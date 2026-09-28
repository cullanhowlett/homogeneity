import os
import pandas as pd
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.patches as mpatches
from python_routines import get_mocks, grab_data
from fit_mock_average_desilike import setup_fitting
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike_routines import CorrelationDimensionLikelihood, LPTVelocileptorsTracerCorrelationDimension


if "__main__" in __name__:

    # Generate a likelihood to compute the model. Let's use a z=0 model.
    rhp = 5
    template = DirectPowerSpectrumTemplate(z=0.0, fiducial="DESI")
    theory = LPTVelocileptorsTracerCorrelationDimension(template=template)
    for param in theory.template.params:
        theory.template.params[param].update(fixed=True)

    AbacusSummit, EZmock = get_mocks("BGS", "NGC", "weighted", rebin=5)
    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        AbacusSummit, rmin=AbacusSummit["r_edges"][0], rmax=AbacusSummit["r_edges"][-1], datatype="D2", mock=None
    )
    likelihood = CorrelationDimensionLikelihood(
        data_x,
        data_y,
        covariance=cov,
        precision=invcov,
        theory=theory,
    )
    likelihood()
    r_range = likelihood.theory.s
    rin_range = likelihood.theory.sin
    redges_range = likelihood.theory.sedges

    param_names = ["b1p", "b2p", "bsp", "alpha0p", "sn0p"]
    param_labels = [
        r"$b_{1}\sigma_{8}$",
        r"$b_{2}\sigma^{2}_{8}$",
        r"$b_{s}\sigma^{2}_{8}$",
        r"$\alpha_{0}$",
        r"$\mathrm{SN}_{0}$",
    ]
    param_ranges = {
        "b1p": (0.5, 3.0),
        "b2p": (-4.0, 4.0),
        "bsp": (-4.0, 4.0),
        "alpha0p": (-100.0, 100.0),
        "sn0p": (-1.0, 4.0),
    }
    default_params = {"b1p": likelihood.theory.pt.sigma8, "b2p": 0.0, "bsp": 0.0, "alpha0p": 0.0, "sn0p": 0.0}

    grid_resolution = 11

    # ============================================================================
    # COMPUTE DATA
    # ============================================================================
    print("Computing RH grids and D2 curves in unified loop...")
    RH_grids, D2_curves, msc_curves, corr_curves = (
        {},
        {name: [] for name in param_names},
        {name: [] for name in param_names},
        {name: [] for name in param_names},
    )
    param_pairs = []

    # First, compute D2 curve for b1p (which would otherwise be missed)
    param_name = param_names[0]
    p_vals = np.linspace(param_ranges[param_name][0], param_ranges[param_name][1], grid_resolution)
    print(p_vals)
    for p_val in p_vals:
        params = default_params.copy()
        params[param_name] = p_val
        loglikelihood = likelihood(**params)
        D2_curves[param_name].append((p_val, likelihood.theory.cd))
        msc_curves[param_name].append((p_val, likelihood.theory.msc))
        corr_curves[param_name].append((p_val, likelihood.theory.corr.corr[0]))

    # For each parameter pair
    for i in range(5):
        param_name_i = param_names[i]

        # Set up grid for parameter i, ensuring default value is in the middle
        p_i_min, p_i_max = param_ranges[param_name_i]
        p_i_default = default_params[param_name_i]
        # Create grid centered on default (grid_resolution should be odd for exact center)
        p_i_vals_grid = np.linspace(p_i_min, p_i_max, grid_resolution)

        # Find index closest to default value (will use this for D2 curves)
        default_idx_i = np.argmin(np.abs(p_i_vals_grid - p_i_default))

        for j in range(i):
            param_name_j = param_names[j]
            pair_name = f"{param_name_i}_{param_name_j}"
            param_pairs.append((i, j, pair_name))

            print(f"  Computing RH grid for {pair_name}...")

            # Set up grid for parameter j, ensuring default value is in the middle
            p_j_min, p_j_max = param_ranges[param_name_j]
            p_j_default = default_params[param_name_j]
            p_j_vals_grid = np.linspace(p_j_min, p_j_max, grid_resolution)

            # Find index closest to default value
            default_idx_j = np.argmin(np.abs(p_j_vals_grid - p_j_default))

            RH_grid = np.zeros((grid_resolution, grid_resolution))

            for ii, p_i in enumerate(p_i_vals_grid):
                for jj, p_j in enumerate(p_j_vals_grid):
                    params = default_params.copy()
                    params[param_name_i], params[param_name_j] = p_i, p_j

                    loglikelihood = likelihood(**params)
                    RH_grid[jj, ii] = likelihood.theory.RHp

                    # Extract D2 curves when other parameter is at default
                    if jj == default_idx_j and j == 0 and i > 0:
                        # Varying param_i, param_j at default -> D2 curve for param_i
                        D2_curves[param_name_i].append((p_i, likelihood.theory.cd))
                        msc_curves[param_name_i].append((p_i, likelihood.theory.msc))
                        corr_curves[param_name_i].append((p_i, likelihood.theory.corr.corr[0]))

            RH_grids[pair_name] = (RH_grid, p_j_vals_grid, p_i_vals_grid)

    print("\nData computation complete!")

    # ============================================================================
    # CREATE D2 FIGURE
    # ============================================================================
    fig = plt.figure(figsize=(4.5, 8))

    # Create 5 panels in a single column
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.05, left=0.15, right=0.92, top=0.97, bottom=0.06)

    axes = []
    for idx, param_name in enumerate(param_names):
        if idx == 0:
            ax = fig.add_subplot(gs[idx, 0])
        else:
            ax = fig.add_subplot(gs[idx, 0], sharex=axes[0])
        axes.append(ax)

        curves = D2_curves[param_name]

        # Color by parameter value
        max_deviation = max(
            abs(param_ranges[param_name][1] - default_params[param_name]),
            abs(param_ranges[param_name][0] - default_params[param_name]),
        )
        norm_min = default_params[param_name] - max_deviation
        norm_max = default_params[param_name] + max_deviation

        param_norm = TwoSlopeNorm(vmin=norm_min, vcenter=default_params[param_name], vmax=norm_max)
        param_cmap = cmr.pride

        for p_val, D2 in curves:
            color = param_cmap(param_norm(p_val))
            ax.plot(r_range, D2, color=color, linewidth=1.5, alpha=0.8)

        # Add colorbar below each panel
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            ax, width="60%", height="5%", loc="lower right", bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=ax.transAxes
        )
        sm = cm.ScalarMappable(norm=param_norm, cmap=param_cmap)
        sm.set_array([])
        cbar_small = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar_small.ax.set_xlim(param_ranges[param_name][0], param_ranges[param_name][1])
        cbar_small.ax.tick_params(labelsize=6, top=True, labeltop=True, bottom=False, labelbottom=False)
        cbar_small.ax.xaxis.set_label_position("top")
        cbar_small.set_label(param_labels[idx], fontsize=7, labelpad=2)

        # Only bottom panel gets x-label and x-tick labels
        if idx == len(param_names) - 1:
            ax.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$", fontsize=10)
            ax.tick_params(labelsize=8, labelbottom=True)
        else:
            ax.tick_params(labelsize=8, labelbottom=False)
        ax.axhline(y=3.00, color="k", ls="-", lw=1.3, zorder=1)
        ax.axhline(y=2.97, color="k", ls="--", lw=1.3, zorder=1)
        ax.set_xlim(20.0, 250.0)
        ax.set_ylim(2.75, 3.02)
        ax.set_ylabel(r"$D_2(r)$", fontsize=10)
        ax.tick_params(labelsize=10)

    print("Saving figure...")
    plt.savefig("./correlation_dimension_theory.png", dpi=300, bbox_inches="tight")
    print("Done!")

    # ============================================================================
    # CREATE MSC FIGURE
    # ============================================================================
    fig = plt.figure(figsize=(4.5, 8))

    # Create 5 panels in a single column
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.05, left=0.15, right=0.92, top=0.97, bottom=0.06)

    axes = []
    for idx, param_name in enumerate(param_names):
        if idx == 0:
            ax = fig.add_subplot(gs[idx, 0])
        else:
            ax = fig.add_subplot(gs[idx, 0], sharex=axes[0])
        axes.append(ax)

        curves = msc_curves[param_name]

        # Color by parameter value
        max_deviation = max(
            abs(param_ranges[param_name][1] - default_params[param_name]),
            abs(param_ranges[param_name][0] - default_params[param_name]),
        )
        norm_min = default_params[param_name] - max_deviation
        norm_max = default_params[param_name] + max_deviation

        param_norm = TwoSlopeNorm(vmin=norm_min, vcenter=default_params[param_name], vmax=norm_max)
        param_cmap = cmr.pride

        for p_val, msc in curves:
            color = param_cmap(param_norm(p_val))
            ax.plot(redges_range, msc, color=color, linewidth=1.5, alpha=0.8)

        # Add colorbar below each panel
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            ax, width="60%", height="5%", loc="upper right", bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=ax.transAxes
        )
        sm = cm.ScalarMappable(norm=param_norm, cmap=param_cmap)
        sm.set_array([])
        cbar_small = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar_small.ax.set_xlim(param_ranges[param_name][0], param_ranges[param_name][1])
        cbar_small.ax.tick_params(labelsize=6, top=True, labeltop=True, bottom=False, labelbottom=False)
        cbar_small.ax.xaxis.set_label_position("top")
        cbar_small.set_label(param_labels[idx], fontsize=7, labelpad=2)

        # Only bottom panel gets x-label and x-tick labels
        if idx == len(param_names) - 1:
            ax.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$", fontsize=10)
            ax.tick_params(labelsize=8, labelbottom=True)
        else:
            ax.tick_params(labelsize=8, labelbottom=False)
        ax.set_xlim(20.0, 250.0)
        ax.set_ylim(0.99, 3.0)
        ax.set_ylabel(r"$N(r)$", fontsize=10)
        ax.tick_params(labelsize=10)

    print("Saving figure...")
    plt.savefig("./MSC_theory.png", dpi=300, bbox_inches="tight")
    print("Done!")

    # ============================================================================
    # CREATE MSC FIGURE
    # ============================================================================
    fig = plt.figure(figsize=(4.5, 8))

    # Create 5 panels in a single column
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.05, left=0.15, right=0.92, top=0.97, bottom=0.06)

    axes = []
    for idx, param_name in enumerate(param_names):
        if idx == 0:
            ax = fig.add_subplot(gs[idx, 0])
        else:
            ax = fig.add_subplot(gs[idx, 0], sharex=axes[0])
        axes.append(ax)

        curves = corr_curves[param_name]

        # Color by parameter value
        max_deviation = max(
            abs(param_ranges[param_name][1] - default_params[param_name]),
            abs(param_ranges[param_name][0] - default_params[param_name]),
        )
        norm_min = default_params[param_name] - max_deviation
        norm_max = default_params[param_name] + max_deviation

        param_norm = TwoSlopeNorm(vmin=norm_min, vcenter=default_params[param_name], vmax=norm_max)
        param_cmap = cmr.pride

        for p_val, corr in curves:
            color = param_cmap(param_norm(p_val))
            ax.plot(rin_range, rin_range**2 * corr, color=color, linewidth=1.5, alpha=0.8)

        # Add colorbar below each panel
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            ax, width="60%", height="5%", loc="upper right", bbox_to_anchor=(0, 0.02, 1, 1), bbox_transform=ax.transAxes
        )
        sm = cm.ScalarMappable(norm=param_norm, cmap=param_cmap)
        sm.set_array([])
        cbar_small = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cbar_small.ax.set_xlim(param_ranges[param_name][0], param_ranges[param_name][1])
        cbar_small.ax.tick_params(labelsize=6, top=True, labeltop=True, bottom=False, labelbottom=False)
        cbar_small.ax.xaxis.set_label_position("top")
        cbar_small.set_label(param_labels[idx], fontsize=7, labelpad=2)

        # Only bottom panel gets x-label and x-tick labels
        if idx == len(param_names) - 1:
            ax.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$", fontsize=10)
            ax.tick_params(labelsize=8, labelbottom=True)
        else:
            ax.tick_params(labelsize=8, labelbottom=False)
        ax.set_xlim(1.0, 250.0)
        ax.set_ylim(-50.0, 150.0)
        ax.set_ylabel(r"$r^{2}\xi(r)$", fontsize=10)
        ax.tick_params(labelsize=10)

    print("Saving figure...")
    plt.savefig("./correlation_function_theory.png", dpi=300, bbox_inches="tight")
    print("Done!")
