# Compare homogeneity scale fits using different bin widths
import os
import numpy as np
import pandas as pd
from fit_mock_average_desilike import setup_fitting
from python_routines import get_data, grab_data, get_mocks
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain
import matplotlib.patheffects as path_effects
from scipy.interpolate import BSpline, PPoly

if "__main__" in __name__:

    tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
    model = "bspline"

    cov_factors = {
        "NGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
        "SGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
    }

    # First plot the mean scaled counts
    fig, axes = plt.subplots(6, 1, figsize=(4.5, 10), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.06, top=0.99, hspace=0.06)

    for t, (tracer, ax, name) in enumerate(zip(tracers, axes, names)):
        for skyarea in ["NGC", "SGC"]:
            Data = get_data(tracer, skyarea, "weighted", rebin=rebin)
            AbacusSummit, EZmock = get_mocks(tracer, skyarea, "weighted", rebin=rebin)

            # first the Mean scaled counts plot
            data_x, data_y, cov, invcov, corrA, corrB = grab_data(
                Data,
                rmin=Data["r_edges"][0],
                rmax=Data["r_edges"][-1],
                datatype="MSC",
                mock="data",
            )
            data_err = np.sqrt(np.diag(cov))
            label_text = f"{name}"

            ax.errorbar(
                data_x,
                data_y - 1.0,
                yerr=data_err,
                color=colors[tracer],
                ls="None",
                marker="o",
                mec="k" if skyarea == "NGC" else colors[tracer],
                mfc=colors[tracer] if skyarea == "NGC" else "w",
                markersize=4,
                alpha=0.8,
                zorder=3,
            )

            # first the Mean scaled counts plot
            data_x, data_y, cov, invcov, corrA, corrB = grab_data(
                AbacusSummit,
                rmin=AbacusSummit["r_edges"][0],
                rmax=AbacusSummit["r_edges"][-1],
                datatype="MSC",
                mock=None,
            )

            ax.errorbar(
                data_x,
                data_y - 1.0,
                color=colors[tracer],
                ls="-" if skyarea == "NGC" else "--",
                marker="None",
                alpha=0.8,
                zorder=3,
            )

        text = ax.text(
            0.98,
            0.92,
            label_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=12,
            fontweight="bold",
            color="k",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3),
        )
        # text.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground="black", alpha=0.5)])
        ax.axhline(y=0.01, color="k", ls=":", lw=1.3, zorder=1)
        ax.set_xlim(50.0, 200.0)
        ax.set_yscale("log")
        ax.set_ylim(0.0001 if tracer == "LRG1" else 0.001, 0.2)
        ax.tick_params(labelsize=10)

        # Only show x-tick labels on bottom panel
        if t < 5:
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(r"$N(r) - 1$", fontsize=12)

    # Add x-label only to bottom panel
    axes[-1].set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$", fontsize=12)

    plt.savefig("./MSC_data_NGCSGC.png", dpi=800)

    # And now the correlation dimensions and polynomial fits
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.98, hspace=0.05, wspace=0.05)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    for t, (tracer, ax, name) in enumerate(zip(tracers, axes_flat, names)):
        RH_galaxy = {}
        for skyarea in ["NGC", "SGC"]:
            Data = get_data(tracer, skyarea, "weighted", rebin=rebin)

            filename = f"{tracer}/CD_data_{skyarea}_weighted_{model}_rebin{rebin}_samples_emcee"
            if model == "bspline":
                first_line = np.loadtxt(filename + ".txt", max_rows=1)
                degree, n_cols = int(first_line[-3]), len(first_line)
                n_coeffs = (n_cols - degree - 4) // 2
                colnames = (
                    [f"a_{i}" for i in range(n_coeffs)]
                    + [f"k_{i}" for i in range(n_cols - n_coeffs - 3)]
                    + ["degree", "R_H", "log_post"]
                )
            else:
                colnames = [f"a_{i}" for i in range(int(model[-1]) + 1)] + ["R_H", "log_post"]
            samples = pd.read_csv(
                filename + ".txt",
                header=None,
                sep=r"\s+",
                names=colnames,
            )
            c = ChainConsumer()
            columns_to_use = [s for s in samples.columns if "k" not in s and s != "degree"]
            c.add_chain(
                Chain(
                    samples=samples[columns_to_use],
                    linewidth=2.0,
                    name=tracer,
                )
            )
            constraints = c.analysis.get_summary()
            post_RH = constraints[tracer]["R_H"]
            print(tracer, skyarea, post_RH.center, post_RH.upper - post_RH.center, post_RH.center - post_RH.lower)

            data_x, data_y, cov, invcov, corrA, corrB = grab_data(
                Data,
                rmin=Data["r_edges"][0],
                rmax=Data["r_edges"][-1],
                datatype="D2",
                mock="data",
            )
            data_err = np.sqrt(np.diag(cov))
            index = np.where((data_x > rmins[tracer]) & (data_x < rmaxs[tracer]))[0]

            # Calculate the values and errors
            RH_galaxy[skyarea] = [post_RH.center, post_RH.upper - post_RH.center, post_RH.center - post_RH.lower]

            ax.errorbar(
                data_x,
                data_y,
                yerr=data_err,
                color=colors[tracer],
                ls="None",
                marker="o",
                mec="k" if skyarea == "NGC" else colors[tracer],
                mfc=colors[tracer] if skyarea == "NGC" else "w",
                markersize=3,
                alpha=0.8,
                zorder=3,
            )
            max_post = np.argmax(samples["log_post"])
            if model == "bspline":
                degree = samples.iloc[max_post].to_numpy()[-3]
                n_coeffs = int((len(samples.columns) - degree - 4) // 2)
                coeffs, knots, degree = (
                    samples.iloc[max_post].to_numpy()[:n_coeffs],
                    samples.iloc[max_post].to_numpy()[n_coeffs:-3],
                    int(samples.iloc[max_post].to_numpy()[-3]),
                )
                spline = BSpline(knots, coeffs, degree)
                best_model = spline(data_x)
            else:
                best_model = np.poly1d(samples.iloc[max_post].to_numpy()[:-2])(data_x / 100.0)
            ax.plot(
                data_x[index],
                best_model[index],
                color=colors[tracer],
                ls="-" if skyarea == "NGC" else "--",
                marker="None",
                zorder=4,
                lw=0.8,
            )

        # Create formatted multi-line label with asymmetric errors
        label_text = (
            f"{name}\n"
            f"$R_{{H,\\mathrm{{NGC}}}}^{{\\mathrm{{galaxy}}}} = {RH_galaxy['NGC'][0]:.1f}^{{+{RH_galaxy['NGC'][1]:.1f}}}_{{-{RH_galaxy['NGC'][2]:.1f}}}\\,h^{{-1}}\\,\\mathrm{{Mpc}}$\n"
            f"$R_{{H,\\mathrm{{SGC}}}}^{{\\mathrm{{galaxy}}}} = {RH_galaxy['SGC'][0]:.1f}^{{+{RH_galaxy['SGC'][1]:.1f}}}_{{-{RH_galaxy['SGC'][2]:.1f}}}\\,h^{{-1}}\\,\\mathrm{{Mpc}}$"
        )

        text = ax.text(
            0.96,
            0.04,
            label_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right",
            fontsize=7.5,
            fontweight="bold",
            color="k",
            bbox=dict(boxstyle="round", facecolor="white", alpha=1.0, pad=0.5),
        )
        # text.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground="black", alpha=0.5)])
        ax.axvline(x=rmins[tracer], color="k", ls=":", alpha=0.5, lw=1.0, zorder=1)
        ax.axvline(x=rmaxs[tracer], color="k", ls=":", alpha=0.5, lw=1.0, zorder=1)
        ax.axhline(y=3.00, color="k", ls="-", lw=1.0, zorder=1)
        ax.axhline(y=2.97, color="k", ls=":", lw=1.0, zorder=1)
        ax.set_xlim(20.0, 215.0)
        ax.set_ylim(2.65, 3.01)
        ax.tick_params(labelsize=7)
        ax.axvline(
            x=RH_galaxy["SGC"][0] + RH_galaxy["SGC"][1], color=colors[tracer], ls="--", alpha=0.5, lw=1.0, zorder=2
        )
        ax.axvline(
            x=RH_galaxy["SGC"][0] - RH_galaxy["SGC"][2], color=colors[tracer], ls="--", alpha=0.5, lw=1.0, zorder=2
        )
        ax.axvspan(
            xmin=RH_galaxy["NGC"][0] - RH_galaxy["NGC"][2],
            xmax=RH_galaxy["NGC"][0] + RH_galaxy["NGC"][1],
            color=colors[tracer],
            alpha=0.3,
            zorder=1,
        )

        # Add x-labels only to bottom row (indices 4 and 5)
        if t >= 3:
            ax.set_xlabel(r"$r\,[h^{-1}\mathrm{Mpc}]$", fontsize=9)

        # Add y-labels only to left column (indices 0, 2, 4)
        if t % 3 == 0:
            ax.set_ylabel(r"$D_{2}(r)$", fontsize=9)

    plt.savefig(f"./CD_data_NGCSGC_weighted_{model}_rebin{rebin}_samples_emcee.png", dpi=800)
