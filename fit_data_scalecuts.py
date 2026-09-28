# Fit the mock average over a grid of (rmin, rmax) scale cuts, for BGS and ELG2 only,
# and plot NGC/SGC heatmaps of the change in R_H relative to the fiducial scale cut,
# in units of the single-realisation error bar.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fit_mock_average_desilike import setup_fitting
from python_routines import get_data, run_bestfit_spline, run_emcee

if "__main__" in __name__:

    full_tracers, all_names, colors, templates, templates_real, rmins_fid, rmaxs_fid, model, rebin = setup_fitting()
    name_map = dict(zip(full_tracers, all_names))
    cov_index = {tracer: i for i, tracer in enumerate(full_tracers)}

    model = "bspline"

    cov_factors = {
        "NGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
        "SGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
    }

    tracers = ["BGS", "ELG2"]

    # Scale-cut grid: rmin from 20-68, rmax from 120-190, both in steps of 4 h^-1 Mpc.
    # Note 190 isn't exactly reached by 4 Mpc/h steps from 120 -> grid tops out at 188.
    rmin_grid = np.arange(20, 68 + 1, 4)  # 20, 24, ..., 68
    rmax_grids = {
        "BGS": np.arange(144, 192 + 1, 4),  # 144, 148, ..., 192
        "ELG2": np.arange(120, 192 + 1, 4),  # 120, 124, ..., 192
    }

    def fit_mean(tracer, skyarea, Data, rmin, rmax, t_idx):
        """Fit the mock average at a given (rmin, rmax); cache to disk like the original script."""
        filename = f"{tracer}/CD_data_{skyarea}_mean_{model}" f"_rebin{rebin}_rmin{rmin:g}_rmax{rmax:g}_samples_emcee"
        if os.path.isfile(filename + ".txt"):
            first_line = np.loadtxt(filename + ".txt", max_rows=1)
            degree, n_cols = int(first_line[-3]), len(first_line)
            n_coeffs = (n_cols - degree - 4) // 2
            colnames = (
                [f"a_{i}" for i in range(n_coeffs)]
                + [f"k_{i}" for i in range(n_cols - n_coeffs - 3)]
                + ["degree", "R_H", "log_post"]
            )
            df = pd.read_csv(filename + ".txt", header=None, sep=r"\s+", names=colnames)
        else:
            coeffs, chi2, polyfit, _ = run_bestfit_spline(
                Data,
                rmin=rmin,
                rmax=rmax,
                datatype="D2",
                plot=False,
                mock="data",
                cov_factor=cov_factors[skyarea][t_idx],
            )
            print(
                f"  [new fit] {tracer} {skyarea} rmin={rmin:g} rmax={rmax:g}: chi2={chi2:.2f}, dof={len(polyfit) - (len(polyfit) // 3 - 4)}"
            )
            df = run_emcee(
                Data,
                coeffs,
                model=model,
                rmin=rmin,
                rmax=rmax,
                datatype="D2",
                mock="data",
                cov_factor=cov_factors[skyarea][t_idx],
            )
            np.savetxt(filename + ".txt", df.to_numpy(), fmt="%g")

        RH_lo, RH_med, RH_hi = np.percentile(df["R_H"], [16, 50, 84])
        RH_bestfit = RH_med
        RH_sigma_mean = (RH_hi - RH_lo) / 2.0
        RH_sigma_single = RH_sigma_mean

        print(
            f"    {tracer} {skyarea} rmin={rmin:6.1f} rmax={rmax:6.1f}: "
            f"R_H = {RH_med:7.3f} +{RH_hi - RH_med:.3f} / -{RH_med - RH_lo:.3f}  "
        )

        return RH_bestfit, RH_sigma_single

    for tracer in tracers:
        t_idx = cov_index[tracer]
        fid_rmin, fid_rmax = rmins_fid[tracer], rmaxs_fid[tracer]
        rmax_grid = rmax_grids[tracer]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, gridspec_kw=dict(wspace=0.08))

        step = rmin_grid[1] - rmin_grid[0]
        extent = [
            rmin_grid[0] - step / 2,
            rmin_grid[-1] + step / 2,
            rmax_grid[0] - step / 2,
            rmax_grid[-1] + step / 2,
        ]

        # First pass: fit both skyareas and collect their delta grids, so the two
        # panels can share a single colour scale (and therefore a single colorbar).
        deltas = {}
        for skyarea in ["NGC", "SGC"]:

            print(f"\n=== {tracer} {skyarea} ===")
            Data = get_data(tracer, skyarea, "weighted", rebin=rebin)

            RH_grid = np.full((len(rmax_grid), len(rmin_grid)), np.nan)
            sigma_grid = np.full((len(rmax_grid), len(rmin_grid)), np.nan)

            for i, rmin in enumerate(rmin_grid):
                for j, rmax in enumerate(rmax_grid):
                    try:
                        RH_bestfit, RH_sigma_single = fit_mean(tracer, skyarea, Data, rmin, rmax, t_idx)
                        RH_grid[j, i] = RH_bestfit
                        sigma_grid[j, i] = RH_sigma_single
                    except Exception as exc:
                        print(f"Skipping {tracer} {skyarea} rmin={rmin} rmax={rmax}: {exc}")
                        continue

            # Fiducial (rmin, rmax) is on the grid by construction -- pull it out directly.
            i_fid = int(np.argmin(np.abs(rmin_grid - fid_rmin)))
            j_fid = int(np.argmin(np.abs(rmax_grid - fid_rmax)))
            RH_fid = RH_grid[j_fid, i_fid]
            sigma_fid = sigma_grid[j_fid, i_fid]

            deltas[skyarea] = (RH_grid - np.mean(RH_grid)) / np.mean(sigma_grid)

        vmax = np.nanmax([np.nanmax(np.abs(d)) for d in deltas.values() if np.any(np.isfinite(d))] or [1.0])

        # Second pass: plot both panels on the shared colour scale.
        for ax, skyarea in zip(axes, ["NGC", "SGC"]):
            im = ax.imshow(
                deltas[skyarea],
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.plot(fid_rmin, fid_rmax, marker="x", color="k", ms=12, mew=2.5, zorder=10)
            ax.set_xlabel(r"$r_{\rm min}\,[h^{-1}\mathrm{Mpc}]$", fontsize=13)
            ax.set_title(skyarea, fontsize=13)

        axes[1].tick_params(axis="y", labelleft=False)
        axes[0].set_ylabel(r"$r_{\rm max}\,[h^{-1}\mathrm{Mpc}]$", fontsize=13)

        cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.9, pad=0.02)
        cb.set_label(r"$\Delta\langle R_H^{\mathrm{galaxy}}\rangle\,[\sigma]$", fontsize=12)

        # Center the title over the two panels (not the colorbar) -- axes positions
        # are only final once the colorbar has been laid out, so draw first.
        fig.canvas.draw()
        x_center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
        fig.suptitle(name_map[tracer], x=x_center, fontsize=14, fontweight="bold", color="k")
        fig.savefig(f"compare_RH_scalecuts_{tracer}.png", dpi=200, bbox_inches="tight")
        plt.show()
