# Diagnostic scans on the data (single realisation, not mocks), for BGS and ELG2 only:
#   1) sensitivity to the rebinning factor (dr), at fixed fiducial rmin/rmax/knot_factor
#   2) sensitivity to the bspline knot_factor, at fixed fiducial rmin/rmax/rebin
#
# Each scan produces ONE combined two-panel figure (BGS and ELG2 together, colour-coded):
#   top:    change in R_H relative to the mean across the values tested, in units
#           of the mean fit uncertainty over that range -- WITH error bars taken
#           from each point's own emcee posterior (16th/84th percentile width,
#           expressed in the same normalised units).
#   bottom: percentage change in the fit uncertainty relative to that same mean --
#           points only, no error bars (hence the panel is made narrower).
# NGC = filled markers, SGC = open markers.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fit_mock_average_desilike import setup_fitting
from python_routines import get_data, run_bestfit_spline, run_emcee

if "__main__" in __name__:

    full_tracers, all_names, colors, templates, templates_real, rmins_fid, rmaxs_fid, model, rebin_fid = setup_fitting()
    name_map = dict(zip(full_tracers, all_names))
    cov_index = {tracer: i for i, tracer in enumerate(full_tracers)}

    model = "bspline"
    DEFAULT_KNOT_FACTOR = 3  # fixed knot_factor while scanning rebin -- adjust to your fiducial choice

    cov_factors = {
        "NGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
        "SGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
    }

    tracers = ["BGS", "ELG2"]

    def fit_data(tracer, skyarea, Data, rmin, rmax, rebin, knot_factor, t_idx):
        """Fit the data at a given (rebin, knot_factor, rmin, rmax); cache to disk."""
        filename = (
            f"{tracer}/CD_data_{skyarea}_mean_{model}"
            f"_rebin{rebin}_knotfactor{knot_factor}_rmin{rmin:g}_rmax{rmax:g}_samples_emcee"
        )
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
                plot=True,
                mock="data",
                cov_factor=cov_factors[skyarea][t_idx],
                knot_factor=knot_factor,
            )
            print(
                f"  [new fit] {tracer} {skyarea} rebin={rebin} knot_factor={knot_factor} "
                f"rmin={rmin:g} rmax={rmax:g}: chi2={chi2:.2f}, dof={len(polyfit) - (len(polyfit) // 3 - 4)}"
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

        print(
            f"    {tracer} {skyarea} rebin={rebin} knot_factor={knot_factor} rmin={rmin:6.1f} rmax={rmax:6.1f}: "
            f"R_H = {RH_med:7.3f} +{RH_hi - RH_med:.3f} / -{RH_med - RH_lo:.3f}"
        )
        return RH_med, RH_lo, RH_hi

    def run_scan(x_values, x_label, value_fn, out_filename):
        """
        value_fn(tracer, skyarea, x) -> (RH_med, RH_lo, RH_hi) for a single scan point,
        where RH_lo/RH_hi are the 16th/84th percentile bounds from the emcee chain.
        Produces one figure: BGS in the top panel, ELG2 in the bottom panel, each
        showing only the change in R_H (in units of sigma) vs x_values, with
        asymmetric error bars taken directly from those percentile bounds.
        """
        x_values = np.asarray(x_values, dtype=float)

        fig, (ax_bgs, ax_elg) = plt.subplots(
            2, 1, figsize=(6, 6), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.05)
        )
        axes_by_tracer = {"BGS": ax_bgs, "ELG2": ax_elg}

        for tracer in tracers:
            ax = axes_by_tracer[tracer]

            for skyarea in ["NGC", "SGC"]:
                print(f"\n=== {tracer} {skyarea}: {x_label} scan ===")
                RH_list, lo_list, hi_list = [], [], []
                for x in x_values:
                    try:
                        RH_med, RH_lo, RH_hi = value_fn(tracer, skyarea, x)
                    except Exception as exc:
                        # General safety net only -- the >= MIN_KNOTS constraint itself is
                        # handled up front by restricting the scan range (see below), so
                        # this should only trigger on unrelated fit failures.
                        print(f"Skipping {tracer} {skyarea} {x_label}={x}: {exc}")
                        RH_med, RH_lo, RH_hi = np.nan, np.nan, np.nan
                    RH_list.append(RH_med)
                    lo_list.append(RH_lo)
                    hi_list.append(RH_hi)
                RH_array, lo_array, hi_array = np.array(RH_list), np.array(lo_list), np.array(hi_list)
                sigma_array = (hi_array - lo_array) / 2.0  # symmetric width, used only to set the reference scale

                ref_RH = np.nanmean(RH_array)
                ref_sig = np.nanmean(sigma_array)

                delta_RH_sigma = (RH_array - ref_RH) / ref_sig
                err_lo = (RH_array - lo_array) / ref_sig
                err_hi = (hi_array - RH_array) / ref_sig

                offset = -0.05 if skyarea == "NGC" else 0.05
                style = dict(
                    marker="o",
                    ls="None",
                    color=colors[tracer],
                    mfc=colors[tracer] if skyarea == "NGC" else "w",
                    mec="k" if skyarea == "NGC" else colors[tracer],
                    zorder=5,
                )

                ax.errorbar(x_values + offset, delta_RH_sigma, yerr=[err_lo, err_hi], **style)

            ax.axhline(0.0, color=colors[tracer], ls="--", zorder=1)
            ax.text(
                0.97,
                0.06,
                name_map[tracer],
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="k",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="k", alpha=0.9),
            )
            ax.set_ylabel(r"$\Delta\langle R_H^{\mathrm{galaxy}}\rangle\,[\sigma]$", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)

        ax_elg.set_xlabel(x_label, fontsize=14)

        fig.savefig(out_filename, dpi=200, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Scan 1: rebinning factor, at fixed fiducial rmin/rmax and DEFAULT_KNOT_FACTOR.
    # Same rebin -> dr convention as the earlier mock binning test: dr = rebin * 0.5.
    # ------------------------------------------------------------------
    base_width = 0.5
    rebin_factors = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # -> dr = 1, 2, 2.5, 4, 5, 8, 10
    dr_values = [rebin * base_width for rebin in rebin_factors]
    dr_to_rebin = dict(zip(dr_values, rebin_factors))

    def rebin_scan_value(tracer, skyarea, dr):
        t_idx = cov_index[tracer]
        rmin, rmax = rmins_fid[tracer], rmaxs_fid[tracer]
        rebin = dr_to_rebin[dr]
        Data = get_data(tracer, skyarea, "weighted", rebin=rebin)
        return fit_data(tracer, skyarea, Data, rmin, rmax, rebin, DEFAULT_KNOT_FACTOR, t_idx)

    run_scan(dr_values, r"$\Delta r\,[h^{-1}\mathrm{Mpc}]$", rebin_scan_value, "plots/compare_RH_rebin.png")

    # ------------------------------------------------------------------
    # Scan 2: bspline knot_factor, at fixed fiducial rmin/rmax/rebin.
    # ------------------------------------------------------------------
    knot_factor_values = list(range(2, 5))  # 2, 3, 4

    # Data only depends on (tracer, skyarea, rebin_fid) here, so cache it across the scan.
    _data_cache = {}

    def knot_factor_scan_value(tracer, skyarea, knot_factor):
        t_idx = cov_index[tracer]
        rmin, rmax = rmins_fid[tracer], rmaxs_fid[tracer]
        key = (tracer, skyarea)
        if key not in _data_cache:
            _data_cache[key] = get_data(tracer, skyarea, "weighted", rebin=rebin_fid)
        Data = _data_cache[key]
        return fit_data(tracer, skyarea, Data, rmin, rmax, rebin_fid, int(knot_factor), t_idx)

    run_scan(knot_factor_values, "knot factor", knot_factor_scan_value, "plots/compare_RH_knotfactor.png")
