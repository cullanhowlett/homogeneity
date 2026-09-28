# Compare homogeneity scale fits using different bin widths
import os
import numpy as np
import pandas as pd
from fit_mock_average_desilike import setup_fitting
from python_routines import get_data, grab_data, run_bestfit_spline
from desilike.samples import Chain
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer, Chain as cChain
import matplotlib.patheffects as path_effects
from rescale_MSC_data import get_rescaled_data
from fit_mock_average_desilike import get_likelihood

if "__main__" in __name__:

    tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
    cov_factors = {
        "NGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
        "SGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
    }
    systematic = {
        "NGC": [0.02, 0.21, 0.01, 0.00, 0.19, 0.19],
        "SGC": [0.22, 0.10, 0.08, 0.11, 0.22, 0.21],
    }
    for skyarea in ["NGC", "SGC"]:

        fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.0), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.98, hspace=0.05, wspace=0.05)

        # Flatten axes for easy iteration
        axes_flat = axes.flatten()

        for t, (tracer, ax, name) in enumerate(zip(tracers, axes_flat, names)):
            Data = get_data(tracer, skyarea, "weighted", rebin=rebin)

            data_x, data_y, cov, invcov, corrA, corrB = grab_data(
                Data,
                rmin=Data["r_edges"][0],
                rmax=Data["r_edges"][-1],
                datatype="D2",
                mock="data",
            )
            data_err = np.sqrt(np.diag(cov))
            index = np.where((data_x > rmins[tracer]) & (data_x < rmaxs[tracer]))[0]

            rhp, p, likelihood, likelihood_real = get_likelihood(
                Data,
                Data["r_edges"][0],
                Data["r_edges"][-1],
                templates[tracer],
                templates_real[tracer],
                mock="data",
                model=model,
                cov_factor=cov_factors[skyarea][t],
            )

            infile = f"{tracer}/CD_data_{skyarea}_weighted_{model}_rebin{rebin}_samples_pocomc.npy"
            chain = Chain.load(infile)
            chain = chain.remove_burnin(0.5)
            rhs_full = chain["RHp"].flatten() if "LPT" in model else chain["RH"].flatten()

            bestind = np.argmax(chain["loglikelihood"].flatten())
            pars = [chain[p].flatten()[bestind] for p in ["b1p", "b2p", "bsp", "alpha0p", "sn0p"]]
            rescaled_Data, old_model = get_rescaled_data(Data, pars, likelihood, likelihood_real)
            coeffs, chi2, rescaled_model, rescaled_rh = run_bestfit_spline(
                rescaled_Data,
                rmin=rmins[tracer],
                rmax=rmaxs[tracer],
                datatype="D2",
                plot=False,
                mock="data",
                cov_factor=cov_factors[skyarea][t],
            )
            print(tracer, skyarea, pars, chi2, len(rescaled_model) - (len(rescaled_model) // 3 - 4))

            ax.errorbar(
                data_x,
                data_y,
                yerr=data_err,
                color=colors[tracer],
                ls="None",
                marker="o",
                mec="k",
                mfc=colors[tracer],
                markersize=3,
                alpha=0.8,
                zorder=3,
            )
            ax.errorbar(
                rescaled_Data["r"],
                rescaled_Data["D2"],
                yerr=np.sqrt(np.diag(rescaled_Data["D2_cov"])),
                color=colors[tracer],
                ls="None",
                marker="o",
                mec=colors[tracer],
                mfc="w",
                markersize=3,
                alpha=0.8,
                zorder=3,
            )
            ax.plot(
                data_x[index],
                old_model[index],
                color=colors[tracer],
                ls="-",
                marker="None",
                zorder=4,
                lw=0.8,
            )
            ax.plot(
                data_x[index],
                rescaled_model,
                color=colors[tracer],
                ls="--",
                marker="None",
                zorder=4,
                lw=0.8,
            )

            models, rhs, rescaled_Data, rescaled_models, rescaled_rhs, pars = np.load(
                f"{tracer}/CD_data_{skyarea}_weighted_{model}_rebin{rebin}_samples_pocomc_rescaled.npz",
                allow_pickle=True,
            ).values()
            rescaled_Data = rescaled_Data.item()

            c = ChainConsumer()
            c.add_chain(cChain(samples=pd.DataFrame({"RH": rhs_full}), linewidth=2.0, name="biased", bins=20))
            c.add_chain(cChain(samples=pd.DataFrame({"RH": rescaled_rhs}), linewidth=2.0, name="rescaled", bins=20))
            constraints = c.analysis.get_summary()
            post_RH = constraints["biased"]["RH"]
            post_rescaled_RH = constraints["rescaled"]["RH"]
            print(post_RH, post_rescaled_RH)

            # Calculate the values and errors
            RH_galaxy = post_RH.center
            RH_galaxy_err_upper = post_RH.upper - post_RH.center
            RH_galaxy_err_lower = post_RH.center - post_RH.lower

            RH_rescaled = post_rescaled_RH.center
            RH_rescaled_err_upper = (
                0 if post_rescaled_RH.upper is None else post_rescaled_RH.upper - post_rescaled_RH.center
            )
            RH_rescaled_err_lower = (
                0 if post_rescaled_RH.lower is None else post_rescaled_RH.center - post_rescaled_RH.lower
            )
            print(RH_rescaled, RH_rescaled_err_lower, RH_rescaled_err_upper)
            RH_rescaled_err_upper = np.sqrt(RH_rescaled_err_upper**2 + systematic[skyarea][t] ** 2)
            RH_rescaled_err_lower = np.sqrt(RH_rescaled_err_lower**2 + systematic[skyarea][t] ** 2)
            print(RH_rescaled, RH_rescaled_err_lower, RH_rescaled_err_upper)

            # Create formatted multi-line label with asymmetric errors
            label_text = (
                f"{name}\n"
                f"$R_{{H,{skyarea}}}^{{\\mathrm{{galaxy}}}} = {RH_galaxy:.1f}^{{+{RH_galaxy_err_upper:.1f}}}_{{-{RH_galaxy_err_lower:.1f}}}\\,h^{{-1}}\\,\\mathrm{{Mpc}}$\n"
                f"$R_{{H,{skyarea}}}^{{\\mathrm{{matter}}}} = {RH_rescaled:.1f}^{{+{RH_rescaled_err_upper:.1f}}}_{{-{RH_rescaled_err_lower:.1f}}}\\,h^{{-1}}\\,\\mathrm{{Mpc}}$"
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
                x=RH_rescaled + RH_rescaled_err_upper, color=colors[tracer], ls="--", alpha=0.5, lw=1.0, zorder=2
            )
            ax.axvline(
                x=RH_rescaled - RH_rescaled_err_lower, color=colors[tracer], ls="--", alpha=0.5, lw=1.0, zorder=2
            )
            ax.axvspan(
                xmin=RH_galaxy - RH_galaxy_err_lower,
                xmax=RH_galaxy + RH_galaxy_err_upper,
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

        plt.savefig(f"./CD_data_{skyarea}_weighted_{model}_rebin{rebin}_samples_pocomc.png", dpi=300)
