# Compare homogeneity scale fits using different bin widths
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desilike import setup_logging
from desilike.samples import Chain
from chainconsumer import ChainConsumer, Chain as cChain
from fit_mock_average_desilike import setup_fitting
import matplotlib.patheffects as path_effects


if "__main__" in __name__:

    tracers, names, colors, templates, templates_real, rmin, rmaxs, model, rebin = setup_fitting()
    dof = np.array([23, 24, 24, 25, 20, 30])
    cov_factors_NGC = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]  # Already included in the original fits
    cov_factors_SGC = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

    # Separate NGC and SGC
    # Create figure with custom width ratios (3:1)
    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(
        len(tracers), 2, width_ratios=[3, 1], hspace=0.05, wspace=0.05, left=0.07, right=0.98, top=0.98, bottom=0.05
    )
    # Accumulate all the measurements
    left_axes, right_axes = [], []
    chi2_values, bestfit_values, std_values = [], [], []
    for t, tracer in enumerate(tracers):

        Rhmin, Rhmax = 1.0e10, -1.0e10
        for s, skyarea in enumerate(["NGC", "SGC"]):

            # Left panels (error bar plots)
            if skyarea == "NGC":
                ax_left = fig.add_subplot(gs[t, 0])
                left_axes.append(ax_left)
            else:
                ax_left = left_axes[t]

            c = ChainConsumer()
            for mc in range(25):
                file = f"{tracer}/AbacusSummit/fits/CD_AbacusSummit_{skyarea}_{mc}_singlecov_{model}_rebin{rebin}_samples_pocomc"
                chain = Chain.load(file + ".npy")
                chain = chain.remove_burnin(0.5)
                params = chain[0].params(varied=True).names() + ["RHp"]
                samples = pd.DataFrame(
                    chain.to_array(params=params, struct=False, derivs=()).reshape(-1, chain.size).T, columns=params
                )
                samples["weights"] = chain.weight.ravel()
                chi2_values.append(
                    -2.0
                    * np.asarray(chain.logposterior.ravel()).max()
                    / (cov_factors_NGC[t] if skyarea == "NGC" else cov_factors_SGC[t])
                )
                # bestfit_values.append(samples["RHp"].values[np.argmax(chain.logposterior.ravel())])
                std_values.append(np.std(samples["RHp"]))

                c.add_chain(
                    cChain(
                        samples=samples,
                        weight_column="weights",
                        name=names[t] + f" {mc}",
                        linewidth=1.0,
                        bins=10 if tracer == "LRG" else 20,
                    )
                )

            chain = Chain.load(
                f"{tracer}/AbacusSummit/fits/CD_AbacusSummit_{skyarea}_mean_singlecov_{model}_rebin{rebin}_samples_pocomc.npy"
            )
            chain = chain.remove_burnin(0.5)
            params = chain[0].params(varied=True).names() + ["RHp"]
            samples = pd.DataFrame(
                chain.to_array(params=params, struct=False, derivs=()).reshape(-1, chain.size).T, columns=params
            )
            samples["weights"] = chain.weight.ravel()
            samples[r"$\chi^{2}$"] = -2.0 * np.asarray(
                chain.logposterior.ravel() / (cov_factors_NGC[t] if skyarea == "NGC" else cov_factors_SGC[t])
            )

            c.add_chain(
                cChain(
                    samples=samples,
                    weight_column="weights",
                    name=names[t] + " Mean",
                    linewidth=1.7,
                    bins=10 if tracer == "LRG" else 20,
                )
            )

            # Return R_H error bounds using ChainConsumer
            results = c.analysis.get_summary()
            Rhs = []
            for n in c._chains:
                if results[n]["RHp"].lower is not None:
                    Rhs.append(results[n]["RHp"].array)
                else:
                    Rhs.append(np.quantile(c._chains[n].samples["RHp"], [0.16, 0.5, 0.84]))
            Rhs = np.array(Rhs)
            bestfit_values.append(Rhs[:, 1])

            Rhmin = np.amin([Rhmin, 0.995 * np.nanmin(Rhs[:, 0])])
            Rhmax = np.amax([Rhmax, 1.020 * np.nanmax(Rhs[:, 2])])

            # Plot error bars
            ax_left.errorbar(
                np.arange(0, 25) - (0.1 if skyarea == "NGC" else -0.1),
                Rhs[:-1, 1],
                yerr=[Rhs[:-1, 1] - Rhs[:-1, 0], Rhs[:-1, 2] - Rhs[:-1, 1]],
                marker="o",
                ls="None",
                color=colors[tracer],
                mfc=colors[tracer] if skyarea == "NGC" else "w",
                mec="k" if skyarea == "NGC" else colors[tracer],
                alpha=1.0,
                zorder=5,
            )
            ax_left.errorbar(
                [25 - (0.1 if skyarea == "NGC" else -0.1)],
                [Rhs[-1, 1]],
                yerr=[[Rhs[-1, 1] - Rhs[-1, 0]], [Rhs[-1, 2] - Rhs[-1, 1]]],
                marker="*",
                ms=14,
                ls="None",
                color=colors[tracer],
                mfc=colors[tracer] if skyarea == "NGC" else "w",
                mec="k" if skyarea == "NGC" else colors[tracer],
                alpha=1.0,
                zorder=5,
            )

            # Add horizontal fill
            Rh_mean, Rh_std = np.mean(Rhs[:-1, 1]), np.std(Rhs[:-1, 1])
            # ax_left.axhline(y=Rh_mean, alpha=0.5, color=colors[tracer], zorder=1, ls="-" if skyarea == "NGC" else "--")
            if skyarea == "NGC":
                ax_left.axhspan(Rh_mean - Rh_std, Rh_mean + Rh_std, alpha=0.3, color=colors[tracer], zorder=0)
            else:
                ax_left.axhline(y=Rh_mean - Rh_std, alpha=0.5, color=colors[tracer], zorder=1, ls="--")
                ax_left.axhline(y=Rh_mean + Rh_std, alpha=0.5, color=colors[tracer], zorder=1, ls="--")

            # First is ratio of std(mean(RH_mock_i)) to <std(RH_mock_i)>
            # Second is ratio of std(mean(RH_mock_i)) to std(RH_mock_mean)
            print(
                tracer,
                skyarea,
                1.0 / (np.array(std_values[(2 * t + s) * 25 : (2 * t + s + 1) * 25]).mean() / Rh_std),
                1.0 / (np.std(samples["RHp"]) / Rh_std),
            )

        # Add text label in top-left
        text = ax_left.text(
            0.02,
            0.92,
            names[t],
            transform=ax_left.transAxes,
            verticalalignment="top",
            fontsize=12,
            fontweight="bold",
            color="k",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        # text.set_path_effects(
        #    [path_effects.Stroke(linewidth=0.5, foreground="black", alpha=0.8), path_effects.Normal()]
        # )

        # Format axes
        ax_left.set_xticks(np.arange(0, 26))
        ax_left.set_ylim((Rhmin, Rhmax))
        ax_left.set_xticklabels([str(x) for x in np.arange(0, 25)] + ["  Mean"], fontsize=12, rotation=0)
        if t < 5:
            ax_left.set_xticklabels([])
        else:
            ax_left.set_xlabel("Mock", fontsize=14)
        ax_left.tick_params(axis="y", labelsize=12)

    all_chi2 = np.array(chi2_values).reshape(len(tracers), 50)
    all_bestfits = np.array(bestfit_values).reshape(len(tracers), 52)

    print(
        np.corrcoef(
            (all_bestfits[:, :25] - all_bestfits[:, 25, None]).flatten(),
            (all_bestfits[:, 26:-1] - all_bestfits[:, -1, None]).flatten(),
        )
    )

    n_bins = 20
    common_bins = np.linspace(all_chi2.min(), all_chi2.max(), n_bins + 1)

    # Plot histograms
    for t, tracer in enumerate(tracers):
        for s, skyarea in enumerate(["NGC", "SGC"]):
            if skyarea == "NGC":
                if t == 0:
                    ax_right = fig.add_subplot(gs[t, 1])
                else:
                    ax_right = fig.add_subplot(gs[t, 1], sharex=right_axes[0])
                right_axes.append(ax_right)
            else:
                ax_right = right_axes[t]

            ax_right.hist(
                all_chi2[t, s * 25 : (s + 1) * 25],
                bins=common_bins,
                orientation="vertical",
                color=colors[tracer],
                alpha=0.4 if skyarea == "NGC" else 1.0,
                lw=0.8 if skyarea == "NGC" else 1.7,
                edgecolor="black" if skyarea == "NGC" else colors[tracer],
                histtype="stepfilled" if skyarea == "NGC" else "step",
            )

            if skyarea == "NGC":
                ax_right.axvline(x=dof[t], color="k", ls="--", lw=1.7)

                if t < 5:
                    ax_right.tick_params(axis="x", labelbottom=False)
                else:
                    ax_right.set_xlabel(r"$\chi^2$", fontsize=14)
                    ax_right.tick_params(axis="x", labelsize=12)

                ax_right.set_yticklabels([])

    # Add shared y-axis label for left panels
    fig.text(
        0.0,
        0.5,
        r"$R^{\mathrm{galaxy}}_{\,\mathrm{H}}\,(h^{-1}\mathrm{Mpc})$",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    plt.show()
