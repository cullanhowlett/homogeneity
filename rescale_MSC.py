# Compare homogeneity scale fits using different bin widths
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from desilike import setup_logging
from desilike.samples import Chain, plotting
from python_routines import get_mocks, grab_data, run_bestfit, run_bestfit_spline
from scipy.interpolate import CubicSpline
from fit_mock_average_desilike import setup_fitting, get_likelihood
import matplotlib.patheffects as path_effects
from chainconsumer import ChainConsumer, Chain as cChain


def get_rescaled_mocks(AbacusSummit, EZmock, pars, likelihood, likelihood_real):

    b1p, b2p, bsp, alpha0p, sn0p = pars

    rescaled_AbacusSummit_MSC = np.zeros(np.shape(AbacusSummit["MSC"]))
    rescaled_AbacusSummit_CD = np.zeros(np.shape(AbacusSummit["D2"]))
    rescaled_EZmock_MSC, rescaled_EZmock_CD = np.zeros(np.shape(EZmock["MSC"])), np.zeros(np.shape(EZmock["D2"]))

    loglikelihood = (
        likelihood(b1p=b1p, b2p=b2p, bsp=bsp, alpha0p=alpha0p, sn0p=sn0p)
        if "LPT" in likelihood.theory.__class__.__name__
        else likelihood(b1=b1p)
    )
    model = likelihood.theory.cd
    rescale = (likelihood_real.theory.msc - 1.0) / (likelihood.theory.msc - 1.0)
    for j in range(len(AbacusSummit["MSC"])):
        rescaled_AbacusSummit_MSC[j] = np.clip(rescale * (AbacusSummit["MSC"][j] - 1.0) + 1.0, 1.0e-30, None)
        rescaled_AbacusSummit_CD[j] = 3.0 + CubicSpline(
            np.log(AbacusSummit["r_edges"]), np.log(rescaled_AbacusSummit_MSC[j])
        ).derivative()(np.log(AbacusSummit["r"]))
    for j in range(len(EZmock["MSC"])):
        rescaled_EZmock_MSC[j] = np.clip(rescale * (EZmock["MSC"][j] - 1.0) + 1.0, 1.0e-30, None)
        rescaled_EZmock_CD[j] = 3.0 + CubicSpline(
            np.log(AbacusSummit["r_edges"]), np.log(rescaled_EZmock_MSC[j])
        ).derivative()(np.log(AbacusSummit["r"]))

    rescaled_EZmock = {
        "r_edges": EZmock["r_edges"],
        "r": EZmock["r"],
        "MSC": rescaled_EZmock_MSC,
        "MSC_mean": np.mean(rescaled_EZmock_MSC, axis=0),
        "MSC_cov": np.cov(rescaled_EZmock_MSC, rowvar=False),
        "D2": rescaled_EZmock_CD,
        "D2_mean": np.mean(rescaled_EZmock_CD, axis=0),
        "D2_cov": np.cov(rescaled_EZmock_CD, rowvar=False),
    }

    rescaled_AbacusSummit = {
        "r_edges": AbacusSummit["r_edges"],
        "r": AbacusSummit["r"],
        "MSC": rescaled_AbacusSummit_MSC,
        "MSC_mean": np.mean(rescaled_AbacusSummit_MSC, axis=0),
        "MSC_cov": rescaled_EZmock["MSC_cov"],
        "D2": rescaled_AbacusSummit_CD,
        "D2_mean": np.mean(rescaled_AbacusSummit_CD, axis=0),
        "D2_cov": rescaled_EZmock["D2_cov"],
    }

    return rescaled_AbacusSummit, rescaled_EZmock, model


def plot_rescaled(
    filename,
    tracer,
    color,
    name,
    data,
    rescaled_data,
    models,
    rescaled_models,
    rhs,
    rescaled_rhs,
    rmin,
    rmax,
    mock="mean",
    systematic=0.0,
):

    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data, rmin=data["r_edges"][0], rmax=data["r_edges"][-1], datatype="D2", mock=None if mock == "mean" else mock
    )
    data_err = np.sqrt(np.diag(cov))
    data_x_fit, _, _, _, _, _ = grab_data(
        data, rmin=rmin, rmax=rmax, datatype="D2", mock=None if mock == "mean" else mock
    )

    c = ChainConsumer()
    c.add_chain(cChain(samples=pd.DataFrame({"RH": rhs}), linewidth=2.0, name="biased", bins=20))
    c.add_chain(cChain(samples=pd.DataFrame({"RH": rescaled_rhs}), linewidth=2.0, name="rescaled", bins=20))
    constraints = c.analysis.get_summary()
    post_RH = constraints["biased"]["RH"]
    post_rescaled_RH = constraints["rescaled"]["RH"]

    # Calculate the values and errors
    RH_galaxy = post_RH.center
    RH_galaxy_err_upper = post_RH.upper - post_RH.center
    RH_galaxy_err_lower = post_RH.center - post_RH.lower

    RH_rescaled = post_rescaled_RH.center
    RH_rescaled_err_upper = 0 if post_rescaled_RH.upper is None else post_rescaled_RH.upper - post_rescaled_RH.center
    RH_rescaled_err_lower = 0 if post_rescaled_RH.lower is None else post_rescaled_RH.center - post_rescaled_RH.lower
    print(RH_rescaled, RH_rescaled_err_upper, RH_rescaled_err_upper)
    RH_rescaled_err_upper = np.sqrt(RH_rescaled_err_upper**2 + systematic**2)
    RH_rescaled_err_lower = np.sqrt(RH_rescaled_err_lower**2 + systematic**2)
    print(RH_rescaled, RH_rescaled_err_upper, RH_rescaled_err_upper)

    # Create formatted multi-line label with asymmetric errors
    label_text = (
        f"{name}\n"
        f"$R_H^{{\\mathrm{{galaxy}}}} = {RH_galaxy:.2f}^{{+{RH_galaxy_err_upper:.2f}}}_{{-{RH_galaxy_err_lower:.2f}}}$\n"
        f"$R_H^{{\\mathrm{{rescaled}}}} = {RH_rescaled:.2f}^{{+{RH_rescaled_err_upper:.2f}}}_{{-{RH_rescaled_err_lower:.2f}}}$"
    )
    print(label_text)

    fig = plt.figure()
    ax = fig.add_axes((0.13, 0.13, 0.85, 0.85))
    ax.errorbar(
        data_x, data_y, yerr=data_err, color=color, ls="None", marker="o", mec="k", markersize=4, alpha=0.4, zorder=3
    )
    ax.errorbar(
        rescaled_data["r"],
        (
            rescaled_data["D2_mean"]
            if mock == "mean"
            else rescaled_data["D2"] if mock == "data" else rescaled_data["D2"][mock]
        ),
        yerr=np.sqrt(np.diag(rescaled_data["D2_cov"])),
        color=color,
        mfc="w",
        mec=color,
        ls="None",
        marker="o",
        markersize=4,
        alpha=0.4,
        zorder=3,
    )
    ax.plot(data_x, np.mean(models, axis=0), color=color, ls="-", lw=1.0, marker="None", zorder=4)
    ax.plot(
        data_x_fit, np.mean(rescaled_models, axis=0), color=color, ls="-", lw=1.0, marker="None", zorder=4, alpha=0.6
    )
    ax.errorbar(
        RH_galaxy,
        2.97,
        xerr=[[RH_galaxy_err_lower], [RH_galaxy_err_upper]],
        color=color,
        mec="k",
        ls="None",
        marker="s",
        markersize=10,
        zorder=4,
    )
    ax.errorbar(
        RH_rescaled,
        2.97,
        xerr=[[RH_rescaled_err_lower], [RH_rescaled_err_upper]],
        color=color,
        mfc="w",
        ls="None",
        marker="s",
        markersize=10,
        zorder=4,
    )
    text = ax.text(
        0.98,
        0.02,
        label_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=14,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round", facecolor="white", alpha=1.0),
    )
    text.set_path_effects(
        [path_effects.withStroke(linewidth=0.5, foreground="black", alpha=0.5)]
    )  # Subtle black outline
    ax.axvline(x=rmin, color="k", ls=":", lw=1.3, zorder=1)
    ax.axvline(x=rmax, color="k", ls=":", lw=1.3, zorder=1)
    ax.axhline(y=3.00, color="k", ls="-", lw=1.3, zorder=1)
    ax.axhline(y=2.97, color="k", ls="--", lw=1.3, zorder=1)
    ax.set_xlim(30.0, 215.0)
    ax.set_ylim(2.70, 3.01)
    ax.set_xlabel(r"$r\,[h^{-1}Mpc]$", fontsize=14)
    ax.set_ylabel(r"$D_{2}(r)$", fontsize=14)
    plt.savefig(filename, dpi=300)


if "__main__" in __name__:

    setup_logging()
    tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
    cov_factors = {
        "NGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
        "SGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
    }

    # Accumulate all the measurements
    for t, tracer in enumerate(tracers):
        for skyarea in ["NGC", "SGC"]:
            AbacusSummit, EZmock = get_mocks(tracer, skyarea, "weighted", rebin=rebin)

            rhp, p, likelihood, likelihood_real = get_likelihood(
                AbacusSummit,
                AbacusSummit["r_edges"][0],
                AbacusSummit["r_edges"][-1],
                templates[tracer],
                templates_real[tracer],
                model=model,
                single_cov=False,
                cov_factor=cov_factors[skyarea][t],
            )

            infile = (
                f"{tracer}/AbacusSummit/fits/CD_AbacusSummit_{skyarea}_mean_{model}_rebin{rebin}_samples_pocomc.npy"
            )
            chain = Chain.load(infile)
            chain = chain.remove_burnin(0.5)
            print(len(chain["RHp"].flatten()))
            rhs_full = (
                chain["RHp"].flatten() if "LPT" in likelihood.theory.__class__.__name__ else chain["RH"].flatten()
            )

            rescaled_file = (
                f"{tracer}/AbacusSummit/fits/CD_AbacusSummit_{skyarea}_mean_{model}_rebin{rebin}_pocomc_rescaled"
            )
            if os.path.isfile(rescaled_file + ".npz"):
                models, rhs, rescaled_AbacusSummit, rescaled_models, rescaled_rhs, pars = np.load(
                    rescaled_file + ".npz",
                    allow_pickle=True,
                ).values()
                rescaled_AbacusSummit = rescaled_AbacusSummit.item()
            else:
                indices = np.random.choice(
                    (
                        len(chain["b1p"].flatten())
                        if "LPT" in likelihood.theory.__class__.__name__
                        else len(chain["b1"].flatten())
                    ),
                    size=1000,
                    replace=False,
                )
                b1ps = (
                    chain["b1p"].flatten()[indices]
                    if "LPT" in likelihood.theory.__class__.__name__
                    else chain["b1"].flatten()[indices]
                )
                b2ps = (
                    chain["b2p"].flatten()[indices]
                    if "LPT" in likelihood.theory.__class__.__name__
                    else np.zeros(len(indices))
                )
                bsps = (
                    chain["bsp"].flatten()[indices]
                    if "LPT" in likelihood.theory.__class__.__name__
                    else np.zeros(len(indices))
                )
                alpha0ps = (
                    chain["alpha0p"].flatten()[indices]
                    if "LPT" in likelihood.theory.__class__.__name__
                    else np.zeros(len(indices))
                )
                sn0p = (
                    chain["sn0p"].flatten()[indices]
                    if "LPT" in likelihood.theory.__class__.__name__
                    else np.zeros(len(indices))
                )
                rhs = (
                    chain["RHp"].flatten()[indices]
                    if "LPT" in likelihood.theory.__class__.__name__
                    else chain["RH"].flatten()[indices]
                )
                pars = np.stack([b1ps, b2ps, bsps, alpha0ps, sn0p]).T

                models, rescaled_models, rescaled_rhs = [], [], []
                for i, par in enumerate(pars):
                    rescaled_AbacusSummit, rescaled_EZmock, old_model = get_rescaled_mocks(
                        AbacusSummit, EZmock, par, likelihood, likelihood_real
                    )
                    models.append(old_model)
                    coeffs, chi2, rescaled_model, rescaled_rh = run_bestfit_spline(
                        rescaled_AbacusSummit,
                        rmin=rmins[tracer],
                        rmax=rmaxs[tracer],
                        single_cov=True,
                        datatype="D2",
                        cov_factor=cov_factors[skyarea][t],
                    )
                    print(i, par, chi2 * 25.0, len(rescaled_model) - 7, rescaled_rh)
                    rescaled_models.append(rescaled_model)
                    rescaled_rhs.append(rescaled_rh)
                models = np.array(models)
                rescaled_models = np.array(rescaled_models)
                rescaled_rhs = np.array(rescaled_rhs)

                np.savez(
                    rescaled_file,
                    models=models,
                    rhs=rhs,
                    rescaled_AbacusSummit=rescaled_AbacusSummit,
                    rescaled_models=rescaled_models,
                    rescaled_rhs=rescaled_rhs,
                    pars=pars,
                )

            c = ChainConsumer()
            c.add_chain(
                cChain(
                    samples=pd.DataFrame(
                        {x: chain[x].flatten() for x in ["b1p", "b2p", "bsp", "alpha0p", "sn0p", "RHp"]}
                    ),
                    linewidth=2.0,
                    name="Full Chain",
                )
            )
            c.add_chain(
                cChain(
                    samples=pd.DataFrame(
                        {x: pars[:, i] for i, x in enumerate(["b1p", "b2p", "bsp", "alpha0p", "sn0p"])}
                        | {"RHp": rhs, "RHp_rescaled": rescaled_rhs}
                    ),
                    linewidth=2.0,
                    name="Sampled Chain",
                )
            )
            constraints = c.analysis.get_summary()
            # fig = c.plotter.plot()
            # plt.show()

            filename = f"{tracer}/CD_AbacusSummit_{skyarea}_mean_{model}_rebin{rebin}_pocomc.png"
            plot_rescaled(
                filename,
                tracer,
                colors[tracer],
                names[t],
                AbacusSummit,
                rescaled_AbacusSummit,
                models,
                rescaled_models,
                rhs_full,
                rescaled_rhs,
                rmins[tracer],
                rmaxs[tracer],
            )
