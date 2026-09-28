# Compare homogeneity scale fits using different bin widths
import os

import numpy as np
import pandas as pd
from desilike import setup_logging
from desilike.samples import Chain, plotting
from python_routines import get_data, grab_data, run_bestfit, run_bestfit_spline
from scipy.interpolate import CubicSpline
from fit_mock_average_desilike import setup_fitting, get_likelihood
from rescale_MSC import plot_rescaled
from chainconsumer import ChainConsumer, Chain as cChain


def get_rescaled_data(Data, pars, likelihood, likelihood_real):

    b1p, b2p, bsp, alpha0p, sn0p = pars

    rescaled_Data_mocks_MSC, rescaled_Data_mocks_CD = np.zeros(np.shape(Data["MSC_mocks"])), np.zeros(
        np.shape(Data["D2_mocks"])
    )

    # Now try obtaining the rescaled MSC using the model(s)
    loglikelihood = (
        likelihood(b1p=b1p, b2p=b2p, bsp=bsp, alpha0p=alpha0p, sn0p=sn0p)
        if "LPT" in likelihood.theory.__class__.__name__
        else likelihood(b1=b1p)
    )
    model = likelihood.theory.cd
    rescale = (likelihood_real.theory.msc - 1.0) / (likelihood.theory.msc - 1.0)
    rescaled_Data_MSC = np.clip(rescale * (Data["MSC"] - 1.0) + 1.0, 1.0e-30, None)
    rescaled_Data_CD = 3.0 + CubicSpline(np.log(Data["r_edges"]), np.log(rescaled_Data_MSC)).derivative()(
        np.log(Data["r"])
    )
    for j in range(len(Data["MSC_mocks"])):
        rescaled_Data_mocks_MSC[j] = np.clip(rescale * (Data["MSC_mocks"][j] - 1.0) + 1.0, 1.0e-30, None)
        rescaled_Data_mocks_CD[j] = 3.0 + CubicSpline(
            np.log(Data["r_edges"]), np.log(rescaled_Data_mocks_MSC[j])
        ).derivative()(np.log(Data["r"]))

    rescaled_Data = {
        "r_edges": Data["r_edges"],
        "r": Data["r"],
        "MSC": rescaled_Data_MSC,
        "MSC_mocks": rescaled_Data_mocks_MSC,
        "MSC_cov": np.cov(rescaled_Data_mocks_MSC, rowvar=False),
        "D2": rescaled_Data_CD,
        "D2_mocks": rescaled_Data_mocks_CD,
        "D2_cov": np.cov(rescaled_Data_mocks_CD, rowvar=False),
    }

    return rescaled_Data, model


if "__main__" in __name__:

    setup_logging()
    tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
    cov_factors = {
        "NGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
        "SGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
    }
    systematic = {
        "NGC": [0.02, 0.21, 0.01, 0.00, 0.19, 0.19],
        "SGC": [0.22, 0.10, 0.08, 0.11, 0.22, 0.21],
    }

    # Accumulate all the measurements
    for t, tracer in enumerate(tracers):
        for skyarea in ["NGC", "SGC"]:
            Data = get_data(tracer, skyarea, "weighted", rebin=rebin)

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
            rhs_full = (
                chain["RHp"].flatten() if "LPT" in likelihood.theory.__class__.__name__ else chain["RH"].flatten()
            )

            rescaled_file = f"{tracer}/CD_data_{skyarea}_weighted_{model}_rebin{rebin}_samples_pocomc_rescaled"
            if os.path.isfile(rescaled_file + ".npz"):
                models, rhs, rescaled_Data, rescaled_models, rescaled_rhs, pars = np.load(
                    rescaled_file + ".npz",
                    allow_pickle=True,
                ).values()
                rescaled_Data = rescaled_Data.item()
            else:
                b = "b1p" if "LPT" in likelihood.theory.__class__.__name__ else "b1"

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
                    rescaled_Data, old_model = get_rescaled_data(Data, par, likelihood, likelihood_real)
                    models.append(old_model)
                    coeffs, chi2, rescaled_model, rescaled_rh = run_bestfit_spline(
                        rescaled_Data,
                        rmin=rmins[tracer],
                        rmax=rmaxs[tracer],
                        datatype="D2",
                        plot=False,
                        mock="data",
                        cov_factor=cov_factors[skyarea][t],
                    )
                    print(i, tracer, skyarea, par, chi2, len(rescaled_model) - (len(rescaled_model) // 3 - 4))
                    rescaled_models.append(rescaled_model)
                    rescaled_rhs.append(rescaled_rh)
                models = np.array(models)
                rescaled_models = np.array(rescaled_models)
                rescaled_rhs = np.array(rescaled_rhs)

                np.savez(
                    rescaled_file,
                    models=models,
                    rhs=rhs,
                    rescaled_Data=rescaled_Data,
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

            filename = f"{tracer}/CD_{tracer}_data_{skyarea}_weighted_{model}_rebin{rebin}_pocomc.png"
            plot_rescaled(
                filename,
                tracer,
                colors[tracer],
                names[t],
                Data,
                rescaled_Data,
                models,
                rescaled_models,
                rhs_full,
                rescaled_rhs,
                rmins[tracer],
                rmaxs[tracer],
                mock="data",
                systematic=systematic[skyarea][t],
            )
