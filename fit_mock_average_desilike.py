# Compare homogeneity scale fits using different bin widths
import os
import numpy as np
import matplotlib.pyplot as plt
from desilike import setup_logging
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike_routines import (
    LPTVelocileptorsTracerCorrelationDimension,
    KaiserTracerCorrelationDimension,
    EFTLikeKaiserTracerCorrelationDimension,
    CorrelationDimensionLikelihood,
    RealSpaceTemplateWrapper,
)
from desilike.profilers import MinuitProfiler
from desilike.samples import Chain, plotting
from desilike.samplers import PocoMCSampler
from python_routines import get_mocks, grab_data


def setup_fitting():

    tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
    names = [
        "BGS: 0.1 < z < 0.4",
        "LRG1: 0.4 < z < 0.6",
        "LRG2: 0.6 < z < 0.8",
        "LRG3: 0.8 < z < 1.1",
        "ELG2: 1.1 < z < 1.6",
        "QSO: 0.8 < z < 2.1",
    ]
    colors = {
        "BGS": "yellowgreen",
        "LRG1": "orange",
        "LRG2": "orangered",
        "LRG3": "firebrick",
        "ELG2": "steelblue",
        "QSO": "seagreen",
    }
    redshifts = np.array([0.295, 0.510, 0.706, 0.919, 1.317, 1.491])
    templates = {t: DirectPowerSpectrumTemplate(z=z, fiducial="DESI") for t, z in zip(tracers, redshifts)}
    # templates_real = {t: DirectPowerSpectrumTemplate(z=z, fiducial="DESI") for t, z in zip(tracers, redshifts)}
    # Real-space templates: wrap in RealSpaceTemplateWrapper
    templates_real = {
        t: RealSpaceTemplateWrapper(DirectPowerSpectrumTemplate(z=z, fiducial="DESI"))
        for t, z in zip(tracers, redshifts)
    }
    rmins = {"BGS": 48.0, "LRG1": 44.0, "LRG2": 44.0, "LRG3": 40.0, "ELG2": 28.0, "QSO": 28.0}
    rmaxs = {"BGS": 160.0, "LRG1": 160.0, "LRG2": 160.0, "LRG3": 160.0, "ELG2": 132.0, "QSO": 170.0}

    model = "LPTVelocileptors"
    rebin = 8

    return tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin


def get_likelihood(
    data, rmin, rmax, template, template_real, mock=None, single_cov=True, model="LPTVelocileptors", cov_factor=1.0
):
    if model == "LPTVelocileptors":
        rhp = 5
        theory = LPTVelocileptorsTracerCorrelationDimension(
            template=template,
        )
        theory_real = LPTVelocileptorsTracerCorrelationDimension(
            template=template_real,
        )
        for param in theory.template.params:
            theory.template.params[param].update(fixed=True)
            theory_real.template.params[param].update(fixed=True)

    if model == "Kaiser":
        rhp = 2
        theory = KaiserTracerCorrelationDimension(
            template=template,
        )
        theory_real = KaiserTracerCorrelationDimension(
            template=template_real,
        )
        for param in theory.template.params:
            theory.template.params[param].update(fixed=True)
            theory_real.template.params[param].update(fixed=True)
        for param in ["sigmaper"]:
            theory.params[param].update(fixed=True)
            theory_real.params[param].update(fixed=True)

    if model == "EFTLikeKaiser":
        rhp = 3
        theory = EFTLikeKaiserTracerCorrelationDimension(
            template=template,
        )
        theory_real = EFTLikeKaiserTracerCorrelationDimension(
            template=template_real,
        )
        for param in theory.template.params:
            theory.template.params[param].update(fixed=True)
            theory_real.template.params[param].update(fixed=True)
        for param in ["ct2_2", "ct4_2"]:
            theory.params[param].update(fixed=True)
            theory_real.params[param].update(fixed=True)
        for param in ["sigmapar"]:
            theory.params[param].update(fixed=False)
            theory_real.params[param].update(fixed=False)

    nparams = 0
    for param in theory.template.params:
        if not theory.template.params[param].fixed:
            nparams += 1
    for param in theory.params:
        if not theory.params[param].fixed:
            nparams += 1
    p = theory.varied_params + ["RHp"] if "LPT" in theory.__class__.__name__ else theory.varied_params + ["RH"]

    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data,
        rmin=rmin,
        rmax=rmax,
        datatype="D2",
        single_cov=single_cov,
        mock=mock,
    )
    corrfac = cov_factor * (1.0 + corrB * (len(data_x) - nparams)) / (1.0 + corrA + corrB * (1.0 + nparams))
    likelihood = CorrelationDimensionLikelihood(
        data_x,
        data_y,
        covariance=cov * corrfac,
        precision=invcov / corrfac,
        theory=theory,
    )
    likelihood_real = CorrelationDimensionLikelihood(
        data_x,
        data_y,
        covariance=cov * corrfac,
        precision=invcov / corrfac,
        theory=theory_real,
    )
    likelihood()
    if "LPT" in theory.__class__.__name__:
        likelihood.varied_params["b1p"].update(prior={"dist": "uniform", "limits": [-0.2, 3.0]})
        likelihood.varied_params["b2p"].update(prior={"dist": "uniform", "limits": [-10.0, 10.0]})
        likelihood.varied_params["bsp"].update(prior={"dist": "uniform", "limits": [-10.0, 10.0]})
        likelihood.varied_params["alpha0p"].update(prior={"dist": "uniform", "limits": [-200.0, 200.0]})
        likelihood.varied_params["sn0p"].update(prior={"dist": "uniform", "limits": [-100.0, 100.0]})
    if "LPT" in theory.__class__.__name__:
        likelihood_real(b1p=likelihood.theory.pt.sigma8, b2p=0.0, bsp=0.0, alpha0p=0.0, sn0p=0.0)
    else:
        likelihood_real(b1=1.0)

    print(f"Redshift-space f: {likelihood.theory.template.f:.6f}")
    print(f"Real-space f: {likelihood_real.theory.template.f:.6f}")  # Should be 0.0

    return rhp, p, likelihood, likelihood_real


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
                rmins[tracer],
                rmaxs[tracer],
                templates[tracer],
                templates_real[tracer],
                model=model,
                single_cov=False,
                cov_factor=cov_factors[skyarea][t],
            )
            print(len(likelihood.xdata))

            sampler = PocoMCSampler(
                likelihood,
                save_fn=f"{tracer}/AbacusSummit/fits/CD_AbacusSummit_{skyarea}_mean_{model}_rebin{rebin}_samples_pocomc",
                seed=42,
            )
            if os.path.isfile(sampler.save_fn[0] + ".npy"):
                chain = Chain.load(sampler.save_fn[0] + ".npy")
            else:
                # Seed used to decide on starting point
                profiler = MinuitProfiler(likelihood, seed=42)
                profiles = profiler.maximize(niterations=5)
                likelihood(**profiler.profiles.bestfit.choice(varied=True))
                print(profiles.to_stats(params=p, tablefmt="pretty"))
                sampler.run(
                    min_iterations=100, check_every=50, check={"max_eigen_gr": 0.01, "min_ess": 50.0}, progress=True
                )
                chain = sampler.chains[0]
            chain = chain.remove_burnin(0.5)
            print(chain.to_stats(params=p, tablefmt="pretty"))
            plotting.plot_triangle(chain, params=p)
    plt.show()
