# Compare homogeneity scale fits using different bin widths
import os
import sys
from desilike import setup_logging
from fit_mock_average_desilike import setup_fitting, get_likelihood
from desilike.profilers import MinuitProfiler
from desilike.samples import Chain
from desilike.samplers import PocoMCSampler
from python_routines import get_mocks

if "__main__" in __name__:

    # Parameters from command line
    tracer = sys.argv[1]
    skyarea = sys.argv[2]
    mock = int(sys.argv[3])

    setup_logging()
    tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
    cov_factors = {
        "NGC": {"BGS": 1.39, "LRG1": 1.15, "LRG2": 1.15, "LRG3": 1.22, "ELG2": 1.29, "QSO": 1.11},
        "SGC": {"BGS": 1.39, "LRG1": 1.15, "LRG2": 1.15, "LRG3": 1.22, "ELG2": 1.29, "QSO": 1.11},
    }

    # Accumulate all the measurements
    AbacusSummit, EZmock = get_mocks(tracer, skyarea, "weighted", rebin=rebin)

    file = f"{tracer}/CD_AbacusSummit_{skyarea}_{mock}_singlecov_{model}_rebin{rebin}_samples_pocomc"
    if os.path.isfile(file + ".npy"):
        chain = Chain.load(file + ".npy")
    else:

        rhp, p, likelihood, likelihood_real = get_likelihood(
            AbacusSummit,
            rmins[tracer],
            rmaxs[tracer],
            templates[tracer],
            templates_real[tracer],
            mock=mock,
            single_cov=True,
            model=model,
            cov_factor=cov_factors[skyarea][tracer],
        )

        # Seed used to decide on starting point
        profiler = MinuitProfiler(likelihood, seed=42)
        profiles = profiler.maximize(niterations=5)
        likelihood(**profiler.profiles.bestfit.choice(varied=True))
        print(profiles.to_stats(params=p, tablefmt="pretty"))

        sampler = PocoMCSampler(likelihood, save_fn=file)
        sampler.run(min_iterations=100, check_every=50, check={"max_eigen_gr": 0.01, "min_ess": 50.0}, progress=True)
        chain = sampler.chains[0]
        chain = chain.remove_burnin(0.5)
        print(chain.to_stats(params=p, tablefmt="pretty"))
