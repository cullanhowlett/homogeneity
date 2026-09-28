# Compare homogeneity scale fits using different bin widths
import os
import matplotlib.pyplot as plt
from desilike import setup_logging
from desilike.profilers import MinuitProfiler
from desilike.samples import Chain, plotting
from desilike.samplers import PocoMCSampler
from python_routines import get_data
from fit_mock_average_desilike import setup_fitting, get_likelihood

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
            Data = get_data(tracer, skyarea, "weighted", rebin=rebin)

            rhp, p, likelihood, likelihood_real = get_likelihood(
                Data,
                rmins[tracer],
                rmaxs[tracer],
                templates[tracer],
                templates_real[tracer],
                mock="data",
                model=model,
                cov_factor=cov_factors[skyarea][t],
            )

            # Seed used to decide on starting point
            profiler = MinuitProfiler(likelihood, seed=42)
            profiles = profiler.maximize(niterations=5)
            likelihood(**profiler.profiles.bestfit.choice(varied=True))
            print(profiles.to_stats(params=p, tablefmt="pretty"))

            sampler = PocoMCSampler(
                likelihood,
                save_fn=f"{tracer}/CD_data_{skyarea}_weighted_{model}_rebin{rebin}_samples_pocomc",
                seed=42,
            )
            if os.path.isfile(sampler.save_fn[0] + ".npy"):
                chain = Chain.load(sampler.save_fn[0] + ".npy")
            else:
                sampler.run(
                    min_iterations=100, check_every=50, check={"max_eigen_gr": 0.01, "min_ess": 50.0}, progress=True
                )
                chain = sampler.chains[0]
            chain = chain.remove_burnin(0.5)
            print(chain.to_stats(params=p, tablefmt="pretty"))
            plotting.plot_triangle(chain, params=p)
    plt.show()
