# Compare homogeneity scale fits using different bin widths
import os
import numpy as np
import pandas as pd
from fit_mock_average_desilike import setup_fitting
from python_routines import get_mocks, run_bestfit, run_bestfit_spline, run_emcee, plot_poly_samples

if "__main__" in __name__:

    tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
    model = "bspline"
    cov_factors = {
        "NGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
        "SGC": [1.39, 1.15, 1.15, 1.22, 1.29, 1.11],
    }

    # Fit the mock average of each tracer with a 3rd, 4th, 5th, 6th rm order polynomial
    for t, tracer in enumerate(tracers):
        for skyarea in ["NGC", "SGC"]:
            AbacusSummit, EZmock = get_mocks(tracer, skyarea, "weighted", complete=False, rebin=rebin)

            samples = {}
            filename = f"{tracer}/CD_AbacusSummit_{skyarea}_mean_{model}_rebin{rebin}_samples_emcee"
            if os.path.isfile(filename + ".txt"):
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
                samples[model] = pd.read_csv(
                    filename + ".txt",
                    header=None,
                    sep=r"\s+",
                    names=colnames,
                )
            else:
                if model == "bspline":
                    coeffs, chi2, polyfit, _ = run_bestfit_spline(
                        AbacusSummit,
                        rmin=rmins[tracer],
                        rmax=rmaxs[tracer],
                        datatype="D2",
                        plot=False,
                        single_cov=False,
                        cov_factor=cov_factors[skyarea][t],
                    )
                    print(tracer, skyarea, chi2, len(polyfit) - (len(polyfit) // 3 - 4))
                else:
                    coeffs, chi2, polyfit, _ = run_bestfit(
                        AbacusSummit,
                        model=model,
                        rmin=rmins[tracer],
                        rmax=rmaxs[tracer],
                        datatype="D2",
                        plot=False,
                        single_cov=False,
                        cov_factor=cov_factors[skyarea][t],
                    )
                    print(tracer, skyarea, chi2, len(polyfit) - int(model[-1]) - 1)
                samples[model] = run_emcee(
                    AbacusSummit,
                    coeffs,
                    model=model,
                    rmin=rmins[tracer],
                    rmax=rmaxs[tracer],
                    datatype="D2",
                    single_cov=False,
                    cov_factor=cov_factors[skyarea][t],
                )
                np.savetxt(filename + ".txt", samples[model].to_numpy(), fmt="%g")

            plot_poly_samples(
                filename + ".png",
                tracer,
                colors[tracer],
                names[t],
                AbacusSummit,
                samples[model],
                rmins[tracer],
                rmaxs[tracer],
                model=model,
                mock="mean",
            )
