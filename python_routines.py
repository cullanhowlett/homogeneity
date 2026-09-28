import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.interpolate import CubicSpline
from chainconsumer import ChainConsumer, Chain
import matplotlib.patheffects as path_effects
from scipy.interpolate import BSpline, PPoly


def get_mocks(tracer, skyarea, sysweight, complete=False, rebin=1, findiff=False):

    def load_mock(sim, prefix, idx):
        base = f"./{tracer}/{sim}/MSC_{tracer}_{prefix}"
        read = lambda p: pd.read_csv(p, sep=r"\s+", names=["r_edges", "MSC", "counts"], header=None, skiprows=1)
        if skyarea == "GCcomb":
            paths = [f"{base}{cap}_mock{idx}_{sysweight}.txt" for cap in ("NGC", "SGC")]
            dfs = [read(p) for p in paths]
            rho = [
                float(dict(s.split("=") for s in open(p).readline().strip("# \n").split())["rand_density"])
                for p in paths
            ]
            MSC = pd.DataFrame(
                {
                    "r_edges": dfs[0]["r_edges"],
                    "counts": dfs[0]["counts"] + dfs[1]["counts"],
                    "MSC": (rho[0] * dfs[0]["MSC"] + rho[1] * dfs[1]["MSC"]) / (rho[0] + rho[1]),
                }
            )
        else:
            MSC = read(f"{base}{skyarea}_mock{idx}_{sysweight}.txt")
        MSC = MSC.iloc[::rebin].reset_index(drop=True)
        radii, lnN = MSC["r_edges"].to_numpy(), np.log(MSC["MSC"].to_numpy())
        lnr, midpoint = np.log(radii), (radii[:-1] + radii[1:]) / 2.0
        CD = 3.0 + (
            (lnN[1:] - lnN[:-1]) / (lnr[1:] - lnr[:-1])
            if findiff
            else CubicSpline(lnr, lnN).derivative()(np.log(midpoint))
        )
        return radii, midpoint, MSC["MSC"].to_numpy(), CD

    # EZmock data (all 1000 required)
    ez = [load_mock("EZmock", "", i + 1) for i in range(1000)]
    radii, midpoint = ez[0][:2]
    nradii, nmidpoint = len(radii), len(midpoint)
    MSC_data, CD_data = np.array([m[2] for m in ez]), np.array([m[3] for m in ez])

    EZmock = {
        "r_edges": radii,
        "r": midpoint,
        "MSC": MSC_data,
        "MSC_mean": np.mean(MSC_data, axis=0),
        "MSC_cov": np.cov(MSC_data, rowvar=False),
        "D2": CD_data,
        "D2_mean": np.mean(CD_data, axis=0),
        "D2_cov": np.cov(CD_data, rowvar=False),
    }

    # AbacusSummit data (tolerates mocks that haven't finished running; missing mocks become NaN rows)
    comp_str = "complete_" if complete else ""
    ab = []
    for i in range(25):
        try:
            ab.append(load_mock("AbacusSummit", comp_str, i))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            ab.append(None)

    valid = np.array([m is not None for m in ab])
    if not valid.any():
        raise FileNotFoundError(f"No AbacusSummit mocks found for {tracer} {skyarea} {sysweight}")
    print(f"AbacusSummit: using {valid.sum()}/25 mocks (missing: {np.where(~valid)[0].tolist()})")

    radii, midpoint, MSC_ref, CD_ref = next(m for m in ab if m is not None)
    MSC_data = np.array([m[2] if m is not None else np.full_like(MSC_ref, np.nan) for m in ab])
    CD_data = np.array([m[3] if m is not None else np.full_like(CD_ref, np.nan) for m in ab])

    AbacusSummit = {
        "r_edges": radii[:nradii],
        "r": midpoint[:nmidpoint],
        "MSC": MSC_data[:, :nradii],
        "MSC_mean": np.nanmean(MSC_data, axis=0)[:nradii],
        "MSC_cov": EZmock["MSC_cov"],
        "D2": CD_data[:, :nmidpoint],
        "D2_mean": np.nanmean(CD_data, axis=0)[:nmidpoint],
        "D2_cov": EZmock["D2_cov"],
        "valid": valid,
        "n_valid": int(valid.sum()),
    }

    return AbacusSummit, EZmock


# Routine to read in the correlation dimension measured from real data
def get_data(tracer, skyarea, sysweight, rebin=1, findiff=False):

    # EZmock data
    MSC_data, CD_data = [], []
    for i in range(1000):
        if skyarea == "GCcomb":
            with open(f"./{tracer}/EZmock/MSC_{tracer}_NGC_mock{i+1}_{sysweight}.txt", "r") as f:
                header_line = f.readline().strip("# \n")
                metadata = dict(item.split("=") for item in header_line.split())
                rand_density_NGC = float(metadata["rand_density"])
            MSC_NGC = pd.read_csv(
                f"./{tracer}/EZmock/MSC_{tracer}_NGC_mock{i+1}_{sysweight}.txt",
                sep=r"\s+",
                names=["r_edges", "MSC", "counts"],
                header=None,
                skiprows=1,
            )
            with open(f"./{tracer}/EZmock/MSC_{tracer}_SGC_mock{i+1}_{sysweight}.txt", "r") as f:
                header_line = f.readline().strip("# \n")
                metadata = dict(item.split("=") for item in header_line.split())
                rand_density_SGC = float(metadata["rand_density"])
            MSC_SGC = pd.read_csv(
                f"./{tracer}/EZmock/MSC_{tracer}_SGC_mock{i+1}_{sysweight}.txt",
                sep=r"\s+",
                names=["r_edges", "MSC", "counts"],
                header=None,
                skiprows=1,
            )
            MSC = pd.DataFrame()
            MSC["r_edges"] = MSC_NGC["r_edges"]
            MSC["MSC"] = (rand_density_NGC * MSC_NGC["MSC"] + rand_density_SGC * MSC_SGC["MSC"]) / (
                rand_density_NGC + rand_density_SGC
            )
            MSC["counts"] = MSC_NGC["counts"] + MSC_SGC["counts"]
        else:
            MSC = pd.read_csv(
                f"./{tracer}/EZmock/MSC_{tracer}_{skyarea}_mock{i+1}_{sysweight}.txt",
                sep=r"\s+",
                names=["r_edges", "MSC", "counts"],
                header=None,
                skiprows=1,
            )
        MSC = MSC.iloc[::rebin].reset_index(drop=True)

        radii = MSC["r_edges"].to_numpy()
        midpoint = (radii[:-1] + radii[1:]) / 2.0
        if findiff:
            lnr = np.log(MSC["r_edges"].to_numpy())
            lnN = np.log(MSC["MSC"].to_numpy())
            CD = 3.0 + (lnN[1:] - lnN[:-1]) / (lnr[1:] - lnr[:-1])
        else:
            CD = 3.0 + CubicSpline(np.log(MSC["r_edges"].to_numpy()), np.log(MSC["MSC"].to_numpy())).derivative()(
                np.log(midpoint)
            )

        MSC_data.append(MSC["MSC"].to_numpy())
        CD_data.append(CD)

    nradii, nmidpoint = len(radii), len(midpoint)
    MSC_data, CD_data = np.array(MSC_data), np.array(CD_data)
    MSC_mean, MSC_cov = np.mean(MSC_data, axis=0), np.cov(MSC_data, rowvar=False)
    CD_mean, CD_cov = np.mean(CD_data, axis=0), np.cov(CD_data, rowvar=False)

    EZmock = {
        "r_edges": radii,
        "r": midpoint,
        "MSC": MSC_data,
        "MSC_mean": MSC_mean,
        "MSC_cov": MSC_cov,
        "D2": CD_data,
        "D2_mean": CD_mean,
        "D2_cov": CD_cov,
    }

    # Real data
    if skyarea == "GCcomb":
        with open(f"./{tracer}/MSC_{tracer}_NGC_data_{sysweight}.txt", "r") as f:
            header_line = f.readline().strip("# \n")
            metadata = dict(item.split("=") for item in header_line.split())
            rand_density_NGC = float(metadata["rand_density"])
        MSC_NGC = pd.read_csv(
            f"./{tracer}/MSC_{tracer}_NGC_data_{sysweight}.txt",
            sep=r"\s+",
            names=["r_edges", "MSC", "counts"],
            header=None,
            skiprows=1,
        )
        with open(f"./{tracer}/MSC_{tracer}_NGC_data_{sysweight}.txt", "r") as f:
            header_line = f.readline().strip("# \n")
            metadata = dict(item.split("=") for item in header_line.split())
            rand_density_SGC = float(metadata["rand_density"])
        MSC_SGC = pd.read_csv(
            f"./{tracer}/MSC_{tracer}_SGC_data_{sysweight}.txt",
            sep=r"\s+",
            names=["r_edges", "MSC", "counts"],
            header=None,
            skiprows=1,
        )
        MSC = pd.DataFrame()
        MSC["r_edges"] = MSC_NGC["r_edges"]
        MSC["MSC"] = (rand_density_NGC * MSC_NGC["MSC"] + rand_density_SGC * MSC_SGC["MSC"]) / (
            rand_density_NGC + rand_density_SGC
        )
        MSC["counts"] = MSC_NGC["counts"] + MSC_SGC["counts"]
    else:
        MSC = pd.read_csv(
            f"./{tracer}/MSC_{tracer}_{skyarea}_data_{sysweight}.txt",
            sep=r"\s+",
            names=["r_edges", "MSC", "counts"],
            header=None,
            skiprows=1,
        )
    MSC = MSC.iloc[::rebin].reset_index(drop=True)

    radii = MSC["r_edges"].to_numpy()
    midpoint = (radii[:-1] + radii[1:]) / 2.0
    if findiff:
        lnr = np.log(MSC["r_edges"].to_numpy())
        lnN = np.log(MSC["MSC"].to_numpy())
        CD = 3.0 + (lnN[1:] - lnN[:-1]) / (lnr[1:] - lnr[:-1])
    else:
        CD = 3.0 + CubicSpline(np.log(MSC["r_edges"].to_numpy()), np.log(MSC["MSC"].to_numpy())).derivative()(
            np.log(midpoint)
        )

    Data = {
        "r_edges": radii[:nradii],
        "r": midpoint[:nmidpoint],
        "MSC": MSC["MSC"].to_numpy()[:nradii],
        "MSC_mocks": EZmock["MSC"],
        "MSC_cov": EZmock["MSC_cov"],
        "D2": CD[:nmidpoint],
        "D2_mocks": EZmock["D2"],
        "D2_cov": EZmock["D2_cov"],
    }

    return Data


# Priors for polynomial fitting
def lnprior_poly(params):
    if np.any(params < -50.0) or np.any(params > 50.0):
        return -np.inf
    return 0.0


# Likelihood for polynomial fitting
def lnlike_poly(xdata, ydata, invcov, corrfac, params):
    model = np.poly1d(params)(xdata / 100.0)
    chi2 = (ydata - model) @ invcov @ (ydata - model)
    return -0.5 * chi2 / corrfac


# Posterior for polynomial fitting
def lnpost_poly(xdata, ydata, invcov, corrfac, params):
    prior = lnprior_poly(params)
    return np.where(np.isinf(prior), -np.inf, lnlike_poly(xdata, ydata, invcov, corrfac, params))


# Add these likelihood functions for B-splines
def lnprior_bspline(coeffs):
    # Adjust bounds based on your data range (for D2, typically 2.8-3.0)
    if np.any(coeffs < 2.0) or np.any(coeffs > 4.0):
        return -np.inf
    return 0.0


def lnlike_bspline(knots, degree, xdata, ydata, invcov, corrfac, coeffs):
    spline = BSpline(knots, coeffs, degree)
    model = spline(xdata)
    chi2 = (ydata - model) @ invcov @ (ydata - model)
    return -0.5 * chi2 / corrfac


def lnpost_bspline(knots, degree, xdata, ydata, invcov, corrfac, coeffs):
    prior = lnprior_bspline(coeffs)
    if np.isinf(prior):
        return -np.inf
    return prior + lnlike_bspline(knots, degree, xdata, ydata, invcov, corrfac, coeffs)


def grab_data(data, mock=None, single_cov=True, rmin=30.0, rmax=200.0, datatype="D2", nsims=1000):

    if datatype == "MSC":
        index = np.where((data["r_edges"] > rmin) & (data["r_edges"] < rmax))[0]
        data_x = data["r_edges"][index]
        data_y = (
            data["MSC_mean"][index]
            if mock is None
            else data["MSC"][index] if mock == "data" else data["MSC"][mock][index]
        )
        cov = (
            data["MSC_cov"][np.ix_(index, index)] / np.shape(data["MSC"])[0]
            if mock is None
            else data["MSC_cov"][np.ix_(index, index)]
        )
    else:
        index = np.where((data["r"] > rmin) & (data["r"] < rmax))[0]
        data_x = data["r"][index]
        data_y = (
            data["D2_mean"][index] if mock is None else data["D2"][index] if mock == "data" else data["D2"][mock][index]
        )
        cov = (
            data["D2_cov"][np.ix_(index, index)] / np.shape(data["D2"])[0]
            if mock is None
            else data["D2_cov"][np.ix_(index, index)]
        )

    if mock is None and single_cov:
        cov *= np.shape(data["MSC" if datatype == "MSC" else "D2"])[0]

    # Need to Hartlap correct covariance!!!!
    invcov = np.linalg.inv(cov)
    invcov *= 1.0 - ((1.0 + np.shape(cov)[0]) / (nsims - 1.0))

    # Correction for the finite number of simulations used to estimate the covariance matrix
    corrA = 2.0 / ((nsims - len(data_x) - 1.0) * (nsims - len(data_x) - 4.0))
    corrB = corrA * (nsims - len(data_x) - 2.0) / 2.0

    return data_x, data_y, cov, invcov, corrA, corrB


# Routine to find the bestfit model for a MSC or D2 data vector. Can use either a polynomial model or a physical model
def run_bestfit(
    data,
    mock=None,
    single_cov=True,
    rmin=30,
    rmax=200,
    model="poly5",
    datatype="D2",
    nsims=1000,
    name=None,
    plot=False,
    cov_factor=1.0,
):

    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data,
        rmin=rmin,
        rmax=rmax,
        datatype=datatype,
        single_cov=single_cov,
        mock=mock,
        nsims=nsims,
    )

    corrfac = (
        cov_factor
        * (1.0 + corrB * (len(data_x) - (int(model[-1]) + 1)))
        / (1.0 + corrA + corrB * (1.0 + (int(model[-1]) + 1)))
    )

    # Design matrix
    X = np.vander(data_x / 100.0, (int(model[-1]) + 1), increasing=True)

    # Weighted least squares: solve (X^T Cov^{-1} X) coeffs = X^T Cov^{-1} data
    A = X.T @ (invcov / corrfac) @ X
    b = X.T @ (invcov / corrfac) @ data_y
    coeffs = np.linalg.solve(A, b)
    chi2 = data_y @ (invcov / corrfac) @ data_y - coeffs @ b
    coeffs = coeffs[::-1]

    R_H = RH(coeffs, rmin=rmin, model=model, crossing=1.01 if datatype == "MSC" else 2.97)[0]

    if plot:
        plot_bestfit(data, coeffs, mock=mock, rmin=rmin, rmax=rmax, model=model, datatype=datatype, name=name)

    return coeffs, chi2, np.poly1d(coeffs)(data_x / 100.0), R_H


# Plot the correlation dimension and the constraints on the homogeneity scale
def plot_bestfit(data, params, mock=None, rmin=30, rmax=200, model="poly5", datatype="D2", nsims=1000, name=None):
    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data,
        rmin=data["r_edges"][0],
        rmax=data["r_edges"][-1],
        datatype=datatype,
        mock=mock,
        nsims=nsims,
    )
    data_err = np.sqrt(np.diag(cov))

    fig = plt.figure()
    ax = fig.add_axes((0.13, 0.13, 0.85, 0.85))
    ax.errorbar(data_x, data_y, yerr=data_err, color="k", ls="None", marker="o", markersize=4, alpha=0.4, zorder=1)
    best_model = np.poly1d(params)(data_x / 100.0)
    R_H = RH(params, rmin=rmin, model=model, crossing=1.01 if datatype == "MSC" else 2.97)[0]
    ax.plot(data_x, best_model, color="r", ls="-", marker="None", zorder=4)
    ax.errorbar(
        R_H,
        1.01 if datatype == "MSC" else 2.97,
        mec="k",
        mfc="b",
        ls="None",
        marker="s",
        markersize=10,
        zorder=5,
    )
    ax.text(
        0.02,
        0.91,
        name,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=14,
        color="k",
    )
    ax.axvline(x=rmin, color="k", ls=":", lw=1.3)
    ax.axvline(x=rmax, color="k", ls=":", lw=1.3)
    ax.set_xlabel(r"$r\,[h^{-1}Mpc]$", fontsize=14)
    if datatype == "MSC":
        ax.axhline(y=1.00, color="k", ls="-", lw=1.3)
        ax.axhline(y=1.01, color="k", ls="--", lw=1.3)
        ax.set_xlim(10.0, 200.0)
        ax.set_ylim(0.99, 3.0)
        ax.set_ylabel(r"$N(<r)$", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.axhline(y=3.00, color="k", ls="-", lw=1.3)
        ax.axhline(y=2.97, color="k", ls="--", lw=1.3)
        ax.set_xlim(30.0, 200.0)
        ax.set_ylim(2.85, 3.01)
        ax.set_ylabel(r"$D_{2}(r)$", fontsize=14)
    plt.show()

    return


def run_bestfit_spline(
    data,
    mock=None,
    single_cov=True,
    rmin=30,
    rmax=200,
    datatype="D2",
    nsims=1000,
    name=None,
    plot=False,
    cov_factor=1.0,
    knot_factor=3,
):

    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data,
        rmin=rmin,
        rmax=rmax,
        datatype=datatype,
        single_cov=single_cov,
        mock=mock,
        nsims=nsims,
    )

    # Create B-spline knots
    n_knots, degree = len(data_x) // knot_factor, 3
    knots_interior = np.linspace(data_x.min(), data_x.max(), n_knots)[1:-1]
    knots = np.concatenate([[data_x.min()] * degree, knots_interior, [data_x.max()] * degree])
    n_basis = len(knots) - degree - 1

    corrfac = cov_factor * (1.0 + corrB * (len(data_x) - n_basis)) / (1.0 + corrA + corrB * (1.0 + n_basis))

    # Design matrix: evaluate each B-spline basis at data points
    X = np.zeros((len(data_x), n_basis))
    for i in range(n_basis):
        coeffs_basis = np.zeros(n_basis)
        coeffs_basis[i] = 1.0
        basis_spline = BSpline(knots, coeffs_basis, degree)
        X[:, i] = basis_spline(data_x)

    # Weighted least squares: solve (X^T Cov^{-1} X) coeffs = X^T Cov^{-1} data
    A = X.T @ (invcov / corrfac) @ X
    b = X.T @ (invcov / corrfac) @ data_y
    coeffs = np.linalg.solve(A, b)
    chi2 = data_y @ (invcov / corrfac) @ data_y - coeffs @ b

    # Create fitted spline (store knots and coeffs for RH function)
    spline_params = (knots, coeffs, degree)

    # Find R_H crossing using RH function
    R_H = RH(spline_params, rmin=rmin, model="bspline", crossing=1.01 if datatype == "MSC" else 2.97)[0]

    if plot:
        plot_bestfit_spline(
            data,
            spline_params,
            mock=mock,
            rmin=rmin,
            rmax=rmax,
            datatype=datatype,
            name=name,
            nsims=nsims,
        )

    # Create fitted spline for evaluation
    fitted_spline = BSpline(knots, coeffs, degree)

    return spline_params, chi2, fitted_spline(data_x), R_H


# Plot the correlation dimension and the constraints on the homogeneity scale using B-spline
def plot_bestfit_spline(data, spline_params, mock=None, rmin=30, rmax=200, datatype="D2", nsims=1000, name=None):
    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data,
        rmin=data["r_edges"][0],
        rmax=data["r_edges"][-1],
        datatype=datatype,
        mock=mock,
        nsims=nsims,
    )
    data_err = np.sqrt(np.diag(cov))

    fig = plt.figure()
    ax = fig.add_axes((0.13, 0.13, 0.85, 0.85))
    ax.errorbar(data_x, data_y, yerr=data_err, color="k", ls="None", marker="o", markersize=4, alpha=0.4, zorder=1)

    # Reconstruct spline
    knots, coeffs, degree = spline_params
    spline = BSpline(knots, coeffs, degree)

    # Evaluate spline on data points
    best_model = spline(data_x)

    # Find R_H crossing
    R_H = RH(spline_params, rmin=rmin, model="bspline", crossing=1.01 if datatype == "MSC" else 2.97)[0]

    ax.plot(data_x, best_model, color="r", ls="-", marker="None", zorder=4)

    ax.errorbar(
        R_H,
        1.01 if datatype == "MSC" else 2.97,
        mec="k",
        mfc="b",
        ls="None",
        marker="s",
        markersize=10,
        zorder=5,
    )
    ax.text(
        0.02,
        0.91,
        name,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=14,
        color="k",
    )
    ax.axvline(x=rmin, color="k", ls=":", lw=1.3)
    ax.axvline(x=rmax, color="k", ls=":", lw=1.3)
    ax.set_xlabel(r"$r\,[h^{-1}Mpc]$", fontsize=14)

    if datatype == "MSC":
        ax.axhline(y=1.00, color="k", ls="-", lw=1.3)
        ax.axhline(y=1.01, color="k", ls="--", lw=1.3)
        ax.set_xlim(10.0, 200.0)
        ax.set_ylim(0.99, 3.0)
        ax.set_ylabel(r"$N(<r)$", fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.axhline(y=3.00, color="k", ls="-", lw=1.3)
        ax.axhline(y=2.97, color="k", ls="--", lw=1.3)
        ax.set_xlim(30.0, 200.0)
        ax.set_ylim(2.85, 3.01)
        ax.set_ylabel(r"$D_{2}(r)$", fontsize=14)

    plt.show()

    return


# Routine to run emcee on a D2 data vector. Can use either a polynomial model or a physical model
def run_emcee(
    data,
    start,
    mock=None,
    single_cov=True,
    rmin=30,
    rmax=200,
    model="poly5",
    datatype="D2",
    nsims=1000,
    max_n=40000,
    cov_factor=1.0,
):

    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data,
        mock=mock,
        rmin=rmin,
        rmax=rmax,
        datatype=datatype,
        nsims=nsims,
        single_cov=single_cov,
    )

    if model == "bspline":
        knots, coeffs, degree = start
        n_basis = len(knots) - degree - 1
        coords = 1.0e-5 * np.random.randn(4 * n_basis, n_basis) + coeffs
        nwalkers, ndim = coords.shape
    else:
        n_basis = int(model[-1]) + 1
        coords = 1.0e-5 * np.random.randn(4 * n_basis, n_basis) + start
        nwalkers, ndim = coords.shape

    corrfac = cov_factor * (1.0 + corrB * (len(data_x) - n_basis)) / (1.0 + corrA + corrB * (1.0 + n_basis))

    if model == "bspline":
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lambda x, *args: lnpost_bspline(*args, x),
            args=(knots, degree, data_x, data_y, invcov, corrfac),
        )
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lambda x, *args: lnpost_poly(*args, x), args=(data_x, data_y, invcov, corrfac)
        )

    # Now we'll sample for up to max_n steps
    old_tau = np.inf
    for _ in sampler.sample(coords, iterations=max_n, progress=True):

        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Check convergence
        tau = sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    if model == "bspline":
        RH_samples = np.array([RH((knots, samp, degree), rmin=rmin, model=model) for samp in samples])

        full_samples = pd.DataFrame(
            np.c_[
                samples,
                np.tile(knots, (len(samples), 1)),
                np.repeat(degree, len(samples)),
                RH_samples,
                log_prob_samples,
            ],
            columns=[f"a_{i}" for i in range(np.shape(samples)[1])]
            + [f"k_{i}" for i in range(len(knots))]
            + ["degree", "R_H", "log_post"],
        )
    else:
        RH_samples = RH(samples, rmin=rmin, model=model)
        full_samples = pd.DataFrame(
            np.c_[samples, RH_samples, log_prob_samples],
            columns=[f"a_{i}" for i in range(np.shape(samples)[1])] + ["R_H", "log_post"],
        )

    return full_samples


# Given a bunch of D2 model find the 1% homogeneity scale for each model
def RH(params, rmin=30, model="poly5", crossing=2.97):
    if model[:-1] == "poly":
        newparams = np.atleast_2d(np.copy(params))
        newparams[:, -1] -= crossing
        roots = [np.roots(n) * 100 for n in newparams]
        RHs = [
            r.real[(abs(r.imag) < 1e-5) & (r.real > rmin) & (r.real <= 240.0)] for r in roots
        ]  # Return only real roots within a wide prior
        RH = np.array([0.0 if len(R) == 0 else np.min(R) for R in RHs])
    elif model == "bspline":
        # params should be a tuple of (knots, coeffs, degree)
        knots, coeffs, degree = params

        # Convert to PPoly to get roots
        ppoly = PPoly.from_spline((knots, coeffs, degree))
        ppoly_shifted = PPoly(ppoly.c, ppoly.x)
        ppoly_shifted.c[-1, :] -= crossing  # Subtract from constant term

        roots = ppoly_shifted.roots()
        RHs = roots.real[(np.abs(roots.imag) < 1e-5) & (roots.real > rmin) & (roots.real <= 240.0)]
        RH = np.array([0.0 if len(RHs) == 0 else np.min(RHs)])
    else:
        RH = np.zeros(len(params))
    return RH


def plot_poly_samples(
    filename,
    tracer,
    color,
    name,
    data,
    samples,
    rmin,
    rmax,
    model="model",
    mock="mean",
):

    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data, rmin=data["r_edges"][0], rmax=data["r_edges"][-1], datatype="D2", mock=None if mock == "mean" else mock
    )
    data_err = np.sqrt(np.diag(cov))
    index = np.where((data_x > rmin) & (data_x < rmax))[0]

    c = ChainConsumer()
    columns_to_use = [s for s in samples.columns if "k" not in s and s != "degree"]
    c.add_chain(
        Chain(
            samples=samples[columns_to_use],
            linewidth=2.0,
            name=tracer,
        )
    )
    constraints = c.analysis.get_summary()
    post_RH = constraints[tracer]["R_H"]
    print(post_RH)

    # Calculate the values and errors
    RH_galaxy = post_RH.center
    RH_galaxy_err_upper = 0 if post_RH.upper is None else post_RH.upper - post_RH.center
    RH_galaxy_err_lower = 0 if post_RH.lower is None else post_RH.center - post_RH.lower

    # Create formatted multi-line label with asymmetric errors
    label_text = (
        f"{name}\n"
        f"$R_H^{{\\mathrm{{galaxy}}}} = {RH_galaxy:.2f}^{{+{RH_galaxy_err_upper:.2f}}}_{{-{RH_galaxy_err_lower:.2f}}}\,h^{{{-1}}}{{\\mathrm{{Mpc}}}}$"
    )

    fig = plt.figure()
    ax = fig.add_axes((0.13, 0.13, 0.85, 0.85))
    ax.errorbar(
        data_x, data_y, yerr=data_err, color=color, ls="None", marker="o", mec="k", markersize=4, alpha=0.4, zorder=3
    )
    max_post = np.argmax(samples["log_post"])
    if model == "bspline":
        degree = samples.iloc[max_post].to_numpy()[-3]
        n_coeffs = int((len(samples.columns) - degree - 4) // 2)
        coeffs, knots, degree = (
            samples.iloc[max_post].to_numpy()[:n_coeffs],
            samples.iloc[max_post].to_numpy()[n_coeffs:-3],
            int(samples.iloc[max_post].to_numpy()[-3]),
        )
        spline = BSpline(knots, coeffs, degree)
        best_model = spline(data_x)
    else:
        best_model = np.poly1d(samples.iloc[max_post].to_numpy()[:-2])(data_x / 100.0)
    ax.plot(data_x[index], best_model[index], color=color, ls="-", marker="None", zorder=4)
    ax.errorbar(
        RH_galaxy,
        2.97,
        xerr=[[RH_galaxy_err_lower], [RH_galaxy_err_upper]],
        color="b",
        mec="b",
        mfc=color,
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
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
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

    return


if "__main__" in __name__:

    # Info on all the measurements we've made
    tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2"]
    skyareas = ["NGC", "SGC", "GCcomb"]
    sysweights = ["weighted"]

    # Accumulate all the measurements
    AbacusSummit = {t: {s: {} for s in skyareas} for t in tracers}
    EZmock = {t: {s: {} for s in skyareas} for t in tracers}
    for tracer in tracers:
        for skyarea in skyareas:
            for sysweight in sysweights:
                AbacusSummit, EZmock = get_mocks(tracer, skyarea, sysweight, rebin=5)
                print(
                    tracer,
                    skyarea,
                    sysweight,
                )
