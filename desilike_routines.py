import numpy as np
import matplotlib.pyplot as plt

import jax
from desilike import setup_logging
from desilike.jax import numpy as jnp
from desilike.jax import jit
from desilike.base import BaseCalculator
from desilike.likelihoods import BaseGaussianLikelihood
from desilike.theories.galaxy_clustering import (
    DirectPowerSpectrumTemplate,
    KaiserTracerCorrelationFunctionMultipoles,
    EFTLikeKaiserTracerCorrelationFunctionMultipoles,
    LPTVelocileptorsTracerCorrelationFunctionMultipoles,
)
from python_routines import get_mocks, grab_data


# Simpson's rule implementation in JAX
@jit
def simpson_jax(y, dx):
    """
    Simpson's rule integration compatible with JAX.
    For arrays with even length, uses Simpson's 3/8 rule for last 4 points.
    """
    n = len(y)

    def odd_length():
        # Standard Simpson's 1/3 rule: integral ≈ dx/3 * (y0 + 4*y1 + 2*y2 + 4*y3 + ... + yn)
        indices = jnp.arange(1, n - 1)
        coeffs = jnp.where(indices % 2 == 1, 4.0, 2.0)
        return dx / 3.0 * (y[0] + jnp.sum(coeffs * y[1:-1]) + y[-1])

    def even_length():
        # Use Simpson's 1/3 for first n-3 points, then Simpson's 3/8 for last 4 points
        # This maintains accuracy for even-length arrays
        n_13 = n - 3
        indices = jnp.arange(1, n_13 - 1)
        coeffs = jnp.where(indices % 2 == 1, 4.0, 2.0)
        integral_13 = dx / 3.0 * (y[0] + jnp.sum(coeffs * y[1 : n_13 - 1]) + y[n_13 - 1])
        # Simpson's 3/8 rule for last 4 points
        integral_38 = 3.0 * dx / 8.0 * (y[n_13 - 1] + 3.0 * y[n_13] + 3.0 * y[n_13 + 1] + y[n_13 + 2])
        return integral_13 + integral_38

    return jax.lax.cond(n % 2 == 1, odd_length, even_length)


# Cubic spline implementation in JAX
@jit
def cubic_spline_coefficients(x, y):
    """
    Compute natural cubic spline coefficients (zero second derivative at boundaries).
    Returns coefficients a, b, c, d for each segment where:
    S_i(t) = a_i + b_i*t + c_i*t^2 + d_i*t^3
    where t = (x - x_i) for x in [x_i, x_{i+1}]
    """
    n = len(x) - 1
    h = jnp.diff(x)

    # Build tridiagonal system for second derivatives (M)
    # Natural spline: M[0] = M[n] = 0

    # For interior points, we have:
    # h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    # Build the tridiagonal matrix
    diag = 2.0 * (h[:-1] + h[1:])
    upper = h[1:-1]
    lower = h[1:-1]

    # Right hand side
    dy = jnp.diff(y)
    rhs = 3.0 * (dy[1:] / h[1:] - dy[:-1] / h[:-1])

    # Solve tridiagonal system using Thomas algorithm (stable for diagonally dominant matrices)
    M_interior = jnp.linalg.solve(jnp.diag(diag) + jnp.diag(upper, 1) + jnp.diag(lower, -1), rhs)

    # Add boundary conditions (natural spline)
    M = jnp.concatenate([jnp.array([0.0]), M_interior, jnp.array([0.0])])

    # Compute spline coefficients for each segment
    a = y[:-1]
    b = dy / h - h * (2.0 * M[:-1] + M[1:]) / 3.0
    c = M[:-1]
    d = (M[1:] - M[:-1]) / (3.0 * h)

    return a, b, c, d


@jit
def evaluate_cubic_spline(x_knots, coeffs, x_eval):
    """Evaluate cubic spline at x_eval points."""
    a, b, c, d = coeffs

    # Find which segment each x_eval point belongs to
    indices = jnp.searchsorted(x_knots[1:], x_eval)
    indices = jnp.clip(indices, 0, len(x_knots) - 2)

    # Local coordinate within segment
    t = x_eval - x_knots[indices]

    # Evaluate polynomial: a + b*t + c*t^2 + d*t^3
    return a[indices] + t * (b[indices] + t * (c[indices] + t * d[indices]))


@jit
def evaluate_cubic_spline_derivative(x_knots, coeffs, x_eval):
    """Evaluate first derivative of cubic spline at x_eval points."""
    a, b, c, d = coeffs

    indices = jnp.searchsorted(x_knots[1:], x_eval)
    indices = jnp.clip(indices, 0, len(x_knots) - 2)

    t = x_eval - x_knots[indices]

    # Derivative: b + 2*c*t + 3*d*t^2
    return b[indices] + t * (2.0 * c[indices] + 3.0 * t * d[indices])


@jit
def find_spline_roots(x_knots, coeffs, y_offset=0.0):
    """
    Find roots of cubic spline (where spline crosses y_offset).
    Returns the first root found after x_knots[0].
    """
    a, b, c, d = coeffs

    # Adjust coefficients for finding roots of (spline - y_offset)
    a_adj = a - y_offset

    def find_root_in_segment(i):
        """Find root in segment [x_knots[i], x_knots[i+1]] using analytical cubic root formula."""
        # We need to solve: a_adj[i] + b[i]*t + c[i]*t^2 + d[i]*t^3 = 0

        # Check if there's a sign change in this segment
        val_start = a_adj[i]
        h = x_knots[i + 1] - x_knots[i]
        val_end = a_adj[i] + b[i] * h + c[i] * h**2 + d[i] * h**3

        has_sign_change = val_start * val_end < 0

        # Use Newton-Raphson to find root if sign change exists
        def newton_raphson():
            # Initial guess: linear interpolation
            t_init = -val_start * h / (val_end - val_start)
            t_init = jnp.clip(t_init, 0.0, h)

            # Newton-Raphson iterations with manual iteration counter
            def body_fun(carry):
                t, t_old, iter_count = carry
                f = a_adj[i] + b[i] * t + c[i] * t**2 + d[i] * t**3
                fp = b[i] + 2.0 * c[i] * t + 3.0 * d[i] * t**2
                # Avoid division by zero
                fp = jnp.where(jnp.abs(fp) < 1e-10, 1e-10, fp)
                t_new = t - f / fp
                t_new = jnp.clip(t_new, 0.0, h)
                return (t_new, t, iter_count + 1)

            def cond_fun(carry):
                t_new, t_old, iter_count = carry
                converged = jnp.abs(t_new - t_old) < 1e-8
                max_iters_reached = iter_count >= 20
                return ~converged & ~max_iters_reached

            t_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, (t_init, t_init + 1.0, 0))
            return x_knots[i] + t_final

        return jax.lax.cond(has_sign_change, newton_raphson, lambda: jnp.inf)

    # Find roots in all segments
    roots = jax.vmap(find_root_in_segment)(jnp.arange(len(x_knots) - 1))

    # Return first valid root (not inf, and greater than x_knots[0])
    valid_roots = jnp.where((roots > x_knots[0]) & (roots < jnp.inf), roots, jnp.inf)
    first_root = jnp.min(valid_roots)

    return jnp.where(first_root < jnp.inf, first_root, 0.0)


# Updated implementations
@jit
def _get_msc_impl(sin, sedges, corr):
    maxinds = jnp.searchsorted(sin, sedges)

    def integrate_segment(maxind):
        # Create a mask instead of slicing
        mask = jnp.arange(len(sin)) < maxind
        y = jnp.where(mask, sin**2 * corr, 0.0)
        return simpson_jax(y, dx=sin[1] - sin[0])

    integrals = jax.vmap(integrate_segment)(maxinds)

    MSC = 1.0 + 3.0 / sedges**3 * integrals
    MSC = jnp.where(MSC < 0.0, 1.0e-30, MSC)
    return MSC


@jit
def _get_cd_impl(sin, sedges, s, corr):
    msc = _get_msc_impl(sin, sedges, corr)

    log_sedges = jnp.log(sedges)
    log_msc = jnp.log(msc)
    log_s = jnp.log(s)

    # Fit cubic spline and evaluate its derivative
    spline_coeffs = cubic_spline_coefficients(log_sedges, log_msc)
    dlog_msc_dlog_s = evaluate_cubic_spline_derivative(log_sedges, spline_coeffs, log_s)

    cd = 3.0 + dlog_msc_dlog_s
    cd = jnp.where(jnp.isnan(cd), 0.0, cd)

    return msc, cd


@jit
def _get_RH_impl(x, cd, crossing=2.97):
    # Fit cubic spline to (x, cd - crossing) and find its roots
    y_shifted = cd - crossing
    spline_coeffs = cubic_spline_coefficients(x, y_shifted)
    root = find_spline_roots(x, spline_coeffs, y_offset=0.0)

    return root


class BaseTheoryCorrelationDimension(BaseCalculator):
    """Base class for theory correlation dimension."""

    _initialize_with_namespace = True

    def initialize(self, s=None):
        if s is None:
            s = np.linspace(20.0, 300, 101)
        self.s = np.array(s, dtype="f8")

    def __getstate__(self):
        state = {}
        for name in ["s", "z", "cd", "fiducial"]:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state


class BaseTheoryCorrelationDimensionFromCorrelationFunctionMultipoles(BaseTheoryCorrelationDimension):
    """Base class for theoretical correlation dimension from correlation function multipoles."""

    _initialize_with_namespace = True

    def initialize(self, s=None, corr=None, interp_order=1, **kwargs):
        if s is None:
            s = np.arange(20.0, 300, 1)
        self.s = np.array(s, dtype="f8")
        self.sedges = np.concatenate(
            [self.s - (self.s[1] - self.s[0]) / 2.0, [self.s[-1] + (self.s[1] - self.s[0]) / 2.0]]
        )
        self.interp_order = {"linear": 1, "cubic": 3}.get(interp_order, interp_order)
        allowed_interp_order = [1, 3]
        if self.interp_order not in allowed_interp_order:
            raise ValueError("interp_order must be one of {}".format(allowed_interp_order))
        if corr is None:
            from desilike.theories.galaxy_clustering.full_shape import KaiserTracerCorrelationFunctionMultipoles

            corr = KaiserTracerCorrelationFunctionMultipoles()
        self.corr = corr
        self.corr.init.update(**kwargs)
        sin = self.corr.init.get("s", None)
        if sin is None:
            self.sin = jnp.arange(0.1, 300, 0.1)
        else:
            self.sin = jnp.array(sin, dtype="f8")
        self.corr.init["s"] = self.sin
        self.set_params()

    def set_params(self):
        self.corr.init.params = self.init.params.copy()
        if "RH" in self.corr.init.params:
            self.corr.init.params.__delitem__("RH")
        if "RHp" in self.corr.init.params:
            self.corr.init.params.__delitem__("RHp")
        [self.init.params.__delitem__(item) for item in self.corr.init.params]

    # def get_msc(self, corr):
    #    maxinds = np.searchsorted(self.sin, self.sedges)
    #    MSC = 1.0 + 3.0 / self.sedges**3 * np.array(
    #        [
    #            integrate.simpson(self.sin[:maxind] ** 2 * corr[:maxind], dx=self.sin[1] - self.sin[0])
    #            for maxind in maxinds
    #        ]
    #    )
    #    MSC[MSC < 0.0] = 1.0e-30
    #    return np.array(MSC)

    # def get_cd(self, corr):
    #    msc = self.get_msc(corr)
    #    cd = 3.0 + CubicSpline(np.log(self.sedges), np.log(msc)).derivative()(np.log(self.s))
    #    cd[np.isnan(cd)] = 0.0
    #    return np.array(msc), np.array(cd)

    # def get_RH(self, x, cd, crossing=2.97):
    #    # Solve for the homogeneity scale by finding the point at which the Correlation Dimension crosses the crossing point
    #    cdspline = CubicSpline(x, cd - crossing)
    #    roots = cdspline.roots()
    #    # For strange models, we occasionally get negative roots or multiple roots at small scales. So put a prior on this.
    #    roots = roots[roots > x[0]]
    #    return 0.0 if len(roots) == 0 else roots[0]

    def get_msc(self, corr):
        return _get_msc_impl(self.sin, self.sedges, corr)

    def get_cd(self, corr):
        return _get_cd_impl(self.sin, self.sedges, self.s, corr)

    def get_RH(self, x, cd, crossing=2.97):
        return _get_RH_impl(x, cd, crossing)

    def calculate(self):
        self.msc, self.cd = self.get_cd(self.corr.corr[0])
        self.RH = self.get_RH(self.s, self.cd)
        self.RHp = self.RH


class BaseTracerCorrelationDimensionFromCorrelationFunctionMultipoles(
    BaseTheoryCorrelationDimensionFromCorrelationFunctionMultipoles
):
    """Base class for perturbation theory tracer correlation function multipoles as Hankel transforms of the power spectrum multipoles."""

    _initialize_with_namespace = True
    config_fn = "correlation_dimension.yaml"

    def initialize(self, *args, pt=None, template=None, **kwargs):
        corr = globals()[self.__class__.__name__.replace("CorrelationDimension", "CorrelationFunctionMultipoles")]()
        if pt is not None:
            corr.init.update(pt=pt)
        if template is not None:
            corr.init.update(template=template)
        super(BaseTracerCorrelationDimensionFromCorrelationFunctionMultipoles, self).initialize(
            *args, corr=corr, **kwargs
        )
        for name in ["z", "ells"]:
            setattr(self, name, getattr(self.corr, name))

    def calculate(self):
        for name in ["z", "ells"]:
            setattr(self, name, getattr(self.corr, name))
        super(BaseTracerCorrelationDimensionFromCorrelationFunctionMultipoles, self).calculate()

    @property
    def pt(self):
        return self.corr.pt

    @property
    def template(self):
        return self.corr.template

    def get(self):
        return self.cd


class KaiserTracerCorrelationDimension(BaseTracerCorrelationDimensionFromCorrelationFunctionMultipoles):
    r"""
    Kaiser tracer correlation dimension.
    For the matter (unbiased) correlation function, set b1=1 and sn0=0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate correlation dimension.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """


class LPTVelocileptorsTracerCorrelationDimension(BaseTracerCorrelationDimensionFromCorrelationFunctionMultipoles):
    r"""
    Velocileptors LPT tracer correlation dimension.
    Can be exactly marginalized over counter terms and stochastic parameters alpha*, sn*.
    For the matter (unbiased) correlation function, set all bias parameters to 0.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    prior_basis : str, default='physical'
        If 'physical', use physically-motivated prior basis for bias parameters, counterterms and stochastic terms:
        :math:`b_{1}^\prime = (1 + b_{1}) \sigma_{8}(z), b_{2}^\prime = b_{2} \sigma_{8}(z)^2, b_{s}^\prime = b_{s} \sigma_{8}(z)^2, b_{3}^\prime = b_{3} \sigma_{8}(z)^3`
        :math:`\alpha_{0} = (1 + b_{1})^{2} \alpha_{0}^\prime, \alpha_{2} = f (1 + b_{1}) (\alpha_{0}^\prime + \alpha_{2}^\prime), \alpha_{4} = f (f \alpha_{2}^\prime + (1 + b_{1}) \alpha_{4}^\prime), \alpha_{6} = f^{2} \alpha_{4}^\prime`.

    **kwargs : dict
        Velocileptors options, defaults to: ``use_Pzel=False, kIR=0.2, cutoff=10, extrap_min=-5, extrap_max=3, N=4000, nthreads=1, jn=5``.


    Reference
    ---------
    - https://arxiv.org/abs/2005.00523
    - https://arxiv.org/abs/2012.04636
    - https://github.com/sfschen/velocileptors
    """

    _params = LPTVelocileptorsTracerCorrelationFunctionMultipoles._params


class EFTLikeKaiserTracerCorrelationDimension(BaseTracerCorrelationDimensionFromCorrelationFunctionMultipoles):
    r"""
    EFT-like Kaiser tracer correlation dimension.
    Can be exactly marginalized over counter terms and stochastic parameters ct*, sn*.

    Parameters
    ----------
    s : array, default=None
        Theory separations where to evaluate multipoles.

    ells : tuple, default=(0, 2, 4)
        Multipoles to compute.

    template : BasePowerSpectrumTemplate
        Power spectrum template. Defaults to :class:`DirectPowerSpectrumTemplate`.

    **kwargs : dict
        Options, defaults to: ``mu=8``.
    """


class CorrelationDimensionLikelihood(BaseGaussianLikelihood):

    def initialize(self, xdata, ydata, covariance=None, precision=None, theory=None):

        self.xdata = xdata
        self.covariance = covariance
        super(CorrelationDimensionLikelihood, self).initialize(ydata, covariance=covariance, precision=precision)
        self.theory = (
            KaiserTracerCorrelationDimension(template=DirectPowerSpectrumTemplate()) if theory is None else theory
        )
        self.theory.init.update(s=self.xdata)

    @property
    def flattheory(self):
        return self.theory.cd

    def plot(self, show=True):
        ax = plt.gca()
        ax.errorbar(
            self.xdata,
            self.flatdata,
            yerr=np.diag(self.covariance) ** 0.5,
            color="k",
            linestyle="none",
            marker="o",
            label="data",
        )
        ax.plot(self.xdata, self.flattheory, color="r", label="theory")
        ax.errorbar(self.theory.RH, 2.97, color="b", marker="s", ls=None, ms=8)
        ax.axhline(y=2.97, ls="--", color="k")
        ax.axvline(x=self.theory.RH, ls="-", color="b")
        ax.grid()
        ax.legend()
        if show:
            plt.show()
        return ax


def plot_theory(data, params, theory=None, mock=None, rmin=50, rmax=295, nsims=1000, name=""):

    # Get the data required to evaluate the likelihood
    data_x, data_y, cov, invcov, corrA, corrB = grab_data(
        data, mock=mock, rmin=rmin, rmax=rmax, datatype="D2", nsims=nsims
    )
    data_err = np.sqrt(np.diag(cov))

    if theory is None:

        # Set up the theory and likelihood
        theory = LPTVelocileptorsTracerCorrelationDimension(
            template=DirectPowerSpectrumTemplate(z=0.0, fiducial="DESI"),
        )
        for param in theory.template.params:
            theory.template.params[param].update(fixed=True)
        for param in ["bsp", "alpha0p", "alpha2p", "alpha4p", "alpha6p"]:
            theory.params[param].update(fixed=True)

    nparams = 0
    for param in theory.template.params:
        if not theory.template.params[param].fixed:
            nparams += 1
    for param in theory.params:
        if not theory.params[param].fixed:
            nparams += 1

    corrfac = (1.0 + corrB * (len(data_x) - nparams)) / (1.0 + corrA + corrB * (1.0 + nparams))
    likelihood = CorrelationDimensionLikelihood(
        data_x,
        data_y,
        covariance=cov * corrfac,
        precision=invcov / corrfac,
        theory=theory,
    )
    likelihood()

    for datatype in ["MSC", "D2"]:
        data_x, data_y, cov, invcov, corrA, corrB = grab_data(
            data, mock=mock, rmin=10.0, rmax=300.0, datatype=datatype, nsims=nsims
        )
        data_err = np.sqrt(np.diag(cov))

        fig = plt.figure()
        cmap = plt.get_cmap("cmr.torch")
        ax = fig.add_axes((0.13, 0.13, 0.85, 0.85))
        ax.errorbar(data_x, data_y, yerr=data_err, color="k", ls="None", marker="o", markersize=4, alpha=0.4, zorder=1)
        for i, (param, c) in enumerate(zip(params, np.linspace(0.1, 0.7, len(params)))):
            loglikelihood = likelihood(b1p=param[0], b2p=param[1], bsp=param[2], alpha6p=param[3])
            s = likelihood.theory.s if datatype == "D2" else likelihood.theory.sedges
            mod = likelihood.theory.cd if datatype == "D2" else likelihood.theory.msc
            ax.plot(s, mod, color=cmap(c), ls="-", marker="None", zorder=4)
            string = f"${{{i}}}: \mathcal{{L}} = {{{loglikelihood:.2f}}}$\n"
            ax.text(
                0.75,
                0.1 * i,
                string,
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=ax.transAxes,
                fontsize=14,
                color=cmap(c),
            )
        ax.text(
            0.02,
            0.91,
            name,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=14,
            color=cmap(0.1),
        )
        ax.axvline(x=rmin, color="k", ls=":", lw=1.3)
        ax.axvline(x=rmax, color="k", ls=":", lw=1.3)
        if datatype == "MSC":
            ax.axhline(y=1.00, color="k", ls="-", lw=1.3)
            ax.axhline(y=1.01, color="k", ls="--", lw=1.3)
            ax.set_xlim(30.0, 300.0)
            ax.set_ylim(0.99, 1.04)
            ax.set_ylabel(r"$N(<r)$", fontsize=14)
        else:
            ax.axhline(y=3.00, color="k", ls="-", lw=1.3)
            ax.axhline(y=2.97, color="k", ls="--", lw=1.3)
            ax.set_xlim(30.0, 300.0)
            ax.set_ylim(2.85, 3.01)
            ax.set_ylabel(r"$D_{2}(r)$", fontsize=14)
    plt.show()

    return


class RealSpaceTemplateWrapper:
    """
    Wrapper that makes any template return f=0 and fsigma8=0

    This delegates all attribute access to the wrapped template,
    except for f and fsigma8 which are forced to 0.
    """

    def __init__(self, template):
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, "_wrapped_template", template)

    def __getattribute__(self, name):
        # Intercept f and fsigma8
        if name in ["f", "fsigma8"]:
            return 0.0
        elif name == "_wrapped_template":
            return object.__getattribute__(self, name)
        else:
            # Delegate to wrapped template
            return getattr(object.__getattribute__(self, "_wrapped_template"), name)

    def __setattr__(self, name, value):
        if name == "_wrapped_template":
            object.__setattr__(self, name, value)
        elif name in ["f", "fsigma8"]:
            # Silently ignore attempts to set f or fsigma8
            pass
        else:
            setattr(self._wrapped_template, name, value)

    def __getattr__(self, name):
        # Fallback for any attributes not handled by __getattribute__
        return getattr(self._wrapped_template, name)


if "__main__" in __name__:

    setup_logging()

    tracer = "BGS"
    tracers = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]
    redshifts = np.array([0.295, 0.510, 0.706, 0.919, 1.317, 1.491])
    templates = {t: DirectPowerSpectrumTemplate(z=z, fiducial="DESI") for t, z in zip(tracers, redshifts)}

    # Accumulate some the measurements
    AbacusSummit, EZmock = get_mocks(tracer, "GCcomb", "weighted", rebin=25, findiff=False)

    # Set up the theory and likelihood
    theory = LPTVelocileptorsTracerCorrelationDimension(
        template=templates[tracer],
    )
    for param in theory.template.params:
        theory.template.params[param].update(fixed=True)
    for param in ["alpha2p", "alpha4p", "alpha6p"]:
        theory.params[param].update(fixed=True)

    # Evaluate the model and likelihood for a given set of parameters
    params = [[1.129, -0.69, 0.11, 6.6]]
    plot_theory(AbacusSummit, params, theory=theory)
