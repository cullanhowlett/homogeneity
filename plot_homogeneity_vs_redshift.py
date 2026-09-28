import numpy as np
import pandas as pd
from cosmoprimo import *
from fit_mock_average_desilike import setup_fitting
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from chainconsumer import ChainConsumer, Chain as cChain
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike_routines import (
    LPTVelocileptorsTracerCorrelationDimension,
    RealSpaceTemplateWrapper,
)

if "__main__" in __name__:

    systematic = {
        "NGC": [0.02, 0.21, 0.01, 0.00, 0.19, 0.19],
        "SGC": [0.22, 0.10, 0.08, 0.11, 0.22, 0.21],
    }

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(4.5, 4.5),
        sharex=True,
        gridspec_kw={"left": 0.15, "right": 0.97, "bottom": 0.10, "top": 0.98, "hspace": 0.05},
    )

    # ------------------------------------------------------------------ #
    # Compute LCDM theory line for the top panel (dense redshift grid)
    # ------------------------------------------------------------------ #
    z_theory = np.linspace(0.0, 2.0, 101)
    rhs_theory = np.zeros_like(z_theory)
    cosmo_desi = fiducial.DESI()
    for i, z in enumerate(z_theory):
        template_real = RealSpaceTemplateWrapper(DirectPowerSpectrumTemplate(z=z, fiducial="DESI"))
        theory_real = LPTVelocileptorsTracerCorrelationDimension(template=template_real)
        theory_real()
        theory_real(b1p=theory_real.pt.sigma8, b2p=0.0, bsp=0.0, alpha0p=0.0, sn0p=0.0)
        rhs_theory[i] = theory_real.RHp
        print(i, z, rhs_theory[i])

    rh_spline = CubicSpline(z_theory, rhs_theory)

    ax_top.plot(z_theory, rhs_theory, color="black", linestyle="--", label=r"$\Lambda$CDM")

    # Bottom panel: fiducial line at 1
    ax_bot.axhline(y=1.0, color="black", linestyle="--", label=r"Fiducial $\Lambda$CDM")

    # ------------------------------------------------------------------ #
    # Literature constraints (Ntelis et al., 2017)
    # ------------------------------------------------------------------ #
    cosmo = fiducial.DESI().clone(
        omega_cdm=0.1198, omega_b=0.02225, h=0.6727, A_s=2.20652e-9, n_s=0.9645, m_ncdm=[0.0, 0.0, 0.0]
    )
    lit_reds = np.array([0.457, 0.511, 0.565, 0.619, 0.673])
    lit_rhs_ngc = np.array([64.2, 65.4, 62.6, 60.4, 59.0])
    lit_rhs_sgc = np.array([66.7, 63.9, 65.2, 60.1, 60.1])
    lit_rhs_err_ngc = np.array([1.3, 0.9, 0.8, 0.8, 0.8])
    lit_rhs_err_sgc = np.array([1.6, 1.5, 1.6, 1.1, 1.8])
    rhs_lcdm_boss = np.zeros_like(lit_reds)
    for i, z in enumerate(lit_reds):
        template_real = RealSpaceTemplateWrapper(DirectPowerSpectrumTemplate(z=z, fiducial=cosmo))
        theory_real = LPTVelocileptorsTracerCorrelationDimension(template=template_real)
        theory_real()
        theory_real(b1p=theory_real.pt.sigma8, b2p=0.0, bsp=0.0, alpha0p=0.0, sn0p=0.0)
        rhs_lcdm_boss[i] = theory_real.RHp
        print(i, z, rhs_lcdm_boss[i])

    boss_kw = dict(color="k", mfc="k", mec="k", marker="o", ls="None", alpha=0.1)
    ax_top.errorbar(
        lit_reds - 0.01,
        lit_rhs_ngc / rhs_lcdm_boss * rh_spline(lit_reds),
        yerr=lit_rhs_err_ngc / rhs_lcdm_boss * rh_spline(lit_reds),
        **boss_kw,
    )
    ax_top.errorbar(
        lit_reds + 0.01,
        lit_rhs_sgc / rhs_lcdm_boss * rh_spline(lit_reds),
        yerr=lit_rhs_err_sgc / rhs_lcdm_boss * rh_spline(lit_reds),
        **boss_kw,
    )
    ax_bot.errorbar(
        lit_reds - 0.01,
        lit_rhs_ngc / rhs_lcdm_boss,
        yerr=lit_rhs_err_ngc / rhs_lcdm_boss,
        **boss_kw,
    )
    ax_bot.errorbar(
        lit_reds + 0.01,
        lit_rhs_sgc / rhs_lcdm_boss,
        yerr=lit_rhs_err_sgc / rhs_lcdm_boss,
        **boss_kw,
    )

    # BOSS DR12 constraints (Laurent et al., 2016)
    lit_reds = np.array([2.4])
    lit_rhs = np.array([26.2])
    lit_rhs_err = np.array([0.9])
    rhs_lcdm = np.array([26.8])
    ax_top.errorbar(
        lit_reds, lit_rhs / rhs_lcdm * rh_spline(lit_reds), yerr=lit_rhs_err / rhs_lcdm * rh_spline(lit_reds), **boss_kw
    )
    ax_bot.errorbar(lit_reds, lit_rhs / rhs_lcdm, yerr=lit_rhs_err / rhs_lcdm, **boss_kw)

    # BOSS DR14 constraints (Goncalves et al., 2018)
    lit_reds = np.array([0.985, 1.350, 1.690, 2.075])
    lit_rhs = np.array([48.78, 40.56, 36.19, 27.91])
    lit_rhs_err = np.array([3.82, 3.39, 3.45, 3.91])
    rhs_lcdm = np.array([52.0, 45.0, 40.0, 34.0])
    ax_top.errorbar(
        lit_reds, lit_rhs / rhs_lcdm * rh_spline(lit_reds), yerr=lit_rhs_err / rhs_lcdm * rh_spline(lit_reds), **boss_kw
    )
    ax_bot.errorbar(lit_reds, lit_rhs / rhs_lcdm, yerr=lit_rhs_err / rhs_lcdm, **boss_kw)

    # BOSS DR16 constraints (Goncalves et al., 2021)
    lit_reds = np.array([2.30, 2.85])
    lit_rhs = np.array([29.66, 24.13])
    lit_rhs_err = np.array([2.22, 3.49])
    rhs_lcdm = np.array([31.59, 27.56])
    ax_top.errorbar(
        lit_reds, lit_rhs / rhs_lcdm * rh_spline(lit_reds), yerr=lit_rhs_err / rhs_lcdm * rh_spline(lit_reds), **boss_kw
    )
    ax_bot.errorbar(lit_reds, lit_rhs / rhs_lcdm, yerr=lit_rhs_err / rhs_lcdm, **boss_kw)

    # WiggleZ constraints (Scrimgeour et al., 2012)
    lit_reds = np.array([0.2, 0.4, 0.6, 0.8])
    lit_rhs = np.array([84, 74, 74, 64])
    lit_rhs_err = np.array([11, 6, 3, 2])
    rhs_lcdm = np.array([88, 82, 73, 66])
    ax_top.errorbar(
        lit_reds, lit_rhs / rhs_lcdm * rh_spline(lit_reds), yerr=lit_rhs_err / rhs_lcdm * rh_spline(lit_reds), **boss_kw
    )
    ax_bot.errorbar(lit_reds, lit_rhs / rhs_lcdm, yerr=lit_rhs_err / rhs_lcdm, **boss_kw)

    # ------------------------------------------------------------------ #
    # DESI data points
    # ------------------------------------------------------------------ #
    tracers, names, colors, templates, templates_real, rmins, rmaxs, model, rebin = setup_fitting()
    redshifts = np.array([0.295, 0.510, 0.706, 0.919, 1.317, 1.491])
    rhs_lcdm = np.zeros_like(redshifts)
    for i, z in enumerate(redshifts):
        template_real = RealSpaceTemplateWrapper(DirectPowerSpectrumTemplate(z=z, fiducial="DESI"))
        theory_real = LPTVelocileptorsTracerCorrelationDimension(template=template_real)
        theory_real()
        theory_real(b1p=theory_real.pt.sigma8, b2p=0.0, bsp=0.0, alpha0p=0.0, sn0p=0.0)
        rhs_lcdm[i] = theory_real.RHp
        print(i, z, rhs_lcdm[i])

    for skyarea in ["NGC", "SGC"]:
        for t, (tracer, name) in enumerate(zip(tracers, names)):
            print(tracer, name, rhs_lcdm[t])

            models, rhs, rescaled_Data, rescaled_models, rescaled_rhs, pars = np.load(
                f"{tracer}/CD_data_{skyarea}_weighted_{model}_rebin{rebin}_samples_pocomc_rescaled.npz",
                allow_pickle=True,
            ).values()
            rescaled_Data = rescaled_Data.item()

            c = ChainConsumer()
            c.add_chain(cChain(samples=pd.DataFrame({"RH": rescaled_rhs}), linewidth=2.0, name="rescaled", bins=20))
            constraints = c.analysis.get_summary()
            post_rescaled_RH = constraints["rescaled"]["RH"]
            print(post_rescaled_RH)

            RH_rescaled = post_rescaled_RH.center
            RH_rescaled_err_upper = (
                0 if post_rescaled_RH.upper is None else post_rescaled_RH.upper - post_rescaled_RH.center
            )
            RH_rescaled_err_lower = (
                0 if post_rescaled_RH.lower is None else post_rescaled_RH.center - post_rescaled_RH.lower
            )
            RH_rescaled_err_upper = np.sqrt(RH_rescaled_err_upper**2 + systematic[skyarea][t] ** 2)
            RH_rescaled_err_lower = np.sqrt(RH_rescaled_err_lower**2 + systematic[skyarea][t] ** 2)
            print(RH_rescaled, RH_rescaled_err_upper, RH_rescaled_err_upper)

            z_offset = redshifts[t] - 0.01 if skyarea == "NGC" else redshifts[t] + 0.01
            errbar_kw = dict(
                color=colors[tracer],
                mfc=colors[tracer] if skyarea == "NGC" else "w",
                mec="k" if skyarea == "NGC" else colors[tracer],
                marker="o",
                # label=name if skyarea == "NGC" else None,
                ls="None",
            )
            yerr = [[RH_rescaled_err_lower], [RH_rescaled_err_upper]]
            yerr_ratio = [[RH_rescaled_err_lower / rhs_lcdm[t]], [RH_rescaled_err_upper / rhs_lcdm[t]]]

            ax_top.errorbar(z_offset, RH_rescaled, yerr=yerr, **errbar_kw)
            ax_bot.errorbar(z_offset, RH_rescaled / rhs_lcdm[t], yerr=yerr_ratio, **errbar_kw)

    # ------------------------------------------------------------------ #
    # Axes formatting
    # ------------------------------------------------------------------ #
    ax_top.set_xlim(0.0, 1.8)
    ax_top.set_ylabel(r"$R_{H}^{\mathrm{matter}}(z)\,[h^{-1}\mathrm{Mpc}]$", fontsize=12, labelpad=12)
    ax_top.tick_params(labelsize=10)
    ax_top.legend(fontsize=12)

    ax_bot.set_xlim(0.0, 1.8)
    ax_bot.set_ylim(0.95, 1.05)
    ax_bot.set_xlabel(r"$z$", fontsize=12)
    ax_bot.set_ylabel(r"$R_{H}^{\mathrm{matter}}(z)/R_{H}^{\Lambda\mathrm{CDM}}(z)$", fontsize=12)
    ax_bot.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig("./plot_homogeneity_vs_redshift.png", dpi=300)
