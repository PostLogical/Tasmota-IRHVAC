"""Bacher-Madsen 2011 forward-selection driver: 1R1C → 2R2C with LR test.

Per Madsen-Bacher 2015 §1.4 + Bacher-Madsen 2011 §1.2:
  1. Fit simplest model (1R1C) and run residual-diagnostic battery.
  2. Fit next-complexity model (2R2C TiTe) and run battery.
  3. Likelihood-ratio test at α=0.05: reject reduced if 2·Δll > χ²_{Δp,α}.
  4. Practical identifiability gate: CV per-param ≤ 10% across restarts AND
     no parameters railed at bounds (Reynders-2014 diagnostic).
  5. Terminal model = highest accepted by both LR test AND identifiability.

Classification per Leprince 2022 (38% good / 38% close / 24% poor):
  good  — LR accepts AND identifiability passes AND residual battery passes
  close — LR accepts AND identifiability OR residual battery has caveats
  poor  — identifiability fails OR residuals fail whiteness with major rails
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from scipy.stats import chi2

from tests.hvac_bench.empirical.cumulated_periodogram import (
    CumulatedPeriodogramResult,
    cumulated_periodogram,
)
from tests.hvac_bench.empirical.pem_fit import (
    DEFAULT_BOUNDS_1R1C,
    DEFAULT_BOUNDS_2R2C,
    Bounds,
    FitResult,
    fit_1r1c,
    fit_2r2c,
)
from tests.hvac_bench.empirical.rc_model import (
    build_1r1c,
    build_2r2c,
    kalman_innovations,
)


N_PARAMS_1R1C: Final = 5
N_PARAMS_2R2C: Final = 8

# Per Bacher-Madsen 2011 LR-test threshold + Leprince 2022 nCPBES guidance
DEFAULT_LR_ALPHA: Final = 0.05
DEFAULT_CV_THRESHOLD: Final = 0.10  # Cárdenas-Rangel 2022
DEFAULT_LJUNG_BOX_ALPHA: Final = 0.05
DEFAULT_NCPBES_GOOD: Final = 0.005  # Leprince 2022 "good fit" envelope
DEFAULT_NCPBES_CLOSE: Final = 0.020  # above this is "poor"


# ── LR test ──────────────────────────────────────────────────────────────


@dataclass
class LRTestResult:
    """Likelihood-ratio test result (Bacher-Madsen 2011 forward selection)."""

    df: int
    statistic: float
    p_value: float
    alpha: float
    accept_full: bool


def lr_test(
    ll_reduced: float,
    ll_full: float,
    df: int,
    *,
    alpha: float = DEFAULT_LR_ALPHA,
) -> LRTestResult:
    """Standard LR test: 2·(ll_full - ll_reduced) ~ χ²_df under H0.

    Returns accept_full=True if p_value < alpha (reject reduced model).
    """
    if df <= 0:
        raise ValueError(f"df must be positive; got {df}")
    statistic = 2.0 * (ll_full - ll_reduced)
    if statistic < 0:
        # Reduced model has higher log-likelihood than full — keep reduced.
        statistic = 0.0
    p_value = float(chi2.sf(statistic, df=df))
    return LRTestResult(
        df=df,
        statistic=float(statistic),
        p_value=p_value,
        alpha=alpha,
        accept_full=p_value < alpha,
    )


# ── Residual battery ─────────────────────────────────────────────────────


@dataclass
class ResidualBatteryResult:
    """Bacher-Madsen residual battery applied to one-step Kalman innovations.

    Sub-tests:
      - Ljung-Box: temporal whiteness (autocorrelation up to lag K)
      - Cumulated periodogram: frequency-domain whiteness (Annex 58 ST3b)
      - Normality: Shapiro-Wilk on standardized residuals
    """

    ljung_box_p: float
    ljung_box_pass: bool
    normality_p: float
    normality_pass: bool
    cp: CumulatedPeriodogramResult
    cp_pass: bool
    n_residuals: int
    overall_pass: bool


def standardize_innovations(
    innovations: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Standardize one-step innovations to unit-variance under the model.

    Returns e_k / sqrt(S_k) for valid (non-NaN, finite, S_k > 0) rows.
    """
    mask = (
        np.isfinite(innovations)
        & np.isfinite(variances)
        & (variances > 0)
    )
    e = innovations[mask]
    s = np.sqrt(variances[mask])
    return e / s


def residual_battery(
    innovations: np.ndarray,
    variances: np.ndarray,
    *,
    n_lags: int = 20,
    ljung_box_alpha: float = DEFAULT_LJUNG_BOX_ALPHA,
    ks_alpha: float = 0.05,
    ncpbes_close_threshold: float = DEFAULT_NCPBES_CLOSE,
) -> ResidualBatteryResult:
    """Run Ljung-Box + CP + Shapiro-Wilk on standardized Kalman innovations.

    Imports residual_diagnostics lazily to avoid pulling its module-level
    dependencies into rc_model_fit code paths that don't need them.
    """
    from tests.hvac_bench import residual_diagnostics as rd

    standardized = standardize_innovations(innovations, variances)
    n = standardized.size
    if n < max(n_lags + 5, 50):
        raise ValueError(f"Too few residuals for diagnostic battery: {n}")

    lb_q, lb_p = rd.ljung_box_test(standardized, n_lags=n_lags)
    nor_test, nor_stat, nor_p = rd.normality_test(standardized)
    cp_result = cumulated_periodogram(standardized, ks_alpha=ks_alpha)

    ljung_pass = lb_p > ljung_box_alpha
    normality_pass = nor_p > ljung_box_alpha
    # CP is "passing" if nCPBES is below the close threshold (loose)
    cp_pass = cp_result.nCPBES < ncpbes_close_threshold
    overall = ljung_pass and cp_pass and normality_pass

    return ResidualBatteryResult(
        ljung_box_p=float(lb_p),
        ljung_box_pass=bool(ljung_pass),
        normality_p=float(nor_p),
        normality_pass=bool(normality_pass),
        cp=cp_result,
        cp_pass=cp_pass,
        n_residuals=n,
        overall_pass=overall,
    )


# ── Identifiability gate ─────────────────────────────────────────────────


@dataclass
class IdentifiabilityGate:
    """Aggregate identifiability verdict from CV + at-bound flags."""

    cv_pass_per_param: dict[str, bool]
    at_bound_per_param: dict[str, bool]
    n_failed_cv: int
    n_at_bound: int
    overall_pass: bool


def identifiability_gate(
    fit: FitResult,
    *,
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
) -> IdentifiabilityGate:
    """Per Cárdenas-Rangel 2022 + Reynders 2014 / lit-pass:
    fail if any parameter's CV across restarts > cv_threshold OR
    rails at a bound.
    """
    cv_per = fit.is_practically_identifiable(cv_threshold=cv_threshold)
    if not cv_per:
        # Fewer than 2 successful restarts; can't compute CV.
        cv_per = {k: False for k in fit.at_bound_per_param}

    n_failed_cv = sum(1 for v in cv_per.values() if not v)
    n_at_bound = sum(1 for v in fit.at_bound_per_param.values() if v)
    overall_pass = (n_failed_cv == 0) and (n_at_bound == 0)

    return IdentifiabilityGate(
        cv_pass_per_param=cv_per,
        at_bound_per_param=dict(fit.at_bound_per_param),
        n_failed_cv=n_failed_cv,
        n_at_bound=n_at_bound,
        overall_pass=overall_pass,
    )


# ── Forward-selection result ─────────────────────────────────────────────


Classification = Literal["good", "close", "poor"]


@dataclass
class ForwardSelectionResult:
    """Aggregated forward-selection outcome per Bacher-Madsen 2011.

    `selected_model` is the one accepted as terminal by the methodology:
    the most-complex model that passes BOTH the LR test AND the
    identifiability gate. If 2R2C fails identifiability despite winning
    the LR test, 1R1C is selected (and the identifiability failure is
    surfaced in 2r2c_identifiability).

    `classification` is per Leprince 2022:
      good  — selected model passes residual battery + identifiability
      close — passes some sub-checks; documented caveats in summary
      poor  — fails identifiability or residual battery on selected model
    """

    selected_model: Literal["1R1C", "2R2C"]
    classification: Classification
    summary: str
    fit_1r1c: FitResult
    fit_2r2c: FitResult | None
    lr_test: LRTestResult | None
    residuals_1r1c: ResidualBatteryResult | None
    residuals_2r2c: ResidualBatteryResult | None
    identifiability_1r1c: IdentifiabilityGate
    identifiability_2r2c: IdentifiabilityGate | None
    rejection_notes: list[str] = field(default_factory=list)


# ── Forward selection driver ─────────────────────────────────────────────


def _classify(
    selected: str,
    selected_battery: ResidualBatteryResult | None,
    selected_id: IdentifiabilityGate,
) -> Classification:
    """Map (model, residual battery, identifiability) to good/close/poor."""
    if selected_battery is None:
        # Battery couldn't run (too few residuals, etc.) — treat as poor.
        return "poor"
    if selected_id.overall_pass and selected_battery.overall_pass:
        # Tighten with nCPBES "good" threshold — Leprince 2022 envelope.
        if selected_battery.cp.nCPBES < DEFAULT_NCPBES_GOOD:
            return "good"
        return "close"
    if selected_id.n_at_bound >= 2 or not selected_battery.ljung_box_pass:
        return "poor"
    return "close"


def forward_select(
    observations: np.ndarray,
    inputs: np.ndarray,
    valid: np.ndarray,
    dt: float,
    *,
    n_restarts: int = 4,
    seed: int = 0,
    bounds_1r1c: Bounds = DEFAULT_BOUNDS_1R1C,
    bounds_2r2c: Bounds = DEFAULT_BOUNDS_2R2C,
    lr_alpha: float = DEFAULT_LR_ALPHA,
    cv_threshold: float = DEFAULT_CV_THRESHOLD,
    fit_2r2c_unconditionally: bool = True,
) -> ForwardSelectionResult:
    """Run Bacher-Madsen 2011 forward selection.

    Args:
        observations, inputs, valid, dt: see fit_1r1c/fit_2r2c.
        n_restarts: per-model multi-restart count.
        seed: master RNG seed; 1R1C uses ``seed``, 2R2C uses ``seed + 1000``.
        bounds_1r1c, bounds_2r2c: per-model bounds.
        lr_alpha: LR test α.
        cv_threshold: identifiability CV threshold.
        fit_2r2c_unconditionally: if True, fit 2R2C even if 1R1C residuals
            already pass — for the bench, we always want both fits for
            comparative reporting.

    Returns ForwardSelectionResult with both fits (when 2R2C ran) and the
    methodology's selection verdict.
    """
    rejection_notes: list[str] = []

    # Step 1: Fit 1R1C
    fit1 = fit_1r1c(
        observations,
        inputs,
        valid,
        dt,
        n_restarts=n_restarts,
        seed=seed,
        bounds=bounds_1r1c,
    )

    id_1r1c = identifiability_gate(fit1, cv_threshold=cv_threshold)

    # Step 2: Residual battery on 1R1C
    battery_1r1c: ResidualBatteryResult | None = None
    try:
        ss1 = build_1r1c(fit1.best.params.to_full(), dt=dt)
        innov1, var1 = kalman_innovations(ss1, observations, inputs, valid)
        battery_1r1c = residual_battery(innov1, var1)
    except (ValueError, np.linalg.LinAlgError) as e:
        rejection_notes.append(f"1R1C residual battery failed: {e}")

    # Step 3: Decide whether to fit 2R2C
    if not fit_2r2c_unconditionally and battery_1r1c is not None and battery_1r1c.overall_pass:
        # 1R1C is sufficient — skip 2R2C fit
        return ForwardSelectionResult(
            selected_model="1R1C",
            classification=_classify("1R1C", battery_1r1c, id_1r1c),
            summary="1R1C accepted; 2R2C skipped (residual battery passed for 1R1C).",
            fit_1r1c=fit1,
            fit_2r2c=None,
            lr_test=None,
            residuals_1r1c=battery_1r1c,
            residuals_2r2c=None,
            identifiability_1r1c=id_1r1c,
            identifiability_2r2c=None,
            rejection_notes=rejection_notes,
        )

    # Step 4: Fit 2R2C
    fit2 = fit_2r2c(
        observations,
        inputs,
        valid,
        dt,
        n_restarts=n_restarts,
        seed=seed + 1000,
        bounds=bounds_2r2c,
    )

    id_2r2c = identifiability_gate(fit2, cv_threshold=cv_threshold)

    battery_2r2c: ResidualBatteryResult | None = None
    try:
        ss2 = build_2r2c(fit2.best.params.to_full(), dt=dt)
        innov2, var2 = kalman_innovations(ss2, observations, inputs, valid)
        battery_2r2c = residual_battery(innov2, var2)
    except (ValueError, np.linalg.LinAlgError) as e:
        rejection_notes.append(f"2R2C residual battery failed: {e}")

    # Step 5: LR test
    df = N_PARAMS_2R2C - N_PARAMS_1R1C
    lr = lr_test(
        ll_reduced=fit1.best.log_likelihood,
        ll_full=fit2.best.log_likelihood,
        df=df,
        alpha=lr_alpha,
    )

    # Step 6: Selection per Bacher-Madsen 2011 + lit-pass identifiability gate
    if lr.accept_full and id_2r2c.overall_pass:
        selected = "2R2C"
        summary = (
            f"2R2C accepted: LR p={lr.p_value:.3g} < {lr_alpha}, "
            f"identifiability passes."
        )
    elif lr.accept_full and not id_2r2c.overall_pass:
        selected = "1R1C"
        rails = [k for k, v in id_2r2c.at_bound_per_param.items() if v]
        cv_fails = [k for k, v in id_2r2c.cv_pass_per_param.items() if not v]
        summary = (
            f"2R2C rejected DESPITE LR p={lr.p_value:.3g} < {lr_alpha}: "
            f"identifiability fails ({id_2r2c.n_at_bound} at-bound rails={rails}, "
            f"{id_2r2c.n_failed_cv} CV-failures={cv_fails}). "
            f"Per Madsen-Bacher 2015 §1.4 + Reynders 2014, rails diagnose "
            f"non-identifiability under wide priors."
        )
        rejection_notes.append(summary)
    else:
        selected = "1R1C"
        summary = (
            f"2R2C rejected: LR p={lr.p_value:.3g} ≥ {lr_alpha} "
            f"(insufficient improvement over 1R1C)."
        )

    selected_battery = battery_1r1c if selected == "1R1C" else battery_2r2c
    selected_id = id_1r1c if selected == "1R1C" else id_2r2c
    classification = _classify(selected, selected_battery, selected_id)

    return ForwardSelectionResult(
        selected_model=selected,
        classification=classification,
        summary=summary,
        fit_1r1c=fit1,
        fit_2r2c=fit2,
        lr_test=lr,
        residuals_1r1c=battery_1r1c,
        residuals_2r2c=battery_2r2c,
        identifiability_1r1c=id_1r1c,
        identifiability_2r2c=id_2r2c,
        rejection_notes=rejection_notes,
    )
