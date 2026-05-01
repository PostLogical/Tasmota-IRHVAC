"""Identifiability-aware assertion harness for bench scenario tests (Phase 1e).

Replaces the ad-hoc ``assert abs(beta - truth) < 0.05`` pattern with
helpers that distinguish *identifiability failures* (the experiment
design can't recover this parameter at the precision required) from
*recovery failures* (the data could in principle support it, but the
estimator missed). The two have different remediations — improve the
probe versus fix the estimator — and the existing flat tolerance test
conflates them.

These assertions are tools for tests, not production code. They take
``IdentifiabilityReport`` and ``ResidualDiagnosticReport`` instances
from Phases 1b and 1d and check their key properties.

Failure messages name the diagnostic that triggered, the threshold
that was violated, and the relevant context (n_obs, σ², CRLB) so the
test author can decide whether to widen the tolerance or fix the probe.
"""

from __future__ import annotations

import math

from tests.hvac_bench.identifiability import IdentifiabilityReport
from tests.hvac_bench.residual_diagnostics import ResidualDiagnosticReport


# ── Identifiability assertions ────────────────────────────────────────


def assert_identifiable(
    report: IdentifiabilityReport,
    feature: str,
    *,
    crlb_max: float | None = None,
    se_max: float | None = None,
) -> None:
    """Assert the experiment design admits identification of ``feature``.

    Either ``crlb_max`` or ``se_max`` (= sqrt(crlb_max)) must be supplied.
    Provide whichever is more natural for the test's claim — variance
    bounds are common in the literature, SE bounds map directly to
    "the experiment can resolve β to within ±X°C".

    Raises
    ------
    AssertionError
        If ``feature`` is not in the regressor, if it is rank-deficient
        (CRLB == +inf), or if the CRLB exceeds the supplied bound.
    """
    if crlb_max is None and se_max is None:
        raise ValueError("supply at least one of crlb_max or se_max")
    if feature not in report.feature_names:
        raise AssertionError(
            f"feature {feature!r} not in regressor "
            f"(have: {report.feature_names})"
        )
    idx = report.feature_names.index(feature)
    crlb = report.crlb[idx]
    if not math.isfinite(crlb):
        raise AssertionError(
            f"feature {feature!r} is unidentifiable (CRLB = +inf); "
            f"rank={report.rank}/{report.n_features}, "
            f"PE order={report.pe_order}, "
            f"κ={report.condition_number:.2f}"
        )
    if crlb_max is not None and crlb > crlb_max:
        raise AssertionError(
            f"CRLB[{feature!r}] = {crlb:.6g} > {crlb_max:.6g}; "
            f"experiment design insufficient for the requested precision. "
            f"n_obs={report.n_observations}, σ²={report.sigma2:.4g}"
        )
    if se_max is not None:
        se = math.sqrt(crlb)
        if se > se_max:
            raise AssertionError(
                f"SE_min[{feature!r}] = {se:.6g} > {se_max:.6g}; "
                f"experiment design insufficient. "
                f"n_obs={report.n_observations}, CRLB={crlb:.6g}"
            )


def assert_pe_order_at_least(
    report: IdentifiabilityReport, n: int
) -> None:
    """Assert PE order ≥ n (Ljung Ch 13.4 requirement for n parameters).

    Below the parameter count, some directions of β-space are
    structurally unidentifiable.
    """
    if report.pe_order < n:
        raise AssertionError(
            f"PE order {report.pe_order} < {n} required; "
            f"rank-deficient regressor on {report.n_features} features. "
            f"σ_min={report.smallest_singular_value:.4g}"
        )


def assert_condition_number_below(
    report: IdentifiabilityReport, max_kappa: float
) -> None:
    """Belsley κ check on the standardized regressor.

    Belsley (1980) thresholds: κ > 30 moderate, κ > 100 strong
    collinearity. The default production WLS κ-gate uses 30.
    """
    if report.condition_number > max_kappa:
        raise AssertionError(
            f"condition number {report.condition_number:.2f} > {max_kappa} "
            f"(Belsley 1980 collinearity threshold). "
            f"σ_min={report.smallest_singular_value:.4g}, "
            f"σ_max={report.largest_singular_value:.4g}"
        )


# ── Residual-diagnostic assertions ────────────────────────────────────


def assert_residuals_white(
    report: ResidualDiagnosticReport, *, alpha: float = 0.05
) -> None:
    """Assert the Ljung-Box test fails to reject whiteness.

    Failure means residuals are autocorrelated → the linear model is
    misspecified for the data-generating process. Per Annex 58 ST3
    part 2, this is the trigger to escalate model order, not a
    correctness failure of the regression itself.
    """
    if report.ljung_box_p_value < alpha:
        raise AssertionError(
            f"Ljung-Box rejects whiteness: p={report.ljung_box_p_value:.4g} "
            f"< α={alpha}. Q={report.ljung_box_statistic:.2f}, "
            f"lag-1 ACF={report.autocorrelation[1] if len(report.autocorrelation) > 1 else 0:.3f}, "
            f"n={report.n_residuals}. Model is misspecified for these data."
        )


def assert_residuals_normal(
    report: ResidualDiagnosticReport, *, alpha: float = 0.05
) -> None:
    """Assert the normality test fails to reject Gaussianity.

    Failure means residuals are non-Gaussian → the FIM/CRLB derivation
    (which assumes Gaussian noise) is approximate. The β estimates are
    still unbiased under OLS/WLS, but the SE → confidence-interval
    mapping no longer covers truth at nominal rates.
    """
    if report.normality_p_value < alpha:
        raise AssertionError(
            f"{report.normality_test_name} rejects normality: "
            f"p={report.normality_p_value:.4g} < α={alpha}. "
            f"statistic={report.normality_statistic:.4f}, "
            f"skew={report.skewness:.3f}, kurtosis={report.kurtosis:.3f}, "
            f"n={report.n_residuals}"
        )


def assert_split_half_stable(
    report: ResidualDiagnosticReport, *, max_rel_change: float = 0.10
) -> None:
    """Assert the split-half max relative β change is below threshold.

    Large changes flag non-stationarity or over-fitting. ``max_rel_change``
    of 10% is the textbook starting threshold (Bacher-Madsen 2011 §4.2);
    bench scenarios with deliberate transients (e.g. window-replacement
    physical change scenarios) should pass a wider threshold.
    """
    if report.split_half_max_rel_change is None:
        raise AssertionError(
            f"split-half stability not computed (insufficient observations). "
            f"n_residuals={report.n_residuals}"
        )
    if report.split_half_max_rel_change > max_rel_change:
        raise AssertionError(
            f"split-half max relative β change "
            f"{report.split_half_max_rel_change:.1%} > "
            f"{max_rel_change:.1%}; model unstable across data subsets. "
            f"first half β: {report.split_half_beta_first}, "
            f"second half β: {report.split_half_beta_second}"
        )


# ── Recovery assertions (β vs truth, CRLB-aware) ──────────────────────


def assert_recovers_within_se(
    *,
    estimated: float,
    truth: float,
    se: float,
    k: float = 3.0,
    name: str = "β",
) -> None:
    """Assert estimator recovers truth to within ``k * se``.

    The supplied SE can be the production WLS standard error, the CRLB
    lower bound, or any other estimator-SE estimate. ``k=3`` corresponds
    to a 99.7% Gaussian band; ``k=2`` is 95%.
    """
    if se <= 0:
        raise ValueError("se must be > 0")
    band = k * se
    gap = abs(estimated - truth)
    if gap > band:
        raise AssertionError(
            f"{name}={estimated:.4f} not within {k}σ of truth {truth:.4f}: "
            f"|gap|={gap:.4f}, k·SE={band:.4f} (SE={se:.4f})"
        )


def assert_recovers_within_crlb(
    *,
    estimated: float,
    truth: float,
    report: IdentifiabilityReport,
    feature: str,
    k: float = 3.0,
) -> None:
    """Assert β recovery is within ``k * sqrt(CRLB)`` of truth.

    Distinguishes identifiability failures (CRLB too wide for *any*
    estimator to resolve the gap) from recovery failures (CRLB tight,
    but the estimator missed). The failure message names which.
    """
    if feature not in report.feature_names:
        raise AssertionError(
            f"feature {feature!r} not in regressor "
            f"(have: {report.feature_names})"
        )
    idx = report.feature_names.index(feature)
    crlb = report.crlb[idx]
    if not math.isfinite(crlb):
        raise AssertionError(
            f"can't assert recovery: feature {feature!r} unidentifiable "
            f"(CRLB = +inf). rank={report.rank}/{report.n_features}"
        )
    se = math.sqrt(crlb)
    band = k * se
    gap = abs(estimated - truth)
    if gap <= band:
        return
    # Failed. Distinguish "identifiability too loose" from "estimator missed".
    # If the truth lies outside the lower-bound band by a large factor
    # (gap >> band), the estimator missed even though the data could
    # have supported it. If gap is comparable to band, identifiability
    # is the binding constraint.
    ratio = gap / band
    if ratio > 5.0:
        kind = "estimator failed to recover"
    else:
        kind = "experiment design (CRLB) is tight against truth"
    raise AssertionError(
        f"{kind}: β[{feature!r}] = {estimated:.4f} not within {k}·SE of "
        f"truth {truth:.4f}: |gap|={gap:.4f}, k·SE={band:.4f} "
        f"(SE={se:.4f}, CRLB={crlb:.6g}, n_obs={report.n_observations}, "
        f"σ²={report.sigma2:.4g})"
    )
