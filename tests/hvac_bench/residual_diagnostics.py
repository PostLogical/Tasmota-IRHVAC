"""Bacher-Madsen residual diagnostics for fitted regressions (Phase 1d).

Post-hoc tests on the residuals of a fitted linear model. These are the
Annex 58 ST3 / Bacher-Madsen 2011 standard diagnostics for grey-box
thermal model identification: whiteness (Ljung-Box), normality (Shapiro-
Wilk or Jarque-Bera), and parameter stability across data subsets
(split-half regression).

Wired into bench scenarios so any reported β estimate carries a
diagnostic verdict. When residuals fail whiteness, the linear model is
mis-specified — autocorrelated residuals violate the i.i.d. Gaussian
assumption underlying both OLS/WLS and the FIM-based CRLB. β estimates
are still computable but the reported standard errors no longer cover
truth at their nominal rates; this is the canonical signature of model
mis-specification (Ljung *System Identification* §16.4).

References:
- Bacher & Madsen 2011, "Identifying suitable models for the heat
  dynamics of buildings", *Energy and Buildings* 43(7) 1511-1522, §4.2
- IEA EBC Annex 58 ST3 part 2 (Madsen & Bacher), statistical guidelines
  for grey-box identification
- Ljung-Box, "On a measure of lack of fit in time series models",
  *Biometrika* 65(2) 297-303 (1978)
- Shapiro & Wilk, "An analysis of variance test for normality",
  *Biometrika* 52 (1965)
- Jarque & Bera, "Efficient tests for normality, homoscedasticity and
  serial independence of regression residuals", *Economics Letters* 6
  (1980)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from custom_components.tasmota_irhvac.pi.batch_learning import (
    Observation,
    weighted_least_squares,
)
from tests.hvac_bench.identifiability import build_regressor_matrix


# ── Result type ────────────────────────────────────────────────────────


@dataclass
class ResidualDiagnosticReport:
    """Bacher-Madsen residual battery for a fitted regression.

    Attributes
    ----------
    n_residuals : int
        Number of residuals fed into the diagnostics (after the WLS
        eligibility filter).
    mean, std, skewness, kurtosis : float
        Sample moments of the residuals. Skewness should be ~0 and
        excess kurtosis ~0 for Gaussian residuals.
    autocorrelation : list[float]
        Sample ACF values at lags 0..n_lags (lag 0 is always 1.0).
    ljung_box_statistic, ljung_box_p_value : float
        Q statistic and chi-square p-value for the joint test that
        autocorrelations at lags 1..n_lags are all zero. p < 0.05
        rejects "white noise".
    is_white_at_alpha_05 : bool
        ``True`` iff Ljung-Box p ≥ 0.05 (cannot reject white noise).
    normality_test_name : str
        ``"shapiro-wilk"`` for n ≤ 5000, ``"jarque-bera"`` otherwise.
    normality_statistic, normality_p_value : float
        Statistic and p-value for the normality test. p < 0.05 rejects
        Gaussianity.
    is_normal_at_alpha_05 : bool
        ``True`` iff normality p ≥ 0.05.
    split_half_beta_first, split_half_beta_second : list[float] | None
        β estimates from the first and second halves of the data. None
        if either half had insufficient observations.
    split_half_max_rel_change : float | None
        Largest ``|β_2 - β_1| / |β_1|`` across features. Should be
        small (<10%) for a stable model on stationary data.
    """

    n_residuals: int

    mean: float
    std: float
    skewness: float
    kurtosis: float

    autocorrelation: list[float]

    ljung_box_statistic: float
    ljung_box_p_value: float
    is_white_at_alpha_05: bool

    normality_test_name: str
    normality_statistic: float
    normality_p_value: float
    is_normal_at_alpha_05: bool

    split_half_beta_first: list[float] | None
    split_half_beta_second: list[float] | None
    split_half_max_rel_change: float | None


# ── Core statistics ────────────────────────────────────────────────────


def autocorrelation(residuals: np.ndarray, max_lag: int = 20) -> list[float]:
    """Sample autocorrelation function up to lag ``max_lag``.

    Lag 0 is always 1.0. Lag k is computed from mean-centered residuals
    as ``Σ_t (r_t - mean)(r_{t+k} - mean) / Σ_t (r_t - mean)²``.

    For ``len(residuals) <= max_lag`` returns a truncated list.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be ≥ 0")
    n = len(residuals)
    if n == 0:
        return []
    x = residuals - residuals.mean()
    denom = float(x @ x)
    if denom <= 0:
        return [1.0] + [0.0] * min(max_lag, n - 1)
    out = [1.0]
    for k in range(1, min(max_lag, n - 1) + 1):
        out.append(float(x[: n - k] @ x[k:]) / denom)
    return out


def ljung_box_test(
    residuals: np.ndarray, n_lags: int = 20
) -> tuple[float, float]:
    """Ljung-Box (1978) Q test for joint zero-autocorrelation.

    ``Q = n(n+2) Σ_{k=1}^{m} ρ_k² / (n-k)``. Under H0 (white noise),
    ``Q ~ χ²(m)``. Returns ``(Q, p_value)``.

    For tiny samples (``n ≤ n_lags``) returns ``(0.0, 1.0)`` — not
    enough data to test.
    """
    n = len(residuals)
    if n <= n_lags + 1:
        return 0.0, 1.0
    rho = autocorrelation(residuals, max_lag=n_lags)
    Q = n * (n + 2) * sum(
        rho[k] ** 2 / (n - k) for k in range(1, len(rho))
    )
    p = float(1.0 - stats.chi2.cdf(Q, df=n_lags))
    return float(Q), p


def normality_test(
    residuals: np.ndarray,
) -> tuple[str, float, float]:
    """Test residuals against the Gaussian null.

    Uses Shapiro-Wilk for ``n ≤ 5000`` (highest power for small
    samples) and Jarque-Bera otherwise (Shapiro-Wilk is undefined for
    n > 5000). Returns ``(name, statistic, p_value)``.
    """
    n = len(residuals)
    if n < 3:
        return "n/a", float("nan"), 1.0
    if n <= 5000:
        stat, p = stats.shapiro(residuals)
        return "shapiro-wilk", float(stat), float(p)
    stat, p = stats.jarque_bera(residuals)
    return "jarque-bera", float(stat), float(p)


def split_half_stability(
    observations: list[Observation],
    feature_order: list[str],
    model_inputs: list[dict] | None = None,
    min_observations: int = 20,
) -> tuple[list[float] | None, list[float] | None, float | None]:
    """Fit WLS on first and second halves, compare β.

    On stationary data with enough observations, β should be similar
    across halves. Large per-feature relative changes flag either
    non-stationarity (the data-generating process is changing) or
    over-fitting (the regression is sensitive to which subset is used).

    Returns ``(beta_first, beta_second, max_rel_change)`` or
    ``(None, None, None)`` if either half has too few observations.
    """
    n = len(observations)
    if n < 2 * min_observations:
        return None, None, None

    half = n // 2
    first = observations[:half]
    second = observations[half:]

    n_features = len(feature_order)
    r1 = weighted_least_squares(
        observations=first,
        n_features=n_features,
        feature_order=feature_order,
        model_inputs=model_inputs,
        min_observations=min_observations,
        detect_lag=False,
    )
    r2 = weighted_least_squares(
        observations=second,
        n_features=n_features,
        feature_order=feature_order,
        model_inputs=model_inputs,
        min_observations=min_observations,
        detect_lag=False,
    )
    if r1 is None or r2 is None:
        return None, None, None

    rel = []
    for i in range(n_features):
        b1, b2 = r1.beta_batch[i], r2.beta_batch[i]
        denom = max(abs(b1), 1e-12)
        rel.append(abs(b2 - b1) / denom)
    return list(r1.beta_batch), list(r2.beta_batch), float(max(rel))


# ── Top-level entry point ─────────────────────────────────────────────


def residual_diagnostic_report(
    observations: list[Observation],
    beta: list[float],
    feature_order: list[str],
    model_inputs: list[dict] | None = None,
    room_rate_threshold: float = 0.02,
    n_lags: int = 20,
    include_split_half: bool = True,
) -> ResidualDiagnosticReport:
    """Compute the Bacher-Madsen residual battery for a fitted regression.

    Builds the regressor matrix exactly as ``weighted_least_squares``
    would, computes residuals ``y - X β``, and runs the diagnostic suite.
    """
    X, y, _weights, _ = build_regressor_matrix(
        observations, feature_order, model_inputs, room_rate_threshold
    )
    n = X.shape[0]
    beta_arr = np.array(beta, dtype=float)
    if n == 0:
        return ResidualDiagnosticReport(
            n_residuals=0,
            mean=0.0, std=0.0, skewness=0.0, kurtosis=0.0,
            autocorrelation=[],
            ljung_box_statistic=0.0,
            ljung_box_p_value=1.0,
            is_white_at_alpha_05=True,
            normality_test_name="n/a",
            normality_statistic=float("nan"),
            normality_p_value=1.0,
            is_normal_at_alpha_05=True,
            split_half_beta_first=None,
            split_half_beta_second=None,
            split_half_max_rel_change=None,
        )
    residuals = y - X @ beta_arr

    moments_skew = float(stats.skew(residuals))
    moments_kurt = float(stats.kurtosis(residuals))  # Fisher (excess)

    acf = autocorrelation(residuals, max_lag=n_lags)
    Q, lb_p = ljung_box_test(residuals, n_lags=n_lags)
    norm_name, norm_stat, norm_p = normality_test(residuals)

    if include_split_half:
        b1, b2, max_rel = split_half_stability(
            observations, feature_order, model_inputs
        )
    else:
        b1 = b2 = None
        max_rel = None

    return ResidualDiagnosticReport(
        n_residuals=int(n),
        mean=float(residuals.mean()),
        std=float(residuals.std(ddof=1)) if n > 1 else 0.0,
        skewness=moments_skew,
        kurtosis=moments_kurt,
        autocorrelation=acf,
        ljung_box_statistic=Q,
        ljung_box_p_value=lb_p,
        is_white_at_alpha_05=lb_p >= 0.05,
        normality_test_name=norm_name,
        normality_statistic=norm_stat,
        normality_p_value=norm_p,
        is_normal_at_alpha_05=norm_p >= 0.05,
        split_half_beta_first=b1,
        split_half_beta_second=b2,
        split_half_max_rel_change=max_rel,
    )
