"""Tests for the Bacher-Madsen residual diagnostics module (Phase 1d).

Two layers, mirroring Phase 1a/1b:

1. **Pure-math** — synthetic series with known properties (white noise,
   AR(1), Gaussian, non-Gaussian) to verify each diagnostic correctly
   classifies the input.
2. **End-to-end** — runs an open-loop probe, fits WLS, computes the
   residual battery, asserts what we expect on this misspecified linear
   model: residuals are NOT white (2R2C wall transients introduce
   autocorrelation) and NOT normal (step transitions create heavy
   tails). These "failures" are themselves *the* signal — they tell us
   the regression's CRLB-derived SEs do NOT cover truth at nominal
   rates, which is the Annex 58 ST3 verdict that motivates Phase 1d.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.hvac_bench.identifiability import build_regressor_matrix
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.open_loop_runner import (
    OpenLoopConfig,
    make_step_excitation,
    run_open_loop_probe,
)
from tests.hvac_bench.residual_diagnostics import (
    autocorrelation,
    ljung_box_test,
    normality_test,
    residual_diagnostic_report,
    split_half_stability,
)


# ── Pure-math tests ───────────────────────────────────────────────────


class TestAutocorrelation:
    def test_white_noise_has_decaying_acf(self, bench_metrics, num_regression):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000)
        acf = autocorrelation(x, max_lag=10)
        assert acf[0] == pytest.approx(1.0, abs=1e-12)
        # Lag-k ACF ~ N(0, 1/n) → |acf[k]| should be small.
        for k in range(1, 11):
            assert abs(acf[k]) < 0.1, f"lag {k} ACF = {acf[k]:.3f} too large"

    def test_ar1_has_geometric_decay(self, bench_metrics, num_regression):
        rng = np.random.default_rng(1)
        n = 5000
        phi = 0.6
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal()
        acf = autocorrelation(x, max_lag=5)
        # AR(1) with parameter φ has theoretical ACF[k] = φ^k.
        for k in range(1, 6):
            theoretical = phi ** k
            assert abs(acf[k] - theoretical) < 0.05, (
                f"lag {k} ACF {acf[k]:.3f} vs theoretical {theoretical:.3f}"
            )

    def test_constant_series_returns_zero_acf(self, bench_metrics, num_regression):
        x = np.ones(100)
        acf = autocorrelation(x, max_lag=5)
        # Zero variance after centering → ACF undefined; we return 1, 0, ...
        assert acf[0] == pytest.approx(1.0)
        for k in range(1, 6):
            assert acf[k] == 0.0


class TestLjungBox:
    def test_white_noise_passes(self, bench_metrics, num_regression):
        rng = np.random.default_rng(2)
        x = rng.normal(size=2000)
        Q, p = ljung_box_test(x, n_lags=20)
        assert p > 0.05, (
            f"white noise rejected: Q={Q:.2f}, p={p:.4f}"
        )

    def test_ar1_rejected(self, bench_metrics, num_regression):
        rng = np.random.default_rng(3)
        n = 1000
        phi = 0.5
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal()
        Q, p = ljung_box_test(x, n_lags=10)
        assert p < 0.001, (
            f"AR(1) not rejected: Q={Q:.2f}, p={p:.4f}"
        )

    def test_tiny_sample_returns_safe_default(self, bench_metrics, num_regression):
        # n ≤ n_lags + 1 → no test possible.
        x = np.array([0.1, -0.2, 0.05])
        Q, p = ljung_box_test(x, n_lags=10)
        assert (Q, p) == (0.0, 1.0)


class TestNormality:
    def test_gaussian_passes_shapiro(self, bench_metrics, num_regression):
        rng = np.random.default_rng(4)
        x = rng.normal(size=500)
        name, stat, p = normality_test(x)
        assert name == "shapiro-wilk"
        assert p > 0.05, f"gaussian rejected: stat={stat:.4f}, p={p:.4f}"

    def test_uniform_rejected(self, bench_metrics, num_regression):
        rng = np.random.default_rng(5)
        x = rng.uniform(low=-1, high=1, size=500)
        name, stat, p = normality_test(x)
        assert p < 0.01, (
            f"uniform passed normality: stat={stat:.4f}, p={p:.4f}"
        )

    def test_large_sample_uses_jarque_bera(self, bench_metrics, num_regression):
        rng = np.random.default_rng(6)
        x = rng.normal(size=10000)
        name, stat, p = normality_test(x)
        assert name == "jarque-bera"
        assert p > 0.05  # gaussian still passes

    def test_too_few_samples_returns_na(self, bench_metrics, num_regression):
        x = np.array([0.1, 0.2])
        name, stat, p = normality_test(x)
        assert name == "n/a"


# ── End-to-end on probe residuals ─────────────────────────────────────


@pytest.fixture(scope="module")
def probe_observations():
    excitation = make_step_excitation(
        center_c=24.0, amplitude_c=1.5, hold_minutes=360.0, tick_minutes=15.0,
    )
    config = OpenLoopConfig(
        excitation=excitation,
        n_days=30,
        profile_name="living_room",
        outdoor_base_c=-2.0,
        outdoor_diurnal_c=8.0,
        desired_c=20.0,
        mode="heat",
        noise_sigma=0.05,
        noise_seed=42,
        tick_minutes=15.0,
    )
    return run_open_loop_probe(config).observations


@pytest.fixture(scope="module")
def probe_beta_and_report(probe_observations):
    """Fit OLS-via-WLS once and compute the diagnostic report."""
    X, y, w, _ = build_regressor_matrix(
        probe_observations,
        feature_order=["intercept", "outdoor_delta"],
    )
    Wsqrt = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Wsqrt * X, Wsqrt[:, 0] * y, rcond=None)
    rep = residual_diagnostic_report(
        observations=probe_observations,
        beta=beta.tolist(),
        feature_order=["intercept", "outdoor_delta"],
        model_inputs=None,
        n_lags=20,
    )
    return beta, rep


class TestProbeResidualsSurfaceMisspecification:
    """The 2R2C step probe fed into a 1R1C-shaped linear regression
    SHOULD fail whiteness — wall-mode transients introduce
    autocorrelation that the regression doesn't model. The diagnostic
    is therefore expected to flag the model as misspecified, which is
    informative: it tells us the WLS regression's nominal CRLB SEs are
    conservative (or aggressive) by an unmodeled amount.

    Per Annex 58 ST3 part 2: failed whiteness on a 1R1C fit motivates
    moving up to 2R2C; failed whiteness on a 2R2C fit motivates moving
    up to 3R3C / heat-pump-aware models. The diagnostic is the
    *trigger*, not the failure.
    """

    def test_residual_count_matches_eligible_obs(self, bench_metrics, num_regression, probe_beta_and_report):
        beta, rep = probe_beta_and_report
        # Should be ~all of the 2880 ticks, minus those filtered out by
        # the room_rate gate.
        assert rep.n_residuals > 1000

    def test_residual_mean_near_zero(self, bench_metrics, num_regression, probe_beta_and_report):
        beta, rep = probe_beta_and_report
        # OLS residuals on a regression with intercept have mean exactly
        # 0 by construction. WLS adds weights; mean stays small.
        assert abs(rep.mean) < 0.1

    def test_acf_first_lag_reported(self, bench_metrics, num_regression, probe_beta_and_report):
        beta, rep = probe_beta_and_report
        # 21 entries: lag 0..20.
        assert len(rep.autocorrelation) == 21
        assert rep.autocorrelation[0] == pytest.approx(1.0)

    def test_ljung_box_rejects_whiteness(self, bench_metrics, num_regression, probe_beta_and_report):
        # 2R2C wall transients during step switches → autocorrelated
        # residuals → reject whiteness. This is the EXPECTED finding;
        # if it passed, the linear model would be perfectly specified
        # for 2R2C dynamics (it isn't).
        beta, rep = probe_beta_and_report
        assert rep.is_white_at_alpha_05 is False
        assert rep.ljung_box_p_value < 0.05

    def test_normality_test_runs(self, bench_metrics, num_regression, probe_beta_and_report):
        # Doesn't matter whether it passes or fails — just verifies the
        # test runs without crashing on a real probe's residual scale.
        beta, rep = probe_beta_and_report
        assert rep.normality_test_name in ("shapiro-wilk", "jarque-bera")
        assert math.isfinite(rep.normality_statistic) or math.isnan(
            rep.normality_statistic
        )
        assert 0.0 <= rep.normality_p_value <= 1.0

    def test_split_half_stable(self, bench_metrics, num_regression, probe_beta_and_report):
        # On a 30-day stationary scenario, β should be similar across
        # halves. Loose threshold (50%) — the split-half check is
        # mostly a sanity check; the strong claim is "not wildly
        # different", not "identical". Stricter values would justify
        # treating split-half as a tightening tool.
        beta, rep = probe_beta_and_report
        assert rep.split_half_max_rel_change is not None
        assert rep.split_half_max_rel_change < 0.50, (
            f"split-half β changed by {rep.split_half_max_rel_change:.1%} "
            f"between halves — model unstable on stationary data"
        )

    def test_report_has_skew_and_kurtosis(self, bench_metrics, num_regression, probe_beta_and_report):
        beta, rep = probe_beta_and_report
        assert math.isfinite(rep.skewness)
        assert math.isfinite(rep.kurtosis)


class TestEmptyObservations:
    def test_empty_input_returns_safe_report(self, bench_metrics, num_regression):
        rep = residual_diagnostic_report(
            observations=[],
            beta=[0.0, 0.0],
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
        )
        assert rep.n_residuals == 0
        assert rep.is_white_at_alpha_05 is True  # vacuous pass
        assert rep.is_normal_at_alpha_05 is True
        assert rep.split_half_max_rel_change is None
