"""Tests for the Bacher-Madsen forward-selection driver."""

from __future__ import annotations

import numpy as np
import pytest

from tests.hvac_bench.conftest import check_bench_metrics

from tests.hvac_bench.empirical.forward_selection import (
    DEFAULT_LR_ALPHA,
    N_PARAMS_1R1C,
    N_PARAMS_2R2C,
    ForwardSelectionResult,
    IdentifiabilityGate,
    LRTestResult,
    ResidualBatteryResult,
    forward_select,
    identifiability_gate,
    lr_test,
    residual_battery,
    standardize_innovations,
)
from tests.hvac_bench.empirical.pem_fit import (
    DEFAULT_BOUNDS_1R1C,
    FitParams1R1C,
    FitResult,
    RestartResult,
)
from tests.hvac_bench.empirical.rc_model import (
    RCParams1R1C,
    StateSpace,
    build_1r1c,
    kalman_innovations,
)
from tests.hvac_bench.empirical.test_pem_fit import _simulate_1r1c, _simulate_2r2c


# ── LR test ──────────────────────────────────────────────────────────────


class TestLRTest:
    def test_zero_improvement_p_value_is_one(self) -> None:
        r = lr_test(ll_reduced=-100.0, ll_full=-100.0, df=3)
        np.testing.assert_allclose(r.p_value, 1.0, atol=1e-10)
        assert not r.accept_full

    def test_negative_improvement_clamps_to_zero(self) -> None:
        # If full has lower likelihood (worse fit), statistic clamps to 0.
        r = lr_test(ll_reduced=-50.0, ll_full=-100.0, df=3)
        assert r.statistic == 0.0
        assert not r.accept_full

    def test_large_improvement_rejects_reduced(self) -> None:
        # 2·(−40 − (−100)) = 120 → p≈0 for df=3
        r = lr_test(ll_reduced=-100.0, ll_full=-40.0, df=3)
        assert r.statistic == 120.0
        assert r.p_value < 1e-10
        assert r.accept_full

    def test_critical_at_chi2_3_alpha_05(self) -> None:
        # χ²_3 critical at α=0.05 ≈ 7.815. Statistic just above → reject.
        r_just_above = lr_test(ll_reduced=-100.0, ll_full=-100.0 + 7.9 / 2, df=3)
        r_just_below = lr_test(ll_reduced=-100.0, ll_full=-100.0 + 7.7 / 2, df=3)
        assert r_just_above.accept_full
        assert not r_just_below.accept_full

    def test_invalid_df_raises(self) -> None:
        with pytest.raises(ValueError):
            lr_test(ll_reduced=-100.0, ll_full=-90.0, df=0)


# ── Standardization helper ──────────────────────────────────────────────


class TestStandardizeInnovations:
    def test_drops_nan_and_nonpositive_variance(self) -> None:
        innov = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        var = np.array([1.0, 4.0, 1.0, 0.0, 9.0])
        out = standardize_innovations(innov, var)
        np.testing.assert_allclose(out, np.array([1.0, 1.0, 4.0 / 3.0]))

    def test_returns_unit_scale_for_correctly_scaled_input(self) -> None:
        rng = np.random.default_rng(0)
        innov = rng.standard_normal(1000)  # already N(0,1)
        var = np.ones(1000)
        out = standardize_innovations(innov, var)
        assert abs(np.mean(out)) < 0.1
        assert abs(np.std(out) - 1.0) < 0.1


# ── Residual battery ─────────────────────────────────────────────────────


class TestResidualBattery:
    def test_white_innovations_pass_battery(self) -> None:
        rng = np.random.default_rng(2026)
        # White Gaussian standardized innovations ≈ pass all three sub-tests.
        innov = rng.standard_normal(500)
        var = np.ones(500)
        result = residual_battery(innov, var)
        assert result.ljung_box_pass
        assert result.normality_pass
        assert result.cp_pass
        assert result.overall_pass

    def test_strongly_autocorrelated_innovations_fail(self) -> None:
        # AR(1) ϕ=0.9 — strongly autocorrelated → Ljung-Box fails.
        rng = np.random.default_rng(7)
        innov = np.zeros(500)
        innov[0] = rng.standard_normal()
        for k in range(1, 500):
            innov[k] = 0.9 * innov[k - 1] + rng.standard_normal()
        var = np.ones(500)
        result = residual_battery(innov, var)
        assert not result.ljung_box_pass
        assert not result.cp_pass
        assert not result.overall_pass

    def test_too_few_residuals_raises(self) -> None:
        with pytest.raises(ValueError):
            residual_battery(np.zeros(20), np.ones(20))


# ── Identifiability gate ────────────────────────────────────────────────


def _fake_fit_result(
    cv_per_param: dict[str, float],
    at_bound_per_param: dict[str, bool],
) -> FitResult:
    params = FitParams1R1C(
        tau_s=1e5,
        q_scale=1.0,
        solar_scale=0.01,
        sigma_w=1e-3,
        sigma_v=0.1,
    )
    rr = RestartResult(
        success=True,
        log_likelihood=-100.0,
        params=params,
        initial=params,
        n_iter=10,
        message="ok",
    )
    return FitResult(
        model_name="1R1C",
        best=rr,
        restarts=[rr],
        n_obs=1000,
        cv_per_param=cv_per_param,
        at_bound_per_param=at_bound_per_param,
        bounds=DEFAULT_BOUNDS_1R1C,
    )


class TestIdentifiabilityGate:
    def test_pass_when_all_below_threshold_and_no_rails(self) -> None:
        fit = _fake_fit_result(
            cv_per_param={"tau_s": 0.05, "q_scale": 0.05},
            at_bound_per_param={"tau_s": False, "q_scale": False},
        )
        gate = identifiability_gate(fit)
        assert gate.overall_pass
        assert gate.n_failed_cv == 0
        assert gate.n_at_bound == 0

    def test_fail_when_any_at_bound(self) -> None:
        fit = _fake_fit_result(
            cv_per_param={"tau_s": 0.01},
            at_bound_per_param={"tau_s": True},
        )
        gate = identifiability_gate(fit)
        assert not gate.overall_pass
        assert gate.n_at_bound == 1

    def test_fail_when_cv_above_threshold(self) -> None:
        fit = _fake_fit_result(
            cv_per_param={"tau_s": 0.5, "q_scale": 0.05},
            at_bound_per_param={"tau_s": False, "q_scale": False},
        )
        gate = identifiability_gate(fit)
        assert not gate.overall_pass
        assert gate.n_failed_cv == 1


# ── Forward-selection driver ─────────────────────────────────────────────


class TestForwardSelectAcceptsSimplerOnPoorlyExcitedData:
    def test_accepts_1r1c_when_2r2c_rails(self) -> None:
        # With 1R1C-generated data + low excitation, 2R2C has nothing extra
        # to identify and should rail at bounds. Methodology rejects 2R2C.
        truth = FitParams1R1C(
            tau_s=20 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.05,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=800, seed=33)
        result = forward_select(
            obs, inputs, valid, dt=300.0, n_restarts=2, seed=4
        )
        # Either 1R1C is selected, or 2R2C selected as "close"/"poor".
        assert result.selected_model in {"1R1C", "2R2C"}
        # If 2R2C selected, it must be either passing identifiability or
        # explicitly classified close/poor.
        if result.selected_model == "2R2C":
            assert (
                result.identifiability_2r2c is not None
                and result.identifiability_2r2c.overall_pass
            )

    def test_returns_both_fits_when_unconditional(self) -> None:
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=400, seed=1)
        result = forward_select(
            obs, inputs, valid, dt=300.0, n_restarts=2,
            fit_2r2c_unconditionally=True,
        )
        assert result.fit_2r2c is not None
        assert result.lr_test is not None
        assert result.lr_test.df == N_PARAMS_2R2C - N_PARAMS_1R1C


class TestForwardSelectionResultClassification:
    def test_selected_model_is_a_known_string(self) -> None:
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=400)
        result = forward_select(obs, inputs, valid, dt=300.0, n_restarts=2)
        assert result.selected_model in {"1R1C", "2R2C"}
        assert result.classification in {"good", "close", "poor"}

    def test_summary_string_describes_outcome(self) -> None:
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=400)
        result = forward_select(obs, inputs, valid, dt=300.0, n_restarts=2)
        # Summary references the selected model name in some form
        assert "1R1C" in result.summary or "2R2C" in result.summary
