"""Unit tests for ``summarize_post_fill`` — the bench's post-buffer-fill
identification-quality scorer.

Stays a unit test (slow tier eligible but cheap): operates on a
hand-constructed ``FullStackResult`` rather than running a sim.
"""

from __future__ import annotations

import math

import pytest

from tests.hvac_bench.full_stack_runner import (
    FullStackResult,
    PostFillId,
    summarize_post_fill,
)
from tests.hvac_bench.conftest import check_bench_metrics


def _make_result(
    *,
    coef_trajectory: list[dict],
    daily_buffer_utilization: list[float],
    true_coefs: dict[str, float] | None = None,
) -> FullStackResult:
    """Minimal FullStackResult with the fields summarize_post_fill reads."""
    return FullStackResult(
        history=[],
        coef_trajectory=coef_trajectory,
        final_coefs={},
        true_coefs=true_coefs or {},
        coef_errors={},
        integral_rms=0.0,
        total_itae=0.0,
        total_violations=0,
        total_reversals=0,
        batches_to_converge=None,
        n_batches=len(coef_trajectory),
        n_ticks=0,
        daily_itae=[], daily_violations=[], daily_reversals=[],
        daily_mae=[], daily_integral_rms=[],
        weekly_itae=[], weekly_violations=[], weekly_reversals=[],
        checkpoint_data=[],
        comfort_hours_pct=0.0, ctrl_comfort_pct=0.0,
        cold_violations=0, warm_violations=0,
        ctrl_violations=0, unctrl_violations=0,
        worst_undershoot=0.0, worst_overshoot=0.0,
        longest_violation_streak=0,
        ctrl_worst_undershoot=0.0, ctrl_worst_overshoot=0.0,
        ctrl_longest_violation_streak=0,
        hp_capacity_violations=0, worst_hp_capacity_error=0.0,
        longest_hp_capacity_streak=0,
        daily_comfort_pct=[], daily_ctrl_comfort_pct=[],
        daily_cold_violations=[], daily_warm_violations=[],
        daily_ff_fraction=[],
        daily_covariance_trace=[],
        daily_buffer_utilization=daily_buffer_utilization,
        batch_kappa=[], batch_covariance_trace=[],
        batch_std_err_trajectory=[],
        ticks_hp_on=0, ticks_uncertain=0, ticks_hp_off=0,
        observation_yield_pct=0.0,
        final_cal_min=0.0, final_cal_max=0.0,
        boundary_updates=0, boundary_stall_count=0,
        boundary_last_n_obs=0,
        daily_setpoint_limited_pct=[],
        daily_rapid_sp_changes=[],
        total_rapid_sp_changes=0,
    )


# ── Empty / edge cases ───────────────────────────────────────────────


class TestSummarizePostFillEdgeCases:
    def test_no_truth_returns_empty(self, bench_metrics, num_regression):
        """No true_coefs → no entries to score."""
        result = _make_result(
            coef_trajectory=[{"outdoor_delta": -0.25}] * 5,
            daily_buffer_utilization=[0.5, 1.0, 1.0],
            true_coefs=None,
        )
        out = summarize_post_fill(
            result, bias_tols={"outdoor_delta": 0.05},
            std_tols={"outdoor_delta": 0.03},
        )
        bench_metrics["n_entries"] = len(out)
        check_bench_metrics(num_regression, bench_metrics)
        assert out == {}

    def test_buffer_never_fills_returns_nan_metrics(self, bench_metrics, num_regression):
        """Without fill_day, all metrics are NaN and converges=False."""
        result = _make_result(
            coef_trajectory=[{"outdoor_delta": -0.20}] * 5,
            daily_buffer_utilization=[0.5, 0.6, 0.7, 0.8, 0.9],
            true_coefs={"outdoor_delta": -0.25},
        )
        out = summarize_post_fill(
            result, bias_tols={"outdoor_delta": 0.05},
            std_tols={"outdoor_delta": 0.03},
        )
        pf = out["outdoor_delta"]
        # bias/std are NaN — filtered by check_bench_metrics; record n_entries
        bench_metrics["n_entries"] = len(out)
        check_bench_metrics(num_regression, bench_metrics)
        assert "outdoor_delta" in out
        assert pf.fill_day is None
        assert math.isnan(pf.bias)
        assert math.isnan(pf.std)
        assert pf.converges is False


# ── Bias / std / drift on known synthetic trajectories ───────────────


class TestSummarizePostFillMetrics:
    def test_constant_trajectory_post_fill_zero_drift_bias_exceeds_tol(self, bench_metrics, num_regression):
        """β stays at -0.18 throughout post-fill: bias = +0.07, std = 0,
        drift = 0.  bias > bias_tol=0.05 so converges=False."""
        # 30 days, fill at day 5, 12h batches → 60 batches over the run.
        coef_trajectory = [{"outdoor_delta": -0.18}] * 60
        daily_util = [0.0] * 5 + [1.0] * 25  # fill_day = 5
        result = _make_result(
            coef_trajectory=coef_trajectory,
            daily_buffer_utilization=daily_util,
            true_coefs={"outdoor_delta": -0.25},
        )
        out = summarize_post_fill(
            result, bias_tols={"outdoor_delta": 0.05},
            std_tols={"outdoor_delta": 0.03},
        )
        pf = out["outdoor_delta"]
        bench_metrics["fill_day"] = pf.fill_day
        bench_metrics["bias"] = pf.bias
        bench_metrics["std"] = pf.std
        bench_metrics["drift_per_day"] = pf.drift_per_day
        check_bench_metrics(num_regression, bench_metrics)
        assert pf.fill_day == 5
        assert pf.bias == pytest.approx(0.07, abs=1e-9)
        assert pf.std == 0.0
        assert pf.drift_per_day == 0.0
        assert pf.converges is False

    def test_within_tol_converges(self, bench_metrics, num_regression):
        """Constant β at -0.24, truth = -0.25 → bias = 0.01 < 0.05,
        std = 0 < 0.03 → converges."""
        coef_trajectory = [{"outdoor_delta": -0.24}] * 60
        daily_util = [0.0] * 5 + [1.0] * 25
        result = _make_result(
            coef_trajectory=coef_trajectory,
            daily_buffer_utilization=daily_util,
            true_coefs={"outdoor_delta": -0.25},
        )
        out = summarize_post_fill(
            result, bias_tols={"outdoor_delta": 0.05},
            std_tols={"outdoor_delta": 0.03},
        )
        pf = out["outdoor_delta"]
        bench_metrics["bias"] = pf.bias
        bench_metrics["std"] = pf.std
        check_bench_metrics(num_regression, bench_metrics)
        assert pf.converges is True

    def test_drift_slope_matches_synthetic_linear_trajectory(self, bench_metrics, num_regression):
        """β_post = -0.25 + 0.005 * day → drift should be 0.005/day."""
        # 30 days, 12h batches → 60 batches, fill at day 0
        # day_j = j / 2 (since 2 batches per day)
        coef_trajectory = [
            {"outdoor_delta": -0.25 + 0.005 * (j / 2.0)}
            for j in range(60)
        ]
        daily_util = [1.0] * 30
        result = _make_result(
            coef_trajectory=coef_trajectory,
            daily_buffer_utilization=daily_util,
            true_coefs={"outdoor_delta": -0.25},
        )
        out = summarize_post_fill(
            result, bias_tols={"outdoor_delta": 1.0},
            std_tols={"outdoor_delta": 1.0},
        )
        pf = out["outdoor_delta"]
        bench_metrics["drift_per_day"] = pf.drift_per_day
        check_bench_metrics(num_regression, bench_metrics)
        assert pf.drift_per_day == 0.005
        assert pf.improves is False

    def test_improves_flag_picks_up_convergence_to_truth(self, bench_metrics, num_regression):
        """β starts off at -0.20, drifts toward -0.25 by end → improves=True."""
        # Linear: β = -0.20 - 0.001 * j (over 50 batches, ends near -0.25)
        coef_trajectory = [
            {"outdoor_delta": -0.20 - 0.001 * j}
            for j in range(50)
        ]
        daily_util = [1.0] * 25
        result = _make_result(
            coef_trajectory=coef_trajectory,
            daily_buffer_utilization=daily_util,
            true_coefs={"outdoor_delta": -0.25},
        )
        out = summarize_post_fill(
            result, bias_tols={"outdoor_delta": 0.10},
            std_tols={"outdoor_delta": 0.10},
        )
        pf = out["outdoor_delta"]
        bench_metrics["drift_per_day"] = pf.drift_per_day
        bench_metrics["bias"] = pf.bias
        check_bench_metrics(num_regression, bench_metrics)
        # |β_end − truth| = |(-0.20 - 0.049) - (-0.25)| = 0.001
        # |β_fill − truth| = |(-0.20) - (-0.25)| = 0.05
        # Strictly closer at end → improves.
        assert pf.improves is True
