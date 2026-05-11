"""Unit tests for the canonical KPI bundle (Phase 2c).

Spec-grounded tests, not output-pinning. Each test fixes a synthetic
history with known properties and asserts the bundle reflects them
exactly. Catches regressions in unit conventions (K·h, kW, hours) and
in the cold/warm split convention.
"""

from __future__ import annotations

import math
import pytest

from tests.hvac_bench.kpis import (
    DEADBAND_C,
    KpiBundle,
    aggregate_mc_bundles,
    attach_learning_kpis,
    compute_control_kpis,
    crlb_coverage,
)
from tests.hvac_bench.conftest import check_bench_metrics


# ── compute_control_kpis ──────────────────────────────────────────────────


def _fixed_history(
    n_ticks: int,
    *,
    error: float = 0.0,
    desired: float = 20.0,
    hp_setpoint: float = 22.0,
    cumulative_kwh_per_tick: float = 0.01,
) -> list[dict]:
    """Generate a flat history with constant error/setpoint/energy slope."""
    return [
        {
            "tick": t,
            "room_temp": desired - error,
            "desired": desired,
            "error": error,
            "hp_setpoint": hp_setpoint,
            "cumulative_kwh": cumulative_kwh_per_tick * (t + 1),
        }
        for t in range(n_ticks)
    ]


class TestControlKpisEmpty:
    def test_empty_history_returns_zeros(self, bench_metrics, num_regression):
        b = compute_control_kpis([], tick_minutes=15.0)
        bench_metrics["n_ticks"] = b.n_ticks
        bench_metrics["tdis_tot"] = b.tdis_tot
        bench_metrics["cold_time_h"] = b.cold_time_h
        bench_metrics["warm_time_h"] = b.warm_time_h
        bench_metrics["ener_tot"] = b.ener_tot
        bench_metrics["peak_kw"] = b.peak_kw
        bench_metrics["setpoint_changes"] = b.setpoint_changes
        check_bench_metrics(num_regression, bench_metrics)
        assert b.n_ticks == 0
        assert b.tdis_tot == 0.0
        assert b.cold_time_h == 0.0
        assert b.warm_time_h == 0.0
        assert b.ener_tot == 0.0
        assert b.peak_kw == 0.0
        assert b.settling_time_h is None
        assert b.setpoint_changes == 0


class TestControlKpisComfort:
    def test_inside_deadband_zero_discomfort(self, bench_metrics, num_regression):
        # Error within deadband (0.5°C) → tdis_tot=0, cold=warm=0
        history = _fixed_history(n_ticks=96, error=0.3)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["tdis_tot"] = b.tdis_tot
        bench_metrics["cold_time_h"] = b.cold_time_h
        bench_metrics["warm_time_h"] = b.warm_time_h
        check_bench_metrics(num_regression, bench_metrics)
        assert b.tdis_tot == 0.0
        assert b.cold_time_h == 0.0
        assert b.warm_time_h == 0.0

    def test_cold_violation_accumulates_kh(self, bench_metrics, num_regression):
        # error=+1.5°C (room cold by 1.5) for 96 ticks of 15 min = 24h
        # excess = 1.5 - 0.5 = 1.0 K; integral = 1.0 * 24 = 24 K·h
        history = _fixed_history(n_ticks=96, error=1.5)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["tdis_tot"] = b.tdis_tot
        bench_metrics["cold_time_h"] = b.cold_time_h
        bench_metrics["warm_time_h"] = b.warm_time_h
        check_bench_metrics(num_regression, bench_metrics)
        assert b.tdis_tot == pytest.approx(24.0, abs=0.01)
        assert b.cold_time_h == pytest.approx(24.0, abs=0.01)
        assert b.warm_time_h == 0.0

    def test_warm_violation_accumulates_kh(self, bench_metrics, num_regression):
        # error=-1.5°C (room warm by 1.5) for 24h
        history = _fixed_history(n_ticks=96, error=-1.5)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["tdis_tot"] = b.tdis_tot
        bench_metrics["cold_time_h"] = b.cold_time_h
        bench_metrics["warm_time_h"] = b.warm_time_h
        check_bench_metrics(num_regression, bench_metrics)
        assert b.tdis_tot == pytest.approx(24.0, abs=0.01)
        assert b.cold_time_h == 0.0
        assert b.warm_time_h == pytest.approx(24.0, abs=0.01)

    def test_deadband_edge_inclusive(self, bench_metrics, num_regression):
        # error == deadband: max(0, 0.5 - 0.5) = 0 → no discomfort
        history = _fixed_history(n_ticks=96, error=DEADBAND_C)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["tdis_tot"] = b.tdis_tot
        bench_metrics["cold_time_h"] = b.cold_time_h
        check_bench_metrics(num_regression, bench_metrics)
        assert b.tdis_tot == 0.0
        assert b.cold_time_h == 0.0


class TestControlKpisEnergy:
    def test_total_kwh_tracks_cumulative(self, bench_metrics, num_regression):
        # 96 ticks, 0.01 kWh per tick → final cumulative_kwh = 0.96
        history = _fixed_history(n_ticks=96, cumulative_kwh_per_tick=0.01)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["ener_tot"] = b.ener_tot
        check_bench_metrics(num_regression, bench_metrics)
        assert b.ener_tot == pytest.approx(0.96, abs=0.001)

    def test_peak_kw_from_per_tick_delta(self, bench_metrics, num_regression):
        # 0.01 kWh per 15-min tick = 0.04 kW peak (constant)
        history = _fixed_history(n_ticks=96, cumulative_kwh_per_tick=0.01)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["peak_kw"] = b.peak_kw
        check_bench_metrics(num_regression, bench_metrics)
        assert b.peak_kw == pytest.approx(0.04, abs=0.001)

    def test_peak_kw_picks_max_tick(self, bench_metrics, num_regression):
        # Spike at tick 50: 0.01 kWh tick generally, then a 0.05 kWh jump
        history = _fixed_history(n_ticks=96, cumulative_kwh_per_tick=0.01)
        # add a spike: tick 50 jumps by an extra 0.04 kWh
        for i in range(50, 96):
            history[i]["cumulative_kwh"] += 0.04
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["peak_kw"] = b.peak_kw
        check_bench_metrics(num_regression, bench_metrics)
        # spike tick: 0.04 + 0.01 = 0.05 kWh, dt=0.25h → 0.20 kW
        assert b.peak_kw == pytest.approx(0.20, abs=0.001)

    def test_no_cumulative_kwh_returns_zero(self, bench_metrics, num_regression):
        # Histories without 'cumulative_kwh' field (no COP model)
        history = [{"tick": t, "room_temp": 20, "desired": 20,
                    "error": 0, "hp_setpoint": 22} for t in range(10)]
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["ener_tot"] = b.ener_tot
        bench_metrics["peak_kw"] = b.peak_kw
        check_bench_metrics(num_regression, bench_metrics)
        assert b.ener_tot == 0.0
        assert b.peak_kw == 0.0


class TestControlKpisSettling:
    def test_always_within_deadband_settles_at_zero(self, bench_metrics, num_regression):
        history = _fixed_history(n_ticks=20, error=0.1)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["settling_time_h"] = b.settling_time_h
        check_bench_metrics(num_regression, bench_metrics)
        assert b.settling_time_h == 0.0

    def test_never_settles_returns_none(self, bench_metrics, num_regression):
        history = _fixed_history(n_ticks=10, error=2.0)
        b = compute_control_kpis(history, tick_minutes=15.0)
        # settling_time_h is None when never settled — filtered by
        # check_bench_metrics, so still call it for consistency.
        bench_metrics["settling_time_h"] = b.settling_time_h
        check_bench_metrics(num_regression, bench_metrics)
        assert b.settling_time_h is None

    def test_late_exit_yields_settling_time(self, bench_metrics, num_regression):
        # First 8 ticks outside deadband, then inside for the rest
        history = _fixed_history(n_ticks=20, error=0.1)
        for i in range(8):
            history[i]["error"] = 2.0
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["settling_time_h"] = b.settling_time_h
        check_bench_metrics(num_regression, bench_metrics)
        # Settled at tick 8 → 8 ticks * 15min/60 = 2.0h
        assert b.settling_time_h == pytest.approx(2.0, abs=0.01)


class TestControlKpisSetpointChanges:
    def test_constant_sp_zero_changes(self, bench_metrics, num_regression):
        history = _fixed_history(n_ticks=20, hp_setpoint=22.0)
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["setpoint_changes"] = b.setpoint_changes
        check_bench_metrics(num_regression, bench_metrics)
        assert b.setpoint_changes == 0

    def test_alternating_counts_per_change(self, bench_metrics, num_regression):
        history = _fixed_history(n_ticks=20, hp_setpoint=22.0)
        for i in range(20):
            history[i]["hp_setpoint"] = 22.0 if i % 2 == 0 else 23.0
        b = compute_control_kpis(history, tick_minutes=15.0)
        bench_metrics["setpoint_changes"] = b.setpoint_changes
        check_bench_metrics(num_regression, bench_metrics)
        # 19 transitions across 20 ticks
        assert b.setpoint_changes == 19


# ── attach_learning_kpis ──────────────────────────────────────────────────


def _baseline_bundle() -> KpiBundle:
    return compute_control_kpis(_fixed_history(10), tick_minutes=15.0)


class TestAttachLearningKpis:
    def test_bias_is_estimated_minus_truth(self, bench_metrics, num_regression):
        b = attach_learning_kpis(
            _baseline_bundle(),
            beta_estimated={"outdoor_delta": -0.20, "solar": -1.5},
            beta_truth={"outdoor_delta": -0.25, "solar": -2.0},
            n_observations=2000,
        )
        bench_metrics["beta_bias_outdoor_delta"] = b.beta_bias["outdoor_delta"]
        bench_metrics["beta_bias_solar"] = b.beta_bias["solar"]
        bench_metrics["n_observations"] = b.n_observations
        check_bench_metrics(num_regression, bench_metrics)
        assert b.beta_bias == {"outdoor_delta": 0.05, "solar": 0.5}
        assert b.n_observations == 2000

    def test_only_shared_features_appear(self, bench_metrics, num_regression):
        b = attach_learning_kpis(
            _baseline_bundle(),
            beta_estimated={"outdoor_delta": -0.20, "solar": -1.5},
            beta_truth={"outdoor_delta": -0.25},  # no solar truth
            n_observations=2000,
        )
        bench_metrics["n_bias_features"] = len(b.beta_bias)
        bench_metrics["n_estimated_features"] = len(b.beta_estimated)
        check_bench_metrics(num_regression, bench_metrics)
        assert set(b.beta_bias) == {"outdoor_delta"}
        assert set(b.beta_estimated) == {"outdoor_delta"}

    def test_passes_diagnostic_flags_through(self, bench_metrics, num_regression):
        b = attach_learning_kpis(
            _baseline_bundle(),
            beta_estimated={"a": 1.0},
            beta_truth={"a": 1.0},
            n_observations=100,
            residual_whiteness_pass=False,
            residual_normality_pass=False,
            condition_number=12.345,
        )
        bench_metrics["condition_number"] = b.condition_number
        check_bench_metrics(num_regression, bench_metrics)
        assert b.residual_whiteness_pass is False
        assert b.residual_normality_pass is False
        assert b.condition_number == 12.345


# ── crlb_coverage ──────────────────────────────────────────────────────────


class TestCrlbCoverage:
    def test_all_realisations_cover_truth_yields_one(self, bench_metrics, num_regression):
        # 10 realisations, all with estimate=-0.25 ± 0.05, truth=-0.25
        ests = [{"outdoor_delta": -0.25} for _ in range(10)]
        ses = [{"outdoor_delta": 0.05} for _ in range(10)]
        cov = crlb_coverage(ests, {"outdoor_delta": -0.25}, ses, k=2.0)
        bench_metrics["coverage_outdoor_delta"] = cov["outdoor_delta"]
        check_bench_metrics(num_regression, bench_metrics)
        assert cov == {"outdoor_delta": 1.0}

    def test_no_realisation_covers_yields_zero(self, bench_metrics, num_regression):
        # estimate=-0.10, truth=-0.25, 2*SE=0.10 → gap=0.15 > 0.10 → out
        ests = [{"outdoor_delta": -0.10} for _ in range(10)]
        ses = [{"outdoor_delta": 0.05} for _ in range(10)]
        cov = crlb_coverage(ests, {"outdoor_delta": -0.25}, ses, k=2.0)
        bench_metrics["coverage_outdoor_delta"] = cov["outdoor_delta"]
        check_bench_metrics(num_regression, bench_metrics)
        assert cov == {"outdoor_delta": 0.0}

    def test_partial_coverage(self, bench_metrics, num_regression):
        # 7 of 10 realisations cover truth
        ests = ([{"a": 1.0}] * 7) + ([{"a": 5.0}] * 3)
        ses = [{"a": 0.5} for _ in range(10)]  # 2σ band = 1.0
        cov = crlb_coverage(ests, {"a": 1.0}, ses, k=2.0)
        bench_metrics["coverage_a"] = cov["a"]
        check_bench_metrics(num_regression, bench_metrics)
        assert cov == {"a": 0.7}

    def test_unidentifiable_counts_as_out(self, bench_metrics, num_regression):
        ests = [{"a": 1.0}, {"a": 1.0}]
        ses = [{"a": 0.1}, {"a": math.inf}]
        cov = crlb_coverage(ests, {"a": 1.0}, ses, k=2.0)
        bench_metrics["coverage_a"] = cov["a"]
        check_bench_metrics(num_regression, bench_metrics)
        # 1/2 in band; the inf realisation is counted as a miss
        assert cov == {"a": 0.5}


# ── aggregate_mc_bundles ──────────────────────────────────────────────────


class TestAggregateMcBundles:
    def test_empty_returns_empty(self, bench_metrics, num_regression):
        agg = aggregate_mc_bundles([])
        bench_metrics["n_kpis"] = len(agg)
        check_bench_metrics(num_regression, bench_metrics)
        assert agg == {}

    def test_summary_stats_per_kpi(self, bench_metrics, num_regression):
        bundles = []
        for tdis in (10.0, 12.0, 14.0):
            b = compute_control_kpis(_fixed_history(96, error=1.0), tick_minutes=15.0)
            # craft a synthetic tdis variation by overriding
            from dataclasses import replace
            bundles.append(replace(b, tdis_tot=tdis))
        agg = aggregate_mc_bundles(bundles)
        bench_metrics["tdis_tot_mean"] = agg["tdis_tot"]["mean"]
        bench_metrics["tdis_tot_n"] = agg["tdis_tot"]["n"]
        check_bench_metrics(num_regression, bench_metrics)
        assert agg["tdis_tot"]["mean"] == 12.0
        assert agg["tdis_tot"]["n"] == 3

    def test_settling_time_only_includes_settled(self, bench_metrics, num_regression):
        from dataclasses import replace
        b0 = compute_control_kpis(_fixed_history(10, error=0.1), tick_minutes=15.0)
        bundles = [
            replace(b0, settling_time_h=1.0),
            replace(b0, settling_time_h=None),
            replace(b0, settling_time_h=2.0),
        ]
        agg = aggregate_mc_bundles(bundles)
        bench_metrics["settling_n_settled"] = agg["settling_time_h"]["n_settled"]
        bench_metrics["settling_n_total"] = agg["settling_time_h"]["n_total"]
        bench_metrics["settling_mean"] = agg["settling_time_h"]["mean"]
        check_bench_metrics(num_regression, bench_metrics)
        assert agg["settling_time_h"]["n_settled"] == 2
        assert agg["settling_time_h"]["n_total"] == 3
        assert agg["settling_time_h"]["mean"] == 1.5
