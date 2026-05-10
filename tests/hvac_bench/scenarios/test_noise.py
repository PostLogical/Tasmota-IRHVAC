"""Sensor noise robustness tests.

Verifies that the controller degrades gracefully (not catastrophically)
when sensor readings include realistic noise levels.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.conftest import check_bench_metrics, record_scenario_rollup
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario


def _make_controller(profile, seed_factor=1.0):
    seed = profile.true_seed
    return TasmotaPIAdapter({
        "pi_outdoor_seed_heat": seed * seed_factor,
        "pi_outdoor_seed_cool": seed * seed_factor,
    })


# ── Mild Noise (RTD typical) ─────────────────────────────────────────────


class TestMildNoise:
    """σ=0.1°C, 0.1°C quantization — typical RTD sensor."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_start_with_noise(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=17.0, outdoor_temp=2.0,
            sensor_noise_sigma=0.1, sensor_quantization=0.1, noise_seed=42,
            hp_lag_minutes=2.0,
        )

        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="heat")
        record_scenario_rollup(bench_metrics, history, profile_name=profile_name,
                               scenario="mild_noise_cold_start", desired=20.5)

        final_error = abs(history[-1]["room_temp"] - 20.5)
        bench_metrics["final_error"] = final_error
        bench_metrics["final_room_temp"] = history[-1]["room_temp"]
        check_bench_metrics(num_regression, bench_metrics)

        # Same tolerance as clean — noise shouldn't make it worse
        assert final_error < 2.5, (
            f"{profile_name}: final error {final_error:.1f}°C with mild noise"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_steady_state_with_noise(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=20.5, outdoor_temp=5.0,
            sensor_noise_sigma=0.1, sensor_quantization=0.1, noise_seed=42,
            hp_lag_minutes=2.0,
        )

        history = run_scenario(ctrl, model, duration_minutes=12 * 60, mode="heat")
        record_scenario_rollup(bench_metrics, history, profile_name=profile_name,
                               scenario="mild_noise_steady_state", desired=20.5)
        check_bench_metrics(num_regression, bench_metrics)

        # Noise shouldn't cause limit cycles — count reversals from rollup.
        reversals = bench_metrics.get("rollup_reversals", 0)
        assert reversals < 10, (
            f"{profile_name}: {reversals} reversals with mild noise"
        )


# ── Heavy Noise (stress test) ────────────────────────────────────────────


class TestHeavyNoise:
    """σ=0.3°C — worst-case for cheap sensors or EMI."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_start_heavy_noise(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=17.0, outdoor_temp=2.0,
            sensor_noise_sigma=0.3, sensor_quantization=0.1, noise_seed=42,
            hp_lag_minutes=2.0,
        )

        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="heat")
        record_scenario_rollup(bench_metrics, history, profile_name=profile_name,
                               scenario="heavy_noise_cold_start", desired=20.5)

        final_error = abs(history[-1]["room_temp"] - 20.5)
        bench_metrics["final_error"] = final_error
        bench_metrics["final_room_temp"] = history[-1]["room_temp"]
        check_bench_metrics(num_regression, bench_metrics)

        # Wider tolerance but should still reach target
        assert final_error < 3.0, (
            f"{profile_name}: final error {final_error:.1f}°C with heavy noise"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_steady_state_heavy_noise(self, bench_metrics, num_regression, profile_name):
        """Heavy noise shouldn't cause runaway or crash."""
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=20.5, outdoor_temp=5.0,
            sensor_noise_sigma=0.3, sensor_quantization=0.1, noise_seed=42,
            hp_lag_minutes=2.0,
        )

        history = run_scenario(ctrl, model, duration_minutes=12 * 60, mode="heat")
        record_scenario_rollup(bench_metrics, history, profile_name=profile_name,
                               scenario="heavy_noise_steady_state", desired=20.5)
        bench_metrics["max_room_temp"] = max(h["room_temp"] for h in history)
        bench_metrics["min_room_temp"] = min(h["room_temp"] for h in history)
        check_bench_metrics(num_regression, bench_metrics)

        # No runaway — room should stay in bounds
        for h in history:
            assert abs(h["room_temp"] - 20.5) < 5.0, (
                f"{profile_name} tick {h['tick']}: room={h['room_temp']:.1f} (runaway?)"
            )
            assert 16 <= h["hp_setpoint"] <= 30
