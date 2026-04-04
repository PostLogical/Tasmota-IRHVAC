"""Sensor noise robustness tests.

Verifies that the controller degrades gracefully (not catastrophically)
when sensor readings include realistic noise levels.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


def _make_controller(seed_factor=1.0):
    return TasmotaPIAdapter({
        "pi_ff_heat_slope": 0.35 * seed_factor,
        "pi_ff_cool_slope": 0.35 * seed_factor,
    })


# ── Mild Noise (RTD typical) ─────────────────────────────────────────────


class TestMildNoise:
    """σ=0.1°C, 0.1°C quantization — typical RTD sensor."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_start_with_noise(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=17.0, outdoor_temp=2.0,
            sensor_noise_sigma=0.1, sensor_quantization=0.1, noise_seed=42,
        )

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat")

        # Same tolerance as clean — noise shouldn't make it worse
        final_error = abs(history[-1]["room_temp"] - 20.5)
        assert final_error < 2.5, (
            f"{profile_name}: final error {final_error:.1f}°C with mild noise"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_steady_state_with_noise(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=20.5, outdoor_temp=5.0,
            sensor_noise_sigma=0.1, sensor_quantization=0.1, noise_seed=42,
        )

        history = run_scenario(ctrl, model, n_ticks=48, mode="heat")
        m = compute_all_metrics(history, desired=20.5)

        # Noise shouldn't cause limit cycles
        assert m["reversals"] < 10, (
            f"{profile_name}: {m['reversals']} reversals with mild noise"
        )


# ── Heavy Noise (stress test) ────────────────────────────────────────────


class TestHeavyNoise:
    """σ=0.3°C — worst-case for cheap sensors or EMI."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_start_heavy_noise(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=17.0, outdoor_temp=2.0,
            sensor_noise_sigma=0.3, sensor_quantization=0.1, noise_seed=42,
        )

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat")

        # Wider tolerance but should still reach target
        final_error = abs(history[-1]["room_temp"] - 20.5)
        assert final_error < 3.0, (
            f"{profile_name}: final error {final_error:.1f}°C with heavy noise"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_steady_state_heavy_noise(self, profile_name):
        """Heavy noise shouldn't cause runaway or crash."""
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=20.5, outdoor_temp=5.0,
            sensor_noise_sigma=0.3, sensor_quantization=0.1, noise_seed=42,
        )

        history = run_scenario(ctrl, model, n_ticks=48, mode="heat")

        # No runaway — room should stay in bounds
        for h in history:
            assert abs(h["room_temp"] - 20.5) < 5.0, (
                f"{profile_name} tick {h['tick']}: room={h['room_temp']:.1f} (runaway?)"
            )
            assert 16 <= h["hp_setpoint"] <= 30
