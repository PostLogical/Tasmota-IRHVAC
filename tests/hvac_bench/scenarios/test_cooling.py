"""Core cooling scenario tests.

Mirror of heating scenarios with inverted conditions.
Room is too warm, outdoor is hot, HP needs to cool.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


SEED_FACTORS = [0.0, 0.5, 1.0, 1.5]


def _make_controller(seed_factor=1.0, **overrides):
    true_slope = 0.35
    config = {
        "pi_ff_heat_slope": true_slope * seed_factor,
        "pi_ff_cool_slope": true_slope * seed_factor,
        **overrides,
    }
    ctrl = TasmotaPIAdapter(config)
    return ctrl


def _make_model(profile, initial_temp=24.0, outdoor=32.0, **kwargs):
    return ThermalModel(profile=profile, initial_temp=initial_temp,
                        outdoor_temp=outdoor, **kwargs)


# ── Warm Start (cool down) ───────────────────────────────────────────────


class TestCoolingWarmStart:
    """Room starts warm (28°C), needs to cool to 24°C."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_warm_start(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=28.0, outdoor=32.0)

        history = run_scenario(ctrl, model, n_ticks=32, mode="cool")

        final_error = abs(history[-1]["room_temp"] - 24.0)
        assert final_error < 2.5, (
            f"{profile_name} seed={seed_factor}: final error {final_error:.1f}°C"
        )

        for h in history:
            assert 16 <= h["hp_setpoint"] <= 30


# ── Heat Wave ─────────────────────────────────────────────────────────────


class TestCoolingHeatWave:
    """Outdoor rises 10°C over 3 hours while room should stay cool."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0])
    def test_heat_wave(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=24.0, outdoor=30.0)

        def outdoor(tick):
            return min(40.0, 30.0 + tick * 0.8)

        history = run_scenario(ctrl, model, n_ticks=32, mode="cool",
                               outdoor_schedule=outdoor)

        for h in history:
            if h["tick"] > 6:
                assert abs(h["room_temp"] - 24.0) < 3.0, (
                    f"{profile_name} seed={seed_factor} tick {h['tick']}: "
                    f"room={h['room_temp']:.1f}"
                )


# ── Cooling Steady State ─────────────────────────────────────────────────


class TestCoolingSteadyState:
    """Room at target in cooling mode, should stay stable."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0])
    def test_steady_state(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=24.0, outdoor=32.0)

        history = run_scenario(ctrl, model, n_ticks=48, mode="cool")

        late_temps = [h["room_temp"] for h in history[-16:]]
        temp_range = max(late_temps) - min(late_temps)
        assert temp_range < 2.0, (
            f"{profile_name} seed={seed_factor}: range {temp_range:.1f}°C"
        )


# ── Solar Rejection (cooling) ────────────────────────────────────────────


class TestCoolingSolarRejection:
    """Solar gain warms room during cooling mode. Controller must compensate."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_solar_rejection(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=24.0, outdoor=30.0)

        def solar(tick):
            if tick < 4:
                return 0.0
            return min(0.8, (tick - 4) * 0.1)

        history = run_scenario(ctrl, model, n_ticks=32, mode="cool",
                               solar_schedule=solar)

        # Room should stay within tolerance despite solar heating
        for h in history:
            if h["tick"] > 8:
                assert abs(h["room_temp"] - 24.0) < 2.5, (
                    f"{profile_name} tick {h['tick']}: room={h['room_temp']:.1f}"
                )
