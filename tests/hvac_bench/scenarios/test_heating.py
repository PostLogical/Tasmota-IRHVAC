"""Core heating scenario tests.

These test basic control quality for heating mode across house profiles.
Every controller should pass these — they define minimum viable behavior.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES, HouseProfile2R2C as HouseProfile
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────

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


def _make_model(profile, initial_temp=20.0, outdoor=5.0, **kwargs):
    return ThermalModel(profile=profile, initial_temp=initial_temp,
                        outdoor_temp=outdoor, **kwargs)


# ── Cold Start ────────────────────────────────────────────────────────────


class TestHeatingColdStart:
    """Room starts cold (17°C), needs to warm to 20.5°C."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", SEED_FACTORS)
    def test_cold_start(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=17.0, outdoor=2.0)

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat")
        m = compute_all_metrics(history, desired=20.5)

        # Must reach target eventually
        final_error = abs(history[-1]["room_temp"] - 20.5)
        assert final_error < 2.0, (
            f"{profile_name} seed={seed_factor}: final error {final_error:.1f}°C"
        )

        # Setpoint stays in bounds
        for h in history:
            assert 16 <= h["hp_setpoint"] <= 30


# ── Cold Snap ─────────────────────────────────────────────────────────────


class TestHeatingColdSnap:
    """Outdoor drops 15°C over 3 hours while room should stay at target."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", SEED_FACTORS)
    def test_cold_snap(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=10.0)

        def outdoor(tick):
            return max(-5.0, 10.0 - tick * 1.25)

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat",
                               outdoor_schedule=outdoor)

        # Room should stay within tolerance
        tol = 3.0 if seed_factor == 0.0 else 2.0
        for h in history:
            if h["tick"] > 6:
                assert abs(h["room_temp"] - 20.5) < tol, (
                    f"{profile_name} seed={seed_factor} tick {h['tick']}: "
                    f"room={h['room_temp']:.1f}"
                )


# ── Setpoint Steps ────────────────────────────────────────────────────────


class TestHeatingSetpointUp:
    """User raises desired temp by 2°C at tick 10."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_setpoint_up(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=5.0)

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat",
                               desired_schedule={10: 22.5})

        # Should reach new target
        final_error = abs(history[-1]["room_temp"] - 22.5)
        assert final_error < 2.0, (
            f"{profile_name} seed={seed_factor}: final error {final_error:.1f}°C"
        )


class TestHeatingSetpointDown:
    """User lowers desired temp by 2°C at tick 10."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_setpoint_down(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(22.5)
        model = _make_model(profile, initial_temp=22.5, outdoor=5.0)

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat",
                               desired_schedule={10: 20.5})

        final_error = abs(history[-1]["room_temp"] - 20.5)
        assert final_error < 2.0


# ── Steady State ──────────────────────────────────────────────────────────


class TestHeatingSteadyState:
    """Room starts at target, should stay stable."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_steady_state(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=5.0)

        history = run_scenario(ctrl, model, n_ticks=48, mode="heat")

        # Last 16 ticks should be stable
        late_temps = [h["room_temp"] for h in history[-16:]]
        temp_range = max(late_temps) - min(late_temps)
        assert temp_range < 2.0, (
            f"{profile_name} seed={seed_factor}: range {temp_range:.1f}°C in last 16 ticks"
        )


# ── Ramp Disturbance ─────────────────────────────────────────────────────


class TestHeatingRampDisturbance:
    """Outdoor drops slowly at 1°C/hour (realistic weather)."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0])
    def test_ramp_disturbance(self, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=5.0)

        # 1°C/hour = 0.25°C per 15-min tick
        def outdoor(tick):
            return 5.0 - tick * 0.25

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat",
                               outdoor_schedule=outdoor)

        # Should track within tolerance despite continuous disturbance
        for h in history:
            if h["tick"] > 8:
                assert abs(h["room_temp"] - 20.5) < 2.5, (
                    f"{profile_name} seed={seed_factor} tick {h['tick']}: "
                    f"room={h['room_temp']:.1f}, outdoor={h['outdoor']:.1f}"
                )
