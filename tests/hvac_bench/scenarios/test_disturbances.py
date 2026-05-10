"""Unmodeled disturbance tests.

Tests controller's ability to compensate for thermal events that are NOT
in the controller's model inputs. The PI integral must handle these alone.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.disturbances import (
    oil_boiler, front_door_open, garage_door_open, cooking, party,
)
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


def _make_controller(profile, seed_factor=1.0):
    seed = profile.true_seed
    return TasmotaPIAdapter({
        "pi_outdoor_seed_heat": seed * seed_factor,
        "pi_outdoor_seed_cool": seed * seed_factor,
    })


def _record_run(bench_metrics, history, *, profile_name, scenario, desired):
    """Record control-quality rollup + disturbance-relevant deviations."""
    bench_metrics["profile_name"] = profile_name
    bench_metrics["scenario"] = scenario
    bench_metrics["n_ticks"] = len(history)
    rollup = compute_all_metrics(history, desired=desired)
    for k, v in rollup.items():
        bench_metrics[f"rollup_{k}"] = v
    if history:
        bench_metrics["max_room_temp"] = max(h["room_temp"] for h in history)
        bench_metrics["min_room_temp"] = min(h["room_temp"] for h in history)
        bench_metrics["final_room_temp"] = history[-1]["room_temp"]


# ── Oil Boiler (unknown heat source) ─────────────────────────────────────


class TestOilBoiler:
    """Oil boiler turns on for 30 min, adding ~3°C of heat.

    Controller has no model input for this — must detect and compensate
    via integral action.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_oil_boiler_recovery(self, bench_metrics, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=5.0)
        model.add_disturbance(oil_boiler(start_tick=8))

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    scenario="oil_boiler", desired=20.5)

        # Room should recover to target after disturbance ends
        post_disturbance = [h for h in history if h["tick"] >= 16]
        for h in post_disturbance:
            assert abs(h["room_temp"] - 20.5) < 2.5, (
                f"{profile_name} tick {h['tick']}: room={h['room_temp']:.1f} "
                f"(failed to recover from oil boiler)"
            )


# ── Front Door Open (tau change) ─────────────────────────────────────────


class TestFrontDoorOpen:
    """Front door open for 15 min — tau drops 50%.

    Rapid heat loss that the controller must compensate for.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_front_door_recovery(self, bench_metrics, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=0.0)
        model.add_disturbance(front_door_open(start_tick=8))

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    scenario="front_door_open", desired=20.5)

        # Should recover within 8 ticks (2 hours) after door closes
        post = [h for h in history if h["tick"] >= 12]
        for h in post:
            assert abs(h["room_temp"] - 20.5) < 3.0, (
                f"{profile_name} tick {h['tick']}: room={h['room_temp']:.1f}"
            )


# ── Garage Door Open (sustained tau change) ──────────────────────────────


class TestGarageDoorOpen:
    """Garage door stays open for 60 min — tau drops 30%.

    Sustained plant parameter change. Tests ability to maintain
    comfort during extended leaky conditions.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_garage_door_during(self, bench_metrics, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=2.0)
        model.add_disturbance(garage_door_open(start_tick=8))

        history = run_scenario(ctrl, model, n_ticks=24, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    scenario="garage_door_open", desired=20.5)

        # Room may drop but shouldn't crash
        min_temp = min(h["room_temp"] for h in history if h["tick"] >= 8)
        assert min_temp > 16.0, (
            f"{profile_name}: room dropped to {min_temp:.1f}°C during garage open"
        )


# ── Cooking (mild heat gain) ─────────────────────────────────────────────


class TestCooking:
    """Cooking for 45 min — moderate unmodeled heat gain."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cooking_no_overshoot(self, bench_metrics, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=5.0)
        model.add_disturbance(cooking(start_tick=8))

        history = run_scenario(ctrl, model, n_ticks=24, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    scenario="cooking", desired=20.5)

        # Cooking adds heat — room should warm slightly, not overshoot wildly
        max_temp = max(h["room_temp"] for h in history)
        assert max_temp < 23.0, (
            f"{profile_name}: room reached {max_temp:.1f}°C during cooking"
        )


# ── Party (sustained large heat gain) ────────────────────────────────────


class TestParty:
    """10 people for 3 hours — sustained +2°C gain.

    Tests long-duration unmodeled disturbance.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_party_recovery(self, bench_metrics, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=5.0)
        model.add_disturbance(party(start_tick=4))

        history = run_scenario(ctrl, model, n_ticks=32, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    scenario="party", desired=20.5)

        # After party ends (tick 16), should recover within 8 ticks
        post_party = [h for h in history if h["tick"] >= 20]
        if post_party:
            for h in post_party:
                assert abs(h["room_temp"] - 20.5) < 3.0, (
                    f"{profile_name} tick {h['tick']}: room={h['room_temp']:.1f}"
                )
