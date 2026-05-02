"""Unit tests for the reference-controller suite (Phase 2a).

Spec-grounded: each test fixes a property (e.g. "bang-bang rails to max
when room is below desired - hysteresis") and asserts it directly.
Avoids tying tests to a specific scenario score — that's Phase 2b's job.
"""

from __future__ import annotations

import pytest

from tests.hvac_bench.controller_protocol import HVACController
from tests.hvac_bench.full_stack_runner import ModelInputSpec
from tests.hvac_bench.reference_controllers import (
    NaiveBangBangController,
    _BatchAwarePIController,
    make_production_pi,
    make_well_tuned_pi,
)


# ── NaiveBangBangController ───────────────────────────────────────────────


class TestNaiveBangBang:
    def test_protocol_conformance(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0)
        assert isinstance(c, HVACController)

    def test_heat_below_band_rails_to_max(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0,
                                    hysteresis_c=0.5)
        c.set_mode("heat")
        c.set_desired_temp(20.0)
        sp = c.tick(room_temp_c=18.0, outdoor_temp_c=-5.0,
                    dt_seconds=900, model_inputs=None)
        assert sp == 30.0

    def test_heat_above_band_drops_to_min(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0,
                                    hysteresis_c=0.5)
        c.set_mode("heat")
        c.set_desired_temp(20.0)
        # First push room hot
        c.tick(room_temp_c=22.0, outdoor_temp_c=-5.0, dt_seconds=900)
        assert c.hp_setpoint == 16.0

    def test_heat_inside_band_holds(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0,
                                    hysteresis_c=0.5,
                                    initial_hp_setpoint=30.0)
        c.set_mode("heat")
        c.set_desired_temp(20.0)
        # First, room cold → rails to max
        c.tick(room_temp_c=18.0, outdoor_temp_c=-5.0, dt_seconds=900)
        # Then within hysteresis: room=20.3 (|err|=0.3 < 0.5)
        sp = c.tick(room_temp_c=20.3, outdoor_temp_c=-5.0, dt_seconds=900)
        assert sp == 30.0  # held

    def test_cool_above_band_pulls_down(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0)
        c.set_mode("cool")
        c.set_desired_temp(24.0)
        sp = c.tick(room_temp_c=27.0, outdoor_temp_c=35.0, dt_seconds=900)
        assert sp == 16.0

    def test_set_mode_resets_setpoint(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0)
        c.set_mode("heat")
        assert c.hp_setpoint == 30.0
        c.set_mode("cool")
        assert c.hp_setpoint == 16.0

    def test_get_state_returns_required_keys(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0)
        s = c.get_state()
        assert "integral" in s
        assert "ff_offset" in s
        assert s["integral"] == 0.0
        assert s["ff_offset"] == 0.0

    def test_batch_update_is_noop(self):
        c = NaiveBangBangController(min_temp=16.0, max_temp=30.0)
        # Should not raise
        c.batch_update(tick=100)


# ── PI factories ──────────────────────────────────────────────────────────


class TestWellTunedPI:
    def test_protocol_conformance(self):
        c = make_well_tuned_pi("living_room", mode="heat")
        assert isinstance(c, HVACController)
        assert isinstance(c, _BatchAwarePIController)

    def test_learning_disabled_in_pi_config(self):
        c = make_well_tuned_pi("living_room", mode="heat")
        # Both batch WLS and plant ID off: well-tuned means frozen.
        assert c.adapter._config["pi_batch_wls_enabled"] is False
        assert c.adapter._config["pi_plant_id_enabled"] is False

    def test_outdoor_seed_matches_profile_truth_heat(self):
        from tests.hvac_bench.house_profiles import PROFILES_2R2C
        profile = PROFILES_2R2C["living_room"]
        c = make_well_tuned_pi("living_room", mode="heat")
        assert c.adapter._config["pi_outdoor_seed_heat"] == profile.true_seed

    def test_outdoor_seed_matches_profile_truth_cool(self):
        from tests.hvac_bench.house_profiles import PROFILES_2R2C
        profile = PROFILES_2R2C["living_room"]
        c = make_well_tuned_pi("living_room", mode="cool")
        # cool seed override goes into pi_outdoor_seed_cool
        assert c.adapter._config["pi_outdoor_seed_cool"] == profile.true_seed

    def test_model_input_seeds_use_true_ff_coef(self):
        mi = ModelInputSpec(
            name="solar",
            entity_id="sensor.solar",
            input_role="solar",
            _true_ff_coef=-2.5,
            seed_heat=0.0,  # explicitly wrong; should be overridden by truth
        )
        c = make_well_tuned_pi("living_room", mode="heat", model_inputs=[mi])
        configured = c.adapter._config["pi_model_inputs"]
        assert len(configured) == 1
        assert configured[0]["seed_heat"] == -2.5

    def test_batch_update_is_noop_when_learning_disabled(self):
        c = make_well_tuned_pi("living_room", mode="heat")
        # Capture the WLS run count before/after; with learning disabled,
        # batch_update should not invoke _run_batch_analysis.
        called = []
        original = c.pi._run_batch_analysis
        c.pi._run_batch_analysis = lambda: called.append(True)
        try:
            c.batch_update(tick=100)
        finally:
            c.pi._run_batch_analysis = original
        assert called == []


class TestProductionPI:
    def test_protocol_conformance(self):
        c = make_production_pi("living_room", mode="heat")
        assert isinstance(c, HVACController)

    def test_learning_enabled_by_default(self):
        c = make_production_pi("living_room", mode="heat")
        assert c.adapter._config["pi_batch_wls_enabled"] is True
        assert c.adapter._config["pi_plant_id_enabled"] is True

    def test_default_seeds_not_truth(self):
        # Without seed_with_truth, the conftest default seed (0.25) is
        # left in place. bunkroom's true_seed (~0.235) differs, so a
        # default-seeded production controller on bunkroom must show
        # 0.25 (conftest default), not bunkroom.true_seed.
        from tests.hvac_bench.house_profiles import PROFILES_2R2C
        profile = PROFILES_2R2C["bunkroom"]
        assert profile.true_seed != 0.25, "test premise broken: profile truth equals conftest default"
        c = make_production_pi("bunkroom", mode="heat")
        assert c.adapter._config["pi_outdoor_seed_heat"] == 0.25

    def test_seed_with_truth_overrides(self):
        from tests.hvac_bench.house_profiles import PROFILES_2R2C
        profile = PROFILES_2R2C["living_room"]
        c = make_production_pi("living_room", mode="heat", seed_with_truth=True)
        assert c.adapter._config["pi_outdoor_seed_heat"] == profile.true_seed

    def test_batch_update_invokes_run_batch_analysis(self):
        c = make_production_pi("living_room", mode="heat")
        called = []
        original = c.pi._run_batch_analysis
        c.pi._run_batch_analysis = lambda: called.append(True)
        try:
            c.batch_update(tick=100)
        finally:
            c.pi._run_batch_analysis = original
        assert called == [True]


# ── Cross-controller behavioral check ─────────────────────────────────────


class TestCrossControllerBehavior:
    """Quick sanity check that the three controllers produce different
    setpoint trajectories on a simple step input.

    Each controller is fed the same room/outdoor sequence; their
    setpoints must differ in the first few ticks (otherwise the suite is
    not discriminating)."""

    def test_first_tick_setpoints_can_differ(self):
        # Cold room (room=18, desired=20), no model inputs, heat mode
        cold_room = 18.0
        outdoor = -5.0
        dt = 900

        # NaiveBangBang: rails to max (30)
        bb = NaiveBangBangController(min_temp=16.0, max_temp=30.0)
        bb.set_mode("heat")
        bb.set_desired_temp(20.0)
        bb_sp = bb.tick(cold_room, outdoor, dt)

        # WellTunedPI: smooth, profile-aware setpoint (something between
        # desired and max — the exact value depends on FF, but it should
        # NOT rail to max when only 2°C below).
        wt = make_well_tuned_pi("living_room", mode="heat")
        wt.set_mode("heat")
        wt.set_desired_temp(20.0)
        wt_sp = wt.tick(cold_room, outdoor, dt)

        # Bang-bang rails; PI does not rail at this small offset.
        assert bb_sp == 30.0
        assert wt_sp != 30.0  # PI does not rail at -2°C error
