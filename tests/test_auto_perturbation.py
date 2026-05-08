"""Tests for AutoPerturbation state machine (Layer 2.5).

Unit tests for the pure state machine + integration tests for PI controller wiring.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from custom_components.tasmota_irhvac.pi.auto_perturbation import (
    AMPLITUDE_C,
    CONVERGENCE_DELTA,
    CONVERGENCE_STOP,
    FORCE_TIMEOUT_S,
    INFORMATIVE_DELTA_C,
    MAX_HOLD_S,
    MIN_HOLD_S,
    RESEARCH_STALL_CAP,
    STALL_CAP,
    STEADY_STATE_DWELL_S,
    AutoPerturbation,
    PerturbState,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make(**kwargs) -> AutoPerturbation:
    """Create an enabled AutoPerturbation with defaults."""
    return AutoPerturbation(enabled=True, **kwargs)


def _tick(ap: AutoPerturbation, now: float, **kw) -> float:
    """Tick with all steady-state conditions met."""
    defaults = dict(
        now_mono=now,
        room_temp=20.0,
        room_temp_rate=0.0,
        integral_change_output=0.0,
        ff_settled_ticks=10,
        is_clamped=False,
        supplemental_active=False,
        learning_suppressed=False,
        plant_test_active=False,
        mode_heating=True,
        plant_confidence=0.3,
        current_hour=3,
    )
    defaults.update(kw)
    return ap.tick(**defaults)


def _tick_unsteady(ap: AutoPerturbation, now: float, **kw) -> float:
    """Tick with room NOT stable."""
    return _tick(ap, now, room_temp_rate=0.05, **kw)


def _run_dwell(ap: AutoPerturbation, t: float, **kw) -> float:
    """Run 11 minutes of steady ticks (exceeds 10-min dwell). Returns final t."""
    for _ in range(12):
        _tick(ap, t, **kw)
        t += 60.0
    return t


def _enter_step(ap: AutoPerturbation, t: float = 0.0, **kw) -> float:
    """Advance through dwell to STEP_ACTIVE. Returns current time."""
    t = _run_dwell(ap, t, **kw)
    assert ap.state == PerturbState.STEP_ACTIVE
    return t


def _complete_step(ap: AutoPerturbation, t: float, **kw) -> float:
    """Advance STEP_ACTIVE past min hold + dwell to RESTORE.

    Default simulates room moving 1°C in response to the perturbation
    (high-quality cycle). Pass `room_temp=20.0` to simulate no movement
    (low-quality cycle).
    """
    kw.setdefault("room_temp", 21.0)
    t += MIN_HOLD_S + 60.0
    _tick_unsteady(ap, t, **kw)
    t += 60.0
    t = _run_dwell(ap, t, **kw)
    assert ap.state == PerturbState.RESTORE
    return t


def _complete_restore(ap: AutoPerturbation, t: float, **kw) -> float:
    """Complete RESTORE via dwell. Returns t. Ends in IDLE or STALLED."""
    t = _run_dwell(ap, t, **kw)
    assert ap.state in (PerturbState.IDLE, PerturbState.STALLED)
    return t


def _run_cycle(ap: AutoPerturbation, t: float, **kw) -> tuple[float, float]:
    """Run one full cycle. Returns (offset_during_step, time_after)."""
    t = _enter_step(ap, t, **kw)
    offset = ap.offset
    t = _complete_step(ap, t, **kw)
    t = _complete_restore(ap, t, **kw)
    return offset, t


# ── Basic lifecycle ──────────────────────────────────────────────────

class TestLifecycle:

    def test_disabled_returns_zero(self):
        ap = AutoPerturbation(enabled=False)
        assert _tick(ap, 0.0) == 0.0
        assert ap.state == PerturbState.IDLE

    def test_needs_dwell_before_starting(self):
        ap = _make()
        for i in range(5):
            _tick(ap, i * 60.0)
        assert ap.state == PerturbState.IDLE  # only 5 min

    def test_starts_after_dwell(self):
        ap = _make()
        _enter_step(ap)
        assert abs(ap.offset) == AMPLITUDE_C

    def test_holds_during_min_hold(self):
        ap = _make()
        t = _enter_step(ap)
        t += MIN_HOLD_S - 120.0  # well under min hold
        _tick(ap, t)
        assert ap.state == PerturbState.STEP_ACTIVE

    def test_advances_after_min_hold_and_new_steady(self):
        ap = _make()
        t = _enter_step(ap)
        t = _complete_step(ap, t)
        assert ap.offset == 0.0  # restored

    def test_max_hold_forces_restore(self):
        ap = _make()
        t = _enter_step(ap)
        t += MAX_HOLD_S + 60.0
        _tick_unsteady(ap, t)
        assert ap.state == PerturbState.RESTORE

    def test_full_cycle(self):
        ap = _make()
        _, _ = _run_cycle(ap, 0.0)
        assert ap.cycles_completed == 1
        assert ap.state == PerturbState.IDLE


# ── Direction alternation ────────────────────────────────────────────

class TestDirection:

    def test_alternates_heating(self):
        ap = _make()
        off1, t = _run_cycle(ap, 0.0)
        off2, _ = _run_cycle(ap, t)
        assert off1 == +AMPLITUDE_C
        assert off2 == -AMPLITUDE_C

    def test_cooling_flips(self):
        ap = _make()
        _enter_step(ap, mode_heating=False)
        assert ap.offset == -AMPLITUDE_C  # more cooling effort


# ── Abort ────────────────────────────────────────────────────────────

class TestAbort:

    def test_abort_on_clamp(self):
        ap = _make()
        t = _enter_step(ap)
        _tick(ap, t + 60, is_clamped=True)
        assert ap.state == PerturbState.IDLE

    def test_abort_on_supplemental(self):
        ap = _make()
        t = _enter_step(ap)
        _tick(ap, t + 60, supplemental_active=True)
        assert ap.state == PerturbState.IDLE

    def test_abort_on_learning_suppressed(self):
        ap = _make()
        t = _enter_step(ap)
        _tick(ap, t + 60, learning_suppressed=True)
        assert ap.state == PerturbState.IDLE

    def test_abort_on_plant_test(self):
        ap = _make()
        t = _enter_step(ap)
        _tick(ap, t + 60, plant_test_active=True)
        assert ap.state == PerturbState.IDLE

    def test_explicit_abort(self):
        ap = _make()
        _enter_step(ap)
        ap.abort("user_setpoint_change")
        assert ap.state == PerturbState.IDLE
        assert ap.offset == 0.0

    def test_abort_during_restore(self):
        ap = _make()
        t = _enter_step(ap)
        t += MAX_HOLD_S + 60.0
        _tick_unsteady(ap, t)
        assert ap.state == PerturbState.RESTORE
        ap.abort("mode_change")
        assert ap.state == PerturbState.IDLE


# ── Steady-state detection ───────────────────────────────────────────

class TestSteadyState:

    def test_dwell_resets_on_interruption(self):
        ap = _make()
        t = 0.0
        for _ in range(8):  # 8 min steady
            _tick(ap, t)
            t += 60.0
        _tick_unsteady(ap, t)  # break
        t += 60.0
        for _ in range(8):  # 8 more — not 10 continuous
            _tick(ap, t)
            t += 60.0
        assert ap.state == PerturbState.IDLE

    def test_integral_unstable_blocks(self):
        ap = _make()
        _run_dwell(ap, 0.0, integral_change_output=0.1)
        assert ap.state == PerturbState.IDLE

    def test_ff_unsettled_blocks(self):
        ap = _make()
        _run_dwell(ap, 0.0, ff_settled_ticks=2)
        assert ap.state == PerturbState.IDLE

    def test_noise_floor_widens_steady_threshold(self):
        """When ``room_rate_noise_floor`` is provided, the steady-state
        threshold becomes ``max(FLOOR, K × noise_floor)`` instead of the
        legacy 0.015 °C/min default.  At a noise floor of 0.005, the
        cadence-adaptive threshold = 4 × 0.005 = 0.02 — wider than the
        default — so a rate of 0.018 (which the default would reject)
        now passes the steady-state gate."""
        ap = _make()
        # rate = 0.018: above default 0.015 floor, below 4 × noise_floor=0.02
        t = 0.0
        for _ in range(12):
            _tick(
                ap, t, room_temp_rate=0.018,
                room_rate_noise_floor=0.005,
            )
            t += 60.0
        assert ap.state == PerturbState.STEP_ACTIVE

    def test_noise_floor_default_floor_used_when_below_floor(self):
        """The cadence-adaptive threshold is bounded below by
        ROOM_RATE_STEADY_FLOOR — even with a tiny noise_floor the
        threshold doesn't shrink below 0.005 °C/min.  rate=0.004
        passes; rate=0.006 doesn't."""
        ap = _make()
        t = 0.0
        for _ in range(12):
            _tick(
                ap, t, room_temp_rate=0.004,
                room_rate_noise_floor=1e-6,  # would give 4e-6 < FLOOR
            )
            t += 60.0
        assert ap.state == PerturbState.STEP_ACTIVE


# ── Convergence gating ───────────────────────────────────────────────

class TestConvergence:

    def test_high_confidence_blocks(self):
        ap = _make()
        _run_dwell(ap, 0.0, plant_confidence=0.95)
        assert ap.state == PerturbState.IDLE

    def test_low_confidence_allows(self):
        ap = _make()
        _enter_step(ap, plant_confidence=0.2)


# ── Research mode ────────────────────────────────────────────────────

class TestResearchMode:
    """Path A0 research-mode behavior: bypass convergence gate, higher stall cap."""

    def test_research_mode_default_off(self):
        ap = _make()
        assert ap.research_mode is False

    def test_research_mode_property_reflects_init(self):
        ap = _make(research_mode=True)
        assert ap.research_mode is True

    def test_research_mode_bypasses_convergence(self):
        """High plant_confidence still permits perturbation in research mode."""
        ap = _make(research_mode=True)
        _enter_step(ap, plant_confidence=0.99)

    def test_research_mode_uses_higher_stall_cap(self):
        """Stall does not trigger after STALL_CAP cycles in research mode."""
        ap = _make(research_mode=True)
        t = 0.0
        for _ in range(STALL_CAP + 5):  # exceed default cap
            _, t = _run_cycle(ap, t, plant_confidence=0.3)
        assert ap.state == PerturbState.IDLE
        assert ap.cycles_without_improvement == STALL_CAP + 5

    def test_research_mode_eventually_stalls(self):
        """Research mode still stalls — at the higher cap."""
        ap = _make(research_mode=True)
        t = 0.0
        for _ in range(RESEARCH_STALL_CAP):
            _, t = _run_cycle(ap, t, plant_confidence=0.3)
        assert ap.state == PerturbState.STALLED

    def test_default_mode_unchanged_stall_cap(self):
        """Regression: default mode still stalls at STALL_CAP."""
        ap = _make()
        t = 0.0
        for _ in range(STALL_CAP):
            _, t = _run_cycle(ap, t, plant_confidence=0.3)
        assert ap.state == PerturbState.STALLED


# ── Data-quality gate (A0.3) ─────────────────────────────────────────

class TestDataQuality:
    """Cycles where room_temp barely moved are flagged low-quality and don't
    increment the stall counter — unlucky timing shouldn't penalize the system.
    """

    def test_low_quality_does_not_increment_stall(self):
        """Cycle with no room movement: completes but skips stall counter."""
        ap = _make()
        t = 0.0
        # Step starts at 20.0, restore also at 20.0 → Δ=0 < threshold
        t = _enter_step(ap, t, room_temp=20.0)
        t = _complete_step(ap, t, room_temp=20.0)
        t = _run_dwell(ap, t, room_temp=20.0)
        assert ap.state == PerturbState.IDLE
        assert ap.cycles_completed == 1
        assert ap.cycles_without_improvement == 0
        assert ap.cycles_low_quality == 1

    def test_high_quality_no_improvement_increments_stall(self):
        """Cycle with movement but no plant_confidence improvement: stall counter +1."""
        ap = _make()
        _, _ = _run_cycle(ap, 0.0, plant_confidence=0.3)
        assert ap.cycles_completed == 1
        assert ap.cycles_without_improvement == 1
        assert ap.cycles_low_quality == 0

    def test_low_quality_cycles_dont_cause_stall(self):
        """Many low-quality cycles in a row never reach STALLED state."""
        ap = _make()
        t = 0.0
        for _ in range(STALL_CAP + 3):
            t = _enter_step(ap, t, room_temp=20.0)
            t = _complete_step(ap, t, room_temp=20.0)
            t = _run_dwell(ap, t, room_temp=20.0)
        assert ap.state == PerturbState.IDLE
        assert ap.cycles_low_quality == STALL_CAP + 3
        assert ap.cycles_without_improvement == 0

    def test_just_above_threshold_is_high_quality(self):
        """Δ slightly above INFORMATIVE_DELTA_C is treated as informative."""
        ap = _make()
        # 0.5°C movement is above 0.4 threshold
        t = _enter_step(ap, 0.0, room_temp=20.0)
        t = _complete_step(ap, t, room_temp=20.0 + INFORMATIVE_DELTA_C + 0.1)
        t = _run_dwell(ap, t, room_temp=20.5)
        assert ap.cycles_low_quality == 0
        assert ap.cycles_without_improvement == 1  # no plant_confidence improvement

    def test_just_below_threshold_is_low_quality(self):
        """Δ just below INFORMATIVE_DELTA_C is flagged low-quality."""
        ap = _make()
        t = _enter_step(ap, 0.0, room_temp=20.0)
        t = _complete_step(ap, t, room_temp=20.0 + INFORMATIVE_DELTA_C - 0.01)
        t = _run_dwell(ap, t, room_temp=20.39)
        assert ap.cycles_low_quality == 1

    def test_low_quality_persists_across_restart(self):
        """cycles_low_quality round-trips through as_dict / restore."""
        ap = _make()
        ap._cycles_low_quality = 7
        ap2 = _make()
        ap2.restore(ap.as_dict())
        assert ap2.cycles_low_quality == 7

    def test_missing_snapshots_treated_as_high_quality(self):
        """Defensive: if snapshots aren't set (shouldn't happen via tick flow),
        cycle is treated as high-quality so we don't accidentally suppress the
        stall counter on stale state."""
        ap = _make()
        ap._step_start_temp = None
        ap._restore_start_temp = 20.0
        assert ap._cycle_was_low_quality() is False
        ap._step_start_temp = 20.0
        ap._restore_start_temp = None
        assert ap._cycle_was_low_quality() is False


# ── Stall detection ──────────────────────────────────────────────────

class TestStall:

    def _cycle_no_improvement(self, ap: AutoPerturbation, t: float) -> float:
        _, t = _run_cycle(ap, t, plant_confidence=0.3)
        return t

    def test_stalls_after_cap(self):
        ap = _make()
        t = 0.0
        for _ in range(STALL_CAP):
            t = self._cycle_no_improvement(ap, t)
        assert ap.state == PerturbState.STALLED

    def test_improvement_resets_counter(self):
        ap = _make()
        t = 0.0
        for _ in range(STALL_CAP - 1):
            t = self._cycle_no_improvement(ap, t)
        # One cycle WITH improvement: start at 0.3, finish at higher
        t = _enter_step(ap, t, plant_confidence=0.3)  # snapshot = 0.3
        improved = 0.3 + CONVERGENCE_DELTA + 0.01
        t = _complete_step(ap, t, plant_confidence=improved)
        t = _complete_restore(ap, t, plant_confidence=improved)
        assert ap.cycles_without_improvement == 0

    def test_stalled_ignores_ticks(self):
        ap = _make()
        t = 0.0
        for _ in range(STALL_CAP):
            t = self._cycle_no_improvement(ap, t)
        _run_dwell(ap, t)
        assert ap.state == PerturbState.STALLED

    def test_stall_issue(self):
        ap = _make()
        t = 0.0
        for _ in range(STALL_CAP):
            t = self._cycle_no_improvement(ap, t)
        issue = ap.get_stall_issue("e1", "heat")
        assert issue is not None
        assert issue[2] == "auto_perturb_stall"

    def test_no_issue_when_not_stalled(self):
        assert _make().get_stall_issue("e1", "heat") is None


# ── perturb_now ──────────────────────────────────────────────────────

class TestForceStart:

    def test_enters_waiting(self):
        ap = _make()
        ap.force_start()
        assert ap.state == PerturbState.WAITING

    def test_starts_on_steady(self):
        ap = _make()
        ap.force_start()
        _enter_step(ap)

    def test_times_out(self):
        ap = _make()
        ap.force_start()
        _tick_unsteady(ap, 0.0)  # initialize timer
        _tick_unsteady(ap, FORCE_TIMEOUT_S + 60.0)
        assert ap.state == PerturbState.IDLE

    def test_clears_stall(self):
        ap = _make()
        ap._state = PerturbState.STALLED
        ap._cycles_without_improvement = STALL_CAP
        ap.force_start()
        assert ap.state == PerturbState.WAITING
        assert ap.cycles_without_improvement == 0

    def test_bypasses_convergence_gate(self):
        ap = _make()
        ap.force_start()
        _enter_step(ap, plant_confidence=0.99)


# ── Time window ──────────────────────────────────────────────────────

class TestTimeWindow:

    def test_no_window(self):
        ap = _make()
        _enter_step(ap, current_hour=14)

    def test_inside_window(self):
        ap = _make(window_start=2, window_end=6)
        _enter_step(ap, current_hour=3)

    def test_outside_window(self):
        ap = _make(window_start=2, window_end=6)
        _run_dwell(ap, 0.0, current_hour=14)
        assert ap.state == PerturbState.IDLE

    def test_midnight_wrap(self):
        ap = _make(window_start=22, window_end=6)
        _enter_step(ap, current_hour=23)

    def test_active_cycle_not_aborted_by_window(self):
        ap = _make(window_start=2, window_end=6)
        t = _enter_step(ap, current_hour=3)
        _tick(ap, t + 300, current_hour=14)  # 5 min later, outside window
        assert ap.state == PerturbState.STEP_ACTIVE


# ── Persistence ──────────────────────────────────────────────────────

class TestPersistence:

    def test_round_trip(self):
        ap = _make()
        ap._cycles_completed = 3
        ap._cycles_without_improvement = 2
        ap._direction = -1.0
        ap._confidence_snapshot = 0.45

        ap2 = _make()
        ap2.restore(ap.as_dict())
        assert ap2.cycles_completed == 3
        assert ap2.cycles_without_improvement == 2
        assert ap2._direction == -1.0

    def test_restores_stall(self):
        ap = _make()
        ap._cycles_without_improvement = STALL_CAP
        ap2 = _make()
        ap2.restore(ap.as_dict())
        assert ap2.state == PerturbState.STALLED

    def test_empty_dict(self):
        ap = _make()
        ap.restore({})
        assert ap.cycles_completed == 0
        assert ap.state == PerturbState.IDLE


# ── PI Controller Integration ────────────────────────────────────────

from tests.conftest import make_pi_config
from tests.test_pi_controller import FakePIEntity


class TestPIControllerIntegration:
    """Verify auto-perturbation wiring in pi_controller.py."""

    def _make_entity(self, **overrides):
        config = make_pi_config({"pi_auto_perturb_enabled": True, **overrides})
        return FakePIEntity(config)

    def test_auto_perturb_composed(self):
        entity = self._make_entity()
        assert entity._pi._auto_perturb.enabled is True

    def test_auto_perturb_disabled_by_default(self):
        entity = FakePIEntity(make_pi_config())
        assert entity._pi._auto_perturb.enabled is False

    def test_research_mode_disabled_by_default(self):
        entity = self._make_entity()
        assert entity._pi._auto_perturb.research_mode is False

    def test_research_mode_plumbed_from_config(self):
        entity = self._make_entity(pi_auto_perturb_research_mode=True)
        assert entity._pi._auto_perturb.research_mode is True

    def test_health_status_includes_perturbation_state(self):
        entity = self._make_entity()
        status = entity._pi.get_health_status()
        assert "auto_perturbation_state" in status
        assert status["auto_perturbation_state"] == "idle"

    def test_perturb_now_method(self):
        entity = self._make_entity()
        pi = entity._pi
        pi.perturb_now()
        assert pi._auto_perturb.state == PerturbState.WAITING

    def test_abort_on_set_temperature(self):
        """User setpoint change aborts active perturbation."""
        entity = self._make_entity()
        pi = entity._pi
        # Force into STEP_ACTIVE
        pi._auto_perturb._state = PerturbState.STEP_ACTIVE
        pi._auto_perturb._direction = 1.0
        assert pi._auto_perturb.offset != 0.0
        # Simulate set_temperature abort
        pi._auto_perturb.abort("user_setpoint_change")
        assert pi._auto_perturb.state == PerturbState.IDLE
        assert pi._auto_perturb.offset == 0.0

    def test_stored_data_round_trip(self):
        """Perturbation counters survive save/restore."""
        entity = self._make_entity()
        pi = entity._pi
        pi._auto_perturb._cycles_completed = 3
        pi._auto_perturb._cycles_without_improvement = 2

        data = pi.get_extra_stored_data()
        assert data is not None
        assert data.auto_perturb_state["cycles_completed"] == 3

        entity2 = self._make_entity()
        entity2._pi.restore_extra_stored_data(data)
        assert entity2._pi._auto_perturb.cycles_completed == 3
        assert entity2._pi._auto_perturb.cycles_without_improvement == 2
