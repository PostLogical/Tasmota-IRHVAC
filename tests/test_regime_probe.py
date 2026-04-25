"""Tests for RegimeProbe state machine.

Unit tests for the pure state machine — state transitions, margin
updates, directional rate checks, cooldown, abort, and persistence.
"""

from __future__ import annotations

import pytest

from custom_components.tasmota_irhvac.pi.regime_probe import (
    BASELINE_MIN_DURATION_S,
    BASELINE_MIN_READINGS,
    CONFIRMED_COOLDOWN_S,
    CONVERGED_COOLDOWN_S,
    INITIAL_COOLDOWN_S,
    MIN_DELTA_BELOW_CURRENT,
    PROBE_MIN_DURATION_S,
    PROBE_MIN_READINGS,
    RATE_CHANGE_THRESHOLD,
    SHRINK_CONFIRMATIONS,
    SHRINK_FACTOR,
    STABILITY_THRESHOLD,
    ProbeState,
    RegimeProbe,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make(**kwargs) -> RegimeProbe:
    return RegimeProbe(enabled=True, **kwargs)


def _tick(rp: RegimeProbe, now: float, **kw):
    defaults = dict(
        now_mono=now,
        room_temp_rate=0.0,
        hp_setpoint=20,
        current_c=20.5,
        min_temp_c=15.0,
        cal_min=-2.0,
        cal_max=2.0,
        is_heating=True,
        is_clamped=False,
        learning_suppressed=False,
        current_hour=12,
        auto_perturb_active=False,
    )
    defaults.update(kw)
    return rp.tick(**defaults)


def _advance_baseline(rp: RegimeProbe, t: float) -> float:
    """Run through baseline phase. Returns time after baseline.

    Assumes BASELINE state has already been entered.  Each tick in
    BASELINE records a reading, so we need MIN_READINGS ticks plus
    enough elapsed time.
    """
    assert rp.state == ProbeState.BASELINE
    for _ in range(BASELINE_MIN_READINGS + 1):
        t += BASELINE_MIN_DURATION_S / BASELINE_MIN_READINGS + 1
        _tick(rp, t)
        if rp.state == ProbeState.PROBE:
            break
    return t


# ── Basic state transitions ──────────────────────────────────────────

class TestStateTransitions:
    def test_starts_idle(self):
        rp = _make()
        assert rp.state == ProbeState.IDLE

    def test_disabled_stays_idle(self):
        rp = RegimeProbe(enabled=False)
        result = _tick(rp, 0.0)
        assert rp.state == ProbeState.IDLE
        assert not result.probe_active

    def test_enters_baseline_when_uncertain(self):
        rp = _make()
        # hp_setpoint=20, current=20.5, cal_min=2.0 → offset=-0.5,
        # |offset|=0.5 < cal_max=1.0 → uncertain
        _tick(rp, 0.0, hp_setpoint=20, current_c=20.5)
        assert rp.state == ProbeState.BASELINE

    def test_stays_idle_when_not_uncertain(self):
        rp = _make()
        # hp_setpoint=25, current=20, offset=5 > cal_min=2.0 → not uncertain
        _tick(rp, 0.0, hp_setpoint=25, current_c=20.0)
        assert rp.state == ProbeState.IDLE

    def test_stays_idle_when_unstable(self):
        rp = _make()
        _tick(rp, 0.0, room_temp_rate=0.05)  # above STABILITY_THRESHOLD
        assert rp.state == ProbeState.IDLE

    def test_stays_idle_when_min_gap_too_small(self):
        rp = _make()
        # current=16, min=15 → gap=1 < MIN_DELTA_BELOW_CURRENT=3
        _tick(rp, 0.0, current_c=16.0, min_temp_c=15.0, hp_setpoint=16)
        assert rp.state == ProbeState.IDLE

    def test_stays_idle_when_auto_perturb_active(self):
        rp = _make()
        _tick(rp, 0.0, auto_perturb_active=True)
        assert rp.state == ProbeState.IDLE

    def test_baseline_to_probe_transition(self):
        rp = _make()
        t = 0.0
        _tick(rp, t)  # enters BASELINE
        assert rp.state == ProbeState.BASELINE

        # Need BASELINE_MIN_READINGS readings AND BASELINE_MIN_DURATION_S elapsed
        for i in range(BASELINE_MIN_READINGS):
            t += BASELINE_MIN_DURATION_S / BASELINE_MIN_READINGS + 1
            result = _tick(rp, t)

        assert rp.state == ProbeState.PROBE
        assert result.probe_active
        assert result.force_min_setpoint

    def test_probe_to_cooldown(self):
        rp = _make()
        t = 0.0
        _tick(rp, t)  # → BASELINE
        t = _advance_baseline(rp, t)

        assert rp.state == ProbeState.PROBE

        # Run probe readings + time
        for i in range(PROBE_MIN_READINGS):
            t += PROBE_MIN_DURATION_S / PROBE_MIN_READINGS + 1
            result = _tick(rp, t)

        assert rp.state == ProbeState.COOLDOWN
        assert not result.probe_active

    def test_cooldown_to_idle(self):
        rp = _make()
        t = 0.0
        _tick(rp, t)
        t = _advance_baseline(rp, t)

        for i in range(PROBE_MIN_READINGS):
            t += PROBE_MIN_DURATION_S / PROBE_MIN_READINGS + 1
            _tick(rp, t)

        assert rp.state == ProbeState.COOLDOWN

        # Advance past initial cooldown
        t += INITIAL_COOLDOWN_S + 1
        _tick(rp, t)
        assert rp.state == ProbeState.IDLE


# ── Uncertainty detection ────────────────────────────────────────────

class TestUncertaintyDetection:
    """delta = current_c - setpoint; uncertain when cal_min <= delta <= cal_max."""

    def test_heating_in_band(self):
        # setpoint=22, current=21, delta=-1.0, band=[-2, 2] → uncertain
        assert RegimeProbe.is_contribution_uncertain(
            hp_setpoint=22, current_c=21.0, cal_min=-2.0,
            cal_max=2.0, is_heating=True,
        )

    def test_heating_definitely_on(self):
        # setpoint=22, current=19, delta=-3.0 < cal_min=-2.0 → definitely on
        assert not RegimeProbe.is_contribution_uncertain(
            hp_setpoint=22, current_c=19.0, cal_min=-2.0,
            cal_max=2.0, is_heating=True,
        )

    def test_heating_definitely_off(self):
        # setpoint=22, current=25, delta=3.0 > cal_max=2.0 → definitely off
        assert not RegimeProbe.is_contribution_uncertain(
            hp_setpoint=22, current_c=25.0, cal_min=-2.0,
            cal_max=2.0, is_heating=True,
        )

    def test_heating_narrowed_band(self):
        # Narrowed band: [-1.5, -0.5], delta=-1.0 → uncertain
        assert RegimeProbe.is_contribution_uncertain(
            hp_setpoint=22, current_c=21.0, cal_min=-1.5,
            cal_max=-0.5, is_heating=True,
        )

    def test_cooling_in_band(self):
        # setpoint=24, current=25, delta=1.0, band=[-2, 2] → uncertain
        assert RegimeProbe.is_contribution_uncertain(
            hp_setpoint=24, current_c=25.0, cal_min=-2.0,
            cal_max=2.0, is_heating=False,
        )

    def test_cooling_definitely_on(self):
        # setpoint=24, current=27, delta=3.0 > cal_max=2.0 → definitely on
        assert not RegimeProbe.is_contribution_uncertain(
            hp_setpoint=24, current_c=27.0, cal_min=-2.0,
            cal_max=2.0, is_heating=False,
        )


# ── Directional rate analysis ────────────────────────────────────────

class TestDirectionalAnalysis:
    def _run_probe_cycle(self, rp, t, baseline_rate, probe_rate, **kw):
        """Run a full probe cycle with specified rates."""
        # Enter baseline
        _tick(rp, t, room_temp_rate=baseline_rate, **kw)
        assert rp.state == ProbeState.BASELINE

        # Baseline readings
        for _ in range(BASELINE_MIN_READINGS):
            t += BASELINE_MIN_DURATION_S / BASELINE_MIN_READINGS + 1
            _tick(rp, t, room_temp_rate=baseline_rate, **kw)

        assert rp.state == ProbeState.PROBE

        # Probe readings
        for _ in range(PROBE_MIN_READINGS):
            t += PROBE_MIN_DURATION_S / PROBE_MIN_READINGS + 1
            _tick(rp, t, room_temp_rate=probe_rate, **kw)

        return t

    def test_heating_hp_was_contributing_rate_decreases(self):
        rp = _make()
        t = self._run_probe_cycle(
            rp, 0.0,
            baseline_rate=0.005,  # slight warming with HP
            probe_rate=0.005 - 0.01,  # cooling after HP removed
        )
        assert rp.state == ProbeState.COOLDOWN
        assert rp.probes_completed == 1

    def test_heating_hp_not_contributing_rate_unchanged(self):
        rp = _make()
        t = self._run_probe_cycle(
            rp, 0.0,
            baseline_rate=0.005,
            probe_rate=0.005,  # no change
        )
        assert rp.state == ProbeState.COOLDOWN
        assert rp.probes_completed == 1


# ── Abort conditions ─────────────────────────────────────────────────

class TestAbort:
    def test_abort_during_baseline(self):
        rp = _make()
        _tick(rp, 0.0)
        assert rp.state == ProbeState.BASELINE
        _tick(rp, 60.0, is_clamped=True)
        assert rp.state == ProbeState.IDLE

    def test_abort_during_probe(self):
        rp = _make()
        t = 0.0
        _tick(rp, t)
        t = _advance_baseline(rp, t)
        assert rp.state == ProbeState.PROBE
        _tick(rp, t + 60, learning_suppressed=True)
        assert rp.state == ProbeState.IDLE

    def test_abort_window_during_baseline(self):
        rp = _make(window_start=8, window_end=20)
        _tick(rp, 0.0, current_hour=12)
        assert rp.state == ProbeState.BASELINE
        # Hour changes to outside window
        _tick(rp, 60.0, current_hour=22)
        assert rp.state == ProbeState.IDLE

    def test_abort_auto_perturb_during_probe(self):
        rp = _make()
        t = 0.0
        _tick(rp, t)
        t = _advance_baseline(rp, t)
        assert rp.state == ProbeState.PROBE
        _tick(rp, t + 60, auto_perturb_active=True)
        assert rp.state == ProbeState.IDLE

    def test_external_abort(self):
        rp = _make()
        _tick(rp, 0.0)
        assert rp.state == ProbeState.BASELINE
        rp.abort("test")
        assert rp.state == ProbeState.IDLE


# ── Margin updates ───────────────────────────────────────────────────

class TestCalibrationUpdates:
    """compute_calibration_updates: evidence shrinks [cal_min, cal_max]."""

    def test_no_change_without_evidence(self):
        rp = _make()
        cal_min, cal_max = rp.compute_calibration_updates(-2.0, 2.0)
        assert cal_min == -2.0
        assert cal_max == 2.0

    def test_shrink_cal_max_requires_confirmations(self):
        rp = _make()
        # "HP not contributing" at delta=0.5 → transition below 0.5 → shrink cal_max
        rp._contribution_evidence_above.append(0.5)
        cal_min, cal_max = rp.compute_calibration_updates(-2.0, 2.0)
        assert cal_max == 2.0  # not enough evidence yet

    def test_shrink_cal_max_with_confirmations(self):
        rp = _make()
        for _ in range(SHRINK_CONFIRMATIONS):
            rp._contribution_evidence_above.append(0.5)
        cal_min, cal_max = rp.compute_calibration_updates(-2.0, 2.0)
        assert cal_max < 2.0  # shrunk
        assert cal_min == -2.0  # min unchanged

    def test_shrink_cal_min_with_confirmations(self):
        rp = _make()
        # "HP was contributing" at delta=-1.0 → transition above -1.0 → shrink cal_min
        for _ in range(SHRINK_CONFIRMATIONS):
            rp._contribution_evidence_below.append(-1.0)
        cal_min, cal_max = rp.compute_calibration_updates(-2.0, 2.0)
        assert cal_min > -2.0  # shrunk
        assert cal_max == 2.0  # max unchanged

    def test_never_widens(self):
        rp = _make()
        rp._no_contribution_count = 10
        cal_min, cal_max = rp.compute_calibration_updates(-2.0, 2.0)
        assert cal_min == -2.0
        assert cal_max == 2.0


# ── Persistence ──────────────────────────────────────────────────────

class TestCoolingMode:
    """Test directional analysis in cooling mode."""

    def test_cooling_hp_contributing_rate_increases(self):
        """In cooling: removing HP (which was cooling) → room warms → rate increases."""
        rp = _make()
        t = 0.0
        # Cooling mode: setpoint=24, current=25, cal_min=2.0
        # offset = 24-25 = -1, |offset|=1 < cal_min → uncertain
        kw = dict(hp_setpoint=24, current_c=25.0, is_heating=False)
        _tick(rp, t, **kw)
        assert rp.state == ProbeState.BASELINE

        for _ in range(BASELINE_MIN_READINGS + 1):
            t += BASELINE_MIN_DURATION_S / BASELINE_MIN_READINGS + 1
            _tick(rp, t, room_temp_rate=-0.005, **kw)
            if rp.state == ProbeState.PROBE:
                break

        # Probe: rate increases (room warms without cooling)
        for _ in range(PROBE_MIN_READINGS + 1):
            t += PROBE_MIN_DURATION_S / PROBE_MIN_READINGS + 1
            _tick(rp, t, room_temp_rate=0.005, **kw)  # +0.01 change
            if rp.state == ProbeState.COOLDOWN:
                break

        assert rp.state == ProbeState.COOLDOWN
        assert rp.probes_completed == 1


class TestCooldownTiers:
    """Test convergence-driven cooldown escalation."""

    def _complete_probe(self, rp, t, **kw):
        """Run a full probe cycle. Returns time after."""
        # Ensure we're in IDLE before starting
        if rp.state == ProbeState.COOLDOWN:
            t += INITIAL_COOLDOWN_S + CONFIRMED_COOLDOWN_S + 1
            _tick(rp, t, **kw)
        assert rp.state == ProbeState.IDLE
        _tick(rp, t, **kw)
        t = _advance_baseline(rp, t)
        for _ in range(PROBE_MIN_READINGS + 1):
            t += PROBE_MIN_DURATION_S / PROBE_MIN_READINGS + 1
            _tick(rp, t, **kw)
            if rp.state == ProbeState.COOLDOWN:
                break
        return t

    def test_initial_cooldown(self):
        rp = _make()
        t = self._complete_probe(rp, 0.0)
        assert rp.state == ProbeState.COOLDOWN
        # Should use INITIAL_COOLDOWN_S from when analysis completed
        _tick(rp, t + INITIAL_COOLDOWN_S - 1)
        assert rp.state == ProbeState.COOLDOWN
        _tick(rp, t + INITIAL_COOLDOWN_S + 1)
        assert rp.state == ProbeState.IDLE

    def test_confirmed_cooldown_after_multiple_probes(self):
        rp = _make()
        t = 0.0
        # Run SHRINK_CONFIRMATIONS probes
        for _ in range(SHRINK_CONFIRMATIONS):
            t = self._complete_probe(rp, t)
            t += INITIAL_COOLDOWN_S + 1
            _tick(rp, t)  # back to IDLE

        # Next probe should use CONFIRMED_COOLDOWN_S
        t = self._complete_probe(rp, t)
        assert rp.state == ProbeState.COOLDOWN
        _tick(rp, t + CONFIRMED_COOLDOWN_S - 1)
        assert rp.state == ProbeState.COOLDOWN
        _tick(rp, t + CONFIRMED_COOLDOWN_S + 1)
        assert rp.state == ProbeState.IDLE

    def test_converged_cooldown(self):
        from custom_components.tasmota_irhvac.pi.regime_probe import CONFIRMATION_THRESHOLD
        rp = _make()
        rp._confirmations_total = CONFIRMATION_THRESHOLD
        rp._probes_completed = 10
        t = self._complete_probe(rp, 0.0)
        assert rp.state == ProbeState.COOLDOWN
        _tick(rp, t + CONVERGED_COOLDOWN_S - 1)
        assert rp.state == ProbeState.COOLDOWN
        _tick(rp, t + CONVERGED_COOLDOWN_S + 1)
        assert rp.state == ProbeState.IDLE


class TestEnabledProperty:
    def test_enabled_true(self):
        rp = _make()
        assert rp.enabled is True

    def test_enabled_false(self):
        rp = RegimeProbe(enabled=False)
        assert rp.enabled is False


class TestPersistence:
    def test_round_trip(self):
        rp = _make()
        rp._probes_completed = 5
        rp._no_contribution_count = 3
        rp._contribution_evidence_above = [1.5, 1.6]
        rp._contribution_evidence_below = [0.7]

        d = rp.as_dict()
        rp2 = _make()
        rp2.restore(d)

        assert rp2._probes_completed == 5
        assert rp2._no_contribution_count == 3
        assert rp2._contribution_evidence_above == [1.5, 1.6]
        assert rp2._contribution_evidence_below == [0.7]

    def test_restore_empty_dict(self):
        rp = _make()
        rp.restore({})
        assert rp._probes_completed == 0
        assert rp._contribution_evidence_above == []


# ── Window ───────────────────────────────────────────────────────────

class TestWindow:
    def test_no_window_always_allowed(self):
        rp = _make()  # no window configured
        _tick(rp, 0.0, current_hour=3)
        assert rp.state == ProbeState.BASELINE  # entered (uncertain zone)

    def test_window_blocks_outside(self):
        rp = _make(window_start=8, window_end=20)
        _tick(rp, 0.0, current_hour=3)
        assert rp.state == ProbeState.IDLE

    def test_window_allows_inside(self):
        rp = _make(window_start=8, window_end=20)
        _tick(rp, 0.0, current_hour=12)
        assert rp.state == ProbeState.BASELINE

    def test_window_wraps_midnight(self):
        rp = _make(window_start=20, window_end=6)
        _tick(rp, 0.0, current_hour=22)
        assert rp.state == ProbeState.BASELINE

        rp2 = _make(window_start=20, window_end=6)
        _tick(rp2, 0.0, current_hour=12)
        assert rp2.state == ProbeState.IDLE
