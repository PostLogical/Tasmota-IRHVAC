"""Tests for CUSUM-driven over-temp regime latch arming (#135).

This is one of several latch-arming triggers (see `LatchArmingTrigger`
enum in `health_checks.py`). The CUSUM trigger wires the existing
CUSUM anomaly detector to the over-temp regime latch via TWO filters:

  1. Sign filter — only fires when the CUSUM arm matching the active mode
     crosses (heat → S-, cool → S+). Mismatched signs would mean room is
     too cold in heat / too hot in cool, where idling HP would actively
     harm comfort. The sign filter is mandatory, not optional.

  2. Overtemp gate — only arms when `overtemp_error > 0` (room genuinely
     above user_desired in heat, or below user_desired in cool). Rejects
     residual-noise events at/below desired where idling HP would be
     wrong.

The two-filter design was validated by `local/tools/_135_path5_fp_audit`
(chronic FPs at perfect FF: 36 unfiltered → 1 with two-filter, 100%
per-disturbance retention).

Contract:
  - `_cusum_armed_this_tick: bool` — transient per-tick flag set in
    `_update_cusum` when sign matches mode AND the config option
    `pi_cusum_overtemp_arming_enabled` is True.
  - Latch-arming block consumes the flag via a new OR-chain term
    `cusum_anomaly = self._cusum_armed_this_tick and overtemp_error > 0`.
  - On consumption (whether True or False), flag clears. There is a
    1-tick latency between CUSUM detection at line ~6114 and latch-block
    consumption at line ~5712 (latch block runs first); cooldown bounds
    arming rate so latency is in the noise vs the 30-min CUSUM cooldown.
  - Event recorded in `_latch_armed_events: list[LatchArmedEvent]` ONLY
    when both filters pass — with `trigger=LatchArmingTrigger.CUSUM_ANOMALY`
    and CUSUM-specific extras (residual, peak_cusum) in `details`. This
    is the generic latch-arming attribution log used for bench analysis;
    other triggers (HP_ESTIMATED_IDLE, MODE_FLIP_OVERTEMP, SUSTAINED_OVERTEMP)
    can backfill emits via the same mechanism (see future_work #136).
"""

from __future__ import annotations

import random
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from homeassistant.components.climate.const import HVACMode

from custom_components.tasmota_irhvac.pi.health_checks import (
    CUSUM_H,
    MIN_RESIDUALS_FOR_DETECTION,
    LatchArmingTrigger,
)
from .conftest import make_pi_config
from .test_pi_controller import FakePIEntity


# ── Helpers ──────────────────────────────────────────────────────────


def _make_cusum_controller(*, cusum_overtemp_arming_enabled: bool = True):
    """Minimal PIController-like object for unit-level CUSUM testing.

    Mirrors `tests/test_anomaly_detection.py::_make_cusum_controller`,
    plus the CUSUM-arming fields. Bypasses __init__ to keep tests focused
    on the CUSUM-side contract.
    """
    from custom_components.tasmota_irhvac.pi.pi_controller import (
        PIController, dt_util,
    )

    ctrl = object.__new__(PIController)
    ctrl._residual_history = deque()
    ctrl._monotonic = lambda: 0.0
    ctrl._utcnow_fn = dt_util.utcnow
    ctrl._cusum_pos = 0.0
    ctrl._cusum_neg = 0.0
    ctrl._anomaly_events = []
    ctrl._exclusion_count = 0
    ctrl._cusum_cooldown_until = None
    ctrl._metrics = MagicMock()
    ctrl._metrics.batch_model_rms = None
    ctrl._pending_events = []
    # CUSUM-arming contract fields (new):
    ctrl._cusum_overtemp_arming_enabled = cusum_overtemp_arming_enabled
    ctrl._cusum_armed_this_tick = False
    ctrl._latch_armed_events = []
    return ctrl


def _feed_residuals(
    ctrl, residuals, *, sigma=0.15, start_mono=1000.0,
    tick_spacing=60.0, is_heating=True, is_cooling=False,
):
    """Feed residuals; calibrate MAD via prefill so CUSUM uses the test σ."""
    random.seed(123)
    prefill_mono = start_mono - tick_spacing
    for _ in range(MIN_RESIDUALS_FOR_DETECTION):
        ctrl._residual_history.append((prefill_mono, random.gauss(0, sigma)))
        prefill_mono -= tick_spacing

    mono = start_mono
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i, r in enumerate(residuals):
        now = base_time + timedelta(seconds=i * tick_spacing)
        ctrl._update_cusum(
            r, mono, is_heating=is_heating, is_cooling=is_cooling, _now=now,
        )
        mono += tick_spacing
    return ctrl


def _make_entity(*, mode: HVACMode = HVACMode.HEAT, outdoor_c: float = 5.0,
                 cusum_overtemp_arming_enabled: bool = True):
    """FakePIEntity with outdoor sensor + CUSUM-arming toggle."""
    overrides = {}
    if not cusum_overtemp_arming_enabled:
        overrides["pi_cusum_overtemp_arming_enabled"] = False
    config = make_pi_config(overrides)
    entity = FakePIEntity(config)
    entity._attr_hvac_mode = mode
    entity._pi._inputs.outdoor_temp = outdoor_c
    return entity


async def _tick(pi, *, mono: float = 1000.0, dt_s: float = 180.0):
    """Standard 3-min production-typical tick (matches overtemp_regime_gate tests)."""
    pi._pi_last_tick_time = mono - dt_s
    pi._monotonic = lambda: mono
    pi._last_setpoint_change_time = mono - 1.0
    await pi._pi_tick()


# ── 1. Sign filter (CUSUM side) ──────────────────────────────────────


class TestCusumArmingSignFilter:
    """`_update_cusum` sets the per-tick flag only when alarm sign matches mode."""

    def test_heat_mode_negative_residual_alarm_sets_flag(self):
        """Heat + S- alarm (negative residual): sign matches → flag set."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        # Sustained -8σ → S- arm crosses H quickly
        _feed_residuals(ctrl, [-8.0 * sigma] * 5, sigma=sigma)
        assert ctrl._cusum_armed_this_tick is True, (
            "Heat-mode S- alarm should set per-tick flag for downstream "
            "latch arming"
        )

    def test_heat_mode_positive_residual_alarm_does_not_set_flag(self):
        """Heat + S+ alarm (positive residual): wrong sign → flag NOT set.

        Idling HP on positive residual in heat mode would freeze the user
        when the system is fighting a cold leak. Sign filter is mandatory.
        """
        ctrl = _make_cusum_controller()
        sigma = 0.15
        _feed_residuals(ctrl, [+8.0 * sigma] * 5, sigma=sigma)
        # Sanity: anomaly was detected at all
        assert len(ctrl._anomaly_events) >= 1, (
            "CUSUM should still fire on positive-residual alarm — the sign "
            "filter is at the CUSUM-arming layer, not at the CUSUM layer"
        )
        assert ctrl._cusum_armed_this_tick is False, (
            "Heat-mode S+ alarm must NOT set the flag — wrong sign"
        )

    def test_cool_mode_positive_residual_alarm_sets_flag(self):
        """Cool + S+ alarm (positive residual): sign matches → flag set.

        Cool mode: hp_setpoint typically BELOW desired (negative
        residual is the steady state). A POSITIVE residual means the
        controller is over-commanding (less cooling than expected),
        which happens when room is unexpectedly cool — idle HP is
        correct (stop cooling).
        """
        ctrl = _make_cusum_controller()
        sigma = 0.15
        _feed_residuals(
            ctrl, [+8.0 * sigma] * 5, sigma=sigma,
            is_heating=False, is_cooling=True,
        )
        assert ctrl._cusum_armed_this_tick is True

    def test_cool_mode_negative_residual_alarm_does_not_set_flag(self):
        """Cool + S- alarm (negative residual): wrong sign → flag NOT set.

        Negative residual in cool mode means over-cooling commanded —
        room is unexpectedly warm. Idling HP would let it warm more.
        """
        ctrl = _make_cusum_controller()
        sigma = 0.15
        _feed_residuals(
            ctrl, [-8.0 * sigma] * 5, sigma=sigma,
            is_heating=False, is_cooling=True,
        )
        assert len(ctrl._anomaly_events) >= 1, "CUSUM should fire"
        assert ctrl._cusum_armed_this_tick is False, (
            "Cool-mode S- alarm must NOT set the flag — wrong sign"
        )

    def test_off_mode_alarm_does_not_set_flag(self):
        """Neither heat nor cool active → no CUSUM arming regardless of sign."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        _feed_residuals(
            ctrl, [-8.0 * sigma] * 5, sigma=sigma,
            is_heating=False, is_cooling=False,
        )
        assert ctrl._cusum_armed_this_tick is False, (
            "OFF mode (no heat/cool active) must not arm the latch via CUSUM"
        )


# ── 2. Kill switch ───────────────────────────────────────────────────


class TestCusumArmingKillSwitch:
    """CUSUM arming inert when `pi_cusum_overtemp_arming_enabled` is False."""

    def test_flag_never_set_when_disabled(self):
        """Sign-matched CUSUM alarm + arming disabled → flag stays False."""
        ctrl = _make_cusum_controller(cusum_overtemp_arming_enabled=False)
        sigma = 0.15
        _feed_residuals(ctrl, [-8.0 * sigma] * 5, sigma=sigma)
        assert len(ctrl._anomaly_events) >= 1, (
            "CUSUM and HA Repairs path still fire when CUSUM arming disabled "
            "— only the latch-arming side is gated"
        )
        assert ctrl._cusum_armed_this_tick is False, (
            "Kill switch off → no CUSUM latch arming"
        )


# ── 3. Overtemp gate + latch arming (full-tick path) ─────────────────


class TestCusumArmingOvertempGate:
    """Latch-arming block applies the overtemp_error > 0 gate.

    These tests directly set `_cusum_armed_this_tick` to True and run a
    tick — bypassing the CUSUM side to isolate the latch-arming contract.
    """

    @pytest.mark.asyncio
    async def test_arms_latch_when_overtemp_positive(self):
        """Flag set + room above user_desired → latch arms via CUSUM trigger."""
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        # Room above user_desired (overtemp_error = +0.5 > 0)
        entity._attr_current_temperature = 22.5
        # HP commanded HIGH (delta < 0 → HP_ESTIMATED_IDLE does NOT fire);
        # no mode flip, no sustained timer → other triggers inert.
        pi._hp_setpoint = 25
        # Pre-set the flag (simulating CUSUM having fired previous tick)
        pi._cusum_armed_this_tick = True

        assert pi._uncontrollable_entry_latch is False
        await _tick(pi)
        assert pi._uncontrollable_entry_latch is True, (
            "CUSUM flag + overtemp>0 should arm the latch via the OR-chain"
        )

    @pytest.mark.asyncio
    async def test_does_not_arm_latch_when_overtemp_negative(self):
        """Flag set + room BELOW user_desired → latch does NOT arm.

        This is the audit's key finding: chronic FPs with negative residuals
        cluster in mild under-temp. Idling HP under-temp would freeze the
        user. The overtemp>0 gate prevents this.
        """
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        # Room BELOW user_desired (overtemp_error = -0.3 < 0)
        entity._attr_current_temperature = 21.7
        pi._hp_setpoint = 25
        pi._cusum_armed_this_tick = True

        await _tick(pi)
        assert pi._uncontrollable_entry_latch is False, (
            "CUSUM trigger must NOT arm latch when room is below "
            "user_desired — overtemp gate rejects (audit-confirmed FP pattern)"
        )

    @pytest.mark.asyncio
    async def test_does_not_arm_latch_when_at_user_desired(self):
        """At-desired (overtemp_error == 0) → does NOT arm. Gate is strict >."""
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        entity._attr_current_temperature = 22.0  # exactly at desired
        pi._hp_setpoint = 25
        pi._cusum_armed_this_tick = True

        await _tick(pi)
        assert pi._uncontrollable_entry_latch is False, (
            "overtemp_error == 0 is not > 0; gate is strict"
        )

    @pytest.mark.asyncio
    async def test_overtemp_error_uses_user_desired_not_effective(self):
        """Gate references user_desired (occupant setpoint), not qref-biased.

        Mirrors existing comfort-regime contract (Bug 1 fix). With qref
        biasing effective_desired DOWN by 0.3°C, a room at 21.9°C is:
          - 0.2 BELOW user_desired (22.0) → overtemp_error = -0.1 < 0
          - 0.1 ABOVE effective_desired (21.7) → would arm if keyed off effective
        CUSUM trigger must use user_desired and reject.
        """
        from unittest.mock import patch
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        pi._supervisor_enabled = True
        pi._supervisor_kind = "qref"
        pi._last_raw_setpoint = 24.0
        entity._attr_current_temperature = 21.9
        pi._hp_setpoint = 25
        pi._cusum_armed_this_tick = True

        with patch.object(pi._qref_biaser, "update", return_value=-0.3):
            await _tick(pi)
        assert pi._uncontrollable_entry_latch is False, (
            "overtemp_error must reference user_desired, not effective_desired"
        )

    @pytest.mark.asyncio
    async def test_flag_cleared_after_tick(self):
        """The per-tick flag is transient — clears every tick regardless of arming."""
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        entity._attr_current_temperature = 22.5
        pi._hp_setpoint = 25
        pi._cusum_armed_this_tick = True

        await _tick(pi)
        assert pi._cusum_armed_this_tick is False, (
            "Flag must clear after each tick (consumed or not) so a stale "
            "True doesn't carry across ticks"
        )


# ── 4. Latch arming integration (doesn't disturb other triggers) ─────


class TestCusumArmingLatchIntegration:
    """CUSUM is a new OR-chain trigger — must not interfere with existing ones."""

    @pytest.mark.asyncio
    async def test_cusum_does_not_disturb_sustained_overtemp_counter(self):
        """CUSUM trigger MUST NOT reset the SUSTAINED_OVERTEMP counter.

        The two triggers are independent arming sources for the same latch;
        each manages its own state.
        """
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._cal_midpoint_warmup_pending = False
        entity._attr_current_temperature = 21.6  # > 1.5°C overtemp
        pi._hp_setpoint = 25

        # Accumulate 5 ticks of SUSTAINED_OVERTEMP counter
        for i in range(5):
            await _tick(pi, mono=1000.0 + i * 180.0)
        assert pi._sustained_overtemp_minutes == 15.0

        # Fire CUSUM trigger mid-stream — must NOT reset the counter
        pi._cusum_armed_this_tick = True
        await _tick(pi, mono=1000.0 + 5 * 180.0)
        assert pi._sustained_overtemp_minutes == 18.0, (
            f"SUSTAINED_OVERTEMP counter must keep accumulating: got "
            f"{pi._sustained_overtemp_minutes}, expected 18.0"
        )

    @pytest.mark.asyncio
    async def test_hp_estimated_idle_still_fires_with_cusum_disabled(self):
        """Disabling CUSUM arming must not affect other triggers (kill switch is local)."""
        entity = _make_entity(cusum_overtemp_arming_enabled=False)
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        # HP_ESTIMATED_IDLE condition: delta > cal_midpoint (=0) → hp_estimated_active=False
        entity._attr_current_temperature = 24.0
        pi._hp_setpoint = 22

        await _tick(pi)
        assert pi._uncontrollable_entry_latch is True, (
            "HP_ESTIMATED_IDLE trigger must still fire when CUSUM arming disabled"
        )


# ── 5. Instrumentation (events recorded for attribution) ─────────────


class TestCusumArmingInstrumentation:
    """Events recorded ONLY when both filters pass — drive bench attribution."""

    @pytest.mark.asyncio
    async def test_event_recorded_when_both_filters_pass(self):
        """Sign + overtemp both pass → LatchArmedEvent appended with CUSUM context."""
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        entity._attr_current_temperature = 22.5  # +0.5°C overtemp
        pi._hp_setpoint = 25
        # Simulate residual context the CUSUM side would set up
        pi._last_residual = -0.5
        pi._cusum_armed_this_tick = True

        assert pi._latch_armed_events == []
        await _tick(pi)
        assert len(pi._latch_armed_events) == 1, (
            "One arming should record exactly one event"
        )
        event = pi._latch_armed_events[0]
        assert event.trigger == LatchArmingTrigger.CUSUM_ANOMALY, (
            f"trigger should be CUSUM_ANOMALY, got {event.trigger}"
        )
        assert event.mode == "heat"
        assert event.overtemp_error == pytest.approx(0.5, abs=0.01)
        # Residual sign should be preserved (negative for heat-mode arming)
        assert event.details["residual"] < 0

    @pytest.mark.asyncio
    async def test_no_event_when_overtemp_gate_rejects(self):
        """Flag set but overtemp gate rejects → no event recorded.

        Distinguishes 'CUSUM alarmed with matching sign' (recorded in
        `_anomaly_events`) from 'CUSUM actually armed the latch'
        (recorded in `_latch_armed_events`). Bench attribution uses the
        latter.
        """
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        entity._attr_current_temperature = 21.7  # below desired
        pi._hp_setpoint = 25
        pi._cusum_armed_this_tick = True

        await _tick(pi)
        assert pi._latch_armed_events == [], (
            "No event when overtemp gate rejects — keeps the attribution "
            "log clean (only actionable arms)"
        )

    @pytest.mark.asyncio
    async def test_events_accumulate_across_multiple_arms(self):
        """Each successful arm appends a new event."""
        entity = _make_entity()
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._cal_midpoint_warmup_pending = False
        entity._attr_current_temperature = 22.5
        pi._hp_setpoint = 25

        for i in range(3):
            pi._cusum_armed_this_tick = True
            # Reset latch each iteration so OR-chain isn't masked by persisted state
            pi._uncontrollable_entry_latch = False
            await _tick(pi, mono=1000.0 + i * 180.0)

        assert len(pi._latch_armed_events) == 3
        assert all(
            e.trigger == LatchArmingTrigger.CUSUM_ANOMALY
            for e in pi._latch_armed_events
        )
