"""Unit tests for the over-temperature regime gate (physical-state anti-windup).

Tests the gate's contract directly via FakePIEntity + single ticks, no physics
simulation.  See `tests/test_overtemp_regime_full_stack.py` for the synthetic
solar-pulse scenario and `tests/test_overtemp_regime_replay.py` for the
bundle-replay validation.

Gate contract:
  Latch: set when `hp_estimated_active=False` (existing cal_midpoint gate
         already considers HP estimated-inactive); reset when room returns to
         or below desired AND existing gate not firing.
  Enter: regime activates when `overtemp_error > ENTER_THRESHOLD` AND latch is
         set (latch precondition excludes cold-snap brief overshoot).
  Exit:  regime deactivates when `overtemp_error < EXIT_THRESHOLD`.  No
         precondition on exit — temperature recovery is the only release.
  Action: while active, HP forced to idle setpoint (min in heat / max in
         cool), `hp_observation_usable=False`, `hp_estimated_active=False`
         which in turn freezes the integrator via `skip_integration`.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from homeassistant.components.climate.const import HVACMode

from .conftest import make_pi_config
from .test_pi_controller import FakePIEntity


def _make_entity(*, mode: HVACMode = HVACMode.HEAT, outdoor_c: float = 5.0):
    """FakePIEntity with outdoor sensor seeded so ticks pass data_complete."""
    config = make_pi_config()
    entity = FakePIEntity(config)
    entity._attr_hvac_mode = mode
    entity._pi._inputs.outdoor_temp = outdoor_c
    return entity


async def _tick(pi, *, mono: float = 1000.0, dt_s: float = 180.0):
    """Default dt 180s = 3 min — production-typical tick cadence (sensor-driven
    with ~60-90s cooldown nominally, 3 min as a representative round number).
    Tests that need a 15-min cadence pass dt_s=900.0 explicitly (e.g. batch
    WLS, lag-tau, steady-state convergence)."""
    pi._pi_last_tick_time = mono - dt_s
    pi._monotonic = lambda: mono
    # Activate setpoint hold so the manually-fixtured `hp_setpoint` isn't
    # overwritten by PI's per-tick math during these state-machine tests.
    # Tests that probe overtemp_regime / cal_midpoint hysteresis fix
    # `hp_setpoint` to set up a specific delta-vs-cal_midpoint relationship;
    # without an active hold, PI's normal output (large FF + integral) would
    # rewrite the setpoint mid-tick and shift the delta away from the
    # state-machine boundary the test is exercising.
    pi._last_setpoint_change_time = mono - 1.0
    await pi._pi_tick()


def _warm_rate_window(
    pi, current_c: float, target_rate: float = 0.0, mono: float = 1000.0
) -> None:
    """Pre-populate `_room_temp_history` so the rate-validity guard passes
    on the very next tick AND the tick's rate recomputation yields
    approximately `target_rate` °C/min.

    The rate is computed during the tick at line ~5193 as
    `(history[-1] − history[0]) / elapsed_min`.  After the tick appends the
    current reading at monotonic time `mono`, history[0] will be our second
    warmup entry (after the pop) and history[-1] will be the new reading.
    To make the recomputed rate equal `target_rate`, set the warmup entries
    to `current_c − target_rate × (mono − 1) / 60`.

    Tests that need a specific rate at the gate's evaluation moment must
    pass both `current_c` (matching the entity's `_attr_current_temperature`)
    and the `mono` they'll use in `_tick`.  See
    `test_regime_gate_does_not_fire_until_rate_history_full` for the
    contract being bypassed."""
    elapsed_min = (mono - 1.0) / 60.0
    warm_temp = current_c - target_rate * elapsed_min
    pi._room_temp_history = [(float(i), warm_temp) for i in range(5)]


# ── Latch behavior ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_latch_sets_when_existing_gate_fires():
    """Latch sets when delta crosses cal_midpoint (existing gate fires)."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Bypass the first-tick cal_midpoint warmup guard — this test exercises
    # post-warmup steady-state latch behavior, not the warmup itself.  See
    # `test_cal_midpoint_warmup_skips_first_tick` for the warmup contract.
    pi._cal_midpoint_warmup_pending = False
    # Room above setpoint by enough that delta > cal_midpoint(=0):
    #   delta = current - setpoint = 24 - 22 = +2 > 0 → hp_estimated_active=False
    entity._attr_current_temperature = 24.0
    pi._hp_setpoint = 22

    assert pi._uncontrollable_entry_latch is False
    await _tick(pi)
    assert pi._uncontrollable_entry_latch is True


@pytest.mark.asyncio
async def test_latch_stays_unset_when_existing_gate_not_firing():
    """Cold-snap analog: room over-temp but HP commanded high → existing gate
    not firing → latch stays False → my gate dormant."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Room over-temp (24°C, 2°C above desired) BUT HP commanded high.
    # delta = 24 - 27 = -3 < cal_midpoint=0 → hp_estimated_active=True
    entity._attr_current_temperature = 24.0
    pi._hp_setpoint = 27

    await _tick(pi)
    assert pi._uncontrollable_entry_latch is False
    assert pi._overtemp_regime is False


@pytest.mark.asyncio
async def test_latch_resets_when_room_returns_to_desired():
    """Latch should reset only when room ≤ desired AND existing gate not firing."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True
    # Room back at desired, HP commanded above (delta < 0 → existing gate
    # not firing).  overtemp_error = 22 - 22 = 0 ≤ 0 → reset.
    entity._attr_current_temperature = 22.0
    pi._hp_setpoint = 24

    await _tick(pi)
    assert pi._uncontrollable_entry_latch is False


@pytest.mark.asyncio
async def test_setpoint_down_handled_by_existing_bumpless_not_latch():
    """Setpoint-down should be handled by the existing Åström-Hägglund
    bumpless transfer in ``set_temperature`` / ``on_remote_change`` (which
    zeros the integral on |Δ|>2°C and applies the P-cancel formula on
    smaller deltas), NOT by arming the regime gate's latch.

    Regression guard: an earlier attempt added a "Path 2-new" event-driven
    latch arming on setpoint-down ≥1°C, which forced the regime to engage
    and HP to MIN on the same tick.  In cold weather with seed-over FF,
    thermal momentum then overshot the room below the new desired by ~3°F
    before the controller could respond.  The bench's setpoint_down_1_5_drafty
    regression (rollup_itae 8048→34558, cold_max_undershoot 0→1.71°C, final
    20.22→20.94) made this clear.  The fix was to delete Path 2-new entirely
    and rely on the bumpless transfer that production-path setpoint changes
    already apply (`pi_controller.py:2168`).
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Tick 1: baseline (no event)
    entity._attr_current_temperature = 22.0
    pi._hp_setpoint = 25
    await _tick(pi, mono=1000.0)
    assert pi._uncontrollable_entry_latch is False

    # Tick 2: directly lower desired by 2°C — simulating the harness path
    # (the production set_temperature would also zero the integral here as
    # bumpless transfer; this test is just asserting the latch is NOT armed
    # by the setpoint change alone).
    pi._desired_temp = 20.0
    pi._hp_setpoint = 25
    await _tick(pi, mono=2000.0)
    assert pi._uncontrollable_entry_latch is False, (
        "Setpoint-down should not arm the latch — existing bumpless transfer "
        "handles it.  See pi_controller.py:2168."
    )


@pytest.mark.asyncio
async def test_latch_arms_on_mode_flip_to_heat_with_overtemp():
    """Path 3-new: user turns on HEAT mode in a room already above desired
    → latch arms via the mode-flip event path even when HP is commanded
    high (existing Path 1 would not fire on the same tick).
    """
    # Start in OFF mode so the flip to HEAT is a real mode change
    entity = _make_entity(mode=HVACMode.OFF)
    pi = entity._pi
    pi._desired_temp = 20.0
    # Tick 1: mode is OFF → passive tick path; _prev_hvac_mode initialized.
    entity._attr_current_temperature = 23.0
    pi._hp_setpoint = 25
    await _tick(pi, mono=1000.0)
    # Latch state during OFF is implementation-defined; force False before flip.
    pi._uncontrollable_entry_latch = False

    # Tick 2: user flips to HEAT mode.  Room (23.0) is 3°C over desired (20).
    entity._attr_hvac_mode = HVACMode.HEAT
    pi._hp_setpoint = 25
    await _tick(pi, mono=2000.0)
    assert pi._uncontrollable_entry_latch is True


@pytest.mark.asyncio
async def test_latch_does_not_arm_on_mode_flip_to_heat_without_overtemp():
    """Mode flip to HEAT when room is at/below desired: event path requires
    `overtemp_error > 0` and does not fire.  This is the normal cold-start
    case — user turning on heat in a cool room.
    """
    entity = _make_entity(mode=HVACMode.OFF)
    pi = entity._pi
    pi._desired_temp = 20.0
    entity._attr_current_temperature = 18.0
    pi._hp_setpoint = 25
    await _tick(pi, mono=1000.0)
    pi._uncontrollable_entry_latch = False

    # Flip to HEAT; room (18) is below desired (20).
    entity._attr_hvac_mode = HVACMode.HEAT
    pi._hp_setpoint = 25
    await _tick(pi, mono=2000.0)
    assert pi._uncontrollable_entry_latch is False


@pytest.mark.asyncio
async def test_path4_arms_after_sustained_overtemp_30min():
    """Path 4: when overtemp_error stays above 1.5°C continuously for 30 min,
    the latch arms even when HP is still commanded high (so Path 1 wouldn't
    fire) and there's no setpoint/mode event (so Paths 2/3 wouldn't fire).
    Threshold + duration grounded in path4_threshold_sweep (2026-05-29).
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 20.0
    pi._cal_midpoint_warmup_pending = False
    # Room sustained at 1.6°C over desired (just above 1.5°C threshold),
    # HP commanded above room so Path 1 doesn't fire.
    entity._attr_current_temperature = 21.6
    pi._hp_setpoint = 25

    # 9 ticks × 3 min = 27 min (under threshold)
    for i in range(9):
        await _tick(pi, mono=1000.0 + i * 180.0)
        assert pi._uncontrollable_entry_latch is False, (
            f"latch armed too early at tick {i+1} "
            f"(elapsed {(i+1)*3} min, threshold 30 min)"
        )

    # 10th tick at minute 30 → arms
    await _tick(pi, mono=1000.0 + 9 * 180.0)
    assert pi._uncontrollable_entry_latch is True


@pytest.mark.asyncio
async def test_path4_does_not_arm_if_overtemp_drops_below_threshold():
    """Path 4 counter resets when overtemp drops at-or-below the threshold,
    so a brief dip mid-episode prevents arming.  This is the intended
    behavior: the path measures *continuous* sustained overtemp, not
    cumulative.

    Asserts on `_sustained_overtemp_minutes` directly (the Path 4 counter),
    not on `_uncontrollable_entry_latch`, because other latch paths
    (regime_probe `force_min_setpoint`, cal_midpoint saturation) may fire
    incidentally at the production-default Ki=0.70.  This test is
    specifically about Path 4's continuous-vs-cumulative semantics.
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 20.0
    pi._cal_midpoint_warmup_pending = False
    entity._attr_current_temperature = 21.6   # 1.6°C over → above threshold
    pi._hp_setpoint = 25

    # 5 ticks sustained → counter at 15 min, below 30 min arm threshold
    for i in range(5):
        await _tick(pi, mono=1000.0 + i * 180.0)
    assert pi._sustained_overtemp_minutes == 15.0, (
        f"After 5 ticks of 1.6°C overtemp: counter={pi._sustained_overtemp_minutes}"
    )

    # Tick at minute 15: dip below threshold (room cools to 21.3°C = +1.3,
    # below the 1.5°C threshold) → counter resets
    entity._attr_current_temperature = 21.3
    await _tick(pi, mono=1000.0 + 5 * 180.0)
    assert pi._sustained_overtemp_minutes == 0.0, (
        f"Counter should reset on dip below threshold: {pi._sustained_overtemp_minutes}"
    )

    # Resume overtemp.  Counter resets, needs 30 more minutes to arm.
    entity._attr_current_temperature = 21.6
    # 9 more ticks (27 min from reset) — still under threshold
    for i in range(9):
        await _tick(pi, mono=1000.0 + (6 + i) * 180.0)
    assert pi._sustained_overtemp_minutes == 27.0, (
        f"After dip + 9 ticks of 1.6°C: counter={pi._sustained_overtemp_minutes}"
    )


@pytest.mark.asyncio
async def test_path4_does_not_arm_below_threshold():
    """Cooking-tier scenario analog: room sustained above desired but ≤1.5°C
    (peak 1.18°C is the empirical cooking signature from path4_threshold_sweep).
    Path 4 explicitly does NOT engage in this regime — its counter only
    accumulates above 1.5°C.

    The cooking-tier *gap* (Path 4 missing this case) is now closed in
    production by the CUSUM_ANOMALY trigger (#135), which can arm the
    latch on sub-1.5°C overtemp episodes via sign-matched CUSUM events.
    This test isolates Path 4's contract by asserting on
    `_sustained_overtemp_minutes` directly (the Path 4 state), not on
    the latch — the latch may be armed by CUSUM_ANOMALY in this scenario
    (correct production behavior), but Path 4's own counter must stay 0.
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 20.0
    pi._cal_midpoint_warmup_pending = False
    # Sustained at 1.2°C over — typical cooking-tier peak (below 1.5°C)
    entity._attr_current_temperature = 21.2
    pi._hp_setpoint = 25

    # 20 ticks = 60 min — well past Path 4's 30-min threshold, but
    # overtemp is below 1.5°C threshold so counter never increments.
    for i in range(20):
        await _tick(pi, mono=1000.0 + i * 180.0)
        assert pi._sustained_overtemp_minutes == 0.0, (
            f"Path 4 counter incorrectly accumulated on sub-threshold "
            f"overtemp at tick {i+1}: {pi._sustained_overtemp_minutes}"
        )


@pytest.mark.asyncio
async def test_path4_counter_resets_on_mode_change():
    """Mode change clears the latch + Path 4's accumulated time, since the
    over-temp concept is direction-aware (heat vs cool)."""
    from homeassistant.components.climate.const import HVACMode
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 20.0
    pi._cal_midpoint_warmup_pending = False
    entity._attr_current_temperature = 21.6
    pi._hp_setpoint = 25

    # Accumulate 20 min of Path 4 counter
    for i in range(7):
        await _tick(pi, mono=1000.0 + i * 180.0)
    assert pi._sustained_overtemp_minutes > 15.0
    assert pi._uncontrollable_entry_latch is False

    # Flip mode → Path 4 counter resets
    entity._attr_hvac_mode = HVACMode.COOL
    await _tick(pi, mono=1000.0 + 7 * 180.0)
    assert pi._sustained_overtemp_minutes == 0.0


@pytest.mark.asyncio
async def test_latch_persists_during_overtemp_episode():
    """Latch persists across ticks where existing gate transiently unfreezes
    (FF nudges setpoint above current) — that's the whole point: my gate
    can fire in those moments because we've ENTERED uncontrollable state."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Bypass the first-tick warmup guard — post-warmup latch dynamics.
    pi._cal_midpoint_warmup_pending = False
    # Tick 1: existing gate fires → latch sets
    entity._attr_current_temperature = 25.0
    pi._hp_setpoint = 22
    await _tick(pi, mono=1000.0)
    assert pi._uncontrollable_entry_latch is True
    # Tick 2: FF nudges setpoint above current, existing gate would unfreeze,
    # but room still over-temp → latch persists, my regime still active.
    pi._hp_setpoint = 27  # FF aggressive
    entity._attr_current_temperature = 25.0
    await _tick(pi, mono=2000.0)
    assert pi._uncontrollable_entry_latch is True


# ── Regime entry / exit ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_regime_enters_when_latched_and_room_not_recovering():
    """Positive case: latch armed AND room not actively cooling
    (`room_temp_rate ≥ −0.02 °C/min`) → regime engages.  Rate-based redesign
    (#108): no absolute temperature threshold."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True       # latch pre-armed
    entity._attr_current_temperature = 22.8     # warm; room above desired
    pi._hp_setpoint = 22                        # any value (regime entry no
                                                # longer checks delta directly;
                                                # the latch is the proxy)
    _warm_rate_window(pi, current_c=22.8, target_rate=0.05)  # rising at +0.05

    await _tick(pi)
    assert pi._overtemp_regime is True


@pytest.mark.asyncio
async def test_regime_does_not_enter_when_room_actively_cooling():
    """Latch armed (HP gauged inactive) BUT the room is recovering on its own
    (`room_temp_rate < −0.02 °C/min`) → regime stays dormant.  Latch +
    integration freeze are sufficient; regime intervention isn't needed.
    Rate-based redesign (#108): replaces the old "below 1.0 °C above desired"
    threshold check, which silently missed slow steady rises."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True       # latch pre-armed
    pi._room_temp_rate = -0.05                  # cooling at meaningful pace
    entity._attr_current_temperature = 23.0     # still warm — but recovering
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._overtemp_regime is False


@pytest.mark.asyncio
async def test_regime_enters_on_slow_steady_rise():
    """The case the old `> 1.0` threshold silently missed: a slow steady rise
    well below any positive-side threshold but cumulatively warming the room
    over hours.  With rate-based entry the gate fires once latch arms, even
    when rate is only barely above zero."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 22.2     # only 0.2 °C above desired
    pi._hp_setpoint = 22
    _warm_rate_window(pi, current_c=22.2, target_rate=0.005)  # +0.005 °C/min ~0.3 °C/hour

    await _tick(pi)
    assert pi._overtemp_regime is True, (
        "Slow steady rise must engage regime — the legacy `> 1.0 °C` "
        "threshold would have silently missed this."
    )


@pytest.mark.asyncio
async def test_regime_does_not_enter_without_latch():
    """Even with the room rising, no latch means the HP is still gauged active
    (e.g., cold-snap ff overshoot where HP commanded high — should NOT be kicked
    out)."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = False      # latch NOT armed
    pi._room_temp_rate = 0.05                   # rising — but HP is in control
    entity._attr_current_temperature = 23.5     # 1.5 °C over (well above legacy threshold)
    pi._hp_setpoint = 28                        # delta=+4.5 → hp gauged active
    pi._hp_estimated_active_state = True

    await _tick(pi)
    assert pi._overtemp_regime is False, (
        "Cold-snap ff overshoot (HP active) must NOT trip the regime — "
        "latch precondition guards this."
    )


# ── Comfort regime references the OCCUPANT setpoint, not qref-effective ──
# Bug 1 (shipped previously): the regime's `overtemp_error` is computed against
# `user_desired_c`, not the qref-biased effective desired.  The rate-based
# redesign (#108) keeps this — entry no longer uses overtemp_error at all (it's
# rate-based), but EXIT and latch-reset still reference overtemp_error against
# user_desired.  See project_qref_overtemp_bumpless_bugs.


@pytest.mark.asyncio
async def test_regime_exit_references_user_desired_not_effective():
    """Bug 1 preservation: regime exit fires when room reaches user_desired,
    not when it reaches the qref-biased effective_desired.  With qref biasing
    effective down by 0.3 °C, an exit keyed off effective would fire 0.3 °C
    early (room still above user_desired)."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    pi._supervisor_enabled = True
    pi._supervisor_kind = "qref"
    pi._last_raw_setpoint = 24.0
    # Room 21.8: 0.2 below user, but 0.1 ABOVE effective (21.7 with -0.3 bias).
    # If exit referenced effective, it would NOT exit yet (overtemp_error_eff
    # = 0.1 > 0).  Against user, overtemp_error_user = -0.2 ≤ 0 → exits.
    entity._attr_current_temperature = 21.8
    pi._hp_setpoint = 22
    with patch.object(pi._qref_biaser, "update", return_value=-0.3):
        await _tick(pi)
    assert pi._overtemp_regime is False, (
        "exit must reference user_desired (Bug 1 fix) — room at user_desired-0.2 "
        "should release regardless of the qref-biased effective reference"
    )


@pytest.mark.asyncio
async def test_regime_exits_when_room_returns_to_desired():
    """Regime exits when overtemp_error ≤ 0 (room ≤ user_desired).  Rate-based
    redesign (#108) drops the +0.5 hysteresis on the exit side — both the
    regime exit and the latch reset now share the SAME event (room reaches
    desired), unifying what were two related conditions."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 22.0    # exactly at desired → error=0
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._overtemp_regime is False


@pytest.mark.asyncio
async def test_regime_stays_active_above_desired():
    """Regime stays active while room is above user_desired, even at small
    positive overtemp_error (no +0.5 exit buffer to release early)."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 22.4    # 0.4 above desired → error=0.4>0
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._overtemp_regime is True, (
        "no +0.5 hysteresis on exit — must stay active until room ≤ desired"
    )


# ── Action when active ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_active_regime_freezes_integrator_in_heat():
    """Integrator must not change while regime is active, even with error."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    pi._pi_integral = 5.0
    entity._attr_current_temperature = 24.0    # error = -2
    pi._hp_setpoint = 22

    initial_i = pi._pi_integral
    await _tick(pi)
    # Integrator may leak by tiny amount (0.9999^dt_factor), but should not
    # change meaningfully — definitely not by the avg_error * dt that would
    # otherwise wind it negative.
    assert abs(pi._pi_integral - initial_i) < 0.01


@pytest.mark.asyncio
async def test_active_regime_forces_hp_to_min_in_heat():
    """In heat mode, HP setpoint forced to vendor min when regime active."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 24.0
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._hp_setpoint == int(pi._min_temp_c)


@pytest.mark.asyncio
async def test_active_regime_forces_hp_to_max_in_cool():
    """In cool mode, HP setpoint forced to vendor max when regime active.
    Cool mode: low setpoint = aggressive cool, high setpoint = idle."""
    entity = _make_entity(mode=HVACMode.COOL, outdoor_c=30.0)
    pi = entity._pi
    pi._desired_temp = 22.0
    # Trigger regime in cool: room < desired - ENTER
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 19.0    # 3°C below desired
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._hp_setpoint == int(pi._max_temp_c)


@pytest.mark.asyncio
async def test_active_regime_excludes_wls_observation():
    """`hp_observation_usable=False` when regime active → WLS buffer should
    not admit observations from this tick.  We verify by checking the
    observation count doesn't grow."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 24.0
    pi._hp_setpoint = 22

    initial_count = len(pi._observation_buffer_heat)
    await _tick(pi)
    assert len(pi._observation_buffer_heat) == initial_count


# ── Cool mode symmetry ──────────────────────────────────────────────


# ── Mode change + restart resets ────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_change_clears_regime_and_latch():
    """Heat → cool with regime active in heat: regime + latch reset on mode
    change so cool-mode evaluation starts fresh under cool semantics."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Engage regime in heat: latch pre-armed + rate not cooling
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 24.0
    pi._hp_setpoint = 22
    _warm_rate_window(pi, current_c=24.0, target_rate=0.05)
    await _tick(pi, mono=1000.0)
    assert pi._overtemp_regime is True
    assert pi._uncontrollable_entry_latch is True

    # Flip to cool — gate state should reset on next tick
    entity._attr_hvac_mode = HVACMode.COOL
    entity._pi._inputs.outdoor_temp = 30.0    # warm outdoor for cool mode
    await _tick(pi, mono=2000.0)
    assert pi._overtemp_regime is False
    assert pi._uncontrollable_entry_latch is False


@pytest.mark.asyncio
async def test_mode_change_to_off_clears_regime():
    """Heat → off should also clear regime state."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 24.0
    pi._hp_setpoint = 22
    pi._prev_hvac_mode = HVACMode.HEAT       # pretend we tracked HEAT prior

    entity._attr_hvac_mode = HVACMode.OFF
    await _tick(pi, mono=1000.0)
    # Mode change → both should clear
    assert pi._overtemp_regime is False
    assert pi._uncontrollable_entry_latch is False


@pytest.mark.asyncio
async def test_restart_starts_clean():
    """A fresh PIController (post-restart simulation) starts with regime and
    latch clear; first tick re-evaluates from scratch."""
    config = make_pi_config()
    entity = FakePIEntity(config)
    pi = entity._pi
    # Initial state immediately after construction
    assert pi._overtemp_regime is False
    assert pi._uncontrollable_entry_latch is False
    assert pi._prev_hvac_mode is None


# ── Post-exit integral-set ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_regime_exit_preserves_integrator():
    """At regime exit, the integrator is preserved at whatever value the
    freeze held it at.  An earlier integral-set design planted I to land
    sp at `ceil(room + 0.4)`; the bench revealed that under sustained-
    learning FF excursions, the 1/Ki amplification of FF chronically drove
    |I| to 280+ across multi-day runs and produced comfort failures.
    Frozen leaves I bounded; FF volatility shows up in the setpoint (which
    clamps harmlessly when extreme) rather than in I (which would take
    O(1/Ki) ticks to unwind).
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    # pre_integral sized so raw_setpoint stays well within [min, max] under
    # the new Ki=0.70 default — otherwise back-calc anti-windup fires and
    # masks the "regime exit doesn't touch the integrator" invariant under
    # test.  At outdoor=5°C / desired=22 / room=22, FF ≈ 0.25*17 = 4.25 in
    # heat.  With Ki=0.70 and integral=1.0: Ki*I = 0.7; raw_sp = 22 + 0 +
    # 0.7 + 4.25 = 26.95 < 30 max → no saturation.
    pre_integral = 1.0
    pi._pi_integral = pre_integral
    entity._attr_current_temperature = 22.0     # exactly at desired → exits
    pi._hp_setpoint = 22

    await _tick(pi, mono=1000.0)

    # After exit: regime False, integrator preserved (modulo tiny leak).
    assert pi._overtemp_regime is False
    assert abs(pi._pi_integral - pre_integral) < 0.01, (
        f"Integrator should be preserved across exit: "
        f"{pre_integral:.3f} → {pi._pi_integral:.3f}"
    )


# ── cal_midpoint gate hysteresis ────────────────────────────────────


@pytest.mark.asyncio
async def test_regime_gate_does_not_fire_until_rate_history_full():
    """Rate-signal-validity guard: the regime gate's entry condition includes
    `len(_room_temp_history) >= 5`.  Until the rate window is fully populated,
    `_room_temp_rate` either is undefined (history empty) or computed over a
    too-short window where σ_rate is wider than the noise margin.  The gate
    must not fire on a signal it can't trust.

    Belt-and-suspenders with the cal_midpoint warmup (which only covers tick
    0): this extends the protection across the rest of the rate window for
    the sensor-recovery / restart / cold-install edge cases.
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Bypass cal_midpoint warmup — we want to test ONLY the rate-validity
    # guard, not the cal_midpoint guard.
    pi._cal_midpoint_warmup_pending = False
    # Pre-arm the latch + set rate to a value that WOULD trigger the gate
    # if the rate signal were trusted.
    pi._uncontrollable_entry_latch = True
    pi._room_temp_rate = 0.05    # well above the -0.02 threshold
    entity._attr_current_temperature = 23.0
    pi._hp_setpoint = 22

    # First 4 ticks: rate history is filling up (1, 2, 3, 4 entries after
    # each tick).  Gate must stay dormant.
    for i in range(4):
        await _tick(pi, mono=1000.0 + i * 200.0)
        assert pi._overtemp_regime is False, (
            f"Regime fired at tick {i + 1} with only {len(pi._room_temp_history)} "
            f"history entries — rate signal not yet trustworthy"
        )

    # 5th tick: history is now full (5 entries).  Gate can fire — verify it
    # does, so we've actually proven the guard was the thing holding it back
    # (not some other condition).
    assert len(pi._room_temp_history) == 4
    await _tick(pi, mono=2000.0)
    assert len(pi._room_temp_history) == 5
    assert pi._overtemp_regime is True, (
        "Regime should fire once the rate-history window is fully populated"
    )


@pytest.mark.asyncio
async def test_cal_midpoint_warmup_does_not_freeze_integration_on_first_tick():
    """First-tick warmup contract — downstream consequence on integration.

    Under steady-state cal_midpoint behavior, when `hp_setpoint <= room` the
    gate flips `_hp_estimated_active_state` to False, which arms
    `skip_integration` via `not hp_estimated_active`.  Under the first-tick
    warmup guard, the gate's update is suppressed; the default
    `_hp_estimated_active_state = True` carries through; `skip_integration`
    does NOT fire from this path on tick 0.

    Net behavior change on the first tick after construction: one tick of
    integration runs that would have been frozen under the pre-warmup
    semantics.  The bound is `avg_error * dt_factor` (~0.5-2.0 units of
    integral) — a one-shot small bump, not a runaway.  This test locks
    that contract in so future changes to the warmup semantics surface
    here first.
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 21.0
    # Setup that would freeze integration under steady-state cal_midpoint
    # behavior: HP at min, room well above setpoint, error positive.
    pi._hp_setpoint = 16
    entity._attr_current_temperature = 19.0      # delta=+3 >> +0.3 hysteresis
    pi._pi_integral = 0.0
    assert pi._cal_midpoint_warmup_pending is True

    await _tick(pi)

    # Warmup clears + integration was NOT frozen by the hp_active path on
    # tick 0 (because the gate update was skipped, so hp_estimated_active
    # stayed True — `not hp_estimated_active` was False, the hp-active path
    # of skip_integration did not fire).  Integral moved (small bump).
    assert pi._cal_midpoint_warmup_pending is False
    assert pi._hp_estimated_active_state is True


@pytest.mark.asyncio
async def test_cal_midpoint_warmup_skips_first_tick():
    """First-tick warmup guard: at construction, `_hp_setpoint` reflects
    `target_temperature` (the user's *desired*), not a controller-issued
    command.  Without the warmup guard, the cal_midpoint gate evaluates
    `current_to_setpoint_delta = current_c − hp_setpoint` against that
    stale init value — and because the test fixture sets target=20.0 with
    room landing at 20.5+, that delta crosses the +0.3 hysteresis threshold
    and flips `_hp_estimated_active_state` to False on tick 0, which then
    arms the over-temp regime latch spuriously.  The warmup guard skips the
    update on the very first tick so the default `True` carries through;
    the next tick evaluates against the real controller-computed setpoint.
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Room is comfortably above the init target_temp (20.0) → delta would
    # cross +0.3 hysteresis if the gate evaluated against the stale init.
    entity._attr_current_temperature = 23.0
    pi._room_temp_rate = 0.05    # rising — would trip regime if gate fires
    # Don't pre-arm the latch; we want to verify the gate doesn't *arm* it
    # on tick 0 by spuriously flipping hp_estimated_active.
    assert pi._cal_midpoint_warmup_pending is True
    assert pi._hp_estimated_active_state is True

    await _tick(pi)

    # After tick 0: gate update was skipped — default True carried through;
    # latch did NOT arm; regime did NOT fire.  Warmup flag is cleared.
    assert pi._cal_midpoint_warmup_pending is False
    assert pi._hp_estimated_active_state is True, (
        "Warmup guard should suppress the gate update on tick 0 — "
        "default `True` must carry through"
    )
    assert pi._uncontrollable_entry_latch is False, (
        "Latch must not arm on tick 0 from a spurious cal_midpoint gate fire"
    )
    assert pi._overtemp_regime is False


@pytest.mark.asyncio
async def test_cal_midpoint_hysteresis_no_chatter_at_boundary():
    """Delta oscillating near cal_midpoint with realistic noise should not
    chatter `_hp_estimated_active_state` — the per-tick boolean would
    otherwise flip every few ticks.  With hysteresis (margin = 0.3°C) and
    σ=0.1°C noise, transitions should be rare across many ticks.
    """
    import random
    random.seed(42)
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Default cal_midpoint = (cal_min + cal_max) / 2 = (-2 + 2) / 2 = 0
    # Drive delta near 0 with σ=0.1°C noise.  Need: current = setpoint + noise.
    pi._hp_setpoint = 22
    transitions = 0
    prev_state = pi._hp_estimated_active_state
    for i in range(100):
        # Room temp = setpoint + small noise around midpoint(=0)
        entity._attr_current_temperature = 22.0 + random.gauss(0.0, 0.1)
        await _tick(pi, mono=1000.0 + i * 900)
        if pi._hp_estimated_active_state != prev_state:
            transitions += 1
            prev_state = pi._hp_estimated_active_state
    # Without hysteresis, expected ~50 transitions (roughly 50% sign flips).
    # With 0.3°C hysteresis vs 0.1°C noise, should be 0 or very few.
    assert transitions <= 5, (
        f"State chattered {transitions} times in 100 ticks despite hysteresis "
        f"(σ=0.1°C noise vs 0.3°C margin should give near-zero transitions)"
    )


@pytest.mark.asyncio
async def test_cal_midpoint_hysteresis_enter_inactive_above_margin():
    """When state is active, only transition to inactive when delta clearly
    exceeds cal_midpoint + hysteresis."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._hp_setpoint = 22
    pi._hp_estimated_active_state = True

    # Within margin: stays active
    entity._attr_current_temperature = 22.2  # delta = +0.2 < 0 + 0.3 hyst
    await _tick(pi, mono=1000.0)
    assert pi._hp_estimated_active_state is True

    # Beyond margin: transitions to inactive
    entity._attr_current_temperature = 22.5  # delta = +0.5 > 0 + 0.3 hyst
    await _tick(pi, mono=2000.0)
    assert pi._hp_estimated_active_state is False


@pytest.mark.asyncio
async def test_cal_midpoint_hysteresis_reenter_active_below_margin():
    """Symmetric: when state is inactive, only transition to active when
    delta clearly below cal_midpoint - hysteresis."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._hp_setpoint = 22
    pi._hp_estimated_active_state = False

    # Within margin (still above lower hyst boundary): stays inactive
    entity._attr_current_temperature = 21.8  # delta = -0.2 > 0 - 0.3 hyst
    await _tick(pi, mono=1000.0)
    assert pi._hp_estimated_active_state is False

    # Beyond margin: transitions to active
    entity._attr_current_temperature = 21.5  # delta = -0.5 < 0 - 0.3 hyst
    await _tick(pi, mono=2000.0)
    assert pi._hp_estimated_active_state is True


@pytest.mark.asyncio
async def test_regime_enters_in_cool_when_room_under():
    """Cool mode mirror: room < desired - ENTER triggers gate."""
    entity = _make_entity(mode=HVACMode.COOL, outdoor_c=30.0)
    pi = entity._pi
    pi._desired_temp = 22.0
    # Bypass the first-tick warmup guard — post-warmup gate dynamics.
    pi._cal_midpoint_warmup_pending = False
    _warm_rate_window(pi, current_c=19.0)        # rate=0 (steady), passes -0.02 threshold
    # Cool mode: existing gate fires when delta = current - setpoint < cal_min
    # (cal_min < 0 in cool).  current=19, setpoint=22 → delta=-3, which makes
    # hp_estimated_active=False in cool (delta >= cal_midpoint is the heat check;
    # for cool the inverted check fires when delta < cal_midpoint, i.e., room
    # cooler than setpoint).
    entity._attr_current_temperature = 19.0    # 3°C below → > ENTER
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._uncontrollable_entry_latch is True
    assert pi._overtemp_regime is True


# ── Rate-based gate: load-bearing scenarios + threshold calibration (#108) ──
#
# The four entry-side T1-T4 cases are covered by the entry/exit block above.
# What follows adds T5/T6 (exit + post-exit chatter), the explicit 0-vs-−0.02
# threshold comparison the user asked for, and a σ_rate noise-floor check
# that validates the threshold has the headroom it claims.


@pytest.mark.asyncio
async def test_t6_no_post_exit_chatter_when_no_new_exogenous_event():
    """T6: after the regime exits at room=desired, the gate must not re-fire
    from the post-exit transient alone.  Under the legacy design (>1.0 entry
    threshold, +0.5 exit hysteresis, latch reset at error≤0), the post-exit
    sp jumped from idle (16) to PI-computed-with-wound-integral (23), which
    reheated the room over +1.0 and re-fired the regime — the limit cycle.
    Under the rate-based redesign (#108), this scenario should keep the
    regime cleanly exited.

    Verified by: (a) the integral is planted at exit per the new design (the
    target_sp formula keeps sp at room+~0.4, not deep into reheat territory),
    (b) the latch reset condition (overtemp_error ≤ 0) fires on this tick,
    and (c) with the latch cleared, even small post-exit room rises don't
    re-arm the gate without a NEW exogenous-heat event.
    """
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    # Engage regime: latch armed, room warm and rising
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 23.5
    pi._sensor_filtered = 23.5
    pi._hp_setpoint = 18
    _warm_rate_window(pi, current_c=23.5, target_rate=0.05)
    await _tick(pi, mono=1000.0)
    assert pi._overtemp_regime is True

    # Drive room to desired AND past it (so the per-tick latch update sees
    # overtemp_error≤0 AND hp_estimated_active becomes True via natural HP
    # response post-exit).  Bypass filter τ — this is a state-machine test.
    entity._attr_current_temperature = 21.9      # just below desired
    pi._sensor_filtered = 21.9
    # Simulate the natural HP response: post-exit integral-set put sp ≈
    # room+0.4; manually reflect that so hp_estimated_active will be True.
    pi._hp_setpoint = 23
    await _tick(pi, mono=2000.0)
    assert pi._overtemp_regime is False, "Regime didn't exit at error≤0"
    # Now overtemp_error<0 AND HP active → latch reset.
    assert pi._uncontrollable_entry_latch is False, (
        "Latch should reset (overtemp_error<0 + hp_active → reset path)"
    )

    # Continue ticking with the room hovering around desired.  No new
    # exogenous heat → latch stays cleared → regime cannot re-fire even on
    # a brief rate excursion.
    for i in range(8):
        room = 22.0 + (0.1 if i % 2 else -0.1)
        entity._attr_current_temperature = room
        pi._sensor_filtered = room
        await _tick(pi, mono=3000.0 + i * 900.0)
        assert pi._overtemp_regime is False, (
            f"Regime re-engaged on tick {i} without new exogenous event — "
            f"the post-exit limit cycle has returned"
        )


@pytest.mark.asyncio
async def test_rate_threshold_negative_002_engages_at_borderline_cooling(
    monkeypatch,
):
    """Threshold = −0.02: at borderline cooling rate (−0.01 °C/min, within the
    noise band), the gate engages — treats "barely cooling" as "not
    recovering at meaningful pace."  This is the codebase default."""
    monkeypatch.setattr(
        "custom_components.tasmota_irhvac.pi.pi_controller."
        "DEFAULT_OVERTEMP_REGIME_RATE_THRESHOLD_C_PER_MIN",
        -0.02,
    )
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 22.5
    pi._hp_setpoint = 18                        # delta=-4.5 → hp inactive
    _warm_rate_window(pi, current_c=22.5, target_rate=-0.01)  # borderline cooling

    await _tick(pi)
    assert pi._overtemp_regime is True, (
        "threshold=−0.02: borderline-cooling room (rate=−0.01, between "
        "thresholds) should engage regime — room isn't recovering at "
        "meaningful pace"
    )


@pytest.mark.asyncio
async def test_rate_threshold_zero_misses_borderline_cooling(monkeypatch):
    """Threshold = 0: at the SAME borderline cooling rate (−0.01 °C/min),
    the gate stays dormant.  Demonstrates the trade-off the threshold
    choice makes: 0 treats any cooling as recovery; −0.02 treats only
    meaningful cooling that way.  Use this to consciously calibrate the
    threshold value rather than infer it from runtime behaviour later."""
    monkeypatch.setattr(
        "custom_components.tasmota_irhvac.pi.pi_controller."
        "DEFAULT_OVERTEMP_REGIME_RATE_THRESHOLD_C_PER_MIN",
        0.0,
    )
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 22.5
    pi._hp_setpoint = 18
    _warm_rate_window(pi, current_c=22.5, target_rate=-0.01)  # SAME borderline cooling

    await _tick(pi)
    assert pi._overtemp_regime is False, (
        "threshold=0: borderline-cooling room (rate=−0.01) should NOT engage "
        "regime — under threshold=0, any negative rate counts as recovery"
    )


@pytest.mark.asyncio
async def test_rate_threshold_both_engage_on_meaningful_rise(monkeypatch):
    """Sanity: both threshold choices (0 and −0.02) agree on meaningful rises
    (rate +0.05 °C/min) — the comparison is only meaningful in the borderline
    band between the two."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 22.5
    pi._hp_setpoint = 18
    _warm_rate_window(pi, current_c=22.5, target_rate=0.05)  # meaningful rise

    for threshold in (0.0, -0.02):
        monkeypatch.setattr(
            "custom_components.tasmota_irhvac.pi.pi_controller."
            "DEFAULT_OVERTEMP_REGIME_RATE_THRESHOLD_C_PER_MIN",
            threshold,
        )
        pi._overtemp_regime = False             # reset between iterations
        await _tick(pi, mono=1000.0 + threshold * 100)  # disambiguate mono
        assert pi._overtemp_regime is True, (
            f"threshold={threshold} should engage at rate +0.05 °C/min — "
            f"meaningful rise is above both thresholds"
        )


def test_rate_noise_floor_calibration_threshold_has_headroom():
    """Calibrates σ_rate empirically for the bench noise model (σ_sensor=0.1
    °C, 5-tick rate window at 3-min ticks) and asserts the regime-gate
    threshold (−0.02 °C/min) sits below at least 0.5σ of the rate-signal
    noise floor.  If sensor noise rises (different hardware) or the rate
    window changes, this test will surface the need to re-calibrate."""
    import random as _random
    import statistics as _stat
    from custom_components.tasmota_irhvac.const import (
        DEFAULT_OVERTEMP_REGIME_RATE_THRESHOLD_C_PER_MIN,
    )

    SIGMA_SENSOR = 0.1     # °C
    TICK_MIN = 3.0
    N_TICKS_RATE = 5       # rate computed over 5 ticks per pi_controller.py
    N_SAMPLES = 10000

    _random.seed(42)
    rates = []
    for _ in range(N_SAMPLES):
        # Steady-state room (true=20.0); 5 noisy readings span 4 intervals
        readings = [20.0 + _random.gauss(0.0, SIGMA_SENSOR) for _ in range(N_TICKS_RATE)]
        elapsed_min = (N_TICKS_RATE - 1) * TICK_MIN
        rates.append((readings[-1] - readings[0]) / elapsed_min)

    sigma_rate = _stat.stdev(rates)
    threshold = DEFAULT_OVERTEMP_REGIME_RATE_THRESHOLD_C_PER_MIN

    # Threshold should sit below zero with at least ~0.5σ of rate noise of
    # headroom — far enough below the noise mean (0) that a single sample
    # below threshold is unusual, but close enough that genuinely-cooling
    # rooms cross it cleanly.
    assert threshold < 0, (
        f"Threshold {threshold} should be negative — see #108 design notes "
        f"(test is on the 'recovering' side)"
    )
    assert abs(threshold) > 0.5 * sigma_rate, (
        f"Threshold {threshold} is within 0.5σ of rate noise floor "
        f"(σ_rate≈{sigma_rate:.3f} °C/min for σ_sensor={SIGMA_SENSOR}, "
        f"{N_TICKS_RATE}-tick window at {TICK_MIN}-min ticks).  Steady-state "
        f"noise alone will frequently dip below threshold and erroneously "
        f"exempt the regime."
    )
    # Document the calibration result for human review.
    print(
        f"\n[rate-noise calibration] σ_sensor={SIGMA_SENSOR}, "
        f"window={N_TICKS_RATE} ticks × {TICK_MIN} min → σ_rate≈"
        f"{sigma_rate:.4f} °C/min; threshold={threshold} = "
        f"{abs(threshold)/sigma_rate:.2f}σ below zero."
    )
