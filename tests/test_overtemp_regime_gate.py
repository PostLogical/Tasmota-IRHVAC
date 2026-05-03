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


async def _tick(pi, *, mono: float = 1000.0):
    pi._pi_last_tick_time = mono - 900.0
    with patch("time.monotonic", return_value=mono):
        await pi._pi_tick()


# ── Latch behavior ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_latch_sets_when_existing_gate_fires():
    """Latch sets when delta crosses cal_midpoint (existing gate fires)."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
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
async def test_latch_persists_during_overtemp_episode():
    """Latch persists across ticks where existing gate transiently unfreezes
    (FF nudges setpoint above current) — that's the whole point: my gate
    can fire in those moments because we've ENTERED uncontrollable state."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
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
async def test_regime_enters_when_overtemp_and_latched():
    """The full positive case: room over by ENTER threshold AND latch set."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    entity._attr_current_temperature = 24.0   # 2°C over → > ENTER (1.0)
    pi._hp_setpoint = 22                       # delta=+2 → existing gate fires

    await _tick(pi)
    assert pi._uncontrollable_entry_latch is True
    assert pi._overtemp_regime is True


@pytest.mark.asyncio
async def test_regime_does_not_enter_below_enter_threshold():
    """Room over-temp but < ENTER threshold (1.0°C) — gate dormant."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True       # latch already set
    entity._attr_current_temperature = 22.5     # only 0.5°C over
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._overtemp_regime is False


@pytest.mark.asyncio
async def test_regime_exits_when_temp_returns_to_band():
    """Regime exits when overtemp_error drops below EXIT threshold (0.5°C)."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._overtemp_regime = True
    pi._uncontrollable_entry_latch = True
    entity._attr_current_temperature = 22.4    # only 0.4°C over → < EXIT
    pi._hp_setpoint = 22

    await _tick(pi)
    assert pi._overtemp_regime is False


@pytest.mark.asyncio
async def test_regime_hysteresis_does_not_chatter_in_band():
    """Room oscillating between EXIT and ENTER thresholds: regime state
    should not chatter — once exited, stays exited until ENTER again."""
    entity = _make_entity()
    pi = entity._pi
    pi._desired_temp = 22.0
    pi._uncontrollable_entry_latch = True
    pi._overtemp_regime = False

    # Room at desired + 0.7 (between EXIT 0.5 and ENTER 1.0):
    # not active → does not enter (above EXIT but below ENTER)
    entity._attr_current_temperature = 22.7
    pi._hp_setpoint = 22
    await _tick(pi, mono=1000.0)
    assert pi._overtemp_regime is False

    # Now jump to 23.5 (above ENTER) → enters
    entity._attr_current_temperature = 23.5
    await _tick(pi, mono=2000.0)
    assert pi._overtemp_regime is True

    # Drop back to 22.7 (between thresholds): active → stays active
    entity._attr_current_temperature = 22.7
    await _tick(pi, mono=3000.0)
    assert pi._overtemp_regime is True


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
    # Engage regime in heat
    entity._attr_current_temperature = 24.0
    pi._hp_setpoint = 22
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


@pytest.mark.asyncio
async def test_regime_enters_in_cool_when_room_under():
    """Cool mode mirror: room < desired - ENTER triggers gate."""
    entity = _make_entity(mode=HVACMode.COOL, outdoor_c=30.0)
    pi = entity._pi
    pi._desired_temp = 22.0
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
