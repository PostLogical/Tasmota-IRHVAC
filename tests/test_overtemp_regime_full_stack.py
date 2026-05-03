"""Full-stack test: solar in FF model, sensor drops while room stays warm.

Reproduces the bundle's failure mechanism (hp_debug_20260503_0407) with the
specific element the prior session's harness was missing: **solar is a
model_input that the FF model knows about**.

Bundle mechanism:
  - Solar Proxy is in the FF model with a configured seed (4.0)
  - FF reads the live sensor → contribution = coeff × scaled_input (with
    heat-source convention, contribution is negative when sun is shining)
  - Sun goes down → sensor drops to 0 → FF contribution drops to 0
  - Room is still warm because thermal mass retains absorbed solar
  - Without gate: existing cal_midpoint freeze releases when FF nudges
    setpoint above current temperature, integrator winds negative
  - With gate: latch set during over-temp + existing-freeze, regime stays
    active even when FF transiently flickers, integrator preserved

This harness drives room temperature programmatically (decoupled from a
thermal model) so the test focuses on gate behavior rather than physics
calibration.  See `test_overtemp_regime_replay.py` for high-fidelity
validation against actual bundle data.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from homeassistant.components.climate.const import HVACMode

from .conftest import make_pi_config
from .test_pi_controller import FakePIEntity


class _MockedSensorEntity(FakePIEntity):
    """FakePIEntity with hass.states.get returning callable-driven values."""

    def __init__(self, config):
        super().__init__(config)
        self._sensor_callables: dict[str, Callable[[], Any]] = {}
        states = MagicMock()
        def states_get(entity_id: str):
            if entity_id in self._sensor_callables:
                value = self._sensor_callables[entity_id]()
                state = MagicMock()
                state.state = str(value)
                state.attributes = {"unit_of_measurement": "°C"}
                return state
            return None
        states.get = states_get
        self.hass.states = states

    def set_sensor(self, entity_id: str, callable_fn: Callable[[], Any]):
        self._sensor_callables[entity_id] = callable_fn


def _make_solar_config():
    """Config with solar_proxy as model_input, seed_heat=4.0 (matches bundle)."""
    return make_pi_config({
        "outdoor_temp_sensor": "sensor.outdoor",
        "pi_model_inputs": [{
            "name": "Solar Proxy",
            "entity_id": "sensor.solar_proxy",
            "seed_heat": 4.0,
            "seed_cool": 0.0,
            "lag_tau": 0,
        }],
    })


@pytest.mark.asyncio
async def test_post_solar_integrator_preserved_with_gate():
    """Drive the bundle's failure scenario with controlled inputs.

    Phase 1 (sun up, room warm): solar_signal=1.0, room=25°C, desired=22°C.
        FF contribution from solar = -4×1 = -4°C (subtracts from setpoint).
        Room is over-temp, existing gate fires, latch sets, regime fires.

    Phase 2 (sun goes down, room descending): solar_signal=0.0, room slowly
        cools from 25 → 22 over 8h.  FF contribution from solar = 0.  FF total
        becomes positive (outdoor seed contribution kicks back in), wants to
        push setpoint up.  Without gate, existing freeze releases, integrator
        winds negative on phantom load.  With gate, latch persists through
        over-temp episode, regime stays active, integrator preserved.

    Assertion: across the entire post-solar window, integrator should not
        wind down significantly from where it was at sunset.
    """
    config = _make_solar_config()
    entity = _MockedSensorEntity(config)
    entity._attr_hvac_mode = HVACMode.HEAT

    desired_c = 22.0
    outdoor_c = 5.0
    solar_signal = [1.0]    # mutable for closure
    room_temp = [25.0]      # mutable for closure

    entity.set_sensor("sensor.solar_proxy", lambda: solar_signal[0])
    entity.set_sensor("sensor.outdoor", lambda: outdoor_c)
    entity._pi._inputs.outdoor_temp = outdoor_c

    pi = entity._pi
    pi._desired_temp = desired_c

    # Phase 1: sun up, room hot, settle a few ticks so latch sets and regime
    # engages.
    pi_interval_s = 900.0
    mono = max(pi_interval_s, pi._last_setpoint_change_time + pi_interval_s)
    integral_history: list[float] = []
    regime_history: list[bool] = []
    latch_history: list[bool] = []
    room_history: list[float] = []
    solar_history: list[float] = []

    async def tick():
        nonlocal mono
        entity._attr_current_temperature = room_temp[0]
        pi._pi_last_tick_time = mono - pi_interval_s
        with patch("time.monotonic", return_value=mono):
            await pi._pi_tick()
        mono += pi_interval_s
        integral_history.append(pi._pi_integral)
        regime_history.append(pi._overtemp_regime)
        latch_history.append(pi._uncontrollable_entry_latch)
        room_history.append(room_temp[0])
        solar_history.append(solar_signal[0])

    # 4 ticks (1 hour) at sun up + room over-temp → latch sets, regime engages
    for _ in range(4):
        await tick()

    assert pi._uncontrollable_entry_latch is True, "Latch should set during sun-up over-temp"
    assert pi._overtemp_regime is True, "Regime should engage during sun-up over-temp"

    # Pre-cooldown integrator value (snapshot)
    pre_cooldown_idx = len(integral_history) - 1
    pre_cooldown_integral = integral_history[pre_cooldown_idx]

    # Phase 2: sun goes down, room descends slowly from 25 → 22 over 32 ticks (8h)
    solar_signal[0] = 0.0
    descent_ticks = 32
    descent_per_tick = (25.0 - 22.5) / descent_ticks    # end at 22.5°C (just inside EXIT band)
    for _ in range(descent_ticks):
        room_temp[0] -= descent_per_tick
        await tick()

    # Cooldown window assertions
    cooldown_min = min(integral_history[pre_cooldown_idx + 1:])
    cooldown_delta = cooldown_min - pre_cooldown_integral
    assert cooldown_delta > -1.5, (
        f"Integrator wound by {cooldown_delta:.2f} during post-solar cooldown "
        f"(pre={pre_cooldown_integral:.2f}, min={cooldown_min:.2f}). "
        f"Gate should prevent this."
    )

    cooldown_active = sum(regime_history[pre_cooldown_idx + 1:])
    cooldown_total = len(regime_history) - (pre_cooldown_idx + 1)
    # Most of cooldown should still be in regime (room still over-temp).
    # Allow last few ticks where room dips into EXIT band.
    assert cooldown_active >= cooldown_total - 4, (
        f"Regime should be active for almost all of cooldown "
        f"({cooldown_active}/{cooldown_total} ticks active)"
    )
