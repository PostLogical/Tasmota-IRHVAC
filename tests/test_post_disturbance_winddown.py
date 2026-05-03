"""Failing test for post-disturbance integral wind-down (bundle 2026-05-03_0407).

After a sunny day the room stays well above setpoint for hours while absorbed
solar gain dissipates from thermal mass.  During this cooldown window the PI
error is negative (room hot), the room temperature rate is also negative
(cooling on its own), and the HP cannot help — but the conditional-integration
freeze releases as soon as `current_c - hp_setpoint` drops below the head
calibration midpoint, and the integrator then winds rapidly negative on what
is effectively phantom load.

Bundle observation: living-room integral went from +6.7 to −9.45 over ~6h of
post-peak cooldown, then fought the morning recovery.

This test reproduces the mechanism with a synthetic supplemental-heat pulse
(absorbed solar) on top of a real PIController + 1R1C room model.  Documented
as a known failing test until the fix lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, UnitOfTemperature

from custom_components.tasmota_irhvac.pi.pi_controller import PIController

from .conftest import make_pi_config


# ── 1R1C room model with HP and supplemental heat input ──────────────


@dataclass
class RoomModel:
    """1R1C room: outdoor wall loss + HP heating + exogenous supplemental.

    `supplemental_heat` (W) represents absorbed solar gain dissipating from
    thermal mass: an exogenous heat injection the controller cannot directly
    observe.
    """

    R_wall: float = 0.015
    C_room: float = 240_000.0
    hp_max_power: float = 3500.0
    hp_min_power: float = 500.0
    hp_response_time: float = 300.0

    room_temp: float = 22.0
    outdoor_temp: float = 5.0
    hp_output: float = 0.0
    supplemental_heat: float = 0.0

    def step(self, hp_setpoint: float, dt: float = 60.0) -> None:
        target = self._hp_target_power(hp_setpoint)
        alpha = min(1.0, dt / self.hp_response_time)
        self.hp_output += alpha * (target - self.hp_output)
        q_wall = (self.outdoor_temp - self.room_temp) / self.R_wall
        q_total = q_wall + self.hp_output + self.supplemental_heat
        self.room_temp += (q_total * dt) / self.C_room

    def _hp_target_power(self, hp_setpoint: float) -> float:
        diff = hp_setpoint - self.room_temp
        if diff > 0:
            return max(self.hp_min_power, self.hp_max_power * min(1.0, diff / 5.0))
        if diff < -1:
            return 0.0
        return self.hp_min_power * max(0.0, diff + 1)


class _SimEntity:
    """Minimal entity for closed-loop integration test."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _ir_temp_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    def __init__(self, config, room_temp_c: float, outdoor_temp_c: float):
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = room_temp_c
        self._attr_target_temperature = 22.0
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "winddown_test"
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)
        self.async_get_last_extra_data = AsyncMock(return_value=None)
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return self._attr_temperature_unit

    @property
    def target_temperature(self):
        if self._pi and self._pi._desired_temp is not None:
            return self._pi._desired_temp
        return self._attr_target_temperature


# ── Solar profile ────────────────────────────────────────────────────


def _solar_pulse(t_hr: float) -> float:
    """Absorbed-solar surrogate (W).

    Ramps from 0 at t=0, peaks at 2500W at t=4h, decays to ~0 by t=10h.
    Shape mirrors the bundle's living-room cooldown: room stays above target
    for several hours while the pulse fades, then settles.
    """
    if t_hr <= 0.0 or t_hr >= 12.0:
        return 0.0
    # Skewed pulse: fast ramp, slow tail (mirrors thermal-mass dissipation)
    if t_hr < 4.0:
        return 2500.0 * (t_hr / 4.0)
    return 2500.0 * math.exp(-(t_hr - 4.0) / 2.5)


# ── Sim runner ───────────────────────────────────────────────────────


@dataclass
class SimTrace:
    time_min: list[float] = field(default_factory=list)
    room_temp_c: list[float] = field(default_factory=list)
    hp_setpoint: list[int] = field(default_factory=list)
    integral: list[float] = field(default_factory=list)
    ff_offset: list[float] = field(default_factory=list)
    error: list[float] = field(default_factory=list)
    integration_frozen: list[bool] = field(default_factory=list)


async def _run_sim(
    entity: _SimEntity,
    room: RoomModel,
    duration_hours: float,
    desired_c: float,
    solar_fn=None,
    dt_room: float = 60.0,
    pi_interval: float = 900.0,
) -> SimTrace:
    """Drive the room model + controller at `dt_room` resolution; tick PI every `pi_interval`."""
    pi = entity._pi
    await pi.set_temperature(desired_c)
    pi._inputs.outdoor_temp = room.outdoor_temp

    steps = int(duration_hours * 3600 / dt_room)
    pi_interval_steps = int(pi_interval / dt_room)
    trace = SimTrace()
    mono_time = max(pi_interval, pi._last_setpoint_change_time + pi_interval)

    for step in range(steps):
        t_sec = step * dt_room
        t_hr = t_sec / 3600.0

        if solar_fn is not None:
            room.supplemental_heat = solar_fn(t_hr)

        room.step(float(pi._hp_setpoint), dt_room)
        entity._attr_current_temperature = room.room_temp

        if step % pi_interval_steps == 0 and step > 0:
            pi._pi_last_tick_time = mono_time - pi_interval
            with patch("time.monotonic", return_value=mono_time):
                await pi._pi_tick()
            mono_time += pi_interval

        if step % pi_interval_steps == 0:
            trace.time_min.append(t_sec / 60.0)
            trace.room_temp_c.append(room.room_temp)
            trace.hp_setpoint.append(int(pi._hp_setpoint))
            trace.integral.append(pi._pi_integral)
            trace.ff_offset.append(pi._ff_offset)
            trace.error.append(desired_c - room.room_temp)
            trace.integration_frozen.append(pi._integration_frozen)

    return trace


# ── The test ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_solar_decay_does_not_wind_integral_negative():
    """Integral must not wind heavily negative on phantom (room cooling on own) error.

    Setup: warmup to equilibrium, then an absorbed-solar pulse heats the room
    well above setpoint for ~4h, after which it decays over ~6h.  During the
    decay the room is hot (error < 0) and cooling on its own (rate < 0): the
    HP cannot help, so the integrator should not accumulate large negative
    debt.

    Pathology: with the current freeze logic, the integrator unfreezes the
    moment `current - hp_setpoint` drops below the head-cal midpoint and
    winds negative on phantom load.  Bundle 2026-05-03_0407 observed
    integral diving from +6.7 to −9.45 over the cooldown.

    Assertion: integral must stay above −2.0 throughout.  Threshold is
    deliberately generous so any reasonable fix passes; current code fails
    by a wide margin.
    """
    config = make_pi_config()
    desired_c = 22.0
    outdoor_c = 5.0

    room = RoomModel(room_temp=20.0, outdoor_temp=outdoor_c)
    entity = _SimEntity(config, room.room_temp, outdoor_c)

    # Warmup: 6h with no solar to settle integrator + ff_offset
    await _run_sim(entity, room, duration_hours=6.0, desired_c=desired_c)

    # Run 16h: solar pulse over t=[0,12]h, then 4h post-pulse settling
    trace = await _run_sim(
        entity, room, duration_hours=16.0, desired_c=desired_c,
        solar_fn=_solar_pulse,
    )

    # Diagnostics for the failure message — point at the worst tick
    min_integral = min(trace.integral)
    min_idx = trace.integral.index(min_integral)
    min_t_hr = trace.time_min[min_idx] / 60.0
    room_at_min = trace.room_temp_c[min_idx]
    error_at_min = trace.error[min_idx]

    assert min_integral > -2.0, (
        f"Integral wound to {min_integral:.2f} at t={min_t_hr:.1f}h "
        f"(room={room_at_min:.2f}°C, error={error_at_min:+.2f}°C). "
        "Post-disturbance cooldown should not produce a large negative "
        "integral — the room is cooling on its own and HP cannot help."
    )
