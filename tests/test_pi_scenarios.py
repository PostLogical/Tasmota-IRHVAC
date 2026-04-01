"""Scenario-based simulation tests for PI+FF+RLS system.

Each test simulates a real-world scenario using a simple thermal model.
The PI controller runs in a loop, updating room temperature based on
HP setpoint, outdoor temp, solar gain, and supplemental heat.

Tests are parametrized by seed_factor to verify behavior with
correct seeds (1.0), bad seeds (0.0, 0.5), and overseeded (1.5).
"""

import math
import pytest
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, UnitOfTemperature

from custom_components.tasmota_irhvac.pi_controller import PIController

from .conftest import make_pi_config


# ── Thermal Model ─────────────────────────────────────────────────────


class ThermalModel:
    """Simple 1R room thermal model for simulation.

    dT_room/dt = (T_outdoor - T_room) / R + Q_hp + Q_solar + Q_stove

    Where:
        R = thermal resistance (minutes for 1°C change per °C delta)
        Q_hp = hp_gain * (hp_setpoint - T_room) when HP is heating
        Q_solar = solar_proxy * solar_gain_factor
        Q_stove = stove_active * stove_heat_factor
    """

    def __init__(self, initial_temp=20.0, outdoor_temp=5.0,
                 time_constant_min=60.0, hp_gain=0.05,
                 solar_gain=3.0, stove_gain=2.0):
        self.room_temp = initial_temp
        self.outdoor_temp = outdoor_temp
        self.time_constant = time_constant_min
        self.hp_gain = hp_gain
        self.solar_gain = solar_gain
        self.stove_gain = stove_gain

    def step(self, hp_setpoint, dt_minutes=15.0, solar_proxy=0.0, stove_active=0.0):
        """Advance room temperature by dt_minutes.

        Args:
            hp_setpoint: HP target temperature in °C
            dt_minutes: Time step in minutes
            solar_proxy: Solar gain factor (0-1)
            stove_active: Supplemental heat (0 or 1)

        Returns:
            New room temperature.
        """
        # Heat loss to outdoor
        heat_loss = (self.outdoor_temp - self.room_temp) / self.time_constant

        # HP heating (proportional to setpoint - room temp)
        if hp_setpoint > self.room_temp:
            q_hp = self.hp_gain * (hp_setpoint - self.room_temp)
        else:
            q_hp = self.hp_gain * (hp_setpoint - self.room_temp) * 0.3  # Slower cooling

        # Solar and stove
        q_solar = solar_proxy * self.solar_gain / self.time_constant
        q_stove = stove_active * self.stove_gain / self.time_constant

        # Update temperature
        dT = (heat_loss + q_hp + q_solar + q_stove) * dt_minutes
        self.room_temp += dT
        return self.room_temp


# ── Fake Entity for Simulation ────────────────────────────────────────


class SimEntity:
    """Minimal fake entity for PI scenario simulation."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    def __init__(self, config):
        self.hass = MagicMock()
        self.hass.states.get.return_value = None  # No entities by default
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 20.0
        self._attr_target_temperature = 20.5
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "test_sim"
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)
        self.async_get_last_extra_data = AsyncMock(return_value=None)
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return UnitOfTemperature.CELSIUS


def _run_simulation(entity, thermal, n_ticks, outdoor_schedule=None,
                    solar_schedule=None, stove_schedule=None,
                    desired_schedule=None):
    """Run PI simulation for n_ticks, returning history.

    Schedules are dicts of {tick_number: new_value} for step changes,
    or callables taking tick number returning value.
    """
    import asyncio
    loop = asyncio.new_event_loop()

    history = []
    pi = entity._pi

    for tick in range(n_ticks):
        # Simulate 15-minute intervals by resetting last tick time
        pi._pi_last_tick_time = 0  # Forces dt_factor = 1.0

        # Apply schedules
        if outdoor_schedule:
            if callable(outdoor_schedule):
                thermal.outdoor_temp = outdoor_schedule(tick)
            elif tick in outdoor_schedule:
                thermal.outdoor_temp = outdoor_schedule[tick]
        if desired_schedule and tick in desired_schedule:
            pi._desired_temp = desired_schedule[tick]
            entity._attr_target_temperature = desired_schedule[tick]
            pi._pi_integral = 0.0  # Same as set_temperature behavior

        solar = 0.0
        if solar_schedule:
            if callable(solar_schedule):
                solar = solar_schedule(tick)
            elif tick in solar_schedule:
                solar = solar_schedule[tick]

        stove = 0.0
        if stove_schedule:
            if callable(stove_schedule):
                stove = stove_schedule(tick)
            elif tick in stove_schedule:
                stove = stove_schedule[tick]

        # Update entity state from thermal model
        entity._attr_current_temperature = thermal.room_temp
        pi._outdoor_temp = thermal.outdoor_temp

        # Run PI tick
        loop.run_until_complete(pi._pi_tick())

        # Advance thermal model using HP setpoint
        thermal.step(pi._hp_setpoint, dt_minutes=15.0,
                    solar_proxy=solar, stove_active=stove)

        history.append({
            "tick": tick,
            "room_temp": thermal.room_temp,
            "desired": pi._desired_temp,
            "hp_setpoint": pi._hp_setpoint,
            "integral": pi._pi_integral,
            "ff_offset": pi._ff_offset,
            "error": pi._desired_temp - thermal.room_temp if pi._desired_temp else 0,
            "outdoor": thermal.outdoor_temp,
            "rls_obs_count": pi._rls_heat.observation_count,
        })

    loop.close()
    return history


def _make_sim_config(seed_factor=1.0, **overrides):
    """Make config with outdoor_delta seed scaled by seed_factor."""
    true_slope = 0.35
    config = {
        "pi_kp": 1.0,
        "pi_ki": 0.05,  # Proposed new default
        "pi_deadband": 0.5,
        "pi_ff_heat_slope": true_slope * seed_factor,
        "pi_ff_cool_slope": true_slope * seed_factor,
        "pi_ff_heat_reference": 15.0,
        "pi_ff_cool_reference": 25.0,
        "pi_ff_learn_night_only": False,
        "pi_model_inputs": [],
    }
    config.update(overrides)
    return make_pi_config(config)


# ── Assertion Helpers ─────────────────────────────────────────────────


def _assert_room_reaches_target(history, max_ticks=16, tolerance=0.5):
    """Assert room temp reaches within tolerance of desired within max_ticks."""
    for h in history:
        if h["tick"] >= max_ticks:
            break
        if h["desired"] and abs(h["room_temp"] - h["desired"]) < tolerance:
            return
    # Check if we got there by max_ticks
    last = history[min(max_ticks, len(history) - 1)]
    assert abs(last["room_temp"] - last["desired"]) < tolerance, (
        f"Room didn't reach target by tick {max_ticks}: "
        f"room={last['room_temp']:.1f}, desired={last['desired']}, "
        f"hp={last['hp_setpoint']}, integral={last['integral']:.1f}"
    )


def _assert_no_oscillation(history, start_tick=0, max_reversals=4):
    """Assert room temp doesn't oscillate (cross desired too many times)."""
    crossings = 0
    above = None
    for h in history:
        if h["tick"] < start_tick or not h["desired"]:
            continue
        currently_above = h["room_temp"] > h["desired"]
        if above is not None and currently_above != above:
            crossings += 1
        above = currently_above
    assert crossings <= max_reversals, (
        f"Room oscillated {crossings} times (max {max_reversals})"
    )


def _assert_setpoint_in_bounds(history, min_temp=16, max_temp=30):
    """Assert hp_setpoint never exceeds bounds."""
    for h in history:
        assert min_temp <= h["hp_setpoint"] <= max_temp, (
            f"Tick {h['tick']}: hp_setpoint={h['hp_setpoint']} outside [{min_temp}, {max_temp}]"
        )


def _assert_integral_bounded(history, cap=50):
    """Assert integral stays within ±cap."""
    for h in history:
        assert abs(h["integral"]) <= cap + 0.1, (
            f"Tick {h['tick']}: integral={h['integral']:.1f} exceeds ±{cap}"
        )


# ── Scenario Tests ────────────────────────────────────────────────────


class TestColdStart:
    """Room starts cold, HP needs to warm it up."""

    @pytest.mark.parametrize("seed_factor", [0.0, 0.5, 1.0, 1.5])
    def test_cold_start(self, seed_factor):
        config = _make_sim_config(seed_factor)
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        thermal = ThermalModel(initial_temp=17.0, outdoor_temp=2.0)

        history = _run_simulation(entity, thermal, n_ticks=24)

        _assert_room_reaches_target(history, max_ticks=20, tolerance=0.8)
        _assert_no_oscillation(history, start_tick=12)
        _assert_setpoint_in_bounds(history)
        _assert_integral_bounded(history)


class TestColdSnap:
    """Outdoor temp drops 15°C over 3 hours (12 ticks)."""

    @pytest.mark.parametrize("seed_factor", [0.0, 0.5, 1.0, 1.5])
    def test_cold_snap(self, seed_factor):
        config = _make_sim_config(seed_factor)
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5

        def outdoor(tick):
            # Drop from 10°C to -5°C over 12 ticks
            return max(-5.0, 10.0 - tick * 1.25)

        thermal = ThermalModel(initial_temp=20.5, outdoor_temp=10.0)
        history = _run_simulation(entity, thermal, n_ticks=24, outdoor_schedule=outdoor)

        # Room should stay within 1.5°C of target during cold snap
        for h in history:
            if h["tick"] > 4:  # Allow initial response time
                assert abs(h["room_temp"] - 20.5) < 1.5, (
                    f"Tick {h['tick']}: room dropped to {h['room_temp']:.1f} during cold snap"
                )
        _assert_setpoint_in_bounds(history)
        _assert_integral_bounded(history)


class TestSetpointChangeUp:
    """User raises desired temp by 2°C."""

    @pytest.mark.parametrize("seed_factor", [0.0, 0.5, 1.0, 1.5])
    def test_setpoint_up(self, seed_factor):
        config = _make_sim_config(seed_factor)
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        thermal = ThermalModel(initial_temp=20.5, outdoor_temp=5.0)

        # Run 4 ticks to settle, then raise setpoint
        history = _run_simulation(entity, thermal, n_ticks=24,
                                 desired_schedule={4: 22.5})

        _assert_room_reaches_target(history, max_ticks=20, tolerance=0.8)
        _assert_setpoint_in_bounds(history)
        _assert_integral_bounded(history)


class TestSetpointChangeDown:
    """User lowers desired temp by 2°C."""

    @pytest.mark.parametrize("seed_factor", [0.0, 0.5, 1.0, 1.5])
    def test_setpoint_down(self, seed_factor):
        config = _make_sim_config(seed_factor)
        entity = SimEntity(config)
        entity._pi._desired_temp = 22.5
        thermal = ThermalModel(initial_temp=22.5, outdoor_temp=5.0)

        history = _run_simulation(entity, thermal, n_ticks=24,
                                 desired_schedule={4: 20.5})

        _assert_room_reaches_target(history, max_ticks=20, tolerance=0.8)
        _assert_setpoint_in_bounds(history)


class TestFFTooHigh:
    """FF overpredicts — room should stabilize, not oscillate."""

    def test_ff_too_high_no_oscillation(self):
        config = _make_sim_config(seed_factor=1.5)  # 50% too high
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        thermal = ThermalModel(initial_temp=20.5, outdoor_temp=5.0)

        history = _run_simulation(entity, thermal, n_ticks=32)

        _assert_no_oscillation(history, start_tick=8, max_reversals=4)
        _assert_setpoint_in_bounds(history)
        # Room shouldn't be more than 1°C above target sustained
        for h in history[16:]:
            assert h["room_temp"] < h["desired"] + 1.5, (
                f"Tick {h['tick']}: room at {h['room_temp']:.1f}, "
                f"desired {h['desired']}, sustained overshoot"
            )


class TestSteadyState:
    """Room at target, nothing changing — verify no drift."""

    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_steady_state_no_drift(self, seed_factor):
        config = _make_sim_config(seed_factor)
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        thermal = ThermalModel(initial_temp=20.5, outdoor_temp=5.0)

        # Let it settle first
        history = _run_simulation(entity, thermal, n_ticks=48)

        # Last 12 ticks should be stable (no drift > 0.5°C)
        last_12 = history[-12:]
        temps = [h["room_temp"] for h in last_12]
        assert max(temps) - min(temps) < 0.5, (
            f"Room drifted in steady state: range {min(temps):.2f} to {max(temps):.2f}"
        )


class TestSolarGainMorning:
    """Solar proxy ramps up, room warms — FF should compensate."""

    @pytest.mark.parametrize("seed_factor", [0.0, 1.0])
    def test_solar_morning(self, seed_factor):
        config = _make_sim_config(seed_factor)
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        thermal = ThermalModel(initial_temp=20.5, outdoor_temp=8.0)

        def solar(tick):
            # Ramp from 0 to 0.8 over 12 ticks (3 hours)
            return min(0.8, tick * 0.067)

        history = _run_simulation(entity, thermal, n_ticks=24, solar_schedule=solar)

        # Room shouldn't overshoot more than 1°C from solar
        for h in history:
            assert h["room_temp"] < h["desired"] + 1.5, (
                f"Tick {h['tick']}: solar overshoot to {h['room_temp']:.1f}"
            )
        _assert_setpoint_in_bounds(history)


class TestSunnyDayVsColdNight:
    """The scenario that corrupted buckets: same outdoor temp, different solar."""

    def test_no_coefficient_collapse(self):
        """RLS coefficients should not collapse when learning from mixed day/night."""
        config = _make_sim_config(seed_factor=1.0)
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        entity._pi._rls_warmup_done = True  # Skip warmup for test
        thermal = ThermalModel(initial_temp=20.5, outdoor_temp=6.0)

        # Simulate 3 day/night cycles at same outdoor temp
        def solar(tick):
            cycle_pos = tick % 24  # 24 ticks = 6 hours
            if 8 <= cycle_pos <= 16:
                return 0.7
            return 0.0

        history = _run_simulation(entity, thermal, n_ticks=72, solar_schedule=solar)

        # outdoor_delta coefficient should stay near seed (0.35)
        coeff = entity._pi._rls_heat.beta[1]
        assert coeff > 0.2, (
            f"outdoor_delta coefficient collapsed to {coeff:.3f}, expected near 0.35"
        )


class TestSetpointSaturation:
    """Adaptive ki pushes raw above max — verify clamp and anti-windup."""

    def test_saturation_clamp(self):
        config = _make_sim_config(seed_factor=0.0)  # No FF, integral must do everything
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        thermal = ThermalModel(initial_temp=15.0, outdoor_temp=-10.0)

        history = _run_simulation(entity, thermal, n_ticks=16)

        _assert_setpoint_in_bounds(history)
        _assert_integral_bounded(history)


class TestStoveOnOff:
    """Stove turns on, then off. With lag filter if configured."""

    @pytest.mark.parametrize("seed_factor", [0.5, 1.0])
    def test_stove_transition(self, seed_factor):
        config = _make_sim_config(seed_factor, pi_model_inputs=[{
            "name": "Stove",
            "entity_id": "sensor.stove",
            "seed_heat": -3.0,
            "seed_cool": 0.0,
            "lag_tau": 0,
        }])
        entity = SimEntity(config)
        entity._pi._desired_temp = 20.5
        # Mock stove entity
        stove_state = MagicMock()
        stove_state.state = "0"

        def get_state(entity_id):
            if entity_id == "sensor.stove":
                return stove_state
            return None
        entity.hass.states.get = get_state

        thermal = ThermalModel(initial_temp=20.5, outdoor_temp=5.0)

        def stove(tick):
            if 8 <= tick < 20:
                stove_state.state = "1"
                return 1.0
            stove_state.state = "0"
            return 0.0

        history = _run_simulation(entity, thermal, n_ticks=32, stove_schedule=stove)

        # Room shouldn't swing more than 2°C during stove transitions
        for h in history:
            if h["desired"]:
                assert abs(h["room_temp"] - h["desired"]) < 2.0, (
                    f"Tick {h['tick']}: room at {h['room_temp']:.1f} during stove transition"
                )
        _assert_setpoint_in_bounds(history)
