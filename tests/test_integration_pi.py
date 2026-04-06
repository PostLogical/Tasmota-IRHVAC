"""Tier 3: PI controller integration tests through real HA machinery.

These tests exercise the PI controller through the full HA stack — entity setup,
unit conversion, dispatcher signals, companion sensors. They catch the class of
bugs that FakePIEntity unit tests cannot (°F/°C mismatches, state machine wiring).
"""

import json
import pytest

from homeassistant.components.climate.const import HVACMode
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import async_fire_mqtt_message

from homeassistant.helpers.dispatcher import async_dispatcher_send

from custom_components.tasmota_irhvac.const import DATA_KEY, SIGNAL_FF_SUPPRESS_UPDATE
from custom_components.tasmota_irhvac.pi_controller import PIExtraStoredData

from .conftest import get_climate_entity, make_mqtt_state_payload


class TestPIClamping:
    """Regression tests for °F/°C limit bugs."""

    @pytest.mark.asyncio
    async def test_pi_limits_are_celsius(self, hass, setup_pi_integration):
        """PI min/max limits must be in °C regardless of entity display unit."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        assert entity._pi._min_temp_c == 16.0
        assert entity._pi._max_temp_c == 30.0

    @pytest.mark.asyncio
    async def test_pi_clamping_uses_celsius(self, hass, setup_pi_integration):
        """Setpoint clamping must use °C limits, not display unit limits.

        Regression: previously clamped to 61-86 (°F) instead of 16-30 (°C).
        """
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Set up a scenario where PI computes a valid 24°C setpoint
        pi._desired_temp = 24.0
        pi._hp_setpoint = 22.0
        pi._pi_integral = 0.0
        pi._pi_last_tick_time = 0  # Deterministic dt_factor
        entity._attr_current_temperature = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        await pi._pi_tick()

        # Setpoint should be > 22 (P term pushes up), NOT clamped to min
        assert pi._hp_setpoint >= 22, (
            f"Setpoint clamped incorrectly to {pi._hp_setpoint}, "
            f"limits should be {pi._min_temp_c}-{pi._max_temp_c}"
        )

    @pytest.mark.asyncio
    async def test_pi_2dof_formula(self, hass, setup_pi_integration):
        """2-DOF P term must be weight * (desired - current), not weight*desired - current.

        Regression: operator precedence bug computed (0.5 * 22) - 20 = -9 instead of
        0.5 * (22 - 20) = 1.
        """
        entry = await setup_pi_integration({"pi_setpoint_weight": 0.5})
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._desired_temp = 22.0
        pi._hp_setpoint = 21.0
        pi._pi_integral = 0.0
        pi._pi_ki = 0.0  # Isolate P term
        pi._ff_offset = 0.0
        pi._pi_last_tick_time = 0
        entity._attr_current_temperature = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        await pi._pi_tick()

        # Correct formula: P positive (room too cold), setpoint should go UP
        # Wrong formula: P = -9, setpoint would drop to min
        assert pi._hp_setpoint >= 21, (
            f"P term pushed setpoint wrong direction: {pi._hp_setpoint}"
        )


class TestPICompanionSensors:
    """Tests for PI companion sensor entities updating via dispatcher."""

    @pytest.mark.asyncio
    async def test_sensor_entities_update_after_tick(self, hass, setup_pi_integration):
        """Companion sensors should update after a PI tick."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Set up conditions for a PI tick that changes hp_setpoint
        pi._desired_temp = 24.0
        pi._hp_setpoint = 22.0
        pi._pi_integral = 0.0
        pi._pi_last_tick_time = 0
        entity._attr_current_temperature = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        await pi._pi_tick()
        await hass.async_block_till_done()

        # Find the hp_setpoint sensor
        all_sensors = hass.states.async_all("sensor")
        hp_sensor = next(
            (s for s in all_sensors if "hp_setpoint" in s.entity_id), None
        )
        assert hp_sensor is not None, "hp_setpoint sensor not found"
        # Sensor should reflect the updated setpoint
        assert hp_sensor.state != "unknown", f"Sensor still unknown: {hp_sensor}"

    @pytest.mark.asyncio
    async def test_binary_sensor_reflects_suppress(self, hass, setup_pi_integration):
        """FF learning binary sensor should show Problem when manually suppressed."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        # Manually suppress learning (set both the flag and the state the binary sensor reads)
        entity._pi._manual_ff_suppress = True
        entity._pi._manual_ff_suppress_reason = "testing"
        entity._pi._disturbance_suppress_active = True

        # Fire the FF suppress signal so binary sensor re-reads state
        async_dispatcher_send(
            hass,
            SIGNAL_FF_SUPPRESS_UPDATE.format(entry.entry_id),
        )
        await hass.async_block_till_done()

        # Find the ff_learning binary sensor
        all_binary = hass.states.async_all("binary_sensor")
        ff_sensor = next(
            (s for s in all_binary if "ff_learning" in s.entity_id), None
        )
        assert ff_sensor is not None, "ff_learning binary sensor not found"
        # "on" means problem detected (learning suppressed)
        assert ff_sensor.state == "on", f"Expected 'on' (suppressed), got {ff_sensor.state}"


class TestPISensorTrigger:
    """Tests for PI reacting to sensor state changes."""

    @pytest.mark.asyncio
    async def test_room_temp_change_triggers_tick(self, hass, setup_pi_integration):
        """Changing room temp sensor should trigger PI recalculation."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Set up heating mode
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._pi_integral = 0.0
        pi._pi_last_tick_time = 0

        old_setpoint = pi._hp_setpoint

        # Change room temp — this should trigger _async_sensor_changed → pi_tick
        hass.states.async_set(
            "sensor.room_temp", "18.0",
            {"unit_of_measurement": "°C"},
        )
        await hass.async_block_till_done()

        # PI should have reacted to the 4°C error
        # (setpoint may or may not change depending on hysteresis, but integral should grow)
        assert pi._pi_integral > 0 or pi._hp_setpoint > old_setpoint, (
            f"PI didn't react to sensor change: integral={pi._pi_integral}, "
            f"setpoint={pi._hp_setpoint}"
        )


class TestPIMQTTEcho:
    """Tests for MQTT echo handling with PI active."""

    @pytest.mark.asyncio
    async def test_mqtt_echo_preserves_desired_temp(self, hass, setup_pi_integration):
        """MQTT state echo should not overwrite user's desired temperature.

        Regression: base class _handle_state_payload sets _attr_target_temperature
        to the HP's whole-°C setpoint. PI must restore _desired_temp afterward.
        """
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        import time as _time
        # User sets desired temp, PI sends command
        pi._desired_temp = 21.5
        entity._attr_target_temperature = 21.5
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._last_send_ir_time = _time.monotonic()

        # HP echoes back a different temp (its whole-°C setpoint)
        payload = make_mqtt_state_payload({"Temp": 24, "Mode": "Heat"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        # desired_temp should be preserved
        assert pi._desired_temp == 21.5, (
            f"MQTT echo overwrote desired_temp: {pi._desired_temp}"
        )
        assert entity.target_temperature == 21.5, (
            f"MQTT echo overwrote target_temperature: {entity.target_temperature}"
        )


class TestExtraStoredData:
    """Tests for PIExtraStoredData persistence."""

    def test_round_trip(self):
        """Serialize and deserialize PIExtraStoredData."""
        data = PIExtraStoredData(
            pi_integral=5.67,
            desired_temp=22.0,
            hp_setpoint=23.0,
        )
        serialized = data.as_dict()
        restored = PIExtraStoredData.from_dict(serialized)

        assert restored is not None
        assert restored.pi_integral == 5.67
        assert restored.desired_temp == 22.0
        assert restored.hp_setpoint == 23.0

    def test_from_dict_invalid_returns_none(self):
        """Invalid data should return None, not crash."""
        assert PIExtraStoredData.from_dict({}) is None
        assert PIExtraStoredData.from_dict({"pi_integral": "bad"}) is None
        assert PIExtraStoredData.from_dict(None) is None

    def test_from_dict_with_legacy_bucket_fields(self):
        """Old stored data with bucket fields should load gracefully."""
        legacy_data = {
            "ff_heat_buckets": {"0": 1.5},
            "ff_cool_buckets": {},
            "pi_integral": 3.0,
            "desired_temp": 22.0,
            "hp_setpoint": 23.0,
            "ff_bucket_observation_counts": {"0": 5},
        }
        restored = PIExtraStoredData.from_dict(legacy_data)
        assert restored is not None
        assert restored.pi_integral == 3.0

    @pytest.mark.asyncio
    async def test_entity_provides_extra_stored_data(self, hass, setup_pi_integration):
        """Climate entity should provide PIExtraStoredData for RestoreEntity."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        data = entity.extra_restore_state_data
        assert data is not None
        assert isinstance(data, PIExtraStoredData)
        assert hasattr(data, "pi_integral")
