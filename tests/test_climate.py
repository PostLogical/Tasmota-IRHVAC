"""Tier 6: Climate entity method tests.

Tests climate entity methods that aren't covered by MQTT or PI integration tests.
"""

import json
import pytest

from homeassistant.components.climate.const import (
    FAN_HIGH,
    HVACAction,
    HVACMode,
    PRESET_AWAY,
    PRESET_NONE,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)
from homeassistant.const import STATE_ON

from pytest_homeassistant_custom_component.common import async_fire_mqtt_message

from .conftest import get_climate_entity, make_mqtt_state_payload


class TestHvacAction:
    """Tests for the hvac_action property."""

    @pytest.mark.asyncio
    async def test_hvac_action_off(self, hass, setup_integration):
        """OFF mode should return HVACAction.OFF."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.OFF
        assert entity.hvac_action == HVACAction.OFF

    @pytest.mark.asyncio
    async def test_hvac_action_heat(self, hass, setup_integration):
        """HEAT mode should return HVACAction.HEATING."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        assert entity.hvac_action == HVACAction.HEATING

    @pytest.mark.asyncio
    async def test_hvac_action_cool(self, hass, setup_integration):
        """COOL mode should return HVACAction.COOLING."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.COOL
        assert entity.hvac_action == HVACAction.COOLING

    @pytest.mark.asyncio
    async def test_hvac_action_dry(self, hass, setup_integration):
        """DRY mode should return HVACAction.DRYING."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.DRY
        assert entity.hvac_action == HVACAction.DRYING

    @pytest.mark.asyncio
    async def test_hvac_action_fan_only(self, hass, setup_integration):
        """FAN_ONLY mode should return HVACAction.FAN."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.FAN_ONLY
        assert entity.hvac_action == HVACAction.FAN


class TestSetTemperature:
    """Tests for async_set_temperature."""

    @pytest.mark.asyncio
    async def test_set_temperature(self, hass, setup_integration):
        """Setting temperature should update target and send IR."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await entity.async_set_temperature(temperature=24)
        assert entity._attr_target_temperature == 24

    @pytest.mark.asyncio
    async def test_set_temperature_none_ignored(self, hass, setup_integration):
        """None temperature should be ignored."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_target_temperature = 22

        await entity.async_set_temperature(temperature=None)
        assert entity._attr_target_temperature == 22

    @pytest.mark.asyncio
    async def test_set_temperature_with_mode(self, hass, setup_integration):
        """Setting temperature with hvac_mode should change both."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_temperature(temperature=24, hvac_mode=HVACMode.COOL)
        assert entity._attr_target_temperature == 24
        assert entity._attr_hvac_mode == HVACMode.COOL


class TestSetHvacMode:
    """Tests for async_set_hvac_mode."""

    @pytest.mark.asyncio
    async def test_set_hvac_mode_heat(self, hass, setup_integration):
        """Setting HEAT mode should update mode and power on."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_hvac_mode(HVACMode.HEAT)
        assert entity._attr_hvac_mode == HVACMode.HEAT
        assert entity.power_mode == STATE_ON

    @pytest.mark.asyncio
    async def test_set_hvac_mode_off(self, hass, setup_integration):
        """Setting OFF mode should power off."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_hvac_mode(HVACMode.OFF)
        assert entity._attr_hvac_mode == HVACMode.OFF


class TestTurnOnOff:
    """Tests for async_turn_on and async_turn_off."""

    @pytest.mark.asyncio
    async def test_turn_on_restores_last_mode(self, hass, setup_integration):
        """Turn on should restore the last non-OFF mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        # Set a mode, then turn off
        entity._last_on_mode = HVACMode.COOL
        await entity.async_turn_on()
        assert entity._attr_hvac_mode == HVACMode.COOL

    @pytest.mark.asyncio
    async def test_turn_on_no_last_mode(self, hass, setup_integration):
        """Turn on with no last mode should use AUTO."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        entity._last_on_mode = None
        await entity.async_turn_on()
        assert entity._attr_hvac_mode == HVACMode.AUTO

    @pytest.mark.asyncio
    async def test_turn_off(self, hass, setup_integration):
        """Turn off should set mode to OFF."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        await entity.async_turn_off()
        assert entity._attr_hvac_mode == HVACMode.OFF


class TestSetFanMode:
    """Tests for async_set_fan_mode."""

    @pytest.mark.asyncio
    async def test_set_valid_fan_mode(self, hass, setup_integration):
        """Setting a valid fan mode should work."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await entity.async_set_fan_mode("low")
        assert entity._attr_fan_mode == "low"

    @pytest.mark.asyncio
    async def test_set_invalid_fan_mode(self, hass, setup_integration):
        """Setting an invalid fan mode should be rejected."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_fan_mode = "auto"

        await entity.async_set_fan_mode("nonexistent")
        assert entity._attr_fan_mode == "auto"  # Unchanged


class TestSetSwingMode:
    """Tests for async_set_swing_mode."""

    @pytest.mark.asyncio
    async def test_set_valid_swing_mode(self, hass, setup_integration):
        """Setting a valid swing mode should work."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await entity.async_set_swing_mode(SWING_VERTICAL)
        assert entity._attr_swing_mode == SWING_VERTICAL

    @pytest.mark.asyncio
    async def test_set_invalid_swing_mode(self, hass, setup_integration):
        """Setting an invalid swing mode should be rejected."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_OFF

        await entity.async_set_swing_mode("nonexistent")
        assert entity._attr_swing_mode == SWING_OFF  # Unchanged


class TestSetSwingV:
    """Tests for async_set_swingv service."""

    @pytest.mark.asyncio
    async def test_set_swingv_from_both(self, hass, setup_integration):
        """Setting swingv non-auto from SWING_BOTH should change to HORIZONTAL."""
        entry = await setup_integration({"default_swingv": "highest"})
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_BOTH

        await entity.async_set_swingv(swingv="highest", state_mode="SendStore")
        assert entity._swingv == "highest"
        # Should switch from BOTH to HORIZONTAL (removed vertical oscillation)
        assert entity._attr_swing_mode == SWING_HORIZONTAL

    @pytest.mark.asyncio
    async def test_set_swingv_auto_from_horizontal(self, hass, setup_integration):
        """Setting swingv auto from HORIZONTAL should change to BOTH."""
        entry = await setup_integration({"default_swingv": "auto"})
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_HORIZONTAL

        await entity.async_set_swingv(swingv="auto", state_mode="SendStore")
        assert entity._attr_swing_mode == SWING_BOTH


class TestPresetModes:
    """Tests for preset mode handling."""

    @pytest.mark.asyncio
    async def test_set_preset_away(self, hass, setup_integration):
        """Setting AWAY preset should set away temp."""
        # Use non-Fujitsu vendor to test base class preset logic
        entry = await setup_integration({"vendor": "MITSUBISHI_AC", "away_temp": 16})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_target_temperature = 22

        await entity.async_set_preset_mode(PRESET_AWAY)
        assert entity._is_away is True
        assert entity._attr_target_temperature == 16

    @pytest.mark.asyncio
    async def test_clear_preset_away(self, hass, setup_integration):
        """Setting NONE preset should restore saved temp."""
        # Use non-Fujitsu vendor to test base class preset logic
        entry = await setup_integration({"vendor": "MITSUBISHI_AC", "away_temp": 16})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_target_temperature = 22

        # Go away
        await entity.async_set_preset_mode(PRESET_AWAY)
        assert entity._attr_target_temperature == 16

        # Come back
        await entity.async_set_preset_mode(PRESET_NONE)
        assert entity._is_away is False
        assert entity._attr_target_temperature == 22


class TestExtraStateAttributes:
    """Tests for extra state attributes."""

    @pytest.mark.asyncio
    async def test_extra_attributes_contain_irhvac(self, hass, setup_integration):
        """Extra attributes should contain IRHVAC-specific fields."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        attrs = entity.extra_state_attributes
        assert "econo" in attrs
        assert "turbo" in attrs
        assert "quiet" in attrs
        assert "light" in attrs

    @pytest.mark.asyncio
    async def test_extra_attributes_with_pi(self, hass, setup_pi_integration):
        """Extra attributes should include PI state when PI enabled."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        attrs = entity.extra_state_attributes
        assert "hp_setpoint" in attrs
        assert "pi_integral" in attrs
        assert "desired_temp" in attrs
        assert "ff_offset" in attrs


class TestPrecision:
    """Tests for the precision property."""

    @pytest.mark.asyncio
    async def test_precision_from_config(self, hass, setup_integration):
        """Precision should use configured value."""
        entry = await setup_integration({"precision": 0.5})
        entity = get_climate_entity(hass, entry)
        assert entity.precision == 0.5

    @pytest.mark.asyncio
    async def test_precision_default(self, hass, setup_integration):
        """Default precision should be 1.0."""
        entry = await setup_integration({"precision": 1.0})
        entity = get_climate_entity(hass, entry)
        assert entity.precision == 1.0


class TestSensorTracking:
    """Tests for temperature and humidity sensor tracking."""

    @pytest.mark.asyncio
    async def test_temp_sensor_updates_current_temperature(self, hass, setup_integration):
        """Temperature sensor state change should update current_temperature."""
        hass.states.async_set(
            "sensor.test_temp", "21.5",
            {"unit_of_measurement": "°C"},
        )
        entry = await setup_integration({"temperature_sensor": "sensor.test_temp"})
        entity = get_climate_entity(hass, entry)

        # Verify initial temp was picked up
        assert entity._attr_current_temperature is not None

        # Change sensor state
        hass.states.async_set(
            "sensor.test_temp", "23.0",
            {"unit_of_measurement": "°C"},
        )
        await hass.async_block_till_done()

        assert entity._attr_current_temperature == 23.0

    @pytest.mark.asyncio
    async def test_humidity_sensor_updates(self, hass, setup_integration):
        """Humidity sensor state change should update current_humidity."""
        hass.states.async_set(
            "sensor.test_humid", "45",
            {"unit_of_measurement": "%"},
        )
        entry = await setup_integration({"humidity_sensor": "sensor.test_humid"})
        entity = get_climate_entity(hass, entry)

        assert entity._attr_current_humidity is not None

        hass.states.async_set(
            "sensor.test_humid", "55",
            {"unit_of_measurement": "%"},
        )
        await hass.async_block_till_done()

        assert entity._attr_current_humidity == 55.0
