"""Fujitsu vendor handler integration tests.

Verifies the FujitsuHandler is correctly wired into the climate entity
and produces the right end-to-end behavior through real HA machinery.

Unit tests for the handler itself are in test_vendor_fujitsu.py.
"""

import json
from unittest.mock import patch, AsyncMock

import pytest

from homeassistant.components.climate.const import (
    HVACMode,
    PRESET_AWAY,
    PRESET_NONE,
)
from homeassistant.const import STATE_ON

from pytest_homeassistant_custom_component.common import async_fire_mqtt_message

from homeassistant.components.climate.const import PRESET_BOOST, PRESET_ECO
from custom_components.tasmota_irhvac.const import PRESET_MIN_HEAT
from custom_components.tasmota_irhvac.vendors.fujitsu import (
    FUJITSU_DATA_ECONO,
    FUJITSU_DATA_MIN_HEAT,
    FUJITSU_DATA_POWERFUL,
    FUJITSU_DATA_SET_V,
    FUJITSU_IR_ECONO,
    FUJITSU_IR_MIN_HEAT,
    FUJITSU_IR_POWERFUL,
    FUJITSU_MODEL_3,
    FujitsuHandler,
)

from .conftest import get_climate_entity


def _fujitsu_mqtt_payload(overrides=None, json_extra=None):
    """Build a Fujitsu MQTT state payload with optional json-level extras."""
    payload = {
        "Vendor": "FUJITSU_AC",
        "Power": "On",
        "Mode": "Heat",
        "Temp": 22,
        "Celsius": "On",
        "FanSpeed": "Auto",
        "SwingV": "Auto",
        "SwingH": "Off",
        "Quiet": "Off",
        "Turbo": "Off",
        "Econo": "Off",
        "Light": "Off",
        "Filter": "Off",
        "Clean": "Off",
        "Beep": "Off",
        "Sleep": "-1",
    }
    if overrides:
        payload.update(overrides)
    wrapper = {"IRHVAC": payload}
    if json_extra:
        wrapper.update(json_extra)
    return json.dumps(wrapper)


# ── Handler wiring ───────────────────────────────────────────────────


class TestFujitsuHandlerWiring:
    """Verify FujitsuHandler is composed into the entity."""

    @pytest.mark.asyncio
    async def test_fujitsu_vendor_uses_handler(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        assert isinstance(entity._vendor_handler, FujitsuHandler)

    @pytest.mark.asyncio
    async def test_non_fujitsu_uses_default(self, hass, setup_integration):
        from custom_components.tasmota_irhvac.vendors.base import VendorHandler
        entry = await setup_integration({"vendor": "DAIKIN"})
        entity = get_climate_entity(hass, entry)
        assert type(entity._vendor_handler) is VendorHandler


# ── Preset modes via handler ─────────────────────────────────────────


class TestFujitsuPresetIntegration:
    """Verify Fujitsu presets work end-to-end through the entity."""

    @pytest.mark.asyncio
    async def test_powerful_sends_raw_ir(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        with patch(
            "homeassistant.components.mqtt.async_publish", new_callable=AsyncMock
        ) as mock_pub:
            await entity.async_set_preset_mode(PRESET_BOOST)

            # Should have sent the raw IR code (not a regular IRHVAC payload)
            calls = [c[0][2] for c in mock_pub.await_args_list]
            assert any(FUJITSU_IR_POWERFUL in str(c) for c in calls)

        assert entity._attr_preset_mode == PRESET_BOOST

    @pytest.mark.asyncio
    async def test_econo_sends_raw_ir(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        with patch(
            "homeassistant.components.mqtt.async_publish", new_callable=AsyncMock
        ) as mock_pub:
            await entity.async_set_preset_mode(PRESET_ECO)

            calls = [c[0][2] for c in mock_pub.await_args_list]
            assert any(FUJITSU_IR_ECONO in str(c) for c in calls)

        assert entity._attr_preset_mode == PRESET_ECO

    @pytest.mark.asyncio
    async def test_min_heat_forces_heat_mode(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.COOL
        entity._attr_target_temperature = 25
        entity.power_mode = STATE_ON

        with patch(
            "homeassistant.components.mqtt.async_publish", new_callable=AsyncMock
        ):
            await entity.async_set_preset_mode(PRESET_MIN_HEAT)

        assert entity._attr_preset_mode == PRESET_MIN_HEAT
        assert entity._attr_hvac_mode == HVACMode.HEAT
        assert entity.power_mode == "on"

    @pytest.mark.asyncio
    async def test_preset_none_clears_and_falls_through(self, hass, setup_integration):
        """PRESET_NONE should clear Fujitsu flags and fall through to base."""
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        # First set a preset
        with patch(
            "homeassistant.components.mqtt.async_publish", new_callable=AsyncMock
        ):
            await entity.async_set_preset_mode(PRESET_ECO)
        assert entity._attr_preset_mode == PRESET_ECO

        # Now clear it
        with patch(
            "homeassistant.components.mqtt.async_publish", new_callable=AsyncMock
        ):
            await entity.async_set_preset_mode(PRESET_NONE)
        assert entity._attr_preset_mode == PRESET_NONE

    @pytest.mark.asyncio
    async def test_away_still_works_with_fujitsu(self, hass, setup_integration):
        """PRESET_AWAY should fall through to base handling."""
        entry = await setup_integration({"vendor": "FUJITSU_AC", "away_temp": 16})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_target_temperature = 22
        entity.power_mode = STATE_ON

        with patch(
            "homeassistant.components.mqtt.async_publish", new_callable=AsyncMock
        ):
            await entity.async_set_preset_mode(PRESET_AWAY)

        assert entity._attr_preset_mode == PRESET_AWAY
        assert entity._attr_target_temperature == 16


# ── State receive: flag→preset mapping ───────────────────────────────


class TestFujitsuStateReceive:
    """Verify Fujitsu state payload processing through the entity."""

    @pytest.mark.asyncio
    async def test_turbo_flag_sets_powerful_preset(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            _fujitsu_mqtt_payload({"Turbo": "On"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_preset_mode == PRESET_BOOST

    @pytest.mark.asyncio
    async def test_econo_flag_sets_econo_preset(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            _fujitsu_mqtt_payload({"Econo": "On"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_preset_mode == PRESET_ECO

    @pytest.mark.asyncio
    async def test_power_off_clears_presets(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)

        # Set a preset first
        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            _fujitsu_mqtt_payload({"Turbo": "On"}),
        )
        await hass.async_block_till_done()
        assert entity._attr_preset_mode == PRESET_BOOST

        # Power off should clear it
        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            _fujitsu_mqtt_payload({"Power": "Off", "Turbo": "Off"}),
        )
        await hass.async_block_till_done()

        assert entity._vendor_handler.should_pause_controller is False

    @pytest.mark.asyncio
    async def test_56bit_powerful_detected(self, hass, setup_integration):
        """Model 3 56-bit Powerful command should set preset without changing temp."""
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        entity._attr_target_temperature = 22
        entity._attr_hvac_mode = HVACMode.HEAT

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            _fujitsu_mqtt_payload(
                {"Model": FUJITSU_MODEL_3, "Temp": 0},  # 56-bit puts garbage temp
                json_extra={"Data": FUJITSU_DATA_POWERFUL, "Bits": 56},
            ),
        )
        await hass.async_block_till_done()

        assert entity._attr_preset_mode == PRESET_BOOST
        # Temp should be restored to pre-56bit value, not the garbage 0
        assert entity._attr_target_temperature == 22
