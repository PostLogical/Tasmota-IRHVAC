"""Tier 7: Fujitsu-specific behavior tests.

NOTE: These tests were written for the pre-handler subclass architecture
(FujitsuTasmotaIrhvac). They access entity._economy, entity._powerful, etc.
which now live on entity._vendor_handler. Equivalent behavior is covered by:
  - tests/test_vendor_fujitsu.py (handler unit tests)
  - tests/test_payload_snapshots.py (end-to-end payload behavior)

Skipped since architecture-rework; preserved for old-branch validation.
"""

import json
import pytest

pytestmark = pytest.mark.skip(reason="pre-handler architecture — see test_vendor_fujitsu.py")

from homeassistant.components.climate.const import (
    HVACMode,
    PRESET_NONE,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)

from pytest_homeassistant_custom_component.common import async_fire_mqtt_message

from custom_components.tasmota_irhvac.const import (
    PRESET_ECONO,
    PRESET_MIN_HEAT,
    PRESET_POWERFUL,
)
from custom_components.tasmota_irhvac.fujitsu import (
    FUJITSU_DATA_ECONO,
    FUJITSU_DATA_MIN_HEAT,
    FUJITSU_DATA_POWERFUL,
    FUJITSU_DATA_SET_H,
    FUJITSU_DATA_SET_V,
    FUJITSU_MODEL_3,
)

from .conftest import get_climate_entity, make_mqtt_state_payload


def _make_56bit_payload(data_hex, model=FUJITSU_MODEL_3, bits=56):
    """Build MQTT payload with 56-bit Fujitsu preset data."""
    return json.dumps({
        "IrReceived": {
            "IRHVAC": {
                "Vendor": "FUJITSU_AC", "Model": model,
                "Power": "On", "Mode": "Heat", "Temp": 22,
                "Celsius": "On", "FanSpeed": "Auto",
                "SwingV": "Auto", "SwingH": "Off",
                "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
                "Light": "Off", "Filter": "Off", "Clean": "Off",
                "Beep": "Off", "Sleep": "-1",
            },
            "Data": data_hex,
            "Bits": bits,
        }
    })


class TestFujitsuEntity:
    """Tests for Fujitsu entity creation."""

    @pytest.mark.asyncio
    async def test_fujitsu_entity_type(self, hass, setup_integration):
        """Fujitsu vendor should create FujitsuTasmotaIrhvac."""
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        assert type(entity).__name__ == "FujitsuTasmotaIrhvac"


class TestFujitsu56BitDetection:
    """Tests for 56-bit IR preset detection from MQTT."""

    @pytest.mark.asyncio
    async def test_detect_powerful(self, hass, setup_integration):
        """FUJITSU_DATA_POWERFUL should set Powerful preset."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        payload = _make_56bit_payload(FUJITSU_DATA_POWERFUL)
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_preset_mode == PRESET_POWERFUL
        assert entity._powerful is True

    @pytest.mark.asyncio
    async def test_detect_econo(self, hass, setup_integration):
        """FUJITSU_DATA_ECONO should set Economy preset."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        payload = _make_56bit_payload(FUJITSU_DATA_ECONO)
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_preset_mode == PRESET_ECONO
        assert entity._economy is True

    @pytest.mark.asyncio
    async def test_detect_min_heat(self, hass, setup_integration):
        """FUJITSU_DATA_MIN_HEAT should set Min Heat preset."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        # Min heat uses full-length data, not 56 bits
        payload = json.dumps({
            "IrReceived": {
                "IRHVAC": {
                    "Vendor": "FUJITSU_AC", "Model": FUJITSU_MODEL_3,
                    "Power": "On", "Mode": "Heat", "Temp": 22,
                    "Celsius": "On", "FanSpeed": "Auto",
                    "SwingV": "Auto", "SwingH": "Off",
                    "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
                    "Light": "Off", "Filter": "Off", "Clean": "On",
                    "Beep": "Off", "Sleep": "-1",
                },
                "Data": FUJITSU_DATA_MIN_HEAT,
            }
        })
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._min_heat is True
        assert entity._attr_preset_mode == PRESET_MIN_HEAT

    @pytest.mark.asyncio
    async def test_detect_set_v_from_both(self, hass, setup_integration):
        """FUJITSU_DATA_SET_V from BOTH should go to HORIZONTAL."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_swing_mode = SWING_BOTH

        payload = _make_56bit_payload(FUJITSU_DATA_SET_V)
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        # Set V stops vertical oscillation: BOTH → HORIZONTAL
        assert entity._attr_swing_mode == SWING_HORIZONTAL

    @pytest.mark.asyncio
    async def test_detect_set_v_from_vertical(self, hass, setup_integration):
        """Set V from VERTICAL should go to OFF."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_swing_mode = SWING_VERTICAL

        payload = _make_56bit_payload(FUJITSU_DATA_SET_V)
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_OFF

    @pytest.mark.asyncio
    async def test_detect_set_h(self, hass, setup_integration):
        """FUJITSU_DATA_SET_H should update swing state."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_swing_mode = SWING_BOTH

        payload = _make_56bit_payload(FUJITSU_DATA_SET_H)
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        # Set H from BOTH → VERTICAL (removed horizontal oscillation)
        assert entity._attr_swing_mode == SWING_VERTICAL

    @pytest.mark.asyncio
    async def test_detect_set_h_from_horizontal(self, hass, setup_integration):
        """Set H from HORIZONTAL should go to OFF."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_swing_mode = SWING_HORIZONTAL

        payload = _make_56bit_payload(FUJITSU_DATA_SET_H)
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        # Set H stops horizontal oscillation: HORIZONTAL → OFF
        assert entity._attr_swing_mode == SWING_OFF


class TestFujitsuPresets:
    """Tests for Fujitsu preset activation/deactivation."""

    @pytest.mark.asyncio
    async def test_activate_powerful(self, hass, setup_integration):
        """Activating Powerful preset should set turbo on."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await entity.async_set_preset_mode(PRESET_POWERFUL)

        assert entity._powerful is True
        assert entity._attr_preset_mode == PRESET_POWERFUL

    @pytest.mark.asyncio
    async def test_activate_econo(self, hass, setup_integration):
        """Activating Economy preset should set econo on."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await entity.async_set_preset_mode(PRESET_ECONO)

        assert entity._economy is True
        assert entity._attr_preset_mode == PRESET_ECONO

    @pytest.mark.asyncio
    async def test_activate_min_heat(self, hass, setup_integration):
        """Activating Min Heat should save temp and pause PI."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_target_temperature = 22

        await entity.async_set_preset_mode(PRESET_MIN_HEAT)

        assert entity._min_heat is True
        assert entity._attr_preset_mode == PRESET_MIN_HEAT
        assert entity._saved_target_temp == 22

    @pytest.mark.asyncio
    async def test_deactivate_min_heat_restores_temp(self, hass, setup_integration):
        """Leaving Min Heat should restore saved temp."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_target_temperature = 22

        await entity.async_set_preset_mode(PRESET_MIN_HEAT)
        assert entity._min_heat is True

        # Switch to none — should restore temp
        await entity.async_set_preset_mode(PRESET_NONE)
        assert entity._min_heat is False
        assert entity._attr_target_temperature == 22

    @pytest.mark.asyncio
    async def test_switching_from_econo_to_powerful(self, hass, setup_integration):
        """Switching from Economy to Powerful should clear econo."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await entity.async_set_preset_mode(PRESET_ECONO)
        assert entity._economy is True

        await entity.async_set_preset_mode(PRESET_POWERFUL)
        assert entity._economy is False
        assert entity._powerful is True

    @pytest.mark.asyncio
    async def test_vendor_mismatch_ignored(self, hass, setup_integration):
        """MQTT from wrong vendor should be ignored by Fujitsu handler."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.OFF

        payload = json.dumps({"IRHVAC": {
            "Vendor": "MITSUBISHI_AC", "Power": "On", "Mode": "Heat", "Temp": 25,
            "Celsius": "On", "FanSpeed": "Auto", "SwingV": "Off", "SwingH": "Off",
            "Quiet": "Off", "Turbo": "Off", "Econo": "Off", "Light": "Off",
            "Filter": "Off", "Clean": "Off", "Beep": "Off", "Sleep": "-1",
        }})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        # Should still be off
        assert entity._attr_hvac_mode == HVACMode.OFF
