"""Phase 3a: HA lifecycle tests for architecture rework.

Verifies the new architectural components work end-to-end through real
HA machinery — not just FakePIEntity or mock patches.
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

from homeassistant.components.climate.const import PRESET_BOOST
from custom_components.tasmota_irhvac.const import DATA_KEY
from custom_components.tasmota_irhvac.controller_protocol import ControllerHook, NullController
from custom_components.tasmota_irhvac.pi_controller import PIController
from custom_components.tasmota_irhvac.vendors.base import VendorHandler
from custom_components.tasmota_irhvac.vendors.fujitsu import FujitsuHandler

from .conftest import get_climate_entity, make_mqtt_state_payload


# ── target_temperature property through real HA ──────────────────────


class TestTargetTemperatureProperty:
    """Verify single-writer target_temperature via real entity."""

    @pytest.mark.asyncio
    async def test_without_pi_reads_attr(self, hass, setup_integration):
        """Non-PI entity: target_temperature reads _attr_target_temperature."""
        entry = await setup_integration({"pi_enabled": False})
        entity = get_climate_entity(hass, entry)
        entity._attr_target_temperature = 25.0

        assert entity.target_temperature == 25.0

    @pytest.mark.asyncio
    async def test_with_pi_reads_desired_temp(self, hass, setup_pi_integration):
        """PI entity: target_temperature reads from controller.desired_temp."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._controller

        # PI desired_temp diverges from _attr_target_temperature
        pi._desired_temp = 23.5
        entity._attr_target_temperature = 99.0  # Would be wrong if read directly

        assert entity.target_temperature == 23.5

    @pytest.mark.asyncio
    async def test_pi_set_temperature_updates_property(self, hass, setup_pi_integration):
        """set_temperature through PI should update target_temperature property."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0

        with patch.object(entity, 'send_ir', new_callable=AsyncMock):
            await entity.async_set_temperature(temperature=24.0)

        assert entity.target_temperature == 24.0

    @pytest.mark.asyncio
    async def test_away_preset_updates_property(self, hass, setup_pi_integration):
        """AWAY preset should update target_temperature via desired_temp."""
        entry = await setup_pi_integration({"away_temp": 16})
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity._controller._desired_temp = 22.0

        with patch.object(entity, 'send_ir', new_callable=AsyncMock):
            await entity.async_set_preset_mode(PRESET_AWAY)

        assert entity.target_temperature == 16


# ── ControllerHook Protocol conformance ──────────────────────────────


class TestProtocolConformance:
    """Verify PIController and NullController satisfy ControllerHook."""

    def test_pi_controller_is_controller_hook(self):
        assert isinstance(PIController, type)
        # runtime_checkable Protocol
        # Can't instantiate PIController without entity, so check methods exist
        required = [
            'is_active', 'desired_temp', 'is_tick_running',
            'async_added_to_hass', 'async_will_remove_from_hass',
            'handle_state_payload', 'sensor_changed', 'fire_dispatcher',
            'set_temperature', 'get_ir_temp',
            'filter_hvac_modes', 'should_reject_hvac_mode',
            'pi_pause', 'pi_resume', 'pi_reset_integral',
            'get_extra_state_attributes', 'get_extra_stored_data',
            'async_reset_ff_seeds', 'async_suppress_ff_learning',
            'async_resume_ff_learning',
        ]
        for attr in required:
            assert hasattr(PIController, attr), f"PIController missing {attr}"

    def test_null_controller_is_controller_hook(self):
        nc = NullController()
        assert isinstance(nc, ControllerHook)

    def test_null_controller_defaults(self):
        nc = NullController()
        assert nc.is_active is False
        assert nc.desired_temp is None
        assert nc.is_tick_running is False
        assert nc.filter_hvac_modes(["heat", "cool"]) == ["heat", "cool"]
        assert nc.should_reject_hvac_mode("auto") is False
        assert nc.get_extra_state_attributes() == {}
        assert nc.get_extra_stored_data() is None

    @pytest.mark.asyncio
    async def test_null_controller_async_noop(self):
        nc = NullController()
        # These should all complete without error
        await nc.async_added_to_hass()
        await nc.handle_state_payload({"Temp": 22})
        await nc.sensor_changed(False)
        await nc.set_temperature(22.0)
        await nc.async_reset_ff_seeds()
        await nc.async_suppress_ff_learning("test")
        await nc.async_resume_ff_learning()
        nc.pi_pause()
        nc.pi_resume()
        nc.pi_reset_integral()
        nc.fire_dispatcher()
        nc.async_will_remove_from_hass()


# ── Controller wiring through real entity ────────────────────────────


class TestControllerWiring:
    """Verify _controller is properly wired in real entities."""

    @pytest.mark.asyncio
    async def test_pi_entity_has_pi_controller(self, hass, setup_pi_integration):
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        assert isinstance(entity._controller, PIController)
        assert entity._controller.is_active is True

    @pytest.mark.asyncio
    async def test_non_pi_entity_has_null_controller(self, hass, setup_integration):
        entry = await setup_integration({"pi_enabled": False})
        entity = get_climate_entity(hass, entry)
        assert isinstance(entity._controller, NullController)
        assert entity._controller.is_active is False

    @pytest.mark.asyncio
    async def test_legacy_pi_alias(self, hass, setup_pi_integration):
        """self._pi should be the same object as self._controller for PI entities."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        assert entity._pi is entity._controller

    @pytest.mark.asyncio
    async def test_legacy_pi_alias_none_without_pi(self, hass, setup_integration):
        """self._pi should be None for non-PI entities."""
        entry = await setup_integration({"pi_enabled": False})
        entity = get_climate_entity(hass, entry)
        assert entity._pi is None


# ── Vendor handler wiring through lifecycle ──────────────────────────


class TestVendorHandlerLifecycle:
    """Verify vendor handler survives full entity lifecycle."""

    @pytest.mark.asyncio
    async def test_fujitsu_handler_persists_through_mqtt(self, hass, setup_integration):
        """Fujitsu handler should process MQTT state correctly."""
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)
        assert isinstance(entity._vendor_handler, FujitsuHandler)

        # Send turbo flag via MQTT
        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Turbo": "On", "Power": "On"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_preset_mode == PRESET_BOOST

    @pytest.mark.asyncio
    async def test_default_handler_for_unknown_vendor(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "DAIKIN"})
        entity = get_climate_entity(hass, entry)
        assert type(entity._vendor_handler) is VendorHandler

    @pytest.mark.asyncio
    async def test_config_model_accessible(self, hass, setup_integration):
        """Entity should have been created from IrhvacConfig."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        # The entity was created via IrhvacConfig.from_config_dict in async_setup_entry
        # Verify key config values propagated correctly
        assert entity._vendor == "FUJITSU_AC"
        assert entity._ir_protocol_unit == "celsius"
