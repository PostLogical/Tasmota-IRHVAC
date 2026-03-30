"""Tier 1: Integration tests for entity setup and lifecycle."""

import pytest

from homeassistant.core import HomeAssistant

from custom_components.tasmota_irhvac.const import DATA_KEY, DOMAIN

from .conftest import get_climate_entity, make_config, make_pi_config


class TestEntitySetup:
    """Tests for climate entity creation through real HA setup."""

    @pytest.mark.asyncio
    async def test_setup_creates_climate_entity(self, hass, setup_integration):
        """Config entry setup should create a climate entity in hass.states."""
        entry = await setup_integration()

        # Entity should be in hass.data
        entity = get_climate_entity(hass, entry)
        assert entity is not None

        # Entity should be registered in state machine
        state = hass.states.get(entity.entity_id)
        assert state is not None
        assert state.state == "off"  # initial_operation_mode

    @pytest.mark.asyncio
    async def test_setup_fujitsu_creates_subclass(self, hass, setup_integration):
        """Fujitsu vendor should create FujitsuTasmotaIrhvac instance."""
        entry = await setup_integration({"vendor": "FUJITSU_AC"})

        entity = get_climate_entity(hass, entry)
        assert entity is not None
        assert type(entity).__name__ == "FujitsuTasmotaIrhvac"

    @pytest.mark.asyncio
    async def test_setup_non_fujitsu_creates_base(self, hass, setup_integration):
        """Non-Fujitsu vendor should create base TasmotaIrhvac instance."""
        entry = await setup_integration({"vendor": "MITSUBISHI_AC"})

        entity = get_climate_entity(hass, entry)
        assert entity is not None
        assert type(entity).__name__ == "TasmotaIrhvac"

    @pytest.mark.asyncio
    async def test_entity_unique_id(self, hass, setup_integration):
        """Entity unique_id should match config entry id."""
        entry = await setup_integration()

        entity = get_climate_entity(hass, entry)
        assert entity.unique_id == entry.entry_id

    @pytest.mark.asyncio
    async def test_entity_has_device_info(self, hass, setup_integration):
        """Entity should provide device info for device registry."""
        entry = await setup_integration()

        entity = get_climate_entity(hass, entry)
        device_info = entity.device_info
        assert device_info is not None

    @pytest.mark.asyncio
    async def test_pi_disabled_no_pi_controller(self, hass, setup_integration):
        """PI disabled should not create a PIController."""
        entry = await setup_integration({"pi_enabled": False})

        entity = get_climate_entity(hass, entry)
        assert entity._pi is None


class TestPIEntitySetup:
    """Tests for PI-enabled entity setup."""

    @pytest.mark.asyncio
    async def test_pi_enabled_creates_controller(self, hass, setup_pi_integration):
        """PI enabled should create a PIController."""
        entry = await setup_pi_integration()

        entity = get_climate_entity(hass, entry)
        assert entity._pi is not None
        assert entity._pi._pi_enabled is True

    @pytest.mark.asyncio
    async def test_pi_creates_sensor_entities(self, hass, setup_pi_integration):
        """PI enabled should create companion sensor entities."""
        entry = await setup_pi_integration()

        entity = get_climate_entity(hass, entry)
        entity_id_base = entity.entity_id.replace("climate.", "")

        # Check sensor entities exist in state machine
        # Sensor names depend on translation keys; check hass.states for any PI sensors
        all_states = hass.states.async_all("sensor")
        sensor_ids = [s.entity_id for s in all_states]
        assert any("hp_setpoint" in s for s in sensor_ids), f"No hp_setpoint sensor found in {sensor_ids}"
        assert any("pi_integral" in s for s in sensor_ids), f"No pi_integral sensor found in {sensor_ids}"
        assert any("ff_offset" in s for s in sensor_ids), f"No ff_offset sensor found in {sensor_ids}"

    @pytest.mark.asyncio
    async def test_pi_creates_binary_sensor(self, hass, setup_pi_integration):
        """PI enabled should create FF learning binary sensor."""
        entry = await setup_pi_integration()

        all_states = hass.states.async_all("binary_sensor")
        binary_ids = [s.entity_id for s in all_states]
        assert any("ff_learning" in s for s in binary_ids), f"No ff_learning binary sensor in {binary_ids}"

    @pytest.mark.asyncio
    async def test_pi_disabled_no_sensors(self, hass, setup_integration):
        """PI disabled should not create sensor or binary_sensor entities."""
        entry = await setup_integration({"pi_enabled": False})

        all_sensors = hass.states.async_all("sensor")
        all_binary = hass.states.async_all("binary_sensor")

        pi_sensors = [s for s in all_sensors if "hp_setpoint" in s.entity_id or "pi_integral" in s.entity_id]
        pi_binary = [s for s in all_binary if "ff_learning" in s.entity_id]

        assert len(pi_sensors) == 0
        assert len(pi_binary) == 0


class TestVaneButtonSetup:
    """Tests for vane button entity creation."""

    @pytest.mark.asyncio
    async def test_vane_buttons_created(self, hass, setup_integration):
        """Vane config should create button entities."""
        entry = await setup_integration({
            "has_set_vertical_vane": True,
            "has_set_horizontal_vane": True,
        })

        all_buttons = hass.states.async_all("button")
        button_ids = [s.entity_id for s in all_buttons]
        assert len(button_ids) >= 2, f"Expected 2+ buttons, got {button_ids}"

    @pytest.mark.asyncio
    async def test_no_vane_buttons_by_default(self, hass, setup_integration):
        """Default config should not create vane buttons."""
        entry = await setup_integration()

        all_buttons = hass.states.async_all("button")
        vane_buttons = [s for s in all_buttons if "set_v" in s.entity_id or "set_h" in s.entity_id]
        assert len(vane_buttons) == 0


class TestUnload:
    """Tests for config entry unload."""

    @pytest.mark.asyncio
    async def test_unload_removes_entity(self, hass, setup_integration):
        """Unloading config entry should clean up hass.data."""
        entry = await setup_integration()

        # Verify entity exists
        assert get_climate_entity(hass, entry) is not None

        # Unload
        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

        # Entity should be removed from hass.data
        assert get_climate_entity(hass, entry) is None
