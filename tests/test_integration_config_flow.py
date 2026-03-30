"""Tier 4: Config flow and options flow integration tests."""

import pytest

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN

from .conftest import make_config, make_pi_config


class TestConfigFlowUserStep:
    """Tests for the initial user step of config flow."""

    @pytest.mark.asyncio
    async def test_user_step_shows_form(self, hass, enable_custom_integrations):
        """User step should show form when no input provided."""
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"

    @pytest.mark.asyncio
    async def test_user_step_no_vendor_error(self, hass, enable_custom_integrations):
        """User step without vendor should show error."""
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                "name": "Test AC",
                "vendor": "",
                "command_topic": "cmnd/test/irhvac",
                "state_topic": "tele/test/RESULT",
            },
        )
        assert result["type"] == FlowResultType.FORM
        assert "vendor" in result.get("errors", {})

    @pytest.mark.asyncio
    async def test_user_step_advances_to_climate(self, hass, enable_custom_integrations):
        """Valid user step should advance to climate step."""
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                "name": "Test AC",
                "vendor": "FUJITSU_AC",
                "command_topic": "cmnd/test/irhvac",
                "state_topic": "tele/test/RESULT",
            },
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "climate"


class TestConfigFlowFullWizard:
    """Tests for completing the full config flow wizard."""

    @pytest.mark.asyncio
    async def test_full_wizard_creates_entry(self, hass, enable_custom_integrations):
        """Completing all steps should create a config entry."""
        # Step 1: User
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                "name": "Test AC",
                "vendor": "FUJITSU_AC",
                "command_topic": "cmnd/test/irhvac",
                "state_topic": "tele/test/RESULT",
            },
        )
        assert result["step_id"] == "climate"

        # Step 2: Climate (use defaults for most fields)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={},  # All optional with defaults
        )
        assert result["step_id"] == "advanced"

        # Step 3: Advanced (no PI — use defaults)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                "pi_enabled": False,
            },
        )

        # Should create entry (no PI step since pi_enabled=False)
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "Test AC"

    @pytest.mark.asyncio
    async def test_wizard_with_pi_shows_pi_step(self, hass, enable_custom_integrations):
        """Enabling PI in advanced step should show PI controller step."""
        # Step 1: User
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                "name": "Test AC",
                "vendor": "FUJITSU_AC",
                "command_topic": "cmnd/test/irhvac",
                "state_topic": "tele/test/RESULT",
            },
        )

        # Step 2: Climate (defaults)
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={},
        )

        # Step 3: Advanced with PI enabled
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={
                "pi_enabled": True,
            },
        )

        # Should advance to PI step
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "pi_controller"


class TestConfigFlowImport:
    """Tests for YAML import."""

    @pytest.mark.asyncio
    async def test_import_creates_entry(self, hass, enable_custom_integrations):
        """YAML import should create a config entry."""
        config = make_config()
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_IMPORT},
            data=config,
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "Test AC"

    @pytest.mark.asyncio
    async def test_import_duplicate_aborted(self, hass, enable_custom_integrations):
        """Duplicate YAML import (same command_topic) should abort."""
        config = make_config()
        # First import
        await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_IMPORT},
            data=config,
        )
        # Second import with same topic
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_IMPORT},
            data=config,
        )
        assert result["type"] == FlowResultType.ABORT
        assert result["reason"] == "already_configured"

    @pytest.mark.asyncio
    async def test_import_transforms_legacy_keys(self, hass, mqtt_mock, enable_custom_integrations):
        """Import should transform legacy suppress/bias entities to disturbance inputs."""
        config = make_config({
            "pi_enabled": True,
            "pi_ff_suppress_learning_entity": "input_boolean.pellet_stove",
            "temperature_sensor": "sensor.room_temp",
            "outdoor_temp_sensor": "sensor.outdoor_temp",
        })
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_IMPORT},
            data=config,
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        entry = result["result"]
        # Legacy key should be transformed — disturbance_inputs goes to options
        disturbance = entry.options.get("pi_disturbance_inputs", [])
        assert len(disturbance) >= 1, f"No disturbance inputs found. options={entry.options}"
        assert disturbance[0]["entity_id"] == "input_boolean.pellet_stove"


class TestOptionsFlow:
    """Tests for the options flow."""

    @pytest.mark.asyncio
    async def test_options_flow_shows_menu(self, hass, setup_integration):
        """Options flow should show menu on init."""
        entry = await setup_integration()

        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result["type"] == FlowResultType.MENU

    @pytest.mark.asyncio
    async def test_options_flow_temperature_step(self, hass, setup_integration):
        """Temperature options step should accept input."""
        entry = await setup_integration()

        result = await hass.config_entries.options.async_init(entry.entry_id)
        # Select temperature from menu
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "temperature"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "temperature"

        # Submit with defaults (all fields optional)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={},
        )
        # Should return to menu or create entry
        assert result["type"] in (FlowResultType.MENU, FlowResultType.CREATE_ENTRY)


class TestMigration:
    """Tests for config entry migration during setup."""

    @pytest.mark.asyncio
    async def test_migration_v1_1_to_v1_3(self, hass, mqtt_mock, enable_custom_integrations):
        """Config entry at v1.1 should migrate to v1.3 on setup."""
        config = make_config()
        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test AC",
            version=1,
            minor_version=1,
        )
        entry.add_to_hass(hass)

        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.minor_version == 3
