"""Tier 4: Config flow and options flow integration tests."""

import pytest

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.config_flow import TasmotaIrhvacConfigFlow

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

        # Should advance to first PI sub-step (gains)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "pi_gains"


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


class TestOptionsFlowSubSteps:
    """Tests for each options flow sub-step."""

    async def _navigate_to_step(self, hass, entry, step_name):
        """Helper: init options flow and navigate to a sub-step."""
        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result["type"] == FlowResultType.MENU
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": step_name},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == step_name
        return result

    @pytest.mark.asyncio
    async def test_options_mqtt_step(self, hass, setup_integration):
        """MQTT options step should accept input and save."""
        entry = await setup_integration()
        result = await self._navigate_to_step(hass, entry, "mqtt")
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_modes_step(self, hass, setup_integration):
        """Modes options step should accept input and save."""
        entry = await setup_integration()
        result = await self._navigate_to_step(hass, entry, "modes")
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_defaults_step(self, hass, setup_integration):
        """Defaults options step should accept input and save."""
        entry = await setup_integration()
        result = await self._navigate_to_step(hass, entry, "defaults")
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_sensors_step(self, hass, setup_integration):
        """Sensors options step should accept input and save."""
        entry = await setup_integration()
        result = await self._navigate_to_step(hass, entry, "sensors")
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_advanced_step(self, hass, setup_integration):
        """Advanced options step should accept input and save."""
        entry = await setup_integration()
        result = await self._navigate_to_step(hass, entry, "advanced_options")
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_pi_controller_menu(self, hass, setup_integration):
        """PI controller options should show sub-menu."""
        entry = await setup_integration()
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_controller"},
        )
        assert result["type"] == FlowResultType.MENU

    @pytest.mark.asyncio
    async def test_options_pi_gains_step(self, hass, setup_integration):
        """PI gains sub-step should accept input and save."""
        entry = await setup_integration()
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_controller"},
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_gains"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "pi_gains"
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_pi_seeds_step(self, hass, setup_integration):
        """PI seeds sub-step should accept input and save."""
        entry = await setup_integration()
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_controller"},
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_seeds"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "pi_seeds"
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_pi_timing_step(self, hass, setup_integration):
        """PI timing sub-step should accept input and save."""
        entry = await setup_integration()
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_controller"},
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_timing"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "pi_timing"
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_pi_advanced_step(self, hass, setup_integration):
        """PI advanced sub-step should accept input and save."""
        entry = await setup_integration()
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_controller"},
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "pi_advanced"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "pi_advanced"
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], user_input={},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY

    @pytest.mark.asyncio
    async def test_options_ir_actions_menu(self, hass, setup_integration):
        """IR actions should show a management menu."""
        entry = await setup_integration()
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "ir_actions"},
        )
        assert result["type"] == FlowResultType.MENU


class TestDisturbanceInputsSubentryFlow:
    """Tests for disturbance inputs via subentry flow."""

    @pytest.mark.asyncio
    async def test_add_disturbance_input(self, hass, setup_integration):
        """Adding a model input via subentry should create a subentry."""
        entry = await setup_integration()

        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, "model_input"),
            context={"source": "user"},
        )
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"

        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"],
            user_input={
                "name": "Test Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.2,
                "seed_cool": 0.0,
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        model_subs = [
            s for s in entry.subentries.values()
            if s.subentry_type == "model_input"
        ]
        assert len(model_subs) >= 1
        assert model_subs[-1].title == "Test Stove"
        assert model_subs[-1].data["seed_heat"] == 3.2


class TestIRActionsFlow:
    """Tests for IR actions add/remove in options flow."""

    @pytest.mark.asyncio
    async def test_add_ir_action(self, hass, setup_integration):
        """Adding an IR action should save to options."""
        entry = await setup_integration()

        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "ir_actions"},
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "ir_actions_add"},
        )
        assert result["step_id"] == "ir_actions_add"

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                "ir_action_name": "Test Button",
                "ir_action_type": "button",
                "ir_action_code": "raw,0,1234,5678",
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        actions = entry.options.get("ir_actions", [])
        assert len(actions) == 1
        assert actions[0]["name"] == "Test Button"

    @pytest.mark.asyncio
    async def test_remove_ir_action(self, hass, mqtt_mock, enable_custom_integrations):
        """Removing an IR action should update options."""
        config = make_config()
        options = {
            "ir_actions": [{
                "name": "Old Button",
                "type": "button",
                "ir_code": "raw,0,1234",
            }],
        }
        entry = MockConfigEntry(
            domain=DOMAIN, data=config, options=options,
            title="Test AC", version=1, minor_version=3,
        )
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "ir_actions"},
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"next_step_id": "ir_actions_remove"},
        )
        assert result["step_id"] == "ir_actions_remove"

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={"ir_actions_to_remove": ["Old Button"]},
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        actions = entry.options.get("ir_actions", [])
        assert len(actions) == 0


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

        assert entry.minor_version == TasmotaIrhvacConfigFlow.MINOR_VERSION
