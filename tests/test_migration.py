"""Tests for config entry migration."""

import pytest

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.__init__ import async_migrate_entry

from .conftest import make_config


class TestConfigMigration:
    """Tests for async_migrate_entry."""

    @pytest.mark.asyncio
    async def test_migrate_v1_1_to_v1_2(self, hass):
        """Migration from v1.1 should add bias_entity and setpoint_weight."""
        config = make_config()
        # Remove the keys that v1.2 adds
        config.pop("pi_ff_bias_entity", None)
        config.pop("pi_setpoint_weight", None)

        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True

        # Verify new keys were added
        assert entry.data.get("pi_ff_bias_entity") == ""
        assert entry.data.get("pi_setpoint_weight") == 1.0
        assert entry.minor_version == 2

    @pytest.mark.asyncio
    async def test_migrate_v1_2_noop(self, hass):
        """Migration of v1.2 entry should be a no-op."""
        config = make_config()  # Already has all keys
        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=2,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == 2

    @pytest.mark.asyncio
    async def test_migrate_preserves_existing_data(self, hass):
        """Migration should not overwrite existing keys."""
        config = make_config()
        config["pi_ff_bias_entity"] = "input_number.my_bias"
        config["pi_setpoint_weight"] = 0.7

        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True

        # setdefault should not overwrite
        assert entry.data["pi_ff_bias_entity"] == "input_number.my_bias"
        assert entry.data["pi_setpoint_weight"] == 0.7
