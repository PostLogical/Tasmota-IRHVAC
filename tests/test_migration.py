"""Tests for config entry migration.

Single migration v1.1 → v1.2 (temp unit conversion + toggle normalization).
"""

import pytest

from homeassistant.const import UnitOfTemperature
from homeassistant.util.unit_system import METRIC_SYSTEM, US_CUSTOMARY_SYSTEM
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.__init__ import MINOR_VERSION, async_migrate_entry

from .conftest import make_config


class TestMigrationV1:
    """v1.1 → v1.2: temp unit conversion, toggle normalization, precision coerce."""

    @pytest.mark.asyncio
    async def test_celsius_to_fahrenheit_system(self, hass):
        """Temps stored in °C should convert to °F for a °F system."""
        hass.config.units = US_CUSTOMARY_SYSTEM

        config = make_config({"min_temp": 16, "max_temp": 30, "target_temp": 22, "away_temp": 16})
        # Simulate pre-migration config with legacy celsius_mode key
        config["celsius_mode"] = "on"
        config.pop("ir_protocol_unit", None)
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == MINOR_VERSION

        assert entry.options["min_temp"] == pytest.approx(60.8, abs=0.1)
        assert entry.options["max_temp"] == pytest.approx(86.0, abs=0.1)
        assert entry.options["target_temp"] == pytest.approx(71.6, abs=0.1)
        assert entry.options["away_temp"] == pytest.approx(60.8, abs=0.1)

    @pytest.mark.asyncio
    async def test_celsius_system_noop(self, hass):
        """Temps on a °C system with celsius_mode=on should not change."""
        hass.config.units = METRIC_SYSTEM

        config = make_config({"min_temp": 16, "max_temp": 30, "target_temp": 22})
        config["celsius_mode"] = "on"
        config.pop("ir_protocol_unit", None)
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.options["min_temp"] == 16
        assert entry.options["max_temp"] == 30

    @pytest.mark.asyncio
    async def test_away_temp_none_skipped(self, hass):
        """away_temp=None should not crash the migration."""
        hass.config.units = US_CUSTOMARY_SYSTEM

        config = make_config({"away_temp": None})
        config["celsius_mode"] = "on"
        config.pop("ir_protocol_unit", None)
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.options.get("away_temp") is None

    @pytest.mark.asyncio
    async def test_toggles_lowercased(self, hass):
        """Capitalised toggle values should be normalized to lowercase."""
        config = make_config({
            "quiet": "Off",
            "turbo": "On",
        })
        # Inject legacy celsius_mode key (pre-migration)
        config["celsius_mode"] = "On"
        config.pop("ir_protocol_unit", None)
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.options["quiet"] == "off"
        assert entry.options["turbo"] == "on"

    @pytest.mark.asyncio
    async def test_celsius_mode_renamed_to_ir_protocol_unit(self, hass):
        """celsius_mode 'on'/'off' should become ir_protocol_unit 'celsius'/'fahrenheit'."""
        config = make_config()
        config["celsius_mode"] = "on"
        config.pop("ir_protocol_unit", None)
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert "celsius_mode" not in entry.options
        assert entry.options["ir_protocol_unit"] == "celsius"

    @pytest.mark.asyncio
    async def test_celsius_mode_off_becomes_fahrenheit(self, hass):
        config = make_config()
        config["celsius_mode"] = "off"
        config.pop("ir_protocol_unit", None)
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.options["ir_protocol_unit"] == "fahrenheit"

    @pytest.mark.asyncio
    async def test_precision_coerced_from_string(self, hass):
        """String precision/temp_step should be coerced to float."""
        config = make_config({"precision": "0.5", "temp_step": "1.0"})
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.options["precision"] == 0.5
        assert isinstance(entry.options["precision"], float)

    @pytest.mark.asyncio
    async def test_current_version_noop(self, hass):
        """Entry already at current version should not be modified."""
        config = make_config()
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=MINOR_VERSION,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == MINOR_VERSION
