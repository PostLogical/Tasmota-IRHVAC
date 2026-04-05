"""Tests for config entry migration.

Tests the clean migration chain (v1.1 → v1.3).
Beta migrations (v1.2–v1.6, v1.9–v1.11) were stripped.
"""

import pytest

from homeassistant.const import UnitOfTemperature
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.__init__ import MINOR_VERSION, async_migrate_entry

from .conftest import make_config


class TestMigrationTempUnit:
    """v1.2: Convert stored temps from celsius_mode unit to system unit."""

    @pytest.mark.asyncio
    async def test_celsius_to_fahrenheit_system(self, hass):
        """Temps stored in °C should convert to °F for a °F system."""
        from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM
        hass.config.units = US_CUSTOMARY_SYSTEM

        config = make_config({
            "celsius_mode": "on",  # IR protocol uses °C
            "min_temp": 16,        # Stored as °C (pre-migration)
            "max_temp": 30,
            "target_temp": 22,
            "away_temp": 16,
        })
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == MINOR_VERSION

        # Temps should now be in °F
        assert entry.options["min_temp"] == pytest.approx(60.8, abs=0.1)
        assert entry.options["max_temp"] == pytest.approx(86.0, abs=0.1)
        assert entry.options["target_temp"] == pytest.approx(71.6, abs=0.1)
        assert entry.options["away_temp"] == pytest.approx(60.8, abs=0.1)

    @pytest.mark.asyncio
    async def test_celsius_system_noop(self, hass):
        """Temps on a °C system with celsius_mode=on should not change."""
        from homeassistant.util.unit_system import METRIC_SYSTEM
        hass.config.units = METRIC_SYSTEM

        config = make_config({
            "celsius_mode": "on",
            "min_temp": 16,
            "max_temp": 30,
            "target_temp": 22,
        })
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True

        assert entry.options["min_temp"] == 16
        assert entry.options["max_temp"] == 30
        assert entry.options["target_temp"] == 22

    @pytest.mark.asyncio
    async def test_away_temp_none_skipped(self, hass):
        """away_temp=None should not crash the migration."""
        from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM
        hass.config.units = US_CUSTOMARY_SYSTEM

        config = make_config({"celsius_mode": "on", "away_temp": None})
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.options.get("away_temp") is None


class TestMigrationToggleCase:
    """v1.3: Normalize toggle values to lowercase."""

    @pytest.mark.asyncio
    async def test_capitalised_toggles_lowered(self, hass):
        config = make_config({
            "quiet": "Off",
            "turbo": "On",
            "celsius_mode": "On",
        })
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=2,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == MINOR_VERSION

        assert entry.options["quiet"] == "off"
        assert entry.options["turbo"] == "on"
        assert entry.options["celsius_mode"] == "on"


class TestMigrationPrecisionCoerce:
    """precision/temp_step should be coerced from string to float."""

    @pytest.mark.asyncio
    async def test_string_precision_coerced(self, hass):
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
        assert entry.options["temp_step"] == 1.0
        assert isinstance(entry.options["temp_step"], float)

    @pytest.mark.asyncio
    async def test_float_precision_unchanged(self, hass):
        config = make_config({"precision": 0.5, "temp_step": 1.0})
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.options["precision"] == 0.5


class TestMigrationCurrentVersion:
    """Entry already at current version should be a noop."""

    @pytest.mark.asyncio
    async def test_current_version_noop(self, hass):
        config = make_config()
        entry = MockConfigEntry(
            domain=DOMAIN, data={}, options=config,
            title="Test", version=1, minor_version=MINOR_VERSION,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == MINOR_VERSION
