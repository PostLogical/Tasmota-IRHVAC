"""Tests for the diagnostics platform."""

import pytest
from unittest.mock import MagicMock

from homeassistant.const import UnitOfTemperature

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.diagnostics import (
    async_get_config_entry_diagnostics,
    _redact,
)

from .conftest import make_config, make_pi_config
from .test_pi_controller import FakePIEntity


def test_redact_sensitive_keys():
    """Sensitive keys should be redacted."""
    data = {
        "unique_id": "abc123",
        "topic": "cmnd/device/irhvac",
        "name": "Living Room",
        "vendor": "FUJITSU_AC",
    }
    redacted = _redact(data)
    assert redacted["unique_id"] == "**REDACTED**"
    assert redacted["topic"] == "**REDACTED**"
    assert redacted["name"] == "Living Room"
    assert redacted["vendor"] == "FUJITSU_AC"


def test_redact_no_sensitive_keys():
    """Non-sensitive data should pass through unchanged."""
    data = {"name": "Test", "vendor": "SAMSUNG"}
    redacted = _redact(data)
    assert redacted == data


class TestDiagnosticsWithEntity:
    """Tests for diagnostics with real HA entities."""

    @pytest.mark.asyncio
    async def test_diagnostics_with_pi_entity(self, hass, setup_pi_integration):
        """Diagnostics should include PI state for PI-enabled entity."""
        from custom_components.tasmota_irhvac.diagnostics import (
            async_get_config_entry_diagnostics,
        )
        entry = await setup_pi_integration()
        diag = await async_get_config_entry_diagnostics(hass, entry)

        assert "config_entry" in diag
        assert "pi_controller" in diag
        assert diag["pi_controller"]["enabled"] is True
        assert "ff_heat_buckets" in diag["pi_controller"]

    @pytest.mark.asyncio
    async def test_diagnostics_without_pi(self, hass, setup_integration):
        """Diagnostics should work for non-PI entity."""
        from custom_components.tasmota_irhvac.diagnostics import (
            async_get_config_entry_diagnostics,
        )
        entry = await setup_integration({"pi_enabled": False})
        diag = await async_get_config_entry_diagnostics(hass, entry)

        assert "config_entry" in diag
        assert "entity" in diag
