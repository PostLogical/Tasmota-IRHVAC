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
        assert "rls_model" in diag["pi_controller"]

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

    @pytest.mark.asyncio
    async def test_diagnostics_batch_state_none_before_run(self, hass, setup_pi_integration):
        """Batch learning section should be None when no batch has run."""
        entry = await setup_pi_integration()
        diag = await async_get_config_entry_diagnostics(hass, entry)

        assert "pi_controller" in diag
        assert diag["pi_controller"]["batch_learning"] is None

    @pytest.mark.asyncio
    async def test_diagnostics_observation_buffer_present(self, hass, setup_pi_integration):
        """Observation buffer stats should appear in diagnostics."""
        entry = await setup_pi_integration()
        diag = await async_get_config_entry_diagnostics(hass, entry)

        assert "observation_buffer_heat" in diag["pi_controller"]
        buf = diag["pi_controller"]["observation_buffer_heat"]
        assert "total" in buf
        assert "eligible" in buf
        assert buf["total"] >= 0
        assert "observation_buffer_cool" in diag["pi_controller"]

    @pytest.mark.asyncio
    async def test_diagnostics_performance_fields(self, hass, setup_pi_integration):
        """Performance metrics should appear in diagnostics."""
        entry = await setup_pi_integration()
        diag = await async_get_config_entry_diagnostics(hass, entry)

        pi_diag = diag["pi_controller"]
        assert "performance" in pi_diag
        perf = pi_diag["performance"]
        assert "itae_accumulator" in perf
        assert "comfort_violation_hours" in perf
        assert "setpoint_changes" in perf

        assert "ff_confidence" in pi_diag
        assert "room_temp_rate" in pi_diag
        assert "tau_estimate" in pi_diag
