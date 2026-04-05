"""Smoke tests to verify test infrastructure works."""

import pytest
from custom_components.tasmota_irhvac.const import DOMAIN, PLATFORMS
from custom_components.tasmota_irhvac.pi_controller import PIController


def test_domain():
    """Verify domain constant."""
    assert DOMAIN == "tasmota_irhvac"


def test_platforms():
    """Verify platforms include climate and sensor."""
    assert "climate" in PLATFORMS
    assert "sensor" in PLATFORMS
