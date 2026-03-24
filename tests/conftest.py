"""Test fixtures for Tasmota IRHVAC integration."""

import pytest
from unittest.mock import AsyncMock, patch

from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN


def make_config(overrides=None):
    """Build a config dict with sensible defaults for testing."""
    config = {
        "name": "Test AC",
        "vendor": "FUJITSU_AC",
        "command_topic": "cmnd/irhvac/irhvac",
        "state_topic": "tele/irhvac/RESULT",
        "availability_topic": "tele/irhvac/LWT",
        "mqtt_delay": "0",
        "min_temp": 16,
        "max_temp": 30,
        "target_temp": 22,
        "precision": 1.0,
        "temp_step": 1.0,
        "celsius_mode": "on",
        "initial_operation_mode": "off",
        "keep_mode_when_off": False,
        "ignore_off_temp": False,
        "supported_modes": ["heat", "cool", "dry", "fan_only", "off"],
        "supported_fan_speeds": ["auto", "low", "medium", "high"],
        "supported_swing_list": ["off", "vertical", "horizontal", "both"],
        "preset_modes": ["none", "away"],
        "away_temp": 16,
        "default_quiet_mode": "off",
        "default_turbo_mode": "off",
        "default_econo_mode": "off",
        "default_light_mode": "off",
        "default_filter_mode": "off",
        "default_clean_mode": "off",
        "default_beep_mode": "off",
        "default_sleep_mode": "-1",
        "hvac_model": -1,
        "default_swingv": "auto",
        "default_swingh": "auto",
        "toggle_list": [],
        "temperature_sensor": "",
        "humidity_sensor": "",
        "power_sensor": "",
        # PI defaults
        "pi_enabled": False,
        "pi_kp": 1.5,
        "pi_ki": 0.05,
        "pi_min_interval": 900,
        "pi_deadband": 0.5,
        "outdoor_temp_sensor": "",
        "pi_ff_heat_reference": 15.0,
        "pi_ff_heat_slope": 0.3,
        "pi_ff_cool_reference": 25.0,
        "pi_ff_cool_slope": 0.3,
        "pi_ff_suppress_learning_entity": "",
        "pi_ff_bias_entity": "",
        "pi_setpoint_weight": 1.0,
    }
    if overrides:
        config.update(overrides)
    return config


def make_pi_config(overrides=None):
    """Build a config dict with PI enabled and a temp sensor."""
    pi_defaults = {
        "pi_enabled": True,
        "temperature_sensor": "sensor.room_temp",
        "outdoor_temp_sensor": "sensor.outdoor_temp",
    }
    if overrides:
        pi_defaults.update(overrides)
    return make_config(pi_defaults)


@pytest.fixture
def mock_config_entry():
    """Return a MockConfigEntry for a non-PI Fujitsu setup."""
    return MockConfigEntry(
        domain=DOMAIN,
        data=make_config(),
        title="Test AC",
    )


@pytest.fixture
def mock_pi_config_entry():
    """Return a MockConfigEntry for a Fujitsu PI setup."""
    return MockConfigEntry(
        domain=DOMAIN,
        data=make_pi_config(),
        title="Test AC PI",
    )
