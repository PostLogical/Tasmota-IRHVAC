"""Test fixtures for Tasmota IRHVAC integration."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DATA_KEY, DOMAIN


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
        "ir_protocol_unit": "celsius",
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
        "special_mode": "",
        "temperature_sensor": "",
        "humidity_sensor": "",
        "power_sensor": "",
        # PI defaults
        "pi_enabled": False,
        "pi_kp": 1.5,
        "pi_ki": 0.15,
        "pi_tick_fallback": 900,
        "pi_deadband": 0.5,
        "outdoor_temp_sensor": "",
        "pi_outdoor_seed_heat": 0.25,
        "pi_outdoor_seed_cool": 0.25,
        "pi_model_inputs": [],
        "pi_setpoint_weight": 0.3,
    }
    if overrides:
        config.update(overrides)
    return config


def make_pi_config(overrides=None):
    """Build a config dict with PI enabled and a temp sensor.

    Test default sets `pi_tau_estimate: 0.0` to preserve the manual-gains
    path the existing test suite was tuned for.  Production code now
    defaults to IMC-on (`DEFAULT_PI_TAU_ESTIMATE = 60.0`); tests that
    want IMC behavior should pass `pi_tau_estimate: 60` (or any positive
    value) in `overrides`.
    """
    pi_defaults = {
        "pi_enabled": True,
        "temperature_sensor": "sensor.room_temp",
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "pi_tau_estimate": 0.0,
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


# ── Integration test fixtures ─────────────────────────────────────────


@pytest.fixture
def setup_integration(hass, mqtt_mock, enable_custom_integrations):
    """Return a factory that sets up a config entry through real HA machinery.

    Usage:
        entry = await setup_integration()
        entry = await setup_integration({"vendor": "MITSUBISHI_AC"})
    """
    async def _setup(config_overrides=None):
        config = make_config(config_overrides or {})
        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title=config.get("name", "Test AC"),
            version=1,
            minor_version=2,
        )
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        return entry
    return _setup


@pytest.fixture
def setup_pi_integration(hass, mqtt_mock, enable_custom_integrations):
    """Return a factory that sets up a PI-enabled config entry with pre-seeded sensors.

    Seeds room temp (21°C) and outdoor temp (5°C) states before setup.
    """
    async def _setup(config_overrides=None):
        hass.states.async_set(
            "sensor.room_temp", "21.0",
            {"unit_of_measurement": "°C"},
        )
        hass.states.async_set(
            "sensor.outdoor_temp", "5.0",
            {"unit_of_measurement": "°C"},
        )
        config = make_pi_config(config_overrides or {})
        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title=config.get("name", "Test AC PI"),
            version=1,
            minor_version=2,
        )
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        return entry
    return _setup


def get_climate_entity(hass, entry):
    """Look up the climate entity for a config entry."""
    return hass.data.get(DATA_KEY, {}).get(entry.entry_id)


def make_mqtt_state_payload(overrides=None):
    """Build a standard IRHVAC MQTT state payload."""
    payload = {
        "Vendor": "FUJITSU_AC",
        "Power": "On",
        "Mode": "Heat",
        "Temp": 22,
        "Celsius": "On",
        "FanSpeed": "Auto",
        "SwingV": "Auto",
        "SwingH": "Off",
        "Quiet": "Off",
        "Turbo": "Off",
        "Econo": "Off",
        "Light": "Off",
        "Filter": "Off",
        "Clean": "Off",
        "Beep": "Off",
        "Sleep": "-1",
    }
    if overrides:
        payload.update(overrides)
    return json.dumps({"IRHVAC": payload})
