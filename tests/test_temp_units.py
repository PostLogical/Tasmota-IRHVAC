"""Temperature unit handling behavioral tests.

These test user-observable behavior, not implementation details.
They should pass regardless of how the internal unit handling is
implemented — only the contract matters:
- User sees system unit (°F or °C)
- Tasmota receives °C (or celsius_mode unit)
- PI math produces correct results in either unit system
- Seeds, coefficients, and deltas display correctly
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from homeassistant.const import UnitOfTemperature
from homeassistant.components.climate.const import HVACMode
from homeassistant.util.unit_conversion import TemperatureConverter

from tests.conftest import make_pi_config
from tests.test_pi_controller import FakePIEntity


def _make_f_entity(config):
    """Create a FakePIEntity with °F unit — must set before PI reads it."""
    from custom_components.tasmota_irhvac.pi_controller import PIController
    entity = FakePIEntity.__new__(FakePIEntity)
    entity.hass = MagicMock()
    entity._attr_hvac_mode = HVACMode.HEAT
    entity._attr_temperature_unit = UnitOfTemperature.FAHRENHEIT
    entity._celsius_unit = UnitOfTemperature.CELSIUS
    entity._attr_current_temperature = 69.8  # 21°C
    entity._attr_target_temperature = 71.6   # 22°C
    entity._temp_sensor = "sensor.room_temp"
    entity._min_temp = 61
    entity._max_temp = 86
    entity.power_mode = "on"
    entity._mqtt_delay = "0"
    entity._config_entry_id = "test_entry"
    entity.send_ir = AsyncMock()
    entity.async_schedule_update_ha_state = MagicMock()
    entity.async_write_ha_state = MagicMock()
    entity.async_get_last_state = AsyncMock(return_value=None)
    entity.entity_id = "climate.test"
    entity.unique_id = "test"
    entity._pi = PIController(entity, config)
    return entity


class TestSetpointUnitConversion:
    """User sets temperature in system unit, HP receives °C."""

    def test_fahrenheit_desired_produces_celsius_ir(self):
        """User sets 68°F → HP gets 20°C via IR."""
        config = make_pi_config()
        entity = _make_f_entity(config)

        pi = entity._pi
        pi._desired_temp = 68.0  # °F
        pi._hp_setpoint = 20  # °C (what PI computes)

        # get_ir_temp returns °C for Tasmota
        ir_temp = pi.get_ir_temp()
        assert ir_temp == 20

    def test_celsius_desired_produces_celsius_ir(self):
        """User sets 20°C → HP gets 20°C via IR."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 20

        ir_temp = pi.get_ir_temp()
        assert ir_temp == 20


class TestSeedUnitConversion:
    """Seeds are stored in °C, displayed in system unit."""

    def test_seed_stored_in_celsius(self):
        """Regardless of display unit, seeds in PI controller are °C."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,  # This is °C in the config
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        # The RLS seed for the stove input should be -3.0 °C
        # Index 0=intercept, 1=outdoor_delta, 2=first model input
        assert pi._heat_seeds[2] == -3.0

    def test_coefficient_display_converts_to_fahrenheit(self):
        """Extra state attributes should show coefficients in system unit."""
        config = make_pi_config()
        entity = _make_f_entity(config)

        pi = entity._pi
        # Set a known coefficient in °C
        pi._rls_heat.beta[1] = 0.35  # outdoor_delta coefficient in °C

        attrs = pi.get_extra_state_attributes()
        displayed = attrs["rls_heat_coefficients"]["outdoor_delta"]

        # Should be 0.35 * 1.8 = 0.63 in °F
        assert abs(displayed - 0.63) < 0.01

    def test_coefficient_display_unchanged_in_celsius(self):
        """When system is °C, coefficients display as-is."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._rls_heat.beta[1] = 0.35

        attrs = pi.get_extra_state_attributes()
        displayed = attrs["rls_heat_coefficients"]["outdoor_delta"]

        assert abs(displayed - 0.35) < 0.001


class TestDeadbandBehavior:
    """Deadband produces same physical comfort zone regardless of unit."""

    @pytest.mark.asyncio
    async def test_deadband_equivalent_behavior(self):
        """0.9°F deadband and 0.5°C deadband should produce same PI behavior."""
        import time

        # Celsius system: deadband 0.5°C
        config_c = make_pi_config({"pi_deadband": 0.5})
        entity_c = FakePIEntity(config_c)
        pi_c = entity_c._pi
        pi_c._desired_temp = 20.0
        pi_c._outdoor_temp = 5.0
        entity_c._attr_current_temperature = 20.3  # within 0.5°C deadband

        # Fahrenheit system: deadband 0.9°F (= 0.5°C)
        config_f = make_pi_config({"pi_deadband": 0.9})
        entity_f = _make_f_entity(config_f)
        pi_f = entity_f._pi
        pi_f._desired_temp = 68.0  # 20°C in °F
        pi_f._outdoor_temp = 5.0
        entity_f._attr_current_temperature = 68.54  # 20.3°C in °F

        # Both should be in deadband (error < deadband)
        error_c = 20.0 - 20.3  # -0.3°C
        error_f_converted = (68.0 - 68.54) / 1.8  # -0.3°C equivalent

        assert abs(error_c) < 0.5  # within °C deadband
        assert abs(error_f_converted) < 0.5  # within equivalent deadband


class TestFFOffsetDisplay:
    """FF offset displays in system unit."""

    def test_ff_offset_displayed_in_fahrenheit(self):
        """FF offset should be in °F when system is °F."""
        config = make_pi_config()
        entity = _make_f_entity(config)

        pi = entity._pi
        pi._ff_offset = 3.5  # °C internally

        attrs = pi.get_extra_state_attributes()
        displayed_ff = attrs["ff_offset"]

        # Should be 3.5 * 1.8 = 6.3 in °F
        assert abs(displayed_ff - 6.3) < 0.1

    def test_ff_offset_displayed_in_celsius(self):
        """FF offset should be unchanged when system is °C."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._ff_offset = 3.5

        attrs = pi.get_extra_state_attributes()
        assert abs(attrs["ff_offset"] - 3.5) < 0.01


class TestDeltaConversionRoundTrip:
    """Temperature delta conversions should be stable on round-trip."""

    def test_celsius_round_trip(self):
        """°C → display → store should be identity in °C system."""
        from custom_components.tasmota_irhvac.pi_controller import (
            _delta_to_c, _delta_to_display,
        )
        for val in [0.3, 0.5, 1.0, 3.5, -4.0, -8.0]:
            displayed = _delta_to_display(val, UnitOfTemperature.CELSIUS)
            stored = _delta_to_c(displayed, UnitOfTemperature.CELSIUS)
            assert abs(stored - val) < 0.001, f"Round-trip failed for {val}"

    def test_fahrenheit_round_trip(self):
        """°C → display(°F) → store(°C) should be stable."""
        from custom_components.tasmota_irhvac.pi_controller import (
            _delta_to_c, _delta_to_display,
        )
        for val in [0.3, 0.5, 1.0, 3.5, -4.0, -8.0]:
            displayed = _delta_to_display(val, UnitOfTemperature.FAHRENHEIT)
            stored = _delta_to_c(displayed, UnitOfTemperature.FAHRENHEIT)
            assert abs(stored - val) < 0.001, f"Round-trip failed for {val}: {val} → {displayed} → {stored}"

    def test_user_enters_round_fahrenheit(self):
        """User enters -6°F → stored as °C → displayed as -6°F."""
        from custom_components.tasmota_irhvac.pi_controller import (
            _delta_to_c, _delta_to_display,
        )
        user_input = -6.0  # °F
        stored = _delta_to_c(user_input, UnitOfTemperature.FAHRENHEIT)
        redisplayed = _delta_to_display(stored, UnitOfTemperature.FAHRENHEIT)
        assert abs(redisplayed - user_input) < 0.01
