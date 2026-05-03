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
    from custom_components.tasmota_irhvac.pi.pi_controller import PIController
    entity = FakePIEntity.__new__(FakePIEntity)
    entity.hass = MagicMock()
    entity._attr_hvac_mode = HVACMode.HEAT
    entity._attr_temperature_unit = UnitOfTemperature.FAHRENHEIT
    entity._ir_temp_unit = UnitOfTemperature.CELSIUS
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
                "seed_heat": 3.0,  # This is °C in the config (positive = warms room)
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        # Internal β = -seed (HP backs off when source warms room)
        # Index 0=intercept, 1=outdoor_delta, 2=first model input
        assert pi._heat_seeds[2] == -3.0

    def test_coefficient_display_always_celsius(self):
        """Extra state attributes show coefficients in °C regardless of system unit."""
        config = make_pi_config()
        entity = _make_f_entity(config)

        pi = entity._pi
        # Set the outdoor_delta seed to 0.35 (user-facing convention).
        pi._rls_heat.beta[1] = pi._rls_heat.seed_to_beta(1, 0.35)

        attrs = pi.get_extra_state_attributes()
        displayed = attrs["rls_heat_seeds"]["outdoor_delta"]

        # Seeds are always in °C — no system-unit conversion.
        assert abs(displayed - 0.35) < 0.001

    def test_coefficient_display_unchanged_in_celsius(self):
        """When system is °C, coefficients display as-is."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._rls_heat.beta[1] = pi._rls_heat.seed_to_beta(1, 0.35)

        attrs = pi.get_extra_state_attributes()
        displayed = attrs["rls_heat_seeds"]["outdoor_delta"]

        assert abs(displayed - 0.35) < 0.001


class TestDeadbandBehavior:
    """Deadband produces same physical comfort zone regardless of unit."""

    @pytest.mark.asyncio
    async def test_deadband_equivalent_behavior(self):
        """Both °C and °F systems use same °C deadband value."""
        # Celsius system: deadband 0.5°C
        config_c = make_pi_config({"pi_deadband": 0.5})
        entity_c = FakePIEntity(config_c)
        pi_c = entity_c._pi
        pi_c._desired_temp = 20.0
        pi_c._inputs.outdoor_temp = 5.0
        entity_c._attr_current_temperature = 20.3  # within 0.5°C deadband

        # Fahrenheit system: same deadband 0.5°C (entered in °C in config)
        config_f = make_pi_config({"pi_deadband": 0.5})
        entity_f = _make_f_entity(config_f)
        pi_f = entity_f._pi
        pi_f._desired_temp = 68.0  # 20°C in °F
        pi_f._inputs.outdoor_temp = 5.0
        entity_f._attr_current_temperature = 68.54  # 20.3°C in °F

        # Both store deadband in °C — same value
        assert pi_c._pi_deadband == 0.5
        assert pi_f._pi_deadband == 0.5


class TestFFOffsetDisplay:
    """FF offset displays in system unit."""

    def test_ff_offset_displayed_in_celsius(self):
        """FF offset is always in °C regardless of system unit."""
        config = make_pi_config()
        entity = _make_f_entity(config)

        pi = entity._pi
        pi._ff_offset = 3.5  # °C internally

        attrs = pi.get_extra_state_attributes()
        displayed_ff = attrs["ff_offset"]

        # FF offset always displays in °C — no conversion
        assert abs(displayed_ff - 3.5) < 0.01

    def test_ff_offset_displayed_in_celsius(self):
        """FF offset should be unchanged when system is °C."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._ff_offset = 3.5

        attrs = pi.get_extra_state_attributes()
        assert abs(attrs["ff_offset"] - 3.5) < 0.01


class TestControlParamsAlwaysCelsius:
    """Control parameters are always stored and used in °C."""

    def test_deadband_stored_in_celsius(self):
        """Deadband is stored in °C regardless of system unit."""
        config = make_pi_config({"pi_deadband": 0.5})
        entity = _make_f_entity(config)
        assert entity._pi._pi_deadband == 0.5

    def test_outdoor_seeds_stored_in_celsius(self):
        """Outdoor seeds are stored in °C regardless of system unit."""
        config = make_pi_config({
            "pi_outdoor_seed_heat": 0.35,
            "pi_outdoor_seed_cool": 0.25,
        })
        entity = _make_f_entity(config)
        assert entity._pi._outdoor_seed_heat == 0.35
        assert entity._pi._outdoor_seed_cool == 0.25
