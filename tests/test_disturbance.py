"""Tests for the disturbance input framework."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import State

from custom_components.tasmota_irhvac.pi_controller import PIController

from .conftest import make_pi_config


# ── Test Entity ──────────────────────────────────────────────────────


class FakePIEntity:
    """Minimal fake entity for disturbance testing."""

    _attr_temperature_unit = UnitOfTemperature.CELSIUS

    def __init__(self, config):
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 72.0
        self._attr_target_temperature = 72.0
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)
        self._config_entry_id = "test_entry"
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return UnitOfTemperature.CELSIUS


def _mock_state(entity_id, state_val, attributes=None):
    """Create a mock State object."""
    return State(entity_id, state_val, attributes or {})


def _setup_hass_states(entity, state_map):
    """Configure hass.states.get to return mock states."""
    def get_state(entity_id):
        if entity_id in state_map:
            return _mock_state(entity_id, state_map[entity_id])
        return None
    entity.hass.states.get = MagicMock(side_effect=get_state)


# ── Tests: _compute_disturbance_effects ──────────────────────────────


class TestComputeDisturbanceEffects:
    """Tests for the disturbance computation logic."""

    def test_no_inputs(self):
        """No disturbance inputs → no suppress, no bias."""
        entity = FakePIEntity(make_pi_config())
        _setup_hass_states(entity, {})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is False
        assert active == []
        assert bias == 0.0

    def test_boolean_on_suppresses(self):
        """Boolean entity 'on' with suppress_learning=True suppresses."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Door",
                "entity_id": "binary_sensor.door",
                "suppress_learning": True,
                "default_bias": 1.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"binary_sensor.door": "on"})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is True
        assert "binary_sensor.door" in active
        assert bias == pytest.approx(1.0)

    def test_boolean_off_inactive(self):
        """Boolean entity 'off' → inactive, no suppress or bias."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Door",
                "entity_id": "binary_sensor.door",
                "suppress_learning": True,
                "default_bias": 1.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"binary_sensor.door": "off"})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is False
        assert active == []
        assert bias == 0.0

    def test_numeric_nonzero_applies_gain(self):
        """Numeric entity with non-zero value → bias = value × gain."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Bias Sensor",
                "entity_id": "sensor.bias",
                "suppress_learning": False,
                "default_bias": 0.0,
                "gain": 0.5,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"sensor.bias": "-4.0"})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is False
        assert active == []
        assert bias == pytest.approx(-2.0)

    def test_numeric_zero_inactive(self):
        """Numeric entity with value 0 → inactive."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Bias Sensor",
                "entity_id": "sensor.bias",
                "suppress_learning": True,
                "default_bias": 0.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"sensor.bias": "0"})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is False
        assert bias == 0.0

    def test_unavailable_entity_skipped(self):
        """Unavailable entities are silently skipped."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Door",
                "entity_id": "binary_sensor.door",
                "suppress_learning": True,
                "default_bias": 5.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"binary_sensor.door": STATE_UNAVAILABLE})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is False
        assert bias == 0.0

    def test_unknown_entity_skipped(self):
        """Unknown entities are silently skipped."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Door",
                "entity_id": "binary_sensor.door",
                "suppress_learning": True,
                "default_bias": 5.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"binary_sensor.door": STATE_UNKNOWN})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is False
        assert bias == 0.0

    def test_missing_entity_skipped(self):
        """Entity not in HA → skipped (hass.states.get returns None)."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Gone",
                "entity_id": "binary_sensor.nonexistent",
                "suppress_learning": True,
                "default_bias": 5.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {})  # entity not in map
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is False

    def test_multiple_inputs_sum_bias(self):
        """Multiple active inputs sum their biases."""
        config = make_pi_config({
            "pi_disturbance_inputs": [
                {
                    "name": "Door",
                    "entity_id": "binary_sensor.door",
                    "suppress_learning": True,
                    "default_bias": 1.5,
                    "gain": 1.0,
                },
                {
                    "name": "Stove",
                    "entity_id": "binary_sensor.stove",
                    "suppress_learning": True,
                    "default_bias": -5.0,
                    "gain": 1.0,
                },
            ]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {
            "binary_sensor.door": "on",
            "binary_sensor.stove": "on",
        })
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is True
        assert len(active) == 2
        assert bias == pytest.approx(-3.5)

    def test_mixed_boolean_and_numeric(self):
        """Boolean and numeric inputs work together."""
        config = make_pi_config({
            "pi_disturbance_inputs": [
                {
                    "name": "Door",
                    "entity_id": "binary_sensor.door",
                    "suppress_learning": True,
                    "default_bias": 1.0,
                    "gain": 1.0,
                },
                {
                    "name": "Solar",
                    "entity_id": "sensor.solar_bias",
                    "suppress_learning": False,
                    "default_bias": 0.0,
                    "gain": 1.0,
                },
            ]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {
            "binary_sensor.door": "on",
            "sensor.solar_bias": "-0.5",
        })
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is True
        assert bias == pytest.approx(0.5)  # 1.0 + (-0.5)

    def test_suppress_without_bias(self):
        """Suppress-only input (bias=0) still suppresses."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Oil Boiler",
                "entity_id": "binary_sensor.oil_zone",
                "suppress_learning": True,
                "default_bias": 0.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"binary_sensor.oil_zone": "on"})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is True
        assert bias == 0.0

    def test_numeric_suppress_when_nonzero(self):
        """Numeric entity with suppress_learning=True suppresses when non-zero."""
        config = make_pi_config({
            "pi_disturbance_inputs": [{
                "name": "Template Bias",
                "entity_id": "sensor.pellet_bias",
                "suppress_learning": True,
                "default_bias": 0.0,
                "gain": 1.0,
            }]
        })
        entity = FakePIEntity(config)
        _setup_hass_states(entity, {"sensor.pellet_bias": "-3.0"})
        suppress, active, bias = entity._pi._compute_disturbance_effects()
        assert suppress is True
        assert "sensor.pellet_bias" in active
        assert bias == pytest.approx(-3.0)


# ── Tests: Manual Suppress ───────────────────────────────────────────


class TestManualSuppress:
    """Tests for manual suppress service methods."""

    @pytest.mark.asyncio
    async def test_suppress_sets_flag(self):
        """suppress_ff_learning sets the manual flag and reason."""
        entity = FakePIEntity(make_pi_config())
        _setup_hass_states(entity, {})
        await entity._pi.async_suppress_ff_learning(reason="testing")
        assert entity._pi._manual_ff_suppress is True
        assert entity._pi._manual_ff_suppress_reason == "testing"

        suppress, _, _ = entity._pi._compute_disturbance_effects()
        assert suppress is True

    @pytest.mark.asyncio
    async def test_resume_clears_flag(self):
        """resume_ff_learning clears the manual flag."""
        entity = FakePIEntity(make_pi_config())
        _setup_hass_states(entity, {})
        await entity._pi.async_suppress_ff_learning(reason="test")
        await entity._pi.async_resume_ff_learning()
        assert entity._pi._manual_ff_suppress is False
        assert entity._pi._manual_ff_suppress_reason == ""

    @pytest.mark.asyncio
    async def test_manual_suppress_empty_reason(self):
        """suppress with no reason works."""
        entity = FakePIEntity(make_pi_config())
        _setup_hass_states(entity, {})
        await entity._pi.async_suppress_ff_learning()
        assert entity._pi._manual_ff_suppress is True
        assert entity._pi._manual_ff_suppress_reason == ""


# ── Tests: Integral Reset on Bias Transition ─────────────────────────


class TestIntegralReset:
    """Tests for integral behavior with RLS model (formerly disturbance bias reset)."""

    @pytest.mark.asyncio
    async def test_integral_accumulates_with_error(self):
        """Integral should accumulate when there is heating error."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        entity._pi._pi_integral = 0.0
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0
        entity._pi._outdoor_temp = 5.0
        entity._pi._pi_last_tick_time = 0

        await entity._pi._pi_tick()

        assert entity._pi._pi_integral > 0.0

    @pytest.mark.asyncio
    async def test_integral_decays_in_deadband(self):
        """Integral should decay (×0.9) but not reset in deadband."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        entity._pi._pi_integral = 5.0
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 22.0
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0
        entity._pi._outdoor_temp = 14.0  # Near reference, low FF offset
        entity._pi._pi_last_tick_time = 0

        await entity._pi._pi_tick()

        # In deadband: integral *= 0.9, so should be ~4.5
        assert entity._pi._pi_integral > 0.0
        assert entity._pi._pi_integral < 5.0


# ── Tests: Migration ────────────────────────────────────────────────


class TestDisturbanceMigration:
    """Test that old suppress/bias config migrates correctly."""

    def test_migrate_suppress_entity(self):
        """Old suppress entity → disturbance input with suppress=True, bias=0."""
        old_config = {
            "pi_ff_suppress_learning_entity": "input_boolean.suppress_learning",
            "pi_ff_bias_entity": "",
        }
        # Simulate migration logic
        disturbance_inputs = []
        old_suppress = old_config.pop("pi_ff_suppress_learning_entity", "")
        old_bias = old_config.pop("pi_ff_bias_entity", "")
        if old_suppress:
            disturbance_inputs.append({
                "name": "Suppress Entity (migrated)",
                "entity_id": old_suppress,
                "suppress_learning": True,
                "default_bias": 0.0,
                "gain": 1.0,
            })
        if old_bias:
            disturbance_inputs.append({
                "name": "Bias Entity (migrated)",
                "entity_id": old_bias,
                "suppress_learning": False,
                "default_bias": 0.0,
                "gain": 1.0,
            })

        assert len(disturbance_inputs) == 1
        assert disturbance_inputs[0]["entity_id"] == "input_boolean.suppress_learning"
        assert disturbance_inputs[0]["suppress_learning"] is True

    def test_migrate_both_entities(self):
        """Both suppress and bias entities migrate to two disturbance inputs."""
        old_config = {
            "pi_ff_suppress_learning_entity": "input_boolean.suppress",
            "pi_ff_bias_entity": "input_number.ff_bias",
        }
        disturbance_inputs = []
        old_suppress = old_config.pop("pi_ff_suppress_learning_entity", "")
        old_bias = old_config.pop("pi_ff_bias_entity", "")
        if old_suppress:
            disturbance_inputs.append({
                "name": "Suppress Entity (migrated)",
                "entity_id": old_suppress,
                "suppress_learning": True,
                "default_bias": 0.0,
                "gain": 1.0,
            })
        if old_bias:
            disturbance_inputs.append({
                "name": "Bias Entity (migrated)",
                "entity_id": old_bias,
                "suppress_learning": False,
                "default_bias": 0.0,
                "gain": 1.0,
            })

        assert len(disturbance_inputs) == 2

    def test_migrate_empty(self):
        """No old entities → empty disturbance inputs."""
        old_config = {
            "pi_ff_suppress_learning_entity": "",
            "pi_ff_bias_entity": "",
        }
        disturbance_inputs = []
        old_suppress = old_config.pop("pi_ff_suppress_learning_entity", "")
        old_bias = old_config.pop("pi_ff_bias_entity", "")
        if old_suppress:
            disturbance_inputs.append({"entity_id": old_suppress})
        if old_bias:
            disturbance_inputs.append({"entity_id": old_bias})

        assert len(disturbance_inputs) == 0
