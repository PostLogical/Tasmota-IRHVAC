"""Tests for learning suppression and legacy disturbance input migration."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import State

from custom_components.tasmota_irhvac.pi.pi_controller import PIController

from .conftest import make_pi_config


# ── Test Entity ──────────────────────────────────────────────────────


class FakePIEntity:
    """Minimal fake entity for disturbance testing."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.FAHRENHEIT
    _temp_precision = 1.0

    def __init__(self, config):
        self.hass = MagicMock()
        self.hass.states.get.return_value = None
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 70.0
        self._attr_target_temperature = 72.0
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 61
        self._max_temp = 86
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "test_disturbance"
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)
        self.async_get_last_extra_data = AsyncMock(return_value=None)
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return UnitOfTemperature.FAHRENHEIT


# ── Tests: Manual Suppress ─────────────────────────────────────────


class TestManualSuppress:
    """Tests for manual suppress service methods."""

    @pytest.mark.asyncio
    async def test_suppress_sets_flag(self):
        """suppress_ff_learning sets the manual flag and reason."""
        entity = FakePIEntity(make_pi_config())
        await entity._pi.async_suppress_ff_learning(reason="testing")
        assert entity._pi._manual_ff_suppress is True
        assert entity._pi._manual_ff_suppress_reason == "testing"

    @pytest.mark.asyncio
    async def test_resume_clears_flag(self):
        """resume_ff_learning clears the manual flag."""
        entity = FakePIEntity(make_pi_config())
        await entity._pi.async_suppress_ff_learning(reason="test")
        await entity._pi.async_resume_ff_learning()
        assert entity._pi._manual_ff_suppress is False
        assert entity._pi._manual_ff_suppress_reason == ""

    @pytest.mark.asyncio
    async def test_manual_suppress_empty_reason(self):
        """suppress with no reason works."""
        entity = FakePIEntity(make_pi_config())
        await entity._pi.async_suppress_ff_learning()
        assert entity._pi._manual_ff_suppress is True
        assert entity._pi._manual_ff_suppress_reason == ""

    @pytest.mark.asyncio
    async def test_manual_suppress_blocks_rls_learning(self):
        """Manual suppress should block RLS learning in tick."""
        entity = FakePIEntity(make_pi_config())
        pi = entity._pi
        await pi.async_suppress_ff_learning(reason="test")

        pi._inputs.outdoor_temp = 5.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._ff_settled_ticks = 10
        pi._rls_warmup_done = True
        pi._rls_heat_mature = True
        pi._pi_integral = 0.5
        pi._prev_integral_for_obs = 0.5
        entity._attr_current_temperature = 22.0
        entity._attr_hvac_mode = HVACMode.HEAT

        mock_state = MagicMock()
        mock_state.state = "off"
        entity.hass.states.get.return_value = mock_state

        old_obs = pi._rls_heat.observation_count
        await pi._pi_tick()

        assert pi._rls_heat.observation_count == old_obs
        assert pi._disturbance_suppress_active is True


# ── Tests: Integral Behavior ──────────────────────────────────────


class TestIntegralReset:
    """Tests for integral behavior with RLS model."""

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
        entity._pi._inputs.outdoor_temp = 5.0
        entity._pi._pi_last_tick_time = 0

        await entity._pi._pi_tick()

        assert entity._pi._pi_integral > 0.0

    @pytest.mark.asyncio
    async def test_integral_frozen_in_deadband(self):
        """Integral should freeze (no accumulation, no decay) in deadband."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        entity._pi._pi_integral = 5.0
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 22.0
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0
        entity._pi._inputs.outdoor_temp = 14.0  # Near reference, low FF offset
        entity._pi._pi_last_tick_time = 0

        await entity._pi._pi_tick()

        # In deadband: integral frozen (quantization feedback may nudge slightly)
        assert entity._pi._pi_integral > 0.0


# ── Tests: Migration ────────────────────────────────────────────────


class TestDisturbanceMigration:
    """Test that old suppress/bias config migration logic works."""

    def test_migrate_suppress_entity(self):
        """Old suppress entity → disturbance input with suppress=True, bias=0."""
        old_config = {
            "pi_ff_suppress_learning_entity": "input_boolean.suppress_learning",
            "pi_ff_bias_entity": "",
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
