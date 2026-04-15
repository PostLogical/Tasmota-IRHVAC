"""Tests for FF learning: integral convergence tracking."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, UnitOfTemperature

from custom_components.tasmota_irhvac.pi_controller import PIController, PIExtraStoredData

from .conftest import make_pi_config


class FakeLearningEntity:
    """Minimal fake entity for learning tests."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    def __init__(self, config):
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 22.0
        self._attr_target_temperature = 22.0
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "test_entry"
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)
        self.async_get_last_extra_data = AsyncMock(return_value=None)
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return UnitOfTemperature.CELSIUS


def _make_config(**overrides):
    """Build PI config with learning-friendly defaults."""
    return make_pi_config(overrides)


def _settled_tick(entity, outdoor_temp=0.0, current=22.0, desired=22.0):
    """Set up entity state for a learning-eligible tick (in deadband, settled)."""
    entity._attr_current_temperature = current
    entity._pi._desired_temp = desired
    entity._pi._hp_setpoint = round(desired)
    entity._pi._pi_integral = 0.0
    entity._pi._outdoor_temp = outdoor_temp
    entity._pi._ff_settled_ticks = 5  # Already settled
    entity._pi._pi_last_tick_time = 0


# ── Integral Convergence Tests ───────────────────────────────────────


class TestIntegralConvergence:
    """Verify integral convergence tracking."""

    @pytest.mark.asyncio
    async def test_convergence_tracks_abs_integral(self):
        """Integral convergence should be EMA of abs(integral)."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        pi._metrics.integral_convergence = 0.0
        pi._pi_integral = 10.0

        _settled_tick(entity, outdoor_temp=5.0, current=20.0, desired=22.0)
        await pi._pi_tick()

        # After one tick with integral ~10: convergence = 0.99*0 + 0.01*|integral|
        assert pi._metrics.integral_convergence > 0

    @pytest.mark.asyncio
    async def test_convergence_decays_with_low_integral(self):
        """Convergence should decay when integral is consistently low."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        pi._metrics.integral_convergence = 20.0  # Was high
        pi._pi_integral = 0.0

        _settled_tick(entity, outdoor_temp=5.0, current=22.0, desired=22.0)
        await pi._pi_tick()

        # Should decay toward 0
        assert pi._metrics.integral_convergence < 20.0


# ── ExtraStoredData Tests ────────────────────────────────────────────


class TestExtraStoredDataLearning:
    """Verify ExtraStoredData handles legacy bucket fields gracefully."""

    def test_round_trip_serialization(self):
        """ExtraStoredData should round-trip through as_dict/from_dict."""
        data = PIExtraStoredData(
            pi_integral=0.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            integral_convergence=3.5,
        )
        serialized = data.as_dict()
        restored = PIExtraStoredData.from_dict(serialized)

        assert restored is not None
        assert restored.integral_convergence == 3.5

    def test_legacy_data_with_bucket_fields_loads(self):
        """Old ExtraStoredData with bucket fields should still load."""
        old_data = {
            "ff_heat_buckets": {"0": 1.0},
            "ff_cool_buckets": {},
            "pi_integral": 0.0,
            "desired_temp": 22.0,
            "hp_setpoint": 22.0,
            "ff_bucket_observation_counts": {"0": 5, "3": 10},
            "integral_convergence": 2.5,
        }
        restored = PIExtraStoredData.from_dict(old_data)

        assert restored is not None
        assert restored.integral_convergence == 2.5
        assert restored.pi_integral == 0.0
