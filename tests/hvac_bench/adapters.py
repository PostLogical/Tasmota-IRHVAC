"""Controller adapters for HVAC benchmark.

Wraps specific controller implementations to conform to HVACController protocol.
"""

import asyncio
import time
from unittest.mock import MagicMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import UnitOfTemperature

from tests.conftest import make_pi_config


class TasmotaPIAdapter:
    """Adapts the Tasmota-IRHVAC PIController to HVACController protocol.

    Wraps the async PIController in a synchronous interface suitable
    for benchmark simulation.
    """

    def __init__(self, config_overrides: dict | None = None):
        """Initialize with optional config overrides.

        Args:
            config_overrides: Dict of PI config values to override defaults.
                Common: {"pi_ki": 0.15, "pi_kd": 0.5, "pi_setpoint_weight": 0.3}
        """
        config = make_pi_config(config_overrides or {})
        self._config = config
        self._entity = _FakeBenchEntity(config)
        self._pi = self._entity._pi
        self._loop = asyncio.new_event_loop()
        self._sim_clock = 0.0
        self._mode = "heat"

    def tick(self, room_temp_c, outdoor_temp_c, dt_seconds,
             model_inputs=None):
        """Run one PI tick and return HP setpoint."""
        self._sim_clock += dt_seconds
        self._entity._attr_current_temperature = room_temp_c
        self._pi._outdoor_temp = outdoor_temp_c

        # Update model inputs if provided
        if model_inputs:
            for i, m_input in enumerate(self._pi._model_inputs):
                name = m_input.get("name", "")
                if name in model_inputs:
                    self._pi._model_input_values[i] = model_inputs[name]

        # Set timing
        self._pi._pi_last_tick_time = self._sim_clock - dt_seconds

        # Mock time.monotonic for the PI controller
        original = time.monotonic
        time.monotonic = lambda: self._sim_clock
        try:
            self._loop.run_until_complete(self._pi._pi_tick())
        finally:
            time.monotonic = original

        return float(self._pi._hp_setpoint)

    def set_desired_temp(self, temp_c):
        self._pi._desired_temp = temp_c

    def set_mode(self, mode):
        self._mode = mode
        if mode == "heat":
            self._entity._attr_hvac_mode = HVACMode.HEAT
        elif mode == "cool":
            self._entity._attr_hvac_mode = HVACMode.COOL
        else:
            self._entity._attr_hvac_mode = HVACMode.HEAT

    def get_state(self):
        return {
            "integral": self._pi._pi_integral,
            "ff_offset": self._pi._ff_offset,
            "d_term": getattr(self._pi, "_pi_d_filtered", 0.0),
            "rls_obs_count": self._pi._rls_heat.observation_count,
            "desired_temp": self._pi._desired_temp,
            "hp_setpoint": self._pi._hp_setpoint,
        }

    def __del__(self):
        if hasattr(self, "_loop") and self._loop and not self._loop.is_closed():
            self._loop.close()


class TextbookPIController:
    """Simple textbook PI controller for cross-validation.

    No feedforward, no RLS, no quantization tricks. Just PI.
    Used to verify benchmark tests aren't too tight.
    """

    def __init__(self, kp=1.0, ki=0.1, setpoint_weight=1.0,
                 min_temp=16.0, max_temp=30.0):
        self.kp = kp
        self.ki = ki
        self.b = setpoint_weight
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.desired = 20.5
        self.integral = 0.0
        self.hp_setpoint = 20.0
        self._mode = "heat"

    def tick(self, room_temp_c, outdoor_temp_c, dt_seconds,
             model_inputs=None):
        error = self.desired - room_temp_c
        dt_factor = dt_seconds / 900.0  # normalize to 15 min

        p_term = self.kp * self.b * error
        self.integral += error * dt_factor
        i_term = self.ki * self.integral

        raw = self.desired + p_term + i_term
        clamped = max(self.min_temp, min(self.max_temp, raw))

        # Simple anti-windup
        if clamped != raw and self.ki != 0:
            if raw > clamped and self.integral > 0:
                self.integral = (clamped - self.desired - p_term) / self.ki
            elif raw < clamped and self.integral < 0:
                self.integral = (clamped - self.desired - p_term) / self.ki

        # Integer quantization with hysteresis
        if clamped > self.hp_setpoint + 0.5:
            self.hp_setpoint = round(clamped)
        elif clamped < self.hp_setpoint - 0.5:
            self.hp_setpoint = round(clamped)
        self.hp_setpoint = int(max(self.min_temp, min(self.max_temp, self.hp_setpoint)))

        return float(self.hp_setpoint)

    def set_desired_temp(self, temp_c):
        self.desired = temp_c

    def set_mode(self, mode):
        self._mode = mode

    def get_state(self):
        return {
            "integral": self.integral,
            "ff_offset": 0.0,
            "desired_temp": self.desired,
            "hp_setpoint": self.hp_setpoint,
            "rls_obs_count": 0,
        }


# ── Fake entity for adapter ──────────────────────────────────────────────


class _FakeBenchEntity:
    """Minimal fake entity for PIController adapter."""

    def __init__(self, config):
        from custom_components.tasmota_irhvac.pi_controller import PIController

        self.hass = MagicMock()
        self._attr_current_temperature = 20.0
        self._attr_target_temperature = 20.0
        self._attr_hvac_mode = HVACMode.HEAT
        self._temp_sensor = "sensor.room_temp"
        self.temperature_unit = UnitOfTemperature.CELSIUS
        self._attr_temperature_unit = UnitOfTemperature.CELSIUS
        self.entity_id = "climate.bench_test"
        self._config_entry_id = "bench_test"
        self.unique_id = "bench_test"
        self._temp_precision = config.get("precision", 1.0)
        self._min_temp = config.get("min_temp", 16)
        self._max_temp = config.get("max_temp", 30)
        self._attr_min_temp = self._min_temp
        self._attr_max_temp = self._max_temp

        self._pi = PIController(self, config)
        self._pi._pi_enabled = True

    @property
    def device_info(self):
        return None

    def async_schedule_update_ha_state(self, force_refresh=False):
        pass

    async def send_ir(self, *args, **kwargs):
        pass
