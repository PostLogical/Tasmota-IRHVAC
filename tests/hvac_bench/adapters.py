"""Controller adapters for HVAC benchmark.

Wraps specific controller implementations to conform to HVACController protocol.
"""

import asyncio
import time
from unittest.mock import MagicMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import UnitOfTemperature

from tests.conftest import _PITestEntityRoomTempMixin, make_pi_config


class TasmotaPIAdapter:
    """Adapts the Tasmota-IRHVAC PIController to HVACController protocol.

    Wraps the async PIController in a synchronous interface suitable
    for benchmark simulation.
    """

    def __init__(self, config_overrides: dict | None = None,
                 head_calibration_bounds: tuple[float, float] = (0.0, 0.0)):
        """Initialize with optional config overrides.

        Args:
            config_overrides: Dict of PI config values to override defaults.
                Common: {"pi_ki": 0.15, "pi_kd": 0.5, "pi_setpoint_weight": 0.3}
            head_calibration_bounds: (cal_min, cal_max) for uncertain zone.
                Default (0.0, 0.0) = no uncertain zone (perfect sensor).
                Use None for production defaults (±2.0°C).
        """
        overrides = dict(config_overrides or {})
        # Pull out test-only seed overrides before make_pi_config (they are
        # not real config keys — production uses fixed DEFAULT_TAU_*_SEED).
        tau_fast_seed = overrides.pop("tau_fast_seed", None)
        tau_slow_seed = overrides.pop("tau_slow_seed", None)

        config = make_pi_config(overrides)
        self._config = config
        self._entity = _FakeBenchEntity(config,
                                        head_calibration_bounds=head_calibration_bounds)
        self._pi = self._entity._pi
        self._loop = asyncio.new_event_loop()
        self._sim_clock = 0.0
        self._mode = "heat"

        if tau_fast_seed is not None or tau_slow_seed is not None:
            self._inject_plant_seeds(tau_fast_seed, tau_slow_seed)

    def _inject_plant_seeds(self, tau_fast_seed: float | None,
                            tau_slow_seed: float | None) -> None:
        """Override the plant identifier's τ seeds for per-profile tuning.

        Production uses fixed DEFAULT_TAU_FAST_SEED / DEFAULT_TAU_SLOW_SEED
        (pre44 maturity gate). Tests that want to validate gain scheduling
        across profiles need to inject profile-derived seeds; otherwise
        every profile gets identical Kp/Ki and the test premise collapses.
        """
        from custom_components.tasmota_irhvac.pi.plant_model import PlantEstimate

        plant_id = self._pi._plant_id
        if tau_fast_seed is not None:
            plant_id._tau_fast_seed = float(tau_fast_seed)
        if tau_slow_seed is not None:
            plant_id._tau_slow_seed = float(tau_slow_seed)

        plant_id._plant = PlantEstimate.from_seeds(
            tau_fast_seed=plant_id._tau_fast_seed,
            tau_slow_seed=plant_id._tau_slow_seed,
            response_lag=plant_id._response_lag,
        )

        if plant_id.enabled:
            gains = plant_id.compute_gains()
            self._pi._pi_kp = gains.kp
            self._pi._pi_ki = gains.ki
            if self._pi._smith is not None:
                self._pi._smith.update_params(tau=gains.tau_fast, lag=gains.lag)

    def tick(self, room_temp_c, outdoor_temp_c, dt_seconds,
             model_inputs=None):
        """Run one PI tick and return HP setpoint."""
        self._sim_clock += dt_seconds
        self._entity._attr_current_temperature = room_temp_c
        self._pi._inputs.outdoor_temp = outdoor_temp_c

        # Update model inputs if provided
        if model_inputs:
            for i, m_input in enumerate(self._pi._model_inputs):
                name = m_input.get("name", "")
                if name in model_inputs:
                    self._pi._inputs.values[i] = model_inputs[name]

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
        smith = self._pi._smith
        return {
            "integral": self._pi._pi_integral,
            "ff_offset": self._pi._ff_offset,
            "d_term": getattr(self._pi, "_pi_d_filtered", 0.0),
            "rls_obs_count": self._pi._rls_heat.observation_count,
            "desired_temp": self._pi._desired_temp,
            "hp_setpoint": self._pi._hp_setpoint,
            "raw_setpoint": getattr(self._pi, "_last_raw_setpoint", 0.0),
            "smith_correction": smith.correction if smith is not None else 0.0,
        }

    def set_hold_time(self, seconds: float):
        """Override the setpoint hold timer for testing."""
        self._pi._SETPOINT_HOLD_SECONDS = seconds

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


class _FakeBenchEntity(_PITestEntityRoomTempMixin):
    """Minimal fake entity for PIController adapter."""

    def __init__(self, config, head_calibration_bounds=None):
        from custom_components.tasmota_irhvac.pi.pi_controller import PIController

        self.hass = MagicMock()
        self.hass.states.get = MagicMock(return_value=None)
        self._pi_test_room_temp = 20.0  # mixin backing field
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
        self._sync_room_temp_to_pi()
        self._pi._pi_enabled = True
        # Head calibration bounds: None = production defaults (±2.0°C),
        # explicit tuple overrides both heat and cool modes.
        if head_calibration_bounds is not None:
            cal_min, cal_max = head_calibration_bounds
            self._pi._head_calibration_min_heat = cal_min
            self._pi._head_calibration_max_heat = cal_max
            self._pi._head_calibration_min_cool = cal_min
            self._pi._head_calibration_max_cool = cal_max

    @property
    def device_info(self):
        return None

    def async_schedule_update_ha_state(self, force_refresh=False):
        pass

    async def send_ir(self, *args, **kwargs):
        pass
