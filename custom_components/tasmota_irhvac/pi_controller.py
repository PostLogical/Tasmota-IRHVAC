"""Vendor-agnostic PI + feedforward controller for IRHVAC climate entities.

Composed object (not a mixin). The climate entity creates a PIController instance
and calls its hook methods at the appropriate points. This avoids MRO issues
and minimizes changes to the upstream-derived climate.py.
"""

import dataclasses
import logging
import time
from datetime import timedelta
from typing import Any, Self

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.restore_state import ExtraStoredData
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.util.unit_conversion import TemperatureConverter

from .const import (
    ATTR_DESIRED_TEMP,
    ATTR_FF_COOL_BUCKETS,
    ATTR_FF_HEAT_BUCKETS,
    ATTR_FF_OFFSET,
    ATTR_HP_SETPOINT,
    ATTR_PI_INTEGRAL,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PI_DEADBAND,
    CONF_PI_DISTURBANCE_INPUTS,
    CONF_PI_ENABLED,
    CONF_PI_FF_ANTICIPATED_CHANGE_ENTITY,
    CONF_PI_FF_ANTICIPATED_CHANGE_GAIN,
    CONF_PI_FF_COOL_REFERENCE,
    CONF_PI_FF_COOL_SLOPE,
    CONF_PI_FF_HEAT_REFERENCE,
    CONF_PI_FF_HEAT_SLOPE,
    CONF_PI_FF_LEARN_NIGHT_ONLY,
    CONF_PI_FF_LEARN_SUNSET_DELAY,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_MIN_INTERVAL,
    CONF_PI_SETPOINT_WEIGHT,
    DEFAULT_PI_DEADBAND,
    DEFAULT_PI_ENABLED,
    DEFAULT_PI_FF_ALPHA,
    DEFAULT_PI_FF_ALPHA_OVERSHOOT_RATIO,
    DEFAULT_PI_FF_ANTICIPATED_CHANGE_GAIN,
    DEFAULT_PI_FF_COOL_REFERENCE,
    DEFAULT_PI_FF_COOL_SLOPE,
    DEFAULT_PI_FF_HEAT_REFERENCE,
    DEFAULT_PI_FF_HEAT_SLOPE,
    DEFAULT_PI_FF_LEARN_NIGHT_ONLY,
    DEFAULT_PI_FF_LEARN_SUNSET_DELAY,
    DEFAULT_PI_KI,
    DEFAULT_PI_KP,
    DEFAULT_PI_MIN_INTERVAL,
    DEFAULT_PI_SETPOINT_WEIGHT,
    SIGNAL_FF_SUPPRESS_UPDATE,
    SIGNAL_PI_UPDATE,
)

_LOGGER = logging.getLogger(__name__)


def _seed_buckets(reference, slope, is_cooling=False):
    """Seed feedforward buckets from a linear approximation."""
    buckets = {}
    for bucket_temp in range(-30, 48, 3):
        if is_cooling:
            delta = max(0, bucket_temp - reference)
            buckets[bucket_temp] = -slope * delta
        else:
            delta = max(0, reference - bucket_temp)
            buckets[bucket_temp] = slope * delta
    return buckets


@dataclasses.dataclass
class PIExtraStoredData(ExtraStoredData):
    """PI controller data persisted via RestoreEntity's ExtraStoredData mechanism.

    Stores FF buckets, integral, desired_temp, and hp_setpoint so they survive
    restarts without bloating the recorder DB on every state write.
    """

    ff_heat_buckets: dict[int, float]
    ff_cool_buckets: dict[int, float]
    pi_integral: float
    desired_temp: float | None
    hp_setpoint: float | None
    ff_bucket_observation_counts: dict[int, int] = dataclasses.field(default_factory=dict)
    integral_convergence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "ff_heat_buckets": {str(k): v for k, v in self.ff_heat_buckets.items()},
            "ff_cool_buckets": {str(k): v for k, v in self.ff_cool_buckets.items()},
            "pi_integral": self.pi_integral,
            "desired_temp": self.desired_temp,
            "hp_setpoint": self.hp_setpoint,
            "ff_bucket_observation_counts": {str(k): v for k, v in self.ff_bucket_observation_counts.items()},
            "integral_convergence": self.integral_convergence,
        }

    @classmethod
    def from_dict(cls, restored: dict[str, Any]) -> Self | None:
        """Deserialize from stored dict."""
        try:
            obs_counts = {}
            if "ff_bucket_observation_counts" in restored:
                obs_counts = {int(k): int(v) for k, v in restored["ff_bucket_observation_counts"].items()}
            return cls(
                ff_heat_buckets={int(k): float(v) for k, v in restored["ff_heat_buckets"].items()},
                ff_cool_buckets={int(k): float(v) for k, v in restored["ff_cool_buckets"].items()},
                pi_integral=float(restored["pi_integral"]),
                desired_temp=restored.get("desired_temp"),
                hp_setpoint=restored.get("hp_setpoint"),
                ff_bucket_observation_counts=obs_counts,
                integral_convergence=float(restored.get("integral_convergence", 0.0)),
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            return None


class PIController:
    """PI + feedforward temperature controller for IRHVAC climate entities.

    Usage in climate entity:
        __init__:           self._pi = PIController(self, config)
        async_added_to_hass:     await self._pi.async_added_to_hass(old_state)
        async_will_remove:       self._pi.async_will_remove_from_hass()
        async_set_temperature:   await self._pi.set_temperature(temp, hvac_mode)
        _handle_state_payload:   await self._pi.handle_state_payload(payload)
        _async_sensor_changed:   await self._pi.sensor_changed(was_none)
        _get_ir_temp:            return self._pi.get_ir_temp()
        extra_state_attributes:  attrs.update(self._pi.get_extra_state_attributes())
        hvac_modes:              return self._pi.filter_hvac_modes(modes)
        async_set_hvac_mode:     if self._pi.should_reject_hvac_mode(mode): return
        async_write_ha_state:    self._pi.fire_dispatcher()

    Vendor subclasses use: pi_pause(), pi_resume(), pi_reset_integral()
    """

    def __init__(self, entity, config):
        """Initialize PI controller.

        Args:
            entity: The climate entity this controller is attached to.
            config: Merged config dict (entry.data + entry.options).
        """
        self._entity = entity

        # Convert entity temp limits to °C for internal PI math
        self._min_temp_c = TemperatureConverter.convert(
            entity._min_temp, entity._attr_temperature_unit, UnitOfTemperature.CELSIUS
        )
        self._max_temp_c = TemperatureConverter.convert(
            entity._max_temp, entity._attr_temperature_unit, UnitOfTemperature.CELSIUS
        )

        # PI controller config
        self._pi_enabled = config.get(CONF_PI_ENABLED, DEFAULT_PI_ENABLED)
        self._pi_kp = config.get(CONF_PI_KP, DEFAULT_PI_KP)
        self._pi_ki = config.get(CONF_PI_KI, DEFAULT_PI_KI)
        self._pi_min_interval = config.get(CONF_PI_MIN_INTERVAL, DEFAULT_PI_MIN_INTERVAL)
        self._pi_deadband = config.get(CONF_PI_DEADBAND, DEFAULT_PI_DEADBAND)
        self._pi_setpoint_weight = config.get(CONF_PI_SETPOINT_WEIGHT, DEFAULT_PI_SETPOINT_WEIGHT)

        # Feedforward config
        self._outdoor_temp_sensor = config.get(CONF_OUTDOOR_TEMP_SENSOR)
        self._ff_heat_reference = config.get(CONF_PI_FF_HEAT_REFERENCE, DEFAULT_PI_FF_HEAT_REFERENCE)
        self._ff_heat_slope = config.get(CONF_PI_FF_HEAT_SLOPE, DEFAULT_PI_FF_HEAT_SLOPE)
        self._ff_cool_reference = config.get(CONF_PI_FF_COOL_REFERENCE, DEFAULT_PI_FF_COOL_REFERENCE)
        self._ff_cool_slope = config.get(CONF_PI_FF_COOL_SLOPE, DEFAULT_PI_FF_COOL_SLOPE)

        # FF learning config
        self._ff_learn_night_only = config.get(CONF_PI_FF_LEARN_NIGHT_ONLY, DEFAULT_PI_FF_LEARN_NIGHT_ONLY)
        self._ff_learn_sunset_delay = config.get(CONF_PI_FF_LEARN_SUNSET_DELAY, DEFAULT_PI_FF_LEARN_SUNSET_DELAY) * 60  # Convert min → sec
        self._ff_alpha = DEFAULT_PI_FF_ALPHA
        self._ff_alpha_overshoot_ratio = DEFAULT_PI_FF_ALPHA_OVERSHOOT_RATIO
        self._ff_min_observation_hours = 4.0  # Hours of data before EMA overwrites seed

        # Anticipated change FF config
        self._anticipated_change_entity = config.get(CONF_PI_FF_ANTICIPATED_CHANGE_ENTITY, "")
        self._anticipated_change_gain = config.get(CONF_PI_FF_ANTICIPATED_CHANGE_GAIN, DEFAULT_PI_FF_ANTICIPATED_CHANGE_GAIN)

        # Disturbance inputs (replaces single suppress/bias entities)
        # Fallback: convert legacy keys if disturbance_inputs is empty (e.g., YAML import)
        self._disturbance_inputs = config.get(CONF_PI_DISTURBANCE_INPUTS, [])
        if not self._disturbance_inputs:
            old_suppress = config.get("pi_ff_suppress_learning_entity", "")
            old_bias = config.get("pi_ff_bias_entity", "")
            if old_suppress:
                self._disturbance_inputs.append({
                    "name": "Suppress Entity (migrated)",
                    "entity_id": old_suppress,
                    "suppress_learning": True,
                    "default_bias": 0.0,
                    "gain": 1.0,
                })
            if old_bias:
                self._disturbance_inputs.append({
                    "name": "Bias Entity (migrated)",
                    "entity_id": old_bias,
                    "suppress_learning": False,
                    "default_bias": 0.0,
                    "gain": 1.0,
                })
        self._manual_ff_suppress = False
        self._manual_ff_suppress_reason = ""
        self._last_disturbance_bias = 0.0
        self._disturbance_suppress_active = False
        self._disturbance_active_suppressors = []
        self._disturbance_total_bias = 0.0

        # PI controller state
        self._desired_temp = entity._attr_target_temperature
        self._hp_setpoint = entity._attr_target_temperature
        self._pi_integral = 0.0
        self._pi_timer_unsub = None
        self._ff_offset = 0.0
        self._pi_command_pending = False
        self._ff_settled_ticks = 0
        self._sensor_unavailable = False
        self._sensor_recovery_pending = False
        self._sensor_recovery_unsub = None
        self._pi_paused = False
        self._pi_last_tick_time = 0.0
        self._pi_last_error = 0.0

        # Feedforward buckets
        self._ff_heat_buckets = _seed_buckets(self._ff_heat_reference, self._ff_heat_slope)
        self._ff_cool_buckets = _seed_buckets(
            self._ff_cool_reference, self._ff_cool_slope, is_cooling=True
        )
        self._ff_bucket_observation_counts: dict[int, int] = {}
        self._ff_bucket_first_obs_time: dict[int, float] = {}

        # Outdoor temp state
        self._outdoor_temp = None

        # Anticipated change state
        self._anticipated_change = 0.0
        self._ff_anticipated_offset = 0.0

        # Night learning state
        self._sun_below_horizon_since = 0.0  # monotonic timestamp

        # Integral convergence tracking (EMA of abs(integral) over ~24hr)
        self._integral_convergence = 0.0

    # ── Shorthand entity access ──────────────────────────────────────

    @property
    def _hass(self):
        return self._entity.hass

    # ── Lifecycle hooks (called by climate entity) ───────────────────

    async def async_added_to_hass(self, old_state=None):
        """Set up PI after entity is added to HA."""
        if not self._pi_enabled:
            return

        e = self._entity

        # Restore PI state — prefer ExtraStoredData, fall back to state attributes
        extra_data = await e.async_get_last_extra_data()
        if extra_data is not None:
            pi_data = PIExtraStoredData.from_dict(extra_data.as_dict())
            if pi_data is not None:
                self.restore_extra_stored_data(pi_data)
                _LOGGER.debug("PI: restored from ExtraStoredData")
        else:
            # Fall back to state attributes (migration from pre-ExtraStoredData versions)
            if old_state is None:
                old_state = await e.async_get_last_state()
            if old_state is not None:
                attrs = old_state.attributes
                if attrs.get(ATTR_PI_INTEGRAL) is not None:
                    self._pi_integral = max(-50, min(50, float(attrs[ATTR_PI_INTEGRAL])))
                if attrs.get(ATTR_DESIRED_TEMP) is not None:
                    self._desired_temp = float(attrs[ATTR_DESIRED_TEMP])
                if attrs.get(ATTR_HP_SETPOINT) is not None:
                    self._hp_setpoint = float(attrs[ATTR_HP_SETPOINT])
                if attrs.get(ATTR_FF_HEAT_BUCKETS) is not None:
                    self._ff_heat_buckets = {
                        int(k): float(v) for k, v in attrs[ATTR_FF_HEAT_BUCKETS].items()
                    }
                if attrs.get(ATTR_FF_COOL_BUCKETS) is not None:
                    self._ff_cool_buckets = {
                        int(k): float(v) for k, v in attrs[ATTR_FF_COOL_BUCKETS].items()
                    }
                _LOGGER.debug("PI: restored from state attributes (legacy)")

        # Fallback: sync with restored _attr_target_temperature
        if self._desired_temp is None and e._attr_target_temperature is not None:
            self._desired_temp = e._attr_target_temperature
        if self._hp_setpoint is None and e._attr_target_temperature is not None:
            self._hp_setpoint = TemperatureConverter.convert(
                e._attr_target_temperature,
                e.temperature_unit,
                UnitOfTemperature.CELSIUS,
            )

        # Register outdoor temp sensor
        if self._outdoor_temp_sensor:
            async_track_state_change_event(
                self._hass,
                self._outdoor_temp_sensor,
                self._async_outdoor_temp_changed,
            )
            outdoor_state = self._hass.states.get(self._outdoor_temp_sensor)
            if outdoor_state is not None:
                self._update_outdoor_temp(outdoor_state)

        # Subscribe to disturbance input entities for real-time updates
        disturbance_entity_ids = [
            d["entity_id"] for d in self._disturbance_inputs if d.get("entity_id")
        ]
        if disturbance_entity_ids:
            async_track_state_change_event(
                self._hass,
                disturbance_entity_ids,
                self._async_disturbance_entity_changed,
            )

        # Register anticipated change entity
        if self._anticipated_change_entity:
            async_track_state_change_event(
                self._hass,
                self._anticipated_change_entity,
                self._async_anticipated_change_changed,
            )
            ac_state = self._hass.states.get(self._anticipated_change_entity)
            if ac_state is not None and ac_state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                try:
                    self._anticipated_change = float(ac_state.state)
                except (ValueError, TypeError):
                    pass

        # Register sun.sun for night-only learning
        if self._ff_learn_night_only:
            async_track_state_change_event(
                self._hass, "sun.sun", self._async_sun_state_changed,
            )
            sun_state = self._hass.states.get("sun.sun")
            if sun_state is not None and sun_state.state == "below_horizon":
                self._sun_below_horizon_since = time.monotonic()

        # Compute initial feedforward offset
        if self._outdoor_temp is not None:
            is_heating = e._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
            buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets
            bucket_key = round(self._outdoor_temp / 3) * 3
            self._ff_offset = buckets.get(bucket_key, 0.0)

        # Start fallback timer
        if e._temp_sensor:
            self._pi_timer_unsub = async_track_time_interval(
                self._hass,
                self._pi_tick,
                timedelta(seconds=self._pi_min_interval),
            )
            if e._attr_current_temperature is not None:
                await self._pi_tick()
            else:
                _LOGGER.debug("PI: skipping initial tick, waiting for sensor")

    def async_will_remove_from_hass(self):
        """Clean up PI timers."""
        if self._pi_timer_unsub:
            self._pi_timer_unsub()
            self._pi_timer_unsub = None
        if self._sensor_recovery_unsub:
            self._sensor_recovery_unsub()
            self._sensor_recovery_unsub = None

    # ── Hook methods (called by climate entity) ──────────────────────

    def get_extra_stored_data(self) -> PIExtraStoredData | None:
        """Return PI data for RestoreEntity's ExtraStoredData persistence."""
        if not self._pi_enabled:
            return None
        return PIExtraStoredData(
            ff_heat_buckets=dict(self._ff_heat_buckets),
            ff_cool_buckets=dict(self._ff_cool_buckets),
            pi_integral=self._pi_integral,
            desired_temp=self._desired_temp,
            hp_setpoint=self._hp_setpoint,
            ff_bucket_observation_counts=dict(self._ff_bucket_observation_counts),
            integral_convergence=self._integral_convergence,
        )

    def restore_extra_stored_data(self, data: PIExtraStoredData) -> None:
        """Restore PI data from ExtraStoredData."""
        self._ff_heat_buckets = data.ff_heat_buckets
        self._ff_cool_buckets = data.ff_cool_buckets
        self._pi_integral = max(-50, min(50, data.pi_integral))
        if data.desired_temp is not None:
            self._desired_temp = data.desired_temp
        if data.hp_setpoint is not None:
            self._hp_setpoint = data.hp_setpoint
        if data.ff_bucket_observation_counts:
            self._ff_bucket_observation_counts = data.ff_bucket_observation_counts
        self._integral_convergence = data.integral_convergence

    async def set_temperature(self, temperature, hvac_mode=None):
        """Handle temperature set when PI is active."""
        if temperature is None:
            return
        e = self._entity
        if hvac_mode is not None:
            await e.set_mode(hvac_mode)
        self._desired_temp = temperature
        e._attr_target_temperature = temperature
        self._pi_integral = 0.0
        if e._attr_hvac_mode != HVACMode.OFF:
            e.power_mode = STATE_ON
        await self._pi_tick()
        e.async_schedule_update_ha_state()

    async def handle_state_payload(self, payload):
        """Handle MQTT state echo. Call after base class processes payload."""
        if not self._pi_enabled or self._desired_temp is None:
            return
        e = self._entity
        # Capture HP setpoint from echo, restore user's desired temp
        if "Temp" in payload and payload["Temp"] > 0:
            self._hp_setpoint = payload["Temp"]
        e._attr_target_temperature = self._desired_temp
        e.async_write_ha_state()
        # Echo detection
        if "Temp" in payload and payload["Temp"] > 0:
            if self._pi_command_pending:
                self._pi_command_pending = False
            else:
                self._desired_temp = e._attr_target_temperature
                await self._pi_tick()

    async def sensor_changed(self, was_none):
        """Handle temp sensor update."""
        await self._pi_async_sensor_changed(was_none=was_none)

    def get_ir_temp(self):
        """Return PI-computed setpoint for IR command."""
        return round(self._hp_setpoint)

    def get_extra_state_attributes(self):
        """Return PI state attributes to merge into entity attributes."""
        if not self._pi_enabled:
            return {}
        return {
            ATTR_HP_SETPOINT: self._hp_setpoint,
            ATTR_PI_INTEGRAL: round(self._pi_integral, 3),
            ATTR_DESIRED_TEMP: self._desired_temp,
            ATTR_FF_OFFSET: round(self._ff_offset, 2),
            ATTR_FF_HEAT_BUCKETS: {
                str(k): round(v, 2) for k, v in self._ff_heat_buckets.items()
            },
            ATTR_FF_COOL_BUCKETS: {
                str(k): round(v, 2) for k, v in self._ff_cool_buckets.items()
            },
            "ff_learning_suppressed": self._disturbance_suppress_active,
            "disturbance_bias": round(self._disturbance_total_bias, 2),
            "ff_anticipated_offset": round(self._ff_anticipated_offset, 2),
            "integral_convergence": round(self._integral_convergence, 2),
        }

    def filter_hvac_modes(self, modes):
        """Filter out auto/heat_cool when PI is enabled."""
        if self._pi_enabled and modes:
            return [m for m in modes if m not in (HVACMode.AUTO, HVACMode.HEAT_COOL)]
        return modes

    def should_reject_hvac_mode(self, hvac_mode):
        """Return True if PI should reject this HVAC mode."""
        return self._pi_enabled and hvac_mode in (HVACMode.AUTO, HVACMode.HEAT_COOL)

    def fire_dispatcher(self):
        """Fire dispatcher signal for companion PI sensors."""
        if self._pi_enabled and hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_PI_UPDATE.format(self._entity._config_entry_id),
            )

    # ── Public API (for vendor subclasses via entity._pi) ────────────

    def pi_pause(self):
        """Pause PI control (e.g., during vendor-specific preset modes)."""
        self._pi_paused = True

    def pi_resume(self):
        """Resume PI control after a pause."""
        self._pi_paused = False

    def pi_reset_integral(self):
        """Zero the integral (e.g., after mode changes that invalidate it)."""
        self._pi_integral = 0.0

    async def async_suppress_ff_learning(self, reason=""):
        """Manually suppress FF learning (service call handler)."""
        self._manual_ff_suppress = True
        self._manual_ff_suppress_reason = reason or ""
        _LOGGER.info("FF learning manually suppressed: %s", reason or "(no reason)")
        if hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._entity._config_entry_id),
            )

    async def async_resume_ff_learning(self):
        """Resume FF learning after manual suppression (service call handler)."""
        self._manual_ff_suppress = False
        self._manual_ff_suppress_reason = ""
        _LOGGER.info("FF learning manual suppress cleared")
        if hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._entity._config_entry_id),
            )

    async def async_reset_ff_buckets(self):
        """Reset feedforward buckets to seed values from config."""
        self._ff_heat_buckets = _seed_buckets(self._ff_heat_reference, self._ff_heat_slope)
        self._ff_cool_buckets = _seed_buckets(
            self._ff_cool_reference, self._ff_cool_slope, is_cooling=True
        )
        self._pi_integral = 0.0
        _LOGGER.info("FF buckets reset to seed values, integral zeroed")
        self._entity.async_schedule_update_ha_state()

    def _compute_disturbance_effects(self):
        """Compute combined suppress and bias from disturbance inputs.

        Returns (suppress: bool, active_suppressors: list[str], total_bias: float).
        """
        suppress = self._manual_ff_suppress
        active_suppressors = []
        total_bias = 0.0

        for d_input in self._disturbance_inputs:
            entity_id = d_input.get("entity_id")
            if not entity_id:
                continue
            state = self._hass.states.get(entity_id)
            if not state or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                continue

            # Try to interpret as numeric first
            try:
                value = float(state.state)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            if is_numeric:
                if value != 0:
                    if d_input.get("suppress_learning"):
                        suppress = True
                        active_suppressors.append(entity_id)
                    total_bias += value * d_input.get("gain", 1.0)
            else:
                if state.state == "on":
                    if d_input.get("suppress_learning"):
                        suppress = True
                        active_suppressors.append(entity_id)
                    total_bias += d_input.get("default_bias", 0.0)

        return suppress, active_suppressors, total_bias

    # ── PI Internals ──────────────────────────────────────────────────

    @callback
    def _update_outdoor_temp(self, state):
        """Update outdoor temperature from sensor state, converting to °C."""
        try:
            temp = float(state.state)
            unit = state.attributes.get("unit_of_measurement", UnitOfTemperature.CELSIUS)
            self._outdoor_temp = TemperatureConverter.convert(
                temp, unit, UnitOfTemperature.CELSIUS
            )
        except (ValueError, TypeError):
            pass

    @callback
    def _async_outdoor_temp_changed(self, event):
        """Handle outdoor temperature sensor state changes."""
        new_state = event.data.get("new_state")
        if new_state is not None:
            self._update_outdoor_temp(new_state)

    @callback
    def _async_anticipated_change_changed(self, event):
        """Handle anticipated change entity state changes."""
        new_state = event.data.get("new_state")
        if new_state is not None and new_state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            try:
                self._anticipated_change = float(new_state.state)
            except (ValueError, TypeError):
                self._anticipated_change = 0.0
        else:
            self._anticipated_change = 0.0

    @callback
    def _async_sun_state_changed(self, event):
        """Track when sun goes below horizon for night-only learning."""
        new_state = event.data.get("new_state")
        if new_state is not None:
            if new_state.state == "below_horizon":
                if self._sun_below_horizon_since == 0.0:
                    self._sun_below_horizon_since = time.monotonic()
            else:
                self._sun_below_horizon_since = 0.0

    def _is_learning_time_allowed(self) -> bool:
        """Check if FF learning is allowed based on night-only setting."""
        if not self._ff_learn_night_only:
            return True
        # Check sun.sun entity state
        sun_state = self._hass.states.get("sun.sun")
        if sun_state is None:
            return True  # No sun entity — allow learning always
        if sun_state.state != "below_horizon":
            return False
        # Check sunset delay
        if self._sun_below_horizon_since == 0.0:
            # First check — set the timestamp now
            self._sun_below_horizon_since = time.monotonic()
            return False
        elapsed = time.monotonic() - self._sun_below_horizon_since
        return elapsed >= self._ff_learn_sunset_delay

    @callback
    def _async_disturbance_entity_changed(self, event):
        """Handle disturbance input entity state changes — update binary sensor."""
        if hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._entity._config_entry_id),
            )

    async def _pi_async_sensor_changed(self, was_none=False):
        """Handle temp sensor update."""
        if not self._pi_enabled:
            return
        if was_none:
            if self._sensor_recovery_unsub:
                self._sensor_recovery_unsub()
                self._sensor_recovery_unsub = None
            self._sensor_recovery_pending = False
            if self._sensor_unavailable:
                _LOGGER.info("PI: temp sensor recovered, resuming full PI control")
                self._sensor_unavailable = False
            else:
                _LOGGER.debug("PI: temp sensor just became available, running immediate tick")
            await self._pi_tick()
            return

        elapsed = time.monotonic() - self._pi_last_tick_time
        min_cooldown = max(60.0, self._pi_min_interval / 3.0)
        if elapsed >= min_cooldown:
            await self._pi_tick()

    async def _check_sensor_recovery(self, _now=None):
        """Called 60s after sensor went unavailable. Fall back to FF-only if still gone."""
        self._sensor_recovery_pending = False
        self._sensor_recovery_unsub = None
        e = self._entity
        if e._attr_current_temperature is not None:
            _LOGGER.info("PI: temp sensor recovered during grace period")
            await self._pi_tick()
            return
        self._sensor_unavailable = True
        _LOGGER.warning("PI: temp sensor confirmed unavailable, using feedforward-only fallback")
        if self._desired_temp is None:
            return
        desired_c = TemperatureConverter.convert(
            self._desired_temp, e.temperature_unit, UnitOfTemperature.CELSIUS,
        )
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        if not is_heating and e._attr_hvac_mode not in (HVACMode.COOL, HVACMode.DRY):
            return
        ff_offset = 0.0
        if self._outdoor_temp is not None:
            bucket_key = round(self._outdoor_temp / 3) * 3
            buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets
            ff_offset = buckets.get(bucket_key, 0.0)
        self._ff_offset = ff_offset
        self._pi_integral = 0.0
        new_setpoint = round(max(self._min_temp_c, min(self._max_temp_c, desired_c + ff_offset)))
        if new_setpoint != self._hp_setpoint:
            _LOGGER.info("PI fallback: setpoint %s -> %s (FF only)", self._hp_setpoint, new_setpoint)
            self._hp_setpoint = new_setpoint
            self._pi_command_pending = True
            await e.send_ir()
        e.async_schedule_update_ha_state()

    async def _pi_tick(self, now=None):
        """PI + feedforward controller tick. Called by timer and sensor events."""
        if not self._pi_enabled:
            return
        e = self._entity
        if e._attr_hvac_mode == HVACMode.OFF:
            self._pi_integral = 0.0
            return
        if self._desired_temp is None:
            return
        if e._attr_current_temperature is None:
            if self._sensor_unavailable or self._sensor_recovery_pending:
                return
            _LOGGER.info("PI: temp sensor unavailable, scheduling 60s recovery check")
            self._sensor_recovery_pending = True
            self._sensor_recovery_unsub = async_call_later(
                self._hass, 60, self._check_sensor_recovery
            )
            return
        if self._pi_paused:
            _LOGGER.debug("PI tick: skipping, paused by vendor")
            return

        # Time since last tick (for time-normalized integral)
        now_mono = time.monotonic()
        if self._pi_last_tick_time > 0:
            dt_seconds = min(now_mono - self._pi_last_tick_time, self._pi_min_interval * 2)
        else:
            dt_seconds = float(self._pi_min_interval)
        self._pi_last_tick_time = now_mono
        dt_factor = dt_seconds / float(self._pi_min_interval)

        # Convert both to °C for PI math
        current_c = TemperatureConverter.convert(
            e._attr_current_temperature,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )
        desired_c = TemperatureConverter.convert(
            self._desired_temp,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )
        error = desired_c - current_c

        # PI only operates in explicit HEAT, COOL, or DRY modes
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        is_cooling = e._attr_hvac_mode in (HVACMode.COOL, HVACMode.DRY)
        if not is_heating and not is_cooling:
            return

        # Feedforward from outdoor temp buckets
        self._ff_offset = 0.0
        if self._outdoor_temp is not None:
            bucket_key = round(self._outdoor_temp / 3) * 3
            if is_heating:
                raw_ff = self._ff_heat_buckets.get(bucket_key, 0.0)
                ff_scale = max(0.0, min(1.0, 1.0 + error / self._pi_deadband))
            else:
                raw_ff = self._ff_cool_buckets.get(bucket_key, 0.0)
                ff_scale = max(0.0, min(1.0, 1.0 - error / self._pi_deadband))
            self._ff_offset = raw_ff * ff_scale

        # Disturbance inputs: compute suppress + bias
        learning_suppressed, active_suppressors, disturbance_bias = (
            self._compute_disturbance_effects()
        )
        self._disturbance_suppress_active = learning_suppressed
        self._disturbance_active_suppressors = active_suppressors
        self._disturbance_total_bias = disturbance_bias
        self._ff_offset += disturbance_bias

        # Anticipated change feedforward
        if self._anticipated_change_entity and self._anticipated_change != 0.0:
            self._ff_anticipated_offset = self._anticipated_change_gain * self._anticipated_change
            self._ff_offset += self._ff_anticipated_offset
        else:
            self._ff_anticipated_offset = 0.0

        # Reset integral on large disturbance bias transitions
        if abs(disturbance_bias - self._last_disturbance_bias) > 1.0:
            _LOGGER.info(
                "PI: Disturbance bias changed by %.1f°C, resetting integral",
                disturbance_bias - self._last_disturbance_bias,
            )
            self._pi_integral = 0.0
        self._last_disturbance_bias = disturbance_bias

        # Adaptive setpoint weight
        abs_error = abs(error)
        if abs_error > self._pi_deadband * 4:
            effective_weight = 1.0
        elif abs_error > self._pi_deadband:
            blend = (abs_error - self._pi_deadband) / (self._pi_deadband * 3)
            effective_weight = self._pi_setpoint_weight + blend * (1.0 - self._pi_setpoint_weight)
        else:
            effective_weight = self._pi_setpoint_weight

        # Deadband: if error is small, skip P term and decay integral
        in_deadband = abs_error < self._pi_deadband
        if in_deadband:
            self._pi_integral *= 0.9
            self._ff_settled_ticks += 1
            # Auto-learn: record offset when settled for 2+ ticks
            can_learn = (
                self._ff_settled_ticks >= 2
                and self._outdoor_temp is not None
                and not learning_suppressed
                and self._is_learning_time_allowed()
            )
            if can_learn:
                # Learn from total need (FF + integral contribution)
                observed_offset = (
                    self._hp_setpoint
                    + (self._pi_ki * self._pi_integral)
                    - desired_c
                )
                bucket_key = round(self._outdoor_temp / 3) * 3
                learn_buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets
                old = learn_buckets.get(bucket_key, 0.0)

                # Seed protection: don't EMA until enough time has passed
                first_obs_time = self._ff_bucket_first_obs_time.get(bucket_key)
                if first_obs_time is None:
                    self._ff_bucket_first_obs_time[bucket_key] = now_mono
                    first_obs_time = now_mono
                self._ff_bucket_observation_counts[bucket_key] = (
                    self._ff_bucket_observation_counts.get(bucket_key, 0) + 1
                )
                hours_observed = (now_mono - first_obs_time) / 3600.0
                if hours_observed >= self._ff_min_observation_hours:
                    # Asymmetric learning: faster for undershoot, slower for overshoot
                    needed_more = (
                        (is_heating and observed_offset > old)
                        or (is_cooling and observed_offset < old)
                    )
                    alpha = self._ff_alpha if needed_more else self._ff_alpha * self._ff_alpha_overshoot_ratio
                    learn_buckets[bucket_key] = (1.0 - alpha) * old + alpha * observed_offset
                # else: keep seed value until enough observations
            elif learning_suppressed and self._ff_settled_ticks >= 2:
                _LOGGER.debug(
                    "PI: FF learning suppressed by %s",
                    active_suppressors if active_suppressors else "manual",
                )
            p_term = 0.0
        else:
            self._ff_settled_ticks = 0
            p_error = effective_weight * (desired_c - current_c)
            p_term = self._pi_kp * p_error
            avg_error = (error + self._pi_last_error) / 2.0
            self._pi_integral += avg_error * dt_factor

        self._pi_last_error = error

        # Discard stale integral on overshoot recovery
        if is_heating and self._pi_integral < 0 and error >= -self._pi_deadband:
            self._pi_integral = 0.0
        elif is_cooling and self._pi_integral > 0 and error <= self._pi_deadband:
            self._pi_integral = 0.0

        # Hard safety cap on integral
        self._pi_integral = max(-50.0, min(50.0, self._pi_integral))

        # Update integral convergence metric (EMA of abs(integral), ~24hr time constant)
        # With 15-min ticks, 96 ticks/day → alpha ≈ 1/96 ≈ 0.01
        convergence_alpha = 0.01
        self._integral_convergence = (
            (1.0 - convergence_alpha) * self._integral_convergence
            + convergence_alpha * abs(self._pi_integral)
        )

        i_term = self._pi_ki * self._pi_integral
        raw_setpoint = desired_c + p_term + i_term + self._ff_offset
        clamped_setpoint = max(self._min_temp_c, min(self._max_temp_c, raw_setpoint))

        # Back-calculation anti-windup
        if self._pi_ki != 0:
            saturation_error = clamped_setpoint - raw_setpoint
            if abs(saturation_error) > 0.01:
                kb = 1.0 / self._pi_ki
                self._pi_integral += kb * saturation_error
                self._pi_integral = max(-50.0, min(50.0, self._pi_integral))

        # Midpoint-crossing hysteresis
        new_setpoint = self._hp_setpoint
        if clamped_setpoint > self._hp_setpoint + 0.5:
            new_setpoint = round(clamped_setpoint)
        elif clamped_setpoint < self._hp_setpoint - 0.5:
            new_setpoint = round(clamped_setpoint)
        new_setpoint = int(max(self._min_temp_c, min(self._max_temp_c, new_setpoint)))

        if new_setpoint != self._hp_setpoint:
            _LOGGER.info(
                "PI: error=%.1f P=%.1f I=%.1f FF=%.1f raw=%.1f setpoint %s -> %s",
                error, p_term, i_term, self._ff_offset, clamped_setpoint,
                self._hp_setpoint, new_setpoint,
            )
            self._hp_setpoint = new_setpoint
            self._pi_command_pending = True
            await e.send_ir()
        else:
            _LOGGER.debug(
                "PI: error=%.1f raw=%.1f setpoint=%s (held)",
                error, clamped_setpoint, self._hp_setpoint,
            )

        e.async_schedule_update_ha_state()
