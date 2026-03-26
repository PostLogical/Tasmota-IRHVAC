"""Vendor-agnostic PI + feedforward controller mixin for IRHVAC climate entities."""

import logging
import time
from datetime import timedelta

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
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
    CONF_PI_FF_COOL_REFERENCE,
    CONF_PI_FF_COOL_SLOPE,
    CONF_PI_FF_HEAT_REFERENCE,
    CONF_PI_FF_HEAT_SLOPE,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_MIN_INTERVAL,
    CONF_PI_SETPOINT_WEIGHT,
    DEFAULT_PI_DEADBAND,
    DEFAULT_PI_ENABLED,
    DEFAULT_PI_FF_COOL_REFERENCE,
    DEFAULT_PI_FF_COOL_SLOPE,
    DEFAULT_PI_FF_HEAT_REFERENCE,
    DEFAULT_PI_FF_HEAT_SLOPE,
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


class PIControllerMixin:
    """Mixin providing PI + feedforward temperature control for IRHVAC entities.

    Vendor subclasses should:
    - Call pi_init(config) in __init__ after super().__init__
    - Call pi_async_added_to_hass() in async_added_to_hass after super()
    - Call pi_async_will_remove_from_hass() in async_will_remove_from_hass
    - Use pi_pause() / pi_resume() / pi_reset_integral() to control PI from presets
    - Override _get_ir_temp() to return self._hp_setpoint when PI active
    """

    # ── PI Initialization ─────────────────────────────────────────────

    def pi_init(self, config):
        """Initialize PI controller state from config. Call after super().__init__."""
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

        # Disturbance inputs (replaces single suppress/bias entities)
        self._disturbance_inputs = config.get(CONF_PI_DISTURBANCE_INPUTS, [])
        self._manual_ff_suppress = False
        self._manual_ff_suppress_reason = ""
        self._last_disturbance_bias = 0.0
        self._disturbance_suppress_active = False
        self._disturbance_active_suppressors = []
        self._disturbance_total_bias = 0.0

        # PI controller state
        self._desired_temp = self._attr_target_temperature
        self._hp_setpoint = self._attr_target_temperature
        self._pi_integral = 0.0
        self._pi_timer_unsub = None
        self._ff_offset = 0.0
        self._pi_command_pending = False
        self._ff_settled_ticks = 0
        self._sensor_unavailable = False
        self._sensor_recovery_pending = False
        self._sensor_recovery_unsub = None
        self._pi_paused = False
        self._pi_last_tick_time = 0.0  # monotonic time of last tick
        self._pi_last_error = 0.0  # previous error for trapezoidal integral

        # Feedforward buckets
        self._ff_heat_buckets = _seed_buckets(self._ff_heat_reference, self._ff_heat_slope)
        self._ff_cool_buckets = _seed_buckets(
            self._ff_cool_reference, self._ff_cool_slope, is_cooling=True
        )

        # Outdoor temp state
        self._outdoor_temp = None

    async def pi_async_added_to_hass(self, old_state=None):
        """Set up PI after entity is added. Call after super().async_added_to_hass().

        Args:
            old_state: Previous entity state (optional, avoids duplicate lookup).
        """
        if not self._pi_enabled:
            return

        # Restore PI and feedforward state from previous session
        if old_state is None:
            old_state = await self.async_get_last_state()
        if old_state is not None:
            attrs = old_state.attributes
            if attrs.get(ATTR_PI_INTEGRAL) is not None:
                restored_integral = float(attrs[ATTR_PI_INTEGRAL])
                self._pi_integral = max(-50, min(50, restored_integral))
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

        # Fallback: sync with restored _attr_target_temperature from super()
        if self._desired_temp is None and self._attr_target_temperature is not None:
            self._desired_temp = self._attr_target_temperature
        if self._hp_setpoint is None and self._attr_target_temperature is not None:
            self._hp_setpoint = TemperatureConverter.convert(
                self._attr_target_temperature,
                self.temperature_unit,
                UnitOfTemperature.CELSIUS,
            )

        # Register outdoor temp sensor
        if self._outdoor_temp_sensor:
            async_track_state_change_event(
                self.hass,
                self._outdoor_temp_sensor,
                self._async_outdoor_temp_changed,
            )
            outdoor_state = self.hass.states.get(self._outdoor_temp_sensor)
            if outdoor_state is not None:
                self._update_outdoor_temp(outdoor_state)

        # Subscribe to disturbance input entities for real-time updates
        disturbance_entity_ids = [
            d["entity_id"] for d in self._disturbance_inputs if d.get("entity_id")
        ]
        if disturbance_entity_ids:
            async_track_state_change_event(
                self.hass,
                disturbance_entity_ids,
                self._async_disturbance_entity_changed,
            )

        # Compute initial feedforward offset
        if self._outdoor_temp is not None:
            is_heating = self._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
            buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets
            bucket_key = round(self._outdoor_temp / 3) * 3
            self._ff_offset = buckets.get(bucket_key, 0.0)

        # Start fallback timer (catches outdoor temp changes when room sensor is stable)
        # Primary ticking is event-driven via _pi_async_sensor_changed
        if self._temp_sensor:
            self._pi_timer_unsub = async_track_time_interval(
                self.hass,
                self._pi_tick,
                timedelta(seconds=self._pi_min_interval),
            )
            if self._attr_current_temperature is not None:
                await self._pi_tick()
            else:
                _LOGGER.debug("PI: skipping initial tick, waiting for sensor")

    def pi_async_will_remove_from_hass(self):
        """Clean up PI timers. Call in async_will_remove_from_hass."""
        if self._pi_timer_unsub:
            self._pi_timer_unsub()
            self._pi_timer_unsub = None
        if self._sensor_recovery_unsub:
            self._sensor_recovery_unsub()
            self._sensor_recovery_unsub = None

    # ── PI Public API (for vendor subclasses) ─────────────────────────

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
        if hasattr(self, "_config_entry_id"):
            async_dispatcher_send(
                self.hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._config_entry_id),
            )

    async def async_resume_ff_learning(self):
        """Resume FF learning after manual suppression (service call handler)."""
        self._manual_ff_suppress = False
        self._manual_ff_suppress_reason = ""
        _LOGGER.info("FF learning manual suppress cleared")
        if hasattr(self, "_config_entry_id"):
            async_dispatcher_send(
                self.hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._config_entry_id),
            )

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
            state = self.hass.states.get(entity_id)
            if not state or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                continue

            # Try to interpret as numeric first
            try:
                value = float(state.state)
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False

            if is_numeric:
                # Numeric entity: active when non-zero, bias = value × gain
                if value != 0:
                    if d_input.get("suppress_learning"):
                        suppress = True
                        active_suppressors.append(entity_id)
                    total_bias += value * d_input.get("gain", 1.0)
            else:
                # Boolean entity: active when "on", bias = default_bias
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
    def _async_disturbance_entity_changed(self, event):
        """Handle disturbance input entity state changes — update binary sensor."""
        if hasattr(self, "_config_entry_id"):
            async_dispatcher_send(
                self.hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._config_entry_id),
            )

    async def _pi_async_sensor_changed(self, was_none=False):
        """Handle temp sensor update. Call from _async_sensor_changed override.

        Args:
            was_none: True if sensor was previously unavailable (recovery case).
        """
        if not self._pi_enabled:
            return
        if was_none:
            # Sensor recovery — cancel pending recovery callback
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

        # Event-driven tick: run PI on every sensor update, respecting min cooldown
        elapsed = time.monotonic() - self._pi_last_tick_time
        min_cooldown = max(60.0, self._pi_min_interval / 3.0)  # At least 60s, at most interval/3
        if elapsed >= min_cooldown:
            await self._pi_tick()

    async def _check_sensor_recovery(self, _now=None):
        """Called 60s after sensor went unavailable. Fall back to FF-only if still gone."""
        self._sensor_recovery_pending = False
        self._sensor_recovery_unsub = None
        if self._attr_current_temperature is not None:
            _LOGGER.info("PI: temp sensor recovered during grace period")
            await self._pi_tick()
            return
        # Confirmed unavailable — set flag, fall back to feedforward-only setpoint
        self._sensor_unavailable = True
        _LOGGER.warning("PI: temp sensor confirmed unavailable, using feedforward-only fallback")
        if self._desired_temp is None:
            return
        desired_c = TemperatureConverter.convert(
            self._desired_temp, self.temperature_unit, UnitOfTemperature.CELSIUS,
        )
        is_heating = self._attr_hvac_mode == HVACMode.HEAT
        if not is_heating and self._attr_hvac_mode not in (HVACMode.COOL, HVACMode.DRY):
            return
        ff_offset = 0.0
        if self._outdoor_temp is not None:
            bucket_key = round(self._outdoor_temp / 3) * 3
            buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets
            ff_offset = buckets.get(bucket_key, 0.0)
        self._ff_offset = ff_offset
        self._pi_integral = 0.0
        new_setpoint = round(max(self._min_temp, min(self._max_temp, desired_c + ff_offset)))
        if new_setpoint != self._hp_setpoint:
            _LOGGER.info("PI fallback: setpoint %s -> %s (FF only)", self._hp_setpoint, new_setpoint)
            self._hp_setpoint = new_setpoint
            self._pi_command_pending = True
            await self.send_ir()
        self.async_schedule_update_ha_state()

    async def _pi_tick(self, now=None):
        """PI + feedforward controller tick. Called by timer and sensor events."""
        if not self._pi_enabled:
            return
        if self._attr_hvac_mode == HVACMode.OFF:
            self._pi_integral = 0.0
            return
        if self._desired_temp is None:
            return
        if self._attr_current_temperature is None:
            if self._sensor_unavailable or self._sensor_recovery_pending:
                return
            _LOGGER.info("PI: temp sensor unavailable, scheduling 60s recovery check")
            self._sensor_recovery_pending = True
            self._sensor_recovery_unsub = async_call_later(
                self.hass, 60, self._check_sensor_recovery
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
            dt_seconds = float(self._pi_min_interval)  # First tick: assume one interval
        self._pi_last_tick_time = now_mono
        dt_factor = dt_seconds / float(self._pi_min_interval)  # Normalize to reference interval

        # Convert both to °C for PI math
        current_c = TemperatureConverter.convert(
            self._attr_current_temperature,
            self.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )
        desired_c = TemperatureConverter.convert(
            self._desired_temp,
            self.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )
        error = desired_c - current_c

        # PI only operates in explicit HEAT, COOL, or DRY modes
        is_heating = self._attr_hvac_mode == HVACMode.HEAT
        is_cooling = self._attr_hvac_mode in (HVACMode.COOL, HVACMode.DRY)
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

        # Reset integral on large disturbance bias transitions
        if abs(disturbance_bias - self._last_disturbance_bias) > 1.0:
            _LOGGER.info(
                "PI: Disturbance bias changed by %.1f°C, resetting integral",
                disturbance_bias - self._last_disturbance_bias,
            )
            self._pi_integral = 0.0
        self._last_disturbance_bias = disturbance_bias

        # Adaptive setpoint weight: full P (b=1) for large errors,
        # blend to configured weight as error approaches deadband
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
            if self._ff_settled_ticks >= 2 and self._outdoor_temp is not None and not learning_suppressed:
                observed_offset = self._hp_setpoint - desired_c
                bucket_key = round(self._outdoor_temp / 3) * 3
                learn_buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets
                old = learn_buckets.get(bucket_key, 0.0)
                learn_buckets[bucket_key] = 0.8 * old + 0.2 * observed_offset
            elif learning_suppressed and self._ff_settled_ticks >= 2:
                _LOGGER.debug(
                    "PI: FF learning suppressed by %s",
                    active_suppressors if active_suppressors else "manual",
                )
            p_term = 0.0
        else:
            self._ff_settled_ticks = 0
            # 2-DOF setpoint weighting with adaptive blend
            p_error = effective_weight * desired_c - current_c
            p_term = self._pi_kp * p_error
            # Time-normalized integral: trapezoidal (Tustin) method
            # Accumulates (avg of current + previous error) * dt_factor
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

        i_term = self._pi_ki * self._pi_integral
        raw_setpoint = desired_c + p_term + i_term + self._ff_offset
        clamped_setpoint = max(self._min_temp, min(self._max_temp, raw_setpoint))

        # Back-calculation anti-windup
        if self._pi_ki != 0:
            saturation_error = clamped_setpoint - raw_setpoint
            if abs(saturation_error) > 0.01:
                kb = 1.0 / self._pi_ki
                self._pi_integral += kb * saturation_error
                self._pi_integral = max(-50.0, min(50.0, self._pi_integral))

        # Midpoint-crossing hysteresis: only change HP setpoint when the raw
        # value crosses the midpoint between integers. Prevents 1°C limit cycles.
        new_setpoint = self._hp_setpoint  # Default: keep current
        if clamped_setpoint > self._hp_setpoint + 0.5:
            new_setpoint = round(clamped_setpoint)
        elif clamped_setpoint < self._hp_setpoint - 0.5:
            new_setpoint = round(clamped_setpoint)
        new_setpoint = int(max(self._min_temp, min(self._max_temp, new_setpoint)))

        if new_setpoint != self._hp_setpoint:
            _LOGGER.info(
                "PI: error=%.1f P=%.1f I=%.1f FF=%.1f raw=%.1f setpoint %s -> %s",
                error, p_term, i_term, self._ff_offset, clamped_setpoint,
                self._hp_setpoint, new_setpoint,
            )
            self._hp_setpoint = new_setpoint
            self._pi_command_pending = True
            await self.send_ir()
        else:
            _LOGGER.debug(
                "PI: error=%.1f raw=%.1f setpoint=%s (held)",
                error, clamped_setpoint, self._hp_setpoint,
            )

        self.async_schedule_update_ha_state()

    # ── Method overrides (MRO: Mixin → TasmotaIrhvac → ClimateEntity) ──

    async def _async_sensor_changed(self, entity_id_or_event, old_state=None, new_state=None):
        """Override to add PI event-driven ticking on sensor updates."""
        was_none = self._attr_current_temperature is None
        await super()._async_sensor_changed(entity_id_or_event, old_state, new_state)
        if self._attr_current_temperature is not None:
            await self._pi_async_sensor_changed(was_none=was_none)

    def _get_ir_temp(self):
        """Override to return PI-computed setpoint when active."""
        if self._pi_enabled and self._attr_hvac_mode != HVACMode.OFF:
            return round(self._hp_setpoint)
        return super()._get_ir_temp()

    def async_write_ha_state(self):
        """Override to fire dispatcher signal for companion PI sensors."""
        super().async_write_ha_state()
        if self._pi_enabled and hasattr(self, "_config_entry_id"):
            async_dispatcher_send(
                self.hass,
                SIGNAL_PI_UPDATE.format(self._config_entry_id),
            )

    @property
    def hvac_modes(self):
        """Override to filter auto/heat_cool when PI is enabled."""
        modes = self._attr_hvac_modes
        if self._pi_enabled and modes:
            return [m for m in modes if m not in (HVACMode.AUTO, HVACMode.HEAT_COOL)]
        return modes

    async def async_set_hvac_mode(self, hvac_mode):
        """Override to reject auto/heat_cool when PI is enabled."""
        if self._pi_enabled and hvac_mode in (HVACMode.AUTO, HVACMode.HEAT_COOL):
            _LOGGER.warning(
                "PI mode does not support %s — use HEAT or COOL explicitly", hvac_mode,
            )
            return
        await super().async_set_hvac_mode(hvac_mode)

    async def async_set_temperature(self, **kwargs):
        """Override to route through PI when active."""
        temperature = kwargs.get("temperature")
        if temperature is None:
            return
        hvac_mode = kwargs.get("hvac_mode")
        if hvac_mode is not None:
            await super().async_set_hvac_mode(hvac_mode)
        if self._pi_enabled:
            self._desired_temp = temperature
            self._attr_target_temperature = temperature
            self._pi_integral = 0.0
            if self._attr_hvac_mode != HVACMode.OFF:
                self.power_mode = STATE_ON
            await self._pi_tick()
            self.async_schedule_update_ha_state()
            return
        await super().async_set_temperature(**kwargs)

    @property
    def extra_state_attributes(self):
        """Override to include PI controller state attributes."""
        attrs = super().extra_state_attributes
        if self._pi_enabled:
            attrs.update({
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
            })
        return attrs

    async def _handle_state_payload(self, json_payload, payload):
        """Override to add PI MQTT hooks."""
        await super()._handle_state_payload(json_payload, payload)
        # Restore desired_temp over payload temp
        if self._pi_enabled and self._desired_temp is not None:
            if "Temp" in payload and payload["Temp"] > 0:
                self._hp_setpoint = payload["Temp"]
            self._attr_target_temperature = self._desired_temp
        # Handle temp echo detection
        if self._pi_enabled and "Temp" in payload and payload["Temp"] > 0:
            if self._pi_command_pending:
                self._pi_command_pending = False
            else:
                self._desired_temp = self._attr_target_temperature
                await self._pi_tick()

    async def async_reset_ff_buckets(self):
        """Reset feedforward buckets to seed values from config."""
        self._ff_heat_buckets = _seed_buckets(self._ff_heat_reference, self._ff_heat_slope)
        self._ff_cool_buckets = _seed_buckets(
            self._ff_cool_reference, self._ff_cool_slope, is_cooling=True
        )
        self._pi_integral = 0.0
        _LOGGER.info("FF buckets reset to seed values, integral zeroed")
        self.async_schedule_update_ha_state()
