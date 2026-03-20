"""Fujitsu-specific IRHVAC with PI closed-loop temperature control and auto-learned feedforward."""

import asyncio
import logging
from datetime import timedelta

from homeassistant.components import mqtt
from homeassistant.components.climate.const import (
    ATTR_PRESET_MODE,
    HVACMode,
    PRESET_NONE,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)
from homeassistant.const import ATTR_TEMPERATURE, STATE_ON, UnitOfTemperature
from homeassistant.core import callback
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.util.unit_conversion import TemperatureConverter

from .climate import TasmotaIrhvac
from .const import (
    ATTR_DESIRED_TEMP,
    ATTR_FF_COOL_BUCKETS,
    ATTR_FF_HEAT_BUCKETS,
    ATTR_FF_OFFSET,
    ATTR_HP_SETPOINT,
    ATTR_PI_INTEGRAL,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PI_DEADBAND,
    CONF_PI_ENABLED,
    CONF_PI_FF_COOL_REFERENCE,
    CONF_PI_FF_COOL_SLOPE,
    CONF_PI_FF_HEAT_REFERENCE,
    CONF_PI_FF_HEAT_SLOPE,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_MIN_INTERVAL,
    DEFAULT_PI_DEADBAND,
    DEFAULT_PI_ENABLED,
    DEFAULT_PI_FF_COOL_REFERENCE,
    DEFAULT_PI_FF_COOL_SLOPE,
    DEFAULT_PI_FF_HEAT_REFERENCE,
    DEFAULT_PI_FF_HEAT_SLOPE,
    DEFAULT_PI_KI,
    DEFAULT_PI_KP,
    DEFAULT_PI_MIN_INTERVAL,
    PRESET_ECONO,
    PRESET_MIN_HEAT,
    PRESET_POWERFUL,
    PRESET_SET_H,
    PRESET_SET_V,
)

_LOGGER = logging.getLogger(__name__)

# Raw IR codes sent via cmnd/{device}/irsend
FUJITSU_RAW_PREFIX = "raw,0,3324,1574,448,390,1182,"
FUJITSU_IR_STOP = FUJITSU_RAW_PREFIX + "001010001100011000000000000010000000100001000000101111111"
FUJITSU_IR_POWERFUL = FUJITSU_RAW_PREFIX + "00101000110001100000000000001000000010001001110001100011"
FUJITSU_IR_ECONO = FUJITSU_RAW_PREFIX + "00101000110001100000000000001000000010001001000001101111"
FUJITSU_IR_MIN_HEAT = (
    FUJITSU_RAW_PREFIX
    + "00101000110001100000000000001000000010000111111110010000"
    + "000011001000110011010000010000000000000000000000000000000000010001001110"
)
FUJITSU_IR_SET_V = FUJITSU_RAW_PREFIX + "001010001100011000000000000010000000100000110110110010011"
FUJITSU_IR_SET_H = FUJITSU_RAW_PREFIX + "00101000110001100000000000001000000010001001111001100001"

# Data field hex values for detecting received IR presets
FUJITSU_DATA_POWERFUL = "0x146300101039C6"
FUJITSU_DATA_ECONO = "0x146300101009F6"
FUJITSU_DATA_SET_V = "0x14630010106C93"
FUJITSU_DATA_SET_H = "0x14630010107986"
FUJITSU_DATA_MIN_HEAT = "0x1463001010FE0930800B000000002025"

POWERFUL_TIMEOUT_SECONDS = 1200  # 20 minutes
FUJITSU_MODEL_3 = 3


def _seed_buckets(reference, slope, is_cooling=False):
    """Seed feedforward buckets from a linear approximation."""
    buckets = {}
    # Cover outdoor temps from -30°C to 45°C in 3°C steps
    for bucket_temp in range(-30, 48, 3):
        if is_cooling:
            delta = max(0, bucket_temp - reference)
            buckets[bucket_temp] = -slope * delta
        else:
            delta = max(0, reference - bucket_temp)
            buckets[bucket_temp] = slope * delta
    return buckets


class FujitsuTasmotaIrhvac(TasmotaIrhvac):
    """Fujitsu IRHVAC with PI + feedforward temperature control and preset modes."""

    def __init__(self, hass, vendor, config):
        super().__init__(hass, vendor, config)

        # PI controller config
        self._pi_enabled = config.get(CONF_PI_ENABLED, DEFAULT_PI_ENABLED)
        self._pi_kp = config.get(CONF_PI_KP, DEFAULT_PI_KP)
        self._pi_ki = config.get(CONF_PI_KI, DEFAULT_PI_KI)
        self._pi_min_interval = config.get(CONF_PI_MIN_INTERVAL, DEFAULT_PI_MIN_INTERVAL)
        self._pi_deadband = config.get(CONF_PI_DEADBAND, DEFAULT_PI_DEADBAND)

        # Feedforward config
        self._outdoor_temp_sensor = config.get(CONF_OUTDOOR_TEMP_SENSOR)
        self._ff_heat_reference = config.get(CONF_PI_FF_HEAT_REFERENCE, DEFAULT_PI_FF_HEAT_REFERENCE)
        self._ff_heat_slope = config.get(CONF_PI_FF_HEAT_SLOPE, DEFAULT_PI_FF_HEAT_SLOPE)
        self._ff_cool_reference = config.get(CONF_PI_FF_COOL_REFERENCE, DEFAULT_PI_FF_COOL_REFERENCE)
        self._ff_cool_slope = config.get(CONF_PI_FF_COOL_SLOPE, DEFAULT_PI_FF_COOL_SLOPE)

        # PI controller state
        self._desired_temp = self._attr_target_temperature
        self._hp_setpoint = self._attr_target_temperature
        self._pi_integral = 0.0
        self._pi_timer_unsub = None
        self._ff_offset = 0.0
        self._ff_settled_ticks = 0

        # Feedforward buckets (seeded from linear config, refined by auto-learning)
        self._ff_heat_buckets = _seed_buckets(self._ff_heat_reference, self._ff_heat_slope)
        self._ff_cool_buckets = _seed_buckets(self._ff_cool_reference, self._ff_cool_slope, is_cooling=True)

        # Outdoor temp state
        self._outdoor_temp = None

        # Fujitsu preset state
        self._min_heat = False
        self._economy = False
        self._powerful = False
        self._powerful_timer_unsub = None
        self._saved_state_for_preset = {}

    async def async_added_to_hass(self):
        await super().async_added_to_hass()

        # Restore PI and feedforward state from previous session
        old_state = await self.async_get_last_state()
        if old_state is not None:
            attrs = old_state.attributes
            if attrs.get(ATTR_PI_INTEGRAL) is not None:
                self._pi_integral = float(attrs[ATTR_PI_INTEGRAL])
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
            # Restore Fujitsu preset flags
            preset = attrs.get(ATTR_PRESET_MODE)
            if preset == PRESET_MIN_HEAT:
                self._min_heat = True
            elif preset == PRESET_ECONO:
                self._economy = True
            elif preset == PRESET_POWERFUL:
                self._powerful = True

        # Fallback: sync with restored _attr_target_temperature from super()
        # desired_temp stays in the entity's display unit (e.g. °F); PI converts internally
        if self._desired_temp is None and self._attr_target_temperature is not None:
            self._desired_temp = self._attr_target_temperature
        if self._hp_setpoint is None and self._attr_target_temperature is not None:
            self._hp_setpoint = TemperatureConverter.convert(
                self._attr_target_temperature,
                self.temperature_unit,
                UnitOfTemperature.CELSIUS,
            )

        # Register outdoor temp sensor with state change listener
        if self._outdoor_temp_sensor:
            async_track_state_change_event(
                self.hass,
                self._outdoor_temp_sensor,
                self._async_outdoor_temp_changed,
            )
            outdoor_state = self.hass.states.get(self._outdoor_temp_sensor)
            if outdoor_state is not None:
                self._update_outdoor_temp(outdoor_state)

        # Compute initial feedforward offset for display
        if self._outdoor_temp is not None and self._pi_enabled:
            is_heating = self._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
            buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets
            bucket_key = round(self._outdoor_temp / 3) * 3
            self._ff_offset = buckets.get(bucket_key, 0.0)

        # Start PI timer and run first tick immediately
        if self._pi_enabled and self._temp_sensor:
            self._pi_timer_unsub = async_track_time_interval(
                self.hass,
                self._pi_tick,
                timedelta(seconds=self._pi_min_interval),
            )
            await self._pi_tick()

    async def async_will_remove_from_hass(self):
        if self._pi_timer_unsub:
            self._pi_timer_unsub()
            self._pi_timer_unsub = None
        if self._powerful_timer_unsub:
            self._powerful_timer_unsub()
            self._powerful_timer_unsub = None
        await super().async_will_remove_from_hass()

    # ── PI Controller ──────────────────────────────────────────────────

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

    async def _pi_tick(self, now=None):
        """Periodic PI + feedforward controller tick."""
        if not self._pi_enabled:
            return
        if self._attr_hvac_mode == HVACMode.OFF:
            self._pi_integral = 0.0
            return
        if self._attr_current_temperature is None or self._desired_temp is None:
            return
        # Don't send IR while a Fujitsu preset is active (would cancel it)
        if self._min_heat or self._powerful or self._economy:
            return

        # Convert both to °C for PI math (HP operates in °C)
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

        # Select bucket set based on HVAC mode
        is_heating = self._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL)
        buckets = self._ff_heat_buckets if is_heating else self._ff_cool_buckets

        # Feedforward: lookup learned offset for current outdoor temp
        self._ff_offset = 0.0
        if self._outdoor_temp is not None:
            bucket_key = round(self._outdoor_temp / 3) * 3
            self._ff_offset = buckets.get(bucket_key, 0.0)

        # Deadband: if error is small, decay integral and optionally learn
        if abs(error) < self._pi_deadband:
            self._pi_integral *= 0.9
            self._ff_settled_ticks += 1
            # Auto-learn: record offset when settled for 2+ ticks
            if self._ff_settled_ticks >= 2 and self._outdoor_temp is not None:
                observed_offset = self._hp_setpoint - desired_c
                bucket_key = round(self._outdoor_temp / 3) * 3
                old = buckets.get(bucket_key, 0.0)
                buckets[bucket_key] = 0.8 * old + 0.2 * observed_offset
            self.async_schedule_update_ha_state()
            return

        self._ff_settled_ticks = 0

        # PI computation
        p_term = self._pi_kp * error
        self._pi_integral += error

        # Anti-windup: clamp integral so setpoint stays in valid range
        if self._pi_ki != 0:
            max_integral = (self._max_temp - desired_c - p_term - self._ff_offset) / self._pi_ki
            min_integral = (self._min_temp - desired_c - p_term - self._ff_offset) / self._pi_ki
            if min_integral > max_integral:
                min_integral, max_integral = max_integral, min_integral
            self._pi_integral = max(min_integral, min(max_integral, self._pi_integral))

        i_term = self._pi_ki * self._pi_integral
        raw_setpoint = desired_c + p_term + i_term + self._ff_offset
        new_setpoint = round(max(self._min_temp, min(self._max_temp, raw_setpoint)))

        if new_setpoint != self._hp_setpoint:
            _LOGGER.info(
                "PI: error=%.1f P=%.1f I=%.1f FF=%.1f setpoint %s -> %s",
                error, p_term, i_term, self._ff_offset,
                self._hp_setpoint, new_setpoint,
            )
            self._hp_setpoint = new_setpoint
            await self.send_ir()
        else:
            _LOGGER.debug("PI: error=%.1f, setpoint unchanged at %s", error, self._hp_setpoint)

        self.async_schedule_update_ha_state()

    def _get_ir_temp(self):
        """Return PI-computed HP setpoint instead of user target temp."""
        if self._pi_enabled and self._attr_hvac_mode != HVACMode.OFF:
            return round(self._hp_setpoint)
        return super()._get_ir_temp()

    async def async_set_temperature(self, **kwargs):
        """Set new desired room temperature. PI computes HP setpoint."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return

        hvac_mode = kwargs.get("hvac_mode")
        if hvac_mode is not None:
            await self.set_mode(hvac_mode)

        if self._pi_enabled:
            self._desired_temp = temperature
            self._attr_target_temperature = temperature
            if self._attr_hvac_mode != HVACMode.OFF:
                self.power_mode = STATE_ON
            await self._pi_tick()
            self.async_schedule_update_ha_state()
        else:
            await super().async_set_temperature(**kwargs)

    # ── State Payload Handling ─────────────────────────────────────────

    async def _handle_state_payload(self, json_payload, payload):
        """Handle MQTT state with Fujitsu-specific preset detection and PI integration."""
        if payload.get("Vendor") != self._vendor:
            return

        # Save prev state for Model 3 / 56-bit preset restoration
        prev_model3 = False
        if "Model" in payload and payload["Model"] == FUJITSU_MODEL_3:
            prev_model3 = True
            self._saved_state_for_preset = {
                "hvac_mode": self._attr_hvac_mode,
                "target_temp": self._attr_target_temperature,
                "fan_mode": self._attr_fan_mode,
                "swingv": self._swingv,
                "swingh": self._swingh,
            }

        # Standard IRHVAC processing
        await super()._handle_state_payload(json_payload, payload)

        # Map turbo/econo/clean flags to Fujitsu presets
        if self.power_mode == "off":
            self._min_heat = False
            self._powerful = False
            self._economy = False
        if self._turbo == "on":
            self._attr_preset_mode = PRESET_POWERFUL
            self._powerful = True
        if self._econo == "on":
            self._attr_preset_mode = PRESET_ECONO
            self._economy = True
        if self._clean == "on":
            self._attr_preset_mode = PRESET_MIN_HEAT
            self._min_heat = True

        # 56-bit / special data detection
        if "Data" in json_payload and prev_model3:
            data = json_payload["Data"]
            # Restore saved state (56-bit presets put filler values)
            saved = self._saved_state_for_preset
            self._attr_hvac_mode = saved.get("hvac_mode", self._attr_hvac_mode)
            self._attr_target_temperature = saved.get("target_temp", self._attr_target_temperature)
            self._attr_fan_mode = saved.get("fan_mode", self._attr_fan_mode)
            self._swingv = saved.get("swingv", self._swingv)
            self._swingh = saved.get("swingh", self._swingh)

            bits = json_payload.get("Bits")
            if bits == 56:
                if data == FUJITSU_DATA_POWERFUL:
                    self._powerful = True
                    self._attr_preset_mode = PRESET_POWERFUL
                elif data == FUJITSU_DATA_ECONO:
                    self._economy = True
                    self._attr_preset_mode = PRESET_ECONO
                elif data == FUJITSU_DATA_SET_V:
                    self._attr_preset_mode = PRESET_SET_V
                    self._swingv = None
                    if self._attr_swing_mode == SWING_BOTH:
                        self._attr_swing_mode = SWING_HORIZONTAL
                    elif self._attr_swing_mode == SWING_VERTICAL:
                        self._attr_swing_mode = SWING_OFF
                elif data == FUJITSU_DATA_SET_H:
                    self._attr_preset_mode = PRESET_SET_H
                    self._swingh = None
                    if self._attr_swing_mode == SWING_BOTH:
                        self._attr_swing_mode = SWING_VERTICAL
                    elif self._attr_swing_mode == SWING_HORIZONTAL:
                        self._attr_swing_mode = SWING_OFF
            elif data == FUJITSU_DATA_MIN_HEAT:
                self._min_heat = True
                self._attr_preset_mode = PRESET_MIN_HEAT
                self.power_mode = "on"
                self._attr_hvac_mode = HVACMode.HEAT
                self._attr_target_temperature = 50
                self._econo = "off"
                self._turbo = "off"
                self._clean = "off"

        # PI: if temp received from physical remote, treat as new desired room temp
        if self._pi_enabled and "Temp" in payload and payload["Temp"] > 0:
            if not (prev_model3 and "Data" in json_payload):
                self._desired_temp = self._attr_target_temperature
                await self._pi_tick()

        self.async_schedule_update_ha_state()

    # ── Preset Mode Handling ───────────────────────────────────────────

    async def async_set_preset_mode(self, preset_mode):
        """Set Fujitsu preset mode via raw IR codes."""
        # Exit active presets first
        if preset_mode != PRESET_MIN_HEAT and self._min_heat:
            await self._send_raw_ir(FUJITSU_IR_STOP)
            if hasattr(self, "_saved_target_temp") and self._saved_target_temp:
                self._attr_target_temperature = self._saved_target_temp
            self._min_heat = False

        if (
            preset_mode not in (PRESET_ECONO, PRESET_SET_V, PRESET_SET_H)
            and self._economy
        ):
            await self._send_raw_ir(FUJITSU_IR_ECONO)
            self._economy = False

        if preset_mode == PRESET_POWERFUL:
            if not self._powerful:
                await self._send_raw_ir(FUJITSU_IR_POWERFUL)
                self._powerful = True
                self._attr_preset_mode = PRESET_POWERFUL
                # Auto-clear after 20 minutes
                if self._powerful_timer_unsub:
                    self._powerful_timer_unsub()
                self._powerful_timer_unsub = async_call_later(
                    self.hass, POWERFUL_TIMEOUT_SECONDS, self._clear_powerful
                )
            self.async_schedule_update_ha_state()
            return

        elif preset_mode == PRESET_ECONO:
            if not self._economy:
                await self._send_raw_ir(FUJITSU_IR_ECONO)
                self._economy = True
                self._attr_preset_mode = PRESET_ECONO
            self.async_schedule_update_ha_state()
            return

        elif preset_mode == PRESET_MIN_HEAT:
            if not self._min_heat:
                self._saved_target_temp = self._attr_target_temperature
                await self._send_raw_ir(FUJITSU_IR_MIN_HEAT)
                self.power_mode = "on"
                self._min_heat = True
                self._attr_hvac_mode = HVACMode.HEAT
                self._attr_target_temperature = 50
                self._econo = "off"
                self._economy = False
                self._powerful = False
                self._turbo = "off"
                self._clean = "off"
                self._attr_preset_mode = PRESET_MIN_HEAT
            self.async_schedule_update_ha_state()
            return

        elif preset_mode == PRESET_SET_V:
            await self._send_raw_ir(FUJITSU_IR_SET_V)
            self._swingv = None
            if self._attr_swing_mode == SWING_BOTH:
                self._attr_swing_mode = SWING_HORIZONTAL
            elif self._attr_swing_mode == SWING_VERTICAL:
                self._attr_swing_mode = SWING_OFF
            self._attr_preset_mode = self._attr_preset_mode  # maintain current preset
            self.async_schedule_update_ha_state()
            return

        elif preset_mode == PRESET_SET_H:
            await self._send_raw_ir(FUJITSU_IR_SET_H)
            self._swingh = None
            if self._attr_swing_mode == SWING_BOTH:
                self._attr_swing_mode = SWING_VERTICAL
            elif self._attr_swing_mode == SWING_HORIZONTAL:
                self._attr_swing_mode = SWING_OFF
            self._attr_preset_mode = self._attr_preset_mode  # maintain current preset
            self.async_schedule_update_ha_state()
            return

        elif preset_mode == PRESET_NONE:
            self._turbo = "off"
            self._econo = "off"
            self._clean = "off"
            self._economy = False
            self._min_heat = False
            self._powerful = False
            self._attr_preset_mode = PRESET_NONE
            await self.send_ir()
            return

        # PRESET_AWAY and others: delegate to parent
        await super().async_set_preset_mode(preset_mode)

    @callback
    def _clear_powerful(self, _now=None):
        """Auto-clear Powerful preset after timeout."""
        self._powerful = False
        if self._attr_preset_mode == PRESET_POWERFUL:
            self._attr_preset_mode = PRESET_NONE
        self._powerful_timer_unsub = None
        self.async_schedule_update_ha_state()

    # ── Raw IR Helper ──────────────────────────────────────────────────

    async def _send_raw_ir(self, raw_code):
        """Send a raw IR code via Tasmota's IRSend command."""
        path = self.topic.split("/")
        irsend_topic = f"cmnd/{path[1]}/irsend"
        if float(self._mqtt_delay) != 0.0:
            await asyncio.sleep(float(self._mqtt_delay))
        await mqtt.async_publish(self.hass, irsend_topic, raw_code)

    # ── Extra State Attributes ─────────────────────────────────────────

    @property
    def extra_state_attributes(self):
        """Return state attributes including PI controller state."""
        attrs = super().extra_state_attributes
        if self._pi_enabled:
            attrs[ATTR_HP_SETPOINT] = self._hp_setpoint
            attrs[ATTR_PI_INTEGRAL] = round(self._pi_integral, 3)
            attrs[ATTR_DESIRED_TEMP] = self._desired_temp
            attrs[ATTR_FF_OFFSET] = round(self._ff_offset, 2)
            attrs[ATTR_FF_HEAT_BUCKETS] = {
                str(k): round(v, 2) for k, v in self._ff_heat_buckets.items()
            }
            attrs[ATTR_FF_COOL_BUCKETS] = {
                str(k): round(v, 2) for k, v in self._ff_cool_buckets.items()
            }
        return attrs
