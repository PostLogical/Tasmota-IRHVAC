"""Fujitsu-specific IRHVAC with preset modes and PI controller integration."""

import asyncio
import logging

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
from homeassistant.const import ATTR_TEMPERATURE, STATE_ON
from homeassistant.core import callback
from homeassistant.helpers.event import async_call_later

from .climate import TasmotaIrhvac
from .const import (
    PRESET_ECONO,
    PRESET_MIN_HEAT,
    PRESET_POWERFUL,
    PRESET_SET_H,
    PRESET_SET_V,
)
from .pi_controller import PIControllerMixin

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


class FujitsuTasmotaIrhvac(PIControllerMixin, TasmotaIrhvac):
    """Fujitsu IRHVAC with PI + feedforward temperature control and preset modes."""

    def __init__(self, hass, vendor, config):
        super().__init__(hass, vendor, config)

        # Initialize PI controller (vendor-agnostic)
        self.pi_init(config)

        # Fujitsu preset state
        self._min_heat = False
        self._economy = False
        self._powerful = False
        self._powerful_timer_unsub = None
        self._saved_state_for_preset = {}

    async def async_added_to_hass(self):
        await super().async_added_to_hass()

        # Restore Fujitsu preset flags from previous session
        old_state = await self.async_get_last_state()
        if old_state is not None:
            preset = old_state.attributes.get(ATTR_PRESET_MODE)
            if preset == PRESET_MIN_HEAT:
                self._min_heat = True
                self.pi_pause()
            elif preset == PRESET_ECONO:
                self._economy = True
                self.pi_pause()
            elif preset == PRESET_POWERFUL:
                self._powerful = True
                self.pi_pause()

        # Initialize PI controller (sets up timers, restores state, etc.)
        await self.pi_async_added_to_hass(old_state=old_state)

    async def async_will_remove_from_hass(self):
        self.pi_async_will_remove_from_hass()
        if self._powerful_timer_unsub:
            self._powerful_timer_unsub()
            self._powerful_timer_unsub = None
        await super().async_will_remove_from_hass()

    # ── PI Integration Overrides ──────────────────────────────────────

    async def _async_sensor_changed(self, entity_id_or_event, old_state=None, new_state=None):
        """Override to trigger PI tick when temp sensor first becomes available."""
        was_none = self._attr_current_temperature is None
        await super()._async_sensor_changed(entity_id_or_event, old_state, new_state)
        if was_none and self._attr_current_temperature is not None:
            await self._pi_async_sensor_changed()

    def _get_ir_temp(self):
        """Return PI-computed HP setpoint instead of user target temp."""
        pi_temp = self.pi_get_ir_temp()
        if pi_temp is not None:
            return pi_temp
        return super()._get_ir_temp()

    def async_write_ha_state(self):
        """Write state and notify companion PI sensors."""
        super().async_write_ha_state()
        self.pi_write_ha_state()

    @property
    def hvac_modes(self):
        """Filter out auto/heat_cool when PI is enabled."""
        return self.pi_filter_hvac_modes(self._attr_hvac_modes)

    async def async_set_hvac_mode(self, hvac_mode):
        """Reject auto/heat_cool when PI is enabled."""
        if self.pi_reject_hvac_mode(hvac_mode):
            return
        await super().async_set_hvac_mode(hvac_mode)

    async def async_set_temperature(self, **kwargs):
        """Set new desired room temperature. PI computes HP setpoint."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        if temperature is None:
            return

        hvac_mode = kwargs.get("hvac_mode")
        if hvac_mode is not None:
            await self.set_mode(hvac_mode)

        if not await self.pi_set_temperature(temperature):
            await super().async_set_temperature(**kwargs)

    @property
    def extra_state_attributes(self):
        """Return state attributes including PI controller state."""
        attrs = super().extra_state_attributes
        attrs.update(self.pi_extra_state_attributes())
        return attrs

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

        # PI: restore desired_temp over payload temp
        self.pi_restore_desired_temp(payload)

        # Map turbo/econo/clean flags to Fujitsu presets
        if self.power_mode == "off":
            self._min_heat = False
            self._powerful = False
            self._economy = False
            self.pi_resume()
        if self._turbo == "on":
            self._attr_preset_mode = PRESET_POWERFUL
            self._powerful = True
            self.pi_pause()
        if self._econo == "on":
            self._attr_preset_mode = PRESET_ECONO
            self._economy = True
            self.pi_pause()
        if self._clean == "on":
            self._attr_preset_mode = PRESET_MIN_HEAT
            self._min_heat = True
            self.pi_pause()

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
                    self.pi_pause()
                elif data == FUJITSU_DATA_ECONO:
                    self._economy = True
                    self._attr_preset_mode = PRESET_ECONO
                    self.pi_pause()
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
                self._attr_target_temperature = 10  # 10°C / 50°F
                self.pi_pause()
                self.pi_reset_integral()
                self._econo = "off"
                self._turbo = "off"
                self._clean = "off"

        # PI: handle temp from MQTT payload
        if self._pi_enabled and "Temp" in payload and payload["Temp"] > 0:
            is_echo = self._pi_command_pending
            if self.pi_handle_mqtt_temp(is_echo):
                if not (prev_model3 and "Data" in json_payload):
                    await self._pi_tick()

        self.async_schedule_update_ha_state()

    # ── Preset Mode Handling ───────────────────────────────────────────

    async def async_set_preset_mode(self, preset_mode):
        """Set Fujitsu preset mode via raw IR codes."""
        # Exit active presets first
        if preset_mode != PRESET_MIN_HEAT and self._min_heat:
            await self._send_raw_ir(FUJITSU_IR_STOP)
            await asyncio.sleep(1)  # let HP process STOP before next command
            if hasattr(self, "_saved_target_temp") and self._saved_target_temp:
                self._attr_target_temperature = self._saved_target_temp
            self._min_heat = False
            self.pi_resume()

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
                self.pi_pause()
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
                self.pi_pause()
            self.async_schedule_update_ha_state()
            return

        elif preset_mode == PRESET_MIN_HEAT:
            if not self._min_heat:
                self._saved_target_temp = self._attr_target_temperature
                await self._send_raw_ir(FUJITSU_IR_MIN_HEAT)
                self.power_mode = "on"
                self._min_heat = True
                self._attr_hvac_mode = HVACMode.HEAT
                self._attr_target_temperature = 10  # 10°C / 50°F
                self.pi_pause()
                self.pi_reset_integral()
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
            self._attr_preset_mode = self._attr_preset_mode
            self.async_schedule_update_ha_state()
            return

        elif preset_mode == PRESET_SET_H:
            await self._send_raw_ir(FUJITSU_IR_SET_H)
            self._swingh = None
            if self._attr_swing_mode == SWING_BOTH:
                self._attr_swing_mode = SWING_VERTICAL
            elif self._attr_swing_mode == SWING_HORIZONTAL:
                self._attr_swing_mode = SWING_OFF
            self._attr_preset_mode = self._attr_preset_mode
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
            self.pi_resume()
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
        self.pi_resume()
        self.async_schedule_update_ha_state()

    # ── Raw IR Helper ──────────────────────────────────────────────────

    async def _send_raw_ir(self, raw_code):
        """Send a raw IR code via Tasmota's IRSend command."""
        path = self.topic.split("/")
        irsend_topic = f"cmnd/{path[1]}/irsend"
        if float(self._mqtt_delay) != 0.0:
            await asyncio.sleep(float(self._mqtt_delay))
        await mqtt.async_publish(self.hass, irsend_topic, raw_code)
