"""Vendor handler for Fujitsu heat pumps.

Handles:
- Preset modes (Powerful, Economy, Min Heat) via raw IR codes
- 56-bit special command detection from physical remotes
- Model 3 state save/restore (56-bit commands carry garbage HVAC fields)
- Powerful mode 20-minute auto-timeout
- Vane button swing state tracking (SET_V / SET_H)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from homeassistant.components.climate.const import (
    HVACMode,
    PRESET_BOOST,
    PRESET_ECO,
    PRESET_NONE,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)

from ..const import PRESET_MIN_HEAT
from . import register
from .base import (
    EntityState,
    IRDecode,
    ModelInfo,
    PresetResult,
    TimerRequest,
    VendorCapabilities,
    VendorHandler,
)

_LOGGER = logging.getLogger(__name__)

# ── Raw IR codes (sent via cmnd/{device}/irsend) ────────────────────

FUJITSU_RAW_PREFIX = "raw,0,3324,1574,448,390,1182,"

FUJITSU_IR_STOP = (
    FUJITSU_RAW_PREFIX
    + "001010001100011000000000000010000000100001000000101111111"
)
FUJITSU_IR_POWERFUL = (
    FUJITSU_RAW_PREFIX
    + "00101000110001100000000000001000000010001001110001100011"
)
FUJITSU_IR_ECONO = (
    FUJITSU_RAW_PREFIX
    + "00101000110001100000000000001000000010001001000001101111"
)
FUJITSU_IR_MIN_HEAT = (
    FUJITSU_RAW_PREFIX
    + "00101000110001100000000000001000000010000111111110010000"
    + "000011001000110011010000010000000000000000000000000000000000010001001110"
)

# ── Data field hex values for detecting received IR presets ──────────

FUJITSU_DATA_POWERFUL = "0x146300101039C6"
FUJITSU_DATA_ECONO = "0x146300101009F6"
FUJITSU_DATA_SET_V = "0x14630010106C93"
FUJITSU_DATA_SET_H = "0x14630010107986"
FUJITSU_DATA_MIN_HEAT = "0x1463001010FE0930800B000000002025"

POWERFUL_TIMEOUT_SECONDS = 1200  # 20 minutes
FUJITSU_MODEL_3 = 3

# Min Heat forces this temperature (Celsius).  The entity converts to
# its display unit when applying state_restore.
MIN_HEAT_TEMP_C = 10.0


@register("FUJITSU")
class FujitsuHandler(VendorHandler):
    """Fujitsu preset modes, 56-bit detection, and state management."""

    vendor_id = "FUJITSU"

    def __init__(self) -> None:
        # Preset flags
        self._min_heat = False
        self._economy = False
        self._powerful = False

        # State snapshots
        self._saved_entity_state: EntityState | None = None
        self._saved_target_temp: float | None = None

        # Post-hook state the entity reads
        self._active_preset: str | None = None
        self._should_reset_integral = False
        self._clear_toggles = False
        self._state_restore: EntityState | None = None

    # ── Capabilities ─────────────────────────────────────────────────

    @classmethod
    def capabilities(cls) -> VendorCapabilities:
        return VendorCapabilities(
            supported_toggles=frozenset(
                {"beep", "turbo", "quiet", "econo", "light"}
            ),
            extra_preset_modes=(
                PRESET_BOOST,
                PRESET_ECO,
                PRESET_MIN_HEAT,
            ),
            has_raw_ir=True,
            celsius_default=True,
            models=(
                ModelInfo(id="-1", label="Auto-detect"),
                ModelInfo(id="1", label="ARRAH2E (Model 1)"),
                ModelInfo(id="3", label="ARDB1 (Model 3)"),
            ),
        )

    # ── Restore ──────────────────────────────────────────────────────

    def on_restore_state(self, restored_preset: str | None) -> None:
        if restored_preset == PRESET_MIN_HEAT:
            self._min_heat = True
        elif restored_preset == PRESET_ECO:
            self._economy = True
        elif restored_preset == PRESET_BOOST:
            self._powerful = True

    # ── State processing ─────────────────────────────────────────────

    def _reset_post_hooks(self) -> None:
        """Clear transient post-hook state before each processing cycle."""
        self._active_preset = None
        self._should_reset_integral = False
        self._clear_toggles = False
        self._state_restore = None

    def pre_state_processing(
        self, decode: IRDecode, entity_state: EntityState
    ) -> None:
        self._reset_post_hooks()

        # Model 3 puts garbage values in HVAC fields for 56-bit commands.
        # Save the current entity state so we can restore it afterward.
        model = decode.irhvac.get("Model")
        if model == FUJITSU_MODEL_3:
            self._saved_entity_state = entity_state

    def post_state_processing(self, decode: IRDecode) -> None:
        power = decode.irhvac.get("Power", "off")
        turbo = decode.irhvac.get("Turbo", "off")
        econo = decode.irhvac.get("Econo", "off")
        clean = decode.irhvac.get("Clean", "off")

        # Power off clears all presets
        if power.lower() == "off":
            self._min_heat = False
            self._powerful = False
            self._economy = False
            return  # should_pause_controller → False (resumes PI)

        # Map hardware flags to presets
        if turbo.lower() == "on":
            self._powerful = True
            self._active_preset = PRESET_BOOST
        if econo.lower() == "on":
            self._economy = True
            self._active_preset = PRESET_ECO
        if clean.lower() == "on":
            self._min_heat = True
            self._active_preset = PRESET_MIN_HEAT

        # 56-bit / special data detection
        self._detect_special_commands(decode)

    def _detect_special_commands(self, decode: IRDecode) -> None:
        """Detect 56-bit preset commands and extended Min Heat data."""
        if self._saved_entity_state is None or decode.data is None:
            return

        # Restore state saved in pre_state_processing (56-bit payloads
        # carry garbage HVAC field values that base extraction applied).
        self._state_restore = self._saved_entity_state

        if decode.bits == 56:
            if decode.data == FUJITSU_DATA_POWERFUL:
                self._powerful = True
                self._active_preset = PRESET_BOOST
            elif decode.data == FUJITSU_DATA_ECONO:
                self._economy = True
                self._active_preset = PRESET_ECO
            elif decode.data == FUJITSU_DATA_SET_V:
                self._apply_vane_set_v()
            elif decode.data == FUJITSU_DATA_SET_H:
                self._apply_vane_set_h()
        elif decode.data == FUJITSU_DATA_MIN_HEAT:
            self._min_heat = True
            self._active_preset = PRESET_MIN_HEAT
            self._should_reset_integral = True
            self._clear_toggles = True
            # Override the restored state with Min Heat specifics.
            # target_temperature is in Celsius — entity converts to display units.
            self._state_restore = EntityState(
                hvac_mode=HVACMode.HEAT,
                target_temperature=MIN_HEAT_TEMP_C,
                fan_mode=self._saved_entity_state.fan_mode,
                swing_mode=self._saved_entity_state.swing_mode,
                swingv=self._saved_entity_state.swingv,
                swingh=self._saved_entity_state.swingh,
                power_mode="on",
            )

        # Consumed — don't re-apply on next cycle
        self._saved_entity_state = None

    def _apply_vane_set_v(self) -> None:
        """Physical remote cycled vertical vane — update swing tracking."""
        if self._state_restore is None:
            return
        swing = self._state_restore.swing_mode
        if swing == SWING_BOTH:
            new_swing = SWING_HORIZONTAL
        elif swing == SWING_VERTICAL:
            new_swing = SWING_OFF
        else:
            new_swing = swing
        self._state_restore = EntityState(
            hvac_mode=self._state_restore.hvac_mode,
            target_temperature=self._state_restore.target_temperature,
            fan_mode=self._state_restore.fan_mode,
            swing_mode=new_swing,
            swingv=None,
            swingh=self._state_restore.swingh,
            power_mode=self._state_restore.power_mode,
        )

    def _apply_vane_set_h(self) -> None:
        """Physical remote cycled horizontal vane — update swing tracking."""
        if self._state_restore is None:
            return
        swing = self._state_restore.swing_mode
        if swing == SWING_BOTH:
            new_swing = SWING_VERTICAL
        elif swing == SWING_HORIZONTAL:
            new_swing = SWING_OFF
        else:
            new_swing = swing
        self._state_restore = EntityState(
            hvac_mode=self._state_restore.hvac_mode,
            target_temperature=self._state_restore.target_temperature,
            fan_mode=self._state_restore.fan_mode,
            swing_mode=new_swing,
            swingv=self._state_restore.swingv,
            swingh=None,
            power_mode=self._state_restore.power_mode,
        )

    # ── Preset handling ──────────────────────────────────────────────

    async def handle_preset(
        self,
        preset_mode: str,
        entity_state: EntityState,
        send_raw_ir: Callable[[str], Awaitable[None]],
    ) -> PresetResult | None:
        self._reset_post_hooks()

        # Exit active presets before entering a new one
        await self._exit_active_presets(preset_mode, send_raw_ir)

        if preset_mode == PRESET_BOOST:
            return await self._enter_powerful(send_raw_ir)

        if preset_mode == PRESET_ECO:
            return await self._enter_econo(send_raw_ir)

        if preset_mode == PRESET_MIN_HEAT:
            return await self._enter_min_heat(entity_state, send_raw_ir)

        if preset_mode == PRESET_NONE:
            self._clear_all_presets()
            return None  # fall through to base for AWAY→NONE and send_ir

        # PRESET_AWAY and IR action presets: fall through to base
        return None

    async def _exit_active_presets(
        self,
        new_preset: str,
        send_raw_ir: Callable[[str], Awaitable[None]],
    ) -> None:
        """Exit currently active Fujitsu presets before entering a new one."""
        if new_preset != PRESET_MIN_HEAT and self._min_heat:
            await send_raw_ir(FUJITSU_IR_STOP)
            await asyncio.sleep(1)  # HP needs time to process STOP
            self._min_heat = False

        if new_preset != PRESET_ECO and self._economy:
            await send_raw_ir(FUJITSU_IR_ECONO)  # Toggle off
            self._economy = False

    async def _enter_powerful(
        self,
        send_raw_ir: Callable[[str], Awaitable[None]],
    ) -> PresetResult:
        if not self._powerful:
            await send_raw_ir(FUJITSU_IR_POWERFUL)
            self._powerful = True
        self._active_preset = PRESET_BOOST
        return PresetResult(
            timer_request=TimerRequest(
                delay_seconds=POWERFUL_TIMEOUT_SECONDS,
                callback_id="clear_powerful",
            ),
        )

    async def _enter_econo(
        self,
        send_raw_ir: Callable[[str], Awaitable[None]],
    ) -> PresetResult:
        if not self._economy:
            await send_raw_ir(FUJITSU_IR_ECONO)
            self._economy = True
        self._active_preset = PRESET_ECO
        return PresetResult()

    async def _enter_min_heat(
        self,
        entity_state: EntityState,
        send_raw_ir: Callable[[str], Awaitable[None]],
    ) -> PresetResult:
        if not self._min_heat:
            self._saved_target_temp = entity_state.target_temperature
            await send_raw_ir(FUJITSU_IR_MIN_HEAT)
            self._min_heat = True
            self._should_reset_integral = True
            self._clear_toggles = True
            self._state_restore = EntityState(
                hvac_mode=HVACMode.HEAT,
                target_temperature=MIN_HEAT_TEMP_C,
                power_mode="on",
            )
        self._active_preset = PRESET_MIN_HEAT
        return PresetResult()

    def _clear_all_presets(self) -> None:
        """Clear all Fujitsu preset flags (for PRESET_NONE)."""
        self._economy = False
        self._min_heat = False
        self._powerful = False
        self._clear_toggles = True

    # ── Timer callback ───────────────────────────────────────────────

    def on_timer(self, callback_id: str) -> None:
        if callback_id == "clear_powerful":
            self._powerful = False
            if self._active_preset == PRESET_BOOST:
                self._active_preset = PRESET_NONE

    # ── Properties the entity reads ──────────────────────────────────

    @property
    def active_preset(self) -> str | None:
        return self._active_preset

    @property
    def should_pause_controller(self) -> bool:
        return self._min_heat or self._economy or self._powerful

    @property
    def should_reset_integral(self) -> bool:
        return self._should_reset_integral

    @property
    def clear_toggles(self) -> bool:
        return self._clear_toggles

    @property
    def state_restore(self) -> EntityState | None:
        return self._state_restore

    @property
    def saved_target_temp(self) -> float | None:
        """Target temp saved before Min Heat, for restoration on exit."""
        return self._saved_target_temp
