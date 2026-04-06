"""Base vendor handler types and default implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable


@dataclass(frozen=True)
class IRDecode:
    """Parsed IR message from Tasmota, constructed once by the base entity.

    Standardises access to Tasmota's IRHVAC payload so vendor handlers never
    need to dig into raw json_payload dicts.
    """

    irhvac: dict[str, Any]
    protocol: str | None = None
    bits: int | None = None
    data: str | None = None
    ir_received: bool = False


@dataclass(frozen=True)
class EntityState:
    """Snapshot of entity state passed to handlers for context.

    Handlers may read this but never mutate entity state directly.
    The entity creates this snapshot before calling handler hooks.
    """

    hvac_mode: str | None = None
    target_temperature: float | None = None
    fan_mode: str | None = None
    swing_mode: str | None = None
    swingv: str | None = None
    swingh: str | None = None
    power_mode: str | None = None


@dataclass(frozen=True)
class TimerRequest:
    """Instruction for the entity to schedule a callback."""

    delay_seconds: float
    callback_id: str  # handler method name the entity should call back


@dataclass
class PresetResult:
    """Returned by handle_preset when the handler fully owns the preset."""

    timer_request: TimerRequest | None = None


@dataclass(frozen=True)
class ModelInfo:
    """A known model variant for a vendor."""

    id: str
    label: str


@dataclass(frozen=True)
class VendorCapabilities:
    """Declares what a vendor supports — drives config flow defaults.

    Empty supported_toggles means 'show all' (backward-compatible default).
    """

    supported_toggles: frozenset[str] = frozenset()
    extra_preset_modes: tuple[str, ...] = ()
    has_raw_ir: bool = False
    celsius_default: bool = True
    models: tuple[ModelInfo, ...] | None = None


class VendorHandler:
    """Base vendor handler — default pass-through for all vendors.

    One instance per climate entity, composed (not inherited).
    Holds its own state; entity reads handler properties after calling hooks.
    """

    vendor_id: str = "default"

    @classmethod
    def capabilities(cls) -> VendorCapabilities:
        """Vendor capabilities for config flow and feature gating."""
        return VendorCapabilities()

    # ── Initialisation hooks ─────────────────────────────────────────

    def transform_fan_modes(self, fan_modes: list[str]) -> list[str]:
        """Transform the fan modes list during entity init."""
        return fan_modes

    def on_restore_state(self, restored_preset: str | None) -> None:
        """Called during async_added_to_hass with the previous preset."""

    # ── Fan mode remapping ───────────────────────────────────────────

    def remap_fan_to_ir(self, fan_mode: str) -> str:
        """Remap HA fan mode → IR fan speed for send_ir."""
        return fan_mode

    # ── State processing hooks ───────────────────────────────────────

    def pre_state_processing(
        self, decode: IRDecode, entity_state: EntityState
    ) -> None:
        """Called before base field extraction.

        Handler may save entity_state for later restoration (e.g. 56-bit).
        """

    def post_state_processing(self, decode: IRDecode) -> None:
        """Called after base field extraction.

        Handler reads decode.irhvac for vendor flags and updates its own
        internal state (active_preset, should_pause_controller, etc.).
        The entity reads handler properties afterward.
        """

    # ── Preset handling ──────────────────────────────────────────────

    async def handle_preset(
        self,
        preset_mode: str,
        entity_state: EntityState,
        send_raw_ir: Callable[[str], Awaitable[None]],
    ) -> PresetResult | None:
        """Handle a vendor-specific preset mode.

        Returns PresetResult if the handler fully owns this preset.
        Returns None to fall through to base preset handling (AWAY, NONE, etc.).
        """
        return None

    def on_timer(self, callback_id: str) -> None:
        """Called when a previously-requested timer fires."""

    # ── State the entity reads after hooks ────────────────────────────

    @property
    def active_preset(self) -> str | None:
        """Preset mode override, or None to leave entity's preset unchanged."""
        return None

    @property
    def should_pause_controller(self) -> bool:
        """Whether the PI controller should be paused."""
        return False

    @property
    def should_reset_integral(self) -> bool:
        """Whether the PI controller should reset its integral term."""
        return False

    @property
    def clear_toggles(self) -> bool:
        """Whether econo/turbo/clean should be forced to 'off'."""
        return False

    @property
    def state_restore(self) -> EntityState | None:
        """If set, entity should apply these values (replaces garbage from special commands)."""
        return None
