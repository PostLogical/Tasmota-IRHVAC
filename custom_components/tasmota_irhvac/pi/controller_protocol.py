"""ControllerHook Protocol and NullController.

Defines the interface between the climate entity and an optional controller
(PI, or future alternatives). NullController provides no-op implementations
so the entity never needs `if self._pi:` guards.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from homeassistant.components.climate.const import HVACMode
    from homeassistant.core import State
    from homeassistant.helpers.restore_state import ExtraStoredData

    from .snapshot import TickOutput


@runtime_checkable
class ControllerHook(Protocol):
    """Interface the climate entity uses to interact with a controller.

    PIController implements this. NullController provides pass-through
    defaults for when no controller is configured.
    """

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """Whether a real controller is managing temperature."""
        ...

    @property
    def desired_temp(self) -> float | None: ...

    @desired_temp.setter
    def desired_temp(self, value: float) -> None: ...

    @property
    def is_tick_running(self) -> bool: ...

    @property
    def last_tick(self) -> TickOutput:
        """Most-recent typed tick output. Always non-None."""
        ...

    # ── Lifecycle ────────────────────────────────────────────────────

    async def async_added_to_hass(
        self,
        *,
        old_state: State | None = None,
        pi_autosave: dict[str, Any] | None = None,
    ) -> None: ...

    def async_will_remove_from_hass(self) -> None: ...

    def schedule_batch_analysis(self) -> None: ...

    # ── State processing ─────────────────────────────────────────────

    async def on_remote_change(self, reported_temp: float) -> bool: ...

    async def pi_tick(self, now: datetime | None = None) -> bool: ...

    async def sensor_changed(self, was_none: bool) -> bool: ...

    def fire_dispatcher(self) -> None: ...

    # ── Temperature ──────────────────────────────────────────────────

    async def set_temperature(
        self, temperature: float, hvac_mode: str | None = None
    ) -> bool: ...

    def get_ir_temp(self) -> float: ...

    # ── Mode filtering ───────────────────────────────────────────────

    def filter_hvac_modes(self, modes: list[HVACMode]) -> list[HVACMode]: ...

    def should_reject_hvac_mode(self, mode: str) -> bool: ...

    # ── Pause / resume ───────────────────────────────────────────────

    def pi_pause(self) -> None: ...

    def pi_resume(self) -> None: ...

    def pi_reset_integral(self) -> None: ...

    # ── State attributes & persistence ───────────────────────────────

    def get_extra_state_attributes(self) -> dict[str, Any]: ...

    def get_extra_stored_data(self) -> ExtraStoredData | None: ...

    # ── FF learning services ─────────────────────────────────────────

    async def async_reset_ff_seeds(self, mode: str | None = None) -> None: ...

    async def async_suppress_ff_learning(self, reason: str = "") -> None: ...

    async def async_resume_ff_learning(self) -> None: ...

    async def async_flush_observation_buffer(self, mode: str | None = None) -> None: ...

    async def async_learning_reset(
        self, targets: list[str], mode: str | None = None
    ) -> None: ...

    def get_learning_snapshot(self) -> dict[str, Any]: ...

    def apply_learning_snapshot(self, data: dict[str, Any]) -> None: ...


class NullController:
    """No-op controller for entities without PI enabled.

    Every method is a safe default — the entity can call any ControllerHook
    method unconditionally without checking if a controller exists.
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular dependency.
        from .snapshot import TickOutput
        self._last_tick = TickOutput.empty(zone_label="null")

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return False

    @property
    def desired_temp(self) -> float | None:
        return None

    @desired_temp.setter
    def desired_temp(self, value: float) -> None:
        pass

    @property
    def is_tick_running(self) -> bool:
        return False

    @property
    def last_tick(self) -> TickOutput:
        return self._last_tick

    # ── Lifecycle ────────────────────────────────────────────────────

    async def async_added_to_hass(
        self,
        *,
        old_state: State | None = None,
        pi_autosave: dict[str, Any] | None = None,
    ) -> None:
        pass

    def async_will_remove_from_hass(self) -> None:
        pass

    def schedule_batch_analysis(self) -> None:
        pass

    # ── State processing ─────────────────────────────────────────────

    async def on_remote_change(self, reported_temp: float) -> bool:
        return False

    async def pi_tick(self, now: datetime | None = None) -> bool:
        return False

    async def sensor_changed(self, was_none: bool) -> bool:
        return False

    def fire_dispatcher(self) -> None:
        pass

    # ── Temperature ──────────────────────────────────────────────────

    async def set_temperature(
        self, temperature: float, hvac_mode: str | None = None
    ) -> bool:
        return False

    def get_ir_temp(self) -> float:
        # Should not be called — entity checks is_active before calling
        raise RuntimeError("get_ir_temp called on NullController")

    # ── Mode filtering ───────────────────────────────────────────────

    def filter_hvac_modes(self, modes: list[HVACMode]) -> list[HVACMode]:
        return modes

    def should_reject_hvac_mode(self, mode: str) -> bool:
        return False

    # ── Pause / resume ───────────────────────────────────────────────

    def pi_pause(self) -> None:
        pass

    def pi_resume(self) -> None:
        pass

    def pi_reset_integral(self) -> None:
        pass

    # ── State attributes & persistence ───────────────────────────────

    def get_extra_state_attributes(self) -> dict[str, Any]:
        return {}

    def get_extra_stored_data(self) -> ExtraStoredData | None:
        return None

    # ── FF learning services ─────────────────────────────────────────

    async def async_reset_ff_seeds(self, mode: str | None = None) -> None:
        pass

    async def async_suppress_ff_learning(self, reason: str = "") -> None:
        pass

    async def async_resume_ff_learning(self) -> None:
        pass

    async def async_flush_observation_buffer(self, mode: str | None = None) -> None:
        pass

    async def async_learning_reset(
        self, targets: list[str], mode: str | None = None
    ) -> None:
        pass

    def get_learning_snapshot(self) -> dict[str, Any]:
        return {}

    def apply_learning_snapshot(self, data: dict[str, Any]) -> None:
        pass
