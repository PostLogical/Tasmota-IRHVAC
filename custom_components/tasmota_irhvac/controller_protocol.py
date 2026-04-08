"""ControllerHook Protocol and NullController.

Defines the interface between the climate entity and an optional controller
(PI, or future alternatives). NullController provides no-op implementations
so the entity never needs `if self._pi:` guards.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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

    # ── Lifecycle ────────────────────────────────────────────────────

    async def async_added_to_hass(self, *, old_state: Any = None) -> None: ...

    def async_will_remove_from_hass(self) -> None: ...

    # ── State processing ─────────────────────────────────────────────

    async def on_remote_change(self, reported_temp: float) -> bool: ...

    async def pi_tick(self, now: Any = None) -> bool: ...

    async def sensor_changed(self, was_none: bool) -> bool: ...

    def fire_dispatcher(self) -> None: ...

    # ── Temperature ──────────────────────────────────────────────────

    async def set_temperature(
        self, temperature: float, hvac_mode: str | None = None
    ) -> bool: ...

    def get_ir_temp(self) -> float: ...

    # ── Mode filtering ───────────────────────────────────────────────

    def filter_hvac_modes(self, modes: list[str]) -> list[str]: ...

    def should_reject_hvac_mode(self, mode: str) -> bool: ...

    # ── Pause / resume ───────────────────────────────────────────────

    def pi_pause(self) -> None: ...

    def pi_resume(self) -> None: ...

    def pi_reset_integral(self) -> None: ...

    # ── State attributes & persistence ───────────────────────────────

    def get_extra_state_attributes(self) -> dict[str, Any]: ...

    def get_extra_stored_data(self) -> Any: ...

    # ── FF learning services ─────────────────────────────────────────

    async def async_reset_ff_seeds(self) -> None: ...

    async def async_suppress_ff_learning(self, reason: str = "") -> None: ...

    async def async_resume_ff_learning(self) -> None: ...


class NullController:
    """No-op controller for entities without PI enabled.

    Every method is a safe default — the entity can call any ControllerHook
    method unconditionally without checking if a controller exists.
    """

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

    # ── Lifecycle ────────────────────────────────────────────────────

    async def async_added_to_hass(self, *, old_state=None):
        pass

    def async_will_remove_from_hass(self):
        pass

    # ── State processing ─────────────────────────────────────────────

    async def on_remote_change(self, reported_temp):
        return False

    async def pi_tick(self, now=None):
        return False

    async def sensor_changed(self, was_none):
        return False

    def fire_dispatcher(self):
        pass

    # ── Temperature ──────────────────────────────────────────────────

    async def set_temperature(self, temperature, hvac_mode=None):
        return False

    def get_ir_temp(self):
        # Should not be called — entity checks is_active before calling
        raise RuntimeError("get_ir_temp called on NullController")

    # ── Mode filtering ───────────────────────────────────────────────

    def filter_hvac_modes(self, modes):
        return modes

    def should_reject_hvac_mode(self, mode):
        return False

    # ── Pause / resume ───────────────────────────────────────────────

    def pi_pause(self):
        pass

    def pi_resume(self):
        pass

    def pi_reset_integral(self):
        pass

    # ── State attributes & persistence ───────────────────────────────

    def get_extra_state_attributes(self):
        return {}

    def get_extra_stored_data(self):
        return None

    # ── FF learning services ─────────────────────────────────────────

    async def async_reset_ff_seeds(self):
        pass

    async def async_suppress_ff_learning(self, reason=""):
        pass

    async def async_resume_ff_learning(self):
        pass
