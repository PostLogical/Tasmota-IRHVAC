"""DataUpdateCoordinator for typed PI controller state.

Migrates the integration from a custom dispatcher signal pattern
(`SIGNAL_PI_UPDATE`) to HA's recommended `DataUpdateCoordinator` —
the standard pattern for any integration with multiple entities
sharing data from one source.

Architecture:

- One `TasmotaIRHVACCoordinator` per config entry, paired 1:1 with a
  `PIController`. Stored on `hass.data[DATA_KEY_COORDINATORS][entry_id]`
  alongside the climate entity reference.
- Push-based: no polling. `update_interval=None` disables HA's poll
  loop. The controller calls `coordinator.async_set_updated_data(tick)`
  whenever it needs to notify subscribers — same cadence as the
  legacy `fire_dispatcher()`.
- Sensors and binary_sensors extend `CoordinatorEntity[Coordinator]`
  and read `self.coordinator.data.<typed_field>` — no string-key
  dict access, no manual signal subscription.
- `coordinator.data` is the most recently published `TickOutput`.
  Initialized to `controller.last_tick` (an empty `TickOutput.empty(...)`
  before the first tick), so consumers can rely on `coordinator.data`
  always being non-None.

Push-based usage of `DataUpdateCoordinator` is fully supported by HA —
the same pattern MQTT integrations and WebSocket integrations use.
Polling is an optional convenience built on top, which we opt out of.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .snapshot import TickOutput

if TYPE_CHECKING:
    from .pi_controller import PIController

_LOGGER = logging.getLogger(__name__)


class TasmotaIRHVACCoordinator(DataUpdateCoordinator[TickOutput]):
    """Push-based coordinator that distributes typed `TickOutput` to entities.

    There is no `_async_update_data` override (which `DataUpdateCoordinator`
    would call on its polling schedule) because we set `update_interval=None`
    — updates are pushed by `PIController` via `publish(tick)`. Calling
    `async_request_refresh()` falls back to re-publishing the controller's
    current `last_tick` to keep the API contract intact.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        controller: PIController,
        *,
        name: str,
    ) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=name,
            update_interval=None,  # push-based; no polling
        )
        self._controller = controller
        # Seed `data` with the controller's current snapshot so consumers
        # see a non-None value before the first publish.
        self.data = controller.last_tick

    @property
    def controller(self) -> PIController:
        """Return the paired controller (for service handlers)."""
        return self._controller

    async def _async_update_data(self) -> TickOutput:
        """Fallback for `async_request_refresh()` — return controller's snapshot.

        Called by `DataUpdateCoordinator.async_request_refresh()` when a
        consumer explicitly requests a refresh. Returns the controller's
        current `last_tick` rather than computing fresh, since refreshes
        in this push-based model should reflect "the tick state we
        already published."
        """
        return self._controller.last_tick

    def publish(self, tick: TickOutput) -> None:
        """Push a new tick to subscribers — replaces `fire_dispatcher`.

        Called by `PIController.fire_dispatcher` (or its successor) at
        the end of each tick rebuild. Notifies all `CoordinatorEntity`
        subscribers synchronously, same cadence as the legacy
        `SIGNAL_PI_UPDATE` dispatcher.
        """
        self.async_set_updated_data(tick)
