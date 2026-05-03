"""Unit tests for TasmotaIRHVACCoordinator (Stage 7a)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.tasmota_irhvac.pi.coordinator import TasmotaIRHVACCoordinator
from custom_components.tasmota_irhvac.pi.snapshot import TickOutput


def _stub_controller(zone_label: str = "test") -> MagicMock:
    """Return a minimal controller stub with a `last_tick` property."""
    controller = MagicMock()
    controller.last_tick = TickOutput.empty(zone_label=zone_label)
    return controller


@pytest.mark.asyncio
async def test_coordinator_initial_data_is_controller_last_tick(hass):
    """Coordinator.data starts as controller.last_tick (no None gap)."""
    controller = _stub_controller("living_room")
    coord = TasmotaIRHVACCoordinator(hass, controller, name="test")

    assert coord.data is controller.last_tick
    assert coord.data.zone_label == "living_room"


@pytest.mark.asyncio
async def test_coordinator_publish_pushes_new_tick(hass):
    """publish(tick) replaces coordinator.data with the new snapshot."""
    controller = _stub_controller()
    coord = TasmotaIRHVACCoordinator(hass, controller, name="test")

    new_tick = TickOutput.empty(zone_label="updated")
    coord.publish(new_tick)

    assert coord.data is new_tick
    assert coord.data.zone_label == "updated"


@pytest.mark.asyncio
async def test_coordinator_publish_notifies_listeners(hass):
    """publish triggers async listeners — replaces SIGNAL_PI_UPDATE dispatch."""
    controller = _stub_controller()
    coord = TasmotaIRHVACCoordinator(hass, controller, name="test")

    seen: list[TickOutput] = []

    def listener() -> None:
        seen.append(coord.data)

    coord.async_add_listener(listener)

    new_tick = TickOutput.empty(zone_label="published")
    coord.publish(new_tick)
    await hass.async_block_till_done()

    assert seen
    assert seen[-1].zone_label == "published"


@pytest.mark.asyncio
async def test_coordinator_request_refresh_returns_controller_state(hass):
    """async_request_refresh re-reads from controller.last_tick (push fallback)."""
    controller = _stub_controller("initial")
    coord = TasmotaIRHVACCoordinator(hass, controller, name="test")

    # Mutate controller's last_tick directly (simulating a tick that
    # ran without calling publish — defensive path)
    controller.last_tick = TickOutput.empty(zone_label="mutated")

    await coord.async_request_refresh()
    await hass.async_block_till_done()

    assert coord.data.zone_label == "mutated"


@pytest.mark.asyncio
async def test_coordinator_no_polling(hass):
    """Coordinator runs in push-only mode — update_interval is None."""
    controller = _stub_controller()
    coord = TasmotaIRHVACCoordinator(hass, controller, name="test")

    assert coord.update_interval is None


@pytest.mark.asyncio
async def test_coordinator_exposes_controller(hass):
    """controller property gives service handlers access to the paired PIController."""
    controller = _stub_controller()
    coord = TasmotaIRHVACCoordinator(hass, controller, name="test")

    assert coord.controller is controller
