"""Spec tests for PI tick → publish coupling.

Invariant: a logical PI tick (a single user-visible state change in the
PI controller) results in exactly one ``coordinator.publish(tick)`` call.

Bug under test (2026-05-08 bundle audit, CUSUM): the same alarm event was
seen in 14 different TickOutputs, all from a single sensor update. Root
cause was ``async_write_ha_state`` overridden to call ``fire_dispatcher``,
combined with HA's tendency to schedule multiple state writes during one
control flow. Fix: decouple ``fire_dispatcher`` from
``async_write_ha_state`` and call it explicitly at PI-tick boundaries.

These tests are written FIRST per TDD; they fail against the unfixed
code and pass after the decoupling lands.
"""

from __future__ import annotations

from typing import Any

import pytest

from custom_components.tasmota_irhvac.pi.snapshot import (
    TickEventKind,
    TickOutput,
)

from .conftest import get_climate_entity


def _wrap_publish_counter(coord) -> list[TickOutput]:
    """Wrap coord.publish to record every call. Returns the recording list."""
    recorded: list[TickOutput] = []
    original = coord.publish

    def counting_publish(tick: TickOutput) -> None:
        recorded.append(tick)
        original(tick)

    coord.publish = counting_publish  # type: ignore[method-assign]
    return recorded


@pytest.mark.asyncio
async def test_pi_tick_publishes_exactly_once(hass, setup_pi_integration):
    """A single ``_pi_tick()`` results in exactly one publish."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    pi._desired_temp = 21.0
    # Drain any startup publishes.
    pi.fire_dispatcher()

    recorded = _wrap_publish_counter(coord)
    await pi._pi_tick()

    assert len(recorded) == 1, (
        f"_pi_tick should publish exactly once; got {len(recorded)} "
        f"publish calls"
    )


@pytest.mark.asyncio
async def test_async_write_ha_state_does_not_publish(hass, setup_pi_integration):
    """Calling ``async_write_ha_state`` alone must not publish a PI tick.

    Climate-entity state writes (target_temperature, current_temperature,
    hvac_mode, etc.) are independent of PI tick publication. Only logical
    PI ticks should publish via the coordinator.
    """
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    coord = climate.coordinator
    assert coord is not None

    # Drain any startup publishes.
    climate._controller.fire_dispatcher()

    recorded = _wrap_publish_counter(coord)

    # Direct call — simulates HA's automatic state writes (e.g., after
    # set_temperature, async_set_hvac_mode, etc. internal flows).
    climate.async_write_ha_state()

    assert len(recorded) == 0, (
        f"async_write_ha_state must not publish a PI tick; "
        f"got {len(recorded)} publish calls"
    )


@pytest.mark.asyncio
async def test_pi_set_temperature_publishes_exactly_once(hass, setup_pi_integration):
    """Controller-side ``set_temperature`` (which runs a tick) publishes once."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    pi._desired_temp = 21.0
    pi.fire_dispatcher()  # baseline drain

    recorded = _wrap_publish_counter(coord)
    await pi.set_temperature(temperature=22.5)

    assert len(recorded) == 1, (
        f"pi.set_temperature should publish exactly once "
        f"(integrated _pi_tick); got {len(recorded)}"
    )


@pytest.mark.asyncio
async def test_pi_on_remote_change_publishes_exactly_once(hass, setup_pi_integration):
    """``on_remote_change`` (physical remote detected) publishes once."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    pi._desired_temp = 21.0
    pi.fire_dispatcher()

    recorded = _wrap_publish_counter(coord)
    await pi.on_remote_change(reported_temp_ir_unit=22.0)

    assert len(recorded) == 1, (
        f"pi.on_remote_change should publish exactly once; got {len(recorded)}"
    )


@pytest.mark.asyncio
async def test_suppress_ff_learning_publishes_exactly_once(
    hass, setup_pi_integration,
):
    """The ff-learning suppress service handler publishes exactly once."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    pi.fire_dispatcher()
    recorded = _wrap_publish_counter(coord)

    await pi.async_suppress_ff_learning(reason="test")

    assert len(recorded) == 1, (
        f"async_suppress_ff_learning should publish once; got {len(recorded)}"
    )


@pytest.mark.asyncio
async def test_resume_ff_learning_publishes_exactly_once(
    hass, setup_pi_integration,
):
    """The ff-learning resume service handler publishes exactly once."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    pi.fire_dispatcher()
    recorded = _wrap_publish_counter(coord)

    await pi.async_resume_ff_learning()

    assert len(recorded) == 1, (
        f"async_resume_ff_learning should publish once; got {len(recorded)}"
    )


@pytest.mark.asyncio
async def test_initial_setup_publishes_controller_reload_once(
    hass, setup_pi_integration,
):
    """Initial controller setup publishes exactly one CONTROLLER_RELOAD event.

    Currently startup fans out into multiple publishes via the
    ``async_write_ha_state`` override. Spec: one publish at the end of
    ``async_added_to_hass`` carrying CONTROLLER_RELOAD.
    """
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    # Inspect the most recent published tick — it should carry exactly one
    # CONTROLLER_RELOAD event (the startup publish).
    reloads = [
        e for e in coord.data.events
        if e.kind == TickEventKind.CONTROLLER_RELOAD
    ]
    assert len(reloads) == 1, (
        f"Initial publish should carry exactly one CONTROLLER_RELOAD; "
        f"got {len(reloads)}"
    )


@pytest.mark.asyncio
async def test_event_appears_in_only_one_published_tick(
    hass, setup_pi_integration,
):
    """A single emitted event appears in the events tuple of exactly one publish.

    Regression test for the CUSUM bug — same alarm event re-emitted in 14
    successive TickOutputs. After fix, every event is consumed by exactly
    one ``_build_tick_output()`` call (the first publish following the
    emit).
    """
    from custom_components.tasmota_irhvac.pi.snapshot import (
        SetpointChangeUserPayload,
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    pi._desired_temp = 21.0
    pi.fire_dispatcher()

    # Emit a unique sentinel event then trigger a tick path.
    pi._emit_event(
        TickEventKind.SETPOINT_CHANGE_USER,
        SetpointChangeUserPayload(from_setpoint=21.0, to_setpoint=22.0),
    )

    recorded = _wrap_publish_counter(coord)
    pi.fire_dispatcher()  # First publish should consume the pending event.
    pi.fire_dispatcher()  # Second publish should NOT see it again.
    pi.fire_dispatcher()  # Nor third.

    sentinel_count = 0
    for tick in recorded:
        for evt in tick.events:
            if (
                evt.kind == TickEventKind.SETPOINT_CHANGE_USER
                and isinstance(evt.payload, SetpointChangeUserPayload)
                and evt.payload.to_setpoint == 22.0
            ):
                sentinel_count += 1
    assert sentinel_count == 1, (
        f"Sentinel event must appear in exactly one published tick; "
        f"got {sentinel_count}"
    )
