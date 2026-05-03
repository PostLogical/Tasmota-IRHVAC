"""Multi-zone isolation tests (Stage 11).

User runs 4 zones on Condenser A in production. Verifies that
multiple PIController + TasmotaIRHVACCoordinator instances co-exist
without cross-contamination of state — no shared mutable singletons,
each instance carries its own zone_label, each coordinator publishes
only its own ticks.
"""

from __future__ import annotations

import pytest

from custom_components.tasmota_irhvac.pi.coordinator import (
    TasmotaIRHVACCoordinator,
)
from custom_components.tasmota_irhvac.pi.snapshot import TickOutput


@pytest.mark.asyncio
async def test_two_zones_have_independent_last_tick(hass, setup_pi_integration):
    """Two PIControllers don't share `_last_tick` state."""
    entry_a = await setup_pi_integration({
        "name": "Zone A", "pi_tau_estimate": 60,
    })
    entry_b = await setup_pi_integration({
        "name": "Zone B", "pi_tau_estimate": 60,
    })

    from custom_components.tasmota_irhvac.const import DATA_KEY
    entity_a = hass.data[DATA_KEY][entry_a.entry_id]
    entity_b = hass.data[DATA_KEY][entry_b.entry_id]
    pi_a = entity_a._controller
    pi_b = entity_b._controller

    # Initial state: both empty, both with zone-specific labels
    assert pi_a.last_tick is not pi_b.last_tick
    assert pi_a.last_tick.zone_label != pi_b.last_tick.zone_label

    # Mutate B's state, fire its dispatcher — A must NOT see the change.
    pi_b._desired_temp = 25.0
    pi_b.fire_dispatcher()

    # A's last_tick still reflects A's state (default-empty desired_temp)
    # B's last_tick reflects the mutation
    assert pi_b.last_tick.desired_temp == 25.0
    assert pi_a.last_tick.desired_temp != 25.0


@pytest.mark.asyncio
async def test_two_zones_have_independent_pending_events(hass, setup_pi_integration):
    """Event accumulators are per-controller — emitting on B doesn't leak to A."""
    from custom_components.tasmota_irhvac.pi.snapshot import (
        ModeChangePayload, TickEventKind,
    )

    entry_a = await setup_pi_integration({"name": "Zone A"})
    entry_b = await setup_pi_integration({"name": "Zone B"})

    from custom_components.tasmota_irhvac.const import DATA_KEY
    pi_a = hass.data[DATA_KEY][entry_a.entry_id]._controller
    pi_b = hass.data[DATA_KEY][entry_b.entry_id]._controller

    pi_b._emit_event(
        TickEventKind.MODE_CHANGE,
        ModeChangePayload(from_mode="off", to_mode="heat"),
    )

    assert pi_b._pending_events
    assert pi_a._pending_events == []


@pytest.mark.asyncio
async def test_two_zones_have_independent_coordinators(hass, setup_pi_integration):
    """Each entity gets its own coordinator; subscribing to one doesn't see the other."""
    entry_a = await setup_pi_integration({"name": "Zone A"})
    entry_b = await setup_pi_integration({"name": "Zone B"})

    from custom_components.tasmota_irhvac.const import DATA_KEY
    entity_a = hass.data[DATA_KEY][entry_a.entry_id]
    entity_b = hass.data[DATA_KEY][entry_b.entry_id]

    coord_a = entity_a.coordinator
    coord_b = entity_b.coordinator
    assert coord_a is not coord_b
    assert coord_a is not None and coord_b is not None
    assert isinstance(coord_a, TasmotaIRHVACCoordinator)
    assert isinstance(coord_b, TasmotaIRHVACCoordinator)

    seen_a: list[TickOutput] = []
    seen_b: list[TickOutput] = []
    coord_a.async_add_listener(lambda: seen_a.append(coord_a.data))
    coord_b.async_add_listener(lambda: seen_b.append(coord_b.data))

    # Publish on A only
    new_tick = TickOutput.empty(zone_label="A_only")
    coord_a.publish(new_tick)
    await hass.async_block_till_done()

    assert seen_a, "A's listener should fire"
    assert seen_a[-1].zone_label == "A_only"
    assert seen_b == [], "B's listener should NOT fire when only A publishes"


@pytest.mark.asyncio
async def test_zone_labels_match_entity_ids(hass, setup_pi_integration):
    """Each zone's `last_tick.zone_label` matches its entity's entity_id."""
    entry_a = await setup_pi_integration({"name": "Zone A"})
    entry_b = await setup_pi_integration({"name": "Zone B"})

    from custom_components.tasmota_irhvac.const import DATA_KEY
    entity_a = hass.data[DATA_KEY][entry_a.entry_id]
    entity_b = hass.data[DATA_KEY][entry_b.entry_id]
    pi_a = entity_a._controller
    pi_b = entity_b._controller

    pi_a.fire_dispatcher()
    pi_b.fire_dispatcher()

    assert pi_a.last_tick.zone_label == entity_a.entity_id
    assert pi_b.last_tick.zone_label == entity_b.entity_id
