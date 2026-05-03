"""Contract tests for the tick-first snapshot architecture.

These tests are written BEFORE the implementation lands and use
`@pytest.mark.xfail` until their target stage is complete. As each stage
ships, the corresponding xfail marker is removed.

Stages (from `~/.claude/plans/why-don-t-you-give-dynamic-chipmunk.md`):

- Stage 1: snapshot.py module exists with TickOutput dataclass
- Stage 2: controller.last_tick populated after tick(); SCHEMA_VERSION=1
- Stage 3: set_debug_capture service toggles full_p_heat/cool fields
- Stage 4: get_full_diagnostics() reduced to last_tick.diagnostics().to_dict()
- Stage 5: sensor-feeding getters read from last_tick
- Stage 6: get_diagnostic_dump path removed entirely (no consumer)
"""

from __future__ import annotations

import pytest


# ── Stage 1: TickOutput dataclass exists ──────────────────────────────


def test_snapshot_module_importable():
    """The snapshot module exists and exports TickOutput + SCHEMA_VERSION.

    Stage 1: COMPLETE.
    """
    from custom_components.tasmota_irhvac.pi.snapshot import TickOutput

    assert TickOutput.SCHEMA_VERSION == 1


def test_tick_output_roundtrip():
    """TickOutput.from_dict(tick.to_dict()) == tick.

    Sets the contract for future event-log readers: any record written
    to the log must round-trip cleanly.

    Stage 1: COMPLETE.
    """
    from custom_components.tasmota_irhvac.pi.snapshot import TickOutput

    tick = TickOutput.empty(zone_label="contract_test")
    serialized = tick.to_dict()
    restored = TickOutput.from_dict(serialized)
    assert restored == tick


# ── Stage 2: last_tick populated after tick() ─────────────────────────


@pytest.mark.asyncio
async def test_last_tick_populated_after_fire_dispatcher(hass, setup_pi_integration):
    """controller.last_tick is non-None after fire_dispatcher; schema version is 1.

    Stage 2: COMPLETE.
    """
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # fire_dispatcher() rebuilds last_tick before sending the signal
    pi.fire_dispatcher()

    assert pi.last_tick is not None
    assert pi.last_tick.SCHEMA_VERSION == 1


@pytest.mark.asyncio
async def test_last_tick_zone_label(hass, setup_pi_integration):
    """last_tick carries zone_label from the climate entity.

    Stage 2: COMPLETE.
    """
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller

    pi.fire_dispatcher()

    # zone_label is the entity_id (or empty if not yet registered)
    assert pi.last_tick.zone_label  # non-empty after entity registration


@pytest.mark.asyncio
async def test_coordinator_publish_ordering(hass, setup_pi_integration):
    """Coordinator listeners see fresh `last_tick` when notified.

    The coordinator's listener fires after `controller._last_tick` is
    rebuilt, so any consumer that reads `pi.last_tick` from a listener
    callback sees current state — not stale state from before the tick.

    Stage 2/7d: COMPLETE. Replaces the legacy SIGNAL_PI_UPDATE ordering
    test now that sensor refreshes flow through DataUpdateCoordinator.
    """
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    coord = climate.coordinator
    assert coord is not None

    seen_ts: list[float] = []

    def listener() -> None:
        # Listener reads coordinator.data — should be the just-published tick
        seen_ts.append(coord.data.ts_mono)

    coord.async_add_listener(listener)

    pi.fire_dispatcher()
    await hass.async_block_till_done()

    # Listener saw a non-zero ts_mono — meaning the tick was rebuilt and
    # published before the listener fired (initial empty tick has
    # ts_mono=0.0).
    assert seen_ts
    assert seen_ts[-1] > 0.0


@pytest.mark.asyncio
async def test_null_controller_last_tick_compat(hass, setup_integration):
    """NullController.last_tick is a default-empty TickOutput (not None).

    Sensors must not crash when reading from a non-PI integration.

    Stage 2: COMPLETE.
    """
    from .conftest import get_climate_entity

    entry = await setup_integration({"pi_enabled": False})
    pi = get_climate_entity(hass, entry)._controller

    assert pi.last_tick is not None
    assert pi.last_tick.SCHEMA_VERSION == 1


# ── Stage 3: set_debug_capture service ────────────────────────────────


@pytest.mark.asyncio
async def test_debug_capture_service_toggles_full_p(hass, setup_pi_integration):
    """set_debug_capture(full_p=True) makes full_p_heat/cool appear in diagnostics.

    Stage 3: COMPLETE.
    """
    from custom_components.tasmota_irhvac.const import DOMAIN
    from custom_components.tasmota_irhvac.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)

    # Initially: no full_p
    diag = await async_get_config_entry_diagnostics(hass, entry)
    pi_diag = diag["pi_controller"]
    assert "full_p_heat" not in pi_diag
    assert "full_p_cool" not in pi_diag

    # Call service: full_p=True
    await hass.services.async_call(
        DOMAIN,
        "set_debug_capture",
        {"full_p": True, "entity_id": climate.entity_id},
        blocking=True,
    )

    # After service call: full_p_heat and full_p_cool present
    diag = await async_get_config_entry_diagnostics(hass, entry)
    pi_diag = diag["pi_controller"]
    assert "full_p_heat" in pi_diag
    assert "full_p_cool" in pi_diag
    # Should be square matrices
    n = len(pi_diag["full_p_heat"])
    assert all(len(row) == n for row in pi_diag["full_p_heat"])

    # Toggle off again
    await hass.services.async_call(
        DOMAIN,
        "set_debug_capture",
        {"full_p": False, "entity_id": climate.entity_id},
        blocking=True,
    )
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert "full_p_heat" not in diag["pi_controller"]
    assert "full_p_cool" not in diag["pi_controller"]


# ── Stage 4: get_full_diagnostics() reduced to one-liner ─────────────


def test_get_full_diagnostics_is_one_liner():
    """After Stage 4, get_full_diagnostics() body is a single return statement.

    Stage 4: COMPLETE.
    """
    import inspect

    from custom_components.tasmota_irhvac.pi.pi_controller import PIController

    src = inspect.getsource(PIController.get_full_diagnostics)
    # Allow docstring + return; strict body should be the return
    body_lines = [
        line.strip()
        for line in src.splitlines()
        if line.strip()
        and not line.strip().startswith('"""')
        and not line.strip().startswith("def ")
    ]
    # Drop docstring lines (anything between triple quotes)
    in_doc = False
    body_no_doc: list[str] = []
    for line in body_lines:
        if '"""' in line:
            in_doc = not in_doc
            continue
        if not in_doc:
            body_no_doc.append(line)

    return_lines = [line for line in body_no_doc if line.startswith("return")]
    assert len(return_lines) == 1
    assert "diagnostics()" in return_lines[0]


# ── Stage 6: diagnostic_dump path removed entirely ─────────────────────


def test_get_diagnostic_dump_removed():
    """After Stage 6, get_diagnostic_dump no longer exists on PIController.

    Removed as redundant with the upcoming export_debug_bundle service
    (Stage 10). The 6 fields unique to the legacy dump are preserved on
    BatchLearningSnapshot (beta_std_err, blend_gains, beta_blended,
    feature_vif, detected_tau, plant_snapshot) so no diagnostic data is
    lost.
    """
    from custom_components.tasmota_irhvac.pi.pi_controller import PIController

    assert not hasattr(PIController, "get_diagnostic_dump")
