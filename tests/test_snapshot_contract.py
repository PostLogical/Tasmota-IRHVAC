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
- Stage 6: get_diagnostic_dump() reduced to offline_bundle().to_dict()
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
async def test_signal_pi_update_ordering(hass, setup_pi_integration):
    """SIGNAL_PI_UPDATE fires AFTER last_tick is set, not before.

    Sensors subscribe to this signal and must see the new tick state when
    they re-read, not the previous tick's state.

    Stage 2: COMPLETE.
    """
    from custom_components.tasmota_irhvac.const import SIGNAL_PI_UPDATE
    from homeassistant.helpers.dispatcher import async_dispatcher_connect

    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller

    seen_ts: list[float] = []

    def callback() -> None:
        # Signal handler reads last_tick — should reflect the just-built tick
        seen_ts.append(pi.last_tick.ts_mono)

    async_dispatcher_connect(
        hass,
        SIGNAL_PI_UPDATE.format(climate._config_entry_id),
        callback,
    )

    pi.fire_dispatcher()
    await hass.async_block_till_done()

    # Signal handler saw a non-zero ts_mono — meaning last_tick was rebuilt
    # before the signal fired (initial empty tick has ts_mono=0.0).
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


@pytest.mark.xfail(reason="Stage 3 not yet implemented", strict=False)
@pytest.mark.asyncio
async def test_debug_capture_service_toggles_full_p(hass, setup_pi_integration):
    """set_debug_capture(full_p=True) makes full_p_heat/cool appear in diagnostics."""
    from custom_components.tasmota_irhvac.const import DOMAIN
    from custom_components.tasmota_irhvac.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})

    # Initially: no full_p
    diag = await async_get_config_entry_diagnostics(hass, entry)
    pi_diag = diag["pi_controller"]
    assert "full_p_heat" not in pi_diag.get("observation_buffer_heat", {})

    # Call service: full_p=True
    await hass.services.async_call(
        DOMAIN,
        "set_debug_capture",
        {"full_p": True, "entity_id": "climate.test_ac_pi"},
        blocking=True,
    )

    # After service call: full_p_heat and full_p_cool present
    diag = await async_get_config_entry_diagnostics(hass, entry)
    pi_diag = diag["pi_controller"]
    assert "full_p_heat" in pi_diag
    assert "full_p_cool" in pi_diag


# ── Stage 4: get_full_diagnostics() reduced to one-liner ─────────────


@pytest.mark.xfail(reason="Stage 4 not yet implemented", strict=False)
def test_get_full_diagnostics_is_one_liner():
    """After Stage 4, get_full_diagnostics() body is a single return statement."""
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


# ── Stage 6: get_diagnostic_dump() reduced to one-liner ──────────────


@pytest.mark.xfail(reason="Stage 6 not yet implemented", strict=False)
def test_get_diagnostic_dump_is_one_liner():
    """After Stage 6, get_diagnostic_dump() body is a single return statement."""
    import inspect

    from custom_components.tasmota_irhvac.pi.pi_controller import PIController

    src = inspect.getsource(PIController.get_diagnostic_dump)
    body_lines = [
        line.strip()
        for line in src.splitlines()
        if line.strip()
        and not line.strip().startswith('"""')
        and not line.strip().startswith("def ")
    ]
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
    assert "offline_bundle()" in return_lines[0]
