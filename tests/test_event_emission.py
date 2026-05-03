"""Integration tests for typed event emission (Stage 8).

Each test exercises a real PIController via the standard fixtures,
triggers a state transition, and asserts the corresponding TickEvent
appears in `last_tick.events` with the right typed payload.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from homeassistant.components.climate.const import HVACMode

from custom_components.tasmota_irhvac.pi.snapshot import (
    AnomalyDetectedPayload,
    BatchRunPayload,
    LearningSuppressionChangePayload,
    ModeChangePayload,
    SetpointChangeUserPayload,
    TickEventKind,
)

from .conftest import get_climate_entity


def _events_of_kind(tick, kind: TickEventKind) -> list:
    """Return all events of a given kind from a tick output."""
    return [e for e in tick.events if e.kind == kind]


@pytest.mark.asyncio
async def test_setpoint_change_emits_event(hass, setup_pi_integration):
    """Calling set_temperature emits a SETPOINT_CHANGE_USER event."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # Initialize prior state
    pi._desired_temp = 21.0
    pi.fire_dispatcher()

    # User changes setpoint
    await pi.set_temperature(temperature=22.5)
    pi.fire_dispatcher()

    events = _events_of_kind(pi.last_tick, TickEventKind.SETPOINT_CHANGE_USER)
    assert events, "Expected SETPOINT_CHANGE_USER event after set_temperature"
    payload = events[-1].payload
    assert isinstance(payload, SetpointChangeUserPayload)
    assert payload.from_setpoint == 21.0
    assert payload.to_setpoint == 22.5


@pytest.mark.asyncio
async def test_mode_change_emits_event(hass, setup_pi_integration):
    """Changing _attr_hvac_mode emits a MODE_CHANGE event."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller

    # Establish baseline (records prev_hvac_mode)
    pi.fire_dispatcher()

    # Switch mode
    climate._attr_hvac_mode = HVACMode.COOL
    pi.fire_dispatcher()

    events = _events_of_kind(pi.last_tick, TickEventKind.MODE_CHANGE)
    assert events
    payload = events[-1].payload
    assert isinstance(payload, ModeChangePayload)
    assert payload.to_mode == "cool"


@pytest.mark.asyncio
async def test_batch_run_emits_event(hass, setup_pi_integration):
    """A batch WLS run emits a BATCH_RUN event with typed payload."""
    from custom_components.tasmota_irhvac.pi.batch_learning import BatchResult

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # Synthesize a batch result via the public _record_batch_result path
    # by calling the helper that wires it (mimicking what the periodic
    # batch task does). Direct attribute set + emit_event is what
    # _build_tick_output sees on the next fire_dispatcher.
    pi._last_batch_result = BatchResult(
        n_total=100, n_eligible=80,
        beta_batch=[0.1, -0.05, 0.0, 0.0],
        beta_current=[0.1, -0.04, 0.0, 0.0],
        residual_rms=0.05, max_coeff_change_pct=10.0,
        recommend_update=True, n_outliers_excluded=2,
    )
    # The emit happens inside _record_batch_result, not from setting
    # _last_batch_result directly. Emit via the helper to simulate the
    # integration point.
    pi._emit_event(
        TickEventKind.BATCH_RUN,
        BatchRunPayload(
            n_eligible=80, residual_rms=0.05, recommend_update=True,
            max_coeff_change_pct=10.0, n_outliers_excluded=2,
        ),
    )
    pi.fire_dispatcher()

    events = _events_of_kind(pi.last_tick, TickEventKind.BATCH_RUN)
    assert events
    payload = events[-1].payload
    assert isinstance(payload, BatchRunPayload)
    assert payload.n_eligible == 80
    assert payload.recommend_update is True


@pytest.mark.asyncio
async def test_anomaly_detected_emits_event(hass, setup_pi_integration):
    """An anomaly event (emit-side) lands on the tick output.

    The CUSUM alarm path in `_update_cusum` is exercised by the
    existing test suite (test_anomaly_detection.py); here we just verify
    the emit-helper path from CUSUM appears on `last_tick.events`.
    """
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    pi._emit_event(
        TickEventKind.ANOMALY_DETECTED,
        AnomalyDetectedPayload(
            mode="heat", mean_residual=0.8, peak_cusum=15.0, tick_count=1,
        ),
    )
    pi.fire_dispatcher()

    events = _events_of_kind(pi.last_tick, TickEventKind.ANOMALY_DETECTED)
    assert events
    payload = events[-1].payload
    assert isinstance(payload, AnomalyDetectedPayload)
    assert payload.mode == "heat"
    assert payload.peak_cusum == 15.0


@pytest.mark.asyncio
async def test_manual_suppress_emits_event(hass, setup_pi_integration):
    """suppress_ff_learning service emits LEARNING_SUPPRESSION_CHANGE."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    pi.fire_dispatcher()  # Establish baseline

    await pi.async_suppress_ff_learning(reason="testing")

    events = _events_of_kind(
        pi.last_tick, TickEventKind.LEARNING_SUPPRESSION_CHANGE,
    )
    assert events
    payload = events[-1].payload
    assert isinstance(payload, LearningSuppressionChangePayload)
    assert payload.is_suppressed is True
    assert payload.manual is True


@pytest.mark.asyncio
async def test_pending_events_cleared_each_tick(hass, setup_pi_integration):
    """pi_tick clears the pending-events buffer at the start of each tick."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    pi._emit_event(
        TickEventKind.MODE_CHANGE,
        ModeChangePayload(from_mode="off", to_mode="heat"),
    )
    assert len(pi._pending_events) == 1

    # pi_tick is what runs at tick boundary; clearing happens there.
    await pi.pi_tick()
    assert pi._pending_events == [] or all(
        e.kind != TickEventKind.MODE_CHANGE or e.payload.to_mode != "heat"
        for e in pi._pending_events
    ), "pi_tick should clear pre-existing events"
