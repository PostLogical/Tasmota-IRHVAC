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
    ControllerReloadPayload,
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
async def test_batch_learning_snapshot_carries_lag_tau_diagnostics(
    hass, setup_pi_integration,
):
    """BatchResult.detected_tau_diagnostics flows through to the tick log.

    Pipes a BatchResult with one accepted and one rejected diagnostic
    through `_build_tick_output` and asserts both land on
    `tick.batch_learning.detected_tau_diagnostics` with all fields
    preserved (including the rejected entry's reject_reason).
    """
    from custom_components.tasmota_irhvac.pi.batch_learning import (
        BatchResult,
        LagTauDiagnostic as BatchLagTauDiagnostic,
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    accepted = BatchLagTauDiagnostic(
        tau=5400.0, tau_opt_raw=5400.0,
        bic_gain=12.34, bic_threshold=5.30,
        r2_improvement=0.18, beta_at_tau=-2.45,
        n_eff=180, accepted=True, reject_reason="",
    )
    rejected = BatchLagTauDiagnostic(
        tau=0.0, tau_opt_raw=420.0,
        bic_gain=1.5, bic_threshold=5.0,
        r2_improvement=0.01, beta_at_tau=-0.4,
        n_eff=150, accepted=False, reject_reason="below_floor",
    )
    pi._last_batch_result = BatchResult(
        n_total=200, n_eligible=180,
        beta_batch=[0.1, -0.05, 0.0, 0.0],
        beta_current=[0.1, -0.04, 0.0, 0.0],
        residual_rms=0.05, max_coeff_change_pct=2.0,
        recommend_update=False,
        detected_tau={"solar": 5400.0, "stove": 0.0},
        detected_tau_diagnostics={"solar": accepted, "stove": rejected},
    )
    pi.fire_dispatcher()

    bl = pi.last_tick.batch_learning
    assert bl is not None
    diag = bl.detected_tau_diagnostics
    assert set(diag.keys()) == {"solar", "stove"}
    # Accepted entry: τ matches, reject_reason empty, BIC clears threshold.
    assert diag["solar"].tau == 5400.0
    assert diag["solar"].accepted is True
    assert diag["solar"].reject_reason == ""
    assert diag["solar"].bic_gain > diag["solar"].bic_threshold
    # Rejected entry: τ snapped to 0 but tau_opt_raw preserved.
    assert diag["stove"].tau == 0.0
    assert diag["stove"].tau_opt_raw == 420.0
    assert diag["stove"].accepted is False
    assert diag["stove"].reject_reason == "below_floor"


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
async def test_auto_perturb_state_transition_emits_event(hass, setup_pi_integration):
    """Changing auto-perturbation state across fire_dispatcher emits AUTO_PERTURB_STATE."""
    from custom_components.tasmota_irhvac.pi.snapshot import (
        AutoPerturbStatePayload, TickEventKind,
    )
    from unittest.mock import MagicMock

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    pi.fire_dispatcher()  # Establish baseline; records prev_auto_perturb_state

    # Force a state change in the transition-detection cache. Current
    # state is "idle" (auto-perturbation is in IDLE in basic setup);
    # setting prev to something different triggers the transition emit.
    pi._prev_auto_perturb_state = "DIFFERENT_FROM_CURRENT"
    pi.fire_dispatcher()

    events = _events_of_kind(pi.last_tick, TickEventKind.AUTO_PERTURB_STATE)
    assert events
    payload = events[-1].payload
    assert isinstance(payload, AutoPerturbStatePayload)
    assert payload.from_state == "DIFFERENT_FROM_CURRENT"
    assert payload.to_state == "idle"


@pytest.mark.asyncio
async def test_boundary_update_emits_event_on_significant_shift(
    hass, setup_pi_integration,
):
    """Posterior mean shifting > 0.1°C between fire_dispatchers emits BOUNDARY_UPDATE."""
    from custom_components.tasmota_irhvac.pi.snapshot import (
        BoundaryUpdatePayload, TickEventKind,
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # Seed prior posterior so the transition can be detected
    pi._prev_boundary_posterior_mean = 20.0
    # Mutate the boundary estimator's posterior to trigger the > 0.1°C threshold
    pi._boundary_estimator._posterior_mean = 21.0
    pi.fire_dispatcher()

    events = _events_of_kind(pi.last_tick, TickEventKind.BOUNDARY_UPDATE)
    assert events
    payload = events[-1].payload
    assert isinstance(payload, BoundaryUpdatePayload)
    assert payload.posterior_mean_after == 21.0


@pytest.mark.asyncio
async def test_maturity_gate_emits_event_on_source_change(
    hass, setup_pi_integration,
):
    """Plant-ID source changing across fire_dispatchers emits MATURITY_GATE."""
    from custom_components.tasmota_irhvac.pi.snapshot import (
        MaturityGatePayload, TickEventKind,
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller
    pi.fire_dispatcher()  # Establish baseline; records prev_plant_id_sources

    # Force a source mismatch by mutating the cached prior source. Real
    # transition (seed → estimate) happens deep inside the plant
    # identifier's update path; here we verify the emit path itself.
    pi._prev_plant_id_sources = {
        "tau_fast": "DIFFERENT",
        "tau_slow": "DIFFERENT",
        "k": "DIFFERENT",
        "theta": "DIFFERENT",
    }
    pi.fire_dispatcher()

    events = _events_of_kind(pi.last_tick, TickEventKind.MATURITY_GATE)
    assert events
    payload = events[-1].payload
    assert isinstance(payload, MaturityGatePayload)
    assert payload.source_before == "DIFFERENT"


@pytest.mark.asyncio
async def test_controller_reload_emits_once_on_first_tick(
    hass, setup_pi_integration,
):
    """async_added_to_hass stages CONTROLLER_RELOAD; it lands on the first
    published tick and is consumed (not re-emitted on later ticks).
    """
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # First fire_dispatcher after async_added_to_hass should carry the event.
    pi.fire_dispatcher()
    first = _events_of_kind(pi.last_tick, TickEventKind.CONTROLLER_RELOAD)
    assert len(first) == 1, "CONTROLLER_RELOAD should fire on first published tick"
    payload = first[0].payload
    assert isinstance(payload, ControllerReloadPayload)
    # Fresh setup — no prior stored data, so restored_from_storage=False
    # and prior_run_age_s is None.
    assert payload.restored_from_storage is False
    assert payload.prior_run_age_s is None
    # Reason is one of the documented strings.
    assert payload.reason in {"ha_start", "integration_reload"}

    # A pi_tick clears the per-tick event accumulator. After that, no
    # CONTROLLER_RELOAD should appear on subsequent ticks.
    await pi.pi_tick()
    pi.fire_dispatcher()
    second = _events_of_kind(pi.last_tick, TickEventKind.CONTROLLER_RELOAD)
    assert second == [], "CONTROLLER_RELOAD must be one-shot per init"


@pytest.mark.asyncio
async def test_pending_events_cleared_each_tick(hass, setup_pi_integration):
    """pi_tick clears the pending-events buffer at the start of each tick."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # Drain any pre-staged init events (e.g. CONTROLLER_RELOAD) so this
    # test exercises only the manual-emit clearing semantics.
    await pi.pi_tick()
    pi.fire_dispatcher()
    await pi.pi_tick()
    pi._pending_events.clear()

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
