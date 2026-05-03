"""Unit tests for the typed snapshot dataclasses (Stage 1).

These test the dataclasses in isolation: construction, to_dict() shape,
from_dict() roundtrip, frozen behavior, schema versioning. No PI controller
integration — that's Stage 2+.
"""

from __future__ import annotations

import dataclasses

import pytest

from custom_components.tasmota_irhvac.pi.snapshot import (
    AnomalyDetectedPayload,
    AutoPerturbStatePayload,
    BatchCoeff,
    BatchLearningSnapshot,
    BatchRunPayload,
    BoundaryUpdatePayload,
    ControllerConfig,
    CorrelatedPair,
    DiagnosticsBundle,
    DriftCoefficient,
    DriftDetection,
    FFContribution,
    FFContributionsSnapshot,
    LagFilterSnapshot,
    LagFilterState,
    LearningSuppressionChangePayload,
    MaturityGatePayload,
    ModeChangePayload,
    MulticollinearityStats,
    ObservationBufferSnapshot,
    ObservationContext,
    PerformanceSnapshot,
    ResidualPattern,
    RLSModelSnapshot,
    SetpointChangeUserPayload,
    TickEvent,
    TickEventKind,
    TickOutput,
)


# ── Schema version ─────────────────────────────────────────────────────


def test_tick_output_schema_version_is_one():
    assert TickOutput.SCHEMA_VERSION == 1


# ── Per-sub-dataclass roundtrip ────────────────────────────────────────


def test_controller_config_roundtrip():
    c = ControllerConfig(
        kp=1.5, ki=0.15, deadband=0.5, setpoint_weight=0.3,
        tick_fallback=900.0, outdoor_temp_sensor="sensor.outdoor",
        model_inputs=[{"name": "solar", "entity_id": "sensor.solar"}],
        ff_enabled=True, batch_wls_enabled=True, plant_id_enabled=False,
    )
    assert ControllerConfig.from_dict(c.to_dict()) == c


def test_rls_model_snapshot_roundtrip():
    r = RLSModelSnapshot(
        heat_seeds={"intercept": 0.1, "outdoor_delta": 0.04},
        cool_seeds={"intercept": -0.1, "outdoor_delta": -0.05},
        heat_uncertainty={"intercept": 0.01, "outdoor_delta": 0.02},
        heat_observation_count=42, cool_observation_count=10,
        learning_suppressed=False, manual_suppress_reason="",
        last_residual=0.05, last_gain_vector=(0.1, 0.05),
        frozen_mask_heat=(False, False, True),
        frozen_mask_cool=(False, False, True),
        cusum_pos=2.5, cusum_neg=1.2,
    )
    assert RLSModelSnapshot.from_dict(r.to_dict()) == r


def test_rls_model_snapshot_with_none_residual_and_gain():
    """Online RLS removed in pre45; last_residual/gain may be None."""
    r = RLSModelSnapshot(
        heat_seeds={}, cool_seeds={}, heat_uncertainty={},
        heat_observation_count=0, cool_observation_count=0,
        learning_suppressed=False, manual_suppress_reason="",
        last_residual=None, last_gain_vector=None,
        frozen_mask_heat=(), frozen_mask_cool=(),
        cusum_pos=0.0, cusum_neg=0.0,
    )
    assert RLSModelSnapshot.from_dict(r.to_dict()) == r


def test_performance_snapshot_roundtrip():
    p = PerformanceSnapshot(
        itae_accumulator=15.4, comfort_violation_hours=0.3,
        setpoint_changes=2, controllable_itae=12.0, uncontrollable_itae=3.4,
        controllable_cvh=0.2, uncontrollable_cvh=0.1,
        ff_load_fraction=0.65, batch_model_rms=0.045,
    )
    assert PerformanceSnapshot.from_dict(p.to_dict()) == p


def test_performance_snapshot_with_none_batch_rms():
    p = PerformanceSnapshot(
        itae_accumulator=0.0, comfort_violation_hours=0.0,
        setpoint_changes=0, controllable_itae=0.0, uncontrollable_itae=0.0,
        controllable_cvh=0.0, uncontrollable_cvh=0.0,
        ff_load_fraction=0.0, batch_model_rms=None,
    )
    assert PerformanceSnapshot.from_dict(p.to_dict()) == p


def test_batch_coeff_roundtrip():
    b = BatchCoeff(current=0.1, batch=0.12)
    assert BatchCoeff.from_dict(b.to_dict()) == b


def test_drift_coefficient_roundtrip():
    d = DriftCoefficient(index=2, name="solar", consecutive_cycles=3)
    assert DriftCoefficient.from_dict(d.to_dict()) == d


def test_drift_detection_roundtrip():
    d = DriftDetection(
        drifting_coefficients=(
            DriftCoefficient(index=2, name="solar", consecutive_cycles=3),
        ),
        correction_history={"solar": (1, -1, 1, 1), "outdoor": (1,)},
    )
    assert DriftDetection.from_dict(d.to_dict()) == d


def test_residual_pattern_roundtrip():
    p = ResidualPattern(start_hour=14, end_hour=16, mean_residual=-0.6, n_observations=24)
    assert ResidualPattern.from_dict(p.to_dict()) == p


def test_batch_learning_snapshot_roundtrip():
    b = BatchLearningSnapshot(
        last_run_mono=12345.6, last_run_wallclock="2026-05-02T10:00:00Z",
        n_total=500, n_eligible=420, residual_rms=0.045,
        recommend_update=True, max_coeff_change_pct=12.5,
        coefficients={"intercept": BatchCoeff(0.1, 0.12)},
        held_features=("sin_hour", "cos_hour"),
        n_outliers_excluded=8,
        drift_detection=DriftDetection(
            drifting_coefficients=(),
            correction_history={"intercept": (1, 1, 1)},
        ),
        residual_patterns=(
            ResidualPattern(14, 16, -0.6, 24),
            ResidualPattern(2, 4, 0.4, 18),
        ),
        beta_std_err=(0.01, 0.02, 0.03),
        beta_blended=(0.105, -0.04, 0.0),
        blend_gains=(0.5, 0.3, 0.0),
        feature_vif=(1.2, 3.4, 1.0),
        detected_tau={"solar": 300.0, "boiler": 60.0},
        plant_snapshot={"tau_slow": {"value": 120.0, "observations": 50}},
    )
    assert BatchLearningSnapshot.from_dict(b.to_dict()) == b


def test_batch_learning_snapshot_legacy_dict_missing_extras():
    """from_dict tolerates legacy serialized form missing the BatchResult extras."""
    legacy = {
        "last_run_mono": 0.0, "last_run_wallclock": None,
        "n_total": 0, "n_eligible": 0, "residual_rms": 0.0,
        "recommend_update": False, "max_coeff_change_pct": 0.0,
        "coefficients": {}, "held_features": [], "n_outliers_excluded": 0,
        "drift_detection": {"drifting_coefficients": [], "correction_history": {}},
        "residual_patterns": [],
        # No beta_std_err / blend_gains / etc. — defaults to empty
    }
    b = BatchLearningSnapshot.from_dict(legacy)
    assert b.beta_std_err == ()
    assert b.blend_gains == ()
    assert b.detected_tau == {}


def test_ff_contribution_roundtrip():
    c = FFContribution(coef=-0.04, filtered=12.5, contribution=-0.5)
    assert FFContribution.from_dict(c.to_dict()) == c


def test_ff_contributions_snapshot_roundtrip():
    s = FFContributionsSnapshot(
        contributions={
            "intercept": FFContribution(0.1, 1.0, 0.1),
            "outdoor_delta": FFContribution(-0.04, 12.5, -0.5),
        },
        sum=-0.4, blended_offset=-0.43,
    )
    assert FFContributionsSnapshot.from_dict(s.to_dict()) == s


def test_observation_buffer_snapshot_roundtrip_full():
    o = ObservationBufferSnapshot(
        total=500, eligible=420, max_size=500,
        leverage_min=0.001, leverage_median=0.005, leverage_max=0.05,
        oldest_age_hours=72.0,
    )
    assert ObservationBufferSnapshot.from_dict(o.to_dict()) == o


def test_observation_buffer_snapshot_roundtrip_minimal():
    """Empty buffer: optional leverage/age fields not present."""
    o = ObservationBufferSnapshot(
        total=0, eligible=0, max_size=500,
        leverage_min=None, leverage_median=None, leverage_max=None,
        oldest_age_hours=None,
    )
    assert ObservationBufferSnapshot.from_dict(o.to_dict()) == o


def test_correlated_pair_roundtrip():
    p = CorrelatedPair(feature_a="solar", feature_b="outdoor_delta", r=0.85)
    assert CorrelatedPair.from_dict(p.to_dict()) == p


def test_multicollinearity_stats_roundtrip():
    m = MulticollinearityStats(
        condition_number=42.3, condition_rating="moderate",
        correlated_pairs=(CorrelatedPair("solar", "outdoor_delta", 0.85),),
        feature_active_counts={"solar": 200, "outdoor_delta": 500},
    )
    assert MulticollinearityStats.from_dict(m.to_dict()) == m


def test_multicollinearity_stats_insufficient_data():
    m = MulticollinearityStats(
        condition_number=None, condition_rating="insufficient_data",
        correlated_pairs=(), feature_active_counts={},
    )
    assert MulticollinearityStats.from_dict(m.to_dict()) == m


def test_lag_filter_state_roundtrip():
    s = LagFilterState(
        entity_id="sensor.solar", filtered_value=12.5, decay_constant_s=300.0,
    )
    assert LagFilterState.from_dict(s.to_dict()) == s


def test_lag_filter_snapshot_roundtrip():
    s = LagFilterSnapshot(states=(
        LagFilterState("sensor.solar", 12.5, 300.0),
        LagFilterState("sensor.boiler", 0.0, 60.0),
    ))
    assert LagFilterSnapshot.from_dict(s.to_dict()) == s


def test_observation_context_roundtrip():
    o = ObservationContext(
        admitted=True, clamped=False, clamped_reason="",
        leverage_score=0.012, mode="heat",
        raw_readings={"sensor.outdoor": 5.0, "sensor.solar": 200.0},
        feature_vector=(1.0, 12.5, 0.5, -0.85),
    )
    assert ObservationContext.from_dict(o.to_dict()) == o


def test_alert_roundtrip():
    """Alert.from_dict reconstructs a typed alert."""
    from custom_components.tasmota_irhvac.pi.snapshot import Alert
    a = Alert(message="High integral", code="integral_warn", severity="Warning")
    assert Alert.from_dict(a.to_dict()) == a


def test_tick_event_batch_run_roundtrip():
    e = TickEvent(
        kind=TickEventKind.BATCH_RUN,
        payload=BatchRunPayload(
            n_eligible=420, residual_rms=0.045, recommend_update=True,
            max_coeff_change_pct=12.5, n_outliers_excluded=8,
        ),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


def test_tick_event_anomaly_detected_roundtrip():
    e = TickEvent(
        kind=TickEventKind.ANOMALY_DETECTED,
        payload=AnomalyDetectedPayload(
            mode="heat", mean_residual=0.8, peak_cusum=12.5, tick_count=42,
        ),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


def test_tick_event_mode_change_roundtrip():
    e = TickEvent(
        kind=TickEventKind.MODE_CHANGE,
        payload=ModeChangePayload(from_mode="off", to_mode="heat"),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


def test_tick_event_setpoint_change_roundtrip():
    e = TickEvent(
        kind=TickEventKind.SETPOINT_CHANGE_USER,
        payload=SetpointChangeUserPayload(from_setpoint=21.0, to_setpoint=22.5),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


def test_tick_event_maturity_gate_roundtrip():
    e = TickEvent(
        kind=TickEventKind.MATURITY_GATE,
        payload=MaturityGatePayload(
            parameter="tau_slow", source_before="seed", source_after="estimate",
            value=120.5, observations=15,
        ),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


def test_tick_event_learning_suppression_roundtrip():
    e = TickEvent(
        kind=TickEventKind.LEARNING_SUPPRESSION_CHANGE,
        payload=LearningSuppressionChangePayload(
            was_suppressed=False, is_suppressed=True,
            active_suppressors=("setpoint_change", "manual"),
            manual=True,
        ),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


def test_tick_event_auto_perturb_roundtrip():
    e = TickEvent(
        kind=TickEventKind.AUTO_PERTURB_STATE,
        payload=AutoPerturbStatePayload(
            from_state="IDLE", to_state="STEP_ACTIVE", cycles_completed=3,
        ),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


def test_tick_event_boundary_update_roundtrip():
    e = TickEvent(
        kind=TickEventKind.BOUNDARY_UPDATE,
        payload=BoundaryUpdatePayload(
            posterior_mean_before=20.5, posterior_mean_after=21.0,
            posterior_std=0.3, n_observations=12, confident=True,
        ),
    )
    assert TickEvent.from_dict(e.to_dict()) == e


# ── TickOutput end-to-end ──────────────────────────────────────────────


def _minimal_tick(zone_label: str = "test") -> TickOutput:
    """Return a minimal TickOutput for testing — all required fields populated."""
    return TickOutput.empty(zone_label=zone_label)


def test_tick_output_empty_constructor():
    """TickOutput.empty() produces a valid roundtrippable dataclass."""
    tick = _minimal_tick()
    assert tick.zone_label == "test"
    assert tick.SCHEMA_VERSION == 1


def test_tick_output_roundtrip_minimal():
    tick = _minimal_tick()
    assert TickOutput.from_dict(tick.to_dict()) == tick


def test_tick_output_roundtrip_with_observation():
    tick = dataclasses.replace(
        _minimal_tick(),
        observation=ObservationContext(
            admitted=True, clamped=False, clamped_reason="",
            leverage_score=0.012, mode="heat",
            raw_readings={"sensor.outdoor": 5.0},
            feature_vector=(1.0, 12.5),
        ),
    )
    assert TickOutput.from_dict(tick.to_dict()) == tick


def test_tick_output_roundtrip_with_events():
    tick = dataclasses.replace(
        _minimal_tick(),
        events=(
            TickEvent(
                kind=TickEventKind.MODE_CHANGE,
                payload=ModeChangePayload(from_mode="heat", to_mode="cool"),
            ),
            TickEvent(
                kind=TickEventKind.SETPOINT_CHANGE_USER,
                payload=SetpointChangeUserPayload(from_setpoint=21.0, to_setpoint=22.0),
            ),
        ),
    )
    assert TickOutput.from_dict(tick.to_dict()) == tick


def test_tick_output_to_dict_includes_underscore_fields():
    tick = _minimal_tick("living_room")
    d = tick.to_dict()
    assert d["_schema_version"] == 1
    assert d["_zone_label"] == "living_room"
    assert "_ts_mono" in d
    assert "_ts_wall" in d
    assert "_lag_filter" in d


def test_tick_output_to_dict_has_legacy_top_level_keys():
    """Wire format compatibility: keys that current consumers expect."""
    d = _minimal_tick().to_dict()
    for legacy_key in (
        "enabled", "paused", "desired_temp", "hp_setpoint",
        "integral", "integral_convergence", "ff_offset", "ff_confidence",
        "outdoor_temp", "sensor_unavailable", "sensor_recovery_pending",
        "room_temp_rate", "tau_estimate", "tau_fast", "tau_slow",
        "config", "rls_model", "performance", "batch_learning",
        "observation_buffer_heat", "observation_buffer_cool",
        "greybox_observer", "greybox_bridge", "greybox_buffer",
        "boundary_estimator", "regime_probe", "ff_contributions",
        "plant_identification",
    ):
        assert legacy_key in d, f"missing legacy key: {legacy_key}"


def test_tick_output_from_dict_rejects_wrong_schema_version():
    tick = _minimal_tick()
    d = tick.to_dict()
    d["_schema_version"] = 99
    with pytest.raises(ValueError, match="schema version mismatch"):
        TickOutput.from_dict(d)


def test_tick_output_is_frozen():
    tick = _minimal_tick()
    with pytest.raises(dataclasses.FrozenInstanceError):
        tick.zone_label = "other"  # type: ignore[misc]


def test_sub_snapshots_are_frozen():
    """Spot-check: sub-snapshots also frozen."""
    c = ControllerConfig(
        kp=1.5, ki=0.15, deadband=0.5, setpoint_weight=0.3,
        tick_fallback=900.0, outdoor_temp_sensor=None,
        model_inputs=[], ff_enabled=False, batch_wls_enabled=False,
        plant_id_enabled=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.kp = 2.0  # type: ignore[misc]


# ── DiagnosticsBundle ──────────────────────────────────────────────────


def _empty_multicollinearity() -> MulticollinearityStats:
    return MulticollinearityStats(
        condition_number=None, condition_rating="insufficient_data",
        correlated_pairs=(), feature_active_counts={},
    )


def test_diagnostics_bundle_to_dict_merges_multicollinearity():
    bundle = DiagnosticsBundle(
        tick=_minimal_tick(),
        heat_multicollinearity=MulticollinearityStats(
            condition_number=42.0, condition_rating="moderate",
            correlated_pairs=(CorrelatedPair("a", "b", 0.8),),
            feature_active_counts={"a": 100},
        ),
        cool_multicollinearity=_empty_multicollinearity(),
    )
    d = bundle.to_dict()
    # Heavies merged into the buffer dict
    assert d["observation_buffer_heat"]["condition_number"] == 42.0
    assert d["observation_buffer_heat"]["condition_rating"] == "moderate"
    assert len(d["observation_buffer_heat"]["correlated_pairs"]) == 1
    # Cool buffer also has its (empty) heavies merged
    assert d["observation_buffer_cool"]["condition_rating"] == "insufficient_data"


def test_diagnostics_bundle_full_p_omitted_by_default():
    bundle = DiagnosticsBundle(
        tick=_minimal_tick(),
        heat_multicollinearity=_empty_multicollinearity(),
        cool_multicollinearity=_empty_multicollinearity(),
    )
    d = bundle.to_dict()
    assert "full_p_heat" not in d
    assert "full_p_cool" not in d


def test_diagnostics_bundle_full_p_present_when_set():
    bundle = DiagnosticsBundle(
        tick=_minimal_tick(),
        heat_multicollinearity=_empty_multicollinearity(),
        cool_multicollinearity=_empty_multicollinearity(),
        full_p_heat=((1.0, 0.0), (0.0, 1.0)),
        full_p_cool=((1.0, 0.0), (0.0, 1.0)),
    )
    d = bundle.to_dict()
    assert d["full_p_heat"] == [[1.0, 0.0], [0.0, 1.0]]
    assert d["full_p_cool"] == [[1.0, 0.0], [0.0, 1.0]]


