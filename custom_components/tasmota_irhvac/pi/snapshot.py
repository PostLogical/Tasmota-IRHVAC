"""Typed per-tick snapshot of PI controller state.

`TickOutput` is the canonical contract for "what just happened in this tick" —
sensors, the diagnostics endpoint, and the event log all read from it instead
of reaching into private controller state. `DiagnosticsBundle` extends
`TickOutput` with on-demand heavy fields (multicollinearity κ, correlated
feature pairs, feature_active_counts) that are too expensive to compute every
tick but cheap when the diagnostics endpoint is queried. `OfflineBundle`
extends `DiagnosticsBundle` further with raw observation buffer dumps for
offline analysis paths.

Schema versioning lives on `TickOutput.SCHEMA_VERSION`. Evolution rule:
**additive only** — new fields default to None or empty; never rename or
remove existing fields. When a breaking change is unavoidable, bump
`SCHEMA_VERSION` and add migration logic to `from_dict()`.

This versioning is entirely separate from HA's config-entry migration
system (`MINOR_VERSION` in `__init__.py`); `SCHEMA_VERSION` is internal
to the snapshot/event-log subsystem.

All sub-dataclasses are `frozen=True, slots=True` for memory efficiency
and immutability. NOTE: `frozen=True` prevents top-level field reassignment
but does NOT make nested mutable containers (lists, dicts) immutable.
By convention, consumers must not mutate `last_tick` contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


# ── Configuration sub-snapshot ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """PI controller configuration at tick time."""

    kp: float
    ki: float
    deadband: float
    setpoint_weight: float
    tick_fallback: bool
    outdoor_temp_sensor: str | None
    model_inputs: list[dict[str, Any]]
    ff_enabled: bool
    batch_wls_enabled: bool
    plant_id_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kp": self.kp,
            "ki": self.ki,
            "deadband": self.deadband,
            "setpoint_weight": self.setpoint_weight,
            "tick_fallback": self.tick_fallback,
            "outdoor_temp_sensor": self.outdoor_temp_sensor,
            "model_inputs": list(self.model_inputs),
            "ff_enabled": self.ff_enabled,
            "batch_wls_enabled": self.batch_wls_enabled,
            "plant_id_enabled": self.plant_id_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControllerConfig:
        return cls(
            kp=data["kp"],
            ki=data["ki"],
            deadband=data["deadband"],
            setpoint_weight=data["setpoint_weight"],
            tick_fallback=data["tick_fallback"],
            outdoor_temp_sensor=data["outdoor_temp_sensor"],
            model_inputs=list(data["model_inputs"]),
            ff_enabled=data["ff_enabled"],
            batch_wls_enabled=data["batch_wls_enabled"],
            plant_id_enabled=data["plant_id_enabled"],
        )


# ── RLS model sub-snapshot ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RLSModelSnapshot:
    """RLS coefficient state for both modes plus per-tick learning signals.

    The "_heat" / "_cool" suffix on per-mode fields lets analysis tooling
    correlate residual + gain across modes within a single tick record.
    """

    heat_coefficients: dict[str, float]
    cool_coefficients: dict[str, float]
    heat_uncertainty: dict[str, float]
    heat_observation_count: int
    cool_observation_count: int
    learning_suppressed: bool
    manual_suppress_reason: str

    # Per-tick learning signals — populated only on the tick that updated
    # the RLS model. None on ticks where no observation was admitted.
    last_residual_heat: float | None
    last_residual_cool: float | None
    last_gain_vector_heat: tuple[float, ...] | None
    last_gain_vector_cool: tuple[float, ...] | None
    frozen_mask_heat: tuple[bool, ...]
    frozen_mask_cool: tuple[bool, ...]

    # CUSUM detector state (always present, even when no event is open)
    cusum_pos_heat: float
    cusum_neg_heat: float
    cusum_pos_cool: float
    cusum_neg_cool: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "heat_coefficients": dict(self.heat_coefficients),
            "cool_coefficients": dict(self.cool_coefficients),
            "heat_uncertainty": dict(self.heat_uncertainty),
            "heat_observation_count": self.heat_observation_count,
            "cool_observation_count": self.cool_observation_count,
            "learning_suppressed": self.learning_suppressed,
            "manual_suppress_reason": self.manual_suppress_reason,
            "last_residual_heat": self.last_residual_heat,
            "last_residual_cool": self.last_residual_cool,
            "last_gain_vector_heat": (
                list(self.last_gain_vector_heat)
                if self.last_gain_vector_heat is not None else None
            ),
            "last_gain_vector_cool": (
                list(self.last_gain_vector_cool)
                if self.last_gain_vector_cool is not None else None
            ),
            "frozen_mask_heat": list(self.frozen_mask_heat),
            "frozen_mask_cool": list(self.frozen_mask_cool),
            "cusum_pos_heat": self.cusum_pos_heat,
            "cusum_neg_heat": self.cusum_neg_heat,
            "cusum_pos_cool": self.cusum_pos_cool,
            "cusum_neg_cool": self.cusum_neg_cool,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLSModelSnapshot:
        return cls(
            heat_coefficients=dict(data["heat_coefficients"]),
            cool_coefficients=dict(data["cool_coefficients"]),
            heat_uncertainty=dict(data["heat_uncertainty"]),
            heat_observation_count=data["heat_observation_count"],
            cool_observation_count=data["cool_observation_count"],
            learning_suppressed=data["learning_suppressed"],
            manual_suppress_reason=data["manual_suppress_reason"],
            last_residual_heat=data["last_residual_heat"],
            last_residual_cool=data["last_residual_cool"],
            last_gain_vector_heat=(
                tuple(data["last_gain_vector_heat"])
                if data["last_gain_vector_heat"] is not None else None
            ),
            last_gain_vector_cool=(
                tuple(data["last_gain_vector_cool"])
                if data["last_gain_vector_cool"] is not None else None
            ),
            frozen_mask_heat=tuple(data["frozen_mask_heat"]),
            frozen_mask_cool=tuple(data["frozen_mask_cool"]),
            cusum_pos_heat=data["cusum_pos_heat"],
            cusum_neg_heat=data["cusum_neg_heat"],
            cusum_pos_cool=data["cusum_pos_cool"],
            cusum_neg_cool=data["cusum_neg_cool"],
        )


# ── Performance metrics sub-snapshot ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    """Cumulative control-quality metrics."""

    itae_accumulator: float
    comfort_violation_hours: float
    setpoint_changes: int
    controllable_itae: float
    uncontrollable_itae: float
    controllable_cvh: float
    uncontrollable_cvh: float
    ff_load_fraction: float
    batch_model_rms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "itae_accumulator": self.itae_accumulator,
            "comfort_violation_hours": self.comfort_violation_hours,
            "setpoint_changes": self.setpoint_changes,
            "controllable_itae": self.controllable_itae,
            "uncontrollable_itae": self.uncontrollable_itae,
            "controllable_cvh": self.controllable_cvh,
            "uncontrollable_cvh": self.uncontrollable_cvh,
            "ff_load_fraction": self.ff_load_fraction,
            "batch_model_rms": self.batch_model_rms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceSnapshot:
        return cls(
            itae_accumulator=data["itae_accumulator"],
            comfort_violation_hours=data["comfort_violation_hours"],
            setpoint_changes=data["setpoint_changes"],
            controllable_itae=data["controllable_itae"],
            uncontrollable_itae=data["uncontrollable_itae"],
            controllable_cvh=data["controllable_cvh"],
            uncontrollable_cvh=data["uncontrollable_cvh"],
            ff_load_fraction=data["ff_load_fraction"],
            batch_model_rms=data["batch_model_rms"],
        )


# ── Batch learning sub-snapshot ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BatchCoeff:
    """One coefficient's current vs. batch-recommended value."""

    current: float
    batch: float

    def to_dict(self) -> dict[str, Any]:
        return {"current": self.current, "batch": self.batch}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchCoeff:
        return cls(current=data["current"], batch=data["batch"])


@dataclass(frozen=True, slots=True)
class DriftCoefficient:
    """A coefficient flagged as drifting persistently."""

    index: int
    name: str
    consecutive_cycles: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "consecutive_cycles": self.consecutive_cycles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftCoefficient:
        return cls(
            index=data["index"],
            name=data["name"],
            consecutive_cycles=data["consecutive_cycles"],
        )


@dataclass(frozen=True, slots=True)
class DriftDetection:
    """Drift correction signs and currently-drifting coefficients."""

    drifting_coefficients: tuple[DriftCoefficient, ...]
    correction_history: dict[str, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drifting_coefficients": [d.to_dict() for d in self.drifting_coefficients],
            "correction_history": {
                k: list(v) for k, v in self.correction_history.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftDetection:
        return cls(
            drifting_coefficients=tuple(
                DriftCoefficient.from_dict(d) for d in data["drifting_coefficients"]
            ),
            correction_history={
                k: tuple(v) for k, v in data["correction_history"].items()
            },
        )


@dataclass(frozen=True, slots=True)
class ResidualPattern:
    """Time-of-day residual bucket from the last batch run."""

    start_hour: int
    end_hour: int
    mean_residual: float
    n_observations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "mean_residual": self.mean_residual,
            "n_observations": self.n_observations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualPattern:
        return cls(
            start_hour=data["start_hour"],
            end_hour=data["end_hour"],
            mean_residual=data["mean_residual"],
            n_observations=data["n_observations"],
        )


@dataclass(frozen=True, slots=True)
class BatchLearningSnapshot:
    """Last batch WLS run results."""

    last_run_mono: float
    last_run_wallclock: str | None
    n_total: int
    n_eligible: int
    residual_rms: float
    recommend_update: bool
    max_coeff_change_pct: float
    coefficients: dict[str, BatchCoeff]
    held_features: tuple[str, ...]
    n_outliers_excluded: int
    drift_detection: DriftDetection
    residual_patterns: tuple[ResidualPattern, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_run_mono": self.last_run_mono,
            "last_run_wallclock": self.last_run_wallclock,
            "n_total": self.n_total,
            "n_eligible": self.n_eligible,
            "residual_rms": self.residual_rms,
            "recommend_update": self.recommend_update,
            "max_coeff_change_pct": self.max_coeff_change_pct,
            "coefficients": {k: v.to_dict() for k, v in self.coefficients.items()},
            "held_features": list(self.held_features),
            "n_outliers_excluded": self.n_outliers_excluded,
            "drift_detection": self.drift_detection.to_dict(),
            "residual_patterns": [p.to_dict() for p in self.residual_patterns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchLearningSnapshot:
        return cls(
            last_run_mono=data["last_run_mono"],
            last_run_wallclock=data["last_run_wallclock"],
            n_total=data["n_total"],
            n_eligible=data["n_eligible"],
            residual_rms=data["residual_rms"],
            recommend_update=data["recommend_update"],
            max_coeff_change_pct=data["max_coeff_change_pct"],
            coefficients={
                k: BatchCoeff.from_dict(v) for k, v in data["coefficients"].items()
            },
            held_features=tuple(data["held_features"]),
            n_outliers_excluded=data["n_outliers_excluded"],
            drift_detection=DriftDetection.from_dict(data["drift_detection"]),
            residual_patterns=tuple(
                ResidualPattern.from_dict(p) for p in data["residual_patterns"]
            ),
        )


# ── FF contributions sub-snapshot ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FFContribution:
    """One feature's contribution to the FF offset."""

    coef: float
    filtered: float
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coef": self.coef,
            "filtered": self.filtered,
            "contribution": self.contribution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FFContribution:
        return cls(
            coef=data["coef"],
            filtered=data["filtered"],
            contribution=data["contribution"],
        )


@dataclass(frozen=True, slots=True)
class FFContributionsSnapshot:
    """Per-feature decomposition of current FF offset.

    Wire format inlines `_sum` and `_blended_offset` at the top level
    alongside per-feature entries (legacy shape).
    """

    contributions: dict[str, FFContribution]
    sum: float
    blended_offset: float | None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            name: c.to_dict() for name, c in self.contributions.items()
        }
        result["_sum"] = self.sum
        result["_blended_offset"] = self.blended_offset
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FFContributionsSnapshot:
        contributions = {
            k: FFContribution.from_dict(v)
            for k, v in data.items()
            if not k.startswith("_")
        }
        return cls(
            contributions=contributions,
            sum=data["_sum"],
            blended_offset=data["_blended_offset"],
        )


# ── Observation buffer sub-snapshots ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class ObservationBufferSnapshot:
    """Per-mode observation buffer summary (light fields).

    Heavy fields (multicollinearity, correlated_pairs, feature_active_counts)
    live in `MulticollinearityStats` and merge in at `DiagnosticsBundle`
    serialization time.
    """

    total: int
    eligible: int
    max_size: int
    leverage_min: float | None
    leverage_median: float | None
    leverage_max: float | None
    oldest_age_hours: float | None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "total": self.total,
            "eligible": self.eligible,
            "max_size": self.max_size,
        }
        if self.leverage_min is not None:
            out["leverage_min"] = self.leverage_min
        if self.leverage_median is not None:
            out["leverage_median"] = self.leverage_median
        if self.leverage_max is not None:
            out["leverage_max"] = self.leverage_max
        if self.oldest_age_hours is not None:
            out["oldest_age_hours"] = self.oldest_age_hours
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationBufferSnapshot:
        return cls(
            total=data["total"],
            eligible=data["eligible"],
            max_size=data["max_size"],
            leverage_min=data.get("leverage_min"),
            leverage_median=data.get("leverage_median"),
            leverage_max=data.get("leverage_max"),
            oldest_age_hours=data.get("oldest_age_hours"),
        )


@dataclass(frozen=True, slots=True)
class CorrelatedPair:
    """Two features whose buffer columns are highly correlated."""

    feature_a: str
    feature_b: str
    r: float

    def to_dict(self) -> dict[str, Any]:
        return {"feature_a": self.feature_a, "feature_b": self.feature_b, "r": self.r}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrelatedPair:
        return cls(
            feature_a=data["feature_a"],
            feature_b=data["feature_b"],
            r=data["r"],
        )


@dataclass(frozen=True, slots=True)
class MulticollinearityStats:
    """Heavy multicollinearity stats — only computed at diagnostics-query time."""

    condition_number: float | None
    condition_rating: str  # "weak" | "moderate" | "severe" | "insufficient_data"
    correlated_pairs: tuple[CorrelatedPair, ...]
    feature_active_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "condition_rating": self.condition_rating,
            "correlated_pairs": [p.to_dict() for p in self.correlated_pairs],
        }
        if self.condition_number is not None:
            out["condition_number"] = self.condition_number
        if self.feature_active_counts:
            out["feature_active_counts"] = dict(self.feature_active_counts)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MulticollinearityStats:
        return cls(
            condition_number=data.get("condition_number"),
            condition_rating=data["condition_rating"],
            correlated_pairs=tuple(
                CorrelatedPair.from_dict(p) for p in data["correlated_pairs"]
            ),
            feature_active_counts=dict(data.get("feature_active_counts", {})),
        )


# ── Lag filter sub-snapshot ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LagFilterState:
    """One model input's lag-filter state."""

    entity_id: str
    filtered_value: float
    decay_constant_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "filtered_value": self.filtered_value,
            "decay_constant_s": self.decay_constant_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LagFilterState:
        return cls(
            entity_id=data["entity_id"],
            filtered_value=data["filtered_value"],
            decay_constant_s=data["decay_constant_s"],
        )


@dataclass(frozen=True, slots=True)
class LagFilterSnapshot:
    """All model inputs' lag-filter states at tick time."""

    states: tuple[LagFilterState, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"states": [s.to_dict() for s in self.states]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LagFilterSnapshot:
        return cls(
            states=tuple(LagFilterState.from_dict(s) for s in data["states"]),
        )


# ── Per-tick observation context + events ──────────────────────────────


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """What happened with this tick's observation, if one was recorded.

    None when this tick didn't produce an observation (HP off, sensor
    unavailable, etc.).
    """

    admitted: bool
    clamped: bool
    clamped_reason: str
    leverage_score: float | None
    mode: str  # "heat" or "cool"
    raw_readings: dict[str, float]  # entity_id → value at this tick
    feature_vector: tuple[float, ...]  # x vector that fed RLS

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "clamped": self.clamped,
            "clamped_reason": self.clamped_reason,
            "leverage_score": self.leverage_score,
            "mode": self.mode,
            "raw_readings": dict(self.raw_readings),
            "feature_vector": list(self.feature_vector),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationContext:
        return cls(
            admitted=data["admitted"],
            clamped=data["clamped"],
            clamped_reason=data["clamped_reason"],
            leverage_score=data["leverage_score"],
            mode=data["mode"],
            raw_readings=dict(data["raw_readings"]),
            feature_vector=tuple(data["feature_vector"]),
        )


@dataclass(frozen=True, slots=True)
class TickEvent:
    """A discrete event that fired during this tick.

    `kind` discriminates the payload shape. Known kinds (will tighten to
    enum once emitters land in stage 2):
    - "batch_run": full BatchResult-shaped payload
    - "anomaly_started" / "anomaly_detected": CUSUM event boundaries
    - "mode_change": heat ↔ cool ↔ off
    - "setpoint_change": user-initiated
    - "maturity_gate": seed → estimate transition for τ_slow / kp
    - "learning_suppression_change": disturbance suppress flip
    - "auto_perturbation_state_change"
    - "boundary_estimator_update"
    """

    kind: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TickEvent:
        return cls(kind=data["kind"], payload=dict(data["payload"]))


# ── Top-level tick output ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TickOutput:
    """First-class output of one PI tick.

    Sensors, diagnostics, and the event log all read from this dataclass
    instead of reaching into controller internals. `to_dict()` produces
    output matching the legacy `get_full_diagnostics()` shape so existing
    consumers don't break. New top-level fields prefixed with `_` are
    additive.

    Schema versioning: see module docstring for evolution rules.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    # Timing + framing
    ts_mono: float
    ts_wall: float
    zone_label: str

    # Top-level live state (matches current get_full_diagnostics top level)
    enabled: bool
    paused: bool
    desired_temp: float | None
    hp_setpoint: float | None
    integral: float
    integral_convergence: float
    ff_offset: float
    ff_confidence: float
    outdoor_temp: float | None
    sensor_unavailable: bool
    sensor_recovery_pending: bool
    room_temp_rate: float
    tau_estimate: float | None
    tau_fast: float | None
    tau_slow: float | None

    # Typed sub-snapshots
    config: ControllerConfig
    rls_model: RLSModelSnapshot
    performance: PerformanceSnapshot
    batch_learning: BatchLearningSnapshot | None
    observation_buffer_heat: ObservationBufferSnapshot
    observation_buffer_cool: ObservationBufferSnapshot
    ff_contributions: FFContributionsSnapshot
    lag_filter: LagFilterSnapshot

    # Already-encapsulated state — referenced as opaque dicts since their
    # owning modules expose `.as_dict()` and the shapes are stable.
    plant_identification: dict[str, Any] | None
    greybox_observer: dict[str, Any] | None
    greybox_bridge: dict[str, Any] | None
    greybox_buffer: dict[str, Any]
    boundary_estimator: dict[str, Any]
    regime_probe: dict[str, Any]

    # Per-tick fields with day-1 defaults — populated by emitters in
    # stage 2+.
    observation: ObservationContext | None = None
    events: tuple[TickEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the legacy `get_full_diagnostics()` wire format.

        New top-level fields (`_schema_version`, `_ts_mono`, `_ts_wall`,
        `_zone_label`, `_observation`, `_events`, `_lag_filter`) are
        prefixed with `_` to keep them out of the legacy field set.
        Existing consumers reading named fields like `enabled`, `config`,
        `rls_model`, etc. see no change.
        """
        out: dict[str, Any] = {
            "_schema_version": self.SCHEMA_VERSION,
            "_ts_mono": self.ts_mono,
            "_ts_wall": self.ts_wall,
            "_zone_label": self.zone_label,
            "enabled": self.enabled,
            "paused": self.paused,
            "desired_temp": self.desired_temp,
            "hp_setpoint": self.hp_setpoint,
            "integral": self.integral,
            "integral_convergence": self.integral_convergence,
            "ff_offset": self.ff_offset,
            "ff_confidence": self.ff_confidence,
            "outdoor_temp": self.outdoor_temp,
            "sensor_unavailable": self.sensor_unavailable,
            "sensor_recovery_pending": self.sensor_recovery_pending,
            "room_temp_rate": self.room_temp_rate,
            "tau_estimate": self.tau_estimate,
            "tau_fast": self.tau_fast,
            "tau_slow": self.tau_slow,
            "plant_identification": self.plant_identification,
            "config": self.config.to_dict(),
            "rls_model": self.rls_model.to_dict(),
            "performance": self.performance.to_dict(),
            "batch_learning": (
                self.batch_learning.to_dict()
                if self.batch_learning is not None else None
            ),
            "greybox_observer": self.greybox_observer,
            "greybox_bridge": self.greybox_bridge,
            "observation_buffer_heat": self.observation_buffer_heat.to_dict(),
            "observation_buffer_cool": self.observation_buffer_cool.to_dict(),
            "greybox_buffer": self.greybox_buffer,
            "boundary_estimator": self.boundary_estimator,
            "regime_probe": self.regime_probe,
            "ff_contributions": self.ff_contributions.to_dict(),
            "_lag_filter": self.lag_filter.to_dict(),
        }
        if self.observation is not None:
            out["_observation"] = self.observation.to_dict()
        if self.events:
            out["_events"] = [e.to_dict() for e in self.events]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TickOutput:
        """Reconstruct a TickOutput from its serialized form.

        Raises ValueError if the schema_version doesn't match. Once we
        bump SCHEMA_VERSION, add migration logic here.
        """
        version = data.get("_schema_version")
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"TickOutput schema version mismatch: expected "
                f"{cls.SCHEMA_VERSION}, got {version}"
            )
        return cls(
            ts_mono=data["_ts_mono"],
            ts_wall=data["_ts_wall"],
            zone_label=data["_zone_label"],
            enabled=data["enabled"],
            paused=data["paused"],
            desired_temp=data["desired_temp"],
            hp_setpoint=data["hp_setpoint"],
            integral=data["integral"],
            integral_convergence=data["integral_convergence"],
            ff_offset=data["ff_offset"],
            ff_confidence=data["ff_confidence"],
            outdoor_temp=data["outdoor_temp"],
            sensor_unavailable=data["sensor_unavailable"],
            sensor_recovery_pending=data["sensor_recovery_pending"],
            room_temp_rate=data["room_temp_rate"],
            tau_estimate=data["tau_estimate"],
            tau_fast=data["tau_fast"],
            tau_slow=data["tau_slow"],
            config=ControllerConfig.from_dict(data["config"]),
            rls_model=RLSModelSnapshot.from_dict(data["rls_model"]),
            performance=PerformanceSnapshot.from_dict(data["performance"]),
            batch_learning=(
                BatchLearningSnapshot.from_dict(data["batch_learning"])
                if data["batch_learning"] is not None else None
            ),
            observation_buffer_heat=ObservationBufferSnapshot.from_dict(
                data["observation_buffer_heat"]
            ),
            observation_buffer_cool=ObservationBufferSnapshot.from_dict(
                data["observation_buffer_cool"]
            ),
            ff_contributions=FFContributionsSnapshot.from_dict(
                data["ff_contributions"]
            ),
            lag_filter=LagFilterSnapshot.from_dict(data["_lag_filter"]),
            plant_identification=data["plant_identification"],
            greybox_observer=data["greybox_observer"],
            greybox_bridge=data["greybox_bridge"],
            greybox_buffer=data["greybox_buffer"],
            boundary_estimator=data["boundary_estimator"],
            regime_probe=data["regime_probe"],
            observation=(
                ObservationContext.from_dict(data["_observation"])
                if "_observation" in data else None
            ),
            events=tuple(
                TickEvent.from_dict(e) for e in data.get("_events", [])
            ),
        )

    @classmethod
    def empty(cls, zone_label: str = "") -> TickOutput:
        """Return a default-empty TickOutput for use before the first tick.

        Used by `NullController.last_tick` and as the initial value of
        `PIController._last_tick` so consumers don't need to handle None.
        """
        return cls(
            ts_mono=0.0,
            ts_wall=0.0,
            zone_label=zone_label,
            enabled=False,
            paused=False,
            desired_temp=None,
            hp_setpoint=None,
            integral=0.0,
            integral_convergence=0.0,
            ff_offset=0.0,
            ff_confidence=0.0,
            outdoor_temp=None,
            sensor_unavailable=False,
            sensor_recovery_pending=False,
            room_temp_rate=0.0,
            tau_estimate=None,
            tau_fast=None,
            tau_slow=None,
            config=ControllerConfig(
                kp=0.0, ki=0.0, deadband=0.0, setpoint_weight=0.0,
                tick_fallback=False, outdoor_temp_sensor=None,
                model_inputs=[], ff_enabled=False, batch_wls_enabled=False,
                plant_id_enabled=False,
            ),
            rls_model=RLSModelSnapshot(
                heat_coefficients={}, cool_coefficients={},
                heat_uncertainty={}, heat_observation_count=0,
                cool_observation_count=0, learning_suppressed=False,
                manual_suppress_reason="",
                last_residual_heat=None, last_residual_cool=None,
                last_gain_vector_heat=None, last_gain_vector_cool=None,
                frozen_mask_heat=(), frozen_mask_cool=(),
                cusum_pos_heat=0.0, cusum_neg_heat=0.0,
                cusum_pos_cool=0.0, cusum_neg_cool=0.0,
            ),
            performance=PerformanceSnapshot(
                itae_accumulator=0.0, comfort_violation_hours=0.0,
                setpoint_changes=0, controllable_itae=0.0,
                uncontrollable_itae=0.0, controllable_cvh=0.0,
                uncontrollable_cvh=0.0, ff_load_fraction=0.0,
                batch_model_rms=None,
            ),
            batch_learning=None,
            observation_buffer_heat=ObservationBufferSnapshot(
                total=0, eligible=0, max_size=0,
                leverage_min=None, leverage_median=None, leverage_max=None,
                oldest_age_hours=None,
            ),
            observation_buffer_cool=ObservationBufferSnapshot(
                total=0, eligible=0, max_size=0,
                leverage_min=None, leverage_median=None, leverage_max=None,
                oldest_age_hours=None,
            ),
            ff_contributions=FFContributionsSnapshot(
                contributions={}, sum=0.0, blended_offset=None,
            ),
            lag_filter=LagFilterSnapshot(states=()),
            plant_identification=None,
            greybox_observer=None,
            greybox_bridge=None,
            greybox_buffer={},
            boundary_estimator={},
            regime_probe={},
        )


# ── Diagnostics-time and offline bundles ───────────────────────────────


@dataclass(frozen=True, slots=True)
class DiagnosticsBundle:
    """`TickOutput` plus heavy fields computed only at diagnostics-query time.

    The HA diagnostics endpoint returns `bundle.to_dict()`, which merges
    multicollinearity stats into the per-buffer snapshots so the wire
    format matches the legacy `get_full_diagnostics()` shape exactly.

    Optional `full_p_heat` / `full_p_cool` populate when the
    `set_debug_capture(full_p=True)` service has been called (Stage 3).
    """

    tick: TickOutput
    heat_multicollinearity: MulticollinearityStats
    cool_multicollinearity: MulticollinearityStats
    full_p_heat: tuple[tuple[float, ...], ...] | None = None
    full_p_cool: tuple[tuple[float, ...], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = self.tick.to_dict()
        out["observation_buffer_heat"] = {
            **out["observation_buffer_heat"],
            **self.heat_multicollinearity.to_dict(),
        }
        out["observation_buffer_cool"] = {
            **out["observation_buffer_cool"],
            **self.cool_multicollinearity.to_dict(),
        }
        if self.full_p_heat is not None:
            out["full_p_heat"] = [list(row) for row in self.full_p_heat]
        if self.full_p_cool is not None:
            out["full_p_cool"] = [list(row) for row in self.full_p_cool]
        return out


@dataclass(frozen=True, slots=True)
class OfflineBundle:
    """`DiagnosticsBundle` plus raw observation buffer dumps for offline analysis.

    Used by the `get_diagnostic_dump()` path. Raw buffers are large; only
    materialize this bundle when the caller actually needs the dumps.
    """

    diagnostics: DiagnosticsBundle
    raw_buffer_heat: tuple[dict[str, Any], ...]
    raw_buffer_cool: tuple[dict[str, Any], ...]
    model_input_configs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        out = self.diagnostics.to_dict()
        # `get_diagnostic_dump`-shaped layout: raw buffers replace the
        # summary buffers; counts and condition stats added separately
        # at the top level.
        out["observation_buffer_heat"] = list(self.raw_buffer_heat)
        out["observation_buffer_cool"] = list(self.raw_buffer_cool)
        out["buffer_size_heat"] = len(self.raw_buffer_heat)
        out["buffer_size_cool"] = len(self.raw_buffer_cool)
        out["model_input_configs"] = list(self.model_input_configs)
        return out
