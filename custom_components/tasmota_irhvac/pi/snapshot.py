"""Typed per-tick snapshot of PI controller state.

`TickOutput` is the canonical contract for "what just happened in this tick" —
sensors, the diagnostics endpoint, and the future event log all read from it
instead of reaching into private controller state. `DiagnosticsBundle` extends
`TickOutput` with on-demand heavy fields (multicollinearity κ, correlated
feature pairs, feature_active_counts) that are too expensive to compute every
tick but cheap when the diagnostics endpoint is queried.

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
from enum import StrEnum
from typing import Any, ClassVar


# ── Configuration sub-snapshot ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """PI controller configuration at tick time."""

    kp: float
    ki: float
    deadband: float
    setpoint_weight: float
    tick_fallback: float  # fallback tick interval in seconds
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

    # Coefficients in user-facing seed space (positive = warms room).
    # Converted from internal β via `RLSModel.beta_to_seed` so consumers
    # don't need to track the negation convention; see rls_model.py:281.
    heat_seeds: dict[str, float]
    cool_seeds: dict[str, float]
    heat_uncertainty: dict[str, float]
    heat_observation_count: int
    cool_observation_count: int
    learning_suppressed: bool
    manual_suppress_reason: str

    # Per-tick learning signals. `last_residual` is the most recent
    # prediction residual `(hp_setpoint - desired) - rls.predict(x)`
    # from the currently-active RLS (heat OR cool, not both — only one
    # is active per tick). None if no residual has been computed yet.
    last_residual: float | None
    # Reserved for future per-tick RLS update gain vectors. Online RLS
    # was removed in pre45, so this is None today; field retained so the
    # schema doesn't bump when a future update path lands.
    last_gain_vector: tuple[float, ...] | None
    # Per-mode static frozen masks (locked coefficients).
    frozen_mask_heat: tuple[bool, ...]
    frozen_mask_cool: tuple[bool, ...]

    # CUSUM detector state — shared across modes in production. Single
    # pos/neg pair, regardless of which mode the residual came from.
    cusum_pos: float
    cusum_neg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "heat_seeds": dict(self.heat_seeds),
            "cool_seeds": dict(self.cool_seeds),
            "heat_uncertainty": dict(self.heat_uncertainty),
            "heat_observation_count": self.heat_observation_count,
            "cool_observation_count": self.cool_observation_count,
            "learning_suppressed": self.learning_suppressed,
            "manual_suppress_reason": self.manual_suppress_reason,
            "last_residual": self.last_residual,
            "last_gain_vector": (
                list(self.last_gain_vector)
                if self.last_gain_vector is not None else None
            ),
            "frozen_mask_heat": list(self.frozen_mask_heat),
            "frozen_mask_cool": list(self.frozen_mask_cool),
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLSModelSnapshot:
        return cls(
            heat_seeds=dict(data["heat_seeds"]),
            cool_seeds=dict(data["cool_seeds"]),
            heat_uncertainty=dict(data["heat_uncertainty"]),
            heat_observation_count=data["heat_observation_count"],
            cool_observation_count=data["cool_observation_count"],
            learning_suppressed=data["learning_suppressed"],
            manual_suppress_reason=data["manual_suppress_reason"],
            last_residual=data["last_residual"],
            last_gain_vector=(
                tuple(data["last_gain_vector"])
                if data["last_gain_vector"] is not None else None
            ),
            frozen_mask_heat=tuple(data["frozen_mask_heat"]),
            frozen_mask_cool=tuple(data["frozen_mask_cool"]),
            cusum_pos=data["cusum_pos"],
            cusum_neg=data["cusum_neg"],
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
class UnlockEvaluationRecord:
    """Per-feature outcome of a single unlock-evaluation cycle.

    Captures *why* a frozen feature stayed frozen (or was unfrozen)
    based on the full-model batch result. Built by
    `_evaluate_feature_unlocks` so bundles can answer "why is feature X
    still held" without re-running batch WLS offline.

    `gate_failed` is one of:
      - "held"     — feature in `full_result.held_features` (variance gate)
      - "std_err"  — full-model std_err non-finite
      - "vif"      — full-model VIF >= 10
      - "kappa"    — adjacent_zone feature, condition number >= 100
      - None       — all gates passed; `unfrozen` is True
    """

    feature_name: str
    coefficient_index: int
    gate_failed: str | None
    unfrozen: bool
    in_full_model_held: bool
    full_model_std_err: float | None
    full_model_vif: float | None
    is_adjacent_zone: bool
    kappa_at_decision: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "coefficient_index": self.coefficient_index,
            "gate_failed": self.gate_failed,
            "unfrozen": self.unfrozen,
            "in_full_model_held": self.in_full_model_held,
            "full_model_std_err": self.full_model_std_err,
            "full_model_vif": self.full_model_vif,
            "is_adjacent_zone": self.is_adjacent_zone,
            "kappa_at_decision": self.kappa_at_decision,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnlockEvaluationRecord:
        return cls(
            feature_name=data["feature_name"],
            coefficient_index=data["coefficient_index"],
            gate_failed=data.get("gate_failed"),
            unfrozen=data.get("unfrozen", False),
            in_full_model_held=data.get("in_full_model_held", False),
            full_model_std_err=data.get("full_model_std_err"),
            full_model_vif=data.get("full_model_vif"),
            is_adjacent_zone=data.get("is_adjacent_zone", False),
            kappa_at_decision=data.get("kappa_at_decision"),
        )


@dataclass(frozen=True, slots=True)
class LagTauDiagnostic:
    """Per-input τ search diagnostics surfaced into the tick log.

    Mirrors the dataclass of the same name in
    ``batch_learning.LagTauDiagnostic`` but is intentionally a separate
    type — snapshot.py is the public schema boundary and consumers shouldn't
    depend on internals of the WLS module.
    """

    tau: float
    tau_opt_raw: float
    bic_gain: float
    bic_threshold: float
    r2_improvement: float
    beta_at_tau: float
    n_eff: int
    accepted: bool
    reject_reason: str
    search_max_used: float = 0.0
    boundary_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "tau_opt_raw": self.tau_opt_raw,
            "bic_gain": self.bic_gain,
            "bic_threshold": self.bic_threshold,
            "r2_improvement": self.r2_improvement,
            "beta_at_tau": self.beta_at_tau,
            "n_eff": self.n_eff,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "search_max_used": self.search_max_used,
            "boundary_hit": self.boundary_hit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LagTauDiagnostic:
        return cls(
            tau=data["tau"],
            tau_opt_raw=data["tau_opt_raw"],
            bic_gain=data["bic_gain"],
            bic_threshold=data["bic_threshold"],
            r2_improvement=data["r2_improvement"],
            beta_at_tau=data["beta_at_tau"],
            n_eff=data["n_eff"],
            accepted=data["accepted"],
            reject_reason=data["reject_reason"],
            search_max_used=data.get("search_max_used", 0.0),
            boundary_hit=data.get("boundary_hit", False),
        )


@dataclass(frozen=True, slots=True)
class BatchLearningSnapshot:
    """Last batch WLS run results.

    Carries the full `BatchResult` field set (typed). `beta_std_err`,
    `blend_gains`, `feature_vif`, `beta_blended`, `plant_snapshot`,
    `detected_tau`, `detected_tau_diagnostics` are additive over the basic
    regression-fit fields — they tell offline analysis tools how identifiable
    each coefficient was on the last batch and what blended update was actually
    applied.
    """

    last_run_mono: float | None
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
    # Raw BatchResult fields preserved for offline analysis. Empty
    # when not produced by the underlying regression (e.g., legacy
    # batch results that pre-date these fields).
    beta_std_err: tuple[float, ...]
    beta_blended: tuple[float, ...]
    blend_gains: tuple[float, ...]
    feature_vif: tuple[float, ...]
    detected_tau: dict[str, float]
    plant_snapshot: dict[str, Any]
    # Per-feature outcome of the last unlock evaluation. Empty before
    # the first batch cycle has run, or when no frozen features remain.
    unlock_evaluation: tuple[UnlockEvaluationRecord, ...] = ()
    # Per-input τ search diagnostics: BIC gain vs threshold, R²
    # improvement, β at τ_opt, accept/reject reason. Sibling to
    # ``detected_tau`` (which the online EMA path reads); empty when
    # ``detect_lag=False`` or no inputs were searched. Default empty
    # so legacy bundles deserialize without bumping SCHEMA_VERSION.
    detected_tau_diagnostics: dict[str, LagTauDiagnostic] = field(default_factory=dict)

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
            "beta_std_err": list(self.beta_std_err),
            "beta_blended": list(self.beta_blended),
            "blend_gains": list(self.blend_gains),
            "feature_vif": list(self.feature_vif),
            "detected_tau": dict(self.detected_tau),
            "plant_snapshot": dict(self.plant_snapshot),
            "unlock_evaluation": [r.to_dict() for r in self.unlock_evaluation],
            "detected_tau_diagnostics": {
                k: v.to_dict() for k, v in self.detected_tau_diagnostics.items()
            },
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
            beta_std_err=tuple(data.get("beta_std_err", ())),
            beta_blended=tuple(data.get("beta_blended", ())),
            blend_gains=tuple(data.get("blend_gains", ())),
            feature_vif=tuple(data.get("feature_vif", ())),
            detected_tau=dict(data.get("detected_tau", {})),
            plant_snapshot=dict(data.get("plant_snapshot", {})),
            unlock_evaluation=tuple(
                UnlockEvaluationRecord.from_dict(r)
                for r in data.get("unlock_evaluation", [])
            ),
            detected_tau_diagnostics={
                k: LagTauDiagnostic.from_dict(v)
                for k, v in data.get("detected_tau_diagnostics", {}).items()
            },
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


# ── State-machine snapshots ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Alert:
    """One health-check finding.

    Produced by `pi/health_checks.py` check functions; aggregated into
    `HealthSnapshot.alerts`. The legacy `get_health_status` wire format
    splits these into parallel `alerts` (messages) and `reasons` (codes)
    lists; the typed snapshot keeps them paired.
    """

    message: str
    code: str
    severity: str  # "Warning" | "Critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "code": self.code,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Alert:
        return cls(
            message=data["message"],
            code=data["code"],
            severity=data["severity"],
        )


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Health state-machine output.

    Carries only the state-machine result. Diagnostic fields exposed via
    the health sensor's extra_state_attributes (pi_integral, hp_setpoint,
    rls_intercept, etc.) are read from the broader TickOutput by the
    legacy `get_health_status` transformer — no duplication here.

    `state == "Disabled"` is reserved for `pi_enabled=False`. `OK` means
    no warning- or critical-severity alerts. `Warning`/`Critical` reflect
    the highest severity across `alerts`.
    """

    state: str  # "OK" | "Warning" | "Critical" | "Disabled"
    alerts: tuple[Alert, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "alerts": [a.to_dict() for a in self.alerts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthSnapshot:
        return cls(
            state=data["state"],
            alerts=tuple(Alert.from_dict(a) for a in data["alerts"]),
        )


@dataclass(frozen=True, slots=True)
class LearningSnapshot:
    """Learning state-machine output.

    `state` reflects how many of the auto-frozen-at-init features have
    been unfrozen by batch-WLS evidence (held_features / VIF / std_err).

    `observation_count` is the SUM of the heat and cool observation
    BUFFER lengths (real-time, grows per tick). NOT to be confused with
    `RLSModelSnapshot.{heat,cool}_observation_count` which post-pre45
    reflects the most recent batch's `n_eligible` (lags by up to ~12h).

    `condition_number` is the most recently computed κ for the active
    buffer; `None` until the multicollinearity gate has been evaluated
    against enough data.
    """

    state: str  # "Observing" | "Learning" | "Optimizing" | "Optimized"
    frozen_features: tuple[str, ...]
    active_features: tuple[str, ...]
    observation_count: int
    condition_number: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "frozen_features": list(self.frozen_features),
            "active_features": list(self.active_features),
            "observation_count": self.observation_count,
            "condition_number": self.condition_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningSnapshot:
        return cls(
            state=data["state"],
            frozen_features=tuple(data["frozen_features"]),
            active_features=tuple(data["active_features"]),
            observation_count=data["observation_count"],
            condition_number=data["condition_number"],
        )


@dataclass(frozen=True, slots=True)
class GreyboxSnapshot:
    """Grey-box state-machine output.

    Carries only the state-machine label and an optional reason for
    `Failed`/`Learning` cases. The full fit details (c0, ua_c, k_c,
    alpha_c, residual_rms, etc.) live in `TickOutput.greybox_observer`
    (opaque dict from `GreyboxResult.as_dict()`); bridge details live in
    `TickOutput.greybox_bridge`. The legacy `get_greybox_state`
    transformer assembles the wire format from both sources.
    """

    state: str  # "Failed" | "Learning" | "Adequate" | "Good" | "Degraded"
    reason: str | None  # "scipy unavailable" | "buffer empty" | None

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GreyboxSnapshot:
        return cls(state=data["state"], reason=data["reason"])


@dataclass(frozen=True, slots=True)
class LearningSuppressionSnapshot:
    """Effective FF-learning suppression state.

    `effective_suppressed` reflects the OR of manual suppression
    (`_manual_ff_suppress`) and disturbance-driven suppression
    (`_disturbance_suppress_active`). The manual flag and reason live in
    `RLSModelSnapshot` already and are not duplicated here.
    """

    effective_suppressed: bool
    active_suppressors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_suppressed": self.effective_suppressed,
            "active_suppressors": list(self.active_suppressors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningSuppressionSnapshot:
        return cls(
            effective_suppressed=data["effective_suppressed"],
            active_suppressors=tuple(data["active_suppressors"]),
        )


# ── Per-tick observation context + events ──────────────────────────────


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """What happened with this tick's observation, if one was recorded.

    None when this tick didn't produce an observation (HP off, sensor
    unavailable, etc.).

    `admitted` reflects actual buffer admission for the WLS heat/cool
    buffer (corrected from earlier semantics where it meant "we tried
    to call add()"). `evicted_timestamp` and `min_incumbent_score`
    are populated when the WLS buffer was at capacity at decision time.
    `rejection_reason` is set on rejection (e.g. "leverage_rejected"
    when the active policy declined; "no_outdoor_temp" upstream).

    `gb_*` fields mirror the same observability for the grey-box
    buffer, which has independent admission rules (admits HP-off
    observations, rejects only `outdoor_temp_c=None`).
    """

    admitted: bool
    clamped: bool
    clamped_reason: str
    score: float | None
    mode: str  # "heat" or "cool"
    raw_readings: dict[str, float]  # entity_id → value at this tick
    feature_vector: tuple[float, ...]  # x vector that fed RLS
    evicted_timestamp: float | None = None
    min_incumbent_score: float | None = None
    rejection_reason: str | None = None
    gb_admitted: bool | None = None
    gb_score: float | None = None
    gb_evicted_timestamp: float | None = None
    gb_min_incumbent_score: float | None = None
    gb_rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "clamped": self.clamped,
            "clamped_reason": self.clamped_reason,
            "score": self.score,
            "mode": self.mode,
            "raw_readings": dict(self.raw_readings),
            "feature_vector": list(self.feature_vector),
            "evicted_timestamp": self.evicted_timestamp,
            "min_incumbent_score": self.min_incumbent_score,
            "rejection_reason": self.rejection_reason,
            "gb_admitted": self.gb_admitted,
            "gb_score": self.gb_score,
            "gb_evicted_timestamp": self.gb_evicted_timestamp,
            "gb_min_incumbent_score": self.gb_min_incumbent_score,
            "gb_rejection_reason": self.gb_rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationContext:
        return cls(
            admitted=data["admitted"],
            clamped=data["clamped"],
            clamped_reason=data["clamped_reason"],
            score=data["score"],
            mode=data["mode"],
            raw_readings=dict(data["raw_readings"]),
            feature_vector=tuple(data["feature_vector"]),
            evicted_timestamp=data.get("evicted_timestamp"),
            min_incumbent_score=data.get("min_incumbent_score"),
            rejection_reason=data.get("rejection_reason"),
            gb_admitted=data.get("gb_admitted"),
            gb_score=data.get("gb_score"),
            gb_evicted_timestamp=data.get("gb_evicted_timestamp"),
            gb_min_incumbent_score=data.get("gb_min_incumbent_score"),
            gb_rejection_reason=data.get("gb_rejection_reason"),
        )


class TickEventKind(StrEnum):
    """Typed kinds for `TickEvent.kind`. Stable wire format (string values)."""

    BATCH_RUN = "batch_run"
    ANOMALY_DETECTED = "anomaly_detected"
    MODE_CHANGE = "mode_change"
    SETPOINT_CHANGE_USER = "setpoint_change_user"
    MATURITY_GATE = "maturity_gate"
    LEARNING_SUPPRESSION_CHANGE = "learning_suppression_change"
    AUTO_PERTURB_STATE = "auto_perturb_state"
    BOUNDARY_UPDATE = "boundary_update"
    CONTROLLER_RELOAD = "controller_reload"
    BUFFER_RESET = "buffer_reset"


# Per-kind payload dataclasses. Each is frozen+slotted; `to_dict()` keeps
# only JSON-compatible field types so events round-trip through the
# event log cleanly.


@dataclass(frozen=True, slots=True)
class BatchRunPayload:
    """Emitted when a batch WLS run completes."""
    n_eligible: int
    residual_rms: float
    recommend_update: bool
    max_coeff_change_pct: float
    n_outliers_excluded: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_eligible": self.n_eligible,
            "residual_rms": self.residual_rms,
            "recommend_update": self.recommend_update,
            "max_coeff_change_pct": self.max_coeff_change_pct,
            "n_outliers_excluded": self.n_outliers_excluded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchRunPayload:
        return cls(
            n_eligible=data["n_eligible"],
            residual_rms=data["residual_rms"],
            recommend_update=data["recommend_update"],
            max_coeff_change_pct=data["max_coeff_change_pct"],
            n_outliers_excluded=data["n_outliers_excluded"],
        )


@dataclass(frozen=True, slots=True)
class AnomalyDetectedPayload:
    """Emitted when a CUSUM anomaly event closes (alarm threshold crossed)."""
    mode: str  # "heat" | "cool"
    mean_residual: float
    peak_cusum: float
    tick_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mean_residual": self.mean_residual,
            "peak_cusum": self.peak_cusum,
            "tick_count": self.tick_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyDetectedPayload:
        return cls(
            mode=data["mode"],
            mean_residual=data["mean_residual"],
            peak_cusum=data["peak_cusum"],
            tick_count=data["tick_count"],
        )


@dataclass(frozen=True, slots=True)
class ModeChangePayload:
    """Emitted when HVAC mode transitions."""
    from_mode: str
    to_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {"from_mode": self.from_mode, "to_mode": self.to_mode}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModeChangePayload:
        return cls(from_mode=data["from_mode"], to_mode=data["to_mode"])


@dataclass(frozen=True, slots=True)
class SetpointChangeUserPayload:
    """Emitted when the user changes the setpoint via service / UI."""
    from_setpoint: float | None
    to_setpoint: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_setpoint": self.from_setpoint,
            "to_setpoint": self.to_setpoint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetpointChangeUserPayload:
        return cls(
            from_setpoint=data["from_setpoint"],
            to_setpoint=data["to_setpoint"],
        )


@dataclass(frozen=True, slots=True)
class MaturityGatePayload:
    """Emitted when a plant-ID parameter graduates from seed to estimate."""
    parameter: str  # "tau_fast" | "tau_slow" | "k" | "theta"
    source_before: str  # "seed" | "estimate"
    source_after: str
    value: float
    observations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "source_before": self.source_before,
            "source_after": self.source_after,
            "value": self.value,
            "observations": self.observations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaturityGatePayload:
        return cls(
            parameter=data["parameter"],
            source_before=data["source_before"],
            source_after=data["source_after"],
            value=data["value"],
            observations=data["observations"],
        )


@dataclass(frozen=True, slots=True)
class LearningSuppressionChangePayload:
    """Emitted when effective FF-learning suppression toggles."""
    was_suppressed: bool
    is_suppressed: bool
    active_suppressors: tuple[str, ...]
    manual: bool  # True if change was user-initiated via service

    def to_dict(self) -> dict[str, Any]:
        return {
            "was_suppressed": self.was_suppressed,
            "is_suppressed": self.is_suppressed,
            "active_suppressors": list(self.active_suppressors),
            "manual": self.manual,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningSuppressionChangePayload:
        return cls(
            was_suppressed=data["was_suppressed"],
            is_suppressed=data["is_suppressed"],
            active_suppressors=tuple(data["active_suppressors"]),
            manual=data["manual"],
        )


@dataclass(frozen=True, slots=True)
class AutoPerturbStatePayload:
    """Emitted when the auto-perturbation FSM transitions."""
    from_state: str
    to_state: str
    cycles_completed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "cycles_completed": self.cycles_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AutoPerturbStatePayload:
        return cls(
            from_state=data["from_state"],
            to_state=data["to_state"],
            cycles_completed=data["cycles_completed"],
        )


@dataclass(frozen=True, slots=True)
class BoundaryUpdatePayload:
    """Emitted when the boundary estimator's posterior shifts notably."""
    posterior_mean_before: float
    posterior_mean_after: float
    posterior_std: float
    n_observations: int
    confident: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "posterior_mean_before": self.posterior_mean_before,
            "posterior_mean_after": self.posterior_mean_after,
            "posterior_std": self.posterior_std,
            "n_observations": self.n_observations,
            "confident": self.confident,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundaryUpdatePayload:
        return cls(
            posterior_mean_before=data["posterior_mean_before"],
            posterior_mean_after=data["posterior_mean_after"],
            posterior_std=data["posterior_std"],
            n_observations=data["n_observations"],
            confident=data["confident"],
        )


@dataclass(frozen=True, slots=True)
class ControllerReloadPayload:
    """Emitted on the first published tick after PI controller __init__ runs.

    Captures why the controller is starting fresh, which lets bundle
    readers distinguish "reset boundary" from "anomaly" when they see
    PI state discontinuities (integral=0, observation_count=0,
    ff_confidence reset, etc.).

    `reason` values: "ha_start" | "integration_reload" | "options_change" |
    "config_change" | "unknown". Strings (not StrEnum) because the
    distinction set may grow without a wire-format bump.
    """
    reason: str
    restored_from_storage: bool
    prior_run_age_s: float | None  # None if no prior wallclock available

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "restored_from_storage": self.restored_from_storage,
            "prior_run_age_s": self.prior_run_age_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControllerReloadPayload:
        return cls(
            reason=data["reason"],
            restored_from_storage=data["restored_from_storage"],
            prior_run_age_s=data["prior_run_age_s"],
        )


@dataclass(frozen=True, slots=True)
class BufferResetPayload:
    """Emitted when an observation buffer is cleared.

    Surfaces buffer wipes (manual service call, schema migration,
    model_inputs change, rls reset cascading, etc.) that previously
    only left a `state-discontinuity` shape in the data with no
    explicit signal. Bundle readers rely on this to distinguish
    "buffer policy / curation worked as expected" from "we erased
    history and the next batch fits a tiny cohort."

    Fields:
      buffer: which buffer was cleared. "wls_heat" | "wls_cool" |
          "greybox" | "all_wls" (heat+cool together).
      reason: why the clear happened. "service_call" |
          "schema_migration" | "model_inputs_changed" |
          "rls_reset_cascade" | "manual" | "other".
      before_count: buffer size immediately before the clear, or None
          if not measured at the call site.
    """

    buffer: str
    reason: str
    before_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer,
            "reason": self.reason,
            "before_count": self.before_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BufferResetPayload:
        return cls(
            buffer=data["buffer"],
            reason=data["reason"],
            before_count=data.get("before_count"),
        )


# Tagged-union of payload types. New kinds: add to TickEventKind, define
# their payload dataclass, and append the type to this union and to the
# `_PAYLOAD_BY_KIND` dispatch in `TickEvent.from_dict`.
TickEventPayload = (
    BatchRunPayload
    | AnomalyDetectedPayload
    | ModeChangePayload
    | SetpointChangeUserPayload
    | MaturityGatePayload
    | LearningSuppressionChangePayload
    | AutoPerturbStatePayload
    | BoundaryUpdatePayload
    | ControllerReloadPayload
    | BufferResetPayload
)


_PAYLOAD_BY_KIND: dict[TickEventKind, type[TickEventPayload]] = {
    TickEventKind.BATCH_RUN: BatchRunPayload,
    TickEventKind.ANOMALY_DETECTED: AnomalyDetectedPayload,
    TickEventKind.MODE_CHANGE: ModeChangePayload,
    TickEventKind.SETPOINT_CHANGE_USER: SetpointChangeUserPayload,
    TickEventKind.MATURITY_GATE: MaturityGatePayload,
    TickEventKind.LEARNING_SUPPRESSION_CHANGE: LearningSuppressionChangePayload,
    TickEventKind.AUTO_PERTURB_STATE: AutoPerturbStatePayload,
    TickEventKind.BOUNDARY_UPDATE: BoundaryUpdatePayload,
    TickEventKind.CONTROLLER_RELOAD: ControllerReloadPayload,
    TickEventKind.BUFFER_RESET: BufferResetPayload,
}


@dataclass(frozen=True, slots=True)
class TickEvent:
    """A discrete event that fired during this tick.

    Tagged-union: `kind` selects the type of `payload`. Roundtrip-clean
    through the event log via `to_dict()` / `from_dict()`.
    """

    kind: TickEventKind
    payload: TickEventPayload

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "payload": self.payload.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TickEvent:
        kind = TickEventKind(data["kind"])
        payload_cls = _PAYLOAD_BY_KIND[kind]
        return cls(kind=kind, payload=payload_cls.from_dict(data["payload"]))


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

    SCHEMA_VERSION: ClassVar[int] = 2

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

    # State-machine snapshots (Stage 5c) — typed views over the four
    # legacy sensor-feeding getters.
    health: HealthSnapshot
    learning: LearningSnapshot
    greybox: GreyboxSnapshot
    learning_suppression: LearningSuppressionSnapshot

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
    # Over-temperature regime gate state — True when the gate forced HP
    # to its idle setpoint and froze the integrator on this tick.
    # Visible in debug bundles for post-deployment verification.
    overtemp_regime: bool = False
    # Stable-conditions combined-bias EMA used as bumpless-transfer target
    # at regime exit.  None until enough stable observations have populated
    # it.  Visible in debug bundles for tuning the bumpless behavior.
    stable_combined_bias_ema: float | None = None

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
            "_health": self.health.to_dict(),
            "_learning": self.learning.to_dict(),
            "_greybox": self.greybox.to_dict(),
            "_learning_suppression": self.learning_suppression.to_dict(),
        }
        if self.observation is not None:
            out["_observation"] = self.observation.to_dict()
        if self.events:
            out["_events"] = [e.to_dict() for e in self.events]
        out["_overtemp_regime"] = self.overtemp_regime
        out["_stable_combined_bias_ema"] = self.stable_combined_bias_ema
        return out

    @staticmethod
    def _migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
        """Rewrite v1 records into v2 shape.

        v1 used `leverage_score` / `min_incumbent_leverage` /
        `gb_leverage_score` / `gb_min_incumbent_leverage` on the
        `_observation` sub-dict; v2 renamed all four to `score` /
        `min_incumbent_score` / `gb_score` / `gb_min_incumbent_score`.
        Kept indefinitely — v1 daily JSONL files exist in user installs
        on disk from before the rename.

        Returns a shallow copy with the migrated `_observation` block;
        leaves the input untouched.
        """
        if "_observation" not in data:
            return {**data, "_schema_version": 2}
        obs = dict(data["_observation"])
        for old, new in (
            ("leverage_score", "score"),
            ("min_incumbent_leverage", "min_incumbent_score"),
            ("gb_leverage_score", "gb_score"),
            ("gb_min_incumbent_leverage", "gb_min_incumbent_score"),
        ):
            if new not in obs and old in obs:
                obs[new] = obs.pop(old)
        return {**data, "_observation": obs, "_schema_version": 2}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TickOutput:
        """Reconstruct a TickOutput from its serialized form.

        Migrates older schema versions forward in-place; raises
        ValueError on unknown versions.
        """
        version = data.get("_schema_version")
        if version == 1:
            data = cls._migrate_v1_to_v2(data)
            version = 2
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
            health=HealthSnapshot.from_dict(data["_health"]),
            learning=LearningSnapshot.from_dict(data["_learning"]),
            greybox=GreyboxSnapshot.from_dict(data["_greybox"]),
            learning_suppression=LearningSuppressionSnapshot.from_dict(
                data["_learning_suppression"]
            ),
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
            overtemp_regime=data.get("_overtemp_regime", False),
            stable_combined_bias_ema=data.get("_stable_combined_bias_ema"),
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
                tick_fallback=0.0, outdoor_temp_sensor=None,
                model_inputs=[], ff_enabled=False, batch_wls_enabled=False,
                plant_id_enabled=False,
            ),
            rls_model=RLSModelSnapshot(
                heat_seeds={}, cool_seeds={},
                heat_uncertainty={}, heat_observation_count=0,
                cool_observation_count=0, learning_suppressed=False,
                manual_suppress_reason="",
                last_residual=None, last_gain_vector=None,
                frozen_mask_heat=(), frozen_mask_cool=(),
                cusum_pos=0.0, cusum_neg=0.0,
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
            health=HealthSnapshot(state="Disabled", alerts=()),
            learning=LearningSnapshot(
                state="Observing", frozen_features=(), active_features=(),
                observation_count=0, condition_number=None,
            ),
            greybox=GreyboxSnapshot(state="Failed", reason="not initialized"),
            learning_suppression=LearningSuppressionSnapshot(
                effective_suppressed=False, active_suppressors=(),
            ),
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


