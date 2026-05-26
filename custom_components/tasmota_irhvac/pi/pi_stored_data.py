"""PI controller persistence via HomeAssistant's ExtraStoredData mechanism.

Stores RLS models, integral, desired_temp, and hp_setpoint so they survive
restarts without bloating the recorder DB on every state write.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Self

from homeassistant.helpers.restore_state import ExtraStoredData

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class PIExtraStoredData(ExtraStoredData):
    """PI controller data persisted via RestoreEntity's ExtraStoredData mechanism.

    Stores RLS models, integral, desired_temp, and hp_setpoint so they survive
    restarts without bloating the recorder DB on every state write.

    Fields typed dict[str, Any] are opaque serialized blobs from subsystems
    (RLS, plant ID, auto-perturbation).  The stored data module doesn't parse
    them — it passes them through to the subsystem's restore() method.
    """

    pi_integral: float
    desired_temp: float | None
    hp_setpoint: float | None
    integral_convergence: float = 0.0
    itae_accumulator: float = 0.0
    comfort_violation_hours: float = 0.0
    setpoint_changes: int = 0
    controllable_itae: float = 0.0
    uncontrollable_itae: float = 0.0
    controllable_cvh: float = 0.0
    uncontrollable_cvh: float = 0.0
    ff_load_fraction: float = 0.5
    rls_heat_model: dict[str, Any] = dataclasses.field(default_factory=dict)  # RLSModel.as_dict()
    rls_cool_model: dict[str, Any] = dataclasses.field(default_factory=dict)  # RLSModel.as_dict()
    lag_filter_states: dict[str, float] = dataclasses.field(default_factory=dict)
    heat_seeds_at_learn: list[float] = dataclasses.field(default_factory=list)
    cool_seeds_at_learn: list[float] = dataclasses.field(default_factory=list)
    ki_at_save: float = 0.0
    tau_estimate: float = 0.0  # backward compat (tau_fast)
    tau_observations: int = 0  # backward compat (tau_fast)
    plant_identifier_state: dict[str, Any] = dataclasses.field(default_factory=dict)  # PlantIdentifier.as_dict()
    observation_buffer_heat: list[dict[str, Any]] = dataclasses.field(default_factory=list)  # Observation.as_dict()
    observation_buffer_cool: list[dict[str, Any]] = dataclasses.field(default_factory=list)  # Observation.as_dict()
    drift_correction_signs: list[list[int]] = dataclasses.field(default_factory=list)
    last_batch_result: dict[str, Any] | None = None  # BatchResult via dataclasses.asdict()
    last_batch_wallclock: str = ""  # ISO-8601 wall-clock time of last batch run
    tuning_alert_counters: dict[str, int] = dataclasses.field(default_factory=dict)
    tuning_alert_snapshots: dict[str, float] = dataclasses.field(default_factory=dict)
    hp_deadband_estimate_heat: float = 0.5
    hp_deadband_estimate_cool: float = 0.5
    head_calibration_min_heat: float = -2.0
    head_calibration_max_heat: float = 2.0
    head_calibration_min_cool: float = -2.0
    head_calibration_max_cool: float = 2.0
    regime_probe_state: dict[str, Any] = dataclasses.field(default_factory=dict)  # RegimeProbe.as_dict()
    boundary_estimator_state: dict[str, Any] = dataclasses.field(default_factory=dict)  # BoundaryEstimator.as_dict()
    exclusion_count: int = 0
    auto_perturb_state: dict[str, Any] = dataclasses.field(default_factory=dict)  # AutoPerturbation.as_dict()
    manual_override_heat: list[bool | None] = dataclasses.field(default_factory=list)
    manual_override_cool: list[bool | None] = dataclasses.field(default_factory=list)
    greybox_buffer: list[dict[str, Any]] = dataclasses.field(default_factory=list)  # Observation.as_dict()
    batch_cycle_count: int = 0
    # Runtime subsystem toggles (persisted so they survive restarts)
    control_active: bool = True
    ff_enabled: bool = True
    batch_wls_enabled: bool = True
    plant_id_enabled: bool = True
    detected_lag_tau: dict[str, float] = dataclasses.field(default_factory=dict)  # input name → auto-detected EMA tau (seconds)
    detected_lag_tau_counts: dict[str, int] = dataclasses.field(default_factory=dict)  # input name → consistent detection count
    pi_event_log_enabled: bool = False  # opt-in: persistent JSONL event log under <config>/tasmota_irhvac/log/
    # ISO-8601 UTC wall-clock at which this snapshot was assembled. Used by
    # async_added_to_hass to compute prior_run_age_s for the CONTROLLER_RELOAD
    # event. Empty string on legacy stored data (pre-introduction).
    saved_at_wallclock: str = ""
    # CUSUM anomaly-detection state.  Persisted so a controller restart
    # doesn't erase accumulated drift evidence (cusum_pos / cusum_neg
    # would otherwise restart from 0, masking on-going anomalies for the
    # MIN_RESIDUALS_FOR_DETECTION ramp-up) or the cooldown timer (otherwise
    # a re-init within the 30-min cooldown re-emits the same anomaly).
    # ``cusum_cooldown_until_epoch`` is the absolute wall-clock epoch the
    # cooldown ends.  0.0 means no cooldown active.  Float (not ISO) keeps
    # save/restore symmetric for the naive datetimes used by the detector.
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    cusum_cooldown_until_epoch: float = 0.0
    # Rolling residual window for MAD scale estimate.  Persisted so
    # MIN_RESIDUALS_FOR_DETECTION ramp-up doesn't blank detection for
    # 10 ticks after every restart.
    cusum_residual_history: list[float] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "pi_integral": self.pi_integral,
            "desired_temp": self.desired_temp,
            "hp_setpoint": self.hp_setpoint,
            "integral_convergence": self.integral_convergence,
            "itae_accumulator": self.itae_accumulator,
            "comfort_violation_hours": self.comfort_violation_hours,
            "setpoint_changes": self.setpoint_changes,
            "controllable_itae": self.controllable_itae,
            "uncontrollable_itae": self.uncontrollable_itae,
            "controllable_cvh": self.controllable_cvh,
            "uncontrollable_cvh": self.uncontrollable_cvh,
            "ff_load_fraction": self.ff_load_fraction,
            "rls_heat_model": self.rls_heat_model,
            "rls_cool_model": self.rls_cool_model,
            "lag_filter_states": self.lag_filter_states,
            "heat_seeds_at_learn": self.heat_seeds_at_learn,
            "cool_seeds_at_learn": self.cool_seeds_at_learn,
            "ki_at_save": self.ki_at_save,
            "tau_estimate": self.tau_estimate,
            "tau_observations": self.tau_observations,
            "plant_identifier_state": self.plant_identifier_state,
            "observation_buffer_heat": self.observation_buffer_heat,
            "observation_buffer_cool": self.observation_buffer_cool,
            "drift_correction_signs": self.drift_correction_signs,
            "last_batch_result": self.last_batch_result,
            "last_batch_wallclock": self.last_batch_wallclock,
            "tuning_alert_counters": self.tuning_alert_counters,
            "tuning_alert_snapshots": self.tuning_alert_snapshots,
            "hp_deadband_estimate_heat": self.hp_deadband_estimate_heat,
            "hp_deadband_estimate_cool": self.hp_deadband_estimate_cool,
            "head_calibration_min_heat": self.head_calibration_min_heat,
            "head_calibration_max_heat": self.head_calibration_max_heat,
            "head_calibration_min_cool": self.head_calibration_min_cool,
            "head_calibration_max_cool": self.head_calibration_max_cool,
            "regime_probe_state": self.regime_probe_state,
            "boundary_estimator_state": self.boundary_estimator_state,
            "exclusion_count": self.exclusion_count,
            "auto_perturb_state": self.auto_perturb_state,
            "manual_override_heat": self.manual_override_heat,
            "manual_override_cool": self.manual_override_cool,
            "greybox_buffer": self.greybox_buffer,
            "batch_cycle_count": self.batch_cycle_count,
            "control_active": self.control_active,
            "ff_enabled": self.ff_enabled,
            "batch_wls_enabled": self.batch_wls_enabled,
            "plant_id_enabled": self.plant_id_enabled,
            "detected_lag_tau": self.detected_lag_tau,
            "detected_lag_tau_counts": self.detected_lag_tau_counts,
            "pi_event_log_enabled": self.pi_event_log_enabled,
            "saved_at_wallclock": self.saved_at_wallclock,
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "cusum_cooldown_until_epoch": self.cusum_cooldown_until_epoch,
            "cusum_residual_history": self.cusum_residual_history,
        }

    @classmethod
    def from_dict(cls, restored: dict[str, Any]) -> Self | None:
        """Deserialize from stored dict.

        Gracefully ignores legacy bucket fields (ff_heat_buckets, ff_cool_buckets,
        ff_bucket_observation_counts) from pre-removal stored data.
        """
        try:
            return cls(
                pi_integral=float(restored["pi_integral"]),
                desired_temp=restored.get("desired_temp"),
                hp_setpoint=restored.get("hp_setpoint"),
                integral_convergence=float(restored.get("integral_convergence", 0.0)),
                itae_accumulator=float(restored.get("itae_accumulator", 0.0)),
                comfort_violation_hours=float(restored.get("comfort_violation_hours", 0.0)),
                setpoint_changes=int(restored.get("setpoint_changes", 0)),
                controllable_itae=float(restored.get("controllable_itae", 0.0)),
                uncontrollable_itae=float(restored.get("uncontrollable_itae", 0.0)),
                controllable_cvh=float(restored.get("controllable_cvh", 0.0)),
                uncontrollable_cvh=float(restored.get("uncontrollable_cvh", 0.0)),
                ff_load_fraction=float(restored.get("ff_load_fraction", 0.5)),
                rls_heat_model=restored.get("rls_heat_model", {}),
                rls_cool_model=restored.get("rls_cool_model", {}),
                lag_filter_states=restored.get("lag_filter_states", {}),
                heat_seeds_at_learn=restored.get("heat_seeds_at_learn", []),
                cool_seeds_at_learn=restored.get("cool_seeds_at_learn", []),
                ki_at_save=float(restored.get("ki_at_save", 0.0)),
                tau_estimate=float(restored.get("tau_estimate", 0.0)),
                tau_observations=int(restored.get("tau_observations", 0)),
                plant_identifier_state=restored.get("plant_identifier_state", {}),
                observation_buffer_heat=restored.get("observation_buffer_heat", []),
                observation_buffer_cool=restored.get("observation_buffer_cool", []),
                drift_correction_signs=restored.get("drift_correction_signs", []),
                last_batch_result=restored.get("last_batch_result"),
                last_batch_wallclock=str(restored.get("last_batch_wallclock", "")),
                tuning_alert_counters=restored.get("tuning_alert_counters", {}),
                # Migration from pre39: extract snapshots from old combined dict.
                # Remove fallback once all installs have restarted on pre39+.
                tuning_alert_snapshots=restored.get("tuning_alert_snapshots", {
                    k: v for k, v in restored.get("tuning_alert_counters", {}).items()
                    if k.startswith("freeze_rms_")
                }),
                hp_deadband_estimate_heat=float(restored.get("hp_deadband_estimate_heat", 0.5)),
                hp_deadband_estimate_cool=float(restored.get("hp_deadband_estimate_cool", 0.5)),
                head_calibration_min_heat=float(restored.get("head_calibration_min_heat", -2.0)),
                head_calibration_max_heat=float(restored.get("head_calibration_max_heat", 2.0)),
                head_calibration_min_cool=float(restored.get("head_calibration_min_cool", -2.0)),
                head_calibration_max_cool=float(restored.get("head_calibration_max_cool", 2.0)),
                regime_probe_state=restored.get("regime_probe_state", {}),
                boundary_estimator_state=restored.get("boundary_estimator_state", {}),
                exclusion_count=int(restored.get("exclusion_count", 0)),
                auto_perturb_state=restored.get("auto_perturb_state", {}),
                manual_override_heat=restored.get("manual_override_heat", []),
                manual_override_cool=restored.get("manual_override_cool", []),
                greybox_buffer=restored.get("greybox_buffer", []),
                batch_cycle_count=int(restored.get("batch_cycle_count", 0)),
                control_active=bool(restored.get("control_active", True)),
                ff_enabled=bool(restored.get("ff_enabled", True)),
                batch_wls_enabled=bool(restored.get("batch_wls_enabled", True)),
                plant_id_enabled=bool(restored.get("plant_id_enabled", True)),
                detected_lag_tau=restored.get("detected_lag_tau", {}),
                detected_lag_tau_counts={
                    k: int(v) for k, v in restored.get("detected_lag_tau_counts", {}).items()
                },
                pi_event_log_enabled=bool(restored.get("pi_event_log_enabled", False)),
                saved_at_wallclock=str(restored.get("saved_at_wallclock", "")),
                cusum_pos=float(restored.get("cusum_pos", 0.0)),
                cusum_neg=float(restored.get("cusum_neg", 0.0)),
                cusum_cooldown_until_epoch=float(
                    restored.get("cusum_cooldown_until_epoch", 0.0),
                ),
                cusum_residual_history=[
                    float(r) for r in restored.get("cusum_residual_history", [])
                ],
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            # Returning None lets callers fall through to the next
            # restore source, but the bare-swallow used to hide the
            # failure entirely — including the case where the only
            # persisted-False boolean (pi_event_log_enabled) silently
            # flipped back to default. Log loudly so future restore
            # regressions are visible in HA logs and the failing
            # field/value is captured in the traceback.
            keys_preview = (
                sorted(restored.keys())[:20] if isinstance(restored, dict)
                else f"<not-a-dict: {type(restored).__name__}>"
            )
            _LOGGER.exception(
                "PIExtraStoredData.from_dict failed; callers will fall "
                "back to next restore source (auto-save / legacy "
                "attributes / defaults). Affected keys present in input: "
                "%s",
                keys_preview,
            )
            return None
