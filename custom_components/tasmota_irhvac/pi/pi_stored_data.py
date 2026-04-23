"""PI controller persistence via HomeAssistant's ExtraStoredData mechanism.

Stores RLS models, integral, desired_temp, and hp_setpoint so they survive
restarts without bloating the recorder DB on every state write.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Self

from homeassistant.helpers.restore_state import ExtraStoredData


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
    exclusion_count: int = 0
    auto_perturb_state: dict[str, Any] = dataclasses.field(default_factory=dict)  # AutoPerturbation.as_dict()
    manual_override_heat: list[bool | None] = dataclasses.field(default_factory=list)
    manual_override_cool: list[bool | None] = dataclasses.field(default_factory=list)
    greybox_buffer: list[dict[str, Any]] = dataclasses.field(default_factory=list)  # Observation.as_dict()

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
            "exclusion_count": self.exclusion_count,
            "auto_perturb_state": self.auto_perturb_state,
            "manual_override_heat": self.manual_override_heat,
            "manual_override_cool": self.manual_override_cool,
            "greybox_buffer": self.greybox_buffer,
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
                exclusion_count=int(restored.get("exclusion_count", 0)),
                auto_perturb_state=restored.get("auto_perturb_state", {}),
                manual_override_heat=restored.get("manual_override_heat", []),
                manual_override_cool=restored.get("manual_override_cool", []),
                greybox_buffer=restored.get("greybox_buffer", []),
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            return None
