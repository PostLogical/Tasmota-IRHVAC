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
    rls_heat_model: dict = dataclasses.field(default_factory=dict)
    rls_cool_model: dict = dataclasses.field(default_factory=dict)
    lag_filter_states: dict = dataclasses.field(default_factory=dict)
    heat_seeds_at_learn: list = dataclasses.field(default_factory=list)
    cool_seeds_at_learn: list = dataclasses.field(default_factory=list)
    ki_at_save: float = 0.0
    tau_estimate: float = 0.0
    observation_buffer: list = dataclasses.field(default_factory=list)
    drift_correction_signs: list = dataclasses.field(default_factory=list)
    last_batch_result: dict | None = None
    tuning_alert_counters: dict = dataclasses.field(default_factory=dict)

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
            "observation_buffer": self.observation_buffer,
            "drift_correction_signs": self.drift_correction_signs,
            "last_batch_result": self.last_batch_result,
            "tuning_alert_counters": self.tuning_alert_counters,
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
                observation_buffer=restored.get("observation_buffer", []),
                drift_correction_signs=restored.get("drift_correction_signs", []),
                last_batch_result=restored.get("last_batch_result"),
                tuning_alert_counters=restored.get("tuning_alert_counters", {}),
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            return None
