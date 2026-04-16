"""Performance metrics accumulator for PIController.

Pure computation — no Home Assistant dependencies.  Owns the running totals
that track controller quality (ITAE, comfort‐violation hours, FF load
fraction, etc.).  PIController calls the ``accumulate_*`` methods each tick
and reads the public attributes for persistence / diagnostics.
"""

from __future__ import annotations

from typing import Any


class PerformanceMetrics:
    """Running performance accumulators for PI controller diagnostics."""

    def __init__(self) -> None:
        # Integral convergence tracking (EMA of abs(integral) over ~24hr)
        self.integral_convergence: float = 0.0

        # Performance metrics (running totals, persisted via ExtraStoredData)
        self.itae_accumulator: float = 0.0
        self.itae_tick_count: int = 0
        self.comfort_violation_hours: float = 0.0
        self.setpoint_changes: int = 0
        self.controllable_itae: float = 0.0
        self.uncontrollable_itae: float = 0.0
        self.controllable_cvh: float = 0.0
        self.uncontrollable_cvh: float = 0.0

        # FF load fraction: EMA of |ff_offset| / (|ff_offset| + |ki*integral|).
        self.ff_load_fraction: float = 0.5

        # Batch model RMS: last residual_rms from batch WLS analysis.
        self.batch_model_rms: float | None = None

    # ── Tick accumulation ────────────────────────────────────────────

    def accumulate_convergence(self, pi_integral: float) -> None:
        """Update integral convergence EMA (~24hr time constant).

        Called every tick, unconditionally.
        """
        alpha = 0.01  # With 15-min ticks, 96 ticks/day → alpha ≈ 1/96
        self.integral_convergence = (
            (1.0 - alpha) * self.integral_convergence
            + alpha * abs(pi_integral)
        )

    def accumulate_tick(
        self,
        *,
        abs_error: float,
        dt_seconds: float,
        pi_deadband: float,
        is_heating: bool,
        is_cooling: bool,
        error: float,
        hp_setpoint: float,
        min_temp_c: float,
        max_temp_c: float,
    ) -> None:
        """Accumulate ITAE, CVH, and controllable/uncontrollable split.

        Called when not in tracking mode (HP is responsible for comfort).
        """
        self.itae_tick_count += 1
        effective_error = max(0.0, abs_error - pi_deadband)
        self.itae_accumulator += self.itae_tick_count * effective_error
        if abs_error > 1.0:
            self.comfort_violation_hours += dt_seconds / 3600.0

        # Controllable/uncontrollable split: "uncontrollable" when the HP
        # is clamped at its limit in the direction that would help.
        saturated_wrong_end = (
            (is_heating and error < 0 and hp_setpoint <= min_temp_c)
            or (is_cooling and error > 0 and hp_setpoint >= max_temp_c)
        )
        itae_increment = self.itae_tick_count * effective_error
        if saturated_wrong_end:
            self.uncontrollable_itae += itae_increment
            if abs_error > 1.0:
                self.uncontrollable_cvh += dt_seconds / 3600.0
        else:
            self.controllable_itae += itae_increment
            if abs_error > 1.0:
                self.controllable_cvh += dt_seconds / 3600.0

    def accumulate_ff_load(self, ki_integral: float, ff_offset: float) -> None:
        """Update FF load fraction EMA (~24hr time constant).

        Called every tick, unconditionally.
        """
        i_correction = abs(ki_integral)
        ff_mag = abs(ff_offset)
        total_effort = ff_mag + i_correction
        if total_effort > 0.1:  # avoid noise when both are near zero
            instant_ff_load = ff_mag / total_effort
            self.ff_load_fraction += 0.01 * (instant_ff_load - self.ff_load_fraction)

    def record_setpoint_change(self) -> None:
        """Increment setpoint change counter."""
        self.setpoint_changes += 1

    # ── Persistence ──────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Serialize all fields for ExtraStoredData persistence."""
        return {
            "integral_convergence": self.integral_convergence,
            "itae_accumulator": self.itae_accumulator,
            "itae_tick_count": self.itae_tick_count,
            "comfort_violation_hours": self.comfort_violation_hours,
            "setpoint_changes": self.setpoint_changes,
            "controllable_itae": self.controllable_itae,
            "uncontrollable_itae": self.uncontrollable_itae,
            "controllable_cvh": self.controllable_cvh,
            "uncontrollable_cvh": self.uncontrollable_cvh,
            "ff_load_fraction": self.ff_load_fraction,
            "batch_model_rms": self.batch_model_rms,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore fields from persisted dict."""
        self.integral_convergence = float(data.get("integral_convergence", 0.0))
        self.itae_accumulator = float(data.get("itae_accumulator", 0.0))
        self.itae_tick_count = int(data.get("itae_tick_count", 0))
        self.comfort_violation_hours = float(data.get("comfort_violation_hours", 0.0))
        self.setpoint_changes = int(data.get("setpoint_changes", 0))
        self.controllable_itae = float(data.get("controllable_itae", 0.0))
        self.uncontrollable_itae = float(data.get("uncontrollable_itae", 0.0))
        self.controllable_cvh = float(data.get("controllable_cvh", 0.0))
        self.uncontrollable_cvh = float(data.get("uncontrollable_cvh", 0.0))
        self.ff_load_fraction = float(data.get("ff_load_fraction", 0.5))
        self.batch_model_rms = data.get("batch_model_rms")
