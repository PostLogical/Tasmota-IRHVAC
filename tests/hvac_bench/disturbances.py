"""Disturbance library for benchmark scenarios.

Disturbances are events that affect the thermal model but are NOT in
the controller's model inputs. They test the controller's ability to
compensate for unknown perturbations using integral action alone.

All timing is wall-clock (minutes) so disturbances behave identically
across bench cadences.  The thermal model passes
``minute = tick * dt_minutes`` to ``intensity()`` and ``is_active()``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Disturbance:
    """A time-limited perturbation to the thermal model.

    Attributes:
        name: Human-readable label.
        heat_gain_c_per_min: Additional heat input (positive = warming).
        tau_factor: Multiplier on the room's time constant (< 1.0 = leakier).
            Applied only during the disturbance window.
        start_minute: Wall-clock minute when disturbance begins.
        duration_minutes: How many minutes the disturbance lasts.
        ramp_minutes: Minutes to ramp up/down (0 = instant step).
    """
    name: str
    heat_gain_c_per_min: float = 0.0
    tau_factor: float = 1.0
    start_minute: float = 0.0
    duration_minutes: float = 15.0
    ramp_minutes: float = 0.0

    def is_active(self, minute: float) -> bool:
        """Whether the disturbance is active at this wall-clock minute."""
        return self.start_minute <= minute < self.start_minute + self.duration_minutes

    def intensity(self, minute: float) -> float:
        """Disturbance intensity at this minute (0-1, with ramp)."""
        if not self.is_active(minute):
            return 0.0
        minutes_in = minute - self.start_minute
        minutes_remaining = (self.start_minute + self.duration_minutes) - minute
        if self.ramp_minutes > 0:
            ramp_up = min(minutes_in / self.ramp_minutes, 1.0)
            ramp_down = min(minutes_remaining / self.ramp_minutes, 1.0)
            return min(ramp_up, ramp_down)
        return 1.0


# ── Standard disturbance library ──────────────────────────────────────────
#
# Defaults preserve the original 15-min-tick semantics: start_minute=150
# was former ``start_tick=10`` at 15-min cadence.

def oil_boiler(start_minute=150.0):
    """Oil boiler cycles on for 30 min. Sudden heat gain."""
    return Disturbance(
        name="oil_boiler",
        heat_gain_c_per_min=0.2,  # ~3°C over 15 min
        start_minute=start_minute,
        duration_minutes=30.0,
        ramp_minutes=0.0,
    )


def front_door_open(start_minute=150.0):
    """Front door open for 15 min. Tau drops 50%."""
    return Disturbance(
        name="front_door_open",
        tau_factor=0.5,
        start_minute=start_minute,
        duration_minutes=15.0,
    )


def garage_door_open(start_minute=150.0):
    """Garage door stays open for 60 min. Tau drops 30%."""
    return Disturbance(
        name="garage_door_open",
        tau_factor=0.7,
        start_minute=start_minute,
        duration_minutes=60.0,
    )


def cooking(start_minute=150.0):
    """Cooking in kitchen for 45 min. Moderate heat gain."""
    return Disturbance(
        name="cooking",
        heat_gain_c_per_min=0.07,  # ~1°C over 15 min
        start_minute=start_minute,
        duration_minutes=45.0,
        ramp_minutes=15.0,
    )


def party(start_minute=150.0):
    """10 people for 3 hours. Sustained heat gain."""
    return Disturbance(
        name="party",
        heat_gain_c_per_min=0.13,  # ~2°C over 15 min
        start_minute=start_minute,
        duration_minutes=180.0,
        ramp_minutes=30.0,
    )


def window_open_summer(start_minute=150.0):
    """Window open in summer for 30 min. Tau drops + warm air ingress."""
    return Disturbance(
        name="window_open_summer",
        tau_factor=0.4,
        heat_gain_c_per_min=0.13,
        start_minute=start_minute,
        duration_minutes=30.0,
        ramp_minutes=0.0,
    )
