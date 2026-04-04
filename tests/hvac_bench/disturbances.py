"""Disturbance library for benchmark scenarios.

Disturbances are events that affect the thermal model but are NOT in
the controller's model inputs. They test the controller's ability to
compensate for unknown perturbations using integral action alone.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Disturbance:
    """A time-limited perturbation to the thermal model.

    Attributes:
        name: Human-readable label.
        heat_gain_c_per_min: Additional heat input (positive = warming).
        tau_factor: Multiplier on the room's time constant (< 1.0 = leakier).
            Applied only during the disturbance window.
        start_tick: Tick when disturbance begins.
        duration_ticks: How many ticks the disturbance lasts.
        ramp_ticks: Ticks to ramp up/down (0 = instant step).
    """
    name: str
    heat_gain_c_per_min: float = 0.0
    tau_factor: float = 1.0
    start_tick: int = 0
    duration_ticks: int = 1
    ramp_ticks: int = 0

    def is_active(self, tick):
        """Whether the disturbance is active at this tick."""
        return self.start_tick <= tick < self.start_tick + self.duration_ticks

    def intensity(self, tick):
        """Disturbance intensity at this tick (0-1, with ramp)."""
        if not self.is_active(tick):
            return 0.0
        ticks_in = tick - self.start_tick
        ticks_remaining = (self.start_tick + self.duration_ticks) - tick
        if self.ramp_ticks > 0:
            ramp_up = min(ticks_in / self.ramp_ticks, 1.0)
            ramp_down = min(ticks_remaining / self.ramp_ticks, 1.0)
            return min(ramp_up, ramp_down)
        return 1.0


# ── Standard disturbance library ──────────────────────────────────────────

def oil_boiler(start_tick=10):
    """Oil boiler cycles on for 30 min (2 ticks). Sudden heat gain."""
    return Disturbance(
        name="oil_boiler",
        heat_gain_c_per_min=0.2,  # ~3°C over 15 min
        start_tick=start_tick,
        duration_ticks=2,
        ramp_ticks=0,
    )


def front_door_open(start_tick=10):
    """Front door open for 15 min (1 tick). Tau drops 50%."""
    return Disturbance(
        name="front_door_open",
        tau_factor=0.5,
        start_tick=start_tick,
        duration_ticks=1,
    )


def garage_door_open(start_tick=10):
    """Garage door stays open for 60 min (4 ticks). Tau drops 30%."""
    return Disturbance(
        name="garage_door_open",
        tau_factor=0.7,
        start_tick=start_tick,
        duration_ticks=4,
    )


def cooking(start_tick=10):
    """Cooking in kitchen for 45 min (3 ticks). Moderate heat gain."""
    return Disturbance(
        name="cooking",
        heat_gain_c_per_min=0.07,  # ~1°C over 15 min
        start_tick=start_tick,
        duration_ticks=3,
        ramp_ticks=1,
    )


def party(start_tick=10):
    """10 people for 3 hours (12 ticks). Sustained heat gain."""
    return Disturbance(
        name="party",
        heat_gain_c_per_min=0.13,  # ~2°C over 15 min
        start_tick=start_tick,
        duration_ticks=12,
        ramp_ticks=2,
    )


def window_open_summer(start_tick=10):
    """Window open in summer for 30 min. Tau drops + warm air ingress."""
    return Disturbance(
        name="window_open_summer",
        tau_factor=0.4,
        heat_gain_c_per_min=0.13,
        start_tick=start_tick,
        duration_ticks=2,
        ramp_ticks=0,
    )
