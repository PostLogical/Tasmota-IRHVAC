#!/usr/bin/env python3
"""Compare old vs new PI tuning parameters.

Runs the same scenarios with old defaults (Kp=1.5, Ki=0.05, b=1.0, no hysteresis)
vs new defaults (Kp=0.8, Ki=0.02, b=0.5, hysteresis=1) side by side.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Import the base sim components
from pi_simulator import RoomModel, SimResult


def run_sim(room, pi, duration_hours=24.0, dt=60.0, events=None):
    """Run simulation with time-normalized PI ticks."""
    events = sorted(events or [], key=lambda e: e[0])
    event_idx = 0
    steps = int(duration_hours * 3600 / dt)
    pi_interval_steps = int(pi.interval / dt)
    result = SimResult()
    times = []

    for step in range(steps):
        t_seconds = step * dt
        t_hours = t_seconds / 3600.0

        while event_idx < len(events) and events[event_idx][0] <= t_hours:
            events[event_idx][1](room, pi)
            event_idx += 1

        if step % pi_interval_steps == 0:
            pi.tick(room.room_temp, room.outdoor_temp, dt_seconds=pi.interval)

        room.step(pi.hp_setpoint, pi.desired_temp, dt)

        times.append(t_hours)
        result.room_temp.append(room.room_temp)
        result.desired_temp.append(pi.desired_temp)
        result.hp_setpoint.append(pi.hp_setpoint)
        result.outdoor_temp.append(room.outdoor_temp)
        result.integral.append(pi.integral)
        result.ff_offset.append(pi.ff_offset)
        result.hp_output.append(room.hp_output)

    result.time_hours = np.array(times)
    return result


@dataclass
class PIController:
    """PI+FF controller with optional setpoint hysteresis."""

    kp: float = 0.8
    ki: float = 0.02
    deadband: float = 0.5
    setpoint_weight: float = 0.5
    min_temp: float = 16.0
    max_temp: float = 30.0
    ff_heat_slope: float = 0.3
    ff_heat_reference: float = 15.0
    interval: float = 900.0
    hysteresis: float = 1.0  # Only change setpoint if diff > hysteresis from last sent

    # State
    integral: float = 0.0
    hp_setpoint: float = 22.0
    ff_offset: float = 0.0
    desired_temp: float = 22.0
    settled_ticks: int = 0
    paused: bool = False
    ff_buckets: dict = field(default_factory=dict)
    _last_error: float = 0.0
    _tick_count: int = 0

    def tick(self, current_temp: float, outdoor_temp: Optional[float] = None, dt_seconds: float = 900.0) -> float:
        """Run one PI tick. Matches pi_controller.py algorithm."""
        if self.paused:
            return self.hp_setpoint

        self._tick_count += 1
        dt_factor = dt_seconds / self.interval  # Time normalization

        error = self.desired_temp - current_temp

        # Feedforward
        self.ff_offset = 0.0
        if outdoor_temp is not None:
            delta = max(0, self.ff_heat_reference - outdoor_temp)
            raw_ff = self.ff_heat_slope * delta
            ff_scale = max(0.0, min(1.0, 1.0 + error / self.deadband))
            self.ff_offset = raw_ff * ff_scale

        # Adaptive setpoint weight
        abs_error = abs(error)
        if abs_error > self.deadband * 4:
            effective_weight = 1.0
        elif abs_error > self.deadband:
            blend = (abs_error - self.deadband) / (self.deadband * 3)
            effective_weight = self.setpoint_weight + blend * (1.0 - self.setpoint_weight)
        else:
            effective_weight = self.setpoint_weight

        # Deadband
        in_deadband = abs_error < self.deadband
        if in_deadband:
            self.integral *= 0.9
            self.settled_ticks += 1
            p_term = 0.0
        else:
            self.settled_ticks = 0
            p_error = effective_weight * self.desired_temp - current_temp
            p_term = self.kp * p_error
            # Time-normalized trapezoidal integral
            avg_error = (error + self._last_error) / 2.0
            self.integral += avg_error * dt_factor

        self._last_error = error

        # Stale integral clamp
        if self.integral < 0 and error >= -self.deadband:
            self.integral = 0.0

        # Hard cap
        self.integral = max(-50.0, min(50.0, self.integral))

        i_term = self.ki * self.integral
        raw_setpoint = self.desired_temp + p_term + i_term + self.ff_offset
        clamped = max(self.min_temp, min(self.max_temp, raw_setpoint))

        # Back-calculation anti-windup
        if self.ki != 0:
            sat_err = clamped - raw_setpoint
            if abs(sat_err) > 0.01:
                self.integral += (1.0 / self.ki) * sat_err
                self.integral = max(-50.0, min(50.0, self.integral))

        # Midpoint-crossing hysteresis
        if self.hysteresis > 0:
            new_setpoint = self.hp_setpoint  # default: hold
            if clamped > self.hp_setpoint + 0.5:
                new_setpoint = round(clamped)
            elif clamped < self.hp_setpoint - 0.5:
                new_setpoint = round(clamped)
            self.hp_setpoint = int(max(self.min_temp, min(self.max_temp, new_setpoint)))
        else:
            self.hp_setpoint = round(max(self.min_temp, min(self.max_temp, clamped)))

        return self.hp_setpoint

    def set_desired(self, temp: float):
        self.desired_temp = temp
        self.integral = 0.0


def make_old_pi(**kwargs):
    """Old defaults (v0.4.2)."""
    defaults = dict(kp=1.5, ki=0.05, setpoint_weight=1.0, hysteresis=0.0)
    defaults.update(kwargs)
    return PIController(**defaults)


def make_new_pi(**kwargs):
    """New defaults (v0.5.0) with all improvements."""
    defaults = dict(kp=1.0, ki=0.02, setpoint_weight=0.5, hysteresis=1.0)
    defaults.update(kwargs)
    return PIController(**defaults)


def comparison_plot(results: dict, title: str, filename: str):
    """Plot old vs new comparison."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    colors = {"Old (Kp=1.5, b=1.0)": "tab:red", "New (Kp=1.0, b=0.5, hyst)": "tab:blue"}

    for label, r in results.items():
        c = colors.get(label, "gray")
        t = r.time_hours
        axes[0].plot(t, r.room_temp, color=c, label=label, linewidth=1.5)
        axes[1].plot(t, r.hp_setpoint, color=c, label=label, linewidth=1.5)
        axes[2].plot(t, r.integral, color=c, label=label, linewidth=1.5)
        axes[3].plot(t, [w/1000 for w in r.hp_output], color=c, label=label, linewidth=1)

    axes[0].axhline(r.desired_temp[0], color="green", linestyle="--", alpha=0.5, label="Target")
    axes[0].set_ylabel("Room Temp (°C)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("HP Setpoint (°C)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Integral")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("HP Power (kW)")
    axes[3].set_xlabel("Time (hours)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def compare_steady_state():
    """Compare steady state behavior."""
    print("Compare 1: Steady state (outdoor 0°C, target 22°C)")
    results = {}
    for label, make_fn in [("Old (Kp=1.5, b=1.0)", make_old_pi), ("New (Kp=1.0, b=0.5, hyst)", make_new_pi)]:
        # Realistic room: higher thermal mass, slower HP
        room = RoomModel(room_temp=22.0, outdoor_temp=0.0, C_room=2_000_000, hp_response_time=600)
        pi = make_fn(desired_temp=22.0, hp_setpoint=24.0)
        run_sim(room, pi, duration_hours=4)  # warm up
        results[label] = run_sim(room, pi, duration_hours=12)

    comparison_plot(results, "Steady State Comparison\n(Outdoor 0°C, Target 22°C, 12h)", "cmp_steady_state.png")

    for label, r in results.items():
        changes = sum(1 for i in range(1, len(r.hp_setpoint)) if r.hp_setpoint[i] != r.hp_setpoint[i-1])
        temp_range = max(r.room_temp) - min(r.room_temp)
        print(f"  {label}: setpoint changes={changes}, temp range={temp_range:.2f}°C")


def compare_cold_start():
    """Compare cold start response."""
    print("Compare 2: Cold start (15°C → 22°C, outdoor 0°C)")
    results = {}
    for label, make_fn in [("Old (Kp=1.5, b=1.0)", make_old_pi), ("New (Kp=1.0, b=0.5, hyst)", make_new_pi)]:
        room = RoomModel(room_temp=15.0, outdoor_temp=0.0)
        pi = make_fn(desired_temp=22.0, hp_setpoint=22.0)
        results[label] = run_sim(room, pi, duration_hours=8)

    comparison_plot(results, "Cold Start Comparison\n(15°C → 22°C, Outdoor 0°C)", "cmp_cold_start.png")

    for label, r in results.items():
        # Time to reach within 0.5°C of target
        target = 22.0
        settled = None
        for i, temp in enumerate(r.room_temp):
            if abs(temp - target) < 0.5:
                settled = r.time_hours[i]
                break
        if settled is not None:
            print(f"  {label}: settled in {settled:.1f}h, final={r.room_temp[-1]:.2f}°C")
        else:
            print(f"  {label}: not settled in 8h, final={r.room_temp[-1]:.2f}°C")


def compare_setpoint_change():
    """Compare setpoint step response."""
    print("Compare 3: Setpoint change 22→24°C at t=2h")
    results = {}
    for label, make_fn in [("Old (Kp=1.5, b=1.0)", make_old_pi), ("New (Kp=1.0, b=0.5, hyst)", make_new_pi)]:
        room = RoomModel(room_temp=22.0, outdoor_temp=0.0)
        pi = make_fn(desired_temp=22.0, hp_setpoint=24.0)
        run_sim(room, pi, duration_hours=4)
        events = [(2.0, lambda r, p: p.set_desired(24.0))]
        results[label] = run_sim(room, pi, duration_hours=10, events=events)

    comparison_plot(results, "Setpoint Change Comparison\n(22→24°C at t=2h, Outdoor 0°C)", "cmp_setpoint_change.png")


def compare_outdoor_drop():
    """Compare outdoor temp disturbance response."""
    print("Compare 4: Outdoor drop 5→-10°C at t=2h")
    results = {}
    for label, make_fn in [("Old (Kp=1.5, b=1.0)", make_old_pi), ("New (Kp=1.0, b=0.5, hyst)", make_new_pi)]:
        room = RoomModel(room_temp=22.0, outdoor_temp=5.0)
        pi = make_fn(desired_temp=22.0, hp_setpoint=23.0)
        run_sim(room, pi, duration_hours=4)
        events = [(2.0, lambda r, p: setattr(r, 'outdoor_temp', -10.0))]
        results[label] = run_sim(room, pi, duration_hours=12, events=events)

    comparison_plot(results, "Outdoor Temp Drop Comparison\n(5→-10°C at t=2h, Target 22°C)", "cmp_outdoor_drop.png")

    for label, r in results.items():
        min_temp = min(r.room_temp)
        print(f"  {label}: min room temp={min_temp:.2f}°C")


def compare_saturation_recovery():
    """Compare saturation recovery."""
    print("Compare 5: Saturation recovery (-20→5°C at t=6h)")
    results = {}
    for label, make_fn in [("Old (Kp=1.5, b=1.0)", make_old_pi), ("New (Kp=1.0, b=0.5, hyst)", make_new_pi)]:
        room = RoomModel(room_temp=18.0, outdoor_temp=-20.0)
        pi = make_fn(desired_temp=22.0, hp_setpoint=22.0)
        events = [(6.0, lambda r, p: setattr(r, 'outdoor_temp', 5.0))]
        results[label] = run_sim(room, pi, duration_hours=16, events=events)

    comparison_plot(results, "Saturation Recovery Comparison\n(Outdoor -20→5°C at t=6h, Target 22°C)", "cmp_saturation.png")

    for label, r in results.items():
        idx_6h = int(6 * 3600 / 60)
        print(f"  {label}: integral at warmup={r.integral[idx_6h]:.1f}")


def compare_ff():
    """Compare PI+FF old vs new on outdoor disturbance."""
    print("Compare 6: PI+FF old vs new (outdoor 5→-10°C)")
    results = {}
    for label, make_fn in [("Old (Kp=1.5, b=1.0)", make_old_pi), ("New (Kp=1.0, b=0.5, hyst)", make_new_pi)]:
        room = RoomModel(room_temp=22.0, outdoor_temp=5.0)
        pi = make_fn(desired_temp=22.0, hp_setpoint=23.0, ff_heat_slope=0.3)
        run_sim(room, pi, duration_hours=4)
        events = [(2.0, lambda r, p: setattr(r, 'outdoor_temp', -10.0))]
        results[label] = run_sim(room, pi, duration_hours=10, events=events)

    comparison_plot(results, "PI+FF Comparison\n(Outdoor 5→-10°C at t=2h, FF slope=0.3)", "cmp_ff.png")


if __name__ == "__main__":
    print("PI Tuning Comparison: Old vs New Defaults")
    print("=" * 60)
    print("Old: Kp=1.5, Ki=0.05, b=1.0, no hysteresis")
    print("New: Kp=0.8, Ki=0.02, b=0.5, hysteresis=1°C")
    print("=" * 60)
    print()

    compare_steady_state()
    print()
    compare_cold_start()
    print()
    compare_setpoint_change()
    print()
    compare_outdoor_drop()
    print()
    compare_saturation_recovery()
    print()
    compare_ff()
    print()
    print("=" * 60)
    print(f"All comparison plots saved to {PLOTS_DIR}/")
