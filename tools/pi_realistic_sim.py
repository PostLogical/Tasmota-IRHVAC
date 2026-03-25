#!/usr/bin/env python3
"""Realistic PI simulation using actual house parameters from HA data.

Living Room parameters (from 10-day analysis):
  - Time constant: ~60 min
  - Settling time: ~81 min
  - Comfort band: 65.7-67.8°F (IQR 2.1°F)
  - Typical hold: 27.5 hours
  - Pellet stove: +2.3°F/hr

Bunkroom parameters:
  - Time constant: ~23 min
  - Settling time: ~30-40 min
  - Comfort band: 67.9-69.9°F (IQR 2.0°F)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Realistic Room Models (from HA data) ──────────────────────────────

def make_living_room(room_temp=19.3, outdoor_temp=0.0):
    """Living room: TC=60min, slow response, pellet stove accessible."""
    # TC = R * C. With TC=3600s and reasonable R:
    # R=0.006 °C/W, C=600000 J/°C → TC=3600s (60 min)
    return RoomModel(
        name="Living Room",
        room_temp=room_temp,
        outdoor_temp=outdoor_temp,
        R_wall=0.006,
        C_room=600_000,
        hp_max_power=3500,
        hp_response_time=300,
    )

def make_bunkroom(room_temp=20.4, outdoor_temp=0.0):
    """Bunkroom: TC=23min, fast response, runs warmer."""
    # TC=1380s → R=0.006, C=230000
    return RoomModel(
        name="Bunkroom",
        room_temp=room_temp,
        outdoor_temp=outdoor_temp,
        R_wall=0.006,
        C_room=230_000,
        hp_max_power=3500,
        hp_response_time=180,
    )


@dataclass
class RoomModel:
    name: str = "Room"
    R_wall: float = 0.006
    C_room: float = 600_000
    hp_max_power: float = 3500.0
    hp_min_power: float = 500.0
    hp_response_time: float = 300.0
    room_temp: float = 20.0
    outdoor_temp: float = 0.0
    hp_output: float = 0.0
    supplemental_heat: float = 0.0

    def step(self, hp_setpoint, desired_temp, dt=60.0):
        hp_target = self._hp_power(hp_setpoint)
        alpha = min(1.0, dt / self.hp_response_time)
        self.hp_output += alpha * (hp_target - self.hp_output)
        q_wall = (self.outdoor_temp - self.room_temp) / self.R_wall
        q_total = q_wall + self.hp_output + self.supplemental_heat
        self.room_temp += (q_total * dt) / self.C_room

    def _hp_power(self, hp_setpoint):
        diff = hp_setpoint - self.room_temp
        if diff > 0:
            return max(self.hp_min_power, self.hp_max_power * min(1.0, diff / 5.0))
        elif diff < -1:
            return 0.0
        else:
            return self.hp_min_power * max(0, diff + 1)


# ── PI Controller (matches v0.5.0 pi_controller.py) ──────────────────

@dataclass
class PIController:
    kp: float = 1.0
    ki: float = 0.02
    deadband: float = 0.5
    deadband_above: float = 0.5  # Asymmetric: tolerance above desired (heating)
    deadband_below: float = 0.5  # Asymmetric: tolerance below desired (heating)
    setpoint_weight: float = 0.5
    min_temp: float = 16.0
    max_temp: float = 30.0
    ff_heat_slope: float = 0.3
    ff_heat_reference: float = 15.0
    interval: float = 900.0

    # State
    integral: float = 0.0
    hp_setpoint: float = 22.0
    ff_offset: float = 0.0
    desired_temp: float = 22.0
    settled_ticks: int = 0
    _last_error: float = 0.0
    _tick_count: int = 0

    def tick(self, current_temp, outdoor_temp=None, dt_seconds=900.0):
        self._tick_count += 1
        dt_factor = dt_seconds / self.interval
        error = self.desired_temp - current_temp

        # FF
        self.ff_offset = 0.0
        if outdoor_temp is not None:
            delta = max(0, self.ff_heat_reference - outdoor_temp)
            raw_ff = self.ff_heat_slope * delta
            ff_scale = max(0.0, min(1.0, 1.0 + error / self.deadband))
            self.ff_offset = raw_ff * ff_scale

        # Adaptive setpoint weight
        abs_error = abs(error)
        if abs_error > self.deadband * 4:
            ew = 1.0
        elif abs_error > self.deadband:
            blend = (abs_error - self.deadband) / (self.deadband * 3)
            ew = self.setpoint_weight + blend * (1.0 - self.setpoint_weight)
        else:
            ew = self.setpoint_weight

        # Asymmetric deadband (heating mode)
        if error > 0:
            in_deadband = error < self.deadband_below
        else:
            in_deadband = abs(error) < self.deadband_above

        if in_deadband:
            self.integral *= 0.9
            self.settled_ticks += 1
            p_term = 0.0
        else:
            self.settled_ticks = 0
            p_error = ew * self.desired_temp - current_temp
            p_term = self.kp * p_error
            avg_error = (error + self._last_error) / 2.0
            self.integral += avg_error * dt_factor

        self._last_error = error

        if self.integral < 0 and error >= -self.deadband:
            self.integral = 0.0

        self.integral = max(-50.0, min(50.0, self.integral))

        i_term = self.ki * self.integral
        raw = self.desired_temp + p_term + i_term + self.ff_offset
        clamped = max(self.min_temp, min(self.max_temp, raw))

        if self.ki != 0:
            sat_err = clamped - raw
            if abs(sat_err) > 0.01:
                self.integral += (1.0 / self.ki) * sat_err
                self.integral = max(-50.0, min(50.0, self.integral))

        # Midpoint hysteresis
        new_sp = self.hp_setpoint
        if clamped > self.hp_setpoint + 0.5:
            new_sp = round(clamped)
        elif clamped < self.hp_setpoint - 0.5:
            new_sp = round(clamped)
        self.hp_setpoint = int(max(self.min_temp, min(self.max_temp, new_sp)))
        return self.hp_setpoint

    def set_desired(self, temp):
        self.desired_temp = temp
        self.integral = 0.0
        self._last_error = 0.0


# ── Simulation ────────────────────────────────────────────────────────

@dataclass
class SimResult:
    time_min: np.ndarray = field(default_factory=lambda: np.array([]))
    room_temp_f: list = field(default_factory=list)
    desired_temp_f: list = field(default_factory=list)
    hp_setpoint: list = field(default_factory=list)
    outdoor_temp_f: list = field(default_factory=list)
    integral: list = field(default_factory=list)
    ff_offset: list = field(default_factory=list)
    setpoint_changes: int = 0


def c_to_f(c):
    return c * 9/5 + 32

def f_to_c(f):
    return (f - 32) * 5/9


def run_sim(room, pi, duration_hours, dt=60.0, events=None):
    events = sorted(events or [], key=lambda e: e[0])
    event_idx = 0
    steps = int(duration_hours * 3600 / dt)
    pi_interval_steps = int(pi.interval / dt)
    result = SimResult()
    times = []
    prev_setpoint = pi.hp_setpoint

    for step in range(steps):
        t_sec = step * dt
        t_min = t_sec / 60.0
        t_hr = t_sec / 3600.0

        while event_idx < len(events) and events[event_idx][0] <= t_hr:
            events[event_idx][1](room, pi)
            event_idx += 1

        if step % pi_interval_steps == 0 and step > 0:
            pi.tick(room.room_temp, room.outdoor_temp, dt_seconds=pi.interval)
            if pi.hp_setpoint != prev_setpoint:
                result.setpoint_changes += 1
                prev_setpoint = pi.hp_setpoint

        room.step(pi.hp_setpoint, pi.desired_temp, dt)

        times.append(t_min)
        result.room_temp_f.append(c_to_f(room.room_temp))
        result.desired_temp_f.append(c_to_f(pi.desired_temp))
        result.hp_setpoint.append(pi.hp_setpoint)
        result.outdoor_temp_f.append(c_to_f(room.outdoor_temp))
        result.integral.append(pi.integral)
        result.ff_offset.append(pi.ff_offset)

    result.time_min = np.array(times)
    return result


def plot_result(result, title, filename, comfort_band=None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    t = result.time_min

    ax = axes[0]
    ax.plot(t, result.room_temp_f, "b-", label="Room Temp (°F)", linewidth=1.5)
    ax.plot(t, result.desired_temp_f, "r--", label="Desired (°F)", linewidth=1.5)
    ax.plot(t, result.outdoor_temp_f, "g:", label="Outdoor (°F)", linewidth=1, alpha=0.7)
    if comfort_band:
        desired_f = result.desired_temp_f[0]
        ax.axhspan(desired_f - comfort_band[0], desired_f + comfort_band[1],
                    alpha=0.1, color="green", label=f"Comfort band (-{comfort_band[0]}/+{comfort_band[1]}°F)")
    ax.set_ylabel("Temperature (°F)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, result.hp_setpoint, "m-", label="HP Setpoint (°C)", linewidth=1.5)
    ax.set_ylabel("HP Setpoint (°C)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, result.integral, "c-", label="Integral", linewidth=1)
    ax.plot(t, result.ff_offset, "orange", label="FF Offset", linewidth=1)
    ax.set_ylabel("PI State")
    ax.set_xlabel("Time (minutes)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(PLOTS_DIR, filename)
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"  Saved: {fp}")
    print(f"  Setpoint changes: {result.setpoint_changes}")


# ── Scenarios ─────────────────────────────────────────────────────────


def scenario_steady_state():
    """Steady state: room at desired, outdoor 25°F (-4°C). Should barely touch setpoint."""
    print("\n=== Steady State (LR at 67°F, outdoor 25°F, 6 hours) ===")
    room = make_living_room(room_temp=f_to_c(67), outdoor_temp=f_to_c(25))
    pi = PIController(desired_temp=f_to_c(67), hp_setpoint=24, ff_heat_slope=0.30)
    # Warm up to reach equilibrium
    run_sim(room, pi, duration_hours=4)
    result = run_sim(room, pi, duration_hours=6)
    plot_result(result, "Steady State: LR at 67°F, Outdoor 25°F\n6h observation",
                "real_steady_state.png", comfort_band=(1.0, 2.0))
    rng = max(result.room_temp_f) - min(result.room_temp_f)
    print(f"  Room temp range: {rng:.1f}°F")


def scenario_cold_recovery():
    """Recovery: door was open, room dropped to 63°F. How fast to get back to 67°F?"""
    print("\n=== Cold Recovery (LR: 63°F → 67°F, outdoor 25°F) ===")
    room = make_living_room(room_temp=f_to_c(63), outdoor_temp=f_to_c(25))
    pi = PIController(desired_temp=f_to_c(67), hp_setpoint=24, ff_heat_slope=0.30)
    result = run_sim(room, pi, duration_hours=4)
    plot_result(result, "Cold Recovery: LR dropped to 63°F, target 67°F\nOutdoor 25°F",
                "real_cold_recovery.png", comfort_band=(1.0, 2.0))
    # Time to reach 66°F (within comfort band)
    for i, temp in enumerate(result.room_temp_f):
        if temp >= 66.0:
            print(f"  Time to 66°F (comfort): {result.time_min[i]:.0f} min")
            break


def scenario_setpoint_change():
    """User raises desired from 67°F to 70°F. How fast?"""
    print("\n=== Setpoint Change (LR: 67→70°F, outdoor 30°F) ===")
    room = make_living_room(room_temp=f_to_c(67), outdoor_temp=f_to_c(30))
    pi = PIController(desired_temp=f_to_c(67), hp_setpoint=24, ff_heat_slope=0.30)
    run_sim(room, pi, duration_hours=4)  # Equilibrium

    def raise_temp(r, p):
        p.set_desired(f_to_c(70))

    events = [(0.5, raise_temp)]  # Change at 30 min
    result = run_sim(room, pi, duration_hours=4, events=events)
    plot_result(result, "Setpoint Change: LR 67→70°F at t=30min\nOutdoor 30°F",
                "real_setpoint_change.png", comfort_band=(1.0, 2.0))
    for i, temp in enumerate(result.room_temp_f):
        if result.time_min[i] > 30 and temp >= 69.0:
            print(f"  Time to 69°F (after change): {result.time_min[i] - 30:.0f} min")
            break


def scenario_outdoor_drop():
    """Outdoor temp drops from 35°F to 10°F over 2 hours. Room should hold."""
    print("\n=== Outdoor Drop (LR: 35→10°F over 2h, target 67°F) ===")
    room = make_living_room(room_temp=f_to_c(67), outdoor_temp=f_to_c(35))
    pi = PIController(desired_temp=f_to_c(67), hp_setpoint=23, ff_heat_slope=0.30)
    run_sim(room, pi, duration_hours=4)

    def gradual_drop(r, p):
        # Drop 1°F per 5 min tick (gradual)
        new_outdoor = max(f_to_c(10), r.outdoor_temp - 0.5)
        r.outdoor_temp = new_outdoor

    # Create events every 5 minutes for 2 hours
    events = [(1.0 + i * 5/60, gradual_drop) for i in range(24)]
    result = run_sim(room, pi, duration_hours=6, events=events)
    plot_result(result, "Outdoor Drop: 35→10°F over 2h\nLR target 67°F, FF active",
                "real_outdoor_drop.png", comfort_band=(1.0, 2.0))
    min_room = min(result.room_temp_f)
    print(f"  Min room temp: {min_room:.1f}°F (target: 67°F)")


def scenario_pellet_stove():
    """Pellet stove turns on, HP drops to 64°F, stove off after 4h."""
    print("\n=== Pellet Stove (LR: stove on at 1h, off at 5h) ===")
    room = make_living_room(room_temp=f_to_c(67), outdoor_temp=f_to_c(20))
    pi = PIController(desired_temp=f_to_c(67), hp_setpoint=24, ff_heat_slope=0.30)
    run_sim(room, pi, duration_hours=4)

    def stove_on(r, p):
        r.supplemental_heat = 2000  # 2kW pellet stove
        p.set_desired(f_to_c(64))  # Drop HP target

    def stove_off(r, p):
        r.supplemental_heat = 0
        p.set_desired(f_to_c(67))  # Restore target

    events = [(1.0, stove_on), (5.0, stove_off)]
    result = run_sim(room, pi, duration_hours=8, events=events)
    plot_result(result, "Pellet Stove: on at 1h (+2kW), off at 5h\nHP drops to 64°F, outdoor 20°F",
                "real_pellet_stove.png", comfort_band=(1.0, 2.0))


def scenario_bunkroom_fast():
    """Bunkroom cold recovery — should be faster than LR."""
    print("\n=== Bunkroom Cold Recovery (64°F → 69°F, outdoor 25°F) ===")
    room = make_bunkroom(room_temp=f_to_c(64), outdoor_temp=f_to_c(25))
    pi = PIController(desired_temp=f_to_c(69), hp_setpoint=24, ff_heat_slope=0.35)
    result = run_sim(room, pi, duration_hours=3)
    plot_result(result, "Bunkroom Cold Recovery: 64→69°F\nOutdoor 25°F (fast zone, TC=23min)",
                "real_bunkroom_recovery.png", comfort_band=(1.0, 2.0))
    for i, temp in enumerate(result.room_temp_f):
        if temp >= 68.0:
            print(f"  Time to 68°F (comfort): {result.time_min[i]:.0f} min")
            break


def scenario_extreme_cold():
    """Extreme cold night: outdoor -10°F, HP at capacity."""
    print("\n=== Extreme Cold (LR: outdoor -10°F, target 67°F) ===")
    room = make_living_room(room_temp=f_to_c(67), outdoor_temp=f_to_c(15))
    pi = PIController(desired_temp=f_to_c(67), hp_setpoint=26, ff_heat_slope=0.30)
    run_sim(room, pi, duration_hours=4)

    def cold_snap(r, p):
        r.outdoor_temp = f_to_c(-10)

    events = [(1.0, cold_snap)]
    result = run_sim(room, pi, duration_hours=8, events=events)
    plot_result(result, "Extreme Cold: outdoor drops to -10°F at 1h\nLR target 67°F",
                "real_extreme_cold.png", comfort_band=(1.0, 2.0))
    min_room = min(result.room_temp_f)
    print(f"  Min room temp: {min_room:.1f}°F")


if __name__ == "__main__":
    print("Realistic PI Simulation (from HA historical data)")
    print("=" * 60)
    print(f"Defaults: Kp={1.0}, Ki={0.02}, deadband=0.5°C, b=0.5 adaptive")
    print(f"Interval: 900s (15 min), FF slope: 0.30 (LR), 0.35 (BR)")

    scenario_steady_state()
    scenario_cold_recovery()
    scenario_setpoint_change()
    scenario_outdoor_drop()
    scenario_pellet_stove()
    scenario_bunkroom_fast()
    scenario_extreme_cold()

    print("\n" + "=" * 60)
    print(f"All plots saved to {PLOTS_DIR}/")
