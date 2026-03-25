#!/usr/bin/env python3
"""PI controller simulator with simple room thermal model.

Simulates the PI+feedforward controller against an RC thermal model
to verify algorithm behavior before deploying to real hardware.

Usage:
    python tools/pi_simulator.py

Outputs PNG plots to tools/plots/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

# Add project root so we can import actual constants
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Thermal Model ─────────────────────────────────────────────────────


@dataclass
class RoomModel:
    """Simple 2R1C thermal model of a room.

    Models:
    - Wall thermal resistance (outdoor → room)
    - Room thermal capacitance (thermal mass)
    - Heat pump input (setpoint-dependent heating/cooling power)
    - Optional supplemental heat source

    Units: °C, Watts, seconds
    """

    # Thermal properties
    R_wall: float = 0.01  # Wall thermal resistance (°C/W) — lower = leakier building
    C_room: float = 500_000.0  # Room thermal capacity (J/°C) — higher = more thermal mass
    # A typical well-insulated room: R=0.01, C=500000 gives ~83 min time constant

    # Heat pump model (simplified)
    hp_max_power: float = 3500.0  # Max heating power (W) — typical mini-split
    hp_min_power: float = 500.0  # Min modulated power (W)
    hp_response_time: float = 300.0  # Time for HP to reach new setpoint output (s)

    # State
    room_temp: float = 20.0  # Current room temp (°C)
    outdoor_temp: float = 0.0  # Outdoor temp (°C)
    hp_output: float = 0.0  # Current HP heat output (W)
    supplemental_heat: float = 0.0  # External heat source (W)

    def step(self, hp_setpoint: float, desired_temp: float, dt: float = 60.0):
        """Advance the model by dt seconds.

        Args:
            hp_setpoint: HP setpoint from PI controller (°C)
            desired_temp: The room temp the HP is trying to reach (for power calc)
            dt: Time step in seconds
        """
        # HP power model: proportional to (setpoint - room_temp), clamped
        # Higher setpoint = more power output (HP tries harder)
        hp_target_power = self._hp_power_from_setpoint(hp_setpoint, self.room_temp)

        # Smooth HP response (first-order lag)
        alpha = min(1.0, dt / self.hp_response_time)
        self.hp_output += alpha * (hp_target_power - self.hp_output)

        # Heat flows
        q_wall = (self.outdoor_temp - self.room_temp) / self.R_wall  # Heat loss through walls
        q_hp = self.hp_output  # Heat from HP
        q_supp = self.supplemental_heat  # Supplemental heat

        # Temperature change: dT = (Q_total * dt) / C
        q_total = q_wall + q_hp + q_supp
        self.room_temp += (q_total * dt) / self.C_room

    def _hp_power_from_setpoint(self, hp_setpoint: float, room_temp: float) -> float:
        """Model HP heat output based on setpoint vs room temp.

        When setpoint > room_temp: HP heats (positive power)
        When setpoint < room_temp: HP cools (negative power, or just off)
        Power scales with the difference, clamped to HP capacity.
        """
        diff = hp_setpoint - room_temp
        if diff > 0:
            # Heating: power proportional to diff, with diminishing returns
            power = self.hp_max_power * min(1.0, diff / 5.0)
            return max(self.hp_min_power, power)
        elif diff < -1:
            # Cooling (or HP effectively off)
            return 0.0
        else:
            # Near setpoint: minimum power to maintain
            return self.hp_min_power * max(0, diff + 1)


# ── PI Controller (mirrors actual implementation) ─────────────────────


@dataclass
class PIController:
    """PI+FF controller matching pi_controller.py implementation."""

    kp: float = 1.5
    ki: float = 0.05
    deadband: float = 0.5
    setpoint_weight: float = 1.0
    min_temp: float = 16.0
    max_temp: float = 30.0
    ff_heat_slope: float = 0.3
    ff_heat_reference: float = 15.0
    interval: float = 900.0  # seconds

    # State
    integral: float = 0.0
    hp_setpoint: float = 22.0
    ff_offset: float = 0.0
    desired_temp: float = 22.0  # in °C
    settled_ticks: int = 0
    paused: bool = False

    # FF buckets (simplified — use linear model instead of buckets for sim)
    ff_buckets: dict = field(default_factory=dict)

    def tick(self, current_temp: float, outdoor_temp: Optional[float] = None) -> float:
        """Run one PI tick. Returns new HP setpoint."""
        if self.paused:
            return self.hp_setpoint

        error = self.desired_temp - current_temp

        # Feedforward from outdoor temp
        self.ff_offset = 0.0
        if outdoor_temp is not None:
            delta = max(0, self.ff_heat_reference - outdoor_temp)
            raw_ff = self.ff_heat_slope * delta
            # Graduated ramp on overshoot
            ff_scale = max(0.0, min(1.0, 1.0 + error / self.deadband))
            self.ff_offset = raw_ff * ff_scale

        # Deadband
        in_deadband = abs(error) < self.deadband
        if in_deadband:
            self.integral *= 0.9
            self.settled_ticks += 1
            p_term = 0.0
        else:
            self.settled_ticks = 0
            # 2-DOF setpoint weighting
            p_error = self.setpoint_weight * self.desired_temp - current_temp
            p_term = self.kp * p_error
            self.integral += error

        # Stale integral clamp (heating mode)
        if self.integral < 0 and error >= -self.deadband:
            self.integral = 0.0

        # Hard safety cap
        self.integral = max(-50.0, min(50.0, self.integral))

        i_term = self.ki * self.integral
        raw_setpoint = self.desired_temp + p_term + i_term + self.ff_offset
        new_setpoint = round(max(self.min_temp, min(self.max_temp, raw_setpoint)))

        # Back-calculation anti-windup
        if self.ki != 0:
            saturation_error = new_setpoint - raw_setpoint
            if abs(saturation_error) > 0.01:
                kb = 1.0 / self.ki
                self.integral += kb * saturation_error
                self.integral = max(-50.0, min(50.0, self.integral))

        self.hp_setpoint = new_setpoint
        return new_setpoint

    def set_desired(self, temp: float):
        """User changes the desired temperature."""
        self.desired_temp = temp
        self.integral = 0.0  # Zero on user input


# ── Simulation Runner ─────────────────────────────────────────────────


@dataclass
class SimResult:
    """Stores simulation results for plotting."""
    time_hours: np.ndarray = field(default_factory=lambda: np.array([]))
    room_temp: list = field(default_factory=list)
    desired_temp: list = field(default_factory=list)
    hp_setpoint: list = field(default_factory=list)
    outdoor_temp: list = field(default_factory=list)
    integral: list = field(default_factory=list)
    ff_offset: list = field(default_factory=list)
    hp_output: list = field(default_factory=list)


def run_sim(
    room: RoomModel,
    pi: PIController,
    duration_hours: float = 24.0,
    dt: float = 60.0,  # model step (seconds)
    events: Optional[list] = None,
) -> SimResult:
    """Run a simulation.

    Args:
        room: Thermal model
        pi: PI controller
        duration_hours: Simulation duration
        dt: Model time step
        events: List of (time_hours, callback) tuples for mid-sim changes
    """
    events = sorted(events or [], key=lambda e: e[0])
    event_idx = 0

    steps = int(duration_hours * 3600 / dt)
    pi_interval_steps = int(pi.interval / dt)

    result = SimResult()
    times = []

    for step in range(steps):
        t_seconds = step * dt
        t_hours = t_seconds / 3600.0

        # Process events
        while event_idx < len(events) and events[event_idx][0] <= t_hours:
            events[event_idx][1](room, pi)
            event_idx += 1

        # PI tick at controller interval
        if step % pi_interval_steps == 0:
            pi.tick(room.room_temp, room.outdoor_temp)

        # Advance thermal model
        room.step(pi.hp_setpoint, pi.desired_temp, dt)

        # Record
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


# ── Plotting ──────────────────────────────────────────────────────────


def plot_result(result: SimResult, title: str, filename: str):
    """Generate a 4-panel plot of simulation results."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    t = result.time_hours

    # Panel 1: Temperatures
    ax = axes[0]
    ax.plot(t, result.room_temp, "b-", label="Room Temp", linewidth=1.5)
    ax.plot(t, result.desired_temp, "r--", label="Desired Temp", linewidth=1.5)
    ax.plot(t, result.outdoor_temp, "g:", label="Outdoor Temp", linewidth=1)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: HP Setpoint
    ax = axes[1]
    ax.plot(t, result.hp_setpoint, "m-", label="HP Setpoint", linewidth=1.5)
    ax.plot(t, result.desired_temp, "r--", label="Desired", linewidth=1, alpha=0.5)
    ax.set_ylabel("HP Setpoint (°C)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: PI State
    ax = axes[2]
    ax.plot(t, result.integral, "c-", label="Integral", linewidth=1.5)
    ax.plot(t, result.ff_offset, "orange", label="FF Offset", linewidth=1.5)
    ax.set_ylabel("PI State")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 4: HP Power Output
    ax = axes[3]
    ax.plot(t, [w / 1000 for w in result.hp_output], "k-", label="HP Output (kW)", linewidth=1)
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time (hours)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


# ── Scenarios ─────────────────────────────────────────────────────────


def scenario_1_steady_state():
    """Basic heating: cold start to steady state."""
    print("Scenario 1: Cold start to steady state (0°C outdoor, target 22°C)")
    room = RoomModel(room_temp=15.0, outdoor_temp=0.0)
    pi = PIController(desired_temp=22.0, hp_setpoint=22.0)
    result = run_sim(room, pi, duration_hours=8)
    plot_result(result, "Scenario 1: Cold Start to Steady State\n(Outdoor 0°C, Target 22°C, Start 15°C)", "s1_cold_start.png")
    print(f"  Final room temp: {result.room_temp[-1]:.2f}°C (target: 22.0°C)")
    print(f"  Final integral: {result.integral[-1]:.3f}")


def scenario_2_setpoint_change():
    """Setpoint step change: 22°C → 24°C at t=2h."""
    print("Scenario 2: Setpoint change 22→24°C at t=2h")
    room = RoomModel(room_temp=22.0, outdoor_temp=0.0)
    pi = PIController(desired_temp=22.0, hp_setpoint=24.0)

    # Pre-run to reach steady state
    pre = run_sim(room, pi, duration_hours=4)

    events = [
        (2.0, lambda r, p: p.set_desired(24.0)),
    ]
    result = run_sim(room, pi, duration_hours=8, events=events)
    plot_result(result, "Scenario 2: Setpoint Change 22→24°C at t=2h\n(Outdoor 0°C)", "s2_setpoint_change.png")
    print(f"  Final room temp: {result.room_temp[-1]:.2f}°C (target: 24.0°C)")


def scenario_3_setpoint_weight_comparison():
    """Compare setpoint weight b=1.0 vs b=0.5 on setpoint step."""
    print("Scenario 3: Setpoint weight comparison (b=1.0 vs b=0.5)")

    results = {}
    for weight in [1.0, 0.5]:
        room = RoomModel(room_temp=22.0, outdoor_temp=0.0)
        pi = PIController(desired_temp=22.0, hp_setpoint=24.0, setpoint_weight=weight)
        # Reach steady state first
        run_sim(room, pi, duration_hours=4)
        events = [(1.0, lambda r, p: p.set_desired(24.0))]
        results[weight] = run_sim(room, pi, duration_hours=6, events=events)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Scenario 3: Setpoint Weight Comparison\n(22→24°C step at t=1h, Outdoor 0°C)", fontsize=14, fontweight="bold")

    ax = axes[0]
    for w, r in results.items():
        ax.plot(r.time_hours, r.room_temp, label=f"b={w}", linewidth=1.5)
    ax.axhline(24.0, color="r", linestyle="--", alpha=0.5, label="Target")
    ax.set_ylabel("Room Temp (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for w, r in results.items():
        ax.plot(r.time_hours, r.hp_setpoint, label=f"b={w}", linewidth=1.5)
    ax.set_ylabel("HP Setpoint (°C)")
    ax.set_xlabel("Time (hours)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, "s3_setpoint_weight.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def scenario_4_outdoor_temp_drop():
    """Disturbance: outdoor temp drops from 5°C to -10°C at t=2h."""
    print("Scenario 4: Outdoor temp drop (5→-10°C at t=2h)")
    room = RoomModel(room_temp=22.0, outdoor_temp=5.0)
    pi = PIController(desired_temp=22.0, hp_setpoint=22.0)
    # Steady state first
    run_sim(room, pi, duration_hours=4)

    def drop_outdoor(r, p):
        r.outdoor_temp = -10.0

    events = [(2.0, drop_outdoor)]
    result = run_sim(room, pi, duration_hours=12, events=events)
    plot_result(result, "Scenario 4: Outdoor Temp Drop 5→-10°C at t=2h\n(Target 22°C, FF active)", "s4_outdoor_drop.png")
    print(f"  Max room temp dip: {min(result.room_temp):.2f}°C")
    print(f"  Final room temp: {result.room_temp[-1]:.2f}°C")


def scenario_5_pellet_stove():
    """Supplemental heat: pellet stove on, HP target dropped to 18°C."""
    print("Scenario 5: Pellet stove scenario (HP drops to 18°C, stove provides heat)")
    room = RoomModel(room_temp=22.0, outdoor_temp=0.0)
    pi = PIController(desired_temp=22.0, hp_setpoint=24.0)
    # Steady state
    run_sim(room, pi, duration_hours=4)

    def stove_on(r, p):
        r.supplemental_heat = 2000.0  # 2kW pellet stove
        p.set_desired(18.0)  # Drop HP target

    def stove_off(r, p):
        r.supplemental_heat = 0.0
        p.set_desired(22.0)  # Restore HP target

    events = [
        (2.0, stove_on),
        (6.0, stove_off),
    ]
    result = run_sim(room, pi, duration_hours=12, events=events)
    plot_result(result, "Scenario 5: Pellet Stove On/Off\n(Stove 2kW at t=2h, off at t=6h, outdoor 0°C)", "s5_pellet_stove.png")
    print(f"  Room temp during stove: {max(result.room_temp[int(3*3600/60):int(6*3600/60)]):.1f}°C peak")


def scenario_6_saturation_recovery():
    """Anti-windup: HP pegged at max during extreme cold, then warms up."""
    print("Scenario 6: Saturation recovery (extreme cold → warm up)")
    room = RoomModel(room_temp=18.0, outdoor_temp=-20.0)
    pi = PIController(desired_temp=22.0, hp_setpoint=22.0)

    def warm_up(r, p):
        r.outdoor_temp = 5.0

    events = [(6.0, warm_up)]
    result = run_sim(room, pi, duration_hours=16, events=events)
    plot_result(result, "Scenario 6: Saturation Recovery\n(Outdoor -20°C→5°C at t=6h, target 22°C)", "s6_saturation_recovery.png")
    print(f"  HP setpoint during cold: {max(result.hp_setpoint[:int(6*3600/60)])}")
    print(f"  Integral at warm-up: {result.integral[int(6*3600/60)]:.1f}")
    print(f"  Time to recover to 22°C: check plot")


def scenario_7_deadband_behavior():
    """Verify deadband prevents oscillation in steady state."""
    print("Scenario 7: Deadband behavior in steady state")
    room = RoomModel(room_temp=22.0, outdoor_temp=5.0)
    pi = PIController(desired_temp=22.0, hp_setpoint=23.0)
    # Long steady state run
    run_sim(room, pi, duration_hours=4)
    result = run_sim(room, pi, duration_hours=12)

    # Count setpoint changes
    changes = sum(1 for i in range(1, len(result.hp_setpoint)) if result.hp_setpoint[i] != result.hp_setpoint[i-1])
    plot_result(result, "Scenario 7: Deadband Steady State\n(Outdoor 5°C, target 22°C, 12h observation)", "s7_deadband.png")
    print(f"  Setpoint changes in 12h: {changes}")
    print(f"  Room temp range: {min(result.room_temp):.2f} - {max(result.room_temp):.2f}°C")


def scenario_8_no_ff_vs_ff():
    """Compare PI-only vs PI+FF on outdoor temp disturbance."""
    print("Scenario 8: PI-only vs PI+FF comparison")

    results = {}
    for label, slope in [("No FF", 0.0), ("With FF", 0.3)]:
        room = RoomModel(room_temp=22.0, outdoor_temp=5.0)
        pi = PIController(desired_temp=22.0, hp_setpoint=23.0, ff_heat_slope=slope)
        run_sim(room, pi, duration_hours=4)

        def drop(r, p):
            r.outdoor_temp = -10.0
        events = [(2.0, drop)]
        results[label] = run_sim(room, pi, duration_hours=10, events=events)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Scenario 8: PI-Only vs PI+FF\n(Outdoor drops 5→-10°C at t=2h)", fontsize=14, fontweight="bold")

    ax = axes[0]
    for label, r in results.items():
        ax.plot(r.time_hours, r.room_temp, label=label, linewidth=1.5)
    ax.axhline(22.0, color="r", linestyle="--", alpha=0.5)
    ax.set_ylabel("Room Temp (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label, r in results.items():
        ax.plot(r.time_hours, r.hp_setpoint, label=label, linewidth=1.5)
    ax.set_ylabel("HP Setpoint (°C)")
    ax.set_xlabel("Time (hours)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(PLOTS_DIR, "s8_ff_comparison.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("PI Controller Simulation Suite")
    print("=" * 60)
    print()

    scenario_1_steady_state()
    print()
    scenario_2_setpoint_change()
    print()
    scenario_3_setpoint_weight_comparison()
    print()
    scenario_4_outdoor_temp_drop()
    print()
    scenario_5_pellet_stove()
    print()
    scenario_6_saturation_recovery()
    print()
    scenario_7_deadband_behavior()
    print()
    scenario_8_no_ff_vs_ff()
    print()
    print("=" * 60)
    print(f"All plots saved to {PLOTS_DIR}/")
