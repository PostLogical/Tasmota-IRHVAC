#!/usr/bin/env python3
"""A/B simulation: variable-rate vs full-rate deadband integration.

Uses the real PIController from pi/pi_controller.py with the realistic
RoomModel from pi_realistic_sim.py.  Runs each scenario twice and
compares ITAE, reversals, settling time, and comfort band metrics.

Usage:
    python tools/sim_variable_rate.py
"""

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, UnitOfTemperature

from custom_components.tasmota_irhvac.pi.pi_controller import PIController
from tests.conftest import make_pi_config
from tests.benchmark_metrics import (
    compute_itae,
    compute_overshoot,
    compute_settling_time,
    count_reversals,
    count_setpoint_changes,
)

# Reuse room models from the existing sim
from tools.pi_realistic_sim import RoomModel, c_to_f, f_to_c


def make_living_room(room_temp=20.0, outdoor_temp=0.0):
    """Living room: τ≈60min, properly sized HP for -15°F design temp."""
    # R_wall=0.015: HP (3500W) maintains 22°C at outdoor=-26°C (-15°F)
    # C_room=360000: τ = R*C = 0.015 * 360000 / 60 ≈ 90min (well-insulated)
    # Adjusted to τ≈60min: C = 60*60/0.015 = 240000
    return RoomModel(
        name="Living Room",
        room_temp=room_temp,
        outdoor_temp=outdoor_temp,
        R_wall=0.015,
        C_room=240_000,
        hp_max_power=3500,
        hp_response_time=300,
    )


def make_bunkroom(room_temp=20.0, outdoor_temp=0.0):
    """Bunkroom: τ≈23min, same HP sizing, faster response."""
    # τ = R*C/60 = 0.015 * 92000 / 60 ≈ 23min
    return RoomModel(
        name="Bunkroom",
        room_temp=room_temp,
        outdoor_temp=outdoor_temp,
        R_wall=0.015,
        C_room=92_000,
        hp_max_power=3500,
        hp_response_time=180,
    )

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "local", "data",
                            "variable_rate_sim_results.json")


# ── Entity Adapter ───────────────────────────────────────────────────

class SimEntity:
    """Minimal entity that PIController can read from during simulation."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.AUTO, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _ir_temp_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    def __init__(self, config, room_temp_c=20.0, outdoor_temp_c=0.0):
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = room_temp_c
        self._attr_target_temperature = 22.0
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "sim_entry"

        self.send_ir = MagicMock()  # Not async in sim — just track calls
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = MagicMock(return_value=None)

        self._pi = PIController(self, config)

    @property
    def target_temperature(self):
        if self._pi and self._pi._desired_temp is not None:
            return self._pi._desired_temp
        return self._attr_target_temperature

    @property
    def temperature_unit(self):
        return self._attr_temperature_unit


# ── Simulation Loop ──────────────────────────────────────────────────

@dataclass
class SimTrace:
    """Per-tick trace data for one simulation run."""
    time_min: list = field(default_factory=list)
    room_temp_c: list = field(default_factory=list)
    hp_setpoint: list = field(default_factory=list)
    integral: list = field(default_factory=list)
    ff_offset: list = field(default_factory=list)
    error: list = field(default_factory=list)


async def run_sim(entity, room, duration_hours, desired_c,
                  outdoor_events=None, dt_room=60.0, pi_interval=900.0):
    """Run closed-loop simulation with real PIController and RoomModel.

    Args:
        entity: SimEntity wrapping PIController.
        room: RoomModel from pi_realistic_sim.
        duration_hours: Total simulation time.
        desired_c: Target temperature in °C.
        outdoor_events: List of (hour, outdoor_temp_c) for outdoor temp changes.
        dt_room: Room model timestep in seconds.
        pi_interval: PI tick interval in seconds.
    """
    pi = entity._pi
    await pi.set_temperature(desired_c)
    pi._inputs.outdoor_temp = room.outdoor_temp

    outdoor_events = sorted(outdoor_events or [], key=lambda e: e[0])
    event_idx = 0
    steps = int(duration_hours * 3600 / dt_room)
    pi_interval_steps = int(pi_interval / dt_room)
    trace = SimTrace()
    # Sync mono_time with any existing _last_setpoint_change_time so dwell
    # calculations work correctly across warmup → measurement boundaries.
    mono_time = max(pi_interval, pi._last_setpoint_change_time + pi_interval)

    for step in range(steps):
        t_sec = step * dt_room
        t_hr = t_sec / 3600.0

        # Apply outdoor temp events
        while event_idx < len(outdoor_events) and outdoor_events[event_idx][0] <= t_hr:
            room.outdoor_temp = outdoor_events[event_idx][1]
            pi._inputs.outdoor_temp = room.outdoor_temp
            event_idx += 1

        # Room physics step
        room.step(float(pi._hp_setpoint), desired_c, dt_room)

        # Update entity with new room temp
        entity._attr_current_temperature = room.room_temp

        # PI tick at interval
        if step % pi_interval_steps == 0 and step > 0:
            pi._pi_last_tick_time = mono_time - pi_interval
            with patch("time.monotonic", return_value=mono_time):
                await pi._pi_tick()
            mono_time += pi_interval

        # Record trace every PI interval
        if step % pi_interval_steps == 0:
            trace.time_min.append(t_sec / 60.0)
            trace.room_temp_c.append(room.room_temp)
            trace.hp_setpoint.append(int(pi._hp_setpoint))
            trace.integral.append(pi._pi_integral)
            trace.ff_offset.append(pi._ff_offset)
            trace.error.append(desired_c - room.room_temp)

    return trace


def trace_to_history(trace, desired_c, outdoor_c):
    """Convert SimTrace to benchmark_metrics history format."""
    history = []
    for i in range(len(trace.time_min)):
        history.append({
            "tick": i,
            "room_temp": trace.room_temp_c[i],
            "desired": desired_c,
            "hp_setpoint": trace.hp_setpoint[i],
            "integral": trace.integral[i],
            "ff_offset": trace.ff_offset[i],
            "error": trace.error[i],
            "outdoor": outdoor_c,
            "rls_obs_count": 0,
        })
    return history


# ── A/B Runner ───────────────────────────────────────────────────────

async def run_ab(label, room_factory, desired_c, duration_hours,
                 outdoor_events=None, config_overrides=None):
    """Run a scenario with both variable-rate and full-rate, compare metrics."""

    config = make_pi_config(config_overrides)
    outdoor_c = room_factory().outdoor_temp  # Initial outdoor

    results = {}
    for policy, rate_fn in [("variable-rate", None), ("full-rate", lambda ae: 1.0)]:
        room = room_factory()
        entity = SimEntity(config, room.room_temp, room.outdoor_temp)
        if rate_fn:
            entity._pi._deadband_integration_rate = rate_fn

        trace = await run_sim(entity, room, duration_hours, desired_c,
                              outdoor_events=outdoor_events)
        history = trace_to_history(trace, desired_c, outdoor_c)

        metrics = {
            "itae": compute_itae(history),
            "overshoot": compute_overshoot(history),
            "settling_time": compute_settling_time(history),
            "reversals": count_reversals(history),
            "setpoint_changes": count_setpoint_changes(history),
            "final_room_c": trace.room_temp_c[-1] if trace.room_temp_c else 0,
            "final_integral": trace.integral[-1] if trace.integral else 0,
            "max_deviation": max(abs(e) for e in trace.error) if trace.error else 0,
        }
        results[policy] = {"trace": trace, "metrics": metrics}

    # Print comparison
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25} {'Variable-rate':>15} {'Full-rate':>15} {'Diff':>10}")
    print(f"  {'-' * 65}")
    for key in ["itae", "overshoot", "settling_time", "reversals",
                "setpoint_changes", "final_room_c", "max_deviation"]:
        v = results["variable-rate"]["metrics"][key]
        f = results["full-rate"]["metrics"][key]
        vs = f"{v:.2f}" if isinstance(v, float) else str(v)
        fs = f"{f:.2f}" if isinstance(f, float) else str(f)
        if v is not None and f is not None and isinstance(v, (int, float)):
            d = f - v
            ds = f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}"
        else:
            ds = "—"
        print(f"  {key:<25} {vs:>15} {fs:>15} {ds:>10}")

    # Print setpoint trajectory summary
    for policy in ["variable-rate", "full-rate"]:
        t = results[policy]["trace"]
        sps = t.hp_setpoint
        # Show setpoint at key time points
        n = len(sps)
        indices = [0, n//6, n//3, n//2, 2*n//3, 5*n//6, n-1]
        summary = [f"{t.time_min[i]:.0f}min:{sps[i]}°C" for i in indices if i < n]
        print(f"  {policy} setpoints: {', '.join(summary)}")

    return results


# ── Scenarios ────────────────────────────────────────────────────────

async def run_ab_with_warmup(label, room_factory, desired_c, duration_hours,
                            warmup_hours=6, outdoor_events=None,
                            config_overrides=None):
    """Run A/B with a warmup phase so the system starts near equilibrium.

    The warmup runs with variable-rate (current policy) to establish the
    HP setpoint and integral state, then copies that state to the full-rate
    entity.  This ensures both runs start from the same operating point.
    """
    config = make_pi_config(config_overrides)
    outdoor_c = room_factory().outdoor_temp

    # Warmup: run variable-rate to equilibrium
    warmup_room = room_factory()
    warmup_entity = SimEntity(config, warmup_room.room_temp, warmup_room.outdoor_temp)
    await run_sim(warmup_entity, warmup_room, warmup_hours, desired_c)
    warmup_pi = warmup_entity._pi

    # Snapshot the equilibrium state
    eq_room_temp = warmup_room.room_temp
    eq_hp_setpoint = int(warmup_pi._hp_setpoint)
    eq_integral = warmup_pi._pi_integral
    eq_ff_offset = warmup_pi._ff_offset
    eq_ff_confidence = warmup_pi._ff_confidence

    print(f"\n  Warmup equilibrium: room={eq_room_temp:.2f}°C, "
          f"HP={eq_hp_setpoint}°C, integral={eq_integral:.2f}, "
          f"ff={eq_ff_offset:.2f}")

    results = {}
    for policy, rate_fn in [("variable-rate", None), ("full-rate", lambda ae: 1.0)]:
        room = room_factory()
        room.room_temp = eq_room_temp
        # Let room physics also reach equilibrium
        room.hp_output = warmup_room.hp_output

        entity = SimEntity(config, eq_room_temp, room.outdoor_temp)
        pi = entity._pi
        # Restore equilibrium PI state
        pi._hp_setpoint = eq_hp_setpoint
        pi._pi_integral = eq_integral
        pi._ff_offset = eq_ff_offset
        pi._ff_confidence = eq_ff_confidence
        pi._desired_temp = desired_c
        pi._inputs.outdoor_temp = room.outdoor_temp

        if rate_fn:
            pi._deadband_integration_rate = rate_fn

        trace = await run_sim(entity, room, duration_hours, desired_c,
                              outdoor_events=outdoor_events)
        history = trace_to_history(trace, desired_c, outdoor_c)

        metrics = {
            "itae": compute_itae(history),
            "overshoot": compute_overshoot(history),
            "settling_time": compute_settling_time(history),
            "reversals": count_reversals(history),
            "setpoint_changes": count_setpoint_changes(history),
            "final_room_c": trace.room_temp_c[-1] if trace.room_temp_c else 0,
            "final_integral": trace.integral[-1] if trace.integral else 0,
            "max_deviation": max(abs(e) for e in trace.error) if trace.error else 0,
        }
        results[policy] = {"trace": trace, "metrics": metrics}

    # Print comparison
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25} {'Variable-rate':>15} {'Full-rate':>15} {'Diff':>10}")
    print(f"  {'-' * 65}")
    for key in ["itae", "overshoot", "settling_time", "reversals",
                "setpoint_changes", "final_room_c", "max_deviation"]:
        v = results["variable-rate"]["metrics"][key]
        f = results["full-rate"]["metrics"][key]
        vs = f"{v:.2f}" if isinstance(v, float) else str(v)
        fs = f"{f:.2f}" if isinstance(f, float) else str(f)
        if v is not None and f is not None and isinstance(v, (int, float)):
            d = f - v
            ds = f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}"
        else:
            ds = "—"
        print(f"  {key:<25} {vs:>15} {fs:>15} {ds:>10}")

    for policy in ["variable-rate", "full-rate"]:
        t = results[policy]["trace"]
        sps = t.hp_setpoint
        n = len(sps)
        indices = [0, n//6, n//3, n//2, 2*n//3, 5*n//6, n-1]
        summary = [f"{t.time_min[i]:.0f}m:{sps[i]}" for i in indices if i < n]
        print(f"  {policy} SP: {', '.join(summary)}")

    return results


async def main():
    print("Variable-Rate vs Full-Rate Deadband Integration")
    print("Using real PIController + realistic room physics")
    print("Outdoor temps chosen so HP can reach target (deadband relevant)")
    print("=" * 70)

    all_results = {}

    # 1. Living room mild winter — HP can reach target, deadband active
    # outdoor=10°C → HP@26 maintains ~22°C (equilibrium)
    all_results["lr_mild"] = await run_ab_with_warmup(
        "Living Room: Mild conditions (target 22°C, outdoor 10°C, 12h)",
        lambda: make_living_room(room_temp=20.0, outdoor_temp=10.0),
        desired_c=22.0, duration_hours=12,
    )

    # 2. Living room shoulder season — barely needs HP
    # outdoor=18°C → HP@23 maintains target easily
    all_results["lr_shoulder"] = await run_ab_with_warmup(
        "Living Room: Shoulder season (target 22°C, outdoor 18°C, 12h)",
        lambda: make_living_room(room_temp=21.0, outdoor_temp=18.0),
        desired_c=22.0, duration_hours=12,
    )

    # 3. Bunkroom mild — fast τ, most oscillation-prone, HP can reach target
    all_results["br_mild"] = await run_ab_with_warmup(
        "Bunkroom: Mild conditions (target 21°C, outdoor 10°C, 12h)",
        lambda: make_bunkroom(room_temp=19.0, outdoor_temp=10.0),
        desired_c=21.0, duration_hours=12,
    )

    # 4. Bunkroom shoulder — fast τ, minimal HP effort, most sensitive to rate
    all_results["br_shoulder"] = await run_ab_with_warmup(
        "Bunkroom: Shoulder season (target 21°C, outdoor 18°C, 12h)",
        lambda: make_bunkroom(room_temp=20.5, outdoor_temp=18.0),
        desired_c=21.0, duration_hours=12,
    )

    # 5. Living room cold front starting from equilibrium
    all_results["lr_cold_front"] = await run_ab_with_warmup(
        "Living Room: Cold front from equilibrium (10→0°C over 3h, 10h)",
        lambda: make_living_room(room_temp=22.0, outdoor_temp=10.0),
        desired_c=22.0, duration_hours=10,
        outdoor_events=[(1.0 + i * 5/60, 10.0 - i * (10.0/36))
                        for i in range(37)],  # 10→0°C over 3h
    )

    # 6. Bunkroom perturbation from equilibrium — the key oscillation test
    # Start at equilibrium, then a door opens (room drops 0.5°C)
    all_results["br_perturbation"] = await run_ab_with_warmup(
        "Bunkroom: Door-open perturbation from equilibrium (12h)",
        lambda: make_bunkroom(room_temp=20.5, outdoor_temp=10.0),
        desired_c=21.0, duration_hours=12,
    )

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Full-rate minus Variable-rate")
    print(f"{'=' * 70}")
    print(f"  {'Scenario':<30} {'ΔITAE':>10} {'ΔReversals':>12} {'ΔChanges':>10} {'ΔOvershoot':>12}")
    print(f"  {'-' * 74}")
    for key, res in all_results.items():
        vm = res["variable-rate"]["metrics"]
        fm = res["full-rate"]["metrics"]
        print(f"  {key:<30} {fm['itae']-vm['itae']:>+10.1f} "
              f"{fm['reversals']-vm['reversals']:>+12d} "
              f"{fm['setpoint_changes']-vm['setpoint_changes']:>+10d} "
              f"{fm['overshoot']-vm['overshoot']:>+12.2f}")

    # Save raw metrics
    saved = {}
    for key, res in all_results.items():
        saved[key] = {
            p: res[p]["metrics"] for p in ["variable-rate", "full-rate"]
        }
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(saved, f, indent=2, default=str)
    print(f"\nRaw metrics saved to {RESULTS_FILE}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.INFO)  # Suppress PI debug spam, keep warnings
    asyncio.run(main())
