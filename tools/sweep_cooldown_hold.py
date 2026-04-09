#!/usr/bin/env python3
"""Parameter sweep: sensor cooldown and hold timer.

Runs the real PI controller (via TasmotaPIAdapter) through heating scenarios
at different tick intervals (simulating sensor cooldown) and hold timer values.

Tick interval maps to sensor cooldown:
  1 min  → 60s cooldown (PI runs every minute)
  2 min  → 120s cooldown
  5 min  → 300s cooldown (current production)
  15 min → 900s (timer-only, no sensor ticks between)

Hold timer is the minimum time between setpoint changes (currently 1800s).

Usage:
    python -m tools.sweep_cooldown_hold
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.benchmark_metrics import (
    compute_itae,
    compute_settling_time,
    count_setpoint_changes,
    compute_comfort_violations,
)


TICK_INTERVALS = {
    "60s": 1.0,      # 1 minute
    "120s": 2.0,     # 2 minutes
    "300s": 5.0,     # 5 minutes (current)
}

HOLD_TIMES = {
    "0m": 0,
    "10m": 600,
    "15m": 900,
    "30m": 1800,     # current
    "60m": 3600,
}


def _make_controller(hold_seconds):
    """Create a PI controller with the specified hold time.

    The hold time is hardcoded at 1800s in _pi_tick_inner. We patch it by
    shifting _last_setpoint_change_time before each tick so that the elapsed
    time check sees the desired hold duration.
    """
    ctrl = TasmotaPIAdapter({"pi_ff_heat_slope": 0.35})
    if hold_seconds != 1800:
        original_tick = ctrl.tick
        hold_delta = 1800.0 - hold_seconds

        def patched_tick(room_temp_c, outdoor_temp_c, dt_seconds, model_inputs=None):
            # Shift time so hold check sees desired threshold
            ctrl._pi._last_setpoint_change_time -= hold_delta
            result = original_tick(room_temp_c, outdoor_temp_c, dt_seconds, model_inputs)
            ctrl._pi._last_setpoint_change_time += hold_delta
            return result

        ctrl.tick = patched_tick
    return ctrl


HP_LAG = 15.0  # minutes — typical residential HP response lag


def run_cold_start(tick_min, hold_seconds, profile, outdoor=2.0):
    """Cold start: room at 17°C, desired 20.5°C, outdoor 2°C."""
    ctrl = _make_controller(hold_seconds)
    ctrl.set_desired_temp(20.5)
    model = ThermalModel(profile=profile, initial_temp=17.0, outdoor_temp=outdoor,
                         sensor_noise_sigma=0.05, noise_seed=42,
                         hp_lag_minutes=HP_LAG)
    n_ticks = int(8 * 60 / tick_min)
    return run_scenario(ctrl, model, n_ticks=n_ticks, mode="heat",
                        tick_interval_min=tick_min)


def run_setpoint_change(tick_min, hold_seconds, profile, outdoor=5.0):
    """Steady state at 20.5°C, then setpoint change to 22°C at hour 2."""
    ctrl = _make_controller(hold_seconds)
    ctrl.set_desired_temp(20.5)
    model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=outdoor,
                         sensor_noise_sigma=0.05, noise_seed=42,
                         hp_lag_minutes=HP_LAG)
    change_tick = int(2 * 60 / tick_min)
    n_ticks = int(8 * 60 / tick_min)
    return run_scenario(ctrl, model, n_ticks=n_ticks, mode="heat",
                        tick_interval_min=tick_min,
                        desired_schedule={change_tick: 22.0})


def run_disturbance(tick_min, hold_seconds, profile):
    """Steady state at 20.5°C, outdoor drops from 5°C to -5°C at hour 2."""
    ctrl = _make_controller(hold_seconds)
    ctrl.set_desired_temp(20.5)
    model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=5.0,
                         sensor_noise_sigma=0.05, noise_seed=42,
                         hp_lag_minutes=HP_LAG)
    drop_tick = int(2 * 60 / tick_min)
    n_ticks = int(8 * 60 / tick_min)
    return run_scenario(ctrl, model, n_ticks=n_ticks, mode="heat",
                        tick_interval_min=tick_min,
                        outdoor_schedule={drop_tick: -5.0})


def analyze(history, desired, tick_min, deadband=0.3):
    """Extract key metrics from history, converted to wall-clock time."""
    settling_ticks = compute_settling_time(history, deadband * 2)
    settling_min = round(settling_ticks * tick_min) if settling_ticks is not None else None
    changes = count_setpoint_changes(history)
    violation_ticks, _, max_undershoot = compute_comfort_violations(history, deadband)
    total_ticks = len(history) or 1
    final_err = abs(history[-1]["room_temp"] - desired)

    # Time-weighted absolute error in real minutes
    itae_min = sum(
        (i * tick_min) * abs(h["desired"] - h["room_temp"])
        for i, h in enumerate(history)
    )

    return {
        "itae_min": round(itae_min, 0),
        "settling_min": settling_min,
        "changes": changes,
        "violations_pct": round(100 * violation_ticks / total_ticks, 1),
        "final_err": round(final_err, 2),
    }


def run_mild_disturbance(tick_min, hold_seconds, profile):
    """Near steady state with mild outdoor temp ramp — exercises hold timer band.

    Room at 20.5°C (target), outdoor slowly drops from 5°C to 0°C over 4 hours.
    This creates a gradual error buildup in the 0.3-1.0°C range where the
    hold timer is the binding constraint.
    """
    ctrl = _make_controller(hold_seconds)
    ctrl.set_desired_temp(20.5)
    model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=5.0,
                         sensor_noise_sigma=0.05, noise_seed=42,
                         hp_lag_minutes=HP_LAG)

    n_ticks = int(8 * 60 / tick_min)
    ramp_start = int(1 * 60 / tick_min)  # start ramp at hour 1
    ramp_end = int(5 * 60 / tick_min)    # end ramp at hour 5

    def outdoor_fn(tick):
        if tick < ramp_start:
            return 5.0
        if tick > ramp_end:
            return 0.0
        progress = (tick - ramp_start) / (ramp_end - ramp_start)
        return 5.0 - 5.0 * progress  # 5°C → 0°C

    return run_scenario(ctrl, model, n_ticks=n_ticks, mode="heat",
                        tick_interval_min=tick_min,
                        outdoor_schedule=outdoor_fn)


def run_small_setpoint_bump(tick_min, hold_seconds, profile):
    """Setpoint change of just 0.5°C — stays in the hold timer band.

    Room at 20.5°C, bump to 21.0°C at hour 1. Error stays 0.3-0.5°C,
    which is between deadband and the 1°C bypass threshold.
    """
    ctrl = _make_controller(hold_seconds)
    ctrl.set_desired_temp(20.5)
    model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=5.0,
                         sensor_noise_sigma=0.05, noise_seed=42,
                         hp_lag_minutes=HP_LAG)

    change_tick = int(1 * 60 / tick_min)
    n_ticks = int(8 * 60 / tick_min)

    return run_scenario(ctrl, model, n_ticks=n_ticks, mode="heat",
                        tick_interval_min=tick_min,
                        desired_schedule={change_tick: 21.0})


def run_solar_gain(tick_min, hold_seconds, profile):
    """Solar gain during heating: room at 20.5°C, strong solar ramp starting hour 1.

    Models the solar saturation scenario: cold morning (outdoor 0°C), HP heating,
    then strong solar gain pushes room above desired. PI needs to drop setpoint.
    Solar proxy ramps from 0 to 0.8 over 2 hours (morning sun hitting windows).
    """
    ctrl = _make_controller(hold_seconds)
    ctrl.set_desired_temp(20.5)
    model = ThermalModel(profile=profile, initial_temp=19.0, outdoor_temp=0.0,
                         sensor_noise_sigma=0.05, noise_seed=42,
                         hp_lag_minutes=HP_LAG)

    solar_start = int(1 * 60 / tick_min)
    solar_peak = int(3 * 60 / tick_min)
    n_ticks = int(8 * 60 / tick_min)

    def solar_fn(tick):
        if tick < solar_start:
            return 0.0
        if tick > solar_peak:
            return 0.8
        progress = (tick - solar_start) / (solar_peak - solar_start)
        return 0.8 * progress

    return run_scenario(ctrl, model, n_ticks=n_ticks, mode="heat",
                        tick_interval_min=tick_min,
                        solar_schedule=solar_fn)


def main():
    profile_name = "standard_residential"
    profile = QUICK_PROFILES[profile_name]

    scenarios = {
        "cold_start": (run_cold_start, 20.5),
        "setpoint_change": (run_setpoint_change, 22.0),
        "outdoor_drop": (run_disturbance, 20.5),
        "mild_disturbance": (run_mild_disturbance, 20.5),
        "small_bump": (run_small_setpoint_bump, 21.0),
        "solar_gain": (run_solar_gain, 20.5),
    }

    for scenario_name, (run_fn, desired) in scenarios.items():
        print(f"\n{'='*72}")
        print(f"  SCENARIO: {scenario_name} ({profile_name})")
        print(f"{'='*72}")
        print(f"  {'Cooldown':>8s} {'Hold':>6s} | {'ITAE':>9s} {'Settle':>8s} "
              f"{'Changes':>8s} {'Viol%':>6s} {'FinalE':>7s}")
        print(f"  {'-'*8} {'-'*6} | {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*7}")

        for cd_label, tick_min in TICK_INTERVALS.items():
            for ht_label, hold_s in HOLD_TIMES.items():
                history = run_fn(tick_min, hold_s, profile)
                m = analyze(history, desired, tick_min)
                settle_str = f"{m['settling_min']}m" if m['settling_min'] is not None else "N/A"
                print(f"  {cd_label:>8s} {ht_label:>6s} | {m['itae_min']:>9.0f} "
                      f"{settle_str:>8s} "
                      f"{m['changes']:>8d} {m['violations_pct']:>6.1f} "
                      f"{m['final_err']:>7.2f}")


if __name__ == "__main__":
    main()
