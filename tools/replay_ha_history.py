#!/usr/bin/env python3
"""Replay HA history data through the PI controller with different parameters.

Reads pi_regression_data.json (or any HA history export), extracts sensor
timeseries, and runs the PI controller in simulation with real-world
conditions. Compares parameter sets (e.g., kd=0 vs kd=0.5) using the
same weather/temperature history.

Usage:
    python tools/replay_ha_history.py [data_file] [--zone ZONE_NAME]

Default data file: pi_regression_data.json
Default zone: first climate entity found
"""

import argparse
import json
import logging
import math
import sys
import os
import time
from datetime import datetime, timezone

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_pi_scenarios import SimEntity, _make_sim_config
from tests.benchmark_metrics import compute_all_metrics


# ── Data Parsing ──────────────────────────────────────────────────────────


def parse_sensor_ts(records, convert_f_to_c=False):
    """Parse sensor records into [(timestamp_epoch, value)] sorted by time."""
    points = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r["last_changed"])
            val = float(r["state"])
        except (ValueError, TypeError, KeyError):
            continue
        if convert_f_to_c:
            val = (val - 32) * 5 / 9
        points.append((ts.timestamp(), val))
    points.sort()
    return points


def parse_climate_ts(records, ha_unit_f=True):
    """Parse climate entity records into timestamped state dicts.

    Returns [(epoch, {room_temp_c, desired_c, hp_setpoint, integral, ff_offset})]
    """
    points = []
    for r in records:
        attrs = r.get("attributes", {})
        try:
            ts = datetime.fromisoformat(r["last_changed"])
        except (ValueError, KeyError):
            continue

        current_temp = attrs.get("current_temperature")
        if current_temp is None:
            continue

        current_temp = float(current_temp)
        if ha_unit_f:
            current_temp = (current_temp - 32) * 5 / 9

        desired = attrs.get("desired_temp")
        if desired is not None:
            desired = float(desired)
        else:
            temp_attr = attrs.get("temperature")
            if temp_attr is not None:
                desired = float(temp_attr)
                if ha_unit_f:
                    desired = (desired - 32) * 5 / 9

        points.append((ts.timestamp(), {
            "room_temp_c": current_temp,
            "desired_c": desired,
            "hp_setpoint": attrs.get("hp_setpoint"),
            "integral": attrs.get("pi_integral"),
            "ff_offset": attrs.get("ff_offset"),
        }))
    points.sort()
    return points


def interpolate_at(ts_points, target_epoch):
    """Linear interpolation of (epoch, value) timeseries at target_epoch."""
    if not ts_points:
        return None
    if target_epoch <= ts_points[0][0]:
        return ts_points[0][1]
    if target_epoch >= ts_points[-1][0]:
        return ts_points[-1][1]
    for i in range(len(ts_points) - 1):
        t0, v0 = ts_points[i]
        t1, v1 = ts_points[i + 1]
        if t0 <= target_epoch <= t1:
            if t1 == t0:
                return v0
            frac = (target_epoch - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    return ts_points[-1][1]


# ── Replay Engine ─────────────────────────────────────────────────────────


def replay_with_params(climate_ts, outdoor_ts, model_input_ts_list,
                       config_overrides, tick_interval_s=900):
    """Replay real-world history through PI controller with given params.

    Instead of using a thermal model, we feed real room temperatures directly.
    The PI controller sees the actual room temp at each tick and computes
    what HP setpoint it would have commanded.

    Returns history list compatible with benchmark_metrics.
    """
    if not climate_ts or not outdoor_ts:
        return []

    # Determine time range
    start_epoch = climate_ts[0][0]
    end_epoch = climate_ts[-1][0]

    # Build config
    seed_slope = config_overrides.pop("seed_slope", 0.35)
    config = _make_sim_config(seed_factor=1.0)
    config.update(config_overrides)
    config["pi_ff_heat_slope"] = seed_slope
    config["pi_ff_cool_slope"] = seed_slope

    entity = SimEntity(config)
    pi = entity._pi

    # Get initial desired temp from data
    initial_desired = climate_ts[0][1].get("desired_c")
    if initial_desired:
        pi._desired_temp = initial_desired

    history = []
    tick = 0
    current_epoch = start_epoch

    import asyncio
    loop = asyncio.new_event_loop()

    while current_epoch <= end_epoch:
        # Get real room temp at this time
        climate_state = None
        for i, (t, state) in enumerate(climate_ts):
            if t >= current_epoch:
                climate_state = state
                break
        if climate_state is None:
            climate_state = climate_ts[-1][1]

        room_temp_c = climate_state["room_temp_c"]
        desired_c = climate_state.get("desired_c")
        if desired_c is not None:
            pi._desired_temp = desired_c

        # Set entity state
        entity._attr_current_temperature = room_temp_c

        # Get outdoor temp
        outdoor_c = interpolate_at(outdoor_ts, current_epoch)
        if outdoor_c is not None:
            pi._outdoor_temp = outdoor_c

        # Get model input values
        for i, mi_ts in enumerate(model_input_ts_list):
            if i < len(pi._model_input_values):
                val = interpolate_at(mi_ts, current_epoch)
                if val is not None:
                    pi._model_input_values[i] = val

        # Set timing for PI tick
        pi._pi_last_tick_time = current_epoch - tick_interval_s
        sim_clock = [current_epoch]
        original_monotonic = time.monotonic
        time.monotonic = lambda: sim_clock[0]

        try:
            loop.run_until_complete(pi._pi_tick())
        finally:
            time.monotonic = original_monotonic

        # Record history
        error = (desired_c or pi._desired_temp) - room_temp_c
        history.append({
            "tick": tick,
            "room_temp": room_temp_c,
            "desired": desired_c or pi._desired_temp,
            "hp_setpoint": pi._hp_setpoint,
            "integral": pi._pi_integral,
            "ff_offset": pi._ff_offset,
            "error": error,
            "outdoor": outdoor_c or 0.0,
            "rls_obs_count": pi._rls_heat.observation_count,
            "epoch": current_epoch,
        })

        tick += 1
        current_epoch += tick_interval_s

    loop.close()
    return history


# ── Main ──────────────────────────────────────────────────────────────────


def find_climate_entities(data):
    """Find climate entity keys in the data."""
    return [k for k in data.keys() if k.startswith("climate.")]


def find_outdoor_sensor(data):
    """Find outdoor temperature sensor in the data."""
    candidates = [
        k for k in data.keys()
        if "outdoor" in k.lower() or "weather" in k.lower() or "pirate" in k.lower()
    ]
    for c in candidates:
        if "temperature" in c.lower() or "temp" in c.lower():
            return c
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Replay HA history with different PI parameters")
    parser.add_argument("data_file", nargs="?", default="pi_regression_data.json",
                        help="Path to HA history JSON file")
    parser.add_argument("--zone", default=None, help="Climate entity to replay (default: first found)")
    parser.add_argument("--unit-f", action="store_true", default=True,
                        help="HA temperatures are in °F (default: True)")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.data_file}...")
    with open(args.data_file) as f:
        data = json.load(f)

    # Find entities
    climate_entities = find_climate_entities(data)
    if not climate_entities:
        print("No climate entities found in data!")
        sys.exit(1)

    zone = args.zone
    if zone and zone not in data:
        # Try matching by name
        matches = [e for e in climate_entities if zone.lower() in e.lower()]
        zone = matches[0] if matches else None
    if not zone:
        zone = climate_entities[0]

    outdoor_sensor = find_outdoor_sensor(data)

    print(f"Climate entity: {zone}")
    print(f"Outdoor sensor: {outdoor_sensor}")
    print(f"Available zones: {climate_entities}")
    print()

    # Parse timeseries
    climate_ts = parse_climate_ts(data[zone], ha_unit_f=args.unit_f)
    outdoor_ts = parse_sensor_ts(data[outdoor_sensor], convert_f_to_c=args.unit_f) if outdoor_sensor else []

    if not climate_ts:
        print("No climate data points found!")
        sys.exit(1)

    t0 = datetime.fromtimestamp(climate_ts[0][0])
    t1 = datetime.fromtimestamp(climate_ts[-1][0])
    duration_h = (climate_ts[-1][0] - climate_ts[0][0]) / 3600
    print(f"Time range: {t0} to {t1} ({duration_h:.0f} hours)")
    print(f"Climate data points: {len(climate_ts)}")
    print(f"Outdoor data points: {len(outdoor_ts)}")
    print()

    # Define parameter sets to compare
    PARAM_SETS = [
        {"name": "Optimal (ki=0.15, kd=0, b=0.3)",
         "pi_ki": 0.15, "pi_kd": 0.0, "pi_setpoint_weight": 0.3},
        {"name": "With D (ki=0.15, kd=0.5, b=0.3)",
         "pi_ki": 0.15, "pi_kd": 0.5, "pi_setpoint_weight": 0.3},
        {"name": "With D (ki=0.15, kd=1.0, b=0.3)",
         "pi_ki": 0.15, "pi_kd": 1.0, "pi_setpoint_weight": 0.3},
        {"name": "Higher b (ki=0.15, kd=0, b=0.5)",
         "pi_ki": 0.15, "pi_kd": 0.0, "pi_setpoint_weight": 0.5},
        {"name": "Lower ki (ki=0.08, kd=0, b=0.3)",
         "pi_ki": 0.08, "pi_kd": 0.0, "pi_setpoint_weight": 0.3},
    ]

    # Run replays
    print("=" * 100)
    print(f"REPLAY COMPARISON — {zone}")
    print("=" * 100)

    header = (f"{'Config':<42} {'ITAE':>8} {'Cold':>6} {'MaxRun':>7} "
              f"{'Revers':>7} {'SPchg':>6} {'IntRMS':>8} {'RLS':>5}")
    print(header)
    print("-" * 100)

    all_results = []
    for params in PARAM_SETS:
        name = params.pop("name")
        config = dict(params)
        config["pi_kd_filter_n"] = 8

        history = replay_with_params(
            climate_ts, outdoor_ts, [],
            config_overrides=config,
            tick_interval_s=900,
        )

        if not history:
            print(f"{name:<42} (no data)")
            continue

        m = compute_all_metrics(history)
        print(f"{name:<42} {m['itae']:>8.1f} {m['cold_ticks']:>6} {m['cold_max_run']:>7} "
              f"{m['reversals']:>7} {m['setpoint_changes']:>6} {m['integral_rms']:>8.1f} "
              f"{m['rls_obs_count']:>5}")

        all_results.append({"name": name, "metrics": m, "history_len": len(history)})

    print("=" * 100)

    # Save results
    output_file = f"replay_results_{zone.replace('.', '_')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Also save the actual vs simulated setpoints for the best config
    if all_results:
        print(f"\nTo plot: use the history data in {output_file}")


if __name__ == "__main__":
    main()
