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
from custom_components.tasmota_irhvac.pi.batch_learning import (
    DiversityAwareBuffer,
    Observation,
    weighted_least_squares,
    compare_and_report,
    compute_blended_update,
)


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
                       config_overrides, tick_interval_s=900,
                       room_sensor_ts=None, temp_unit=None):
    """Replay real-world history through PI controller with given params.

    Instead of using a thermal model, we feed real room temperatures directly.
    The PI controller sees the actual room temp at each tick and computes
    what HP setpoint it would have commanded.

    Args:
        room_sensor_ts: Optional [(epoch, temp_c)] from the raw temperature
            sensor. If provided, room temperature is interpolated from this
            instead of from the climate entity's current_temperature attribute,
            which HA rounds to the entity's precision (often whole degrees).

    Returns history list compatible with benchmark_metrics.
    """
    if not climate_ts or not outdoor_ts:
        return [], None

    # Determine time range from raw sensor if available, else climate entity
    if room_sensor_ts:
        start_epoch = max(climate_ts[0][0], room_sensor_ts[0][0])
        end_epoch = min(climate_ts[-1][0], room_sensor_ts[-1][0])
    else:
        start_epoch = climate_ts[0][0]
        end_epoch = climate_ts[-1][0]

    # Build config
    seed_slope = config_overrides.pop("seed_slope", 0.35)
    config = _make_sim_config(seed_factor=1.0)
    config.update(config_overrides)
    config["pi_outdoor_seed_heat"] = seed_slope
    config["pi_outdoor_seed_cool"] = seed_slope

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
        # Get desired temp from climate entity
        climate_state = None
        for i, (t, state) in enumerate(climate_ts):
            if t >= current_epoch:
                climate_state = state
                break
        if climate_state is None:
            climate_state = climate_ts[-1][1]

        desired_c = climate_state.get("desired_c")
        if desired_c is not None:
            pi._desired_temp = desired_c

        # Get room temp: prefer raw sensor (full precision) over climate
        # entity attribute (rounded to entity precision, often whole degrees)
        if room_sensor_ts:
            room_temp_c = interpolate_at(room_sensor_ts, current_epoch)
        else:
            room_temp_c = climate_state["room_temp_c"]

        if room_temp_c is None:
            current_epoch += tick_interval_s
            continue

        # Set entity state
        entity._attr_current_temperature = room_temp_c

        # Get outdoor temp
        outdoor_c = interpolate_at(outdoor_ts, current_epoch)
        if outdoor_c is not None:
            pi._inputs.outdoor_temp = outdoor_c

        # Get model input values
        for i, mi_ts in enumerate(model_input_ts_list):
            if i < len(pi._inputs.values):
                val = interpolate_at(mi_ts, current_epoch)
                if val is not None:
                    # delta_from_room: convert to °C, store raw for obs, then delta
                    m_input = pi._inputs.model_inputs[i] if i < len(pi._inputs.model_inputs) else {}
                    if m_input.get("delta_from_room") and room_temp_c is not None:
                        from homeassistant.util.unit_conversion import TemperatureConverter
                        from homeassistant.const import UnitOfTemperature
                        unit = temp_unit or UnitOfTemperature.CELSIUS
                        val_c = TemperatureConverter.convert(
                            val, unit, UnitOfTemperature.CELSIUS
                        )
                        pi._inputs._raw_for_obs[i] = val_c
                        val = val_c - room_temp_c
                    else:
                        pi._inputs._raw_for_obs[i] = val
                    pi._inputs.values[i] = val

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
    return history, pi


def _run_estimator_batch(buf, current_beta, n_features, obs_count, epoch,
                         drift_correction_signs):
    """Run batch WLS on the buffer and return a snapshot.

    Mirrors the logic of PIController._run_batch_analysis but operates
    on a standalone buffer — no PI controller or RLS model involved.
    """
    if buf.needs_recompute:
        buf.recompute_info_matrix()

    observations = buf.get_all()
    if len(observations) < 20:
        return None

    result = weighted_least_squares(
        observations, n_features=n_features, current_beta=current_beta,
        room_rate_threshold=0.02, min_observations=20,
    )
    if result is None:
        return None

    coeff_names = ["intercept", "outdoor_delta"]
    compare_and_report(
        result, current_beta, coeff_names,
        change_threshold_pct=20.0, min_observations=20,
        log_prefix="[replay] ",
    )
    compute_blended_update(result, prior_std=1.0, max_step=1.0)

    # Determine post-update coefficients
    if result.recommend_update and result.beta_blended:
        beta_after = list(result.beta_blended)
    else:
        beta_after = list(current_beta)

    # Drift detection: track per-coefficient correction direction
    if result.beta_blended and result.beta_current:
        n = min(len(result.beta_blended), len(result.beta_current))
        signs = []
        for i in range(n):
            delta = result.beta_blended[i] - result.beta_current[i]
            if abs(delta) < 1e-4:
                signs.append(0)
            elif delta > 0:
                signs.append(1)
            else:
                signs.append(-1)
        if not drift_correction_signs:
            drift_correction_signs.extend([] for _ in range(n))
        while len(drift_correction_signs) < n:
            drift_correction_signs.append([])
        for i in range(n):
            drift_correction_signs[i].append(signs[i])
            if len(drift_correction_signs[i]) > 10:
                drift_correction_signs[i].pop(0)

    leverage_scores = buf.get_leverage_scores()

    return {
        "obs_count": obs_count,
        "epoch": epoch,
        "timestamp": datetime.fromtimestamp(epoch).isoformat(),
        # Coefficients
        "beta_before": [round(v, 6) for v in current_beta],
        "beta_batch": [round(v, 6) for v in result.beta_batch],
        "beta_blended": [round(v, 6) for v in result.beta_blended] if result.beta_blended else None,
        "beta_after": [round(v, 6) for v in beta_after],
        "beta_std_err": [round(v, 6) for v in result.beta_std_err] if result.beta_std_err else [],
        "blend_gains": [round(v, 4) for v in result.blend_gains] if result.blend_gains else [],
        "recommend_update": result.recommend_update,
        "max_coeff_change_pct": round(result.max_coeff_change_pct, 2),
        # Batch fit quality
        "batch_rms": round(result.residual_rms, 4),
        "n_total": result.n_total,
        "n_eligible": result.n_eligible,
        "n_outliers_excluded": result.n_outliers_excluded,
        "held_features": sorted(result.held_features),
        # Buffer composition
        "buffer_size": len(buf),
        "leverage_min": round(min(leverage_scores), 6) if leverage_scores else 0.0,
        "leverage_max": round(max(leverage_scores), 6) if leverage_scores else 0.0,
        "leverage_mean": round(sum(leverage_scores) / len(leverage_scores), 6) if leverage_scores else 0.0,
        "leverage_std": round(_std(leverage_scores), 6) if leverage_scores else 0.0,
        # Drift detection
        "drift_correction_signs": [list(h) for h in drift_correction_signs],
    }


def _print_long_run_report(snapshots, n_features, drift_correction_signs):
    """Print the per-cycle summary table and convergence analysis."""
    coeff_names = ["intercept", "outdoor_Δ"] + [f"input_{i}" for i in range(n_features - 2)]

    # Per-cycle summary table
    print(f"{'Cycle':>5} {'Timestamp':<20} {'Buf':>5} {'Elig':>5} {'Outlr':>5} "
          f"{'RMS':>8} {'Chg%':>6} {'Upd':>3} {'LevMin':>8} {'LevStd':>8} ", end="")
    for cn in coeff_names:
        print(f" {cn:>10}", end="")
    print()
    print("-" * (75 + 11 * n_features))

    for i, s in enumerate(snapshots):
        upd = "Y" if s["recommend_update"] else "n"
        coeff_str = ""
        for v in s["beta_after"]:
            coeff_str += f" {v:>10.4f}"

        print(f"{i + 1:>5} {s['timestamp']:<20} {s['buffer_size']:>5} "
              f"{s['n_eligible']:>5} {s['n_outliers_excluded']:>5} "
              f"{s['batch_rms']:>8.4f} {s['max_coeff_change_pct']:>6.1f} "
              f"{upd:>3} {s['leverage_min']:>8.5f} {s['leverage_std']:>8.5f}"
              f"{coeff_str}")

    print("-" * (75 + 11 * n_features))

    # Drift summary
    drifting = _get_drifting(drift_correction_signs, coeff_names, threshold=5)
    if drifting:
        print("\nDRIFT DETECTED:")
        for name, count in drifting:
            print(f"  {name}: {count} consecutive same-direction corrections")
    else:
        print("\nNo persistent drift detected.")

    # Convergence summary
    if len(snapshots) >= 2:
        first_beta = snapshots[0]["beta_after"]
        last_beta = snapshots[-1]["beta_after"]
        print("\nCoefficient trajectory (first → last cycle):")
        for j, name in enumerate(coeff_names):
            if j < len(first_beta) and j < len(last_beta):
                delta = last_beta[j] - first_beta[j]
                std_err = ""
                if snapshots[-1].get("beta_std_err") and j < len(snapshots[-1]["beta_std_err"]):
                    se = snapshots[-1]["beta_std_err"][j]
                    std_err = f"  (±{se:.4f})"
                print(f"  {name:<15} {first_beta[j]:>8.4f} → {last_beta[j]:>8.4f}  "
                      f"(Δ={delta:+.4f}){std_err}")


def _print_cross_validation(buf, n_features, current_beta):
    """Ljung §16.4 cross-validation: fit on first 80%, predict on last 20%.

    Reports prediction RMS on held-out data and residual autocorrelation
    at lag 1 to check for model adequacy.
    """
    all_obs = buf.get_all()
    # Sort by timestamp to ensure temporal ordering
    all_obs.sort(key=lambda o: o.timestamp)
    n = len(all_obs)
    if n < 40:
        print("\nCross-validation: insufficient observations (need ≥ 40).")
        return

    split = int(n * 0.8)
    train = all_obs[:split]
    test = all_obs[split:]

    # Fit on training set
    train_result = weighted_least_squares(
        train, n_features=n_features, current_beta=current_beta,
        room_rate_threshold=0.02, min_observations=20,
    )
    if train_result is None:
        print("\nCross-validation: insufficient eligible training observations.")
        return

    beta_fit = train_result.beta_batch

    # Predict on test set (same eligibility filter)
    residuals = []
    for o in test:
        if o.clamped or abs(o.room_rate) >= 0.02:
            continue
        x = o.features[:n_features]
        while len(x) < n_features:
            x.append(0.0)
        y_true = o.hp_setpoint - o.current_c
        y_pred = sum(b * xi for b, xi in zip(beta_fit, x))
        residuals.append(y_true - y_pred)

    if len(residuals) < 5:
        print("\nCross-validation: insufficient eligible test observations.")
        return

    pred_rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))

    # Lag-1 autocorrelation (should be near 0 for adequate model)
    mean_r = sum(residuals) / len(residuals)
    centered = [r - mean_r for r in residuals]
    var = sum(c * c for c in centered)
    if var > 1e-12 and len(centered) > 1:
        cov1 = sum(centered[i] * centered[i + 1] for i in range(len(centered) - 1))
        autocorr = cov1 / var
    else:
        autocorr = 0.0

    print(f"\nCross-validation (train={split}, test={n - split}, eligible test={len(residuals)}):")
    print(f"  Prediction RMS on held-out data: {pred_rms:.4f}")
    print(f"  Residual lag-1 autocorrelation:  {autocorr:.4f}", end="")
    if abs(autocorr) > 0.3:
        print("  ← HIGH (model may be missing dynamics)")
    elif abs(autocorr) > 0.15:
        print("  ← moderate")
    else:
        print("  ← OK (residuals approximately white)")
    print(f"  Training batch fit coefficients: {['%.4f' % b for b in beta_fit]}")


def _get_drifting(drift_signs, coeff_names, threshold=5):
    """Check for persistent same-direction corrections."""
    drifting = []
    for i, history in enumerate(drift_signs):
        if len(history) < threshold:
            continue
        recent = history[-threshold:]
        if all(s == 1 for s in recent) or all(s == -1 for s in recent):
            name = coeff_names[i] if i < len(coeff_names) else f"β{i}"
            count = len([s for s in history if s == recent[0]])
            drifting.append((name, count))
    return drifting


def _std(values):
    """Standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


# ── Main ──────────────────────────────────────────────────────────────────


def find_climate_entities(data):
    """Find climate entity keys in the data."""
    return [k for k in data.keys() if k.startswith("climate.")]


def find_room_sensor(data, zone):
    """Find the raw temperature sensor for a climate zone.

    HA's climate entity rounds current_temperature to entity precision
    (often whole degrees). The raw sensor has full precision (e.g. 0.01°F).
    """
    # Extract zone name (e.g. "living_room" from "climate.living_room_heat_pump")
    zone_name = zone.replace("climate.", "").replace("_heat_pump", "")

    # Common sensor naming patterns
    candidates = [
        k for k in data.keys()
        if k.startswith("sensor.") and "temperature" in k.lower()
        and zone_name.replace("_", " ").split()[0] in k.lower()
    ]
    # Prefer air/room sensors over weather sensors
    for c in candidates:
        if "air_sensor" in c or "room" in c:
            return c
    return candidates[0] if candidates else None


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
    parser.add_argument("--room-sensor", default=None,
                        help="Raw room temperature sensor entity_id (overrides auto-detect)")
    parser.add_argument("--unit-f", action="store_true", default=True,
                        help="HA temperatures are in °F (default: True)")
    parser.add_argument("--long-run", action="store_true",
                        help="Enable batch WLS cycling during replay")
    parser.add_argument("--batch-every", type=int, default=48,
                        help="Ticks between batch WLS cycles (default: 48 = 12h at 15min ticks)")
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
    room_sensor = args.room_sensor or find_room_sensor(data, zone)

    print(f"Climate entity: {zone}")
    print(f"Room sensor:    {room_sensor or '(none — using climate entity, WARNING: rounded)'}")
    print(f"Outdoor sensor: {outdoor_sensor}")
    print(f"Available zones: {climate_entities}")
    print()

    # Parse timeseries
    climate_ts = parse_climate_ts(data[zone], ha_unit_f=args.unit_f)
    outdoor_ts = parse_sensor_ts(data[outdoor_sensor], convert_f_to_c=args.unit_f) if outdoor_sensor else []
    room_sensor_ts = parse_sensor_ts(data[room_sensor], convert_f_to_c=args.unit_f) if room_sensor else None

    if not climate_ts:
        print("No climate data points found!")
        sys.exit(1)

    t0 = datetime.fromtimestamp(climate_ts[0][0])
    t1 = datetime.fromtimestamp(climate_ts[-1][0])
    duration_h = (climate_ts[-1][0] - climate_ts[0][0]) / 3600
    print(f"Time range: {t0} to {t1} ({duration_h:.0f} hours)")
    print(f"Climate data points: {len(climate_ts)}")
    if room_sensor_ts:
        print(f"Room sensor points: {len(room_sensor_ts)} (full precision)")
    print(f"Outdoor data points: {len(outdoor_ts)}")
    print()

    if args.long_run:
        _run_long_run(climate_ts, outdoor_ts, room_sensor_ts, zone,
                      args.batch_every)
    else:
        _run_comparison(climate_ts, outdoor_ts, room_sensor_ts, zone)


def _run_long_run(climate_ts, outdoor_ts, room_sensor_ts, zone,
                  batch_every):
    """Replay real observations through the estimator (not the controller).

    Reconstructs Observation objects from HA history and feeds them into
    a DiversityAwareBuffer, running batch WLS analysis periodically.
    This validates buffer convergence, residual filtering, and drift
    detection on real closed-loop data without the open-loop divergence
    problem of running the PI controller on non-responsive room temps.

    Follows Åström & Wittenmark §12.5: replay the estimator on actual
    closed-loop data, don't re-close the loop in simulation.
    """
    if not climate_ts or not outdoor_ts:
        print("No data to replay!")
        return

    FF_HEAT_REFERENCE = 15.0  # default; matches production config
    MIN_TEMP_C = 16
    MAX_TEMP_C = 30

    # n_features = intercept + outdoor_delta (no model inputs in replay)
    n_features = 2
    buf = DiversityAwareBuffer(n_features=n_features)
    # Seed coefficients matching production defaults
    current_beta = [0.0, 0.35]  # intercept=0, outdoor_delta=seed_slope

    interval_h = batch_every * 15 / 60
    print(f"LONG-RUN ESTIMATOR REPLAY: batch WLS every {batch_every} obs ({interval_h:.0f}h equivalent)")
    print(f"Buffer capacity: {buf._max_size}, features: {n_features}")
    print("=" * 110)

    # Build observation stream from real HA data
    obs_count = 0
    snapshots = []
    drift_correction_signs: list[list[int]] = []
    prev_room_epoch = None
    prev_room_c = None

    for epoch, state in climate_ts:
        hp_setpoint = state.get("hp_setpoint")
        if hp_setpoint is None:
            continue

        # Room temp: prefer raw sensor for full precision
        if room_sensor_ts:
            current_c = interpolate_at(room_sensor_ts, epoch)
        else:
            current_c = state["room_temp_c"]
        if current_c is None:
            continue

        desired_c = state.get("desired_c")
        if desired_c is None:
            continue

        # Outdoor temp → outdoor_delta
        outdoor_c = interpolate_at(outdoor_ts, epoch)
        if outdoor_c is None:
            continue
        outdoor_delta = max(0.0, FF_HEAT_REFERENCE - outdoor_c)

        # Room rate: compute from consecutive readings
        room_rate = 0.0
        if prev_room_epoch is not None and prev_room_c is not None:
            dt_min = (epoch - prev_room_epoch) / 60.0
            if dt_min > 0:
                room_rate = (current_c - prev_room_c) / dt_min
        prev_room_epoch = epoch
        prev_room_c = current_c

        clamped = (hp_setpoint <= MIN_TEMP_C or hp_setpoint >= MAX_TEMP_C)

        obs = Observation(
            timestamp=epoch,
            features=[1.0, outdoor_delta],
            hp_setpoint=float(hp_setpoint),
            current_c=current_c,
            desired_c=desired_c,
            room_rate=room_rate,
            clamped=clamped,
            pi_integral=state.get("integral") or 0.0,
            ff_offset=state.get("ff_offset") or 0.0,
            ff_confidence=1.0,
            raw_c=current_c,
        )
        buf.add(obs)
        obs_count += 1

        # Run batch analysis at intervals
        if obs_count > 0 and obs_count % batch_every == 0:
            snapshot = _run_estimator_batch(
                buf, current_beta, n_features, obs_count, epoch,
                drift_correction_signs,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
                # Apply blended update to current_beta for next cycle
                if snapshot["beta_after"]:
                    current_beta = list(snapshot["beta_after"])

    # Final batch at end
    if obs_count > 0 and obs_count % batch_every != 0:
        snapshot = _run_estimator_batch(
            buf, current_beta, n_features, obs_count,
            climate_ts[-1][0], drift_correction_signs,
        )
        if snapshot is not None:
            snapshots.append(snapshot)

    print(f"\nObservations fed: {obs_count}")
    print(f"Buffer size: {len(buf)}")
    print(f"Batch cycles completed: {len(snapshots)}")
    print()

    if not snapshots:
        print("No batch cycles ran (insufficient observations after filtering).")
        return

    _print_long_run_report(snapshots, n_features, drift_correction_signs)

    # Cross-validation: hold out last 20% of observations
    _print_cross_validation(buf, n_features, current_beta)

    # Save detailed results
    output_file = f"longrun_{zone.replace('.', '_')}.json"
    output = {
        "zone": zone,
        "batch_every": batch_every,
        "total_observations": obs_count,
        "final_buffer_size": len(buf),
        "batch_snapshots": snapshots,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")


def _run_comparison(climate_ts, outdoor_ts, room_sensor_ts, zone):
    """Run parameter-sweep comparison (original mode)."""
    PARAM_SETS = [
        # Current production defaults (const.py)
        {"name": "Current (Kp=1.0 Ki=0.15 b=0.30)",
         "pi_kp": 1.0, "pi_ki": 0.15, "pi_kd": 0.0, "pi_setpoint_weight": 0.3},
        # Sweep winner: conservative (Kp=1.0)
        {"name": "Sweep A (Kp=1.0 Ki=0.20 b=0.15)",
         "pi_kp": 1.0, "pi_ki": 0.20, "pi_kd": 0.0, "pi_setpoint_weight": 0.15},
        # Sweep winner: fine grid best
        {"name": "Sweep B (Kp=0.8 Ki=0.24 b=0.20)",
         "pi_kp": 0.8, "pi_ki": 0.24, "pi_kd": 0.0, "pi_setpoint_weight": 0.20},
        # Middle ground
        {"name": "Sweep C (Kp=1.0 Ki=0.22 b=0.20)",
         "pi_kp": 1.0, "pi_ki": 0.22, "pi_kd": 0.0, "pi_setpoint_weight": 0.20},
        # Test Kd contribution
        {"name": "Sweep D (Kp=1.0 Ki=0.20 b=0.15 Kd=1)",
         "pi_kp": 1.0, "pi_ki": 0.20, "pi_kd": 1.0, "pi_setpoint_weight": 0.15},
        # IMC baseline for reference
        {"name": "IMC τ=60 (Kp=3.00)",
         "pi_tau_estimate": 60.0, "pi_response_lag": 15.0,
         "pi_kd": 0.0, "pi_setpoint_weight": 0.3},
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

        history, pi = replay_with_params(
            climate_ts, outdoor_ts, [],
            config_overrides=config,
            tick_interval_s=900,
            room_sensor_ts=room_sensor_ts,
        )

        if not history:
            print(f"{name:<42} (no data)")
            continue

        m = compute_all_metrics(history)

        # IMC state: learned τ and final gains
        imc_info = ""
        plant_id = getattr(pi, "_plant_id", None)
        if plant_id is not None and plant_id.enabled:
            imc_info = (f"  → τ learned: {plant_id.tau:.0f} min "
                        f"({plant_id.observations} obs), "
                        f"Kp={pi._pi_kp:.2f} Ki={pi._pi_ki:.3f}")

        print(f"{name:<42} {m['itae']:>8.1f} {m['cold_ticks']:>6} {m['cold_max_run']:>7} "
              f"{m['reversals']:>7} {m['setpoint_changes']:>6} {m['integral_rms']:>8.1f} "
              f"{m['rls_obs_count']:>5}")
        if imc_info:
            print(imc_info)

        result = {"name": name, "metrics": m, "history_len": len(history)}
        if plant_id is not None and plant_id.enabled:
            result["tau_learned"] = round(plant_id.tau, 1)
            result["tau_observations"] = plant_id.observations
            result["final_kp"] = round(pi._pi_kp, 3)
            result["final_ki"] = round(pi._pi_ki, 4)
        all_results.append(result)

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
