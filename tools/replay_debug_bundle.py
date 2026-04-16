#!/usr/bin/env python3
"""Replay debug bundle data to calibrate room model parameters.

Two-phase approach:
1. Open-loop: feed production HP setpoints into ThermalModel,
   sweep room model params to minimize RMSE vs production room temp.
2. Closed-loop: run TasmotaPIAdapter with production RLS coefficients
   through calibrated room model, compare HP setpoints + room temp.

Usage:
    python -m tools.replay_debug_bundle \
        --bundle local/debug_bundles/hp_debug_20260416 \
        --zone living_room
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

F_TO_C = lambda f: (f - 32) * 5 / 9


class TimeseriesPoint(NamedTuple):
    t: float  # seconds since epoch
    v: float


def _parse_ts(s: str) -> float:
    """Parse ISO timestamp to epoch seconds."""
    dt = datetime.fromisoformat(s)
    return dt.timestamp()


def load_room_temp(bundle: Path, zone: str) -> list[TimeseriesPoint]:
    """Load high-resolution room temp (°F → °C)."""
    path = bundle / f"room_temp_{zone}_72h.csv"
    pts: list[TimeseriesPoint] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            t = _parse_ts(row["timestamp"])
            temp_c = F_TO_C(float(row["temp_f"]))
            pts.append(TimeseriesPoint(t, temp_c))
    return pts


def load_pi_state(bundle: Path, zone: str) -> list[dict]:
    """Load PI state rows with parsed timestamps and RLS coefficients."""
    path = bundle / f"pi_state_{zone}_72h.csv"
    rows: list[dict] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            # Skip rows with missing critical fields
            if not row.get("hp_setpoint") or not row.get("temperature"):
                continue
            try:
                parsed = {
                    "t": _parse_ts(row["timestamp"]),
                    "current_temperature_f": float(row["current_temperature"]) if row["current_temperature"] else None,
                    "desired_f": float(row["temperature"]),
                    "hp_setpoint_c": float(row["hp_setpoint"]),
                    "pi_integral": float(row["pi_integral"]) if row["pi_integral"] else 0.0,
                    "ff_offset": float(row["ff_offset"]) if row["ff_offset"] else 0.0,
                    "tau_estimate": float(row["tau_estimate"]) if row["tau_estimate"] else 60.0,
                    "rls_obs_count": int(row["rls_observation_count"]) if row["rls_observation_count"] else 0,
                    "rls_coefficients": json.loads(row["rls_heat_coefficients"]) if row["rls_heat_coefficients"] else {},
                    "smith_correction": float(row["smith_correction"]) if row["smith_correction"] else 0.0,
                }
                rows.append(parsed)
            except (ValueError, json.JSONDecodeError):
                continue
    return rows


def load_environment(bundle: Path) -> dict[str, list[TimeseriesPoint]]:
    """Load environment series. Returns dict of series_name -> points."""
    path = bundle / "environment_72h.csv"
    series: dict[str, list[TimeseriesPoint]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["series"]
            t = _parse_ts(row["timestamp"])
            # Try numeric, fall back to binary
            val_str = row["value"]
            if val_str in ("on", "off"):
                val = 1.0 if val_str == "on" else 0.0
            elif val_str in ("unavailable", "unknown", ""):
                continue
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    continue
            series.setdefault(name, []).append(TimeseriesPoint(t, val))
    # Sort each series by time
    for pts in series.values():
        pts.sort(key=lambda p: p.t)
    return series


def interpolate(pts: list[TimeseriesPoint], t: float) -> float:
    """Linear interpolation, clamped at edges."""
    if not pts:
        return 0.0
    if t <= pts[0].t:
        return pts[0].v
    if t >= pts[-1].t:
        return pts[-1].v
    # Binary search
    lo, hi = 0, len(pts) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if pts[mid].t <= t:
            lo = mid
        else:
            hi = mid
    frac = (t - pts[lo].t) / (pts[hi].t - pts[lo].t) if pts[hi].t != pts[lo].t else 0.0
    return pts[lo].v + frac * (pts[hi].v - pts[lo].v)


def step_interpolate(pts: list[TimeseriesPoint], t: float) -> float:
    """Step interpolation (hold last value) for binary signals."""
    if not pts:
        return 0.0
    if t <= pts[0].t:
        return pts[0].v
    # Find last point <= t
    result = pts[0].v
    for p in pts:
        if p.t <= t:
            result = p.v
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Aligned production data on uniform grid
# ---------------------------------------------------------------------------

@dataclass
class AlignedData:
    """Production data interpolated to a uniform time grid."""
    t0: float  # epoch seconds of first tick
    dt_seconds: float  # tick interval
    n_ticks: int

    # Per-tick arrays (length = n_ticks)
    room_temp_c: list[float]     # production room temp (°C) - ground truth
    hp_setpoint_c: list[float]   # production HP setpoint (°C)
    desired_c: list[float]       # production desired temp (°C)
    outdoor_c: list[float]       # outdoor temp (°C)
    solar_proxy: list[float]     # solar proxy (0-1)
    pellet_stove: list[float]    # pellet stove binary
    outdoor_rate: list[float]    # outdoor temp rate (°C/hr → converted from °F/hr)
    pi_integral: list[float]     # production integral
    ff_offset: list[float]       # production FF offset

    # Masks
    active_heating: list[bool] | None = None  # True when HP is actively heating

    # Metadata
    initial_rls_coefficients: dict[str, float] | None = None
    initial_integral: float = 0.0
    initial_ff_offset: float = 0.0


def align_data(bundle: Path, zone: str,
               dt_seconds: float = 60.0) -> AlignedData:
    """Load and align all production data to a uniform time grid.

    Uses 1-minute ticks by default for fine-grained room model accuracy.
    """
    room_temp = load_room_temp(bundle, zone)
    pi_state = load_pi_state(bundle, zone)
    env = load_environment(bundle)

    # Convert outdoor from °F to °C
    outdoor_f = env.get("outdoor_f", [])
    outdoor_c_pts = [TimeseriesPoint(p.t, F_TO_C(p.v)) for p in outdoor_f]

    # Convert outdoor rate from °F/hr to °C/hr
    outdoor_rate_f = env.get("outdoor_rate_f_per_hr", [])
    outdoor_rate_c = [TimeseriesPoint(p.t, p.v * 5.0 / 9.0) for p in outdoor_rate_f]

    solar_pts = env.get("solar_proxy", [])
    pellet_pts = env.get("pellet_supplementing", [])

    # Build pi_state timeseries for interpolation
    hp_setpoint_pts = [TimeseriesPoint(r["t"], r["hp_setpoint_c"]) for r in pi_state]
    desired_f_pts = [TimeseriesPoint(r["t"], r["desired_f"]) for r in pi_state]
    desired_c_pts = [TimeseriesPoint(p.t, F_TO_C(p.v)) for p in desired_f_pts]
    integral_pts = [TimeseriesPoint(r["t"], r["pi_integral"]) for r in pi_state]
    ff_pts = [TimeseriesPoint(r["t"], r["ff_offset"]) for r in pi_state]

    # Time range: intersection of all data
    t_start = max(room_temp[0].t, pi_state[0]["t"])
    t_end = min(room_temp[-1].t, pi_state[-1]["t"])
    n_ticks = int((t_end - t_start) / dt_seconds)

    aligned = AlignedData(
        t0=t_start,
        dt_seconds=dt_seconds,
        n_ticks=n_ticks,
        room_temp_c=[0.0] * n_ticks,
        hp_setpoint_c=[0.0] * n_ticks,
        desired_c=[0.0] * n_ticks,
        outdoor_c=[0.0] * n_ticks,
        solar_proxy=[0.0] * n_ticks,
        pellet_stove=[0.0] * n_ticks,
        outdoor_rate=[0.0] * n_ticks,
        pi_integral=[0.0] * n_ticks,
        ff_offset=[0.0] * n_ticks,
        initial_rls_coefficients=pi_state[0]["rls_coefficients"],
        initial_integral=pi_state[0]["pi_integral"],
        initial_ff_offset=pi_state[0]["ff_offset"],
    )

    for i in range(n_ticks):
        t = t_start + i * dt_seconds
        aligned.room_temp_c[i] = interpolate(room_temp, t)
        aligned.hp_setpoint_c[i] = step_interpolate(hp_setpoint_pts, t)
        aligned.desired_c[i] = interpolate(desired_c_pts, t)
        aligned.outdoor_c[i] = interpolate(outdoor_c_pts, t)
        aligned.solar_proxy[i] = interpolate(solar_pts, t)
        aligned.pellet_stove[i] = step_interpolate(pellet_pts, t)
        aligned.outdoor_rate[i] = interpolate(outdoor_rate_c, t)
        aligned.pi_integral[i] = interpolate(integral_pts, t)
        aligned.ff_offset[i] = interpolate(ff_pts, t)

    # Mask: HP is actively heating when setpoint is above desired temp
    # (when HP setpoint < desired, the HP is idling — solar/other heat is enough)
    aligned.active_heating = [
        aligned.hp_setpoint_c[i] > aligned.desired_c[i] + 0.5
        for i in range(n_ticks)
    ]
    n_active = sum(aligned.active_heating)
    print(f"Active heating: {n_active}/{n_ticks} ticks "
          f"({n_active/n_ticks*100:.1f}%)")

    return aligned


# ---------------------------------------------------------------------------
# Open-loop room model simulation
# ---------------------------------------------------------------------------

def run_open_loop(data: AlignedData,
                  tau_minutes: float,
                  hp_gain: float,
                  solar_gain: float,
                  stove_gain: float,
                  hp_lag_minutes: float = 5.0,
                  reset_interval: int = 0) -> list[float]:
    """Run room model with production HP setpoints, return sim room temp.

    This is a pure thermal model simulation — no PI controller involved.
    Uses exact exponential integration matching ThermalModel.step().

    Args:
        reset_interval: If > 0, reset sim room temp to production every N ticks.
            This isolates short-term model fidelity from long-term drift,
            since open-loop sims diverge when production HP setpoints were
            computed for the real room temp, not the sim room temp.
    """
    dt_min = data.dt_seconds / 60.0
    n = data.n_ticks

    room_temp = data.room_temp_c[0]  # start at production room temp
    effective_sp = data.hp_setpoint_c[0]
    sim_temps: list[float] = []

    for i in range(n):
        # Periodic reset to production room temp
        if reset_interval > 0 and i > 0 and i % reset_interval == 0:
            room_temp = data.room_temp_c[i]
            effective_sp = data.hp_setpoint_c[i]

        # HP lag
        hp_sp = data.hp_setpoint_c[i]
        if hp_lag_minutes > 0:
            lag_decay = math.exp(-dt_min / hp_lag_minutes)
            effective_sp = hp_sp + (effective_sp - hp_sp) * lag_decay
        else:
            effective_sp = hp_sp

        # Heat inputs
        solar_heat = solar_gain * data.solar_proxy[i] * dt_min
        stove_heat = stove_gain * data.pellet_stove[i] * dt_min

        outdoor = data.outdoor_c[i]

        # Equilibrium temperature
        total_gain = 1.0 / tau_minutes + hp_gain
        if total_gain == 0:
            sim_temps.append(room_temp)
            continue

        t_eq = (
            outdoor / tau_minutes
            + hp_gain * effective_sp
            + solar_heat / dt_min
            + stove_heat / dt_min
        ) / total_gain

        # Exponential decay
        decay = math.exp(-dt_min / tau_minutes)
        room_temp = t_eq + (room_temp - t_eq) * decay

        sim_temps.append(room_temp)

    return sim_temps


def compute_rmse(sim: list[float], actual: list[float],
                 mask: list[bool] | None = None) -> float:
    """Root mean squared error between two series, optionally masked."""
    assert len(sim) == len(actual)
    if mask is None:
        pairs = list(zip(sim, actual))
    else:
        pairs = [(s, a) for s, a, m in zip(sim, actual, mask) if m]
    n = len(pairs)
    if n == 0:
        return float("inf")
    return math.sqrt(sum((s - a) ** 2 for s, a in pairs) / n)


def compute_mae(sim: list[float], actual: list[float],
                mask: list[bool] | None = None) -> float:
    """Mean absolute error between two series, optionally masked."""
    assert len(sim) == len(actual)
    if mask is None:
        pairs = list(zip(sim, actual))
    else:
        pairs = [(s, a) for s, a, m in zip(sim, actual, mask) if m]
    n = len(pairs)
    if n == 0:
        return float("inf")
    return sum(abs(s - a) for s, a in pairs) / n


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    tau_minutes: float
    hp_gain: float
    solar_gain: float
    stove_gain: float
    hp_lag_minutes: float
    rmse_c: float
    mae_c: float


def sweep_open_loop(data: AlignedData,
                    tau_range: list[float] | None = None,
                    hp_gain_range: list[float] | None = None,
                    solar_gain_range: list[float] | None = None,
                    stove_gain_range: list[float] | None = None,
                    hp_lag_range: list[float] | None = None,
                    reset_interval: int = 60,
                    ) -> list[CalibrationResult]:
    """Sweep room model parameters, return sorted results (best first).

    Args:
        reset_interval: Reset sim to production every N ticks (default 60 = 1hr).
            Prevents open-loop divergence from masking model quality.
    """
    # Defaults: coarse grid around expected values
    if tau_range is None:
        tau_range = [40, 50, 60, 70, 80, 90, 100, 110, 120]
    if hp_gain_range is None:
        hp_gain_range = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    if solar_gain_range is None:
        solar_gain_range = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    if stove_gain_range is None:
        stove_gain_range = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    if hp_lag_range is None:
        hp_lag_range = [3.0, 5.0, 7.0]

    total = (len(tau_range) * len(hp_gain_range) * len(solar_gain_range)
             * len(stove_gain_range) * len(hp_lag_range))
    print(f"Sweeping {total} parameter combinations "
          f"(reset every {reset_interval} ticks = "
          f"{reset_interval * data.dt_seconds / 60:.0f} min)...")

    mask = data.active_heating
    results: list[CalibrationResult] = []
    for i, (tau, hpg, sg, stg, lag) in enumerate(
        product(tau_range, hp_gain_range, solar_gain_range,
                stove_gain_range, hp_lag_range)
    ):
        sim = run_open_loop(data, tau, hpg, sg, stg, lag,
                            reset_interval=reset_interval)
        rmse = compute_rmse(sim, data.room_temp_c, mask)
        mae = compute_mae(sim, data.room_temp_c, mask)
        results.append(CalibrationResult(tau, hpg, sg, stg, lag, rmse, mae))

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total} done, best RMSE so far: "
                  f"{min(r.rmse_c for r in results):.4f}°C")

    results.sort(key=lambda r: r.rmse_c)
    return results


def refine_around_best(data: AlignedData, best: CalibrationResult,
                       n_steps: int = 5,
                       reset_interval: int = 60) -> list[CalibrationResult]:
    """Fine grid around the best coarse result."""
    def _range(center, coarse_step, n=n_steps):
        half = coarse_step * 1.5
        return [center + half * (2 * i / (n - 1) - 1) for i in range(n)]

    return sweep_open_loop(
        data,
        tau_range=_range(best.tau_minutes, 10),
        hp_gain_range=[max(0.005, v) for v in _range(best.hp_gain, 0.01)],
        solar_gain_range=[max(0.0, v) for v in _range(best.solar_gain, 0.05)],
        stove_gain_range=[max(0.0, v) for v in _range(best.stove_gain, 0.05)],
        hp_lag_range=_range(best.hp_lag_minutes, 2),
        reset_interval=reset_interval,
    )


# ---------------------------------------------------------------------------
# Closed-loop validation (phase 2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model input config for closed-loop replay
# ---------------------------------------------------------------------------

# Production LR model inputs (from pi_state CSV rls_heat_coefficients).
# Map each to a column in AlignedData for feeding the PI controller.
# "data_key" is the AlignedData attribute name; "name" must match RLS.

LIVING_ROOM_MODEL_INPUTS = [
    {"name": "Pellet Stove",      "data_key": "pellet_stove",  "seed_heat": -8.0, "typical_value": 0.5},
    {"name": "Solar Proxy",       "data_key": "solar_proxy",   "seed_heat": -8.0, "typical_value": 0.5},
    {"name": "Outdoor Temp Rate", "data_key": "outdoor_rate",  "seed_heat": 0.7,  "typical_value": 0.5},
    {"name": "Living Room Boiler","data_key": None,            "seed_heat": 0.0,  "typical_value": 0.5},
    {"name": "Dining Room Boiler","data_key": None,            "seed_heat": 0.0,  "typical_value": 0.5},
    {"name": "Sunroom Boiler",    "data_key": None,            "seed_heat": 0.0,  "typical_value": 0.5},
]


def run_closed_loop(data: AlignedData,
                    tau_minutes: float,
                    hp_gain: float,
                    solar_gain: float,
                    stove_gain: float,
                    hp_lag_minutes: float = 5.0,
                    model_input_configs: list[dict] | None = None,
                    pi_overrides: dict | None = None,
                    ) -> dict:
    """Run PI controller through calibrated room model.

    Args:
        model_input_configs: List of dicts, each with:
            - name: matches production RLS coefficient name
            - data_key: AlignedData attribute name (or None to skip feeding)
            - seed_heat: initial coefficient value
            - typical_value: feature scale for RLS normalization
        pi_overrides: Extra config overrides for the TasmotaPIAdapter.

    Returns dict with sim_room_temp, sim_hp_setpoint, and metrics.
    """
    # Import here to avoid hard dependency for open-loop-only runs
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tests.hvac_bench.adapters import TasmotaPIAdapter
    from tests.hvac_bench.house_profiles import HouseProfile
    from tests.hvac_bench.thermal_model import ThermalModel

    profile = HouseProfile(
        name="calibrated_living_room",
        tau_minutes=tau_minutes,
        hp_gain=hp_gain,
        solar_gain=solar_gain,
        stove_gain=stove_gain,
    )

    model = ThermalModel(
        profile=profile,
        initial_temp=data.room_temp_c[0],
        outdoor_temp=data.outdoor_c[0],
        sensor_noise_sigma=0.055,  # σ=0.098°C measured, but halved for sim
        sensor_quantization=0.01,  # 0.02°F ≈ 0.011°C
        noise_seed=42,
        hp_lag_minutes=hp_lag_minutes,
    )

    # Build PI model inputs config for the adapter
    mi_configs = model_input_configs or []
    pi_model_inputs = []
    for mi in mi_configs:
        pi_model_inputs.append({
            "name": mi["name"],
            "entity_id": f"sensor.fake_{mi['name'].lower().replace(' ', '_')}",
            "seed_heat": mi.get("seed_heat", 0.0),
            "seed_cool": mi.get("seed_cool", 0.0),
            "typical_value": mi.get("typical_value", 0.5),
        })

    config = {"pi_ki": 0.15}
    if pi_model_inputs:
        config["pi_model_inputs"] = pi_model_inputs
    if pi_overrides:
        config.update(pi_overrides)

    adapter = TasmotaPIAdapter(config_overrides=config)

    # Inject production RLS coefficients
    rls = adapter._pi._rls_heat
    coeffs = data.initial_rls_coefficients or {}

    # Build the coefficient name list matching RLS index order
    coeff_names = ["intercept", "outdoor_delta"]
    for mi in adapter._pi._model_inputs:
        coeff_names.append(mi.get("name", ""))

    print(f"RLS structure: {rls.n} coefficients, names: {coeff_names}")
    for i, name in enumerate(coeff_names):
        phys_val = coeffs.get(name, 0.0)
        rls.beta[i] = phys_val * rls.feature_scales[i]
        print(f"  [{i}] {name}: phys={phys_val:.4f}, "
              f"scale={rls.feature_scales[i]:.1f}, "
              f"beta_norm={rls.beta[i]:.4f}")
    rls.observation_count = 183

    # Inject initial PI state from production
    adapter._pi._pi_integral = data.initial_integral
    adapter._pi._hp_setpoint = data.hp_setpoint_c[0]
    # Disable hold timer so setpoint can update freely each tick
    adapter.set_hold_time(0)
    # Monkey-patch _read_model_input_values to no-op — the adapter
    # sets values before the tick, but the PI tick would overwrite them
    # by reading from (non-existent) HA entities.
    adapter._pi._read_model_input_values = lambda: None
    print(f"Initial integral: {data.initial_integral:.3f}, "
          f"HP setpoint: {data.hp_setpoint_c[0]:.0f}°C")

    # Set initial desired temp
    adapter.set_desired_temp(data.desired_c[0])

    dt_seconds = data.dt_seconds
    sim_room_temps: list[float] = []
    sim_hp_setpoints: list[float] = []

    for i in range(data.n_ticks):
        model.outdoor_temp = data.outdoor_c[i]
        adapter.set_desired_temp(data.desired_c[i])
        sensor = model.read_sensor()

        # Build model inputs from data for each configured input
        tick_inputs = {}
        for mi in mi_configs:
            data_key = mi.get("data_key")
            if data_key and hasattr(data, data_key):
                tick_inputs[mi["name"]] = getattr(data, data_key)[i]

        hp_sp = adapter.tick(
            room_temp_c=sensor,
            outdoor_temp_c=data.outdoor_c[i],
            dt_seconds=dt_seconds,
            model_inputs=tick_inputs if tick_inputs else None,
        )

        model.step(
            hp_setpoint=hp_sp,
            dt_minutes=dt_seconds / 60.0,
            solar_proxy=data.solar_proxy[i],
            stove_active=data.pellet_stove[i],
            tick=i,
            mode="heat",
        )

        sim_room_temps.append(model.room_temp)
        sim_hp_setpoints.append(hp_sp)

    # Metrics
    mask = data.active_heating
    room_rmse = compute_rmse(sim_room_temps, data.room_temp_c)
    room_rmse_active = compute_rmse(sim_room_temps, data.room_temp_c, mask)
    room_mae = compute_mae(sim_room_temps, data.room_temp_c)

    sp_match = sum(
        1 for s, p in zip(sim_hp_setpoints, data.hp_setpoint_c)
        if abs(s - p) < 0.5
    ) / data.n_ticks * 100

    return {
        "sim_room_temp": sim_room_temps,
        "sim_hp_setpoint": sim_hp_setpoints,
        "room_rmse_c": room_rmse,
        "room_rmse_active_c": room_rmse_active,
        "room_mae_c": room_mae,
        "setpoint_match_pct": sp_match,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(data: AlignedData, best: CalibrationResult,
                 closed_loop: dict | None = None) -> None:
    """Print calibration results."""
    print("\n" + "=" * 60)
    print("ROOM MODEL CALIBRATION RESULTS")
    print("=" * 60)

    print(f"\nData: {data.n_ticks} ticks @ {data.dt_seconds}s = "
          f"{data.n_ticks * data.dt_seconds / 3600:.1f} hours")
    print(f"Production room temp range: "
          f"{min(data.room_temp_c):.2f} to {max(data.room_temp_c):.2f} °C "
          f"({min(data.room_temp_c) * 9/5 + 32:.1f} to "
          f"{max(data.room_temp_c) * 9/5 + 32:.1f} °F)")

    print(f"\n── Open-Loop Calibration ──")
    print(f"  tau_minutes:    {best.tau_minutes:.1f}")
    print(f"  hp_gain:        {best.hp_gain:.4f}")
    print(f"  solar_gain:     {best.solar_gain:.4f}")
    print(f"  stove_gain:     {best.stove_gain:.4f}")
    print(f"  hp_lag_minutes: {best.hp_lag_minutes:.1f}")
    print(f"  RMSE:           {best.rmse_c:.4f} °C ({best.rmse_c * 9/5:.4f} °F)")
    print(f"  MAE:            {best.mae_c:.4f} °C ({best.mae_c * 9/5:.4f} °F)")

    print(f"\n── HouseProfile for bench tests ──")
    print(f'  HouseProfile(')
    print(f'      name="living_room_calibrated",')
    print(f'      tau_minutes={best.tau_minutes:.1f},')
    print(f'      hp_gain={best.hp_gain:.4f},')
    print(f'      solar_gain={best.solar_gain:.4f},')
    print(f'      stove_gain={best.stove_gain:.4f},')
    print(f'  )')

    if closed_loop:
        print(f"\n── Closed-Loop Validation ──")
        print(f"  Room temp RMSE (all):    {closed_loop['room_rmse_c']:.4f} °C")
        print(f"  Room temp RMSE (active): {closed_loop.get('room_rmse_active_c', 0):.4f} °C")
        print(f"  Room temp MAE:           {closed_loop['room_mae_c']:.4f} °C")
        print(f"  HP setpoint match:       {closed_loop['setpoint_match_pct']:.1f}%")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_comparison(data: AlignedData, sim_temps: list[float],
                      out_path: Path,
                      closed_loop: dict | None = None) -> None:
    """Export production vs sim comparison to CSV for plotting."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        headers = [
            "tick", "minutes",
            "prod_room_c", "sim_room_c", "error_c",
            "prod_hp_setpoint_c", "outdoor_c",
            "solar_proxy", "pellet_stove", "outdoor_rate",
            "prod_integral", "prod_ff_offset",
        ]
        if closed_loop:
            headers.extend(["cl_room_c", "cl_hp_setpoint_c"])
        w.writerow(headers)

        for i in range(data.n_ticks):
            row = [
                i,
                i * data.dt_seconds / 60.0,
                f"{data.room_temp_c[i]:.4f}",
                f"{sim_temps[i]:.4f}",
                f"{sim_temps[i] - data.room_temp_c[i]:.4f}",
                f"{data.hp_setpoint_c[i]:.1f}",
                f"{data.outdoor_c[i]:.2f}",
                f"{data.solar_proxy[i]:.4f}",
                f"{data.pellet_stove[i]:.0f}",
                f"{data.outdoor_rate[i]:.4f}",
                f"{data.pi_integral[i]:.3f}",
                f"{data.ff_offset[i]:.3f}",
            ]
            if closed_loop:
                row.extend([
                    f"{closed_loop['sim_room_temp'][i]:.4f}",
                    f"{closed_loop['sim_hp_setpoint'][i]:.1f}",
                ])
            w.writerow(row)

    print(f"\nExported comparison to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate room model from debug bundle data")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Path to debug bundle directory")
    parser.add_argument("--zone", default="living_room",
                        help="Zone name (default: living_room)")
    parser.add_argument("--dt", type=float, default=60.0,
                        help="Tick interval in seconds (default: 60)")
    parser.add_argument("--skip-closed-loop", action="store_true",
                        help="Skip closed-loop validation")
    parser.add_argument("--export", type=Path, default=None,
                        help="Export comparison CSV")
    args = parser.parse_args()

    print(f"Loading data from {args.bundle} zone={args.zone}...")
    data = align_data(args.bundle, args.zone, args.dt)
    print(f"Aligned {data.n_ticks} ticks over "
          f"{data.n_ticks * data.dt_seconds / 3600:.1f} hours")
    print(f"Initial RLS coefficients: {data.initial_rls_coefficients}")

    # Phase 1: Coarse sweep with 2-hour reset windows
    # Scored only on ticks where HP is actively heating (setpoint > desired)
    reset_ticks = 120  # 120 min at 60s ticks
    print(f"\n── Phase 1: Coarse parameter sweep (reset every {reset_ticks} min, active-heating only) ──")
    results = sweep_open_loop(
        data, reset_interval=reset_ticks,
        # Range centered on expected values from physics:
        # tau: 200-700 for residential (varies by insulation/mass)
        # hp_gain: 0.004-0.02 for mini-split heads
        tau_range=[200, 250, 300, 350, 400, 500, 600, 700],
        hp_gain_range=[0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02],
    )
    best = results[0]
    print(f"\nBest coarse: RMSE={best.rmse_c:.4f}°C, "
          f"tau={best.tau_minutes}, hp_gain={best.hp_gain}, "
          f"solar={best.solar_gain}, stove={best.stove_gain}, "
          f"lag={best.hp_lag_minutes}")

    # Phase 1b: Refine
    print("\n── Phase 1b: Refinement sweep ──")
    refined = refine_around_best(data, best, reset_interval=reset_ticks)
    best = refined[0]
    print(f"\nBest refined: RMSE={best.rmse_c:.4f}°C, "
          f"tau={best.tau_minutes}, hp_gain={best.hp_gain}, "
          f"solar={best.solar_gain}, stove={best.stove_gain}, "
          f"lag={best.hp_lag_minutes}")

    # Top 5 for inspection
    print("\n── Top 5 results ──")
    for i, r in enumerate(refined[:5]):
        print(f"  {i+1}. RMSE={r.rmse_c:.4f}°C  tau={r.tau_minutes:.1f} "
              f"hp_gain={r.hp_gain:.4f} solar={r.solar_gain:.4f} "
              f"stove={r.stove_gain:.4f} lag={r.hp_lag_minutes:.1f}")

    # Generate sim with best params — both reset and free-running
    sim_temps = run_open_loop(
        data, best.tau_minutes, best.hp_gain,
        best.solar_gain, best.stove_gain, best.hp_lag_minutes,
        reset_interval=reset_ticks)
    sim_free = run_open_loop(
        data, best.tau_minutes, best.hp_gain,
        best.solar_gain, best.stove_gain, best.hp_lag_minutes,
        reset_interval=0)

    mask = data.active_heating
    print(f"\nActive-heating RMSE (reset):  {compute_rmse(sim_temps, data.room_temp_c, mask):.4f}°C")
    print(f"All-ticks RMSE (reset):       {compute_rmse(sim_temps, data.room_temp_c):.4f}°C")
    print(f"Free-running RMSE:            {compute_rmse(sim_free, data.room_temp_c):.4f}°C")

    # Phase 2: Closed-loop
    closed_loop = None
    if not args.skip_closed_loop:
        print("\n── Phase 2: Closed-loop validation ──")
        try:
            closed_loop = run_closed_loop(
                data, best.tau_minutes, best.hp_gain,
                best.solar_gain, best.stove_gain, best.hp_lag_minutes,
                model_input_configs=LIVING_ROOM_MODEL_INPUTS)
        except Exception as e:
            print(f"Closed-loop failed: {e}")
            import traceback
            traceback.print_exc()
            print("Run with --skip-closed-loop to skip this phase.")

    # Report
    print_report(data, best, closed_loop)

    # Export
    export_path = args.export or (args.bundle / f"calibration_{args.zone}.csv")
    export_comparison(data, sim_temps, export_path, closed_loop)


if __name__ == "__main__":
    main()
