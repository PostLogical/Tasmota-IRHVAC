"""CSV and data adapters for full-stack simulation weather/input schedules.

Primary format: simple CSV with timestamp + named columns.
Adapters convert from HA history JSON, debug bundles, and open-meteo.

All adapters return schedule dicts: {field_name: callable(minute) -> value}
compatible with FullStackConfig.outdoor_schedule / model input schedules.
``minute`` is sim-minutes from the data's first timestamp.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


# ── Simple CSV ───────────────────────────────────────────────────────────


def load_csv(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Load a simple CSV into timeseries dict.

    Expected format:
        timestamp_utc,outdoor_c,solar_w_m2,sunroom_delta_c,...
        2026-01-15T00:00:00,-5.2,0,-3.1
        2026-01-15T00:15:00,-5.3,0,-3.0

    Returns:
        Dict mapping column names to sorted [(epoch_seconds, value)] lists.
        The 'timestamp_utc' column is consumed as the time axis.
    """
    path = Path(path)
    result: dict[str, list[tuple[float, float]]] = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp_utc") or row.get("timestamp")
            if not ts_str:
                continue
            try:
                epoch = datetime.fromisoformat(ts_str).timestamp()
            except ValueError:
                continue

            for col, val_str in row.items():
                if col in ("timestamp_utc", "timestamp"):
                    continue
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    continue
                result.setdefault(col, []).append((epoch, val))

    # Sort by time
    for series in result.values():
        series.sort()

    return result


def csv_to_schedules(
    csv_data: dict[str, list[tuple[float, float]]],
) -> dict[str, Callable[[float], float]]:
    """Convert CSV timeseries to minute-indexed schedule callables.

    Each callable interpolates the CSV data at sim-minute ``m`` past the
    data's first timestamp. Minute 0 maps to that first timestamp.
    """
    # Find earliest timestamp across all series
    all_starts = [series[0][0] for series in csv_data.values() if series]
    if not all_starts:
        return {}
    t0 = min(all_starts)

    schedules: dict[str, Callable[[float], float]] = {}
    for name, series in csv_data.items():
        schedules[name] = _make_interpolator(series, t0)

    return schedules


def _make_interpolator(
    series: list[tuple[float, float]],
    t0: float,
) -> Callable[[float], float]:
    """Create a minute -> value interpolator from a timeseries."""
    def interpolate(minute: float) -> float:
        t = t0 + minute * 60.0
        # Binary search for bracketing interval
        if t <= series[0][0]:
            return series[0][1]
        if t >= series[-1][0]:
            return series[-1][1]

        lo, hi = 0, len(series) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if series[mid][0] <= t:
                lo = mid
            else:
                hi = mid

        t0_seg, v0 = series[lo]
        t1_seg, v1 = series[hi]
        dt = t1_seg - t0_seg
        if dt == 0:
            return v0
        frac = (t - t0_seg) / dt
        return v0 + frac * (v1 - v0)

    return interpolate


# ── Open-Meteo CSV ───────────────────────────────────────────────────────


def from_open_meteo_csv(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Convert open-meteo hourly CSV to timeseries dict.

    Open-meteo CSV format:
        time,temperature_2m,direct_radiation,...
        2026-01-15T00:00,−5.2,0,...

    Maps:
        temperature_2m -> outdoor_c
        direct_radiation -> solar_w_m2  (W/m², typically 0-1000)
    """
    path = Path(path)
    result: dict[str, list[tuple[float, float]]] = {}
    field_map = {
        "temperature_2m": "outdoor_c",
        "direct_radiation": "solar_w_m2",
        "shortwave_radiation": "solar_w_m2",  # alternate name
    }

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("time") or row.get("date")
            if not ts_str:
                continue
            try:
                epoch = datetime.fromisoformat(ts_str).timestamp()
            except ValueError:
                continue

            for src_col, dst_col in field_map.items():
                val_str = row.get(src_col)
                if val_str is None:
                    continue
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    continue
                result.setdefault(dst_col, []).append((epoch, val))

    for series in result.values():
        series.sort()

    return result


# ── HA History JSON ──────────────────────────────────────────────────────


def from_ha_history_json(
    path: str | Path,
    entity_map: dict[str, str],
    convert_f_to_c: bool = True,
) -> dict[str, list[tuple[float, float]]]:
    """Convert HA history JSON export to timeseries dict.

    Args:
        path: Path to JSON file (list of entity history arrays).
        entity_map: Maps entity_id to output field name.
            Example: {"sensor.outdoor_temp": "outdoor_c",
                      "sensor.solar_proxy": "solar_w_m2"}
        convert_f_to_c: If True, apply F→C conversion to temperature
            entities (those mapping to fields containing 'temp' or '_c').
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    # HA exports as list of entity arrays or dict keyed by entity_id
    if isinstance(data, dict):
        entities = data
    elif isinstance(data, list):
        # Each item is a list of states for one entity
        entities = {}
        for entity_states in data:
            if entity_states and isinstance(entity_states, list):
                eid = entity_states[0].get("entity_id", "")
                entities[eid] = entity_states
    else:
        return {}

    result: dict[str, list[tuple[float, float]]] = {}
    for entity_id, field_name in entity_map.items():
        states = entities.get(entity_id, [])
        is_temp = "temp" in field_name.lower() or field_name.endswith("_c")
        for record in states:
            try:
                ts = datetime.fromisoformat(
                    record.get("last_changed", record.get("last_updated", ""))
                )
                val = float(record["state"])
            except (ValueError, TypeError, KeyError):
                continue
            if convert_f_to_c and is_temp:
                val = (val - 32) * 5 / 9
            result.setdefault(field_name, []).append((ts.timestamp(), val))

    for series in result.values():
        series.sort()

    return result


# ── Debug Bundle ─────────────────────────────────────────────────────────


def from_debug_bundle(
    bundle_path: str | Path,
    zone: str,
) -> dict[str, list[tuple[float, float]]]:
    """Convert a debug bundle's CSVs to timeseries dict.

    Looks for:
        {bundle_path}/room_temp_{zone}_72h.csv  -> room_temp_c (F→C)
        {bundle_path}/outdoor_temp_72h.csv      -> outdoor_c (F→C)
        {bundle_path}/pi_state_{zone}_72h.csv   -> various PI fields

    Returns timeseries dict compatible with csv_to_schedules().
    """
    bundle = Path(bundle_path)
    result: dict[str, list[tuple[float, float]]] = {}

    # Outdoor temp
    outdoor_path = bundle / "outdoor_temp_72h.csv"
    if outdoor_path.exists():
        with open(outdoor_path) as f:
            for row in csv.DictReader(f):
                try:
                    epoch = datetime.fromisoformat(row["timestamp"]).timestamp()
                    temp_f = float(row.get("temp_f", row.get("temperature", "")))
                    temp_c = (temp_f - 32) * 5 / 9
                except (ValueError, TypeError, KeyError):
                    continue
                result.setdefault("outdoor_c", []).append((epoch, temp_c))

    # Room temp
    room_path = bundle / f"room_temp_{zone}_72h.csv"
    if room_path.exists():
        with open(room_path) as f:
            for row in csv.DictReader(f):
                try:
                    epoch = datetime.fromisoformat(row["timestamp"]).timestamp()
                    temp_f = float(row["temp_f"])
                    temp_c = (temp_f - 32) * 5 / 9
                except (ValueError, TypeError, KeyError):
                    continue
                result.setdefault("room_temp_c", []).append((epoch, temp_c))

    for series in result.values():
        series.sort()

    return result
