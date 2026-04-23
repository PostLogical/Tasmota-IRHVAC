#!/usr/bin/env python3
"""Analyze seasonal patterns in LTS data to validate grey-box buffer stratification.

Reads local/data/pi_replay_lts.json and produces:
1. Outdoor temp distribution per calendar quarter (Q1-Q4)
2. Solar radiation distribution per quarter
3. Room temp drift rates per quarter (proxy for ua_c signal)
4. Overlap analysis: do quarters capture meaningfully different regimes?
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "local" / "data" / "pi_replay_lts.json"


def quarter_of(iso_ts: str) -> int:
    """Map ISO timestamp to quarter 1-4."""
    # Handle both '+00:00' and '+00:00' style offsets
    dt = datetime.fromisoformat(iso_ts)
    return (dt.month - 1) // 3 + 1


def month_of(iso_ts: str) -> int:
    return datetime.fromisoformat(iso_ts).month


def load_sensor(stats: dict, key: str) -> list[tuple[str, float]]:
    """Extract (timestamp, mean) pairs, skipping nulls."""
    result = []
    for entry in stats.get(key, []):
        val = entry.get("mean")
        if val is not None:
            result.append((entry["t"], float(val)))
    return result


def compute_hourly_rate(data: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Compute hourly dT/dt in °C/hr from consecutive hourly readings."""
    rates = []
    for i in range(1, len(data)):
        t_prev = datetime.fromisoformat(data[i - 1][0])
        t_curr = datetime.fromisoformat(data[i][0])
        dt_hours = (t_curr - t_prev).total_seconds() / 3600
        if 0.5 < dt_hours < 2.0:  # only consecutive hours
            rate = (data[i][1] - data[i - 1][1]) / dt_hours
            rates.append((data[i][0], rate))
    return rates


def stats_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = sum(values_sorted) / n
    variance = sum((v - mean) ** 2 for v in values_sorted) / n
    std = variance ** 0.5
    return {
        "n": n,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": round(values_sorted[0], 2),
        "p10": round(values_sorted[int(n * 0.1)], 2),
        "p25": round(values_sorted[int(n * 0.25)], 2),
        "median": round(values_sorted[n // 2], 2),
        "p75": round(values_sorted[int(n * 0.75)], 2),
        "p90": round(values_sorted[int(n * 0.9)], 2),
        "max": round(values_sorted[-1], 2),
    }


def overlap_coefficient(vals_a: list[float], vals_b: list[float], bins: int = 50) -> float:
    """Compute overlap coefficient between two distributions (0=disjoint, 1=identical)."""
    if not vals_a or not vals_b:
        return 0.0
    lo = min(min(vals_a), min(vals_b))
    hi = max(max(vals_a), max(vals_b))
    if hi == lo:
        return 1.0
    bin_width = (hi - lo) / bins
    hist_a = [0.0] * bins
    hist_b = [0.0] * bins
    for v in vals_a:
        idx = min(int((v - lo) / bin_width), bins - 1)
        hist_a[idx] += 1.0 / len(vals_a)
    for v in vals_b:
        idx = min(int((v - lo) / bin_width), bins - 1)
        hist_b[idx] += 1.0 / len(vals_b)
    return sum(min(a, b) for a, b in zip(hist_a, hist_b))


def main():
    print(f"Loading {DATA_PATH} ...")
    with open(DATA_PATH) as f:
        data = json.load(f)
    stats = data["stats"]

    # --- Outdoor temperature ---
    outdoor = load_sensor(stats, "sensor.openmeteo_archive_air_temperature")
    print(f"\nOutdoor temperature: {len(outdoor)} hourly readings")
    print(f"  Date range: {outdoor[0][0][:10]} to {outdoor[-1][0][:10]}")

    outdoor_by_q: dict[int, list[float]] = defaultdict(list)
    outdoor_by_month: dict[int, list[float]] = defaultdict(list)
    for ts, val in outdoor:
        outdoor_by_q[quarter_of(ts)].append(val)
        outdoor_by_month[month_of(ts)].append(val)

    print("\n=== OUTDOOR TEMP BY CALENDAR QUARTER (°C) ===")
    q_names = {1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)", 3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"}
    for q in [1, 2, 3, 4]:
        s = stats_summary(outdoor_by_q[q])
        print(f"  {q_names[q]}: {s}")

    print("\n=== OUTDOOR TEMP BY MONTH (°C) ===")
    m_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    for m in range(1, 13):
        s = stats_summary(outdoor_by_month[m])
        print(f"  {m_names[m]:>3}: n={s['n']:>5}  mean={s.get('mean',''):>6}  std={s.get('std',''):>5}  "
              f"range=[{s.get('min',''):>6}, {s.get('max',''):>5}]")

    print("\n=== QUARTER OVERLAP (outdoor temp distributions) ===")
    print("  Overlap coefficient: 0=completely different, 1=identical")
    for i in [1, 2, 3, 4]:
        for j in range(i + 1, 5):
            oc = overlap_coefficient(outdoor_by_q[i], outdoor_by_q[j])
            print(f"  {q_names[i]} vs {q_names[j]}: {oc:.3f}")

    # --- Solar radiation ---
    solar = load_sensor(stats, "sensor.openmeteo_archive_solar_radiation")
    if solar:
        solar_by_q: dict[int, list[float]] = defaultdict(list)
        for ts, val in solar:
            solar_by_q[quarter_of(ts)].append(val)

        print("\n=== SOLAR RADIATION BY QUARTER (W/m²) ===")
        for q in [1, 2, 3, 4]:
            s = stats_summary(solar_by_q[q])
            print(f"  {q_names[q]}: mean={s.get('mean',''):>6}  std={s.get('std',''):>6}  "
                  f"max={s.get('max',''):>6}  n={s['n']}")

    # --- Room temp drift rates ---
    room_sensors = [
        ("Living Room", "sensor.living_room_air_sensor_temperature"),
        ("Kitchen", "sensor.kitchen_air_sensor_temperature"),
        ("Nursery", "sensor.nursery_air_sensor_temperature"),
        ("Bedroom", "sensor.bedroom_air_sensor_temperature"),
    ]

    print("\n=== ROOM TEMP DRIFT RATES BY QUARTER (°C/hr) ===")
    print("  (Proxy for thermal envelope signal — larger |drift| = more grey-box information)")
    for name, sensor_key in room_sensors:
        room_data = load_sensor(stats, sensor_key)
        if not room_data:
            continue
        rates = compute_hourly_rate(room_data)
        rates_by_q: dict[int, list[float]] = defaultdict(list)
        for ts, rate in rates:
            rates_by_q[quarter_of(ts)].append(rate)

        print(f"\n  {name}:")
        for q in [1, 2, 3, 4]:
            s = stats_summary(rates_by_q[q])
            if s["n"] > 0:
                # Also compute % of readings with |rate| > 0.2°C/hr (significant drift)
                significant = sum(1 for r in rates_by_q[q] if abs(r) > 0.2)
                pct = 100 * significant / s["n"]
                print(f"    {q_names[q]}: mean={s.get('mean',''):>6}  std={s.get('std',''):>5}  "
                      f"|rate|>0.2: {pct:>4.1f}%  n={s['n']}")

    # --- Heating vs non-heating regime estimate ---
    print("\n=== THERMAL REGIME ANALYSIS ===")
    print("  Estimating heating/cooling/shoulder based on outdoor temp thresholds")
    print("  Heating-dominant: outdoor < 15°C (~59°F)")
    print("  Cooling-dominant: outdoor > 24°C (~75°F)")
    print("  Shoulder: 15-24°C")

    regime_by_q: dict[int, dict[str, int]] = defaultdict(lambda: {"heating": 0, "cooling": 0, "shoulder": 0})
    for ts, val in outdoor:
        q = quarter_of(ts)
        if val < 15:
            regime_by_q[q]["heating"] += 1
        elif val > 24:
            regime_by_q[q]["cooling"] += 1
        else:
            regime_by_q[q]["shoulder"] += 1

    for q in [1, 2, 3, 4]:
        r = regime_by_q[q]
        total = sum(r.values())
        if total > 0:
            print(f"  {q_names[q]}: heating={100*r['heating']/total:.0f}%  "
                  f"shoulder={100*r['shoulder']/total:.0f}%  "
                  f"cooling={100*r['cooling']/total:.0f}%")

    # --- Recommendation ---
    print("\n=== STRATIFICATION RECOMMENDATION ===")
    # Check if Q2 and Q4 are similar (shoulder seasons)
    q2_q4_overlap = overlap_coefficient(outdoor_by_q[2], outdoor_by_q[4])
    q1_q3_overlap = overlap_coefficient(outdoor_by_q[1], outdoor_by_q[3])

    if q2_q4_overlap > 0.6:
        print(f"  Q2 and Q4 overlap significantly ({q2_q4_overlap:.2f}) — they're both shoulder seasons.")
        print("  Consider: 3-season split (heating/shoulder/cooling) instead of 4 quarters.")
    else:
        print(f"  Q2 and Q4 are distinct ({q2_q4_overlap:.2f}) — calendar quarters work well.")

    if q1_q3_overlap < 0.2:
        print(f"  Q1 and Q3 are very different ({q1_q3_overlap:.2f}) — good separation between winter and summer.")

    # Check for strong asymmetry in data availability
    q_counts = {q: len(outdoor_by_q[q]) for q in [1, 2, 3, 4]}
    min_q = min(q_counts.values())
    max_q = max(q_counts.values())
    if max_q > 2 * min_q:
        print(f"  Warning: Uneven quarter sizes ({q_counts}). "
              f"Consider adjusting buffer allocation proportionally.")

    print("\nDone.")


if __name__ == "__main__":
    main()
