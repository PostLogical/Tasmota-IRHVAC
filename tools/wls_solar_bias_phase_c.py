"""Phase C: direct buffer-dynamics observation using pre49 ObservationContext.

Phases A and B inferred mechanism indirectly from β_solar trajectories.
This phase uses the per-tick admission/eviction observability that pre49
shipped (`pi._last_observation_context`) to *watch* what the leverage policy
is actually doing.

Captures per tick (via the bench history augmentation):
- obs_admitted: was this tick's obs added to the WLS buffer?
- obs_leverage: leverage score the candidate received
- obs_evicted_ts: monotonic timestamp of any obs displaced by this admission
- obs_min_incumbent_lev: min leverage of incumbents at decision time
- obs_rejection_reason: "low_leverage" or None

End-of-run: dumps the live buffer (solar value, ToD, outdoor_delta per obs).

Analyses:
- Solar distribution: buffer vs population
- ToD distribution: buffer vs population
- Eviction rate over time
- Rejection reasons over time

Reading guide:
- If buffer's high-solar fraction is much lower than population's → eviction
  is systematically shedding solar-rich obs. Confirms the buffer is the bias
  source even with raw-leverage.
- If buffer's distribution matches population's but rejection rate is high
  during midday → leverage scoring is rejecting solar candidates at admission
  rather than evicting them later.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any


def main() -> int:
    logging.getLogger("custom_components.tasmota_irhvac").setLevel(logging.ERROR)

    from tests.hvac_bench.adapters import TasmotaPIAdapter
    from tests.hvac_bench.full_stack_runner import run_full_stack
    from tests.hvac_bench.scenarios.test_seasonal_convergence import (
        _make_real_config,
    )

    captured: dict[str, Any] = {}
    orig_init = TasmotaPIAdapter.__init__

    def patched(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        captured["pi"] = self._pi

    TasmotaPIAdapter.__init__ = patched
    try:
        config = _make_real_config("spring", n_days=90)
        mi = config.model_inputs[0]
        solar_eid = mi.entity_id

        print("=" * 78)
        print("Phase C: direct buffer-dynamics observation (spring 90d, raw-leverage)")
        print("=" * 78)
        print(f"Truth: β_solar = {mi._true_ff_coef:.2f}, lag_tau = {mi.lag_tau} s")
        print()

        result = run_full_stack(config)
        pi = captured["pi"]
    finally:
        TasmotaPIAdapter.__init__ = orig_init

    print("Final β_solar:", result.final_coefs.get(mi.name, float("nan")))
    print()

    history = result.history

    # ── Tick population: all ticks that had an obs admission attempt ──
    # An admission attempt happened when obs_admitted is not None (i.e., the
    # controller called buf.add). Filter to that population.
    attempts = [h for h in history if h.get("obs_admitted") is not None]
    if not attempts:
        print("No admission attempts recorded — observability plumbing missing?")
        return 1
    print(f"Total ticks: {len(history)}")
    print(f"Admission attempts (filtered through controller eligibility): {len(attempts)}")

    admitted = [h for h in attempts if h["obs_admitted"]]
    rejected = [h for h in attempts if not h["obs_admitted"]]
    evicted = [h for h in attempts if h.get("obs_evicted_ts") is not None]
    print(f"  admitted               : {len(admitted)} ({100 * len(admitted) / len(attempts):.1f}%)")
    print(f"  rejected (low_leverage): {len(rejected)} ({100 * len(rejected) / len(attempts):.1f}%)")
    print(f"  caused eviction        : {len(evicted)} ({100 * len(evicted) / len(attempts):.1f}%)")

    # ── ToD-binned analysis: admission/rejection by hour ──
    # Bench tick is 15 min, so hour = (tick * 15) / 60 mod 24.
    print()
    print("Hourly admission rate (ticks where buf.add was called):")
    print(f"  {'hour':>4} {'attempts':>9} {'admitted':>9} {'evictions':>10} {'rejected':>9} "
          f"{'mean solar':>11}")
    print("  " + "-" * 64)
    by_hour: dict[int, list[dict]] = {h: [] for h in range(24)}
    for h in attempts:
        hour = int((h["tick"] * 15) / 60) % 24
        by_hour[hour].append(h)
    solar_key = f"input_{mi.name}"
    for hour in range(24):
        bucket = by_hour[hour]
        if not bucket:
            continue
        adm = sum(1 for x in bucket if x["obs_admitted"])
        evt = sum(1 for x in bucket if x.get("obs_evicted_ts") is not None)
        rej = sum(1 for x in bucket if not x["obs_admitted"])
        solars = [x.get(solar_key, 0.0) or 0.0 for x in bucket]
        mean_solar = sum(solars) / len(solars) if solars else 0.0
        print(f"  {hour:>4d} {len(bucket):>9d} {adm:>9d} {evt:>10d} {rej:>9d} "
              f"{mean_solar:>11.4f}")

    # ── Solar-bin analysis: eviction by solar value ──
    print()
    print("Solar-bin: rejection rate at admission (was the candidate too "
          "low-leverage to enter?)")
    print(f"  {'bin (solar)':<14} {'attempts':>9} {'admitted':>9} {'rejected':>9} "
          f"{'evicted':>9}")
    print("  " + "-" * 60)
    bins = [(0.0, 0.0001, "= 0"), (0.0001, 0.05, "(0, 0.05]"),
            (0.05, 0.15, "(0.05, 0.15]"), (0.15, 0.30, "(0.15, 0.30]"),
            (0.30, 0.50, "(0.30, 0.50]"), (0.50, 1.01, "> 0.50")]
    for lo, hi, label in bins:
        bucket = [x for x in attempts
                  if lo <= (x.get(solar_key, 0.0) or 0.0) < hi]
        if not bucket:
            continue
        adm = sum(1 for x in bucket if x["obs_admitted"])
        evt = sum(1 for x in bucket if x.get("obs_evicted_ts") is not None)
        rej = sum(1 for x in bucket if not x["obs_admitted"])
        print(f"  {label:<14} {len(bucket):>9d} {adm:>9d} {rej:>9d} {evt:>9d}")

    # ── Time-windowed eviction rate ──
    print()
    print("Eviction rate by 10-day window:")
    print(f"  {'days':<10} {'attempts':>9} {'evictions':>10} {'rate':>7}")
    print("  " + "-" * 40)
    ticks_per_day = 4 * 24  # 15-min ticks
    n_days = 90
    for d_start in range(0, n_days, 10):
        d_end = min(d_start + 10, n_days)
        t_lo, t_hi = d_start * ticks_per_day, d_end * ticks_per_day
        bucket = [x for x in attempts if t_lo <= x["tick"] < t_hi]
        if not bucket:
            continue
        evt = sum(1 for x in bucket if x.get("obs_evicted_ts") is not None)
        rate = evt / len(bucket) if bucket else 0.0
        print(f"  {d_start:>3d}-{d_end:<3d}    {len(bucket):>9d} "
              f"{evt:>10d} {rate:>7.1%}")

    # ── End-of-run buffer composition ──
    print()
    buf = pi._observation_buffer_heat
    obs_list = buf.get_all()
    print(f"End-of-run buffer (heat): {len(obs_list)} observations "
          f"(max_size={buf._max_size})")
    if obs_list:
        solar_vals = [o.raw_readings.get(solar_eid, 0.0) for o in obs_list]
        # Hour from wall_time (assume seconds since epoch-like; bench uses
        # tick * 15min from t=0)
        hours = [(o.wall_time / 3600.0) % 24.0 for o in obs_list]
        print()
        print("Buffer solar value distribution:")
        print(f"  {'bin':<14} {'count':>7} {'%':>6}")
        for lo, hi, label in bins:
            cnt = sum(1 for v in solar_vals if lo <= v < hi)
            pct = 100 * cnt / len(solar_vals)
            print(f"  {label:<14} {cnt:>7d} {pct:>5.1f}%")

        print()
        print("Buffer ToD distribution (4-hour bins):")
        print(f"  {'hours':<10} {'count':>7} {'%':>6}")
        tod_bins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
        for lo, hi in tod_bins:
            cnt = sum(1 for h in hours if lo <= h < hi)
            pct = 100 * cnt / len(hours)
            print(f"  {lo:>2d}-{hi:<2d}     {cnt:>7d} {pct:>5.1f}%")

        # Population for comparison
        print()
        print("POPULATION (admission attempts) for comparison:")
        all_solars = [x.get(solar_key, 0.0) or 0.0 for x in attempts]
        all_hours = [(x["tick"] * 15 / 60.0) % 24.0 for x in attempts]
        print(f"  Solar bins:")
        for lo, hi, label in bins:
            buf_pct = 100 * sum(1 for v in solar_vals if lo <= v < hi) / len(solar_vals)
            pop_pct = 100 * sum(1 for v in all_solars if lo <= v < hi) / len(all_solars)
            delta = buf_pct - pop_pct
            arrow = " ↑" if delta > 1 else (" ↓" if delta < -1 else "  ")
            print(f"    {label:<14} buf {buf_pct:>5.1f}%  pop {pop_pct:>5.1f}%  "
                  f"Δ {delta:>+5.1f}{arrow}")
        print(f"  Hour bins:")
        for lo, hi in tod_bins:
            buf_pct = 100 * sum(1 for h in hours if lo <= h < hi) / len(hours)
            pop_pct = 100 * sum(1 for h in all_hours if lo <= h < hi) / len(all_hours)
            delta = buf_pct - pop_pct
            arrow = " ↑" if delta > 1 else (" ↓" if delta < -1 else "  ")
            print(f"    {lo:>2d}-{hi:<2d}     buf {buf_pct:>5.1f}%  pop {pop_pct:>5.1f}%  "
                  f"Δ {delta:>+5.1f}{arrow}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
