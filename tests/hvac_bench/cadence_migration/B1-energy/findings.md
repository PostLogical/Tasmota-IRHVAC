# B1 — Energy tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-10 · **Files:** `tests/hvac_bench/scenarios/test_energy.py`

## Outcome summary

9 tests, all passing at all phases. 0 status flips.

| Run | Tick cadence | Result |
|---|---|---|
| Phase 1 baseline (Pass A) | 15 min | 9/9 passed (7s) |
| Phase 2 (refactored, Pass B) | 15 min | 9/9 passed; deltas at 10^-5-10^-6 magnitude — float-noise from `dt_factor=1.0` multiplications (q-feedback + leaky decay) |
| Phase 3 | 3 min | 9/9 passed (18s) |

Both diffs (Pass A→B and 15-min→3-min) clean: 0 status flips.

## Notable energy-specific findings

- **`cop_mean` shifts ≤1.4%** at 3-min (e.g. `test_energy_accumulates`: 2.61 → 2.65 +1.4%). At higher cadence the controller settles faster → less time at low setpoints during transient → slightly higher COP.
- **`final_cumulative_kwh` shifts ≤4.4%** (e.g. `test_overseed_wastes_energy[standard_residential]`: kwh_correct_seed 0.732 → 0.700 -4.4%). Same dynamics: faster cadence → faster convergence → less wasted heat during transient.
- **`kwh_delta_overseed_vs_correct` flipped sign** at 3-min: -0.008 → +0.028 kWh (correct-seed used MORE at 15-min, over-seed uses MORE at 3-min). At 15-min the under-integration of the correctly-seeded controller (q-feedback dominance) made it less efficient than at 3-min where the integral builds properly. The sign-flip is a real cadence-sensitivity finding worth investigating but doesn't break any contract.

## Observation: ITAE-as-defined is not cadence-portable

`tests/benchmark_metrics.py:117` defines:
```python
itae += h["tick"] * effective_error
```

This multiplies error by **tick index**, not wall-clock time. Three problems:

1. **Cross-cadence comparisons are meaningless.** Same physical scenario at 15-min has weights 1+2+...+24 = 300; at 3-min has weights 1+2+...+120 = 7260. **24× more weight at 3-min** for the same trajectory. Explains the 8000-30000% ITAE deltas in 3-min vs 15-min diffs.
2. **Variable production cadence breaks ITAE entirely.** Production ticks are sensor-driven, ~60s typical but ranging from seconds (during fast temp changes) to 15-min (sensor stale fallback). Tick index in production reflects "how many sensor updates" not "how much time" — so a stable system might tick 50× in an hour while a chattery one ticks 200×. ITAE values have no physical meaning when ticks are variable.
3. **Even within a single fixed-cadence sim, the design intent of ITAE (penalize late errors more than early ones) is realized as "penalize errors at high tick numbers more"** — which only matches "penalize late errors more" if cadence is fixed.

Recommended fix (separate from this migration): replace with **time-weighted IAE** in degree-minutes:
```python
itae += h["minute"] * abs(h["error"]) * tick_minutes
```
Or for energy-design intent, normalize by total_minutes:
```python
itae /= total_minutes
```

Until this lands, ITAE values in `bench_metrics` should be interpreted only within-phase, not across cadences. Other rollup metrics with similar issues (already documented in B0/B1-cooling findings):
- `rollup_settling_time` (in ticks)
- `rollup_kwh_per_degree_hour` (cadence-dependent denominator)
- `rollup_reversals` / `rollup_setpoint_changes` (event counts, not rates)

## What changed in code

- `tests/hvac_bench/scenarios/test_energy.py`: bench_metrics wired into all 5 test methods; converted to wall-clock API (`duration_minutes`); `_make_model` helper added with `hp_lag_minutes=2.0` default.

No production changes for this batch.
