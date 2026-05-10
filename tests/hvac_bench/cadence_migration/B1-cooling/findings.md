# B1 — Cooling tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-10 · **Files:** `tests/hvac_bench/scenarios/test_cooling.py`

## Outcome summary

24 tests, all passing at both cadences post-q-feedback fix.

| Run | Tick cadence | Result |
|---|---|---|
| Phase 1 baseline (Pass A) | 15 min | 24/24 passed |
| Phase 2 (refactored, Pass B) | 15 min | 24/24 passed (8.6e-06°C float-noise drift vs Phase 1) |
| Phase 3 | 3 min | 24/24 passed |

Status flips between phases: **0**. The cadence change introduces real metric drift but no test contracts break.

## What this batch surfaced — beyond the cooling tests

The cooling steady_state cases revealed a **real production-side cadence-dependence bug** in the controller's q-feedback (quantization-error feedback) mechanism. Q-feedback's integral nudge fired the same fixed amount per tick regardless of sample period, so its per-wall-clock effective gain scaled with cadence — 5× too strong at 3-min, 15× too strong at 60s production cadence vs the 15-min cadence at which gain=0.4 was tuned.

Fix landed in commit `ea1cc50` (`pi: scale q-feedback by dt_factor for cadence-invariance`) before completing this batch's findings doc. Without that fix, the 3-min cooling steady_state tests would have shown limit cycles (HP setpoint flipping 21↔22 every 30-60 min) caused by q-feedback over-suppressing integral build below the level needed to compensate the under-seeded FF.

See [`tests/hvac_bench/cadence_migration/B0-heating/findings.md`](../B0-heating/findings.md) for the cadence-fidelity story; this batch contributes the controller-side fix that batch alone didn't surface.

## Cooling-specific patterns at 3-min

Direction of metric changes 3-min vs 15-min (post-q-fix):

| Pattern | Affected tests | Direction |
|---|---|---|
| Limit cycle eliminated | `test_heat_wave[1.0-standard_residential]`, `test_heat_wave[1.0-well_insulated]` (13→0, 10→0 reversals) | **Major improvement** |
| Reduced cycling | `test_warm_start[1.5-*]` (under-seeded cooling cases) | **Improvement** |
| Unchanged | `test_solar_rejection[*]`, `test_warm_start[0.5-*]`, well-converged steady_state | **Same** |
| Slightly more cycling | `test_steady_state[0.5-{standard,well}_insulated]` (20→28, 13→19 reversals); `test_warm_start[1.0-drafty]` (2→12) | **More integrator activity, tracking unchanged** |

The "more cycling" cases all keep `post_settle_max_abs_dev ≤ 0.45°C` — well within tracking spec. The increased reversal count reflects the controller responding to 5× more sensor samples per hour, which is expected at finer cadence and matches what production at ~60s already does.

## Cadence-dependent rollup metrics (carry-over from B0)

Same caveats as `B0-heating/findings.md`:
- `rollup_itae`, `rollup_settling_time` (in ticks), `rollup_kwh_per_degree_hour`, `rollup_reversals` and `rollup_setpoint_changes` are not cadence-invariant and need normalization for cross-cadence comparison.

Not blocking the migration; documented for the eventual `tests/benchmark_metrics.py` cleanup.

## What changed in code

- `tests/hvac_bench/scenarios/test_cooling.py`: bench_metrics wired into all 4 test methods; converted to wall-clock API (`duration_minutes` + minute-keyed schedules); `hp_lag_minutes=2.0` default added.
- `custom_components/tasmota_irhvac/pi/pi_controller.py`: q-feedback nudge multiplied by `dt_factor` (commit `ea1cc50`).

## Carryforward for B1+ batches

The q-feedback fix is **production-side** and now applies to ALL bench tests — not just B1 cooling. Subsequent batches inherit cadence-invariant q-feedback automatically. No re-baseline of earlier batches needed since the fix is mathematically a no-op at 15-min cadence (verified across heating + cooling, 81 tests, 0 status flips).
