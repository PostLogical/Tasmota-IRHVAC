# B7 — Input taxonomy cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **File:** `tests/hvac_bench/scenarios/test_input_taxonomy.py`

## Outcome summary

| Run | Tick cadence | Wall time | Result |
|---|---|---|---|
| Pass A baseline | 15 min | ~16 min | 15 passed + 1 xfailed (matches HEAD) |
| Pass B (refactored) | 15 min | 10:15 | 15 passed + 1 xfailed, byte-identical |
| Pass B (refactored + perf patches) | 15 min | 10:47 (incl. test_smith_predictor sanity) | All clean |
| Phase 3 (3-min, --force-regen) | 3 min | **13:29** | 15 passed + 1 xfailed, baselines created |
| Phase 3 confirm (concurrent w/ 2 other masters) | 3 min | 48:11 | 15 passed + 1 xfailed |

**Status flips: 0.** The 1 xfail is `TestDirectActiveSource::test_stove_beta_recovers_meaningful_magnitude` per #88 (stove × outdoor identifiability confound), unchanged across cadences.

## What changed in code

### `tests/hvac_bench/scenarios/test_input_taxonomy.py`

Three hardcoded `tick_minutes=15.0` callsites in schedule-helper invocations replaced with `TICK_MINUTES_DEFAULT`:

- `make_pellet_stove_schedule(15.0)` → `make_pellet_stove_schedule(TICK_MINUTES_DEFAULT)`
- `make_dr_temp_schedule(15.0)` → `make_dr_temp_schedule(TICK_MINUTES_DEFAULT)`
- `make_passive_zone_schedule(15.0)` → `make_passive_zone_schedule(TICK_MINUTES_DEFAULT)`

Without this fix, at 3-min cadence the schedule helpers would interpret each tick as 15 min of wall-clock time inside `hour = (tick * tick_minutes / 60.0) % 24.0` — rolling over the diurnal cycle 5× faster than real wall-clock. Same cadence-coupling-bug pattern as `test_statistical.py`'s `lambda tick: 10.0 - tick * 1.25` from B5.

### Collateral bench-perf changes (separate commit)

B7 surfaced two perf bottlenecks via py-spy that affect every full_stack-runner-using test (B6/B7/B8/B9). Both landed as a separate commit:

1. **`full_stack_runner.py` freezegun hoist** — was per-tick `with freeze_time(_sim_dt):` × 14,400 ticks per 3-min/30-day test. Each `__enter__` hashes `sys.modules` (O(N_modules)). Now single freeze_time outside the loop + `factory.move_to()` per tick.
2. **`pi_controller.py` + `adapters.py` `skip_tick_output` flag** — bench-only, default True in `TasmotaPIAdapter`. Skips `_build_tick_output()` (which recomputes `ObservationBufferSnapshot.get_leverage_scores()` for every buffer entry per tick). Bench doesn't consume the dispatcher payload.

Combined impact: B7 Phase 3 at 3-min went from projected ~100 min to **13:29**. The deeper batched-buffer-admission architecture fix is captured in future_work prompt #95.

## Refactor equivalence verification

Pass B at 15-min produced **byte-identical** metrics against existing `regression_data/default/test_input_taxonomy/*.csv` (pytest-regressions rtol=1e-6 enforced). 16 tests × all metrics intact.

## Cadence-sensitivity findings (15-min → 3-min)

Selected from CSV diff (recap; full numerical table left implicit, focus on patterns):

| Pattern | Tests affected | Direction | Diagnosis |
|---|---|---|---|
| `final_*_beta` coef trajectories | All TestDirectActiveSource, TestAdjacentZoneProxy | Small drift (~5-15%) | Real — finer cadence WLS observes more data per batch, modest β shift |
| `n_batches` counts | All | ~5× | Sample-count scaling artifact (5× more ticks → 5× more batches at default 12h spacing) |
| `ctrl_comfort_pct` | All | Mostly stable ±2pp | Real, minor |
| `outdoor_delta` β | All | Within 0.05 of expected | Real, well-bounded — the test's assertion `abs(od - expected) < 0.1` holds at both cadences |
| `final_passive_zone_beta` | Passive-zone tests | Drift but assertion still holds | Real cadence-sensitivity, no flip |
| `solar_proxy_beta` (anomaly tests) | TestAnomalyRobustness | Within ±0.1 | Real, modest |

## Concurrent-CPU contention finding

Pre-perf-patch B7 Phase 3 at 3-min: projected ~150 min (interrupted by py-spy investigation at 1h32m, ~10/16 tests done).

Post-perf-patch B7 Phase 3 at 3-min, **single pytest master**: 13:29.

Post-perf-patch B7 Phase 3 at 3-min, **3 concurrent pytest masters**: 48:11 (~3.5× slowdown).

Implication for the rest of the migration: stacking parallel pytest masters with overlapping CPU footprint tanks per-master performance, even on a 10-core machine. The optimal strategy is **serial heavy-batch runs** rather than parallel masters — the user's intuition was correct.

The `--dist=load` within-file parallelism (used for `test_boundary_learning` Master A — 4 workers × 12 tests = 6:34 wall) is genuinely beneficial and doesn't have the same contention pattern because all 4 workers share one pytest master / one xdist scheduler.

## Carry-forward

- The same `tick_minutes=15.0` literal pattern in schedule-helper callsites might exist in other bench files; B5 already caught test_statistical; B7 catches this; B6/B9 still pending.
- `B6` (full_stack_learning) and `B9` (seasonal_convergence) have analogous `tick_min = 15.0` / `tick_minutes=15.0` patterns in their config builders, addressed alongside their own refactors (see those batches' findings).
