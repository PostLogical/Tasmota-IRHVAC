# B9 — Seasonal convergence cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **File:** `tests/hvac_bench/scenarios/test_seasonal_convergence.py`

## Outcome summary

**Refactor-only.** All 4 tests are `@pytest.mark.design` tier, gated behind a single class decorator. No Phase 3 baselines created in this batch — the design-tier opt-in run is deferred to a follow-up matching the original #91 plan's heavy-batch carve-out.

| Step | Status |
|---|---|
| Audit: cadence-coupling | `_make_synth_config` had `tick_min = 15.0` hardcoded; `_make_real_config` (default) uses CSV-bound cadence and is cadence-invariant by construction |
| Refactor: `tick_min = TICK_MINUTES_DEFAULT` | ✅ |
| Import `TICK_MINUTES_DEFAULT` from constants | ✅ |
| Pass A baseline run | n/a — no fast-tier tests collect |
| Pass B 15-min | n/a — no fast-tier tests collect |
| Phase 3 3-min | **Deferred** — all 4 tests are @design |

## What changed in code

### `tests/hvac_bench/scenarios/test_seasonal_convergence.py`

- Added `from tests.hvac_bench.constants import TICK_MINUTES_DEFAULT`
- `_make_synth_config`: `tick_min = 15.0` → `tick_min = TICK_MINUTES_DEFAULT`. This affects the `WeatherState(n_ticks=..., tick_minutes=tick_min)` and the `_make_outdoor_schedule(season, weather_state, tick_min)` calls. Without this fix, `--tick-minutes=3.0` would still build WeatherState as if cadence were 15-min, producing a wall-clock-mis-scaled AR(1) trajectory.

`_make_real_config` (the default `_make_config`) was untouched — it uses `windowed_real_weather` which is CSV-bound and cadence-invariant by construction.

## Why no Phase 3 in this batch

Per #91 plan and `feedback_test_tier_policy.md`, the @design tier is opt-in and runs ~10 min per scenario. test_seasonal_convergence has 4 tests × 4 seasons + module-scoped fixture × multi-month sims — total Phase 3 wall time at 3-min cadence with our perf patches would be ~1-2 hours. Combined with B3's other timed-out @design files (buffer_variants, buffer_variants_synth) + the real_csv all-season fixture in B8, the design-tier carve-out can be done in one dedicated session with `--timeout=7200 -n 1` (serial workers) to avoid the concurrent-CPU contention that bit B3 and B8.

## Carry-forward

- Phase 3 design-tier run for test_seasonal_convergence (4 tests).
- Combined with: test_buffer_variants (2 tests), test_buffer_variants_synth (2 tests), test_greybox_2r2c_real_csv::TestGreybox2R2CRealCSV (4 tests).
- Recommended: `pytest <files> -n 1 --tick-minutes=3.0 --force-regen -m design --timeout=7200` (serial worker, 2-hour budget). Expected ~3-4 hours total wall time.
