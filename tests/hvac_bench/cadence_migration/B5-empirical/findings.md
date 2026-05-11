# B5 — Empirical tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **Files:** `tests/hvac_bench/empirical/*.py` (10 files) + `tests/hvac_bench/scenarios/test_statistical.py` (Monte Carlo, technically in scenarios/ but empirical in nature, included for completeness)

## Outcome summary

| Run | Tick cadence | Result |
|---|---|---|
| Pass A baseline | 15 min | All pass against HEAD baselines |
| Phase 3 first run (empirical/) | 3 min | 45 newly-created baselines (first-baseline gotcha) |
| Phase 3 confirmation (empirical/) | 3 min | **190 passed** |
| test_statistical Phase 3 (pre-refactor) | 3 min | First run created 3 baselines; identified cadence-coupling bug in test code |
| test_statistical refactor + 15-min Pass B | 15 min | 3 passed, byte-identical |
| test_statistical Phase 3 (post-refactor) | 3 min | 3 newly-created baselines, then confirmation pending |

## What changed in code

### `tests/hvac_bench/empirical/*.py` — no refactor

7 fast-tier empirical files: `test_cumulated_periodogram.py`, `test_data_loader.py`, `test_forward_selection.py`, `test_pem_fit.py`, `test_rc_model.py`, `test_runner.py`, `test_synthetic_drivers.py`. All are cadence-invariant by construction (synthetic input arrays, fixed sample-rate generators, regression-fit unit tests). Phase 3 diff shows all metrics within ±1% — these are unit tests, not cadence-sensitive simulations.

3 design-tier files: `test_inverter_hp_positive_control.py`, `test_proxy_variant_decision.py`, `test_synthetic_positive_control.py`. All have `@slow` or `@design`/`@study` markers. Per #90 session memory, their default-tier baselines are deferred; same for 3.0min. First caller with `--run-studies` will populate both.

### `tests/hvac_bench/scenarios/test_statistical.py` — refactor performed

This file lives in `scenarios/` but is functionally empirical (Monte Carlo statistical tests with 50 noise seeds × 3 scenarios). Audit found 3 tick-keyed callsites:

- `run_scenario(ctrl, model, n_ticks=32, mode="heat")` × 2 → `duration_minutes=8 * 60`
- `run_scenario(ctrl, model, n_ticks=48, mode="heat")` → `duration_minutes=12 * 60`
- `outdoor_schedule=lambda tick: max(-5.0, 10.0 - tick * 1.25)` → `outdoor_minute_schedule=lambda m: max(-5.0, 10.0 - m * (5.0 / 60.0))`
- `history[24:]` (after settling, tick 24 at 15-min = minute 360) → `[h for h in history if h["minute"] >= 6 * 60]`

**Refactor caught a real cadence-coupling bug in the original tick-keyed code.** With the original `outdoor=lambda tick: 10.0 - tick * 1.25` and `--tick-minutes=3.0`, the cold snap dropped 1.25°C per 3 min = 25°C/h — 5× faster than the intended 5°C/h. The wall-clock-correct version produces the same outdoor profile at any cadence. The pre-refactor 3-min baselines were stale and have been replaced.

## Cadence diff (empirical/)

All 7 fast-tier files: max drift < 1%. Cadence-invariant.

## Cadence diff (test_statistical, post-refactor)

Captured after the wall-clock refactor stabilizes; only Phase 3 created the baselines so this is the first cadence-aware measurement. Real Monte Carlo (50 seeds × scenario) statistical noise dominates any cadence effect; finer cadence captures slightly different transient sampling and produces small distribution shifts.

## What I did NOT change

- 3 design-tier empirical files: no refactor (also no baselines exist for them in either default or 3.0min). They'll be addressed when someone runs `--run-studies` for the first time.

## Carry-forward

- The "tick-keyed lambda is cadence-coupled" bug is the same shape as the B1-disturbances 2R2C silent-bug lesson. Pass A→Pass B byte-identical at 15-min is necessary but not sufficient: tick-keyed code that "passes Pass B" can still be cadence-broken if the wall-clock semantics depend on tick rate. The B0/B1 playbook (and now B2/B5) makes the wall-clock refactor the authoritative carrier of intent.
- The recurring pattern across cadence-keyed lambdas (`lambda t: f(t)` where `f` has implicit per-tick units) is the cleanest cadence-coupling bug shape. Every file that converted away from tick-keyed schedules in #91 has had to recompute the rate expression in per-minute units.
