# B11 — Solution verification cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **File:** `tests/hvac_bench/scenarios/test_solution_verification.py`

## Outcome summary

**No refactor performed.** The file is **deliberately cadence-sweeping** as its core test design and does not depend on `TICK_MINUTES_DEFAULT`.

## Why no refactor

`test_solution_verification.py` implements Roache GCI + tick-rate spread for grid-convergence study. It pins its own internal cadence sweep at `RICHARDSON_TICK_MINUTES = (30.0, 15.0, 5.0)` (defined in `tests/hvac_bench/reference_solution_verification.py:80`) and runs each (scenario × controller × KPI) cell at each of those three cadences, then computes the discretization sensitivity bound.

Key observations:

- Per-test `_scenario_at_tick(scenario_name, tick_minutes)` does `replace(base, tick_minutes=tick)` — the test explicitly varies cadence as the independent variable.
- The `--tick-minutes=N` global flag does not affect this file; it's neutralized by the per-cell `tick_minutes` override on each scenario.
- All three test functions (`test_solution_verification_within_tolerance`, `test_convergence_shape_invariants`, `test_richardson_reports_have_finite_bands`) are `@pytest.mark.study`, opted in only with `--run-studies`.
- No regression baselines exist under `regression_data/default/test_solution_verification/` because the tests have never been run with `--run-studies` in this branch. The first caller with `--run-studies` will populate the default baseline; the 3.0min equivalent is moot because the file ignores the global cadence.

## What would change at production cadence?

Nothing about this file. The Richardson sweep itself is a methodology test — it confirms numerical convergence behavior — not a closed-loop scenario whose results depend on the bench's chosen default cadence.

## Carry-forward

When prompt #93 flips `TICK_MINUTES_DEFAULT` to 3.0, this file remains untouched. The `RICHARDSON_TICK_MINUTES` tuple is its own contract.
