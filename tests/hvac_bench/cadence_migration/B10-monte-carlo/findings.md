# B10 — Monte Carlo runner cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **File:** `tests/hvac_bench/test_monte_carlo.py`

## Outcome summary

| Run | Tick cadence | Result |
|---|---|---|
| Pass A baseline | 15 min | 2 passed (HEAD baselines) |
| Phase 3 (3-min, --force-regen) | 3 min | 2 newly-created baselines |
| Phase 3 confirmation | 3 min | 2 passed |

**No refactor needed.** Both tests stub out `run_full_stack` via `unittest.mock.patch` so the cadence-sensitive simulation never executes. The tests verify Monte Carlo seed-scaling overrides via captured `pi_overrides` dicts.

## Cadence diff

Byte-identical between `regression_data/default/test_monte_carlo/*.csv` and `regression_data/3.0min/test_monte_carlo/*.csv`. The captured override dicts (integer seeds, dict shapes) do not depend on tick cadence.

## Carry-forward

When #93 flips `TICK_MINUTES_DEFAULT` to 3.0, the test_monte_carlo baselines remain unchanged. No follow-up needed.
