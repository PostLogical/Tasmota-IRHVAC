# B3 — Supplemental learning cadence migration findings (Phase 3 addendum)

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **File:** `tests/hvac_bench/scenarios/test_supplemental_learning.py`

## Outcome summary

This addendum completes the supplemental_learning portion of B3. Refactor + Pass B byte-identical were captured in the original B3 findings doc; Phase 3 at 3-min was deferred and is now done.

| Run | Tick cadence | Result |
|---|---|---|
| Pass A baseline | 15 min | 5 passed (HEAD baselines) |
| Pass B (refactored) | 15 min | 5 passed, byte-identical against existing baselines |
| Phase 3 (3-min, --force-regen) | 3 min | 5 baselines created |
| Phase 3 confirm (3-min) | 3 min | 5 passed against newly-committed baselines |

## What was refactored (recap from B3 findings)

`run_with_stove` helper converted from tick-keyed to wall-clock:

- `n_ticks=N, mode="heat"` → `duration_minutes=N * 15, mode="heat"`
- `stove_on_ticks=(start_tick, end_tick)` → `stove_on_minutes=(start_min, end_min)`
- `outdoor_schedule=lambda tick: ...` → `outdoor_minute_schedule=lambda minute: ...`
- Internal `tick_interval_min` defaults to `TICK_MINUTES_DEFAULT`
- Stove on/off transition: `tick == stove_start` → minute-window match `stove_start_min <= minute < stove_start_min + tick_interval_min` (fires on the first tick spanning the transition minute, cadence-invariant)
- Added `minute` field to history dict for clean wall-clock slicing in tests

All 6 callsites in test_supplemental_learning.py converted to wall-clock semantics.

## Cadence-sensitivity findings (15-min → 3-min)

| Test | 15-min recorded | 3-min recorded | Direction |
|---|---|---|---|
| test_setpoint_differs_burn_vs_idle | (snapshot) | (snapshot) | Real difference in burn/idle setpoint contrast |
| test_before_after_gives_reasonable_coefficient | (snapshot) | (snapshot) | Inferred stove coefficient drifts slightly |
| test_phase_aware_is_more_accurate | (snapshot) | (snapshot) | Phase-aware advantage may change with finer cadence |
| test_before_after_robust_to_outdoor_change | (snapshot) | (snapshot) | Outdoor-ramp robustness preserved |
| test_hp_resumes_correctly_after_stove | (snapshot) | (snapshot) | HP resume behavior preserved |

All assertions hold at both cadences — refactor + cadence change are well-behaved for this file.

## Carry-forward

None — supplemental_learning is fully migrated.
