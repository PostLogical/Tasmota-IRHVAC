# B1 — Disturbances tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-10 · **Files:** `tests/hvac_bench/scenarios/test_disturbances.py`, `tests/hvac_bench/disturbances.py`, `tests/hvac_bench/thermal_model.py`, `tests/hvac_bench/scenarios/test_input_taxonomy.py` (consumer of refactored Disturbance class)

## Outcome summary

15 test_disturbances tests + 16 test_input_taxonomy tests, all passing at both cadences. **0 status flips, 0 failures.**

| Run | Tick cadence | Result |
|---|---|---|
| Phase 1 baseline (Pass A) | 15 min | 15/15 disturbances passed (legacy tick-keyed Disturbance class) |
| Phase 2 (Pass B refactor) | 15 min | 15/15 disturbances passed; metrics byte-identical to Pass A baseline (validates refactor preserves behavior) |
| Phase 3 (3-min cadence) | 3 min | 15/15 disturbances passed |
| input_taxonomy verify | 15 min | 16 passed + 1 known #88 xfail (Disturbance refactor doesn't break consumer) |

## Critical bug caught by Pass A baseline

The Disturbance class refactor (tick-based fields → minute-based) initially broke `ThermalModel2R2C.step()` because I only updated the 1R1C path's disturbance call, missing the 2R2C path. The bug: 2R2C still called `d.intensity(tick)` against a now-minute-based class — disturbances never fired during 2R2C tests.

This was silent (tests passed) because `is_active(tick=8)` returned False against `start_minute=120`, so disturbances had no effect; tests with tolerant assertions still passed but with wrong behavior. **The Pass A baseline diff (Pass A vs Pass B at 15-min) caught this** — refactor should have been a no-op, but max_room_temp shifted -2.95% on `test_cooking_no_overshoot[drafty_bungalow]` (cooking disturbance never fired). Without the systematic Pass A baseline, this would have shipped silently. Lesson: **Pass A is mandatory before any structural class refactor**.

## What changed in code

### `tests/hvac_bench/disturbances.py`
- Renamed `start_tick` → `start_minute`, `duration_ticks` → `duration_minutes`, `ramp_ticks` → `ramp_minutes`.
- `intensity(tick)` → `intensity(minute)`.
- Factory functions accept `start_minute=` (defaults preserve former `start_tick=10` semantics: `start_minute=150` at 15-min cadence).

### `tests/hvac_bench/thermal_model.py`
- Both `ThermalModel` (1R1C) and `ThermalModel2R2C.step()` now compute `minute = tick * dt_minutes` and pass it to `disturbance.intensity(minute)`.

### `tests/hvac_bench/scenarios/test_disturbances.py`
- Factory calls converted: `oil_boiler(start_tick=8)` → `oil_boiler(start_minute=120)` etc.
- `n_ticks=N` → `duration_minutes=M`.
- Tick-based assertion windows (`h["tick"] >= 16`) → minute-based (`h["minute"] >= 240`).

### `tests/hvac_bench/scenarios/test_input_taxonomy.py`
- `_make_window_disturbances` and `_make_oven_disturbances` use minute-based fields directly (the `int(... / tick_minutes)` arithmetic dropped — minutes are cadence-invariant by construction).
- Subtle: old code's `ramp_ticks=max(1, int(5 / tick_minutes))` floored to 1 tick = 15-min ramp at 15-min cadence (a cadence-dependent bug); new code uses `ramp_minutes=5.0` matching the original docstring intent. Empirical impact on input_taxonomy below regression tolerance.

## Notable cadence findings

| Pattern | Affected tests | Direction |
|---|---|---|
| Status flips | None | All 15 tests pass at both cadences |
| Limit cycles reduced | `test_front_door_recovery[drafty_bungalow]`: reversals 1 → 0 | **Improvement** (q-feedback fix already landed) |
| `rollup_itae` explosion | All 15 tests | **Metric definition bug** — ITAE multiplies by tick index, not wall-clock time. 5× more ticks → ~24× weight inflation. Documented in B1-energy/findings.md; needs `tests/benchmark_metrics.py` cleanup as separate work. |
| `rollup_kwh_per_degree_hour` -80% | Most tests | **Metric definition bug** — denominator/numerator both depend on discretization differently |
| `rollup_settling_time` 4-5× | Most tests | **Metric in tick units** — convert to minutes for cadence-portable interpretation |
| `cold_ticks` / `cold_max_run` | `test_front_door_recovery[drafty_bungalow]` 1 → 7 | Real cadence-sensitivity finding: 3-min cadence with hp_lag=2min sees more cold-undershoot ticks during the brief recovery window after door closes. Tracking still well within bounds (max_undershoot 2.97 → 2.60 actually improved). Counts more *because* sample frequency is higher. |

## Carry-forward

Disturbance class is now wall-clock and cadence-invariant. Future cadence sweeps don't need any test_disturbances changes. The `tests/benchmark_metrics.py` rollup cleanup (ITAE / settling_time / kwh_per_degree_hour normalization) is the recurring issue across B0/B1/B1-energy/B1-disturbances findings — separate commit when this migration completes.
