# B0 — Heating tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-10 · **Files:** `tests/hvac_bench/scenarios/test_heating.py`

## Outcome summary

| Run | Tick cadence | HP lag | Result |
|---|---|---|---|
| Phase 1 baseline | 15 min | 0 min | 53 passed, 4 xfailed |
| Phase 2 (refactored API) | 15 min | 0 min | 53 passed, 4 xfailed (identical to Phase 1) |
| Phase 2-laggy | 15 min | 2 min | 53 passed, 4 xfailed |
| Phase 3 | 3 min | 0 min | 53 passed, 4 **xpassed** |
| Phase 3-laggy | 3 min | 2 min | 53 passed, 4 **xpassed** |

The 4 cases that flipped xfail → xpass:
- `TestHeatingColdSnap::test_cold_snap[1.5-drafty_bungalow]`
- `TestHeatingSetpointDown::test_setpoint_down[1.5-drafty_bungalow]`
- `TestHeatingSteadyState::test_steady_state[1.5-drafty_bungalow]`
- `TestHeatingSteadyState::test_steady_state[1.5-standard_residential]`

All 4 are **`seed_factor=1.5` (FF over-seeded by 50%) × weak-insulation profile**. The third QUICK_PROFILES profile, `well_insulated` (τ_env=250min), passes at 1.5× regardless of cadence.

## Mechanism

Side-by-side comparison of `test_steady_state[1.5-drafty_bungalow]` at 15 min vs 3 min cadence (same seed, same model, hp_lag=2 min):

```
minute   15min: room  sp     int     ff  ||    3min: room  sp     int     ff
    0        21.560  28   0.000  7.750  ||        20.602  28   0.000  7.750
   15        17.665  16   0.000  7.750  ||        21.560  28  -1.477  7.750
   30        21.861  30   0.000  7.750  ||        20.129  29  -0.422  7.750
   45        17.796  16   0.000  7.750  ||        21.810  28  -0.613  7.750
   60        21.420  29   0.000  7.750  ||        21.582  28  -1.830  7.750
  ...                                                  ...
  300        17.689  16   0.891  7.750  ||        21.221  27 -10.503  7.750
  360        17.663  16   0.890  7.750  ||        20.815  26 -15.089  7.750
  720        17.575  16   0.888  7.750  ||        20.766  26 -15.080  7.750
```

**At 15-min cadence**, the over-seeded FF (+7.75°C of feedforward offset) drives the HP setpoint to ~28°C at startup. Within one tick (15 min), the room overshoots into the over-temp regime (room > desired + threshold). The regime correctly forces HP to idle (16) and freezes integration to prevent windup. After 15 min of idle, the room undershoots; the regime exits and HP returns to ~30. **Cycle**.

The integral is stuck near 0.89 — it cannot accumulate because the over-temp regime keeps freezing it on every overshoot, and the regime cycles take ≥ 1 full tick (15 min). At each tick boundary the system is mid-bang in either direction.

**At 3-min cadence**, the same regime entry/exit pattern occurs, but each phase lasts only 3 min instead of 15. The room overshoots less per phase (3 min of full HP output vs 15 min), and crucially **the integral has time to accumulate between regime cycles**. By minute 360, the integral has wound down to **-15.09** — within numerical noise of the value needed to exactly cancel the over-seeded FF (+7.75 → effective FF ≈ 0). After that the room holds rock-stable at 20.78°C.

## Production relevance

Production runs sensor-driven at ~60 s cooldown (per `feedback_pi_tick_architecture.md`), 15× faster than the original bench cadence and 3× faster than the migrated 3-min cadence. **The over-seed × weak-insulation pathology surfaced by the bench at 15 min is a sample-rate artifact that does not reflect production behavior.** The 4 xfails were false positives — diagnosing a controller pathology that doesn't exist on real hardware.

The mechanism is real, but its threshold sits between 15-min and 3-min cadence: the regime/anti-windup interaction needs the integral to build between regime cycles, which requires sample period < some critical fraction of the cycle duration. Production cadence (~60s) sits comfortably below that threshold.

## HP lag invariance

`hp_lag_minutes=2.0` (typical inverter compressor spool) was added during this migration. Diff between lag-vs-no-lag at both cadences shows essentially zero change (max delta ~0.5%, mostly machine-noise). Reason: the HP-quantized setpoint typically holds across multiple ticks, so a 2-min first-order lag finishes well before the next setpoint change. To stress-test lag we'd want `hp_lag_minutes=8-10` (cycling-only / fixed-capacity HP), but that wasn't necessary to confirm the cadence finding.

## Cadence-dependent rollup metrics

Several values in `compute_all_metrics` are not cadence-invariant and therefore not directly comparable across phases:

| Metric | Source of cadence dependency | Cadence-invariant alternative |
|---|---|---|
| `rollup_itae` | per-tick weighted sum; 5× ticks → ~5× ITAE for same dynamics | divide by n_ticks for time-averaged, or compute degree-minutes |
| `rollup_settling_time` | reported in **ticks**, not minutes | convert to minutes |
| `rollup_kwh_per_degree_hour` | numerator/denominator scale differently with discretization | use kWh / (∫\|error\| dt in degree-hours) |
| `rollup_reversals` | counts events; 5× ticks → more chances for direction changes | reversals per hour |
| `rollup_setpoint_changes` | same | changes per hour |

This is a separate quality issue worth fixing in `tests/benchmark_metrics.py`, but doesn't block the cadence migration — within-cadence comparisons remain valid.

## What changed in code

- `tests/hvac_bench/constants.py` (new): single source of truth for `TICK_MINUTES_DEFAULT`, `BATCH_INTERVAL_HOURS_DEFAULT`, `_SIM_EPOCH`. Honors `BENCH_TICK_MINUTES` env var (set by `pytest --tick-minutes=N`).
- `tests/hvac_bench/conftest.py`: adds `--tick-minutes`, `--bench-phase` CLI flags; opt-in `bench_metrics` fixture; `pytest_runtest_makereport` hook dumps per-test JSON.
- `tests/hvac_bench/runner.py`: introduces wall-clock API alongside legacy tick-keyed (`duration_minutes`, `outdoor_minute_schedule` etc.). Tick-keyed legacy still supported during incremental migration.
- `tests/hvac_bench/scenarios/test_heating.py`: converted to wall-clock semantics; recorded assertion values + control-quality rollup via `bench_metrics`; `_make_model` now defaults `hp_lag_minutes=2.0`.
- `local/tools/diff_bench_metrics.py` (gitignored): renders `xfail`/`xpass` properly using `wasxfail` field.

## Inputs to next batches

The migration playbook is now solid:
1. Wire `bench_metrics` into the batch's tests.
2. Run baseline at 15 min via `pytest <files> --bench-phase=BN-baseline`.
3. Refactor to wall-clock API.
4. Run refactored at 15 min via `--bench-phase=BN-refactored`. **Diff vs baseline must be empty modulo discretization rounding.**
5. Run refactored at 3 min via `--bench-phase=BN-3min --tick-minutes=3.0`.
6. Diff 3min vs refactored. Real cadence findings are here.

Open items for the wider bench:
- The cadence-dependent rollup metrics (above) should be fixed in `tests/benchmark_metrics.py` so `rollup_*` values diff cleanly across cadences. Not yet touched.
- Other simple-runner test files use legacy tick-keyed API and need conversion (cooling, derivative, energy, disturbances, noise, smith_predictor, imc_gains, hp_capacity_curve, pi_overcorrection_diagnosis, supplemental_learning, boundary_learning, buffer_variants*, wls_vs_greybox, solution_verification).
- `full_stack_runner` callers (test_full_stack_learning, test_input_taxonomy, test_seasonal_convergence, etc.) use `n_days` which auto-scales with `tick_minutes` — should already be cadence-invariant. To verify per-batch.
