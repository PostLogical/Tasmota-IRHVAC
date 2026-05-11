# B2 — Closed-loop advanced tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **Files:** `tests/hvac_bench/scenarios/test_smith_predictor.py`, `tests/hvac_bench/scenarios/test_imc_gains.py`, `tests/hvac_bench/scenarios/test_hp_capacity_curve.py`, `tests/hvac_bench/scenarios/test_pi_overcorrection_diagnosis.py`

## Outcome summary

| Run | Tick cadence | Result |
|---|---|---|
| Phase 1 baseline (Pass A) | 15 min | 57 passed + 1 xfailed (matches HEAD, no drift) |
| Phase 2 (refactored, Pass B) | 15 min | 34 smith + 21 imc + 2 hp_capacity all clean. pytest-regressions enforced rtol=1e-6 byte-identical against existing default baselines. |
| Phase 3 (3-min) | 3 min | 57 passed + 1 xfailed against newly-created `regression_data/3.0min/` baselines |

**Status flips: 0.** All test contracts hold at both cadences. The 1 xfail is `test_pi_overcorrection_diagnosis::test_tracking_error_grows_with_finer_ticks_than_gain_calibration` per #92 (not in B2 scope).

## What changed in code

### `tests/hvac_bench/scenarios/test_smith_predictor.py`

- `_run_pair` and `_run_hold_comparison` signatures: `n_ticks` → `duration_minutes`, `outdoor_schedule` → `outdoor_minute_schedule`, `desired_schedule` → `desired_minute_schedule`.
- All callsites: `n_ticks=48 → duration_minutes=12*60` (preserves 12-hour wall-clock duration), `n_ticks=32 → duration_minutes=8*60`, `n_ticks=64 → duration_minutes=16*60`.
- Lambda conversions: `lambda t: max(-5.0, 10.0 - t * 1.25)` → `lambda m: max(-5.0, 10.0 - m * (5.0 / 60.0))` (1.25°C/tick at 15-min = 5°C/h, expressed in cadence-invariant per-minute units).
- Dict schedules: `desired_schedule={10: 22.5}` → `desired_minute_schedule={150: 22.5}` (tick 10 at 15-min = minute 150).
- `SCENARIOS` lists in `TestSmithAggregate` and `TestHoldTimerReduction` propagated.

### `tests/hvac_bench/scenarios/test_imc_gains.py`

- Same pattern as test_smith_predictor: `_run_pair` signature, callsites, two `SCENARIOS` lists.
- Additional lambda: `lambda t: 5.0 - t * 0.25` → `lambda m: 5.0 - m * (1.0 / 60.0)` (0.25°C/tick at 15-min = 1°C/h).

### `tests/hvac_bench/scenarios/test_hp_capacity_curve.py`

- Imported `TICK_MINUTES_DEFAULT` from `constants`.
- `n_ticks = int(_N_DAYS * 24 * 60 / 15.0)` → `int(_N_DAYS * 24 * 60 / TICK_MINUTES_DEFAULT)`.
- `WeatherState(..., tick_minutes=15.0)` → `WeatherState(..., tick_minutes=TICK_MINUTES_DEFAULT)`.
- `diurnal_outdoor(t, ..., 15.0, weather)` → `diurnal_outdoor(t, ..., TICK_MINUTES_DEFAULT, weather)`.
- `diurnal_solar(..., tick_minutes=15.0, ...)` → `diurnal_solar(..., tick_minutes=TICK_MINUTES_DEFAULT, ...)`.

### `tests/hvac_bench/scenarios/test_pi_overcorrection_diagnosis.py`

No changes. The file explicitly sweeps cadence as the independent variable (`replace(base, tick_minutes=5.0)` vs `tick_minutes=30.0`) for its premise; this is intentional cadence-comparison, not cadence-coupling. Test is xfailed per #92.

## Refactor equivalence verification

Pass A → Pass B at 15-min cadence: **byte-identical against existing `regression_data/default/<test_module>/*.csv` baselines** (pytest-regressions enforced `rtol=1e-6, atol=1e-9`).

Spot-check sanity of conversion math:
- `n_ticks=48 × 15min = 720min = 12h` ✓
- `lambda t: 10.0 - t * 1.25` at tick=1 returns `8.75`; new `lambda m: 10.0 - m * (5.0 / 60.0)` at minute=15 returns `8.75` ✓
- `desired_schedule={10: 22.5}` triggers at tick=10 (= minute=150 at 15-min); `desired_minute_schedule={150: 22.5}` triggers at minute=150 ✓

Confidence: refactor is semantically equivalent; the Phase 3 (3-min) divergence is exclusively cadence-driven.

## Cadence-sensitivity findings (15-min → 3-min)

Top-level table from `regression_data/default/` vs `regression_data/3.0min/` CSV diff. **Real vs metric-artifact split investigated for each pattern** (see "Diagnosis" below).

| Pattern | Affected tests | Direction | Diagnosis |
|---|---|---|---|
| `imc_itae` / `smith_itae` / `flat_itae` raw | All scenarios | +37% mean to +1600% max | Mixed — see below |
| `imc_overshoot` / `smith_overshoot` raw | All | +28% mean to +206% max | **Metric artifact** — `compute_overshoot = max(|err|)` includes initial-state error, which 15-min cadence has 15 min to decay before first sample; 3-min cadence has 3 min. |
| `imc_reversals` / `smith_reversals` raw counts | Most | +50% mean to +1000% max | **Mixed** — normalized to per-hour: real cycling rate up 3-11× in worst cases (Smith cold_start [standard_residential] imc: 0.08 → 0.92 rev/hr). |
| `*_setpoint_changes` raw counts | Most | +50% mean to +500% max | **Mixed** — same story. flat cooling: 0.38 → 1.50 changes/hr (4× rate). |
| `imc_settling_time` | Some | +73% mean to +1287% max | **Metric definition** — `compute_settling_time` was migrated to wall-clock minutes per `tests/benchmark_metrics.py:73`, but legacy baselines still reflect tick units. Cleanup pending. |
| `*_total_kwh` | All | ±8% (essentially stable) | **Real, minor** — energy is wall-clock-integrated; stable as expected. |
| `sat_fixed_pct` (hp_capacity_curve) | 1 | -78% (2.83 → 0.63) | **Real** — finer PI updates respond to overcorrection 5× faster, less stuck-at-rail. Production at ~60s cadence should match 3-min better than 15-min. |
| `sat_capacity_pct` | 1 | -37% (78.05 → 49.12) | **Real** — same mechanism. |
| `cold_violations_capacity` | 1 | +401% (993 → 4982) | **Mostly artifact** — count of cold-side ticks; 5× ticks → 5× count for same wall-clock duration of being cold. |
| `ctrl_comfort_*_pct` | 1 | ±6% | **Real, minor** — comfort is fraction-of-ticks, mostly stable. |
| pi_overcorrection_diagnosis all metrics | 1 | 0% | File uses internal cadence sweep, unaffected by `--tick-minutes`. |

## Diagnosis — distinguishing artifact from real cadence-sensitivity

The most interesting finding from this batch is that several "regression" signals are metric-definition artifacts, not control quality changes:

### Overshoot inflation is initial-state capture

Traced `test_cooling_bounded[drafty_bungalow]` side-by-side. The recorded `imc_overshoot` jumped 2.31°C → 3.98°C. But the trajectory tells a different story:

- **Initial state**: room=28°C, desired=24°C (true initial |err|=4°C).
- **15-min first sample (after one model.step of 15 min)**: room=26.305°C, |err|=2.305°C.
- **3-min first sample (after one model.step of 3 min)**: room=27.982°C, |err|=3.982°C.

The `Max |err|` over the entire history occurs at **minute=0 in both cases** — i.e. the metric is reading the initial-state error, not a control overshoot. Coarser cadence has 15 min to decay before recording; finer cadence captures 12 min more of the initial transient. **Both report the actual peak undershoot during recovery as ~0.59°C — identical real behavior.**

### Reversals/hr show genuine controller cycling increase

After normalizing raw counts to per-hour rates, the controller IS cycling more in wall-clock terms at finer cadence:

| Test | Controller | rev/hr 15min | rev/hr 3min | Ratio |
|---|---|---|---|---|
| `test_cold_start_bounded[standard_residential]` | smith | 0.33 | 1.17 | 3.5× |
| `test_cold_start_bounded[standard_residential]` | imc | 0.08 | 0.92 | 11× |
| `test_cooling_bounded[drafty_bungalow]` | flat | 0.25 | 1.25 | 5× |

This is **production-relevant**: the 15-min bench was smoothing over real controller cycling that happens at finer sampling rates. Production at ~60s cooldown (per `feedback_pi_tick_architecture.md`) sees this same cycling. The bench is finally surfacing what production has been doing.

### hp_capacity_curve saturation drop is production-relevant

`sat_fixed_pct` going 2.83% → 0.63% (without capacity curve) means the PI loop pulls off the rail 5× faster at finer sampling. The 15-min bench was over-reporting saturated time. The capacity curve's claim ("enabling it raises saturation by >10pp") still holds at both cadences, so the test assertion isn't broken — but the absolute saturation level is much lower than the 15-min bench suggested.

## Carry-forward

- **Metric definitions** in `tests/benchmark_metrics.py` have known cadence-sensitivity:
  - `compute_overshoot` includes initial-state error (overbroad)
  - `compute_settling_time` mid-migration: now returns minutes (per code comment) but legacy baselines may still reflect ticks
  - Per-tick counts (`compute_reversals`, `compute_setpoint_changes`) scale linearly with sample rate
  - Per-tick integrals like `compute_itae` are mathematically valid wall-clock integrals but sample early-time errors more accurately at finer cadence
- These are documented per-findings; a unified `tests/benchmark_metrics.py` review + per-hour-normalized rollup is deferred to its own commit, not in #91 scope.
- The real wall-clock finding from B2 — **finer cadence reveals genuine controller cycling and saturation reduction that 15-min sampled away** — is the production-relevant signal worth tracking through B3+ heavy batches.
