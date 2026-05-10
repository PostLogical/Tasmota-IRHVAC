# B1 — Derivative tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-10 · **Files:** `tests/hvac_bench/scenarios/test_derivative.py`

## Outcome summary

13 tests, all passing at all phases. 0 status flips.

| Run | Tick cadence | Result |
|---|---|---|
| Phase 1 baseline (Pass A) | 15 min | 13/13 passed (44s) |
| Phase 2 (refactored, Pass B) | 15 min | 13/13 passed, diff vs baseline empty modulo float noise |
| Phase 3 | 3 min | 13/13 passed (3m24s, ~5× slower) |

## Notable metrics 3-min vs 15-min

- **kd=5.0 max_abs_d (cold_start)**: 0.60 → 0.54 (-10.5%) — D-term peaks slightly less at finer cadence (smaller per-tick rate-of-change → smaller raw D before filter).
- **d_at_step values**: all parametrizations show 1e-4-magnitude D at the setpoint-step moment, scaled smoothly with Kd. Cadence change scales these by ~-8% across all Kd values — consistent with finer time discretization.
- **kd=0.5 max_abs_d (cold_snap)**: +2-4% increase at 3-min — opposite direction from cold_start, expected since the cold_snap scenario has continuous outdoor variation (more visible D-term excitation at finer cadence).
- **total_kwh**: shifts ≤ 0.3% across all parametrizations — energy outcomes are stable.

All deltas are within expected float-noise + cadence-discretization ranges. The asserted derivative-on-measurement property (`d_at_step / max_abs_d < 0.5`) holds at both cadences.

## Notes

- `test_kd_sweep_noisy_steady` runs with `sensor_noise_sigma=0.15` (only place in B1 with non-zero sensor noise). Behavior at 3-min: `d_std` (controller D-term standard deviation) scales with Kd as expected — the noise-amplification check still works.
- `test_kd_sweep_setpoint_step` reads D at a specific step moment. Refactor uses cadence-aware lookup (`next(i for i, h in enumerate(history) if h["minute"] == 150)`) instead of hardcoded `d_terms[10]`.
- Aggressive-PI oscillation test (`test_kd_damps_oscillation`) uses 16h run with last 8h as the oscillation window; at 3-min cadence the late_temp_std measure correctly uses minute-based windowing, not tick count.

## What changed in code

- `tests/hvac_bench/scenarios/test_derivative.py`: bench_metrics wired into all 5 test methods with per-Kd metric flattening (`kd_{kd}_{metric}` keys); converted to wall-clock API (`duration_minutes`, minute-keyed schedules); `hp_lag_minutes=2.0` default; cadence-aware step-tick lookup; minute-based late-window selection.

No production-side changes for this batch (the q-feedback fix from `ea1cc50` already applies).
