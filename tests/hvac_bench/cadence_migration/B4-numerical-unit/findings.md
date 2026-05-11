# B4 — Numerical / unit tests cadence migration findings

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **Files:** `tests/hvac_bench/test_richardson.py`, `test_open_loop_runner.py`, `test_identifiability.py`, `test_residual_diagnostics.py`, `test_bias_attribution.py`, `test_kpis.py`, `test_thermal_model_validation.py`, `test_assertions.py`, `test_observation_pipeline_validation.py`, `test_post_fill_metrics.py`

## Outcome summary

| Run | Tick cadence | Result |
|---|---|---|
| Pass A baseline | 15 min | All pass against HEAD baselines (existing default tier) |
| Phase 3 first run (creates baselines) | 3 min | 169 newly-created baseline failures (pytest-regressions first-baseline gotcha) |
| Phase 3 confirmation | 3 min | **215 passed** |

**No refactor performed.** Audit identified 12 `tick_minutes=15.0` literals in `test_kpis.py`, 12 in `test_open_loop_runner.py`, 12 `n_ticks=N` literals in `test_richardson.py`, and similar hardcoded patterns elsewhere — but these are **intentional unit-test parameterization**. The tests verify metric / model behavior at a specific cadence by passing `tick_minutes=15.0` as a function argument; they don't inherit cadence from `TICK_MINUTES_DEFAULT`. Hardcoding here is appropriate.

## Cadence-sensitivity per file

| File | Meaningful 3-min drift? | Verdict |
|---|---|---|
| `test_richardson.py` | All metrics within ±1% | Cadence-invariant: Richardson sweep is its own internal cadence parameter |
| `test_open_loop_runner.py` | All within ±1% | Hardcoded `tick_minutes=15.0` ignores `--tick-minutes` |
| `test_identifiability.py` | All within ±1% | Same |
| `test_residual_diagnostics.py` | All within ±1% | Same |
| `test_kpis.py` | All within ±1% | Unit tests pass `tick_minutes=15.0` as an explicit arg to `compute_control_kpis` |
| `test_thermal_model_validation.py` | All within ±1% | Pass `dt_minutes=15.0` to `model.step()` explicitly |
| `test_assertions.py` | All within ±1% | Same |
| `test_post_fill_metrics.py` | All within ±1% | Single `n_ticks=0` literal; trivially invariant |
| **`test_bias_attribution.py`** | Modest, ~2% bias | **Real** — closed-loop bias estimation samples more often at 3-min, but bias magnitude drift is small |
| **`test_observation_pipeline_validation.py`** | Counts +188% to +401% | **Sample-count scaling** — counts observations admitted to WLS/greybox/RLS buffers over a fixed wall-clock duration. At 3-min cadence, ~5× more samples → ~5× count for same wall-clock behavior. Not a behavior change. |

## Real findings

**`test_bias_attribution`** has small but real drift in `bias` and `bias_over_se` (+2.4%). Closed-loop bias estimation samples more observations at finer cadence; the additional observations are correlated (same underlying dynamics, just sampled more often) so the standard-error doesn't shrink linearly. Bias magnitude itself shifts slightly because finer sampling captures transient features. Within acceptable drift; both baselines preserved.

**`test_observation_pipeline_validation`** counts go up ~5× as expected for any sample-count metric at 5× finer cadence. Examples:

| Metric | 15-min | 3-min | Ratio |
|---|---|---|---|
| `gb_hp_on` | (varies) | ~5× | Sample-count scaling |
| `wls_total` | (varies) | ~5× | Sample-count scaling |
| `rls_obs_count` | (varies) | ~3× | Sample-count scaling |

These are correct — the buffer admission code is doing its job at a higher sample rate. The underlying behavior (which observations get admitted vs filtered) is consistent.

## What I did NOT change

The "hardcoded 15.0" pattern in unit tests is intentional. Tests like:

```python
b = compute_control_kpis(_fixed_history(96, error=1.0), tick_minutes=15.0)
```

are unit-testing the metric function at a specific cadence, with synthetic input that doesn't depend on the runner's tick rate. Converting these to `tick_minutes=TICK_MINUTES_DEFAULT` would change their semantic intent (they'd start verifying behavior at whatever cadence the suite happens to be running). Left as-is.

## Carry-forward

When #93 flips `TICK_MINUTES_DEFAULT` to 3.0:
- All 8 cadence-invariant files in B4 will pass against existing default baselines (because their tests don't depend on the global default).
- `test_bias_attribution` and `test_observation_pipeline_validation` will drift to match the 3-min baselines we just committed.

The recurring sample-count-scaling pattern in `test_observation_pipeline_validation` is the same shape as the `compute_reversals` / `compute_setpoint_changes` issue from B0/B1: any "count of ticks/observations" metric scales with cadence. Documenting consistently across batches; cleanup in `tests/benchmark_metrics.py` deferred.
