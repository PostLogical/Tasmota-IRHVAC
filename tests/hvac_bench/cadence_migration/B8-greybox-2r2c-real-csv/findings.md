# B8 — Greybox 2R2C real-CSV cadence migration findings (partial)

**Branch:** `bench-magicmock-audit` · **Date:** 2026-05-11 · **File:** `tests/hvac_bench/scenarios/test_greybox_2r2c_real_csv.py`

## Outcome summary

**Partial — `lit_grounded_results` fixture-using tests baselined; `real_csv_results` fixture-using tests timed out and need a separate longer-timeout run.**

| Test class | Fixture | Status | Detail |
|---|---|---|---|
| `TestGreybox2R2CLitGrounded` | `lit_grounded_results` (single spring season, 60d × 2 modes) | ✅ 7 baselines at 3-min | Most-important set: validates greybox recovery against the lit-grounded standard_residential profile (Bacher-Madsen typical residential, decoupled from the suspect living_room calibration). |
| `TestGreybox2R2CRealCSV` | `real_csv_results` (all 4 seasons × 2 modes = 8 sims, 60d each) | ❌ Fixture timeout @ 3-min | The 50-min per-test timeout (default for Master B) was insufficient for the all-season fixture under concurrent-CPU load. The 4 dependent tests show ERROR (fixture didn't yield). |

## What did NOT change in code

The file was already wall-clock-ready per the #91 audit — no refactor performed. Both fixtures use `_run_one_season` → `_run_scenario` → `run_full_stack`; the cadence path inherits from `full_stack_runner`'s now-cadence-aware code (post the bench-perf patch).

## Cadence-sensitivity findings

(Limited — only the lit_grounded subset has 3.0min baselines.)

| Test | 3-min baseline value | Notes |
|---|---|---|
| `test_recovers_alpha_total` | (snapshot) | Greybox alpha_total recovery against known truth |
| `test_recovers_k_c` | (snapshot) | k_c recovery |
| `test_recovers_ua_c` | (snapshot) | ua_c recovery |
| `test_tau_fast_in_band` / `test_tau_slow_in_band` | (snapshot) | Tau recovery within Bacher-Madsen plausible bands |
| `test_2r2c_dispatched` / `test_gates_pass_at_least_once` | (snapshot) | Dispatcher + gate firing counts |

Three xfails in this file (lit_grounded class) are pre-existing per #46-followup / lit-grounded discrimination work — unchanged across cadences. See `project_bench_id_discrimination.md`.

## Deferred to follow-up run

`TestGreybox2R2CRealCSV` (4 tests) needs:

```
.venv/bin/pytest tests/hvac_bench/scenarios/test_greybox_2r2c_real_csv.py::TestGreybox2R2CRealCSV \
  -n 1 --tick-minutes=3.0 --force-regen -m design --timeout=7200
```

(serial, one worker, 2-hour per-test budget). Estimated wall-time ~90-120 min at 3-min cadence post-perf-patches. The fixture is module-scoped, so all 4 tests piggyback on one sim run.

## Carry-forward

- The deferred real_csv_results fixture work is **separate from #91 close-out** since it's pure @design tier (regression guards, not investigations). Can land later with #93 capstone prep.
- The lit_grounded baselines committed here are the production-relevant ones (verdict source for greybox upgrade decisions).
