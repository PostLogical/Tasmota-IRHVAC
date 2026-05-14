# Cadence migration — closed 2026-05-14 (#93)

`TICK_MINUTES_DEFAULT` flipped from 15.0 → 3.0 in `tests/hvac_bench/constants.py`,
closing the cadence migration thread that ran from #90 through #104.

## What changed at the flip

- `tests/hvac_bench/constants.py:TICK_MINUTES_DEFAULT = 3.0` (was 15.0).
- Baseline directories renamed:
  - `regression_data/default/` → `regression_data/15.0min/` (preserved
    pre-#93 baselines so `--tick-minutes=15.0` still runs).
  - `regression_data/3.0min/` → `regression_data/default/` (post-cadence
    baselines became the no-flag default).
- 5 cadence-irrelevant @design baselines (empirical/test_runner.py and
  empirical/test_inverter_hp_positive_control.py) copied from
  `15.0min/` to `default/` — they read pre-canned bundle data, not bench
  sims, so their values are cadence-invariant.
- 4 unconditional `pytest.mark.xfail` wrappers removed from
  `test_heating.py` (cold_snap[1.5-drafty_bungalow],
  setpoint_down[1.5-drafty_bungalow],
  steady_state[1.5-{drafty_bungalow,standard_residential}]).
  These wrappers acknowledged real over-seed × weak-insulation
  oscillation that fired in the 32–48-tick-window 15-min regime
  because batch WLS didn't have time to fire.  At 3-min the same
  test windows give batch WLS room to correct, so the assertions
  pass naturally.  At 15-min the tests will now hard-fail —
  documented as expected; 15-min is no longer a default-supported
  cadence, only a runnable historical reference.
- 1 additional `pytest.mark.xfail` wrapper removed from
  `test_input_taxonomy.py::test_stove_beta_recovers_meaningful_magnitude`
  after surfacing as XPASS during the #93 verification run.  Pre-#93
  the stove × outdoor identifiability confound attenuated β_stove to
  ~-0.96 (truth -3.0); at the new 3-min default the WLS sees ~5×
  more observations and β_stove reaches the ≥1.5 threshold.  The
  underlying multi-collinearity issue (#88) remains as a research
  topic but no longer fails this assertion at the production-fidelity
  cadence.  Will hard-fail at `--tick-minutes=15.0`.

## What stays xfail

- `test_bench_fidelity.py::TestTickIntervalSensitivity::test_tick_interval_sweep`:
  cross-cadence comfort drift between 30-min and 15-min still ~5.36%
  (just over the 5% bound) post #100/#103/#104.  Real cadence-coupled
  property change from the `ea1cc50` Q-feedback dt_factor fix.  Wrapper
  is doing its job; resolution gated on #89 q-feedback design study.
- `test_input_taxonomy.py::TestDirectActiveSource::test_stove_unlocks_in_30_days`
  (and other #88-tracked stove-related assertions if any): kept where
  the underlying multi-collinearity bug genuinely persists.
- All `TICK_MINUTES_DEFAULT < 15.0` cadence-gated xfails from
  `0141a16` (#98 sub-item 2): now firing as expected at the new 3-min
  default.  These track real cadence-coupled regressions in tests
  that pass at 15-min and fail at 3-min — they wait on either
  smoothing or per-cadence threshold work.

## What was preserved

- Every per-batch `B*/findings.md` doc is preserved verbatim as a
  historical artifact of the migration.  They reference 15-min as the
  default of the time and that remains accurate to their authoring date.
- 15-min baselines under `regression_data/15.0min/` so any future
  cross-cadence investigation can still compare 3-min vs 15-min
  without re-running all of pre-#93.
- The `--tick-minutes=N` CLI flag continues to work for all cadences;
  see `feedback_pi_tick_architecture.md` for production-cadence context.

## Migration thread (commits in chronological order)

- **#90** — per-test recording wired across structurally-wired bench
  files (DONE 2026-05-11)
- **#91** — B0–B11 wall-clock-API refactor + dual-cadence baseline
  pass (mostly DONE 2026-05-11; design tier carved out as #98)
- **#92** — two q-fix-induced xfails + 3 bench-root migration files
  (still open; deferred until #89 q-feedback design study resolves)
- **#97** — bench batch-WLS timing aligned to production 07:00/19:00
  (DONE 2026-05-12)
- **#98** — design-tier 3-min carve-out + post-#97 hard-assertion
  follow-up (DONE 2026-05-14; remaining items deferred to #95 / #56)
- **#100** — solar β covariance collapse after BIC tau rejection at
  fine cadence (DONE 2026-05-12 via HAC + ESS-substituted BIC)
- **#101** — bench schedule API migrated to single minute-keyed API
  (DONE 2026-05-13)
- **#103** — dict-keyed schedules → step-function semantic
  (DONE 2026-05-14)
- **#104** — collapse `BENCH_TICK_MINUTES` env var + `--tick-minutes`
  CLI flag to single canonical entry (DONE 2026-05-14)
- **#93** — flip `TICK_MINUTES_DEFAULT` to 3.0 (DONE 2026-05-14, this
  doc)

## Remaining open

- **#92** — two q-fix-induced xfails (`test_bench_fidelity` /
  `test_pi_overcorrection_diagnosis`) gated on #89 q-feedback design.
  Behaviour is correct at the new default; the xfail decisions are
  about what to assert on, not whether the underlying code is right.
