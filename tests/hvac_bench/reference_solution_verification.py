"""Locked solution-verification scores (Phase 3b).

Per-KPI Richardson + tick-rate-spread numbers for the canonical
reference scenarios run at three tick rates. Locked numbers here serve
the same role for *numerical-error sensitivity* as
``reference_scores.py`` does for *single-grid KPI values*: any bench
change that shifts the discretization profile must be a deliberate
relock.

Each locked entry is a :class:`SolutionVerificationExpectation` with:

* ``value_at_finest`` — the KPI value at the finest tick (5 min). The
  value any production-relevant test would actually quote.
* ``value_tolerance`` — absolute tolerance on the finest value (catches
  bench drift on the production-relevant cell).
* ``bound_max`` — upper ceiling on ``max(GCI, tick_rate_spread)``.
  Catches the kernel/controller's grid-convergence shape changing.
* ``expected_in_regime`` — locked Roache regime classification. Flags
  when a previously-asymptotic cell stops fitting the power-law form
  or vice versa (a bench change that shifts which cells the GCI
  bound applies to).

The Phase 3 plan calls for grid-converged KPIs reported with a
discretization error band alongside the MC band. Phase 3b populates
that with two scenarios — ``lr_heat_step`` and ``lr_cool_step`` —
using two reference controllers:

* ``naive_bang_bang`` — exposes the worst-case tick-coupling. Its
  ``tdis_tot`` cycles at the tick rate (Åström & Wittenmark sampled-
  data limit-cycle aliasing), so its extrapolation is misleading and
  the spread is the truer bound. We lock it anyway because that
  characterisation is itself a load-bearing fact about the bench.
* ``well_tuned_pi`` — production-relevant deterministic reference.
  Most KPIs are clean Richardson regime; ``lr_cool_step``'s
  ``tdis_tot`` exposes the 5-minute PI overcorrection that Phase 3c
  diagnoses.

The solar scenario is intentionally excluded: its hard-coded
``tick_minutes=15.0`` solar schedule aliases against the sweep, and
β_solar identification is on the Phase 4 fidelity track (see
``project_bench_solar_fidelity.md``).

Update protocol when a number is expected to change:

1. Re-run with the print harness::

       BENCH_PRINT_RICHARDSON=1 pytest \\
           tests/hvac_bench/scenarios/test_solution_verification.py -s -m slow

2. Audit deltas. A shift in ``bound_max`` is a meaningful bench change
   — either the kernel's convergence order changed or a new tick-rate-
   dependent term entered the controller path. A flip in
   ``expected_in_regime`` is also load-bearing and must be explained.

3. Update the numbers and the rationale comment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolutionVerificationExpectation:
    """Locked solution-verification expectation for a single KPI."""

    value_at_finest: float
    value_tolerance: float       # absolute tolerance on finest-tick value
    bound_max: float             # ceiling on max(GCI, tick_rate_spread)
    expected_in_regime: bool     # locked Roache regime classification

    def passes_value(self, observed: float) -> bool:
        return abs(observed - self.value_at_finest) <= self.value_tolerance

    def passes_bound(self, observed_bound: float) -> bool:
        return observed_bound <= self.bound_max


# Tick rates used in the locked sweep. Sorted descending (coarsest first).
# Updated #102 (2026-05-14): shifted from (30, 15, 5) to (15, 5, 1) to
# bracket the post-#93 production-default cadence (3 min).  The finest
# grid (1 min) is now finer than production, so the convergence study
# validates behavior at the operating point rather than above it.
RICHARDSON_TICK_MINUTES: tuple[float, ...] = (15.0, 5.0, 1.0)


# Layout: ``RICHARDSON_SCORES[scenario_name][controller_name][kpi_name]``
# = ``SolutionVerificationExpectation``.
#
# Numbers produced by ``BENCH_PRINT_RICHARDSON=1 pytest -s -m slow``.
# Tolerances calibrated to comfortably exceed the noise floor while
# flagging genuine bench drift.

RICHARDSON_SCORES: dict[
    str, dict[str, dict[str, SolutionVerificationExpectation]]
] = {
    # All values re-locked #102 (2026-05-14) after RICHARDSON_TICK_MINUTES
    # shifted from (30, 15, 5) to (15, 5, 1) to bracket the post-#93
    # production-default cadence.  At the new finer grids many KPIs that
    # previously sat in the asymptotic regime now fall into fallback fits
    # because their signal is at the noise floor — well-tuned PI tdis_tot
    # in particular is essentially zero at 1-min, leaving residual variance
    # dominated by sensor noise rather than discretization error.  This
    # is a real regime shift, not a regression.
    #
    # ── lr_heat_step ──────────────────────────────────────────────────
    #
    # Heat scenario, 3 days, outdoor base -5°C, desired 20.5°C.
    "lr_heat_step": {
        # Naive bang-bang's tdis_tot cycle rate IS the tick rate.
        # Sequence (15,5,1min): 58.4, 12.4, 1.0 K·h.  Spread=57.4.  Math
        # regime is power-law clean (limit-cycle aliasing follows h^1).
        "naive_bang_bang": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=1.03,
                value_tolerance=0.30,
                bound_max=60.0,             # spread-dominated; controller property
                expected_in_regime=True,    # math regime is fine, magnitude isn't
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=6.68,
                value_tolerance=0.10,
                bound_max=1.0,
                expected_in_regime=True,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.262,
                value_tolerance=0.010,
                bound_max=0.05,
                expected_in_regime=False,   # peak essentially flat; nan order
            ),
        },
        # well-tuned PI: tdis_tot is non-monotonic at the finer grids
        # (0.22 @ 15-min → 0.01 @ 5-min → 0.03 @ 1-min) — the controller's
        # warm-up transient is so small that 1-min residuals are dominated
        # by sensor noise / anomaly events, not by discretization error.
        # Power-law fit fails → fallback regime.  Real shape change from
        # operating in a finer regime than the test was originally locked
        # for; not a regression.
        "well_tuned_pi": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=0.033,
                value_tolerance=0.10,       # essentially zero; absolute tolerance
                bound_max=0.5,
                expected_in_regime=False,   # noise-floor; non-monotonic
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=6.006,
                value_tolerance=0.05,
                bound_max=0.10,
                # Flipped False→True in #126 (2026-05-30) Ki=0.7 retune.
                # At higher Ki the controller is more responsive, the energy
                # trajectory is cleaner across cadences, and the power-law
                # fit succeeds (observed order ~0.87).
                expected_in_regime=True,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.227,
                value_tolerance=0.010,
                bound_max=0.05,
                expected_in_regime=True,
            ),
        },
    },
    # ── lr_cool_step ──────────────────────────────────────────────────
    #
    # Cool scenario, 3 days, outdoor base 28°C, desired 23°C.
    "lr_cool_step": {
        # Same limit-cycle aliasing as heat.  Sequence (15,5,1min):
        # 28.5, 6.3, 0.65.
        "naive_bang_bang": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=0.65,
                value_tolerance=0.20,
                bound_max=30.0,
                expected_in_regime=True,
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=1.57,
                value_tolerance=0.05,
                bound_max=0.30,
                expected_in_regime=True,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.087,
                value_tolerance=0.010,
                bound_max=0.05,
                expected_in_regime=True,
            ),
        },
        # well-tuned PI cool: similar noise-floor regime shift at fine
        # grids (sequence 0.30 @ 15-min → 0.225 @ 5-min → 0.224 @ 1-min,
        # essentially saturated at the noise floor).
        "well_tuned_pi": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=0.224,
                value_tolerance=0.10,
                bound_max=0.20,
                expected_in_regime=False,   # noise-floor; nearly flat at fine grids
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=1.446,
                value_tolerance=0.05,
                bound_max=0.020,
                expected_in_regime=False,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.046,
                value_tolerance=0.010,
                bound_max=0.020,
                expected_in_regime=False,
            ),
        },
    },
    # ── lr_heat_with_solar ────────────────────────────────────────────
    #
    # Heat scenario with solar input (β_truth = -2.0).  Added #102
    # (2026-05-14) after #101 made `_solar_schedule` cadence-clean
    # (minute-keyed); pre-#101 the schedule's hardcoded `tick_minutes=15.0`
    # aliased against the sweep, so this scenario was excluded from
    # solution verification.
    "lr_heat_with_solar": {
        # Same limit-cycle aliasing as lr_heat_step at (15, 5, 1).
        "naive_bang_bang": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=1.07,
                value_tolerance=0.30,
                bound_max=60.0,
                expected_in_regime=True,
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=6.40,
                value_tolerance=0.10,
                bound_max=1.0,
                expected_in_regime=True,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.267,
                value_tolerance=0.010,
                bound_max=0.05,
                expected_in_regime=True,
            ),
        },
        # well-tuned PI with solar: tdis_tot sequence (15,5,1min):
        # 0.70, 0.47, 0.25.  Solar disturbance keeps the warm-up
        # signal larger than lr_heat_step at the noise floor, but
        # power-law fit still fails → fallback.
        "well_tuned_pi": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=0.253,
                value_tolerance=0.10,
                bound_max=1.0,
                expected_in_regime=False,
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=5.726,
                value_tolerance=0.05,
                bound_max=0.10,
                expected_in_regime=False,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.228,
                value_tolerance=0.010,
                bound_max=0.05,
                expected_in_regime=True,
            ),
        },
    },
}


# ── Convergence-shape invariants ──────────────────────────────────────────
#
# Independent of locked numbers, certain shape invariants must hold.

CONVERGENCE_INVARIANTS: dict[str, dict] = {
    # The naive bang-bang's heating ``tdis_tot`` is a clean limit-cycle-
    # aliasing case: the cycle rate IS the tick rate, so tdis_tot scales
    # ~linearly with h.  At (15, 5, 1) the sequence is (58.4, 12.4, 1.0)
    # K·h with order ≈ 1.39 — power-law clean.  A flip to non-asymptotic
    # indicates either a controller-side change (bang-bang modified to
    # add hysteresis dwell or rate limit) or a kernel-side change that
    # broke aliasing-based limit-cycle convergence.
    #
    # (Pre-#102: this invariant was on well_tuned_pi.tdis_tot — but at the
    # post-#102 finer grids that KPI is at the noise floor and falls into
    # fallback regime.  Switched to naive_bang_bang where the discretization
    # signal is large enough to be load-bearing for shape detection.)
    "naive_bangbang_heat_tdis_is_aliasing_kernel_error": {
        "scenario": "lr_heat_step",
        "controller": "naive_bang_bang",
        "kpi": "tdis_tot",
        "must_be_in_regime": True,
    },
}
