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
RICHARDSON_TICK_MINUTES: tuple[float, ...] = (30.0, 15.0, 5.0)


# Layout: ``RICHARDSON_SCORES[scenario_name][controller_name][kpi_name]``
# = ``SolutionVerificationExpectation``.
#
# Numbers produced by ``BENCH_PRINT_RICHARDSON=1 pytest -s -m slow``.
# Tolerances calibrated to comfortably exceed the noise floor while
# flagging genuine bench drift.

RICHARDSON_SCORES: dict[
    str, dict[str, dict[str, SolutionVerificationExpectation]]
] = {
    # ── lr_heat_step ──────────────────────────────────────────────────
    #
    # Heat scenario, 3 days, outdoor base -5°C, desired 20.5°C.
    "lr_heat_step": {
        # Naive bang-bang's tdis_tot cycle rate IS the tick rate.
        # Sequence (30,15,5min): 125.8, 58.4, 12.4 K·h. Spread=113.4.
        # Extrapolation produces a non-physical -11.7 K·h because the
        # data isn't in true Richardson regime (limit-cycle aliasing,
        # not power-law convergence). Locked numbers reflect this:
        # the value at 5min is the actual quoted score; the bound is
        # the spread (the truer bound when extrapolation lies).
        "naive_bang_bang": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=12.40,
                value_tolerance=2.0,
                bound_max=120.0,            # spread-dominated; controller property, not kernel
                expected_in_regime=True,    # math regime is fine, magnitude isn't
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=6.50,
                value_tolerance=0.20,
                bound_max=1.0,
                expected_in_regime=True,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.264,
                value_tolerance=0.020,
                bound_max=0.05,
                expected_in_regime=False,   # fallback fit (insufficient curvature)
            ),
        },
        # well-tuned PI: tdis_tot dominated by warm-up transient.
        # Sequence (30,15,5min): 0.000, 0.932, 1.152 K·h. Clean Richardson
        # regime with p=2.29; GCI=0.024.
        "well_tuned_pi": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=1.15,
                value_tolerance=0.20,
                bound_max=1.5,              # spread dominates (warm-up transient sensitivity)
                expected_in_regime=True,
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=5.97,
                value_tolerance=0.15,
                bound_max=0.20,
                expected_in_regime=False,   # fallback fit (signal too small to fit)
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.227,
                value_tolerance=0.020,
                bound_max=0.05,
                expected_in_regime=False,
            ),
        },
    },
    # ── lr_cool_step ──────────────────────────────────────────────────
    #
    # Cool scenario, 3 days, outdoor base 28°C, desired 23°C.
    "lr_cool_step": {
        # Same limit-cycle aliasing as heat. Sequence: 61.0, 28.5, 6.3.
        "naive_bang_bang": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=6.35,
                value_tolerance=1.5,
                bound_max=60.0,
                expected_in_regime=True,
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=1.52,
                value_tolerance=0.10,
                bound_max=0.30,
                expected_in_regime=False,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.085,
                value_tolerance=0.010,
                bound_max=0.020,
                expected_in_regime=False,
            ),
        },
        # well-tuned PI cool: tdis_tot 0.155 (30min) → 0.415 (15min) →
        # 1.406 (5min). Sequence INCREASES as h decreases — controller
        # overcorrects at fine ticks. Phase 3c diagnoses; Phase 3b just
        # locks the property. Non-asymptotic by Roache classification
        # because the fit lands in fallback.
        "well_tuned_pi": {
            "tdis_tot": SolutionVerificationExpectation(
                value_at_finest=1.41,
                value_tolerance=0.30,
                bound_max=2.5,
                expected_in_regime=False,   # 5-min PI overcorrection signature
            ),
            "ener_tot": SolutionVerificationExpectation(
                value_at_finest=1.45,
                value_tolerance=0.10,
                bound_max=0.020,
                expected_in_regime=False,
            ),
            "peak_kw": SolutionVerificationExpectation(
                value_at_finest=0.049,
                value_tolerance=0.010,
                bound_max=0.020,
                expected_in_regime=False,
            ),
        },
    },
}


# ── Convergence-shape invariants ──────────────────────────────────────────
#
# Independent of locked numbers, certain shape invariants must hold.

CONVERGENCE_INVARIANTS: dict[str, dict] = {
    # The well-tuned PI's heating ``tdis_tot`` is dominated by the
    # warm-up transient, which is a proper kernel-discretization error
    # (room temperature evolves continuously; finer h → smaller
    # truncation error in the integration). It must be in Roache
    # asymptotic regime. A flip to non-asymptotic indicates either a
    # controller-side change (e.g. a new tick-coupled term in the PI
    # path) or a kernel-side change (e.g. switching the integration
    # scheme away from matrix-exp).
    "well_tuned_pi_heat_warmup_is_kernel_error": {
        "scenario": "lr_heat_step",
        "controller": "well_tuned_pi",
        "kpi": "tdis_tot",
        "must_be_in_regime": True,
    },
}
