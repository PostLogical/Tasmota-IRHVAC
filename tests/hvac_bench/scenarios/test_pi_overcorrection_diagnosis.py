"""PI overcorrection at fine ticks — literature-grounded version.

The PI gains (Kp=1.5, Ki=0.15) are calibrated for a 15-minute nominal
cadence (``pi_tick_fallback = 900s``).  At finer ticks the same gains
deliver action more frequently → the closed loop is effectively
under-damped → tracking error grows.

The original Phase 3c test (``test_5min_overcorrection_signature``)
locked specific numerical signatures (4.5× setpoint_changes/hour,
1.9× mean_abs_error) that turned out to be inflated by the wall-clock
leak fixed in #84 (sin/cos ToD nuisance regressors stuck at near-
constant → β_outdoor absorbed diurnal variance → controller acted on
contaminated FF).  Per the test_bench_self_validation header, those
specific multipliers were *bench artefact*.

This rewrite asserts the *underlying property* — gains tuned for 15-min
cadence ⇒ tracking error larger at finer ticks — using BOPTEST's
canonical thermal-discomfort integral ``tdis_tot`` (Blum et al. 2021,
Annex 71 / IEA EBC).  ``tdis_tot`` is integrated over the run, so it
captures both transient excursions and steady-state ringing without
depending on PI internals or wall-clock-leak-coupled signatures.

Dual threshold (per `feedback_test_to_spec_not_output.md`):

* Hard floor: ``tdis_tot(fine) > tdis_tot(coarse)`` — if not,
  the gain-calibration property is broken (good thing — but a
  deliberate change worth surfacing).
* Tight sensitivity: ``tdis_tot(fine) > tdis_tot(coarse) * 2.0`` —
  catches subtle regressions in the cadence/gain coupling without
  locking against a specific magnitude.  Calibrated against an
  observed ratio that exceeds the threshold by a meaningful margin.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.reference_scenarios import (
    CANONICAL_SCENARIOS,
    make_well_tuned_for_scenario,
    run_reference_scenario,
)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Premise invalidated by q-feedback dt_factor fix (commit ea1cc50). "
        "The test measured cadence-dependent over-correction that was "
        "dominated by q-feedback's effective-gain scaling with sample rate. "
        "After the fix, q-feedback is cadence-invariant; PI integration is "
        "also cadence-invariant by design (avg_error * dt_factor); so finer "
        "ticks no longer produce more tracking error — they produce LESS "
        "(better discretization of the disturbance response).  "
        "Empirical post-fix: fine_tdis ≈ 0.22, coarse_tdis ≈ 0.84.  "
        "Replace this test with a 'cadence-invariance' assertion or remove."
    ),
)
def test_tracking_error_grows_with_finer_ticks_than_gain_calibration(bench_metrics, num_regression):
    """Closed-loop ``tdis_tot`` rises when ticks are finer than 15-min nominal.

    PI gains were tuned for ``pi_tick_fallback=900s``.  Running the same
    scenario at 5-min ticks produces 3× the action density at the same
    gain → underdamped response → larger ``tdis_tot``.  The 30-min run
    serves as the comparison baseline (matched scenario, slower action
    rate, gains nominal-or-conservative).
    """
    base = CANONICAL_SCENARIOS["lr_cool_step"]

    fine_scenario = replace(base, tick_minutes=5.0)
    fine_controller = make_well_tuned_for_scenario(fine_scenario)
    _, fine_bundle = run_reference_scenario(fine_controller, fine_scenario)

    coarse_scenario = replace(base, tick_minutes=30.0)
    coarse_controller = make_well_tuned_for_scenario(coarse_scenario)
    _, coarse_bundle = run_reference_scenario(coarse_controller, coarse_scenario)

    fine_tdis = fine_bundle.tdis_tot
    coarse_tdis = coarse_bundle.tdis_tot

    bench_metrics["fine_tdis_tot"] = fine_tdis
    bench_metrics["coarse_tdis_tot"] = coarse_tdis
    bench_metrics["tdis_ratio"] = fine_tdis / coarse_tdis if coarse_tdis > 1e-9 else 0.0
    check_bench_metrics(num_regression, bench_metrics)

    # Hard floor: directional property must hold.  If fine ≤ coarse the
    # cadence/gain coupling has been fundamentally changed (e.g. gain
    # scheduling against tick rate added) — a deliberate redesign worth
    # surfacing, not a regression to silently absorb.
    assert fine_tdis > coarse_tdis, (
        f"tdis_tot at 5-min ({fine_tdis:.4g} K·h) should exceed 30-min "
        f"({coarse_tdis:.4g} K·h) since PI gains are tuned for 15-min "
        f"nominal cadence (Kp=1.5, Ki=0.15, pi_tick_fallback=900s). "
        f"If the controller now auto-scales gains to cadence, this test "
        f"should be relaxed deliberately."
    )

    # Tight sensitivity: ratio should be substantially > 1.  Threshold of
    # 2.0× catches subtle regressions in the cadence/gain coupling
    # (e.g. partial gain auto-scaling, hold mechanism weakening) without
    # locking the specific magnitude.  Set well below the observed
    # ratio post-#84 to absorb seed/weather noise but well above 1× to
    # keep the bound informative.
    if coarse_tdis > 1e-9:  # avoid div-by-zero on perfect coarse runs
        ratio = fine_tdis / coarse_tdis
        assert ratio > 2.0, (
            f"tdis_tot ratio fine/coarse = {ratio:.2f} < 2.0 "
            f"(fine={fine_tdis:.4g}, coarse={coarse_tdis:.4g}). "
            f"Cadence/gain coupling has weakened; verify whether the "
            f"controller's effective gain is no longer cadence-sensitive."
        )
