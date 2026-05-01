"""Locked reference scores (Phase 2b).

Checked-in numbers for every (controller × scenario) tuple. Any bench
change that moves a score outside its tolerance is flagged by the
regression test in ``tests/hvac_bench/scenarios/test_reference_scores.py``.

Update protocol when a score is expected to change:

1. Run the regression test in print mode::

       pytest tests/hvac_bench/scenarios/test_reference_scores.py \\
           -k generate -s

   This prints fresh numbers for every (controller × scenario).

2. Audit the deltas. A score moving by *more than its tolerance* must
   come from a deliberate bench change — physics fix, profile
   recalibration, controller behavior change. Capture the *why* in the
   commit message.

3. Update the numbers below. Tolerances rarely change — only when noise
   characterisation or scenario length changes substantially.

The point of locking these numbers is not "freeze the bench" — it's
"any bench change that moves discriminative power must be a deliberate
commit." That's what BOPTEST's locked KPI list buys.

## Tolerance philosophy

Single-realisation point estimates with conservative ± tolerances. Tighter
than this requires Monte-Carlo sampling, which is a follow-up. The
tolerances are calibrated by hand from the variance audit below
(comments per row); they should comfortably exceed natural noise
variation but flag any genuine bench drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreExpectation:
    """Expected score with absolute tolerance for a single KPI."""

    expected: float
    abs_tolerance: float

    def passes(self, observed: float) -> bool:
        return abs(observed - self.expected) <= self.abs_tolerance


# Layout: ``REFERENCE_SCORES[scenario_name][controller_name][kpi_name]``
# = ``ScoreExpectation``. Flat dict so the regression test can iterate
# without recursion. Tolerance choices documented inline; absolute (not
# relative) so small-value KPIs (e.g. ``warm_time_h ≈ 0``) get a
# meaningful floor.

REFERENCE_SCORES: dict[str, dict[str, dict[str, ScoreExpectation]]] = {
    # ── lr_heat_step ──────────────────────────────────────────────
    # Living room, heat mode, -5°C base / 6°C diurnal, 3 days, no inputs.
    "lr_heat_step": {
        "naive_bang_bang": {
            "tdis_tot": ScoreExpectation(58.416, 0.5),       # K·h, dominated by deep undershoot
            "ener_tot": ScoreExpectation(6.046, 0.05),        # kWh
            "peak_kw": ScoreExpectation(0.241, 0.005),
            "cold_time_h": ScoreExpectation(30.25, 0.5),
            "warm_time_h": ScoreExpectation(29.75, 0.5),
            "setpoint_changes": ScoreExpectation(232, 5),     # rails alternate every other tick
        },
        "well_tuned_pi": {
            "tdis_tot": ScoreExpectation(0.932, 0.10),
            "ener_tot": ScoreExpectation(6.018, 0.05),
            "peak_kw": ScoreExpectation(0.215, 0.005),
            "cold_time_h": ScoreExpectation(2.50, 0.5),
            "warm_time_h": ScoreExpectation(1.75, 0.5),
            "setpoint_changes": ScoreExpectation(27, 3),
        },
        "production_pi": {
            # Production behaves like well-tuned over 3 days because
            # batch WLS κ-gate rejects coefficient updates during early
            # learning (κ severe). Differentiation is expected on longer
            # horizons; the current locked numbers exist to detect
            # *regressions* in the short-horizon behavior.
            "tdis_tot": ScoreExpectation(0.932, 0.10),
            "ener_tot": ScoreExpectation(6.018, 0.05),
            "peak_kw": ScoreExpectation(0.215, 0.005),
            "cold_time_h": ScoreExpectation(2.50, 0.5),
            "warm_time_h": ScoreExpectation(1.75, 0.5),
            "setpoint_changes": ScoreExpectation(27, 3),
        },
    },
    # ── lr_cool_step ──────────────────────────────────────────────
    # Living room, cool mode, 28°C base / 6°C diurnal, 3 days, no inputs.
    "lr_cool_step": {
        "naive_bang_bang": {
            "tdis_tot": ScoreExpectation(28.548, 0.5),
            "ener_tot": ScoreExpectation(1.413, 0.03),
            "peak_kw": ScoreExpectation(0.075, 0.003),
            "cold_time_h": ScoreExpectation(26.25, 0.5),
            "warm_time_h": ScoreExpectation(21.25, 0.5),
            "setpoint_changes": ScoreExpectation(175, 5),
        },
        "well_tuned_pi": {
            "tdis_tot": ScoreExpectation(0.415, 0.10),
            "ener_tot": ScoreExpectation(1.450, 0.03),
            "peak_kw": ScoreExpectation(0.046, 0.003),
            "cold_time_h": ScoreExpectation(2.75, 0.5),
            "warm_time_h": ScoreExpectation(0.0, 0.25),
            "setpoint_changes": ScoreExpectation(20, 3),
        },
        "production_pi": {
            "tdis_tot": ScoreExpectation(0.484, 0.10),
            "ener_tot": ScoreExpectation(1.452, 0.03),
            "peak_kw": ScoreExpectation(0.046, 0.003),
            "cold_time_h": ScoreExpectation(3.25, 0.5),
            "warm_time_h": ScoreExpectation(0.25, 0.25),
            "setpoint_changes": ScoreExpectation(19, 3),
        },
    },
    # ── lr_heat_with_solar ────────────────────────────────────────
    # Living room, heat mode, with solar input (β_truth = -2.0).
    "lr_heat_with_solar": {
        "naive_bang_bang": {
            "tdis_tot": ScoreExpectation(58.187, 0.5),
            "ener_tot": ScoreExpectation(5.762, 0.05),
            "peak_kw": ScoreExpectation(0.250, 0.005),
            "cold_time_h": ScoreExpectation(30.75, 0.5),
            "warm_time_h": ScoreExpectation(30.75, 0.5),
            "setpoint_changes": ScoreExpectation(238, 5),
        },
        "well_tuned_pi": {
            "tdis_tot": ScoreExpectation(0.936, 0.10),
            "ener_tot": ScoreExpectation(5.740, 0.05),
            "peak_kw": ScoreExpectation(0.212, 0.005),
            "cold_time_h": ScoreExpectation(2.25, 0.5),
            "warm_time_h": ScoreExpectation(2.25, 0.5),
            "setpoint_changes": ScoreExpectation(25, 3),
        },
        "production_pi": {
            "tdis_tot": ScoreExpectation(0.932, 0.10),
            "ener_tot": ScoreExpectation(5.734, 0.05),
            "peak_kw": ScoreExpectation(0.212, 0.005),
            "cold_time_h": ScoreExpectation(2.25, 0.5),
            "warm_time_h": ScoreExpectation(2.00, 0.5),
            "setpoint_changes": ScoreExpectation(26, 3),
        },
    },
}


# Discriminative-power invariants that hold regardless of the exact
# locked numbers. These catch "bench broke and now naive scores like
# well-tuned" without requiring the full numeric lock — they're the
# load-bearing checks for bench credibility.
DISCRIMINATIVE_INVARIANTS = {
    # Bang-bang must be substantially worse than well-tuned PI on
    # comfort across every scenario. If this stops holding, the bench's
    # discriminative power has collapsed.
    "naive_worse_than_well_tuned_on_comfort": {
        "metric": "tdis_tot",
        "min_ratio": 5.0,
        # Naive's tdis_tot must be ≥ 5× well-tuned's on every scenario.
    },
}
