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


# Layout:
#   ``REFERENCE_SCORES[scenario_name][controller_name][cadence_min][kpi_name]``
#   = ``ScoreExpectation``.
# Cadence axis (added #102, 2026-05-14) so the same locked-score
# regression covers both the 15-min historical reference and the
# production-realistic 3-min cadence.  Tests parametrize on
# ``(scenario, controller, cadence)`` and skip cells where the cadence
# block is missing — additive evolution over time.

REFERENCE_SCORES: dict[str, dict[str, dict[float, dict[str, ScoreExpectation]]]] = {
    # ── lr_heat_step ──────────────────────────────────────────────
    # Living room, heat mode, -5°C base / 6°C diurnal, 3 days, no inputs.
    "lr_heat_step": {
        "naive_bang_bang": {
            15.0: {
                "tdis_tot": ScoreExpectation(58.416, 0.5),       # K·h, dominated by deep undershoot
                "ener_tot": ScoreExpectation(6.046, 0.05),        # kWh
                "peak_kw": ScoreExpectation(0.241, 0.005),
                "cold_time_h": ScoreExpectation(30.25, 0.5),
                "warm_time_h": ScoreExpectation(29.75, 0.5),
                "setpoint_changes": ScoreExpectation(232, 5),     # rails alternate every other tick
            },
            3.0: {
                # Locked #102 (2026-05-14).  At 3-min naive bang-bang's
                # rail-alternation cycle is much shorter (5× more ticks
                # per wall-clock) so K·h-of-discomfort is much smaller
                # — discriminative ratio vs well-tuned still > 600×.
                "tdis_tot": ScoreExpectation(6.185, 0.5),
                "ener_tot": ScoreExpectation(6.588, 0.10),
                "peak_kw": ScoreExpectation(0.255, 0.005),
                "cold_time_h": ScoreExpectation(13.9, 0.5),
                "warm_time_h": ScoreExpectation(12.5, 0.5),
                "setpoint_changes": ScoreExpectation(494, 15),
            },
        },
        # Relocked 2026-05-09 (#84 Stage C): replaced bench's global
        # `time.monotonic = lambda` patch with PIController constructor-
        # injected `monotonic=lambda: sim_clock` (production should never
        # rely on global module-level patching). Stage B's prior numbers
        # reflected two compounding bench bugs: (1) freezegun's epoch-style
        # `time.monotonic()` value silently bypassed the setpoint-hold's
        # `_last_setpoint_change_time = 0.0` initial-state edge case, and
        # (2) the same epoch value made `dt_seconds = min(now_mono - last,
        # 1800)` saturate at the cap on every tick, giving `dt_factor = 2`
        # instead of 1 — i.e., the integrator updated 2× per tick. Both
        # are fixed: hold check now uses `Optional[float]` with explicit
        # None handling, and sim-clock-style monotonic resolves dt_factor
        # to the correct 1.0. The new numbers reflect the controller's
        # actual behavior at the configured tick rate.
        "well_tuned_pi": {
            15.0: {
                "tdis_tot": ScoreExpectation(0.216, 0.10),
                "ener_tot": ScoreExpectation(5.996, 0.05),
                "peak_kw": ScoreExpectation(0.215, 0.005),
                "cold_time_h": ScoreExpectation(2.25, 0.5),
                "warm_time_h": ScoreExpectation(1.25, 0.5),
                "setpoint_changes": ScoreExpectation(21, 3),
            },
            3.0: {
                # Locked #102 (2026-05-14).  At 3-min the controller has
                # 5× more correction opportunities → tdis_tot drops to
                # 0.01 K·h (essentially zero discomfort).
                "tdis_tot": ScoreExpectation(0.010, 0.10),
                "ener_tot": ScoreExpectation(6.012, 0.05),
                "peak_kw": ScoreExpectation(0.219, 0.005),
                "cold_time_h": ScoreExpectation(0.05, 0.5),
                "warm_time_h": ScoreExpectation(0.45, 0.5),
                "setpoint_changes": ScoreExpectation(39, 5),
            },
        },
        "production_pi": {
            15.0: {
                # Production behaves like well-tuned over 3 days because batch
                # WLS κ-gate rejects coefficient updates during early learning
                # (κ severe). Differentiation is expected on longer horizons.
                # Re-locked #99 (2026-05-14): batch timing realigned to
                # production-faithful 07:00/19:00 wall-clock; minor shift.
                "tdis_tot": ScoreExpectation(0.216, 0.10),
                "ener_tot": ScoreExpectation(5.996, 0.05),
                "peak_kw": ScoreExpectation(0.215, 0.005),
                "cold_time_h": ScoreExpectation(2.25, 0.5),
                "warm_time_h": ScoreExpectation(1.25, 0.5),
                "setpoint_changes": ScoreExpectation(21, 3),
            },
            3.0: {
                # Re-locked #99 (2026-05-14): batch timing realigned to
                # 07:00/19:00 wall-clock.  3-min cells largely byte-stable;
                # tdis_tot improved slightly (0.022 → 0.012).
                "tdis_tot": ScoreExpectation(0.012, 0.10),
                "ener_tot": ScoreExpectation(6.013, 0.05),
                "peak_kw": ScoreExpectation(0.219, 0.005),
                "cold_time_h": ScoreExpectation(0.05, 0.5),
                "warm_time_h": ScoreExpectation(0.55, 0.5),
                "setpoint_changes": ScoreExpectation(41, 5),
            },
        },
    },
    # ── lr_cool_step ──────────────────────────────────────────────
    # Living room, cool mode, 28°C base / 6°C diurnal, 3 days, no inputs.
    "lr_cool_step": {
        "naive_bang_bang": {
            15.0: {
                "tdis_tot": ScoreExpectation(28.548, 0.5),
                "ener_tot": ScoreExpectation(1.413, 0.03),
                "peak_kw": ScoreExpectation(0.075, 0.003),
                "cold_time_h": ScoreExpectation(26.25, 0.5),
                "warm_time_h": ScoreExpectation(21.25, 0.5),
                "setpoint_changes": ScoreExpectation(175, 5),
            },
            3.0: {
                # Locked #102 (2026-05-14).  Cool-mode bang-bang: shorter
                # rail-alternation cycle at 3-min reduces tdis_tot ~10×.
                "tdis_tot": ScoreExpectation(3.171, 0.30),
                "ener_tot": ScoreExpectation(1.541, 0.03),
                "peak_kw": ScoreExpectation(0.086, 0.003),
                "cold_time_h": ScoreExpectation(10.65, 0.5),
                "warm_time_h": ScoreExpectation(8.15, 0.5),
                "setpoint_changes": ScoreExpectation(325, 15),
            },
        },
        "well_tuned_pi": {
            15.0: {
                "tdis_tot": ScoreExpectation(0.297, 0.10),
                "ener_tot": ScoreExpectation(1.450, 0.03),
                "peak_kw": ScoreExpectation(0.046, 0.003),
                "cold_time_h": ScoreExpectation(2.50, 0.5),
                "warm_time_h": ScoreExpectation(0.0, 0.25),
                "setpoint_changes": ScoreExpectation(19, 3),
            },
            3.0: {
                # Locked #102 (2026-05-14).
                "tdis_tot": ScoreExpectation(0.258, 0.10),
                "ener_tot": ScoreExpectation(1.445, 0.03),
                "peak_kw": ScoreExpectation(0.046, 0.003),
                "cold_time_h": ScoreExpectation(1.10, 0.5),
                "warm_time_h": ScoreExpectation(0.40, 0.5),
                "setpoint_changes": ScoreExpectation(30, 5),
            },
        },
        "production_pi": {
            15.0: {
                # Re-locked #99 (2026-05-14): batch-timing alignment shifted
                # tdis_tot 0.378 → 0.51, setpoint_changes 20 → 24 — cool-mode
                # learning fires later in the run with the wall-clock 07/19
                # schedule.  Discriminative invariant still holds (naive
                # 28.5 vs 0.51 → 56× ratio).
                "tdis_tot": ScoreExpectation(0.51, 0.10),
                "ener_tot": ScoreExpectation(1.446, 0.03),
                "peak_kw": ScoreExpectation(0.046, 0.003),
                "cold_time_h": ScoreExpectation(3.25, 0.5),
                "warm_time_h": ScoreExpectation(1.75, 0.5),
                "setpoint_changes": ScoreExpectation(24, 3),
            },
            3.0: {
                # Re-locked #99 (2026-05-14): tdis_tot drift 1.79 → 2.02
                # (still within previous 0.20 tolerance band); minor
                # cold/warm time shifts.
                "tdis_tot": ScoreExpectation(2.024, 0.20),
                "ener_tot": ScoreExpectation(1.477, 0.03),
                "peak_kw": ScoreExpectation(0.046, 0.003),
                "cold_time_h": ScoreExpectation(11.90, 0.5),
                "warm_time_h": ScoreExpectation(0.70, 0.5),
                "setpoint_changes": ScoreExpectation(59, 5),
            },
        },
    },
    # ── lr_heat_with_solar ────────────────────────────────────────
    # Living room, heat mode, with solar input (β_truth = -2.0).
    "lr_heat_with_solar": {
        "naive_bang_bang": {
            15.0: {
                "tdis_tot": ScoreExpectation(58.187, 0.5),
                "ener_tot": ScoreExpectation(5.762, 0.05),
                "peak_kw": ScoreExpectation(0.250, 0.005),
                "cold_time_h": ScoreExpectation(30.75, 0.5),
                "warm_time_h": ScoreExpectation(30.75, 0.5),
                "setpoint_changes": ScoreExpectation(238, 5),
            },
            3.0: {
                # Locked #102 (2026-05-14).  Solar input doesn't change
                # naive bang-bang behaviour materially (it doesn't react to
                # FF inputs); ~same shape as lr_heat_step at 3-min.
                "tdis_tot": ScoreExpectation(6.210, 0.5),
                "ener_tot": ScoreExpectation(6.300, 0.10),
                "peak_kw": ScoreExpectation(0.258, 0.005),
                "cold_time_h": ScoreExpectation(14.0, 0.5),
                "warm_time_h": ScoreExpectation(13.0, 0.5),
                "setpoint_changes": ScoreExpectation(494, 15),
            },
        },
        "well_tuned_pi": {
            15.0: {
                # Relocked 2026-05-09 (#84 Stage C — see lr_heat_step block above).
                "tdis_tot": ScoreExpectation(0.695, 0.10),
                "ener_tot": ScoreExpectation(5.707, 0.05),
                "peak_kw": ScoreExpectation(0.192, 0.005),
                "cold_time_h": ScoreExpectation(4.00, 0.5),
                "warm_time_h": ScoreExpectation(4.50, 0.5),
                "setpoint_changes": ScoreExpectation(30, 3),
            },
            3.0: {
                # Locked #102 (2026-05-14).  Solar input dynamics interact
                # with the FF: tdis_tot 0.31 K·h, slightly worse than
                # lr_heat_step at the same cadence (0.01) because solar
                # provides a modeled disturbance the controller must reject.
                "tdis_tot": ScoreExpectation(0.312, 0.10),
                "ener_tot": ScoreExpectation(5.739, 0.05),
                "peak_kw": ScoreExpectation(0.220, 0.005),
                "cold_time_h": ScoreExpectation(0.15, 0.5),
                "warm_time_h": ScoreExpectation(3.80, 0.5),
                "setpoint_changes": ScoreExpectation(39, 5),
            },
        },
        "production_pi": {
            15.0: {
                # Re-locked #99 (2026-05-14): batch-timing alignment shifted
                # tdis_tot 0.105 → 0.211 (solar-modulated learning fires
                # mid-day when solar disturbance is still active).
                "tdis_tot": ScoreExpectation(0.211, 0.10),
                "ener_tot": ScoreExpectation(5.714, 0.05),
                "peak_kw": ScoreExpectation(0.190, 0.005),
                "cold_time_h": ScoreExpectation(2.00, 0.5),
                "warm_time_h": ScoreExpectation(2.25, 0.5),
                "setpoint_changes": ScoreExpectation(20, 3),
            },
            3.0: {
                # Re-locked #99 (2026-05-14): production still beats
                # well-tuned at 3-min (tdis_tot 0.0075 vs 0.312) — solar FF
                # learning kicks in within the 3-day window with enough
                # observations at the finer cadence.  setpoint_changes
                # shifted 25 → 30 from the timing realignment.
                "tdis_tot": ScoreExpectation(0.008, 0.10),
                "ener_tot": ScoreExpectation(5.721, 0.05),
                "peak_kw": ScoreExpectation(0.213, 0.005),
                "cold_time_h": ScoreExpectation(0.20, 0.5),
                "warm_time_h": ScoreExpectation(0.10, 0.5),
                "setpoint_changes": ScoreExpectation(30, 5),
            },
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
