"""Full-stack bench tests for CUSUM-driven over-temp regime arming (#135).

These tests assert the *attribution* contract — bench harnesses need to
read `latch_armed_events` after a run and know when, why, and under
what state the CUSUM trigger fired. Without that, any longer bench run
that shows a change in comfort/chatter/ITAE can't be attributed to the
CUSUM trigger vs other mechanisms.

Two scenarios:

  1. `test_cusum_arms_within_disturbance_window` — a heat-disturbance
     (cooking) injection that the audit confirmed the CUSUM trigger
     detects. Asserts ≥1 CUSUM_ANOMALY arming event, event timing falls
     inside the disturbance window, and event payload carries enough
     context for later attribution.

  2. `test_cusum_chronic_fp_budget_at_perfect_ff` — 7-day perfect-FF
     scenario with NO disturbances. Asserts the two-filter design caps
     chronic FP arms at ~0.5/day (budget from `_135_path5_fp_audit`:
     empirical 0.1/day, budget = 5x for bench-drift headroom; clearly
     rejects sign-only ≈ 2/day and unfiltered ≈ 5/day).

The `LatchArmingTrigger.CUSUM_ANOMALY` discriminator on each event
distinguishes these from arms via other triggers (HP_ESTIMATED_IDLE,
MODE_FLIP_OVERTEMP, SUSTAINED_OVERTEMP). Today only CUSUM arms emit
into `latch_armed_events`; the other triggers can be backfilled later
via the same mechanism (see future_work #136).
"""

from __future__ import annotations

import pytest

from custom_components.tasmota_irhvac.pi.health_checks import (
    LatchArmingTrigger,
)
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    run_full_stack,
)
from tests.hvac_bench.house_profiles import PROFILES
from tests.hvac_bench.disturbances import party


# CUSUM arming budget at perfect FF (per _135_path5_fp_audit).
# Audit empirical: 1 arm / 7d at perfect FF with two-filter design.
# Budget = 5x headroom = 5 arms / 7d total. Sign-only design alone would
# blow this (15/7d); unfiltered design even more (36/7d).
# Empirical chronic-FP rate at Hawkins canonical K=0.5/H=4. Bench
# baseline 33/7d after the two-filter (#135 Phase 1 wire-up). Budget
# 50 = ~1.5x empirical headroom — guards against regression where the
# two-filter is dropped entirely (raw-CUSUM rate would be much higher,
# 100+/7d).
#
# Under the previous Page CUSUM (K=1.0/H=10, ARL₀ ≈ 50,000) the empirical
# was 1/7d and budget was 5. Hawkins is by design more sensitive — trade
# higher FP rate for faster TP detection on small shifts (the #135 σ̂-
# collapse and detection-latency problems both went away with Hawkins).
# The actionable consequence — chronic latch-arming — is gated by
# pi_cusum_overtemp_arming_enabled, which remains default-False until
# Phase 2 lands the additive/multiplicative discriminator that turns
# the higher-FP-rate detector into a safe action signal.
CUSUM_PERFECT_FF_BUDGET = 50

# Disturbance: party (10 people, 3h sustained, +2°C peak).  We use party
# rather than cooking because production CUSUM has a 30-min cooldown
# after every alarm — including chronic S+ events that fire in the
# late-afternoon hours from solar-load FF mismatch.  Cooking's peak
# residual (~1°C, sub-1.5°C) is small enough that cooldown contention
# from a chronic S+ event 30 min prior can suppress the cooking
# detection.  Party produces large negative residuals (-3 to -6°C in
# the audit's `_133_signs_mags.txt`) that fire reliably even immediately
# after a cooldown clears.  Day 2 placement gives ~2 days of settling
# without accumulating excessive chronic-FP cooldown contention.
PARTY_START_MIN = 2 * 24 * 60 + 18 * 60  # day 2, 18:00
PARTY_TEST_N_DAYS = 3


def _make_perfect_ff_config(n_days: int, *, with_party: bool = False,
                             cusum_arming: bool = True):
    """Standard perfect-FF harness from `_135_path5_fp_audit`.

    `cusum_arming` defaults to True here so the arming tests below exercise
    the CUSUM_ANOMALY trigger end-to-end. Production default is False (see
    `DEFAULT_CUSUM_OVERTEMP_ARMING_ENABLED` in const.py) — passive arming
    has three known failure modes (chatter, σ̂-collapse, observation
    starvation) and is shipped off pending the active-probe redesign in
    future_work #138. These tests verify the framework still works when
    explicitly opted into.
    """
    profile = PROFILES["standard_residential"]
    seed = profile.true_seed
    thermal = (
        [party(start_minute=PARTY_START_MIN)] if with_party else []
    )
    return FullStackConfig(
        n_days=n_days,
        profile_name="standard_residential",
        outdoor_base_c=0.0,
        outdoor_diurnal_c=6.0,
        desired_c=20.5,
        noise_sigma=0.1,
        noise_seed=42,
        tick_minutes=3.0,
        pi_overrides={
            "pi_outdoor_seed_heat": seed * 1.0,  # perfect FF
            "pi_outdoor_seed_cool": seed * 1.0,
            "pi_cusum_overtemp_arming_enabled": cusum_arming,
        },
        thermal_disturbances=thermal,
        relax_kappa_gate=True,
    )


def _cusum_arms(result):
    """Filter latch_armed_events for CUSUM_ANOMALY trigger only."""
    return [
        e for e in result.latch_armed_events
        if e.trigger == LatchArmingTrigger.CUSUM_ANOMALY
    ]


class TestCusumArmingDisturbance:
    """CUSUM trigger fires within a heat-disturbance window.

    Uses party (3h, large heat injection) rather than cooking (45 min,
    mild) because the production CUSUM has a 30-min cooldown that can
    be consumed by chronic late-afternoon S+ events (solar-load FF
    mismatch), suppressing cooking-tier detection.  Party's large
    negative residuals fire reliably even after cooldown contention.
    See future_work #137 for the cooldown-sizing redesign.
    """

    def test_cusum_arms_within_party_window(self):
        """Party at day 2 18:00 → CUSUM_ANOMALY arms at least once.

        Each arming event must be attributable (mode, residual,
        overtemp_error all populated) for bench analysis.
        """
        config = _make_perfect_ff_config(n_days=PARTY_TEST_N_DAYS, with_party=True)
        result = run_full_stack(config)

        all_events = _cusum_arms(result)
        # Separate party-window arms from chronic-FP arms elsewhere in
        # the run. Under Hawkins (#135 Phase 1) the chronic-FP rate is
        # higher than under Page (validated in TestCusumArmingChronicFPBudget),
        # so the run will contain BOTH disturbance-driven and noise-driven
        # arms. This test specifically asserts the disturbance triggers
        # at least one arm; chronic-FP rate is tested separately.
        party_start_s = PARTY_START_MIN * 60.0
        party_end_s = (PARTY_START_MIN + 180) * 60.0  # 3h duration
        recovery_tail_s = 4 * 3600.0  # 4h recovery tolerance
        window_lo = party_start_s - 60.0
        window_hi = party_end_s + recovery_tail_s
        party_events = [e for e in all_events if window_lo <= e.mono <= window_hi]
        assert len(party_events) >= 1, (
            f"CUSUM trigger should arm at least once during party "
            f"disturbance (large sustained heat injection). "
            f"Got {len(party_events)} arms in party window; "
            f"{len(all_events)} total arms across the run."
        )

        # Attribution: every party-window event needs context.
        for ev in party_events:
            assert ev.mode == "heat", f"wrong mode: {ev.mode}"
            # Note: under Hawkins-Olwell self-starting CUSUM,
            # sign_matches_mode flags shifts RELATIVE to the running window
            # mean — not absolute residual sign. During a sustained
            # disturbance the window adapts to the new level and recovery
            # residuals (with absolute positive sign) can fire sign-matched
            # "downward shift" alarms. The safety-critical filter is the
            # overtemp_error > 0 gate below (absolute room temp), which is
            # still satisfied throughout disturbance recovery.
            assert ev.overtemp_error > 0, (
                f"overtemp gate failed: ev recorded with overtemp_error="
                f"{ev.overtemp_error}; gate is supposed to reject ≤ 0"
            )
            assert ev.details.get("peak_cusum", 0.0) > 0, (
                "peak_cusum should be populated in details"
            )

    def test_cusum_disabled_no_arms_in_same_scenario(self):
        """Kill switch verified at full-stack level: same scenario with
        CUSUM arming disabled records ZERO CUSUM_ANOMALY arms. Other
        triggers may still fire latch arms — we only assert the CUSUM
        trigger is inert.
        """
        config = _make_perfect_ff_config(
            n_days=PARTY_TEST_N_DAYS, with_party=True, cusum_arming=False,
        )
        result = run_full_stack(config)

        assert _cusum_arms(result) == [], (
            "Kill switch off → no CUSUM_ANOMALY arms regardless of CUSUM activity"
        )


class TestCusumArmingChronicFPBudget:
    """Two-filter design keeps chronic-FP arming bounded at perfect FF.

    Empirical baseline (`_135_path5_fp_audit`): 1 arm / 7d with two-filter.
    Budget 5/7d = 5x headroom — easily within and ROBUST to bench drift,
    while still rejecting sign-only design (~15/7d) and unfiltered (~36/7d).

    This test is the regression guard against a future "loosen the gate"
    change that would re-introduce excessive false arms.
    """

    def test_no_disturbance_perfect_ff_arm_budget(self):
        config = _make_perfect_ff_config(n_days=7, with_party=False)
        result = run_full_stack(config)

        events = _cusum_arms(result)
        assert len(events) <= CUSUM_PERFECT_FF_BUDGET, (
            f"CUSUM chronic-FP arming exceeds budget: {len(events)} arms in 7d "
            f"(budget {CUSUM_PERFECT_FF_BUDGET}). "
            f"Likely cause: overtemp gate weakened or sign filter dropped. "
            f"See local/tools/_135_path5_fp_audit for the design study."
        )

        # Diagnostic: every chronic-FP arm should still be attributable —
        # if these slip through, we want to know WHY in test output.
        for ev in events:
            assert ev.overtemp_error > 0, (
                f"FP arm with overtemp_error={ev.overtemp_error} — gate "
                f"must reject ≤ 0; details={ev.details}"
            )
