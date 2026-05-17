"""Over-temp recovery scenario: stylized 04-25-like pathology test.

REGRESSION TEST for the user-reported 2026-04-25 LR pathology:
solar-driven afternoon overshoot → regime gate fires → solar wanes →
cold overnight → does the room undershoot desired by 2°F+ (the user's
comfort-gap threshold)?

This test does NOT use the LR calibrated profile (which has 3 rails per
Reynders 2014 — non-sensical fit on passive operational data). Instead
it uses a literature-typical 2R2C profile so we're testing the controller's
behavior against believable physics, not a degenerate fit.

If this test PASSES, the regime gate + bumpless transfer chain handles
the synthetic version of the failure mode. The user-reported pathology
either (a) requires real-world dynamics the bench doesn't capture, or
(b) is already fixed and the original observation was specific to pre-pre44
controller behavior.

If this test FAILS, we've reproduced the failure mode in controlled
conditions and can iterate on the fix.
"""

from __future__ import annotations

import math

import pytest

from tests.hvac_bench.disturbances import Disturbance
from tests.hvac_bench.full_stack_runner import FullStackConfig, run_full_stack


# ── Scenario definition ──────────────────────────────────────────────────


def outdoor_schedule_aprlike(minute: float) -> float:
    """Spring-shoulder diurnal outdoor temperature in °C.

    Modeled on the user's 04-25 conditions: cold morning low, mild
    afternoon high, cold overnight.  Sinusoidal with min at 04:00 local
    (~ minute 240) and max at 14:00 local (~ minute 840).
    """
    # Phase: cosine with min at 04:00 (240 min) and max at 16:00 (960 min).
    # Period = 24h. Amplitude = 4°C. Mean = 4°C.
    day_minute = minute % 1440.0
    # Shift so that t=0 is midnight, peak is at t=14h:
    phase = 2 * math.pi * (day_minute - 14 * 60) / 1440.0
    return 4.0 + 4.0 * math.cos(phase)


def solar_overshoot_disturbance(day: int = 0) -> Disturbance:
    """A 7h afternoon solar pulse strong enough to push room ~5°C above
    desired, triggering the regime gate.

    Start at 10:00 local on the given day, duration 7h, ramp 60min.
    heat_gain_c_per_min=0.15 chosen to overshoot desired meaningfully
    against ~150min envelope cooling. First-cut numbers may need tuning.
    """
    start_minute = day * 1440.0 + 10 * 60.0
    return Disturbance(
        name=f"solar_pulse_day{day}",
        heat_gain_c_per_min=0.15,
        start_minute=start_minute,
        duration_minutes=7 * 60.0,
        ramp_minutes=60.0,
    )


# ── Test ─────────────────────────────────────────────────────────────────


class TestOverTempRecovery:
    """Stylized 04-25 pathology: solar overshoot → regime → cold night."""

    def test_overnight_does_not_deeply_undershoot(self, bench_metrics):
        """After a solar-overshoot day, room should not sustainably
        undershoot desired by ≥2°F (~1.1°C) overnight.

        Threshold matches the user-reported comfort-gap symptom: the
        04-25 production data showed 3°F undershoot sustained for 3+
        hours. If HEAD's controller handles this scenario correctly,
        overnight room min should stay within 2°F of desired.
        """
        desired_c = 20.0  # 68°F (slightly above the user's 67°F desired)
        config = FullStackConfig(
            n_days=2,  # 2 days: warm-up day + pathological day
            profile_name="standard_residential_2r2c",
            desired_c=desired_c,
            mode="heat",
            outdoor_schedule=outdoor_schedule_aprlike,
            noise_sigma=0.1,   # realistic sensor noise per feedback_test_noise_realism
            thermal_disturbances=[
                # Solar pulse on day 1 (after day 0 warm-up)
                solar_overshoot_disturbance(day=1),
            ],
            head_sensor_offset=0.0,  # idealized — HP sees what we see
        )
        result = run_full_stack(config)
        history = result.history

        # Find peak room temp on day 1 (verify the disturbance actually
        # triggered overshoot)
        day1_start_min = 1 * 1440.0
        day1_history = [h for h in history if h["minute"] >= day1_start_min]
        peak_temp = max(h["room_temp"] for h in day1_history)
        bench_metrics["day1_peak_room_c"] = peak_temp
        bench_metrics["day1_peak_overshoot_c"] = max(0.0, peak_temp - desired_c)

        # Overnight window: 2200 → 0700 next day (in minutes-of-day terms)
        # Use day 1 evening + day 2 early morning.
        overnight_start = 1 * 1440.0 + 22 * 60.0  # day 1, 22:00
        overnight_end = 2 * 1440.0 + 7 * 60.0     # day 2, 07:00
        overnight = [
            h for h in history
            if overnight_start <= h["minute"] < overnight_end
        ]
        assert len(overnight) > 30, "Not enough overnight samples — check schedule"

        overnight_min = min(h["room_temp"] for h in overnight)
        bench_metrics["overnight_min_room_c"] = overnight_min
        bench_metrics["overnight_max_undershoot_c"] = max(0.0, desired_c - overnight_min)

        # How long was the room below desired - 1.0°C (≈ 2°F)?
        threshold = desired_c - 1.0
        below_count = sum(1 for h in overnight if h["room_temp"] < threshold)
        below_minutes = below_count * config.tick_minutes
        bench_metrics["overnight_minutes_under_2F"] = below_minutes

        # Deepest negative integral overnight (catches the bumpless-transfer
        # negative-wind-up mode I theorized from the open-loop replay)
        overnight_integ = [h.get("integral", 0.0) for h in overnight]
        deepest_neg_integ = min(overnight_integ) if overnight_integ else 0.0
        bench_metrics["deepest_neg_integral"] = deepest_neg_integ

        # Did regime gate fire?  The runner doesn't capture _overtemp_regime
        # in history, so we detect it indirectly: integral stays unchanged
        # for a sustained period while room is over desired (regime freezes
        # integration).  Look for ≥30min of |Δintegral| < 0.01 while room
        # is > desired + 1.0°C.
        regime_minutes = 0.0
        prev_integ = None
        for h in day1_history:
            if h["room_temp"] > desired_c + 1.0 and prev_integ is not None:
                if abs(h["integral"] - prev_integ) < 0.01:
                    regime_minutes += config.tick_minutes
            prev_integ = h["integral"]
        bench_metrics["regime_minutes_estimated"] = regime_minutes

        # Prerequisite 1: meaningful overshoot
        assert peak_temp > desired_c + 2.0, (
            f"Disturbance didn't trigger meaningful overshoot "
            f"(peak {peak_temp:.2f}°C, desired {desired_c}°C). "
            f"Increase solar_overshoot_disturbance.heat_gain_c_per_min."
        )
        # Prerequisite 2: regime gate must actually fire (integral frozen for
        # a sustained period during overshoot)
        assert regime_minutes >= 30.0, (
            f"Regime gate didn't fire long enough (estimated {regime_minutes:.0f} "
            f"min frozen-integral during overshoot — need ≥30 to be a meaningful "
            f"regime-exit test).  Peak {peak_temp:.2f}°C may not have triggered "
            f"the gate (entry threshold 1.0°C above desired + uncontrollable latch)."
        )

        # ── CORE ASSERTION ────────────────────────────────────────
        # Overnight, room should NOT sustainably undershoot desired by
        # 2°F (1.1°C).  Sustained = more than 60 min.
        assert below_minutes < 60.0, (
            f"Overnight pathology reproduced: room was below "
            f"{threshold:.1f}°C ({desired_c - threshold:.1f}°C under desired) "
            f"for {below_minutes:.0f} minutes overnight. "
            f"Min temp: {overnight_min:.2f}°C. "
            f"Peak overshoot: {peak_temp:.2f}°C. "
            f"Deepest integral: {deepest_neg_integ:+.2f}. "
            f"Regime fired for {regime_active_count * config.tick_minutes:.0f} min."
        )
