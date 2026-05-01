"""5-minute PI overcorrection — empirical mechanism diagnosis (Phase 3c).

Phase 3b found that ``well_tuned_pi`` on ``lr_cool_step`` produces
``tdis_tot`` increasing as h decreases (0.155 → 0.415 → 1.406 K·h at
30/15/5 min). The Richardson regime classifier flags this as
non-asymptotic. Phase 3c's job per the plan is to *diagnose* the
mechanism — Oberkampf-Roy V&V §8: "diagnose, don't hide."

## Hypothesis space (per the plan)

a. **Thermal-model integration scheme inadequate at high cadence** —
   unlikely. The 2R2C thermal model uses matrix-exponential propagation
   per :class:`tests.hvac_bench.thermal_model.ThermalModel2R2C`, which
   is exact for the linearised dynamics regardless of step size.

b. **PI's anti-windup or hold-time interacting badly with finer ticks
   — the working hypothesis.** Production PI has
   ``pi_setpoint_hold = 1200s`` (20-minute minimum dwell after a
   setpoint change, bypassed only by ``abs_error > 1.0°C``). At 5-min
   ticks the controller evaluates 4× per hold window but can only act
   once. Each evaluation uses the latest state to compute a candidate
   setpoint; only the first post-hold evaluation actually applies.
   Between evaluations the integral keeps accumulating (correctly,
   via dt_factor normalisation), but the controller's *action* lags
   the observed evolution.

c. **Genuine controller overcorrection that should be exposed as a
   known property** — likely overlapping with (b). At fine ticks, the
   FF samples outdoor more accurately, so during diurnal swings the
   FF fires harder; the integral accumulates concurrently; when the
   hold finally expires, the combined raw setpoint can overshoot.

This test empirically distinguishes (b) and (c). It runs the same
scenario at 30-min and 5-min ticks and locks a diagnostic signature
showing:

* Hold-blocked tick rate (fraction of ticks where the PI computed a
  new candidate setpoint but the hold prevented application). Higher
  at finer ticks confirms (b).
* Mean ``|raw_setpoint - hp_setpoint|`` gap (the difference between
  what the PI wants right now and what's actually applied). Larger
  at finer ticks confirms the hold *is* the active mechanism.
* Setpoint-changes per hour (controlled by both bypass and hold
  expiry). Comparable at fine vs coarse if the hold dominates;
  divergent if the bypass fires more at one rate than the other.
* Mean absolute error inside the deadband.

The locked numbers serve as a regression: any future change to the
hold mechanism, integral handling, or FF coupling that reshapes this
signature should be a deliberate commit.

## Verdict (locked at 2026-05-01)

Observed signature on lr_cool_step / well_tuned_pi (3 days):

| Metric | 5-min | 30-min | ratio |
|---|---|---|---|
| hold_block_ratio | 0.17 | 0.00 | — |
| mean abs(raw - hp) gap | 0.23 | 0.22 | 1.0x |
| setpoint_changes_per_hour | 0.89 | 0.19 | 4.5× |
| mean abs error | 0.34 | 0.18 | 1.9× |

The mechanism is **(c) action-rate-driven mild ringing**, with (b)
playing a secondary role:

* The hold *is* engaged at 5-min (17% of ticks have a candidate
  setpoint blocked), confirming (b) as a real but partial factor.
* The wanted-vs-applied gap is the *same* magnitude at both rates
  (~0.22) — so the hold doesn't dominate. Whatever the PI wants now,
  it converges to similar values at both cadences.
* The *dominant* signature is a 4.5× increase in setpoint changes
  per hour at 5-min. After warm-up, post-hold the controller can act
  every 20-min hold-expiry (i.e. every 4 ticks at 5-min) vs every
  single tick at 30-min — so the *effective* action rate is ~50%
  faster at fine ticks, not 6× as the raw tick ratio would suggest.
* Faster action with the same gain produces mild ringing, doubling
  the mean absolute error (0.18 → 0.34) and pushing the
  ``tdis_tot`` integral up (0.155 → 1.406 K·h).

This is the textbook signature of a controller running at finer
ticks than its gain-tuning was calibrated for. The PI gains
(``Kp = 1.5``, ``Ki = 0.15``) were tuned for a 15-minute nominal
cadence (``pi_tick_fallback = 900s``); at 5-min the gain-per-action
is the same but the actions land closer together → underdamped
response.

## Implication

5-minute ticks are not a bench fidelity problem — they are a
controller property the bench should *expose*, exactly per Oberkampf-
Roy V&V §8 ("diagnose, don't hide"). Hypothesis (a) — kernel
discretization scheme — is rejected: the matrix-exp propagation in
``ThermalModel2R2C`` is exact for the linearised dynamics, and naive
bang-bang's ``ener_tot`` (which has no controller cleverness) shows
clean monotone convergence under refinement.

For production, this suggests gain scheduling against the actual
sensor cadence — at fine ticks, ``Kp / Ki`` should scale down — but
that's a production-side question outside Phase 3 scope.

## Phase 3c contract

This test fails if the diagnostic signature *shape* changes
materially:

* Hold-block-ratio at 5-min staying near zero would indicate the
  hold mechanism stopped engaging.
* Setpoint-change rate ratio collapsing toward 1.0 would indicate
  the hold expiry (20-min) no longer dominates the action cadence.
* Mean abs error ratio collapsing toward 1.0 would indicate the
  bench stopped being able to expose tick-rate-coupled overcorrection.

Each of those would be a deliberate bench/controller change worth
flagging, not a regression to silently absorb.
"""

from __future__ import annotations

import math
import os
import time as _time
from dataclasses import replace
from typing import Any

import pytest

from tests.hvac_bench.full_stack_runner import diurnal_outdoor
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.reference_scenarios import (
    CANONICAL_SCENARIOS,
    make_well_tuned_for_scenario,
)
from tests.hvac_bench.thermal_model import COPModel, ThermalModel2R2C


PRINT_DIAGNOSIS = os.environ.get("BENCH_PRINT_PI_DIAGNOSIS") == "1"


# ── Instrumented runner ───────────────────────────────────────────────────


def _run_instrumented(tick_minutes: float) -> dict[str, float]:
    """Run lr_cool_step / well_tuned_pi at ``tick_minutes`` with PI-internal capture.

    Uses the canonical lr_cool_step scenario, swapped to the requested
    tick rate. Captures per-tick:

    * ``raw - hp`` gap (PI's wanted vs actually-applied setpoint).
    * Hold-blocked indicator (`time_since_last < SETPOINT_HOLD_SECONDS`
      AND `abs_error <= 1.0` AND the rounded raw differs from current).
    * Setpoint-change marker.
    * Absolute error.

    Returns aggregated diagnostic stats.
    """
    base = CANONICAL_SCENARIOS["lr_cool_step"]
    scenario = replace(base, tick_minutes=tick_minutes)
    profile = PROFILES_2R2C[scenario.profile_name]

    controller = make_well_tuned_for_scenario(scenario)
    adapter = controller.adapter
    pi = controller.pi

    initial_temp = scenario.initial_temp_c if scenario.initial_temp_c is not None else scenario.desired_c
    model = ThermalModel2R2C(
        profile=profile,
        initial_temp=initial_temp,
        outdoor_temp=scenario.outdoor_base_c,
        cop_model=COPModel(),
        sensor_noise_sigma=scenario.noise_sigma,
        noise_seed=scenario.noise_seed,
    )

    controller.set_mode(scenario.mode)
    controller.set_desired_temp(scenario.desired_c)

    n_ticks = int(scenario.n_days * 24 * 60 / tick_minutes)
    dt_seconds = tick_minutes * 60.0

    # Capture buffers
    raw_minus_hp: list[float] = []
    hold_blocked: list[bool] = []
    setpoint_changes = 0
    abs_errors: list[float] = []
    prev_setpoint: float | None = None

    for tick in range(n_ticks):
        model.outdoor_temp = diurnal_outdoor(
            tick, scenario.outdoor_base_c, scenario.outdoor_diurnal_c, tick_minutes,
        )
        sensor_reading = model.read_sensor()

        # Capture pre-tick state for hold-blocked detection
        prev_hp_setpoint = pi._hp_setpoint
        prev_change_time = pi._last_setpoint_change_time
        prev_sim_clock = adapter._sim_clock

        hp_setpoint = controller.tick(
            room_temp_c=sensor_reading,
            outdoor_temp_c=model.outdoor_temp,
            dt_seconds=dt_seconds,
            model_inputs=None,
        )

        model.step(
            hp_setpoint=hp_setpoint,
            dt_minutes=tick_minutes,
            tick=tick,
            mode=scenario.mode,
        )

        # Post-tick captures
        raw_setpoint = pi._last_raw_setpoint
        if math.isnan(raw_setpoint):
            raw_setpoint = float(hp_setpoint)
        rounded_raw = int(max(pi._min_temp_c, min(pi._max_temp_c, round(raw_setpoint))))

        # Hold-blocked: PI wanted to change setpoint but the hold prevented it
        # (and the bypass abs_error > 1.0 didn't fire).
        time_since_last = prev_sim_clock - prev_change_time if prev_change_time > 0 else float("inf")
        candidate_differs = rounded_raw != prev_hp_setpoint
        abs_err = abs(scenario.desired_c - model.room_temp)
        held = (
            candidate_differs
            and time_since_last < pi._SETPOINT_HOLD_SECONDS
            and abs_err <= 1.0
        )
        hold_blocked.append(held)

        raw_minus_hp.append(raw_setpoint - hp_setpoint)
        abs_errors.append(abs_err)

        if prev_setpoint is not None and hp_setpoint != prev_setpoint:
            setpoint_changes += 1
        prev_setpoint = hp_setpoint

    n = len(hold_blocked)
    hold_block_ratio = sum(hold_blocked) / n if n else 0.0
    mean_abs_raw_gap = sum(abs(g) for g in raw_minus_hp) / n if n else 0.0
    sim_hours = n * tick_minutes / 60.0
    sp_changes_per_hour = setpoint_changes / sim_hours if sim_hours else 0.0
    mean_abs_error = sum(abs_errors) / n if n else 0.0

    return {
        "tick_minutes": tick_minutes,
        "n_ticks": n,
        "hold_block_ratio": hold_block_ratio,
        "mean_abs_raw_gap": mean_abs_raw_gap,
        "setpoint_changes_per_hour": sp_changes_per_hour,
        "mean_abs_error": mean_abs_error,
    }


# ── Diagnosis test ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_5min_overcorrection_signature():
    """Locks the diagnostic signature of 5-min PI overcorrection.

    The mechanism is **action-rate-driven mild ringing** (verdict in
    module docstring): post-hold action rate ~50% faster at 5-min,
    same gains tuned for 15-min cadence → underdamped response → 2×
    mean absolute error.

    A regression that flips any of these signature shapes is a
    deliberate change worth surfacing:

    * 30-min hold_block_ratio ~ 0 (each tick exceeds the 20-min hold).
    * 5-min hold_block_ratio meaningfully positive (hold engages).
    * Setpoint-change rate at 5-min substantially higher than 30-min.
    * Mean abs error at 5-min meaningfully higher than 30-min.
    """
    fine = _run_instrumented(5.0)
    coarse = _run_instrumented(30.0)

    if PRINT_DIAGNOSIS:
        print()
        for label, stats in [("5-min", fine), ("30-min", coarse)]:
            print(f"  {label:>6}: ", end="")
            print(", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in stats.items() if k != "tick_minutes"
            ))

    # (b) hold-mechanism signature: 30-min should never block (each tick
    # exceeds the 20-min hold); 5-min should engage the hold non-trivially.
    assert coarse["hold_block_ratio"] <= 0.05, (
        f"30-min hold_block_ratio={coarse['hold_block_ratio']:.2%} > 5%; "
        "expected ~0 because each tick exceeds the 20-min hold"
    )
    assert fine["hold_block_ratio"] >= 0.10, (
        f"5-min hold_block_ratio={fine['hold_block_ratio']:.2%} < 10%; "
        "expected the hold to engage non-trivially at 5-min cadence"
    )
    assert fine["hold_block_ratio"] > coarse["hold_block_ratio"] + 0.10, (
        f"fine block_ratio={fine['hold_block_ratio']:.2%} not meaningfully "
        f"higher than coarse={coarse['hold_block_ratio']:.2%}"
    )

    # (c) action-rate-driven ringing: 5-min has substantially more
    # setpoint changes per hour and meaningfully higher mean abs error.
    # The change-rate ratio reflects post-hold action density;
    # the abs-error ratio reflects the resulting controller performance.
    sp_change_ratio = (
        fine["setpoint_changes_per_hour"]
        / max(coarse["setpoint_changes_per_hour"], 1e-6)
    )
    assert sp_change_ratio >= 2.5, (
        f"setpoint_changes/hour fine/coarse ratio={sp_change_ratio:.2f} < 2.5; "
        "expected ≥2.5× more frequent action at 5-min ticks"
    )

    err_ratio = fine["mean_abs_error"] / max(coarse["mean_abs_error"], 1e-6)
    assert err_ratio >= 1.4, (
        f"mean_abs_error fine/coarse ratio={err_ratio:.2f} < 1.4; "
        "expected ≥1.4× larger excursions at 5-min (action-rate ringing)"
    )
