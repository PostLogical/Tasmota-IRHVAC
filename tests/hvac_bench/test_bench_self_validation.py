"""Bench self-validation: tests that the bench is honest about what the
controller observed.

These tests exist because of #84: the bench was leaking real wall-clock
into `time.time()` (and via `_time.time = lambda: ...` in `full_stack_runner`,
plus a subtler leak in `adapters.py` where `time.time` was never patched at
all on the `reference_scenarios` path). The leak made `Observation.wall_time`
advance by microseconds-of-test-runtime instead of sim hours, which collapsed
sin/cos ToD features to constant — making the FF model unable to learn
diurnal patterns, and silently invalidating cadence-dependent verdicts (see
the deleted `test_pi_overcorrection_diagnosis.py`).

These tests catch the bug class going forward:

1. `test_wall_time_spans_sim_duration` — direct check that observation
   wall_time covers the sim's 24-hour span. Would have caught the leak.
2. `test_tod_features_cover_diurnal_cycle` — derived check that the sin/cos
   feature pair traverses the unit circle. Would have caught the leak even
   if wall_time semantics changed.
3. `test_kpis_consistent_across_cadence` — regression catcher for cadence-
   coupled bench artifacts. Phase 3b's original `tdis_tot` non-monotone-in-
   cadence finding was real; Phase 3c's locked verdict (action-rate ringing,
   4.5x sp_changes) was bench-artifact. This test pins the post-#84
   behavior so any future cadence-coupled regression surfaces immediately.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.tasmota_irhvac.pi.model_input_manager import tod_features
from tests.hvac_bench.reference_scenarios import (
    CANONICAL_SCENARIOS,
    make_well_tuned_for_scenario,
    run_reference_scenario,
)


# ── Test 1: wall_time covers sim duration ─────────────────────────────────


def test_wall_time_spans_sim_duration():
    """Observation `wall_time` must span the sim's 24-hour duration.

    Direct regression catcher for the #84 wall-clock leak: prior to Stage B,
    `time.time()` was either unpatched (reference_scenarios path) or patched
    inconsistently (full_stack_runner path), so `wall_time` reflected
    test-runtime microseconds instead of sim hours.
    """
    base = CANONICAL_SCENARIOS["lr_heat_step"]
    scenario = replace(base, n_days=1.0)
    controller = make_well_tuned_for_scenario(scenario)
    run_reference_scenario(controller, scenario)

    buf = controller.pi._observation_buffer_heat.get_all()
    times = [o.wall_time for o in buf if o.wall_time > 0]
    assert len(times) >= 30, (
        f"observation buffer too sparse for the test ({len(times)} obs); "
        "scenario should have populated the buffer"
    )
    span_seconds = max(times) - min(times)
    # Allow a generous floor: 1-day sim should produce ~86400s span; require
    # ≥80000s to leave room for tick-boundary edge cases. A wall-clock-leaked
    # bench would produce a span on the order of milliseconds.
    assert span_seconds >= 80000, (
        f"observation wall_time span = {span_seconds:.1f}s; expected ~86400s "
        f"for a 24h sim. Wall-clock leak likely — observations are not "
        f"advancing with sim_clock."
    )


# ── Test 2: ToD features traverse the diurnal cycle ──────────────────────


def test_tod_features_cover_diurnal_cycle():
    """sin/cos ToD features must traverse the unit circle over a 24h sim.

    Derived check: even if `wall_time` semantics shift in the future,
    the sin/cos feature pair as a whole must cover the diurnal cycle for
    the FF model to be able to learn time-of-day patterns. A near-constant
    sin/cos pair signals either a wall-clock leak (the case #84 fixed) or
    a future regression in `tod_features()` itself.
    """
    base = CANONICAL_SCENARIOS["lr_heat_step"]
    scenario = replace(base, n_days=1.0)
    controller = make_well_tuned_for_scenario(scenario)
    run_reference_scenario(controller, scenario)

    buf = controller.pi._observation_buffer_heat.get_all()
    sins: list[float] = []
    coss: list[float] = []
    for o in buf:
        if o.wall_time <= 0:
            continue
        s, c = tod_features(o.wall_time)
        sins.append(s)
        coss.append(c)
    assert len(sins) >= 30, "buffer too sparse for coverage check"

    # Over a full 24h, sin and cos each traverse [-1, +1] — span ≥ 1.5
    # leaves room for tick-boundary truncation. Constant features (the leak
    # signature) would produce span ≪ 0.1.
    assert max(sins) - min(sins) >= 1.5, (
        f"sin_hour range too narrow ({max(sins) - min(sins):.3f}); "
        f"expected ≥1.5 over 24h. ToD features may be stuck at near-constant."
    )
    assert max(coss) - min(coss) >= 1.5, (
        f"cos_hour range too narrow ({max(coss) - min(coss):.3f}); "
        f"expected ≥1.5 over 24h. ToD features may be stuck at near-constant."
    )


# ── Test 3: KPIs consistent across reasonable tick cadences ──────────────


@pytest.mark.slow
def test_kpis_consistent_across_cadence():
    """Comfort metrics should not vary wildly across reasonable tick rates.

    Phase 3b found `tdis_tot` non-monotone in cadence on the buggy bench.
    Under the post-#84 sim-coherent bench, real cadence sensitivity DOES
    exist — but smaller than Stage B's bench suggested. The current
    well-tuned values on `lr_heat_step` are ~1.07 / 0.22 / 0.0 K·h at
    5/15/30 min. The 5-min cadence has higher comfort dissatisfaction
    because the controller's gains were tuned for a 15-min nominal cadence.

    This test does NOT lock the cadence-dependence to specific numbers
    — that's a future research question (gain scheduling vs cadence).
    Instead, it pins an *absolute ceiling* that catches:

    1. A cadence where comfort blows up (e.g., a cadence-coupled artifact
       that returns the controller to the buggy-bench baseline of
       ~1.4 K·h at 5-min — close to the current ceiling).
    2. A spread between cadences that grows beyond what real
       gain-scheduling-mismatch produces.

    If/when production grows gain scheduling that matches gains to
    cadence, these thresholds should tighten — but the test then becomes
    a regression catcher for that fix rather than a "bench is honest"
    sentinel.
    """
    base = CANONICAL_SCENARIOS["lr_heat_step"]
    cadences = [5.0, 15.0, 30.0]
    tdis_values: dict[float, float] = {}
    for tick_min in cadences:
        scenario = replace(base, tick_minutes=tick_min)
        controller = make_well_tuned_for_scenario(scenario)
        _, bundle = run_reference_scenario(controller, scenario)
        tdis_values[tick_min] = bundle.tdis_tot

    max_tdis = max(tdis_values.values())
    spread = max(tdis_values.values()) - min(tdis_values.values())

    # Ceiling: 1.5 K·h covers the current 5-min worst (~1.07) plus margin.
    assert max_tdis < 1.5, (
        f"max tdis_tot = {max_tdis:.4f} K·h across cadences "
        f"(values: {tdis_values}); expected <1.5 K·h under sim-coherent bench. "
        f"A cadence-coupled artifact may have regressed."
    )
    # Spread: 1.5 K·h again (paired with ceiling).
    assert spread < 1.5, (
        f"tdis_tot spread = {spread:.4f} K·h across cadences "
        f"(values: {tdis_values}); expected <1.5 K·h."
    )
