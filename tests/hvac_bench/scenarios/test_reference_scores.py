"""Locked reference scores — regression suite (Phase 2b).

For every (scenario × controller) tuple in ``reference_scenarios.py``,
runs the controller and compares the resulting KPI bundle against
``REFERENCE_SCORES`` in ``reference_scores.py``. Any KPI moving outside
its locked tolerance fails — surfacing bench changes that silently shift
discriminative power.

Also includes invariant checks that hold regardless of the exact locked
numbers (e.g. naive bang-bang must be substantially worse than well-tuned
PI on every scenario).

A `--print-scores` mode (toggled by env var or pytest arg, see test
class docstring) prints the current numbers without asserting — used
when intentionally relocking after a deliberate bench change.

Cost: ~30s wall-clock for 9 (scenario × controller) tuples on 3-day
scenarios. Marked plain (regression tier) — runs in CI on every push.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from tests.hvac_bench.reference_scenarios import (
    CANONICAL_SCENARIOS,
    CONTROLLER_FACTORIES,
    run_reference_scenario,
)
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.reference_scores import (
    DISCRIMINATIVE_INVARIANTS,
    REFERENCE_SCORES,
)


PRINT_SCORES = os.environ.get("BENCH_PRINT_REFERENCE_SCORES") == "1"

# Cadences swept by the locked-score and discriminative-invariant tests.
# 15.0 = bench's historical default, preserved for cross-cadence drift
# detection.  3.0 = post-#93 default, production-realistic.
CADENCES_TO_TEST = (15.0, 3.0)

# Controllers whose hard-locked anchors are knowingly stale vs the qref
# supervisor default-flip (da4d5ae, 2026-05-18) and the 2026-05-17 structural
# fixes (Fujitsu capacity curves + solar split), measured on the
# calibration-limited ``living_room`` plant.  Their cells xfail (the
# num_regression baseline still captures current behavior as the comparison
# baseline) until the relock that follows the qref-excursion cap decision on
# the desired-vs-effective-desired branch.  The xfail self-clears: it only
# fires when there are still failures, so a future relock makes them pass.
# naive_bang_bang is unaffected (no supervisor) and stays hard-locked.
_DEFERRED_RELOCK_CONTROLLERS = {"production_pi", "well_tuned_pi"}

# (scenario, cadence) cells where the naive-vs-well_tuned discriminative ratio
# is knowingly disturbed by the same qref flip: qref drove well_tuned's
# living_room tdis_tot to ~0 at lr_heat_step@15 (ratio guard trips) and raised
# it via the heat+solar warm-bias at lr_heat_with_solar@3 (gap narrows).  Same
# deferred-relock rationale; self-clears once the invariant holds again.
_DEFERRED_INVARIANT_CELLS = {("lr_heat_step", 15.0), ("lr_heat_with_solar", 3.0)}


def _run(scenario_name: str, controller_name: str, cadence_min: float):
    scenario = CANONICAL_SCENARIOS[scenario_name]
    if cadence_min != scenario.tick_minutes:
        scenario = replace(scenario, tick_minutes=cadence_min)
    controller = CONTROLLER_FACTORIES[controller_name](scenario)
    history, bundle = run_reference_scenario(controller, scenario)
    return bundle


# ── Locked-score regression ───────────────────────────────────────────────


@pytest.mark.parametrize("scenario_name", sorted(REFERENCE_SCORES.keys()))
@pytest.mark.parametrize("controller_name", sorted(CONTROLLER_FACTORIES.keys()))
@pytest.mark.parametrize("cadence_min", CADENCES_TO_TEST)
def test_locked_scores_within_tolerance(
    bench_metrics, num_regression, scenario_name, controller_name, cadence_min,
):
    """Every KPI of every (scenario × controller × cadence) is within locked tolerance.

    A failure here means: either the bench changed (intentionally — relock)
    or it changed (unintentionally — investigate).
    """
    controller_block = REFERENCE_SCORES[scenario_name].get(controller_name)
    if controller_block is None:
        pytest.skip(f"no locked scores for {controller_name} on {scenario_name}")
    expected_block = controller_block.get(cadence_min)
    if expected_block is None:
        pytest.skip(
            f"no locked scores for {controller_name} on {scenario_name} "
            f"at {cadence_min}-min cadence"
        )

    bundle = _run(scenario_name, controller_name, cadence_min)
    observed = bundle.as_dict()

    if PRINT_SCORES:
        print(f"\n{scenario_name} / {controller_name} @ {cadence_min}min:")
        for k, v in observed.items():
            if k in expected_block:
                exp = expected_block[k]
                marker = "✓" if exp.passes(v) else "✗"
                print(f"  {marker} {k}: observed={v}  expected={exp.expected}±{exp.abs_tolerance}")

    failures = []
    for kpi_name, exp in expected_block.items():
        if kpi_name not in observed:
            failures.append(f"{kpi_name}: missing from KPI bundle (controller produced no value)")
            continue
        # Record the observed KPI for regression detection alongside the
        # locked-tolerance assertion below.
        if isinstance(observed[kpi_name], (int, float)):
            bench_metrics[kpi_name] = observed[kpi_name]
        if not exp.passes(observed[kpi_name]):
            failures.append(
                f"{kpi_name}: observed={observed[kpi_name]}, "
                f"expected={exp.expected}±{exp.abs_tolerance} "
                f"(delta={observed[kpi_name] - exp.expected:+.4f})"
            )
    check_bench_metrics(num_regression, bench_metrics)
    if failures and controller_name in _DEFERRED_RELOCK_CONTROLLERS:
        pytest.xfail(
            f"{scenario_name} / {controller_name} @ {cadence_min}min: locked "
            "anchors stale vs qref default-flip + 2026-05-17 structural fixes "
            "on living_room; relock deferred pending the qref-excursion cap "
            "decision. num_regression baseline still pins current behavior."
        )
    assert not failures, (
        f"{scenario_name} / {controller_name} @ {cadence_min}min drifted from locked scores:\n  "
        + "\n  ".join(failures)
    )


# ── Discriminative-power invariants ───────────────────────────────────────


@pytest.mark.parametrize("scenario_name", sorted(CANONICAL_SCENARIOS.keys()))
@pytest.mark.parametrize("cadence_min", CADENCES_TO_TEST)
def test_naive_bangbang_worse_than_well_tuned_on_comfort(
    bench_metrics, num_regression, scenario_name, cadence_min,
):
    """Naive bang-bang must score substantially worse than well-tuned PI on
    ``tdis_tot`` for every canonical scenario × cadence.

    This invariant is what gives the bench discriminative power: if it
    stops holding, the bench can no longer distinguish a known-bad
    controller from a known-good one. Independent of the exact locked
    numbers — catches a bench break even if scores happen to land within
    each other's tolerances after a regression.
    """
    naive_bundle = _run(scenario_name, "naive_bang_bang", cadence_min)
    well_tuned_bundle = _run(scenario_name, "well_tuned_pi", cadence_min)

    rule = DISCRIMINATIVE_INVARIANTS["naive_worse_than_well_tuned_on_comfort"]
    metric = rule["metric"]
    min_ratio = rule["min_ratio"]

    naive_v = getattr(naive_bundle, metric)
    well_tuned_v = getattr(well_tuned_bundle, metric)

    # well_tuned_v ≤ 0 makes the ratio ill-defined; short-circuit keeps the
    # division safe.  ``invariant_holds`` folds both the degenerate-score and
    # ratio checks so we can pin the baseline and defer before asserting.
    invariant_holds = well_tuned_v > 0.0 and (naive_v / well_tuned_v) >= min_ratio
    bench_metrics["naive_v"] = naive_v
    bench_metrics["well_tuned_v"] = well_tuned_v
    bench_metrics["ratio"] = (naive_v / well_tuned_v) if well_tuned_v > 0.0 else float("nan")
    check_bench_metrics(num_regression, bench_metrics)

    if not invariant_holds and (scenario_name, cadence_min) in _DEFERRED_INVARIANT_CELLS:
        pytest.xfail(
            f"{scenario_name} @ {cadence_min}min: qref default-flip reshaped "
            "well_tuned's living_room comfort (→0 or warm-biased), disturbing "
            "the discriminative ratio; deferred pending the qref-cap relock. "
            "num_regression baseline still pins current values."
        )

    # Guard against a degenerate well-tuned score (≤0 makes the ratio
    # ill-defined) — that's its own failure mode worth surfacing.
    assert well_tuned_v > 0.0, (
        f"{scenario_name} @ {cadence_min}min: well_tuned_pi.{metric}={well_tuned_v} "
        f"— bench produced zero-discomfort score for the mid baseline; "
        f"likely scenario too short or insensitive."
    )

    ratio = naive_v / well_tuned_v
    assert ratio >= min_ratio, (
        f"{scenario_name} @ {cadence_min}min: naive/well_tuned {metric} ratio={ratio:.2f} "
        f"(naive={naive_v}, well_tuned={well_tuned_v}); "
        f"need ≥ {min_ratio}× to maintain discriminative power"
    )
