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

import pytest

from tests.hvac_bench.reference_scenarios import (
    CANONICAL_SCENARIOS,
    CONTROLLER_FACTORIES,
    run_reference_scenario,
)
from tests.hvac_bench.reference_scores import (
    DISCRIMINATIVE_INVARIANTS,
    REFERENCE_SCORES,
)


PRINT_SCORES = os.environ.get("BENCH_PRINT_REFERENCE_SCORES") == "1"


def _run(scenario_name: str, controller_name: str):
    scenario = CANONICAL_SCENARIOS[scenario_name]
    controller = CONTROLLER_FACTORIES[controller_name](scenario)
    history, bundle = run_reference_scenario(controller, scenario)
    return bundle


# ── Locked-score regression ───────────────────────────────────────────────


@pytest.mark.parametrize("scenario_name", sorted(REFERENCE_SCORES.keys()))
@pytest.mark.parametrize("controller_name", sorted(CONTROLLER_FACTORIES.keys()))
def test_locked_scores_within_tolerance(scenario_name, controller_name):
    """Every KPI of every (scenario × controller) is within locked tolerance.

    A failure here means: either the bench changed (intentionally — relock)
    or it changed (unintentionally — investigate).
    """
    expected_block = REFERENCE_SCORES[scenario_name].get(controller_name)
    if expected_block is None:
        pytest.skip(f"no locked scores for {controller_name} on {scenario_name}")

    bundle = _run(scenario_name, controller_name)
    observed = bundle.as_dict()

    if PRINT_SCORES:
        print(f"\n{scenario_name} / {controller_name}:")
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
        if not exp.passes(observed[kpi_name]):
            failures.append(
                f"{kpi_name}: observed={observed[kpi_name]}, "
                f"expected={exp.expected}±{exp.abs_tolerance} "
                f"(delta={observed[kpi_name] - exp.expected:+.4f})"
            )
    assert not failures, (
        f"{scenario_name} / {controller_name} drifted from locked scores:\n  "
        + "\n  ".join(failures)
    )


# ── Discriminative-power invariants ───────────────────────────────────────


@pytest.mark.parametrize("scenario_name", sorted(CANONICAL_SCENARIOS.keys()))
def test_naive_bangbang_worse_than_well_tuned_on_comfort(scenario_name):
    """Naive bang-bang must score substantially worse than well-tuned PI on
    ``tdis_tot`` for every canonical scenario.

    This invariant is what gives the bench discriminative power: if it
    stops holding, the bench can no longer distinguish a known-bad
    controller from a known-good one. Independent of the exact locked
    numbers — catches a bench break even if scores happen to land within
    each other's tolerances after a regression.
    """
    naive_bundle = _run(scenario_name, "naive_bang_bang")
    well_tuned_bundle = _run(scenario_name, "well_tuned_pi")

    rule = DISCRIMINATIVE_INVARIANTS["naive_worse_than_well_tuned_on_comfort"]
    metric = rule["metric"]
    min_ratio = rule["min_ratio"]

    naive_v = getattr(naive_bundle, metric)
    well_tuned_v = getattr(well_tuned_bundle, metric)

    # Guard against a degenerate well-tuned score (≤0 makes the ratio
    # ill-defined) — that's its own failure mode worth surfacing.
    assert well_tuned_v > 0.0, (
        f"{scenario_name}: well_tuned_pi.{metric}={well_tuned_v} "
        f"— bench produced zero-discomfort score for the mid baseline; "
        f"likely scenario too short or insensitive."
    )

    ratio = naive_v / well_tuned_v
    assert ratio >= min_ratio, (
        f"{scenario_name}: naive/well_tuned {metric} ratio={ratio:.2f} "
        f"(naive={naive_v}, well_tuned={well_tuned_v}); "
        f"need ≥ {min_ratio}× to maintain discriminative power"
    )
