"""Solution verification with Roache GCI + tick-rate spread (Phase 3b).

Runs ``naive_bang_bang`` and ``well_tuned_pi`` against the canonical
``lr_heat_step`` and ``lr_cool_step`` scenarios at three tick rates
(30, 15, 5 minutes), feeds the resulting KPI bundles into
``kpi_richardson_sweep``, and asserts that:

1. The KPI value at the finest tick is within tolerance of the
   locked number (catches bench drift on the production-relevant cell).
2. The combined discretization-sensitivity bound
   ``max(GCI, tick_rate_spread)`` is below the locked ceiling
   (catches kernel/controller grid-convergence shape regressions).
3. The Roache asymptotic-regime classification matches the locked
   expectation (flags a bench change that flips a previously-
   asymptotic cell to non-asymptotic or vice versa).
4. Specific shape invariants hold (e.g. well-tuned PI's heating
   ``tdis_tot`` must remain in Roache asymptotic regime — its
   warm-up transient is a proper kernel-discretization error).

Why both GCI and spread:

* For "in asymptotic regime" cells the Richardson power-law fit is
  reliable, GCI = 1.25·|finest-extrap| is the literature-grounded
  bound, and ``max(GCI, spread) ≈ GCI``.
* For "non-asymptotic" cells the fit is degenerate (negative
  extrapolated values, or the sequence going the wrong direction
  under refinement); spread is the truer bound, GCI is conservative
  with Fs=3.0, and we take ``max(GCI, spread)`` for safety.

This dual approach handles the bang-bang limit-cycle aliasing case
(KPI math is "in regime" but extrapolation is non-physical because
hysteresis cycle rate IS the tick rate, per Åström-Wittenmark) without
KPI-semantic-aware logic in the Richardson module.

Runtime: ~12-18s wall-clock for the four (scenario × controller) cells
× three tick rates. Marked ``slow`` because of the 5-minute tick cells.

Solar scenario excluded — its hard-coded ``tick_minutes=15.0`` solar
schedule aliases against the sweep, and β_solar is on the Phase 4
fidelity track.

Print mode: ``BENCH_PRINT_RICHARDSON=1 pytest -s -m slow
tests/hvac_bench/scenarios/test_solution_verification.py`` prints the
observed Richardson tables — used during relock.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import pytest

from tests.hvac_bench.kpis import KpiBundle
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.reference_scenarios import (
    CANONICAL_SCENARIOS,
    CONTROLLER_FACTORIES,
    ReferenceScenario,
    run_reference_scenario,
)
from tests.hvac_bench.reference_solution_verification import (
    CONVERGENCE_INVARIANTS,
    RICHARDSON_SCORES,
    RICHARDSON_TICK_MINUTES,
)
from tests.hvac_bench.richardson import (
    RichardsonReport,
    format_richardson_table,
    kpi_richardson_sweep,
    tick_rate_spread,
)


PRINT_RICHARDSON = os.environ.get("BENCH_PRINT_RICHARDSON") == "1"


# ── Sweep harness ─────────────────────────────────────────────────────────


def _scenario_at_tick(scenario_name: str, tick_minutes: float) -> ReferenceScenario:
    """Return a copy of the canonical scenario with ``tick_minutes`` overridden."""
    base = CANONICAL_SCENARIOS[scenario_name]
    return replace(base, tick_minutes=tick_minutes)


def _run_sweep(
    scenario_name: str,
    controller_name: str,
    tick_rates: tuple[float, ...],
) -> dict[float, KpiBundle]:
    """Run a (scenario × controller) at each tick rate, collect KPI bundles."""
    factory = CONTROLLER_FACTORIES[controller_name]
    out: dict[float, KpiBundle] = {}
    for tick in tick_rates:
        scenario = _scenario_at_tick(scenario_name, tick)
        controller = factory(scenario)
        _, bundle = run_reference_scenario(controller, scenario)
        out[tick] = bundle
    return out


def _combined_bound(report: RichardsonReport) -> float:
    """``max(GCI, tick_rate_spread)`` — the conservative discretization bound."""
    return max(report.gci, tick_rate_spread(report))


# ── Locked-score regression ───────────────────────────────────────────────


@pytest.mark.study
@pytest.mark.parametrize("scenario_name", sorted(RICHARDSON_SCORES.keys()))
@pytest.mark.parametrize("controller_name", sorted({
    c for s in RICHARDSON_SCORES.values() for c in s
}))
def test_solution_verification_within_tolerance(bench_metrics, num_regression, scenario_name, controller_name):
    """Locked finest-tick value, combined bound, and regime classification.

    A failure here means: either the bench's discretization profile has
    changed (intentional → relock with ``BENCH_PRINT_RICHARDSON=1``) or
    the kernel / controller's grid-convergence behavior has shifted
    (unintentional → investigate).
    """
    expected_block = RICHARDSON_SCORES[scenario_name].get(controller_name)
    if expected_block is None:
        pytest.skip(
            f"no Richardson scores for {controller_name} on {scenario_name}"
        )

    bundles = _run_sweep(scenario_name, controller_name, RICHARDSON_TICK_MINUTES)
    reports = kpi_richardson_sweep(bundles, kpi_names=tuple(expected_block.keys()))

    if PRINT_RICHARDSON:
        print(f"\n{scenario_name} / {controller_name}:")
        print(format_richardson_table(reports))

    failures: list[str] = []
    for kpi_name, exp in expected_block.items():
        rep = reports[kpi_name]
        observed_value = rep.kpi_values[-1]
        observed_bound = _combined_bound(rep)
        observed_regime = rep.in_asymptotic_regime

        bench_metrics[f"{kpi_name}__value"] = observed_value
        bench_metrics[f"{kpi_name}__bound"] = observed_bound

        if not exp.passes_value(observed_value):
            failures.append(
                f"{kpi_name}: value={observed_value:.4f} at h={rep.tick_rates_minutes[-1]}min "
                f"(expected {exp.value_at_finest}±{exp.value_tolerance})"
            )
        if not exp.passes_bound(observed_bound):
            failures.append(
                f"{kpi_name}: bound=max(gci={rep.gci:.4f}, spread={tick_rate_spread(rep):.4f}) "
                f"= {observed_bound:.4f} > ceiling {exp.bound_max}"
            )
        if observed_regime != exp.expected_in_regime:
            failures.append(
                f"{kpi_name}: in_asymptotic_regime={observed_regime} "
                f"(expected {exp.expected_in_regime}); "
                f"order={rep.observed_order:.2f} ({rep.fit_method})"
            )
    check_bench_metrics(num_regression, bench_metrics)
    assert not failures, (
        f"{scenario_name} / {controller_name} drifted from locked solution-verification scores:\n  "
        + "\n  ".join(failures)
    )


# ── Shape invariants ──────────────────────────────────────────────────────


@pytest.mark.study
@pytest.mark.parametrize("invariant_name", sorted(CONVERGENCE_INVARIANTS.keys()))
def test_convergence_shape_invariants(bench_metrics, num_regression, invariant_name):
    """Per-rule asymptotic-regime invariants on specific (scenario × controller × KPI).

    Catches kernel/controller changes that flip the regime status of
    KPIs whose theoretical convergence behavior is well-understood.
    Independent of locked numeric values.
    """
    rule = CONVERGENCE_INVARIANTS[invariant_name]
    bundles = _run_sweep(
        rule["scenario"], rule["controller"], RICHARDSON_TICK_MINUTES,
    )
    reports = kpi_richardson_sweep(bundles, kpi_names=(rule["kpi"],))
    rep = reports[rule["kpi"]]

    if PRINT_RICHARDSON:
        print(f"\n{invariant_name} ({rule['scenario']}/{rule['controller']}/{rule['kpi']}):")
        print(format_richardson_table(reports))

    bench_metrics["observed_order"] = rep.observed_order
    bench_metrics["finest_value"] = rep.kpi_values[-1]
    bench_metrics["gci"] = rep.gci
    check_bench_metrics(num_regression, bench_metrics)
    if rule.get("must_be_in_regime", False):
        assert rep.in_asymptotic_regime, (
            f"{invariant_name}: expected asymptotic regime, got non-asymptotic; "
            f"order={rep.observed_order:.2f} ({rep.fit_method}); "
            f"sequence={rep.kpi_values}"
        )


# ── Sanity check: error bands are non-negative ────────────────────────────


@pytest.mark.study
def test_richardson_reports_have_finite_bands(bench_metrics, num_regression):
    """All RichardsonReport bands are finite and non-negative.

    Cheap sanity check that the sweep doesn't produce NaN bands (which
    would silently pass the locked-score test) for any cell.
    """
    bundles = _run_sweep(
        "lr_heat_step", "well_tuned_pi", RICHARDSON_TICK_MINUTES,
    )
    reports = kpi_richardson_sweep(bundles)

    for name, rep in reports.items():
        assert rep.error_band >= 0.0, f"{name}: error_band={rep.error_band} < 0"
        assert not math.isnan(rep.error_band), f"{name}: error_band is NaN"
        assert rep.gci >= 0.0, f"{name}: gci={rep.gci} < 0"
        assert not math.isnan(rep.gci), f"{name}: gci is NaN"
        assert tick_rate_spread(rep) >= 0.0, (
            f"{name}: spread < 0 (max < min?!)"
        )
        bench_metrics[f"{name}__error_band"] = rep.error_band
        bench_metrics[f"{name}__gci"] = rep.gci
    check_bench_metrics(num_regression, bench_metrics)
