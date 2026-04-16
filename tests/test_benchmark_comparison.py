"""Benchmark comparison: old PI+RLS (rls-model) vs new PID+RLS (rls-rework).

Run with: python -m pytest tests/test_benchmark_comparison.py -v -s
The -s flag is important to see the comparison table output.
"""

import json
from pathlib import Path

import pytest

from tests.test_pi_scenarios import (
    ThermalModel,
    SimEntity,
    _run_simulation,
    _make_sim_config,
)
from tests.benchmark_metrics import compute_all_metrics


# ── Controller configurations ────────────────────────────────────────────

# Approximate rls-model behavior (before principled rework)
OLD_CONFIG = {
    "pi_ki": 0.05,
    "pi_kd": 0.0,  # No derivative
    "pi_kd_filter_n": 8,
    "pi_setpoint_weight": 1.0,  # Full P gain (old default in tests)
}

# Current rls-rework defaults
NEW_CONFIG = {
    "pi_ki": 0.08,
    "pi_kd": 1.0,  # Filtered derivative
    "pi_kd_filter_n": 8,
    "pi_setpoint_weight": 0.5,  # 2-DOF weight
}


# ── Scenario definitions ─────────────────────────────────────────────────

HOUSE_TYPES = [
    ("drafty", 30),
    ("typical", 60),
    ("insulated", 120),
]

SEED_FACTORS = [0.0, 0.5, 1.0, 1.5]


def _outdoor_cold_snap(tick):
    """Outdoor drops from 10°C to -5°C over 32 ticks."""
    return max(-5.0, 10.0 - tick * 1.25)


def _outdoor_bunkroom(tick):
    """Outdoor drops from 2°C to -3°C over 32 ticks."""
    return 2.0 - tick * (5.0 / 32.0)


def _solar_morning(tick):
    """Solar ramps up from tick 8 to tick 20."""
    if tick < 8:
        return 0.0
    if tick > 20:
        return 1.0
    return (tick - 8) / 12.0


SCENARIOS = [
    {
        "name": "cold_start",
        "initial_temp": 17.0,
        "desired": 20.5,
        "outdoor": 5.0,
        "n_ticks": 32,
    },
    {
        "name": "cold_snap",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 10.0,
        "outdoor_schedule": _outdoor_cold_snap,
        "n_ticks": 32,
    },
    {
        "name": "setpoint_up",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 5.0,
        "desired_schedule": {10: 22.5},
        "n_ticks": 32,
    },
    {
        "name": "setpoint_down",
        "initial_temp": 22.5,
        "desired": 22.5,
        "outdoor": 5.0,
        "desired_schedule": {10: 20.5},
        "n_ticks": 32,
    },
    {
        "name": "steady_state",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 5.0,
        "n_ticks": 48,
    },
    {
        "name": "solar_morning",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 5.0,
        "solar_schedule": _solar_morning,
        "n_ticks": 32,
    },
    {
        "name": "bunkroom",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 2.0,
        "outdoor_schedule": _outdoor_bunkroom,
        "tau_override": 23,
        "n_ticks": 48,
        "seed_factors": [1.0],  # Bunkroom only tests at seed=1.0
    },
]


# ── Runner ────────────────────────────────────────────────────────────────


def _run_scenario(scenario, house_name, tau, seed_factor, config_overrides):
    """Run a single scenario and return metrics."""
    sim_config = _make_sim_config(seed_factor, **config_overrides)
    entity = SimEntity(sim_config)
    entity._pi._desired_temp = scenario["desired"]

    tau_actual = scenario.get("tau_override", tau)
    thermal = ThermalModel(
        initial_temp=scenario["initial_temp"],
        outdoor_temp=scenario["outdoor"],
        time_constant_min=tau_actual,
    )

    kwargs = {"n_ticks": scenario["n_ticks"]}
    if "outdoor_schedule" in scenario:
        kwargs["outdoor_schedule"] = scenario["outdoor_schedule"]
    if "solar_schedule" in scenario:
        kwargs["solar_schedule"] = scenario["solar_schedule"]
    if "desired_schedule" in scenario:
        kwargs["desired_schedule"] = scenario["desired_schedule"]

    # _run_simulation is sync (creates its own event loop internally)
    history = _run_simulation(entity, thermal, **kwargs)

    return compute_all_metrics(history, desired=scenario["desired"])


# ── Test that collects and prints comparison ──────────────────────────────


class TestBenchmarkComparison:
    """Run all scenarios with old and new configs, print comparison table."""

    def test_full_comparison(self, capsys):
        """Run complete benchmark comparison and print results."""
        results = []

        for scenario in SCENARIOS:
            seed_factors = scenario.get("seed_factors", SEED_FACTORS)
            house_types = HOUSE_TYPES
            if scenario.get("tau_override"):
                house_types = [("custom", scenario["tau_override"])]

            for house_name, tau in house_types:
                for seed in seed_factors:
                    label = f"{scenario['name']}|{house_name}|seed={seed}"

                    old_metrics = _run_scenario(
                        scenario, house_name, tau, seed, OLD_CONFIG
                    )
                    new_metrics = _run_scenario(
                        scenario, house_name, tau, seed, NEW_CONFIG
                    )

                    results.append({
                        "scenario": label,
                        "old": old_metrics,
                        "new": new_metrics,
                    })

        # Print comparison table
        with capsys.disabled():
            _print_comparison(results)

        # Also write JSON for further analysis
        out_dir = Path(__file__).resolve().parent.parent / "local" / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Basic sanity: at least some scenarios ran
        assert len(results) > 0


def _print_comparison(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print("BENCHMARK COMPARISON: OLD (rls-model approx) vs NEW (rls-rework)")
    print("=" * 120)

    header = f"{'Scenario':<45} {'Metric':<18} {'OLD':>10} {'NEW':>10} {'Δ':>10} {'Winner':>8}"
    print(header)
    print("-" * 120)

    metrics_to_compare = [
        ("itae", "ITAE", True),           # lower is better
        ("overshoot", "Overshoot°C", True),  # lower is better
        ("settling_time", "Settle(ticks)", True),  # lower is better
        ("reversals", "Reversals", True),  # lower is better
        ("setpoint_changes", "SP changes", True),  # lower is better
        ("integral_rms", "Integral RMS", True),  # lower is better
        ("rls_obs_count", "RLS obs", False),  # higher is better (more learning)
    ]

    # Summary counters
    wins = {"old": 0, "new": 0, "tie": 0}

    for r in results:
        first_metric = True
        for key, label, lower_is_better in metrics_to_compare:
            old_val = r["old"][key]
            new_val = r["new"][key]

            # Handle None (never settled)
            if old_val is None and new_val is None:
                delta_str = "—"
                winner = "tie"
            elif old_val is None:
                delta_str = "—"
                winner = "new" if lower_is_better else "old"
            elif new_val is None:
                delta_str = "—"
                winner = "old" if lower_is_better else "new"
            else:
                delta = new_val - old_val
                delta_str = f"{delta:+.2f}" if isinstance(delta, float) else f"{delta:+d}"
                if abs(delta) < 0.01:
                    winner = "tie"
                elif (delta < 0) == lower_is_better:
                    winner = "NEW"
                else:
                    winner = "OLD"

            old_str = f"{old_val:.2f}" if isinstance(old_val, float) else str(old_val)
            new_str = f"{new_val:.2f}" if isinstance(new_val, float) else str(new_val)

            scenario_label = r["scenario"] if first_metric else ""
            print(f"{scenario_label:<45} {label:<18} {old_str:>10} {new_str:>10} {delta_str:>10} {winner:>8}")
            first_metric = False

            if winner in ("NEW", "new"):
                wins["new"] += 1
            elif winner in ("OLD", "old"):
                wins["old"] += 1
            else:
                wins["tie"] += 1

        print()  # Blank line between scenarios

    print("=" * 120)
    print(f"SUMMARY: NEW wins {wins['new']}, OLD wins {wins['old']}, Ties {wins['tie']}")
    print("=" * 120)
