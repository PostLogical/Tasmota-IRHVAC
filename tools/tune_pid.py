#!/usr/bin/env python3
"""PID parameter sweep: find optimal (ki, kd, b) across all scenarios.

Run: python tools/tune_pid.py

Searches a grid of parameter values, evaluates composite cost across all
scenarios × house types × seed factors, and reports the best parameters.
"""

import logging
import sys
import os
import itertools
import json

# Suppress all logging from PI controller during sweep
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_pi_scenarios import (
    ThermalModel,
    SimEntity,
    _run_simulation,
    _make_sim_config,
)
from tests.benchmark_metrics import compute_all_metrics


# ── Search space ──────────────────────────────────────────────────────────

KI_VALUES = [0.03, 0.05, 0.08, 0.10, 0.12]
KD_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
B_VALUES = [0.3, 0.5, 0.7, 0.9, 1.0]

# Hold kp=1.0 fixed (standard for unity-gain systems)
KP = 1.0


# ── Scenarios ─────────────────────────────────────────────────────────────

def _outdoor_cold_snap(tick):
    return max(-5.0, 10.0 - tick * 1.25)

def _outdoor_bunkroom(tick):
    return 2.0 - tick * (5.0 / 32.0)

def _solar_morning(tick):
    if tick < 8: return 0.0
    if tick > 20: return 1.0
    return (tick - 8) / 12.0

SCENARIOS = [
    {"name": "cold_start", "initial_temp": 17.0, "desired": 20.5, "outdoor": 5.0, "n_ticks": 32},
    {"name": "cold_snap", "initial_temp": 20.5, "desired": 20.5, "outdoor": 10.0,
     "outdoor_schedule": _outdoor_cold_snap, "n_ticks": 32},
    {"name": "setpoint_up", "initial_temp": 20.5, "desired": 20.5, "outdoor": 5.0,
     "desired_schedule": {10: 22.5}, "n_ticks": 32},
    {"name": "setpoint_down", "initial_temp": 22.5, "desired": 22.5, "outdoor": 5.0,
     "desired_schedule": {10: 20.5}, "n_ticks": 32},
    {"name": "steady_state", "initial_temp": 20.5, "desired": 20.5, "outdoor": 5.0, "n_ticks": 48},
    {"name": "solar_morning", "initial_temp": 20.5, "desired": 20.5, "outdoor": 5.0,
     "solar_schedule": _solar_morning, "n_ticks": 32},
    {"name": "bunkroom", "initial_temp": 20.5, "desired": 20.5, "outdoor": 2.0,
     "outdoor_schedule": _outdoor_bunkroom, "tau_override": 23, "n_ticks": 48,
     "seed_factors": [1.0]},
]

HOUSE_TYPES = [("drafty", 30), ("typical", 60), ("insulated", 120)]
SEED_FACTORS = [0.0, 0.5, 1.0, 1.5]


# ── Cost function ─────────────────────────────────────────────────────────

# Normalization constants (approximate max values from benchmark data)
ITAE_NORM = 700.0
COLD_TICKS_NORM = 32.0
REVERSALS_NORM = 20.0
SP_CHANGES_NORM = 20.0

# Weights
W_ITAE = 0.4
W_COLD = 0.3
W_REVERSALS = 0.2
W_SP_CHANGES = 0.1


def composite_cost(metrics):
    """Compute weighted composite cost from a single scenario's metrics."""
    return (
        W_ITAE * min(metrics["itae"] / ITAE_NORM, 1.0)
        + W_COLD * min(metrics["cold_ticks"] / COLD_TICKS_NORM, 1.0)
        + W_REVERSALS * min(metrics["reversals"] / REVERSALS_NORM, 1.0)
        + W_SP_CHANGES * min(metrics["setpoint_changes"] / SP_CHANGES_NORM, 1.0)
    )


def evaluate_params(ki, kd, b):
    """Evaluate a parameter set across all scenarios. Returns average cost."""
    config_overrides = {
        "pi_kp": KP,
        "pi_ki": ki,
        "pi_kd": kd,
        "pi_kd_filter_n": 8,
        "pi_setpoint_weight": b,
    }

    total_cost = 0.0
    n_scenarios = 0

    for scenario in SCENARIOS:
        seed_factors = scenario.get("seed_factors", SEED_FACTORS)
        house_types = HOUSE_TYPES
        if scenario.get("tau_override"):
            house_types = [("custom", scenario["tau_override"])]

        for house_name, tau in house_types:
            for seed in seed_factors:
                sim_config = _make_sim_config(seed, **config_overrides)
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

                history = _run_simulation(entity, thermal, **kwargs)
                metrics = compute_all_metrics(history, desired=scenario["desired"])
                total_cost += composite_cost(metrics)
                n_scenarios += 1

    return total_cost / n_scenarios if n_scenarios > 0 else float("inf")


# ── Main sweep ────────────────────────────────────────────────────────────

def main():
    grid = list(itertools.product(KI_VALUES, KD_VALUES, B_VALUES))
    print(f"Sweeping {len(grid)} parameter combinations across ~73 scenarios each...")
    print(f"Cost weights: ITAE={W_ITAE}, cold_ticks={W_COLD}, reversals={W_REVERSALS}, sp_changes={W_SP_CHANGES}")
    print()

    results = []
    best_cost = float("inf")
    best_params = None

    for i, (ki, kd, b) in enumerate(grid):
        cost = evaluate_params(ki, kd, b)
        results.append({"ki": ki, "kd": kd, "b": b, "cost": round(cost, 5)})

        if cost < best_cost:
            best_cost = cost
            best_params = (ki, kd, b)

        # Progress
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(grid)}] ki={ki:.2f} kd={kd:.1f} b={b:.1f} cost={cost:.5f}"
                  f"  (best so far: ki={best_params[0]:.2f} kd={best_params[1]:.1f} b={best_params[2]:.1f} cost={best_cost:.5f})")

    # Sort by cost
    results.sort(key=lambda r: r["cost"])

    print()
    print("=" * 70)
    print("TOP 10 PARAMETER SETS")
    print("=" * 70)
    print(f"{'Rank':<6} {'ki':<8} {'kd':<8} {'b':<8} {'Cost':<12}")
    print("-" * 70)
    for i, r in enumerate(results[:10]):
        marker = " ← BEST" if i == 0 else ""
        print(f"{i+1:<6} {r['ki']:<8.2f} {r['kd']:<8.1f} {r['b']:<8.1f} {r['cost']:<12.5f}{marker}")

    print()
    print("WORST 5 (for comparison):")
    print("-" * 70)
    for r in results[-5:]:
        print(f"       {r['ki']:<8.2f} {r['kd']:<8.1f} {r['b']:<8.1f} {r['cost']:<12.5f}")

    # Save full results
    with open("tune_pid_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to tune_pid_results.json")

    # Show best vs current
    print()
    print("=" * 70)
    print(f"OPTIMAL:  ki={best_params[0]:.2f}  kd={best_params[1]:.1f}  b={best_params[2]:.1f}  cost={best_cost:.5f}")
    current_cost = evaluate_params(0.08, 1.0, 0.5)
    print(f"CURRENT:  ki=0.08  kd=1.0  b=0.5  cost={current_cost:.5f}")
    improvement = (current_cost - best_cost) / current_cost * 100
    print(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
