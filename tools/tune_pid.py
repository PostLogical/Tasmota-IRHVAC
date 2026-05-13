#!/usr/bin/env python3
"""PI parameter sweep: find optimal (kp, ki, kd, b, deadband) across 2R2C scenarios.

Run: python tools/tune_pid.py

Searches a grid of parameter values, evaluates composite cost across
QUICK_PROFILES × scenarios × seed factors, and reports the best parameters.

Uses the hvac_bench 2R2C infrastructure for realistic thermal dynamics.
"""

import logging
import sys
import os
import itertools
import json
import time as _time

# Suppress all logging from PI controller during sweep
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


# ── Search space ──────────────────────────────────────────────────────────

KP_VALUES = [0.8, 1.0, 1.2]
KI_VALUES = [0.16, 0.18, 0.20, 0.22, 0.24]
KD_VALUES = [0.0, 0.5, 1.0]
B_VALUES = [0.10, 0.15, 0.20, 0.25]
# Deadband held fixed: sweeping it biases the cost function (ITAE is
# deadband-adjusted), so a wider db wins by measuring less error, not
# by controlling better.  Sweep deadband separately if needed.
DEADBAND = 0.5

SEED_FACTORS = [0.0, 0.5, 1.0, 1.5]


# ── Scenarios ─────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "cold_start",
        "initial_temp": 17.0,
        "desired": 20.5,
        "outdoor": 2.0,
        "n_ticks": 32,
    },
    {
        "name": "setpoint_up",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 5.0,
        "desired_schedule": {150: 22.5},
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
        "name": "cold_snap",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 10.0,
        "outdoor_schedule": lambda minute: max(-5.0, 10.0 - minute * (1.25 / 15.0)),
        "n_ticks": 32,
    },
    {
        "name": "ramp_disturbance",
        "initial_temp": 20.5,
        "desired": 20.5,
        "outdoor": 5.0,
        "outdoor_schedule": lambda minute: 5.0 - minute / 60.0,  # 1°C/hour
        "n_ticks": 32,
    },
]


# ── Cost function ─────────────────────────────────────────────────────────

# Normalization constants (approximate max values)
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


def evaluate_params(kp, ki, kd, b):
    """Evaluate a parameter set across all scenarios. Returns average cost."""
    total_cost = 0.0
    n_evals = 0

    for profile_name, profile in QUICK_PROFILES.items():
        for scenario in SCENARIOS:
            for seed_factor in SEED_FACTORS:
                seed = profile.true_seed
                config = {
                    "pi_kp": kp,
                    "pi_ki": ki,
                    "pi_kd": kd,
                    "pi_kd_filter_n": 8,
                    "pi_setpoint_weight": b,
                    "pi_deadband": DEADBAND,
                    "pi_outdoor_seed_heat": seed * seed_factor,
                    "pi_outdoor_seed_cool": seed * seed_factor,
                }
                ctrl = TasmotaPIAdapter(config)
                ctrl.set_desired_temp(scenario["desired"])

                model = ThermalModel(
                    profile=profile,
                    initial_temp=scenario["initial_temp"],
                    outdoor_temp=scenario["outdoor"],
                    sensor_noise_sigma=0.1,  # realistic DHT noise
                    noise_seed=42,
                )

                kwargs = {"n_ticks": scenario["n_ticks"]}
                for key in ("outdoor_schedule", "solar_schedule",
                            "desired_schedule", "stove_schedule"):
                    if key in scenario:
                        kwargs[key] = scenario[key]

                history = run_scenario(ctrl, model, **kwargs)
                metrics = compute_all_metrics(history, desired=scenario["desired"],
                                              deadband=DEADBAND)
                total_cost += composite_cost(metrics)
                n_evals += 1

    return total_cost / n_evals if n_evals > 0 else float("inf")


# ── Main sweep ────────────────────────────────────────────────────────────

def main():
    grid = list(itertools.product(KP_VALUES, KI_VALUES, KD_VALUES, B_VALUES))
    n_profiles = len(QUICK_PROFILES)
    n_scenarios = len(SCENARIOS)
    n_seeds = len(SEED_FACTORS)
    evals_per = n_profiles * n_scenarios * n_seeds
    print(f"Sweeping {len(grid)} parameter combinations × {evals_per} evals each "
          f"({n_profiles} profiles × {n_scenarios} scenarios × {n_seeds} seeds)")
    print(f"Fixed deadband={DEADBAND}")
    print(f"Cost weights: ITAE={W_ITAE}, cold_ticks={W_COLD}, "
          f"reversals={W_REVERSALS}, sp_changes={W_SP_CHANGES}")
    print()

    results = []
    best_cost = float("inf")
    best_params = None
    t0 = _time.time()

    for i, (kp, ki, kd, b) in enumerate(grid):
        cost = evaluate_params(kp, ki, kd, b)
        results.append({
            "kp": kp, "ki": ki, "kd": kd, "b": b,
            "cost": round(cost, 5),
        })

        if cost < best_cost:
            best_cost = cost
            best_params = (kp, ki, kd, b)

        # Progress every 25 combos or first
        if (i + 1) % 25 == 0 or i == 0:
            elapsed = _time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(grid) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{len(grid)}] kp={kp:.1f} ki={ki:.2f} kd={kd:.1f} "
                  f"b={b:.2f} cost={cost:.5f}  "
                  f"(best: kp={best_params[0]:.1f} ki={best_params[1]:.2f} "
                  f"kd={best_params[2]:.1f} b={best_params[3]:.2f} "
                  f"cost={best_cost:.5f})  "
                  f"[{elapsed:.0f}s, ETA {eta:.0f}s]")

    # Sort by cost
    results.sort(key=lambda r: r["cost"])

    elapsed = _time.time() - t0
    print()
    print("=" * 80)
    print(f"TOP 15 PARAMETER SETS  (completed in {elapsed:.0f}s)")
    print("=" * 80)
    print(f"{'Rank':<6} {'kp':<6} {'ki':<7} {'kd':<6} {'b':<6} {'Cost':<12}")
    print("-" * 80)
    for i, r in enumerate(results[:15]):
        marker = " ← BEST" if i == 0 else ""
        print(f"{i+1:<6} {r['kp']:<6.1f} {r['ki']:<7.2f} {r['kd']:<6.1f} "
              f"{r['b']:<6.2f} {r['cost']:<12.5f}{marker}")

    print()
    print("WORST 5 (for comparison):")
    print("-" * 80)
    for r in results[-5:]:
        print(f"       {r['kp']:<6.1f} {r['ki']:<7.2f} {r['kd']:<6.1f} "
              f"{r['b']:<6.2f} {r['cost']:<12.5f}")

    # Save full results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "local", "data", "tune_pid_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    # Show best vs current defaults (from const.py)
    print()
    print("=" * 80)
    bp = best_params
    print(f"OPTIMAL:  kp={bp[0]:.1f}  ki={bp[1]:.2f}  kd={bp[2]:.1f}  "
          f"b={bp[3]:.2f}  cost={best_cost:.5f}  (deadband={DEADBAND})")
    current_cost = evaluate_params(kp=1.0, ki=0.15, kd=0.0, b=0.3)
    print(f"CURRENT:  kp=1.0  ki=0.15  kd=0.0  b=0.30  "
          f"cost={current_cost:.5f}  (deadband={DEADBAND})")
    improvement = (current_cost - best_cost) / current_cost * 100
    print(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
