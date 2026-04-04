"""Performance metrics for HVAC benchmark.

Re-exports core metrics from benchmark_metrics.py and adds energy/COP metrics.
"""

import math

# Re-export existing metrics
from tests.benchmark_metrics import (
    compute_itae,
    compute_overshoot,
    compute_settling_time,
    count_reversals,
    count_setpoint_changes,
    compute_integral_rms,
    compute_comfort_violations,
    compute_rls_convergence,
    compute_all_metrics as _base_all_metrics,
)


def compute_energy_metrics(history):
    """Compute energy-related metrics from history with COP data.

    History entries should include 'cumulative_kwh' and 'cop' fields
    (added by the benchmark runner when COP model is active).
    """
    if not history:
        return {"total_kwh": 0.0, "avg_cop": 0.0, "kwh_per_degree_hour": 0.0}

    total_kwh = history[-1].get("cumulative_kwh", 0.0)

    # Average COP (weighted by energy input)
    cops = [h.get("cop", 0) for h in history if h.get("cop", 0) > 0]
    avg_cop = sum(cops) / len(cops) if cops else 0.0

    # Comfort-normalized energy: kWh per degree-hour maintained
    # (degree-hours = sum of |desired - outdoor| * dt_hours)
    degree_hours = 0.0
    for h in history:
        dt_hours = 15.0 / 60.0  # assume 15-min ticks
        degree_hours += abs(h.get("desired", 20) - h.get("outdoor", 5)) * dt_hours

    kwh_per_dh = total_kwh / degree_hours if degree_hours > 0 else 0.0

    return {
        "total_kwh": round(total_kwh, 3),
        "avg_cop": round(avg_cop, 2),
        "kwh_per_degree_hour": round(kwh_per_dh, 4),
    }


def compute_all_metrics(history, desired=None, deadband=0.5, include_energy=True):
    """Compute all metrics including energy."""
    base = _base_all_metrics(history, desired, deadband)
    if include_energy:
        base.update(compute_energy_metrics(history))
    return base
