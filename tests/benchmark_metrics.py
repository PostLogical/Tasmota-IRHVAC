"""Performance metrics for PID+RLS controller benchmarking.

All functions take a history list from _run_simulation() and return scalar metrics.
History entries have: tick, room_temp, desired, hp_setpoint, integral, ff_offset,
error, outdoor, rls_obs_count.
"""

import math


def compute_itae(history, deadband=0.5):
    """Integral Time-weighted Absolute Error, deadband-adjusted.

    Ignores errors within the deadband since those are caused by 1°C
    quantization, not controller failure. Weights later errors more
    heavily to penalize sustained deviations.
    """
    itae = 0.0
    for h in history:
        effective_error = max(0.0, abs(h["error"]) - deadband)
        itae += h["tick"] * effective_error
    return itae


def compute_overshoot(history, desired=None):
    """Maximum deviation past setpoint during scenario (°C).

    Returns the peak error magnitude across the entire scenario.
    """
    if desired is None:
        desired = history[0]["desired"] if history else 0.0
    return max(abs(h["room_temp"] - desired) for h in history) if history else 0.0


def compute_settling_time(history, deadband=0.5):
    """First tick after which |error| stays within deadband for the remainder.

    Returns None if the system never settles.
    """
    if not history:
        return None
    # Walk backwards to find last tick outside deadband
    last_outside = -1
    for h in reversed(history):
        if abs(h["error"]) >= deadband:
            last_outside = h["tick"]
            break
    if last_outside == -1:
        return 0  # Always within deadband
    if last_outside == history[-1]["tick"]:
        return None  # Never settled
    return last_outside + 1


def count_reversals(history):
    """Count direction changes in HP setpoint sequence."""
    reversals = 0
    last_dir = 0
    prev_sp = None
    for h in history:
        sp = h["hp_setpoint"]
        if prev_sp is not None and sp != prev_sp:
            direction = 1 if sp > prev_sp else -1
            if last_dir != 0 and direction != last_dir:
                reversals += 1
            last_dir = direction
        prev_sp = sp
    return reversals


def count_setpoint_changes(history):
    """Total number of HP setpoint changes (fewer = less compressor wear)."""
    changes = 0
    prev_sp = None
    for h in history:
        if prev_sp is not None and h["hp_setpoint"] != prev_sp:
            changes += 1
        prev_sp = h["hp_setpoint"]
    return changes


def compute_integral_rms(history):
    """RMS of integral over scenario. Lower = FF is doing its job."""
    if not history:
        return 0.0
    sum_sq = sum(h["integral"] ** 2 for h in history)
    return math.sqrt(sum_sq / len(history))


def compute_comfort_violations(history, threshold=1.0):
    """Count ticks where room is 1°C+ below desired (too cold).

    Returns (violation_ticks, max_consecutive, max_undershoot).
    Each tick represents 15 minutes of real time.
    """
    if not history:
        return 0, 0, 0.0
    violation_ticks = 0
    max_consecutive = 0
    current_consecutive = 0
    max_undershoot = 0.0
    for h in history:
        undershoot = h["desired"] - h["room_temp"]
        if undershoot > threshold:
            violation_ticks += 1
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
            max_undershoot = max(max_undershoot, undershoot)
        else:
            current_consecutive = 0
    return violation_ticks, max_consecutive, round(max_undershoot, 2)


def compute_rls_convergence(history):
    """RLS observation count at final tick."""
    if not history:
        return 0
    return history[-1].get("rls_obs_count", 0)


def compute_all_metrics(history, desired=None, deadband=0.5):
    """Compute all metrics and return as dict."""
    viol_ticks, viol_consec, viol_max = compute_comfort_violations(history)
    return {
        "itae": round(compute_itae(history, deadband), 2),
        "overshoot": round(compute_overshoot(history, desired), 2),
        "settling_time": compute_settling_time(history, deadband),
        "reversals": count_reversals(history),
        "setpoint_changes": count_setpoint_changes(history),
        "integral_rms": round(compute_integral_rms(history), 2),
        "rls_obs_count": compute_rls_convergence(history),
        "cold_ticks": viol_ticks,          # ticks 1°C+ below desired
        "cold_max_run": viol_consec,       # longest consecutive cold streak
        "cold_max_undershoot": viol_max,   # worst undershoot (°C)
    }
