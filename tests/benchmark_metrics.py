"""Performance metrics for PID+RLS controller benchmarking.

All functions take a history list from _run_simulation() and return scalar metrics.
History entries have: tick, room_temp, desired, hp_setpoint, integral, ff_offset,
error, outdoor, rls_obs_count.
"""

import math


def _derive_tick_minutes(history):
    """Derive the wall-clock minutes-per-tick from history entries.

    History entries should carry a ``"minute"`` field (added 2026-05-10
    when the bench moved to wall-clock metrics).  Falls back to 15.0
    for legacy callers that haven't populated ``minute`` yet.
    """
    if len(history) >= 2 and "minute" in history[0] and "minute" in history[1]:
        dt = history[1]["minute"] - history[0]["minute"]
        if dt > 0:
            return dt
    return 15.0


def _entry_minute(h, tick_minutes):
    """Wall-clock minute of a history entry, with legacy fallback."""
    if "minute" in h:
        return h["minute"]
    return h["tick"] * tick_minutes


def compute_itae(history, deadband=0.5):
    """Integral Time-weighted Absolute Error, in degree·minutes².

    Discrete approximation of ``∫ t |e(t)| dt`` (Åström/Hägglund §3),
    where ``t`` is wall-clock minutes from start.  Deadband-adjusted:
    ignores errors within the deadband since those are caused by 1°C
    quantization, not controller failure.

    Cadence-invariant: same wall-clock-duration scenario produces
    matching ITAE values at any tick cadence (modulo discretization
    error that decreases as cadence shrinks).  Pre-2026-05-10 the
    weighting used tick *index* instead of minutes, making ITAE values
    incomparable across cadences (5× ticks ≈ 24× weight inflation).
    """
    if not history:
        return 0.0
    tick_minutes = _derive_tick_minutes(history)
    itae = 0.0
    for h in history:
        effective_error = max(0.0, abs(h["error"]) - deadband)
        t = _entry_minute(h, tick_minutes)
        itae += t * effective_error * tick_minutes
    return itae


def compute_overshoot(history, desired=None):
    """Maximum deviation past setpoint during scenario (°C).

    Returns the peak error magnitude across the entire scenario.
    """
    if desired is None:
        desired = history[0]["desired"] if history else 0.0
    return max(abs(h["room_temp"] - desired) for h in history) if history else 0.0


def compute_settling_time(history, deadband=0.5):
    """First wall-clock minute after which ``|error|`` stays within
    deadband for the remainder of the run.

    Returns None if the system never settles, 0.0 if always within
    deadband.  Pre-2026-05-10 returned a tick index (cadence-coupled);
    now returns minutes for cadence-portable comparison.
    """
    if not history:
        return None
    tick_minutes = _derive_tick_minutes(history)
    # Walk backwards, find last entry outside deadband.
    last_outside_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if abs(history[i]["error"]) >= deadband:
            last_outside_idx = i
            break
    if last_outside_idx == -1:
        return 0.0  # Always within deadband
    if last_outside_idx == len(history) - 1:
        return None  # Never settled
    # Settling time = wall-clock minute of the next entry (when system
    # was first within deadband and stayed there).
    next_h = history[last_outside_idx + 1]
    return _entry_minute(next_h, tick_minutes)


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

    Returns (violation_ticks, max_consecutive, max_undershoot).  Counts
    are in tick units (cadence-dependent — 1 tick = ``tick_minutes``
    of real time, configurable via ``constants.TICK_MINUTES_DEFAULT`` /
    ``--tick-minutes``).  For cadence-portable comparison divide by
    total tick count for fraction, or use ``violation_minutes =
    violation_ticks * tick_minutes`` if a wall-clock figure is needed.
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
