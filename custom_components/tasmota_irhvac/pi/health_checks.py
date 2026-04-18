"""Pure-function health checks for PIController diagnostics.

Each function evaluates one aspect of controller health and returns
an (alert_message, reason_code, severity) tuple, or None if healthy.
PIController.get_health_status() calls these and assembles the result.

Tuning-repair functions (check_slope_divergence_repair, etc.) return
(translation_key, placeholders, should_create) for HA Repairs issues.
"""

from __future__ import annotations

from typing import Any


def check_comfort(
    error_c: float,
    warn_threshold: float,
    crit_threshold: float,
) -> tuple[str, str, str] | None:
    """Check temperature error against comfort thresholds."""
    if error_c > crit_threshold:
        return (
            f"Temperature {error_c:.1f}°C from setpoint — comfort critical",
            "comfort_critical",
            "Critical",
        )
    if error_c > warn_threshold:
        return (
            f"Temperature {error_c:.1f}°C from setpoint — comfort warning",
            "comfort_warn",
            "Warning",
        )
    return None


def check_integral(
    ki_integral: float,
    threshold: float,
) -> tuple[str, str, str] | None:
    """Check PI integral correction magnitude (in °C)."""
    if ki_integral > threshold:
        return (
            f"PI integral correction {ki_integral:.1f}°C — controller struggling",
            "integral_high",
            "Warning",
        )
    return None


def check_ff_confidence(
    ff_confidence: float,
    threshold: float = 0.5,
) -> tuple[str, str, str] | None:
    """Check if FF confidence is sustained low."""
    if ff_confidence < threshold:
        return (
            f"FF confidence at {ff_confidence:.0%} — model prediction unreliable",
            "ff_confidence_low",
            "Warning",
        )
    return None


def check_intercept_drift(
    intercept: float,
    threshold: float,
    has_observations: bool,
) -> tuple[str, str, str] | None:
    """Check RLS intercept drift from expected near-zero."""
    if has_observations and abs(intercept) > threshold:
        return (
            f"RLS intercept drifted to {intercept:.3f} (expect near 0)",
            "intercept_drift",
            "Warning",
        )
    return None


def check_slope_drift(
    outdoor_slope: float,
    expected_slope: float,
    drift_pct_threshold: float,
    drift_abs_floor: float,
    has_observations: bool,
) -> tuple[str, str, str] | None:
    """Check outdoor delta slope drift from seed value."""
    if not has_observations or expected_slope == 0:
        return None
    drift_abs = abs(outdoor_slope - expected_slope)
    drift_pct = (drift_abs / abs(expected_slope)) * 100
    if drift_pct > drift_pct_threshold and drift_abs > drift_abs_floor:
        return (
            f"RLS outdoor slope {outdoor_slope:.4f} drifted "
            f"{drift_pct:.0f}% from seed {expected_slope:.4f}",
            "slope_drift",
            "Warning",
        )
    return None


def check_model_drift(
    drifting_coefficients: list[tuple[int, str, int]],
) -> list[tuple[str, str, str]]:
    """Check for persistent same-direction batch corrections."""
    results = []
    for _idx, name, count in drifting_coefficients:
        results.append((
            f"Batch consistently correcting {name} in same direction "
            f"({count} cycles) — possible physical change",
            "model_drift",
            "Warning",
        ))
    return results


def check_feature_diversity(
    observations: list[Any],
    n_features: int,
    feature_names: list[str],
    min_activity_pct: float,
    min_observations: int,
) -> tuple[str, str, str] | None:
    """Check observation buffer feature diversity."""
    total_obs = len(observations)
    if total_obs < min_observations:
        return None
    starved: list[str] = []
    for j in range(2, n_features):  # skip intercept & outdoor_delta
        active = sum(
            1 for o in observations
            if j < len(o.features) and abs(o.features[j]) > 1e-6
        )
        if active / total_obs < min_activity_pct:
            name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
            starved.append(name)
    if starved:
        return (
            f"Low feature diversity: {', '.join(starved)} "
            f"active in <{min_activity_pct:.0%} of {total_obs} observations",
            "low_feature_diversity",
            "Warning",
        )
    return None


# ── Tuning repair checks (for HA Repairs panel) ────────────────────


def check_slope_divergence_repair(
    learned_slope: float,
    configured_slope: float,
    sustained_cycles: int,
    mode: str,
    create_threshold_pct: float = 30.0,
    clear_threshold_pct: float = 15.0,
    min_sustained_cycles: int = 6,
    abs_floor: float = 0.05,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if learned outdoor slope has diverged from configured seed.

    Returns (translation_key, placeholders, should_create) or None if
    the condition is in the hysteresis band (no change needed).
    """
    if configured_slope == 0:
        return None
    drift_abs = abs(learned_slope - configured_slope)
    drift_pct = (drift_abs / abs(configured_slope)) * 100

    if drift_pct > create_threshold_pct and drift_abs > abs_floor and sustained_cycles >= min_sustained_cycles:
        return (
            "slope_divergence",
            {
                "mode": mode,
                "mode_cap": mode.capitalize(),
                "learned": f"{learned_slope:.4f}",
                "configured": f"{configured_slope:.4f}",
                "drift_pct": f"{drift_pct:.0f}",
            },
            True,
        )
    if drift_pct < clear_threshold_pct:
        return (
            "slope_divergence",
            {},
            False,
        )
    # In hysteresis band — no change
    return None


def check_save_seeds_repair(
    integral_convergence: float,
    seeds_match_learned: bool,
    already_notified: bool,
    convergence_threshold: float = 2.0,
    outdoor_delta_heat: float = 0.0,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if model has converged and seeds should be saved.

    One-shot: fires once when converged, clears when seeds are saved.
    Does not re-fire if already notified (until model changes significantly).
    """
    if seeds_match_learned:
        return ("save_seeds", {}, False)

    if already_notified:
        return None

    if integral_convergence < convergence_threshold:
        return (
            "save_seeds",
            {"outdoor_delta": f"{outdoor_delta_heat:.4f}"},
            True,
        )
    return None


def check_high_integral_repair(
    ki_integral_correction: float,
    sustained_cycles: int,
    observation_count: int,
    learned_slope: float,
    configured_slope: float,
    uncontrollable_cvh: float,
    total_cvh: float,
    pi_ki: float,
    integral_convergence: float,
    mode: str,
    create_threshold: float = 2.0,
    clear_threshold: float = 1.0,
    min_sustained_cycles: int = 6,
    maturity_obs: int = 50,
    slope_gap_pct: float = 20.0,
) -> tuple[str, dict[str, str], bool] | None:
    """Diagnose high integral correction and return the most specific cause.

    Sub-cases checked in priority order:
    1. Immature model (not enough observations)
    2. FF slope gap (learned vs configured mismatch)
    3. Equipment limits (high uncontrollable fraction)
    4. Tuning (suggest Ki reduction)

    Returns None in hysteresis band.
    """
    if ki_integral_correction < clear_threshold:
        return ("high_integral_immature", {}, False)

    if ki_integral_correction < create_threshold or sustained_cycles < min_sustained_cycles:
        return None

    correction_str = f"{ki_integral_correction:.1f}"

    # Sub-case 1: Model still learning
    if observation_count < maturity_obs:
        return (
            "high_integral_immature",
            {"correction": correction_str, "count": str(observation_count)},
            True,
        )

    # Sub-case 2: FF slope gap
    if configured_slope != 0:
        gap_pct = (abs(learned_slope - configured_slope) / abs(configured_slope)) * 100
        if gap_pct > slope_gap_pct:
            return (
                "high_integral_slope_gap",
                {
                    "mode": mode,
                    "configured": f"{configured_slope:.4f}",
                    "learned": f"{learned_slope:.4f}",
                    "correction": correction_str,
                },
                True,
            )

    # Sub-case 3: Equipment at limits
    if total_cvh > 0 and uncontrollable_cvh / total_cvh > 0.5:
        return (
            "high_integral_equipment",
            {"correction": correction_str},
            True,
        )

    # Sub-case 4: Tuning — suggest specific Ki
    suggested_ki = pi_ki * (1.0 / ki_integral_correction)
    suggested_ki = max(0.01, min(suggested_ki, pi_ki))  # clamp to reasonable range
    return (
        "high_integral_tuning",
        {
            "correction": correction_str,
            "current_ki": f"{pi_ki:.3f}",
            "suggested_ki": f"{suggested_ki:.3f}",
        },
        True,
    )


def check_covariance_collapse_repair(
    coeff_index: int,
    coeff_name: str,
    coeff_value: float,
    clamp: tuple[float, float] | None,
    p_diagonal: float,
    delta: float = 0.001,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if a coefficient is stuck at its clamp with collapsed uncertainty.

    When P[i,i] ≈ delta and the coefficient is at a clamp boundary,
    online RLS learning cannot recover — the covariance has collapsed.
    """
    if clamp is None:
        return None

    lo, hi = clamp
    at_lower = abs(coeff_value - lo) < 1e-3
    at_upper = abs(coeff_value - hi) < 1e-3

    if not (at_lower or at_upper):
        # Coefficient not at clamp — clear any existing issue
        return ("covariance_collapse", {}, False)

    if p_diagonal < 3 * delta:
        clamp_value = lo if at_lower else hi
        return (
            "covariance_collapse",
            {
                "coeff_name": coeff_name,
                "value": f"{coeff_value:.4f}",
                "clamp_value": f"{clamp_value:.4f}",
                "p_diagonal": f"{p_diagonal:.6f}",
            },
            True,
        )

    # At clamp but P hasn't collapsed yet — no issue
    return None


def check_model_drift_repair(
    drifting_coefficients: list[tuple[int, str, int]],
    has_had_stable_batch: bool,
    min_consecutive: int = 5,
) -> list[tuple[str, dict[str, str], bool]]:
    """Check for persistent model drift with maturity gate.

    Suppresses drift alerts until at least one batch cycle has had
    recommend_update=False (system stabilized at least once).
    """
    results = []
    if not has_had_stable_batch:
        return results

    for idx, name, count in drifting_coefficients:
        if count < min_consecutive:
            continue
        direction = "upward" if count > 0 else "downward"
        # Per-coefficient suggestions
        if name == "outdoor_delta":
            suggestion = "Check window seals, insulation, or HVAC ducting changes."
        elif name == "intercept":
            suggestion = "Check sensor calibration or look for an unmodeled heat/cool source."
        else:
            suggestion = f"Check whether {name} has changed (different fuel, settings, schedule)."

        results.append((
            "model_drift",
            {
                "coeff_name": name,
                "direction": direction,
                "count": str(abs(count)),
                "suggestion": suggestion,
            },
            True,
        ))
    return results


def check_intercept_absorbing_repair(
    intercept_value: float,
    coefficients: list[tuple[str, float, tuple[float, float] | None, float]],
    delta: float = 0.001,
    intercept_threshold: float = 1.0,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if intercept has grown large by absorbing a clamped coefficient's effect.

    coefficients: list of (name, value, clamp, p_diagonal) for non-intercept coefficients.
    """
    if abs(intercept_value) < intercept_threshold:
        return ("intercept_absorbing", {}, False)

    # Find any coefficient with covariance collapse at its clamp
    for name, value, clamp, p_diag in coefficients:
        if clamp is None:
            continue
        lo, hi = clamp
        at_clamp = abs(value - lo) < 1e-3 or abs(value - hi) < 1e-3
        collapsed = p_diag < 3 * delta
        if at_clamp and collapsed:
            clamp_value = lo if abs(value - lo) < 1e-3 else hi
            return (
                "intercept_absorbing",
                {
                    "intercept_value": f"{intercept_value:.2f}",
                    "absorbed_name": name,
                    "absorbed_clamp": f"{clamp_value:.4f}",
                },
                True,
            )

    return None


def check_batch_online_disagreement_repair(
    coeff_index: int,
    coeff_name: str,
    drift_signs: list[int],
    current_beta: float,
    last_blended_beta: float | None,
    min_same_direction: int = 3,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if batch and online learning are fighting over a coefficient.

    Detected when the last N batch corrections are in the same direction
    but the current beta has drifted back in the opposite direction.
    """
    if last_blended_beta is None or len(drift_signs) < min_same_direction:
        return None

    recent = drift_signs[-min_same_direction:]
    # Check all recent corrections are in the same direction (all +1 or all -1)
    if not (all(s == 1 for s in recent) or all(s == -1 for s in recent)):
        return None

    correction_direction = recent[0]  # +1 or -1
    drift_back = current_beta - last_blended_beta

    # Disagreement: current beta drifted in opposite direction from corrections
    if correction_direction * drift_back < 0 and abs(drift_back) > 0.01:
        direction = "upward" if correction_direction == 1 else "downward"
        return (
            "batch_online_disagreement",
            {
                "coeff_name": coeff_name,
                "direction": direction,
            },
            True,
        )

    # Corrections in same direction but no drift-back — corrections are sticking
    return ("batch_online_disagreement", {}, False)


def check_residual_pattern_repair(
    start_hour: int,
    end_hour: int,
    mean_residual: float,
    n_observations: int,
    sustained_cycles: int,
    create_threshold: float = 0.5,
    clear_threshold: float = 0.3,
    min_sustained_cycles: int = 3,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if a time-of-day residual pattern indicates an unmodeled disturbance.

    A consistent positive residual means the HP needs more offset than the
    model predicts (unmodeled heat loss at that time). Negative means
    unmodeled heat gain (solar, occupancy, scheduled heating).

    Args:
        start_hour: start of the pattern span (0-23).
        end_hour: end of the pattern span (0-23).
        mean_residual: signed mean residual in °C.
        n_observations: total observations in the span.
        sustained_cycles: how many consecutive batch cycles detected this.
        create_threshold: minimum |mean_residual| to create issue.
        clear_threshold: |mean_residual| below which to clear.
        min_sustained_cycles: minimum sustained cycles before creating.
    """
    if abs(mean_residual) < clear_threshold:
        return ("residual_pattern", {}, False)

    if abs(mean_residual) < create_threshold or sustained_cycles < min_sustained_cycles:
        return None  # hysteresis band

    if mean_residual < 0:
        direction = "negative"
        cause_hint = "unmodeled heat gain (common causes: solar gain, occupancy, scheduled heating from another source)"
    else:
        direction = "positive"
        cause_hint = "unmodeled heat loss (common causes: drafts, scheduled ventilation, door/window opening patterns)"

    if start_hour == end_hour:
        time_range = f"{start_hour:02d}:00–{start_hour:02d}:59"
    else:
        time_range = f"{start_hour:02d}:00–{end_hour:02d}:59"

    return (
        "residual_pattern",
        {
            "time_range": time_range,
            "direction": direction,
            "mean_residual": f"{mean_residual:+.2f}",
            "n_observations": str(n_observations),
            "cause_hint": cause_hint,
        },
        True,
    )


def check_multicollinearity_repair(
    condition_number: float,
    correlated_pairs: list[tuple[str, str, float]],
    sustained_cycles: int,
    create_threshold: float = 30.0,
    clear_threshold: float = 20.0,
    min_sustained_cycles: int = 3,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if features are multicollinear (condition number too high).

    When two input features are highly correlated, the RLS cannot
    separate their effects and coefficient estimates become unstable.

    Args:
        condition_number: spectral condition number √(λ_max/λ_min) of X^T X + λI.
        correlated_pairs: (name_i, name_j, r) for pairs with |r| > 0.7.
        sustained_cycles: how many consecutive batch cycles above threshold.
        create_threshold: condition number above which to create issue.
            Belsley (1980): κ > 30 = moderate multicollinearity.
        clear_threshold: condition number below which to clear issue.
            κ < 20 = weak dependencies, coefficients reliable.
        min_sustained_cycles: minimum sustained cycles before creating.

    Reference: Belsley, Kuh & Welsch, "Regression Diagnostics" (1980), Ch. 3.
    """
    if condition_number < clear_threshold:
        return ("multicollinearity", {}, False)

    if condition_number < create_threshold or sustained_cycles < min_sustained_cycles:
        return None  # hysteresis band

    # Build human-readable pair list
    if correlated_pairs:
        pair_strs = [f"{a} and {b} (r={r:.2f})" for a, b, r in correlated_pairs[:3]]
        pairs_text = "; ".join(pair_strs)
    else:
        pairs_text = "unknown (condition number high but no single pair dominates)"

    return (
        "multicollinearity",
        {
            "condition_number": f"{condition_number:.0f}",
            "pairs": pairs_text,
        },
        True,
    )
