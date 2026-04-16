"""Pure-function health checks for PIController diagnostics.

Each function evaluates one aspect of controller health and returns
an (alert_message, reason_code, severity) tuple, or None if healthy.
PIController.get_health_status() calls these and assembles the result.
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
