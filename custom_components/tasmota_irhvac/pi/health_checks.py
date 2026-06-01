"""Pure-function health checks for PIController diagnostics.

Each function evaluates one aspect of controller health and returns
an (alert_message, reason_code, severity) tuple, or None if healthy.
PIController.get_health_status() calls these and assembles the result.

Tuning-repair functions (check_slope_divergence_repair, etc.) return
(translation_key, placeholders, should_create) for HA Repairs issues.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

# Self-starting CUSUM needs t→Normal transform. Use scipy.special (not
# scipy.stats) — same math, much smaller import surface, and avoids
# scipy.stats's eager doc generation in _new_distributions which can
# conflict with coverage's numpy reload (scipy 1.16+ behavior).
#   stdtr(df, x): Student-t CDF at x with df degrees of freedom
#   ndtri(p):    inverse standard-normal CDF (= norm.ppf)
try:
    from scipy.special import ndtri as _ndtri
    from scipy.special import stdtr as _stdtr
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover — scipy is a hard dependency in this codebase
    _SCIPY_AVAILABLE = False


# ── CUSUM anomaly detection constants ────────────────────────────────
# Basseville & Nikiforov (1993), Detection of Abrupt Changes
CUSUM_K: float = 1.0       # reference value — dead zone for shifts < 2σ
CUSUM_H: float = 10.0      # threshold — ARL₀ ≈ 50,000 ticks
MIN_RESIDUALS_FOR_DETECTION: int = 10  # minimum history for reliable MAD
MIN_SIGMA_FLOOR: float = 0.05         # absolute σ floor (°C)
MIN_EVENT_DURATION_SEC: float = 600.0  # 10 minutes wall-clock minimum
CUSUM_COOLDOWN_SEC: float = 1800.0     # 30 minutes wall-clock cooldown
# Residual history window for MAD-σ̂ estimation.  Time-based (not count-
# based) so the window stays meaningful across the wide production tick
# cadence range (60s minimum to 15min maximum, typically 3-5min average).
#
# 12h chosen from:
#   - Jensen-Jones-Farmer-Champ-Woodall 2006 minimum: ≥200 obs at the
#     stable end of cadence (3-5min) → 144-240 obs at 12h, above the
#     stability threshold for Phase-I estimation.
#   - House et al. 2006: HVAC settling time ~60min → 12h is 12× settling,
#     well past steady-state.
#   - Captures half a diurnal cycle (one full peak + one full trough) so
#     small sustained residuals from diurnal FF mismatch fall inside the
#     natural noise envelope rather than firing as anomalies.
#   - Bench validated at 60s/3min/5min cadences: at 12h all three
#     disturbance classes (party, oil_boiler 30-min step, cooking) are
#     reliably detected; at 24h+ oil_boiler is lost.
# See `local/tools/_133_cadence_sweep.txt` for the validation data.
CUSUM_RESIDUAL_WINDOW_S: float = 12 * 3600.0


@dataclass
class AnomalyEvent:
    """A completed anomalous period detected by CUSUM."""

    start_time: datetime
    start_mono: float
    end_time: datetime
    end_mono: float
    tick_count: int
    mean_residual: float      # signed, physical units (°C)
    peak_cusum: float         # max(S⁺, S⁻) — severity measure
    mode: str                 # "heat" or "cool" at detection time
    # Snapshot of room state at event time — used by event_in_overtemp
    # and the repair_qualifies / latch_qualifies filter helpers.
    # Defaults to 0.0 (= room at desired) for backwards-compatible
    # construction in test fixtures that don't model temperature state.
    current_c: float = 0.0
    desired_c: float = 0.0

    @property
    def sign_matches_mode(self) -> bool:
        """True when residual sign indicates additive-in-mode disturbance.

        Heating: residual<0 (room warmer than predicted → additive heat).
        Cooling: residual>0 (room cooler than predicted → additive cooling).
        Used by the over-temp latch trigger (with overtemp gate).
        """
        return (
            (self.mode == "heat" and self.mean_residual < 0)
            or (self.mode == "cool" and self.mean_residual > 0)
        )

    @property
    def sign_inverse_mode(self) -> bool:
        """True when residual sign indicates additive-against-mode disturbance.

        Heating: residual>0 (room cooler than predicted → additive heat-loss).
        Cooling: residual<0 (room warmer than predicted → additive heat).
        Used by HA Repairs notification (open-window / unmodelled-input in
        opposite direction from the current mode's expected disturbance).
        """
        return (
            (self.mode == "heat" and self.mean_residual > 0)
            or (self.mode == "cool" and self.mean_residual < 0)
        )


def event_in_overtemp(event: AnomalyEvent) -> bool:
    """Whether the controller was in overtemp/undertemp at event time.

    Uses the event's captured current_c/desired_c snapshot.
    "Overtemp" semantics flip with mode:
      Heating mode: current > desired (room warmer than target)
      Cooling mode: current < desired (room cooler than target)
    """
    if event.mode == "heat":
        return event.current_c > event.desired_c
    return event.current_c < event.desired_c  # cool


def latch_qualifies(event: AnomalyEvent) -> bool:
    """Production over-temp latch arming trigger: sign-match AND overtemp.

    Conservative — only fires on additive-heat-during-heating or
    additive-cool-during-cooling. Mirrors the production two-filter at
    pi_controller.py (when `_cusum_overtemp_arming_enabled` is True).
    """
    return event.sign_matches_mode and event_in_overtemp(event)


def repair_qualifies(event: AnomalyEvent) -> bool:
    """HA Repairs notification trigger: any unmodelled-input direction.

    Catches both additive-in-mode (sign_matches + overtemp) AND
    additive-against-mode (sign_inverse + undertemp). User should be
    notified of unmodelled inputs in either direction — open window in
    winter (sign_inverse + undertemp in heat mode) is just as valuable
    to surface as cooking heat (sign_matches + overtemp in heat mode).

    Latch arming uses only the sign_matches branch (safety-critical
    overtemp). Repairs surfaces both for user awareness.
    """
    if event.sign_matches_mode and event_in_overtemp(event):
        return True
    # Sign_inverse + undertemp/overtemp-mirror (depends on mode)
    if event.mode == "heat":
        # heat-loss case: room is cool (current < desired) AND residual indicates HP overcompensating
        return event.sign_inverse_mode and event.current_c < event.desired_c
    # cool mode: room is warm (current > desired) AND residual indicates AC undercooling
    return event.sign_inverse_mode and event.current_c > event.desired_c


# ── Self-starting Hawkins-Olwell CUSUM (replaces MAD path) ───────────
#
# The legacy MAD-based path (compute_mad_sigma + MIN_SIGMA_FLOOR) is
# hyper-sensitive in quiet residual regimes: when MAD collapses below
# the floor, σ̂ stops tracking the data and z-scores explode. Per the
# #135 audit, 44% of historical CUSUM events fired at the σ̂ floor.
# Two-filter (sign_matches_mode + overtemp) masks the actionable harm
# but the underlying alarm spam remains.
#
# Hawkins-Olwell 1998 self-starting CUSUM:
#   1. Maintain Welford running stats on a rolling window (12h here)
#   2. Standardize new obs against PRE-update mean/σ̂:
#      T = (x - mean_n) / sigma_n  (Student-t distributed under H0)
#   3. Transform to Normal via scaled t-CDF then Φ⁻¹:
#      U = Φ⁻¹(F_{n-1}(sqrt(n/(n+1)) · T))   ~ N(0,1) under H0
#   4. CUSUM with K=0.5, H=4 (Hawkins canonical, ARL₀ ≈ 370)
#   5. Suppress alarms during warmup (n < 20) — transform unstable
#
# References:
# - Hawkins (1987) "Self-starting CUSUM charts for location and scale"
# - Hawkins & Olwell (1998) "CUMULATIVE SUM CHARTS AND CHARTING FOR
#   QUALITY IMPROVEMENT" §7.2
# - arxiv:2509.07112 — rolling window for locally-stationary data

CUSUM_K_HAWKINS: float = 0.5
"""Reference value: half-sigma dead zone (Hawkins-Olwell canonical)."""

CUSUM_H_HAWKINS: float = 4.0
"""Threshold (multiples of σ̂): ARL₀ ≈ 370 (Hawkins-Olwell canonical)."""

CUSUM_WARMUP_N: int = 20
"""Min observations before alarms fire (transform unstable below this)."""

# Rolling window for σ̂ estimation. Same 12h as the legacy MAD path so
# the design rationale documented at CUSUM_RESIDUAL_WINDOW_S applies
# (diurnal-cycle capture, Jensen-Jones-Farmer 2006 minimum, etc.).
CUSUM_WINDOW_S: float = 12 * 3600.0


@dataclass
class SelfStartingCusumState:
    """Hawkins-Olwell self-starting CUSUM state.

    Replaces the MAD+floor σ̂ path. Welford running stats on a 12h
    rolling window estimate mean and σ̂; the t→Normal transform makes
    the standardized statistic exactly N(0,1) under H0 regardless of
    σ̂'s sampling-error noise.

    Persistence: `as_dict`/`from_dict` serialize alongside other PI
    state so the chart doesn't have to re-warm-up after HA restart.

    Window observations stored as ``(timestamp_mono, residual)`` pairs;
    eviction is age-based at ``CUSUM_WINDOW_S``.
    """

    # CUSUM accumulators (persisted across ticks)
    s_pos: float = 0.0
    s_neg: float = 0.0
    # Rolling window of (timestamp_mono, residual) — basis for Welford
    # mean/variance computation each update. Deque so age-based eviction
    # is O(1) at the front.
    window: deque[tuple[float, float]] = field(default_factory=deque)

    @property
    def n(self) -> int:
        """Current window size (= n in the Hawkins-Olwell formulas)."""
        return len(self.window)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "s_pos": self.s_pos,
            "s_neg": self.s_neg,
            "window": [(float(t), float(r)) for t, r in self.window],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelfStartingCusumState":
        """Restore from persisted state."""
        state = cls(
            s_pos=float(d.get("s_pos", 0.0)),
            s_neg=float(d.get("s_neg", 0.0)),
        )
        for t, r in d.get("window", []):
            state.window.append((float(t), float(r)))
        return state

    def reset_accumulators(self) -> None:
        """Reset S⁺/S⁻ to 0 (after an alarm fires; Lucas-Crosier 1982).

        Window is preserved — σ̂ continues to track residual distribution
        across the alarm event.
        """
        self.s_pos = 0.0
        self.s_neg = 0.0


def update_self_starting_cusum(
    state: SelfStartingCusumState,
    residual: float,
    now_mono: float,
    k: float = CUSUM_K_HAWKINS,
    h: float = CUSUM_H_HAWKINS,
    warmup_n: int = CUSUM_WARMUP_N,
    window_s: float = CUSUM_WINDOW_S,
) -> tuple[bool, float, float]:
    """Update self-starting CUSUM with one residual.

    Steps (Hawkins-Olwell 1998 §7.2):
      1. Evict window observations older than ``window_s`` seconds
      2. Compute Welford mean and σ̂ on PRE-update window
      3. Standardize: T = (residual - mean) / σ̂
      4. Scale: T_scaled = sqrt(n/(n+1)) · T  (~ Student-t with n-1 dof)
      5. Transform: U = Φ⁻¹(F_{n-1}(T_scaled))  ~ N(0,1) under H0
      6. CUSUM update with U
      7. Append (now_mono, residual) to window for next call
      8. Alarm if max(S⁺, S⁻) > h AND n ≥ warmup_n

    Returns:
        (alarmed, U, sigma_hat) where U is the standardized statistic
        (for diagnostics/logging) and sigma_hat is the pre-update σ̂
        (NaN if n < 2).

    Side effects:
        Mutates ``state``. Caller is responsible for calling
        ``state.reset_accumulators()`` after consuming an alarm.
    """
    # 1. Evict old observations from window
    cutoff = now_mono - window_s
    while state.window and state.window[0][0] < cutoff:
        state.window.popleft()

    n_pre = len(state.window)

    # 2-3. Insufficient history → just append and return
    if n_pre < 2:
        state.window.append((now_mono, residual))
        return False, 0.0, float("nan")

    # Welford-style mean + variance from window (one pass; n ≤ ~240 at 3min
    # cadence so O(n) per tick is sub-microsecond)
    residuals = [r for _, r in state.window]
    mean_pre = sum(residuals) / n_pre
    var_pre = sum((r - mean_pre) ** 2 for r in residuals) / (n_pre - 1)
    sigma_pre = math.sqrt(var_pre) if var_pre > 0 else 0.0

    if sigma_pre <= 0:
        # All residuals identical → can't standardize. Just append.
        state.window.append((now_mono, residual))
        return False, 0.0, sigma_pre

    # 4. Standardize
    T = (residual - mean_pre) / sigma_pre
    T_scaled = math.sqrt(n_pre / (n_pre + 1)) * T

    # 5. Transform Student-t → Normal via scipy.special (lightweight)
    if _SCIPY_AVAILABLE:
        cdf = float(_stdtr(n_pre - 1, T_scaled))
        # Clip to avoid ndtri(0)=-∞ or ndtri(1)=+∞ from extreme T_scaled
        cdf = max(min(cdf, 1.0 - 1e-12), 1e-12)
        U = float(_ndtri(cdf))
    else:  # pragma: no cover — scipy is required in this codebase
        # Fallback: treat scaled T as Gaussian. Reasonable for n > 30.
        U = T_scaled

    # 6. CUSUM update
    state.s_pos = max(0.0, state.s_pos + U - k)
    state.s_neg = max(0.0, state.s_neg - U - k)

    # 7. Append to window for next call
    state.window.append((now_mono, residual))

    # 8. Alarm gating: threshold AND warmup
    alarmed = (
        (state.s_pos > h or state.s_neg > h)
        and len(state.window) >= warmup_n
    )
    return alarmed, U, sigma_pre


class LatchArmingTrigger(StrEnum):
    """Why the over-temp regime latch armed on a given tick.

    Generic discriminator across all arming sources for
    `_uncontrollable_entry_latch`. Each new trigger gets a stable string
    value here; the event payload's `details` dict carries trigger-specific
    extras (e.g. CUSUM_ANOMALY records residual + peak_cusum).

    Currently only CUSUM_ANOMALY emits LatchArmedEvent. Backfilling the
    other three is tracked in future_work #136 — they don't emit today
    because their edge-detection semantics differ (continuous vs event vs
    threshold-crossing) and each needs its own emit-site review.
    """

    HP_ESTIMATED_IDLE = "hp_estimated_idle"     # cal_midpoint inferred HP idle
    MODE_FLIP_OVERTEMP = "mode_flip_overtemp"   # user flipped to HEAT in hot room
    SUSTAINED_OVERTEMP = "sustained_overtemp"   # error > 1.5°C for 30+ min
    CUSUM_ANOMALY = "cusum_anomaly"             # sign-matched CUSUM + overtemp gate


@dataclass(frozen=True, slots=True)
class LatchArmedEvent:
    """A latch-arming event with attribution to a specific trigger.

    Recorded post-hoc for bench analysis and any future Repairs check
    that wants to know "which trigger fired N times this week."
    """

    time: datetime
    mono: float
    trigger: LatchArmingTrigger
    mode: str                       # "heat" | "cool"
    overtemp_error: float           # at arming time, against user_desired
    details: dict[str, Any] = field(default_factory=dict)


def compute_mad_sigma(residuals: deque[float] | list[float]) -> float:
    """Compute robust scale estimate using Median Absolute Deviation.

    σ̂ = 1.4826 × median(|rᵢ - median(r)|)

    The 1.4826 factor makes MAD consistent with σ for Gaussian data,
    while remaining robust to up to 50% contamination (Huber, 1981).

    Returns MIN_SIGMA_FLOOR if MAD is zero (all residuals identical).
    """
    n = len(residuals)
    if n == 0:
        return MIN_SIGMA_FLOOR

    sorted_r = sorted(residuals)
    if n % 2 == 1:
        median_r = sorted_r[n // 2]
    else:
        median_r = (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2.0

    abs_devs = sorted(abs(r - median_r) for r in residuals)
    if n % 2 == 1:
        mad = abs_devs[n // 2]
    else:
        mad = (abs_devs[n // 2 - 1] + abs_devs[n // 2]) / 2.0

    sigma = 1.4826 * mad
    return max(sigma, MIN_SIGMA_FLOOR)


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


def obs_raw_reading(o: Any, entity_id: str) -> float:
    """Get a raw sensor reading from a v2 Observation by entity_id.

    Returns 0.0 if the observation doesn't have a reading for this entity.
    """
    if hasattr(o, "raw_readings") and isinstance(o.raw_readings, dict):
        return float(o.raw_readings.get(entity_id, 0.0))
    return 0.0


def check_feature_diversity(
    observations: list[Any],
    n_features: int,
    feature_names: list[str],
    min_activity_pct: float,
    min_observations: int,
    model_inputs: list[dict[str, Any]] | None = None,
    model_input_start: int = 2,
) -> tuple[str, str, str] | None:
    """Check observation buffer feature diversity.

    Args:
        model_inputs: model input config dicts, used to resolve entity_ids
            for raw_readings lookup.  Required for v2 observations.
        model_input_start: first feature index that is a model input
            (skips intercept, outdoor_delta, and any automatic features).
    """
    total_obs = len(observations)
    if total_obs < min_observations:
        return None
    m_inputs = model_inputs or []
    starved: list[str] = []
    for j in range(model_input_start, n_features):
        input_idx = j - model_input_start
        name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
        entity_id = m_inputs[input_idx].get("entity_id", "") if input_idx < len(m_inputs) else ""
        active = sum(
            1 for o in observations
            if abs(obs_raw_reading(o, entity_id)) > 1e-6
        ) if entity_id else 0
        if active / total_obs < min_activity_pct:
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
    coefficient_summary: str = "",
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
            {
                "outdoor_delta": f"{outdoor_delta_heat:.4f}",
                "coefficient_summary": coefficient_summary,
            },
            True,
        )
    return None


def build_coefficient_summary(
    coeff_names: list[str],
    coefficients: dict[int, float],
    seeds: list[float],
    std_errors: list[float],
    uncertainty_ratio_threshold: float = 1.0,
) -> str:
    """Build human-readable coefficient summary with uncertainty flags.

    Flags coefficients where the batch WLS standard error rivals or exceeds
    the coefficient magnitude (coefficient of variation > threshold) — the
    estimate is not well-determined and saving it as a seed would anchor the
    system to a noisy value.

    Args:
        coeff_names: Feature names [intercept, outdoor_delta, ...].
        coefficients: Physical-unit coefficients {index: value}.
        seeds: Current seed values (physical units).
        std_errors: Per-coefficient standard error in physical units, from the
            most recent batch WLS for this mode.  A non-finite entry marks a
            non-estimable coefficient (always flagged uncertain).  Empty/short
            when no batch has run yet for this mode — those coefficients are
            left unflagged (can't assess without an estimate).
        uncertainty_ratio_threshold: Flag when std_err/|coeff| > this.

    Returns:
        Multi-line string like:
          outdoor_delta: 0.35 → 0.42
          Solar Proxy: -4.00 → -3.12 (uncertain)
    """
    lines = []
    for i in range(1, len(coeff_names)):  # skip intercept
        name = coeff_names[i] if i < len(coeff_names) else f"coeff_{i}"
        value = coefficients.get(i, 0.0)
        seed = seeds[i] if i < len(seeds) else 0.0

        se = std_errors[i] if i < len(std_errors) else None
        # Flag when the standard error rivals the coefficient magnitude (high
        # coefficient of variation) or the coefficient is non-estimable
        # (non-finite std_err).  No std_err yet → can't assess → don't flag.
        uncertain = (
            se is not None
            and abs(value) > 1e-6
            and (not math.isfinite(se) or se / abs(value) > uncertainty_ratio_threshold)
        )

        line = f"{name}: {seed:.3f} → {value:.3f}"
        if uncertain:
            line += " (uncertain)"
        lines.append(line)
    return "; ".join(lines)


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


def check_model_drift_repair(
    drifting_coefficients: list[tuple[int, str, int]],
    has_had_stable_batch: bool,
    min_consecutive: int = 5,
) -> list[tuple[str, dict[str, str], bool]]:
    """Check for persistent model drift with maturity gate.

    Suppresses drift alerts until at least one batch cycle has had
    recommend_update=False (system stabilized at least once).
    """
    results: list[tuple[str, dict[str, str], bool]] = []
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
    coefficients: list[tuple[str, float, tuple[float, float] | None]],
    intercept_threshold: float = 1.0,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if intercept has grown large by absorbing a clamped coefficient's effect.

    coefficients: list of (name, value, clamp) for non-intercept coefficients.
    Batch WLS clips to clamps post-solve, so a coefficient sitting at its
    boundary means the data wanted to push past the bound and the residual
    is being absorbed by the intercept.
    """
    if abs(intercept_value) < intercept_threshold:
        return ("intercept_absorbing", {}, False)

    for name, value, clamp in coefficients:
        if clamp is None:
            continue
        lo, hi = clamp
        if abs(value - lo) < 1e-3 or abs(value - hi) < 1e-3:
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
    collinear_groups: list[Any] | None = None,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if features are multicollinear (condition number too high).

    When two input features are highly correlated, the RLS cannot
    separate their effects and coefficient estimates become unstable.

    Args:
        condition_number: spectral condition number √(λ_max/λ_min) of X^T X + λI.
            Excludes intercept (Belsley 1980 §3.3).
        correlated_pairs: (name_i, name_j, r) for pairs with |r| > 0.7.
        sustained_cycles: how many consecutive batch cycles above threshold.
        create_threshold: condition number above which to create issue.
            Belsley (1980): κ > 30 = moderate multicollinearity.
        clear_threshold: condition number below which to clear issue.
            κ < 20 = weak dependencies, coefficients reliable.
        min_sustained_cycles: minimum sustained cycles before creating.
        collinear_groups: Belsley VDP groups from compute_belsley_diagnostics().
            When available, provides per-variable diagnosis of which features
            share ill-conditioned components — more precise than pairwise r.

    Reference: Belsley, Kuh & Welsch, "Regression Diagnostics" (1980), Ch. 3.
    """
    if condition_number < clear_threshold:
        return ("multicollinearity", {}, False)

    if condition_number < create_threshold or sustained_cycles < min_sustained_cycles:
        return None  # hysteresis band

    # Build human-readable description of the collinearity.
    # Prefer Belsley VDP groups (identifies multivariate dependencies)
    # over pairwise correlations (only captures bivariate).
    if collinear_groups:
        group_strs = []
        for g in collinear_groups[:3]:
            names = " and ".join(g.features)
            group_strs.append(f"{names} (CI={g.condition_index:.0f})")
        pairs_text = "; ".join(group_strs)
    elif correlated_pairs:
        above_threshold = [p for p in correlated_pairs if abs(p[2]) > 0.7]
        pair_strs = [f"{a} and {b} (r={r:.2f})" for a, b, r in correlated_pairs[:3]]
        if above_threshold:
            pairs_text = "; ".join(pair_strs)
        else:
            pairs_text = (
                f"distributed across inputs (highest: {pair_strs[0]})"
            )
    else:
        pairs_text = "distributed across inputs (not enough data to identify pairs)"

    return (
        "multicollinearity",
        {
            "condition_number": f"{condition_number:.0f}",
            "pairs": pairs_text,
        },
        True,
    )


def check_freeze_impact_repair(
    coeff_name: str,
    mode: str,
    rms_at_freeze: float,
    current_rms: float,
    sustained_cycles: int,
    rms_increase_pct: float = 20.0,
    clear_pct: float = 5.0,
    min_sustained_cycles: int = 3,
) -> tuple[str, dict[str, str], bool] | None:
    """Check if a frozen coefficient is degrading model fit.

    Compares current batch residual RMS to the RMS recorded when the
    coefficient was frozen.  A sustained increase suggests the freeze
    is preventing the model from tracking a real change.

    Args:
        coeff_name: human-readable coefficient name.
        mode: "heat" or "cool".
        rms_at_freeze: batch residual RMS when the coefficient was frozen.
        current_rms: latest batch residual RMS.
        sustained_cycles: consecutive batch cycles with RMS above threshold.
        rms_increase_pct: % increase in RMS to trigger issue (default 20%).
        clear_pct: % increase below which to clear the issue.
        min_sustained_cycles: minimum sustained cycles before creating.
    """
    if rms_at_freeze <= 0:
        return None

    increase_pct = ((current_rms - rms_at_freeze) / rms_at_freeze) * 100.0

    if increase_pct < clear_pct:
        return ("freeze_impact", {}, False)

    if increase_pct < rms_increase_pct or sustained_cycles < min_sustained_cycles:
        return None  # hysteresis band

    return (
        "freeze_impact",
        {
            "coeff_name": coeff_name,
            "mode": mode,
            "rms_at_freeze": f"{rms_at_freeze:.3f}",
            "current_rms": f"{current_rms:.3f}",
            "increase_pct": f"{increase_pct:.0f}",
        },
        True,
    )
