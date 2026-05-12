"""Offline batch learning for RLS feedforward model.

Standalone module — no Home Assistant dependencies. Periodically analyzes
accumulated near-equilibrium observations via weighted least squares (WLS)
and reports how the batch estimate compares to the current online RLS model.

Observation-only mode: logs recommendations but does not modify the model.
Set apply_updates=True to enable automatic model updates.

References:
- Ljung, L. "System Identification: Theory for the User" — batch estimation
- Åström & Wittenmark, "Adaptive Control" — offline vs online identification
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    from scipy.optimize import minimize_scalar as _minimize_scalar
    _SCIPY_AVAILABLE = True
except ImportError:
    _minimize_scalar = None
    _SCIPY_AVAILABLE = False

from .buffer_policies import BufferPolicy, LeveragePolicy, SlevPolicy
from .model_input_manager import tod_features

_LOGGER = logging.getLogger(__name__)


@dataclass
class Observation:
    """A single physical observation for batch learning.

    Stores **what the house experienced** — raw sensor readings keyed by
    entity_id — rather than pre-computed regression features.  Feature
    vectors are built at WLS time from raw_readings + current config,
    decoupling storage from model input configuration (lag_tau, entity
    swaps, feature additions/removals).

    This means:
    - Changing lag_tau does NOT invalidate the buffer
    - Adding/removing model inputs does NOT corrupt old observations
    - Old observations contribute to features they have data for and are
      excluded from features they predate (hierarchical regression)
    """

    timestamp: float  # monotonic time (for ordering/age)
    wall_time: float  # UTC epoch seconds (for sun position, time-of-day)
    hp_setpoint: float | None  # HP setpoint (°C), None for passive observations
    current_c: float  # filtered room temperature (°C)
    desired_c: float  # target temperature (°C)
    outdoor_temp_c: float | None  # absolute outdoor temperature (°C)
    room_rate: float  # dT/dt in °C/min at observation time
    raw_readings: dict[str, float]  # entity_id → raw sensor value at obs time
    clamped: bool  # True if HP output is unusable for learning
    clamped_reason: str = ""  # "", "no_output", "saturated_low", "saturated_high", "observe_only"
    supplemental_active: bool = False  # supplemental source tracking or assisting
    hp_contribution_uncertain: bool = False  # |hp_offset| within regime margin
    # Auto-perturbation state at observation time. True when the auto-
    # perturbation state machine has a non-zero setpoint offset applied
    # (STEP_ACTIVE or RESTORE phases). Used by Stage B greybox fit to
    # partition perturbation-regime observations from operational ones.
    during_perturbation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "v": 2,  # schema version
            "t": self.timestamp,
            "wt": self.wall_time,
            "sp": self.hp_setpoint,
            "cur": self.current_c,
            "des": self.desired_c,
            "ot": self.outdoor_temp_c,
            "rate": self.room_rate,
            "rr": self.raw_readings,
            "clamp": self.clamped,
            "cr": self.clamped_reason,
            "sa": self.supplemental_active,
            "hcu": self.hp_contribution_uncertain,
            "dp": self.during_perturbation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observation:
        """Restore a v2 Observation from a serialized dict.

        Raises ValueError for v1 observations (pre-raw_readings format)
        or corrupt data, so callers can skip gracefully.
        """
        if d.get("v", 1) < 2:
            raise ValueError("v1 observation cannot be restored")
        return cls(
            timestamp=d["t"],
            wall_time=d["wt"],
            hp_setpoint=d["sp"],
            current_c=d["cur"],
            desired_c=d["des"],
            outdoor_temp_c=d.get("ot"),
            room_rate=d["rate"],
            raw_readings=d.get("rr", {}),
            clamped=d["clamp"],
            clamped_reason=d.get("cr", ""),
            supplemental_active=d.get("sa", False),
            hp_contribution_uncertain=d.get("hcu", False),
            during_perturbation=d.get("dp", False),
        )


# Default capacity for diversity-aware buffer.  ~2000 observations at
# 4/hr fills in ~3 weeks, after which leverage scoring governs eviction.
DEFAULT_DIVERSITY_BUFFER_SIZE = 2000

# Regularization for (X^T X)^{-1} to keep it invertible before the
# buffer fills and during early operation when rank may be deficient.
INFO_MATRIX_REGULARIZATION = 1e-4

# Wall-clock correlation time for thermal residuals (minutes).  Used to
# anchor the HAC bandwidth in physical time rather than sample count, so
# the bandwidth covers the same wall-clock window regardless of sampling
# cadence.  Building thermal residuals are correlated over roughly
# (envelope τ ÷ EMA-tau pre-filter pass) ≈ 30 min for typical residential
# buildings (Bacher-Madsen 2011 thermal time constants).  Default 60 min
# rather than 30: the Bartlett kernel undershoots highly-autocorrelated
# variance amplification (effective bandwidth ≈ L/2), so we double the
# nominal correlation time to recover most of the correction.  A
# Quadratic-Spectral kernel would let us hold T_corr=30 with a closer-to-
# unity effective coverage but adds compute (Andrews 1991).
# See ~/.claude/plans/wls_cadence_corrections.md (open questions 1, 2) —
# both T_corr default and kernel choice should be validated against
# actual residual ACF on B9/B8 reproducers and tuned if the empirical
# decay differs.
HAC_T_CORR_MIN = 60.0


# ── Cadence-invariant statistics: HAC std_err + ESS ─────────────────────
#
# Classical OLS/WLS std_err shrinks like 1/√N regardless of whether new
# samples carry independent information.  At fine sampling cadences,
# thermal residuals are autocorrelated — N grows but ESS doesn't, so
# IID std_err underestimates true sampling variance, K = σ²_prior/(σ²_prior
# + σ²_batch) becomes too aggressive, and a single batch can dominate
# the persistent state.  The Newey-West (1987) HAC sandwich corrects the
# variance estimator using a Bartlett-kernel-weighted residual ACF.
# Bandwidth is wall-clock-anchored (HAC_T_CORR_MIN ÷ Δt) rather than
# Newey-West's N-only formula so the correction is cadence-invariant by
# construction.
#
# Both HAC std_err and ESS-substituted BIC share the residual ACF as
# their primitive; the helpers below compute it once and reuse.


def _estimate_observation_dt_min(
    observations: list[Observation],
) -> float:
    """Median consecutive wall-clock dt in minutes across observations.

    Observations may have gaps (filtered passive/transient ones removed
    from the buffer); the median of consecutive deltas is the cleanest
    cadence proxy.  Returns 1.0 if unavailable so HAC bandwidth degenerates
    to a safe minimum rather than blowing up.  None entries (test seams
    that pass placeholder lists) are tolerated.
    """
    if len(observations) < 2:
        return 1.0
    times = sorted(
        o.wall_time for o in observations
        if o is not None and o.wall_time > 0
    )
    if len(times) < 2:
        return 1.0
    deltas = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return 1.0
    deltas.sort()
    median_seconds = deltas[len(deltas) // 2]
    return median_seconds / 60.0


def _hac_bandwidth(
    n_obs: int, dt_min: float, t_corr_min: float = HAC_T_CORR_MIN,
) -> int:
    """Wall-clock-anchored HAC bandwidth.

    L = ceil(t_corr_min / dt_min), capped at n_obs/5 to guarantee small-
    sample stability (Andrews 1991 plug-in could replace this for fully
    data-driven bandwidth selection — see plan open question 2).
    """
    if dt_min <= 0:
        return 1
    raw = math.ceil(t_corr_min / dt_min)
    cap = max(1, n_obs // 5)
    return max(1, min(raw, cap))


def _bartlett_acf(
    residuals: list[float],
    weights: list[float],
    max_lag: int,
) -> list[float]:
    """Bartlett-weighted residual autocovariances γ̂(0)..γ̂(max_lag).

    Bartlett kernel weights `(1 - k/(max_lag+1))` ensure the implied
    spectral density / variance estimator is positive semi-definite
    (Newey-West 1987).  Weights are applied multiplicatively to each
    residual before computing the lagged products, treating WLS weights
    as importance weights for the variance computation.
    """
    n = len(residuals)
    if n == 0:
        return [0.0] * (max_lag + 1)
    we = [weights[i] * residuals[i] for i in range(n)]
    acf = [0.0] * (max_lag + 1)
    for k in range(max_lag + 1):
        s = 0.0
        for t in range(k, n):
            s += we[t] * we[t - k]
        acf[k] = s / n
    return acf


def _ess_from_acf(acf: list[float], n_obs: int) -> float:
    """Effective sample size from Bartlett-weighted residual ACF.

    ESS = N / (1 + 2·Σ_{k=1..L} (1 - k/(L+1)) · ρ̂(k))
    where ρ̂(k) = γ̂(k)/γ̂(0).  Bartlett weighting on the sum (not just
    the ACF estimator) keeps ESS positive and bounded in [1, N].

    Returns N when γ̂(0) ≤ 0 (degenerate residuals) — degrade to classical.
    """
    if not acf or acf[0] <= 0 or n_obs <= 0:
        return float(max(1, n_obs))
    L = len(acf) - 1
    if L <= 0:
        return float(n_obs)
    rho_sum = 0.0
    for k in range(1, L + 1):
        bartlett = 1.0 - k / (L + 1)
        rho_sum += bartlett * (acf[k] / acf[0])
    denom = 1.0 + 2.0 * rho_sum
    if denom <= 0:
        # Pathological ACF (negative correlations dominate) — clamp to N.
        return float(n_obs)
    ess = n_obs / denom
    return max(1.0, min(float(n_obs), ess))


def _hac_meat_matrix(
    X_normalized: list[list[float]],
    weights: list[float],
    residuals: list[float],
    bandwidth: int,
) -> list[list[float]]:
    """Newey-West sandwich meat: X' W Σ̂ W X with Bartlett kernel.

    Σ̂[t,s] = Σ_{k=-L..L} (1 - |k|/(L+1)) · γ̂(k) · 1{s = t-k}

    Equivalent (and cheaper) form:
        meat = Σ_t (w_t · X_t · e_t)(w_t · X_t · e_t)'
             + Σ_{k=1..L} (1 - k/(L+1)) · [Σ_t (w_t·X_t·e_t)(w_{t-k}·X_{t-k}·e_{t-k})' + transpose]
    """
    n = len(residuals)
    if n == 0:
        return [[]]
    p = len(X_normalized[0]) if X_normalized and X_normalized[0] else 0
    if p == 0:
        return [[]]
    # Score vectors s_t = w_t · e_t · X_t
    s_vecs = [
        [weights[t] * residuals[t] * X_normalized[t][j] for j in range(p)]
        for t in range(n)
    ]
    meat = [[0.0] * p for _ in range(p)]
    # Lag 0
    for t in range(n):
        for i in range(p):
            for j in range(p):
                meat[i][j] += s_vecs[t][i] * s_vecs[t][j]
    # Lags 1..L
    for k in range(1, bandwidth + 1):
        bartlett = 1.0 - k / (bandwidth + 1)
        for t in range(k, n):
            for i in range(p):
                for j in range(p):
                    cross = s_vecs[t][i] * s_vecs[t - k][j]
                    meat[i][j] += bartlett * cross
                    meat[j][i] += bartlett * cross
    return meat


def _hac_std_err_from_sandwich(
    XtWX_inv: list[list[float]],
    meat: list[list[float]],
    col_scales: list[float],
    classical_std_err: list[float],
) -> list[float]:
    """Assemble HAC sandwich variance and return per-coefficient std_err.

    Var(β̂) = (X'WX)⁻¹ · meat · (X'WX)⁻¹
    De-normalize by col_scales (matches existing convention in WLS solvers).

    Safeguard: if HAC std_err < classical std_err for any coefficient,
    fall back to classical for that coefficient.  HAC under
    autocorrelation should always be ≥ classical; smaller indicates
    numerical issue (e.g. negative meat diagonal from pathological ACF).
    """
    p = len(XtWX_inv)
    if p == 0:
        return []
    # M_inv · meat · M_inv (manual matrix multiply — keep pure-Python
    # path for parity with rest of module).
    inv_meat = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            s = 0.0
            for k in range(p):
                s += XtWX_inv[i][k] * meat[k][j]
            inv_meat[i][j] = s
    var_hac = [0.0] * p
    for i in range(p):
        s = 0.0
        for k in range(p):
            s += inv_meat[i][k] * XtWX_inv[k][i]
        var_hac[i] = s

    out: list[float] = []
    for i in range(p):
        scale_sq = col_scales[i] * col_scales[i] if i < len(col_scales) else 1.0
        v = max(0.0, var_hac[i]) / scale_sq if scale_sq > 0 else 0.0
        hac_se = math.sqrt(v) if v > 0 else 0.0
        cse = classical_std_err[i] if i < len(classical_std_err) else float("inf")
        if math.isfinite(cse) and hac_se < cse:
            # Numerical safeguard: HAC should always be ≥ classical for
            # autocorrelated residuals.  Smaller means meat had a
            # negative diagonal contribution — fall back.
            _LOGGER.debug(
                "HAC std_err[%d]=%.4f < classical=%.4f; falling back to classical",
                i, hac_se, cse,
            )
            out.append(cse)
        else:
            out.append(hac_se if math.isfinite(hac_se) else cse)
    return out


def _hac_std_err_univariate(
    r_z: list[float],
    weights: list[float],
    residuals: list[float],
    wrzrz: float,
    bandwidth: int,
    classical_std_err: float,
) -> float:
    """HAC std_err for a univariate FWL partial regression.

    For β = (r_z' W y) / (r_z' W r_z), the sandwich is:
        Var(β̂) = (1/wrzrz²) · Σ_{|k|≤L} κ(k) · Σ_t (w_t·r_z[t]·e_t)(w_{t-k}·r_z[t-k]·e_{t-k})

    Same Bartlett kernel and safeguard as the full-matrix variant.
    """
    n = len(residuals)
    if n == 0 or wrzrz <= 0:
        return classical_std_err
    s = [weights[t] * r_z[t] * residuals[t] for t in range(n)]
    meat = sum(x * x for x in s)
    for k in range(1, bandwidth + 1):
        bartlett = 1.0 - k / (bandwidth + 1)
        cross = sum(s[t] * s[t - k] for t in range(k, n))
        meat += 2.0 * bartlett * cross
    if meat < 0:
        return classical_std_err
    var_hac = meat / (wrzrz * wrzrz)
    hac_se = math.sqrt(var_hac) if var_hac > 0 else 0.0
    if math.isfinite(classical_std_err) and hac_se < classical_std_err:
        _LOGGER.debug(
            "HAC univariate std_err=%.4f < classical=%.4f; falling back",
            hac_se, classical_std_err,
        )
        return classical_std_err
    return hac_se if math.isfinite(hac_se) else classical_std_err


@dataclass(frozen=True, slots=True)
class BufferAddResult:
    """Outcome of a single observation admission attempt.

    Returned by `DiversityAwareBuffer.add()` (and subclass overrides) so
    callers can populate observability snapshots without re-deriving
    buffer-internal state.

    Fields:
      admitted: True iff the observation is now in the buffer.
      candidate_score: policy score the candidate received during the
          decision (e.g. leverage under LeveragePolicy). None when the
          candidate was rejected before scoring (e.g. greybox
          `no_outdoor_temp`).
      evicted_timestamp: monotonic timestamp of the displaced incumbent
          when admission into a full buffer caused an eviction. None
          otherwise.
      min_incumbent_score: policy score of the worst-scoring incumbent
          at decision time, present only when the buffer was at capacity
          (so the candidate was actually compared). None when the buffer
          had room or admission was rejected before scoring.
      rejection_reason: short string identifying why admission was
          declined.  Policy-driven rejections use ``f"{policy_name}_rejected"``
          (e.g. "leverage_rejected"); upstream filters use other literals
          like "no_outdoor_temp".  None when `admitted` is True.
      policy_name: name of the policy that made the decision (e.g.
          "leverage"), for observability across mixed-policy runs.
    """

    admitted: bool
    candidate_score: float | None
    evicted_timestamp: float | None
    min_incumbent_score: float | None
    rejection_reason: str | None
    policy_name: str


def build_feature_vector_from_raw(
    obs: Observation,
    model_inputs: list[dict[str, Any]],
    feature_order: list[str],
    filtered_overrides: dict[str, float] | None = None,
) -> list[float] | None:
    """Build a feature vector from raw readings + current config.

    Returns an ordered list aligned to ``feature_order``, or None if the
    observation is missing readings for any required model input entity.

    Feature construction:
    - "intercept": always 1.0
    - "outdoor_delta": obs.outdoor_temp_c - obs.desired_c (requires outdoor_temp_c)
      References desired temp (exogenous), not room temp, to match the online
      model and prevent endogeneity in the regressor (Ljung, System Identification).
    - model inputs: raw_readings[entity_id], with delta_from_room adjustment
      if configured (entity_temp_c - current_c).  raw_readings stores °C
      absolute temps; the delta is computed here at batch time.
    - "sin_hour", "cos_hour": sinusoidal time-of-day features computed from
      obs.wall_time (local fractional hour).

    When ``filtered_overrides`` is provided, its values replace raw_readings
    for matching entity_ids.  Used by batch WLS to apply retrospective EMA
    filtering so the batch regression matches the online filtered signal.
    """
    if obs.outdoor_temp_c is None:
        return None

    features: dict[str, float] = {
        "intercept": 1.0,
        "outdoor_delta": obs.outdoor_temp_c - obs.desired_c,
    }

    for m_input in model_inputs:
        entity_id = m_input.get("entity_id", "")
        name = m_input.get("name", entity_id)
        if not entity_id or entity_id not in obs.raw_readings:
            return None  # incomplete — skip this observation for this feature set
        # Use filtered override if available, else raw reading
        if filtered_overrides and entity_id in filtered_overrides:
            value = filtered_overrides[entity_id]
        else:
            value = obs.raw_readings[entity_id]
        # Apply delta_from_room: raw_readings stores the °C absolute temp;
        # subtract the observation's room temp to get the delta.
        if m_input.get("delta_from_room"):
            value = value - obs.current_c
        features[name] = value

    # Time-of-day sinusoidal features from observation wall clock
    sin_h, cos_h = tod_features(obs.wall_time)
    features["sin_hour"] = sin_h
    features["cos_hour"] = cos_h

    return [features.get(name, 0.0) for name in feature_order]


# ── Retrospective EMA + auto lag-tau detection ─────────────────────────
#
# The online path applies EMA filtering tick-by-tick (model_input_manager),
# but batch WLS regresses against raw instantaneous values.  For inputs
# with thermal lag (solar through walls), this attenuates the estimated
# coefficient.  These functions apply EMA retrospectively across the
# observation buffer and auto-detect the optimal lag per input.
#
# Reference: Ljung, "System Identification" — pre-filtering inputs to
# match plant dynamics before identification.

# Golden-section ratio for pure-Python bracket search.
_PHI = (math.sqrt(5) - 1) / 2  # ≈ 0.618

# Bounds for tau search in seconds: 0 (no lag) to 8 hours (default).
_TAU_SEARCH_MIN = 0.0
_TAU_SEARCH_MAX = 28800.0

# Per-role upper rails for the tau search (seconds).  Physical reasoning:
# an EMA with τ filters out signal frequencies above 1/τ, so a search rail
# matched to the input's plausible thermal timescale prevents the optimizer
# from running off into the diurnal band where solar lives (τ ≥ 12h ≈ half
# the diurnal period collapses EMA(solar) to a near-DC signal — see the
# Forssell-Ljung 1999 / Raue 2009 practical-non-identifiability argument).
#
# Inputs with no role match fall back to ``_TAU_SEARCH_MAX``.
_TAU_SEARCH_MAX_BY_ROLE: dict[str, float] = {
    "solar": 21600.0,         # 6h — well below 12h diurnal half-period
    "heat_source": 3600.0,    # 1h — convective response of radiators / stoves
    "adjacent_zone": 43200.0, # 12h — party-wall conduction timescale
}

# Optimum lands within this fraction of either rail → flag as
# practically non-identifiable (Raue 2009 profile-likelihood).
_TAU_BOUNDARY_FRACTION = 0.05

# Minimum detectable tau (seconds).  Lags shorter than observation
# spacing (~15 min) are indistinguishable from noise — snap to 0.
_TAU_MIN_MEANINGFUL = 600.0  # 10 minutes

# Acceptance is via BIC test (Schwarz, 1978): accept τ if
# n·log(RSS_0/RSS_τ) > log(n), i.e. ΔBIC < 0 for one extra parameter.
# References: Ljung, "System Identification" §16.4; Söderström & Stoica
# §11.4. The BIC threshold scales with n, so it tightens automatically
# on small buffers and loosens on large ones — unlike a fixed R² floor.


@dataclass(frozen=True)
class LagTauDiagnostic:
    """Per-input τ search diagnostics from one batch run.

    Surfaces the BIC/R² evidence behind the chosen τ so debug bundles can
    distinguish "decisive accept" from "marginal accept", and explain why
    a τ was rejected (BIC failed vs sub-floor snap).

    `tau` is the value that flows through to ``BatchResult.detected_tau``
    and the online EMA — for rejected searches this is 0.0 even though
    ``tau_opt_raw`` may be larger.
    """

    tau: float
    """τ applied (seconds). 0.0 for rejected searches and below-floor snaps."""

    tau_opt_raw: float
    """τ at the RSS minimum before BIC/floor gates (seconds)."""

    bic_gain: float
    """n·log(RSS_0/RSS_τ). Accepted when ``bic_gain > log(n_eff)``."""

    bic_threshold: float
    """log(n_eff). The acceptance bar bic_gain has to clear."""

    r2_improvement: float
    """1 - RSS_τ/RSS_0. Reported only — the gate is BIC, not R²."""

    beta_at_tau: float
    """Recovered input coefficient at ``tau_opt_raw`` from the joint fit."""

    n_eff: int
    """Observations with a valid raw reading for this entity."""

    accepted: bool
    """True iff BIC passed AND τ_opt ≥ floor. Mirrors ``tau > 0``."""

    reject_reason: str
    """"" if accepted, else one of {"bic_failed", "below_floor", "boundary_hit"}."""

    search_max_used: float = 0.0
    """Upper rail of the τ search (seconds).  Per-role when configured,
    falls back to the global ``_TAU_SEARCH_MAX``.  Surfaced so debug
    bundles can tell which rail a ``boundary_hit`` rejection bumped against."""

    boundary_hit: bool = False
    """True when ``tau_opt_raw`` lands within ``_TAU_BOUNDARY_FRACTION`` of
    the upper search rail.  By Raue 2009 profile-likelihood semantics this
    is practical non-identifiability: the optimizer has no preference for
    stopping inside the search range, so the value is not informative."""


def _apply_retrospective_ema(
    observations: list[Observation],
    entity_id: str,
    tau_seconds: float,
) -> list[float | None]:
    """Apply EMA filtering retrospectively across observations.

    Sorts observations by wall_time and applies the same EMA formula as
    the online path: alpha = 1 - exp(-dt/tau).  Returns a list parallel
    to the input list with filtered values, or None where the entity_id
    is missing from raw_readings.

    For tau <= 0, returns raw values (no filtering).
    """
    n = len(observations)
    if n == 0:
        return []

    # Build (original_index, wall_time, raw_value) sorted by wall_time
    indexed: list[tuple[int, float, float | None]] = []
    for i, obs in enumerate(observations):
        val = obs.raw_readings.get(entity_id)
        indexed.append((i, obs.wall_time, val))
    indexed.sort(key=lambda t: t[1])

    result: list[float | None] = [None] * n

    if tau_seconds <= 0:
        # No filtering — return raw values
        for orig_idx, _, raw in indexed:
            result[orig_idx] = raw
        return result

    ema_state: float | None = None
    prev_wt: float = 0.0

    for orig_idx, wt, raw in indexed:
        if raw is None:
            # Missing reading — propagate None, keep EMA state
            continue

        if ema_state is None:
            ema_state = raw
            prev_wt = wt
        else:
            dt = wt - prev_wt
            if dt > 0:
                alpha = 1.0 - math.exp(-dt / tau_seconds)
                ema_state = alpha * raw + (1.0 - alpha) * ema_state
            # dt == 0: simultaneous observations, keep previous state
            prev_wt = wt

        result[orig_idx] = ema_state

    return result


def _detect_optimal_tau(
    observations: list[Observation],
    y_values: list[float],
    weights: list[float],
    entity_id: str,
    delta_from_room: bool = False,
    base_X: list[list[float]] | None = None,
    tod_cols: list[tuple[float, float]] | None = None,
    input_role: str | None = None,
) -> LagTauDiagnostic | None:
    """Find the optimal EMA tau for a model input via joint regression sweep.

    Builds X = [1, od, sin, cos, EMA(input, τ)] at candidate τ values
    and picks the τ with lowest weighted RSS from the full joint regression.
    This avoids the suppression problem where diurnal correlation between
    solar and outdoor_delta hides the solar signal in partial residuals.

    Returns a ``LagTauDiagnostic`` carrying both the chosen τ and the
    BIC/R² evidence, or None when there's nothing to diagnose
    (insufficient observations, degenerate RSS). Rejected searches still
    return a diagnostic with ``accepted=False`` and ``tau=0`` so debug
    bundles can distinguish "BIC failed" from "below-floor snap".

    Uses scipy.optimize.minimize_scalar when available, else golden-section.
    """
    n = len(observations)
    if n < 10:
        return None

    # Effective regression sample size: observations with a valid raw
    # reading for this entity (eligibility is τ-invariant — the EMA
    # cannot create a value where the raw is missing).
    n_eff = sum(
        1 for o in observations if o.raw_readings.get(entity_id) is not None
    )
    if n_eff < 10:
        return None

    def _compute_rss(tau: float) -> float:
        """Weighted RSS from joint regression y ~ [1, od, sin, cos, filtered_input]."""
        filtered = _apply_retrospective_ema(observations, entity_id, tau)

        # Build rows for observations with valid filtered values
        rows: list[int] = []
        x_input: list[float] = []
        for i in range(n):
            val_f = filtered[i]
            if val_f is None:
                continue
            x_val = val_f
            if delta_from_room:
                x_val = x_val - observations[i].current_c
            rows.append(i)
            x_input.append(x_val)

        m = len(rows)
        if m < 10:
            return float("inf")

        if _NUMPY_AVAILABLE and base_X is not None and tod_cols is not None:
            # Build design matrix: [1, od, sin, cos, input]
            X = np.empty((m, 5))
            y_arr = np.empty(m)
            w_arr = np.empty(m)
            for j, i in enumerate(rows):
                X[j, 0] = base_X[i][0]  # intercept
                X[j, 1] = base_X[i][1]  # outdoor_delta
                X[j, 2] = tod_cols[i][0]  # sin
                X[j, 3] = tod_cols[i][1]  # cos
                X[j, 4] = x_input[j]
                y_arr[j] = y_values[i]
                w_arr[j] = weights[i]
            # Weighted least squares: scale rows by sqrt(w)
            sw = np.sqrt(w_arr)
            Xw = X * sw[:, None]
            yw = y_arr * sw
            beta, rss_arr, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            if len(rss_arr) > 0:
                return float(rss_arr[0])
            # Fallback: compute RSS manually
            pred = Xw @ beta
            return float(np.sum((yw - pred) ** 2))
        else:
            # Pure-Python 5-feature normal equations
            p = 5
            XtWX = [[0.0] * p for _ in range(p)]
            XtWy = [0.0] * p
            for j, i in enumerate(rows):
                od_i = base_X[i][1] if base_X else 0.0
                s_i, c_i = tod_cols[i] if tod_cols else (0.0, 0.0)
                row = [1.0, od_i, s_i, c_i, x_input[j]]
                wi = weights[i]
                for a in range(p):
                    XtWy[a] += row[a] * wi * y_values[i]
                    for b in range(p):
                        XtWX[a][b] += row[a] * wi * row[b]
            for a in range(p):
                XtWX[a][a] += 1e-6
            beta_pp = _solve_symmetric(XtWX, XtWy, p)
            if beta_pp is None:
                return float("inf")
            rss = 0.0
            for j, i in enumerate(rows):
                od_i = base_X[i][1] if base_X else 0.0
                s_i, c_i = tod_cols[i] if tod_cols else (0.0, 0.0)
                row = [1.0, od_i, s_i, c_i, x_input[j]]
                pred = sum(beta_pp[a] * row[a] for a in range(p))
                rss += weights[i] * (y_values[i] - pred) ** 2
            return rss

    # Per-role upper rail for the τ search.  Inputs whose role has a
    # tighter physical timescale (e.g. solar bounded by room thermal
    # mass) get a tighter rail than the global default — reduces the
    # chance of the optimum running off into a band where the EMA
    # smooths the signal it's supposed to identify.
    search_max = _TAU_SEARCH_MAX_BY_ROLE.get(
        input_role or "", _TAU_SEARCH_MAX,
    )

    # Compute RSS at tau=0 (raw) for comparison
    rss_raw = _compute_rss(0.0)

    if _SCIPY_AVAILABLE and _minimize_scalar is not None:
        result = _minimize_scalar(
            _compute_rss,
            bounds=(_TAU_SEARCH_MIN, search_max),
            method="bounded",
            options={"xatol": 60.0},  # 1-minute precision
        )
        tau_opt = float(result.x)
        rss_opt = float(result.fun)
    else:
        # Pure-Python golden-section search
        a, b = _TAU_SEARCH_MIN, search_max
        c = b - _PHI * (b - a)
        d = a + _PHI * (b - a)
        fc = _compute_rss(c)
        fd = _compute_rss(d)

        while (b - a) > 60.0:  # 1-minute precision
            if fc < fd:
                b = d
                d, fd = c, fc
                c = b - _PHI * (b - a)
                fc = _compute_rss(c)
            else:
                a = c
                c, fc = d, fd
                d = a + _PHI * (b - a)
                fd = _compute_rss(d)

        tau_opt = (a + b) / 2
        rss_opt = _compute_rss(tau_opt)

    # Degenerate RSS (perfect fit at τ=0, or empty regression) — no
    # diagnostic is meaningful; the BIC test would divide by zero.
    if rss_raw <= 0 or rss_raw == float("inf") or rss_opt <= 0:
        return None

    r2_improvement = 1.0 - rss_opt / rss_raw

    # Recover the input coefficient at tau_opt — always, even when
    # rejected, so debug bundles can compare β at the candidate τ
    # against β at τ=0 and reason about why the search was rejected.
    # Also compute residuals + weights at tau_opt for ESS computation
    # below (BIC test uses ESS in place of raw n_eff to honor the
    # autocorrelation-induced information collapse at fine cadence).
    filtered = _apply_retrospective_ema(observations, entity_id, tau_opt)
    rows_f: list[int] = []
    x_input_f: list[float] = []
    for i in range(n):
        val_f = filtered[i]
        if val_f is None:
            continue
        x_val = val_f
        if delta_from_room:
            x_val = x_val - observations[i].current_c
        rows_f.append(i)
        x_input_f.append(x_val)
    p = 5
    XtWX_f = [[0.0] * p for _ in range(p)]
    XtWy_f = [0.0] * p
    for j, i in enumerate(rows_f):
        od_i = base_X[i][1] if base_X else 0.0
        s_i, c_i = tod_cols[i] if tod_cols else (0.0, 0.0)
        row = [1.0, od_i, s_i, c_i, x_input_f[j]]
        wi = weights[i]
        for a in range(p):
            XtWy_f[a] += row[a] * wi * y_values[i]
            for b_idx in range(p):
                XtWX_f[a][b_idx] += row[a] * wi * row[b_idx]
    for a in range(p):
        XtWX_f[a][a] += 1e-6
    beta_f = _solve_symmetric(XtWX_f, XtWy_f, p)
    beta_input = beta_f[4] if beta_f else 0.0

    # ESS-substituted BIC: replace raw n_eff with ESS computed from the
    # tau_opt-fitted residuals.  Schwarz 1978 BIC assumes IID errors;
    # thermal residuals are heavily autocorrelated at fine cadence, so
    # raw n overstates the information content.  ESS = N/(1+2Σρ_k) via
    # Bartlett-weighted ACF (Bartlett 1946; Newey-West 1987 kernel).
    # Falls back to raw n_eff if residuals are degenerate or beta_f
    # failed (preserves classical behavior in pathological cases).
    n_for_bic: float = float(n_eff)
    if beta_f is not None and rows_f:
        m_f = len(rows_f)
        resid_f = [0.0] * m_f
        w_f = [0.0] * m_f
        for j, i in enumerate(rows_f):
            od_i = base_X[i][1] if base_X else 0.0
            s_i, c_i = tod_cols[i] if tod_cols else (0.0, 0.0)
            row = [1.0, od_i, s_i, c_i, x_input_f[j]]
            pred = sum(beta_f[a] * row[a] for a in range(p))
            resid_f[j] = y_values[i] - pred
            w_f[j] = weights[i]
        # Wall-clock-anchored bandwidth — same primitive as HAC std_err.
        rows_obs = [observations[i] for i in rows_f]
        dt_min_bic = _estimate_observation_dt_min(rows_obs)
        bw_bic = _hac_bandwidth(m_f, dt_min_bic)
        acf = _bartlett_acf(resid_f, w_f, bw_bic)
        n_for_bic = _ess_from_acf(acf, m_f)

    # BIC test: τ adds one nuisance parameter (k=1).  Accept iff
    # ESS·log(RSS_0/RSS_τ) > log(ESS) (equivalently ΔBIC < 0).  Standard
    # nested-model criterion in system identification, with ESS replacing
    # raw N to honor autocorrelated residuals (see ESS computation above).
    bic_gain = n_for_bic * math.log(rss_raw / rss_opt)
    bic_threshold = math.log(n_for_bic)

    # Boundary-hit detection: τ_opt within ε of the upper rail is the
    # Raue 2009 practical-non-identifiability signal — the optimizer has
    # no interior preference, so the reported value is the rail itself,
    # not an estimate.  See module-level rationale on _TAU_SEARCH_MAX_BY_ROLE.
    boundary_hit = (
        search_max > 0
        and (search_max - tau_opt) / search_max < _TAU_BOUNDARY_FRACTION
    )

    # Apply acceptance gates: BIC first, then sub-floor snap, then
    # boundary-hit (only after BIC and floor pass — a bic_failed rail-hit
    # is more usefully reported as bic_failed).
    if bic_gain < bic_threshold:
        return LagTauDiagnostic(
            tau=0.0, tau_opt_raw=tau_opt,
            bic_gain=bic_gain, bic_threshold=bic_threshold,
            r2_improvement=r2_improvement, beta_at_tau=beta_input,
            n_eff=n_eff, accepted=False, reject_reason="bic_failed",
            search_max_used=search_max, boundary_hit=boundary_hit,
        )

    if tau_opt < _TAU_MIN_MEANINGFUL:
        return LagTauDiagnostic(
            tau=0.0, tau_opt_raw=tau_opt,
            bic_gain=bic_gain, bic_threshold=bic_threshold,
            r2_improvement=r2_improvement, beta_at_tau=beta_input,
            n_eff=n_eff, accepted=False, reject_reason="below_floor",
            search_max_used=search_max, boundary_hit=boundary_hit,
        )

    if boundary_hit:
        return LagTauDiagnostic(
            tau=0.0, tau_opt_raw=tau_opt,
            bic_gain=bic_gain, bic_threshold=bic_threshold,
            r2_improvement=r2_improvement, beta_at_tau=beta_input,
            n_eff=n_eff, accepted=False, reject_reason="boundary_hit",
            search_max_used=search_max, boundary_hit=True,
        )

    return LagTauDiagnostic(
        tau=tau_opt, tau_opt_raw=tau_opt,
        bic_gain=bic_gain, bic_threshold=bic_threshold,
        r2_improvement=r2_improvement, beta_at_tau=beta_input,
        n_eff=n_eff, accepted=True, reject_reason="",
        search_max_used=search_max, boundary_hit=False,
    )


class DiversityAwareBuffer:
    """Policy-driven observation buffer for long-term diverse data retention.

    Instead of FIFO eviction, retains observations selected by a
    pluggable ``BufferPolicy``.  The default ``SlevPolicy(alpha=0.0)``
    performs uniform-random retention — the bench-validated unbiased
    baseline.  Other policies (``LeveragePolicy``, ``DOptimalPolicy``,
    ``MinEigPolicy``, ``AOptimalPolicy``, ``SlidingWindowPolicy``) can be
    selected explicitly; they trade unbiasedness for rare-regime retention
    or recency.  When the buffer is full, a candidate displaces an
    incumbent iff the policy says so.

    Feature vectors are built on the fly from raw readings + current
    model config.  Scoring adapts when config changes — an observation
    that was low-score under the old config may become high-score under
    the new one (e.g., after adding a solar input, observations from
    sunny days become more valuable).

    The buffer owns storage, info-matrix maintenance, and Sherman-Morrison
    up/downdates; the policy only sees vectors and matrices for the
    admit/evict decision.  See ``buffer_policies`` for available policies.

    References:
    - Chowdhary & Johnson, ACC 2011 — concurrent learning history stack
    - Atkinson & Donev, "Optimum Experimental Designs" — D-optimal sequential design
    """

    def __init__(
        self,
        n_features: int,
        max_size: int = DEFAULT_DIVERSITY_BUFFER_SIZE,
        feature_order: list[str] | None = None,
        model_inputs: list[dict[str, Any]] | None = None,
        policy: BufferPolicy | None = None,
    ) -> None:
        if feature_order is not None and len(feature_order) != n_features:
            raise ValueError(
                f"feature_order length ({len(feature_order)}) must equal "
                f"n_features ({n_features}); legacy code silently truncated "
                f"feature vectors when these disagreed, hiding configuration "
                f"errors."
            )
        self._buffer: list[Observation] = []
        self._max_size = max_size
        self._n_features = n_features
        self._feature_order: list[str] | None = feature_order
        self._model_inputs: list[dict[str, Any]] = model_inputs or []
        # Default: SlevPolicy(alpha=0.0) — uniform-random eviction.  Replaces
        # historical LeveragePolicy default after the 2026-05-07 finding that
        # deterministic top-N leverage selection biases β_solar by 35–40% on
        # real-CSV corpora (`project_bench_solar_fidelity.md`).  Uniform random
        # is the bench-validated unbiased baseline; LeveragePolicy stays
        # available for explicit selection.
        self._policy: BufferPolicy = policy if policy is not None else SlevPolicy(alpha=0.0)
        # (X^T X + λI)^{-1} — the inverse information matrix, n×n.
        # Initialized to (1/λ) * I (no data yet).
        n = n_features
        reg_inv = 1.0 / INFO_MATRIX_REGULARIZATION
        self._info_inv: list[list[float]] = [
            [reg_inv if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        # Forward matrix X^T X + λI, maintained incrementally alongside
        # _info_inv via the Sherman-Morrison up/downdate paths.  Some
        # policies (e.g. MinEigPolicy) need a fresh forward matrix each
        # tick to compute eigenvalues without paying for a full recompute.
        self._xtx_matrix: list[list[float]] = [
            [INFO_MATRIX_REGULARIZATION if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        # Counter for incremental updates since last full recomputation.
        self._updates_since_recompute: int = 0

    @property
    def n_features(self) -> int:
        return self._n_features

    def clear(self) -> None:
        """Remove all observations and reset the information matrix."""
        self._buffer.clear()
        n = self._n_features
        reg_inv = 1.0 / INFO_MATRIX_REGULARIZATION
        self._info_inv = [
            [reg_inv if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        self._xtx_matrix = [
            [INFO_MATRIX_REGULARIZATION if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        self._updates_since_recompute = 0

    def update_config(
        self,
        feature_order: list[str],
        model_inputs: list[dict[str, Any]],
    ) -> None:
        """Atomically update feature order and model config.

        Both must be updated together — feature_order defines which columns
        the info matrix tracks, and model_inputs defines how to build those
        columns from raw_readings.  Updating one without the other causes
        silent leverage scoring corruption.
        """
        self._feature_order = feature_order
        self._n_features = len(feature_order)
        self._model_inputs = model_inputs
        self.recompute_info_matrix()

    def filter_inactive(self, mode: str) -> int:
        """Remove observations where the HP had zero output.

        When the HP setpoint is below room temp in heating (or above in
        cooling), the HP's internal thermostat turns off the compressor.
        These observations carry no plant information (Ljung §13.3) and
        corrupt the WLS regression with passive room dynamics.

        Re-evaluates the condition from stored hp_setpoint and current_c
        on each observation.  Call once after adding the hp_no_output
        condition to purge historical poisoned data.

        Args:
            mode: "heat" or "cool"

        Returns:
            Number of observations removed.
        """
        before = len(self._buffer)
        if mode == "heat":
            self._buffer = [
                o for o in self._buffer
                if o.clamped_reason == "no_output"
                or o.hp_setpoint is None
                or not (o.hp_setpoint < o.current_c)
            ]
        else:
            self._buffer = [
                o for o in self._buffer
                if o.clamped_reason == "no_output"
                or o.hp_setpoint is None
                or not (o.hp_setpoint > o.current_c)
            ]
        removed = before - len(self._buffer)
        if removed:
            self.recompute_info_matrix()
        return removed

    def exclude_time_range(self, start: float, end: float) -> int:
        """Remove observations within a monotonic timestamp range.

        Used by anomaly detection to purge contaminated observations.
        Same pattern as filter_inactive(): filter + recompute info matrix.

        Returns number of observations removed.
        """
        before = len(self._buffer)
        self._buffer = [
            o for o in self._buffer
            if o.timestamp < start or o.timestamp > end
        ]
        removed = before - len(self._buffer)
        if removed:
            self.recompute_info_matrix()
        return removed

    def add(self, obs: Observation) -> BufferAddResult:
        """Add an observation, using the configured policy when full.

        Returns a `BufferAddResult` describing the decision so callers
        can populate observability snapshots without re-deriving state.

        Dispatch:
        - Buffer not full → unconditionally admit; ``score_candidate``
          is called only for observability.
        - Buffer full with a joint policy (``hasattr(policy,
          'attempt_exchange')``) → policy returns ``ExchangeChoice``
          covering both admit and evictee selection.
        - Buffer full with a simple policy → use ``score_candidate``
          + ``find_evictee`` + ``should_admit`` independently.
        """
        x = self._get_feature_vector(obs)
        policy_name = self._policy.name

        if len(self._buffer) < self._max_size:
            # Buffer not full — always accept.
            cand_score = self._policy.score_candidate(
                x, self._info_inv, self._xtx_matrix, len(self._buffer),
            )
            self._buffer.append(obs)
            self._sherman_morrison_update(x)
            return BufferAddResult(
                admitted=True,
                candidate_score=cand_score,
                evicted_timestamp=None,
                min_incumbent_score=None,
                rejection_reason=None,
                policy_name=policy_name,
            )

        feature_vectors = [self._get_feature_vector(o) for o in self._buffer]

        # Joint-optimization policies (e.g. DOptimalPolicy) supply
        # ``attempt_exchange`` so admit/evict can consider the candidate↔
        # incumbent interaction together.  Falls back to the simple path
        # when not implemented.
        attempt = getattr(self._policy, "attempt_exchange", None)
        if attempt is not None:
            decision = attempt(
                x, self._buffer, feature_vectors,
                self._info_inv, self._xtx_matrix,
            )
            if decision is not None:  # pragma: no branch — attempt_exchange returns None only when buffer is empty
                if decision.admit:
                    evicted_ts = self._buffer[decision.evictee_index].timestamp
                    old_x = feature_vectors[decision.evictee_index]
                    self._sherman_morrison_downdate(old_x)
                    self._buffer[decision.evictee_index] = obs
                    self._sherman_morrison_update(x)
                    return BufferAddResult(
                        admitted=True,
                        candidate_score=decision.candidate_score,
                        evicted_timestamp=evicted_ts,
                        min_incumbent_score=decision.evictee_score,
                        rejection_reason=None,
                        policy_name=policy_name,
                    )
                return BufferAddResult(
                    admitted=False,
                    candidate_score=decision.candidate_score,
                    evicted_timestamp=None,
                    min_incumbent_score=decision.evictee_score,
                    rejection_reason=(
                        decision.rejection_reason
                        or f"{policy_name}_rejected"
                    ),
                    policy_name=policy_name,
                )

        # Simple-policy path: independent score / find_evictee / admit.
        cand_score = self._policy.score_candidate(
            x, self._info_inv, self._xtx_matrix, len(self._buffer),
        )
        evictee = self._policy.find_evictee(
            self._buffer, feature_vectors, self._info_inv, self._xtx_matrix,
        )
        if self._policy.should_admit(cand_score, evictee.score):
            evicted_ts = self._buffer[evictee.index].timestamp
            old_x = feature_vectors[evictee.index]
            self._sherman_morrison_downdate(old_x)
            self._buffer[evictee.index] = obs
            self._sherman_morrison_update(x)
            return BufferAddResult(
                admitted=True,
                candidate_score=cand_score,
                evicted_timestamp=evicted_ts,
                min_incumbent_score=evictee.score,
                rejection_reason=None,
                policy_name=policy_name,
            )

        return BufferAddResult(
            admitted=False,
            candidate_score=cand_score,
            evicted_timestamp=None,
            min_incumbent_score=evictee.score,
            rejection_reason=f"{policy_name}_rejected",
            policy_name=policy_name,
        )

    def _find_min_leverage_idx(self) -> tuple[int, float]:
        """Find the index and value of the lowest-leverage observation.

        Uses numpy vectorized einsum when available (55× faster for
        large buffers), falls back to pure-Python loop.
        """
        if _NUMPY_AVAILABLE and len(self._buffer) > 50:
            return self._find_min_leverage_idx_np()
        # Pure-Python fallback.
        min_idx = 0
        min_lev = self._compute_leverage(self._get_feature_vector(self._buffer[0]))
        for i in range(1, len(self._buffer)):
            lev = self._compute_leverage(self._get_feature_vector(self._buffer[i]))
            if lev < min_lev:
                min_lev = lev
                min_idx = i
        return min_idx, min_lev

    def _find_min_leverage_idx_np(self) -> tuple[int, float]:
        """Vectorized min-leverage scan using numpy."""
        n = self._n_features
        m = len(self._buffer)
        X = np.empty((m, n), dtype=np.float64)
        for i, obs in enumerate(self._buffer):
            X[i] = self._get_feature_vector(obs)
        info_inv = np.array(self._info_inv, dtype=np.float64)
        # leverages[i] = X[i] @ info_inv @ X[i]
        leverages = np.einsum('ij,jk,ik->i', X, info_inv, X)
        min_idx = int(np.argmin(leverages))
        return min_idx, float(leverages[min_idx])

    def get_all(self) -> list[Observation]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def as_list(self) -> list[dict[str, Any]]:
        return [o.as_dict() for o in self._buffer]

    @classmethod
    def from_list(
        cls,
        data: list[dict[str, Any]],
        n_features: int,
        feature_order: list[str],
        model_inputs: list[dict[str, Any]],
        max_size: int = DEFAULT_DIVERSITY_BUFFER_SIZE,
    ) -> DiversityAwareBuffer:
        """Deserialize from stored dicts, recomputing the info matrix.

        Corrupt or unreadable entries are silently skipped.
        """
        buf = cls(n_features, max_size, feature_order=feature_order,
                  model_inputs=model_inputs)
        observations: list[Observation] = []
        for d in data:
            try:
                observations.append(Observation.from_dict(d))
            except (KeyError, TypeError, ValueError):
                continue
        n_skipped = len(data) - len(observations)
        if n_skipped > 0:
            _LOGGER.info(
                "Skipped %d unreadable observations during buffer restore",
                n_skipped,
            )
        if len(observations) > max_size:
            observations = observations[-max_size:]
        buf._buffer = observations
        buf.recompute_info_matrix()
        return buf

    def _get_feature_vector(self, obs: Observation) -> list[float]:
        """Build ordered feature vector from raw readings for leverage scoring.

        Returns a complete vector when all model inputs are present, or a
        partial vector (intercept + outdoor_delta, zeros for missing inputs)
        when the observation predates a model input.  Partial vectors get
        low leverage on model-input dimensions and will be evicted naturally
        as complete observations accumulate.
        """
        if self._feature_order is not None and self._model_inputs is not None:
            vec = build_feature_vector_from_raw(
                obs, self._model_inputs, self._feature_order,
            )
            if vec is not None:
                return vec
        # Partial: base features only (intercept + outdoor_delta)
        partial = [0.0] * self._n_features
        if self._n_features > 0:  # pragma: no branch — n_features is always ≥ 2
            partial[0] = 1.0  # intercept
        if self._n_features > 1 and obs.outdoor_temp_c is not None:  # pragma: no branch — n always ≥ 2; obs filtered upstream
            partial[1] = obs.outdoor_temp_c - obs.desired_c
        return partial

    def recompute_info_matrix(self) -> None:
        """Recompute (X^T X + λI)^{-1} from scratch.

        Call periodically (e.g., at each 12h batch cycle) to prevent
        numerical drift from incremental Sherman-Morrison updates.
        Also stores the forward matrix for condition number estimation.
        """
        n = self._n_features
        # Build X^T X + λI
        xtx = [
            [INFO_MATRIX_REGULARIZATION if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        for obs in self._buffer:
            x = self._get_feature_vector(obs)
            for i in range(n):
                for j in range(n):
                    xtx[i][j] += x[i] * x[j]

        # Store forward matrix for condition number estimation
        self._xtx_matrix = [row[:] for row in xtx]

        # Invert via Gaussian elimination
        inv = self._invert_matrix(xtx, n)
        if inv is not None:
            self._info_inv = inv
        else:
            # Fallback: increase regularization
            for i in range(n):
                xtx[i][i] += 1e-2
            inv = self._invert_matrix(xtx, n)
            if inv is not None:  # pragma: no branch — extra regularization makes the system invertible
                self._info_inv = inv

        self._updates_since_recompute = 0

    def _ensure_xtx_matrix(self) -> None:
        """Ensure the forward X'X matrix is available for VIF/κ computation.

        The Sherman-Morrison incremental updates only maintain the inverse.
        The forward matrix is computed from scratch by recompute_info_matrix()
        and stored separately.  Call this before any method that needs X'X.
        """
        if self._xtx_matrix is None and len(self._buffer) > 0:
            self.recompute_info_matrix()

    def _corr_matrix_no_intercept(self) -> list[list[float]] | None:
        """Build column-normalized correlation matrix, excluding the intercept.

        Belsley, Kuh & Welsch (1980) recommend excluding the constant
        column when diagnosing multicollinearity.  The intercept creates
        "essential ill-conditioning" — a structural artifact that inflates
        κ and VIF without reflecting actual feature-to-feature collinearity.

        Returns an (n-1)×(n-1) correlation matrix for features 1..n-1,
        or None if the matrix cannot be computed.
        """
        if len(self._buffer) == 0:
            # No data → feature correlations are undefined, even though
            # _xtx_matrix carries the λI regularization seed.
            return None
        self._ensure_xtx_matrix()
        if self._xtx_matrix is None:
            return None
        n = self._n_features
        if n < 3:
            # With only intercept + 1 feature, no feature-feature κ to compute
            return None

        # Extract the (n-1)×(n-1) sub-matrix excluding row/col 0 (intercept)
        sub = [
            [self._xtx_matrix[i][j] for j in range(1, n)]
            for i in range(1, n)
        ]
        m = n - 1

        # Column-normalize to correlation matrix
        diag_sqrt = [
            math.sqrt(sub[i][i]) if sub[i][i] > 1e-15 else 1.0
            for i in range(m)
        ]
        corr = [
            [sub[i][j] / (diag_sqrt[i] * diag_sqrt[j]) for j in range(m)]
            for i in range(m)
        ]
        return corr

    def compute_condition_number(self) -> float:
        """Compute the spectral condition number, excluding the intercept.

        Operates on the column-normalized correlation matrix of features
        1..n-1 (excluding the constant intercept at index 0).  This
        measures true feature-to-feature multicollinearity without the
        structural inflation from the intercept column (Belsley 1980).

        κ = √(λ_max / λ_min) where λ are eigenvalues of the correlation
        matrix.

        Thresholds (Belsley, Kuh & Welsch, "Regression Diagnostics", 1980):
        - κ > 30: moderate multicollinearity, coefficients becoming unreliable
        - κ > 100: severe multicollinearity, coefficient estimates numerically unstable

        Uses power iteration for λ_max and inverse iteration for λ_min.
        Returns inf if the matrix cannot be computed (empty buffer).
        """
        corr = self._corr_matrix_no_intercept()
        if corr is None:
            n = self._n_features
            if n == 2:
                return 1.0  # intercept + 1 feature → no collinearity
            return float('inf')
        m = len(corr)

        # Compute eigenvalues of the m×m correlation matrix.
        # For small m (typical: 1-6 features after excluding intercept),
        # use QR iteration or direct formula.
        eigenvalues = self._eigenvalues_symmetric(corr, m)
        if eigenvalues is None:
            return float('inf')

        lambda_max = max(eigenvalues)
        lambda_min = min(eigenvalues)
        if lambda_min < 1e-15:
            return float('inf')

        return math.sqrt(lambda_max / lambda_min)

    def compute_vif(self) -> list[float]:
        """Compute per-feature Variance Inflation Factor, excluding intercept.

        VIF_j = diag((corr)⁻¹)_j where corr is the correlation matrix of
        features 1..n-1 (excluding the constant intercept at index 0).
        VIF > 10 indicates the feature is too collinear with other features
        for reliable estimation (Belsley 1980).

        Returns a list of n_features VIF values.  Index 0 (intercept) is
        always 1.0.  Returns all inf if the matrix cannot be computed.
        """
        n = self._n_features
        if n < 3:
            # intercept + ≤1 feature → no feature-feature collinearity
            return [1.0] * n

        corr = self._corr_matrix_no_intercept()
        if corr is None:
            return [float('inf')] * n
        m = len(corr)  # n - 1

        corr_inv = self._invert_matrix([row[:] for row in corr], m)
        if corr_inv is None:
            return [float('inf')] * n

        # Build result: intercept VIF = 1.0, rest from corr_inv diagonal
        result = [1.0]  # index 0 = intercept
        for i in range(m):
            result.append(max(corr_inv[i][i], 1.0))
        return result

    def get_pairwise_correlations(
        self, feature_names: list[str] | None = None,
        *, include_top: bool = False,
    ) -> list[tuple[str, str, float]]:
        """Compute pairwise Pearson correlations between features.

        Returns (name_i, name_j, r) for all pairs with |r| > 0.7,
        skipping the intercept (always 1.0, undefined correlation).
        Only uses unclamped observations for relevance to WLS.

        If *include_top* is True and no pair exceeds 0.7, the single
        highest-|r| pair is returned so callers always have context.
        """
        n = self._n_features
        if n < 3 or len(self._buffer) < 20:
            return []

        names = feature_names or [f"feature_{i}" for i in range(n)]
        unclamped = [o for o in self._buffer if not o.clamped]
        if len(unclamped) < 20:
            return []

        m = len(unclamped)
        # Extract columns (skip intercept at index 0)
        cols: list[list[float]] = []
        for j in range(1, n):
            col: list[float] = []
            for k in range(m):
                x = self._get_feature_vector(unclamped[k])
                col.append(x[j] if j < len(x) else 0.0)
            cols.append(col)

        results: list[tuple[str, str, float]] = []
        top_pair: tuple[str, str, float] | None = None
        nc = len(cols)
        for a in range(nc):
            for b in range(a + 1, nc):
                mean_a = sum(cols[a]) / m
                mean_b = sum(cols[b]) / m
                cov_ab = sum((cols[a][k] - mean_a) * (cols[b][k] - mean_b) for k in range(m)) / m
                var_a = sum((cols[a][k] - mean_a) ** 2 for k in range(m)) / m
                var_b = sum((cols[b][k] - mean_b) ** 2 for k in range(m)) / m
                denom = math.sqrt(var_a * var_b)
                if denom < 1e-12:
                    continue
                r = cov_ab / denom
                if abs(r) > 0.7:
                    # a, b are 0-indexed into cols which starts at feature 1
                    results.append((names[a + 1], names[b + 1], r))
                if top_pair is None or abs(r) > abs(top_pair[2]):
                    top_pair = (names[a + 1], names[b + 1], r)

        if not results and include_top and top_pair is not None:
            results.append(top_pair)

        return results

    def get_leverage_scores(self) -> list[float]:
        """Compute and return current leverage scores (for diagnostics)."""
        if _NUMPY_AVAILABLE and len(self._buffer) > 50:
            n = self._n_features
            X = np.empty((len(self._buffer), n), dtype=np.float64)
            for i, obs in enumerate(self._buffer):
                X[i] = self._get_feature_vector(obs)
            info_inv = np.array(self._info_inv, dtype=np.float64)
            result: list[float] = np.einsum('ij,jk,ik->i', X, info_inv, X).tolist()
            return result
        return [self._compute_leverage(self._get_feature_vector(o)) for o in self._buffer]

    def get_min_leverage(self) -> float:
        """Return the minimum leverage score in the buffer."""
        if not self._buffer:
            return 0.0
        _, min_lev = self._find_min_leverage_idx()
        return min_lev

    def _compute_leverage(self, x: list[float]) -> float:
        """Compute leverage score: x^T (X^T X + λI)^{-1} x."""
        n = self._n_features
        # (X^T X + λI)^{-1} x
        inv_x = [0.0] * n
        for i in range(n):
            for j in range(n):
                inv_x[i] += self._info_inv[i][j] * x[j]
        # x^T (result)
        return sum(x[i] * inv_x[i] for i in range(n))

    def _sherman_morrison_update(self, x: list[float]) -> None:
        """Rank-1 downdate of the inverse: (A + xx^T)^{-1} via Sherman-Morrison.

        (A + xx^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)

        Also maintains the forward matrix _xtx_matrix via the matching
        rank-1 outer-product addition, keeping the two views consistent.
        """
        n = self._n_features
        # A^{-1} x
        inv_x = [sum(self._info_inv[i][j] * x[j] for j in range(n)) for i in range(n)]
        # 1 + x^T A^{-1} x
        denom = 1.0 + sum(x[i] * inv_x[i] for i in range(n))
        if abs(denom) < 1e-15:
            return  # near-singular, skip incremental update
        # A^{-1} -= (A^{-1} x)(x^T A^{-1}) / denom
        for i in range(n):
            for j in range(n):
                self._info_inv[i][j] -= inv_x[i] * inv_x[j] / denom
        # Forward: _xtx_matrix += xx^T
        for i in range(n):
            for j in range(n):
                self._xtx_matrix[i][j] += x[i] * x[j]
        self._updates_since_recompute += 1

    def _sherman_morrison_downdate(self, x: list[float]) -> None:
        """Rank-1 update for removing an observation: (A - xx^T)^{-1}.

        (A - xx^T)^{-1} = A^{-1} + (A^{-1} x x^T A^{-1}) / (1 - x^T A^{-1} x)

        Also maintains the forward matrix _xtx_matrix via the matching
        rank-1 outer-product subtraction.
        """
        n = self._n_features
        inv_x = [sum(self._info_inv[i][j] * x[j] for j in range(n)) for i in range(n)]
        denom = 1.0 - sum(x[i] * inv_x[i] for i in range(n))
        if abs(denom) < 1e-15:
            # Near-singular downdate — schedule full recomputation instead.
            self._updates_since_recompute = 999
            return
        for i in range(n):
            for j in range(n):
                self._info_inv[i][j] += inv_x[i] * inv_x[j] / denom
        # Forward: _xtx_matrix -= xx^T
        for i in range(n):
            for j in range(n):
                self._xtx_matrix[i][j] -= x[i] * x[j]
        self._updates_since_recompute += 1

    @property
    def needs_recompute(self) -> bool:
        """Whether the info matrix should be recomputed from scratch."""
        return self._updates_since_recompute > 500

    @staticmethod
    def _invert_matrix(A: list[list[float]], n: int) -> list[list[float]] | None:
        """Invert an n×n matrix via Gauss-Jordan elimination.

        Returns None if singular.  For n=3-8 this is instantaneous.
        """
        # Build [A | I]
        M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        for col in range(n):
            # Partial pivoting
            max_row = col
            max_val = abs(M[col][col])
            for row in range(col + 1, n):
                if abs(M[row][col]) > max_val:
                    max_val = abs(M[row][col])
                    max_row = row
            if max_val < 1e-14:
                return None
            M[col], M[max_row] = M[max_row], M[col]

            # Scale pivot row
            pivot = M[col][col]
            for j in range(2 * n):
                M[col][j] /= pivot

            # Eliminate column
            for row in range(n):
                if row == col:
                    continue
                factor = M[row][col]
                for j in range(2 * n):
                    M[row][j] -= factor * M[col][j]

        # Extract inverse from right half
        return [M[i][n:] for i in range(n)]

    @staticmethod
    def _eigenvalues_symmetric(
        A: list[list[float]], n: int,
    ) -> list[float] | None:
        """Compute eigenvalues of an n×n symmetric matrix.

        Uses numpy when available (LAPACK, exact).  Falls back to direct
        formula (n≤2) or power/inverse iteration with random start (n>2).
        Returns None if computation fails.
        """
        if _NUMPY_AVAILABLE:
            try:
                eigs: list[float] = np.linalg.eigvalsh(np.array(A)).tolist()
                return eigs
            except np.linalg.LinAlgError:
                return None

        if n == 1:
            return [A[0][0]]
        if n == 2:
            tr = A[0][0] + A[1][1]
            det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
            disc = max(0.0, tr * tr - 4 * det)
            sd = math.sqrt(disc)
            return [(tr + sd) / 2, (tr - sd) / 2]

        # n > 2 without numpy: power iteration with random start for
        # λ_max, inverse iteration for λ_min.  Random start avoids the
        # uniform-vector pitfall where [1/√n, ...] aligns with an
        # eigenvector of correlation matrices.
        import random as _rng
        rng = _rng.Random(42)

        v = [rng.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(vi * vi for vi in v))
        v = [vi / norm for vi in v]
        lam_max = 0.0
        for _ in range(100):
            w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
            lam_max = sum(v[i] * w[i] for i in range(n))
            norm = math.sqrt(sum(wi * wi for wi in w))
            if norm < 1e-15:
                return None
            v = [wi / norm for wi in w]

        A_inv = DiversityAwareBuffer._invert_matrix([row[:] for row in A], n)
        if A_inv is None:
            return None
        v = [rng.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(vi * vi for vi in v))
        v = [vi / norm for vi in v]
        inv_lam_min = 0.0
        for _ in range(100):
            w = [sum(A_inv[i][j] * v[j] for j in range(n)) for i in range(n)]
            inv_lam_min = sum(v[i] * w[i] for i in range(n))
            norm = math.sqrt(sum(wi * wi for wi in w))
            if norm < 1e-15:
                return None
            v = [wi / norm for wi in w]

        if inv_lam_min < 1e-15:  # pragma: no cover — IEEE 754 underflow guard
            return None
        lam_min = 1.0 / inv_lam_min
        return [lam_max, lam_min]


@dataclass
class BatchResult:
    """Result of a batch WLS analysis."""

    n_total: int  # total observations in buffer
    n_eligible: int  # observations that passed filters
    beta_batch: list[float]  # WLS coefficient estimate
    beta_current: list[float]  # current RLS coefficients (physical units)
    residual_rms: float  # RMS residual of batch fit
    max_coeff_change_pct: float  # largest coefficient change (%)
    recommend_update: bool  # True if batch differs significantly
    n_outliers_excluded: int = 0  # observations excluded by residual filter
    held_features: set[int] = field(default_factory=set)  # indices held at current (insufficient variance)
    beta_std_err: list[float] = field(default_factory=list)  # per-coefficient standard error from WLS
    beta_blended: list[float] = field(default_factory=list)  # safe update after covariance-weighted blend
    blend_gains: list[float] = field(default_factory=list)  # per-coefficient Kalman gain K_i ∈ [0, 1]
    plant_snapshot: dict[str, Any] = field(default_factory=dict)  # plant ID state at batch time
    feature_vif: list[float] = field(default_factory=list)  # per-feature VIF from regression data
    detected_tau: dict[str, float] = field(default_factory=dict)  # input name → auto-detected EMA tau (seconds)
    detected_tau_diagnostics: dict[str, LagTauDiagnostic] = field(default_factory=dict)
    """Per-input τ search diagnostics (BIC gain, R², β, accept/reject reason).

    Sibling to ``detected_tau`` rather than a replacement: the online path
    reads ``detected_tau`` directly for the smoothed/confirmed EMA filter;
    diagnostics are an additive observability surface for debug bundles.
    Empty when ``detect_lag=False`` or when no model inputs are configured.
    """


def _weighted_variance(values: list[float], weights: list[float]) -> float:
    """Compute weighted variance of a feature column."""
    w_sum = sum(weights)
    if w_sum < 1e-12:
        return 0.0
    mean = sum(v * w for v, w in zip(values, weights)) / w_sum
    return sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / w_sum


# Minimum weighted variance for a feature to be included in the regression.
# Only blocks structurally unidentifiable features (constant columns that
# are collinear with the intercept).  Column normalization + ridge
# regularization handle low-variance-but-varying features numerically.
MIN_FEATURE_VARIANCE = 1e-12


@dataclass
class _RegressionContext:
    """Classified observation data ready for regression.

    Built once by weighted_least_squares(), consumed by _solve_joint()
    or _solve_fwl().  Separating data preparation from the solve makes
    it easy to swap in a grey-box solver later.
    """

    n: int  # total feature count
    n_base: int  # always 2 (intercept + outdoor_delta)
    m_base: int  # number of base-eligible observations
    base_eligible: list[Observation]
    y_base: list[float]
    w_base: list[float]
    X_base: list[list[float]]
    col_scales_base: list[float]
    XtWX_base: list[list[float]]
    beta_base: list[float]
    ridge: float
    # Per model-input classification
    m_inputs: list[dict[str, Any]]
    input_entity_ids: list[str]
    input_values_by_obs: list[list[float | None]]
    feature_obs_counts: dict[int, int]
    active_input_indices: list[int]
    held: set[int]
    complete_indices: list[int]
    min_feature_variance: float


def _solve_joint(
    ctx: _RegressionContext,
) -> tuple[list[float], list[float]] | None:
    """Joint OLS on observations with ALL active model inputs.

    Returns (beta, std_err) or None if the joint solve fails.
    Mathematically equivalent to standard OLS — no approximation.

    When observations span sufficient time diversity, sin/cos
    time-of-day columns are included as nuisance regressors to
    decorrelate diurnally confounded features (e.g. solar proxy
    vs outdoor_delta).  Their coefficients are estimated but
    discarded — only the decorrelation effect on other features
    matters.  This is algebraically equivalent to the FWL
    augmented partialling in ``_solve_fwl``.
    """
    n_active = len(ctx.active_input_indices)
    m_complete = len(ctx.complete_indices)

    # Build joint feature matrix [intercept, outdoor_delta, input_0, ...]
    joint_cols: list[list[float]] = []
    for j in range(ctx.n_base):
        joint_cols.append([ctx.X_base[k][j] for k in ctx.complete_indices])
    for fi in ctx.active_input_indices:
        joint_cols.append([ctx.input_values_by_obs[k][fi] for k in ctx.complete_indices])  # type: ignore[misc]  # complete_indices guarantees not-None

    # Augment with sin/cos nuisance columns for diurnal decorrelation.
    # Same logic as _solve_fwl: skip if observations lack time diversity.
    # If an active feature is collinear with sin/cos (e.g. a sinusoidal
    # input schedule), the augmented X'WX is singular → _solve_symmetric
    # returns None → caller falls back to _solve_fwl, which handles
    # rank deficiency via staged residualization (FWL theorem).
    n_tod = 0
    if m_complete >= 4 and ctx.base_eligible[ctx.complete_indices[0]] is not None:  # pragma: no branch — base_eligible is always populated when m_complete ≥ 4
        tod_sin = [0.0] * m_complete
        tod_cos = [0.0] * m_complete
        for idx, k in enumerate(ctx.complete_indices):
            obs = ctx.base_eligible[k]
            wt = obs.wall_time if obs is not None else 0.0
            s, c = tod_features(wt)
            tod_sin[idx] = s
            tod_cos[idx] = c
        mean_s = sum(tod_sin) / m_complete
        mean_c = sum(tod_cos) / m_complete
        var_s = sum((s - mean_s) ** 2 for s in tod_sin) / m_complete
        var_c = sum((c - mean_c) ** 2 for c in tod_cos) / m_complete
        if var_s >= 0.001 or var_c >= 0.001:
            joint_cols.append(tod_sin)
            joint_cols.append(tod_cos)
            n_tod = 2

    n_joint = ctx.n_base + n_active + n_tod

    # Column normalization
    col_scales = [1.0] * n_joint
    col_scales[1] = ctx.col_scales_base[1]
    for jj in range(ctx.n_base, n_joint):
        col = joint_cols[jj]
        mean_j = sum(col) / m_complete
        var_j = sum((c - mean_j) ** 2 for c in col) / m_complete
        col_scales[jj] = math.sqrt(var_j) if var_j > 1e-12 else 1.0

    y = [ctx.y_base[k] for k in ctx.complete_indices]
    w = [ctx.w_base[k] for k in ctx.complete_indices]

    XtWX = [[0.0] * n_joint for _ in range(n_joint)]
    XtWy = [0.0] * n_joint
    for idx in range(m_complete):
        for ii in range(n_joint):
            xi = joint_cols[ii][idx] / col_scales[ii]
            XtWy[ii] += xi * w[idx] * y[idx]
            for jj in range(n_joint):
                xj = joint_cols[jj][idx] / col_scales[jj]
                XtWX[ii][jj] += xi * w[idx] * xj
    for ii in range(n_joint):
        XtWX[ii][ii] += ctx.ridge

    beta_norm = _solve_symmetric(XtWX, XtWy, n_joint)
    if beta_norm is None:
        return None

    # Denormalize into full-size beta vector.
    # Sin/cos nuisance coefficients are estimated but discarded.
    beta = [0.0] * ctx.n
    beta[0] = beta_norm[0] / col_scales[0]
    beta[1] = beta_norm[1] / col_scales[1]
    for jj, fi in enumerate(ctx.active_input_indices):
        beta[fi + 2] = beta_norm[ctx.n_base + jj] / col_scales[ctx.n_base + jj]

    # Standard errors: classical (X'WX)⁻¹·σ² baseline + Newey-West HAC
    # sandwich (uses Bartlett-weighted residual ACF; cadence-invariant via
    # wall-clock-anchored bandwidth).  HAC kicks in for autocorrelated
    # residuals; safeguard falls back to classical per-coefficient if the
    # sandwich produces a smaller value (numerical pathology, not a real
    # correction).
    std_err = [float("inf")] * ctx.n
    cov_diag = _diagonal_of_inverse(XtWX, n_joint)
    if cov_diag is not None:  # pragma: no branch — XtWX is well-conditioned when _solve_symmetric succeeded
        all_beta_norm = [beta_norm[jj] / col_scales[jj] for jj in range(n_joint)]
        resid = [
            y[idx] - sum(joint_cols[jj][idx] * all_beta_norm[jj] for jj in range(n_joint))
            for idx in range(m_complete)
        ]
        rms_sq = sum(r * r for r in resid) / max(1, m_complete - n_joint)

        # Classical per-coefficient std_err keyed back to ctx.n positions.
        classical_norm = [0.0] * n_joint
        for i in range(n_joint):
            v = rms_sq * max(0.0, cov_diag[i])
            classical_norm[i] = math.sqrt(v) if v > 0 else 0.0

        # HAC sandwich on the normalized design matrix.
        complete_obs = [ctx.base_eligible[k] for k in ctx.complete_indices]
        dt_min = _estimate_observation_dt_min(complete_obs)
        bandwidth = _hac_bandwidth(m_complete, dt_min)
        XtWX_inv = DiversityAwareBuffer._invert_matrix(
            [row[:] for row in XtWX], n_joint,
        )
        if XtWX_inv is not None:
            X_rows = [
                [joint_cols[jj][idx] / col_scales[jj] for jj in range(n_joint)]
                for idx in range(m_complete)
            ]
            meat = _hac_meat_matrix(X_rows, w, resid, bandwidth)
            hac_norm = _hac_std_err_from_sandwich(
                XtWX_inv, meat,
                col_scales=[1.0] * n_joint,  # de-norm done below
                classical_std_err=classical_norm,
            )
        else:  # pragma: no cover — XtWX inversion mirrors cov_diag presence
            hac_norm = classical_norm

        # De-normalize and place into the ctx.n positions (skip sin/cos
        # nuisance coefficients which aren't reported).
        for i in range(ctx.n_base):
            std_err[i] = hac_norm[i] / col_scales[i] if col_scales[i] > 0 else 0.0
        for jj, fi in enumerate(ctx.active_input_indices):
            idx_in = ctx.n_base + jj
            std_err[fi + 2] = (
                hac_norm[idx_in] / col_scales[idx_in] if col_scales[idx_in] > 0 else 0.0
            )

    return beta, std_err


def _solve_fwl(
    ctx: _RegressionContext,
) -> tuple[list[float], list[float]]:
    """Frisch-Waugh-Lovell partial regression for ragged data.

    Used when the complete-data subset is too small for joint OLS but
    individual features have enough observations in their subsets.
    Each feature coefficient is unbiased (base regressors partialled out).
    Base intercept and outdoor_delta are re-estimated afterward.

    Augmented partialling: in addition to [1, outdoor_delta], we partial
    out [sin_hour, cos_hour] from both z and y_sub.  By the FWL theorem
    (Frisch-Waugh-Lovell 1933/63) this is algebraically equivalent to
    including sin/cos in the base regression — decorrelating diurnally
    confounded features (e.g. solar proxy) without changing n_base.
    """
    beta = [0.0] * ctx.n
    beta[0] = ctx.beta_base[0]
    beta[1] = ctx.beta_base[1]
    std_err = [float("inf")] * ctx.n
    held = ctx.held

    residuals_base = [
        ctx.y_base[k] - sum(ctx.beta_base[i] * ctx.X_base[k][i] for i in range(ctx.n_base))
        for k in range(ctx.m_base)
    ]

    # Build sin/cos columns for augmented partialling.
    # Check variance — if observations don't span enough of the day,
    # fall back to base-only partialling (no diurnal decorrelation).
    tod_sin = [0.0] * ctx.m_base
    tod_cos = [0.0] * ctx.m_base
    _use_tod = False
    if ctx.m_base >= 4 and ctx.base_eligible[0] is not None:
        for k in range(ctx.m_base):
            obs_k = ctx.base_eligible[k]
            wt = obs_k.wall_time if obs_k is not None else 0.0
            s, c = tod_features(wt)
            tod_sin[k] = s
            tod_cos[k] = c
        mean_s = sum(tod_sin) / ctx.m_base
        mean_c = sum(tod_cos) / ctx.m_base
        var_s = sum((s - mean_s) ** 2 for s in tod_sin) / ctx.m_base
        var_c = sum((c - mean_c) ** 2 for c in tod_cos) / ctx.m_base
        _use_tod = var_s >= 0.001 or var_c >= 0.001

    for fi in ctx.active_input_indices:
        coeff_idx = fi + 2
        subset_indices = [k for k in range(ctx.m_base) if ctx.input_values_by_obs[k][fi] is not None]
        subset_values: list[float] = [ctx.input_values_by_obs[k][fi] for k in subset_indices]  # type: ignore[misc]  # filtered not-None via subset_indices
        m_sub = len(subset_indices)
        w_sub = [ctx.w_base[k] for k in subset_indices]
        y_sub = [residuals_base[k] for k in subset_indices]
        X_base_sub = [ctx.X_base[k] for k in subset_indices]

        # Partial out base + optional sin/cos: regress z on [1, od, sin, cos]
        n_partial = ctx.n_base + (2 if _use_tod else 0)
        XtWX_zb = [[0.0] * n_partial for _ in range(n_partial)]
        XtWy_zb = [0.0] * n_partial
        for i_sub in range(m_sub):
            k = subset_indices[i_sub]
            row: list[float] = list(X_base_sub[i_sub])
            if _use_tod:
                row.append(tod_sin[k])
                row.append(tod_cos[k])
            for a in range(n_partial):
                XtWy_zb[a] += row[a] * w_sub[i_sub] * subset_values[i_sub]
                for b in range(n_partial):
                    XtWX_zb[a][b] += row[a] * w_sub[i_sub] * row[b]
        for a in range(n_partial):
            XtWX_zb[a][a] += ctx.ridge
        gamma = _solve_symmetric(XtWX_zb, XtWy_zb, n_partial)

        if gamma is not None:
            r_z: list[float] = []
            for i in range(m_sub):
                k = subset_indices[i]
                row_p: list[float] = list(X_base_sub[i])
                if _use_tod:
                    row_p.append(tod_sin[k])
                    row_p.append(tod_cos[k])
                r_z.append(subset_values[i] - sum(gamma[a] * row_p[a] for a in range(n_partial)))
        else:
            r_z = list(subset_values)

        # Also partial out sin/cos from y_sub (FWL requires both sides)
        if _use_tod and gamma is not None:
            sin_sub = [tod_sin[k] for k in subset_indices]
            cos_sub = [tod_cos[k] for k in subset_indices]
            # Regress y_sub on [sin, cos] (no intercept — already residualized)
            XtWX_y = [[0.0, 0.0], [0.0, 0.0]]
            XtWy_y = [0.0, 0.0]
            for i_sub in range(m_sub):
                sc = [sin_sub[i_sub], cos_sub[i_sub]]
                for a in range(2):
                    XtWy_y[a] += sc[a] * w_sub[i_sub] * y_sub[i_sub]
                    for b in range(2):
                        XtWX_y[a][b] += sc[a] * w_sub[i_sub] * sc[b]
            XtWX_y[0][0] += 1e-8
            XtWX_y[1][1] += 1e-8
            gamma_y = _solve_symmetric(XtWX_y, XtWy_y, 2)
            if gamma_y is not None:  # pragma: no branch — 2x2 ridge-regularized system is always invertible
                y_sub = [y_sub[i] - gamma_y[0] * sin_sub[i] - gamma_y[1] * cos_sub[i] for i in range(m_sub)]

        if _weighted_variance(r_z, w_sub) < ctx.min_feature_variance:
            held.add(coeff_idx)
            continue

        wrzry = sum(w_sub[i] * r_z[i] * y_sub[i] for i in range(m_sub))
        wrzrz = sum(w_sub[i] * r_z[i] * r_z[i] for i in range(m_sub))
        if wrzrz < 1e-15:  # pragma: no cover — IEEE 754 cancellation guard
            held.add(coeff_idx)
            continue

        beta[coeff_idx] = wrzry / wrzrz
        sub_resid = [y_sub[i] - beta[coeff_idx] * r_z[i] for i in range(m_sub)]
        sub_rms_sq = sum(r * r for r in sub_resid) / max(1, m_sub - 1)
        classical_se = math.sqrt(sub_rms_sq / wrzrz) if wrzrz > 1e-15 else float("inf")

        # HAC sandwich for univariate FWL — replaces classical IID std_err
        # with autocorrelation-consistent variance.  Wall-clock-anchored
        # bandwidth keeps it cadence-invariant.
        sub_obs = [ctx.base_eligible[k] for k in subset_indices]
        dt_min_sub = _estimate_observation_dt_min(sub_obs)
        bw = _hac_bandwidth(m_sub, dt_min_sub)
        std_err[coeff_idx] = _hac_std_err_univariate(
            r_z, w_sub, sub_resid, wrzrz, bw, classical_se,
        )

    # Re-estimate base to absorb model input contributions
    y_adj = list(ctx.y_base)
    for k in range(ctx.m_base):
        for fi in ctx.active_input_indices:
            coeff_idx = fi + 2
            val = ctx.input_values_by_obs[k][fi]
            if coeff_idx not in held and val is not None:
                y_adj[k] -= beta[coeff_idx] * val
    XtWX_adj = [[0.0] * ctx.n_base for _ in range(ctx.n_base)]
    XtWy_adj = [0.0] * ctx.n_base
    for k in range(ctx.m_base):
        for i in range(ctx.n_base):
            xi = ctx.X_base[k][i] / ctx.col_scales_base[i]
            XtWy_adj[i] += xi * ctx.w_base[k] * y_adj[k]
            for j in range(ctx.n_base):
                xj = ctx.X_base[k][j] / ctx.col_scales_base[j]
                XtWX_adj[i][j] += xi * ctx.w_base[k] * xj
    for i in range(ctx.n_base):
        XtWX_adj[i][i] += ctx.ridge
    beta_base_adj = _solve_symmetric(XtWX_adj, XtWy_adj, ctx.n_base)
    if beta_base_adj is not None:  # pragma: no branch — 2x2 ridge-regularized system is always invertible
        beta[0] = beta_base_adj[0] / ctx.col_scales_base[0]
        beta[1] = beta_base_adj[1] / ctx.col_scales_base[1]

    # Base std errors from adjusted system
    cov_diag = _diagonal_of_inverse(XtWX_adj if beta_base_adj else ctx.XtWX_base, ctx.n_base)
    if cov_diag is not None:  # pragma: no branch — XtWX_adj is always invertible after ridge
        # Use sub-residual RMS as rough sigma estimate
        # Use sub-residual RMS as rough sigma estimate
        rms_base = math.sqrt(sum(r * r for r in residuals_base) / max(1, ctx.m_base - ctx.n_base))
        rms_sq = rms_base * rms_base if rms_base > 0 else 1e-12
        for i in range(ctx.n_base):
            var_i = rms_sq * max(0.0, cov_diag[i]) / (ctx.col_scales_base[i] ** 2)
            std_err[i] = math.sqrt(var_i) if var_i > 0 else 0.0

    return beta, std_err


def weighted_least_squares(
    observations: list[Observation],
    n_features: int,
    current_beta: list[float] | None = None,
    room_rate_threshold: float = 0.02,
    min_observations: int = 20,
    min_feature_variance: float = MIN_FEATURE_VARIANCE,
    outlier_sigma: float = 3.0,
    min_feature_representation: int = 10,
    feature_order: list[str] | None = None,
    model_inputs: list[dict[str, Any]] | None = None,
    frozen_features: set[int] | None = None,
    detect_lag: bool = True,
) -> BatchResult | None:
    """Run weighted least squares on physical observations.

    Orchestrates three phases:
    1. **Filter & classify** — exclude ineligible observations, classify
       features as active/held, partition data by completeness.
    2. **Solve** — joint OLS on complete data (exact), or FWL on partial
       data (unbiased per-feature).  Grey-box solvers plug in here.
    3. **Package** — compute residuals, exclude outliers, return BatchResult.

    When ``detect_lag`` is True, auto-detects optimal EMA tau per model
    input by minimizing base-regression residuals.  Detected tau values
    are stored in ``BatchResult.detected_tau`` and used to pre-filter
    inputs for the main regression.

    Returns None if insufficient eligible observations.
    """
    # ── Phase 1: Filter & classify ──────────────────────────────────
    eligible = [
        o for o in observations
        if not o.clamped
        and o.hp_setpoint is not None
        and abs(o.room_rate) < room_rate_threshold
        and not o.hp_contribution_uncertain
    ]

    if len(eligible) < min_observations:
        return None

    n = n_features
    if n < 2:
        return None
    m_inputs = model_inputs or []

    base_eligible = [o for o in eligible if o.outdoor_temp_c is not None]
    if len(base_eligible) < min_observations:
        return None

    m_base = len(base_eligible)
    y_base: list[float] = []
    for o in base_eligible:
        assert o.hp_setpoint is not None
        y_base.append(o.hp_setpoint - o.current_c)
    w_base = [1.0 / (1.0 + (o.room_rate / room_rate_threshold) ** 2) for o in base_eligible]
    X_base: list[list[float]] = [[1.0, o.outdoor_temp_c - o.desired_c] for o in base_eligible]  # type: ignore[operator]  # filtered not-None above

    # Scale outdoor_delta column
    n_base = 2
    col_scales_base = [1.0, 1.0]
    col_od = [X_base[k][1] for k in range(m_base)]
    mean_od = sum(col_od) / m_base
    var_od = sum((c - mean_od) ** 2 for c in col_od) / m_base
    col_scales_base[1] = math.sqrt(var_od) if var_od > 1e-12 else 1.0

    ridge = 1e-6
    XtWX_base = [[0.0] * n_base for _ in range(n_base)]
    XtWy_base = [0.0] * n_base
    for k in range(m_base):
        for i in range(n_base):
            xi = X_base[k][i] / col_scales_base[i]
            XtWy_base[i] += xi * w_base[k] * y_base[k]
            for j in range(n_base):
                xj = X_base[k][j] / col_scales_base[j]
                XtWX_base[i][j] += xi * w_base[k] * xj
    for i in range(n_base):
        XtWX_base[i][i] += ridge

    beta_base_norm = _solve_symmetric(XtWX_base, XtWy_base, n_base)
    if beta_base_norm is None:
        return None
    beta_base = [beta_base_norm[i] / col_scales_base[i] for i in range(n_base)]

    # ── Auto lag-tau detection ──────────────────────────────────────
    # For each model input, detect optimal EMA tau by minimizing
    # base-regression residuals.  Pre-compute filtered values using
    # the detected tau for use in the main regression.
    detected_tau: dict[str, float] = {}
    detected_tau_diagnostics: dict[str, LagTauDiagnostic] = {}
    # Per-entity filtered values: entity_id → list[float|None] parallel to base_eligible
    _filtered_cache: dict[str, list[float | None]] = {}

    if detect_lag and m_inputs:
        # Auto lag-tau via joint regression sweep.  For each model input,
        # build X = [1, od, sin, cos, EMA(input, τ)] at candidate τ values
        # and pick the τ with lowest weighted RSS.  This is the profile
        # likelihood approach — the full joint regression naturally handles
        # diurnal correlation without needing separate FWL partialling.
        #
        # Uses numpy when available (np.linalg.lstsq is ~100× faster than
        # pure-Python normal equations for the ~200×5 matrices involved).
        _tod_cols: list[tuple[float, float]] = [
            tod_features(base_eligible[k].wall_time) for k in range(m_base)
        ]

        for m_input in m_inputs:
            entity_id = m_input.get("entity_id", "")
            name = m_input.get("name", entity_id)
            dfr = bool(m_input.get("delta_from_room"))
            if not entity_id:
                continue

            tau_result = _detect_optimal_tau(
                base_eligible, y_base, w_base,
                entity_id=entity_id,
                delta_from_room=dfr,
                base_X=X_base,
                tod_cols=_tod_cols,
                input_role=m_input.get("input_role"),
            )

            if tau_result is not None:
                detected_tau[name] = tau_result.tau
                detected_tau_diagnostics[name] = tau_result
                if tau_result.tau > 0:
                    _LOGGER.info(
                        "Lag-tau detection: %s τ=%.0fs (%.0f min), "
                        "R²_improvement=%.3f, β=%.3f, BIC gain=%.2f (>%.2f)",
                        name, tau_result.tau, tau_result.tau / 60,
                        tau_result.r2_improvement, tau_result.beta_at_tau,
                        tau_result.bic_gain, tau_result.bic_threshold,
                    )
                # Cache filtered values for this entity
                _filtered_cache[entity_id] = _apply_retrospective_ema(
                    base_eligible, entity_id, tau_result.tau,
                )

    # Classify model input features
    input_entity_ids = [m.get("entity_id", "") for m in m_inputs]
    input_values_by_obs: list[list[float | None]] = []
    for k, o in enumerate(base_eligible):
        row: list[float | None] = []
        for feat_idx, m_input in enumerate(m_inputs):
            entity_id = input_entity_ids[feat_idx]
            if entity_id and entity_id in o.raw_readings:
                # Use filtered value if available, else raw
                value: float | None
                if entity_id in _filtered_cache and _filtered_cache[entity_id][k] is not None:
                    value = _filtered_cache[entity_id][k]
                else:
                    value = o.raw_readings[entity_id]
                if value is not None and m_input.get("delta_from_room"):
                    value = value - o.current_c
                row.append(value)
            else:
                row.append(None)
        input_values_by_obs.append(row)

    feature_obs_counts: dict[int, int] = {}
    for feat_idx in range(len(m_inputs)):
        feature_obs_counts[feat_idx + 2] = sum(
            1 for row in input_values_by_obs if row[feat_idx] is not None
        )

    held: set[int] = set()
    # Frozen features (from per-feature gating) are treated as held so
    # batch WLS matches the online RLS partial model — both estimators
    # see the same features, preventing batch-online oscillation.
    _frozen = frozen_features or set()
    active_input_indices: list[int] = []
    for feat_idx, m_input in enumerate(m_inputs):
        coeff_idx = feat_idx + 2
        if coeff_idx in _frozen:
            held.add(coeff_idx)
            continue
        entity_id = input_entity_ids[feat_idx]
        if not entity_id:
            held.add(coeff_idx)
            continue
        if feature_obs_counts[coeff_idx] < max(min_observations, 10):
            held.add(coeff_idx)
            continue
        vals: list[float] = [input_values_by_obs[k][feat_idx]  # type: ignore[misc]  # filtered not-None
                for k in range(m_base) if input_values_by_obs[k][feat_idx] is not None]
        w_vals = [w_base[k] for k in range(m_base) if input_values_by_obs[k][feat_idx] is not None]
        if _weighted_variance(vals, w_vals) < min_feature_variance:
            held.add(coeff_idx)
            continue
        active_input_indices.append(feat_idx)

    complete_indices = [
        k for k in range(m_base)
        if all(input_values_by_obs[k][fi] is not None for fi in active_input_indices)
    ]

    ctx = _RegressionContext(
        n=n, n_base=n_base, m_base=m_base,
        base_eligible=base_eligible, y_base=y_base, w_base=w_base,
        X_base=X_base, col_scales_base=col_scales_base,
        XtWX_base=XtWX_base, beta_base=beta_base, ridge=ridge,
        m_inputs=m_inputs, input_entity_ids=input_entity_ids,
        input_values_by_obs=input_values_by_obs,
        feature_obs_counts=feature_obs_counts,
        active_input_indices=active_input_indices, held=held,
        complete_indices=complete_indices,
        min_feature_variance=min_feature_variance,
    )

    # ── Phase 2: Solve ──────────────────────────────────────────────
    n_active = len(active_input_indices)
    if n_active > 0 and len(complete_indices) >= min_observations:
        result = _solve_joint(ctx)
    else:
        result = None

    if result is not None:
        beta, std_err = result
    elif n_active > 0:
        beta, std_err = _solve_fwl(ctx)
    else:
        beta = [0.0] * n
        beta[0] = beta_base[0]
        beta[1] = beta_base[1]
        std_err = [float("inf")] * n
        cov_diag = _diagonal_of_inverse(XtWX_base, n_base)
        if cov_diag is not None:  # pragma: no branch — XtWX_base ridge-regularized is always invertible
            resid_base = [
                y_base[k] - sum(beta_base[i] * X_base[k][i] for i in range(n_base))
                for k in range(m_base)
            ]
            rms_base = math.sqrt(
                sum(r * r for r in resid_base) / max(1, m_base - n_base)
            )
            rms_sq = rms_base * rms_base if rms_base > 0 else 1e-12

            classical_norm = [0.0] * n_base
            for i in range(n_base):
                v = rms_sq * max(0.0, cov_diag[i])
                classical_norm[i] = math.sqrt(v) if v > 0 else 0.0

            # HAC sandwich on the base-only design matrix.
            dt_min = _estimate_observation_dt_min(base_eligible)
            bandwidth = _hac_bandwidth(m_base, dt_min)
            XtWX_inv = DiversityAwareBuffer._invert_matrix(
                [row[:] for row in XtWX_base], n_base,
            )
            if XtWX_inv is not None:
                X_norm_rows = [
                    [X_base[k][i] / col_scales_base[i] for i in range(n_base)]
                    for k in range(m_base)
                ]
                meat = _hac_meat_matrix(X_norm_rows, w_base, resid_base, bandwidth)
                hac_norm = _hac_std_err_from_sandwich(
                    XtWX_inv, meat,
                    col_scales=[1.0] * n_base,
                    classical_std_err=classical_norm,
                )
            else:  # pragma: no cover — XtWX_base inversion mirrors cov_diag presence
                hac_norm = classical_norm

            for i in range(n_base):
                std_err[i] = (
                    hac_norm[i] / col_scales_base[i] if col_scales_base[i] > 0 else 0.0
                )

    # Fill held features from current model
    fallback = current_beta if current_beta else [0.0] * n
    for j in held:
        beta[j] = fallback[j] if j < len(fallback) else 0.0

    # ── Phase 3: Package (residuals, outlier exclusion) ─────────────
    if feature_order and m_inputs:
        full_obs: list[Observation] = []
        full_X: list[list[float]] = []
        full_y: list[float] = []
        for k, o in enumerate(base_eligible):
            # Build per-observation filtered overrides from cache
            f_overrides: dict[str, float] | None = None
            if _filtered_cache:
                fo: dict[str, float] = {}
                for eid, fvals in _filtered_cache.items():
                    if k < len(fvals) and fvals[k] is not None:
                        fo[eid] = fvals[k]  # type: ignore[assignment]
                if fo:
                    f_overrides = fo
            vec = build_feature_vector_from_raw(
                o, m_inputs, feature_order, filtered_overrides=f_overrides,
            )
            if vec is not None:
                assert o.hp_setpoint is not None
                full_obs.append(o)
                full_X.append(vec)
                full_y.append(o.hp_setpoint - o.current_c)
        m_full = len(full_obs)
    else:
        full_obs = list(base_eligible)
        full_X = X_base
        full_y = y_base
        m_full = m_base

    residuals = [
        full_y[k] - sum(beta[i] * (full_X[k][i] if i < len(full_X[k]) else 0.0) for i in range(n))
        for k in range(m_full)
    ]
    rms = math.sqrt(sum(r * r for r in residuals) / m_full) if m_full > 0 else 0.0

    # Outlier exclusion with rare-feature protection
    n_excluded = 0
    if outlier_sigma > 0 and rms > 0 and m_full > min_observations + 5:
        threshold_val = outlier_sigma * rms
        keep = []
        for k in range(m_full):
            if abs(residuals[k]) <= threshold_val:
                keep.append(k)
            else:
                has_rare = False
                for feat_idx, m_input in enumerate(m_inputs):
                    entity_id = m_input.get("entity_id", "")
                    if entity_id and entity_id in full_obs[k].raw_readings:  # pragma: no branch — short-circuit on empty entity_id / missing reading
                        val = full_obs[k].raw_readings[entity_id]
                        if abs(val) > 1e-6 and feature_obs_counts.get(feat_idx + 2, 0) < min_feature_representation:
                            has_rare = True
                            break
                if has_rare:
                    keep.append(k)
                else:
                    n_excluded += 1

        if n_excluded > 0 and len(keep) >= min_observations:
            residuals = [residuals[k] for k in keep]
            rms = math.sqrt(sum(r * r for r in residuals) / len(keep)) if keep else 0.0
            m_full = len(keep)
            _LOGGER.debug(
                "Residual filter: excluded %d observations (threshold=%.3f)",
                n_excluded, threshold_val,
            )

    # ── VIF from regression data ──────────────────────────────────────
    # Compute per-feature VIF from the eligible-only feature matrix.
    # This uses the same observations the regression used, not the full
    # buffer (which may include passive/clamped observations with different
    # correlation structure).
    vif = _compute_vif_from_features(full_X, n, m_full)

    return BatchResult(
        n_total=len(observations),
        n_eligible=m_full,
        n_outliers_excluded=n_excluded,
        beta_batch=beta,
        beta_current=[],
        residual_rms=rms,
        max_coeff_change_pct=0.0,
        recommend_update=False,
        held_features=held,
        beta_std_err=std_err,
        feature_vif=vif,
        detected_tau=detected_tau,
        detected_tau_diagnostics=detected_tau_diagnostics,
    )


def _compute_vif_from_features(
    X: list[list[float]], n_features: int, n_obs: int,
) -> list[float]:
    """Compute per-feature VIF from a feature matrix, excluding intercept.

    VIF_j = diag((corr)⁻¹)_j where corr is the correlation matrix of
    features 1..n-1 (excluding the constant intercept at index 0).
    Uses the same eligible-only data the regression was fitted on.

    Returns [VIF_0, ..., VIF_{n-1}] where VIF_0 = 1.0 (intercept).
    All inf if matrix is singular.
    """
    n = n_features
    if n_obs < 2 or n < 3:
        return [1.0] * n

    # Build X'X for features 1..n-1 (exclude intercept at index 0)
    m = n - 1
    xtx = [[0.0] * m for _ in range(m)]
    for k in range(n_obs):
        row = X[k]
        for i in range(m):
            ri = i + 1  # skip intercept
            if ri >= len(row):
                continue
            for j in range(m):
                rj = j + 1
                if rj >= len(row):
                    continue
                xtx[i][j] += row[ri] * row[rj]

    # Column-normalize to correlation matrix
    diag_sqrt = [
        math.sqrt(xtx[i][i]) if xtx[i][i] > 1e-15 else 1.0
        for i in range(m)
    ]
    corr = [
        [xtx[i][j] / (diag_sqrt[i] * diag_sqrt[j]) for j in range(m)]
        for i in range(m)
    ]

    # Invert correlation matrix
    corr_inv = DiversityAwareBuffer._invert_matrix(corr, m)
    if corr_inv is None:
        return [float('inf')] * n

    # Build result: intercept VIF = 1.0, rest from corr_inv diagonal
    result = [1.0]  # index 0 = intercept
    for i in range(m):
        result.append(max(corr_inv[i][i], 1.0))
    return result


@dataclass
class CollinearGroup:
    """A group of features sharing an ill-conditioned component.

    Belsley (1980): a collinearity problem exists when a component with
    condition index ≥ 30 has variance decomposition proportion ≥ 0.5 for
    two or more features simultaneously.
    """

    condition_index: float
    features: list[str]  # names of involved features
    feature_indices: list[int]  # indices into the feature vector
    proportions: list[float]  # VDP values for each involved feature


def compute_belsley_diagnostics(
    X: list[list[float]],
    n_features: int,
    n_obs: int,
    feature_names: list[str] | None = None,
    ci_threshold: float = 30.0,
    vdp_threshold: float = 0.5,
) -> list[CollinearGroup]:
    """Compute Belsley (1980) collinearity diagnostics via SVD.

    Identifies groups of features that share ill-conditioned components
    in the design matrix — the proper per-variable collinearity diagnostic
    that tells you *which* features are confounded with *which*.

    A feature is degraded by collinearity only when:
    1. A condition index is ≥ ci_threshold (default 30), AND
    2. That same component has VDP ≥ vdp_threshold for 2+ features

    Requires numpy.  Returns empty list if numpy is unavailable or
    if no collinearity groups are detected.

    Args:
        X: Feature matrix (m × n), including intercept at column 0.
        n_features: Number of features (columns in X).
        n_obs: Number of observations (rows in X).
        feature_names: Human-readable names, one per feature.
        ci_threshold: Condition index threshold (Belsley: 30).
        vdp_threshold: Variance decomposition proportion threshold (Belsley: 0.5).

    Returns:
        List of CollinearGroup, one per ill-conditioned component that
        involves 2+ features.  Empty if no problems detected.
    """
    if not _NUMPY_AVAILABLE:
        return []
    if n_obs < n_features or n_features < 2:
        return []

    names = feature_names or [f"feature_{i}" for i in range(n_features)]

    X_np = np.array(X[:n_obs])
    if X_np.shape != (n_obs, n_features):
        X_np = X_np[:, :n_features]

    # Column-normalize to unit length (Belsley 1991 §3.3 recommendation).
    # Do NOT center columns — centering obscures intercept dependencies.
    norms = np.linalg.norm(X_np, axis=0)
    norms[norms < 1e-15] = 1.0
    X_scaled = X_np / norms

    # SVD: X = U Σ Vᵀ
    try:
        _U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    except np.linalg.LinAlgError:
        return []

    V = Vt.T  # (n, n) — columns are right singular vectors

    # Condition indices: CI_j = σ_max / σ_j
    if s[-1] < 1e-15:
        # Singular matrix — can't decompose
        return []
    cond_indices = s[0] / s

    # Variance decomposition proportions.
    # φ[k, j] = V[k,j]² / σ_j²  (unnormalized contribution of component j
    #                               to Var(β_k))
    # VDP[j, k] = φ[k,j] / Σ_j φ[k,j]  (proportion, summing to 1 over j
    #                                      for each feature k)
    phi = V ** 2 / (s ** 2)  # (n_features, n_components)
    row_sums = phi.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-15] = 1.0
    vdp = (phi / row_sums).T  # (n_components, n_features)

    # Identify collinear groups: components where CI ≥ threshold AND
    # 2+ features have VDP ≥ vdp_threshold.
    groups: list[CollinearGroup] = []
    for j in range(len(cond_indices)):
        if cond_indices[j] < ci_threshold:
            continue
        involved = []
        for k in range(n_features):
            if vdp[j, k] >= vdp_threshold:
                involved.append(k)
        if len(involved) >= 2:
            groups.append(CollinearGroup(
                condition_index=float(cond_indices[j]),
                features=[names[k] for k in involved],
                feature_indices=involved,
                proportions=[float(vdp[j, k]) for k in involved],
            ))

    return groups


def _diagonal_of_inverse(A: list[list[float]], n: int) -> list[float] | None:
    """Compute diagonal of A⁻¹ by solving A x = eᵢ for each column.

    Returns [diag(A⁻¹)₀, ..., diag(A⁻¹)ₙ₋₁] or None if singular.
    Only needs the diagonal, but for small n (3-8) solving n systems
    is negligible.
    """
    diag = []
    for i in range(n):
        e_i = [1.0 if j == i else 0.0 for j in range(n)]
        col = _solve_symmetric(A, e_i, n)
        if col is None:
            return None
        diag.append(col[i])
    return diag


def _solve_symmetric(A: list[list[float]], b: list[float], n: int) -> list[float] | None:
    """Solve Ax = b for symmetric positive-definite A via Gaussian elimination.

    Simple implementation for small n (typically 3-8). Returns None if
    the system is singular.
    """
    # Copy to avoid mutation
    M = [row[:] + [bi] for row, bi in zip(A, b)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row
        if max_val < 1e-12:
            return None
        M[col], M[max_row] = M[max_row], M[col]

        # Eliminate below
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if abs(M[i][i]) < 1e-12:  # pragma: no cover
            # Currently unreachable: forward elimination only modifies rows
            # below the pivot, so M[i][i] is the same value that passed the
            # > 1e-12 pivot check. Retained as a guard for future changes to
            # the elimination (e.g. pivot-row scaling, threshold changes).
            return None
        x[i] /= M[i][i]

    return x


def compare_and_report(
    result: BatchResult,
    current_beta_physical: list[float],
    coeff_names: list[str] | None = None,
    change_threshold_pct: float = 20.0,
    min_observations: int = 20,
    log_prefix: str = "",
) -> BatchResult:
    """Compare batch WLS result with current RLS and log findings.

    Populates result.beta_current, result.max_coeff_change_pct, and
    result.recommend_update.  Held features (insufficient variance) are
    logged but excluded from max_change and update recommendation.
    """
    result.beta_current = list(current_beta_physical)
    n = len(result.beta_batch)
    names = coeff_names or [f"β{i}" for i in range(n)]
    held = result.held_features

    max_change = 0.0
    changes: list[tuple[str, float, float, float, bool]] = []
    # When |current| is below this floor (post-reset / near-seed), percent
    # change is meaningless — cap at 100% so the threshold gate still works.
    _NEAR_ZERO_FLOOR = 0.1
    for i in range(min(n, len(current_beta_physical))):
        current = current_beta_physical[i]
        batch = result.beta_batch[i]
        is_held = i in held
        if is_held:
            pct = 0.0
        elif abs(current) > _NEAR_ZERO_FLOOR:
            pct = abs(batch - current) / abs(current) * 100
        elif abs(batch - current) > 1e-6:
            pct = 100.0
        else:
            pct = 0.0
        if not is_held:
            max_change = max(max_change, pct)
        name = names[i] if i < len(names) else f"β{i}"
        changes.append((name, current, batch, pct, is_held))

    result.max_coeff_change_pct = max_change
    result.recommend_update = (
        max_change > change_threshold_pct
        and result.n_eligible >= min_observations
    )

    # Log the analysis
    n_held = len(held)
    n_outliers = result.n_outliers_excluded
    _LOGGER.info(
        "%sBatch WLS analysis: %d/%d observations eligible, RMS=%.3f%s%s",
        log_prefix, result.n_eligible, result.n_total, result.residual_rms,
        f", {n_held} feature{'s' if n_held != 1 else ''} held (no variance)"
        if n_held else "",
        f", {n_outliers} outlier{'s' if n_outliers != 1 else ''} excluded"
        if n_outliers else "",
    )
    for name, current, batch, pct, is_held in changes:
        if is_held:
            _LOGGER.info(
                "%s  %s: held at %.4f (insufficient variance)",
                log_prefix, name, current,
            )
        else:
            marker = " ***" if pct > change_threshold_pct else ""
            _LOGGER.info(
                "%s  %s: current=%.4f batch=%.4f (%.1f%% change)%s",
                log_prefix, name, current, batch, pct, marker,
            )
    if result.recommend_update:
        _LOGGER.warning(
            "%sBatch WLS recommends model update (max change %.0f%%, %d observations)",
            log_prefix, max_change, result.n_eligible,
        )
    else:
        _LOGGER.info(
            "%sBatch WLS: model is consistent (max change %.0f%%)",
            log_prefix, max_change,
        )

    return result


# ── Covariance-weighted blended update ───────────────────────────────

# Maximum per-coefficient change (absolute) per batch cycle.  Prevents
# a single noisy batch from making a large destructive step even if the
# covariance says it's confident.
MAX_STEP_ABS = 1.0

# Prior standard deviation for the current model's coefficients.
# Represents "moderate confidence" in the online RLS estimate.
# When the batch std_err is much smaller than this, the batch dominates;
# when batch std_err is comparable or larger, the current model holds.
# This is a scalar prior — a more precise version would use the RLS
# covariance diagonal, but that requires plumbing it through the caller.
DEFAULT_PRIOR_STD = 1.0


def compute_blended_update(
    result: BatchResult,
    prior_std: float = DEFAULT_PRIOR_STD,
    max_step: float = MAX_STEP_ABS,
    max_step_per_feature: list[float] | None = None,
) -> BatchResult:
    """Compute a safe blended update via covariance-weighted fusion.

    Per-coefficient Kalman-style gain (Ljung §11.4):
        K_i = σ²_prior / (σ²_prior + σ²_batch_i)

    When σ²_batch is small (tight estimate), K → 1 and the batch pulls
    the coefficient strongly.  When σ²_batch is large (uncertain), K → 0
    and the current value holds.  Held features (σ²_batch = ∞) always
    get K = 0.

    Safety net: per-coefficient step cap of ±max_step regardless of gain.
    If *max_step_per_feature* is provided, each coefficient uses its own
    cap (e.g. enlarged for recently-unfrozen features).

    Populates result.beta_blended and result.blend_gains.
    """
    current = result.beta_current
    batch = result.beta_batch
    std_err = result.beta_std_err
    n = min(len(current), len(batch))

    prior_var = prior_std * prior_std
    gains = [0.0] * n
    blended = list(current[:n])

    for i in range(n):
        se = std_err[i] if i < len(std_err) else float("inf")
        batch_var = se * se
        if math.isinf(batch_var) or (prior_var + batch_var) < 1e-12:
            gains[i] = 0.0
        else:
            gains[i] = prior_var / (prior_var + batch_var)

        step_cap = (
            max_step_per_feature[i]
            if max_step_per_feature is not None and i < len(max_step_per_feature)
            else max_step
        )
        delta = gains[i] * (batch[i] - current[i])
        if abs(delta) > step_cap:
            delta = step_cap if delta > 0 else -step_cap
        blended[i] = current[i] + delta

    result.beta_blended = blended
    result.blend_gains = gains

    _LOGGER.info(
        "Batch blend: prior_std=%.2f, max_step=%.1f%s",
        prior_std, max_step,
        f", per-feature caps active" if max_step_per_feature else "",
    )
    for i in range(n):
        se = std_err[i] if i < len(std_err) else float("inf")
        step_cap = (
            max_step_per_feature[i]
            if max_step_per_feature is not None and i < len(max_step_per_feature)
            else max_step
        )
        if math.isinf(se):
            _LOGGER.info(
                "  β%d: held (no batch uncertainty estimate)",
                i,
            )
        elif current[i] != blended[i]:
            cap_note = f", cap={step_cap:.1f}" if step_cap != max_step else ""
            _LOGGER.info(
                "  β%d: %.4f → %.4f (K=%.3f, σ_batch=%.4f, Δ=%.4f%s)",
                i, current[i], blended[i], gains[i], se,
                blended[i] - current[i], cap_note,
            )
        else:
            _LOGGER.debug(
                "  β%d: %.4f (unchanged, K=%.3f, σ_batch=%.4f)",
                i, current[i], gains[i], se,
            )

    return result


def fuse_batch_greybox(
    result: BatchResult,
    greybox_beta: list[float | None],
    greybox_std_err: list[float],
) -> None:
    """Fuse WLS and grey-box β estimates via inverse-variance weighting.

    For each coefficient where grey-box provides an estimate (not None)
    and has finite std_err, combines the two independent estimates:

        1/σ²_fused = 1/σ²_wls + 1/σ²_gb
        β_fused = σ²_fused × (β_wls/σ²_wls + β_gb/σ²_gb)

    Modifies result.beta_batch and result.beta_std_err in-place so
    the downstream compute_blended_update() sees the fused estimate.

    When grey-box estimate is None or std_err is infinite, the WLS
    estimate passes through unchanged.
    """
    n = min(len(result.beta_batch), len(greybox_beta))
    fused_count = 0
    for i in range(n):
        gb = greybox_beta[i]
        gb_se = greybox_std_err[i] if i < len(greybox_std_err) else float("inf")
        wls_se = result.beta_std_err[i] if i < len(result.beta_std_err) else float("inf")

        if gb is None or math.isinf(gb_se) or math.isinf(wls_se):
            continue

        gb_var = gb_se * gb_se
        wls_var = wls_se * wls_se

        if gb_var < 1e-12 or wls_var < 1e-12:
            continue

        # Inverse-variance weighting
        fused_prec = 1.0 / wls_var + 1.0 / gb_var
        fused_var = 1.0 / fused_prec
        fused_beta = fused_var * (
            result.beta_batch[i] / wls_var + gb / gb_var
        )
        fused_se = math.sqrt(fused_var)

        _LOGGER.info(
            "  β%d fuse: WLS=%.4f (σ=%.4f) + GB=%.4f (σ=%.4f) → %.4f (σ=%.4f)",
            i, result.beta_batch[i], wls_se, gb, gb_se, fused_beta, fused_se,
        )

        result.beta_batch[i] = fused_beta
        result.beta_std_err[i] = fused_se
        fused_count += 1

    if fused_count > 0:
        _LOGGER.info(
            "Fused %d coefficient(s) from WLS + grey-box",
            fused_count,
        )


# ─��� Residual time-of-day analysis ─────���────────────────────────────


@dataclass
class HourlyResidualPattern:
    """A detected time-of-day residual pattern."""

    start_hour: int  # inclusive
    end_hour: int  # inclusive
    mean_residual: float  # signed mean residual (°C)
    n_observations: int  # total obs in the span


def analyze_residuals_by_hour(
    observations: list[Observation],
    beta: list[float],
    n_features: int,
    room_rate_threshold: float = 0.02,
    min_obs_per_hour: int = 5,
    residual_threshold: float = 0.5,
    feature_order: list[str] | None = None,
    model_inputs: list[dict[str, Any]] | None = None,
) -> list[HourlyResidualPattern]:
    """Detect systematic time-of-day residual patterns.

    Computes residual = (hp_setpoint - current_c) - Σ βᵢ xᵢ for each
    eligible observation, bins by wall-clock hour, and finds contiguous
    hour spans where the mean residual consistently exceeds the threshold.
    A positive residual means the HP needed more offset than the model
    predicted (unmodeled heat loss); negative means less (unmodeled gain).

    Args:
        observations: all observations (filtered internally).
        beta: current model coefficients.
        n_features: number of features.
        room_rate_threshold: max |room_rate| for eligibility.
        min_obs_per_hour: minimum observations per hour bucket.
        residual_threshold: minimum |mean residual| to flag (°C).
        model_inputs: current model input config for feature building.

    Returns:
        List of detected patterns (contiguous hour spans with consistent
        bias). Empty if no patterns exceed threshold.
    """
    from homeassistant.util import dt as dt_util

    # Bin residuals by UTC hour — same convention as model_input_manager.tod_features.
    hour_residuals: dict[int, list[float]] = {h: [] for h in range(24)}

    for o in observations:
        if o.clamped or o.hp_setpoint is None:
            continue
        if abs(o.room_rate) >= room_rate_threshold:
            continue
        # Derive UTC hour from wall_time (epoch → UTC hour) — see
        # model_input_manager.tod_features for the architectural rationale.
        wall_hour = dt_util.utc_from_timestamp(o.wall_time).hour if o.wall_time > 0 else -1
        if wall_hour < 0:
            continue
        # Build feature vector from raw readings + current config
        if feature_order is None or model_inputs is None:
            continue
        x = build_feature_vector_from_raw(o, model_inputs, feature_order)
        if x is None:
            continue
        n_use = min(n_features, len(beta), len(x))
        predicted = sum(beta[i] * x[i] for i in range(n_use))
        actual = o.hp_setpoint - o.current_c
        residual = actual - predicted
        hour_residuals[wall_hour].append(residual)

    # Compute per-hour means
    hour_means: dict[int, float] = {}
    hour_counts: dict[int, int] = {}
    for h in range(24):
        vals = hour_residuals[h]
        hour_counts[h] = len(vals)
        if len(vals) >= min_obs_per_hour:
            hour_means[h] = sum(vals) / len(vals)
        else:
            hour_means[h] = 0.0  # insufficient data, treat as neutral

    # Find contiguous spans where mean residual exceeds threshold
    # with consistent sign
    patterns: list[HourlyResidualPattern] = []
    visited: set[int] = set()

    for start in range(24):
        if start in visited:
            continue
        if hour_counts[start] < min_obs_per_hour:
            continue
        if abs(hour_means[start]) < residual_threshold:
            continue

        sign = 1 if hour_means[start] > 0 else -1
        end = start
        total_residual = 0.0
        total_obs = 0

        # Extend the span forward (wrapping at 24)
        for offset in range(24):  # pragma: no branch — hour loop always completes the 0..24 range
            h = (start + offset) % 24
            if hour_counts[h] < min_obs_per_hour:
                break
            if abs(hour_means[h]) < residual_threshold:
                break
            h_sign = 1 if hour_means[h] > 0 else -1
            if h_sign != sign:
                break
            end = h
            visited.add(h)
            total_residual += sum(hour_residuals[h])
            total_obs += hour_counts[h]

        if total_obs > 0:  # pragma: no branch — total_obs > 0 when residuals were collected
            patterns.append(HourlyResidualPattern(
                start_hour=start,
                end_hour=end,
                mean_residual=total_residual / total_obs,
                n_observations=total_obs,
            ))

    return patterns
