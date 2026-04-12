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

_LOGGER = logging.getLogger(__name__)

# Maximum observations to retain (48h at ~15-min ticks ≈ 192)
MAX_BUFFER_SIZE = 300


@dataclass
class Observation:
    """A single observation for batch learning and debug analysis."""

    timestamp: float  # monotonic time
    features: list[float]  # [1, outdoor_delta, model_input_1, ...]
    hp_setpoint: float  # integer HP setpoint (°C)
    current_c: float  # filtered room temperature (°C)
    desired_c: float  # target temperature (°C)
    room_rate: float  # dT/dt in °C/min at observation time
    clamped: bool  # True if HP setpoint was at min or max
    # Debug fields (not used by WLS, but needed for diagnostics)
    pi_integral: float = 0.0
    ff_offset: float = 0.0
    ff_confidence: float = 1.0
    raw_c: float = 0.0  # unfiltered room temperature

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": self.timestamp,
            "x": self.features,
            "sp": self.hp_setpoint,
            "cur": self.current_c,
            "des": self.desired_c,
            "rate": self.room_rate,
            "clamp": self.clamped,
            "integ": self.pi_integral,
            "ff": self.ff_offset,
            "ffc": self.ff_confidence,
            "raw": self.raw_c,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Observation:
        return cls(
            timestamp=d["t"],
            features=d["x"],
            hp_setpoint=d["sp"],
            current_c=d["cur"],
            desired_c=d["des"],
            room_rate=d["rate"],
            clamped=d["clamp"],
            pi_integral=d.get("integ", 0.0),
            ff_offset=d.get("ff", 0.0),
            ff_confidence=d.get("ffc", 1.0),
            raw_c=d.get("raw", d["cur"]),
        )


class ObservationBuffer:
    """Ring buffer of observations for batch learning.

    Stores every PI tick's state regardless of learning gate, so batch
    analysis has a richer dataset than online-only learning.  Capped at
    MAX_BUFFER_SIZE entries (~48h).
    """

    def __init__(self, max_size: int = MAX_BUFFER_SIZE) -> None:
        self._buffer: list[Observation] = []
        self._max_size = max_size

    def add(self, obs: Observation) -> None:
        """Add an observation, evicting oldest if full."""
        self._buffer.append(obs)
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)

    def get_all(self) -> list[Observation]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def as_list(self) -> list[dict[str, Any]]:
        return [o.as_dict() for o in self._buffer]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]], max_size: int = MAX_BUFFER_SIZE) -> ObservationBuffer:
        buf = cls(max_size)
        for d in data[-max_size:]:  # keep only most recent
            try:
                buf._buffer.append(Observation.from_dict(d))
            except (KeyError, TypeError, ValueError):
                continue
        return buf


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
    held_features: set[int] = field(default_factory=set)  # indices held at current (insufficient variance)
    beta_std_err: list[float] = field(default_factory=list)  # per-coefficient standard error from WLS
    beta_blended: list[float] = field(default_factory=list)  # safe update after covariance-weighted blend
    blend_gains: list[float] = field(default_factory=list)  # per-coefficient Kalman gain K_i ∈ [0, 1]


def _weighted_variance(values: list[float], weights: list[float]) -> float:
    """Compute weighted variance of a feature column."""
    w_sum = sum(weights)
    if w_sum < 1e-12:
        return 0.0
    mean = sum(v * w for v, w in zip(values, weights)) / w_sum
    return sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / w_sum


# Minimum weighted variance for a feature to be considered identifiable.
# Binary features (0/1) that toggled in ~10% of observations have variance
# ~0.09; this threshold is well below that.
MIN_FEATURE_VARIANCE = 1e-4


def weighted_least_squares(
    observations: list[Observation],
    n_features: int,
    current_beta: list[float] | None = None,
    room_rate_threshold: float = 0.02,
    min_observations: int = 20,
    min_feature_variance: float = MIN_FEATURE_VARIANCE,
) -> BatchResult | None:
    """Run weighted least squares on filtered observations.

    Filters:
    - Excludes clamped observations (censored data)
    - Excludes observations with room_rate > threshold (not at equilibrium)
    - Per-feature persistent excitation check: features with insufficient
      weighted variance are held at their current_beta values (Ljung §13.3).

    Weights: 1.0 / (1.0 + |error_c|) — at-target observations get full
    weight, observations further from target are downweighted.

    Returns None if insufficient eligible observations.
    """
    # Filter to near-equilibrium, unclamped observations
    eligible = [
        o for o in observations
        if not o.clamped and abs(o.room_rate) < room_rate_threshold
    ]

    if len(eligible) < min_observations:
        return None

    n = n_features
    m = len(eligible)

    y = [o.hp_setpoint - o.current_c for o in eligible]
    X = [o.features[:n] for o in eligible]
    w = [1.0 / (1.0 + abs(o.current_c - o.desired_c)) for o in eligible]

    # ── Persistent excitation check per feature ──────────────────────
    # Feature 0 (intercept, always 1.0) is always identifiable — its
    # "variance" is zero but it's structurally needed for the regression.
    held: set[int] = set()
    for j in range(1, n):  # skip intercept
        col = [X[k][j] if j < len(X[k]) else 0.0 for k in range(m)]
        if _weighted_variance(col, w) < min_feature_variance:
            held.add(j)

    # Identifiable feature indices
    active = [j for j in range(n) if j not in held]

    if len(active) < 1:
        return None

    # ── Subtract held-feature contributions from y ───────────────────
    # y_adj = y - Σ(held j) current_beta[j] * x[j]
    # This lets WLS solve only for the active subset while accounting
    # for held features' known contributions.
    fallback = current_beta if current_beta else [0.0] * n
    y_adj = list(y)
    for j in held:
        bj = fallback[j] if j < len(fallback) else 0.0
        for k in range(m):
            xj = X[k][j] if j < len(X[k]) else 0.0
            y_adj[k] -= bj * xj

    # ── Build reduced system: X_a'W X_a β_a = X_a'W y_adj ───────────
    na = len(active)
    XtWX = [[0.0] * na for _ in range(na)]
    XtWy = [0.0] * na

    for k in range(m):
        for ii, i in enumerate(active):
            xi = X[k][i] if i < len(X[k]) else 0.0
            XtWy[ii] += xi * w[k] * y_adj[k]
            for jj, j in enumerate(active):
                xj = X[k][j] if j < len(X[k]) else 0.0
                XtWX[ii][jj] += xi * w[k] * xj

    ridge = 1e-6
    for ii in range(na):
        XtWX[ii][ii] += ridge

    beta_active = _solve_symmetric(XtWX, XtWy, na)
    if beta_active is None:
        return None

    # ── Reassemble full beta vector ──────────────────────────────────
    beta = [0.0] * n
    for j in held:
        beta[j] = fallback[j] if j < len(fallback) else 0.0
    for ii, j in enumerate(active):
        beta[j] = beta_active[ii]

    # Compute RMS residual (using full beta, original y)
    ss = 0.0
    for k in range(m):
        pred = sum(beta[i] * (X[k][i] if i < len(X[k]) else 0.0) for i in range(n))
        ss += (y[k] - pred) ** 2
    rms = math.sqrt(ss / m) if m > 0 else 0.0

    # ── Per-coefficient standard error from (X'WX)⁻¹ ────────────────
    # σ²(βᵢ) = RMS² × diag((X'WX)⁻¹)ᵢ  (Ljung §9.4)
    # Solve (X'WX) eᵢ = eᵢ for each column to get diagonal of inverse.
    std_err = [float("inf")] * n  # held features get inf (unknown)
    cov_diag = _diagonal_of_inverse(XtWX, na)
    if cov_diag is not None:
        rms_sq = rms * rms if rms > 0 else 1e-12
        for ii, j in enumerate(active):
            var_j = rms_sq * max(0.0, cov_diag[ii])
            std_err[j] = math.sqrt(var_j) if var_j > 0 else 0.0

    return BatchResult(
        n_total=len(observations),
        n_eligible=m,
        beta_batch=beta,
        beta_current=[],  # filled in by caller
        residual_rms=rms,
        max_coeff_change_pct=0.0,  # filled in by caller
        recommend_update=False,  # filled in by caller
        held_features=held,
        beta_std_err=std_err,
    )


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
        if abs(M[i][i]) < 1e-12:
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
    for i in range(min(n, len(current_beta_physical))):
        current = current_beta_physical[i]
        batch = result.beta_batch[i]
        is_held = i in held
        if is_held:
            pct = 0.0
        elif abs(current) > 1e-6:
            pct = abs(batch - current) / abs(current) * 100
        elif abs(batch) > 1e-6:
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
    _LOGGER.info(
        "%sBatch WLS analysis: %d/%d observations eligible, RMS=%.3f%s",
        log_prefix, result.n_eligible, result.n_total, result.residual_rms,
        f", {n_held} feature{'s' if n_held != 1 else ''} held (no variance)"
        if n_held else "",
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
) -> BatchResult:
    """Compute a safe blended update via covariance-weighted fusion.

    Per-coefficient Kalman-style gain (Ljung §11.4):
        K_i = σ²_prior / (σ²_prior + σ²_batch_i)

    When σ²_batch is small (tight estimate), K → 1 and the batch pulls
    the coefficient strongly.  When σ²_batch is large (uncertain), K → 0
    and the current value holds.  Held features (σ²_batch = ∞) always
    get K = 0.

    Safety net: per-coefficient step cap of ±max_step regardless of gain.

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

        delta = gains[i] * (batch[i] - current[i])
        if abs(delta) > max_step:
            delta = max_step if delta > 0 else -max_step
        blended[i] = current[i] + delta

    result.beta_blended = blended
    result.blend_gains = gains

    _LOGGER.info(
        "Batch blend: prior_std=%.2f, max_step=%.1f",
        prior_std, max_step,
    )
    for i in range(n):
        se = std_err[i] if i < len(std_err) else float("inf")
        if math.isinf(se):
            _LOGGER.info(
                "  β%d: held (no batch uncertainty estimate)",
                i,
            )
        elif current[i] != blended[i]:
            _LOGGER.info(
                "  β%d: %.4f → %.4f (K=%.3f, σ_batch=%.4f, Δ=%.4f)",
                i, current[i], blended[i], gains[i], se,
                blended[i] - current[i],
            )
        else:
            _LOGGER.debug(
                "  β%d: %.4f (unchanged, K=%.3f, σ_batch=%.4f)",
                i, current[i], gains[i], se,
            )

    return result
