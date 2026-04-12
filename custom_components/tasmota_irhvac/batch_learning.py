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


def weighted_least_squares(
    observations: list[Observation],
    n_features: int,
    room_rate_threshold: float = 0.02,
    min_observations: int = 20,
) -> BatchResult | None:
    """Run weighted least squares on filtered observations.

    Filters:
    - Excludes clamped observations (censored data)
    - Excludes observations with room_rate > threshold (not at equilibrium)

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

    # Build matrices: y = Xβ, solve β = (X'WX)⁻¹ X'Wy
    n = n_features
    m = len(eligible)

    # Observation: hp_setpoint - current_c (equilibrium offset)
    y = [o.hp_setpoint - o.current_c for o in eligible]
    X = [o.features[:n] for o in eligible]  # truncate/pad to n_features
    w = [1.0 / (1.0 + abs(o.current_c - o.desired_c)) for o in eligible]

    # X'WX (n×n) and X'Wy (n×1)
    XtWX = [[0.0] * n for _ in range(n)]
    XtWy = [0.0] * n

    for k in range(m):
        for i in range(n):
            xi = X[k][i] if i < len(X[k]) else 0.0
            XtWy[i] += xi * w[k] * y[k]
            for j in range(n):
                xj = X[k][j] if j < len(X[k]) else 0.0
                XtWX[i][j] += xi * w[k] * xj

    # Add small ridge for numerical stability
    ridge = 1e-6
    for i in range(n):
        XtWX[i][i] += ridge

    # Solve via Cholesky-like approach (n is small, typically 3-8)
    beta = _solve_symmetric(XtWX, XtWy, n)
    if beta is None:
        return None

    # Compute RMS residual
    ss = 0.0
    for k in range(m):
        pred = sum(beta[i] * (X[k][i] if i < len(X[k]) else 0.0) for i in range(n))
        ss += (y[k] - pred) ** 2
    rms = math.sqrt(ss / m) if m > 0 else 0.0

    return BatchResult(
        n_total=len(observations),
        n_eligible=m,
        beta_batch=beta,
        beta_current=[],  # filled in by caller
        residual_rms=rms,
        max_coeff_change_pct=0.0,  # filled in by caller
        recommend_update=False,  # filled in by caller
    )


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
    result.recommend_update.  Always logs — does not modify the model.
    """
    result.beta_current = list(current_beta_physical)
    n = len(result.beta_batch)
    names = coeff_names or [f"β{i}" for i in range(n)]

    max_change = 0.0
    changes = []
    for i in range(min(n, len(current_beta_physical))):
        current = current_beta_physical[i]
        batch = result.beta_batch[i]
        if abs(current) > 1e-6:
            pct = abs(batch - current) / abs(current) * 100
        elif abs(batch) > 1e-6:
            pct = 100.0
        else:
            pct = 0.0
        max_change = max(max_change, pct)
        name = names[i] if i < len(names) else f"β{i}"
        changes.append((name, current, batch, pct))

    result.max_coeff_change_pct = max_change
    result.recommend_update = (
        max_change > change_threshold_pct
        and result.n_eligible >= min_observations
    )

    # Log the analysis
    _LOGGER.info(
        "%sBatch WLS analysis: %d/%d observations eligible, RMS=%.3f",
        log_prefix, result.n_eligible, result.n_total, result.residual_rms,
    )
    for name, current, batch, pct in changes:
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
