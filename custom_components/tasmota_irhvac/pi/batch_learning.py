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
    hp_setpoint: float  # integer HP setpoint (°C)
    current_c: float  # filtered room temperature (°C)
    desired_c: float  # target temperature (°C)
    outdoor_temp_c: float | None  # absolute outdoor temperature (°C)
    room_rate: float  # dT/dt in °C/min at observation time
    raw_readings: dict[str, float]  # entity_id → raw sensor value at obs time
    clamped: bool  # True if HP setpoint was at min or max
    clamped_reason: str = ""  # "", "no_output", "saturated_low", "saturated_high"
    supplemental_active: bool = False  # supplemental source tracking or assisting

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
        )


# Default capacity for diversity-aware buffer.  ~2000 observations at
# 4/hr fills in ~3 weeks, after which leverage scoring governs eviction.
DEFAULT_DIVERSITY_BUFFER_SIZE = 2000

# Regularization for (X^T X)^{-1} to keep it invertible before the
# buffer fills and during early operation when rank may be deficient.
INFO_MATRIX_REGULARIZATION = 1e-4


def build_feature_vector_from_raw(
    obs: Observation,
    model_inputs: list[dict[str, Any]],
    feature_order: list[str],
) -> list[float] | None:
    """Build a feature vector from raw readings + current config.

    Returns an ordered list aligned to ``feature_order``, or None if the
    observation is missing readings for any required model input entity.

    Feature construction:
    - "intercept": always 1.0
    - "outdoor_delta": obs.outdoor_temp_c - obs.current_c (requires outdoor_temp_c)
    - model inputs: raw_readings[entity_id], with delta_from_room adjustment
      if configured (entity_value - current_c)

    No EMA is applied — batch WLS operates on raw instantaneous values.
    The online RLS uses EMA for tick-by-tick smoothing, but the batch
    fits across many diverse observations where individual noise averages out.
    """
    if obs.outdoor_temp_c is None:
        return None

    features: dict[str, float] = {
        "intercept": 1.0,
        "outdoor_delta": obs.outdoor_temp_c - obs.current_c,
    }

    for m_input in model_inputs:
        entity_id = m_input.get("entity_id", "")
        name = m_input.get("name", entity_id)
        if not entity_id or entity_id not in obs.raw_readings:
            return None  # incomplete — skip this observation for this feature set
        value = obs.raw_readings[entity_id]
        # Apply delta_from_room if configured (same transform as live path)
        if m_input.get("delta_from_room"):
            value = value - obs.current_c
        features[name] = value

    return [features.get(name, 0.0) for name in feature_order]




class DiversityAwareBuffer:
    """Leverage-scored observation buffer for long-term diverse data retention.

    Instead of FIFO eviction, retains observations that maximize the
    information content of the buffer (D-optimal design).  Each observation's
    leverage score measures how much unique information it contributes:

        leverage(x) = x^T (X^T X + λI)^{-1} x

    When the buffer is full, a new observation replaces the stored observation
    with the lowest leverage score, but only if the new observation has higher
    leverage.  This naturally retains rare operating conditions (cold snaps,
    pellet stove events, door transitions) while shedding redundant
    steady-state observations.

    Leverage scoring uses raw readings + current model config to build
    feature vectors on the fly.  This means scoring adapts when config
    changes — an observation that was low-leverage under the old config
    may become high-leverage under the new one (e.g., after adding a
    solar input, observations from sunny days become more valuable).

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
    ) -> None:
        self._buffer: list[Observation] = []
        self._max_size = max_size
        self._n_features = n_features
        self._feature_order: list[str] | None = feature_order
        self._model_inputs: list[dict[str, Any]] = model_inputs or []
        # (X^T X + λI)^{-1} — the inverse information matrix, n×n.
        # Initialized to (1/λ) * I (no data yet).
        n = n_features
        reg_inv = 1.0 / INFO_MATRIX_REGULARIZATION
        self._info_inv: list[list[float]] = [
            [reg_inv if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        # Forward matrix X^T X + λI for condition number estimation.
        self._xtx_matrix: list[list[float]] | None = None
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
            self._buffer = [o for o in self._buffer if not (o.hp_setpoint < o.current_c)]
        else:
            self._buffer = [o for o in self._buffer if not (o.hp_setpoint > o.current_c)]
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

    def add(self, obs: Observation) -> None:
        """Add an observation, using leverage-scored eviction when full."""
        x = self._get_feature_vector(obs)
        new_leverage = self._compute_leverage(x)

        if len(self._buffer) < self._max_size:
            # Buffer not full — always accept.
            self._buffer.append(obs)
            self._sherman_morrison_update(x)
        else:
            # Find the lowest-leverage observation using current info matrix.
            min_idx = 0
            min_lev = self._compute_leverage(self._get_feature_vector(self._buffer[0]))
            for i in range(1, len(self._buffer)):
                lev = self._compute_leverage(self._get_feature_vector(self._buffer[i]))
                if lev < min_lev:
                    min_lev = lev
                    min_idx = i
            if new_leverage > min_lev:
                # Downdate the evicted observation, then update with new.
                old_x = self._get_feature_vector(self._buffer[min_idx])
                self._sherman_morrison_downdate(old_x)
                self._buffer[min_idx] = obs
                self._sherman_morrison_update(x)
            # else: new observation is less informative than everything
            # in the buffer — discard it silently.

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
        if self._n_features > 0:
            partial[0] = 1.0  # intercept
        if self._n_features > 1 and obs.outdoor_temp_c is not None:
            partial[1] = obs.outdoor_temp_c - obs.current_c
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
            if inv is not None:
                self._info_inv = inv

        self._updates_since_recompute = 0

    def compute_condition_number(self) -> float:
        """Compute the spectral condition number of the column-normalized X'X.

        Column-normalizes X'X by dividing each entry (i,j) by
        √(diag[i]) · √(diag[j]), converting it to a correlation matrix.
        This removes scale-induced conditioning and reports only true
        multicollinearity.

        κ = √(λ_max / λ_min) where λ are eigenvalues of the correlation
        matrix + λI regularization.

        Thresholds (Belsley, Kuh & Welsch, "Regression Diagnostics", 1980):
        - κ > 30: moderate multicollinearity, coefficients becoming unreliable
        - κ > 100: severe multicollinearity, coefficient estimates numerically unstable

        Uses power iteration for λ_max and inverse iteration for λ_min.
        Returns inf if the forward matrix hasn't been computed yet.
        """
        if self._xtx_matrix is None:
            return float('inf')
        n = self._n_features
        if n < 2:
            return 1.0

        # Column-normalize X'X → correlation matrix.
        # corr[i][j] = xtx[i][j] / (√xtx[i][i] · √xtx[j][j])
        diag_sqrt = [
            math.sqrt(self._xtx_matrix[i][i])
            if self._xtx_matrix[i][i] > 1e-15 else 1.0
            for i in range(n)
        ]
        corr = [
            [self._xtx_matrix[i][j] / (diag_sqrt[i] * diag_sqrt[j])
             for j in range(n)]
            for i in range(n)
        ]

        # Power iteration for λ_max of correlation matrix
        v = [1.0 / math.sqrt(n)] * n
        lambda_max = 0.0
        for _ in range(50):
            w = [sum(corr[i][j] * v[j] for j in range(n)) for i in range(n)]
            lambda_max = sum(v[i] * w[i] for i in range(n))
            norm = math.sqrt(sum(wi * wi for wi in w))
            if norm < 1e-15:
                return float('inf')
            v = [wi / norm for wi in w]

        # Inverse iteration for λ_min: need inverse of corr matrix
        corr_copy = [row[:] for row in corr]
        corr_inv = self._invert_matrix(corr_copy, n)
        if corr_inv is None:
            return float('inf')

        v = [1.0 / math.sqrt(n)] * n
        v[0] += 0.1
        norm = math.sqrt(sum(vi * vi for vi in v))
        v = [vi / norm for vi in v]

        inv_lambda_min = 0.0
        for _ in range(50):
            w = [sum(corr_inv[i][j] * v[j] for j in range(n)) for i in range(n)]
            inv_lambda_min = sum(v[i] * w[i] for i in range(n))
            norm = math.sqrt(sum(wi * wi for wi in w))
            if norm < 1e-15:
                return float('inf')
            v = [wi / norm for wi in w]

        if inv_lambda_min < 1e-15:
            return float('inf')

        lambda_min = 1.0 / inv_lambda_min
        if lambda_min < 1e-15:
            return float('inf')

        return math.sqrt(lambda_max / lambda_min)

    def get_pairwise_correlations(
        self, feature_names: list[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        """Compute pairwise Pearson correlations between features.

        Returns (name_i, name_j, r) for all pairs with |r| > 0.7,
        skipping the intercept (always 1.0, undefined correlation).
        Only uses unclamped observations for relevance to WLS.
        """
        n = self._n_features
        if n < 3 or len(self._buffer) < 20:
            return []

        names = feature_names or [f"feature_{i}" for i in range(n)]
        unclamped = [o for o in self._buffer if o.clamped_reason not in ("no_output", "clamped")]
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

        return results

    def get_leverage_scores(self) -> list[float]:
        """Compute and return current leverage scores (for diagnostics)."""
        return [self._compute_leverage(self._get_feature_vector(o)) for o in self._buffer]

    def get_min_leverage(self) -> float:
        """Return the minimum leverage score in the buffer."""
        if not self._buffer:
            return 0.0
        return min(self._compute_leverage(self._get_feature_vector(o)) for o in self._buffer)

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
        self._updates_since_recompute += 1

    def _sherman_morrison_downdate(self, x: list[float]) -> None:
        """Rank-1 update for removing an observation: (A - xx^T)^{-1}.

        (A - xx^T)^{-1} = A^{-1} + (A^{-1} x x^T A^{-1}) / (1 - x^T A^{-1} x)
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
    """
    n_active = len(ctx.active_input_indices)
    n_joint = ctx.n_base + n_active
    m_complete = len(ctx.complete_indices)

    # Build joint feature matrix [intercept, outdoor_delta, input_0, ...]
    joint_cols: list[list[float]] = []
    for j in range(ctx.n_base):
        joint_cols.append([ctx.X_base[k][j] for k in ctx.complete_indices])
    for fi in ctx.active_input_indices:
        joint_cols.append([ctx.input_values_by_obs[k][fi] for k in ctx.complete_indices])

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

    # Denormalize into full-size beta vector
    beta = [0.0] * ctx.n
    beta[0] = beta_norm[0] / col_scales[0]
    beta[1] = beta_norm[1] / col_scales[1]
    for jj, fi in enumerate(ctx.active_input_indices):
        beta[fi + 2] = beta_norm[ctx.n_base + jj] / col_scales[ctx.n_base + jj]

    # Standard errors from (X'WX)^-1
    std_err = [float("inf")] * ctx.n
    cov_diag = _diagonal_of_inverse(XtWX, n_joint)
    if cov_diag is not None:
        resid = [
            y[idx] - sum(
                joint_cols[jj][idx] * (beta[0], beta[1], *[beta[fi + 2] for fi in ctx.active_input_indices])[jj]
                for jj in range(n_joint)
            )
            for idx in range(m_complete)
        ]
        rms_sq = sum(r * r for r in resid) / max(1, m_complete - n_joint)
        for i in range(ctx.n_base):
            var_i = rms_sq * max(0.0, cov_diag[i]) / (col_scales[i] ** 2)
            std_err[i] = math.sqrt(var_i) if var_i > 0 else 0.0
        for jj, fi in enumerate(ctx.active_input_indices):
            idx_in = ctx.n_base + jj
            var_i = rms_sq * max(0.0, cov_diag[idx_in]) / (col_scales[idx_in] ** 2)
            std_err[fi + 2] = math.sqrt(var_i) if var_i > 0 else 0.0

    return beta, std_err


def _solve_fwl(
    ctx: _RegressionContext,
) -> tuple[list[float], list[float]]:
    """Frisch-Waugh-Lovell partial regression for ragged data.

    Used when the complete-data subset is too small for joint OLS but
    individual features have enough observations in their subsets.
    Each feature coefficient is unbiased (base regressors partialled out).
    Base intercept and outdoor_delta are re-estimated afterward.
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

    for fi in ctx.active_input_indices:
        coeff_idx = fi + 2
        subset_indices = [k for k in range(ctx.m_base) if ctx.input_values_by_obs[k][fi] is not None]
        subset_values = [ctx.input_values_by_obs[k][fi] for k in subset_indices]
        m_sub = len(subset_indices)
        w_sub = [ctx.w_base[k] for k in subset_indices]
        y_sub = [residuals_base[k] for k in subset_indices]
        X_base_sub = [ctx.X_base[k] for k in subset_indices]

        # Partial out base regressors: regress z on [intercept, outdoor_delta]
        XtWX_zb = [[0.0] * ctx.n_base for _ in range(ctx.n_base)]
        XtWy_zb = [0.0] * ctx.n_base
        for i_sub in range(m_sub):
            for a in range(ctx.n_base):
                xa = X_base_sub[i_sub][a]
                XtWy_zb[a] += xa * w_sub[i_sub] * subset_values[i_sub]
                for b in range(ctx.n_base):
                    XtWX_zb[a][b] += xa * w_sub[i_sub] * X_base_sub[i_sub][b]
        for a in range(ctx.n_base):
            XtWX_zb[a][a] += ctx.ridge
        gamma = _solve_symmetric(XtWX_zb, XtWy_zb, ctx.n_base)

        r_z = (
            [subset_values[i] - sum(gamma[a] * X_base_sub[i][a] for a in range(ctx.n_base)) for i in range(m_sub)]
            if gamma is not None else subset_values
        )

        if _weighted_variance(r_z, w_sub) < ctx.min_feature_variance:
            held.add(coeff_idx)
            continue

        wrzry = sum(w_sub[i] * r_z[i] * y_sub[i] for i in range(m_sub))
        wrzrz = sum(w_sub[i] * r_z[i] * r_z[i] for i in range(m_sub))
        if wrzrz < 1e-15:
            held.add(coeff_idx)
            continue

        beta[coeff_idx] = wrzry / wrzrz
        sub_resid = [y_sub[i] - beta[coeff_idx] * r_z[i] for i in range(m_sub)]
        sub_rms_sq = sum(r * r for r in sub_resid) / max(1, m_sub - 1)
        std_err[coeff_idx] = math.sqrt(sub_rms_sq / wrzrz) if wrzrz > 1e-15 else float("inf")

    # Re-estimate base to absorb model input contributions
    y_adj = list(ctx.y_base)
    for k in range(ctx.m_base):
        for fi in ctx.active_input_indices:
            coeff_idx = fi + 2
            if coeff_idx not in held and ctx.input_values_by_obs[k][fi] is not None:
                y_adj[k] -= beta[coeff_idx] * ctx.input_values_by_obs[k][fi]
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
    if beta_base_adj is not None:
        beta[0] = beta_base_adj[0] / ctx.col_scales_base[0]
        beta[1] = beta_base_adj[1] / ctx.col_scales_base[1]

    # Base std errors from adjusted system
    cov_diag = _diagonal_of_inverse(XtWX_adj if beta_base_adj else ctx.XtWX_base, ctx.n_base)
    if cov_diag is not None:
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
) -> BatchResult | None:
    """Run weighted least squares on physical observations.

    Orchestrates three phases:
    1. **Filter & classify** — exclude ineligible observations, classify
       features as active/held, partition data by completeness.
    2. **Solve** — joint OLS on complete data (exact), or FWL on partial
       data (unbiased per-feature).  Grey-box solvers plug in here.
    3. **Package** — compute residuals, exclude outliers, return BatchResult.

    Returns None if insufficient eligible observations.
    """
    # ── Phase 1: Filter & classify ──────────────────────────────────
    _EXCLUDE_REASONS = ("no_output", "clamped")
    eligible = [
        o for o in observations
        if o.clamped_reason not in _EXCLUDE_REASONS
        and abs(o.room_rate) < room_rate_threshold
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
    y_base = [o.hp_setpoint - o.current_c for o in base_eligible]
    w_base = [1.0 / (1.0 + (o.room_rate / room_rate_threshold) ** 2) for o in base_eligible]
    X_base = [[1.0, o.outdoor_temp_c - o.current_c] for o in base_eligible]

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

    # Classify model input features
    input_entity_ids = [m.get("entity_id", "") for m in m_inputs]
    input_values_by_obs: list[list[float | None]] = []
    for k, o in enumerate(base_eligible):
        row: list[float | None] = []
        for feat_idx, m_input in enumerate(m_inputs):
            entity_id = input_entity_ids[feat_idx]
            if entity_id and entity_id in o.raw_readings:
                value = o.raw_readings[entity_id]
                if m_input.get("delta_from_room"):
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
    active_input_indices: list[int] = []
    for feat_idx, m_input in enumerate(m_inputs):
        coeff_idx = feat_idx + 2
        entity_id = input_entity_ids[feat_idx]
        if not entity_id:
            held.add(coeff_idx)
            continue
        if feature_obs_counts[coeff_idx] < max(min_observations, 10):
            held.add(coeff_idx)
            continue
        vals = [input_values_by_obs[k][feat_idx]
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
        if cov_diag is not None:
            rms_base = math.sqrt(
                sum((y_base[k] - sum(beta_base[i] * X_base[k][i] for i in range(n_base))) ** 2
                    for k in range(m_base)) / max(1, m_base - n_base)
            )
            rms_sq = rms_base * rms_base if rms_base > 0 else 1e-12
            for i in range(n_base):
                var_i = rms_sq * max(0.0, cov_diag[i]) / (col_scales_base[i] ** 2)
                std_err[i] = math.sqrt(var_i) if var_i > 0 else 0.0

    # Fill held features from current model
    fallback = current_beta if current_beta else [0.0] * n
    for j in held:
        beta[j] = fallback[j] if j < len(fallback) else 0.0

    # ── Phase 3: Package (residuals, outlier exclusion) ─────────────
    if feature_order and m_inputs:
        full_obs: list[Observation] = []
        full_X: list[list[float]] = []
        full_y: list[float] = []
        for o in base_eligible:
            vec = build_feature_vector_from_raw(o, m_inputs, feature_order)
            if vec is not None:
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
                    if entity_id and entity_id in full_obs[k].raw_readings:
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


# ── Residual time-of-day analysis ──────────────────────────────────


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
    import datetime as _dt

    # Bin residuals by wall-clock hour
    hour_residuals: dict[int, list[float]] = {h: [] for h in range(24)}

    for o in observations:
        if o.clamped_reason in ("no_output", "clamped") or abs(o.room_rate) >= room_rate_threshold:
            continue
        # Derive wall hour from wall_time (UTC epoch → local hour)
        wall_hour = _dt.datetime.fromtimestamp(o.wall_time).hour if o.wall_time > 0 else -1
        if wall_hour < 0:
            continue
        # Build feature vector from raw readings + current config
        if feature_order is None or model_inputs is None:
            continue
        x = build_feature_vector_from_raw(o, model_inputs, feature_order)
        if x is None:
            continue
        predicted = sum(beta[i] * x[i] for i in range(n_features))
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
        for offset in range(24):
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

        if total_obs > 0:
            patterns.append(HourlyResidualPattern(
                start_hour=start,
                end_hour=end,
                mean_residual=total_residual / total_obs,
                n_observations=total_obs,
            ))

    return patterns
