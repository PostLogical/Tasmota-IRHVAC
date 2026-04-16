"""Recursive Least Squares feedforward model.

Standalone module — no Home Assistant dependencies. Used by PIController
to learn optimal HP setpoint offsets from multiple input features.
"""

from __future__ import annotations

import logging
from typing import Any

from ..const import (
    DEFAULT_RLS_DELTA,
    DEFAULT_RLS_LAMBDA_BASE,
    DEFAULT_RLS_LAMBDA_MIN,
    DEFAULT_RLS_P_INIT,
)

_LOGGER = logging.getLogger(__name__)


class RLSModel:
    """Recursive Least Squares feedforward model.

    Predicts HP setpoint offset from multiple input factors. Learns online
    via RLS with variable forgetting factor and ridge regularization.

    Each model input has:
        - A current value (read from HA entity)
        - A learned coefficient (updated by RLS)
        - An optional lag filter (exponential smoothing)
        - Min/max coefficient clamps (physical bounds)
    """

    def __init__(
        self,
        n_inputs: int,
        seed_coefficients: list[float] | None = None,
        lambda_base: float = DEFAULT_RLS_LAMBDA_BASE,
        lambda_min: float = DEFAULT_RLS_LAMBDA_MIN,
        delta: float = DEFAULT_RLS_DELTA,
        p_init: float = DEFAULT_RLS_P_INIT,
        coeff_clamps: list[tuple[float, float] | None] | None = None,
        feature_scales: list[float] | None = None,
    ) -> None:
        """Initialize RLS model.

        Args:
            n_inputs: Number of input features (excluding intercept).
            seed_coefficients: Initial β vector [intercept, β₁, β₂, ...] in
                              physical units. Length n_inputs + 1. Defaults to zeros.
            lambda_base: Base forgetting factor (0.99-0.999).
            lambda_min: Minimum λ when residuals are large.
            delta: Covariance regularization constant (added to P diagonal
                   each step to prevent covariance windup).
            p_init: Initial covariance diagonal value (uniform for all dimensions).
            coeff_clamps: List of (min, max) tuples per coefficient in physical
                         units, or None.
            feature_scales: Typical magnitude of each feature [1, scale₁, ...].
                           Features are normalized by these scales before entering
                           the RLS, so all dimensions are O(1). Coefficients are
                           stored internally in normalized space and converted to
                           physical units for prediction and external access.
        """
        self.n: int = n_inputs + 1  # +1 for intercept
        self.lambda_base: float = lambda_base
        self.lambda_min: float = lambda_min
        self.delta: float = delta
        self.p_init: float = p_init

        # Feature scales: normalize features to O(1) before RLS math.
        # This gives truly balanced learning rates across all dimensions.
        self.feature_scales: list[float] = feature_scales or [1.0] * self.n
        while len(self.feature_scales) < self.n:
            self.feature_scales.append(1.0)

        # Coefficient vector β in normalized space.
        # Physical β_phys[i] = β_norm[i] / scale[i]
        # So β_norm[i] = β_phys[i] * scale[i]
        self.beta: list[float]
        if seed_coefficients is not None:
            self.beta = [
                seed_coefficients[i] * self.feature_scales[i]
                if i < len(seed_coefficients) else 0.0
                for i in range(self.n)
            ]
        else:
            self.beta = [0.0] * self.n

        # Seed values in normalized space (for seed change detection)
        self.beta_seed: list[float] = list(self.beta)

        # Covariance matrix P (n × n, stored as flat list row-major)
        # Uniform initialization — feature normalization handles scale balance.
        self.P: list[float] = [0.0] * (self.n * self.n)
        for i in range(self.n):
            self.P[i * self.n + i] = p_init

        # Coefficient clamps in normalized space
        self.coeff_clamps: list[tuple[float, float] | None]
        if coeff_clamps is not None:
            self.coeff_clamps = [
                (clamp[0] * self.feature_scales[i], clamp[1] * self.feature_scales[i])
                if clamp is not None and i < len(self.feature_scales)
                else clamp
                for i, clamp in enumerate(coeff_clamps)
            ]
        else:
            self.coeff_clamps = [None] * self.n

        # Observation counter
        self.observation_count: int = 0

    def _normalize(self, x: list[float]) -> list[float]:
        """Normalize raw feature vector to O(1) by dividing by scales."""
        return [x[i] / self.feature_scales[i] for i in range(self.n)]

    def predict(self, x: list[float]) -> float:
        """Predict offset from feature vector.

        Args:
            x: Feature vector [1, x₁, x₂, ...] in physical units.
               Length must equal self.n.

        Returns:
            Predicted offset (float).
        """
        # β_norm · x_norm = Σ (β_phys * scale) * (x / scale) = Σ β_phys * x
        x_norm = self._normalize(x)
        return sum(self.beta[i] * x_norm[i] for i in range(self.n))

    def update(self, x: list[float], y: float) -> float:
        """Update coefficients via RLS with one observation.

        Args:
            x: Feature vector [1, x₁, x₂, ...] in physical units.
            y: Observed offset (float).

        Returns:
            Residual (y - prediction before update).
        """
        n = self.n
        x_norm = self._normalize(x)

        # Prediction error (residual)
        y_pred = sum(self.beta[i] * x_norm[i] for i in range(n))
        residual = y - y_pred

        # Kalman gain: K = P·x / (λ + x'·P·x)
        Px = [sum(self.P[i * n + j] * x_norm[j] for j in range(n)) for i in range(n)]
        xPx = sum(x_norm[i] * Px[i] for i in range(n))

        # Variable forgetting factor based on normalized residual.
        # When residual^2 >> expected prediction variance (xPx), the model
        # is surprised → decrease λ for faster adaptation. When residuals
        # are within expected variance, keep λ high for stability.
        normalized_sq = (residual * residual) / max(xPx, 0.01)
        blend = min(normalized_sq / 9.0, 1.0)  # 9 = 3-sigma threshold squared
        lam = self.lambda_base - (self.lambda_base - self.lambda_min) * blend

        denom = lam + xPx
        if denom == 0:
            return residual
        K = [Px[i] / denom for i in range(n)]

        # Update coefficients: standard RLS (no ad-hoc beta penalty).
        # Seed anchoring comes from initial conditions and the delta*I
        # term in the P update, which prevents covariance collapse.
        for i in range(n):
            self.beta[i] += K[i] * residual

        # Apply coefficient clamps with P projection.
        # When a coefficient hits a boundary, zero its row/col in P
        # so the estimator knows this dimension is constrained.
        for i in range(n):
            clamp = self.coeff_clamps[i] if i < len(self.coeff_clamps) else None
            if clamp is not None:
                lo, hi = clamp
                unclamped = self.beta[i]
                self.beta[i] = max(lo, min(hi, self.beta[i]))
                if self.beta[i] != unclamped:
                    for j in range(n):
                        self.P[i * n + j] = 0.0
                        self.P[j * n + i] = 0.0
                    self.P[i * n + i] = self.delta

        # Update covariance: P = (P - K·x'·P) / λ + δ·I
        new_P = [0.0] * (n * n)
        for i in range(n):
            for j in range(n):
                col_j = sum(x_norm[k] * self.P[k * n + j] for k in range(n))
                new_P[i * n + j] = (self.P[i * n + j] - K[i] * col_j) / lam

        # Regularization: prevent covariance windup by adding δ·I each step
        for i in range(n):
            new_P[i * n + i] += self.delta

        self.P = new_P
        self.observation_count += 1

        return residual

    def get_coefficients(self) -> dict[int, float]:
        """Return coefficient dict in physical units: {index: value}."""
        return {i: self.beta[i] / self.feature_scales[i] for i in range(self.n)}

    def get_covariance_diagonal(self) -> list[float]:
        """Return diagonal of P (uncertainty per coefficient)."""
        return [self.P[i * self.n + i] for i in range(self.n)]

    def as_dict(self) -> dict[str, Any]:
        """Serialize model state to dict (beta in normalized space)."""
        return {
            "beta": list(self.beta),
            "P": list(self.P),
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], n_inputs: int, **kwargs: Any) -> RLSModel:
        """Restore model from serialized dict.

        Beta and P are stored in normalized space. Handles length mismatches
        when model inputs are added/removed.
        """
        model = cls(n_inputs, **kwargs)
        if "beta" in data:
            beta = data["beta"]
            if len(beta) == model.n:
                model.beta = [float(v) for v in beta]
            elif len(beta) < model.n:
                for i in range(len(beta)):
                    model.beta[i] = float(beta[i])
                _LOGGER.info("RLS restore: stored %d coefficients, model needs %d — seeding new inputs",
                           len(beta), model.n)
            else:
                for i in range(model.n):
                    model.beta[i] = float(beta[i])
                _LOGGER.info("RLS restore: stored %d coefficients, model needs %d — truncating",
                           len(beta), model.n)
        if "P" in data:
            P = data["P"]
            if len(P) == model.n * model.n:
                model.P = [float(v) for v in P]
            else:
                _LOGGER.info("RLS restore: covariance matrix size mismatch, using initial P")
        if "observation_count" in data:
            model.observation_count = int(data["observation_count"])
        return model
