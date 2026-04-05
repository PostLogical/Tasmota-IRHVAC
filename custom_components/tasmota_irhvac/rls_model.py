"""Recursive Least Squares feedforward model.

Standalone module — no Home Assistant dependencies. Used by PIController
to learn optimal HP setpoint offsets from multiple input features.
"""

import logging

from .const import (
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

    def __init__(self, n_inputs, seed_coefficients=None,
                 lambda_base=DEFAULT_RLS_LAMBDA_BASE,
                 lambda_min=DEFAULT_RLS_LAMBDA_MIN,
                 delta=DEFAULT_RLS_DELTA,
                 p_init=DEFAULT_RLS_P_INIT,
                 coeff_clamps=None,
                 feature_scales=None):
        """Initialize RLS model.

        Args:
            n_inputs: Number of input features (excluding intercept).
            seed_coefficients: Initial β vector [intercept, β₁, β₂, ...].
                              Length n_inputs + 1. Defaults to zeros.
            lambda_base: Base forgetting factor (0.99-0.999).
            lambda_min: Minimum λ when residuals are large.
            delta: Covariance regularization constant (added to P diagonal
                   each step to prevent covariance windup).
            p_init: Base initial covariance diagonal value.
            coeff_clamps: List of (min, max) tuples per coefficient, or None.
            feature_scales: Typical magnitude of each feature [1, scale₁, ...].
                           Used to scale P initialization so all dimensions
                           have balanced learning rates. Defaults to all 1.0.
        """
        self.n = n_inputs + 1  # +1 for intercept
        self.lambda_base = lambda_base
        self.lambda_min = lambda_min
        self.delta = delta

        # Coefficient vector β (intercept + n_inputs)
        if seed_coefficients is not None:
            self.beta = list(seed_coefficients)
            # Pad with zeros if seed is shorter
            while len(self.beta) < self.n:
                self.beta.append(0.0)
        else:
            self.beta = [0.0] * self.n

        # Seed values (retained for blend and seed change detection)
        self.beta_seed = list(self.beta)

        # Feature scales for P initialization (retained for seed change reset)
        self.feature_scales = feature_scales or [1.0] * self.n
        while len(self.feature_scales) < self.n:
            self.feature_scales.append(1.0)

        # Covariance matrix P (n × n, stored as flat list row-major)
        # Scale each diagonal by inverse feature magnitude squared so all
        # dimensions have balanced initial learning rates.
        self.P = [0.0] * (self.n * self.n)
        for i in range(self.n):
            scale = self.feature_scales[i]
            self.P[i * self.n + i] = p_init / max(scale * scale, 0.01)

        # Coefficient clamps: [(min, max), ...] for each coefficient
        self.coeff_clamps = coeff_clamps or [None] * self.n

        # Observation counter
        self.observation_count = 0

    def predict(self, x):
        """Predict offset from feature vector.

        Args:
            x: Feature vector [1, x₁, x₂, ...] with leading 1 for intercept.
               Length must equal self.n.

        Returns:
            Predicted offset (float).
        """
        return sum(self.beta[i] * x[i] for i in range(self.n))

    def update(self, x, y):
        """Update coefficients via RLS with one observation.

        Args:
            x: Feature vector [1, x₁, x₂, ...].
            y: Observed offset (float).

        Returns:
            Residual (y - prediction before update).
        """
        n = self.n
        # Prediction error (residual)
        y_pred = self.predict(x)
        residual = y - y_pred

        # Kalman gain: K = P·x / (λ + x'·P·x)
        Px = [sum(self.P[i * n + j] * x[j] for j in range(n)) for i in range(n)]
        xPx = sum(x[i] * Px[i] for i in range(n))

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
                col_j = sum(x[k] * self.P[k * n + j] for k in range(n))
                new_P[i * n + j] = (self.P[i * n + j] - K[i] * col_j) / lam

        # Regularization: prevent covariance windup by adding δ·I each step
        for i in range(n):
            new_P[i * n + i] += self.delta

        self.P = new_P
        self.observation_count += 1

        return residual

    def get_coefficients(self):
        """Return coefficient dict: {index: value}."""
        return {i: self.beta[i] for i in range(self.n)}

    def get_covariance_diagonal(self):
        """Return diagonal of P (uncertainty per coefficient)."""
        return [self.P[i * self.n + i] for i in range(self.n)]

    def as_dict(self):
        """Serialize model state to dict."""
        return {
            "beta": list(self.beta),
            "P": list(self.P),
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, data, n_inputs, **kwargs):
        """Restore model from serialized dict.

        Handles length mismatches when model inputs are added/removed:
        - If stored beta matches current length: restore exactly
        - If shorter (inputs added): restore existing, new inputs use seeds
        - If longer (inputs removed): restore only what fits
        """
        model = cls(n_inputs, **kwargs)
        if "beta" in data:
            beta = data["beta"]
            if len(beta) == model.n:
                # Exact match — restore all
                model.beta = [float(v) for v in beta]
            elif len(beta) < model.n:
                # Inputs were added — restore old coefficients, keep seeds for new
                for i in range(len(beta)):
                    model.beta[i] = float(beta[i])
                _LOGGER.info("RLS restore: stored %d coefficients, model needs %d — seeding new inputs",
                           len(beta), model.n)
            else:
                # Inputs were removed — restore what fits
                for i in range(model.n):
                    model.beta[i] = float(beta[i])
                _LOGGER.info("RLS restore: stored %d coefficients, model needs %d — truncating",
                           len(beta), model.n)
        if "P" in data:
            P = data["P"]
            if len(P) == model.n * model.n:
                model.P = [float(v) for v in P]
            else:
                # Covariance matrix size mismatch — keep initial P (high uncertainty for new inputs)
                _LOGGER.info("RLS restore: covariance matrix size mismatch, using initial P")
        if "observation_count" in data:
            model.observation_count = int(data["observation_count"])
        return model
