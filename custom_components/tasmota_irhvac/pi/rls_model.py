"""Linear feedforward model: deployed coefficients + prediction.

Standalone module — no Home Assistant dependencies.  Holds the feedforward
coefficient vector that PIController applies each tick to predict the HP
setpoint offset.  Coefficients are fit externally by batch WLS (see
``batch_learning``) and written into ``beta`` once per learning cycle; this
class stores them and evaluates predictions.  It does NOT learn online — the
recursive-least-squares ``update`` path was removed in 4d7e77a, leaving batch
WLS as the sole coefficient estimator.  The class name is retained for
persistence/compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)


class RLSModel:
    """Linear feedforward model holding deployed coefficients.

    Predicts the HP setpoint offset from multiple input factors via a linear
    model β·x.  Coefficients are fit externally by batch WLS and assigned to
    ``beta``; this class stores them (in feature-normalized space) and
    evaluates ``predict``.  The historical online recursive-least-squares
    ``update`` was removed in 4d7e77a — see ``batch_learning`` for the
    estimator.

    Each coefficient has:
        - A learned value (``beta``, written from the batch)
        - Optional min/max physical clamps (``coeff_clamps``, applied by the
          batch when it writes new coefficients)
        - An optional freeze flag (``frozen``)
    """

    def __init__(
        self,
        n_inputs: int,
        seed_coefficients: list[float] | None = None,
        coeff_clamps: list[tuple[float, float] | None] | None = None,
        feature_scales: list[float] | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            n_inputs: Number of input features (excluding intercept).
            seed_coefficients: Initial β vector [intercept, β₁, β₂, ...] in
                              physical units. Length n_inputs + 1. Defaults to zeros.
            coeff_clamps: List of (min, max) tuples per coefficient in physical
                         units, or None.
            feature_scales: Typical magnitude of each feature [1, scale₁, ...].
                           Features are normalized by these scales, so all
                           dimensions are O(1). Coefficients are stored
                           internally in normalized space and converted to
                           physical units for prediction and external access.
        """
        self.n: int = n_inputs + 1  # +1 for intercept

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

        # Per-coefficient freeze mask: when True, coefficient is locked
        # (K[i] zeroed so beta[i] and P row/col are unchanged by updates).
        self.frozen: list[bool] = [False] * self.n

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

    def get_coefficient_physical(self, index: int) -> float:
        """Return single coefficient in physical (de-normalized) units."""
        return self.beta[index] / self.feature_scales[index]

    def get_coefficients(self) -> dict[int, float]:
        """Return coefficient dict in physical units: {index: value}."""
        return {i: self.get_coefficient_physical(i) for i in range(self.n)}

    def beta_to_seed(self, index: int) -> float:
        """Convert internal β to seed convention (positive = warms room).

        Seed = -β_physical. Everything funnels through get_coefficient_physical.
        """
        return -self.get_coefficient_physical(index)

    def seed_to_beta(self, index: int, seed: float) -> float:
        """Convert seed (positive = warms room) to internal normalized β.

        β_normalized = -seed * feature_scale.
        """
        return -seed * self.feature_scales[index]

    def rescale_features(self, old_scales: list[float]) -> None:
        """Apply similarity transform when feature scales change.

        Adjusts the (normalized) coefficients so that physical predictions are
        preserved after a scale change.  Call after constructing with new
        scales but before reading predictions.

        Math: beta_norm_new[i] = beta_norm_old[i] * (new_scale[i] / old_scale[i])
        """
        n = self.n
        for i in range(min(len(old_scales), n)):
            ratio = self.feature_scales[i] / old_scales[i] if old_scales[i] != 0 else 1.0
            if abs(ratio - 1.0) < 1e-12:
                continue
            self.beta[i] *= ratio
            self.beta_seed[i] *= ratio

    def as_dict(self) -> dict[str, Any]:
        """Serialize model state to dict (beta in normalized space)."""
        result: dict[str, Any] = {
            "beta": list(self.beta),
            "observation_count": self.observation_count,
            "feature_scales": list(self.feature_scales),
        }
        if any(self.frozen):
            result["frozen"] = list(self.frozen)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], n_inputs: int, **kwargs: Any) -> RLSModel:
        """Restore model from serialized dict.

        Beta is stored in normalized space. Handles length mismatches
        when model inputs are added/removed, and feature scale changes
        (e.g. user edited typical_value between restarts).
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
        if "observation_count" in data:
            model.observation_count = int(data["observation_count"])
        if "frozen" in data:
            frozen = data["frozen"]
            for i in range(min(len(frozen), model.n)):
                model.frozen[i] = bool(frozen[i])
        # Detect feature scale changes and apply similarity transform so
        # beta remains consistent with the new normalization.
        old_scales = data.get("feature_scales")
        if old_scales and len(old_scales) == model.n:
            scales_changed = any(
                abs(old_scales[i] - model.feature_scales[i]) > 1e-12
                for i in range(model.n)
            )
            if scales_changed:
                _LOGGER.info(
                    "RLS restore: feature scales changed, applying similarity transform"
                )
                model.rescale_features(old_scales)
        return model
