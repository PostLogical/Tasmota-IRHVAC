"""Admit/evict policies for ``DiversityAwareBuffer``.

A buffer policy decides three things given current buffer state:

  - ``score_candidate``: how valuable is a new candidate observation?
  - ``find_evictee``:   which incumbent is least valuable?
  - ``should_admit``:    given the two scores, does the candidate displace
                         the evictee?

Policies are stateless apart from their ``name``.  The buffer owns
storage, info-matrix maintenance, and Sherman-Morrison up/downdates;
the policy only sees vectors and matrices for the decision.

References:
- Atkinson & Donev, "Optimum Experimental Designs" — D-optimal sequential design
- Chowdhary & Johnson, ACC 2011 — concurrent learning history stack
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

try:  # pragma: no cover — exercised by base buffer
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False

if TYPE_CHECKING:
    from .batch_learning import Observation


@dataclass(frozen=True, slots=True)
class EvicteeChoice:
    """Result of identifying the lowest-value incumbent."""

    index: int
    score: float


class BufferPolicy(Protocol):
    """Strategy interface for ``DiversityAwareBuffer`` admit/evict decisions."""

    name: str

    def score_candidate(
        self,
        x: list[float],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
        n_buffered: int,
    ) -> float:
        """Score a candidate observation given current buffer state."""

    def find_evictee(
        self,
        observations: list[Observation],
        feature_vectors: list[list[float]],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
    ) -> EvicteeChoice:
        """Return the lowest-value incumbent's index and score."""

    def should_admit(
        self,
        candidate_score: float,
        evictee_score: float,
    ) -> bool:
        """Decide whether the candidate displaces the evictee."""


class LeveragePolicy:
    """D-optimal sequential admission via leverage scoring.

    Score: ``x^T (X^T X + λI)^{-1} x`` — the diagonal of the hat matrix.
    Admits iff the candidate's leverage exceeds the lowest incumbent's,
    so the buffer accumulates points that locally maximize ``det(X^T X)``.

    Equivalent to the original ``DiversityAwareBuffer`` admission rule;
    this is the baseline policy that preserves prior behavior.
    """

    name = "leverage"

    def score_candidate(
        self,
        x: list[float],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
        n_buffered: int,
    ) -> float:
        # Use the info matrix's dimension as authoritative — feature
        # vectors longer than n_features are truncated (matches the
        # historical _compute_leverage behavior).
        n = len(info_inv)
        inv_x = [
            sum(info_inv[i][j] * x[j] for j in range(n))
            for i in range(n)
        ]
        return sum(x[i] * inv_x[i] for i in range(n))

    def find_evictee(
        self,
        observations: list[Observation],
        feature_vectors: list[list[float]],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
    ) -> EvicteeChoice:
        m = len(feature_vectors)
        if m == 0:
            return EvicteeChoice(index=-1, score=float("inf"))

        n = len(info_inv)
        if _NUMPY_AVAILABLE and m > 50:
            # Slice each feature vector to ``n`` columns to match the
            # info matrix dimension (silent truncation, as in the legacy
            # implementation).
            X = np.array(
                [fv[:n] for fv in feature_vectors], dtype=np.float64,
            )
            inv = np.array(info_inv, dtype=np.float64)
            scores = np.einsum("ij,jk,ik->i", X, inv, X)
            idx = int(np.argmin(scores))
            return EvicteeChoice(index=idx, score=float(scores[idx]))

        min_idx = 0
        min_score = self.score_candidate(feature_vectors[0], info_inv, xtx, 0)
        for i in range(1, m):
            s = self.score_candidate(feature_vectors[i], info_inv, xtx, 0)
            if s < min_score:
                min_score = s
                min_idx = i
        return EvicteeChoice(index=min_idx, score=min_score)

    def should_admit(
        self,
        candidate_score: float,
        evictee_score: float,
    ) -> bool:
        return candidate_score > evictee_score
