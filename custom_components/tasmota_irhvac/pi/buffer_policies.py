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
        n = len(x)
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

        if _NUMPY_AVAILABLE and m > 50:
            X = np.array(feature_vectors, dtype=np.float64)
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


class MinEigPolicy:
    """Concurrent-learning admission via minimum-eigenvalue increase.

    Admits a candidate iff the marginal gain it would deliver to
    ``λ_min(X^T X)`` exceeds the marginal loss from evicting the least-
    valuable incumbent.  The minimum eigenvalue of the regressor stack
    bounds the worst-case parameter estimation variance, so this policy
    grows the buffer along the direction that's currently least-
    explored — exactly the conditioning concern that leverage scoring
    only addresses indirectly.

    Score (candidate): ``λ_min(X^T X + xx^T) − λ_min(X^T X)`` — the
    marginal increase from admitting ``x``.

    Evictee: argmin over incumbents ``i`` of
    ``λ_min(X^T X) − λ_min(X^T X − xᵢxᵢ^T)`` — the obs whose removal
    hurts ``λ_min`` the LEAST.

    Admit iff candidate gain > evictee removal cost (net improvement to
    the worst-conditioned eigenvalue).

    Numerical: relies on numpy's ``eigvalsh``.  For ``find_evictee`` it
    runs ``m+1`` symmetric eigenvalue decompositions on an ``n × n``
    matrix per full-buffer admission attempt — fine for the small ``n``
    (≤10) feature counts used here, but quadratic in ``m`` would be
    expensive at the buffer-add cadence; see ``score_candidate`` notes.

    The ``λI`` regularization that the buffer adds to ``X^T X`` shifts
    every eigenvalue by ``λ``; the *gain* and *cost* are invariant to
    that shift, so we can operate directly on ``_xtx_matrix``.

    References:
    - Chowdhary, G. & Johnson, E. N. (2011), ACC — concurrent learning
      history stack with eigenvalue-driven admission.
    - Chowdhary, G., Mühlegg, M. & Johnson, E. N. (2014) — exponential
      parameter convergence without persistent excitation, given the
      eigenvalue admission rule.
    """

    name = "min_eig"

    def _min_eig(self, M: list[list[float]]) -> float:
        """Smallest eigenvalue of an n×n symmetric matrix.

        Uses numpy's eigvalsh when available.  The pure-Python fallback
        delegates to the buffer's _eigenvalues_symmetric helper to avoid
        duplicating the n×n eigendecomposition logic.
        """
        if _NUMPY_AVAILABLE:
            return float(np.linalg.eigvalsh(np.asarray(M, dtype=np.float64))[0])
        # Local import keeps the policy module decoupled at load time.
        from .batch_learning import DiversityAwareBuffer
        eigs = DiversityAwareBuffer._eigenvalues_symmetric(M, len(M))
        if eigs is None:
            return float("nan")
        return min(eigs)

    def score_candidate(
        self,
        x: list[float],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
        n_buffered: int,
    ) -> float:
        if xtx is None:
            return 0.0
        n = len(x)
        base = self._min_eig(xtx)
        # Augmented = xtx + outer(x, x).
        aug = [
            [xtx[i][j] + x[i] * x[j] for j in range(n)]
            for i in range(n)
        ]
        new = self._min_eig(aug)
        return new - base

    def find_evictee(
        self,
        observations: list[Observation],
        feature_vectors: list[list[float]],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
    ) -> EvicteeChoice:
        m = len(feature_vectors)
        if m == 0 or xtx is None:
            return EvicteeChoice(index=-1, score=float("inf"))

        n = len(feature_vectors[0])
        if _NUMPY_AVAILABLE:
            XtX = np.asarray(xtx, dtype=np.float64)
            base = float(np.linalg.eigvalsh(XtX)[0])
            best_idx = 0
            best_cost = float("inf")
            for i, xi in enumerate(feature_vectors):
                xi_arr = np.asarray(xi, dtype=np.float64)
                removed = XtX - np.outer(xi_arr, xi_arr)
                new_eig = float(np.linalg.eigvalsh(removed)[0])
                cost = base - new_eig
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            return EvicteeChoice(index=best_idx, score=best_cost)

        base = self._min_eig(xtx)
        best_idx = 0
        best_cost = float("inf")
        for i, xi in enumerate(feature_vectors):
            removed = [
                [xtx[a][b] - xi[a] * xi[b] for b in range(n)]
                for a in range(n)
            ]
            new_eig = self._min_eig(removed)
            cost = base - new_eig
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        return EvicteeChoice(index=best_idx, score=best_cost)

    def should_admit(
        self,
        candidate_score: float,
        evictee_score: float,
    ) -> bool:
        return candidate_score > evictee_score
