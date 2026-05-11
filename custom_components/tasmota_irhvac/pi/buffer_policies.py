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

import random
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


@dataclass(frozen=True, slots=True)
class ExchangeChoice:
    """Outcome of a joint candidate↔incumbent admission evaluation.

    Returned by the optional ``BufferPolicy.attempt_exchange`` method
    when the policy's admission decision depends on the candidate↔
    incumbent interaction rather than independent scores (e.g. Fedorov
    D-optimal exchange where the optimal evictee depends on which
    incumbent's removal best complements the candidate).

    Fields:
      admit: True iff the policy decided to admit the candidate.
      evictee_index: index of the displaced incumbent.  Use ``-1`` when
          ``admit`` is False (no exchange chosen).
      candidate_score, evictee_score: policy-defined scalar metrics for
          BufferAddResult observability — the meaning is policy-specific
          (typically leverage or λ_min gain) but should be comparable
          across consecutive ticks of the same policy.
      rejection_reason: short label when ``admit`` is False; the buffer
          falls back to ``f"{policy.name}_rejected"`` if None is given.
    """

    admit: bool
    evictee_index: int
    candidate_score: float | None
    evictee_score: float | None
    rejection_reason: str | None = None


class BufferPolicy(Protocol):
    """Strategy interface for ``DiversityAwareBuffer`` admit/evict decisions.

    Required methods (all policies):
      - ``score_candidate``: scalar metric for an incoming observation
      - ``find_evictee``:    pick the lowest-value incumbent
      - ``should_admit``:    compare two scalar scores

    Optional method (joint-optimization policies):
      - ``attempt_exchange``: evaluate the candidate↔incumbent swap
        jointly.  Used by ``DOptimalPolicy`` (Fedorov exchange) where
        the optimal evictee depends on the candidate.

    The buffer's ``add()`` dispatches on ``hasattr(policy,
    'attempt_exchange')``: if present, the policy's joint decision wins;
    otherwise the simple ``score_candidate`` / ``find_evictee`` /
    ``should_admit`` path runs.  This split lets the simple policies
    (LeveragePolicy, MinEigPolicy, SlidingWindowPolicy) stay decoupled
    while joint policies opt in.

    Long-term: if a joint-optimization policy ends up being the
    production choice, refactor to a single ``evaluate_admission``
    method that all policies implement uniformly — that removes the
    two-paths-in-add() dispatch.  See branch ``buffer-policy-strategy``
    discussion (commit 3) for the analysis.
    """

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


class DOptimalPolicy(LeveragePolicy):
    """D-optimal sequential exchange (Fedorov 1972).

    Maximizes ``det(X^T X)`` via joint candidate↔incumbent exchange.
    For each incumbent ``xᵢ`` and the candidate ``x``, evaluates the
    determinant ratio of the swap:

        ratio_i = (1 − ℓᵢ) · (1 + ℓ(x) + cross_i² / (1 − ℓᵢ))
                = (1 − ℓᵢ)(1 + ℓ(x)) + cross_i²

    where:
        ℓᵢ      = xᵢᵀ A⁻¹ xᵢ           (incumbent leverage)
        ℓ(x)    = xᵀ  A⁻¹ x            (candidate leverage)
        cross_i = xᵀ  A⁻¹ xᵢ           (bilinear cross term)

    The cross term distinguishes Fedorov from LeveragePolicy: when ``x``
    and ``xᵢ`` are correlated (large cross_i), the exchange is more
    valuable than leverage scoring alone would suggest — sometimes
    favoring eviction of an incumbent whose own leverage is high but
    whose direction is already covered by the candidate.

    Admit iff ``max_i ratio_i > 1`` (strictly positive log-det gain).
    Evictee = argmax of ``ratio_i``.

    Inherits ``score_candidate`` / ``find_evictee`` / ``should_admit``
    from LeveragePolicy as fallbacks for the buffer-not-full path; the
    joint logic only runs when the buffer is full and add() routes to
    ``attempt_exchange``.

    References:
    - Fedorov, V. V. (1972), "Theory of Optimal Experiments"
    - Mitchell, T. J. (1974), "An algorithm for the construction of
      D-optimal experimental designs" (DETMAX)
    - Atkinson, A. C. & Donev, A. N. (1992), "Optimum Experimental Designs"
    """

    name = "d_optimal"

    def attempt_exchange(
        self,
        candidate: list[float],
        observations: list[Observation],
        feature_vectors: list[list[float]],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
    ) -> ExchangeChoice | None:
        m = len(feature_vectors)
        if m == 0:
            return None  # buffer empty — buffer's add() handles unconditional admission

        if _NUMPY_AVAILABLE:
            x = np.asarray(candidate, dtype=np.float64)
            Ainv = np.asarray(info_inv, dtype=np.float64)
            X_inc = np.asarray(feature_vectors, dtype=np.float64)

            Ainv_x = Ainv @ x
            leverage_x = float(x @ Ainv_x)

            # Per-incumbent leverage and bilinear cross term.
            Ainv_Xinc = X_inc @ Ainv  # (m, n)
            leverage_i = np.einsum("ij,ij->i", X_inc, Ainv_Xinc)  # (m,)
            cross_i = X_inc @ Ainv_x  # (m,)

            # Determinant-ratio formula for the joint swap.  Equivalent to
            # log-det gain via log(ratio_i).  Maximize ratio_i directly.
            ratios = (1.0 - leverage_i) * (1.0 + leverage_x) + cross_i**2

            best_idx = int(np.argmax(ratios))
            best_ratio = float(ratios[best_idx])
            evictee_lev = float(leverage_i[best_idx])
        else:
            n = len(candidate)
            leverage_x = sum(
                candidate[i] * sum(info_inv[i][j] * candidate[j] for j in range(n))
                for i in range(n)
            )
            best_idx = 0
            best_ratio = -float("inf")
            evictee_lev = 0.0
            for k in range(m):
                xi = feature_vectors[k]
                # leverage_i = xᵢᵀ A⁻¹ xᵢ
                Ainv_xi = [
                    sum(info_inv[i][j] * xi[j] for j in range(n))
                    for i in range(n)
                ]
                ell_i = sum(xi[i] * Ainv_xi[i] for i in range(n))
                # cross_i = xᵀ A⁻¹ xᵢ
                cross = sum(candidate[i] * Ainv_xi[i] for i in range(n))
                ratio = (1.0 - ell_i) * (1.0 + leverage_x) + cross * cross
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = k
                    evictee_lev = ell_i

        if best_ratio > 1.0:
            return ExchangeChoice(
                admit=True,
                evictee_index=best_idx,
                candidate_score=leverage_x,
                evictee_score=evictee_lev,
                rejection_reason=None,
            )
        return ExchangeChoice(
            admit=False,
            evictee_index=-1,
            candidate_score=leverage_x,
            evictee_score=evictee_lev,
            rejection_reason=None,  # buffer fills in f"{policy.name}_rejected"
        )


class SlidingWindowPolicy:
    """FIFO sliding-window admission — keep the most-recent ``max_size``
    observations, evict the oldest.

    Score (candidate): the candidate's monotonic timestamp; newer = higher.
    Evictee: the incumbent with the smallest timestamp (oldest).
    Admit: always (FIFO never rejects when buffer is full).

    Mirrors the historical ``FIFOBuffer`` ad-hoc subclass in
    ``tests/hvac_bench/scenarios/test_buffer_variants.py`` (now
    deprecated — use ``policy=SlidingWindowPolicy()`` instead).

    Recency-biased: discards old observations regardless of how rare
    or informative their operating regime was.  Compare with
    LeveragePolicy/DOptimalPolicy/MinEigPolicy which retain rare
    regimes and shed redundant ones.

    Reference: Fortescue, T. R. (1981) — "Implementation of self-tuning
    regulators with variable forgetting factors", Automatica.  The
    sliding window is the hard-cutoff special case of forgetting; for
    a soft variant, see future ``VFFPolicy``.
    """

    name = "sliding_window"

    def score_candidate(
        self,
        x: list[float],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
        n_buffered: int,
    ) -> float:
        # The candidate's timestamp is its score; the buffer passes the
        # observation's timestamp implicitly via the ordering of add().
        # Concretely: the buffer's add() always appends in arrival order,
        # so the *latest* candidate has the highest implicit "timestamp"
        # by construction.  Returning a positive score guarantees admission
        # under the should_admit comparator below.
        return float("inf")

    def find_evictee(
        self,
        observations: list[Observation],
        feature_vectors: list[list[float]],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
    ) -> EvicteeChoice:
        m = len(observations)
        if m == 0:
            return EvicteeChoice(index=-1, score=float("inf"))
        oldest_idx = 0
        oldest_ts = observations[0].timestamp
        for i in range(1, m):
            if observations[i].timestamp < oldest_ts:
                oldest_ts = observations[i].timestamp
                oldest_idx = i
        return EvicteeChoice(index=oldest_idx, score=oldest_ts)

    def should_admit(
        self,
        candidate_score: float,
        evictee_score: float,
    ) -> bool:
        # Always admit when the buffer is full.  The candidate's
        # implicit "now" timestamp is always > the evictee's stored
        # timestamp by the time add() is called.
        return True


class AOptimalPolicy:
    """A-optimal sequential admission — minimize ``trace((X^T X)⁻¹)``.

    The trace of ``(X^T X)⁻¹`` equals the sum of variances of OLS
    coefficient estimates, so A-optimality directly minimizes the
    *average* parameter-estimation variance.  In contrast,
    LeveragePolicy/DOptimalPolicy maximize ``det`` (geometric mean of
    eigenvalues) and MinEigPolicy maximizes ``λ_min`` (worst-case
    direction).  A-optimal trades worst-case protection for average-
    case efficiency.

    By Sherman-Morrison, the trace gain from admitting ``x`` is

        trace(A⁻¹) − trace((A + xx^T)⁻¹)
            = trace(A⁻¹ x x^T A⁻¹) / (1 + x^T A⁻¹ x)
            = ‖A⁻¹ x‖² / (1 + ℓ(x))

    The trace cost from removing incumbent ``xᵢ`` is

        trace((A − xᵢxᵢ^T)⁻¹) − trace(A⁻¹) = ‖A⁻¹ xᵢ‖² / (1 − ℓᵢ)

    Both depend only on individual incumbents, so A-optimal uses the
    simple BufferPolicy interface (``score_candidate`` /
    ``find_evictee`` / ``should_admit``) — no joint exchange.  Admit
    iff candidate gain > evictee removal cost.

    References:
    - Atkinson, A. C. & Donev, A. N. (1992) — A-optimal design
    - Pukelsheim, F. (1993), "Optimal Design of Experiments" — A vs. D
      vs. E criteria comparison
    """

    name = "a_optimal"

    def _trace_change(
        self,
        x: list[float],
        info_inv: list[list[float]],
        denom_sign: float,
    ) -> float:
        """``‖A⁻¹ x‖² / (1 + denom_sign · ℓ(x))``.

        denom_sign = +1 for admission (gain), −1 for removal (cost).
        Both use the same numerator and the same leverage ``ℓ(x)`` on
        the current matrix; the sign flip captures whether we're adding
        or subtracting the rank-1 update.

        Returns ``inf`` when the denominator is ≤ 0: for removal this
        flags the singular regime (``ℓ(x) ≥ 1``) where ``A − xx^T`` is
        not positive semidefinite — removing such a point would
        destroy the matrix's invertibility, so the cost is infinite.
        Production buffers maintain ``ℓ < 1`` for buffered points via
        regularization (Cook 1977), but defensive guarding keeps the
        formula well-defined under degenerate hand-constructed inputs.
        """
        n = len(x)
        Ainv_x = [
            sum(info_inv[i][j] * x[j] for j in range(n))
            for i in range(n)
        ]
        leverage = sum(x[i] * Ainv_x[i] for i in range(n))
        denom = 1.0 + denom_sign * leverage
        if denom <= 1e-15:
            return float("inf")
        numerator = sum(v * v for v in Ainv_x)
        return numerator / denom

    def score_candidate(
        self,
        x: list[float],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
        n_buffered: int,
    ) -> float:
        return self._trace_change(x, info_inv, denom_sign=+1.0)

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
            X = np.asarray(feature_vectors, dtype=np.float64)
            Ainv = np.asarray(info_inv, dtype=np.float64)
            Ainv_X = X @ Ainv  # (m, n)
            leverages = np.einsum("ij,ij->i", X, Ainv_X)  # (m,)
            row_norms_sq = np.einsum("ij,ij->i", Ainv_X, Ainv_X)  # ‖Ainv xᵢ‖²
            denoms = 1.0 - leverages
            with np.errstate(divide="ignore", invalid="ignore"):
                costs = np.where(
                    denoms > 1e-15,
                    row_norms_sq / denoms,
                    np.inf,
                )
            idx = int(np.argmin(costs))
            return EvicteeChoice(index=idx, score=float(costs[idx]))

        min_idx = 0
        min_cost = self._trace_change(feature_vectors[0], info_inv, -1.0)
        for i in range(1, m):
            cost = self._trace_change(feature_vectors[i], info_inv, -1.0)
            if cost < min_cost:
                min_cost = cost
                min_idx = i
        return EvicteeChoice(index=min_idx, score=min_cost)

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


class SlevPolicy:
    """Shrunken-leverage probabilistic admission (Ma-Mahoney-Yu 2015 SLEV).

    Streaming-adapted form of MMY's offline SLEV: admission probability is
    a convex combination of leverage and uniform.

    Per-call score:
        score(x) = α · leverage(x) + (1−α) · noise · noise_scale

    where ``noise`` is a fresh ``U(0, 1)`` draw and ``noise_scale`` is the
    maximum incumbent leverage (so the noise component matches the
    leverage component's scale dynamically — at α=0.5 they have roughly
    equal influence on ranking).

    α=1: reduces to LeveragePolicy.  α=0: pure random scoring → eviction
    is uniform over incumbents → uniform-random retention in the limit.
    α∈(0, 1): mix.

    Why probabilistic helps: deterministic top-N selection (any criterion)
    biases parameter estimation by concentrating selection on extremes
    (Ma-Mahoney-Yu 2015 — algorithmic leveraging pathology).  Probabilistic
    selection avoids the bias by giving interior observations non-zero
    retention probability.

    Caveat vs offline MMY: the offline form gives each observation a
    persistent ``π_i`` and samples without replacement.  This streaming
    form uses ephemeral noise per call — captures the spirit (probabilistic
    admission with leverage bias modulated by α) but isn't algebraically
    equivalent.  Empirical behavior in single-season offline testing
    matched: at any α ∈ [0, 1], probabilistic SLEV recovered β_solar to
    within 0.08 of truth on the spring 90d real-CSV corpus.

    References:
    - Ma, P., Mahoney, M. W. & Yu, B. (2015), JMLR 16 — A Statistical
      Perspective on Algorithmic Leveraging.  SLEV (eq. 4) and its bias-
      variance analysis.
    """

    name = "slev"

    def __init__(self, alpha: float = 0.5, seed: int | None = 0) -> None:
        # seed=0 (not None): SLEV admission is pseudo-random across
        # observations within a buffer (the policy exists to break the
        # deterministic top-N leverage bias), but the draw sequence must
        # be reproducible across processes. ``random.Random(None)`` seeds
        # from os.urandom — different per process — which propagates
        # non-determinism into buffer composition, WLS β, FF, and integral.
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        self.alpha = alpha
        self._rng = random.Random(seed)

    def _leverage(
        self,
        x: list[float],
        info_inv: list[list[float]],
    ) -> float:
        n = len(x)
        inv_x = [
            sum(info_inv[i][j] * x[j] for j in range(n))
            for i in range(n)
        ]
        return max(0.0, sum(x[i] * inv_x[i] for i in range(n)))

    def score_candidate(
        self,
        x: list[float],
        info_inv: list[list[float]],
        xtx: list[list[float]] | None,
        n_buffered: int,
    ) -> float:
        """Score a candidate.  Noise scale is set by the candidate's own
        leverage; for the eviction comparison, ``find_evictee`` re-scales
        consistently against incumbent leverages, so the candidate score
        here is a placeholder used only when the buffer isn't full.
        """
        leverage = self._leverage(x, info_inv)
        noise = self._rng.random() * max(leverage, 1e-9)
        return self.alpha * leverage + (1.0 - self.alpha) * noise

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
            X = np.asarray(feature_vectors, dtype=np.float64)
            inv = np.asarray(info_inv, dtype=np.float64)
            leverages = np.einsum("ij,jk,ik->i", X, inv, X)
            leverages = np.clip(leverages, 0.0, None)
            # Dynamic noise scaling: match the incumbent leverage range.
            noise_scale = float(leverages.max()) if leverages.max() > 1e-12 else 1.0
            noise = np.array(
                [self._rng.random() for _ in range(m)], dtype=np.float64
            ) * noise_scale
            scores = self.alpha * leverages + (1.0 - self.alpha) * noise
            idx = int(np.argmin(scores))
            return EvicteeChoice(index=idx, score=float(scores[idx]))

        # Pure-Python fallback
        leverages = [self._leverage(x, info_inv) for x in feature_vectors]
        noise_scale = max(leverages) if max(leverages) > 1e-12 else 1.0
        scores = [
            self.alpha * leverages[i]
            + (1.0 - self.alpha) * self._rng.random() * noise_scale
            for i in range(m)
        ]
        min_idx = min(range(m), key=lambda i: scores[i])
        return EvicteeChoice(index=min_idx, score=scores[min_idx])

    def should_admit(
        self,
        candidate_score: float,
        evictee_score: float,
    ) -> bool:
        return candidate_score > evictee_score
