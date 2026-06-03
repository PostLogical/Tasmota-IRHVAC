"""Unit tests for the ``BufferPolicy`` abstraction and built-in policies."""

from __future__ import annotations

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    DiversityAwareBuffer,
    Observation,
)
from custom_components.tasmota_irhvac.pi.buffer_policies import (
    AOptimalPolicy,
    DOptimalPolicy,
    EvicteeChoice,
    ExchangeChoice,
    LeveragePolicy,
    MinEigPolicy,
    SlevPolicy,
    SlidingWindowPolicy,
    TimeWindowPolicy,
)


# Minimal feature config: intercept + outdoor_delta + one model input.
TEST_FEATURE_ORDER = ["intercept", "outdoor_delta", "Solar Proxy"]
TEST_MODEL_INPUTS = [
    {
        "name": "Solar Proxy",
        "entity_id": "sensor.solar",
        "lag_tau": 0.0,
        "delta_from_room": False,
    },
]


def _make_obs(t: float, outdoor: float = 5.0, solar: float = 0.0) -> Observation:
    return Observation(
        timestamp=t, wall_time=t,
        hp_setpoint=21.0, current_c=20.0, desired_c=20.0,
        outdoor_temp_c=outdoor, room_rate=0.0,
        raw_readings={"sensor.solar": solar},
        clamped=False,
    )


# ── LeveragePolicy: scoring contract ─────────────────────────────────


class TestLeveragePolicyScoring:
    def test_score_candidate_is_quadratic_form(self):
        """``score_candidate(x, A^{-1}, ...)`` evaluates ``x^T A^{-1} x``."""
        policy = LeveragePolicy()
        # 2×2 inverse info matrix with known structure.
        info_inv = [[2.0, 0.5], [0.5, 3.0]]
        x = [1.0, 2.0]
        # Expected: 1*2*1 + 1*0.5*2 + 2*0.5*1 + 2*3*2 = 2 + 1 + 1 + 12 = 16
        assert policy.score_candidate(x, info_inv, None, 0) == pytest.approx(16.0)

    def test_score_is_zero_for_zero_vector(self):
        policy = LeveragePolicy()
        info_inv = [[1.0, 0.0], [0.0, 1.0]]
        assert policy.score_candidate([0.0, 0.0], info_inv, None, 0) == 0.0

    def test_should_admit_strict_inequality(self):
        """Equal candidate and incumbent scores → reject (matches old rule)."""
        policy = LeveragePolicy()
        assert policy.should_admit(0.5, 0.4) is True
        assert policy.should_admit(0.5, 0.5) is False
        assert policy.should_admit(0.4, 0.5) is False


# ── LeveragePolicy: evictee selection ────────────────────────────────


class TestLeveragePolicyFindEvictee:
    def test_empty_buffer_returns_sentinel(self):
        """No incumbents → sentinel index -1, +inf score."""
        policy = LeveragePolicy()
        info_inv = [[1.0, 0.0], [0.0, 1.0]]
        choice = policy.find_evictee([], [], info_inv, None)
        assert choice.index == -1
        assert choice.score == float("inf")

    def test_picks_lowest_scoring_incumbent_pure_python(self):
        """Below the numpy threshold (50), pure-Python loop finds the min."""
        policy = LeveragePolicy()
        info_inv = [[1.0, 0.0], [0.0, 1.0]]
        # Three feature vectors with hand-computable scores: 5, 1, 25.
        feature_vectors = [
            [1.0, 2.0],   # score = 5
            [1.0, 0.0],   # score = 1  ← min
            [3.0, 4.0],   # score = 25
        ]
        observations = [_make_obs(t=float(i)) for i in range(3)]
        choice = policy.find_evictee(observations, feature_vectors, info_inv, None)
        assert choice.index == 1
        assert choice.score == pytest.approx(1.0)

    def test_picks_lowest_scoring_incumbent_numpy_vectorized(self):
        """Above the numpy threshold (>50 stack), vectorized path agrees."""
        policy = LeveragePolicy()
        info_inv = [[1.0, 0.0], [0.0, 1.0]]
        feature_vectors = [[1.0, float(i)] for i in range(60)]  # score = 1 + i^2
        observations = [_make_obs(t=float(i)) for i in range(60)]
        choice = policy.find_evictee(observations, feature_vectors, info_inv, None)
        assert choice.index == 0  # i=0 gives the smallest score (=1)
        assert choice.score == pytest.approx(1.0)


# ── Policy injection at the buffer level ─────────────────────────────


class TestBufferPolicyInjection:
    def test_default_policy_is_slev(self):
        """Buffer constructed without a policy uses SlevPolicy(alpha=0.0).

        Default switched from leverage on 2026-05-07 after the leverage-
        curation β_solar bias finding (`project_bench_solar_fidelity.md`).
        Uniform-random retention is the bench-validated unbiased baseline;
        leverage stays available for explicit selection.
        """
        buf = DiversityAwareBuffer(
            n_features=3, max_size=5,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
        )
        r = buf.add(_make_obs(t=0.0))
        assert r.policy_name == "slev"

    def test_explicit_leverage_policy_overrides_default(self):
        """Passing a LeveragePolicy() explicitly opts out of the slev default."""
        buf_default = DiversityAwareBuffer(
            n_features=3, max_size=5,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
        )
        buf_explicit = DiversityAwareBuffer(
            n_features=3, max_size=5,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=LeveragePolicy(),
        )
        r_def = buf_default.add(_make_obs(t=0.0, outdoor=5.0))
        r_exp = buf_explicit.add(_make_obs(t=0.0, outdoor=5.0))
        assert r_def.policy_name == "slev"
        assert r_exp.policy_name == "leverage"


# ── EvicteeChoice value type ─────────────────────────────────────────


class TestEvicteeChoice:
    def test_is_frozen(self):
        """Mutation is rejected — policy results should be immutable."""
        choice = EvicteeChoice(index=2, score=0.5)
        with pytest.raises(AttributeError):
            choice.index = 5  # type: ignore[misc]


# ── MinEigPolicy: scoring contract ───────────────────────────────────


class TestMinEigPolicyScoring:
    """Direct tests against the score formula on hand-constructed matrices.

    For a candidate x and forward matrix XtX, the score is
    λ_min(XtX + xx^T) − λ_min(XtX).
    """

    def test_score_in_unexplored_direction_improves_min_eig(self):
        """Adding a point along the worst-conditioned direction raises λ_min.

        XtX = diag(10, 0.1) — eigenvalues are 10 and 0.1.  Adding a point
        x = [0, 1] (purely in the second direction) lifts λ_min from 0.1
        toward 1.1, so the score should be positive.
        """
        policy = MinEigPolicy()
        xtx = [[10.0, 0.0], [0.0, 0.1]]
        score = policy.score_candidate([0.0, 1.0], info_inv=[], xtx=xtx, n_buffered=10)
        assert score > 0.5  # 1.1 - 0.1 = 1.0; with eigvalsh accuracy

    def test_score_in_already_strong_direction_barely_helps(self):
        """Adding a point along the well-conditioned direction barely changes λ_min."""
        policy = MinEigPolicy()
        xtx = [[10.0, 0.0], [0.0, 0.1]]
        score = policy.score_candidate([1.0, 0.0], info_inv=[], xtx=xtx, n_buffered=10)
        assert score < 0.01  # λ_min stays ≈ 0.1; only λ_max moves

    def test_score_zero_vector_yields_zero(self):
        """Zero candidate adds no information."""
        policy = MinEigPolicy()
        xtx = [[5.0, 0.0], [0.0, 5.0]]
        score = policy.score_candidate([0.0, 0.0], info_inv=[], xtx=xtx, n_buffered=10)
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_score_returns_zero_when_xtx_unavailable(self):
        """Empty buffers / pre-init paths return 0 score."""
        policy = MinEigPolicy()
        score = policy.score_candidate([1.0, 1.0], info_inv=[], xtx=None, n_buffered=0)
        assert score == 0.0

    def test_should_admit_strict_inequality(self):
        policy = MinEigPolicy()
        assert policy.should_admit(0.5, 0.4) is True
        assert policy.should_admit(0.5, 0.5) is False
        assert policy.should_admit(0.4, 0.5) is False


# ── MinEigPolicy: evictee selection ──────────────────────────────────


class TestMinEigPolicyFindEvictee:
    """Cheapest-to-evict = obs whose removal hurts λ_min the least."""

    def test_empty_buffer_returns_sentinel(self):
        policy = MinEigPolicy()
        choice = policy.find_evictee(
            observations=[], feature_vectors=[],
            info_inv=[], xtx=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert choice.index == -1
        assert choice.score == float("inf")

    def test_picks_redundant_incumbent_over_unique_one(self):
        """When 4 incumbents share a direction and 1 is unique, the unique
        one is the LAST we should evict — its removal hurts λ_min most.
        find_evictee returns the cheapest, which is one of the redundant.
        """
        policy = MinEigPolicy()
        # Three points along x=[1,0] and one along x=[0,1].
        # XtX = 3 * [[1,0],[0,0]] + [[0,0],[0,1]] = [[3,0],[0,1]]
        feature_vectors = [
            [1.0, 0.0],  # one of three redundant
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],  # the unique direction
        ]
        xtx = [[3.0, 0.0], [0.0, 1.0]]
        # base λ_min = 1.0 (the unique direction).
        # Removing a redundant: XtX → [[2,0],[0,1]], λ_min still 1.0 → cost 0.
        # Removing the unique: XtX → [[3,0],[0,0]], λ_min → 0 → cost 1.0.
        # So evictee should be one of indices 0, 1, 2 (cost 0).
        observations = []  # not used by min_eig
        choice = policy.find_evictee(observations, feature_vectors, info_inv=[], xtx=xtx)
        assert choice.index in (0, 1, 2)
        assert choice.score == pytest.approx(0.0, abs=1e-9)


# ── MinEigPolicy: integration through DiversityAwareBuffer ───────────


class TestMinEigBufferIntegration:
    """End-to-end behavior with a buffer using MinEigPolicy."""

    def _buf(self, max_size: int) -> DiversityAwareBuffer:
        return DiversityAwareBuffer(
            n_features=3, max_size=max_size,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=MinEigPolicy(),
        )

    def test_buffer_reports_min_eig_policy_name(self):
        buf = self._buf(max_size=5)
        r = buf.add(_make_obs(t=0.0, outdoor=5.0))
        assert r.policy_name == "min_eig"

    def test_full_buffer_admits_complementary_direction(self):
        """A candidate spanning a previously-unexplored direction wins
        admission against a buffer of redundant incumbents."""
        buf = self._buf(max_size=4)
        # Fill with redundant observations (all in the same direction).
        for i in range(4):
            buf.add(_make_obs(t=float(i), outdoor=5.0, solar=0.0))
        # A novel observation with strong solar reading explores a new
        # direction in feature space.
        r = buf.add(_make_obs(t=10.0, outdoor=5.0, solar=10.0))
        assert r.admitted is True
        assert r.policy_name == "min_eig"
        assert r.candidate_score > r.min_incumbent_score

    def test_buffer_full_decision_returns_min_eig_score_metadata(self):
        """When the buffer is full, the result reports per-policy scores
        and (on rejection) a ``min_eig_rejected`` literal — validates
        wiring without over-specifying eigenvalue arithmetic."""
        buf = self._buf(max_size=3)
        buf.add(_make_obs(t=0.0, outdoor=5.0, solar=1.0))
        buf.add(_make_obs(t=1.0, outdoor=-5.0, solar=2.0))
        buf.add(_make_obs(t=2.0, outdoor=0.0, solar=10.0))
        r = buf.add(_make_obs(t=3.0, outdoor=2.0, solar=5.0))
        # We don't assert admit/reject here — depends on eigenvalue
        # arithmetic that the policy unit tests already cover.  We assert
        # the result *shape*: scores populated, policy name reported,
        # rejection literal (if any) carries the policy name.
        assert r.policy_name == "min_eig"
        assert r.candidate_score is not None
        assert r.min_incumbent_score is not None
        if not r.admitted:
            assert r.rejection_reason == "min_eig_rejected"


# ── DOptimalPolicy: Fedorov-exchange contract ────────────────────────


class TestDOptimalPolicyExchange:
    """``attempt_exchange`` realizes the Fedorov D-optimal swap criterion.

    The determinant ratio of swapping incumbent ``xᵢ`` for candidate
    ``x`` is ``(1 − ℓᵢ)(1 + ℓ(x)) + cross_i²`` where ``cross_i =
    xᵀ A⁻¹ xᵢ``.  Admit iff max-over-i of that ratio exceeds 1.
    """

    def test_admit_into_orthogonal_space_with_redundant_incumbents(self):
        """Buffer full of redundant points along [1,0]; candidate along
        [0,1] should be admitted, and the cheapest evictee is one of the
        redundants (cross_i = 0 means no cross-term salvage).
        """
        policy = DOptimalPolicy()
        # Buffer state: A⁻¹ is the inverse of XᵀX + λI.  Three [1,0]
        # incumbents and one [0,1] gives XᵀX = diag(3, 1).  With small
        # regularization (λ ≈ 0), A⁻¹ ≈ diag(1/3, 1).
        info_inv = [[1.0 / 3.0, 0.0], [0.0, 1.0]]
        feature_vectors = [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        candidate = [0.0, 1.0]  # along [0,1] direction (already covered)
        decision = policy.attempt_exchange(
            candidate, observations=[],
            feature_vectors=feature_vectors,
            info_inv=info_inv, xtx=None,
        )
        assert decision is not None
        # Candidate's leverage = [0,1] · diag(1/3, 1) · [0,1] = 1.
        # Best swap is one of the redundant [1,0] incumbents (their
        # removal preserves the [0,1] direction; cross is 0 so no
        # salvage).  Accept the swap with positive ratio gain.
        assert decision.evictee_index in (0, 1, 2)
        assert decision.candidate_score == pytest.approx(1.0, rel=1e-9)

    def test_cross_term_changes_evictee_vs_pure_leverage(self):
        """Demonstrates Fedorov diverges from LeveragePolicy.

        Construct three incumbents along [1,0]: leverage 1/3 each.
        One incumbent along [0,1]: leverage 1.0 (highest).
        Pure LeveragePolicy would call the [1,0] incumbents the lowest-
        leverage and prefer them for eviction.

        With a candidate along [1, 0], leverage(x) = 1/3, cross_i:
        - vs each [1,0] incumbent: cross_i = 1/3 (parallel)
        - vs [0,1] incumbent:      cross_i = 0 (orthogonal)

        Ratio formulas:
        - swap with [1,0]:  (1 − 1/3)(1 + 1/3) + (1/3)² = (2/3)(4/3) + 1/9 = 8/9 + 1/9 = 1.0
        - swap with [0,1]:  (1 − 1)(1 + 1/3) + 0² = 0 — singular!

        So Fedorov picks one of the [1,0] incumbents (ratio = 1.0,
        boundary).  Pure leverage would also pick a [1,0] (ratio
        coincidence here).  The cross term ensures we never pick the
        [0,1] incumbent — its removal makes the design singular.
        """
        policy = DOptimalPolicy()
        info_inv = [[1.0 / 3.0, 0.0], [0.0, 1.0]]
        feature_vectors = [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],  # singular-removal incumbent
        ]
        candidate = [1.0, 0.0]
        decision = policy.attempt_exchange(
            candidate, observations=[],
            feature_vectors=feature_vectors,
            info_inv=info_inv, xtx=None,
        )
        assert decision is not None
        # Must NOT pick index 3 — cross term protects against singular
        # design under Fedorov.  Other indices are tied at ratio = 1.0.
        assert decision.evictee_index != 3

    def test_reject_when_no_swap_improves_determinant(self):
        """A candidate that's a duplicate of a unique-direction incumbent
        in a well-conditioned buffer can't improve the determinant.

        Buffer: 4 obs along orthogonal directions of an n=2 feature
        space — XᵀX = diag(2, 2), well-conditioned.  Candidate equals
        one of the obs exactly.  No swap improves det.
        """
        policy = DOptimalPolicy()
        # XᵀX = sum of x_i x_i^T = [[1,0],[0,0]] + [[1,0],[0,0]]
        #                          + [[0,0],[0,1]] + [[0,0],[0,1]]
        #                        = diag(2, 2).  A⁻¹ ≈ diag(0.5, 0.5).
        info_inv = [[0.5, 0.0], [0.0, 0.5]]
        feature_vectors = [
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
        ]
        candidate = [1.0, 0.0]  # duplicate
        decision = policy.attempt_exchange(
            candidate, observations=[],
            feature_vectors=feature_vectors,
            info_inv=info_inv, xtx=None,
        )
        assert decision is not None
        # leverage_x = 0.5; for each [1,0] incumbent ℓᵢ = 0.5,
        # cross_i = 0.5; ratio = (0.5)(1.5) + 0.25 = 0.75 + 0.25 = 1.0
        # For each [0,1] incumbent ℓᵢ = 0.5, cross_i = 0;
        # ratio = (0.5)(1.5) + 0 = 0.75 < 1
        # max_ratio = 1.0 — boundary, admit-iff-strict-positive returns False.
        assert decision.admit is False
        assert decision.candidate_score == pytest.approx(0.5, abs=1e-9)


class TestExchangeChoiceShape:
    def test_is_frozen(self):
        e = ExchangeChoice(
            admit=True, evictee_index=2,
            candidate_score=0.5, evictee_score=0.3,
            rejection_reason=None,
        )
        with pytest.raises(AttributeError):
            e.admit = False  # type: ignore[misc]


# ── DOptimalPolicy: integration through DiversityAwareBuffer ─────────


class TestDOptimalBufferIntegration:
    def _buf(self, max_size: int) -> DiversityAwareBuffer:
        return DiversityAwareBuffer(
            n_features=3, max_size=max_size,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=DOptimalPolicy(),
        )

    def test_buffer_dispatches_to_attempt_exchange_when_full(self):
        """Full buffer + DOptimalPolicy uses the joint exchange path,
        carrying through ``policy_name`` to ``BufferAddResult``."""
        buf = self._buf(max_size=3)
        buf.add(_make_obs(t=0.0, outdoor=5.0, solar=1.0))
        buf.add(_make_obs(t=1.0, outdoor=-5.0, solar=2.0))
        buf.add(_make_obs(t=2.0, outdoor=0.0, solar=10.0))
        r = buf.add(_make_obs(t=3.0, outdoor=3.0, solar=4.0))
        assert r.policy_name == "d_optimal"
        # Whichever way the swap goes, the score fields are populated.
        assert r.candidate_score is not None
        assert r.min_incumbent_score is not None
        if not r.admitted:
            assert r.rejection_reason == "d_optimal_rejected"

    def test_buffer_not_full_falls_through_simple_path(self):
        """``attempt_exchange`` is only consulted when the buffer is
        full.  For not-full adds, the inherited simple-path methods run
        (LeveragePolicy ``score_candidate``); admitted=True regardless
        of policy."""
        buf = self._buf(max_size=10)
        r = buf.add(_make_obs(t=0.0, outdoor=5.0, solar=0.0))
        assert r.admitted is True
        assert r.policy_name == "d_optimal"
        assert r.candidate_score is not None
        assert r.evicted_timestamp is None
        assert r.min_incumbent_score is None  # buffer wasn't full


# ── AOptimalPolicy: trace-reduction contract ─────────────────────────


class TestAOptimalPolicyScoring:
    """Score formula: ``‖A⁻¹ x‖² / (1 + ℓ(x))`` — trace reduction from
    admitting the candidate (Sherman-Morrison applied to ``trace((A +
    xx^T)⁻¹)``).
    """

    def test_score_is_positive_for_nonzero_candidate(self):
        """Any non-zero candidate reduces ``trace(A⁻¹)`` strictly."""
        policy = AOptimalPolicy()
        info_inv = [[2.0, 0.0], [0.0, 3.0]]  # diag → ‖A⁻¹ x‖² for [1, 1] = 4 + 9
        x = [1.0, 1.0]
        # Expected: ‖[2, 3]‖² / (1 + (1·2·1 + 1·3·1)) = 13 / 6 ≈ 2.1667
        score = policy.score_candidate(x, info_inv, xtx=None, n_buffered=0)
        assert score == pytest.approx(13.0 / 6.0, rel=1e-9)

    def test_score_zero_for_zero_candidate(self):
        policy = AOptimalPolicy()
        info_inv = [[1.0, 0.0], [0.0, 1.0]]
        score = policy.score_candidate([0.0, 0.0], info_inv, xtx=None, n_buffered=0)
        assert score == 0.0

    def test_score_higher_for_high_inverse_norm_direction(self):
        """A candidate aligned with a large-variance (small-eigenvalue)
        direction yields a larger trace reduction than one aligned with
        an already-tight direction.
        """
        policy = AOptimalPolicy()
        # info_inv = diag(0.1, 10) — so direction 0 has variance 0.1,
        # direction 1 has variance 10 (poorly estimated).
        info_inv = [[0.1, 0.0], [0.0, 10.0]]
        score_tight = policy.score_candidate(
            [1.0, 0.0], info_inv, xtx=None, n_buffered=0,
        )
        score_loose = policy.score_candidate(
            [0.0, 1.0], info_inv, xtx=None, n_buffered=0,
        )
        # tight: 0.01 / 1.1 ≈ 0.0091
        # loose: 100  / 11  ≈ 9.09 — reducing variance in the loose
        # direction matters far more.
        assert score_loose > 100 * score_tight

    def test_should_admit_strict_inequality(self):
        policy = AOptimalPolicy()
        assert policy.should_admit(0.5, 0.4) is True
        assert policy.should_admit(0.5, 0.5) is False
        assert policy.should_admit(0.4, 0.5) is False


class TestAOptimalPolicyFindEvictee:
    """``argmin_i ‖A⁻¹ xᵢ‖² / (1 − ℓᵢ)`` — cheapest incumbent to remove."""

    def test_empty_buffer_returns_sentinel(self):
        policy = AOptimalPolicy()
        choice = policy.find_evictee(
            observations=[], feature_vectors=[],
            info_inv=[[1.0, 0.0], [0.0, 1.0]], xtx=None,
        )
        assert choice.index == -1
        assert choice.score == float("inf")

    def test_picks_low_inverse_norm_incumbent_pure_python(self):
        """Below the numpy threshold, the pure-Python loop selects the
        incumbent with the smallest ``‖A⁻¹ xᵢ‖² / (1 − ℓᵢ)``.

        Constructed with ``info_inv = diag(0.25, 0.25)`` so the example
        feature vectors have leverage well below 1 (matches the
        production invariant that buffered-point leverages ≤ 1).
        """
        policy = AOptimalPolicy()
        info_inv = [[0.25, 0.0], [0.0, 0.25]]
        feature_vectors = [
            [1.0, 0.0],   # ‖Ainv·x‖² = 0.0625, ℓ = 0.25, cost = 0.0625/0.75 ≈ 0.0833 ← min
            [1.5, 0.0],   # ‖Ainv·x‖² = 0.140625, ℓ = 0.5625, cost ≈ 0.3214
            [0.0, 1.7],   # ‖Ainv·x‖² = 0.180625, ℓ = 0.7225, cost ≈ 0.6510
        ]
        observations = []
        choice = policy.find_evictee(observations, feature_vectors, info_inv, None)
        assert choice.index == 0
        assert choice.score == pytest.approx(0.0625 / 0.75, rel=1e-9)

    def test_singular_removal_yields_infinite_cost(self):
        """An incumbent with ``ℓ ≥ 1`` (singular removal) gets cost =
        inf, so it's never preferred for eviction.  Production buffers
        maintain ``ℓ < 1`` via regularization, but the policy guards
        against degenerate inputs defensively."""
        policy = AOptimalPolicy()
        info_inv = [[1.0, 0.0], [0.0, 1.0]]
        feature_vectors = [
            [1.0, 0.0],   # ℓ = 1, denom = 0, cost = inf
            [0.5, 0.0],   # ℓ = 0.25, cost finite
        ]
        choice = policy.find_evictee([], feature_vectors, info_inv, None)
        assert choice.index == 1
        assert choice.score < float("inf")


class TestAOptimalBufferIntegration:
    def _buf(self, max_size: int) -> DiversityAwareBuffer:
        return DiversityAwareBuffer(
            n_features=3, max_size=max_size,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=AOptimalPolicy(),
        )

    def test_buffer_reports_a_optimal_policy_name(self):
        buf = self._buf(max_size=5)
        r = buf.add(_make_obs(t=0.0, outdoor=5.0))
        assert r.policy_name == "a_optimal"

    def test_full_buffer_decision_carries_score_metadata(self):
        """End-to-end smoke: full buffer + AOptimal supplies scores and,
        on rejection, the literal ``a_optimal_rejected``."""
        buf = self._buf(max_size=3)
        buf.add(_make_obs(t=0.0, outdoor=5.0, solar=1.0))
        buf.add(_make_obs(t=1.0, outdoor=-5.0, solar=2.0))
        buf.add(_make_obs(t=2.0, outdoor=0.0, solar=10.0))
        r = buf.add(_make_obs(t=3.0, outdoor=2.0, solar=5.0))
        assert r.policy_name == "a_optimal"
        assert r.candidate_score is not None
        assert r.min_incumbent_score is not None
        if not r.admitted:
            assert r.rejection_reason == "a_optimal_rejected"


# ── SlidingWindowPolicy: FIFO recency contract ───────────────────────


class TestSlidingWindowPolicy:
    """Plain FIFO admission — newest in, oldest out, always admit."""

    def test_find_evictee_returns_oldest_timestamp(self):
        policy = SlidingWindowPolicy()
        obs1 = _make_obs(t=10.0)
        obs2 = _make_obs(t=5.0)   # oldest
        obs3 = _make_obs(t=20.0)
        feature_vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        choice = policy.find_evictee(
            observations=[obs1, obs2, obs3],
            feature_vectors=feature_vectors,
            info_inv=[[1.0, 0.0], [0.0, 1.0]],
            xtx=None,
        )
        assert choice.index == 1
        assert choice.score == 5.0

    def test_find_evictee_empty_buffer_returns_sentinel(self):
        policy = SlidingWindowPolicy()
        choice = policy.find_evictee(
            observations=[], feature_vectors=[],
            info_inv=[], xtx=None,
        )
        assert choice.index == -1

    def test_should_admit_always_true(self):
        """FIFO never rejects when full."""
        policy = SlidingWindowPolicy()
        # Comparator is unconditional; cand/evict scores irrelevant.
        assert policy.should_admit(0.0, 0.0) is True
        assert policy.should_admit(-100.0, 100.0) is True


class TestSlidingWindowBufferIntegration:
    def _buf(self, max_size: int) -> DiversityAwareBuffer:
        return DiversityAwareBuffer(
            n_features=3, max_size=max_size,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=SlidingWindowPolicy(),
        )

    def test_full_buffer_evicts_oldest_admits_new(self):
        """Newest in, oldest out — regardless of leverage."""
        buf = self._buf(max_size=3)
        buf.add(_make_obs(t=0.0, outdoor=5.0, solar=1.0))
        buf.add(_make_obs(t=1.0, outdoor=-5.0, solar=2.0))
        buf.add(_make_obs(t=2.0, outdoor=0.0, solar=10.0))
        # Add a 4th — t=0.0 (the oldest) must be evicted regardless of
        # whether the new obs is "informative" by leverage standards.
        r = buf.add(_make_obs(t=3.0, outdoor=0.0, solar=0.0))
        assert r.admitted is True
        assert r.evicted_timestamp == 0.0
        assert r.policy_name == "sliding_window"
        timestamps = [o.timestamp for o in buf.get_all()]
        assert 0.0 not in timestamps
        assert 3.0 in timestamps

    def test_redundant_candidate_still_admitted(self):
        """A duplicate of an existing obs is admitted (FIFO is content-blind),
        unlike LeveragePolicy/DOptimal which would reject."""
        buf = self._buf(max_size=3)
        buf.add(_make_obs(t=0.0, outdoor=5.0, solar=0.0))
        buf.add(_make_obs(t=1.0, outdoor=5.0, solar=0.0))
        buf.add(_make_obs(t=2.0, outdoor=5.0, solar=0.0))
        # All identical content — newest still wins.
        r = buf.add(_make_obs(t=3.0, outdoor=5.0, solar=0.0))
        assert r.admitted is True
        assert r.evicted_timestamp == 0.0


# ── TimeWindowPolicy: time-defined sliding window ────────────────────


class TestTimeWindowPolicyContract:
    """Unit-level contract for TimeWindowPolicy: FIFO when full, plus
    expired_indices() prunes by time relative to the reference timestamp.
    """

    def test_init_rejects_non_positive_window(self):
        with pytest.raises(ValueError, match="must be > 0"):
            TimeWindowPolicy(window_seconds=0)
        with pytest.raises(ValueError, match="must be > 0"):
            TimeWindowPolicy(window_seconds=-1.0)

    def test_find_evictee_returns_oldest(self):
        """When buffer is full, behaves like FIFO (memory safety net)."""
        policy = TimeWindowPolicy(window_seconds=60.0)
        obs1 = _make_obs(t=10.0)
        obs2 = _make_obs(t=5.0)
        obs3 = _make_obs(t=20.0)
        choice = policy.find_evictee(
            observations=[obs1, obs2, obs3],
            feature_vectors=[[1.0, 0.0]] * 3,
            info_inv=[[1.0, 0.0], [0.0, 1.0]],
            xtx=None,
        )
        assert choice.index == 1
        assert choice.score == 5.0

    def test_should_admit_always_true(self):
        policy = TimeWindowPolicy(window_seconds=60.0)
        assert policy.should_admit(0.0, 0.0) is True

    def test_expired_indices_empty_buffer(self):
        policy = TimeWindowPolicy(window_seconds=60.0)
        assert policy.expired_indices([], reference_timestamp=100.0) == []

    def test_expired_indices_all_within_window(self):
        """Reference = 100, window = 60, all obs within [40, 100] kept."""
        policy = TimeWindowPolicy(window_seconds=60.0)
        observations = [_make_obs(t=ts) for ts in (40.0, 50.0, 100.0)]
        assert policy.expired_indices(observations, reference_timestamp=100.0) == []

    def test_expired_indices_some_older_than_cutoff(self):
        """Reference = 100, window = 60 → cutoff = 40 → obs older than 40 expire."""
        policy = TimeWindowPolicy(window_seconds=60.0)
        observations = [
            _make_obs(t=10.0),   # expired (10 < 40)
            _make_obs(t=30.0),   # expired (30 < 40)
            _make_obs(t=40.0),   # boundary: 40 < 40 is False → kept
            _make_obs(t=50.0),   # kept
            _make_obs(t=100.0),  # kept
        ]
        # Sorted descending so caller can `del observations[idx]` safely.
        assert policy.expired_indices(observations, reference_timestamp=100.0) == [1, 0]

    def test_expired_indices_uses_reference_not_wall_clock(self):
        """Pruning is relative to passed reference, not system time —
        critical for persistence-restore semantics."""
        policy = TimeWindowPolicy(window_seconds=60.0)
        # Observations with very old timestamps (e.g. restored from disk).
        old_observations = [_make_obs(t=ts) for ts in (1000.0, 1010.0, 1050.0)]
        # Reference = 1060 → cutoff = 1000 → first one (t=1000, equal) kept
        # because the predicate is strict (`<`); 1010 and 1050 kept.
        assert policy.expired_indices(old_observations, 1060.0) == []
        # Reference = 1070 → cutoff = 1010 → t=1000 expires (idx 0).
        assert policy.expired_indices(old_observations, 1070.0) == [0]


class TestTimeWindowBufferIntegration:
    """Buffer-level: time-window pruning on every add() via
    `_apply_time_window_expiry`."""

    def _buf(self, *, window_seconds: float, max_size: int = 100):
        return DiversityAwareBuffer(
            n_features=3, max_size=max_size,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=TimeWindowPolicy(window_seconds=window_seconds),
        )

    def test_observations_within_window_retained(self):
        buf = self._buf(window_seconds=100.0)
        for ts in (0.0, 30.0, 60.0, 90.0):
            buf.add(_make_obs(t=ts, outdoor=ts * 0.1))
        # Latest = 90, window = 100, cutoff = -10 → all kept.
        assert sorted(o.timestamp for o in buf.get_all()) == [0.0, 30.0, 60.0, 90.0]

    def test_older_than_window_pruned_on_admission(self):
        """Each new admission shifts the cutoff; obs older than window
        from the new latest are pruned automatically."""
        buf = self._buf(window_seconds=50.0)
        buf.add(_make_obs(t=0.0))
        buf.add(_make_obs(t=10.0))
        buf.add(_make_obs(t=20.0))
        # Latest=20, cutoff = -30 → all 3 kept.
        assert len(buf.get_all()) == 3
        # Now jump forward — latest=100, cutoff=50 → 0, 10, 20 all expire.
        buf.add(_make_obs(t=100.0))
        timestamps = sorted(o.timestamp for o in buf.get_all())
        assert timestamps == [100.0]

    def test_persistence_restore_semantics(self):
        """Reference timestamp is the *latest admitted* obs, not wall clock.
        Restored buffer with old timestamps should not auto-expire on the
        next add unless that add itself shifts the window past them.
        """
        buf = self._buf(window_seconds=100.0)
        # Pretend these came from persistence — already-old timestamps.
        for ts in (1000.0, 1020.0, 1050.0, 1080.0):
            buf.add(_make_obs(t=ts))
        # Latest = 1080, cutoff = 980 → all 4 retained.
        assert len(buf.get_all()) == 4

    def test_max_size_safety_net(self):
        """If sensor cadence is so dense that the time window holds more
        than max_size obs, FIFO eviction kicks in to cap memory."""
        # window=1000s, but only 5 slots. Densely-spaced observations
        # would fit in the window count-wise too, except we exceed max_size.
        buf = self._buf(window_seconds=1000.0, max_size=5)
        for ts in range(8):
            buf.add(_make_obs(t=float(ts)))
        # All 8 are within the 1000s window. Buffer capped at 5 → FIFO kept
        # the 5 newest (3..7).
        timestamps = sorted(o.timestamp for o in buf.get_all())
        assert len(timestamps) == 5
        assert timestamps == [3.0, 4.0, 5.0, 6.0, 7.0]

    def test_non_time_window_policy_unaffected(self):
        """The `_apply_time_window_expiry` no-op for policies without
        `expired_indices` — sliding-window/leverage/etc. behave unchanged.
        """
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=SlidingWindowPolicy(),
        )
        for ts in (0.0, 10.0, 100.0, 10000.0):
            buf.add(_make_obs(t=ts))
        assert len(buf.get_all()) == 4


# ── SlevPolicy: probabilistic admission contract ─────────────────────


class TestSlevPolicyContract:
    """Unit-level contract: SLEV scoring degenerates to LeveragePolicy at
    α=1, and to uniform-noise scoring at α=0.
    """

    def test_alpha_must_be_in_unit_interval(self):
        with pytest.raises(ValueError):
            SlevPolicy(alpha=-0.1)
        with pytest.raises(ValueError):
            SlevPolicy(alpha=1.5)

    def test_alpha_one_score_equals_leverage(self):
        """At α=1, score_candidate matches LeveragePolicy score."""
        slev = SlevPolicy(alpha=1.0, seed=42)
        lev = LeveragePolicy()
        info_inv = [[2.0, 0.5], [0.5, 3.0]]
        x = [1.0, 2.0]
        s_slev = slev.score_candidate(x, info_inv, xtx=None, n_buffered=0)
        s_lev = lev.score_candidate(x, info_inv, xtx=None, n_buffered=0)
        assert s_slev == pytest.approx(s_lev, rel=1e-9)

    def test_alpha_zero_score_uses_only_noise(self):
        """At α=0, the score is purely noise (independent of leverage)."""
        slev = SlevPolicy(alpha=0.0, seed=42)
        info_inv = [[2.0, 0.5], [0.5, 3.0]]
        # Two different x vectors, same call with different RNG draws
        s1 = slev.score_candidate([1.0, 2.0], info_inv, xtx=None, n_buffered=0)
        s2 = slev.score_candidate([5.0, 7.0], info_inv, xtx=None, n_buffered=0)
        # Both depend only on noise; they should differ (different noise draws)
        # but neither has any leverage component.  The exact values depend on
        # RNG; we just verify they're non-negative and not equal.
        assert s1 >= 0
        assert s2 >= 0
        assert s1 != s2  # different RNG draws

    def test_should_admit_strict_inequality(self):
        slev = SlevPolicy(alpha=0.5, seed=42)
        assert slev.should_admit(0.5, 0.4) is True
        assert slev.should_admit(0.5, 0.5) is False
        assert slev.should_admit(0.4, 0.5) is False

    def test_find_evictee_empty_buffer_returns_sentinel(self):
        slev = SlevPolicy(alpha=0.5, seed=42)
        choice = slev.find_evictee(
            observations=[], feature_vectors=[],
            info_inv=[[1.0, 0.0], [0.0, 1.0]], xtx=None,
        )
        assert choice.index == -1


class TestSlevBufferIntegration:
    def _buf(self, max_size: int, alpha: float, seed: int = 42) -> DiversityAwareBuffer:
        return DiversityAwareBuffer(
            n_features=3, max_size=max_size,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=SlevPolicy(alpha=alpha, seed=seed),
        )

    def test_buffer_reports_slev_policy_name(self):
        buf = self._buf(max_size=5, alpha=0.5)
        r = buf.add(_make_obs(t=0.0, outdoor=5.0))
        assert r.policy_name == "slev"

    def test_alpha_zero_eviction_is_random(self):
        """At α=0, eviction is determined by ephemeral noise — admit
        decisions should be approximately uniform over candidates regardless
        of leverage.

        Smoke test: with α=0 and a full buffer, repeated identical-leverage
        admissions should sometimes succeed (random eviction picks SOME
        incumbent each time).
        """
        buf = self._buf(max_size=5, alpha=0.0)
        for i in range(5):
            buf.add(_make_obs(t=float(i), outdoor=5.0, solar=0.0))
        # Now at capacity; try to admit duplicates and count admissions.
        n_admitted = 0
        for i in range(20):
            r = buf.add(_make_obs(t=10.0 + i, outdoor=5.0, solar=0.0))
            if r.admitted:
                n_admitted += 1
        # With α=0 (random), some fraction should be admitted (>0% probability).
        # Far weaker than LeveragePolicy which would reject all duplicates
        # since their leverage equals incumbents'.
        assert n_admitted > 0, (
            f"Expected some random-admit events at α=0; got {n_admitted}/20"
        )

    def test_alpha_one_behaves_like_leverage_policy(self):
        """At α=1, SlevPolicy admission decisions match LeveragePolicy
        on the same input sequence (for the score_candidate path).
        """
        # Note: find_evictee at α=1 still uses noise=0 contribution, so should
        # match LeveragePolicy exactly.
        buf_slev = DiversityAwareBuffer(
            n_features=3, max_size=4,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=SlevPolicy(alpha=1.0, seed=42),
        )
        buf_lev = DiversityAwareBuffer(
            n_features=3, max_size=4,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
            policy=LeveragePolicy(),
        )
        # Fill both buffers identically.
        for i in range(4):
            buf_slev.add(_make_obs(t=float(i), outdoor=5.0 + i, solar=0.0))
            buf_lev.add(_make_obs(t=float(i), outdoor=5.0 + i, solar=0.0))
        # Try a high-leverage candidate; both should accept.
        r_s = buf_slev.add(_make_obs(t=100.0, outdoor=30.0, solar=0.0))
        r_l = buf_lev.add(_make_obs(t=100.0, outdoor=30.0, solar=0.0))
        assert r_s.admitted == r_l.admitted


# ── Coverage backfill ────────────────────────────────────────────────
#
# The remaining gaps in buffer_policies coverage are paths gated on
# either ``not _NUMPY_AVAILABLE`` (pure-Python fallbacks) or
# ``_NUMPY_AVAILABLE and m > 50`` (numpy fast-paths used at production
# buffer scale, but not by the small-buffer fixtures elsewhere in this
# file).  These tests close those gaps without depending on a 50+
# observation buffer for every case.


class TestPolicyPurePythonFallbacks:
    """Cover the ``not _NUMPY_AVAILABLE`` branches in each policy.

    Patches the module-level ``_NUMPY_AVAILABLE`` flag so the same code
    can be exercised on a system where numpy is installed (CI / dev).
    The pure-Python paths are the deployment fallback for HA installs
    that don't ship numpy with the integration's runtime — they need
    to stay correct.
    """

    def test_d_optimal_attempt_exchange_pure_python_path(self, monkeypatch):
        from custom_components.tasmota_irhvac.pi import buffer_policies as bp
        monkeypatch.setattr(bp, "_NUMPY_AVAILABLE", False)

        policy = DOptimalPolicy()
        info_inv = [[1.0 / 3.0, 0.0], [0.0, 1.0]]
        feature_vectors = [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        candidate = [0.0, 1.0]
        decision = policy.attempt_exchange(
            candidate, observations=[],
            feature_vectors=feature_vectors,
            info_inv=info_inv, xtx=None,
        )
        assert decision is not None
        # Pure-Python branch should produce the same ratio shape as numpy:
        # a [1,0] incumbent gets evicted, candidate score == 1.0.
        assert decision.evictee_index in (0, 1, 2)
        assert decision.candidate_score == pytest.approx(1.0, rel=1e-9)

    def test_d_optimal_attempt_exchange_empty_buffer_returns_none(self):
        """Empty-buffer guard in ``attempt_exchange``: returns None and
        defers to the buffer's unconditional admission path."""
        policy = DOptimalPolicy()
        decision = policy.attempt_exchange(
            candidate=[1.0, 0.0],
            observations=[],
            feature_vectors=[],
            info_inv=[[1.0, 0.0], [0.0, 1.0]],
            xtx=None,
        )
        assert decision is None

    def test_a_optimal_find_evictee_numpy_path_m_over_50(self):
        """A-optimal: with m > 50 the numpy einsum path runs.  Build a
        56-row buffer of well-spread feature vectors and verify the
        cheapest-to-remove incumbent is selected with finite cost.
        """
        policy = AOptimalPolicy()
        # Build 56 well-spread 2D vectors (dominantly varied along axis 0).
        feature_vectors = [[float(i), 1.0 + (i % 3) * 0.1] for i in range(56)]
        # Approximate A^-1 = (X^T X + λI)^-1; we don't need exact, just non-degenerate.
        # Build XtX → invert via a small numpy assist; pure regularization.
        import numpy as np
        X = np.asarray(feature_vectors)
        XtX = X.T @ X + 1e-3 * np.eye(2)
        Ainv = np.linalg.inv(XtX).tolist()

        choice = policy.find_evictee(
            observations=[], feature_vectors=feature_vectors,
            info_inv=Ainv, xtx=None,
        )
        assert 0 <= choice.index < 56
        assert choice.score < float("inf")

    def test_min_eig_min_eig_pure_python_fallback(self, monkeypatch):
        from custom_components.tasmota_irhvac.pi import buffer_policies as bp
        monkeypatch.setattr(bp, "_NUMPY_AVAILABLE", False)

        policy = MinEigPolicy()
        # Symmetric PSD matrix; pure-Python fallback delegates to
        # DiversityAwareBuffer._eigenvalues_symmetric, which we let run.
        M = [[2.0, 0.5], [0.5, 1.5]]
        eig = policy._min_eig(M)
        assert eig > 0  # smallest eigenvalue of a PD matrix is positive

    def test_min_eig_min_eig_returns_nan_when_eigensolver_fails(
        self, monkeypatch,
    ):
        """When ``_eigenvalues_symmetric`` returns None, ``_min_eig``
        propagates NaN — defensive path covering the n×n eigendecomp
        failure mode in pure-Python deployments."""
        import math
        from custom_components.tasmota_irhvac.pi import buffer_policies as bp
        from custom_components.tasmota_irhvac.pi import batch_learning as bl
        monkeypatch.setattr(bp, "_NUMPY_AVAILABLE", False)
        monkeypatch.setattr(
            bl.DiversityAwareBuffer, "_eigenvalues_symmetric",
            staticmethod(lambda M, n: None),
        )
        result = MinEigPolicy()._min_eig([[1.0, 0.0], [0.0, 1.0]])
        assert math.isnan(result)

    def test_min_eig_find_evictee_pure_python_path(self, monkeypatch):
        from custom_components.tasmota_irhvac.pi import buffer_policies as bp
        monkeypatch.setattr(bp, "_NUMPY_AVAILABLE", False)

        policy = MinEigPolicy()
        feature_vectors = [
            [1.0, 0.0],
            [1.0, 0.0],  # redundant — cheapest to remove
            [0.0, 1.0],
        ]
        # Build XtX = sum x_i x_iT.
        xtx = [[2.0, 0.0], [0.0, 1.0]]
        choice = policy.find_evictee(
            observations=[], feature_vectors=feature_vectors,
            info_inv=[], xtx=xtx,
        )
        # Removing one of the redundant [1,0]s costs the least.
        assert choice.index in (0, 1)
        assert choice.score < float("inf")

    def test_slev_find_evictee_numpy_path_m_over_50(self):
        """SLEV find_evictee: m > 50 triggers the numpy fast-path."""
        policy = SlevPolicy(alpha=0.5, seed=123)
        # 60-element buffer; non-trivial leverage spread.
        feature_vectors = [[1.0, float(i % 7)] for i in range(60)]
        import numpy as np
        X = np.asarray(feature_vectors)
        XtX = X.T @ X + 1e-3 * np.eye(2)
        Ainv = np.linalg.inv(XtX).tolist()
        choice = policy.find_evictee(
            observations=[], feature_vectors=feature_vectors,
            info_inv=Ainv, xtx=None,
        )
        assert 0 <= choice.index < 60
        assert choice.score >= 0.0
