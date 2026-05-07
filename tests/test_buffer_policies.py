"""Unit tests for the ``BufferPolicy`` abstraction and built-in policies."""

from __future__ import annotations

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    DiversityAwareBuffer,
    Observation,
)
from custom_components.tasmota_irhvac.pi.buffer_policies import (
    EvicteeChoice,
    LeveragePolicy,
    MinEigPolicy,
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
    def test_default_policy_is_leverage(self):
        """Buffer constructed without a policy uses LeveragePolicy."""
        buf = DiversityAwareBuffer(
            n_features=3, max_size=5,
            feature_order=TEST_FEATURE_ORDER,
            model_inputs=TEST_MODEL_INPUTS,
        )
        r = buf.add(_make_obs(t=0.0))
        assert r.policy_name == "leverage"

    def test_explicit_leverage_policy_matches_default(self):
        """Passing a LeveragePolicy() explicitly is equivalent to default."""
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
        # Same admission decision on identical input.
        r_def = buf_default.add(_make_obs(t=0.0, outdoor=5.0))
        r_exp = buf_explicit.add(_make_obs(t=0.0, outdoor=5.0))
        assert r_def.admitted == r_exp.admitted
        assert r_def.candidate_score == pytest.approx(r_exp.candidate_score)
        assert r_def.policy_name == r_exp.policy_name == "leverage"


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
