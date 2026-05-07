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
