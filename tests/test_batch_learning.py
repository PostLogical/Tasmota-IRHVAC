"""Tests for batch WLS offline learning module."""

import math

import pytest

from custom_components.tasmota_irhvac.batch_learning import (
    Observation,
    ObservationBuffer,
    BatchResult,
    compare_and_report,
    weighted_least_squares,
)


# ── ObservationBuffer ─────────────────────────────────────────────────


class TestObservationBuffer:
    def _make_obs(self, t=0.0, sp=22.0, cur=20.0, des=20.0, rate=0.0, clamped=False):
        return Observation(
            timestamp=t, features=[1.0, 5.0], hp_setpoint=sp,
            current_c=cur, desired_c=des, room_rate=rate, clamped=clamped,
        )

    def test_add_and_get(self):
        buf = ObservationBuffer(max_size=10)
        for i in range(5):
            buf.add(self._make_obs(t=float(i)))
        assert len(buf) == 5
        assert buf.get_all()[0].timestamp == 0.0
        assert buf.get_all()[-1].timestamp == 4.0

    def test_eviction(self):
        buf = ObservationBuffer(max_size=3)
        for i in range(5):
            buf.add(self._make_obs(t=float(i)))
        assert len(buf) == 3
        assert buf.get_all()[0].timestamp == 2.0  # oldest evicted

    def test_serialization_roundtrip(self):
        buf = ObservationBuffer()
        for i in range(3):
            buf.add(self._make_obs(t=float(i), sp=20.0 + i))
        serialized = buf.as_list()
        restored = ObservationBuffer.from_list(serialized)
        assert len(restored) == 3
        assert restored.get_all()[2].hp_setpoint == 22.0

    def test_from_list_handles_corrupt_entries(self):
        data = [{"t": 1.0, "bad": True}, {"t": 2.0, "x": [1.0], "sp": 22, "cur": 20, "des": 20, "rate": 0, "clamp": False}]
        buf = ObservationBuffer.from_list(data)
        assert len(buf) == 1  # only valid entry kept


# ── Weighted Least Squares ────────────────────────────────────────────


class TestWeightedLeastSquares:
    def _make_obs(self, features, sp, cur, des=20.0, rate=0.005, clamped=False):
        return Observation(
            timestamp=0.0, features=features, hp_setpoint=sp,
            current_c=cur, desired_c=des, room_rate=rate, clamped=clamped,
        )

    def test_recovers_known_intercept(self):
        """With constant features, WLS should recover the mean offset."""
        obs = []
        for i in range(30):
            # hp_setpoint=22, current_c=20 → offset = 2.0
            obs.append(self._make_obs([1.0], sp=22.0, cur=20.0))
        result = weighted_least_squares(obs, n_features=1, min_observations=20)
        assert result is not None
        assert abs(result.beta_batch[0] - 2.0) < 0.01

    def test_recovers_linear_relationship(self):
        """WLS should recover β₀ + β₁*x from synthetic data."""
        # True model: offset = 1.0 + 0.5 * outdoor_delta
        obs = []
        for i in range(40):
            outdoor_delta = float(i % 10)
            true_offset = 1.0 + 0.5 * outdoor_delta
            obs.append(self._make_obs(
                features=[1.0, outdoor_delta],
                sp=20.0 + true_offset,  # hp_setpoint
                cur=20.0,  # at target
            ))
        result = weighted_least_squares(obs, n_features=2, min_observations=20)
        assert result is not None
        assert abs(result.beta_batch[0] - 1.0) < 0.05
        assert abs(result.beta_batch[1] - 0.5) < 0.05

    def test_excludes_clamped(self):
        """Clamped observations should be excluded."""
        obs = []
        for i in range(30):
            obs.append(self._make_obs([1.0], sp=22.0, cur=20.0, clamped=True))
        result = weighted_least_squares(obs, n_features=1, min_observations=20)
        assert result is None  # all excluded

    def test_excludes_unstable(self):
        """Observations with high room_rate should be excluded."""
        obs = []
        for i in range(30):
            obs.append(self._make_obs([1.0], sp=22.0, cur=20.0, rate=0.05))
        result = weighted_least_squares(obs, n_features=1, min_observations=20)
        assert result is None  # all excluded

    def test_returns_none_if_insufficient(self):
        obs = [self._make_obs([1.0], sp=22.0, cur=20.0) for _ in range(5)]
        result = weighted_least_squares(obs, n_features=1, min_observations=20)
        assert result is None

    def test_distance_weighting(self):
        """Observations at target should have more influence than distant ones."""
        obs = []
        # 20 at-target observations saying offset=2.0
        for _ in range(20):
            obs.append(self._make_obs([1.0], sp=22.0, cur=20.0, des=20.0))
        # 20 far-from-target observations saying offset=4.0
        for _ in range(20):
            obs.append(self._make_obs([1.0], sp=27.0, cur=23.0, des=20.0))
        result = weighted_least_squares(obs, n_features=1, min_observations=20)
        assert result is not None
        # At-target obs (weight=1.0) should dominate over distant (weight=0.25)
        # Weighted mean: (20*1.0*2.0 + 20*0.25*4.0) / (20*1.0 + 20*0.25) = 60/25 = 2.4
        assert result.beta_batch[0] < 3.0  # closer to 2.0 than 4.0
        assert result.beta_batch[0] > 2.0  # but pulled slightly by distant obs


# ── Compare and Report ────────────────────────────────────────────────


class TestCompareAndReport:
    def test_no_update_when_similar(self):
        result = BatchResult(
            n_total=50, n_eligible=40,
            beta_batch=[1.0, 0.5],
            beta_current=[], residual_rms=0.1,
            max_coeff_change_pct=0.0, recommend_update=False,
        )
        result = compare_and_report(result, [1.05, 0.48], ["intercept", "slope"])
        assert not result.recommend_update
        assert result.max_coeff_change_pct < 20.0

    def test_recommends_update_when_different(self):
        result = BatchResult(
            n_total=50, n_eligible=40,
            beta_batch=[2.0, 0.5],
            beta_current=[], residual_rms=0.1,
            max_coeff_change_pct=0.0, recommend_update=False,
        )
        result = compare_and_report(result, [1.0, 0.5], ["intercept", "slope"])
        assert result.recommend_update
        assert result.max_coeff_change_pct > 50.0

    def test_no_update_with_few_observations(self):
        result = BatchResult(
            n_total=15, n_eligible=10,
            beta_batch=[2.0, 0.5],
            beta_current=[], residual_rms=0.1,
            max_coeff_change_pct=0.0, recommend_update=False,
        )
        result = compare_and_report(result, [1.0, 0.5], min_observations=20)
        assert not result.recommend_update  # too few observations
