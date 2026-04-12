"""Tests for batch WLS offline learning module."""

import math

import pytest

from custom_components.tasmota_irhvac.batch_learning import (
    DEFAULT_PRIOR_STD,
    MIN_FEATURE_VARIANCE,
    MAX_STEP_ABS,
    Observation,
    ObservationBuffer,
    BatchResult,
    _diagonal_of_inverse,
    _weighted_variance,
    compare_and_report,
    compute_blended_update,
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

    def test_held_features_excluded_from_max_change(self):
        """Held features should not contribute to max_coeff_change_pct."""
        result = BatchResult(
            n_total=50, n_eligible=40,
            beta_batch=[1.0, 0.5, -8.0],  # feature 2 held at -8.0
            beta_current=[], residual_rms=0.1,
            max_coeff_change_pct=0.0, recommend_update=False,
            held_features={2},
        )
        # current[2] = 0.0, batch[2] = -8.0 → would be 100% if not held
        result = compare_and_report(result, [1.05, 0.48, 0.0], ["int", "slope", "pellet"])
        assert result.max_coeff_change_pct < 20.0
        assert not result.recommend_update

    def test_held_features_logged(self, caplog):
        """Held features should log 'held at' instead of a change percentage."""
        import logging
        result = BatchResult(
            n_total=50, n_eligible=40,
            beta_batch=[1.0, 0.5, -8.0],
            beta_current=[], residual_rms=0.1,
            max_coeff_change_pct=0.0, recommend_update=False,
            held_features={2},
        )
        with caplog.at_level(logging.INFO):
            compare_and_report(result, [1.0, 0.5, -8.0], ["int", "slope", "pellet"])
        assert any("pellet: held at" in r.message for r in caplog.records)
        assert any("insufficient variance" in r.message for r in caplog.records)


# ── Persistent Excitation Filtering ──────────────────────────────────


class TestPersistentExcitation:
    def _make_obs(self, features, sp, cur, des=20.0, rate=0.005, clamped=False):
        return Observation(
            timestamp=0.0, features=features, hp_setpoint=sp,
            current_c=cur, desired_c=des, room_rate=rate, clamped=clamped,
        )

    def test_weighted_variance_constant(self):
        """Constant column has zero variance."""
        assert _weighted_variance([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0

    def test_weighted_variance_varying(self):
        """Varying column has positive variance."""
        var = _weighted_variance([0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        assert var > 0.2

    def test_invariant_feature_held_at_current(self):
        """A feature that never varies should be held at current_beta."""
        # 3 features: intercept, outdoor_delta (varies), pellet (constant 0)
        obs = []
        for i in range(30):
            outdoor = float(i % 10)
            true_offset = 1.0 + 0.5 * outdoor  # + 0*pellet
            obs.append(self._make_obs(
                features=[1.0, outdoor, 0.0],  # pellet always off
                sp=20.0 + true_offset, cur=20.0,
            ))
        current = [0.5, 0.3, -8.0]  # current pellet seed
        result = weighted_least_squares(obs, n_features=3, current_beta=current)
        assert result is not None
        assert 2 in result.held_features
        # Pellet coefficient should be held at -8.0
        assert result.beta_batch[2] == -8.0
        # Intercept and outdoor_delta should still be estimated accurately
        assert abs(result.beta_batch[0] - 1.0) < 0.1
        assert abs(result.beta_batch[1] - 0.5) < 0.1

    def test_varying_feature_not_held(self):
        """A feature that varies should be estimated, not held."""
        obs = []
        for i in range(30):
            outdoor = float(i % 10)
            boiler = 1.0 if i % 5 == 0 else 0.0
            true_offset = 1.0 + 0.5 * outdoor + 2.0 * boiler
            obs.append(self._make_obs(
                features=[1.0, outdoor, boiler],
                sp=20.0 + true_offset, cur=20.0,
            ))
        current = [0.0, 0.0, 0.0]
        result = weighted_least_squares(obs, n_features=3, current_beta=current)
        assert result is not None
        assert 2 not in result.held_features
        assert abs(result.beta_batch[2] - 2.0) < 0.3

    def test_multiple_held_features(self):
        """Multiple invariant features should all be held."""
        obs = []
        for i in range(30):
            outdoor = float(i % 10)
            obs.append(self._make_obs(
                features=[1.0, outdoor, 0.0, 0.0, 0.0],
                sp=20.0 + 1.0 + 0.5 * outdoor, cur=20.0,
            ))
        current = [0.0, 0.0, -8.0, -1.0, 3.5]
        result = weighted_least_squares(obs, n_features=5, current_beta=current)
        assert result is not None
        assert result.held_features == {2, 3, 4}
        assert result.beta_batch[2] == -8.0
        assert result.beta_batch[3] == -1.0
        assert result.beta_batch[4] == 3.5

    def test_held_subtraction_preserves_active_estimates(self):
        """Held feature contributions are subtracted from y, so active
        estimates remain accurate even when held features have large seeds."""
        obs = []
        for i in range(40):
            outdoor = float(i % 10)
            # Pellet is always on (constant 1.0) with true coeff 3.0
            # But since it's constant, WLS can't identify it — it must
            # be held. The intercept absorbs pellet's constant contribution
            # unless we subtract it.
            true_offset = 1.0 + 0.5 * outdoor + 3.0 * 1.0
            obs.append(self._make_obs(
                features=[1.0, outdoor, 1.0],
                sp=20.0 + true_offset, cur=20.0,
            ))
        # current_beta has pellet at 3.0 (correct)
        result = weighted_least_squares(
            obs, n_features=3, current_beta=[0.0, 0.0, 3.0],
        )
        assert result is not None
        assert 2 in result.held_features
        # With correct subtraction, intercept should recover ~1.0
        assert abs(result.beta_batch[0] - 1.0) < 0.1
        assert abs(result.beta_batch[1] - 0.5) < 0.1

    def test_no_current_beta_defaults_to_zero(self):
        """Without current_beta, held features default to 0.0."""
        obs = []
        for i in range(30):
            outdoor = float(i % 10)
            obs.append(self._make_obs(
                features=[1.0, outdoor, 0.0],
                sp=20.0 + 1.0 + 0.5 * outdoor, cur=20.0,
            ))
        result = weighted_least_squares(obs, n_features=3)
        assert result is not None
        assert 2 in result.held_features
        assert result.beta_batch[2] == 0.0

    def test_existing_tests_unaffected(self):
        """Original 2-feature case still works without current_beta."""
        obs = []
        for i in range(40):
            outdoor = float(i % 10)
            true_offset = 1.0 + 0.5 * outdoor
            obs.append(self._make_obs(
                features=[1.0, outdoor],
                sp=20.0 + true_offset, cur=20.0,
            ))
        result = weighted_least_squares(obs, n_features=2, min_observations=20)
        assert result is not None
        assert result.held_features == set()  # outdoor varies → not held
        assert abs(result.beta_batch[0] - 1.0) < 0.05
        assert abs(result.beta_batch[1] - 0.5) < 0.05


# ── Blended Update ───────────────────────────────────────────────────


class TestComputeBlendedUpdate:
    def _make_result(self, beta_batch, beta_current, std_err, held=None):
        return BatchResult(
            n_total=100,
            n_eligible=60,
            beta_batch=beta_batch,
            beta_current=beta_current,
            residual_rms=0.5,
            max_coeff_change_pct=0.0,
            recommend_update=True,
            held_features=held or set(),
            beta_std_err=std_err,
        )

    def test_tight_batch_pulls_strongly(self):
        """Low batch std_err → high gain → pulls toward batch."""
        # σ_batch = 0.1 → σ²_batch = 0.01.  K = 1.0/(1.0+0.01) ≈ 0.99
        r = self._make_result(
            beta_batch=[1.5, 0.6],
            beta_current=[1.0, 0.5],
            std_err=[0.1, 0.1],
        )
        r = compute_blended_update(r)
        assert r.blend_gains[0] > 0.95
        assert abs(r.beta_blended[0] - 1.5) < 0.05
        assert abs(r.beta_blended[1] - 0.6) < 0.05

    def test_wide_batch_barely_moves(self):
        """High batch std_err → low gain → stays near current."""
        # σ_batch = 5.0 → σ²_batch = 25.  K = 1.0/(1.0+25) ≈ 0.038
        r = self._make_result(
            beta_batch=[5.0, 0.5],
            beta_current=[1.0, 0.5],
            std_err=[5.0, 0.1],
        )
        r = compute_blended_update(r)
        assert r.blend_gains[0] < 0.05
        assert abs(r.beta_blended[0] - 1.0) < 0.2  # barely moved

    def test_per_feature_gains_differ(self):
        """Each coefficient gets its own gain based on its std_err."""
        r = self._make_result(
            beta_batch=[3.0, 3.0],
            beta_current=[1.0, 1.0],
            std_err=[0.1, 5.0],  # first tight, second wide
        )
        r = compute_blended_update(r)
        # First feature has high gain, second has low gain
        assert r.blend_gains[0] > 0.9
        assert r.blend_gains[1] < 0.1
        # First hits step cap (K*2.0 > max_step), second barely moves
        assert abs(r.beta_blended[0] - 2.0) < 1e-9  # 1.0 + max_step
        assert r.beta_blended[1] < 1.2  # barely moved from 1.0

    def test_step_cap_limits_large_change(self):
        """Even with high gain, step cap limits per-cycle movement."""
        # σ_batch very small → K ≈ 1.0, but delta = 9.0 exceeds max_step
        r = self._make_result(
            beta_batch=[10.0, 0.5],
            beta_current=[1.0, 0.5],
            std_err=[0.01, 0.1],
        )
        r = compute_blended_update(r)
        assert abs(r.beta_blended[0] - (1.0 + MAX_STEP_ABS)) < 1e-9

    def test_step_cap_negative_direction(self):
        """Step cap works in both directions."""
        r = self._make_result(
            beta_batch=[-10.0, 0.5],
            beta_current=[1.0, 0.5],
            std_err=[0.01, 0.1],
        )
        r = compute_blended_update(r)
        assert abs(r.beta_blended[0] - (1.0 - MAX_STEP_ABS)) < 1e-9

    def test_held_features_gain_zero(self):
        """Held features (inf std_err) get K=0 and don't move."""
        r = self._make_result(
            beta_batch=[1.5, 0.6, -8.0],
            beta_current=[1.0, 0.5, -8.0],
            std_err=[0.1, 0.1, float("inf")],
            held={2},
        )
        r = compute_blended_update(r)
        assert r.blend_gains[2] == 0.0
        assert r.beta_blended[2] == -8.0

    def test_convergence_over_multiple_cycles(self):
        """Repeated blending with tight std_err converges to batch."""
        current = [1.0, 0.5]
        target = [3.0, 0.8]
        for _ in range(20):
            r = self._make_result(
                beta_batch=target,
                beta_current=current,
                std_err=[0.1, 0.1],
            )
            r = compute_blended_update(r)
            current = list(r.beta_blended)
        assert abs(current[0] - 3.0) < 0.01
        assert abs(current[1] - 0.8) < 0.01

    def test_custom_prior_std_and_max_step(self):
        """Custom prior_std changes the gain calculation."""
        # prior_std=0.5 → prior_var=0.25.  σ_batch=0.5 → batch_var=0.25
        # K = 0.25/(0.25+0.25) = 0.5
        r = self._make_result(
            beta_batch=[3.0, 0.5],
            beta_current=[1.0, 0.5],
            std_err=[0.5, 0.1],
        )
        r = compute_blended_update(r, prior_std=0.5, max_step=0.5)
        assert abs(r.blend_gains[0] - 0.5) < 0.01
        # delta = 0.5 * (3.0 - 1.0) = 1.0, but step cap = 0.5
        assert abs(r.beta_blended[0] - 1.5) < 0.01

    def test_wls_std_err_populated(self):
        """WLS returns per-coefficient standard errors for active features."""
        obs = []
        for i in range(40):
            outdoor = float(i % 10)
            obs.append(Observation(
                timestamp=0.0, features=[1.0, outdoor, 0.0],
                hp_setpoint=20.0 + 1.0 + 0.5 * outdoor,
                current_c=20.0, desired_c=20.0, room_rate=0.005, clamped=False,
            ))
        result = weighted_least_squares(obs, n_features=3, current_beta=[0, 0, -8])
        assert result is not None
        # Active features (0, 1) should have finite std_err
        assert result.beta_std_err[0] < 10.0
        assert result.beta_std_err[1] < 10.0
        # Held feature should have inf
        assert math.isinf(result.beta_std_err[2])

    def test_diagonal_of_inverse(self):
        """Diagonal of inverse for a known 2×2 matrix."""
        # A = [[4, 2], [2, 3]]  →  A⁻¹ = [[3/8, -2/8], [-2/8, 4/8]]
        # diag = [3/8, 4/8] = [0.375, 0.5]
        A = [[4.0, 2.0], [2.0, 3.0]]
        diag = _diagonal_of_inverse(A, 2)
        assert diag is not None
        assert abs(diag[0] - 0.375) < 1e-9
        assert abs(diag[1] - 0.5) < 1e-9
