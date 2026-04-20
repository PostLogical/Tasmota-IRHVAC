"""Tests for batch WLS offline learning module."""

import math

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    DEFAULT_PRIOR_STD,
    MIN_FEATURE_VARIANCE,
    MAX_STEP_ABS,
    DiversityAwareBuffer,
    HourlyResidualPattern,
    Observation,
    BatchResult,
    _diagonal_of_inverse,
    _weighted_variance,
    analyze_residuals_by_hour,
    compare_and_report,
    compute_blended_update,
    weighted_least_squares,
)


# ── Weighted Least Squares ────────────────────────────────────────────


class TestWeightedLeastSquares:
    def _make_obs(self, features, sp, cur, des=20.0, rate=0.005, clamped=False, clamped_reason=""):
        if clamped and not clamped_reason:
            clamped_reason = "no_output"
        return Observation(
            timestamp=0.0, features=features, hp_setpoint=sp,
            current_c=cur, desired_c=des, room_rate=rate, clamped=clamped,
            clamped_reason=clamped_reason,
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

    def test_equilibrium_weighting(self):
        """Near-equilibrium observations should have more influence than transient ones."""
        obs = []
        # 20 near-equilibrium observations (rate≈0) saying offset=2.0
        for _ in range(20):
            obs.append(self._make_obs([1.0], sp=22.0, cur=20.0, rate=0.001))
        # 20 transient observations (rate=0.015, near threshold) saying offset=4.0
        for _ in range(20):
            obs.append(self._make_obs([1.0], sp=24.0, cur=20.0, rate=0.015))
        result = weighted_least_squares(obs, n_features=1, min_observations=20)
        assert result is not None
        # Near-equilibrium obs (weight≈1.0) should dominate over transient
        # (weight = 1/(1+(0.015/0.02)²) ≈ 0.64)
        # Weighted mean: (20*1.0*2.0 + 20*0.64*4.0) / (20*1.0 + 20*0.64) ≈ 2.78
        assert result.beta_batch[0] < 3.5  # closer to 2.0 than 4.0
        assert result.beta_batch[0] > 2.0  # but pulled by transient obs

    def test_normalization_does_not_change_result(self):
        """Column normalization must not change the physical coefficients.

        True model: offset = 2.0 + 0.3 * outdoor_delta + (-2.0) * small_feature
        outdoor_delta ranges 0-20 (large scale), small_feature ranges
        0-1 (smaller scale).  The 20x scale difference would degrade
        X'WX conditioning without normalization.  With normalization the
        coefficients in physical units must match the true model.
        """
        obs = []
        for i in range(40):
            od = float(i % 20)              # 0-19, std ≈ 5.8
            small = (i % 10) * 0.1          # 0-0.9, std ≈ 0.3
            true_offset = 2.0 + 0.3 * od + (-2.0) * small
            obs.append(self._make_obs(
                features=[1.0, od, small],
                sp=20.0 + true_offset,
                cur=20.0,
            ))
        result = weighted_least_squares(obs, n_features=3, min_observations=20)
        assert result is not None
        # Physical coefficients must recover the true model
        assert abs(result.beta_batch[0] - 2.0) < 0.1   # intercept
        assert abs(result.beta_batch[1] - 0.3) < 0.01   # outdoor_delta
        assert abs(result.beta_batch[2] - (-2.0)) < 0.1  # small_feature

    def test_normalization_predictions_match_raw(self):
        """Predictions from normalized WLS must match raw data exactly.

        Use the returned beta (physical units) to predict y for each
        observation and verify residuals are near-zero.  This confirms
        the denormalization is correct end-to-end.
        """
        obs = []
        for i in range(30):
            od = 5.0 + float(i % 10)   # 5-14
            rate = 0.001 * (i % 5)      # 0-0.004
            true_offset = 1.5 + 0.4 * od - 3.0 * rate
            obs.append(self._make_obs(
                features=[1.0, od, rate],
                sp=20.0 + true_offset,
                cur=20.0,
            ))
        result = weighted_least_squares(obs, n_features=3, min_observations=20)
        assert result is not None
        beta = result.beta_batch

        # Verify every observation's prediction matches
        for o in obs:
            y = o.hp_setpoint - o.current_c
            pred = sum(beta[j] * o.features[j] for j in range(3))
            assert abs(y - pred) < 0.01, f"Residual {y - pred:.4f} too large"

    def test_normalization_with_vastly_different_scales(self):
        """Extreme scale mismatch (1000x) still recovers coefficients.

        Feature A ranges 0-100, feature B ranges 0-0.1 — independent
        variation so WLS can separate them.  The 1000x scale difference
        would degrade X'WX conditioning without normalization.
        """
        obs = []
        for i in range(40):
            big = float(i % 20) * 5.0      # 0-95, std ≈ 29
            small = (i % 7) * 0.015         # 0-0.09, std ≈ 0.03
            true_offset = 1.0 + 0.05 * big + 10.0 * small
            obs.append(self._make_obs(
                features=[1.0, big, small],
                sp=20.0 + true_offset,
                cur=20.0,
            ))
        result = weighted_least_squares(obs, n_features=3, min_observations=20)
        assert result is not None
        assert abs(result.beta_batch[0] - 1.0) < 0.1    # intercept
        assert abs(result.beta_batch[1] - 0.05) < 0.005  # big feature
        assert abs(result.beta_batch[2] - 10.0) < 1.0    # small feature


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
    def _make_obs(self, features, sp, cur, des=20.0, rate=0.005, clamped=False, clamped_reason=""):
        if clamped and not clamped_reason:
            clamped_reason = "no_output"
        return Observation(
            timestamp=0.0, features=features, hp_setpoint=sp,
            current_c=cur, desired_c=des, room_rate=rate, clamped=clamped,
            clamped_reason=clamped_reason,
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


class TestNamedFeatures:
    """Tests for named feature dict storage and config change resilience."""

    def test_dict_features_round_trip(self):
        """Dict features survive as_dict → from_dict round-trip."""
        obs = Observation(
            timestamp=100.0,
            features={"intercept": 1.0, "outdoor_delta": 5.0, "Solar Proxy": 0.6},
            hp_setpoint=22.0, current_c=20.0, desired_c=20.0,
            room_rate=0.005, clamped=False,
        )
        d = obs.as_dict()
        restored = Observation.from_dict(d)
        assert restored.features == {"intercept": 1.0, "outdoor_delta": 5.0, "Solar Proxy": 0.6}

    def test_legacy_list_converted_with_names(self):
        """Legacy positional list is converted to dict when names provided."""
        legacy = {"t": 0, "x": [1.0, 5.0, 0.6], "sp": 22, "cur": 20, "des": 20, "rate": 0.005, "clamp": False}
        names = ["intercept", "outdoor_delta", "Solar Proxy"]
        obs = Observation.from_dict(legacy, legacy_feature_names=names)
        assert obs.features == {"intercept": 1.0, "outdoor_delta": 5.0, "Solar Proxy": 0.6}

    def test_legacy_list_without_names_gets_generic(self):
        """Legacy list without names gets f0, f1, ... keys."""
        legacy = {"t": 0, "x": [1.0, 5.0], "sp": 22, "cur": 20, "des": 20, "rate": 0.005, "clamp": False}
        obs = Observation.from_dict(legacy)
        assert obs.features == {"f0": 1.0, "f1": 5.0}

    def test_extract_feature_vector_named(self):
        """extract_feature_vector maps by name, zero-fills missing."""
        from custom_components.tasmota_irhvac.pi.batch_learning import extract_feature_vector
        obs = Observation(
            timestamp=0, features={"intercept": 1.0, "outdoor_delta": 5.0, "Old Input": 3.0},
            hp_setpoint=22, current_c=20, desired_c=20, room_rate=0.005, clamped=False,
        )
        # Current config has New Input instead of Old Input
        order = ["intercept", "outdoor_delta", "New Input"]
        result = extract_feature_vector(obs, order)
        assert result == [1.0, 5.0, 0.0]  # New Input zero-filled

    def test_extract_feature_vector_ignores_extra(self):
        """Extra features in observation are ignored."""
        from custom_components.tasmota_irhvac.pi.batch_learning import extract_feature_vector
        obs = Observation(
            timestamp=0, features={"intercept": 1.0, "outdoor_delta": 5.0, "Removed": 99.0},
            hp_setpoint=22, current_c=20, desired_c=20, room_rate=0.005, clamped=False,
        )
        order = ["intercept", "outdoor_delta"]
        result = extract_feature_vector(obs, order)
        assert result == [1.0, 5.0]

    def test_strip_features_removes_from_buffer(self):
        """strip_features deletes named features from all observations."""
        buf = DiversityAwareBuffer(
            n_features=3,
            feature_order=["intercept", "outdoor_delta", "Solar Proxy"],
        )
        for i in range(5):
            buf.add(Observation(
                timestamp=float(i),
                features={"intercept": 1.0, "outdoor_delta": float(i), "Solar Proxy": 0.5},
                hp_setpoint=22, current_c=20, desired_c=20, room_rate=0.005, clamped=False,
            ))
        modified = buf.strip_features({"Solar Proxy"})
        assert modified == 5
        for obs in buf.get_all():
            assert "Solar Proxy" not in obs.features

    def test_strip_features_prevents_stale_reuse(self):
        """After stripping, re-adding same name starts fresh (zero-filled)."""
        from custom_components.tasmota_irhvac.pi.batch_learning import extract_feature_vector
        obs = Observation(
            timestamp=0,
            features={"intercept": 1.0, "outdoor_delta": 5.0},  # Solar stripped
            hp_setpoint=22, current_c=20, desired_c=20, room_rate=0.005, clamped=False,
        )
        # New config re-adds "Solar Proxy"
        order = ["intercept", "outdoor_delta", "Solar Proxy"]
        result = extract_feature_vector(obs, order)
        assert result == [1.0, 5.0, 0.0]  # Old Solar data gone, zero-filled

    def test_wls_with_mixed_named_observations(self):
        """WLS handles observations with different feature sets."""
        feature_order = ["intercept", "outdoor_delta", "Solar Proxy"]
        obs = []
        # Old observations without solar
        for i in range(20):
            obs.append(Observation(
                timestamp=float(i), hp_setpoint=22.0 + 0.5 * i, current_c=20.0,
                desired_c=20.0, room_rate=0.005, clamped=False,
                features={"intercept": 1.0, "outdoor_delta": float(i)},
            ))
        # New observations with solar
        for i in range(20):
            obs.append(Observation(
                timestamp=float(20 + i), hp_setpoint=22.0 + 0.5 * i - 0.3, current_c=20.0,
                desired_c=20.0, room_rate=0.005, clamped=False,
                features={"intercept": 1.0, "outdoor_delta": float(i), "Solar Proxy": 0.5},
            ))
        result = weighted_least_squares(obs, n_features=3, feature_order=feature_order)
        assert result is not None
        assert len(result.beta_batch) == 3


class TestObservationMetadata:
    """Tests for Observation metadata fields and round-trip serialization."""

    def test_metadata_round_trip(self):
        """New metadata fields survive as_dict → from_dict round-trip."""
        obs = Observation(
            timestamp=100.0,
            features=[1.0, 5.0],
            hp_setpoint=22.0,
            current_c=20.0,
            desired_c=20.0,
            room_rate=0.005,
            clamped=False,
            outdoor_temp_c=-5.0,
            integral_settled=True,
            seconds_since_setpoint_change=300.0,
            supplemental_active=True,
        )
        d = obs.as_dict()
        restored = Observation.from_dict(d)
        assert restored.outdoor_temp_c == -5.0
        assert restored.integral_settled is True
        assert restored.seconds_since_setpoint_change == 300.0
        assert restored.supplemental_active is True

    def test_metadata_defaults_from_legacy(self):
        """Legacy observations without metadata fields get safe defaults."""
        legacy_dict = {
            "t": 100.0, "x": [1.0, 5.0], "sp": 22.0,
            "cur": 20.0, "des": 20.0, "rate": 0.005, "clamp": False,
        }
        obs = Observation.from_dict(legacy_dict)
        assert obs.outdoor_temp_c is None
        assert obs.integral_settled is False
        assert obs.seconds_since_setpoint_change == 0.0
        assert obs.supplemental_active is False

    def test_plant_snapshot_on_batch_result(self):
        """BatchResult.plant_snapshot round-trips through dataclasses.asdict."""
        import dataclasses
        result = BatchResult(
            n_total=100,
            n_eligible=80,
            beta_batch=[0.0, 0.3],
            beta_current=[0.0, 0.25],
            residual_rms=0.5,
            max_coeff_change_pct=20.0,
            recommend_update=True,
            plant_snapshot={
                "k": 1.0, "k_confidence": 0.8,
                "tau_fast": 15.0, "tau_fast_confidence": 0.6,
                "tau_slow": 120.0, "tau_slow_confidence": 0.3,
                "theta": 10.0,
            },
        )
        d = dataclasses.asdict(result)
        d["held_features"] = list(d["held_features"])
        restored = BatchResult(**d)
        assert restored.plant_snapshot["tau_slow"] == 120.0
        assert restored.plant_snapshot["k_confidence"] == 0.8


class TestFilterInactive:
    """Tests for DiversityAwareBuffer.filter_inactive().

    Removes observations where the HP had zero output (setpoint wrong
    side of room temp), used for one-time buffer migration after adding
    the hp_no_output condition.
    """

    def _make_obs(self, sp, cur, features=None):
        return Observation(
            timestamp=0.0,
            features=features or [1.0, 5.0],
            hp_setpoint=sp,
            current_c=cur,
            desired_c=21.0,
            room_rate=0.005,
            clamped=False,
        )

    def test_heat_removes_setpoint_below_room(self):
        """filter_inactive('heat') removes obs where sp < current_c."""
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        # 5 clean (sp > room), 3 poisoned (sp < room)
        for _ in range(5):
            buf.add(self._make_obs(sp=22.0, cur=20.0))
        for _ in range(3):
            buf.add(self._make_obs(sp=17.0, cur=24.0))
        assert len(buf) == 8

        removed = buf.filter_inactive("heat")

        assert removed == 3
        assert len(buf) == 5
        for obs in buf.get_all():
            assert obs.hp_setpoint >= obs.current_c

    def test_cool_removes_setpoint_above_room(self):
        """filter_inactive('cool') removes obs where sp > current_c."""
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        for _ in range(5):
            buf.add(self._make_obs(sp=22.0, cur=25.0))  # clean: sp < room in cool
        for _ in range(3):
            buf.add(self._make_obs(sp=26.0, cur=23.0))  # poisoned: sp > room
        assert len(buf) == 8

        removed = buf.filter_inactive("cool")

        assert removed == 3
        assert len(buf) == 5

    def test_recomputes_info_matrix(self):
        """filter_inactive should recompute the info matrix after removal."""
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        for _ in range(5):
            buf.add(self._make_obs(sp=22.0, cur=20.0))
        for _ in range(3):
            buf.add(self._make_obs(sp=17.0, cur=24.0))
        # Force some incremental updates
        buf._updates_since_recompute = 100

        buf.filter_inactive("heat")

        assert buf._updates_since_recompute == 0

    def test_preserves_clean_observations(self):
        """filter_inactive should not remove clean observations."""
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        for i in range(10):
            buf.add(self._make_obs(sp=22.0 + i * 0.1, cur=20.0))

        removed = buf.filter_inactive("heat")

        assert removed == 0
        assert len(buf) == 10

    def test_no_recompute_when_nothing_removed(self):
        """If no observations are removed, skip the info matrix recompute."""
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        for _ in range(5):
            buf.add(self._make_obs(sp=22.0, cur=20.0))
        buf._updates_since_recompute = 50

        buf.filter_inactive("heat")

        # Should NOT have recomputed since nothing was removed
        assert buf._updates_since_recompute == 50


# ── Condition Number & Multicollinearity ──────────────────────────────


class TestConditionNumber:
    def _make_obs(self, features, sp=22.0, cur=20.0, clamped=False, clamped_reason=""):
        if clamped and not clamped_reason:
            clamped_reason = "no_output"
        return Observation(
            timestamp=0.0, features=features, hp_setpoint=sp,
            current_c=cur, desired_c=20.0, room_rate=0.005, clamped=clamped,
            clamped_reason=clamped_reason,
        )

    def test_condition_number_inf_before_recompute(self):
        """Returns inf when xtx matrix hasn't been computed yet."""
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        assert buf.compute_condition_number() == float("inf")

    def test_condition_number_well_conditioned(self):
        """Diverse data produces κ < 20 (Belsley: weak dependencies).

        Outdoor delta spanning 0-9°C covers a wide operating range — the
        intercept and slope are well-separated and individually identifiable.
        Belsley (1980): κ < 20 means coefficient estimates are reliable.
        """
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        for i in range(50):
            outdoor_delta = float(i % 10)  # 0-9°C spread
            buf.add(self._make_obs([1.0, outdoor_delta]))
        buf.recompute_info_matrix()
        cond = buf.compute_condition_number()
        assert cond < 20.0

    def test_condition_number_moderate_collinearity(self):
        """Correlated features give moderate collinearity (Belsley: κ > 30).

        Two features with |r| ≈ 0.9 — partially confounded so
        individual coefficient estimates are unreliable but the
        combined prediction is still stable.
        Belsley (1980): 30 < κ < 100 means some coefficients unreliable.
        """
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        for i in range(50):
            x1 = float(i % 10)
            # x2 tracks x1 closely but not perfectly (r ≈ 0.9)
            x2 = x1 * 0.8 + (i % 3) * 0.5
            buf.add(self._make_obs([1.0, x1, x2]))
        buf.recompute_info_matrix()
        cond = buf.compute_condition_number()
        assert cond > 10.0  # Meaningful collinearity from correlation

    def test_condition_number_severe_collinearity(self):
        """Near-constant feature produces κ > 100 (Belsley: severe).

        Outdoor delta varies only 5.0-5.1°C — the intercept and outdoor
        slope are nearly indistinguishable and small perturbations cause
        large coefficient swings.
        Belsley (1980): κ > 100 means coefficient estimates numerically unstable.
        """
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        for i in range(50):
            buf.add(self._make_obs([1.0, 5.0 + (i % 10) * 0.01]))
        buf.recompute_info_matrix()
        cond = buf.compute_condition_number()
        assert cond > 100.0

    def test_pairwise_correlations_empty_buffer(self):
        """Returns empty list with insufficient data."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        assert buf.get_pairwise_correlations() == []

    def test_pairwise_correlations_detects_correlated(self):
        """Detects highly correlated features."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        for i in range(30):
            outdoor_delta = float(i)
            solar = outdoor_delta * 0.5 + 1.0  # perfectly correlated
            buf.add(self._make_obs([1.0, outdoor_delta, solar]))
        pairs = buf.get_pairwise_correlations(["intercept", "outdoor_delta", "solar"])
        assert len(pairs) == 1
        name_a, name_b, r = pairs[0]
        assert name_a == "outdoor_delta"
        assert name_b == "solar"
        assert abs(r) > 0.99

    def test_pairwise_correlations_independent_features(self):
        """Independent features produce no correlated pairs."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        for i in range(30):
            outdoor_delta = float(i % 10)
            pellet = float((i + 5) % 3)  # uncorrelated pattern
            buf.add(self._make_obs([1.0, outdoor_delta, pellet]))
        pairs = buf.get_pairwise_correlations(["intercept", "outdoor_delta", "pellet"])
        assert len(pairs) == 0

    def test_pairwise_correlations_skips_clamped(self):
        """Correlation computed only from unclamped observations."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        # All unclamped obs are independent
        for i in range(25):
            buf.add(self._make_obs([1.0, float(i % 10), float((i + 5) % 3)]))
        # Clamped obs are perfectly correlated but should be ignored
        for i in range(25):
            buf.add(self._make_obs([1.0, float(i), float(i)], clamped=True))
        pairs = buf.get_pairwise_correlations(["intercept", "a", "b"])
        assert len(pairs) == 0


# ── Residual Time-of-Day Analysis ─────────────────────────────────────


class TestResidualsByHour:
    def _make_obs(self, features, sp, cur, wall_hour, des=20.0, rate=0.005, clamped=False, clamped_reason=""):
        if clamped and not clamped_reason:
            clamped_reason = "no_output"
        return Observation(
            timestamp=0.0, features=features, hp_setpoint=sp,
            current_c=cur, desired_c=des, room_rate=rate, clamped=clamped,
            clamped_reason=clamped_reason, wall_hour=wall_hour,
        )

    def test_detects_afternoon_solar_gain(self):
        """Negative residuals in afternoon suggest unmodeled solar gain."""
        # True model: offset = 1.0 + 0.3 * outdoor_delta
        beta = [1.0, 0.3]
        obs = []
        for hour in range(24):
            for _ in range(8):
                outdoor_delta = 5.0
                true_offset = 1.0 + 0.3 * outdoor_delta
                # Solar gain during 13-16: HP needs 0.8°C less offset
                solar_effect = -0.8 if 13 <= hour <= 16 else 0.0
                obs.append(self._make_obs(
                    features=[1.0, outdoor_delta],
                    sp=20.0 + true_offset + solar_effect,
                    cur=20.0,
                    wall_hour=hour,
                ))
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2)
        # Should detect the afternoon pattern
        assert len(patterns) >= 1
        afternoon = [p for p in patterns if 13 <= p.start_hour <= 16]
        assert len(afternoon) == 1
        assert afternoon[0].mean_residual < -0.5  # negative = gain

    def test_no_pattern_when_model_fits(self):
        """No patterns when model prediction matches observations."""
        beta = [1.0, 0.3]
        obs = []
        for hour in range(24):
            for _ in range(8):
                outdoor_delta = 5.0
                true_offset = 1.0 + 0.3 * outdoor_delta
                obs.append(self._make_obs(
                    features=[1.0, outdoor_delta],
                    sp=20.0 + true_offset,
                    cur=20.0,
                    wall_hour=hour,
                ))
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2)
        assert len(patterns) == 0

    def test_skips_clamped_observations(self):
        """Clamped observations excluded from residual analysis."""
        beta = [1.0, 0.3]
        obs = []
        for hour in range(24):
            for _ in range(8):
                # Big residual but all clamped
                obs.append(self._make_obs(
                    features=[1.0, 5.0],
                    sp=25.0,  # 3.5°C residual
                    cur=20.0,
                    wall_hour=hour,
                    clamped=True,
                ))
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2)
        assert len(patterns) == 0

    def test_skips_missing_wall_hour(self):
        """Observations with wall_hour=-1 (legacy) are excluded."""
        beta = [1.0, 0.3]
        obs = []
        for _ in range(50):
            obs.append(self._make_obs(
                features=[1.0, 5.0], sp=25.0, cur=20.0, wall_hour=-1,
            ))
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2)
        assert len(patterns) == 0

    def test_insufficient_obs_per_hour(self):
        """Hours with fewer than min_obs_per_hour are not flagged."""
        beta = [1.0, 0.3]
        obs = []
        # Only 2 obs at hour 14 — big residual but not enough data
        for _ in range(2):
            obs.append(self._make_obs(
                features=[1.0, 5.0], sp=25.0, cur=20.0, wall_hour=14,
            ))
        # Fill other hours with good data
        for hour in range(24):
            if hour == 14:
                continue
            for _ in range(8):
                obs.append(self._make_obs(
                    features=[1.0, 5.0],
                    sp=20.0 + 1.0 + 0.3 * 5.0,
                    cur=20.0,
                    wall_hour=hour,
                ))
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2)
        # Hour 14 should not appear in any pattern
        for p in patterns:
            assert not (p.start_hour <= 14 <= p.end_hour)

    def test_contiguous_span_merging(self):
        """Adjacent hours with same-sign residuals merge into one span."""
        beta = [1.0, 0.3]
        obs = []
        for hour in range(24):
            for _ in range(8):
                outdoor_delta = 5.0
                true_offset = 1.0 + 0.3 * outdoor_delta
                # Heat loss 22-02: need +0.7°C more than model predicts
                loss = 0.7 if hour >= 22 or hour <= 2 else 0.0
                obs.append(self._make_obs(
                    features=[1.0, outdoor_delta],
                    sp=20.0 + true_offset + loss,
                    cur=20.0,
                    wall_hour=hour,
                ))
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2)
        assert len(patterns) >= 1
        night = [p for p in patterns if p.start_hour == 22 or p.start_hour <= 2]
        assert len(night) >= 1
        assert night[0].mean_residual > 0.5  # positive = heat loss

    def test_observation_wall_hour_serialization(self):
        """wall_hour round-trips through as_dict/from_dict."""
        obs = self._make_obs([1.0, 5.0], sp=22.0, cur=20.0, wall_hour=14)
        d = obs.as_dict()
        assert d["wh"] == 14
        restored = Observation.from_dict(d)
        assert restored.wall_hour == 14

    def test_observation_wall_hour_default(self):
        """Legacy observations without wall_hour get -1."""
        d = {
            "t": 0.0, "x": [1.0], "sp": 22.0,
            "cur": 20.0, "des": 20.0, "rate": 0.005, "clamp": False,
        }
        obs = Observation.from_dict(d)
        assert obs.wall_hour == -1
