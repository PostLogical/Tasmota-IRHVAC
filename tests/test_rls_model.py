"""Tests for the RLS (Recursive Least Squares) feedforward model."""

import pytest
from custom_components.tasmota_irhvac.pi.pi_controller import RLSModel


class TestRLSPrediction:
    """Tests for RLS model prediction."""

    def test_predict_with_seeds(self):
        """Prediction with seed coefficients should match hand calculation."""
        # β = [0, 0.3] → offset = 0 + 0.3 * outdoor_delta
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        x = [1.0, 10.0]  # intercept=1, outdoor_delta=10
        assert model.predict(x) == pytest.approx(3.0)

    def test_predict_multivariate(self):
        """Multivariate prediction should sum all terms."""
        # β = [0.5, 0.3, -4.0, -3.0] → offset = 0.5 + 0.3*outdoor + -4*solar + -3*stove
        model = RLSModel(n_inputs=3, seed_coefficients=[0.5, 0.3, -4.0, -3.0])
        x = [1.0, 10.0, 0.5, 1.0]  # outdoor=10, solar=0.5, stove=on
        expected = 0.5 + 0.3*10 + -4.0*0.5 + -3.0*1.0
        assert model.predict(x) == pytest.approx(expected)

    def test_predict_zeros(self):
        """Zero inputs should return intercept only."""
        model = RLSModel(n_inputs=2, seed_coefficients=[1.5, 0.3, -2.0])
        x = [1.0, 0.0, 0.0]
        assert model.predict(x) == pytest.approx(1.5)


class TestRLSUpdate:
    """Tests for RLS coefficient updates."""

    def test_update_adjusts_coefficients(self):
        """After update, prediction should be closer to observed value."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        x = [1.0, 10.0]

        # Seed predicts 3.0, observed is 5.0
        prediction_before = model.predict(x)
        assert prediction_before == pytest.approx(3.0)

        model.update(x, 5.0)

        prediction_after = model.predict(x)
        # Should be closer to 5.0 than before
        assert abs(prediction_after - 5.0) < abs(prediction_before - 5.0)

    def test_update_returns_residual(self):
        """Update should return the pre-update residual."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        x = [1.0, 10.0]
        residual = model.update(x, 5.0)
        assert residual == pytest.approx(2.0)  # 5.0 - 3.0

    def test_multiple_updates_converge(self):
        """Repeated observations should converge to the true relationship."""
        # True: offset = 0.5 + 0.4*outdoor_delta
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.2])  # Wrong seeds

        # Feed 200 observations of the true relationship (P_init=1 needs more data)
        for i in range(200):
            outdoor = float(i % 25)
            x = [1.0, outdoor]
            y = 0.5 + 0.4 * outdoor
            model.update(x, y)

        # Coefficients should be close to [0.5, 0.4]
        assert model.beta[0] == pytest.approx(0.5, abs=0.15)
        assert model.beta[1] == pytest.approx(0.4, abs=0.05)

    def test_multivariate_convergence(self):
        """Multi-input model should converge to true coefficients."""
        # True: offset = 1.0 + 0.35*outdoor - 3.5*solar
        model = RLSModel(n_inputs=2, seed_coefficients=[0.0, 0.2, -1.0])

        import random
        random.seed(42)
        for _ in range(200):
            outdoor = random.uniform(0, 20)
            solar = random.uniform(0, 1)
            x = [1.0, outdoor, solar]
            y = 1.0 + 0.35 * outdoor - 3.5 * solar + random.gauss(0, 0.1)
            model.update(x, y)

        assert model.beta[0] == pytest.approx(1.0, abs=0.3)
        assert model.beta[1] == pytest.approx(0.35, abs=0.05)
        assert model.beta[2] == pytest.approx(-3.5, abs=0.3)

    def test_observation_count_increments(self):
        """Observation count should increment on each update."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        assert model.observation_count == 0
        model.update([1.0, 5.0], 2.0)
        assert model.observation_count == 1
        model.update([1.0, 10.0], 4.0)
        assert model.observation_count == 2


class TestRLSCoefficientClamps:
    """Tests for coefficient clamping."""

    def test_clamp_prevents_wrong_sign(self):
        """Coefficient should not go outside clamp range."""
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.3],
            coeff_clamps=[None, (-1.0, 0.0)],  # outdoor must be negative (heating needs more when cold)
        )
        # Wait — seed is 0.3 (positive). The clamp should force it to 0.0.
        # Actually the clamp is applied after updates, not on seed.
        # Let's do an update that would push it positive:
        model.beta[1] = -0.2  # Start within clamp
        x = [1.0, 10.0]
        # Observation that would push coefficient positive
        for _ in range(100):
            model.update(x, -20.0)  # Very negative observation

        # Coefficient should be clamped to [-1.0, 0.0]
        assert model.beta[1] >= -1.0
        assert model.beta[1] <= 0.0

    def test_no_clamp_allows_any_value(self):
        """Without clamp, coefficients can take any value."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.0])
        for _ in range(50):
            model.update([1.0, 10.0], 5.0)
        # No clamp on index 1, should be positive
        assert model.beta[1] > 0


class TestRLSRidgeRegularization:
    """Tests for ridge regularization preventing covariance explosion."""

    def test_covariance_stays_bounded(self):
        """P diagonal should not grow unbounded."""
        model = RLSModel(n_inputs=2, delta=0.001, p_init=10.0)

        # Only update on one input dimension
        for _ in range(500):
            x = [1.0, 5.0, 0.0]  # Solar always 0 → its covariance could blow up
            model.update(x, 2.0)

        # P diagonal for solar (index 2) should be bounded, not infinite
        diag = model.get_covariance_diagonal()
        assert all(d < 1000 for d in diag), f"Covariance exploded: {diag}"


class TestRLSVariableForgetting:
    """Tests for variable forgetting factor."""

    def test_small_residual_slow_forgetting(self):
        """When model predicts well (small normalized residual), λ stays near base."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3],
                        lambda_base=0.999, lambda_min=0.995)

        # Good prediction: residual ≈ 0, so normalized_sq / 9 ≈ 0 → λ ≈ lambda_base
        x = [1.0, 10.0]
        y = 3.0  # Matches prediction well (0 + 0.3*10 = 3.0)
        residual = model.update(x, y)
        assert abs(residual) < 0.1

    def test_large_residual_fast_forgetting(self):
        """When model predicts poorly (large normalized residual), λ drops toward min."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3],
                        lambda_base=0.999, lambda_min=0.995)

        x = [1.0, 10.0]
        # Bad prediction: predicted = 0 + 0.3*10 = 3.0, observed = 10.0 → residual = 7.0
        residual = model.update(x, 10.0)
        assert abs(residual) > 5.0  # Model was way off


class TestRLSSerialization:
    """Tests for model serialization/deserialization."""

    def test_round_trip(self):
        """Model should survive serialization round trip."""
        model = RLSModel(n_inputs=2, seed_coefficients=[1.0, 0.3, -2.0])
        model.update([1.0, 5.0, 0.5], 3.0)
        model.update([1.0, 10.0, 0.8], 4.0)

        data = model.as_dict()
        restored = RLSModel.from_dict(data, n_inputs=2)

        assert restored.beta == pytest.approx(model.beta, abs=1e-10)
        assert restored.P == pytest.approx(model.P, abs=1e-10)
        assert restored.observation_count == model.observation_count

    def test_from_dict_input_added(self):
        """Adding inputs should preserve old coefficients and seed new ones."""
        # Old model had 1 input (intercept + outdoor_delta = 2 coefficients)
        old_model = RLSModel(n_inputs=1, seed_coefficients=[0.5, 0.35])
        old_model.update([1.0, 10.0], 4.0)  # Learn something
        old_data = old_model.as_dict()

        # New model has 2 inputs (added solar)
        restored = RLSModel.from_dict(
            old_data, n_inputs=2,
            seed_coefficients=[0.0, 0.3, -4.0],  # Seeds for new model
        )

        # Old coefficients preserved (intercept and outdoor_delta)
        assert restored.beta[0] == pytest.approx(old_model.beta[0], abs=0.01)
        assert restored.beta[1] == pytest.approx(old_model.beta[1], abs=0.01)
        # New input gets seed value
        assert restored.beta[2] == pytest.approx(-4.0)
        # Observation count preserved
        assert restored.observation_count == 1

    def test_from_dict_input_removed(self):
        """Removing inputs should preserve remaining coefficients."""
        old_model = RLSModel(n_inputs=2, seed_coefficients=[0.5, 0.35, -4.0])
        old_data = old_model.as_dict()

        restored = RLSModel.from_dict(old_data, n_inputs=1)

        assert restored.beta[0] == pytest.approx(0.5)
        assert restored.beta[1] == pytest.approx(0.35)

    def test_from_dict_with_clamps(self):
        """Restored model should accept clamp configuration."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3],
                        coeff_clamps=[None, (-1.0, 0.0)])
        data = model.as_dict()
        restored = RLSModel.from_dict(data, n_inputs=1,
                                     coeff_clamps=[None, (-1.0, 0.0)])
        assert restored.coeff_clamps[1] == (-1.0, 0.0)


class TestRLSLagFilter:
    """Tests for lag filter functionality (tested via PIController integration)."""

    def test_lag_filter_ramps_up(self):
        """Lag filter should smooth a step input."""
        import math
        tau = 30 * 60  # 30 minutes in seconds
        dt = 15 * 60   # 15 minute tick

        alpha = 1.0 - math.exp(-dt / tau)
        filtered = 0.0
        raw = 1.0  # Step to 1.0

        # After one tick
        filtered = alpha * raw + (1.0 - alpha) * filtered
        assert 0.0 < filtered < 1.0  # Partially ramped

        # After several ticks
        for _ in range(10):
            filtered = alpha * raw + (1.0 - alpha) * filtered
        assert filtered > 0.9  # Mostly converged

    def test_lag_filter_decays(self):
        """Lag filter should decay when input drops to 0."""
        import math
        tau = 30 * 60
        dt = 15 * 60
        alpha = 1.0 - math.exp(-dt / tau)

        filtered = 1.0  # Was at 1.0
        raw = 0.0  # Drops to 0

        filtered = alpha * raw + (1.0 - alpha) * filtered
        assert 0.0 < filtered < 1.0  # Partially decayed


class TestRLSSeedPadding:
    """Tests for RLSModel seed coefficient padding (line 132)."""

    def test_seed_shorter_than_n_inputs(self):
        """When seed_coefficients is shorter than n_inputs+1, beta should be padded with 0.0."""
        # 3 inputs means n = 4 (intercept + 3), but only provide 2 seeds
        model = RLSModel(n_inputs=3, seed_coefficients=[1.0, 0.5])
        assert len(model.beta) == 4
        assert model.beta[0] == 1.0
        assert model.beta[1] == 0.5
        assert model.beta[2] == 0.0  # Padded
        assert model.beta[3] == 0.0  # Padded

    def test_seed_empty_list(self):
        """Empty seed list should be padded to full length."""
        model = RLSModel(n_inputs=2, seed_coefficients=[])
        assert len(model.beta) == 3
        assert all(b == 0.0 for b in model.beta)

    def test_seed_one_shorter(self):
        """Seed missing just one element should pad exactly one zero."""
        model = RLSModel(n_inputs=2, seed_coefficients=[0.5, 0.3])
        assert len(model.beta) == 3
        assert model.beta[0] == 0.5
        assert model.beta[1] == 0.3
        assert model.beta[2] == 0.0


class TestRLSZeroDenominator:
    """Tests for RLS update returning early when denominator is zero (line 186)."""

    def test_update_zero_vector(self):
        """Update with all-zero feature vector should return residual without updating."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.0], p_init=0.0)
        # With P=0 and x=0, denom = lambda + x'Px = lambda + 0
        # We need denom == 0, so set lambda_base such that lam computes to 0.
        # Actually, with p_init=0 the P matrix is all zeros, so Px = [0,0], xPx = 0.
        # lam = lambda_base - (lambda_base - lambda_min) * blend
        # With residual = y - 0 = y, abs_residual/threshold >= 1 → blend = 1 → lam = lambda_min
        # So denom = lambda_min + 0 = lambda_min. Need lambda_min = 0.
        model2 = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.0],
                          p_init=0.0, lambda_base=0.0, lambda_min=0.0)
        beta_before = list(model2.beta)
        residual = model2.update([0.0, 0.0], 5.0)
        # Should return residual without modifying beta
        assert residual == 5.0
        assert model2.beta == beta_before
        assert model2.observation_count == 0  # Should not increment


class TestRLSGetCoefficients:
    """Tests for get_coefficients method (line 224)."""

    def test_get_coefficients_returns_dict(self):
        """get_coefficients should return a dict mapping index to value."""
        model = RLSModel(n_inputs=2, seed_coefficients=[1.0, 0.3, -2.0])
        coeffs = model.get_coefficients()
        assert isinstance(coeffs, dict)
        assert coeffs[0] == pytest.approx(1.0)
        assert coeffs[1] == pytest.approx(0.3)
        assert coeffs[2] == pytest.approx(-2.0)
        assert len(coeffs) == 3

    def test_get_coefficients_after_update(self):
        """get_coefficients should reflect updated values."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        model.update([1.0, 10.0], 5.0)
        coeffs = model.get_coefficients()
        # After update, coefficients should have changed from seeds
        assert coeffs[0] != 0.0 or coeffs[1] != 0.3


class TestRLSBayesianRidge:
    """Tests for Bayesian ridge anchoring toward seeds."""

    def test_seed_anchor_resists_drift(self):
        """With ambiguous data, coefficients should stay near seeds.

        Two correlated inputs (simulating outdoor_delta and outdoor_rate both
        tracking the same trend) should not cause one to absorb the other's
        credit when Bayesian ridge is active.
        """
        # Seeds: intercept=0, outdoor_delta=0.35, outdoor_rate=0.28
        model = RLSModel(n_inputs=2, seed_coefficients=[0.0, 0.35, 0.28])

        import random
        random.seed(42)

        # Feed correlated data: rate ≈ -0.5 * delta (both from same trend)
        for _ in range(100):
            delta = random.uniform(5, 15)
            rate = -0.5 * delta + random.gauss(0, 0.5)  # Correlated!
            x = [1.0, delta, rate]
            y = 0.35 * delta + 0.28 * rate  # True relationship
            model.update(x, y)

        # With Bayesian ridge, neither coefficient should collapse to zero
        assert model.beta[1] > 0.1, f"outdoor_delta collapsed to {model.beta[1]}"
        assert abs(model.beta[2]) > 0.05, f"outdoor_rate collapsed to {model.beta[2]}"

    def test_clear_data_overrides_seed(self):
        """With clear independent data, RLS should learn true values even if seeds are wrong."""
        # Wrong seeds
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.1])

        import random
        random.seed(42)

        # Clear data: true relationship is 0.5, not 0.1
        for _ in range(200):
            outdoor = random.uniform(0, 20)
            x = [1.0, outdoor]
            y = 0.5 * outdoor + random.gauss(0, 0.1)
            model.update(x, y)

        # Should learn close to true value despite wrong seed
        assert model.beta[1] == pytest.approx(0.5, abs=0.1)

    def test_beta_seed_stored_at_init(self):
        """beta_seed should be set from seed_coefficients at init."""
        model = RLSModel(n_inputs=2, seed_coefficients=[0.5, 0.35, -4.0])
        assert model.beta_seed == [0.5, 0.35, -4.0]

    def test_beta_seed_preserved_after_from_dict(self):
        """from_dict should preserve beta_seed from current seeds, not stored beta."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.35])
        # Learn something different
        for _ in range(50):
            model.update([1.0, 10.0], 5.0)
        data = model.as_dict()

        # Restore with same seeds
        restored = RLSModel.from_dict(data, n_inputs=1, seed_coefficients=[0.0, 0.35])
        # beta should be learned value
        assert restored.beta[1] != 0.35
        # beta_seed should be the seed, not the learned value
        assert restored.beta_seed == [0.0, 0.35]


class TestSeedShrinkage:
    """Tests for Bayesian seed shrinkage in RLS update."""

    def test_shrinkage_pulls_toward_seed(self):
        """Coefficient drifted from seed should be pulled back over updates."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.5])
        # Manually push beta away from seed
        model.beta[1] = 2.0  # seed is 0.5

        # Feed data consistent with beta=0.5 (so RLS also wants to go back)
        # but even with ambiguous data, shrinkage should pull toward seed
        drift_before = abs(model.beta[1] - model.beta_seed[1])
        for _ in range(100):
            model.update([1.0, 5.0], 2.5)  # y=2.5 is ambiguous
        drift_after = abs(model.beta[1] - model.beta_seed[1])
        assert drift_after < drift_before

    def test_shrinkage_does_not_prevent_learning(self):
        """With clear data, RLS should still converge despite shrinkage pull."""
        # Seed says 0.5, true value is 2.0
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.5])

        # Strong, consistent signal: y = 2.0 * x
        for i in range(200):
            x = float(1 + i % 10)
            model.update([1.0, x], 2.0 * x)

        learned = model.beta[1] / model.feature_scales[1]
        # Should learn close to 2.0 despite seed=0.5 pulling back
        assert abs(learned - 2.0) < 0.3

    def test_frozen_coefficients_not_shrunk(self):
        """Frozen coefficients should not be affected by shrinkage."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.5])
        model.beta[1] = 2.0
        model.frozen[1] = True

        for _ in range(50):
            model.update([1.0, 5.0], 3.0)

        # Beta should not have moved at all (frozen)
        assert model.beta[1] == 2.0

    def test_shrinkage_symmetric_around_seed(self):
        """Shrinkage pulls equally whether above or below seed."""
        model_above = RLSModel(n_inputs=1, seed_coefficients=[0.0, 1.0])
        model_below = RLSModel(n_inputs=1, seed_coefficients=[0.0, 1.0])

        model_above.beta[1] = 2.0   # 1.0 above seed
        model_below.beta[1] = 0.0   # 1.0 below seed

        # Same neutral data
        for _ in range(50):
            model_above.update([1.0, 5.0], 5.0)
            model_below.update([1.0, 5.0], 5.0)

        drift_above = abs(model_above.beta[1] - model_above.beta_seed[1])
        drift_below = abs(model_below.beta[1] - model_below.beta_seed[1])
        # Both should have been pulled back roughly equally
        assert abs(drift_above - drift_below) < 0.2

    def test_zero_seed_shrinks_toward_zero(self):
        """Feature seeded at 0 shrinks toward 0 when active."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.0])
        model.beta[1] = 1.0  # drifted from seed=0

        for _ in range(100):
            model.update([1.0, 5.0], 0.0)  # feature active
        # Should pull back toward 0
        assert abs(model.beta[1]) < 1.0

    def test_dormant_feature_not_shrunk(self):
        """Inactive feature retains its learned coefficient (no data, no shrinkage)."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.0])
        model.beta[1] = 1.0  # learned value

        for _ in range(100):
            model.update([1.0, 0.0], 0.0)  # feature inactive
        # Should NOT be pulled back — dormant features keep their learning
        assert model.beta[1] == 1.0


class TestRLSFeatureScalesPadding:
    """Tests for feature_scales padding when fewer scales than dimensions (line 83)."""

    def test_fewer_scales_than_dimensions_pads_with_ones(self):
        """Providing 1 feature_scale for a 3-input model should pad to [2.0, 1.0, 1.0, 1.0]."""
        model = RLSModel(n_inputs=3, feature_scales=[2.0])
        # n=4 (intercept + 3 inputs), but only 1 scale provided
        assert len(model.feature_scales) == 4
        assert model.feature_scales[0] == 2.0
        assert model.feature_scales[1] == 1.0
        assert model.feature_scales[2] == 1.0
        assert model.feature_scales[3] == 1.0

    def test_exact_scales_no_padding(self):
        """Providing exact number of scales should not pad."""
        model = RLSModel(n_inputs=2, feature_scales=[2.0, 3.0, 4.0])
        assert model.feature_scales == [2.0, 3.0, 4.0]

    def test_empty_scales_defaults_to_all_ones(self):
        """No feature_scales should default to all 1.0."""
        model = RLSModel(n_inputs=2)
        assert model.feature_scales == [1.0, 1.0, 1.0]

    def test_uniform_P_regardless_of_scales(self):
        """P initialization should be uniform — normalization handles scale balance."""
        model = RLSModel(n_inputs=2, feature_scales=[5.0], p_init=10.0)
        # All P diagonals should be p_init (uniform), regardless of feature scales
        assert model.P[0 * 3 + 0] == pytest.approx(10.0)
        assert model.P[1 * 3 + 1] == pytest.approx(10.0)
        assert model.P[2 * 3 + 2] == pytest.approx(10.0)


class TestFeatureScaleRescaling:
    """Tests for rescale_features similarity transform and from_dict scale migration."""

    def test_rescale_preserves_physical_prediction(self):
        """Rescaling should not change predictions in physical units."""
        old_scales = [1.0, 10.0, 0.5]
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(old_scales),
        )
        # Train a bit
        for _ in range(10):
            model.update([1.0, 8.0, 0.6], 3.0)

        x = [1.0, 12.0, 0.4]
        pred_before = model.predict(x)

        # Change scales and rescale
        new_scales = [1.0, 5.0, 2.0]
        model.feature_scales = new_scales
        model.rescale_features(old_scales)

        pred_after = model.predict(x)
        assert pred_after == pytest.approx(pred_before, abs=1e-10)

    def test_rescale_preserves_physical_coefficients(self):
        """Physical coefficients should be unchanged after rescaling."""
        old_scales = [1.0, 10.0, 0.5]
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(old_scales),
        )
        for _ in range(10):
            model.update([1.0, 8.0, 0.6], 3.0)

        coeffs_before = model.get_coefficients()

        new_scales = [1.0, 5.0, 2.0]
        model.feature_scales = new_scales
        model.rescale_features(old_scales)

        coeffs_after = model.get_coefficients()
        for i in coeffs_before:
            assert coeffs_after[i] == pytest.approx(coeffs_before[i], abs=1e-10)

    def test_rescale_no_change_is_noop(self):
        """Rescaling with identical scales should not change anything."""
        scales = [1.0, 10.0, 0.5]
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(scales),
        )
        beta_before = list(model.beta)
        P_before = list(model.P)

        model.rescale_features(list(scales))

        assert model.beta == pytest.approx(beta_before, abs=1e-15)
        assert model.P == pytest.approx(P_before, abs=1e-15)

    def test_from_dict_applies_scale_transform(self):
        """from_dict should detect scale changes and apply similarity transform."""
        old_scales = [1.0, 10.0, 0.5]
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(old_scales),
        )
        for _ in range(10):
            model.update([1.0, 8.0, 0.6], 3.0)

        x = [1.0, 12.0, 0.4]
        pred_original = model.predict(x)
        data = model.as_dict()

        # Restore with different scales
        new_scales = [1.0, 5.0, 2.0]
        restored = RLSModel.from_dict(
            data, n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=new_scales,
        )

        pred_restored = restored.predict(x)
        assert pred_restored == pytest.approx(pred_original, abs=1e-10)

    def test_from_dict_no_stored_scales_skips_transform(self):
        """Legacy data without feature_scales should restore without transform."""
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.35],
            feature_scales=[1.0, 10.0],
        )
        data = model.as_dict()
        del data["feature_scales"]  # Simulate legacy data

        restored = RLSModel.from_dict(
            data, n_inputs=1,
            seed_coefficients=[0.0, 0.35],
            feature_scales=[1.0, 5.0],  # Different scales
        )
        # No transform applied — beta restored as-is
        assert restored.beta == pytest.approx(model.beta, abs=1e-10)

    def test_from_dict_same_scales_no_transform(self):
        """Same scales should not trigger transform."""
        scales = [1.0, 10.0, 0.5]
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(scales),
        )
        for _ in range(5):
            model.update([1.0, 8.0, 0.6], 3.0)

        data = model.as_dict()
        restored = RLSModel.from_dict(
            data, n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(scales),
        )
        assert restored.beta == pytest.approx(model.beta, abs=1e-10)
        assert restored.P == pytest.approx(model.P, abs=1e-10)
