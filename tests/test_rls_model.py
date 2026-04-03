"""Tests for the RLS (Recursive Least Squares) feedforward model."""

import pytest
from custom_components.tasmota_irhvac.pi_controller import RLSModel


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
        """When model predicts well, forgetting should be slow (λ near base)."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3],
                        lambda_base=0.999, lambda_min=0.995, residual_threshold=3.0)

        # Good prediction: residual ≈ 0
        x = [1.0, 10.0]
        y = 3.0  # Matches prediction
        residual = model.update(x, y)
        assert abs(residual) < 0.1

    def test_large_residual_fast_forgetting(self):
        """When model predicts poorly, forgetting should be faster."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3],
                        lambda_base=0.999, lambda_min=0.995, residual_threshold=3.0)

        x = [1.0, 10.0]
        # Bad prediction: residual = 7.0 (>> threshold of 3)
        residual = model.update(x, 10.0)
        assert abs(residual) > 5.0  # Model was way off
        # Second update should converge faster due to lower λ


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
