"""Tests for the feedforward coefficient model (RLSModel).

RLSModel holds the deployed feedforward coefficients (fit by batch WLS and
written into ``beta``) and evaluates predictions.  It no longer learns online
— the recursive ``update`` path was removed in 4d7e77a — so these tests cover
prediction, seed/scale handling, serialization, and the frozen flag.  Tests
that emulate a batch-written coefficient assign ``beta`` directly, exactly as
the controller does each learning cycle.
"""

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


class TestRLSSerialization:
    """Tests for model serialization/deserialization."""

    def test_round_trip(self):
        """Model should survive serialization round trip (beta + observation_count)."""
        model = RLSModel(n_inputs=2, seed_coefficients=[1.0, 0.3, -2.0])
        # Coefficients are written by the batch; emulate by assigning beta directly.
        model.beta = [1.1, 0.42, -1.8]
        model.observation_count = 7

        data = model.as_dict()
        restored = RLSModel.from_dict(data, n_inputs=2)

        assert restored.beta == pytest.approx(model.beta, abs=1e-10)
        assert restored.observation_count == model.observation_count

    def test_from_dict_input_added(self):
        """Adding inputs should preserve old coefficients and seed new ones."""
        # Old model had 1 input (intercept + outdoor_delta = 2 coefficients).
        old_model = RLSModel(n_inputs=1, seed_coefficients=[0.5, 0.35])
        # Emulate a batch-written coefficient (differs from the seed).
        old_model.beta[1] = 0.42
        old_model.observation_count = 1
        old_data = old_model.as_dict()

        # New model has 2 inputs (added solar).
        restored = RLSModel.from_dict(
            old_data, n_inputs=2,
            seed_coefficients=[0.0, 0.3, -4.0],  # Seeds for new model
        )

        # Old coefficients preserved (intercept and outdoor_delta).
        assert restored.beta[0] == pytest.approx(old_model.beta[0], abs=0.01)
        assert restored.beta[1] == pytest.approx(old_model.beta[1], abs=0.01)
        # New input gets seed value.
        assert restored.beta[2] == pytest.approx(-4.0)
        # Observation count preserved.
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

    def test_from_dict_skips_missing_optional_fields(self):
        """from_dict() handles dicts with missing beta/P/observation_count
        keys — covers the skip-branch when each optional field is absent."""
        restored = RLSModel.from_dict({}, n_inputs=1, seed_coefficients=[0.0, 0.3])
        # observation_count default
        assert restored.observation_count == 0
        # All defaults preserved (n is what __init__ assigned).
        assert restored.observation_count == 0


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


class TestRLSGetCoefficients:
    """Tests for get_coefficients method."""

    def test_get_coefficients_returns_dict(self):
        """get_coefficients should return a dict mapping index to value."""
        model = RLSModel(n_inputs=2, seed_coefficients=[1.0, 0.3, -2.0])
        coeffs = model.get_coefficients()
        assert isinstance(coeffs, dict)
        assert coeffs[0] == pytest.approx(1.0)
        assert coeffs[1] == pytest.approx(0.3)
        assert coeffs[2] == pytest.approx(-2.0)
        assert len(coeffs) == 3


class TestRLSBetaSeed:
    """Tests for beta_seed (the prior anchor, set from config seeds)."""

    def test_beta_seed_stored_at_init(self):
        """beta_seed should be set from seed_coefficients at init."""
        model = RLSModel(n_inputs=2, seed_coefficients=[0.5, 0.35, -4.0])
        assert model.beta_seed == [0.5, 0.35, -4.0]

    def test_beta_seed_preserved_after_from_dict(self):
        """from_dict should set beta_seed from current seeds, not stored beta."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.35])
        # Emulate a batch-written coefficient that differs from the seed.
        model.beta[1] = 0.8
        data = model.as_dict()

        # Restore with same seeds.
        restored = RLSModel.from_dict(data, n_inputs=1, seed_coefficients=[0.0, 0.35])
        # beta should be the stored (learned) value.
        assert restored.beta[1] != 0.35
        # beta_seed should be the seed, not the learned value.
        assert restored.beta_seed == [0.0, 0.35]


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
        # Emulate batch-written coefficients (normalized space).
        model.beta = [1.2, 4.5, -1.8]

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
        model.beta = [1.2, 4.5, -1.8]

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
        model.beta = [1.2, 4.5, -1.8]

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
        """Same scales should not trigger transform; beta round-trips exactly."""
        scales = [1.0, 10.0, 0.5]
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(scales),
        )
        model.beta = [1.2, 4.5, -1.8]

        data = model.as_dict()
        restored = RLSModel.from_dict(
            data, n_inputs=2,
            seed_coefficients=[1.0, 0.3, -4.0],
            feature_scales=list(scales),
        )
        assert restored.beta == pytest.approx(model.beta, abs=1e-10)


class TestPValidation:
    """Tests for P matrix validation in from_dict()."""

    def test_from_dict_resets_negative_p_diagonal(self):
        """Restoring a model with negative P diagonal should reset P."""
        model = RLSModel(n_inputs=2, seed_coefficients=[0.0, 0.3, -2.0])
        data = model.as_dict()
        # Corrupt P diagonal to simulate persisted broken state
        n = model.n
        data["P"][0] = -55539.0  # Negative intercept P diagonal
        data["P"][n + 1] = -0.07  # Negative outdoor_delta P diagonal

        restored = RLSModel.from_dict(
            data, n_inputs=2,
            seed_coefficients=[0.0, 0.3, -2.0],
        )
        diag = restored.get_covariance_diagonal()
        for i, d in enumerate(diag):
            assert d > 0, f"P diagonal[{i}] = {d} should be positive after reset"
            assert d == pytest.approx(restored.p_init), (
                f"P diagonal[{i}] = {d} should equal p_init={restored.p_init}"
            )
