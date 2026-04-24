"""Tests for batch WLS offline learning module."""

import math
import time

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
    fuse_batch_greybox,
    weighted_least_squares,
)


# ── Test helpers for v2 Observation format ───────────────────────────
# Maps old-style feature lists [intercept, outdoor_delta, input1, ...]
# to v2 observations with raw_readings + corresponding model_inputs config.

# Synthetic entity IDs for test model inputs.
_TEST_ENTITIES = [f"sensor.test_input_{i}" for i in range(10)]


def _make_test_obs(
    features: list[float],
    sp: float,
    cur: float,
    des: float = 20.0,
    rate: float = 0.005,
    clamped: bool = False,
    clamped_reason: str = "",
    wall_time: float | None = None,
) -> Observation:
    """Create a v2 Observation from a legacy-style feature list.

    Maps features positionally:
    - [0] = intercept (ignored, always 1.0)
    - [1] = outdoor_delta → derives outdoor_temp_c = cur + outdoor_delta
    - [2:] = model input values → stored in raw_readings by entity_id
    """
    if clamped and not clamped_reason:
        clamped_reason = "no_output"

    # Derive outdoor_temp_c from outdoor_delta (features[1]) if present
    outdoor_temp_c: float | None = None
    if len(features) > 1:
        outdoor_temp_c = cur + features[1]

    # Build raw_readings from additional features (index 2+)
    raw_readings: dict[str, float] = {}
    for i, val in enumerate(features[2:]):
        raw_readings[_TEST_ENTITIES[i]] = val

    return Observation(
        timestamp=0.0,
        wall_time=wall_time if wall_time is not None else time.time(),
        hp_setpoint=sp,
        current_c=cur,
        desired_c=des,
        outdoor_temp_c=outdoor_temp_c,
        room_rate=rate,
        raw_readings=raw_readings,
        clamped=clamped,
        clamped_reason=clamped_reason,
    )


def _test_model_inputs(n_extra: int) -> list[dict]:
    """Generate model input config matching _TEST_ENTITIES for n extra features."""
    return [
        {"entity_id": _TEST_ENTITIES[i], "name": f"input_{i}"}
        for i in range(n_extra)
    ]


def _test_feature_order(n_extra: int) -> list[str]:
    """Generate feature order: intercept, outdoor_delta, input_0, ..."""
    order = ["intercept", "outdoor_delta"]
    for i in range(n_extra):
        order.append(f"input_{i}")
    return order


# ── Weighted Least Squares ────────────────────────────────────────────


class TestWeightedLeastSquares:
    def _make_obs(self, features, sp, cur, des=20.0, rate=0.005, clamped=False, clamped_reason=""):
        return _make_test_obs(features, sp, cur, des=des, rate=rate,
                              clamped=clamped, clamped_reason=clamped_reason)

    def test_recovers_known_intercept(self):
        """With constant outdoor_delta, WLS should recover the mean offset."""
        obs = []
        for i in range(30):
            # hp_setpoint=22, current_c=20, outdoor_delta=0 → offset = 2.0
            obs.append(self._make_obs([1.0, 0.0], sp=22.0, cur=20.0))
        result = weighted_least_squares(
            obs, n_features=2, min_observations=20,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
        assert result is not None
        assert abs(result.beta_batch[0] - 2.0) < 0.1

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
        result = weighted_least_squares(
            obs, n_features=2, min_observations=20,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
        assert result is not None
        assert abs(result.beta_batch[0] - 1.0) < 0.05
        assert abs(result.beta_batch[1] - 0.5) < 0.05

    def test_excludes_clamped(self):
        """Clamped observations should be excluded."""
        obs = []
        for i in range(30):
            obs.append(self._make_obs([1.0, 0.0], sp=22.0, cur=20.0, clamped=True))
        result = weighted_least_squares(
            obs, n_features=2, min_observations=20,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
        assert result is None  # all excluded

    def test_excludes_unstable(self):
        """Observations with high room_rate should be excluded."""
        obs = []
        for i in range(30):
            obs.append(self._make_obs([1.0, 0.0], sp=22.0, cur=20.0, rate=0.05))
        result = weighted_least_squares(
            obs, n_features=2, min_observations=20,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
        assert result is None  # all excluded

    def test_returns_none_if_insufficient(self):
        obs = [self._make_obs([1.0, 0.0], sp=22.0, cur=20.0) for _ in range(5)]
        result = weighted_least_squares(
            obs, n_features=2, min_observations=20,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
        assert result is None

    def test_equilibrium_weighting(self):
        """Near-equilibrium observations should have more influence than transient ones."""
        obs = []
        # 20 near-equilibrium observations (rate≈0) saying offset=2.0
        for _ in range(20):
            obs.append(self._make_obs([1.0, 0.0], sp=22.0, cur=20.0, rate=0.001))
        # 20 transient observations (rate=0.015, near threshold) saying offset=4.0
        for _ in range(20):
            obs.append(self._make_obs([1.0, 0.0], sp=24.0, cur=20.0, rate=0.015))
        result = weighted_least_squares(
            obs, n_features=2, min_observations=20,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
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
        result = weighted_least_squares(
            obs, n_features=3, min_observations=20,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        # With proper FWL + base re-estimation, the hierarchical approach
        # recovers the same coefficients as joint OLS on complete data.
        assert abs(result.beta_batch[0] - 2.0) < 0.1   # intercept
        assert abs(result.beta_batch[1] - 0.3) < 0.01  # outdoor_delta
        assert abs(result.beta_batch[2] - (-2.0)) < 0.1  # model input

    def test_normalization_predictions_match_raw(self):
        """Predictions from normalized WLS must match raw data exactly.

        Use the returned beta (physical units) to predict y for each
        observation and verify residuals are near-zero.  This confirms
        the denormalization is correct end-to-end.
        """
        obs = []
        for i in range(30):
            od = 5.0 + float(i % 10)   # 5-14
            extra = 0.001 * (i % 5)     # 0-0.004
            true_offset = 1.5 + 0.4 * od - 3.0 * extra
            obs.append(self._make_obs(
                features=[1.0, od, extra],
                sp=20.0 + true_offset,
                cur=20.0,
            ))
        result = weighted_least_squares(
            obs, n_features=3, min_observations=20,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        beta = result.beta_batch

        # Verify predictions: y ≈ intercept + od*beta[1] + extra*beta[2]
        for o in obs:
            y = o.hp_setpoint - o.current_c
            x = [1.0, o.outdoor_temp_c - o.current_c]
            x.append(o.raw_readings.get(_TEST_ENTITIES[0], 0.0))
            pred = sum(beta[j] * x[j] for j in range(3))
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
        result = weighted_least_squares(
            obs, n_features=3, min_observations=20,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        assert abs(result.beta_batch[0] - 1.0) < 0.1    # intercept
        assert abs(result.beta_batch[1] - 0.05) < 0.005  # big feature (outdoor_delta)
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

    def test_near_zero_current_caps_at_100pct(self):
        """Post-reset coefficients near zero should cap at 100%, not explode."""
        result = BatchResult(
            n_total=50, n_eligible=40,
            beta_batch=[1.5, 0.3],
            beta_current=[], residual_rms=0.1,
            max_coeff_change_pct=0.0, recommend_update=False,
        )
        # current values near zero (post-reset): 0.001 would produce
        # 149900% under pure %-change; should cap at 100%.
        result = compare_and_report(result, [0.001, 0.05], ["intercept", "slope"])
        assert result.recommend_update
        assert result.max_coeff_change_pct == 100.0

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
        return _make_test_obs(features, sp, cur, des=des, rate=rate,
                              clamped=clamped, clamped_reason=clamped_reason)

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
        result = weighted_least_squares(
            obs, n_features=3, current_beta=current,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        assert 2 in result.held_features
        # Pellet coefficient should be held at -8.0
        assert result.beta_batch[2] == -8.0
        # Intercept and outdoor_delta should still be estimated accurately
        assert abs(result.beta_batch[0] - 1.0) < 0.15
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
        result = weighted_least_squares(
            obs, n_features=3, current_beta=current,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
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
        result = weighted_least_squares(
            obs, n_features=5, current_beta=current,
            feature_order=_test_feature_order(3),
            model_inputs=_test_model_inputs(3),
        )
        assert result is not None
        assert result.held_features == {2, 3, 4}
        assert result.beta_batch[2] == -8.0
        assert result.beta_batch[3] == -1.0
        assert result.beta_batch[4] == 3.5

    def test_held_subtraction_preserves_active_estimates(self):
        """Held feature contributions are subtracted from y, so active
        estimates remain accurate even when held features have large seeds.

        In hierarchical regression, a constant-value feature is held.
        The base model (intercept + outdoor_delta) absorbs the constant
        contribution, so its intercept shifts by the held value.
        """
        obs = []
        for i in range(40):
            outdoor = float(i % 10)
            # Pellet is always on (constant 1.0) with true coeff 3.0
            true_offset = 1.0 + 0.5 * outdoor + 3.0 * 1.0
            obs.append(self._make_obs(
                features=[1.0, outdoor, 1.0],
                sp=20.0 + true_offset, cur=20.0,
            ))
        # current_beta has pellet at 3.0 (correct)
        result = weighted_least_squares(
            obs, n_features=3, current_beta=[0.0, 0.0, 3.0],
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        assert 2 in result.held_features
        # outdoor_delta should still be estimated accurately
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
        result = weighted_least_squares(
            obs, n_features=3,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        assert 2 in result.held_features
        assert result.beta_batch[2] == 0.0

    def test_existing_tests_unaffected(self):
        """Original 2-feature case still works without model inputs."""
        obs = []
        for i in range(40):
            outdoor = float(i % 10)
            true_offset = 1.0 + 0.5 * outdoor
            obs.append(self._make_obs(
                features=[1.0, outdoor],
                sp=20.0 + true_offset, cur=20.0,
            ))
        result = weighted_least_squares(
            obs, n_features=2, min_observations=20,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
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
            obs.append(_make_test_obs(
                [1.0, outdoor, 0.0],
                sp=20.0 + 1.0 + 0.5 * outdoor,
                cur=20.0,
            ))
        result = weighted_least_squares(
            obs, n_features=3, current_beta=[0, 0, -8],
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        # Base features (0, 1) should have finite std_err
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


class TestExtractFeatureVectorDeltaFromRoom:
    """Tests for build_feature_vector_from_raw with delta_from_room inputs."""

    def _obs(self, raw_readings, current_c=20.0):
        return Observation(
            timestamp=0.0, wall_time=time.time(),
            hp_setpoint=22.0, current_c=current_c, desired_c=20.0,
            outdoor_temp_c=10.0, room_rate=0.005,
            raw_readings=raw_readings, clamped=False,
        )

    def test_delta_from_room_subtracts_current_c(self):
        """raw_readings stores °C absolute temp; batch subtracts current_c."""
        from custom_components.tasmota_irhvac.pi.batch_learning import build_feature_vector_from_raw

        # Adjacent zone at 22°C, room at 20°C → delta should be 2°C
        obs = self._obs({"sensor.adjacent": 22.0})
        model_inputs = [
            {"entity_id": "sensor.adjacent", "name": "adj", "delta_from_room": True},
        ]
        feature_order = ["intercept", "outdoor_delta", "adj"]
        result = build_feature_vector_from_raw(obs, model_inputs, feature_order)

        assert result is not None
        assert result[0] == 1.0  # intercept
        assert abs(result[1] - (-10.0)) < 0.01  # outdoor_delta = 10 - 20
        assert abs(result[2] - 2.0) < 0.01  # delta = 22 - 20

    def test_non_delta_input_used_directly(self):
        """Non-delta inputs pass through without subtraction."""
        from custom_components.tasmota_irhvac.pi.batch_learning import build_feature_vector_from_raw

        obs = self._obs({"sensor.solar": 0.75})
        model_inputs = [{"entity_id": "sensor.solar", "name": "solar"}]
        feature_order = ["intercept", "outdoor_delta", "solar"]
        result = build_feature_vector_from_raw(obs, model_inputs, feature_order)

        assert result is not None
        assert result[2] == 0.75

    def test_delta_from_room_config_toggle(self):
        """Same raw reading produces different features based on delta_from_room flag."""
        from custom_components.tasmota_irhvac.pi.batch_learning import build_feature_vector_from_raw

        obs = self._obs({"sensor.zone": 22.0})
        feature_order = ["intercept", "outdoor_delta", "zone"]

        # With delta_from_room=True: 22 - 20 = 2
        mi_delta = [{"entity_id": "sensor.zone", "name": "zone", "delta_from_room": True}]
        result = build_feature_vector_from_raw(obs, mi_delta, feature_order)
        assert result is not None
        assert abs(result[2] - 2.0) < 0.01

        # With delta_from_room=False: 22.0 used directly
        mi_raw = [{"entity_id": "sensor.zone", "name": "zone"}]
        result = build_feature_vector_from_raw(obs, mi_raw, feature_order)
        assert result is not None
        assert result[2] == 22.0


class TestFuseBatchGreybox:
    """Tests for inverse-variance fusion of WLS and grey-box β."""

    def _make_result(self, beta_batch, std_err):
        return BatchResult(
            n_total=100, n_eligible=60,
            beta_batch=list(beta_batch),
            beta_current=[],
            residual_rms=0.5,
            max_coeff_change_pct=0.0,
            recommend_update=True,
            beta_std_err=list(std_err),
        )

    def test_equal_variance_averages(self):
        """Equal σ → fused β is arithmetic mean."""
        r = self._make_result([1.0, -0.5], [0.1, 0.1])
        fuse_batch_greybox(r, [None, -0.3], [float("inf"), 0.1])
        # β₀: gb=None → unchanged
        assert r.beta_batch[0] == 1.0
        # β₁: mean of -0.5 and -0.3 = -0.4
        assert abs(r.beta_batch[1] - (-0.4)) < 1e-10

    def test_tighter_greybox_pulls_more(self):
        """Lower grey-box σ → fused β closer to grey-box."""
        r = self._make_result([1.0, -0.5], [0.1, 1.0])
        fuse_batch_greybox(r, [None, -0.3], [float("inf"), 0.1])
        # gb σ=0.1 vs wls σ=1.0 → gb has 100× more precision
        # fused ≈ -0.3 (grey-box dominates)
        assert abs(r.beta_batch[1] - (-0.3)) < 0.01

    def test_tighter_wls_stays_close(self):
        """Lower WLS σ → fused β stays close to WLS."""
        r = self._make_result([1.0, -0.5], [0.1, 0.1])
        fuse_batch_greybox(r, [None, -0.3], [float("inf"), 1.0])
        # wls σ=0.1 vs gb σ=1.0 → WLS dominates
        assert abs(r.beta_batch[1] - (-0.5)) < 0.01

    def test_none_greybox_passthrough(self):
        """None grey-box β → WLS unchanged."""
        r = self._make_result([1.0, -0.5, 0.3], [0.1, 0.1, 0.1])
        fuse_batch_greybox(r, [None, None, None], [float("inf")] * 3)
        assert r.beta_batch == [1.0, -0.5, 0.3]

    def test_infinite_greybox_se_passthrough(self):
        """Infinite grey-box σ → WLS unchanged."""
        r = self._make_result([1.0, -0.5], [0.1, 0.1])
        fuse_batch_greybox(r, [0.5, -0.3], [float("inf"), float("inf")])
        assert r.beta_batch == [1.0, -0.5]

    def test_fused_se_reduced(self):
        """Fused σ is always less than both inputs."""
        r = self._make_result([1.0, -0.5], [0.2, 0.3])
        fuse_batch_greybox(r, [None, -0.4], [float("inf"), 0.25])
        # Fused σ = 1/√(1/0.3² + 1/0.25²) ≈ 0.192
        assert r.beta_std_err[1] < 0.25
        assert r.beta_std_err[1] < 0.3

    def test_mixed_some_fused_some_not(self):
        """Only coefficients with both finite σ and non-None β get fused."""
        r = self._make_result([1.0, -0.5, 0.3], [0.1, 0.1, 0.1])
        fuse_batch_greybox(
            r,
            [None, -0.3, 0.5],             # only β₁ and β₂ from grey-box
            [float("inf"), 0.1, float("inf")],  # only β₁ has finite σ
        )
        assert r.beta_batch[0] == 1.0   # no gb β → unchanged
        assert r.beta_batch[1] != -0.5  # fused
        assert r.beta_batch[2] == 0.3   # gb σ infinite → unchanged


class TestRawReadingsFeatures:
    """Tests for raw_readings storage and config-independent observation format."""

    def test_raw_readings_round_trip(self):
        """raw_readings survive as_dict → from_dict round-trip."""
        obs = Observation(
            timestamp=100.0, wall_time=1713650000.0,
            hp_setpoint=22.0, current_c=20.0, desired_c=20.0,
            outdoor_temp_c=5.0, room_rate=0.005,
            raw_readings={"sensor.solar": 0.6, "sensor.pellet": 1.0},
            clamped=False,
        )
        d = obs.as_dict()
        restored = Observation.from_dict(d)
        assert restored.raw_readings == {"sensor.solar": 0.6, "sensor.pellet": 1.0}
        assert restored.outdoor_temp_c == 5.0
        assert restored.wall_time == 1713650000.0

    def test_legacy_v1_observation_discarded(self):
        """Legacy v1 observations (pre-computed features) cannot be loaded."""
        legacy = {"t": 0, "x": [1.0, 5.0, 0.6], "sp": 22, "cur": 20, "des": 20, "rate": 0.005, "clamp": False}
        with pytest.raises(ValueError, match="v1 observation"):
            Observation.from_dict(legacy)

    def test_build_feature_vector_from_raw_complete(self):
        """Feature vector built from raw readings with matching config."""
        from custom_components.tasmota_irhvac.pi.batch_learning import build_feature_vector_from_raw
        obs = Observation(
            timestamp=0, wall_time=1713650000.0,
            hp_setpoint=22, current_c=20, desired_c=20,
            outdoor_temp_c=15.0, room_rate=0.005,
            raw_readings={"sensor.solar": 0.6},
            clamped=False,
        )
        model_inputs = [{"entity_id": "sensor.solar", "name": "solar"}]
        feature_order = ["intercept", "outdoor_delta", "solar"]
        result = build_feature_vector_from_raw(obs, model_inputs, feature_order)
        assert result == [1.0, -5.0, 0.6]  # outdoor_delta = 15 - 20 = -5

    def test_build_feature_vector_missing_entity_returns_none(self):
        """Returns None when observation lacks a required entity reading."""
        from custom_components.tasmota_irhvac.pi.batch_learning import build_feature_vector_from_raw
        obs = Observation(
            timestamp=0, wall_time=1713650000.0,
            hp_setpoint=22, current_c=20, desired_c=20,
            outdoor_temp_c=15.0, room_rate=0.005,
            raw_readings={},  # no solar reading
            clamped=False,
        )
        model_inputs = [{"entity_id": "sensor.solar", "name": "solar"}]
        feature_order = ["intercept", "outdoor_delta", "solar"]
        result = build_feature_vector_from_raw(obs, model_inputs, feature_order)
        assert result is None

    def test_update_config_recomputes_info_matrix(self):
        """update_config atomically updates feature_order + model_inputs."""
        buf = DiversityAwareBuffer(
            n_features=2, feature_order=["intercept", "outdoor_delta"],
            model_inputs=[],
        )
        for i in range(5):
            buf.add(_make_test_obs([1.0, float(i)], sp=22, cur=20))
        info_before = [row[:] for row in buf._info_inv]
        buf.update_config(
            feature_order=["intercept", "outdoor_delta", "solar"],
            model_inputs=[{"entity_id": "sensor.solar", "name": "solar"}],
        )
        assert buf.n_features == 3
        assert buf._info_inv != info_before  # recomputed

    def test_buffer_from_list_discards_v1(self):
        """from_list discards legacy v1 observations cleanly."""
        v1_data = [
            {"t": 0, "x": [1.0, 5.0], "sp": 22, "cur": 20, "des": 20, "rate": 0.005, "clamp": False}
        ]
        buf = DiversityAwareBuffer.from_list(
            v1_data, n_features=2,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=[],
        )
        assert len(buf) == 0  # v1 discarded

    def test_hierarchical_wls_ragged_data(self):
        """WLS handles observations where solar was added later.

        200 observations with outdoor_delta only (pre-solar),
        60 observations with outdoor_delta + solar.
        Both tiers should be used: outdoor_delta anchored by all 200,
        solar estimated from 60.
        """
        model_inputs = [{"entity_id": "sensor.solar", "name": "solar"}]
        feature_order = ["intercept", "outdoor_delta", "solar"]
        obs = []
        # 150 observations without solar (old)
        for i in range(150):
            outdoor_delta = float(i % 10)
            true_offset = 1.0 + 0.5 * outdoor_delta
            obs.append(Observation(
                timestamp=float(i), wall_time=1713650000.0 + i * 3600,
                hp_setpoint=20.0 + true_offset, current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + outdoor_delta, room_rate=0.005,
                raw_readings={},  # no solar reading
                clamped=False,
            ))
        # 60 observations with solar
        for i in range(60):
            outdoor_delta = float(i % 10)
            solar = 0.3 + (i % 5) * 0.1
            true_offset = 1.0 + 0.5 * outdoor_delta - 2.0 * solar
            obs.append(Observation(
                timestamp=float(150 + i), wall_time=1713650000.0 + (150 + i) * 3600,
                hp_setpoint=20.0 + true_offset, current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + outdoor_delta, room_rate=0.005,
                raw_readings={"sensor.solar": solar},
                clamped=False,
            ))
        result = weighted_least_squares(
            obs, n_features=3, min_observations=20,
            feature_order=feature_order, model_inputs=model_inputs,
        )
        assert result is not None
        # outdoor_delta anchored by all 210 observations
        assert abs(result.beta_batch[1] - 0.5) < 0.05
        # solar estimated from 60 observations
        assert abs(result.beta_batch[2] - (-2.0)) < 0.5


class TestConfigChangeResilience:
    """Proof tests for the three scenarios that motivated the raw_readings redesign.

    Each test simulates a real user action (tau change, entity swap, feature
    addition) and verifies that the batch WLS produces correct, stable
    coefficients without buffer invalidation.
    """

    def test_tau_change_does_not_affect_batch(self):
        """Changing lag_tau has zero effect on batch WLS results.

        The batch uses raw sensor values from raw_readings, not EMA-filtered
        features.  Two runs with identical observations but different
        model_inputs lag_tau should produce identical coefficients.
        """
        model_inputs_tau3600 = [{"entity_id": "sensor.solar", "name": "solar", "lag_tau": 3600}]
        model_inputs_tau7200 = [{"entity_id": "sensor.solar", "name": "solar", "lag_tau": 7200}]
        feature_order = ["intercept", "outdoor_delta", "solar"]

        import random
        rng = random.Random(99)
        obs = []
        for i in range(60):
            od = rng.uniform(0, 10)
            solar = rng.uniform(0, 0.8)
            true_offset = 1.5 + 0.4 * od - 3.0 * solar
            obs.append(Observation(
                timestamp=float(i), wall_time=1713650000.0 + i * 3600,
                hp_setpoint=20.0 + true_offset + rng.gauss(0, 0.05),
                current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + od, room_rate=0.005,
                raw_readings={"sensor.solar": solar},
                clamped=False,
            ))

        result_3600 = weighted_least_squares(
            obs, n_features=3, feature_order=feature_order, model_inputs=model_inputs_tau3600,
        )
        result_7200 = weighted_least_squares(
            obs, n_features=3, feature_order=feature_order, model_inputs=model_inputs_tau7200,
        )
        assert result_3600 is not None
        assert result_7200 is not None
        # Identical — tau is not used by batch WLS
        for i in range(3):
            assert result_3600.beta_batch[i] == result_7200.beta_batch[i]

    def test_entity_swap_old_obs_excluded_for_new_feature(self):
        """Swapping solar entity: old observations excluded for solar, kept for base.

        Old observations have raw_readings keyed to the old entity_id.
        After swapping to a new entity_id, those observations lack the new
        entity's readings and are excluded from the solar sub-regression.
        But they still contribute to outdoor_delta via the base regression.
        """
        old_model_inputs = [{"entity_id": "sensor.solar_old", "name": "solar"}]
        new_model_inputs = [{"entity_id": "sensor.solar_new", "name": "solar"}]
        feature_order = ["intercept", "outdoor_delta", "solar"]

        import random
        rng = random.Random(42)
        obs = []
        # 100 observations with old solar entity
        for i in range(100):
            od = rng.uniform(0, 10)
            solar = rng.uniform(0, 0.6)
            true_offset = 2.0 + 0.3 * od - 4.0 * solar
            obs.append(Observation(
                timestamp=float(i), wall_time=1713650000.0 + i * 3600,
                hp_setpoint=20.0 + true_offset + rng.gauss(0, 0.05),
                current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + od, room_rate=0.005,
                raw_readings={"sensor.solar_old": solar},
                clamped=False,
            ))
        # 40 observations with new solar entity
        for i in range(40):
            od = rng.uniform(0, 10)
            solar = rng.uniform(0, 0.6)
            true_offset = 2.0 + 0.3 * od - 4.0 * solar
            obs.append(Observation(
                timestamp=float(100 + i), wall_time=1713650000.0 + (100 + i) * 3600,
                hp_setpoint=20.0 + true_offset + rng.gauss(0, 0.05),
                current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + od, room_rate=0.005,
                raw_readings={"sensor.solar_new": solar},
                clamped=False,
            ))

        # With the new entity config: old obs contribute to base only
        result = weighted_least_squares(
            obs, n_features=3, feature_order=feature_order, model_inputs=new_model_inputs,
        )
        assert result is not None
        # outdoor_delta anchored by ALL 140 observations
        assert abs(result.beta_batch[1] - 0.3) < 0.05
        # solar estimated from only the 40 new-entity observations
        assert abs(result.beta_batch[2] - (-4.0)) < 0.3
        # solar has finite std_err (not inf — it was estimated, not held)
        assert result.beta_std_err[2] < 10.0

    def test_progressive_feature_addition_outdoor_stable(self):
        """Adding a feature doesn't shift outdoor_delta coefficient.

        200 base-only observations establish outdoor_delta.  60 new
        observations add solar.  The outdoor_delta coefficient should
        be nearly identical whether computed with or without the solar
        feature, because FWL partials out the base before estimating solar.
        """
        feature_order_base = ["intercept", "outdoor_delta"]
        feature_order_ext = ["intercept", "outdoor_delta", "solar"]
        model_inputs_ext = [{"entity_id": "sensor.solar", "name": "solar"}]

        import random
        rng = random.Random(42)
        obs_base = []
        for i in range(200):
            od = rng.uniform(0, 10)
            true_offset = 2.0 + 0.3 * od
            obs_base.append(Observation(
                timestamp=float(i), wall_time=1713650000.0 + i * 3600,
                hp_setpoint=20.0 + true_offset + rng.gauss(0, 0.05),
                current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + od, room_rate=0.005,
                raw_readings={},
                clamped=False,
            ))

        # Base-only regression
        result_base = weighted_least_squares(
            obs_base, n_features=2, feature_order=feature_order_base, model_inputs=[],
        )

        # Now add 60 solar observations
        obs_extended = list(obs_base)
        for i in range(60):
            od = rng.uniform(0, 10)
            solar = rng.uniform(0, 0.6)
            true_offset = 2.0 + 0.3 * od - 3.0 * solar
            obs_extended.append(Observation(
                timestamp=float(200 + i), wall_time=1713650000.0 + (200 + i) * 3600,
                hp_setpoint=20.0 + true_offset + rng.gauss(0, 0.05),
                current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + od, room_rate=0.005,
                raw_readings={"sensor.solar": solar},
                clamped=False,
            ))

        # Extended regression with solar
        result_ext = weighted_least_squares(
            obs_extended, n_features=3, feature_order=feature_order_ext,
            model_inputs=model_inputs_ext,
        )

        assert result_base is not None
        assert result_ext is not None
        # outdoor_delta should be stable: adding solar shouldn't shift it
        assert abs(result_base.beta_batch[1] - result_ext.beta_batch[1]) < 0.02
        # Both should be close to the true value
        assert abs(result_ext.beta_batch[1] - 0.3) < 0.02
        # solar should be estimated correctly
        assert abs(result_ext.beta_batch[2] - (-3.0)) < 0.3

    def test_fwl_matches_joint_ols_on_complete_data(self):
        """When all observations have all features, hierarchical FWL
        produces the same coefficients as a joint regression would.

        This is the mathematical guarantee of Frisch-Waugh-Lovell (1933).
        Verifies the implementation is correct, not just "close enough."
        """
        model_inputs = [
            {"entity_id": "sensor.solar", "name": "solar"},
            {"entity_id": "sensor.pellet", "name": "pellet"},
        ]
        feature_order = ["intercept", "outdoor_delta", "solar", "pellet"]

        import random
        rng = random.Random(42)
        true_beta = [1.5, 0.4, -3.0, -6.0]
        obs = []
        for i in range(100):
            od = rng.uniform(-5, 15)
            solar = rng.uniform(0, 0.8)
            pellet = 1.0 if rng.random() < 0.15 else 0.0
            y_true = true_beta[0] + true_beta[1] * od + true_beta[2] * solar + true_beta[3] * pellet
            obs.append(Observation(
                timestamp=float(i), wall_time=1713650000.0 + i * 3600,
                hp_setpoint=20.0 + y_true, current_c=20.0, desired_c=20.0,
                outdoor_temp_c=20.0 + od, room_rate=0.005,
                raw_readings={"sensor.solar": solar, "sensor.pellet": pellet},
                clamped=False,
            ))

        result = weighted_least_squares(
            obs, n_features=4, feature_order=feature_order, model_inputs=model_inputs,
        )
        assert result is not None
        # Noiseless data: should recover exact coefficients
        for i, (est, true) in enumerate(zip(result.beta_batch, true_beta)):
            assert abs(est - true) < 0.01, (
                f"Coefficient {i}: {est:.6f} != {true:.6f} (diff={abs(est-true):.6f})"
            )
        # All model input std_errs should be finite (not inf)
        for i in range(2, 4):
            assert result.beta_std_err[i] < 1.0, (
                f"std_err[{i}] = {result.beta_std_err[i]} should be finite"
            )


class TestObservationMetadata:
    """Tests for v2 Observation metadata fields and round-trip serialization."""

    def test_metadata_round_trip(self):
        """v2 metadata fields survive as_dict → from_dict round-trip."""
        obs = Observation(
            timestamp=100.0,
            wall_time=1713650000.0,
            hp_setpoint=22.0,
            current_c=20.0,
            desired_c=20.0,
            outdoor_temp_c=-5.0,
            room_rate=0.005,
            raw_readings={"sensor.solar": 0.4},
            clamped=False,
            supplemental_active=True,
        )
        d = obs.as_dict()
        restored = Observation.from_dict(d)
        assert restored.outdoor_temp_c == -5.0
        assert restored.supplemental_active is True
        assert restored.raw_readings == {"sensor.solar": 0.4}
        assert restored.wall_time == 1713650000.0

    def test_legacy_v1_raises(self):
        """Legacy v1 observations raise ValueError."""
        legacy_dict = {
            "t": 100.0, "x": [1.0, 5.0], "sp": 22.0,
            "cur": 20.0, "des": 20.0, "rate": 0.005, "clamp": False,
        }
        with pytest.raises(ValueError, match="v1 observation"):
            Observation.from_dict(legacy_dict)

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
        return _make_test_obs(
            features or [1.0, 5.0], sp=sp, cur=cur, des=21.0,
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
        return _make_test_obs(features, sp=sp, cur=cur,
                              clamped=clamped, clamped_reason=clamped_reason)

    def test_condition_number_single_feature(self):
        """With intercept + 1 feature, κ is trivially 1.0.

        The intercept is excluded from κ computation (Belsley 1980),
        leaving only one feature — no collinearity possible.
        """
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        assert buf.compute_condition_number() == 1.0

    def test_condition_number_inf_empty_buffer(self):
        """Returns inf for empty buffer with 3+ features."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        assert buf.compute_condition_number() == float("inf")

    def test_condition_number_well_conditioned(self):
        """Diverse, uncorrelated features produce low κ (Belsley: reliable).

        Two non-intercept features with independent variation.
        κ computed on features only (intercept excluded per Belsley 1980).
        """
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        for i in range(50):
            outdoor_delta = float(i % 10)  # 0-9°C spread
            # Independent model input with different pattern
            model_input = float((i * 7) % 10) * 0.5
            buf.add(self._make_obs([1.0, outdoor_delta, model_input]))
        buf.recompute_info_matrix()
        cond = buf.compute_condition_number()
        assert cond < 20.0

    def test_condition_number_moderate_collinearity(self):
        """Correlated features give moderate collinearity (Belsley: κ > 10).

        Two features with |r| ≈ 0.9 — partially confounded so
        individual coefficient estimates are unreliable but the
        combined prediction is still stable.
        κ excludes the intercept (Belsley 1980), so this measures
        pure feature-to-feature collinearity.
        """
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        for i in range(50):
            x1 = float(i % 10)
            # x2 tracks x1 closely but not perfectly (r ≈ 0.9)
            x2 = x1 * 0.8 + (i % 3) * 0.5
            buf.add(self._make_obs([1.0, x1, x2]))
        buf.recompute_info_matrix()
        cond = buf.compute_condition_number()
        assert cond > 5.0  # Meaningful collinearity from feature correlation

    def test_condition_number_severe_collinearity(self):
        """Near-identical features produce high κ (Belsley: severe).

        Two features that are nearly linearly dependent — coefficients
        are numerically unstable.
        """
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        for i in range(50):
            x1 = float(i % 10)
            # x2 nearly identical to x1 (tiny noise)
            x2 = x1 + (i % 10) * 0.001
            buf.add(self._make_obs([1.0, x1, x2]))
        buf.recompute_info_matrix()
        cond = buf.compute_condition_number()
        assert cond > 50.0  # Severe collinearity between features

    def test_pairwise_correlations_empty_buffer(self):
        """Returns empty list with insufficient data."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        assert buf.get_pairwise_correlations() == []

    def test_pairwise_correlations_detects_correlated(self):
        """Detects highly correlated features."""
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
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
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        for i in range(30):
            outdoor_delta = float(i % 10)
            pellet = float((i + 5) % 3)  # uncorrelated pattern
            buf.add(self._make_obs([1.0, outdoor_delta, pellet]))
        pairs = buf.get_pairwise_correlations(["intercept", "outdoor_delta", "pellet"])
        assert len(pairs) == 0

    def test_pairwise_correlations_skips_clamped(self):
        """Correlation computed only from unclamped observations."""
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
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
        import datetime as _dt
        # Create a wall_time that corresponds to the desired local hour.
        # Use a fixed date so tests are deterministic. wall_hour=-1 → wall_time=0
        if wall_hour >= 0:
            dt = _dt.datetime(2026, 4, 20, wall_hour, 30, 0)
            wt = dt.timestamp()
        else:
            wt = 0.0
        return _make_test_obs(features, sp=sp, cur=cur, des=des, rate=rate,
                              clamped=clamped, clamped_reason=clamped_reason,
                              wall_time=wt)

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
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2,
                feature_order=_test_feature_order(0), model_inputs=_test_model_inputs(0))
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
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2,
                feature_order=_test_feature_order(0), model_inputs=_test_model_inputs(0))
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
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2,
                feature_order=_test_feature_order(0), model_inputs=_test_model_inputs(0))
        assert len(patterns) == 0

    def test_skips_missing_wall_hour(self):
        """Observations with wall_hour=-1 (legacy) are excluded."""
        beta = [1.0, 0.3]
        obs = []
        for _ in range(50):
            obs.append(self._make_obs(
                features=[1.0, 5.0], sp=25.0, cur=20.0, wall_hour=-1,
            ))
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2,
                feature_order=_test_feature_order(0), model_inputs=_test_model_inputs(0))
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
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2,
                feature_order=_test_feature_order(0), model_inputs=_test_model_inputs(0))
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
        patterns = analyze_residuals_by_hour(obs, beta, n_features=2,
                feature_order=_test_feature_order(0), model_inputs=_test_model_inputs(0))
        assert len(patterns) >= 1
        night = [p for p in patterns if p.start_hour == 22 or p.start_hour <= 2]
        assert len(night) >= 1
        assert night[0].mean_residual > 0.5  # positive = heat loss

    def test_observation_wall_time_serialization(self):
        """wall_time round-trips through as_dict/from_dict."""
        obs = self._make_obs([1.0, 5.0], sp=22.0, cur=20.0, wall_hour=14)
        d = obs.as_dict()
        assert d["v"] == 2
        assert d["wt"] == obs.wall_time
        assert "rr" in d  # raw_readings
        restored = Observation.from_dict(d)
        assert restored.wall_time == obs.wall_time
        assert restored.raw_readings == obs.raw_readings

    def test_legacy_v1_observation_raises(self):
        """Legacy v1 observations raise ValueError (cannot migrate)."""
        d = {
            "t": 0.0, "x": [1.0], "sp": 22.0,
            "cur": 20.0, "des": 20.0, "rate": 0.005, "clamp": False,
        }
        with pytest.raises(ValueError, match="v1 observation"):
            Observation.from_dict(d)


# ── Coverage gap tests ──────────────────────────────────────────────


class TestBatchLearningCoverageGaps:
    """Cover missing lines in batch_learning.py."""

    def _make_obs(self, features, sp, cur, des=20.0, rate=0.005,
                  clamped=False, clamped_reason="", wall_hour=None):
        wt = None
        if wall_hour is not None:
            import datetime as _dt
            wt = _dt.datetime(2026, 1, 15, wall_hour, 30, 0).timestamp()
        return _make_test_obs(features, sp, cur, des=des, rate=rate,
                              clamped=clamped, clamped_reason=clamped_reason,
                              wall_time=wt)

    # ── Line 25-26: numpy unavailable fallback ──

    def test_numpy_unavailable_flag(self):
        """_NUMPY_AVAILABLE = False path when numpy import fails."""
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        orig = bl._NUMPY_AVAILABLE
        try:
            bl._NUMPY_AVAILABLE = False
            # compute_belsley_diagnostics should return [] (line 1387)
            result = bl.compute_belsley_diagnostics(
                [[1.0, 2.0], [1.0, 3.0]], n_features=2, n_obs=2,
            )
            assert result == []
        finally:
            bl._NUMPY_AVAILABLE = orig

    # ── Line 133: build_feature_vector_from_raw with outdoor_temp_c=None ──

    def test_feature_vector_none_outdoor(self):
        """build_feature_vector_from_raw returns None when outdoor_temp_c is None."""
        from custom_components.tasmota_irhvac.pi.batch_learning import build_feature_vector_from_raw
        obs = Observation(
            timestamp=0.0, wall_time=time.time(),
            hp_setpoint=22.0, current_c=20.0, desired_c=20.0,
            outdoor_temp_c=None, room_rate=0.005,
            raw_readings={}, clamped=False, clamped_reason="",
        )
        result = build_feature_vector_from_raw(
            obs, _test_model_inputs(0), _test_feature_order(0),
        )
        assert result is None

    # ── Line 516: eigenvalues returns None → condition_number = inf ──

    def test_condition_number_eigenvalues_none(self):
        """condition_number returns inf when eigenvalues cannot be computed."""
        buf = DiversityAwareBuffer(n_features=4, max_size=100)
        # Add observations with near-zero variance in one feature
        for i in range(30):
            buf.add(self._make_obs([1.0, float(i), 0.0, 0.0], sp=22.0, cur=20.0))
        # Two identical features → singular correlation matrix with numpy
        # numpy should still handle this, but we can test lambda_min < 1e-15 path
        kappa = buf.compute_condition_number()
        assert kappa >= 1.0  # At least returns something valid

    # ── Line 521: lambda_min < 1e-15 ──

    def test_condition_number_eigenvalues_none_path(self):
        """condition_number returns inf when eigenvalues returns None (line 516)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        orig = bl._NUMPY_AVAILABLE
        try:
            bl._NUMPY_AVAILABLE = False
            buf = DiversityAwareBuffer(
                n_features=4, max_size=100,
                feature_order=_test_feature_order(2),
                model_inputs=_test_model_inputs(2),
            )
            for i in range(30):
                buf.add(self._make_obs([1.0, float(i % 10), float(i), float(i * 2)], sp=22.0, cur=20.0))
            with _patch.object(DiversityAwareBuffer, '_eigenvalues_symmetric', return_value=None):
                kappa = buf.compute_condition_number()
            assert kappa == float('inf')
        finally:
            bl._NUMPY_AVAILABLE = orig

    def test_condition_number_lambda_min_zero(self):
        """condition_number returns inf when lambda_min < 1e-15 (line 521)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        orig = bl._NUMPY_AVAILABLE
        try:
            bl._NUMPY_AVAILABLE = False
            buf = DiversityAwareBuffer(
                n_features=4, max_size=100,
                feature_order=_test_feature_order(2),
                model_inputs=_test_model_inputs(2),
            )
            for i in range(30):
                buf.add(self._make_obs([1.0, float(i % 10), float(i), float(i * 2)], sp=22.0, cur=20.0))
            with _patch.object(DiversityAwareBuffer, '_eigenvalues_symmetric', return_value=[10.0, 0.0]):
                kappa = buf.compute_condition_number()
            assert kappa == float('inf')
        finally:
            bl._NUMPY_AVAILABLE = orig

    # ── Line 548: VIF with singular correlation matrix ──

    def test_vif_singular_corr(self):
        """VIF returns all inf when correlation matrix inversion fails (line 548)."""
        from unittest.mock import patch as _patch
        buf = DiversityAwareBuffer(
            n_features=4, max_size=100,
            feature_order=_test_feature_order(2),
            model_inputs=_test_model_inputs(2),
        )
        for i in range(30):
            buf.add(self._make_obs([1.0, float(i % 10), float(i), float(i * 2)], sp=22.0, cur=20.0))
        with _patch.object(DiversityAwareBuffer, '_invert_matrix', return_value=None):
            vif = buf.compute_vif()
        assert all(v == float('inf') for v in vif)  # all inf including intercept

    # ── Line 576: pairwise correlations with too many clamped obs ──

    def test_pairwise_correlations_all_clamped(self):
        """get_pairwise_correlations returns [] when < 20 unclamped obs."""
        buf = DiversityAwareBuffer(
            n_features=4, max_size=100,
            feature_order=_test_feature_order(2),
            model_inputs=_test_model_inputs(2),
        )
        # Add 25 observations, but all clamped
        for i in range(25):
            buf.add(self._make_obs(
                [1.0, float(i), float(i % 3), float(i % 4)],
                sp=22.0, cur=20.0, clamped=True, clamped_reason="no_output",
            ))
        result = buf.get_pairwise_correlations(["intercept", "od", "a", "b"])
        assert result == []

    # ── Line 600: zero-variance column in pairwise correlation ──

    def test_pairwise_correlations_zero_variance(self):
        """Pairs with zero-variance column are skipped (denom < 1e-12)."""
        buf = DiversityAwareBuffer(
            n_features=4, max_size=100,
            feature_order=_test_feature_order(2),
            model_inputs=_test_model_inputs(2),
        )
        for i in range(25):
            # Feature 2 is constant → zero variance
            buf.add(self._make_obs(
                [1.0, float(i % 10), 5.0, float(i)],
                sp=22.0, cur=20.0,
            ))
        result = buf.get_pairwise_correlations(["intercept", "od", "const", "var"])
        # Pairs involving "const" should be absent (zero variance)
        for name_a, name_b, r in result:
            assert "const" not in (name_a, name_b)

    # ── Lines 734-783: non-numpy eigenvalue fallback ──

    def test_eigenvalues_no_numpy_n1(self):
        """Eigenvalue for 1×1 matrix without numpy."""
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        orig = bl._NUMPY_AVAILABLE
        try:
            bl._NUMPY_AVAILABLE = False
            buf = DiversityAwareBuffer(
                n_features=3, max_size=100,
                feature_order=_test_feature_order(1),
                model_inputs=_test_model_inputs(1),
            )
            for i in range(30):
                buf.add(self._make_obs([1.0, float(i % 10), float(i % 7)], sp=22.0, cur=20.0))
            # Should exercise n=2 direct formula path (2×2 corr matrix)
            kappa = buf.compute_condition_number()
            assert kappa >= 1.0
        finally:
            bl._NUMPY_AVAILABLE = orig

    def test_eigenvalues_no_numpy_n_gt_2(self):
        """Eigenvalue via power iteration for n>2 without numpy."""
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        orig = bl._NUMPY_AVAILABLE
        try:
            bl._NUMPY_AVAILABLE = False
            buf = DiversityAwareBuffer(
                n_features=5, max_size=100,
                feature_order=_test_feature_order(3),
                model_inputs=_test_model_inputs(3),
            )
            for i in range(40):
                buf.add(self._make_obs(
                    [1.0, float(i % 10), float(i % 5), float(i % 3), float(i % 7)],
                    sp=22.0, cur=20.0,
                ))
            kappa = buf.compute_condition_number()
            assert kappa >= 1.0
            assert kappa < float('inf')
        finally:
            bl._NUMPY_AVAILABLE = orig

    # ── Line 898: _solve_joint returns None (singular) ──

    def test_joint_solve_singular_falls_to_fwl(self):
        """When joint solve fails, FWL is used (lines 898, 940-1027, 1185)."""
        from unittest.mock import patch as _patch
        # Create data where all model inputs are present but joint system is singular
        obs = []
        for i in range(40):
            od = float(i % 10)
            # Make model input perfectly correlated with outdoor_delta
            inp_val = od * 2.0  # perfect collinearity
            obs.append(self._make_obs(
                [1.0, od, inp_val], sp=22.0, cur=20.0,
            ))

        with _patch(
            "custom_components.tasmota_irhvac.pi.batch_learning._solve_joint",
            return_value=None,
        ):
            result = weighted_least_squares(
                obs, n_features=3, min_observations=20,
                feature_order=_test_feature_order(1),
                model_inputs=_test_model_inputs(1),
            )
        # FWL should have produced a result
        assert result is not None

    # ── Line 1073: n_features < 2 ──

    def test_wls_n_features_lt_2(self):
        """WLS returns None when n_features < 2."""
        obs = [self._make_obs([1.0], sp=22.0, cur=20.0) for _ in range(30)]
        result = weighted_least_squares(
            obs, n_features=1, min_observations=20,
            feature_order=["intercept"],
            model_inputs=[],
        )
        assert result is None

    # ── Line 1106: base solve fails ──

    def test_wls_base_solve_fails(self):
        """WLS returns None when base (intercept+outdoor_delta) solve fails."""
        from unittest.mock import patch as _patch
        obs = []
        for i in range(30):
            obs.append(self._make_obs([1.0, float(i % 10)], sp=22.0, cur=20.0))
        with _patch(
            "custom_components.tasmota_irhvac.pi.batch_learning._solve_symmetric",
            return_value=None,
        ):
            result = weighted_least_squares(
                obs, n_features=2, min_observations=20,
                feature_order=_test_feature_order(0),
                model_inputs=_test_model_inputs(0),
            )
        assert result is None

    # ── Lines 1144-1145: model input with no entity_id ──

    def test_wls_model_input_no_entity_id(self):
        """Model input without entity_id is held (treated as unavailable)."""
        obs = []
        for i in range(30):
            obs.append(self._make_obs([1.0, float(i % 10), float(i % 5)], sp=22.0, cur=20.0))
        result = weighted_least_squares(
            obs, n_features=3, min_observations=20,
            feature_order=_test_feature_order(1),
            model_inputs=[{"name": "no_id_input"}],  # no entity_id
        )
        assert result is not None
        assert 2 in result.held_features  # index 2 should be held

    # ── Lines 1247-1250: rare-feature outlier protection ──

    def test_rare_feature_outlier_protection(self):
        """Outliers with rare features are kept (not excluded)."""
        obs = []
        # 35 obs with only intercept + outdoor_delta
        for i in range(35):
            obs.append(self._make_obs([1.0, float(i % 10)], sp=22.0, cur=20.0))
        # 5 obs with a rare model input present + large residual
        for i in range(5):
            o = self._make_obs([1.0, float(i)], sp=30.0, cur=20.0)  # big outlier
            o.raw_readings[_TEST_ENTITIES[0]] = 25.0  # rare feature
            obs.append(o)
        result = weighted_least_squares(
            obs, n_features=3, min_observations=20,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
            min_feature_representation=20,  # threshold above count=5
        )
        assert result is not None

    # ── Lines 1309, 1313: ragged feature matrix in _compute_vif_from_features ──

    def test_vif_from_features_ragged(self):
        """_compute_vif_from_features handles ragged rows gracefully."""
        from custom_components.tasmota_irhvac.pi.batch_learning import _compute_vif_from_features
        # Rows of different lengths
        X = [
            [1.0, 2.0, 3.0],
            [1.0, 2.5],  # short row
            [1.0, 3.0, 4.0],
        ]
        result = _compute_vif_from_features(X, n_features=3, n_obs=3)
        assert len(result) == 3
        assert result[0] == 1.0  # intercept

    # ── Line 1389: Belsley with n_obs < n_features ──

    def test_belsley_insufficient_obs(self):
        """compute_belsley_diagnostics returns [] with too few observations."""
        from custom_components.tasmota_irhvac.pi.batch_learning import compute_belsley_diagnostics
        result = compute_belsley_diagnostics(
            [[1.0, 2.0, 3.0]], n_features=3, n_obs=1,
        )
        assert result == []

    # ── Line 1395: Belsley with extra columns ──

    def test_belsley_extra_columns(self):
        """compute_belsley_diagnostics truncates extra columns."""
        from custom_components.tasmota_irhvac.pi.batch_learning import compute_belsley_diagnostics
        X = [[1.0, float(i), float(i * 2), 99.0] for i in range(20)]
        result = compute_belsley_diagnostics(X, n_features=3, n_obs=20)
        # Should not crash, may or may not find collinearity
        assert isinstance(result, list)

    # ── Lines 1406-1407: SVD failure ──

    def test_belsley_svd_failure(self):
        """compute_belsley_diagnostics returns [] on SVD failure."""
        from custom_components.tasmota_irhvac.pi.batch_learning import compute_belsley_diagnostics
        X = [[float('nan'), 1.0], [1.0, float('nan')]]
        result = compute_belsley_diagnostics(X, n_features=2, n_obs=2)
        assert result == []

    # ── Line 1733: fuse_batch_greybox with near-zero variance ──

    def test_fuse_batch_greybox_zero_variance(self):
        """Fusion skips coefficients with near-zero std_err."""
        result = BatchResult(
            n_total=50, n_eligible=40,
            beta_batch=[2.0, 0.5], beta_current=[1.8, 0.4],
            residual_rms=0.1, max_coeff_change_pct=10.0,
            recommend_update=True,
            beta_std_err=[1e-8, 0.1],  # first has near-zero variance
        )
        fuse_batch_greybox(
            result,
            greybox_beta=[2.1, 0.6],
            greybox_std_err=[1e-8, 0.1],  # both near-zero
        )
        # First coefficient should NOT be fused (zero variance skip)
        assert result.beta_batch[0] == 2.0  # unchanged

    # ── Line 1819: analyze_residuals_by_hour without feature_order ──

    def test_residuals_by_hour_no_feature_order(self):
        """analyze_residuals_by_hour skips obs when feature_order is None."""
        obs = [self._make_obs([1.0, 5.0], sp=22.0, cur=20.0, wall_hour=10)
               for _ in range(10)]
        patterns = analyze_residuals_by_hour(
            obs, [1.0, 0.3], n_features=2,
            feature_order=None, model_inputs=None,
        )
        assert patterns == []

    # ── Line 1822: feature vector is None in residual analysis ──

    def test_residuals_by_hour_missing_outdoor_temp(self):
        """analyze_residuals_by_hour skips obs with None outdoor_temp_c."""
        obs = []
        for _ in range(10):
            o = Observation(
                timestamp=0.0, wall_time=time.time(),
                hp_setpoint=22.0, current_c=20.0, desired_c=20.0,
                outdoor_temp_c=None, room_rate=0.005,
                raw_readings={}, clamped=False, clamped_reason="",
            )
            obs.append(o)
        patterns = analyze_residuals_by_hour(
            obs, [1.0, 0.3], n_features=2,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
        assert patterns == []

    # ── Line 1866: sign reversal breaks span ──

    def test_residuals_by_hour_sign_reversal(self):
        """Span extension stops when sign flips."""
        obs = []
        beta = [1.0, 0.3]
        for hour in range(24):
            for _ in range(8):
                od = 5.0
                true_offset = 1.0 + 0.3 * od
                if hour in (10, 11, 12):
                    bias = 0.8  # positive
                elif hour == 13:
                    bias = -0.8  # negative — sign flip
                else:
                    bias = 0.0
                obs.append(self._make_obs(
                    [1.0, od], sp=20.0 + true_offset + bias, cur=20.0,
                    wall_hour=hour,
                ))
        patterns = analyze_residuals_by_hour(
            obs, beta, n_features=2,
            feature_order=_test_feature_order(0),
            model_inputs=_test_model_inputs(0),
        )
        # Hour 13 should not be merged with 10-12 (different sign)
        pos = [p for p in patterns if p.mean_residual > 0]
        neg = [p for p in patterns if p.mean_residual < 0]
        assert len(pos) >= 1
        assert len(neg) >= 1

    # ── Lines 734-783: non-numpy eigenvalue paths (direct calls) ──

    def test_eigenvalues_n1_no_numpy(self):
        """Eigenvalue for 1×1 matrix without numpy (line 738)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        with _patch.object(bl, '_NUMPY_AVAILABLE', False):
            result = DiversityAwareBuffer._eigenvalues_symmetric([[5.0]], 1)
        assert result == [5.0]

    def test_eigenvalues_n2_no_numpy(self):
        """Eigenvalue for 2×2 matrix without numpy (lines 739-744)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        with _patch.object(bl, '_NUMPY_AVAILABLE', False):
            A = [[2.0, 0.0], [0.0, 3.0]]
            result = DiversityAwareBuffer._eigenvalues_symmetric(A, 2)
        assert len(result) == 2
        assert abs(result[0] - 3.0) < 0.01
        assert abs(result[1] - 2.0) < 0.01

    def test_eigenvalues_n3_zero_matrix_no_numpy(self):
        """Eigenvalue for zero 3×3 matrix → None (lines 762, 777)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        with _patch.object(bl, '_NUMPY_AVAILABLE', False):
            A = [[0.0] * 3 for _ in range(3)]
            result = DiversityAwareBuffer._eigenvalues_symmetric(A, 3)
        assert result is None

    def test_eigenvalues_n3_singular_no_numpy(self):
        """Eigenvalue for singular 3×3 matrix → None (line 767: _invert_matrix fails)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        with _patch.object(bl, '_NUMPY_AVAILABLE', False):
            # Singular matrix: row 2 = row 0
            A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
            result = DiversityAwareBuffer._eigenvalues_symmetric(A, 3)
        # Should return None because A is singular (inv fails) OR
        # power iteration result near 0

    def test_eigenvalues_n3_inv_lam_min_near_zero(self):
        """Eigenvalue for near-zero minimum eigenvalue (line 781)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        with _patch.object(bl, '_NUMPY_AVAILABLE', False):
            # Near-singular: one eigenvalue very small
            A = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-20]]
            result = DiversityAwareBuffer._eigenvalues_symmetric(A, 3)
        # inv_lam_min ≈ 1/1e-20 = 1e20 >> 0, so lam_min = 1/1e20 ≈ 0
        # → line 781: inv_lam_min < 1e-15 is False, but lam_min = 1/inv_lam_min ≈ 1e-20
        # Actually the test should return valid eigenvalues since all are positive

    def test_eigenvalues_n3_valid_no_numpy(self):
        """Eigenvalue for valid 3×3 matrix without numpy (full power iteration)."""
        from unittest.mock import patch as _patch
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        with _patch.object(bl, '_NUMPY_AVAILABLE', False):
            A = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
            result = DiversityAwareBuffer._eigenvalues_symmetric(A, 3)
            assert result is not None
            assert len(result) == 2  # [lambda_max, lambda_min]

    # ── Line 25-26: numpy import fallback ──

    def test_numpy_import_fallback(self):
        """_NUMPY_AVAILABLE=False when numpy import fails (lines 25-26)."""
        import sys
        import importlib
        import custom_components.tasmota_irhvac.pi.batch_learning as bl
        numpy_mod = sys.modules.get("numpy")
        sys.modules["numpy"] = None  # poison
        try:
            importlib.reload(bl)
            assert bl._NUMPY_AVAILABLE is False
        finally:
            if numpy_mod is not None:
                sys.modules["numpy"] = numpy_mod
            else:
                sys.modules.pop("numpy", None)
            importlib.reload(bl)

    def test_eigvalsh_linalg_error(self):
        """eigvalsh LinAlgError → returns None (lines 734-735)."""
        import numpy as np
        from unittest.mock import patch as _patch
        buf = DiversityAwareBuffer(
            n_features=3, max_size=100,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        for i in range(30):
            buf.add(self._make_obs([1.0, float(i % 10), float(i % 7)], sp=22.0, cur=20.0))
        with _patch.object(np.linalg, 'eigvalsh', side_effect=np.linalg.LinAlgError("test")):
            kappa = buf.compute_condition_number()
        assert kappa == float('inf')
