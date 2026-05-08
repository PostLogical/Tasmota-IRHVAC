"""Tests for the diversity-aware observation buffer and residual filtering.

Covers 26 scenarios from the design discussion:
- Buffer diversity and eviction (1-5)
- Residual filtering at WLS time (6-10)
- Physical changes and drift behavior (11-14)
- Model input changes (15-17)
- Buffer mechanics (18-23)
- Regime coverage and convergence (24-26)
"""

import math
import random

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    BufferAddResult,
    DiversityAwareBuffer as _DiversityAwareBuffer,
    Observation,
    weighted_least_squares,
)
from custom_components.tasmota_irhvac.pi.buffer_policies import LeveragePolicy


def DiversityAwareBuffer(*args, **kwargs):
    """Test-local wrapper that pins ``policy=LeveragePolicy()`` by default.

    The production default switched from leverage to ``SlevPolicy(alpha=0.0)``
    after the 2026-05-07 finding that leverage curation biases β_solar; the
    26 diversity scenarios in this file were authored to characterize
    leverage-policy behavior specifically (rare-regime retention, high-score
    eviction, coverage maintenance).  Pinning the policy here keeps the test
    intent intact without sprinkling ``policy=LeveragePolicy()`` across 22
    construction sites.
    """
    kwargs.setdefault("policy", LeveragePolicy())
    return _DiversityAwareBuffer(*args, **kwargs)


DiversityAwareBuffer.from_list = _DiversityAwareBuffer.from_list  # type: ignore[attr-defined]


import time as _time

# Synthetic entity IDs for test model inputs
_SOLAR_ENTITY = "sensor.test_solar"
_PELLET_ENTITY = "sensor.test_pellet"
_DOOR_ENTITY = "sensor.test_door"

TEST_MODEL_INPUTS = [
    {"entity_id": _SOLAR_ENTITY, "name": "solar"},
    {"entity_id": _PELLET_ENTITY, "name": "pellet"},
    {"entity_id": _DOOR_ENTITY, "name": "door"},
]

TEST_FEATURE_ORDER = ["intercept", "outdoor_delta", "solar", "pellet", "door"]


def _make_obs(
    t: float = 0.0,
    outdoor_delta: float = 5.0,
    solar: float = 0.0,
    pellet: float = 0.0,
    door: float = 0.0,
    sp: float = 22.0,
    cur: float = 20.0,
    des: float = 20.0,
    rate: float = 0.001,
    clamped: bool = False,
    n_features: int = 5,
) -> Observation:
    """Build a v2 observation with raw sensor readings.

    Features: [intercept=1, outdoor_delta, solar, pellet, door]
    """
    raw_readings: dict[str, float] = {}
    if n_features > 2:
        raw_readings[_SOLAR_ENTITY] = solar
    if n_features > 3:
        raw_readings[_PELLET_ENTITY] = pellet
    if n_features > 4:
        raw_readings[_DOOR_ENTITY] = door
    return Observation(
        timestamp=t, wall_time=1713650000.0 + t,
        hp_setpoint=sp, current_c=cur, desired_c=des,
        outdoor_temp_c=cur + outdoor_delta,
        room_rate=rate, raw_readings=raw_readings, clamped=clamped,
    )


# ── Buffer diversity and eviction (1-5) ──────────────────────────────


class TestBufferDiversity:

    def test_1_steady_state_does_not_evict_rare(self):
        """Steady-state flooding doesn't evict rare observations."""
        buf = DiversityAwareBuffer(n_features=5, max_size=100, feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS)
        # Fill with mild-night observations
        for i in range(100):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0, solar=0.0))

        # Insert 5 cold-night observations
        cold_obs = []
        for i in range(5):
            obs = _make_obs(t=100.0 + i, outdoor_delta=20.0, solar=0.0)
            buf.add(obs)
            cold_obs.append(obs)

        # Flood with 200 more mild-night observations
        for i in range(200):
            buf.add(_make_obs(t=200.0 + i, outdoor_delta=5.0, solar=0.0))

        # Cold-night observations should still be in the buffer
        all_obs = buf.get_all()
        cold_deltas = [o for o in all_obs if (o.outdoor_temp_c - o.current_c) > 15.0]
        assert len(cold_deltas) >= 3, (
            f"Expected cold-night observations to survive, found {len(cold_deltas)}"
        )

    def test_2_all_dimensions_retain_representation(self):
        """All feature dimensions retain representation under single-dim flooding."""
        buf = DiversityAwareBuffer(n_features=5, max_size=200, feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS)
        # Seed buffer with diverse observations
        for i in range(50):
            buf.add(_make_obs(t=float(i), outdoor_delta=float(i % 10),
                              solar=0.5 if i % 7 == 0 else 0.0,
                              pellet=1.0 if i % 15 == 0 else 0.0,
                              door=1.0 if i % 12 == 0 else 0.0))
        # Flood with outdoor_delta-only variation
        for i in range(500):
            buf.add(_make_obs(t=100.0 + i, outdoor_delta=float(i % 10)))

        all_obs = buf.get_all()
        has_solar = any(o.raw_readings.get(_SOLAR_ENTITY, 0.0) > 0.3 for o in all_obs)
        has_pellet = any(o.raw_readings.get(_PELLET_ENTITY, 0.0) > 0.5 for o in all_obs)
        has_door = any(o.raw_readings.get(_DOOR_ENTITY, 0.0) > 0.5 for o in all_obs)
        assert has_solar, "Solar observations should survive"
        assert has_pellet, "Pellet observations should survive"
        assert has_door, "Door observations should survive"

    def test_3_diurnal_cycling_retained(self):
        """Buffer captures all phases of daily solar cycling."""
        buf = DiversityAwareBuffer(n_features=5, max_size=200, feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS)
        # 5 days of solar cycling: 0 at night, peak 0.6 midday
        for day in range(5):
            for hour in range(24):
                t = day * 24 + hour
                solar = max(0.0, 0.6 * math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
                buf.add(_make_obs(t=float(t), outdoor_delta=5.0, solar=solar))

        all_obs = buf.get_all()
        solar_vals = [o.raw_readings.get(_SOLAR_ENTITY, 0.0) for o in all_obs]
        has_zero = any(s < 0.01 for s in solar_vals)
        has_low = any(0.1 < s < 0.3 for s in solar_vals)
        has_peak = any(s > 0.4 for s in solar_vals)
        assert has_zero and has_low and has_peak, (
            f"Buffer should retain all solar phases: zero={has_zero}, low={has_low}, peak={has_peak}"
        )

    def test_4_intermittent_feature_survives_gap(self):
        """Pellet stove observations survive a week-long gap."""
        buf = DiversityAwareBuffer(n_features=5, max_size=200, feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS)
        # 2 days of pellet on
        for i in range(48):  # 48 ticks = 2 days at 1/hr
            buf.add(_make_obs(t=float(i), pellet=1.0, outdoor_delta=15.0))
        # 7 days of pellet off
        for i in range(168):
            buf.add(_make_obs(t=48.0 + i, pellet=0.0, outdoor_delta=5.0))

        all_obs = buf.get_all()
        pellet_on = [o for o in all_obs if o.raw_readings.get(_PELLET_ENTITY, 0.0) > 0.5]
        assert len(pellet_on) >= 5, (
            f"Pellet-on observations should survive 7-day gap, found {len(pellet_on)}"
        )

    def test_5_novel_combination_retained(self):
        """Novel feature combination displaces redundant observations."""
        buf = DiversityAwareBuffer(n_features=5, max_size=50, feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS)
        # Fill with common observations
        for i in range(50):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0))

        # Insert novel: pellet ON AND solar high (never seen before)
        novel = _make_obs(t=100.0, pellet=1.0, solar=0.5, outdoor_delta=5.0)
        buf.add(novel)

        all_obs = buf.get_all()
        has_novel = any(o.raw_readings.get(_PELLET_ENTITY, 0.0) > 0.5 and o.raw_readings.get(_SOLAR_ENTITY, 0.0) > 0.3 for o in all_obs)
        assert has_novel, "Novel combination should be retained"


# ── Residual filtering at WLS time (6-10) ────────────────────────────


class TestResidualFiltering:

    def _generate_clean_observations(self, n=100, n_features=3, noise_std=0.1):
        """Generate observations from known coefficients: y = 2 + 0.3*outdoor - 4*solar + noise."""
        true_beta = [2.0, 0.3, -4.0][:n_features]
        while len(true_beta) < n_features:
            true_beta.append(0.0)
        obs = []
        rng = random.Random(42)
        for i in range(n):
            outdoor = rng.uniform(-5, 15)
            solar = rng.uniform(0, 0.6)
            features = [1.0, outdoor, solar][:n_features]
            while len(features) < n_features:
                features.append(0.0)
            y_true = sum(b * x for b, x in zip(true_beta, features))
            y_noisy = y_true + rng.gauss(0, noise_std)
            # hp_setpoint - current_c = y, so sp = y + cur
            cur = 20.0
            sp = y_noisy + cur
            obs.append(_make_obs(
                t=float(i), outdoor_delta=outdoor, solar=solar,
                sp=sp, cur=cur, n_features=n_features,
            ))
        return obs, true_beta

    def test_6_single_glitch_excluded(self):
        """Single sensor glitch excluded from fit."""
        obs, true_beta = self._generate_clean_observations(n=100, n_features=3)
        # Add a glitch: y-value 10°C off
        glitch = _make_obs(
            t=999.0, outdoor_delta=5.0, solar=0.3,
            sp=40.0, n_features=3,
        )
        obs_with_glitch = obs + [glitch]

        result_clean = weighted_least_squares(obs, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1], detect_lag=False)
        result_glitch = weighted_least_squares(obs_with_glitch, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1], detect_lag=False)

        assert result_glitch is not None
        assert result_glitch.n_outliers_excluded >= 1, "Glitch should be excluded"
        # Coefficients should be similar to clean fit
        for i in range(3):
            assert abs(result_glitch.beta_batch[i] - result_clean.beta_batch[i]) < 0.3, (
                f"Coefficient {i}: glitch fit {result_glitch.beta_batch[i]:.3f} "
                f"vs clean {result_clean.beta_batch[i]:.3f}"
            )

    def test_7_glitch_observation_stays_in_buffer(self):
        """Glitch observation stays in buffer after WLS exclusion."""
        buf = DiversityAwareBuffer(n_features=3, max_size=200, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        obs, _ = self._generate_clean_observations(n=100, n_features=3)
        for o in obs:
            buf.add(o)

        glitch = _make_obs(
            t=999.0, outdoor_delta=5.0, solar=0.3,
            sp=40.0, n_features=3,
        )
        buf.add(glitch)

        # Run WLS — the glitch should be excluded from the fit
        all_obs = buf.get_all()
        result = weighted_least_squares(all_obs, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        assert result is not None

        # But the glitch should still be in the buffer
        all_obs_after = buf.get_all()
        has_glitch = any(o.hp_setpoint > 35.0 for o in all_obs_after)
        assert has_glitch, "Glitch observation should remain in buffer"

    def test_8_first_pellet_observations_not_excluded(self):
        """First pellet stove observations are NOT excluded as outliers."""
        # 100 observations with pellet=0
        obs, _ = self._generate_clean_observations(n=100, n_features=4)
        # Add 5 observations with pellet=1 that have large residuals
        # under the pellet-unaware model
        for i in range(5):
            # pellet=1, y is much lower (stove heats room, HP needs less offset)
            obs.append(_make_obs(
                t=200.0 + i, outdoor_delta=5.0, pellet=1.0,
                sp=14.0, n_features=4,
            ))

        result = weighted_least_squares(obs, n_features=4, feature_order=TEST_FEATURE_ORDER[:4], model_inputs=TEST_MODEL_INPUTS[:2],
                                        min_feature_representation=10)
        assert result is not None
        # The pellet observations should NOT have been excluded
        # because pellet has < 10 active observations.
        # The pellet coefficient should reflect their influence.
        pellet_coeff = result.beta_batch[3]
        assert pellet_coeff < -1.0, (
            f"Pellet coefficient should be strongly negative, got {pellet_coeff:.3f}"
        )

    def test_9_outlier_in_well_represented_feature_excluded(self):
        """After minimum representation, outliers in that feature ARE excluded."""
        rng = random.Random(42)
        obs = []
        for i in range(80):
            pellet = 1.0 if i < 20 else 0.0  # 20 pellet-on observations (> threshold)
            outdoor = rng.uniform(0, 10)
            y_true = 2.0 + 0.3 * outdoor - 6.0 * pellet
            sp = y_true + 20.0 + rng.gauss(0, 0.1)
            obs.append(_make_obs(
                t=float(i), outdoor_delta=outdoor, pellet=pellet,
                sp=sp, n_features=4,
            ))
        # Add 1 anomalous pellet-on observation
        obs.append(_make_obs(
            t=999.0, outdoor_delta=5.0, pellet=1.0,
            sp=30.0, n_features=4,  # should be ~14 with pellet on
        ))

        result = weighted_least_squares(obs, n_features=4, feature_order=TEST_FEATURE_ORDER[:4], model_inputs=TEST_MODEL_INPUTS[:2],
                                        min_feature_representation=10)
        assert result is not None
        assert result.n_outliers_excluded >= 1, "Anomalous pellet obs should be excluded"

    def test_10_systematic_sensor_drift_absorbed(self):
        """Consistent sensor bias absorbed into intercept."""
        rng = random.Random(42)
        obs = []
        bias = 0.3  # sensor reads 0.3°C high
        for i in range(100):
            outdoor = rng.uniform(0, 10)
            y_true = 2.0 + 0.3 * outdoor
            sp = y_true + 20.0 + bias + rng.gauss(0, 0.05)
            obs.append(_make_obs(
                t=float(i), outdoor_delta=outdoor,
                sp=sp, n_features=3,
            ))

        result = weighted_least_squares(obs, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        assert result is not None
        # Intercept should absorb the bias: ~2.3 instead of 2.0
        assert abs(result.beta_batch[0] - 2.3) < 0.15, (
            f"Intercept should absorb bias: got {result.beta_batch[0]:.3f}"
        )
        # Outdoor slope should be unaffected: ~0.3
        assert abs(result.beta_batch[1] - 0.3) < 0.1, (
            f"Outdoor slope should be unaffected: got {result.beta_batch[1]:.3f}"
        )


# ── Physical changes and drift behavior (11-14) ──────────────────────


class TestPhysicalChanges:

    def test_11_intercept_change_batch_converges(self):
        """Batch converges after intercept change without intervention."""
        rng = random.Random(42)
        buf = DiversityAwareBuffer(n_features=3, max_size=500, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        # 200 observations with intercept=3.0
        for i in range(200):
            outdoor = rng.uniform(0, 10)
            sp = 3.0 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
            buf.add(_make_obs(
                t=float(i), outdoor_delta=outdoor,
                sp=sp, n_features=3,
            ))
        # 100 observations with intercept=2.5 (window sealed)
        for i in range(100):
            outdoor = rng.uniform(0, 10)
            sp = 2.5 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
            buf.add(_make_obs(
                t=200.0 + i, outdoor_delta=outdoor,
                sp=sp, n_features=3,
            ))

        result = weighted_least_squares(buf.get_all(), n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        assert result is not None
        # Intercept should be between 2.5 and 3.0, trending toward 2.5
        # as more new observations accumulate
        assert 2.3 < result.beta_batch[0] < 3.1, (
            f"Intercept should blend old and new: got {result.beta_batch[0]:.3f}"
        )

    def test_12_intercept_change_other_coefficients_stable(self):
        """Other coefficients stay stable during intercept transition."""
        rng = random.Random(42)
        buf = DiversityAwareBuffer(n_features=3, max_size=500, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        # Phase 1: intercept=3.0, slope=0.3
        for i in range(200):
            outdoor = rng.uniform(0, 10)
            sp = 3.0 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
            buf.add(_make_obs(
                t=float(i), outdoor_delta=outdoor,
                sp=sp, n_features=3,
            ))
        # Phase 2: intercept=2.5, slope still 0.3
        for i in range(100):
            outdoor = rng.uniform(0, 10)
            sp = 2.5 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
            buf.add(_make_obs(
                t=200.0 + i, outdoor_delta=outdoor,
                sp=sp, n_features=3,
            ))

        result = weighted_least_squares(buf.get_all(), n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        assert result is not None
        assert abs(result.beta_batch[1] - 0.3) < 0.05, (
            f"Outdoor slope should stay stable: got {result.beta_batch[1]:.3f}"
        )

    def test_13_pi_gap_during_convergence_is_small(self):
        """Batch error during transition is small enough for PI to cover."""
        rng = random.Random(42)
        buf = DiversityAwareBuffer(n_features=3, max_size=500, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        # 300 observations with intercept=3.0
        for i in range(300):
            outdoor = rng.uniform(0, 10)
            sp = 3.0 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
            buf.add(_make_obs(
                t=float(i), outdoor_delta=outdoor,
                sp=sp, n_features=3,
            ))
        # 50 observations with intercept=2.5
        for i in range(50):
            outdoor = rng.uniform(0, 10)
            sp = 2.5 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
            buf.add(_make_obs(
                t=300.0 + i, outdoor_delta=outdoor,
                sp=sp, n_features=3,
            ))

        result = weighted_least_squares(buf.get_all(), n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        assert result is not None
        # Gap between batch intercept and true new intercept (2.5) should be < 2°C
        gap = abs(result.beta_batch[0] - 2.5)
        assert gap < 2.0, (
            f"Intercept gap {gap:.2f}°C too large for PI to cover"
        )

    def test_14_persistent_correction_direction_detectable(self):
        """Persistent same-direction corrections are detectable from batch results."""
        rng = random.Random(42)
        # Simulate: true intercept is 2.5, but buffer history pulls toward 3.0
        # Each batch cycle, track whether the correction is in the same direction
        corrections: list[float] = []
        for cycle in range(6):
            buf = DiversityAwareBuffer(n_features=3, max_size=500, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
            # Old data: intercept=3.0
            for i in range(200):
                outdoor = rng.uniform(0, 10)
                sp = 3.0 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
                buf.add(_make_obs(
                    t=float(i), outdoor_delta=outdoor,
                    sp=sp, n_features=3,
                ))
            # New data: intercept=2.5 (growing each cycle)
            n_new = 30 * (cycle + 1)
            for i in range(n_new):
                outdoor = rng.uniform(0, 10)
                sp = 2.5 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
                buf.add(_make_obs(
                    t=200.0 + i, outdoor_delta=outdoor,
                    sp=sp, n_features=3,
                ))

            result = weighted_least_squares(buf.get_all(), n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
            if result is not None:
                # Correction: batch says intercept should be X, online is at 3.0
                corrections.append(result.beta_batch[0] - 3.0)

        # All corrections should be negative (pulling intercept down toward 2.5)
        same_direction = all(c < 0 for c in corrections)
        assert same_direction, (
            f"Corrections should consistently pull downward: {corrections}"
        )


# ── Model input changes (15-17) ──────────────────────────────────────


class TestModelInputChanges:

    def test_15_old_observations_participate_for_existing_features(self):
        """Old observations (missing new feature) still contribute to other coefficients."""
        rng = random.Random(42)
        # 100 observations with 3 features (intercept, outdoor, solar)
        obs_old = []
        for i in range(100):
            outdoor = rng.uniform(0, 10)
            solar = rng.uniform(0, 0.5)
            sp = 2.0 + 0.3 * outdoor - 4.0 * solar + 20.0 + rng.gauss(0, 0.1)
            obs_old.append(_make_obs(
                t=float(i), outdoor_delta=outdoor, solar=solar,
                sp=sp, n_features=3,  # no pellet column
            ))

        # Fit with 4 features — old observations have features[:4] which pads pellet=0
        result = weighted_least_squares(obs_old, n_features=4, feature_order=TEST_FEATURE_ORDER[:4], model_inputs=TEST_MODEL_INPUTS[:2])
        assert result is not None
        # Intercept and outdoor slope should still be well-identified
        assert abs(result.beta_batch[0] - 2.0) < 0.3
        assert abs(result.beta_batch[1] - 0.3) < 0.1

    def test_16_removed_feature_old_observations_usable(self):
        """Old observations with extra feature column are usable after feature removal."""
        rng = random.Random(42)
        obs = []
        for i in range(100):
            outdoor = rng.uniform(0, 10)
            pellet = 1.0 if i < 20 else 0.0
            sp = 2.0 + 0.3 * outdoor - 6.0 * pellet + 20.0 + rng.gauss(0, 0.1)
            obs.append(_make_obs(
                t=float(i), outdoor_delta=outdoor, pellet=pellet,
                sp=sp, n_features=4,
            ))

        # Fit with only 3 features (pellet removed) — old obs have extra raw_readings
        result = weighted_least_squares(obs, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        assert result is not None
        assert abs(result.beta_batch[1] - 0.3) < 0.15, (
            f"Outdoor slope should be stable: got {result.beta_batch[1]:.3f}"
        )

    def test_17_missing_feature_does_not_contaminate(self):
        """Old observations with missing pellet data don't bias pellet coefficient."""
        rng = random.Random(42)
        # 50 old observations: no pellet data (filled as 0.0)
        obs = []
        for i in range(50):
            outdoor = rng.uniform(0, 10)
            sp = 2.0 + 0.3 * outdoor + 20.0 + rng.gauss(0, 0.1)
            obs.append(_make_obs(
                t=float(i), outdoor_delta=outdoor,
                sp=sp, n_features=4,  # pellet=0 (missing, backfilled)
            ))
        # 30 new observations with actual pellet data
        for i in range(30):
            outdoor = rng.uniform(0, 10)
            pellet = 1.0 if i < 15 else 0.0
            sp = 2.0 + 0.3 * outdoor - 6.0 * pellet + 20.0 + rng.gauss(0, 0.1)
            obs.append(_make_obs(
                t=50.0 + i, outdoor_delta=outdoor, pellet=pellet,
                sp=sp, n_features=4,
            ))

        result = weighted_least_squares(obs, n_features=4, feature_order=TEST_FEATURE_ORDER[:4], model_inputs=TEST_MODEL_INPUTS[:2])
        assert result is not None
        # Pellet coefficient should be learned from the 30 new observations
        assert abs(result.beta_batch[3] - (-6.0)) < 1.5, (
            f"Pellet coefficient should be ~-6.0, got {result.beta_batch[3]:.3f}"
        )


# ── Buffer mechanics (18-23) ─────────────────────────────────────────


class TestBufferMechanics:

    def test_18_leverage_identifies_novel_observations(self):
        """Novel observations have higher leverage than redundant ones."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        # Fill with observations at outdoor_delta=5
        for i in range(50):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0, n_features=3))

        # Compute leverage for redundant vs novel observation
        redundant = [1.0, 5.0, 0.0]
        novel = [1.0, 20.0, 0.0]
        lev_redundant = buf._compute_leverage(redundant)
        lev_novel = buf._compute_leverage(novel)

        assert lev_novel > lev_redundant, (
            f"Novel leverage {lev_novel:.4f} should exceed redundant {lev_redundant:.4f}"
        )

    def test_19_eviction_displaces_lowest_leverage(self):
        """Eviction targets lowest-leverage observation, not oldest."""
        buf = DiversityAwareBuffer(n_features=3, max_size=10, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        # Add diverse observations
        for i in range(10):
            buf.add(_make_obs(t=float(i), outdoor_delta=float(i * 2), n_features=3))

        # The observation with lowest leverage (most redundant) should be evicted
        scores_before = buf.get_leverage_scores()
        min_score_before = min(scores_before)

        # Add a novel observation
        buf.add(_make_obs(t=100.0, outdoor_delta=30.0, solar=0.5, n_features=3))

        scores_after = buf.get_leverage_scores()
        # The minimum leverage should be >= what it was, because the
        # least-informative observation was replaced
        min_score_after = min(scores_after)
        assert min_score_after >= min_score_before * 0.9, (
            f"Min leverage should not decrease: {min_score_before:.6f} → {min_score_after:.6f}"
        )

    def test_20_recomputation_matches_incremental(self):
        """Periodic recomputation matches incremental Sherman-Morrison updates."""
        buf = DiversityAwareBuffer(n_features=3, max_size=200, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        for i in range(100):
            buf.add(_make_obs(t=float(i), outdoor_delta=float(i % 10),
                              solar=0.1 * (i % 5), n_features=3))

        # Get leverage scores from incremental updates
        scores_incremental = buf.get_leverage_scores()

        # Force recomputation from scratch
        buf.recompute_info_matrix()
        scores_recomputed = buf.get_leverage_scores()

        for i in range(len(scores_incremental)):
            assert abs(scores_incremental[i] - scores_recomputed[i]) < 0.01, (
                f"Score {i}: incremental {scores_incremental[i]:.6f} "
                f"vs recomputed {scores_recomputed[i]:.6f}"
            )

    def test_21_regularization_prevents_singularity(self):
        """λI regularization keeps matrix invertible with fewer obs than features."""
        buf = DiversityAwareBuffer(n_features=5, max_size=100, feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS)
        # Add only 3 observations for 5 features (rank deficient)
        buf.add(_make_obs(t=0.0, outdoor_delta=5.0))
        buf.add(_make_obs(t=1.0, outdoor_delta=10.0))
        buf.add(_make_obs(t=2.0, outdoor_delta=15.0))

        # Leverage scores should be finite
        scores = buf.get_leverage_scores()
        assert all(math.isfinite(s) for s in scores), (
            f"All leverage scores should be finite: {scores}"
        )

    def test_22_migration_from_serialized_data(self):
        """Diversity buffer initializes correctly from serialized observation dicts."""
        # Simulate old serialized data (list of dicts, as stored in HA)
        serialized = [
            _make_obs(t=float(i), outdoor_delta=float(i % 10),
                      solar=0.1 * (i % 5)).as_dict()
            for i in range(50)
        ]

        # Import into diversity buffer
        new_buf = DiversityAwareBuffer.from_list(
            serialized, n_features=5,
            feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS,
        )
        assert len(new_buf) == 50
        scores = new_buf.get_leverage_scores()
        assert len(scores) == 50
        assert all(math.isfinite(s) for s in scores)

    def test_23_serialization_roundtrip(self):
        """Serialize and deserialize preserves observations."""
        buf = DiversityAwareBuffer(n_features=5, max_size=100, feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS)
        for i in range(30):
            buf.add(_make_obs(t=float(i), outdoor_delta=float(i % 10),
                              pellet=1.0 if i % 10 == 0 else 0.0))

        serialized = buf.as_list()
        restored = DiversityAwareBuffer.from_list(
            serialized, n_features=5,
            feature_order=TEST_FEATURE_ORDER, model_inputs=TEST_MODEL_INPUTS,
        )

        assert len(restored) == len(buf)
        orig = buf.get_all()
        rest = restored.get_all()
        for i in range(len(orig)):
            assert orig[i].timestamp == rest[i].timestamp
            assert orig[i].hp_setpoint == rest[i].hp_setpoint


# ── Regime coverage and convergence (24-26) ───────────────────────────


class TestRegimeCoverage:

    def test_24_cold_snap_after_mild_both_regimes_in_buffer(self):
        """Both mild and cold regimes contribute after a cold snap."""
        buf = DiversityAwareBuffer(n_features=3, max_size=200, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        rng = random.Random(42)
        # 2 weeks of mild weather (outdoor_delta ≈ 5)
        for i in range(336):  # 14 days * 24 hours
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0 + rng.gauss(0, 1),
                              n_features=3))
        # 48h of cold snap (outdoor_delta ≈ 20)
        for i in range(48):
            buf.add(_make_obs(t=336.0 + i, outdoor_delta=20.0 + rng.gauss(0, 1),
                              n_features=3))

        all_obs = buf.get_all()
        mild = [o for o in all_obs if (o.outdoor_temp_c - o.current_c) < 10]
        cold = [o for o in all_obs if (o.outdoor_temp_c - o.current_c) > 15]
        assert len(mild) > 10, f"Should retain mild observations: {len(mild)}"
        assert len(cold) > 10, f"Should retain cold observations: {len(cold)}"

    def test_25_multi_season_tighter_estimates(self):
        """Multi-season data produces tighter coefficient estimates than single season."""
        rng = random.Random(42)
        # Single season: only winter (outdoor_delta 10-20)
        obs_winter = []
        for i in range(200):
            outdoor = rng.uniform(10, 20)
            solar = rng.uniform(0, 0.2)  # low solar in winter
            sp = 2.0 + 0.3 * outdoor - 4.0 * solar + 20.0 + rng.gauss(0, 0.2)
            obs_winter.append(_make_obs(
                t=float(i), outdoor_delta=outdoor, solar=solar,
                sp=sp, n_features=3,
            ))

        # Multi-season: winter + spring + summer
        obs_multi = list(obs_winter)
        for i in range(100):
            outdoor = rng.uniform(0, 5)  # spring
            solar = rng.uniform(0, 0.6)  # more solar
            sp = 2.0 + 0.3 * outdoor - 4.0 * solar + 20.0 + rng.gauss(0, 0.2)
            obs_multi.append(_make_obs(
                t=200.0 + i, outdoor_delta=outdoor, solar=solar,
                sp=sp, n_features=3,
            ))
        for i in range(100):
            outdoor = rng.uniform(-5, 0)  # summer (negative delta = above reference)
            solar = rng.uniform(0.2, 0.6)
            sp = 2.0 + 0.3 * outdoor - 4.0 * solar + 20.0 + rng.gauss(0, 0.2)
            obs_multi.append(_make_obs(
                t=300.0 + i, outdoor_delta=outdoor, solar=solar,
                sp=sp, n_features=3,
            ))

        result_winter = weighted_least_squares(obs_winter, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        result_multi = weighted_least_squares(obs_multi, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])

        assert result_winter is not None and result_multi is not None
        # Multi-season should have smaller standard errors
        for i in range(3):
            se_w = result_winter.beta_std_err[i]
            se_m = result_multi.beta_std_err[i]
            if math.isfinite(se_w) and math.isfinite(se_m):
                assert se_m <= se_w * 1.1, (
                    f"Coefficient {i}: multi-season SE {se_m:.4f} should be "
                    f"<= winter-only SE {se_w:.4f}"
                )

    def test_26_convergence_to_true_coefficients(self):
        """Coefficients converge within 2σ of true values."""
        rng = random.Random(42)
        true_beta = [2.0, 0.3, -4.0]
        obs = []
        for i in range(500):
            outdoor = rng.uniform(-5, 20)
            solar = rng.uniform(0, 0.6)
            y_true = sum(b * x for b, x in zip(true_beta, [1.0, outdoor, solar]))
            sp = y_true + 20.0 + rng.gauss(0, 0.15)
            obs.append(_make_obs(
                t=float(i), outdoor_delta=outdoor, solar=solar,
                sp=sp, n_features=3,
            ))

        result = weighted_least_squares(obs, n_features=3, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        assert result is not None
        for i in range(3):
            se = result.beta_std_err[i]
            if math.isfinite(se) and se > 0:
                error = abs(result.beta_batch[i] - true_beta[i])
                assert error < 2.0 * se, (
                    f"Coefficient {i}: {result.beta_batch[i]:.3f} not within "
                    f"2σ ({2*se:.3f}) of true {true_beta[i]:.3f}"
                )


# ── Buffer clear ────────────────────────────────────────────────────


class TestBufferClear:
    """Tests for DiversityAwareBuffer.clear()."""

    def test_clear_empties_buffer(self):
        """clear() should remove all observations."""
        buf = DiversityAwareBuffer(n_features=3, max_size=50, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        for i in range(20):
            buf.add(_make_obs(t=float(i), outdoor_delta=float(i)))
        assert len(buf) == 20

        buf.clear()
        assert len(buf) == 0
        assert buf.get_all() == []

    def test_clear_resets_info_matrix(self):
        """After clear, info matrix should be back to regularized identity."""
        from custom_components.tasmota_irhvac.pi.batch_learning import INFO_MATRIX_REGULARIZATION

        buf = DiversityAwareBuffer(n_features=3, max_size=50, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        for i in range(20):
            buf.add(_make_obs(t=float(i), outdoor_delta=float(i)))

        buf.clear()

        reg_inv = 1.0 / INFO_MATRIX_REGULARIZATION
        for i in range(3):
            for j in range(3):
                expected = reg_inv if i == j else 0.0
                assert buf._info_inv[i][j] == pytest.approx(expected)

    def test_clear_allows_refill(self):
        """Buffer should accept new observations after clear."""
        buf = DiversityAwareBuffer(n_features=3, max_size=10, feature_order=TEST_FEATURE_ORDER[:3], model_inputs=TEST_MODEL_INPUTS[:1])
        for i in range(10):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0))
        assert len(buf) == 10

        buf.clear()
        for i in range(5):
            buf.add(_make_obs(t=100.0 + i, outdoor_delta=15.0))
        assert len(buf) == 5


class TestMulticollinearityWithFewFeatures:
    """compute_condition_number / compute_vif fall back when n_features < 3."""

    def test_condition_number_two_features_returns_one(self):
        """With intercept + outdoor_delta only, no feature-feature κ."""
        buf = DiversityAwareBuffer(
            n_features=2, max_size=20,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=[],
        )
        for i in range(15):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0 + i * 0.1, n_features=2))
        # n_features=2 → returns 1.0 (no collinearity to compute)
        assert buf.compute_condition_number() == 1.0

    def test_vif_two_features_returns_ones(self):
        """VIF with n_features<3 falls back to [1.0]*n."""
        buf = DiversityAwareBuffer(
            n_features=2, max_size=20,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=[],
        )
        for i in range(15):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0 + i * 0.1, n_features=2))
        vifs = buf.compute_vif()
        assert vifs == [1.0, 1.0]

    def test_correlations_include_top_pair_when_below_threshold(self):
        """include_top=True returns the strongest pair even if no |r| > 0.7."""
        buf = DiversityAwareBuffer(
            n_features=4, max_size=80,
            feature_order=["intercept", "outdoor_delta", "solar", "pellet"],
            model_inputs=[
                {"entity_id": _SOLAR_ENTITY, "name": "solar"},
                {"entity_id": _PELLET_ENTITY, "name": "pellet"},
            ],
        )
        # Independent random-ish features → no |r| crosses 0.7
        import random
        rng = random.Random(42)
        for i in range(60):
            buf.add(_make_obs(
                t=float(i),
                outdoor_delta=rng.uniform(-2.0, 2.0),
                solar=rng.uniform(0.0, 1.0),
                pellet=rng.uniform(0.0, 0.5),
                n_features=4,
            ))
        names = ["intercept", "outdoor_delta", "solar", "pellet"]
        # Without include_top: empty (no pair > 0.7)
        regular = buf.get_pairwise_correlations(names, include_top=False)
        assert regular == []
        # With include_top: returns the single strongest pair
        with_top = buf.get_pairwise_correlations(names, include_top=True)
        assert len(with_top) == 1
        assert with_top[0][0] in names and with_top[0][1] in names


# ── BufferAddResult contract (admission observability) ───────────────


class TestAddReturnsDecision:
    """Decision metadata returned by `DiversityAwareBuffer.add()`.

    Exposes admission outcome, candidate leverage, eviction target, and
    the worst-incumbent leverage at decision time so callers (the
    controller's tick output) don't have to re-derive buffer state.
    """

    def _buf(self, max_size: int = 5) -> DiversityAwareBuffer:
        return DiversityAwareBuffer(
            n_features=3, max_size=max_size,
            feature_order=TEST_FEATURE_ORDER[:3],
            model_inputs=TEST_MODEL_INPUTS[:1],
        )

    def test_add_into_empty_buffer_returns_admitted_no_eviction(self):
        buf = self._buf()
        r = buf.add(_make_obs(t=0.0, outdoor_delta=5.0, n_features=3))
        # Structural check rather than isinstance — `test_batch_learning.py`
        # reloads the batch_learning module to exercise the no-numpy path,
        # which replaces `BufferAddResult` in module globals so an
        # isinstance check against the test's frozen import would fail
        # depending on test execution order.
        assert hasattr(r, "admitted")
        assert r.admitted is True
        assert r.candidate_score is not None and r.candidate_score > 0
        assert r.evicted_timestamp is None
        assert r.min_incumbent_score is None
        assert r.rejection_reason is None
        assert r.policy_name == "leverage"

    def test_add_into_partially_full_buffer_returns_admitted_no_eviction(self):
        buf = self._buf(max_size=5)
        for i in range(3):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0 + i, n_features=3))
        r = buf.add(_make_obs(t=10.0, outdoor_delta=20.0, n_features=3))
        assert r.admitted is True
        assert r.evicted_timestamp is None
        assert r.min_incumbent_score is None
        assert r.rejection_reason is None
        assert r.policy_name == "leverage"

    def test_add_into_full_buffer_with_higher_leverage_admits_and_reports_evicted(self):
        buf = self._buf(max_size=5)
        # Fill with redundant observations.
        for i in range(5):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0, n_features=3))

        # Snapshot current min-leverage timestamp via the buffer's API
        scores = buf.get_leverage_scores()
        min_idx = scores.index(min(scores))
        evicted_ts = buf._buffer[min_idx].timestamp

        # A novel observation (large outdoor_delta) should beat the worst incumbent.
        r = buf.add(_make_obs(t=100.0, outdoor_delta=30.0, n_features=3))
        assert r.admitted is True
        assert r.evicted_timestamp == evicted_ts
        assert r.min_incumbent_score is not None
        assert r.candidate_score is not None
        assert r.candidate_score > r.min_incumbent_score
        assert r.rejection_reason is None
        assert r.policy_name == "leverage"

    def test_add_into_full_buffer_with_lower_leverage_rejects_with_reason(self):
        buf = self._buf(max_size=5)
        # Fill with identical observations.  Each slot ends up with the
        # same leverage; another identical candidate has equal leverage,
        # which fails the strict `new > min` admission check.
        for i in range(5):
            buf.add(_make_obs(t=float(i), outdoor_delta=5.0, n_features=3))

        r = buf.add(_make_obs(t=100.0, outdoor_delta=5.0, n_features=3))
        assert r.admitted is False
        assert r.evicted_timestamp is None
        assert r.min_incumbent_score is not None
        assert r.candidate_score is not None
        assert r.candidate_score <= r.min_incumbent_score + 1e-9
        assert r.rejection_reason == "leverage_rejected"
        assert r.policy_name == "leverage"
        # Buffer size unchanged; original timestamps still present.
        assert len(buf._buffer) == 5
        timestamps = [o.timestamp for o in buf._buffer]
        assert 100.0 not in timestamps
