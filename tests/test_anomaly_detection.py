"""Tests for CUSUM anomaly detection (#21 Phase 1).

Tests cover:
- MAD scale estimation (compute_mad_sigma)
- CUSUM detection on synthetic residual sequences
- Detection latency vs ARL₁ predictions
- False positive resistance
- Cooldown and clamped-tick behavior
"""

import math
import random
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from custom_components.tasmota_irhvac.pi.health_checks import (
    AnomalyEvent,
    CUSUM_COOLDOWN_SEC,
    CUSUM_H,
    CUSUM_K,
    MIN_EVENT_DURATION_SEC,
    MIN_RESIDUALS_FOR_DETECTION,
    MIN_SIGMA_FLOOR,
    compute_mad_sigma,
)


# ── compute_mad_sigma ────────────────────────────────────────────────


class TestComputeMadSigma:
    """Tests for MAD-based robust scale estimation."""

    def test_gaussian_samples(self):
        """MAD should approximate true σ for Gaussian data."""
        random.seed(42)
        samples = [random.gauss(0, 1.0) for _ in range(60)]
        sigma = compute_mad_sigma(samples)
        # Should be within ~30% of true σ=1.0 for n=60
        assert 0.6 < sigma < 1.5

    def test_contaminated_samples(self):
        """MAD should be robust to 40% outlier contamination."""
        random.seed(42)
        clean = [random.gauss(0, 0.15) for _ in range(36)]
        outliers = [random.gauss(5.0, 0.5) for _ in range(24)]  # 40% contamination
        sigma = compute_mad_sigma(clean + outliers)
        # MAD should track the clean component, not the outliers
        # Clean σ = 0.15, should be within factor of 3
        assert sigma < 0.5

    def test_identical_residuals(self):
        """All-identical residuals should return MIN_SIGMA_FLOOR."""
        samples = [0.1] * 60
        sigma = compute_mad_sigma(samples)
        assert sigma == MIN_SIGMA_FLOOR

    def test_empty_returns_floor(self):
        """Empty input returns MIN_SIGMA_FLOOR."""
        assert compute_mad_sigma([]) == MIN_SIGMA_FLOOR
        assert compute_mad_sigma(deque()) == MIN_SIGMA_FLOOR

    def test_single_sample(self):
        """Single sample returns floor (MAD of one point is 0)."""
        assert compute_mad_sigma([1.0]) == MIN_SIGMA_FLOOR

    def test_two_samples(self):
        """Two samples should give reasonable result."""
        sigma = compute_mad_sigma([0.0, 1.0])
        assert sigma > 0

    def test_deque_input(self):
        """Works with deque (the actual usage type)."""
        d = deque([0.1, -0.2, 0.05, -0.15, 0.3], maxlen=60)
        sigma = compute_mad_sigma(d)
        assert sigma > MIN_SIGMA_FLOOR


# ── CUSUM detection via PIController._update_cusum ───────────────────


def _make_cusum_controller():
    """Create a minimal PIController-like object for CUSUM testing.

    Only initializes the fields needed by _update_cusum and
    _finalize_anomaly_event, bypassing the full __init__.
    """
    from custom_components.tasmota_irhvac.pi.pi_controller import PIController

    # Use object.__new__ to skip __init__
    ctrl = object.__new__(PIController)
    ctrl._residual_history = deque(maxlen=60)
    ctrl._cusum_pos = 0.0
    ctrl._cusum_neg = 0.0
    ctrl._anomaly_events = []
    ctrl._exclusion_count = 0
    ctrl._cusum_cooldown_until = None
    ctrl._metrics = MagicMock()
    ctrl._metrics.batch_model_rms = None
    ctrl._pending_events = []  # Stage 8: event accumulator (used by _emit_event)
    return ctrl


def _feed_residuals(ctrl, residuals, sigma=0.15, start_mono=1000.0, tick_spacing=60.0):
    """Feed a sequence of residuals and return the controller.

    Pre-fills residual history with normal data so MAD is calibrated,
    then feeds the provided sequence with synthetic wall-clock time.
    """
    random.seed(123)
    # Pre-fill with normal residuals to establish MAD baseline
    for _ in range(MIN_RESIDUALS_FOR_DETECTION):
        ctrl._residual_history.append(random.gauss(0, sigma))

    mono = start_mono
    base_time = datetime(2026, 1, 1)
    for i, r in enumerate(residuals):
        now = base_time + timedelta(seconds=i * tick_spacing)
        ctrl._update_cusum(r, mono, is_heating=True, _now=now)
        mono += tick_spacing
    return ctrl


class TestCusumDetectionShortTerm:
    """Short-term disturbance detection scenarios."""

    def test_window_open_winter(self):
        """Large negative step (-8σ) detected quickly."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        residuals = [-8.0 * sigma] * 5
        _feed_residuals(ctrl, residuals, sigma=sigma)

        assert len(ctrl._anomaly_events) >= 1
        event = ctrl._anomaly_events[0]
        assert event.mean_residual < 0
        assert event.peak_cusum > CUSUM_H

    def test_window_open_spring(self):
        """Moderate negative step (-3σ) detected."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        residuals = [-3.0 * sigma] * 15
        _feed_residuals(ctrl, residuals, sigma=sigma)

        assert len(ctrl._anomaly_events) >= 1

    def test_cooking_positive_shift(self):
        """+2.5σ sustained detected as positive event."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        residuals = [2.5 * sigma] * 15
        _feed_residuals(ctrl, residuals, sigma=sigma)

        assert len(ctrl._anomaly_events) >= 1
        event = ctrl._anomaly_events[0]
        assert event.mean_residual > 0

    def test_sensor_glitch_no_alarm(self):
        """Single moderate spike (+4σ) should NOT trigger alarm."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        # +4σ spike: z ≈ 4, accumulates (4-1)=3 → well below h=10
        residuals = [4.0 * sigma] + [0.0] * 20
        _feed_residuals(ctrl, residuals, sigma=sigma)

        assert len(ctrl._anomaly_events) == 0


class TestCusumDetectionLongTerm:
    """Long-term disturbance detection scenarios."""

    def test_all_day_solar_gain(self):
        """Sustained +1.5σ shift detected."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        residuals = [1.5 * sigma] * 50
        _feed_residuals(ctrl, residuals, sigma=sigma)

        assert len(ctrl._anomaly_events) >= 1

    def test_space_heater_hours(self):
        """+4σ sustained 100 ticks — detected early."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        residuals = [4.0 * sigma] * 20
        _feed_residuals(ctrl, residuals, sigma=sigma)

        assert len(ctrl._anomaly_events) >= 1
        event = ctrl._anomaly_events[0]
        assert event.peak_cusum > CUSUM_H

    def test_gradual_drift_hidden_in_noise(self):
        """Small drift buried in noise should NOT trigger alarm."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(55)
        # Feed noisy residuals with a small +0.3σ bias.
        # The noise keeps MAD ≈ σ, so z ≈ 0.3 < k=1.0 → no accumulation.
        mono = 1000.0
        base = datetime(2026, 1, 1)
        for i in range(200):
            r = random.gauss(0.3 * sigma, sigma)
            ctrl._update_cusum(r, mono, is_heating=True, _now=base + timedelta(seconds=i*60))
            mono += 60.0

        assert len(ctrl._anomaly_events) == 0

    def test_overnight_window_two_events(self):
        """-3σ for 40 ticks, normal 60 ticks, -3σ for 40 ticks → 2+ events."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        # First anomaly, then normal (>30 min cooldown at 60s/tick),
        # then second anomaly.
        residuals = (
            [-3.0 * sigma] * 40   # first anomaly
            + [0.0] * 60          # 60 min normal (>30 min cooldown)
            + [-3.0 * sigma] * 40 # second anomaly
        )
        _feed_residuals(ctrl, residuals, sigma=sigma, tick_spacing=60.0)

        # Should get events from both anomaly periods
        assert len(ctrl._anomaly_events) >= 2


class TestCusumFalsePositiveResistance:
    """False positive resistance scenarios."""

    def test_normal_operation_no_alarm(self):
        """1000 ticks of N(0,σ) should not trigger alarm."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(99)
        residuals = [random.gauss(0, sigma) for _ in range(1000)]
        _feed_residuals(ctrl, residuals, sigma=sigma)

        # ARL₀ ≈ 50,000 — 1000 ticks is well below
        assert len(ctrl._anomaly_events) == 0

    def test_setpoint_change_moderate_no_alarm(self):
        """Moderate transient (+2σ for 2 ticks) doesn't trigger with good MAD."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(88)
        # Pre-fill with 60 clean samples for well-calibrated MAD
        ctrl._residual_history.clear()
        for _ in range(60):
            ctrl._residual_history.append(random.gauss(0, sigma))

        # +2σ for 2 ticks: z ≈ 2, accumulates (2-1)*2=2, well below h=10
        base = datetime(2026, 1, 1)
        mono = 1000.0
        transient = [2.0 * sigma] * 2 + [0.0] * 10
        for i, r in enumerate(transient):
            ctrl._update_cusum(r, mono, is_heating=True, _now=base + timedelta(seconds=i*60))
            mono += 60.0

        assert len(ctrl._anomaly_events) == 0

    def test_noisy_sensor(self):
        """Elevated noise (1.5× normal) absorbed by MAD."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(77)
        # Pre-fill with elevated noise so MAD adapts
        for _ in range(MIN_RESIDUALS_FOR_DETECTION):
            ctrl._residual_history.append(random.gauss(0, sigma * 1.5))
        # Continue with elevated noise
        mono = 1000.0
        base = datetime(2026, 1, 1)
        for i in range(500):
            r = random.gauss(0, sigma * 1.5)
            ctrl._update_cusum(r, mono, is_heating=True, _now=base + timedelta(seconds=i*60))
            mono += 60.0

        assert len(ctrl._anomaly_events) == 0


class TestCusumClampedAndCooldown:
    """Clamped and cooldown behavior."""

    def test_cooldown_suppresses_detection(self):
        """During cooldown, new anomalies are not detected."""
        ctrl = _make_cusum_controller()
        sigma = 0.15

        base = datetime(2026, 1, 1)
        # Set cooldown to 30 min from base
        ctrl._cusum_cooldown_until = base + timedelta(minutes=30)

        # Pre-fill history
        for _ in range(MIN_RESIDUALS_FOR_DETECTION):
            ctrl._residual_history.append(random.gauss(0, sigma))

        # Feed large anomaly during cooldown (all within 20 minutes of base)
        mono = 1000.0
        for i in range(20):
            now = base + timedelta(seconds=i * 60)  # 0-19 min, within cooldown
            ctrl._update_cusum(-8.0 * sigma, mono, is_heating=True, _now=now)
            mono += 60.0

        # Should NOT have detected — cooldown suppressed
        assert len(ctrl._anomaly_events) == 0
        # But residuals should still accumulate in history
        assert len(ctrl._residual_history) > MIN_RESIDUALS_FOR_DETECTION

    def test_cooldown_expires(self):
        """After cooldown expires, detection resumes."""
        ctrl = _make_cusum_controller()
        sigma = 0.15

        base = datetime(2026, 1, 1)
        # Set cooldown to before base (already expired)
        ctrl._cusum_cooldown_until = base - timedelta(minutes=1)

        residuals = [-5.0 * sigma] * 15
        # _feed_residuals starts time from datetime(2026,1,1) → after cooldown
        _feed_residuals(ctrl, residuals, sigma=sigma)

        # Should detect after cooldown expired
        assert len(ctrl._anomaly_events) >= 1


class TestCusumDetectionLatency:
    """Verify detection latency matches ARL₁ predictions."""

    def _detect_ticks(self, shift_sigma: float, sigma: float = 0.15) -> int:
        """Feed a step shift and return ticks until CUSUM crosses h."""
        ctrl = _make_cusum_controller()
        random.seed(42)
        # Pre-fill with 60 clean samples for well-calibrated MAD
        for _ in range(60):
            ctrl._residual_history.append(random.gauss(0, sigma))

        mono = 1000.0
        base = datetime(2026, 1, 1)
        for tick in range(1, 200):
            now = base + timedelta(seconds=tick * 60)
            ctrl._update_cusum(shift_sigma * sigma, mono, is_heating=True, _now=now)
            mono += 60.0
            if len(ctrl._anomaly_events) > 0:
                return tick
        return 200  # never detected

    def test_10sigma_immediate(self):
        """10σ shift detected in 1-2 ticks."""
        ticks = self._detect_ticks(10.0)
        assert ticks <= 2

    def test_5sigma_fast(self):
        """5σ shift detected in 2-4 ticks."""
        ticks = self._detect_ticks(5.0)
        assert ticks <= 4

    def test_3sigma_moderate(self):
        """3σ shift detected in 4-7 ticks."""
        ticks = self._detect_ticks(3.0)
        assert ticks <= 8

    def test_2sigma_slow(self):
        """2σ shift detected in ~10 ticks."""
        ticks = self._detect_ticks(2.0)
        assert ticks <= 15

    def test_half_sigma_below_deadzone(self):
        """0.5σ shift buried in noise should not trigger in 200 ticks."""
        # z ≈ 0.5 < k=1.0 → expected CUSUM increment ≈ -0.5 per tick
        # (decays, never accumulates)
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(42)
        for _ in range(60):
            ctrl._residual_history.append(random.gauss(0, sigma))

        mono = 1000.0
        base = datetime(2026, 1, 1)
        for tick in range(200):
            r = random.gauss(0.5 * sigma, sigma)
            now = base + timedelta(seconds=tick * 60)
            ctrl._update_cusum(r, mono, is_heating=True, _now=now)
            mono += 60.0

        assert len(ctrl._anomaly_events) == 0


# ── Buffer exclusion ─────────────────────────────────────────────────


class TestBufferExclusion:
    """Tests for DiversityAwareBuffer.exclude_time_range."""

    def test_exclude_removes_matching_observations(self):
        """Observations in time range are removed."""
        from custom_components.tasmota_irhvac.pi.batch_learning import (
            DiversityAwareBuffer, Observation,
        )
        buf = DiversityAwareBuffer(n_features=2, max_size=50)
        for i in range(20):
            buf.add(Observation(
                timestamp=1000.0 + i * 60,
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i),
                room_rate=0.0,
                raw_readings={},
                clamped=False,
            ))
        assert len(buf) == 20

        # Exclude observations 5-14 (timestamps 1300-1840)
        removed = buf.exclude_time_range(1300.0, 1840.0)
        assert removed == 10
        assert len(buf) == 10

        # Verify remaining timestamps are outside the range
        for o in buf._buffer:
            assert o.timestamp < 1300.0 or o.timestamp > 1840.0

    def test_exclude_no_match(self):
        """No-op when range doesn't match any observations."""
        from custom_components.tasmota_irhvac.pi.batch_learning import (
            DiversityAwareBuffer, Observation,
        )
        buf = DiversityAwareBuffer(n_features=2, max_size=50)
        for i in range(10):
            buf.add(Observation(
                timestamp=1000.0 + i * 60,
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i),
                room_rate=0.0,
                raw_readings={},
                clamped=False,
            ))

        removed = buf.exclude_time_range(5000.0, 6000.0)
        assert removed == 0
        assert len(buf) == 10

    def test_exclude_all_observations(self):
        """Excluding all observations empties the buffer."""
        from custom_components.tasmota_irhvac.pi.batch_learning import (
            DiversityAwareBuffer, Observation,
        )
        buf = DiversityAwareBuffer(n_features=2, max_size=50)
        for i in range(5):
            buf.add(Observation(
                timestamp=1000.0 + i * 60,
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i),
                room_rate=0.0,
                raw_readings={},
                clamped=False,
            ))

        removed = buf.exclude_time_range(0.0, 99999.0)
        assert removed == 5
        assert len(buf) == 0

    def test_exclude_recomputes_info_matrix(self):
        """Info matrix is recomputed after exclusion."""
        from custom_components.tasmota_irhvac.pi.batch_learning import (
            DiversityAwareBuffer, Observation,
        )
        buf = DiversityAwareBuffer(
            n_features=2, max_size=50,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=[],
        )
        for i in range(10):
            buf.add(Observation(
                timestamp=1000.0 + i * 60,
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i) * 0.5,
                room_rate=0.0,
                raw_readings={},
                clamped=False,
            ))

        # Record info matrix state before
        info_before = [row[:] for row in buf._info_inv]

        buf.exclude_time_range(1300.0, 1600.0)

        # Info matrix should have changed
        info_after = buf._info_inv
        assert info_before != info_after


class TestPIControllerExclusion:
    """Tests for PIController.exclude_observations_by_time."""

    @pytest.mark.asyncio
    async def test_exclude_increments_count(self, hass, setup_pi_integration):
        """Exclusion increments _exclusion_count."""
        from .conftest import get_climate_entity
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Add observations to heat buffer
        for i in range(10):
            pi._observation_buffer_heat.add(Observation(
                timestamp=1000.0 + i * 60,
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i),
                room_rate=0.0,
                raw_readings={},
                clamped=False,
            ))

        assert pi._exclusion_count == 0
        removed = pi.exclude_observations_by_time(1300.0, 1600.0)
        assert removed > 0
        assert pi._exclusion_count == 1

    @pytest.mark.asyncio
    async def test_exclude_no_match_no_increment(self, hass, setup_pi_integration):
        """No-op exclusion does not increment count."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        removed = pi.exclude_observations_by_time(99999.0, 99999.9)
        assert removed == 0
        assert pi._exclusion_count == 0


class TestExclusionPersistence:
    """Tests for exclusion_count persistence in PIStoredData."""

    def test_exclusion_count_round_trip(self):
        """exclusion_count survives serialize/deserialize."""
        from custom_components.tasmota_irhvac.pi.pi_stored_data import PIExtraStoredData

        data = PIExtraStoredData(
            pi_integral=0.0,
            desired_temp=21.0,
            hp_setpoint=22.0,
            exclusion_count=5,
        )
        d = data.as_dict()
        assert d["exclusion_count"] == 5

        restored = PIExtraStoredData.from_dict(d)
        assert restored is not None
        assert restored.exclusion_count == 5

    def test_exclusion_count_defaults_to_zero(self):
        """Missing exclusion_count in stored data defaults to 0."""
        from custom_components.tasmota_irhvac.pi.pi_stored_data import PIExtraStoredData

        d = {"pi_integral": 0.0, "desired_temp": 21.0, "hp_setpoint": 22.0}
        restored = PIExtraStoredData.from_dict(d)
        assert restored is not None
        assert restored.exclusion_count == 0


class TestAnomalyEventDataclass:
    """Basic AnomalyEvent tests."""

    def test_creation(self):
        """AnomalyEvent can be created with all fields."""
        now = datetime.now()
        event = AnomalyEvent(
            start_time=now,
            start_mono=100.0,
            end_time=now + timedelta(minutes=15),
            end_mono=1000.0,
            tick_count=15,
            mean_residual=-0.8,
            peak_cusum=25.0,
            mode="heat",
        )
        assert event.mode == "heat"
        assert event.mean_residual < 0
        assert event.tick_count == 15

    def test_mode_stored(self):
        """Mode is preserved for cause hint generation."""
        now = datetime.now()
        heat_event = AnomalyEvent(
            start_time=now, start_mono=0, end_time=now,
            end_mono=0, tick_count=1, mean_residual=0.5,
            peak_cusum=11.0, mode="heat",
        )
        cool_event = AnomalyEvent(
            start_time=now, start_mono=0, end_time=now,
            end_mono=0, tick_count=1, mean_residual=0.5,
            peak_cusum=11.0, mode="cool",
        )
        assert heat_event.mode != cool_event.mode
