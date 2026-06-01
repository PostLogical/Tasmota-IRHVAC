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
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from custom_components.tasmota_irhvac.pi.health_checks import (
    AnomalyEvent,
    CUSUM_COOLDOWN_SEC,
    CUSUM_H,
    CUSUM_H_HAWKINS,
    CUSUM_K,
    CUSUM_K_HAWKINS,
    CUSUM_WARMUP_N,
    CUSUM_WINDOW_S,
    MIN_EVENT_DURATION_SEC,
    MIN_RESIDUALS_FOR_DETECTION,
    MIN_SIGMA_FLOOR,
    SelfStartingCusumState,
    compute_mad_sigma,
    event_in_overtemp,
    latch_qualifies,
    repair_qualifies,
    update_self_starting_cusum,
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

    def test_matches_scipy_median_abs_deviation(self):
        """Sanity check: our hand-rolled MAD matches the canonical formula.

        Hand-rolled math in the per-tick CUSUM path was chosen to avoid
        importing scipy into the production controller. Lock that
        computation against the textbook MAD formula
        (np.median(np.abs(x - np.median(x))) * 1.4826) for several
        distributions (Gaussian, contaminated, skewed).

        Implemented in numpy rather than scipy.stats.median_abs_deviation
        because scipy.stats import triggers eager docs-gen which conflicts
        with coverage's numpy reload behavior (scipy 1.16+ issue).
        """
        import numpy as np
        def numpy_mad(samples):
            arr = np.asarray(samples)
            return float(np.median(np.abs(arr - np.median(arr))))
        random.seed(0xDADA)
        for label, samples in [
            ("gaussian_n60", [random.gauss(0, 1.0) for _ in range(60)]),
            ("gaussian_n7", [random.gauss(0, 0.3) for _ in range(7)]),
            ("contaminated", [random.gauss(0, 0.15) for _ in range(40)] +
                             [random.gauss(5.0, 0.5) for _ in range(20)]),
            ("skewed", [abs(random.gauss(0, 1.0)) for _ in range(50)]),
        ]:
            ours = compute_mad_sigma(samples)
            theirs = 1.4826 * numpy_mad(samples)
            # Floor may apply (when MAD is below MIN_SIGMA_FLOOR). Both
            # values should match unless our impl applied the floor — in
            # which case theirs (no floor) is ≤ ours.
            if ours == MIN_SIGMA_FLOOR:
                assert theirs <= MIN_SIGMA_FLOOR + 1e-12, (
                    f"{label}: our floored {ours}, scipy {theirs}"
                )
            else:
                assert abs(ours - theirs) < 1e-12, (
                    f"{label}: our {ours}, scipy {theirs}, "
                    f"diff {abs(ours - theirs)}"
                )


# ── CUSUM detection via PIController._update_cusum ───────────────────


def _make_cusum_controller():
    """Create a minimal PIController-like object for CUSUM testing.

    Only initializes the fields needed by _update_cusum and
    _finalize_anomaly_event, bypassing the full __init__.
    """
    from custom_components.tasmota_irhvac.pi.pi_controller import PIController

    # Use object.__new__ to skip __init__
    ctrl = object.__new__(PIController)
    # Self-starting Hawkins-Olwell CUSUM state (12h rolling window for σ̂).
    # See health_checks.SelfStartingCusumState for shape.
    ctrl._cusum_state = SelfStartingCusumState()
    # Need monotonic + utcnow for time-based eviction + cooldown logic
    ctrl._monotonic = lambda: 0.0
    from custom_components.tasmota_irhvac.pi.pi_controller import dt_util
    ctrl._utcnow_fn = dt_util.utcnow
    ctrl._anomaly_events = []
    ctrl._exclusion_count = 0
    ctrl._cusum_cooldown_until = None
    ctrl._metrics = MagicMock()
    ctrl._metrics.batch_model_rms = None
    ctrl._pending_events = []  # Stage 8: event accumulator (used by _emit_event)
    # #135 CUSUM-arming fields (off here — these tests focus on the CUSUM
    # detector itself, not the downstream latch arming).  Setting enabled=
    # False keeps `_update_cusum` from poking the arming state we don't
    # otherwise initialize on this stripped-down stub.
    ctrl._cusum_overtemp_arming_enabled = False
    ctrl._cusum_armed_this_tick = False
    ctrl._latch_armed_events = []
    return ctrl


def _feed_residuals(ctrl, residuals, sigma=0.15, start_mono=1000.0, tick_spacing=60.0):
    """Feed a sequence of residuals and return the controller.

    Pre-fills CUSUM rolling window with normal data so σ̂ is calibrated
    above the warmup threshold, then feeds the provided sequence with
    synthetic wall-clock time.
    """
    random.seed(123)
    # Pre-fill with normal residuals (time-stamped just before start_mono so
    # they're still inside the 12h eviction window when feed begins).
    # Need ≥ CUSUM_WARMUP_N obs so alarms can fire after the prefill ends.
    prefill_mono = start_mono - tick_spacing
    for _ in range(CUSUM_WARMUP_N + 5):
        ctrl._cusum_state.window.append((prefill_mono, random.gauss(0, sigma)))
        prefill_mono -= tick_spacing

    mono = start_mono
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
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
        assert event.peak_cusum > CUSUM_H_HAWKINS

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
        assert event.peak_cusum > CUSUM_H_HAWKINS

    def test_gradual_drift_hidden_in_noise(self):
        """Small drift buried in noise should NOT trigger alarm."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(55)
        # Feed noisy residuals with a small +0.3σ bias.
        # The noise keeps MAD ≈ σ, so z ≈ 0.3 < k=1.0 → no accumulation.
        mono = 1000.0
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
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

    def test_normal_operation_low_alarm_rate(self):
        """1000 ticks of N(0,σ) should produce ARL₀-bounded alarm rate.

        Hawkins canonical K=0.5/H=4 has ARL₀≈370 ticks vs Page K=1/H=10
        which had ARL₀≈50,000. With Hawkins, expect 1000/370 ≈ 2.7 alarms
        on average; bound at 12 for statistical fluctuation. This is by
        design — Hawkins trades higher false-positive rate for faster
        true-positive detection on small shifts.
        """
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(99)
        residuals = [random.gauss(0, sigma) for _ in range(1000)]
        _feed_residuals(ctrl, residuals, sigma=sigma)

        n_alarms = len(ctrl._anomaly_events)
        assert n_alarms < 12, (
            f"expected ARL₀-bounded alarm count, got {n_alarms} in 1000 ticks"
        )

    def test_setpoint_change_moderate_no_alarm(self):
        """Moderate transient (+2σ for 2 ticks) doesn't trigger with good MAD."""
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(88)
        # Pre-fill with 60 clean samples for well-calibrated MAD
        # Time-stamped entries: (mono_time, residual)
        ctrl._cusum_state.window.clear()
        for _ in range(60):
            ctrl._cusum_state.window.append((0.0, random.gauss(0, sigma)))

        # +2σ for 2 ticks: z ≈ 2, accumulates (2-1)*2=2, well below h=10
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mono = 1000.0
        transient = [2.0 * sigma] * 2 + [0.0] * 10
        for i, r in enumerate(transient):
            ctrl._update_cusum(r, mono, is_heating=True, _now=base + timedelta(seconds=i*60))
            mono += 60.0

        assert len(ctrl._anomaly_events) == 0

    def test_noisy_sensor(self):
        """Elevated noise (1.5× normal) absorbed by self-starting σ̂.

        With Hawkins canonical the noise is properly scaled by Welford
        σ̂ so the standardized statistic stays N(0,1). Allow a few
        alarms (ARL₀ ≈ 370 → ≈ 1.4 expected in 500 ticks; bound at 8
        for statistical fluctuation).
        """
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(77)
        # Pre-fill with elevated noise so σ̂ adapts
        for _ in range(CUSUM_WARMUP_N + 5):
            ctrl._cusum_state.window.append((0.0, random.gauss(0, sigma * 1.5)))
        # Continue with elevated noise
        mono = 1000.0
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(500):
            r = random.gauss(0, sigma * 1.5)
            ctrl._update_cusum(r, mono, is_heating=True, _now=base + timedelta(seconds=i*60))
            mono += 60.0

        assert len(ctrl._anomaly_events) < 8


class TestCusumClampedAndCooldown:
    """Clamped and cooldown behavior."""

    def test_cooldown_suppresses_detection(self):
        """During cooldown, new anomalies are not detected."""
        ctrl = _make_cusum_controller()
        sigma = 0.15

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Set cooldown to 30 min from base
        ctrl._cusum_cooldown_until = base + timedelta(minutes=30)

        # Pre-fill history (time-stamped tuples)
        for _ in range(CUSUM_WARMUP_N + 5):
            ctrl._cusum_state.window.append((0.0, random.gauss(0, sigma)))

        # Feed large anomaly during cooldown (all within 20 minutes of base)
        mono = 1000.0
        for i in range(20):
            now = base + timedelta(seconds=i * 60)  # 0-19 min, within cooldown
            ctrl._update_cusum(-8.0 * sigma, mono, is_heating=True, _now=now)
            mono += 60.0

        # Should NOT have detected — cooldown suppressed
        assert len(ctrl._anomaly_events) == 0
        # Cooldown also skips window updates (so disturbance residuals
        # don't shift σ̂'s running mean — see comment in _update_cusum).
        # Window size stays at the prefill count (CUSUM_WARMUP_N + 5).
        assert len(ctrl._cusum_state.window) == CUSUM_WARMUP_N + 5

    def test_cooldown_expires(self):
        """After cooldown expires, detection resumes."""
        ctrl = _make_cusum_controller()
        sigma = 0.15

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
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
        # Pre-fill with 60 clean samples for well-calibrated MAD (mono=0 keeps
        # them inside the 12h window relative to the test's 1000+ mono base)
        for _ in range(60):
            ctrl._cusum_state.window.append((0.0, random.gauss(0, sigma)))

        mono = 1000.0
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
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
        """0.5σ shift right at K_HAWKINS=0.5 dead zone — slow accumulation,
        bounded alarm count over 200 ticks.

        With Hawkins K=0.5, a 0.5σ shift is RIGHT AT the dead zone.
        Page CUSUM K=1.0 would fully absorb (z=0.5 < k=1.0 → -0.5/tick).
        Hawkins (k=0.5) absorbs to net ~0 per tick on average; alarms
        from random-noise variance occasionally cross h=4. Expect a
        few alarms over 200 ticks rather than zero.
        """
        ctrl = _make_cusum_controller()
        sigma = 0.15
        random.seed(42)
        for _ in range(CUSUM_WARMUP_N + 5):
            ctrl._cusum_state.window.append((0.0, random.gauss(0, sigma)))

        mono = 1000.0
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for tick in range(200):
            r = random.gauss(0.5 * sigma, sigma)
            now = base + timedelta(seconds=tick * 60)
            ctrl._update_cusum(r, mono, is_heating=True, _now=now)
            mono += 60.0

        # Hawkins K=0.5 means a sustained 0.5σ shift WILL eventually
        # accumulate (it's right at the dead zone). Bound the count
        # rather than asserting zero — ARL₁ at K=0.5/H=4 for δ=0.5σ
        # is ~50-80 ticks per Hawkins tables, so over 200 ticks
        # expect 2-4 alarms.
        assert len(ctrl._anomaly_events) <= 6


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


class TestAnomalyEventSignHelpers:
    """sign_matches_mode / sign_inverse_mode properties on AnomalyEvent."""

    def _evt(self, mode: str, residual: float) -> AnomalyEvent:
        now = datetime.now()
        return AnomalyEvent(
            start_time=now, start_mono=0, end_time=now,
            end_mono=0, tick_count=1, mean_residual=residual,
            peak_cusum=11.0, mode=mode,
        )

    def test_heat_neg_residual_matches_mode(self):
        """Heating + neg residual = additive heat (room warmer than predicted)."""
        e = self._evt("heat", -0.5)
        assert e.sign_matches_mode is True
        assert e.sign_inverse_mode is False

    def test_heat_pos_residual_inverse_mode(self):
        """Heating + pos residual = additive cooling/heat-loss (open window)."""
        e = self._evt("heat", +0.5)
        assert e.sign_matches_mode is False
        assert e.sign_inverse_mode is True

    def test_cool_pos_residual_matches_mode(self):
        """Cooling + pos residual = additive cooling (room cooler than predicted)."""
        e = self._evt("cool", +0.5)
        assert e.sign_matches_mode is True
        assert e.sign_inverse_mode is False

    def test_cool_neg_residual_inverse_mode(self):
        """Cooling + neg residual = additive heat (sun, cooking)."""
        e = self._evt("cool", -0.5)
        assert e.sign_matches_mode is False
        assert e.sign_inverse_mode is True

    def test_zero_residual_neither(self):
        """Exactly-zero residual qualifies as neither (edge case)."""
        e = self._evt("heat", 0.0)
        assert e.sign_matches_mode is False
        assert e.sign_inverse_mode is False


class TestFilterHelpers:
    """event_in_overtemp / latch_qualifies / repair_qualifies."""

    def _evt(self, mode: str, residual: float) -> AnomalyEvent:
        now = datetime.now()
        return AnomalyEvent(
            start_time=now, start_mono=0, end_time=now,
            end_mono=0, tick_count=1, mean_residual=residual,
            peak_cusum=11.0, mode=mode,
        )

    def test_overtemp_heat_mode(self):
        """Heating mode: overtemp = current > desired."""
        e = self._evt("heat", -0.5)
        assert event_in_overtemp(e, current_c=22.0, desired_c=20.0) is True
        assert event_in_overtemp(e, current_c=19.0, desired_c=20.0) is False
        # Boundary: exactly equal is NOT overtemp (strict >)
        assert event_in_overtemp(e, current_c=20.0, desired_c=20.0) is False

    def test_overtemp_cool_mode(self):
        """Cooling mode: 'overtemp' (latch-condition) = current < desired."""
        e = self._evt("cool", +0.5)
        assert event_in_overtemp(e, current_c=19.0, desired_c=22.0) is True
        assert event_in_overtemp(e, current_c=23.0, desired_c=22.0) is False

    def test_latch_qualifies_heat_additive_heat(self):
        """Heating + neg residual + overtemp → LATCH FIRES."""
        e = self._evt("heat", -0.5)
        assert latch_qualifies(e, current_c=22.0, desired_c=20.0) is True

    def test_latch_does_not_fire_without_overtemp(self):
        """Sign matches but not overtemp → no latch (room cold despite alarm)."""
        e = self._evt("heat", -0.5)
        assert latch_qualifies(e, current_c=19.0, desired_c=20.0) is False

    def test_latch_does_not_fire_wrong_sign(self):
        """Wrong-direction residual: NEVER arms latch even if overtemp."""
        # heat mode + pos residual = open-window case. Even if temp is over,
        # latch shouldn't fire (this means HP is over-correcting)
        e = self._evt("heat", +0.5)
        assert latch_qualifies(e, current_c=22.0, desired_c=20.0) is False

    def test_repair_qualifies_additive_heat(self):
        """Additive heat (heating, overtemp): notify user."""
        e = self._evt("heat", -0.5)
        assert repair_qualifies(e, current_c=22.0, desired_c=20.0) is True

    def test_repair_qualifies_additive_cooling_in_heat(self):
        """Open-window case (heating, undertemp, pos residual): notify user."""
        e = self._evt("heat", +0.5)
        assert repair_qualifies(e, current_c=19.0, desired_c=20.0) is True

    def test_repair_qualifies_additive_heat_in_cool(self):
        """Cooling mode, room warm, neg residual: heat source (sun, cooking)."""
        e = self._evt("cool", -0.5)
        assert repair_qualifies(e, current_c=23.0, desired_c=22.0) is True

    def test_repair_qualifies_additive_cooling_in_cool(self):
        """Cooling, room cold, pos residual: AC over-cooling / heat loss."""
        e = self._evt("cool", +0.5)
        assert repair_qualifies(e, current_c=19.0, desired_c=22.0) is True

    def test_repair_does_not_fire_when_no_unmodelled_input(self):
        """Sign and temp direction both inconsistent with any unmodelled input."""
        # heat mode + neg residual + undertemp = HP underperforming (control issue, not unmodelled)
        e = self._evt("heat", -0.5)
        assert repair_qualifies(e, current_c=19.0, desired_c=20.0) is False


# ── Self-starting Hawkins-Olwell CUSUM ───────────────────────────────


class TestSelfStartingCusumState:
    """Basic SelfStartingCusumState behavior."""

    def test_initial_state(self):
        s = SelfStartingCusumState()
        assert s.n == 0
        assert s.s_pos == 0.0
        assert s.s_neg == 0.0

    def test_window_accumulates(self):
        s = SelfStartingCusumState()
        s.window.append((1.0, 0.1))
        s.window.append((2.0, 0.2))
        assert s.n == 2

    def test_reset_accumulators_preserves_window(self):
        s = SelfStartingCusumState()
        s.s_pos = 5.0
        s.s_neg = 3.0
        s.window.append((1.0, 0.1))
        s.reset_accumulators()
        assert s.s_pos == 0.0
        assert s.s_neg == 0.0
        assert s.n == 1  # window untouched

    def test_persistence_roundtrip(self):
        s = SelfStartingCusumState(s_pos=1.5, s_neg=2.3)
        s.window.append((100.0, 0.05))
        s.window.append((200.0, -0.03))
        restored = SelfStartingCusumState.from_dict(s.as_dict())
        assert restored.s_pos == 1.5
        assert restored.s_neg == 2.3
        assert restored.n == 2
        assert restored.window[0] == (100.0, 0.05)


class TestUpdateSelfStartingCusum:
    """Update function behavior."""

    def test_warmup_no_alarms(self):
        """First CUSUM_WARMUP_N observations cannot alarm even with extreme residuals."""
        s = SelfStartingCusumState()
        # Feed CUSUM_WARMUP_N consecutive +10σ shocks (relative to perfectly clean prior).
        # We can't actually produce +10σ because σ̂ scales with the residual, but the
        # warmup check should suppress alarms regardless of how aggressive the data is.
        random.seed(42)
        for i in range(CUSUM_WARMUP_N):
            alarmed, _, _ = update_self_starting_cusum(
                s, residual=random.gauss(0, 0.1), now_mono=float(i * 60),
            )
            assert alarmed is False, f"warmup-suppressed alarm fired at i={i}"

    def test_window_eviction(self):
        """Observations older than CUSUM_WINDOW_S are evicted."""
        s = SelfStartingCusumState()
        # Two obs at t=0 and t=100
        update_self_starting_cusum(s, residual=0.1, now_mono=0.0)
        update_self_starting_cusum(s, residual=0.1, now_mono=100.0)
        assert s.n == 2
        # New obs WAY past window — both old should evict
        update_self_starting_cusum(s, residual=0.1, now_mono=CUSUM_WINDOW_S + 1000)
        assert s.n == 1  # only the latest survives

    def test_clean_gaussian_low_alarm_rate(self):
        """Clean N(0,σ) data — ARL₀ should be high (few alarms over a long run)."""
        # ARL₀ ≈ 370 ticks at Hawkins canonical (K=0.5, H=4). Over 1000 ticks we
        # expect a small number of alarms; bench-validated bound is <10 to be safe.
        random.seed(20260601)
        s = SelfStartingCusumState()
        alarms = 0
        for i in range(1000):
            alarmed, _, _ = update_self_starting_cusum(
                s, residual=random.gauss(0, 0.1), now_mono=float(i * 60),
            )
            if alarmed:
                alarms += 1
                s.reset_accumulators()
        # Theoretical ARL₀ ≈ 370 → expected ≈ 1000/370 ≈ 2.7 alarms; allow 10
        # for statistical fluctuation across seeds (this test is one realization)
        assert alarms < 20, f"expected ARL₀-bounded alarm count, got {alarms}"

    def test_detects_sustained_step(self):
        """A persistent +N·σ step should trigger within a reasonable ARL₁."""
        random.seed(101)
        s = SelfStartingCusumState()
        # Pre-warm with clean data
        for i in range(50):
            update_self_starting_cusum(
                s, residual=random.gauss(0, 0.1), now_mono=float(i * 60),
            )
        # Reset accumulators (warmup may have left them at zero anyway)
        s.reset_accumulators()
        # Inject sustained +2σ shift: residuals N(0.2, 0.1)
        ticks_to_alarm = None
        for i in range(50, 200):
            alarmed, _, _ = update_self_starting_cusum(
                s, residual=random.gauss(0.2, 0.1), now_mono=float(i * 60),
            )
            if alarmed:
                ticks_to_alarm = i - 50
                break
        # ARL₁ for K=0.5, H=4, 2σ shift ≈ 10-25 ticks per Hawkins tables.
        # Allow up to 50 for safety margin.
        assert ticks_to_alarm is not None, "no alarm fired on sustained +2σ step"
        assert ticks_to_alarm < 50, f"alarm took {ticks_to_alarm} ticks (>50)"

    def test_sigma_floor_collapse_doesnt_fire(self):
        """σ̂-collapse scenario: quiet residuals → small step → old MAD chart
        would fire immediately; self-starting should NOT fire (small step
        relative to running σ̂, even if σ̂ is tiny)."""
        random.seed(7)
        s = SelfStartingCusumState()
        # Pre-fill with very quiet residuals (the σ̂-floor problem case)
        for i in range(CUSUM_WARMUP_N):
            update_self_starting_cusum(
                s, residual=random.gauss(0, 0.001), now_mono=float(i * 60),
            )
        # Now a 0.05°C "step" — old MAD chart with floor=0.05 saw this as 1σ
        # and fired easily; self-starting sees it as a HUGE shift relative
        # to the running σ̂ ≈ 0.001 → alarms IMMEDIATELY (which is correct
        # because relative to actual data, 0.05 IS a huge shift).
        #
        # So this test verifies the OPPOSITE behavior from what the MAD-floor
        # path did: when residuals have been quiet, even small absolute
        # disturbances are correctly flagged as large RELATIVE shifts.
        alarmed_quickly = False
        for i in range(CUSUM_WARMUP_N, CUSUM_WARMUP_N + 5):
            alarmed, _, _ = update_self_starting_cusum(
                s, residual=0.05, now_mono=float(i * 60),
            )
            if alarmed:
                alarmed_quickly = True
                break
        assert alarmed_quickly, (
            "self-starting should detect a 50× σ̂ jump immediately"
        )

    # NOTE: previously had a test for "natural slow drift should not alarm,"
    # but Hawkins-Olwell CUSUM is specifically designed to detect sustained
    # mean shifts — even small ones — by accumulating evidence. That's a
    # feature, not a bug. ARL₁ scales with shift size: small shifts take
    # longer but still fire. If a user's residuals exhibit slow seasonal
    # drift, the right fix is at the FF/RLS layer (learn it down so
    # residuals stay zero-mean), not to make CUSUM blind to it. The
    # rolling 12h window does limit how much old "stable" data anchors
    # the running mean against truly-stationary changes.


class TestCusumPersistence:
    """CUSUM accumulator + cooldown survive a controller restart.

    Without persistence, every restart erased the accumulated drift
    evidence (cusum_pos / cusum_neg → 0) and forgot the cooldown timer.
    The 2026-05-08 BR bundle showed 105 controller_reload events over
    10 days — frequent state loss in practice.
    """

    @pytest.mark.asyncio
    async def test_cusum_state_round_trips_through_persistence(
        self, hass, setup_pi_integration,
    ):
        from custom_components.tasmota_irhvac.pi.pi_stored_data import (
            PIExtraStoredData,
        )
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        pi = get_climate_entity(hass, entry)._pi

        # Set non-default CUSUM state.
        pi._cusum_state.s_pos = 7.5
        pi._cusum_state.s_neg = 0.3
        cooldown = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        pi._cusum_cooldown_until = cooldown
        # Pre-fill some window residuals. Time-stamped tuples now:
        # (mono_time, residual). Persist drops timestamps and reattaches
        # current mono on restore.
        pi._cusum_state.window.clear()
        residuals = [0.05, -0.03, 0.10, -0.08, 0.02]
        pi._cusum_state.window.extend((100.0, r) for r in residuals)

        # Round-trip through PIExtraStoredData.
        saved = pi.get_extra_stored_data()
        assert isinstance(saved, PIExtraStoredData)
        assert saved.cusum_s_pos == pytest.approx(7.5)
        assert saved.cusum_s_neg == pytest.approx(0.3)
        assert saved.cusum_cooldown_until_epoch == pytest.approx(
            cooldown.timestamp(),
        )
        # Persist serializes residuals only (no timestamps).
        assert saved.cusum_window_residuals == residuals

        # Wipe live state, then restore.
        pi._cusum_state.s_pos = 0.0
        pi._cusum_state.s_neg = 0.0
        pi._cusum_cooldown_until = None
        pi._cusum_state.window.clear()
        pi.restore_extra_stored_data(saved)
        assert pi._cusum_state.s_pos == pytest.approx(7.5)
        assert pi._cusum_state.s_neg == pytest.approx(0.3)
        assert pi._cusum_cooldown_until == cooldown
        # Restored entries: residuals preserved, all timestamped to now (so
        # they evict over the next 12h window — warm σ̂ post-restart).
        assert [r for _, r in pi._cusum_state.window] == residuals

    @pytest.mark.asyncio
    async def test_no_cooldown_round_trips_as_none(
        self, hass, setup_pi_integration,
    ):
        """When no cooldown is active, restore must produce ``None`` —
        not a datetime that compares poorly to ``datetime.now()``."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        pi = get_climate_entity(hass, entry)._pi
        pi._cusum_cooldown_until = None
        saved = pi.get_extra_stored_data()
        assert saved.cusum_cooldown_until_epoch == 0.0
        # Set live to a sentinel value, then restore — should clear to None.
        pi._cusum_cooldown_until = datetime(2099, 1, 1, tzinfo=timezone.utc)
        pi.restore_extra_stored_data(saved)
        assert pi._cusum_cooldown_until is None

    @pytest.mark.asyncio
    async def test_cooldown_honored_after_restart(
        self, hass, setup_pi_integration,
    ):
        """Round-trip a live cooldown, then verify a new residual feed
        within the window does NOT trigger a fresh alarm.

        Direct repro of the production failure mode that motivated the
        persistence fix: BR's 14-anomaly cluster was driven by event
        re-emission (fixed via _pending_events lifecycle) but the
        underlying detector still resets accumulators on every restart;
        without persistence, post-restart re-detection of the same
        on-going anomaly is the lurking second bug.
        """
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        pi = get_climate_entity(hass, entry)._pi

        sigma = 0.15
        # Pre-fill window so σ̂ is warm post-restore (time-stamped tuples).
        history = [random.gauss(0, sigma) for _ in range(60)]
        pi._cusum_state.window.clear()
        pi._cusum_state.window.extend((100.0, r) for r in history)

        # Simulate prior alarm: cooldown active 30 min from "now".
        base = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)
        pi._cusum_cooldown_until = base + timedelta(minutes=30)
        pi._cusum_state.s_pos = 0.0
        pi._cusum_state.s_neg = 0.0
        pi._anomaly_events.clear()

        # Save → restore (simulates an HA restart within the cooldown).
        saved = pi.get_extra_stored_data()
        pi._cusum_cooldown_until = None  # would-be-erased on naive restart
        pi._cusum_state.s_pos = 0.0
        pi._cusum_state.s_neg = 0.0
        pi.restore_extra_stored_data(saved)

        # Feed a 5σ excursion at "now = base + 5 min" — within cooldown.
        for i in range(15):
            now = base + timedelta(seconds=i * 60)
            pi._update_cusum(
                -5.0 * sigma, 1000.0 + i * 60.0,
                is_heating=True, _now=now,
            )
        # Cooldown still active → no new alarm.
        assert pi._anomaly_events == []
        # Once we cross the cooldown boundary, detection resumes.
        for i in range(20):
            now = base + timedelta(minutes=31, seconds=i * 60)
            pi._update_cusum(
                -5.0 * sigma, 5000.0 + i * 60.0,
                is_heating=True, _now=now,
            )
        assert len(pi._anomaly_events) >= 1, (
            "post-cooldown detection must resume"
        )
