"""Unit tests for BoundaryEstimator — pure computation, no HA deps."""

from __future__ import annotations

import math

import numpy as np
import pytest

from custom_components.tasmota_irhvac.pi.boundary_estimator import (
    BoundaryEstimator,
    BoundaryEstimateResult,
    BoundaryEvidence,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_step_data(
    breakpoint: float = 0.0,
    hp_on_rate: float = 0.03,
    hp_off_rate: float = 0.0,
    noise_sigma: float = 0.005,
    n_per_side: int = 60,
    delta_range: tuple[float, float] = (-4.0, 4.0),
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Generate (delta, residual) pairs with a step at breakpoint."""
    rng = np.random.default_rng(seed)
    data = []
    deltas = np.linspace(delta_range[0], delta_range[1], n_per_side * 2)
    for d in deltas:
        rate = hp_on_rate if d < breakpoint else hp_off_rate
        noisy = rate + rng.normal(0, noise_sigma)
        data.append((float(d), float(noisy)))
    return data


# ── Test: Synthetic step function ───────────────────────────────────────


class TestBreakpointDetection:
    """Core functionality: detect a known breakpoint."""

    def test_finds_breakpoint_at_zero(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        data = _make_step_data(breakpoint=0.0, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.estimated_breakpoint is not None
        assert abs(result.estimated_breakpoint - 0.0) < 0.5

    def test_finds_breakpoint_at_positive_offset(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        data = _make_step_data(breakpoint=1.5, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 1.5) < 0.5

    def test_finds_breakpoint_at_negative_offset(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        data = _make_step_data(breakpoint=-1.0, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - (-1.0)) < 0.5

    def test_gap_magnitude_positive(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        data = _make_step_data(breakpoint=0.0, hp_on_rate=0.04, hp_off_rate=0.0)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.gap_magnitude > 0.02  # should be ~0.04

    def test_p_value_significant(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        data = _make_step_data(breakpoint=0.0, hp_on_rate=0.04, noise_sigma=0.003)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.p_value is not None
        assert result.p_value < 0.05


# ── Test: Outlier robustness ────────────────────────────────────────────


class TestOutlierRobustness:
    """Median-based estimation should be robust to open-window outliers."""

    def test_10pct_negative_outliers(self):
        """10% of observations have large negative residuals (open window)."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        rng = np.random.default_rng(99)
        data = _make_step_data(breakpoint=0.5, n_per_side=60, seed=42)
        # Inject outliers: 10% get large negative residuals
        for i, (d, r) in enumerate(data):
            if rng.random() < 0.10:
                r = -0.1  # open window: sharp cooling
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 0.5) < 0.8

    def test_5pct_positive_outliers(self):
        """5% solar spikes (large positive residuals when HP is off)."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        rng = np.random.default_rng(77)
        data = _make_step_data(breakpoint=0.0, n_per_side=60, seed=42)
        for i, (d, r) in enumerate(data):
            if d > 0.0 and rng.random() < 0.05:  # solar spike on HP-off side
                r = 0.05
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 0.0) < 0.8


# ── Test: Bound updates ────────────────────────────────────────────────


class TestBoundUpdates:
    """Verify bound movement logic: shrink, widen, step cap, floor."""

    def test_shrinks_toward_breakpoint(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=5.0,  # large step
            rng_seed=42,
        )
        data = _make_step_data(breakpoint=0.0, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        # Band should shrink from [-2, 2] toward [bp-0.3, bp+0.3]
        assert result.new_cal_min > -2.0
        assert result.new_cal_max < 2.0

    def test_widens_when_breakpoint_outside_band(self):
        """Start with tight band (0, 0), true breakpoint at +1.5."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=5.0,
            rng_seed=42,
        )
        data = _make_step_data(breakpoint=1.5, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(0.0, 0.0)
        assert result.confident
        # Band should widen: cal_max should increase above 0
        assert result.new_cal_max > 0.0

    def test_max_step_cap(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=0.2,
            rng_seed=42,
        )
        data = _make_step_data(breakpoint=0.0, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        # Movement should be at most 0.2 from -2.0 and 2.0
        assert result.new_cal_min <= -2.0 + 0.2 + 0.01  # small float tolerance
        assert result.new_cal_max >= 2.0 - 0.2 - 0.01

    def test_min_band_floor(self):
        """Band should never collapse below min_band_width."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.1, min_band_width=0.8,
            max_step_per_update=5.0,
            rng_seed=42,
        )
        data = _make_step_data(breakpoint=0.0, n_per_side=60, noise_sigma=0.001)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-0.5, 0.5)
        assert result.confident
        assert result.new_cal_max - result.new_cal_min >= 0.8 - 0.001


# ── Test: Stall detection ──────────────────────────────────────────────


class TestStallDetection:
    """Verify stall counter and probe trigger."""

    def test_stall_increments_on_insufficient_data(self):
        est = BoundaryEstimator(min_observations=50, stall_threshold=3)
        # Only 10 observations — not enough
        for i in range(10):
            est.add_evidence(float(i), 0.01, float(i))

        est.estimate_boundary(-2.0, 2.0)
        assert est.stall_count == 1

        est.estimate_boundary(-2.0, 2.0)
        assert est.stall_count == 2

    def test_stall_increments_on_no_breakpoint(self):
        """Flat signal with no step → no confident estimate."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            min_gap_threshold=0.003, stall_threshold=2,
            rng_seed=42,
        )
        rng = np.random.default_rng(42)
        # All residuals ~0 (no HP contribution anywhere)
        for i in range(100):
            d = rng.uniform(-3, 3)
            r = rng.normal(0, 0.001)
            est.add_evidence(d, r, float(i))

        est.estimate_boundary(-2.0, 2.0)
        est.estimate_boundary(-2.0, 2.0)
        assert est.stall_count >= 2
        assert est.should_trigger_probe

    def test_confident_estimate_resets_stall(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            stall_threshold=3, rng_seed=42,
        )
        # Stall first
        for i in range(10):
            est.add_evidence(float(i), 0.0, float(i))
        est.estimate_boundary(-2.0, 2.0)
        assert est.stall_count == 1

        # Now add real data with a step
        data = _make_step_data(breakpoint=0.0, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i + 100))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert est.stall_count == 0

    def test_reset_stall_manual(self):
        est = BoundaryEstimator(stall_threshold=3)
        est._stall_count = 5
        est.reset_stall()
        assert est.stall_count == 0
        assert not est.should_trigger_probe

    def test_should_trigger_probe_threshold(self):
        est = BoundaryEstimator(stall_threshold=3)
        assert not est.should_trigger_probe
        est._stall_count = 2
        assert not est.should_trigger_probe
        est._stall_count = 3
        assert est.should_trigger_probe


# ── Test: Bootstrap (no greybox) ───────────────────────────────────────


class TestBootstrap:
    """Raw room_rate (no residualization) should still find approximate breakpoint."""

    def test_raw_rate_with_outdoor_effect(self):
        """Simulate raw room_rate with outdoor cooling effect + HP step."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            min_gap_threshold=0.002, rng_seed=42,
        )
        rng = np.random.default_rng(42)
        # outdoor_effect is constant (not subtracted in bootstrap mode)
        outdoor_effect = -0.01  # °C/min cooling from outdoor
        hp_effect = 0.03  # °C/min from HP
        breakpoint = 0.5

        for i in range(120):
            d = rng.uniform(-4, 4)
            hp = hp_effect if d < breakpoint else 0.0
            raw_rate = outdoor_effect + hp + rng.normal(0, 0.005)
            est.add_evidence(d, raw_rate, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        # Less accurate due to outdoor bias, but should still detect step
        assert abs(result.estimated_breakpoint - 0.5) < 1.0


# ── Test: Delta diversity ──────────────────────────────────────────────


class TestDeltaDiversity:
    """Buffer should enforce diversity across delta bins."""

    def test_bin_cap_prevents_redundancy(self):
        est = BoundaryEstimator(buffer_max_size=200)
        # Add 100 observations at the same delta
        for i in range(100):
            est.add_evidence(1.0, 0.01, float(i))
        # Should be capped at _MAX_PER_BIN (20)
        assert len(est._buffer) <= 25  # some slack for bin edge

    def test_diverse_deltas_retained(self):
        est = BoundaryEstimator(buffer_max_size=200)
        # Add observations across many bins
        for i in range(200):
            d = (i % 10) * 0.5 - 2.0  # 10 bins
            est.add_evidence(d, 0.01, float(i))
        # Should retain all (within buffer max)
        assert len(est._buffer) == 200


# ── Test: Persistence ──────────────────────────────────────────────────


class TestPersistence:
    """as_dict()/restore() round-trip."""

    def test_round_trip(self):
        est = BoundaryEstimator(rng_seed=42)
        for i in range(30):
            est.add_evidence(float(i) * 0.1, 0.01 * i, float(i))
        est._stall_count = 2
        est._updates_applied = 5

        state = est.as_dict()
        assert len(state["buffer"]) == 30
        assert state["stall_count"] == 2
        assert state["updates_applied"] == 5

        est2 = BoundaryEstimator(rng_seed=42)
        est2.restore(state)
        assert len(est2._buffer) == 30
        assert est2._stall_count == 2
        assert est2._updates_applied == 5
        assert abs(est2._buffer[0].delta - 0.0) < 0.001
        assert abs(est2._buffer[0].residual_rate - 0.0) < 0.001

    def test_empty_restore(self):
        est = BoundaryEstimator()
        est.restore({})
        assert len(est._buffer) == 0
        assert est._stall_count == 0


# ── Test: Permutation test calibration ─────────────────────────────────


class TestPermutationTest:
    """Verify p-value is well-calibrated under the null."""

    def test_null_distribution_not_significant(self):
        """Pure noise (no step) should yield p > 0.05 most of the time."""
        rng = np.random.default_rng(42)
        significant_count = 0
        n_trials = 20

        for trial in range(n_trials):
            est = BoundaryEstimator(
                min_observations=20, min_per_side=5,
                min_gap_threshold=0.001,  # low threshold to test p-value
                permutation_n=100,
                rng_seed=trial,
            )
            for i in range(80):
                d = rng.uniform(-3, 3)
                r = rng.normal(0, 0.005)
                est.add_evidence(d, r, float(i))

            result = est.estimate_boundary(-2.0, 2.0)
            if result.confident:
                significant_count += 1

        # Under the null, expect <20% significant (alpha=0.05 + noise)
        assert significant_count < n_trials * 0.3


# ── Test: updates_applied counter ──────────────────────────────────────


class TestUpdatesApplied:

    def test_increments_on_confident_update(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, rng_seed=42,
        )
        assert est.updates_applied == 0

        data = _make_step_data(breakpoint=0.0, n_per_side=60)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert est.updates_applied == 1

    def test_does_not_increment_on_stall(self):
        est = BoundaryEstimator(min_observations=50, rng_seed=42)
        for i in range(10):
            est.add_evidence(float(i), 0.0, float(i))

        est.estimate_boundary(-2.0, 2.0)
        assert est.updates_applied == 0
