"""Unit tests for BoundaryEstimator — piecewise linear (hockey stick) fit."""

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


def _make_ramp_data(
    breakpoint: float = 0.0,
    slope: float = -0.005,
    baseline: float = 0.0,
    noise_sigma: float = 0.002,
    n_per_side: int = 60,
    delta_range: tuple[float, float] = (-4.0, 4.0),
    seed: int = 42,
) -> list[tuple[float, float]]:
    """Generate (delta, residual) pairs with a ramp-to-flat transition.

    HP-on side (delta < breakpoint): residual = slope × (delta - bp) + baseline
    HP-off side (delta >= breakpoint): residual = baseline
    """
    rng = np.random.default_rng(seed)
    data = []
    deltas = np.linspace(delta_range[0], delta_range[1], n_per_side * 2)
    for d in deltas:
        if d < breakpoint:
            rate = slope * (d - breakpoint) + baseline
        else:
            rate = baseline
        noisy = rate + rng.normal(0, noise_sigma)
        data.append((float(d), float(noisy)))
    return data


# ── Test: Breakpoint detection ──────────────────────────────────────────


class TestBreakpointDetection:
    """Core functionality: detect a known breakpoint from ramp-to-flat data."""

    def test_finds_breakpoint_at_zero(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        data = _make_ramp_data(breakpoint=0.0, slope=-0.005)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.estimated_breakpoint is not None
        assert abs(result.estimated_breakpoint - 0.0) < 0.5

    def test_finds_breakpoint_at_positive_offset(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        data = _make_ramp_data(breakpoint=1.5, slope=-0.005)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 1.5) < 0.5

    def test_finds_breakpoint_at_negative_offset(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        data = _make_ramp_data(breakpoint=-1.0, slope=-0.005)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - (-1.0)) < 0.5

    def test_slope_is_negative(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        data = _make_ramp_data(breakpoint=0.0, slope=-0.008)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.slope is not None
        assert result.slope < 0

    def test_breakpoint_std_err_small(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        data = _make_ramp_data(breakpoint=0.0, slope=-0.008, noise_sigma=0.001)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.breakpoint_std_err is not None
        assert result.breakpoint_std_err < 0.5

    def test_baseline_near_zero_with_good_residualization(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        data = _make_ramp_data(breakpoint=0.0, baseline=0.0, noise_sigma=0.001)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.baseline) < 0.005


# ── Test: Outlier robustness ────────────────────────────────────────────


class TestOutlierRobustness:
    """curve_fit with ramp model is reasonably robust to outliers."""

    def test_10pct_negative_outliers(self):
        """10% of observations have large negative residuals (open window)."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        rng = np.random.default_rng(99)
        data = _make_ramp_data(breakpoint=0.5, slope=-0.005)
        for i, (d, r) in enumerate(data):
            if rng.random() < 0.10:
                r = -0.1  # open window: sharp cooling
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 0.5) < 1.5


# ── Test: Bound updates ────────────────────────────────────────────────


class TestBoundUpdates:
    """Verify bound movement logic: shrink, widen, step cap, floor."""

    def test_shrinks_toward_breakpoint(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=5.0,
        )
        data = _make_ramp_data(breakpoint=0.0, slope=-0.005)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.new_cal_min > -2.0
        assert result.new_cal_max < 2.0

    def test_widens_when_breakpoint_outside_band(self):
        """Start with tight band (0, 0), true breakpoint at +1.5."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=5.0,
        )
        data = _make_ramp_data(breakpoint=1.5, slope=-0.005)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(0.0, 0.0)
        assert result.confident
        assert result.new_cal_max > 0.0

    def test_max_step_cap(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=0.2,
        )
        data = _make_ramp_data(breakpoint=0.0, slope=-0.005)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert result.new_cal_min <= -2.0 + 0.2 + 0.01
        assert result.new_cal_max >= 2.0 - 0.2 - 0.01

    def test_min_band_floor(self):
        """Band should never collapse below min_band_width."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.1, min_band_width=0.8,
            max_step_per_update=5.0,
        )
        data = _make_ramp_data(breakpoint=0.0, slope=-0.008, noise_sigma=0.001)
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
        for i in range(10):
            est.add_evidence(float(i), 0.01, float(i))

        est.estimate_boundary(-2.0, 2.0)
        assert est.stall_count == 1
        est.estimate_boundary(-2.0, 2.0)
        assert est.stall_count == 2

    def test_stall_increments_on_no_ramp(self):
        """Flat signal → positive slope or poor fit → not confident."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, stall_threshold=2,
        )
        rng = np.random.default_rng(42)
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
            min_observations=20, min_per_side=5, stall_threshold=3,
        )
        # Stall first
        for i in range(10):
            est.add_evidence(float(i), 0.0, float(i))
        est.estimate_boundary(-2.0, 2.0)
        assert est.stall_count == 1

        # Now add real ramp data
        data = _make_ramp_data(breakpoint=0.0, slope=-0.005)
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
    """Raw room_rate should still find approximate breakpoint if ramp is strong."""

    def test_raw_rate_with_offset_baseline(self):
        """Simulate raw room_rate with outdoor cooling as baseline offset."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        rng = np.random.default_rng(42)
        hp_slope = -0.005
        breakpoint = 0.5
        outdoor_baseline = -0.01  # constant offset from outdoor cooling

        for i in range(120):
            d = rng.uniform(-4, 4)
            if d < breakpoint:
                rate = hp_slope * (d - breakpoint) + outdoor_baseline
            else:
                rate = outdoor_baseline
            rate += rng.normal(0, 0.002)
            est.add_evidence(d, rate, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        # Baseline absorbs outdoor_baseline, breakpoint still accurate
        assert abs(result.estimated_breakpoint - 0.5) < 1.0
        assert abs(result.baseline - outdoor_baseline) < 0.01


# ── Test: Delta diversity ──────────────────────────────────────────────


class TestDeltaDiversity:
    """Buffer should enforce diversity across delta bins."""

    def test_bin_cap_prevents_redundancy(self):
        est = BoundaryEstimator(buffer_max_size=200)
        for i in range(100):
            est.add_evidence(1.0, 0.01, float(i))
        assert len(est._buffer) <= 25  # capped at _MAX_PER_BIN (20) + slack

    def test_diverse_deltas_retained(self):
        est = BoundaryEstimator(buffer_max_size=200)
        for i in range(200):
            d = (i % 10) * 0.5 - 2.0
            est.add_evidence(d, 0.01, float(i))
        assert len(est._buffer) == 200


# ── Test: Persistence ──────────────────────────────────────────────────


class TestPersistence:
    """as_dict()/restore() round-trip."""

    def test_round_trip(self):
        est = BoundaryEstimator()
        for i in range(30):
            est.add_evidence(float(i) * 0.1, 0.01 * i, float(i))
        est._stall_count = 2
        est._updates_applied = 5

        state = est.as_dict()
        assert len(state["buffer"]) == 30
        assert state["stall_count"] == 2
        assert state["updates_applied"] == 5

        est2 = BoundaryEstimator()
        est2.restore(state)
        assert len(est2._buffer) == 30
        assert est2._stall_count == 2
        assert est2._updates_applied == 5

    def test_empty_restore(self):
        est = BoundaryEstimator()
        est.restore({})
        assert len(est._buffer) == 0
        assert est._stall_count == 0


# ── Test: updates_applied counter ──────────────────────────────────────


class TestUpdatesApplied:

    def test_increments_on_confident_update(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        assert est.updates_applied == 0

        data = _make_ramp_data(breakpoint=0.0, slope=-0.005)
        for i, (d, r) in enumerate(data):
            est.add_evidence(d, r, float(i))

        result = est.estimate_boundary(-2.0, 2.0)
        assert result.confident
        assert est.updates_applied == 1

    def test_does_not_increment_on_stall(self):
        est = BoundaryEstimator(min_observations=50)
        for i in range(10):
            est.add_evidence(float(i), 0.0, float(i))

        est.estimate_boundary(-2.0, 2.0)
        assert est.updates_applied == 0
