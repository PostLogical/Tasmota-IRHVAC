"""Unit tests for BoundaryEstimator — greybox profile sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from custom_components.tasmota_irhvac.pi.boundary_estimator import (
    BoundaryEstimator,
    BoundaryEstimateResult,
)


# ── Fake Observation ────────────────────────────────────────────────────


@dataclass
class FakeObs:
    """Minimal observation for boundary estimator tests."""

    current_c: float
    hp_setpoint: float | None
    desired_c: float
    outdoor_temp_c: float | None
    room_rate: float
    raw_readings: dict[str, Any]


def _make_observations(
    breakpoint: float = 0.0,
    k_c: float = 0.005,
    ua_c: float = 0.001,
    c0: float = 0.0,
    alpha_c: float = 0.002,
    noise_sigma: float = 0.002,
    n: int = 200,
    delta_range: tuple[float, float] = (-5.0, 4.0),
    desired: float = 20.5,
    outdoor_base: float = 10.0,
    seed: int = 42,
) -> list[FakeObs]:
    """Generate synthetic observations with a known HP boundary.

    room_rate = c0 + ua_c × (outdoor - room) + k_c × hp_offset × I(delta < bp) + alpha_c × solar + noise
    """
    rng = np.random.default_rng(seed)
    obs = []
    for i in range(n):
        # Vary delta by varying the setpoint
        delta = rng.uniform(*delta_range)
        room = desired + rng.normal(0, 0.3)
        setpoint = room - delta  # delta = room - setpoint
        outdoor = outdoor_base + rng.normal(0, 3)
        solar = max(0, rng.normal(0.3, 0.2))

        outdoor_delta = outdoor - room
        hp_on = delta < breakpoint
        hp_offset = (setpoint - room) if hp_on else 0.0

        rate = c0 + ua_c * outdoor_delta + k_c * hp_offset + alpha_c * solar
        rate += rng.normal(0, noise_sigma)

        obs.append(FakeObs(
            current_c=room,
            hp_setpoint=setpoint,
            desired_c=desired,
            outdoor_temp_c=outdoor,
            room_rate=rate,
            raw_readings={"sensor.solar": solar},
        ))
    return obs


# ── Test: Breakpoint detection ──────────────────────────────────────────


class TestBreakpointDetection:
    """Core: detect a known breakpoint from synthetic data."""

    def test_finds_breakpoint_at_zero(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 0.0) < 0.5

    def test_finds_breakpoint_at_positive(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=1.5)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 1.5) < 0.5

    def test_finds_breakpoint_at_negative(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=-1.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - (-1.0)) < 0.5

    def test_k_c_positive(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0, k_c=0.008)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert result.slope_k_c is not None
        assert result.slope_k_c > 0

    def test_no_solar_model_input(self):
        """Works without solar as a model input."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0, alpha_c=0.0)
        # No solar in model inputs
        result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 0.0) < 0.5


# ── Test: Bound updates ────────────────────────────────────────────────


class TestBoundUpdates:
    """Verify bound movement logic."""

    def test_shrinks_toward_breakpoint(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=5.0,
        )
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert result.new_cal_min > -2.0
        assert result.new_cal_max < 2.0

    def test_widens_when_breakpoint_outside_band(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=5.0,
        )
        obs = _make_observations(breakpoint=1.5)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, 0.0, 0.0)
        assert result.confident
        assert result.new_cal_max > 0.0

    def test_max_step_cap(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=0.2,
        )
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert result.new_cal_min <= -2.0 + 0.2 + 0.01
        assert result.new_cal_max >= 2.0 - 0.2 - 0.01

    def test_min_band_floor(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.1, min_band_width=0.8,
            max_step_per_update=5.0,
        )
        obs = _make_observations(breakpoint=0.0, noise_sigma=0.001)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -0.5, 0.5)
        assert result.confident
        assert result.new_cal_max - result.new_cal_min >= 0.8 - 0.001


# ── Test: Stall detection ──────────────────────────────────────────────


class TestStallDetection:

    def test_stall_on_insufficient_data(self):
        est = BoundaryEstimator(min_observations=50, stall_threshold=3)
        obs = _make_observations(n=10)
        mi = []
        est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert est.stall_count == 1
        est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert est.stall_count == 2

    def test_stall_on_flat_signal(self):
        """No HP contribution → no breakpoint → stall."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, stall_threshold=2,
        )
        # All observations identical (no HP effect)
        rng = np.random.default_rng(42)
        obs = []
        for i in range(100):
            obs.append(FakeObs(
                current_c=20.5 + rng.normal(0, 0.1),
                hp_setpoint=22.0,
                desired_c=20.5,
                outdoor_temp_c=10.0 + rng.normal(0, 1),
                room_rate=rng.normal(0, 0.001),
                raw_readings={},
            ))
        mi = []
        est.estimate_boundary(obs, mi, -2.0, 2.0)
        est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert est.stall_count >= 2
        assert est.should_trigger_probe

    def test_confident_resets_stall(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, stall_threshold=3,
        )
        # Stall first
        est.estimate_boundary([], [], -2.0, 2.0)
        assert est.stall_count == 1

        # Now good data
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert est.stall_count == 0

    def test_reset_stall_manual(self):
        est = BoundaryEstimator(stall_threshold=3)
        est._stall_count = 5
        est.reset_stall()
        assert est.stall_count == 0

    def test_trigger_threshold(self):
        est = BoundaryEstimator(stall_threshold=3)
        assert not est.should_trigger_probe
        est._stall_count = 3
        assert est.should_trigger_probe


# ── Test: Persistence ──────────────────────────────────────────────────


class TestPersistence:

    def test_round_trip(self):
        est = BoundaryEstimator()
        est._stall_count = 2
        est._updates_applied = 5
        state = est.as_dict()
        assert state["stall_count"] == 2
        assert state["updates_applied"] == 5

        est2 = BoundaryEstimator()
        est2.restore(state)
        assert est2._stall_count == 2
        assert est2._updates_applied == 5

    def test_empty_restore(self):
        est = BoundaryEstimator()
        est.restore({})
        assert est._stall_count == 0
        assert est._updates_applied == 0


# ── Test: updates_applied ──────────────────────────────────────────────


class TestUpdatesApplied:

    def test_increments_on_confident(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert est.updates_applied == 1

    def test_no_increment_on_stall(self):
        est = BoundaryEstimator(min_observations=50)
        est.estimate_boundary([], [], -2.0, 2.0)
        assert est.updates_applied == 0


# ── Test: Coarse + fine sweep ──────────────────────────────────────────


class TestSweepRefinement:

    def test_fine_sweep_improves_precision(self):
        """Fine sweep around coarse minimum should give sub-0.25°C precision."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            coarse_step=0.5, fine_step=0.05,
        )
        obs = _make_observations(breakpoint=0.3, noise_sigma=0.001, n=300)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert abs(result.estimated_breakpoint - 0.3) < 0.25
        assert result.n_candidates > 20  # coarse + fine candidates


# ── Test: Data asymmetry detection ────────────────────────────────────


class TestDataAsymmetry:
    """Closed-loop PI can produce heavily skewed observation distributions.

    When the HP is on 93%+ of the time (or off 93%+), the sweep
    optimizes for observation balance rather than physical truth,
    converging to the wrong breakpoint with high confidence.
    The asymmetry check rejects these estimates.
    """

    def test_asymmetric_data_rejected(self):
        """Strongly skewed data (95% HP-on) → not confident, data_asymmetric."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            max_imbalance=10.0,
        )
        # Generate observations where bp is at +4.0, so almost all
        # deltas in [-5, 4] are below bp → 95%+ HP-on.
        obs = _make_observations(breakpoint=4.0, n=200, delta_range=(-5.0, 4.5))
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert not result.confident
        assert result.data_asymmetric
        assert est.stall_count == 1

    def test_balanced_data_not_flagged(self):
        """Balanced data (bp near center of delta range) → not asymmetric."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            max_imbalance=10.0,
        )
        obs = _make_observations(breakpoint=0.0, n=200, delta_range=(-5.0, 4.0))
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert not result.data_asymmetric

    def test_moderate_imbalance_accepted(self):
        """5:1 imbalance (below 10:1 threshold) → still confident."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            max_imbalance=10.0,
        )
        # bp at 2.5 with range [-5, 4] → roughly 75% HP-on (ratio ~3:1)
        obs = _make_observations(breakpoint=2.5, n=200, delta_range=(-5.0, 4.0))
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident
        assert not result.data_asymmetric

    def test_asymmetry_stalls_toward_probe(self):
        """Repeated asymmetric batches accumulate stalls → trigger probe."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            max_imbalance=10.0, stall_threshold=3,
        )
        obs = _make_observations(breakpoint=4.0, n=200, delta_range=(-5.0, 4.5))
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        for _ in range(3):
            est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert est.should_trigger_probe
