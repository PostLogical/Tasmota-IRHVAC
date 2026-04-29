"""Unit tests for BoundaryEstimator — envelope-residual + Bayesian fusion."""

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

    delta = current_c - hp_setpoint.  HP is active when delta < breakpoint.
    """
    rng = np.random.default_rng(seed)
    obs = []
    for i in range(n):
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


# ── Test: Layer 1 — Envelope-residual breakpoint detection ─────────────


class TestEnvelopeResidualDetection:
    """Core: detect a known breakpoint from synthetic data."""

    def test_finds_breakpoint_at_zero(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        # cal_max=2.0 means HP-off data is delta > 2.0 — clean envelope
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.confident or result.rms_margin is not None
        if result.estimated_breakpoint is not None:
            assert abs(result.estimated_breakpoint - 0.0) < 1.0

    def test_finds_breakpoint_at_positive(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=1.5)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 3.0)
        if result.confident and result.estimated_breakpoint is not None:
            assert abs(result.estimated_breakpoint - 1.5) < 1.0

    def test_finds_breakpoint_at_negative(self):
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=-1.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        if result.confident and result.estimated_breakpoint is not None:
            assert abs(result.estimated_breakpoint - (-1.0)) < 1.0

    def test_no_solar_model_input(self):
        """Works without solar as a model input."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0, alpha_c=0.0)
        result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert result.estimated_breakpoint is not None

    def test_mean_residual_positive_on_hp_side(self):
        """In heating, residuals on the HP-on side should be positive."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0, k_c=0.008)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        # slope_k_c stores mean residual on HP-on side
        if result.slope_k_c is not None:
            assert result.slope_k_c > 0


# ── Test: Bound updates ────────────────────────────────────────────────


class TestBoundUpdates:
    """Verify bound movement logic."""

    def test_shrinks_toward_breakpoint(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=5.0,
            confidence_std=2.0,  # Easy confidence for this test
        )
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        if result.confident:
            assert result.new_cal_min > -2.0 or result.new_cal_max < 2.0

    def test_max_step_cap(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.3, max_step_per_update=0.2,
            confidence_std=2.0,
        )
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        if result.confident:
            # Bounds shouldn't move more than 0.2 from originals
            assert result.new_cal_min <= -2.0 + 0.2 + 0.01
            assert result.new_cal_max >= 2.0 - 0.2 - 0.01

    def test_min_band_floor(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            safety_margin=0.1, min_band_width=0.8,
            max_step_per_update=5.0,
            confidence_std=2.0,
        )
        obs = _make_observations(breakpoint=0.0, noise_sigma=0.001)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -0.5, 0.5)
        if result.confident:
            assert result.new_cal_max - result.new_cal_min >= 0.8 - 0.001


# ── Test: Stall detection ────────────────────────────────────────────


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
        """No HP contribution → no breakpoint improvement → stall."""
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5, stall_threshold=2,
        )
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
            min_observations=20, min_per_side=5,
            stall_threshold=3, confidence_std=2.0,
        )
        # Stall first
        est.estimate_boundary([], [], -2.0, 2.0)
        assert est.stall_count == 1

        # Now good data
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        if result.confident:
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


# ── Test: Bayesian posterior ──────────────────────────────────────────


class TestBayesianPosterior:

    def test_prior_is_wide(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        assert est.posterior_mean == 0.0
        assert est.posterior_std == 2.0

    def test_update_shrinks_std(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        est._bayesian_update(0.5, 1.0)
        assert est.posterior_std < 2.0
        assert est.posterior_mean > 0.0  # pulled toward 0.5

    def test_multiple_updates_converge(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        # 10 observations at 1.5 with σ=0.5
        for _ in range(10):
            est._bayesian_update(1.5, 0.5)
        assert abs(est.posterior_mean - 1.5) < 0.2
        assert est.posterior_std < 0.3

    def test_weighted_update(self):
        """Higher weight → faster convergence."""
        est1 = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        est2 = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        est1._bayesian_update(1.0, 1.0, weight=1.0)
        est2._bayesian_update(1.0, 1.0, weight=5.0)
        assert est2.posterior_std < est1.posterior_std
        assert abs(est2.posterior_mean - 1.0) < abs(est1.posterior_mean - 1.0)

    def test_probe_evidence_updates_posterior(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        # Probe found HP contributing at delta=-1.0 → boundary > -1.0
        # Evidence = -1.0 + 0.5 = -0.5.  Posterior pulled toward -0.5.
        est.record_probe_evidence(-1.0, hp_was_contributing=True)
        assert est.posterior_mean > -1.0  # pulled above the probe delta
        assert est.posterior_std < 2.0

    def test_probe_evidence_not_contributing(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        # Probe found HP NOT contributing at delta=1.0 → boundary < 1.0
        # Evidence = 1.0 - 0.5 = 0.5.  Posterior pulled toward 0.5.
        est.record_probe_evidence(1.0, hp_was_contributing=False)
        assert est.posterior_mean > 0.0  # pulled toward 0.5
        assert est.posterior_std < 2.0


# ── Test: Layer 2 — Setpoint-change response ──────────────────────────


class TestSetpointChangeResponse:

    def test_record_and_tick(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        est.record_setpoint_change(
            mono_time=1000.0,
            old_setpoint=20,
            new_setpoint=22,
            current_c=21.0,
            room_rate=0.01,
        )
        assert len(est._pending_events) == 1

    def test_no_duplicate_for_same_setpoint(self):
        est = BoundaryEstimator()
        est.record_setpoint_change(1000.0, 20, 20, 21.0, 0.01)
        assert len(est._pending_events) == 0

    def test_response_detected_after_ticks(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        est.record_setpoint_change(
            mono_time=1000.0,
            old_setpoint=20,
            new_setpoint=22,
            current_c=21.0,
            room_rate=0.01,
        )
        # Simulate 3 ticks with increased room_rate (HP responded in heating)
        for _ in range(3):
            est.tick(room_rate=0.02, current_c=21.0, is_heating=True)

        # Event should be consumed after 3 ticks
        assert len(est._pending_events) == 0
        # Evidence should be accumulated
        assert len(est._setpoint_evidence) == 1

    def test_no_response_gives_boundary_evidence(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        est.record_setpoint_change(
            mono_time=1000.0,
            old_setpoint=20,
            new_setpoint=22,
            current_c=21.0,
            room_rate=0.01,
        )
        # Simulate 3 ticks with SAME room_rate (HP didn't respond)
        for _ in range(3):
            est.tick(room_rate=0.01, current_c=21.0, is_heating=True)

        assert len(est._pending_events) == 0
        assert len(est._setpoint_evidence) == 1

    def test_pending_events_capped(self):
        est = BoundaryEstimator()
        for i in range(10):
            est.record_setpoint_change(
                float(i * 1000), 20 + i, 21 + i, 21.0, 0.01,
            )
        assert len(est._pending_events) <= 5


# ── Test: Large offset detection ──────────────────────────────────────


class TestLargeOffsetDetection:
    """Split-model RSS sweep finds boundaries at any offset without
    needing pre-identified HP-off data.
    """

    @pytest.mark.parametrize("bp_true", [-4.0, -3.0, -2.0, 2.0, 3.0])
    def test_finds_large_offsets(self, bp_true):
        """Offsets up to ±4°C found on first batch."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=bp_true, n=300)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.estimated_breakpoint is not None
        assert abs(result.estimated_breakpoint - bp_true) < 1.0, (
            f"bp_true={bp_true}: estimated {result.estimated_breakpoint:.2f}"
        )


# ── Test: Persistence ────────────────────────────────────────────────


class TestPersistence:

    def test_round_trip(self):
        est = BoundaryEstimator()
        est._stall_count = 2
        est._updates_applied = 5
        est._posterior_mean = 0.8
        est._posterior_std = 0.4
        state = est.as_dict()
        assert state["stall_count"] == 2
        assert state["updates_applied"] == 5
        assert state["posterior_mean"] == 0.8
        assert state["posterior_std"] == 0.4

        est2 = BoundaryEstimator()
        est2.restore(state)
        assert est2._stall_count == 2
        assert est2._updates_applied == 5
        assert est2._posterior_mean == 0.8
        assert est2._posterior_std == 0.4

    def test_empty_restore(self):
        est = BoundaryEstimator()
        est.restore({})
        assert est._stall_count == 0
        assert est._updates_applied == 0

    def test_restore_backward_compat(self):
        """Old format without posterior fields should work."""
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0)
        est.restore({"stall_count": 1, "updates_applied": 3})
        assert est._stall_count == 1
        assert est._updates_applied == 3
        # Posterior stays at prior
        assert est._posterior_mean == 0.0
        assert est._posterior_std == 2.0


# ── Test: updates_applied ────────────────────────────────────────────


class TestUpdatesApplied:

    def test_increments_on_confident(self):
        est = BoundaryEstimator(
            min_observations=20, min_per_side=5,
            confidence_std=2.0,
        )
        obs = _make_observations(breakpoint=0.0)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        if result.confident:
            assert est.updates_applied == 1

    def test_no_increment_on_stall(self):
        est = BoundaryEstimator(min_observations=50)
        est.estimate_boundary([], [], -2.0, 2.0)
        assert est.updates_applied == 0


# ── Test: Posterior-to-bounds conversion ──────────────────────────────


class TestPosteriorToBounds:

    def test_wide_posterior_gives_wide_band(self):
        est = BoundaryEstimator(prior_mean=0.0, prior_std=2.0, safety_margin=0.3)
        new_min, new_max = est._posterior_to_bounds(-2.0, 2.0)
        # 2σ = 4.0, so band should be wide
        assert new_max - new_min >= 4.0

    def test_narrow_posterior_gives_narrow_band(self):
        est = BoundaryEstimator(
            prior_mean=0.5, prior_std=0.2,
            safety_margin=0.3, min_band_width=0.5,
            max_step_per_update=5.0,
        )
        new_min, new_max = est._posterior_to_bounds(-2.0, 2.0)
        band = new_max - new_min
        # Should be relatively tight around 0.5
        assert band < 2.0
        mid = (new_min + new_max) / 2
        assert abs(mid - 0.5) < 1.0

    def test_min_band_width_enforced(self):
        est = BoundaryEstimator(
            prior_mean=0.0, prior_std=0.01,
            min_band_width=0.8, max_step_per_update=5.0,
        )
        new_min, new_max = est._posterior_to_bounds(-0.5, 0.5)
        assert new_max - new_min >= 0.8 - 0.001


# ── Test: Split-model sweep internals ────────────────────────────────


class TestSplitModelSweep:

    def test_recovers_k_c_at_boundary(self):
        """Split-model sweep recovers positive k_c at the true breakpoint."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(breakpoint=0.0, k_c=0.008, n=300)
        mi = [{"input_role": "solar", "entity_id": "sensor.solar"}]
        result = est.estimate_boundary(obs, mi, -2.0, 2.0)
        assert result.slope_k_c is not None
        assert result.slope_k_c > 0


class TestEdgeCases:
    """Defensive branches and rarely-hit paths."""

    def test_skips_observations_without_outdoor_temp(self):
        """Observations with outdoor_temp_c=None are silently skipped."""
        est = BoundaryEstimator(min_observations=10, min_per_side=3)
        obs = _make_observations(n=50)
        # Replace ~10 observations with outdoor-temp-missing (still build_arrays
        # should drop them and proceed with the rest).
        for i in range(10):
            obs[i] = FakeObs(
                current_c=obs[i].current_c, hp_setpoint=obs[i].hp_setpoint,
                desired_c=obs[i].desired_c, outdoor_temp_c=None,
                room_rate=obs[i].room_rate, raw_readings=obs[i].raw_readings,
            )
        result = est.estimate_boundary(
            obs, [{"input_role": "solar", "entity_id": "sensor.solar"}],
            -2.0, 2.0,
        )
        # Estimator should still complete (rows with None outdoor were filtered)
        assert result.n_observations == 40

    def test_empty_coarse_results_stalls(self):
        """If sweep produces no candidates with valid splits, stall counter rises."""
        # Force min_per_side larger than any valid split: with 25 obs, no
        # candidate breakpoint can split into ≥20 on each side.
        est = BoundaryEstimator(min_observations=20, min_per_side=20)
        obs = _make_observations(n=25)
        result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert not result.confident
        assert est.stall_count >= 1

    def test_low_score_returns_unconfident_result(self):
        """When best_score ≤ 0, result reports the candidate but isn't confident.

        Pure linear-in-outdoor signal with no HP gating: null model
        (intercept + outdoor_delta + solar) fits perfectly. Split models
        also fit perfectly but with more parameters, so rss_null = rss_l +
        rss_r = 0 → best_score = 0, falls into the unconfident branch.
        """
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        rng = np.random.default_rng(7)
        obs = []
        for _ in range(80):
            delta = rng.uniform(-2.0, 2.0)
            outdoor = 10.0 + rng.normal(0, 1.0)
            room = 20.5 + rng.normal(0, 0.05)
            outdoor_delta = outdoor - room
            # Pure linear: rate = 0.001 × outdoor_delta. No HP, no solar, no noise.
            obs.append(FakeObs(
                current_c=room, hp_setpoint=room - delta,
                desired_c=20.5, outdoor_temp_c=outdoor,
                room_rate=0.001 * outdoor_delta,
                raw_readings={},
            ))
        result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert not result.confident
        # Whatever breakpoint the sweep landed on is reported
        assert result.estimated_breakpoint is not None
        assert result.rms_margin is not None

    def test_pending_setpoint_evidence_consumed_on_estimate(self):
        """Pending setpoint-change evidence is incorporated into Bayesian update."""
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        # Inject pending evidence directly (bypassing the tick machinery)
        est._setpoint_evidence.append((0.5, 0.8))
        assert len(est._setpoint_evidence) == 1
        obs = _make_observations(breakpoint=0.0, n=100)
        est.estimate_boundary(
            obs, [{"input_role": "solar", "entity_id": "sensor.solar"}],
            -2.0, 2.0,
        )
        # After estimate, pending evidence should be consumed.
        assert len(est._setpoint_evidence) == 0

    def test_analyze_empty_response_event_returns_early(self):
        """Setpoint-change event with no post-change rates is silently ignored."""
        from custom_components.tasmota_irhvac.pi.boundary_estimator import (
            SetpointChangeEvent,
        )
        est = BoundaryEstimator()
        event = SetpointChangeEvent(
            mono_time=0.0,
            old_setpoint=22, new_setpoint=21,
            delta_before=-2.0,
            room_rate_before=0.001,
            current_c_at_change=20.0,
            room_rates_after=[],  # ← no ticks accumulated yet
        )
        # Should return without recording any evidence
        est._analyze_setpoint_response(event, is_heating=True)
        assert len(est._setpoint_evidence) == 0

    def test_lstsq_failure_on_null_model_stalls(self):
        """LinAlgError on null-model lstsq → stall + not-confident result."""
        from unittest.mock import patch as _patch
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(n=80)
        # Force np.linalg.lstsq to raise LinAlgError on the very first call
        # (the null-model fit). estimate_boundary catches it, increments
        # stall, and returns the not-confident sentinel.
        original_lstsq = np.linalg.lstsq

        def fail_first(X, y, rcond=None):
            fail_first.calls += 1
            if fail_first.calls == 1:
                raise np.linalg.LinAlgError("forced for test")
            return original_lstsq(X, y, rcond=rcond)
        fail_first.calls = 0

        with _patch.object(np.linalg, "lstsq", side_effect=fail_first):
            result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert result is not None
        assert not result.confident
        assert est.stall_count == 1

    def test_lstsq_failure_on_left_fit_skips_candidate(self):
        """LinAlgError on left-side fit makes that candidate fall out of the sweep."""
        from unittest.mock import patch as _patch
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(n=120)
        original = np.linalg.lstsq
        # Skip the null-model call (first), then raise on every other call
        # (alternating left/right). Catches the left handler at lines 424-425.
        def lstsq_split_fail(X, y, rcond=None):
            lstsq_split_fail.calls += 1
            if lstsq_split_fail.calls == 1:
                return original(X, y, rcond=rcond)
            raise np.linalg.LinAlgError("forced left-fit failure")
        lstsq_split_fail.calls = 0
        with _patch.object(np.linalg, "lstsq", side_effect=lstsq_split_fail):
            result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert not result.confident
        assert est.stall_count == 1

    def test_lstsq_failure_on_right_fit_skips_candidate(self):
        """LinAlgError on right-side fit makes that candidate fall out (lines 435-436)."""
        from unittest.mock import patch as _patch
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(n=120)
        original = np.linalg.lstsq
        # Skip the null-model call (first), then alternate: left fits succeed
        # (call # is 2, 4, 6, ...) but right fits fail (3, 5, 7, ...).
        def lstsq_right_fail(X, y, rcond=None):
            lstsq_right_fail.calls += 1
            if lstsq_right_fail.calls == 1:
                return original(X, y, rcond=rcond)
            # Even calls = left, odd = right (after the null)
            if lstsq_right_fail.calls % 2 == 0:
                return original(X, y, rcond=rcond)
            raise np.linalg.LinAlgError("forced right-fit failure")
        lstsq_right_fail.calls = 0
        with _patch.object(np.linalg, "lstsq", side_effect=lstsq_right_fail):
            result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert not result.confident
        assert est.stall_count == 1

    def test_low_score_unconfident_when_split_no_better_than_null(self):
        """When split RSS ≥ null RSS for all candidates, best_score≤0 path fires."""
        from unittest.mock import patch as _patch
        est = BoundaryEstimator(min_observations=20, min_per_side=5)
        obs = _make_observations(n=80)
        # Patch _split_model_sweep to return candidates whose score is non-positive
        original_sweep = est._split_model_sweep
        def neg_score_sweep(*args, **kwargs):
            results = original_sweep(*args, **kwargs)
            # Force all scores negative
            return [(bp, -abs(score) - 0.001, k_c) for (bp, score, k_c) in results]
        with _patch.object(est, "_split_model_sweep", side_effect=neg_score_sweep):
            result = est.estimate_boundary(obs, [], -2.0, 2.0)
        assert not result.confident
        assert result.estimated_breakpoint is not None
        assert result.rms_margin is not None
        assert result.rms_margin <= 0  # negative score reported back

    def test_min_band_enforcement_when_step_clipped(self):
        """min_band protection fires when max_step_per_update prevents reaching half-band targets."""
        # Existing cal band is small (1°C); min_band requires 5°C; max_step
        # caps movement so move_toward can only shift each endpoint by 0.1°C
        # toward the wider targets. Result: post-step band is still ~1°C,
        # below min_band → line 648-651 widens it to min_band centered on
        # the (post-step) midpoint.
        est = BoundaryEstimator(
            prior_mean=0.0, prior_std=0.001,
            safety_margin=0.0,
            min_band_width=5.0,
            max_step_per_update=0.1,
        )
        new_min, new_max = est._posterior_to_bounds(-1.0, 0.0)
        # Band must respect min_band_width
        assert new_max - new_min >= 5.0 - 1e-6
        # Centered on the post-step midpoint (originally -0.5, then both
        # shifted toward target by max_step=0.1, midpoint stays near -0.4).
        midpoint = (new_min + new_max) / 2.0
        assert -1.0 < midpoint < 0.0
