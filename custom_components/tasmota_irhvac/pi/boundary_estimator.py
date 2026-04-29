"""Boundary estimation for HP on/off transition via split-model RSS sweep.

Three-layer approach to learning the offset between the room sensor and the
HP head unit's internal sensor:

**Layer 1 — Split-model RSS sweep (batch, every 12h)**:
  For each candidate breakpoint, fit a split model:
    Left (delta < bp):  room_rate = c0 + ua_c×od + k_c×hp_offset [+ α×solar]
    Right (delta >= bp): room_rate = c0 + ua_c×od [+ α×solar]  (no HP term)

  Score = RSS_null - RSS_split: how much the piecewise model improves
  over fitting envelope-only on all data.  The boundary is where adding
  hp_offset to the model stops helping — a structural break.

  Each side gets its own intercept and coefficients, absorbing PI-induced
  systematic biases.  No pre-identification of HP-off data needed, so
  this works at any offset from the first batch cycle.

**Layer 2 — Setpoint-change response (per-tick, opportunistic)**:
  When the HP setpoint changes, the room_rate response (or lack thereof)
  reveals whether the HP was affected.  Each setpoint change is an
  exogenous event — the PI decided to change based on the previous tick,
  and the response plays out over subsequent ticks.  These events provide
  high-quality evidence that works at any offset.

**Bayesian fusion**:
  Both layers update a Gaussian posterior N(μ, σ²) for the boundary
  location.  The posterior mean gives the best estimate; σ gives
  principled uncertainty; cal bounds = μ ± k×σ.

Literature:
  - Regression discontinuity design (Thistlethwaite & Campbell, 1960)
  - Bayesian changepoint detection (Adams & MacKay, 2007)
  - Closed-loop identification (Ljung, 1999; Hjalmarsson, 2005)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

# Layer 2: setpoint-change response detection
RESPONSE_WAIT_TICKS: int = 3
"""Ticks to wait after a setpoint change before measuring response."""

RESPONSE_MIN_RATE_CHANGE: float = 0.003
"""°C/min — minimum room_rate change to classify as "HP responded"."""

# Bayesian update weights (information content per evidence type)
BATCH_EVIDENCE_WEIGHT: float = 1.0
"""Weight for envelope-residual batch evidence."""

SETPOINT_CHANGE_WEIGHT: float = 3.0
"""Weight for setpoint-change response evidence (higher quality)."""

PROBE_EVIDENCE_WEIGHT: float = 5.0
"""Weight for active probe evidence (highest quality)."""


# ── Data structures ────────────────────────────────────────────────────


@dataclass
class BoundaryEstimateResult:
    """Result of a boundary estimation cycle."""

    confident: bool
    estimated_breakpoint: float | None
    breakpoint_rms: float | None  # envelope-residual RMS at best candidate
    rms_margin: float | None  # residual contrast (HP-on vs HP-off)
    slope_k_c: float | None  # mean residual in HP-on zone (≈ HP contribution)
    new_cal_min: float
    new_cal_max: float
    n_observations: int
    n_left: int  # observations left of breakpoint (HP on side)
    n_right: int  # observations right of breakpoint (HP off side)
    n_candidates: int  # candidates evaluated
    data_asymmetric: bool = False  # not enough HP-off data for envelope fit
    posterior_mean: float | None = None
    posterior_std: float | None = None


@dataclass
class SetpointChangeEvent:
    """A pending setpoint-change response observation."""

    mono_time: float  # when the setpoint changed
    old_setpoint: int  # HP setpoint before change
    new_setpoint: int  # HP setpoint after change
    delta_before: float  # current_c - old_setpoint at change time
    room_rate_before: float  # room_rate at change time
    current_c_at_change: float  # room temp when change occurred
    ticks_since: int = 0  # ticks elapsed since the change
    room_rates_after: list[float] = field(default_factory=list)


# ── Estimator ──────────────────────────────────────────────────────────


class BoundaryEstimator:
    """Three-layer boundary estimator with Bayesian fusion.

    The caller (pi_controller) is responsible for:
    - Calling estimate_boundary() each batch cycle (Layer 1)
    - Calling record_setpoint_change() when HP setpoint changes
    - Calling tick() each PI tick to advance Layer 2 state
    - Applying the returned bound updates
    """

    def __init__(
        self,
        *,
        sweep_step: float = 0.25,
        sweep_range: tuple[float, float] = (-5.0, 5.0),
        min_observations: int = 50,
        min_per_side: int = 10,
        max_step_per_update: float = 0.5,
        min_band_width: float = 0.5,
        safety_margin: float = 0.3,
        stall_threshold: int = 3,
        prior_mean: float = 0.0,
        prior_std: float = 2.0,
        confidence_std: float = 0.5,
    ):
        """Initialize.

        Args:
            sweep_step: Delta step for candidate sweep (°C).
            sweep_range: (min_delta, max_delta) range to search.
            min_observations: Minimum total observations needed.
            min_per_side: Minimum observations per side of candidate.
            max_step_per_update: Max cal bound movement per batch cycle.
            min_band_width: Floor for cal_max - cal_min.
            safety_margin: Buffer around posterior mean for bounds.
            stall_threshold: Batch cycles without confident update before probe.
            prior_mean: Prior boundary location (°C delta).
            prior_std: Prior uncertainty (°C).
            confidence_std: Posterior std below which estimate is "confident".
        """
        self._sweep_step = sweep_step
        self._sweep_min, self._sweep_max = sweep_range
        self._min_obs = min_observations
        self._min_per_side = min_per_side
        self._max_step = max_step_per_update
        self._min_band = min_band_width
        self._safety_margin = safety_margin
        self._stall_threshold = stall_threshold
        self._confidence_std = confidence_std

        # Bayesian state: boundary ~ N(μ, σ²)
        self._posterior_mean: float = prior_mean
        self._posterior_std: float = prior_std

        self._stall_count: int = 0
        self._updates_applied: int = 0
        self._last_result: BoundaryEstimateResult | None = None

        # Layer 2: pending setpoint-change events
        self._pending_events: list[SetpointChangeEvent] = []
        self._setpoint_evidence: list[tuple[float, float]] = []  # (bp_evidence, weight)

    # ── Layer 1: Envelope-residual batch estimation ────────────────

    def estimate_boundary(
        self,
        observations: list,
        model_inputs: list[dict],
        current_cal_min: float,
        current_cal_max: float,
    ) -> BoundaryEstimateResult:
        """Estimate boundary via envelope-residual sweep.

        Fits an envelope model on HP-off data, then sweeps candidates
        to find where residuals transition from non-zero to ~zero.
        Updates the Bayesian posterior with the result.

        Args:
            observations: Greybox buffer observations (Observation objects).
            model_inputs: PI model_inputs config (for solar entity lookup).
            current_cal_min: Current lower cal bound.
            current_cal_max: Current upper cal bound.

        Returns:
            BoundaryEstimateResult with posterior-derived bounds.
        """
        not_confident = BoundaryEstimateResult(
            confident=False,
            estimated_breakpoint=self._posterior_mean,
            breakpoint_rms=None,
            rms_margin=None,
            slope_k_c=None,
            new_cal_min=current_cal_min,
            new_cal_max=current_cal_max,
            n_observations=len(observations),
            n_left=0,
            n_right=0,
            n_candidates=0,
            posterior_mean=self._posterior_mean,
            posterior_std=self._posterior_std,
        )

        arrays = self._build_arrays(observations, model_inputs)
        if arrays is None or len(arrays[0]) < self._min_obs:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        deltas, outdoor_deltas, hp_offsets, solar_vals, room_rates, has_solar = arrays
        n = len(deltas)

        # Null model: envelope only (no HP term) on all data.
        # Only include solar column if the user configured a solar input.
        if has_solar:
            X_null = np.column_stack([np.ones(n), outdoor_deltas, solar_vals])
        else:
            X_null = np.column_stack([np.ones(n), outdoor_deltas])
        try:
            beta_null, _, _, _ = np.linalg.lstsq(X_null, room_rates, rcond=None)
        except np.linalg.LinAlgError:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident
        rss_null = float(np.sum((room_rates - X_null @ beta_null) ** 2))

        # Two-pass sweep: coarse (0.25°C) then fine (0.05°C) around best.
        coarse_candidates = np.arange(
            self._sweep_min,
            self._sweep_max + self._sweep_step / 2,
            self._sweep_step,
        )
        coarse_results = self._split_model_sweep(
            coarse_candidates, deltas, outdoor_deltas, hp_offsets,
            solar_vals, room_rates, rss_null, has_solar,
        )

        if not coarse_results:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        coarse_best = max(coarse_results, key=lambda x: x[1])

        # Fine sweep around coarse best
        fine_step = self._sweep_step / 5.0
        fine_candidates = np.arange(
            coarse_best[0] - self._sweep_step,
            coarse_best[0] + self._sweep_step + fine_step / 2,
            fine_step,
        )
        fine_results = self._split_model_sweep(
            fine_candidates, deltas, outdoor_deltas, hp_offsets,
            solar_vals, room_rates, rss_null, has_solar,
        )

        all_results = coarse_results + fine_results
        best = max(all_results, key=lambda x: x[1])
        best_bp, best_score, best_kc = best

        n_left = int((deltas < best_bp).sum())
        n_right = n - n_left

        # Confidence: score must be meaningfully positive
        if best_score <= 0:
            self._stall_count += 1
            result = BoundaryEstimateResult(
                confident=False,
                estimated_breakpoint=float(best_bp),
                breakpoint_rms=None,
                rms_margin=float(best_score),
                slope_k_c=float(best_kc),
                new_cal_min=current_cal_min,
                new_cal_max=current_cal_max,
                n_observations=n,
                n_left=n_left,
                n_right=n_right,
                n_candidates=len(all_results),
                posterior_mean=self._posterior_mean,
                posterior_std=self._posterior_std,
            )
            self._last_result = result
            return result

        # Bayesian update: observation noise inversely proportional
        # to RSS improvement (stronger signal = tighter observation).
        frac_explained = best_score / max(rss_null, 1e-12)
        obs_std = max(self._sweep_step, 0.3 / max(frac_explained, 0.01))
        obs_std = min(obs_std, 2.0)  # cap observation noise
        self._bayesian_update(float(best_bp), obs_std, BATCH_EVIDENCE_WEIGHT)

        # Also incorporate any pending setpoint-change evidence
        for bp_ev, weight in self._setpoint_evidence:
            self._bayesian_update(bp_ev, 1.0, weight)
        self._setpoint_evidence.clear()

        # Compute bounds from posterior
        confident = self._posterior_std < self._confidence_std
        new_min, new_max = self._posterior_to_bounds(
            current_cal_min, current_cal_max,
        )

        if confident:
            self._stall_count = 0
            self._updates_applied += 1
        else:
            self._stall_count += 1

        result = BoundaryEstimateResult(
            confident=confident,
            estimated_breakpoint=self._posterior_mean,
            breakpoint_rms=None,
            rms_margin=float(best_score),
            slope_k_c=float(best_kc),
            new_cal_min=new_min if confident else current_cal_min,
            new_cal_max=new_max if confident else current_cal_max,
            n_observations=n,
            n_left=n_left,
            n_right=n_right,
            n_candidates=len(all_results),
            posterior_mean=self._posterior_mean,
            posterior_std=self._posterior_std,
        )
        self._last_result = result
        return result

    def _build_arrays(
        self, observations: list, model_inputs: list[dict],
    ) -> tuple | None:
        """Extract numpy arrays from observations.

        Returns (deltas, outdoor_deltas, hp_offsets, solar_vals,
        room_rates, has_solar) or None if no valid observations.
        """
        solar_entity = None
        for mi in model_inputs:
            if mi.get("input_role") == "solar":
                solar_entity = mi["entity_id"]
                break

        deltas = []
        outdoor_deltas = []
        hp_offsets = []
        solar_vals = []
        room_rates = []

        for o in observations:
            if o.outdoor_temp_c is None:
                continue
            sp = o.hp_setpoint if o.hp_setpoint is not None else o.desired_c
            deltas.append(o.current_c - sp)
            outdoor_deltas.append(o.outdoor_temp_c - o.current_c)
            hp_offsets.append(sp - o.current_c)  # setpoint - room
            solar_v = 0.0
            if solar_entity:
                solar_v = o.raw_readings.get(solar_entity, 0.0)
            solar_vals.append(solar_v)
            room_rates.append(o.room_rate)

        if not deltas:
            return None

        return (
            np.array(deltas),
            np.array(outdoor_deltas),
            np.array(hp_offsets),
            np.array(solar_vals),
            np.array(room_rates),
            solar_entity is not None,
        )

    def _split_model_sweep(
        self,
        candidates: np.ndarray,
        deltas: np.ndarray,
        outdoor_deltas: np.ndarray,
        hp_offsets: np.ndarray,
        solar_vals: np.ndarray,
        room_rates: np.ndarray,
        rss_null: float,
        has_solar: bool,
    ) -> list[tuple[float, float, float]]:
        """Sweep candidates using split-model RSS reduction.

        For each candidate bp:
          - Left (delta < bp): room_rate = c0 + ua_c×od + k_c×hp_offset [+ α×solar]
          - Right (delta >= bp): room_rate = c0 + ua_c×od [+ α×solar] (no HP)

        Each side gets its own intercept and coefficients, which absorbs
        PI-induced systematic biases.  Score = RSS_null - RSS_split:
        how much the piecewise model (with hp_offset on the left) improves
        over the envelope-only null model.

        This works at any offset because it doesn't require pre-identifying
        HP-off data.  The boundary is where adding hp_offset to the model
        stops helping — a structural break in the data-generating process.

        Returns [(bp, score, k_c), ...].
        """
        results: list[tuple[float, float, float]] = []

        for bp in candidates:
            left = deltas < bp
            right = ~left
            n_l, n_r = int(left.sum()), int(right.sum())
            if n_l < self._min_per_side or n_r < self._min_per_side:
                continue

            # Left: full model with hp_offset
            cols_l = [np.ones(n_l), outdoor_deltas[left], hp_offsets[left]]
            if has_solar:
                cols_l.append(solar_vals[left])
            X_l = np.column_stack(cols_l)
            try:
                beta_l, _, _, _ = np.linalg.lstsq(X_l, room_rates[left], rcond=None)
                rss_l = float(np.sum((room_rates[left] - X_l @ beta_l) ** 2))
                k_c = float(beta_l[2])  # hp_offset coefficient
            except np.linalg.LinAlgError:
                continue

            # Right: envelope only (no hp_offset)
            cols_r = [np.ones(n_r), outdoor_deltas[right]]
            if has_solar:
                cols_r.append(solar_vals[right])
            X_r = np.column_stack(cols_r)
            try:
                beta_r, _, _, _ = np.linalg.lstsq(X_r, room_rates[right], rcond=None)
                rss_r = float(np.sum((room_rates[right] - X_r @ beta_r) ** 2))
            except np.linalg.LinAlgError:
                continue

            score = rss_null - (rss_l + rss_r)
            results.append((float(bp), score, k_c))

        return results

    # ── Layer 2: Setpoint-change response detection ────────────────

    def record_setpoint_change(
        self,
        mono_time: float,
        old_setpoint: int,
        new_setpoint: int,
        current_c: float,
        room_rate: float,
    ) -> None:
        """Record that the HP setpoint changed.

        Called by pi_controller when a setpoint change is actually sent.
        Starts tracking the response over the next few ticks.

        Only tracks events where delta is within the plausible boundary
        region.  When delta is very large (HP running hard or fully off),
        a ±1°C setpoint change won't produce a detectable room_rate
        change, leading to spurious "no response" evidence.
        """
        if old_setpoint == new_setpoint:
            return
        delta = current_c - old_setpoint
        # Only track events near the plausible boundary region.
        # Far from the boundary, the HP is either fully on or fully off
        # and won't respond detectably to small setpoint changes.
        if abs(delta) > 4.0:
            return
        event = SetpointChangeEvent(
            mono_time=mono_time,
            old_setpoint=old_setpoint,
            new_setpoint=new_setpoint,
            delta_before=delta,
            room_rate_before=room_rate,
            current_c_at_change=current_c,
        )
        self._pending_events.append(event)
        # Keep only the most recent few events
        if len(self._pending_events) > 5:
            self._pending_events.pop(0)

    def tick(self, room_rate: float, current_c: float, is_heating: bool) -> None:
        """Advance Layer 2 state each PI tick.

        Tracks room_rate after pending setpoint changes to detect
        whether the HP responded.
        """
        completed = []
        for event in self._pending_events:
            event.ticks_since += 1
            event.room_rates_after.append(room_rate)

            if event.ticks_since >= RESPONSE_WAIT_TICKS:
                self._analyze_setpoint_response(event, is_heating)
                completed.append(event)

        for event in completed:
            self._pending_events.remove(event)

    def _analyze_setpoint_response(
        self, event: SetpointChangeEvent, is_heating: bool,
    ) -> None:
        """Analyze a completed setpoint-change response event.

        If the HP responded (room_rate changed meaningfully), the boundary
        is NOT between the old and new deltas — both were in the active zone.
        If the HP didn't respond, the boundary IS between them — one delta
        was in the active zone and the other was past the boundary.
        """
        if not event.room_rates_after:
            return

        avg_rate_after = sum(event.room_rates_after) / len(event.room_rates_after)
        rate_change = avg_rate_after - event.room_rate_before

        delta_before = event.delta_before
        delta_after = event.current_c_at_change - event.new_setpoint

        # Did the HP respond to the setpoint change?
        # In heating: lowering setpoint (delta increases) should reduce HP
        # output → room_rate decreases.  Raising setpoint (delta decreases)
        # should increase HP output → room_rate increases.
        setpoint_increased = event.new_setpoint > event.old_setpoint
        if is_heating:
            hp_responded = (
                (setpoint_increased and rate_change > RESPONSE_MIN_RATE_CHANGE)
                or (not setpoint_increased and rate_change < -RESPONSE_MIN_RATE_CHANGE)
            )
        else:
            hp_responded = (
                (setpoint_increased and rate_change < -RESPONSE_MIN_RATE_CHANGE)
                or (not setpoint_increased and rate_change > RESPONSE_MIN_RATE_CHANGE)
            )

        if hp_responded:
            # Both deltas are in the active zone — boundary is beyond both.
            # This means boundary > max(delta_before, delta_after) for heating
            # (or < min for cooling).
            # Evidence: boundary is at least as far as the farther delta.
            farther = max(delta_before, delta_after)
            _LOGGER.info(
                "Boundary L2: HP responded to setpoint %d→%d "
                "(rate %.4f→%.4f), boundary > %.1f",
                event.old_setpoint, event.new_setpoint,
                event.room_rate_before, avg_rate_after, farther,
            )
            # Push posterior away from these deltas — boundary is farther out
            # We don't know exactly where, but it's beyond farther.
            # Use farther + 0.5 as the evidence point.
            self._setpoint_evidence.append(
                (farther + 0.5, SETPOINT_CHANGE_WEIGHT)
            )
        else:
            # HP did NOT respond — one delta was past the boundary.
            # The boundary is between the two deltas.
            midpoint = (delta_before + delta_after) / 2.0
            _LOGGER.info(
                "Boundary L2: HP did NOT respond to setpoint %d→%d "
                "(rate %.4f→%.4f), boundary ≈ %.1f",
                event.old_setpoint, event.new_setpoint,
                event.room_rate_before, avg_rate_after, midpoint,
            )
            self._setpoint_evidence.append(
                (midpoint, SETPOINT_CHANGE_WEIGHT)
            )

    # ── Layer 3: Probe evidence (called externally) ────────────────

    def record_probe_evidence(
        self,
        delta_at_probe: float,
        hp_was_contributing: bool,
    ) -> None:
        """Record evidence from an active probe.

        Called by pi_controller after a regime probe completes.

        Args:
            delta_at_probe: current_c - hp_setpoint at probe time.
            hp_was_contributing: True if probe detected HP contribution.
        """
        if hp_was_contributing:
            # HP was active at this delta → boundary is above this delta
            bp_evidence = delta_at_probe + 0.5
        else:
            # HP was NOT active → boundary is below this delta
            bp_evidence = delta_at_probe - 0.5

        _LOGGER.info(
            "Boundary L3: probe at delta=%.1f, HP %s → boundary evidence %.1f",
            delta_at_probe,
            "active" if hp_was_contributing else "idle",
            bp_evidence,
        )
        self._bayesian_update(bp_evidence, 0.5, PROBE_EVIDENCE_WEIGHT)

    # ── Bayesian posterior ─────────────────────────────────────────

    def _bayesian_update(
        self,
        observation: float,
        obs_std: float,
        weight: float = 1.0,
    ) -> None:
        """Update Gaussian posterior with a new observation.

        Conjugate Gaussian update:
          precision_new = precision_prior + weight * precision_obs
          mean_new = (precision_prior * mean_prior + weight * precision_obs * obs) / precision_new

        Args:
            observation: Observed boundary location.
            obs_std: Standard deviation of the observation noise.
            weight: Multiplier for the observation's information content.
        """
        prior_prec = 1.0 / (self._posterior_std ** 2)
        obs_prec = weight / (obs_std ** 2)

        new_prec = prior_prec + obs_prec
        new_mean = (prior_prec * self._posterior_mean + obs_prec * observation) / new_prec
        new_std = math.sqrt(1.0 / new_prec)

        self._posterior_mean = new_mean
        self._posterior_std = new_std

    def _posterior_to_bounds(
        self,
        current_cal_min: float,
        current_cal_max: float,
    ) -> tuple[float, float]:
        """Convert posterior to cal bounds with rate limiting.

        Bounds = posterior_mean ± max(safety_margin, 2σ), but never move
        faster than max_step per update cycle.
        """
        half_band = max(self._safety_margin, 2.0 * self._posterior_std)
        half_band = max(half_band, self._min_band / 2.0)

        target_min = self._posterior_mean - half_band
        target_max = self._posterior_mean + half_band

        new_min = self._move_toward(current_cal_min, target_min)
        new_max = self._move_toward(current_cal_max, target_max)

        # Enforce minimum band width
        if new_max - new_min < self._min_band:
            mid = (new_max + new_min) / 2.0
            new_min = mid - self._min_band / 2.0
            new_max = mid + self._min_band / 2.0

        return new_min, new_max

    def _move_toward(self, current: float, target: float) -> float:
        """Move current toward target by at most max_step."""
        diff = target - current
        if abs(diff) <= self._max_step:
            return target
        return current + math.copysign(self._max_step, diff)

    # ── Stall detection ────────────────────────────────────────────

    @property
    def stall_count(self) -> int:
        """Number of consecutive batch cycles without confident update."""
        return self._stall_count

    @property
    def should_trigger_probe(self) -> bool:
        """Whether passive estimation has stalled and active probe needed."""
        return self._stall_count >= self._stall_threshold

    @property
    def updates_applied(self) -> int:
        """Total confident boundary updates applied."""
        return self._updates_applied

    @property
    def last_result(self) -> BoundaryEstimateResult | None:
        """Most recent estimation result (for diagnostics)."""
        return self._last_result

    @property
    def posterior_mean(self) -> float:
        """Current boundary estimate (posterior mean)."""
        return self._posterior_mean

    @property
    def posterior_std(self) -> float:
        """Current estimate uncertainty (posterior std)."""
        return self._posterior_std

    def reset_stall(self) -> None:
        """Reset stall counter (called after active probe completes)."""
        self._stall_count = 0

    # ── Persistence ────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Serialize state for persistence across restarts."""
        return {
            "stall_count": self._stall_count,
            "updates_applied": self._updates_applied,
            "posterior_mean": self._posterior_mean,
            "posterior_std": self._posterior_std,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from persisted data."""
        self._stall_count = int(data.get("stall_count", 0))
        self._updates_applied = int(data.get("updates_applied", 0))
        if "posterior_mean" in data:
            self._posterior_mean = float(data["posterior_mean"])
        if "posterior_std" in data:
            self._posterior_std = float(data["posterior_std"])
