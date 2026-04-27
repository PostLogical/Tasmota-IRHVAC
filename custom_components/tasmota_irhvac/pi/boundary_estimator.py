"""Passive regime boundary estimation via greybox profile sweep.

Estimates where the HP transitions from actively contributing to idle
by fitting the greybox energy balance model with different candidate
boundary locations and selecting the one that minimizes fit residual.

For each candidate breakpoint bp:
  - Observations with delta < bp get hp_offset = setpoint - room (HP on)
  - Observations with delta >= bp get hp_offset = 0 (HP off)
  - Fit: room_rate = c0 + ua_c × outdoor_delta + k_c × hp_offset + α_c × solar
  - Record fit RMS

The bp with lowest RMS is where hp_offset assignments best explain the
observed room_rates — i.e., the true HP transition.

Runs each batch cycle using the greybox observation buffer.  No per-tick
evidence accumulation needed.

Literature: profile likelihood over a nuisance parameter (boundary location).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class BoundaryEstimateResult:
    """Result of a boundary estimation cycle."""

    confident: bool
    estimated_breakpoint: float | None
    breakpoint_rms: float | None  # RMS at the best candidate
    rms_margin: float | None  # RMS difference between best and runner-up region
    slope_k_c: float | None  # fitted k_c at the best candidate
    new_cal_min: float
    new_cal_max: float
    n_observations: int
    n_left: int  # observations left of breakpoint (HP on side)
    n_right: int  # observations right of breakpoint (HP off side)
    n_candidates: int  # candidates evaluated
    data_asymmetric: bool = False  # observation balance too skewed for reliable estimate


# ── Estimator ───────────────────────────────────────────────────────────


class BoundaryEstimator:
    """Passive regime boundary estimation via greybox profile sweep.

    Each batch cycle, sweeps candidate boundary locations and fits
    an OLS energy balance at each.  The candidate with lowest RMS
    is the estimated HP transition boundary.

    The caller (pi_controller) is responsible for:
    - Providing the greybox observation buffer each batch cycle
    - Providing the model_inputs config (for solar entity lookup)
    - Calling estimate_boundary() each batch cycle
    - Applying the returned bound updates
    """

    def __init__(
        self,
        *,
        coarse_step: float = 0.25,
        fine_step: float = 0.05,
        sweep_range: tuple[float, float] = (-5.0, 5.0),
        min_observations: int = 50,
        min_per_side: int = 10,
        min_rms_margin: float = 0.0001,
        max_step_per_update: float = 0.3,
        min_band_width: float = 0.5,
        safety_margin: float = 0.3,
        stall_threshold: int = 3,
        max_imbalance: float = 10.0,
    ):
        """Initialize.

        Args:
            coarse_step: Delta step for initial coarse sweep (°C).
            fine_step: Delta step for refinement around coarse minimum (°C).
            sweep_range: (min_delta, max_delta) range to search.
            min_observations: Minimum greybox buffer observations needed.
            min_per_side: Minimum observations on each side of candidate.
            min_rms_margin: Minimum RMS improvement over flat model to accept.
            max_step_per_update: Max cal bound movement per batch cycle.
            min_band_width: Floor for cal_max - cal_min.
            safety_margin: Buffer around estimated breakpoint for bounds.
            stall_threshold: Batch cycles without confident estimate
                before should_trigger_probe fires.
            max_imbalance: Maximum allowed ratio of majority/minority
                observations at the best breakpoint.  Above this, the
                closed-loop data is too asymmetric for reliable
                identification and the estimate is rejected.
        """
        self._coarse_step = coarse_step
        self._fine_step = fine_step
        self._sweep_min, self._sweep_max = sweep_range
        self._min_obs = min_observations
        self._min_per_side = min_per_side
        self._min_rms_margin = min_rms_margin
        self._max_step = max_step_per_update
        self._min_band = min_band_width
        self._safety_margin = safety_margin
        self._stall_threshold = stall_threshold
        self._max_imbalance = max_imbalance

        self._stall_count: int = 0
        self._updates_applied: int = 0
        self._last_result: BoundaryEstimateResult | None = None

    # ── Core estimation ─────────────────────────────────────────────

    def estimate_boundary(
        self,
        observations: list,
        model_inputs: list[dict],
        current_cal_min: float,
        current_cal_max: float,
    ) -> BoundaryEstimateResult:
        """Estimate the HP on/off transition boundary.

        Sweeps candidate breakpoints, fits OLS energy balance at each,
        selects the candidate with lowest residual RMS.

        Args:
            observations: Greybox buffer observations (Observation objects).
            model_inputs: PI model_inputs config (for solar entity lookup).
            current_cal_min: Current lower cal bound.
            current_cal_max: Current upper cal bound.

        Returns:
            BoundaryEstimateResult with confident flag and proposed bounds.
        """
        not_confident = BoundaryEstimateResult(
            confident=False,
            estimated_breakpoint=None,
            breakpoint_rms=None,
            rms_margin=None,
            slope_k_c=None,
            new_cal_min=current_cal_min,
            new_cal_max=current_cal_max,
            n_observations=len(observations),
            n_left=0,
            n_right=0,
            n_candidates=0,
        )

        # Precompute arrays from observations
        arrays = self._build_arrays(observations, model_inputs)
        if arrays is None or len(arrays[0]) < self._min_obs:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        deltas, outdoor_deltas, hp_setpoints, room_temps, solar_vals, room_rates = arrays
        n = len(deltas)

        # Coarse sweep (split model: separate OLS for each side)
        coarse_candidates = np.arange(
            self._sweep_min, self._sweep_max + self._coarse_step / 2,
            self._coarse_step,
        )
        coarse_results = self._sweep(
            coarse_candidates, deltas, outdoor_deltas, hp_setpoints,
            room_temps, solar_vals, room_rates,
        )
        # Also run single-model sweep as fallback — the split model's
        # separate intercepts can cause k_c to go negative for large
        # offsets due to endogeneity, but the single model handles
        # these cases better.
        coarse_results_single = self._sweep_single(
            coarse_candidates, deltas, outdoor_deltas, hp_setpoints,
            room_temps, solar_vals, room_rates,
        )

        if not coarse_results and not coarse_results_single:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        # Find coarse minimum (k_c must be positive).
        # Prefer split model; fall back to single model when split
        # produces no valid candidates (all k_c < 0 from endogeneity).
        valid = [(bp, rms, kc) for bp, rms, kc in coarse_results if kc > 0]
        using_single = False
        if not valid:
            valid = [(bp, rms, kc) for bp, rms, kc in coarse_results_single if kc > 0]
            using_single = True
        if not valid:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        coarse_best = min(valid, key=lambda x: x[1])

        # Fine sweep around coarse minimum using the same method
        fine_candidates = np.arange(
            coarse_best[0] - self._coarse_step,
            coarse_best[0] + self._coarse_step + self._fine_step / 2,
            self._fine_step,
        )
        sweep_fn = self._sweep_single if using_single else self._sweep
        fine_results = sweep_fn(
            fine_candidates, deltas, outdoor_deltas, hp_setpoints,
            room_temps, solar_vals, room_rates,
        )

        all_results = (coarse_results_single if using_single else coarse_results) + fine_results
        valid = [(bp, rms, kc) for bp, rms, kc in all_results if kc > 0]
        if not valid:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        sorted_valid = sorted(valid, key=lambda x: x[1])
        best_bp, best_rms, best_kc = sorted_valid[0]

        # Count observations on each side
        n_left = int((deltas < best_bp).sum())
        n_right = n - n_left

        # Confidence checks
        # 0. Data asymmetry: closed-loop PI can produce heavily skewed
        #    observation distributions (e.g., HP on 93% of ticks).
        #    The sweep optimizes for observation balance rather than
        #    physical truth, converging confidently to the wrong bp.
        #    Refuse to commit when the data can't support identification.
        minority = min(n_left, n_right)
        majority = max(n_left, n_right)
        balance_ratio = majority / max(minority, 1)
        if balance_ratio > self._max_imbalance:
            self._stall_count += 1
            result = BoundaryEstimateResult(
                confident=False,
                estimated_breakpoint=float(best_bp),
                breakpoint_rms=float(best_rms),
                rms_margin=None,
                slope_k_c=float(best_kc),
                new_cal_min=current_cal_min,
                new_cal_max=current_cal_max,
                n_observations=n,
                n_left=n_left,
                n_right=n_right,
                n_candidates=len(all_results),
                data_asymmetric=True,
            )
            self._last_result = result
            return result

        # 1. Enough observations on each side
        if n_left < self._min_per_side or n_right < self._min_per_side:
            self._stall_count += 1
            result = BoundaryEstimateResult(
                confident=False,
                estimated_breakpoint=float(best_bp),
                breakpoint_rms=float(best_rms),
                rms_margin=None,
                slope_k_c=float(best_kc),
                new_cal_min=current_cal_min,
                new_cal_max=current_cal_max,
                n_observations=n,
                n_left=n_left,
                n_right=n_right,
                n_candidates=len(all_results),
            )
            self._last_result = result
            return result

        # 2. RMS margin: best must be meaningfully better than distant candidates.
        #    Compare best RMS against the RMS at ±2°C away (or edges).
        distant = [
            rms for bp, rms, kc in valid
            if abs(bp - best_bp) > 1.5 and kc > 0
        ]
        if distant:
            rms_margin = float(np.median(distant) - best_rms)
        else:
            rms_margin = 0.0

        confident = rms_margin > self._min_rms_margin

        if not confident:
            self._stall_count += 1
            result = BoundaryEstimateResult(
                confident=False,
                estimated_breakpoint=float(best_bp),
                breakpoint_rms=float(best_rms),
                rms_margin=rms_margin,
                slope_k_c=float(best_kc),
                new_cal_min=current_cal_min,
                new_cal_max=current_cal_max,
                n_observations=n,
                n_left=n_left,
                n_right=n_right,
                n_candidates=len(all_results),
            )
            self._last_result = result
            return result

        # Confident — compute new bounds
        target_min = float(best_bp) - self._safety_margin
        target_max = float(best_bp) + self._safety_margin

        new_min = self._move_toward(current_cal_min, target_min)
        new_max = self._move_toward(current_cal_max, target_max)

        # Enforce minimum band width
        if new_max - new_min < self._min_band:
            mid = (new_max + new_min) / 2.0
            new_min = mid - self._min_band / 2.0
            new_max = mid + self._min_band / 2.0

        self._stall_count = 0
        self._updates_applied += 1

        result = BoundaryEstimateResult(
            confident=True,
            estimated_breakpoint=float(best_bp),
            breakpoint_rms=float(best_rms),
            rms_margin=rms_margin,
            slope_k_c=float(best_kc),
            new_cal_min=new_min,
            new_cal_max=new_max,
            n_observations=n,
            n_left=n_left,
            n_right=n_right,
            n_candidates=len(all_results),
        )
        self._last_result = result
        return result

    def _build_arrays(
        self, observations: list, model_inputs: list[dict],
    ) -> tuple | None:
        """Extract numpy arrays from observations."""
        solar_entity = None
        for mi in model_inputs:
            if mi.get("input_role") == "solar":
                solar_entity = mi["entity_id"]
                break

        deltas = []
        outdoor_deltas = []
        hp_setpoints = []
        room_temps = []
        solar_vals = []
        room_rates = []

        for o in observations:
            if o.outdoor_temp_c is None:
                continue
            sp = o.hp_setpoint if o.hp_setpoint is not None else o.desired_c
            deltas.append(o.current_c - sp)
            outdoor_deltas.append(o.outdoor_temp_c - o.current_c)
            hp_setpoints.append(sp)
            room_temps.append(o.current_c)
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
            np.array(hp_setpoints),
            np.array(room_temps),
            np.array(solar_vals),
            np.array(room_rates),
        )

    def _sweep(
        self,
        candidates: np.ndarray,
        deltas: np.ndarray,
        outdoor_deltas: np.ndarray,
        hp_setpoints: np.ndarray,
        room_temps: np.ndarray,
        solar_vals: np.ndarray,
        room_rates: np.ndarray,
    ) -> list[tuple[float, float, float]]:
        """Sweep candidate breakpoints using split-model OLS.

        For each candidate bp, fit SEPARATE models on each side:
          Left (delta < bp):  room_rate = c0 + ua_c*od + k_c*hp_offset + a*solar
          Right (delta >= bp): room_rate = c0 + ua_c*od + a*solar  (no HP term)

        The split model allows different intercepts for HP-on and HP-off
        regimes, which removes PI-induced systematic biases that confound
        the single-model approach.

        Returns [(bp, combined_rms, k_c), ...].
        """
        n = len(deltas)
        hp_offset = hp_setpoints - room_temps
        results: list[tuple[float, float, float]] = []

        for bp in candidates:
            left = deltas < bp
            right = ~left
            n_l, n_r = int(left.sum()), int(right.sum())
            if n_l < self._min_per_side or n_r < self._min_per_side:
                continue

            # Left model: with HP term
            X_l = np.column_stack([
                np.ones(n_l), outdoor_deltas[left],
                hp_offset[left], solar_vals[left],
            ])
            try:
                beta_l, _, _, _ = np.linalg.lstsq(
                    X_l, room_rates[left], rcond=None,
                )
                rms_l = np.mean((room_rates[left] - X_l @ beta_l) ** 2)
                k_c = float(beta_l[2])
            except np.linalg.LinAlgError:
                continue

            # Right model: no HP term
            X_r = np.column_stack([
                np.ones(n_r), outdoor_deltas[right], solar_vals[right],
            ])
            try:
                beta_r, _, _, _ = np.linalg.lstsq(
                    X_r, room_rates[right], rcond=None,
                )
                rms_r = np.mean((room_rates[right] - X_r @ beta_r) ** 2)
            except np.linalg.LinAlgError:
                continue

            # Combined weighted RMS
            combined_rms = float(
                math.sqrt((rms_l * n_l + rms_r * n_r) / n)
            )
            results.append((float(bp), combined_rms, k_c))

        return results

    def _sweep_single(
        self,
        candidates: np.ndarray,
        deltas: np.ndarray,
        outdoor_deltas: np.ndarray,
        hp_setpoints: np.ndarray,
        room_temps: np.ndarray,
        solar_vals: np.ndarray,
        room_rates: np.ndarray,
    ) -> list[tuple[float, float, float]]:
        """Single-model sweep: one OLS with hp_offset = 0 for HP-off side.

        Fallback for large offsets where the split model's separate
        intercepts cause endogeneity-driven negative k_c.  The single
        model shares intercept/ua_c across both sides, which is less
        accurate but more robust to closed-loop confounding.
        """
        n = len(deltas)
        results: list[tuple[float, float, float]] = []

        for bp in candidates:
            hp_on_mask = deltas < bp
            n_l = int(hp_on_mask.sum())
            n_r = n - n_l
            if n_l < self._min_per_side or n_r < self._min_per_side:
                continue

            hp_offset = np.where(
                hp_on_mask,
                hp_setpoints - room_temps,
                0.0,
            )
            X = np.column_stack([
                np.ones(n), outdoor_deltas, hp_offset, solar_vals,
            ])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, room_rates, rcond=None)
                predicted = X @ beta
                rms = float(math.sqrt(np.mean((room_rates - predicted) ** 2)))
                k_c = float(beta[2])
                results.append((float(bp), rms, k_c))
            except np.linalg.LinAlgError:
                continue

        return results

    def _move_toward(self, current: float, target: float) -> float:
        """Move current toward target by at most max_step."""
        diff = target - current
        if abs(diff) <= self._max_step:
            return target
        return current + math.copysign(self._max_step, diff)

    # ── Stall detection ─────────────────────────────────────────────

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

    def reset_stall(self) -> None:
        """Reset stall counter (called after active probe completes)."""
        self._stall_count = 0

    # ── Persistence ─────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Serialize state for persistence across restarts."""
        return {
            "stall_count": self._stall_count,
            "updates_applied": self._updates_applied,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from persisted data."""
        self._stall_count = int(data.get("stall_count", 0))
        self._updates_applied = int(data.get("updates_applied", 0))
