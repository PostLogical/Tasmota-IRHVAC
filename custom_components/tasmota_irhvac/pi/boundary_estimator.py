"""Passive regime boundary estimation from residual rate vs delta.

Estimates where the HP transitions from actively contributing to idle
by detecting a step change in the residual room_rate (after subtracting
modeled environmental effects) as a function of delta = current_c - hp_setpoint.

Three layers:
    1. Per-tick evidence accumulation (add_evidence)
    2. Periodic boundary estimation via median-split breakpoint search
    3. Stall detection → active probe trigger

Literature: Muggeo (2003) segmented regression, Porter & Yu (2015)
regression discontinuity with unknown cutoff.  We use a simpler
median-split approach that is robust to outliers (open windows)
and doesn't require external packages.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class BoundaryEvidence:
    """A single (delta, residual_rate) observation for boundary estimation."""

    delta: float  # current_c - hp_setpoint
    residual_rate: float  # room_rate minus modeled environmental contribution
    timestamp: float  # monotonic time


@dataclass
class BoundaryEstimateResult:
    """Result of a boundary estimation cycle."""

    confident: bool
    estimated_breakpoint: float | None
    gap_magnitude: float
    p_value: float | None
    new_cal_min: float
    new_cal_max: float
    n_observations: int
    n_left: int  # observations left of breakpoint (HP on side)
    n_right: int  # observations right of breakpoint (HP off side)


# ── Constants ───────────────────────────────────────────────────────────

# Delta diversity binning: observations binned by this width,
# capped per bin to prevent steady-state redundancy.
_DELTA_BIN_WIDTH = 0.5  # °C
_MAX_PER_BIN = 20


# ── Estimator ───────────────────────────────────────────────────────────


class BoundaryEstimator:
    """Passive regime boundary estimation from residual rate vs delta.

    Accumulates (delta, residual_rate) evidence per tick and periodically
    estimates the HP on/off transition boundary via a median-split
    breakpoint search.

    The caller (pi_controller) is responsible for:
    - Computing the residual_rate (room_rate - modeled environment)
    - Gating evidence collection (skip anomalies, probes, clamped ticks)
    - Calling estimate_boundary() each batch cycle
    - Applying the returned bound updates
    """

    def __init__(
        self,
        *,
        buffer_max_size: int = 200,
        min_observations: int = 50,
        min_per_side: int = 5,
        min_gap_threshold: float = 0.003,
        max_step_per_update: float = 0.3,
        min_band_width: float = 0.5,
        safety_margin: float = 0.3,
        stall_threshold: int = 3,
        permutation_n: int = 200,
        permutation_p_threshold: float = 0.05,
        rng_seed: int | None = None,
    ):
        """Initialize.

        Args:
            buffer_max_size: Max evidence observations retained.
            min_observations: Minimum evidence needed for estimation.
            min_per_side: Minimum observations on each side of a split.
            min_gap_threshold: Minimum median gap (°C/min) to accept.
            max_step_per_update: Max cal bound movement per batch cycle.
            min_band_width: Floor for cal_max - cal_min.
            safety_margin: Buffer around estimated breakpoint for bounds.
            stall_threshold: Batch cycles without confident estimate
                before should_trigger_probe fires.
            permutation_n: Number of permutations for confidence test.
            permutation_p_threshold: p-value threshold for confidence.
            rng_seed: Seed for permutation test reproducibility.
        """
        self._buffer_max = buffer_max_size
        self._min_observations = min_observations
        self._min_per_side = min_per_side
        self._min_gap = min_gap_threshold
        self._max_step = max_step_per_update
        self._min_band = min_band_width
        self._safety_margin = safety_margin
        self._stall_threshold = stall_threshold
        self._perm_n = permutation_n
        self._perm_p = permutation_p_threshold

        self._buffer: list[BoundaryEvidence] = []
        # Delta diversity tracking: bin -> count
        self._bin_counts: dict[int, int] = {}

        self._stall_count: int = 0
        self._updates_applied: int = 0
        self._last_result: BoundaryEstimateResult | None = None
        self._rng = np.random.default_rng(rng_seed)

    # ── Layer 1: Per-tick evidence ──────────────────────────────────

    def add_evidence(
        self, delta: float, residual_rate: float, timestamp: float
    ) -> None:
        """Add one (delta, residual_rate) observation.

        The caller gates this: no anomalies, no active probe, no clamped,
        no auto-perturbation, outdoor_temp available.
        """
        if not math.isfinite(delta) or not math.isfinite(residual_rate):
            return

        # Delta diversity: bin and cap
        bin_idx = int(math.floor(delta / _DELTA_BIN_WIDTH))
        count = self._bin_counts.get(bin_idx, 0)
        if count >= _MAX_PER_BIN:
            # Bin is full — evict oldest in this bin, then add
            self._evict_oldest_in_bin(bin_idx)
        self._bin_counts[bin_idx] = self._bin_counts.get(bin_idx, 0) + 1

        self._buffer.append(BoundaryEvidence(delta, residual_rate, timestamp))

        # Global cap: evict oldest regardless of bin
        while len(self._buffer) > self._buffer_max:
            evicted = self._buffer.pop(0)
            evicted_bin = int(math.floor(evicted.delta / _DELTA_BIN_WIDTH))
            self._bin_counts[evicted_bin] = max(
                0, self._bin_counts.get(evicted_bin, 1) - 1
            )

    def _evict_oldest_in_bin(self, target_bin: int) -> None:
        """Remove the oldest observation in a specific delta bin."""
        for i, ev in enumerate(self._buffer):
            if int(math.floor(ev.delta / _DELTA_BIN_WIDTH)) == target_bin:
                self._buffer.pop(i)
                self._bin_counts[target_bin] = max(
                    0, self._bin_counts.get(target_bin, 1) - 1
                )
                return

    # ── Layer 2: Boundary estimation ────────────────────────────────

    def estimate_boundary(
        self,
        current_cal_min: float,
        current_cal_max: float,
    ) -> BoundaryEstimateResult:
        """Estimate the HP on/off transition boundary.

        Call every batch cycle.  Returns updated cal bounds if confident.

        Args:
            current_cal_min: Current lower cal bound.
            current_cal_max: Current upper cal bound.

        Returns:
            BoundaryEstimateResult with confident flag and proposed bounds.
        """
        n = len(self._buffer)
        not_confident = BoundaryEstimateResult(
            confident=False,
            estimated_breakpoint=None,
            gap_magnitude=0.0,
            p_value=None,
            new_cal_min=current_cal_min,
            new_cal_max=current_cal_max,
            n_observations=n,
            n_left=0,
            n_right=0,
        )

        if n < self._min_observations:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        # Sort by delta
        sorted_ev = sorted(self._buffer, key=lambda e: e.delta)
        deltas = np.array([e.delta for e in sorted_ev])
        residuals = np.array([e.residual_rate for e in sorted_ev])

        # Find best breakpoint: max median gap
        best_gap = 0.0
        best_split = 0
        min_side = self._min_per_side

        for split in range(min_side, n - min_side):
            left_med = np.median(residuals[:split])
            right_med = np.median(residuals[split:])
            gap = left_med - right_med  # HP on = higher residual
            if gap > best_gap:
                best_gap = gap
                best_split = split

        if best_gap < self._min_gap or best_split == 0:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        breakpoint = float((deltas[best_split - 1] + deltas[best_split]) / 2.0)

        # Permutation test: is this gap significant?
        p_value = self._permutation_test(residuals, best_gap)

        if p_value > self._perm_p:
            self._stall_count += 1
            result = BoundaryEstimateResult(
                confident=False,
                estimated_breakpoint=breakpoint,
                gap_magnitude=best_gap,
                p_value=p_value,
                new_cal_min=current_cal_min,
                new_cal_max=current_cal_max,
                n_observations=n,
                n_left=best_split,
                n_right=n - best_split,
            )
            self._last_result = result
            return result

        # Confident — compute new bounds
        target_min = breakpoint - self._safety_margin
        target_max = breakpoint + self._safety_margin

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
            estimated_breakpoint=breakpoint,
            gap_magnitude=best_gap,
            p_value=p_value,
            new_cal_min=new_min,
            new_cal_max=new_max,
            n_observations=n,
            n_left=best_split,
            n_right=n - best_split,
        )
        self._last_result = result
        return result

    def _move_toward(self, current: float, target: float) -> float:
        """Move current toward target by at most max_step."""
        diff = target - current
        if abs(diff) <= self._max_step:
            return target
        return current + math.copysign(self._max_step, diff)

    def _permutation_test(self, residuals: np.ndarray, observed_gap: float) -> float:
        """Permutation test for breakpoint significance.

        Shuffles residuals and recomputes max gap.  Returns p-value:
        fraction of permuted gaps >= observed gap.
        """
        n = len(residuals)
        min_side = self._min_per_side
        exceedances = 0

        for _ in range(self._perm_n):
            perm = self._rng.permutation(residuals)
            perm_gap = 0.0
            for split in range(min_side, n - min_side):
                left_med = np.median(perm[:split])
                right_med = np.median(perm[split:])
                gap = left_med - right_med
                if gap > perm_gap:
                    perm_gap = gap
            if perm_gap >= observed_gap:
                exceedances += 1

        return (exceedances + 1) / (self._perm_n + 1)

    # ── Layer 3: Stall detection ────────────────────────────────────

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
            "buffer": [
                {"d": e.delta, "r": e.residual_rate, "t": e.timestamp}
                for e in self._buffer
            ],
            "stall_count": self._stall_count,
            "updates_applied": self._updates_applied,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from persisted data."""
        self._buffer.clear()
        self._bin_counts.clear()
        for item in data.get("buffer", []):
            ev = BoundaryEvidence(
                delta=float(item["d"]),
                residual_rate=float(item["r"]),
                timestamp=float(item["t"]),
            )
            bin_idx = int(math.floor(ev.delta / _DELTA_BIN_WIDTH))
            self._bin_counts[bin_idx] = self._bin_counts.get(bin_idx, 0) + 1
            self._buffer.append(ev)
        self._stall_count = int(data.get("stall_count", 0))
        self._updates_applied = int(data.get("updates_applied", 0))
