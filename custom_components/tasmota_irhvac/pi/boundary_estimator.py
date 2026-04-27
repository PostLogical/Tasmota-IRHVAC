"""Passive regime boundary estimation from residual rate vs delta.

Estimates where the HP transitions from actively contributing to idle
by fitting a piecewise linear ("hockey stick") model to the residual
room_rate as a function of delta = current_c - hp_setpoint.

The HP is a proportional controller: its contribution is
k × max(0, setpoint - room) = k × max(0, -delta + offset).
This creates a RAMP on the HP-on side, not a binary step.
The boundary is where the ramp meets the flat HP-off baseline.

Model:  residual = slope × (delta - bp) + baseline   if delta < bp
        residual = baseline                           if delta >= bp

Three layers:
    1. Per-tick evidence accumulation (add_evidence)
    2. Periodic piecewise linear fit via scipy.optimize.curve_fit
    3. Stall detection → active probe trigger

Literature: Muggeo (2003) segmented regression with unknown breakpoint.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

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
    breakpoint_std_err: float | None  # from curve_fit covariance
    slope: float | None  # ramp slope (should be negative in heating)
    baseline: float | None  # flat HP-off residual level
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


# ── Piecewise linear model ─────────────────────────────────────────────


def _hockey_stick(delta: np.ndarray, slope: float, bp: float,
                  baseline: float) -> np.ndarray:
    """Piecewise linear: ramp on HP-on side, flat on HP-off side.

    For heating mode (delta = current_c - hp_setpoint):
      - delta < bp: HP is on, residual = slope × (delta - bp) + baseline
      - delta >= bp: HP is off, residual = baseline

    slope should be negative (residual decreases as delta increases
    toward boundary, because HP contribution shrinks).
    """
    return np.where(delta < bp, slope * (delta - bp) + baseline, baseline)


# ── Estimator ───────────────────────────────────────────────────────────


class BoundaryEstimator:
    """Passive regime boundary estimation from residual rate vs delta.

    Accumulates (delta, residual_rate) evidence per tick and periodically
    fits a piecewise linear model to estimate the HP on/off boundary.

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
        max_step_per_update: float = 0.3,
        min_band_width: float = 0.5,
        safety_margin: float = 0.3,
        stall_threshold: int = 3,
        bp_std_err_max: float = 1.0,
    ):
        """Initialize.

        Args:
            buffer_max_size: Max evidence observations retained.
            min_observations: Minimum evidence needed for estimation.
            min_per_side: Minimum observations on each side of breakpoint.
            max_step_per_update: Max cal bound movement per batch cycle.
            min_band_width: Floor for cal_max - cal_min.
            safety_margin: Buffer around estimated breakpoint for bounds.
            stall_threshold: Batch cycles without confident estimate
                before should_trigger_probe fires.
            bp_std_err_max: Maximum acceptable breakpoint standard error
                (°C) from curve_fit.  Above this, estimate is not confident.
        """
        self._buffer_max = buffer_max_size
        self._min_observations = min_observations
        self._min_per_side = min_per_side
        self._max_step = max_step_per_update
        self._min_band = min_band_width
        self._safety_margin = safety_margin
        self._stall_threshold = stall_threshold
        self._bp_std_err_max = bp_std_err_max

        self._buffer: list[BoundaryEvidence] = []
        # Delta diversity tracking: bin -> count
        self._bin_counts: dict[int, int] = {}

        self._stall_count: int = 0
        self._updates_applied: int = 0
        self._last_result: BoundaryEstimateResult | None = None

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

        Fits a piecewise linear model (ramp + flat) to the accumulated
        evidence.  The breakpoint where the ramp meets the flat baseline
        is the estimated HP transition.

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
            breakpoint_std_err=None,
            slope=None,
            baseline=None,
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

        deltas = np.array([e.delta for e in self._buffer])
        residuals = np.array([e.residual_rate for e in self._buffer])

        # Initial guess: breakpoint at midpoint of current band,
        # slope negative, baseline near median of high-delta residuals.
        bp_guess = (current_cal_min + current_cal_max) / 2.0
        high_mask = deltas > bp_guess
        if high_mask.sum() > 0:
            baseline_guess = float(np.median(residuals[high_mask]))
        else:
            baseline_guess = float(np.median(residuals))
        slope_guess = -0.005  # typical negative slope

        def residual_fn(params: np.ndarray) -> np.ndarray:
            return _hockey_stick(deltas, *params) - residuals

        try:
            result_fit = least_squares(
                residual_fn,
                x0=[slope_guess, bp_guess, baseline_guess],
                loss="huber",
                f_scale=0.005,  # residual scale for Huber (��C/min)
                max_nfev=2000,
            )
        except (RuntimeError, ValueError):
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        if not result_fit.success:
            self._stall_count += 1
            self._last_result = not_confident
            return not_confident

        slope, bp, baseline = result_fit.x
        # Standard errors from Jacobian: J^T J ≈ inverse covariance.
        # Use MAD-based robust variance (consistent with Huber loss)
        # to avoid outlier inflation.
        J = result_fit.jac
        try:
            mad = float(np.median(np.abs(result_fit.fun)))
            r_var = (mad * 1.4826) ** 2  # MAD → σ estimate
            cov = np.linalg.inv(J.T @ J) * r_var
            diag = np.diag(cov)
            if diag[1] > 0:
                bp_se = float(np.sqrt(diag[1]))
            else:
                bp_se = float("inf")
        except np.linalg.LinAlgError:
            bp_se = float("inf")

        # Count observations on each side
        n_left = int((deltas < bp).sum())
        n_right = n - n_left

        # Confidence checks
        confident = (
            bp_se < self._bp_std_err_max
            and n_left >= self._min_per_side
            and n_right >= self._min_per_side
            and slope < 0  # ramp must slope downward (HP on → higher residual at lower delta)
        )

        if not confident:
            self._stall_count += 1
            result = BoundaryEstimateResult(
                confident=False,
                estimated_breakpoint=float(bp),
                breakpoint_std_err=bp_se,
                slope=float(slope),
                baseline=float(baseline),
                new_cal_min=current_cal_min,
                new_cal_max=current_cal_max,
                n_observations=n,
                n_left=n_left,
                n_right=n_right,
            )
            self._last_result = result
            return result

        # Confident — compute new bounds
        target_min = float(bp) - self._safety_margin
        target_max = float(bp) + self._safety_margin

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
            estimated_breakpoint=float(bp),
            breakpoint_std_err=bp_se,
            slope=float(slope),
            baseline=float(baseline),
            new_cal_min=new_min,
            new_cal_max=new_max,
            n_observations=n,
            n_left=n_left,
            n_right=n_right,
        )
        self._last_result = result
        return result

    def _move_toward(self, current: float, target: float) -> float:
        """Move current toward target by at most max_step."""
        diff = target - current
        if abs(diff) <= self._max_step:
            return target
        return current + math.copysign(self._max_step, diff)

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
