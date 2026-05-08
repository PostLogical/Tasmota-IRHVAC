"""Closed-loop SOPDT identification (Option 5 — cross-check).

Pure computation — no Home Assistant dependencies.  Fits an SOPDT plant
model to observed closed-loop data by accounting for the PI controller's
effect on the HP setpoint trajectory.

Unlike the step response and area method providers (which assume a
constant HP setpoint during the observation), this provider records the
actual HP setpoint at each tick and fits the plant model to the full
input/output trajectory.  This corrects for the PI controller modifying
the setpoint during the observation.

Method: time-domain grid search over (τ_fast, τ_slow) with K solved
analytically as a linear least-squares coefficient.  θ is fixed at the
configured response_lag.  No scipy required — runs in milliseconds.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..plant_model import ObservationContext, ParameterEstimate

_LOGGER = logging.getLogger(__name__)

# Grid search values for τ_fast and τ_slow (minutes)
_TAU_FAST_GRID = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]
_TAU_SLOW_GRID = [30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 200.0, 300.0]

# Minimum observation duration before running the fit (minutes)
_MIN_DURATION_MIN = 60.0

# Maximum observation duration (minutes)
_MAX_DURATION_MIN = 480.0

# Minimum number of data points for a reliable fit.
# With 15-min ticks, 10 points = 2.5 hours — need enough to see both
# fast and slow dynamics.
_MIN_DATA_POINTS = 10

# Minimum R² for accepting the fit.  Higher threshold than typical because
# the grid search is coarse and we want high-confidence cross-checks.
_MIN_R_SQUARED = 0.8


def _sopdt_step(t_min: float, tau_fast: float, tau_slow: float) -> float:
    """Unit SOPDT step response at time t (minutes).

    Returns the normalized response y/K for a unit step input.
    For G(s) = K / [(τ_fast·s+1)(τ_slow·s+1)]:
    y(t)/K = 1 - τ_slow/(τ_slow-τ_fast)·exp(-t/τ_slow) + τ_fast/(τ_slow-τ_fast)·exp(-t/τ_fast)
    """
    if t_min <= 0:
        return 0.0
    if abs(tau_fast - tau_slow) < 0.1:
        # Degenerate: single time constant
        return 1.0 - math.exp(-t_min / tau_fast)
    e_fast = math.exp(-t_min / tau_fast) if tau_fast > 0 else 0.0
    e_slow = math.exp(-t_min / tau_slow) if tau_slow > 0 else 0.0
    return 1.0 - (tau_slow * e_slow - tau_fast * e_fast) / (tau_slow - tau_fast)


class ClosedLoopProvider:
    """Fits SOPDT plant model to closed-loop observation data.

    Accumulates (time, room_temp, hp_setpoint) tuples during a
    step-response observation, then fits (K, τ_fast, τ_slow) via
    grid search when the observation window closes.
    """

    def __init__(self, response_lag: float) -> None:
        self._response_lag: float = response_lag

        # Observation state
        self._ctx: ObservationContext | None = None
        self._active: bool = False
        self._data: list[tuple[float, float, float]] = []  # (time_min, room_c, hp_setpoint_c)
        self._last_hp_setpoint: float | None = None

        # Disturbance gating
        self._ff_change_threshold: float = 1.0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._active

    # ── Observation lifecycle ────────────────────────────────────────

    def start_observation(self, ctx: ObservationContext) -> None:
        """Begin accumulating closed-loop data."""
        if abs(ctx.step_magnitude) < 1.0:
            return
        self._ctx = ctx
        self._active = True
        self._data = []
        self._last_hp_setpoint = None
        _LOGGER.debug(
            "Closed-loop observation started: step=%.1f°C, room=%.1f°C",
            ctx.step_magnitude, ctx.baseline_temp,
        )

    def accumulate(
        self,
        now_mono: float,
        current_c: float,
        hp_setpoint_c: float | None = None,
        ff_offset: float = 0.0,
    ) -> list[ParameterEstimate] | None:
        """Accumulate one tick of closed-loop data.

        Returns list of [tau_fast_est, tau_slow_est] when the observation
        completes and the fit succeeds, else None.
        """
        if not self._active or self._ctx is None:
            return None
        if hp_setpoint_c is None:
            return None  # Need HP setpoint data

        elapsed_min = (now_mono - self._ctx.start_time) / 60.0

        # Record data point
        self._data.append((elapsed_min, current_c, float(hp_setpoint_c)))
        self._last_hp_setpoint = hp_setpoint_c

        # Timeout
        if elapsed_min > _MAX_DURATION_MIN:
            _LOGGER.debug("Closed-loop observation timed out after %.0f min", elapsed_min)
            self._active = False
            return self._try_fit()

        # Check for settling (same as area method: response near steady state)
        if elapsed_min >= _MIN_DURATION_MIN and len(self._data) >= _MIN_DATA_POINTS:
            expected_change = self._ctx.step_magnitude
            if abs(expected_change) > 0.5:  # pragma: no branch — expected_change small — settling check skipped
                actual_change = current_c - self._ctx.baseline_temp
                fraction = actual_change / expected_change
                if fraction >= 0.90:
                    # Response appears settled — try to fit
                    return self._try_fit()

        return None

    def cancel_observation(self) -> None:
        """Cancel any in-progress observation."""
        if self._active:
            _LOGGER.debug("Closed-loop observation cancelled")
            self._active = False
            self._data = []

    # ── Grid search fitting ──────────────────────────────────────────

    def _try_fit(self) -> list[ParameterEstimate] | None:
        """Run grid search to fit SOPDT parameters."""
        self._active = False

        if len(self._data) < _MIN_DATA_POINTS:
            _LOGGER.debug(
                "Closed-loop fit: insufficient data (%d points)", len(self._data)
            )
            return None

        # Extract input changes (Δu) from the HP setpoint trajectory
        # Each Δu is a setpoint change relative to the initial setpoint
        t0 = self._data[0][0]
        u_initial = self._data[0][2]
        y_initial = self._data[0][1]

        # Build input change events: (time_min, delta_u)
        input_changes: list[tuple[float, float]] = []
        prev_u = u_initial
        for t, _y, u in self._data:
            if u != prev_u:
                input_changes.append((t, u - prev_u))
                prev_u = u
        # Include the initial step
        if self._ctx is not None:
            input_changes.insert(0, (0.0, self._ctx.step_magnitude))

        if not input_changes:
            return None

        # Observed output (relative to initial)
        y_obs = [y - y_initial for _t, y, _u in self._data]
        t_obs = [t - t0 for t, _y, _u in self._data]

        # Grid search: find (τ_fast, τ_slow) that minimizes residual
        best_rss = float("inf")
        best_tau_fast = 15.0
        best_tau_slow = 60.0
        best_k = 1.0
        best_r2 = 0.0

        y_mean = sum(y_obs) / len(y_obs) if y_obs else 0.0
        ss_tot = sum((y - y_mean) ** 2 for y in y_obs)
        if ss_tot < 1e-10:
            return None  # No variation in output

        for tau_fast in _TAU_FAST_GRID:
            for tau_slow in _TAU_SLOW_GRID:
                if tau_slow <= tau_fast:
                    continue  # τ_slow must be > τ_fast

                # Simulate the predicted output for unit K
                y_pred_unit: list[float] = []
                for t in t_obs:
                    y_sum = 0.0
                    for t_change, delta_u in input_changes:
                        dt = t - t_change - self._response_lag
                        if dt > 0:
                            y_sum += delta_u * _sopdt_step(dt, tau_fast, tau_slow)
                    y_pred_unit.append(y_sum)

                # Solve K analytically: K = Σ(y_obs × y_pred) / Σ(y_pred²)
                num = sum(yo * yp for yo, yp in zip(y_obs, y_pred_unit))
                den = sum(yp * yp for yp in y_pred_unit)
                if den < 1e-10:
                    continue
                k = num / den
                if k <= 0:
                    continue  # Non-physical (anti-correlated data)

                # Compute residual sum of squares
                rss = sum(
                    (yo - k * yp) ** 2
                    for yo, yp in zip(y_obs, y_pred_unit)
                )

                if rss < best_rss:
                    best_rss = rss
                    best_tau_fast = tau_fast
                    best_tau_slow = tau_slow
                    best_k = k
                    best_r2 = 1.0 - rss / ss_tot

        if best_r2 < _MIN_R_SQUARED:
            _LOGGER.info(
                "Closed-loop fit rejected: R²=%.3f < %.1f (τ_fast=%.0f, τ_slow=%.0f, K=%.2f)",
                best_r2, _MIN_R_SQUARED, best_tau_fast, best_tau_slow, best_k,
            )
            return None

        _LOGGER.info(
            "Closed-loop fit: τ_fast=%.0f min, τ_slow=%.0f min, K=%.3f, "
            "R²=%.3f (%d data points)",
            best_tau_fast, best_tau_slow, best_k, best_r2, len(self._data),
        )

        return [
            ParameterEstimate(
                value=best_tau_fast,
                confidence=min(1.0, best_r2),
                source="closed_loop",
                observations=len(self._data),
            ),
            ParameterEstimate(
                value=best_tau_slow,
                confidence=min(1.0, best_r2),
                source="closed_loop",
                observations=len(self._data),
            ),
        ]

    # ── Persistence ──────────────────────────────────────────────────
    # Not persisted — observations are transient. The results flow into
    # PlantEstimate which IS persisted.
