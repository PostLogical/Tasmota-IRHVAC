"""Step-response τ_fast estimation (Layer 1).

Pure computation — no Home Assistant dependencies.  Observes room temperature
response to HP setpoint changes and estimates τ_fast (the fast/air time
constant) from the 63.2% crossing of the step response.

Extracted from tau_estimator.py — same logic, adapted to the provider
interface (ObservationContext in, ParameterEstimate out).
"""

from __future__ import annotations

import logging
from typing import Any

from ..plant_model import ObservationContext, ParameterEstimate

_LOGGER = logging.getLogger(__name__)


class StepResponseProvider:
    """Estimates τ_fast from step-response 63.2% crossing."""

    def __init__(self, response_lag: float) -> None:
        self._response_lag: float = response_lag
        self._tau_fast: float = 0.0  # current EMA estimate (0 = no observations)
        self._observations: int = 0

        # Observation state
        self._ctx: ObservationContext | None = None
        self._active: bool = False

        # Outlier rejection: observed τ must be within this factor of current
        # estimate to be accepted (e.g., 3.0 means 1/3× to 3× current τ).
        self._outlier_factor: float = 3.0
        # Disturbance gating: reject observation if FF offset changed by more
        # than this threshold (°C) during the observation window.
        self._ff_change_threshold: float = 1.0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def tau_fast(self) -> float:
        """Current τ_fast estimate (minutes). 0.0 if no observations."""
        return self._tau_fast

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def active(self) -> bool:
        return self._active

    # Test access
    @property
    def step_temp(self) -> float | None:
        return self._ctx.baseline_temp if self._ctx else None

    @property
    def step_target(self) -> float | None:
        return self._ctx.target_temp if self._ctx else None

    @property
    def step_magnitude(self) -> float:
        return self._ctx.step_magnitude if self._ctx else 0.0

    # ── Observation lifecycle ────────────────────────────────────────

    def start_observation(self, ctx: ObservationContext) -> None:
        """Begin observing a step response for τ_fast estimation.

        Called by the orchestrator when hp_setpoint changes by ≥1°C.
        """
        if abs(ctx.step_magnitude) < 1.0:
            return
        self._ctx = ctx
        self._active = True
        _LOGGER.debug(
            "τ_fast observation started: step=%.1f°C, room=%.1f°C, "
            "target=%.1f°C, ff=%.2f",
            ctx.step_magnitude, ctx.baseline_temp, ctx.target_temp, ctx.ff_offset,
        )

    def check_observation(
        self, now_mono: float, current_c: float, ff_offset: float = 0.0
    ) -> ParameterEstimate | None:
        """Check if the room has reached 63.2% of the step response.

        Returns ParameterEstimate for τ_fast if threshold crossed, else None.
        Observations are rejected if:
        - FF offset changed significantly (disturbance contamination)
        - Observed τ is an outlier vs. current estimate (≥2 prior observations)
        """
        if not self._active or self._ctx is None:
            return None

        elapsed_min = (now_mono - self._ctx.start_time) / 60.0

        # Timeout: 4× current estimate or 4 hours
        ref_tau = self._tau_fast if self._tau_fast > 0 else 60.0
        timeout = max(4.0 * ref_tau, 240.0)
        if elapsed_min > timeout:
            _LOGGER.debug("τ_fast observation timed out after %.0f min", elapsed_min)
            self._active = False
            return None

        expected_change = self._ctx.step_magnitude  # K_eff = 1.0
        if abs(expected_change) < 0.5:
            self._active = False
            return None

        actual_change = current_c - self._ctx.baseline_temp
        fraction = actual_change / expected_change

        # 63.2% threshold (1 - 1/e)
        if fraction >= 0.632:
            raw_tau = elapsed_min - self._response_lag
            observed_tau = max(raw_tau, 15.0)  # Floor: no room has τ < 15 min

            # ── Disturbance gate ─────────────────────────────────────
            ff_delta = abs(ff_offset - self._ctx.ff_offset)
            if ff_delta > self._ff_change_threshold:
                _LOGGER.info(
                    "τ_fast observation rejected: FF offset changed %.2f°C "
                    "(threshold %.1f) — disturbance contamination",
                    ff_delta, self._ff_change_threshold,
                )
                self._active = False
                return None

            # ── Outlier rejection ────────────────────────────────────
            if self._observations >= 2 and self._tau_fast > 0:
                ratio = observed_tau / self._tau_fast
                if ratio > self._outlier_factor or ratio < 1.0 / self._outlier_factor:
                    _LOGGER.info(
                        "τ_fast observation rejected as outlier: observed=%.1f min "
                        "vs estimate=%.1f min (ratio=%.2f, limit=%.1f×)",
                        observed_tau, self._tau_fast, ratio, self._outlier_factor,
                    )
                    self._active = False
                    return None

            # ── EMA update ───────────────────────────────────────────
            n = self._observations
            alpha = max(0.3, 1.0 / (2.0 + n))
            if self._tau_fast > 0:
                self._tau_fast = (1.0 - alpha) * self._tau_fast + alpha * observed_tau
            else:
                # First observation with no prior estimate — use directly
                self._tau_fast = observed_tau
            self._observations += 1
            self._active = False

            confidence = min(1.0, self._observations / 5.0)

            _LOGGER.info(
                "τ_fast observed: %.1f min (raw=%.1f, lag=%.1f, ff_Δ=%.2f). "
                "EMA: %.1f min (n=%d, α=%.2f)",
                observed_tau, elapsed_min, self._response_lag, ff_delta,
                self._tau_fast, self._observations, alpha,
            )

            return ParameterEstimate(
                value=self._tau_fast,
                confidence=confidence,
                source="step_response",
                observations=self._observations,
            )

        return None

    def cancel_observation(self) -> None:
        """Cancel any in-progress observation."""
        if self._active:
            _LOGGER.debug("τ_fast observation cancelled")
            self._active = False

    # ── Persistence ──────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau_fast": self._tau_fast,
            "observations": self._observations,
        }

    def restore(self, d: dict[str, Any]) -> None:
        self._tau_fast = float(d.get("tau_fast", 0.0))
        self._observations = int(d.get("observations", 0))
