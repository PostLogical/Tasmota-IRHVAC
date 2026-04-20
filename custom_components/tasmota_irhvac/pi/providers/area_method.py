"""Area-method τ_slow estimation (Layer 2).

Pure computation — no Home Assistant dependencies.  Continues observing
the step response after the 63.2% crossing (which gives τ_fast) and
integrates the remaining response curve to extract τ_slow.

For an SOPDT system, the area under the normalized step response equals
the sum of all time constants plus dead time:

    A = ∫₀^∞ (1 - y_normalized) dt = τ_fast + τ_slow + θ

Since τ_fast is known from Layer 1 and θ is configured, we solve:

    τ_slow = A - τ_fast - θ

The area is computed incrementally using trapezoidal integration,
which is noise-robust (integration averages out sensor noise).
"""

from __future__ import annotations

import logging
from typing import Any

from ..plant_model import ObservationContext, ParameterEstimate

_LOGGER = logging.getLogger(__name__)

# Minimum area observation duration (minutes) before computing τ_slow.
# Need enough of the tail to capture the slow dynamics.
_MIN_AREA_DURATION_MIN = 60.0

# Maximum area observation duration (minutes). Beyond this, the remaining
# area contribution is negligible for any realistic τ_slow.
_MAX_AREA_DURATION_MIN = 480.0  # 8 hours

# Response fraction threshold: stop when response exceeds this fraction
# of the expected change (the remaining area is negligible).
_SETTLING_FRACTION = 0.95

# Minimum consecutive ticks with fraction above settling to declare settled.
_SETTLING_TICKS = 3


class AreaMethodProvider:
    """Estimates τ_slow from the area under the step-response tail."""

    def __init__(self, response_lag: float) -> None:
        self._response_lag: float = response_lag
        self._tau_slow: float = 0.0  # current EMA estimate (0 = no observations)
        self._observations: int = 0

        # Observation state
        self._ctx: ObservationContext | None = None
        self._active: bool = False
        self._area_integral: float = 0.0
        self._last_time: float = 0.0
        self._last_fraction: float = 0.0
        self._settled_ticks: int = 0
        self._peak_fraction: float = 0.0  # track for reversal detection

        # Disturbance gating threshold (same as step response provider)
        self._ff_change_threshold: float = 1.0
        # Outlier rejection factor (same as step response provider)
        self._outlier_factor: float = 3.0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def tau_slow(self) -> float:
        """Current τ_slow estimate (minutes). 0.0 if no observations."""
        return self._tau_slow

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def active(self) -> bool:
        return self._active

    # ── Observation lifecycle ────────────────────────────────────────

    def start_observation(self, ctx: ObservationContext) -> None:
        """Begin area observation alongside step response.

        The area method starts accumulating from the beginning of the step
        response (not from the 63.2% crossing), so it captures the full area.
        """
        if abs(ctx.step_magnitude) < 1.0:
            return
        self._ctx = ctx
        self._active = True
        self._area_integral = 0.0
        self._last_time = ctx.start_time
        self._last_fraction = 0.0
        self._settled_ticks = 0
        self._peak_fraction = 0.0
        _LOGGER.debug(
            "τ_slow area observation started: step=%.1f°C, room=%.1f°C",
            ctx.step_magnitude, ctx.baseline_temp,
        )

    def accumulate(
        self,
        now_mono: float,
        current_c: float,
        ff_offset: float = 0.0,
        tau_fast: float = 0.0,
    ) -> ParameterEstimate | None:
        """Accumulate one tick of area integral.

        Called every PI tick while observation is active.  Uses trapezoidal
        integration for accuracy with ~15-min tick intervals.

        Args:
            tau_fast: Current τ_fast estimate from Layer 1 (needed to
                extract τ_slow = area - τ_fast - θ).

        Returns ParameterEstimate for τ_slow when observation completes,
        else None.
        """
        if not self._active or self._ctx is None:
            return None

        elapsed_min = (now_mono - self._ctx.start_time) / 60.0
        dt_min = (now_mono - self._last_time) / 60.0

        if dt_min <= 0:
            return None

        # Compute normalized response fraction
        expected_change = self._ctx.step_magnitude
        if abs(expected_change) < 0.5:
            self._active = False
            return None

        actual_change = current_c - self._ctx.baseline_temp
        fraction = actual_change / expected_change

        # ── Reversal detection ───────────────────────────────────
        # If the response reverses (fraction drops significantly from peak),
        # the observation is contaminated by a disturbance.
        if fraction > self._peak_fraction:
            self._peak_fraction = fraction
        elif self._peak_fraction > 0.3 and fraction < self._peak_fraction - 0.1:
            _LOGGER.info(
                "τ_slow area observation rejected: response reversed "
                "(peak=%.2f, current=%.2f) — disturbance contamination",
                self._peak_fraction, fraction,
            )
            self._active = False
            return None

        # ── Trapezoidal integration ──────────────────────────────
        # Area under (1 - y_normalized) curve
        integrand_prev = max(0.0, 1.0 - self._last_fraction)
        integrand_now = max(0.0, 1.0 - fraction)
        self._area_integral += 0.5 * (integrand_prev + integrand_now) * dt_min

        self._last_time = now_mono
        self._last_fraction = fraction

        # ── Settling detection ───────────────────────────────────
        if fraction >= _SETTLING_FRACTION:
            self._settled_ticks += 1
        else:
            self._settled_ticks = 0

        # ── Timeout ──────────────────────────────────────────────
        timed_out = elapsed_min >= _MAX_AREA_DURATION_MIN

        # ── Check completion ─────────────────────────────────────
        settled = self._settled_ticks >= _SETTLING_TICKS
        enough_data = elapsed_min >= _MIN_AREA_DURATION_MIN

        if not (settled or timed_out) or not enough_data:
            # Still accumulating
            if timed_out:
                _LOGGER.debug(
                    "τ_slow area observation timed out after %.0f min "
                    "(area=%.1f, fraction=%.2f) — insufficient data",
                    elapsed_min, self._area_integral, fraction,
                )
                self._active = False
            return None

        # ── Disturbance gate ─────────────────────────────────────
        ff_delta = abs(ff_offset - self._ctx.ff_offset)
        if ff_delta > self._ff_change_threshold:
            _LOGGER.info(
                "τ_slow area observation rejected: FF offset changed %.2f°C "
                "(threshold %.1f) — disturbance contamination",
                ff_delta, self._ff_change_threshold,
            )
            self._active = False
            return None

        # ── Extract τ_slow from area ─────────────────────────────
        # For SOPDT: area = τ_fast + τ_slow + θ
        # If response hasn't fully settled, the area underestimates.
        # Correct for remaining tail: if fraction < 1.0, the unobserved
        # area ≈ tau_slow * (1 - fraction) (exponential tail approximation).
        # For settled responses (fraction ≥ 0.95), correction is small.
        raw_area = self._area_integral
        if fraction < 1.0 and fraction > 0.5:
            # Tail correction: remaining area ≈ current tau_slow_estimate * (1-fraction)
            # Bootstrap: use raw_area - tau_fast - theta as initial estimate
            raw_tau_slow = raw_area - tau_fast - self._response_lag
            if raw_tau_slow > 0:
                tail_correction = raw_tau_slow * (1.0 - fraction)
                raw_area += tail_correction

        observed_tau_slow = raw_area - tau_fast - self._response_lag

        # Physical constraint: τ_slow must be positive and ≥ τ_fast
        if observed_tau_slow < tau_fast and tau_fast > 0:
            _LOGGER.info(
                "τ_slow area observation rejected: computed τ_slow=%.1f < τ_fast=%.1f "
                "(area=%.1f, τ_fast=%.1f, θ=%.1f) — likely not a second-order system",
                observed_tau_slow, tau_fast, self._area_integral, tau_fast,
                self._response_lag,
            )
            self._active = False
            return None

        observed_tau_slow = max(observed_tau_slow, 15.0)  # Same floor as τ_fast

        # ── Outlier rejection ────────────────────────────────────
        if self._observations >= 2 and self._tau_slow > 0:
            ratio = observed_tau_slow / self._tau_slow
            if ratio > self._outlier_factor or ratio < 1.0 / self._outlier_factor:
                _LOGGER.info(
                    "τ_slow area observation rejected as outlier: "
                    "observed=%.1f vs estimate=%.1f (ratio=%.2f)",
                    observed_tau_slow, self._tau_slow, ratio,
                )
                self._active = False
                return None

        # ── EMA update ───────────────────────────────────────────
        n = self._observations
        alpha = max(0.3, 1.0 / (2.0 + n))
        if self._tau_slow > 0:
            self._tau_slow = (1.0 - alpha) * self._tau_slow + alpha * observed_tau_slow
        else:
            self._tau_slow = observed_tau_slow
        self._observations += 1
        self._active = False

        confidence = min(1.0, self._observations / 5.0)

        _LOGGER.info(
            "τ_slow observed: %.1f min (area=%.1f, τ_fast=%.1f, θ=%.1f, "
            "fraction=%.2f, %s). EMA: %.1f min (n=%d, α=%.2f)",
            observed_tau_slow, self._area_integral, tau_fast,
            self._response_lag, fraction,
            "settled" if settled else "timeout",
            self._tau_slow, self._observations, alpha,
        )

        return ParameterEstimate(
            value=self._tau_slow,
            confidence=confidence,
            source="area_method",
            observations=self._observations,
        )

    def cancel_observation(self) -> None:
        """Cancel any in-progress observation."""
        if self._active:
            _LOGGER.debug("τ_slow area observation cancelled")
            self._active = False

    # ── Persistence ──────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        return {
            "tau_slow": self._tau_slow,
            "observations": self._observations,
        }

    def restore(self, d: dict[str, Any]) -> None:
        self._tau_slow = float(d.get("tau_slow", 0.0))
        self._observations = int(d.get("observations", 0))
