"""Online τ estimation via step-response observation + IMC gain scheduling.

Pure computation — no Home Assistant dependencies.  Observes room temperature
response to HP setpoint changes and estimates the first-order time constant τ.
Derives PI gains (Kp, Ki) from τ using modified IMC tuning rules.
"""

from __future__ import annotations

import dataclasses
import logging
import math

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GainUpdate:
    """Result of a gain recomputation."""

    kp: float
    ki: float
    tau: float
    lag: float


class TauEstimator:
    """Online τ estimation via step-response observation + IMC gain scheduling."""

    def __init__(
        self,
        tau_seed: float,
        response_lag: float,
        imc_lambda: float,
    ) -> None:
        self._tau_seed: float = tau_seed
        self._response_lag: float = response_lag
        self._imc_lambda_config: float = imc_lambda
        self._tau_estimate: float = tau_seed
        self._enabled: bool = tau_seed > 0

        # Step-response observation state
        self._step_time: float = 0.0
        self._step_temp: float | None = None
        self._step_target: float | None = None
        self._step_magnitude: float = 0.0
        self._step_ff_offset: float = 0.0  # FF offset at step start
        self._step_active: bool = False
        self._observations: int = 0

        # Outlier rejection: observed τ must be within this factor of current
        # estimate to be accepted (e.g., 3.0 means 1/3× to 3× current τ).
        self._outlier_factor: float = 3.0
        # Disturbance gating: reject observation if FF offset changed by more
        # than this threshold (°C) during the observation window.
        self._ff_change_threshold: float = 1.0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def tau(self) -> float:
        """Current τ estimate (minutes)."""
        return self._tau_estimate

    @tau.setter
    def tau(self, value: float) -> None:
        self._tau_estimate = value

    @property
    def observations(self) -> int:
        """Number of τ observations made."""
        return self._observations

    @property
    def active(self) -> bool:
        """Whether a step-response observation is in progress."""
        return self._step_active

    @property
    def enabled(self) -> bool:
        """Whether IMC gain scheduling is enabled (tau_seed > 0)."""
        return self._enabled

    @property
    def response_lag(self) -> float:
        """Configured HP response lag (minutes)."""
        return self._response_lag

    # Step-response state (for test access / diagnostics)
    @property
    def step_temp(self) -> float | None:
        return self._step_temp

    @property
    def step_target(self) -> float | None:
        return self._step_target

    @property
    def step_magnitude(self) -> float:
        return self._step_magnitude

    # ── Gain computation ─────────────────────────────────────────────

    def compute_gains(self) -> GainUpdate:
        """Derive Kp and Ki from τ estimate via modified IMC tuning rule.

        IMC for first-order + delay (FOPDT):
            Kp = τ / (K_eff * (λ + L))

        Integral time uses Ti = τ/3 instead of standard Ti = τ (Skogestad SIMC
        modification for HVAC: faster integral action for disturbance rejection,
        since HVAC systems face continuous disturbances from weather/occupancy):
            Ki = Kp / (τ/3) = 3 * Kp / τ

        K_eff = 1.0 (unit gain: 1°C HP setpoint offset → 1°C room temp at SS).
        L = HP response lag (compressor → room sensor, config, default 15 min).

        λ (closed-loop speed) defaults to L/3 when not explicitly configured.
        Bench sweep across 3 house profiles (τ=25,50,120) × 6 scenarios showed:
        - λ=τ/2 (old default) too conservative: gains barely differ from flat Kp=1.5
        - λ=L/3≈5 gives 17% aggregate ITAE reduction vs flat gains, 0 regressions
        - Biggest win on well-insulated (τ=120): 64% ITAE reduction (Kp 1.5→6.0)
        - λ tied to L (not τ) because the transport delay is the physical constraint
          on how aggressively we can close the loop, regardless of house thermal mass
        Override via pi_imc_lambda config for manual tuning.
        """
        tau = max(self._tau_estimate, 1.0)  # Floor at 1 min to avoid division issues
        lag = self._response_lag
        lam = self._imc_lambda_config if self._imc_lambda_config > 0 else lag / 3.0
        k_eff = 1.0

        kp = tau / (k_eff * (lam + lag))
        ti = tau / 3.0  # Aggressive integral time for HVAC disturbance rejection
        ki = kp / ti

        return GainUpdate(kp=kp, ki=ki, tau=tau, lag=lag)

    # ── Step-response observation ────────────────────────────────────

    def start_observation(
        self,
        now_mono: float,
        current_c: float,
        desired_c: float,
        step_magnitude: float,
        ff_offset: float = 0.0,
    ) -> None:
        """Begin observing a step response for τ estimation.

        Called when hp_setpoint changes by ≥1°C.  Records the starting conditions
        so check_observation can detect when the room reaches 63.2% of the
        expected response.  Also records the FF offset so we can reject
        observations contaminated by large disturbance changes.
        """
        if not self._enabled:
            return
        # Only observe steps with clear direction and magnitude
        if abs(step_magnitude) < 1.0:
            return
        self._step_time = now_mono
        self._step_temp = current_c
        self._step_target = desired_c
        self._step_magnitude = step_magnitude
        self._step_ff_offset = ff_offset
        self._step_active = True
        _LOGGER.debug(
            "τ observation started: step=%.1f°C, room=%.1f°C, target=%.1f°C, ff=%.2f",
            step_magnitude, current_c, desired_c, ff_offset,
        )

    def check_observation(
        self, now_mono: float, current_c: float, ff_offset: float = 0.0
    ) -> GainUpdate | None:
        """Check if the room has reached 63.2% of the step response.

        τ is the time from setpoint change to 63.2% of the total expected
        temperature change (first-order system definition).  On observation,
        update the running τ estimate with an EMA.

        Observations are rejected if:
        - FF offset changed significantly (disturbance contamination)
        - Observed τ is an outlier vs. current estimate (≥2 prior observations)

        Returns GainUpdate if τ changed and gains were recomputed, else None.
        """
        if not self._step_active or self._step_temp is None:
            return None

        elapsed_min = (now_mono - self._step_time) / 60.0

        # Timeout: if we haven't seen 63.2% response in 4× the current estimate
        # (or 4 hours if no estimate), abandon this observation.
        timeout = max(4.0 * self._tau_estimate, 240.0) if self._tau_estimate > 0 else 240.0
        if elapsed_min > timeout:
            _LOGGER.debug("τ observation timed out after %.0f min", elapsed_min)
            self._step_active = False
            return None

        # Expected total change: step drives room from step_temp toward target.
        # For FOPDT, the expected SS change = step_magnitude * K_eff (K_eff=1).
        expected_change = self._step_magnitude  # K_eff = 1.0
        if abs(expected_change) < 0.5:
            self._step_active = False
            return None

        actual_change = current_c - self._step_temp
        fraction = actual_change / expected_change

        # 63.2% threshold (1 - 1/e)
        if fraction >= 0.632:
            # Subtract response lag — τ is the thermal time constant, not
            # including the HP's own delay to start affecting room temp.
            raw_tau = elapsed_min - self._response_lag
            observed_tau = max(raw_tau, 15.0)  # Floor: no room has τ < 15 min

            # ── Disturbance gate ─────────────────────────────────────
            # Reject observations where the FF offset changed significantly
            # during the window — the step response is contaminated by
            # concurrent disturbance changes (solar, boiler, outdoor rate).
            ff_delta = abs(ff_offset - self._step_ff_offset)
            if ff_delta > self._ff_change_threshold:
                _LOGGER.info(
                    "τ observation rejected: FF offset changed %.2f°C "
                    "(threshold %.1f) during observation — disturbance "
                    "contamination",
                    ff_delta, self._ff_change_threshold,
                )
                self._step_active = False
                return None

            # ── Outlier rejection ────────────────────────────────────
            # After ≥2 observations, reject values that are >3× or <1/3×
            # the current estimate.  A single bad observation (e.g., from
            # integral windup driving a fast apparent response) shouldn't
            # destroy a well-established estimate.
            if self._observations >= 2:
                ratio = observed_tau / self._tau_estimate if self._tau_estimate > 0 else 1.0
                if ratio > self._outlier_factor or ratio < 1.0 / self._outlier_factor:
                    _LOGGER.info(
                        "τ observation rejected as outlier: observed=%.1f min "
                        "vs estimate=%.1f min (ratio=%.2f, limit=%.1f×)",
                        observed_tau, self._tau_estimate, ratio, self._outlier_factor,
                    )
                    self._step_active = False
                    return None

            # ── EMA update ───────────────────────────────────────────
            # Weight new observations more when we have few.
            # +2 offset ensures the seed is never obliterated on a single
            # observation (n=0 → α=0.5, n=1 → α≈0.33, converges to 0.3).
            n = self._observations
            alpha = max(0.3, 1.0 / (2.0 + n))  # Starts at 0.5, decays to 0.3
            old_tau = self._tau_estimate
            self._tau_estimate = (1.0 - alpha) * old_tau + alpha * observed_tau
            self._observations += 1
            self._step_active = False

            gains = self.compute_gains()

            _LOGGER.info(
                "τ observed: %.1f min (raw=%.1f, lag=%.1f, ff_Δ=%.2f). "
                "EMA τ: %.1f → %.1f min (n=%d, α=%.2f). Kp=%.3f Ki=%.4f",
                observed_tau, elapsed_min, self._response_lag, ff_delta,
                old_tau, self._tau_estimate, self._observations, alpha,
                gains.kp, gains.ki,
            )

            return gains

        return None

    def cancel_observation(self) -> None:
        """Cancel any in-progress τ observation (e.g., mode change, setpoint change)."""
        if self._step_active:
            _LOGGER.debug("τ observation cancelled")
            self._step_active = False

    # ── Persistence ──────────────────────────────────────────────────

    def restore(self, tau_estimate: float) -> GainUpdate:
        """Restore τ estimate and return recomputed gains.

        Caller is responsible for applying gains to PI state and Smith predictor.
        """
        self._tau_estimate = tau_estimate
        return self.compute_gains()
