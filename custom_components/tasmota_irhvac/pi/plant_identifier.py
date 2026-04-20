"""Plant identification orchestrator.

Coordinates identification providers (step response, area method, etc.)
and maintains the current PlantEstimate.  Derives PI gains via IMC tuning
rules.  Pure computation — no Home Assistant dependencies.

Replaces tau_estimator.py with a multi-provider architecture that
separates τ_fast (for Smith predictor) from τ_slow (for Kp scheduling).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from .plant_model import (
    GainUpdate,
    ObservationContext,
    ParameterEstimate,
    PlantEstimate,
)
from .providers.area_method import AreaMethodProvider
from .providers.step_response import StepResponseProvider

_LOGGER = logging.getLogger(__name__)


class PlantIdentifier:
    """Orchestrates plant identification and derives PI gains.

    Manages one or more identification providers, each estimating
    different plant parameters.  Produces GainUpdate objects consumed
    by the PI controller.
    """

    def __init__(
        self,
        tau_seed: float,
        response_lag: float,
        imc_lambda: float,
    ) -> None:
        self._tau_seed: float = tau_seed
        self._response_lag: float = response_lag
        self._imc_lambda_config: float = imc_lambda
        self._enabled: bool = tau_seed > 0

        # Current best plant estimate — starts from seeds
        self._plant: PlantEstimate = PlantEstimate.from_seeds(
            tau_seed=tau_seed, response_lag=response_lag
        )

        # Providers
        self._step_provider = StepResponseProvider(response_lag=response_lag)
        self._area_provider = AreaMethodProvider(response_lag=response_lag)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def plant(self) -> PlantEstimate:
        """Current best plant estimate."""
        return self._plant

    @property
    def enabled(self) -> bool:
        """Whether plant identification is enabled (tau_seed > 0)."""
        return self._enabled

    @property
    def tau(self) -> float:
        """Backward compat: returns τ_fast value."""
        return self._plant.tau_fast.value

    @property
    def observations(self) -> int:
        """Backward compat: returns τ_fast observation count."""
        return self._plant.tau_fast.observations

    @property
    def active(self) -> bool:
        """Whether any provider has an active observation."""
        return self._step_provider.active or self._area_provider.active

    @property
    def response_lag(self) -> float:
        """Configured HP response lag (minutes)."""
        return self._response_lag

    # ── Observation lifecycle ────────────────────────────────────────

    def start_observation(
        self,
        now_mono: float,
        current_c: float,
        desired_c: float,
        step_magnitude: float,
        ff_offset: float = 0.0,
    ) -> None:
        """Begin a step-response observation, fanning out to all providers."""
        if not self._enabled:
            return
        if abs(step_magnitude) < 1.0:
            return

        ctx = ObservationContext(
            start_time=now_mono,
            baseline_temp=current_c,
            target_temp=desired_c,
            step_magnitude=step_magnitude,
            ff_offset=ff_offset,
        )

        self._step_provider.start_observation(ctx)
        self._area_provider.start_observation(ctx)

    def check_observation(
        self, now_mono: float, current_c: float, ff_offset: float = 0.0
    ) -> GainUpdate | None:
        """Check all providers for new estimates.

        Called every PI tick.  If any provider produces a new estimate,
        updates the PlantEstimate and returns a new GainUpdate.
        """
        if not self._enabled:
            return None

        plant_changed = False

        # Layer 1: step response → τ_fast
        tau_fast_est = self._step_provider.check_observation(
            now_mono, current_c, ff_offset
        )
        if tau_fast_est is not None:
            self._plant = dataclasses.replace(self._plant, tau_fast=tau_fast_est)
            plant_changed = True

        # Layer 2: area method → τ_slow (continues after step response fires)
        tau_slow_est = self._area_provider.accumulate(
            now_mono, current_c, ff_offset,
            tau_fast=self._plant.tau_fast.value,
        )
        if tau_slow_est is not None:
            self._plant = dataclasses.replace(self._plant, tau_slow=tau_slow_est)
            plant_changed = True

        if plant_changed:
            gains = self.compute_gains()
            _LOGGER.info(
                "Plant updated: τ_fast=%.1f (n=%d), τ_slow=%.1f (n=%d). "
                "Kp=%.3f Ki=%.4f",
                self._plant.tau_fast.value, self._plant.tau_fast.observations,
                self._plant.tau_slow.value, self._plant.tau_slow.observations,
                gains.kp, gains.ki,
            )
            return gains

        return None

    def cancel_observation(self) -> None:
        """Cancel all in-progress observations."""
        self._step_provider.cancel_observation()
        self._area_provider.cancel_observation()

    # ── Gain computation ─────────────────────────────────────────────

    def compute_gains(self) -> GainUpdate:
        """Derive Kp and Ki from plant estimate via modified IMC tuning rule.

        IMC for FOPDT: Kp = τ / (K_eff * (λ + L))
        Uses τ_slow for Kp scheduling (dominant dynamics determine loop bandwidth).
        Uses τ_fast for Smith predictor (fast air response for transport delay model).

        Integral time Ti = τ_slow/3 (Skogestad SIMC modification for HVAC:
        faster integral action for disturbance rejection).
        Ki = Kp / Ti = 3 * Kp / τ_slow

        λ defaults to L/3 (bench-validated: 17% ITAE reduction, 0 regressions).
        """
        tau_fast = max(self._plant.tau_fast.value, 1.0)
        tau_slow = max(self._plant.tau_slow.value, 1.0)
        lag = self._response_lag
        lam = self._imc_lambda_config if self._imc_lambda_config > 0 else max(lag / 3.0, 1.0)
        k_eff = max(self._plant.k.value, 0.1)

        # Kp uses tau_slow (dominant dynamics — wall/mass time constant).
        # Until tau_slow is identified (Layer 2), it stays at the seed value,
        # giving stable Kp independent of tau_fast fluctuations.
        tau_for_gains = tau_slow
        kp = tau_for_gains / (k_eff * (lam + lag))
        ti = tau_for_gains / 3.0
        ki = kp / ti

        return GainUpdate(
            kp=kp, ki=ki, tau_fast=tau_fast, tau_slow=tau_slow, lag=lag
        )

    # ── Persistence ──────────────────────────────────────────────────

    def restore(self, data: dict[str, Any]) -> GainUpdate:
        """Restore plant estimate and provider states from persistence.

        Handles migration from old single-tau format.
        """
        # Try to restore full PlantEstimate
        plant_dict = data.get("plant_estimate")
        if plant_dict:
            restored = PlantEstimate.from_dict(plant_dict)
            if restored is not None:
                self._plant = restored

        # Migration: old single-tau data → tau_fast
        elif data.get("tau_fast", 0) > 0 or data.get("tau_estimate", 0) > 0:
            tau_val = float(data.get("tau_fast", 0) or data.get("tau_estimate", 0))
            tau_obs = int(data.get("tau_fast_observations", 0) or data.get("tau_observations", 0))
            tau_slow_val = float(data.get("tau_slow", 0))
            tau_slow_obs = int(data.get("tau_slow_observations", 0))

            self._plant = dataclasses.replace(
                self._plant,
                tau_fast=ParameterEstimate(
                    value=tau_val,
                    confidence=min(1.0, tau_obs / 5.0),
                    source="step_response" if tau_obs > 0 else "seed",
                    observations=tau_obs,
                ),
                **({"tau_slow": ParameterEstimate(
                    value=tau_slow_val,
                    confidence=min(1.0, tau_slow_obs / 5.0),
                    source="area_method" if tau_slow_obs > 0 else "seed",
                    observations=tau_slow_obs,
                )} if tau_slow_val > 0 else {}),
            )

        # Restore provider states
        step_state = data.get("step_provider")
        if step_state:
            self._step_provider.restore(step_state)
        elif self._plant.tau_fast.observations > 0:
            # Migration: sync provider state from restored plant
            self._step_provider.restore({
                "tau_fast": self._plant.tau_fast.value,
                "observations": self._plant.tau_fast.observations,
            })

        area_state = data.get("area_provider")
        if area_state:
            self._area_provider.restore(area_state)
        elif self._plant.tau_slow.observations > 0 and self._plant.tau_slow.source != "seed":
            self._area_provider.restore({
                "tau_slow": self._plant.tau_slow.value,
                "observations": self._plant.tau_slow.observations,
            })

        return self.compute_gains()

    def as_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "plant_estimate": self._plant.as_dict(),
            "step_provider": self._step_provider.as_dict(),
            "area_provider": self._area_provider.as_dict(),
            # Backward compat — old code reads these
            "tau_estimate": self._plant.tau_fast.value,
            "tau_observations": self._plant.tau_fast.observations,
        }
