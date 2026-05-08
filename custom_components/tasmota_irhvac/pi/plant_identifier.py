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
    PlantTestCommand,
)
from .providers.area_method import AreaMethodProvider
from .providers.closed_loop import ClosedLoopProvider
from .providers.plant_test import PlantTestProvider
from .providers.step_response import StepResponseProvider

_LOGGER = logging.getLogger(__name__)


class PlantIdentifier:
    """Orchestrates plant identification and derives PI gains.

    Manages one or more identification providers, each estimating
    different plant parameters.  Produces GainUpdate objects consumed
    by the PI controller.
    """

    # Minimum non-seed observations before τ from plant ID may influence
    # IMC gains.  Below this threshold, compute_gains() falls back to the
    # conservative seed.  Skogestad SIMC + cautious adaptation pattern —
    # don't apply gain updates derived from a single observation.
    MIN_TAU_OBSERVATIONS_FOR_GATE = 3

    def __init__(
        self,
        tau_fast_seed: float,
        tau_slow_seed: float,
        response_lag: float,
        imc_lambda: float,
        enabled: bool = True,
    ) -> None:
        self._tau_fast_seed: float = tau_fast_seed
        self._tau_slow_seed: float = tau_slow_seed
        self._response_lag: float = response_lag
        self._imc_lambda_config: float = imc_lambda
        self._enabled: bool = enabled

        # Current best plant estimate — starts from seeds
        self._plant: PlantEstimate = PlantEstimate.from_seeds(
            tau_fast_seed=tau_fast_seed,
            tau_slow_seed=tau_slow_seed,
            response_lag=response_lag,
        )

        # Providers
        self._step_provider = StepResponseProvider(response_lag=response_lag)
        self._area_provider = AreaMethodProvider(response_lag=response_lag)
        self._closed_loop_provider = ClosedLoopProvider(response_lag=response_lag)
        self._last_cross_check: tuple[ParameterEstimate, ParameterEstimate] | None = None
        self._plant_test: PlantTestProvider | None = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def plant(self) -> PlantEstimate:
        """Current best plant estimate."""
        return self._plant

    @property
    def enabled(self) -> bool:
        """Whether plant identification (IMC formula) is enabled."""
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
        return (
            self._step_provider.active
            or self._area_provider.active
            or self.plant_test_active
        )

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
        self._closed_loop_provider.start_observation(ctx)

    def check_observation(
        self,
        now_mono: float,
        current_c: float,
        ff_offset: float = 0.0,
        hp_setpoint_c: float | None = None,
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

        # Option 5: closed-loop identification — cross-check + interim estimates.
        cl_results = self._closed_loop_provider.accumulate(
            now_mono, current_c, hp_setpoint_c=hp_setpoint_c, ff_offset=ff_offset,
        )
        if cl_results is not None and len(cl_results) >= 2:
            cl_tau_fast, cl_tau_slow = cl_results[0], cl_results[1]
            self._last_cross_check = (cl_tau_fast, cl_tau_slow)

            # (1) Cross-validate: log agreement/disagreement
            agreement = self._cross_validate(cl_tau_fast, cl_tau_slow)

            # (2) Boost/reduce confidence on primary estimates
            self._adjust_confidence(agreement)

            # (3) Interim estimate: if primary hasn't fired AND the
            # closed-loop Kp change would be modest (< 50%), use it.
            # This fills the gap between seed and first area method observation.
            if self._plant.tau_slow.source == "seed" and cl_tau_slow.confidence >= 0.8:  # pragma: no branch — closed-loop bridge update — defensive on source/confidence combo
                ratio = cl_tau_slow.value / self._plant.tau_slow.value
                if 0.5 <= ratio <= 2.0:  # Modest change only
                    interim = ParameterEstimate(
                        value=cl_tau_slow.value,
                        confidence=cl_tau_slow.confidence * 0.7,  # Discount vs primary
                        source="closed_loop",
                        observations=cl_tau_slow.observations,
                    )
                    self._plant = dataclasses.replace(self._plant, tau_slow=interim)
                    plant_changed = True
                    _LOGGER.info(
                        "Closed-loop interim τ_slow=%.0f (primary not yet available, "
                        "ratio=%.2f from seed=%.0f)",
                        cl_tau_slow.value, ratio, self._plant.tau_slow.value,
                    )
                else:
                    _LOGGER.info(
                        "Closed-loop τ_slow=%.0f rejected as interim: ratio=%.2f "
                        "from seed=%.0f too large",
                        cl_tau_slow.value, ratio, self._plant.tau_slow.value,
                    )

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
        self._closed_loop_provider.cancel_observation()

    def reset(self) -> None:
        """Reset plant estimate to seeds and cancel all observations."""
        self.abort_plant_test()
        self.cancel_observation()
        self._plant = PlantEstimate.from_seeds(
            tau_fast_seed=self._tau_fast_seed,
            tau_slow_seed=self._tau_slow_seed,
            response_lag=self._response_lag,
        )
        self._last_cross_check = None

    def _cross_validate(
        self, cl_tau_fast: ParameterEstimate, cl_tau_slow: ParameterEstimate
    ) -> dict[str, bool]:
        """Cross-validate closed-loop against primary estimates.

        Returns dict of {"tau_fast": agrees, "tau_slow": agrees}.
        Logs agreement/disagreement with 30% tolerance.
        """
        agreement: dict[str, bool] = {}

        for name, cl_est, primary in [
            ("τ_fast", cl_tau_fast, self._plant.tau_fast),
            ("τ_slow", cl_tau_slow, self._plant.tau_slow),
        ]:
            if primary.source == "seed" or primary.value <= 0:
                agreement[name] = True  # No primary to compare against
                continue
            ratio = cl_est.value / primary.value
            agrees = 0.7 <= ratio <= 1.3
            agreement[name] = agrees
            if agrees:
                _LOGGER.info(
                    "Cross-validation: %s agrees (CL=%.0f vs %s=%.0f, ratio=%.2f)",
                    name, cl_est.value, primary.source, primary.value, ratio,
                )
            else:
                _LOGGER.warning(
                    "Cross-validation: %s DISAGREES (CL=%.0f vs %s=%.0f, ratio=%.2f)"
                    " — plant model may have changed",
                    name, cl_est.value, primary.source, primary.value, ratio,
                )

        return agreement

    def _adjust_confidence(self, agreement: dict[str, bool]) -> None:
        """Adjust PlantEstimate confidence based on cross-validation.

        Agreement boosts confidence toward 1.0. Disagreement reduces it.
        Only adjusts parameters that have primary (non-seed) estimates.
        """
        for name, field in [("τ_fast", "tau_fast"), ("τ_slow", "tau_slow")]:
            current: ParameterEstimate = getattr(self._plant, field)
            if current.source == "seed":
                continue

            agrees = agreement.get(name, True)
            if agrees:
                # Boost: nudge toward 1.0
                new_conf = min(1.0, current.confidence + 0.1)
            else:
                # Reduce: nudge toward 0.5 (don't go too low — primary is still best)
                new_conf = max(0.5, current.confidence - 0.15)

            if new_conf != current.confidence:  # pragma: no branch — new_conf == current.confidence — boundary
                updated = ParameterEstimate(
                    value=current.value,
                    confidence=new_conf,
                    source=current.source,
                    observations=current.observations,
                )
                self._plant = dataclasses.replace(self._plant, **{field: updated})

    # ── Grey-box τ provider (Layer 4) ─────────────────────────────────

    def update_from_greybox(
        self,
        tau_eff: float,
        ua_c_cv: float,
        tau_fast: float | None = None,
        tau_fast_cv: float | None = None,
    ) -> GainUpdate | None:
        """Accept grey-box τ estimates and update plant state if appropriate.

        ``tau_eff`` always updates tau_slow (1R1C: the only τ; 2R2C: τ_slow).
        When ``tau_fast`` is provided (2R2C only), tau_fast is also updated.

        Confidence derived from coefficient of variation:
            confidence = max(0, 1 - 2×CV).  CV=0→conf=1, CV≥0.5→conf=0.

        tau_slow accepts grey-box updates over seed / closed_loop / prior
        greybox; it never overrides area_method or step_response primaries.
        tau_fast accepts grey-box updates over seed / prior greybox only —
        the step_response primary keeps precedence.

        Both fields enforce a modest-change ratio (0.5×–2× of current).
        """
        if not self._enabled or tau_eff <= 0:
            return None

        slow_updated = self._apply_greybox_update(
            field_name="tau_slow",
            new_value=tau_eff,
            cv=ua_c_cv,
            overridable={"seed", "closed_loop", "greybox"},
            label="τ_slow",
        )

        fast_updated = False
        if tau_fast is not None and tau_fast > 0:
            cv_fast = tau_fast_cv if tau_fast_cv is not None else ua_c_cv
            fast_updated = self._apply_greybox_update(
                field_name="tau_fast",
                new_value=tau_fast,
                cv=cv_fast,
                # Don't override the step_response primary; allow over seed
                # and prior greybox only.
                overridable={"seed", "greybox"},
                label="τ_fast",
            )

        if not (slow_updated or fast_updated):
            return None

        gains = self.compute_gains()
        _LOGGER.info(
            "Grey-box plant update applied (slow=%s, fast=%s) → Kp=%.3f Ki=%.4f",
            slow_updated, fast_updated, gains.kp, gains.ki,
        )
        return gains

    def _apply_greybox_update(
        self,
        field_name: str,
        new_value: float,
        cv: float,
        overridable: set[str],
        label: str,
    ) -> bool:
        """Helper: apply a grey-box τ update to one PlantEstimate field.

        Returns True if the field was updated, False otherwise.  Gates:
        confidence ≥ 0.3, current source in ``overridable``, and ratio
        between 0.5× and 2× of current value.
        """
        confidence = max(0.0, 1.0 - 2.0 * cv)
        if confidence < 0.3:
            _LOGGER.debug(
                "Grey-box %s=%.1f rejected: low confidence (CV=%.2f → conf=%.2f)",
                label, new_value, cv, confidence,
            )
            return False

        current: ParameterEstimate = getattr(self._plant, field_name)
        if current.source not in overridable:
            _LOGGER.debug(
                "Grey-box %s=%.1f: not overriding %s estimate (current=%.1f)",
                label, new_value, current.source, current.value,
            )
            return False

        if current.value > 0:
            ratio = new_value / current.value
            if not (0.5 <= ratio <= 2.0):
                _LOGGER.info(
                    "Grey-box %s=%.1f rejected: ratio=%.2f from current=%.1f too large",
                    label, new_value, ratio, current.value,
                )
                return False

        est = ParameterEstimate(
            value=new_value,
            confidence=confidence * 0.8,
            source="greybox",
            observations=1,
        )
        self._plant = dataclasses.replace(self._plant, **{field_name: est})
        _LOGGER.info(
            "Grey-box %s=%.1f (CV=%.2f, conf=%.2f) accepted",
            label, new_value, cv, est.confidence,
        )
        return True

    # ── Plant test (Layer 3: active identification) ──────────────────

    def start_plant_test(
        self,
        baseline_setpoint_c: float,
        amplitude_c: float,
        current_c: float,
        comfort_min_c: float,
        comfort_max_c: float,
        n_cycles: int = 4,
    ) -> None:
        """Start an active plant identification test."""
        if not self._enabled:
            return
        # Cancel any passive observations
        self.cancel_observation()

        self._plant_test = PlantTestProvider()
        self._plant_test.start(
            baseline_setpoint_c=baseline_setpoint_c,
            amplitude_c=amplitude_c,
            current_c=current_c,
            comfort_min_c=comfort_min_c,
            comfort_max_c=comfort_max_c,
            n_cycles=n_cycles,
            response_lag=self._response_lag,
        )

    def tick_plant_test(
        self, now_mono: float, current_c: float
    ) -> PlantTestCommand:
        """Advance the plant test by one tick.

        Returns PlantTestCommand with the setpoint to send.
        When the test transitions to step_hold, starts the passive
        providers to capture τ_fast and τ_slow from the hold phase.
        """
        if self._plant_test is None:
            return PlantTestCommand(setpoint_c=0, phase="aborted")

        cmd = self._plant_test.tick(now_mono, current_c)

        # When step-hold starts, feed the step to passive providers
        if cmd.phase == "step_hold" and self._plant_test._step_hold_ctx is not None:
            ctx = self._plant_test._step_hold_ctx
            if not self._step_provider.active:  # pragma: no branch — step_provider.active state — depends on prior plant test phase
                self._step_provider.start_observation(ctx)
            if not self._area_provider.active:  # pragma: no branch — area_provider.active state — depends on prior plant test phase
                self._area_provider.start_observation(ctx)
            self._plant_test._step_hold_ctx = None  # Only start once

        # During step-hold, accumulate data in passive providers
        if cmd.phase == "step_hold":
            tau_fast_est = self._step_provider.check_observation(now_mono, current_c)
            if tau_fast_est is not None:
                self._plant = dataclasses.replace(self._plant, tau_fast=tau_fast_est)
            tau_slow_est = self._area_provider.accumulate(
                now_mono, current_c, tau_fast=self._plant.tau_fast.value,
            )
            if tau_slow_est is not None:
                self._plant = dataclasses.replace(self._plant, tau_slow=tau_slow_est)

        # On completion, extract relay results
        if cmd.phase in ("complete", "aborted"):
            if cmd.phase == "complete":  # pragma: no branch — plant_test phase ∈ {complete, aborted}
                results = self._plant_test.get_results()
                if results:  # pragma: no branch — plant_test results dict empty — defensive
                    _LOGGER.info(
                        "Plant test complete: K_u=%.2f, T_u=%.1f min, a=%.3f°C",
                        results["k_u"].value,
                        results["period"].value,
                        results["amplitude"].value,
                    )
            self._plant_test = None

        return cmd

    def abort_plant_test(self) -> None:
        """Abort any active plant test."""
        if self._plant_test is not None:
            self._plant_test.abort()
            self._plant_test = None
        # Also cancel any passive observations started during step-hold
        self.cancel_observation()

    @property
    def plant_test_active(self) -> bool:
        """Whether an active plant test is running."""
        return self._plant_test is not None and self._plant_test.active

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

        Maturity gate (Skogestad SIMC + cautious adaptation): until a
        plant-ID parameter has accumulated MIN_TAU_OBSERVATIONS_FOR_GATE
        observations from a non-seed source, the conservative seed is
        used in the IMC formula instead of the live estimate.  This
        prevents single-observation outliers from driving large gain
        swings before evidence is strong enough to act on.
        """
        tau_fast = self._gated_tau(self._plant.tau_fast, self._tau_fast_seed)
        tau_slow = self._gated_tau(self._plant.tau_slow, self._tau_slow_seed)
        lag = self._response_lag
        lam = self._imc_lambda_config if self._imc_lambda_config > 0 else max(lag / 3.0, 1.0)
        k_eff = max(self._plant.k.value, 0.1)

        # Kp uses tau_slow (dominant dynamics — wall/mass time constant).
        kp = tau_slow / (k_eff * (lam + lag))
        ti = tau_slow / 3.0
        ki = kp / ti

        return GainUpdate(
            kp=kp, ki=ki, tau_fast=tau_fast, tau_slow=tau_slow, lag=lag,
            imc_lambda=lam,
        )

    def _gated_tau(self, estimate: ParameterEstimate, seed: float) -> float:
        """Return the seed unless the estimate has graduated past the maturity gate."""
        mature = (
            estimate.source != "seed"
            and estimate.observations >= self.MIN_TAU_OBSERVATIONS_FOR_GATE
        )
        return max(estimate.value if mature else seed, 1.0)

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

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic data for the debug bundle."""
        diag: dict[str, Any] = {
            "plant_estimate": self._plant.as_dict(),
            "providers": {
                "step_response": {"active": self._step_provider.active},
                "area_method": {"active": self._area_provider.active},
                "closed_loop": {"active": self._closed_loop_provider.active},
            },
        }

        if self._last_cross_check is not None:
            cl_fast, cl_slow = self._last_cross_check
            diag["cross_check"] = {
                "tau_fast": cl_fast.as_dict(),
                "tau_slow": cl_slow.as_dict(),
            }

        if self._plant_test is not None:
            test_diag: dict[str, Any] = {
                "active": self._plant_test.active,
                "phase": self._plant_test.phase,
                "cycle_count": self._plant_test.cycle_count,
            }
            if not self._plant_test.active:
                results = self._plant_test.get_results()
                if results:  # pragma: no branch — plant_test diagnostics with empty results — defensive
                    test_diag["results"] = {
                        k: v.as_dict() for k, v in results.items()
                    }
            diag["plant_test"] = test_diag

        return diag

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
