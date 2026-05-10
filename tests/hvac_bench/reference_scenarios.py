"""Reference scenarios + locked-score harness (Phase 2b).

Defines the canonical (controller × scenario) grid the bench locks numbers
against. Pattern follows BOPTEST: a fixed list of well-defined scenarios,
each with locked KPI scores and tolerances. CI fails if any score moves
outside tolerance — making bench changes that silently break
discriminative power immediately visible.

Scope of this initial cut:

* **Three scenarios** — heating step, cooling step, heating with a solar
  input. Each is short (3 sim-days at 15-min ticks) so the regression
  suite runs in tens of seconds, not minutes.
* **Three controllers** — :class:`NaiveBangBangController`,
  :func:`make_well_tuned_pi`, :func:`make_production_pi`.
* **Control-side KPIs only** — ``tdis_tot``, ``ener_tot``, ``peak_kw``,
  ``cold_time_h``, ``warm_time_h``, ``setpoint_changes``. Learning KPIs
  (β bias, CRLB coverage, whiteness) need Monte-Carlo realisations to
  be meaningful and are deferred to a follow-up.
* **Single MC seed per (controller × scenario)** — the locked numbers
  are point estimates with tolerances calibrated from a small variance
  audit. MC bands are a follow-up.

The locked numbers are produced by ``--update-reference-scores`` mode (a
flag the regression test exposes); changing a number must be a deliberate
commit, with a comment explaining what changed about the bench.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable

from tests.hvac_bench.controller_protocol import HVACController
from tests.hvac_bench.full_stack_runner import (
    ModelInputSpec,
    diurnal_outdoor,
    diurnal_solar,
)
from tests.hvac_bench.house_profiles import PROFILES, PROFILES_2R2C
from tests.hvac_bench.kpis import KpiBundle, compute_control_kpis
from tests.hvac_bench.thermal_model import COPModel, ThermalModel2R2C


# ── Scenario definition ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ReferenceScenario:
    """A single locked scenario.

    Attributes are intentionally minimal — anything missing here is a
    runner default. Adding a knob to a scenario is itself a bench change
    and should require updating locked scores.
    """

    name: str
    profile_name: str
    mode: str = "heat"
    outdoor_base_c: float = -2.0
    outdoor_diurnal_c: float = 8.0
    desired_c: float = 20.5
    n_days: float = 3.0
    tick_minutes: float = 15.0
    noise_sigma: float = 0.1
    noise_seed: int = 42
    initial_temp_c: float | None = None  # None = desired_c
    initial_wall_temp_c: float | None = None  # None = initial_temp_c
    model_inputs: tuple[ModelInputSpec, ...] = field(default_factory=tuple)
    batch_interval_hours: float = 12.0
    head_calibration_bounds: tuple[float, float] = (0.0, 0.0)


# ── Reference runner ──────────────────────────────────────────────────────


def _resolve_profile(profile_name: str):
    if profile_name in PROFILES_2R2C:
        return PROFILES_2R2C[profile_name]
    if profile_name in PROFILES:
        return PROFILES[profile_name]
    raise ValueError(f"Unknown profile: {profile_name}")


def run_reference_scenario(
    controller: HVACController,
    scenario: ReferenceScenario,
) -> tuple[list[dict], KpiBundle]:
    """Run a controller against a reference scenario, returning history + KPIs.

    The runner is intentionally lighter than ``full_stack_runner.run_full_stack``
    — it doesn't track per-day rollups, learning trajectories, comfort
    sub-categorisation, or zone-model classification. It produces the
    minimum needed for ``compute_control_kpis``. Heavier diagnostics
    (residual whiteness on the buffer, β recovery) are computed by callers
    on the returned ``history`` if needed.
    """
    profile = _resolve_profile(scenario.profile_name)
    initial_temp = scenario.initial_temp_c if scenario.initial_temp_c is not None else scenario.desired_c

    # Resolve model-input physics on a per-run copy so the module-level
    # CANONICAL_SCENARIOS instances stay pristine across runs (mutating
    # them silently corrupts any future run that uses a different hp_gain).
    model_inputs = tuple(copy.copy(mi) for mi in scenario.model_inputs)
    for mi in model_inputs:
        mi.resolve(profile.hp_gain)

    # Solar inputs feed the model's built-in 2R2C solar pathway
    # (matches full_stack_runner.run_full_stack semantics).
    solar_thermal_gain = sum(
        mi.true_thermal_effect for mi in model_inputs
        if mi.input_role == "solar"
    )

    model = ThermalModel2R2C(
        profile=profile,
        initial_temp=initial_temp,
        outdoor_temp=scenario.outdoor_base_c,
        cop_model=COPModel(),
        sensor_noise_sigma=scenario.noise_sigma,
        noise_seed=scenario.noise_seed,
        initial_wall_temp=scenario.initial_wall_temp_c,
        solar_gain=solar_thermal_gain,
    )

    tick_min = scenario.tick_minutes
    n_ticks = int(scenario.n_days * 24 * 60 / tick_min)
    batch_interval_ticks = max(1, int(scenario.batch_interval_hours * 60 / tick_min))

    controller.set_mode(scenario.mode)
    controller.set_desired_temp(scenario.desired_c)

    history: list[dict] = []

    for tick in range(n_ticks):
        dt_seconds = tick_min * 60.0

        # Drive outdoor schedule (synthetic diurnal, no weather state coupling
        # — that's a Phase 4 / multi-realisation concern).
        model.outdoor_temp = diurnal_outdoor(
            tick, scenario.outdoor_base_c, scenario.outdoor_diurnal_c, tick_min
        )

        # Compute model input values per role.
        input_values: dict[str, float] = {}
        solar_proxy_value = 0.0
        q_air_extra = 0.0
        q_wall_extra = 0.0
        for mi in model_inputs:
            val = mi.schedule(tick) if mi.schedule is not None else 0.0
            input_values[mi.name] = val
            if mi.input_role == "solar":
                solar_proxy_value += val
                continue
            feature_val = val - model.room_temp if mi.delta_from_room else val
            q_total = mi.true_thermal_effect * feature_val
            if mi.input_role == "adjacent_zone":
                q_wall_extra += q_total
            else:
                # Default ASHRAE 0.3 air / 0.7 wall split for radiant sources.
                q_air_extra += 0.3 * q_total
                q_wall_extra += 0.7 * q_total

        sensor_reading = model.read_sensor()

        hp_setpoint = controller.tick(
            room_temp_c=sensor_reading,
            outdoor_temp_c=model.outdoor_temp,
            dt_seconds=dt_seconds,
            model_inputs=input_values,
        )

        model.step(
            hp_setpoint=hp_setpoint,
            dt_minutes=tick_min,
            solar_proxy=solar_proxy_value,
            q_air_extra=q_air_extra,
            q_wall_extra=q_wall_extra,
            tick=tick,
            mode=scenario.mode,
        )

        state = controller.get_state()
        desired = state.get("desired_temp", scenario.desired_c)

        history.append({
            "tick": tick,
            "room_temp": model.room_temp,
            "sensor_reading": sensor_reading,
            "desired": desired,
            "hp_setpoint": hp_setpoint,
            "error": desired - model.room_temp,
            "outdoor": model.outdoor_temp,
            "cumulative_kwh": model.cumulative_kwh,
            "integral": state.get("integral", 0.0),
            "ff_offset": state.get("ff_offset", 0.0),
        })

        # Trigger learning batches at the configured interval. Controllers
        # without learning expose batch_update as a no-op.
        if hasattr(controller, "batch_update"):
            if tick > 0 and tick % batch_interval_ticks == 0:
                controller.batch_update(tick)

    bundle = compute_control_kpis(history, tick_minutes=tick_min)
    return history, bundle


# ── Canonical scenario set ────────────────────────────────────────────────


def _solar_schedule(tick: int) -> float:
    """Solar input schedule: matches the bench's canonical diurnal solar."""
    return diurnal_solar(tick, peak=0.8, tick_minutes=15.0)


CANONICAL_SCENARIOS: dict[str, ReferenceScenario] = {
    "lr_heat_step": ReferenceScenario(
        name="lr_heat_step",
        profile_name="living_room",
        mode="heat",
        outdoor_base_c=-5.0,
        outdoor_diurnal_c=6.0,
        desired_c=20.5,
        n_days=3.0,
    ),
    "lr_cool_step": ReferenceScenario(
        name="lr_cool_step",
        profile_name="living_room",
        mode="cool",
        outdoor_base_c=28.0,
        outdoor_diurnal_c=6.0,
        desired_c=23.0,
        n_days=3.0,
    ),
    "lr_heat_with_solar": ReferenceScenario(
        name="lr_heat_with_solar",
        profile_name="living_room",
        mode="heat",
        outdoor_base_c=-5.0,
        outdoor_diurnal_c=6.0,
        desired_c=20.5,
        n_days=3.0,
        model_inputs=(
            ModelInputSpec(
                name="solar",
                entity_id="sensor.solar",
                input_role="solar",
                _true_ff_coef=-2.0,
                seed_heat=0.0,  # unused for naive / overridden by truth-seeding
                schedule=_solar_schedule,
                clamp_min=0.0,
            ),
        ),
    ),
}


# ── Controller factories per (scenario, controller-id) ───────────────────


def make_naive_for_scenario(scenario: ReferenceScenario) -> HVACController:
    """Build NaiveBangBang configured for the scenario."""
    from tests.hvac_bench.reference_controllers import NaiveBangBangController
    profile = _resolve_profile(scenario.profile_name)
    # Use HVACController defaults; min/max aligned with conftest defaults.
    return NaiveBangBangController(min_temp=16.0, max_temp=30.0, hysteresis_c=0.5)


def make_well_tuned_for_scenario(scenario: ReferenceScenario) -> HVACController:
    """Build WellTunedPI configured for the scenario (FF seeded with truth)."""
    from tests.hvac_bench.reference_controllers import make_well_tuned_pi
    return make_well_tuned_pi(
        scenario.profile_name,
        mode=scenario.mode,
        model_inputs=list(scenario.model_inputs) or None,
        head_calibration_bounds=scenario.head_calibration_bounds,
    )


def make_production_for_scenario(scenario: ReferenceScenario) -> HVACController:
    """Build ProductionPI for the scenario (default seeds, learning ON).

    For the locked-score harness we keep the realistic fresh-deployment
    posture: no truth-seeding. This means the production controller may
    initially perform worse than well-tuned on short scenarios and recover
    over longer horizons — exactly the behavior Phase 2's discriminative
    power is meant to expose.
    """
    from tests.hvac_bench.reference_controllers import make_production_pi
    return make_production_pi(
        scenario.profile_name,
        mode=scenario.mode,
        model_inputs=list(scenario.model_inputs) or None,
        seed_with_truth=False,
        head_calibration_bounds=scenario.head_calibration_bounds,
    )


CONTROLLER_FACTORIES: dict[str, Callable[[ReferenceScenario], HVACController]] = {
    "naive_bang_bang": make_naive_for_scenario,
    "well_tuned_pi": make_well_tuned_for_scenario,
    "production_pi": make_production_for_scenario,
}
