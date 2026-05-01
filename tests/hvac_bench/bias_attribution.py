"""Closed-loop bias attribution (Phase 1c).

Runs the same thermal physics under two controllers and compares the
recovered β coefficients:

- **Closed-loop arm.** The production PI controller drives the bench;
  batch WLS runs on the resulting observation buffer; final β is the
  estimator's converged value.
- **Open-loop arm.** A scripted setpoint trajectory drives the bench
  with no feedback; the same WLS path runs on the resulting observation
  list.

The gap (closed-loop β) − (open-loop β) is the closed-loop bias. Per
Forssell & Ljung (Automatica 1999) this gap quantifies how much the
feedback path biases identification on a given system; non-zero values
are the canonical fingerprint of closed-loop ID.

Both arms share an identical outdoor schedule / solar schedule / model
inputs / sensor noise seed so the only difference between them is
controller behaviour. Differences in HP runtime, room trajectories,
and observation distributions are themselves *the* signal — that's how
closed-loop bias enters.

Phase 1c output is a per-scenario credibility annotation that gets
attached to bench scenarios (Phase 2 reference-controller suite consumes
this) and an explicit comparison test that surfaces when the bias
becomes load-bearing on a learning-algorithm decision (e.g. #34, #41,
#47-followup).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from custom_components.tasmota_irhvac.pi.batch_learning import (
    weighted_least_squares,
)
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    FullStackResult,
    ModelInputSpec,
    diurnal_outdoor,
    diurnal_solar,
    run_full_stack,
)
from tests.hvac_bench.identifiability import (
    IdentifiabilityReport,
    identifiability_report,
)
from tests.hvac_bench.open_loop_runner import (
    Excitation,
    OpenLoopConfig,
    OpenLoopResult,
    make_step_excitation,
    run_open_loop_probe,
)


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class BiasAttributionConfig:
    """Single source of truth driving both arms.

    The shared fields (profile, weather, model inputs, sensor) are
    threaded into both ``FullStackConfig`` and ``OpenLoopConfig`` so the
    two arms exercise identical thermal physics. Only the controller
    differs.
    """

    n_days: int = 30
    profile_name: str = "living_room"

    outdoor_base_c: float = -2.0
    outdoor_diurnal_c: float = 8.0
    desired_c: float = 20.0
    mode: str = "heat"

    noise_sigma: float = 0.05
    noise_seed: int = 42

    tick_minutes: float = 15.0

    model_inputs: list[ModelInputSpec] = field(default_factory=list)

    # Open-loop arm uses this excitation (default: 6h step ±1.5°C above
    # closed-loop equilibrium so HP is always on, matching Phase 1a's
    # well-posed test scenario).
    probe_excitation_factory: Callable[[float, float], Excitation] = (
        lambda desired_c, tick_minutes: make_step_excitation(
            center_c=desired_c + 4.0,
            amplitude_c=1.5,
            hold_minutes=360.0,
            tick_minutes=tick_minutes,
        )
    )

    # Optional explicit schedules — when None, both arms get the same
    # canonical diurnal schedule keyed off the shared outdoor_base_c /
    # outdoor_diurnal_c.
    outdoor_schedule: Callable[[int], float] | None = None
    solar_schedule: Callable[[int], float] | None = None


# ── Result types ───────────────────────────────────────────────────────


@dataclass
class ArmResult:
    """One arm of the bias-attribution comparison.

    ``arm`` is ``"closed_loop"`` or ``"open_loop"``. ``beta`` is keyed
    by feature name; ``identifiability`` is the corresponding Phase 1b
    report. ``n_eligible_obs`` is how many observations passed the WLS
    filter (room_rate, clamped, etc.) and entered the regression.
    """

    arm: str
    beta: dict[str, float]
    identifiability: IdentifiabilityReport
    n_eligible_obs: int
    raw_full_stack: FullStackResult | None = None
    raw_open_loop: OpenLoopResult | None = None


@dataclass
class BiasAttributionResult:
    """Two-arm comparison output.

    ``closed_loop_bias[name] = closed_loop.beta[name] - open_loop.beta[name]``.
    Non-zero entries are the closed-loop bias contribution for each
    parameter, in the parameter's natural units. ``shared_features`` is
    the intersection of the two arms' feature sets (only those features
    have a meaningful gap).
    """

    closed_loop: ArmResult
    open_loop: ArmResult
    shared_features: list[str]
    closed_loop_bias: dict[str, float]


# ── Runner ─────────────────────────────────────────────────────────────


def run_bias_attribution(config: BiasAttributionConfig) -> BiasAttributionResult:
    """Run both arms and return the side-by-side comparison.

    Both arms get identical environmental schedules so the gap between
    them is attributable to controller behaviour, not weather.
    """
    tick_min = config.tick_minutes

    outdoor_fn = config.outdoor_schedule or (
        lambda t: diurnal_outdoor(
            t,
            config.outdoor_base_c,
            config.outdoor_diurnal_c,
            tick_min,
        )
    )
    solar_fn = config.solar_schedule or (
        lambda t: diurnal_solar(t, tick_minutes=tick_min)
    )

    # ── Closed-loop arm: full PI controller ─────────────────────────
    closed_loop_config = FullStackConfig(
        n_days=config.n_days,
        profile_name=config.profile_name,
        outdoor_base_c=config.outdoor_base_c,
        outdoor_diurnal_c=config.outdoor_diurnal_c,
        desired_c=config.desired_c,
        mode=config.mode,
        noise_sigma=config.noise_sigma,
        noise_seed=config.noise_seed,
        tick_minutes=tick_min,
        model_inputs=list(config.model_inputs),
        outdoor_schedule=outdoor_fn,
        solar_schedule=solar_fn,
    )
    closed_loop_full = run_full_stack(closed_loop_config)

    # The PI runner doesn't return the WLS observation buffer directly,
    # so the closed-loop β comes from final_coefs (production WLS at end
    # of run). For the identifiability report on the closed-loop arm,
    # we'd need the buffer state — which is internal to PI. Phase 1c
    # ships with closed-loop identifiability as None for now; Phase 1e
    # adds it via a checkpoint that snapshots the buffer.
    cl_beta_named = dict(closed_loop_full.final_coefs)
    closed_loop_arm = ArmResult(
        arm="closed_loop",
        beta=cl_beta_named,
        identifiability=identifiability_report(
            observations=[],
            feature_order=list(cl_beta_named.keys()),
        ),
        n_eligible_obs=closed_loop_full.ticks_hp_on,
        raw_full_stack=closed_loop_full,
    )

    # ── Open-loop arm: scripted setpoint ─────────────────────────────
    excitation = config.probe_excitation_factory(
        config.desired_c, tick_min
    )
    open_loop_config = OpenLoopConfig(
        excitation=excitation,
        n_days=config.n_days,
        profile_name=config.profile_name,
        outdoor_base_c=config.outdoor_base_c,
        outdoor_diurnal_c=config.outdoor_diurnal_c,
        desired_c=config.desired_c,
        mode=config.mode,
        noise_sigma=config.noise_sigma,
        noise_seed=config.noise_seed,
        tick_minutes=tick_min,
        model_inputs=list(config.model_inputs),
        outdoor_schedule=outdoor_fn,
        solar_schedule=solar_fn,
    )
    open_loop_run = run_open_loop_probe(open_loop_config)

    # Build feature_order matching what closed-loop reports (intercept,
    # outdoor_delta, then any model inputs).
    feature_order = ["intercept", "outdoor_delta"]
    feature_order += [mi.name for mi in config.model_inputs]
    pi_model_inputs = [
        {
            "entity_id": mi.entity_id,
            "name": mi.name,
            "delta_from_room": mi.delta_from_room,
        }
        for mi in config.model_inputs
    ]

    ol_wls = weighted_least_squares(
        observations=open_loop_run.observations,
        n_features=len(feature_order),
        feature_order=feature_order,
        model_inputs=pi_model_inputs or None,
        min_observations=20,
        detect_lag=False,
    )
    if ol_wls is None:
        raise RuntimeError(
            "Open-loop WLS returned None — probe produced too few "
            "eligible observations for the configured feature set"
        )
    ol_beta_named = {
        feature_order[i]: ol_wls.beta_batch[i]
        for i in range(len(feature_order))
    }

    ol_id = identifiability_report(
        observations=open_loop_run.observations,
        feature_order=feature_order,
        model_inputs=pi_model_inputs or None,
        sigma2=None,
        beta=ol_wls.beta_batch,
    )
    open_loop_arm = ArmResult(
        arm="open_loop",
        beta=ol_beta_named,
        identifiability=ol_id,
        n_eligible_obs=ol_id.n_observations,
        raw_open_loop=open_loop_run,
    )

    # ── Compute bias ─────────────────────────────────────────────────
    shared = sorted(set(cl_beta_named) & set(ol_beta_named))
    bias = {
        name: cl_beta_named[name] - ol_beta_named[name]
        for name in shared
    }

    return BiasAttributionResult(
        closed_loop=closed_loop_arm,
        open_loop=open_loop_arm,
        shared_features=shared,
        closed_loop_bias=bias,
    )


# ── Reporting helpers ─────────────────────────────────────────────────


def format_bias_report(result: BiasAttributionResult) -> str:
    """Render a side-by-side text table for inclusion in test logs."""
    lines = [
        "=" * 72,
        "  Closed-loop bias attribution",
        "=" * 72,
        f"{'Feature':<24}{'Closed-loop β':>14}{'Open-loop β':>14}{'Bias':>14}",
        "-" * 72,
    ]
    for name in result.shared_features:
        cl = result.closed_loop.beta.get(name, float("nan"))
        ol = result.open_loop.beta.get(name, float("nan"))
        bias = result.closed_loop_bias.get(name, float("nan"))
        lines.append(
            f"{name:<24}{cl:>14.4f}{ol:>14.4f}{bias:>+14.4f}"
        )
    lines.append("-" * 72)
    ol_id = result.open_loop.identifiability
    lines.append(
        f"Open-loop n_obs={ol_id.n_observations}, κ={ol_id.condition_number:.2f}, "
        f"σ²={ol_id.sigma2:.4f}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)
