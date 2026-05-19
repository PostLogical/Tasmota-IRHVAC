"""Open-loop probe runner for closed-loop bias attribution (Phase 1a).

Replaces the PI controller with a scripted setpoint trajectory so the
identification path receives data with **no closed-loop coupling**. The
same thermal model, weather, and model-input plumbing as
``full_stack_runner`` is reused; only the controller is bypassed.

Per Forssell & Ljung (Automatica 1999, "Closed-loop identification
revisited"), open-loop data is the upper bound on what an experiment
design can identify. The gap between an estimator's closed-loop β and
its open-loop β on identical thermal physics is the **closed-loop bias**
contribution. This module produces the open-loop side of that
comparison; ``full_stack_runner`` produces the closed-loop side.

Excitation choices follow Ljung *System Identification* (1999) Ch. 13.4:
step (PE order 2, DC-gain identification), PRBS (broadband, persistently
exciting of any order ≤ register length), and Schroeder-phased multi-sine
(frequency-targeted, low crest factor — Pintelon & Schoukens 2012).

The output is a list of ``Observation`` objects ready for
``weighted_least_squares``. The probe has no buffer eviction, no
``hp_contribution_uncertain`` masking, no clamping — every tick produces
one observation. Filtering is the caller's responsibility (typically the
``room_rate < threshold`` filter inside WLS).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable

from custom_components.tasmota_irhvac.pi.batch_learning import Observation
from tests.hvac_bench.full_stack_runner import (
    TICK_MINUTES_DEFAULT,
    _SIM_EPOCH,
    ModelInputSpec,
    diurnal_outdoor,
    diurnal_solar,
)
from tests.hvac_bench.house_profiles import PROFILES, PROFILES_2R2C
from tests.hvac_bench.thermal_model import ThermalModel2R2C


Excitation = Callable[[int], float]


# ── Excitation generators ───────────────────────────────────────────────


def make_step_excitation(
    *,
    center_c: float,
    amplitude_c: float,
    hold_minutes: float,
    tick_minutes: float = TICK_MINUTES_DEFAULT,
) -> Excitation:
    """Square-wave setpoint between ``center_c ± amplitude_c``.

    Holds at each level for ``hold_minutes``. PE order 2 (Ljung §13.4).
    DC-gain estimator: with ``hold_minutes ≥ 4·τ_eff`` each hold reaches
    quasi-steady state and the regression sees clean (sp, room) pairs.

    Returns a callable ``tick → hp_setpoint``.
    """
    if hold_minutes <= 0:
        raise ValueError("hold_minutes must be > 0")
    if amplitude_c <= 0:
        raise ValueError("amplitude_c must be > 0")
    hold_ticks = max(1, int(round(hold_minutes / tick_minutes)))

    def excite(tick: int) -> float:
        phase = (tick // hold_ticks) % 2
        return center_c + amplitude_c if phase == 0 else center_c - amplitude_c

    return excite


def make_prbs_excitation(
    *,
    center_c: float,
    amplitude_c: float,
    min_hold_minutes: float,
    seed: int = 0,
    tick_minutes: float = TICK_MINUTES_DEFAULT,
) -> Excitation:
    """Pseudo-random binary sequence with minimum hold time.

    Maximum-Length-Sequence-style binary signal flipping between
    ``center_c ± amplitude_c``. Stays at each level for at least
    ``min_hold_minutes``, then flips with probability 0.5 per tick.
    PE order is unbounded asymptotically (Ljung §13.4); for finite-length
    runs the realised PE order is bounded by the number of switches.

    Per Söderström & Stoica (1989) §5.6, ``min_hold_minutes ≈ τ/4`` is the
    standard rule of thumb so the system has time to start responding
    before the next switch but the input remains broadband relative to
    system bandwidth.

    Returns a callable ``tick → hp_setpoint``. The function caches its
    state per call — calling out of order produces wrong results; call
    with monotonically increasing ticks.
    """
    if min_hold_minutes <= 0:
        raise ValueError("min_hold_minutes must be > 0")
    if amplitude_c <= 0:
        raise ValueError("amplitude_c must be > 0")
    min_hold_ticks = max(1, int(round(min_hold_minutes / tick_minutes)))

    rng = random.Random(seed)
    state = {"sign": 1, "next_eligible_tick": 0, "last_tick": -1}

    def excite(tick: int) -> float:
        if tick != state["last_tick"] + 1 and state["last_tick"] != -1:
            raise ValueError(
                f"PRBS excitation expects monotonic ticks; got {tick} "
                f"after {state['last_tick']}"
            )
        state["last_tick"] = tick
        if tick >= state["next_eligible_tick"] and rng.random() < 0.5:
            state["sign"] *= -1
            state["next_eligible_tick"] = tick + min_hold_ticks
        return center_c + state["sign"] * amplitude_c

    return excite


def make_multisine_excitation(
    *,
    center_c: float,
    amplitude_c: float,
    n_components: int,
    period_min_minutes: float,
    period_max_minutes: float,
    tick_minutes: float = TICK_MINUTES_DEFAULT,
) -> Excitation:
    """Schroeder-phased multi-sine spanning ``[period_min, period_max]``.

    Sum of ``n_components`` sinusoids at logarithmically-spaced periods,
    energy-normalized, Schroeder-phased to minimize crest factor:

        s(t) = amplitude × Σ_k sin(2π t/τ_k + φ_k) / sqrt(M)

    where ``φ_k = -π k(k-1) / M`` (Schroeder 1970, *IEEE Trans. Inf.
    Theory* 16:85-89). Frequencies span the bandwidth where thermal
    dynamics live; ``period_min ≈ τ_fast/2`` and ``period_max ≈ 4·τ_slow``
    is the standard rule of thumb.

    Lower crest factor than a random-phase multisine so the actuator
    isn't pushed to extreme amplitudes for short bursts (Pintelon &
    Schoukens 2012, *System Identification: A Frequency Domain
    Approach*, 2nd ed., Ch. 4).

    Returns a callable ``tick → hp_setpoint``.
    """
    if n_components < 1:
        raise ValueError("n_components must be ≥ 1")
    if period_min_minutes <= 0 or period_max_minutes <= period_min_minutes:
        raise ValueError("require 0 < period_min_minutes < period_max_minutes")
    if amplitude_c <= 0:
        raise ValueError("amplitude_c must be > 0")

    log_min = math.log(period_min_minutes)
    log_max = math.log(period_max_minutes)
    if n_components == 1:
        periods = [math.exp((log_min + log_max) / 2.0)]
    else:
        step = (log_max - log_min) / (n_components - 1)
        periods = [math.exp(log_min + k * step) for k in range(n_components)]
    schroeder_phases = [-math.pi * k * (k - 1) / n_components for k in range(n_components)]
    norm = math.sqrt(n_components)

    def excite(tick: int) -> float:
        t_minutes = tick * tick_minutes
        s = 0.0
        for k in range(n_components):
            s += math.sin(2 * math.pi * t_minutes / periods[k] + schroeder_phases[k])
        return center_c + amplitude_c * s / norm

    return excite


def clip_to_bounds(
    excitation: Excitation, *, min_c: float, max_c: float
) -> Excitation:
    """Wrap an excitation so its output is clipped to ``[min_c, max_c]``.

    HP setpoint must lie within the controller's configured range
    (typically 16–30°C). Clipping is the caller's choice — for some
    fidelity comparisons an unclipped probe is desired so saturation
    effects are exposed rather than hidden.
    """
    def clipped(tick: int) -> float:
        return max(min_c, min(max_c, excitation(tick)))

    return clipped


# ── Configuration ──────────────────────────────────────────────────────


@dataclass
class OpenLoopConfig:
    """Configuration for an open-loop probe run.

    Mirrors ``FullStackConfig`` for the environmental knobs (profile,
    weather, model inputs, sensor noise) but replaces the PI controller
    with a scripted ``excitation`` callable.
    """

    excitation: Excitation
    n_days: int = 30
    profile_name: str = "living_room"

    # Environmental conditions
    outdoor_base_c: float = -2.0
    outdoor_diurnal_c: float = 8.0
    desired_c: float = 20.5
    mode: str = "heat"

    # Sensor
    noise_sigma: float = 0.1
    noise_seed: int = 42

    # Tick interval (minutes). Fixed; no CSV path here yet.
    tick_minutes: float = TICK_MINUTES_DEFAULT

    # Model inputs (solar, stove, sunroom, etc.)
    model_inputs: list[ModelInputSpec] = field(default_factory=list)

    # Outdoor/solar schedule overrides
    outdoor_schedule: Callable[[float], float] | None = None
    solar_schedule: Callable[[float], float] | None = None


@dataclass
class OpenLoopResult:
    """Output of an open-loop probe run."""

    observations: list[Observation]
    history: list[dict]
    n_ticks: int
    setpoint_trajectory: list[float]
    profile_name: str
    tick_minutes: float
    desired_c: float
    true_coefs: dict[str, float]


# ── Runner ─────────────────────────────────────────────────────────────


def run_open_loop_probe(config: OpenLoopConfig) -> OpenLoopResult:
    """Run a scripted-setpoint probe and return WLS-ready observations.

    No PI controller. No batch triggering. No buffer. The probe drives
    the thermal model with ``config.excitation`` and records one
    ``Observation`` per tick. The caller passes the observation list to
    ``weighted_least_squares`` (or any other estimator) to recover β
    on data with zero closed-loop coupling.
    """
    if config.profile_name in PROFILES_2R2C:
        profile = PROFILES_2R2C[config.profile_name]
    elif config.profile_name in PROFILES:
        profile = PROFILES[config.profile_name]
    else:
        raise ValueError(f"Unknown profile: {config.profile_name}")

    tick_min = config.tick_minutes
    n_ticks = int(config.n_days * 24 * 60 / tick_min)

    # Per-run copies so the runner's .resolve() doesn't mutate the
    # caller's specs (matches full_stack_runner / reference_scenarios).
    model_inputs = [copy.copy(mi) for mi in config.model_inputs]
    for mi in model_inputs:
        mi.resolve(profile.hp_gain)

    solar_thermal_gain = sum(
        mi.true_thermal_effect
        for mi in model_inputs
        if mi.input_role == "solar"
    )

    initial_outdoor = config.outdoor_base_c
    if config.outdoor_schedule is not None:
        initial_outdoor = config.outdoor_schedule(0)
    model = ThermalModel2R2C(
        profile=profile,
        initial_temp=config.desired_c,
        outdoor_temp=initial_outdoor,
        sensor_noise_sigma=config.noise_sigma,
        noise_seed=config.noise_seed,
        solar_gain=solar_thermal_gain,
        stove_gain=0.0,
    )

    outdoor_fn = config.outdoor_schedule or (
        lambda m: diurnal_outdoor(m, config.outdoor_base_c,
                                  config.outdoor_diurnal_c)
    )
    solar_fn = config.solar_schedule or (
        lambda m: diurnal_solar(m)
    )

    # Truth coefficients for downstream comparison.
    #
    # Physics-space (signed β), matching full_stack_runner's convention so
    # callers can consume either runner's true_coefs interchangeably.
    #
    # NOTE: open-loop and closed-loop have DIFFERENT outdoor_delta truths.
    # Closed loop:  -1/(g·τ_env)        = -profile.true_seed
    # Open loop:    -1/(g·τ_env + 1)    (asymptote of fixed-setpoint regression)
    # See test_open_loop_runner.test_beta_outdoor_differs_from_closed_loop_truth
    # for the derivation.
    open_loop_outdoor_truth = -1.0 / (profile.hp_gain * profile.tau_env + 1.0)
    true_coefs: dict[str, float] = {
        "intercept": 0.0,
        "outdoor_delta": open_loop_outdoor_truth,
    }
    for mi in model_inputs:
        true_coefs[mi.name] = mi.true_ff_coef(profile.hp_gain)

    observations: list[Observation] = []
    history: list[dict] = []
    setpoint_trajectory: list[float] = []
    sim_clock = 0.0
    sim_epoch_ts = _SIM_EPOCH.timestamp()
    prev_sensor: float | None = None

    for tick in range(n_ticks):
        dt_seconds = tick_min * 60.0
        minute = tick * tick_min
        sim_clock += dt_seconds

        # Update outdoor.
        model.outdoor_temp = outdoor_fn(minute)

        # Compute model input values + per-node heat injection (mirrors
        # full_stack_runner's ASHRAE/Madsen routing).
        input_values: dict[str, float] = {}
        solar_proxy_value = 0.0
        q_air_extra = 0.0
        q_wall_extra = 0.0
        raw_readings: dict[str, float] = {}
        for mi in model_inputs:
            val = mi.schedule(minute) if mi.schedule is not None else 0.0
            input_values[mi.name] = val
            raw_readings[mi.entity_id] = val
            if mi.input_role == "solar":
                solar_proxy_value += val
                continue
            feature_val = val - model.room_temp if mi.delta_from_room else val
            q_total = mi.true_thermal_effect * feature_val
            if mi.input_role == "adjacent_zone":
                q_wall_extra += q_total
            else:
                q_air_extra += q_total * 0.3
                q_wall_extra += q_total * 0.7

        # Default solar contribution from the diurnal generator (only
        # active when no model input owns the solar role).
        if not any(mi.input_role == "solar" for mi in model_inputs):
            solar_proxy_value = solar_fn(minute)

        # Read sensor at the START of the tick (room_rate is computed
        # from consecutive readings).
        sensor_reading = model.read_sensor()
        if prev_sensor is None:
            room_rate = 0.0
        else:
            room_rate = (sensor_reading - prev_sensor) / tick_min
        prev_sensor = sensor_reading

        # Drive HP from the scripted excitation (no feedback).
        hp_setpoint = float(config.excitation(tick))

        # Advance thermal model.
        model.step(
            hp_setpoint=hp_setpoint,
            dt_minutes=tick_min,
            solar_proxy=solar_proxy_value,
            q_air_extra=q_air_extra,
            q_wall_extra=q_wall_extra,
            tick=tick,
            mode=config.mode,
        )

        observations.append(
            Observation(
                timestamp=sim_clock,
                wall_time=sim_epoch_ts + sim_clock,
                hp_setpoint=hp_setpoint,
                current_c=sensor_reading,
                desired_c=config.desired_c,
                # Open-loop runner has no supervisor in the loop, so the
                # effective reference equals the user-stated value.
                effective_desired_c=config.desired_c,
                outdoor_temp_c=model.outdoor_temp,
                room_rate=room_rate,
                raw_readings=raw_readings,
                clamped=False,
                clamped_reason="",
                supplemental_active=False,
                hp_contribution_uncertain=False,
            )
        )
        setpoint_trajectory.append(hp_setpoint)

        history.append({
            "tick": tick,
            "room_temp": model.room_temp,
            "sensor_reading": sensor_reading,
            "hp_setpoint": hp_setpoint,
            "outdoor": model.outdoor_temp,
            "room_rate": room_rate,
            "solar_proxy": solar_proxy_value,
            **{f"input_{mi.name}": input_values[mi.name]
               for mi in model_inputs},
        })

    return OpenLoopResult(
        observations=observations,
        history=history,
        n_ticks=n_ticks,
        setpoint_trajectory=setpoint_trajectory,
        profile_name=config.profile_name,
        tick_minutes=tick_min,
        desired_c=config.desired_c,
        true_coefs=true_coefs,
    )
