"""Full-stack simulation engine for PI+RLS+WLS learning validation.

Runs the real PIController against a 2R2C thermal model with batch WLS
triggering, checkpoint callbacks, CSV weather input, and disturbance
injection.  Extracted from test_adaptive_step_cap.py::run_full_system()
and generalized for multi-timescale validation.

Usage:
    config = FullStackConfig(n_days=30, profile_name="living_room", ...)
    result = run_full_stack(config)
    # or with checkpoints:
    result = run_full_stack(config, checkpoints=[
        Checkpoint(interval_days=7, callback=assert_weekly_progress),
    ])
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Callable

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import PROFILES, PROFILES_2R2C
from tests.hvac_bench.thermal_model import ThermalModel2R2C


# ── Constants ────────────────────────────────────────────────────────────

TICK_MINUTES_DEFAULT = 15.0
TICKS_PER_HOUR = int(60 / TICK_MINUTES_DEFAULT)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR  # 96
BATCH_INTERVAL_HOURS_DEFAULT = 12
DEADBAND = 0.5


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class ModelInputSpec:
    """A model input with ground-truth thermal effect and FF coefficient."""

    name: str
    entity_id: str
    input_role: str  # "solar", "adjacent_zone", "heat_source", "other"
    true_thermal_effect: float  # °C room per unit input per minute
    true_ff_coef: float  # ground-truth WLS coefficient
    seed_heat: float = 0.0
    seed_cool: float = 0.0
    schedule: Callable[[int], float] | None = None  # tick -> value
    delta_from_room: bool = False
    lag_tau: int = 0
    clamp_min: float | None = None  # min in seed space (positive = warms room)
    clamp_max: float | None = None  # max in seed space


@dataclass
class Disturbance:
    """Inject an anomaly into a field at a specific tick range.

    The field can be 'room_temp_offset' (sensor grab), 'solar_override',
    or any model input name.
    """

    start_tick: int
    duration_ticks: int
    field: str  # 'room_temp_offset' or model input name
    value: float  # override/offset value


@dataclass
class FullStackConfig:
    """Configuration for a full-stack simulation."""

    # Duration
    n_days: int = 30

    # House profile (key in PROFILES_2R2C or PROFILES)
    profile_name: str = "living_room"

    # Environmental conditions
    outdoor_base_c: float = -2.0
    outdoor_diurnal_c: float = 8.0
    desired_c: float = 20.5
    mode: str = "heat"

    # Sensor
    noise_sigma: float = 0.1
    noise_seed: int = 42

    # Tick interval (minutes).  Fixed for synthetic; overridden by CSV.
    tick_minutes: float = TICK_MINUTES_DEFAULT

    # Batch WLS interval in hours
    batch_interval_hours: float = BATCH_INTERVAL_HOURS_DEFAULT

    # Model inputs
    model_inputs: list[ModelInputSpec] = field(default_factory=list)

    # PI config overrides (passed to TasmotaPIAdapter)
    pi_overrides: dict = field(default_factory=dict)

    # Outdoor/solar schedule overrides (callable: tick -> value)
    outdoor_schedule: Callable[[int], float] | None = None
    solar_schedule: Callable[[int], float] | None = None

    # Disturbances to inject
    disturbances: list[Disturbance] = field(default_factory=list)

    # If True, relax κ gate (for short sims with limited diversity)
    relax_kappa_gate: bool = False

    # True coefficients for convergence checking.
    # If None, computed from profile + model inputs.
    true_coefs: dict[str, float] | None = None


# ── Checkpoint system ────────────────────────────────────────────────────


@dataclass
class CheckpointState:
    """State snapshot passed to checkpoint callbacks."""

    tick: int
    day: int
    elapsed_days: float
    history: list[dict]
    coef_trajectory: list[dict]
    current_coefs: dict[str, float]
    true_coefs: dict[str, float]
    coef_errors: dict[str, float]
    integral_rms: float
    batch_count: int
    batches_to_converge: int | None

    # Per-period metrics (since last checkpoint or start)
    period_itae: float
    period_violations: int
    period_reversals: int
    period_mae: float

    # Cumulative metrics
    total_itae: float
    total_violations: int

    # Learning state
    ff_fraction: float  # current |FF| / (|FF| + |integral|)
    covariance_trace: float  # current tr(P)
    buffer_utilization: float  # current buffer len / max

    # Access to PI internals
    pi: object  # PIController instance


@dataclass
class Checkpoint:
    """Define a periodic checkpoint with a callback.

    The callback receives CheckpointState and can assert, log, or collect.
    If it raises, the simulation stops with the error.
    """

    interval_days: int  # fire every N days
    callback: Callable[[CheckpointState], None]


# ── Result ───────────────────────────────────────────────────────────────


@dataclass
class FullStackResult:
    """Results from a full-stack simulation."""

    history: list[dict]
    coef_trajectory: list[dict]  # per-batch coefficient snapshots
    final_coefs: dict[str, float]
    true_coefs: dict[str, float]
    coef_errors: dict[str, float]
    integral_rms: float
    total_itae: float
    total_violations: int
    total_reversals: int
    batches_to_converge: int | None
    n_batches: int
    n_ticks: int

    # Per-day rollups
    daily_itae: list[float]
    daily_violations: list[int]
    daily_reversals: list[int]
    daily_mae: list[float]
    daily_integral_rms: list[float]

    # Per-week rollups
    weekly_itae: list[float]
    weekly_violations: list[int]
    weekly_reversals: list[int]

    # Checkpoint results (collected by callbacks if they return values)
    checkpoint_data: list[dict]

    # ── Comfort metrics ──────────────────────────────────────────────
    comfort_hours_pct: float  # % of ticks within deadband
    cold_violations: int  # ticks where room < desired - deadband
    warm_violations: int  # ticks where room > desired + deadband
    worst_undershoot: float  # max (desired - room) when room < desired
    worst_overshoot: float  # max (room - desired) when room > desired
    longest_violation_streak: int  # max consecutive ticks outside deadband

    # Per-day comfort rollups
    daily_comfort_pct: list[float]
    daily_cold_violations: list[int]
    daily_warm_violations: list[int]

    # ── Learning metrics ─────────────────────────────────────────────
    daily_ff_fraction: list[float]  # |FF| / (|FF| + |integral|)
    daily_covariance_trace: list[float]  # tr(P) from RLS
    daily_buffer_utilization: list[float]  # buffer len / max_size

    # Per-batch snapshots
    batch_kappa: list[float]  # condition number at each batch
    batch_covariance_trace: list[float]  # tr(P) at each batch

    # ── Setpoint behavior metrics ────────────────────────────────────
    # We send IR setpoints to the HP's thermostat; we don't control the
    # compressor directly.  These measure our setpoint behavior, not
    # compressor cycling.
    daily_setpoint_limited_pct: list[float]  # % ticks at min or max SP (no room to adjust)
    daily_rapid_sp_changes: list[int]  # SP reversals within 30 min of each other
    total_rapid_sp_changes: int


# ── Default weather schedules ────────────────────────────────────────────


def diurnal_outdoor(tick: int, base_c: float, amplitude_c: float,
                    tick_minutes: float = TICK_MINUTES_DEFAULT) -> float:
    """Sinusoidal outdoor temp with multi-day weather-front drift.

    Coldest at 6AM, warmest at 3PM. Includes ±8°C 5-day weather drift
    to provide outdoor_delta diversity for batch WLS.
    """
    hour = (tick * tick_minutes / 60.0) % 24.0
    day = tick * tick_minutes / (60.0 * 24.0)
    weather_drift = 8.0 * math.sin(2 * math.pi * day / 5.0)
    return base_c + weather_drift + amplitude_c * math.cos(
        2 * math.pi * (hour - 15) / 24
    )


def diurnal_solar(tick: int, peak: float = 0.8,
                  tick_minutes: float = TICK_MINUTES_DEFAULT) -> float:
    """Solar proxy: 0 at night, peaks at noon. Variable cloud cover."""
    hour = (tick * tick_minutes / 60.0) % 24.0
    day = tick * tick_minutes / (60.0 * 24.0)
    if hour < 6 or hour > 18:
        return 0.0
    base = peak * math.sin(math.pi * (hour - 6) / 12)
    # Cloud factor varies by day (different period than weather drift)
    cloud = 0.5 + 0.5 * math.cos(2 * math.pi * day / 3.0 + 1.0)
    return base * cloud


# ── Metric helpers ───────────────────────────────────────────────────────


def _count_reversals(history: list[dict]) -> int:
    """Count HP setpoint direction changes."""
    prev_dir = 0
    count = 0
    for i in range(1, len(history)):
        delta = history[i]["hp_setpoint"] - history[i - 1]["hp_setpoint"]
        if delta > 0:
            d = 1
        elif delta < 0:
            d = -1
        else:
            continue
        if prev_dir != 0 and d != prev_dir:
            count += 1
        prev_dir = d
    return count


def _zero_crossing_rate(history: list[dict], field: str = "error") -> float:
    """Zero-crossings per hour of the error signal (oscillation proxy)."""
    if len(history) < 2:
        return 0.0
    crossings = 0
    for i in range(1, len(history)):
        if history[i - 1][field] * history[i][field] < 0:
            crossings += 1
    hours = len(history) * TICK_MINUTES_DEFAULT / 60.0
    return crossings / hours if hours > 0 else 0.0


# ── Core simulation ─────────────────────────────────────────────────────


def run_full_stack(
    config: FullStackConfig,
    checkpoints: list[Checkpoint] | None = None,
    csv_schedule: dict[str, list[tuple[float, float]]] | None = None,
) -> FullStackResult:
    """Run a full-stack simulation with the real PIController.

    Args:
        config: Simulation configuration.
        checkpoints: Optional checkpoint callbacks fired at intervals.
        csv_schedule: Optional CSV-derived schedules. Dict mapping field
            names ('outdoor_c', 'solar', or model input names) to sorted
            lists of (epoch_seconds, value) tuples. When provided, these
            override synthetic schedules. Tick dt is derived from the
            outdoor_c timestamps if present.

    Returns:
        FullStackResult with full history, rollups, and convergence data.
    """
    checkpoints = checkpoints or []

    # Resolve profile
    if config.profile_name in PROFILES_2R2C:
        profile = PROFILES_2R2C[config.profile_name]
    elif config.profile_name in PROFILES:
        profile = PROFILES[config.profile_name]
    else:
        raise ValueError(f"Unknown profile: {config.profile_name}")

    tick_min = config.tick_minutes
    n_ticks = int(config.n_days * 24 * 60 / tick_min)
    ticks_per_day = int(24 * 60 / tick_min)
    batch_interval_ticks = int(config.batch_interval_hours * 60 / tick_min)

    # Build model input config for PIController
    pi_model_inputs = []
    for mi in config.model_inputs:
        entry: dict = {
            "entity_id": mi.entity_id,
            "name": mi.name,
            "input_role": mi.input_role,
            "seed_heat": mi.seed_heat,
            "seed_cool": mi.seed_cool,
            "lag_tau": mi.lag_tau,
            "delta_from_room": mi.delta_from_room,
        }
        if mi.clamp_min is not None:
            entry["clamp_min"] = mi.clamp_min
        if mi.clamp_max is not None:
            entry["clamp_max"] = mi.clamp_max
        pi_model_inputs.append(entry)

    # Create adapter
    pi_config = {
        "pi_model_inputs": pi_model_inputs,
        "pi_outdoor_seed_heat": profile.true_seed,
        "pi_outdoor_seed_cool": profile.true_seed,
        "pi_ki": 0.15,
        "pi_kp": 1.5,
        "pi_deadband": DEADBAND,
        "pi_setpoint_weight": 0.3,
        **config.pi_overrides,
    }
    adapter = TasmotaPIAdapter(pi_config)
    pi = adapter._pi

    if config.relax_kappa_gate:
        pi._batch_kappa_threshold = 10000

    # Compute solar_gain for the thermal model from solar model inputs.
    # The thermal model applies this via 2R2C physics (30% air, 70% wall).
    solar_thermal_gain = sum(
        mi.true_thermal_effect for mi in config.model_inputs
        if mi.input_role == "solar"
    )

    # Create thermal model
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

    adapter.set_desired_temp(config.desired_c)
    adapter.set_mode(config.mode)

    # Resolve outdoor/solar schedules
    outdoor_fn = config.outdoor_schedule or (
        lambda t: diurnal_outdoor(t, config.outdoor_base_c,
                                  config.outdoor_diurnal_c, tick_min)
    )
    solar_fn = config.solar_schedule or (
        lambda t: diurnal_solar(t, tick_minutes=tick_min)
    )

    # Build disturbance lookup
    disturbance_map: dict[int, list[Disturbance]] = {}
    for d in config.disturbances:
        for t in range(d.start_tick, d.start_tick + d.duration_ticks):
            disturbance_map.setdefault(t, []).append(d)

    # True coefficients
    true_coefs = config.true_coefs
    if true_coefs is None:
        true_coefs = {"intercept": 0.0, "outdoor_delta": profile.true_seed}
        for mi in config.model_inputs:
            true_coefs[mi.name] = mi.true_ff_coef

    # ── Tracking state ───────────────────────────────────────────────
    history: list[dict] = []
    coef_trajectory: list[dict] = []
    batch_count = 0
    batches_to_converge: int | None = None
    integral_sq_sum = 0.0
    total_itae = 0.0
    total_violations = 0
    cold_violations = 0
    warm_violations = 0
    worst_undershoot = 0.0
    worst_overshoot = 0.0
    cur_violation_streak = 0
    longest_violation_streak = 0
    total_rapid_sp_changes = 0

    # Per-day accumulators
    day_itae = 0.0
    day_violations = 0
    day_cold_viols = 0
    day_warm_viols = 0
    day_integral_sq = 0.0
    day_ff_sum = 0.0
    day_ff_plus_int_sum = 0.0
    day_saturated = 0
    day_start_tick = 0

    daily_itae: list[float] = []
    daily_violations: list[int] = []
    daily_reversals: list[int] = []
    daily_mae: list[float] = []
    daily_integral_rms: list[float] = []
    daily_comfort_pct: list[float] = []
    daily_cold_violations: list[int] = []
    daily_warm_violations: list[int] = []
    daily_ff_fraction: list[float] = []
    daily_covariance_trace: list[float] = []
    daily_buffer_utilization: list[float] = []
    daily_setpoint_limited_pct: list[float] = []
    daily_rapid_sp_changes: list[int] = []

    batch_kappa: list[float] = []
    batch_covariance_trace: list[float] = []

    checkpoint_data: list[dict] = []
    last_checkpoint_tick = 0

    # ── Main loop ────────────────────────────────────────────────────

    for tick in range(n_ticks):
        dt_seconds = tick_min * 60.0

        # Update outdoor temp
        model.outdoor_temp = outdoor_fn(tick)

        # Compute model input values
        input_values: dict[str, float] = {}
        solar_proxy_value = 0.0
        extra_heat_rate = 0.0
        for mi in config.model_inputs:
            val = mi.schedule(tick) if mi.schedule is not None else 0.0
            input_values[mi.name] = val
            if mi.input_role == "solar":
                # Solar goes through the thermal model's 2R2C physics
                # (30% convective to air, 70% radiative to walls).
                solar_proxy_value += val
            else:
                # Non-solar inputs (stove, adjacent zone) add heat directly
                extra_heat_rate += mi.true_thermal_effect * val

        # Apply disturbances
        room_temp_offset = 0.0
        if tick in disturbance_map:
            for d in disturbance_map[tick]:
                if d.field == "room_temp_offset":
                    room_temp_offset = d.value
                elif d.field in input_values:
                    input_values[d.field] = d.value
                    # Update solar_proxy if the disturbance overrides it
                    for mi in config.model_inputs:
                        if mi.name == d.field and mi.input_role == "solar":
                            solar_proxy_value = d.value

        # Apply non-solar extra heat directly to room
        if extra_heat_rate != 0:
            model.room_temp += extra_heat_rate * tick_min

        # Read sensor (with optional disturbance offset)
        sensor_reading = model.read_sensor() + room_temp_offset

        # Controller tick
        adapter._sim_clock += dt_seconds
        adapter._entity._attr_current_temperature = sensor_reading
        pi._inputs.outdoor_temp = model.outdoor_temp

        # Mock model input entity states
        _mock_states: dict = {}
        for mi in config.model_inputs:
            val = input_values[mi.name]
            ms = type("MockState", (), {
                "state": str(val),
                "attributes": {"unit_of_measurement": None},
            })()
            _mock_states[mi.entity_id] = ms
        pi._hass.states.get = lambda eid, _s=_mock_states: _s.get(eid)

        original = _time.monotonic
        _time.monotonic = lambda: adapter._sim_clock
        try:
            adapter._loop.run_until_complete(pi._pi_tick())
        finally:
            _time.monotonic = original

        hp_setpoint = float(pi._hp_setpoint)

        # Advance thermal model (solar goes through 2R2C air/wall split)
        model.step(
            hp_setpoint=hp_setpoint,
            dt_minutes=tick_min,
            solar_proxy=solar_proxy_value,
            tick=tick,
            mode=config.mode,
        )

        # Metrics
        error = config.desired_c - model.room_temp
        abs_error = abs(error)
        integral_sq_sum += pi._pi_integral ** 2
        day_integral_sq += pi._pi_integral ** 2

        deadband_error = max(0, abs_error - DEADBAND)
        t_hours = (tick % ticks_per_day) * tick_min / 60.0  # day-relative
        day_itae += t_hours * deadband_error
        total_itae += tick * tick_min * deadband_error  # absolute

        is_violation = abs_error > DEADBAND
        if is_violation:
            total_violations += 1
            day_violations += 1
            cur_violation_streak += 1
            if cur_violation_streak > longest_violation_streak:
                longest_violation_streak = cur_violation_streak
            # Asymmetric: cold vs warm
            if error > 0:  # error = desired - room, positive = room too cold
                cold_violations += 1
                day_cold_viols += 1
                if error > worst_undershoot:
                    worst_undershoot = error
            else:
                warm_violations += 1
                day_warm_viols += 1
                if -error > worst_overshoot:
                    worst_overshoot = -error
        else:
            cur_violation_streak = 0

        # FF fraction: |FF| / (|FF| + |integral|)
        ff_abs = abs(pi._ff_offset)
        int_abs = abs(pi._pi_integral)
        denom = ff_abs + int_abs
        day_ff_sum += ff_abs
        day_ff_plus_int_sum += denom

        # Saturation: at min or max setpoint
        if hp_setpoint <= pi._min_temp_c or hp_setpoint >= pi._max_temp_c:
            day_saturated += 1

        # Record history
        history.append({
            "tick": tick,
            "room_temp": model.room_temp,
            "sensor_reading": sensor_reading,
            "hp_setpoint": hp_setpoint,
            "integral": pi._pi_integral,
            "ff_offset": pi._ff_offset,
            "ff_fraction": ff_abs / denom if denom > 0 else 0.0,
            "error": error,
            "outdoor": model.outdoor_temp,
            "d_term": getattr(pi, "_pi_d_filtered", 0.0),
            "rls_obs_count": pi._rls_heat.observation_count,
            **{f"input_{mi.name}": input_values[mi.name]
               for mi in config.model_inputs},
        })

        # Trigger batch WLS at intervals
        if tick > 0 and tick % batch_interval_ticks == 0:
            pi._run_batch_analysis()
            batch_count += 1

            # Snapshot coefficients
            _snapshot_coefs(pi, batch_count, config.model_inputs,
                            true_coefs, coef_trajectory)

            # κ and covariance trace at batch time
            rls = pi._rls_heat
            p_diag = rls.get_covariance_diagonal()
            batch_covariance_trace.append(sum(p_diag))
            kappa = pi._cached_kappa
            batch_kappa.append(kappa if kappa is not None else float("inf"))

            # Check convergence
            if batches_to_converge is None:
                snap = coef_trajectory[-1]
                converged = all(
                    abs(snap.get(name, 0) - true_val) <= 0.3
                    for name, true_val in true_coefs.items()
                    if name in snap and not snap.get(f"{name}_frozen", True)
                )
                if converged:
                    batches_to_converge = batch_count

        # End-of-day rollup
        if (tick + 1) % ticks_per_day == 0 and tick > 0:
            day_slice = history[day_start_tick:tick + 1]
            day_len = len(day_slice)
            day_errors = [abs(h["room_temp"] - config.desired_c)
                          for h in day_slice]

            daily_itae.append(day_itae)
            daily_violations.append(day_violations)
            daily_reversals.append(_count_reversals(day_slice))
            daily_mae.append(sum(day_errors) / day_len)
            daily_integral_rms.append(
                math.sqrt(day_integral_sq / ticks_per_day)
            )

            # Comfort
            in_deadband = sum(1 for e in day_errors if e <= DEADBAND)
            daily_comfort_pct.append(in_deadband / day_len * 100.0)
            daily_cold_violations.append(day_cold_viols)
            daily_warm_violations.append(day_warm_viols)

            # Learning
            daily_ff_fraction.append(
                day_ff_sum / day_ff_plus_int_sum
                if day_ff_plus_int_sum > 0 else 0.0
            )
            rls = pi._rls_heat
            p_diag = rls.get_covariance_diagonal()
            daily_covariance_trace.append(sum(p_diag))
            buf = pi._observation_buffer_heat
            buf_max = getattr(buf, "_max_size", 500)
            daily_buffer_utilization.append(
                len(buf) / buf_max if buf_max > 0 else 0.0
            )

            # Equipment
            daily_setpoint_limited_pct.append(day_saturated / day_len * 100.0)
            daily_rapid_sp_changes.append(
                _count_rapid_sp_changes(day_slice, tick_min)
            )
            total_rapid_sp_changes += daily_rapid_sp_changes[-1]

            # Reset accumulators
            day_itae = 0.0
            day_violations = 0
            day_cold_viols = 0
            day_warm_viols = 0
            day_integral_sq = 0.0
            day_ff_sum = 0.0
            day_ff_plus_int_sum = 0.0
            day_saturated = 0
            day_start_tick = tick + 1

        # Checkpoints
        current_day = tick // ticks_per_day
        for cp in checkpoints:
            cp_interval_ticks = cp.interval_days * ticks_per_day
            if (tick + 1) % cp_interval_ticks == 0 and tick > 0:
                # Build period metrics (since last checkpoint)
                period_slice = history[last_checkpoint_tick:tick + 1]
                period_errors = [abs(h["room_temp"] - config.desired_c)
                                 for h in period_slice]
                period_viols = sum(1 for e in period_errors if e > DEADBAND)
                period_itae_val = sum(
                    (i * tick_min / 60.0) * max(0, e - DEADBAND)
                    for i, e in enumerate(period_errors)
                )

                # Current coefficients
                current_coefs, current_errors = _get_coef_state(
                    pi, config.model_inputs, true_coefs
                )

                # Learning state for checkpoint
                rls_cp = pi._rls_heat
                p_diag_cp = rls_cp.get_covariance_diagonal()
                ff_a = abs(pi._ff_offset)
                int_a = abs(pi._pi_integral)
                denom_cp = ff_a + int_a
                buf_cp = pi._observation_buffer_heat
                buf_max_cp = getattr(buf_cp, "_max_size", 500)

                state = CheckpointState(
                    tick=tick,
                    day=current_day,
                    elapsed_days=(tick + 1) * tick_min / (60.0 * 24.0),
                    history=history,
                    coef_trajectory=coef_trajectory,
                    current_coefs=current_coefs,
                    true_coefs=true_coefs,
                    coef_errors=current_errors,
                    integral_rms=math.sqrt(
                        integral_sq_sum / (tick + 1)
                    ),
                    batch_count=batch_count,
                    batches_to_converge=batches_to_converge,
                    period_itae=period_itae_val,
                    period_violations=period_viols,
                    period_reversals=_count_reversals(period_slice),
                    period_mae=(sum(period_errors) / len(period_errors)
                                if period_errors else 0.0),
                    total_itae=total_itae,
                    total_violations=total_violations,
                    ff_fraction=ff_a / denom_cp if denom_cp > 0 else 0.0,
                    covariance_trace=sum(p_diag_cp),
                    buffer_utilization=(
                        len(buf_cp) / buf_max_cp if buf_max_cp > 0 else 0.0
                    ),
                    pi=pi,
                )
                cp.callback(state)
                last_checkpoint_tick = tick + 1

    # ── Final results ────────────────────────────────────────────────

    final_coefs, final_errors = _get_coef_state(
        pi, config.model_inputs, true_coefs
    )

    # Weekly rollups
    weekly_itae = _rollup(daily_itae, 7)
    weekly_violations = _rollup(daily_violations, 7)
    weekly_reversals = _rollup(daily_reversals, 7)

    # Comfort totals
    in_deadband_total = sum(1 for h in history
                           if abs(h["room_temp"] - config.desired_c) <= DEADBAND)
    comfort_pct = in_deadband_total / n_ticks * 100.0 if n_ticks else 0.0

    return FullStackResult(
        history=history,
        coef_trajectory=coef_trajectory,
        final_coefs=final_coefs,
        true_coefs=true_coefs,
        coef_errors=final_errors,
        integral_rms=math.sqrt(integral_sq_sum / n_ticks) if n_ticks else 0.0,
        total_itae=total_itae,
        total_violations=total_violations,
        total_reversals=_count_reversals(history),
        batches_to_converge=batches_to_converge,
        n_batches=batch_count,
        n_ticks=n_ticks,
        daily_itae=daily_itae,
        daily_violations=daily_violations,
        daily_reversals=daily_reversals,
        daily_mae=daily_mae,
        daily_integral_rms=daily_integral_rms,
        weekly_itae=weekly_itae,
        weekly_violations=weekly_violations,
        weekly_reversals=weekly_reversals,
        checkpoint_data=checkpoint_data,
        # Comfort
        comfort_hours_pct=comfort_pct,
        cold_violations=cold_violations,
        warm_violations=warm_violations,
        worst_undershoot=worst_undershoot,
        worst_overshoot=worst_overshoot,
        longest_violation_streak=longest_violation_streak,
        daily_comfort_pct=daily_comfort_pct,
        daily_cold_violations=daily_cold_violations,
        daily_warm_violations=daily_warm_violations,
        # Learning
        daily_ff_fraction=daily_ff_fraction,
        daily_covariance_trace=daily_covariance_trace,
        daily_buffer_utilization=daily_buffer_utilization,
        batch_kappa=batch_kappa,
        batch_covariance_trace=batch_covariance_trace,
        # Equipment
        daily_setpoint_limited_pct=daily_setpoint_limited_pct,
        daily_rapid_sp_changes=daily_rapid_sp_changes,
        total_rapid_sp_changes=total_rapid_sp_changes,
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _snapshot_coefs(pi, batch_count, model_inputs, true_coefs, trajectory):
    """Take a coefficient snapshot after a batch cycle."""
    coef_dict = pi._rls_heat.get_coefficients()
    names = ["intercept", "outdoor_delta"] + [mi.name for mi in model_inputs]
    snapshot = {"batch": batch_count}
    for idx, name in enumerate(names):
        if idx < pi._rls_heat.n:
            snapshot[name] = coef_dict[idx]
            snapshot[f"{name}_frozen"] = pi._rls_heat.frozen[idx]
    trajectory.append(snapshot)


def _get_coef_state(pi, model_inputs, true_coefs):
    """Get current coefficients and errors vs truth."""
    coef_dict = pi._rls_heat.get_coefficients()
    names = ["intercept", "outdoor_delta"] + [mi.name for mi in model_inputs]
    current = {}
    errors = {}
    for idx, name in enumerate(names):
        if idx < pi._rls_heat.n:
            val = coef_dict[idx]
            current[name] = val
            if name in true_coefs:
                errors[name] = abs(val - true_coefs[name])
    return current, errors


def _count_rapid_sp_changes(history: list[dict], tick_min: float,
                            window_min: float = 30.0) -> int:
    """Count setpoint reversals within `window_min` minutes of each other.

    Rapid setpoint changes cause unnecessary IR blasts to the HP's
    thermostat.  We don't control the compressor directly, but rapid
    setpoint oscillation is undesirable behavior regardless.
    """
    window_ticks = int(window_min / tick_min)
    reversal_ticks = []
    prev_dir = 0
    for i in range(1, len(history)):
        delta = history[i]["hp_setpoint"] - history[i - 1]["hp_setpoint"]
        if delta > 0:
            d = 1
        elif delta < 0:
            d = -1
        else:
            continue
        if prev_dir != 0 and d != prev_dir:
            reversal_ticks.append(i)
        prev_dir = d

    short = 0
    for i in range(1, len(reversal_ticks)):
        if reversal_ticks[i] - reversal_ticks[i - 1] <= window_ticks:
            short += 1
    return short


def _rollup(daily: list, period: int) -> list:
    """Sum daily values into period-sized buckets."""
    result = []
    for i in range(0, len(daily), period):
        chunk = daily[i:i + period]
        if isinstance(chunk[0], int):
            result.append(sum(chunk))
        else:
            result.append(sum(chunk))
    return result
