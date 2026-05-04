"""Synthetic-truth drivers for Phase 4 lite positive-control experiments.

Two modes (per Tier 1.3 plan):

  1. Synthetic excitation (POC) — fully-synthetic open-loop inputs designed
     to maximise informativeness for 1R1C identification. Decoupled from
     bundle data; tests methodology reach on cleanest possible signals.

  2. Bundle excitation — replace `room_temp_c` in a real `ZoneTelemetry`
     with a series simulated from a known-truth 1R1C kernel driven by the
     bundle's recorded outdoor/setpoint/solar trajectories. Tests whether
     the bundle's excitation pattern is sufficient for the methodology.

Both modes use the same Van Loan-discretized kernel as production
(`build_1r1c`) and add Gaussian measurement noise per `RCParams1R1C.sigma_v`.

Truth parameters for Tier 1.3 are literature-grounded (Levermore 2020
residential winter envelope), NOT calibrated from house data — see
project_phase4_data_survey.md for the calibration concerns this avoids.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tests.hvac_bench.empirical.data_loader import (
    REQUIRED_SIGNALS,
    ProxyVariant,
    ZoneInfo,
    ZoneTelemetry,
    derive_q_heat_proxy_w,
)
from tests.hvac_bench.empirical.rc_model import RCParams1R1C, build_1r1c
from tests.hvac_bench.house_profiles import HouseProfile2R2C
from tests.hvac_bench.thermal_model import ThermalModel


# ── Forward simulator ────────────────────────────────────────────────────


def simulate_1r1c_room_temp(
    params: RCParams1R1C,
    inputs: pd.DataFrame,
    *,
    dt: float = 300.0,
    initial_temp_c: float = 20.0,
    seed: int = 0,
) -> pd.Series:
    """Forward-simulate `room_temp_c` through a 1R1C kernel driven by inputs.

    `inputs` must contain columns `outdoor_temp_c_om`, `q_heat_proxy_w`,
    `shortwave_w_m2` (the runner's input convention). Returns a Series
    indexed identically to `inputs` with Gaussian measurement noise per
    `params.sigma_v`.

    Process noise is included if `params.sigma_w > 0` (Van Loan via
    `build_1r1c`). For positive-control tests we typically set sigma_w=0
    so the synthetic data has known measurement-only noise structure.
    """
    ss = build_1r1c(params, dt=dt)
    rng = np.random.default_rng(seed)

    n = len(inputs)
    u = inputs[["outdoor_temp_c_om", "q_heat_proxy_w", "shortwave_w_m2"]].to_numpy()
    x = np.array([[initial_temp_c]])
    obs = np.empty(n)

    process_chol = (
        np.linalg.cholesky(ss.Q) if (ss.Q > 0).any() else np.zeros_like(ss.Q)
    )
    process_active = bool((ss.Q > 0).any())

    for k in range(n):
        # Measurement: y_k = H x_k + v_k
        v_k = rng.standard_normal() * params.sigma_v
        obs[k] = (ss.H @ x).item() + v_k
        if k < n - 1:
            x = ss.A @ x + ss.B @ u[k].reshape(-1, 1)
            if process_active:
                w_k = process_chol @ rng.standard_normal((ss.A.shape[0], 1))
                x = x + w_k

    return pd.Series(obs, index=inputs.index, name="room_temp_c")


# ── Open-loop synthetic excitation (Tier 1.3a) ───────────────────────────


def make_open_loop_inputs(
    n_steps: int,
    *,
    dt: float = 300.0,
    seed: int = 0,
    start: datetime | None = None,
    outdoor_mean_c: float = -5.0,
    outdoor_diurnal_amp_c: float = 8.0,
    setpoint_levels_c: tuple[float, float] = (20.0, 22.0),
    setpoint_period_steps: int | None = None,
    solar_peak_wm2: float = 500.0,
    cloud_seed_period_h: float = 72.0,
    nominal_capacity_w: float = 3000.0,
    prbs_avg_period_steps: int = 12,
    prbs_duty_cycle: float = 0.65,
) -> pd.DataFrame:
    """Generate an open-loop synthetic excitation pattern for system ID.

    Designed to maximise informativeness for 1R1C parameter recovery
    (per Bacher-Madsen 2011 / Madsen-Holst 1995):

    - Setpoint: square-wave between two levels with a long period →
      persistent excitation for τ.
    - Outdoor: diurnal sinusoid → natural-decay information.
    - Solar: half-sine bell with multi-day cloud factor → β_solar
      identifiability not aliased with diurnal.
    - HP active: pseudo-random binary sequence (PRBS) with configurable
      average switching period and duty cycle. Decorrelated from solar
      diurnal pattern → q_scale identifiable independently. Standard SI
      literature recipe (Ljung 1999; Bacher-Madsen 2011 §2.4).
    - q_heat: nominal capacity when active (binary proxy at this layer).

    `prbs_avg_period_steps` controls the average dwell time between switches;
    default 12 steps × 5 min = 60 min average dwell. `prbs_duty_cycle` is
    the fraction of steps with HP on; default 0.65 (typical residential
    heating duty cycle in moderate weather).

    Returns a DataFrame with bundle's required-signal columns (suitable
    for wrapping into a ZoneTelemetry via `make_synthetic_zone_telemetry`).
    """
    if setpoint_period_steps is None:
        # Default: ~48-h square-wave at 5-min ticks → 576 steps
        setpoint_period_steps = int(48.0 * 3600.0 / dt)

    rng = np.random.default_rng(seed)
    if start is None:
        start = datetime(2026, 3, 21, 0, 0, tzinfo=timezone.utc)

    idx = pd.date_range(start=start, periods=n_steps, freq=f"{int(dt)}s", tz="UTC")
    t_s = np.arange(n_steps) * dt

    # Outdoor: diurnal sinusoid (period 24h)
    outdoor = outdoor_mean_c + outdoor_diurnal_amp_c * np.sin(
        2 * np.pi * t_s / (24 * 3600.0)
    )

    # Solar: half-sine within daylight hours (06:00–18:00 local) × cloud factor
    # local hour-of-day from t_s + start.hour
    hour_of_day = (start.hour + t_s / 3600.0) % 24.0
    in_daylight = (hour_of_day >= 6.0) & (hour_of_day <= 18.0)
    daylight_frac = np.clip((hour_of_day - 6.0) / 12.0, 0.0, 1.0)
    bell = np.where(in_daylight, np.sin(np.pi * daylight_frac), 0.0)
    cloud_period_steps = max(1, int(cloud_seed_period_h * 3600.0 / dt))
    raw_cloud = rng.standard_normal(n_steps // cloud_period_steps + 2)
    # Smooth piecewise-linear cloud factor in [0.3, 1.0]
    cloud_indices = np.arange(n_steps) // cloud_period_steps
    cloud_factor = 0.65 + 0.35 * np.tanh(raw_cloud[cloud_indices])
    shortwave = bell * cloud_factor * solar_peak_wm2

    # Setpoint: square-wave between levels
    cycles = (np.arange(n_steps) // setpoint_period_steps) % 2
    setpoint = np.where(cycles == 0, setpoint_levels_c[0], setpoint_levels_c[1])

    # HP active: two-state Markov chain with geometric dwell times.
    # Expected on-dwell = prbs_avg_period_steps; off-dwell scales to keep
    # stationary duty cycle = prbs_duty_cycle. Standard PRBS-style SI input
    # (Ljung 1999 §13.3; Bacher-Madsen 2011 §2.4): persistent excitation
    # for q_scale identifiability, decorrelated from outdoor diurnal pattern.
    if prbs_avg_period_steps <= 0:
        raise ValueError("prbs_avg_period_steps must be positive")
    if not 0.0 < prbs_duty_cycle < 1.0:
        raise ValueError("prbs_duty_cycle must be in (0, 1)")
    p_off = 1.0 / prbs_avg_period_steps  # P(switch off | currently on)
    p_on = p_off * prbs_duty_cycle / (1.0 - prbs_duty_cycle)  # symmetry
    hp_active = np.empty(n_steps, dtype=bool)
    state = bool(rng.random() < prbs_duty_cycle)
    for k in range(n_steps):
        hp_active[k] = state
        if state and rng.random() < p_off:
            state = False
        elif (not state) and rng.random() < p_on:
            state = True
    q_heat = hp_active.astype(float) * nominal_capacity_w

    df = pd.DataFrame(
        {
            # placeholder; overwritten by simulator with the actual evolved state
            "room_temp_c": np.full(n_steps, float(setpoint_levels_c[0])),
            "hp_setpoint_c": setpoint,
            "desired_temp_c": setpoint,
            "user_setpoint_c": setpoint,
            "mode": "heat",
            "hvac_action": "heating",
            "pi_integral": 0.0,
            "ff_offset": 0.0,
            "outdoor_temp_c_om": outdoor,
            "shortwave_w_m2": shortwave,
            "solar_gain_proxy": shortwave / 1000.0,
            "pellet_burning": False,
            "boiler_calling": False,
            "hp_active": hp_active,
            "q_heat_proxy_w": q_heat,
            "valid": True,
        },
        index=idx,
    )
    df.index.name = "ts_utc"

    # Verify required signals present (sanity for downstream consumers)
    missing = [c for c in REQUIRED_SIGNALS if c not in df.columns]
    if missing:
        raise AssertionError(f"open-loop synth missing required signals: {missing}")

    return df


def make_synthetic_zone_telemetry(
    params: RCParams1R1C,
    *,
    n_steps: int = 5184,  # 18 days at 5-min, mirrors bundle train window length
    dt: float = 300.0,
    seed: int = 0,
    initial_temp_c: float = 20.0,
    nominal_capacity_w: float = 3000.0,
    info: ZoneInfo | None = None,
) -> ZoneTelemetry:
    """Tier 1.3a: build a fully-synthetic ZoneTelemetry from literature truth.

    Combines `make_open_loop_inputs` + `simulate_1r1c_room_temp` and wraps
    in a ZoneTelemetry suitable for passing through `run_zone()`. The
    `info` defaults to a synthetic ZoneInfo; pass an existing one to
    parametrize.
    """
    inputs = make_open_loop_inputs(
        n_steps,
        dt=dt,
        seed=seed,
        nominal_capacity_w=nominal_capacity_w,
    )
    room_temp = simulate_1r1c_room_temp(
        params,
        inputs,
        dt=dt,
        initial_temp_c=initial_temp_c,
        seed=seed + 1,
    )
    inputs["room_temp_c"] = room_temp

    if info is None:
        info = ZoneInfo(
            name="synthetic_open_loop",
            condenser="synth",
            fit_target=True,
            exclusion_windows=(),
        )

    # Synthetic bundle path is a sentinel; not used by run_zone.
    return ZoneTelemetry(
        info=info,
        df=inputs,
        nominal_capacity_w=nominal_capacity_w,
        bundle_path=__file__,  # type: ignore[arg-type]
        applied_exclusions=(),
    )


# ── Bundle-excitation replacement (Tier 1.3b) ────────────────────────────


def replace_room_temp_with_synthetic(
    real_telemetry: ZoneTelemetry,
    params: RCParams1R1C,
    *,
    dt: float = 300.0,
    seed: int = 0,
    initial_temp_c: float | None = None,
) -> ZoneTelemetry:
    """Tier 1.3b: replace `room_temp_c` in a real ZoneTelemetry with a
    series simulated from `params` driven by the bundle's recorded
    outdoor/q_heat_proxy/shortwave inputs.

    All other columns (mode, setpoint, valid mask, exclusions) are
    preserved unchanged. The resulting telemetry has known-truth physics
    but the bundle's actual excitation pattern.

    `initial_temp_c` defaults to the first valid row's real `room_temp_c`.
    """
    df = real_telemetry.df.copy()
    if initial_temp_c is None:
        first_valid = df.loc[df["valid"], "room_temp_c"]
        initial_temp_c = (
            float(first_valid.iloc[0]) if len(first_valid) else 20.0
        )

    synth_room = simulate_1r1c_room_temp(
        params,
        df,
        dt=dt,
        initial_temp_c=initial_temp_c,
        seed=seed,
    )
    df["room_temp_c"] = synth_room

    return ZoneTelemetry(
        info=real_telemetry.info,
        df=df,
        nominal_capacity_w=real_telemetry.nominal_capacity_w,
        bundle_path=real_telemetry.bundle_path,
        applied_exclusions=real_telemetry.applied_exclusions,
    )


# ── Option A — proportional inverter HP forward simulator ────────────────


def simulate_inverter_hp_room_temp(
    inputs: pd.DataFrame,
    *,
    profile: HouseProfile2R2C,
    dt: float = 300.0,
    initial_temp_c: float = 20.0,
    seed: int = 0,
    solar_gain: float = 0.012,
    sensor_noise_sigma: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    """Forward-simulate `room_temp_c` using ThermalModel's proportional inverter
    HP physics with outdoor-temp capacity curve, NOT the linear 1R1C kernel.

    The HP modulates output proportional to (setpoint − room) and idles
    when room ≥ setpoint (heating). Capacity scales with outdoor temp via
    `profile.hp_capacity` (e.g. FUJITSU_HYPERHEAT_CAPACITY). HP cycling
    emerges from the physics — `inputs` need not provide an `hp_active`
    column (the realised series is returned).

    `inputs` must contain `outdoor_temp_c_om`, `hp_setpoint_c`, and either
    `solar_gain_proxy` or `shortwave_w_m2` (the solar input is taken from
    `solar_gain_proxy` if present, else `shortwave_w_m2 / 1000.0`).
    `dt` is in seconds; the underlying ThermalModel works in minutes.

    Returns (room_temp_c, hp_active) series indexed identically to `inputs`.
    """
    if "outdoor_temp_c_om" not in inputs.columns:
        raise ValueError("inputs must contain 'outdoor_temp_c_om'")
    if "hp_setpoint_c" not in inputs.columns:
        raise ValueError("inputs must contain 'hp_setpoint_c'")
    if "solar_gain_proxy" in inputs.columns:
        solar = inputs["solar_gain_proxy"].to_numpy()
    elif "shortwave_w_m2" in inputs.columns:
        solar = inputs["shortwave_w_m2"].to_numpy() / 1000.0
    else:
        raise ValueError(
            "inputs must contain 'solar_gain_proxy' or 'shortwave_w_m2'"
        )

    outdoor = inputs["outdoor_temp_c_om"].to_numpy()
    setpoint = inputs["hp_setpoint_c"].to_numpy()
    n = len(inputs)
    dt_min = dt / 60.0

    model = ThermalModel(
        profile=profile,
        initial_temp=initial_temp_c,
        outdoor_temp=float(outdoor[0]),
        sensor_noise_sigma=sensor_noise_sigma,
        noise_seed=seed,
        solar_gain=solar_gain,
    )

    room_temp = np.empty(n)
    hp_active = np.empty(n, dtype=bool)

    for k in range(n):
        # Update outdoor before stepping so capacity-curve sees current weather
        model.outdoor_temp = float(outdoor[k])
        # Snapshot whether HP is "active" this step (matches ThermalModel.step
        # internal logic with hp_lag_minutes=0 and head_sensor_offset=0).
        hp_active[k] = bool(model.room_temp < float(setpoint[k]))
        model.step(
            hp_setpoint=float(setpoint[k]),
            dt_minutes=dt_min,
            solar_proxy=float(solar[k]),
            stove_active=0.0,
            tick=k,
            mode="heat",
        )
        # Sensor reading (with noise) — what the empirical pipeline observes.
        room_temp[k] = model.read_sensor()

    room_series = pd.Series(room_temp, index=inputs.index, name="room_temp_c")
    active_series = pd.Series(hp_active, index=inputs.index, name="hp_active")
    return room_series, active_series


def make_inverter_hp_zone_telemetry(
    *,
    profile: HouseProfile2R2C,
    n_steps: int = 5184,
    dt: float = 300.0,
    seed: int = 0,
    initial_temp_c: float = 20.0,
    nominal_capacity_w: float = 1500.0,
    proxy_variant: ProxyVariant = "constant",
    solar_gain: float = 0.012,
    sensor_noise_sigma: float = 0.05,
    setpoint_levels_c: tuple[float, float] = (20.0, 22.0),
    outdoor_mean_c: float = -5.0,
    outdoor_diurnal_amp_c: float = 8.0,
    info: ZoneInfo | None = None,
) -> ZoneTelemetry:
    """Option A wrapper: build a ZoneTelemetry whose `room_temp_c` was
    simulated by a proportional inverter HP (with outdoor capacity curve),
    while the empirical pipeline sees only the linear `q_heat_proxy_w` form.

    Setpoint, outdoor, and solar excitation come from `make_open_loop_inputs`
    (square-wave setpoint between `setpoint_levels_c`, diurnal outdoor sinusoid,
    bell-curve solar with multi-day cloud factor). The PRBS `hp_active` from
    that helper is overwritten with the on/off cycling that emerges from
    the HP physics.

    `proxy_variant` selects how the linear empirical proxy approximates the
    realised inverter heat output ('constant' = `nominal·hp_active`;
    'setpoint_modulated' = `nominal·hp_active·clip((sp−T)/5, 0, 1)`).
    """
    inputs = make_open_loop_inputs(
        n_steps,
        dt=dt,
        seed=seed,
        nominal_capacity_w=nominal_capacity_w,
        setpoint_levels_c=setpoint_levels_c,
        outdoor_mean_c=outdoor_mean_c,
        outdoor_diurnal_amp_c=outdoor_diurnal_amp_c,
    )

    room_series, active_series = simulate_inverter_hp_room_temp(
        inputs,
        profile=profile,
        dt=dt,
        initial_temp_c=initial_temp_c,
        seed=seed + 1,
        solar_gain=solar_gain,
        sensor_noise_sigma=sensor_noise_sigma,
    )
    inputs["room_temp_c"] = room_series
    inputs["hp_active"] = active_series

    # Re-derive q_heat_proxy_w from the realised hp_active and room_temp,
    # mirroring the data_loader pipeline a real bundle goes through.
    inputs["q_heat_proxy_w"] = derive_q_heat_proxy_w(
        inputs["hp_active"],
        nominal_capacity_w,
        variant=proxy_variant,
        hp_setpoint_c=inputs["hp_setpoint_c"],
        room_temp_c=inputs["room_temp_c"],
    )

    if info is None:
        info = ZoneInfo(
            name="synthetic_inverter_hp",
            condenser="synth",
            fit_target=True,
            exclusion_windows=(),
        )

    missing = [c for c in REQUIRED_SIGNALS if c not in inputs.columns]
    if missing:
        raise AssertionError(f"inverter-HP synth missing required signals: {missing}")

    return ZoneTelemetry(
        info=info,
        df=inputs,
        nominal_capacity_w=nominal_capacity_w,
        bundle_path=__file__,  # type: ignore[arg-type]
        applied_exclusions=(),
    )
