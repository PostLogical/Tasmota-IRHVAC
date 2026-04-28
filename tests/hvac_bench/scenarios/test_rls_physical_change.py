"""Scenario: physical change after convergence + unmodeled disturbances.

Timeline (90 days):
- Days 1-30: Normal operation, system converges from correct seeds
- Days 31-60: Spring arrives — user opens windows randomly (unmodeled disturbance,
  not captured by any model input). Affects observations but coefficients shouldn't change.
- Days 61-90: Window replacement — new double-pane windows reduce heat loss.
  outdoor_delta coefficient actually drops (better R-value). System must adapt.

Tests whether online RLS helps detect real physical changes faster than batch-only,
and whether it's corrupted by unmodeled disturbances that batch filters out.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from tests.hvac_bench.full_stack_runner import (
    FullStackConfig, FullStackResult, ModelInputSpec,
    diurnal_outdoor, diurnal_solar, run_full_stack,
    TICK_MINUTES_DEFAULT,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.thermal_model import ThermalModel2R2C


# ── Custom thermal model with window opening + R-value change ────────────


TICKS_PER_DAY = int(24 * 60 / TICK_MINUTES_DEFAULT)
WINDOW_CHANGE_DAY = 60  # Day windows get replaced
DISTURBANCE_START_DAY = 30  # Day open-window disturbances begin


def _window_open_schedule(tick: int, seed: int = 123) -> float:
    """Random window-opening events starting at day 31.

    Returns extra heat loss rate (°C/min) when windows are open.
    - Downstairs: random 1-3 hour windows during day (10am-8pm)
    - Upstairs: random 1-2 hour windows at night (9pm-6am)
    - ~30% of eligible hours have an open window
    """
    day = tick * TICK_MINUTES_DEFAULT / (60 * 24)
    if day < DISTURBANCE_START_DAY:
        return 0.0

    hour = (tick * TICK_MINUTES_DEFAULT / 60.0) % 24.0
    rng = random.Random(seed + tick // 4)  # changes every hour

    # Daytime: downstairs windows (10am-8pm)
    if 10 <= hour < 20:
        if rng.random() < 0.3:
            return 0.015  # ~1°C/hr extra heat loss
    # Nighttime: upstairs windows (9pm-6am)
    elif hour >= 21 or hour < 6:
        if rng.random() < 0.2:
            return 0.010  # ~0.6°C/hr extra heat loss

    return 0.0


def _outdoor_with_spring(tick: int, base_c: float, amplitude_c: float) -> float:
    """Outdoor temp that warms toward spring.

    Days 1-30: winter (-5°C base)
    Days 31-60: early spring (warming +5°C over 30 days)
    Days 61-90: mid-spring (+5°C base)
    """
    day = tick * TICK_MINUTES_DEFAULT / (60.0 * 24.0)
    if day < 30:
        effective_base = base_c
    elif day < 60:
        progress = (day - 30) / 30.0
        effective_base = base_c + 5.0 * progress
    else:
        effective_base = base_c + 5.0

    return diurnal_outdoor(tick, effective_base, amplitude_c)


class PhysicalChangeModel(ThermalModel2R2C):
    """Extended thermal model with window opening and R-value change."""

    def __init__(self, *args, window_seed: int = 123,
                 new_r_value_factor: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self._window_seed = window_seed
        self._new_r_value_factor = new_r_value_factor
        self._window_change_applied = False

    def step(self, hp_setpoint: float, dt_minutes: float, tick: int = 0,
             solar_proxy: float = 0.0, mode: str = "heat", **kwargs) -> None:
        # Apply window replacement at day 61: increase tau_env (less heat loss)
        day = tick * dt_minutes / (60.0 * 24.0)
        if day >= WINDOW_CHANGE_DAY and not self._window_change_applied:
            from dataclasses import replace
            # Copy the profile so we don't mutate the module-level singleton
            new_tau = self.profile.tau_env / self._new_r_value_factor
            self.profile = replace(self.profile, tau_env=new_tau)
            self._window_change_applied = True

        # Apply open window heat loss
        window_loss = _window_open_schedule(tick, self._window_seed)
        if window_loss > 0:
            # Open window: extra heat loss proportional to indoor-outdoor delta
            delta = self.room_temp - self.outdoor_temp
            self.room_temp -= window_loss * delta * dt_minutes / 10.0

        super().step(hp_setpoint=hp_setpoint, dt_minutes=dt_minutes,
                     tick=tick, solar_proxy=solar_proxy, mode=mode)


# ── Comparison runner ────────────────────────────────────────────────────


@dataclass
class PhysicalChangeResult:
    """Results from a physical change scenario."""

    rls_on: FullStackResult
    rls_off: FullStackResult

    # Per-phase metrics
    phase1_comfort_on: float  # days 1-30
    phase1_comfort_off: float
    phase2_comfort_on: float  # days 31-60 (disturbances)
    phase2_comfort_off: float
    phase3_comfort_on: float  # days 61-90 (physical change)
    phase3_comfort_off: float

    # outdoor_delta at key points
    od_day30_on: float  # converged value before disturbances
    od_day30_off: float
    od_day60_on: float  # after disturbances, before physical change
    od_day60_off: float
    od_day90_on: float  # final value after adaptation
    od_day90_off: float
    od_true_before: float  # true outdoor_delta before window change
    od_true_after: float  # true outdoor_delta after window change


def _run_physical_change_comparison() -> PhysicalChangeResult:
    """Run the physical change scenario with RLS on vs off."""
    profile = PROFILES_2R2C["living_room"]
    n_days = 90
    new_r_factor = 0.6  # 40% less heat loss after window replacement

    # True outdoor_delta: -(UA/K_hp) = -profile.true_seed (negated for β convention)
    od_true_before = profile.true_seed
    od_true_after = profile.true_seed * new_r_factor

    config = FullStackConfig(
        n_days=n_days,
        profile_name="living_room",
        outdoor_base_c=-5.0,
        outdoor_diurnal_c=6.0,
        desired_c=20.5,
        noise_sigma=0.1,
        noise_seed=42,
        outdoor_schedule=lambda t: _outdoor_with_spring(t, -5.0, 6.0),
        solar_schedule=lambda t: diurnal_solar(t),
        model_inputs=[
            ModelInputSpec(
                name="Solar Proxy",
                entity_id="sensor.solar_proxy",
                input_role="solar",
                _true_ff_coef=-2.0,
                seed_heat=0.0,
                lag_tau=120,
                clamp_min=0,
                schedule=lambda t: diurnal_solar(t),
            ),
        ],
        pi_overrides={
            "pi_outdoor_seed_heat": profile.true_seed,
        },
        relax_kappa_gate=True,
    )

    # We can't use the standard run_full_stack because we need the custom
    # thermal model. Instead, run manually with the adapter.
    from tests.hvac_bench.adapters import TasmotaPIAdapter
    import time as _time

    results = {}
    for rls_enabled in [True, False]:
        pi_config = {
            "pi_model_inputs": [{
                "entity_id": "sensor.solar_proxy",
                "name": "Solar Proxy",
                "input_role": "solar",
                "seed_heat": 0.0,
                "seed_cool": 0.0,
                "lag_tau": 120,
                "delta_from_room": False,
                "clamp_min": 0,
            }],
            "pi_outdoor_seed_heat": profile.true_seed,
            "pi_outdoor_seed_cool": profile.true_seed,
            "pi_ki": 0.15,
            "pi_kp": 1.5,
            "pi_deadband": 0.5,
            "pi_setpoint_weight": 0.3,
        }
        adapter = TasmotaPIAdapter(pi_config)
        pi = adapter._pi
        pi._batch_kappa_threshold = 10000

        if not rls_enabled:
            pi._rls_online_learning = False

        model = PhysicalChangeModel(
            profile=profile,
            initial_temp=20.5,
            outdoor_temp=-5.0,
            sensor_noise_sigma=0.1,
            noise_seed=42,
            solar_gain=0.005,
            stove_gain=0.0,
            window_seed=123,
            new_r_value_factor=new_r_factor,
        )

        adapter.set_desired_temp(20.5)
        adapter.set_mode("heat")

        tick_min = TICK_MINUTES_DEFAULT
        n_ticks = int(n_days * 24 * 60 / tick_min)
        batch_interval = int(12 * 60 / tick_min)
        history = []
        coef_snapshots = []  # (day, outdoor_delta)

        for tick in range(n_ticks):
            dt_seconds = tick_min * 60.0
            model.outdoor_temp = _outdoor_with_spring(tick, -5.0, 6.0)
            solar_val = diurnal_solar(tick)

            sensor_reading = model.read_sensor()

            adapter._sim_clock += dt_seconds
            adapter._entity._attr_current_temperature = sensor_reading
            pi._inputs.outdoor_temp = model.outdoor_temp

            # Mock solar entity state
            _mock_states = {}
            ms = type("MockState", (), {
                "state": str(solar_val),
                "attributes": {"unit_of_measurement": None},
            })()
            _mock_states["sensor.solar_proxy"] = ms
            pi._hass.states.get = lambda eid, _s=_mock_states: _s.get(eid)

            original = _time.monotonic
            _time.monotonic = lambda: adapter._sim_clock
            try:
                adapter._loop.run_until_complete(pi._pi_tick())
            finally:
                _time.monotonic = original

            hp_setpoint = float(pi._hp_setpoint)
            model.step(hp_setpoint=hp_setpoint, dt_minutes=tick_min,
                       tick=tick, solar_proxy=solar_val, mode="heat")

            day = tick * tick_min / (60 * 24)
            error = 20.5 - model.room_temp
            history.append({
                "tick": tick,
                "day": day,
                "room_temp": model.room_temp,
                "hp_setpoint": hp_setpoint,
                "error": error,
                "integral": pi._pi_integral,
            })

            # Snapshot coefficients daily
            if tick % TICKS_PER_DAY == 0 and tick > 0:
                coefs = pi._rls_heat.get_coefficients()
                coef_snapshots.append((day, coefs.get(1, 0)))  # index 1 = outdoor_delta physical

            # Trigger batch every 12h
            if tick > 0 and tick % batch_interval == 0:
                pi._run_batch_analysis()

        # Extract phase comfort
        ticks_per_phase = n_ticks // 3
        phase1 = history[:ticks_per_phase]
        phase2 = history[ticks_per_phase:2*ticks_per_phase]
        phase3 = history[2*ticks_per_phase:]

        def comfort_pct(phase):
            in_band = sum(1 for h in phase if abs(h["error"]) <= 0.5)
            return 100.0 * in_band / len(phase) if phase else 0.0

        # Get outdoor_delta at phase boundaries
        od_day30 = coef_snapshots[29][1] if len(coef_snapshots) > 29 else 0.0
        od_day60 = coef_snapshots[59][1] if len(coef_snapshots) > 59 else 0.0
        od_day90 = coef_snapshots[-1][1] if coef_snapshots else 0.0

        results[rls_enabled] = {
            "comfort_p1": comfort_pct(phase1),
            "comfort_p2": comfort_pct(phase2),
            "comfort_p3": comfort_pct(phase3),
            "od_day30": od_day30,
            "od_day60": od_day60,
            "od_day90": od_day90,
            "coef_snapshots": coef_snapshots,
            "history": history,
        }

    on = results[True]
    off = results[False]

    return PhysicalChangeResult(
        rls_on=None,  # type: ignore  # not using FullStackResult here
        rls_off=None,  # type: ignore
        phase1_comfort_on=on["comfort_p1"],
        phase1_comfort_off=off["comfort_p1"],
        phase2_comfort_on=on["comfort_p2"],
        phase2_comfort_off=off["comfort_p2"],
        phase3_comfort_on=on["comfort_p3"],
        phase3_comfort_off=off["comfort_p3"],
        od_day30_on=on["od_day30"],
        od_day30_off=off["od_day30"],
        od_day60_on=on["od_day60"],
        od_day60_off=off["od_day60"],
        od_day90_on=on["od_day90"],
        od_day90_off=off["od_day90"],
        od_true_before=od_true_before,
        od_true_after=od_true_after,
    )


class TestPhysicalChange:
    """Test online RLS value for detecting real physical changes."""

    def test_physical_change_adaptation(self):
        """After window replacement, does online RLS adapt faster?"""
        r = _run_physical_change_comparison()

        print(f"\n{'='*70}")
        print(f"  Physical Change Scenario (90 days)")
        print(f"  Days 1-30: converge | Days 31-60: open windows | Days 61-90: new windows")
        print(f"{'='*70}")
        print(f"  True outdoor_delta: before={r.od_true_before:.4f}, after={r.od_true_after:.4f}")
        print(f"\n{'Phase':<20} {'RLS ON':>10} {'RLS OFF':>10} {'Delta':>10}")
        print(f"{'-'*50}")
        print(f"{'Phase 1 comfort':<20} {r.phase1_comfort_on:>9.1f}% {r.phase1_comfort_off:>9.1f}% {r.phase1_comfort_off - r.phase1_comfort_on:>+9.1f}%")
        print(f"{'Phase 2 comfort':<20} {r.phase2_comfort_on:>9.1f}% {r.phase2_comfort_off:>9.1f}% {r.phase2_comfort_off - r.phase2_comfort_on:>+9.1f}%")
        print(f"{'Phase 3 comfort':<20} {r.phase3_comfort_on:>9.1f}% {r.phase3_comfort_off:>9.1f}% {r.phase3_comfort_off - r.phase3_comfort_on:>+9.1f}%")

        print(f"\n{'outdoor_delta trajectory:'}")
        print(f"  {'Day 30 (converged)':<25} ON={r.od_day30_on:>8.4f}  OFF={r.od_day30_off:>8.4f}  true={r.od_true_before:.4f}")
        print(f"  {'Day 60 (post-disturb)':<25} ON={r.od_day60_on:>8.4f}  OFF={r.od_day60_off:>8.4f}  true={r.od_true_before:.4f}")
        print(f"  {'Day 90 (post-change)':<25} ON={r.od_day90_on:>8.4f}  OFF={r.od_day90_off:>8.4f}  true={r.od_true_after:.4f}")

        # How close did each get to the new true value?
        err_on = abs(r.od_day90_on - (-r.od_true_after))  # negate for beta convention
        err_off = abs(r.od_day90_off - (-r.od_true_after))
        print(f"\n  Day 90 error vs new truth:")
        print(f"    RLS ON:  {err_on:.4f}")
        print(f"    RLS OFF: {err_off:.4f}")
        print(f"    {'RLS ON wins' if err_on < err_off else 'RLS OFF wins (or tie)'}")

        # Sanity: both should maintain comfort > 70%
        assert r.phase1_comfort_on > 70.0
        assert r.phase1_comfort_off > 70.0
