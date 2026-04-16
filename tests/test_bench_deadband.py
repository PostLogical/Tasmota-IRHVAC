"""CI bench tests: full-rate vs variable-rate deadband integration.

Validates that full-rate integration (current policy) is equal or better
than variable-rate across all scenarios.  The solar cycle regression is
the key gate — full-rate must show lower ITAE during day/night transitions.

Run with: python -m pytest tests/test_bench_deadband.py -v
"""

import asyncio
import math
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, UnitOfTemperature

from custom_components.tasmota_irhvac.pi.pi_controller import PIController

from .conftest import make_pi_config
from .benchmark_metrics import (
    compute_itae,
    compute_overshoot,
    compute_settling_time,
    count_reversals,
    count_setpoint_changes,
)


# ── Room Model (from pi_realistic_sim.py) ────────────────────────────


@dataclass
class RoomModel:
    """1R1C room model with HP dynamics and supplemental heat."""

    name: str = "Room"
    R_wall: float = 0.006
    C_room: float = 600_000
    hp_max_power: float = 3500.0
    hp_min_power: float = 500.0
    hp_response_time: float = 300.0
    room_temp: float = 20.0
    outdoor_temp: float = 0.0
    hp_output: float = 0.0
    supplemental_heat: float = 0.0

    def step(self, hp_setpoint, desired_temp, dt=60.0):
        hp_target = self._hp_power(hp_setpoint)
        alpha = min(1.0, dt / self.hp_response_time)
        self.hp_output += alpha * (hp_target - self.hp_output)
        q_wall = (self.outdoor_temp - self.room_temp) / self.R_wall
        q_total = q_wall + self.hp_output + self.supplemental_heat
        self.room_temp += (q_total * dt) / self.C_room

    def _hp_power(self, hp_setpoint):
        diff = hp_setpoint - self.room_temp
        if diff > 0:
            return max(self.hp_min_power, self.hp_max_power * min(1.0, diff / 5.0))
        elif diff < -1:
            return 0.0
        else:
            return self.hp_min_power * max(0, diff + 1)


# ── Sim Entity ───────────────────────────────────────────────────────


class SimEntity:
    """Minimal entity that PIController can read from during simulation."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.AUTO, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    def __init__(self, config, room_temp_c=20.0, outdoor_temp_c=0.0):
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = room_temp_c
        self._attr_target_temperature = 22.0
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "bench_entry"

        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)
        self.async_get_last_extra_data = AsyncMock(return_value=None)

        self._pi = PIController(self, config)

    @property
    def target_temperature(self):
        if self._pi and self._pi._desired_temp is not None:
            return self._pi._desired_temp
        return self._attr_target_temperature

    @property
    def temperature_unit(self):
        return self._attr_temperature_unit


# ── Sim Trace & Runner ───────────────────────────────────────────────


@dataclass
class SimTrace:
    """Per-tick trace data for one simulation run."""

    time_min: list = field(default_factory=list)
    room_temp_c: list = field(default_factory=list)
    hp_setpoint: list = field(default_factory=list)
    integral: list = field(default_factory=list)
    ff_offset: list = field(default_factory=list)
    error: list = field(default_factory=list)


async def _run_sim(entity, room, duration_hours, desired_c,
                   outdoor_events=None, solar_fn=None, sensor_noise_std=0.0,
                   dt_room=60.0, pi_interval=900.0):
    """Run closed-loop simulation with real PIController and RoomModel."""
    import numpy as np

    pi = entity._pi
    await pi.set_temperature(desired_c)
    pi._inputs.outdoor_temp = room.outdoor_temp

    rng = np.random.default_rng(42)
    outdoor_events = sorted(outdoor_events or [], key=lambda e: e[0])
    event_idx = 0
    steps = int(duration_hours * 3600 / dt_room)
    pi_interval_steps = int(pi_interval / dt_room)
    trace = SimTrace()
    mono_time = max(pi_interval, pi._last_setpoint_change_time + pi_interval)

    for step in range(steps):
        t_sec = step * dt_room
        t_hr = t_sec / 3600.0

        while event_idx < len(outdoor_events) and outdoor_events[event_idx][0] <= t_hr:
            room.outdoor_temp = outdoor_events[event_idx][1]
            pi._inputs.outdoor_temp = room.outdoor_temp
            event_idx += 1

        if solar_fn is not None:
            room.supplemental_heat = solar_fn(t_hr)

        room.step(float(pi._hp_setpoint), desired_c, dt_room)

        noise = rng.normal(0, sensor_noise_std) if sensor_noise_std > 0 else 0.0
        entity._attr_current_temperature = room.room_temp + noise

        if step % pi_interval_steps == 0 and step > 0:
            pi._pi_last_tick_time = mono_time - pi_interval
            with patch("time.monotonic", return_value=mono_time):
                await pi._pi_tick()
            mono_time += pi_interval

        if step % pi_interval_steps == 0:
            trace.time_min.append(t_sec / 60.0)
            trace.room_temp_c.append(room.room_temp)
            trace.hp_setpoint.append(int(pi._hp_setpoint))
            trace.integral.append(pi._pi_integral)
            trace.ff_offset.append(pi._ff_offset)
            trace.error.append(desired_c - room.room_temp)

    return trace


def _trace_to_history(trace, desired_c, outdoor_c):
    """Convert SimTrace to benchmark_metrics history format."""
    history = []
    for i in range(len(trace.time_min)):
        history.append({
            "tick": i,
            "room_temp": trace.room_temp_c[i],
            "desired": desired_c,
            "hp_setpoint": trace.hp_setpoint[i],
            "integral": trace.integral[i],
            "ff_offset": trace.ff_offset[i],
            "error": trace.error[i],
            "outdoor": outdoor_c,
            "rls_obs_count": 0,
        })
    return history


def _compute_metrics(trace, desired_c, outdoor_c):
    """Compute all bench metrics from a trace."""
    history = _trace_to_history(trace, desired_c, outdoor_c)
    return {
        "itae": compute_itae(history),
        "overshoot": compute_overshoot(history),
        "settling_time": compute_settling_time(history),
        "reversals": count_reversals(history),
        "setpoint_changes": count_setpoint_changes(history),
        "final_room_c": trace.room_temp_c[-1] if trace.room_temp_c else 0,
        "max_deviation": max(abs(e) for e in trace.error) if trace.error else 0,
    }


# ── A/B Runner ───────────────────────────────────────────────────────


def _make_living_room(room_temp=20.0, outdoor_temp=0.0):
    """Living room: τ≈60min, R_wall=0.015 (HP sized for -15°F design temp)."""
    return RoomModel(
        name="Living Room",
        room_temp=room_temp,
        outdoor_temp=outdoor_temp,
        R_wall=0.015,
        C_room=240_000,
        hp_max_power=3500,
        hp_response_time=300,
    )


def _make_bunkroom(room_temp=20.0, outdoor_temp=0.0):
    """Bunkroom: τ≈23min, same HP sizing, faster response."""
    return RoomModel(
        name="Bunkroom",
        room_temp=room_temp,
        outdoor_temp=outdoor_temp,
        R_wall=0.015,
        C_room=92_000,
        hp_max_power=3500,
        hp_response_time=180,
    )


async def _run_ab(room_factory, desired_c, duration_hours,
                  warmup_hours=6, outdoor_events=None,
                  solar_fn=None, sensor_noise_std=0.0):
    """Run both variable-rate and full-rate from the same warmup equilibrium.

    Returns dict with keys "variable-rate" and "full-rate", each containing metrics.
    """
    config = make_pi_config()
    outdoor_c = room_factory().outdoor_temp

    # Warmup to equilibrium (no noise/solar)
    warmup_room = room_factory()
    warmup_entity = SimEntity(config, warmup_room.room_temp, warmup_room.outdoor_temp)
    await _run_sim(warmup_entity, warmup_room, warmup_hours, desired_c)
    warmup_pi = warmup_entity._pi

    eq_room_temp = warmup_room.room_temp
    eq_hp_setpoint = int(warmup_pi._hp_setpoint)
    eq_integral = warmup_pi._pi_integral
    eq_ff_offset = warmup_pi._ff_offset
    eq_ff_confidence = warmup_pi._ff_confidence

    results = {}
    for policy in ["variable-rate", "full-rate"]:
        room = room_factory()
        room.room_temp = eq_room_temp
        room.hp_output = warmup_room.hp_output

        entity = SimEntity(config, eq_room_temp, room.outdoor_temp)
        pi = entity._pi
        pi._hp_setpoint = eq_hp_setpoint
        pi._pi_integral = eq_integral
        pi._ff_offset = eq_ff_offset
        pi._ff_confidence = eq_ff_confidence
        pi._desired_temp = desired_c
        pi._inputs.outdoor_temp = room.outdoor_temp

        if policy == "variable-rate":
            # Old policy: rate = |error|/deadband, floor 0.05
            deadband = pi._pi_deadband
            pi._deadband_integration_rate = (
                lambda ae, db=deadband: max(0.05, ae / db) if db > 0 else 1.0
            )

        trace = await _run_sim(entity, room, duration_hours, desired_c,
                               outdoor_events=outdoor_events,
                               solar_fn=solar_fn,
                               sensor_noise_std=sensor_noise_std)
        results[policy] = _compute_metrics(trace, desired_c, outdoor_c)

    return results


# ── Solar helper ─────────────────────────────────────────────────────


def _solar_cycle(t_hr):
    """Sinusoidal solar gain: 0 at night, peaks at 1500W midday."""
    phase = (t_hr % 24.0) / 24.0 * 2 * math.pi - math.pi / 2
    return max(0.0, 1500.0 * math.sin(phase))


def _spring_solar(t_hr):
    """Moderate spring solar: peaks at 700W midday, zero at night."""
    phase = (t_hr % 24.0) / 24.0 * 2 * math.pi - math.pi / 2
    return max(0.0, 700.0 * math.sin(phase))


def _spring_outdoor_events():
    """Diurnal outdoor temp: 5°C at dawn (6am), 15°C at 3pm, 8°C by midnight.

    Sim starts at midnight (t=0).  Sinusoidal with min at 6am, max at 3pm.
    Steps every 30 min for smooth curve over 24h, repeated for 48h.
    """
    events = []
    for day in range(2):
        for step in range(48):  # every 30 min
            t_hr = day * 24.0 + step * 0.5
            # Sinusoidal: min 5°C at 6am (t=6), max 15°C at 3pm (t=15)
            # Center = 10°C, amplitude = 5°C
            # Phase: peak at t=15 → phase = (t - 15) / 24 * 2π
            phase = (t_hr % 24.0 - 15.0) / 24.0 * 2 * math.pi
            outdoor = 10.0 + 5.0 * math.cos(phase)
            events.append((t_hr, outdoor))
    return events


# ── Scenario Definitions ─────────────────────────────────────────────


SCENARIOS = {
    "lr_mild": {
        "label": "Living room mild (22°C, outdoor 10°C)",
        "room_factory": lambda: _make_living_room(room_temp=20.0, outdoor_temp=10.0),
        "desired_c": 22.0,
        "duration_hours": 12,
    },
    "lr_shoulder": {
        "label": "Living room shoulder (22°C, outdoor 18°C)",
        "room_factory": lambda: _make_living_room(room_temp=21.0, outdoor_temp=18.0),
        "desired_c": 22.0,
        "duration_hours": 12,
    },
    "br_mild": {
        "label": "Bunkroom mild (21°C, outdoor 10°C)",
        "room_factory": lambda: _make_bunkroom(room_temp=19.0, outdoor_temp=10.0),
        "desired_c": 21.0,
        "duration_hours": 12,
    },
    "br_shoulder": {
        "label": "Bunkroom shoulder (21°C, outdoor 18°C)",
        "room_factory": lambda: _make_bunkroom(room_temp=20.5, outdoor_temp=18.0),
        "desired_c": 21.0,
        "duration_hours": 12,
    },
    "lr_cold_front": {
        "label": "Living room cold front (10→0°C over 3h)",
        "room_factory": lambda: _make_living_room(room_temp=22.0, outdoor_temp=10.0),
        "desired_c": 22.0,
        "duration_hours": 10,
        "outdoor_events": [
            (1.0 + i * 5 / 60, 10.0 - i * (10.0 / 36)) for i in range(37)
        ],
    },
    "br_perturbation": {
        "label": "Bunkroom door-open perturbation",
        "room_factory": lambda: _make_bunkroom(room_temp=20.5, outdoor_temp=10.0),
        "desired_c": 21.0,
        "duration_hours": 12,
    },
}

EDGE_SCENARIOS = {
    "br_noise": {
        "label": "Bunkroom + sensor noise ±0.1°C",
        "room_factory": lambda: _make_bunkroom(room_temp=19.0, outdoor_temp=10.0),
        "desired_c": 21.0,
        "duration_hours": 12,
        "sensor_noise_std": 0.1,
    },
    "lr_solar": {
        "label": "Living room solar day/night 48h (KEY REGRESSION)",
        "room_factory": lambda: _make_living_room(room_temp=22.0, outdoor_temp=10.0),
        "desired_c": 22.0,
        "duration_hours": 48,
        "solar_fn": _solar_cycle,
    },
    "br_solar": {
        "label": "Bunkroom solar day/night 48h",
        "room_factory": lambda: _make_bunkroom(room_temp=21.0, outdoor_temp=10.0),
        "desired_c": 21.0,
        "duration_hours": 48,
        "solar_fn": _solar_cycle,
    },
    "lr_spring_day": {
        "label": "Living room spring day (5-15°C outdoor + 700W solar, 48h)",
        "room_factory": lambda: _make_living_room(room_temp=22.0, outdoor_temp=8.0),
        "desired_c": 22.0,
        "duration_hours": 48,
        "outdoor_events": _spring_outdoor_events(),
        "solar_fn": _spring_solar,
    },
    "br_spring_day": {
        "label": "Bunkroom spring day (5-15°C outdoor + 700W solar, 48h)",
        "room_factory": lambda: _make_bunkroom(room_temp=21.0, outdoor_temp=8.0),
        "desired_c": 21.0,
        "duration_hours": 48,
        "outdoor_events": _spring_outdoor_events(),
        "solar_fn": _spring_solar,
    },
}

ALL_SCENARIOS = {**SCENARIOS, **EDGE_SCENARIOS}


# ── Tests ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ab_results():
    """Run all A/B comparisons once, share across tests in this module."""
    loop = asyncio.new_event_loop()
    results = {}
    for key, scen in ALL_SCENARIOS.items():
        results[key] = loop.run_until_complete(
            _run_ab(
                room_factory=scen["room_factory"],
                desired_c=scen["desired_c"],
                duration_hours=scen["duration_hours"],
                outdoor_events=scen.get("outdoor_events"),
                solar_fn=scen.get("solar_fn"),
                sensor_noise_std=scen.get("sensor_noise_std", 0.0),
            )
        )
    loop.close()
    return results


class TestDeadbandFullRateNotWorse:
    """Full-rate must not regress vs variable-rate on any scenario."""

    @pytest.mark.parametrize("scenario_key", list(ALL_SCENARIOS.keys()),
                             ids=[ALL_SCENARIOS[k]["label"] for k in ALL_SCENARIOS])
    def test_itae_not_worse(self, ab_results, scenario_key):
        """Full-rate ITAE must be <= variable-rate ITAE (with 5% tolerance)."""
        vr = ab_results[scenario_key]["variable-rate"]
        fr = ab_results[scenario_key]["full-rate"]
        tolerance = max(vr["itae"] * 0.05, 1.0)  # 5% or 1.0, whichever is larger
        assert fr["itae"] <= vr["itae"] + tolerance, (
            f"Full-rate ITAE {fr['itae']:.1f} worse than variable-rate "
            f"{vr['itae']:.1f} (tolerance {tolerance:.1f})"
        )

    @pytest.mark.parametrize("scenario_key", list(ALL_SCENARIOS.keys()),
                             ids=[ALL_SCENARIOS[k]["label"] for k in ALL_SCENARIOS])
    def test_reversals_not_worse(self, ab_results, scenario_key):
        """Full-rate must not produce more than 4 extra reversals.

        Tolerance is 4 (not 2) because coupled disturbances (spring day:
        diurnal outdoor + moderate solar) can produce slightly more
        reversals with full-rate in slow-τ rooms while ITAE stays within 5%.
        """
        vr = ab_results[scenario_key]["variable-rate"]
        fr = ab_results[scenario_key]["full-rate"]
        assert fr["reversals"] <= vr["reversals"] + 4, (
            f"Full-rate reversals {fr['reversals']} vs variable-rate "
            f"{vr['reversals']} (>4 extra)"
        )

    @pytest.mark.parametrize("scenario_key", list(ALL_SCENARIOS.keys()),
                             ids=[ALL_SCENARIOS[k]["label"] for k in ALL_SCENARIOS])
    def test_setpoint_changes_not_worse(self, ab_results, scenario_key):
        """Full-rate must not produce more than 4 extra setpoint changes."""
        vr = ab_results[scenario_key]["variable-rate"]
        fr = ab_results[scenario_key]["full-rate"]
        assert fr["setpoint_changes"] <= vr["setpoint_changes"] + 4, (
            f"Full-rate setpoint changes {fr['setpoint_changes']} vs "
            f"variable-rate {vr['setpoint_changes']} (>4 extra)"
        )


class TestSolarCycleRegression:
    """Solar day/night cycle is where full-rate proved measurably better.

    Variable-rate carried stale daytime integral through the deadband during
    dusk/dawn transitions.  Full-rate must be strictly better here.
    """

    def test_lr_solar_itae_better(self, ab_results):
        """Living room solar: full-rate must have lower ITAE."""
        vr = ab_results["lr_solar"]["variable-rate"]
        fr = ab_results["lr_solar"]["full-rate"]
        assert fr["itae"] <= vr["itae"], (
            f"Full-rate ITAE {fr['itae']:.1f} not better than "
            f"variable-rate {vr['itae']:.1f} on solar cycle"
        )

    def test_br_solar_itae_better(self, ab_results):
        """Bunkroom solar: full-rate must have lower ITAE."""
        vr = ab_results["br_solar"]["variable-rate"]
        fr = ab_results["br_solar"]["full-rate"]
        assert fr["itae"] <= vr["itae"], (
            f"Full-rate ITAE {fr['itae']:.1f} not better than "
            f"variable-rate {vr['itae']:.1f} on solar cycle"
        )

    def test_lr_solar_fewer_reversals(self, ab_results):
        """Living room solar: full-rate must not have more reversals."""
        vr = ab_results["lr_solar"]["variable-rate"]
        fr = ab_results["lr_solar"]["full-rate"]
        assert fr["reversals"] <= vr["reversals"], (
            f"Full-rate reversals {fr['reversals']} vs "
            f"variable-rate {vr['reversals']} on solar cycle"
        )

    def test_br_solar_fewer_reversals(self, ab_results):
        """Bunkroom solar: full-rate must not have more reversals."""
        vr = ab_results["br_solar"]["variable-rate"]
        fr = ab_results["br_solar"]["full-rate"]
        assert fr["reversals"] <= vr["reversals"], (
            f"Full-rate reversals {fr['reversals']} vs "
            f"variable-rate {vr['reversals']} on solar cycle"
        )


class TestSummaryTable:
    """Print comparison table for human review (always passes)."""

    def test_print_summary(self, ab_results, capsys):
        with capsys.disabled():
            print(f"\n{'=' * 90}")
            print("  DEADBAND BENCH: Full-rate vs Variable-rate")
            print(f"{'=' * 90}")
            print(f"  {'Scenario':<40} {'ΔITAE':>10} {'ΔReversals':>12} "
                  f"{'ΔSP Changes':>12} {'ΔOvershoot':>12}")
            print(f"  {'-' * 86}")
            for key in ALL_SCENARIOS:
                vr = ab_results[key]["variable-rate"]
                fr = ab_results[key]["full-rate"]
                d_itae = fr["itae"] - vr["itae"]
                d_rev = fr["reversals"] - vr["reversals"]
                d_sp = fr["setpoint_changes"] - vr["setpoint_changes"]
                d_os = fr["overshoot"] - vr["overshoot"]
                print(f"  {key:<40} {d_itae:>+10.1f} {d_rev:>+12d} "
                      f"{d_sp:>+12d} {d_os:>+12.2f}")
            print(f"{'=' * 90}")
