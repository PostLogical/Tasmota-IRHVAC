"""Core cooling scenario tests.

Mirror of heating scenarios with inverted conditions.
Room is too warm, outdoor is hot, HP needs to cool.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


SEED_FACTORS = [0.0, 0.5, 1.0, 1.5]


def _make_controller(profile, seed_factor=1.0, **overrides):
    seed = profile.true_seed
    config = {
        "pi_outdoor_seed_heat": seed * seed_factor,
        "pi_outdoor_seed_cool": seed * seed_factor,
        **overrides,
    }
    ctrl = TasmotaPIAdapter(config)
    return ctrl


def _make_model(profile, initial_temp=24.0, outdoor=32.0, **kwargs):
    """Build a 2R2C thermal model for the cooling tests.

    Mirror of test_heating's _make_model.  Default ``hp_lag_minutes=2.0``
    (typical inverter compressor spool); other realism knobs left at
    idealized defaults.
    """
    kwargs.setdefault("hp_lag_minutes", 2.0)
    return ThermalModel(profile=profile, initial_temp=initial_temp,
                        outdoor_temp=outdoor, **kwargs)


def _record_run(bench_metrics, history, *, profile_name, seed_factor,
                desired, deadband=0.5):
    """Record parametrization + control-quality rollup from a cooling run."""
    bench_metrics["profile_name"] = profile_name
    bench_metrics["seed_factor"] = seed_factor
    bench_metrics["n_ticks"] = len(history)
    rollup = compute_all_metrics(history, desired=desired, deadband=deadband)
    for k, v in rollup.items():
        bench_metrics[f"rollup_{k}"] = v


# ── Warm Start (cool down) ───────────────────────────────────────────────


class TestCoolingWarmStart:
    """Room starts warm (28°C), needs to cool to 24°C."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_warm_start(self, bench_metrics, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=28.0, outdoor=32.0)

        # 8h cool-down (was n_ticks=32 at 15-min cadence).
        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="cool")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=24.0)

        final_error = abs(history[-1]["room_temp"] - 24.0)
        bench_metrics["final_error"] = final_error
        bench_metrics["final_room_temp"] = history[-1]["room_temp"]
        sp_min = min(h["hp_setpoint"] for h in history)
        sp_max = max(h["hp_setpoint"] for h in history)
        bench_metrics["setpoint_min"] = sp_min
        bench_metrics["setpoint_max"] = sp_max
        assert final_error < 2.5, (
            f"{profile_name} seed={seed_factor}: final error {final_error:.1f}°C"
        )

        for h in history:
            assert 16 <= h["hp_setpoint"] <= 30


# ── Heat Wave ─────────────────────────────────────────────────────────────


class TestCoolingHeatWave:
    """Outdoor rises 10°C over 3 hours while room should stay cool."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0])
    def test_heat_wave(self, bench_metrics, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=24.0, outdoor=30.0)

        # Outdoor rises 30°C → 40°C over the first ~3.1h, then holds.
        # Original (15-min ticks): 0.8°C per tick = 3.2°C/h.
        def outdoor_schedule(minute):
            return min(40.0, 30.0 + minute * (3.2 / 60.0))

        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="cool",
                               outdoor_minute_schedule=outdoor_schedule)
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=24.0)

        # 90-min settle (was tick > 6 at 15-min cadence).
        post_settle = [h for h in history if h["minute"] > 90]
        post_settle_devs = [abs(h["room_temp"] - 24.0) for h in post_settle]
        bench_metrics["post_settle_max_abs_dev"] = (
            max(post_settle_devs) if post_settle_devs else 0.0
        )
        bench_metrics["post_settle_mean_abs_dev"] = (
            sum(post_settle_devs) / len(post_settle_devs) if post_settle_devs else 0.0
        )
        for h in post_settle:
            assert abs(h["room_temp"] - 24.0) < 3.0, (
                f"{profile_name} seed={seed_factor} tick {h['tick']}: "
                f"room={h['room_temp']:.1f}"
            )


# ── Cooling Steady State ─────────────────────────────────────────────────


class TestCoolingSteadyState:
    """Room at target in cooling mode, should stay stable."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0])
    def test_steady_state(self, bench_metrics, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=24.0, outdoor=32.0)

        # 12h run; last 4h should be stable (was n_ticks=48, last 16
        # ticks at 15-min cadence = 4h).
        history = run_scenario(ctrl, model, duration_minutes=12 * 60, mode="cool")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=24.0)

        late = [h for h in history if h["minute"] >= 8 * 60]
        late_temps = [h["room_temp"] for h in late]
        temp_range = max(late_temps) - min(late_temps)
        bench_metrics["late_temp_range"] = temp_range
        bench_metrics["late_temp_max"] = max(late_temps)
        bench_metrics["late_temp_min"] = min(late_temps)
        bench_metrics["late_temp_mean"] = sum(late_temps) / len(late_temps)
        assert temp_range < 2.0, (
            f"{profile_name} seed={seed_factor}: range {temp_range:.1f}°C"
        )


# ── Solar Rejection (cooling) ────────────────────────────────────────────


class TestCoolingSolarRejection:
    """Solar gain warms room during cooling mode. Controller must compensate."""

    # Zone-level solar sensitivity per archetype (was formerly baked into profiles)
    SOLAR_GAINS = {"drafty_bungalow": 0.5, "standard_residential": 0.4, "well_insulated": 0.15}

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_solar_rejection(self, bench_metrics, profile_name):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=24.0, outdoor=30.0,
                            solar_gain=self.SOLAR_GAINS[profile_name])

        # Solar starts after 60 min, ramps at 0.1/tick (15-min) = 0.4/h,
        # caps at 0.8.  Original: tick<4 = 0; (tick-4)*0.1.
        def solar_schedule(minute):
            if minute < 60.0:
                return 0.0
            return min(0.8, (minute - 60.0) * (0.4 / 60.0))

        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="cool",
                               solar_minute_schedule=solar_schedule)
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=1.0, desired=24.0)

        # 120-min settle (was tick > 8 at 15-min cadence).
        post_settle = [h for h in history if h["minute"] > 120]
        post_settle_devs = [abs(h["room_temp"] - 24.0) for h in post_settle]
        bench_metrics["post_settle_max_abs_dev"] = (
            max(post_settle_devs) if post_settle_devs else 0.0
        )
        bench_metrics["post_settle_mean_abs_dev"] = (
            sum(post_settle_devs) / len(post_settle_devs) if post_settle_devs else 0.0
        )
        for h in post_settle:
            assert abs(h["room_temp"] - 24.0) < 2.5, (
                f"{profile_name} tick {h['tick']}: room={h['room_temp']:.1f}"
            )
