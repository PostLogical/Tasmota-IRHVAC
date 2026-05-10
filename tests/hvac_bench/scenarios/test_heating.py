"""Core heating scenario tests.

These test basic control quality for heating mode across house profiles.
Every controller should pass these — they define minimum viable behavior.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES, HouseProfile2R2C as HouseProfile
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────

SEED_FACTORS = [0.0, 0.5, 1.0, 1.5]


def _make_controller(profile: HouseProfile, seed_factor=1.0, **overrides):
    seed = profile.true_seed
    config = {
        "pi_outdoor_seed_heat": seed * seed_factor,
        "pi_outdoor_seed_cool": seed * seed_factor,
        **overrides,
    }
    ctrl = TasmotaPIAdapter(config)
    return ctrl


def _make_model(profile, initial_temp=20.0, outdoor=5.0, **kwargs):
    """Build a 2R2C thermal model for the heating tests.

    The remaining knobs (sensor noise/quantization, head sensor offset,
    head calibration bounds, solar/stove inputs) are deliberately left
    at their idealized defaults — these tests check minimum-viable PI
    control quality on a clean thermal model.  Full-stack realism (noise,
    head offset, model inputs, real weather) lives in
    ``test_full_stack_learning.py``.

    The one realism knob we DO set: ``hp_lag_minutes=2.0`` (typical
    inverter compressor spool time).  Without this, the HP delivers
    commanded heat instantaneously, which becomes increasingly
    unrealistic as the bench cadence approaches the spool timescale.
    Tests can override via kwargs.
    """
    kwargs.setdefault("hp_lag_minutes", 2.0)
    return ThermalModel(profile=profile, initial_temp=initial_temp,
                        outdoor_temp=outdoor, **kwargs)


def _record_run(bench_metrics, history, *, profile_name, seed_factor,
                desired, deadband=0.5):
    """Record parametrization + control-quality rollup from a heating run.

    Captures every key in ``compute_all_metrics`` so phase-to-phase diffs
    surface drift in itae, overshoot, settling_time, reversals,
    setpoint_changes, integral_rms, comfort violations, and energy.
    Test-specific assertion values are recorded by the caller next to the
    assert (e.g. ``final_error``, ``late_temp_range``).
    """
    bench_metrics["profile_name"] = profile_name
    bench_metrics["seed_factor"] = seed_factor
    bench_metrics["n_ticks"] = len(history)
    rollup = compute_all_metrics(history, desired=desired, deadband=deadband)
    for k, v in rollup.items():
        bench_metrics[f"rollup_{k}"] = v


# ── Cold Start ────────────────────────────────────────────────────────────


class TestHeatingColdStart:
    """Room starts cold (17°C), needs to warm to 20.5°C."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", SEED_FACTORS)
    def test_cold_start(self, bench_metrics, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=17.0, outdoor=2.0)

        # 8 hours of recovery from a 3.5°C cold start (was n_ticks=32 at
        # 15-min cadence).  Duration is the contract; tick count is a
        # discretization detail and now scales with TICK_MINUTES_DEFAULT.
        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=20.5)

        # Must reach target eventually
        final_error = abs(history[-1]["room_temp"] - 20.5)
        bench_metrics["final_error"] = final_error
        bench_metrics["final_room_temp"] = history[-1]["room_temp"]
        assert final_error < 2.0, (
            f"{profile_name} seed={seed_factor}: final error {final_error:.1f}°C"
        )

        # Setpoint stays in bounds
        sp_min = min(h["hp_setpoint"] for h in history)
        sp_max = max(h["hp_setpoint"] for h in history)
        bench_metrics["setpoint_min"] = sp_min
        bench_metrics["setpoint_max"] = sp_max
        for h in history:
            assert 16 <= h["hp_setpoint"] <= 30


# ── Cold Snap ─────────────────────────────────────────────────────────────


class TestHeatingColdSnap:
    """Outdoor drops 15°C over 3 hours while room should stay at target."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", SEED_FACTORS)
    def test_cold_snap(self, bench_metrics, request, profile_name, seed_factor):
        if seed_factor == 1.5 and profile_name == "drafty_bungalow":
            request.node.add_marker(pytest.mark.xfail(
                strict=False,
                reason=(
                    "Post-#84 sim-coherent bench reveals real over-seed × "
                    "weak-insulation oscillation that the wall-clock leak "
                    "previously masked.  Test window (32 ticks) is too "
                    "short for batch WLS to correct the seed.  Eventual "
                    "fix: gain scheduling against insulation, OR rewrite "
                    "as a learning-window test with batch correction.  "
                    "See project_bench_audit_20260509.md."
                ),
            ))
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=10.0)

        # Outdoor drops from 10°C to -5°C over the first 3 hours, then
        # holds.  Original (15-min ticks): 1.25°C per tick = 5°C/h.
        def outdoor_schedule(minute):
            return max(-5.0, 10.0 - minute * (5.0 / 60.0))

        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="heat",
                               outdoor_minute_schedule=outdoor_schedule)
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=20.5)

        # Room should stay within tolerance after a 90-min settle window
        # (was tick > 6 at 15-min cadence; converted to minutes for
        # cadence independence).
        tol = 3.0 if seed_factor == 0.0 else 2.0
        post_settle = [h for h in history if h["minute"] > 90]
        post_settle_devs = [abs(h["room_temp"] - 20.5) for h in post_settle]
        bench_metrics["post_settle_max_abs_dev"] = (
            max(post_settle_devs) if post_settle_devs else 0.0
        )
        bench_metrics["post_settle_mean_abs_dev"] = (
            sum(post_settle_devs) / len(post_settle_devs) if post_settle_devs else 0.0
        )
        bench_metrics["tolerance_threshold"] = tol
        for h in post_settle:
            assert abs(h["room_temp"] - 20.5) < tol, (
                f"{profile_name} seed={seed_factor} tick {h['tick']}: "
                f"room={h['room_temp']:.1f}"
            )


# ── Setpoint Steps ────────────────────────────────────────────────────────


class TestHeatingSetpointUp:
    """User raises desired temp by 2°C at tick 10."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_setpoint_up(self, bench_metrics, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=5.0)

        # Setpoint step at 150 min = 2.5h (was tick=10 at 15-min cadence).
        # Total run 8h.
        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="heat",
                               desired_minute_schedule={150.0: 22.5})
        # Final desired is 22.5; rollup is computed against final desired
        # so post-step tracking error dominates the metrics.
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=22.5)

        # Should reach new target
        final_error = abs(history[-1]["room_temp"] - 22.5)
        bench_metrics["final_error"] = final_error
        bench_metrics["final_room_temp"] = history[-1]["room_temp"]
        # Step-response shape (post-step):
        post_step = [h for h in history if h["minute"] >= 150]
        if post_step:
            bench_metrics["post_step_max_room_temp"] = max(
                h["room_temp"] for h in post_step
            )
        assert final_error < 2.0, (
            f"{profile_name} seed={seed_factor}: final error {final_error:.1f}°C"
        )


class TestHeatingSetpointDown:
    """User lowers desired temp by 2°C at tick 10."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_setpoint_down(self, bench_metrics, request, profile_name, seed_factor):
        if seed_factor == 1.5 and profile_name == "drafty_bungalow":
            request.node.add_marker(pytest.mark.xfail(
                strict=False,
                reason=(
                    "Post-#84 sim-coherent bench reveals real over-seed × "
                    "weak-insulation oscillation that the wall-clock leak "
                    "previously masked.  Test window (32 ticks) is too "
                    "short for batch WLS to correct the seed.  Eventual "
                    "fix: gain scheduling against insulation, OR rewrite "
                    "as a learning-window test.  "
                    "See project_bench_audit_20260509.md."
                ),
            ))
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(22.5)
        model = _make_model(profile, initial_temp=22.5, outdoor=5.0)

        # Setpoint step down at 150 min = 2.5h (was tick=10 at 15-min).
        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="heat",
                               desired_minute_schedule={150.0: 20.5})
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=20.5)

        final_error = abs(history[-1]["room_temp"] - 20.5)
        bench_metrics["final_error"] = final_error
        bench_metrics["final_room_temp"] = history[-1]["room_temp"]
        post_step = [h for h in history if h["minute"] >= 150]
        if post_step:
            bench_metrics["post_step_min_room_temp"] = min(
                h["room_temp"] for h in post_step
            )
        assert final_error < 2.0


# ── Steady State ──────────────────────────────────────────────────────────


class TestHeatingSteadyState:
    """Room starts at target, should stay stable."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5])
    def test_steady_state(self, bench_metrics, request, profile_name, seed_factor):
        if seed_factor == 1.5 and profile_name in (
            "drafty_bungalow", "standard_residential",
        ):
            request.node.add_marker(pytest.mark.xfail(
                strict=False,
                reason=(
                    "Post-#84 sim-coherent bench reveals real over-seed × "
                    "weak-insulation oscillation that the wall-clock leak "
                    "previously masked.  Test window (48 ticks) is too "
                    "short for batch WLS to correct the seed.  Eventual "
                    "fix: gain scheduling against insulation, OR rewrite "
                    "as a learning-window test.  "
                    "See project_bench_audit_20260509.md."
                ),
            ))
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=5.0)

        # 12h run; the last 4h should be stable (was n_ticks=48, last
        # 16 ticks at 15-min cadence = 4h).
        history = run_scenario(ctrl, model, duration_minutes=12 * 60, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=20.5)

        late = [h for h in history if h["minute"] >= 8 * 60]
        late_temps = [h["room_temp"] for h in late]
        temp_range = max(late_temps) - min(late_temps)
        bench_metrics["late_temp_range"] = temp_range
        bench_metrics["late_temp_max"] = max(late_temps)
        bench_metrics["late_temp_min"] = min(late_temps)
        bench_metrics["late_temp_mean"] = sum(late_temps) / len(late_temps)
        assert temp_range < 2.0, (
            f"{profile_name} seed={seed_factor}: range {temp_range:.1f}°C in last 4h"
        )


# ── Ramp Disturbance ─────────────────────────────────────────────────────


class TestHeatingRampDisturbance:
    """Outdoor drops slowly at 1°C/hour (realistic weather)."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    @pytest.mark.parametrize("seed_factor", [0.5, 1.0])
    def test_ramp_disturbance(self, bench_metrics, profile_name, seed_factor):
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor=5.0)

        # Outdoor drops at 1°C/h continuously over 8h (was 0.25°C per
        # 15-min tick = 1°C/h).
        def outdoor_schedule(minute):
            return 5.0 - minute * (1.0 / 60.0)

        history = run_scenario(ctrl, model, duration_minutes=8 * 60, mode="heat",
                               outdoor_minute_schedule=outdoor_schedule)
        _record_run(bench_metrics, history, profile_name=profile_name,
                    seed_factor=seed_factor, desired=20.5)

        # Should track within tolerance after a 2h settle window
        # (was tick > 8 at 15-min cadence = 120 min).
        post_settle = [h for h in history if h["minute"] > 120]
        post_settle_devs = [abs(h["room_temp"] - 20.5) for h in post_settle]
        bench_metrics["ramp_max_abs_dev"] = (
            max(post_settle_devs) if post_settle_devs else 0.0
        )
        bench_metrics["ramp_mean_abs_dev"] = (
            sum(post_settle_devs) / len(post_settle_devs) if post_settle_devs else 0.0
        )
        for h in post_settle:
            assert abs(h["room_temp"] - 20.5) < 2.5, (
                f"{profile_name} seed={seed_factor} tick {h['tick']}: "
                f"room={h['room_temp']:.1f}, outdoor={h['outdoor']:.1f}"
            )
