"""Energy efficiency and COP tests.

Verifies that the controller makes energy-efficient decisions,
not just comfortable ones.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel, COPModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


def _make_controller(profile, seed_factor=1.0):
    seed = profile.true_seed
    return TasmotaPIAdapter({
        "pi_outdoor_seed_heat": seed * seed_factor,
        "pi_outdoor_seed_cool": seed * seed_factor,
    })


def _make_model(profile, **kwargs):
    """2R2C model with hp_lag_minutes=2.0 default."""
    kwargs.setdefault("hp_lag_minutes", 2.0)
    if "cop_model" not in kwargs:
        kwargs["cop_model"] = COPModel()
    return ThermalModel(profile=profile, **kwargs)


def _record_run(bench_metrics, history, *, profile_name, scenario, desired):
    """Record control-quality rollup + COP/energy summary."""
    bench_metrics["profile_name"] = profile_name
    bench_metrics["scenario"] = scenario
    bench_metrics["n_ticks"] = len(history)
    rollup = compute_all_metrics(history, desired=desired)
    for k, v in rollup.items():
        bench_metrics[f"rollup_{k}"] = v
    cops = [h["cop"] for h in history if h["cop"] > 0]
    if cops:
        bench_metrics["cop_min"] = min(cops)
        bench_metrics["cop_max"] = max(cops)
        bench_metrics["cop_mean"] = sum(cops) / len(cops)
    bench_metrics["final_cumulative_kwh"] = history[-1]["cumulative_kwh"] if history else 0.0
    bench_metrics["final_room_temp"] = history[-1]["room_temp"] if history else None


# ── COP Tracking ─────────────────────────────────────────────────────────


class TestCOPTracking:
    """Verify COP is tracked and physically reasonable."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_heating_cop_range(self, bench_metrics, num_regression, profile_name):
        """COP should be in realistic range during heating."""
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor_temp=5.0)

        # 6h run (was n_ticks=24 at 15-min cadence).
        history = run_scenario(ctrl, model, duration_minutes=6 * 60, mode="heat")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    scenario="heating_cop_range", desired=20.5)
        check_bench_metrics(num_regression, bench_metrics)

        cops = [h["cop"] for h in history if h["cop"] > 0]
        assert len(cops) > 0, "No COP data recorded"
        assert all(1.0 <= c <= 6.0 for c in cops), (
            f"COP out of range: {min(cops):.1f}-{max(cops):.1f}"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cooling_cop_range(self, bench_metrics, num_regression, profile_name):
        """COP should be in realistic range during cooling."""
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(24.0)
        model = _make_model(profile, initial_temp=24.0, outdoor_temp=32.0)

        # 6h run.
        history = run_scenario(ctrl, model, duration_minutes=6 * 60, mode="cool")
        _record_run(bench_metrics, history, profile_name=profile_name,
                    scenario="cooling_cop_range", desired=24.0)
        check_bench_metrics(num_regression, bench_metrics)

        cops = [h["cop"] for h in history if h["cop"] > 0]
        assert len(cops) > 0
        assert all(1.0 <= c <= 7.0 for c in cops)

    def test_energy_accumulates(self, bench_metrics, num_regression):
        """Cumulative kWh should increase over time."""
        profile = QUICK_PROFILES["standard_residential"]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=17.0, outdoor_temp=0.0)

        # 6h run.
        history = run_scenario(ctrl, model, duration_minutes=6 * 60, mode="heat")
        _record_run(bench_metrics, history, profile_name="standard_residential",
                    scenario="energy_accumulates", desired=20.5)
        check_bench_metrics(num_regression, bench_metrics)

        # Energy should be monotonically increasing
        kwhs = [h["cumulative_kwh"] for h in history]
        for i in range(1, len(kwhs)):
            assert kwhs[i] >= kwhs[i-1], (
                f"Energy decreased at tick {i}: {kwhs[i-1]:.3f} → {kwhs[i]:.3f}"
            )
        assert kwhs[-1] > 0, "No energy consumed in 6h of heating from cold"


# ── Custom COP Model ─────────────────────────────────────────────────────


class TestCustomCOP:
    """Verify custom COP function override works."""

    def test_custom_cop_fn(self, bench_metrics, num_regression):
        """Custom COP function should be used instead of default."""
        def constant_cop(outdoor_c, setpoint_c, mode):
            return 3.0

        profile = QUICK_PROFILES["standard_residential"]
        ctrl = _make_controller(profile, seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = _make_model(profile, initial_temp=20.5, outdoor_temp=5.0,
                            cop_model=COPModel(cop_fn=constant_cop))

        # 2h run (was n_ticks=8 at 15-min cadence).
        history = run_scenario(ctrl, model, duration_minutes=2 * 60, mode="heat")
        _record_run(bench_metrics, history, profile_name="standard_residential",
                    scenario="custom_cop_fn", desired=20.5)
        check_bench_metrics(num_regression, bench_metrics)

        for h in history:
            assert h["cop"] == 3.0


# ── Overshoot Energy Cost ────────────────────────────────────────────────


class TestOvershootEnergyCost:
    """Controllers that overshoot waste energy.

    Compare energy use between overseeded (1.5x) and correctly seeded (1.0x).
    Overseed should use more energy due to overshoot compensation.
    """

    @pytest.mark.parametrize("profile_name", ["standard_residential"])
    def test_overseed_wastes_energy(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]

        # Correctly seeded
        ctrl_good = _make_controller(profile, seed_factor=1.0)
        ctrl_good.set_desired_temp(20.5)
        model_good = _make_model(profile, initial_temp=17.0, outdoor_temp=0.0)
        # 8h run (was n_ticks=32 at 15-min cadence).
        hist_good = run_scenario(ctrl_good, model_good, duration_minutes=8 * 60,
                                 mode="heat")

        # Overseeded
        ctrl_over = _make_controller(profile, seed_factor=1.5)
        ctrl_over.set_desired_temp(20.5)
        model_over = _make_model(profile, initial_temp=17.0, outdoor_temp=0.0)
        hist_over = run_scenario(ctrl_over, model_over, duration_minutes=8 * 60,
                                 mode="heat")

        # Both should reach target
        assert abs(hist_good[-1]["room_temp"] - 20.5) < 2.0
        assert abs(hist_over[-1]["room_temp"] - 20.5) < 2.0

        # Overseeded typically uses more energy (overshoot → higher setpoints → lower COP)
        # This is informational — we log it but don't strictly assert
        kwh_good = hist_good[-1]["cumulative_kwh"]
        kwh_over = hist_over[-1]["cumulative_kwh"]
        bench_metrics["profile_name"] = profile_name
        bench_metrics["scenario"] = "overseed_wastes_energy"
        bench_metrics["kwh_correct_seed"] = kwh_good
        bench_metrics["kwh_over_seed"] = kwh_over
        bench_metrics["kwh_delta_overseed_vs_correct"] = kwh_over - kwh_good
        bench_metrics["final_room_correct_seed"] = hist_good[-1]["room_temp"]
        bench_metrics["final_room_over_seed"] = hist_over[-1]["room_temp"]
        check_bench_metrics(num_regression, bench_metrics)
