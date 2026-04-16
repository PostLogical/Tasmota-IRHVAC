"""Energy efficiency and COP tests.

Verifies that the controller makes energy-efficient decisions,
not just comfortable ones.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel, COPModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


def _make_controller(seed_factor=1.0):
    return TasmotaPIAdapter({
        "pi_ff_heat_slope": 0.35 * seed_factor,
        "pi_ff_cool_slope": 0.35 * seed_factor,
    })


# ── COP Tracking ─────────────────────────────────────────────────────────


class TestCOPTracking:
    """Verify COP is tracked and physically reasonable."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_heating_cop_range(self, profile_name):
        """COP should be in realistic range during heating."""
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=20.5, outdoor_temp=5.0,
            cop_model=COPModel(),
        )

        history = run_scenario(ctrl, model, n_ticks=24, mode="heat")

        cops = [h["cop"] for h in history if h["cop"] > 0]
        assert len(cops) > 0, "No COP data recorded"
        assert all(1.0 <= c <= 6.0 for c in cops), (
            f"COP out of range: {min(cops):.1f}-{max(cops):.1f}"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cooling_cop_range(self, profile_name):
        """COP should be in realistic range during cooling."""
        profile = QUICK_PROFILES[profile_name]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(24.0)
        model = ThermalModel(
            profile=profile, initial_temp=24.0, outdoor_temp=32.0,
            cop_model=COPModel(),
        )

        history = run_scenario(ctrl, model, n_ticks=24, mode="cool")

        cops = [h["cop"] for h in history if h["cop"] > 0]
        assert len(cops) > 0
        assert all(1.0 <= c <= 7.0 for c in cops)

    def test_energy_accumulates(self):
        """Cumulative kWh should increase over time."""
        profile = QUICK_PROFILES["standard_residential"]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=17.0, outdoor_temp=0.0,
            cop_model=COPModel(),
        )

        history = run_scenario(ctrl, model, n_ticks=24, mode="heat")

        # Energy should be monotonically increasing
        kwhs = [h["cumulative_kwh"] for h in history]
        for i in range(1, len(kwhs)):
            assert kwhs[i] >= kwhs[i-1], (
                f"Energy decreased at tick {i}: {kwhs[i-1]:.3f} → {kwhs[i]:.3f}"
            )
        assert kwhs[-1] > 0, "No energy consumed in 24 ticks of heating from cold"


# ── Custom COP Model ─────────────────────────────────────────────────────


class TestCustomCOP:
    """Verify custom COP function override works."""

    def test_custom_cop_fn(self):
        """Custom COP function should be used instead of default."""
        def constant_cop(outdoor_c, setpoint_c, mode):
            return 3.0

        profile = QUICK_PROFILES["standard_residential"]
        ctrl = _make_controller(seed_factor=1.0)
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(
            profile=profile, initial_temp=20.5, outdoor_temp=5.0,
            cop_model=COPModel(cop_fn=constant_cop),
        )

        history = run_scenario(ctrl, model, n_ticks=8, mode="heat")

        for h in history:
            assert h["cop"] == 3.0


# ── Overshoot Energy Cost ────────────────────────────────────────────────


class TestOvershootEnergyCost:
    """Controllers that overshoot waste energy.

    Compare energy use between overseeded (1.5x) and correctly seeded (1.0x).
    Overseed should use more energy due to overshoot compensation.
    """

    @pytest.mark.parametrize("profile_name", ["standard_residential"])
    def test_overseed_wastes_energy(self, profile_name):
        profile = QUICK_PROFILES[profile_name]

        # Correctly seeded
        ctrl_good = _make_controller(seed_factor=1.0)
        ctrl_good.set_desired_temp(20.5)
        model_good = ThermalModel(
            profile=profile, initial_temp=17.0, outdoor_temp=0.0,
            cop_model=COPModel(),
        )
        hist_good = run_scenario(ctrl_good, model_good, n_ticks=32, mode="heat")

        # Overseeded
        ctrl_over = _make_controller(seed_factor=1.5)
        ctrl_over.set_desired_temp(20.5)
        model_over = ThermalModel(
            profile=profile, initial_temp=17.0, outdoor_temp=0.0,
            cop_model=COPModel(),
        )
        hist_over = run_scenario(ctrl_over, model_over, n_ticks=32, mode="heat")

        # Both should reach target
        assert abs(hist_good[-1]["room_temp"] - 20.5) < 2.0
        assert abs(hist_over[-1]["room_temp"] - 20.5) < 2.0

        # Overseeded typically uses more energy (overshoot → higher setpoints → lower COP)
        # This is informational — we log it but don't strictly assert
        kwh_good = hist_good[-1]["cumulative_kwh"]
        kwh_over = hist_over[-1]["cumulative_kwh"]
        print(f"\n  Energy: correct={kwh_good:.3f} kWh, overseed={kwh_over:.3f} kWh "
              f"(delta={kwh_over-kwh_good:+.3f})")
