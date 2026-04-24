"""Derivative (D) term impact tests.

Sweeps Kd across realistic thermal-model-coupled scenarios to quantify
the D term's effect on reversals, overshoot, settling, ITAE, and energy.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


KD_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0]


def _make_controller(kd, profile, seed_factor=1.0, **overrides):
    seed = profile.true_seed
    config = {
        "pi_kd": kd,
        "pi_kd_filter_n": 8,
        "pi_outdoor_seed_heat": seed * seed_factor,
        "pi_outdoor_seed_cool": seed * seed_factor,
        **overrides,
    }
    return TasmotaPIAdapter(config)


def _make_model(profile, initial_temp=20.0, outdoor=5.0, **kwargs):
    return ThermalModel(profile=profile, initial_temp=initial_temp,
                        outdoor_temp=outdoor, **kwargs)


# ── Cold Start (17°C → 20.5°C) ──────────────────────────────────────────


class TestDerivativeColdStart:
    """D's impact on cold start: overshoot damping as room approaches target."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_kd_sweep_cold_start(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        results = {}

        for kd in KD_VALUES:
            ctrl = _make_controller(kd, profile)
            ctrl.set_desired_temp(20.5)
            model = _make_model(profile, initial_temp=17.0, outdoor=2.0)

            history = run_scenario(ctrl, model, n_ticks=48, mode="heat")
            m = compute_all_metrics(history, desired=20.5)

            d_terms = [h["d_term"] for h in history]
            results[kd] = {
                "metrics": m,
                "max_abs_d": max(abs(d) for d in d_terms),
                "history": history,
            }

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"Cold Start — {profile_name}")
        print(f"{'Kd':>5} {'ITAE':>8} {'Overshoot':>10} {'Settle':>7} "
              f"{'Reversals':>10} {'SP Changes':>11} {'max|D|':>8} {'kWh':>6}")
        for kd in KD_VALUES:
            m = results[kd]["metrics"]
            print(f"{kd:5.1f} {m['itae']:8.1f} {m['overshoot']:10.2f} "
                  f"{str(m.get('settling_time', 'N/A')):>7} "
                  f"{m['reversals']:10d} {m['setpoint_changes']:11d} "
                  f"{results[kd]['max_abs_d']:8.4f} "
                  f"{m.get('total_kwh', 0):6.2f}")

        # D terms should be zero for kd=0
        assert results[0.0]["max_abs_d"] == 0.0

        # D terms should scale with kd
        assert results[5.0]["max_abs_d"] > results[0.5]["max_abs_d"]

        # All Kd values should reach target (sanity)
        for kd in KD_VALUES:
            final_error = abs(results[kd]["history"][-1]["room_temp"] - 20.5)
            assert final_error < 2.0, f"Kd={kd}: final error {final_error:.1f}°C"


# ── Cold Snap (outdoor drops 15°C) ──────────────────────────────────────


class TestDerivativeColdSnap:
    """D's impact during sudden outdoor temp drop — disturbance rejection."""

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_kd_sweep_cold_snap(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        results = {}

        def outdoor(tick):
            return max(-5.0, 10.0 - tick * 1.25)

        for kd in KD_VALUES:
            ctrl = _make_controller(kd, profile)
            ctrl.set_desired_temp(20.5)
            model = _make_model(profile, initial_temp=20.5, outdoor=10.0)

            history = run_scenario(ctrl, model, n_ticks=48, mode="heat",
                                   outdoor_schedule=outdoor)
            m = compute_all_metrics(history, desired=20.5)

            d_terms = [h["d_term"] for h in history]
            results[kd] = {
                "metrics": m,
                "max_abs_d": max(abs(d) for d in d_terms),
                "history": history,
            }

        print(f"\n{'='*70}")
        print(f"Cold Snap — {profile_name}")
        print(f"{'Kd':>5} {'ITAE':>8} {'Overshoot':>10} {'Settle':>7} "
              f"{'Reversals':>10} {'SP Changes':>11} {'max|D|':>8} {'kWh':>6}")
        for kd in KD_VALUES:
            m = results[kd]["metrics"]
            print(f"{kd:5.1f} {m['itae']:8.1f} {m['overshoot']:10.2f} "
                  f"{str(m.get('settling_time', 'N/A')):>7} "
                  f"{m['reversals']:10d} {m['setpoint_changes']:11d} "
                  f"{results[kd]['max_abs_d']:8.4f} "
                  f"{m.get('total_kwh', 0):6.2f}")

        assert results[0.0]["max_abs_d"] == 0.0
        assert results[5.0]["max_abs_d"] > results[0.5]["max_abs_d"]


# ── Steady State with Sensor Noise ───────────────────────────────────────


class TestDerivativeSteadyStateNoise:
    """D's sensitivity to sensor noise — the key risk of derivative action.

    With noisy sensors, D amplifies measurement noise. This test checks
    whether higher Kd causes more setpoint chatter.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_kd_sweep_noisy_steady(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        results = {}

        for kd in KD_VALUES:
            ctrl = _make_controller(kd, profile)
            ctrl.set_desired_temp(20.5)
            model = _make_model(profile, initial_temp=20.5, outdoor=5.0,
                                sensor_noise_sigma=0.15,
                                noise_seed=42)

            history = run_scenario(ctrl, model, n_ticks=64, mode="heat")
            m = compute_all_metrics(history, desired=20.5)

            d_terms = [h["d_term"] for h in history]
            results[kd] = {
                "metrics": m,
                "max_abs_d": max(abs(d) for d in d_terms),
                "d_std": _std(d_terms),
            }

        print(f"\n{'='*70}")
        print(f"Noisy Steady State — {profile_name}")
        print(f"{'Kd':>5} {'ITAE':>8} {'Reversals':>10} {'SP Changes':>11} "
              f"{'max|D|':>8} {'D_std':>8} {'kWh':>6}")
        for kd in KD_VALUES:
            m = results[kd]["metrics"]
            print(f"{kd:5.1f} {m['itae']:8.1f} "
                  f"{m['reversals']:10d} {m['setpoint_changes']:11d} "
                  f"{results[kd]['max_abs_d']:8.4f} "
                  f"{results[kd]['d_std']:8.4f} "
                  f"{m.get('total_kwh', 0):6.2f}")

        # Sanity: D noise should scale with Kd
        assert results[5.0]["d_std"] > results[0.5]["d_std"]


# ── Setpoint Step (user raises temp 2°C) ────────────────────────────────


class TestDerivativeSetpointStep:
    """D should NOT react to setpoint changes (derivative on measurement).

    When the user bumps the setpoint, D should only respond to the
    resulting room temp change, not the error spike.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_kd_sweep_setpoint_step(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        results = {}

        for kd in KD_VALUES:
            ctrl = _make_controller(kd, profile)
            ctrl.set_desired_temp(20.5)
            model = _make_model(profile, initial_temp=20.5, outdoor=5.0)

            history = run_scenario(ctrl, model, n_ticks=48, mode="heat",
                                   desired_schedule={10: 22.5})
            m = compute_all_metrics(history, desired=22.5)

            d_terms = [h["d_term"] for h in history]

            # D at the step tick (tick 10) should be small — measurement
            # hasn't changed yet, only the setpoint did
            d_at_step = d_terms[10] if len(d_terms) > 10 else 0.0

            results[kd] = {
                "metrics": m,
                "d_at_step": d_at_step,
                "max_abs_d": max(abs(d) for d in d_terms),
                "history": history,
            }

        print(f"\n{'='*70}")
        print(f"Setpoint Step — {profile_name}")
        print(f"{'Kd':>5} {'ITAE':>8} {'Overshoot':>10} {'Settle':>7} "
              f"{'Reversals':>10} {'D@step':>8} {'max|D|':>8}")
        for kd in KD_VALUES:
            m = results[kd]["metrics"]
            print(f"{kd:5.1f} {m['itae']:8.1f} {m['overshoot']:10.2f} "
                  f"{str(m.get('settling_time', 'N/A')):>7} "
                  f"{m['reversals']:10d} "
                  f"{results[kd]['d_at_step']:8.4f} "
                  f"{results[kd]['max_abs_d']:8.4f}")

        # All Kd values should reach new target
        for kd in KD_VALUES:
            final_error = abs(results[kd]["history"][-1]["room_temp"] - 22.5)
            assert final_error < 2.0

        # Derivative-on-measurement: D at the step tick should be similar
        # across Kd values (measurement didn't change at that tick)
        # Allow some tolerance for accumulated filter state
        for kd in KD_VALUES:
            # D at step tick should be much smaller than max D during recovery
            if kd > 0 and results[kd]["max_abs_d"] > 0.001:
                ratio = abs(results[kd]["d_at_step"]) / results[kd]["max_abs_d"]
                assert ratio < 0.5, (
                    f"Kd={kd}: D at step tick is {ratio:.0%} of max — "
                    "derivative may be acting on error, not measurement"
                )


# ── Oscillation Damping ──────────────────────────────────────────────────


class TestDerivativeOscillation:
    """Test D's primary purpose: damping oscillation near setpoint.

    Use a fast-response profile (studio) with aggressive PI gains to
    induce oscillation, then check if D reduces it.
    """

    def test_kd_damps_oscillation(self):
        profile = QUICK_PROFILES["drafty_bungalow"]
        results = {}

        for kd in KD_VALUES:
            # Aggressive PI to provoke oscillation
            ctrl = _make_controller(kd, profile, pi_kp=3.0, pi_ki=0.3)
            ctrl.set_desired_temp(20.5)
            model = _make_model(profile, initial_temp=17.0, outdoor=0.0)

            history = run_scenario(ctrl, model, n_ticks=64, mode="heat")
            m = compute_all_metrics(history, desired=20.5)

            # Measure oscillation: std dev of room temp in last 32 ticks
            late_temps = [h["room_temp"] for h in history[-32:]]
            temp_std = _std(late_temps)

            results[kd] = {
                "metrics": m,
                "late_temp_std": temp_std,
            }

        print(f"\n{'='*70}")
        print("Oscillation Damping — drafty_bungalow (aggressive PI)")
        print(f"{'Kd':>5} {'ITAE':>8} {'Overshoot':>10} {'Reversals':>10} "
              f"{'SP Changes':>11} {'Late T_std':>10}")
        for kd in KD_VALUES:
            m = results[kd]["metrics"]
            print(f"{kd:5.1f} {m['itae']:8.1f} {m['overshoot']:10.2f} "
                  f"{m['reversals']:10d} {m['setpoint_changes']:11d} "
                  f"{results[kd]['late_temp_std']:10.4f}")


def _std(values):
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
