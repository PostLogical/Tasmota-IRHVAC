"""Simulation tests: RLS model behavior over realistic diurnal scenarios.

These tests simulate multi-day cycles with realistic outdoor temp, solar gain,
and supplemental heat patterns to validate RLS convergence and stability.
"""

import math
import random
import pytest

from custom_components.tasmota_irhvac.pi_controller import RLSModel


def _simulate_diurnal_cycle(hours=72, seed=42):
    """Generate 72 hours of realistic input data.

    Returns list of dicts with: hour, outdoor_temp, solar_proxy, pellet_stove,
    and true_offset (what the room actually needed).
    """
    random.seed(seed)
    data = []

    # True coefficients (from regression)
    TRUE_INTERCEPT = 0.5
    TRUE_OUTDOOR = 0.35
    TRUE_SOLAR = -4.0
    TRUE_STOVE = -3.2

    for h in range(hours):
        hour_of_day = h % 24
        day = h // 24

        # Outdoor temp: sinusoidal diurnal cycle, colder at night
        # Base: 5°C, amplitude: 8°C, min at 5am, max at 3pm
        outdoor = 5.0 + 8.0 * math.sin(math.radians((hour_of_day - 5) * 15 - 90))

        # Solar proxy: 0 at night, peaks at noon
        if 7 <= hour_of_day <= 19:
            sun_elevation = 45.0 * math.sin(math.radians((hour_of_day - 7) * 15))
            cloud = random.uniform(0, 0.5)  # Mostly clear
            solar = max(0, math.sin(math.radians(sun_elevation))) * (1 - cloud)
        else:
            solar = 0.0

        # Pellet stove: on from 6pm to 10pm
        stove = 1.0 if 18 <= hour_of_day <= 22 else 0.0

        # Outdoor delta (reference = 15°C)
        outdoor_delta = max(0, 15.0 - outdoor)

        # True offset with noise
        true_offset = (
            TRUE_INTERCEPT
            + TRUE_OUTDOOR * outdoor_delta
            + TRUE_SOLAR * solar
            + TRUE_STOVE * stove
            + random.gauss(0, 0.3)  # Measurement/model noise
        )

        data.append({
            "hour": h,
            "hour_of_day": hour_of_day,
            "outdoor_temp": outdoor,
            "outdoor_delta": outdoor_delta,
            "solar_proxy": solar,
            "pellet_stove": stove,
            "true_offset": true_offset,
        })

    return data


class TestRLSDiurnalConvergence:
    """Test RLS convergence over a realistic 3-day diurnal cycle."""

    def test_converges_to_true_coefficients(self):
        """RLS should converge to true coefficients within 3 days."""
        data = _simulate_diurnal_cycle(hours=72)

        # Start with wrong seeds
        model = RLSModel(
            n_inputs=3,
            seed_coefficients=[0.0, 0.20, -1.0, 0.0],  # Wrong seeds
            coeff_clamps=[None, (-1.0, 1.0), (-10.0, 0.0), (-10.0, 0.0)],
        )

        for d in data:
            x = [1.0, d["outdoor_delta"], d["solar_proxy"], d["pellet_stove"]]
            model.update(x, d["true_offset"])

        # After 72 hours (72 observations at hourly), coefficients should be close
        assert model.beta[0] == pytest.approx(0.5, abs=0.5)   # Intercept
        assert model.beta[1] == pytest.approx(0.35, abs=0.1)  # Outdoor
        assert model.beta[2] == pytest.approx(-4.0, abs=1.0)  # Solar
        assert model.beta[3] == pytest.approx(-3.2, abs=1.0)  # Stove

    def test_prediction_error_decreases(self):
        """Prediction error should decrease over time as model learns."""
        data = _simulate_diurnal_cycle(hours=72)

        model = RLSModel(
            n_inputs=3,
            seed_coefficients=[0.0, 0.20, -1.0, 0.0],  # Wrong seeds
        )

        # Track prediction errors in first vs last day
        first_day_errors = []
        last_day_errors = []

        for d in data:
            x = [1.0, d["outdoor_delta"], d["solar_proxy"], d["pellet_stove"]]
            prediction = model.predict(x)
            error = abs(d["true_offset"] - prediction)

            if d["hour"] < 24:
                first_day_errors.append(error)
            elif d["hour"] >= 48:
                last_day_errors.append(error)

            model.update(x, d["true_offset"])

        avg_first = sum(first_day_errors) / len(first_day_errors)
        avg_last = sum(last_day_errors) / len(last_day_errors)

        assert avg_last < avg_first, (
            f"Error should decrease: first day avg={avg_first:.2f}, last day avg={avg_last:.2f}"
        )

    def test_solar_coefficient_not_corrupted_by_night(self):
        """Night observations (solar=0) should not corrupt the solar coefficient."""
        data = _simulate_diurnal_cycle(hours=72)

        model = RLSModel(
            n_inputs=3,
            seed_coefficients=[0.0, 0.35, -4.0, -3.2],  # Correct seeds
        )

        for d in data:
            x = [1.0, d["outdoor_delta"], d["solar_proxy"], d["pellet_stove"]]
            model.update(x, d["true_offset"])

        # Solar coefficient should stay near -4.0, not drift toward 0
        assert model.beta[2] == pytest.approx(-4.0, abs=1.5), (
            f"Solar coefficient drifted to {model.beta[2]:.2f}, expected near -4.0"
        )

    def test_outdoor_coefficient_isolated_from_solar(self):
        """Outdoor temp coefficient should not absorb solar's effect."""
        data = _simulate_diurnal_cycle(hours=72)

        model = RLSModel(
            n_inputs=3,
            seed_coefficients=[0.0, 0.20, -1.0, 0.0],  # Wrong seeds
        )

        for d in data:
            x = [1.0, d["outdoor_delta"], d["solar_proxy"], d["pellet_stove"]]
            model.update(x, d["true_offset"])

        # Outdoor coefficient should converge to 0.35, NOT be biased by solar
        assert model.beta[1] == pytest.approx(0.35, abs=0.1), (
            f"Outdoor coefficient is {model.beta[1]:.3f}, expected ~0.35. "
            f"Solar may be leaking into outdoor."
        )


class TestRLSSolarCorruption:
    """Test that RLS handles the specific scenario that corrupted buckets."""

    def test_sunny_afternoon_does_not_corrupt_outdoor_coefficient(self):
        """The scenario that killed buckets: sunny afternoon at 6°C.

        Buckets wrote a low offset for the 6°C bin because solar was helping.
        RLS should attribute the low offset to solar, not outdoor temp.
        """
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[0.0, 0.35, -4.0],  # Outdoor + solar
        )

        # 20 night observations at 6°C (no solar, high offset needed)
        for _ in range(20):
            outdoor_delta = 15.0 - 6.0  # = 9
            x = [1.0, outdoor_delta, 0.0]  # No solar
            y = 0.35 * 9.0 + random.gauss(0, 0.2)  # True need: ~3.15
            model.update(x, y)

        # 5 sunny afternoon observations at 6°C (solar helps, low offset)
        for _ in range(5):
            outdoor_delta = 15.0 - 6.0
            solar = 0.7  # Strong sun
            x = [1.0, outdoor_delta, solar]
            y = 0.35 * 9.0 + (-4.0) * 0.7 + random.gauss(0, 0.2)  # ~0.35
            model.update(x, y)

        # Outdoor coefficient should still be near 0.35 (not corrupted)
        assert model.beta[1] == pytest.approx(0.35, abs=0.1), (
            f"Outdoor coefficient corrupted to {model.beta[1]:.3f} by solar observations"
        )
        # Solar coefficient should be near -4.0
        assert model.beta[2] == pytest.approx(-4.0, abs=1.5), (
            f"Solar coefficient is {model.beta[2]:.3f}, expected near -4.0"
        )


class TestRLSCoefficientStability:
    """Test coefficient stability over long periods."""

    def test_coefficients_stable_when_model_is_correct(self):
        """Once converged, coefficients should not drift significantly."""
        data = _simulate_diurnal_cycle(hours=168)  # 1 week

        model = RLSModel(
            n_inputs=3,
            seed_coefficients=[0.5, 0.35, -4.0, -3.2],  # Correct seeds
        )

        # Train for 5 days
        for d in data[:120]:
            x = [1.0, d["outdoor_delta"], d["solar_proxy"], d["pellet_stove"]]
            model.update(x, d["true_offset"])

        # Record coefficients at day 5
        beta_day5 = list(model.beta)

        # Continue for 2 more days
        for d in data[120:]:
            x = [1.0, d["outdoor_delta"], d["solar_proxy"], d["pellet_stove"]]
            model.update(x, d["true_offset"])

        # Coefficients should not have drifted much
        for i in range(4):
            assert abs(model.beta[i] - beta_day5[i]) < 0.3, (
                f"Coefficient {i} drifted from {beta_day5[i]:.3f} to {model.beta[i]:.3f}"
            )

    def test_ridge_prevents_explosion_with_correlated_inputs(self):
        """Correlated inputs (outdoor + solar) should not cause coefficient explosion."""
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[0.0, 0.3, -2.0],
            delta=0.001,
        )

        random.seed(42)
        for _ in range(500):
            # Correlated: when outdoor is warm (low delta), solar is high
            outdoor_delta = random.uniform(0, 15)
            solar = max(0, 0.8 - outdoor_delta / 20 + random.gauss(0, 0.1))
            x = [1.0, outdoor_delta, solar]
            y = 0.3 * outdoor_delta - 2.0 * solar + random.gauss(0, 0.5)
            model.update(x, y)

        # Coefficients should be reasonable, not exploded
        assert abs(model.beta[1]) < 2.0, f"Outdoor coefficient exploded: {model.beta[1]}"
        assert abs(model.beta[2]) < 10.0, f"Solar coefficient exploded: {model.beta[2]}"
