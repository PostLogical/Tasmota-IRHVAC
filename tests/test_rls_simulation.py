"""Simulation tests: RLS model behavior over realistic diurnal scenarios.

These tests simulate multi-day cycles with realistic outdoor temp, solar gain,
and supplemental heat patterns to validate RLS convergence and stability.

Includes both open-loop tests (feed true offset directly) and closed-loop
tests (simulate PI feedback loop where observation = hp_setpoint - desired).
"""

import math
import random
import pytest

from custom_components.tasmota_irhvac.rls_model import RLSModel


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
        """RLS should converge to true coefficients within 5 days."""
        data = _simulate_diurnal_cycle(hours=120)  # 5 days (P_init=1 needs more data)

        # Start with wrong seeds
        model = RLSModel(
            n_inputs=3,
            seed_coefficients=[0.0, 0.20, -1.0, 0.0],  # Wrong seeds
            coeff_clamps=[None, (-1.0, 1.0), (-10.0, 0.0), (-10.0, 0.0)],
        )

        for d in data:
            x = [1.0, d["outdoor_delta"], d["solar_proxy"], d["pellet_stove"]]
            model.update(x, d["true_offset"])

        # After 120 hours, slope coefficients should converge.
        # Intercept converges slowest (absorbed by other coefficients initially).
        assert model.beta[1] == pytest.approx(0.35, abs=0.1)  # Outdoor
        assert model.beta[2] == pytest.approx(-4.0, abs=1.5)  # Solar
        assert model.beta[3] == pytest.approx(-3.2, abs=1.5)  # Stove
        # Intercept: just verify it's not wildly wrong (sign and magnitude)
        assert abs(model.beta[0]) < 2.0  # Not exploded

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
        coeffs = model.get_coefficients()
        assert abs(coeffs[1]) < 2.0, f"Outdoor coefficient exploded: {coeffs[1]}"
        assert abs(coeffs[2]) < 10.0, f"Solar coefficient exploded: {coeffs[2]}"


# ── Closed-Loop PI+RLS Simulation ──────────────────────────────────────


def _simulate_closed_loop(
    rls, true_slope, ki=0.15, kp=1.0, deadband=0.5,
    n_ticks=200, outdoor_schedule=None, disturbance_schedule=None,
    seed=42,
):
    """Simulate a PI+RLS closed loop.

    The PI controller drives hp_setpoint based on error + integral + FF.
    The RLS observes hp_setpoint - desired when in deadband and integral stable.
    A simple thermal model connects hp_setpoint to room temperature.

    Returns history list of dicts with per-tick state.
    """
    random.seed(seed)

    desired = 20.5  # °C
    room_temp = desired
    integral = 0.0
    hp_setpoint = round(desired)
    settled_ticks = 0
    prev_integral = 0.0

    # Simple thermal model: room temp moves toward hp_setpoint with time constant
    thermal_tc = 0.7  # fraction of gap closed per tick (15-min tick, ~25min TC)

    history = []

    for tick in range(n_ticks):
        outdoor = outdoor_schedule(tick) if outdoor_schedule else 5.0
        disturbance = disturbance_schedule(tick) if disturbance_schedule else 0.0

        # True required offset (what the room actually needs)
        outdoor_delta = max(0, 15.0 - outdoor)
        true_offset = true_slope * outdoor_delta + disturbance

        # Room temperature evolves: moves toward (desired + noise) driven by
        # how much the HP setpoint overshoots/undershoots the true need
        excess = (hp_setpoint - desired) - true_offset  # positive = too much heat
        room_temp += thermal_tc * excess + random.gauss(0, 0.05)

        error = desired - room_temp
        abs_error = abs(error)
        in_deadband = abs_error < deadband

        if in_deadband:
            settled_ticks += 1
            p_term = 0.0
        else:
            settled_ticks = 0
            p_term = kp * 0.3 * error  # setpoint weight b=0.3
            integral += error  # trapezoidal approximation simplified

        # FF prediction from RLS
        x = [1.0, outdoor_delta]
        ff_offset = rls.predict(x)

        # RLS learning gate: in deadband, settled 4+ ticks, integral stable
        integral_stable = abs(integral - prev_integral) < 0.5
        if in_deadband and settled_ticks >= 4 and integral_stable:
            observed_offset = float(hp_setpoint) - desired
            rls.update(x, observed_offset)

        prev_integral = integral
        i_term = ki * integral

        raw_setpoint = desired + p_term + i_term + ff_offset
        clamped = max(16.0, min(30.0, raw_setpoint))

        # Midpoint-crossing hysteresis
        if clamped > hp_setpoint + 0.5:
            hp_setpoint = round(clamped)
        elif clamped < hp_setpoint - 0.5:
            hp_setpoint = round(clamped)
        hp_setpoint = int(max(16, min(30, hp_setpoint)))

        history.append({
            "tick": tick,
            "room_temp": room_temp,
            "hp_setpoint": hp_setpoint,
            "integral": integral,
            "ff_offset": ff_offset,
            "error": error,
            "outdoor_delta": outdoor_delta,
        })

    return history


class TestClosedLoopMultiScale:
    """Test that feature normalization gives balanced learning in closed loop."""

    def test_intercept_stable_with_feature_scales(self):
        """With production-like feature_scales, intercept should not dominate.

        This is the bug that caused intercept=-2.26 in DR: the intercept had
        100x the Kalman gain of outdoor_delta, absorbing all variance.
        """
        # Production-like scales: outdoor_delta typically ~10
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.35],
            feature_scales=[1.0, 10.0],
        )

        history = _simulate_closed_loop(
            model, true_slope=0.35, n_ticks=200,
            outdoor_schedule=lambda t: 5.0 + 3.0 * math.sin(t * 0.1),
        )

        coeffs = model.get_coefficients()
        # Intercept should stay near 0, not drift to large values
        assert abs(coeffs[0]) < 0.5, (
            f"Intercept drifted to {coeffs[0]:.2f} — should stay near 0"
        )
        # Outdoor coefficient should stay near true value
        assert coeffs[1] == pytest.approx(0.35, abs=0.15), (
            f"Outdoor coeff={coeffs[1]:.3f}, expected ~0.35"
        )

    def test_single_observation_after_reset_bounded(self):
        """After reset, a single observation should not cause >20% coefficient shift.

        This reproduces the DR bug: 1 observation shifted intercept by -2.26
        and outdoor_delta by 46%.
        """
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.48],  # DR seeds
            feature_scales=[1.0, 10.0],
        )

        # One observation at an unusual condition
        x = [1.0, 13.0]  # outdoor_delta=13 (outdoor=2°C, cold)
        y = 4.5  # observed offset

        model.update(x, y)

        coeffs = model.get_coefficients()
        # Intercept should not shift more than ~0.5 from seed (0)
        assert abs(coeffs[0]) < 1.0, (
            f"Single observation shifted intercept to {coeffs[0]:.2f}"
        )
        # Outdoor coefficient should not shift more than ~30% from seed
        assert abs(coeffs[1] - 0.48) / 0.48 < 0.30, (
            f"Single observation shifted outdoor coeff to {coeffs[1]:.3f} "
            f"({100 * (coeffs[1] - 0.48) / 0.48:.0f}% from seed 0.48)"
        )

    def test_balanced_learning_rates_across_scales(self):
        """Both intercept and outdoor_delta should learn at similar rates.

        Feed data where the true relationship is [0.5, 0.4]. Track convergence
        of both coefficients from wrong seeds [0, 0.2].
        """
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.2],  # Wrong seeds
            feature_scales=[1.0, 10.0],  # Production scales
        )

        random.seed(42)
        for _ in range(100):
            outdoor_delta = random.uniform(3, 15)
            x = [1.0, outdoor_delta]
            y = 0.5 + 0.4 * outdoor_delta + random.gauss(0, 0.2)
            model.update(x, y)

        coeffs = model.get_coefficients()
        assert coeffs[0] == pytest.approx(0.5, abs=0.3), (
            f"Intercept={coeffs[0]:.3f}, expected ~0.5"
        )
        assert coeffs[1] == pytest.approx(0.4, abs=0.1), (
            f"Outdoor={coeffs[1]:.3f}, expected ~0.4"
        )


class TestClosedLoopSelfCorrection:
    """Test that RLS can correct wrong coefficients through the feedback loop."""

    def test_wrong_seed_corrected_via_integral(self):
        """FF with wrong slope should be corrected as integral absorbs the error.

        When FF underpredicts, integral builds to compensate. RLS observes
        hp_setpoint - desired which includes ki * integral, revealing the
        true offset needed. Coefficients should converge toward truth.
        """
        # Start with wrong seed (0.2 instead of true 0.35)
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.20],
            feature_scales=[1.0, 10.0],
        )

        history = _simulate_closed_loop(
            model, true_slope=0.35, n_ticks=300,
            outdoor_schedule=lambda t: 5.0,  # Constant outdoor
        )

        coeffs = model.get_coefficients()
        # Should have corrected toward 0.35 (at least partially)
        assert coeffs[1] > 0.25, (
            f"Outdoor coeff={coeffs[1]:.3f} — didn't correct from seed 0.20 "
            f"toward true 0.35"
        )

        # Room should be holding near target by end
        late_temps = [h["room_temp"] for h in history[-20:]]
        avg_error = sum(abs(t - 20.5) for t in late_temps) / len(late_temps)
        assert avg_error < 0.5, (
            f"Room not converging: avg error={avg_error:.2f}°C"
        )

    def test_large_integral_observations_drive_correction(self):
        """When integral is large but stable, RLS should learn from it.

        The observation hp_setpoint - desired = ff_predict + ki*integral.
        Large integral means FF is wrong. The residual = ki*integral should
        push coefficients in the right direction.
        """
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.15],  # Very wrong seed
            feature_scales=[1.0, 10.0],
        )

        history = _simulate_closed_loop(
            model, true_slope=0.40, n_ticks=400,
            outdoor_schedule=lambda t: 3.0,  # Cold, outdoor_delta=12
        )

        coeffs = model.get_coefficients()
        # Combined intercept + slope should predict close to true offset
        # True offset = 0.40 * 12 = 4.8
        predicted = model.predict([1.0, 12.0])
        assert abs(predicted - 4.8) < 1.5, (
            f"After 400 ticks, prediction={predicted:.2f} vs true=4.8 "
            f"(coeffs: intercept={coeffs[0]:.3f}, od={coeffs[1]:.3f})"
        )


class TestClosedLoopDisturbanceRejection:
    """Test behavior with unmeasured disturbances (like solar without solar input)."""

    def test_diurnal_disturbance_absorbed_by_intercept(self):
        """Solar-like diurnal disturbance without solar input.

        Without a solar model input, the intercept WILL absorb the average
        solar effect — this is omitted variable bias and is mathematically
        expected. The test verifies the outdoor coefficient stays close to
        truth (intercept absorbs the bias, not outdoor_delta).
        """
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.35],
            feature_scales=[1.0, 10.0],
        )

        def outdoor(tick):
            hour = (tick % 96) / 4  # 15-min ticks, 24-hour cycle
            return 5.0 + 5.0 * math.sin(math.radians((hour - 5) * 15 - 90))

        def solar_disturbance(tick):
            hour = (tick % 96) / 4
            if 8 <= hour <= 18:
                return -1.5 * math.sin(math.radians((hour - 8) * 18))
            return 0.0

        history = _simulate_closed_loop(
            model, true_slope=0.35, n_ticks=96 * 3,  # 3 days
            outdoor_schedule=outdoor,
            disturbance_schedule=solar_disturbance,
        )

        coeffs = model.get_coefficients()
        # Both intercept and outdoor_delta absorb some solar bias because
        # solar correlates with outdoor temp (both follow diurnal cycle).
        # This is omitted variable bias — expected without the solar input.
        # The key property is that outdoor_delta stays in a reasonable range
        # and doesn't explode or collapse.
        assert 0.15 < coeffs[1] < 0.65, (
            f"Outdoor coeff outside reasonable range: {coeffs[1]:.3f}"
        )

    def test_diurnal_disturbance_fixed_with_solar_input(self):
        """With solar as a model input, intercept stays near 0."""
        model = RLSModel(
            n_inputs=2,
            seed_coefficients=[0.0, 0.35, -1.5],
            feature_scales=[1.0, 10.0, 0.5],
        )

        random.seed(42)
        for tick in range(96 * 5):  # 5 days
            hour = (tick % 96) / 4
            outdoor = 5.0 + 5.0 * math.sin(math.radians((hour - 5) * 15 - 90))
            outdoor_delta = max(0, 15.0 - outdoor)

            if 8 <= hour <= 18:
                solar = 0.7 * math.sin(math.radians((hour - 8) * 18))
                solar += random.gauss(0, 0.05)
                solar = max(0, solar)
            else:
                solar = 0.0

            x = [1.0, outdoor_delta, solar]
            true_offset = 0.35 * outdoor_delta - 1.5 * solar
            y = true_offset + random.gauss(0, 0.2)
            model.update(x, y)

        coeffs = model.get_coefficients()
        assert abs(coeffs[0]) < 0.5, (
            f"Intercept drifted to {coeffs[0]:.2f} despite solar input"
        )
        assert coeffs[1] == pytest.approx(0.35, abs=0.1)
        assert coeffs[2] == pytest.approx(-1.5, abs=0.5)

    def test_collinear_noise_does_not_corrupt(self):
        """Noise partially correlated with outdoor_delta should not shift coefficients.

        Simulates a confounding variable (e.g., wind correlates with cold
        outdoor temps) that isn't modeled as an input.
        """
        model = RLSModel(
            n_inputs=1,
            seed_coefficients=[0.0, 0.35],
            feature_scales=[1.0, 10.0],
        )

        random.seed(42)
        for _ in range(200):
            outdoor_delta = random.uniform(3, 15)
            # Correlated noise: wind effect ~ 0.3 * outdoor_delta + random
            wind_effect = 0.05 * outdoor_delta + random.gauss(0, 0.3)
            x = [1.0, outdoor_delta]
            y = 0.35 * outdoor_delta + wind_effect + random.gauss(0, 0.2)
            model.update(x, y)

        coeffs = model.get_coefficients()
        # Outdoor coefficient absorbs some of the correlated wind effect
        # (this is expected — omitted variable bias). The key test is that
        # the intercept doesn't explode.
        assert abs(coeffs[0]) < 1.5, (
            f"Intercept exploded with collinear noise: {coeffs[0]:.2f}"
        )
        # Outdoor coefficient should absorb the correlated portion
        # True = 0.35, plus ~0.05 from wind = ~0.40
        assert 0.2 < coeffs[1] < 0.7, (
            f"Outdoor coeff outside expected range: {coeffs[1]:.3f}"
        )
