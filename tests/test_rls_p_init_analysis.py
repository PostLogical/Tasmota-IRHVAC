"""Analysis tests: explore P_INIT and MIN_RLS_OBS parameter space.

These tests use the real RLSModel to simulate the Living Room failure
and explore parameter ranges. Results are printed for human review —
assertions validate only the known-bad current configuration and
basic sanity bounds.
"""

import math
import random

import pytest

from custom_components.tasmota_irhvac.pi.rls_model import RLSModel


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_rls(seed_slope: float, p_init: float, n_inputs: int = 1,
              feature_scales: list[float] | None = None) -> RLSModel:
    """Create an RLS model with given seed slope and P_INIT."""
    seeds = [0.0, seed_slope]  # intercept=0, outdoor_delta=seed_slope
    for _ in range(n_inputs - 1):
        seeds.append(0.0)
    scales = feature_scales or [1.0, 10.0] + [0.5] * (n_inputs - 1)
    clamps = [(-0.5, 0.5), (0.0, 2.0)] + [None] * (n_inputs - 1)
    return RLSModel(
        n_inputs=n_inputs,
        seed_coefficients=seeds,
        coeff_clamps=clamps,
        feature_scales=scales,
        p_init=p_init,
    )


def _blended_predict(rls: RLSModel, seeds: list[float], x: list[float],
                     min_rls_obs: int) -> float:
    """Reproduce the pi_controller blend between seed and RLS prediction."""
    seed_offset = sum(s * xi for s, xi in zip(seeds, x))
    rls_offset = rls.predict(x)
    alpha = min(rls.observation_count / min_rls_obs, 1.0)
    return (1.0 - alpha) * seed_offset + alpha * rls_offset


def _simulate_closed_loop_with_blend(
    rls: RLSModel,
    seeds: list[float],
    true_slope: float,
    min_rls_obs: int,
    p_init: float,
    ki: float = 0.15,
    kp: float = 1.0,
    deadband: float = 0.5,
    n_ticks: int = 200,
    outdoor_schedule=None,
    disturbance_schedule=None,
    seed_rng: int = 42,
) -> list[dict]:
    """Closed-loop PI+RLS simulation with seed-to-RLS blend.

    Like test_rls_simulation._simulate_closed_loop but includes the
    MIN_RLS_OBS blend that the production code uses.
    """
    random.seed(seed_rng)

    desired = 20.5  # °C
    room_temp = desired
    integral = 0.0
    hp_setpoint = round(desired)
    settled_ticks = 0
    prev_integral = 0.0
    thermal_tc = 0.7

    history: list[dict] = []

    for tick in range(n_ticks):
        outdoor = outdoor_schedule(tick) if outdoor_schedule else 5.0
        disturbance = disturbance_schedule(tick) if disturbance_schedule else 0.0

        outdoor_delta = max(0, 15.0 - outdoor)
        true_offset = true_slope * outdoor_delta + disturbance

        excess = (hp_setpoint - desired) - true_offset
        room_temp += thermal_tc * excess + random.gauss(0, 0.05)

        error = desired - room_temp
        in_deadband = abs(error) < deadband

        if in_deadband:
            settled_ticks += 1
            p_term = 0.0
        else:
            settled_ticks = 0
            p_term = kp * 0.3 * error
            integral += error

        x = [1.0, outdoor_delta]
        ff_offset = _blended_predict(rls, seeds, x, min_rls_obs)

        integral_stable = abs(integral - prev_integral) < 0.5
        if in_deadband and settled_ticks >= 4 and integral_stable:
            observed_offset = float(hp_setpoint) - desired
            rls.update(x, observed_offset)

        prev_integral = integral
        i_term = ki * integral

        raw_setpoint = desired + p_term + i_term + ff_offset
        clamped = max(16.0, min(30.0, raw_setpoint))

        if clamped > hp_setpoint + 0.5:
            hp_setpoint = round(clamped)
        elif clamped < hp_setpoint - 0.5:
            hp_setpoint = round(clamped)
        hp_setpoint = int(max(16, min(30, hp_setpoint)))

        slope_phys = rls.beta[1] / rls.feature_scales[1]

        history.append({
            "tick": tick,
            "room_temp": room_temp,
            "hp_setpoint": hp_setpoint,
            "integral": integral,
            "ff_offset": ff_offset,
            "error": error,
            "outdoor_delta": outdoor_delta,
            "slope": slope_phys,
            "obs_count": rls.observation_count,
            "alpha": min(rls.observation_count / min_rls_obs, 1.0),
        })

    return history


# ── Scenario A: Reproduce LR Failure ────────────────────────────────────


class TestScenarioA:
    """Reproduce the Living Room runaway: P_INIT=10, seed=0.35, 21 obs.

    The real failure mechanism: the PI integral was large during early
    operation (room cold, integral accumulating). RLS observed
    hp_setpoint - desired which included the integral contribution,
    attributing the high setpoint to outdoor_delta slope rather than
    the transient integral. This is systematic upward bias, not noise.
    """

    def test_lr_failure_seed_truth_mismatch(self):
        """With P_INIT=10 and seed below true slope, slope overshoots to clamp.

        This reproduces the actual LR failure: the seed (0.35) was too low
        for the real conditions. With P_INIT=10, the RLS overcorrects from
        noisy observations and overshoots far past the true slope.
        Scenario B confirms this pattern across the full parameter grid.
        """
        rls = _make_rls(seed_slope=0.35, p_init=10.0)
        rng = random.Random(42)

        # Varying outdoor conditions (realistic diurnal variation)
        true_slope = 0.6  # Seed 0.35 is too low
        for i in range(21):
            outdoor_delta = 8.0 + 4.0 * math.sin(i * 0.3) + rng.gauss(0, 1.0)
            outdoor_delta = max(0, outdoor_delta)
            x = [1.0, outdoor_delta]
            observed = true_slope * outdoor_delta + rng.gauss(0, 1.5)
            rls.update(x, observed)

        slope = rls.beta[1] / rls.feature_scales[1]
        intercept = rls.beta[0] / rls.feature_scales[0]
        print(f"\nScenario A: P_INIT=10, seed=0.35, true=0.6, varying outdoor")
        print(f"  Final slope: {slope:.4f} (seed: 0.35, clamp: 2.0)")
        print(f"  Final intercept: {intercept:.4f}")
        print(f"  Drift from seed: {abs(slope - 0.35) / 0.35 * 100:.0f}%")
        print(f"  Overshoot past true: {slope - 0.6:.4f}")

        # With P_INIT=10, slope should overshoot significantly past 0.6
        assert slope > 1.0, (
            f"Expected slope overshoot > 1.0, got {slope:.4f}"
        )

    def test_lr_failure_sweep_p_init(self):
        """Show how different P_INIT values handle seed-truth mismatch."""
        print("\nScenario A (sweep): Seed=0.35, true=0.6, varying outdoor")
        print(f"{'P_INIT':>8} {'Slope@21':>9} {'Intercept':>10} "
              f"{'Overshoot':>10} {'Drift%':>7}")

        for p_init in [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05]:
            rls = _make_rls(seed_slope=0.35, p_init=p_init)
            rng = random.Random(42)

            true_slope = 0.6
            for i in range(21):
                outdoor_delta = 8.0 + 4.0 * math.sin(i * 0.3) + rng.gauss(0, 1.0)
                outdoor_delta = max(0, outdoor_delta)
                x = [1.0, outdoor_delta]
                observed = true_slope * outdoor_delta + rng.gauss(0, 1.5)
                rls.update(x, observed)

            slope = rls.beta[1] / rls.feature_scales[1]
            intercept = rls.beta[0] / rls.feature_scales[0]
            drift = abs(slope - 0.35) / 0.35 * 100
            overshoot = slope - 0.6
            print(f"{p_init:8.2f} {slope:9.4f} {intercept:10.4f} "
                  f"{overshoot:10.4f} {drift:6.0f}%")


# ── Scenario B: P_INIT Sweep with Seed-Truth Mismatch ──────────────────


class TestScenarioB:
    """Sweep P_INIT values with various true slopes against seed=0.35."""

    P_INIT_VALUES = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05]
    TRUE_SLOPES = [0.2, 0.35, 0.5, 0.8]

    def test_p_init_sweep(self):
        """Print coefficient trajectories across P_INIT × true_slope grid."""
        print("\nScenario B: P_INIT sweep (seed=0.35)")
        print(f"{'P_INIT':>8} {'True':>6} {'@21obs':>8} {'@50obs':>8} "
              f"{'@100obs':>8} {'drift21%':>9}")

        for p_init in self.P_INIT_VALUES:
            for true_slope in self.TRUE_SLOPES:
                rls = _make_rls(seed_slope=0.35, p_init=p_init)
                rng = random.Random(42)

                outdoor_delta = 10.0
                x = [1.0, outdoor_delta]
                slopes_at = {}

                for i in range(100):
                    observed = true_slope * outdoor_delta + rng.gauss(0, 1.5)
                    rls.update(x, observed)
                    obs_n = i + 1
                    if obs_n in (21, 50, 100):
                        slopes_at[obs_n] = rls.beta[1] / rls.feature_scales[1]

                drift_21 = abs(slopes_at[21] - 0.35) / 0.35 * 100
                print(f"{p_init:8.2f} {true_slope:6.2f} {slopes_at[21]:8.4f} "
                      f"{slopes_at[50]:8.4f} {slopes_at[100]:8.4f} "
                      f"{drift_21:8.0f}%")

    def test_p_init_prevents_runaway_at_low_values(self):
        """With P_INIT <= 0.5, slope shouldn't hit the clamp from noise alone."""
        for p_init in [0.5, 0.1, 0.05]:
            rls = _make_rls(seed_slope=0.35, p_init=p_init)
            rng = random.Random(42)
            outdoor_delta = 10.0
            x = [1.0, outdoor_delta]

            # True slope matches seed — any drift is pure noise
            for i in range(21):
                observed = 0.35 * outdoor_delta + rng.gauss(0, 2.0)
                rls.update(x, observed)

            slope = rls.beta[1] / rls.feature_scales[1]
            assert slope < 1.5, (
                f"P_INIT={p_init}: slope={slope:.4f} hit near-clamp from noise"
            )

    def test_wrong_seed_eventually_corrects(self):
        """Even with conservative P_INIT, slope should approach truth by 100 obs."""
        for p_init in [0.5, 0.1, 0.05]:
            rls = _make_rls(seed_slope=0.35, p_init=p_init)
            rng = random.Random(42)
            outdoor_delta = 10.0
            x = [1.0, outdoor_delta]

            true_slope = 0.8  # Seed is significantly wrong
            for i in range(100):
                observed = true_slope * outdoor_delta + rng.gauss(0, 1.0)
                rls.update(x, observed)

            slope = rls.beta[1] / rls.feature_scales[1]
            # Should be moving toward 0.8, even if not there yet
            assert slope > 0.5, (
                f"P_INIT={p_init}: slope={slope:.4f} didn't move toward "
                f"true={true_slope} after 100 obs"
            )


# ── Scenario C: Zero-Seed Comfort During Learning ──────────────────────


class TestScenarioC:
    """With seed=0, does PI maintain comfort while RLS learns?"""

    P_INIT_VALUES = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]

    def test_zero_seed_comfort(self):
        """Measure room temp error during learning, not convergence speed."""
        print("\nScenario C: Zero-seed comfort (true_slope=0.35)")
        print(f"{'P_INIT':>8} {'MinObs':>7} {'MaxErr°C':>9} {'AvgErr°C':>9} "
              f"{'Slope@end':>10} {'MaxHP':>6}")

        true_slope = 0.35
        for p_init in self.P_INIT_VALUES:
            for min_obs in [10, 50]:
                rls = _make_rls(seed_slope=0.0, p_init=p_init)
                seeds = [0.0, 0.0]

                history = _simulate_closed_loop_with_blend(
                    rls, seeds, true_slope, min_obs, p_init,
                    n_ticks=300,
                )

                errors = [abs(h["error"]) for h in history]
                max_err = max(errors)
                avg_err = sum(errors) / len(errors)
                max_hp = max(h["hp_setpoint"] for h in history)
                final_slope = history[-1]["slope"]

                print(f"{p_init:8.2f} {min_obs:7d} {max_err:9.3f} "
                      f"{avg_err:9.3f} {final_slope:10.4f} {max_hp:6d}")

    def test_zero_seed_pi_maintains_comfort(self):
        """With conservative P_INIT, room error should stay manageable."""
        for p_init in [0.5, 0.1]:
            rls = _make_rls(seed_slope=0.0, p_init=p_init)
            seeds = [0.0, 0.0]

            history = _simulate_closed_loop_with_blend(
                rls, seeds, true_slope=0.35, min_rls_obs=50, p_init=p_init,
                n_ticks=300,
            )

            # After initial transient (first 20 ticks), error should be bounded
            steady_errors = [abs(h["error"]) for h in history[20:]]
            max_steady_err = max(steady_errors)
            # PI should keep room within ~2°C even with zero FF
            assert max_steady_err < 3.0, (
                f"P_INIT={p_init}: max steady error {max_steady_err:.2f}°C "
                f"— PI not maintaining comfort"
            )


# ── Scenario D: Cold Snap Extrapolation ─────────────────────────────────


class TestScenarioD:
    """Train in mild weather, predict in cold snap."""

    P_INIT_VALUES = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]

    def test_cold_snap_extrapolation(self):
        """Train at outdoor_delta=5, predict at outdoor_delta=30."""
        print("\nScenario D: Cold snap extrapolation")
        print(f"{'P_INIT':>8} {'Slope@50':>9} {'FF@delta5':>10} "
              f"{'FF@delta30':>11} {'TrueFF@30':>10} {'Error@30':>9}")

        true_slope = 0.35
        for p_init in self.P_INIT_VALUES:
            rls = _make_rls(seed_slope=0.35, p_init=p_init)
            rng = random.Random(42)

            # Train: 50 obs at mild weather (outdoor_delta=5)
            x_mild = [1.0, 5.0]
            for _ in range(50):
                observed = true_slope * 5.0 + rng.gauss(0, 1.0)
                rls.update(x_mild, observed)

            slope_50 = rls.beta[1] / rls.feature_scales[1]

            # Predict at cold snap (outdoor_delta=30)
            x_cold = [1.0, 30.0]
            ff_cold = rls.predict(x_cold)
            true_ff_cold = true_slope * 30.0  # 10.5
            ff_mild = rls.predict(x_mild)
            error_cold = ff_cold - true_ff_cold

            print(f"{p_init:8.2f} {slope_50:9.4f} {ff_mild:10.3f} "
                  f"{ff_cold:11.3f} {true_ff_cold:10.3f} {error_cold:9.3f}")

    def test_blend_protects_extrapolation(self):
        """MIN_RLS_OBS blend should anchor prediction during extrapolation."""
        print("\nScenario D (blend): MIN_RLS_OBS protects cold snap")
        print(f"{'P_INIT':>8} {'MinObs':>7} {'Obs':>5} {'Alpha':>6} "
              f"{'BlendFF':>8} {'RawFF':>8} {'SeedFF':>8} {'TrueFF':>8}")

        true_slope = 0.35
        seeds = [0.0, 0.35]

        for p_init in [10.0, 1.0, 0.1]:
            for min_obs in [10, 50]:
                rls = _make_rls(seed_slope=0.35, p_init=p_init)
                rng = random.Random(42)

                x_mild = [1.0, 5.0]
                for _ in range(21):
                    observed = true_slope * 5.0 + rng.gauss(0, 1.5)
                    rls.update(x_mild, observed)

                x_cold = [1.0, 30.0]
                blend_ff = _blended_predict(rls, seeds, x_cold, min_obs)
                raw_ff = rls.predict(x_cold)
                seed_ff = sum(s * xi for s, xi in zip(seeds, x_cold))
                true_ff = true_slope * 30.0

                alpha = min(rls.observation_count / min_obs, 1.0)
                print(f"{p_init:8.2f} {min_obs:7d} {rls.observation_count:5d} "
                      f"{alpha:6.2f} {blend_ff:8.3f} {raw_ff:8.3f} "
                      f"{seed_ff:8.3f} {true_ff:8.3f}")


# ── Scenario E: MIN_RLS_OBS Sweep ───────────────────────────────────────


class TestScenarioE:
    """Sweep MIN_RLS_OBS with P_INIT candidates from B/C."""

    MIN_OBS_VALUES = [10, 20, 30, 50, 75, 100]

    def test_min_obs_sweep_closed_loop(self):
        """Closed-loop comfort across MIN_RLS_OBS values."""
        print("\nScenario E: MIN_RLS_OBS sweep (closed-loop, seed=0.35, true=0.35)")
        print(f"{'P_INIT':>8} {'MinObs':>7} {'MaxErr':>7} {'AvgErr':>7} "
              f"{'MaxHP':>6} {'Slope@end':>10}")

        for p_init in [1.0, 0.5, 0.1]:
            for min_obs in self.MIN_OBS_VALUES:
                rls = _make_rls(seed_slope=0.35, p_init=p_init)
                seeds = [0.0, 0.35]

                history = _simulate_closed_loop_with_blend(
                    rls, seeds, true_slope=0.35, min_rls_obs=min_obs,
                    p_init=p_init, n_ticks=300,
                )

                errors = [abs(h["error"]) for h in history]
                print(f"{p_init:8.2f} {min_obs:7d} {max(errors):7.3f} "
                      f"{sum(errors)/len(errors):7.3f} "
                      f"{max(h['hp_setpoint'] for h in history):6d} "
                      f"{history[-1]['slope']:10.4f}")

    def test_min_obs_sweep_wrong_seed(self):
        """How does MIN_RLS_OBS affect learning when seed is wrong?"""
        print("\nScenario E: MIN_RLS_OBS sweep (seed=0.35, true=0.6)")
        print(f"{'P_INIT':>8} {'MinObs':>7} {'MaxErr':>7} {'AvgErr':>7} "
              f"{'Slope@end':>10}")

        for p_init in [1.0, 0.5, 0.1]:
            for min_obs in self.MIN_OBS_VALUES:
                rls = _make_rls(seed_slope=0.35, p_init=p_init)
                seeds = [0.0, 0.35]

                history = _simulate_closed_loop_with_blend(
                    rls, seeds, true_slope=0.6, min_rls_obs=min_obs,
                    p_init=p_init, n_ticks=300,
                )

                errors = [abs(h["error"]) for h in history]
                print(f"{p_init:8.2f} {min_obs:7d} {max(errors):7.3f} "
                      f"{sum(errors)/len(errors):7.3f} "
                      f"{history[-1]['slope']:10.4f}")


# ── Scenario F: Overnight Solar Coefficient ─────────────────────────────


class TestScenarioF:
    """Does the solar coefficient gain false confidence from night-only data?"""

    def test_overnight_solar_confidence(self):
        """10 night observations (solar=0), then check solar coefficient state."""
        print("\nScenario F: Overnight solar coefficient")
        print(f"{'P_INIT':>8} {'SolarSlope@10':>14} {'SolarSeed':>10} "
              f"{'P_solar_diag':>13}")

        for p_init in [10.0, 1.0, 0.5, 0.1]:
            # 2-input model: outdoor_delta + solar_proxy
            seeds = [0.0, 0.35, -4.0]  # intercept, outdoor, solar
            scales = [1.0, 10.0, 0.5]
            clamps = [(-0.5, 0.5), (0.0, 2.0), (-8.0, 0.0)]

            rls = RLSModel(
                n_inputs=2,
                seed_coefficients=seeds,
                coeff_clamps=clamps,
                feature_scales=scales,
                p_init=p_init,
            )

            rng = random.Random(42)
            # 10 night observations: solar=0, outdoor_delta=10
            for _ in range(10):
                x = [1.0, 10.0, 0.0]  # no solar at night
                observed = 0.35 * 10.0 + rng.gauss(0, 1.0)  # ~3.5
                rls.update(x, observed)

            solar_coeff = rls.beta[2] / scales[2]
            n = rls.n
            p_solar = rls.P[2 * n + 2]

            print(f"{p_init:8.2f} {solar_coeff:14.4f} {-4.0:10.1f} "
                  f"{p_solar:13.6f}")

    def test_solar_not_corrupted_by_night_then_day(self):
        """After night-only training, does the first sunny observation cause a jump?"""
        print("\nScenario F (extended): Night training then sunny observation")
        print(f"{'P_INIT':>8} {'Solar@night':>12} {'Solar@1sun':>12} "
              f"{'Solar@5sun':>12} {'SeedSolar':>10}")

        for p_init in [10.0, 1.0, 0.5, 0.1]:
            seeds = [0.0, 0.35, -4.0]
            scales = [1.0, 10.0, 0.5]
            clamps = [(-0.5, 0.5), (0.0, 2.0), (-8.0, 0.0)]

            rls = RLSModel(
                n_inputs=2, seed_coefficients=seeds,
                coeff_clamps=clamps, feature_scales=scales, p_init=p_init,
            )

            rng = random.Random(42)
            # 10 night observations
            for _ in range(10):
                x = [1.0, 10.0, 0.0]
                observed = 0.35 * 10.0 + rng.gauss(0, 1.0)
                rls.update(x, observed)

            solar_after_night = rls.beta[2] / scales[2]

            # 1 sunny observation: solar=1.0, offset drops because sun helps
            x_sun = [1.0, 10.0, 1.0]
            observed = 0.35 * 10.0 + (-4.0) * 1.0 + rng.gauss(0, 0.5)
            rls.update(x_sun, observed)
            solar_after_1sun = rls.beta[2] / scales[2]

            # 4 more sunny observations
            for _ in range(4):
                observed = 0.35 * 10.0 + (-4.0) * 1.0 + rng.gauss(0, 0.5)
                rls.update(x_sun, observed)
            solar_after_5sun = rls.beta[2] / scales[2]

            print(f"{p_init:8.2f} {solar_after_night:12.4f} "
                  f"{solar_after_1sun:12.4f} {solar_after_5sun:12.4f} "
                  f"{-4.0:10.1f}")

    def test_solar_not_corrupted_by_night(self):
        """Solar coefficient should stay near seed after night-only obs."""
        for p_init in [1.0, 0.5, 0.1]:
            seeds = [0.0, 0.35, -4.0]
            scales = [1.0, 10.0, 0.5]
            clamps = [(-0.5, 0.5), (0.0, 2.0), (-8.0, 0.0)]

            rls = RLSModel(
                n_inputs=2,
                seed_coefficients=seeds,
                coeff_clamps=clamps,
                feature_scales=scales,
                p_init=p_init,
            )

            rng = random.Random(42)
            for _ in range(10):
                x = [1.0, 10.0, 0.0]
                observed = 0.35 * 10.0 + rng.gauss(0, 1.0)
                rls.update(x, observed)

            solar_coeff = rls.beta[2] / scales[2]
            # Solar coefficient should stay near -4.0 since solar=0
            # means solar dimension carries no information
            assert abs(solar_coeff - (-4.0)) < 2.0, (
                f"P_INIT={p_init}: solar coeff {solar_coeff:.2f} drifted "
                f"from seed -4.0 despite zero solar input"
            )


# ── Scenario G: Intercept Clamp Analysis ────────────────────────────────


class TestScenarioG:
    """Should the intercept be clamped at (-0.5, 0.5)?"""

    def test_intercept_clamp_effect(self):
        """Compare clamped vs unclamped intercept across scenarios."""
        print("\nScenario G: Intercept clamp effect")
        print(f"{'P_INIT':>8} {'True':>5} {'Clamp':>6} {'Slope':>7} "
              f"{'Intcpt':>8} {'IntHitClamp':>12}")

        for p_init in [10.0, 2.0, 1.0, 0.5]:
            for true_slope in [0.2, 0.6, 0.8]:
                for clamp_intercept in [True, False]:
                    if clamp_intercept:
                        clamps = [(-0.5, 0.5), (0.0, 2.0)]
                    else:
                        clamps = [None, (0.0, 2.0)]

                    rls = RLSModel(
                        n_inputs=1,
                        seed_coefficients=[0.0, 0.35],
                        coeff_clamps=clamps,
                        feature_scales=[1.0, 10.0],
                        p_init=p_init,
                    )
                    rng = random.Random(42)

                    for i in range(50):
                        od = 8.0 + 4.0 * math.sin(i * 0.3) + rng.gauss(0, 1.0)
                        od = max(0, od)
                        x = [1.0, od]
                        observed = true_slope * od + rng.gauss(0, 1.0)
                        rls.update(x, observed)

                    slope = rls.beta[1] / rls.feature_scales[1]
                    intercept = rls.beta[0] / rls.feature_scales[0]
                    hit = "YES" if clamp_intercept and abs(intercept) >= 0.49 else ""

                    label = "clmpd" if clamp_intercept else "free "
                    print(f"{p_init:8.2f} {true_slope:5.2f} {label:>6} "
                          f"{slope:7.4f} {intercept:8.4f} {hit:>12}")


# ── Scenario H: Seasonal / Long-Term Behavior ──────────────────────────


class TestScenarioH:
    """What happens over a full year with changing conditions?"""

    def test_seasonal_behavior(self):
        """Simulate 365 days of varying outdoor temps and check adaptation."""
        print("\nScenario H: Seasonal behavior (365 days, ~6 obs/day)")
        print(f"{'P_INIT':>8} {'Winter1':>8} {'Spring':>8} {'Summer':>8} "
              f"{'Fall':>8} {'Winter2':>8} {'Seed':>6}")

        # True slope varies slightly by season (insulation works differently
        # at extreme vs moderate outdoor temps)
        def seasonal_outdoor(day: int) -> float:
            """Outdoor temp in °C: sinusoidal annual cycle."""
            # Day 0 = Jan 1. Range: -10°C (winter) to 25°C (summer)
            return 7.5 + 17.5 * math.sin((day - 90) * 2 * math.pi / 365)

        for p_init in [10.0, 2.0, 1.0, 0.5]:
            rls = _make_rls(seed_slope=0.35, p_init=p_init)
            rng = random.Random(42)

            true_slope = 0.4  # Slightly different from seed
            reference = 15.0
            slopes_at = {}

            obs_count = 0
            for day in range(365):
                outdoor = seasonal_outdoor(day)
                outdoor_delta = max(0, reference - outdoor)

                # ~6 observations per day (only during heating season)
                n_obs_today = 6 if outdoor_delta > 0 else 0
                for _ in range(n_obs_today):
                    x = [1.0, outdoor_delta + rng.gauss(0, 0.5)]
                    observed = true_slope * outdoor_delta + rng.gauss(0, 1.0)
                    rls.update(x, observed)
                    obs_count += 1

                # Record at season transitions
                if day in (45, 120, 210, 300, 360):
                    slopes_at[day] = rls.beta[1] / rls.feature_scales[1]

            print(f"{p_init:8.2f} {slopes_at[45]:8.4f} {slopes_at[120]:8.4f} "
                  f"{slopes_at[210]:8.4f} {slopes_at[300]:8.4f} "
                  f"{slopes_at[360]:8.4f} {0.35:6.2f}")
            print(f"{'':>8} (total obs: {obs_count})")

    def test_winter_to_winter_memory(self):
        """Does the system remember last winter's learning?

        With lambda=0.999 and ~6 obs/day during heating season (~200 days),
        effective memory is ~1000 obs. By next winter, summer observations
        (zero heating) don't accumulate, but the P matrix evolves via the
        forgetting factor even without observations. Check if winter 2
        coefficients benefit from winter 1.
        """
        print("\nScenario H (memory): Winter-to-winter coefficient retention")
        print(f"{'P_INIT':>8} {'W1_end':>8} {'Summer':>8} {'W2_start':>9} "
              f"{'W2_obs10':>9} {'ObsTotal':>9}")

        def seasonal_outdoor(day: int) -> float:
            return 7.5 + 17.5 * math.sin((day - 90) * 2 * math.pi / 365)

        for p_init in [2.0, 1.0, 0.5]:
            rls = _make_rls(seed_slope=0.35, p_init=p_init)
            rng = random.Random(42)

            true_slope = 0.5
            reference = 15.0
            total_obs = 0

            # Winter 1: days 0-120 (Jan-Apr)
            for day in range(121):
                outdoor = seasonal_outdoor(day)
                od = max(0, reference - outdoor)
                if od > 0:
                    for _ in range(6):
                        x = [1.0, od + rng.gauss(0, 0.5)]
                        observed = true_slope * od + rng.gauss(0, 1.0)
                        rls.update(x, observed)
                        total_obs += 1

            w1_end = rls.beta[1] / rls.feature_scales[1]

            # Summer: days 121-270, no heating observations
            # (lambda decay still happens implicitly when next obs arrives)
            summer_slope = rls.beta[1] / rls.feature_scales[1]

            # Winter 2 start: day 271 (Oct)
            w2_start = rls.beta[1] / rls.feature_scales[1]

            # Winter 2: 10 observations
            for day in range(271, 281):
                outdoor = seasonal_outdoor(day)
                od = max(0, reference - outdoor)
                if od > 0:
                    for _ in range(6):
                        x = [1.0, od + rng.gauss(0, 0.5)]
                        observed = true_slope * od + rng.gauss(0, 1.0)
                        rls.update(x, observed)
                        total_obs += 1

            w2_obs10 = rls.beta[1] / rls.feature_scales[1]

            print(f"{p_init:8.2f} {w1_end:8.4f} {summer_slope:8.4f} "
                  f"{w2_start:9.4f} {w2_obs10:9.4f} {total_obs:9d}")


# ── Scenario I: Max Error Deep Dive ─────────────────────────────────────


class TestScenarioI:
    """Where does the 5°C max error come from? Is it the initial transient?"""

    def test_error_timeline(self):
        """Print error over time to see when the max error occurs."""
        print("\nScenario I: Error timeline (P_INIT=1.0, seed=0.35, true=0.35)")
        print(f"{'Tick':>5} {'Error°C':>8} {'HP':>4} {'FF':>7} "
              f"{'Integral':>9} {'Slope':>7}")

        rls = _make_rls(seed_slope=0.35, p_init=1.0)
        seeds = [0.0, 0.35]

        history = _simulate_closed_loop_with_blend(
            rls, seeds, true_slope=0.35, min_rls_obs=50, p_init=1.0,
            n_ticks=50,
        )

        for h in history[:30]:  # First 30 ticks
            print(f"{h['tick']:5d} {h['error']:8.3f} {h['hp_setpoint']:4d} "
                  f"{h['ff_offset']:7.3f} {h['integral']:9.3f} "
                  f"{h['slope']:7.4f}")

    def test_error_timeline_wrong_seed(self):
        """Error timeline when seed is wrong (0.35 vs true 0.6)."""
        print("\nScenario I: Error timeline (P_INIT=1.0, seed=0.35, true=0.6)")
        print(f"{'Tick':>5} {'Error°C':>8} {'HP':>4} {'FF':>7} "
              f"{'Integral':>9} {'Slope':>7}")

        rls = _make_rls(seed_slope=0.35, p_init=1.0)
        seeds = [0.0, 0.35]

        history = _simulate_closed_loop_with_blend(
            rls, seeds, true_slope=0.6, min_rls_obs=50, p_init=1.0,
            n_ticks=50,
        )

        for h in history[:30]:
            print(f"{h['tick']:5d} {h['error']:8.3f} {h['hp_setpoint']:4d} "
                  f"{h['ff_offset']:7.3f} {h['integral']:9.3f} "
                  f"{h['slope']:7.4f}")

        # Report the max error and when it occurs
        max_err_tick = max(range(len(history)), key=lambda i: abs(history[i]["error"]))
        print(f"\n  Max error: {abs(history[max_err_tick]['error']):.3f}°C "
              f"at tick {max_err_tick}")
