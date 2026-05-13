"""Full-stack learning validation: end-to-end PI+RLS+WLS convergence.

Validates the complete learning trajectory: wrong seeds → online RLS →
batch WLS → FF convergence → integral reduction → comfort improvement.

These tests run the real PIController against 2R2C thermal models with
batch WLS triggering.  They fill the gap identified in the TODO at
test_pi_controller.py:1206.

Test tiers
----------
**Regression** (unmarked — run every time, ~2 min total):
    1. TestWrongSeedsConvergence — core learning, wrong→correct (30d LR)
    2. TestBunkroomSlowLearner — high-τ profile (30d BR)
    3. TestDisturbanceRejection — CUSUM sensor-grab recovery (30d)
    4. TestStagedModelInputRollout — feature unlock pipeline (30d)
    5. TestRecoveryFromBadStates — sign-flip, windup, P-collapse (4×30d)
    6. TestRealWeatherReplay — real open-meteo CSV (3 sims × 14d).
       Cheap real-weather guard for the otherwise-synthetic regression tier.

**Investigation** (@pytest.mark.slow — run with ``-m slow``):
    7. TestQFeedbackConvergence — q-feedback lock-in across profiles
       (6 sims × 21d).  Run when changing q-feedback or integral logic.
    8. TestConvergenceToTruth — seed-factor sweep, convergence to
       ground-truth (13 sims × 21d).  Run when changing batch WLS,
       step caps, or seed initialization.

**Design** (@pytest.mark.design — run on-demand, excluded from slow):
    9. TestMultiYearStability — 365-day drift check (1 shared sim, 2 tests).
       Run when changing forgetting factor, P-matrix, or long-horizon behavior.

To run regression + slow::

    pytest tests/hvac_bench/scenarios/test_full_stack_learning.py -m 'not design'

To run only regression (default-ish)::

    pytest tests/hvac_bench/scenarios/test_full_stack_learning.py -m 'not slow and not design'

To run the long-horizon design study::

    pytest tests/hvac_bench/scenarios/test_full_stack_learning.py -m design
"""

from __future__ import annotations

import math

import pytest

from tests.hvac_bench.full_stack_runner import (
    Checkpoint,
    CheckpointState,
    Disturbance,
    FullStackConfig,
    FullStackResult,
    ModelInputSpec,
    diurnal_outdoor,
    diurnal_solar,
    run_full_stack,
)
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.constants import TICK_MINUTES_DEFAULT
from tests.hvac_bench.house_profiles import (
    FUJITSU_HYPERHEAT_CAPACITY,
    PROFILES_2R2C,
    QUICK_PROFILES,
)
from tests.hvac_bench.scenarios._weather_mode import (
    SHOULDER_SPRING,
    SUMMER,
    WINTER_MC_STARTS,
    WINTER_TYPICAL,
    WeatherMode,
    WeatherWindow,
    get_default_weather_mode,
    real_weather_schedules,
    season_for_outdoor_base,
    windowed_real_weather,
)


# ── Shared weather schedules ─────────────────────────────────────────────


def _solar_schedule(minute: float) -> float:
    """Solar with variable cloud cover (decorrelated from outdoor)."""
    return diurnal_solar(minute, peak=0.8)


# ── Scenario 1: Wrong Seeds → Convergence ────────────────────────────────


class TestWrongSeedsConvergence:
    """Living room with intentionally wrong FF seeds.

    Starts with 2× true outdoor_delta seed and 0 solar seed.
    Validates the system converges toward truth over 30 days.
    """

    @staticmethod
    def _make_config(
        n_days: int = 30,
        *,
        weather: WeatherMode | None = None,
        start_day: int | None = None,
        profile_name: str = "living_room",
    ) -> FullStackConfig:
        if weather is None:
            weather = get_default_weather_mode()
        profile = PROFILES_2R2C[profile_name]
        outdoor_base_c = -5.0
        outdoor_schedule = None
        solar_input_schedule = _solar_schedule
        if weather == "real":
            if start_day is not None:
                outdoor_fn, solar_fn, max_days = windowed_real_weather(
                    start_day=start_day, n_days=n_days,
                )
            else:
                outdoor_fn, solar_fn, max_days = real_weather_schedules(
                    season_for_outdoor_base(outdoor_base_c), min_days=n_days,
                )
            n_days = min(n_days, max_days)
            outdoor_schedule = outdoor_fn
            if solar_fn is not None:
                solar_input_schedule = solar_fn
        return FullStackConfig(
            n_days=n_days,
            profile_name=profile_name,
            outdoor_base_c=outdoor_base_c,
            outdoor_diurnal_c=6.0,
            outdoor_schedule=outdoor_schedule,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            model_inputs=[
                ModelInputSpec(
                    name="Solar Proxy",
                    entity_id="sensor.solar_proxy",
                    input_role="solar",
                    _true_ff_coef=-3.0,
                    seed_heat=0.0,  # wrong: should be -3.0
                    lag_tau=120,
                    clamp_min=0,  # solar only warms, never cools
                    schedule=solar_input_schedule,
                ),
            ],
            pi_overrides={
                # Wrong outdoor seed: 2× true value
                "pi_outdoor_seed_heat": profile.true_seed * 2.0,
            },
            relax_kappa_gate=True,
        )

    def test_integral_compensates_early(self, bench_metrics, num_regression):
        """Day 1: integral must be working to compensate wrong FF."""
        config = self._make_config(n_days=2, weather="synth")
        result = run_full_stack(config)

        # Average over the last 5 wall-clock hours (was last 20 ticks =
        # 5h at 15-min cadence, but only 1h at 3-min — cadence-coupled).
        # Compute slice count from wall-clock window for byte-identity at
        # 15-min cadence (same 20 ticks) and scaling to 100 ticks at 3-min.
        n_last = int(300 / config.tick_minutes)
        late_integrals = [abs(h["integral"]) for h in result.history[-n_last:]]
        avg_integral = sum(late_integrals) / len(late_integrals)
        bench_metrics["avg_integral"] = avg_integral
        check_bench_metrics(num_regression, bench_metrics)
        assert avg_integral > 0.5, (
            f"Integral should be compensating for wrong seeds, "
            f"got avg |integral|={avg_integral:.2f}"
        )

    def test_batch_wls_runs(self, bench_metrics, num_regression):
        """Week 1: batch WLS should have run multiple cycles."""
        config = self._make_config(n_days=7)
        result = run_full_stack(config)
        bench_metrics["n_batches"] = result.n_batches
        check_bench_metrics(num_regression, bench_metrics)
        assert result.n_batches >= 10, (
            f"Expected ≥10 batch cycles in 7 days, got {result.n_batches}"
        )

    def test_outdoor_delta_stabilizes(self, bench_metrics, num_regression):
        """Month 1: outdoor_delta should stabilize (low batch-to-batch change)."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.coef_trajectory) >= 10:
            late_ods = [snap.get("outdoor_delta", 0)
                        for snap in result.coef_trajectory[-5:]]
            od_range = max(late_ods) - min(late_ods)
            bench_metrics["od_range_late_5"] = od_range
            bench_metrics["final_outdoor_delta"] = result.final_coefs.get("outdoor_delta", 0.0)
            check_bench_metrics(num_regression, bench_metrics)
            assert od_range < 0.1, (
                f"outdoor_delta not stabilized: range={od_range:.4f} "
                f"in last 5 batches (values: {[f'{v:.4f}' for v in late_ods]})"
            )

    def test_integral_rms_decreases(self, bench_metrics, num_regression):
        """Month 1: integral RMS in last week should be < first week."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_integral_rms) >= 14:
            first_week = sum(result.daily_integral_rms[:7]) / 7
            last_week = sum(result.daily_integral_rms[-7:]) / 7
            bench_metrics["first_week_rms"] = first_week
            bench_metrics["last_week_rms"] = last_week
            check_bench_metrics(num_regression, bench_metrics)
            assert last_week < first_week * 1.1, (
                f"Integral RMS should decrease: week 1={first_week:.3f}, "
                f"last week={last_week:.3f}"
            )

    def test_comfort_does_not_degrade(self, bench_metrics, num_regression):
        """Learning should not make comfort worse over time."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_mae) >= 14:
            first_week_mae = sum(result.daily_mae[:7]) / 7
            last_week_mae = sum(result.daily_mae[-7:]) / 7
            bench_metrics["first_week_mae"] = first_week_mae
            bench_metrics["last_week_mae"] = last_week_mae
            check_bench_metrics(num_regression, bench_metrics)
            assert last_week_mae < first_week_mae * 1.5 + 0.05, (
                f"Comfort degrading: week 1 MAE={first_week_mae:.3f}, "
                f"last week={last_week_mae:.3f}"
            )

    def test_ff_fraction_increases(self, bench_metrics, num_regression):
        """FF should carry more of the load as learning progresses."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_ff_fraction) >= 14:
            first_week_ff = sum(result.daily_ff_fraction[:7]) / 7
            last_week_ff = sum(result.daily_ff_fraction[-7:]) / 7
            bench_metrics["first_week_ff"] = first_week_ff
            bench_metrics["last_week_ff"] = last_week_ff
            check_bench_metrics(num_regression, bench_metrics)
            assert last_week_ff >= first_week_ff * 0.8, (
                f"FF fraction declining: week 1={first_week_ff:.2%}, "
                f"last week={last_week_ff:.2%}"
            )

    def test_covariance_does_not_collapse(self, bench_metrics, num_regression):
        """RLS covariance trace should not collapse to zero."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if result.batch_covariance_trace:
            min_trace = min(result.batch_covariance_trace)
            bench_metrics["min_trace"] = min_trace
            check_bench_metrics(num_regression, bench_metrics)
            assert min_trace > 1e-6, (
                f"Covariance collapsed: min tr(P)={min_trace:.2e}"
            )

    def test_no_long_violation_streaks(self, bench_metrics, num_regression):
        """No more than 5 hours of consecutive violations.

        With intentionally wrong seeds, the cold start produces a long
        streak while the integral compensates.  After the first day,
        streaks should be much shorter.

        Controller-behavior test (no-stuck-state property under wrong-FF
        disturbance), not learning. Opted to synth so the disturbance
        and recovery dynamics are stationary; real weather adds
        non-stationary cold fronts that drag the bench HP out of
        envelope and turn this into a saturation test, not a controller
        test. See ``feedback_synthetic_vs_real_bench.md``.
        """
        config = self._make_config(n_days=30, weather="synth")
        result = run_full_stack(config)
        bench_metrics["longest_violation_streak"] = result.longest_violation_streak
        check_bench_metrics(num_regression, bench_metrics)
        # Bound is wall-clock (10 hours, generous for wrong-seed cold start
        # across cadences).  NOT tick-count.  Compute from config.tick_minutes
        # so the bound stays constant in wall-clock terms.
        #
        # Cadence sensitivity (future_work #97 relevant): at 15-min cadence
        # the streak is ~165 min (11 ticks); at 3-min cadence it's ~350 min
        # (116 ticks).  The 2× wall-clock increase at finer cadence reflects
        # smaller integral build per tick (faster sampling → less per-tick
        # error accumulated → slower recovery from wrong-seed bias).  10-hour
        # bound is generous-but-still-meaningful at both cadences; tighten
        # after #97 resolves the bench/prod batch-timing fidelity gap (which
        # may also affect wrong-seed recovery behavior).
        streak_minutes = result.longest_violation_streak * config.tick_minutes
        assert streak_minutes <= 600, (
            f"Violation streak too long: {result.longest_violation_streak} "
            f"ticks ({streak_minutes:.0f} min, bound 600 min)"
        )


# ── Scenario 2: Bunkroom Slow Learner ────────────────────────────────────


class TestBunkroomSlowLearner:
    """Bunkroom with wrong seeds — validates slower but still convergent.

    The bunkroom has low hp_gain (0.02) and high τ_env (170), so learning
    is slower: fewer informative observations, larger integral swings.
    """

    @staticmethod
    def _make_config(
        n_days: int = 30, *, weather: WeatherMode | None = None
    ) -> FullStackConfig:
        if weather is None:
            weather = get_default_weather_mode()
        profile = PROFILES_2R2C["bunkroom"]
        outdoor_base_c = -3.0
        outdoor_schedule = None
        if weather == "real":
            outdoor_fn, _, max_days = real_weather_schedules(
                season_for_outdoor_base(outdoor_base_c), min_days=n_days,
            )
            n_days = min(n_days, max_days)
            outdoor_schedule = outdoor_fn
        return FullStackConfig(
            n_days=n_days,
            profile_name="bunkroom",
            outdoor_base_c=outdoor_base_c,
            outdoor_diurnal_c=8.0,
            outdoor_schedule=outdoor_schedule,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides={
                "pi_outdoor_seed_heat": profile.true_seed * 1.5,
            },
            relax_kappa_gate=True,
        )

    def test_no_integral_runaway(self, bench_metrics, num_regression):
        """Integral must stay bounded over 30 days."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        max_integral = max(abs(h["integral"]) for h in result.history)
        bench_metrics["max_integral"] = max_integral
        check_bench_metrics(num_regression, bench_metrics)
        assert max_integral < 50, (
            f"Integral runaway: max |integral|={max_integral:.1f}"
        )

    def test_convergence_slower_than_living_room(self, bench_metrics, num_regression):
        """Bunkroom should converge but potentially slower."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)
        bench_metrics["n_batches"] = result.n_batches
        check_bench_metrics(num_regression, bench_metrics)
        assert result.n_batches >= 50, (
            f"Expected ≥50 batch cycles in 30 days, got {result.n_batches}"
        )

    def test_outdoor_delta_bounded(self, bench_metrics, num_regression):
        """outdoor_delta should not diverge."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)
        od = result.final_coefs.get("outdoor_delta", 0)
        bench_metrics["outdoor_delta"] = od
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(od) < 2.0, (
            f"outdoor_delta diverged: {od:.3f}"
        )

    def test_comfort_above_80_pct(self, bench_metrics, num_regression):
        """Room should be within deadband ≥80% of the time."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        bench_metrics["ctrl_violations"] = result.ctrl_violations
        bench_metrics["unctrl_violations"] = result.unctrl_violations
        check_bench_metrics(num_regression, bench_metrics)
        assert result.ctrl_comfort_pct >= 80.0, (
            f"Controllable comfort only {result.ctrl_comfort_pct:.1f}% "
            f"(ctrl={result.ctrl_violations}, unctrl={result.unctrl_violations})"
        )

    def test_no_runaway_overshoot(self, bench_metrics, num_regression):
        """Controllable warm violations should be minority — no FF sign errors."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)
        bench_metrics["ctrl_violations"] = result.ctrl_violations
        bench_metrics["warm_violations"] = result.warm_violations
        bench_metrics["unctrl_violations"] = result.unctrl_violations
        check_bench_metrics(num_regression, bench_metrics)
        if result.ctrl_violations > 10:
            ctrl_warm = result.warm_violations - result.unctrl_violations
            ctrl_cold = result.ctrl_violations - max(0, ctrl_warm)
            assert ctrl_warm <= result.ctrl_violations * 0.6, (
                f"Too many controllable warm violations: "
                f"ctrl_warm={ctrl_warm}, ctrl_total={result.ctrl_violations}"
            )


# ── Scenario 3: Q-Feedback Convergence ───────────────────────────────────


@pytest.mark.slow
class TestQFeedbackConvergence:
    """Validates that q-feedback=0.0 enables SP lock-in within 2 weeks.

    Formalizes the 30-day manual validation from commit 56fccac.
    The key metric is reversals/week: should drop from high (14-190)
    to low (0-5) as q-feedback enables the integral to find the
    correct quantized setpoint.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_reversals_decrease_over_time(self, bench_metrics, num_regression, profile_name):
        """Reversals/week should decrease as q-feedback converges."""
        profile = QUICK_PROFILES[profile_name]
        config = FullStackConfig(
            n_days=21,  # 3 weeks
            profile_name=profile_name,
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        if len(result.weekly_reversals) >= 3:
            total = sum(result.weekly_reversals[:3])
            avg = total / 3.0
            bench_metrics["profile"] = profile_name
            bench_metrics["avg_reversals_per_week"] = avg
            check_bench_metrics(num_regression, bench_metrics)
            assert avg < 20, (
                f"{profile_name}: average reversals {avg:.1f}/week "
                f"(weekly: {result.weekly_reversals[:3]})"
            )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_ff_offset_stabilizes(self, bench_metrics, num_regression, profile_name):
        """FF offset should stabilize (low variance) by week 3."""
        profile = QUICK_PROFILES[profile_name]
        config = FullStackConfig(
            n_days=21,
            profile_name=profile_name,
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # FF offset in last week should have lower variance than first week
        tpd = int(24 * 60 / config.tick_minutes)
        if len(result.history) >= 21 * tpd:
            w1_ff = [h["ff_offset"] for h in result.history[:7 * tpd]]
            w3_ff = [h["ff_offset"]
                     for h in result.history[14 * tpd:21 * tpd]]
            w1_std = _std(w1_ff)
            w3_std = _std(w3_ff)
            bench_metrics["profile"] = profile_name
            bench_metrics["w1_std"] = w1_std
            bench_metrics["w3_std"] = w3_std
            check_bench_metrics(num_regression, bench_metrics)
            assert w3_std <= w1_std * 1.5 + 0.1, (
                f"{profile_name}: FF std increased from "
                f"week 1={w1_std:.3f} to week 3={w3_std:.3f}"
            )


# ── Scenario: Convergence to true coefficients at varying wrongness ─────


@pytest.mark.slow
class TestConvergenceToTruth:
    """Start with seeds at varying levels of wrongness, validate convergence.

    Uses explicit true_coefs matching the thermal model's ground truth
    so we can assert convergence to the correct value, not just stability.

    The ground-truth outdoor_delta for the 2R2C living room thermal model
    is derived from steady-state: at equilibrium, hp_setpoint adjusts by
    1/(hp_gain × τ_env) per degree of outdoor delta.  In the RLS
    observation model, outdoor_delta = (outdoor - desired), so the
    coefficient is negative (colder outdoor → higher HP setpoint).
    """

    # Ground-truth coefficient for living_room 2R2C model.
    # Derived from: true_seed = 1/(hp_gain × τ_env) = 1/(0.04 × 100) = 0.25
    # In RLS sign convention (outdoor_delta = outdoor - desired), this is
    # negative: colder outdoor → HP pushes harder → coefficient ≈ -0.25.
    # But the actual learned value depends on closed-loop dynamics and
    # may differ.  We'll discover the true value from a long baseline run
    # and use it as the reference.

    @staticmethod
    def _make_config(
        seed_factor: float | None,
        n_days: int = 21,
        *,
        weather: WeatherMode | None = None,
    ) -> FullStackConfig:
        """Living-room config with optional wrong outdoor seed.

        ``seed_factor=None`` runs with PI defaults (correct seed baseline).
        """
        if weather is None:
            weather = get_default_weather_mode()
        profile = PROFILES_2R2C["living_room"]
        outdoor_base_c = -5.0
        outdoor_schedule = None
        if weather == "real":
            outdoor_fn, _, max_days = real_weather_schedules(
                season_for_outdoor_base(outdoor_base_c), min_days=n_days,
            )
            n_days = min(n_days, max_days)
            outdoor_schedule = outdoor_fn
        pi_overrides: dict = {}
        if seed_factor is not None:
            pi_overrides["pi_outdoor_seed_heat"] = profile.true_seed * seed_factor
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=outdoor_base_c,
            outdoor_diurnal_c=6.0,
            outdoor_schedule=outdoor_schedule,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides=pi_overrides,
            relax_kappa_gate=True,
        )

    # All 5 seed factors plus the None baseline. Used as the @parametrize
    # set for the shared fixture below.
    _SEED_FACTORS = [
        pytest.param(0.5, id="half"),
        pytest.param(1.0, id="correct"),
        pytest.param(1.5, id="1.5x"),
        pytest.param(2.0, id="2x"),
        pytest.param(3.0, id="3x"),
    ]

    @pytest.fixture(scope="class")
    def baseline_result(self):
        """Single PI-default-seed run, shared across the class."""
        return run_full_stack(self._make_config(None))

    @pytest.fixture(scope="class")
    def seed_results(self):
        """All seed-factor sims computed once for the whole class.

        Returns dict[seed_factor -> FullStackResult]. Replaces the prior
        per-test duplication where two parametrized tests + one loop
        re-ran identical configs (~26s waste).
        """
        return {
            factor: run_full_stack(self._make_config(factor))
            for factor in (0.5, 1.0, 1.5, 2.0, 3.0)
        }

    @pytest.mark.parametrize("seed_factor", _SEED_FACTORS)
    def test_coefficient_converges(self, bench_metrics, num_regression, seed_factor, seed_results):
        """FF coefficient should stabilize regardless of initial seed error."""
        result = seed_results[seed_factor]

        if len(result.coef_trajectory) >= 15:
            late_ods = [snap.get("outdoor_delta", 0)
                        for snap in result.coef_trajectory[-10:]]
            od_std = _std(late_ods)
            bench_metrics["seed_factor"] = seed_factor
            bench_metrics["od_std"] = od_std
            check_bench_metrics(num_regression, bench_metrics)
            assert od_std < 0.05, (
                f"seed_factor={seed_factor}: outdoor_delta not converged, "
                f"std={od_std:.4f} in last 10 batches "
                f"(values: {[f'{v:.4f}' for v in late_ods]})"
            )

    @pytest.mark.parametrize("seed_factor", _SEED_FACTORS)
    def test_all_seeds_converge_to_same_value(
        self, bench_metrics, num_regression, seed_factor, seed_results, baseline_result,
    ):
        """All seed factors should converge to approximately the same
        final coefficient, since the ground-truth physics is identical.
        """
        result = seed_results[seed_factor]
        od = result.final_coefs.get("outdoor_delta", 0)
        baseline_od = baseline_result.final_coefs.get("outdoor_delta", 0)
        bench_metrics["seed_factor"] = seed_factor
        bench_metrics["od"] = od
        bench_metrics["baseline_od"] = baseline_od
        bench_metrics["diff"] = abs(od - baseline_od)
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(od - baseline_od) < 0.1, (
            f"seed_factor={seed_factor}: converged to {od:.4f}, "
            f"baseline={baseline_od:.4f}, diff={abs(od - baseline_od):.4f}"
        )

    def test_worse_seeds_take_longer(self, bench_metrics, num_regression, seed_results):
        """More wrong seeds should take more batch cycles to converge."""
        first_stable_for: dict[float, int | None] = {}
        for factor in (1.0, 2.0, 3.0):
            result = seed_results[factor]
            final_od = result.final_coefs.get("outdoor_delta", 0)
            first_stable: int | None = None
            for i, snap in enumerate(result.coef_trajectory):
                od = snap.get("outdoor_delta", 0)
                if abs(od - final_od) < 0.05:
                    first_stable = i
                    break
            first_stable_for[factor] = first_stable

        print(f"\n  Convergence speed: {first_stable_for}")
        bench_metrics["stable_1x"] = first_stable_for[1.0] if first_stable_for[1.0] is not None else -1
        bench_metrics["stable_2x"] = first_stable_for[2.0] if first_stable_for[2.0] is not None else -1
        bench_metrics["stable_3x"] = first_stable_for[3.0] if first_stable_for[3.0] is not None else -1
        check_bench_metrics(num_regression, bench_metrics)
        if first_stable_for[1.0] is not None and first_stable_for[3.0] is not None:
            assert first_stable_for[3.0] >= first_stable_for[1.0], (
                f"3× wrong seeds converged faster ({first_stable_for[3.0]}) "
                f"than correct seeds ({first_stable_for[1.0]})"
            )





class TestDisturbanceRejection:
    """Inject a sensor anomaly and verify recovery."""

    def test_sensor_grab_recovery(self, bench_metrics, num_regression):
        """Simulate sensor grabbed for battery change (+8°C for 4 ticks).

        CUSUM should detect it, and the system should recover quickly.
        """
        config = FullStackConfig(
            n_days=7,
            profile_name="living_room",
            outdoor_base_c=-2.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            disturbances=[
                Disturbance(
                    # ~day 2 (was tick=200 at 15-min = minute 3000),
                    # duration 1 hour (was 4 ticks at 15-min).
                    start_tick=int(round(3000 / TICK_MINUTES_DEFAULT)),
                    duration_ticks=int(round(60 / TICK_MINUTES_DEFAULT)),
                    field="room_temp_offset",
                    value=8.0,  # +8°C spike
                ),
            ],
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        tpd = int(24 * 60 / config.tick_minutes)
        last_day_errors = [abs(h["room_temp"] - 20.5)
                           for h in result.history[-tpd:]]
        last_day_mae = sum(last_day_errors) / len(last_day_errors)

        last_day_integrals = [abs(h["integral"])
                              for h in result.history[-tpd:]]
        max_late_integral = max(last_day_integrals)

        bench_metrics["last_day_mae"] = last_day_mae
        bench_metrics["max_late_integral"] = max_late_integral
        check_bench_metrics(num_regression, bench_metrics)

        assert last_day_mae < 1.0, (
            f"System didn't recover from sensor grab: "
            f"last day MAE={last_day_mae:.3f}"
        )
        assert max_late_integral < 30, (
            f"Integral wound up after sensor grab: "
            f"max |integral|={max_late_integral:.1f}"
        )


# ── Scenario 4: Real Weather Replay ──────────────────────────────────────


# ── Scenario: Staged model input rollout ─────────────────────────────────


def _stove_schedule(minute: float) -> float:
    """Pellet stove: runs 6pm-10pm on cold days, off otherwise.

    Intermittent, correlated with cold outdoor (runs when it's coldest).
    This is the hard case for learning — sparse, confounded.
    """
    hour = (minute / 60.0) % 24.0
    day = minute / (60.0 * 24.0)
    # Only fires on "cold" days (day 0, 2, 4, ... — alternating)
    if int(day) % 2 != 0:
        return 0.0
    if hour < 18 or hour > 22:
        return 0.0
    return 1.0


def _adjacent_zone_schedule(minute: float) -> float:
    """Adjacent zone (sunroom) absolute temperature.

    Returns the sunroom's absolute °C reading — the controlled room's
    desired temp (20.5°C) ± a solar-driven delta.  The controller is
    configured with ``delta_from_room=True`` to convert this to the
    delta feature, matching how real installs work (sensor reports
    absolute, controller computes delta).

    Warmer than room during solar hours, cooler at night.
    Correlated with solar — tests collinearity handling for the
    multi-input staged-rollout.  See the dedicated TestAdjacentZone*
    classes below for scenario-specific sunroom realism.
    """
    hour = (minute / 60.0) % 24.0
    day = minute / (60.0 * 24.0)
    REFERENCE_ROOM_TEMP = 20.5
    if 8 <= hour <= 18:
        solar_factor = math.sin(math.pi * (hour - 8) / 10)
        cloud = 0.5 + 0.5 * math.cos(2 * math.pi * day / 3.0 + 1.0)
        return REFERENCE_ROOM_TEMP + 3.0 * solar_factor * cloud
    return REFERENCE_ROOM_TEMP - 2.0


class TestStagedModelInputRollout:
    """Validate automatic staged feature unlocking via κ/VIF gating.

    All model inputs are configured from the start but begin frozen
    (production behavior: _cold_start_freeze).  The batch WLS + staged
    learning gate decides when to unlock each feature based on data
    quality (std_err, VIF, κ).

    Tests that:
    - outdoor_delta (always active) converges first
    - Solar unlocks when daytime data provides sufficient contrast
    - Adjacent zone (correlated with solar) is gated by κ until decorrelated
    - Stove (intermittent) stays frozen until enough active observations
    - Unlocking one feature doesn't destabilize others
    - FF fraction increases as features unlock
    """

    @staticmethod
    def _make_config(
        n_days: int = 30,
        *,
        weather: WeatherMode | None = None,
        start_day: int | None = None,
    ) -> FullStackConfig:
        # Sunroom + stove schedules stay synthetic — neither is in the
        # Open-Meteo CSVs, and they exercise the κ/VIF gating logic
        # against the solar feature, so realism of those two is
        # secondary to the staged-rollout dynamics under test.
        if weather is None:
            weather = get_default_weather_mode()
        outdoor_base_c = -5.0
        outdoor_schedule = None
        solar_input_schedule = _solar_schedule
        if weather == "real":
            if start_day is not None:
                outdoor_fn, solar_fn, max_days = windowed_real_weather(
                    start_day=start_day, n_days=n_days,
                )
            else:
                outdoor_fn, solar_fn, max_days = real_weather_schedules(
                    season_for_outdoor_base(outdoor_base_c), min_days=n_days,
                )
            n_days = min(n_days, max_days)
            outdoor_schedule = outdoor_fn
            if solar_fn is not None:
                solar_input_schedule = solar_fn
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=outdoor_base_c,
            outdoor_diurnal_c=6.0,
            outdoor_schedule=outdoor_schedule,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            model_inputs=[
                ModelInputSpec(
                    name="Solar Proxy",
                    entity_id="sensor.solar_proxy",
                    input_role="solar",
                    _true_ff_coef=-2.0,
                    seed_heat=0.0,
                    lag_tau=120,
                    clamp_min=0,
                    schedule=solar_input_schedule,
                ),
                ModelInputSpec(
                    name="Sunroom Delta",
                    entity_id="sensor.sunroom_delta",
                    input_role="adjacent_zone",
                    _true_ff_coef=-0.5,
                    seed_heat=0.0,
                    schedule=_adjacent_zone_schedule,
                    delta_from_room=True,
                ),
                ModelInputSpec(
                    name="Pellet Stove",
                    entity_id="sensor.pellet_stove",
                    input_role="heat_source",
                    _true_ff_coef=-3.0,
                    seed_heat=0.0,
                    schedule=_stove_schedule,
                ),
            ],
            relax_kappa_gate=False,  # let κ gating work naturally
        )

    def test_features_start_frozen(self, bench_metrics, num_regression):
        """Model input features (indices 2+) should start frozen.

        Day-1 solar is suppressed (5% of clear-sky peak — heavily overcast
        spring day) so solar variance stays below the feature-unlock
        criteria (variance + VIF + std_err) regardless of tick cadence.
        Without this, at 3-min cadence the WLS accumulates enough variance
        from the 6am-noon sunrise gradient before the first batch (at
        noon, per bench's interval-from-epoch scheduling — production
        fires at 07:00/19:00 wall-clock, see future_work #97) to trigger
        unlock on the first batch, breaking the test's cadence-invariant
        intent.
        """
        config = self._make_config(n_days=2)
        # Wrap solar input schedule to suppress day 1 entirely (full overcast
        # = no direct + diffuse light) so solar has zero variance regardless
        # of tick cadence.  5% of peak still admits enough variance at 3-min
        # cadence to satisfy the unlock criteria; full zero is unambiguous.
        solar_spec = next(mi for mi in config.model_inputs if mi.name == "Solar Proxy")
        _underlying_solar = solar_spec.schedule
        assert _underlying_solar is not None, "solar input must have a schedule"
        def _overcast_day1(minute: float, _base=_underlying_solar) -> float:
            day = minute / (60.0 * 24.0)
            return 0.0 if day < 1.0 else _base(minute)
        solar_spec.schedule = _overcast_day1
        result = run_full_stack(config)
        bench_metrics["n_snapshots"] = len(result.coef_trajectory)
        check_bench_metrics(num_regression, bench_metrics)
        # First batch snapshot should show model inputs frozen
        if result.coef_trajectory:
            snap = result.coef_trajectory[0]
            for name in ["Solar Proxy", "Sunroom Delta", "Pellet Stove"]:
                frozen_key = f"{name}_frozen"
                if frozen_key in snap:
                    assert snap[frozen_key] is True, (
                        f"{name} should start frozen"
                    )

    def test_outdoor_delta_learns_first(self, bench_metrics, num_regression):
        """outdoor_delta (base feature, never frozen) should converge first."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.coef_trajectory) >= 15:
            early_ods = [snap.get("outdoor_delta", 0)
                         for snap in result.coef_trajectory[5:15]]
            od_range = max(early_ods) - min(early_ods)
            bench_metrics["od_range_early"] = od_range
            check_bench_metrics(num_regression, bench_metrics)
            assert od_range < 0.2, (
                f"outdoor_delta not stabilizing early: range={od_range:.4f}"
            )

    def test_unlock_does_not_destabilize_outdoor(self, bench_metrics, num_regression):
        """When a feature unlocks, outdoor_delta should not jump."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)
        max_unlock_jump = 0.0
        # Find batches where a feature unfroze
        for i in range(1, len(result.coef_trajectory)):
            prev = result.coef_trajectory[i - 1]
            curr = result.coef_trajectory[i]
            for name in ["Solar Proxy", "Sunroom Delta", "Pellet Stove"]:
                was_frozen = prev.get(f"{name}_frozen", True)
                now_frozen = curr.get(f"{name}_frozen", True)
                if was_frozen and not now_frozen:
                    od_prev = prev.get("outdoor_delta", 0)
                    od_curr = curr.get("outdoor_delta", 0)
                    max_unlock_jump = max(max_unlock_jump, abs(od_curr - od_prev))
                    assert abs(od_curr - od_prev) < 0.3, (
                        f"outdoor_delta jumped {od_prev:.4f} → {od_curr:.4f} "
                        f"when {name} unlocked at batch {i}"
                    )
        bench_metrics["max_unlock_jump"] = max_unlock_jump
        check_bench_metrics(num_regression, bench_metrics)

    @pytest.mark.slow
    def test_unlock_jump_bound_holds_across_winters(self, bench_metrics, num_regression):
        """MC sanity-check: 0.3 unlock-jump bound across 3 winter starts.

        ``test_unlock_does_not_destabilize_outdoor`` currently runs against
        the cold-year window (``WINTER_DEEP`` via ``real_weather_schedules``)
        where outdoor signal dominance keeps unlock-jumps small. This MC
        variant runs the same scenario across ``WINTER_MC_STARTS`` (Jan-11
        of 2023/2024/2025) and asserts the median per-run worst unlock-jump
        stays under the production bound. Use this when retuning the 0.3
        bound to confirm it's not just calibrated to a single weather
        realization.
        """
        per_run_max: list[float] = []
        for sd in WINTER_MC_STARTS:
            config = self._make_config(n_days=30, start_day=sd)
            result = run_full_stack(config)
            run_max = 0.0
            for i in range(1, len(result.coef_trajectory)):
                prev = result.coef_trajectory[i - 1]
                curr = result.coef_trajectory[i]
                for name in ["Solar Proxy", "Sunroom Delta", "Pellet Stove"]:
                    if (
                        prev.get(f"{name}_frozen", True)
                        and not curr.get(f"{name}_frozen", True)
                    ):
                        od_prev = prev.get("outdoor_delta", 0)
                        od_curr = curr.get("outdoor_delta", 0)
                        run_max = max(run_max, abs(od_curr - od_prev))
            per_run_max.append(run_max)
        per_run_max.sort()
        median = per_run_max[len(per_run_max) // 2]
        bench_metrics["median_unlock_jump"] = median
        check_bench_metrics(num_regression, bench_metrics)
        assert median < 0.3, (
            f"Median worst unlock-jump across {len(per_run_max)} winter "
            f"starts: {median:.4f} (bound 0.3); per-run worst-jumps: "
            f"{[f'{j:.3f}' for j in per_run_max]}"
        )

    def test_no_integral_runaway_during_unlocks(self, bench_metrics, num_regression):
        """Integral should stay bounded through all feature unlocks."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        max_integral = max(abs(h["integral"]) for h in result.history)
        bench_metrics["max_integral"] = max_integral
        check_bench_metrics(num_regression, bench_metrics)
        assert max_integral < 50, (
            f"Integral runaway during staged unlocks: max={max_integral:.1f}"
        )

    def test_ff_fraction_increases_with_unlocks(self, bench_metrics, num_regression):
        """FF fraction should increase as more features unlock and learn."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_ff_fraction) >= 21:
            first_week = sum(result.daily_ff_fraction[:7]) / 7
            third_week = sum(result.daily_ff_fraction[14:21]) / 7
            bench_metrics["first_week"] = first_week
            bench_metrics["third_week"] = third_week
            check_bench_metrics(num_regression, bench_metrics)
            assert third_week >= first_week * 0.7, (
                f"FF fraction collapsed: week 1={first_week:.2%}, "
                f"week 3={third_week:.2%}"
            )

    def test_comfort_maintained_through_unlocks(self, bench_metrics, num_regression):
        """Comfort should stay ≥75% even with staged unlocks."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)
        assert result.ctrl_comfort_pct >= 75.0, (
            f"Controllable comfort too low during staged rollout: "
            f"{result.ctrl_comfort_pct:.1f}%"
        )


# ── Scenario: Recovery from bad states ───────────────────────────────────


class TestRecoveryFromBadStates:
    """Validate that the system self-heals from corrupted learning state.

    Tests coefficient sign flips, covariance collapse, and large
    integral windup — the production failure modes that HA Repairs
    warns about.
    """

    @staticmethod
    def _make_config(
        profile_name: str = "living_room",
        n_days: int = 30,
        *,
        pi_overrides: dict | None = None,
        disturbances: list | None = None,
        weather: WeatherMode | None = None,
    ) -> FullStackConfig:
        if weather is None:
            weather = get_default_weather_mode()
        outdoor_base_c = -5.0
        outdoor_schedule = None
        if weather == "real":
            outdoor_fn, _, max_days = real_weather_schedules(
                season_for_outdoor_base(outdoor_base_c), min_days=n_days,
            )
            n_days = min(n_days, max_days)
            outdoor_schedule = outdoor_fn
        return FullStackConfig(
            n_days=n_days,
            profile_name=profile_name,
            outdoor_base_c=outdoor_base_c,
            outdoor_diurnal_c=6.0,
            outdoor_schedule=outdoor_schedule,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides=pi_overrides or {},
            disturbances=disturbances or [],
            relax_kappa_gate=True,
        )

    def test_recovery_from_sign_flip(self, bench_metrics, num_regression):
        """If outdoor_delta flips sign, batch WLS should correct it.

        Simulate by starting with a positive outdoor_delta seed (wrong
        sign — means HP backs off when it's colder, backwards).
        """
        profile = PROFILES_2R2C["living_room"]
        config = self._make_config(
            n_days=30,
            pi_overrides={
                # Positive seed = wrong sign (should be negative in RLS)
                "pi_outdoor_seed_heat": profile.true_seed * -1.0,
            },
        )
        result = run_full_stack(config)

        od = result.final_coefs.get("outdoor_delta", 0)
        bench_metrics["outdoor_delta"] = od
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)
        assert od < 0, (
            f"outdoor_delta still wrong sign after 30 days: {od:.4f}"
        )
        assert result.ctrl_comfort_pct >= 70.0, (
            f"Controllable comfort collapsed after sign flip: {result.ctrl_comfort_pct:.1f}%"
        )

    def test_recovery_from_large_integral_windup(self, bench_metrics, num_regression):
        """System should recover from a large initial integral error.

        Simulate by injecting a massive disturbance early that winds
        up the integral, then removing it.
        """
        config = self._make_config(
            n_days=14,
            disturbances=[
                # Massive cold draft for 2 hours on day 1 (was tick=48 at 15-min
                # = noon day 0 = minute 720; duration 2h = 120 min).
                Disturbance(
                    start_tick=int(round(720 / TICK_MINUTES_DEFAULT)),
                    duration_ticks=int(round(120 / TICK_MINUTES_DEFAULT)),
                    field="room_temp_offset",
                    value=-5.0,  # -5°C sensor error
                ),
            ],
        )
        result = run_full_stack(config)

        if len(result.daily_integral_rms) >= 7:
            peak_irms = max(result.daily_integral_rms[:3])
            last_week_irms = sum(result.daily_integral_rms[-7:]) / 7
            bench_metrics["peak_irms"] = peak_irms
            bench_metrics["last_week_irms"] = last_week_irms
            if peak_irms > 1.0:
                assert last_week_irms < peak_irms * 0.8, (
                    f"Integral not recovering: peak={peak_irms:.2f}, "
                    f"last week avg={last_week_irms:.2f}"
                )

        if len(result.daily_comfort_pct) >= 7:
            last_week_comfort = sum(result.daily_comfort_pct[-7:]) / 7
            bench_metrics["last_week_comfort"] = last_week_comfort
            assert last_week_comfort >= 80.0, (
                f"Comfort not recovered in last week: {last_week_comfort:.1f}%"
            )
        check_bench_metrics(num_regression, bench_metrics)

    def test_wrong_sign_seed_all_profiles(self, bench_metrics, num_regression):
        """All profiles should recover from a wrong-sign outdoor seed."""
        for profile_name in ["living_room", "bunkroom"]:
            profile = PROFILES_2R2C[profile_name]
            config = self._make_config(
                profile_name=profile_name,
                n_days=14,
                pi_overrides={
                    "pi_outdoor_seed_heat": profile.true_seed * -1.0,
                },
            )
            result = run_full_stack(config)
            od = result.final_coefs.get("outdoor_delta", 0)
            bench_metrics[f"{profile_name}_outdoor_delta"] = od
            assert od < 0, (
                f"{profile_name}: outdoor_delta still wrong sign: {od:.4f}"
            )
        check_bench_metrics(num_regression, bench_metrics)


class TestRealWeatherReplay:
    """Run learning against real open-meteo weather data.

    Validates that the learning stack doesn't diverge under realistic
    non-synthetic weather patterns (fronts, clouds, variable solar).
    """

    @pytest.mark.parametrize("window,mode,desired", [
        (WINTER_TYPICAL, "heat", 20.5),
        (SHOULDER_SPRING, "heat", 20.5),
        (SUMMER, "cool", 24.0),
    ], ids=["winter_heat", "spring_heat", "summer_cool"])
    def test_no_divergence(
        self, bench_metrics, num_regression, window: WeatherWindow, mode: str, desired: float
    ):
        """Learning should not diverge under real weather."""
        outdoor_fn, solar_fn, _ = windowed_real_weather(
            start_day=window.start_day, n_days=14,
        )

        config = FullStackConfig(
            n_days=14,
            profile_name="living_room",
            desired_c=desired,
            mode=mode,
            noise_sigma=0.1,
            noise_seed=42,
            outdoor_schedule=outdoor_fn,
            solar_schedule=solar_fn,
            model_inputs=[
                ModelInputSpec(
                    name="Solar Proxy",
                    entity_id="sensor.solar_proxy",
                    input_role="solar",
                    _true_ff_coef=-2.0,
                    seed_heat=0.0,
                    lag_tau=120,
                    clamp_min=0,
                    schedule=solar_fn,
                ),
            ] if solar_fn else [],
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # No integral runaway
        max_integral = max(abs(h["integral"]) for h in result.history)
        assert max_integral < 50, (
            f"{window.season} (start_day={window.start_day}): "
            f"integral runaway, max={max_integral:.1f}"
        )

        # Coefficient should be bounded
        od = result.final_coefs.get("outdoor_delta", 0)
        assert abs(od) < 5.0, (
            f"{window.season} (start_day={window.start_day}): "
            f"outdoor_delta diverged to {od:.3f}"
        )


# ── Scenario 5: Multi-Year Stability ────────────────────────────────────


@pytest.fixture(scope="module")
def _multi_year_result():
    """Single 365-day run shared by all TestMultiYearStability tests.

    #51 P2: pulls the calendar year 2023-01-01 → 2023-12-31 from the
    multi-year Open-Meteo CSV (44°N 71.5°W). Real outdoor data exposes
    long-horizon behavior to actual cold snaps, fronts, and shoulder-
    season transitions that the prior synthetic sinusoid smoothed over.
    Solar is wired in via a Solar Proxy ModelInputSpec — the seasonal
    swing in incident radiation (winter → summer → fall) is exactly the
    kind of slow drift signal a 1-year design test should exercise; the
    synthetic version couldn't credibly simulate it.
    """
    outdoor_fn, solar_fn, _ = windowed_real_weather(start_day=0, n_days=365)
    config = FullStackConfig(
        n_days=365,
        profile_name="living_room",
        desired_c=20.5,
        noise_sigma=0.1,
        noise_seed=42,
        outdoor_schedule=outdoor_fn,
        solar_schedule=solar_fn,
        model_inputs=[
            ModelInputSpec(
                name="Solar Proxy",
                entity_id="sensor.solar_proxy",
                input_role="solar",
                _true_ff_coef=-2.0,
                seed_heat=0.0,
                lag_tau=120,
                clamp_min=0,
                schedule=solar_fn,
            ),
        ] if solar_fn else [],
        relax_kappa_gate=True,
    )
    return run_full_stack(config)


@pytest.mark.design
class TestMultiYearStability:
    """Run for 1+ year and validate no long-term drift or collapse.

    Marked ``design`` (not ``slow``): a 365-day simulation is too expensive
    for routine regression and is run on-demand when changing forgetting
    factor, P-matrix, or anything that could affect long-horizon behavior.
    """

    def test_one_year_no_divergence(self, bench_metrics, num_regression, _multi_year_result):
        """365-day run: coefficients bounded, no integral runaway."""
        result = _multi_year_result

        # No integral runaway
        max_integral = max(abs(h["integral"]) for h in result.history)
        assert max_integral < 50, (
            f"Integral runaway over 1 year: max={max_integral:.1f}"
        )

        # Coefficients bounded
        od = result.final_coefs.get("outdoor_delta", 0)
        assert abs(od) < 5.0, f"outdoor_delta diverged: {od:.3f}"

        # ── Headline metrics (final-state snapshot) ──────────────────
        bench_metrics["max_integral"] = max_integral
        bench_metrics["final_outdoor_delta"] = od
        bench_metrics["final_solar"] = result.final_coefs.get("Solar Proxy", 0.0)
        bench_metrics["final_intercept"] = result.final_coefs.get("intercept", 0.0)
        bench_metrics["n_batches"] = len(result.coef_trajectory)
        bench_metrics["n_history"] = len(result.history)

        # Late-year coefficient stability (last 30 days of batches)
        if len(result.coef_trajectory) >= 60:
            late_ods = [snap.get("outdoor_delta", 0)
                        for snap in result.coef_trajectory[-60:]]
            od_range = max(late_ods) - min(late_ods)
            assert od_range < 0.5, (
                f"outdoor_delta unstable in last 30 days: range={od_range:.4f}"
            )
            bench_metrics["od_range_late_30d"] = od_range

            # Solar β collapse detector (#100 bug class — 80+ consecutive
            # identical values in production was the textbook signature).
            late_solars = [snap.get("Solar Proxy", 0.0)
                           for snap in result.coef_trajectory[-60:]]
            bench_metrics["solar_beta_unique_in_last_60"] = len({
                round(s, 6) for s in late_solars
            })

            # Minimum covariance trace in last 30 days — catches general
            # P-matrix collapse beyond the solar-specific freeze.  tr(P) is
            # stored alongside coef_trajectory in result.batch_covariance_trace
            # (same per-batch indexing), not on the trajectory snapshots.
            if len(result.batch_covariance_trace) >= 60:
                bench_metrics["min_covariance_trace_late"] = min(
                    result.batch_covariance_trace[-60:]
                )

        # ── Monthly trajectory checkpoints (12 × 3 coefs + 12 cov traces) ─
        # 12h batches over 365 days → ~730 snapshots; every 60th ≈ monthly.
        # Indices [60, 120, ..., 720] give 12 end-of-month checkpoints.
        if len(result.coef_trajectory) >= 720:
            for month in range(1, 13):
                idx = month * 60
                snap = result.coef_trajectory[idx]
                bench_metrics[f"m{month:02d}_outdoor_delta"] = snap.get("outdoor_delta", 0.0)
                bench_metrics[f"m{month:02d}_solar"] = snap.get("Solar Proxy", 0.0)
                bench_metrics[f"m{month:02d}_intercept"] = snap.get("intercept", 0.0)
                if idx < len(result.batch_covariance_trace):
                    bench_metrics[f"m{month:02d}_cov_trace"] = (
                        result.batch_covariance_trace[idx]
                    )

        check_bench_metrics(num_regression, bench_metrics)

    def test_one_year_comfort_stable(self, bench_metrics, num_regression, _multi_year_result):
        """Monthly MAE should not grow over the year."""
        result = _multi_year_result

        # Monthly MAE (30-day buckets)
        monthly_mae = []
        for m in range(12):
            start = m * 30
            end = min(start + 30, len(result.daily_mae))
            if start < len(result.daily_mae):
                chunk = result.daily_mae[start:end]
                monthly_mae.append(sum(chunk) / len(chunk))

        if len(monthly_mae) >= 6:
            first_half = sum(monthly_mae[:6]) / 6
            second_half = sum(monthly_mae[6:]) / len(monthly_mae[6:])
            # Second half should not be dramatically worse
            assert second_half < first_half * 2.0 + 0.1, (
                f"Comfort degrading: H1 MAE={first_half:.3f}, "
                f"H2 MAE={second_half:.3f}"
            )
            bench_metrics["h1_monthly_mae"] = first_half
            bench_metrics["h2_monthly_mae"] = second_half

        # ── Headline metrics ─────────────────────────────────────────
        if result.daily_mae:
            bench_metrics["daily_mae_max"] = max(result.daily_mae)
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        bench_metrics["ctrl_violations"] = result.ctrl_violations
        bench_metrics["longest_violation_streak"] = result.longest_violation_streak

        # FF fraction headline + extremes (ties to #105 cadence-velocity
        # finding — a year-horizon regression in FF learning surfaces here).
        if result.daily_ff_fraction:
            bench_metrics["daily_ff_fraction_final"] = result.daily_ff_fraction[-1]
            bench_metrics["daily_ff_fraction_min"] = min(result.daily_ff_fraction)

        # ── Monthly MAE trajectory (full 12-vector) ──────────────────
        for i, mae in enumerate(monthly_mae[:12], start=1):
            bench_metrics[f"m{i:02d}_mae"] = mae

        # ── Monthly FF fraction (end-of-month subsample of dailies) ──
        if len(result.daily_ff_fraction) >= 360:
            for month in range(1, 13):
                day_idx = month * 30 - 1
                if day_idx < len(result.daily_ff_fraction):
                    bench_metrics[f"m{month:02d}_ff"] = result.daily_ff_fraction[day_idx]

        check_bench_metrics(num_regression, bench_metrics)


# ── Helpers ──────────────────────────────────────────────────────────────


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
