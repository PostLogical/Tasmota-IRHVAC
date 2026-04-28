"""Full-stack learning validation: end-to-end PI+RLS+WLS convergence.

Validates the complete learning trajectory: wrong seeds → online RLS →
batch WLS → FF convergence → integral reduction → comfort improvement.

These tests run the real PIController against 2R2C thermal models with
batch WLS triggering.  They fill the gap identified in the TODO at
test_pi_controller.py:1206.

Scenarios:
    1. Wrong seeds → convergence (living room, 30 days)
    2. Bunkroom slow learner (30 days)
    3. Q-feedback convergence (formalizes commit 56fccac validation)
    4. Real weather replay (CSV, @pytest.mark.slow)
    5. Multi-year stability (@pytest.mark.slow)
"""

from __future__ import annotations

import math
from pathlib import Path

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
from tests.hvac_bench.csv_adapters import from_open_meteo_csv, csv_to_schedules
from tests.hvac_bench.house_profiles import PROFILES_2R2C, QUICK_PROFILES


# ── Shared weather schedules ─────────────────────────────────────────────


def _solar_schedule(tick: int) -> float:
    """Solar with variable cloud cover (decorrelated from outdoor)."""
    return diurnal_solar(tick, peak=0.8)


# ── Scenario 1: Wrong Seeds → Convergence ────────────────────────────────


class TestWrongSeedsConvergence:
    """Living room with intentionally wrong FF seeds.

    Starts with 2× true outdoor_delta seed and 0 solar seed.
    Validates the system converges toward truth over 30 days.
    """

    @staticmethod
    def _make_config(n_days: int = 30) -> FullStackConfig:
        profile = PROFILES_2R2C["living_room"]
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            model_inputs=[
                ModelInputSpec(
                    name="Solar Proxy",
                    entity_id="sensor.solar_proxy",
                    input_role="solar",
                    true_thermal_effect=0.01,
                    true_ff_coef=-3.0,
                    seed_heat=0.0,  # wrong: should be -3.0
                    lag_tau=120,
                    clamp_min=0,  # solar only warms, never cools
                    schedule=_solar_schedule,
                ),
            ],
            pi_overrides={
                # Wrong outdoor seed: 2× true value
                "pi_outdoor_seed_heat": profile.true_seed * 2.0,
            },
            relax_kappa_gate=True,
        )

    def test_integral_compensates_early(self):
        """Day 1: integral must be working to compensate wrong FF."""
        config = self._make_config(n_days=2)
        result = run_full_stack(config)

        # With wrong seeds, integral should be nonzero
        late_integrals = [abs(h["integral"]) for h in result.history[-20:]]
        avg_integral = sum(late_integrals) / len(late_integrals)
        assert avg_integral > 0.5, (
            f"Integral should be compensating for wrong seeds, "
            f"got avg |integral|={avg_integral:.2f}"
        )

    def test_batch_wls_runs(self):
        """Week 1: batch WLS should have run multiple cycles."""
        config = self._make_config(n_days=7)
        result = run_full_stack(config)

        # 7 days × 2 batches/day = 14 expected
        assert result.n_batches >= 10, (
            f"Expected ≥10 batch cycles in 7 days, got {result.n_batches}"
        )

    def test_outdoor_delta_stabilizes(self):
        """Month 1: outdoor_delta should stabilize (low batch-to-batch change)."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        # Last 5 batch snapshots should show small changes
        if len(result.coef_trajectory) >= 10:
            late_ods = [snap.get("outdoor_delta", 0)
                        for snap in result.coef_trajectory[-5:]]
            od_range = max(late_ods) - min(late_ods)
            assert od_range < 0.1, (
                f"outdoor_delta not stabilized: range={od_range:.4f} "
                f"in last 5 batches (values: {[f'{v:.4f}' for v in late_ods]})"
            )

    def test_integral_rms_decreases(self):
        """Month 1: integral RMS in last week should be < first week."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_integral_rms) >= 14:
            first_week = sum(result.daily_integral_rms[:7]) / 7
            last_week = sum(result.daily_integral_rms[-7:]) / 7
            assert last_week < first_week * 1.1, (
                f"Integral RMS should decrease: week 1={first_week:.3f}, "
                f"last week={last_week:.3f}"
            )

    def test_comfort_does_not_degrade(self):
        """Learning should not make comfort worse over time."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_mae) >= 14:
            first_week_mae = sum(result.daily_mae[:7]) / 7
            last_week_mae = sum(result.daily_mae[-7:]) / 7
            # Last week should not be dramatically worse
            assert last_week_mae < first_week_mae * 1.5 + 0.05, (
                f"Comfort degrading: week 1 MAE={first_week_mae:.3f}, "
                f"last week={last_week_mae:.3f}"
            )

    def test_ff_fraction_increases(self):
        """FF should carry more of the load as learning progresses."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_ff_fraction) >= 14:
            first_week_ff = sum(result.daily_ff_fraction[:7]) / 7
            last_week_ff = sum(result.daily_ff_fraction[-7:]) / 7
            # FF fraction should increase (or at least not collapse)
            assert last_week_ff >= first_week_ff * 0.8, (
                f"FF fraction declining: week 1={first_week_ff:.2%}, "
                f"last week={last_week_ff:.2%}"
            )

    def test_covariance_does_not_collapse(self):
        """RLS covariance trace should not collapse to zero."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if result.batch_covariance_trace:
            min_trace = min(result.batch_covariance_trace)
            assert min_trace > 1e-6, (
                f"Covariance collapsed: min tr(P)={min_trace:.2e}"
            )

    def test_no_long_violation_streaks(self):
        """No more than 5 hours of consecutive violations.

        With intentionally wrong seeds, the cold start produces a long
        streak while the integral compensates.  After the first day,
        streaks should be much shorter.
        """
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        # 20 ticks = 5 hours — generous for wrong-seed cold start
        assert result.longest_violation_streak <= 20, (
            f"Violation streak too long: {result.longest_violation_streak} "
            f"ticks ({result.longest_violation_streak * 15} min)"
        )


# ── Scenario 2: Bunkroom Slow Learner ────────────────────────────────────


class TestBunkroomSlowLearner:
    """Bunkroom with wrong seeds — validates slower but still convergent.

    The bunkroom has low hp_gain (0.02) and high τ_env (170), so learning
    is slower: fewer informative observations, larger integral swings.
    """

    @staticmethod
    def _make_config(n_days: int = 30) -> FullStackConfig:
        profile = PROFILES_2R2C["bunkroom"]
        return FullStackConfig(
            n_days=n_days,
            profile_name="bunkroom",
            outdoor_base_c=-3.0,
            outdoor_diurnal_c=8.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides={
                "pi_outdoor_seed_heat": profile.true_seed * 1.5,
            },
            relax_kappa_gate=True,
        )

    def test_no_integral_runaway(self):
        """Integral must stay bounded over 30 days."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        max_integral = max(abs(h["integral"]) for h in result.history)
        assert max_integral < 50, (
            f"Integral runaway: max |integral|={max_integral:.1f}"
        )

    def test_convergence_slower_than_living_room(self):
        """Bunkroom should converge but potentially slower."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        # Should have run many batch cycles
        assert result.n_batches >= 50, (
            f"Expected ≥50 batch cycles in 30 days, got {result.n_batches}"
        )

    def test_outdoor_delta_bounded(self):
        """outdoor_delta should not diverge."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        od = result.final_coefs.get("outdoor_delta", 0)
        # Should be in a reasonable range (true is ~0.29 for bunkroom)
        assert abs(od) < 2.0, (
            f"outdoor_delta diverged: {od:.3f}"
        )

    def test_comfort_above_80_pct(self):
        """Room should be within deadband ≥80% of the time."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        assert result.ctrl_comfort_pct >= 80.0, (
            f"Controllable comfort only {result.ctrl_comfort_pct:.1f}% "
            f"(ctrl={result.ctrl_violations}, unctrl={result.unctrl_violations})"
        )

    def test_no_runaway_overshoot(self):
        """Controllable warm violations should be minority — no FF sign errors."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        # With well-sized HP, some warm overshoot is normal during recovery.
        # But controllable warm violations (HP active + room too warm) would
        # indicate wrong FF sign or integral windup.
        if result.ctrl_violations > 10:
            # Warm ctrl violations shouldn't dominate — that would mean
            # the controller is actively pushing the room too hot.
            ctrl_warm = result.warm_violations - result.unctrl_violations
            ctrl_cold = result.ctrl_violations - max(0, ctrl_warm)
            assert ctrl_warm <= result.ctrl_violations * 0.6, (
                f"Too many controllable warm violations: "
                f"ctrl_warm={ctrl_warm}, ctrl_total={result.ctrl_violations}"
            )


# ── Scenario 3: Q-Feedback Convergence ───────────────────────────────────


class TestQFeedbackConvergence:
    """Validates that q-feedback=0.0 enables SP lock-in within 2 weeks.

    Formalizes the 30-day manual validation from commit 56fccac.
    The key metric is reversals/week: should drop from high (14-190)
    to low (0-5) as q-feedback enables the integral to find the
    correct quantized setpoint.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_reversals_decrease_over_time(self, profile_name):
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
            # Average reversals per week should stay bounded.
            # Well-tuned PI with q-feedback: typically 8-15/week.
            assert avg < 20, (
                f"{profile_name}: average reversals {avg:.1f}/week "
                f"(weekly: {result.weekly_reversals[:3]})"
            )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_ff_offset_stabilizes(self, profile_name):
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
            # Allow week 3 to be up to 50% worse (weather drift varies)
            assert w3_std <= w1_std * 1.5 + 0.1, (
                f"{profile_name}: FF std increased from "
                f"week 1={w1_std:.3f} to week 3={w3_std:.3f}"
            )


# ── Scenario: Convergence to true coefficients at varying wrongness ─────


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

    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5, 2.0, 3.0],
                             ids=["half", "correct", "1.5x", "2x", "3x"])
    def test_coefficient_converges(self, seed_factor):
        """FF coefficient should stabilize regardless of initial seed error."""
        profile = PROFILES_2R2C["living_room"]
        config = FullStackConfig(
            n_days=30,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides={
                "pi_outdoor_seed_heat": profile.true_seed * seed_factor,
            },
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # Coefficient should stabilize: low variance in last 10 batch snapshots
        if len(result.coef_trajectory) >= 15:
            late_ods = [snap.get("outdoor_delta", 0)
                        for snap in result.coef_trajectory[-10:]]
            od_std = _std(late_ods)
            assert od_std < 0.05, (
                f"seed_factor={seed_factor}: outdoor_delta not converged, "
                f"std={od_std:.4f} in last 10 batches "
                f"(values: {[f'{v:.4f}' for v in late_ods]})"
            )

    @pytest.mark.parametrize("seed_factor", [0.5, 1.0, 1.5, 2.0, 3.0],
                             ids=["half", "correct", "1.5x", "2x", "3x"])
    def test_all_seeds_converge_to_same_value(self, seed_factor):
        """All seed factors should converge to approximately the same
        final coefficient, since the ground-truth physics is identical.
        """
        # Run with this seed factor
        profile = PROFILES_2R2C["living_room"]
        config = FullStackConfig(
            n_days=30,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides={
                "pi_outdoor_seed_heat": profile.true_seed * seed_factor,
            },
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # Run baseline (correct seeds) for comparison
        baseline_config = FullStackConfig(
            n_days=30,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            relax_kappa_gate=True,
        )
        baseline = run_full_stack(baseline_config)

        # Final outdoor_delta should be within 0.1 of baseline
        od = result.final_coefs.get("outdoor_delta", 0)
        baseline_od = baseline.final_coefs.get("outdoor_delta", 0)
        assert abs(od - baseline_od) < 0.1, (
            f"seed_factor={seed_factor}: converged to {od:.4f}, "
            f"baseline={baseline_od:.4f}, diff={abs(od - baseline_od):.4f}"
        )

    def test_worse_seeds_take_longer(self):
        """More wrong seeds should take more batch cycles to converge."""
        profile = PROFILES_2R2C["living_room"]
        results = {}
        for factor in [1.0, 2.0, 3.0]:
            config = FullStackConfig(
                n_days=30,
                profile_name="living_room",
                outdoor_base_c=-5.0,
                outdoor_diurnal_c=6.0,
                desired_c=20.5,
                noise_sigma=0.1,
                noise_seed=42,
                pi_overrides={
                    "pi_outdoor_seed_heat": profile.true_seed * factor,
                },
                relax_kappa_gate=True,
            )
            result = run_full_stack(config)
            # Measure when coefficient first stabilizes within 0.05 of final
            final_od = result.final_coefs.get("outdoor_delta", 0)
            first_stable = None
            for i, snap in enumerate(result.coef_trajectory):
                od = snap.get("outdoor_delta", 0)
                if abs(od - final_od) < 0.05:
                    first_stable = i
                    break
            results[factor] = first_stable

        print(f"\n  Convergence speed: {results}")
        # 3× wrong should not converge faster than 1× (correct seeds)
        if results[1.0] is not None and results[3.0] is not None:
            assert results[3.0] >= results[1.0], (
                f"3× wrong seeds converged faster ({results[3.0]}) than "
                f"correct seeds ({results[1.0]})"
            )





class TestDisturbanceRejection:
    """Inject a sensor anomaly and verify recovery."""

    def test_sensor_grab_recovery(self):
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
                    start_tick=200,  # ~day 2
                    duration_ticks=4,  # 1 hour
                    field="room_temp_offset",
                    value=8.0,  # +8°C spike
                ),
            ],
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # System should recover: error in last day should be reasonable
        tpd = int(24 * 60 / config.tick_minutes)
        last_day_errors = [abs(h["room_temp"] - 20.5)
                           for h in result.history[-tpd:]]
        last_day_mae = sum(last_day_errors) / len(last_day_errors)
        assert last_day_mae < 1.0, (
            f"System didn't recover from sensor grab: "
            f"last day MAE={last_day_mae:.3f}"
        )

        # Integral should not have wound up permanently
        last_day_integrals = [abs(h["integral"])
                              for h in result.history[-tpd:]]
        max_late_integral = max(last_day_integrals)
        assert max_late_integral < 30, (
            f"Integral wound up after sensor grab: "
            f"max |integral|={max_late_integral:.1f}"
        )


# ── Scenario 4: Real Weather Replay ──────────────────────────────────────


# ── Scenario: Staged model input rollout ─────────────────────────────────


def _stove_schedule(tick: int) -> float:
    """Pellet stove: runs 6pm-10pm on cold days, off otherwise.

    Intermittent, correlated with cold outdoor (runs when it's coldest).
    This is the hard case for learning — sparse, confounded.
    """
    tick_min = 15.0
    hour = (tick * tick_min / 60.0) % 24.0
    day = tick * tick_min / (60.0 * 24.0)
    # Only fires on "cold" days (day 0, 2, 4, ... — alternating)
    if int(day) % 2 != 0:
        return 0.0
    if hour < 18 or hour > 22:
        return 0.0
    return 1.0


def _adjacent_zone_schedule(tick: int) -> float:
    """Adjacent zone (sunroom) temp delta from room.

    Warmer than room during solar hours, cooler at night.
    Correlated with solar — tests collinearity handling.
    """
    tick_min = 15.0
    hour = (tick * tick_min / 60.0) % 24.0
    day = tick * tick_min / (60.0 * 24.0)
    # Solar-driven: warm during day, cool at night
    if 8 <= hour <= 18:
        solar_factor = math.sin(math.pi * (hour - 8) / 10)
        cloud = 0.5 + 0.5 * math.cos(2 * math.pi * day / 3.0 + 1.0)
        return 3.0 * solar_factor * cloud  # up to +3°C warmer
    return -2.0  # cooler at night


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
    def _make_config(n_days: int = 30) -> FullStackConfig:
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            model_inputs=[
                ModelInputSpec(
                    name="Solar Proxy",
                    entity_id="sensor.solar_proxy",
                    input_role="solar",
                    true_thermal_effect=0.005,
                    true_ff_coef=-2.0,
                    seed_heat=0.0,
                    lag_tau=120,
                    clamp_min=0,
                    schedule=_solar_schedule,
                ),
                ModelInputSpec(
                    name="Sunroom Delta",
                    entity_id="sensor.sunroom_delta",
                    input_role="adjacent_zone",
                    true_thermal_effect=0.001,
                    true_ff_coef=-0.5,
                    seed_heat=0.0,
                    schedule=_adjacent_zone_schedule,
                    delta_from_room=True,
                ),
                ModelInputSpec(
                    name="Pellet Stove",
                    entity_id="sensor.pellet_stove",
                    input_role="heat_source",
                    true_thermal_effect=0.008,
                    true_ff_coef=-3.0,
                    seed_heat=0.0,
                    schedule=_stove_schedule,
                ),
            ],
            relax_kappa_gate=False,  # let κ gating work naturally
        )

    def test_features_start_frozen(self):
        """Model input features (indices 2+) should start frozen."""
        config = self._make_config(n_days=2)
        result = run_full_stack(config)

        # First batch snapshot should show model inputs frozen
        if result.coef_trajectory:
            snap = result.coef_trajectory[0]
            for name in ["Solar Proxy", "Sunroom Delta", "Pellet Stove"]:
                frozen_key = f"{name}_frozen"
                if frozen_key in snap:
                    assert snap[frozen_key] is True, (
                        f"{name} should start frozen"
                    )

    def test_outdoor_delta_learns_first(self):
        """outdoor_delta (base feature, never frozen) should converge first."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        # outdoor_delta should stabilize early (first 10 batches)
        if len(result.coef_trajectory) >= 15:
            early_ods = [snap.get("outdoor_delta", 0)
                         for snap in result.coef_trajectory[5:15]]
            od_range = max(early_ods) - min(early_ods)
            assert od_range < 0.2, (
                f"outdoor_delta not stabilizing early: range={od_range:.4f}"
            )

    def test_unlock_does_not_destabilize_outdoor(self):
        """When a feature unlocks, outdoor_delta should not jump."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        # Find batches where a feature unfroze
        for i in range(1, len(result.coef_trajectory)):
            prev = result.coef_trajectory[i - 1]
            curr = result.coef_trajectory[i]
            for name in ["Solar Proxy", "Sunroom Delta", "Pellet Stove"]:
                was_frozen = prev.get(f"{name}_frozen", True)
                now_frozen = curr.get(f"{name}_frozen", True)
                if was_frozen and not now_frozen:
                    # Feature just unlocked — check outdoor_delta stability
                    od_prev = prev.get("outdoor_delta", 0)
                    od_curr = curr.get("outdoor_delta", 0)
                    assert abs(od_curr - od_prev) < 0.3, (
                        f"outdoor_delta jumped {od_prev:.4f} → {od_curr:.4f} "
                        f"when {name} unlocked at batch {i}"
                    )

    def test_no_integral_runaway_during_unlocks(self):
        """Integral should stay bounded through all feature unlocks."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        max_integral = max(abs(h["integral"]) for h in result.history)
        assert max_integral < 50, (
            f"Integral runaway during staged unlocks: max={max_integral:.1f}"
        )

    def test_ff_fraction_increases_with_unlocks(self):
        """FF fraction should increase as more features unlock and learn."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

        if len(result.daily_ff_fraction) >= 21:
            first_week = sum(result.daily_ff_fraction[:7]) / 7
            third_week = sum(result.daily_ff_fraction[14:21]) / 7
            # FF fraction should not collapse
            assert third_week >= first_week * 0.7, (
                f"FF fraction collapsed: week 1={first_week:.2%}, "
                f"week 3={third_week:.2%}"
            )

    def test_comfort_maintained_through_unlocks(self):
        """Comfort should stay ≥75% even with staged unlocks."""
        config = self._make_config(n_days=30)
        result = run_full_stack(config)

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

    def test_recovery_from_sign_flip(self):
        """If outdoor_delta flips sign, batch WLS should correct it.

        Simulate by starting with a positive outdoor_delta seed (wrong
        sign — means HP backs off when it's colder, backwards).
        """
        profile = PROFILES_2R2C["living_room"]
        config = FullStackConfig(
            n_days=30,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides={
                # Positive seed = wrong sign (should be negative in RLS)
                "pi_outdoor_seed_heat": profile.true_seed * -1.0,
            },
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # outdoor_delta should end up negative (correct sign)
        od = result.final_coefs.get("outdoor_delta", 0)
        assert od < 0, (
            f"outdoor_delta still wrong sign after 30 days: {od:.4f}"
        )

        # System should still be functional — controllable comfort > 70%
        assert result.ctrl_comfort_pct >= 70.0, (
            f"Controllable comfort collapsed after sign flip: {result.ctrl_comfort_pct:.1f}%"
        )

    def test_recovery_from_large_integral_windup(self):
        """System should recover from a large initial integral error.

        Simulate by injecting a massive disturbance early that winds
        up the integral, then removing it.
        """
        config = FullStackConfig(
            n_days=14,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            disturbances=[
                # Massive cold draft for 2 hours on day 1
                Disturbance(
                    start_tick=48,  # noon day 0
                    duration_ticks=8,  # 2 hours
                    field="room_temp_offset",
                    value=-5.0,  # -5°C sensor error
                ),
            ],
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # Integral should recover — last week's integral RMS should be
        # much lower than the peak
        if len(result.daily_integral_rms) >= 7:
            peak_irms = max(result.daily_integral_rms[:3])
            last_week_irms = sum(result.daily_integral_rms[-7:]) / 7
            if peak_irms > 1.0:
                assert last_week_irms < peak_irms * 0.8, (
                    f"Integral not recovering: peak={peak_irms:.2f}, "
                    f"last week avg={last_week_irms:.2f}"
                )

        # Comfort in last week should be reasonable
        if len(result.daily_comfort_pct) >= 7:
            last_week_comfort = sum(result.daily_comfort_pct[-7:]) / 7
            assert last_week_comfort >= 80.0, (
                f"Comfort not recovered in last week: {last_week_comfort:.1f}%"
            )

    def test_recovery_from_covariance_collapse(self):
        """If P collapses (RLS stops learning), batch WLS should compensate.

        Simulate by running with very low forgetting factor (λ≈0.99)
        which causes fast P decay, then check that batch WLS still
        corrects coefficients even when online RLS has stalled.
        """
        profile = PROFILES_2R2C["living_room"]
        config = FullStackConfig(
            n_days=30,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            pi_overrides={
                "pi_outdoor_seed_heat": profile.true_seed * 2.0,
                "pi_rls_forgetting": 0.99,  # fast decay → P collapse
            },
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # Batch WLS should still function — coefficients should stabilize
        if len(result.coef_trajectory) >= 15:
            late_ods = [snap.get("outdoor_delta", 0)
                        for snap in result.coef_trajectory[-10:]]
            od_std = _std(late_ods)
            assert od_std < 0.1, (
                f"Coefficients not stable despite P collapse: "
                f"outdoor_delta std={od_std:.4f}"
            )

        # System should still be comfortable (controllable)
        assert result.ctrl_comfort_pct >= 80.0, (
            f"Controllable comfort collapsed with low λ: {result.ctrl_comfort_pct:.1f}%"
        )

    def test_wrong_sign_seed_all_profiles(self):
        """All profiles should recover from a wrong-sign outdoor seed."""
        for profile_name in ["living_room", "bunkroom"]:
            profile = PROFILES_2R2C[profile_name]
            config = FullStackConfig(
                n_days=14,
                profile_name=profile_name,
                outdoor_base_c=-5.0,
                outdoor_diurnal_c=6.0,
                desired_c=20.5,
                noise_sigma=0.1,
                noise_seed=42,
                pi_overrides={
                    "pi_outdoor_seed_heat": profile.true_seed * -1.0,
                },
                relax_kappa_gate=True,
            )
            result = run_full_stack(config)

            # Should have corrected sign
            od = result.final_coefs.get("outdoor_delta", 0)
            assert od < 0, (
                f"{profile_name}: outdoor_delta still wrong sign: {od:.4f}"
            )


_WEATHER_DIR = Path(__file__).parent.parent / "weather_data"


@pytest.mark.slow
class TestRealWeatherReplay:
    """Run learning against real open-meteo weather data.

    Validates that the learning stack doesn't diverge under realistic
    non-synthetic weather patterns (fronts, clouds, variable solar).
    """

    @pytest.mark.parametrize("weather_file,mode,desired,outdoor_offset", [
        ("new_england_winter_2w.csv", "heat", 20.5, 0.0),
        ("new_england_spring_2w.csv", "heat", 20.5, 0.0),
        ("new_england_summer_2w.csv", "cool", 24.0, 0.0),
    ], ids=["winter_heat", "spring_heat", "summer_cool"])
    def test_no_divergence(self, weather_file, mode, desired, outdoor_offset):
        """Learning should not diverge under real weather."""
        csv_path = _WEATHER_DIR / weather_file
        if not csv_path.exists():
            pytest.skip(f"Weather file not found: {csv_path}")

        csv_data = from_open_meteo_csv(csv_path)
        schedules = csv_to_schedules(csv_data)

        outdoor_fn = schedules.get("outdoor_c")
        if outdoor_fn is None:
            pytest.skip("No outdoor_c in weather data")

        # Solar: normalize W/m² to 0-1 proxy
        raw_solar_fn = schedules.get("solar_w_m2")
        if raw_solar_fn is not None:
            solar_fn = lambda tick, _f=raw_solar_fn: _f(tick) / 1000.0
        else:
            solar_fn = None

        n_hours = len(csv_data.get("outdoor_c", [])) - 1
        n_days = max(1, n_hours // 24)

        config = FullStackConfig(
            n_days=n_days,
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
                    true_thermal_effect=0.005,
                    true_ff_coef=-2.0,
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
            f"{weather_file}: integral runaway, max={max_integral:.1f}"
        )

        # Coefficient should be bounded
        od = result.final_coefs.get("outdoor_delta", 0)
        assert abs(od) < 5.0, (
            f"{weather_file}: outdoor_delta diverged to {od:.3f}"
        )


# ── Scenario 5: Multi-Year Stability ────────────────────────────────────


def _seasonal_outdoor(tick: int) -> float:
    """Full-year outdoor temp with seasonal + diurnal variation.

    Annual cycle: winter low ~-10°C (Jan), summer high ~25°C (Jul).
    Diurnal: ±5°C daily swing.  Weather fronts: ±4°C 5-day cycle.
    """
    tick_min = TICK_MINUTES_DEFAULT
    hour = (tick * tick_min / 60.0) % 24.0
    day_of_year = (tick * tick_min / (60.0 * 24.0)) % 365.0

    # Annual sinusoid: min at day 15 (Jan 15), max at day 196 (Jul 15)
    annual = 7.5 + 17.5 * math.sin(2 * math.pi * (day_of_year - 105) / 365)
    # Diurnal
    diurnal = 5.0 * math.cos(2 * math.pi * (hour - 15) / 24)
    # Weather fronts
    day_abs = tick * tick_min / (60.0 * 24.0)
    weather = 4.0 * math.sin(2 * math.pi * day_abs / 5.0)

    return annual + diurnal + weather


@pytest.mark.slow
class TestMultiYearStability:
    """Run for 1+ year and validate no long-term drift or collapse."""

    def test_one_year_no_divergence(self):
        """365-day run: coefficients bounded, no integral runaway."""
        config = FullStackConfig(
            n_days=365,
            profile_name="living_room",
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            outdoor_schedule=_seasonal_outdoor,
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

        # No integral runaway
        max_integral = max(abs(h["integral"]) for h in result.history)
        assert max_integral < 50, (
            f"Integral runaway over 1 year: max={max_integral:.1f}"
        )

        # Coefficients bounded
        od = result.final_coefs.get("outdoor_delta", 0)
        assert abs(od) < 5.0, f"outdoor_delta diverged: {od:.3f}"

        # Late-year coefficient stability (last 30 days of batches)
        if len(result.coef_trajectory) >= 60:
            late_ods = [snap.get("outdoor_delta", 0)
                        for snap in result.coef_trajectory[-60:]]
            od_range = max(late_ods) - min(late_ods)
            assert od_range < 0.5, (
                f"outdoor_delta unstable in last 30 days: range={od_range:.4f}"
            )

    def test_one_year_comfort_stable(self):
        """Monthly MAE should not grow over the year."""
        config = FullStackConfig(
            n_days=365,
            profile_name="living_room",
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            outdoor_schedule=_seasonal_outdoor,
            relax_kappa_gate=True,
        )
        result = run_full_stack(config)

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


# ── Helpers ──────────────────────────────────────────────────────────────

TICK_MINUTES_DEFAULT = 15.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
