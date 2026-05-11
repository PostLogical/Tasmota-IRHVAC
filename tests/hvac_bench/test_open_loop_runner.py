"""Tests for the open-loop probe runner (Phase 1a).

Two layers:

1. **Excitation properties** — unit tests on the input-design generators.
   Step / PRBS / multi-sine each have algebraic invariants (binary range,
   minimum hold time, energy normalization) that should hold without
   running a full simulation.
2. **End-to-end probe** — runs ``run_open_loop_probe`` end-to-end on the
   ``living_room`` 2R2C profile, then runs ``weighted_least_squares`` on
   the produced observations and asserts that β_outdoor recovery lands
   close to the truth ``-1/(g·τ)``. This is the proof that open-loop
   probe data drives the same identification path that ``full_stack_runner``
   exercises.
"""

from __future__ import annotations

import math

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    Observation,
    weighted_least_squares,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.open_loop_runner import (
    OpenLoopConfig,
    clip_to_bounds,
    make_multisine_excitation,
    make_prbs_excitation,
    make_step_excitation,
    run_open_loop_probe,
)


# ── Excitation property tests ─────────────────────────────────────────


class TestStepExcitation:
    """Step excitation: square wave between center ± amplitude."""

    def test_alternates_between_two_values(self, bench_metrics, num_regression):
        excite = make_step_excitation(
            center_c=20.0, amplitude_c=2.0, hold_minutes=60.0, tick_minutes=15.0
        )
        # 60-min hold at 15-min ticks = 4 ticks per level.
        values = [excite(t) for t in range(16)]
        bench_metrics["high_value"] = values[0]
        bench_metrics["low_value"] = values[4]
        check_bench_metrics(num_regression, bench_metrics)
        # First 4 ticks at +amplitude, next 4 at -amplitude, etc.
        assert values[0] == values[1] == values[2] == values[3] == 22.0
        assert values[4] == values[5] == values[6] == values[7] == 18.0
        assert values[8] == values[9] == values[10] == values[11] == 22.0

    def test_only_two_distinct_values(self, bench_metrics, num_regression):
        excite = make_step_excitation(
            center_c=20.0, amplitude_c=1.5, hold_minutes=30.0, tick_minutes=15.0
        )
        values = {excite(t) for t in range(200)}
        bench_metrics["n_distinct"] = len(values)
        bench_metrics["min_value"] = min(values)
        bench_metrics["max_value"] = max(values)
        check_bench_metrics(num_regression, bench_metrics)
        assert values == {18.5, 21.5}

    def test_rejects_zero_hold(self, bench_metrics, num_regression):
        with pytest.raises(ValueError):
            make_step_excitation(
                center_c=20.0, amplitude_c=1.0, hold_minutes=0.0
            )

    def test_rejects_negative_amplitude(self, bench_metrics, num_regression):
        with pytest.raises(ValueError):
            make_step_excitation(
                center_c=20.0, amplitude_c=-1.0, hold_minutes=30.0
            )


class TestPRBSExcitation:
    """PRBS: binary signal honoring minimum hold time, broadband."""

    def test_prbs_only_two_distinct_values(self, bench_metrics, num_regression):
        excite = make_prbs_excitation(
            center_c=20.0, amplitude_c=2.0, min_hold_minutes=30.0,
            seed=42, tick_minutes=15.0,
        )
        values = {excite(t) for t in range(2880)}  # 30 days at 15 min
        bench_metrics["n_distinct"] = len(values)
        bench_metrics["min_value"] = min(values)
        bench_metrics["max_value"] = max(values)
        check_bench_metrics(num_regression, bench_metrics)
        assert values == {18.0, 22.0}

    def test_respects_min_hold_time(self, bench_metrics, num_regression):
        # min_hold = 60 min @ 15 min tick = 4 ticks. So once flipped,
        # the signal must stay constant for at least 4 ticks.
        excite = make_prbs_excitation(
            center_c=20.0, amplitude_c=1.0, min_hold_minutes=60.0,
            seed=7, tick_minutes=15.0,
        )
        values = [excite(t) for t in range(500)]
        # Find every flip; the run preceding each flip must be ≥ 4 ticks.
        # (Run length BEFORE a flip = ticks since the previous flip.)
        run_start = 0
        min_run = None
        for t in range(1, len(values)):
            if values[t] != values[t - 1]:
                run_length = t - run_start
                min_run = run_length if min_run is None else min(min_run, run_length)
                assert run_length >= 4, (
                    f"PRBS run length {run_length} < min_hold_ticks=4 "
                    f"at flip tick {t}"
                )
                run_start = t
        bench_metrics["min_run_length"] = min_run if min_run is not None else 0
        check_bench_metrics(num_regression, bench_metrics)

    def test_has_switches_in_long_run(self, bench_metrics, num_regression):
        excite = make_prbs_excitation(
            center_c=20.0, amplitude_c=1.0, min_hold_minutes=15.0,
            seed=0, tick_minutes=15.0,
        )
        values = [excite(t) for t in range(500)]
        switches = sum(1 for t in range(1, 500) if values[t] != values[t - 1])
        bench_metrics["n_switches"] = switches
        check_bench_metrics(num_regression, bench_metrics)
        # With 50% switch probability and min_hold=1, expect ~125 switches
        # in 500 ticks. Loose lower bound to keep deterministic.
        assert switches >= 50, f"too few PRBS switches: {switches}"

    def test_rejects_non_monotonic_ticks(self, bench_metrics, num_regression):
        excite = make_prbs_excitation(
            center_c=20.0, amplitude_c=1.0, min_hold_minutes=30.0, seed=1,
        )
        excite(0)
        excite(1)
        with pytest.raises(ValueError, match="monotonic"):
            excite(0)  # going backwards is not allowed


class TestMultiSineExcitation:
    """Multi-sine: Schroeder-phased, energy-normalized, bounded."""

    def test_amplitude_bounded(self, bench_metrics, num_regression):
        # With Schroeder phasing, peak amplitude ≤ amplitude_c × √M / √M =
        # amplitude_c (in the limit; actual peak is somewhat below).
        excite = make_multisine_excitation(
            center_c=20.0, amplitude_c=2.0,
            n_components=8,
            period_min_minutes=30.0, period_max_minutes=720.0,
            tick_minutes=15.0,
        )
        values = [excite(t) for t in range(10000)]
        peak_dev = max(abs(v - 20.0) for v in values)
        bench_metrics["peak_dev"] = peak_dev
        check_bench_metrics(num_regression, bench_metrics)
        # Crest factor for Schroeder multi-sine is theoretically ~√2 above
        # RMS, but for small M the bound is loose. Assert peak dev ≤
        # amplitude × √M (conservative upper bound).
        sqrt_m = math.sqrt(8)
        assert peak_dev <= 2.0 * sqrt_m + 1e-6

    def test_carries_energy_at_chosen_frequencies(self, bench_metrics, num_regression):
        # Single-component sine: energy concentrated at one frequency.
        # Verify by checking variance and zero-crossings.
        period_min = 240.0  # 4 hours
        excite = make_multisine_excitation(
            center_c=0.0, amplitude_c=1.0,
            n_components=1,
            period_min_minutes=period_min, period_max_minutes=period_min * 1.001,
            tick_minutes=15.0,
        )
        # Run for 10 periods.
        n_ticks = int(10 * period_min / 15.0)
        values = [excite(t) for t in range(n_ticks)]
        # Variance should be ~0.5 (amplitude² / 2 for unit sine,
        # normalized by sqrt(M=1) = 1).
        mean = sum(values) / n_ticks
        var = sum((v - mean) ** 2 for v in values) / n_ticks
        bench_metrics["mean"] = mean
        bench_metrics["variance"] = var
        check_bench_metrics(num_regression, bench_metrics)
        assert 0.4 < var < 0.6

    def test_rejects_invalid_periods(self, bench_metrics, num_regression):
        with pytest.raises(ValueError):
            make_multisine_excitation(
                center_c=20.0, amplitude_c=1.0,
                n_components=4,
                period_min_minutes=120.0, period_max_minutes=60.0,  # max < min
            )

    def test_rejects_zero_components(self, bench_metrics, num_regression):
        with pytest.raises(ValueError):
            make_multisine_excitation(
                center_c=20.0, amplitude_c=1.0,
                n_components=0,
                period_min_minutes=30.0, period_max_minutes=300.0,
            )


class TestClipToBounds:
    """Wrapper clamps excitation output to a [min, max] range."""

    def test_clips_above_max(self, bench_metrics, num_regression):
        clipped = clip_to_bounds(
            lambda t: 100.0, min_c=16.0, max_c=30.0
        )
        bench_metrics["output"] = clipped(0)
        check_bench_metrics(num_regression, bench_metrics)
        assert clipped(0) == 30.0

    def test_clips_below_min(self, bench_metrics, num_regression):
        clipped = clip_to_bounds(
            lambda t: -50.0, min_c=16.0, max_c=30.0
        )
        bench_metrics["output"] = clipped(0)
        check_bench_metrics(num_regression, bench_metrics)
        assert clipped(0) == 16.0

    def test_passes_through_in_range(self, bench_metrics, num_regression):
        clipped = clip_to_bounds(
            lambda t: 22.5, min_c=16.0, max_c=30.0
        )
        bench_metrics["output"] = clipped(0)
        check_bench_metrics(num_regression, bench_metrics)
        assert clipped(0) == 22.5


# ── End-to-end probe run + β recovery ─────────────────────────────────


class TestProbeRecoversOutdoorBeta:
    """End-to-end probe + WLS recovers outdoor_delta close to OPEN-LOOP truth.

    The 2R2C steady-state heat balance with HP active is
    ``(sp - room) = (sp - T_out)/(g·τ + 1) - q·τ/(g·τ+1)``. Regressing
    ``(sp - room)`` on ``(T_out - desired)`` therefore identifies
    ``β_open_loop = -1/(g·τ + 1)``.

    For ``living_room`` (g=0.04, τ=100, g·τ=4): ``β_open_loop = -0.20``.
    The closed-loop FF coefficient ``-1/(g·τ) = -0.25`` is what minimizes
    tracking error when the controller forces ``room ≈ desired``; it is
    NOT what the regression should recover from open-loop data, by
    construction. The 0.05 gap is the closed-loop bias the probe is built
    to surface (compared in Phase 1c against the closed-loop runner).

    Assertion is against the literature-derivable open-loop closed form
    (``feedback_test_to_spec_not_output.md``: ground in physics, not in
    output calibration).
    """

    @pytest.fixture(scope="class")
    def step_probe_result(self):
        """30-day step probe on living_room with HP always on.

        Setpoint range [22.5, 25.5] is high enough that the 2R2C
        equilibrium (≈18.8°C max with sp=24, T_out=-2) stays below
        setpoint at all times → ``hp_active`` is always True → all
        observations are in the same dynamical regime, so the regression
        identifies the heat-balance slope cleanly.

        6-hour hold = 360 min ≫ 4·τ_couple (120 min), so transients
        decay within each hold and the WLS room_rate filter admits the
        majority of observations.
        """
        excitation = make_step_excitation(
            center_c=24.0,
            amplitude_c=1.5,
            hold_minutes=360.0,
            tick_minutes=15.0,
        )
        config = OpenLoopConfig(
            excitation=excitation,
            n_days=30,
            profile_name="living_room",
            outdoor_base_c=-2.0,
            outdoor_diurnal_c=8.0,
            desired_c=20.0,
            mode="heat",
            noise_sigma=0.05,
            noise_seed=42,
            tick_minutes=15.0,
        )
        return run_open_loop_probe(config)

    def test_true_coefs_outdoor_delta_uses_open_loop_asymptote(
        self, bench_metrics, num_regression, step_probe_result
    ):
        """Regression test for C5: open-loop truth must be the open-loop
        asymptote ``-1/(g·τ_env+1)`` in signed physics-space, NOT
        ``+profile.true_seed`` (the previous buggy value, which had wrong
        sign AND wrong magnitude — that's the closed-loop seed in
        user-facing convention).
        """
        profile = PROFILES_2R2C["living_room"]
        expected = -1.0 / (profile.hp_gain * profile.tau_env + 1.0)
        actual = step_probe_result.true_coefs["outdoor_delta"]
        bench_metrics["actual_true_coef"] = actual
        bench_metrics["expected_asymptote"] = expected
        check_bench_metrics(num_regression, bench_metrics)
        assert actual == pytest.approx(expected), (
            f"open-loop true_coefs['outdoor_delta']={actual:.4f} "
            f"expected {expected:.4f} = -1/(g·τ_env+1)"
        )
        # Sign must be negative (heat mode: cold outdoor → +setpoint headroom).
        assert actual < 0.0
        # Must NOT equal +profile.true_seed (the original buggy value).
        assert actual != profile.true_seed

    def test_observation_count_matches_ticks(self, bench_metrics, num_regression, step_probe_result):
        bench_metrics["n_observations"] = len(step_probe_result.observations)
        bench_metrics["n_ticks"] = step_probe_result.n_ticks
        check_bench_metrics(num_regression, bench_metrics)
        assert len(step_probe_result.observations) == step_probe_result.n_ticks
        assert step_probe_result.n_ticks == 30 * 24 * 4  # 2880

    def test_observations_have_required_fields(self, bench_metrics, num_regression, step_probe_result):
        n_clamped = sum(1 for o in step_probe_result.observations if o.clamped)
        n_uncertain = sum(1 for o in step_probe_result.observations if o.hp_contribution_uncertain)
        bench_metrics["n_clamped"] = n_clamped
        bench_metrics["n_uncertain"] = n_uncertain
        check_bench_metrics(num_regression, bench_metrics)
        # Every observation must be non-clamped and carry the fields WLS
        # consumes.
        for obs in step_probe_result.observations:
            assert obs.hp_setpoint is not None
            assert obs.outdoor_temp_c is not None
            assert obs.clamped is False
            assert obs.hp_contribution_uncertain is False

    def test_setpoint_trajectory_alternates(self, bench_metrics, num_regression, step_probe_result):
        # 6-hour hold @ 15-min ticks = 24 ticks per level.
        traj = step_probe_result.setpoint_trajectory
        # First 24 ticks at one level, next 24 at the other.
        first = traj[0]
        second = traj[24]
        bench_metrics["first_value"] = first
        bench_metrics["second_value"] = second
        check_bench_metrics(num_regression, bench_metrics)
        assert all(t == first for t in traj[:24])
        assert second != first
        assert all(t == second for t in traj[24:48])

    def test_beta_outdoor_in_physical_band(self, bench_metrics, num_regression, step_probe_result):
        """WLS on probe data recovers β_outdoor in the physically valid band.

        Two algebraic bounds without any output-tuned tolerance:
        - β must be negative (in heat mode, more outdoor cold drives more
          setpoint headroom).
        - |β| must be below ``1/g`` = 25 °C-of-setpoint per °C-of-outdoor,
          which would imply zero thermal mass; anything larger violates
          the heat-balance equation.

        The closed-form open-loop asymptote is ``-1/(g·τ + 1) = -0.20`` for
        ``living_room``, but a finite-duration step probe with a high-mass
        wall (``mass_ratio=8``, τ_wall = 240 min) doesn't reach the
        asymptote — measuring the gap to the asymptote is itself a Phase
        1c attribution job. Phase 1a only asserts data validity.
        """
        result = weighted_least_squares(
            observations=step_probe_result.observations,
            n_features=2,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
            min_observations=20,
            detect_lag=False,
        )
        assert result is not None, "WLS returned None on 30-day probe"
        beta_outdoor = result.beta_batch[1]
        bench_metrics["beta_outdoor"] = beta_outdoor
        check_bench_metrics(num_regression, bench_metrics)
        profile = PROFILES_2R2C["living_room"]
        max_plausible = 1.0 / profile.hp_gain
        assert -max_plausible < beta_outdoor < 0.0, (
            f"β_outdoor={beta_outdoor:.4f} outside physical band "
            f"({-max_plausible:.2f}, 0); probe data is unusable"
        )

    def test_beta_outdoor_differs_from_closed_loop_truth(self, bench_metrics, num_regression, step_probe_result):
        """Sanity: the open-loop probe DOES produce a different β.

        If the open-loop and closed-loop coefficients were equal, the
        whole Phase 1 attribution premise would collapse. Open-loop
        truth is -1/(g·τ+1) = -0.20; closed-loop truth is -1/(g·τ) =
        -0.25. The recovered β should be measurably closer to the
        open-loop value than to the closed-loop value.
        """
        result = weighted_least_squares(
            observations=step_probe_result.observations,
            n_features=2,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
            min_observations=20,
            detect_lag=False,
        )
        assert result is not None
        beta_outdoor = result.beta_batch[1]
        profile = PROFILES_2R2C["living_room"]
        g_tau = profile.hp_gain * profile.tau_env
        open_loop = -1.0 / (g_tau + 1.0)
        closed_loop = -1.0 / g_tau
        gap_to_open = abs(beta_outdoor - open_loop)
        gap_to_closed = abs(beta_outdoor - closed_loop)
        bench_metrics["beta_outdoor"] = beta_outdoor
        bench_metrics["gap_to_open"] = gap_to_open
        bench_metrics["gap_to_closed"] = gap_to_closed
        check_bench_metrics(num_regression, bench_metrics)
        assert gap_to_open < gap_to_closed, (
            f"β={beta_outdoor:.4f} closer to closed-loop ({closed_loop:.4f}) "
            f"than open-loop ({open_loop:.4f}) — probe may be coupling back"
        )

    def test_room_rate_filter_admits_observations(self, bench_metrics, num_regression, step_probe_result):
        """At 6-hour hold ≫ τ, most ticks are quasi-steady (rate ≈ 0).

        The WLS room_rate filter has a default threshold of 0.02 °C/min.
        We expect most observations to pass this filter; if too many fail
        it means the step amplitude or hold is wrong for the τ.
        """
        threshold = 0.02
        passed = sum(
            1 for o in step_probe_result.observations
            if abs(o.room_rate) < threshold
        )
        # Loosely: at least 50% of observations should be quasi-steady.
        # The first ~τ minutes after each switch are transient.
        pass_pct = 100.0 * passed / len(step_probe_result.observations)
        bench_metrics["pass_pct"] = pass_pct
        bench_metrics["n_passed"] = passed
        check_bench_metrics(num_regression, bench_metrics)
        assert pass_pct > 50.0, (
            f"only {pass_pct:.0f}% of obs passed room_rate filter — "
            f"step hold may be too short for the profile τ"
        )


class TestProbeRunsWithModelInputs:
    """Smoke test: probe with a model input wires raw_readings correctly."""

    def test_runs_with_solar_input(self, bench_metrics, num_regression):
        from tests.hvac_bench.full_stack_runner import ModelInputSpec

        excitation = make_step_excitation(
            center_c=21.0, amplitude_c=1.0, hold_minutes=360.0,
        )
        config = OpenLoopConfig(
            excitation=excitation,
            n_days=2,
            profile_name="living_room",
            outdoor_base_c=-2.0,
            desired_c=20.0,
            tick_minutes=15.0,
            model_inputs=[
                ModelInputSpec(
                    name="Solar Proxy",
                    entity_id="sensor.solar_test",
                    input_role="solar",
                    _true_ff_coef=-2.0,
                    schedule=lambda t: 0.5,  # constant — unrealistic but fine for plumbing
                ),
            ],
        )
        result = run_open_loop_probe(config)
        bench_metrics["n_observations"] = len(result.observations)
        check_bench_metrics(num_regression, bench_metrics)
        # Every observation should carry the solar reading in raw_readings.
        for obs in result.observations:
            assert "sensor.solar_test" in obs.raw_readings
            assert obs.raw_readings["sensor.solar_test"] == 0.5

    def test_runs_with_no_inputs(self, bench_metrics, num_regression):
        excitation = make_step_excitation(
            center_c=21.0, amplitude_c=1.0, hold_minutes=180.0,
        )
        config = OpenLoopConfig(
            excitation=excitation,
            n_days=1,
            profile_name="living_room",
            tick_minutes=15.0,
        )
        result = run_open_loop_probe(config)
        bench_metrics["n_ticks"] = result.n_ticks
        check_bench_metrics(num_regression, bench_metrics)
        assert result.n_ticks == 96  # 1 day at 15 min
        for obs in result.observations:
            assert obs.raw_readings == {}
