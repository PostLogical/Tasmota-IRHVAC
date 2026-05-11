"""Boundary estimator bench validation: passive sweep + probe escalation.

Validates the full chain from boundary estimation to downstream effects:
  1. Band narrows toward true HP on/off breakpoint
  2. Observation yield increases (more definitive, fewer uncertain)
  3. Learning quality improves (FF fraction rises, integral falls)
  4. Comfort maintained or improved throughout

Physics: delta = current_c - hp_setpoint.  HP activates when
room_temp + offset < setpoint, so the true breakpoint is at
delta = -offset.  The sweep range [-5, +5] covers any offset in that
range; the initial +/-2.0 band is a starting point, not a limit.

Scenarios:
    1. Zero offset, 30d spring -- band narrows around bp=0
    2. Positive offset (+1), 30d spring -- band shifts to bp=-1
    3. Large offset (+2, +3), 60d spring -- iterative convergence
    4. Negative offset (-1), 60d spring -- band shifts to bp=+1
    5. Passive-to-probe escalation -- stall triggers active probe
    6. Parameter sensitivity sweep -- min_band, max_step, safety_margin
"""

from __future__ import annotations

import math

import pytest

from tests.hvac_bench.full_stack_runner import (
    Checkpoint,
    CheckpointState,
    FullStackConfig,
    run_full_stack,
)
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.scenarios._weather_mode import (
    SPRING_MC_STARTS,
    WeatherMode,
    get_default_weather_mode,
    real_weather_schedules,
    windowed_real_weather,
)


# ── Shared spring config ────────────────────────────────────────────────


def _spring_config(
    *,
    offset: float = 0.0,
    n_days: int = 30,
    outdoor_base_c: float = 10.0,
    outdoor_diurnal_c: float = 8.0,
    pi_overrides: dict | None = None,
    weather: WeatherMode | None = None,
    start_day: int | None = None,
    profile_name: str = "living_room",
) -> FullStackConfig:
    """Spring shoulder-season config with configurable head sensor offset.

    ``weather`` defaults to ``BENCH_WEATHER`` (``"real"`` unless
    ``--weather=synth`` was passed). Real mode swaps the synthetic
    diurnal outdoor schedule for the New England spring Open-Meteo CSV;
    the diurnal-amplitude knob is ignored under real weather, so tests
    that rely on a *narrow* diurnal swing (e.g. probe-escalation) may
    behave differently and need re-tuning under real (#49 Phase 3).
    """
    if weather is None:
        weather = get_default_weather_mode()
    outdoor_schedule = None
    if weather == "real":
        if start_day is not None:
            outdoor_fn, _, max_days = windowed_real_weather(
                start_day=start_day, n_days=n_days,
            )
        else:
            outdoor_fn, _, max_days = real_weather_schedules(
                "spring", min_days=n_days,
            )
        n_days = min(n_days, max_days)
        outdoor_schedule = outdoor_fn
    return FullStackConfig(
        n_days=n_days,
        profile_name=profile_name,
        outdoor_base_c=outdoor_base_c,
        outdoor_diurnal_c=outdoor_diurnal_c,
        outdoor_schedule=outdoor_schedule,
        desired_c=20.5,
        noise_sigma=0.1,
        noise_seed=42,
        head_sensor_offset=offset,
        head_calibration_bounds=None,  # production default +/-2.0
        relax_kappa_gate=True,
        pi_overrides=pi_overrides or {},
    )


# ── Scenario 1: Zero offset -- band narrows ─────────────────────────────


class TestZeroOffsetBandNarrowing:
    """offset=0, 30d spring.  True breakpoint at delta=0.

    With no sensor mismatch the estimator should find bp~0 and
    narrow the +/-2.0 band, improving observation yield and learning.
    """

    @pytest.mark.slow
    def test_band_narrows_and_learning_improves(self, bench_metrics, num_regression):
        config = _spring_config(offset=0.0, n_days=21)
        result = run_full_stack(config)

        bench_metrics["final_cal_min"] = result.final_cal_min
        bench_metrics["final_cal_max"] = result.final_cal_max
        bench_metrics["boundary_updates"] = result.boundary_updates
        bench_metrics["observation_yield_pct"] = result.observation_yield_pct
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct

        # -- Boundary convergence --
        band_width = result.final_cal_max - result.final_cal_min
        assert band_width < 4.0, (
            f"Band should narrow from initial 4.0, got {band_width:.2f}"
        )
        assert result.boundary_updates >= 1, (
            f"Expected at least 1 confident update, got {result.boundary_updates}"
        )
        assert (
            result.final_cal_min > -2.0 or result.final_cal_max < 2.0
        ), "At least one bound should have moved from initial +/-2.0"

        # -- Observation yield --
        assert result.observation_yield_pct > 50.0, (
            f"Narrower band should yield >50% definitive obs, "
            f"got {result.observation_yield_pct:.1f}%"
        )

        # -- Learning quality: WLS posterior covariance trace must tighten.
        # Replaces a previous FF-fraction trajectory check that was
        # confounded by the wall-clock leak fixed in #84 (pre-fix, ToD
        # nuisance regressors were stuck and β_outdoor absorbed diurnal
        # variance, inflating |FF| / (|FF|+|integral|); the trajectory's
        # nominal "growth" was leak-driven).
        #
        # Σ σ̂(β_i)² (sum of squared WLS standard errors at each batch) is
        # the trace of the WLS covariance matrix — the literature-standard
        # CRLB / information-accumulation metric (Ljung 1999 §11), computed
        # batch-side over the buffer.  As more diverse observations enter
        # the buffer, this trace decreases.  Exactly the causal claim the
        # boundary-narrowing test makes: narrower band → more usable
        # observations → tighter parameter estimates.  Multi-coefficient,
        # truth-agnostic, decoupled from FF-vs-integral controller mechanics.
        #
        # Note: ``batch_covariance_trace`` (RLS prior) is constant since
        # online RLS was removed; we use the batch-side equivalent.
        # Sum σ̂² over identifiable features only — held features (sin/cos
        # nuisance regressors during low-diversity buffer state) get inf
        # σ̂ as a sentinel.  Excluding them gives the trace of the
        # active-features-only Fisher information, which IS the right
        # CRLB on what the model is currently learning.
        traces = [
            sum(s * s for s in row if math.isfinite(s))
            for row in result.batch_std_err_trajectory
            if any(math.isfinite(s) for s in row)
        ]
        assert len(traces) >= 5, (
            f"Need at least 5 batches with std_err to assert decay; "
            f"got {len(traces)}"
        )
        # Hard floor: trace at end must be strictly less than early-run
        # baseline.  Information must accumulate; if not, learning is
        # fundamentally broken.  Use batch index 2 to skip very-small-n
        # batches where σ̂ is dominated by buffer-fill noise.
        early = traces[2]
        late = traces[-1]
        assert late < early, (
            f"WLS Σσ̂² must decrease as the buffer fills with diverse "
            f"observations (CRLB / information accumulation). Got "
            f"early[batch 2]={early:.4g}, late[final]={late:.4g}; "
            f"trajectory: {[f'{t:.3g}' for t in traces[::max(1, len(traces)//8)]]}"
        )
        # Tighter sensitivity: a 21-day run with 12h batches accumulates
        # ~40× more observations than the early-run baseline.  CRLB scaling
        # predicts Σσ̂² ∝ σ²_w / n_eff, so the ratio should drop
        # substantially.  Threshold of 0.5 is well above the theoretical
        # asymptote (~1/40) and well below "barely decreasing" (~1.0) —
        # catches subtle regressions in feature diversity or WLS
        # weighting without locking against a specific magnitude.
        assert late < early * 0.5, (
            f"WLS Σσ̂² decreased less than expected for a 21-day run "
            f"(CRLB sensitivity threshold). Got ratio "
            f"late/early={late/early:.3f}, expected <0.5. "
            f"trajectory: {[f'{t:.3g}' for t in traces[::max(1, len(traces)//8)]]}"
        )

        # -- Comfort --
        assert result.ctrl_comfort_pct >= 70.0, (
            f"Comfort should be maintained, got {result.ctrl_comfort_pct:.1f}%"
        )

        # -- No major undershoot during learning.
        # Controller-behavior assertion (boundary estimator's exploration
        # shouldn't cause deep cold dives). Run a synth-mode config so the
        # disturbance is stationary and the bound reflects what the
        # estimator did, not weather variance. See
        # ``feedback_synthetic_vs_real_bench.md``.
        synth_result = run_full_stack(_spring_config(
            offset=0.0, n_days=21, weather="synth",
        ))
        bench_metrics["synth_worst_undershoot"] = synth_result.worst_undershoot
        bench_metrics["early_trace"] = early
        bench_metrics["late_trace"] = late
        check_bench_metrics(num_regression, bench_metrics)
        assert synth_result.worst_undershoot < 2.0, (
            f"No major undershoot during learning, "
            f"got {synth_result.worst_undershoot:.2f}"
        )


# ── Scenario 2: Positive offset -- band shifts ──────────────────────────


class TestPositiveOffsetBandShift:
    """offset=+1, 30d spring.  True breakpoint at delta=-1.

    The estimator should shift the band negative to capture the
    actual HP transition, recovering observation yield.
    """

    @pytest.mark.slow
    def test_band_shifts_and_comfort_improves(self, bench_metrics, num_regression):
        config = _spring_config(offset=1.0, n_days=21)
        result = run_full_stack(config)

        bench_metrics["final_cal_min"] = result.final_cal_min
        bench_metrics["final_cal_max"] = result.final_cal_max
        bench_metrics["observation_yield_pct"] = result.observation_yield_pct
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct

        # -- Boundary convergence --
        assert result.final_cal_max < 2.0, (
            f"Upper bound should move down from 2.0, got {result.final_cal_max:.2f}"
        )
        band_center = (result.final_cal_min + result.final_cal_max) / 2
        assert band_center < 0.5, (
            f"Band center should shift toward bp=-1, got {band_center:.2f}"
        )
        band_width = result.final_cal_max - result.final_cal_min
        assert band_width < 3.0, (
            f"Band should narrow below 3.0, got {band_width:.2f}"
        )

        # -- Observation yield --
        assert result.observation_yield_pct > 40.0, (
            f"Yield should improve, got {result.observation_yield_pct:.1f}%"
        )

        # -- Learning: later MAE should be better than early.
        # Trajectory checkpoint — the 1.10 ratio bound is fragile to a
        # single non-stationary spring window (the canonical
        # SHOULDER_SPRING window was offset by 1 day specifically to land
        # on the right side of this bound). Run across three independent
        # spring starts and assert the median ratio. Bound (≤1.10) unchanged.
        ratios: list[float] = []
        for sd in SPRING_MC_STARTS:
            mc_result = run_full_stack(
                _spring_config(offset=1.0, n_days=21, start_day=sd)
            )
            n_days_actual = len(mc_result.daily_mae)
            if n_days_actual >= 14:
                first_week_mae = sum(mc_result.daily_mae[:7]) / 7
                last_week_mae = sum(mc_result.daily_mae[-7:]) / 7
                if first_week_mae > 0:
                    ratios.append(last_week_mae / first_week_mae)
        if ratios:
            ratios.sort()
            assert ratios[len(ratios) // 2] <= 1.1, (
                f"Median (last-week / first-week) MAE ratio across "
                f"{len(ratios)} spring starts should be ≤1.10, "
                f"got {[round(r, 3) for r in ratios]}"
            )

        # -- Comfort --
        assert result.ctrl_comfort_pct >= 65.0, (
            f"Comfort maintained, got {result.ctrl_comfort_pct:.1f}%"
        )
        bench_metrics["mc_median_ratio"] = (
            ratios[len(ratios) // 2] if ratios else 0.0
        )
        check_bench_metrics(num_regression, bench_metrics)


# ── Scenario 3: Large offset -- iterative convergence ────────────────────


class TestLargeOffsetConvergence:
    """offset=+2 and +3, 60d spring.  Breakpoints at -2 and -3.

    These are hard cases: the true breakpoint is at or beyond the
    initial band edge.  The estimator must iteratively shift bounds
    over multiple batch cycles.

    Known limitation: with offset=3 (bp=-3.0), passive estimation
    can't reach the true breakpoint without probes.  The initial
    band [-2, 2] misclassifies observations near the true bp, and
    the sweep finds a local minimum inside the band.  Probes
    (pi_auto_perturb_enabled) would help but aren't configured in
    this pure-passive test.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("offset", [2.0, 3.0], ids=["offset_2", "offset_3"])
    def test_iterative_convergence(self, bench_metrics, num_regression, offset: float):
        config = _spring_config(offset=offset, n_days=45)
        result = run_full_stack(config)

        bench_metrics["offset"] = offset
        bench_metrics["final_cal_min"] = result.final_cal_min
        bench_metrics["final_cal_max"] = result.final_cal_max
        bench_metrics["boundary_updates"] = result.boundary_updates
        bench_metrics["observation_yield_pct"] = result.observation_yield_pct
        if result.daily_integral_rms:
            bench_metrics["max_daily_integral_rms"] = max(result.daily_integral_rms)
        check_bench_metrics(num_regression, bench_metrics)

        # -- Boundary convergence --
        assert result.boundary_updates >= 3, (
            f"offset={offset}: expected >=3 iterative updates, "
            f"got {result.boundary_updates}"
        )
        assert result.final_cal_max < 0.5, (
            f"offset={offset}: cal_max should move far from 2.0, "
            f"got {result.final_cal_max:.2f}"
        )
        band_width = result.final_cal_max - result.final_cal_min
        assert band_width < 3.0, (
            f"offset={offset}: band should narrow, got {band_width:.2f}"
        )

        # For offset=3, both bounds should be on the negative side.
        # Passive-only can't reach bp=-3 without probe evidence.
        if offset >= 3.0:
            if result.final_cal_max >= 0.0:
                pytest.xfail(
                    f"Bootstrap problem: offset={offset} band "
                    f"[{result.final_cal_min:.2f}, {result.final_cal_max:.2f}] "
                    f"— passive sweep can't reach bp=-3 without probes"
                )

        # -- Observation yield improved from initial --
        assert result.observation_yield_pct > 50.0, (
            f"offset={offset}: yield should improve, "
            f"got {result.observation_yield_pct:.1f}%"
        )

        # -- Stability: integral should not diverge --
        if result.daily_integral_rms:
            max_integral = max(result.daily_integral_rms)
            assert max_integral < 10.0, (
                f"offset={offset}: integral should stay bounded, "
                f"max daily RMS={max_integral:.2f}"
            )


# ── Scenario 4: Negative offset -- positive breakpoint ──────────────────


class TestNegativeOffsetConvergence:
    """offset=-1, 60d spring.  True breakpoint at delta=+1.

    Mirror of scenario 2 but with positive breakpoint.  The HP runs
    longer than expected (thinks room is cooler), so the band must
    shift positive.

    Known limitation: with offset=-1, the HP is on for 93% of
    observations (delta < 1.0 almost always).  The sweep can't
    find the true bp=+1 because there are too few HP-off
    observations to create RMS contrast there.  The estimator
    converges confidently to bp~-0.5 where the observation
    balance is better — wrong answer, high confidence.  This is
    exactly the case where the active probe is needed: generate
    deliberate HP-off data near the true transition.
    """

    @pytest.mark.slow
    def test_band_shifts_positive(self, bench_metrics, num_regression):
        config = _spring_config(offset=-1.0, n_days=30)
        result = run_full_stack(config)

        bench_metrics["final_cal_min"] = result.final_cal_min
        bench_metrics["final_cal_max"] = result.final_cal_max
        bench_metrics["boundary_updates"] = result.boundary_updates
        bench_metrics["observation_yield_pct"] = result.observation_yield_pct
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)

        # -- Boundary convergence --
        band_center = (result.final_cal_min + result.final_cal_max) / 2
        band_width = result.final_cal_max - result.final_cal_min

        # The band narrows (the estimator IS confident, just about
        # the wrong breakpoint).  Band should be tight.
        assert band_width < 2.0, (
            f"Band should narrow, got {band_width:.2f}"
        )
        assert result.boundary_updates >= 1, (
            f"Expected at least 1 update, got {result.boundary_updates}"
        )

        # The center should shift positive toward bp=+1.
        # Currently fails: converges to center~-0.5 due to data
        # asymmetry (HP almost always on).
        if band_center <= 0.0:
            pytest.xfail(
                f"Data asymmetry: offset=-1 gives 93% HP-on observations. "
                f"Sweep converges to bp~-0.5 (center={band_center:.2f}) "
                f"instead of true bp=+1.0 — needs probe for HP-off evidence"
            )

        # -- Observation yield (this passes even with wrong bp) --
        assert result.observation_yield_pct > 80.0, (
            f"Tight band should give high yield, "
            f"got {result.observation_yield_pct:.1f}%"
        )

        # -- Comfort --
        assert result.ctrl_comfort_pct >= 65.0, (
            f"Comfort maintained, got {result.ctrl_comfort_pct:.1f}%"
        )


# ── Scenario 5: Passive → probe escalation ──────────────────────────────


class TestPassiveToProbeEscalation:
    """Probe fires in ambiguous conditions and completes without
    wrecking comfort.

    Narrow diurnal swing + cooler base = deltas cluster near the
    breakpoint.  With auto_perturb enabled, the probe fires when
    the HP is in the uncertain zone and generates definitive evidence
    about HP contribution.

    Historical note: before the is_clamped fix, the probe's own
    force_min_setpoint caused is_clamped=True on the next tick,
    self-aborting every probe attempt.
    """

    @pytest.mark.slow
    def test_probe_completes(self, bench_metrics, num_regression):
        config = _spring_config(
            offset=0.5,
            n_days=30,
            outdoor_base_c=5.0,
            outdoor_diurnal_c=3.0,
            pi_overrides={"pi_auto_perturb_enabled": True},
        )

        probes_completed = [0]

        def _track_probe(state: CheckpointState) -> None:
            probes_completed[0] = state.pi._regime_probe._probes_completed

        result = run_full_stack(
            config,
            checkpoints=[Checkpoint(interval_days=10, callback=_track_probe)],
        )

        bench_metrics["probes_completed"] = probes_completed[0]
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        bench_metrics["observation_yield_pct"] = result.observation_yield_pct
        check_bench_metrics(num_regression, bench_metrics)

        # -- Probes completed --
        assert probes_completed[0] >= 1, (
            f"Active probe should complete at least once, "
            f"probes_completed = {probes_completed[0]}"
        )

        # -- Comfort not wrecked by probe interventions --
        assert result.ctrl_comfort_pct >= 60.0, (
            f"Comfort should survive probes, got {result.ctrl_comfort_pct:.1f}%"
        )


# ── Scenario 6: Parameter sensitivity ───────────────────────────────────


class TestBoundaryParameterSensitivity:
    """Sweep boundary estimator parameters to verify behavior.

    Uses a day-1 checkpoint to override BoundaryEstimator attributes
    on the live PIController instance.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "param_name,param_value",
        [
            ("_min_band", 0.3),
            ("_min_band", 1.0),
            ("_max_step", 0.1),
            ("_max_step", 0.5),
            ("_safety_margin", 0.1),
            ("_safety_margin", 0.5),
        ],
        ids=[
            "tight_band_0.3",
            "wide_band_1.0",
            "slow_step_0.1",
            "fast_step_0.5",
            "tight_margin_0.1",
            "wide_margin_0.5",
        ],
    )
    def test_parameter_effect(self, bench_metrics, num_regression, param_name: str, param_value: float):
        config = _spring_config(offset=0.0, n_days=14)

        applied = [False]

        def _apply_override(state: CheckpointState) -> None:
            if not applied[0]:
                setattr(state.pi._boundary_estimator, param_name, param_value)
                applied[0] = True

        result = run_full_stack(
            config,
            checkpoints=[Checkpoint(interval_days=1, callback=_apply_override)],
        )

        # -- Band narrowed from initial --
        band_width = result.final_cal_max - result.final_cal_min

        bench_metrics["param_value"] = param_value
        bench_metrics["band_width"] = band_width
        bench_metrics["final_cal_min"] = result.final_cal_min
        bench_metrics["final_cal_max"] = result.final_cal_max
        bench_metrics["boundary_updates"] = result.boundary_updates
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)

        assert band_width < 4.0, (
            f"{param_name}={param_value}: band should narrow, got {band_width:.2f}"
        )

        # -- Floor respected --
        if param_name == "_min_band":
            assert band_width >= param_value - 0.01, (
                f"Band width {band_width:.2f} should respect "
                f"min_band_width={param_value}"
            )

        # -- Learning occurred --
        assert result.boundary_updates >= 1, (
            f"{param_name}={param_value}: expected updates, "
            f"got {result.boundary_updates}"
        )

        # -- Comfort --
        assert result.ctrl_comfort_pct >= 70.0, (
            f"{param_name}={param_value}: comfort should be maintained, "
            f"got {result.ctrl_comfort_pct:.1f}%"
        )
