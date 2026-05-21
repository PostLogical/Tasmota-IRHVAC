"""System learning + performance report — holistic, pinned observability.

Purpose (future_work #113, generalized): instead of testing one theory
(qref endogeneity, excitation, lag-τ, …) in isolation, run the system *as
configured* (production defaults) across a few real-weather scenarios and
pin a comprehensive learning + performance snapshot via pytest-regressions.

Then any code change can be run against this test, and the regression diff
shows exactly how the system's behaviour moved — coefficient identification,
detected lag-τ, learning progression, comfort, and chatter — relative to
truth, without having to trust the particular code under test.

Scenarios: the four real-weather heating seasons (winter / fall / spring /
spring_plus_60) plus a summer cooling scenario (mode=cool, 24°C setpoint).
Together they span the SNR / excitation range and both HP modes.

Per scenario, pinned per cell:

  Learning (per coefficient with a known truth):
    {coef}_truth                    physics-space truth (from result.true_coefs)
    {coef}_final / _err_final       live applied coef at end, |Δ truth|
    {coef}_at_fill / _err_at_fill   live applied coef at buffer-fill day
    {coef}_at_unlock / _err_…       live applied coef the batch it unfroze
    {coef}_unlock_day               day the feature unfroze (NaN = never)
    {coef}_traj_dNN                 progression at days 5/15/30/45/60/75/90
    {coef}_post_fill_bias/std/drift summarize_post_fill (Belsley/seasonal-tol)
    {coef}_tau_at_fill/at_unlock/final   detected EMA lag-τ at each stage (s)

  Convergence / buffer:
    fill_day, final_util, batches_to_converge

  Performance (chatter == HP beeps; comfort band = ±1°F per the 1°F-step UI):
    comfort_pct_1f          fraction of ticks within ±1°F of user setpoint
    ctrl_comfort_pct_1f     same, restricted to HP-active ticks
    hp_changes_per_day      HP integer-setpoint transitions / day (beep proxy)
    total_itae, ctrl_violations, worst_undershoot, worst_overshoot

Only loose, mode-agnostic non-divergence asserts gate pass/fail; the value
of the test is the pinned snapshot, not hand-tuned thresholds (see feedback:
test to spec, not output).  When the reported numbers settle, the
`_learning_report` extractor should move to full_stack_runner alongside
summarize_post_fill.
"""
from __future__ import annotations

import math

import pytest

from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    FullStackResult,
    ModelInputSpec,
    run_full_stack,
    summarize_post_fill,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.scenarios._weather_mode import windowed_real_weather
from tests.hvac_bench.scenarios.test_seasonal_convergence import _make_config

# Summer cooling window: June 1 → end of August (real CSV starts 2023-01-01;
# 2025-06-01 is day 882, the cooling season proper).  Kept to Jun–Aug so the
# run doesn't drift into cold autumn nights where cool-mode can't act.
_SUMMER_START_DAY = 882
_SUMMER_N_DAYS = 92

_TICK_MINUTES = 5.0
_N_DAYS = 90
_BAND_1F_C = 1.0 * 5.0 / 9.0  # ±1°F in °C ≈ 0.5556
_BATCHES_PER_DAY = 2.0  # batch WLS cadence is 12h
_TRAJ_DAYS = (5, 15, 30, 45, 60, 75, 90)

# Post-fill tolerances reused from the seasonal convergence / buffer-variant
# tests (outdoor_delta tight, solar loose — Belsley + seasonal-convergence pins).
_POST_FILL_BIAS_TOL = {"outdoor_delta": 0.05, "Solar Proxy": 0.20}
_POST_FILL_STD_TOL = {"outdoor_delta": 0.05, "Solar Proxy": 0.20}

# Scenario set: four heating seasons (real-CSV defaults) + one summer cooling.
# Heating seasons resolve through test_seasonal_convergence._make_config; the
# cooling scenario is built locally because that helper is heating-only.
_HEATING_SEASONS = ("winter", "fall", "spring", "spring_plus_60")
_SUMMER_COOL = "summer_cool"
_SCENARIOS = (*_HEATING_SEASONS, _SUMMER_COOL)

_NAN = float("nan")


def _make_summer_cool_config(n_days: int) -> FullStackConfig:
    """Real-CSV summer window in cooling mode (67°F setpoint).

    Mirrors the heating ``_make_real_config`` (same building, σ, seed, 2×-wrong
    outdoor seed) but flips mode→cool, setpoint→67°F, and seeds the cool model.
    Setpoint is expressed on the °F grid (the user's HP quantizes in °F; the
    heating scenarios' 20.5°C is likewise 69°F).  A low 67°F (not 21/24°C): at
    this northern latitude the room sits below a milder setpoint most of the
    time, so cooling rarely engages and the learner is excitation-starved (21°C
    never identified solar).  67°F keeps cooling demand sustained enough to
    actually exercise cool-mode learning.
    Solar drives the 2R2C model through the Solar Proxy input schedule (same as
    the heating path); ``_true_ff_coef`` stays −2.0 (more sun → more cooling
    demand → lower setpoint → negative coefficient, same sign as heating).
    Window is fixed to Jun–Aug (ignoring ``n_days``) so the run doesn't drift
    into cold autumn nights the cooler can't act on.
    """
    outdoor_fn, solar_fn, served_days = windowed_real_weather(
        start_day=_SUMMER_START_DAY, n_days=_SUMMER_N_DAYS,
    )
    profile = PROFILES_2R2C["living_room"]
    return FullStackConfig(
        n_days=served_days,
        profile_name="living_room",
        desired_c=(67.0 - 32.0) * 5.0 / 9.0,  # 67°F ≈ 19.44°C (user's grid is °F)
        mode="cool",
        noise_sigma=0.1,
        noise_seed=42,
        outdoor_schedule=outdoor_fn,
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
        ],
        pi_overrides={"pi_outdoor_seed_cool": profile.true_seed * 2.0},
        relax_kappa_gate=True,
    )


def _make_scenario(name: str, n_days: int) -> FullStackConfig:
    if name == _SUMMER_COOL:
        return _make_summer_cool_config(n_days)
    return _make_config(name, n_days=n_days)


def _at(traj: list[dict], idx: int | None, key: str) -> float:
    if not traj or idx is None:
        return _NAN
    idx = max(0, min(idx, len(traj) - 1))
    v = traj[idx].get(key, _NAN)
    return float(v) if v is not None else _NAN


def _fill_day(result: FullStackResult) -> int | None:
    return next(
        (i for i, u in enumerate(result.daily_buffer_utilization) if u >= 0.999),
        None,
    )


def _unlock_batch(traj: list[dict], coef: str) -> int | None:
    """First batch index where the coefficient is no longer frozen."""
    for i, snap in enumerate(traj):
        if snap.get(f"{coef}_frozen", True) is False:
            return i
    return None


def _learning_report(result: FullStackResult, desired_c: float) -> dict[str, float]:
    """Flatten a FullStackResult into the pinned learning + perf snapshot."""
    traj = result.coef_trajectory
    truths = result.true_coefs
    fill_day = _fill_day(result)
    fill_idx = int(fill_day * _BATCHES_PER_DAY) if fill_day is not None else None
    post_fill = summarize_post_fill(
        result, bias_tols=_POST_FILL_BIAS_TOL, std_tols=_POST_FILL_STD_TOL
    )

    m: dict[str, float] = {}
    m["fill_day"] = float(fill_day) if fill_day is not None else _NAN
    m["final_util"] = (
        result.daily_buffer_utilization[-1]
        if result.daily_buffer_utilization else _NAN
    )
    m["batches_to_converge"] = (
        float(result.batches_to_converge)
        if result.batches_to_converge is not None else _NAN
    )

    for coef, truth in truths.items():
        key = coef.replace(" ", "_")
        final = result.final_coefs.get(coef, _NAN)
        at_fill = _at(traj, fill_idx, coef)
        ub = _unlock_batch(traj, coef)
        # The unlock flag flips at batch ub, but the coefficient is still HELD
        # at its seed there: _evaluate_feature_unlocks runs *after* the blend is
        # applied, with the WLS for that batch computed under the still-frozen
        # set.  The first *learned* value lands the next batch.  In prod the
        # held value is the seed (e.g. solar −4.0), not 0 — there is no reset.
        learned_idx = (ub + 1) if ub is not None else None
        at_unlock = _at(traj, learned_idx, coef)
        unlock_day = (ub / _BATCHES_PER_DAY) if ub is not None else _NAN

        m[f"{key}_truth"] = float(truth)
        m[f"{key}_final"] = float(final)
        m[f"{key}_err_final"] = abs(final - truth)
        m[f"{key}_at_fill"] = at_fill
        m[f"{key}_err_at_fill"] = abs(at_fill - truth)
        m[f"{key}_at_unlock"] = at_unlock
        m[f"{key}_err_at_unlock"] = abs(at_unlock - truth)
        m[f"{key}_unlock_day"] = unlock_day

        for d in _TRAJ_DAYS:
            m[f"{key}_traj_d{d:02d}"] = _at(traj, int(d * _BATCHES_PER_DAY), coef)

        # detected EMA lag-τ at each stage (only inputs with a lag carry it)
        m[f"{key}_tau_at_fill"] = _at(traj, fill_idx, f"{coef}_tau")
        m[f"{key}_tau_at_unlock"] = _at(traj, learned_idx, f"{coef}_tau")
        m[f"{key}_tau_final"] = _at(traj, len(traj) - 1, f"{coef}_tau")

        pf = post_fill.get(coef)
        m[f"{key}_post_fill_bias"] = pf.bias if pf else _NAN
        m[f"{key}_post_fill_std"] = pf.std if pf else _NAN
        m[f"{key}_post_fill_drift_per_day"] = pf.drift_per_day if pf else _NAN

    # ── Performance: comfort (±1°F) + chatter (HP setpoint changes/day) ──
    hist = result.history
    n_in_band = n_total = 0
    n_ctrl_in_band = n_ctrl = 0
    hp_changes = 0
    prev_sp = None
    worst_under = worst_over = 0.0
    for h in hist:
        room = h.get("room_temp")
        if room is None:
            continue
        dev = room - desired_c
        n_total += 1
        if abs(dev) <= _BAND_1F_C:
            n_in_band += 1
        if dev < 0:
            worst_under = max(worst_under, -dev)
        else:
            worst_over = max(worst_over, dev)
        if h.get("hp_estimated_active_state", True):
            n_ctrl += 1
            if abs(dev) <= _BAND_1F_C:
                n_ctrl_in_band += 1
        sp = h.get("hp_setpoint")
        if sp is not None and prev_sp is not None and sp != prev_sp:
            hp_changes += 1
        if sp is not None:
            prev_sp = sp

    n_days = result.n_ticks * _TICK_MINUTES / (60.0 * 24.0)
    m["comfort_pct_1f"] = (n_in_band / n_total) if n_total else _NAN
    m["ctrl_comfort_pct_1f"] = (n_ctrl_in_band / n_ctrl) if n_ctrl else _NAN
    m["hp_changes_per_day"] = (hp_changes / n_days) if n_days else _NAN
    m["total_itae"] = result.total_itae
    m["ctrl_violations"] = float(result.ctrl_violations)
    m["worst_undershoot"] = worst_under
    m["worst_overshoot"] = worst_over
    return m


def _run_and_pin(scenario, *, supervisor_enabled, bench_metrics, num_regression):
    """Run one scenario (qref on/off) and pin the learning + perf report."""
    config = _make_scenario(scenario, _N_DAYS)
    config.tick_minutes = _TICK_MINUTES
    if not supervisor_enabled:
        config.pi_overrides = {
            **config.pi_overrides, "pi_supervisor_enabled": False,
        }
    result = run_full_stack(config)

    report = _learning_report(result, config.desired_c)
    for k, v in report.items():
        bench_metrics[k] = v
    check_bench_metrics(num_regression, bench_metrics)

    # ── Loose, mode-agnostic non-divergence gates only ──────────────────
    od = result.final_coefs.get("outdoor_delta", _NAN)
    solar = result.final_coefs.get("Solar Proxy", _NAN)
    assert abs(od) < 5.0, f"{scenario}: outdoor_delta diverged: {od:.4f}"
    assert -10.0 < solar < 5.0, f"{scenario}: Solar Proxy diverged: {solar:.4f}"
    assert not math.isnan(report["comfort_pct_1f"]), f"{scenario}: no comfort data"


@pytest.mark.design
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_system_learning_report(scenario, bench_metrics, num_regression) -> None:
    """Production config (qref supervisor ON) on one real-weather scenario;
    pin the full learning + performance report.  Regression diff = how a code
    change moved system behaviour relative to truth and comfort.
    """
    _run_and_pin(scenario, supervisor_enabled=True,
                 bench_metrics=bench_metrics, num_regression=num_regression)


@pytest.mark.design
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_system_learning_report_qref_off(
    scenario, bench_metrics, num_regression,
) -> None:
    """Same report with the qref supervisor DISABLED.  qref pins the raw
    setpoint to cut HP chatter; the hypothesis is that pinning reduces
    excitation and hurts identification.  Diffing this against the qref-ON
    report (above) isolates the supervisor's effect on excitation, coefficient
    identification, comfort, and beeps — repeatably.
    """
    _run_and_pin(scenario, supervisor_enabled=False,
                 bench_metrics=bench_metrics, num_regression=num_regression)
