"""Real-CSV seasonal validation of 2R2C grey-box (#47 sibling success criterion #2).

Sibling to ``test_wls_vs_greybox.py``: same WLS-only / GB-only / Fused
comparison, but driven by real Open-Meteo CSV windows for fall, winter,
and spring instead of a single synthetic spring scenario.

Per ``project_greybox_1r1c_limitation.md`` Phase 3, the synthetic spring
result was 30/119 gates passed and β_outdoor within ~10% of truth — a
structural improvement over the 1R1C floor of 0/359. This file is the
next-stop validation: do those wins survive real cloud clustering, weather
fronts, and seasonal day-length shifts? ``feedback_synthetic_vs_real_bench.md``
is the precedent; the FIFO/leverage finding looked clean on synth and
inverted on real weather. Same risk applies here.

Marked ``@design``: 9 sims (3 seasons × 3 modes × ~60d each), runs on
demand alongside the synthetic-spring sibling.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.full_stack_runner import TICK_MINUTES_DEFAULT
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.scenarios._weather_mode import (
    SHOULDER_FALL,
    SHOULDER_SPRING,
    WINTER_TYPICAL,
    WeatherWindow,
    windowed_real_weather,
)
from tests.hvac_bench.thermal_model import ThermalModel2R2C


# Fall=2024-10-01, Winter=2024-01-01 (typical), Spring=2025-03-31. Same
# windows as test_seasonal_convergence.py so cross-test comparisons stay
# apples-to-apples.
SEASONS: dict[str, WeatherWindow] = {
    "fall": SHOULDER_FALL,
    "winter": WINTER_TYPICAL,
    "spring": SHOULDER_SPRING,
}


@dataclass
class ScenarioResult:
    season: str
    mode: str
    comfort: float
    integral_rms: float
    od_error: float
    solar_error: float
    gb_gates_passed: int
    gb_total_batches: int
    final_outdoor_beta: float
    final_solar_beta: float
    is_2r2c_dispatched: bool
    final_tau_fast: float | None
    final_tau_slow: float | None
    # Diagnostic capture (2026-05-06 re-verdict re-run)
    n_2r2c_batches: int = 0
    gate_failure_counts: dict[str, int] | None = None
    final_greybox_summary: dict[str, float | int | str | None] | None = None


def _run_scenario(
    *,
    mode: str,
    outdoor_fn,
    solar_fn,
    n_days: int,
    initial_outdoor: float,
    season_label: str,
) -> ScenarioResult:
    """One real-CSV scenario at a given (mode, season).

    mode: "wls_only", "gb_only", "fused". Same custom batch-intercept
    pattern as ``test_wls_vs_greybox._run_spring_scenario``.
    """
    profile = PROFILES_2R2C["living_room"]

    pi_config = {
        "pi_model_inputs": [{
            "entity_id": "sensor.solar_proxy",
            "name": "Solar Proxy",
            "input_role": "solar",
            "seed_heat": 0.0,
            "seed_cool": 0.0,
            "lag_tau": 120,
            "delta_from_room": False,
            "clamp_min": 0,
        }],
        "pi_outdoor_seed_heat": profile.true_seed,
        "pi_outdoor_seed_cool": profile.true_seed,
        "pi_ki": 0.15,
        "pi_kp": 1.5,
        "pi_deadband": 0.5,
        "pi_setpoint_weight": 0.3,
        "pi_greybox_blending": (mode == "fused"),
    }

    adapter = TasmotaPIAdapter(pi_config)
    pi = adapter._pi
    pi._batch_kappa_threshold = 10000
    pi._rls_online_learning = False  # batch-only for all modes

    if mode == "gb_only":
        pi._greybox_blending_enabled = True

    model = ThermalModel2R2C(
        profile=profile,
        initial_temp=20.5,
        outdoor_temp=initial_outdoor,
        sensor_noise_sigma=0.1,
        noise_seed=42,
        solar_gain=0.06,
        stove_gain=0.0,
    )

    adapter.set_desired_temp(20.5)
    adapter.set_mode("heat")

    tick_min = TICK_MINUTES_DEFAULT
    n_ticks = int(n_days * 24 * 60 / tick_min)
    batch_interval = int(12 * 60 / tick_min)

    history: list[dict] = []
    gb_gates_passed = 0
    gb_total = 0
    is_2r2c_seen = False
    n_2r2c_batches = 0
    gate_failure_counts: dict[str, int] = {}

    if mode == "gb_only":
        _orig_run_batch = pi._run_batch_analysis

        def _gb_only_batch():
            beta_before = list(pi._rls_heat.beta)
            _orig_run_batch()
            if (pi._last_greybox_bridge is None or
                    not pi._last_greybox_bridge.gates_passed):
                for i in range(len(beta_before)):
                    pi._rls_heat.beta[i] = beta_before[i]
            else:
                bridge = pi._last_greybox_bridge
                for i in range(min(len(bridge.beta), pi._rls_heat.n)):
                    if (bridge.beta[i] is not None
                            and math.isfinite(bridge.beta_std_err[i])):
                        pi._rls_heat.beta[i] = (
                            bridge.beta[i] * pi._rls_heat.feature_scales[i]
                        )

        pi._run_batch_analysis = _gb_only_batch

    for tick in range(n_ticks):
        dt_seconds = tick_min * 60.0
        model.outdoor_temp = outdoor_fn(tick)
        solar_val = solar_fn(tick) if solar_fn is not None else 0.0

        sensor_reading = model.read_sensor()

        adapter._sim_clock += dt_seconds
        adapter._entity._attr_current_temperature = sensor_reading
        pi._inputs.outdoor_temp = model.outdoor_temp

        ms = type("MockState", (), {
            "state": str(solar_val),
            "attributes": {"unit_of_measurement": None},
        })()
        _mock_states = {"sensor.solar_proxy": ms}
        pi._hass.states.get = lambda eid, _s=_mock_states: _s.get(eid)

        original = _time.monotonic
        _time.monotonic = lambda: adapter._sim_clock
        try:
            adapter._loop.run_until_complete(pi._pi_tick())
        finally:
            _time.monotonic = original

        hp_setpoint = float(pi._hp_setpoint)
        model.step(hp_setpoint=hp_setpoint, dt_minutes=tick_min,
                   tick=tick, solar_proxy=solar_val, mode="heat")

        error = 20.5 - model.room_temp
        history.append({"error": error})

        if tick > 0 and tick % batch_interval == 0:
            pi._run_batch_analysis()
            gb_total += 1
            bridge = pi._last_greybox_bridge
            if bridge is not None:
                if bridge.gates_passed:
                    gb_gates_passed += 1
                if bridge.greybox.is_2r2c:
                    is_2r2c_seen = True
                    n_2r2c_batches += 1
                for gate_name, passed in bridge.gate_details.items():
                    if not passed:
                        gate_failure_counts[gate_name] = (
                            gate_failure_counts.get(gate_name, 0) + 1
                        )

    in_band = sum(1 for h in history if abs(h["error"]) <= 0.5)
    comfort = 100.0 * in_band / len(history)
    integral_sq = sum(h["error"] ** 2 for h in history)
    integral_rms = math.sqrt(integral_sq / len(history))

    final_coefs = pi._rls_heat.get_coefficients()
    od_final = final_coefs.get(1, 0.0)
    solar_final = final_coefs.get(2, 0.0) if pi._rls_heat.n > 2 else 0.0

    od_true = profile.true_seed
    solar_true = -2.0  # parity with test_wls_vs_greybox

    last_bridge = pi._last_greybox_bridge
    final_tau_fast = (last_bridge.tau_fast if last_bridge is not None
                      else None)
    final_tau_slow = (last_bridge.tau_slow if last_bridge is not None
                      else None)

    final_greybox_summary: dict[str, float | int | str | None] | None = None
    last_gb = pi._last_greybox_result
    if last_gb is not None:
        final_greybox_summary = {
            "is_2r2c": last_gb.is_2r2c,
            "c0": last_gb.c0,
            "ua_c": last_gb.ua_c,
            "k_c": last_gb.k_c,
            "alpha_c": last_gb.alpha_c,
            "k_w": last_gb.k_w,
            "mass_ratio": last_gb.mass_ratio,
            "tau_eff": last_gb.tau_eff,
            "tau_fast": last_gb.tau_fast,
            "tau_slow": last_gb.tau_slow,
            "residual_rms": last_gb.residual_rms,
            "cost": last_gb.cost,
            "n_function_evals": last_gb.n_function_evals,
            "n_observations": last_gb.n_observations,
            "n_hp_on": last_gb.n_hp_on,
            "n_hp_off": last_gb.n_hp_off,
            "param_std_err": dict(last_gb.param_std_err)
            if last_gb.param_std_err else None,
        }

    return ScenarioResult(
        season=season_label,
        mode=mode,
        comfort=comfort,
        integral_rms=integral_rms,
        od_error=abs(od_final - (-od_true)),
        solar_error=abs(solar_final - solar_true),
        gb_gates_passed=gb_gates_passed,
        gb_total_batches=gb_total,
        final_outdoor_beta=od_final,
        final_solar_beta=solar_final,
        is_2r2c_dispatched=is_2r2c_seen,
        final_tau_fast=final_tau_fast,
        final_tau_slow=final_tau_slow,
        n_2r2c_batches=n_2r2c_batches,
        gate_failure_counts=gate_failure_counts,
        final_greybox_summary=final_greybox_summary,
    )


def _run_one_season(season: str, n_days: int = 60) -> dict[str, ScenarioResult]:
    """Run all 3 modes for one season and return {mode: ScenarioResult}."""
    window = SEASONS[season]
    outdoor_fn, solar_fn, max_days = windowed_real_weather(
        start_day=window.start_day, n_days=n_days,
    )
    initial_outdoor = outdoor_fn(0)
    out: dict[str, ScenarioResult] = {}
    for mode in ("wls_only", "gb_only", "fused"):
        out[mode] = _run_scenario(
            mode=mode,
            outdoor_fn=outdoor_fn,
            solar_fn=solar_fn,
            n_days=max_days,
            initial_outdoor=initial_outdoor,
            season_label=season,
        )
    return out


def _print_summary(
    all_results: dict[str, dict[str, ScenarioResult]],
) -> None:
    """3 seasons × 3 modes table: gates, β errors, comfort."""
    print(f"\n{'=' * 90}")
    print(f"  2R2C Grey-box Real-CSV Validation (living_room, ~60d per season)")
    print(f"{'=' * 90}")
    header = (f"{'Season':<8}{'Mode':<11}{'Gates':>10}"
              f"{'2R2C':>7}{'β_out err':>11}{'β_sol err':>11}"
              f"{'Comfort %':>11}{'τ_fast':>9}{'τ_slow':>9}")
    print(header)
    print("-" * len(header))
    for season, by_mode in all_results.items():
        for mode in ("wls_only", "gb_only", "fused"):
            r = by_mode[mode]
            gates = f"{r.gb_gates_passed}/{r.gb_total_batches}"
            tf = (f"{r.final_tau_fast:>9.0f}"
                  if r.final_tau_fast is not None else f"{'-':>9}")
            ts = (f"{r.final_tau_slow:>9.0f}"
                  if r.final_tau_slow is not None else f"{'-':>9}")
            print(
                f"{r.season:<8}{r.mode:<11}{gates:>10}"
                f"{('Y' if r.is_2r2c_dispatched else 'n'):>7}"
                f"{r.od_error:>11.4f}{r.solar_error:>11.4f}"
                f"{r.comfort:>10.1f}%{tf}{ts}"
            )
        print()


def _print_diagnostics(
    all_results: dict[str, dict[str, ScenarioResult]],
) -> None:
    """Per-(season, mode) breakdown: gate failure histogram + final fit dump."""
    print(f"\n{'=' * 90}")
    print(f"  Diagnostic detail (gate-failure counts + final fit values)")
    print(f"{'=' * 90}")
    for season, by_mode in all_results.items():
        for mode in ("wls_only", "gb_only", "fused"):
            r = by_mode[mode]
            print(f"\n[{season} / {mode}]  "
                  f"2R2C dispatched on {r.n_2r2c_batches}/{r.gb_total_batches} batches")
            if r.gate_failure_counts:
                sorted_fails = sorted(
                    r.gate_failure_counts.items(),
                    key=lambda kv: kv[1], reverse=True,
                )
                fail_str = ", ".join(
                    f"{name}={count}" for name, count in sorted_fails
                )
                print(f"  Gate failures (out of {r.gb_total_batches}): {fail_str}")
            else:
                print(f"  Gate failures: none recorded")
            s = r.final_greybox_summary
            if s is None:
                print(f"  Final fit: <no greybox result captured>")
                continue
            kind = "2R2C" if s["is_2r2c"] else "1R1C"
            print(f"  Final fit ({kind}): "
                  f"n_obs={s['n_observations']} "
                  f"n_hp_on={s['n_hp_on']} n_hp_off={s['n_hp_off']} "
                  f"cost={s['cost']:.5f} nfev={s['n_function_evals']} "
                  f"residual_rms={s['residual_rms']:.5f}")
            print(f"    c0={s['c0']:.5f}  ua_c={s['ua_c']:.5f}  "
                  f"k_c={s['k_c']:.5f}  alpha_c={s['alpha_c']:.5f}")
            if s["is_2r2c"]:
                k_w = s["k_w"] if s["k_w"] is not None else float("nan")
                mr = s["mass_ratio"] if s["mass_ratio"] is not None else float("nan")
                tf = s["tau_fast"] if s["tau_fast"] is not None else float("nan")
                ts = s["tau_slow"] if s["tau_slow"] is not None else float("nan")
                print(f"    k_w={k_w:.5f}  mass_ratio={mr:.2f}  "
                      f"τ_fast={tf:.0f}min  τ_slow={ts:.0f}min")
            else:
                print(f"    τ_eff={s['tau_eff']:.0f}min")
            pse = s["param_std_err"]
            if pse:
                cv_parts = []
                raw = {
                    "ua_c": s["ua_c"], "k_c": s["k_c"], "alpha_c": s["alpha_c"],
                    "k_w": s["k_w"], "mass_ratio": s["mass_ratio"],
                }
                for name, sigma in pse.items():
                    val = raw.get(name)
                    if val is not None and abs(val) > 1e-12:
                        cv_parts.append(f"{name}: σ={sigma:.5f} CV={sigma/abs(val):.3f}")
                    else:
                        cv_parts.append(f"{name}: σ={sigma:.5f}")
                print(f"    std_err: {', '.join(cv_parts)}")


def run_greybox_real_csv_summary() -> None:
    """CLI entry: run all (season × mode) combos and print the summary."""
    results = {s: _run_one_season(s) for s in SEASONS}
    _print_summary(results)
    _print_diagnostics(results)


@pytest.fixture(scope="module")
def real_csv_results() -> dict[str, dict[str, ScenarioResult]]:
    """Run all (season × mode) combos once. 9 sims, ~5–10 minutes."""
    return {s: _run_one_season(s) for s in SEASONS}


@pytest.mark.design
class TestGreybox2R2CRealCSV:
    """Validate 2R2C grey-box on real Open-Meteo CSVs across heating seasons.

    Marked ``design``: this is the verdict source for the 1R1C → 2R2C
    upgrade (#47 sibling success criterion #2). Production only consumes
    the grey-box bridge when ``pi_greybox_blending`` is on for a zone, so
    this test exists to validate the multi-season behavior before any
    flag flip.
    """

    def test_2r2c_dispatch_fires_each_season(self, real_csv_results):
        """At ~60 days × 96 ticks/day, the 2R2C gate (≥1500 obs AND ≥14 days)
        must fire in every season for at least one mode. If 2R2C never
        dispatches, the test is degenerate and the rest of the assertions
        say nothing about the upgrade."""
        for season, by_mode in real_csv_results.items():
            assert any(r.is_2r2c_dispatched for r in by_mode.values()), (
                f"{season}: 2R2C never dispatched in any mode "
                f"(60d × 96 ticks should easily clear the ≥1500 obs and "
                f"≥14-day span gates). Tau dispatch logic regression?"
            )

    @pytest.mark.xfail(
        reason="2R2C grey-box passes 0/119 gates per season on real CSV — "
        "synthetic-spring 30/119 win did NOT transfer (verdict 2026-05-01). "
        "τ_fast collapses to 700+ min and τ_slow to 100,000+ min, both far "
        "outside plausible bands. See project_greybox_2r2c_real_csv_finding.md.",
        strict=True,
    )
    def test_gates_pass_at_least_once_per_season(self, real_csv_results):
        """The 1R1C floor was 0/N gates passed. The 2R2C upgrade has to
        clear that floor on real weather; otherwise the synthetic-spring
        win at 30/119 was an artifact of clean synth distributions
        (the precedent in feedback_synthetic_vs_real_bench.md)."""
        for season, by_mode in real_csv_results.items():
            r = by_mode["fused"]
            assert r.gb_gates_passed > 0, (
                f"{season}: fused arm passed 0/{r.gb_total_batches} grey-box "
                f"gates — 2R2C wins didn't transfer from synth to real CSV"
            )

    def test_no_arm_diverges_in_outdoor(self, real_csv_results):
        """Sanity: every (season, mode) ends with a bounded outdoor β."""
        for season, by_mode in real_csv_results.items():
            for mode, r in by_mode.items():
                assert -2.0 < r.final_outdoor_beta < 0.0, (
                    f"{season}/{mode}: outdoor β={r.final_outdoor_beta:.4f} "
                    f"out of plausible range"
                )

    @pytest.mark.xfail(
        reason="On real CSV the 2R2C fit converges to τ_fast≈700–800 min and "
        "τ_slow≈100,000+ min — degenerate solutions with τ_fast above the "
        "5–60 plausible band. Same root cause as test_gates_pass_at_least_"
        "once_per_season. See project_greybox_2r2c_real_csv_finding.md.",
        strict=True,
    )
    def test_tau_fast_in_plausible_range_when_2r2c(self, real_csv_results):
        """Where 2R2C dispatched, the final τ_fast must land in the plant-ID
        plausible band (5–60 min). Verifies the dual-τ provider feeds
        plant_identifier with sensible numbers — the user-visible point of
        the 2R2C upgrade per Phase 1 design."""
        any_checked = False
        for season, by_mode in real_csv_results.items():
            for mode, r in by_mode.items():
                if r.is_2r2c_dispatched and r.final_tau_fast is not None:
                    any_checked = True
                    assert 5.0 <= r.final_tau_fast <= 60.0, (
                        f"{season}/{mode}: final τ_fast={r.final_tau_fast:.1f} "
                        f"min outside plausible range (5–60)"
                    )
        assert any_checked, (
            "No (season, mode) combination ever dispatched 2R2C with a "
            "non-None τ_fast — dispatch test should have caught this first"
        )

    def test_summary_print(self, real_csv_results):
        """Print-only: surface the comparison table for memory updates."""
        _print_summary(real_csv_results)
        _print_diagnostics(real_csv_results)
