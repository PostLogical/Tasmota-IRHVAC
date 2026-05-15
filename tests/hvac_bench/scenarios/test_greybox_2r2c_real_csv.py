"""Real-CSV seasonal validation of 2R2C grey-box (#47 sibling success criterion #2).

Sibling to ``test_wls_vs_greybox.py``: same WLS-only vs Fused comparison,
but driven by real Open-Meteo CSV windows for fall, winter, and spring
instead of a synthetic spring scenario.

Per ``project_greybox_1r1c_limitation.md`` Phase 3, the synthetic spring
result was 30/119 gates passed and β_outdoor within ~10% of truth — a
structural improvement over the 1R1C floor of 0/359. This file is the
next-stop validation: do those wins survive real cloud clustering, weather
fronts, and seasonal day-length shifts? ``feedback_synthetic_vs_real_bench.md``
is the precedent; the FIFO/leverage finding looked clean on synth and
inverted on real weather. Same risk applies here.

History note: this file previously ran an inline simulation loop with a
third arm (``gb_only``) that monkey-patched ``pi._run_batch_analysis`` to
revert WLS coefficient writes. That arm was dropped (2026-05-06) when
migrating to ``full_stack_runner``: the verdict memory established that
gb_only rails the same way wls_only and fused do. The inline loop also
hardcoded ``solar_gain=0.06`` while asserting ``solar_true=-2.0``, producing
a physics inconsistency (β_solar implied by physics is −solar_gain/hp_gain
= −1.5, not −2.0). Migrating to ``full_stack_runner`` fixes this:
``ModelInputSpec.resolve()`` derives ``solar_thermal_gain = |β_solar_truth|
× hp_gain`` so the model and the asserted truth agree.

Marked ``@design``: 6 sims (3 seasons × 2 modes × ~60d each), runs on
demand alongside the synthetic-spring sibling.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    ModelInputSpec,
    run_full_stack,
    TICK_MINUTES_DEFAULT,
)
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.house_profiles import PROFILES, PROFILES_2R2C
from tests.hvac_bench.scenarios._weather_mode import (
    SHOULDER_FALL,
    SHOULDER_SPRING,
    WINTER_TYPICAL,
    WeatherWindow,
    windowed_real_weather,
)


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
    n_2r2c_batches: int = 0
    gate_failure_counts: dict[str, int] | None = None
    final_greybox_summary: dict[str, float | int | str | None] | None = None


def _resolve_profile(profile_name: str):
    """Look up a HouseProfile2R2C from either PROFILES (lit-grounded) or
    PROFILES_2R2C (production-calibrated). Mirrors full_stack_runner.py
    lookup so callers can use names from either dict."""
    if profile_name in PROFILES_2R2C:
        return PROFILES_2R2C[profile_name]
    if profile_name in PROFILES:
        return PROFILES[profile_name]
    raise ValueError(f"Unknown profile: {profile_name}")


def _run_scenario(
    *,
    mode: str,
    outdoor_fn,
    solar_fn,
    n_days: int,
    season_label: str,
    profile_name: str = "living_room",
) -> ScenarioResult:
    """One real-CSV scenario at a given (mode, season, profile).

    mode: "wls_only" (greybox blending off) or "fused" (greybox blending on).
    profile_name: looked up in PROFILES_2R2C first, then PROFILES (lit-grounded).
    """
    profile = _resolve_profile(profile_name)

    # Per-batch state captured via callback
    state: dict = {
        "gb_gates_passed": 0,
        "gb_total": 0,
        "is_2r2c_seen": False,
        "n_2r2c_batches": 0,
        "gate_failure_counts": {},
        "final_greybox_summary": None,
        "final_tau_fast": None,
        "final_tau_slow": None,
    }

    def _on_batch(batch_idx: int, pi: object) -> None:
        state["gb_total"] += 1
        bridge = getattr(pi, "_last_greybox_bridge", None)
        last_gb = getattr(pi, "_last_greybox_result", None)
        if bridge is not None:
            if bridge.gates_passed:
                state["gb_gates_passed"] += 1
            if bridge.greybox.is_2r2c:
                state["is_2r2c_seen"] = True
                state["n_2r2c_batches"] += 1
            for gate_name, passed in bridge.gate_details.items():
                if not passed:
                    state["gate_failure_counts"][gate_name] = (
                        state["gate_failure_counts"].get(gate_name, 0) + 1
                    )
            state["final_tau_fast"] = bridge.tau_fast
            state["final_tau_slow"] = bridge.tau_slow
        if last_gb is not None:
            state["final_greybox_summary"] = {
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

    config = FullStackConfig(
        n_days=n_days,
        profile_name=profile_name,
        desired_c=20.5,
        mode="heat",
        noise_sigma=0.1,
        noise_seed=42,
        tick_minutes=TICK_MINUTES_DEFAULT,
        outdoor_schedule=outdoor_fn,
        model_inputs=[
            ModelInputSpec(
                name="Solar Proxy",
                entity_id="sensor.solar_proxy",
                input_role="solar",
                _true_ff_coef=-2.0,
                seed_heat=0.0,
                seed_cool=0.0,
                lag_tau=120,
                clamp_min=0,
                schedule=solar_fn,
            ),
        ],
        pi_overrides={
            "pi_outdoor_seed_heat": profile.true_seed,
            "pi_outdoor_seed_cool": profile.true_seed,
            "pi_ki": 0.15,
            "pi_kp": 1.5,
            "pi_setpoint_weight": 0.3,
            "pi_greybox_blending": (mode == "fused"),
            "pi_rls_online_learning": False,
        },
        relax_kappa_gate=True,
        batch_callback=_on_batch,
    )

    result = run_full_stack(config)

    od_final = result.final_coefs.get("outdoor_delta", 0.0)
    solar_final = result.final_coefs.get("Solar Proxy", 0.0)
    od_truth = result.true_coefs.get("outdoor_delta", profile.true_seed)
    solar_truth = result.true_coefs.get("Solar Proxy", -2.0)

    return ScenarioResult(
        season=season_label,
        mode=mode,
        comfort=result.comfort_hours_pct,
        integral_rms=result.integral_rms,
        od_error=abs(od_final - od_truth),
        solar_error=abs(solar_final - solar_truth),
        gb_gates_passed=state["gb_gates_passed"],
        gb_total_batches=state["gb_total"],
        final_outdoor_beta=od_final,
        final_solar_beta=solar_final,
        is_2r2c_dispatched=state["is_2r2c_seen"],
        final_tau_fast=state["final_tau_fast"],
        final_tau_slow=state["final_tau_slow"],
        n_2r2c_batches=state["n_2r2c_batches"],
        gate_failure_counts=state["gate_failure_counts"],
        final_greybox_summary=state["final_greybox_summary"],
    )


def _run_one_season(
    season: str,
    n_days: int = 60,
    profile_name: str = "living_room",
) -> dict[str, ScenarioResult]:
    """Run wls_only + fused for one season and return {mode: ScenarioResult}."""
    window = SEASONS[season]
    outdoor_fn, solar_fn, max_days = windowed_real_weather(
        start_day=window.start_day, n_days=n_days,
    )
    out: dict[str, ScenarioResult] = {}
    for mode in ("wls_only", "fused"):
        out[mode] = _run_scenario(
            mode=mode,
            outdoor_fn=outdoor_fn,
            solar_fn=solar_fn,
            n_days=max_days,
            season_label=season,
            profile_name=profile_name,
        )
    return out


def _print_summary(
    all_results: dict[str, dict[str, ScenarioResult]],
) -> None:
    """3 seasons × 2 modes table: gates, β errors, comfort."""
    print(f"\n{'=' * 90}")
    print("  2R2C Grey-box Real-CSV Validation (living_room, ~60d per season)")
    print(f"{'=' * 90}")
    header = (f"{'Season':<8}{'Mode':<11}{'Gates':>10}"
              f"{'2R2C':>7}{'β_out err':>11}{'β_sol err':>11}"
              f"{'Comfort %':>11}{'τ_fast':>9}{'τ_slow':>9}")
    print(header)
    print("-" * len(header))
    for season, by_mode in all_results.items():
        for mode in ("wls_only", "fused"):
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
    print("  Diagnostic detail (gate-failure counts + final fit values)")
    print(f"{'=' * 90}")
    for season, by_mode in all_results.items():
        for mode in ("wls_only", "fused"):
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
                print("  Gate failures: none recorded")
            s = r.final_greybox_summary
            if s is None:
                print("  Final fit: <no greybox result captured>")
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
                        cv_parts.append(f"{name}: σ={sigma:.5f} CV={sigma / abs(val):.3f}")
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
    """Run all (season × mode) combos once. 6 sims, ~5–10 minutes."""
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

    def test_2r2c_dispatch_fires_each_season(self, bench_metrics, num_regression, real_csv_results):
        """At ~60 days × 96 ticks/day, the 2R2C gate (≥1500 obs AND ≥14 days)
        must fire in every season for at least one mode. If 2R2C never
        dispatches, the test is degenerate and the rest of the assertions
        say nothing about the upgrade."""
        for season, by_mode in real_csv_results.items():
            for mode, r in by_mode.items():
                bench_metrics[f"{season}__{mode}__n_2r2c_batches"] = r.n_2r2c_batches
                # Locked count alone is uninformative — "200 dispatches" means
                # different things at 400 total vs 4000 total batches.  Lock
                # the denominator + ratio.
                bench_metrics[f"{season}__{mode}__gb_total_batches"] = r.gb_total_batches
                if r.gb_total_batches > 0:
                    bench_metrics[f"{season}__{mode}__dispatch_rate"] = (
                        r.n_2r2c_batches / r.gb_total_batches
                    )
        check_bench_metrics(num_regression, bench_metrics)
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
    def test_gates_pass_at_least_once_per_season(self, bench_metrics, num_regression, real_csv_results):
        """The 1R1C floor was 0/N gates passed. The 2R2C upgrade has to
        clear that floor on real weather; otherwise the synthetic-spring
        win at 30/119 was an artifact of clean synth distributions
        (the precedent in feedback_synthetic_vs_real_bench.md)."""
        for season, by_mode in real_csv_results.items():
            r = by_mode["fused"]
            bench_metrics[f"{season}__fused__gb_gates_passed"] = r.gb_gates_passed
            bench_metrics[f"{season}__fused__gb_total_batches"] = r.gb_total_batches
            # When this xfail eventually xpasses (2R2C upgrade lands), the
            # diagnostic question is WHICH gates are now passing.  Lock the
            # per-gate failure counts so the xpass diff shows what changed.
            if r.gate_failure_counts:
                for gate_name, count in sorted(r.gate_failure_counts.items()):
                    bench_metrics[f"{season}__fused__gate_fail__{gate_name}"] = count
        check_bench_metrics(num_regression, bench_metrics)
        for season, by_mode in real_csv_results.items():
            r = by_mode["fused"]
            assert r.gb_gates_passed > 0, (
                f"{season}: fused arm passed 0/{r.gb_total_batches} grey-box "
                f"gates — 2R2C wins didn't transfer from synth to real CSV"
            )

    def test_no_arm_diverges_in_outdoor(self, bench_metrics, num_regression, real_csv_results):
        """Sanity: every (season, mode) ends with a bounded outdoor β."""
        for season, by_mode in real_csv_results.items():
            for mode, r in by_mode.items():
                bench_metrics[f"{season}__{mode}__outdoor_beta"] = r.final_outdoor_beta
                bench_metrics[f"{season}__{mode}__solar_beta"] = r.final_solar_beta
        check_bench_metrics(num_regression, bench_metrics)
        for season, by_mode in real_csv_results.items():
            for mode, r in by_mode.items():
                assert -2.0 < r.final_outdoor_beta < 0.0, (
                    f"{season}/{mode}: outdoor β={r.final_outdoor_beta:.4f} "
                    f"out of plausible range"
                )

    def test_tau_fast_in_plausible_range_when_2r2c(self, bench_metrics, num_regression, real_csv_results):
        """Where 2R2C dispatched, the final τ_fast must land in the plant-ID
        plausible band (5–60 min). Verifies the dual-τ provider feeds
        plant_identifier with sensible numbers — the user-visible point of
        the 2R2C upgrade per Phase 1 design."""
        any_checked = False
        for season, by_mode in real_csv_results.items():
            for mode, r in by_mode.items():
                if r.is_2r2c_dispatched and r.final_tau_fast is not None:
                    any_checked = True
                    bench_metrics[f"{season}__{mode}__tau_fast"] = r.final_tau_fast
                    assert 5.0 <= r.final_tau_fast <= 60.0, (
                        f"{season}/{mode}: final τ_fast={r.final_tau_fast:.1f} "
                        f"min outside plausible range (5–60)"
                    )
        check_bench_metrics(num_regression, bench_metrics)
        assert any_checked, (
            "No (season, mode) combination ever dispatched 2R2C with a "
            "non-None τ_fast — dispatch test should have caught this first"
        )

    def test_summary_print(self, bench_metrics, num_regression, real_csv_results):
        """Print-only: surface the comparison table for memory updates."""
        _print_summary(real_csv_results)
        _print_diagnostics(real_csv_results)


# ── Lit-grounded validation against PROFILES["standard_residential"] ─────
#
# Bacher-Madsen 2011 typical residential params, used as ground truth so
# greybox recovery can be validated against known values rather than
# fitted-from-uncertain-data placeholders. Per
# feedback_living_room_calibration_suspect.md, the production-calibrated
# living_room profile shouldn't be cited as truth; this scenario fills
# the gap.


# Truth values derived from PROFILES["standard_residential"]:
#   tau_env=100, tau_couple=80, mass_ratio=8, hp_gain=0.025
_LIT_TRUE_UA_C = 1.0 / 100.0      # = 0.01 min⁻¹
_LIT_TRUE_K_C = 0.025              # min⁻¹
_LIT_TRUE_K_W = 1.0 / 80.0         # = 0.0125 min⁻¹
_LIT_TRUE_MASS_RATIO = 8.0
_LIT_TRUE_FF_COEF = -2.0           # asserted; α_total = 2.0 × 0.025 = 0.05
_LIT_TRUE_ALPHA_TOTAL = abs(_LIT_TRUE_FF_COEF) * _LIT_TRUE_K_C
_LIT_TRUE_BETA_OUTDOOR = -_LIT_TRUE_UA_C / _LIT_TRUE_K_C   # = -0.4
_LIT_TRUE_BETA_SOLAR = _LIT_TRUE_FF_COEF                     # = -2.0


@pytest.fixture(scope="module")
def lit_grounded_results() -> dict[str, dict[str, ScenarioResult]]:
    """Single season (spring) on standard_residential, both modes.

    Spring chosen because shoulder-season weather mixes outdoor variation
    with non-trivial solar — best regime for excitation per
    project_buffer_seasonal_findings.md.
    """
    return {"spring": _run_one_season(
        "spring", n_days=60, profile_name="standard_residential",
    )}


@pytest.mark.design
class TestGreybox2R2CLitGrounded:
    """Lit-grounded recovery validation: does greybox recover known truth
    parameters when driven by real Open-Meteo weather + the lit-grounded
    standard_residential profile (Bacher-Madsen typical residential)?

    This is the validation that was missing — the existing real-CSV test
    used the production-calibrated living_room profile (suspect — see
    feedback_living_room_calibration_suspect.md), so its results couldn't
    be assessed against any reference. With known-truth driving the
    thermal model, fit quality is directly measurable.
    """

    def test_2r2c_dispatched(self, bench_metrics, num_regression, lit_grounded_results):
        """Sanity: 60 days × 96 ticks/day clears the 2R2C dispatch gates."""
        for season, by_mode in lit_grounded_results.items():
            for mode, r in by_mode.items():
                bench_metrics[f"{season}__{mode}__n_2r2c_batches"] = r.n_2r2c_batches
        check_bench_metrics(num_regression, bench_metrics)
        for season, by_mode in lit_grounded_results.items():
            assert any(r.is_2r2c_dispatched for r in by_mode.values()), (
                f"{season}: 2R2C never dispatched — dispatch logic regression?"
            )

    @pytest.mark.xfail(
        reason="ua_c lands at ~14× underestimate of standard_residential "
        "truth on real-CSV bench. Diagnosis: rate-convention bug — "
        "production room_rate is 5-tick trailing FD (averaged); residual "
        "evaluates instantaneous predicted rate at i. Magnitude-scale "
        "bias preserves ratios (β_outdoor ≈ truth) but underestimates "
        "raw RC params. residual_rms ≈ 0.0075 °C/min is below the "
        "sensor-noise rate floor — fit is converging to sub-physical "
        "scaled params. See project_greybox_rate_convention_bug.md "
        "Probes 2/3 — greybox-on-synthetic-with-instantaneous-rate "
        "recovers truth.",
        strict=True,
    )
    def test_recovers_ua_c(self, bench_metrics, num_regression, lit_grounded_results):
        """Recovered ua_c within 30% of truth (= 0.01 min⁻¹).

        ua_c is a free parameter (not pinned by priors) so this is a
        direct test of greybox identification on real-weather data.
        """
        r = lit_grounded_results["spring"]["fused"]
        s = r.final_greybox_summary
        assert s is not None and s["is_2r2c"], "no 2R2C summary captured"
        ua_c = s["ua_c"]
        rel_err = abs(ua_c - _LIT_TRUE_UA_C) / _LIT_TRUE_UA_C
        bench_metrics["ua_c"] = ua_c
        bench_metrics["rel_err"] = rel_err
        check_bench_metrics(num_regression, bench_metrics)
        assert rel_err < 0.30, (
            f"ua_c={ua_c:.5f} vs truth {_LIT_TRUE_UA_C} ({100 * rel_err:.0f}% off)"
        )

    @pytest.mark.xfail(
        reason="k_c underestimated by same factor as ua_c (rate-convention "
        "bug); ratios preserved so β_outdoor ≈ truth despite raw param "
        "scale being wrong. See test_recovers_ua_c.",
        strict=True,
    )
    def test_recovers_k_c(self, bench_metrics, num_regression, lit_grounded_results):
        """Recovered k_c within 30% of truth (= 0.025 min⁻¹). Free param."""
        r = lit_grounded_results["spring"]["fused"]
        s = r.final_greybox_summary
        assert s is not None and s["is_2r2c"]
        k_c = s["k_c"]
        rel_err = abs(k_c - _LIT_TRUE_K_C) / _LIT_TRUE_K_C
        bench_metrics["k_c"] = k_c
        bench_metrics["rel_err"] = rel_err
        check_bench_metrics(num_regression, bench_metrics)
        assert rel_err < 0.30, (
            f"k_c={k_c:.5f} vs truth {_LIT_TRUE_K_C} ({100 * rel_err:.0f}% off)"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Last known-passing at a116154 (2026-05-06). Observed at 4cbb9ac "
        "and HEAD (2026-05-15): 2R2C dispatched every batch; optimizer lands at "
        "ua_c≈0.00044 (truth 0.01), k_c≈0.001 (truth 0.025), α_total≈0.00007 "
        "(truth 0.05 — measured here, ≈700× under), with k_w=0.02 / mass_ratio=8 "
        "pinned at Bayesian priors; residual_rms ≈ 0.55 °C/min (≈ 70× larger "
        "than the rate-convention-bug residual cited by test_recovers_ua_c's "
        "xfail). Root cause not yet diagnosed; may overlap with test_recovers_ua_c "
        "but the much larger residual indicates at least one additional factor "
        "not described there.",
    )
    def test_recovers_alpha_total(self, bench_metrics, num_regression, lit_grounded_results):
        """Recovered α_total within 50% of truth (= 0.05). Free param.

        Looser tolerance than ua_c/k_c because α_total identification is
        harder under closed-loop solar (project_bench_solar_fidelity.md);
        50% covers the lit-grounded "right sign + same order of magnitude"
        bar from earlier sessions.
        """
        r = lit_grounded_results["spring"]["fused"]
        s = r.final_greybox_summary
        assert s is not None and s["is_2r2c"]
        alpha_c = s["alpha_c"]
        rel_err = abs(alpha_c - _LIT_TRUE_ALPHA_TOTAL) / _LIT_TRUE_ALPHA_TOTAL
        bench_metrics["alpha_c"] = alpha_c
        bench_metrics["rel_err"] = rel_err
        check_bench_metrics(num_regression, bench_metrics)
        assert rel_err < 0.50, (
            f"α_total={alpha_c:.5f} vs truth {_LIT_TRUE_ALPHA_TOTAL} "
            f"({100 * rel_err:.0f}% off)"
        )

    def test_tau_fast_in_band(self, bench_metrics, num_regression, lit_grounded_results):
        """Final τ_fast in [5, 60] min plausible band."""
        r = lit_grounded_results["spring"]["fused"]
        assert r.final_tau_fast is not None
        bench_metrics["tau_fast"] = r.final_tau_fast
        check_bench_metrics(num_regression, bench_metrics)
        assert 5.0 <= r.final_tau_fast <= 60.0, (
            f"τ_fast={r.final_tau_fast:.1f} min outside [5, 60]"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Last known-passing at a116154 (2026-05-06). Observed at 4cbb9ac "
        "and HEAD (2026-05-15): 2R2C dispatched every batch; optimizer lands at "
        "ua_c≈0.00044 (truth 0.01), k_c≈0.001 (truth 0.025), α_total≈0.00007 "
        "(truth 0.05), with k_w=0.02 / mass_ratio=8 pinned at Bayesian priors; "
        "residual_rms ≈ 0.55 °C/min (≈ 70× larger than the rate-convention-bug "
        "residual cited by test_recovers_ua_c's xfail). The slow eigenvalue "
        "derived from these parameters lands at ≈ 21000 min — measured here, "
        "outside [60, 3500]. Root cause not yet diagnosed; may overlap with "
        "test_recovers_ua_c but the much larger residual indicates at least one "
        "additional factor not described there.",
    )
    def test_tau_slow_in_band(self, bench_metrics, num_regression, lit_grounded_results):
        """Final τ_slow in [60, 3500] min plausible band.

        Truth-implied τ_slow with priors: ~1250 min (priors pin k_w=0.02,
        mass_ratio=8; ua_c truth gives this τ_slow). Should comfortably
        clear the gate.
        """
        r = lit_grounded_results["spring"]["fused"]
        assert r.final_tau_slow is not None
        bench_metrics["tau_slow"] = r.final_tau_slow
        check_bench_metrics(num_regression, bench_metrics)
        assert 60.0 <= r.final_tau_slow <= 3500.0, (
            f"τ_slow={r.final_tau_slow:.0f} min outside [60, 3500]"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Last known-passing at a116154 (2026-05-06). Observed at 4cbb9ac "
        "and HEAD (2026-05-15): 2R2C dispatched every batch (120/120) but all "
        "120 batches fail the full gate set — measured here, 0/120 pass. "
        "Optimizer lands at ua_c≈0.00044 (truth 0.01), k_c≈0.001 (truth 0.025), "
        "α_total≈0.00007 (truth 0.05), with k_w=0.02 / mass_ratio=8 pinned at "
        "Bayesian priors; residual_rms ≈ 0.55 °C/min (≈ 70× larger than the "
        "rate-convention-bug residual cited by test_recovers_ua_c's xfail). "
        "Recurring failed gates: param_precision_alpha_c, tau_slow_plausible, "
        "residual_rms. Root cause not yet diagnosed; may overlap with "
        "test_recovers_ua_c but the much larger residual indicates at least one "
        "additional factor not described there.",
    )
    def test_gates_pass_at_least_once(self, bench_metrics, num_regression, lit_grounded_results):
        """At least one batch passes the full gate set on standard_residential
        spring data. This is the headline: with lit-grounded truth + priors,
        does the greybox actually pass its quality gates?"""
        r = lit_grounded_results["spring"]["fused"]
        bench_metrics["gb_gates_passed"] = r.gb_gates_passed
        bench_metrics["gb_total_batches"] = r.gb_total_batches
        check_bench_metrics(num_regression, bench_metrics)
        assert r.gb_gates_passed > 0, (
            f"0/{r.gb_total_batches} batches passed gates on lit-grounded "
            f"profile — greybox doesn't deliver usable bridge under "
            f"realistic-weather + known-truth conditions"
        )

    def test_summary_print_lit_grounded(self, bench_metrics, num_regression, lit_grounded_results):
        """Print-only: surface the lit-grounded comparison vs truth."""
        print(f"\n{'=' * 90}")
        print("  Lit-Grounded Validation (standard_residential, spring, 60d)")
        print(f"  Truth: ua_c={_LIT_TRUE_UA_C}, k_c={_LIT_TRUE_K_C}, "
              f"α_total={_LIT_TRUE_ALPHA_TOTAL}, mass_ratio={_LIT_TRUE_MASS_RATIO}")
        print(f"  Truth-derived: β_outdoor={_LIT_TRUE_BETA_OUTDOOR}, "
              f"β_solar={_LIT_TRUE_BETA_SOLAR}")
        print(f"{'=' * 90}")
        _print_summary(lit_grounded_results)
        _print_diagnostics(lit_grounded_results)
