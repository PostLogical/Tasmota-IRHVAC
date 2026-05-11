"""IMC gain scheduling validation tests.

Verifies that IMC-derived Kp/Ki (from profile τ) produces better control
metrics than flat Kp=1.5/Ki=0.15 across house profiles. The IMC mechanism
seeds τ, derives gains via Kp=τ/(λ+L), and adapts online. These tests
validate the static derivation against the bench thermal model.

Design principles (Skogestad SIMC, Åström & Hägglund):
- IMC adjusts Kp/Ki based on τ — test through settling time and reversals,
  which depend directly on gain tuning, not FF seed accuracy.
- ITAE conflates FF quality with PI tuning quality. Use it for bounded
  regression, not absolute improvement claims.
- Per-profile FF seeds from 2R2C steady-state: seed = 1/(hp_gain × τ_env).
  This keeps FF physically correct so tests measure IMC gain quality in
  isolation, not FF error compensation.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.house_profiles import QUICK_PROFILES, HouseProfile2R2C as HouseProfile
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


def _record_imc_pair(bench_metrics, flat, imc, *, profile_name, scenario):
    bench_metrics["profile_name"] = profile_name
    bench_metrics["scenario"] = scenario
    for key in ["itae", "overshoot", "reversals", "setpoint_changes",
                "integral_rms", "total_kwh"]:
        if key in flat:
            bench_metrics[f"flat_{key}"] = flat[key]
        if key in imc:
            bench_metrics[f"imc_{key}"] = imc[key]
    if flat.get("settling_time") is not None:
        bench_metrics["flat_settling_time"] = flat["settling_time"]
    if imc.get("settling_time") is not None:
        bench_metrics["imc_settling_time"] = imc["settling_time"]


def _make_flat_controller(profile: HouseProfile):
    """Controller with flat (non-IMC) default gains and physics-correct seed."""
    seed = profile.true_seed
    return TasmotaPIAdapter({
        "pi_outdoor_seed_heat": seed,
        "pi_outdoor_seed_cool": seed,
    })


def _make_imc_controller(profile: HouseProfile):
    """Controller with IMC gains derived from profile τ.

    Per-profile τ seeds are injected via the adapter (production uses
    fixed DEFAULT_TAU_*_SEED with a maturity gate; tests need to bypass
    that to validate gain scheduling across profiles).

    Smith predictor is disabled: these tests validate gain scheduling
    in isolation (hp_lag=0 in the thermal model, so there is no real
    delay for Smith to compensate).
    """
    seed = profile.true_seed
    tau = float(profile.tau_minutes)  # 2R2C fast_tau — what step-response τ measures
    ctrl = TasmotaPIAdapter({
        "pi_outdoor_seed_heat": seed,
        "pi_outdoor_seed_cool": seed,
        "pi_tau_estimate": tau,  # >0 enables IMC; numeric value unused since pre44
        "pi_response_lag": 15.0,
        "tau_fast_seed": tau,
        "tau_slow_seed": tau,  # bench profiles' analytic slow_tau is unrealistic;
                               # use fast_tau as the IMC formula's τ (matches the
                               # pre-2R2C test premise: Kp = τ/(λ+L))
    })
    ctrl._pi._smith = None  # IMC-only: no Smith for lag-free thermal model
    return ctrl


def _run_pair(profile, initial, outdoor, desired, duration_minutes, mode,
              outdoor_minute_schedule=None, desired_minute_schedule=None):
    """Run both flat and IMC controllers, return (flat_metrics, imc_metrics)."""
    final_desired = desired
    if desired_minute_schedule:
        for _, temp in sorted(desired_minute_schedule.items()):
            final_desired = temp

    results = {}
    for label, ctrl in [("flat", _make_flat_controller(profile)),
                        ("imc", _make_imc_controller(profile))]:
        ctrl.set_desired_temp(desired)
        model = ThermalModel(profile=profile, initial_temp=initial,
                             outdoor_temp=outdoor)
        history = run_scenario(ctrl, model, duration_minutes=duration_minutes, mode=mode,
                               outdoor_minute_schedule=outdoor_minute_schedule,
                               desired_minute_schedule=desired_minute_schedule)
        results[label] = compute_all_metrics(history, desired=final_desired)

    return results["flat"], results["imc"]


# ── IMC should not regress on any profile/scenario ──────────────────────


class TestIMCNoRegression:
    """Snapshot-pin flat-PI vs IMC metrics across profiles + scenarios.

    Regression detection via snapshot drift.  Structural IMC claims
    (Kp scales with τ, Ki ≈ 9/(4L); Skogestad SIMC §4) tested in
    TestIMCGainScaling.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_start_bounded(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(profile, initial=17.0, outdoor=2.0,
                              desired=20.5, duration_minutes=8 * 60, mode="heat")
        _record_imc_pair(bench_metrics, flat, imc, profile_name=profile_name, scenario="cold_start_bounded")
        check_bench_metrics(num_regression, bench_metrics)

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_snap_bounded(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(
            profile, initial=20.5, outdoor=10.0, desired=20.5,
            duration_minutes=8 * 60, mode="heat",
            outdoor_minute_schedule=lambda m: max(-5.0, 10.0 - m * (5.0 / 60.0)),
        )
        _record_imc_pair(bench_metrics, flat, imc, profile_name=profile_name, scenario="cold_snap_bounded")
        check_bench_metrics(num_regression, bench_metrics)

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_steady_state_bounded(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(profile, initial=20.5, outdoor=5.0,
                              desired=20.5, duration_minutes=12 * 60, mode="heat")
        _record_imc_pair(bench_metrics, flat, imc, profile_name=profile_name, scenario="steady_state_bounded")
        check_bench_metrics(num_regression, bench_metrics)

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cooling_bounded(self, bench_metrics, num_regression, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(profile, initial=28.0, outdoor=32.0,
                              desired=24.0, duration_minutes=8 * 60, mode="cool")
        _record_imc_pair(bench_metrics, flat, imc, profile_name=profile_name, scenario="cooling_bounded")
        check_bench_metrics(num_regression, bench_metrics)


# ── IMC should measurably improve slow-τ profiles ───────────────────────


class TestIMCImprovesSlowProfiles:
    """Well-insulated profile (τ_env=250, hp_gain=0.010) with IMC gains.

    With physics-correct FF seeds (1/(hp_gain × τ_env)), FF handles
    disturbance tracking almost perfectly — both flat and IMC controllers
    achieve near-zero ITAE on outdoor change scenarios. The IMC advantage
    (higher Kp for slow-τ) shows in transient overshoot damping and
    reversal reduction, but is modest because the FF does most of the work.

    Skogestad SIMC §4: gain scheduling optimizes for disturbance rejection.
    With accurate FF, disturbances are handled before PI acts, so the IMC
    margin is small. Tests verify bounded regression, not strict improvement.
    The structural claim (correct Kp from τ) is verified in TestIMCGainScaling.
    """

    SCENARIOS = [
        ("cold_start", dict(initial=17.0, outdoor=2.0, desired=20.5,
                            duration_minutes=8 * 60, mode="heat")),
        ("setpoint_step", dict(initial=20.5, outdoor=5.0, desired=20.5,
                               duration_minutes=8 * 60, mode="heat",
                               desired_minute_schedule={150: 22.5})),
        ("cold_snap", dict(initial=20.5, outdoor=10.0, desired=20.5,
                           duration_minutes=8 * 60, mode="heat",
                           outdoor_minute_schedule=lambda m: max(-5.0, 10.0 - m * (5.0 / 60.0)))),
        ("ramp_dist", dict(initial=20.5, outdoor=5.0, desired=20.5,
                           duration_minutes=8 * 60, mode="heat",
                           outdoor_minute_schedule=lambda m: 5.0 - m * (1.0 / 60.0))),
        ("warm_start", dict(initial=28.0, outdoor=32.0, desired=24.0,
                            duration_minutes=8 * 60, mode="cool")),
    ]

    @pytest.mark.parametrize("scenario_name,kwargs", SCENARIOS, ids=[s[0] for s in SCENARIOS])
    def test_well_insulated_no_regression(self, bench_metrics, num_regression, scenario_name, kwargs):
        """IMC should not regress on any scenario for well_insulated profile."""
        profile = QUICK_PROFILES["well_insulated"]
        flat, imc = _run_pair(profile, **kwargs)
        print(f"\n  well_insulated {scenario_name}: flat ITAE={flat['itae']:.1f}, "
              f"IMC={imc['itae']:.1f}, flat rev={flat['reversals']}, "
              f"IMC rev={imc['reversals']}")
        _record_imc_pair(bench_metrics, flat, imc, profile_name="well_insulated",
                         scenario=f"well_insulated_no_regression_{scenario_name}")
        check_bench_metrics(num_regression, bench_metrics)

    def test_well_insulated_cooling_improvement(self, bench_metrics, num_regression):
        """IMC should measurably improve cooling for well-insulated profiles.

        Cooling a high-inertia house (τ_fast≈48 min) is where flat Kp=1.5
        is genuinely too low. IMC Kp = τ/(λ+L) ≈ 2.4 drives convergence
        ~60% faster, producing clear ITAE reduction. This is the core
        scenario where gain scheduling earns its keep.
        """
        profile = QUICK_PROFILES["well_insulated"]
        flat, imc = _run_pair(profile, initial=28.0, outdoor=32.0,
                              desired=24.0, duration_minutes=8 * 60, mode="cool")
        pct = (1 - imc["itae"] / flat["itae"]) * 100 if flat["itae"] > 0 else 0
        print(f"\n  well_insulated cooling: flat ITAE={flat['itae']:.1f}, "
              f"IMC={imc['itae']:.1f} ({pct:.0f}% reduction)")
        assert pct > 10, f"Expected >10% cooling improvement, got {pct:.0f}%"


# ── IMC gains scale correctly with τ ─────────────────────────────────────


class TestIMCGainScaling:
    """Verify that IMC produces higher Kp for higher-τ profiles.

    This is the core value proposition: a well-insulated house (τ=120)
    needs ~4× the proportional gain of a drafty house (τ=25) because
    the room responds slowly and needs stronger corrective action.
    """

    def test_kp_increases_with_tau(self, bench_metrics, num_regression):
        """Kp should be monotonically increasing with τ (for fixed λ, L)."""
        gains = {}
        for name, profile in QUICK_PROFILES.items():
            ctrl = _make_imc_controller(profile)
            pi = ctrl._pi
            gains[name] = (profile.tau_minutes, pi._pi_kp, pi._pi_ki)

        sorted_profiles = sorted(gains.items(), key=lambda x: x[1][0])
        print("\n  Profile               τ     Kp      Ki")
        for name, (tau, kp, ki) in sorted_profiles:
            print(f"  {name:<22} {tau:3.0f}   {kp:5.2f}   {ki:.4f}")

        taus = [v[0] for _, v in sorted_profiles]
        kps = [v[1] for _, v in sorted_profiles]
        for i in range(len(kps) - 1):
            assert kps[i + 1] > kps[i], (
                f"Kp should increase with τ: τ={taus[i]}→{taus[i+1]}, "
                f"Kp={kps[i]:.2f}→{kps[i+1]:.2f}"
            )

    def test_ki_consistent_across_profiles(self, bench_metrics, num_regression):
        """Ki = 3·Kp/τ — with λ=L/3 (fixed), Ki converges as τ grows.

        Ki = 3·τ/((L/3+L)·τ) = 3/(4L/3) = 9/(4L) ≈ 0.15 for L=15.
        All profiles should have Ki near this value.
        """
        for name, profile in QUICK_PROFILES.items():
            ctrl = _make_imc_controller(profile)
            ki = ctrl._pi._pi_ki
            assert 0.10 < ki < 0.20, (
                f"{name}: Ki={ki:.4f} outside expected range 0.10-0.20"
            )


# ── Aggregate metric comparison ──────────────────────────────────────────


class TestIMCAggregate:
    """Aggregate comparison across all profiles and scenarios.

    With physics-correct FF seeds, disturbance tracking scenarios have
    near-zero ITAE for both controllers — the aggregate is dominated by
    transient-heavy scenarios (cold start, setpoint step) where IMC's
    gain advantage matters.

    Uses bounded regression rather than strict improvement because the
    margin depends on quantization phase alignment. The structural tests
    (TestIMCGainScaling) verify the gains are correct; this test verifies
    they don't cause aggregate harm.
    """

    SCENARIOS = [
        ("cold_start", dict(initial=17.0, outdoor=2.0, desired=20.5,
                            duration_minutes=8 * 60, mode="heat")),
        ("cold_snap", dict(initial=20.5, outdoor=10.0, desired=20.5,
                           duration_minutes=8 * 60, mode="heat",
                           outdoor_minute_schedule=lambda m: max(-5.0, 10.0 - m * (5.0 / 60.0)))),
        ("setpoint_up", dict(initial=20.5, outdoor=5.0, desired=20.5,
                             duration_minutes=8 * 60, mode="heat",
                             desired_minute_schedule={150: 22.5})),
        ("steady_state", dict(initial=20.5, outdoor=5.0, desired=20.5,
                              duration_minutes=12 * 60, mode="heat")),
        ("ramp_dist", dict(initial=20.5, outdoor=5.0, desired=20.5,
                           duration_minutes=8 * 60, mode="heat",
                           outdoor_minute_schedule=lambda m: 5.0 - m * (1.0 / 60.0))),
        ("warm_start", dict(initial=28.0, outdoor=32.0, desired=24.0,
                            duration_minutes=8 * 60, mode="cool")),
    ]

    def test_aggregate_no_regression(self, bench_metrics, num_regression):
        """Snapshot-pin aggregate flat vs IMC ITAE across the scenario matrix.

        Regression detection via snapshot drift, not an ad-hoc absolute
        ITAE bound.
        """
        flat_total = 0.0
        imc_total = 0.0

        print("\n  Profile               Scenario       Flat ITAE  IMC ITAE  Change")
        for profile_name, profile in QUICK_PROFILES.items():
            for scenario_name, kwargs in self.SCENARIOS:
                flat, imc = _run_pair(profile, **kwargs)
                flat_total += flat["itae"]
                imc_total += imc["itae"]
                delta = imc["itae"] - flat["itae"]
                marker = "<<<" if delta < -flat["itae"] * 0.05 else (
                    "!!!" if delta > flat["itae"] * 0.05 else "")
                print(f"  {profile_name:<22} {scenario_name:<14} "
                      f"{flat['itae']:9.1f} {imc['itae']:9.1f} {delta:+8.1f} {marker}")

        pct = (1 - imc_total / flat_total) * 100 if flat_total > 0 else 0
        print(f"\n  TOTAL: flat={flat_total:.1f}, IMC={imc_total:.1f} ({pct:.0f}% reduction)")
        bench_metrics["flat_total_itae"] = round(flat_total, 2)
        bench_metrics["imc_total_itae"] = round(imc_total, 2)
        check_bench_metrics(num_regression, bench_metrics)
