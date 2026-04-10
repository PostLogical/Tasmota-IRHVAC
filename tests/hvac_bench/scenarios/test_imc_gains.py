"""IMC gain scheduling validation tests.

Verifies that IMC-derived Kp/Ki (from profile τ) produces better control
metrics than flat Kp=1.5/Ki=0.15 across house profiles. The IMC mechanism
seeds τ, derives gains via Kp=τ/(λ+L), and adapts online. These tests
validate the static derivation against the bench thermal model.

Key insight from bench sweep (2026-04-09):
- Flat Kp=1.5 is near-optimal for drafty (τ=25) but 3-4× too low for
  well-insulated (τ=120). IMC corrects this automatically.
- Default λ=L/3≈5 gives 17% aggregate ITAE reduction, 0 regressions.
- Largest win: well-insulated 64% ITAE reduction, 90% on warm start.
"""

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import QUICK_PROFILES, HouseProfile
from tests.hvac_bench.thermal_model import ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


def _make_flat_controller(seed_factor=1.0):
    """Controller with flat (non-IMC) default gains."""
    return TasmotaPIAdapter({
        "pi_ff_heat_slope": 0.35 * seed_factor,
        "pi_ff_cool_slope": 0.35 * seed_factor,
    })


def _make_imc_controller(profile: HouseProfile, seed_factor=1.0):
    """Controller with IMC gains derived from profile τ."""
    return TasmotaPIAdapter({
        "pi_ff_heat_slope": 0.35 * seed_factor,
        "pi_ff_cool_slope": 0.35 * seed_factor,
        "pi_tau_estimate": float(profile.tau_minutes),
        "pi_response_lag": 15.0,
        # lambda=0 → uses default L/3
    })


def _run_pair(profile, initial, outdoor, desired, n_ticks, mode,
              outdoor_schedule=None, desired_schedule=None, seed_factor=1.0):
    """Run both flat and IMC controllers, return (flat_metrics, imc_metrics)."""
    final_desired = desired
    if desired_schedule:
        for _, temp in sorted(desired_schedule.items()):
            final_desired = temp

    results = {}
    for label, ctrl in [("flat", _make_flat_controller(seed_factor)),
                        ("imc", _make_imc_controller(profile, seed_factor))]:
        ctrl.set_desired_temp(desired)
        model = ThermalModel(profile=profile, initial_temp=initial,
                             outdoor_temp=outdoor)
        history = run_scenario(ctrl, model, n_ticks=n_ticks, mode=mode,
                               outdoor_schedule=outdoor_schedule,
                               desired_schedule=desired_schedule)
        results[label] = compute_all_metrics(history, desired=final_desired)

    return results["flat"], results["imc"]


# ── IMC should not regress on any profile/scenario ──────────────────────


class TestIMCNoRegression:
    """IMC gains must not produce catastrophically worse ITAE than flat.

    λ=L/3 is a compromise: optimal for slow-τ houses, slightly worse for
    fast-τ in some scenarios. Small regressions (<50% or <5 ITAE points)
    are acceptable trade-offs when the aggregate is a 17% win. These
    bounds catch real problems (wrong formula, sign errors) without
    over-fitting to quantization phase alignment.
    """

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_start_bounded(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(profile, initial=17.0, outdoor=2.0,
                              desired=20.5, n_ticks=32, mode="heat")
        assert imc["itae"] <= flat["itae"] * 1.50 + 5.0, (
            f"{profile_name}: IMC ITAE {imc['itae']:.1f} vs flat {flat['itae']:.1f}"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cold_snap_bounded(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(
            profile, initial=20.5, outdoor=10.0, desired=20.5,
            n_ticks=32, mode="heat",
            outdoor_schedule=lambda t: max(-5.0, 10.0 - t * 1.25),
        )
        assert imc["itae"] <= flat["itae"] * 1.50 + 5.0, (
            f"{profile_name}: IMC ITAE {imc['itae']:.1f} vs flat {flat['itae']:.1f}"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_steady_state_bounded(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(profile, initial=20.5, outdoor=5.0,
                              desired=20.5, n_ticks=48, mode="heat")
        assert imc["itae"] <= flat["itae"] * 1.50 + 5.0, (
            f"{profile_name}: IMC ITAE {imc['itae']:.1f} vs flat {flat['itae']:.1f}"
        )

    @pytest.mark.parametrize("profile_name", QUICK_PROFILES.keys())
    def test_cooling_bounded(self, profile_name):
        profile = QUICK_PROFILES[profile_name]
        flat, imc = _run_pair(profile, initial=28.0, outdoor=32.0,
                              desired=24.0, n_ticks=32, mode="cool")
        assert imc["itae"] <= flat["itae"] * 1.50 + 5.0, (
            f"{profile_name}: IMC ITAE {imc['itae']:.1f} vs flat {flat['itae']:.1f}"
        )


# ── IMC should measurably improve slow-τ profiles ───────────────────────


class TestIMCImprovesSlowProfiles:
    """Well-insulated (τ=120) benefits most from IMC — flat Kp=1.5 is
    dramatically too low. Assert meaningful ITAE improvement.

    Thresholds from bench sweep: well_insulated sees 35-94% ITAE
    reduction depending on scenario. We assert ≥20% to leave margin.
    """

    def test_cold_start_improvement(self):
        profile = QUICK_PROFILES["well_insulated"]
        flat, imc = _run_pair(profile, initial=17.0, outdoor=2.0,
                              desired=20.5, n_ticks=32, mode="heat")
        pct = (1 - imc["itae"] / flat["itae"]) * 100 if flat["itae"] > 0 else 0
        print(f"\n  well_insulated cold_start: flat ITAE={flat['itae']:.1f}, "
              f"IMC={imc['itae']:.1f} ({pct:.0f}% reduction)")
        assert pct > 20, f"Expected >20% improvement, got {pct:.0f}%"

    def test_cold_snap_improvement(self):
        profile = QUICK_PROFILES["well_insulated"]
        flat, imc = _run_pair(
            profile, initial=20.5, outdoor=10.0, desired=20.5,
            n_ticks=32, mode="heat",
            outdoor_schedule=lambda t: max(-5.0, 10.0 - t * 1.25),
        )
        pct = (1 - imc["itae"] / flat["itae"]) * 100 if flat["itae"] > 0 else 0
        print(f"\n  well_insulated cold_snap: flat ITAE={flat['itae']:.1f}, "
              f"IMC={imc['itae']:.1f} ({pct:.0f}% reduction)")
        assert pct > 20, f"Expected >20% improvement, got {pct:.0f}%"

    def test_warm_start_improvement(self):
        profile = QUICK_PROFILES["well_insulated"]
        flat, imc = _run_pair(profile, initial=28.0, outdoor=32.0,
                              desired=24.0, n_ticks=32, mode="cool")
        pct = (1 - imc["itae"] / flat["itae"]) * 100 if flat["itae"] > 0 else 0
        print(f"\n  well_insulated warm_start: flat ITAE={flat['itae']:.1f}, "
              f"IMC={imc['itae']:.1f} ({pct:.0f}% reduction)")
        assert pct > 20, f"Expected >20% improvement, got {pct:.0f}%"

    def test_ramp_disturbance_improvement(self):
        profile = QUICK_PROFILES["well_insulated"]
        flat, imc = _run_pair(
            profile, initial=20.5, outdoor=5.0, desired=20.5,
            n_ticks=32, mode="heat",
            outdoor_schedule=lambda t: 5.0 - t * 0.25,
        )
        pct = (1 - imc["itae"] / flat["itae"]) * 100 if flat["itae"] > 0 else 0
        print(f"\n  well_insulated ramp_dist: flat ITAE={flat['itae']:.1f}, "
              f"IMC={imc['itae']:.1f} ({pct:.0f}% reduction)")
        assert pct > 20, f"Expected >20% improvement, got {pct:.0f}%"


# ── IMC gains scale correctly with τ ─────────────────────────────────────


class TestIMCGainScaling:
    """Verify that IMC produces higher Kp for higher-τ profiles.

    This is the core value proposition: a well-insulated house (τ=120)
    needs ~4× the proportional gain of a drafty house (τ=25) because
    the room responds slowly and needs stronger corrective action.
    """

    def test_kp_increases_with_tau(self):
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

    def test_ki_consistent_across_profiles(self):
        """Ki = 3·Kp/τ — with λ=L/3 (fixed), Ki converges as τ grows.

        Ki = 3·τ/((L/3+L)·τ) = 3/(4L/3) = 9/(4L) ≈ 0.15 for L=15.
        All profiles should have Ki near this value.
        """
        for name, profile in QUICK_PROFILES.items():
            ctrl = _make_imc_controller(profile)
            ki = ctrl._pi._pi_ki
            # Ki should be in a reasonable range (not wildly different per profile)
            assert 0.10 < ki < 0.20, (
                f"{name}: Ki={ki:.4f} outside expected range 0.10-0.20"
            )


# ── Aggregate metric comparison ──────────────────────────────────────────


class TestIMCAggregate:
    """Aggregate ITAE across all profiles and scenarios.

    IMC should produce lower total ITAE than flat gains. This is the
    summary statistic: gain scheduling helps overall, not just for one
    profile or scenario.
    """

    SCENARIOS = [
        ("cold_start", dict(initial=17.0, outdoor=2.0, desired=20.5, n_ticks=32, mode="heat")),
        ("cold_snap", dict(initial=20.5, outdoor=10.0, desired=20.5, n_ticks=32, mode="heat",
                           outdoor_schedule=lambda t: max(-5.0, 10.0 - t * 1.25))),
        ("setpoint_up", dict(initial=20.5, outdoor=5.0, desired=20.5, n_ticks=32, mode="heat",
                             desired_schedule={10: 22.5})),
        ("steady_state", dict(initial=20.5, outdoor=5.0, desired=20.5, n_ticks=48, mode="heat")),
        ("ramp_dist", dict(initial=20.5, outdoor=5.0, desired=20.5, n_ticks=32, mode="heat",
                           outdoor_schedule=lambda t: 5.0 - t * 0.25)),
        ("warm_start", dict(initial=28.0, outdoor=32.0, desired=24.0, n_ticks=32, mode="cool")),
    ]

    def test_aggregate_itae_improvement(self):
        """Total ITAE across all profiles × scenarios should be ≥10% lower with IMC."""
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
        assert pct > 10, f"Expected >10% aggregate ITAE improvement, got {pct:.0f}%"
