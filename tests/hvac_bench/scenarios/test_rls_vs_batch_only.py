"""Comparison: online RLS + batch WLS vs batch WLS only.

Tests whether the online RLS tick-by-tick learning adds value beyond
what the 12h batch WLS provides.  Runs identical scenarios with
_rls_online_learning=True vs False and compares:
- Coefficient convergence speed (batches to reach <10% error)
- Coefficient trajectory stability (oscillation)
- Comfort (% time in deadband)
- Integral RMS (lower = better FF)

VERDICT (captured in project_online_rls_verdict.md, 2026-04-26): online
RLS adds no value across 5 scenarios.  The assertions in this file are
intentionally weak ("not catastrophically worse") — they're regression-
guard floors, not strict pass criteria.  The win/loss verdict comes from
the printed metrics, not the asserts.  Don't strengthen the assertions
to match the verdict; the win is small (1.3% comfort delta in the most
favorable RLS case) and tightening would cause flakes.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.hvac_bench.full_stack_runner import (
    FullStackConfig, FullStackResult, ModelInputSpec,
    diurnal_solar, run_full_stack,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C


@dataclass
class ComparisonResult:
    """Side-by-side results from online RLS vs batch-only."""

    rls_on: FullStackResult
    rls_off: FullStackResult
    config_name: str


def _run_comparison(config: FullStackConfig, name: str) -> ComparisonResult:
    """Run same config with online RLS enabled vs disabled."""
    # Run with online RLS (default behavior)
    result_on = run_full_stack(config)

    # Run with online RLS disabled
    from tests.hvac_bench.adapters import TasmotaPIAdapter
    # Monkey-patch the adapter to disable online learning
    _orig_init = TasmotaPIAdapter.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._pi._rls_online_learning = False

    TasmotaPIAdapter.__init__ = _patched_init
    try:
        result_off = run_full_stack(config)
    finally:
        TasmotaPIAdapter.__init__ = _orig_init

    return ComparisonResult(rls_on=result_on, rls_off=result_off, config_name=name)


def _print_comparison(c: ComparisonResult) -> None:
    """Print side-by-side metrics."""
    print(f"\n{'='*60}")
    print(f"  {c.config_name}")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'RLS ON':>12} {'RLS OFF':>12} {'Delta':>10}")
    print(f"{'-'*60}")
    print(f"{'Comfort % (total)':<30} {c.rls_on.comfort_hours_pct:>11.1f}% {c.rls_off.comfort_hours_pct:>11.1f}% {c.rls_off.comfort_hours_pct - c.rls_on.comfort_hours_pct:>+9.1f}%")
    print(f"{'Comfort % (controllable)':<30} {c.rls_on.ctrl_comfort_pct:>11.1f}% {c.rls_off.ctrl_comfort_pct:>11.1f}% {c.rls_off.ctrl_comfort_pct - c.rls_on.ctrl_comfort_pct:>+9.1f}%")
    print(f"{'Integral RMS':<30} {c.rls_on.integral_rms:>12.3f} {c.rls_off.integral_rms:>12.3f} {c.rls_off.integral_rms - c.rls_on.integral_rms:>+10.3f}")
    print(f"{'Total ITAE':<30} {c.rls_on.total_itae:>12.1f} {c.rls_off.total_itae:>12.1f} {c.rls_off.total_itae - c.rls_on.total_itae:>+10.1f}")
    print(f"{'Longest violation streak':<30} {c.rls_on.longest_violation_streak:>12d} {c.rls_off.longest_violation_streak:>12d} {c.rls_off.longest_violation_streak - c.rls_on.longest_violation_streak:>+10d}")
    print(f"{'Batches to converge':<30} {str(c.rls_on.batches_to_converge):>12} {str(c.rls_off.batches_to_converge):>12}")
    print(f"{'Total reversals':<30} {c.rls_on.total_reversals:>12d} {c.rls_off.total_reversals:>12d} {c.rls_off.total_reversals - c.rls_on.total_reversals:>+10d}")

    # Coefficient errors
    print(f"\n{'Coefficient errors vs truth:'}")
    for name in c.rls_on.coef_errors:
        err_on = c.rls_on.coef_errors.get(name, 0)
        err_off = c.rls_off.coef_errors.get(name, 0)
        print(f"  {name:<24} {err_on:>10.4f} {err_off:>10.4f} {err_off - err_on:>+10.4f}")

    # Weekly comfort progression
    if c.rls_on.daily_comfort_pct and c.rls_off.daily_comfort_pct:
        print(f"\n{'Weekly comfort progression:'}")
        n_days = min(len(c.rls_on.daily_comfort_pct), len(c.rls_off.daily_comfort_pct))
        for week_start in range(0, n_days, 7):
            week_end = min(week_start + 7, n_days)
            on_avg = sum(c.rls_on.daily_comfort_pct[week_start:week_end]) / (week_end - week_start)
            off_avg = sum(c.rls_off.daily_comfort_pct[week_start:week_end]) / (week_end - week_start)
            print(f"  Days {week_start+1}-{week_end}: ON={on_avg:.1f}%  OFF={off_avg:.1f}%  delta={off_avg-on_avg:+.1f}%")

    # Coefficient trajectory (per-batch outdoor_delta)
    if c.rls_on.coef_trajectory and c.rls_off.coef_trajectory:
        print(f"\n{'outdoor_delta trajectory (first 10 batches):'}")
        for i in range(min(10, len(c.rls_on.coef_trajectory), len(c.rls_off.coef_trajectory))):
            od_on = c.rls_on.coef_trajectory[i].get("outdoor_delta", 0)
            od_off = c.rls_off.coef_trajectory[i].get("outdoor_delta", 0)
            print(f"  Batch {i+1:2d}: ON={od_on:>8.4f}  OFF={od_off:>8.4f}")


# ── Scenario configs ─────────────────────────────────────────────────────


def _correct_seed_config(n_days: int = 30) -> FullStackConfig:
    """Living room with correct seeds — tests steady-state behavior."""
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
                _true_ff_coef=-2.0,
                seed_heat=0.0,
                lag_tau=120,
                clamp_min=0,
                schedule=lambda t: diurnal_solar(t),
            ),
        ],
        pi_overrides={
            "pi_outdoor_seed_heat": profile.true_seed,
        },
        relax_kappa_gate=True,
    )


def _wrong_seed_config(n_days: int = 30) -> FullStackConfig:
    """Living room with wrong seeds — tests convergence from bad start."""
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
                _true_ff_coef=-2.0,
                seed_heat=0.0,
                lag_tau=120,
                clamp_min=0,
                schedule=lambda t: diurnal_solar(t),
            ),
        ],
        pi_overrides={
            # Wrong: 2x true value
            "pi_outdoor_seed_heat": profile.true_seed * 2.0,
        },
        relax_kappa_gate=True,
    )


def _bunkroom_config(n_days: int = 30) -> FullStackConfig:
    """Bunkroom (slow thermal mass) — tests slow learner scenario."""
    profile = PROFILES_2R2C["bunkroom"]
    return FullStackConfig(
        n_days=n_days,
        profile_name="bunkroom",
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
                _true_ff_coef=-1.5,
                seed_heat=0.0,
                lag_tau=120,
                clamp_min=0,
                schedule=lambda t: diurnal_solar(t),
            ),
        ],
        pi_overrides={
            "pi_outdoor_seed_heat": profile.true_seed,
        },
        relax_kappa_gate=True,
    )


# ── Test class ───────────────────────────────────────────────────────────


class TestOnlineRLSValue:
    """Compare online RLS + batch vs batch-only across scenarios."""

    def test_correct_seeds(self):
        """With correct seeds, online RLS should not degrade performance."""
        c = _run_comparison(_correct_seed_config(), "Correct Seeds (30 days)")
        _print_comparison(c)

        # Batch-only should be at least as good (no worse than 5% controllable comfort)
        assert c.rls_off.ctrl_comfort_pct >= c.rls_on.ctrl_comfort_pct - 5.0

    def test_wrong_seeds(self):
        """With wrong seeds, check if online RLS helps early convergence."""
        c = _run_comparison(_wrong_seed_config(), "Wrong Seeds 2x (30 days)")
        _print_comparison(c)

        # Just collect data — no hard assertion on which is better
        assert c.rls_off.ctrl_comfort_pct > 50.0  # sanity check

    def test_bunkroom_slow(self):
        """Slow thermal mass — does online RLS help or hurt?"""
        c = _run_comparison(_bunkroom_config(), "Bunkroom Slow (30 days)")
        _print_comparison(c)

        assert c.rls_off.ctrl_comfort_pct > 50.0  # sanity check
