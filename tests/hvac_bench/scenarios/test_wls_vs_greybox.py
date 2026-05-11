"""Comparison: WLS-only vs WLS+grey-box fused.

Uses a spring scenario where outdoor temp is mild enough that the HP
cycles on/off naturally — essential for grey-box to have HP-off data
to separate ua_c from k_c.

The grey-box observer needs both HP-on (to identify k_c) and HP-off
(to identify ua_c) observations.  Pure winter heating rarely provides
HP-off data; spring/shoulder season does.

History note: this file previously had a third arm (``gb_only``) using an
inline simulation loop that monkey-patched ``pi._run_batch_analysis`` to
revert WLS coefficient writes. That arm was dropped (2026-05-06) when
migrating to ``full_stack_runner``: the verdict memory established that
gb_only rails the same way wls_only and fused do — it didn't carry
diagnostic information the other arms didn't already provide. The inline
loop also hardcoded ``solar_gain=0.06`` while asserting ``solar_true=-2.0``,
producing a physics inconsistency (β_solar implied by physics is
−solar_gain/hp_gain = −1.5, not −2.0). Migrating to
``full_stack_runner`` fixes the inconsistency: ``ModelInputSpec.resolve()``
derives ``solar_thermal_gain = |β_solar_truth| × hp_gain`` so the model
and the asserted truth agree.

Validation harness for #47 (2R2C grey-box upgrade); not a current verdict
producer because 1R1C grey-box rails at τ=1000 and tau_plausible blocks
every batch. See ``project_greybox_1r1c_limitation.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    ModelInputSpec,
    diurnal_outdoor,
    diurnal_solar,
    run_full_stack,
    TICK_MINUTES_DEFAULT,
)
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.house_profiles import PROFILES_2R2C


def _spring_outdoor(tick: int) -> float:
    """Late spring weather: warm base (16°C) with ±8°C diurnal + weather fronts.

    Afternoons regularly exceed desired (20.5°C). Combined with solar,
    HP should be off 30-50% of the time — matching production patterns.
    With 5-day weather drift ±8°C, warm spells push outdoor to 32°C.
    """
    return diurnal_outdoor(tick, base_c=16.0, amplitude_c=8.0)


def _spring_solar(tick: int) -> float:
    """Strong spring solar (peak 0.8)."""
    return diurnal_solar(tick, peak=0.8)


@dataclass
class ArmResult:
    comfort: float
    integral_rms: float
    od_error: float
    solar_error: float
    coef_trajectory: list[dict]
    gb_gates_passed: int
    gb_total: int
    final_outdoor_beta: float
    final_solar_beta: float


def _run_arm(mode: str, n_days: int = 60) -> ArmResult:
    """Run spring scenario via full_stack_runner.

    mode: "wls_only" (no greybox blending) or "fused" (greybox blending on).
    """
    profile = PROFILES_2R2C["living_room"]

    gb_gates_passed = [0]
    gb_total = [0]

    def _on_batch(batch_idx: int, pi: object) -> None:
        gb_total[0] += 1
        bridge = getattr(pi, "_last_greybox_bridge", None)
        if bridge is not None and bridge.gates_passed:
            gb_gates_passed[0] += 1

    config = FullStackConfig(
        n_days=n_days,
        profile_name="living_room",
        outdoor_base_c=16.0,
        outdoor_diurnal_c=8.0,
        desired_c=20.5,
        mode="heat",
        noise_sigma=0.1,
        noise_seed=42,
        tick_minutes=TICK_MINUTES_DEFAULT,
        outdoor_schedule=_spring_outdoor,
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
                schedule=_spring_solar,
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

    return ArmResult(
        comfort=result.comfort_hours_pct,
        integral_rms=result.integral_rms,
        od_error=abs(od_final - od_truth),
        solar_error=abs(solar_final - solar_truth),
        coef_trajectory=result.coef_trajectory,
        gb_gates_passed=gb_gates_passed[0],
        gb_total=gb_total[0],
        final_outdoor_beta=od_final,
        final_solar_beta=solar_final,
    )


@pytest.mark.design
class TestWLSvsGreybox:
    """Compare WLS-only vs WLS+grey-box fused estimators on synthetic spring.

    Marked ``design``: validation harness for #47 (2R2C grey-box upgrade);
    not a current verdict producer because 1R1C grey-box rails at τ=1000
    and tau_plausible blocks every batch. See
    ``project_greybox_1r1c_limitation.md``.
    """

    def test_spring_comparison(self, bench_metrics, num_regression):
        """Spring scenario with HP cycling — grey-box should have data."""
        wls = _run_arm("wls_only", n_days=60)
        fused = _run_arm("fused", n_days=60)

        print(f"\n{'=' * 70}")
        print("  WLS vs Grey-box-fused Comparison (Spring, 60 days)")
        print(f"{'=' * 70}")
        print(f"  Grey-box gates passed: WLS arm={wls.gb_gates_passed}/{wls.gb_total}, "
              f"Fused arm={fused.gb_gates_passed}/{fused.gb_total}")
        print(f"\n{'Metric':<25} {'WLS Only':>12} {'Fused':>12}")
        print("-" * 51)
        print(f"{'Comfort %':<25} {wls.comfort:>11.1f}% {fused.comfort:>11.1f}%")
        print(f"{'Integral RMS':<25} {wls.integral_rms:>12.3f} "
              f"{fused.integral_rms:>12.3f}")
        print(f"{'|outdoor_delta err|':<25} {wls.od_error:>12.4f} "
              f"{fused.od_error:>12.4f}")
        print(f"{'|Solar err|':<25} {wls.solar_error:>12.4f} "
              f"{fused.solar_error:>12.4f}")
        print(f"{'final β_outdoor':<25} {wls.final_outdoor_beta:>12.4f} "
              f"{fused.final_outdoor_beta:>12.4f}")
        print(f"{'final β_solar':<25} {wls.final_solar_beta:>12.4f} "
              f"{fused.final_solar_beta:>12.4f}")

        # Trajectory snapshot
        print("\noutdoor_delta trajectory (every 5th batch):")
        max_len = min(len(wls.coef_trajectory), len(fused.coef_trajectory))
        for i in range(0, max_len, 5):
            w = wls.coef_trajectory[i].get("outdoor_delta", 0)
            f = fused.coef_trajectory[i].get("outdoor_delta", 0)
            print(f"  Batch {i + 1:3d}: WLS={w:>8.4f}  Fused={f:>8.4f}")

        bench_metrics["wls_comfort"] = wls.comfort
        bench_metrics["fused_comfort"] = fused.comfort
        bench_metrics["wls_integral_rms"] = wls.integral_rms
        bench_metrics["fused_integral_rms"] = fused.integral_rms
        bench_metrics["wls_od_error"] = wls.od_error
        bench_metrics["fused_od_error"] = fused.od_error
        bench_metrics["wls_solar_error"] = wls.solar_error
        bench_metrics["fused_solar_error"] = fused.solar_error
        bench_metrics["wls_final_outdoor_beta"] = wls.final_outdoor_beta
        bench_metrics["fused_final_outdoor_beta"] = fused.final_outdoor_beta
        bench_metrics["wls_final_solar_beta"] = wls.final_solar_beta
        bench_metrics["fused_final_solar_beta"] = fused.final_solar_beta
        bench_metrics["wls_gb_gates_passed"] = wls.gb_gates_passed
        bench_metrics["fused_gb_gates_passed"] = fused.gb_gates_passed
        check_bench_metrics(num_regression, bench_metrics)

        # Sanity
        assert wls.comfort > 50.0
        assert fused.comfort > 50.0
