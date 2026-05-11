"""HP capacity curve validation (#43): cold weather drives saturation.

The fixed-gain bench underestimates how often a real ASHP rails in cold —
this is the regime Tobit (#40) is meant to handle. Adding a piecewise
linear capacity factor in outdoor_c makes the bench's saturation rate
behave like the production system, so Tobit's censoring fix can be
demonstrated and validated.

Compares two 30-day winter scenarios on the calibrated living_room
profile:
  - living_room              (no capacity scaling, fixed hp_gain)
  - living_room_capacity     (STANDARD_HP_CAPACITY: 0 at -15°C, 1.0 at 7°C)

Asserts that enabling the capacity curve materially increases the fraction
of ticks where the HP setpoint is at the rail.  Exact numbers depend on
the profile's headroom (the calibrated living_room is "marginal at -15°F"
per its description), so the test asserts a directional minimum rather
than a tight bracket.
"""

from __future__ import annotations

import logging

import pytest

from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.constants import TICK_MINUTES_DEFAULT
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    FullStackResult,
    ModelInputSpec,
    WeatherState,
    diurnal_outdoor,
    diurnal_solar,
    print_full_stack_summary,
    run_full_stack,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C


# Winter scenario shared by both runs.  Mirrors test_seasonal_convergence's
# winter season (base -5°C, diurnal ±4°C, peak solar 0.5).  14 days is
# enough for the saturation regime to dominate the daily-mean stat.
_WINTER_BASE_C = -5.0
_WINTER_DIURNAL_C = 4.0
_WINTER_SOLAR_PEAK = 0.5
_N_DAYS = 14


def _make_config(profile_name: str) -> FullStackConfig:
    """Build a winter FullStackConfig pinned to the given profile."""
    base_profile = PROFILES_2R2C["living_room"]  # for true_seed reference
    n_ticks = int(_N_DAYS * 24 * 60 / TICK_MINUTES_DEFAULT)
    weather = WeatherState(
        n_ticks=n_ticks, seed=42, persistence_hours=36.0,
        tick_minutes=TICK_MINUTES_DEFAULT,
    )
    return FullStackConfig(
        n_days=_N_DAYS,
        profile_name=profile_name,
        outdoor_base_c=_WINTER_BASE_C,
        outdoor_diurnal_c=_WINTER_DIURNAL_C,
        outdoor_schedule=lambda t: diurnal_outdoor(
            t, _WINTER_BASE_C, _WINTER_DIURNAL_C, TICK_MINUTES_DEFAULT, weather
        ),
        desired_c=20.5,
        noise_sigma=0.1,
        noise_seed=42,
        model_inputs=[
            ModelInputSpec(
                name="Solar Proxy",
                entity_id="sensor.solar_proxy",
                input_role="solar",
                _true_ff_coef=-2.0,
                lag_tau=120,
                clamp_min=0,
                schedule=lambda t: diurnal_solar(
                    t, peak=_WINTER_SOLAR_PEAK,
                    tick_minutes=TICK_MINUTES_DEFAULT,
                    weather_state=weather,
                ),
            ),
        ],
        pi_overrides={"pi_outdoor_seed_heat": base_profile.true_seed},
        relax_kappa_gate=True,
    )


def _saturation_pct(result: FullStackResult) -> float:
    """Mean % of ticks per day where setpoint is at min or max rail."""
    if not result.daily_setpoint_limited_pct:
        return 0.0
    return sum(result.daily_setpoint_limited_pct) / len(
        result.daily_setpoint_limited_pct
    )


def _compute_capacity_runs() -> dict[str, FullStackResult]:
    """Run both variants once. Shared by the pytest fixture and the CLI runner."""
    pi_logger = logging.getLogger("custom_components.tasmota_irhvac")
    prev = pi_logger.level
    pi_logger.setLevel(logging.ERROR)
    try:
        return {
            "fixed": run_full_stack(_make_config("living_room")),
            "capacity": run_full_stack(_make_config("living_room_capacity")),
        }
    finally:
        pi_logger.setLevel(prev)


def _print_capacity_summary(capacity_runs: dict[str, FullStackResult]) -> None:
    """Side-by-side fixed-vs-capacity saturation/comfort/cold-violations table."""
    sat_fixed = _saturation_pct(capacity_runs["fixed"])
    sat_cap = _saturation_pct(capacity_runs["capacity"])
    ctrl_fixed = capacity_runs["fixed"].ctrl_comfort_pct
    ctrl_cap = capacity_runs["capacity"].ctrl_comfort_pct
    cold_fixed = capacity_runs["fixed"].cold_violations
    cold_cap = capacity_runs["capacity"].cold_violations
    print(f"\n{'=' * 60}")
    print("  HP Capacity Curve: winter saturation, 14d, living_room")
    print(f"{'=' * 60}")
    print(f"{'Variant':<22}{'Saturation%':>14}{'CtrlComfort%':>14}"
          f"{'ColdViols':>12}")
    print("-" * 62)
    print(f"{'fixed (no capacity)':<22}{sat_fixed:>13.2f}%"
          f"{ctrl_fixed:>13.2f}%{cold_fixed:>12d}")
    print(f"{'capacity curve':<22}{sat_cap:>13.2f}%"
          f"{ctrl_cap:>13.2f}%{cold_cap:>12d}")
    print(f"{'Δ':<22}{sat_cap - sat_fixed:>+13.2f}pp"
          f"{ctrl_cap - ctrl_fixed:>+13.2f}pp"
          f"{cold_cap - cold_fixed:>+12d}")


def run_hp_capacity_summary() -> None:
    """CLI entry: per-run generic summary for each variant + capacity comparison."""
    runs = _compute_capacity_runs()
    for name, result in runs.items():
        print_full_stack_summary(result, label=f"hp_capacity / {name}")
    _print_capacity_summary(runs)


@pytest.fixture(scope="module")
def capacity_runs() -> dict[str, FullStackResult]:
    """Run both variants once for the whole test module."""
    return _compute_capacity_runs()


@pytest.mark.slow
class TestHPCapacityCurveSaturation:
    """Capacity curve raises winter saturation rate (#43 validation)."""

    def test_capacity_increases_saturation(self, bench_metrics, num_regression, capacity_runs):
        """Enabling the curve must raise the rail-time fraction.

        The bench's pre-#43 fixed-gain model rails ~2-5% in winter (PI
        almost always has setpoint headroom).  With STANDARD_HP_CAPACITY
        applied to the 'marginal at -15°F' calibrated living_room, real
        cold-snap behavior dominates and saturation jumps materially.
        Assert the directional change without pinning a specific number;
        the magnitude depends on the profile's reserve capacity.
        """
        sat_fixed = _saturation_pct(capacity_runs["fixed"])
        sat_cap = _saturation_pct(capacity_runs["capacity"])
        bench_metrics["sat_fixed_pct"] = sat_fixed
        bench_metrics["sat_capacity_pct"] = sat_cap
        bench_metrics["sat_delta_pp"] = sat_cap - sat_fixed
        bench_metrics["ctrl_comfort_fixed_pct"] = capacity_runs["fixed"].ctrl_comfort_pct
        bench_metrics["ctrl_comfort_capacity_pct"] = capacity_runs["capacity"].ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)
        assert sat_fixed < 10.0, (
            f"Pre-capacity saturation should be low (<10%): {sat_fixed:.2f}%"
        )
        assert sat_cap > sat_fixed + 10.0, (
            f"Capacity curve should raise saturation by >10pp: "
            f"fixed={sat_fixed:.2f}%, capacity={sat_cap:.2f}%"
        )

    def test_capacity_increases_cold_violations(self, bench_metrics, num_regression, capacity_runs):
        """Cold-side comfort violations should rise too (less control authority)."""
        cold_fixed = capacity_runs["fixed"].cold_violations
        cold_cap = capacity_runs["capacity"].cold_violations
        bench_metrics["cold_violations_fixed"] = cold_fixed
        bench_metrics["cold_violations_capacity"] = cold_cap
        bench_metrics["cold_violations_delta"] = cold_cap - cold_fixed
        check_bench_metrics(num_regression, bench_metrics)
        assert cold_cap > cold_fixed, (
            f"Capacity curve should produce more cold violations: "
            f"fixed={cold_fixed}, capacity={cold_cap}"
        )
