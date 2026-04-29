"""Buffer variant sweep against real open-meteo weather (NE, 90 days/season).

Mirror of ``test_buffer_variants.py`` but with synthetic schedules replaced
by real hourly weather from ``new_england_{fall,winter,spring}_90d.csv``.

Same coordinates (44°N, 71.5°W) and same synthetic-test buffer variants:
  - default-2000 (leverage)
  - half-1000    (leverage)
  - double-4000  (leverage)
  - FIFO-2000    (oldest-eviction)

Goal: confirm or refute the synthetic finding that leverage-scored eviction
biases the Solar Proxy slope toward zero. Real weather has fronts, partly
cloudy days, dawn/dusk gradients, and seasonal day-length shifts that the
synthetic profiles approximate with sinusoids — those distributional
differences are exactly what a leverage policy could be sensitive to.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.hvac_bench.csv_adapters import csv_to_schedules, from_open_meteo_csv
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    FullStackResult,
    ModelInputSpec,
    run_full_stack,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.scenarios.test_buffer_variants import (
    VARIANTS,
    _run_with_variant,  # we'll override the inner config builder
    FIFOBuffer,
    _replace_buffers,
)
from tests.hvac_bench.adapters import TasmotaPIAdapter


_WEATHER_DIR = Path(__file__).parent.parent / "weather_data"

REAL_SEASONS = ["fall", "winter", "spring"]


# ── Config builder for real-weather runs ────────────────────────────────


def _make_real_config(season: str, n_days: int = 90) -> FullStackConfig:
    """Build FullStackConfig driven by the open-meteo CSV for this season."""
    csv_path = _WEATHER_DIR / f"new_england_{season}_90d.csv"
    csv_data = from_open_meteo_csv(csv_path)
    schedules = csv_to_schedules(csv_data)

    outdoor_fn = schedules["outdoor_c"]
    raw_solar_fn = schedules["solar_w_m2"]
    # Open-meteo direct_radiation is W/m² (peak ~700-1000); normalize to 0-1
    # proxy to match the synthetic convention the thermal model expects.
    solar_fn = lambda t, _f=raw_solar_fn: _f(t) / 1000.0

    n_hours = len(csv_data.get("outdoor_c", [])) - 1
    n_days_max = max(1, n_hours // 24)
    n_days = min(n_days, n_days_max)

    profile = PROFILES_2R2C["living_room"]
    return FullStackConfig(
        n_days=n_days,
        profile_name="living_room",
        desired_c=20.5,
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
        pi_overrides={
            # 2× wrong outdoor seed — same as synthetic test
            "pi_outdoor_seed_heat": profile.true_seed * 2.0,
        },
        relax_kappa_gate=True,
    )


def _run_real_variant(season: str, *, max_size: int, fifo: bool,
                      n_days: int = 90) -> FullStackResult:
    """Run a real-weather season with a patched buffer."""
    orig_init = TasmotaPIAdapter.__init__

    def patched(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _replace_buffers(self._pi, max_size=max_size, fifo=fifo)

    TasmotaPIAdapter.__init__ = patched
    try:
        return run_full_stack(_make_real_config(season, n_days=n_days))
    finally:
        TasmotaPIAdapter.__init__ = orig_init


# ── Fixture ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_variant_results() -> dict[str, dict[str, FullStackResult]]:
    """Run all (variant × season) combos once. 12 sims, ~8 minutes."""
    pi_logger = logging.getLogger("custom_components.tasmota_irhvac")
    prev = pi_logger.level
    pi_logger.setLevel(logging.ERROR)
    try:
        out: dict[str, dict[str, FullStackResult]] = {}
        for vname, size, fifo in VARIANTS:
            out[vname] = {}
            for season in REAL_SEASONS:
                out[vname][season] = _run_real_variant(
                    season, max_size=size, fifo=fifo, n_days=90,
                )
        return out
    finally:
        pi_logger.setLevel(prev)


# ── Pretty printing ──────────────────────────────────────────────────────


def _print_final_table(
    results: dict[str, dict[str, FullStackResult]],
    coef: str,
    truth: float,
) -> None:
    print(f"\n=== Real-weather final {coef} (truth ≈ {truth:.2f}) ===")
    header = f"{'Variant':<16}" + "".join(f"{s:>11}" for s in REAL_SEASONS) + f"{'Spread':>10}"
    print(header)
    print("-" * len(header))
    for vname in results:
        vals = [results[vname][s].final_coefs.get(coef, 0.0) for s in REAL_SEASONS]
        spread = max(vals) - min(vals)
        row = "".join(f"{v:>11.4f}" for v in vals)
        print(f"{vname:<16}{row}{spread:>10.4f}")


def _print_drift_table(
    results: dict[str, dict[str, FullStackResult]],
    coef: str,
) -> None:
    days = [15, 30, 45, 60, 75, 90]
    print(f"\n=== Real-weather {coef} trajectory (day 15 → day 90) ===")
    header = f"{'Variant':<16}{'Season':<10}" + "".join(f"d{d:>3}".rjust(11) for d in days)
    print(header)
    print("-" * len(header))
    for vname in results:
        for s in REAL_SEASONS:
            traj = results[vname][s].coef_trajectory
            row = []
            for d in days:
                target = int(d * 2)
                if not traj:
                    row.append(float("nan"))
                else:
                    idx = min(target, len(traj) - 1)
                    row.append(traj[idx].get(coef, 0.0))
            row_s = "".join(f"{v:>11.4f}" for v in row)
            print(f"{vname:<16}{s:<10}{row_s}")
        print()


def _print_buffer_fill(results: dict[str, dict[str, FullStackResult]]) -> None:
    print(f"\n=== Real-weather buffer fill day (first 100% utilization) ===")
    header = f"{'Variant':<16}" + "".join(f"{s:>11}" for s in REAL_SEASONS)
    print(header)
    print("-" * len(header))
    for vname in results:
        row = []
        for s in REAL_SEASONS:
            r = results[vname][s]
            day_full = next(
                (i for i, u in enumerate(r.daily_buffer_utilization) if u >= 0.999),
                None,
            )
            row.append(f"{day_full}" if day_full is not None else "never")
        print(f"{vname:<16}" + "".join(f"{v:>11}" for v in row))


def _print_weather_stats(results: dict[str, dict[str, FullStackResult]]) -> None:
    """Show what the real weather looks like per season for context."""
    print(f"\n=== Real-weather context (90-day windows, 44°N 71.5°W) ===")
    header = f"{'Season':<10}{'OutMin':>9}{'OutMax':>9}{'OutMean':>9}{'SolarMax':>10}{'SunHrs':>9}"
    print(header)
    print("-" * len(header))
    # Use any variant — weather is the same, only buffer differs.
    any_variant = next(iter(results))
    for s in REAL_SEASONS:
        csv_path = _WEATHER_DIR / f"new_england_{s}_90d.csv"
        data = from_open_meteo_csv(csv_path)
        outdoors = [v for _, v in data["outdoor_c"]]
        solars = [v for _, v in data["solar_w_m2"]]
        sun_hrs = sum(1 for v in solars if v > 0) / 90.0  # avg per day
        print(f"{s:<10}{min(outdoors):>9.1f}{max(outdoors):>9.1f}"
              f"{sum(outdoors) / len(outdoors):>9.1f}"
              f"{max(solars):>10.0f}{sun_hrs:>9.1f}")


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestBufferVariantsRealWeather:
    """Buffer variant sweep against real open-meteo weather."""

    def test_print_summary(self, real_variant_results):
        _print_weather_stats(real_variant_results)
        _print_buffer_fill(real_variant_results)
        _print_final_table(real_variant_results, "outdoor_delta", truth=-0.25)
        _print_final_table(real_variant_results, "Solar Proxy", truth=-2.0)
        _print_drift_table(real_variant_results, "Solar Proxy")

    def test_no_variant_diverges(self, real_variant_results):
        """Sanity: every (variant, season) finishes with bounded coefs."""
        for vname, by_season in real_variant_results.items():
            for sname, r in by_season.items():
                od = r.final_coefs.get("outdoor_delta", 0.0)
                solar = r.final_coefs.get("Solar Proxy", 0.0)
                assert -2.0 < od < 0.0, f"{vname}/{sname}: outdoor_delta={od:.4f}"
                assert -5.0 < solar < 1.0, f"{vname}/{sname}: solar={solar:.4f}"
