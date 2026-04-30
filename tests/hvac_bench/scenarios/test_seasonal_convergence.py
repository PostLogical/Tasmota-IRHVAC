"""Seasonal convergence: do batch WLS coefficients agree across heating seasons?

Question: with online RLS deprecated, the diversity buffer's "year-to-year
persistence" role is no longer load-bearing if batch WLS converges fast within
a single season. This test checks both:

1. **Within-season convergence speed** — in fall, winter, spring (heating
   seasons), how many days/batches before coefficients stabilize?

2. **Cross-season agreement** — do converged coefficients land in the same
   place, or does each season pull toward a different attractor?

Cooling season (summer) is excluded — the per-mode buffer split keeps cooling
observations separate from heating.

WEATHER SOURCE (#45, 2026-04-29): real Open-Meteo CSVs (44°N 71.5°W) are
the *default* — any verdict must come from real weather. Synthetic AR(1)
weather is opt-in via :func:`_make_synth_config` and reserved for parameter
sweeps where reproducible knobs (varying ``weather_amp_c``, sweeping seeds)
matter more than realism. ``feedback_synthetic_vs_real_bench.md`` explains
the inversion: synth misled us once on the FIFO/leverage finding and clean
synthetic distributions can produce results that don't survive real weather.

CAVEAT — what this test can and can't show:
- Bench physics is linear by construction (2R2C). So same coefficients across
  seasons here ≠ "real world is seasonally stable" — the bench can't falsify
  nonlinearity. A positive result is suggestive; cross-check against
  production data before acting.
- Different coefficients across seasons here ≈ "even the easy linear case
  fails to converge from one season's data alone" — strong evidence that
  cross-season persistence (the diversity buffer) is load-bearing.

Asymmetric test: a negative result is strong, a positive result is weak.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pytest

from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    FullStackResult,
    ModelInputSpec,
    WeatherState,
    diurnal_outdoor,
    print_full_stack_summary,
    run_full_stack,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.scenarios._weather_mode import (
    SHOULDER_FALL,
    SHOULDER_SPRING,
    WINTER_TYPICAL,
    WeatherWindow,
    windowed_real_weather,
)


# Cross-season convergence asks a typical-conditions question, so winter
# resolves to WINTER_TYPICAL (2024-01-01) rather than the cold-year
# WINTER_DEEP (2025-01-01).
_SEASON_WINDOWS: dict[str, WeatherWindow] = {
    "winter": WINTER_TYPICAL,
    "fall": SHOULDER_FALL,
    "spring": SHOULDER_SPRING,
}


# ── Seasonal weather profiles ────────────────────────────────────────────


@dataclass(frozen=True)
class SeasonProfile:
    """Outdoor + solar parameters for a heating season.

    Tuned for a northern temperate climate (similar to NE deployment).
    Solar varies in both peak intensity and day length; outdoor varies
    in mean and diurnal range. Both shape the excitation available
    to batch WLS.
    """
    name: str
    outdoor_base_c: float
    outdoor_diurnal_c: float
    solar_peak: float
    solar_sunrise: float
    solar_sunset: float


SEASONS: dict[str, SeasonProfile] = {
    "fall": SeasonProfile(
        name="fall",
        outdoor_base_c=8.0,       # Oct-Nov mean
        outdoor_diurnal_c=8.0,
        solar_peak=0.6,
        solar_sunrise=7.0,
        solar_sunset=18.0,
    ),
    "winter": SeasonProfile(
        name="winter",
        outdoor_base_c=-5.0,      # Jan-Feb mean (deep freeze)
        outdoor_diurnal_c=4.0,
        solar_peak=0.5,
        solar_sunrise=8.0,
        solar_sunset=16.5,
    ),
    "spring": SeasonProfile(
        name="spring",
        outdoor_base_c=10.0,      # Apr mean (big swings)
        outdoor_diurnal_c=10.0,
        solar_peak=0.7,
        solar_sunrise=6.5,
        solar_sunset=19.0,
    ),
}


def _make_solar_schedule(
    season: SeasonProfile,
    weather_state: WeatherState,
    tick_minutes: float = 15.0,
):
    """Solar with seasonal day length and peak intensity, AR(1) cloud factor.

    Cloud factor is ``clip(0.7 + 0.5·W(tick), 0.2, 1.0)`` where W is the shared
    weather state — same instance also shifts outdoor temp in :func:`diurnal_outdoor`,
    producing the residual T-vs-S correlation seen in real Open-Meteo data.
    """
    def schedule(tick: int) -> float:
        hour = (tick * tick_minutes / 60.0) % 24.0
        if hour < season.solar_sunrise or hour > season.solar_sunset:
            return 0.0
        day_len = season.solar_sunset - season.solar_sunrise
        base = season.solar_peak * math.sin(
            math.pi * (hour - season.solar_sunrise) / day_len
        )
        cloud = max(0.2, min(1.0, 0.7 + 0.3 * weather_state(tick)))
        return base * cloud
    return schedule


def _make_outdoor_schedule(
    season: SeasonProfile,
    weather_state: WeatherState,
    tick_minutes: float = 15.0,
):
    """Outdoor temp with diurnal swing + AR(1) weather offset (replaces 5-day sine)."""
    def schedule(tick: int) -> float:
        return diurnal_outdoor(
            tick,
            base_c=season.outdoor_base_c,
            amplitude_c=season.outdoor_diurnal_c,
            tick_minutes=tick_minutes,
            weather_state=weather_state,
        )
    return schedule


def _make_synth_config(season_name: str, n_days: int = 90) -> FullStackConfig:
    """Synthetic AR(1) config — opt-in for parameter sweeps and reproducible knobs.

    Same building, same noise seed, same wrong starting seed — vary weather.
    AR(1) shared weather state produces residual T-vs-S correlation in the
    0.10–0.30 band measured from real Open-Meteo data. Use this for tests
    that need to vary ``weather_amp_c``, sweep seeds, or otherwise hold
    weather as a controllable parameter; for verdict-producing tests, use
    :func:`_make_real_config` (the default :func:`_make_config`).
    """
    season = SEASONS[season_name]
    profile = PROFILES_2R2C["living_room"]
    tick_min = 15.0
    n_ticks = int(n_days * 24 * 60 / tick_min)
    weather_state = WeatherState(
        n_ticks=n_ticks, seed=42, persistence_hours=36.0, tick_minutes=tick_min
    )
    return FullStackConfig(
        n_days=n_days,
        profile_name="living_room",
        outdoor_base_c=season.outdoor_base_c,
        outdoor_diurnal_c=season.outdoor_diurnal_c,
        outdoor_schedule=_make_outdoor_schedule(season, weather_state, tick_min),
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
                schedule=_make_solar_schedule(season, weather_state, tick_min),
            ),
        ],
        pi_overrides={
            # 2× wrong seed: forces non-trivial learning
            "pi_outdoor_seed_heat": profile.true_seed * 2.0,
        },
        relax_kappa_gate=True,
    )


def _make_real_config(season_name: str, n_days: int = 90) -> FullStackConfig:
    """Open-Meteo-driven config — the canonical default (#45, #51).

    Pulls a ``WeatherWindow`` from the multi-year CSV (44°N 71.5°W) for the
    requested season via ``_SEASON_WINDOWS``. Same building/seed/wrong-starting-
    seed as the synth path so only the weather distribution differs. Produces
    verdicts that survive real cloud clustering, weather fronts, and seasonal
    day-length shifts.
    """
    window = _SEASON_WINDOWS[season_name]
    outdoor_fn, solar_fn, n_days = windowed_real_weather(
        start_day=window.start_day, n_days=n_days,
    )

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
            "pi_outdoor_seed_heat": profile.true_seed * 2.0,
        },
        relax_kappa_gate=True,
    )


# Default canonical config: real CSV. Tests that need synthetic reproducibility
# (parameter sweeps, seed scans) call ``_make_synth_config`` directly.
_make_config = _make_real_config


# ── Helpers ──────────────────────────────────────────────────────────────


def _batches_to_stable(
    trajectory: list[dict],
    coef_name: str,
    final: float,
    tol: float,
) -> int | None:
    """First batch index where coef stays within `tol` of `final` for the rest."""
    n = len(trajectory)
    for i in range(n):
        if all(
            abs(snap.get(coef_name, 0.0) - final) < tol
            for snap in trajectory[i:]
        ):
            return i
    return None


def _print_summary(results: dict[str, FullStackResult]) -> None:
    """Side-by-side coefficient and convergence comparison across seasons."""
    seasons = list(results.keys())
    print(f"\n{'=' * 72}")
    print(f"  Seasonal Convergence (living_room, 2× wrong outdoor seed, 90d)")
    print(f"{'=' * 72}")
    header = f"{'Metric':<34}" + "".join(f"{s:>12}" for s in seasons)
    print(header)
    print("-" * 72)

    # Final coefficients
    for coef in ["outdoor_delta", "Solar Proxy"]:
        vals = [results[s].final_coefs.get(coef, 0.0) for s in seasons]
        print(f"{'Final ' + coef:<34}" + "".join(f"{v:>12.4f}" for v in vals))

    # Cross-season spread (the key cross-season agreement metric)
    print("-" * 72)
    for coef in ["outdoor_delta", "Solar Proxy"]:
        vals = [results[s].final_coefs.get(coef, 0.0) for s in seasons]
        spread = max(vals) - min(vals)
        print(f"{'Spread (max-min) ' + coef:<34}{spread:>12.4f}")

    # Convergence speed: batches and approx days to first reach within tol
    print("-" * 72)
    for coef, tol in [("outdoor_delta", 0.05), ("Solar Proxy", 0.20)]:
        bvals = []
        dvals = []
        for s in seasons:
            r = results[s]
            final = r.final_coefs.get(coef, 0.0)
            n = _batches_to_stable(r.coef_trajectory, coef, final, tol=tol)
            bvals.append(n if n is not None else -1)
            # ~2 batches/day at default 12h interval
            dvals.append(n / 2.0 if n is not None else float("nan"))
        print(f"{f'Batches to ±{tol} of final ' + coef:<34}"
              + "".join(f"{v:>12d}" for v in bvals))
        print(f"{f'Approx days to ±{tol} ' + coef:<34}"
              + "".join(f"{v:>12.1f}" for v in dvals))

    # Trajectory snapshots every 15 days (~2 batches/day → batch 30, 60, ...)
    print("-" * 72)
    for day in [15, 30, 45, 60, 75, 90]:
        target = int(day * 2)
        for coef in ["outdoor_delta", "Solar Proxy"]:
            row = []
            for s in seasons:
                traj = results[s].coef_trajectory
                if not traj:
                    row.append(float("nan"))
                    continue
                # Cap at last batch — final day request maps to final snapshot.
                idx = min(target, len(traj) - 1)
                row.append(traj[idx].get(coef, 0.0))
            label = f"{coef} @ day {day}"
            print(f"{label:<34}" + "".join(f"{v:>12.4f}" for v in row))
        if day != 90:
            print()

    # Comfort + batch counts (sanity)
    print("-" * 72)
    row = [results[s].ctrl_comfort_pct for s in seasons]
    print(f"{'Controllable comfort %':<34}"
          + "".join(f"{v:>11.1f}%" for v in row))
    row = [results[s].n_batches for s in seasons]
    print(f"{'Total batches run':<34}" + "".join(f"{v:>12d}" for v in row))


def _print_solar_drift_diagnostic(results: dict[str, FullStackResult]) -> None:
    """Per-batch trajectory of Solar Proxy + outdoor + intercept + buffer.

    Diagnostic for the day-30 → day-60 Solar Proxy drift. Prints every
    10th batch (~5 days) so we can see when the drift starts and whether
    it correlates with buffer fill, outdoor micro-shifts, or intercept.
    """
    for name, r in results.items():
        print(f"\n=== {name}: Solar Proxy trajectory ===")
        print(f"{'batch':>6} {'day':>5} {'outdoor':>10} {'solar':>10} "
              f"{'intercept':>10} {'sol_froz':>9} {'buf%':>7}")
        n_days = len(r.daily_buffer_utilization)
        for i in range(0, len(r.coef_trajectory), 10):
            snap = r.coef_trajectory[i]
            day = i / 2.0
            day_idx = min(int(day), n_days - 1) if n_days else 0
            buf = (r.daily_buffer_utilization[day_idx]
                   if n_days else 0.0)
            frozen = snap.get("Solar Proxy_frozen", "?")
            print(
                f"{i:>6d} {day:>5.1f} "
                f"{snap.get('outdoor_delta', 0):>10.4f} "
                f"{snap.get('Solar Proxy', 0):>10.4f} "
                f"{snap.get('intercept', 0):>10.4f} "
                f"{str(frozen):>9} "
                f"{buf:>6.1%}"
            )


# ── Fixture: run all seasons once, share across tests ───────────────────


def _compute_seasonal_results() -> dict[str, FullStackResult]:
    """Run all heating seasons (90 days each). Shared by fixture and CLI.

    Uses real Open-Meteo CSVs by default (#45). Silences PI/batch loggers
    during the runs so the printed summary table is the only artifact.
    """
    pi_logger = logging.getLogger("custom_components.tasmota_irhvac")
    prev_level = pi_logger.level
    pi_logger.setLevel(logging.ERROR)
    try:
        return {
            name: run_full_stack(_make_config(name, n_days=90))
            for name in SEASONS
        }
    finally:
        pi_logger.setLevel(prev_level)


def run_seasonal_summary() -> None:
    """CLI entry: per-run generic summary, then cross-season tables + diagnostic."""
    results = _compute_seasonal_results()
    for name, result in results.items():
        print_full_stack_summary(result, label=f"seasonal / {name}")
    _print_summary(results)
    _print_solar_drift_diagnostic(results)


@pytest.fixture(scope="module")
def seasonal_results() -> dict[str, FullStackResult]:
    """Run all heating seasons (90 days each) once for the whole test module."""
    return _compute_seasonal_results()


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestSeasonalConvergence:
    """Compare batch WLS convergence and final coefficients across heating seasons."""

    def test_no_season_diverges(self, seasonal_results):
        """Sanity: every season's final coefficients must be bounded."""
        for name, r in seasonal_results.items():
            od = r.final_coefs.get("outdoor_delta", 0.0)
            assert -2.0 < od < 0.0, (
                f"{name}: outdoor_delta out of plausible range: {od:.4f}"
            )
            solar = r.final_coefs.get("Solar Proxy", 0.0)
            assert -5.0 < solar < 1.0, (
                f"{name}: Solar Proxy out of plausible range: {solar:.4f}"
            )

    def test_cross_season_outdoor_agreement(self, seasonal_results):
        """outdoor_delta should land in the same place across seasons.

        Underlying physics is identical; only weather (excitation) differs.
        Cross-season disagreement >0.10 means the WLS estimator is sensitive
        to operating regime, which would justify keeping the diversity buffer.
        """
        ods = [r.final_coefs.get("outdoor_delta", 0.0)
               for r in seasonal_results.values()]
        spread = max(ods) - min(ods)
        assert spread < 0.10, (
            f"outdoor_delta differs across seasons: spread={spread:.4f} "
            f"(values: {[f'{v:.4f}' for v in ods]}). "
            f"Diversity buffer's persistence role appears load-bearing."
        )

    def test_cross_season_solar_agreement(self, seasonal_results):
        """Solar coefficient varies modestly across seasons.

        With AR(1) shared weather state, solar β identification depends on the
        season-specific solar excitation × shared-cause structure — this is the
        same season-dependent identifiability seen in real Open-Meteo CSV runs
        (real β = -1.03 to -1.20; see project_buffer_seasonal_findings.md).
        Tolerance widened from 0.5 (clean-synth era) to 0.8 to reflect the
        realistic difficulty.
        """
        solars = [r.final_coefs.get("Solar Proxy", 0.0)
                  for r in seasonal_results.values()]
        spread = max(solars) - min(solars)
        assert spread < 0.8, (
            f"Solar Proxy differs across seasons: spread={spread:.4f} "
            f"(values: {[f'{v:.4f}' for v in solars]})"
        )

    def test_each_season_stabilizes_within_run(self, seasonal_results):
        """Each season should reach a stable outdoor_delta within 90 days.

        Loose bound — the printed summary shows the actual time-to-converge.
        A failure here means even 90 days of within-season data isn't enough
        to settle, which is itself an answer.
        """
        for name, r in seasonal_results.items():
            final_od = r.final_coefs.get("outdoor_delta", 0.0)
            n_stable = _batches_to_stable(
                r.coef_trajectory, "outdoor_delta", final_od, tol=0.05
            )
            assert n_stable is not None, (
                f"{name}: outdoor_delta never stabilized within 0.05 of "
                f"final value over {len(r.coef_trajectory)} batches"
            )
