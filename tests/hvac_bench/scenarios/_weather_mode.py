"""Weather-mode resolver + real-CSV schedule builder for full-stack scenarios.

Tests in the full-stack regression layer (test_full_stack_learning,
test_boundary_learning) accept a ``weather="real"|"synth"`` parameter on
their config builders. When ``weather`` is omitted, callers consult
:func:`get_default_weather_mode`, which reads ``BENCH_WEATHER`` set by
the ``--weather`` pytest CLI option in ``tests/hvac_bench/conftest.py``.

The default is ``"real"``: per ``feedback_synthetic_vs_real_bench.md``,
learning-algorithm tests must be validated on real weather. ``"synth"``
remains available for reproducible parameter sweeps and seed scans.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal

from tests.hvac_bench.csv_adapters import csv_to_schedules, from_open_meteo_csv


WeatherMode = Literal["real", "synth"]
Season = Literal["winter", "fall", "spring", "summer"]

_WEATHER_DIR = Path(__file__).parent.parent / "weather_data"


def get_default_weather_mode() -> WeatherMode:
    """Return the env-driven default ('real' unless BENCH_WEATHER=synth)."""
    val = os.environ.get("BENCH_WEATHER", "real").lower()
    return "synth" if val == "synth" else "real"


def season_for_outdoor_base(outdoor_base_c: float) -> Season:
    """Map a synthetic ``outdoor_base_c`` to the closest NE season CSV.

    ``feedback_synthetic_vs_real_bench.md`` says learning verdicts must
    survive real weather; this mapping picks the season whose CSV mean is
    closest to the synth base, so a real-mode swap doesn't accidentally
    turn a winter test into a spring one.
    """
    if outdoor_base_c <= 0.0:
        return "winter"
    if outdoor_base_c <= 8.0:
        return "fall"
    return "spring"


def real_weather_schedules(
    season: Season,
    *,
    min_days: int = 0,
    weather_dir: Path | None = None,
) -> tuple[Callable[[int], float], Callable[[int], float] | None, int]:
    """Load the right Open-Meteo CSV for a season → (outdoor_fn, solar_fn, max_days).

    Prefers the 90d CSV when ``min_days > 14`` (winter/fall/spring); falls
    back to the 2w CSV otherwise. ``solar_fn`` normalizes W/m² to the 0-1
    proxy the thermal model expects. ``max_days`` lets callers clamp the
    requested ``n_days`` to what the CSV actually contains.
    """
    weather_dir = weather_dir or _WEATHER_DIR

    candidates: list[Path] = []
    if min_days > 14:
        candidates.append(weather_dir / f"new_england_{season}_90d.csv")
    candidates.append(weather_dir / f"new_england_{season}_2w.csv")
    candidates.append(weather_dir / f"new_england_{season}_90d.csv")

    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"No CSV found for season={season} (looked in {weather_dir})"
        )

    csv_data = from_open_meteo_csv(csv_path)
    schedules = csv_to_schedules(csv_data)

    outdoor_fn = schedules["outdoor_c"]
    raw_solar_fn = schedules.get("solar_w_m2")
    solar_fn: Callable[[int], float] | None
    if raw_solar_fn is not None:
        solar_fn = lambda t, _f=raw_solar_fn: _f(t) / 1000.0
    else:
        solar_fn = None

    n_hours = len(csv_data.get("outdoor_c", [])) - 1
    max_days = max(1, n_hours // 24)
    return outdoor_fn, solar_fn, max_days
