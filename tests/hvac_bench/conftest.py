"""Bench-suite pytest configuration: weather-source toggle.

Per ``feedback_synthetic_vs_real_bench.md`` and the seasonal-convergence
precedent (#45), learning-algorithm tests are validated against real
Open-Meteo weather. The bench-scenario layer flipped its default to real
in #45; this conftest extends the same default to the full-stack
regression layer (#49 Phase 1).

Tests in this tree that author both real and synth config builders consult
``BENCH_WEATHER`` (set here from ``--weather``) at config-build time.
Defaults to ``real``; pass ``--weather=synth`` for reproducible parameter
sweeps, seed scans, or controller-behavior debugging.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--weather",
        action="store",
        default=None,
        choices=("real", "synth"),
        help=(
            "Weather source for bench scenario tests: 'real' (Open-Meteo CSV, "
            "default) or 'synth' (AR(1) synthetic). Sets BENCH_WEATHER."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    val = config.getoption("--weather")
    if val:
        os.environ["BENCH_WEATHER"] = val
