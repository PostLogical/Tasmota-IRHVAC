"""Bench-suite pytest configuration.

Surfaces several cross-cutting bench knobs:

- ``--weather``: real-vs-synth weather source (per #45 / #49 Phase 1).
- ``--run-studies``: opt-in for ad-hoc ``@pytest.mark.study`` tests.
- ``--tick-minutes``: override ``constants.TICK_MINUTES_DEFAULT`` for this
  run (sets ``BENCH_TICK_MINUTES`` env var, read by ``constants.py`` at
  module import time during collection).
- ``--bench-phase``: tag the ``bench_metrics`` recorder output with a
  phase label (e.g. ``B0-baseline``, ``B0-refactored``, ``B0-3min``) so
  multiple runs land in distinct directories.

The ``bench_metrics`` fixture is opt-in (declare it as a parameter).
Tests that opt in record assertion-relevant values via
``bench_metrics["key"] = value``; the fixture writes one JSON per test
under ``local/bench_metrics/<phase>/<nodeid>.json`` containing the
recorded metrics plus pass/fail/duration captured from
``pytest_runtest_makereport``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


# ── CLI options ──────────────────────────────────────────────────────────


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
    parser.addoption(
        "--run-studies",
        action="store_true",
        default=False,
        help=(
            "Include @pytest.mark.study tests (ad-hoc verdict-recapture "
            "studies; deselected by default — too expensive even for design)."
        ),
    )
    parser.addoption(
        "--tick-minutes",
        action="store",
        default=None,
        help=(
            "Override TICK_MINUTES_DEFAULT for this pytest invocation. Sets "
            "BENCH_TICK_MINUTES env var which constants.py reads at import. "
            "Use during cadence sweeps; without this flag the constants.py "
            "default applies."
        ),
    )
    parser.addoption(
        "--bench-phase",
        action="store",
        default=None,
        help=(
            "Tag bench_metrics output directory with a phase label "
            "(e.g. 'B0-baseline', 'B0-3min'). Without this flag, output "
            "lands under 'adhoc/'."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    weather = config.getoption("--weather")
    if weather:
        os.environ["BENCH_WEATHER"] = weather

    tick_min = config.getoption("--tick-minutes")
    if tick_min is not None:
        os.environ["BENCH_TICK_MINUTES"] = str(tick_min)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip @study tests unless --run-studies passed."""
    if config.getoption("--run-studies"):
        return
    skip_study = pytest.mark.skip(
        reason="ad-hoc study; opt in with --run-studies"
    )
    for item in items:
        if "study" in item.keywords:
            item.add_marker(skip_study)


# ── bench_metrics fixture + makereport capture ──────────────────────────


# Stash key for cross-hook metrics state. Per pytest convention we use a
# StashKey so the report hook can find what the fixture stored without
# polluting the item's public namespace.
_METRICS_KEY = pytest.StashKey[dict[str, Any]]()


@pytest.fixture
def bench_metrics(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Per-test metric recorder.

    Tests opt in by declaring ``bench_metrics`` as a parameter and writing
    ``bench_metrics["final_error"] = final_error`` next to relevant
    assertions.  The fixture stashes the dict on the test item so
    ``pytest_runtest_makereport`` can attach pass/fail/duration on
    teardown and write the merged record to disk.

    Output: ``local/bench_metrics/<phase>/<nodeid>.json``.  Nodeid is
    flattened (path separators → ``__``, ``::`` → ``..``) for filesystem
    safety.
    """
    metrics: dict[str, Any] = {}
    request.node.stash[_METRICS_KEY] = metrics
    return metrics


def _bench_metrics_dir(config: pytest.Config) -> Path:
    phase = config.getoption("--bench-phase") or "adhoc"
    # Always write under repo-root/local/bench_metrics/<phase>/.
    repo_root = Path(config.rootpath)
    return repo_root / "local" / "bench_metrics" / phase


def _flatten_nodeid(nodeid: str) -> str:
    # Filesystem-safe filename. Replace path separators and `::` so that
    # downstream tooling can recover nodeid via reverse mapping.
    return nodeid.replace("/", "__").replace("::", "..")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()
    if report.when != "call":
        return

    # Only persist if the test opted in (stash entry exists).
    metrics = item.stash.get(_METRICS_KEY, None)
    if metrics is None:
        return

    record = {
        "nodeid": item.nodeid,
        "outcome": report.outcome,  # 'passed' / 'failed' / 'skipped'
        "duration_s": report.duration,
        "wasxfail": getattr(report, "wasxfail", None),
        "marks": sorted(m.name for m in item.iter_markers()),
        "metrics": dict(metrics),
    }

    out_dir = _bench_metrics_dir(item.config)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{_flatten_nodeid(item.nodeid)}.json"
    out_file.write_text(json.dumps(record, indent=2, default=str))
