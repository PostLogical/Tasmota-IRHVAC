"""Shared bench constants — single source of truth.

Centralizes values that previously lived (and duplicated) across
``runner.py``, ``full_stack_runner.py``, ``adapters.py``, and
``open_loop_runner.py``.  Importing from here lets a single edit retune
the entire bench cadence or sim epoch.

``TICK_MINUTES_DEFAULT`` may be overridden at test invocation via
``pytest --tick-minutes=N`` (see ``conftest.py``); without that flag the
default below applies to any callsite that doesn't pass an explicit
``tick_interval_min`` / ``tick_minutes`` argument.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


# ── Cadence ──────────────────────────────────────────────────────────────
#
# Production runs sensor-driven at ~60s cooldown (per
# ``feedback_pi_tick_architecture.md``).  Bench default kept at 15 min
# during the wall-clock-refactor migration so a Phase-2 run at the
# refactored API can be diffed against the Phase-1 baseline without the
# cadence change confounding the comparison.  The final post-migration
# commit will flip this to 3.0 to bring bench fidelity closer to
# production.
#
# The ``BENCH_TICK_MINUTES`` environment variable provides a runtime
# override (set by conftest.pytest_configure when ``--tick-minutes`` is
# passed), so test invocations can sweep cadence without editing this
# file.

TICK_MINUTES_DEFAULT: float = float(os.environ.get("BENCH_TICK_MINUTES", "15.0"))


# ── Batch WLS cadence ───────────────────────────────────────────────────

BATCH_INTERVAL_HOURS_DEFAULT: float = 12.0


# ── Simulated wall-clock epoch ──────────────────────────────────────────

_SIM_EPOCH: datetime = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)


# ── Derived helpers ─────────────────────────────────────────────────────
#
# These are functions, not module-level constants, because
# ``TICK_MINUTES_DEFAULT`` can be overridden by the env var (set after
# import time by pytest_configure).  Materialising them as constants at
# import time would freeze the value before the override lands.


def ticks_per_hour(tick_minutes: float = TICK_MINUTES_DEFAULT) -> int:
    return int(round(60.0 / tick_minutes))


def ticks_per_day(tick_minutes: float = TICK_MINUTES_DEFAULT) -> int:
    return 24 * ticks_per_hour(tick_minutes)
