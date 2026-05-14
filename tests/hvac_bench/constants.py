"""Shared bench constants — single source of truth.

Centralizes values that previously lived (and duplicated) across
``runner.py``, ``full_stack_runner.py``, ``adapters.py``, and
``open_loop_runner.py``.  Importing from here lets a single edit retune
the entire bench cadence or sim epoch.

``TICK_MINUTES_DEFAULT`` may be overridden at test invocation via
``pytest --tick-minutes=N`` (see ``conftest.py``); without that flag the
default below applies to any callsite that doesn't pass an explicit
``tick_interval_min`` / ``tick_minutes`` argument.

Standalone scripts that need a non-default cadence call
``set_tick_minutes_default(value)`` *before* importing any module that
captures ``TICK_MINUTES_DEFAULT`` as a function default argument.
"""

from __future__ import annotations

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

TICK_MINUTES_DEFAULT: float = 15.0


def set_tick_minutes_default(value: float) -> None:
    """Set ``TICK_MINUTES_DEFAULT`` for the current process.

    Single canonical entry point — ``conftest.pytest_configure`` calls
    this when ``--tick-minutes=N`` is passed, and standalone scripts
    invoke it before importing bench runner modules.  Eliminates the
    pre-#104 trap where setting ``BENCH_TICK_MINUTES`` env var alone
    changed runtime cadence but not pytest-regressions baseline routing.
    """
    global TICK_MINUTES_DEFAULT
    TICK_MINUTES_DEFAULT = float(value)


# ── Simulated wall-clock epoch ──────────────────────────────────────────

_SIM_EPOCH: datetime = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)


# ── Derived helpers ─────────────────────────────────────────────────────
#
# Sentinel default (None → look up at call time) so a mutation of
# ``TICK_MINUTES_DEFAULT`` after this module's import is observed by
# subsequent calls.  Function-default arguments would otherwise freeze
# the import-time value.


def ticks_per_hour(tick_minutes: float | None = None) -> int:
    if tick_minutes is None:
        tick_minutes = TICK_MINUTES_DEFAULT
    return int(round(60.0 / tick_minutes))


def ticks_per_day(tick_minutes: float | None = None) -> int:
    return 24 * ticks_per_hour(tick_minutes)
