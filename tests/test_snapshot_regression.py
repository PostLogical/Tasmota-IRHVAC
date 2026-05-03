"""Regression tests pinning the existing diagnostic wire format.

These tests capture the output of `get_full_diagnostics()` and the five
sensor-feeding getters in JSON fixtures, then assert byte-for-byte
equality on subsequent runs.

They land BEFORE the tick-first refactor begins so any drift in the wire
format introduced by the refactor is caught immediately.

Non-deterministic fields (timestamps, monotonic counters, RNG-seeded
values) are normalized to sentinel strings before comparison; see
`_normalize`.

To regenerate fixtures intentionally (e.g., after an additive change):
    REGEN_DIAG_FIXTURES=1 pytest tests/test_snapshot_regression.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REGEN = os.environ.get("REGEN_DIAG_FIXTURES") == "1"

# Field-name patterns whose values are non-deterministic and should be
# normalized to a sentinel before fixture comparison.
_NORMALIZE_KEYS = {
    "last_run_mono",  # monotonic timestamp
    "last_run_wallclock",  # ISO-8601
    "oldest_age_hours",  # monotonic-derived
    "last_fit",  # ISO-8601 from greybox
    "_ts_mono",
    "_ts_wall",
}

# Keys whose entire subtree is non-deterministic (e.g., time-of-day FF
# contributions computed from current wall clock).
_NORMALIZE_SUBTREE_KEYS = {
    "sin_hour",
    "cos_hour",
}

# Regex patterns for values that should be normalized regardless of key.
_NORMALIZE_VALUE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),  # ISO-8601 timestamps
]


def _normalize(obj: Any) -> Any:
    """Replace non-deterministic values with sentinel strings.

    Recursively walks dicts/lists. Returns a deep copy with normalized values.
    """
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _NORMALIZE_KEYS:
                result[k] = "<NORMALIZED>" if v is not None else None
            elif k in _NORMALIZE_SUBTREE_KEYS:
                result[k] = "<NORMALIZED_SUBTREE>"
            else:
                result[k] = _normalize(v)
        return result
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize(v) for v in obj)
    if isinstance(obj, str):
        for pattern in _NORMALIZE_VALUE_PATTERNS:
            if pattern.match(obj):
                return "<NORMALIZED>"
        return obj
    if isinstance(obj, float):
        # Round floats to 6 decimal places to absorb tiny FP noise that
        # comes from non-deterministic insertion order in dict/buffer state
        return round(obj, 6)
    return obj


def _compare_or_regen(captured: Any, fixture_path: Path, label: str) -> None:
    """Compare captured output to fixture, or regenerate if REGEN=1.

    Both `captured` and the fixture are normalized before comparison.
    Mismatch produces a diff-style assertion failure.
    """
    captured_norm = _normalize(captured)
    if REGEN or not fixture_path.exists():
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(
            json.dumps(captured_norm, indent=2, sort_keys=True, default=str)
            + "\n"
        )
        if not REGEN:
            pytest.skip(
                f"Generated fixture {fixture_path.name} — re-run to verify"
            )
        return

    expected = json.loads(fixture_path.read_text())
    captured_json = json.loads(
        json.dumps(captured_norm, sort_keys=True, default=str)
    )

    if captured_json != expected:
        # Produce a more useful diff
        import difflib

        expected_str = json.dumps(expected, indent=2, sort_keys=True)
        captured_str = json.dumps(captured_json, indent=2, sort_keys=True)
        diff = "\n".join(
            difflib.unified_diff(
                expected_str.splitlines(),
                captured_str.splitlines(),
                fromfile=f"fixture/{fixture_path.name}",
                tofile=f"captured/{label}",
                n=3,
            )
        )
        pytest.fail(
            f"\n{label} drift detected vs {fixture_path.name}:\n{diff}\n"
            f"To accept these changes, run:\n"
            f"    REGEN_DIAG_FIXTURES=1 pytest tests/test_snapshot_regression.py"
        )


# ── Test 1: get_full_diagnostics fixture ──────────────────────────────


@pytest.mark.asyncio
async def test_get_full_diagnostics_basic_state(hass, setup_pi_integration):
    """Pin get_full_diagnostics output for the basic post-setup state.

    This captures the COMMON path (no batch run, no greybox fit, no
    residual patterns yet, IDLE regime probe). Branches with more complex
    state are covered by separate fixtures.
    """
    from custom_components.tasmota_irhvac.diagnostics import (
        async_get_config_entry_diagnostics,
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    diag = await async_get_config_entry_diagnostics(hass, entry)

    # Strip config_entry section — entry_id is non-deterministic and
    # already covered by other tests.
    diag.pop("config_entry", None)

    _compare_or_regen(
        diag,
        FIXTURES_DIR / "diagnostics_full_basic.json",
        "diagnostics_full_basic",
    )


# ── Test 2: sensor-feeding getter fixtures ────────────────────────────


@pytest.mark.asyncio
async def test_get_health_status_fixture(hass, setup_pi_integration):
    """Pin get_health_status() output for basic state."""
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller
    status = pi.get_health_status()
    _compare_or_regen(
        status,
        FIXTURES_DIR / "sensor_health_basic.json",
        "get_health_status",
    )


@pytest.mark.asyncio
async def test_get_learning_state_fixture(hass, setup_pi_integration):
    """Pin get_learning_state() output for basic state."""
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller
    state = pi.get_learning_state()
    _compare_or_regen(
        state,
        FIXTURES_DIR / "sensor_learning_state_basic.json",
        "get_learning_state",
    )


@pytest.mark.asyncio
async def test_get_greybox_state_fixture(hass, setup_pi_integration):
    """Pin get_greybox_state() output for basic state."""
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller
    state = pi.get_greybox_state()
    _compare_or_regen(
        state,
        FIXTURES_DIR / "sensor_greybox_state_basic.json",
        "get_greybox_state",
    )


@pytest.mark.asyncio
async def test_get_learning_status_fixture(hass, setup_pi_integration):
    """Pin get_learning_status() output (binary_sensor) for basic state."""
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller
    status = pi.get_learning_status()
    _compare_or_regen(
        status,
        FIXTURES_DIR / "sensor_learning_status_basic.json",
        "get_learning_status",
    )


@pytest.mark.asyncio
async def test_get_drifting_coefficients_fixture(hass, setup_pi_integration):
    """Pin get_drifting_coefficients() output for basic state."""
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller
    drift = pi.get_drifting_coefficients()
    # Convert tuples to lists for JSON
    drift_serializable = [list(item) for item in drift]
    _compare_or_regen(
        drift_serializable,
        FIXTURES_DIR / "sensor_drifting_coefficients_basic.json",
        "get_drifting_coefficients",
    )
