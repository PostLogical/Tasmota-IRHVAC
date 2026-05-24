"""Unit tests for the production-bundle assertion helpers.

Focus: `extract_room_temp_c`, which reads the zone's OWN controlled room
temperature from the `_current_room_temp_c` tick field (future_work #116).
It deliberately has NO fallback to `_observation.raw_readings`: model
inputs are exogenous (a zone's only `sensor.*_temperature` there is often
an *adjacent* zone, e.g. dining's `sensor.living_room_air_sensor_
temperature`), so guessing from them returned a neighbour's temperature —
the bug #116 fixes. Bundles predating #116 return None and skip
temperature assertions rather than score the wrong zone.
"""

from __future__ import annotations

from tests.production_replay.assertions import extract_room_temp_c


# ── Own-temp field is the only source ────────────────────────────────────


def test_reads_own_temp_field_ignoring_adjacent_sensor():
    """`_current_room_temp_c` is used, even though the only
    `sensor.*_temperature` in raw_readings is an adjacent zone."""
    tick = {
        "_current_room_temp_c": 18.30,
        "_observation": {
            "raw_readings": {
                "sensor.solar_gain_proxy": 0.4,
                "binary_sensor.dining_room_boiler_calling": 0.0,
                # Adjacent zone — a raw_readings heuristic would return this.
                "sensor.living_room_air_sensor_temperature": 19.92,
            },
        },
    }
    assert extract_room_temp_c(tick) == 18.30


def test_own_temp_field_is_celsius_not_fahrenheit_converted():
    """The own-temp field is the controller's internal °C value — returned
    verbatim, never run through F→C range detection."""
    tick = {"_current_room_temp_c": 20.61, "_observation": {"raw_readings": {}}}
    assert extract_room_temp_c(tick) == 20.61


# ── No field → None, regardless of what's in raw_readings (no fallback) ──


def test_non_finite_field_returns_none():
    tick = {
        "_current_room_temp_c": float("nan"),
        "_observation": {"raw_readings": {"sensor.kitchen_air_temperature": 21.0}},
    }
    assert extract_room_temp_c(tick) is None


def test_no_field_returns_none_even_with_temperature_sensor():
    """A pre-#116 bundle whose only temp sensor is a model input must NOT be
    scored — that sensor is exogenous (an adjacent zone), not the own room."""
    tick = {"_observation": {"raw_readings": {"sensor.living_room_air_sensor_temperature": 19.9}}}
    assert extract_room_temp_c(tick) is None


def test_no_field_no_observation_returns_none():
    assert extract_room_temp_c({}) is None
