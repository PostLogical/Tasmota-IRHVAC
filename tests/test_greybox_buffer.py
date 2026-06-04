"""Tests for GreyboxBuffer — append-only fill-and-wipe buffer.

2026-06-04 architecture: GreyboxBuffer no longer inherits from
DiversityAwareBuffer and no longer carries leverage-scoring machinery.
It is simply an observation list with:

  * Hard admission gate: reject ``outdoor_temp_c is None``.
  * FIFO drop on overflow (memory safety net only — normal weekly
    fill-and-wipe never approaches max_size).
  * ``clear()`` for the controller to call after each successful batch.
  * Round-trippable persistence via ``as_list`` / ``from_list``.

The prior leverage-policy test suite was removed when the inheritance
dropped; those tests characterized behavior that no longer exists.
"""

from __future__ import annotations

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    BufferAddResult,
    Observation,
)
from custom_components.tasmota_irhvac.pi.greybox_buffer import (
    DEFAULT_GREYBOX_BUFFER_SIZE,
    GreyboxBuffer,
)


def _make_obs(
    *,
    timestamp: float = 0.0,
    outdoor_temp_c: float | None = 10.0,
    hp_setpoint: float | None = 22.0,
    current_c: float = 20.0,
    clamped_reason: str = "",
) -> Observation:
    """Construct an Observation with reasonable defaults for buffer tests."""
    return Observation(
        timestamp=timestamp,
        wall_time=timestamp,
        hp_setpoint=hp_setpoint,
        current_c=current_c,
        desired_c=20.0,
        outdoor_temp_c=outdoor_temp_c,
        room_rate=0.0,
        raw_readings={},
        clamped=clamped_reason != "",
        clamped_reason=clamped_reason,
    )


# ── Admission ─────────────────────────────────────────────────────────


class TestAdmission:
    def test_admits_valid_observation(self):
        buf = GreyboxBuffer()
        result = buf.add(_make_obs())
        assert isinstance(result, BufferAddResult)
        assert result.admitted is True
        assert result.rejection_reason is None
        assert result.candidate_score is None  # no scoring under fill-and-wipe
        assert result.evicted_timestamp is None
        assert result.min_incumbent_score is None
        assert len(buf) == 1

    def test_rejects_observation_with_no_outdoor_temp(self):
        """The energy balance has T_out in every term — without it the
        observation cannot contribute and the buffer should reject."""
        buf = GreyboxBuffer()
        result = buf.add(_make_obs(outdoor_temp_c=None))
        assert result.admitted is False
        assert result.rejection_reason == "no_outdoor_temp"
        assert len(buf) == 0

    def test_admits_hp_off_observation(self):
        """HP-off observations carry information about ua_c (envelope
        heat loss with no HP forcing) and must be retained."""
        buf = GreyboxBuffer()
        result = buf.add(_make_obs(
            hp_setpoint=None,
            clamped_reason="no_output",
        ))
        assert result.admitted is True
        assert len(buf) == 1


# ── Fill-and-wipe lifecycle ───────────────────────────────────────────


class TestFillAndWipe:
    def test_get_all_returns_admitted_observations(self):
        buf = GreyboxBuffer()
        for i in range(5):
            buf.add(_make_obs(timestamp=float(i)))
        observations = buf.get_all()
        assert len(observations) == 5
        assert [o.timestamp for o in observations] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_get_all_returns_snapshot_not_shared_reference(self):
        """Iterating ``get_all()`` shouldn't be affected by concurrent
        ``add()`` / ``clear()`` activity on the buffer."""
        buf = GreyboxBuffer()
        for i in range(3):
            buf.add(_make_obs(timestamp=float(i)))
        snapshot = buf.get_all()
        buf.clear()
        # snapshot is independent of the now-empty buffer
        assert len(snapshot) == 3
        assert len(buf) == 0

    def test_clear_empties_buffer(self):
        buf = GreyboxBuffer()
        for i in range(10):
            buf.add(_make_obs(timestamp=float(i)))
        assert len(buf) == 10
        buf.clear()
        assert len(buf) == 0
        assert buf.get_all() == []

    def test_can_refill_after_clear(self):
        """The fill-and-wipe cycle: fill, clear, fill again — controller
        does this each weekly batch."""
        buf = GreyboxBuffer()
        buf.add(_make_obs(timestamp=0.0))
        buf.clear()
        buf.add(_make_obs(timestamp=100.0))
        observations = buf.get_all()
        assert len(observations) == 1
        assert observations[0].timestamp == 100.0


# ── Memory safety: FIFO on overflow ───────────────────────────────────


class TestMemorySafety:
    def test_fifo_drops_oldest_when_full(self):
        """If the buffer ever approaches max_size without a clear() (e.g.
        controller bug or extreme cadence), the oldest is FIFO-dropped to
        keep memory bounded."""
        buf = GreyboxBuffer(max_size=3)
        for i in range(5):
            result = buf.add(_make_obs(timestamp=float(i)))
            assert result.admitted is True
        # Buffer caps at 3, oldest dropped first
        assert len(buf) == 3
        observations = buf.get_all()
        assert [o.timestamp for o in observations] == [2.0, 3.0, 4.0]

    def test_evicted_timestamp_reported_on_overflow(self):
        """The eviction is reflected in the BufferAddResult so observability
        tooling can record the unusual event."""
        buf = GreyboxBuffer(max_size=2)
        buf.add(_make_obs(timestamp=0.0))
        buf.add(_make_obs(timestamp=1.0))
        result = buf.add(_make_obs(timestamp=2.0))
        assert result.admitted is True
        assert result.evicted_timestamp == 0.0


# ── Solar entity wiring ───────────────────────────────────────────────


class TestSolarEntity:
    def test_solar_entity_stored(self):
        buf = GreyboxBuffer(solar_entity="sensor.solar_proxy")
        assert buf.solar_entity == "sensor.solar_proxy"

    def test_solar_entity_settable(self):
        buf = GreyboxBuffer()
        assert buf.solar_entity is None
        buf.solar_entity = "sensor.solar_proxy"
        assert buf.solar_entity == "sensor.solar_proxy"


# ── Diagnostics ───────────────────────────────────────────────────────


class TestDiagnostics:
    def test_empty_buffer_diagnostics(self):
        buf = GreyboxBuffer(max_size=100)
        diag = buf.get_diagnostics()
        assert diag["total"] == 0
        assert diag["max_size"] == 100
        assert diag["hp_off_pct"] is None
        assert diag["min_leverage"] is None  # retained as None for backward-compat

    def test_hp_off_percentage(self):
        buf = GreyboxBuffer()
        # 3 HP-on, 1 HP-off
        for i in range(3):
            buf.add(_make_obs(timestamp=float(i)))
        buf.add(_make_obs(timestamp=4.0, hp_setpoint=None, clamped_reason="no_output"))
        diag = buf.get_diagnostics()
        assert diag["total"] == 4
        assert diag["hp_off_pct"] == 25.0


# ── Persistence ───────────────────────────────────────────────────────


class TestPersistence:
    def test_round_trip_preserves_observations(self):
        buf = GreyboxBuffer()
        for i in range(5):
            buf.add(_make_obs(timestamp=float(i), current_c=20.0 + i * 0.1))
        snapshot = buf.as_list()
        assert len(snapshot) == 5

        restored = GreyboxBuffer.from_list(snapshot)
        observations = restored.get_all()
        assert len(observations) == 5
        # Numeric fields round-trip cleanly
        assert [o.timestamp for o in observations] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert observations[0].current_c == pytest.approx(20.0)
        assert observations[4].current_c == pytest.approx(20.4)

    def test_round_trip_drops_invalid_observations(self):
        """Persisted entries with corrupt fields are silently skipped on
        restore so a single bad observation doesn't crash startup."""
        buf = GreyboxBuffer()
        buf.add(_make_obs(timestamp=0.0))
        snapshot = buf.as_list()
        # Inject a corrupt entry that's missing required fields
        snapshot.append({"timestamp": "not_a_number"})

        restored = GreyboxBuffer.from_list(snapshot)
        assert len(restored.get_all()) == 1

    def test_round_trip_drops_no_outdoor_temp_entries(self):
        """If persisted state somehow contains an observation with
        ``outdoor_temp_c=None``, it should not survive the restore
        (same gate as live ``add()``)."""
        buf = GreyboxBuffer()
        buf.add(_make_obs(timestamp=0.0, outdoor_temp_c=20.0))
        snapshot = buf.as_list()
        # Hand-craft a serialized entry with outdoor_temp_c=None (uses
        # the abbreviated schema keys produced by Observation.as_dict).
        snapshot.append({
            **snapshot[0],
            "t": 1.0,
            "ot": None,
        })

        restored = GreyboxBuffer.from_list(snapshot)
        observations = restored.get_all()
        assert len(observations) == 1
        assert observations[0].timestamp == 0.0

    def test_round_trip_respects_max_size(self):
        """When restoring more entries than ``max_size`` allows, the
        newest are retained (matches the FIFO-on-overflow rule of live
        ``add()``)."""
        buf = GreyboxBuffer()
        for i in range(5):
            buf.add(_make_obs(timestamp=float(i)))
        snapshot = buf.as_list()

        restored = GreyboxBuffer.from_list(snapshot, max_size=3)
        observations = restored.get_all()
        assert len(observations) == 3
        assert [o.timestamp for o in observations] == [2.0, 3.0, 4.0]


# ── Default size ──────────────────────────────────────────────────────


class TestDefaults:
    def test_default_max_size_is_memory_safety_cap(self):
        """The default is a memory-safety upper bound, sized for many weeks
        of high-cadence observations. Normal weekly fill-and-wipe never
        approaches it."""
        buf = GreyboxBuffer()
        assert buf._max_size == DEFAULT_GREYBOX_BUFFER_SIZE
        # Sanity: high enough to hold a week of 60s ticks (10 080) with
        # a lot of headroom.
        assert DEFAULT_GREYBOX_BUFFER_SIZE >= 20000
