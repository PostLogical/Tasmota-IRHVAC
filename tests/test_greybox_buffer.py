"""Tests for GreyboxBuffer — leverage-scored grey-box observation buffer."""

import math
import time

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    BufferAddResult,
    Observation,
)
from custom_components.tasmota_irhvac.pi.greybox_buffer import (
    DEFAULT_GREYBOX_BUFFER_SIZE,
    GreyboxBuffer,
    _GREYBOX_N_FEATURES,
)


def _make_obs(
    outdoor_temp_c: float | None = 10.0,
    hp_setpoint: float | None = 22.0,
    current_c: float = 21.0,
    room_rate: float = 0.001,
    wall_time: float | None = None,
    clamped_reason: str = "",
    raw_readings: dict | None = None,
) -> Observation:
    """Create a test observation with sensible defaults."""
    return Observation(
        timestamp=time.monotonic(),
        wall_time=wall_time or time.time(),
        hp_setpoint=hp_setpoint,
        current_c=current_c,
        desired_c=22.0,
        outdoor_temp_c=outdoor_temp_c,
        room_rate=room_rate,
        raw_readings=raw_readings or {},
        clamped=bool(clamped_reason),
        clamped_reason=clamped_reason,
        supplemental_active=False,
    )


# ── Admission ────────────────────────────────────────────────────────


class TestAdmission:
    """Admission policy: everything with outdoor_temp_c, including HP-off."""

    def test_admits_hp_on(self):
        buf = GreyboxBuffer(max_size=100)
        buf.add(_make_obs(outdoor_temp_c=5.0, hp_setpoint=22.0))
        assert len(buf) == 1

    def test_admits_hp_off_no_output(self):
        buf = GreyboxBuffer(max_size=100)
        buf.add(_make_obs(
            outdoor_temp_c=5.0, hp_setpoint=None, clamped_reason="no_output",
        ))
        assert len(buf) == 1

    def test_admits_passive_tick(self):
        buf = GreyboxBuffer(max_size=100)
        buf.add(_make_obs(
            outdoor_temp_c=15.0, hp_setpoint=None, clamped_reason="no_output",
        ))
        assert len(buf) == 1

    def test_admits_saturated(self):
        buf = GreyboxBuffer(max_size=100)
        buf.add(_make_obs(outdoor_temp_c=5.0, clamped_reason="saturated_high"))
        assert len(buf) == 1

    def test_rejects_no_outdoor_temp(self):
        buf = GreyboxBuffer(max_size=100)
        buf.add(_make_obs(outdoor_temp_c=None))
        assert len(buf) == 0

    def test_fills_to_max_size(self):
        buf = GreyboxBuffer(max_size=200)
        for i in range(300):
            buf.add(_make_obs(outdoor_temp_c=float(i % 40 - 10)))
        assert len(buf) == 200


# ── Feature vector ───────────────────────────────────────────────────


class TestFeatureVector:
    """Test the 4-feature grey-box leverage vector."""

    def test_hp_on_features(self):
        buf = GreyboxBuffer(max_size=100)
        obs = _make_obs(outdoor_temp_c=5.0, hp_setpoint=22.0, current_c=20.0, room_rate=0.01)
        vec = buf._get_feature_vector(obs)
        assert len(vec) == _GREYBOX_N_FEATURES
        assert vec[0] == pytest.approx(5.0 - 20.0)  # outdoor_delta
        assert vec[1] == pytest.approx(22.0 - 20.0)  # hp_offset
        assert vec[2] == pytest.approx(0.0)  # solar (no entity)
        assert vec[3] == pytest.approx(0.01)  # room_rate

    def test_hp_off_features(self):
        buf = GreyboxBuffer(max_size=100)
        obs = _make_obs(outdoor_temp_c=5.0, hp_setpoint=None, current_c=20.0)
        vec = buf._get_feature_vector(obs)
        assert vec[1] == pytest.approx(0.0)  # hp_offset = 0 when off

    def test_solar_entity_features(self):
        buf = GreyboxBuffer(max_size=100, solar_entity="sensor.solar")
        obs = _make_obs(outdoor_temp_c=5.0, raw_readings={"sensor.solar": 250.0})
        vec = buf._get_feature_vector(obs)
        assert vec[2] == pytest.approx(250.0)

    def test_solar_entity_missing_reading(self):
        buf = GreyboxBuffer(max_size=100, solar_entity="sensor.solar")
        obs = _make_obs(outdoor_temp_c=5.0, raw_readings={})
        vec = buf._get_feature_vector(obs)
        assert vec[2] == pytest.approx(0.0)


# ── Leverage-scored eviction ─────────────────────────────────────────


class TestLeverageEviction:
    """Leverage scoring retains diverse observations."""

    def test_retains_diverse_temps(self):
        """Buffer should retain observations across the full temp range."""
        buf = GreyboxBuffer(max_size=200)
        for i in range(500):
            t = -20 + 50 * (i % 50) / 49
            buf.add(_make_obs(outdoor_temp_c=t, wall_time=1_000_000.0 + i))
        assert len(buf) == 200
        temps = [o.outdoor_temp_c for o in buf.get_all()]
        assert min(temps) < -15, f"Lost cold extreme: min={min(temps)}"
        assert max(temps) > 25, f"Lost warm extreme: max={max(temps)}"

    def test_retains_hp_off_via_leverage(self):
        """HP-off observations have different feature vectors (hp_offset=0)
        and should be retained naturally by leverage scoring."""
        buf = GreyboxBuffer(max_size=200)
        wt = 1_000_000.0
        # 180 HP-on observations at various temps
        for i in range(180):
            buf.add(_make_obs(
                outdoor_temp_c=float(i % 30 - 10),
                hp_setpoint=22.0,
                wall_time=wt + i,
            ))
        # 20 HP-off observations at same temps
        for i in range(20):
            buf.add(_make_obs(
                outdoor_temp_c=float(i % 30 - 10),
                hp_setpoint=None,
                clamped_reason="no_output",
                wall_time=wt + 180 + i,
            ))
        # Add more HP-on to force eviction
        for i in range(200):
            buf.add(_make_obs(
                outdoor_temp_c=float(i % 30 - 10),
                hp_setpoint=22.0,
                wall_time=wt + 300 + i,
            ))
        hp_off = sum(1 for o in buf._buffer
                     if o.hp_setpoint is None or o.clamped_reason == "no_output")
        # Leverage should retain some HP-off (they have unique feature vectors)
        assert hp_off > 0, "All HP-off observations were evicted"

    def test_new_temp_range_replaces_redundant(self):
        """Adding observations at a new temp range should evict redundant ones."""
        buf = GreyboxBuffer(max_size=100)
        # Fill with 100 obs all at 10°C
        for i in range(100):
            buf.add(_make_obs(outdoor_temp_c=10.0, wall_time=1_000_000.0 + i))
        # Now add one at -20°C — high leverage, should be accepted
        buf.add(_make_obs(outdoor_temp_c=-20.0, wall_time=1_000_100.0))
        temps = [o.outdoor_temp_c for o in buf.get_all()]
        assert -20.0 in temps, "High-leverage cold observation was rejected"


# ── Simulated year ───────────────────────────────────────────────────


class TestSimulatedYear:
    """Test with realistic seasonal data over a simulated year."""

    def test_year_preserves_full_range(self):
        """After a simulated year, buffer should cover the full temp range."""
        buf = GreyboxBuffer(max_size=600)
        base_wt = 1_000_000.0
        for hour in range(8760):
            day = hour / 24.0
            seasonal = 7.5 + 17.5 * math.sin(2 * math.pi * (day - 80) / 365)
            diurnal = 5.0 * math.sin(2 * math.pi * hour / 24)
            outdoor = seasonal + diurnal

            hp_on = outdoor < 15.0
            buf.add(_make_obs(
                outdoor_temp_c=outdoor,
                hp_setpoint=22.0 if hp_on else None,
                clamped_reason="" if hp_on else "no_output",
                wall_time=base_wt + hour * 3600,
            ))

        assert len(buf) == 600
        temps = [o.outdoor_temp_c for o in buf.get_all()]
        assert min(temps) < -5, f"Lost winter data: min={min(temps)}"
        assert max(temps) > 25, f"Lost summer data: max={max(temps)}"

    def test_year_retains_hp_off(self):
        """HP-off observations should survive a full year of data."""
        buf = GreyboxBuffer(max_size=600)
        base_wt = 1_000_000.0
        for hour in range(8760):
            day = hour / 24.0
            outdoor = 7.5 + 17.5 * math.sin(2 * math.pi * (day - 80) / 365)
            hp_on = outdoor < 15.0
            buf.add(_make_obs(
                outdoor_temp_c=outdoor,
                hp_setpoint=22.0 if hp_on else None,
                clamped_reason="" if hp_on else "no_output",
                wall_time=base_wt + hour * 3600,
            ))

        hp_off = sum(1 for o in buf._buffer
                     if o.hp_setpoint is None or o.clamped_reason == "no_output")
        assert hp_off > 0, "No HP-off observations survived"
        assert hp_off > 20, f"Only {hp_off} HP-off survived — too few"


# ── Persistence ──────────────────────────────────────────────────────


class TestPersistence:
    """Serialization round-trip."""

    def test_round_trip(self):
        buf = GreyboxBuffer(max_size=200)
        for i in range(150):
            buf.add(_make_obs(
                outdoor_temp_c=-10.0 + 40.0 * i / 149,
                wall_time=1_000_000.0 + i,
            ))

        serialized = buf.as_list()
        restored = GreyboxBuffer.from_list(serialized, max_size=200)

        assert len(restored) == len(buf)
        orig_temps = sorted(o.outdoor_temp_c for o in buf.get_all())
        rest_temps = sorted(o.outdoor_temp_c for o in restored.get_all())
        assert orig_temps == rest_temps

    def test_skips_corrupt_entries(self):
        buf = GreyboxBuffer(max_size=100)
        buf.add(_make_obs(outdoor_temp_c=10.0))
        serialized = buf.as_list()
        serialized.append({"garbage": True})
        serialized.append({"v": 1, "old": "format"})

        restored = GreyboxBuffer.from_list(serialized, max_size=100)
        assert len(restored) == 1

    def test_truncates_to_max_size(self):
        buf = GreyboxBuffer(max_size=200)
        for i in range(200):
            buf.add(_make_obs(
                outdoor_temp_c=float(i % 40 - 10),
                wall_time=1_000_000.0 + i,
            ))
        serialized = buf.as_list()
        restored = GreyboxBuffer.from_list(serialized, max_size=100)
        assert len(restored) == 100

    def test_empty_round_trip(self):
        buf = GreyboxBuffer(max_size=100)
        serialized = buf.as_list()
        restored = GreyboxBuffer.from_list(serialized, max_size=100)
        assert len(restored) == 0

    def test_solar_entity_preserved(self):
        buf = GreyboxBuffer(solar_entity="sensor.solar")
        buf.add(_make_obs(outdoor_temp_c=10.0, raw_readings={"sensor.solar": 100.0}))
        serialized = buf.as_list()
        restored = GreyboxBuffer.from_list(serialized, solar_entity="sensor.solar")
        assert restored.solar_entity == "sensor.solar"


# ── Diagnostics ──────────────────────────────────────────────────────


class TestDiagnostics:
    """Test diagnostic output."""

    def test_diagnostics_empty(self):
        buf = GreyboxBuffer(max_size=100)
        diag = buf.get_diagnostics()
        assert diag["total"] == 0
        assert diag["max_size"] == 100
        assert diag["hp_off_pct"] is None

    def test_diagnostics_with_data(self):
        buf = GreyboxBuffer(max_size=300)
        for i in range(200):
            hp_off = i % 5 == 0
            buf.add(_make_obs(
                outdoor_temp_c=float(i % 40 - 10),
                hp_setpoint=None if hp_off else 22.0,
                clamped_reason="no_output" if hp_off else "",
                wall_time=1_000_000.0 + i,
            ))
        diag = buf.get_diagnostics()
        assert diag["total"] == 200
        assert diag["hp_off_pct"] is not None
        assert 15 < diag["hp_off_pct"] < 25
        assert diag["min_leverage"] is not None


# ── Solar entity config ──────────────────────────────────────────────


class TestSolarEntity:
    def test_setter(self):
        buf = GreyboxBuffer(solar_entity="sensor.old")
        assert buf.solar_entity == "sensor.old"
        buf.solar_entity = "sensor.new"
        assert buf.solar_entity == "sensor.new"


# ── get_all ──────────────────────────────────────────────────────────


class TestGetAll:
    def test_returns_copy(self):
        buf = GreyboxBuffer(max_size=100)
        buf.add(_make_obs(outdoor_temp_c=10.0))
        result = buf.get_all()
        result.clear()
        assert len(buf) == 1


# ── BufferAddResult contract ─────────────────────────────────────────


class TestAddReturnsDecision:
    """Same admission-observability contract as DiversityAwareBuffer,
    plus the greybox-specific `no_outdoor_temp` rejection reason that
    fires before leverage is even computed.
    """

    def test_no_outdoor_temp_rejection_returns_named_reason(self):
        buf = GreyboxBuffer(max_size=10)
        r = buf.add(_make_obs(outdoor_temp_c=None))
        # Structural check rather than isinstance — test_batch_learning.py
        # reloads the module which replaces BufferAddResult in module
        # globals, breaking isinstance against the frozen test import.
        assert hasattr(r, "admitted")
        assert r.admitted is False
        assert r.rejection_reason == "no_outdoor_temp"
        assert r.candidate_score is None
        assert r.evicted_timestamp is None
        assert r.min_incumbent_score is None
        assert r.policy_name == "leverage"

    def test_admitted_into_empty_buffer_returns_admitted(self):
        buf = GreyboxBuffer(max_size=10)
        r = buf.add(_make_obs(outdoor_temp_c=5.0))
        assert r.admitted is True
        assert r.rejection_reason is None
        assert r.candidate_score is not None and r.candidate_score >= 0
        assert r.evicted_timestamp is None
        assert r.min_incumbent_score is None
        assert r.policy_name == "leverage"

    def test_full_buffer_with_higher_leverage_admits_and_reports_evicted(self):
        buf = GreyboxBuffer(max_size=4)
        for i in range(4):
            buf.add(_make_obs(outdoor_temp_c=10.0 + i, room_rate=0.001))

        r = buf.add(_make_obs(outdoor_temp_c=-30.0, room_rate=0.05))
        assert r.admitted is True
        assert r.candidate_score is not None
        assert r.min_incumbent_score is not None
        assert r.evicted_timestamp is not None
        assert r.rejection_reason is None

    def test_full_buffer_with_equal_score_rejects_with_policy_reason(self):
        buf = GreyboxBuffer(max_size=4)
        for i in range(4):
            buf.add(_make_obs(outdoor_temp_c=10.0, room_rate=0.001))

        r = buf.add(_make_obs(outdoor_temp_c=10.0, room_rate=0.001))
        assert r.admitted is False
        assert r.rejection_reason == "leverage_rejected"
        assert r.candidate_score is not None
        assert r.min_incumbent_score is not None
        assert r.evicted_timestamp is None
        assert len(buf) == 4
