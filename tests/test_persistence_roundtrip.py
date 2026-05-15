"""Persistence round-trip tests for new PIExtraStoredData fields (Stage 11).

The two flags added during the tick-first refactor — `debug_capture_full_p`
(Stage 3) and `pi_event_log_enabled` (Stage 9) — must survive a save →
restart → restore cycle. Otherwise users would have to re-toggle them
after every HA restart.
"""

from __future__ import annotations

import dataclasses

import pytest

from custom_components.tasmota_irhvac.pi.pi_stored_data import PIExtraStoredData


def _minimal_stored_data(**overrides) -> PIExtraStoredData:
    """Build a PIExtraStoredData with sensible defaults; overrides win."""
    base: dict = dict(
        pi_integral=0.0,
        desired_temp=22.0,
        hp_setpoint=22.0,
    )
    base.update(overrides)
    return PIExtraStoredData(**base)


def test_debug_capture_full_p_default_false():
    """New field defaults to False."""
    data = _minimal_stored_data()
    assert data.debug_capture_full_p is False


def test_pi_event_log_enabled_default_false():
    """New field defaults to False."""
    data = _minimal_stored_data()
    assert data.pi_event_log_enabled is False


def test_debug_capture_full_p_roundtrips_when_true():
    """as_dict + from_dict preserves debug_capture_full_p=True."""
    original = _minimal_stored_data(debug_capture_full_p=True)
    serialized = original.as_dict()
    assert serialized["debug_capture_full_p"] is True

    restored = PIExtraStoredData.from_dict(serialized)
    assert restored is not None
    assert restored.debug_capture_full_p is True


def test_pi_event_log_enabled_roundtrips_when_true():
    """as_dict + from_dict preserves pi_event_log_enabled=True."""
    original = _minimal_stored_data(pi_event_log_enabled=True)
    serialized = original.as_dict()
    assert serialized["pi_event_log_enabled"] is True

    restored = PIExtraStoredData.from_dict(serialized)
    assert restored is not None
    assert restored.pi_event_log_enabled is True


def test_legacy_dict_missing_new_fields_falls_back_to_default():
    """Restoring from a legacy serialized form predating these fields succeeds."""
    legacy = _minimal_stored_data().as_dict()
    # Simulate a stored-data dict from before Stages 3 + 9 by deleting
    # the keys (mimics restoration from older versions).
    legacy.pop("debug_capture_full_p", None)
    legacy.pop("pi_event_log_enabled", None)

    restored = PIExtraStoredData.from_dict(legacy)
    assert restored is not None
    # Both default to False — no migration loss
    assert restored.debug_capture_full_p is False
    assert restored.pi_event_log_enabled is False


def test_both_flags_roundtrip_independently():
    """Both flags can be set independently — no aliasing."""
    a = _minimal_stored_data(debug_capture_full_p=True, pi_event_log_enabled=False)
    b = _minimal_stored_data(debug_capture_full_p=False, pi_event_log_enabled=True)

    restored_a = PIExtraStoredData.from_dict(a.as_dict())
    restored_b = PIExtraStoredData.from_dict(b.as_dict())

    assert restored_a is not None and restored_b is not None
    assert restored_a.debug_capture_full_p is True
    assert restored_a.pi_event_log_enabled is False
    assert restored_b.debug_capture_full_p is False
    assert restored_b.pi_event_log_enabled is True


def test_saved_at_wallclock_default_empty():
    """saved_at_wallclock defaults to empty string (signals no prior save)."""
    assert _minimal_stored_data().saved_at_wallclock == ""


def test_saved_at_wallclock_roundtrips():
    """ISO-8601 wallclock survives save → restore."""
    iso = "2026-05-04T12:34:56+00:00"
    original = _minimal_stored_data(saved_at_wallclock=iso)
    restored = PIExtraStoredData.from_dict(original.as_dict())
    assert restored is not None
    assert restored.saved_at_wallclock == iso


def test_legacy_dict_missing_saved_at_wallclock_defaults_empty():
    """Restoring pre-introduction stored data leaves saved_at_wallclock empty."""
    legacy = _minimal_stored_data().as_dict()
    legacy.pop("saved_at_wallclock", None)
    restored = PIExtraStoredData.from_dict(legacy)
    assert restored is not None
    assert restored.saved_at_wallclock == ""


# ── Loud-failure logging on from_dict (pre52 followup) ────────────────


def test_from_dict_missing_required_field_logs_exception(caplog):
    """A bare-swallow used to hide all restore failures.

    Pre-pre52, `from_dict` returned None on any exception with no log
    output — the failure mode behind pi_event_log_enabled silently
    flipping back to default. Now it logs the exception with traceback
    so future regressions are visible.
    """
    broken = _minimal_stored_data().as_dict()
    del broken["pi_integral"]  # required field, bracket-indexed
    with caplog.at_level("ERROR"):
        restored = PIExtraStoredData.from_dict(broken)
    assert restored is None
    failure_logs = [
        r for r in caplog.records
        if "PIExtraStoredData.from_dict failed" in r.message
    ]
    assert len(failure_logs) == 1, (
        f"expected one exception log, got: {[r.message for r in caplog.records]}"
    )
    # _LOGGER.exception attaches traceback info; pytest's caplog
    # exposes it via exc_info.
    assert failure_logs[0].exc_info is not None
    assert failure_logs[0].exc_info[0] is KeyError


def test_from_dict_bad_type_logs_exception(caplog):
    """ValueError from float() coercion is logged the same way."""
    broken = _minimal_stored_data().as_dict()
    broken["pi_integral"] = "not-a-number"
    with caplog.at_level("ERROR"):
        restored = PIExtraStoredData.from_dict(broken)
    assert restored is None
    failure_logs = [
        r for r in caplog.records
        if "PIExtraStoredData.from_dict failed" in r.message
    ]
    assert len(failure_logs) == 1
    assert failure_logs[0].exc_info[0] is ValueError


# ── _compute_prior_run_age_s helper ───────────────────────────────────


def test_compute_prior_run_age_s_returns_none_for_empty():
    from custom_components.tasmota_irhvac.pi.pi_controller import (
        _compute_prior_run_age_s,
    )
    assert _compute_prior_run_age_s("") is None


def test_compute_prior_run_age_s_returns_none_for_garbage():
    from custom_components.tasmota_irhvac.pi.pi_controller import (
        _compute_prior_run_age_s,
    )
    assert _compute_prior_run_age_s("not-an-iso-timestamp") is None


def test_compute_prior_run_age_s_positive_for_past_timestamp():
    from datetime import datetime, timedelta, timezone
    from custom_components.tasmota_irhvac.pi.pi_controller import (
        _compute_prior_run_age_s,
    )
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    age = _compute_prior_run_age_s(past)
    assert age is not None
    # 2h ± a few seconds for test latency
    assert 7195 <= age <= 7210


def test_compute_prior_run_age_s_clamps_negative_to_zero():
    """Future timestamps (clock skew) clamp to 0 instead of going negative."""
    from datetime import datetime, timedelta, timezone
    from custom_components.tasmota_irhvac.pi.pi_controller import (
        _compute_prior_run_age_s,
    )
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert _compute_prior_run_age_s(future) == 0.0


def test_compute_prior_run_age_s_treats_naive_iso_as_utc():
    """Naive ISO timestamps are interpreted as UTC (no tz crash)."""
    from datetime import datetime, timezone
    from custom_components.tasmota_irhvac.pi.pi_controller import (
        _compute_prior_run_age_s,
    )
    naive = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    age = _compute_prior_run_age_s(naive)
    assert age is not None
    assert 0.0 <= age <= 5.0
