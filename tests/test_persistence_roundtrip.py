"""Persistence round-trip tests for PIExtraStoredData flags.

The `pi_event_log_enabled` flag (Stage 9) must survive a save → restart →
restore cycle. Otherwise users would have to re-toggle it after every HA
restart. (The `debug_capture_full_p` flag was removed in #117 along with the
RLS P matrix it captured; the stale key from older saves must still load.)
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


def test_pi_event_log_enabled_default_false():
    """New field defaults to False."""
    data = _minimal_stored_data()
    assert data.pi_event_log_enabled is False


def test_pi_event_log_enabled_roundtrips_when_true():
    """as_dict + from_dict preserves pi_event_log_enabled=True."""
    original = _minimal_stored_data(pi_event_log_enabled=True)
    serialized = original.as_dict()
    assert serialized["pi_event_log_enabled"] is True

    restored = PIExtraStoredData.from_dict(serialized)
    assert restored is not None
    assert restored.pi_event_log_enabled is True


def test_legacy_dict_falls_back_and_ignores_removed_field():
    """Restoring from legacy serialized forms succeeds.

    Covers a pre-Stage-9 dict (missing pi_event_log_enabled) and an older
    save that still carries the removed `debug_capture_full_p` key (#117) —
    from_dict must default the missing flag and silently ignore the stale
    key rather than choking on it.
    """
    legacy = _minimal_stored_data().as_dict()
    legacy.pop("pi_event_log_enabled", None)
    legacy["debug_capture_full_p"] = True  # stale key from a pre-#117 save

    restored = PIExtraStoredData.from_dict(legacy)
    assert restored is not None
    assert restored.pi_event_log_enabled is False
    # The removed field is silently ignored — no attribute, no error.
    assert not hasattr(restored, "debug_capture_full_p")


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
