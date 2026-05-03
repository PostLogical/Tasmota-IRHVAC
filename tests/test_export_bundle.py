"""Tests for the debug bundle exporter (Stage 10)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from custom_components.tasmota_irhvac.pi.event_log import EventLogWriter
from custom_components.tasmota_irhvac.pi.export_bundle import (
    _profile_filter,
    _write_logs_to_bundle,
    _write_manifest_and_readme,
)
from custom_components.tasmota_irhvac.pi.snapshot import (
    ModeChangePayload,
    TickEvent,
    TickEventKind,
    TickOutput,
)


def _basic_tick(zone: str = "test") -> TickOutput:
    return TickOutput.empty(zone_label=zone)


def _stub_hass_with_sync_executor() -> MagicMock:
    hass = MagicMock()
    def run_sync(fn, *args):
        return fn(*args)
    hass.async_add_executor_job = run_sync
    return hass


# ── Profile filter ─────────────────────────────────────────────────────


def test_profile_filter_all_returns_none():
    """`all` profile = no filtering (returns None sentinel)."""
    assert _profile_filter("all") is None


def test_profile_filter_unknown_falls_back_to_all():
    """Unknown profile name logs warning and returns None (no filtering)."""
    assert _profile_filter("nonexistent") is None


def test_profile_filter_rls_drift_keeps_rls_fields():
    keep = _profile_filter("rls_drift")
    assert keep is not None
    assert keep("rls_model")
    assert keep("batch_learning")
    assert keep("ff_contributions")
    # Filtered out: high-level state-machine summaries not in the profile
    assert not keep("health")
    assert not keep("learning")


def test_profile_filter_keeps_underscore_metadata():
    """Profile filters always retain underscore-prefixed metadata fields."""
    keep = _profile_filter("comfort")
    assert keep is not None
    assert keep("_schema_version")
    assert keep("_ts_mono")
    assert keep("_zone_label")


def test_profile_filter_comfort_keeps_pi_state():
    keep = _profile_filter("comfort")
    assert keep is not None
    assert keep("integral")
    assert keep("ff_offset")
    assert keep("hp_setpoint")
    assert keep("desired_temp")
    assert not keep("rls_model")


def test_profile_filter_phase4_keeps_estimation_fields():
    keep = _profile_filter("phase4_validation")
    assert keep is not None
    assert keep("rls_model")
    assert keep("greybox_observer")
    assert keep("plant_identification")
    assert keep("boundary_estimator")


# ── _write_logs_to_bundle ──────────────────────────────────────────────


def test_write_logs_writes_tick_and_event_files(tmp_path: Path):
    """Bundle writer creates tick_log.jsonl + event_log.jsonl with correct counts."""
    import dataclasses

    log_dir = tmp_path / "log"
    bundle_dir = tmp_path / "bundle"
    hass = _stub_hass_with_sync_executor()

    writer = EventLogWriter(hass, log_dir, "test")
    plain_tick = _basic_tick()
    tick_with_event = dataclasses.replace(
        _basic_tick(),
        events=(
            TickEvent(
                kind=TickEventKind.MODE_CHANGE,
                payload=ModeChangePayload(from_mode="off", to_mode="heat"),
            ),
        ),
    )
    writer._sync_append(plain_tick, date(2026, 5, 1))
    writer._sync_append(tick_with_event, date(2026, 5, 1))

    tick_count, event_count = _write_logs_to_bundle(
        bundle_dir, log_dir, "test",
        date(2026, 5, 1), date(2026, 5, 1),
        profile="all",
    )

    assert tick_count == 2
    assert event_count == 1

    tick_lines = (bundle_dir / "tick_log.jsonl").read_text().strip().splitlines()
    event_lines = (bundle_dir / "event_log.jsonl").read_text().strip().splitlines()
    assert len(tick_lines) == 2
    assert len(event_lines) == 1

    event_record = json.loads(event_lines[0])
    assert event_record["kind"] == "mode_change"
    assert event_record["zone_label"] == "test"
    assert event_record["payload"]["to_mode"] == "heat"


def test_write_logs_respects_window(tmp_path: Path):
    """Files outside the window are skipped."""
    log_dir = tmp_path / "log"
    bundle_dir = tmp_path / "bundle"
    hass = _stub_hass_with_sync_executor()

    writer = EventLogWriter(hass, log_dir, "test")
    for d in [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)]:
        writer._sync_append(_basic_tick(), d)

    tick_count, _ = _write_logs_to_bundle(
        bundle_dir, log_dir, "test",
        date(2026, 5, 2), date(2026, 5, 2),
        profile="all",
    )
    assert tick_count == 1


def test_write_logs_applies_profile(tmp_path: Path):
    """Profile filter slims tick records to the keep set + underscore metadata."""
    log_dir = tmp_path / "log"
    bundle_dir = tmp_path / "bundle"
    hass = _stub_hass_with_sync_executor()

    writer = EventLogWriter(hass, log_dir, "test")
    writer._sync_append(_basic_tick(), date(2026, 5, 1))

    _write_logs_to_bundle(
        bundle_dir, log_dir, "test",
        date(2026, 5, 1), date(2026, 5, 1),
        profile="comfort",
    )

    [line] = (bundle_dir / "tick_log.jsonl").read_text().strip().splitlines()
    record = json.loads(line)
    # comfort profile keeps integral, ff_offset, etc., plus all `_*` metadata
    assert "integral" in record
    assert "_schema_version" in record
    # Should NOT have rls_model
    assert "rls_model" not in record


# ── Manifest / README ─────────────────────────────────────────────────


def test_manifest_and_readme_written(tmp_path: Path):
    """Manifest and README are produced with the requested fields."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest = {
        "generated_at": "2026-05-02T12:00:00+00:00",
        "zone_label": "climate.living_room",
        "window_days": 7,
        "window_start": "2026-04-25",
        "window_end": "2026-05-02",
        "profile": "all",
        "tick_record_count": 100,
        "event_record_count": 5,
        "ha_history_record_count": 250,
        "integration_version": "0.19.2-pre47",
        "schema_version": 1,
    }
    _write_manifest_and_readme(bundle_dir, manifest, ["sensor.outdoor"])

    written = json.loads((bundle_dir / "manifest.json").read_text())
    assert written == manifest

    readme = (bundle_dir / "README.md").read_text()
    assert "climate.living_room" in readme
    assert "100 TickOutput" in readme
    assert "5 sparse" in readme  # event count
    assert "ha_history.jsonl" in readme  # since history_entity_ids was non-empty


def test_readme_omits_ha_history_section_when_no_entities(tmp_path: Path):
    """When include_ha_history is False, the README's history section is dropped."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest = {
        "generated_at": "2026-05-02T12:00:00+00:00",
        "zone_label": "climate.test",
        "window_days": 1, "window_start": "2026-05-01", "window_end": "2026-05-02",
        "profile": "all", "tick_record_count": 0, "event_record_count": 0,
        "ha_history_record_count": 0, "integration_version": "test",
        "schema_version": 1,
    }
    _write_manifest_and_readme(bundle_dir, manifest, None)
    readme = (bundle_dir / "README.md").read_text()
    assert "ha_history.jsonl" not in readme


# ── End-to-end via export_bundle ──────────────────────────────────────


@pytest.mark.asyncio
async def test_export_bundle_end_to_end(hass, tmp_path: Path, monkeypatch):
    """export_bundle produces a complete directory layout."""
    from custom_components.tasmota_irhvac.pi.export_bundle import export_bundle

    # Redirect <config> path resolution to tmp_path so the bundle and
    # log dirs land somewhere we can inspect.
    monkeypatch.setattr(
        hass.config, "path", lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    # Pre-populate the event log
    log_dir = tmp_path / "tasmota_irhvac" / "log"
    writer = EventLogWriter(hass, log_dir, "test_zone")
    today = date.today()
    writer._sync_append(_basic_tick("test_zone"), today)

    bundle_path = await export_bundle(
        hass,
        zone_label="test_zone",
        window_days=1,
        profile="all",
        include_ha_history=False,
    )

    assert bundle_path.exists()
    assert (bundle_path / "tick_log.jsonl").exists()
    assert (bundle_path / "event_log.jsonl").exists()
    assert (bundle_path / "manifest.json").exists()
    assert (bundle_path / "README.md").exists()

    manifest = json.loads((bundle_path / "manifest.json").read_text())
    assert manifest["zone_label"] == "test_zone"
    assert manifest["window_days"] == 1
    assert manifest["tick_record_count"] >= 1
