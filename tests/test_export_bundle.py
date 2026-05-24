"""Tests for the debug bundle exporter (Stage 10)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import Observation
from custom_components.tasmota_irhvac.pi.event_log import EventLogWriter
from custom_components.tasmota_irhvac.pi.export_bundle import (
    _profile_filter,
    _write_logs_to_bundle,
    _write_manifest_and_readme,
    _write_observation_buffers,
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
    hass.states.get = MagicMock(return_value=None)
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
    # Own controlled room temp (#116) rides the `_` metadata prefix, so the
    # comfort profile carries it — comfort assertions need room temperature.
    assert keep("_current_room_temp_c")
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


# ── Observation buffer dump ────────────────────────────────────────────


def _make_obs(timestamp: float, wall_time: float, current_c: float = 21.0) -> Observation:
    return Observation(
        timestamp=timestamp,
        wall_time=wall_time,
        hp_setpoint=22.0,
        current_c=current_c,
        desired_c=22.0,
        outdoor_temp_c=10.0,
        room_rate=0.001,
        raw_readings={"sensor.outdoor": 10.0},
        clamped=False,
    )


def test_write_observation_buffers_writes_jsonl_per_buffer(tmp_path: Path):
    """Each buffer dumped to `<stem>.jsonl`; counts returned per stem."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    buffers = {
        "observation_buffer_heat": [
            _make_obs(timestamp=1.0, wall_time=1000.0),
            _make_obs(timestamp=2.0, wall_time=1060.0, current_c=21.5),
        ],
        "observation_buffer_cool": [],
        "greybox_buffer": [_make_obs(timestamp=3.0, wall_time=1120.0)],
    }
    counts = _write_observation_buffers(bundle_dir, buffers)
    assert counts == {
        "observation_buffer_heat": 2,
        "observation_buffer_cool": 0,
        "greybox_buffer": 1,
    }
    heat_lines = (bundle_dir / "observation_buffer_heat.jsonl").read_text().splitlines()
    assert len(heat_lines) == 2
    first = json.loads(heat_lines[0])
    # Observation.as_dict format includes schema version + compact keys
    roundtrip = Observation.from_dict(first)
    assert roundtrip.timestamp == 1.0
    assert roundtrip.wall_time == 1000.0
    # Empty buffer file exists but is empty.
    cool_path = bundle_dir / "observation_buffer_cool.jsonl"
    assert cool_path.exists()
    assert cool_path.read_text() == ""


def test_readme_lists_buffer_files_when_present(tmp_path: Path):
    """README mentions each buffer artifact present in manifest counts."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    manifest = {
        "generated_at": "2026-05-04T12:00:00+00:00",
        "zone_label": "climate.living_room",
        "window_days": 1, "window_start": "2026-05-03", "window_end": "2026-05-04",
        "profile": "all", "tick_record_count": 10, "event_record_count": 0,
        "ha_history_record_count": 0,
        "buffer_record_counts": {
            "observation_buffer_heat": 711,
            "observation_buffer_cool": 0,
            "greybox_buffer": 507,
        },
        "integration_version": "0.19.2-pre49",
        "schema_version": 1,
    }
    _write_manifest_and_readme(bundle_dir, manifest, None)
    readme = (bundle_dir / "README.md").read_text()
    assert "observation_buffer_heat.jsonl" in readme
    assert "711" in readme
    assert "greybox_buffer.jsonl" in readme
    assert "507" in readme


# ── End-to-end via export_bundle ──────────────────────────────────────


@pytest.mark.asyncio
async def test_write_ha_history_writes_state_records(hass, tmp_path: Path, monkeypatch):
    """_write_ha_history pulls Recorder states and writes ha_history.jsonl."""
    from custom_components.tasmota_irhvac.pi.export_bundle import _write_ha_history

    # Mock the Recorder API to return a deterministic 2-state history
    fake_states = {
        "sensor.outdoor": [
            MagicMock(state="5.0", last_changed=MagicMock(isoformat=lambda: "2026-05-01T00:00:00")),
            MagicMock(state="6.0", last_changed=MagicMock(isoformat=lambda: "2026-05-01T01:00:00")),
        ],
        "sensor.solar": [
            MagicMock(state="100.0", last_changed=MagicMock(isoformat=lambda: "2026-05-01T00:30:00")),
        ],
    }

    def fake_get_states(_hass, _start, _end, _entities):
        return fake_states

    monkeypatch.setattr(
        "homeassistant.components.recorder.history.get_significant_states",
        fake_get_states,
    )

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    count = await _write_ha_history(
        hass, bundle_dir, ["sensor.outdoor", "sensor.solar"],
        date(2026, 5, 1), date(2026, 5, 1),
    )

    assert count == 3
    history = (bundle_dir / "ha_history.jsonl").read_text().strip().splitlines()
    assert len(history) == 3
    records = [json.loads(line) for line in history]
    assert any(r["entity_id"] == "sensor.outdoor" and r["state"] == "5.0" for r in records)
    assert any(r["entity_id"] == "sensor.solar" for r in records)


def test_read_integration_version_reads_manifest(tmp_path: Path):
    """`_read_integration_version` returns the manifest's version string when readable."""
    from custom_components.tasmota_irhvac.pi.export_bundle import _read_integration_version

    # Stage a manifest.json at the expected path under a fake config_dir
    custom_components_dir = tmp_path / "custom_components" / "tasmota_irhvac"
    custom_components_dir.mkdir(parents=True)
    (custom_components_dir / "manifest.json").write_text(
        '{"version": "test-version"}'
    )

    hass = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.config.config_dir = str(tmp_path)

    assert _read_integration_version(hass) == "test-version"


def test_read_integration_version_falls_back_on_error(tmp_path: Path):
    """`_read_integration_version` returns 'unknown' when the manifest can't be read."""
    from custom_components.tasmota_irhvac.pi.export_bundle import _read_integration_version

    hass = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.config.config_dir = str(tmp_path / "nonexistent")

    assert _read_integration_version(hass) == "unknown"


@pytest.mark.asyncio
async def test_climate_handler_invokes_export_bundle(hass, tmp_path: Path, monkeypatch):
    """async_export_debug_bundle handler resolves entity_ids + calls exporter + notifies."""
    from .conftest import get_climate_entity, setup_pi_integration as _setup

    # Use the standard fixture indirectly — we need a real PI integration
    # to drive the handler.
    monkeypatch.setattr(
        hass.config, "path", lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    # Register persistent_notification.create stub so the handler can fire it
    hass.services.async_register(
        "persistent_notification", "create", lambda call: None,
    )


@pytest.mark.asyncio
async def test_climate_handler_no_pi_logs_warning(hass, setup_integration, caplog):
    """async_export_debug_bundle on a non-PI entity logs warning and returns."""
    from .conftest import get_climate_entity

    entry = await setup_integration({"pi_enabled": False})
    entity = get_climate_entity(hass, entry)

    await entity.async_export_debug_bundle(window_days=1)
    # No raise; just early-return + warning log
    assert any("no PI controller" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_climate_handler_full_bundle(hass, setup_pi_integration, tmp_path, monkeypatch):
    """Full handler path: resolves history_ids, calls exporter, fires notification."""
    from .conftest import get_climate_entity

    monkeypatch.setattr(
        hass.config, "path", lambda *parts: str(tmp_path.joinpath(*parts)),
    )
    hass.services.async_register(
        "persistent_notification", "create", lambda call: None,
    )

    # Patch the recorder to return empty history (no need for real data)
    monkeypatch.setattr(
        "homeassistant.components.recorder.history.get_significant_states",
        lambda *args, **kwargs: {},
    )

    # Configure a model input so the for-loop in the handler executes
    entry = await setup_pi_integration({
        "pi_tau_estimate": 60,
        "pi_model_inputs": [{
            "name": "solar",
            "entity_id": "sensor.solar_proxy",
            "input_role": "solar",
        }],
    })
    entity = get_climate_entity(hass, entry)

    # Trigger the full handler — should not raise
    await entity.async_export_debug_bundle(
        window_days=1, profile="all", include_ha_history=True,
    )

    # Verify a bundle directory was created
    bundle_root = tmp_path / "tasmota_irhvac" / "bundles"
    assert bundle_root.exists()
    bundles = list(bundle_root.iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "manifest.json").exists()


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
