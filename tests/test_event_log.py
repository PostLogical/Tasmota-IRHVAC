"""Tests for EventLogWriter and EventLogReader (Stage 9)."""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from custom_components.tasmota_irhvac.pi.event_log import (
    EventLogReader,
    EventLogWriter,
    _sanitize_zone_label,
)
from custom_components.tasmota_irhvac.pi.snapshot import (
    ModeChangePayload,
    TickEvent,
    TickEventKind,
    TickOutput,
)


# ── Filename sanitization ─────────────────────────────────────────────


def test_sanitize_strips_dots_and_special_chars():
    assert _sanitize_zone_label("climate.living_room") == "climate_living_room"
    assert _sanitize_zone_label("foo/bar:baz") == "foo_bar_baz"


def test_sanitize_empty_falls_back():
    assert _sanitize_zone_label("") == "unknown_zone"


def test_sanitize_preserves_alphanumeric_and_underscores():
    assert _sanitize_zone_label("climate-test_1") == "climate-test_1"


# ── Writer ─────────────────────────────────────────────────────────────


def _stub_hass_with_sync_executor() -> MagicMock:
    """Return a MagicMock hass whose async_add_executor_job runs jobs synchronously."""
    hass = MagicMock()
    # Sync executor — runs the function inline so test assertions can
    # inspect the file immediately.
    def run_sync(fn, *args):
        return fn(*args)
    hass.async_add_executor_job = run_sync
    return hass


def _basic_tick(zone: str = "test") -> TickOutput:
    return TickOutput.empty(zone_label=zone)


def test_writer_creates_daily_file(tmp_path: Path):
    """First append creates today's .jsonl file with one record."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "climate.living_room")

    tick = _basic_tick("living_room")
    writer.append(tick)

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("climate_living_room_")
    assert files[0].suffix == ".jsonl"


def test_writer_appends_one_record_per_call(tmp_path: Path):
    """Multiple appends produce multiple JSONL lines in the same file."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    for _ in range(3):
        writer.append(_basic_tick())

    [path] = list(tmp_path.iterdir())
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    # Each line is valid JSON of a TickOutput
    for line in lines:
        data = json.loads(line)
        assert data["_schema_version"] == 1


def test_writer_gzips_yesterday_on_rollover(tmp_path: Path):
    """When date changes, prior day's file is gzipped in place."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    # First write: yesterday's date
    yesterday = date(2026, 5, 1)
    writer._sync_append(_basic_tick(), yesterday)
    yesterday_file = tmp_path / f"test_{yesterday.isoformat()}.jsonl"
    assert yesterday_file.exists()

    # Second write: today — triggers rotation of yesterday
    today = date(2026, 5, 2)
    writer._sync_append(_basic_tick(), today)

    # Yesterday's file is now gzipped, today's is plain
    assert not yesterday_file.exists()
    assert (tmp_path / f"test_{yesterday.isoformat()}.jsonl.gz").exists()
    assert (tmp_path / f"test_{today.isoformat()}.jsonl").exists()


def test_writer_creates_log_dir_if_missing(tmp_path: Path):
    """Writer creates the log directory if it doesn't exist."""
    log_dir = tmp_path / "deep" / "nested" / "log"
    assert not log_dir.exists()

    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, log_dir, "test")
    writer.append(_basic_tick())

    assert log_dir.exists()
    assert any(log_dir.iterdir())


def test_writer_handles_corrupted_prior_day_gracefully(tmp_path: Path, caplog):
    """If gzip of yesterday's file fails, today's append still succeeds."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    # Seed an unreadable yesterday file by making it a directory (gzip fails)
    yesterday = date(2026, 5, 1)
    fake_path = tmp_path / f"test_{yesterday.isoformat()}.jsonl"
    fake_path.mkdir()  # Now gzip will fail trying to open as a file
    writer._current_date = yesterday

    today = date(2026, 5, 2)
    writer._sync_append(_basic_tick(), today)

    # Today's file should still have been written
    today_file = tmp_path / f"test_{today.isoformat()}.jsonl"
    assert today_file.exists()
    assert today_file.read_text().strip()


# ── Reader ─────────────────────────────────────────────────────────────


def test_reader_yields_ticks_from_jsonl(tmp_path: Path):
    """Reader recovers TickOutput records from a plain .jsonl file."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    tick = _basic_tick("living_room")
    writer.append(tick)
    writer.append(tick)

    reader = EventLogReader(tmp_path, "test")
    ticks = list(reader.iter_ticks())
    assert len(ticks) == 2
    assert all(t.zone_label == "living_room" for t in ticks)


def test_reader_yields_ticks_from_gzip(tmp_path: Path):
    """Reader recovers records from rotated .jsonl.gz files."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    # Force a rotation so we have a .gz file
    yesterday = date(2026, 5, 1)
    writer._sync_append(_basic_tick(), yesterday)
    today = date(2026, 5, 2)
    writer._sync_append(_basic_tick(), today)

    reader = EventLogReader(tmp_path, "test")
    ticks = list(reader.iter_ticks())
    assert len(ticks) == 2  # one from gz, one from current jsonl


def test_reader_filters_by_date_range(tmp_path: Path):
    """start/end bounds skip files outside the range."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    for d in [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)]:
        writer._sync_append(_basic_tick(), d)

    reader = EventLogReader(tmp_path, "test")
    middle_only = list(reader.iter_ticks(
        start=date(2026, 5, 2), end=date(2026, 5, 2),
    ))
    assert len(middle_only) == 1


def test_reader_skips_malformed_lines(tmp_path: Path, caplog):
    """A garbled line in the middle of a file doesn't kill the reader."""
    path = tmp_path / "test_2026-05-01.jsonl"
    valid_line = json.dumps(_basic_tick().to_dict())
    path.write_text(f"{valid_line}\nNOT JSON\n{valid_line}\n")

    reader = EventLogReader(tmp_path, "test")
    ticks = list(reader.iter_ticks())
    assert len(ticks) == 2


def test_reader_returns_empty_when_log_dir_missing(tmp_path: Path):
    """Missing log dir is not an error — yields nothing."""
    reader = EventLogReader(tmp_path / "nonexistent", "test")
    assert list(reader.iter_ticks()) == []


def test_reader_zone_label_matches_writer_after_sanitize(tmp_path: Path):
    """Reader uses sanitized zone label so it matches writer's filenames."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "climate.living_room")
    writer.append(_basic_tick())

    # Reader called with the entity-id form should still find the file
    reader = EventLogReader(tmp_path, "climate.living_room")
    assert list(reader.iter_ticks())


# ── Roundtrip ─────────────────────────────────────────────────────────


def test_writer_handles_oserror_on_write(tmp_path: Path, caplog, monkeypatch):
    """If the file write itself fails, the writer logs and moves on."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    # Force the open() inside _sync_append to raise OSError.
    real_path_open = Path.open
    call_count = {"n": 0}
    def maybe_fail_open(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:  # First call (the actual write)
            raise OSError("disk full")
        return real_path_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", maybe_fail_open)

    # Should not raise — writer logs and returns
    writer._sync_append(_basic_tick(), date(2026, 5, 1))


def test_gzip_file_no_op_when_src_missing(tmp_path: Path):
    """Calling _gzip_file with a path that doesn't exist is a no-op (defensive)."""
    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")
    writer._gzip_file(tmp_path / "nonexistent.jsonl")
    # No file created, no exception raised
    assert not list(tmp_path.iterdir())


def test_reader_skips_non_file_entries(tmp_path: Path):
    """A subdirectory in the log dir is ignored."""
    (tmp_path / "test_2026-05-01.jsonl").write_text("")
    (tmp_path / "test_subdir").mkdir()  # not a file — should be skipped

    reader = EventLogReader(tmp_path, "test")
    files = list(reader._iter_files_in_range(None, None))
    assert all(p.is_file() for p in files)


def test_reader_skips_wrong_prefix(tmp_path: Path):
    """Files for other zones are skipped."""
    (tmp_path / "test_2026-05-01.jsonl").write_text("")
    (tmp_path / "other_zone_2026-05-01.jsonl").write_text("")

    reader = EventLogReader(tmp_path, "test")
    files = list(reader._iter_files_in_range(None, None))
    assert len(files) == 1
    assert "test_" in files[0].name
    assert "other_zone" not in files[0].name


def test_reader_skips_files_with_unrecognized_suffix(tmp_path: Path):
    """Files that don't end with .jsonl[.gz] are skipped."""
    (tmp_path / "test_2026-05-01.jsonl").write_text("")
    (tmp_path / "test_2026-05-02.txt").write_text("")  # wrong suffix

    reader = EventLogReader(tmp_path, "test")
    files = list(reader._iter_files_in_range(None, None))
    assert len(files) == 1


def test_reader_skips_files_with_malformed_date(tmp_path: Path):
    """A file matching the prefix but with a non-date in place of YYYY-MM-DD is skipped."""
    (tmp_path / "test_2026-05-01.jsonl").write_text("")
    (tmp_path / "test_NOT_A_DATE.jsonl").write_text("")

    reader = EventLogReader(tmp_path, "test")
    files = list(reader._iter_files_in_range(None, None))
    assert len(files) == 1


def test_reader_skips_empty_lines(tmp_path: Path):
    """Empty / whitespace-only lines in a JSONL file are skipped without warning."""
    valid = json.dumps(_basic_tick().to_dict())
    (tmp_path / "test_2026-05-01.jsonl").write_text(f"{valid}\n\n   \n{valid}\n")

    reader = EventLogReader(tmp_path, "test")
    ticks = list(reader.iter_ticks())
    assert len(ticks) == 2


def test_reader_handles_oserror_on_read(tmp_path: Path, caplog, monkeypatch):
    """File-open OSError during read is logged; reader continues."""
    path = tmp_path / "test_2026-05-01.jsonl"
    path.write_text(json.dumps(_basic_tick().to_dict()) + "\n")

    real_open = open
    def failing_open(*args, **kwargs):
        raise OSError("permission denied")
    monkeypatch.setattr("builtins.open", failing_open)

    reader = EventLogReader(tmp_path, "test")
    # Should not raise — yields nothing
    assert list(reader.iter_ticks()) == []


@pytest.mark.asyncio
async def test_controller_set_event_log_enabled_creates_writer(
    hass, setup_pi_integration, tmp_path: Path, monkeypatch,
):
    """Toggling the flag via controller wires the writer on next fire_dispatcher."""
    from .conftest import get_climate_entity
    from custom_components.tasmota_irhvac.pi.event_log import EventLogWriter

    monkeypatch.setattr(
        hass.config, "path", lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # Initially disabled — no writer
    assert pi._event_log_writer is None

    # Enable + fire — lazy-creates the writer
    pi.set_event_log_enabled(enabled=True)
    pi.fire_dispatcher()

    assert isinstance(pi._event_log_writer, EventLogWriter)


@pytest.mark.asyncio
async def test_controller_set_event_log_disabled_drops_writer(
    hass, setup_pi_integration, tmp_path: Path, monkeypatch,
):
    """Disabling the flag drops the cached writer reference."""
    from .conftest import get_climate_entity

    monkeypatch.setattr(
        hass.config, "path", lambda *parts: str(tmp_path.joinpath(*parts)),
    )

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    pi.set_event_log_enabled(enabled=True)
    pi.fire_dispatcher()
    assert pi._event_log_writer is not None

    pi.set_event_log_enabled(enabled=False)
    assert pi._event_log_writer is None


@pytest.mark.asyncio
async def test_climate_handler_set_event_log_enabled(hass, setup_pi_integration):
    """Climate handler async_set_event_log_enabled toggles the controller flag."""
    from .conftest import get_climate_entity

    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    entity = get_climate_entity(hass, entry)
    pi = entity._controller

    assert pi._pi_event_log_enabled is False
    await entity.async_set_event_log_enabled(enabled=True)
    assert pi._pi_event_log_enabled is True
    await entity.async_set_event_log_enabled(enabled=False)
    assert pi._pi_event_log_enabled is False


def test_writer_reader_roundtrip_preserves_events(tmp_path: Path):
    """A tick with typed events roundtrips through writer + reader losslessly."""
    import dataclasses

    hass = _stub_hass_with_sync_executor()
    writer = EventLogWriter(hass, tmp_path, "test")

    tick = dataclasses.replace(
        _basic_tick("living_room"),
        events=(
            TickEvent(
                kind=TickEventKind.MODE_CHANGE,
                payload=ModeChangePayload(from_mode="off", to_mode="heat"),
            ),
        ),
    )
    writer.append(tick)

    reader = EventLogReader(tmp_path, "test")
    [recovered] = list(reader.iter_ticks())
    assert recovered.zone_label == "living_room"
    assert len(recovered.events) == 1
    assert recovered.events[0].kind == TickEventKind.MODE_CHANGE
    assert recovered.events[0].payload.to_mode == "heat"
