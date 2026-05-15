"""Persistent event log for typed TickOutput records.

Append-only JSONL log under `<config>/tasmota_irhvac/log/`, rotated
daily and gzipped after rollover. The original goal of the tick-first
refactor: long-term per-tick history that analysis tools can recover
weeks or months later.

Schema versioning lives on `TickOutput.SCHEMA_VERSION`. Records on disk
include the version field so a future-version reader can migrate them.

Configuration: opt-in via the `pi_event_log_enabled` flag in
`PIExtraStoredData`. Default off so existing installations don't start
writing files without the user choosing it. Retention is unlimited by
default — for the user who's been waiting on long-term debug data,
that's the goal — with optional `max_size_gb` cap for users who want
one (not implemented in this stage; future enhancement).

Disk pressure: ~3 GB/year compressed for 4 zones at 1-min ticks.

Threading: file I/O happens on HA's executor (never blocks the event
loop). The sync `append()` entry point hands off to an executor job;
the caller returns immediately. The reader is sync — analysis tools
typically run synchronously after the integration is stopped or in a
service handler that explicitly awaits.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
from collections import Counter
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

from .snapshot import TickOutput

_LOGGER = logging.getLogger(__name__)


# Match anything that's not safe in a filename; replaces with _.
# Dots are intentionally NOT in the safe set — they'd create secondary
# extensions in `<zone>_<date>.jsonl` filenames (e.g., the entity_id
# "climate.living_room" would produce "climate.living_room_2026-05-02.jsonl"
# which globs as "climate.*").
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _sanitize_zone_label(zone_label: str) -> str:
    """Convert an entity_id-like zone label to a safe filename component.

    `climate.living_room_heat_pump` → `climate_living_room_heat_pump`.
    Falls back to `unknown_zone` if the input is empty.
    """
    if not zone_label:
        return "unknown_zone"
    return _UNSAFE_FILENAME_RE.sub("_", zone_label)


class EventLogWriter:
    """Append-only JSONL writer with daily rotation + gzip-on-rollover."""

    def __init__(
        self,
        hass: HomeAssistant,
        log_dir: Path,
        zone_label: str,
    ) -> None:
        self._hass = hass
        self._log_dir = log_dir
        self._zone_label = _sanitize_zone_label(zone_label)
        # Lazy: detected on first write. None means "haven't written yet."
        self._current_date: date | None = None

    def append(self, tick: TickOutput) -> None:
        """Schedule an append. Returns immediately; I/O runs in the executor.

        Caller is typically a coordinator listener (sync). The actual
        write is offloaded to HA's executor pool.
        """
        # Capture today's date here (event-loop side) so the executor
        # job sees a consistent value. async_add_executor_job returns a
        # task we don't await — fire-and-forget.
        today = date.today()
        self._hass.async_add_executor_job(self._sync_append, tick, today)

    def _sync_append(self, tick: TickOutput, today: date) -> None:
        """Synchronous: detect rollover, gzip prior day, append to today's file.

        Runs on HA's executor — never on the event loop. Defensive
        against rotation/gzip failures: a corrupted prior-day file
        shouldn't stop today's writes from landing.
        """
        if self._current_date is not None and today != self._current_date:
            try:
                self._gzip_file(self._daily_path(self._current_date))
            except Exception:  # noqa: BLE001 — defense in depth
                _LOGGER.exception(
                    "Failed to gzip previous day's log %s — continuing",
                    self._daily_path(self._current_date),
                )
        self._current_date = today

        path = self._daily_path(today)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(tick.to_dict(), default=str))
                f.write("\n")
        except OSError:
            _LOGGER.exception("Failed to append tick to %s", path)

    def _daily_path(self, day: date) -> Path:
        return self._log_dir / f"{self._zone_label}_{day.isoformat()}.jsonl"

    def _gzip_file(self, src: Path) -> None:
        """Gzip-in-place: writes src.gz, removes src on success."""
        if not src.exists():
            return
        gz_path = src.with_suffix(src.suffix + ".gz")
        with src.open("rb") as src_f, gzip.open(gz_path, "wb") as gz_f:
            shutil.copyfileobj(src_f, gz_f)
        src.unlink()
        _LOGGER.info("Rotated event log: %s → %s", src.name, gz_path.name)


class EventLogReader:
    """Iterates TickOutput records from the log directory.

    Sync — reading is typically done in service handlers (which await
    the result) or in offline analysis scripts (which run outside HA).
    Handles both `.jsonl` (today's) and `.jsonl.gz` (rotated) files
    transparently.
    """

    def __init__(self, log_dir: Path, zone_label: str) -> None:
        self._log_dir = log_dir
        self._zone_label = _sanitize_zone_label(zone_label)

    def iter_ticks(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> Iterator[TickOutput]:
        """Yield TickOutputs in chronological order across the date range.

        `start` / `end` inclusive. None = no bound on that side. Files
        outside the range are skipped without opening.
        """
        for path in self._iter_files_in_range(start, end):
            yield from self._read_file(path)

    def _iter_files_in_range(
        self, start: date | None, end: date | None,
    ) -> Iterator[Path]:
        """Yield log file paths sorted by date, filtered by the range."""
        if not self._log_dir.exists():
            return
        prefix = f"{self._zone_label}_"
        candidates: list[tuple[date, Path]] = []
        for entry in self._log_dir.iterdir():
            if not entry.is_file():
                continue
            name = entry.name
            if not name.startswith(prefix):
                continue
            # Strip prefix and the .jsonl[.gz] suffix to get YYYY-MM-DD
            stem = name[len(prefix):]
            if stem.endswith(".jsonl.gz"):
                date_str = stem[:-len(".jsonl.gz")]
            elif stem.endswith(".jsonl"):
                date_str = stem[:-len(".jsonl")]
            else:
                continue
            try:
                day = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            candidates.append((day, entry))
        for _, path in sorted(candidates):
            yield path

    def _read_file(self, path: Path) -> Iterator[TickOutput]:
        """Yield TickOutputs from a single .jsonl or .jsonl.gz file.

        Per-line parse failures are aggregated and logged once at file
        close — the prior per-line warning was noisy enough to hide
        actionable schema-drift signals (e.g. a uniform ``KeyError`` on
        one renamed field across thousands of lines). The summary names
        the field on ``KeyError`` so future schema drift is
        self-diagnosing.
        """
        opener: object
        if path.suffix == ".gz":
            opener = gzip.open
        else:
            opener = open
        parse_errors: Counter[str] = Counter()
        ok_count = 0
        try:
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield TickOutput.from_dict(data)
                        ok_count += 1
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        detail = (
                            f"KeyError({e!s})" if isinstance(e, KeyError)
                            else type(e).__name__
                        )
                        parse_errors[detail] += 1
                        continue
            if parse_errors:
                _LOGGER.warning(
                    "Event log %s: %d records ok, %d skipped (%s)",
                    path.name, ok_count, sum(parse_errors.values()),
                    ", ".join(
                        f"{k}×{v}"
                        for k, v in parse_errors.most_common(5)
                    ),
                )
        except OSError:
            _LOGGER.exception("Failed to read event log %s", path)
