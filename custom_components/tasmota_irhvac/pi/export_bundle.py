"""Debug bundle exporter (Stage 10).

Packages a window of tick-log records + (optional) HA Recorder history
into a directory that analysis tools can pick up. Triggered by the
`tasmota_irhvac.export_debug_bundle` service.

Bundle layout under `<config>/tasmota_irhvac/bundles/<timestamp>_<zone>/`:
- `tick_log.jsonl` — TickOutput records for the requested window
- `event_log.jsonl` — events extracted from the same window (sparse)
- `ha_history.jsonl` — optional HA-side state history (room temp,
  outdoor, solar, model inputs) joined for cross-correlation
- `manifest.json` — window, zone, profile, schema versions, generated-at
- `README.md` — auto-generated; what each file contains, how to load

All file I/O runs on HA's executor (never blocks the event loop).
Recorder access uses the proper async APIs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .event_log import EventLogReader, _sanitize_zone_label
from .snapshot import TickEventKind, TickOutput

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Integration version — manifest references it so future readers can
# correlate bundle contents with the integration version that produced
# them.
_INTEGRATION_VERSION_FALLBACK = "unknown"


def _read_integration_version(hass: HomeAssistant) -> str:
    """Read version from manifest.json. Returns "unknown" on any failure."""
    try:
        manifest_path = (
            Path(hass.config.config_dir)
            / "custom_components"
            / "tasmota_irhvac"
            / "manifest.json"
        )
        data = json.loads(manifest_path.read_text())
        return str(data.get("version", _INTEGRATION_VERSION_FALLBACK))
    except (OSError, json.JSONDecodeError, KeyError):
        return _INTEGRATION_VERSION_FALLBACK


async def export_bundle(
    hass: HomeAssistant,
    *,
    zone_label: str,
    window_days: int,
    profile: str = "all",
    include_ha_history: bool = True,
    ha_history_entity_ids: list[str] | None = None,
) -> Path:
    """Generate a debug bundle for one zone over the requested window.

    Returns the bundle directory path. Writes are async-safe (executor
    for file I/O; native async for Recorder).

    `ha_history_entity_ids` is the list of HA-side entities (room temp
    sensor, outdoor temp, solar proxy, model inputs) to include in the
    history join. If None, only tick log + manifest are emitted.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=max(window_days, 0))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_zone = _sanitize_zone_label(zone_label)
    bundle_dir = (
        Path(hass.config.path("tasmota_irhvac/bundles"))
        / f"{timestamp}_{safe_zone}"
    )

    log_dir = Path(hass.config.path("tasmota_irhvac/log"))
    integration_version = _read_integration_version(hass)

    # Read tick log on the executor — file I/O may be substantial
    tick_count, event_count = await hass.async_add_executor_job(
        _write_logs_to_bundle,
        bundle_dir, log_dir, zone_label, start_date, end_date, profile,
    )

    history_count = 0
    if include_ha_history and ha_history_entity_ids:
        history_count = await _write_ha_history(
            hass, bundle_dir, ha_history_entity_ids, start_date, end_date,
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zone_label": zone_label,
        "window_days": window_days,
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "profile": profile,
        "tick_record_count": tick_count,
        "event_record_count": event_count,
        "ha_history_record_count": history_count,
        "integration_version": integration_version,
        "schema_version": TickOutput.SCHEMA_VERSION,
    }
    await hass.async_add_executor_job(
        _write_manifest_and_readme, bundle_dir, manifest, ha_history_entity_ids,
    )

    _LOGGER.info(
        "Debug bundle written to %s (%d ticks, %d events, %d history)",
        bundle_dir, tick_count, event_count, history_count,
    )
    return bundle_dir


def _write_logs_to_bundle(
    bundle_dir: Path,
    log_dir: Path,
    zone_label: str,
    start: date,
    end: date,
    profile: str,
) -> tuple[int, int]:
    """Sync helper (runs on executor): write tick + event logs to the bundle.

    Returns (tick_count, event_count).
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    reader = EventLogReader(log_dir, zone_label)

    tick_path = bundle_dir / "tick_log.jsonl"
    event_path = bundle_dir / "event_log.jsonl"

    tick_count = 0
    event_count = 0
    keep_field = _profile_filter(profile)

    with tick_path.open("w", encoding="utf-8") as tick_f, \
         event_path.open("w", encoding="utf-8") as event_f:
        for tick in reader.iter_ticks(start=start, end=end):
            tick_dict = tick.to_dict()
            if keep_field is not None:
                tick_dict = {k: v for k, v in tick_dict.items() if keep_field(k)}
            tick_f.write(json.dumps(tick_dict, default=str))
            tick_f.write("\n")
            tick_count += 1
            # Event records: sparse stream, one line per emitted event
            for event in tick.events:
                event_record = {
                    "ts_mono": tick.ts_mono,
                    "ts_wall": tick.ts_wall,
                    "zone_label": tick.zone_label,
                    **event.to_dict(),
                }
                event_f.write(json.dumps(event_record, default=str))
                event_f.write("\n")
                event_count += 1

    return tick_count, event_count


def _profile_filter(
    profile: str,
) -> "Callable[[str], bool] | None":
    """Return a `keep(field_name)` predicate for the requested profile.

    None = keep everything (the "all" profile, which is the default).
    Profiles are field-selectors over the TickOutput shape — the
    underlying log captures everything, profiles just slim down the
    export for specific debugging contexts.
    """
    if profile == "all":
        return None

    # Profile field-selectors. New profiles: add a key here. Each value
    # is the set of TICK-LEVEL keys to keep. Fields prefixed with `_`
    # are metadata (schema version, ts, etc.) and always retained.
    profile_keep: dict[str, set[str]] = {
        "rls_drift": {
            "rls_model", "batch_learning", "ff_contributions",
            "observation_buffer_heat", "observation_buffer_cool",
            "config", "performance",
        },
        "comfort": {
            "integral", "integral_convergence", "ff_offset", "ff_confidence",
            "hp_setpoint", "desired_temp", "outdoor_temp", "room_temp_rate",
            "performance", "config",
        },
        "phase4_validation": {
            "rls_model", "batch_learning", "greybox_observer", "greybox_bridge",
            "plant_identification", "boundary_estimator", "regime_probe",
            "observation_buffer_heat", "observation_buffer_cool",
        },
    }

    keep_set = profile_keep.get(profile)
    if keep_set is None:
        # Unknown profile: log and fall through to "all"
        _LOGGER.warning("Unknown export profile %r — including all fields", profile)
        return None

    def _keep(field_name: str) -> bool:
        return field_name.startswith("_") or field_name in keep_set

    return _keep


async def _write_ha_history(
    hass: HomeAssistant,
    bundle_dir: Path,
    entity_ids: list[str],
    start: date,
    end: date,
) -> int:
    """Write HA Recorder state history for the requested entities.

    Returns the total number of state records written across all entities.
    """
    from homeassistant.components.recorder.history import get_significant_states

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.max.time(), tzinfo=timezone.utc)

    states_by_entity = await hass.async_add_executor_job(
        get_significant_states,
        hass, start_dt, end_dt, entity_ids,
    )

    history_path = bundle_dir / "ha_history.jsonl"
    record_count = 0

    def _write_history() -> int:
        count = 0
        with history_path.open("w", encoding="utf-8") as f:
            for entity_id, states in states_by_entity.items():
                for state in states:
                    record = {
                        "entity_id": entity_id,
                        "state": state.state if hasattr(state, "state") else state.get("s"),
                        "last_changed": (
                            state.last_changed.isoformat()
                            if hasattr(state, "last_changed") else None
                        ),
                    }
                    f.write(json.dumps(record, default=str))
                    f.write("\n")
                    count += 1
        return count

    record_count = await hass.async_add_executor_job(_write_history)
    return record_count


def _write_manifest_and_readme(
    bundle_dir: Path,
    manifest: dict[str, Any],
    ha_history_entity_ids: list[str] | None,
) -> None:
    """Sync helper: write manifest.json and auto-generated README.md."""
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )

    readme_lines = [
        f"# Tasmota IRHVAC Debug Bundle — {manifest['zone_label']}",
        "",
        f"Generated: {manifest['generated_at']}  ",
        f"Window: {manifest['window_start']} → {manifest['window_end']}  "
        f"({manifest['window_days']} days)  ",
        f"Profile: `{manifest['profile']}`  ",
        f"Integration version: `{manifest['integration_version']}`  ",
        f"Tick schema version: `{manifest['schema_version']}`  ",
        "",
        "## Files",
        "",
        f"- **`tick_log.jsonl`** — {manifest['tick_record_count']} TickOutput "
        "records, one per line. Each line is the JSON serialization of a "
        "`TickOutput`; load via "
        "`TickOutput.from_dict(json.loads(line))`.",
        f"- **`event_log.jsonl`** — {manifest['event_record_count']} sparse "
        "event records (mode change, batch run, anomaly, etc.). Each "
        "record carries `ts_mono`, `ts_wall`, `zone_label`, `kind`, "
        "and a typed `payload`.",
    ]
    if ha_history_entity_ids:
        readme_lines.extend([
            f"- **`ha_history.jsonl`** — {manifest['ha_history_record_count']} "
            "state-change records from HA's recorder for the joined "
            "entities (room temp, outdoor, solar, model inputs).",
        ])
    readme_lines.extend([
        "- **`manifest.json`** — bundle metadata (this README's source).",
        "",
        "## Loading",
        "",
        "```python",
        "import json",
        "from custom_components.tasmota_irhvac.pi.snapshot import TickOutput",
        "",
        "ticks = []",
        'with open("tick_log.jsonl") as f:',
        "    for line in f:",
        "        ticks.append(TickOutput.from_dict(json.loads(line)))",
        "```",
    ])
    (bundle_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
