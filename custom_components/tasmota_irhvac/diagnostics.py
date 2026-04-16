"""Diagnostics support for Tasmota IRHVAC."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DATA_KEY
from .pi import PIController

REDACT_KEYS = {"unique_id", "topic", "state_topic", "availability_topic"}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    data: dict[str, Any] = {
        "config_entry": {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "data": _redact(dict(entry.data)),
            "options": _redact(dict(entry.options)),
        },
    }

    climate_entity = hass.data.get(DATA_KEY, {}).get(entry.entry_id)
    if climate_entity is None:
        data["entity"] = None
        return data

    data["entity"] = {
        "entity_id": climate_entity.entity_id,
        "unique_id": "**REDACTED**",
        "vendor": getattr(climate_entity, "_vendor", None),
        "hvac_mode": str(climate_entity._attr_hvac_mode),
        "target_temperature": climate_entity._attr_target_temperature,
        "current_temperature": climate_entity._attr_current_temperature,
        "fan_mode": climate_entity._attr_fan_mode,
        "swing_mode": climate_entity._attr_swing_mode,
        "preset_mode": getattr(climate_entity, "_attr_preset_mode", None),
        "available": climate_entity.available,
    }

    pi = climate_entity._pi
    if isinstance(pi, PIController) and pi.is_active:
        data["pi_controller"] = pi.get_full_diagnostics()

    return data


def _redact(data: dict) -> dict:
    """Redact sensitive keys from a dict."""
    return {
        k: "**REDACTED**" if k in REDACT_KEYS else v
        for k, v in data.items()
    }
