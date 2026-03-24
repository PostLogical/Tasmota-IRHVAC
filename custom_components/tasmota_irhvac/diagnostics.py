"""Diagnostics support for Tasmota IRHVAC."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DATA_KEY
from .pi_controller import PIControllerMixin

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

    if isinstance(climate_entity, PIControllerMixin) and climate_entity._pi_enabled:
        data["pi_controller"] = {
            "enabled": True,
            "paused": climate_entity._pi_paused,
            "desired_temp": climate_entity._desired_temp,
            "hp_setpoint": climate_entity._hp_setpoint,
            "integral": round(climate_entity._pi_integral, 3),
            "ff_offset": round(climate_entity._ff_offset, 2),
            "outdoor_temp": climate_entity._outdoor_temp,
            "sensor_unavailable": climate_entity._sensor_unavailable,
            "sensor_recovery_pending": climate_entity._sensor_recovery_pending,
            "config": {
                "kp": climate_entity._pi_kp,
                "ki": climate_entity._pi_ki,
                "deadband": climate_entity._pi_deadband,
                "setpoint_weight": climate_entity._pi_setpoint_weight,
                "min_interval": climate_entity._pi_min_interval,
                "outdoor_temp_sensor": climate_entity._outdoor_temp_sensor,
                "suppress_learning_entity": climate_entity._ff_suppress_learning_entity,
                "bias_entity": climate_entity._ff_bias_entity,
            },
            "ff_heat_buckets": {
                str(k): round(v, 2) for k, v in climate_entity._ff_heat_buckets.items()
            },
            "ff_cool_buckets": {
                str(k): round(v, 2) for k, v in climate_entity._ff_cool_buckets.items()
            },
        }

    return data


def _redact(data: dict) -> dict:
    """Redact sensitive keys from a dict."""
    return {
        k: "**REDACTED**" if k in REDACT_KEYS else v
        for k, v in data.items()
    }
