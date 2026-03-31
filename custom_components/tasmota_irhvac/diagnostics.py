"""Diagnostics support for Tasmota IRHVAC."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DATA_KEY
from .pi_controller import PIController

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
    if pi and pi._pi_enabled:
        # RLS model state
        coeff_names = ["intercept", "outdoor_delta"]
        for m_input in pi._model_inputs:
            coeff_names.append(m_input.get("name", "unknown"))

        rls_heat_coeffs = {coeff_names[i]: round(pi._rls_heat.beta[i], 4)
                          for i in range(min(len(coeff_names), len(pi._rls_heat.beta)))}
        rls_cool_coeffs = {coeff_names[i]: round(pi._rls_cool.beta[i], 4)
                          for i in range(min(len(coeff_names), len(pi._rls_cool.beta)))}
        rls_heat_uncertainty = {coeff_names[i]: round(pi._rls_heat.get_covariance_diagonal()[i], 4)
                               for i in range(min(len(coeff_names), len(pi._rls_heat.beta)))}

        data["pi_controller"] = {
            "enabled": True,
            "paused": pi._pi_paused,
            "desired_temp": pi._desired_temp,
            "hp_setpoint": pi._hp_setpoint,
            "integral": round(pi._pi_integral, 3),
            "integral_convergence": round(pi._integral_convergence, 2),
            "ff_offset": round(pi._ff_offset, 2),
            "ff_offset_buckets": round(pi._ff_offset_buckets, 2),
            "outdoor_temp": pi._outdoor_temp,
            "sensor_unavailable": pi._sensor_unavailable,
            "sensor_recovery_pending": pi._sensor_recovery_pending,
            "config": {
                "kp": pi._pi_kp,
                "ki": pi._pi_ki,
                "deadband": pi._pi_deadband,
                "setpoint_weight": pi._pi_setpoint_weight,
                "min_interval": pi._pi_min_interval,
                "outdoor_temp_sensor": pi._outdoor_temp_sensor,
                "model_inputs": pi._model_inputs,
            },
            "rls_model": {
                "heat_coefficients": rls_heat_coeffs,
                "cool_coefficients": rls_cool_coeffs,
                "heat_uncertainty": rls_heat_uncertainty,
                "heat_observation_count": pi._rls_heat.observation_count,
                "cool_observation_count": pi._rls_cool.observation_count,
                "learning_suppressed": pi._manual_ff_suppress,
                "manual_suppress_reason": pi._manual_ff_suppress_reason,
            },
            "legacy_buckets": {
                "ff_heat_buckets": {
                    str(k): round(v, 2) for k, v in pi._ff_heat_buckets.items()
                },
                "ff_cool_buckets": {
                    str(k): round(v, 2) for k, v in pi._ff_cool_buckets.items()
                },
            },
        }

    return data


def _redact(data: dict) -> dict:
    """Redact sensitive keys from a dict."""
    return {
        k: "**REDACTED**" if k in REDACT_KEYS else v
        for k, v in data.items()
    }
