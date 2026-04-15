"""Diagnostics support for Tasmota IRHVAC."""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .batch_learning import BatchResult
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

        heat_phys = pi._rls_heat.get_coefficients()
        cool_phys = pi._rls_cool.get_coefficients()
        rls_heat_coeffs = {coeff_names[i]: round(heat_phys[i], 4)
                          for i in range(min(len(coeff_names), len(heat_phys)))}
        rls_cool_coeffs = {coeff_names[i]: round(cool_phys[i], 4)
                          for i in range(min(len(coeff_names), len(cool_phys)))}
        rls_heat_uncertainty = {coeff_names[i]: round(pi._rls_heat.get_covariance_diagonal()[i], 4)
                               for i in range(min(len(coeff_names), len(pi._rls_heat.beta)))}

        data["pi_controller"] = {
            "enabled": True,
            "paused": pi._pi_paused,
            "desired_temp": pi._desired_temp,
            "hp_setpoint": pi._hp_setpoint,
            "integral": round(pi._pi_integral, 3),
            "integral_convergence": round(pi._metrics.integral_convergence, 2),
            "ff_offset": round(pi._ff_offset, 2),
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
        }

        # Batch learning state
        if pi._last_batch_result is not None:
            br = pi._last_batch_result
            batch_names = coeff_names[:len(br.beta_batch)]
            data["pi_controller"]["batch_learning"] = {
                "last_run_mono": pi._last_batch_timestamp,
                "n_total": br.n_total,
                "n_eligible": br.n_eligible,
                "residual_rms": round(br.residual_rms, 4),
                "recommend_update": br.recommend_update,
                "max_coeff_change_pct": round(br.max_coeff_change_pct, 1),
                "coefficients": {
                    batch_names[i]: {
                        "current": round(br.beta_current[i], 4),
                        "batch": round(br.beta_batch[i], 4),
                    }
                    for i in range(len(batch_names))
                    if i < len(br.beta_current)
                },
                "held_features": [
                    batch_names[i] for i in br.held_features
                    if i < len(batch_names)
                ],
                "n_outliers_excluded": br.n_outliers_excluded,
            }
            # Drift detection state
            drifting = pi.get_drifting_coefficients()
            data["pi_controller"]["batch_learning"]["drift_detection"] = {
                "drifting_coefficients": [
                    {"index": idx, "name": name, "consecutive_cycles": count}
                    for idx, name, count in drifting
                ],
                "correction_history": {
                    (coeff_names[i] if i < len(coeff_names) else f"β{i}"): signs
                    for i, signs in enumerate(pi._drift_correction_signs)
                },
            }
        else:
            data["pi_controller"]["batch_learning"] = None

        # Observation buffer stats
        obs = pi._observation_buffer.get_all()
        n_eligible = sum(
            1 for o in obs
            if not o.clamped and abs(o.room_rate) < 0.02
        )
        buf_stats: dict = {
            "total": len(obs),
            "eligible": n_eligible,
        }
        # Diversity buffer stats (if using DiversityAwareBuffer)
        if hasattr(pi._observation_buffer, 'get_leverage_scores'):
            scores = pi._observation_buffer.get_leverage_scores()
            if scores:
                buf_stats["leverage_min"] = round(min(scores), 6)
                buf_stats["leverage_median"] = round(sorted(scores)[len(scores) // 2], 6)
                buf_stats["leverage_max"] = round(max(scores), 6)
            if obs:
                import time as time_mod
                now = time_mod.monotonic()
                oldest = min(o.timestamp for o in obs)
                buf_stats["oldest_age_hours"] = round((now - oldest) / 3600, 1)
            buf_stats["max_size"] = pi._observation_buffer._max_size
            # Feature composition: count observations with each feature active
            n_features = pi._observation_buffer.n_features
            coeff_names = ["intercept", "outdoor_delta"]
            for m_input in pi._model_inputs:
                coeff_names.append(m_input.get("name", "input"))
            feature_active: dict[str, int] = {}
            for j in range(2, n_features):  # skip intercept & outdoor_delta (always present)
                name = coeff_names[j] if j < len(coeff_names) else f"feature_{j}"
                feature_active[name] = sum(
                    1 for o in obs
                    if j < len(o.features) and abs(o.features[j]) > 1e-6
                )
            if feature_active:
                buf_stats["feature_active_counts"] = feature_active
        data["pi_controller"]["observation_buffer"] = buf_stats

        # PI performance metrics
        data["pi_controller"]["performance"] = {
            "itae_accumulator": round(pi._metrics.itae_accumulator, 2),
            "comfort_violation_hours": round(pi._metrics.comfort_violation_hours, 2),
            "setpoint_changes": pi._metrics.setpoint_changes,
            "controllable_itae": round(pi._metrics.controllable_itae, 2),
            "uncontrollable_itae": round(pi._metrics.uncontrollable_itae, 2),
            "controllable_cvh": round(pi._metrics.controllable_cvh, 2),
            "uncontrollable_cvh": round(pi._metrics.uncontrollable_cvh, 2),
            "ff_load_fraction": round(pi._metrics.ff_load_fraction, 4),
            "batch_model_rms": round(pi._metrics.batch_model_rms, 3) if pi._metrics.batch_model_rms is not None else None,
        }
        data["pi_controller"]["ff_confidence"] = round(pi._ff_confidence, 4)
        data["pi_controller"]["room_temp_rate"] = round(pi._room_temp_rate, 4)
        data["pi_controller"]["tau_estimate"] = (
            round(pi._tau_estimate, 1) if pi._imc_enabled else None
        )

    return data


def _redact(data: dict) -> dict:
    """Redact sensitive keys from a dict."""
    return {
        k: "**REDACTED**" if k in REDACT_KEYS else v
        for k, v in data.items()
    }
