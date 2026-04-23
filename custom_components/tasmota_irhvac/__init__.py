"""The Tasmota IRHVAC integration."""

from __future__ import annotations

import asyncio  # TODO: unused after async_write_ha_state migration; keep for upstream compat
import logging
from datetime import datetime
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, UnitOfTemperature
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.typing import ConfigType

from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import (
    CONF_PI_ENABLED,
    CONF_OUTDOOR_TEMP_SENSOR,
    DATA_KEY,
    DOMAIN,
    PLATFORMS,
    SIGNAL_PI_BATCH_COMPLETE,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Tasmota IRHVAC integration."""
    hass.data.setdefault(DOMAIN, {})
    hass.data.setdefault(DATA_KEY, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Tasmota IRHVAC from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data.setdefault(DATA_KEY, {})

    # Silver tier: raise ConfigEntryNotReady if MQTT isn't available yet.
    # HA will auto-retry setup with exponential backoff.
    try:
        from homeassistant.components import mqtt
        await mqtt.async_wait_for_mqtt_client(hass)
    except Exception as err:
        from homeassistant.exceptions import ConfigEntryNotReady
        raise ConfigEntryNotReady("MQTT not available") from err

    _register_services(hass)

    # Forward climate first so entity is in hass.data before sensor/button setup
    await hass.config_entries.async_forward_entry_setups(entry, ["climate"])
    await hass.config_entries.async_forward_entry_setups(entry, ["sensor", "button", "binary_sensor"])

    # Defer config issue checks to give other integrations time to load entities
    @callback
    def _deferred_check(_now: datetime) -> None:
        _check_config_issues(hass, entry)
        _check_tuning_health_issues(hass, entry)

    async_call_later(hass, 120, _deferred_check)

    # Listen for batch completion to re-check tuning health
    @callback
    def _on_batch_complete() -> None:
        _check_tuning_health_issues(hass, entry, from_batch=True)

    unsub = async_dispatcher_connect(
        hass, SIGNAL_PI_BATCH_COMPLETE.format(entry.entry_id), _on_batch_complete,
    )
    entry.async_on_unload(unsub)

    return True


MINOR_VERSION = 3

async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to current version."""
    _LOGGER.debug("Migrating config entry from version %s.%s", entry.version, entry.minor_version)

    if entry.version != 1:
        return True

    if entry.minor_version < 2:
        from homeassistant.util.unit_conversion import TemperatureConverter

        new_data = {**entry.data}
        new_options = {**entry.options}

        # Convert stored config temps from IR protocol unit to system unit.
        celsius_mode = new_options.get("celsius_mode", new_data.get("celsius_mode", "on"))
        ir_unit = (
            UnitOfTemperature.CELSIUS if celsius_mode.lower() in ("on", "celsius")
            else UnitOfTemperature.FAHRENHEIT
        )
        system_unit = hass.config.units.temperature_unit
        if ir_unit != system_unit:
            temp_keys = ("min_temp", "max_temp", "target_temp", "away_temp")
            for store in (new_data, new_options):
                for key in temp_keys:
                    if key in store and store[key] is not None:
                        store[key] = round(TemperatureConverter.convert(
                            float(store[key]), ir_unit, system_unit
                        ), 1)
            _LOGGER.info("Migrated config temps from %s to %s", ir_unit, system_unit)

        # Normalize toggle values to lowercase
        toggle_keys = (
            "celsius_mode", "beep", "turbo", "quiet", "econo",
            "light", "filter", "clean", "sleep", "swingv", "swingh",
        )
        for store in (new_data, new_options):
            for key in toggle_keys:
                if key in store and isinstance(store[key], str):
                    store[key] = store[key].lower()

        # Coerce precision/temp_step to float
        for store in (new_data, new_options):
            for key in ("precision", "temp_step"):
                if key in store and isinstance(store[key], str):
                    store[key] = float(store[key])

        # Rename celsius_mode → ir_protocol_unit ("on"→"celsius", "off"→"fahrenheit")
        for store in (new_data, new_options):
            old_val = store.pop("celsius_mode", None)
            if old_val is not None:
                store["ir_protocol_unit"] = (
                    "celsius" if old_val.lower() in ("on", "celsius") else "fahrenheit"
                )

        hass.config_entries.async_update_entry(
            entry, data=new_data, options=new_options,
            minor_version=2, version=1,
        )

    if entry.minor_version < 3:
        new_options = {**entry.options}

        # Rename outdoor slope keys (values stay positive — same convention)
        for old_key, new_key in [
            ("pi_ff_heat_slope", "pi_outdoor_seed_heat"),
            ("pi_ff_cool_slope", "pi_outdoor_seed_cool"),
        ]:
            if old_key in new_options:
                new_options[new_key] = new_options.pop(old_key)

        # Remove reference temperatures (no longer used)
        new_options.pop("pi_ff_heat_reference", None)
        new_options.pop("pi_ff_cool_reference", None)

        # Merge outdoor clamps: keep heat values as shared, drop cool-specific
        for old_key, new_key in [
            ("pi_outdoor_delta_clamp_heat_min", "pi_outdoor_seed_clamp_min"),
            ("pi_outdoor_delta_clamp_heat_max", "pi_outdoor_seed_clamp_max"),
        ]:
            if old_key in new_options:
                new_options[new_key] = new_options.pop(old_key)
        new_options.pop("pi_outdoor_delta_clamp_cool_min", None)
        new_options.pop("pi_outdoor_delta_clamp_cool_max", None)

        # Negate model input seeds: old convention was "effect on setpoint"
        # (negative = warms room), new convention is "thermal effect on room"
        # (positive = warms room). Key names stay seed_heat/seed_cool.
        model_inputs = new_options.get("pi_model_inputs", [])
        if model_inputs:
            migrated_inputs = []
            for m_input in model_inputs:
                m = dict(m_input)
                if "seed_heat" in m:
                    m["seed_heat"] = -m["seed_heat"]
                if "seed_cool" in m:
                    m["seed_cool"] = -m["seed_cool"]
                # Negate clamps from old internal β space to seed space
                if "clamp_min" in m and "clamp_max" in m:
                    old_min = m["clamp_min"]
                    old_max = m["clamp_max"]
                    m["clamp_min"] = -old_max
                    m["clamp_max"] = -old_min
                migrated_inputs.append(m)
            new_options["pi_model_inputs"] = migrated_inputs

        hass.config_entries.async_update_entry(
            entry, options=new_options,
            minor_version=MINOR_VERSION, version=1,
        )

        # Migrate subentry data (model inputs and supplemental sources)
        if hasattr(entry, "subentries"):
            for subentry in entry.subentries.values():
                sub_data = dict(subentry.data)
                changed = False
                if "seed_heat" in sub_data:
                    sub_data["seed_heat"] = -sub_data["seed_heat"]
                    changed = True
                if "seed_cool" in sub_data:
                    sub_data["seed_cool"] = -sub_data["seed_cool"]
                    changed = True
                if "clamp_min" in sub_data and "clamp_max" in sub_data:
                    old_min = sub_data["clamp_min"]
                    old_max = sub_data["clamp_max"]
                    sub_data["clamp_min"] = -old_max
                    sub_data["clamp_max"] = -old_min
                    changed = True
                if changed:
                    hass.config_entries.async_update_subentry(
                        entry, subentry, data=sub_data,
                    )

        _LOGGER.info("Migrated to positive-warms-room sign convention (v1.3)")

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
        hass.data.get(DATA_KEY, {}).pop(entry.entry_id, None)
    return unload_ok


def _check_config_issues(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Check for config issues and surface via HA Repairs panel."""
    config = {**entry.data, **entry.options}

    # Only check PI-related entities if PI is enabled
    pi_enabled = config.get(CONF_PI_ENABLED, False)

    # Check outdoor temp sensor
    outdoor_entity = config.get(CONF_OUTDOOR_TEMP_SENSOR)
    outdoor_issue_id = f"outdoor_sensor_not_found_{entry.entry_id}"
    if outdoor_entity and pi_enabled:
        if hass.states.get(outdoor_entity) is None:
            ir.async_create_issue(
                hass, DOMAIN, outdoor_issue_id, is_fixable=False,
                severity=ir.IssueSeverity.WARNING,
                translation_key="outdoor_sensor_not_found",
                translation_placeholders={"entity_id": outdoor_entity},
            )
        else:
            ir.async_delete_issue(hass, DOMAIN, outdoor_issue_id)
    else:
        ir.async_delete_issue(hass, DOMAIN, outdoor_issue_id)

    # Check model input entities (from options or subentries)
    from .const import CONF_PI_MODEL_INPUTS, SUBENTRY_MODEL_INPUT
    model_inputs = list(config.get(CONF_PI_MODEL_INPUTS, []))
    if hasattr(entry, "subentries"):
        model_inputs.extend(
            dict(sub.data) for sub in entry.subentries.values()
            if sub.subentry_type == SUBENTRY_MODEL_INPUT
        )
    if pi_enabled:
        for m_input in model_inputs:
            entity_id = m_input.get("entity_id", "")
            name = m_input.get("name", entity_id)
            issue_id = f"model_input_entity_not_found_{entry.entry_id}_{entity_id}"
            if entity_id and hass.states.get(entity_id) is None:
                ir.async_create_issue(
                    hass, DOMAIN, issue_id, is_fixable=False,
                    severity=ir.IssueSeverity.WARNING,
                    translation_key="model_input_entity_not_found",
                    translation_placeholders={
                        "entity_id": entity_id,
                        "name": name,
                    },
                )
            else:
                ir.async_delete_issue(hass, DOMAIN, issue_id)

    # Clean up legacy issue IDs
    for legacy_key in ("suppress_learning_entity_not_found", "bias_entity_not_found",
                       "disturbance_entity_not_found"):
        ir.async_delete_issue(hass, DOMAIN, f"{legacy_key}_{entry.entry_id}")


def _check_tuning_health_issues(hass: HomeAssistant, entry: ConfigEntry, *, from_batch: bool = False) -> None:
    """Check PI tuning health and surface/clear issues via HA Repairs."""
    climate_entity = hass.data.get(DATA_KEY, {}).get(entry.entry_id)
    if climate_entity is None:
        return

    from .pi import PIController
    pi = climate_entity._pi
    if not isinstance(pi, PIController):
        return

    issues = pi._check_tuning_health(from_batch=from_batch)
    for issue_id, severity, translation_key, placeholders, should_create, is_fixable, data in issues:
        if should_create:
            kwargs: dict[str, Any] = dict(
                is_fixable=is_fixable,
                severity=(ir.IssueSeverity.WARNING if severity == "warning"
                          else ir.IssueSeverity.CRITICAL if severity == "critical"
                          else ir.IssueSeverity.WARNING),
                translation_key=translation_key,
                translation_placeholders=placeholders,
            )
            if data is not None:
                kwargs["data"] = data
            ir.async_create_issue(hass, DOMAIN, issue_id, **kwargs)
        else:
            ir.async_delete_issue(hass, DOMAIN, issue_id)


def _register_services(hass: HomeAssistant) -> None:
    """Register IRHVAC services (idempotent)."""
    from .climate import SERVICE_TO_METHOD, IRHVAC_SERVICE_SCHEMA

    if hass.services.has_service(DOMAIN, "set_econo"):
        return

    async def async_service_handler(service: ServiceCall) -> None:
        """Map services to methods on TasmotaIrhvac."""
        method_info: dict[str, Any] = SERVICE_TO_METHOD.get(service.service, {})
        params: dict[str, Any] = {
            key: value for key, value in service.data.items() if key != ATTR_ENTITY_ID
        }
        entity_ids = service.data.get(ATTR_ENTITY_ID)
        if entity_ids:
            devices = [
                device
                for device in hass.data[DATA_KEY].values()
                if device.entity_id in entity_ids
            ]
        else:  # pragma: no cover — schema requires entity_id, defensive only
            devices = list(hass.data[DATA_KEY].values())

        method_name: str = method_info["method"]
        for device in devices:
            if not hasattr(device, method_name):  # pragma: no cover — defensive for vendor subclasses
                continue
            await getattr(device, method_name)(**params)
            device.async_write_ha_state()

    for irhvac_service in SERVICE_TO_METHOD:
        svc_schema = SERVICE_TO_METHOD[irhvac_service].get("schema", IRHVAC_SERVICE_SCHEMA)
        hass.services.async_register(
            DOMAIN, irhvac_service, async_service_handler, schema=svc_schema  # type: ignore[arg-type]
        )
