"""The Tasmota IRHVAC integration."""

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_PI_DISTURBANCE_INPUTS,
    CONF_PI_ENABLED,
    CONF_OUTDOOR_TEMP_SENSOR,
    DATA_KEY,
    DOMAIN,
    PLATFORMS,
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

    _register_services(hass)

    # Forward climate first so entity is in hass.data before sensor/button setup
    await hass.config_entries.async_forward_entry_setups(entry, ["climate"])
    await hass.config_entries.async_forward_entry_setups(entry, ["sensor", "button", "binary_sensor"])

    # Defer config issue checks to give other integrations time to load entities
    @callback
    def _deferred_check(_now):
        _check_config_issues(hass, entry)

    async_call_later(hass, 120, _deferred_check)

    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to current version."""
    _LOGGER.debug("Migrating config entry from version %s.%s", entry.version, entry.minor_version)

    if entry.version == 1:
        new_data = {**entry.data}
        new_options = {**entry.options}

        if entry.minor_version < 2:
            # v1.2: Added pi_ff_bias_entity and pi_setpoint_weight
            new_data.setdefault("pi_ff_bias_entity", "")
            new_data.setdefault("pi_setpoint_weight", 1.0)
            new_options.setdefault("pi_ff_bias_entity", "")
            new_options.setdefault("pi_setpoint_weight", 1.0)

        if entry.minor_version < 3:
            # v1.3: Migrate suppress/bias entities → disturbance_inputs list
            for store in (new_data, new_options):
                disturbance_inputs = store.get("pi_disturbance_inputs", [])
                old_suppress = store.pop("pi_ff_suppress_learning_entity", "")
                old_bias = store.pop("pi_ff_bias_entity", "")
                if old_suppress:
                    disturbance_inputs.append({
                        "name": "Suppress Entity (migrated)",
                        "entity_id": old_suppress,
                        "suppress_learning": True,
                        "default_bias": 0.0,
                        "gain": 1.0,
                    })
                if old_bias:
                    disturbance_inputs.append({
                        "name": "Bias Entity (migrated)",
                        "entity_id": old_bias,
                        "suppress_learning": False,
                        "default_bias": 0.0,
                        "gain": 1.0,
                    })
                store["pi_disturbance_inputs"] = disturbance_inputs

        hass.config_entries.async_update_entry(
            entry, data=new_data, options=new_options, minor_version=3, version=1,
        )
        _LOGGER.info("Migrated config entry to version %s.%s", entry.version, entry.minor_version)

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

    # Check disturbance input entities
    disturbance_inputs = config.get(CONF_PI_DISTURBANCE_INPUTS, [])
    if pi_enabled:
        for d_input in disturbance_inputs:
            entity_id = d_input.get("entity_id", "")
            name = d_input.get("name", entity_id)
            issue_id = f"disturbance_entity_not_found_{entry.entry_id}_{entity_id}"
            if entity_id and hass.states.get(entity_id) is None:
                ir.async_create_issue(
                    hass, DOMAIN, issue_id, is_fixable=False,
                    severity=ir.IssueSeverity.WARNING,
                    translation_key="disturbance_entity_not_found",
                    translation_placeholders={
                        "entity_id": entity_id,
                        "name": name,
                    },
                )
            else:
                ir.async_delete_issue(hass, DOMAIN, issue_id)

    # Clean up legacy issue IDs from v1.2
    for legacy_key in ("suppress_learning_entity_not_found", "bias_entity_not_found"):
        ir.async_delete_issue(hass, DOMAIN, f"{legacy_key}_{entry.entry_id}")


def _register_services(hass: HomeAssistant) -> None:
    """Register IRHVAC services (idempotent)."""
    from .climate import SERVICE_TO_METHOD, IRHVAC_SERVICE_SCHEMA

    if hass.services.has_service(DOMAIN, "set_econo"):
        return

    async def async_service_handler(service):
        """Map services to methods on TasmotaIrhvac."""
        method = SERVICE_TO_METHOD.get(service.service, {})
        params = {
            key: value for key, value in service.data.items() if key != ATTR_ENTITY_ID
        }
        entity_ids = service.data.get(ATTR_ENTITY_ID)
        if entity_ids:
            devices = [
                device
                for device in hass.data[DATA_KEY].values()
                if device.entity_id in entity_ids
            ]
        else:
            devices = hass.data[DATA_KEY].values()

        update_tasks = []
        for device in devices:
            if not hasattr(device, method["method"]):
                continue
            await getattr(device, method["method"])(**params)
            update_tasks.append(
                asyncio.create_task(device.async_update_ha_state(True))
            )

        if update_tasks:
            await asyncio.wait(update_tasks)

    for irhvac_service, method_info in SERVICE_TO_METHOD.items():
        schema = method_info.get("schema", IRHVAC_SERVICE_SCHEMA)
        hass.services.async_register(
            DOMAIN, irhvac_service, async_service_handler, schema=schema
        )
