"""The Tasmota IRHVAC integration."""

import asyncio
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.typing import ConfigType

from .const import (
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
    def _deferred_check(_now):
        _check_config_issues(hass, entry)

    async_call_later(hass, 120, _deferred_check)

    return True


MINOR_VERSION = 3

async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to current version.

    Clean migration chain for upstream (post config-flow).
    Beta migrations (v1.2–v1.6, v1.9–v1.11) stripped.
    """
    _LOGGER.debug("Migrating config entry from version %s.%s", entry.version, entry.minor_version)

    if entry.version != 1:
        return True

    new_data = {**entry.data}
    new_options = {**entry.options}

    # v1.2: Convert stored config temps from celsius_mode unit to system unit.
    # Config flow stores temps in celsius_mode unit; entity expects system unit.
    # Without this, double-conversion occurs when user edits via UI.
    if entry.minor_version < 2:
        from homeassistant.util.unit_conversion import TemperatureConverter
        celsius_mode = new_options.get("celsius_mode", new_data.get("celsius_mode", "on"))
        celsius_unit = (
            UnitOfTemperature.CELSIUS if celsius_mode.lower() == "on"
            else UnitOfTemperature.FAHRENHEIT
        )
        system_unit = hass.config.units.temperature_unit
        if celsius_unit != system_unit:
            temp_keys = ("min_temp", "max_temp", "target_temp", "away_temp")
            for store in (new_data, new_options):
                for key in temp_keys:
                    if key in store and store[key] is not None:
                        old_val = store[key]
                        store[key] = round(TemperatureConverter.convert(
                            float(old_val), celsius_unit, system_unit
                        ), 1)
            _LOGGER.info("Migrated config temps from %s to %s", celsius_unit, system_unit)

    # v1.3: Normalize on/off toggle values to lowercase.
    # Tasmota sends "On"/"Off" (capitalized) but selectors expect "on"/"off".
    if entry.minor_version < 3:
        toggle_keys = (
            "celsius_mode", "beep", "turbo", "quiet", "econo",
            "light", "filter", "clean", "sleep", "swingv", "swingh",
        )
        for store in (new_data, new_options):
            for key in toggle_keys:
                if key in store and isinstance(store[key], str):
                    store[key] = store[key].lower()

    # Coerce precision/temp_step to float (config flow may store as string)
    for store in (new_data, new_options):
        for key in ("precision", "temp_step"):
            if key in store and isinstance(store[key], str):
                store[key] = float(store[key])

    hass.config_entries.async_update_entry(
        entry, data=new_data, options=new_options,
        minor_version=MINOR_VERSION, version=1,
    )
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
        else:  # pragma: no cover — schema requires entity_id, defensive only
            devices = hass.data[DATA_KEY].values()

        update_tasks = []
        for device in devices:
            if not hasattr(device, method["method"]):  # pragma: no cover — defensive for vendor subclasses
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
