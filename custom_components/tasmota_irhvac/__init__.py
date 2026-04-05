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


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old config entries."""
    if entry.version == 1 and entry.minor_version < 5:
        # v1.5: precision/temp_step stored as floats instead of strings
        new_options = dict(entry.options)
        changed = False
        for key in ("precision", "temp_step"):
            if key in new_options and isinstance(new_options[key], str):
                new_options[key] = float(new_options[key])
                changed = True
        if changed:
            hass.config_entries.async_update_entry(entry, options=new_options)
        entry.minor_version = 5
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

        if entry.minor_version < 4:
            # v1.4: Migrate disturbance_inputs → model_inputs
            for store in (new_data, new_options):
                disturbance_inputs = store.pop("pi_disturbance_inputs", [])
                model_inputs = store.get("pi_model_inputs", [])
                for d_input in disturbance_inputs:
                    model_input = {
                        "name": d_input.get("name", "Migrated Input"),
                        "entity_id": d_input.get("entity_id", ""),
                        "seed_heat": float(d_input.get("default_bias", 0.0)),
                        "seed_cool": float(d_input.get("default_bias", 0.0)),
                        "lag_tau": 0,
                    }
                    gain = d_input.get("gain", 1.0)
                    if gain != 1.0:
                        # Old gain was a multiplier on entity value; approximate as seed
                        model_input["seed_heat"] = float(gain)
                        model_input["seed_cool"] = float(gain)
                    model_inputs.append(model_input)
                store["pi_model_inputs"] = model_inputs

        hass.config_entries.async_update_entry(
            entry, data=new_data, options=new_options, minor_version=4, version=1,
        )
        _LOGGER.info("Migrated config entry to version 1.4")

    # v1.6: Migrate model_inputs from options list → subentries
    if entry.version == 1 and entry.minor_version < 6:
        from .const import SUBENTRY_MODEL_INPUT, CONF_PI_MODEL_INPUTS
        from homeassistant.config_entries import ConfigSubentry
        model_inputs = list(entry.options.get(CONF_PI_MODEL_INPUTS, []))
        if model_inputs:
            existing_ids = {
                sub.unique_id for sub in entry.subentries.values()
                if sub.subentry_type == SUBENTRY_MODEL_INPUT and sub.unique_id
            }
            for m_input in model_inputs:
                entity_id = m_input.get("entity_id", "")
                if entity_id in existing_ids:
                    continue
                subentry = ConfigSubentry(
                    data=m_input,
                    subentry_type=SUBENTRY_MODEL_INPUT,
                    title=m_input.get("name", "Model Input"),
                    unique_id=entity_id or None,
                )
                hass.config_entries.async_add_subentry(entry, subentry)
            new_options = {k: v for k, v in entry.options.items() if k != CONF_PI_MODEL_INPUTS}
            hass.config_entries.async_update_entry(
                entry, options=new_options, minor_version=6, version=1,
            )
            _LOGGER.info("Migrated %d model inputs to subentries", len(model_inputs))
        else:
            hass.config_entries.async_update_entry(
                entry, minor_version=6, version=1,
            )

    # v1.7: Convert stored config temps from celsius_mode unit to system unit.
    # Previously, temps were stored in celsius_mode unit and converted at runtime
    # on every boot. This caused double-conversion when user edited values via UI
    # (saved in system unit, then reconverted on next boot as if still celsius_mode).
    if entry.version == 1 and entry.minor_version < 7:
        from homeassistant.util.unit_conversion import TemperatureConverter
        celsius_mode = entry.options.get("celsius_mode", entry.data.get("celsius_mode", "on"))
        celsius_unit = (
            UnitOfTemperature.CELSIUS if celsius_mode.lower() == "on"
            else UnitOfTemperature.FAHRENHEIT
        )
        system_unit = hass.config.units.temperature_unit
        if celsius_unit != system_unit:
            temp_keys = ("min_temp", "max_temp", "target_temp", "away_temp")
            new_options = {**entry.options}
            new_data = {**entry.data}
            for store in (new_data, new_options):
                for key in temp_keys:
                    if key in store and store[key] is not None:
                        old_val = store[key]
                        store[key] = round(TemperatureConverter.convert(
                            float(old_val), celsius_unit, system_unit
                        ), 1)
            hass.config_entries.async_update_entry(
                entry, data=new_data, options=new_options, minor_version=7, version=1,
            )
            _LOGGER.info("Migrated config temps from %s to %s", celsius_unit, system_unit)
        else:
            hass.config_entries.async_update_entry(
                entry, minor_version=7, version=1,
            )

    # v1.8: Normalize on/off toggle values to lowercase.
    # Tasmota sends "On"/"Off" (capitalized) but our selectors expect "on"/"off".
    if entry.version == 1 and entry.minor_version < 8:
        toggle_keys = (
            "celsius_mode", "beep", "turbo", "quiet", "econo",
            "light", "filter", "clean", "sleep", "swingv", "swingh",
        )
        new_options = {**entry.options}
        new_data = {**entry.data}
        for store in (new_data, new_options):
            for key in toggle_keys:
                if key in store and isinstance(store[key], str):
                    store[key] = store[key].lower()
        hass.config_entries.async_update_entry(
            entry, data=new_data, options=new_options, minor_version=8, version=1,
        )

    # v1.9: Convert remaining temp values from °C to system unit.
    # v1.7 converted min/max/target/away. This covers deadband, FF references,
    # and recovery_margin (supplemental sources).
    if entry.version == 1 and entry.minor_version < 9:
        from homeassistant.util.unit_conversion import TemperatureConverter
        celsius_mode = entry.options.get("celsius_mode", entry.data.get("celsius_mode", "on"))
        celsius_unit = (
            UnitOfTemperature.CELSIUS if celsius_mode.lower() == "on"
            else UnitOfTemperature.FAHRENHEIT
        )
        system_unit = hass.config.units.temperature_unit
        if celsius_unit != system_unit:
            new_options = {**entry.options}
            new_data = {**entry.data}
            # Absolute temperatures: convert using TemperatureConverter
            abs_keys = ("pi_ff_heat_reference", "pi_ff_cool_reference")
            # Temperature deltas: multiply by 1.8 for °C→°F (or /1.8 for °F→°C)
            delta_keys = ("pi_deadband",)
            for store in (new_data, new_options):
                for key in abs_keys:
                    if key in store and store[key] is not None:
                        store[key] = round(TemperatureConverter.convert(
                            float(store[key]), celsius_unit, system_unit
                        ), 1)
                for key in delta_keys:
                    if key in store and store[key] is not None:
                        old_val = float(store[key])
                        if system_unit == UnitOfTemperature.FAHRENHEIT:
                            store[key] = round(old_val * 1.8, 2)
                        else:
                            store[key] = round(old_val / 1.8, 2)
            hass.config_entries.async_update_entry(
                entry, data=new_data, options=new_options, minor_version=9, version=1,
            )
        else:
            hass.config_entries.async_update_entry(
                entry, minor_version=9, version=1,
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
