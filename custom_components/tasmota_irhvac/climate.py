"""Adds support for generic thermostat units."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import Event, EventStateChangedData, State
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from homeassistant.helpers.event import CALLBACK_TYPE

    from .config_model import IrhvacConfig
    from .vendors.base import TimerRequest, VendorHandler

import homeassistant.helpers.config_validation as cv
import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components import mqtt
from homeassistant.config_entries import ConfigEntry

from homeassistant.components.mqtt.schemas import MQTT_ENTITY_COMMON_SCHEMA

from homeassistant.components.climate import PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA

# try:
#     from homeassistant.components.climate import ClimateEntity
# except ImportError:
#     from homeassistant.components.binary_sensor import ClimateDevice as ClimateEntity
from homeassistant.components.climate import ClimateEntity
from homeassistant.components.climate.const import (
    ATTR_FAN_MODE,
    ATTR_HVAC_MODE,
    ATTR_PRESET_MODE,
    ATTR_SWING_MODE,
    FAN_AUTO,
    FAN_DIFFUSE,
    FAN_FOCUS,
    FAN_HIGH,
    FAN_LOW,
    FAN_MEDIUM,
    FAN_MIDDLE,
    FAN_OFF,
    FAN_ON,
    PRESET_AWAY,
    PRESET_NONE,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, cached_property, callback
from homeassistant.helpers import event as ha_event
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.restore_state import ExtraStoredData, RestoreEntity
from homeassistant.util.unit_conversion import TemperatureConverter

from .const import (
    ATTR_BEEP,
    ATTR_CLEAN,
    ATTR_ECONO,
    ATTR_FILTERS,
    ATTR_LAST_ON_MODE,
    ATTR_LIGHT,
    ATTR_QUIET,
    ATTR_SLEEP,
    ATTR_STATE_MODE,
    ATTR_SWINGH,
    ATTR_SWINGV,
    ATTR_TURBO,
    ATTRIBUTES_IRHVAC,
    CONF_AVAILABILITY_TOPIC,
    CONF_IR_ACTIONS,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PI_DEADBAND,
    CONF_PI_ENABLED,
    CONF_PI_FF_COOL_REFERENCE,
    CONF_PI_FF_COOL_SLOPE,
    CONF_PI_FF_HEAT_REFERENCE,
    CONF_PI_FF_HEAT_SLOPE,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_MIN_INTERVAL,
    CONF_PI_SETPOINT_WEIGHT,
    CONF_PRESET_MODES_LIST,
    CONF_AWAY_TEMP,
    CONF_BEEP,
    CONF_CELSIUS,
    CONF_CLEAN,
    CONF_COMMAND_TOPIC,
    CONF_ECONO,
    CONF_EXCLUSIVE_GROUP_VENDOR,
    CONF_FAN_LIST,
    CONF_FILTER,
    CONF_HUMIDITY_SENSOR,
    CONF_IGNORE_OFF_TEMP,
    CONF_INITIAL_OPERATION_MODE,
    CONF_KEEP_MODE,
    CONF_LIGHT,
    CONF_MAX_TEMP,
    CONF_MIN_TEMP,
    CONF_MODEL,
    CONF_MODES_LIST,
    CONF_MQTT_DELAY,
    CONF_POWER_SENSOR,
    CONF_PRECISION,
    CONF_PROTOCOL,
    CONF_QUIET,
    CONF_SLEEP,
    CONF_SPECIAL_MODE,
    CONF_STATE_TOPIC,
    CONF_STATE_TOPIC_2,
    CONF_SWING_LIST,
    CONF_SWINGH,
    CONF_SWINGV,
    CONF_TARGET_TEMP,
    CONF_TEMP_SENSOR,
    CONF_TEMP_STEP,
    CONF_TOGGLE_LIST,
    CONF_TURBO,
    CONF_VENDOR,
    DATA_KEY,
    DEFAULT_COMMAND_TOPIC,
    DEFAULT_CONF_BEEP,
    DEFAULT_CONF_CELSIUS,
    DEFAULT_CONF_CLEAN,
    DEFAULT_CONF_ECONO,
    DEFAULT_CONF_FILTER,
    DEFAULT_CONF_KEEP_MODE,
    DEFAULT_CONF_LIGHT,
    DEFAULT_CONF_MODEL,
    DEFAULT_CONF_QUIET,
    DEFAULT_CONF_SLEEP,
    DEFAULT_CONF_TURBO,
    DEFAULT_FAN_LIST,
    DEFAULT_IGNORE_OFF_TEMP,
    DEFAULT_MAX_TEMP,
    DEFAULT_PI_DEADBAND,
    DEFAULT_PI_ENABLED,
    DEFAULT_PI_FF_COOL_REFERENCE,
    DEFAULT_PI_FF_COOL_SLOPE,
    DEFAULT_PI_FF_HEAT_REFERENCE,
    DEFAULT_PI_FF_HEAT_SLOPE,
    DEFAULT_PI_KI,
    DEFAULT_PI_KP,
    DEFAULT_PI_MIN_INTERVAL,
    DEFAULT_PI_SETPOINT_WEIGHT,
    DEFAULT_MIN_TEMP,
    DEFAULT_MQTT_DELAY,
    DEFAULT_NAME,
    DEFAULT_PRECISION,
    DEFAULT_STATE_MODE,
    DEFAULT_STATE_TOPIC,
    DEFAULT_TARGET_TEMP,
    DOMAIN,
    HVAC_FAN_AUTO_MAX,
    HVAC_FAN_MAX,
    HVAC_FAN_MAX_HIGH,
    HVAC_FAN_MIN,
    HVAC_MODE_AUTO_FAN,
    HVAC_MODE_FAN_AUTO,
    HVAC_MODES,
    ON_OFF_LIST,
    SERVICE_BEEP_MODE,
    SERVICE_CLEAN_MODE,
    SERVICE_ECONO_MODE,
    SERVICE_FILTERS_MODE,
    SERVICE_LIGHT_MODE,
    SERVICE_QUIET_MODE,
    SERVICE_SET_SWINGH,
    SERVICE_SET_SWINGV,
    SERVICE_SLEEP_MODE,
    SERVICE_TURBO_MODE,
    SWING_AUTO,
    STATE_MODE_LIST,
    TOGGLE_ALL_LIST,
)

DEFAULT_MODES_LIST = [
    HVACMode.COOL,
    HVACMode.HEAT,
    HVACMode.DRY,
    HVAC_MODE_AUTO_FAN,
    HVAC_MODE_FAN_AUTO,
]

DEFAULT_SWING_LIST = [SWING_OFF, SWING_VERTICAL]
DEFAULT_INITIAL_OPERATION_MODE = HVACMode.OFF

_LOGGER = logging.getLogger(__name__)

SUPPORT_FLAGS = ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.FAN_MODE

if hasattr(ClimateEntityFeature, "TURN_ON"):
    SUPPORT_FLAGS |= ClimateEntityFeature.TURN_ON | ClimateEntityFeature.TURN_OFF

PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        vol.Exclusive(CONF_VENDOR, CONF_EXCLUSIVE_GROUP_VENDOR): cv.string,
        vol.Exclusive(CONF_PROTOCOL, CONF_EXCLUSIVE_GROUP_VENDOR): cv.string,
        vol.Required(
            CONF_COMMAND_TOPIC, default=DEFAULT_COMMAND_TOPIC
        ): mqtt.valid_publish_topic,
        vol.Optional(CONF_AVAILABILITY_TOPIC): mqtt.util.valid_topic,
        vol.Optional(CONF_TEMP_SENSOR): cv.entity_id,
        vol.Optional(CONF_HUMIDITY_SENSOR): cv.entity_id,
        vol.Optional(CONF_POWER_SENSOR): cv.entity_id,
        vol.Optional(
            CONF_STATE_TOPIC, default=DEFAULT_STATE_TOPIC
        ): mqtt.valid_subscribe_topic,
        vol.Optional(CONF_STATE_TOPIC + "_2"): mqtt.util.valid_topic,
        vol.Optional(CONF_MQTT_DELAY, default=DEFAULT_MQTT_DELAY): vol.Coerce(float),
        vol.Optional(CONF_MAX_TEMP, default=DEFAULT_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_TEMP, default=DEFAULT_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP, default=DEFAULT_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(
            CONF_INITIAL_OPERATION_MODE, default=DEFAULT_INITIAL_OPERATION_MODE
        ): vol.In(HVAC_MODES),
        vol.Optional(CONF_AWAY_TEMP): vol.Coerce(float),
        vol.Optional(CONF_PRECISION, default=DEFAULT_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_TEMP_STEP, default=PRECISION_WHOLE): vol.In(
            [PRECISION_HALVES, PRECISION_WHOLE, 2.0]
        ),
        vol.Optional(CONF_MODES_LIST, default=DEFAULT_MODES_LIST): vol.All(
            cv.ensure_list, [vol.In(HVAC_MODES)]
        ),
        vol.Optional(CONF_FAN_LIST, default=DEFAULT_FAN_LIST): vol.All(
            cv.ensure_list,
            [
                vol.In(
                    [
                        FAN_ON,
                        FAN_OFF,
                        FAN_AUTO,
                        FAN_LOW,
                        FAN_MEDIUM,
                        FAN_HIGH,
                        FAN_MIDDLE,
                        FAN_FOCUS,
                        FAN_DIFFUSE,
                        HVAC_FAN_MIN,
                        FAN_MEDIUM,
                        HVAC_FAN_MAX,
                        FAN_AUTO,
                        HVAC_FAN_MAX_HIGH,
                        HVAC_FAN_AUTO_MAX,
                    ]
                )
            ],
        ),
        vol.Optional(CONF_SWING_LIST, default=DEFAULT_SWING_LIST): vol.All(
            cv.ensure_list,
            [vol.In([SWING_OFF, SWING_BOTH, SWING_VERTICAL, SWING_HORIZONTAL])],
        ),
        vol.Optional(CONF_QUIET, default=DEFAULT_CONF_QUIET): cv.string,
        vol.Optional(CONF_TURBO, default=DEFAULT_CONF_TURBO): cv.string,
        vol.Optional(CONF_ECONO, default=DEFAULT_CONF_ECONO): cv.string,
        vol.Optional(CONF_MODEL, default=DEFAULT_CONF_MODEL): cv.string,
        vol.Optional(CONF_CELSIUS, default=DEFAULT_CONF_CELSIUS): cv.string,
        vol.Optional(CONF_LIGHT, default=DEFAULT_CONF_LIGHT): cv.string,
        vol.Optional(CONF_FILTER, default=DEFAULT_CONF_FILTER): cv.string,
        vol.Optional(CONF_CLEAN, default=DEFAULT_CONF_CLEAN): cv.string,
        vol.Optional(CONF_BEEP, default=DEFAULT_CONF_BEEP): cv.string,
        vol.Optional(CONF_SLEEP, default=DEFAULT_CONF_SLEEP): cv.string,
        vol.Optional(CONF_KEEP_MODE, default=DEFAULT_CONF_KEEP_MODE): cv.boolean,
        vol.Optional(CONF_SWINGV): cv.string,
        vol.Optional(CONF_SWINGH): cv.string,
        vol.Optional(CONF_TOGGLE_LIST, default=[]): vol.All(
            cv.ensure_list,
            [vol.In(TOGGLE_ALL_LIST)],
        ),
        vol.Optional(CONF_IGNORE_OFF_TEMP, default=DEFAULT_IGNORE_OFF_TEMP): cv.boolean,
        vol.Optional(CONF_SPECIAL_MODE, default=""): cv.string,
        vol.Optional(CONF_PRESET_MODES_LIST): vol.All(
            cv.ensure_list, [cv.string]
        ),
        vol.Optional(CONF_PI_ENABLED, default=DEFAULT_PI_ENABLED): cv.boolean,
        vol.Optional(CONF_PI_KP, default=DEFAULT_PI_KP): vol.Coerce(float),
        vol.Optional(CONF_PI_KI, default=DEFAULT_PI_KI): vol.Coerce(float),
        vol.Optional(CONF_PI_MIN_INTERVAL, default=DEFAULT_PI_MIN_INTERVAL): vol.Coerce(int),
        vol.Optional(CONF_PI_DEADBAND, default=DEFAULT_PI_DEADBAND): vol.Coerce(float),
        vol.Optional(CONF_OUTDOOR_TEMP_SENSOR): cv.entity_id,
        vol.Optional(CONF_PI_FF_HEAT_REFERENCE, default=DEFAULT_PI_FF_HEAT_REFERENCE): vol.Coerce(float),
        vol.Optional(CONF_PI_FF_HEAT_SLOPE, default=DEFAULT_PI_FF_HEAT_SLOPE): vol.Coerce(float),
        vol.Optional(CONF_PI_FF_COOL_REFERENCE, default=DEFAULT_PI_FF_COOL_REFERENCE): vol.Coerce(float),
        vol.Optional(CONF_PI_FF_COOL_SLOPE, default=DEFAULT_PI_FF_COOL_SLOPE): vol.Coerce(float),
        vol.Optional(CONF_PI_SETPOINT_WEIGHT, default=DEFAULT_PI_SETPOINT_WEIGHT): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=1.0)
        ),
    }
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(MQTT_ENTITY_COMMON_SCHEMA.schema)
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(mqtt.config.MQTT_BASE_SCHEMA.schema)

IRHVAC_SERVICE_SCHEMA = vol.Schema({vol.Required(ATTR_ENTITY_ID): cv.entity_ids})

SERVICE_SCHEMA_ECONO_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_ECONO): vol.In(ON_OFF_LIST),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_TURBO_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_TURBO): vol.In(ON_OFF_LIST),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_QUIET_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_QUIET): vol.In(ON_OFF_LIST),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_LIGHT_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_LIGHT): vol.In(ON_OFF_LIST),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_FILTERS_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_FILTERS): vol.In(ON_OFF_LIST),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_CLEAN_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_CLEAN): vol.In(ON_OFF_LIST),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_BEEP_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_BEEP): vol.In(ON_OFF_LIST),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_SLEEP_MODE = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_SLEEP): cv.string,
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_SET_SWINGV = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_SWINGV): vol.In(
            ["off", "auto", "highest", "high", "middle", "low", "lowest"]
        ),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)
SERVICE_SCHEMA_SET_SWINGH = IRHVAC_SERVICE_SCHEMA.extend(
    {
        vol.Required(ATTR_SWINGH): vol.In(
            ["off", "auto", "left max", "left", "middle", "right", "right max", "wide"]
        ),
        vol.Optional(ATTR_STATE_MODE, default=DEFAULT_STATE_MODE): vol.In(
            STATE_MODE_LIST
        ),
    }
)

SERVICE_TO_METHOD = {
    SERVICE_ECONO_MODE: {
        "method": "async_set_econo",
        "schema": SERVICE_SCHEMA_ECONO_MODE,
    },
    SERVICE_TURBO_MODE: {
        "method": "async_set_turbo",
        "schema": SERVICE_SCHEMA_TURBO_MODE,
    },
    SERVICE_QUIET_MODE: {
        "method": "async_set_quiet",
        "schema": SERVICE_SCHEMA_QUIET_MODE,
    },
    SERVICE_LIGHT_MODE: {
        "method": "async_set_light",
        "schema": SERVICE_SCHEMA_LIGHT_MODE,
    },
    SERVICE_FILTERS_MODE: {
        "method": "async_set_filters",
        "schema": SERVICE_SCHEMA_FILTERS_MODE,
    },
    SERVICE_CLEAN_MODE: {
        "method": "async_set_clean",
        "schema": SERVICE_SCHEMA_CLEAN_MODE,
    },
    SERVICE_BEEP_MODE: {
        "method": "async_set_beep",
        "schema": SERVICE_SCHEMA_BEEP_MODE,
    },
    SERVICE_SLEEP_MODE: {
        "method": "async_set_sleep",
        "schema": SERVICE_SCHEMA_SLEEP_MODE,
    },
    SERVICE_SET_SWINGV: {
        "method": "async_set_swingv",
        "schema": SERVICE_SCHEMA_SET_SWINGV,
    },
    SERVICE_SET_SWINGH: {
        "method": "async_set_swingh",
        "schema": SERVICE_SCHEMA_SET_SWINGH,
    },
    "reset_ff_seeds": {
        "method": "async_reset_ff_seeds",
        "schema": IRHVAC_SERVICE_SCHEMA,
    },
    "suppress_ff_learning": {
        "method": "async_suppress_ff_learning",
        "schema": IRHVAC_SERVICE_SCHEMA.extend(
            {vol.Optional("reason"): cv.string}
        ),
    },
    "resume_ff_learning": {
        "method": "async_resume_ff_learning",
        "schema": IRHVAC_SERVICE_SCHEMA,
    },
}


async def async_setup_platform(
    hass: HomeAssistant, config: dict[str, Any], async_add_entities: AddEntitiesCallback,
    discovery_info: Any = None,
) -> None:
    """Set up via YAML (deprecated — triggers config entry import)."""
    _LOGGER.warning(
        "Configuration of Tasmota IRHVAC via YAML is deprecated. "
        "Your configuration has been imported. Please remove the YAML "
        "configuration and restart Home Assistant."
    )
    from homeassistant.components.persistent_notification import async_create
    async_create(
        hass,
        "Your Tasmota IRHVAC YAML configuration has been imported into the UI. "
        "Please remove the `platform: tasmota_irhvac` entry from your "
        "configuration.yaml and restart Home Assistant.",
        title="Tasmota IRHVAC YAML Import",
        notification_id="tasmota_irhvac_yaml_import",
    )
    hass.async_create_task(
        hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_IMPORT},
            data=dict(config),
        )
    )


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback,
) -> bool | None:
    """Set up Tasmota IRHVAC climate from a config entry."""
    hass.data.setdefault(DATA_KEY, {})

    config = {**entry.data, **entry.options}

    # Build model_inputs and supplemental_sources from subentries (if available)
    # Falls back to options-based list for pre-subentry config entries.
    if hasattr(entry, "subentries") and entry.subentries:
        from .const import SUBENTRY_MODEL_INPUT, SUBENTRY_SUPPLEMENTAL_SOURCE, CONF_PI_MODEL_INPUTS
        model_inputs_from_subentries = [
            dict(sub.data) for sub in entry.subentries.values()
            if sub.subentry_type == SUBENTRY_MODEL_INPUT
        ]
        supplemental_sources = [
            dict(sub.data) for sub in entry.subentries.values()
            if sub.subentry_type == SUBENTRY_SUPPLEMENTAL_SOURCE
        ]
        if model_inputs_from_subentries:
            config[CONF_PI_MODEL_INPUTS] = model_inputs_from_subentries
        config["pi_supplemental_sources"] = supplemental_sources

    from .config_model import IrhvacConfig
    from .vendors import get_handler

    irhvac_config = IrhvacConfig.from_config_dict(config)

    if not irhvac_config.vendor:
        _LOGGER.error("No vendor configured for %s", entry.title)
        return False

    entity = TasmotaIrhvac(hass, irhvac_config, get_handler(irhvac_config.vendor))

    if entity.unique_id is None:
        entity._attr_unique_id = entry.entry_id
    entity._config_entry_id = entry.entry_id

    hass.data[DATA_KEY][entry.entry_id] = entity
    async_add_entities([entity])
    return None


from .pi_controller import PIController


class TasmotaIrhvac(RestoreEntity, ClimateEntity):
    """Representation of a Generic Thermostat device."""

    # It can remove from HA >= 2025.1
    # see https://developers.home-assistant.io/blog/2024/01/24/climate-climateentityfeatures-expanded/
    _enable_turn_on_off_backwards_compatibility = False

    _last_on_mode: HVACMode | None
    _config_entry_id: str
    _vendor_timer_unsub: CALLBACK_TYPE | None

    def __init__(
        self,
        hass: HomeAssistant,
        config: IrhvacConfig | dict[str, Any],
        vendor_handler: VendorHandler | None = None,
    ) -> None:
        """Initialize the thermostat.

        Args:
            hass: Home Assistant instance.
            config: IrhvacConfig (frozen dataclass) or raw dict (legacy/tests).
            vendor_handler: VendorHandler instance (from registry).
        """
        from .vendors.base import VendorHandler
        from .config_model import IrhvacConfig

        # Accept both IrhvacConfig and raw dict (for backward compat in tests)
        if isinstance(config, dict):
            cfg = IrhvacConfig.from_config_dict(config)
            raw_config = config
        else:
            cfg = config
            raw_config = cfg.pi_raw_config

        self._vendor_handler = vendor_handler or VendorHandler()
        self.topic = cfg.command_topic
        self.hass = hass
        self._vendor = cfg.vendor
        self._temp_sensor = cfg.temp_sensor
        self._humidity_sensor = cfg.humidity_sensor
        self._power_sensor = cfg.power_sensor
        self.state_topic = cfg.state_topic
        self.state_topic2 = cfg.state_topic_2
        self._away_temp: float | None = cfg.away_temp
        self._saved_target_temp: float | None = cfg.target_temp or cfg.away_temp
        self._temp_precision: float | None = cfg.precision
        self._enabled: bool = False
        self.power_mode: str | None = None
        self._active: bool = False
        self._mqtt_delay = cfg.mqtt_delay
        self._min_temp = cfg.min_temp
        self._max_temp = cfg.max_temp
        self._def_target_temp = cfg.target_temp
        self._is_away = False
        self._modes_list = cfg.modes_list
        self._quiet = cfg.quiet
        self._turbo = cfg.turbo
        self._econo = cfg.econo
        self._model = cfg.model
        self._ir_protocol_unit = cfg.ir_protocol_unit  # "celsius" or "fahrenheit"
        self._light = cfg.light
        self._filter = cfg.filter
        self._clean = cfg.clean
        self._beep = cfg.beep
        self._sleep = cfg.sleep
        self._sub_state: str | None = None
        self._keep_mode = cfg.keep_mode
        self._last_on_mode: HVACMode | None = None
        self._swingv: str | None = cfg.swingv
        self._swingh: str | None = cfg.swingh
        self._fix_swingv: str | None = None
        self._fix_swingh: str | None = None
        self._toggle_list = cfg.toggle_list
        self._state_mode = DEFAULT_STATE_MODE
        self._ignore_off_temp = cfg.ignore_off_temp
        self._special_mode = cfg.special_mode
        self._use_track_state_change_event: bool = False
        self._unsubscribes: list[Any] = []

        self.availability_topic: str = cfg.availability_topic or ""
        if not self.availability_topic:
            path = self.topic.split("/")
            self.availability_topic = "tele/" + path[1] + "/LWT"

        # Set _attr_*
        self._attr_unique_id = cfg.unique_id
        self._attr_name = cfg.name
        self._attr_should_poll = False
        # Entity temperature unit = HA system unit (what the user sees).
        # _ir_temp_unit is the unit the IR protocol speaks (for Tasmota payloads).
        self._attr_temperature_unit = hass.config.units.temperature_unit
        self._ir_temp_unit = (
            UnitOfTemperature.CELSIUS
            if self._ir_protocol_unit == "celsius"
            else UnitOfTemperature.FAHRENHEIT
        )
        # Config temps are stored in system unit after v1.7 migration.
        # No runtime conversion needed.
        self._attr_hvac_mode = cfg.initial_operation_mode  # type: ignore[assignment]
        self._attr_target_temperature_step = cfg.temp_step
        self._attr_hvac_modes = cfg.modes_list  # type: ignore[assignment]
        self._attr_fan_modes = cfg.fan_list
        if isinstance(self._attr_fan_modes, list):
            self._attr_fan_modes = self._vendor_handler.transform_fan_modes(
                self._attr_fan_modes
            ) or None
        self._attr_fan_mode = (
            self._attr_fan_modes[0]
            if isinstance(self._attr_fan_modes, list) and len(self._attr_fan_modes)
            else None
        )
        self._attr_swing_modes = cfg.swing_list
        self._attr_swing_mode = (
            self._attr_swing_modes[0]
            if isinstance(self._attr_swing_modes, list) and len(self._attr_swing_modes)
            else None
        )
        self._attr_preset_mode = None
        self._attr_current_temperature = None
        self._attr_current_humidity = None
        self._attr_target_temperature = None

        self._support_flags = SUPPORT_FLAGS
        if self._attr_swing_mode is not None:
            self._support_flags = self._support_flags | ClimateEntityFeature.SWING_MODE

        # Build preset list from base (Away) + vendor config, deduplicated
        presets = [PRESET_NONE]
        if self._away_temp:
            presets.append(PRESET_AWAY)
        if cfg.preset_modes_list:
            presets.extend(cfg.preset_modes_list)
        # Deduplicate while preserving order
        seen = set()
        unique_presets = []
        for mode in presets:
            if mode not in seen:
                seen.add(mode)
                unique_presets.append(mode)
        # Add IR action presets from config
        self._ir_action_presets = {}
        for action in cfg.ir_actions:
            if action.get("type") == "preset":
                name = action["name"]
                if name not in seen:
                    unique_presets.append(name)
                    seen.add(name)
                self._ir_action_presets[name] = action

        if len(unique_presets) > 1 or self._away_temp or self._ir_action_presets:
            self._attr_preset_modes = unique_presets
            self._support_flags |= ClimateEntityFeature.PRESET_MODE
        else:
            self._attr_preset_modes = None

        # Controller: PIController when enabled, NullController otherwise.
        # All calls are unconditional — no `if self._pi:` guards needed.
        from .controller_protocol import NullController
        self._controller = PIController(self, raw_config) if cfg.pi_enabled else NullController()
        # Legacy alias for tests that reference self._pi directly
        self._pi = self._controller if cfg.pi_enabled else None

        # Echo classification state (only active when PI is active)
        self._has_sent_once: bool = False
        self._expected_state: dict[str, Any] = {}

        # PI recovery subscription (PI owns its own fallback timer)
        self._pi_recovery_unsub: CALLBACK_TYPE | None = None

    async def async_added_to_hass(self) -> None:
        def regist_track_state_change_event(entity_id: str) -> None:
            ha_event.async_track_state_change_event(
                self.hass, entity_id, self._async_sensor_changed
            )

        # Make sure MQTT integration is enabled and the client is available
        await mqtt.async_wait_for_mqtt_client(self.hass)

        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listener
        self._unsubscribes = await self._subscribe_topics()

        # Check If we have an old state
        old_state = await self.async_get_last_state()
        if old_state is not None:
            # If we have no initial temperature, restore
            if old_state.attributes.get(ATTR_TEMPERATURE) is not None:
                self._attr_target_temperature = TemperatureConverter.convert(
                    float(old_state.attributes[ATTR_TEMPERATURE]),
                    self.hass.config.units.temperature_unit,
                    self.temperature_unit,
                )
            if old_state.attributes.get(ATTR_PRESET_MODE) == PRESET_AWAY:
                self._is_away = True
            if old_state.attributes.get(ATTR_FAN_MODE) is not None:
                self._attr_fan_mode = old_state.attributes.get(ATTR_FAN_MODE)
            if old_state.attributes.get(ATTR_SWING_MODE) is not None:
                self._attr_swing_mode = old_state.attributes.get(ATTR_SWING_MODE)
            if old_state.attributes.get(ATTR_LAST_ON_MODE) is not None:
                self._last_on_mode = old_state.attributes.get(ATTR_LAST_ON_MODE)

            for attr, prop in ATTRIBUTES_IRHVAC.items():
                val = old_state.attributes.get(attr)
                if val is not None:
                    setattr(self, "_" + prop, val)
            if old_state.state:
                self._attr_hvac_mode = (
                    HVACMode.OFF
                    if old_state.state in [STATE_UNKNOWN, STATE_UNAVAILABLE]
                    else old_state.state  # type: ignore[assignment]
                )
                self._enabled = self._attr_hvac_mode != HVACMode.OFF
                if self._enabled:
                    self._last_on_mode = self._attr_hvac_mode
            if self._swingv != "auto":
                self._fix_swingv = self._swingv
            if self._swingh != "auto":
                self._fix_swingh = self._swingh

        # Let vendor handler restore its state from the previous preset
        restored_preset = (
            old_state.attributes.get(ATTR_PRESET_MODE) if old_state else None
        )
        self._vendor_handler.on_restore_state(restored_preset)
        if self._vendor_handler.should_pause_controller:
            self._controller.pi_pause()

        # No previous target temperature, try and restore defaults
        if self._attr_target_temperature is None or self._attr_target_temperature < 1:
            self._attr_target_temperature = self._def_target_temp
            _LOGGER.warning(
                "No previously saved target temperature, setting to default value %s",
                self._attr_target_temperature,
            )

        if self._attr_hvac_mode is HVACMode.OFF:
            self.power_mode = STATE_OFF
            self._enabled = False
        else:
            self.power_mode = STATE_ON
            self._enabled = True

        for key in self._toggle_list:
            setattr(self, "_" + key.lower(), "off")

        if self._temp_sensor:
            regist_track_state_change_event(self._temp_sensor)

            temp_sensor_state = self.hass.states.get(self._temp_sensor)
            if (
                temp_sensor_state
                and temp_sensor_state.state != STATE_UNKNOWN
                and temp_sensor_state.state != STATE_UNAVAILABLE
            ):
                self._async_update_temp(temp_sensor_state)

        if self._humidity_sensor:
            regist_track_state_change_event(self._humidity_sensor)

            humidity_sensor_state = self.hass.states.get(self._humidity_sensor)
            if (
                humidity_sensor_state
                and humidity_sensor_state.state != STATE_UNKNOWN
                and humidity_sensor_state.state != STATE_UNAVAILABLE
            ):
                self._async_update_humidity(humidity_sensor_state)

        if self._power_sensor:
            regist_track_state_change_event(self._power_sensor)

        # Initialize PI controller (restores state, no I/O)
        await self._controller.async_added_to_hass(old_state=old_state)

        # PI fallback timer — PI reschedules after every tick, climate.py
        # provides the callback that bridges timer fire → send_ir.
        if self._controller.is_active and self._temp_sensor:
            @callback
            def _pi_timer_fired(_now):
                self.hass.async_create_task(self._on_pi_timer())
            self._controller._pi_timer_callback = _pi_timer_fired
            self._controller.schedule_batch_analysis()
            if self._attr_current_temperature is not None:
                @callback
                def _deferred_initial_tick(_now):
                    self.hass.async_create_task(self._on_pi_timer())
                async_call_later(self.hass, 5, _deferred_initial_tick)

    async def _subscribe_topics(self) -> list[Any]:
        """(Re)Subscribe to topics."""

        @callback
        async def available_message_received(message: mqtt.ReceiveMessage) -> None:
            msg = message.payload
            _LOGGER.debug(msg)
            if msg == "Online" or msg == "Offline":
                self._attr_available = True if msg == "Online" else False
                self.async_schedule_update_ha_state()

        @callback
        async def state_message_received(message: mqtt.ReceiveMessage) -> None:
            """Handle MQTT state from any topic (tele or stat)."""
            await self._process_mqtt_state(message)

        unsubscribe = []
        unsubscribe.append(
            await mqtt.async_subscribe(
                self.hass, self.state_topic, state_message_received
            )
        )
        unsubscribe.append(
            await mqtt.async_subscribe(
                self.hass, self.availability_topic, available_message_received
            )
        )
        if self.state_topic2:
            unsubscribe.append(
                await mqtt.async_subscribe(
                    self.hass, self.state_topic2, state_message_received
                )
            )

        return unsubscribe

    async def _process_mqtt_state(
        self, message: mqtt.ReceiveMessage,
    ) -> None:
        """Parse MQTT message, extract ir_received transport fact, dispatch."""
        try:
            json_payload = json.loads(message.payload)
        except ValueError:
            _LOGGER.error("Unable to parse MQTT payload as JSON: %s", message.payload)
            return
        _LOGGER.debug(json_payload)

        # IrReceived wrapper means the IR receiver decoded a signal —
        # could be our own transmission bouncing back or a physical remote.
        # This is a transport fact; the controller combines it with timing
        # to classify the message.
        ir_received = "IrReceived" in json_payload
        if ir_received:
            json_payload = json_payload["IrReceived"]

        if "IRHVAC" not in json_payload:
            return

        payload = json_payload["IRHVAC"]
        await self._handle_state_payload(
            json_payload, payload,
            ir_received=ir_received,
        )

    def _payload_matches_expected(self, payload: dict[str, Any]) -> bool:
        """Compare incoming payload against expected state using vendor precision.

        Returns True if all fields match (echo/confirmation), False if any differ.
        """
        if not self._expected_state:
            return False
        prec = self._temp_precision or 1.0
        for key, expected in self._expected_state.items():
            if key not in payload:
                continue
            incoming = payload[key]
            if key == "Temp":
                # Compare at vendor precision to avoid float→int mismatch
                if round(incoming / prec) * prec != round(expected / prec) * prec:
                    return False
            elif key == "Sleep":
                # We send "off", Tasmota echoes -1; both mean "no timer"
                _SLEEP_OFF = {-1, "-1", "off"}
                if (incoming in _SLEEP_OFF) != (expected in _SLEEP_OFF):
                    return False
            else:
                # String fields: case-insensitive
                if str(incoming).lower() != str(expected).lower():
                    return False
        return True

    async def _handle_state_payload(
        self, json_payload: dict[str, Any], payload: dict[str, Any],
        *, ir_received: bool = False,
    ) -> None:
        """Process IRHVAC state payload."""
        if payload["Vendor"] == self._vendor:
            # ── Echo classification (PI active + has sent at least once) ──
            if self._controller.is_active and self._has_sent_once:
                matches = self._payload_matches_expected(payload)
                if matches:
                    # Cases 1 & 3: echo or confirmation — no state change
                    _LOGGER.debug(
                        "%s MQTT %s: payload matches expected, ignoring",
                        self.entity_id, "echo" if ir_received else "confirmation",
                    )
                    return
                if ir_received:
                    # Case 2: physical remote — state changed via IrReceived
                    _LOGGER.info("%s Physical remote detected (state diff)", self.entity_id)
                    # Fall through to apply state, then notify PI
                else:
                    # Case 4: mismatch without IrReceived — resend
                    _LOGGER.warning(
                        "%s Telemetry mismatch (no IrReceived), resending",
                        self.entity_id,
                    )
                    await self.send_ir()
                    return

            # Build IRDecode + EntityState for vendor handler hooks
            from .vendors.base import IRDecode, EntityState
            decode = IRDecode(
                irhvac=payload,
                protocol=json_payload.get("Protocol"),
                bits=json_payload.get("Bits"),
                data=json_payload.get("Data"),
                ir_received=ir_received,
            )
            entity_state = EntityState(
                hvac_mode=self._attr_hvac_mode,
                target_temperature=self.target_temperature,
                fan_mode=self._attr_fan_mode,
                swing_mode=self._attr_swing_mode,
                swingv=self._swingv,
                swingh=self._swingh,
                power_mode=self.power_mode,
            )
            self._vendor_handler.pre_state_processing(decode, entity_state)
            # All values in the payload are Optional
            prev_power = self.power_mode
            if "Power" in payload:
                self.power_mode = payload["Power"].lower()
            if "Mode" in payload:
                self._attr_hvac_mode = payload["Mode"].lower()
                # Some vendors send/receive mode as fan instead of fan_only
                if self._attr_hvac_mode == HVACAction.FAN:
                    self._attr_hvac_mode = HVACMode.FAN_ONLY
            if "Temp" in payload:
                if payload["Temp"] > 0:
                    # Sanity check: reject impossible temps (no HVAC uses 0-50°C range)
                    temp_c = TemperatureConverter.convert(
                        payload["Temp"], self._ir_temp_unit, UnitOfTemperature.CELSIUS
                    )
                    if temp_c < 0 or temp_c > 50:
                        _LOGGER.warning(
                            "MQTT state: ignoring impossible Temp %s (%s°C)",
                            payload["Temp"], temp_c,
                        )
                    elif self.power_mode == STATE_OFF and self._ignore_off_temp:
                        pass  # Keep existing target temp
                    elif self._controller.is_active:
                        pass  # PI handler manages target temp separately
                    else:
                        # Convert from IR unit (celsius_mode) to entity unit (system)
                        temp = payload["Temp"]
                        if self._ir_temp_unit != self._attr_temperature_unit:
                            temp = TemperatureConverter.convert(
                                temp, self._ir_temp_unit, self._attr_temperature_unit
                            )
                        self._attr_target_temperature = temp
            if "Celsius" in payload:
                # Tasmota reports "On"/"Off" for Celsius field; map to our format
                self._ir_protocol_unit = (
                    "celsius" if payload["Celsius"].lower() == "on" else "fahrenheit"
                )
            if "Quiet" in payload:
                self._quiet = payload["Quiet"].lower()
            if "Turbo" in payload:
                self._turbo = payload["Turbo"].lower()
            if "Econo" in payload:
                self._econo = payload["Econo"].lower()
            if "Light" in payload:
                self._light = payload["Light"].lower()
            if "Filter" in payload:
                self._filter = payload["Filter"].lower()
            if "Clean" in payload:
                self._clean = payload["Clean"].lower()
            if "Beep" in payload:
                self._beep = payload["Beep"].lower()
            if "Sleep" in payload:
                self._sleep = payload["Sleep"]
            if "SwingV" in payload:
                self._swingv = payload["SwingV"].lower()
                if self._swingv != "auto":
                    self._fix_swingv = self._swingv
            if "SwingH" in payload:
                self._swingh = payload["SwingH"].lower()
                if self._swingh != "auto":
                    self._fix_swingh = self._swingh
            if (
                "SwingV" in payload
                and payload["SwingV"].lower() == SWING_AUTO
                and "SwingH" in payload
                and payload["SwingH"].lower() == SWING_AUTO
            ):
                if SWING_BOTH in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_BOTH
                elif SWING_VERTICAL in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_VERTICAL
                elif SWING_HORIZONTAL in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_HORIZONTAL
                else:
                    self._attr_swing_mode = SWING_OFF
            elif (
                "SwingV" in payload
                and payload["SwingV"].lower() == SWING_AUTO
                and SWING_VERTICAL in (self._attr_swing_modes or [])
            ):
                self._attr_swing_mode = SWING_VERTICAL
            elif (
                "SwingH" in payload
                and payload["SwingH"].lower() == SWING_AUTO
                and SWING_HORIZONTAL in (self._attr_swing_modes or [])
            ):
                self._attr_swing_mode = SWING_HORIZONTAL
            else:
                self._attr_swing_mode = SWING_OFF

            if "FanSpeed" in payload:
                fan_mode = payload["FanSpeed"].lower()
                # ELECTRA_AC fan modes fix
                if HVAC_FAN_MAX_HIGH in (  # pragma: no cover — upstream bug, see #184
                    self._attr_fan_modes or []
                ) and HVAC_FAN_AUTO_MAX in (self._attr_fan_modes or []):
                    # NOTE: This block is unreachable because __init__ transforms
                    # HVAC_FAN_MAX_HIGH/HVAC_FAN_AUTO_MAX out of _attr_fan_modes,
                    # so the enclosing `if` condition always fails.
                    # Upstream bug: the init transformation was added after this
                    # mapping code, making it dead. See upstream issue #184.
                    if fan_mode == HVAC_FAN_MAX:
                        self._attr_fan_mode = FAN_HIGH
                    elif fan_mode == FAN_AUTO:  # pragma: no cover
                        self._attr_fan_mode = HVAC_FAN_MAX
                    else:
                        self._attr_fan_mode = fan_mode
                else:
                    self._attr_fan_mode = fan_mode
                _LOGGER.debug(self._attr_fan_mode)

            if self._attr_hvac_mode is not HVACMode.OFF:
                self._last_on_mode = self._attr_hvac_mode

            # Set default state to off
            if self.power_mode == STATE_OFF:
                self._attr_hvac_mode = HVACMode.OFF
                self._enabled = False
            else:
                self._enabled = True

            for key in self._toggle_list:
                setattr(self, "_" + key.lower(), "off")

            # Vendor handler post-processing (preset detection, 56-bit, etc.)
            self._vendor_handler.post_state_processing(decode)

            self._apply_vendor_handler_state()

            # If physical remote detected (ir_received + state diff), notify PI
            if ir_received and self._controller.is_active and "Temp" in payload and payload["Temp"] > 0:
                if await self._controller.on_remote_change(payload["Temp"]):
                    await self.send_ir()
            self.async_schedule_update_ha_state()

            # Check power sensor state
            if (
                self._power_sensor
                and prev_power is not None
                and prev_power != self.power_mode
            ):
                await asyncio.sleep(3)
                state = self.hass.states.get(self._power_sensor)
                # It's probably running in a special mode, such as an automatic cleaning function.
                is_special_mode = (
                    True if state is not None and state.state else False
                )
                await self._async_power_sensor_changed(None, state, is_special_mode)

    # ── Vendor handler helpers ───────────────────────────────────────

    def _apply_vendor_state_restore(self) -> None:
        """Apply state_restore from vendor handler to entity attributes."""
        restore = self._vendor_handler.state_restore
        if restore is None:
            return
        if restore.hvac_mode is not None:
            self._attr_hvac_mode = restore.hvac_mode  # type: ignore[assignment]
        if restore.target_temperature is not None:
            temp = restore.target_temperature
            if self._ir_temp_unit != self._attr_temperature_unit:
                temp = TemperatureConverter.convert(
                    temp, UnitOfTemperature.CELSIUS,
                    self._attr_temperature_unit,
                )
            self._attr_target_temperature = temp
        if restore.fan_mode is not None:
            self._attr_fan_mode = restore.fan_mode
        if restore.swing_mode is not None:
            self._attr_swing_mode = restore.swing_mode
        # swingv/swingh: always apply from restore (None means "clear")
        self._swingv = restore.swingv
        self._swingh = restore.swingh
        if restore.power_mode is not None:
            self.power_mode = restore.power_mode

    def _apply_vendor_handler_state(self) -> None:
        """Read vendor handler properties and apply to entity state."""
        if self._vendor_handler.active_preset is not None:
            self._attr_preset_mode = self._vendor_handler.active_preset
        self._apply_vendor_state_restore()
        if self._vendor_handler.clear_toggles:
            self._econo = "off"
            self._turbo = "off"
            self._clean = "off"
        if self._vendor_handler.should_pause_controller:
            self._controller.pi_pause()
        elif not self._vendor_handler.should_pause_controller:
            self._controller.pi_resume()
        if self._vendor_handler.should_reset_integral:
            self._controller.pi_reset_integral()

    async def _send_raw_ir(self, raw_code: str) -> None:
        """Send a raw IR code via Tasmota's IRSend command."""
        path = self.topic.split("/")
        irsend_topic = f"cmnd/{path[1]}/irsend"
        if float(self._mqtt_delay) != 0.0:
            await asyncio.sleep(float(self._mqtt_delay))
        await mqtt.async_publish(self.hass, irsend_topic, raw_code)

    def _schedule_vendor_timer(self, timer_request: TimerRequest) -> None:
        """Schedule a vendor handler timer callback."""
        # Cancel any existing vendor timer
        if hasattr(self, "_vendor_timer_unsub") and self._vendor_timer_unsub:
            self._vendor_timer_unsub()

        @callback
        def _on_vendor_timer(_now=None):
            self._vendor_timer_unsub = None
            self._vendor_handler.on_timer(timer_request.callback_id)
            # Apply handler state after timer fires
            if self._vendor_handler.active_preset is not None:
                self._attr_preset_mode = self._vendor_handler.active_preset
            if not self._vendor_handler.should_pause_controller:
                self._controller.pi_resume()
            self.async_schedule_update_ha_state()

        self._vendor_timer_unsub = async_call_later(
            self.hass, timer_request.delay_seconds, _on_vendor_timer
        )

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe when removed."""
        if hasattr(self, "_vendor_timer_unsub") and self._vendor_timer_unsub:
            self._vendor_timer_unsub()
            self._vendor_timer_unsub = None
        if self._pi_recovery_unsub:
            self._pi_recovery_unsub()
            self._pi_recovery_unsub = None
        self._controller.async_will_remove_from_hass()
        for unsubscribe in self._unsubscribes:
            unsubscribe()

    def async_write_ha_state(self) -> None:
        """Write state and fire PI dispatcher signal for companion sensors."""
        super().async_write_ha_state()
        self._controller.fire_dispatcher()

    @property
    def extra_restore_state_data(self) -> ExtraStoredData | None:
        """Return PI data for ExtraStoredData persistence."""
        return self._controller.get_extra_stored_data()

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to register this entity in the device registry."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.unique_id or self._config_entry_id)},
            name=self._attr_name,
            manufacturer=self._vendor,
        )

    @property
    def precision(self) -> float:
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    # This extension property is written throughout the instance, so use @property instead of @cached_property.
    @property
    def hvac_action(self) -> HVACAction | None:
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._attr_hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        elif self._attr_hvac_mode == HVACMode.HEAT:
            return HVACAction.HEATING
        elif self._attr_hvac_mode == HVACMode.COOL:
            return HVACAction.COOLING
        elif self._attr_hvac_mode == HVACMode.DRY:
            return HVACAction.DRYING
        elif self._attr_hvac_mode == HVACMode.FAN_ONLY:
            return HVACAction.FAN
        return None

    @property
    def target_temperature(self) -> float | None:
        """Return target temperature — single source of truth.

        When controller is active (PI), reads from controller.desired_temp.
        Otherwise reads from _attr_target_temperature (entity-owned).
        """
        if self._controller.is_active and self._controller.desired_temp is not None:
            return self._controller.desired_temp
        return self._attr_target_temperature

    # This extension property is written throughout the instance, so use @property instead of @cached_property.
    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the device."""
        attrs = {
            attr: getattr(self, "_" + prop) for attr, prop in ATTRIBUTES_IRHVAC.items()
        }
        attrs.update(self._controller.get_extra_state_attributes())
        return attrs

    @property
    def last_on_mode(self) -> HVACMode | None:
        """Return the last non-idle mode ie. heat, cool."""
        return self._last_on_mode

    @property
    def hvac_modes(self) -> list[Any]:
        """Return the list of available HVAC modes."""
        return self._controller.filter_hvac_modes(self._attr_hvac_modes)

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if self._controller.should_reject_hvac_mode(hvac_mode):
            _LOGGER.warning(
                "PI mode does not support %s — use HEAT or COOL explicitly",
                hvac_mode,
            )
            return
        await self.set_mode(hvac_mode)
        # Ensure we update the current operation after changing the mode
        await self.async_send_cmd()

    async def async_turn_on(self) -> None:
        """Turn thermostat on."""
        self._attr_hvac_mode = (
            self._last_on_mode if self._last_on_mode is not None else HVACMode.AUTO
        )
        self.power_mode = STATE_ON
        await self.async_send_cmd()

    async def async_turn_off(self) -> None:
        """Turn thermostat off."""
        self._attr_hvac_mode = HVACMode.OFF
        self.power_mode = STATE_OFF
        await self.async_send_cmd()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        temperature = kwargs.get(ATTR_TEMPERATURE)
        hvac_mode = kwargs.get(ATTR_HVAC_MODE)
        if temperature is None:
            return

        # Controller handles its own setpoint logic when active
        if self._controller.is_active:
            _LOGGER.info(
                "async_set_temperature: temp=%s unit=%s max=%s "
                "BEFORE target=%s desired=%s",
                temperature, self.temperature_unit, self.max_temp,
                self.target_temperature, self._controller.desired_temp,
            )
            if await self._controller.set_temperature(temperature, hvac_mode):
                await self.send_ir()
            self.async_schedule_update_ha_state()
            return

        if hvac_mode is not None:
            await self.set_mode(hvac_mode)

        self._attr_target_temperature = temperature
        if not self._attr_hvac_mode == HVACMode.OFF:
            self.power_mode = STATE_ON
        await self.async_send_cmd()

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Set new target fan mode."""
        if fan_mode not in (self._attr_fan_modes or []):
            # tweak for some ELECTRA_AC devices
            if HVAC_FAN_MAX_HIGH in (  # pragma: no cover — upstream bug #184
                self._attr_fan_modes or []
            ) and HVAC_FAN_AUTO_MAX in (self._attr_fan_modes or []):
                if fan_mode != FAN_HIGH and fan_mode != HVAC_FAN_MAX:
                    _LOGGER.error(
                        "Invalid swing mode selected. Got '%s'. Allowed modes are:",
                        fan_mode,
                    )
                    _LOGGER.error(self._attr_fan_modes)
                    return
            else:
                _LOGGER.error(
                    "Invalid swing mode selected. Got '%s'. Allowed modes are:",
                    fan_mode,
                )
                _LOGGER.error(self._attr_fan_modes)
                return
        self._attr_fan_mode = fan_mode
        if not self._attr_hvac_mode == HVACMode.OFF:
            self.power_mode = STATE_ON
        await self.async_send_cmd()

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        """Set new target swing operation."""
        if swing_mode not in (self._attr_swing_modes or []):
            _LOGGER.error(
                "Invalid swing mode selected. Got '%s'. Allowed modes are:", swing_mode
            )
            _LOGGER.error(self._attr_swing_modes)
            return
        self._attr_swing_mode = swing_mode
        # note: set _swingv and _swingh in send_ir() later
        if not self._attr_hvac_mode == HVACMode.OFF:
            self.power_mode = STATE_ON
        await self.async_send_cmd()

    async def async_set_econo(self, econo: str, state_mode: str) -> None:
        """Set new target econo mode."""
        if econo not in ON_OFF_LIST:
            return
        self._econo = econo.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_turbo(self, turbo: str, state_mode: str) -> None:
        """Set new target turbo mode."""
        if turbo not in ON_OFF_LIST:
            return
        self._turbo = turbo.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_quiet(self, quiet: str, state_mode: str) -> None:
        """Set new target quiet mode."""
        if quiet not in ON_OFF_LIST:
            return
        self._quiet = quiet.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_light(self, light: str, state_mode: str) -> None:
        """Set new target light mode."""
        if light not in ON_OFF_LIST:
            return
        self._light = light.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_filters(self, filters: str, state_mode: str) -> None:
        """Set new target filters mode."""
        if filters not in ON_OFF_LIST:
            return
        self._filter = filters.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_clean(self, clean: str, state_mode: str) -> None:
        """Set new target clean mode."""
        if clean not in ON_OFF_LIST:
            return
        self._clean = clean.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_beep(self, beep: str, state_mode: str) -> None:
        """Set new target beep mode."""
        if beep not in ON_OFF_LIST:
            return
        self._beep = beep.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_sleep(self, sleep: str, state_mode: str) -> None:
        """Set new target sleep mode."""
        self._sleep = sleep.lower()
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_swingv(self, swingv: str, state_mode: str) -> None:
        """Set new target swingv."""
        self._swingv = swingv.lower()
        if self._swingv != "auto":
            self._fix_swingv = self._swingv
            if self._attr_swing_mode == SWING_BOTH:
                if SWING_HORIZONTAL in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_HORIZONTAL
            elif self._attr_swing_mode == SWING_VERTICAL:
                self._attr_swing_mode = SWING_OFF
        else:
            if self._attr_swing_mode == SWING_HORIZONTAL:
                if SWING_BOTH in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_BOTH
            else:
                if SWING_VERTICAL in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_VERTICAL
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_set_swingh(self, swingh: str, state_mode: str) -> None:
        """Set new target swingh."""
        self._swingh = swingh.lower()
        if self._swingh != "auto":
            self._fix_swingh = self._swingh
            if self._attr_swing_mode == SWING_BOTH:
                if SWING_VERTICAL in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_VERTICAL
            elif self._attr_swing_mode == SWING_HORIZONTAL:
                self._attr_swing_mode = SWING_OFF
        else:
            if self._attr_swing_mode == SWING_VERTICAL:
                if SWING_BOTH in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_BOTH
            else:
                if SWING_HORIZONTAL in (self._attr_swing_modes or []):
                    self._attr_swing_mode = SWING_HORIZONTAL
        self._state_mode = state_mode
        await self.async_send_cmd()

    async def async_send_cmd(self) -> None:
        await self.send_ir()

    @cached_property
    def min_temp(self) -> float:
        """Return the minimum temperature."""
        if self._min_temp:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @cached_property
    def max_temp(self) -> float:
        """Return the maximum temperature."""
        if self._max_temp:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp

    def _check_pi_recovery_needed(self) -> None:
        """If PI flagged a sensor recovery check, schedule it."""
        from .pi_controller import PIController
        pi = self._pi if isinstance(self._pi, PIController) else None
        if pi is not None and pi._recovery_check_needed:
            pi._recovery_check_needed = False
            if self._pi_recovery_unsub:
                self._pi_recovery_unsub()
            self._pi_recovery_unsub = async_call_later(
                self.hass, 60, self._on_pi_recovery
            )

    async def _on_pi_timer(self, now: Any = None) -> None:
        """PI timer tick — climate.py owns the timer, PI does computation."""
        if await self._controller.pi_tick(now):
            await self.send_ir()
        self._check_pi_recovery_needed()
        self.async_schedule_update_ha_state()

    async def _on_pi_recovery(self, _now: Any = None) -> None:
        """Sensor recovery check — 60s after sensor went unavailable."""
        self._pi_recovery_unsub = None
        from .pi_controller import PIController
        pi = self._pi if isinstance(self._pi, PIController) else None
        if pi is not None and await pi._check_sensor_recovery(_now):
            await self.send_ir()
        self.async_schedule_update_ha_state()

    async def _async_sensor_changed(
        self, entity_id_or_event: Event[EventStateChangedData],
        old_state: State | None = None, new_state: State | None = None
    ) -> None:
        # Replacing `async_track_state_change` with `async_track_state_change_event`
        entity_id = entity_id_or_event.data["entity_id"]
        old_state = entity_id_or_event.data["old_state"]
        new_state = entity_id_or_event.data["new_state"]

        if new_state is None:
            return

        if entity_id == self._temp_sensor:
            was_none = self._attr_current_temperature is None
            self._async_update_temp(new_state)
            if await self._controller.sensor_changed(was_none):
                await self.send_ir()
            # Cancel pending recovery if sensor came back
            if was_none and self._pi_recovery_unsub and self._attr_current_temperature is not None:
                self._pi_recovery_unsub()
                self._pi_recovery_unsub = None
            self._check_pi_recovery_needed()
            self.async_schedule_update_ha_state()
        elif entity_id == self._humidity_sensor:
            self._async_update_humidity(new_state)
            self.async_schedule_update_ha_state()
        elif entity_id == self._power_sensor:
            await self._async_power_sensor_changed(old_state, new_state)

    async def _async_power_sensor_changed(
        self, old_state: State | None, new_state: State | None, is_special_mode: bool = False
    ) -> None:
        """Handle power sensor changes."""
        if new_state is None:  # pragma: no cover — _async_sensor_changed already filters None
            return

        if old_state is not None and new_state.state == old_state.state:  # pragma: no cover — HA only fires on state change
            return

        if new_state.state == STATE_ON:
            if self._attr_hvac_mode == HVACMode.OFF or self.power_mode == STATE_OFF:
                self._attr_hvac_mode = (
                    self._special_mode  # type: ignore[assignment]
                    if self._special_mode and is_special_mode
                    else self._last_on_mode
                )
                self.power_mode = STATE_ON
                self.async_schedule_update_ha_state()

        elif new_state.state == STATE_OFF:
            if self._attr_hvac_mode != HVACMode.OFF or self.power_mode == STATE_ON:
                self._attr_hvac_mode = HVACMode.OFF
                self.power_mode = STATE_OFF
                self.async_schedule_update_ha_state()

    @callback
    def _async_update_temp(self, state: State) -> None:
        """Update thermostat with latest state from sensor."""
        try:
            self._attr_current_temperature = TemperatureConverter.convert(
                float(state.state),
                state.attributes["unit_of_measurement"],
                self.temperature_unit,
            )
        except ValueError as ex:
            _LOGGER.debug("Unable to update from sensor: %s", ex)

    @callback
    def _async_update_humidity(self, state: State) -> None:
        """Update thermostat with latest state from humidity sensor."""
        try:
            if state.state != STATE_UNKNOWN and state.state != STATE_UNAVAILABLE:
                self._attr_current_humidity = int(float(state.state))
        except ValueError as ex:
            _LOGGER.error("Unable to update from humidity sensor: %s", ex)

    @property
    def _is_device_active(self) -> bool:
        """If the toggleable device is currently active."""
        return self.power_mode == STATE_ON

    @cached_property
    def supported_features(self) -> ClimateEntityFeature:
        """Return the list of supported features."""
        return self._support_flags

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set new preset mode.

        This method must be run in the event loop and returns a coroutine.
        """
        # Handle IR action presets
        if hasattr(self, "_ir_action_presets") and preset_mode in self._ir_action_presets:
            action = self._ir_action_presets[preset_mode]
            await self._activate_ir_action_preset(preset_mode, action)
            return

        # Deactivate IR action preset if switching away from one
        if (
            hasattr(self, "_ir_action_presets")
            and self._attr_preset_mode in self._ir_action_presets
            and preset_mode != self._attr_preset_mode
        ):
            old_action = self._ir_action_presets[self._attr_preset_mode]
            # Send exit IR code if defined
            if old_action.get("exit_ir_code"):
                path = self.topic.split("/")
                irsend_topic = f"cmnd/{path[1]}/irsend"
                if float(self._mqtt_delay) != 0.0:
                    await asyncio.sleep(float(self._mqtt_delay))
                await mqtt.async_publish(self.hass, irsend_topic, old_action["exit_ir_code"])
            # Resume PI if it was paused
            if old_action.get("pause_pi"):
                self._controller.pi_resume()

        # Vendor-specific preset handling
        from .vendors.base import EntityState
        vendor_entity_state = EntityState(
            hvac_mode=self._attr_hvac_mode,
            target_temperature=self._attr_target_temperature,
            fan_mode=self._attr_fan_mode,
            swing_mode=self._attr_swing_mode,
            swingv=self._swingv,
            swingh=self._swingh,
            power_mode=self.power_mode,
        )
        vendor_result = await self._vendor_handler.handle_preset(
            preset_mode, vendor_entity_state, self._send_raw_ir,
        )
        if vendor_result is not None:
            # Handler fully owned this preset
            self._apply_vendor_handler_state()
            if vendor_result.timer_request:
                self._schedule_vendor_timer(vendor_result.timer_request)
            self.async_schedule_update_ha_state()
            return

        # PRESET_NONE from vendor handler clears its flags but falls through
        # to base logic for AWAY→NONE handling and send_ir
        self._apply_vendor_handler_state()

        if preset_mode == PRESET_AWAY and not self._is_away:
            self._is_away = True
            self._saved_target_temp = self.target_temperature  # property reads from controller when active
            self._attr_target_temperature = self._away_temp
            self._controller.desired_temp = self._away_temp  # type: ignore[assignment]  # guarded by preset list build
        elif preset_mode == PRESET_NONE and self._is_away:
            self._is_away = False
            self._attr_target_temperature = self._saved_target_temp
            self._controller.desired_temp = self._saved_target_temp  # type: ignore[assignment]  # was saved from target_temperature
        self._attr_preset_mode = PRESET_AWAY if self._is_away else PRESET_NONE
        await self.send_ir()

    async def _activate_ir_action_preset(self, preset_name, action):
        """Activate a user-defined IR action preset."""
        # Send the IR code
        path = self.topic.split("/")
        irsend_topic = f"cmnd/{path[1]}/irsend"
        if float(self._mqtt_delay) != 0.0:
            await asyncio.sleep(float(self._mqtt_delay))
        await mqtt.async_publish(self.hass, irsend_topic, action["ir_code"])

        self._attr_preset_mode = preset_name

        # Optionally pause PI
        if action.get("pause_pi"):
            self._controller.pi_pause()

        # Optionally auto-clear after timeout
        auto_clear = action.get("auto_clear_seconds")
        if auto_clear and auto_clear > 0:
            from homeassistant.helpers.event import async_call_later

            @callback
            def _clear_preset(_now):
                self._attr_preset_mode = PRESET_NONE
                if action.get("pause_pi"):
                    self._controller.pi_resume()
                self.async_schedule_update_ha_state()

            async_call_later(self.hass, auto_clear, _clear_preset)

        self.async_schedule_update_ha_state()
        _LOGGER.info("IR action preset '%s' activated", preset_name)

    # ── PI service delegations (called by SERVICE_TO_METHOD handler) ──

    async def async_reset_ff_seeds(self) -> None:
        """Reset feedforward RLS models to seed values."""
        await self._controller.async_reset_ff_seeds()
        self.async_schedule_update_ha_state()

    async def async_suppress_ff_learning(self, reason: str = "") -> None:
        """Manually suppress FF learning."""
        await self._controller.async_suppress_ff_learning(reason=reason)

    async def async_resume_ff_learning(self) -> None:
        """Resume FF learning after manual suppression."""
        await self._controller.async_resume_ff_learning()

    async def set_mode(self, hvac_mode: str) -> None:
        """Set hvac mode."""
        hvac_mode = hvac_mode.lower()
        if hvac_mode not in self._attr_hvac_modes or hvac_mode == HVACMode.OFF:
            self._attr_hvac_mode = HVACMode.OFF
            self._enabled = False
            self.power_mode = STATE_OFF
        else:
            self._attr_hvac_mode = self._last_on_mode = hvac_mode  # type: ignore[assignment]
            self._enabled = True
            self.power_mode = STATE_ON

    def _get_ir_temp(self) -> float:
        """Return temperature for IR payload (in celsius_mode unit)."""
        if self._controller.is_active and self._attr_hvac_mode != HVACMode.OFF:
            return self._controller.get_ir_temp()
        # Convert from entity unit (system) to IR unit (celsius_mode)
        temp: float = self._attr_target_temperature or 0.0
        if self._ir_temp_unit != self._attr_temperature_unit:
            temp = TemperatureConverter.convert(
                temp, self._attr_temperature_unit, self._ir_temp_unit
            )
        prec = self._temp_precision or 1.0
        temp = round(temp / prec) * prec
        # Safety clamp: no residential HVAC accepts temps outside 0-50°C
        temp_c = (
            temp if self._ir_temp_unit == UnitOfTemperature.CELSIUS
            else TemperatureConverter.convert(temp, UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.CELSIUS)
        )
        if temp_c < 0 or temp_c > 50:
            _LOGGER.error("IR temp %.1f°C out of safe range, clamping", temp_c)
            temp_c = max(0, min(50, temp_c))
            temp = (
                temp_c if self._ir_temp_unit == UnitOfTemperature.CELSIUS
                else TemperatureConverter.convert(temp_c, UnitOfTemperature.CELSIUS, UnitOfTemperature.FAHRENHEIT)
            )
        return temp

    async def send_ir(self) -> None:
        """Send the payload to tasmota mqtt topic."""
        fan_speed = self._vendor_handler.remap_fan_to_ir(self.fan_mode or "")

        # Set the swing mode - default off
        self._swingv = STATE_OFF if self._fix_swingv is None else self._fix_swingv
        self._swingh = STATE_OFF if self._fix_swingh is None else self._fix_swingh

        if SWING_BOTH in (self._attr_swing_modes or []) or SWING_VERTICAL in (
            self._attr_swing_modes or []
        ):
            if (
                self._attr_swing_mode == SWING_BOTH
                or self._attr_swing_mode == SWING_VERTICAL
            ):
                self._swingv = SWING_AUTO

        if SWING_BOTH in (self._attr_swing_modes or []) or SWING_HORIZONTAL in (
            self._attr_swing_modes or []
        ):
            if (
                self._attr_swing_mode == SWING_BOTH
                or self._attr_swing_mode == SWING_HORIZONTAL
            ):
                self._swingh = SWING_AUTO

        _dt = dt_util.now()
        _min = _dt.hour * 60 + _dt.minute

        # Populate the payload
        payload_data = {
            "StateMode": self._state_mode,
            "Vendor": self._vendor,
            "Model": self._model,
            "Power": self.power_mode,
            "Mode": self._last_on_mode if self._keep_mode else self._attr_hvac_mode,
            "Celsius": "on" if self._ir_protocol_unit == "celsius" else "off",
            "Temp": self._get_ir_temp(),
            "FanSpeed": fan_speed,
            "SwingV": self._swingv,
            "SwingH": self._swingh,
            "Quiet": self._quiet,
            "Turbo": self._turbo,
            "Econo": self._econo,
            "Light": self._light,
            "Filter": self._filter,
            "Clean": self._clean,
            "Beep": self._beep,
            "Sleep": self._sleep,
            "Clock": int(_min),
            "Weekday": int(_dt.weekday()),
        }
        self._state_mode = DEFAULT_STATE_MODE
        for key in self._toggle_list:
            setattr(self, "_" + key.lower(), "off")

        # Snapshot expected state for echo classification
        self._has_sent_once = True
        self._expected_state = {
            "Power": payload_data["Power"],
            "Mode": payload_data["Mode"],
            "Temp": payload_data["Temp"],
            "FanSpeed": payload_data["FanSpeed"],
            "SwingV": payload_data["SwingV"],
            "SwingH": payload_data["SwingH"],
            "Quiet": payload_data["Quiet"],
            "Turbo": payload_data["Turbo"],
            "Econo": payload_data["Econo"],
            "Light": payload_data["Light"],
            "Filter": payload_data["Filter"],
            "Clean": payload_data["Clean"],
            "Beep": payload_data["Beep"],
            "Sleep": payload_data["Sleep"],
        }

        payload = json.dumps(payload_data)

        # Publish mqtt message
        if float(self._mqtt_delay) != float(DEFAULT_MQTT_DELAY):
            await asyncio.sleep(float(self._mqtt_delay))

        _LOGGER.debug(
            "%s send_ir: Temp=%s Power=%s Mode=%s topic=%s",
            self.entity_id, payload_data["Temp"], payload_data["Power"],
            payload_data["Mode"], self.topic,
        )

        await mqtt.async_publish(self.hass, self.topic, payload)

        # Update HA UI and State
        self.async_schedule_update_ha_state()
