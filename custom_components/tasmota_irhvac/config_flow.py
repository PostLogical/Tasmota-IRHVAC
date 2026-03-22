"""Config flow for Tasmota IRHVAC integration."""

import logging

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.climate.const import (
    FAN_AUTO,
    FAN_DIFFUSE,
    FAN_FOCUS,
    FAN_HIGH,
    FAN_LOW,
    FAN_MEDIUM,
    FAN_MIDDLE,
    FAN_OFF,
    FAN_ON,
    HVACMode,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)
from homeassistant.const import (
    CONF_NAME,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)

from .const import (
    CONF_AVAILABILITY_TOPIC,
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
    DEFAULT_IGNORE_OFF_TEMP,
    DEFAULT_MAX_TEMP,
    DEFAULT_MIN_TEMP,
    DEFAULT_MQTT_DELAY,
    DEFAULT_NAME,
    DEFAULT_PRECISION,
    DEFAULT_STATE_TOPIC,
    DEFAULT_TARGET_TEMP,
    DOMAIN,
    HVAC_FAN_AUTO,
    HVAC_FAN_AUTO_MAX,
    HVAC_FAN_MAX,
    HVAC_FAN_MAX_HIGH,
    HVAC_FAN_MEDIUM,
    HVAC_FAN_MIN,
    HVAC_MODE_AUTO_FAN,
    HVAC_MODE_FAN_AUTO,
    HVAC_MODES,
    TOGGLE_ALL_LIST,
)

_LOGGER = logging.getLogger(__name__)

# Keys stored in entry.data (connection/identity)
DATA_KEYS = {
    CONF_NAME,
    CONF_VENDOR,
    CONF_COMMAND_TOPIC,
    CONF_STATE_TOPIC,
    CONF_STATE_TOPIC_2,
    CONF_AVAILABILITY_TOPIC,
}

# All valid fan speed options
ALL_FAN_SPEEDS = [
    FAN_ON, FAN_OFF, FAN_AUTO, FAN_LOW, FAN_MEDIUM, FAN_HIGH,
    FAN_MIDDLE, FAN_FOCUS, FAN_DIFFUSE,
    HVAC_FAN_MIN, HVAC_FAN_MEDIUM, HVAC_FAN_MAX, HVAC_FAN_AUTO,
    HVAC_FAN_MAX_HIGH, HVAC_FAN_AUTO_MAX,
]

# Default fan speed list
DEFAULT_FAN_LIST = [HVAC_FAN_AUTO_MAX, HVAC_FAN_MAX_HIGH, HVAC_FAN_MEDIUM, HVAC_FAN_MIN]

# Default modes list
DEFAULT_MODES_LIST = [
    HVACMode.COOL, HVACMode.HEAT, HVACMode.DRY,
    HVAC_MODE_AUTO_FAN, HVAC_MODE_FAN_AUTO,
]

DEFAULT_SWING_LIST = [SWING_OFF, SWING_VERTICAL]


class TasmotaIrhvacConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Tasmota IRHVAC."""

    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self._user_input = {}

    async def async_step_user(self, user_input=None):
        """Step 1: Device Setup."""
        errors = {}

        if user_input is not None:
            # Normalize protocol -> vendor
            vendor = user_input.get(CONF_VENDOR, "")
            if not vendor:
                errors[CONF_VENDOR] = "vendor_required"
            else:
                self._user_input.update(user_input)
                return await self.async_step_climate()

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME, default=DEFAULT_NAME): TextSelector(),
                    vol.Required(CONF_VENDOR): TextSelector(),
                    vol.Required(
                        CONF_COMMAND_TOPIC, default="cmnd/your_device/irhvac"
                    ): TextSelector(),
                    vol.Required(
                        CONF_STATE_TOPIC, default="tele/your_device/RESULT"
                    ): TextSelector(),
                    vol.Optional(CONF_STATE_TOPIC_2): TextSelector(),
                    vol.Optional(CONF_AVAILABILITY_TOPIC): TextSelector(),
                    vol.Optional(
                        CONF_MQTT_DELAY, default=DEFAULT_MQTT_DELAY
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=30, step=0.1, mode=NumberSelectorMode.BOX
                        )
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_climate(self, user_input=None):
        """Step 2: Climate Settings."""
        if user_input is not None:
            self._user_input.update(user_input)
            return await self.async_step_advanced()

        return self.async_show_form(
            step_id="climate",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MIN_TEMP, default=DEFAULT_MIN_TEMP
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=50, step=1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_MAX_TEMP, default=DEFAULT_MAX_TEMP
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=50, step=1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_TARGET_TEMP, default=DEFAULT_TARGET_TEMP
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=50, step=1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_PRECISION, default=DEFAULT_PRECISION
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": str(PRECISION_TENTHS), "label": "0.1"},
                                {"value": str(PRECISION_HALVES), "label": "0.5"},
                                {"value": str(PRECISION_WHOLE), "label": "1"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_TEMP_STEP, default=PRECISION_WHOLE
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": str(PRECISION_HALVES), "label": "0.5"},
                                {"value": str(PRECISION_WHOLE), "label": "1"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_CELSIUS, default=DEFAULT_CONF_CELSIUS
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["on", "off"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(CONF_AWAY_TEMP): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=50, step=1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional(
                        CONF_IGNORE_OFF_TEMP, default=DEFAULT_IGNORE_OFF_TEMP
                    ): BooleanSelector(),
                    vol.Optional(
                        CONF_MODES_LIST, default=DEFAULT_MODES_LIST
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=HVAC_MODES,
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_FAN_LIST, default=DEFAULT_FAN_LIST
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=ALL_FAN_SPEEDS,
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_SWING_LIST, default=DEFAULT_SWING_LIST
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[SWING_OFF, SWING_VERTICAL, SWING_HORIZONTAL, SWING_BOTH],
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_INITIAL_OPERATION_MODE, default=HVACMode.OFF
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=HVAC_MODES,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_KEEP_MODE, default=DEFAULT_CONF_KEEP_MODE
                    ): BooleanSelector(),
                }
            ),
        )

    async def async_step_advanced(self, user_input=None):
        """Step 3: Advanced & Sensors."""
        if user_input is not None:
            self._user_input.update(user_input)
            return await self._create_entry()

        return self.async_show_form(
            step_id="advanced",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_QUIET, default=DEFAULT_CONF_QUIET
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "on"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_TURBO, default=DEFAULT_CONF_TURBO
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "on"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_ECONO, default=DEFAULT_CONF_ECONO
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "on"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_MODEL, default=DEFAULT_CONF_MODEL
                    ): TextSelector(),
                    vol.Optional(
                        CONF_LIGHT, default=DEFAULT_CONF_LIGHT
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "on"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_FILTER, default=DEFAULT_CONF_FILTER
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "on"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_CLEAN, default=DEFAULT_CONF_CLEAN
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "on"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_BEEP, default=DEFAULT_CONF_BEEP
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "on"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_SLEEP, default=DEFAULT_CONF_SLEEP
                    ): TextSelector(),
                    vol.Optional(CONF_SWINGV): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "auto", "highest", "high", "middle", "low", "lowest"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(CONF_SWINGH): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "auto", "left max", "left", "middle", "right", "right max", "wide"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_TOGGLE_LIST, default=[]
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=TOGGLE_ALL_LIST,
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_SPECIAL_MODE, default=""
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["", "auto", "cool", "dry", "fan_only", "heat", "off"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(CONF_TEMP_SENSOR): EntitySelector(
                        EntitySelectorConfig(domain="sensor")
                    ),
                    vol.Optional(CONF_HUMIDITY_SENSOR): EntitySelector(
                        EntitySelectorConfig(domain="sensor")
                    ),
                    vol.Optional(CONF_POWER_SENSOR): EntitySelector(
                        EntitySelectorConfig(domain=["binary_sensor", "sensor"])
                    ),
                }
            ),
        )

    async def _create_entry(self):
        """Create the config entry from accumulated user input."""
        vendor = self._user_input.get(CONF_VENDOR, "")
        topic = self._user_input.get(CONF_COMMAND_TOPIC, "")

        await self.async_set_unique_id(f"{vendor}_{topic}")
        self._abort_if_unique_id_configured()

        data = {k: v for k, v in self._user_input.items() if k in DATA_KEYS}
        options = {k: v for k, v in self._user_input.items() if k not in DATA_KEYS}

        return self.async_create_entry(
            title=data.get(CONF_NAME, DEFAULT_NAME),
            data=data,
            options=options,
        )

    async def async_step_import(self, import_data):
        """Handle YAML import."""
        # Normalize protocol -> vendor
        if CONF_PROTOCOL in import_data and CONF_VENDOR not in import_data:
            import_data[CONF_VENDOR] = import_data.pop(CONF_PROTOCOL)

        # Handle the old state_topic + "_2" key
        old_key = CONF_STATE_TOPIC + "_2"
        if old_key in import_data and CONF_STATE_TOPIC_2 not in import_data:
            import_data[CONF_STATE_TOPIC_2] = import_data.pop(old_key)

        vendor = import_data.get(CONF_VENDOR, "")
        topic = import_data.get(CONF_COMMAND_TOPIC, "")
        await self.async_set_unique_id(f"{vendor}_{topic}")
        self._abort_if_unique_id_configured()

        data = {k: v for k, v in import_data.items() if k in DATA_KEYS}
        options = {k: v for k, v in import_data.items() if k not in DATA_KEYS}

        return self.async_create_entry(
            title=data.get(CONF_NAME, DEFAULT_NAME),
            data=data,
            options=options,
        )

    async def async_step_reconfigure(self, user_input=None):
        """Handle reconfiguration of connection/identity settings."""
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        errors = {}

        if user_input is not None:
            vendor = user_input.get(CONF_VENDOR, "")
            if not vendor:
                errors[CONF_VENDOR] = "vendor_required"
            else:
                return self.async_update_reload_and_abort(
                    entry,
                    data={**entry.data, **user_input},
                )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_NAME, default=entry.data.get(CONF_NAME, DEFAULT_NAME)
                    ): TextSelector(),
                    vol.Required(
                        CONF_VENDOR, default=entry.data.get(CONF_VENDOR, "")
                    ): TextSelector(),
                    vol.Required(
                        CONF_COMMAND_TOPIC,
                        default=entry.data.get(CONF_COMMAND_TOPIC, ""),
                    ): TextSelector(),
                    vol.Required(
                        CONF_STATE_TOPIC,
                        default=entry.data.get(CONF_STATE_TOPIC, ""),
                    ): TextSelector(),
                    vol.Optional(
                        CONF_STATE_TOPIC_2,
                        description={"suggested_value": entry.data.get(CONF_STATE_TOPIC_2)},
                    ): TextSelector(),
                    vol.Optional(
                        CONF_AVAILABILITY_TOPIC,
                        description={"suggested_value": entry.data.get(CONF_AVAILABILITY_TOPIC)},
                    ): TextSelector(),
                }
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow handler."""
        return TasmotaIrhvacOptionsFlow(config_entry)


class TasmotaIrhvacOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Tasmota IRHVAC."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Show the options menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "mqtt",
                "temperature",
                "modes",
                "defaults",
                "sensors",
                "advanced_options",
            ],
        )

    async def async_step_mqtt(self, user_input=None):
        """MQTT options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self._config_entry.options, **user_input}
            )

        opts = self._config_entry.options
        return self.async_show_form(
            step_id="mqtt",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MQTT_DELAY,
                        default=opts.get(CONF_MQTT_DELAY, DEFAULT_MQTT_DELAY),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=30, step=0.1, mode=NumberSelectorMode.BOX
                        )
                    ),
                }
            ),
        )

    async def async_step_temperature(self, user_input=None):
        """Temperature options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self._config_entry.options, **user_input}
            )

        opts = self._config_entry.options
        return self.async_show_form(
            step_id="temperature",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MIN_TEMP, default=opts.get(CONF_MIN_TEMP, DEFAULT_MIN_TEMP)
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_MAX_TEMP, default=opts.get(CONF_MAX_TEMP, DEFAULT_MAX_TEMP)
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_TARGET_TEMP, default=opts.get(CONF_TARGET_TEMP, DEFAULT_TARGET_TEMP)
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_PRECISION, default=opts.get(CONF_PRECISION, DEFAULT_PRECISION)
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": str(PRECISION_TENTHS), "label": "0.1"},
                                {"value": str(PRECISION_HALVES), "label": "0.5"},
                                {"value": str(PRECISION_WHOLE), "label": "1"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_TEMP_STEP, default=opts.get(CONF_TEMP_STEP, PRECISION_WHOLE)
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                {"value": str(PRECISION_HALVES), "label": "0.5"},
                                {"value": str(PRECISION_WHOLE), "label": "1"},
                            ],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_CELSIUS, default=opts.get(CONF_CELSIUS, DEFAULT_CONF_CELSIUS)
                    ): SelectSelector(
                        SelectSelectorConfig(options=["on", "off"], mode=SelectSelectorMode.DROPDOWN)
                    ),
                    vol.Optional(
                        CONF_AWAY_TEMP,
                        description={"suggested_value": opts.get(CONF_AWAY_TEMP)},
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_IGNORE_OFF_TEMP,
                        default=opts.get(CONF_IGNORE_OFF_TEMP, DEFAULT_IGNORE_OFF_TEMP),
                    ): BooleanSelector(),
                }
            ),
        )

    async def async_step_modes(self, user_input=None):
        """Mode options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self._config_entry.options, **user_input}
            )

        opts = self._config_entry.options
        return self.async_show_form(
            step_id="modes",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_MODES_LIST, default=opts.get(CONF_MODES_LIST, DEFAULT_MODES_LIST)
                    ): SelectSelector(
                        SelectSelectorConfig(options=HVAC_MODES, multiple=True, mode=SelectSelectorMode.DROPDOWN)
                    ),
                    vol.Optional(
                        CONF_FAN_LIST, default=opts.get(CONF_FAN_LIST, DEFAULT_FAN_LIST)
                    ): SelectSelector(
                        SelectSelectorConfig(options=ALL_FAN_SPEEDS, multiple=True, mode=SelectSelectorMode.DROPDOWN)
                    ),
                    vol.Optional(
                        CONF_SWING_LIST, default=opts.get(CONF_SWING_LIST, DEFAULT_SWING_LIST)
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[SWING_OFF, SWING_VERTICAL, SWING_HORIZONTAL, SWING_BOTH],
                            multiple=True, mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_INITIAL_OPERATION_MODE,
                        default=opts.get(CONF_INITIAL_OPERATION_MODE, HVACMode.OFF),
                    ): SelectSelector(
                        SelectSelectorConfig(options=HVAC_MODES, mode=SelectSelectorMode.DROPDOWN)
                    ),
                    vol.Optional(
                        CONF_KEEP_MODE,
                        default=opts.get(CONF_KEEP_MODE, DEFAULT_CONF_KEEP_MODE),
                    ): BooleanSelector(),
                }
            ),
        )

    async def async_step_defaults(self, user_input=None):
        """Default values options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self._config_entry.options, **user_input}
            )

        opts = self._config_entry.options
        on_off = SelectSelectorConfig(options=["off", "on"], mode=SelectSelectorMode.DROPDOWN)
        return self.async_show_form(
            step_id="defaults",
            data_schema=vol.Schema(
                {
                    vol.Optional(CONF_QUIET, default=opts.get(CONF_QUIET, DEFAULT_CONF_QUIET)): SelectSelector(on_off),
                    vol.Optional(CONF_TURBO, default=opts.get(CONF_TURBO, DEFAULT_CONF_TURBO)): SelectSelector(on_off),
                    vol.Optional(CONF_ECONO, default=opts.get(CONF_ECONO, DEFAULT_CONF_ECONO)): SelectSelector(on_off),
                    vol.Optional(CONF_MODEL, default=opts.get(CONF_MODEL, DEFAULT_CONF_MODEL)): TextSelector(),
                    vol.Optional(CONF_LIGHT, default=opts.get(CONF_LIGHT, DEFAULT_CONF_LIGHT)): SelectSelector(on_off),
                    vol.Optional(CONF_FILTER, default=opts.get(CONF_FILTER, DEFAULT_CONF_FILTER)): SelectSelector(on_off),
                    vol.Optional(CONF_CLEAN, default=opts.get(CONF_CLEAN, DEFAULT_CONF_CLEAN)): SelectSelector(on_off),
                    vol.Optional(CONF_BEEP, default=opts.get(CONF_BEEP, DEFAULT_CONF_BEEP)): SelectSelector(on_off),
                    vol.Optional(CONF_SLEEP, default=opts.get(CONF_SLEEP, DEFAULT_CONF_SLEEP)): TextSelector(),
                    vol.Optional(
                        CONF_SWINGV,
                        description={"suggested_value": opts.get(CONF_SWINGV)},
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "auto", "highest", "high", "middle", "low", "lowest"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_SWINGH,
                        description={"suggested_value": opts.get(CONF_SWINGH)},
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["off", "auto", "left max", "left", "middle", "right", "right max", "wide"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    async def async_step_sensors(self, user_input=None):
        """Sensor entity options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self._config_entry.options, **user_input}
            )

        opts = self._config_entry.options
        return self.async_show_form(
            step_id="sensors",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TEMP_SENSOR,
                        description={"suggested_value": opts.get(CONF_TEMP_SENSOR)},
                    ): EntitySelector(EntitySelectorConfig(domain="sensor")),
                    vol.Optional(
                        CONF_HUMIDITY_SENSOR,
                        description={"suggested_value": opts.get(CONF_HUMIDITY_SENSOR)},
                    ): EntitySelector(EntitySelectorConfig(domain="sensor")),
                    vol.Optional(
                        CONF_POWER_SENSOR,
                        description={"suggested_value": opts.get(CONF_POWER_SENSOR)},
                    ): EntitySelector(
                        EntitySelectorConfig(domain=["binary_sensor", "sensor"])
                    ),
                }
            ),
        )

    async def async_step_advanced_options(self, user_input=None):
        """Advanced options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self._config_entry.options, **user_input}
            )

        opts = self._config_entry.options
        return self.async_show_form(
            step_id="advanced_options",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TOGGLE_LIST, default=opts.get(CONF_TOGGLE_LIST, [])
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=TOGGLE_ALL_LIST,
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Optional(
                        CONF_SPECIAL_MODE, default=opts.get(CONF_SPECIAL_MODE, "")
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=["", "auto", "cool", "dry", "fan_only", "heat", "off"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )
