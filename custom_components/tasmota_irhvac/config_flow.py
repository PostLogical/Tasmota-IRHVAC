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
from homeassistant.config_entries import OptionsFlowWithReload
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
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PI_DEADBAND,
    CONF_PI_ENABLED,
    CONF_PI_FF_COOL_REFERENCE,
    CONF_PI_FF_COOL_SLOPE,
    CONF_PI_FF_HEAT_REFERENCE,
    CONF_PI_FF_HEAT_SLOPE,
    CONF_PI_DISTURBANCE_INPUTS,
    CONF_PI_FF_SUPPRESS_LEARNING_ENTITY,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_MIN_INTERVAL,
    CONF_PI_SETPOINT_WEIGHT,
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
    CONF_HAS_SET_H,
    CONF_HAS_SET_V,
    CONF_IR_ACTIONS,
    CONF_PRESET_MODES_LIST,
    PRESET_ECONO,
    PRESET_MIN_HEAT,
    PRESET_POWERFUL,
    PRESET_SET_H,
    PRESET_SET_V,
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
    HVACMode.OFF, HVACMode.HEAT, HVACMode.COOL, HVACMode.DRY,
    HVACMode.FAN_ONLY, HVACMode.AUTO,
]

DEFAULT_SWING_LIST = [SWING_OFF, SWING_VERTICAL]

# Keys whose SelectSelector values need coercion from str to float
_FLOAT_KEYS = (CONF_PRECISION, CONF_TEMP_STEP)

# Reusable selector configs
_ON_OFF_SELECTOR = SelectSelectorConfig(
    options=["off", "on"], mode=SelectSelectorMode.DROPDOWN
)

_PRECISION_SELECTOR = SelectSelector(
    SelectSelectorConfig(
        options=[
            {"value": str(PRECISION_TENTHS), "label": "0.1"},
            {"value": str(PRECISION_HALVES), "label": "0.5"},
            {"value": str(PRECISION_WHOLE), "label": "1"},
        ],
        mode=SelectSelectorMode.DROPDOWN,
    )
)

_TEMP_STEP_SELECTOR = SelectSelector(
    SelectSelectorConfig(
        options=[
            {"value": str(PRECISION_HALVES), "label": "0.5"},
            {"value": str(PRECISION_WHOLE), "label": "1"},
            {"value": "2.0", "label": "2"},
        ],
        mode=SelectSelectorMode.DROPDOWN,
    )
)


def _stringify_floats(data: dict) -> dict:
    """Ensure SelectSelector numeric keys are stored as strings for UI consistency."""
    for key in _FLOAT_KEYS:
        if key in data and not isinstance(data[key], str):
            data[key] = str(data[key])
    return data


# ---------------------------------------------------------------------------
# Options flow schemas (defined once, populated via add_suggested_values_to_schema)
# ---------------------------------------------------------------------------

OPTIONS_MQTT_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_MQTT_DELAY): NumberSelector(
            NumberSelectorConfig(min=0, max=30, step=0.1, mode=NumberSelectorMode.BOX)
        ),
    }
)

OPTIONS_TEMPERATURE_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_MIN_TEMP): NumberSelector(
            NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_MAX_TEMP): NumberSelector(
            NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_TARGET_TEMP): NumberSelector(
            NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PRECISION): _PRECISION_SELECTOR,
        vol.Optional(CONF_TEMP_STEP): _TEMP_STEP_SELECTOR,
        vol.Optional(CONF_CELSIUS): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_AWAY_TEMP): NumberSelector(
            NumberSelectorConfig(min=0, max=50, step=1, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_IGNORE_OFF_TEMP): BooleanSelector(),
    }
)

OPTIONS_MODES_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_MODES_LIST): SelectSelector(
            SelectSelectorConfig(
                options=HVAC_MODES, multiple=True, mode=SelectSelectorMode.DROPDOWN
            )
        ),
        vol.Optional(CONF_FAN_LIST): SelectSelector(
            SelectSelectorConfig(
                options=ALL_FAN_SPEEDS, multiple=True, mode=SelectSelectorMode.DROPDOWN
            )
        ),
        vol.Optional(CONF_SWING_LIST): SelectSelector(
            SelectSelectorConfig(
                options=[SWING_OFF, SWING_VERTICAL, SWING_HORIZONTAL, SWING_BOTH],
                multiple=True,
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(CONF_INITIAL_OPERATION_MODE): SelectSelector(
            SelectSelectorConfig(
                options=HVAC_MODES, mode=SelectSelectorMode.DROPDOWN
            )
        ),
        vol.Optional(CONF_KEEP_MODE): BooleanSelector(),
    }
)

OPTIONS_DEFAULTS_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_QUIET): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_TURBO): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_ECONO): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_MODEL): TextSelector(),
        vol.Optional(CONF_LIGHT): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_FILTER): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_CLEAN): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_BEEP): SelectSelector(_ON_OFF_SELECTOR),
        vol.Optional(CONF_SLEEP): TextSelector(),
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
    }
)

OPTIONS_SENSORS_SCHEMA = vol.Schema(
    {
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
)

OPTIONS_ADVANCED_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_TOGGLE_LIST): SelectSelector(
            SelectSelectorConfig(
                options=TOGGLE_ALL_LIST,
                multiple=True,
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(CONF_SPECIAL_MODE): SelectSelector(
            SelectSelectorConfig(
                options=["", "auto", "cool", "dry", "fan_only", "heat", "off"],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
    }
)

OPTIONS_PI_CONTROLLER_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_PI_ENABLED, default=DEFAULT_PI_ENABLED): BooleanSelector(),
        vol.Optional(CONF_PI_KP, default=DEFAULT_PI_KP): NumberSelector(
            NumberSelectorConfig(min=0, max=20, step=0.1, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PI_KI, default=DEFAULT_PI_KI): NumberSelector(
            NumberSelectorConfig(min=0, max=5, step=0.01, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PI_MIN_INTERVAL, default=DEFAULT_PI_MIN_INTERVAL): NumberSelector(
            NumberSelectorConfig(min=60, max=3600, step=60, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PI_DEADBAND, default=DEFAULT_PI_DEADBAND): NumberSelector(
            NumberSelectorConfig(min=0, max=5, step=0.1, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_OUTDOOR_TEMP_SENSOR): EntitySelector(
            EntitySelectorConfig(domain="sensor")
        ),
        vol.Optional(CONF_PI_FF_HEAT_REFERENCE, default=DEFAULT_PI_FF_HEAT_REFERENCE): NumberSelector(
            NumberSelectorConfig(min=-20, max=50, step=0.5, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PI_FF_HEAT_SLOPE, default=DEFAULT_PI_FF_HEAT_SLOPE): NumberSelector(
            NumberSelectorConfig(min=0, max=5, step=0.01, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PI_FF_COOL_REFERENCE, default=DEFAULT_PI_FF_COOL_REFERENCE): NumberSelector(
            NumberSelectorConfig(min=0, max=60, step=0.5, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PI_FF_COOL_SLOPE, default=DEFAULT_PI_FF_COOL_SLOPE): NumberSelector(
            NumberSelectorConfig(min=0, max=5, step=0.01, mode=NumberSelectorMode.BOX)
        ),
        vol.Optional(CONF_PI_SETPOINT_WEIGHT, default=DEFAULT_PI_SETPOINT_WEIGHT): NumberSelector(
            NumberSelectorConfig(min=0, max=1, step=0.05, mode=NumberSelectorMode.BOX)
        ),
    }
)


# ---------------------------------------------------------------------------
# Config flow
# ---------------------------------------------------------------------------

class TasmotaIrhvacConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Tasmota IRHVAC."""

    VERSION = 1
    MINOR_VERSION = 3  # Disturbance inputs (replaces suppress/bias entities)

    def __init__(self):
        """Initialize the config flow."""
        self._user_input = {}

    async def async_step_import(self, import_data=None):
        """Handle YAML import. Check for duplicates before creating entry."""
        if import_data is None:
            return self.async_abort(reason="already_configured")

        # Check if an entry with the same command topic already exists
        command_topic = import_data.get(CONF_COMMAND_TOPIC, "")
        for entry in self._async_current_entries():
            if entry.data.get(CONF_COMMAND_TOPIC) == command_topic:
                return self.async_abort(reason="already_configured")

        # Skip the UI steps — create entry directly from YAML data
        name = import_data.get(CONF_NAME, DEFAULT_NAME)
        return self.async_create_entry(title=name, data=import_data)

    async def async_step_user(self, user_input=None):
        """Step 1: Device Setup."""
        errors = {}

        if user_input is not None:
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
                        CONF_PRECISION, default=str(DEFAULT_PRECISION)
                    ): _PRECISION_SELECTOR,
                    vol.Optional(
                        CONF_TEMP_STEP, default=str(PRECISION_WHOLE)
                    ): _TEMP_STEP_SELECTOR,
                    vol.Optional(
                        CONF_CELSIUS, default=DEFAULT_CONF_CELSIUS
                    ): SelectSelector(_ON_OFF_SELECTOR),
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
                        CONF_PRESET_MODES_LIST
                    ): SelectSelector(
                        SelectSelectorConfig(
                            options=[
                                PRESET_POWERFUL,
                                PRESET_ECONO,
                                PRESET_MIN_HEAT,
                            ],
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                            custom_value=True,
                        )
                    ),
                    vol.Optional(
                        CONF_KEEP_MODE, default=DEFAULT_CONF_KEEP_MODE
                    ): BooleanSelector(),
                }
            ),
        )

    def _vendor_is_fujitsu(self):
        """Check if the configured vendor is a Fujitsu model."""
        vendor = self._user_input.get(CONF_VENDOR, "")
        return vendor.upper().startswith("FUJITSU")

    async def async_step_advanced(self, user_input=None):
        """Step 3: Advanced & Sensors."""
        if user_input is not None:
            self._user_input.update(user_input)
            if user_input.get(CONF_PI_ENABLED):
                return await self.async_step_pi_controller()
            return await self._create_entry()

        return self.async_show_form(
            step_id="advanced",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_QUIET, default=DEFAULT_CONF_QUIET
                    ): SelectSelector(_ON_OFF_SELECTOR),
                    vol.Optional(
                        CONF_TURBO, default=DEFAULT_CONF_TURBO
                    ): SelectSelector(_ON_OFF_SELECTOR),
                    vol.Optional(
                        CONF_ECONO, default=DEFAULT_CONF_ECONO
                    ): SelectSelector(_ON_OFF_SELECTOR),
                    vol.Optional(
                        CONF_MODEL, default=DEFAULT_CONF_MODEL
                    ): TextSelector(),
                    vol.Optional(
                        CONF_LIGHT, default=DEFAULT_CONF_LIGHT
                    ): SelectSelector(_ON_OFF_SELECTOR),
                    vol.Optional(
                        CONF_FILTER, default=DEFAULT_CONF_FILTER
                    ): SelectSelector(_ON_OFF_SELECTOR),
                    vol.Optional(
                        CONF_CLEAN, default=DEFAULT_CONF_CLEAN
                    ): SelectSelector(_ON_OFF_SELECTOR),
                    vol.Optional(
                        CONF_BEEP, default=DEFAULT_CONF_BEEP
                    ): SelectSelector(_ON_OFF_SELECTOR),
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
                    vol.Optional(
                        CONF_PI_ENABLED, default=DEFAULT_PI_ENABLED
                    ): BooleanSelector(),
                    vol.Optional(
                        CONF_HAS_SET_V, default=False
                    ): BooleanSelector(),
                    vol.Optional(
                        CONF_HAS_SET_H, default=False
                    ): BooleanSelector(),
                }
            ),
        )

    async def async_step_pi_controller(self, user_input=None):
        """Step 4: PI Controller settings (shown when PI is enabled)."""
        if user_input is not None:
            self._user_input.update(user_input)
            return await self._create_entry()

        return self.async_show_form(
            step_id="pi_controller",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_PI_KP, default=DEFAULT_PI_KP
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=20, step=0.1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_PI_KI, default=DEFAULT_PI_KI
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=5, step=0.01, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_PI_MIN_INTERVAL, default=DEFAULT_PI_MIN_INTERVAL
                    ): NumberSelector(
                        NumberSelectorConfig(min=60, max=3600, step=60, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_PI_DEADBAND, default=DEFAULT_PI_DEADBAND
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=5, step=0.1, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(CONF_OUTDOOR_TEMP_SENSOR): EntitySelector(
                        EntitySelectorConfig(domain="sensor")
                    ),
                    vol.Optional(
                        CONF_PI_FF_HEAT_REFERENCE, default=DEFAULT_PI_FF_HEAT_REFERENCE
                    ): NumberSelector(
                        NumberSelectorConfig(min=-20, max=50, step=0.5, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_PI_FF_HEAT_SLOPE, default=DEFAULT_PI_FF_HEAT_SLOPE
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=5, step=0.01, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_PI_FF_COOL_REFERENCE, default=DEFAULT_PI_FF_COOL_REFERENCE
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=60, step=0.5, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(
                        CONF_PI_FF_COOL_SLOPE, default=DEFAULT_PI_FF_COOL_SLOPE
                    ): NumberSelector(
                        NumberSelectorConfig(min=0, max=5, step=0.01, mode=NumberSelectorMode.BOX)
                    ),
                    vol.Optional(CONF_PI_SETPOINT_WEIGHT, default=DEFAULT_PI_SETPOINT_WEIGHT): NumberSelector(
                        NumberSelectorConfig(min=0, max=1, step=0.05, mode=NumberSelectorMode.BOX)
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
        options = _stringify_floats(
            {k: v for k, v in self._user_input.items() if k not in DATA_KEYS}
        )

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

        # Migrate legacy suppress/bias entities → disturbance_inputs
        disturbance_inputs = import_data.get(CONF_PI_DISTURBANCE_INPUTS, [])
        if not disturbance_inputs:
            old_suppress = import_data.pop("pi_ff_suppress_learning_entity", "")
            old_bias = import_data.pop("pi_ff_bias_entity", "")
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
            if disturbance_inputs:
                import_data[CONF_PI_DISTURBANCE_INPUTS] = disturbance_inputs
        else:
            # Clean up old keys if disturbance_inputs already present
            import_data.pop("pi_ff_suppress_learning_entity", None)
            import_data.pop("pi_ff_bias_entity", None)

        vendor = import_data.get(CONF_VENDOR, "")
        topic = import_data.get(CONF_COMMAND_TOPIC, "")
        await self.async_set_unique_id(f"{vendor}_{topic}")
        self._abort_if_unique_id_configured()

        data = {k: v for k, v in import_data.items() if k in DATA_KEYS}
        options = _stringify_floats(
            {k: v for k, v in import_data.items() if k not in DATA_KEYS}
        )

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
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required(CONF_NAME): TextSelector(),
                        vol.Required(CONF_VENDOR): TextSelector(),
                        vol.Required(CONF_COMMAND_TOPIC): TextSelector(),
                        vol.Required(CONF_STATE_TOPIC): TextSelector(),
                        vol.Optional(CONF_STATE_TOPIC_2): TextSelector(),
                        vol.Optional(CONF_AVAILABILITY_TOPIC): TextSelector(),
                    }
                ),
                entry.data,
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow handler."""
        return TasmotaIrhvacOptionsFlow()


# ---------------------------------------------------------------------------
# Options flow
# ---------------------------------------------------------------------------

class TasmotaIrhvacOptionsFlow(OptionsFlowWithReload):
    """Handle options flow for Tasmota IRHVAC."""

    def _vendor_is_fujitsu(self):
        """Check if the configured vendor is a Fujitsu model."""
        vendor = self.config_entry.data.get(CONF_VENDOR, "")
        return vendor.upper().startswith("FUJITSU")

    async def async_step_init(self, user_input=None):
        """Show the options menu."""
        menu = [
            "mqtt",
            "temperature",
            "modes",
            "defaults",
            "sensors",
            "advanced_options",
            "pi_controller",
            "disturbance_inputs",
            "ir_actions",
        ]
        return self.async_show_menu(
            step_id="init",
            menu_options=menu,
        )

    async def async_step_mqtt(self, user_input=None):
        """MQTT options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self.config_entry.options, **user_input}
            )

        return self.async_show_form(
            step_id="mqtt",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_MQTT_SCHEMA, self.config_entry.options
            ),
        )

    async def async_step_temperature(self, user_input=None):
        """Temperature options."""
        if user_input is not None:
            return self.async_create_entry(
                data=_stringify_floats({**self.config_entry.options, **user_input})
            )

        return self.async_show_form(
            step_id="temperature",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_TEMPERATURE_SCHEMA, self.config_entry.options
            ),
        )

    async def async_step_modes(self, user_input=None):
        """Mode options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self.config_entry.options, **user_input}
            )

        return self.async_show_form(
            step_id="modes",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_MODES_SCHEMA, self.config_entry.options
            ),
        )

    async def async_step_defaults(self, user_input=None):
        """Default values options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self.config_entry.options, **user_input}
            )

        return self.async_show_form(
            step_id="defaults",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_DEFAULTS_SCHEMA, self.config_entry.options
            ),
        )

    async def async_step_sensors(self, user_input=None):
        """Sensor entity options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self.config_entry.options, **user_input}
            )

        return self.async_show_form(
            step_id="sensors",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_SENSORS_SCHEMA, self.config_entry.options
            ),
        )

    async def async_step_advanced_options(self, user_input=None):
        """Advanced options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self.config_entry.options, **user_input}
            )

        return self.async_show_form(
            step_id="advanced_options",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_ADVANCED_SCHEMA, self.config_entry.options
            ),
        )

    async def async_step_pi_controller(self, user_input=None):
        """PI controller options."""
        if user_input is not None:
            return self.async_create_entry(
                data={**self.config_entry.options, **user_input}
            )

        return self.async_show_form(
            step_id="pi_controller",
            data_schema=self.add_suggested_values_to_schema(
                OPTIONS_PI_CONTROLLER_SCHEMA, self.config_entry.options
            ),
        )

    # ── Disturbance Inputs ─────────────────────────────────────────────

    async def async_step_disturbance_inputs(self, user_input=None):
        """Disturbance inputs management menu."""
        inputs = self.config_entry.options.get(CONF_PI_DISTURBANCE_INPUTS, [])
        menu = ["disturbance_inputs_add"]
        if inputs:
            menu.append("disturbance_inputs_remove")
        return self.async_show_menu(
            step_id="disturbance_inputs",
            menu_options=menu,
            description_placeholders={
                "count": str(len(inputs)),
                "inputs": ", ".join(d["name"] for d in inputs) if inputs else "none",
            },
        )

    async def async_step_disturbance_inputs_add(self, user_input=None):
        """Add a new disturbance input."""
        if user_input is not None:
            inputs = list(self.config_entry.options.get(CONF_PI_DISTURBANCE_INPUTS, []))
            new_input = {
                "name": user_input["disturbance_name"],
                "entity_id": user_input["disturbance_entity"],
                "suppress_learning": user_input.get("disturbance_suppress", False),
                "default_bias": float(user_input.get("disturbance_default_bias", 0.0)),
                "gain": float(user_input.get("disturbance_gain", 1.0)),
            }
            inputs.append(new_input)
            return self.async_create_entry(
                data={**self.config_entry.options, CONF_PI_DISTURBANCE_INPUTS: inputs}
            )

        return self.async_show_form(
            step_id="disturbance_inputs_add",
            data_schema=vol.Schema(
                {
                    vol.Required("disturbance_name"): TextSelector(),
                    vol.Required("disturbance_entity"): EntitySelector(
                        EntitySelectorConfig()
                    ),
                    vol.Optional("disturbance_suppress", default=True): BooleanSelector(),
                    vol.Optional("disturbance_default_bias", default=0.0): NumberSelector(
                        NumberSelectorConfig(
                            min=-20, max=20, step=0.1, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional("disturbance_gain", default=1.0): NumberSelector(
                        NumberSelectorConfig(
                            min=-10, max=10, step=0.1, mode=NumberSelectorMode.BOX
                        )
                    ),
                }
            ),
        )

    async def async_step_disturbance_inputs_remove(self, user_input=None):
        """Remove disturbance inputs."""
        inputs = list(self.config_entry.options.get(CONF_PI_DISTURBANCE_INPUTS, []))
        if user_input is not None:
            names_to_remove = set(user_input.get("disturbance_inputs_to_remove", []))
            inputs = [d for d in inputs if d["name"] not in names_to_remove]
            return self.async_create_entry(
                data={**self.config_entry.options, CONF_PI_DISTURBANCE_INPUTS: inputs}
            )

        input_names = [d["name"] for d in inputs]
        if not input_names:
            return await self.async_step_disturbance_inputs()

        return self.async_show_form(
            step_id="disturbance_inputs_remove",
            data_schema=vol.Schema(
                {
                    vol.Required("disturbance_inputs_to_remove"): SelectSelector(
                        SelectSelectorConfig(
                            options=input_names,
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )

    # ── IR Actions ────────────────────────────────────────────────────

    async def async_step_ir_actions(self, user_input=None):
        """IR actions management menu."""
        actions = self.config_entry.options.get(CONF_IR_ACTIONS, [])
        menu = ["ir_actions_add"]
        if actions:
            menu.append("ir_actions_remove")
        return self.async_show_menu(
            step_id="ir_actions",
            menu_options=menu,
            description_placeholders={
                "count": str(len(actions)),
                "actions": ", ".join(a["name"] for a in actions) if actions else "none",
            },
        )

    async def async_step_ir_actions_add(self, user_input=None):
        """Add a new IR action."""
        if user_input is not None:
            actions = list(self.config_entry.options.get(CONF_IR_ACTIONS, []))
            new_action = {
                "name": user_input["ir_action_name"],
                "type": user_input["ir_action_type"],
                "ir_code": user_input["ir_action_code"],
            }
            # Optional fields for presets
            if user_input.get("ir_action_exit_code"):
                new_action["exit_ir_code"] = user_input["ir_action_exit_code"]
            if user_input.get("ir_action_auto_clear"):
                new_action["auto_clear_seconds"] = int(user_input["ir_action_auto_clear"])
            if user_input.get("ir_action_pause_pi"):
                new_action["pause_pi"] = True

            actions.append(new_action)
            return self.async_create_entry(
                data={**self.config_entry.options, CONF_IR_ACTIONS: actions}
            )

        return self.async_show_form(
            step_id="ir_actions_add",
            data_schema=vol.Schema(
                {
                    vol.Required("ir_action_name"): TextSelector(),
                    vol.Required("ir_action_type", default="button"): SelectSelector(
                        SelectSelectorConfig(
                            options=["button", "preset"],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                    vol.Required("ir_action_code"): TextSelector(),
                    vol.Optional("ir_action_exit_code"): TextSelector(),
                    vol.Optional("ir_action_auto_clear"): NumberSelector(
                        NumberSelectorConfig(
                            min=0, max=7200, step=60, mode=NumberSelectorMode.BOX
                        )
                    ),
                    vol.Optional("ir_action_pause_pi", default=False): BooleanSelector(),
                }
            ),
        )

    async def async_step_ir_actions_remove(self, user_input=None):
        """Remove IR actions."""
        actions = list(self.config_entry.options.get(CONF_IR_ACTIONS, []))
        if user_input is not None:
            names_to_remove = set(user_input.get("ir_actions_to_remove", []))
            actions = [a for a in actions if a["name"] not in names_to_remove]
            return self.async_create_entry(
                data={**self.config_entry.options, CONF_IR_ACTIONS: actions}
            )

        action_names = [a["name"] for a in actions]
        if not action_names:
            return await self.async_step_ir_actions()

        return self.async_show_form(
            step_id="ir_actions_remove",
            data_schema=vol.Schema(
                {
                    vol.Required("ir_actions_to_remove"): SelectSelector(
                        SelectSelectorConfig(
                            options=action_names,
                            multiple=True,
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    ),
                }
            ),
        )
