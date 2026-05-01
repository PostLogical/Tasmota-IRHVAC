"""Provides the constants needed for component."""

from homeassistant.components.climate.const import FAN_MEDIUM, HVACMode

# Swing oscillation value sent to/from Tasmota (not an HA swing mode)
SWING_AUTO = "auto"

# Tasmota fan speed values (not all map to HA FAN_* constants)
# HA provides: FAN_AUTO="auto", FAN_LOW="low", FAN_MEDIUM="medium", FAN_HIGH="high"
# Tasmota uses: "auto", "min", "medium", "max" — "min" and "max" have no HA equivalent
HVAC_FAN_MIN = "min"
HVAC_FAN_MAX = "max"

# Vendor mode swaps: some devices interchange auto/fan_only
HVAC_MODE_AUTO_FAN = "auto_fan_only"
HVAC_MODE_FAN_AUTO = "fan_only_auto"

# ELECTRA_AC fan speed labels (remapped to standard names by ElectraHandler)
HVAC_FAN_MAX_HIGH = "max_high"
HVAC_FAN_AUTO_MAX = "auto_max"

# Hvac moed list
HVAC_MODES = [
    HVACMode.OFF,
    HVACMode.HEAT,
    HVACMode.COOL,
    HVACMode.HEAT_COOL,
    HVACMode.AUTO,
    HVACMode.DRY,
    HVACMode.FAN_ONLY,
    HVAC_MODE_AUTO_FAN,
    HVAC_MODE_FAN_AUTO,
]

# Platform specific config entry names
CONF_EXCLUSIVE_GROUP_VENDOR = "exclusive_group_vendor"
CONF_VENDOR = "vendor"
CONF_PROTOCOL = "protocol"  # Soon to be deprecated
CONF_COMMAND_TOPIC = "command_topic"
CONF_STATE_TOPIC = "state_topic"
CONF_AVAILABILITY_TOPIC = "availability_topic"
CONF_TEMP_SENSOR = "temperature_sensor"
CONF_HUMIDITY_SENSOR = "humidity_sensor"
CONF_POWER_SENSOR = "power_sensor"
CONF_MQTT_DELAY = "mqtt_delay"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_TARGET_TEMP = "target_temp"
CONF_INITIAL_OPERATION_MODE = "initial_operation_mode"
CONF_AWAY_TEMP = "away_temp"
CONF_PRECISION = "precision"
CONF_TEMP_STEP = "temp_step"
CONF_MODES_LIST = "supported_modes"
CONF_FAN_LIST = "supported_fan_speeds"
CONF_SWING_LIST = "supported_swing_list"
CONF_QUIET = "default_quiet_mode"
CONF_TURBO = "default_turbo_mode"
CONF_ECONO = "default_econo_mode"
CONF_MODEL = "hvac_model"
CONF_CELSIUS = "celsius_mode"  # Legacy key — use CONF_IR_PROTOCOL_UNIT
CONF_IR_PROTOCOL_UNIT = "ir_protocol_unit"
CONF_LIGHT = "default_light_mode"
CONF_FILTER = "default_filter_mode"
CONF_CLEAN = "default_clean_mode"
CONF_BEEP = "default_beep_mode"
CONF_SLEEP = "default_sleep_mode"
CONF_KEEP_MODE = "keep_mode_when_off"
CONF_SWINGV = "default_swingv"
CONF_SWINGH = "default_swingh"
CONF_TOGGLE_LIST = "toggle_list"
CONF_IGNORE_OFF_TEMP = "ignore_off_temp"
CONF_SPECIAL_MODE = "special_mode"

# Platform specific default values
DEFAULT_NAME = "IR AirConditioner"
DEFAULT_STATE_TOPIC = "state"
DEFAULT_COMMAND_TOPIC = "topic"
DEFAULT_MQTT_DELAY = 0
DEFAULT_TARGET_TEMP = 26
DEFAULT_MIN_TEMP = 16
DEFAULT_MAX_TEMP = 32
DEFAULT_PRECISION = 1
DEFAULT_FAN_LIST = [HVAC_FAN_AUTO_MAX, HVAC_FAN_MAX_HIGH, FAN_MEDIUM, HVAC_FAN_MIN]
DEFAULT_CONF_QUIET = "off"
DEFAULT_CONF_TURBO = "off"
DEFAULT_CONF_ECONO = "off"
DEFAULT_CONF_MODEL = "-1"
DEFAULT_CONF_CELSIUS = "on"  # Legacy — use DEFAULT_IR_PROTOCOL_UNIT
DEFAULT_IR_PROTOCOL_UNIT = "celsius"
DEFAULT_CONF_LIGHT = "off"
DEFAULT_CONF_FILTER = "off"
DEFAULT_CONF_CLEAN = "off"
DEFAULT_CONF_BEEP = "off"
DEFAULT_CONF_SLEEP = "-1"
DEFAULT_CONF_KEEP_MODE = False
DEFAULT_STATE_MODE = "SendStore"
DEFAULT_IGNORE_OFF_TEMP = False

ATTR_NAME = "name"
ATTR_VALUE = "value"

CONF_STATE_TOPIC_2 = "state_topic_2"

DATA_KEY = "tasmota_irhvac.climate"
DOMAIN = "tasmota_irhvac"
PLATFORMS = ["climate", "sensor", "button", "binary_sensor"]

ATTR_ECONO = "econo"
ATTR_TURBO = "turbo"
ATTR_QUIET = "quiet"
ATTR_LIGHT = "light"
ATTR_FILTERS = "filters"
ATTR_CLEAN = "clean"
ATTR_BEEP = "beep"
ATTR_SLEEP = "sleep"
ATTR_LAST_ON_MODE = "last_on_mode"
ATTR_SWINGV = "swingv"
ATTR_SWINGH = "swingh"
ATTR_FIX_SWINGV = "fix_swingv"
ATTR_FIX_SWINGH = "fix_swingh"
ATTR_STATE_MODE = "state_mode"

SERVICE_ECONO_MODE = "set_econo"
SERVICE_TURBO_MODE = "set_turbo"
SERVICE_QUIET_MODE = "set_quiet"
SERVICE_LIGHT_MODE = "set_light"
SERVICE_FILTERS_MODE = "set_filters"
SERVICE_CLEAN_MODE = "set_clean"
SERVICE_BEEP_MODE = "set_beep"
SERVICE_SLEEP_MODE = "set_sleep"
SERVICE_SET_SWINGV = "set_swingv"
SERVICE_SET_SWINGH = "set_swingh"
SERVICE_SUPPRESS_FF_LEARNING = "suppress_ff_learning"
SERVICE_RESUME_FF_LEARNING = "resume_ff_learning"

# Map attributes to properties of the state object
ATTRIBUTES_IRHVAC = {
    ATTR_ECONO: "econo",
    ATTR_TURBO: "turbo",
    ATTR_QUIET: "quiet",
    ATTR_LIGHT: "light",
    ATTR_FILTERS: "filter",
    ATTR_CLEAN: "clean",
    ATTR_BEEP: "beep",
    ATTR_SLEEP: "sleep",
    ATTR_LAST_ON_MODE: "last_on_mode",
    ATTR_SWINGV: "swingv",
    ATTR_SWINGH: "swingh",
    ATTR_FIX_SWINGV: "fix_swingv",
    ATTR_FIX_SWINGH: "fix_swingh",
}

ON_OFF_LIST = ["ON", "OFF", "On", "Off", "on", "off"]

TOGGLE_ALL_LIST = [
    "SwingV",
    "SwingH",
    "Quiet",
    "Turbo",
    "Econo",
    "Light",
    "Filter",
    "Clean",
    "Beep",
    "Sleep",
]

STATE_MODE_LIST = ["StoreOnly", "SendStore"]

# Fujitsu preset modes (Min Heat has no HA standard equivalent)
PRESET_MIN_HEAT = "min_heat"

# Vane button config
CONF_HAS_SET_V = "has_set_vertical_vane"
CONF_HAS_SET_H = "has_set_horizontal_vane"

# IR actions config
CONF_IR_ACTIONS = "ir_actions"
CONF_PRESET_MODES_LIST = "supported_preset_modes"

# PI controller config keys
CONF_PI_ENABLED = "pi_enabled"
CONF_PI_KP = "pi_kp"
CONF_PI_KI = "pi_ki"
CONF_PI_KD = "pi_kd"
CONF_PI_KD_FILTER_N = "pi_kd_filter_n"
CONF_PI_TICK_FALLBACK = "pi_tick_fallback"
CONF_PI_DEADBAND = "pi_deadband"
CONF_OUTDOOR_TEMP_SENSOR = "outdoor_temp_sensor"
CONF_PI_OUTDOOR_SEED_HEAT = "pi_outdoor_seed_heat"
CONF_PI_OUTDOOR_SEED_COOL = "pi_outdoor_seed_cool"
CONF_PI_OUTDOOR_SEED_CLAMP_MIN = "pi_outdoor_seed_clamp_min"
CONF_PI_OUTDOOR_SEED_CLAMP_MAX = "pi_outdoor_seed_clamp_max"
CONF_PI_SETPOINT_WEIGHT = "pi_setpoint_weight"
CONF_PI_TAU_ESTIMATE = "pi_tau_estimate"
CONF_PI_RESPONSE_LAG = "pi_response_lag"
CONF_PI_IMC_LAMBDA = "pi_imc_lambda"
CONF_PI_SENSOR_FILTER_TAU = "pi_sensor_filter_tau"
CONF_PI_SMITH_ENABLED = "pi_smith_enabled"
CONF_PI_SETPOINT_HOLD = "pi_setpoint_hold"
CONF_PI_AUTO_PERTURB_ENABLED = "pi_auto_perturb_enabled"
CONF_PI_AUTO_PERTURB_WINDOW_START = "pi_auto_perturb_window_start"
CONF_PI_AUTO_PERTURB_WINDOW_END = "pi_auto_perturb_window_end"
CONF_PI_GREYBOX_BLENDING = "pi_greybox_blending"
CONF_PI_INTERCEPT_SEED_HEAT = "pi_intercept_seed_heat"
CONF_PI_INTERCEPT_SEED_COOL = "pi_intercept_seed_cool"
CONF_PI_FF_ENABLED = "pi_ff_enabled"
CONF_PI_RLS_ONLINE_ENABLED = "pi_rls_online_enabled"
CONF_PI_BATCH_WLS_ENABLED = "pi_batch_wls_enabled"
CONF_PI_PLANT_ID_ENABLED = "pi_plant_id_enabled"
CONF_PI_TOBIT_ENABLED = "pi_tobit_enabled"
CONF_PI_MODEL_INPUTS = "pi_model_inputs"

# Subentry types
SUBENTRY_MODEL_INPUT = "model_input"
SUBENTRY_SUPPLEMENTAL_SOURCE = "supplemental_source"

# Supplemental source config keys
CONF_SUPPLEMENTAL_NAME = "name"
CONF_SUPPLEMENTAL_ENTITY = "entity_id"
CONF_SUPPLEMENTAL_SEED_HEAT = "seed_heat"
CONF_SUPPLEMENTAL_SEED_COOL = "seed_cool"
CONF_SUPPLEMENTAL_FAILURE_THRESHOLD = "failure_threshold"
CONF_SUPPLEMENTAL_RECOVERY_MARGIN = "recovery_margin"
CONF_SUPPLEMENTAL_AUTO_MODEL_INPUT = "auto_model_input"

# Supplemental source defaults
DEFAULT_SUPPLEMENTAL_FAILURE_THRESHOLD = 900  # 15 minutes
DEFAULT_SUPPLEMENTAL_RECOVERY_MARGIN = 0.3    # °C
DEFAULT_SUPPLEMENTAL_SEED = 3.0                # Typical pellet stove: warms room by ~3°C equiv

# PID controller defaults
DEFAULT_PI_ENABLED = False
DEFAULT_PI_KP = 1.0
DEFAULT_PI_KI = 0.20             # Optimized via 2R2C parameter sweep (Apr 2026)
DEFAULT_PI_KD = 0.0              # Literature + 72h replay: D contraindicated for quantized HVAC
DEFAULT_PI_KD_FILTER_N = 8       # Derivative filter coefficient: Tf = Td/N. Higher N = less filtering.
DEFAULT_PI_TICK_FALLBACK = 900
DEFAULT_PI_DEADBAND = 0.5
DEFAULT_PI_OUTDOOR_SEED_HEAT = 0.25
DEFAULT_PI_OUTDOOR_SEED_COOL = 0.25
DEFAULT_PI_OUTDOOR_SEED_CLAMP_MIN = 0.0
DEFAULT_PI_OUTDOOR_SEED_CLAMP_MAX = 2.0
DEFAULT_PI_SETPOINT_WEIGHT = 0.15 # 2-DOF: p_term = kp * b * error. Lower b since FF handles setpoint steps.
DEFAULT_PI_TAU_ESTIMATE = 60.0     # DEPRECATED: now an IMC enable flag only (>0 = IMC on, 0 = manual Kp/Ki).
                                   # The numeric value is no longer used as a τ seed — see DEFAULT_TAU_FAST_SEED
                                   # and DEFAULT_TAU_SLOW_SEED below.  Field is hidden from config flow; existing
                                   # installs with explicit 0 keep manual gains.

# Plant-ID seeds (internal — not config-exposed).  τ_fast and τ_slow are
# physically distinct (air-mass coupling vs wall-mass coupling) and need
# separate defaults.  Values are conservative low-end residential figures
# (Madsen & Holst, ASHRAE Ch 18): under-estimating τ → lower kp → sluggish
# but stable, biased away from the dangerous over-aggressive direction.
DEFAULT_TAU_FAST_SEED = 20.0       # Air-mass response (minutes). Typical mini-split air loop.
DEFAULT_TAU_SLOW_SEED = 60.0       # Wall-mass response (minutes). Low-end residential thermal mass.

# Maturity gate: minimum non-seed observations before τ from plant ID
# may drive IMC gains.  Below this threshold, compute_gains() falls back
# to the conservative seed.  Skogestad SIMC + cautious adaptation pattern.
MIN_TAU_OBSERVATIONS_FOR_GATE = 3
DEFAULT_PI_RESPONSE_LAG = 15.0     # HP response lag L (minutes). Compressor→room first-order lag.
DEFAULT_PI_IMC_LAMBDA = 0.0        # IMC closed-loop speed λ (minutes). 0 = auto (τ/2).
DEFAULT_PI_SENSOR_FILTER_TAU = 120  # Measurement low-pass filter τ (seconds). 0 = disabled.
DEFAULT_PI_SMITH_ENABLED = True     # Smith predictor for dead-time compensation. Requires IMC (tau>0).
DEFAULT_PI_SETPOINT_HOLD = 1200     # Min seconds between HP setpoint changes. 0 = disabled.
DEFAULT_PI_INTERCEPT_SEED_HEAT = 0.0
DEFAULT_PI_INTERCEPT_SEED_COOL = 0.0
DEFAULT_PI_FF_ENABLED = True
# Online RLS is retired per project_online_rls_verdict.md — batch WLS
# is the sole production estimator. Default off to match the verdict;
# users who want online RLS for experimentation can opt in via config
# flow. Existing user configs with the key explicitly set keep their
# stored value; only new installs see this default.
DEFAULT_PI_RLS_ONLINE_ENABLED = False
DEFAULT_PI_BATCH_WLS_ENABLED = True
DEFAULT_PI_PLANT_ID_ENABLED = True
# Tobit / censored WLS at HP saturation (#40). Default off — opt-in until
# bench validation in Session 5 of #40 confirms convergence-quality wins.
# When enabled: relaxes the buffer admission gate to admit
# saturated_high / saturated_low / no_output observations alongside
# uncensored ones, and partitions them at WLS time. The actual Tobit MLE
# solve is gated on Session 2's solver landing.
DEFAULT_PI_TOBIT_ENABLED = False

# RLS model defaults
DEFAULT_RLS_LAMBDA_BASE = 0.999  # Base forgetting factor (~10 day effective memory)
DEFAULT_RLS_LAMBDA_MIN = 0.995   # Minimum λ when residuals are large
DEFAULT_RLS_DELTA = 0.001        # Covariance regularization (added to P diagonal each step)
DEFAULT_RLS_P_INIT = 1.0         # Initial covariance diagonal (prior uncertainty per coeff)

# PI controller extra state attributes
ATTR_HP_SETPOINT = "hp_setpoint"
ATTR_PI_INTEGRAL = "pi_integral"
ATTR_DESIRED_TEMP = "desired_temp"
ATTR_FF_OFFSET = "ff_offset"

# Dispatcher signals (format with entry_id)
SIGNAL_PI_UPDATE = "tasmota_irhvac_pi_update_{}"
SIGNAL_FF_SUPPRESS_UPDATE = "tasmota_irhvac_ff_suppress_update_{}"
SIGNAL_PI_BATCH_COMPLETE = "tasmota_irhvac_batch_complete_{}"
