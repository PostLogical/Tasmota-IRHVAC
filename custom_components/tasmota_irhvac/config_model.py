"""Typed, frozen config model for TasmotaIrhvac.

Parsed once in async_setup_entry from the raw config dict.
All defaults applied here — no default-handling scattered through __init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homeassistant.const import CONF_NAME, CONF_UNIQUE_ID

from .const import (
    CONF_AVAILABILITY_TOPIC,
    CONF_AWAY_TEMP,
    CONF_BEEP,
    CONF_CELSIUS,
    CONF_CLEAN,
    CONF_IR_PROTOCOL_UNIT,
    CONF_COMMAND_TOPIC,
    CONF_ECONO,
    CONF_FAN_LIST,
    CONF_FILTER,
    CONF_IGNORE_OFF_TEMP,
    CONF_INITIAL_OPERATION_MODE,
    CONF_IR_ACTIONS,
    CONF_KEEP_MODE,
    CONF_LIGHT,
    CONF_MAX_TEMP,
    CONF_MIN_TEMP,
    CONF_MODEL,
    CONF_MODES_LIST,
    CONF_MQTT_DELAY,
    CONF_PI_ENABLED,
    CONF_POWER_SENSOR,
    CONF_PRECISION,
    CONF_PRESET_MODES_LIST,
    CONF_PROTOCOL,
    CONF_QUIET,
    CONF_SLEEP,
    CONF_SPECIAL_MODE,
    CONF_STATE_TOPIC,
    CONF_STATE_TOPIC_2,
    CONF_SWING_LIST,
    CONF_SWINGH,
    CONF_SWINGV,
    CONF_TEMP_SENSOR,
    CONF_TEMP_STEP,
    CONF_HUMIDITY_SENSOR,
    CONF_TARGET_TEMP,
    CONF_TOGGLE_LIST,
    CONF_TURBO,
    CONF_VENDOR,
    DEFAULT_CONF_BEEP,
    DEFAULT_CONF_CLEAN,
    DEFAULT_IR_PROTOCOL_UNIT,
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
    DEFAULT_TARGET_TEMP,
)


def _parse_ir_protocol_unit(config: dict) -> str:
    """Parse IR protocol unit from config, handling legacy celsius_mode format."""
    # New key takes priority
    val: str | None = config.get(CONF_IR_PROTOCOL_UNIT)
    if val is not None:
        return val  # Already "celsius" or "fahrenheit"
    # Fall back to legacy celsius_mode ("on"/"off")
    legacy = config.get(CONF_CELSIUS, "on")
    if isinstance(legacy, str) and legacy.lower() in ("on", "celsius"):
        return "celsius"
    return "fahrenheit"


@dataclass(frozen=True)
class IrhvacConfig:
    """Complete, immutable config for a TasmotaIrhvac entity.

    All string toggles are stored lowercase ("on"/"off").
    """

    # ── Identity & transport ─────────────────────────────────────────
    name: str | None
    unique_id: str | None
    vendor: str
    command_topic: str
    state_topic: str
    state_topic_2: str | None
    availability_topic: str | None
    mqtt_delay: float
    model: str

    # ── Temperature ──────────────────────────────────────────────────
    min_temp: float
    max_temp: float
    target_temp: float
    precision: float
    temp_step: float
    ir_protocol_unit: str  # "celsius" or "fahrenheit" — IR encoding unit
    away_temp: float | None
    ignore_off_temp: bool

    # ── Modes & features ─────────────────────────────────────────────
    modes_list: list[str]
    fan_list: list[str] | None
    swing_list: list[str] | None
    initial_operation_mode: str | None
    keep_mode: bool
    preset_modes_list: list[str] | None
    ir_actions: list[dict[str, Any]]

    # ── Toggle defaults ──────────────────────────────────────────────
    quiet: str
    turbo: str
    econo: str
    light: str
    filter: str
    clean: str
    beep: str
    sleep: str

    # ── Swing defaults ───────────────────────────────────────────────
    swingv: str | None
    swingh: str | None
    toggle_list: list[str]
    special_mode: str

    # ── Sensors ──────────────────────────────────────────────────────
    temp_sensor: str | None
    humidity_sensor: str | None
    power_sensor: str | None

    # ── PI controller (raw config dict passed through) ───────────────
    pi_enabled: bool
    pi_raw_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> IrhvacConfig:
        """Parse a raw config dict into an IrhvacConfig.

        Handles all defaults and normalization in one place.
        """
        vendor = config.get(CONF_VENDOR) or config.get(CONF_PROTOCOL) or ""
        swingv_raw = config.get(CONF_SWINGV)
        swingh_raw = config.get(CONF_SWINGH)

        return cls(
            # Identity & transport
            name=config.get(CONF_NAME),
            unique_id=config.get(CONF_UNIQUE_ID),
            vendor=vendor,
            command_topic=config.get(CONF_COMMAND_TOPIC, ""),
            state_topic=config.get(CONF_STATE_TOPIC, ""),
            state_topic_2=(
                config.get(CONF_STATE_TOPIC_2)
                or config.get(CONF_STATE_TOPIC + "_2")
            ),
            availability_topic=config.get(CONF_AVAILABILITY_TOPIC),
            mqtt_delay=float(config.get(CONF_MQTT_DELAY, DEFAULT_MQTT_DELAY)),
            model=config.get(CONF_MODEL, DEFAULT_CONF_MODEL),
            # Temperature
            min_temp=float(config.get(CONF_MIN_TEMP, DEFAULT_MIN_TEMP)),
            max_temp=float(config.get(CONF_MAX_TEMP, DEFAULT_MAX_TEMP)),
            target_temp=float(config.get(CONF_TARGET_TEMP, DEFAULT_TARGET_TEMP)),
            precision=float(config.get(CONF_PRECISION, 1.0)),
            temp_step=float(config.get(CONF_TEMP_STEP, 1.0)),
            ir_protocol_unit=_parse_ir_protocol_unit(config),
            away_temp=config.get(CONF_AWAY_TEMP),
            ignore_off_temp=config.get(CONF_IGNORE_OFF_TEMP, DEFAULT_IGNORE_OFF_TEMP),
            # Modes & features
            modes_list=config.get(CONF_MODES_LIST, []),
            fan_list=config.get(CONF_FAN_LIST),
            swing_list=config.get(CONF_SWING_LIST),
            initial_operation_mode=config.get(CONF_INITIAL_OPERATION_MODE),
            keep_mode=config.get(CONF_KEEP_MODE, DEFAULT_CONF_KEEP_MODE),
            preset_modes_list=config.get(CONF_PRESET_MODES_LIST),
            ir_actions=config.get(CONF_IR_ACTIONS, []),
            # Toggles (always lowercase)
            quiet=config.get(CONF_QUIET, DEFAULT_CONF_QUIET).lower(),
            turbo=config.get(CONF_TURBO, DEFAULT_CONF_TURBO).lower(),
            econo=config.get(CONF_ECONO, DEFAULT_CONF_ECONO).lower(),
            light=config.get(CONF_LIGHT, DEFAULT_CONF_LIGHT).lower(),
            filter=config.get(CONF_FILTER, DEFAULT_CONF_FILTER).lower(),
            clean=config.get(CONF_CLEAN, DEFAULT_CONF_CLEAN).lower(),
            beep=config.get(CONF_BEEP, DEFAULT_CONF_BEEP).lower(),
            sleep=config.get(CONF_SLEEP, DEFAULT_CONF_SLEEP).lower(),
            # Swing
            swingv=swingv_raw.lower() if swingv_raw else None,
            swingh=swingh_raw.lower() if swingh_raw else None,
            toggle_list=config.get(CONF_TOGGLE_LIST, []),
            special_mode=config.get(CONF_SPECIAL_MODE, ""),
            # Sensors
            temp_sensor=config.get(CONF_TEMP_SENSOR) or None,
            humidity_sensor=config.get(CONF_HUMIDITY_SENSOR) or None,
            power_sensor=config.get(CONF_POWER_SENSOR) or None,
            # PI
            pi_enabled=config.get(CONF_PI_ENABLED, False),
            pi_raw_config=dict(config),
        )
