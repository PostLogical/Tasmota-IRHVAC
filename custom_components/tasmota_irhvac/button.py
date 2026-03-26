"""Button entities for Tasmota IRHVAC (vane position control)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from homeassistant.components import mqtt
from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from homeassistant.components.climate.const import (
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_HAS_SET_H, CONF_HAS_SET_V, DATA_KEY

_LOGGER = logging.getLogger(__name__)

# Fujitsu raw IR codes for vane position cycling
FUJITSU_RAW_PREFIX = "raw,0,3324,1574,448,390,1182,"
IR_SET_V = FUJITSU_RAW_PREFIX + "001010001100011000000000000010000000100000110110110010011"
IR_SET_H = FUJITSU_RAW_PREFIX + "00101000110001100000000000001000000010001001111001100001"


@dataclass(frozen=True, kw_only=True)
class VaneButtonDescription(ButtonEntityDescription):
    """Describe a vane button."""

    ir_code: str
    clears_swing_axis: str  # "vertical" or "horizontal"


VANE_BUTTON_DESCRIPTIONS: dict[str, VaneButtonDescription] = {
    CONF_HAS_SET_V: VaneButtonDescription(
        key="set_vertical_vane",
        translation_key="set_vertical_vane",
        ir_code=IR_SET_V,
        clears_swing_axis="vertical",
    ),
    CONF_HAS_SET_H: VaneButtonDescription(
        key="set_horizontal_vane",
        translation_key="set_horizontal_vane",
        ir_code=IR_SET_H,
        clears_swing_axis="horizontal",
    ),
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up vane button entities from a config entry."""
    climate_entity = hass.data.get(DATA_KEY, {}).get(entry.entry_id)
    if climate_entity is None:
        return

    config = {**entry.data, **entry.options}
    buttons = []

    for conf_key, description in VANE_BUTTON_DESCRIPTIONS.items():
        if config.get(conf_key):
            buttons.append(
                VaneButton(
                    climate_entity=climate_entity,
                    description=description,
                )
            )

    if buttons:
        async_add_entities(buttons)


class VaneButton(ButtonEntity):
    """Button that sends a raw IR code to cycle vane position."""

    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(
        self,
        climate_entity,
        description: VaneButtonDescription,
    ) -> None:
        """Initialize the vane button."""
        self.entity_description = description
        self._climate = climate_entity
        self._attr_unique_id = f"{climate_entity.unique_id}_{description.key}"

    @property
    def device_info(self):
        """Return device info to group button with climate entity."""
        return self._climate.device_info

    @property
    def available(self) -> bool:
        """Button is available when the climate entity is available."""
        return self._climate.available

    async def async_press(self) -> None:
        """Send the raw IR code and update swing state."""
        description = self.entity_description

        # Send the raw IR code via MQTT
        topic = self._climate.topic
        path = topic.split("/")
        irsend_topic = f"cmnd/{path[1]}/irsend"
        mqtt_delay = float(getattr(self._climate, "_mqtt_delay", "0"))
        if mqtt_delay > 0:
            await asyncio.sleep(mqtt_delay)
        await mqtt.async_publish(self.hass, irsend_topic, description.ir_code)

        # Update climate entity swing state — vane is now fixed (not oscillating)
        if description.clears_swing_axis == "vertical":
            self._climate._swingv = None
            if self._climate._attr_swing_mode == SWING_VERTICAL:
                self._climate._attr_swing_mode = SWING_OFF
            elif self._climate._attr_swing_mode == SWING_BOTH:
                self._climate._attr_swing_mode = SWING_HORIZONTAL
        elif description.clears_swing_axis == "horizontal":
            self._climate._swingh = None
            if self._climate._attr_swing_mode == SWING_HORIZONTAL:
                self._climate._attr_swing_mode = SWING_OFF
            elif self._climate._attr_swing_mode == SWING_BOTH:
                self._climate._attr_swing_mode = SWING_VERTICAL

        self._climate.async_schedule_update_ha_state()
        _LOGGER.debug("Vane button %s pressed, IR sent", description.key)
