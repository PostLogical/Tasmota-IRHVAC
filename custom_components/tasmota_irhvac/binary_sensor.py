"""Binary sensor entities for Tasmota IRHVAC (FF learning suppression status)."""

from __future__ import annotations

import logging

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DATA_KEY, SIGNAL_FF_SUPPRESS_UPDATE

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up binary sensor entities from a config entry."""
    climate_entity = hass.data.get(DATA_KEY, {}).get(entry.entry_id)
    if climate_entity is None:
        return

    from .pi_controller import PIControllerMixin

    if not isinstance(climate_entity, PIControllerMixin):
        return
    if not climate_entity._pi_enabled:
        return

    async_add_entities([
        FFLearningSuppressedBinarySensor(
            climate_entity=climate_entity,
            entry_id=entry.entry_id,
        )
    ])


class FFLearningSuppressedBinarySensor(BinarySensorEntity):
    """Binary sensor showing whether FF auto-learning is suppressed."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "ff_learning_suppressed"

    def __init__(self, climate_entity, entry_id: str) -> None:
        """Initialize the binary sensor."""
        self._climate = climate_entity
        self._entry_id = entry_id
        self._attr_unique_id = f"{climate_entity.unique_id}_ff_learning_suppressed"

    @property
    def device_info(self):
        """Return device info to group with climate entity."""
        return self._climate.device_info

    @property
    def available(self) -> bool:
        """Available when the climate entity is available."""
        return self._climate.available

    @property
    def is_on(self) -> bool:
        """True when FF learning is suppressed."""
        suppress, _, _ = self._climate._compute_disturbance_effects()
        return suppress

    @property
    def extra_state_attributes(self):
        """Return details about what is suppressing learning."""
        suppress, active_entities, total_bias = (
            self._climate._compute_disturbance_effects()
        )
        return {
            "manual_suppress": self._climate._manual_ff_suppress,
            "manual_suppress_reason": self._climate._manual_ff_suppress_reason,
            "active_entity_suppressors": active_entities,
            "total_bias": round(total_bias, 2),
        }

    async def async_added_to_hass(self) -> None:
        """Subscribe to dispatcher signal for state updates."""

        @callback
        def _update_sensor():
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._entry_id),
                _update_sensor,
            )
        )
