"""Binary sensor entities for Tasmota IRHVAC (FF learning suppression status)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .climate import TasmotaIrhvac
    from .pi import PIController

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DATA_KEY, SIGNAL_FF_SUPPRESS_UPDATE, SIGNAL_PI_UPDATE

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

    if not climate_entity._pi:
        return
    if not climate_entity._pi.is_active:
        return

    async_add_entities([
        FFLearningSuppressedBinarySensor(
            climate_entity=climate_entity,
            entry_id=entry.entry_id,
        ),
        ModelDriftingBinarySensor(
            climate_entity=climate_entity,
            entry_id=entry.entry_id,
        ),
    ])


class FFLearningSuppressedBinarySensor(BinarySensorEntity):
    """Binary sensor showing whether FF auto-learning is suppressed."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "ff_learning"

    def __init__(self, climate_entity: TasmotaIrhvac, entry_id: str) -> None:
        """Initialize the binary sensor."""
        self._climate = climate_entity
        self._entry_id = entry_id
        self._attr_unique_id = f"{climate_entity.unique_id}_ff_learning"

    @property
    def device_info(self):
        """Return device info to group with climate entity."""
        return self._climate.device_info

    @property
    def available(self) -> bool:
        """Available when the climate entity is available."""
        return bool(self._climate.available)

    @property
    def _pi(self) -> PIController | None:
        """Return the PI controller, narrowed from the union type."""
        from .pi import PIController
        pi = self._climate._pi
        return pi if isinstance(pi, PIController) else None

    @property
    def is_on(self) -> bool:
        """True when FF learning is suppressed."""
        pi = self._pi
        if pi is None:
            return False
        return bool(pi.get_learning_status()["suppressed"])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return details about what is suppressing learning."""
        pi = self._pi
        if pi is None:
            return {}
        status = pi.get_learning_status()
        return {
            "manual_suppress": status["manual_suppress"],
            "manual_suppress_reason": status["manual_suppress_reason"],
            "active_suppressors": status["active_suppressors"],
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


class ModelDriftingBinarySensor(BinarySensorEntity):
    """Binary sensor indicating persistent same-direction batch correction."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "model_drifting"

    def __init__(self, climate_entity: TasmotaIrhvac, entry_id: str) -> None:
        """Initialize the binary sensor."""
        self._climate = climate_entity
        self._entry_id = entry_id
        self._attr_unique_id = f"{climate_entity.unique_id}_model_drifting"

    @property
    def device_info(self):
        """Return device info to group with climate entity."""
        return self._climate.device_info

    @property
    def available(self) -> bool:
        """Available when the climate entity is available."""
        return bool(self._climate.available)

    @property
    def _pi(self) -> PIController | None:
        """Return the PI controller, narrowed from the union type."""
        from .pi import PIController
        pi = self._climate._pi
        return pi if isinstance(pi, PIController) else None

    @property
    def is_on(self) -> bool:
        """True when any coefficient shows persistent same-direction drift."""
        pi = self._pi
        if pi is None:
            return False
        return bool(pi.get_drifting_coefficients())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return details about which coefficients are drifting."""
        pi = self._pi
        if pi is None:
            return {}
        drifting = pi.get_drifting_coefficients()
        if not drifting:
            return {}
        return {
            "drifting_coefficients": [
                {"name": name, "consecutive_cycles": count}
                for _idx, name, count in drifting
            ],
        }

    async def async_added_to_hass(self) -> None:
        """Subscribe to dispatcher signal for state updates."""

        @callback
        def _update_sensor():
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_PI_UPDATE.format(self._entry_id),
                _update_sensor,
            )
        )
