"""Binary sensor entities for Tasmota IRHVAC.

All binary sensors are `CoordinatorEntity[TasmotaIRHVACCoordinator]`
subclasses — refresh is automatic when the controller publishes a new
TickOutput.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.helpers.device_registry import DeviceInfo

    from .climate import TasmotaIrhvac
    from .pi import PIController

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DATA_KEY
from .pi.coordinator import TasmotaIRHVACCoordinator

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

    coordinator = climate_entity.coordinator
    if coordinator is None:  # pragma: no cover — unreachable: PI-enabled entities always have a coordinator
        return

    async_add_entities([
        FFLearningSuppressedBinarySensor(
            coordinator=coordinator, climate_entity=climate_entity,
        ),
        ModelDriftingBinarySensor(
            coordinator=coordinator, climate_entity=climate_entity,
        ),
    ])


class FFLearningSuppressedBinarySensor(
    CoordinatorEntity[TasmotaIRHVACCoordinator], BinarySensorEntity,
):
    """Binary sensor showing whether FF auto-learning is suppressed.

    Reads typed `coordinator.data.learning_suppression.effective_suppressed`
    for the on/off state. Manual-suppress fields read from the typed
    `coordinator.data.rls_model` for the rich attribute set.
    """

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "ff_learning"

    def __init__(
        self,
        coordinator: TasmotaIRHVACCoordinator,
        climate_entity: TasmotaIrhvac,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator)
        self._climate = climate_entity
        self._attr_unique_id = f"{climate_entity.unique_id}_ff_learning"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group with climate entity."""
        return self._climate.device_info

    @property
    def available(self) -> bool:
        """Available when the climate entity is available."""
        return bool(self._climate.available)

    @property
    def is_on(self) -> bool:
        """True when FF learning is effectively suppressed."""
        return self.coordinator.data.learning_suppression.effective_suppressed

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return details about what is suppressing learning."""
        ls = self.coordinator.data.learning_suppression
        rls = self.coordinator.data.rls_model
        return {
            "manual_suppress": rls.learning_suppressed,
            "manual_suppress_reason": rls.manual_suppress_reason,
            "active_suppressors": list(ls.active_suppressors),
        }


class ModelDriftingBinarySensor(
    CoordinatorEntity[TasmotaIRHVACCoordinator], BinarySensorEntity,
):
    """Binary sensor indicating persistent same-direction batch correction.

    Reads `coordinator.data.batch_learning.drift_detection.drifting_coefficients`
    for the on/off state and the per-coefficient details.
    """

    _attr_has_entity_name = True
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_translation_key = "model_drifting"

    def __init__(
        self,
        coordinator: TasmotaIRHVACCoordinator,
        climate_entity: TasmotaIrhvac,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator)
        self._climate = climate_entity
        self._attr_unique_id = f"{climate_entity.unique_id}_model_drifting"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group with climate entity."""
        return self._climate.device_info

    @property
    def available(self) -> bool:
        """Available when the climate entity is available."""
        return bool(self._climate.available)

    def _drifting(self) -> tuple:
        """Return the drifting coefficients tuple, or empty if no batch run yet."""
        batch = self.coordinator.data.batch_learning
        if batch is None:
            return ()
        return batch.drift_detection.drifting_coefficients

    @property
    def is_on(self) -> bool:
        """True when any coefficient shows persistent same-direction drift."""
        return bool(self._drifting())

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return details about which coefficients are drifting."""
        drifting = self._drifting()
        if not drifting:
            return {}
        return {
            "drifting_coefficients": [
                {"name": d.name, "consecutive_cycles": d.consecutive_cycles}
                for d in drifting
            ],
        }
