"""Companion sensor entities for the Tasmota IRHVAC PI controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DATA_KEY, SIGNAL_PI_UPDATE

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class TasmotaIrhvacPISensorDescription(SensorEntityDescription):
    """Describe a Tasmota IRHVAC PI sensor."""

    climate_attr: str


PI_SENSOR_DESCRIPTIONS: tuple[TasmotaIrhvacPISensorDescription, ...] = (
    TasmotaIrhvacPISensorDescription(
        key="hp_setpoint",
        translation_key="hp_setpoint",
        climate_attr="_hp_setpoint",
        device_class=SensorDeviceClass.TEMPERATURE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=None,
        suggested_display_precision=1,
    ),
    TasmotaIrhvacPISensorDescription(
        key="pi_integral",
        translation_key="pi_integral",
        climate_attr="_pi_integral",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=3,
    ),
    TasmotaIrhvacPISensorDescription(
        key="ff_offset",
        translation_key="ff_offset",
        climate_attr="_ff_offset",
        device_class=None,
        native_unit_of_measurement="°C",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="disturbance_bias",
        translation_key="disturbance_bias",
        climate_attr="_disturbance_total_bias",
        device_class=None,
        native_unit_of_measurement="°C",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="ff_anticipated_offset",
        translation_key="ff_anticipated_offset",
        climate_attr="_ff_anticipated_offset",
        device_class=None,
        native_unit_of_measurement="°C",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="integral_convergence",
        translation_key="integral_convergence",
        climate_attr="_integral_convergence",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up PI sensor entities from a config entry."""
    climate_entity = hass.data.get(DATA_KEY, {}).get(entry.entry_id)
    if climate_entity is None:
        return

    if not climate_entity._pi:
        return
    if not climate_entity._pi._pi_enabled:
        return

    sensors = [
        TasmotaIrhvacPISensor(
            climate_entity=climate_entity,
            entry_id=entry.entry_id,
            description=desc,
        )
        for desc in PI_SENSOR_DESCRIPTIONS
    ]
    async_add_entities(sensors)


class TasmotaIrhvacPISensor(SensorEntity):
    """Sensor that mirrors a PI controller value from the climate entity."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    entity_description: TasmotaIrhvacPISensorDescription

    def __init__(
        self,
        climate_entity,
        entry_id: str,
        description: TasmotaIrhvacPISensorDescription,
    ) -> None:
        """Initialize the PI sensor."""
        self.entity_description = description
        self._climate = climate_entity
        self._entry_id = entry_id
        self._attr_unique_id = f"{climate_entity.unique_id}_{description.key}"

    @property
    def device_info(self):
        """Return device info to group sensor with climate entity."""
        return self._climate.device_info

    @property
    def native_value(self):
        """Read current value from the PI controller."""
        pi = self._climate._pi
        if pi is None:
            return None
        return getattr(pi, self.entity_description.climate_attr, None)

    @property
    def available(self) -> bool:
        """Sensor is available when the climate entity is available."""
        return self._climate.available

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
