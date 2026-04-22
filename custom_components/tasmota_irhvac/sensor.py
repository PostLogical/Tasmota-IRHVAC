"""Companion sensor entities for the Tasmota IRHVAC PI controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.helpers.device_registry import DeviceInfo

    from .climate import TasmotaIrhvac
    from .pi import PIController

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfTemperature, UnitOfTime
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
        key="integral_convergence",
        translation_key="integral_convergence",
        climate_attr="_integral_convergence",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="itae",
        translation_key="itae",
        climate_attr="_itae_accumulator",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=1,
    ),
    TasmotaIrhvacPISensorDescription(
        key="comfort_violation_hours",
        translation_key="comfort_violation_hours",
        climate_attr="_comfort_violation_hours",
        device_class=None,
        native_unit_of_measurement="h",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="setpoint_changes",
        translation_key="setpoint_changes",
        climate_attr="_setpoint_changes",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=0,
    ),
    TasmotaIrhvacPISensorDescription(
        key="controllable_itae",
        translation_key="controllable_itae",
        climate_attr="_controllable_itae",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=1,
    ),
    TasmotaIrhvacPISensorDescription(
        key="uncontrollable_itae",
        translation_key="uncontrollable_itae",
        climate_attr="_uncontrollable_itae",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=1,
    ),
    TasmotaIrhvacPISensorDescription(
        key="controllable_cvh",
        translation_key="controllable_cvh",
        climate_attr="_controllable_cvh",
        device_class=None,
        native_unit_of_measurement="h",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="uncontrollable_cvh",
        translation_key="uncontrollable_cvh",
        climate_attr="_uncontrollable_cvh",
        device_class=None,
        native_unit_of_measurement="h",
        state_class=SensorStateClass.TOTAL_INCREASING,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="ff_load_fraction",
        translation_key="ff_load_fraction",
        climate_attr="_ff_load_fraction",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=2,
    ),
    TasmotaIrhvacPISensorDescription(
        key="batch_model_rms",
        translation_key="batch_model_rms",
        climate_attr="_batch_model_rms",
        device_class=None,
        native_unit_of_measurement="°C",
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=3,
    ),
    TasmotaIrhvacPISensorDescription(
        key="buffer_eligible",
        translation_key="buffer_eligible",
        climate_attr="buffer_eligible",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=0,
    ),
    TasmotaIrhvacPISensorDescription(
        key="buffer_total",
        translation_key="buffer_total",
        climate_attr="buffer_total",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=0,
    ),
    TasmotaIrhvacPISensorDescription(
        key="buffer_oldest_age_hours",
        translation_key="buffer_oldest_age_hours",
        climate_attr="buffer_oldest_age_hours",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.HOURS,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=1,
    ),
    TasmotaIrhvacPISensorDescription(
        key="buffer_leverage_max",
        translation_key="buffer_leverage_max",
        climate_attr="buffer_leverage_max",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=4,
    ),
    TasmotaIrhvacPISensorDescription(
        key="batch_outliers_excluded",
        translation_key="batch_outliers_excluded",
        climate_attr="batch_outliers_excluded",
        device_class=None,
        native_unit_of_measurement=None,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=0,
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
    if not climate_entity._pi.is_active:
        return

    sensors: list[SensorEntity] = [
        TasmotaIrhvacPISensor(
            climate_entity=climate_entity,
            entry_id=entry.entry_id,
            description=desc,
        )
        for desc in PI_SENSOR_DESCRIPTIONS
    ]
    sensors.append(
        TasmotaIrhvacHealthSensor(
            climate_entity=climate_entity,
            entry_id=entry.entry_id,
        )
    )
    sensors.append(
        TasmotaIrhvacLearningSensor(
            climate_entity=climate_entity,
            entry_id=entry.entry_id,
        )
    )
    async_add_entities(sensors)


class TasmotaIrhvacPISensor(SensorEntity):
    """Sensor that mirrors a PI controller value from the climate entity."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    entity_description: TasmotaIrhvacPISensorDescription

    def __init__(
        self,
        climate_entity: TasmotaIrhvac,
        entry_id: str,
        description: TasmotaIrhvacPISensorDescription,
    ) -> None:
        """Initialize the PI sensor."""
        self.entity_description = description
        self._climate = climate_entity
        self._entry_id = entry_id
        self._attr_unique_id = f"{climate_entity.unique_id}_{description.key}"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group sensor with climate entity."""
        return self._climate.device_info

    @property
    def native_value(self) -> float | str | None:
        """Read current value from the PI controller."""
        pi = self._climate._pi
        if pi is None:
            return None
        return getattr(pi, self.entity_description.climate_attr, None)

    @property
    def available(self) -> bool:
        """Sensor is available when the climate entity is available."""
        return bool(self._climate.available)

    async def async_added_to_hass(self) -> None:
        """Subscribe to dispatcher signal for state updates."""

        @callback
        def _update_sensor() -> None:
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_PI_UPDATE.format(self._entry_id),
                _update_sensor,
            )
        )


class TasmotaIrhvacHealthSensor(SensorEntity):
    """Sensor that evaluates PI controller health status."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["OK", "Warning", "Critical", "Disabled"]
    _attr_translation_key = "health"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        climate_entity: TasmotaIrhvac,
        entry_id: str,
    ) -> None:
        """Initialize the health sensor."""
        self._climate = climate_entity
        self._entry_id = entry_id
        self._attr_unique_id = f"{climate_entity.unique_id}_health"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group sensor with climate entity."""
        return self._climate.device_info

    @property
    def _pi(self) -> PIController | None:
        """Return the PI controller, narrowed from the union type."""
        from .pi import PIController
        pi = self._climate._pi
        return pi if isinstance(pi, PIController) else None

    @property
    def native_value(self) -> str | None:
        """Return current health state."""
        pi = self._pi
        if pi is None:
            return None
        return str(pi.get_health_status()["state"])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return health details as attributes."""
        pi = self._pi
        if pi is None:
            return {}
        status = pi.get_health_status()
        attrs = dict(status)
        # Join alerts as semicolon-separated string for HA display
        attrs["alerts"] = "; ".join(status["alerts"]) if status["alerts"] else ""
        return attrs

    @property
    def available(self) -> bool:
        """Sensor is available when the climate entity is available."""
        return bool(self._climate.available)

    async def async_added_to_hass(self) -> None:
        """Subscribe to dispatcher signal for state updates."""

        @callback
        def _update_sensor() -> None:
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_PI_UPDATE.format(self._entry_id),
                _update_sensor,
            )
        )


class TasmotaIrhvacLearningSensor(SensorEntity):
    """Sensor that reports learning state of the PI controller."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["Learning", "Optimizing", "Optimized"]
    _attr_translation_key = "learning"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        climate_entity: TasmotaIrhvac,
        entry_id: str,
    ) -> None:
        """Initialize the learning sensor."""
        self._climate = climate_entity
        self._entry_id = entry_id
        self._attr_unique_id = f"{climate_entity.unique_id}_learning"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group sensor with climate entity."""
        return self._climate.device_info

    @property
    def _pi(self) -> PIController | None:
        """Return the PI controller, narrowed from the union type."""
        from .pi import PIController
        pi = self._climate._pi
        return pi if isinstance(pi, PIController) else None

    @property
    def native_value(self) -> str | None:
        """Return current learning state."""
        pi = self._pi
        if pi is None:
            return None
        return str(pi.get_learning_state()["state"])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return learning details as attributes."""
        pi = self._pi
        if pi is None:
            return {}
        state = pi.get_learning_state()
        attrs = dict(state)
        attrs["frozen_features"] = ", ".join(state["frozen_features"]) if state["frozen_features"] else ""
        attrs["active_features"] = ", ".join(state["active_features"]) if state["active_features"] else ""
        return attrs

    @property
    def available(self) -> bool:
        """Sensor is available when the climate entity is available."""
        return bool(self._climate.available)

    async def async_added_to_hass(self) -> None:
        """Subscribe to dispatcher signal for state updates."""

        @callback
        def _update_sensor() -> None:
            self.async_write_ha_state()

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                SIGNAL_PI_UPDATE.format(self._entry_id),
                _update_sensor,
            )
        )
