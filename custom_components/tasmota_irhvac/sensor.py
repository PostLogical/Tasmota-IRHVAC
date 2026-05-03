"""Companion sensor entities for the Tasmota IRHVAC PI controller.

All sensors are `CoordinatorEntity[TasmotaIRHVACCoordinator]` subclasses
— refresh is automatic when the controller publishes a new TickOutput.
"""

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
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DATA_KEY
from .pi.coordinator import TasmotaIRHVACCoordinator

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
        key="user_setpoint",
        translation_key="user_setpoint",
        climate_attr="desired_temp_celsius",
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
    TasmotaIrhvacPISensorDescription(
        key="tau_estimate",
        translation_key="tau_estimate",
        climate_attr="tau_estimate",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
        suggested_display_precision=1,
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

    coordinator = climate_entity.coordinator
    if coordinator is None:  # pragma: no cover — unreachable: PI-enabled entities always have a coordinator
        return

    sensors: list[SensorEntity] = [
        TasmotaIrhvacPISensor(
            coordinator=coordinator,
            climate_entity=climate_entity,
            description=desc,
        )
        for desc in PI_SENSOR_DESCRIPTIONS
    ]
    sensors.append(
        TasmotaIrhvacHealthSensor(
            coordinator=coordinator, climate_entity=climate_entity,
        )
    )
    sensors.append(
        TasmotaIrhvacLearningSensor(
            coordinator=coordinator, climate_entity=climate_entity,
        )
    )
    sensors.append(
        TasmotaIrhvacGreyboxSensor(
            coordinator=coordinator, climate_entity=climate_entity,
        )
    )
    async_add_entities(sensors)


class TasmotaIrhvacPISensor(
    CoordinatorEntity[TasmotaIRHVACCoordinator], SensorEntity,
):
    """Sensor that mirrors a PI controller value via the coordinator."""

    _attr_has_entity_name = True
    entity_description: TasmotaIrhvacPISensorDescription

    def __init__(
        self,
        coordinator: TasmotaIRHVACCoordinator,
        climate_entity: TasmotaIrhvac,
        description: TasmotaIrhvacPISensorDescription,
    ) -> None:
        """Initialize the PI sensor."""
        super().__init__(coordinator)
        self.entity_description = description
        self._climate = climate_entity
        self._attr_unique_id = f"{climate_entity.unique_id}_{description.key}"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group sensor with climate entity."""
        return self._climate.device_info

    @property
    def native_value(self) -> float | str | None:
        """Read current value from the PI controller.

        Reads private PIController attributes via `climate_attr` —
        these stay updated by the tick path; the coordinator just
        triggers refresh.
        """
        pi = self._climate._pi
        if pi is None:
            return None
        return getattr(pi, self.entity_description.climate_attr, None)

    @property
    def available(self) -> bool:
        """Sensor is available when the climate entity is available."""
        return bool(self._climate.available)


class TasmotaIrhvacHealthSensor(
    CoordinatorEntity[TasmotaIRHVACCoordinator], SensorEntity,
):
    """Sensor that evaluates PI controller health status.

    Reads typed `coordinator.data.health` for state; falls back to the
    legacy getter for the rich attribute set (which already reads from
    `last_tick` since Stage 5c).
    """

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["OK", "Warning", "Critical", "Disabled"]
    _attr_translation_key = "health"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: TasmotaIRHVACCoordinator,
        climate_entity: TasmotaIrhvac,
    ) -> None:
        """Initialize the health sensor."""
        super().__init__(coordinator)
        self._climate = climate_entity
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
        """Return current health state from the typed snapshot."""
        return self.coordinator.data.health.state

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


class TasmotaIrhvacGreyboxSensor(
    CoordinatorEntity[TasmotaIRHVACCoordinator], SensorEntity,
):
    """Sensor that reports grey-box model identification state."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["Failed", "Learning", "Adequate", "Good", "Degraded"]
    _attr_translation_key = "greybox"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: TasmotaIRHVACCoordinator,
        climate_entity: TasmotaIrhvac,
    ) -> None:
        """Initialize the grey-box sensor."""
        super().__init__(coordinator)
        self._climate = climate_entity
        self._attr_unique_id = f"{climate_entity.unique_id}_greybox"

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
        """Return current grey-box model state from the typed snapshot."""
        return self.coordinator.data.greybox.state

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return grey-box model details as attributes."""
        pi = self._pi
        if pi is None:
            return {}
        state = pi.get_greybox_state()
        attrs = dict(state)
        # Convert gate_details dict to semicolon-separated string for HA display.
        if "gate_details" in attrs and isinstance(attrs["gate_details"], dict):
            attrs["gate_details"] = "; ".join(
                f"{k}={v}" for k, v in attrs["gate_details"].items()
            )
        return attrs

    @property
    def available(self) -> bool:
        """Sensor is available when the climate entity is available."""
        return bool(self._climate.available)


class TasmotaIrhvacLearningSensor(
    CoordinatorEntity[TasmotaIRHVACCoordinator], SensorEntity,
):
    """Sensor that reports learning state of the PI controller."""

    _attr_has_entity_name = True
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["Observing", "Learning", "Optimizing", "Optimized"]
    _attr_translation_key = "learning"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: TasmotaIRHVACCoordinator,
        climate_entity: TasmotaIrhvac,
    ) -> None:
        """Initialize the learning sensor."""
        super().__init__(coordinator)
        self._climate = climate_entity
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
        """Return current learning state from the typed snapshot."""
        return self.coordinator.data.learning.state

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
