"""Model input sensor management: reading, lag filtering, feature vector building.

Callers resolve HA entity states and pass them in.  Uses HA temperature
conversion utilities but no HA runtime state.  Owns the runtime state
(current values, filtered values, outdoor temp) but not the config-derived
RLS setup (seeds, clamps, feature scales stay in PIController for RLS init).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from homeassistant.util.unit_conversion import TemperatureConverter
from homeassistant.const import UnitOfTemperature

_LOGGER = logging.getLogger(__name__)

# States that map to 1.0 for non-numeric entities (binary sensors, climate, etc.)
_ACTIVE_STATES = frozenset({
    "on", "heat", "cool", "dry", "fan_only",
    "heating", "cooling", "burning", "igniting",
})



class ModelInputManager:
    """Manages model input sensor values, lag filtering, and feature vector building."""

    def __init__(
        self,
        model_inputs: list[dict[str, Any]],
        outdoor_temp_sensor: str | None,
    ) -> None:
        self._model_inputs: list[dict[str, Any]] = model_inputs
        self._outdoor_temp_sensor: str | None = outdoor_temp_sensor
        self.values: list[float] = [0.0] * len(model_inputs)
        self.filtered: list[float] = [0.0] * len(model_inputs)
        # Pre-delta sensor readings in °C for observation storage.
        # For delta_from_room inputs this is the °C-converted absolute temp;
        # for all others it mirrors self.values.  Batch WLS applies
        # delta_from_room at regression time using the observation's current_c.
        self._raw_for_obs: list[float] = [0.0] * len(model_inputs)
        self.outdoor_temp: float | None = None

    @property
    def model_inputs(self) -> list[dict[str, Any]]:
        """Config dicts for each model input."""
        return self._model_inputs

    @property
    def outdoor_temp_sensor(self) -> str | None:
        return self._outdoor_temp_sensor

    @property
    def n_model_inputs(self) -> int:
        """Total feature count: 1 (outdoor_delta) + len(model_inputs)."""
        return 1 + len(self._model_inputs)

    def update_outdoor_temp(self, state_value: str, unit: str) -> None:
        """Update outdoor temperature from a sensor reading, converting to °C."""
        try:
            temp = float(state_value)
            self.outdoor_temp = TemperatureConverter.convert(
                temp, unit, UnitOfTemperature.CELSIUS
            )
        except (ValueError, TypeError):
            pass

    def read_values(
        self,
        entity_states: dict[str, tuple[str, bool, str | None]],
        room_temp_c: float | None = None,
    ) -> None:
        """Update model input values from resolved entity states.

        Args:
            entity_states: Map of entity_id → (state_string, is_available, unit).
                PIController resolves these from hass.states before calling.
                Also includes gate entity states when gate_entity is configured.
                ``unit`` is the ``unit_of_measurement`` attribute (may be None).
            room_temp_c: Current room temperature in °C.  Required for
                delta_from_room inputs — the stored value becomes
                (entity_temp_c − room_temp_c).
        """
        for i, m_input in enumerate(self._model_inputs):
            entity_id = m_input.get("entity_id", "")
            if not entity_id:
                continue
            if entity_id not in entity_states or not entity_states[entity_id][1]:
                _LOGGER.debug(
                    "Model input '%s' (%s) unavailable, using last value %.2f",
                    m_input.get("name", "?"), entity_id, self.values[i],
                )
                continue
            state_str = entity_states[entity_id][0]
            prev_value = self.values[i]
            try:
                self.values[i] = float(state_str)
            except (ValueError, TypeError):
                self.values[i] = 1.0 if state_str in _ACTIVE_STATES else 0.0

            # Delta-from-room: convert entity temp to °C and subtract room temp.
            # _raw_for_obs gets the °C absolute temp (for batch WLS);
            # self.values gets the delta (for online RLS / lag filter).
            if m_input.get("delta_from_room") and room_temp_c is not None:
                unit = entity_states[entity_id][2]
                if unit not in (UnitOfTemperature.CELSIUS, UnitOfTemperature.FAHRENHEIT):
                    # Unit unknown — keep previous value rather than corrupt
                    # with a wrong-unit subtraction.
                    self.values[i] = prev_value
                    _LOGGER.warning(
                        "Model input '%s' (%s): delta_from_room skipped — "
                        "unit_of_measurement is %r",
                        m_input.get("name", "?"), entity_id, unit,
                    )
                    continue
                entity_temp_c = TemperatureConverter.convert(
                    self.values[i], unit, UnitOfTemperature.CELSIUS
                )
                self._raw_for_obs[i] = entity_temp_c
                self.values[i] = entity_temp_c - room_temp_c
            else:
                self._raw_for_obs[i] = self.values[i]

            # Enabled toggle: force value to zero when disabled.
            if not m_input.get("input_enabled", True):
                self.values[i] = 0.0

            # Gate: force value to zero when gate entity is inactive (or
            # active if inverted).  Gate check runs after delta_from_room so
            # the computed delta is what gets zeroed, not the raw temperature.
            gate_id = m_input.get("gate_entity", "")
            if gate_id:
                if gate_id not in entity_states or not entity_states[gate_id][1]:
                    # Gate entity unavailable — keep last value (don't zero).
                    _LOGGER.debug(
                        "Gate entity '%s' for input '%s' unavailable, keeping value",
                        gate_id, m_input.get("name", "?"),
                    )
                else:
                    gate_state = entity_states[gate_id][0]
                    gate_active = gate_state in _ACTIVE_STATES
                    gate_invert = m_input.get("gate_invert", False)
                    gate_open = gate_active != gate_invert
                    if not gate_open:
                        self.values[i] = 0.0

    def any_unavailable(
        self,
        entity_states: dict[str, tuple[str, bool, str | None]],
    ) -> bool:
        """Check if any enabled model input entity is currently unavailable.

        Disabled inputs (input_enabled=False) are skipped — their value is
        forced to zero regardless, so unavailability doesn't affect data quality.
        """
        for m_input in self._model_inputs:
            if not m_input.get("input_enabled", True):
                continue
            entity_id = m_input.get("entity_id", "")
            if not entity_id:
                continue
            if entity_id not in entity_states or not entity_states[entity_id][1]:
                return True
        return False

    def update_lag_filters(self, dt_seconds: float) -> None:
        """Apply exponential lag filters to model input values."""
        for i, m_input in enumerate(self._model_inputs):
            tau = float(m_input.get("lag_tau", 0))
            raw = self.values[i]
            if tau > 0 and dt_seconds > 0:
                alpha = 1.0 - math.exp(-dt_seconds / tau)
                self.filtered[i] = alpha * raw + (1.0 - alpha) * self.filtered[i]
            else:
                self.filtered[i] = raw

    def build_feature_vector(self, outdoor_delta: float) -> list[float]:
        """Build feature vector [1, outdoor_delta, input1_filtered, ...] for RLS."""
        x: list[float] = [1.0, outdoor_delta]
        for i in range(len(self._model_inputs)):
            x.append(self.filtered[i])
        return x

    def build_feature_names(self) -> list[str]:
        """Return ordered feature names matching build_feature_vector layout."""
        names: list[str] = ["intercept", "outdoor_delta"]
        for m_input in self._model_inputs:
            names.append(m_input.get("name", f"input_{len(names) - 2}"))
        return names

    def build_named_features(self, outdoor_delta: float) -> dict[str, float]:
        """Build feature dict {name: value} for Observation storage."""
        features: dict[str, float] = {
            "intercept": 1.0,
            "outdoor_delta": outdoor_delta,
        }
        for i, m_input in enumerate(self._model_inputs):
            name = m_input.get("name", f"input_{i}")
            features[name] = self.filtered[i]
        return features

    def build_raw_readings(self) -> dict[str, float]:
        """Build raw sensor readings dict: entity_id → current value in °C.

        Returns pre-EMA, pre-delta values for each model input, keyed by
        entity_id.  For delta_from_room inputs this is the °C-converted
        absolute temperature — batch WLS applies the delta transform at
        regression time using the observation's current_c.
        """
        readings: dict[str, float] = {}
        for i, m_input in enumerate(self._model_inputs):
            entity_id = m_input.get("entity_id", "")
            if entity_id:
                readings[entity_id] = self._raw_for_obs[i]
        return readings

    def get_lag_states(self) -> dict[str, float]:
        """Get current lag filter states for persistence."""
        return {
            m_input.get("name", str(i)): self.filtered[i]
            for i, m_input in enumerate(self._model_inputs)
        }

    def restore_lag_states(self, states: dict[str, float]) -> None:
        """Restore lag filter states from persistence."""
        for i, m_input in enumerate(self._model_inputs):
            key = m_input.get("name", str(i))
            if key in states:
                self.filtered[i] = float(states[key])
