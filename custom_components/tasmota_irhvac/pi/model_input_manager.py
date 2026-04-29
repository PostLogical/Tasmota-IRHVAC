"""Model input sensor management: reading, lag filtering, feature vector building.

Callers resolve HA entity states and pass them in.  Uses HA temperature
conversion utilities but no HA runtime state.  Owns the runtime state
(current values, filtered values, outdoor temp) but not the config-derived
RLS setup (seeds, clamps, feature scales stay in PIController for RLS init).
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from dataclasses import dataclass
from typing import Any

from homeassistant.util.unit_conversion import TemperatureConverter
from homeassistant.const import UnitOfTemperature

_LOGGER = logging.getLogger(__name__)

# States that map to 1.0 for non-numeric entities (binary sensors, climate, etc.)
_ACTIVE_STATES = frozenset({
    "on", "heat", "cool", "dry", "fan_only",
    "heating", "cooling", "burning", "igniting",
})

# Time-of-day feature names (automatic sinusoidal features for diurnal
# decorrelation — always present, no entity_id or config needed).
TOD_FEATURE_NAMES: tuple[str, str] = ("sin_hour", "cos_hour")
N_TOD_FEATURES: int = len(TOD_FEATURE_NAMES)
_TWO_PI_OVER_24 = 2.0 * math.pi / 24.0


def tod_features(wall_time: float | None) -> tuple[float, float]:
    """Compute sin/cos of local fractional hour from UTC epoch seconds.

    Returns (sin(2π·hour/24), cos(2π·hour/24)) using the system's local
    timezone, consistent with batch_learning's wall-hour derivation.
    """
    if wall_time is None or wall_time <= 0:
        return 0.0, 0.0
    dt = _dt.datetime.fromtimestamp(wall_time)
    hour_frac = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    angle = _TWO_PI_OVER_24 * hour_frac
    return math.sin(angle), math.cos(angle)


# ── Feature layout: single source of truth for feature vector structure ───


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one element of the feature vector.

    Seeds and clamps are in **internal β space** (already negated from
    user-facing seed convention where positive = warms room).
    """

    name: str
    role: str  # "intercept", "outdoor_delta", "time_of_day", "model_input"
    seed_heat: float
    seed_cool: float
    clamp: tuple[float, float] | None
    scale: float
    frozen_at_init: bool


class FeatureLayout:
    """Single source of truth for feature vector layout.

    Built once per PIController lifetime; every consumer reads from here
    instead of reconstructing names/seeds/clamps/scales ad-hoc.
    """

    def __init__(self, specs: list[FeatureSpec]) -> None:
        self._specs = list(specs)

    @property
    def n(self) -> int:
        """Total feature count including intercept."""
        return len(self._specs)

    @property
    def n_inputs(self) -> int:
        """Feature count excluding intercept (for RLSModel n_inputs)."""
        return len(self._specs) - 1

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def seeds(self, mode: str) -> list[float]:
        """Return seed list in internal β space for the given mode."""
        if mode == "cool":
            return [s.seed_cool for s in self._specs]
        return [s.seed_heat for s in self._specs]

    def clamps(self) -> list[tuple[float, float] | None]:
        return [s.clamp for s in self._specs]

    @property
    def scales(self) -> list[float]:
        return [s.scale for s in self._specs]

    def role(self, index: int) -> str:
        if 0 <= index < len(self._specs):
            return self._specs[index].role
        return "other"

    def frozen_mask(self) -> list[bool]:
        return [s.frozen_at_init for s in self._specs]

    @property
    def model_input_start(self) -> int:
        """First index that is a user-configured model input."""
        for i, s in enumerate(self._specs):
            if s.role == "model_input":
                return i
        return len(self._specs)

    def model_input_index(self, feature_index: int) -> int:
        """Map a feature vector index to its position in the model_inputs config list.

        Raises IndexError if the feature_index is not a model_input.
        """
        offset = feature_index - self.model_input_start
        if offset < 0:
            raise IndexError(f"Feature {feature_index} is not a model_input")
        return offset


# ── Feature layout: single source of truth for feature vector structure ───


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one element of the feature vector.

    Seeds and clamps are in **internal β space** (already negated from
    user-facing seed convention where positive = warms room).
    """

    name: str
    role: str  # "intercept", "outdoor_delta", "time_of_day", "model_input"
    seed_heat: float
    seed_cool: float
    clamp: tuple[float, float] | None
    scale: float
    frozen_at_init: bool


class FeatureLayout:
    """Single source of truth for feature vector layout.

    Built once per PIController lifetime; every consumer reads from here
    instead of reconstructing names/seeds/clamps/scales ad-hoc.
    """

    def __init__(self, specs: list[FeatureSpec]) -> None:
        self._specs = list(specs)

    @property
    def n(self) -> int:
        """Total feature count including intercept."""
        return len(self._specs)

    @property
    def n_inputs(self) -> int:
        """Feature count excluding intercept (for RLSModel n_inputs)."""
        return len(self._specs) - 1

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def seeds(self, mode: str) -> list[float]:
        """Return seed list in internal β space for the given mode."""
        if mode == "cool":
            return [s.seed_cool for s in self._specs]
        return [s.seed_heat for s in self._specs]

    def clamps(self) -> list[tuple[float, float] | None]:
        return [s.clamp for s in self._specs]

    @property
    def scales(self) -> list[float]:
        return [s.scale for s in self._specs]

    def role(self, index: int) -> str:
        if 0 <= index < len(self._specs):
            return self._specs[index].role
        return "other"

    def frozen_mask(self) -> list[bool]:
        return [s.frozen_at_init for s in self._specs]

    @property
    def model_input_start(self) -> int:
        """First index that is a user-configured model input."""
        for i, s in enumerate(self._specs):
            if s.role == "model_input":
                return i
        return len(self._specs)

    def model_input_index(self, feature_index: int) -> int:
        """Map a feature vector index to its position in the model_inputs config list.

        Raises IndexError if the feature_index is not a model_input.
        """
        offset = feature_index - self.model_input_start
        if offset < 0:
            raise IndexError(f"Feature {feature_index} is not a model_input")
        return offset



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
        """Total non-intercept feature count: outdoor_delta + model_inputs + ToD."""
        return 1 + len(self._model_inputs) + N_TOD_FEATURES

    def update_outdoor_temp(self, state_value: str, unit: str) -> None:
        """Update outdoor temperature from a sensor reading, converting to °C.

        Sets to None on parse failure (defense-in-depth) rather than
        silently retaining a stale value.  The primary unavailability
        check is in _async_outdoor_temp_changed, but this catches edge
        cases like NaN strings or corrupted state values.
        """
        try:
            temp = float(state_value)
            self.outdoor_temp = TemperatureConverter.convert(
                temp, unit, UnitOfTemperature.CELSIUS
            )
        except (ValueError, TypeError):
            self.outdoor_temp = None

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

    def build_feature_vector(
        self, outdoor_delta: float, wall_time: float | None = None,
    ) -> list[float]:
        """Build feature vector [1, outdoor_delta, inputs..., sin_hour, cos_hour]."""
        x: list[float] = [1.0, outdoor_delta]
        for i in range(len(self._model_inputs)):
            x.append(self.filtered[i])
        sin_h, cos_h = tod_features(wall_time)
        x.append(sin_h)
        x.append(cos_h)
        return x

    def build_feature_names(self) -> list[str]:
        """Return ordered feature names matching build_feature_vector layout."""
        names: list[str] = ["intercept", "outdoor_delta"]
        for m_input in self._model_inputs:
            names.append(m_input.get("name", f"input_{len(names) - 2}"))
        names.extend(TOD_FEATURE_NAMES)
        return names

    def build_named_features(
        self, outdoor_delta: float, wall_time: float | None = None,
    ) -> dict[str, float]:
        """Build feature dict {name: value} for Observation storage."""
        features: dict[str, float] = {
            "intercept": 1.0,
            "outdoor_delta": outdoor_delta,
        }
        for i, m_input in enumerate(self._model_inputs):
            name = m_input.get("name", f"input_{i}")
            features[name] = self.filtered[i]
        sin_h, cos_h = tod_features(wall_time)
        features["sin_hour"] = sin_h
        features["cos_hour"] = cos_h
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
