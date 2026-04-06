"""Vendor-agnostic PI + feedforward controller for IRHVAC climate entities.

Composed object (not a mixin). The climate entity creates a PIController instance
and calls its hook methods at the appropriate points. This avoids MRO issues
and minimizes changes to the upstream-derived climate.py.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State
    from homeassistant.helpers.event import CALLBACK_TYPE

    from .climate import TasmotaIrhvac

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.restore_state import ExtraStoredData
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.util.unit_conversion import TemperatureConverter

import math

from .const import (
    ATTR_DESIRED_TEMP,
    ATTR_FF_OFFSET,
    ATTR_HP_SETPOINT,
    ATTR_PI_INTEGRAL,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PI_DEADBAND,
    CONF_PI_ENABLED,
    CONF_PI_FF_COOL_REFERENCE,
    CONF_PI_FF_COOL_SLOPE,
    CONF_PI_FF_HEAT_REFERENCE,
    CONF_PI_FF_HEAT_SLOPE,
    CONF_PI_KD,
    CONF_PI_KD_FILTER_N,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_MIN_INTERVAL,
    CONF_PI_MODEL_INPUTS,
    CONF_PI_SETPOINT_WEIGHT,
    DEFAULT_PI_DEADBAND,
    DEFAULT_PI_ENABLED,
    DEFAULT_PI_FF_COOL_REFERENCE,
    DEFAULT_PI_FF_COOL_SLOPE,
    DEFAULT_PI_FF_HEAT_REFERENCE,
    DEFAULT_PI_FF_HEAT_SLOPE,
    DEFAULT_PI_KD,
    DEFAULT_PI_KD_FILTER_N,
    DEFAULT_PI_KI,
    DEFAULT_PI_KP,
    DEFAULT_PI_MIN_INTERVAL,
    DEFAULT_PI_SETPOINT_WEIGHT,
    SIGNAL_FF_SUPPRESS_UPDATE,
    SIGNAL_PI_UPDATE,
)

from .const import DEFAULT_RLS_P_INIT
from .rls_model import RLSModel

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class PIExtraStoredData(ExtraStoredData):
    """PI controller data persisted via RestoreEntity's ExtraStoredData mechanism.

    Stores RLS models, integral, desired_temp, and hp_setpoint so they survive
    restarts without bloating the recorder DB on every state write.
    """

    pi_integral: float
    desired_temp: float | None
    hp_setpoint: float | None
    integral_convergence: float = 0.0
    rls_heat_model: dict = dataclasses.field(default_factory=dict)
    rls_cool_model: dict = dataclasses.field(default_factory=dict)
    lag_filter_states: dict = dataclasses.field(default_factory=dict)
    heat_seeds_at_learn: list = dataclasses.field(default_factory=list)
    cool_seeds_at_learn: list = dataclasses.field(default_factory=list)
    ki_at_save: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "pi_integral": self.pi_integral,
            "desired_temp": self.desired_temp,
            "hp_setpoint": self.hp_setpoint,
            "integral_convergence": self.integral_convergence,
            "rls_heat_model": self.rls_heat_model,
            "rls_cool_model": self.rls_cool_model,
            "lag_filter_states": self.lag_filter_states,
            "heat_seeds_at_learn": self.heat_seeds_at_learn,
            "cool_seeds_at_learn": self.cool_seeds_at_learn,
            "ki_at_save": self.ki_at_save,
        }

    @classmethod
    def from_dict(cls, restored: dict[str, Any]) -> Self | None:
        """Deserialize from stored dict.

        Gracefully ignores legacy bucket fields (ff_heat_buckets, ff_cool_buckets,
        ff_bucket_observation_counts) from pre-removal stored data.
        """
        try:
            return cls(
                pi_integral=float(restored["pi_integral"]),
                desired_temp=restored.get("desired_temp"),
                hp_setpoint=restored.get("hp_setpoint"),
                integral_convergence=float(restored.get("integral_convergence", 0.0)),
                rls_heat_model=restored.get("rls_heat_model", {}),
                rls_cool_model=restored.get("rls_cool_model", {}),
                lag_filter_states=restored.get("lag_filter_states", {}),
                heat_seeds_at_learn=restored.get("heat_seeds_at_learn", []),
                cool_seeds_at_learn=restored.get("cool_seeds_at_learn", []),
                ki_at_save=float(restored.get("ki_at_save", 0.0)),
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            return None


class PIController:
    """PI + feedforward temperature controller for IRHVAC climate entities.

    Usage in climate entity:
        __init__:           self._pi = PIController(self, config)
        async_added_to_hass:     await self._pi.async_added_to_hass(old_state)
        async_will_remove:       self._pi.async_will_remove_from_hass()
        async_set_temperature:   await self._pi.set_temperature(temp, hvac_mode)
        _handle_state_payload:   await self._pi.handle_state_update(payload, ir_received=...)
        _async_sensor_changed:   await self._pi.sensor_changed(was_none)
        _get_ir_temp:            return self._pi.get_ir_temp()
        extra_state_attributes:  attrs.update(self._pi.get_extra_state_attributes())
        hvac_modes:              return self._pi.filter_hvac_modes(modes)
        async_set_hvac_mode:     if self._pi.should_reject_hvac_mode(mode): return
        async_write_ha_state:    self._pi.fire_dispatcher()

    Vendor subclasses use: pi_pause(), pi_resume(), pi_reset_integral()
    """

    def __init__(self, entity: TasmotaIrhvac, config: dict[str, Any]) -> None:
        """Initialize PI controller.

        Args:
            entity: The climate entity this controller is attached to.
            config: Merged config dict (entry.data + entry.options).
        """
        self._entity = entity

        # Convert entity temp limits to °C for internal PI math
        self._min_temp_c: float = TemperatureConverter.convert(
            entity._min_temp, entity._attr_temperature_unit, UnitOfTemperature.CELSIUS
        )
        self._max_temp_c: float = TemperatureConverter.convert(
            entity._max_temp, entity._attr_temperature_unit, UnitOfTemperature.CELSIUS
        )

        # PID controller config
        self._pi_enabled: bool = config.get(CONF_PI_ENABLED, DEFAULT_PI_ENABLED)
        self._pi_kp: float = config.get(CONF_PI_KP, DEFAULT_PI_KP)
        self._pi_ki: float = config.get(CONF_PI_KI, DEFAULT_PI_KI)
        self._pi_kd: float = config.get(CONF_PI_KD, DEFAULT_PI_KD)
        self._pi_kd_filter_n: float = config.get(CONF_PI_KD_FILTER_N, DEFAULT_PI_KD_FILTER_N)
        self._pi_min_interval: float = config.get(CONF_PI_MIN_INTERVAL, DEFAULT_PI_MIN_INTERVAL)
        # Control parameters are stored in °C always — read directly, no conversion
        self._pi_deadband: float = config.get(CONF_PI_DEADBAND, DEFAULT_PI_DEADBAND)
        self._ff_heat_reference: float = config.get(CONF_PI_FF_HEAT_REFERENCE, DEFAULT_PI_FF_HEAT_REFERENCE)
        self._ff_cool_reference: float = config.get(CONF_PI_FF_COOL_REFERENCE, DEFAULT_PI_FF_COOL_REFERENCE)
        self._pi_setpoint_weight: float = config.get(CONF_PI_SETPOINT_WEIGHT, DEFAULT_PI_SETPOINT_WEIGHT)

        # Feedforward config
        self._outdoor_temp_sensor: str | None = config.get(CONF_OUTDOOR_TEMP_SENSOR)
        self._ff_heat_slope: float = config.get(CONF_PI_FF_HEAT_SLOPE, DEFAULT_PI_FF_HEAT_SLOPE)
        self._ff_cool_slope: float = config.get(CONF_PI_FF_COOL_SLOPE, DEFAULT_PI_FF_COOL_SLOPE)

        # Learning suppression state (manual service + model input suppress_learning flags)
        self._manual_ff_suppress: bool = False
        self._manual_ff_suppress_reason: str = ""
        self._disturbance_suppress_active: bool = False
        self._disturbance_active_suppressors: list[str] = []

        # PI controller state
        # _desired_temp is in entity unit (system); _hp_setpoint is always °C (for IR)
        self._desired_temp: float | None = entity._attr_target_temperature
        self._hp_setpoint: float | None = (
            TemperatureConverter.convert(
                entity._attr_target_temperature,
                entity._attr_temperature_unit,
                UnitOfTemperature.CELSIUS,
            )
            if entity._attr_target_temperature is not None
            else None
        )
        self._pi_integral: float = 0.0
        self._pi_timer_unsub: CALLBACK_TYPE | None = None
        self._ff_offset: float = 0.0
        self._pi_tick_running: bool = False
        self._last_send_ir_time: float = 0.0
        self._last_setpoint_change_time: float = 0.0
        self._ff_settled_ticks: int = 0
        self._sensor_unavailable: bool = False
        self._sensor_recovery_pending: bool = False
        self._sensor_recovery_unsub: CALLBACK_TYPE | None = None
        self._pi_paused: bool = False
        self._pi_last_tick_time: float = 0.0
        self._pi_last_error: float = 0.0
        self._pi_d_filtered: float = 0.0    # Filtered derivative term
        self._pi_last_measurement: float | None = None  # Previous temperature measurement for derivative

        # Supplemental heat source selector/override control
        self._supplemental_sources: list[dict[str, Any]] = config.get("pi_supplemental_sources", [])
        self._tracking_mode: bool = False          # True = HP defers to supplemental
        self._tracking_sources: list[str] = []  # Names of active overriding sources
        self._supplemental_failure_start: float | None = None
        self._supplemental_assist_active: bool = False
        self._supplemental_last_override: bool = False  # Edge detection

        # Model inputs (replaces disturbance inputs for RLS)
        # Each: {"name": str, "entity_id": str, "seed_heat": float, "seed_cool": float,
        #         "clamp_min": float, "clamp_max": float, "lag_tau": float (seconds)}
        self._model_inputs: list[dict[str, Any]] = list(config.get(CONF_PI_MODEL_INPUTS, []))

        # Auto-generate model inputs from supplemental sources with auto_model_input=true.
        # These use the supplemental's climate entity as a binary signal (heat/cool=1, else=0).
        # The PI controller reads the entity state each tick to update the value.
        self._supplemental_auto_inputs: list[dict[str, Any]] = []
        for source in self._supplemental_sources:
            if not source.get("auto_model_input", True):
                continue
            entity_id = source.get("entity_id", "")
            name = source.get("name", entity_id)
            # Check if user already has a manual model input for this entity
            existing = any(m.get("entity_id") == entity_id for m in self._model_inputs)
            if existing:
                _LOGGER.debug("Supplemental %s: skipping auto model input (manual input exists)", name)
                continue
            auto_input = {
                "name": f"{name} (auto)",
                "entity_id": entity_id,
                "seed_heat": float(source.get("seed_heat", -3.0)),
                "seed_cool": float(source.get("seed_cool", 0.0)),
                "lag_tau": 0,
                "suppress_learning": True,  # Learning deferred per research
                "_auto_supplemental": True,  # Internal flag for signal generation
            }
            self._model_inputs.append(auto_input)
            self._supplemental_auto_inputs.append(auto_input)
        # Outdoor delta is always the first model input (index 1, after intercept)
        # Other model inputs follow in order of _model_inputs list
        self._n_model_inputs = 1 + len(self._model_inputs)  # outdoor_delta + configured inputs

        # Build seed coefficients and clamps
        # Index 0: intercept (seed 0)
        # Index 1: outdoor_delta (seed = configured slope)
        # Index 2+: model inputs in order
        self._heat_seeds = [0.0, self._ff_heat_slope]
        self._cool_seeds = [0.0, -self._ff_cool_slope]  # Negative: hotter outdoor → lower HP setpoint
        self._rls_clamps = [None, (0.0, 2.0)]  # Intercept unclamped, outdoor_delta positive
        for m_input in self._model_inputs:
            self._heat_seeds.append(float(m_input.get("seed_heat", 0.0)))
            self._cool_seeds.append(float(m_input.get("seed_cool", 0.0)))
            clamp_min = m_input.get("clamp_min")
            clamp_max = m_input.get("clamp_max")
            if clamp_min is not None and clamp_max is not None:
                self._rls_clamps.append((float(clamp_min), float(clamp_max)))
            else:
                self._rls_clamps.append(None)

        # Feature scales for balanced P initialization.
        # intercept=1.0, outdoor_delta typical ~10, model inputs ~0.5
        self._feature_scales = [1.0, 10.0]
        for m_input in self._model_inputs:
            self._feature_scales.append(float(m_input.get("typical_value", 0.5)))

        # RLS models (separate for heating and cooling)
        self._rls_heat = RLSModel(
            n_inputs=self._n_model_inputs,
            seed_coefficients=self._heat_seeds,
            coeff_clamps=self._rls_clamps,
            feature_scales=self._feature_scales,
        )
        self._rls_cool = RLSModel(
            n_inputs=self._n_model_inputs,
            seed_coefficients=self._cool_seeds,
            coeff_clamps=self._rls_clamps,
            feature_scales=self._feature_scales,
        )

        # Model input current values and lag filter states
        self._model_input_values = [0.0] * len(self._model_inputs)
        self._model_input_filtered = [0.0] * len(self._model_inputs)
        self._model_input_last_values = [0.0] * len(self._model_inputs)  # For lag filter

        # Outdoor temp state
        self._outdoor_temp: float | None = None


        # Integral convergence tracking (EMA of abs(integral) over ~24hr)
        self._integral_convergence: float = 0.0

        # Performance metrics (running accumulators, reset daily)
        self._itae_accumulator: float = 0.0      # Σ(tick * |effective_error|)
        self._itae_tick_count: int = 0         # ticks since last reset
        self._comfort_violation_hours: float = 0.0  # hours spent >1°C from setpoint
        self._setpoint_changes_today: int = 0  # setpoint change count since reset

        # RLS learning gate: track integral stability
        self._prev_integral_for_rls: float = 0.0

    # ── Shorthand entity access ──────────────────────────────────────

    @property
    def _hass(self) -> HomeAssistant:
        return self._entity.hass

    # ── ControllerHook Protocol properties ─────────────────────────

    @property
    def is_active(self) -> bool:
        return bool(self._pi_enabled)

    @property
    def desired_temp(self) -> float | None:
        return self._desired_temp

    @desired_temp.setter
    def desired_temp(self, value: float) -> None:
        self._desired_temp = value

    @property
    def is_tick_running(self) -> bool:
        return self._pi_tick_running

    # ── Lifecycle hooks (called by climate entity) ───────────────────

    async def async_added_to_hass(self, old_state: State | None = None) -> None:
        """Set up PI after entity is added to HA."""
        if not self._pi_enabled:
            return

        e = self._entity

        # Restore PI state — prefer ExtraStoredData, fall back to state attributes
        extra_data = await e.async_get_last_extra_data()
        if extra_data is not None:
            pi_data = PIExtraStoredData.from_dict(extra_data.as_dict())
            if pi_data is not None:
                self.restore_extra_stored_data(pi_data)
                _LOGGER.debug("PI: restored from ExtraStoredData")
        else:
            # Fall back to state attributes (migration from pre-ExtraStoredData versions)
            if old_state is None:
                old_state = await e.async_get_last_state()
            if old_state is not None:
                attrs = old_state.attributes
                if attrs.get(ATTR_PI_INTEGRAL) is not None:
                    self._pi_integral = float(attrs[ATTR_PI_INTEGRAL])
                if attrs.get(ATTR_DESIRED_TEMP) is not None:
                    self._desired_temp = float(attrs[ATTR_DESIRED_TEMP])
                if attrs.get(ATTR_HP_SETPOINT) is not None:
                    self._hp_setpoint = float(attrs[ATTR_HP_SETPOINT])
                _LOGGER.debug("PI: restored from state attributes (legacy)")

        # Fallback: sync with restored _attr_target_temperature
        if self._desired_temp is None and e._attr_target_temperature is not None:
            self._desired_temp = e._attr_target_temperature
        if self._hp_setpoint is None and e._attr_target_temperature is not None:
            self._hp_setpoint = TemperatureConverter.convert(
                e._attr_target_temperature,
                e.temperature_unit,
                UnitOfTemperature.CELSIUS,
            )

        # Register outdoor temp sensor
        if self._outdoor_temp_sensor:
            async_track_state_change_event(
                self._hass,
                self._outdoor_temp_sensor,
                self._async_outdoor_temp_changed,
            )
            outdoor_state = self._hass.states.get(self._outdoor_temp_sensor)
            if outdoor_state is not None:
                self._update_outdoor_temp(outdoor_state)

        # Register model input entities
        model_entity_ids = [
            m["entity_id"] for m in self._model_inputs if m.get("entity_id")
        ]
        if model_entity_ids:
            async_track_state_change_event(
                self._hass,
                model_entity_ids,
                self._async_model_input_changed,
            )
        # Read initial model input values
        self._read_model_input_values()

        # Compute initial feedforward offset
        if self._outdoor_temp is not None:
            is_heating = e._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
            outdoor_delta = self._ff_heat_reference - self._outdoor_temp if is_heating else self._outdoor_temp - self._ff_cool_reference
            outdoor_delta = max(0, outdoor_delta)
            x = self._build_feature_vector(outdoor_delta)
            rls = self._rls_heat if is_heating else self._rls_cool
            self._ff_offset = rls.predict(x)

        # Start fallback timer
        if e._temp_sensor:
            self._pi_timer_unsub = async_track_time_interval(
                self._hass,
                self._pi_tick,
                timedelta(seconds=self._pi_min_interval),
            )
            if e._attr_current_temperature is not None:
                # Delay initial tick by 5s to avoid burst of IR sends
                # on restart when all sensors come online together.
                @callback
                def _deferred_initial_tick(_now):
                    self._hass.async_create_task(self._pi_tick())
                async_call_later(self._hass, 5, _deferred_initial_tick)
            else:
                _LOGGER.debug("PI: skipping initial tick, waiting for sensor")

    def async_will_remove_from_hass(self) -> None:
        """Clean up PI timers."""
        if self._pi_timer_unsub:
            self._pi_timer_unsub()
            self._pi_timer_unsub = None
        if self._sensor_recovery_unsub:
            self._sensor_recovery_unsub()
            self._sensor_recovery_unsub = None

    # ── Hook methods (called by climate entity) ──────────────────────

    def get_extra_stored_data(self) -> PIExtraStoredData | None:
        """Return PI data for RestoreEntity's ExtraStoredData persistence."""
        if not self._pi_enabled:
            return None
        lag_states = {}
        for i, m_input in enumerate(self._model_inputs):
            lag_states[m_input.get("name", str(i))] = self._model_input_filtered[i]
        return PIExtraStoredData(
            pi_integral=self._pi_integral,
            desired_temp=self._desired_temp,
            hp_setpoint=self._hp_setpoint,
            integral_convergence=self._integral_convergence,
            rls_heat_model=self._rls_heat.as_dict(),
            rls_cool_model=self._rls_cool.as_dict(),
            lag_filter_states=lag_states,
            heat_seeds_at_learn=list(self._heat_seeds),
            cool_seeds_at_learn=list(self._cool_seeds),
            ki_at_save=self._pi_ki,
        )

    def restore_extra_stored_data(self, data: PIExtraStoredData) -> None:
        """Restore PI data from ExtraStoredData."""
        # Scale integral if ki changed since last save, so the I-term
        # contribution (ki * integral) stays the same magnitude.
        if data.ki_at_save > 0 and data.ki_at_save != self._pi_ki:
            scale = data.ki_at_save / self._pi_ki
            _LOGGER.info(
                "PI: scaling integral %.2f by %.2f (ki changed %.3f → %.3f)",
                data.pi_integral, scale, data.ki_at_save, self._pi_ki,
            )
            self._pi_integral = data.pi_integral * scale
        else:
            self._pi_integral = data.pi_integral
        if data.desired_temp is not None:
            self._desired_temp = data.desired_temp
        if data.hp_setpoint is not None:
            self._hp_setpoint = data.hp_setpoint
        self._integral_convergence = data.integral_convergence
        # Restore RLS models if available. Pass seed_coefficients so that
        # if model inputs changed (different vector length), new inputs get
        # seeded instead of zeroed.
        if data.rls_heat_model:
            self._rls_heat = RLSModel.from_dict(
                data.rls_heat_model, self._n_model_inputs,
                seed_coefficients=self._heat_seeds,
                coeff_clamps=self._rls_clamps,
            )
        if data.rls_cool_model:
            self._rls_cool = RLSModel.from_dict(
                data.rls_cool_model, self._n_model_inputs,
                seed_coefficients=self._cool_seeds,
                coeff_clamps=self._rls_clamps,
            )
        # Seed change detection: if user edited a seed since last save,
        # reset that coefficient to the new seed and increase its uncertainty.
        # Coefficients with unchanged seeds keep their learned values.
        self._apply_seed_changes(data.heat_seeds_at_learn, self._heat_seeds, self._rls_heat)
        self._apply_seed_changes(data.cool_seeds_at_learn, self._cool_seeds, self._rls_cool)
        # Restore lag filter states
        if data.lag_filter_states:
            for i, m_input in enumerate(self._model_inputs):
                key = m_input.get("name", str(i))
                if key in data.lag_filter_states:
                    self._model_input_filtered[i] = float(data.lag_filter_states[key])

    def _apply_seed_changes(
        self, old_seeds: list[float], new_seeds: list[float], rls_model: RLSModel
    ) -> None:
        """Detect seed changes and selectively reset affected coefficients.

        Compares seeds stored at last persist with current config seeds.
        For each coefficient where the seed changed, reset to new seed
        and increase P diagonal (high uncertainty → fast re-learning).
        """
        if not old_seeds:
            return  # No stored seeds (first run or pre-seed-detection data)
        for i in range(min(len(old_seeds), len(new_seeds), rls_model.n)):
            if abs(old_seeds[i] - new_seeds[i]) > 0.001:
                old_val = rls_model.beta[i]
                rls_model.beta[i] = new_seeds[i]
                rls_model.beta_seed[i] = new_seeds[i]
                # Reset P for this coefficient using scaled initialization
                scale = rls_model.feature_scales[i] if i < len(rls_model.feature_scales) else 1.0
                rls_model.P[i * rls_model.n + i] = DEFAULT_RLS_P_INIT / max(scale * scale, 0.01)
                _LOGGER.info(
                    "Seed changed for coefficient %d: %.4f → %.4f (learned was %.4f, reset)",
                    i, old_seeds[i], new_seeds[i], old_val,
                )

    async def set_temperature(
        self, temperature: float | None, hvac_mode: str | None = None
    ) -> None:
        """Handle temperature set when PI is active."""
        if temperature is None:
            return
        e = self._entity
        if hvac_mode is not None:
            await e.set_mode(hvac_mode)
        self._desired_temp = temperature
        # target_temperature property reads from desired_temp — no dual-write needed
        self._pi_integral = 0.0
        if e._attr_hvac_mode != HVACMode.OFF:
            e.power_mode = STATE_ON
        await self._pi_tick()
        e.async_schedule_update_ha_state()

    async def handle_state_update(
        self, payload: dict[str, Any], *, ir_received: bool = False
    ) -> None:
        """Handle MQTT state update. Called by climate entity after base state applied.

        Classifies the message using ir_received (transport fact from climate.py)
        combined with internal timing state to determine the appropriate action:

        Case 1 — Our echo: IrReceived within 2s of our send, or any non-IrReceived
                  message (stat echo / telemetry) where temp matches hp_setpoint.
                  Action: confirm hp_setpoint, no other changes.

        Case 2 — Physical remote: IrReceived WITHOUT a recent send (>2s).
                  Action: update desired_temp, zero integral if changed, tick.

        Case 3 — Telemetry mismatch: non-IrReceived message where reported temp
                  differs from hp_setpoint. Our IR send may have failed.
                  Action: resend current setpoint.

        State writes are handled by climate.py after this method returns.
        """
        if not self._pi_enabled or self._desired_temp is None or self._pi_paused:
            return
        e = self._entity
        if "Temp" not in payload or payload["Temp"] <= 0:
            return
        reported_temp = payload["Temp"]
        elapsed = time.monotonic() - self._last_send_ir_time

        _LOGGER.debug(
            "State update: reported=%s hp=%s desired=%s "
            "elapsed=%.1f ir_received=%s",
            reported_temp, self._hp_setpoint, self._desired_temp,
            elapsed, ir_received,
        )

        if ir_received and elapsed > 2.0:
            # Case 2: Physical remote — IrReceived with no recent send.
            # Someone used the physical remote. The reported Temp is their
            # desired room temperature in the IR protocol's unit.
            reported_c = TemperatureConverter.convert(
                reported_temp, e._ir_temp_unit, UnitOfTemperature.CELSIUS
            )
            desired_in_entity_unit = TemperatureConverter.convert(
                reported_temp, e._ir_temp_unit, e.temperature_unit
            )
            _LOGGER.info(
                "Physical remote detected: reported=%s (%.1f°C) -> desired=%s",
                reported_temp, reported_c, desired_in_entity_unit,
            )
            old_desired = self._desired_temp
            self._desired_temp = desired_in_entity_unit
            self._hp_setpoint = reported_temp
            # Only zero integral when the user actually changed the target
            if old_desired != desired_in_entity_unit:
                self._pi_integral = 0.0
            await self._pi_tick()

        elif ir_received:
            # Case 1a: Our self-echo — IrReceived within 2s of our send.
            # The IR receiver picked up our own transmission. Confirm hp_setpoint.
            _LOGGER.debug(
                "Self-echo (IrReceived, %.1fs since send): confirming hp=%s",
                elapsed, reported_temp,
            )
            self._hp_setpoint = reported_temp

        elif reported_temp != self._hp_setpoint:
            # Case 3: Telemetry mismatch — no IrReceived, temp differs.
            # Periodic telemetry (TelePeriod) or stat echo reports a different
            # temp than we expect. Our IR send may have failed. Resend.
            _LOGGER.warning(
                "Telemetry mismatch: reported=%s expected hp=%s. Resending.",
                reported_temp, self._hp_setpoint,
            )
            await e.send_ir()

        else:
            # Case 1b: Telemetry/stat confirmation — temp matches hp_setpoint.
            _LOGGER.debug(
                "State confirmed: reported=%s matches hp=%s",
                reported_temp, self._hp_setpoint,
            )

    async def sensor_changed(self, was_none: bool) -> None:
        """Handle temp sensor update."""
        await self._pi_async_sensor_changed(was_none=was_none)

    def get_ir_temp(self) -> float:
        """Return PI-computed setpoint for IR command.

        Safety clamp: no residential HVAC accepts temps outside 0-50°C.
        This catches corrupted _min_temp_c/_max_temp_c (e.g., from bad
        config migration) that would let the PI send impossible temps.
        """
        assert self._hp_setpoint is not None
        return max(0, min(50, round(self._hp_setpoint)))

    def get_extra_state_attributes(self) -> dict[str, Any]:
        """Return PI state attributes to merge into entity attributes."""
        if not self._pi_enabled:
            return {}
        # RLS coefficient names
        coeff_names = ["intercept", "outdoor_delta"]
        for m_input in self._model_inputs:
            coeff_names.append(m_input.get("name", "unknown"))

        # Coefficients are in °C — display as-is
        rls_heat_coeffs = {coeff_names[i]: round(self._rls_heat.beta[i], 4)
                          for i in range(min(len(coeff_names), len(self._rls_heat.beta)))}
        rls_cool_coeffs = {coeff_names[i]: round(self._rls_cool.beta[i], 4)
                          for i in range(min(len(coeff_names), len(self._rls_cool.beta)))}

        return {
            ATTR_HP_SETPOINT: self._hp_setpoint,
            ATTR_PI_INTEGRAL: round(self._pi_integral, 3),
            "d_term": round(self._pi_d_filtered, 3),
            "tracking_mode": self._tracking_mode,
            "tracking_sources": self._tracking_sources,
            "supplemental_assist": self._supplemental_assist_active,
            ATTR_DESIRED_TEMP: self._desired_temp,
            ATTR_FF_OFFSET: round(self._ff_offset, 2),
            "rls_heat_coefficients": rls_heat_coeffs,
            "rls_cool_coefficients": rls_cool_coeffs,
            "rls_observation_count": self._rls_heat.observation_count,
            "ff_learning_suppressed": self._disturbance_suppress_active,
            "integral_convergence": round(self._integral_convergence, 2),
        }

    def filter_hvac_modes(self, modes: list[Any]) -> list[Any]:
        """Filter out auto/heat_cool when PI is enabled."""
        if self._pi_enabled and modes:
            return [m for m in modes if m not in (HVACMode.AUTO, HVACMode.HEAT_COOL)]
        return modes

    def should_reject_hvac_mode(self, hvac_mode: str) -> bool:
        """Return True if PI should reject this HVAC mode."""
        return self._pi_enabled and hvac_mode in (HVACMode.AUTO, HVACMode.HEAT_COOL)

    def fire_dispatcher(self) -> None:
        """Fire dispatcher signal for companion PI sensors."""
        if self._pi_enabled and hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_PI_UPDATE.format(self._entity._config_entry_id),
            )

    # ── Public API (for vendor subclasses via entity._pi) ────────────

    def pi_pause(self) -> None:
        """Pause PI control (e.g., during vendor-specific preset modes)."""
        self._pi_paused = True

    def pi_resume(self) -> None:
        """Resume PI control after a pause."""
        self._pi_paused = False

    def pi_reset_integral(self) -> None:
        """Zero the integral (e.g., after mode changes that invalidate it)."""
        self._pi_integral = 0.0

    async def async_suppress_ff_learning(self, reason: str = "") -> None:
        """Manually suppress FF learning (service call handler)."""
        self._manual_ff_suppress = True
        self._manual_ff_suppress_reason = reason or ""
        _LOGGER.info("FF learning manually suppressed: %s", reason or "(no reason)")
        if hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._entity._config_entry_id),
            )

    async def async_resume_ff_learning(self) -> None:
        """Resume FF learning after manual suppression (service call handler)."""
        self._manual_ff_suppress = False
        self._manual_ff_suppress_reason = ""
        _LOGGER.info("FF learning manual suppress cleared")
        if hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._entity._config_entry_id),
            )

    async def async_reset_ff_seeds(self) -> None:
        """Reset feedforward RLS models to seed values from config."""
        # Reset RLS models to seed coefficients
        heat_seeds = [0.0, self._ff_heat_slope]
        cool_seeds = [0.0, -self._ff_cool_slope]  # Negative: hotter outdoor → lower HP setpoint
        for m_input in self._model_inputs:
            heat_seeds.append(float(m_input.get("seed_heat", 0.0)))
            cool_seeds.append(float(m_input.get("seed_cool", 0.0)))
        self._rls_heat.beta = heat_seeds + [0.0] * (self._rls_heat.n - len(heat_seeds))
        self._rls_cool.beta = cool_seeds + [0.0] * (self._rls_cool.n - len(cool_seeds))
        # Reset covariance to scaled initial uncertainty
        for i in range(self._rls_heat.n):
            for j in range(self._rls_heat.n):
                if i == j:
                    scale = self._feature_scales[i] if i < len(self._feature_scales) else 1.0
                    val = DEFAULT_RLS_P_INIT / max(scale * scale, 0.01)
                else:
                    val = 0.0
                self._rls_heat.P[i * self._rls_heat.n + j] = val
                self._rls_cool.P[i * self._rls_cool.n + j] = val
        self._rls_heat.observation_count = 0
        self._rls_cool.observation_count = 0
        self._pi_integral = 0.0
        _LOGGER.info("FF models reset to seed values, integral zeroed")
        self._entity.async_schedule_update_ha_state()

    # ── Supplemental Source Override/Selector ────────────────────────

    def _evaluate_supplemental_override(self, error_c: float, now_mono: float) -> bool:
        """Evaluate whether supplemental sources are active and update tracking mode.

        Implements override/selector control pattern:
        - When supplemental is active: HP enters tracking mode (computes but doesn't send IR)
        - When supplemental can't keep up: HP assists (sends IR alongside supplemental)
        - When supplemental stops: bumpless transfer (HP resumes with current integral)

        Returns True if the HP should send IR commands, False if tracking.
        """
        if not self._supplemental_sources:
            return True  # No supplemental sources configured, HP always active

        active_sources = []
        for source in self._supplemental_sources:
            entity_id = source.get("entity_id", "")
            if not entity_id:
                continue
            state = self._hass.states.get(entity_id)
            if state is None or state.state in ("unavailable", "unknown"):
                continue
            # Climate entity in heat or cool mode = supplemental is managing the room
            if state.state in ("heat", "cool"):
                active_sources.append(source.get("name", entity_id))

        was_tracking = self._tracking_mode

        if not active_sources:
            # No supplemental active → HP is in charge
            self._tracking_mode = False
            self._tracking_sources = []
            self._supplemental_failure_start = None
            self._supplemental_assist_active = False

            if was_tracking:
                _LOGGER.info(
                    "Supplemental override ended (sources: %s). HP resuming with integral=%.2f, setpoint=%s",
                    self._tracking_sources if self._tracking_sources else "none",
                    self._pi_integral, self._hp_setpoint,
                )
                # Bumpless transfer: clear hold timer so first IR send isn't blocked
                self._last_setpoint_change_time = 0.0
            return True  # HP active

        # At least one supplemental is active
        self._tracking_sources = active_sources

        # Failure detection: is the supplemental keeping up?
        min_threshold = min(
            s.get("failure_threshold", 900) for s in self._supplemental_sources
            if s.get("name", "") in active_sources
        ) if active_sources else 900

        # recovery_margin is stored in °C — read directly
        recovery_margin = min(
            s.get("recovery_margin", 0.3) for s in self._supplemental_sources
            if s.get("name", "") in active_sources
        ) if active_sources else 0.3

        if error_c > self._pi_deadband:
            # Room is below desired
            if self._supplemental_failure_start is None:
                self._supplemental_failure_start = now_mono
            time_below = now_mono - self._supplemental_failure_start
            if time_below >= min_threshold:
                if not self._supplemental_assist_active:
                    _LOGGER.info(
                        "Supplemental can't keep up (%.0fs below desired). HP assisting.",
                        time_below,
                    )
                self._supplemental_assist_active = True
        else:
            if error_c < -recovery_margin:
                # Room above desired + margin → supplemental caught up
                if self._supplemental_assist_active:
                    _LOGGER.info("Supplemental recovered. HP deferring again.")
                self._supplemental_assist_active = False
            self._supplemental_failure_start = None

        if self._supplemental_assist_active:
            self._tracking_mode = False  # HP active (assisting)
        else:
            self._tracking_mode = True   # HP tracking (deferred)

        if self._tracking_mode and not was_tracking:
            _LOGGER.info(
                "Supplemental override started: %s. HP entering tracking mode.",
                ", ".join(active_sources),
            )

        return not self._tracking_mode

    # ── PI Internals ──────────────────────────────────────────────────

    @callback
    def _update_outdoor_temp(self, state: State) -> None:
        """Update outdoor temperature from sensor state, converting to °C."""
        try:
            temp = float(state.state)
            unit = state.attributes.get("unit_of_measurement", UnitOfTemperature.CELSIUS)
            self._outdoor_temp = TemperatureConverter.convert(
                temp, unit, UnitOfTemperature.CELSIUS
            )
        except (ValueError, TypeError):
            pass

    @callback
    def _async_outdoor_temp_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle outdoor temperature sensor state changes."""
        new_state = event.data.get("new_state")
        if new_state is not None:
            self._update_outdoor_temp(new_state)

    def _build_feature_vector(self, outdoor_delta: float) -> list[float]:
        """Build the feature vector for RLS prediction/update.

        Returns [1, outdoor_delta, input1_filtered, input2_filtered, ...].
        """
        x: list[float] = [1.0, outdoor_delta]
        for i in range(len(self._model_inputs)):
            x.append(self._model_input_filtered[i])
        return x

    def _update_lag_filters(self, dt_seconds: float) -> None:
        """Update exponential lag filters for model inputs."""
        for i, m_input in enumerate(self._model_inputs):
            tau = float(m_input.get("lag_tau", 0))
            raw = self._model_input_values[i]
            if tau > 0 and dt_seconds > 0:
                alpha = 1.0 - math.exp(-dt_seconds / tau)
                self._model_input_filtered[i] = (
                    alpha * raw + (1.0 - alpha) * self._model_input_filtered[i]
                )
            else:
                self._model_input_filtered[i] = raw

    def _read_model_input_values(self) -> None:
        """Read current values from all model input entities."""
        for i, m_input in enumerate(self._model_inputs):
            entity_id = m_input.get("entity_id", "")
            if not entity_id:
                continue
            state = self._hass.states.get(entity_id)
            if not state or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                _LOGGER.debug("Model input '%s' (%s) unavailable, using last value %.2f",
                             m_input.get("name", "?"), entity_id, self._model_input_values[i])
                continue
            try:
                self._model_input_values[i] = float(state.state)
            except (ValueError, TypeError):
                # Non-numeric: treat as active/inactive
                # Covers binary_sensor (on/off), climate (heat/cool/off), etc.
                active_states = {"on", "heat", "cool", "dry", "fan_only",
                                 "heating", "cooling", "burning", "igniting"}
                self._model_input_values[i] = 1.0 if state.state in active_states else 0.0

    def _any_model_input_unavailable(self) -> bool:
        """Check if any model input entity is currently unavailable."""
        for m_input in self._model_inputs:
            entity_id = m_input.get("entity_id", "")
            if not entity_id:
                continue
            state = self._hass.states.get(entity_id)
            if not state or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                return True
        return False

    @callback
    def _async_model_input_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle model input entity state changes — update binary sensor."""
        if hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_FF_SUPPRESS_UPDATE.format(self._entity._config_entry_id),
            )

    async def _pi_async_sensor_changed(self, was_none: bool = False) -> None:
        """Handle temp sensor update."""
        if not self._pi_enabled:
            return
        if was_none:
            # Verify the sensor actually has a numeric value — transitions from
            # None to 'unavailable' fire was_none=True but aren't real recoveries.
            if self._entity._attr_current_temperature is None:
                return
            if self._sensor_recovery_unsub:
                self._sensor_recovery_unsub()
                self._sensor_recovery_unsub = None
            self._sensor_recovery_pending = False
            if self._sensor_unavailable:
                _LOGGER.info("PI: temp sensor recovered, resuming full PI control")
                self._sensor_unavailable = False
            else:
                _LOGGER.debug("PI: temp sensor just became available, running immediate tick")
            # Guard against multiple sensors coming online simultaneously
            # (e.g., temp + humidity both fire was_none=True within milliseconds).
            # Allow recovery tick only if no tick ran in the last 2 seconds.
            elapsed = time.monotonic() - self._pi_last_tick_time
            if elapsed < 2.0:
                _LOGGER.debug("PI: skipping recovery tick, another ran %.1fs ago", elapsed)
                return
            await self._pi_tick()
            return

        elapsed = time.monotonic() - self._pi_last_tick_time
        min_cooldown = max(60.0, self._pi_min_interval / 3.0)
        if elapsed >= min_cooldown:
            await self._pi_tick()

    async def _check_sensor_recovery(self, _now: datetime | None = None) -> None:
        """Called 60s after sensor went unavailable. Fall back to FF-only if still gone."""
        self._sensor_recovery_pending = False
        self._sensor_recovery_unsub = None
        e = self._entity
        if e._attr_current_temperature is not None:
            _LOGGER.info("PI: temp sensor recovered during grace period")
            await self._pi_tick()
            return
        self._sensor_unavailable = True
        _LOGGER.warning("PI: temp sensor confirmed unavailable, using feedforward-only fallback")
        if self._desired_temp is None:
            return
        desired_c = TemperatureConverter.convert(
            self._desired_temp, e.temperature_unit, UnitOfTemperature.CELSIUS,
        )
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        if not is_heating and e._attr_hvac_mode not in (HVACMode.COOL, HVACMode.DRY):
            return
        # Use RLS model for FF-only fallback
        if self._outdoor_temp is not None:
            if is_heating:
                outdoor_delta = max(0, self._ff_heat_reference - self._outdoor_temp)
            else:
                outdoor_delta = max(0, self._outdoor_temp - self._ff_cool_reference)
            self._read_model_input_values()
            x = self._build_feature_vector(outdoor_delta)
            rls = self._rls_heat if is_heating else self._rls_cool
            self._ff_offset = rls.predict(x)
        else:
            self._ff_offset = 0.0
        self._pi_integral = 0.0
        new_setpoint = round(max(self._min_temp_c, min(self._max_temp_c, desired_c + self._ff_offset)))
        if new_setpoint != self._hp_setpoint:
            _LOGGER.info("PI fallback: setpoint %s -> %s (FF only)", self._hp_setpoint, new_setpoint)
            self._hp_setpoint = new_setpoint
            self._last_send_ir_time = time.monotonic()
            await e.send_ir()
        e.async_schedule_update_ha_state()

    async def _pi_tick(self, now: datetime | None = None) -> None:
        """PI + feedforward controller tick. Called by timer and sensor events."""
        if not self._pi_enabled:
            return
        if self._pi_tick_running:
            _LOGGER.debug("PI tick: skipping, already running (reentrant call)")
            return
        self._pi_tick_running = True
        try:
            await self._pi_tick_inner(now)
        finally:
            self._pi_tick_running = False

    async def _pi_tick_inner(self, now: datetime | None = None) -> None:
        """PI + feedforward controller tick implementation."""
        e = self._entity
        if e._attr_hvac_mode == HVACMode.OFF:
            self._pi_integral = 0.0
            return
        if self._desired_temp is None or self._hp_setpoint is None:
            return
        if e._attr_current_temperature is None:
            if self._sensor_unavailable or self._sensor_recovery_pending:
                return
            _LOGGER.info("PI: temp sensor unavailable, scheduling 60s recovery check")
            self._sensor_recovery_pending = True
            self._sensor_recovery_unsub = async_call_later(
                self._hass, 60, self._check_sensor_recovery
            )
            return
        if self._pi_paused:
            _LOGGER.debug("PI tick: skipping, paused by vendor")
            return

        # Time since last tick (for time-normalized integral)
        now_mono = time.monotonic()
        if self._pi_last_tick_time > 0:
            dt_seconds = min(now_mono - self._pi_last_tick_time, self._pi_min_interval * 2)
        else:
            dt_seconds = float(self._pi_min_interval)
        self._pi_last_tick_time = now_mono
        dt_factor = dt_seconds / float(self._pi_min_interval)

        # Convert both to °C for PI math
        current_c = TemperatureConverter.convert(
            e._attr_current_temperature,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )
        desired_c = TemperatureConverter.convert(
            self._desired_temp,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )
        error = desired_c - current_c

        # Evaluate supplemental heat source override (selector control)
        now_mono = time.monotonic()
        hp_should_send_ir = self._evaluate_supplemental_override(error, now_mono)

        # Filtered derivative on measurement (not error — avoids derivative kick).
        # D(s) = -Kd * s / (1 + Tf*s) where Tf = Kd/N.
        # Discrete: D[n] = (Tf/(Tf+dt))*D[n-1] - (Kd/(Tf+dt))*(y[n]-y[n-1])
        if self._pi_last_measurement is not None and dt_seconds > 0:
            td = self._pi_kd  # derivative time constant (minutes, used as gain)
            tf = td / max(self._pi_kd_filter_n, 1)  # filter time (minutes)
            dt_min = dt_seconds / 60.0  # convert to minutes for consistency with Kd units
            alpha_d = tf / (tf + dt_min)
            dy = current_c - self._pi_last_measurement
            self._pi_d_filtered = alpha_d * self._pi_d_filtered - (td / (tf + dt_min)) * dy
        self._pi_last_measurement = current_c

        # PID only operates in explicit HEAT, COOL, or DRY modes
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        is_cooling = e._attr_hvac_mode in (HVACMode.COOL, HVACMode.DRY)
        if not is_heating and not is_cooling:
            return

        # Read model input values and update lag filters
        self._read_model_input_values()
        self._update_lag_filters(dt_seconds)

        # Compute outdoor delta (always first model input)
        if self._outdoor_temp is not None:
            if is_heating:
                outdoor_delta = max(0, self._ff_heat_reference - self._outdoor_temp)
            else:
                outdoor_delta = max(0, self._outdoor_temp - self._ff_cool_reference)
        else:
            outdoor_delta = 0.0

        # Build feature vector and predict FF offset via RLS model
        x = self._build_feature_vector(outdoor_delta)
        rls = self._rls_heat if is_heating else self._rls_cool
        seeds = self._heat_seeds if is_heating else self._cool_seeds

        # Blend seed prediction with RLS prediction based on observation count.
        # Early RLS estimates are unstable (large P matrix gives each observation
        # outsized influence). Linear blend from pure seeds to pure RLS over
        # MIN_RLS_OBS observations prevents discontinuities when transitioning
        # from prior knowledge to learned coefficients.
        MIN_RLS_OBS = 10
        seed_offset = sum(s * xi for s, xi in zip(seeds, x))
        rls_offset = rls.predict(x)
        alpha = min(rls.observation_count / MIN_RLS_OBS, 1.0)
        self._ff_offset = (1.0 - alpha) * seed_offset + alpha * rls_offset

        # Learning suppression: manual service + per-input suppress_learning flag
        learning_suppressed = self._manual_ff_suppress
        active_suppressors = []
        if self._manual_ff_suppress:
            active_suppressors.append("manual")
        for i, m_input in enumerate(self._model_inputs):
            if m_input.get("suppress_learning") and self._model_input_values[i] > 0.5:
                learning_suppressed = True
                active_suppressors.append(m_input.get("name", f"input_{i}"))
        self._disturbance_suppress_active = learning_suppressed
        self._disturbance_active_suppressors = active_suppressors

        abs_error = abs(error)

        # Deadband: if error is small, skip P term and freeze integral.
        in_deadband = abs_error < self._pi_deadband
        if in_deadband:
            # Freeze integral — system is close enough to target.
            # Quantization-error feedback (below) handles the X.5 boundary.
            # RLS learning absorbs persistent integral into FF over time.
            self._ff_settled_ticks += 1
            # RLS learns when integral is stable (not still converging), regardless
            # of magnitude. Large stable integral = FF is wrong, observation is valid.
            integral_stable = abs(self._pi_integral - self._prev_integral_for_rls) < 0.5
            can_learn_rls = (
                self._ff_settled_ticks >= 4
                and self._outdoor_temp is not None
                and not learning_suppressed
                and integral_stable
                and not self._any_model_input_unavailable()
                and not self._tracking_mode
                and not self._supplemental_assist_active
            )
            if not can_learn_rls and self._ff_settled_ticks == 4:
                # Log why learning was blocked (once, at the gate threshold)
                reasons = []
                if self._outdoor_temp is None:
                    reasons.append("no outdoor temp")
                if learning_suppressed:
                    reasons.append("manually suppressed")
                if not integral_stable:
                    reasons.append(f"integral not stable (|I|={abs(self._pi_integral):.1f}, rate={abs(self._pi_integral - self._prev_integral_for_rls):.2f})")
                if self._any_model_input_unavailable():
                    reasons.append("model input unavailable")
                if reasons:
                    _LOGGER.debug("RLS learning blocked: %s", ", ".join(reasons))
            if can_learn_rls:
                # Observe what actually worked: the HP setpoint that achieved
                # the target temperature. This directly measures the needed
                # offset without coupling to integral state or ki.
                observed_offset = float(self._hp_setpoint) - desired_c
                beta_before = list(rls.beta)
                residual = rls.update(x, observed_offset)
                _LOGGER.debug(
                    "RLS update: observed=%.2f predicted=%.2f residual=%.2f obs_count=%d",
                    observed_offset, observed_offset - residual, residual,
                    rls.observation_count,
                )
                # Log significant coefficient changes
                for idx in range(len(rls.beta)):
                    if beta_before[idx] != 0 and abs(rls.beta[idx] - beta_before[idx]) / abs(beta_before[idx]) > 0.1:
                        _LOGGER.info(
                            "RLS coefficient[%d] changed %.3f -> %.3f (%.0f%%)",
                            idx, beta_before[idx], rls.beta[idx],
                            100 * (rls.beta[idx] - beta_before[idx]) / beta_before[idx],
                        )

            self._prev_integral_for_rls = self._pi_integral

            if learning_suppressed and self._ff_settled_ticks >= 2:
                _LOGGER.debug(
                    "PI: FF learning suppressed by %s",
                    active_suppressors if active_suppressors else "manual",
                )
            p_term = 0.0
        else:
            self._ff_settled_ticks = 0
            # Setpoint weighting: reduce P-term to prevent overshoot while
            # integral drives steady-state accuracy. Standard 2-DOF technique
            # (Astrom & Hagglund). b < 1 reduces proportional kick.
            p_term = self._pi_kp * self._pi_setpoint_weight * error
            avg_error = (error + self._pi_last_error) / 2.0
            self._pi_integral += avg_error * dt_factor

        self._pi_last_error = error

        # Update integral convergence metric (EMA of abs(integral), ~24hr time constant)
        # With 15-min ticks, 96 ticks/day → alpha ≈ 1/96 ≈ 0.01
        convergence_alpha = 0.01
        self._integral_convergence = (
            (1.0 - convergence_alpha) * self._integral_convergence
            + convergence_alpha * abs(self._pi_integral)
        )

        # Performance metrics accumulation (paused during tracking — HP not responsible)
        if not self._tracking_mode:
            self._itae_tick_count += 1
            effective_error = max(0.0, abs_error - self._pi_deadband)
            self._itae_accumulator += self._itae_tick_count * effective_error
            if abs_error > 1.0:
                self._comfort_violation_hours += dt_seconds / 3600.0

        i_term = self._pi_ki * self._pi_integral
        d_term = self._pi_d_filtered
        raw_setpoint = desired_c + p_term + i_term + d_term + self._ff_offset
        clamped_setpoint = max(self._min_temp_c, min(self._max_temp_c, raw_setpoint))

        # Conditional anti-windup: stop integral from growing in the saturated
        # direction. Freeze at the value that would produce the clamped output.
        if clamped_setpoint != raw_setpoint and self._pi_ki != 0:
            if raw_setpoint > clamped_setpoint and self._pi_integral > 0:
                max_i = (clamped_setpoint - desired_c - p_term - self._ff_offset) / self._pi_ki
                self._pi_integral = min(self._pi_integral, max_i)
            elif raw_setpoint < clamped_setpoint and self._pi_integral < 0:
                min_i = (clamped_setpoint - desired_c - p_term - self._ff_offset) / self._pi_ki
                self._pi_integral = max(self._pi_integral, min_i)

        # Quantization-error feedback: push integral toward values where
        # raw_setpoint lands near an integer, avoiding the X.5 boundary that
        # causes limit cycles with 1°C HP steps.
        if in_deadband and self._pi_ki != 0:
            q_error = float(self._hp_setpoint) - clamped_setpoint
            if abs(q_error) > 0.3:
                self._pi_integral += (q_error / self._pi_ki) * 0.4

        # Midpoint-crossing hysteresis
        new_setpoint = self._hp_setpoint
        if clamped_setpoint > self._hp_setpoint + 0.5:
            new_setpoint = round(clamped_setpoint)
        elif clamped_setpoint < self._hp_setpoint - 0.5:
            new_setpoint = round(clamped_setpoint)
        new_setpoint = int(max(self._min_temp_c, min(self._max_temp_c, new_setpoint)))

        if new_setpoint != self._hp_setpoint:
            # Minimum hold time: don't change setpoint more often than every 30 min.
            # A 1°C change takes 15-30 min to affect room temp — wait to see its effect.
            # Bypass for large corrections (>1°C) and active ramps (error >> deadband).
            change = new_setpoint - self._hp_setpoint
            time_since_last = now_mono - self._last_setpoint_change_time
            can_change = time_since_last >= 1800.0  # 30 minutes
            if not in_deadband:
                can_change = True  # Outside deadband = active demand, bypass hold
            if not can_change:
                _LOGGER.debug(
                    "PI: setpoint %s -> %s held (%.0fs since last change, need 1800s)",
                    self._hp_setpoint, new_setpoint, time_since_last,
                )
            else:
                self._hp_setpoint = new_setpoint
                if not hp_should_send_ir:
                    # Tracking mode: update internal setpoint but don't send IR
                    _LOGGER.debug(
                        "PI tracking: setpoint %s -> %s (IR suppressed, override by %s)",
                        self._hp_setpoint, new_setpoint, ", ".join(self._tracking_sources),
                    )
                else:
                    _LOGGER.info(
                        "PI: error=%.1f P=%.1f I=%.1f D=%.1f FF=%.1f raw=%.1f setpoint %s -> %s",
                        error, p_term, i_term, d_term, self._ff_offset, clamped_setpoint,
                        self._hp_setpoint, new_setpoint,
                    )
                    self._last_setpoint_change_time = now_mono
                    self._setpoint_changes_today += 1
                    self._last_send_ir_time = time.monotonic()
                    await e.send_ir()
        else:
            _LOGGER.debug(
                "PI: error=%.1f raw=%.1f setpoint=%s (held)",
                error, clamped_setpoint, self._hp_setpoint,
            )

        e.async_schedule_update_ha_state()
