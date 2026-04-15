"""Vendor-agnostic PI + feedforward controller for IRHVAC climate entities.

Pure computation — the controller never publishes MQTT or writes HA state.
climate.py owns all I/O: it calls PI methods that return bool (True = send
needed), then decides whether to call send_ir().

Composed object (not a mixin). The climate entity creates a PIController instance
and calls its hook methods at the appropriate points. This avoids MRO issues
and minimizes changes to the upstream-derived climate.py.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State

    from .climate import TasmotaIrhvac

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.restore_state import ExtraStoredData
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
)
from homeassistant.util.unit_conversion import TemperatureConverter

import math

from .batch_learning import BatchResult, DiversityAwareBuffer, Observation, ObservationBuffer, weighted_least_squares, compare_and_report, compute_blended_update

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
    CONF_PI_IMC_LAMBDA,
    CONF_PI_KD,
    CONF_PI_SENSOR_FILTER_TAU,
    CONF_PI_KD_FILTER_N,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_MIN_INTERVAL,
    CONF_PI_MODEL_INPUTS,
    CONF_PI_RESPONSE_LAG,
    CONF_PI_SETPOINT_HOLD,
    CONF_PI_SETPOINT_WEIGHT,
    CONF_PI_SMITH_ENABLED,
    CONF_PI_TAU_ESTIMATE,
    DEFAULT_PI_DEADBAND,
    DEFAULT_PI_ENABLED,
    DEFAULT_PI_FF_COOL_REFERENCE,
    DEFAULT_PI_FF_COOL_SLOPE,
    DEFAULT_PI_FF_HEAT_REFERENCE,
    DEFAULT_PI_FF_HEAT_SLOPE,
    DEFAULT_PI_IMC_LAMBDA,
    DEFAULT_PI_KD,
    DEFAULT_PI_SENSOR_FILTER_TAU,
    DEFAULT_PI_KD_FILTER_N,
    DEFAULT_PI_KI,
    DEFAULT_PI_KP,
    DEFAULT_PI_MIN_INTERVAL,
    DEFAULT_PI_RESPONSE_LAG,
    DEFAULT_PI_SETPOINT_HOLD,
    DEFAULT_PI_SETPOINT_WEIGHT,
    DEFAULT_PI_SMITH_ENABLED,
    DEFAULT_PI_TAU_ESTIMATE,
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
    itae_accumulator: float = 0.0
    comfort_violation_hours: float = 0.0
    setpoint_changes: int = 0
    controllable_itae: float = 0.0
    uncontrollable_itae: float = 0.0
    controllable_cvh: float = 0.0
    uncontrollable_cvh: float = 0.0
    ff_load_fraction: float = 0.5
    rls_heat_model: dict = dataclasses.field(default_factory=dict)
    rls_cool_model: dict = dataclasses.field(default_factory=dict)
    lag_filter_states: dict = dataclasses.field(default_factory=dict)
    heat_seeds_at_learn: list = dataclasses.field(default_factory=list)
    cool_seeds_at_learn: list = dataclasses.field(default_factory=list)
    ki_at_save: float = 0.0
    tau_estimate: float = 0.0
    observation_buffer: list = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "pi_integral": self.pi_integral,
            "desired_temp": self.desired_temp,
            "hp_setpoint": self.hp_setpoint,
            "integral_convergence": self.integral_convergence,
            "itae_accumulator": self.itae_accumulator,
            "comfort_violation_hours": self.comfort_violation_hours,
            "setpoint_changes": self.setpoint_changes,
            "controllable_itae": self.controllable_itae,
            "uncontrollable_itae": self.uncontrollable_itae,
            "controllable_cvh": self.controllable_cvh,
            "uncontrollable_cvh": self.uncontrollable_cvh,
            "ff_load_fraction": self.ff_load_fraction,
            "rls_heat_model": self.rls_heat_model,
            "rls_cool_model": self.rls_cool_model,
            "lag_filter_states": self.lag_filter_states,
            "heat_seeds_at_learn": self.heat_seeds_at_learn,
            "cool_seeds_at_learn": self.cool_seeds_at_learn,
            "ki_at_save": self.ki_at_save,
            "tau_estimate": self.tau_estimate,
            "observation_buffer": self.observation_buffer,
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
                itae_accumulator=float(restored.get("itae_accumulator", 0.0)),
                comfort_violation_hours=float(restored.get("comfort_violation_hours", 0.0)),
                setpoint_changes=int(restored.get("setpoint_changes", 0)),
                controllable_itae=float(restored.get("controllable_itae", 0.0)),
                uncontrollable_itae=float(restored.get("uncontrollable_itae", 0.0)),
                controllable_cvh=float(restored.get("controllable_cvh", 0.0)),
                uncontrollable_cvh=float(restored.get("uncontrollable_cvh", 0.0)),
                ff_load_fraction=float(restored.get("ff_load_fraction", 0.5)),
                rls_heat_model=restored.get("rls_heat_model", {}),
                rls_cool_model=restored.get("rls_cool_model", {}),
                lag_filter_states=restored.get("lag_filter_states", {}),
                heat_seeds_at_learn=restored.get("heat_seeds_at_learn", []),
                cool_seeds_at_learn=restored.get("cool_seeds_at_learn", []),
                ki_at_save=float(restored.get("ki_at_save", 0.0)),
                tau_estimate=float(restored.get("tau_estimate", 0.0)),
                observation_buffer=restored.get("observation_buffer", []),
            )
        except (KeyError, ValueError, TypeError, AttributeError):
            return None


class SmithPredictor:
    """Modified Smith predictor for dead-time compensation (Åström Ch. 7 §7.3).

    Uses an internal FOPDT model to estimate the temperature change "in the
    pipeline" — corrections that have been commanded but haven't reached the
    sensor due to transport delay L.  Subtracting this from the PI error
    signal lets the controller react as if there were no delay.

    Two internal first-order models run in parallel:
      nodelay  — receives current HP setpoint immediately
      delayed  — receives HP setpoint from L minutes ago (ring buffer lookup)

    The correction term (nodelay − delayed) represents the pending temperature
    change.  At steady state the term is zero; after a setpoint change it
    transiently grows then decays as the real plant catches up.

    Robust to ±50% parameter mismatch: degrades gracefully (slower convergence)
    without instability, because the outer PI loop still closes on the real
    measurement.
    """

    def __init__(self, tau: float, lag: float, k_eff: float = 1.0) -> None:
        self.tau: float = max(tau, 1.0)   # thermal time constant (minutes)
        self.lag: float = lag             # transport delay L (minutes)
        self.k_eff: float = k_eff        # process gain (1.0 = unit gain)
        self._model_nodelay: float = 0.0  # no-delay model state (°C)
        self._model_delayed: float = 0.0  # delayed model state (°C)
        # Ring buffer: (monotonic_time, hp_setpoint) for delayed lookup
        self._setpoint_history: list[tuple[float, float]] = []
        self._initialized: bool = False

    def initialize(self, room_temp: float, hp_setpoint: float, now_mono: float) -> None:
        """Initialize model states to current room temperature.

        Both models start at the same value so correction = 0 on first tick.
        The Smith predictor is inert until the first setpoint change creates
        a divergence between the nodelay and delayed models.
        """
        self._model_nodelay = room_temp
        self._model_delayed = room_temp
        self._setpoint_history = [(now_mono, hp_setpoint)]
        self._initialized = True

    def update_params(self, tau: float, lag: float) -> None:
        """Update model parameters when τ estimate changes."""
        self.tau = max(tau, 1.0)
        self.lag = lag

    def record_setpoint(self, hp_setpoint: float, now_mono: float) -> None:
        """Record current HP setpoint for delayed lookup."""
        self._setpoint_history.append((now_mono, hp_setpoint))
        # Trim: keep enough history to look back L + margin
        cutoff = now_mono - (self.lag + 5.0) * 60.0
        while len(self._setpoint_history) > 2 and self._setpoint_history[0][0] < cutoff:
            self._setpoint_history.pop(0)

    def _delayed_setpoint(self, now_mono: float) -> float:
        """Look up HP setpoint from L minutes ago."""
        target_time = now_mono - self.lag * 60.0
        if not self._setpoint_history:
            return 0.0
        # Find last entry at or before target_time
        result = self._setpoint_history[0][1]
        for t, sp in self._setpoint_history:
            if t <= target_time:
                result = sp
            else:
                break
        return result

    def step(self, hp_setpoint: float, dt_seconds: float, now_mono: float) -> None:
        """Advance both internal models by one time step (exact exponential)."""
        if not self._initialized:
            return
        dt_min = dt_seconds / 60.0
        tau = self.tau
        decay = math.exp(-dt_min / tau) if tau > 0 else 0.0

        # No-delay model: receives current setpoint immediately
        eq_nd = self.k_eff * hp_setpoint
        self._model_nodelay = eq_nd + (self._model_nodelay - eq_nd) * decay

        # Delayed model: receives setpoint from L minutes ago
        u_delayed = self._delayed_setpoint(now_mono)
        eq_d = self.k_eff * u_delayed
        self._model_delayed = eq_d + (self._model_delayed - eq_d) * decay

    @property
    def correction(self) -> float:
        """Pipeline temperature: pending change that hasn't reached the sensor.

        Returns nodelay − delayed model prediction.  Positive means the room
        should get warmer than the sensor currently reads (heating in pipeline).
        """
        if not self._initialized:
            return 0.0
        return self._model_nodelay - self._model_delayed

    def reset(self, room_temp: float, hp_setpoint: float, now_mono: float) -> None:
        """Reset model states (mode change, large regime shift)."""
        self.initialize(room_temp, hp_setpoint, now_mono)


class PIController:
    """PI + feedforward temperature controller for IRHVAC climate entities.

    Pure computation — never publishes MQTT or writes HA state. Methods return
    bool (True = new setpoint ready, caller should send IR).

    Usage in climate entity:
        __init__:           self._pi = PIController(self, config)
        async_added_to_hass:     await self._pi.async_added_to_hass(old_state)
        async_will_remove:       self._pi.async_will_remove_from_hass()
        async_set_temperature:   send = await self._pi.set_temperature(temp, hvac_mode)
        on_remote_change:        send = await self._pi.on_remote_change(reported_temp)
        sensor_changed:          send = await self._pi.sensor_changed(was_none)
        pi_tick:                 send = await self._pi.pi_tick()
        _get_ir_temp:            return self._pi.get_ir_temp()
        extra_state_attributes:  attrs.update(self._pi.get_extra_state_attributes())
        hvac_modes:              return self._pi.filter_hvac_modes(modes)
        async_set_hvac_mode:     if self._pi.should_reject_hvac_mode(mode): return
        async_write_ha_state:    self._pi.fire_dispatcher()

    Vendor subclasses use: pi_pause(), pi_resume(), pi_reset_integral()
    """

    # Health check thresholds
    HEALTH_COMFORT_WARN: float = 1.1        # °C error (~2°F)
    HEALTH_COMFORT_CRIT: float = 1.7        # °C error (~3°F)
    HEALTH_INTEGRAL_WARN: float = 2.0       # |ki × integral| correction in °C
    HEALTH_INTERCEPT_WARN: float = 1.0      # RLS intercept drift
    HEALTH_SLOPE_DRIFT_PCT: float = 30.0    # % drift from configured ff slope
    HEALTH_SLOPE_DRIFT_FLOOR: float = 0.05  # minimum absolute drift to trigger

    def __init__(self, entity: TasmotaIrhvac, config: dict[str, Any]) -> None:
        """Initialize PI controller.

        Args:
            entity: The climate entity this controller is attached to.
            config: Merged config dict (entry.data + entry.options).
        """
        self._entity = entity
        self._log_prefix: str = ""  # set in async_added when entity_id is known

        # Convert entity temp limits to °C for internal PI math
        self._min_temp_c: float = TemperatureConverter.convert(
            entity._min_temp, entity._attr_temperature_unit, UnitOfTemperature.CELSIUS
        )
        self._max_temp_c: float = TemperatureConverter.convert(
            entity._max_temp, entity._attr_temperature_unit, UnitOfTemperature.CELSIUS
        )

        # PID controller config
        self._pi_enabled: bool = config.get(CONF_PI_ENABLED, DEFAULT_PI_ENABLED)
        self._pi_kp_config: float = config.get(CONF_PI_KP, DEFAULT_PI_KP)
        self._pi_ki_config: float = config.get(CONF_PI_KI, DEFAULT_PI_KI)
        self._pi_kd: float = config.get(CONF_PI_KD, DEFAULT_PI_KD)
        self._pi_kd_filter_n: float = config.get(CONF_PI_KD_FILTER_N, DEFAULT_PI_KD_FILTER_N)
        self._sensor_filter_tau: float = config.get(
            CONF_PI_SENSOR_FILTER_TAU, DEFAULT_PI_SENSOR_FILTER_TAU
        )  # Low-pass filter τ on room temperature (seconds). 0 = disabled.
        self._smith_enabled: bool = config.get(CONF_PI_SMITH_ENABLED, DEFAULT_PI_SMITH_ENABLED)
        self._SETPOINT_HOLD_SECONDS: float = float(
            config.get(CONF_PI_SETPOINT_HOLD, DEFAULT_PI_SETPOINT_HOLD)
        )
        self._pi_min_interval: float = config.get(CONF_PI_MIN_INTERVAL, DEFAULT_PI_MIN_INTERVAL)

        # IMC gain scheduling from τ estimate
        self._tau_seed: float = config.get(CONF_PI_TAU_ESTIMATE, DEFAULT_PI_TAU_ESTIMATE)
        self._response_lag: float = config.get(CONF_PI_RESPONSE_LAG, DEFAULT_PI_RESPONSE_LAG)
        self._imc_lambda_config: float = config.get(CONF_PI_IMC_LAMBDA, DEFAULT_PI_IMC_LAMBDA)
        self._tau_estimate: float = self._tau_seed  # Online estimate, updated by step-response
        self._imc_enabled: bool = self._tau_seed > 0

        # Smith predictor for dead-time compensation.
        # Requires IMC (tau > 0) AND pi_smith_enabled=true.
        # Must be created before _recompute_imc_gains which calls update_params.
        self._smith: SmithPredictor | None = None
        if self._imc_enabled and self._smith_enabled:
            self._smith = SmithPredictor(
                tau=self._tau_estimate, lag=self._response_lag
            )

        # Derive effective Kp/Ki: IMC formula or manual config
        self._pi_kp: float = 0.0
        self._pi_ki: float = 0.0
        if self._imc_enabled:
            self._recompute_imc_gains()
        else:
            self._pi_kp = self._pi_kp_config
            self._pi_ki = self._pi_ki_config

        # Step-response τ observation state
        self._tau_step_time: float = 0.0        # monotonic time of last HP setpoint change
        self._tau_step_temp: float | None = None  # room temp at step time
        self._tau_step_target: float | None = None  # expected final room temp (desired_c)
        self._tau_step_magnitude: float = 0.0   # signed HP setpoint change (°C)
        self._tau_step_active: bool = False      # True while observing a step response
        self._tau_observations: int = 0          # total τ observations made

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
        self._vendor_precision: float = entity._temp_precision or 1.0
        # hp_setpoint is always in °C. The first PI tick will round to vendor
        # precision. No init rounding needed — the 18.889 mismatch bug is
        # prevented by climate.py's state-diff echo classification, not by
        # rounding the stored setpoint.
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
        self._ff_offset: float = 0.0
        self._pi_tick_running: bool = False
        self._last_setpoint_change_time: float = 0.0
        self._ff_settled_ticks: int = 0
        self._sensor_unavailable: bool = False
        self._sensor_recovery_pending: bool = False
        self._recovery_check_needed: bool = False
        self._pi_paused: bool = False
        self._pi_last_tick_time: float = 0.0
        self._pi_timer_unsub: Any | None = None    # cancel handle for fallback timer
        self._pi_timer_callback: Any | None = None  # set by climate.py during setup
        self._pi_last_error: float = 0.0
        self._pi_d_filtered: float = 0.0    # Filtered derivative term
        self._pi_last_measurement: float | None = None  # Previous temperature measurement for derivative
        self._sensor_filtered: float | None = None  # Low-pass filtered room temp (°C)
        self._last_raw_setpoint: float = 0.0  # Pre-quantization setpoint from last tick

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
        self._rls_heat_clamps: list[tuple[float, float] | None] = [None, (0.0, 2.0)]
        self._rls_cool_clamps: list[tuple[float, float] | None] = [None, (-2.0, 0.0)]
        for m_input in self._model_inputs:
            self._heat_seeds.append(float(m_input.get("seed_heat", 0.0)))
            self._cool_seeds.append(float(m_input.get("seed_cool", 0.0)))
            clamp_min = m_input.get("clamp_min")
            clamp_max = m_input.get("clamp_max")
            clamp: tuple[float, float] | None = (
                (float(clamp_min), float(clamp_max))
                if clamp_min is not None and clamp_max is not None else None
            )
            self._rls_heat_clamps.append(clamp)
            self._rls_cool_clamps.append(clamp)

        # Feature scales for balanced P initialization.
        # intercept=1.0, outdoor_delta typical ~10, model inputs ~0.5
        self._feature_scales = [1.0, 10.0]
        for m_input in self._model_inputs:
            self._feature_scales.append(float(m_input.get("typical_value", 0.5)))

        # RLS models (separate for heating and cooling)
        self._rls_heat = RLSModel(
            n_inputs=self._n_model_inputs,
            seed_coefficients=self._heat_seeds,
            coeff_clamps=self._rls_heat_clamps,
            feature_scales=self._feature_scales,
        )
        self._rls_cool = RLSModel(
            n_inputs=self._n_model_inputs,
            seed_coefficients=self._cool_seeds,
            coeff_clamps=self._rls_cool_clamps,
            feature_scales=self._feature_scales,
        )

        # Model input current values and lag filter states
        self._model_input_values = [0.0] * len(self._model_inputs)
        self._model_input_filtered = [0.0] * len(self._model_inputs)
        self._model_input_last_values = [0.0] * len(self._model_inputs)  # For lag filter

        # Outdoor temp state
        self._outdoor_temp: float | None = None


        # Health check state
        self._health_prev_desired: float | None = None
        self._health_comfort_skip: int = 0

        # Integral convergence tracking (EMA of abs(integral) over ~24hr)
        self._integral_convergence: float = 0.0

        # Performance metrics (running totals, persisted via ExtraStoredData)
        self._itae_accumulator: float = 0.0      # Σ(tick * |effective_error|)
        self._itae_tick_count: int = 0         # ticks since last reset
        self._comfort_violation_hours: float = 0.0  # hours spent >1°C from setpoint
        self._setpoint_changes: int = 0  # total setpoint change count
        # Controllable/uncontrollable split: only accumulate "controllable"
        # when HP has headroom (not clamped at min/max for the error direction).
        self._controllable_itae: float = 0.0
        self._uncontrollable_itae: float = 0.0
        self._controllable_cvh: float = 0.0
        self._uncontrollable_cvh: float = 0.0

        # RLS learning gate: track integral stability
        self._prev_integral_for_rls: float = 0.0
        # Out-of-deadband steady-state learning: consecutive ticks with
        # low room_temp_rate while outside deadband but within learning zone.
        self._stable_oodb_ticks: int = 0
        self._prev_integral_for_oodb: float = 0.0  # separate from deadband tracker

        # FF confidence (EMA-smoothed to prevent limit cycling from
        # tick-to-tick confidence changes near integer setpoint boundaries)
        self._ff_confidence: float = 1.0

        # FF load fraction: EMA of |ff_offset| / (|ff_offset| + |ki*integral|).
        # Trends toward 1.0 as the model takes load off the integral.
        self._ff_load_fraction: float = 0.5

        # Batch model RMS: last residual_rms from batch WLS analysis.
        self._batch_model_rms: float | None = None

        # Conditional integration freeze: track state for edge-triggered logging.
        self._integration_frozen: bool = False

        # Drift detection: per-coefficient history of batch correction signs.
        # Each entry is +1 (batch pushed up), -1 (batch pushed down), or 0.
        # Tracked across batch cycles to detect persistent same-direction
        # corrections that indicate a physical change.
        self._drift_correction_signs: list[list[int]] = []
        # Number of consecutive same-direction corrections to trigger alert.
        self._drift_threshold: int = 5

        # Batch learning: diversity-aware buffer of every tick's state for
        # periodic offline WLS analysis.  Records regardless of learning gate.
        # n_features = intercept + outdoor_delta + model_inputs
        self._observation_buffer = DiversityAwareBuffer(
            n_features=2 + len(self._model_inputs),
        )
        self._batch_analysis_timer: Any | None = None
        self._last_batch_result: BatchResult | None = None
        self._last_batch_timestamp: float | None = None

        # Room temperature rate of change tracking (°C/min)
        self._room_temp_history: list[tuple[float, float]] = []  # [(monotonic_time, temp_c), ...]
        self._room_temp_rate: float = 0.0  # °C/min, updated each tick

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
        # Build log prefix from entity name (e.g. "Dining Room" from "climate.dining_room")
        eid = getattr(self._entity, "entity_id", None) or ""
        short = eid.removeprefix("climate.").replace("_", " ").title()
        self._log_prefix = f"[{short}] " if short else ""

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

        # Timer and initial tick are set up by climate.py after this returns.

    def async_will_remove_from_hass(self) -> None:
        """Clean up PI state."""
        if self._pi_timer_unsub:
            self._pi_timer_unsub()
            self._pi_timer_unsub = None
        if self._batch_analysis_timer:
            self._batch_analysis_timer()
            self._batch_analysis_timer = None

    def schedule_batch_analysis(self) -> None:
        """Schedule periodic batch WLS analysis (called by climate.py after setup)."""
        from datetime import timedelta
        from homeassistant.helpers.event import async_track_time_interval

        BATCH_INTERVAL = timedelta(hours=12)

        @callback
        def _run_batch(_now: Any) -> None:
            self._run_batch_analysis()

        self._batch_analysis_timer = async_track_time_interval(
            self._hass, _run_batch, BATCH_INTERVAL,
        )

    def _run_batch_analysis(self) -> None:
        """Run batch WLS analysis and apply blended updates to RLS.

        Analyzes accumulated near-equilibrium observations via weighted
        least squares, then applies covariance-weighted Kalman fusion
        to blend batch estimates with the current online model.  Safety:
        ±1.0°C step cap per coefficient per 12h cycle.
        """
        e = self._entity
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        rls = self._rls_heat if is_heating else self._rls_cool

        # Periodic recomputation of info matrix to prevent numerical drift.
        if hasattr(self._observation_buffer, 'recompute_info_matrix'):
            if self._observation_buffer.needs_recompute:
                self._observation_buffer.recompute_info_matrix()

        observations = self._observation_buffer.get_all()
        if len(observations) < 20:
            _LOGGER.debug(
                "%sBatch WLS: insufficient observations (%d < 20)",
                self._log_prefix, len(observations),
            )
            return

        # Get current physical coefficients before WLS so held features
        # can be filled from the online model (persistent excitation filter).
        coeff_dict = rls.get_coefficients()
        current_phys = [coeff_dict[i] for i in range(rls.n)]

        result = weighted_least_squares(
            observations, n_features=rls.n, current_beta=current_phys,
            room_rate_threshold=0.02, min_observations=20,
        )
        if result is None:
            _LOGGER.debug(
                "%sBatch WLS: insufficient eligible observations after filtering",
                self._log_prefix,
            )
            return
        coeff_names = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            coeff_names.append(m.get("name", "input"))

        compare_and_report(
            result, current_phys, coeff_names,
            change_threshold_pct=20.0, min_observations=20,
            log_prefix=self._log_prefix,
        )

        compute_blended_update(result, prior_std=1.0, max_step=1.0)
        if result.recommend_update and result.beta_blended:
            for i, val in enumerate(result.beta_blended):
                if i < rls.n:
                    rls.beta[i] = val * rls.feature_scales[i]
            _LOGGER.info(
                "%sBatch WLS: applied blended update to %s model",
                self._log_prefix, "heat" if is_heating else "cool",
            )

        self._last_batch_result = result
        self._last_batch_timestamp = time.monotonic()
        self._batch_model_rms = result.residual_rms

        # ── Drift detection: track per-coefficient correction direction ──
        if result.beta_blended and result.beta_current:
            n = min(len(result.beta_blended), len(result.beta_current))
            signs = []
            for i in range(n):
                delta = result.beta_blended[i] - result.beta_current[i]
                if abs(delta) < 1e-4:
                    signs.append(0)
                elif delta > 0:
                    signs.append(1)
                else:
                    signs.append(-1)
            # Initialize history on first cycle
            if not self._drift_correction_signs:
                self._drift_correction_signs = [[] for _ in range(n)]
            # Extend history if model grew (new input added)
            while len(self._drift_correction_signs) < n:
                self._drift_correction_signs.append([])
            for i in range(n):
                self._drift_correction_signs[i].append(signs[i])
                # Keep only the last 10 cycles (5 days) of history
                if len(self._drift_correction_signs[i]) > 10:
                    self._drift_correction_signs[i].pop(0)

    def get_drifting_coefficients(self) -> list[tuple[int, str, int]]:
        """Return coefficients with persistent same-direction correction.

        Returns list of (index, direction_label, consecutive_count) for
        coefficients that have been corrected in the same direction for
        >= drift_threshold consecutive cycles.
        """
        drifting = []
        coeff_names = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            coeff_names.append(m.get("name", "input"))

        for i, history in enumerate(self._drift_correction_signs):
            if len(history) < self._drift_threshold:
                continue
            # Check the most recent drift_threshold entries
            recent = history[-self._drift_threshold:]
            if all(s == 1 for s in recent):
                name = coeff_names[i] if i < len(coeff_names) else f"β{i}"
                drifting.append((i, name, len([s for s in history if s == recent[0]])))
            elif all(s == -1 for s in recent):
                name = coeff_names[i] if i < len(coeff_names) else f"β{i}"
                drifting.append((i, name, len([s for s in history if s == recent[0]])))
        return drifting

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
            itae_accumulator=self._itae_accumulator,
            comfort_violation_hours=self._comfort_violation_hours,
            setpoint_changes=self._setpoint_changes,
            controllable_itae=self._controllable_itae,
            uncontrollable_itae=self._uncontrollable_itae,
            controllable_cvh=self._controllable_cvh,
            uncontrollable_cvh=self._uncontrollable_cvh,
            ff_load_fraction=self._ff_load_fraction,
            rls_heat_model=self._rls_heat.as_dict(),
            rls_cool_model=self._rls_cool.as_dict(),
            lag_filter_states=lag_states,
            heat_seeds_at_learn=list(self._heat_seeds),
            cool_seeds_at_learn=list(self._cool_seeds),
            ki_at_save=self._pi_ki,
            tau_estimate=self._tau_estimate,
            observation_buffer=self._observation_buffer.as_list(),
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
        self._itae_accumulator = data.itae_accumulator
        self._comfort_violation_hours = data.comfort_violation_hours
        self._setpoint_changes = data.setpoint_changes
        self._controllable_itae = data.controllable_itae
        self._uncontrollable_itae = data.uncontrollable_itae
        self._controllable_cvh = data.controllable_cvh
        self._uncontrollable_cvh = data.uncontrollable_cvh
        self._ff_load_fraction = data.ff_load_fraction
        # Restore RLS models if available. Pass seed_coefficients so that
        # if model inputs changed (different vector length), new inputs get
        # seeded instead of zeroed.
        if data.rls_heat_model:
            self._rls_heat = RLSModel.from_dict(
                data.rls_heat_model, self._n_model_inputs,
                seed_coefficients=self._heat_seeds,
                coeff_clamps=self._rls_heat_clamps,
                feature_scales=self._feature_scales,
            )
        if data.rls_cool_model:
            self._rls_cool = RLSModel.from_dict(
                data.rls_cool_model, self._n_model_inputs,
                seed_coefficients=self._cool_seeds,
                coeff_clamps=self._rls_cool_clamps,
                feature_scales=self._feature_scales,
            )
        # Restore observation buffer for batch learning.
        # Accepts data from both legacy FIFO and diversity-aware buffers.
        if data.observation_buffer:
            self._observation_buffer = DiversityAwareBuffer.from_list(
                data.observation_buffer,
                n_features=2 + len(self._model_inputs),
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
        # Restore τ estimate and recompute IMC gains
        if self._imc_enabled and data.tau_estimate > 0:
            old_ki = self._pi_ki
            self._tau_estimate = data.tau_estimate
            self._recompute_imc_gains()
            # Re-scale integral for the restored ki (overrides the earlier scaling
            # which used the seed-derived ki, not the restored-τ-derived ki)
            if old_ki > 0 and old_ki != self._pi_ki:
                rescale = old_ki / self._pi_ki
                self._pi_integral *= rescale
                _LOGGER.debug(
                    "PI: re-scaled integral for restored τ (ki %.4f → %.4f, scale %.2f)",
                    old_ki, self._pi_ki, rescale,
                )

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
                scale = rls_model.feature_scales[i]
                old_val_phys = rls_model.beta[i] / scale
                rls_model.beta[i] = new_seeds[i] * scale  # Store in normalized space
                rls_model.beta_seed[i] = new_seeds[i] * scale
                # Reset P for this coefficient — uniform, normalization handles scaling
                rls_model.P[i * rls_model.n + i] = DEFAULT_RLS_P_INIT
                _LOGGER.info(
                    "Seed changed for coefficient %d: %.4f → %.4f (learned was %.4f, reset)",
                    i, old_seeds[i], new_seeds[i], old_val_phys,
                )

    async def set_temperature(
        self, temperature: float | None, hvac_mode: str | None = None
    ) -> bool:
        """Handle temperature set when PI is active. Returns True if send needed."""
        if temperature is None:
            return False
        e = self._entity
        if hvac_mode is not None:
            await e.set_mode(hvac_mode)
        old_desired = self._desired_temp
        self._desired_temp = temperature
        self._cancel_tau_observation()
        # Bumpless transfer (Åström & Hägglund): keep output continuous
        if old_desired is not None:
            old_c = TemperatureConverter.convert(
                old_desired, e.temperature_unit, UnitOfTemperature.CELSIUS,
            )
            new_c = TemperatureConverter.convert(
                temperature, e.temperature_unit, UnitOfTemperature.CELSIUS,
            )
            if abs(old_c - new_c) > 2.0:
                # Large regime shift — zero integral and reset Smith model
                self._pi_integral = 0.0
                if self._smith is not None:
                    self._smith._initialized = False
            else:
                # Adjust integral to keep output continuous
                self._pi_integral += self._pi_kp * (1 - self._pi_setpoint_weight) * (old_c - new_c)
        if e._attr_hvac_mode != HVACMode.OFF:
            e.power_mode = STATE_ON
        return await self._pi_tick()

    async def on_remote_change(self, reported_temp_ir_unit: float) -> bool:
        """Physical remote detected — update desired temp. Returns True if send needed.

        Called by climate.py when state-diff classification determines a physical
        remote was used (IrReceived + state changed). The reported temperature is
        in IR protocol units (typically °C).
        """
        if not self._pi_enabled or self._desired_temp is None or self._pi_paused:
            return False
        e = self._entity
        desired_in_entity_unit = TemperatureConverter.convert(
            reported_temp_ir_unit, e._ir_temp_unit, e.temperature_unit
        )
        _LOGGER.info(
            "%sPhysical remote detected: reported=%s -> desired=%s",
            self._log_prefix, reported_temp_ir_unit, desired_in_entity_unit,
        )
        old_desired = self._desired_temp
        self._desired_temp = desired_in_entity_unit
        self._hp_setpoint = reported_temp_ir_unit
        # Bumpless transfer: adjust integral to keep output continuous
        if old_desired != desired_in_entity_unit:
            old_c = TemperatureConverter.convert(
                old_desired, e.temperature_unit, UnitOfTemperature.CELSIUS,
            )
            new_c = TemperatureConverter.convert(
                desired_in_entity_unit, e.temperature_unit, UnitOfTemperature.CELSIUS,
            )
            if abs(old_c - new_c) > 2.0:
                self._pi_integral = 0.0
            else:
                self._pi_integral += self._pi_kp * (1 - self._pi_setpoint_weight) * (old_c - new_c)
        return await self._pi_tick()

    async def pi_tick(self, now: datetime | None = None) -> bool:
        """Run PI tick. Returns True if send needed. Public API for climate.py."""
        return await self._pi_tick(now)

    async def sensor_changed(self, was_none: bool) -> bool:
        """Handle temp sensor update. Returns True if send needed."""
        return await self._pi_async_sensor_changed(was_none=was_none)

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

        # Convert from normalized to physical units for display
        heat_phys = self._rls_heat.get_coefficients()
        cool_phys = self._rls_cool.get_coefficients()
        rls_heat_coeffs = {coeff_names[i]: round(heat_phys[i], 4)
                          for i in range(min(len(coeff_names), len(heat_phys)))}
        rls_cool_coeffs = {coeff_names[i]: round(cool_phys[i], 4)
                          for i in range(min(len(coeff_names), len(cool_phys)))}

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
            "room_temp_rate": round(self._room_temp_rate, 4),  # °C/min
            "effective_kp": round(self._pi_kp, 3),
            "effective_ki": round(self._pi_ki, 4),
            "tau_estimate": round(self._tau_estimate, 1) if self._imc_enabled else None,
            "tau_observations": self._tau_observations if self._imc_enabled else None,
            "smith_correction": (
                round(self._smith.correction, 3) if self._smith is not None else None
            ),
            "setpoint_changes_total": self._setpoint_changes,
            "sensor_filtered": (
                round(self._sensor_filtered, 3)
                if self._sensor_filtered is not None else None
            ),
        }

    def get_diagnostic_dump(self) -> dict[str, Any]:
        """Return full diagnostic state for offline analysis.

        Contains the observation buffer, current RLS coefficients, and
        config — everything needed to reproduce this session's debug
        bundle without manual HA CLI work.
        """
        coeff_names = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            coeff_names.append(m.get("name", "input"))
        heat_dict = self._rls_heat.get_coefficients()
        cool_dict = self._rls_cool.get_coefficients()
        return {
            "observation_buffer": self._observation_buffer.as_list(),
            "buffer_size": len(self._observation_buffer),
            "rls_heat": {
                "coefficients": {coeff_names[i] if i < len(coeff_names) else f"β{i}": heat_dict[i]
                                 for i in range(self._rls_heat.n)},
                "observation_count": self._rls_heat.observation_count,
            },
            "rls_cool": {
                "coefficients": {coeff_names[i] if i < len(coeff_names) else f"β{i}": cool_dict[i]
                                 for i in range(self._rls_cool.n)},
                "observation_count": self._rls_cool.observation_count,
            },
            "pi_state": {
                "integral": self._pi_integral,
                "ff_offset": round(self._ff_offset, 4),
                "ff_confidence": round(self._ff_confidence, 4),
                "hp_setpoint": self._hp_setpoint,
                "desired_temp": self._desired_temp,
                "outdoor_temp": self._outdoor_temp,
                "room_temp_rate": round(self._room_temp_rate, 6),
                "integral_convergence": round(self._integral_convergence, 4),
                "tau_estimate": round(self._tau_estimate, 1) if self._imc_enabled else None,
            },
        }

    def get_health_status(self) -> dict[str, Any]:
        """Evaluate PI controller health and return status with alerts."""
        if not self._pi_enabled:
            return {
                "state": "Disabled",
                "alerts": ["PI controller not enabled"],
                "reasons": ["pi_disabled"],
                "alert_count": 0,
            }

        if self._entity._attr_hvac_mode == HVACMode.OFF:
            return {
                "state": "OK",
                "alerts": [],
                "reasons": [],
                "alert_count": 0,
            }

        alerts: list[str] = []
        reasons: list[str] = []
        severity = "OK"

        e = self._entity

        # Grace period: suppress comfort check for one tick after setpoint change
        if self._desired_temp != self._health_prev_desired:
            self._health_comfort_skip = 1
            self._health_prev_desired = self._desired_temp

        # Check 1: Comfort error
        if self._health_comfort_skip > 0:
            self._health_comfort_skip -= 1
        elif (
            e._attr_current_temperature is not None
            and self._desired_temp is not None
        ):
            cur_c = TemperatureConverter.convert(
                e._attr_current_temperature,
                e._attr_temperature_unit,
                UnitOfTemperature.CELSIUS,
            )
            desired_c = TemperatureConverter.convert(
                self._desired_temp,
                e._attr_temperature_unit,
                UnitOfTemperature.CELSIUS,
            )
            error_c = abs(cur_c - desired_c)
            if error_c > self.HEALTH_COMFORT_CRIT:
                severity = "Critical"
                alerts.append(
                    f"Temperature {error_c:.1f}°C from setpoint — comfort critical"
                )
                reasons.append("comfort_critical")
            elif error_c > self.HEALTH_COMFORT_WARN:
                severity = "Warning"
                alerts.append(
                    f"Temperature {error_c:.1f}°C from setpoint — comfort warning"
                )
                reasons.append("comfort_warn")

        # Check 2: PI integral correction magnitude (in °C, not raw integral)
        # With FF confidence scaling, large integrals are expected when the
        # model is adapting — the system is handling it.  Alert on the actual
        # correction the integral is applying, which is the real load.
        ki_integral = abs(self._pi_ki * self._pi_integral)
        if ki_integral > self.HEALTH_INTEGRAL_WARN:
            if severity != "Critical":
                severity = "Warning"
            alerts.append(
                f"PI integral correction {ki_integral:.1f}°C — controller struggling"
            )
            reasons.append("integral_high")

        # Check 2b: FF confidence sustained low
        if self._ff_confidence < 0.5:
            if severity != "Critical":
                severity = "Warning"
            alerts.append(
                f"FF confidence at {self._ff_confidence:.0%} — model prediction unreliable"
            )
            reasons.append("ff_confidence_low")

        # Determine active RLS model for checks 3 & 4
        is_heating = e._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
        rls = self._rls_heat if is_heating else self._rls_cool
        expected_slope = (
            self._ff_heat_slope if is_heating else -self._ff_cool_slope
        )
        has_obs = rls.observation_count > 0

        coeffs = rls.get_coefficients() if has_obs else []
        intercept = coeffs[0] if len(coeffs) > 0 else 0.0
        outdoor_slope = coeffs[1] if len(coeffs) > 1 else expected_slope

        # Check 3: RLS intercept drift
        if has_obs and abs(intercept) > self.HEALTH_INTERCEPT_WARN:
            if severity != "Critical":
                severity = "Warning"
            alerts.append(
                f"RLS intercept drifted to {intercept:.3f} (expect near 0)"
            )
            reasons.append("intercept_drift")

        # Check 4: Outdoor delta slope drift
        if has_obs and expected_slope != 0:
            drift_abs = abs(outdoor_slope - expected_slope)
            drift_pct = (drift_abs / abs(expected_slope)) * 100
            if (
                drift_pct > self.HEALTH_SLOPE_DRIFT_PCT
                and drift_abs > self.HEALTH_SLOPE_DRIFT_FLOOR
            ):
                if severity != "Critical":
                    severity = "Warning"
                alerts.append(
                    f"RLS outdoor slope {outdoor_slope:.4f} drifted "
                    f"{drift_pct:.0f}% from seed {expected_slope:.4f}"
                )
                reasons.append("slope_drift")

        # Check 5: Persistent same-direction batch correction (model drift)
        drifting = self.get_drifting_coefficients()
        if drifting:
            if severity != "Critical":
                severity = "Warning"
            for _idx, name, count in drifting:
                alerts.append(
                    f"Batch consistently correcting {name} in same direction "
                    f"({count} cycles) — possible physical change"
                )
            reasons.append("model_drift")

        return {
            "state": severity,
            "alerts": alerts,
            "reasons": reasons,
            "alert_count": len(alerts),
            "pi_integral": round(self._pi_integral, 3),
            "hp_setpoint": self._hp_setpoint,
            "ff_offset": round(self._ff_offset, 2),
            "rls_intercept": round(intercept, 4),
            "rls_outdoor_slope": round(outdoor_slope, 4),
            "expected_slope": round(expected_slope, 4),
            "rls_obs_count": rls.observation_count,
            "ff_confidence": round(self._ff_confidence, 3),
            "integral_convergence": round(self._integral_convergence, 2),
            "tau_estimate": round(self._tau_estimate, 1) if self._imc_enabled else None,
            "smith_correction": (
                round(self._smith.correction, 3) if self._smith is not None else None
            ),
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
        # Convert physical-space seeds to normalized space
        n = self._rls_heat.n
        heat_norm = [
            heat_seeds[i] * self._feature_scales[i] if i < len(heat_seeds) else 0.0
            for i in range(n)
        ]
        cool_norm = [
            cool_seeds[i] * self._feature_scales[i] if i < len(cool_seeds) else 0.0
            for i in range(n)
        ]
        self._rls_heat.beta = heat_norm
        self._rls_cool.beta = cool_norm
        # Reset covariance to uniform initial uncertainty
        for i in range(n):
            for j in range(n):
                val = DEFAULT_RLS_P_INIT if i == j else 0.0
                self._rls_heat.P[i * n + j] = val
                self._rls_cool.P[i * n + j] = val
        self._rls_heat.observation_count = 0
        self._rls_cool.observation_count = 0
        self._pi_integral = 0.0
        _LOGGER.info("FF models reset to seed values, integral zeroed")

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

    # ── IMC Gain Scheduling ─────────────────────────────────────────

    def _recompute_imc_gains(self) -> None:
        """Derive Kp and Ki from τ estimate via modified IMC tuning rule.

        IMC for first-order + delay (FOPDT):
            Kp = τ / (K_eff * (λ + L))

        Integral time uses Ti = τ/3 instead of standard Ti = τ (Skogestad SIMC
        modification for HVAC: faster integral action for disturbance rejection,
        since HVAC systems face continuous disturbances from weather/occupancy):
            Ki = Kp / (τ/3) = 3 * Kp / τ

        K_eff = 1.0 (unit gain: 1°C HP setpoint offset → 1°C room temp at SS).
        L = HP response lag (compressor → room sensor, config, default 15 min).

        λ (closed-loop speed) defaults to L/3 when not explicitly configured.
        Bench sweep across 3 house profiles (τ=25,50,120) × 6 scenarios showed:
        - λ=τ/2 (old default) too conservative: gains barely differ from flat Kp=1.5
        - λ=L/3≈5 gives 17% aggregate ITAE reduction vs flat gains, 0 regressions
        - Biggest win on well-insulated (τ=120): 64% ITAE reduction (Kp 1.5→6.0)
        - λ tied to L (not τ) because the transport delay is the physical constraint
          on how aggressively we can close the loop, regardless of house thermal mass
        Override via pi_imc_lambda config for manual tuning.
        """
        tau = max(self._tau_estimate, 1.0)  # Floor at 1 min to avoid division issues
        lag = self._response_lag
        lam = self._imc_lambda_config if self._imc_lambda_config > 0 else lag / 3.0
        k_eff = 1.0

        old_kp = self._pi_kp
        old_ki = self._pi_ki
        self._pi_kp = tau / (k_eff * (lam + lag))
        ti = tau / 3.0  # Aggressive integral time for HVAC disturbance rejection
        self._pi_ki = self._pi_kp / ti

        if old_kp > 0 and (abs(self._pi_kp - old_kp) / old_kp > 0.05):
            _LOGGER.info(
                "IMC gains updated: τ=%.1f λ=%.1f L=%.1f → Kp=%.3f Ki=%.4f (was Kp=%.3f Ki=%.4f)",
                tau, lam, lag, self._pi_kp, self._pi_ki, old_kp, old_ki,
            )

        # Keep Smith predictor model in sync with updated τ
        if self._smith is not None:
            self._smith.update_params(tau=tau, lag=lag)

    def _start_tau_observation(
        self, now_mono: float, current_c: float, desired_c: float, step_magnitude: float
    ) -> None:
        """Begin observing a step response for τ estimation.

        Called when hp_setpoint changes by ≥1°C. Records the starting conditions
        so _check_tau_observation can detect when the room reaches 63.2% of the
        expected response.
        """
        if not self._imc_enabled:
            return
        # Only observe steps with clear direction and magnitude
        if abs(step_magnitude) < 1.0:
            return
        self._tau_step_time = now_mono
        self._tau_step_temp = current_c
        self._tau_step_target = desired_c
        self._tau_step_magnitude = step_magnitude
        self._tau_step_active = True
        _LOGGER.debug(
            "τ observation started: step=%.1f°C, room=%.1f°C, target=%.1f°C",
            step_magnitude, current_c, desired_c,
        )

    def _check_tau_observation(self, now_mono: float, current_c: float) -> None:
        """Check if the room has reached 63.2% of the step response.

        τ is the time from setpoint change to 63.2% of the total expected
        temperature change (first-order system definition). On observation,
        update the running τ estimate with an EMA.
        """
        if not self._tau_step_active or self._tau_step_temp is None:
            return

        elapsed_min = (now_mono - self._tau_step_time) / 60.0

        # Timeout: if we haven't seen 63.2% response in 4× the current estimate
        # (or 4 hours if no estimate), abandon this observation.
        timeout = max(4.0 * self._tau_estimate, 240.0) if self._tau_estimate > 0 else 240.0
        if elapsed_min > timeout:
            _LOGGER.debug("τ observation timed out after %.0f min", elapsed_min)
            self._tau_step_active = False
            return

        # Expected total change: step drives room from step_temp toward target.
        # For FOPDT, the expected SS change = step_magnitude * K_eff (K_eff=1).
        expected_change = self._tau_step_magnitude  # K_eff = 1.0
        if abs(expected_change) < 0.5:
            self._tau_step_active = False
            return

        actual_change = current_c - self._tau_step_temp
        fraction = actual_change / expected_change

        # 63.2% threshold (1 - 1/e)
        if fraction >= 0.632:
            # Subtract response lag — τ is the thermal time constant, not
            # including the HP's own delay to start affecting room temp.
            raw_tau = elapsed_min - self._response_lag
            observed_tau = max(raw_tau, 5.0)  # Floor: no house has τ < 5 min

            # EMA update: weight new observations more when we have few
            n = self._tau_observations
            alpha = max(0.3, 1.0 / (1.0 + n))  # Starts at 0.5, decays to 0.3
            old_tau = self._tau_estimate
            self._tau_estimate = (1.0 - alpha) * old_tau + alpha * observed_tau
            self._tau_observations += 1
            self._tau_step_active = False

            _LOGGER.info(
                "τ observed: %.1f min (raw=%.1f, lag=%.1f). "
                "EMA τ: %.1f → %.1f min (n=%d, α=%.2f). Kp=%.3f Ki=%.4f",
                observed_tau, elapsed_min, self._response_lag,
                old_tau, self._tau_estimate, self._tau_observations, alpha,
                self._pi_kp, self._pi_ki,
            )

            # Recompute gains with updated τ
            self._recompute_imc_gains()

    def _cancel_tau_observation(self) -> None:
        """Cancel any in-progress τ observation (e.g., mode change, setpoint change)."""
        if self._tau_step_active:
            _LOGGER.debug("τ observation cancelled")
            self._tau_step_active = False

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

    async def _pi_async_sensor_changed(self, was_none: bool = False) -> bool:
        """Handle temp sensor update. Returns True if send needed."""
        if not self._pi_enabled:
            return False
        if was_none:
            # Verify the sensor actually has a numeric value — transitions from
            # None to 'unavailable' fire was_none=True but aren't real recoveries.
            if self._entity._attr_current_temperature is None:
                return False
            self._sensor_recovery_pending = False
            self._recovery_check_needed = False
            if self._sensor_unavailable:
                _LOGGER.info("%sPI: temp sensor recovered, resuming full PI control", self._log_prefix)
                self._sensor_unavailable = False
            else:
                _LOGGER.debug("PI: temp sensor just became available, running immediate tick")
            # Guard against multiple sensors coming online simultaneously
            # (e.g., temp + humidity both fire was_none=True within milliseconds).
            # Allow recovery tick only if no tick ran in the last 2 seconds.
            elapsed = time.monotonic() - self._pi_last_tick_time
            if elapsed < 2.0:
                _LOGGER.debug("PI: skipping recovery tick, another ran %.1fs ago", elapsed)
                return False
            return await self._pi_tick()

        elapsed = time.monotonic() - self._pi_last_tick_time
        min_cooldown = max(60.0, self._pi_min_interval / 3.0)
        if elapsed >= min_cooldown:
            return await self._pi_tick()
        return False

    async def _check_sensor_recovery(self, _now: datetime | None = None) -> bool:
        """Called 60s after sensor went unavailable. Returns True if send needed."""
        self._sensor_recovery_pending = False
        self._sensor_recovery_unsub = None
        e = self._entity
        if e._attr_current_temperature is not None:
            _LOGGER.info("%sPI: temp sensor recovered during grace period", self._log_prefix)
            return await self._pi_tick()
        self._sensor_unavailable = True
        _LOGGER.warning("%sPI: temp sensor confirmed unavailable, using feedforward-only fallback", self._log_prefix)
        if self._desired_temp is None:
            return False
        desired_c = TemperatureConverter.convert(
            self._desired_temp, e.temperature_unit, UnitOfTemperature.CELSIUS,
        )
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        if not is_heating and e._attr_hvac_mode not in (HVACMode.COOL, HVACMode.DRY):
            return False
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
            _LOGGER.info("%sPI fallback: setpoint %s -> %s (FF only)", self._log_prefix, self._hp_setpoint, new_setpoint)
            self._hp_setpoint = new_setpoint
            return True
        return False

    async def _pi_tick(self, now: datetime | None = None) -> bool:
        """PI + feedforward controller tick. Returns True if send needed."""
        if not self._pi_enabled:
            return False
        if self._pi_tick_running:
            _LOGGER.debug("%sPI tick: skipping, already running (reentrant call)", self._log_prefix)
            return False
        self._pi_tick_running = True
        try:
            result = await self._pi_tick_inner(now)
            # Reschedule fallback timer after every tick
            if self._pi_timer_unsub:
                self._pi_timer_unsub()
            if self._pi_timer_callback:
                self._pi_timer_unsub = async_call_later(
                    self._hass, self._pi_min_interval, self._pi_timer_callback)
            return result
        finally:
            self._pi_tick_running = False

    async def _pi_tick_inner(self, now: datetime | None = None) -> bool:
        """PI + feedforward controller tick implementation. Returns True if send needed."""
        e = self._entity
        if e._attr_hvac_mode == HVACMode.OFF:
            self._pi_integral = 0.0
            self._cancel_tau_observation()
            if self._smith is not None:
                self._smith._initialized = False
            return False
        if self._desired_temp is None or self._hp_setpoint is None:
            return False
        if e._attr_current_temperature is None:
            if self._sensor_unavailable or self._sensor_recovery_pending:
                return False
            _LOGGER.info("%sPI: temp sensor unavailable, requesting 60s recovery check", self._log_prefix)
            self._sensor_recovery_pending = True
            self._recovery_check_needed = True
            return False
        if self._pi_paused:
            _LOGGER.debug("%sPI tick: skipping, paused by vendor", self._log_prefix)
            return False

        # Time since last tick (for time-normalized integral)
        now_mono = time.monotonic()
        if self._pi_last_tick_time > 0:
            dt_seconds = min(now_mono - self._pi_last_tick_time, self._pi_min_interval * 2)
        else:
            dt_seconds = float(self._pi_min_interval)
        self._pi_last_tick_time = now_mono
        dt_factor = dt_seconds / float(self._pi_min_interval)

        # Convert both to °C for PI math
        raw_c = TemperatureConverter.convert(
            e._attr_current_temperature,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )
        desired_c = TemperatureConverter.convert(
            self._desired_temp,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )

        # Low-pass filter on room temperature measurement.
        # Reduces sensor noise amplified through Kp.  Uses raw reading for
        # room_temp_history (rate calc, RLS gate) so those reflect reality.
        # α = 1 - exp(-dt/τ): short dt → small α (gentle), long dt → large α.
        if self._sensor_filter_tau > 0 and dt_seconds > 0:
            if self._sensor_filtered is None:
                self._sensor_filtered = raw_c  # Initialize on first reading
            alpha = 1.0 - math.exp(-dt_seconds / self._sensor_filter_tau)
            self._sensor_filtered = alpha * raw_c + (1.0 - alpha) * self._sensor_filtered
            current_c = self._sensor_filtered
        else:
            current_c = raw_c

        error = desired_c - current_c

        # Track room temperature rate of change (°C/min) from RAW readings.
        # Keep last 5 readings (~5 ticks). Compute rate from oldest to newest.
        self._room_temp_history.append((now_mono, raw_c))
        if len(self._room_temp_history) > 5:
            self._room_temp_history.pop(0)
        if len(self._room_temp_history) >= 2:
            t0, temp0 = self._room_temp_history[0]
            t1, temp1 = self._room_temp_history[-1]
            elapsed_min = (t1 - t0) / 60.0
            if elapsed_min > 0:
                self._room_temp_rate = (temp1 - temp0) / elapsed_min

        # Check ongoing τ step-response observation (raw — measures real plant)
        self._check_tau_observation(now_mono, raw_c)

        # Evaluate supplemental heat source override (selector control)
        now_mono = time.monotonic()
        hp_should_send_ir = self._evaluate_supplemental_override(error, now_mono)

        # Smith predictor: compensate transport delay (Åström Ch. 7 §7.3,
        # I-PI variant per Normey-Rico & Camacho 2007).
        #
        # The correction term (nodelay − delayed model) represents temperature
        # change in the pipeline.  Applied ONLY to the P-term — the integral
        # accumulates on raw error for robustness to model mismatch:
        #   - Deadband uses raw error (physical room state)
        #   - P-term uses Smith-corrected error (anticipatory action)
        #   - Integral uses raw error (correct steady-state, mismatch-robust)
        #   - Metrics use raw error (actual comfort)
        smith_correction = 0.0
        if self._smith is not None:
            if not self._smith._initialized:
                self._smith.initialize(raw_c, float(self._hp_setpoint), now_mono)
            self._smith.record_setpoint(float(self._hp_setpoint), now_mono)
            self._smith.step(float(self._hp_setpoint), dt_seconds, now_mono)
            smith_correction = self._smith.correction

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
            return False

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
        # With few observations the RLS may have learned from narrow conditions
        # (e.g. only mild weather) and extrapolation can be wrong. The blend
        # anchors predictions to seeds until enough observations have covered
        # a representative range of conditions (~1-2 weeks at ~6 obs/day).
        MIN_RLS_OBS = 50
        seed_offset = sum(s * xi for s, xi in zip(seeds, x))
        rls_offset = rls.predict(x)
        alpha = min(rls.observation_count / MIN_RLS_OBS, 1.0)
        blended_offset = (1.0 - alpha) * seed_offset + alpha * rls_offset

        # Integral-based FF confidence: when the integral opposes the FF
        # offset direction, the model prediction is wrong in sign/magnitude
        # and the feedback loop is fighting it.  Scale FF down so the
        # integral has less to correct, accelerating convergence.
        #
        # Only activates when integral OPPOSES FF — meaning FF predicts
        # an offset the integral is trying to undo.  When they agree
        # (both wanting more/less heat), the model direction is right
        # and reducing FF would worsen an undersized-HP situation.
        #
        # Threshold of 3°C: below this, full FF trust (the integral is
        # handling normal residuals).  Above, FF scales smoothly toward
        # the integral-corrected value.  EMA-smoothed to prevent
        # limit cycling at integer setpoint boundaries.
        FF_CONFIDENCE_THRESHOLD = 3.0
        integral_opposes_ff = (self._pi_integral * blended_offset) < 0
        if integral_opposes_ff:
            model_error = abs(self._pi_ki * self._pi_integral)
            raw_confidence = 1.0 / (1.0 + max(0.0, model_error - FF_CONFIDENCE_THRESHOLD) / FF_CONFIDENCE_THRESHOLD)
        else:
            raw_confidence = 1.0
        # EMA smoothing (~10 ticks ≈ 2.5h) prevents tick-to-tick jitter
        self._ff_confidence += 0.1 * (raw_confidence - self._ff_confidence)
        self._ff_offset = blended_offset * self._ff_confidence

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

        # Deadband: if error is small, skip P term but integrate at reduced rate.
        # Variable-rate integration replaces the old integral freeze (Åström §3.5
        # warns against stopping integration near setpoint). Leaky decay bounds
        # integral growth universally. P=0 in deadband is preserved — simulation
        # confirmed it prevents quantization-driven limit cycles.
        in_deadband = abs_error < self._pi_deadband
        avg_error = (error + self._pi_last_error) / 2.0

        # One-sided actuator anti-windup (Astrom & Hagglund, "Advanced PID
        # Control" §3.5 — conditional integration): heat-only HP cannot cool,
        # cool-only cannot heat.  When the integral has wound up significantly
        # in the direction the actuator cannot act, accelerated decay limits
        # the windup while still allowing brief overshoot correction.
        # Threshold: |ki*integral| > deadband in the constrained direction.
        _ki_integral = self._pi_ki * self._pi_integral
        integral_wound_against_actuator = (
            (is_heating and error < 0 and _ki_integral < -self._pi_deadband)
            or (is_cooling and error > 0 and _ki_integral > self._pi_deadband)
        )

        # Conditional integration (Åström & Hägglund, "Advanced PID
        # Control" §3.5): freeze the integrator when the HP is at the
        # limit opposite to what its mode can deliver AND the error is
        # in the direction the actuator can't help.
        #
        # Heating at min + room above target: HP can only heat but the
        # room is already too hot.  Integration is pointless and creates
        # an integral debt that delays recovery at sunset.
        #
        # Heating at min + room BELOW target: HP IS helping (heating a
        # cold room at its minimum output).  Integration must continue
        # so the integral can recover and drive the setpoint up.
        #
        # Symmetric for cooling mode at max setpoint.
        skip_integration = (
            (is_heating and self._hp_setpoint <= self._min_temp_c and error < 0)
            or (is_cooling and self._hp_setpoint >= self._max_temp_c and error > 0)
        )

        # Log transitions into/out of conditional integration freeze.
        if skip_integration and not self._integration_frozen:
            _LOGGER.debug(
                "%sIntegration frozen: %s at %s limit, error=%.2f°C",
                self._log_prefix,
                "heating" if is_heating else "cooling",
                "min" if is_heating else "max",
                error,
            )
        elif not skip_integration and self._integration_frozen:
            _LOGGER.debug(
                "%sIntegration unfrozen: error=%.2f°C, setpoint=%.1f°C",
                self._log_prefix, error, self._hp_setpoint,
            )
        self._integration_frozen = skip_integration

        if in_deadband:
            self._ff_settled_ticks += 1
            # Variable-rate: smooth taper from full rate at deadband edge to 5%
            # floor at setpoint. Prevents integral starvation while reducing
            # accumulation speed near target.
            rate = max(0.05, min(1.0, abs_error / self._pi_deadband))
            if not skip_integration:
                self._pi_integral += avg_error * dt_factor * rate
            # No accelerated decay in deadband — small overshoots are normal
            # control behavior, not actuator saturation.  The variable-rate
            # integration (5% floor) already limits accumulation speed.
            # RLS learning gate: require integral settled and room temp stable.
            # The old gate also required rate < 0.2 (error < 10% of deadband),
            # but that starved zones with coarse sensors (1°F) that only briefly
            # sit at exact target.  Being inside deadband (< 0.5°C) with stable
            # room temp and low integral change is sufficient for a valid
            # observation.
            integral_change = abs(self._pi_integral - self._prev_integral_for_rls)
            integral_settling = integral_change < 0.3
            # Room temperature must be genuinely settled — not coasting from a
            # recent setpoint change or external disturbance.
            room_settling = abs(self._room_temp_rate) < 0.02  # °C/min
            can_learn_rls = (
                self._ff_settled_ticks >= 4
                and self._outdoor_temp is not None
                and not learning_suppressed
                and integral_settling
                and room_settling
                and not self._any_model_input_unavailable()
                and not self._tracking_mode
                and not self._supplemental_assist_active
            )
            if not can_learn_rls and self._ff_settled_ticks >= 4 and self._ff_settled_ticks % 4 == 0:
                # Log why learning was blocked (every 4th tick ≈ 60min)
                reasons = []
                if self._outdoor_temp is None:
                    reasons.append("no outdoor temp")
                if learning_suppressed:
                    reasons.append("manually suppressed")
                if not integral_settling:
                    reasons.append(f"integral not settled (rate={rate:.2f}, d_integral={integral_change:.2f})")
                if not room_settling:
                    reasons.append(f"room not settled (dT/dt={self._room_temp_rate:.4f} °C/min)")
                if self._any_model_input_unavailable():
                    reasons.append("model input unavailable")
                if (is_heating and error < 0) or (is_cooling and error > 0):
                    reasons.append(
                        f"actuator no authority ({'heating' if is_heating else 'cooling'}, error={error:.2f})"
                    )
                if reasons:
                    _LOGGER.debug("RLS learning blocked: %s", ", ".join(reasons))
            if can_learn_rls:
                # Observe what the plant actually experienced: the integer
                # HP setpoint that maintained the target temperature.
                # This is a direct input-output observation at the operating
                # point (Ljung, System Identification §7.4).
                observed_offset = float(self._hp_setpoint) - desired_c
                beta_before = list(rls.beta)
                residual = rls.update(x, observed_offset)
                ki_integral = self._pi_ki * self._pi_integral
                _LOGGER.debug(
                    "RLS update: observed=%.2f predicted=%.2f residual=%.2f obs_count=%d "
                    "ki_integral=%.3f dT_dt=%.4f",
                    observed_offset, observed_offset - residual, residual,
                    rls.observation_count, ki_integral, self._room_temp_rate,
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
            # Out-of-deadband steady-state learning: when the room is at
            # thermal equilibrium but outside the control deadband, feed a
            # corrected observation to the RLS.  This breaks the vicious cycle
            # where a miscalibrated model keeps the room away from target,
            # preventing the normal learning gate from opening.
            #
            # Corrected observation: hp_setpoint - current_c tells the RLS
            # "what offset maintains room temp at current conditions."  At
            # equilibrium this mapping is valid at any operating point (Ljung,
            # System Identification §7.4).  Bias grows with distance from
            # target (~K_loss/K_hp × |error|), so we require proportionally
            # more settling ticks at larger errors.
            #
            # Gate on clamping: if the HP setpoint is at min/max the
            # observation is censored — the controller wanted to go further
            # but couldn't — so the learned offset would be biased.
            room_stable = abs(self._room_temp_rate) < 0.015  # stricter than deadband gate
            # Use a separate integral baseline for oodb — the deadband
            # tracker (_prev_integral_for_rls) goes stale for zones that
            # rarely enter deadband, making integral_change enormous.
            # This tracker updates every tick outside deadband so it
            # measures RECENT integral movement, not cumulative drift.
            integral_change = abs(self._pi_integral - self._prev_integral_for_oodb)
            integral_stable = integral_change < 0.5  # integral not actively winding
            self._prev_integral_for_oodb = self._pi_integral
            setpoint_clamped = (
                self._hp_setpoint <= self._min_temp_c
                or self._hp_setpoint >= self._max_temp_c
            )
            # Require more settling at larger errors: the equilibrium offset
            # has bias ~(K_loss/K_hp)×|error| from using current_c instead of
            # desired_c (linear heat-loss approximation).  Longer settling
            # ensures we only learn from genuinely stable states at larger
            # deviations, and naturally reduces the frequency of noisier
            # far-from-target observations.
            min_oodb_ticks = 8 + int(abs_error * 4)  # +4 ticks per °C of error
            if room_stable and integral_stable and not setpoint_clamped:
                self._stable_oodb_ticks += 1
            else:
                if self._stable_oodb_ticks > 0:
                    reasons = []
                    if not room_stable:
                        reasons.append(f"room not settled (dT/dt={self._room_temp_rate:.4f})")
                    if not integral_stable:
                        reasons.append(f"integral not settled (d_integral={integral_change:.2f})")
                    if setpoint_clamped:
                        reasons.append("setpoint clamped")
                    _LOGGER.debug(
                        "OODB gate reset at %d/%d ticks: %s",
                        self._stable_oodb_ticks, min_oodb_ticks,
                        ", ".join(reasons) if reasons else "conditions changed",
                    )
                self._stable_oodb_ticks = 0
            if (
                self._stable_oodb_ticks >= min_oodb_ticks
                and self._outdoor_temp is not None
                and not learning_suppressed
                and not self._any_model_input_unavailable()
                and not self._tracking_mode
                and not self._supplemental_assist_active
            ):
                corrected_offset = float(self._hp_setpoint) - current_c
                beta_before = list(rls.beta)
                residual = rls.update(x, corrected_offset)
                _LOGGER.debug(
                    "RLS steady-state update (oodb): observed=%.2f predicted=%.2f "
                    "residual=%.2f obs_count=%d error=%.2f dT_dt=%.4f",
                    corrected_offset, corrected_offset - residual, residual,
                    rls.observation_count, error, self._room_temp_rate,
                )
                for idx in range(len(rls.beta)):
                    if beta_before[idx] != 0 and abs(rls.beta[idx] - beta_before[idx]) / abs(beta_before[idx]) > 0.1:
                        _LOGGER.info(
                            "RLS coefficient[%d] changed %.3f -> %.3f (%.0f%%)",
                            idx, beta_before[idx], rls.beta[idx],
                            100 * (rls.beta[idx] - beta_before[idx]) / beta_before[idx],
                        )
                self._stable_oodb_ticks = 0  # one observation per settled window

            # Setpoint weighting: reduce P-term to prevent overshoot while
            # integral drives steady-state accuracy. Standard 2-DOF technique
            # (Astrom & Hagglund). b < 1 reduces proportional kick.
            # Smith predictor I-PI variant: only apply correction when it
            # prevents overshoot (same sign as error). When signs differ, the
            # correction would fight the current control direction — ignore it.
            # Heating + warming in pipeline → reduce P (prevent overshoot). ✓
            # Heating + cooling in pipeline → ignore (don't fight recovery). ✓
            effective_smith = smith_correction if smith_correction * error > 0 else 0.0
            p_term = self._pi_kp * self._pi_setpoint_weight * (error - effective_smith)
            if not skip_integration:
                self._pi_integral += avg_error * dt_factor
            if integral_wound_against_actuator and not skip_integration:
                # Accelerated decay: exponential toward zero limits windup
                # while still allowing the integral to correct brief
                # overshoot.  Time constant matches the building's thermal
                # response (Astrom back-calculation Tt ≈ Ti; tau_estimate
                # ~60min is same order as Ti = 1/ki ~100min).  Floor at
                # 10min prevents bumps on recovery if tau is very small.
                tau_min = max(
                    self._tau_estimate if self._tau_estimate > 0 else 60.0,
                    10.0,
                )
                self._pi_integral *= math.exp(-(dt_seconds / 60.0) / tau_min)

        # Leaky integrator: weak decay bounds integral growth universally.
        # α=0.9999 per nominal tick ≈ 10000-tick time constant (~104 days at
        # 15-min ticks). Scaled by dt_factor for variable sample intervals.
        # Weaker than α=0.999 to avoid draining integral correction needed by
        # slow-τ houses (well-insulated). Sim-validated across 3 profiles.
        self._pi_integral *= 0.9999 ** dt_factor

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

            # Controllable/uncontrollable split: "uncontrollable" when the HP
            # is clamped at its limit in the direction that would help.
            # Heating + room above target + at min → can't cool further.
            # Cooling + room below target + at max → can't heat further.
            saturated_wrong_end = (
                (is_heating and error < 0 and self._hp_setpoint <= self._min_temp_c)
                or (is_cooling and error > 0 and self._hp_setpoint >= self._max_temp_c)
            )
            itae_increment = self._itae_tick_count * effective_error
            if saturated_wrong_end:
                self._uncontrollable_itae += itae_increment
                if abs_error > 1.0:
                    self._uncontrollable_cvh += dt_seconds / 3600.0
            else:
                self._controllable_itae += itae_increment
                if abs_error > 1.0:
                    self._controllable_cvh += dt_seconds / 3600.0

        # FF load fraction: how much of the control effort comes from the
        # feedforward model vs the integral.  EMA ~24h (alpha ≈ 0.01).
        i_correction = abs(self._pi_ki * self._pi_integral)
        ff_mag = abs(self._ff_offset)
        total_effort = ff_mag + i_correction
        if total_effort > 0.1:  # avoid noise when both are near zero
            instant_ff_load = ff_mag / total_effort
            self._ff_load_fraction += 0.01 * (instant_ff_load - self._ff_load_fraction)

        i_term = self._pi_ki * self._pi_integral
        d_term = self._pi_d_filtered
        raw_setpoint = desired_c + p_term + i_term + d_term + self._ff_offset
        self._last_raw_setpoint = raw_setpoint
        clamped_setpoint = max(self._min_temp_c, min(self._max_temp_c, raw_setpoint))

        # Back-calculation anti-windup: cap integral at the value that
        # produces the clamped output.  Skipped when conditional integration
        # already froze the integrator — the two mechanisms serve the same
        # purpose and the freeze is the tighter constraint.
        if clamped_setpoint != raw_setpoint and self._pi_ki != 0 and not skip_integration:
            if raw_setpoint > clamped_setpoint and self._pi_integral > 0:
                max_i = (clamped_setpoint - desired_c - p_term - self._ff_offset) / self._pi_ki
                self._pi_integral = min(self._pi_integral, max_i)
            elif raw_setpoint < clamped_setpoint and self._pi_integral < 0:
                min_i = (clamped_setpoint - desired_c - p_term - self._ff_offset) / self._pi_ki
                self._pi_integral = max(self._pi_integral, min_i)

        # Quantization-error feedback: nudge integral so clamped_setpoint
        # lands near an integer, preventing limit cycles from 1°C HP steps.
        # Only acts on small misalignments (≤ 0.5°C, half a step) — large
        # gaps are real integral corrections, not quantization artifacts.
        if in_deadband and self._pi_ki != 0:
            q_error = float(self._hp_setpoint) - clamped_setpoint
            if 0.3 < abs(q_error) <= 0.5:
                self._pi_integral += (q_error / self._pi_ki) * 0.4

        # Record observation for batch learning (every tick, regardless of gate)
        self._observation_buffer.add(Observation(
            timestamp=now_mono,
            features=list(x),
            hp_setpoint=float(self._hp_setpoint),
            current_c=current_c,
            desired_c=desired_c,
            room_rate=self._room_temp_rate,
            clamped=(
                self._hp_setpoint <= self._min_temp_c
                or self._hp_setpoint >= self._max_temp_c
            ),
            pi_integral=self._pi_integral,
            ff_offset=self._ff_offset,
            ff_confidence=self._ff_confidence,
            raw_c=raw_c,
        ))

        # Midpoint-crossing hysteresis
        new_setpoint = self._hp_setpoint
        if clamped_setpoint > self._hp_setpoint + 0.5:
            new_setpoint = round(clamped_setpoint)
        elif clamped_setpoint < self._hp_setpoint - 0.5:
            new_setpoint = round(clamped_setpoint)
        new_setpoint = int(max(self._min_temp_c, min(self._max_temp_c, new_setpoint)))

        if new_setpoint != self._hp_setpoint:
            # Minimum dwell time: wait after a setpoint change before allowing
            # another.  A 1°C change takes ~15-25 min to propagate through the
            # HP response chain (compressor → heat exchanger → room → sensor).
            # Without a hold the PI reacts to incomplete information and
            # oscillates in the 0.3-1.0°C error band.
            #
            # With Smith predictor active the pipeline effect is modelled
            # explicitly, so the hold can be shorter (10 min safety net vs
            # the original 30 min).  Without Smith the 10-min hold still
            # works because the 1°C urgent bypass covers large errors and
            # hysteresis prevents sub-step changes.
            #
            # Bypass: error >1°C skips the hold (urgent demand).
            change = new_setpoint - self._hp_setpoint
            time_since_last = now_mono - self._last_setpoint_change_time
            can_change = time_since_last >= self._SETPOINT_HOLD_SECONDS
            if abs_error > 1.0:
                can_change = True  # Large error = urgent demand, bypass hold
            if not can_change:
                _LOGGER.debug(
                    "%sPI: setpoint %s -> %s held (%.0fs since last change, need %.0fs)",
                    self._log_prefix, self._hp_setpoint, new_setpoint, time_since_last,
                    self._SETPOINT_HOLD_SECONDS,
                )
            else:
                old_setpoint = self._hp_setpoint
                self._hp_setpoint = new_setpoint
                if not hp_should_send_ir:
                    # Tracking mode: update internal setpoint but don't send IR
                    _LOGGER.debug(
                        "%sPI tracking: setpoint %s -> %s (IR suppressed, override by %s)",
                        self._log_prefix, old_setpoint, new_setpoint, ", ".join(self._tracking_sources),
                    )
                else:
                    _LOGGER.info(
                        "%sPI: error=%.1f smith=%.2f P=%.1f I=%.1f D=%.1f FF=%.1f raw=%.1f setpoint %s -> %s",
                        self._log_prefix, error, smith_correction, p_term, i_term, d_term, self._ff_offset,
                        clamped_setpoint, old_setpoint, new_setpoint,
                    )
                    self._last_setpoint_change_time = now_mono
                    self._setpoint_changes += 1
                    # Start τ observation on significant setpoint changes
                    self._start_tau_observation(now_mono, current_c, desired_c, float(change))
                    return True
        else:
            _LOGGER.debug(
                "%sPI: error=%.1f raw=%.1f setpoint=%s (held)",
                self._log_prefix, error, clamped_setpoint, self._hp_setpoint,
            )

        return False
