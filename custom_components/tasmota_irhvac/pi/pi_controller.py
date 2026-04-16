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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import CALLBACK_TYPE, Event, EventStateChangedData, HomeAssistant, State

    from ..climate import TasmotaIrhvac

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.restore_state import ExtraStoredData
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_time_change,
)
from homeassistant.util.unit_conversion import TemperatureConverter

import math

from .batch_learning import BatchResult, DiversityAwareBuffer, Observation, ObservationBuffer, weighted_least_squares, compare_and_report, compute_blended_update

from ..const import (
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

from ..const import DEFAULT_RLS_P_INIT
from .rls_model import RLSModel
from .pi_stored_data import PIExtraStoredData
from .model_input_manager import ModelInputManager
from .health_checks import (
    check_comfort,
    check_feature_diversity,
    check_ff_confidence,
    check_integral,
    check_intercept_drift,
    check_model_drift,
    check_slope_drift,
)
from .performance_metrics import PerformanceMetrics
from .smith_predictor import SmithPredictor
from .supplemental_controller import SupplementalController
from .tau_estimator import GainUpdate, TauEstimator

_LOGGER = logging.getLogger(__name__)


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
    HEALTH_FEATURE_DIVERSITY_MIN: float = 0.05   # 5% activity floor per feature
    HEALTH_FEATURE_DIVERSITY_MIN_OBS: int = 100   # don't fire until enough data

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

        # IMC gain scheduling + τ estimation
        self._tau_estimator = TauEstimator(
            tau_seed=config.get(CONF_PI_TAU_ESTIMATE, DEFAULT_PI_TAU_ESTIMATE),
            response_lag=config.get(CONF_PI_RESPONSE_LAG, DEFAULT_PI_RESPONSE_LAG),
            imc_lambda=config.get(CONF_PI_IMC_LAMBDA, DEFAULT_PI_IMC_LAMBDA),
        )

        # Smith predictor for dead-time compensation.
        # Requires IMC (tau > 0) AND pi_smith_enabled=true.
        self._smith: SmithPredictor | None = None
        if self._tau_estimator.enabled and self._smith_enabled:
            self._smith = SmithPredictor(
                tau=self._tau_estimator.tau, lag=self._tau_estimator.response_lag
            )

        # Derive effective Kp/Ki: IMC formula or manual config
        self._pi_kp: float = 0.0
        self._pi_ki: float = 0.0
        if self._tau_estimator.enabled:
            gains = self._tau_estimator.compute_gains()
            self._pi_kp = gains.kp
            self._pi_ki = gains.ki
        else:
            self._pi_kp = self._pi_kp_config
            self._pi_ki = self._pi_ki_config

        # Control parameters are stored in °C always — read directly, no conversion
        self._pi_deadband: float = config.get(CONF_PI_DEADBAND, DEFAULT_PI_DEADBAND)
        self._ff_heat_reference: float = config.get(CONF_PI_FF_HEAT_REFERENCE, DEFAULT_PI_FF_HEAT_REFERENCE)
        self._ff_cool_reference: float = config.get(CONF_PI_FF_COOL_REFERENCE, DEFAULT_PI_FF_COOL_REFERENCE)
        self._pi_setpoint_weight: float = config.get(CONF_PI_SETPOINT_WEIGHT, DEFAULT_PI_SETPOINT_WEIGHT)

        # Feedforward config
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
        self._pi_timer_unsub: CALLBACK_TYPE | None = None
        self._pi_timer_callback: Callable[[datetime], None] | None = None
        self._pi_last_error: float = 0.0
        self._pi_d_filtered: float = 0.0    # Filtered derivative term
        self._pi_last_measurement: float | None = None  # Previous temperature measurement for derivative
        self._sensor_filtered: float | None = None  # Low-pass filtered room temp (°C)
        self._last_raw_setpoint: float = 0.0  # Pre-quantization setpoint from last tick

        # Supplemental heat source selector/override control
        supplemental_sources: list[dict[str, Any]] = config.get("pi_supplemental_sources", [])
        self._supplemental = SupplementalController(
            sources=supplemental_sources,
            deadband=self._pi_deadband,
        )

        # Model inputs (replaces disturbance inputs for RLS)
        # Each: {"name": str, "entity_id": str, "seed_heat": float, "seed_cool": float,
        #         "clamp_min": float, "clamp_max": float, "lag_tau": float (seconds)}
        self._model_inputs: list[dict[str, Any]] = list(config.get(CONF_PI_MODEL_INPUTS, []))

        # Auto-generate model inputs from supplemental sources with auto_model_input=true.
        # These use the supplemental's climate entity as a binary signal (heat/cool=1, else=0).
        # The PI controller reads the entity state each tick to update the value.
        for source in supplemental_sources:
            if not source.get("auto_model_input", True):
                continue
            entity_id = source.get("entity_id", "")
            name = source.get("name", entity_id)
            # Check if user already has a manual model input for this entity
            existing = any(m.get("entity_id") == entity_id for m in self._model_inputs)
            if existing:
                _LOGGER.debug("Supplemental %s: skipping auto model input (manual input exists)", name)
                continue
            self._model_inputs.append({
                "name": f"{name} (auto)",
                "entity_id": entity_id,
                "seed_heat": float(source.get("seed_heat", -3.0)),
                "seed_cool": float(source.get("seed_cool", 0.0)),
                "lag_tau": 0,
                "suppress_learning": True,  # Learning deferred per research
                "_auto_supplemental": True,  # Internal flag for filtering
            })
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

        # Model input runtime state (values, lag filters, outdoor temp)
        self._inputs = ModelInputManager(
            model_inputs=self._model_inputs,
            outdoor_temp_sensor=config.get(CONF_OUTDOOR_TEMP_SENSOR),
        )

        # Health check state
        self._health_prev_desired: float | None = None
        self._health_comfort_skip: int = 0

        # Performance metrics (ITAE, CVH, FF load fraction, integral convergence)
        self._metrics = PerformanceMetrics()

        # RLS learning gate: track integral stability
        self._prev_integral_for_rls: float = 0.0
        # Out-of-deadband steady-state learning: consecutive ticks with
        # low room_temp_rate while outside deadband but within learning zone.
        self._stable_oodb_ticks: int = 0
        self._prev_integral_for_oodb: float = 0.0  # separate from deadband tracker

        # FF confidence (EMA-smoothed to prevent limit cycling from
        # tick-to-tick confidence changes near integer setpoint boundaries)
        self._ff_confidence: float = 1.0

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
        self._batch_analysis_timer: CALLBACK_TYPE | None = None
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
        if self._inputs.outdoor_temp_sensor:
            async_track_state_change_event(
                self._hass,
                self._inputs.outdoor_temp_sensor,
                self._async_outdoor_temp_changed,
            )
            outdoor_state = self._hass.states.get(self._inputs.outdoor_temp_sensor)
            if outdoor_state is not None:
                unit = outdoor_state.attributes.get("unit_of_measurement", UnitOfTemperature.CELSIUS)
                self._inputs.update_outdoor_temp(outdoor_state.state, unit)

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
        if self._inputs.outdoor_temp is not None:
            is_heating = e._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
            outdoor_delta = self._ff_heat_reference - self._inputs.outdoor_temp if is_heating else self._inputs.outdoor_temp - self._ff_cool_reference
            outdoor_delta = max(0, outdoor_delta)
            x = self._inputs.build_feature_vector(outdoor_delta)
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
        """Schedule batch WLS analysis at 07:00 and 19:00 local time.

        Uses wall-clock scheduling so reboots don't reset the countdown.
        """

        @callback
        def _run_batch(_now: datetime) -> None:
            self._run_batch_analysis()

        self._batch_analysis_timer = async_track_time_change(
            self._hass, _run_batch, hour=(7, 19), minute=0, second=0,
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
        self._metrics.batch_model_rms = result.residual_rms

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

    # ── Buffer / batch sensor properties ──────────────────────────────

    @property
    def buffer_eligible(self) -> int:
        """Count of eligible (unclamped, low-rate) observations in the buffer."""
        obs = self._observation_buffer.get_all()
        return sum(1 for o in obs if not o.clamped and abs(o.room_rate) < 0.02)

    @property
    def buffer_total(self) -> int:
        """Total observations currently in the buffer."""
        return len(self._observation_buffer)

    @property
    def buffer_oldest_age_hours(self) -> float | None:
        """Age of the oldest observation in hours, or None if buffer is empty."""
        obs = self._observation_buffer.get_all()
        if not obs:
            return None
        import time as time_mod

        now = time_mod.monotonic()
        oldest = min(o.timestamp for o in obs)
        return round((now - oldest) / 3600, 1)

    @property
    def buffer_leverage_max(self) -> float | None:
        """Maximum leverage score in the buffer, or None if unavailable."""
        if not hasattr(self._observation_buffer, "get_leverage_scores"):
            return None
        scores = self._observation_buffer.get_leverage_scores()
        if not scores:
            return None
        return round(max(scores), 6)

    @property
    def batch_outliers_excluded(self) -> int | None:
        """Number of outliers excluded in the most recent batch run."""
        if self._last_batch_result is None:
            return None
        return self._last_batch_result.n_outliers_excluded

    # ── Hook methods (called by climate entity) ──────────────────────

    def get_extra_stored_data(self) -> PIExtraStoredData | None:
        """Return PI data for RestoreEntity's ExtraStoredData persistence."""
        if not self._pi_enabled:
            return None
        return PIExtraStoredData(
            pi_integral=self._pi_integral,
            desired_temp=self._desired_temp,
            hp_setpoint=self._hp_setpoint,
            integral_convergence=self._metrics.integral_convergence,
            itae_accumulator=self._metrics.itae_accumulator,
            comfort_violation_hours=self._metrics.comfort_violation_hours,
            setpoint_changes=self._metrics.setpoint_changes,
            controllable_itae=self._metrics.controllable_itae,
            uncontrollable_itae=self._metrics.uncontrollable_itae,
            controllable_cvh=self._metrics.controllable_cvh,
            uncontrollable_cvh=self._metrics.uncontrollable_cvh,
            ff_load_fraction=self._metrics.ff_load_fraction,
            rls_heat_model=self._rls_heat.as_dict(),
            rls_cool_model=self._rls_cool.as_dict(),
            lag_filter_states=self._inputs.get_lag_states(),
            heat_seeds_at_learn=list(self._heat_seeds),
            cool_seeds_at_learn=list(self._cool_seeds),
            ki_at_save=self._pi_ki,
            tau_estimate=self._tau_estimator.tau,
            observation_buffer=self._observation_buffer.as_list(),
            drift_correction_signs=self._drift_correction_signs,
            last_batch_result=(
                {
                    **dataclasses.asdict(self._last_batch_result),
                    "held_features": list(self._last_batch_result.held_features),
                }
                if self._last_batch_result is not None
                else None
            ),
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
        self._metrics.integral_convergence = data.integral_convergence
        self._metrics.itae_accumulator = data.itae_accumulator
        self._metrics.comfort_violation_hours = data.comfort_violation_hours
        self._metrics.setpoint_changes = data.setpoint_changes
        self._metrics.controllable_itae = data.controllable_itae
        self._metrics.uncontrollable_itae = data.uncontrollable_itae
        self._metrics.controllable_cvh = data.controllable_cvh
        self._metrics.uncontrollable_cvh = data.uncontrollable_cvh
        self._metrics.ff_load_fraction = data.ff_load_fraction
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
        # Restore drift detection history
        if data.drift_correction_signs:
            self._drift_correction_signs = data.drift_correction_signs
        # Restore last batch result for diagnostics continuity
        if data.last_batch_result is not None:
            br = data.last_batch_result
            # Convert held_features back to set (serialized as list)
            if "held_features" in br and isinstance(br["held_features"], list):
                br["held_features"] = set(br["held_features"])
            self._last_batch_result = BatchResult(**br)
            self._metrics.batch_model_rms = self._last_batch_result.residual_rms
        # Seed change detection: if user edited a seed since last save,
        # reset that coefficient to the new seed and increase its uncertainty.
        # Coefficients with unchanged seeds keep their learned values.
        self._apply_seed_changes(data.heat_seeds_at_learn, self._heat_seeds, self._rls_heat)
        self._apply_seed_changes(data.cool_seeds_at_learn, self._cool_seeds, self._rls_cool)
        # Restore lag filter states
        if data.lag_filter_states:
            self._inputs.restore_lag_states(data.lag_filter_states)
        # Restore τ estimate and recompute IMC gains
        if self._tau_estimator.enabled and data.tau_estimate > 0:
            old_ki = self._pi_ki
            gains = self._tau_estimator.restore(data.tau_estimate)
            self._apply_gain_update(gains)
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
        self._tau_estimator.cancel_observation()
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
            "tracking_mode": self._supplemental.tracking_mode,
            "tracking_sources": self._supplemental.tracking_sources,
            "supplemental_assist": self._supplemental.assist_active,
            ATTR_DESIRED_TEMP: self._desired_temp,
            ATTR_FF_OFFSET: round(self._ff_offset, 2),
            "rls_heat_coefficients": rls_heat_coeffs,
            "rls_cool_coefficients": rls_cool_coeffs,
            "rls_observation_count": self._rls_heat.observation_count,
            "ff_learning_suppressed": self._disturbance_suppress_active,
            "integral_convergence": round(self._metrics.integral_convergence, 2),
            "room_temp_rate": round(self._room_temp_rate, 4),  # °C/min
            "effective_kp": round(self._pi_kp, 3),
            "effective_ki": round(self._pi_ki, 4),
            "tau_estimate": round(self._tau_estimator.tau, 1) if self._tau_estimator.enabled else None,
            "tau_observations": self._tau_estimator.observations if self._tau_estimator.enabled else None,
            "smith_correction": (
                round(self._smith.correction, 3) if self._smith is not None else None
            ),
            "setpoint_changes_total": self._metrics.setpoint_changes,
            "sensor_filtered": (
                round(self._sensor_filtered, 3)
                if self._sensor_filtered is not None else None
            ),
        }

    def _coeff_names(self) -> list[str]:
        """Build coefficient name list: intercept, outdoor_delta, then model inputs."""
        names = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            names.append(m.get("name", "input"))
        return names

    def get_diagnostic_dump(self) -> dict[str, Any]:
        """Return full diagnostic state for offline analysis (debug bundles)."""
        coeff_names = self._coeff_names()
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
                "outdoor_temp": self._inputs.outdoor_temp,
                "room_temp_rate": round(self._room_temp_rate, 6),
                "integral_convergence": round(self._metrics.integral_convergence, 4),
                "tau_estimate": round(self._tau_estimator.tau, 1) if self._tau_estimator.enabled else None,
            },
        }

    def get_full_diagnostics(self) -> dict[str, Any]:
        """Return complete PI state for HA diagnostics platform.

        This is the single entry point for diagnostics.py — it should not
        need to reach into PI internals beyond this method.
        """
        import time as time_mod

        coeff_names = self._coeff_names()
        heat_phys = self._rls_heat.get_coefficients()
        cool_phys = self._rls_cool.get_coefficients()

        result: dict[str, Any] = {
            "enabled": True,
            "paused": self._pi_paused,
            "desired_temp": self._desired_temp,
            "hp_setpoint": self._hp_setpoint,
            "integral": round(self._pi_integral, 3),
            "integral_convergence": round(self._metrics.integral_convergence, 2),
            "ff_offset": round(self._ff_offset, 2),
            "ff_confidence": round(self._ff_confidence, 4),
            "outdoor_temp": self._inputs.outdoor_temp,
            "sensor_unavailable": self._sensor_unavailable,
            "sensor_recovery_pending": self._sensor_recovery_pending,
            "room_temp_rate": round(self._room_temp_rate, 4),
            "tau_estimate": round(self._tau_estimator.tau, 1) if self._tau_estimator.enabled else None,
            "config": {
                "kp": self._pi_kp,
                "ki": self._pi_ki,
                "deadband": self._pi_deadband,
                "setpoint_weight": self._pi_setpoint_weight,
                "min_interval": self._pi_min_interval,
                "outdoor_temp_sensor": self._inputs.outdoor_temp_sensor,
                "model_inputs": self._model_inputs,
            },
            "rls_model": {
                "heat_coefficients": {
                    coeff_names[i]: round(heat_phys[i], 4)
                    for i in range(min(len(coeff_names), len(heat_phys)))
                },
                "cool_coefficients": {
                    coeff_names[i]: round(cool_phys[i], 4)
                    for i in range(min(len(coeff_names), len(cool_phys)))
                },
                "heat_uncertainty": {
                    coeff_names[i]: round(self._rls_heat.get_covariance_diagonal()[i], 4)
                    for i in range(min(len(coeff_names), len(self._rls_heat.beta)))
                },
                "heat_observation_count": self._rls_heat.observation_count,
                "cool_observation_count": self._rls_cool.observation_count,
                "learning_suppressed": self._manual_ff_suppress,
                "manual_suppress_reason": self._manual_ff_suppress_reason,
            },
            "performance": {
                "itae_accumulator": round(self._metrics.itae_accumulator, 2),
                "comfort_violation_hours": round(self._metrics.comfort_violation_hours, 2),
                "setpoint_changes": self._metrics.setpoint_changes,
                "controllable_itae": round(self._metrics.controllable_itae, 2),
                "uncontrollable_itae": round(self._metrics.uncontrollable_itae, 2),
                "controllable_cvh": round(self._metrics.controllable_cvh, 2),
                "uncontrollable_cvh": round(self._metrics.uncontrollable_cvh, 2),
                "ff_load_fraction": round(self._metrics.ff_load_fraction, 4),
                "batch_model_rms": (
                    round(self._metrics.batch_model_rms, 3)
                    if self._metrics.batch_model_rms is not None else None
                ),
            },
        }

        # Batch learning
        if self._last_batch_result is not None:
            br = self._last_batch_result
            batch_names = coeff_names[:len(br.beta_batch)]
            batch: dict[str, Any] = {
                "last_run_mono": self._last_batch_timestamp,
                "n_total": br.n_total,
                "n_eligible": br.n_eligible,
                "residual_rms": round(br.residual_rms, 4),
                "recommend_update": br.recommend_update,
                "max_coeff_change_pct": round(br.max_coeff_change_pct, 1),
                "coefficients": {
                    batch_names[i]: {
                        "current": round(br.beta_current[i], 4),
                        "batch": round(br.beta_batch[i], 4),
                    }
                    for i in range(len(batch_names))
                    if i < len(br.beta_current)
                },
                "held_features": [
                    batch_names[i] for i in br.held_features
                    if i < len(batch_names)
                ],
                "n_outliers_excluded": br.n_outliers_excluded,
            }
            drifting = self.get_drifting_coefficients()
            batch["drift_detection"] = {
                "drifting_coefficients": [
                    {"index": idx, "name": name, "consecutive_cycles": count}
                    for idx, name, count in drifting
                ],
                "correction_history": {
                    (coeff_names[i] if i < len(coeff_names) else f"β{i}"): signs
                    for i, signs in enumerate(self._drift_correction_signs)
                },
            }
            result["batch_learning"] = batch
        else:
            result["batch_learning"] = None

        # Observation buffer stats
        obs = self._observation_buffer.get_all()
        n_eligible = sum(
            1 for o in obs if not o.clamped and abs(o.room_rate) < 0.02
        )
        buf_stats: dict[str, Any] = {"total": len(obs), "eligible": n_eligible}
        if hasattr(self._observation_buffer, "get_leverage_scores"):
            scores = self._observation_buffer.get_leverage_scores()
            if scores:
                buf_stats["leverage_min"] = round(min(scores), 6)
                buf_stats["leverage_median"] = round(sorted(scores)[len(scores) // 2], 6)
                buf_stats["leverage_max"] = round(max(scores), 6)
            if obs:
                now = time_mod.monotonic()
                oldest = min(o.timestamp for o in obs)
                buf_stats["oldest_age_hours"] = round((now - oldest) / 3600, 1)
            buf_stats["max_size"] = self._observation_buffer._max_size
            n_features = self._observation_buffer.n_features
            feature_active: dict[str, int] = {}
            for j in range(2, n_features):
                name = coeff_names[j] if j < len(coeff_names) else f"feature_{j}"
                feature_active[name] = sum(
                    1 for o in obs if j < len(o.features) and abs(o.features[j]) > 1e-6
                )
            if feature_active:
                buf_stats["feature_active_counts"] = feature_active
        result["observation_buffer"] = buf_stats

        return result

    def get_learning_status(self) -> dict[str, Any]:
        """Return FF learning suppression state for binary_sensor platform."""
        return {
            "suppressed": self._disturbance_suppress_active,
            "manual_suppress": self._manual_ff_suppress,
            "manual_suppress_reason": self._manual_ff_suppress_reason,
            "active_suppressors": list(self._disturbance_active_suppressors),
        }

    def has_rls_observations(self) -> bool:
        """Whether any RLS observations have been recorded (for button availability)."""
        return self._rls_heat.observation_count > 0

    def get_learned_seed_config(self) -> dict[str, Any]:
        """Return current learned coefficients formatted for config entry options.

        Used by the 'save learned seeds' button to write back to config.
        Returns a dict with keys matching CONF_PI_FF_HEAT_SLOPE, CONF_PI_FF_COOL_SLOPE,
        and updated model_inputs list with seed_heat/seed_cool per input.
        """
        heat_beta = self._rls_heat.beta
        cool_beta = self._rls_cool.beta
        result: dict[str, Any] = {}

        if len(heat_beta) > 1:
            result["heat_slope"] = round(heat_beta[1], 4)
        if len(cool_beta) > 1:
            result["cool_slope"] = round(abs(cool_beta[1]), 4)

        input_seeds: list[dict[str, float]] = []
        for i in range(len(self._model_inputs)):
            beta_idx = i + 2  # 0=intercept, 1=outdoor_delta, 2+=model inputs
            seeds: dict[str, float] = {}
            if beta_idx < len(heat_beta):
                seeds["seed_heat"] = round(heat_beta[beta_idx], 4)
            if beta_idx < len(cool_beta):
                seeds["seed_cool"] = round(cool_beta[beta_idx], 4)
            input_seeds.append(seeds)
        result["input_seeds"] = input_seeds

        return result

    def apply_saved_seeds(self) -> None:
        """Update internal seeds to match current coefficients.

        Called after saving seeds to config so seed-change detection
        doesn't flag the save as a user edit.
        """
        heat_beta = self._rls_heat.beta
        cool_beta = self._rls_cool.beta
        for i in range(min(len(heat_beta), len(self._heat_seeds), self._rls_heat.n)):
            self._heat_seeds[i] = round(heat_beta[i], 4)
            self._rls_heat.beta_seed[i] = round(heat_beta[i], 4)
        for i in range(min(len(cool_beta), len(self._cool_seeds), self._rls_cool.n)):
            self._cool_seeds[i] = round(cool_beta[i], 4)
            self._rls_cool.beta_seed[i] = round(cool_beta[i], 4)

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

        e = self._entity
        checks: list[tuple[str, str, str] | None] = []

        # Grace period: suppress comfort check for one tick after setpoint change
        if self._desired_temp != self._health_prev_desired:
            self._health_comfort_skip = 1
            self._health_prev_desired = self._desired_temp

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
            checks.append(check_comfort(
                abs(cur_c - desired_c),
                self.HEALTH_COMFORT_WARN, self.HEALTH_COMFORT_CRIT,
            ))

        checks.append(check_integral(
            abs(self._pi_ki * self._pi_integral), self.HEALTH_INTEGRAL_WARN,
        ))
        checks.append(check_ff_confidence(self._ff_confidence))

        # Active RLS model for coefficient checks
        is_heating = e._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
        rls = self._rls_heat if is_heating else self._rls_cool
        expected_slope = self._ff_heat_slope if is_heating else -self._ff_cool_slope
        has_obs = rls.observation_count > 0
        coeffs = rls.get_coefficients() if has_obs else {}
        intercept = coeffs.get(0, 0.0)
        outdoor_slope = coeffs.get(1, expected_slope)

        checks.append(check_intercept_drift(
            intercept, self.HEALTH_INTERCEPT_WARN, has_obs,
        ))
        checks.append(check_slope_drift(
            outdoor_slope, expected_slope,
            self.HEALTH_SLOPE_DRIFT_PCT, self.HEALTH_SLOPE_DRIFT_FLOOR, has_obs,
        ))

        checks.extend(check_model_drift(self.get_drifting_coefficients()))

        feature_names = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            feature_names.append(m.get("name", "input"))
        checks.append(check_feature_diversity(
            self._observation_buffer.get_all(),
            getattr(self._observation_buffer, "n_features", 0),
            feature_names,
            self.HEALTH_FEATURE_DIVERSITY_MIN,
            self.HEALTH_FEATURE_DIVERSITY_MIN_OBS,
        ))

        # Assemble results — highest severity wins
        alerts: list[str] = []
        reasons: list[str] = []
        severity = "OK"
        for result in checks:
            if result is None:
                continue
            msg, reason, sev = result
            alerts.append(msg)
            reasons.append(reason)
            if sev == "Critical":
                severity = "Critical"
            elif sev == "Warning" and severity != "Critical":
                severity = "Warning"

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
            "integral_convergence": round(self._metrics.integral_convergence, 2),
            "tau_estimate": round(self._tau_estimator.tau, 1) if self._tau_estimator.enabled else None,
            "smith_correction": (
                round(self._smith.correction, 3) if self._smith is not None else None
            ),
        }

    def filter_hvac_modes(self, modes: list[HVACMode]) -> list[HVACMode]:
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

    def _resolve_active_supplemental_sources(self) -> list[str]:
        """Resolve which supplemental sources are currently active from HA state."""
        active: list[str] = []
        for source in self._supplemental.source_configs:
            entity_id = source.get("entity_id", "")
            if not entity_id:
                continue
            state = self._hass.states.get(entity_id)
            if state is None or state.state in ("unavailable", "unknown"):
                continue
            if state.state in ("heat", "cool"):
                active.append(source.get("name", entity_id))
        return active

    def _evaluate_supplemental_override(self, error_c: float, now_mono: float) -> bool:
        """Evaluate supplemental override and apply side-effects. Returns hp_should_send_ir."""
        active = self._resolve_active_supplemental_sources()
        result = self._supplemental.evaluate(
            error_c, now_mono, active,
            pi_integral=self._pi_integral, hp_setpoint=self._hp_setpoint,
        )
        if result.should_reset_hold_timer:
            self._last_setpoint_change_time = 0.0
        return result.hp_should_send_ir

    def _deadband_integration_rate(self, abs_error: float) -> float:
        """Integration rate inside the deadband.

        Full-rate: the integrator accumulates the actual error regardless of
        proximity to setpoint.  Anti-cycling is handled by hysteresis, dwell
        timer, and leaky integrator — not by throttling the integration signal.

        Literature basis: standard PI practice for quantized actuators
        (McMillan, Åström & Hägglund).  Confirmed by A/B simulation across
        steady-state, sensor noise, and solar day/night cycling scenarios.
        Full-rate produces lower ITAE, fewer reversals, and fewer setpoint
        changes than the previous variable-rate (error²/deadband) policy
        during regime transitions through the deadband.
        """
        return 1.0

    # ── RLS learning ─────────────────────────────────────────────────

    def _rls_shared_gate_open(self, learning_suppressed: bool) -> bool:
        """Check shared RLS learning preconditions (outdoor temp, suppression, tracking)."""
        return (
            self._inputs.outdoor_temp is not None
            and not learning_suppressed
            and not self._any_model_input_unavailable()
            and not self._supplemental.tracking_mode
            and not self._supplemental.assist_active
        )

    def _rls_learn_observation(
        self,
        rls: RLSModel,
        x: list[float],
        observed_offset: float,
        label: str,
    ) -> float:
        """Update RLS model with observation and log. Returns residual."""
        beta_before = list(rls.beta)
        residual = rls.update(x, observed_offset)
        _LOGGER.debug(
            "%s: observed=%.2f predicted=%.2f residual=%.2f obs_count=%d dT_dt=%.4f",
            label, observed_offset, observed_offset - residual, residual,
            rls.observation_count, self._room_temp_rate,
        )
        for idx in range(len(rls.beta)):
            if beta_before[idx] != 0 and abs(rls.beta[idx] - beta_before[idx]) / abs(beta_before[idx]) > 0.1:
                _LOGGER.info(
                    "RLS coefficient[%d] changed %.3f -> %.3f (%.0f%%)",
                    idx, beta_before[idx], rls.beta[idx],
                    100 * (rls.beta[idx] - beta_before[idx]) / beta_before[idx],
                )
        return residual

    def _log_learning_blocked(
        self,
        learning_suppressed: bool,
        integral_change: float,
        room_rate: float,
        is_heating: bool,
        is_cooling: bool,
        error: float,
    ) -> None:
        """Log reasons why RLS learning gate is blocked (deadband path)."""
        reasons: list[str] = []
        if self._inputs.outdoor_temp is None:
            reasons.append("no outdoor temp")
        if learning_suppressed:
            reasons.append("manually suppressed")
        if integral_change * self._pi_ki >= 0.045:
            reasons.append(f"integral not settled (d_output={integral_change * self._pi_ki:.3f})")
        if abs(room_rate) >= 0.02:
            reasons.append(f"room not settled (dT/dt={room_rate:.4f} °C/min)")
        if self._any_model_input_unavailable():
            reasons.append("model input unavailable")
        if (is_heating and error < 0) or (is_cooling and error > 0):
            reasons.append(
                f"actuator no authority ({'heating' if is_heating else 'cooling'}, error={error:.2f})"
            )
        if reasons:
            _LOGGER.debug("RLS learning blocked: %s", ", ".join(reasons))

    # ── IMC Gain Scheduling (delegated to TauEstimator) ──────────────

    def _apply_gain_update(self, gains: GainUpdate) -> None:
        """Apply a GainUpdate from TauEstimator to PI state and Smith predictor."""
        self._pi_kp = gains.kp
        self._pi_ki = gains.ki
        if self._smith is not None:
            self._smith.update_params(tau=gains.tau, lag=gains.lag)

    def _recompute_imc_gains(self) -> None:
        """Recompute IMC gains from current τ estimate and apply them."""
        gains = self._tau_estimator.compute_gains()
        self._apply_gain_update(gains)

    # ── PI Internals ──────────────────────────────────────────────────

    def _resolve_model_input_states(self) -> dict[str, tuple[str, bool]]:
        """Resolve all model input entity states from HA for ModelInputManager."""
        states: dict[str, tuple[str, bool]] = {}
        for m_input in self._model_inputs:
            entity_id = m_input.get("entity_id", "")
            if not entity_id:
                continue
            state = self._hass.states.get(entity_id)
            if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                states[entity_id] = ("", False)
            else:
                states[entity_id] = (state.state, True)
        return states

    def _read_model_input_values(self) -> None:
        """Resolve HA entity states and update model input manager."""
        self._inputs.read_values(self._resolve_model_input_states())

    def _any_model_input_unavailable(self) -> bool:
        """Check if any model input entity is currently unavailable in HA."""
        return self._inputs.any_unavailable(self._resolve_model_input_states())

    @callback
    def _async_outdoor_temp_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle outdoor temperature sensor state changes."""
        new_state = event.data.get("new_state")
        if new_state is not None:
            unit = new_state.attributes.get("unit_of_measurement", UnitOfTemperature.CELSIUS)
            self._inputs.update_outdoor_temp(new_state.state, unit)

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
        if self._inputs.outdoor_temp is not None:
            if is_heating:
                outdoor_delta = max(0, self._ff_heat_reference - self._inputs.outdoor_temp)
            else:
                outdoor_delta = max(0, self._inputs.outdoor_temp - self._ff_cool_reference)
            self._read_model_input_values()
            x = self._inputs.build_feature_vector(outdoor_delta)
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
            self._tau_estimator.cancel_observation()
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
        tau_gain_update = self._tau_estimator.check_observation(now_mono, raw_c)
        if tau_gain_update is not None:
            self._apply_gain_update(tau_gain_update)

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
        self._inputs.update_lag_filters(dt_seconds)

        # Compute outdoor delta (always first model input)
        if self._inputs.outdoor_temp is not None:
            if is_heating:
                outdoor_delta = max(0, self._ff_heat_reference - self._inputs.outdoor_temp)
            else:
                outdoor_delta = max(0, self._inputs.outdoor_temp - self._ff_cool_reference)
        else:
            outdoor_delta = 0.0

        # Build feature vector and predict FF offset via RLS model
        x = self._inputs.build_feature_vector(outdoor_delta)
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
            if m_input.get("suppress_learning") and self._inputs.values[i] > 0.5:
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
            # Full-rate integration in deadband: accumulate the actual error.
            # P-term is zero here; the integrator is the only mechanism
            # correcting steady-state offset from HP quantization.
            rate = self._deadband_integration_rate(abs_error)
            if not skip_integration:
                self._pi_integral += avg_error * dt_factor * rate

            # IDB learning gate: require integral settled and room temp stable.
            # Output-normalized: compare Ki × Δintegral against a fixed output
            # threshold (0.045°C ≈ 0.3 × 0.15 at the default Ki).  This makes
            # the gate Ki-invariant — higher Ki needs proportionally smaller
            # integral swings to produce the same output change.
            integral_change = abs(self._pi_integral - self._prev_integral_for_rls)
            output_change = integral_change * self._pi_ki
            branch_ready = (
                self._ff_settled_ticks >= 4
                and output_change < 0.045
                and abs(self._room_temp_rate) < 0.02
            )
            if branch_ready and self._rls_shared_gate_open(learning_suppressed):
                # Observe hp_setpoint - desired_c: what offset maintained target
                self._rls_learn_observation(
                    rls, x, float(self._hp_setpoint) - desired_c, "RLS update",
                )
            elif self._ff_settled_ticks >= 4 and self._ff_settled_ticks % 4 == 0:
                self._log_learning_blocked(
                    learning_suppressed, integral_change, self._room_temp_rate,
                    is_heating, is_cooling, error,
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

            # OODB learning: at thermal equilibrium outside deadband, feed a
            # corrected observation (hp_setpoint - current_c) to the RLS.
            # Breaks the cycle where a miscalibrated model keeps the room away
            # from target, preventing the normal gate from opening.
            # Bias grows with distance from target, so require proportionally
            # more settling ticks at larger errors.
            room_stable = abs(self._room_temp_rate) < 0.015  # stricter than IDB gate
            integral_change = abs(self._pi_integral - self._prev_integral_for_oodb)
            # Output-normalized: 0.075°C ≈ 0.5 × 0.15 at default Ki
            integral_stable = integral_change * self._pi_ki < 0.075
            self._prev_integral_for_oodb = self._pi_integral
            setpoint_clamped = (
                self._hp_setpoint <= self._min_temp_c
                or self._hp_setpoint >= self._max_temp_c
            )
            min_oodb_ticks = 8 + int(abs_error * 4)  # +4 ticks per °C of error

            if room_stable and integral_stable and not setpoint_clamped:
                self._stable_oodb_ticks += 1
            else:
                if self._stable_oodb_ticks > 0:
                    reasons = []
                    if not room_stable:
                        reasons.append(f"room not settled (dT/dt={self._room_temp_rate:.4f})")
                    if not integral_stable:
                        reasons.append(f"integral not settled (d_output={integral_change * self._pi_ki:.3f})")
                    if setpoint_clamped:
                        reasons.append("setpoint clamped")
                    _LOGGER.debug(
                        "OODB gate reset at %d/%d ticks: %s",
                        self._stable_oodb_ticks, min_oodb_ticks,
                        ", ".join(reasons) if reasons else "conditions changed",
                    )
                self._stable_oodb_ticks = 0

            branch_ready = self._stable_oodb_ticks >= min_oodb_ticks
            if branch_ready and self._rls_shared_gate_open(learning_suppressed):
                # Observe hp_setpoint - current_c: what offset maintains equilibrium
                self._rls_learn_observation(
                    rls, x, float(self._hp_setpoint) - current_c, "RLS oodb",
                )
                self._stable_oodb_ticks = 0  # one observation per settled window

            # P-term with setpoint weighting (2-DOF, Åström & Hägglund).
            # Smith correction only applied when it prevents overshoot
            # (same sign as error) — don't fight recovery when signs differ.
            effective_smith = smith_correction if smith_correction * error > 0 else 0.0
            p_term = self._pi_kp * self._pi_setpoint_weight * (error - effective_smith)
            if not skip_integration:
                self._pi_integral += avg_error * dt_factor

        # Leaky integrator: weak decay bounds integral growth universally.
        # α=0.9999 per nominal tick ≈ 10000-tick time constant (~104 days at
        # 15-min ticks). Scaled by dt_factor for variable sample intervals.
        # Weaker than α=0.999 to avoid draining integral correction needed by
        # slow-τ houses (well-insulated). Sim-validated across 3 profiles.
        self._pi_integral *= 0.9999 ** dt_factor

        self._pi_last_error = error

        # Performance metrics accumulation
        self._metrics.accumulate_convergence(self._pi_integral)
        if not self._supplemental.tracking_mode:
            self._metrics.accumulate_tick(
                abs_error=abs_error,
                dt_seconds=dt_seconds,
                pi_deadband=self._pi_deadband,
                is_heating=is_heating,
                is_cooling=is_cooling,
                error=error,
                hp_setpoint=self._hp_setpoint,
                min_temp_c=self._min_temp_c,
                max_temp_c=self._max_temp_c,
            )
        self._metrics.accumulate_ff_load(
            ki_integral=self._pi_ki * self._pi_integral,
            ff_offset=self._ff_offset,
        )

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
                        self._log_prefix, old_setpoint, new_setpoint, ", ".join(self._supplemental.tracking_sources),
                    )
                else:
                    _LOGGER.info(
                        "%sPI: error=%.1f smith=%.2f P=%.1f I=%.1f D=%.1f FF=%.1f raw=%.1f setpoint %s -> %s",
                        self._log_prefix, error, smith_correction, p_term, i_term, d_term, self._ff_offset,
                        clamped_setpoint, old_setpoint, new_setpoint,
                    )
                    self._last_setpoint_change_time = now_mono
                    self._metrics.record_setpoint_change()
                    # Start τ observation on significant setpoint changes
                    self._tau_estimator.start_observation(now_mono, current_c, desired_c, float(change))
                    return True
        else:
            _LOGGER.debug(
                "%sPI: error=%.1f raw=%.1f setpoint=%s (held)",
                self._log_prefix, error, clamped_setpoint, self._hp_setpoint,
            )

        return False
