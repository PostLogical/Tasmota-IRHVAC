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
from collections import deque
from datetime import datetime, timedelta, timezone
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

from .batch_learning import BatchResult, CollinearGroup, DiversityAwareBuffer, HourlyResidualPattern, MAX_STEP_ABS, Observation, UNLOCK_FIRST_STEP, analyze_residuals_by_hour, build_feature_vector_from_raw, compute_belsley_diagnostics, fuse_batch_greybox, weighted_least_squares, compare_and_report, compute_blended_update
from .greybox_buffer import GreyboxBuffer
from .greybox_observer import (
    GreyboxBridgeResult,
    GreyboxResult,
    SCIPY_AVAILABLE,
    find_solar_entity,
    fit_greybox,
    greybox_to_beta,
    log_greybox_result,
)
from .health_checks import AnomalyEvent, compute_mad_sigma, CUSUM_K, CUSUM_H, MIN_RESIDUALS_FOR_DETECTION, MIN_SIGMA_FLOOR, MIN_EVENT_DURATION_SEC, CUSUM_COOLDOWN_SEC, obs_raw_reading

from ..const import (
    ATTR_DESIRED_TEMP,
    ATTR_FF_OFFSET,
    ATTR_HP_SETPOINT,
    ATTR_PI_INTEGRAL,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PI_DEADBAND,
    CONF_PI_ENABLED,
    CONF_PI_INTERCEPT_SEED_COOL,
    CONF_PI_INTERCEPT_SEED_HEAT,
    CONF_PI_OUTDOOR_SEED_COOL,
    CONF_PI_OUTDOOR_SEED_HEAT,
    CONF_PI_OUTDOOR_SEED_CLAMP_MAX,
    CONF_PI_OUTDOOR_SEED_CLAMP_MIN,
    CONF_PI_IMC_LAMBDA,
    CONF_PI_KD,
    CONF_PI_SENSOR_FILTER_TAU,
    CONF_PI_KD_FILTER_N,
    CONF_PI_KI,
    CONF_PI_KP,
    CONF_PI_TICK_FALLBACK,
    CONF_PI_MODEL_INPUTS,
    CONF_PI_RESPONSE_LAG,
    CONF_PI_SETPOINT_HOLD,
    CONF_PI_SETPOINT_WEIGHT,
    CONF_PI_SMITH_ENABLED,
    CONF_PI_AUTO_PERTURB_ENABLED,
    CONF_PI_AUTO_PERTURB_WINDOW_START,
    CONF_PI_AUTO_PERTURB_WINDOW_END,
    CONF_PI_BATCH_WLS_ENABLED,
    CONF_PI_FF_ENABLED,
    CONF_PI_PLANT_ID_ENABLED,
    CONF_PI_RLS_ONLINE_ENABLED,
    CONF_PI_TAU_ESTIMATE,
    DEFAULT_PI_BATCH_WLS_ENABLED,
    DEFAULT_PI_DEADBAND,
    DEFAULT_PI_ENABLED,
    DEFAULT_PI_FF_ENABLED,
    DEFAULT_PI_INTERCEPT_SEED_COOL,
    DEFAULT_PI_INTERCEPT_SEED_HEAT,
    DEFAULT_PI_OUTDOOR_SEED_COOL,
    DEFAULT_PI_OUTDOOR_SEED_HEAT,
    DEFAULT_PI_OUTDOOR_SEED_CLAMP_MAX,
    DEFAULT_PI_OUTDOOR_SEED_CLAMP_MIN,
    DEFAULT_PI_IMC_LAMBDA,
    DEFAULT_PI_KD,
    DEFAULT_PI_SENSOR_FILTER_TAU,
    DEFAULT_PI_KD_FILTER_N,
    DEFAULT_PI_KI,
    DEFAULT_PI_KP,
    DEFAULT_PI_PLANT_ID_ENABLED,
    DEFAULT_PI_RLS_ONLINE_ENABLED,
    DEFAULT_PI_TICK_FALLBACK,
    DEFAULT_PI_RESPONSE_LAG,
    DEFAULT_PI_SETPOINT_HOLD,
    DEFAULT_PI_SETPOINT_WEIGHT,
    DEFAULT_PI_SMITH_ENABLED,
    DEFAULT_PI_TAU_ESTIMATE,
    SIGNAL_FF_SUPPRESS_UPDATE,
    SIGNAL_PI_BATCH_COMPLETE,
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
from .auto_perturbation import AutoPerturbation
from .smith_predictor import SmithPredictor
from .supplemental_controller import SupplementalController
from .plant_identifier import PlantIdentifier
from .plant_model import GainUpdate

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
        self._greybox_blending_enabled: bool = config.get("pi_greybox_blending", False)
        self._pi_ff_enabled: bool = config.get(CONF_PI_FF_ENABLED, DEFAULT_PI_FF_ENABLED)
        self._pi_rls_online_enabled: bool = config.get(
            CONF_PI_RLS_ONLINE_ENABLED, DEFAULT_PI_RLS_ONLINE_ENABLED)
        self._pi_batch_wls_enabled: bool = config.get(
            CONF_PI_BATCH_WLS_ENABLED, DEFAULT_PI_BATCH_WLS_ENABLED)
        self._pi_plant_id_enabled: bool = config.get(
            CONF_PI_PLANT_ID_ENABLED, DEFAULT_PI_PLANT_ID_ENABLED)
        self._SETPOINT_HOLD_SECONDS: float = float(
            config.get(CONF_PI_SETPOINT_HOLD, DEFAULT_PI_SETPOINT_HOLD)
        )
        self._pi_tick_fallback: float = config.get(CONF_PI_TICK_FALLBACK, DEFAULT_PI_TICK_FALLBACK)

        # Plant identification + IMC gain scheduling
        self._plant_id = PlantIdentifier(
            tau_seed=config.get(CONF_PI_TAU_ESTIMATE, DEFAULT_PI_TAU_ESTIMATE),
            response_lag=config.get(CONF_PI_RESPONSE_LAG, DEFAULT_PI_RESPONSE_LAG),
            imc_lambda=config.get(CONF_PI_IMC_LAMBDA, DEFAULT_PI_IMC_LAMBDA),
        )

        # Smith predictor for dead-time compensation.
        # Requires IMC (tau > 0) AND pi_smith_enabled=true.
        self._smith: SmithPredictor | None = None
        if self._plant_id.enabled and self._smith_enabled:
            self._smith = SmithPredictor(
                tau=self._plant_id.plant.tau_fast.value,
                lag=self._plant_id.response_lag,
            )

        # Auto-perturbation for plant identification (Layer 2.5).
        self._auto_perturb = AutoPerturbation(
            enabled=config.get(CONF_PI_AUTO_PERTURB_ENABLED, False),
            window_start=config.get(CONF_PI_AUTO_PERTURB_WINDOW_START),
            window_end=config.get(CONF_PI_AUTO_PERTURB_WINDOW_END),
        )

        # Derive effective Kp/Ki: IMC formula or manual config
        self._pi_kp: float = 0.0
        self._pi_ki: float = 0.0
        if self._plant_id.enabled:
            gains = self._plant_id.compute_gains()
            self._pi_kp = gains.kp
            self._pi_ki = gains.ki
        else:
            self._pi_kp = self._pi_kp_config
            self._pi_ki = self._pi_ki_config

        # Control parameters are stored in °C always — read directly, no conversion
        self._pi_deadband: float = config.get(CONF_PI_DEADBAND, DEFAULT_PI_DEADBAND)
        self._pi_setpoint_weight: float = config.get(CONF_PI_SETPOINT_WEIGHT, DEFAULT_PI_SETPOINT_WEIGHT)

        # Feedforward config — seeds are positive = warms room, negated to internal β
        self._outdoor_seed_heat: float = config.get(
            CONF_PI_OUTDOOR_SEED_HEAT, DEFAULT_PI_OUTDOOR_SEED_HEAT)
        self._outdoor_seed_cool: float = config.get(
            CONF_PI_OUTDOOR_SEED_COOL, DEFAULT_PI_OUTDOOR_SEED_COOL)
        # Intercept seeds — direct baseline offset (no negation, not directional)
        self._intercept_seed_heat: float = config.get(
            CONF_PI_INTERCEPT_SEED_HEAT, DEFAULT_PI_INTERCEPT_SEED_HEAT)
        self._intercept_seed_cool: float = config.get(
            CONF_PI_INTERCEPT_SEED_COOL, DEFAULT_PI_INTERCEPT_SEED_COOL)

        # Sensor unavailability tracking for repairs.
        # Set when outdoor_temp transitions from valid to None (not on startup).
        self._outdoor_temp_unavailable_since: float | None = None
        self._init_time: float = time.monotonic()

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
        # Each: {"name": str, "entity_id": str, "seed_heat": float,
        #         "seed_cool": float, "clamp_min": float, "clamp_max": float,
        #         "lag_tau": float (seconds)}
        self._model_inputs: list[dict[str, Any]] = list(config.get(CONF_PI_MODEL_INPUTS, []))

        # Auto-generate model inputs from supplemental sources with auto_model_input=true.
        # These use the supplemental's climate entity as a binary signal (heat/cool=1, else=0).
        # The PI controller reads the entity state each tick to update the value.
        for source in supplemental_sources:
            if not source.get("input_enabled", True):
                continue
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
                "seed_heat": float(source.get("seed_heat", 3.0)),
                "seed_cool": float(source.get("seed_cool", 3.0)),
                "lag_tau": 0,
                "suppress_learning": True,  # Learning deferred per research
                "_auto_supplemental": True,  # Internal flag for filtering
            })
        # Outdoor delta is always the first model input (index 1, after intercept)
        # Other model inputs follow in order of _model_inputs list
        self._n_model_inputs = 1 + len(self._model_inputs)  # outdoor_delta + configured inputs

        # Build seed coefficients and clamps
        # Index 0: intercept (direct baseline offset, no negation)
        # Index 1: outdoor_delta (β = -seed, since warming → HP backs off)
        # Index 2+: model inputs in order
        # Convention: user-facing seeds are positive for "warms room".
        # Internal β = -seed (HP backs off when source warms room).
        # Intercept is not directional — stored as-is.
        self._heat_seeds = [self._intercept_seed_heat, -self._outdoor_seed_heat]
        self._cool_seeds = [self._intercept_seed_cool, -self._outdoor_seed_cool]
        # Clamp in internal β space: seed (0, 2) → β (-2, 0)
        thermal_clamp_min = config.get(CONF_PI_OUTDOOR_SEED_CLAMP_MIN, DEFAULT_PI_OUTDOOR_SEED_CLAMP_MIN)
        thermal_clamp_max = config.get(CONF_PI_OUTDOOR_SEED_CLAMP_MAX, DEFAULT_PI_OUTDOOR_SEED_CLAMP_MAX)
        beta_clamp = (-thermal_clamp_max, -thermal_clamp_min)
        self._rls_heat_clamps: list[tuple[float, float] | None] = [None, beta_clamp]
        self._rls_cool_clamps: list[tuple[float, float] | None] = [None, beta_clamp]
        for m_input in self._model_inputs:
            self._heat_seeds.append(-float(m_input.get("seed_heat", 0.0)))
            self._cool_seeds.append(-float(m_input.get("seed_cool", 0.0)))
            clamp_min = m_input.get("clamp_min")
            clamp_max = m_input.get("clamp_max")
            # Clamps are in seed space (positive = warms room).
            # Internal β = -seed, so negate and flip.
            # Either side can be set independently; missing side → ±inf.
            clamp: tuple[float, float] | None
            if clamp_min is not None or clamp_max is not None:
                seed_lo = float(clamp_min) if clamp_min is not None else -math.inf
                seed_hi = float(clamp_max) if clamp_max is not None else math.inf
                clamp = (-seed_hi, -seed_lo)
            else:
                clamp = None
            self._rls_heat_clamps.append(clamp)
            self._rls_cool_clamps.append(clamp)

        # Feature scales = expected σ of each feature (van der Sluis 1969,
        # Haykin Adaptive Filter Theory §13). Normalizes features to O(1)
        # for balanced P-matrix conditioning and learning rates.
        # intercept=1.0, outdoor_delta σ ≈ range/4 ≈ 50/4 ≈ 13, model inputs ~0.5
        self._feature_scales = [1.0, 13.0]
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

        # Cold start: freeze model input features (index 2+) until batch
        # WLS establishes per-feature confidence.  Intercept (0) and
        # outdoor_delta (1) are always identifiable from base regression.
        for i in range(2, self._rls_heat.n):
            self._rls_heat.frozen[i] = True
            self._rls_cool.frozen[i] = True

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

        # HP thermostat deadband learning: the HP's internal thermostat has
        # its own hysteresis, so the compressor may still cycle even when
        # hp_setpoint is slightly below room temp (heating) or above (cooling).
        # We learn the effective deadband width from rate-based observations
        # and use it as a fast-path margin for integration freeze decisions.
        # Separate estimates per mode — asymmetric compressor cycling logic.
        self._hp_deadband_estimate_heat: float = 0.5
        self._hp_deadband_estimate_cool: float = 0.5
        self._hp_no_output_ticks: int = 0

        # Drift detection: per-coefficient history of batch correction signs.
        # Each entry is +1 (batch pushed up), -1 (batch pushed down), or 0.
        # Tracked across batch cycles to detect persistent same-direction
        # corrections that indicate a physical change.
        self._drift_correction_signs: list[list[int]] = []
        # Number of consecutive same-direction corrections to trigger alert.
        self._drift_threshold: int = 5

        # Batch learning: per-mode diversity-aware buffers of every tick's
        # state for periodic offline WLS analysis.  Heat and cool models
        # learn independently — outdoor_delta sign differs — so observations
        # must not be mixed.  Records regardless of learning gate.
        # n_features = intercept + outdoor_delta + model_inputs
        self._feature_order: list[str] = self._inputs.build_feature_names()
        _n_buf_features = len(self._feature_order)
        self._observation_buffer_heat = DiversityAwareBuffer(
            n_features=_n_buf_features,
            feature_order=self._feature_order,
            model_inputs=self._model_inputs,
        )
        self._observation_buffer_cool = DiversityAwareBuffer(
            n_features=_n_buf_features,
            feature_order=self._feature_order,
            model_inputs=self._model_inputs,
        )
        # Grey-box buffer: mode-agnostic, admits HP-off, temp-quantile-stratified.
        self._greybox_buffer = GreyboxBuffer(
            solar_entity=find_solar_entity(self._model_inputs),
        )
        # Cached serializations — refreshed only at batch time (every 12h) to
        # avoid serializing thousands of observations on every state write (60s).
        self._obs_buffer_heat_cache: list[dict[str, Any]] = []
        self._obs_buffer_cool_cache: list[dict[str, Any]] = []
        self._greybox_buffer_cache: list[dict[str, Any]] = []
        self._batch_analysis_timer: CALLBACK_TYPE | None = None
        self._last_batch_result: BatchResult | None = None
        self._last_greybox_result: GreyboxResult | None = None
        self._last_greybox_bridge: GreyboxBridgeResult | None = None
        self._last_greybox_timestamp_iso: str | None = None
        self._greybox_has_been_good: bool = False
        self._last_batch_timestamp: float | None = None
        self._last_batch_wallclock: str = ""  # ISO-8601 wall-clock time
        self._last_residual_patterns: list[HourlyResidualPattern] = []
        self._has_had_stable_batch: bool = False
        # Batch-first gate: RLS online updates are frozen until the first
        # batch WLS cycle has run and recommended an update.  Until then,
        # observations are buffered for batch but rls.update() is not called.
        # PI + seed-based FF handles comfort in the interim.
        self._rls_heat_mature: bool = False
        self._rls_cool_mature: bool = False
        self._tuning_alert_counters: dict[str, int] = {}
        self._tuning_alert_snapshots: dict[str, float] = {}

        # Track last active mode for passive tick buffer selection
        self._last_active_heating: bool = True

        # ── Per-feature confidence gating ───────────────────────────
        # Three-state manual override per coefficient:
        #   None = auto-gating decides (default)
        #   True = force unfrozen (user manually unfroze)
        #   False = force frozen (user manually froze)
        # Auto-gating skips features where override is not None.
        self._manual_override_heat: list[bool | None] = [None] * (self._n_model_inputs + 1)
        self._manual_override_cool: list[bool | None] = [None] * (self._n_model_inputs + 1)
        # κ-gated lambda: original lambda_base cached for restoration
        self._original_lambda_base_heat: float = self._rls_heat.lambda_base
        self._original_lambda_base_cool: float = self._rls_cool.lambda_base
        self._cached_kappa: float | None = None
        self._cached_collinear_groups: list[CollinearGroup] = []
        # Adaptive batch step cap: track when each feature was unlocked
        # so we can allow an enlarged first step.  None = never unlocked
        # or unlocked long enough ago that normal cap applies.
        self._batch_cycle_count: int = 0
        self._unlock_batch_cycle: list[int | None] = [None] * self._rls_heat.n

        # ── CUSUM anomaly detection state ───────────────────────────
        self._residual_history: deque[float] = deque(maxlen=60)
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._anomaly_events: list[AnomalyEvent] = []
        self._exclusion_count: int = 0
        self._cusum_cooldown_until: datetime | None = None

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
    def desired_temp_celsius(self) -> float | None:
        if self._desired_temp is None:
            return None
        return round(
            TemperatureConverter.convert(
                self._desired_temp,
                self._entity.temperature_unit,
                UnitOfTemperature.CELSIUS,
            ),
            1,
        )

    @property
    def tau_estimate(self) -> float | None:
        if not self._plant_id.enabled:
            return None
        return round(self._plant_id.tau, 1)

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

        # Register model input entities (including gate entities)
        model_entity_ids = [
            m["entity_id"] for m in self._model_inputs if m.get("entity_id")
        ]
        gate_entity_ids = [
            m["gate_entity"] for m in self._model_inputs
            if m.get("gate_entity") and m["gate_entity"] not in model_entity_ids
        ]
        track_ids = model_entity_ids + gate_entity_ids
        if track_ids:
            async_track_state_change_event(
                self._hass,
                track_ids,
                self._async_model_input_changed,
            )
        # Read initial model input values
        self._read_model_input_values()

        # Compute initial feedforward offset (requires outdoor temp and desired temp)
        # outdoor_delta references desired temp, not room temp, to keep FF
        # exogenous — prevents positive feedback during transients (Åström §5).
        desired_c = self.desired_temp_celsius
        if (self._pi_ff_enabled
                and self._inputs.outdoor_temp is not None and desired_c is not None):
            is_heating = e._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
            outdoor_delta = self._inputs.outdoor_temp - desired_c
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
        ±1.0°C step cap per coefficient per 12h cycle (enlarged to ±3.0°C
        for recently-unlocked features when batch quality gates pass).
        """
        if not self._pi_batch_wls_enabled or not self._pi_ff_enabled:
            _LOGGER.debug(
                "%sBatch WLS: skipped (ff_enabled=%s, batch_wls_enabled=%s)",
                self._log_prefix, self._pi_ff_enabled, self._pi_batch_wls_enabled,
            )
            return

        self._batch_cycle_count += 1
        e = self._entity
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        rls = self._rls_heat if is_heating else self._rls_cool
        buffer = self._observation_buffer_heat if is_heating else self._observation_buffer_cool

        # Periodic recomputation of info matrix to prevent numerical drift.
        if hasattr(buffer, 'recompute_info_matrix'):
            if buffer.needs_recompute:
                buffer.recompute_info_matrix()

        observations = buffer.get_all()
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

        # Partial model: frozen features held — matches online RLS.
        # Coefficients from this run get applied via compute_blended_update.
        frozen_set = self._get_frozen_feature_set(rls)
        result = weighted_least_squares(
            observations, n_features=rls.n, current_beta=current_phys,
            room_rate_threshold=0.02, min_observations=20,
            feature_order=self._feature_order,
            model_inputs=self._inputs.model_inputs,
            frozen_features=frozen_set,
        )
        if result is None:
            _LOGGER.debug(
                "%sBatch WLS: insufficient eligible observations after filtering",
                self._log_prefix,
            )
            return

        # Full model: all features estimated — for unlock evaluation only.
        # Coefficients are discarded; only std_err, held_features, and the
        # buffer VIF are used to decide if frozen features are identifiable.
        full_result: BatchResult | None = None
        if frozen_set:
            full_result = weighted_least_squares(
                observations, n_features=rls.n, current_beta=current_phys,
                room_rate_threshold=0.02, min_observations=20,
                feature_order=self._feature_order,
                model_inputs=self._inputs.model_inputs,
                # No frozen_features → estimates everything
            )

        coeff_names = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            coeff_names.append(m.get("name", "input"))

        compare_and_report(
            result, current_phys, coeff_names,
            change_threshold_pct=20.0, min_observations=20,
            log_prefix=self._log_prefix,
        )

        # Record plant ID snapshot (needed for grey-box seeding).
        plant = self._plant_id.plant
        result.plant_snapshot = {
            "k": plant.k.value,
            "k_confidence": plant.k.confidence,
            "theta": plant.theta.value,
            "tau_fast": plant.tau_fast.value,
            "tau_fast_confidence": plant.tau_fast.confidence,
            "tau_slow": plant.tau_slow.value,
            "tau_slow_confidence": plant.tau_slow.confidence,
        }

        # ── Grey-box 1R1C energy balance + steady-state bridge ──
        # Run BEFORE fusion+blend so this cycle's grey-box can inform
        # the coefficient update (no one-cycle lag).
        # Grey-box uses its own buffer (includes HP-off observations).
        greybox_observations = self._greybox_buffer.get_all()
        gb_diag = self._greybox_buffer.get_diagnostics()
        _LOGGER.info(
            "%sGrey-box buffer: %d/%d obs, hp_off=%.1f%%, min_leverage=%s",
            self._log_prefix,
            gb_diag["total"], gb_diag["max_size"],
            gb_diag.get("hp_off_pct") or 0.0,
            gb_diag.get("min_leverage"),
        )
        greybox = fit_greybox(
            greybox_observations,
            model_inputs=self._model_inputs,
            plant_tau_slow=plant.tau_slow.value if plant.tau_slow.confidence > 0 else None,
            plant_tau_slow_confidence=plant.tau_slow.confidence,
        )
        if greybox is not None:
            log_greybox_result(greybox, log_prefix=self._log_prefix)
            self._last_greybox_result = greybox
            self._last_greybox_timestamp_iso = datetime.now(tz=timezone.utc).isoformat()

            # Bridge: convert rate coefficients to WLS-compatible β
            bridge = greybox_to_beta(
                greybox,
                model_inputs=self._model_inputs,
                log_prefix=self._log_prefix,
            )
            self._last_greybox_bridge = bridge
            if bridge.gates_passed:
                self._greybox_has_been_good = True

            # Cross-validation: compare grey-box β against WLS β
            if result.beta_batch:
                self._log_greybox_wls_comparison(bridge, result)

            # Feed τ_eff to plant ID as a grey-box τ_slow estimate
            if bridge.gates_passed and self._plant_id.enabled:
                gb_se = greybox.param_std_err
                ua_c_cv = gb_se.get("ua_c", float("inf")) / max(abs(greybox.ua_c), 1e-12)
                gain_update = self._plant_id.update_from_greybox(
                    tau_eff=bridge.tau_eff,
                    ua_c_cv=ua_c_cv,
                )
                if gain_update is not None:
                    self._pi_kp = gain_update.kp
                    self._pi_ki = gain_update.ki
                    self._imc_lambda = gain_update.imc_lambda

        # ── Grey-box fusion ──
        # If bridge available, gates pass, and blending enabled, fuse
        # grey-box β into the batch estimate before blending with the
        # online model.  When disabled (default), the bridge still runs
        # for logging/diagnostics but doesn't influence coefficients.
        if (
            self._greybox_blending_enabled
            and self._last_greybox_bridge is not None
            and self._last_greybox_bridge.gates_passed
        ):
            _LOGGER.info(
                "%sGrey-box blending: fusing into batch estimate",
                self._log_prefix,
            )
            fuse_batch_greybox(
                result,
                self._last_greybox_bridge.beta,
                self._last_greybox_bridge.beta_std_err,
            )

        # Compute and cache κ before blending (needed for per-feature caps).
        n_eligible = sum(
            1 for o in buffer.get_all()
            if o.clamped_reason not in ("no_output", "clamped")
            and abs(o.room_rate) < 0.02
        )
        if n_eligible >= 2 * buffer.n_features:
            kappa = buffer.compute_condition_number()
            self._cached_kappa = kappa if not math.isinf(kappa) else None
        else:
            kappa = float("inf")
            self._cached_kappa = None

        # Build per-feature step caps: enlarged for recently-unlocked
        # features when batch quality gates pass.
        per_feature_caps = self._build_per_feature_step_caps(
            result, buffer, n_eligible, kappa,
        )
        compute_blended_update(
            result, prior_std=1.0, max_step=1.0,
            max_step_per_feature=per_feature_caps,
        )

        # κ gate: reject batch recommendation when condition number indicates
        # severe multicollinearity.  The coefficients may look different from
        # current values but the data geometry can't reliably separate them.
        kappa_threshold = getattr(self, '_batch_kappa_threshold', 100)
        if result.recommend_update:
            if not math.isinf(kappa) and kappa > kappa_threshold:
                _LOGGER.warning(
                    "%sBatch WLS: κ=%.0f (severe) — rejecting recommendation "
                    "until data geometry improves",
                    self._log_prefix, kappa,
                )
                result.recommend_update = False

        if result.recommend_update and result.beta_blended:
            for i, val in enumerate(result.beta_blended):
                if i < rls.n:
                    rls.beta[i] = val * rls.feature_scales[i]

            # P-aware update: reduce covariance for updated coefficients
            # so RLS treats the batch correction as real posterior
            # information and doesn't immediately drift back.
            # P[i,i] *= (1 - K_i): higher batch confidence → lower P.
            # Floor = max(batch_var_normalized, delta) to preserve
            # adaptability to real physical changes.
            if result.blend_gains:
                n = min(len(result.blend_gains), rls.n)
                for i in range(n):
                    k_i = result.blend_gains[i]
                    if k_i <= 0:
                        continue
                    # Compute floor from batch std_err in normalized space
                    se = (
                        result.beta_std_err[i]
                        if i < len(result.beta_std_err)
                        else float("inf")
                    )
                    if math.isinf(se):
                        continue
                    se_norm = se * rls.feature_scales[i]
                    p_floor = max(se_norm * se_norm, rls.delta)
                    # Reduce diagonal by (1 - K_i)
                    old_pii = rls.P[i * rls.n + i]
                    rls.P[i * rls.n + i] = max(p_floor, old_pii * (1 - k_i))
                    # Zero off-diagonal elements for updated coefficient
                    # (same projection as clamp behavior) so cross-
                    # correlations don't pull the corrected value back.
                    for j in range(rls.n):
                        if j != i:
                            rls.P[i * rls.n + j] = 0.0
                            rls.P[j * rls.n + i] = 0.0

            # Mark RLS as mature — batch has validated the data geometry
            # and provided a well-conditioned baseline.  Online RLS tracking
            # is now safe to run.
            if is_heating:
                if not self._rls_heat_mature:
                    _LOGGER.info(
                        "%sBatch-first gate: heat RLS now mature",
                        self._log_prefix,
                    )
                self._rls_heat_mature = True
            else:
                if not self._rls_cool_mature:
                    _LOGGER.info(
                        "%sBatch-first gate: cool RLS now mature",
                        self._log_prefix,
                    )
                self._rls_cool_mature = True

            _LOGGER.info(
                "%sBatch WLS: applied blended update to %s model",
                self._log_prefix, "heat" if is_heating else "cool",
            )

        # ── Per-feature confidence gating ──
        # Evaluate unlock conditions using the full-model result (all features
        # estimated).  The full result's std_err, held_features, and VIF tell
        # us whether each frozen feature is identifiable from current data.
        if full_result is not None:
            self._evaluate_feature_unlocks(full_result, rls, is_heating)

        # ── κ-gated learning rate ──
        # When condition number is elevated, slow online RLS by pushing
        # λ toward 1.0 (no forgetting).  Linear blend: κ≤30 → no change,
        # κ≥100 → λ=1.0.  Cached per batch cycle.
        self._apply_kappa_gated_lambda(rls, is_heating)

        self._last_batch_result = result
        self._last_batch_timestamp = time.monotonic()
        self._last_batch_wallclock = datetime.now().isoformat(timespec="seconds")
        self._metrics.batch_model_rms = result.residual_rms

        # Refresh buffer serialization caches (avoids re-serializing thousands
        # of observations on every 60s state write — only at batch time).
        self._obs_buffer_heat_cache = self._observation_buffer_heat.as_list()
        self._obs_buffer_cool_cache = self._observation_buffer_cool.as_list()
        self._greybox_buffer_cache = self._greybox_buffer.as_list()

        # ── Residual time-of-day analysis ──
        beta_for_residuals = result.beta_blended if result.beta_blended else result.beta_batch
        self._last_residual_patterns = analyze_residuals_by_hour(
            observations, beta_for_residuals, n_features=rls.n,
            feature_order=self._feature_order,
            model_inputs=self._inputs.model_inputs,
        )
        if self._last_residual_patterns:
            for p in self._last_residual_patterns:
                _LOGGER.info(
                    "%sResidual pattern: %02d:00–%02d:59 mean=%.2f°C (%d obs)",
                    self._log_prefix, p.start_hour, p.end_hour,
                    p.mean_residual, p.n_observations,
                )

        # ── Belsley VDP collinearity diagnostic ──
        # Compute per-variable variance decomposition to identify which
        # features share ill-conditioned components.  Cached for the
        # multicollinearity repair check.
        coeff_names_vdp = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            coeff_names_vdp.append(m.get("name", "input"))
        eligible = [
            o for o in observations
            if o.clamped_reason not in ("no_output", "clamped")
            and abs(o.room_rate) < 0.02
            and o.hp_setpoint is not None
        ]
        if len(eligible) >= 2 * rls.n:
            vdp_X: list[list[float]] = []
            for o in eligible:
                vec = build_feature_vector_from_raw(
                    o, self._inputs.model_inputs, self._feature_order,
                )
                if vec is not None:
                    vdp_X.append(vec)
            if len(vdp_X) >= 2 * rls.n:
                self._cached_collinear_groups = compute_belsley_diagnostics(
                    vdp_X, n_features=rls.n, n_obs=len(vdp_X),
                    feature_names=coeff_names_vdp,
                )
                for g in self._cached_collinear_groups:
                    _LOGGER.info(
                        "%sBelsley VDP: %s share ill-conditioned component (CI=%.0f)",
                        self._log_prefix,
                        " and ".join(g.features),
                        g.condition_index,
                    )
        else:
            self._cached_collinear_groups = []

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

        # Track maturity: at least one batch cycle without recommending update
        if not result.recommend_update:
            self._has_had_stable_batch = True

        # Signal batch completion for tuning health checks
        if hasattr(self._entity, "_config_entry_id"):
            async_dispatcher_send(
                self._hass,
                SIGNAL_PI_BATCH_COMPLETE.format(self._entity._config_entry_id),
            )

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
    def _active_buffer(self) -> DiversityAwareBuffer:
        """Return the observation buffer for the active HVAC mode.

        Defaults to heat buffer when mode is ambiguous (OFF, HEAT_COOL,
        None) — heat is the dominant mode and buffer sensors should
        report useful data in the default state.
        """
        is_cooling = self._entity._attr_hvac_mode in (HVACMode.COOL, HVACMode.DRY)
        return self._observation_buffer_cool if is_cooling else self._observation_buffer_heat

    @property
    def buffer_eligible(self) -> int:
        """Count of eligible (unclamped, low-rate) observations in the active buffer."""
        obs = self._active_buffer.get_all()
        return sum(1 for o in obs if o.clamped_reason not in ("no_output", "clamped") and abs(o.room_rate) < 0.02)

    @property
    def buffer_total(self) -> int:
        """Total observations in the active mode's buffer."""
        return len(self._active_buffer)

    @property
    def buffer_oldest_age_hours(self) -> float | None:
        """Age of the oldest observation in hours, or None if buffer is empty."""
        obs = self._active_buffer.get_all()
        if not obs:
            return None
        import time as time_mod

        now = time_mod.monotonic()
        oldest = min(o.timestamp for o in obs)
        return round((now - oldest) / 3600, 1)

    @property
    def buffer_leverage_max(self) -> float | None:
        """Maximum leverage score in the active buffer, or None if empty."""
        scores = self._active_buffer.get_leverage_scores()
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
            tau_estimate=self._plant_id.tau,  # backward compat
            tau_observations=self._plant_id.observations,  # backward compat
            plant_identifier_state=self._plant_id.as_dict(),
            observation_buffer_heat=self._obs_buffer_heat_cache,
            observation_buffer_cool=self._obs_buffer_cool_cache,
            greybox_buffer=self._greybox_buffer_cache,
            drift_correction_signs=self._drift_correction_signs,
            last_batch_result=(
                {
                    **dataclasses.asdict(self._last_batch_result),
                    "held_features": list(self._last_batch_result.held_features),
                }
                if self._last_batch_result is not None
                else None
            ),
            last_batch_wallclock=self._last_batch_wallclock,
            tuning_alert_counters={
                **self._tuning_alert_counters,
                "had_stable_batch": int(self._has_had_stable_batch),
                "greybox_has_been_good": int(self._greybox_has_been_good),
            },
            tuning_alert_snapshots=dict(self._tuning_alert_snapshots),
            hp_deadband_estimate_heat=self._hp_deadband_estimate_heat,
            hp_deadband_estimate_cool=self._hp_deadband_estimate_cool,
            exclusion_count=self._exclusion_count,
            auto_perturb_state=self._auto_perturb.as_dict(),
            manual_override_heat=list(self._manual_override_heat),
            manual_override_cool=list(self._manual_override_cool),
            batch_cycle_count=self._batch_cycle_count,
            unlock_batch_cycle=list(self._unlock_batch_cycle),
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
            # If the restored model had observations, it was past the
            # batch-first gate before restart — restore that state.
            if self._rls_heat.observation_count > 0:
                self._rls_heat_mature = True
        if data.rls_cool_model:
            self._rls_cool = RLSModel.from_dict(
                data.rls_cool_model, self._n_model_inputs,
                seed_coefficients=self._cool_seeds,
                coeff_clamps=self._rls_cool_clamps,
                feature_scales=self._feature_scales,
            )
            if self._rls_cool.observation_count > 0:
                self._rls_cool_mature = True
        # Restore per-mode observation buffers for batch learning.
        _n_buf_features = len(self._feature_order)
        _model_inputs_cfg = self._inputs.model_inputs
        if data.observation_buffer_heat:
            self._observation_buffer_heat = DiversityAwareBuffer.from_list(
                data.observation_buffer_heat,
                n_features=_n_buf_features,
                feature_order=self._feature_order,
                model_inputs=_model_inputs_cfg,
            )
            self._obs_buffer_heat_cache = data.observation_buffer_heat
        if data.observation_buffer_cool:
            self._observation_buffer_cool = DiversityAwareBuffer.from_list(
                data.observation_buffer_cool,
                n_features=_n_buf_features,
                feature_order=self._feature_order,
                model_inputs=_model_inputs_cfg,
            )
            self._obs_buffer_cool_cache = data.observation_buffer_cool

        # Restore grey-box observation buffer.
        if data.greybox_buffer:
            self._greybox_buffer = GreyboxBuffer.from_list(
                data.greybox_buffer,
                solar_entity=find_solar_entity(self._model_inputs),
            )
            self._greybox_buffer_cache = data.greybox_buffer

        # Restore learned HP thermostat deadband estimates
        self._hp_deadband_estimate_heat = data.hp_deadband_estimate_heat
        self._hp_deadband_estimate_cool = data.hp_deadband_estimate_cool
        if self._hp_deadband_estimate_heat > 0 or self._hp_deadband_estimate_cool > 0:
            _LOGGER.debug(
                "%sRestored HP deadband estimates: heat=%.2f°C, cool=%.2f°C",
                self._log_prefix,
                self._hp_deadband_estimate_heat,
                self._hp_deadband_estimate_cool,
            )

        # Restore anomaly exclusion count
        self._exclusion_count = data.exclusion_count

        # Restore auto-perturbation counters
        if data.auto_perturb_state:
            self._auto_perturb.restore(data.auto_perturb_state)

        # Restore manual override state (three-state: None/True/False).
        # Empty list means pre-override data → all None (auto-gating).
        n = self._rls_heat.n
        if data.manual_override_heat:
            self._manual_override_heat = [
                v if v is not None and isinstance(v, bool) else None
                for v in data.manual_override_heat
            ]
            # Pad if model grew
            while len(self._manual_override_heat) < n:
                self._manual_override_heat.append(None)
        if data.manual_override_cool:
            self._manual_override_cool = [
                v if v is not None and isinstance(v, bool) else None
                for v in data.manual_override_cool
            ]
            while len(self._manual_override_cool) < n:
                self._manual_override_cool.append(None)

        # Restore adaptive batch step cap state
        self._batch_cycle_count = data.batch_cycle_count
        if data.unlock_batch_cycle:
            self._unlock_batch_cycle = list(data.unlock_batch_cycle)
            # Pad if model grew
            while len(self._unlock_batch_cycle) < n:
                self._unlock_batch_cycle.append(None)

        # Restore drift detection history
        if data.drift_correction_signs:
            self._drift_correction_signs = data.drift_correction_signs
        # Restore tuning alert counters and snapshots
        if data.tuning_alert_counters:
            # Migration from pre39: filter out snapshot keys from old combined dict.
            # Remove this filter once all installs have restarted on pre39+.
            self._tuning_alert_counters = {
                k: int(v) for k, v in data.tuning_alert_counters.items()
                if not k.startswith("freeze_rms_")
            }
            self._has_had_stable_batch = bool(
                self._tuning_alert_counters.get("had_stable_batch", False)
            )
            self._greybox_has_been_good = bool(
                self._tuning_alert_counters.get("greybox_has_been_good", False)
            )
        if data.tuning_alert_snapshots:
            self._tuning_alert_snapshots = {
                k: float(v) for k, v in data.tuning_alert_snapshots.items()
            }
        # Restore last batch result for diagnostics continuity
        if data.last_batch_result is not None:
            br = data.last_batch_result
            # Convert held_features back to set (serialized as list)
            if "held_features" in br and isinstance(br["held_features"], list):
                br["held_features"] = set(br["held_features"])
            self._last_batch_result = BatchResult(**br)
            self._metrics.batch_model_rms = self._last_batch_result.residual_rms
        if data.last_batch_wallclock:
            self._last_batch_wallclock = data.last_batch_wallclock
        # Seed change detection: if user edited a seed since last save,
        # reset that coefficient to the new seed and increase its uncertainty.
        # Coefficients with unchanged seeds keep their learned values.
        self._apply_seed_changes(data.heat_seeds_at_learn, self._heat_seeds, self._rls_heat)
        self._apply_seed_changes(data.cool_seeds_at_learn, self._cool_seeds, self._rls_cool)
        # Restore lag filter states
        if data.lag_filter_states:
            self._inputs.restore_lag_states(data.lag_filter_states)
        # Restore plant estimate and recompute IMC gains
        if self._plant_id.enabled:
            restore_data = data.plant_identifier_state or {}
            # Migration: if no plant_identifier_state, use old single-tau fields
            if not restore_data and data.tau_estimate > 0:
                restore_data = {
                    "tau_estimate": data.tau_estimate,
                    "tau_observations": data.tau_observations,
                }
            if restore_data:
                old_ki = self._pi_ki
                gains = self._plant_id.restore(restore_data)
                self._apply_gain_update(gains)
                # Re-scale integral for the restored ki (overrides the earlier
                # scaling which used the seed-derived ki, not the restored ki)
                if old_ki > 0 and old_ki != self._pi_ki:
                    rescale = old_ki / self._pi_ki
                    self._pi_integral *= rescale
                    _LOGGER.debug(
                        "PI: re-scaled integral for restored plant "
                        "(ki %.4f → %.4f, scale %.2f)",
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
        self._auto_perturb.abort("user_setpoint_change")
        self._plant_id.cancel_observation()
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
        # TODO: round() assumes 1°C vendor resolution; use vendor setpoint_step
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
            "ff_enabled": self._pi_ff_enabled,
            "ff_learning_suppressed": self._disturbance_suppress_active,
            "integral_convergence": round(self._metrics.integral_convergence, 2),
            "room_temp_rate": round(self._room_temp_rate, 4),  # °C/min
            "effective_kp": round(self._pi_kp, 3),
            "effective_ki": round(self._pi_ki, 4),
            "tau_estimate": round(self._plant_id.tau, 1) if self._plant_id.enabled else None,  # backward compat
            "tau_fast": round(self._plant_id.plant.tau_fast.value, 1) if self._plant_id.enabled else None,
            "tau_slow": round(self._plant_id.plant.tau_slow.value, 1) if self._plant_id.enabled else None,
            "tau_observations": self._plant_id.observations if self._plant_id.enabled else None,  # backward compat
            "tau_fast_observations": self._plant_id.plant.tau_fast.observations if self._plant_id.enabled else None,
            "tau_slow_observations": self._plant_id.plant.tau_slow.observations if self._plant_id.enabled else None,
            "smith_correction": (
                round(self._smith.correction, 3) if self._smith is not None else None
            ),
            "setpoint_changes_total": self._metrics.setpoint_changes,
            "sensor_filtered": (
                round(self._sensor_filtered, 3)
                if self._sensor_filtered is not None else None
            ),
            "greybox_tau_eff": (
                round(self._last_greybox_result.tau_eff, 1)
                if self._last_greybox_result is not None else None
            ),
            "greybox_tau_agreement_pct": (
                round(self._last_greybox_result.tau_agreement_pct, 1)
                if self._last_greybox_result is not None
                and self._last_greybox_result.tau_agreement_pct is not None
                else None
            ),
            "greybox_rms": (
                round(self._last_greybox_result.residual_rms, 5)
                if self._last_greybox_result is not None else None
            ),
            "greybox_last_run": self._last_greybox_timestamp_iso,
            "greybox_gates_passed": (
                self._last_greybox_bridge.gates_passed
                if self._last_greybox_bridge is not None else None
            ),
            "greybox_k_eff": (
                round(self._last_greybox_bridge.k_eff, 3)
                if self._last_greybox_bridge is not None else None
            ),
            "greybox_buffer_size": len(self._greybox_buffer),
        }

    def _log_greybox_wls_comparison(
        self,
        bridge: GreyboxBridgeResult,
        wls_result: BatchResult,
    ) -> None:
        """Log side-by-side comparison of grey-box bridge β vs WLS β."""
        names = self._coeff_names()
        wls_beta = wls_result.beta_batch
        wls_se = wls_result.beta_std_err

        _LOGGER.info(
            "%sGrey-box vs WLS cross-validation (gates %s):",
            self._log_prefix,
            "PASS" if bridge.gates_passed else "FAIL",
        )
        for i in range(min(len(bridge.beta), len(wls_beta))):
            name = names[i] if i < len(names) else f"β{i}"
            gb = bridge.beta[i]
            ws = wls_beta[i]
            gb_se = bridge.beta_std_err[i]
            ws_se = wls_se[i] if i < len(wls_se) else float("inf")

            if gb is None:
                _LOGGER.info(
                    "%s  %s: WLS=%.4f (σ=%.4f), grey-box=N/A",
                    self._log_prefix, name, ws, ws_se,
                )
            else:
                diff = abs(gb - ws)
                pct = diff / max(abs(ws), 1e-6) * 100
                _LOGGER.info(
                    "%s  %s: WLS=%.4f (σ=%.4f), grey-box=%.4f (σ=%.4f), "
                    "diff=%.4f (%.0f%%)",
                    self._log_prefix, name, ws, ws_se, gb, gb_se, diff, pct,
                )

    def _coeff_names(self) -> list[str]:
        """Build coefficient name list: intercept, outdoor_delta, then model inputs."""
        names = ["intercept", "outdoor_delta"]
        for m in self._model_inputs:
            names.append(m.get("name", "input"))
        return names

    def _coeff_role(self, index: int) -> str:
        """Map coefficient index to its input_role string.

        Returns "intercept" (0), "outdoor_delta" (1), or the model input's
        input_role config value (2+).  Default role is "other".
        """
        if index == 0:
            return "intercept"
        if index == 1:
            return "outdoor_delta"
        m_idx = index - 2
        if m_idx < len(self._model_inputs):
            return str(self._model_inputs[m_idx].get("input_role", "other"))
        return "other"

    def _get_frozen_feature_set(self, rls: RLSModel) -> set[int]:
        """Return the set of frozen coefficient indices for batch WLS."""
        return {i for i in range(rls.n) if rls.frozen[i]}

    def _build_per_feature_step_caps(
        self,
        result: BatchResult,
        buffer: DiversityAwareBuffer,
        n_eligible: int,
        kappa: float,
    ) -> list[float] | None:
        """Build per-feature step caps for recently-unlocked features.

        Returns None (all features use default cap) unless at least one
        feature qualifies for the enlarged cap.  Quality gates per
        Belsley (1980): diagnose collinearity per-variable, not globally.

        Global gate:
        - n_eligible ≥ 40 (2× minimum — more data for a larger step)

        Per-feature gates:
        - VIF < 5 (feature not confounded with others; stricter than
          unlock threshold of 10)
        - σ_batch < 1.0 (empirical confirmation estimate is precise)
        - Feature was unlocked within the last 2 batch cycles
        """
        cycle = self._batch_cycle_count
        n = len(self._unlock_batch_cycle)
        if n == 0:
            return None

        # Global quality gate
        if n_eligible < 40:
            return None

        # Per-feature VIF from the buffer
        vif = buffer.compute_vif() if hasattr(buffer, 'compute_vif') else []

        any_enlarged = False
        caps: list[float] = [MAX_STEP_ABS] * n
        for i in range(n):
            unlock_cycle = self._unlock_batch_cycle[i]
            if unlock_cycle is None:
                continue
            # Recently unlocked: within 2 batch cycles of unlock
            if cycle - unlock_cycle > 2:
                # No longer recently unlocked — clear tracker
                self._unlock_batch_cycle[i] = None
                continue

            # Per-feature quality gates
            feat_vif = vif[i] if i < len(vif) else float("inf")
            if feat_vif >= 5.0:
                continue
            se = (
                result.beta_std_err[i]
                if i < len(result.beta_std_err)
                else float("inf")
            )
            if math.isinf(se) or se >= 1.0:
                continue

            caps[i] = UNLOCK_FIRST_STEP
            any_enlarged = True
            coeff_names = self._coeff_names()
            name = coeff_names[i] if i < len(coeff_names) else f"β{i}"
            _LOGGER.info(
                "%sAdaptive step cap: %s[%d] enlarged to ±%.1f°C "
                "(unlocked cycle %d, now cycle %d, VIF=%.1f, σ=%.3f)",
                self._log_prefix, name, i, UNLOCK_FIRST_STEP,
                unlock_cycle, cycle, feat_vif, se,
            )

        return caps if any_enlarged else None

    def _evaluate_feature_unlocks(
        self,
        full_result: BatchResult,
        rls: RLSModel,
        is_heating: bool,
    ) -> None:
        """Evaluate per-feature unlock conditions using the full-model result.

        Args:
            full_result: BatchResult from WLS with ALL features estimated
                (no frozen_features held).  Coefficients are discarded —
                only std_err, held_features, and VIF are used as evidence
                for whether each frozen feature is identifiable.

        Each frozen coefficient independently unfreezes when ALL of:
        1. Feature not in full_result.held_features (sufficient variance)
        2. full_result.beta_std_err[i] is finite (feature estimable)
        3. Per-feature VIF < 10 (Belsley 1980)
        4. For adjacent_zone inputs: additionally κ < 100
        Auto-gating skips features with a manual override (not None).
        """
        n = rls.n
        coeff_names = self._coeff_names()
        manual_override = (
            self._manual_override_heat if is_heating
            else self._manual_override_cool
        )
        mode = "heat" if is_heating else "cool"

        # VIF from the full-model regression (eligible-only data)
        vif = full_result.feature_vif

        # Compute κ once for adjacent_zone gate
        kappa = self._cached_kappa

        unlocked_any = False
        for i in range(n):
            if not rls.frozen[i]:
                continue  # Already unfrozen
            if i < len(manual_override) and manual_override[i] is not None:
                continue  # Manual override — auto-gating doesn't touch

            name = coeff_names[i] if i < len(coeff_names) else f"β{i}"

            # 1. Not held in full model (sufficient variance in data)
            if i in full_result.held_features:
                _LOGGER.debug(
                    "%sFeature unlock: %s[%d] — held (insufficient variance)",
                    self._log_prefix, name, i,
                )
                continue

            # 2. Finite std_err in full model (feature is estimable)
            se = (
                full_result.beta_std_err[i]
                if i < len(full_result.beta_std_err)
                else float("inf")
            )
            if not math.isfinite(se):
                _LOGGER.debug(
                    "%sFeature unlock: %s[%d] — infinite std_err",
                    self._log_prefix, name, i,
                )
                continue

            # 3. VIF < 10 (per-feature multicollinearity check)
            feat_vif = vif[i] if i < len(vif) else float("inf")
            if feat_vif >= 10.0:
                _LOGGER.debug(
                    "%sFeature unlock: %s[%d] — VIF=%.1f (≥10, collinear)",
                    self._log_prefix, name, i, feat_vif,
                )
                continue

            # 4. Adjacent zone: additionally require κ < 100
            if self._coeff_role(i) == "adjacent_zone":
                if kappa is not None and kappa >= 100:
                    _LOGGER.debug(
                        "%sFeature unlock: %s[%d] — adjacent_zone gated by κ=%.0f",
                        self._log_prefix, name, i, kappa,
                    )
                    continue

            # All conditions met — unfreeze
            self.set_frozen(mode, i, frozen=False, manual=False)
            unlocked_any = True
            # Track unlock cycle for adaptive batch step cap
            if i < len(self._unlock_batch_cycle):
                self._unlock_batch_cycle[i] = self._batch_cycle_count
            _LOGGER.info(
                "%sFeature unlock: %s[%d] unfrozen (σ=%.4f, VIF=%.1f)",
                self._log_prefix, name, i, se, feat_vif,
            )

        # Mark RLS as mature when first feature unlocks
        if unlocked_any:
            if is_heating:
                self._rls_heat_mature = True
            else:
                self._rls_cool_mature = True

    def _apply_kappa_gated_lambda(self, rls: RLSModel, is_heating: bool) -> None:
        """Adjust RLS forgetting factor based on cached condition number.

        κ ≤ 30: no change (original λ_base).
        30 < κ < 100: linear interpolation toward λ=1.0.
        κ ≥ 100: λ=1.0 (no forgetting, maximum stability).
        """
        kappa = self._cached_kappa
        original = (
            self._original_lambda_base_heat if is_heating
            else self._original_lambda_base_cool
        )
        if kappa is None or kappa <= 30:
            rls.lambda_base = original
            return

        blend = min((kappa - 30) / 70.0, 1.0)
        new_lambda = original + blend * (1.0 - original)
        if abs(new_lambda - rls.lambda_base) > 0.001:
            _LOGGER.info(
                "%sκ-gated λ: κ=%.0f → λ=%.4f (original=%.4f, blend=%.0f%%)",
                self._log_prefix, kappa, new_lambda, original, blend * 100,
            )
        rls.lambda_base = new_lambda

    def get_learning_state(self) -> dict[str, Any]:
        """Return learning state for the learning sensor.

        Returns dict with:
        - state: "Learning" | "Optimizing" | "Optimized"
        - frozen_features: list of frozen feature names
        - active_features: list of unfrozen feature names
        - observation_count: total observations (heat + cool)
        - ff_confidence: current FF confidence value
        - condition_number: cached κ
        - batch_cycles: count of batch cycles since reset
        """
        coeff_names = self._coeff_names()
        # Use whichever RLS matches the current mode, default to heat
        e = self._entity
        is_heating = e._attr_hvac_mode != HVACMode.COOL
        rls = self._rls_heat if is_heating else self._rls_cool
        n = rls.n

        frozen_names = []
        active_names = []
        # Only track model inputs (2+) for learning state; intercept and
        # outdoor_delta are never frozen by auto-gating.
        for i in range(2, n):
            name = coeff_names[i] if i < len(coeff_names) else f"β{i}"
            if rls.frozen[i]:
                frozen_names.append(name)
            else:
                active_names.append(name)

        n_model = n - 2  # Model input features only
        n_frozen = len(frozen_names)
        if n_model == 0:
            state = "Optimized"  # No model inputs → nothing to learn
        elif n_frozen == n_model:
            state = "Learning"
        elif n_frozen == 0:
            state = "Optimized"
        else:
            state = "Optimizing"

        obs_count = len(self._observation_buffer_heat) + len(self._observation_buffer_cool)

        return {
            "state": state,
            "frozen_features": frozen_names,
            "active_features": active_names,
            "observation_count": obs_count,
            "ff_confidence": round(self._ff_confidence, 3),
            "condition_number": (
                round(self._cached_kappa, 1) if self._cached_kappa is not None
                else None
            ),
        }

    def get_greybox_state(self) -> dict[str, Any]:
        """Return grey-box model state for the greybox statistics sensor.

        Returns dict with:
        - state: "Failed" | "Learning" | "Adequate" | "Good" | "Degraded"
        - Plus model outputs and buffer diagnostics as attributes.
        """
        # Unconditional failures.
        if not SCIPY_AVAILABLE:
            return {"state": "Failed", "reason": "scipy unavailable"}
        if len(self._greybox_buffer) == 0:
            return {"state": "Failed", "reason": "buffer empty"}

        # No fit yet.
        if self._last_greybox_result is None:
            return {
                "state": "Learning",
                "buffer_total": len(self._greybox_buffer),
                "buffer_max": self._greybox_buffer._max_size,
            }

        result = self._last_greybox_result
        bridge = self._last_greybox_bridge

        # Determine state from gates + history.
        gates_passed = bridge is not None and bridge.gates_passed
        if gates_passed:
            state = "Good"
        elif self._greybox_has_been_good:
            state = "Degraded"
        else:
            state = "Adequate"

        # Build attributes.
        gb_diag = self._greybox_buffer.get_diagnostics()
        attrs: dict[str, Any] = {
            "state": state,
            "tau_eff": round(result.tau_eff, 1),
            "ua_c": round(result.ua_c, 6),
            "k_c": round(result.k_c, 6),
            "alpha_c": round(result.alpha_c, 6),
            "residual_rms": round(result.residual_rms, 5),
            "n_observations": result.n_observations,
            "n_hp_on": result.n_hp_on,
            "n_hp_off": result.n_hp_off,
            "last_fit": self._last_greybox_timestamp_iso,
            "buffer_total": gb_diag["total"],
            "buffer_max": gb_diag["max_size"],
            "buffer_hp_off_pct": gb_diag.get("hp_off_pct"),
        }
        if bridge is not None:
            attrs["k_eff"] = round(bridge.k_eff, 3)
            attrs["gates_passed"] = bridge.gates_passed
            attrs["gate_details"] = {
                k: "pass" if v else "fail"
                for k, v in bridge.gate_details.items()
            }
        if result.tau_agreement_pct is not None:
            attrs["tau_agreement_pct"] = round(result.tau_agreement_pct, 1)
        if result.param_std_err:
            attrs["param_std_err"] = {
                k: round(v, 6) for k, v in result.param_std_err.items()
            }
        return attrs

    def get_diagnostic_dump(self) -> dict[str, Any]:
        """Return full diagnostic state for offline analysis (debug bundles)."""
        coeff_names = self._coeff_names()
        heat_dict = self._rls_heat.get_coefficients()
        cool_dict = self._rls_cool.get_coefficients()
        dump: dict[str, Any] = {
            "observation_buffer_heat": self._observation_buffer_heat.as_list(),
            "observation_buffer_cool": self._observation_buffer_cool.as_list(),
            "buffer_size_heat": len(self._observation_buffer_heat),
            "buffer_size_cool": len(self._observation_buffer_cool),
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
                "tau_estimate": round(self._plant_id.tau, 1) if self._plant_id.enabled else None,  # backward compat
                "tau_fast": round(self._plant_id.plant.tau_fast.value, 1) if self._plant_id.enabled else None,
                "tau_slow": round(self._plant_id.plant.tau_slow.value, 1) if self._plant_id.enabled else None,
                "plant_identification": self._plant_id.get_diagnostics() if self._plant_id.enabled else None,
            },
        }
        # Multicollinearity per buffer — gate on sufficient data
        for label, buf in [("heat", self._observation_buffer_heat), ("cool", self._observation_buffer_cool)]:
            n_eligible = sum(1 for o in buf.get_all() if o.clamped_reason not in ("no_output", "clamped") and abs(o.room_rate) < 0.02)
            if n_eligible >= 2 * buf.n_features:
                cond = buf.compute_condition_number()
                if not math.isinf(cond):
                    dump[f"condition_number_{label}"] = round(cond, 1)
                corr = buf.get_pairwise_correlations(coeff_names)
                dump[f"correlated_pairs_{label}"] = [
                    {"feature_a": a, "feature_b": b, "r": round(r, 3)} for a, b, r in corr
                ]
            else:
                dump[f"correlated_pairs_{label}"] = []
        # Residual patterns
        dump["residual_patterns"] = [
            {
                "start_hour": p.start_hour,
                "end_hour": p.end_hour,
                "mean_residual": round(p.mean_residual, 3),
                "n_observations": p.n_observations,
            }
            for p in self._last_residual_patterns
        ]
        # Batch result
        if self._last_batch_result is not None:
            dump["batch_result"] = {
                **dataclasses.asdict(self._last_batch_result),
                "held_features": list(self._last_batch_result.held_features),
            }
        # Grey-box observer result + bridge
        dump["greybox_observer"] = (
            self._last_greybox_result.as_dict()
            if self._last_greybox_result is not None else None
        )
        dump["greybox_bridge"] = (
            self._last_greybox_bridge.as_dict()
            if self._last_greybox_bridge is not None else None
        )
        dump["greybox_buffer"] = self._greybox_buffer.get_diagnostics()
        # Model input configs (roles, names, flags for interpreting feature vectors)
        dump["model_input_configs"] = [
            {
                "name": m.get("name", ""),
                "input_role": m.get("input_role", "other"),
                "delta_from_room": m.get("delta_from_room", False),
                "suppress_learning": m.get("suppress_learning", False),
            }
            for m in self._model_inputs
        ]
        return dump

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
            "tau_estimate": round(self._plant_id.tau, 1) if self._plant_id.enabled else None,  # backward compat
            "tau_fast": round(self._plant_id.plant.tau_fast.value, 1) if self._plant_id.enabled else None,
            "tau_slow": round(self._plant_id.plant.tau_slow.value, 1) if self._plant_id.enabled else None,
            "plant_identification": self._plant_id.get_diagnostics() if self._plant_id.enabled else None,
            "config": {
                "kp": self._pi_kp,
                "ki": self._pi_ki,
                "deadband": self._pi_deadband,
                "setpoint_weight": self._pi_setpoint_weight,
                "tick_fallback": self._pi_tick_fallback,
                "outdoor_temp_sensor": self._inputs.outdoor_temp_sensor,
                "model_inputs": self._model_inputs,
                "ff_enabled": self._pi_ff_enabled,
                "rls_online_enabled": self._pi_rls_online_enabled,
                "batch_wls_enabled": self._pi_batch_wls_enabled,
                "plant_id_enabled": self._pi_plant_id_enabled,
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
                "last_run_wallclock": self._last_batch_wallclock or None,
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

        # Grey-box observer + bridge
        result["greybox_observer"] = (
            self._last_greybox_result.as_dict()
            if self._last_greybox_result is not None else None
        )
        result["greybox_bridge"] = (
            self._last_greybox_bridge.as_dict()
            if self._last_greybox_bridge is not None else None
        )

        # Observation buffer stats (per-mode)
        def _buf_stats(buf: DiversityAwareBuffer) -> dict[str, Any]:
            obs = buf.get_all()
            n_eligible = sum(
                1 for o in obs if o.clamped_reason not in ("no_output", "clamped") and abs(o.room_rate) < 0.02
            )
            stats: dict[str, Any] = {"total": len(obs), "eligible": n_eligible}
            scores = buf.get_leverage_scores()
            if scores:
                stats["leverage_min"] = round(min(scores), 6)
                stats["leverage_median"] = round(sorted(scores)[len(scores) // 2], 6)
                stats["leverage_max"] = round(max(scores), 6)
            if obs:
                now = time_mod.monotonic()
                oldest = min(o.timestamp for o in obs)
                stats["oldest_age_hours"] = round((now - oldest) / 3600, 1)
            stats["max_size"] = buf._max_size
            n_features = buf.n_features
            feature_active: dict[str, int] = {}
            for j in range(2, n_features):
                input_idx = j - 2
                name = coeff_names[j] if j < len(coeff_names) else f"feature_{j}"
                entity_id = self._model_inputs[input_idx].get("entity_id", "") if input_idx < len(self._model_inputs) else ""
                feature_active[name] = sum(
                    1 for o in obs
                    if abs(obs_raw_reading(o, entity_id)) > 1e-6
                ) if entity_id else 0
            if feature_active:
                stats["feature_active_counts"] = feature_active
            # Multicollinearity diagnostics — gate on sufficient data
            # to avoid rank-deficient noise from the regularizer.
            if n_eligible >= 2 * n_features:
                cond = buf.compute_condition_number()
                if not math.isinf(cond):
                    stats["condition_number"] = round(cond, 1)
                    if cond > 100:
                        stats["condition_rating"] = "severe"
                    elif cond > 30:
                        stats["condition_rating"] = "moderate"
                    else:
                        stats["condition_rating"] = "weak"
                corr = buf.get_pairwise_correlations(coeff_names)
                stats["correlated_pairs"] = [
                    {"feature_a": a, "feature_b": b, "r": round(r, 3)}
                    for a, b, r in corr
                ]
            else:
                stats["condition_rating"] = "insufficient_data"
                stats["correlated_pairs"] = []
            return stats

        result["observation_buffer_heat"] = _buf_stats(self._observation_buffer_heat)
        result["observation_buffer_cool"] = _buf_stats(self._observation_buffer_cool)
        result["greybox_buffer"] = self._greybox_buffer.get_diagnostics()

        # Residual time-of-day patterns — under batch_learning
        if result.get("batch_learning") is not None:
            result["batch_learning"]["residual_patterns"] = [
                {
                    "start_hour": p.start_hour,
                    "end_hour": p.end_hour,
                    "mean_residual": round(p.mean_residual, 3),
                    "n_observations": p.n_observations,
                }
                for p in self._last_residual_patterns
            ]

        # FF decomposition: per-feature breakdown of current ff_offset.
        ff_contribs: dict[str, Any] = {}
        for i, name in enumerate(coeff_names):
            coef = round(heat_phys[i], 4) if i < len(heat_phys) else 0.0
            if i == 0:
                filtered_val = 1.0  # intercept
            elif i == 1:
                # outdoor_delta = outdoor_temp - desired_temp (exogenous)
                desired_c = self.desired_temp_celsius
                if self._inputs.outdoor_temp is not None and desired_c is not None:
                    filtered_val = round(self._inputs.outdoor_temp - desired_c, 4)
                else:
                    filtered_val = 0.0
            else:
                input_idx = i - 2
                filtered_val = round(self._inputs.filtered[input_idx], 4) if input_idx < len(self._inputs.filtered) else 0.0
            contribution = round(coef * filtered_val, 4)
            ff_contribs[name] = {
                "coef": coef,
                "filtered": filtered_val,
                "contribution": contribution,
            }
        ff_sum = round(sum(v["contribution"] for v in ff_contribs.values()), 4)
        ff_contribs["_sum"] = ff_sum
        ff_contribs["_blended_offset"] = round(
            self._ff_offset / self._ff_confidence, 4
        ) if self._ff_confidence > 0.001 else None
        result["ff_contributions"] = ff_contribs

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
        """Return current learned coefficients as seed values for config.

        Used by the 'save learned seeds' button to write back to config.
        Uses beta_to_seed() for consistent β→seed conversion.
        """
        result: dict[str, Any] = {}

        if self._rls_heat.n > 1:
            result["outdoor_seed_heat"] = round(self._rls_heat.beta_to_seed(1), 4)
        if self._rls_cool.n > 1:
            result["outdoor_seed_cool"] = round(self._rls_cool.beta_to_seed(1), 4)

        input_seeds: list[dict[str, float]] = []
        for i in range(len(self._model_inputs)):
            beta_idx = i + 2  # 0=intercept, 1=outdoor_delta, 2+=model inputs
            seeds: dict[str, float] = {}
            if beta_idx < self._rls_heat.n:
                seeds["seed_heat"] = round(self._rls_heat.beta_to_seed(beta_idx), 4)
            if beta_idx < self._rls_cool.n:
                seeds["seed_cool"] = round(self._rls_cool.beta_to_seed(beta_idx), 4)
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

    def exclude_observations_by_time(self, start: float, end: float) -> int:
        """Exclude observations from both buffers by monotonic timestamp range.

        Called by the anomaly repair flow when the user chooses to exclude
        contaminated observations. Removes from both heat and cool buffers
        since the anomaly affects the zone regardless of mode.

        Returns total number of observations removed.
        """
        removed = 0
        removed += self._observation_buffer_heat.exclude_time_range(start, end)
        removed += self._observation_buffer_cool.exclude_time_range(start, end)
        if removed:
            self._exclusion_count += 1
            _LOGGER.info(
                "Excluded %d observations in time range [%.0f, %.0f]",
                removed, start, end,
            )
        return removed

    @property
    def coeff_names(self) -> list[str]:
        """Public access to coefficient name list."""
        return self._coeff_names()

    @property
    def n_coefficients(self) -> int:
        """Number of coefficients per mode."""
        return self._rls_heat.n

    def _rls_for_mode(self, mode: str) -> RLSModel:
        """Return the RLS model for the given mode."""
        if mode == "cool":
            return self._rls_cool
        return self._rls_heat

    def get_coefficient(self, mode: str, index: int) -> float | None:
        """Get coefficient value in physical units."""
        rls = self._rls_for_mode(mode)
        if index >= rls.n:
            return None
        return rls.beta[index] / rls.feature_scales[index]

    def set_coefficient(self, mode: str, index: int, value: float) -> None:
        """Set coefficient value (physical units). Converts to normalized space."""
        rls = self._rls_for_mode(mode)
        if index >= rls.n:
            return
        rls.beta[index] = value * rls.feature_scales[index]
        _LOGGER.info(
            "Coefficient %s[%d] manually set to %.4f (%s mode)",
            self._coeff_names()[index] if index < len(self._coeff_names()) else f"β{index}",
            index, value, mode,
        )

    def get_frozen(self, mode: str, index: int) -> bool:
        """Get whether a coefficient is frozen."""
        rls = self._rls_for_mode(mode)
        if index >= rls.n:
            return False
        return rls.frozen[index]

    def set_frozen(self, mode: str, index: int, frozen: bool, *, manual: bool = True) -> None:
        """Set freeze state for a coefficient.

        When freezing, snapshots the current batch residual RMS so
        check_freeze_impact_repair can detect degradation.

        Args:
            manual: If True (default for user calls), sets a manual override
                so auto-gating skips this feature entirely.  Internal calls
                from _evaluate_feature_unlocks pass manual=False.
        """
        rls = self._rls_for_mode(mode)
        if index >= rls.n:
            return
        rls.frozen[index] = frozen
        # Three-state manual override: None=auto, True=force unfrozen, False=force frozen
        if manual:
            override = self._manual_override_heat if mode == "heat" else self._manual_override_cool
            if index < len(override):
                override[index] = not frozen  # True=unfrozen, False=frozen
        snapshot_key = f"freeze_rms_{mode}_{index}"
        counter_key = f"freeze_impact_{mode}_{index}"
        if frozen:
            # Snapshot current batch RMS for later comparison
            current_rms = self._metrics.batch_model_rms
            if current_rms is not None:
                self._tuning_alert_snapshots[snapshot_key] = current_rms
        else:
            # Clear snapshot and sustained counter on unfreeze
            self._tuning_alert_snapshots.pop(snapshot_key, None)
            self._tuning_alert_counters.pop(counter_key, None)
        name = self._coeff_names()[index] if index < len(self._coeff_names()) else f"β{index}"
        _LOGGER.info(
            "Coefficient %s[%d] %s (%s mode)",
            name, index, "frozen" if frozen else "unfrozen", mode,
        )

    def get_coefficient_clamp(self, mode: str, index: int) -> tuple[float, float] | None:
        """Get coefficient clamp in physical units, or None if unclamped."""
        rls = self._rls_for_mode(mode)
        if index >= rls.n or index >= len(rls.coeff_clamps):
            return None
        clamp = rls.coeff_clamps[index]
        if clamp is None:
            return None
        scale = rls.feature_scales[index]
        return (clamp[0] / scale, clamp[1] / scale)

    def _flush_buffers(self, mode: str | None = None) -> None:
        """Clear observation buffer(s) and batch learning state (no greybox)."""
        if mode in (None, "heat"):
            self._observation_buffer_heat.clear()
        if mode in (None, "cool"):
            self._observation_buffer_cool.clear()
        self._last_batch_result = None
        self._last_batch_timestamp = None
        self._last_batch_wallclock = ""
        self._drift_correction_signs = []
        self._has_had_stable_batch = False
        self._tuning_alert_counters = {}
        self._tuning_alert_snapshots = {}

    def _flush_greybox(self) -> None:
        """Clear greybox buffer, results, and bridge."""
        self._greybox_has_been_good = False
        self._greybox_buffer.clear()
        self._greybox_buffer_cache = []
        self._last_greybox_result = None
        self._last_greybox_bridge = None
        self._last_greybox_timestamp_iso = None

    def flush_observation_buffer(self, mode: str | None = None) -> None:
        """Clear observation buffer(s) and reset batch learning state.

        Nuclear option for major renovation or equipment change.

        Args:
            mode: "heat", "cool", or None (both).  When flushing a single
                  mode, drift correction history and batch result are also
                  cleared because they reference the now-invalid data.
        """
        self._flush_buffers(mode)
        self._flush_greybox()
        label = mode or "heat+cool"
        _LOGGER.info("Observation buffer (%s) flushed — batch learning will restart from scratch", label)

    def _check_tuning_health(self, *, from_batch: bool = False) -> list[tuple[str, str, str, dict[str, str], bool, bool, dict[str, Any] | None]]:
        """Evaluate tuning health and return issues for HA Repairs.

        Returns list of (issue_id, severity, translation_key, placeholders,
        should_create, is_fixable, data) tuples.  Called after each batch
        cycle and on startup.

        Args:
            from_batch: True when called after a batch cycle.  Sustained-cycle
                counters are only incremented during batch calls so that
                HA restarts don't inflate them.
        """
        from .health_checks import (
            check_batch_online_disagreement_repair,
            check_covariance_collapse_repair,
            check_freeze_impact_repair,
            check_high_integral_repair,
            check_intercept_absorbing_repair,
            check_model_drift_repair,
            check_multicollinearity_repair,
            check_residual_pattern_repair,
            check_save_seeds_repair,
            check_slope_divergence_repair,
        )

        entry_id = getattr(self._entity, "_config_entry_id", "unknown")
        issues: list[tuple[str, str, str, dict[str, str], bool, bool, dict[str, Any] | None]] = []

        if not self._pi_enabled:
            return issues

        # ── Slope divergence (heat and cool) ────────────────────────
        for mode, rls, configured in [
            ("heat", self._rls_heat, self._outdoor_seed_heat),
            ("cool", self._rls_cool, self._outdoor_seed_cool),
        ]:
            if rls.observation_count == 0:
                continue
            learned = rls.beta_to_seed(1) if rls.n > 1 else configured

            counter_key = f"slope_div_{mode}"
            # Check if currently drifting
            drift_abs = abs(learned - configured)
            drift_pct = (drift_abs / abs(configured)) * 100 if configured != 0 else 0
            if from_batch:
                if drift_pct > 30.0 and drift_abs > 0.05:
                    self._tuning_alert_counters[counter_key] = self._tuning_alert_counters.get(counter_key, 0) + 1
                else:
                    self._tuning_alert_counters[counter_key] = 0

            result = check_slope_divergence_repair(
                learned_slope=learned,
                configured_slope=configured,
                sustained_cycles=self._tuning_alert_counters.get(counter_key, 0),
                mode=mode,
            )
            if result is not None:
                key, placeholders, should_create = result
                fix_data = {
                    "repair_type": "slope_divergence",
                    "entry_id": entry_id,
                    "mode": mode,
                    "learned_slope": learned,
                    "configured_slope": configured,
                } if should_create else None
                issues.append((
                    f"{key}_{entry_id}_{mode}",
                    "warning",
                    key,
                    placeholders,
                    should_create,
                    should_create, fix_data,
                ))

        # ── Save seeds ──────────────────────────────────────────────
        # Check if seeds match learned values (within rounding tolerance)
        seeds_match = True
        for i in range(min(len(self._heat_seeds), self._rls_heat.n)):
            learned_phys = self._rls_heat.beta[i] / self._rls_heat.feature_scales[i] if self._rls_heat.feature_scales[i] != 0 else 0
            if abs(round(learned_phys, 4) - round(self._heat_seeds[i], 4)) > 0.001:
                seeds_match = False
                break

        heat_coeffs = self._rls_heat.get_coefficients()
        outdoor_delta_heat = heat_coeffs.get(1, -self._outdoor_seed_heat)

        from .health_checks import build_coefficient_summary
        coeff_names = self._coeff_names()
        coeff_summary = build_coefficient_summary(
            coeff_names=coeff_names,
            coefficients=heat_coeffs,
            seeds=self._heat_seeds,
            uncertainties=self._rls_heat.get_covariance_diagonal(),
            feature_scales=self._rls_heat.feature_scales,
        )

        result = check_save_seeds_repair(
            integral_convergence=self._metrics.integral_convergence,
            seeds_match_learned=seeds_match,
            already_notified=bool(self._tuning_alert_counters.get("save_seeds_notified")),
            outdoor_delta_heat=outdoor_delta_heat,
            coefficient_summary=coeff_summary,
        )
        if result is not None:
            key, placeholders, should_create = result
            if should_create:
                self._tuning_alert_counters["save_seeds_notified"] = 1
            elif not should_create and seeds_match:
                # Seeds were saved — allow re-notification after next significant change
                self._tuning_alert_counters["save_seeds_notified"] = 0
            fix_data = {
                "repair_type": "save_seeds",
                "entry_id": entry_id,
                "coefficient_summary": coeff_summary,
            } if should_create else None
            issues.append((
                f"{key}_{entry_id}",
                "warning",
                key,
                placeholders,
                should_create,
                should_create,  # is_fixable only when creating
                fix_data,
            ))

        # ── High integral (diagnosed) ───────────────────────────────
        is_heating = self._entity._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
        active_mode = "heat" if is_heating else "cool"
        rls = self._rls_heat if is_heating else self._rls_cool
        configured_seed = self._outdoor_seed_heat if is_heating else self._outdoor_seed_cool
        learned_slope = rls.beta_to_seed(1) if rls.n > 1 else configured_seed

        ki_correction = abs(self._pi_ki * self._metrics.integral_convergence)

        counter_key = "high_integral"
        if from_batch:
            if ki_correction > 2.0:
                self._tuning_alert_counters[counter_key] = self._tuning_alert_counters.get(counter_key, 0) + 1
            else:
                self._tuning_alert_counters[counter_key] = 0

        result = check_high_integral_repair(
            ki_integral_correction=ki_correction,
            sustained_cycles=self._tuning_alert_counters.get(counter_key, 0),
            observation_count=rls.observation_count,
            learned_slope=learned_slope,
            configured_slope=configured_seed,
            uncontrollable_cvh=self._metrics.uncontrollable_cvh,
            total_cvh=self._metrics.comfort_violation_hours,
            pi_ki=self._pi_ki,
            integral_convergence=self._metrics.integral_convergence,
            mode=active_mode,
        )
        if result is not None:
            key, placeholders, should_create = result
            # Only sub-case 4 (tuning) is fixable — others are diagnostic
            is_fixable = should_create and key == "high_integral_tuning"
            fix_data = {
                "repair_type": "high_integral_tuning",
                "entry_id": entry_id,
                "current_ki": self._pi_ki,
                "suggested_ki": float(placeholders.get("suggested_ki", self._pi_ki)),
            } if is_fixable else None
            issues.append((
                f"high_integral_{entry_id}",
                "warning",
                key,
                placeholders,
                should_create,
                is_fixable, fix_data,
            ))

        # ── Covariance collapse at clamp ────────────────────────────
        from ..const import DEFAULT_RLS_DELTA
        for mode_label, rls_model, clamps in [
            ("heat", self._rls_heat, self._rls_heat_clamps),
            ("cool", self._rls_cool, self._rls_cool_clamps),
        ]:
            if rls_model.observation_count == 0:
                continue
            coeffs = rls_model.get_coefficients()
            p_diag = rls_model.get_covariance_diagonal()
            coeff_names = ["intercept", "outdoor_delta"]
            for m in self._model_inputs:
                coeff_names.append(m.get("name", "input"))

            for i in range(1, rls_model.n):  # skip intercept (no clamp)
                clamp = clamps[i] if i < len(clamps) else None
                name = coeff_names[i] if i < len(coeff_names) else f"coeff_{i}"
                result = check_covariance_collapse_repair(
                    coeff_index=i,
                    coeff_name=name,
                    coeff_value=coeffs.get(i, 0.0),
                    clamp=clamp,
                    p_diagonal=p_diag[i] if i < len(p_diag) else 1.0,
                    delta=DEFAULT_RLS_DELTA,
                )
                if result is not None:
                    key, placeholders, should_create = result
                    issues.append((
                        f"{key}_{entry_id}_{mode_label}_{name}",
                        "warning",
                        key,
                        placeholders,
                        should_create,
                        False, None,
                    ))

        # ── Model drift with maturity gate ──────────────────────────
        drift_results = check_model_drift_repair(
            drifting_coefficients=self.get_drifting_coefficients(),
            has_had_stable_batch=self._has_had_stable_batch,
        )
        for key, placeholders, should_create in drift_results:
            coeff_name = placeholders.get("coeff_name", "unknown")
            issues.append((
                f"{key}_{entry_id}_{coeff_name}",
                "warning",
                key,
                placeholders,
                should_create,
                False, None,
            ))

        # ── Intercept absorbing coefficient ─────────────────────────
        # Check both heat and cool models
        for mode_label, rls_model, clamps in [
            ("heat", self._rls_heat, self._rls_heat_clamps),
            ("cool", self._rls_cool, self._rls_cool_clamps),
        ]:
            if rls_model.observation_count == 0:
                continue
            coeffs = rls_model.get_coefficients()
            p_diag = rls_model.get_covariance_diagonal()
            intercept = coeffs.get(0, 0.0)
            coeff_names = ["intercept", "outdoor_delta"]
            for m in self._model_inputs:
                coeff_names.append(m.get("name", "input"))

            coeff_tuples = []
            for i in range(1, rls_model.n):
                name = coeff_names[i] if i < len(coeff_names) else f"coeff_{i}"
                clamp = clamps[i] if i < len(clamps) else None
                coeff_tuples.append((
                    name,
                    coeffs.get(i, 0.0),
                    clamp,
                    p_diag[i] if i < len(p_diag) else 1.0,
                ))

            result = check_intercept_absorbing_repair(
                intercept_value=intercept,
                coefficients=coeff_tuples,
                delta=DEFAULT_RLS_DELTA,
            )
            if result is not None:
                key, placeholders, should_create = result
                issues.append((
                    f"{key}_{entry_id}_{mode_label}",
                    "warning",
                    key,
                    placeholders,
                    should_create,
                    False, None,
                ))

        # ── Batch-online disagreement ───────────────────────────────
        if (
            self._last_batch_result is not None
            and self._last_batch_result.beta_blended
            and self._drift_correction_signs
        ):
            is_heating_active = self._entity._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
            active_rls = self._rls_heat if is_heating_active else self._rls_cool
            active_coeffs = active_rls.get_coefficients()
            coeff_names_list = ["intercept", "outdoor_delta"]
            for m in self._model_inputs:
                coeff_names_list.append(m.get("name", "input"))

            n = min(
                len(self._drift_correction_signs),
                len(self._last_batch_result.beta_blended),
                active_rls.n,
            )
            for i in range(n):
                if i >= len(coeff_names_list):
                    break
                drift_signs = self._drift_correction_signs[i] if i < len(self._drift_correction_signs) else []
                blended = self._last_batch_result.beta_blended[i] if i < len(self._last_batch_result.beta_blended) else None
                current = active_coeffs.get(i, 0.0)

                result = check_batch_online_disagreement_repair(
                    coeff_index=i,
                    coeff_name=coeff_names_list[i],
                    drift_signs=drift_signs,
                    current_beta=current,
                    last_blended_beta=blended,
                )
                if result is not None:
                    key, placeholders, should_create = result
                    issues.append((
                        f"{key}_{entry_id}_{coeff_names_list[i]}",
                        "warning",
                        key,
                        placeholders,
                        should_create,
                        False, None,
                    ))

        # ── Residual time-of-day patterns ──────────────────────────
        for idx, pattern in enumerate(self._last_residual_patterns):
            counter_key = f"residual_pattern_{pattern.start_hour}_{pattern.end_hour}"
            if from_batch:
                if abs(pattern.mean_residual) > 0.5:
                    self._tuning_alert_counters[counter_key] = self._tuning_alert_counters.get(counter_key, 0) + 1
                else:
                    self._tuning_alert_counters[counter_key] = 0

            result = check_residual_pattern_repair(
                start_hour=pattern.start_hour,
                end_hour=pattern.end_hour,
                mean_residual=pattern.mean_residual,
                n_observations=pattern.n_observations,
                sustained_cycles=self._tuning_alert_counters.get(counter_key, 0),
            )
            if result is not None:
                key, placeholders, should_create = result
                issues.append((
                    f"{key}_{entry_id}_{pattern.start_hour}_{pattern.end_hour}",
                    "warning",
                    key,
                    placeholders,
                    should_create,
                    False, None,
                ))

        # ── Multicollinearity / condition number ───────────────────
        is_heating_mc = self._entity._attr_hvac_mode in (HVACMode.HEAT, HVACMode.HEAT_COOL, None)
        mc_buffer = self._observation_buffer_heat if is_heating_mc else self._observation_buffer_cool
        if len(mc_buffer) >= 20:
            cond_num = mc_buffer.compute_condition_number()
            coeff_names_mc = ["intercept", "outdoor_delta"]
            for m in self._model_inputs:
                coeff_names_mc.append(m.get("name", "input"))
            corr_pairs = mc_buffer.get_pairwise_correlations(coeff_names_mc, include_top=True)

            counter_key = "multicollinearity"
            if from_batch:
                if cond_num > 30.0:  # Belsley (1980): κ > 30 = moderate
                    self._tuning_alert_counters[counter_key] = self._tuning_alert_counters.get(counter_key, 0) + 1
                else:
                    self._tuning_alert_counters[counter_key] = 0

            result = check_multicollinearity_repair(
                condition_number=cond_num,
                correlated_pairs=corr_pairs,
                sustained_cycles=self._tuning_alert_counters.get(counter_key, 0),
                collinear_groups=self._cached_collinear_groups or None,
            )
            if result is not None:
                key, placeholders, should_create = result
                issues.append((
                    f"{key}_{entry_id}",
                    "warning",
                    key,
                    placeholders,
                    should_create,
                    False, None,
                ))

        # ── Freeze impact (RMS degradation) ────────────────────────
        current_rms = self._metrics.batch_model_rms
        if current_rms is not None:
            coeff_names = self._coeff_names()
            for mode_label, rls_model in [
                ("heat", self._rls_heat),
                ("cool", self._rls_cool),
            ]:
                if rls_model.observation_count == 0:
                    continue
                for i in range(rls_model.n):
                    if not rls_model.frozen[i]:
                        continue
                    snapshot_key = f"freeze_rms_{mode_label}_{i}"
                    rms_at_freeze = self._tuning_alert_snapshots.get(snapshot_key)
                    if rms_at_freeze is None:
                        continue
                    counter_key = f"freeze_impact_{mode_label}_{i}"
                    increase_pct = ((current_rms - rms_at_freeze) / rms_at_freeze) * 100.0 if rms_at_freeze > 0 else 0.0
                    if from_batch:
                        if increase_pct > 20.0:
                            self._tuning_alert_counters[counter_key] = self._tuning_alert_counters.get(counter_key, 0) + 1
                        elif increase_pct < 5.0:
                            self._tuning_alert_counters[counter_key] = 0
                        # else: hysteresis band, don't change counter

                    name = coeff_names[i] if i < len(coeff_names) else f"coeff_{i}"
                    result = check_freeze_impact_repair(
                        coeff_name=name,
                        mode=mode_label,
                        rms_at_freeze=rms_at_freeze,
                        current_rms=current_rms,
                        sustained_cycles=self._tuning_alert_counters.get(counter_key, 0),
                    )
                    if result is not None:
                        key, placeholders, should_create = result
                        issues.append((
                            f"{key}_{entry_id}_{mode_label}_{name}",
                            "warning",
                            key,
                            placeholders,
                            should_create,
                            False, None,
                        ))

        # ── Anomalous observations (CUSUM) ─────────────────────────
        for event in self._anomaly_events:
            # Cause hint based on mode and residual direction
            if event.mode == "heat":
                direction = "unexpected heat loss" if event.mean_residual > 0 else "unexpected heat gain"
            else:
                direction = "unexpected heat gain" if event.mean_residual > 0 else "unexpected heat loss"

            time_range = f"{event.start_time.strftime('%H:%M')} — {event.end_time.strftime('%H:%M')}"
            issue_key = f"anomalous_observation_{entry_id}_{event.start_time.strftime('%Y%m%d_%H%M')}"
            issues.append((
                issue_key,
                "warning",
                "anomalous_observation",
                {
                    "time_range": time_range,
                    "mean_residual": f"{event.mean_residual:+.2f}",
                    "direction": direction,
                    "peak_cusum": f"{event.peak_cusum:.1f}",
                },
                True,
                True,
                {
                    "repair_type": "anomalous_observation",
                    "entry_id": entry_id,
                    "start_mono": event.start_mono,
                    "end_mono": event.end_mono,
                    "time_range": time_range,
                    "direction": direction,
                    "mean_residual": f"{event.mean_residual:+.2f}",
                },
            ))
        # Clear surfaced events — they're now in the issue registry
        self._anomaly_events.clear()

        # ── Frequent exclusions escalation ─────────────────────────
        if self._exclusion_count >= 3:
            issues.append((
                f"frequent_exclusions_{entry_id}",
                "warning",
                "frequent_exclusions",
                {"count": str(self._exclusion_count)},
                True,
                False, None,
            ))

        # Auto-perturbation stall
        hvac_mode = "heat" if self._entity._attr_hvac_mode == HVACMode.HEAT else "cool"
        stall_issue = self._auto_perturb.get_stall_issue(entry_id, hvac_mode)
        if stall_issue is not None:
            issues.append(stall_issue)

        # ── Outdoor temp sensor unavailability ────────────────────────
        # Only relevant when an outdoor temp sensor is configured.
        # Grace: 5 min after startup (entities often unavailable during HA boot),
        # then 30 min of continuous unavailability triggers the repair.
        if self._inputs.outdoor_temp_sensor:
            STARTUP_GRACE = 300.0   # 5 minutes
            UNAVAIL_THRESHOLD = 1800.0  # 30 minutes
            now_mono = time.monotonic()
            past_startup = (now_mono - self._init_time) > STARTUP_GRACE

            outdoor_unavail = (
                self._outdoor_temp_unavailable_since is not None
                and past_startup
                and (now_mono - self._outdoor_temp_unavailable_since) > UNAVAIL_THRESHOLD
            )
            issues.append((
                f"outdoor_temp_unavailable_{entry_id}",
                "warning",
                "outdoor_temp_unavailable",
                {"sensor": self._inputs.outdoor_temp_sensor},
                outdoor_unavail,
                outdoor_unavail,
                None,
            ))

        return issues

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
        expected_slope = -self._outdoor_seed_heat if is_heating else -self._outdoor_seed_cool
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
        active_buf = self._active_buffer
        checks.append(check_feature_diversity(
            active_buf.get_all(),
            active_buf.n_features,
            feature_names,
            self.HEALTH_FEATURE_DIVERSITY_MIN,
            self.HEALTH_FEATURE_DIVERSITY_MIN_OBS,
            model_inputs=self._model_inputs,
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
            "tau_estimate": round(self._plant_id.tau, 1) if self._plant_id.enabled else None,  # backward compat
            "tau_fast": round(self._plant_id.plant.tau_fast.value, 1) if self._plant_id.enabled else None,
            "tau_slow": round(self._plant_id.plant.tau_slow.value, 1) if self._plant_id.enabled else None,
            "smith_correction": (
                round(self._smith.correction, 3) if self._smith is not None else None
            ),
            "auto_perturbation_state": self._auto_perturb.state.value,
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

    # ── Plant test (Layer 3: active identification) ──────────────────

    def start_plant_test(
        self,
        amplitude_c: float = 2.0,
        comfort_min_c: float | None = None,
        comfort_max_c: float | None = None,
        n_cycles: int = 4,
    ) -> bool:
        """Start an active plant identification test.

        Pauses PI and hands control to the relay test provider.
        Returns True if started, False if preconditions not met.
        """
        if not self._plant_id.enabled:
            _LOGGER.warning("%sPlant test: IMC disabled (tau_seed=0), cannot start", self._log_prefix)
            return False
        if self._plant_id.plant_test_active:
            _LOGGER.warning("%sPlant test: already running", self._log_prefix)
            return False
        if self._hp_setpoint is None:
            return False

        e = self._entity
        if e._attr_current_temperature is None:
            return False

        raw_c = TemperatureConverter.convert(
            e._attr_current_temperature,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )

        # Default comfort bounds: current temp ± 2°C if not specified
        if comfort_min_c is None:
            comfort_min_c = raw_c - 2.0
        if comfort_max_c is None:
            comfort_max_c = raw_c + 2.0

        self._pi_paused = True
        self._plant_id.start_plant_test(
            baseline_setpoint_c=self._hp_setpoint,
            amplitude_c=amplitude_c,
            current_c=raw_c,
            comfort_min_c=comfort_min_c,
            comfort_max_c=comfort_max_c,
            n_cycles=n_cycles,
        )
        _LOGGER.info(
            "%sPlant test started: amplitude=±%d°C, comfort=[%.1f, %.1f]°C, %d cycles",
            self._log_prefix, amplitude_c, comfort_min_c, comfort_max_c, n_cycles,
        )
        return True

    def abort_plant_test(self) -> None:
        """Abort any active plant test and resume PI."""
        if self._plant_id.plant_test_active:
            self._plant_id.abort_plant_test()
            self._pi_paused = False
            _LOGGER.info("%sPlant test aborted, PI resumed", self._log_prefix)

    def perturb_now(self) -> None:
        """Request an auto-perturbation cycle (service call handler)."""
        self._auto_perturb.force_start()

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

    def _reset_seeds(self, mode: str | None = None) -> None:
        """Reset RLS models to seed values from config (no integral change)."""
        n = self._rls_heat.n  # same for both models

        if mode in (None, "heat"):
            heat_seeds = [self._intercept_seed_heat, -self._outdoor_seed_heat]
            for m_input in self._model_inputs:
                heat_seeds.append(-float(m_input.get("seed_heat", 0.0)))
            heat_norm = [
                heat_seeds[i] * self._feature_scales[i] if i < len(heat_seeds) else 0.0
                for i in range(n)
            ]
            self._rls_heat.beta = heat_norm
            for i in range(n):
                for j in range(n):
                    self._rls_heat.P[i * n + j] = DEFAULT_RLS_P_INIT if i == j else 0.0
            self._rls_heat.observation_count = 0
            self._rls_heat_mature = False
            # Seed reset re-freezes model input features (confidence invalidated)
            for i in range(2, n):
                self._rls_heat.frozen[i] = True
            self._manual_override_heat = [None] * n

        if mode in (None, "cool"):
            cool_seeds = [self._intercept_seed_cool, -self._outdoor_seed_cool]
            for m_input in self._model_inputs:
                cool_seeds.append(-float(m_input.get("seed_cool", 0.0)))
            cool_norm = [
                cool_seeds[i] * self._feature_scales[i] if i < len(cool_seeds) else 0.0
                for i in range(n)
            ]
            self._rls_cool.beta = cool_norm
            for i in range(n):
                for j in range(n):
                    self._rls_cool.P[i * n + j] = DEFAULT_RLS_P_INIT if i == j else 0.0
            self._rls_cool.observation_count = 0
            self._rls_cool_mature = False
            # Seed reset re-freezes model input features (confidence invalidated)
            for i in range(2, n):
                self._rls_cool.frozen[i] = True
            self._manual_override_cool = [None] * n

    async def async_reset_ff_seeds(self, mode: str | None = None) -> None:
        """Reset feedforward RLS models to seed values from config.

        Args:
            mode: "heat", "cool", or None (both).  Integral is always zeroed.
        """
        self._reset_seeds(mode)
        self._pi_integral = 0.0
        label = mode or "heat+cool"
        _LOGGER.info("FF models (%s) reset to seed values, integral zeroed", label)

    async def async_flush_observation_buffer(self, mode: str | None = None) -> None:
        """Clear observation buffer(s) and reset batch learning state (service handler)."""
        self.flush_observation_buffer(mode=mode)

    def _reset_plant_id(self) -> None:
        """Abort active plant test, cancel observations, reset estimate to seeds."""
        was_testing = self._plant_id.plant_test_active
        self._plant_id.reset()
        if was_testing:
            self._pi_paused = False

    async def async_learning_reset(
        self, targets: list[str], mode: str | None = None
    ) -> None:
        """Unified reset service — selectively reset learning subsystems.

        Args:
            targets: List of subsystems to reset. Valid values:
                "seeds", "buffers", "integral", "plant_id", "greybox".
            mode: "heat", "cool", or None (both). Applies to seeds and buffers.
        """
        if "seeds" in targets:
            self._reset_seeds(mode)
        if "buffers" in targets:
            self._flush_buffers(mode)
        if "integral" in targets:
            self._pi_integral = 0.0
        if "plant_id" in targets:
            self._reset_plant_id()
        if "greybox" in targets:
            self._flush_greybox()
        _LOGGER.info(
            "Learning reset: targets=%s, mode=%s", targets, mode or "heat+cool"
        )

    def get_learning_snapshot(self) -> dict[str, Any]:
        """Capture current learning state for save/restore."""
        return {
            "rls_heat_model": self._rls_heat.as_dict(),
            "rls_cool_model": self._rls_cool.as_dict(),
            "pi_integral": self._pi_integral,
            "manual_override_heat": list(self._manual_override_heat),
            "manual_override_cool": list(self._manual_override_cool),
            "heat_seeds_at_learn": list(self._heat_seeds),
            "cool_seeds_at_learn": list(self._cool_seeds),
        }

    def apply_learning_snapshot(self, data: dict[str, Any]) -> None:
        """Restore learning state from a saved snapshot."""
        heat_dict = data.get("rls_heat_model", {})
        if heat_dict:
            self._rls_heat = RLSModel.from_dict(
                heat_dict, self._n_model_inputs,
                seed_coefficients=self._heat_seeds,
                coeff_clamps=self._rls_heat_clamps,
                feature_scales=self._feature_scales,
            )
            self._rls_heat_mature = self._rls_heat.observation_count > 0

        cool_dict = data.get("rls_cool_model", {})
        if cool_dict:
            self._rls_cool = RLSModel.from_dict(
                cool_dict, self._n_model_inputs,
                seed_coefficients=self._cool_seeds,
                coeff_clamps=self._rls_cool_clamps,
                feature_scales=self._feature_scales,
            )
            self._rls_cool_mature = self._rls_cool.observation_count > 0

        self._pi_integral = float(data.get("pi_integral", 0.0))
        self._manual_override_heat = data.get(
            "manual_override_heat", [None] * self._rls_heat.n
        )
        self._manual_override_cool = data.get(
            "manual_override_cool", [None] * self._rls_cool.n
        )
        _LOGGER.info("Learning snapshot restored")

    def _resolve_active_supplemental_sources(self) -> list[str]:
        """Resolve which supplemental sources are currently active from HA state."""
        active: list[str] = []
        for source in self._supplemental.source_configs:
            if not source.get("input_enabled", True):
                continue
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

        Full-rate (1.0): the integrator accumulates the actual error.
        This actively corrects integral drift from sensor-noise-driven
        deadband boundary crossings, preventing the ratcheting that
        causes limit cycles with suspended or slow-rate integration.

        Monte Carlo with realistic sensor noise (σ=0.1°C, matching
        production DHT sensors): full-rate 0% limit cycles, variable-
        rate 2%, suspended 62%.  Full-rate's active correction inside
        the deadband counteracts the noise-driven ratcheting that
        accumulates when integration is throttled or frozen.

        Anti-cycling is handled by hysteresis (±0.5°C midpoint
        crossing), dwell timer, leaky integrator (α=0.9999), and
        quantization-error feedback — not by throttling integration.
        """
        return 1.0

    # ── RLS learning ─────────────────────────────────────────────────

    def _rls_shared_gate_open(self, learning_suppressed: bool) -> bool:
        """Check shared RLS learning preconditions (toggle, outdoor temp, suppression, tracking)."""
        return (
            self._pi_rls_online_enabled
            and self._inputs.outdoor_temp is not None
            and not learning_suppressed
            and not self._any_model_input_unavailable()
            and not self._supplemental.tracking_mode
            and not self._supplemental.assist_active
        )

    def _update_cusum(
        self,
        residual: float,
        now_mono: float,
        is_heating: bool,
        _now: datetime | None = None,
    ) -> None:
        """Feed one residual to the two-sided CUSUM anomaly detector.

        Runs on every buffer observation (every PI tick where clamped=False).
        Uses MAD-based robust scale estimation (Huber, 1981) and the
        CUSUM algorithm (Page, 1954; Basseville & Nikiforov, 1993).
        """
        self._residual_history.append(residual)

        now = _now or datetime.now()

        # Cooldown: suppress detection after a recent alarm
        if self._cusum_cooldown_until is not None:
            if now < self._cusum_cooldown_until:
                return
            self._cusum_cooldown_until = None

        # Need enough history for reliable MAD
        if len(self._residual_history) < MIN_RESIDUALS_FOR_DETECTION:
            return

        # Robust scale estimate
        sigma = compute_mad_sigma(self._residual_history)
        if self._metrics.batch_model_rms is not None:
            sigma = max(sigma, 0.5 * self._metrics.batch_model_rms)

        # Standardize
        z = residual / sigma

        # Two-sided CUSUM update
        self._cusum_pos = max(0.0, self._cusum_pos + z - CUSUM_K)
        self._cusum_neg = max(0.0, self._cusum_neg - z - CUSUM_K)

        alarm_triggered = self._cusum_pos > CUSUM_H or self._cusum_neg > CUSUM_H

        if alarm_triggered:
            # CUSUM crossed threshold — record event and reset.
            # "Fast initial response" (Lucas & Crosier, 1982): reset
            # accumulators after detection to avoid massive accumulation
            # during prolonged anomalies.
            peak = max(self._cusum_pos, self._cusum_neg)
            event = AnomalyEvent(
                start_time=now,
                start_mono=now_mono,
                end_time=now,
                end_mono=now_mono,
                tick_count=1,
                mean_residual=residual,
                peak_cusum=peak,
                mode="heat" if is_heating else "cool",
            )
            self._anomaly_events.append(event)
            _LOGGER.info(
                "Anomaly detected: %s, residual=%.3f°C, peak_cusum=%.1f, σ̂=%.4f",
                now.strftime("%H:%M"),
                residual,
                peak,
                sigma,
            )
            # Reset and enter cooldown
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0
            self._cusum_cooldown_until = now + timedelta(seconds=CUSUM_COOLDOWN_SEC)

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
        if not self._pi_rls_online_enabled:
            reasons.append("online RLS disabled")
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

    # ── IMC Gain Scheduling (delegated to PlantIdentifier) ────────────

    def _apply_gain_update(self, gains: GainUpdate) -> None:
        """Apply a GainUpdate from PlantIdentifier to PI state and Smith predictor."""
        self._pi_kp = gains.kp
        self._pi_ki = gains.ki
        if self._smith is not None:
            self._smith.update_params(tau=gains.tau_fast, lag=gains.lag)

    def _recompute_imc_gains(self) -> None:
        """Recompute IMC gains from current τ estimate and apply them."""
        gains = self._plant_id.compute_gains()
        self._apply_gain_update(gains)

    # ── PI Internals ──────────────────────────────────────────────────

    def _resolve_model_input_states(self) -> dict[str, tuple[str, bool, str | None]]:
        """Resolve all model input and gate entity states from HA."""
        states: dict[str, tuple[str, bool, str | None]] = {}
        for m_input in self._model_inputs:
            for key in ("entity_id", "gate_entity"):
                eid = m_input.get(key, "")
                if not eid or eid in states:
                    continue
                state = self._hass.states.get(eid)
                if state is None or state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    states[eid] = ("", False, None)
                else:
                    unit = state.attributes.get("unit_of_measurement")
                    states[eid] = (state.state, True, unit)
        return states

    def _read_model_input_values(self, room_temp_c: float | None = None) -> None:
        """Resolve HA entity states and update model input manager."""
        self._inputs.read_values(self._resolve_model_input_states(), room_temp_c)

    def _any_model_input_unavailable(self) -> bool:
        """Check if any model input entity is currently unavailable in HA."""
        return self._inputs.any_unavailable(self._resolve_model_input_states())

    @callback
    def _async_outdoor_temp_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle outdoor temperature sensor state changes."""
        new_state = event.data.get("new_state")
        if new_state is None:
            return
        if new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            if self._inputs.outdoor_temp is not None:
                # Transition from valid → unavailable: start tracking
                self._outdoor_temp_unavailable_since = time.monotonic()
                _LOGGER.warning(
                    "%sOutdoor temp sensor unavailable — FF frozen, learning paused",
                    self._log_prefix,
                )
            self._inputs.outdoor_temp = None
            return
        unit = new_state.attributes.get("unit_of_measurement", UnitOfTemperature.CELSIUS)
        self._inputs.update_outdoor_temp(new_state.state, unit)
        if self._outdoor_temp_unavailable_since is not None:
            duration = time.monotonic() - self._outdoor_temp_unavailable_since
            _LOGGER.info(
                "%sOutdoor temp sensor recovered after %.0f s",
                self._log_prefix, duration,
            )
            self._outdoor_temp_unavailable_since = None

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
        min_cooldown = 60.0  # seconds between sensor-driven ticks
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
        # Use RLS model for FF-only fallback (sensor unavailable)
        # outdoor_delta references desired temp (exogenous, no PV coupling)
        if self._pi_ff_enabled and self._inputs.outdoor_temp is not None:
            outdoor_delta = self._inputs.outdoor_temp - desired_c
            self._read_model_input_values()
            x = self._inputs.build_feature_vector(outdoor_delta)
            rls = self._rls_heat if is_heating else self._rls_cool
            self._ff_offset = rls.predict(x)
        elif not self._pi_ff_enabled:
            self._ff_offset = 0.0
        # else: ff_enabled but outdoor_temp None → keep frozen offset
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
                    self._hass, self._pi_tick_fallback, self._pi_timer_callback)
            return result
        finally:
            self._pi_tick_running = False

    def _passive_tick(self) -> bool:
        """Observation-only tick when hvac_mode=OFF.

        Runs the observation path (temp tracking, model inputs, plant ID)
        without active control.  Keeps sensor filters warm and enables
        passive learning of building dynamics (natural cooling curves → τ).

        NOTE: Does NOT buffer observations.  The observation buffer
        architecture needs redesign before passive data can be stored —
        WLS and grey-box have fundamentally different data needs
        (see project_buffer_architecture.md).
        """
        e = self._entity

        # Zero control state but preserve observation state
        self._pi_integral = 0.0
        if self._smith is not None:
            self._smith._initialized = False

        # Need a valid temperature reading
        if e._attr_current_temperature is None:
            return False

        now_mono = time.monotonic()
        if self._pi_last_tick_time > 0:
            dt_seconds = min(now_mono - self._pi_last_tick_time, self._pi_tick_fallback * 2)
        else:
            dt_seconds = float(self._pi_tick_fallback)
        self._pi_last_tick_time = now_mono

        # Convert to °C
        raw_c = TemperatureConverter.convert(
            e._attr_current_temperature,
            e.temperature_unit,
            UnitOfTemperature.CELSIUS,
        )

        # Sensor filter (same as active path)
        if self._sensor_filter_tau > 0 and dt_seconds > 0:
            if self._sensor_filtered is None:
                self._sensor_filtered = raw_c
            alpha = 1.0 - math.exp(-dt_seconds / self._sensor_filter_tau)
            self._sensor_filtered = alpha * raw_c + (1.0 - alpha) * self._sensor_filtered
            current_c = self._sensor_filtered
        else:
            current_c = raw_c

        # Track room temperature rate of change (same as active path)
        self._room_temp_history.append((now_mono, raw_c))
        if len(self._room_temp_history) > 5:
            self._room_temp_history.pop(0)
        if len(self._room_temp_history) >= 2:
            t0, temp0 = self._room_temp_history[0]
            t1, temp1 = self._room_temp_history[-1]
            elapsed_min = (t1 - t0) / 60.0
            if elapsed_min > 0:
                self._room_temp_rate = (temp1 - temp0) / elapsed_min

        # Plant ID: continue observations — natural cooling curves give τ
        self._plant_id.check_observation(
            now_mono, raw_c, 0.0,
            hp_setpoint_c=None,
        )

        # Read model inputs and update lag filters (keeps filters warm)
        self._read_model_input_values(current_c)
        self._inputs.update_lag_filters(dt_seconds)

        # Feed grey-box buffer (HP-off passive observations are critical
        # for isolating ua_c — WLS buffer does NOT get these).
        self._greybox_buffer.add(Observation(
            timestamp=now_mono,
            wall_time=time.time(),
            hp_setpoint=None,
            current_c=current_c,
            desired_c=self._desired_temp or current_c,
            outdoor_temp_c=self._inputs.outdoor_temp,
            room_rate=self._room_temp_rate,
            raw_readings=self._inputs.build_raw_readings(),
            clamped=True,
            clamped_reason="no_output",
            supplemental_active=False,
        ))

        return False  # No IR command

    async def _pi_tick_inner(self, now: datetime | None = None) -> bool:
        """PI + feedforward controller tick implementation. Returns True if send needed."""
        e = self._entity
        if e._attr_hvac_mode == HVACMode.OFF:
            return self._passive_tick()
        if self._desired_temp is None or self._hp_setpoint is None:
            return False
        if e._attr_current_temperature is None:
            if self._sensor_unavailable or self._sensor_recovery_pending:
                return False
            _LOGGER.info("%sPI: temp sensor unavailable, requesting 60s recovery check", self._log_prefix)
            self._sensor_recovery_pending = True
            self._recovery_check_needed = True
            return False
        # Plant test (Layer 3): runs instead of normal PI when active.
        # Checked before _pi_paused because the test itself sets paused=True.
        if self._plant_id.plant_test_active:
            now_mono = time.monotonic()
            raw_c = TemperatureConverter.convert(
                e._attr_current_temperature,
                e.temperature_unit,
                UnitOfTemperature.CELSIUS,
            )
            cmd = self._plant_id.tick_plant_test(now_mono, raw_c)
            if cmd.phase in ("complete", "aborted"):
                self._pi_paused = False
                if cmd.phase == "complete":
                    gains = self._plant_id.compute_gains()
                    self._apply_gain_update(gains)
                return True  # send IR to restore normal setpoint
            self._hp_setpoint = cmd.setpoint_c
            return True  # send IR with test setpoint

        if self._pi_paused:
            _LOGGER.debug("%sPI tick: skipping, paused by vendor", self._log_prefix)
            return False

        # Time since last tick (for time-normalized integral)
        now_mono = time.monotonic()
        if self._pi_last_tick_time > 0:
            dt_seconds = min(now_mono - self._pi_last_tick_time, self._pi_tick_fallback * 2)
        else:
            dt_seconds = float(self._pi_tick_fallback)
        self._pi_last_tick_time = now_mono
        dt_factor = dt_seconds / float(self._pi_tick_fallback)

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

        # Auto-perturbation offset (Layer 2.5): inject before error computation.
        # FF sees original desired_c (feature vectors, not error signal).
        is_heating = e._attr_hvac_mode == HVACMode.HEAT
        desired_c += self._auto_perturb.tick(
            now_mono=now_mono,
            room_temp_rate=self._room_temp_rate,
            integral_change_output=(
                abs(self._pi_integral - self._prev_integral_for_oodb) * self._pi_ki
            ),
            ff_settled_ticks=self._ff_settled_ticks,
            is_clamped=(
                self._hp_setpoint is not None
                and (self._hp_setpoint <= self._min_temp_c
                     or self._hp_setpoint >= self._max_temp_c)
            ),
            supplemental_active=(
                self._supplemental.tracking_mode or self._supplemental.assist_active
            ),
            learning_suppressed=self._manual_ff_suppress,
            plant_test_active=self._plant_id.plant_test_active,
            mode_heating=is_heating,
            plant_confidence=min(
                self._plant_id.plant.tau_fast.confidence,
                self._plant_id.plant.tau_slow.confidence,
            ) if self._plant_id.enabled else 1.0,
            current_hour=datetime.now().hour,
        )

        error = desired_c - current_c

        # Check ongoing τ step-response observation (raw — measures real plant).
        # Gated on toggle + all inputs available: plant ID attributes all room
        # temp change to the HP, so unmeasured disturbances contaminate τ.
        if (self._pi_plant_id_enabled
                and self._inputs.outdoor_temp is not None
                and not self._any_model_input_unavailable()):
            tau_gain_update = self._plant_id.check_observation(
                now_mono, raw_c, self._ff_offset,
                hp_setpoint_c=float(self._hp_setpoint) if self._hp_setpoint is not None else None,
            )
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
        self._last_active_heating = is_heating

        # Read model input values and update lag filters
        self._read_model_input_values(current_c)
        self._inputs.update_lag_filters(dt_seconds)

        # ── Feedforward computation ──────────────────────────────────
        # Three branches:
        #   ff_enabled + outdoor_temp available → full FF computation
        #   ff_enabled + outdoor_temp None → freeze offset at last valid value
        #   ff_disabled → zero offset, no feature vector
        rls = self._rls_heat if is_heating else self._rls_cool
        x: list[float] | None = None  # None = no valid feature vector

        if self._pi_ff_enabled and self._inputs.outdoor_temp is not None:
            # Compute outdoor delta: outdoor_temp - desired_temp (signed, same formula both modes)
            # References desired temp (exogenous), not room temp, to prevent positive
            # feedback during transients. Matches industry-standard heating curve
            # formulation: Q_loss = UA × (T_setpoint - T_outdoor).
            outdoor_delta = self._inputs.outdoor_temp - desired_c

            # Build feature vector and predict FF offset via RLS model
            x = self._inputs.build_feature_vector(outdoor_delta)
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
        elif not self._pi_ff_enabled:
            # FF disabled: pure PI, zero offset
            self._ff_offset = 0.0
        # else: ff_enabled but outdoor_temp None → freeze offset at last value

        # Learning suppression: manual service + per-input suppress_learning flag.
        # Also suppress when FF is disabled or feature vector unavailable —
        # RLS needs a valid feature vector to learn from.
        learning_suppressed = self._manual_ff_suppress or x is None
        active_suppressors: list[str] = []
        if self._manual_ff_suppress:
            active_suppressors.append("manual")
        if not self._pi_ff_enabled:
            active_suppressors.append("ff_disabled")
        elif self._inputs.outdoor_temp is None:
            active_suppressors.append("outdoor_temp_unavailable")
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
        # HP compressor inactive: when setpoint < room temp in heating
        # (or > in cooling), the HP's internal thermostat turns off the
        # compressor — zero output, open loop.
        # Ljung §13.3: no plant information during actuator saturation.
        # Åström & Hägglund §6.4: stop integrating when actuator is saturated.
        hp_no_output = (
            (is_heating and self._hp_setpoint < current_c)
            or (is_cooling and self._hp_setpoint > current_c)
        )

        # HP thermostat deadband override: the HP's internal thermostat
        # has its own hysteresis, so the compressor may still cycle even
        # when hp_setpoint is slightly below room temp (heating).  We use
        # a learned deadband estimate as a fast-path margin, plus a rate-
        # based fallback to detect cycling beyond the learned range.
        #
        # hp_no_output stays strict for learning/batch gates — only
        # skip_integration gets the override.
        if hp_no_output:
            self._hp_no_output_ticks += 1
        else:
            self._hp_no_output_ticks = 0

        deadband_margin = (
            self._hp_deadband_estimate_heat if is_heating
            else self._hp_deadband_estimate_cool
        )
        delta = abs(current_c - self._hp_setpoint)

        # Fast path: delta within learned deadband — HP likely still cycling.
        override_learned = hp_no_output and delta <= deadband_margin

        # Slow path: rate-based inference.  After 10 ticks (~10 min) of
        # hp_no_output, if the room isn't cooling (heating) or warming
        # (cooling), the HP must still be producing output despite our
        # prediction.  Updates the learned estimate.
        _OVERRIDE_TICKS = 10
        override_rate = (
            hp_no_output
            and not override_learned
            and self._hp_no_output_ticks >= _OVERRIDE_TICKS
            and (
                (is_heating and self._room_temp_rate >= 0.0)
                or (is_cooling and self._room_temp_rate <= 0.0)
            )
        )
        # Downward learning: HP confirmed off at this delta (room moving
        # in the expected passive direction).  If delta < current estimate,
        # the estimate was too generous — shrink it.  This fires even when
        # delta is within the learned deadband (override_learned=True),
        # because room cooling within the estimate is evidence the estimate
        # is too high (sensor miscalibration, unit serviced, etc.).
        confirmed_off = (
            hp_no_output
            and self._hp_no_output_ticks >= _OVERRIDE_TICKS
            and (
                (is_heating and self._room_temp_rate < 0.0)
                or (is_cooling and self._room_temp_rate > 0.0)
            )
        )
        if confirmed_off and delta < deadband_margin:
            if is_heating:
                _LOGGER.info(
                    "%sHP deadband narrowed (heat): %.2f°C → %.2f°C "
                    "(setpoint=%d°C, room=%.1f°C, rate=%.4f)",
                    self._log_prefix,
                    self._hp_deadband_estimate_heat, delta,
                    self._hp_setpoint, current_c, self._room_temp_rate,
                )
                self._hp_deadband_estimate_heat = delta
            else:
                _LOGGER.info(
                    "%sHP deadband narrowed (cool): %.2f°C → %.2f°C "
                    "(setpoint=%d°C, room=%.1f°C, rate=%.4f)",
                    self._log_prefix,
                    self._hp_deadband_estimate_cool, delta,
                    self._hp_setpoint, current_c, self._room_temp_rate,
                )
                self._hp_deadband_estimate_cool = delta

        if override_rate:
            # Learn: HP is cycling at this delta — grow estimate.
            if is_heating:
                if delta > self._hp_deadband_estimate_heat:
                    _LOGGER.info(
                        "%sHP deadband widened (heat): %.2f°C → %.2f°C "
                        "(setpoint=%d°C, room=%.1f°C, rate=%.4f)",
                        self._log_prefix,
                        self._hp_deadband_estimate_heat, delta,
                        self._hp_setpoint, current_c, self._room_temp_rate,
                    )
                    self._hp_deadband_estimate_heat = delta
            elif delta > self._hp_deadband_estimate_cool:
                _LOGGER.info(
                    "%sHP deadband widened (cool): %.2f°C → %.2f°C "
                    "(setpoint=%d°C, room=%.1f°C, rate=%.4f)",
                    self._log_prefix,
                    self._hp_deadband_estimate_cool, delta,
                    self._hp_setpoint, current_c, self._room_temp_rate,
                )
                self._hp_deadband_estimate_cool = delta

        override_freeze = override_learned or override_rate

        skip_integration = (
            (is_heating and self._hp_setpoint <= self._min_temp_c and error < 0)
            or (is_cooling and self._hp_setpoint >= self._max_temp_c and error > 0)
            or (hp_no_output and not override_freeze)
        )

        # Log transitions into/out of conditional integration freeze.
        if skip_integration and not self._integration_frozen:
            if hp_no_output:
                _LOGGER.debug(
                    "%sIntegration frozen: HP no output "
                    "(setpoint=%d°C, room=%.1f°C, delta=%.2f°C, "
                    "deadband_est=%.2f°C), error=%.2f°C",
                    self._log_prefix, self._hp_setpoint, current_c,
                    delta, deadband_margin, error,
                )
            else:
                _LOGGER.debug(
                    "%sIntegration frozen: %s at %s limit, error=%.2f°C",
                    self._log_prefix,
                    "heating" if is_heating else "cooling",
                    "min" if is_heating else "max",
                    error,
                )
        elif not skip_integration and self._integration_frozen:
            if override_freeze:
                _LOGGER.debug(
                    "%sIntegration unfrozen: HP deadband override "
                    "(%s, delta=%.2f°C, est=%.2f°C, ticks=%d, rate=%.4f)",
                    self._log_prefix,
                    "learned" if override_learned else "rate",
                    delta, deadband_margin,
                    self._hp_no_output_ticks, self._room_temp_rate,
                )
            else:
                _LOGGER.debug(
                    "%sIntegration unfrozen: error=%.2f°C, setpoint=%.1f°C",
                    self._log_prefix, error, self._hp_setpoint,
                )
        self._integration_frozen = skip_integration

        if in_deadband:
            self._ff_settled_ticks += 1
            # Full-rate integration in deadband: accumulate the actual error.
            # Active correction prevents noise-driven integral ratcheting.
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
            rls_mature = self._rls_heat_mature if is_heating else self._rls_cool_mature
            if branch_ready and rls_mature and x is not None and self._rls_shared_gate_open(learning_suppressed):
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
                or hp_no_output
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
            rls_mature = self._rls_heat_mature if is_heating else self._rls_cool_mature
            if branch_ready and rls_mature and x is not None and self._rls_shared_gate_open(learning_suppressed):
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

        # ── Observation recording ────────────────────────────────────
        # Gate on data quality: don't record observations with stale or
        # missing input data.  Good observations are plentiful when all
        # sensors are working; no reason to accept degraded data.
        data_complete = (
            self._inputs.outdoor_temp is not None
            and not self._any_model_input_unavailable()
        )

        # Clamped status computed unconditionally (used by hysteresis below)
        if hp_no_output:
            obs_clamped = True
            obs_clamped_reason = "no_output"
        elif self._hp_setpoint <= self._min_temp_c:
            obs_clamped = True
            obs_clamped_reason = "saturated_low"
        elif self._hp_setpoint >= self._max_temp_c:
            obs_clamped = True
            obs_clamped_reason = "saturated_high"
        else:
            obs_clamped = False
            obs_clamped_reason = ""

        if data_complete:
            # Observation metadata for batch diagnostics.
            obs_integral_change = abs(self._pi_integral - self._prev_integral_for_rls)
            obs_output_change = obs_integral_change * self._pi_ki
            obs_integral_settled = (
                obs_output_change < 0.045
                and abs(self._room_temp_rate) < 0.02
            )
            obs_seconds_since_sp = (
                now_mono - self._last_setpoint_change_time
                if self._last_setpoint_change_time > 0 else 0.0
            )
            obs_supplemental_active = (
                self._supplemental.tracking_mode or self._supplemental.assist_active
            )

            obs = Observation(
                timestamp=now_mono,
                wall_time=time.time(),
                hp_setpoint=float(self._hp_setpoint) if obs_clamped_reason != "no_output" else None,
                current_c=current_c,
                desired_c=desired_c,
                outdoor_temp_c=self._inputs.outdoor_temp,
                room_rate=self._room_temp_rate,
                raw_readings=self._inputs.build_raw_readings(),
                clamped=obs_clamped,
                clamped_reason=obs_clamped_reason,
                supplemental_active=obs_supplemental_active,
            )
            # RLS observation buffer: only when we have a valid feature vector
            # (ff_enabled + outdoor temp available).  HP-off observations are
            # zero-value for regression and waste diversity buffer slots.
            if x is not None and obs_clamped_reason != "no_output":
                active_buffer = self._observation_buffer_heat if is_heating else self._observation_buffer_cool
                active_buffer.add(obs)
            # Grey-box buffer gets ALL observations (including HP-off) when
            # data is complete — greybox learns from room_rate + outdoor temp
            # independently of FF.
            self._greybox_buffer.add(obs)

        # CUSUM anomaly detection — requires valid feature vector (x).
        if x is not None and not obs_clamped:
            cusum_residual = (float(self._hp_setpoint) - desired_c) - rls.predict(x)
            self._update_cusum(cusum_residual, now_mono, is_heating)

        # Midpoint-crossing hysteresis
        new_setpoint = self._hp_setpoint
        if clamped_setpoint > self._hp_setpoint + 0.5:
            new_setpoint = round(clamped_setpoint)
        elif clamped_setpoint < self._hp_setpoint - 0.5:
            new_setpoint = round(clamped_setpoint)
        # TODO: int() assumes 1°C vendor resolution; use vendor setpoint_step
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
                    if (self._pi_plant_id_enabled
                            and self._inputs.outdoor_temp is not None
                            and not self._any_model_input_unavailable()):
                        self._plant_id.start_observation(now_mono, current_c, desired_c, float(change), self._ff_offset)
                    return True
        else:
            _LOGGER.debug(
                "%sPI: error=%.1f raw=%.1f setpoint=%s (held)",
                self._log_prefix, error, clamped_setpoint, self._hp_setpoint,
            )

        return False
