"""Controller adapters for HVAC benchmark.

Wraps specific controller implementations to conform to HVACController protocol.
"""

import asyncio
from datetime import timedelta

from freezegun import freeze_time

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import UnitOfTemperature

from custom_components.tasmota_irhvac.const import DEFAULT_KAPPA_THRESHOLD

from tests.conftest import _PITestEntityRoomTempMixin, make_pi_config
from tests.hvac_bench.constants import _SIM_EPOCH
from tests.hvac_bench.mock_states import MockStates, _BenchHass


class TasmotaPIAdapter:
    """Adapts the Tasmota-IRHVAC PIController to HVACController protocol.

    Wraps the async PIController in a synchronous interface suitable
    for benchmark simulation.
    """

    def __init__(self, config_overrides: dict | None = None,
                 head_calibration_bounds: tuple[float, float] = (0.0, 0.0),
                 *,
                 kappa_threshold: float = DEFAULT_KAPPA_THRESHOLD):
        """Initialize with optional config overrides.

        Args:
            config_overrides: Dict of PI config values to override defaults.
                Common: {"pi_ki": 0.15, "pi_kd": 0.5, "pi_setpoint_weight": 0.3}
            head_calibration_bounds: (cal_min, cal_max) for uncertain zone.
                Default (0.0, 0.0) = no uncertain zone (perfect sensor).
                Use None for production defaults (±2.0°C).
            kappa_threshold: Bench-only test seam — pass 10000 (or larger)
                to disable the production κ-gate for synthetic-learning
                experiments. See PIController.__init__.
        """
        overrides = dict(config_overrides or {})
        # Pull out test-only seed overrides before make_pi_config (they are
        # not real config keys — production uses fixed DEFAULT_TAU_*_SEED).
        tau_fast_seed = overrides.pop("tau_fast_seed", None)
        tau_slow_seed = overrides.pop("tau_slow_seed", None)

        # Initialize sim clock first — the monotonic lambda passed to
        # PIController closes over `self._sim_clock`, so the attribute must
        # exist before construction. Per-tick advances of `_sim_clock` are
        # then visible to the controller's monotonic-time elapsed calculations.
        self._sim_clock = 0.0
        self._mode = "heat"

        config = make_pi_config(overrides)
        self._config = config
        self._entity = _FakeBenchEntity(
            config,
            head_calibration_bounds=head_calibration_bounds,
            kappa_threshold=kappa_threshold,
            monotonic=lambda: self._sim_clock,
            utcnow=lambda: _SIM_EPOCH + timedelta(seconds=self._sim_clock),
        )
        self._pi = self._entity._pi
        self._loop = asyncio.new_event_loop()

        if tau_fast_seed is not None or tau_slow_seed is not None:
            self._inject_plant_seeds(tau_fast_seed, tau_slow_seed)

    @property
    def mock_states(self) -> MockStates:
        """Bench-side states registry — runner calls `.set(eid, value, unit)`."""
        return self._entity.mock_states

    def _inject_plant_seeds(self, tau_fast_seed: float | None,
                            tau_slow_seed: float | None) -> None:
        """Override the plant identifier's τ seeds for per-profile tuning.

        Production uses fixed DEFAULT_TAU_FAST_SEED / DEFAULT_TAU_SLOW_SEED
        (pre44 maturity gate). Tests that want to validate gain scheduling
        across profiles need to inject profile-derived seeds; otherwise
        every profile gets identical Kp/Ki and the test premise collapses.
        """
        from custom_components.tasmota_irhvac.pi.plant_model import PlantEstimate

        plant_id = self._pi._plant_id
        if tau_fast_seed is not None:
            plant_id._tau_fast_seed = float(tau_fast_seed)
        if tau_slow_seed is not None:
            plant_id._tau_slow_seed = float(tau_slow_seed)

        plant_id._plant = PlantEstimate.from_seeds(
            tau_fast_seed=plant_id._tau_fast_seed,
            tau_slow_seed=plant_id._tau_slow_seed,
            response_lag=plant_id._response_lag,
        )

        if plant_id.enabled:
            gains = plant_id.compute_gains()
            self._pi._pi_kp = gains.kp
            self._pi._pi_ki = gains.ki
            if self._pi._smith is not None:
                self._pi._smith.update_params(tau=gains.tau_fast, lag=gains.lag)

    def _apply_pre_tick_state(self, room_temp_c, outdoor_temp_c, dt_seconds,
                              model_inputs=None):
        """Stage sensor/outdoor/inputs/clock before a tick.

        Shared by ``tick()`` and ``apply_user_setpoint_change()`` so both
        tick paths see the same fresh state. Anything that resolves entity
        state via ``hass.states.get`` is updated here (resolver-canonical
        path per #84).
        """
        self._sim_clock += dt_seconds
        self._entity._attr_current_temperature = room_temp_c
        self._pi._inputs.outdoor_temp = outdoor_temp_c

        # Write model inputs to MockStates so the controller's resolver
        # picks them up via the normal `hass.states.get` path. Replaces the
        # earlier `pi._inputs.values[i] = ...` direct write that bypassed
        # the resolver — see #84. Resolver is canonical: production reads
        # state via the resolver, so the bench should too.
        if model_inputs:
            for m_input in self._pi._model_inputs:
                name = m_input.get("name", "")
                if name in model_inputs:
                    entity_id = m_input.get("entity_id", "")
                    unit = "°C" if m_input.get("delta_from_room") else None
                    self.mock_states.set(entity_id, model_inputs[name], unit)

        # Set timing — pre-set to "previous tick's monotonic" so pi_tick's
        # `dt_seconds = now_mono - _pi_last_tick_time` resolves to the
        # configured tick interval rather than the fallback.
        self._pi._pi_last_tick_time = self._sim_clock - dt_seconds

    def tick(self, room_temp_c, outdoor_temp_c, dt_seconds,
             model_inputs=None):
        """Run one PI tick and return HP setpoint."""
        self._apply_pre_tick_state(room_temp_c, outdoor_temp_c, dt_seconds,
                                   model_inputs)

        sim_dt = _SIM_EPOCH + timedelta(seconds=self._sim_clock)
        with freeze_time(sim_dt):
            self._loop.run_until_complete(self._pi._pi_tick())

        return float(self._pi._hp_setpoint)

    def apply_user_setpoint_change(self, temp_c, room_temp_c, outdoor_temp_c,
                                   dt_seconds, model_inputs=None):
        """Drive a user-setpoint-change tick via production's set_temperature.

        Mirrors what production does when a user changes the desired temp in
        the HA UI: bumpless integral transfer (Åström-Hägglund §3.5), emit
        SETPOINT_CHANGE_USER event, abort auto-perturb, cancel plant-id
        observation, set power_mode=ON, then ``_pi_tick()``. See
        ``PIController.set_temperature`` (pi_controller.py:2130).

        This **replaces** (not supplements) the regular sensor-driven
        ``tick()`` at this tick boundary. Rationale: in production,
        ``set_temperature`` calls ``_pi_tick()`` unconditionally
        (pi_controller.py:2174) while the 60s sensor-tick cooldown lives in
        the sensor listener (pi_controller.py:4895-4899). So a user
        setpoint change drives one tick at the user-change moment, and the
        next sensor-driven tick is gated to be at least 60s later — two
        ticks at the same instant never happen.

        With the bench's default 3-min tick spacing (well above the 60s
        cooldown), modeling "user changed setpoint at this boundary" as
        the only tick at this boundary is faithful to production.

        ASSUMPTION — fixed-interval bench ticks. If the bench is ever
        changed to model sensor-driven ticks at variable cadence (e.g.,
        Tasmota's 30s-or-on-change pattern), this "user-driven tick
        replaces the sensor tick at this boundary" contract has to be
        revisited: production fires one tick at the user-change moment
        AND further sensor-driven ticks after the 60s cooldown elapses, so
        the bench would need to model that two-tick pattern instead of
        collapsing both into one boundary.
        """
        self._apply_pre_tick_state(room_temp_c, outdoor_temp_c, dt_seconds,
                                   model_inputs)

        sim_dt = _SIM_EPOCH + timedelta(seconds=self._sim_clock)
        with freeze_time(sim_dt):
            self._loop.run_until_complete(self._pi.set_temperature(temp_c))

        return float(self._pi._hp_setpoint)

    def run_pi_tick_sim_coherent(self) -> None:
        """Run one ``pi._pi_tick()`` with wall-clock frozen to sim time.

        Use when a test needs per-tick state inspection that the high-level
        ``tick()`` API doesn't expose (custom buffer reads, mid-tick state
        injection, etc.) and so has to run its own outer loop.

        Why this exists:
        - ``self._pi._monotonic`` is already sim-coherent (injected at
          construction; see __init__).
        - But the controller also calls ``time.time()`` directly for the
          hour-of-day features (sin_hour / cos_hour). Without
          ``freeze_time``, those reach the real wall clock and the
          features jitter between runs, perturbing eligibility filters
          and buffer counts. The convention is freezegun for wall-clock
          + explicit seam for monotonic (PIController.__init__ docstring
          spells this out). This helper packages both halves so direct
          ``pi._pi_tick()`` callers don't have to remember either.
        """
        sim_dt = _SIM_EPOCH + timedelta(seconds=self._sim_clock)
        with freeze_time(sim_dt):
            self._loop.run_until_complete(self._pi._pi_tick())

    def set_desired_temp(self, temp_c):
        self._pi._desired_temp = temp_c

    def set_mode(self, mode):
        self._mode = mode
        if mode == "heat":
            self._entity._attr_hvac_mode = HVACMode.HEAT
        elif mode == "cool":
            self._entity._attr_hvac_mode = HVACMode.COOL
        else:
            self._entity._attr_hvac_mode = HVACMode.HEAT

    def get_state(self):
        smith = self._pi._smith
        return {
            "integral": self._pi._pi_integral,
            "ff_offset": self._pi._ff_offset,
            "d_term": getattr(self._pi, "_pi_d_filtered", 0.0),
            "rls_obs_count": self._pi._rls_heat.observation_count,
            "desired_temp": self._pi._desired_temp,
            "effective_desired": (
                self._pi._last_effective_desired_c
                if self._pi._last_effective_desired_c is not None
                else self._pi._desired_temp
            ),
            "hp_setpoint": self._pi._hp_setpoint,
            "raw_setpoint": getattr(self._pi, "_last_raw_setpoint", 0.0),
            "smith_correction": smith.correction if smith is not None else 0.0,
        }

    def set_hold_time(self, seconds: float):
        """Override the setpoint hold timer for testing."""
        self._pi._SETPOINT_HOLD_SECONDS = seconds

    def __del__(self):
        if hasattr(self, "_loop") and self._loop and not self._loop.is_closed():
            self._loop.close()


class TextbookPIController:
    """Simple textbook PI controller for cross-validation.

    No feedforward, no RLS, no quantization tricks. Just PI.
    Used to verify benchmark tests aren't too tight.
    """

    def __init__(self, kp=1.0, ki=0.1, setpoint_weight=1.0,
                 min_temp=16.0, max_temp=30.0):
        self.kp = kp
        self.ki = ki
        self.b = setpoint_weight
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.desired = 20.5
        self.integral = 0.0
        self.hp_setpoint = 20.0
        self._mode = "heat"

    def tick(self, room_temp_c, outdoor_temp_c, dt_seconds,
             model_inputs=None):
        error = self.desired - room_temp_c
        dt_factor = dt_seconds / 900.0  # normalize to 15 min

        p_term = self.kp * self.b * error
        self.integral += error * dt_factor
        i_term = self.ki * self.integral

        raw = self.desired + p_term + i_term
        clamped = max(self.min_temp, min(self.max_temp, raw))

        # Simple anti-windup
        if clamped != raw and self.ki != 0:
            if raw > clamped and self.integral > 0:
                self.integral = (clamped - self.desired - p_term) / self.ki
            elif raw < clamped and self.integral < 0:
                self.integral = (clamped - self.desired - p_term) / self.ki

        # Integer quantization with hysteresis
        if clamped > self.hp_setpoint + 0.5:
            self.hp_setpoint = round(clamped)
        elif clamped < self.hp_setpoint - 0.5:
            self.hp_setpoint = round(clamped)
        self.hp_setpoint = int(max(self.min_temp, min(self.max_temp, self.hp_setpoint)))

        return float(self.hp_setpoint)

    def set_desired_temp(self, temp_c):
        self.desired = temp_c

    def set_mode(self, mode):
        self._mode = mode

    def get_state(self):
        return {
            "integral": self.integral,
            "ff_offset": 0.0,
            "desired_temp": self.desired,
            "effective_desired": self.desired,
            "hp_setpoint": self.hp_setpoint,
            "rls_obs_count": 0,
        }


# ── Fake entity for adapter ──────────────────────────────────────────────


class _FakeBenchEntity(_PITestEntityRoomTempMixin):
    """Minimal fake entity for PIController adapter."""

    def __init__(self, config, head_calibration_bounds=None,
                 *, kappa_threshold=DEFAULT_KAPPA_THRESHOLD,
                 monotonic=None, utcnow=None, skip_tick_output=True):
        from custom_components.tasmota_irhvac.pi.pi_controller import PIController

        # Hand-rolled hass fake exposing only the bench-active controller
        # surface (states + is_running).  Replaces the prior
        # ``self.hass = MagicMock()`` pattern that was the architectural
        # root of #83's mock-leak bug class — any new controller hass
        # dependency now raises AttributeError instead of silently
        # consuming a truthy MagicMock child.  See ``_BenchHass`` docstring.
        self.mock_states = MockStates()
        self.hass = _BenchHass(self.mock_states)
        self._pi_test_room_temp = 20.0  # mixin backing field
        self._attr_target_temperature = 20.0
        self._attr_hvac_mode = HVACMode.HEAT
        self._temp_sensor = "sensor.room_temp"
        self.temperature_unit = UnitOfTemperature.CELSIUS
        self._attr_temperature_unit = UnitOfTemperature.CELSIUS
        self.entity_id = "climate.bench_test"
        self._config_entry_id = "bench_test"
        self.unique_id = "bench_test"
        self._temp_precision = config.get("precision", 1.0)
        self._min_temp = config.get("min_temp", 16)
        self._max_temp = config.get("max_temp", 30)
        self._attr_min_temp = self._min_temp
        self._attr_max_temp = self._max_temp

        pi_kwargs = {
            "kappa_threshold": kappa_threshold,
            # Default-True for bench: full_stack runs don't read the
            # TickOutput dispatcher payload, and constructing one per
            # tick (buffer snapshots + leverage scores) is the dominant
            # hot path post-freezegun-fix.  Override to False only when
            # a test asserts on tick_output / coordinator state.
            "skip_tick_output": skip_tick_output,
        }
        if monotonic is not None:
            pi_kwargs["monotonic"] = monotonic
        if utcnow is not None:
            pi_kwargs["utcnow"] = utcnow
        self._pi = PIController(self, config, **pi_kwargs)
        self._sync_room_temp_to_pi()
        self._pi._pi_enabled = True
        # Head calibration bounds: None = production defaults (±2.0°C),
        # explicit tuple overrides both heat and cool modes.
        if head_calibration_bounds is not None:
            cal_min, cal_max = head_calibration_bounds
            self._pi._head_calibration_min_heat = cal_min
            self._pi._head_calibration_max_heat = cal_max
            self._pi._head_calibration_min_cool = cal_min
            self._pi._head_calibration_max_cool = cal_max

    @property
    def device_info(self):
        return None

    def async_schedule_update_ha_state(self, force_refresh=False):
        pass

    async def send_ir(self, *args, **kwargs):
        pass
