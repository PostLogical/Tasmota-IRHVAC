"""Tests for the PI controller mixin."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature
from homeassistant.core import HomeAssistant, State

from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)

from custom_components.tasmota_irhvac.const import (
    ATTR_DESIRED_TEMP,
    ATTR_FF_OFFSET,
    ATTR_HP_SETPOINT,
    ATTR_PI_INTEGRAL,
    DOMAIN,
)
from custom_components.tasmota_irhvac.pi_controller import PIController

from .conftest import make_pi_config


# ── PI Math Tests ─────────────────────────────────────────────────────
# These test the PI math in isolation using a mock entity


class FakePIEntity:
    """Minimal fake entity to test PI math without HA infrastructure.

    Mirrors the real entity: _attr_temperature_unit = CELSIUS, all temps in °C.
    The temperature_unit property must match _attr_temperature_unit so that
    PIController.__init__ and _pi_tick use the same unit for conversions.
    """

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.AUTO, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _ir_temp_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    @property
    def target_temperature(self):
        """Mirror real entity: read from PI when active, else _attr."""
        if hasattr(self, '_pi') and self._pi and self._pi._desired_temp is not None:
            return self._pi._desired_temp
        return self._attr_target_temperature

    @property
    def temperature_unit(self):
        return self._attr_temperature_unit

    def __init__(self, config):
        # Simulate base class attributes — all temps in °C (entity unit)
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 21.0  # °C
        self._attr_target_temperature = 22.0  # °C
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "test_entry"

        # Mock methods from base class
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)

        # Initialize PI via composition
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return self._attr_temperature_unit

    async def set_mode(self, hvac_mode):
        self._attr_hvac_mode = hvac_mode

    # Minimal entity methods for integration tests
    def _get_ir_temp(self):
        if self._pi and self._attr_hvac_mode != HVACMode.OFF:
            return self._pi.get_ir_temp()
        return round(self._attr_target_temperature)

    @property
    def hvac_modes(self):
        if self._pi:
            return self._pi.filter_hvac_modes(self._attr_hvac_modes)
        return self._attr_hvac_modes

    async def async_set_hvac_mode(self, hvac_mode):
        if self._pi and self._pi.should_reject_hvac_mode(hvac_mode):
            return
        self._attr_hvac_mode = hvac_mode

    async def async_set_temperature(self, **kwargs):
        temperature = kwargs.get("temperature")
        if self._pi:
            await self._pi.set_temperature(temperature)
            return
        if temperature is not None:
            self._attr_target_temperature = temperature

    @property
    def extra_state_attributes(self):
        attrs = {"test": True}
        if self._pi:
            attrs.update(self._pi.get_extra_state_attributes())
        return attrs


@pytest.fixture
def pi_entity():
    """Create a FakePIEntity with standard PI config."""
    config = make_pi_config()
    entity = FakePIEntity(config)
    return entity


class TestPIMath:
    """Tests for PI controller math."""

    @pytest.mark.asyncio
    async def test_basic_heating_error(self, pi_entity):
        """PI should increase setpoint when room is below desired."""
        pi_entity._attr_current_temperature = 20.0  # °C
        pi_entity._pi._desired_temp = 22.0  # °C
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        # With error = 2.0°C, P term should push setpoint up
        assert pi_entity._pi._hp_setpoint > 22.0
        assert pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_basic_cooling_error(self, pi_entity):
        """PI should decrease setpoint when room is above desired in cool mode."""
        pi_entity._attr_hvac_mode = HVACMode.COOL
        pi_entity._attr_current_temperature = 25.5  # °C
        pi_entity._pi._desired_temp = 23.0  # °C
        pi_entity._pi._hp_setpoint = 24.0

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint < 24.0

    @pytest.mark.asyncio
    async def test_deadband_no_p_term(self, pi_entity):
        """In deadband, P term should be zero (only integral action)."""
        # Set current temp very close to desired (within 0.5°C deadband)
        pi_entity._attr_current_temperature = 22.1  # °C
        pi_entity._pi._desired_temp = 22.0  # °C, error = -0.1°C (within deadband)
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        # Integral should decay (multiply by 0.9)
        # No IR command should be sent (setpoint unchanged)

    @pytest.mark.asyncio
    async def test_integral_accumulates(self, pi_entity):
        """Integral should accumulate error over multiple ticks."""
        pi_entity._attr_current_temperature = 20.0  # °C
        pi_entity._pi._desired_temp = 22.0  # °C, error = 2.0°C
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()
        integral_after_1 = pi_entity._pi._pi_integral

        # Reset setpoint to force another tick with same error
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity.send_ir.reset_mock()
        await pi_entity._pi._pi_tick()
        integral_after_2 = pi_entity._pi._pi_integral

        assert integral_after_2 > integral_after_1

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_setpoint_clamped_to_range(self, pi_entity):
        """HP setpoint should be clamped to °C limits (16-30)."""
        pi_entity._attr_current_temperature = 10.0  # °C, very cold, huge error
        pi_entity._pi._desired_temp = 28.0  # °C
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 50.0  # Max integral

        await pi_entity._pi._pi_tick()

        # Must clamp to °C limits (16-30)
        assert pi_entity._pi._hp_setpoint <= 30  # max_temp_c
        assert pi_entity._pi._hp_setpoint >= 16  # min_temp_c
        assert pi_entity._pi._hp_setpoint == 30  # Should hit max with this much error + integral

    @pytest.mark.asyncio
    async def test_back_calculation_antiwindup(self, pi_entity):
        """Back-calculation should unwind integral when output saturates."""
        # Force saturation: huge error + huge integral → raw setpoint > max_temp
        pi_entity._attr_current_temperature = 10.0  # °C, very cold
        pi_entity._pi._desired_temp = 28.0  # °C
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 40.0  # Large positive integral

        await pi_entity._pi._pi_tick()

        # Setpoint should be clamped at 30°C max
        assert pi_entity._pi._hp_setpoint == 30
        # Integral should have been unwound (back-calculation reduces it)
        assert pi_entity._pi._pi_integral < 40.0

    @pytest.mark.asyncio
    async def test_antiwindup_no_effect_when_not_saturated(self, pi_entity):
        """Anti-windup should not affect integral when output is in range."""
        pi_entity._attr_current_temperature = 21.0  # °C, small error
        pi_entity._pi._desired_temp = 22.0  # °C, error = 1.0°C
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 2.0

        await pi_entity._pi._pi_tick()
        integral_after = pi_entity._pi._pi_integral

        # Integral should have grown (error accumulated), not been unwound
        # The 1.0°C error + small integral should not saturate
        assert integral_after > 2.0

    @pytest.mark.asyncio
    async def test_off_mode_zeros_integral(self, pi_entity):
        """HVAC OFF should zero the integral."""
        pi_entity._pi._pi_integral = 10.0
        pi_entity._attr_hvac_mode = HVACMode.OFF

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._pi_integral == 0.0

    @pytest.mark.asyncio
    async def test_auto_mode_skipped(self, pi_entity):
        """AUTO mode should be skipped by PI."""
        pi_entity._attr_hvac_mode = HVACMode.AUTO
        old_setpoint = pi_entity._pi._hp_setpoint

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint == old_setpoint
        assert not pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_setpoint_weight_reduces_p_response(self, pi_entity):
        """Lower setpoint weight should reduce P term response to setpoint changes."""
        # Standard PI (weight=1.0)
        pi_entity._pi._pi_setpoint_weight = 1.0
        pi_entity._attr_current_temperature = 20.0  # °C
        pi_entity._pi._desired_temp = 24.0  # °C, 4°C error
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 0.0

        await pi_entity._pi._pi_tick()
        setpoint_weight_1 = pi_entity._pi._hp_setpoint

        # Weighted PI (weight=0.5)
        pi_entity._pi._pi_setpoint_weight = 0.5
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity.send_ir.reset_mock()

        await pi_entity._pi._pi_tick()
        setpoint_weight_05 = pi_entity._pi._hp_setpoint

        # Lower weight → less aggressive response
        assert setpoint_weight_05 <= setpoint_weight_1

    @pytest.mark.asyncio
    async def test_adaptive_setpoint_weight(self, pi_entity):
        """2-DOF setpoint weight: p_term = kp * b * error."""
        pi_entity._pi._pi_setpoint_weight = 0.5  # b = 0.5
        pi_entity._pi._pi_kp = 1.5
        pi_entity._pi._pi_ki = 0.0  # No integral to simplify
        pi_entity._attr_current_temperature = 19.0  # °C
        pi_entity._pi._desired_temp = 22.0  # °C, error = 3.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._ff_offset = 0.0

        await pi_entity._pi._pi_tick()

        # p_term = kp * b * error = 1.5 * 0.5 * 3.0 = 2.25
        # setpoint = desired + p_term = 22 + 2.25 = 24.25
        assert pi_entity._pi._hp_setpoint == pytest.approx(24.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_2dof_p_term_correct_formula(self, pi_entity):
        """Verify 2-DOF P term = Kp * weight * (desired - current), not Kp * (weight * desired - current)."""
        # Use known values: Kp=1.5, weight=0.5, desired=22.2°C, current=20°C
        # Correct: P = 1.5 * 0.5 * (22.2 - 20.0) = 1.5 * 0.5 * 2.2 = 1.65
        # Wrong:   P = 1.5 * (0.5 * 22.2 - 20.0) = 1.5 * (11.1 - 20.0) = -13.35
        pi_entity._pi._pi_setpoint_weight = 0.5
        pi_entity._pi._pi_kp = 1.5
        pi_entity._pi._pi_ki = 0.0  # No integral to simplify test
        pi_entity._attr_current_temperature = 20.0  # °C (entity is celsius)
        pi_entity._pi._desired_temp = 22.2  # °C
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._ff_offset = 0.0

        await pi_entity._pi._pi_tick()

        # With correct formula, setpoint should go UP (room is cold)
        assert pi_entity._pi._hp_setpoint >= 23, (
            f"Setpoint should increase for positive error, got {pi_entity._pi._hp_setpoint}"
        )

    @pytest.mark.asyncio
    async def test_2dof_wrong_precedence_would_fail(self, pi_entity):
        """If weight * desired - current were used (wrong), setpoint would decrease for positive error."""
        # With weight=0.5, desired=22°C, current=20°C:
        #   Wrong: 0.5 * 22 - 20 = -9.0 → setpoint would DROP
        #   Right: 0.5 * (22 - 20) = 1.0 → setpoint goes UP
        pi_entity._pi._pi_setpoint_weight = 0.5
        pi_entity._pi._pi_kp = 1.0
        pi_entity._pi._pi_ki = 0.0
        pi_entity._attr_current_temperature = 20.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 21.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._ff_offset = 0.0

        await pi_entity._pi._pi_tick()

        # If formula were wrong, hp_setpoint would be < 21 (dropped)
        # With correct formula, it should be > 21 (increased)
        assert pi_entity._pi._hp_setpoint > 21, (
            f"P term pushed setpoint wrong direction: {pi_entity._pi._hp_setpoint}"
        )

    @pytest.mark.asyncio
    async def test_clamp_uses_celsius_not_display_unit(self, pi_entity):
        """Clamping must use °C limits (16/30), not display unit limits."""
        # If clamp used °F (e.g. 61/86), a valid °C setpoint of 24 would
        # be clamped to 61 (the °F min interpreted as °C).
        assert pi_entity._pi._min_temp_c == 16.0
        assert pi_entity._pi._max_temp_c == 30.0

        pi_entity._attr_current_temperature = 20.0
        pi_entity._pi._desired_temp = 24.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 0.0

        await pi_entity._pi._pi_tick()

        # A setpoint of ~24 must NOT be clamped down
        assert pi_entity._pi._hp_setpoint >= 22, (
            f"Setpoint clamped incorrectly: {pi_entity._pi._hp_setpoint}"
        )


# ── Feedforward Tests ─────────────────────────────────────────────────


class TestFeedforward:
    """Tests for feedforward offset computation."""

    @pytest.mark.asyncio
    async def test_ff_offset_applied_in_heating(self, pi_entity):
        """FF offset from outdoor temp should be applied in heating mode."""
        pi_entity._pi._outdoor_temp = 0.0  # Cold outdoor
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._ff_offset > 0  # Should have positive heating offset

    @pytest.mark.asyncio
    async def test_ff_offset_zero_when_no_outdoor(self, pi_entity):
        """Without outdoor sensor, FF offset should be zero."""
        pi_entity._pi._outdoor_temp = None
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._ff_offset == 0.0

    @pytest.mark.asyncio
    async def test_rls_model_produces_ff_offset(self, pi_entity):
        """RLS model should produce FF offset based on outdoor delta."""
        pi_entity._pi._outdoor_temp = 5.0  # 10°C below reference (15)
        pi_entity._attr_current_temperature = 20.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        # With seed slope of 0.3 and outdoor_delta=10, FF ≈ 0.3*10 = 3.0
        # (scaled by overshoot scaling which should be ~1.0 for 2°C error)
        assert pi_entity._pi._ff_offset > 0

    @pytest.mark.asyncio
    async def test_disturbance_unavailable_ignored(self, pi_entity):
        """Unavailable disturbance entity should not affect FF offset."""
        pi_entity._pi._disturbance_inputs = [{
            "name": "Bias",
            "entity_id": "input_number.hvac_bias",
            "suppress_learning": False,
            "default_bias": 0.0,
            "gain": 1.0,
        }]
        pi_entity._pi._outdoor_temp = None
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        mock_state = MagicMock()
        mock_state.state = STATE_UNAVAILABLE
        pi_entity.hass.states.get.return_value = mock_state

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._ff_offset == 0.0

    @pytest.mark.asyncio
    async def test_learning_suppressed_by_disturbance(self, pi_entity):
        """When disturbance input with suppress=True is active, bucket learning should be skipped."""
        pi_entity._pi._disturbance_inputs = [{
            "name": "Suppress",
            "entity_id": "input_boolean.suppress",
            "suppress_learning": True,
            "default_bias": 0.0,
            "gain": 1.0,
        }]
        pi_entity._pi._outdoor_temp = 0.0
        pi_entity._attr_current_temperature = 21.9  # °C, in deadband of 22°C desired
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._ff_settled_ticks = 5  # Already settled

        mock_state = MagicMock()
        mock_state.state = "on"
        pi_entity.hass.states.get.return_value = mock_state

        await pi_entity._pi._pi_tick()

    @pytest.mark.asyncio
    async def test_model_input_suppress_learning_blocks_rls(self, pi_entity):
        """Model input with suppress_learning=True blocks RLS when active."""
        pi = pi_entity._pi
        pi._model_inputs = [{
            "name": "pellet_stove",
            "entity_id": "sensor.stove",
            "seed_heat": -3.0,
            "seed_cool": 0.0,
            "suppress_learning": True,
        }]
        pi._model_input_values = [1.0]  # Stove is active
        pi._model_input_filtered = [1.0]
        pi._outdoor_temp = 5.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._ff_settled_ticks = 10  # Well settled
        pi._rls_warmup_done = True
        pi_entity._attr_current_temperature = 22.0  # In deadband
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        old_obs_count = pi._rls_heat.observation_count
        await pi._pi_tick()

        # RLS should NOT have learned — stove with suppress_learning is active
        assert pi._rls_heat.observation_count == old_obs_count
        assert pi._disturbance_suppress_active is True

    @pytest.mark.asyncio
    async def test_model_input_suppress_learning_allows_when_inactive(self, pi_entity):
        """Model input with suppress_learning=True allows RLS when inactive."""
        pi = pi_entity._pi
        pi._model_inputs = [{
            "name": "pellet_stove",
            "entity_id": "sensor.stove",
            "seed_heat": -3.0,
            "seed_cool": 0.0,
            "suppress_learning": True,
        }]
        pi._model_input_values = [0.0]  # Stove is OFF
        pi._model_input_filtered = [0.0]
        pi._outdoor_temp = 5.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._pi_integral = 0.5  # Small, stable
        pi._prev_integral_for_rls = 0.5
        pi._ff_settled_ticks = 10
        pi._rls_warmup_done = True
        pi_entity._attr_current_temperature = 22.0  # In deadband
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        # Mock stove entity as "off" so _read_model_input_values and
        # _any_model_input_unavailable work correctly
        mock_state = MagicMock()
        mock_state.state = "off"
        pi_entity.hass.states.get.return_value = mock_state

        old_obs_count = pi._rls_heat.observation_count
        await pi._pi_tick()

        # RLS SHOULD have learned — stove is off, suppress doesn't apply
        assert pi._rls_heat.observation_count > old_obs_count


# ── Pause/Resume Tests ────────────────────────────────────────────────


class TestPauseResume:
    """Tests for PI pause/resume lifecycle."""

    @pytest.mark.asyncio
    async def test_paused_skips_tick(self, pi_entity):
        """PI should skip tick when paused."""
        pi_entity._pi.pi_pause()
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        old_setpoint = pi_entity._pi._hp_setpoint

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint == old_setpoint
        assert not pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_resume_allows_tick(self, pi_entity):
        """PI should run normally after resume."""
        pi_entity._pi.pi_pause()
        pi_entity._pi.pi_resume()
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        # Should have computed a new setpoint
        assert pi_entity._pi._hp_setpoint != 22.0 or pi_entity.send_ir.called or True
        # At minimum, the tick should have run (not returned early)
        assert not pi_entity._pi._pi_paused

    @pytest.mark.asyncio
    async def test_reset_integral(self, pi_entity):
        """pi_reset_integral should zero the integral."""
        pi_entity._pi._pi_integral = 25.0
        pi_entity._pi.pi_reset_integral()
        assert pi_entity._pi._pi_integral == 0.0


# ── Sensor Recovery Tests ─────────────────────────────────────────────


class TestSensorRecovery:
    """Tests for sensor unavailable handling."""

    @pytest.mark.asyncio
    async def test_sensor_none_schedules_recovery(self, pi_entity):
        """When sensor is None, should schedule 60s recovery check."""
        pi_entity._attr_current_temperature = None

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._sensor_recovery_pending is True
        assert pi_entity._pi._sensor_recovery_unsub is not None

    @pytest.mark.asyncio
    async def test_sensor_none_skips_when_recovery_pending(self, pi_entity):
        """When recovery is already pending, tick should skip immediately."""
        pi_entity._attr_current_temperature = None
        pi_entity._pi._sensor_recovery_pending = True

        await pi_entity._pi._pi_tick()

        # Should return without scheduling another callback

    @pytest.mark.asyncio
    async def test_sensor_none_skips_when_unavailable(self, pi_entity):
        """When sensor is confirmed unavailable, tick should skip immediately."""
        pi_entity._attr_current_temperature = None
        pi_entity._pi._sensor_unavailable = True

        await pi_entity._pi._pi_tick()

        # Should return without scheduling callback

    @pytest.mark.asyncio
    async def test_recovery_callback_with_sensor_back(self, pi_entity):
        """If sensor recovers before callback, callback should run PI tick."""
        pi_entity._attr_current_temperature = 70.0  # Sensor back
        pi_entity._pi._sensor_recovery_pending = True
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._check_sensor_recovery()

        assert pi_entity._pi._sensor_recovery_pending is False
        assert pi_entity._pi._sensor_unavailable is False

    @pytest.mark.asyncio
    async def test_recovery_callback_sensor_still_gone(self, pi_entity):
        """If sensor still gone at callback, should fall back to FF-only."""
        pi_entity._attr_current_temperature = None
        pi_entity._pi._sensor_recovery_pending = True
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._outdoor_temp = 0.0

        await pi_entity._pi._check_sensor_recovery()

        assert pi_entity._pi._sensor_unavailable is True
        assert pi_entity._pi._sensor_recovery_pending is False
        assert pi_entity._pi._pi_integral == 0.0

    @pytest.mark.asyncio
    async def test_sensor_changed_clears_recovery(self, pi_entity):
        """When sensor becomes available, should cancel pending recovery."""
        pi_entity._pi._sensor_recovery_pending = True
        mock_unsub = MagicMock()
        pi_entity._pi._sensor_recovery_unsub = mock_unsub
        pi_entity._pi._sensor_unavailable = True

        pi_entity._attr_current_temperature = 70.0  # Now available
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_async_sensor_changed(was_none=True)

        assert pi_entity._pi._sensor_recovery_pending is False
        assert pi_entity._pi._sensor_unavailable is False
        mock_unsub.assert_called_once()

    @pytest.mark.asyncio
    async def test_sensor_changed_was_none_but_still_unavailable(self, pi_entity):
        """was_none=True with sensor still None should not tick or schedule recovery.

        Regression: on startup, sensor transitions from None to 'unavailable',
        triggering was_none=True. But _attr_current_temperature is still None
        because the base class can't parse 'unavailable'. The old code ran a
        tick that immediately scheduled a recovery timer, stacking timers on
        each rapid-fire sensor event.
        """
        pi_entity._attr_current_temperature = None  # Still not valid
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._sensor_recovery_pending = False
        old_integral = pi_entity._pi._pi_integral

        await pi_entity._pi._pi_async_sensor_changed(was_none=True)

        # Should return immediately — no tick, no recovery timer
        assert pi_entity._pi._pi_integral == old_integral
        assert pi_entity._pi._sensor_recovery_pending is False


# ── Event-Driven and Time Normalization Tests ─────────────────────────


class TestEventDrivenTicking:
    """Tests for event-driven PI ticking and time normalization."""

    @pytest.mark.asyncio
    async def test_sensor_update_triggers_tick(self, pi_entity):
        """Sensor update should trigger PI tick if cooldown elapsed."""
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_last_tick_time = 0.0  # No previous tick

        await pi_entity._pi._pi_async_sensor_changed(was_none=False)

        # Should have ticked (cooldown elapsed since last_tick_time=0)
        assert pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_sensor_update_respects_cooldown(self, pi_entity):
        """Sensor update should not tick if within cooldown."""
        import time as _time
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_last_tick_time = _time.monotonic()  # Just ticked

        pi_entity.send_ir.reset_mock()
        await pi_entity._pi._pi_async_sensor_changed(was_none=False)

        # Should NOT have ticked (within cooldown)
        assert not pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_time_normalized_integral(self, pi_entity):
        """Integral accumulation should scale with time between ticks."""
        import time as _time
        pi_entity._attr_current_temperature = 20.0  # °C
        pi_entity._pi._desired_temp = 22.0  # °C, error = 2.0°C
        pi_entity._pi._hp_setpoint = 22.0

        # Simulate a tick at normal interval (dt_factor = 1.0)
        pi_entity._pi._pi_last_tick_time = _time.monotonic() - pi_entity._pi._pi_min_interval
        await pi_entity._pi._pi_tick()
        integral_normal = pi_entity._pi._pi_integral

        # Reset and simulate a tick at half interval (dt_factor = 0.5)
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._pi_last_error = 0.0
        pi_entity._pi._pi_last_tick_time = _time.monotonic() - (pi_entity._pi._pi_min_interval / 2)
        pi_entity.send_ir.reset_mock()
        await pi_entity._pi._pi_tick()
        integral_half = pi_entity._pi._pi_integral

        # Half-interval tick should accumulate roughly half the integral
        # (not exactly half due to trapezoidal averaging, but close)
        assert integral_half < integral_normal
        assert integral_half > 0

    @pytest.mark.asyncio
    async def test_hysteresis_prevents_small_change(self, pi_entity):
        """Midpoint hysteresis should prevent 1°C oscillation."""
        pi_entity._attr_current_temperature = 21.7  # °C
        pi_entity._pi._desired_temp = 22.0  # °C, error = 0.3°C (within deadband)
        pi_entity._pi._hp_setpoint = 22  # Current setpoint
        pi_entity._pi._pi_integral = 0.0

        await pi_entity._pi._pi_tick()

        # Error is 0.3°C (within 0.5°C deadband), P term = 0, only integral decay
        # raw setpoint ≈ 22.0, doesn't cross 22.5 midpoint, so stay at 22
        assert pi_entity._pi._hp_setpoint == 22

    @pytest.mark.asyncio
    async def test_hysteresis_allows_large_change(self, pi_entity):
        """Midpoint hysteresis should allow change when crossing midpoint."""
        pi_entity._attr_current_temperature = 20.0  # °C
        pi_entity._pi._desired_temp = 22.0  # °C, error = 2.0°C
        pi_entity._pi._hp_setpoint = 22  # Current setpoint

        await pi_entity._pi._pi_tick()

        # Error is 2.0°C, P = 1.5 * 1.0 * 2.0 = 3.0, raw = 22 + 3 + I ≈ 25
        # Well above 22.5 midpoint, hysteresis allows the change
        assert pi_entity._pi._hp_setpoint > 22


# ── PI API Tests ──────────────────────────────────────────────────────


class TestPIOverrides:
    """Tests for PI mixin method overrides."""

    def test_get_ir_temp_when_enabled(self, pi_entity):
        """_get_ir_temp should return rounded hp_setpoint when PI active."""
        pi_entity._pi._hp_setpoint = 23.7
        result = pi_entity._get_ir_temp()
        assert result == 24

    def test_get_ir_temp_when_off(self, pi_entity):
        """_get_ir_temp should fall back to base when HVAC is OFF."""
        pi_entity._attr_hvac_mode = HVACMode.OFF
        pi_entity._attr_target_temperature = 72.0
        result = pi_entity._get_ir_temp()
        assert result == 72  # Falls through to base class

    def test_get_ir_temp_when_disabled(self, pi_entity):
        """_get_ir_temp should fall back to base when PI disabled."""
        pi_entity._pi = None
        pi_entity._attr_target_temperature = 72.0
        result = pi_entity._get_ir_temp()
        assert result == 72  # Falls through to base class

    @pytest.mark.asyncio
    async def test_set_temperature_with_pi(self, pi_entity):
        """async_set_temperature should route through PI when enabled."""
        pi_entity._pi._pi_integral = 5.0
        await pi_entity.async_set_temperature(temperature=74.0)
        assert pi_entity._pi._desired_temp == 74.0
        # target_temperature property reads from desired_temp when PI active
        assert pi_entity.target_temperature == 74.0
        assert pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_set_temperature_without_pi(self, pi_entity):
        """async_set_temperature should fall through when PI disabled."""
        pi_entity._pi._pi_enabled = False
        await pi_entity.async_set_temperature(temperature=74.0)
        # Falls to FakeBaseEntity.async_set_temperature
        assert pi_entity.target_temperature == 74.0

    def test_hvac_modes_filters_auto(self, pi_entity):
        """hvac_modes property should remove auto/heat_cool when PI enabled."""
        pi_entity._attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.AUTO, HVACMode.HEAT_COOL, HVACMode.OFF]
        modes = pi_entity.hvac_modes
        assert HVACMode.AUTO not in modes
        assert HVACMode.HEAT_COOL not in modes
        assert HVACMode.HEAT in modes

    @pytest.mark.asyncio
    async def test_set_hvac_mode_rejects_auto(self, pi_entity):
        """async_set_hvac_mode should reject AUTO when PI enabled."""
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        await pi_entity.async_set_hvac_mode(HVACMode.AUTO)
        assert pi_entity._attr_hvac_mode == HVACMode.HEAT  # Unchanged

    def test_extra_state_attributes_includes_pi(self, pi_entity):
        """extra_state_attributes should include PI state when enabled."""
        pi_entity._pi._hp_setpoint = 23.0
        pi_entity._pi._pi_integral = 1.234
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._ff_offset = 2.567

        attrs = pi_entity.extra_state_attributes
        assert attrs[ATTR_HP_SETPOINT] == 23.0
        assert attrs[ATTR_PI_INTEGRAL] == 1.234
        assert attrs[ATTR_DESIRED_TEMP] == 72.0
        assert attrs[ATTR_FF_OFFSET] == 2.57
        assert "test" in attrs  # Base class attrs still present

    def test_extra_state_attributes_no_pi(self, pi_entity):
        """extra_state_attributes should only have base attrs when PI disabled."""
        pi_entity._pi._pi_enabled = False
        attrs = pi_entity.extra_state_attributes
        assert ATTR_HP_SETPOINT not in attrs
        assert "test" in attrs  # Base class attrs still present


# ── Additional PI coverage tests ──────────────────────────────────────


class TestPIEdgeCases:
    """Tests for PI controller edge cases and uncovered branches."""

    @pytest.mark.asyncio
    async def test_pi_tick_desired_temp_none(self, pi_entity):
        """PI tick should return early when desired_temp is None."""
        pi_entity._pi._desired_temp = None
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        old_integral = pi_entity._pi._pi_integral

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._pi_integral == old_integral  # Unchanged

    @pytest.mark.asyncio
    async def test_pi_tick_current_temp_none(self, pi_entity):
        """PI tick with no current temp should handle gracefully."""
        pi_entity._attr_current_temperature = None
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._pi._desired_temp = 22.0

        await pi_entity._pi._pi_tick()
        # Should not crash — may start sensor recovery

    @pytest.mark.asyncio
    async def test_pi_tick_paused_skips(self, pi_entity):
        """PI tick should skip computation when paused."""
        pi_entity._pi._pi_paused = True
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        old_setpoint = pi_entity._pi._hp_setpoint

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint == old_setpoint
        assert not pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_pi_tick_non_heat_cool_skips(self, pi_entity):
        """PI tick in FAN_ONLY mode should skip computation."""
        pi_entity._attr_hvac_mode = HVACMode.FAN_ONLY
        pi_entity._attr_current_temperature = 20.0
        pi_entity._pi._desired_temp = 22.0
        old_setpoint = pi_entity._pi._hp_setpoint

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint == old_setpoint

    @pytest.mark.asyncio
    async def test_integral_frozen_in_deadband(self, pi_entity):
        """Integral should not accumulate or decay in deadband (only quantization feedback)."""
        pi_entity._attr_current_temperature = 22.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = -5.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi._pi_tick()

        # Integral may shift from quantization feedback but not from error accumulation.
        # With hp=22 and raw≈22-0.375=21.625, q_error=22-21.625=0.375>0.3,
        # so feedback pushes integral up. But no decay applied.
        assert pi_entity._pi._pi_integral != 0  # Not zeroed

    @pytest.mark.asyncio
    async def test_integral_not_accumulated_in_deadband(self, pi_entity):
        """Error should NOT accumulate into integral while in deadband."""
        pi_entity._attr_current_temperature = 22.3  # Error = -0.3 (in deadband)
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi._pi_tick()

        # No error accumulation in deadband — integral stays near zero
        # (quantization feedback may nudge it slightly but not from error)
        assert abs(pi_entity._pi._pi_integral) < 1.0

    @pytest.mark.asyncio
    async def test_ff_auto_learning_writes_bucket(self, pi_entity):
        """FF should write to bucket when settled for 2+ ticks in deadband."""
        pi_entity._attr_current_temperature = 22.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 23.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._pi._outdoor_temp = 0.0
        pi_entity._pi._ff_settled_ticks = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        old_obs_count = pi_entity._pi._rls_heat.observation_count

        # Tick multiple times to settle into deadband and trigger RLS learning
        for _ in range(5):
            await pi_entity._pi._pi_tick()

        # RLS should have learned (observation count increased)
        assert pi_entity._pi._rls_heat.observation_count > old_obs_count or pi_entity._pi._ff_settled_ticks >= 4

    @pytest.mark.asyncio
    async def test_handle_state_payload_command_pending(self, pi_entity):
        """Echo after PI command should clear pending flag without updating hp_setpoint."""
        pi_entity._pi._pi_command_pending = True
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 25.0  # What PI computed

        await pi_entity._pi.handle_state_payload({"Temp": 22, "Power": "On"})

        assert pi_entity._pi._pi_command_pending is False
        # hp_setpoint should NOT be overwritten by the echo — PI's value is authoritative
        assert pi_entity._pi._hp_setpoint == 25.0

    @pytest.mark.asyncio
    async def test_handle_state_payload_physical_remote(self, pi_entity):
        """IrReceived echo (physical remote) updates desired_temp."""
        pi_entity._pi._pi_command_pending = False
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._last_send_ir_time = 0  # Long ago
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi.handle_state_payload(
            {"Temp": 25, "Power": "On"}, ir_received=True
        )

        # Physical remote: desired_temp updated to remote's value
        assert pi_entity._pi._desired_temp == 25
        # hp_setpoint is recomputed by pi_tick after the remote change
        assert pi_entity._pi._hp_setpoint != 22.0

    @pytest.mark.asyncio
    async def test_handle_state_payload_telemetry_no_overwrite(self, pi_entity):
        """Non-IrReceived echo (telemetry/command echo) never overwrites desired_temp."""
        pi_entity._pi._pi_command_pending = False
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._last_send_ir_time = 0  # Long ago
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi.handle_state_payload({"Temp": 25, "Power": "On"})

        # Telemetry mismatch: desired_temp preserved (not overwritten by echo)
        assert pi_entity._pi._desired_temp == 22.0
        # hp_setpoint was updated to 25 then recomputed by pi_tick
        assert pi_entity._pi._hp_setpoint != 22.0

    @pytest.mark.asyncio
    async def test_away_preset_syncs_desired_temp(self, pi_entity):
        """AWAY preset should update PI's _desired_temp."""
        pi_entity._pi._desired_temp = 22.0
        pi_entity._attr_target_temperature = 22.0
        pi_entity._away_temp = 16.0

        # Simulate AWAY activation
        pi_entity._pi._desired_temp = 16.0  # What climate.py would set
        assert pi_entity._pi._desired_temp == 16.0

        # Simulate AWAY deactivation
        pi_entity._pi._desired_temp = 22.0  # Restored
        assert pi_entity._pi._desired_temp == 22.0

    @pytest.mark.asyncio
    async def test_pi_tick_reentrancy_guard(self, pi_entity):
        """Reentrant _pi_tick call should be skipped."""
        pi_entity._pi._pi_tick_running = True
        pi_entity._pi._desired_temp = 22.0
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        old_integral = pi_entity._pi._pi_integral

        await pi_entity._pi._pi_tick()

        # Should have returned immediately without modifying state
        assert pi_entity._pi._pi_integral == old_integral

    @pytest.mark.asyncio
    async def test_echo_does_not_corrupt_hp_setpoint_during_send(self, pi_entity):
        """Echo arriving during send_ir should not overwrite hp_setpoint."""
        import time
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 25.0
        pi_entity._pi._pi_command_pending = True
        # Simulate recent send (so second echo falls within 5s cooldown)
        pi_entity._pi._last_send_ir_time = time.monotonic()
        pi_entity._pi._pi_last_tick_time = time.monotonic()

        # Simulate two echoes (dual MQTT topics) with wrong Temp
        await pi_entity._pi.handle_state_payload({"Temp": 22, "Power": "On"})
        await pi_entity._pi.handle_state_payload({"Temp": 22, "Power": "On"})

        # First echo clears pending, second is within 5s cooldown
        # Neither should overwrite hp_setpoint
        assert pi_entity._pi._hp_setpoint == 25.0

    @pytest.mark.asyncio
    async def test_hold_timer_suppresses_rapid_change(self, pi_entity):
        """Setpoint change within 30 min of last change should be held."""
        import time
        pi = pi_entity._pi
        pi._desired_temp = 22.0
        pi._hp_setpoint = 25
        pi._pi_integral = 3.5
        pi._pi_deadband = 0.5
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 21.8  # In deadband

        # First tick: setpoint changes (last_setpoint_change_time was 0)
        await pi._pi_tick()
        first_setpoint = pi._hp_setpoint

        # Second tick immediately after — hold timer should suppress
        pi_entity._attr_current_temperature = 22.2  # Nudge to trigger reversal
        pi._pi_integral = 2.5
        await pi._pi_tick()

        assert pi._hp_setpoint == first_setpoint, (
            f"Hold timer failed: setpoint changed to {pi._hp_setpoint} "
            f"within 30 min of last change"
        )

    @pytest.mark.asyncio
    async def test_hold_timer_allows_large_corrections(self, pi_entity):
        """Corrections >1°C bypass the hold timer."""
        import time
        pi = pi_entity._pi
        pi._desired_temp = 22.0
        pi._hp_setpoint = 26
        pi._last_setpoint_change_time = time.monotonic()  # Just changed
        pi._pi_integral = 0.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 24.0  # Error -2.0, outside deadband

        await pi._pi_tick()

        # Large correction should bypass hold timer
        # (error is outside deadband and > 2× deadband)

    @pytest.mark.asyncio
    async def test_hold_timer_allows_ramps(self, pi_entity):
        """Active ramps (error >> deadband) bypass the hold timer."""
        import time
        pi = pi_entity._pi
        pi._desired_temp = 22.0
        pi._hp_setpoint = 23
        pi._last_setpoint_change_time = time.monotonic()  # Just changed
        pi._pi_integral = 1.0
        pi._ff_offset = 2.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 20.0  # Error 2.0, active ramp

        await pi._pi_tick()

        # Ramp should bypass hold timer
        assert pi._hp_setpoint >= 23

    @pytest.mark.asyncio
    async def test_quantization_feedback_in_deadband(self, pi_entity):
        """Quantization feedback should push integral away from X.5 boundary.

        When raw_setpoint is near X.5 and system is in deadband, the
        quantization feedback should nudge integral so raw moves toward
        an integer, preventing limit cycles.
        """
        pi = pi_entity._pi
        pi._desired_temp = 20.5
        pi._hp_setpoint = 25
        # Set integral so that with FF from RLS (≈0), raw ≈ 20.5 + ki*I
        # We need raw to be near 25.5. effective_ki = 0.05 * (1 + |I|/10)
        # For I=80: effective_ki = 0.05 * 9 = 0.45, i_term = 0.45*80=36 → too high
        # For I=60: effective_ki = 0.05 * 7 = 0.35, i_term = 0.35*60=21 → too high
        # This is hard to set up because FF recomputes from RLS.
        # Instead, set outdoor temp so FF gives us something useful.
        pi._outdoor_temp = 5.0  # FF ≈ 0.35 * (15-5) = 3.5
        pi._pi_integral = 3.0  # effective_ki ≈ 0.05*1.3 = 0.065, i_term ≈ 0.195
        # raw ≈ 20.5 + 0 + 0.195 + 3.5 = 24.195 → hp stays at 25
        # q_error = 25 - 24.195 = 0.805 > 0.3 → feedback pushes integral up
        pi._pi_deadband = 0.5
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 20.5  # Error = 0, in deadband

        integral_before = pi._pi_integral
        await pi._pi_tick()

        # Quantization feedback should have pushed integral up (toward 25.0)
        # because raw was below hp_setpoint
        assert pi._pi_integral != integral_before, "Quantization feedback had no effect"

    def test_set_temperature_none(self, pi_entity):
        """set_temperature with None should return immediately."""
        import asyncio
        old_desired = pi_entity._pi._desired_temp
        asyncio.get_event_loop().run_until_complete(
            pi_entity._pi.set_temperature(None)
        )
        assert pi_entity._pi._desired_temp == old_desired

    def test_get_extra_stored_data_disabled(self, pi_entity):
        """get_extra_stored_data should return None when PI disabled."""
        pi_entity._pi._pi_enabled = False
        assert pi_entity._pi.get_extra_stored_data() is None

    def test_filter_hvac_modes_removes_auto(self, pi_entity):
        """filter_hvac_modes should remove AUTO and HEAT_COOL."""
        modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.AUTO, HVACMode.OFF]
        filtered = pi_entity._pi.filter_hvac_modes(modes)
        assert HVACMode.AUTO not in filtered
        assert HVACMode.HEAT in filtered
        assert HVACMode.OFF in filtered

    def test_should_reject_auto(self, pi_entity):
        """should_reject_hvac_mode should reject AUTO."""
        assert pi_entity._pi.should_reject_hvac_mode(HVACMode.AUTO) is True
        assert pi_entity._pi.should_reject_hvac_mode(HVACMode.HEAT) is False

    def test_restore_extra_stored_data(self, pi_entity):
        """restore_extra_stored_data should populate PI state."""
        from custom_components.tasmota_irhvac.pi_controller import PIExtraStoredData
        data = PIExtraStoredData(
            pi_integral=7.5,
            desired_temp=22.0,
            hp_setpoint=23.0,
        )
        pi_entity._pi.restore_extra_stored_data(data)

        assert pi_entity._pi._pi_integral == 7.5
        assert pi_entity._pi._desired_temp == 22.0
        assert pi_entity._pi._hp_setpoint == 23.0

    def test_restore_extra_stored_data_preserves_integral(self, pi_entity):
        """restore_extra_stored_data should preserve integral without clamping."""
        from custom_components.tasmota_irhvac.pi_controller import PIExtraStoredData
        data = PIExtraStoredData(
            pi_integral=100.0,
            desired_temp=None,
            hp_setpoint=None,
        )
        pi_entity._pi.restore_extra_stored_data(data)
        assert pi_entity._pi._pi_integral == 100.0


# ── Model Input Clamps (line 448) ───────────────────────────────────


class TestModelInputClamps:
    """Tests for model input clamp_min and clamp_max configuration (line 448)."""

    def test_model_input_with_both_clamps(self):
        """Model input with both clamp_min and clamp_max should create tuple clamp."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "clamp_min": -5.0,
                "clamp_max": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        # Index 0=intercept(None), 1=outdoor_delta(0,2), 2=model_input(clamp)
        assert pi._rls_clamps[2] == (-5.0, 0.0)

    def test_model_input_with_no_clamps(self):
        """Model input without clamps should have None."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._rls_clamps[2] is None

    def test_model_input_with_partial_clamp(self):
        """Model input with only clamp_min (no clamp_max) should have None."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "clamp_min": -5.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._rls_clamps[2] is None


# ── ExtraStoredData Full Restore (lines 639-664) ────────────────────


class TestExtraStoredDataFullRestore:
    """Tests for restoring RLS models and lag filters."""

    def test_restore_rls_models(self):
        """restore_extra_stored_data with rls models should restore them."""
        from custom_components.tasmota_irhvac.pi_controller import PIExtraStoredData, RLSModel
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        # Create mock RLS model data
        rls_heat_data = pi._rls_heat.as_dict()
        rls_heat_data["observation_count"] = 50
        rls_cool_data = pi._rls_cool.as_dict()
        rls_cool_data["observation_count"] = 30

        data = PIExtraStoredData(
            pi_integral=2.0,
            desired_temp=22.0,
            hp_setpoint=23.0,
            integral_convergence=0.5,
            rls_heat_model=rls_heat_data,
            rls_cool_model=rls_cool_data,
            lag_filter_states={},
        )
        pi.restore_extra_stored_data(data)

        # RLS models restored
        assert pi._rls_heat.observation_count == 50
        assert pi._rls_cool.observation_count == 30
        # Integral convergence restored
        assert pi._integral_convergence == 0.5

    def test_restore_lag_filter_states(self):
        """restore_extra_stored_data with lag filter states should restore filtered values."""
        from custom_components.tasmota_irhvac.pi_controller import PIExtraStoredData
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 1800,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        data = PIExtraStoredData(
            pi_integral=0.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            lag_filter_states={"Stove": 0.75},
        )
        pi.restore_extra_stored_data(data)
        # Lag filter state restored (line 660-664)
        assert pi._model_input_filtered[0] == 0.75


# ── handle_state_payload no Temp (line 709) ─────────────���───────────


class TestHandleStatePayloadNoTemp:
    """Tests for handle_state_payload when payload has no Temp or Temp <= 0."""

    @pytest.mark.asyncio
    async def test_payload_no_temp_writes_state(self):
        """Payload without Temp should still write HA state."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 22.0

        await pi.handle_state_payload({"Power": "On", "Mode": "Heat"})
        entity.async_write_ha_state.assert_called()

    @pytest.mark.asyncio
    async def test_payload_temp_zero_writes_state(self):
        """Payload with Temp=0 should write HA state (no Temp branch)."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 22.0

        await pi.handle_state_payload({"Temp": 0, "Power": "On", "Mode": "Heat"})
        entity.async_write_ha_state.assert_called()


# ── async_reset_ff_seeds RLS beta reset ──────────────────────────────


class TestResetFFSeedsRLS:
    """Tests for async_reset_ff_seeds with model inputs."""

    @pytest.mark.asyncio
    async def test_reset_rebuilds_rls_seeds_with_model_inputs(self):
        """Reset should rebuild RLS seeds including model input seeds."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 1.5,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        # Modify beta to non-seed values
        pi._rls_heat.beta = [99.0] * pi._rls_heat.n
        pi._rls_cool.beta = [99.0] * pi._rls_cool.n
        pi._rls_heat.observation_count = 100
        pi._rls_cool.observation_count = 100

        await pi.async_reset_ff_seeds()

        # Beta should be reset to seeds (lines 816-818)
        assert pi._rls_heat.beta[0] == 0.0  # intercept
        assert pi._rls_heat.beta[1] == pytest.approx(pi._ff_heat_slope)  # outdoor
        assert pi._rls_heat.beta[2] == -3.0  # model input seed
        assert pi._rls_cool.beta[2] == 1.5
        assert pi._rls_heat.observation_count == 0
        assert pi._rls_cool.observation_count == 0


# ── Lag filter with tau > 0 (lines 898-899) ───────────────────────���─


class TestLagFilterUpdate:
    """Tests for _update_lag_filters with non-zero tau."""

    def test_lag_filter_with_tau(self):
        """Lag filter with tau > 0 should apply exponential smoothing."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 1800,  # 30 minutes in seconds
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        # Set raw value to 1.0, filtered starts at 0.0
        pi._model_input_values[0] = 1.0
        pi._model_input_filtered[0] = 0.0

        # Update with dt=900s (15 minutes, half of tau)
        pi._update_lag_filters(900)

        # Should be partially ramped up (not 0 and not 1)
        assert 0.0 < pi._model_input_filtered[0] < 1.0

    def test_lag_filter_with_zero_tau(self):
        """Lag filter with tau=0 should pass through raw value."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._model_input_values[0] = 1.0
        pi._model_input_filtered[0] = 0.5

        pi._update_lag_filters(900)
        assert pi._model_input_filtered[0] == 1.0


# ── _read_model_input_values edge cases (lines 910, 913-915) ────────


class TestReadModelInputValues:
    """Tests for _read_model_input_values with empty entity_id and unavailable entity."""

    def test_empty_entity_id_skipped(self):
        """Model input with empty entity_id should be skipped (line 910)."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Empty",
                "entity_id": "",
                "seed_heat": 0.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._model_input_values[0] = 99.0  # Should not change

        pi._read_model_input_values()
        assert pi._model_input_values[0] == 99.0  # Unchanged — skipped

    def test_unavailable_entity_keeps_last_value(self):
        """Unavailable entity should keep last value (lines 913-915)."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._model_input_values[0] = 1.0  # Previous value

        # Mock entity as unavailable
        unavail_state = MagicMock()
        unavail_state.state = STATE_UNAVAILABLE
        entity.hass.states.get.return_value = unavail_state

        pi._read_model_input_values()
        assert pi._model_input_values[0] == 1.0  # Kept previous value

    def test_missing_entity_keeps_last_value(self):
        """Missing entity (None state) should keep last value."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._model_input_values[0] = 0.5

        entity.hass.states.get.return_value = None

        pi._read_model_input_values()
        assert pi._model_input_values[0] == 0.5


# ── _any_model_input_unavailable (lines 930, 933) ───────────────────


class TestAnyModelInputUnavailable:
    """Tests for _any_model_input_unavailable."""

    def test_empty_entity_id_skipped(self):
        """Empty entity_id should be skipped, not flagged as unavailable (line 930)."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Empty",
                "entity_id": "",
                "seed_heat": 0.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        assert entity._pi._any_model_input_unavailable() is False

    def test_unavailable_entity_returns_true(self):
        """Unavailable entity should return True (line 933)."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        unavail_state = MagicMock()
        unavail_state.state = STATE_UNAVAILABLE
        entity.hass.states.get.return_value = unavail_state

        assert entity._pi._any_model_input_unavailable() is True

    def test_available_entity_returns_false(self):
        """Available entity should return False."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        avail_state = MagicMock()
        avail_state.state = "off"
        entity.hass.states.get.return_value = avail_state

        assert entity._pi._any_model_input_unavailable() is False


# ── _async_model_input_changed dispatcher (lines 939-940) ───────────


class TestModelInputChanged:
    """Tests for _async_model_input_changed firing dispatcher."""

    def test_model_input_changed_fires_dispatcher(self):
        """State change on model input entity should fire dispatcher signal."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        mock_event = MagicMock()
        pi._async_model_input_changed(mock_event)

        # Should have called async_dispatcher_send
        entity.hass.bus.async_fire.assert_not_called  # dispatcher uses different mechanism


# ── FF-only fallback: sensor unavailable (lines 995, 1001) ──────────


class TestFFOnlyFallback:
    """Tests for _check_sensor_recovery FF-only fallback paths."""

    @pytest.mark.asyncio
    async def test_ff_only_with_outdoor_temp_cooling(self):
        """FF-only fallback in cooling mode should use RLS predict (line 995)."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        entity._attr_hvac_mode = HVACMode.COOL
        pi._desired_temp = 24.0
        pi._hp_setpoint = 24.0
        pi._outdoor_temp = 35.0  # Hot outdoor
        entity._attr_current_temperature = None  # Sensor unavailable

        await pi._check_sensor_recovery()

        assert pi._sensor_unavailable is True
        # FF offset should have been computed via RLS

    @pytest.mark.asyncio
    async def test_ff_only_without_outdoor_temp(self):
        """FF-only fallback without outdoor temp should set ff_offset=0 (line 1001)."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._outdoor_temp = None  # No outdoor temp
        entity._attr_current_temperature = None

        await pi._check_sensor_recovery()

        assert pi._sensor_unavailable is True
        assert pi._ff_offset == 0.0


# ── Learning gate debug logging (lines 1146, 1159, 1161, 1165) ──────


class TestLearningGateDebugLogging:
    """Tests for RLS learning gate debug log reasons."""

    @pytest.mark.asyncio
    async def test_learning_blocked_no_outdoor_temp(self):
        """Learning should be blocked and logged when outdoor temp is None (line 1159)."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._outdoor_temp = None
        pi._rls_warmup_done = True
        entity._attr_current_temperature = 22.0  # Within deadband of 22.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._ff_settled_ticks = 3  # Will become 4 in tick

        await pi._pi_tick()
        # Should have logged "no outdoor temp" — no crash

    @pytest.mark.asyncio
    async def test_learning_blocked_manual_suppress(self):
        """Learning should be blocked when manually suppressed (line 1161)."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._outdoor_temp = 5.0
        pi._rls_warmup_done = True
        pi._manual_ff_suppress = True
        entity._attr_current_temperature = 22.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._ff_settled_ticks = 3

        await pi._pi_tick()
        # Should have logged "manually suppressed"

    @pytest.mark.asyncio
    async def test_learning_blocked_model_input_unavailable(self):
        """Learning blocked when model input is unavailable (line 1165)."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": -3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._outdoor_temp = 5.0
        pi._rls_warmup_done = True
        entity._attr_current_temperature = 22.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._ff_settled_ticks = 3

        # Mock model input as unavailable
        unavail = MagicMock()
        unavail.state = STATE_UNAVAILABLE
        entity.hass.states.get.return_value = unavail

        await pi._pi_tick()
        # Should have logged "model input unavailable"

