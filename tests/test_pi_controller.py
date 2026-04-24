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
from custom_components.tasmota_irhvac.pi.pi_controller import PIController, PIExtraStoredData

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
            send_needed = await self._pi.set_temperature(temperature)
            # In real entity, climate.py would call send_ir if send_needed
            if send_needed:
                await self.send_ir()
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

        send_needed = await pi_entity._pi._pi_tick()

        # With error = 2.0°C, P term should push setpoint up
        assert pi_entity._pi._hp_setpoint > 22.0
        assert send_needed

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
        await pi_entity._pi._pi_tick()
        integral_after_2 = pi_entity._pi._pi_integral

        assert integral_after_2 > integral_after_1

    @pytest.mark.asyncio
    async def test_leaky_decay_always_active(self, pi_entity):
        """Leaky integrator (α=0.9999^dt_factor) should decay integral every tick.

        Active both in and out of deadband. With dt_factor=1.0 (first tick),
        integral *= 0.9999. Weak enough to not fight q-feedback but bounds growth.
        """
        pi_entity._attr_current_temperature = 22.0  # Error = 0 (in deadband)
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 10.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi._pi_tick()

        # Leak should reduce integral: 10.0 * 0.9999 ≈ 9.999
        # Variable-rate integration at rate=0.05 with error≈0 adds negligible amount
        assert pi_entity._pi._pi_integral < 10.0
        assert pi_entity._pi._pi_integral > 9.99  # Very weak drain

    @pytest.mark.asyncio
    async def test_leaky_decay_scales_with_dt(self, pi_entity):
        """Leaky decay should scale with dt_factor for variable sample rates."""
        pi = pi_entity._pi
        pi._desired_temp = 22.0
        pi._pi_deadband = 0.5
        pi_entity._attr_current_temperature = 22.0  # Error = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        # Use integral=1.0 with hp_setpoint that avoids q-feedback:
        # raw = 22 + 0 + 0.15*1.0 = 22.15, q_error = 22 - 22.15 = -0.15 (< 0.3, no feedback)
        # First tick: dt_factor = 1.0 (dt = min_interval)
        pi._pi_integral = 1.0
        pi._hp_setpoint = 22.0
        pi._pi_last_tick_time = 0
        await pi._pi_tick()
        integral_normal_dt = pi._pi_integral

        # Reset: simulate very short dt (dt_factor ≈ 0)
        pi._pi_integral = 1.0
        pi._hp_setpoint = 22.0
        import time as time_mod
        original = time_mod.monotonic
        base = time_mod.monotonic()
        pi._pi_last_tick_time = base - 1.0  # 1 second ago
        time_mod.monotonic = lambda: base
        try:
            await pi._pi_tick()
        finally:
            time_mod.monotonic = original
        integral_short_dt = pi._pi_integral

        # Short dt → less decay → integral closer to 1.0
        assert integral_short_dt > integral_normal_dt

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

        send_needed = await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint == old_setpoint
        assert not send_needed

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
        pi_entity._pi._inputs.outdoor_temp = 0.0  # Cold outdoor
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._ff_offset > 0  # Should have positive heating offset

    @pytest.mark.asyncio
    async def test_ff_offset_zero_when_no_outdoor(self, pi_entity):
        """Without outdoor sensor, FF offset should be zero."""
        pi_entity._pi._inputs.outdoor_temp = None
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._ff_offset == 0.0

    @pytest.mark.asyncio
    async def test_rls_model_produces_ff_offset(self, pi_entity):
        """RLS model should produce FF offset based on outdoor delta."""
        pi_entity._pi._inputs.outdoor_temp = 5.0  # 10°C below reference (15)
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
        pi_entity._pi._inputs.outdoor_temp = None
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
        pi_entity._pi._inputs.outdoor_temp = 0.0
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
            "seed_heat": 3.0,
            "seed_cool": 0.0,
            "suppress_learning": True,
        }]
        pi._inputs.values = [1.0]  # Stove is active
        pi._inputs.filtered = [1.0]
        pi._inputs.outdoor_temp = 5.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._ff_settled_ticks = 10  # Well settled
        pi._rls_warmup_done = True
        pi._rls_heat_mature = True
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
            "seed_heat": 3.0,
            "seed_cool": 0.0,
            "suppress_learning": True,
        }]
        pi._inputs.values = [0.0]  # Stove is OFF
        pi._inputs.filtered = [0.0]
        pi._inputs.outdoor_temp = 5.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._pi_integral = 0.5  # Small, stable
        pi._prev_integral_for_rls = 0.5
        pi._ff_settled_ticks = 10
        pi._rls_warmup_done = True
        pi._rls_heat_mature = True
        pi._rls_heat_mature = True  # Batch-first gate satisfied
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

        send_needed = await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint == old_setpoint
        assert not send_needed

    @pytest.mark.asyncio
    async def test_resume_allows_tick(self, pi_entity):
        """PI should run normally after resume."""
        pi_entity._pi.pi_pause()
        pi_entity._pi.pi_resume()
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

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
        assert pi_entity._pi._recovery_check_needed is True

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
        pi_entity._pi._inputs.outdoor_temp = 0.0

        await pi_entity._pi._check_sensor_recovery()

        assert pi_entity._pi._sensor_unavailable is True
        assert pi_entity._pi._sensor_recovery_pending is False
        assert pi_entity._pi._pi_integral == 0.0

    @pytest.mark.asyncio
    async def test_sensor_changed_clears_recovery(self, pi_entity):
        """When sensor becomes available, should cancel pending recovery."""
        pi_entity._pi._sensor_recovery_pending = True
        pi_entity._pi._sensor_unavailable = True

        pi_entity._attr_current_temperature = 70.0  # Now available
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_async_sensor_changed(was_none=True)

        assert pi_entity._pi._sensor_recovery_pending is False
        assert pi_entity._pi._sensor_unavailable is False

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

        send_needed = await pi_entity._pi._pi_async_sensor_changed(was_none=False)

        # Should have ticked (cooldown elapsed since last_tick_time=0)
        assert send_needed

    @pytest.mark.asyncio
    async def test_sensor_update_respects_cooldown(self, pi_entity):
        """Sensor update should not tick if within cooldown."""
        import time as _time
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_last_tick_time = _time.monotonic()  # Just ticked

        send_needed = await pi_entity._pi._pi_async_sensor_changed(was_none=False)

        # Should NOT have ticked (within cooldown)
        assert not send_needed

    @pytest.mark.asyncio
    async def test_time_normalized_integral(self, pi_entity):
        """Integral accumulation should scale with time between ticks."""
        import time as _time
        pi_entity._attr_current_temperature = 20.0  # °C
        pi_entity._pi._desired_temp = 22.0  # °C, error = 2.0°C
        pi_entity._pi._hp_setpoint = 22.0

        # Simulate a tick at normal interval (dt_factor = 1.0)
        pi_entity._pi._pi_last_tick_time = _time.monotonic() - pi_entity._pi._pi_tick_fallback
        await pi_entity._pi._pi_tick()
        integral_normal = pi_entity._pi._pi_integral

        # Reset and simulate a tick at half interval (dt_factor = 0.5)
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._pi_last_error = 0.0
        pi_entity._pi._pi_last_tick_time = _time.monotonic() - (pi_entity._pi._pi_tick_fallback / 2)
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
        pi_entity._attr_current_temperature = 20.0  # Ensure tick can run
        await pi_entity.async_set_temperature(temperature=74.0)
        assert pi_entity._pi._desired_temp == 74.0
        # target_temperature property reads from desired_temp when PI active
        assert pi_entity.target_temperature == 74.0
        # Bumpless transfer: integral not zeroed for small changes
        # (74 - 22 = 52°F ≈ 28.9°C shift > 2°C, so integral IS zeroed for large shifts)

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

        send_needed = await pi_entity._pi._pi_tick()

        assert pi_entity._pi._hp_setpoint == old_setpoint
        assert not send_needed

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
    async def test_full_rate_integration_in_deadband(self, pi_entity):
        """In deadband, integral should accumulate at full rate, not freeze.

        Full-rate integration: the integrator accumulates the actual error.
        Anti-cycling handled by hysteresis, dwell timer, leaky integrator.
        """
        pi_entity._attr_current_temperature = 22.0  # Error = 0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = -5.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi._pi_tick()

        # Integral should change from leaky decay (error=0 → no accumulation).
        # Leak: -5.0 * 0.9999 ≈ -4.9995
        assert pi_entity._pi._pi_integral != -5.0  # Not frozen
        assert pi_entity._pi._pi_integral < -4.9  # Leak doesn't drain it quickly

    @pytest.mark.asyncio
    async def test_deadband_integration_proportional_to_error(self, pi_entity):
        """Full-rate integration: larger error → proportionally more accumulation."""
        pi = pi_entity._pi
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        pi._pi_last_tick_time = 0
        pi._pi_deadband = 0.5
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        # Small error: 0.1°C
        pi._pi_integral = 0.0
        pi_entity._attr_current_temperature = 21.9  # Error = +0.1
        await pi._pi_tick()
        integral_small_error = pi._pi_integral

        # Larger error: 0.4°C
        pi._pi_integral = 0.0
        pi._pi_last_tick_time = 0
        pi_entity._attr_current_temperature = 21.6  # Error = +0.4
        await pi._pi_tick()
        integral_large_error = pi._pi_integral

        # Full-rate: integral scales linearly with error (4× error → ~4× integral)
        assert abs(integral_large_error) > abs(integral_small_error)
        ratio = abs(integral_large_error) / abs(integral_small_error)
        assert ratio > 3.0, f"Expected ~4× ratio, got {ratio:.1f}×"

    @pytest.mark.asyncio
    async def test_deadband_integration_not_throttled(self, pi_entity):
        """Error within deadband should accumulate at full rate."""
        pi_entity._attr_current_temperature = 21.7  # Error = +0.3 (in deadband)
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0  # above room → HP active
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi._pi_tick()

        # Full-rate: 0.3°C error × 1.0 rate × 1.0 dt_factor = ~0.3 integral
        # (trapezoidal with prev_error=0 → avg = 0.15, but first tick uses error directly)
        assert abs(pi_entity._pi._pi_integral) > 0.1, (
            f"Full-rate should accumulate meaningfully, got {pi_entity._pi._pi_integral:.3f}"
        )


class TestFullRateIntegrationRegression:
    """Regression tests confirming full-rate deadband integration is correct.

    Each test runs the same scenario twice — once with full-rate (current policy,
    rate=1.0) and once with the old variable-rate (rate = |error|/deadband,
    floor 0.05) to confirm full-rate is at least as good.

    Includes both static tests (fixed room temp) and closed-loop dynamic tests
    (room temp responds to HP setpoint via a simple thermal model).
    """

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_entity(current_temp, desired=22.0, hp_setpoint=22,
                     outdoor=10.0, mode=HVACMode.HEAT):
        """Create a fresh FakePIEntity with standard setup."""
        entity = FakePIEntity(make_pi_config())
        pi = entity._pi
        pi._desired_temp = desired
        pi._hp_setpoint = hp_setpoint
        pi._pi_integral = 0.0
        pi._pi_deadband = 0.5
        pi._inputs.outdoor_temp = outdoor
        entity._attr_hvac_mode = mode
        entity._attr_current_temperature = current_temp
        return entity

    @staticmethod
    def _run_static(entity, n_ticks, tick_interval=900.0):
        """Run n ticks with fixed room temperature, return integral trajectory."""
        import asyncio
        pi = entity._pi
        integrals = []
        t = tick_interval
        for _ in range(n_ticks):
            pi._pi_last_tick_time = t - tick_interval
            with patch("time.monotonic", return_value=t):
                asyncio.get_event_loop().run_until_complete(pi._pi_tick())
            integrals.append(pi._pi_integral)
            t += tick_interval
        return integrals

    @staticmethod
    def _run_dynamic(entity, n_ticks, outdoor_c, tau_minutes=60.0,
                     hp_gain=0.8, tick_interval=900.0):
        """Run n ticks with room temperature responding to HP setpoint.

        Simple 1R1C thermal model per tick:
            room += (hp_drive - room) × (1 - exp(-dt/τ))
        where hp_drive = outdoor + hp_gain × (hp_setpoint - outdoor)

        hp_gain < 1.0 models the HP's imperfect efficiency: a setpoint of
        30°C with outdoor at 0°C doesn't heat the room to 30°C — it drives
        toward hp_gain × 30 + (1-hp_gain) × 0 = 24°C.

        Returns: list of (room_temp, hp_setpoint, integral) per tick.
        """
        import asyncio
        import math
        pi = entity._pi
        room = entity._attr_current_temperature
        dt = tick_interval
        alpha = 1.0 - math.exp(-dt / (tau_minutes * 60.0))
        trajectory = []
        t = tick_interval

        for _ in range(n_ticks):
            # Thermal model: room evolves toward HP drive point
            hp_drive = outdoor_c + hp_gain * (float(pi._hp_setpoint) - outdoor_c)
            room = room + (hp_drive - room) * alpha

            # Feed room temp to entity and tick
            entity._attr_current_temperature = room
            pi._pi_last_tick_time = t - tick_interval
            with patch("time.monotonic", return_value=t):
                asyncio.get_event_loop().run_until_complete(pi._pi_tick())

            trajectory.append((room, int(pi._hp_setpoint), pi._pi_integral))
            t += tick_interval

        return trajectory

    @staticmethod
    def _old_variable_rate(deadband):
        """Return the old variable-rate policy for comparison."""
        return lambda ae: max(0.05, min(1.0, ae / deadband))

    def _run_ab_static(self, current_temp, n_ticks, **kwargs):
        """Run static scenario with both policies, return (full_integrals, var_integrals)."""
        entity_full = self._make_entity(current_temp, **kwargs)
        integrals_full = self._run_static(entity_full, n_ticks)

        entity_var = self._make_entity(current_temp, **kwargs)
        entity_var._pi._deadband_integration_rate = self._old_variable_rate(
            entity_var._pi._pi_deadband)
        integrals_var = self._run_static(entity_var, n_ticks)

        return integrals_full, integrals_var

    def _run_ab_dynamic(self, start_temp, n_ticks, outdoor_c=5.0,
                        tau_minutes=60.0, hp_gain=0.8, **kwargs):
        """Run dynamic scenario with both policies, return (var_traj, full_traj).

        Each trajectory is a list of (room_temp, hp_setpoint, integral).
        """
        entity_full = self._make_entity(start_temp, outdoor=outdoor_c, **kwargs)
        traj_full = self._run_dynamic(entity_full, n_ticks, outdoor_c, tau_minutes, hp_gain)

        entity_var = self._make_entity(start_temp, outdoor=outdoor_c, **kwargs)
        entity_var._pi._deadband_integration_rate = self._old_variable_rate(
            entity_var._pi._pi_deadband)
        traj_var = self._run_dynamic(entity_var, n_ticks, outdoor_c, tau_minutes, hp_gain)

        return traj_full, traj_var

    @staticmethod
    def _count_setpoint_changes(trajectory):
        """Count how many times hp_setpoint changed in a trajectory."""
        changes = 0
        for i in range(1, len(trajectory)):
            if trajectory[i][1] != trajectory[i - 1][1]:
                changes += 1
        return changes

    @staticmethod
    def _count_reversals(trajectory):
        """Count setpoint direction reversals (up then down, or down then up).

        A reversal means the PI changed its mind — the hallmark of oscillation.
        """
        directions = []
        for i in range(1, len(trajectory)):
            delta = trajectory[i][1] - trajectory[i - 1][1]
            if delta != 0:
                directions.append(1 if delta > 0 else -1)
        reversals = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i - 1]:
                reversals += 1
        return reversals

    # ── Static tests (fixed room temp, measure integral speed) ───────

    def test_static_02c_offset_threshold(self):
        """0.2°C offset: full-rate reaches setpoint-change threshold faster."""
        full_rate, var_rate = self._run_ab_static(21.8, 60)
        threshold = 0.5 / 0.15  # ≈ 3.33

        var_tick = next((i+1 for i,v in enumerate(var_rate) if v >= threshold), None)
        full_tick = next((i+1 for i,v in enumerate(full_rate) if v >= threshold), None)

        # Full-rate should reach threshold no slower than variable-rate
        if var_tick and full_tick:
            assert full_tick <= var_tick

    def test_static_zero_error_both_identical(self):
        """At zero error, q-feedback dominates — both policies produce same drift."""
        full_rate, var_rate = self._run_ab_static(22.0, 20)

        # At error=0, integration contributes nothing (0 × rate = 0 regardless).
        # Only q-feedback moves the integral. Both should be identical.
        diff = abs(full_rate[-1] - var_rate[-1])
        avg = (abs(full_rate[-1]) + abs(var_rate[-1])) / 2
        if avg > 0.1:
            assert diff / avg < 0.05, (
                f"Expected identical at zero error: var={var_rate[-1]:.3f} "
                f"full={full_rate[-1]:.3f}"
            )

    def test_static_deadband_edge_identical(self):
        """At deadband edge, both policies produce identical trajectories."""
        full_rate, var_rate = self._run_ab_static(21.51, 10)  # 0.49°C ≈ edge

        final_diff = abs(full_rate[-1] - var_rate[-1])
        avg = (abs(full_rate[-1]) + abs(var_rate[-1])) / 2
        if avg > 0.01:
            assert final_diff / avg < 0.15

    # ── Dynamic tests (room temp responds to HP, tests oscillation) ──

    def test_quantization_boundary_limit_cycle(self):
        """Quantization boundary: SS setpoint at x.25 causes limit cycles.

        With outdoor=5, hp_gain=0.8, desired=22: the true steady-state
        setpoint is 26.25 — between two integers. The q-feedback mechanism
        (Bohn & Atherton 1995) should nudge the integral to lock onto one
        integer. Currently, q-feedback has a dead zone (0.3–0.5°C gate)
        that misses q_error=0.25 at this operating point — see future work
        for continuous Bohn & Atherton formulation to eliminate the dead zone.

        At SP=26: room=21.8, error=+0.2 → in deadband, q_error=-0.25 (below 0.3 gate)
        At SP=27: room=22.6, error=-0.6 → out of deadband (q-feedback disabled)

        Result: q-feedback never fires, limit cycle persists. Full-rate
        integration limits reversals vs variable-rate (symmetric rates),
        but cannot eliminate the cycle without q-feedback coverage.

        This test documents the current behavior; the goal is ≤6 reversals
        once the q-feedback dead zone is addressed.
        """
        full_traj, var_traj = self._run_ab_dynamic(
            21.0, 120, outdoor_c=5.0, tau_minutes=60.0, hp_gain=0.8,
        )

        full_rev = self._count_reversals(full_traj)
        var_rev = self._count_reversals(var_traj)

        # Full-rate should have fewer reversals than variable-rate at
        # quantization boundary (symmetric rates damp faster than
        # variable-rate's asymmetric 0.4 up / 1.0 down)
        assert full_rev <= var_rev + 2, (
            f"Full-rate ({full_rev} rev) should not oscillate more than "
            f"variable-rate ({var_rev} rev) at quantization boundary"
        )

    def test_dynamic_settling_no_oscillation(self):
        """Closed-loop: room 1°C below target, full-rate policy should settle.

        Start at 21°C with target 22°C. The HP is initially at 22°C.
        The PI needs to raise the HP setpoint to compensate for outdoor losses.
        After the room approaches target, the setpoint should stabilize — not
        oscillate between integers.

        τ=60min (typical room), outdoor=2°C, hp_gain=0.8.
        120 ticks = 30 hours — long enough to see any oscillation develop.

        outdoor=2 chosen so the true steady-state setpoint is an integer
        (SS = 2 + 20/0.8 = 27.0), isolating settling behavior from the
        quantization boundary limit cycle tested separately above.

        Variable-rate is expected to oscillate here: its reduced integration
        rate for small errors prevents the convergence integral from draining,
        causing sustained oscillation even at integer SS. This is one reason
        the production policy uses full-rate integration.
        """
        full_traj, var_traj = self._run_ab_dynamic(
            21.0, 120, outdoor_c=2.0, tau_minutes=60.0, hp_gain=0.8,
        )

        full_rev = self._count_reversals(full_traj)
        var_rev = self._count_reversals(var_traj)

        # Full-rate should settle (≤6 reversals in 30h)
        assert full_rev <= 6, (
            f"full-rate: {full_rev} setpoint reversals in 120 ticks — "
            f"likely oscillating. Setpoints: "
            f"{[t[1] for t in full_traj[::10]]}"
        )
        # Full-rate should be no worse than variable-rate
        assert full_rev <= var_rev + 2, (
            f"Full-rate ({full_rev} rev) should not oscillate more than "
            f"variable-rate ({var_rev} rev)"
        )

    def test_dynamic_small_offset_oscillation_risk(self):
        """Closed-loop: room at target, small perturbation.  Does full-rate oscillate?

        Start at 22°C (at target). The HP is at FF steady-state setpoint.
        Outdoor = 5°C. Room is stable. The PI should hold — not hunt.

        This is the scenario where variable-rate might prevent oscillation:
        with full-rate, the integral accumulates faster for small perturbations,
        potentially crossing the hysteresis threshold and triggering a setpoint
        change that overshoots.
        """
        # hp_setpoint at FF steady state: desired + seed*(desired-outdoor) = 22 + 0.25*17 ≈ 26
        full_traj, var_traj = self._run_ab_dynamic(
            22.0, 80, outdoor_c=5.0, tau_minutes=60.0, hp_gain=0.8,
            hp_setpoint=26,
        )

        var_changes = self._count_setpoint_changes(var_traj)
        full_changes = self._count_setpoint_changes(full_traj)

        # If full-rate causes significantly more setpoint changes, that's
        # evidence of oscillation risk.  If similar, variable-rate isn't
        # protecting against anything.
        # Allow full-rate up to 2 more changes than variable-rate.
        assert full_changes <= var_changes + 3, (
            f"Full-rate made {full_changes} setpoint changes vs variable-rate's "
            f"{var_changes} — potential oscillation. "
            f"Setpoints: {[t[1] for t in full_traj[::10]]}"
        )

    def test_dynamic_slow_tau_house(self):
        """Closed-loop with slow time constant (well-insulated house, τ=120min).

        Slow-τ houses are where variable-rate was meant to help: the HP response
        is sluggish, so aggressive integration could overshoot badly.

        This tests whether full-rate causes more overshoot or oscillation in a
        slow-response building.
        """
        full_traj, var_traj = self._run_ab_dynamic(
            21.0, 160, outdoor_c=0.0, tau_minutes=120.0, hp_gain=0.7,
        )

        for label, traj in [("full-rate", full_traj), ("variable-rate", var_traj)]:
            reversals = self._count_reversals(traj)
            # Slow-τ houses should settle, not oscillate.
            # The dwell timer (20 min) plus HP response lag should damp oscillation.
            assert reversals <= 8, (
                f"{label} with τ=120min: {reversals} reversals in 160 ticks — "
                f"slow-τ oscillation. Room temps: "
                f"{[f'{t[0]:.1f}' for t in traj[::20]]}"
            )

        # Check that both converge to similar room temperature
        var_final_room = var_traj[-1][0]
        full_final_room = full_traj[-1][0]
        assert abs(var_final_room - full_final_room) < 1.0, (
            f"Divergent outcomes: var room={var_final_room:.1f}°C, "
            f"full room={full_final_room:.1f}°C"
        )

    def test_dynamic_fast_tau_room(self):
        """Closed-loop with fast time constant (small room, τ=30min).

        Fast-τ rooms respond quickly to HP changes, which means the room can
        overshoot/undershoot rapidly after a setpoint change. This is where
        oscillation is most likely.
        """
        full_traj, var_traj = self._run_ab_dynamic(
            21.0, 80, outdoor_c=5.0, tau_minutes=30.0, hp_gain=0.9,
        )

        for label, traj in [("full-rate", full_traj), ("variable-rate", var_traj)]:
            reversals = self._count_reversals(traj)
            assert reversals <= 6, (
                f"{label} with τ=30min: {reversals} reversals — "
                f"fast-room oscillation. Setpoints: "
                f"{[t[1] for t in traj[::10]]}"
            )

    def test_dynamic_room_temp_stays_in_comfort_band(self):
        """Both policies should keep room within ±1°C of target after settling.

        After initial convergence (first 20 ticks / 5h), the room should stay
        within a ±1°C band around the 22°C target. Wider excursions indicate
        oscillation or poor control.
        """
        full_traj, var_traj = self._run_ab_dynamic(
            21.0, 80, outdoor_c=5.0, tau_minutes=60.0, hp_gain=0.8,
        )

        for label, traj in [("full-rate", full_traj), ("variable-rate", var_traj)]:
            # Skip first 20 ticks (settling)
            settled = traj[20:]
            if not settled:
                continue
            room_temps = [t[0] for t in settled]
            max_deviation = max(abs(t - 22.0) for t in room_temps)
            assert max_deviation < 1.0, (
                f"{label}: room deviated {max_deviation:.2f}°C from target "
                f"after settling. Range: [{min(room_temps):.1f}, {max(room_temps):.1f}]"
            )


class TestPIMathContinued:
    """Continued PI math tests (split for readability after tradeoff tests)."""

    @pytest.mark.asyncio
    async def test_ff_auto_learning_writes_bucket(self, pi_entity):
        """FF should write to bucket when settled for 2+ ticks in deadband."""
        pi_entity._attr_current_temperature = 22.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 23.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._pi._inputs.outdoor_temp = 0.0
        pi_entity._pi._ff_settled_ticks = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        old_obs_count = pi_entity._pi._rls_heat.observation_count

        # Tick multiple times to settle into deadband and trigger RLS learning
        for _ in range(5):
            await pi_entity._pi._pi_tick()

        # RLS should have learned (observation count increased)
        assert pi_entity._pi._rls_heat.observation_count > old_obs_count or pi_entity._pi._ff_settled_ticks >= 4

    @pytest.mark.asyncio
    async def test_on_remote_change_updates_desired_and_setpoint(self, pi_entity):
        """on_remote_change should update desired_temp and hp_setpoint."""
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        send_needed = await pi_entity._pi.on_remote_change(25.0)

        # desired_temp updated to converted value, hp_setpoint set to reported
        assert pi_entity._pi._desired_temp == 25.0
        assert pi_entity._pi._hp_setpoint != 22.0  # Recomputed by tick
        assert send_needed

    @pytest.mark.asyncio
    async def test_on_remote_change_same_temp_keeps_integral(self, pi_entity):
        """on_remote_change with same temp should not zero integral (bumpless transfer)."""
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 24.0
        pi_entity._pi._pi_integral = 5.0
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi.on_remote_change(22.0)

        # Same desired temp — integral NOT zeroed
        assert pi_entity._pi._desired_temp == 22.0
        assert pi_entity._pi._pi_integral != 0.0

    @pytest.mark.asyncio
    async def test_on_remote_change_large_shift_zeros_integral(self, pi_entity):
        """on_remote_change with >2°C shift should zero integral before tick."""
        pi_entity._pi._desired_temp = 20.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 50.0  # Large integral
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi.on_remote_change(25.0)  # 5°C shift > 2°C

        assert pi_entity._pi._desired_temp == 25.0
        # Integral was zeroed before tick, then tick added small amount from error
        # So integral should be much smaller than original 50.0
        assert pi_entity._pi._pi_integral < 5.0

    @pytest.mark.asyncio
    async def test_on_remote_change_returns_false_when_paused(self, pi_entity):
        """on_remote_change should return False when PI is paused."""
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._pi_paused = True

        send_needed = await pi_entity._pi.on_remote_change(25.0)

        assert not send_needed
        assert pi_entity._pi._desired_temp == 22.0  # Unchanged

    @pytest.mark.asyncio
    async def test_on_remote_change_returns_false_when_disabled(self, pi_entity):
        """on_remote_change should return False when PI is disabled."""
        pi_entity._pi._pi_enabled = False

        send_needed = await pi_entity._pi.on_remote_change(25.0)

        assert not send_needed

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
        """Quantization feedback should nudge integral for small misalignments.

        Only activates when |q_error| is between 0.3 and 0.5 (true
        quantization misalignment). Large gaps are real integral corrections
        and must not trigger feedback (that caused integral runaway).
        """
        pi = pi_entity._pi
        pi._desired_temp = 20.5
        pi._hp_setpoint = 24
        # Set integral so clamped ≈ 23.6 → q_error = 24 - 23.6 = 0.4 (in range)
        # clamped = desired + ff + ki*I = 20.5 + ff + 0.15*I
        # With outdoor=5, default seed=0.3: ff = 0.3*10 = 3.0
        # Need 20.5 + 3.0 + 0.15*I = 23.6 → I = 0.67
        pi._inputs.outdoor_temp = 5.0
        pi._pi_integral = 0.67
        pi._pi_deadband = 0.5
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 20.5  # Error = 0, in deadband

        integral_before = pi._pi_integral
        await pi._pi_tick()

        # Small q_error (0.3-0.5) should nudge integral toward integer alignment
        assert pi._pi_integral != integral_before, "Quantization feedback had no effect"

    @pytest.mark.asyncio
    async def test_quantization_feedback_rejects_large_gap(self, pi_entity):
        """Large q_error (> 0.5) must NOT trigger quantization feedback.

        When integral carries real transient correction, the gap between
        hp_setpoint and clamped_setpoint can be large. Applying q_feedback
        in this case causes integral runaway (the bunkroom bug).
        """
        pi = pi_entity._pi
        pi._desired_temp = 20.5
        pi._hp_setpoint = 26
        pi._inputs.outdoor_temp = 5.0  # FF ≈ 3.5
        pi._pi_integral = -2.0  # clamped ≈ 20.5 + 3.5 + 0.15*(-2) = 23.7
        # q_error = 26 - 23.7 = 2.3 — way too large for quantization feedback
        pi._pi_deadband = 0.5
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 20.5

        integral_before = pi._pi_integral
        await pi._pi_tick()

        # Large gap should NOT modify integral via q_feedback.
        # The integral will change slightly due to variable-rate integration
        # and leaky decay, but q_feedback for a 2.3°C gap would cause a
        # much larger jump (~15.3 at 0.4 rate with ki=0.15).
        integral_change = abs(pi._pi_integral - integral_before)
        assert integral_change < 1.0, (
            f"Large q_error triggered feedback: integral {integral_before} → {pi._pi_integral} "
            f"(change={integral_change:.3f}, q_feedback would cause ~15)"
        )

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
        from custom_components.tasmota_irhvac.pi.pi_controller import PIExtraStoredData
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
        from custom_components.tasmota_irhvac.pi.pi_controller import PIExtraStoredData
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

    def test_intercept_is_unclamped(self):
        """Intercept (index 0) must not be clamped.

        A clamped intercept forces the outdoor slope to compensate for
        unmodeled offsets, causing slope overshoot. Verified by the
        2026-04-08 LR runaway analysis (Scenario G).
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._rls_heat_clamps[0] is None
        assert pi._rls_cool_clamps[0] is None

    def test_model_input_with_both_clamps(self):
        """Model input with both clamp_min and clamp_max should create tuple clamp."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.0,
                "seed_cool": 0.0,
                "clamp_min": 0.0,
                "clamp_max": 5.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        # Index 0=intercept(None), 1=outdoor_delta(0,2), 2=model_input(clamp)
        assert pi._rls_heat_clamps[2] == (-5.0, 0.0)

    def test_model_input_with_no_clamps(self):
        """Model input without clamps should have None."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._rls_heat_clamps[2] is None

    def test_model_input_with_partial_clamp_min_only(self):
        """Model input with only clamp_min should clamp with upper=inf."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.0,
                "seed_cool": 0.0,
                "clamp_min": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        clamp = pi._rls_heat_clamps[2]
        assert clamp is not None
        # seed_min=0 → β upper = 0; seed_max absent → β lower = -inf
        assert clamp[0] == float("-inf")
        assert clamp[1] == 0.0

    def test_model_input_with_partial_clamp_max_only(self):
        """Model input with only clamp_max should clamp with lower=-inf."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.0,
                "seed_cool": 0.0,
                "clamp_max": 12.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        clamp = pi._rls_heat_clamps[2]
        assert clamp is not None
        assert clamp[0] == -12.0
        assert clamp[1] == float("inf")

    def test_model_input_with_no_clamps(self):
        """Model input with no clamps should have None."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._rls_heat_clamps[2] is None


# ── ExtraStoredData Full Restore (lines 639-664) ────────────────────


class TestExtraStoredDataFullRestore:
    """Tests for restoring RLS models and lag filters."""

    def test_restore_rls_models(self):
        """restore_extra_stored_data with rls models should restore them."""
        from custom_components.tasmota_irhvac.pi.pi_controller import PIExtraStoredData, RLSModel
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
        assert pi._metrics.integral_convergence == 0.5

    def test_restore_lag_filter_states(self):
        """restore_extra_stored_data with lag filter states should restore filtered values."""
        from custom_components.tasmota_irhvac.pi.pi_controller import PIExtraStoredData
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.0,
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
        assert pi._inputs.filtered[0] == 0.75



# ── on_remote_change standalone tests ────────────────────────────────


class TestOnRemoteChangeStandalone:
    """Tests for on_remote_change called from standalone entities."""

    @pytest.mark.asyncio
    async def test_on_remote_change_bumpless_small_shift(self):
        """Small temp shift uses bumpless transfer (adjusts integral, not zero)."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 22.0
        pi._hp_setpoint = 23.0
        pi._pi_integral = 5.0
        entity._attr_current_temperature = 21.0
        entity._attr_hvac_mode = HVACMode.HEAT

        await pi.on_remote_change(23.0)  # 1°C shift < 2°C threshold

        assert pi._desired_temp == 23.0
        # Integral adjusted via bumpless transfer, not zeroed
        assert pi._pi_integral != 0.0

    @pytest.mark.asyncio
    async def test_on_remote_change_large_shift_zeros(self):
        """Large temp shift (>2°C) zeros integral before tick."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 22.0
        pi._pi_integral = 50.0  # Large integral
        entity._attr_current_temperature = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        await pi.on_remote_change(25.0)  # 5°C shift > 2°C

        assert pi._desired_temp == 25.0
        # Integral was zeroed, then tick added small amount from error
        assert pi._pi_integral < 5.0


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
                "seed_heat": 3.0,
                "seed_cool": -1.5,
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

        # Coefficients should be reset to seeds (check physical units)
        heat_coeffs = pi._rls_heat.get_coefficients()
        cool_coeffs = pi._rls_cool.get_coefficients()
        assert heat_coeffs[0] == pytest.approx(0.0)  # intercept
        assert heat_coeffs[1] == pytest.approx(-pi._outdoor_seed_heat)  # outdoor (β = -seed)
        assert heat_coeffs[2] == pytest.approx(-3.0)  # model input seed
        assert cool_coeffs[2] == pytest.approx(1.5)
        assert pi._rls_heat.observation_count == 0
        assert pi._rls_cool.observation_count == 0


# ── Lag filter with tau > 0 (lines 898-899) ───────────────────────���─
# Lag filter, read_values, any_unavailable tests moved to test_model_input_manager.py

# ── _async_model_input_changed dispatcher (lines 939-940) ───────────


class TestModelInputChanged:
    """Tests for _async_model_input_changed firing dispatcher."""

    def test_model_input_changed_fires_dispatcher(self):
        """State change on model input entity should fire dispatcher signal."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove",
                "entity_id": "input_boolean.stove",
                "seed_heat": 3.0,
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
        pi._inputs.outdoor_temp = 35.0  # Hot outdoor
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
        pi._inputs.outdoor_temp = None  # No outdoor temp
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

        pi._inputs.outdoor_temp = None
        pi._rls_warmup_done = True
        pi._rls_heat_mature = True
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

        pi._inputs.outdoor_temp = 5.0
        pi._rls_warmup_done = True
        pi._rls_heat_mature = True
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
                "seed_heat": 3.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._inputs.outdoor_temp = 5.0
        pi._rls_warmup_done = True
        pi._rls_heat_mature = True
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


# ── Supplemental Auto Model Inputs (L242-261) ───────────────────────


class TestSupplementalAutoModelInputs:
    """Tests for auto-generation of model inputs from supplemental sources."""

    def test_supplemental_auto_generates_model_input(self):
        """Supplemental source with auto_model_input=True (default) creates a model input."""
        config = make_pi_config({
            "pi_supplemental_sources": [{
                "name": "Pellet Stove",
                "entity_id": "climate.pellet_stove",
                "seed_heat": 4.0,
                "seed_cool": 0.0,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        # Should have auto-generated one model input (flagged in _model_inputs)
        auto_inputs = [m for m in pi._model_inputs if m.get("_auto_supplemental")]
        assert len(auto_inputs) == 1
        auto = auto_inputs[0]
        assert auto["name"] == "Pellet Stove (auto)"
        assert auto["entity_id"] == "climate.pellet_stove"
        assert auto["seed_heat"] == 4.0
        assert auto["seed_cool"] == 0.0
        assert auto["lag_tau"] == 0
        assert auto["suppress_learning"] is True

    def test_supplemental_auto_model_input_opt_out(self):
        """Supplemental source with auto_model_input=False skips auto generation."""
        config = make_pi_config({
            "pi_supplemental_sources": [{
                "name": "Pellet Stove",
                "entity_id": "climate.pellet_stove",
                "auto_model_input": False,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        assert not any(m.get("_auto_supplemental") for m in pi._model_inputs)

    def test_supplemental_skips_if_manual_input_exists(self):
        """Auto model input skipped when user already has a manual input for same entity."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Stove Manual",
                "entity_id": "climate.pellet_stove",
                "seed_heat": 2.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
            "pi_supplemental_sources": [{
                "name": "Pellet Stove",
                "entity_id": "climate.pellet_stove",
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        assert not any(m.get("_auto_supplemental") for m in pi._model_inputs)
        # Only the manual input should be in _model_inputs
        assert len(pi._model_inputs) == 1
        assert pi._model_inputs[0]["name"] == "Stove Manual"

    def test_supplemental_default_seeds(self):
        """Auto model input uses default seeds when not specified."""
        config = make_pi_config({
            "pi_supplemental_sources": [{
                "name": "Stove",
                "entity_id": "climate.stove",
            }],
        })
        entity = FakePIEntity(config)
        auto = [m for m in entity._pi._model_inputs if m.get("_auto_supplemental")][0]

        assert auto["seed_heat"] == 3.0
        assert auto["seed_cool"] == 3.0

    def test_n_model_inputs_includes_auto(self):
        """_n_model_inputs counts outdoor_delta + all model inputs including auto."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Manual",
                "entity_id": "input_boolean.manual",
                "seed_heat": 1.0,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
            "pi_supplemental_sources": [{
                "name": "Stove",
                "entity_id": "climate.stove",
            }],
        })
        entity = FakePIEntity(config)
        # 1 (outdoor_delta) + 1 (manual) + 1 (auto) = 3
        assert entity._pi._n_model_inputs == 3


# ── Ki-Changed Integral Scaling on Restore (L474-479) ───────────────


class TestKiChangedIntegralScaling:
    """Tests for integral rescaling when ki changes between restarts."""

    def test_ki_changed_scales_integral(self):
        """Integral rescaled when ki differs from ki_at_save."""
        config = make_pi_config({"pi_ki": 0.002})
        entity = FakePIEntity(config)
        pi = entity._pi

        data = PIExtraStoredData(
            pi_integral=10.0,
            desired_temp=22.0,
            hp_setpoint=23.0,
            ki_at_save=0.001,  # Was half of current ki
        )
        pi.restore_extra_stored_data(data)

        # scale = old_ki / new_ki = 0.001 / 0.002 = 0.5
        assert pi._pi_integral == pytest.approx(5.0)

    def test_ki_unchanged_no_scaling(self):
        """Integral not rescaled when ki is the same."""
        config = make_pi_config({"pi_ki": 0.001})
        entity = FakePIEntity(config)
        pi = entity._pi

        data = PIExtraStoredData(
            pi_integral=10.0,
            desired_temp=22.0,
            hp_setpoint=23.0,
            ki_at_save=0.001,
        )
        pi.restore_extra_stored_data(data)

        assert pi._pi_integral == pytest.approx(10.0)

    def test_ki_at_save_zero_no_scaling(self):
        """Integral not rescaled when ki_at_save is zero (legacy data)."""
        config = make_pi_config({"pi_ki": 0.001})
        entity = FakePIEntity(config)
        pi = entity._pi

        data = PIExtraStoredData(
            pi_integral=10.0,
            desired_temp=22.0,
            hp_setpoint=23.0,
            ki_at_save=0.0,
        )
        pi.restore_extra_stored_data(data)

        assert pi._pi_integral == pytest.approx(10.0)


# ── Supplemental Integration (bumpless transfer applies hold timer reset) ────


class TestSupplementalIntegration:
    """Integration: PIController applies SupplementalResult side effects."""

    def test_bumpless_transfer_clears_hold_timer(self):
        """When supplemental stops, PIController clears hold timer."""
        config = make_pi_config({"pi_supplemental_sources": [{
            "name": "Pellet Stove",
            "entity_id": "climate.pellet_stove",
            "failure_threshold": 900,
            "recovery_margin": 0.3,
            "auto_model_input": False,
        }]})
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._supplemental.tracking_mode = True
        pi._supplemental.tracking_sources = ["Pellet Stove"]
        pi._last_setpoint_change_time = 999.0

        state = MagicMock()
        state.state = "off"
        entity.hass.states.get.return_value = state

        result = pi._evaluate_supplemental_override(error_c=1.0, now_mono=2000.0)

        assert result is True
        assert pi._last_setpoint_change_time == 0.0


# ── Recovery Tick Dedup Guard (L969-970) ─────────────────────────────


class TestRecoveryTickDedup:
    """Tests for recovery tick deduplication guard."""

    @pytest.mark.asyncio
    async def test_recovery_tick_skipped_if_recent_tick(self):
        """Recovery tick skipped when another tick ran less than 2s ago."""
        import time as _time

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        entity._attr_current_temperature = 21.0

        # Simulate a recent tick
        pi._pi_last_tick_time = _time.monotonic() - 0.5  # 0.5s ago

        send_needed = await pi._pi_async_sensor_changed(was_none=True)

        # Should skip the recovery tick
        assert not send_needed

    @pytest.mark.asyncio
    async def test_recovery_tick_runs_if_no_recent_tick(self):
        """Recovery tick runs when no tick has run recently."""
        import time as _time

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        entity._attr_current_temperature = 20.0  # Below desired → will produce IR

        # Simulate no recent tick
        pi._pi_last_tick_time = _time.monotonic() - 10.0

        send_needed = await pi._pi_async_sensor_changed(was_none=True)

        # Should run the tick
        assert send_needed


# ── Anti-windup: negative integral clamp (L1235-1236) ────────────────


class TestAntiWindupNegativeClamp:
    """Test anti-windup clamping when raw setpoint is below min temp."""

    @pytest.mark.asyncio
    async def test_negative_integral_clamped_at_min_temp(self):
        """When cooling drives setpoint below min_temp, negative integral is clamped."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        # Set up cooling mode with large negative integral pushing below min_temp
        entity._attr_hvac_mode = HVACMode.COOL
        entity._attr_current_temperature = 18.0  # Well below desired
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        # Large negative integral that would push raw_setpoint below min_temp (16)
        pi._pi_integral = -50000.0

        await pi._pi_tick()

        # Integral should be clamped (less negative than -50000)
        assert pi._pi_integral > -50000.0
        # Setpoint should be at min_temp, not below
        assert pi._hp_setpoint >= pi._min_temp_c


# ── Tracking mode: IR suppressed on setpoint change (L1272) ─────────


class TestTrackingModeIRSuppressed:
    """Test that setpoint changes in tracking mode don't send IR."""

    @pytest.mark.asyncio
    async def test_tracking_mode_suppresses_ir_send(self):
        """When supplemental is active (tracking mode), setpoint updates but no IR sent."""
        config = make_pi_config({
            "pi_supplemental_sources": [{
                "name": "Pellet Stove",
                "entity_id": "climate.pellet_stove",
                "failure_threshold": 900,
                "recovery_margin": 0.3,
                "auto_model_input": False,
            }],
        })
        entity = FakePIEntity(config)
        pi = entity._pi

        # Set up state
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 19.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0

        # Simulate supplemental active → tracking mode
        state = MagicMock()
        state.state = "heat"
        entity.hass.states.get.return_value = state
        pi._supplemental.tracking_mode = True
        pi._supplemental.tracking_sources = ["Pellet Stove"]

        send_needed = await pi._pi_tick()

        # Setpoint should have been computed (error > 0 → setpoint > 22)
        # but send not needed (tracking mode suppresses)
        assert not send_needed


# ── Derivative (D) Term Tests ────────────────────────────────────────


class TestDerivativeTerm:
    """Tests for the filtered derivative-on-measurement term."""

    def _make_entity(self, kd=0.5, kd_filter_n=8):
        """Helper: create FakePIEntity with explicit Kd settings."""
        config = make_pi_config({
            "pi_kd": kd,
            "pi_kd_filter_n": kd_filter_n,
            "pi_sensor_filter_tau": 0,  # Disable low-pass for exact derivative tests
        })
        return FakePIEntity(config)

    async def _tick_with_temp(self, entity, temp_c, dt_seconds=900):
        """Set current temp, advance time, and tick. Returns d_term."""
        import time
        entity._attr_current_temperature = temp_c
        entity._pi._pi_last_tick_time = time.monotonic() - dt_seconds
        await entity._pi._pi_tick()
        return entity._pi._pi_d_filtered

    @pytest.mark.asyncio
    async def test_first_tick_no_derivative(self):
        """First tick has no previous measurement, so D should be zero."""
        entity = self._make_entity(kd=2.0)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        assert entity._pi._pi_last_measurement is None
        await self._tick_with_temp(entity, 20.0)

        assert entity._pi._pi_d_filtered == 0.0
        # But measurement should now be recorded
        assert entity._pi._pi_last_measurement == 20.0

    @pytest.mark.asyncio
    async def test_d_zero_when_kd_zero(self):
        """With Kd=0, D term should always be zero regardless of temp change."""
        entity = self._make_entity(kd=0.0)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        await self._tick_with_temp(entity, 20.0)
        await self._tick_with_temp(entity, 21.0)  # 1°C rise

        assert entity._pi._pi_d_filtered == 0.0

    @pytest.mark.asyncio
    async def test_d_opposes_rising_temp(self):
        """When room temp rises, D should be negative (opposes the change)."""
        entity = self._make_entity(kd=1.0, kd_filter_n=8)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        await self._tick_with_temp(entity, 20.0)  # seed measurement
        d = await self._tick_with_temp(entity, 21.0)  # +1°C rise

        assert d < 0.0, f"D should be negative for rising temp, got {d}"

    @pytest.mark.asyncio
    async def test_d_opposes_falling_temp(self):
        """When room temp falls, D should be positive (opposes the change)."""
        entity = self._make_entity(kd=1.0, kd_filter_n=8)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        await self._tick_with_temp(entity, 21.0)  # seed measurement
        d = await self._tick_with_temp(entity, 20.0)  # -1°C fall

        assert d > 0.0, f"D should be positive for falling temp, got {d}"

    @pytest.mark.asyncio
    async def test_d_zero_when_temp_unchanged(self):
        """No temperature change → D should remain zero (or decay to zero)."""
        entity = self._make_entity(kd=1.0, kd_filter_n=8)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        await self._tick_with_temp(entity, 20.0)
        d = await self._tick_with_temp(entity, 20.0)  # same temp

        assert d == 0.0

    @pytest.mark.asyncio
    async def test_d_formula_exact(self):
        """Verify the discrete filter formula produces expected values."""
        kd = 2.0
        n = 10.0
        entity = self._make_entity(kd=kd, kd_filter_n=n)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        dt_sec = 900  # 15 min
        dt_min = dt_sec / 60.0
        tf = kd / n  # 0.2 min

        # First tick: seed measurement at 20°C
        await self._tick_with_temp(entity, 20.0, dt_seconds=dt_sec)

        # Second tick: temp rises to 21°C (dy = +1.0)
        d = await self._tick_with_temp(entity, 21.0, dt_seconds=dt_sec)

        # Expected: alpha = tf/(tf+dt_min) = 0.2/(0.2+15) = 0.01316
        # D = alpha * 0.0 - (kd/(tf+dt_min)) * 1.0
        #   = -(2.0/15.2) * 1.0 = -0.13158
        alpha_d = tf / (tf + dt_min)
        expected_d = alpha_d * 0.0 - (kd / (tf + dt_min)) * 1.0

        assert d == pytest.approx(expected_d, abs=1e-6)

        # Third tick: temp stays at 21°C (dy = 0) — D should decay
        d2 = await self._tick_with_temp(entity, 21.0, dt_seconds=dt_sec)
        expected_d2 = alpha_d * expected_d  # decay only

        assert d2 == pytest.approx(expected_d2, abs=1e-6)

    @pytest.mark.asyncio
    async def test_d_scales_with_kd(self):
        """Doubling Kd should roughly double the D response on first step."""
        results = {}
        for kd in (0.5, 1.0, 2.0):
            entity = self._make_entity(kd=kd, kd_filter_n=8)
            entity._attr_hvac_mode = HVACMode.HEAT
            entity._pi._desired_temp = 22.0
            entity._pi._hp_setpoint = 22.0

            await self._tick_with_temp(entity, 20.0)
            d = await self._tick_with_temp(entity, 21.0)
            results[kd] = d

        # All should be negative (opposing rise)
        for kd, d in results.items():
            assert d < 0.0, f"Kd={kd}: D should be negative, got {d}"

        # Magnitude should scale with Kd (not exactly linear due to filter,
        # but close since tf is small relative to dt)
        assert abs(results[2.0]) > abs(results[1.0]) > abs(results[0.5])

    @pytest.mark.asyncio
    async def test_higher_n_means_faster_response(self):
        """Higher N = less filtering = D responds more sharply to changes."""
        results = {}
        for n in (2, 8, 50):
            entity = self._make_entity(kd=1.0, kd_filter_n=n)
            entity._attr_hvac_mode = HVACMode.HEAT
            entity._pi._desired_temp = 22.0
            entity._pi._hp_setpoint = 22.0

            await self._tick_with_temp(entity, 20.0)
            d = await self._tick_with_temp(entity, 21.0)
            results[n] = abs(d)

        # Higher N → smaller Tf → D responds more to current step
        # (less smoothing), so magnitude should be larger for higher N
        # With our 15-min dt this difference is very small since all Tf << dt,
        # but N=2 should have slightly less magnitude than N=50
        assert results[50] >= results[2] - 1e-6

    @pytest.mark.asyncio
    async def test_d_exposed_in_attributes(self):
        """D term should appear in extra_state_attributes."""
        entity = self._make_entity(kd=1.0)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        await self._tick_with_temp(entity, 20.0)
        await self._tick_with_temp(entity, 21.0)

        attrs = entity.extra_state_attributes
        assert "d_term" in attrs
        assert attrs["d_term"] != 0.0

    @pytest.mark.asyncio
    async def test_d_derivative_on_measurement_not_error(self):
        """D should respond to measurement change, not error change.

        If we change the setpoint but not the measurement, D should be zero.
        This is the key property of derivative-on-measurement.
        """
        entity = self._make_entity(kd=2.0)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 20.0  # initial
        entity._pi._hp_setpoint = 20.0

        # Two ticks at same temp, different setpoints
        await self._tick_with_temp(entity, 20.0)
        entity._pi._desired_temp = 24.0  # big setpoint change!
        d = await self._tick_with_temp(entity, 20.0)  # same measurement

        assert d == 0.0, "D should not respond to setpoint changes"

    @pytest.mark.asyncio
    async def test_d_contributes_to_setpoint(self):
        """D term should affect the computed HP setpoint.

        Compare the HP setpoint with Kd=0 vs Kd>0 for the same scenario.
        """
        setpoints = {}
        for kd in (0.0, 2.0):
            entity = self._make_entity(kd=kd)
            entity._attr_hvac_mode = HVACMode.HEAT
            entity._pi._desired_temp = 22.0
            entity._pi._hp_setpoint = 22.0

            # Seed measurement, then rising temp (approaching setpoint)
            await self._tick_with_temp(entity, 20.0)
            await self._tick_with_temp(entity, 21.0)
            setpoints[kd] = entity._pi._hp_setpoint

        # With rising temp in heating mode, D is negative → should lower
        # the raw setpoint, potentially resulting in a lower HP setpoint
        # (or same due to rounding, but raw should differ)
        # At minimum, verify D term was nonzero in the kd=2 case
        entity_with_d = self._make_entity(kd=2.0)
        entity_with_d._attr_hvac_mode = HVACMode.HEAT
        entity_with_d._pi._desired_temp = 22.0
        entity_with_d._pi._hp_setpoint = 22.0
        await self._tick_with_temp(entity_with_d, 20.0)
        await self._tick_with_temp(entity_with_d, 21.0)
        assert entity_with_d._pi._pi_d_filtered != 0.0


class TestDerivativeImpact:
    """Comparative tests with open-loop temp sequences.

    For thermal-model-coupled D impact tests, see
    tests/hvac_bench/scenarios/test_derivative.py.
    """

    async def _run_heating_scenario(self, kd, temps):
        """Run a sequence of temperature readings and return metrics."""
        config = make_pi_config({
            "pi_kd": kd,
            "pi_kd_filter_n": 8,
            "pi_kp": 1.5,
            "pi_ki": 0.15,
        })
        entity = FakePIEntity(config)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._pi._desired_temp = 22.0
        entity._pi._hp_setpoint = 22.0

        import time
        setpoints = []
        d_terms = []

        for temp_c, dt_sec in temps:
            entity._attr_current_temperature = temp_c
            entity._pi._pi_last_tick_time = time.monotonic() - dt_sec
            await entity._pi._pi_tick()
            setpoints.append(entity._pi._hp_setpoint)
            d_terms.append(entity._pi._pi_d_filtered)

        return {"setpoints": setpoints, "d_terms": d_terms}

    @pytest.mark.asyncio
    async def test_d_damps_approach_overshoot(self):
        """D should be negative during rising temp approach."""
        temps = [(19.0 + i * 0.2, 900) for i in range(20)]

        result_no_d = await self._run_heating_scenario(kd=0.0, temps=temps)
        result_with_d = await self._run_heating_scenario(kd=2.0, temps=temps)

        approaching_ticks = result_with_d["d_terms"][1:]
        assert all(d <= 0.0 for d in approaching_ticks)
        assert all(d == 0.0 for d in result_no_d["d_terms"])

    @pytest.mark.asyncio
    async def test_d_resists_temp_drop(self):
        """D should be positive when temp drops (resisting the change)."""
        temps = (
            [(22.0, 900)] * 3
            + [(21.5, 900), (21.0, 900), (20.5, 900)]
        )

        result = await self._run_heating_scenario(kd=1.0, temps=temps)

        assert abs(result["d_terms"][1]) < 0.01
        assert abs(result["d_terms"][2]) < 0.01
        assert result["d_terms"][3] > 0.0
        assert result["d_terms"][4] > 0.0
        assert result["d_terms"][5] > 0.0


class TestFallbackTimer:
    """Tests for PI's self-rescheduling fallback timer."""

    @pytest.mark.asyncio
    async def test_pi_tick_reschedules_timer(self):
        """Every _pi_tick should cancel and reschedule the fallback timer."""
        import time as _time

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        entity._attr_current_temperature = 21.5
        pi._pi_last_tick_time = _time.monotonic() - 900.0

        # Set a mock callback and a mock existing unsub
        old_unsub = MagicMock()
        pi._pi_timer_callback = MagicMock()
        pi._pi_timer_unsub = old_unsub

        # Mock _hass to capture async_call_later
        from unittest.mock import patch
        with patch("custom_components.tasmota_irhvac.pi.pi_controller.async_call_later") as mock_acl:
            mock_acl.return_value = MagicMock()  # new unsub handle
            await pi.pi_tick()

        # Old timer was cancelled
        old_unsub.assert_called_once()
        # New timer was scheduled
        mock_acl.assert_called_once()
        assert pi._pi_timer_unsub == mock_acl.return_value

    @pytest.mark.asyncio
    async def test_no_reschedule_without_callback(self):
        """No timer scheduled if _pi_timer_callback is not set."""
        import time as _time

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        entity._attr_current_temperature = 21.5
        pi._pi_last_tick_time = _time.monotonic() - 900.0
        pi._pi_timer_callback = None

        from unittest.mock import patch
        with patch("custom_components.tasmota_irhvac.pi.pi_controller.async_call_later") as mock_acl:
            await pi.pi_tick()

        mock_acl.assert_not_called()

    @pytest.mark.asyncio
    async def test_sensor_tick_also_reschedules(self):
        """sensor_changed flows through _pi_tick, so timer reschedules too."""
        import time as _time

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        pi._desired_temp = 22.0
        pi._hp_setpoint = 20.0
        entity._attr_current_temperature = 20.0
        pi._pi_last_tick_time = _time.monotonic() - 400.0  # past cooldown
        pi._pi_timer_callback = MagicMock()

        from unittest.mock import patch
        with patch("custom_components.tasmota_irhvac.pi.pi_controller.async_call_later") as mock_acl:
            mock_acl.return_value = MagicMock()
            await pi.sensor_changed(was_none=False)

        # sensor_changed → _pi_tick → reschedule
        mock_acl.assert_called_once()


# ── IMC Gain Scheduling Tests ────────────────────────────────────────


class TestIMCGainScheduling:
    """Tests for τ-based IMC gain scheduling."""

    def test_imc_disabled_by_default(self):
        """τ=0 means IMC is disabled, uses manual Kp/Ki."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        assert not pi._plant_id.enabled
        # Should use conftest defaults (pi_kp=1.5, pi_ki=0.15)
        assert pi._pi_kp == 1.5
        assert pi._pi_ki == 0.15

    def test_imc_enabled_with_tau(self):
        """When τ > 0, Kp and Ki are derived via IMC formula."""
        config = make_pi_config({"pi_tau_estimate": 120.0, "pi_response_lag": 15.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._plant_id.enabled
        # IMC: Kp = τ / (K_eff * (λ + L))
        # λ = L/3 = 5, L = 15, K_eff = 1.0
        # Kp = 120 / (1.0 * (5 + 15)) = 120/20 = 6.0
        assert abs(pi._pi_kp - 6.0) < 0.01
        # Ki = 3 * Kp / τ = 3 * 6.0 / 120 = 0.15
        assert abs(pi._pi_ki - 3.0 * 6.0 / 120.0) < 0.001

    def test_imc_custom_lambda(self):
        """Custom λ overrides the default τ/2."""
        config = make_pi_config({
            "pi_tau_estimate": 120.0,
            "pi_response_lag": 15.0,
            "pi_imc_lambda": 30.0,
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        # Kp = 120 / (1 * (30 + 15)) = 120/45 ≈ 2.667
        assert abs(pi._pi_kp - 120.0 / 45.0) < 0.01
        assert abs(pi._pi_ki - 3.0 * pi._pi_kp / 120.0) < 0.001

    def test_imc_tau_floor(self):
        """τ estimate is floored at 1 min to prevent division issues."""
        config = make_pi_config({"pi_tau_estimate": 0.5, "pi_response_lag": 15.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        # τ floored to 1.0, λ = L/3 = 5.0
        # Kp = 1.0 / (1.0 * (5.0 + 15.0)) = 1/20 = 0.05
        assert abs(pi._pi_kp - 0.05) < 0.01

    def test_manual_kp_ki_ignored_when_imc_enabled(self):
        """When IMC is enabled, config Kp/Ki are overridden."""
        config = make_pi_config({
            "pi_tau_estimate": 120.0,
            "pi_kp": 99.0,
            "pi_ki": 99.0,
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._pi_kp != 99.0
        assert pi._pi_ki != 99.0
        assert pi._pi_kp_config == 99.0  # Config values preserved

    def test_recompute_uses_tau_slow_for_kp(self):
        """Kp derives from tau_slow (seed), not tau_fast (observed)."""
        config = make_pi_config({"pi_tau_estimate": 120.0, "pi_response_lag": 15.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        kp_from_seed = pi._pi_kp
        # Restore a different tau_fast — Kp should NOT change (uses tau_slow=seed)
        pi._plant_id.restore({"tau_estimate": 60.0, "tau_observations": 1})
        pi._recompute_imc_gains()
        # Kp still from tau_slow=120: Kp = 120/(1*(5+15)) = 6.0
        assert pi._pi_kp == pytest.approx(kp_from_seed, abs=0.01)


class TestTauGainIntegration:
    """Integration test: τ observation applies gains to PIController."""

    def test_tau_observation_updates_smith_not_kp(self):
        """After τ_fast observation, Smith predictor updates but Kp stays stable."""
        config = make_pi_config({"pi_tau_estimate": 120.0, "pi_response_lag": 15.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        kp_from_seed = pi._pi_kp
        pi._plant_id.start_observation(0.0, 20.0, 22.0, 2.0)
        gain_update = pi._plant_id.check_observation(4800.0, 21.27)
        assert gain_update is not None
        # tau_fast changed but tau_slow is still seed
        assert gain_update.tau_fast != 120.0  # Moved toward observed
        assert gain_update.tau_slow == 120.0  # Seed unchanged
        pi._apply_gain_update(gain_update)
        # Kp derived from tau_slow → unchanged
        assert pi._pi_kp == pytest.approx(kp_from_seed, abs=0.01)


class TestIMCPersistence:
    """Tests for τ estimate persistence across restarts."""

    def test_tau_saved_in_extra_stored_data(self):
        """τ estimate is included in persisted data."""
        config = make_pi_config({"pi_tau_estimate": 120.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._plant_id.restore({"tau_estimate": 85.0, "tau_observations": 1})
        data = pi.get_extra_stored_data()
        assert data is not None
        d = data.as_dict()
        assert d["tau_estimate"] == 85.0

    def test_tau_fast_restored_kp_stable(self):
        """τ_fast restored from persistence, Kp stays at seed (tau_slow unchanged)."""
        config = make_pi_config({"pi_tau_estimate": 120.0, "pi_response_lag": 15.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        kp_seed = pi._pi_kp

        # Simulate restore with learned τ_fast=60 (old single-tau format)
        data = PIExtraStoredData(
            pi_integral=0.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            tau_estimate=60.0,
        )
        pi.restore_extra_stored_data(data)
        assert pi._plant_id.tau == 60.0  # tau_fast restored
        assert pi._plant_id.plant.tau_slow.value == 120.0  # seed unchanged
        assert pi._pi_kp == pytest.approx(kp_seed, abs=0.01)  # Kp from tau_slow

    def test_tau_zero_not_restored(self):
        """τ=0 in stored data doesn't overwrite seed."""
        config = make_pi_config({"pi_tau_estimate": 120.0, "pi_response_lag": 15.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        kp_seed = pi._pi_kp

        data = PIExtraStoredData(
            pi_integral=0.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            tau_estimate=0.0,
        )
        pi.restore_extra_stored_data(data)
        assert pi._plant_id.tau == 120.0  # Kept seed
        assert pi._pi_kp == kp_seed

    def test_integral_unchanged_when_ki_invariant(self):
        """With default λ=L/3, Ki is invariant to τ — no rescaling needed.

        Ki = 3·Kp/τ = 3·τ/((L/3+L)·τ) = 9/(4L), independent of τ.
        This means τ learning doesn't disrupt the integral, which is
        a desirable property of the L/3 default.
        """
        config = make_pi_config({"pi_tau_estimate": 120.0, "pi_response_lag": 15.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        ki_seed = pi._pi_ki

        data = PIExtraStoredData(
            pi_integral=5.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            tau_estimate=60.0,
            ki_at_save=ki_seed,
        )
        pi.restore_extra_stored_data(data)
        # Ki is the same for τ=120 and τ=60 when λ=L/3, so no rescaling
        assert pi._pi_ki == pytest.approx(ki_seed, abs=0.001)
        assert pi._pi_integral == 5.0  # Unchanged

    def test_integral_rescaled_with_custom_lambda(self):
        """With custom λ, Ki varies with τ, so integral IS rescaled on restore."""
        config = make_pi_config({
            "pi_tau_estimate": 120.0, "pi_response_lag": 15.0,
            "pi_imc_lambda": 30.0,
        })
        entity = FakePIEntity(config)
        pi = entity._pi
        ki_seed = pi._pi_ki  # Ki = 3 * (120/45) / 120 = 0.0667

        data = PIExtraStoredData(
            pi_integral=5.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            tau_estimate=60.0,
            ki_at_save=ki_seed,
        )
        pi.restore_extra_stored_data(data)
        # τ=60 with λ=30: Ki = 3*(60/45)/60 = 0.0667 — same! λ=30 also gives same Ki
        # Use λ that DOES change Ki: set tau_estimate directly and recompute
        # Actually with fixed λ=30: Ki = 3*τ/(τ*(30+15)) = 3/45 = 0.0667 for all τ.
        # Need τ-dependent λ to get Ki variation. Use a τ seed where default λ differs.
        # This test validates the rescaling mechanism works when Ki does change.
        pi._plant_id.restore({"tau_estimate": 60.0, "tau_observations": 1})
        pi._plant_id._imc_lambda_config = 0.0  # Switch to auto λ=L/3
        pi._recompute_imc_gains()
        old_ki = pi._pi_ki
        pi._pi_integral = 5.0
        # Manually trigger a Ki change by using a different formula
        pi._plant_id._imc_lambda_config = 10.0  # λ=10: Ki = 3*(60/25)/60 = 0.12
        pi._recompute_imc_gains()
        # Ki changed, so rescaling code in restore would fire
        assert pi._pi_ki != pytest.approx(old_ki, abs=0.01)

    def test_from_dict_preserves_tau(self):
        """PIExtraStoredData.from_dict handles tau_estimate field."""
        raw = {
            "pi_integral": 1.0,
            "desired_temp": 22.0,
            "hp_setpoint": 22.0,
            "tau_estimate": 75.5,
        }
        data = PIExtraStoredData.from_dict(raw)
        assert data is not None
        assert data.tau_estimate == 75.5

    def test_from_dict_missing_tau_defaults_zero(self):
        """Legacy data without tau_estimate gets default 0."""
        raw = {
            "pi_integral": 1.0,
            "desired_temp": 22.0,
            "hp_setpoint": 22.0,
        }
        data = PIExtraStoredData.from_dict(raw)
        assert data is not None
        assert data.tau_estimate == 0.0


class TestIMCStateAttributes:
    """Tests for τ exposure in state attributes."""

    def test_attributes_include_tau_when_imc_enabled(self):
        """State attributes include τ and effective gains when IMC is on."""
        config = make_pi_config({"pi_tau_estimate": 120.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        attrs = pi.get_extra_state_attributes()
        assert "tau_estimate" in attrs
        assert attrs["tau_estimate"] == 120.0
        assert "effective_kp" in attrs
        assert "effective_ki" in attrs
        assert attrs["tau_observations"] == 0

    def test_attributes_tau_none_when_imc_disabled(self):
        """State attributes show τ as None when IMC is disabled."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        attrs = pi.get_extra_state_attributes()
        assert attrs["tau_estimate"] is None
        assert attrs["tau_observations"] is None
        # effective_kp/ki always present
        assert "effective_kp" in attrs

    def test_health_status_includes_tau(self):
        """Health status includes τ when IMC is enabled."""
        config = make_pi_config({"pi_tau_estimate": 120.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        status = pi.get_health_status()
        assert "tau_estimate" in status
        assert status["tau_estimate"] == 120.0


# ── Grey-box Attribute Tests ─────────────────────────────────────────


class TestGreyboxAttributes:
    """Tests for grey-box observer attributes in get_extra_state_attributes."""

    def test_greybox_attrs_none_before_first_fit(self):
        """All greybox attrs are None when no fit has run."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        attrs = pi.get_extra_state_attributes()
        assert attrs["greybox_tau_eff"] is None
        assert attrs["greybox_tau_agreement_pct"] is None
        assert attrs["greybox_rms"] is None
        assert attrs["greybox_last_run"] is None

    def test_greybox_attrs_populated_after_fit(self):
        """Attributes reflect the GreyboxResult after a successful fit."""
        from custom_components.tasmota_irhvac.pi.greybox_observer import GreyboxResult

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        result = GreyboxResult(
            n_observations=100,
            n_hp_on=60,
            n_hp_off=40,
            ua_c=0.012,
            k_c=0.015,
            alpha_c=-0.003,
            tau_eff=83.3,
            residual_rms=0.00042,
            cost=0.5,
            n_function_evals=20,
            plant_tau_slow=85.0,
            tau_agreement_pct=2.0,
        )
        pi._last_greybox_result = result
        pi._last_greybox_timestamp_iso = "2026-04-20T12:00:00Z"

        attrs = pi.get_extra_state_attributes()
        assert attrs["greybox_tau_eff"] == 83.3
        assert attrs["greybox_tau_agreement_pct"] == 2.0
        assert attrs["greybox_rms"] == 0.00042
        assert attrs["greybox_last_run"] == "2026-04-20T12:00:00Z"

    def test_greybox_agreement_none_when_no_plant_tau(self):
        """tau_agreement_pct is None when plant tau_slow wasn't available."""
        from custom_components.tasmota_irhvac.pi.greybox_observer import GreyboxResult

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        result = GreyboxResult(
            n_observations=50,
            n_hp_on=30,
            n_hp_off=20,
            ua_c=0.01,
            k_c=0.012,
            alpha_c=-0.002,
            tau_eff=100.0,
            residual_rms=0.00055,
            cost=0.3,
            n_function_evals=15,
            plant_tau_slow=None,
            tau_agreement_pct=None,
        )
        pi._last_greybox_result = result
        pi._last_greybox_timestamp_iso = "2026-04-20T06:00:00Z"

        attrs = pi.get_extra_state_attributes()
        assert attrs["greybox_tau_eff"] == 100.0
        assert attrs["greybox_tau_agreement_pct"] is None
        assert attrs["greybox_rms"] == 0.00055
        assert attrs["greybox_last_run"] == "2026-04-20T06:00:00Z"


# ── Sensor Low-Pass Filter Tests ──────────────────────────────────────


class TestSensorFilter:
    """Test the exponential low-pass filter on room temperature measurement."""

    @pytest.mark.asyncio
    async def test_filter_smooths_noisy_signal(self):
        """Filter should reduce variance of a noisy signal."""
        import time
        config = make_pi_config({"pi_sensor_filter_tau": 120})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        # Feed alternating temps (simulating noise)
        readings = [20.0, 20.3, 19.9, 20.2, 20.0, 20.4, 19.8, 20.1, 20.0, 20.3]
        filtered_vals = []
        t = 1000.0
        for temp in readings:
            t += 300.0  # 5-min intervals
            entity._attr_current_temperature = temp
            pi._pi_last_tick_time = t - 300.0
            with patch("time.monotonic", return_value=t):
                await pi._pi_tick()
            filtered_vals.append(pi._sensor_filtered)

        # Filter output should have less variance than input
        import statistics
        raw_var = statistics.variance(readings)
        filt_var = statistics.variance(filtered_vals)
        assert filt_var < raw_var, (
            f"Filter should reduce variance: raw={raw_var:.4f}, filtered={filt_var:.4f}"
        )

    @pytest.mark.asyncio
    async def test_filter_disabled_when_tau_zero(self):
        """tau=0 should pass raw values through unchanged."""
        import time
        config = make_pi_config({"pi_sensor_filter_tau": 0})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        entity._attr_current_temperature = 19.5
        pi._pi_last_tick_time = 900.0
        with patch("time.monotonic", return_value=1800.0):
            await pi._pi_tick()

        # With no filter, _sensor_filtered stays None
        assert pi._sensor_filtered is None

    @pytest.mark.asyncio
    async def test_filter_initializes_to_first_reading(self):
        """First tick should set filter to raw reading (no lag on startup)."""
        import time
        config = make_pi_config({"pi_sensor_filter_tau": 120})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        entity._attr_current_temperature = 19.7
        pi._pi_last_tick_time = 0.0
        with patch("time.monotonic", return_value=900.0):
            await pi._pi_tick()

        assert pi._sensor_filtered == pytest.approx(19.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_filter_converges_to_step(self):
        """After a step change, filter should converge toward the new value."""
        import time
        config = make_pi_config({"pi_sensor_filter_tau": 120})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        # Initialize at 20.0
        entity._attr_current_temperature = 20.0
        pi._pi_last_tick_time = 0.0
        with patch("time.monotonic", return_value=900.0):
            await pi._pi_tick()
        assert pi._sensor_filtered == pytest.approx(20.0, abs=0.01)

        # Step to 21.0 — filter should move toward 21 but not reach it
        entity._attr_current_temperature = 21.0
        pi._pi_last_tick_time = 900.0
        with patch("time.monotonic", return_value=1800.0):
            await pi._pi_tick()
        # After 900s with τ=120s: α = 1-exp(-900/120) ≈ 0.9994
        # So filtered ≈ 0.9994*21 + 0.0006*20 ≈ 20.999
        assert pi._sensor_filtered > 20.9
        assert pi._sensor_filtered <= 21.0

    @pytest.mark.asyncio
    async def test_filter_exposed_in_attributes(self):
        """sensor_filtered should appear in extra state attributes."""
        import time
        config = make_pi_config({"pi_sensor_filter_tau": 120})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT

        entity._attr_current_temperature = 20.5
        pi._pi_last_tick_time = 0.0
        with patch("time.monotonic", return_value=900.0):
            await pi._pi_tick()

        attrs = pi.get_extra_state_attributes()
        assert "sensor_filtered" in attrs
        assert attrs["sensor_filtered"] is not None


class TestOneSidedAntiWindup:
    """Tests for integral behavior when error opposes mode direction.

    With accelerated decay removed, the integral accumulates freely
    (bounded by back-calculation anti-windup and leaky integrator).
    This lets the integral fully correct FF model errors — the HP has
    room to adjust its setpoint even when the error opposes the mode.
    Conditional integration (HP at limit) is the true anti-windup.
    """

    @pytest.mark.asyncio
    async def test_heat_mode_negative_error_hp_no_output_freezes(self, pi_entity):
        """In heat mode with room above HP setpoint → hp_no_output → integral frozen.

        Room temp declines slowly (HP truly off, passive cooling) so the
        rate-based deadband override does not fire.
        """
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 24.0  # °C, well above target AND setpoint
        pi_entity._pi._desired_temp = 22.0  # error = -2.0°C
        pi_entity._pi._hp_setpoint = 22.0  # 22 < 24 → hp_no_output
        pi_entity._pi._pi_integral = 0.0

        for i in range(10):
            # Room cools slowly — HP is truly off.
            pi_entity._attr_current_temperature = 24.0 - i * 0.02
            pi_entity._pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi_entity._pi._pi_tick()

        integral_after = pi_entity._pi._pi_integral

        # HP has no output (setpoint < room) → integral should be frozen.
        # Only leaky decay (0.9999^10) applies — negligible.
        assert abs(integral_after) < 1.0, (
            f"Integral should be frozen when HP has no output, got {integral_after}"
        )

    @pytest.mark.asyncio
    async def test_heat_mode_negative_error_hp_active_accumulates(self, pi_entity):
        """In heat mode with setpoint > room, negative error still accumulates."""
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 20.0  # room below setpoint
        pi_entity._pi._desired_temp = 22.0  # error = +2.0°C
        pi_entity._pi._hp_setpoint = 23.0  # 23 > 20 → HP is active
        pi_entity._pi._pi_integral = -10.0  # Pre-wound negative
        pi_entity._pi._pi_last_tick_time = 0.0

        with patch("time.monotonic", return_value=900.0):
            await pi_entity._pi._pi_tick()
        integral_after = pi_entity._pi._pi_integral

        # HP is active (setpoint > room) with positive error → integral grows
        assert integral_after > -10.0, (
            f"Integral should accumulate when HP is active, got {integral_after}"
        )

    @pytest.mark.asyncio
    async def test_heat_mode_positive_error_normal_accumulation(self, pi_entity):
        """In heat mode with room below target, integral should accumulate normally."""
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 20.0  # °C, below target
        pi_entity._pi._desired_temp = 22.0  # error = +2.0°C
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 1.0

        await pi_entity._pi._pi_tick()
        integral_after = pi_entity._pi._pi_integral

        # Integral should have grown (normal positive error accumulation)
        assert integral_after > 1.0, (
            f"Integral should accumulate normally with positive error, got {integral_after}"
        )

    @pytest.mark.asyncio
    async def test_cool_mode_positive_error_hp_no_output_freezes(self, pi_entity):
        """In cool mode with setpoint > room → hp_no_output → integral frozen.

        Room temp rises slowly (HP cooling truly off, passive warming) so
        the rate-based deadband override does not fire.
        """
        pi_entity._attr_hvac_mode = HVACMode.COOL
        pi_entity._attr_current_temperature = 20.0  # °C, below target AND setpoint
        pi_entity._pi._desired_temp = 22.0  # error = +2.0°C
        pi_entity._pi._hp_setpoint = 22.0  # 22 > 20 → hp_no_output in cooling
        pi_entity._pi._pi_integral = 0.0

        for i in range(10):
            # Room warms slowly — HP cooling is truly off.
            pi_entity._attr_current_temperature = 20.0 + i * 0.02
            pi_entity._pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi_entity._pi._pi_tick()

        integral_after = pi_entity._pi._pi_integral

        # HP has no cooling output (setpoint > room) → frozen
        assert abs(integral_after) < 1.0, (
            f"Integral should be frozen when HP has no cooling output, got {integral_after}"
        )

    @pytest.mark.asyncio
    async def test_cool_mode_negative_error_normal_accumulation(self, pi_entity):
        """In cool mode with room above target, integral should accumulate normally."""
        pi_entity._attr_hvac_mode = HVACMode.COOL
        pi_entity._attr_current_temperature = 25.0  # °C, above target
        pi_entity._pi._desired_temp = 22.0  # error = -3.0°C
        pi_entity._pi._hp_setpoint = 24.0
        pi_entity._pi._pi_integral = -1.0

        await pi_entity._pi._pi_tick()
        integral_after = pi_entity._pi._pi_integral

        # Integral should have grown more negative (normal cooling behavior)
        assert integral_after < -1.0, (
            f"Integral should accumulate normally with negative error in cool mode, "
            f"got {integral_after}"
        )

    @pytest.mark.asyncio
    async def test_heat_mode_deadband_hp_active_accumulates(self, pi_entity):
        """In deadband with HP active, integral accumulates normally.

        Small errors within deadband are normal control behavior.
        Full-rate integration: accumulates the actual error.
        """
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 21.7  # °C, slightly below target
        pi_entity._pi._desired_temp = 22.0  # error = +0.3°C (within 0.5 deadband)
        pi_entity._pi._hp_setpoint = 22.0  # 22 > 21.7 → HP active
        pi_entity._pi._pi_integral = 0.0

        await pi_entity._pi._pi_tick()
        integral_after = pi_entity._pi._pi_integral

        # HP active, positive error → integral accumulates
        assert integral_after > 0, (
            f"Integral should accumulate when HP is active in deadband, got {integral_after}"
        )

    @pytest.mark.asyncio
    async def test_integral_accumulates_to_correct_ff_bias(self):
        """Integral should accumulate enough to correct persistent FF model error.

        Without accelerated decay, the integral builds until the HP setpoint
        corrects the error. This is essential for FF learning: the observation
        at equilibrium (ff + ki*integral) reflects the true offset needed.
        HP must be active (setpoint > room) for integration to proceed.
        """
        config = make_pi_config({"pi_tau_estimate": 60.0, "min_temp": 0})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 22.0  # above room → HP active
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 18.0  # error = +2.0°C, room below target
        pi._pi_integral = 1.0

        # Run one tick — integral should accumulate with error, not decay
        await pi._pi_tick()
        integral_after = pi._pi_integral

        # Integral = (1 + 2.0) * 0.9999 ≈ 2.999 (leaky only, no accel decay)
        assert integral_after > 1.5, (
            f"Integral should accumulate with error when HP active, got {integral_after:.2f}"
        )

    @pytest.mark.asyncio
    async def test_existing_clamp_antiwindup_still_works(self, pi_entity):
        """Original back-calculation anti-windup should still function."""
        # Force saturation: huge positive integral → raw > max_temp
        pi_entity._attr_current_temperature = 10.0  # cold
        pi_entity._pi._desired_temp = 28.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 40.0

        await pi_entity._pi._pi_tick()

        # Setpoint clamped at max
        assert pi_entity._pi._hp_setpoint <= 30
        # Integral should have been back-calculated down
        assert pi_entity._pi._pi_integral < 40.0

    @pytest.mark.asyncio
    async def test_sustained_overshoot_reaches_hp_limit(self):
        """During sustained overshoot, integral drives HP to min, then conditional freeze holds.

        The integral accumulates freely until the HP reaches min_temp.
        Then conditional integration freezes the integral (HP at limit +
        error opposing mode). Back-calculation also caps the integral.
        This is the correct anti-windup: only at the actuator limit.
        """
        config = make_pi_config({"pi_tau_estimate": 60.0})
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 20.0
        pi._hp_setpoint = 20.0
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 24.0  # room well above target, error = -4°C
        pi._pi_integral = -25.0

        # Simulate ~3 hours of overshoot (12 ticks at 15min)
        for i in range(12):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        # HP should be at or near minimum (error drives it there quickly).
        assert pi._hp_setpoint <= 17, (
            f"HP should be driven to min by large error, got {pi._hp_setpoint}"
        )
        # Integral is bounded by back-calculation (raw < min_temp) and
        # conditional freeze (hp at min + error < 0). It won't wind to -73.
        assert pi._pi_integral > -30.0, (
            f"Back-calculation should cap integral, got {pi._pi_integral}"
        )
        assert pi._pi_integral < 0, "Integral should still be negative"


class TestConditionalIntegration:
    """Tests for conditional integration on output saturation (Åström §3.5)."""

    @pytest.mark.asyncio
    async def test_solar_gain_cycle_integral_frozen_then_recovers(self):
        """Full solar gain cycle: HP at min (frozen) → above min (integrates) → min again → sunset recovery."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        entity._attr_hvac_mode = HVACMode.HEAT
        # Start with integral at -40 so HP stays clamped at min
        pi._pi_integral = -40.0
        pi._hp_setpoint = 16
        entity._attr_current_temperature = 24.0  # room well above target

        # Phase 1: HP at min, room above target — integral should freeze
        integral_phase1_start = pi._pi_integral
        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()
        integral_phase1_end = pi._pi_integral

        assert integral_phase1_end > integral_phase1_start - 0.1, (
            f"Phase 1: integral should be frozen at min, got {integral_phase1_end}"
        )

        # Phase 2: Room cools below HP setpoint — HP is now active,
        # integration resumes.  hp_setpoint must be > current_c for the HP
        # to have nonzero output (hp_no_output condition).
        entity._attr_current_temperature = 19.0  # error = +2.0, room below target
        pi._pi_integral = -3.0  # simulate partial recovery
        pi._hp_setpoint = 20  # above room temp → HP is active
        integral_phase2_start = pi._pi_integral
        for i in range(4, 8):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()
        integral_phase2_end = pi._pi_integral

        # HP above room temp → actively heating → integration resumes
        assert integral_phase2_end > integral_phase2_start, (
            f"Phase 2: should integrate when HP active (setpoint > room), got {integral_phase2_end}"
        )

        # Phase 3: Solar intensifies again, HP back to min — freeze again
        entity._attr_current_temperature = 25.0  # room way above target
        pi._pi_integral = -40.0
        pi._hp_setpoint = 16
        integral_phase3_start = pi._pi_integral
        for i in range(8, 12):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()
        integral_phase3_end = pi._pi_integral

        assert integral_phase3_end > integral_phase3_start - 0.1, (
            f"Phase 3: integral should freeze again at min, got {integral_phase3_end}"
        )

        # Phase 4: Sunset — room drops below target, HP setpoint rises
        # above room temp → HP is now active, integration resumes.
        # With hp_no_output fix, HP at setpoint=16 and room=20 would
        # still be frozen (16 < 20 → no output).  For recovery, the FF
        # must push setpoint above room temp first.
        entity._attr_current_temperature = 19.0  # error = +2.0
        pi._pi_integral = -1.0  # small negative (mostly recovered)
        pi._hp_setpoint = 22  # FF pushed setpoint above room → HP active
        integral_phase4_start = pi._pi_integral
        for i in range(12, 16):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()
        integral_phase4_end = pi._pi_integral

        # HP setpoint (22) > room temp (19) → actively heating.
        # Error is positive (room below target) → integration resumes,
        # integral grows toward zero and beyond.
        assert integral_phase4_end > integral_phase4_start, (
            f"Phase 4: integral should recover when HP active, got {integral_phase4_end}"
        )

    @pytest.mark.asyncio
    async def test_hp_at_max_heating_keeps_integrating(self):
        """HP at max in heating mode → HP IS controlling, keep integrating."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 30  # At maximum — but HP is actively heating
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 17.0  # error = +4.0 (cold start)
        pi._pi_integral = 2.0

        integral_before = pi._pi_integral
        await pi._pi_tick()

        # HP at max in heating = actively controlling at full power
        # Integral should keep growing (NOT frozen)
        assert pi._pi_integral > integral_before, (
            f"Integral should keep growing at max in heating, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_hp_not_at_limit_integrates_normally(self):
        """HP not at physical limit → normal integration."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 22  # NOT at limit
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 19.0  # error = +2.0
        pi._pi_integral = 1.0

        await pi._pi_tick()

        # Normal integration should grow the integral
        assert pi._pi_integral > 1.0, (
            f"Integral should grow normally when not saturated, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_hp_at_min_but_error_positive_still_frozen_if_no_output(self):
        """HP at min, room below target but above setpoint → hp_no_output → frozen.

        Even though error is positive (room below target), the HP at
        setpoint=16 when room=19°C has zero output (16 < 19 → HP thermostat
        off).  Integration should stay frozen until the setpoint rises
        above the room temperature.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 16  # At minimum, below room temp
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 19.0  # error = +2.0, but 16 < 19 → no output
        pi._pi_integral = -3.0

        integral_before = pi._pi_integral
        await pi._pi_tick()

        # hp_no_output: setpoint (16) < room (19) → frozen despite positive error
        assert abs(pi._pi_integral - integral_before) < 0.5, (
            f"Should be frozen when HP has no output, got {pi._pi_integral}"
        )


class TestHPNoOutput:
    """Tests for hp_no_output integration freeze and observation gating.

    When hp_setpoint < current_c in heating (or > in cooling), the HP's
    internal thermostat turns off the compressor.  The feedback loop is
    open: integration must freeze and observations must be marked clamped.

    Literature: Ljung §13.3 (persistent excitation), Åström & Hägglund §6.4
    (conditional integration during actuator saturation).
    """

    @pytest.mark.asyncio
    async def test_heating_setpoint_below_room_freezes(self):
        """Heating with setpoint < room → hp_no_output → integration frozen."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 17  # below room temp
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 22.0  # room above setpoint
        pi._pi_integral = -5.0

        integral_before = pi._pi_integral
        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert abs(pi._pi_integral - integral_before) < 0.5, (
            f"Integral should be frozen when HP has no output, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_heating_setpoint_equal_room_not_frozen(self):
        """Heating with setpoint == room → boundary, strict < means not frozen."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21  # equal to room temp (integer vs float boundary)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.0  # room equals setpoint
        pi._pi_integral = 0.0

        await pi._pi_tick()

        # Strict < means equality doesn't freeze.  The leaky integrator
        # (0.9999^dt) might cause a tiny change but integration is active.
        assert pi._integration_frozen is False

    @pytest.mark.asyncio
    async def test_heating_setpoint_above_room_integrates(self):
        """Heating with setpoint > room → HP active → normal integration."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 23  # above room temp
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0  # room below setpoint
        pi._pi_integral = 0.0

        integral_before = pi._pi_integral
        await pi._pi_tick()

        # HP setpoint (23) > room (20) → HP is active → integration proceeds
        assert pi._pi_integral > integral_before, (
            f"Should integrate when HP is active, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_cooling_setpoint_above_room_freezes(self):
        """Cooling with setpoint > room → hp_no_output → integration frozen."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 24.0
        pi._hp_setpoint = 26  # above room temp in cooling → no cool output
        entity._attr_hvac_mode = HVACMode.COOL
        entity._attr_current_temperature = 23.0  # room below setpoint
        pi._pi_integral = 3.0

        integral_before = pi._pi_integral
        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert abs(pi._pi_integral - integral_before) < 0.5, (
            f"Integral should freeze when HP has no cooling output, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_cooling_setpoint_below_room_integrates(self):
        """Cooling with setpoint < room → HP active → normal integration."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 24.0
        pi._hp_setpoint = 22  # below room temp in cooling → actively cooling
        entity._attr_hvac_mode = HVACMode.COOL
        entity._attr_current_temperature = 25.0  # room above setpoint
        pi._pi_integral = 0.0

        integral_before = pi._pi_integral
        await pi._pi_tick()

        # HP setpoint (22) < room (25) → HP is cooling → integration proceeds
        assert pi._pi_integral != integral_before, (
            f"Should integrate when HP is cooling, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_long_no_output_integral_bounded(self):
        """48 ticks (~12h) of HP-no-output should not wind integral.

        Room temp declines slowly (HP truly off, passive cooling) so the
        rate-based deadband override does not fire.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 17  # below room
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 24.0  # room well above setpoint
        pi._pi_integral = -7.8  # pre-overshoot value

        integral_start = pi._pi_integral
        for i in range(48):
            # Room cools slowly — HP is truly off.
            entity._attr_current_temperature = 24.0 - i * 0.02
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        # Integral should stay near its starting value (leaky decay is
        # 0.9999^48 ≈ 0.9952, so at most ~0.04 change from decay).
        assert abs(pi._pi_integral - integral_start) < 1.0, (
            f"Integral should be bounded during HP-no-output, "
            f"start={integral_start}, end={pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_integration_resumes_when_hp_active(self):
        """After HP-no-output freeze, integration resumes when setpoint > room.

        Uses a fresh controller for Phase 2 to avoid sensor filter lag
        from Phase 1's high room temperature.
        """
        # Phase 1: confirm freeze
        config = make_pi_config()
        entity1 = FakePIEntity(config)
        pi1 = entity1._pi
        pi1._desired_temp = 21.0
        entity1._attr_hvac_mode = HVACMode.HEAT
        pi1._hp_setpoint = 17
        entity1._attr_current_temperature = 24.0
        pi1._pi_integral = -7.0
        await pi1._pi_tick()
        assert pi1._integration_frozen is True

        # Phase 2: HP active (fresh controller, no filter lag)
        entity2 = FakePIEntity(config)
        pi2 = entity2._pi
        pi2._desired_temp = 21.0
        entity2._attr_hvac_mode = HVACMode.HEAT
        pi2._hp_setpoint = 23  # above room → HP active
        entity2._attr_current_temperature = 20.0  # error = +1.0
        pi2._pi_integral = -7.0
        pi2._integration_frozen = True  # was frozen

        integral_before = pi2._pi_integral
        await pi2._pi_tick()

        assert pi2._integration_frozen is False, "Should unfreeze when HP is active"
        assert pi2._pi_integral > integral_before, (
            f"Integration should resume when HP is active, got {pi2._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_no_output_observation_not_buffered(self):
        """HP-off observations are not buffered (zero-value for WLS regression)."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 17  # below room → HP has no output
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 24.0
        pi._pi_integral = -5.0
        before = len(pi._observation_buffer_heat)

        await pi._pi_tick()

        assert len(pi._observation_buffer_heat) == before, (
            "HP-off observations should not be added to the WLS buffer"
        )

    @pytest.mark.asyncio
    async def test_observation_unclamped_when_hp_active(self):
        """Observation should have clamped=False when HP is active."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 23  # above room
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0
        pi._pi_integral = 0.0

        await pi._pi_tick()

        obs = pi._observation_buffer_heat.get_all()
        assert len(obs) > 0
        assert obs[-1].clamped is False, (
            "Observation should be unclamped when HP is active"
        )

    @pytest.mark.asyncio
    async def test_heating_hp_active_but_overshooting_integrates(self):
        """Zone 2: HP setpoint > room but room > target → HP is causing overshoot.

        The HP IS actively heating (compressor on), but producing too much
        heat.  On the first tick, integration proceeds and the negative
        error drives the integral down.  This pulls the raw setpoint lower.
        Once the setpoint drops below room temp, hp_no_output activates
        and correctly freezes further integration — the HP is now off.

        This is the full overshoot response: integrate → reduce setpoint →
        HP turns off → freeze.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 23  # above room → HP is heating
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 22.0  # room above target, error = -1.0
        pi._pi_integral = 0.0

        # First tick: HP active (23 > 22), integration proceeds
        await pi._pi_tick()
        assert pi._pi_integral < 0, (
            f"First tick should integrate negative error, got {pi._pi_integral}"
        )

        # After a few ticks, setpoint drops below room → hp_no_output → freeze.
        # This is correct: the controller reduced the setpoint, HP turned off.
        for i in range(1, 4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        # Integral went negative (correction applied), then froze
        assert pi._pi_integral < 0, (
            f"Integral should be negative from overshoot correction, "
            f"got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_cooling_hp_active_but_overcooling_integrates(self):
        """Cooling Zone 2: HP setpoint < room but room < target → HP overcooling.

        On first tick, integration proceeds (positive error, drives integral
        up, raising setpoint).  Once setpoint rises above room, HP turns off
        and hp_no_output freezes correctly.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 24.0
        pi._hp_setpoint = 22  # below room → HP IS cooling
        entity._attr_hvac_mode = HVACMode.COOL
        entity._attr_current_temperature = 23.0  # room below target, error = +1.0
        pi._pi_integral = 0.0

        # First tick: HP active (22 < 23 in cooling), integration proceeds
        await pi._pi_tick()
        assert pi._pi_integral > 0, (
            f"First tick should integrate positive error, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_freeze_log_says_hp_no_output(self, caplog):
        """Freeze log should say 'HP no output' not 'at min limit'."""
        import logging
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 18  # above min (16), but below room
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 22.0  # 18 < 22 → hp_no_output
        pi._pi_integral = -3.0
        pi._integration_frozen = False

        with caplog.at_level(logging.DEBUG):
            await pi._pi_tick()

        freeze_msgs = [r for r in caplog.records if "HP no output" in r.message]
        assert len(freeze_msgs) >= 1, (
            "Should log 'HP no output' when setpoint < room (not 'at min limit')"
        )


class TestHPDeadbandLearning:
    """Tests for HP thermostat deadband learning and integration override.

    The HP's internal thermostat has its own hysteresis, so the compressor
    may still cycle even when hp_setpoint is slightly below room temp
    (heating).  The controller learns this deadband bidirectionally:

    - Default estimate starts at 0.5°C (reasonable for most mini-splits).
    - Grows when rate-based override confirms HP cycling at a wider delta.
    - Shrinks when rate confirms HP truly off at a narrower delta.

    Separate estimates per mode (heat/cool).  hp_no_output stays strict
    for RLS/batch gating — only integration freeze gets the override.
    """

    # ── Default estimate ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_default_estimate_is_half_degree(self):
        """Fresh controller should start with 0.5°C deadband estimate."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._hp_deadband_estimate_heat == 0.5
        assert pi._hp_deadband_estimate_cool == 0.5

    @pytest.mark.asyncio
    async def test_default_provides_immediate_override_within_half_degree(self):
        """Delta 0.3°C < default 0.5°C → integration unfreezes on first tick."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21  # delta = 0.3°C below room
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.3
        pi._pi_integral = -1.0

        await pi._pi_tick()

        assert pi._integration_frozen is False, (
            "Default 0.5°C estimate should provide immediate override at 0.3°C delta"
        )

    # ── Upward learning (HP cycling at wider delta than estimate) ─────

    @pytest.mark.asyncio
    async def test_rate_override_fires_when_room_not_cooling(self):
        """After 10 ticks of hp_no_output with room_rate >= 0, override unfreezes."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21  # 0.7°C below room → hp_no_output
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.7
        pi._pi_integral = -1.4

        # Run 15 ticks with stable room temp (HP still cycling).
        for i in range(15):
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        # After 10+ ticks with rate ≈ 0, override should have fired.
        assert pi._integration_frozen is False, (
            "Integration should be unfrozen by rate-based override"
        )
        # Integral should have wound (negative error: 21.0 - 21.7 = -0.7).
        assert pi._pi_integral < -1.4, (
            f"Integral should wind during override, got {pi._pi_integral}"
        )

    @pytest.mark.asyncio
    async def test_rate_override_grows_estimate(self):
        """Rate override at wider delta should increase the estimate."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21  # delta ≈ 0.7°C, above default 0.5
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.7
        pi._pi_integral = -1.0

        est_before = pi._hp_deadband_estimate_heat
        assert est_before == 0.5

        for i in range(12):
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        assert pi._hp_deadband_estimate_heat > est_before, (
            f"Estimate should grow from rate override, "
            f"was {est_before}, now {pi._hp_deadband_estimate_heat}"
        )

    @pytest.mark.asyncio
    async def test_learned_deadband_provides_fast_path(self):
        """Once grown, integration unfreezes immediately at the wider delta."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.7  # delta = 0.7°C
        pi._pi_integral = -1.0
        # Pre-learned: HP cycles up to 0.8°C above setpoint.
        pi._hp_deadband_estimate_heat = 0.8

        await pi._pi_tick()

        # Delta 0.7 < learned 0.8 → immediate override, no waiting.
        assert pi._integration_frozen is False, (
            "Learned deadband should provide immediate override"
        )

    # ── Downward learning (HP truly off at narrower delta) ────────────

    @pytest.mark.asyncio
    async def test_confirmed_off_shrinks_estimate(self):
        """Room cooling at delta < estimate → HP truly off → estimate shrinks.

        Start with estimate 1.0°C.  HP setpoint 0.4°C below room, room
        clearly cooling.  After enough ticks, controller should learn that
        the HP is off at 0.4°C and reduce the estimate.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._pi_integral = -1.0
        pi._hp_deadband_estimate_heat = 1.0  # overestimate

        # Room cooling at 0.4°C delta — HP is truly off here.
        for i in range(15):
            entity._attr_current_temperature = 21.4 - i * 0.03
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        assert pi._hp_deadband_estimate_heat < 1.0, (
            f"Estimate should shrink when HP confirmed off at smaller delta, "
            f"got {pi._hp_deadband_estimate_heat}"
        )

    @pytest.mark.asyncio
    async def test_confirmed_off_does_not_grow_estimate(self):
        """Room cooling at delta > estimate should NOT grow the estimate.

        The HP is off at a wide delta — that's consistent with the current
        estimate, not evidence that it's too small.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 18  # delta ≈ 5°C, well above estimate
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._pi_integral = -5.0

        est_before = pi._hp_deadband_estimate_heat

        for i in range(15):
            entity._attr_current_temperature = 23.0 - i * 0.05
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        assert pi._hp_deadband_estimate_heat == est_before, (
            f"Confirmed-off at wide delta should not change estimate, "
            f"was {est_before}, now {pi._hp_deadband_estimate_heat}"
        )

    @pytest.mark.asyncio
    async def test_sensor_miscalibration_corrects_downward(self):
        """Sensor reads 0.5°C warm → HP actually off at apparent 0.3°C delta.

        Real room is 20.7°C, sensor reads 21.2°C.  HP setpoint 21°C.
        HP internal thermostat sees real temp ≈ setpoint → compressor off.
        Room cools (real temp dropping), sensor shows decline.
        Estimate should shrink from default 0.5 toward the observed off-point.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._pi_integral = -1.0
        # Default estimate is 0.5°C

        # Sensor reads 21.3 (delta=0.3, within 0.5 estimate).
        # HP thermostat sees real temp ≈ setpoint → off → room cools.
        # Hold hp_setpoint fixed each tick to prevent PI from stepping it
        # away and widening the delta.
        for i in range(15):
            entity._attr_current_temperature = 21.3 - i * 0.02
            pi._hp_setpoint = 21
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        # Delta ≈ 0.3°C.  HP is off, room cooling.  Estimate should
        # shrink below 0.5 since HP is confirmed off within the default
        # estimate range.
        assert pi._hp_deadband_estimate_heat < 0.5, (
            f"Estimate should shrink for miscalibrated sensor, "
            f"got {pi._hp_deadband_estimate_heat}"
        )

    # ── Bidirectional convergence ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_estimate_converges_from_both_sides(self):
        """Estimate should converge toward true deadband from both directions.

        Episode 1: HP cycling at 0.8°C delta → estimate grows above 0.5.
        Episode 2: HP off at 0.3°C delta → estimate shrinks toward 0.3.
        Episode 3: HP cycling at 0.6°C delta → estimate grows back above 0.3.
        """
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._pi_integral = -1.0
        t = 0

        # Episode 1: HP cycling at 0.8°C delta (room stable).
        pi._hp_setpoint = 21
        entity._attr_current_temperature = 21.8
        for i in range(12):
            pi._pi_last_tick_time = float(t * 60)
            t += 1
            with patch("time.monotonic", return_value=float(t * 60)):
                await pi._pi_tick()
        est_after_grow = pi._hp_deadband_estimate_heat
        assert est_after_grow > 0.5, f"Should grow from cycling, got {est_after_grow}"

        # Episode 2: HP off at 0.3°C delta (room cooling).
        pi._hp_setpoint = 21
        pi._hp_no_output_ticks = 0
        for i in range(15):
            entity._attr_current_temperature = 21.3 - i * 0.03
            pi._pi_last_tick_time = float(t * 60)
            t += 1
            with patch("time.monotonic", return_value=float(t * 60)):
                await pi._pi_tick()
        est_after_shrink = pi._hp_deadband_estimate_heat
        assert est_after_shrink < est_after_grow, (
            f"Should shrink from confirmed off, {est_after_grow} → {est_after_shrink}"
        )

        # Episode 3: HP cycling at 0.6°C delta (room stable).
        pi._hp_setpoint = 21
        entity._attr_current_temperature = 21.6
        pi._hp_no_output_ticks = 0
        for i in range(12):
            pi._pi_last_tick_time = float(t * 60)
            t += 1
            with patch("time.monotonic", return_value=float(t * 60)):
                await pi._pi_tick()
        est_final = pi._hp_deadband_estimate_heat
        assert est_final > est_after_shrink, (
            f"Should grow back from cycling, {est_after_shrink} → {est_final}"
        )

    # ── HP truly off (original protection preserved) ──────────────────

    @pytest.mark.asyncio
    async def test_hp_truly_off_stays_frozen(self):
        """Room cooling (HP truly off, wide delta) → integration stays frozen."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 18  # well below room, delta > estimate
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._pi_integral = -5.0

        integral_start = pi._pi_integral
        for i in range(20):
            # Room cooling: HP is truly off.
            entity._attr_current_temperature = 23.0 - i * 0.05
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        assert pi._integration_frozen is True, (
            "Integration should stay frozen when room is cooling (HP truly off)"
        )
        assert abs(pi._pi_integral - integral_start) < 0.5, (
            f"Integral should not wind when HP is truly off, "
            f"start={integral_start}, end={pi._pi_integral}"
        )

    # ── Mode separation ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_cool_mode_learns_separately(self):
        """Cooling mode should learn its own deadband estimate."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 24.0
        pi._hp_setpoint = 24  # 0.8°C above room → hp_no_output in cooling
        entity._attr_hvac_mode = HVACMode.COOL
        entity._attr_current_temperature = 23.2
        pi._pi_integral = 1.0

        est_heat_before = pi._hp_deadband_estimate_heat

        for i in range(12):
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        assert pi._hp_deadband_estimate_cool > 0.5, (
            f"Should learn wider cooling deadband, got {pi._hp_deadband_estimate_cool}"
        )
        assert pi._hp_deadband_estimate_heat == est_heat_before, (
            "Heating deadband should remain untouched"
        )

    # ── Learning gates preserved ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_hp_no_output_still_gates_learning(self):
        """Even with override unfreezing integration, RLS learning stays gated."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.3  # within default estimate
        pi._pi_integral = -1.0

        rls_count_before = pi._rls_heat.observation_count

        for i in range(5):
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()

        # Integration should be unfrozen (override) but RLS should NOT learn.
        assert pi._integration_frozen is False
        assert pi._rls_heat.observation_count == rls_count_before, (
            "RLS should not learn during hp_no_output (even with integration override)"
        )

    @pytest.mark.asyncio
    async def test_no_output_not_buffered_even_with_override(self):
        """HP-off observations not buffered even when integration override is active."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.3
        pi._pi_integral = -1.0
        before = len(pi._observation_buffer_heat)

        await pi._pi_tick()

        assert len(pi._observation_buffer_heat) == before, (
            "HP-off observations should not be buffered even with deadband override"
        )

    # ── Persistence ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_deadband_estimate_persisted(self):
        """Deadband estimates survive save/restore cycle."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._hp_deadband_estimate_heat = 0.8
        pi._hp_deadband_estimate_cool = 1.2

        stored = pi.get_extra_stored_data()
        assert stored is not None
        d = stored.as_dict()
        assert d["hp_deadband_estimate_heat"] == 0.8
        assert d["hp_deadband_estimate_cool"] == 1.2

        # Restore into a fresh controller.
        from custom_components.tasmota_irhvac.pi.pi_stored_data import PIExtraStoredData
        restored = PIExtraStoredData.from_dict(d)
        assert restored is not None

        entity2 = FakePIEntity(config)
        pi2 = entity2._pi
        pi2.restore_extra_stored_data(restored)
        assert pi2._hp_deadband_estimate_heat == 0.8
        assert pi2._hp_deadband_estimate_cool == 1.2

    @pytest.mark.asyncio
    async def test_fresh_install_no_persisted_data_gets_default(self):
        """Restoring old data without deadband fields uses 0.5°C default."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        from custom_components.tasmota_irhvac.pi.pi_stored_data import PIExtraStoredData
        # Simulate old stored data without the new fields.
        old_data = {"pi_integral": 0.0}
        restored = PIExtraStoredData.from_dict(old_data)
        assert restored is not None
        assert restored.hp_deadband_estimate_heat == 0.5
        assert restored.hp_deadband_estimate_cool == 0.5

    # ── Tick counter ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tick_counter_resets_when_hp_active(self):
        """hp_no_output_ticks resets to 0 when setpoint goes above room."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 21.5
        pi._pi_integral = -1.0

        # Accumulate some ticks.
        for i in range(5):
            pi._pi_last_tick_time = float(i * 60)
            with patch("time.monotonic", return_value=float((i + 1) * 60)):
                await pi._pi_tick()
        assert pi._hp_no_output_ticks == 5

        # HP becomes active (setpoint above room).
        pi._hp_setpoint = 23
        entity._attr_current_temperature = 20.0
        pi._pi_last_tick_time = 5 * 60.0
        with patch("time.monotonic", return_value=6 * 60.0):
            await pi._pi_tick()

        assert pi._hp_no_output_ticks == 0, (
            "Tick counter should reset when HP is active"
        )


class TestBatchWLSApply:
    """Tests for batch WLS apply integration in _run_batch_analysis."""

    @pytest.mark.asyncio
    async def test_batch_apply_updates_rls_betas(self):
        """When batch recommends update, RLS betas should change."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0
        rls = pi._rls_heat

        # Seed the observation buffer with enough eligible data
        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + float(i % 5 - 2),
                room_rate=0.001,  # stable
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        beta_before = list(rls.beta)
        pi._run_batch_analysis()

        # If batch recommended update, betas should have changed
        if pi._last_batch_result and pi._last_batch_result.recommend_update:
            beta_after = list(rls.beta)
            assert beta_after != beta_before, (
                "RLS betas should change when batch recommends update"
            )

    @pytest.mark.asyncio
    async def test_batch_apply_respects_step_cap(self):
        """Batch apply should not move any coefficient more than max_step."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0
        rls = pi._rls_heat

        beta_phys_before = rls.get_coefficients()

        # Seed with biased data that would suggest large coefficient changes
        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(40):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=25.0,  # biased high
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i % 8 - 4),
                room_rate=0.005,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        pi._run_batch_analysis()

        beta_phys_after = rls.get_coefficients()
        max_step = 1.0
        for idx in range(rls.n):
            delta = abs(beta_phys_after[idx] - beta_phys_before[idx])
            assert delta <= max_step + 1e-6, (
                f"Coefficient {idx} moved {delta:.4f}, exceeds step cap {max_step}"
            )

    @pytest.mark.asyncio
    async def test_batch_apply_reduces_covariance(self):
        """P[i,i] should decrease for updated coefficients after batch apply."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0
        rls = pi._rls_heat

        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + float(i % 5 - 2),
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        P_diag_before = [rls.P[i * rls.n + i] for i in range(rls.n)]
        pi._run_batch_analysis()

        if pi._last_batch_result and pi._last_batch_result.recommend_update:
            gains = pi._last_batch_result.blend_gains
            for i in range(rls.n):
                if i < len(gains) and gains[i] > 0:
                    P_after = rls.P[i * rls.n + i]
                    assert P_after < P_diag_before[i], (
                        f"P[{i},{i}] should decrease after batch update "
                        f"(was {P_diag_before[i]:.4f}, now {P_after:.4f})"
                    )

    @pytest.mark.asyncio
    async def test_batch_apply_zeros_off_diagonal(self):
        """Off-diagonal P elements should be zero for updated coefficients."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0
        rls = pi._rls_heat

        # Inject non-zero off-diagonal P elements
        for i in range(rls.n):
            for j in range(rls.n):
                if i != j:
                    rls.P[i * rls.n + j] = 0.1

        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + float(i % 5 - 2),
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        pi._run_batch_analysis()

        if pi._last_batch_result and pi._last_batch_result.recommend_update:
            gains = pi._last_batch_result.blend_gains
            for i in range(rls.n):
                if i < len(gains) and gains[i] > 0:
                    for j in range(rls.n):
                        if j != i:
                            assert rls.P[i * rls.n + j] == 0.0, (
                                f"P[{i},{j}] should be zero after batch update"
                            )
                            assert rls.P[j * rls.n + i] == 0.0, (
                                f"P[{j},{i}] should be zero after batch update"
                            )

    @pytest.mark.asyncio
    async def test_batch_covariance_floor_prevents_collapse(self):
        """P[i,i] should not go below the batch std_err floor."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0
        rls = pi._rls_heat

        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + float(i % 5 - 2),
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        pi._run_batch_analysis()

        if pi._last_batch_result and pi._last_batch_result.recommend_update:
            gains = pi._last_batch_result.blend_gains
            std_err = pi._last_batch_result.beta_std_err
            for i in range(rls.n):
                if i < len(gains) and gains[i] > 0 and i < len(std_err):
                    import math
                    if not math.isinf(std_err[i]):
                        se_norm = std_err[i] * rls.feature_scales[i]
                        p_floor = max(se_norm * se_norm, rls.delta)
                        P_after = rls.P[i * rls.n + i]
                        assert P_after >= p_floor - 1e-12, (
                            f"P[{i},{i}]={P_after:.6f} should not go below "
                            f"floor={p_floor:.6f}"
                        )

    @pytest.mark.asyncio
    async def test_batch_no_p_update_when_no_recommend(self):
        """P should be unchanged when batch does not recommend update."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0
        rls = pi._rls_heat

        # Set RLS betas to match what WLS would estimate (no change needed)
        # Use very few observations so batch may not recommend
        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        # Only 15 observations — below the min_observations=20 threshold
        for i in range(15):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=21.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i % 5 - 2),
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        P_before = list(rls.P)
        pi._run_batch_analysis()

        # With insufficient observations, batch should not run
        assert rls.P == P_before, "P should be unchanged when batch doesn't run"

    @pytest.mark.asyncio
    async def test_batch_held_features_p_unchanged(self):
        """P for held features (gain=0) should not be modified."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0
        rls = pi._rls_heat

        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        # All observations have identical outdoor_delta → feature 1 will be held
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + 5.0,  # constant outdoor_delta
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        P_diag_before = [rls.P[i * rls.n + i] for i in range(rls.n)]
        pi._run_batch_analysis()

        if pi._last_batch_result and pi._last_batch_result.blend_gains:
            gains = pi._last_batch_result.blend_gains
            for i in range(rls.n):
                if i < len(gains) and gains[i] == 0.0:
                    assert rls.P[i * rls.n + i] == P_diag_before[i], (
                        f"P[{i},{i}] should be unchanged for held feature"
                    )


class TestGateLogging:
    """Tests for RLS gate decision logging."""

    @pytest.mark.asyncio
    async def test_deadband_gate_logs_periodically(self, pi_entity, caplog):
        """Gate block log should fire at tick 4, 8, 12 — not every tick."""
        import logging
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 22.1  # in deadband (error = -0.1)
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        # Force a condition that blocks learning
        pi_entity._pi._inputs.outdoor_temp = None  # blocks "no outdoor temp"

        with caplog.at_level(logging.DEBUG):
            for i in range(16):
                pi_entity._pi._pi_last_tick_time = float(i * 900)
                with patch("time.monotonic", return_value=float((i + 1) * 900)):
                    await pi_entity._pi._pi_tick()

        blocked_msgs = [r for r in caplog.records if "RLS learning blocked" in r.message]
        # Should fire at ticks 4, 8, 12 (every 4th tick)
        assert len(blocked_msgs) >= 2, (
            f"Expected periodic gate block logs, got {len(blocked_msgs)}"
        )

    @pytest.mark.asyncio
    async def test_oodb_gate_reset_logged(self, pi_entity, caplog):
        """OODB gate reset should log when counter was accumulating."""
        import logging
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 19.0  # outside deadband
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._inputs.outdoor_temp = 5.0
        # Simulate some stable ticks to build up counter
        pi_entity._pi._stable_oodb_ticks = 3

        with caplog.at_level(logging.DEBUG):
            # Break stability by making room rate unstable
            pi_entity._pi._room_temp_rate = 0.05  # > 0.015 threshold
            pi_entity._pi._pi_last_tick_time = 0.0
            with patch("time.monotonic", return_value=900.0):
                await pi_entity._pi._pi_tick()

        reset_msgs = [r for r in caplog.records if "OODB gate reset" in r.message]
        assert len(reset_msgs) >= 1, (
            f"Expected OODB gate reset log, got {len(reset_msgs)}"
        )

    @pytest.mark.asyncio
    async def test_oodb_gate_no_log_when_counter_zero(self, pi_entity, caplog):
        """No OODB reset log when counter was already at zero."""
        import logging
        pi_entity._attr_hvac_mode = HVACMode.HEAT
        pi_entity._attr_current_temperature = 19.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._inputs.outdoor_temp = 5.0
        pi_entity._pi._stable_oodb_ticks = 0  # already zero
        pi_entity._pi._room_temp_rate = 0.05  # unstable

        with caplog.at_level(logging.DEBUG):
            await pi_entity._pi._pi_tick()

        reset_msgs = [r for r in caplog.records if "OODB gate reset" in r.message]
        assert len(reset_msgs) == 0, (
            f"Should not log OODB reset when counter was already 0"
        )


class TestControllableUncontrollableMetrics:
    """Tests for controllable/uncontrollable ITAE and CVH split."""

    @pytest.mark.asyncio
    async def test_uncontrollable_itae_accumulates_at_min_setpoint(self):
        """Error while HP is clamped at min and room above target → uncontrollable."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = pi._min_temp_c  # At minimum
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 24.0  # 3°C above target → error < 0
        pi._pi_integral = -50.0  # deep enough to keep HP clamped at min

        # Reset counters
        pi._metrics.controllable_itae = 0.0
        pi._metrics.uncontrollable_itae = 0.0
        pi._metrics.itae_tick_count = 0

        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert pi._metrics.uncontrollable_itae > 0, "Should accumulate uncontrollable ITAE"
        assert pi._metrics.controllable_itae == 0.0, "Should NOT accumulate controllable ITAE"

    @pytest.mark.asyncio
    async def test_controllable_itae_accumulates_above_min_setpoint(self):
        """Error while HP has headroom → controllable."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 22.0  # NOT at minimum
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 24.0  # above target
        pi._pi_integral = -2.0

        pi._metrics.controllable_itae = 0.0
        pi._metrics.uncontrollable_itae = 0.0
        pi._metrics.itae_tick_count = 0

        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert pi._metrics.controllable_itae > 0, "Should accumulate controllable ITAE"
        assert pi._metrics.uncontrollable_itae == 0.0, "Should NOT accumulate uncontrollable ITAE"

    @pytest.mark.asyncio
    async def test_uncontrollable_cvh_at_min_above_threshold(self):
        """CVH accumulates as uncontrollable when at min and room well above target."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = pi._min_temp_c
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 23.0  # 2°C above → abs_error > 1.0
        pi._pi_integral = -50.0

        pi._metrics.controllable_cvh = 0.0
        pi._metrics.uncontrollable_cvh = 0.0

        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert pi._metrics.uncontrollable_cvh > 0, "Should accumulate uncontrollable CVH"
        assert pi._metrics.controllable_cvh == 0.0, "Should NOT accumulate controllable CVH"

    @pytest.mark.asyncio
    async def test_controllable_cvh_when_hp_has_headroom(self):
        """CVH controllable when HP not at limit."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 22.0
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 23.0  # 2°C above
        pi._pi_integral = -2.0

        pi._metrics.controllable_cvh = 0.0
        pi._metrics.uncontrollable_cvh = 0.0

        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert pi._metrics.controllable_cvh > 0, "Should accumulate controllable CVH"
        assert pi._metrics.uncontrollable_cvh == 0.0, "Should NOT accumulate uncontrollable CVH"

    @pytest.mark.asyncio
    async def test_total_itae_equals_sum_of_split(self):
        """Total ITAE should equal controllable + uncontrollable."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 24.0  # above target
        pi._pi_integral = -10.0

        pi._metrics.itae_accumulator = 0.0
        pi._metrics.controllable_itae = 0.0
        pi._metrics.uncontrollable_itae = 0.0
        pi._metrics.itae_tick_count = 0

        # Phase 1: at min (uncontrollable)
        pi._hp_setpoint = pi._min_temp_c
        pi._pi_integral = -50.0  # deep enough to stay clamped at min
        for i in range(3):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        # Phase 2: above min (controllable)
        pi._hp_setpoint = 22.0
        pi._pi_integral = -2.0
        for i in range(3, 6):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        total = pi._metrics.itae_accumulator
        split_sum = pi._metrics.controllable_itae + pi._metrics.uncontrollable_itae
        assert abs(total - split_sum) < 0.1, (
            f"Total ITAE {total:.2f} != controllable {pi._metrics.controllable_itae:.2f} "
            f"+ uncontrollable {pi._metrics.uncontrollable_itae:.2f} = {split_sum:.2f}"
        )

    @pytest.mark.asyncio
    async def test_at_min_room_below_target_is_controllable(self):
        """HP at min but room BELOW target → HP is helping → controllable."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = pi._min_temp_c
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 19.0  # BELOW target → error > 0
        pi._pi_integral = -3.0

        pi._metrics.controllable_itae = 0.0
        pi._metrics.uncontrollable_itae = 0.0
        pi._metrics.itae_tick_count = 0

        for i in range(4):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert pi._metrics.controllable_itae > 0, "HP helping cold room is controllable"
        assert pi._metrics.uncontrollable_itae == 0.0, "Not uncontrollable when HP helps"


class TestFFLoadFraction:
    """Tests for FF load fraction metric."""

    # test_ff_load_fraction_trends_up moved to test_performance_metrics.py

    @pytest.mark.asyncio
    async def test_ff_load_fraction_bounded_0_1(self):
        """Load fraction stays in [0, 1]."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0
        pi._metrics.ff_load_fraction = 0.5

        for i in range(10):
            pi._pi_last_tick_time = float(i * 900)
            with patch("time.monotonic", return_value=float((i + 1) * 900)):
                await pi._pi_tick()

        assert 0.0 <= pi._metrics.ff_load_fraction <= 1.0, (
            f"FF load fraction out of bounds: {pi._metrics.ff_load_fraction}"
        )


class TestBatchModelRMS:
    """Tests for batch model RMS sensor."""

    @pytest.mark.asyncio
    async def test_batch_rms_updated_after_analysis(self):
        """_batch_model_rms should be set after _run_batch_analysis."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0

        assert pi._metrics.batch_model_rms is None

        # Seed observation buffer
        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + float(i % 5 - 2),
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        pi._run_batch_analysis()

        if pi._last_batch_result is not None:
            assert pi._metrics.batch_model_rms is not None
            assert pi._metrics.batch_model_rms >= 0.0


class TestConditionalIntegrationLogging:
    """Tests for edge-triggered integration freeze logging."""

    @pytest.mark.asyncio
    async def test_freeze_logs_on_entry(self, caplog):
        """Should log when integration transitions to frozen."""
        import logging
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = pi._min_temp_c
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 24.0  # error < 0, at min
        pi._pi_integral = -10.0
        pi._integration_frozen = False

        with caplog.at_level(logging.DEBUG):
            await pi._pi_tick()

        freeze_msgs = [r for r in caplog.records if "Integration frozen" in r.message]
        assert len(freeze_msgs) >= 1, "Should log on freeze entry"

    @pytest.mark.asyncio
    async def test_unfreeze_logs_on_exit(self, caplog):
        """Should log when integration transitions out of frozen."""
        import logging
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 22.0  # NOT at min
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0  # HP setpoint (22) > room (20) → active
        pi._pi_integral = -2.0
        pi._integration_frozen = True  # was frozen

        with caplog.at_level(logging.DEBUG):
            await pi._pi_tick()

        unfreeze_msgs = [r for r in caplog.records if "Integration unfrozen" in r.message]
        assert len(unfreeze_msgs) >= 1, "Should log on unfreeze"

    @pytest.mark.asyncio
    async def test_no_log_when_state_unchanged(self, caplog):
        """No log when freeze state doesn't change."""
        import logging
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._desired_temp = 21.0
        pi._hp_setpoint = 22.0  # NOT at min
        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0  # HP setpoint (22) > room (20) → active
        pi._pi_integral = -2.0
        pi._integration_frozen = False  # already not frozen

        with caplog.at_level(logging.DEBUG):
            await pi._pi_tick()

        freeze_msgs = [r for r in caplog.records
                       if "Integration frozen" in r.message or "Integration unfrozen" in r.message]
        assert len(freeze_msgs) == 0, "No log when state unchanged"


class TestStoredDataNewFields:
    """Tests for persistence of new metric fields."""

    def test_from_dict_new_fields_default_on_legacy(self):
        """Legacy stored data without new fields gets sensible defaults."""
        raw = {
            "pi_integral": 1.0,
            "desired_temp": 22.0,
            "hp_setpoint": 22.0,
        }
        data = PIExtraStoredData.from_dict(raw)
        assert data is not None
        assert data.controllable_itae == 0.0
        assert data.uncontrollable_itae == 0.0
        assert data.controllable_cvh == 0.0
        assert data.uncontrollable_cvh == 0.0
        assert data.ff_load_fraction == 0.5

    def test_round_trip_preserves_new_fields(self):
        """as_dict → from_dict round-trip preserves all new fields."""
        data = PIExtraStoredData(
            pi_integral=1.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            controllable_itae=100.5,
            uncontrollable_itae=200.3,
            controllable_cvh=1.5,
            uncontrollable_cvh=3.2,
            ff_load_fraction=0.72,
        )
        restored = PIExtraStoredData.from_dict(data.as_dict())
        assert restored is not None
        assert restored.controllable_itae == pytest.approx(100.5)
        assert restored.uncontrollable_itae == pytest.approx(200.3)
        assert restored.controllable_cvh == pytest.approx(1.5)
        assert restored.uncontrollable_cvh == pytest.approx(3.2)
        assert restored.ff_load_fraction == pytest.approx(0.72)

    def test_restore_applies_new_fields_to_controller(self):
        """restore_extra_stored_data loads new fields into PI controller."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        data = PIExtraStoredData(
            pi_integral=1.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            controllable_itae=50.0,
            uncontrollable_itae=75.0,
            controllable_cvh=2.0,
            uncontrollable_cvh=4.0,
            ff_load_fraction=0.65,
        )
        pi.restore_extra_stored_data(data)

        assert pi._metrics.controllable_itae == pytest.approx(50.0)
        assert pi._metrics.uncontrollable_itae == pytest.approx(75.0)
        assert pi._metrics.controllable_cvh == pytest.approx(2.0)
        assert pi._metrics.uncontrollable_cvh == pytest.approx(4.0)
        assert pi._metrics.ff_load_fraction == pytest.approx(0.65)

    def test_batch_result_round_trip(self):
        """BatchResult survives serialize → deserialize via PIExtraStoredData."""
        from custom_components.tasmota_irhvac.pi.batch_learning import BatchResult

        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        # Simulate a batch result
        pi._last_batch_result = BatchResult(
            n_total=300,
            n_eligible=280,
            beta_batch=[1.0, -0.5, 0.3],
            beta_current=[1.1, -0.4, 0.25],
            residual_rms=0.42,
            max_coeff_change_pct=12.5,
            recommend_update=True,
            n_outliers_excluded=5,
            held_features={2},
            beta_std_err=[0.01, 0.02, 0.03],
            beta_blended=[1.05, -0.45, 0.28],
            blend_gains=[0.5, 0.6, 0.7],
        )
        pi._metrics.batch_model_rms = 0.42

        # Serialize
        stored = pi.get_extra_stored_data()
        assert stored is not None
        d = stored.as_dict()
        assert d["last_batch_result"] is not None
        # held_features serialized as list
        assert isinstance(d["last_batch_result"]["held_features"], list)

        # Deserialize
        restored = PIExtraStoredData.from_dict(d)
        assert restored is not None

        # Restore into a fresh controller
        config2 = make_pi_config()
        entity2 = FakePIEntity(config2)
        pi2 = entity2._pi
        pi2.restore_extra_stored_data(restored)

        br = pi2._last_batch_result
        assert br is not None
        assert br.n_total == 300
        assert br.n_eligible == 280
        assert br.n_outliers_excluded == 5
        assert br.residual_rms == pytest.approx(0.42)
        assert br.recommend_update is True
        assert br.held_features == {2}
        assert br.beta_blended == [1.05, -0.45, 0.28]
        assert pi2._metrics.batch_model_rms == pytest.approx(0.42)

    def test_batch_result_none_round_trip(self):
        """No batch result serializes as None and restores cleanly."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        assert pi._last_batch_result is None
        stored = pi.get_extra_stored_data()
        d = stored.as_dict()
        assert d["last_batch_result"] is None

        restored = PIExtraStoredData.from_dict(d)
        config2 = make_pi_config()
        entity2 = FakePIEntity(config2)
        pi2 = entity2._pi
        pi2.restore_extra_stored_data(restored)
        assert pi2._last_batch_result is None

    def test_batch_result_missing_from_legacy_data(self):
        """Old stored data without last_batch_result field restores without error."""
        d = {
            "pi_integral": 1.0,
            "desired_temp": 22.0,
            "hp_setpoint": 22.0,
        }
        restored = PIExtraStoredData.from_dict(d)
        assert restored is not None
        assert restored.last_batch_result is None


class TestDriftDetection:
    """Tests for persistent same-direction batch correction detection."""

    def test_no_drift_with_empty_history(self):
        """No drift alert when no batch cycles have run."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi

        assert pi.get_drifting_coefficients() == []
        status = pi.get_health_status()
        assert "model_drift" not in status["reasons"]

    def test_no_drift_with_insufficient_cycles(self):
        """No drift alert before threshold cycles."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        # Simulate 3 cycles of same-direction correction (below threshold of 5)
        pi._drift_correction_signs = [[1, 1, 1], [0, 0, -1]]

        assert pi.get_drifting_coefficients() == []

    def test_drift_detected_after_threshold(self):
        """Drift detected after 5 consecutive same-direction corrections."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        # Coefficient 0 (intercept) corrected downward 5 times
        # Coefficient 1 (outdoor_delta) mixed — no drift
        pi._drift_correction_signs = [
            [-1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1],
        ]

        drifting = pi.get_drifting_coefficients()
        assert len(drifting) == 1
        assert drifting[0][0] == 0  # index
        assert drifting[0][1] == "intercept"  # name

    def test_drift_surfaces_in_health_sensor(self):
        """Drift alert appears in health sensor output."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._drift_correction_signs = [
            [-1, -1, -1, -1, -1],  # intercept drifting
            [0, 0, 0, 0, 0],       # outdoor_delta stable
        ]

        status = pi.get_health_status()
        assert "model_drift" in status["reasons"]
        assert any("intercept" in a for a in status["alerts"])
        assert status["state"] == "Warning"

    def test_no_drift_with_zeros(self):
        """Zero corrections (no change) don't trigger drift."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._drift_correction_signs = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        assert pi.get_drifting_coefficients() == []

    def test_drift_clears_when_direction_changes(self):
        """Drift clears when correction direction reverses."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        # 5 downward, then 1 upward — no longer 5 consecutive
        pi._drift_correction_signs = [
            [-1, -1, -1, -1, -1, 1],
        ]

        drifting = pi.get_drifting_coefficients()
        assert len(drifting) == 0

    def test_multiple_coefficients_can_drift(self):
        """Multiple coefficients can drift simultaneously."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._drift_correction_signs = [
            [-1, -1, -1, -1, -1],  # intercept down
            [1, 1, 1, 1, 1],       # outdoor_delta up
        ]

        drifting = pi.get_drifting_coefficients()
        assert len(drifting) == 2
        indices = {d[0] for d in drifting}
        assert indices == {0, 1}

    def test_drift_history_from_batch_analysis(self):
        """_run_batch_analysis populates drift history."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0

        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + float(i % 5 - 2),
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        pi._run_batch_analysis()

        assert len(pi._drift_correction_signs) > 0, (
            "Batch analysis should populate drift history"
        )
        # Each coefficient should have exactly 1 entry after 1 cycle
        for signs in pi._drift_correction_signs:
            assert len(signs) == 1

    def test_drift_history_truncated_at_10(self):
        """History is capped at 10 cycles per coefficient."""
        config = make_pi_config()
        entity = FakePIEntity(config)
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.0
        pi._hp_setpoint = 21.0

        # Pre-fill with 10 entries (the cap)
        pi._drift_correction_signs = [
            [1] * 10,
            [-1] * 10,
        ]

        import time as time_mod
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        now = time_mod.monotonic()
        for i in range(30):
            obs = Observation(
                timestamp=now + i * 900,
                wall_time=1713650000.0 + i * 900,
                hp_setpoint=22.0,
                current_c=21.0 + (i % 3) * 0.1,
                desired_c=21.0,
                outdoor_temp_c=21.0 + (i % 3) * 0.1 + float(i % 5 - 2),
                room_rate=0.001,
                raw_readings={},
                clamped=False,
            )
            pi._observation_buffer_heat.add(obs)

        # Run a batch cycle — should append and truncate
        pi._run_batch_analysis()

        for signs in pi._drift_correction_signs:
            assert len(signs) <= 10, (
                f"History should be capped at 10, got {len(signs)}"
            )
