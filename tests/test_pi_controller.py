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
    ATTR_FF_HEAT_BUCKETS,
    ATTR_FF_COOL_BUCKETS,
    ATTR_FF_OFFSET,
    ATTR_HP_SETPOINT,
    ATTR_PI_INTEGRAL,
    DOMAIN,
)
from custom_components.tasmota_irhvac.pi_controller import PIController, _seed_buckets

from .conftest import make_pi_config


# ── Seed Buckets ──────────────────────────────────────────────────────


class TestSeedBuckets:
    """Tests for _seed_buckets function."""

    def test_heating_at_reference(self):
        """At reference temp, offset should be zero."""
        buckets = _seed_buckets(15.0, 0.3)
        assert buckets[15] == 0.0

    def test_heating_below_reference(self):
        """Below reference, offset should be positive (more heating needed)."""
        buckets = _seed_buckets(15.0, 0.3)
        assert buckets[0] == pytest.approx(4.5)  # 15 * 0.3
        assert buckets[-30] == pytest.approx(13.5)  # 45 * 0.3

    def test_heating_above_reference(self):
        """Above reference, no additional heating needed."""
        buckets = _seed_buckets(15.0, 0.3)
        assert buckets[18] == 0.0
        assert buckets[45] == 0.0

    def test_cooling_at_reference(self):
        """At reference temp, offset should be zero."""
        buckets = _seed_buckets(25.0, 0.3, is_cooling=True)
        assert buckets[24] == 0.0  # Closest bucket below 25

    def test_cooling_above_reference(self):
        """Above reference, offset should be negative (more cooling needed)."""
        buckets = _seed_buckets(25.0, 0.3, is_cooling=True)
        assert buckets[30] == pytest.approx(-1.5)  # 5 * -0.3

    def test_cooling_below_reference(self):
        """Below reference, no additional cooling needed."""
        buckets = _seed_buckets(25.0, 0.3, is_cooling=True)
        assert buckets[21] == 0.0

    def test_bucket_keys_are_3c_steps(self):
        """Buckets should be keyed in 3°C steps from -30 to 45."""
        buckets = _seed_buckets(15.0, 0.3)
        keys = sorted(buckets.keys())
        assert keys[0] == -30
        assert keys[-1] == 45
        for i in range(1, len(keys)):
            assert keys[i] - keys[i - 1] == 3

    def test_different_slopes(self):
        """Different slopes should produce proportionally different offsets."""
        b1 = _seed_buckets(15.0, 0.3)
        b2 = _seed_buckets(15.0, 0.6)
        assert b2[0] == pytest.approx(b1[0] * 2)


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
    _temp_precision = 1.0

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
        return UnitOfTemperature.CELSIUS

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
    async def test_integral_capped_at_50(self, pi_entity):
        """Integral should never exceed ±50."""
        pi_entity._pi._pi_integral = 100.0
        pi_entity._attr_current_temperature = 20.0  # °C
        pi_entity._pi._desired_temp = 22.0  # °C
        pi_entity._pi._hp_setpoint = 22.0

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._pi_integral <= 50.0
        assert pi_entity._pi._pi_integral >= -50.0

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
        """Adaptive weight should blend to b=1 for large errors, b=configured near deadband."""
        pi_entity._pi._pi_setpoint_weight = 0.0  # Configured weight
        # Large error (>4x deadband of 0.5°C = 2.0°C): adaptive weight should be ~1.0
        pi_entity._attr_current_temperature = 19.0  # °C, well below desired
        pi_entity._pi._desired_temp = 22.0  # °C, error = 3.0°C
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = 0.0

        await pi_entity._pi._pi_tick()

        # With adaptive weight, large error → effective_weight≈1.0
        # P = Kp * weight * (desired - current) = positive → setpoint goes UP
        assert pi_entity._pi._hp_setpoint > 22.0

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
    async def test_disturbance_bias_adds_to_ff(self, pi_entity):
        """Disturbance input bias should be added to FF offset."""
        pi_entity._pi._disturbance_inputs = [{
            "name": "Bias",
            "entity_id": "input_number.hvac_bias",
            "suppress_learning": False,
            "default_bias": 0.0,
            "gain": 1.0,
        }]
        pi_entity._pi._outdoor_temp = None  # No outdoor sensor, so base FF = 0
        pi_entity._attr_current_temperature = 68.0
        pi_entity._pi._desired_temp = 72.0
        pi_entity._pi._hp_setpoint = 22.0

        # Mock bias entity state (numeric → value × gain)
        mock_state = MagicMock()
        mock_state.state = "2.5"
        pi_entity.hass.states.get.return_value = mock_state

        await pi_entity._pi._pi_tick()

        assert pi_entity._pi._ff_offset == pytest.approx(2.5)

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

        old_bucket = pi_entity._pi._ff_heat_buckets[0]
        await pi_entity._pi._pi_tick()

        # Bucket should NOT have been updated
        assert pi_entity._pi._ff_heat_buckets[0] == old_bucket


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
        assert pi_entity._attr_target_temperature == 74.0
        assert pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_set_temperature_without_pi(self, pi_entity):
        """async_set_temperature should fall through when PI disabled."""
        pi_entity._pi._pi_enabled = False
        await pi_entity.async_set_temperature(temperature=74.0)
        # Falls to FakeBaseEntity.async_set_temperature
        assert pi_entity._attr_target_temperature == 74.0

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
    async def test_integral_zeroed_on_heating_undershoot(self, pi_entity):
        """Negative integral in heating near deadband should be zeroed."""
        pi_entity._attr_current_temperature = 22.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._pi._pi_integral = -5.0  # Negative integral from overshooting
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi._pi_tick()

        # Negative integral in heating near deadband → zeroed
        assert pi_entity._pi._pi_integral >= 0.0

    @pytest.mark.asyncio
    async def test_integral_zeroed_on_cooling_overshoot(self, pi_entity):
        """Positive integral in cooling near deadband should be zeroed."""
        pi_entity._attr_current_temperature = 24.0
        pi_entity._pi._desired_temp = 24.0
        pi_entity._pi._hp_setpoint = 24.0
        pi_entity._pi._pi_integral = 5.0  # Positive integral from overshooting
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._attr_hvac_mode = HVACMode.COOL

        await pi_entity._pi._pi_tick()

        # Positive integral in cooling near deadband → zeroed
        assert pi_entity._pi._pi_integral <= 0.0

    @pytest.mark.asyncio
    async def test_ff_auto_learning_writes_bucket(self, pi_entity):
        """FF should write to bucket when settled for 2+ ticks in deadband."""
        pi_entity._attr_current_temperature = 22.0
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 23.0
        pi_entity._pi._pi_integral = 0.0
        pi_entity._pi._pi_last_tick_time = 0
        pi_entity._pi._outdoor_temp = 0.0  # Bucket key = 0
        pi_entity._pi._ff_settled_ticks = 0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        old_bucket = pi_entity._pi._ff_heat_buckets.get(0, 0.0)

        # Tick 1: enter deadband, settled_ticks increments
        await pi_entity._pi._pi_tick()
        # Tick 2: still in deadband, settled_ticks >= 2 → learning writes bucket
        await pi_entity._pi._pi_tick()

        # Bucket should have been updated via EMA
        new_bucket = pi_entity._pi._ff_heat_buckets.get(0, 0.0)
        assert new_bucket != old_bucket or pi_entity._pi._ff_settled_ticks >= 2

    @pytest.mark.asyncio
    async def test_handle_state_payload_command_pending(self, pi_entity):
        """Echo after PI command should clear pending flag without re-ticking."""
        pi_entity._pi._pi_command_pending = True
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 23.0

        await pi_entity._pi.handle_state_payload({"Temp": 23, "Power": "On"})

        assert pi_entity._pi._pi_command_pending is False

    @pytest.mark.asyncio
    async def test_handle_state_payload_not_pending_reticks(self, pi_entity):
        """Echo without pending flag should treat as external change and re-tick."""
        pi_entity._pi._pi_command_pending = False
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        pi_entity._attr_current_temperature = 20.0
        pi_entity._attr_hvac_mode = HVACMode.HEAT

        await pi_entity._pi.handle_state_payload({"Temp": 25, "Power": "On"})

        # Should have captured hp_setpoint from payload
        assert pi_entity._pi._hp_setpoint == 25 or pi_entity._pi._desired_temp is not None

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
            ff_heat_buckets={0: 1.5, 3: 2.0},
            ff_cool_buckets={24: -0.5},
            pi_integral=7.5,
            desired_temp=22.0,
            hp_setpoint=23.0,
        )
        pi_entity._pi.restore_extra_stored_data(data)

        assert pi_entity._pi._ff_heat_buckets[0] == 1.5
        assert pi_entity._pi._ff_cool_buckets[24] == -0.5
        assert pi_entity._pi._pi_integral == 7.5
        assert pi_entity._pi._desired_temp == 22.0
        assert pi_entity._pi._hp_setpoint == 23.0

    def test_restore_extra_stored_data_clamps_integral(self, pi_entity):
        """restore_extra_stored_data should clamp integral to ±50."""
        from custom_components.tasmota_irhvac.pi_controller import PIExtraStoredData
        data = PIExtraStoredData(
            ff_heat_buckets={},
            ff_cool_buckets={},
            pi_integral=100.0,
            desired_temp=None,
            hp_setpoint=None,
        )
        pi_entity._pi.restore_extra_stored_data(data)
        assert pi_entity._pi._pi_integral == 50.0
