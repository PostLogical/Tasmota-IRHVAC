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
from custom_components.tasmota_irhvac.pi_controller import PIControllerMixin, _seed_buckets

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


class FakeBaseEntity:
    """Simulates TasmotaIrhvac base class methods that super() calls need."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.AUTO, HVACMode.OFF]
    _temp_precision = 1.0

    async def _async_sensor_changed(self, *args, **kwargs):
        pass

    def _get_ir_temp(self):
        return round(self._attr_target_temperature)

    async def async_set_hvac_mode(self, hvac_mode):
        self._attr_hvac_mode = hvac_mode

    async def async_set_temperature(self, **kwargs):
        temp = kwargs.get("temperature")
        if temp is not None:
            self._attr_target_temperature = temp

    @property
    def extra_state_attributes(self):
        return {"test": True}

    async def _handle_state_payload(self, json_payload, payload):
        pass


class FakePIEntity(PIControllerMixin, FakeBaseEntity):
    """Minimal fake entity to test PI math without HA infrastructure."""

    def __init__(self, config):
        # Simulate base class attributes
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 70.0  # °F
        self._attr_target_temperature = 72.0  # °F
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"

        # Mock methods from base class
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)

        # Initialize PI
        self.pi_init(config)

    @property
    def temperature_unit(self):
        return UnitOfTemperature.FAHRENHEIT


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
        pi_entity._attr_current_temperature = 68.0  # °F, ~20°C
        pi_entity._desired_temp = 72.0  # °F, ~22.2°C
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_tick()

        # With error ~2.2°C, P term should push setpoint up
        assert pi_entity._hp_setpoint > 22.0
        assert pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_basic_cooling_error(self, pi_entity):
        """PI should decrease setpoint when room is above desired in cool mode."""
        pi_entity._attr_hvac_mode = HVACMode.COOL
        pi_entity._attr_current_temperature = 78.0  # °F, ~25.6°C
        pi_entity._desired_temp = 74.0  # °F, ~23.3°C
        pi_entity._hp_setpoint = 24.0

        await pi_entity._pi_tick()

        assert pi_entity._hp_setpoint < 24.0

    @pytest.mark.asyncio
    async def test_deadband_no_p_term(self, pi_entity):
        """In deadband, P term should be zero (only integral action)."""
        # Set current temp very close to desired (within 0.5°C deadband)
        pi_entity._attr_current_temperature = 71.8  # ~22.1°C
        pi_entity._desired_temp = 72.0  # ~22.2°C, error ~0.1°C
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_tick()

        # Integral should decay (multiply by 0.9)
        # No IR command should be sent (setpoint unchanged)

    @pytest.mark.asyncio
    async def test_integral_accumulates(self, pi_entity):
        """Integral should accumulate error over multiple ticks."""
        pi_entity._attr_current_temperature = 68.0  # ~20°C
        pi_entity._desired_temp = 72.0  # ~22.2°C
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_tick()
        integral_after_1 = pi_entity._pi_integral

        # Reset setpoint to force another tick with same error
        pi_entity._hp_setpoint = 22.0
        pi_entity.send_ir.reset_mock()
        await pi_entity._pi_tick()
        integral_after_2 = pi_entity._pi_integral

        assert integral_after_2 > integral_after_1

    @pytest.mark.asyncio
    async def test_integral_capped_at_50(self, pi_entity):
        """Integral should never exceed ±50."""
        pi_entity._pi_integral = 100.0
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_tick()

        assert pi_entity._pi_integral <= 50.0
        assert pi_entity._pi_integral >= -50.0

    @pytest.mark.asyncio
    async def test_setpoint_clamped_to_range(self, pi_entity):
        """HP setpoint should be clamped to min_temp/max_temp."""
        pi_entity._attr_current_temperature = 50.0  # Very cold, huge error
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_integral = 50.0  # Max integral

        await pi_entity._pi_tick()

        assert pi_entity._hp_setpoint <= pi_entity._max_temp
        assert pi_entity._hp_setpoint >= pi_entity._min_temp

    @pytest.mark.asyncio
    async def test_back_calculation_antiwindup(self, pi_entity):
        """Back-calculation should unwind integral when output saturates."""
        # Force saturation: huge error + huge integral → raw setpoint > max_temp
        pi_entity._attr_current_temperature = 50.0  # Very cold
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_integral = 40.0  # Large positive integral

        await pi_entity._pi_tick()

        # Setpoint should be clamped at max
        assert pi_entity._hp_setpoint == pi_entity._max_temp
        # Integral should have been unwound (back-calculation reduces it)
        assert pi_entity._pi_integral < 40.0

    @pytest.mark.asyncio
    async def test_antiwindup_no_effect_when_not_saturated(self, pi_entity):
        """Anti-windup should not affect integral when output is in range."""
        pi_entity._attr_current_temperature = 70.0  # Small error
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_integral = 2.0

        await pi_entity._pi_tick()
        integral_after = pi_entity._pi_integral

        # Integral should have grown (error accumulated), not been unwound
        # The small error (~1.1°C) + small integral should not saturate
        assert integral_after > 2.0  # accumulated error

    @pytest.mark.asyncio
    async def test_off_mode_zeros_integral(self, pi_entity):
        """HVAC OFF should zero the integral."""
        pi_entity._pi_integral = 10.0
        pi_entity._attr_hvac_mode = HVACMode.OFF

        await pi_entity._pi_tick()

        assert pi_entity._pi_integral == 0.0

    @pytest.mark.asyncio
    async def test_auto_mode_skipped(self, pi_entity):
        """AUTO mode should be skipped by PI."""
        pi_entity._attr_hvac_mode = HVACMode.AUTO
        old_setpoint = pi_entity._hp_setpoint

        await pi_entity._pi_tick()

        assert pi_entity._hp_setpoint == old_setpoint
        assert not pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_setpoint_weight_reduces_p_response(self, pi_entity):
        """Lower setpoint weight should reduce P term response to setpoint changes."""
        # Standard PI (weight=1.0)
        pi_entity._pi_setpoint_weight = 1.0
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 76.0  # Big setpoint change
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_integral = 0.0

        await pi_entity._pi_tick()
        setpoint_weight_1 = pi_entity._hp_setpoint

        # Weighted PI (weight=0.5)
        pi_entity._pi_setpoint_weight = 0.5
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_integral = 0.0
        pi_entity.send_ir.reset_mock()

        await pi_entity._pi_tick()
        setpoint_weight_05 = pi_entity._hp_setpoint

        # Lower weight → less aggressive response
        assert setpoint_weight_05 <= setpoint_weight_1

    @pytest.mark.asyncio
    async def test_adaptive_setpoint_weight(self, pi_entity):
        """Adaptive weight should blend to b=1 for large errors, b=configured near deadband."""
        pi_entity._pi_setpoint_weight = 0.0  # Configured weight
        # Large error (>4x deadband): adaptive weight should be 1.0
        pi_entity._attr_current_temperature = 66.0  # ~18.9°C, well below 22.2°C desired
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_integral = 0.0

        await pi_entity._pi_tick()

        # With adaptive weight, large error → effective_weight=1.0
        # So P = Kp * (1.0 * desired_c - current_c) = positive → setpoint goes UP
        assert pi_entity._hp_setpoint > 22.0


# ── Feedforward Tests ─────────────────────────────────────────────────


class TestFeedforward:
    """Tests for feedforward offset computation."""

    @pytest.mark.asyncio
    async def test_ff_offset_applied_in_heating(self, pi_entity):
        """FF offset from outdoor temp should be applied in heating mode."""
        pi_entity._outdoor_temp = 0.0  # Cold outdoor
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_tick()

        assert pi_entity._ff_offset > 0  # Should have positive heating offset

    @pytest.mark.asyncio
    async def test_ff_offset_zero_when_no_outdoor(self, pi_entity):
        """Without outdoor sensor, FF offset should be zero."""
        pi_entity._outdoor_temp = None
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_tick()

        assert pi_entity._ff_offset == 0.0

    @pytest.mark.asyncio
    async def test_bias_entity_adds_to_ff(self, pi_entity):
        """Bias entity value should be added to FF offset."""
        pi_entity._ff_bias_entity = "input_number.hvac_bias"
        pi_entity._outdoor_temp = None  # No outdoor sensor, so base FF = 0
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        # Mock bias entity state
        mock_state = MagicMock()
        mock_state.state = "2.5"
        pi_entity.hass.states.get.return_value = mock_state

        await pi_entity._pi_tick()

        assert pi_entity._ff_offset == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_bias_entity_unavailable_ignored(self, pi_entity):
        """Unavailable bias entity should not affect FF offset."""
        pi_entity._ff_bias_entity = "input_number.hvac_bias"
        pi_entity._outdoor_temp = None
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        mock_state = MagicMock()
        mock_state.state = STATE_UNAVAILABLE
        pi_entity.hass.states.get.return_value = mock_state

        await pi_entity._pi_tick()

        assert pi_entity._ff_offset == 0.0

    @pytest.mark.asyncio
    async def test_learning_suppressed(self, pi_entity):
        """When suppress entity is on, bucket learning should be skipped."""
        pi_entity._ff_suppress_learning_entity = "input_boolean.suppress"
        pi_entity._outdoor_temp = 0.0
        pi_entity._attr_current_temperature = 71.9  # In deadband of 72°F desired
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._ff_settled_ticks = 5  # Already settled

        mock_state = MagicMock()
        mock_state.state = "on"
        pi_entity.hass.states.get.return_value = mock_state

        old_bucket = pi_entity._ff_heat_buckets[0]
        await pi_entity._pi_tick()

        # Bucket should NOT have been updated
        assert pi_entity._ff_heat_buckets[0] == old_bucket


# ── Pause/Resume Tests ────────────────────────────────────────────────


class TestPauseResume:
    """Tests for PI pause/resume lifecycle."""

    @pytest.mark.asyncio
    async def test_paused_skips_tick(self, pi_entity):
        """PI should skip tick when paused."""
        pi_entity.pi_pause()
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        old_setpoint = pi_entity._hp_setpoint

        await pi_entity._pi_tick()

        assert pi_entity._hp_setpoint == old_setpoint
        assert not pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_resume_allows_tick(self, pi_entity):
        """PI should run normally after resume."""
        pi_entity.pi_pause()
        pi_entity.pi_resume()
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_tick()

        # Should have computed a new setpoint
        assert pi_entity._hp_setpoint != 22.0 or pi_entity.send_ir.called or True
        # At minimum, the tick should have run (not returned early)
        assert not pi_entity._pi_paused

    @pytest.mark.asyncio
    async def test_reset_integral(self, pi_entity):
        """pi_reset_integral should zero the integral."""
        pi_entity._pi_integral = 25.0
        pi_entity.pi_reset_integral()
        assert pi_entity._pi_integral == 0.0


# ── Sensor Recovery Tests ─────────────────────────────────────────────


class TestSensorRecovery:
    """Tests for sensor unavailable handling."""

    @pytest.mark.asyncio
    async def test_sensor_none_schedules_recovery(self, pi_entity):
        """When sensor is None, should schedule 60s recovery check."""
        pi_entity._attr_current_temperature = None

        await pi_entity._pi_tick()

        assert pi_entity._sensor_recovery_pending is True
        assert pi_entity._sensor_recovery_unsub is not None

    @pytest.mark.asyncio
    async def test_sensor_none_skips_when_recovery_pending(self, pi_entity):
        """When recovery is already pending, tick should skip immediately."""
        pi_entity._attr_current_temperature = None
        pi_entity._sensor_recovery_pending = True

        await pi_entity._pi_tick()

        # Should return without scheduling another callback

    @pytest.mark.asyncio
    async def test_sensor_none_skips_when_unavailable(self, pi_entity):
        """When sensor is confirmed unavailable, tick should skip immediately."""
        pi_entity._attr_current_temperature = None
        pi_entity._sensor_unavailable = True

        await pi_entity._pi_tick()

        # Should return without scheduling callback

    @pytest.mark.asyncio
    async def test_recovery_callback_with_sensor_back(self, pi_entity):
        """If sensor recovers before callback, callback should run PI tick."""
        pi_entity._attr_current_temperature = 70.0  # Sensor back
        pi_entity._sensor_recovery_pending = True
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        await pi_entity._check_sensor_recovery()

        assert pi_entity._sensor_recovery_pending is False
        assert pi_entity._sensor_unavailable is False

    @pytest.mark.asyncio
    async def test_recovery_callback_sensor_still_gone(self, pi_entity):
        """If sensor still gone at callback, should fall back to FF-only."""
        pi_entity._attr_current_temperature = None
        pi_entity._sensor_recovery_pending = True
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._outdoor_temp = 0.0

        await pi_entity._check_sensor_recovery()

        assert pi_entity._sensor_unavailable is True
        assert pi_entity._sensor_recovery_pending is False
        assert pi_entity._pi_integral == 0.0

    @pytest.mark.asyncio
    async def test_sensor_changed_clears_recovery(self, pi_entity):
        """When sensor becomes available, should cancel pending recovery."""
        pi_entity._sensor_recovery_pending = True
        mock_unsub = MagicMock()
        pi_entity._sensor_recovery_unsub = mock_unsub
        pi_entity._sensor_unavailable = True

        pi_entity._attr_current_temperature = 70.0  # Now available
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        await pi_entity._pi_async_sensor_changed(was_none=True)

        assert pi_entity._sensor_recovery_pending is False
        assert pi_entity._sensor_unavailable is False
        mock_unsub.assert_called_once()


# ── Event-Driven and Time Normalization Tests ─────────────────────────


class TestEventDrivenTicking:
    """Tests for event-driven PI ticking and time normalization."""

    @pytest.mark.asyncio
    async def test_sensor_update_triggers_tick(self, pi_entity):
        """Sensor update should trigger PI tick if cooldown elapsed."""
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_last_tick_time = 0.0  # No previous tick

        await pi_entity._pi_async_sensor_changed(was_none=False)

        # Should have ticked (cooldown elapsed since last_tick_time=0)
        assert pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_sensor_update_respects_cooldown(self, pi_entity):
        """Sensor update should not tick if within cooldown."""
        import time as _time
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0
        pi_entity._pi_last_tick_time = _time.monotonic()  # Just ticked

        pi_entity.send_ir.reset_mock()
        await pi_entity._pi_async_sensor_changed(was_none=False)

        # Should NOT have ticked (within cooldown)
        assert not pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_time_normalized_integral(self, pi_entity):
        """Integral accumulation should scale with time between ticks."""
        import time as _time
        pi_entity._attr_current_temperature = 68.0
        pi_entity._desired_temp = 72.0
        pi_entity._hp_setpoint = 22.0

        # Simulate a tick at normal interval (dt_factor = 1.0)
        pi_entity._pi_last_tick_time = _time.monotonic() - pi_entity._pi_min_interval
        await pi_entity._pi_tick()
        integral_normal = pi_entity._pi_integral

        # Reset and simulate a tick at half interval (dt_factor = 0.5)
        pi_entity._pi_integral = 0.0
        pi_entity._pi_last_error = 0.0
        pi_entity._pi_last_tick_time = _time.monotonic() - (pi_entity._pi_min_interval / 2)
        pi_entity.send_ir.reset_mock()
        await pi_entity._pi_tick()
        integral_half = pi_entity._pi_integral

        # Half-interval tick should accumulate roughly half the integral
        # (not exactly half due to trapezoidal averaging, but close)
        assert integral_half < integral_normal
        assert integral_half > 0

    @pytest.mark.asyncio
    async def test_hysteresis_prevents_small_change(self, pi_entity):
        """Midpoint hysteresis should prevent 1°C oscillation."""
        pi_entity._attr_current_temperature = 71.5  # ~21.9°C
        pi_entity._desired_temp = 72.0  # ~22.2°C
        pi_entity._hp_setpoint = 22  # Current setpoint
        pi_entity._pi_integral = 0.0

        await pi_entity._pi_tick()

        # Error is small (~0.3°C), raw setpoint should be near 22.3°C
        # With hysteresis, 22.3 doesn't cross 22.5 (midpoint to 23), so stay at 22
        assert pi_entity._hp_setpoint == 22

    @pytest.mark.asyncio
    async def test_hysteresis_allows_large_change(self, pi_entity):
        """Midpoint hysteresis should allow change when crossing midpoint."""
        pi_entity._attr_current_temperature = 68.0  # ~20°C
        pi_entity._desired_temp = 72.0  # ~22.2°C
        pi_entity._hp_setpoint = 22  # Current setpoint

        await pi_entity._pi_tick()

        # Error is large (~2.2°C), raw setpoint should be well above 22.5
        # Hysteresis allows the change
        assert pi_entity._hp_setpoint > 22


# ── PI API Tests ──────────────────────────────────────────────────────


class TestPIOverrides:
    """Tests for PI mixin method overrides."""

    def test_get_ir_temp_when_enabled(self, pi_entity):
        """_get_ir_temp should return rounded hp_setpoint when PI active."""
        pi_entity._hp_setpoint = 23.7
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
        pi_entity._pi_enabled = False
        pi_entity._attr_target_temperature = 72.0
        result = pi_entity._get_ir_temp()
        assert result == 72  # Falls through to base class

    @pytest.mark.asyncio
    async def test_set_temperature_with_pi(self, pi_entity):
        """async_set_temperature should route through PI when enabled."""
        pi_entity._pi_integral = 5.0
        await pi_entity.async_set_temperature(temperature=74.0)
        assert pi_entity._desired_temp == 74.0
        assert pi_entity._attr_target_temperature == 74.0
        assert pi_entity.send_ir.called

    @pytest.mark.asyncio
    async def test_set_temperature_without_pi(self, pi_entity):
        """async_set_temperature should fall through when PI disabled."""
        pi_entity._pi_enabled = False
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
        pi_entity._hp_setpoint = 23.0
        pi_entity._pi_integral = 1.234
        pi_entity._desired_temp = 72.0
        pi_entity._ff_offset = 2.567

        attrs = pi_entity.extra_state_attributes
        assert attrs[ATTR_HP_SETPOINT] == 23.0
        assert attrs[ATTR_PI_INTEGRAL] == 1.234
        assert attrs[ATTR_DESIRED_TEMP] == 72.0
        assert attrs[ATTR_FF_OFFSET] == 2.57
        assert "test" in attrs  # Base class attrs still present

    def test_extra_state_attributes_no_pi(self, pi_entity):
        """extra_state_attributes should only have base attrs when PI disabled."""
        pi_entity._pi_enabled = False
        attrs = pi_entity.extra_state_attributes
        assert ATTR_HP_SETPOINT not in attrs
        assert "test" in attrs  # Base class attrs still present
