"""Tests for FF learning improvements: alpha, seed protection, asymmetric, night-only, anticipated change."""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature

from custom_components.tasmota_irhvac.pi_controller import PIController, PIExtraStoredData

from .conftest import make_pi_config


class FakeLearningEntity:
    """Minimal fake entity for learning tests."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    def __init__(self, config):
        self.hass = MagicMock()
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 22.0
        self._attr_target_temperature = 22.0
        self._temp_sensor = "sensor.room_temp"
        self._min_temp = 16
        self._max_temp = 30
        self.power_mode = STATE_ON
        self._mqtt_delay = "0"
        self._config_entry_id = "test_entry"
        self.send_ir = AsyncMock()
        self.async_schedule_update_ha_state = MagicMock()
        self.async_write_ha_state = MagicMock()
        self.async_get_last_state = AsyncMock(return_value=None)
        self.async_get_last_extra_data = AsyncMock(return_value=None)
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return UnitOfTemperature.CELSIUS


def _make_config(**overrides):
    """Build PI config with learning-friendly defaults."""
    defaults = {
        "pi_ff_learn_night_only": False,  # Disable night-only for most tests
        "pi_ff_learn_sunset_delay": 0,
    }
    defaults.update(overrides)
    return make_pi_config(defaults)


def _settled_tick(entity, outdoor_temp=0.0, current=22.0, desired=22.0):
    """Set up entity state for a learning-eligible tick (in deadband, settled)."""
    entity._attr_current_temperature = current
    entity._pi._desired_temp = desired
    entity._pi._hp_setpoint = round(desired)
    entity._pi._pi_integral = 0.0
    entity._pi._outdoor_temp = outdoor_temp
    entity._pi._ff_settled_ticks = 5  # Already settled
    entity._pi._pi_last_tick_time = 0


# ── Alpha Tests ──────────────────────────────────────────────────────


class TestAlpha:
    """Verify learning rate is 0.05 not 0.2."""

    @pytest.mark.asyncio
    async def test_alpha_is_005(self):
        """Default alpha should be 0.05."""
        entity = FakeLearningEntity(_make_config())
        assert entity._pi._ff_alpha == 0.05

    @pytest.mark.asyncio
    async def test_single_observation_shifts_bucket_slowly(self):
        """One observation should only shift bucket by alpha * delta."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        # Pre-seed bucket and mark as past time threshold
        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600  # 5 hours ago

        _settled_tick(entity, outdoor_temp=0.0, current=22.0, desired=22.0)
        pi._hp_setpoint = 25.0  # Observed offset = 25 + 0 - 22 = 3.0

        await pi._pi_tick()

        new_value = pi._ff_heat_buckets[bucket_key]
        # With alpha=0.05, bucket should shift by ~5% of the delta.
        # Old alpha (0.2) would shift ~20%. Verify the shift is small.
        shift = abs(new_value - seed_value)
        max_shift_at_old_alpha = abs(seed_value) * 0.25  # 0.2 * (some large delta)
        # The shift should be much less than what old alpha would produce
        assert shift < max_shift_at_old_alpha, (
            f"Shift {shift} too large — alpha may not be 0.05"
        )
        # And the bucket should have moved (not still at seed)
        assert new_value != seed_value, "Bucket didn't change at all"


# ── Seed Protection Tests ────────────────────────────────────────────


class TestSeedProtection:
    """Verify seeds are protected until enough time has passed."""

    @pytest.mark.asyncio
    async def test_no_ema_before_threshold(self):
        """Bucket should not change before min observation hours."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]

        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = 30.0  # Large offset that would shift bucket significantly

        await pi._pi_tick()

        # Bucket should still be seed value — first observation, < 4 hours
        assert pi._ff_heat_buckets[bucket_key] == seed_value

    @pytest.mark.asyncio
    async def test_ema_after_threshold(self):
        """Bucket should update after min observation hours."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]

        # Simulate first observation 5 hours ago
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600

        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = 30.0  # Large offset

        await pi._pi_tick()

        # Bucket should have changed
        assert pi._ff_heat_buckets[bucket_key] != seed_value

    @pytest.mark.asyncio
    async def test_first_obs_time_recorded(self):
        """First observation should record the timestamp."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        assert 0 not in pi._ff_bucket_first_obs_time

        _settled_tick(entity, outdoor_temp=0.0)
        await pi._pi_tick()

        assert 0 in pi._ff_bucket_first_obs_time


# ── Asymmetric Learning Tests ────────────────────────────────────────


class TestAsymmetricLearning:
    """Verify faster learning for undershoot, slower for overshoot."""

    @pytest.mark.asyncio
    async def test_undershoot_learns_faster(self):
        """When room needed MORE offset (undershoot), learn at full alpha."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600

        # Observed offset > bucket value → undershoot in heating
        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = seed_value + 22.0 + 2.0  # observed_offset = seed + 2 (more than seed)

        await pi._pi_tick()
        undershoot_value = pi._ff_heat_buckets[bucket_key]

        # Reset
        pi._ff_heat_buckets[bucket_key] = seed_value

        # Observed offset < bucket value → overshoot in heating
        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = seed_value + 22.0 - 2.0  # observed_offset = seed - 2 (less than seed)

        await pi._pi_tick()
        overshoot_value = pi._ff_heat_buckets[bucket_key]

        # Undershoot should have shifted more than overshoot
        undershoot_shift = abs(undershoot_value - seed_value)
        overshoot_shift = abs(overshoot_value - seed_value)
        assert undershoot_shift > overshoot_shift, (
            f"Undershoot shift {undershoot_shift} should be > overshoot shift {overshoot_shift}"
        )

    @pytest.mark.asyncio
    async def test_cooling_undershoot_direction(self):
        """In cooling mode, undershoot means observed < bucket (more negative)."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi
        entity._attr_hvac_mode = HVACMode.COOL

        bucket_key = 30  # Hot outdoor temp
        seed_value = pi._ff_cool_buckets.get(bucket_key, 0.0)
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600

        # Observed offset more negative than bucket → cooling undershoot → full alpha
        _settled_tick(entity, outdoor_temp=30.0, current=24.0, desired=24.0)
        pi._hp_setpoint = 24.0 + seed_value - 2.0  # More negative offset

        await pi._pi_tick()

        # Should have shifted toward more negative
        assert pi._ff_cool_buckets[bucket_key] != seed_value


# ── Learn from Total Need Tests ──────────────────────────────────────


class TestLearnFromTotalNeed:
    """Verify learning includes integral contribution."""

    @pytest.mark.asyncio
    async def test_integral_included_in_observation(self):
        """Observed offset should include ki * integral."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600

        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = 24.0  # HP at 24
        pi._pi_integral = 10.0  # Integral at 10 → ki=0.05 → i_contribution = 0.5
        pi._desired_temp = 22.0

        await pi._pi_tick()

        # observed_offset = 24 + (0.05 * 10) - 22 = 24 + 0.5 - 22 = 2.5
        # Without integral: would be 24 - 22 = 2.0
        # Bucket should reflect the 2.5, not 2.0
        new_value = pi._ff_heat_buckets[bucket_key]
        expected_with_integral = 0.95 * seed_value + 0.05 * 2.5
        expected_without = 0.95 * seed_value + 0.05 * 2.0
        # Should be closer to the with-integral value
        assert abs(new_value - expected_with_integral) < abs(new_value - expected_without)


# ── Night-Only Learning Tests ────────────────────────────────────────


class TestNightOnlyLearning:
    """Verify learning is gated by sun.sun state."""

    @pytest.mark.asyncio
    async def test_learning_blocked_during_day(self):
        """Learning should not write buckets when sun is above horizon."""
        entity = FakeLearningEntity(_make_config(pi_ff_learn_night_only=True))
        pi = entity._pi

        # Mock sun.sun as above horizon
        sun_state = MagicMock()
        sun_state.state = "above_horizon"
        entity.hass.states.get.return_value = sun_state

        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600

        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = 30.0

        await pi._pi_tick()

        # Bucket should not change — daytime
        assert pi._ff_heat_buckets[bucket_key] == seed_value

    @pytest.mark.asyncio
    async def test_learning_allowed_at_night_after_delay(self):
        """Learning should write buckets after sunset + delay."""
        entity = FakeLearningEntity(_make_config(
            pi_ff_learn_night_only=True,
            pi_ff_learn_sunset_delay=0,  # No delay for this test
        ))
        pi = entity._pi

        # Mock sun.sun as below horizon
        sun_state = MagicMock()
        sun_state.state = "below_horizon"
        entity.hass.states.get.return_value = sun_state
        pi._sun_below_horizon_since = time.monotonic() - 7200  # 2 hours ago

        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600

        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = 30.0

        await pi._pi_tick()

        # Bucket should change — nighttime, past delay
        assert pi._ff_heat_buckets[bucket_key] != seed_value

    @pytest.mark.asyncio
    async def test_learning_blocked_during_sunset_delay(self):
        """Learning should not write during sunset delay period."""
        entity = FakeLearningEntity(_make_config(
            pi_ff_learn_night_only=True,
            pi_ff_learn_sunset_delay=90,  # 90 minutes
        ))
        pi = entity._pi

        sun_state = MagicMock()
        sun_state.state = "below_horizon"
        entity.hass.states.get.return_value = sun_state
        pi._sun_below_horizon_since = time.monotonic() - 1800  # Only 30 min ago

        bucket_key = 0
        seed_value = pi._ff_heat_buckets[bucket_key]
        pi._ff_bucket_first_obs_time[bucket_key] = time.monotonic() - 5 * 3600

        _settled_tick(entity, outdoor_temp=0.0)
        pi._hp_setpoint = 30.0

        await pi._pi_tick()

        assert pi._ff_heat_buckets[bucket_key] == seed_value

    @pytest.mark.asyncio
    async def test_learning_fallback_no_sun_entity(self):
        """Without sun.sun, learning should always be allowed."""
        entity = FakeLearningEntity(_make_config(pi_ff_learn_night_only=True))
        pi = entity._pi

        # Mock no sun entity
        entity.hass.states.get.return_value = None

        assert pi._is_learning_time_allowed() is True


# ── Anticipated Change FF Tests ──────────────────────────────────────


class TestAnticipatedChange:
    """Verify anticipated change feedforward."""

    @pytest.mark.asyncio
    async def test_anticipated_change_adds_offset(self):
        """Anticipated outdoor drop should add positive offset in heating."""
        entity = FakeLearningEntity(_make_config(
            pi_ff_anticipated_change_entity="sensor.forecast_delta",
            pi_ff_anticipated_change_gain=0.5,
        ))
        pi = entity._pi

        # Outdoor temp dropping 2°C → gain 0.5 → +1°C offset
        pi._anticipated_change = -2.0

        _settled_tick(entity, outdoor_temp=5.0, current=20.0, desired=22.0)

        await pi._pi_tick()

        assert pi._ff_anticipated_offset == pytest.approx(-2.0 * 0.5)

    @pytest.mark.asyncio
    async def test_anticipated_change_zero_when_disabled(self):
        """No anticipated change entity → offset should be 0."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        _settled_tick(entity, outdoor_temp=5.0, current=20.0, desired=22.0)

        await pi._pi_tick()

        assert pi._ff_anticipated_offset == 0.0

    @pytest.mark.asyncio
    async def test_anticipated_change_listener(self):
        """State change on anticipated entity should update value."""
        entity = FakeLearningEntity(_make_config(
            pi_ff_anticipated_change_entity="sensor.forecast_delta",
        ))
        pi = entity._pi

        # Simulate state change event
        event = MagicMock()
        new_state = MagicMock()
        new_state.state = "-1.5"
        event.data = {"new_state": new_state}

        pi._async_anticipated_change_changed(event)

        assert pi._anticipated_change == -1.5

    @pytest.mark.asyncio
    async def test_anticipated_change_unavailable(self):
        """Unavailable state should set anticipated change to 0."""
        entity = FakeLearningEntity(_make_config(
            pi_ff_anticipated_change_entity="sensor.forecast_delta",
        ))
        pi = entity._pi
        pi._anticipated_change = -2.0  # Had a value

        event = MagicMock()
        new_state = MagicMock()
        new_state.state = STATE_UNAVAILABLE
        event.data = {"new_state": new_state}

        pi._async_anticipated_change_changed(event)

        assert pi._anticipated_change == 0.0


# ── Integral Convergence Tests ───────────────────────────────────────


class TestIntegralConvergence:
    """Verify integral convergence tracking."""

    @pytest.mark.asyncio
    async def test_convergence_tracks_abs_integral(self):
        """Integral convergence should be EMA of abs(integral)."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        pi._integral_convergence = 0.0
        pi._pi_integral = 10.0

        _settled_tick(entity, outdoor_temp=5.0, current=20.0, desired=22.0)
        await pi._pi_tick()

        # After one tick with integral ~10: convergence = 0.99*0 + 0.01*|integral|
        assert pi._integral_convergence > 0

    @pytest.mark.asyncio
    async def test_convergence_decays_with_low_integral(self):
        """Convergence should decay when integral is consistently low."""
        entity = FakeLearningEntity(_make_config())
        pi = entity._pi

        pi._integral_convergence = 20.0  # Was high
        pi._pi_integral = 0.0

        _settled_tick(entity, outdoor_temp=5.0, current=22.0, desired=22.0)
        await pi._pi_tick()

        # Should decay toward 0
        assert pi._integral_convergence < 20.0


# ── Sun State Change Tests ───────────────────────────────────────────


class TestSunStateChange:
    """Verify sun state tracking for night-only learning."""

    def test_sun_below_horizon_sets_timestamp(self):
        """Sun going below horizon should record timestamp."""
        entity = FakeLearningEntity(_make_config(pi_ff_learn_night_only=True))
        pi = entity._pi

        event = MagicMock()
        new_state = MagicMock()
        new_state.state = "below_horizon"
        event.data = {"new_state": new_state}

        pi._async_sun_state_changed(event)

        assert pi._sun_below_horizon_since > 0

    def test_sun_above_horizon_clears_timestamp(self):
        """Sun going above horizon should clear timestamp."""
        entity = FakeLearningEntity(_make_config(pi_ff_learn_night_only=True))
        pi = entity._pi
        pi._sun_below_horizon_since = time.monotonic()

        event = MagicMock()
        new_state = MagicMock()
        new_state.state = "above_horizon"
        event.data = {"new_state": new_state}

        pi._async_sun_state_changed(event)

        assert pi._sun_below_horizon_since == 0.0


# ── ExtraStoredData Tests ────────────────────────────────────────────


class TestExtraStoredDataLearning:
    """Verify ExtraStoredData includes new fields."""

    def test_observation_counts_persisted(self):
        """Observation counts should survive serialization."""
        data = PIExtraStoredData(
            ff_heat_buckets={0: 1.0},
            ff_cool_buckets={},
            pi_integral=0.0,
            desired_temp=22.0,
            hp_setpoint=22.0,
            ff_bucket_observation_counts={0: 5, 3: 10},
            integral_convergence=3.5,
        )
        serialized = data.as_dict()
        restored = PIExtraStoredData.from_dict(serialized)

        assert restored is not None
        assert restored.ff_bucket_observation_counts == {0: 5, 3: 10}
        assert restored.integral_convergence == 3.5

    def test_missing_new_fields_default(self):
        """Old ExtraStoredData without new fields should still load."""
        old_data = {
            "ff_heat_buckets": {"0": 1.0},
            "ff_cool_buckets": {},
            "pi_integral": 0.0,
            "desired_temp": 22.0,
            "hp_setpoint": 22.0,
            # No ff_bucket_observation_counts or integral_convergence
        }
        restored = PIExtraStoredData.from_dict(old_data)

        assert restored is not None
        assert restored.ff_bucket_observation_counts == {}
        assert restored.integral_convergence == 0.0
