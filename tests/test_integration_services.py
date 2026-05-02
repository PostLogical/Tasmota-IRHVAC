"""Tier 5: Integration tests for service calls."""

import pytest

from homeassistant.components.climate.const import HVACMode
from homeassistant.core import HomeAssistant

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.pi.batch_learning import Observation

from .conftest import get_climate_entity


def _dummy_obs() -> Observation:
    """Real Observation for tests that just need a non-empty buffer.

    State-write paths exercised during service calls now read buffer
    contents (via the typed snapshot rebuild), so dict placeholders
    break iteration. Use this helper to seed buffers safely.
    """
    return Observation(
        timestamp=0.0, wall_time=0.0, hp_setpoint=22.0,
        current_c=21.0, desired_c=22.0, outdoor_temp_c=5.0,
        room_rate=0.0, raw_readings={}, clamped=False,
    )


class TestPIServices:
    """Tests for PI-related service calls."""

    @pytest.mark.asyncio
    async def test_suppress_ff_learning(self, hass, setup_pi_integration):
        """suppress_ff_learning service should set manual suppress flag."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await hass.services.async_call(
            DOMAIN, "suppress_ff_learning",
            {"entity_id": entity.entity_id, "reason": "testing"},
            blocking=True,
        )

        assert entity._pi._manual_ff_suppress is True
        assert entity._pi._manual_ff_suppress_reason == "testing"

    @pytest.mark.asyncio
    async def test_resume_ff_learning(self, hass, setup_pi_integration):
        """resume_ff_learning service should clear manual suppress flag."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        # First suppress
        entity._pi._manual_ff_suppress = True
        entity._pi._manual_ff_suppress_reason = "testing"

        await hass.services.async_call(
            DOMAIN, "resume_ff_learning",
            {"entity_id": entity.entity_id},
            blocking=True,
        )

        assert entity._pi._manual_ff_suppress is False

    @pytest.mark.asyncio
    async def test_reset_ff_seeds(self, hass, setup_pi_integration):
        """reset_ff_seeds service should reset RLS models and zero integral."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        # Modify state
        entity._pi._pi_integral = 25.0
        entity._pi._rls_heat.beta[0] = 99.0
        entity._pi._rls_heat.observation_count = 100

        await hass.services.async_call(
            DOMAIN, "reset_ff_seeds",
            {"entity_id": entity.entity_id},
            blocking=True,
        )

        assert entity._pi._pi_integral == 0.0
        assert entity._pi._rls_heat.beta[0] == 0.0  # Reset to seed
        assert entity._pi._rls_heat.observation_count == 0


class TestLearningReset:
    """Tests for the unified learning_reset service."""

    @pytest.mark.asyncio
    async def test_reset_seeds_only(self, hass, setup_pi_integration):
        """Seeds target resets RLS beta/P/frozen but leaves integral untouched."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi
        n = pi._rls_heat.n

        pi._pi_integral = 25.0
        pi._rls_heat.beta[0] = 99.0
        pi._rls_heat.observation_count = 100
        # Corrupt P off-diagonal
        pi._rls_heat.P[1] = 5.0

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["seeds"]},
            blocking=True,
        )

        assert pi._rls_heat.beta[0] == 0.0
        assert pi._rls_heat.observation_count == 0
        assert pi._pi_integral == 25.0  # Untouched
        # P should be reset to diagonal
        from custom_components.tasmota_irhvac.pi.pi_controller import DEFAULT_RLS_P_INIT
        assert pi._rls_heat.P[0] == DEFAULT_RLS_P_INIT  # diagonal
        assert pi._rls_heat.P[1] == 0.0  # off-diagonal cleared
        # Model input features should be re-frozen (indices 2+)
        for i in range(2, n):
            assert pi._rls_heat.frozen[i] is True

    @pytest.mark.asyncio
    async def test_reset_integral_only(self, hass, setup_pi_integration):
        """Integral target zeros integral but leaves RLS untouched."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._pi_integral = 25.0
        pi._rls_heat.beta[0] = 99.0

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["integral"]},
            blocking=True,
        )

        assert pi._pi_integral == 0.0
        assert pi._rls_heat.beta[0] == 99.0  # Untouched

    @pytest.mark.asyncio
    async def test_reset_seeds_and_integral(self, hass, setup_pi_integration):
        """Multiple targets reset together."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._pi_integral = 25.0
        pi._rls_heat.beta[0] = 99.0

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["seeds", "integral"]},
            blocking=True,
        )

        assert pi._pi_integral == 0.0
        assert pi._rls_heat.beta[0] == 0.0

    @pytest.mark.asyncio
    async def test_reset_buffers_without_greybox(self, hass, setup_pi_integration):
        """Buffers target clears observation buffers and batch state but not greybox."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Seed buffer and batch state
        pi._observation_buffer_heat._buffer.append(_dummy_obs())
        pi._greybox_buffer._buffer.append(_dummy_obs())
        pi._last_batch_result = {"rms": 0.5}
        pi._drift_correction_signs = [[1, -1]]
        pi._tuning_alert_counters = {"test": 1}

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["buffers"]},
            blocking=True,
        )

        assert len(pi._observation_buffer_heat._buffer) == 0
        assert pi._last_batch_result is None
        assert pi._drift_correction_signs == []
        assert pi._tuning_alert_counters == {}
        # Greybox untouched
        assert len(pi._greybox_buffer._buffer) == 1

    @pytest.mark.asyncio
    async def test_reset_greybox_without_buffers(self, hass, setup_pi_integration):
        """Greybox target clears greybox buffer and results but not observation buffers."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._observation_buffer_heat._buffer.append(_dummy_obs())
        pi._greybox_buffer._buffer.append(_dummy_obs())
        pi._last_greybox_result = {"tau_eff": 60.0}
        pi._last_greybox_bridge = {"outdoor_delta": -0.3}
        pi._greybox_has_been_good = True

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["greybox"]},
            blocking=True,
        )

        assert len(pi._observation_buffer_heat._buffer) == 1  # Untouched
        assert len(pi._greybox_buffer._buffer) == 0
        assert pi._last_greybox_result is None
        assert pi._last_greybox_bridge is None
        assert pi._greybox_has_been_good is False

    @pytest.mark.asyncio
    async def test_reset_plant_id(self, hass, setup_pi_integration):
        """Plant_id target resets plant estimate to seeds."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Modify plant estimate
        original_tau = pi._plant_id.plant.tau_fast.value
        from custom_components.tasmota_irhvac.pi.plant_model import (
            ParameterEstimate,
            PlantEstimate,
        )
        pi._plant_id._plant = PlantEstimate(
            k=pi._plant_id.plant.k,
            theta=pi._plant_id.plant.theta,
            tau_fast=ParameterEstimate(value=999.0, source="test"),
            tau_slow=pi._plant_id.plant.tau_slow,
        )

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["plant_id"]},
            blocking=True,
        )

        assert pi._plant_id.plant.tau_fast.value == original_tau
        assert pi._plant_id.plant.tau_fast.source == "seed"

    @pytest.mark.asyncio
    async def test_reset_all_targets(self, hass, setup_pi_integration):
        """All targets at once."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._pi_integral = 25.0
        pi._rls_heat.beta[0] = 99.0
        pi._observation_buffer_heat._buffer.append(_dummy_obs())
        pi._greybox_buffer._buffer.append(_dummy_obs())

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {
                "entity_id": entity.entity_id,
                "targets": ["seeds", "buffers", "integral", "plant_id", "greybox"],
            },
            blocking=True,
        )

        assert pi._pi_integral == 0.0
        assert pi._rls_heat.beta[0] == 0.0
        assert len(pi._observation_buffer_heat._buffer) == 0
        assert len(pi._greybox_buffer._buffer) == 0

    @pytest.mark.asyncio
    async def test_reset_with_mode_heat(self, hass, setup_pi_integration):
        """Mode parameter restricts seeds reset to heat only."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._rls_heat.beta[0] = 99.0
        pi._rls_cool.beta[0] = 88.0

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["seeds"], "mode": "heat"},
            blocking=True,
        )

        assert pi._rls_heat.beta[0] == 0.0  # Reset
        assert pi._rls_cool.beta[0] == 88.0  # Untouched


    @pytest.mark.asyncio
    async def test_reset_buffers_mode_cool(self, hass, setup_pi_integration):
        """Mode=cool on buffers only clears cool buffer, leaves heat."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._observation_buffer_heat._buffer.append(_dummy_obs())
        pi._observation_buffer_cool._buffer.append(_dummy_obs())

        await hass.services.async_call(
            DOMAIN, "learning_reset",
            {"entity_id": entity.entity_id, "targets": ["buffers"], "mode": "cool"},
            blocking=True,
        )

        assert len(pi._observation_buffer_heat._buffer) == 1  # Untouched
        assert len(pi._observation_buffer_cool._buffer) == 0


class TestLearningSnapshots:
    """Tests for learning_save and learning_restore services."""

    @pytest.mark.asyncio
    async def test_save_and_restore_round_trip(self, hass, setup_pi_integration):
        """Save and restore preserves RLS beta, P, frozen, obs count, integral."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi
        n = pi._rls_heat.n

        # Set known state across both models
        pi._rls_heat.beta[0] = 42.0
        pi._rls_heat.beta[1] = -3.5
        pi._rls_cool.beta[0] = 17.0
        pi._rls_cool.beta[1] = -2.0
        pi._rls_heat.observation_count = 55
        pi._rls_cool.observation_count = 30
        pi._rls_heat.frozen[0] = True
        pi._pi_integral = 15.0
        pi._manual_override_heat[1] = True
        # Set a non-default P diagonal value
        pi._rls_heat.P[0] = 999.0

        await hass.services.async_call(
            DOMAIN, "learning_save",
            {"entity_id": entity.entity_id, "slot": "test_slot"},
            blocking=True,
        )

        # Trash everything
        pi._rls_heat.beta[0] = 0.0
        pi._rls_heat.beta[1] = 0.0
        pi._rls_cool.beta[0] = 0.0
        pi._rls_cool.beta[1] = 0.0
        pi._rls_heat.observation_count = 0
        pi._rls_cool.observation_count = 0
        pi._rls_heat.frozen[0] = False
        pi._pi_integral = 0.0
        pi._manual_override_heat[1] = None
        pi._rls_heat.P[0] = 1.0

        await hass.services.async_call(
            DOMAIN, "learning_restore",
            {"entity_id": entity.entity_id, "slot": "test_slot"},
            blocking=True,
        )

        # Verify full round-trip
        assert pi._rls_heat.beta[0] == 42.0
        assert pi._rls_heat.beta[1] == -3.5
        assert pi._rls_cool.beta[0] == 17.0
        assert pi._rls_cool.beta[1] == -2.0
        assert pi._rls_heat.observation_count == 55
        assert pi._rls_cool.observation_count == 30
        assert pi._rls_heat.frozen[0] is True
        assert pi._pi_integral == 15.0
        assert pi._manual_override_heat[1] is True
        assert pi._rls_heat.P[0] == 999.0

    @pytest.mark.asyncio
    async def test_max_three_slots(self, hass, setup_pi_integration):
        """Fourth slot is rejected when three already exist."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        for name in ["a", "b", "c"]:
            await hass.services.async_call(
                DOMAIN, "learning_save",
                {"entity_id": entity.entity_id, "slot": name},
                blocking=True,
            )

        # Fourth slot should be rejected
        entity._pi._rls_heat.beta[0] = 77.0
        await hass.services.async_call(
            DOMAIN, "learning_save",
            {"entity_id": entity.entity_id, "slot": "d"},
            blocking=True,
        )

        # Verify slot "d" was not saved by trying to restore it
        entity._pi._rls_heat.beta[0] = 0.0
        await hass.services.async_call(
            DOMAIN, "learning_restore",
            {"entity_id": entity.entity_id, "slot": "d"},
            blocking=True,
        )
        assert entity._pi._rls_heat.beta[0] == 0.0  # Unchanged, slot doesn't exist

    @pytest.mark.asyncio
    async def test_overwrite_existing_slot(self, hass, setup_pi_integration):
        """Overwriting an existing slot works even at max capacity."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        for name in ["a", "b", "c"]:
            await hass.services.async_call(
                DOMAIN, "learning_save",
                {"entity_id": entity.entity_id, "slot": name},
                blocking=True,
            )

        # Overwrite existing slot "b"
        entity._pi._rls_heat.beta[0] = 55.0
        await hass.services.async_call(
            DOMAIN, "learning_save",
            {"entity_id": entity.entity_id, "slot": "b"},
            blocking=True,
        )

        entity._pi._rls_heat.beta[0] = 0.0
        await hass.services.async_call(
            DOMAIN, "learning_restore",
            {"entity_id": entity.entity_id, "slot": "b"},
            blocking=True,
        )
        assert entity._pi._rls_heat.beta[0] == 55.0

    @pytest.mark.asyncio
    async def test_restore_nonexistent_slot(self, hass, setup_pi_integration):
        """Restore from nonexistent slot logs warning, changes nothing."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._pi_integral = 10.0

        await hass.services.async_call(
            DOMAIN, "learning_restore",
            {"entity_id": entity.entity_id, "slot": "nonexistent"},
            blocking=True,
        )

        assert pi._pi_integral == 10.0  # Unchanged


class TestBackwardCompat:
    """Old services still work with their original behavior."""

    @pytest.mark.asyncio
    async def test_reset_ff_seeds_still_zeros_integral(self, hass, setup_pi_integration):
        """reset_ff_seeds backward compat: still zeros integral."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._pi._pi_integral = 25.0
        entity._pi._rls_heat.beta[0] = 99.0

        await hass.services.async_call(
            DOMAIN, "reset_ff_seeds",
            {"entity_id": entity.entity_id},
            blocking=True,
        )

        assert entity._pi._pi_integral == 0.0
        assert entity._pi._rls_heat.beta[0] == 0.0

    @pytest.mark.asyncio
    async def test_flush_observation_buffer_still_clears_greybox(self, hass, setup_pi_integration):
        """flush_observation_buffer backward compat: still clears greybox."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._observation_buffer_heat._buffer.append(_dummy_obs())
        pi._greybox_buffer._buffer.append(_dummy_obs())

        await hass.services.async_call(
            DOMAIN, "flush_observation_buffer",
            {"entity_id": entity.entity_id},
            blocking=True,
        )

        assert len(pi._observation_buffer_heat._buffer) == 0
        assert len(pi._greybox_buffer._buffer) == 0


class TestIRHVACServices:
    """Tests for IRHVAC toggle services."""

    @pytest.mark.asyncio
    async def test_set_econo_on(self, hass, setup_integration):
        """set_econo service should toggle econo mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await hass.services.async_call(
            DOMAIN, "set_econo",
            {"entity_id": entity.entity_id, "econo": "on"},
            blocking=True,
        )

        assert entity._econo == "on"

    @pytest.mark.asyncio
    async def test_set_turbo_on(self, hass, setup_integration):
        """set_turbo service should toggle turbo mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await hass.services.async_call(
            DOMAIN, "set_turbo",
            {"entity_id": entity.entity_id, "turbo": "on"},
            blocking=True,
        )

        assert entity._turbo == "on"

    @pytest.mark.asyncio
    async def test_set_quiet_on(self, hass, setup_integration):
        """set_quiet service should toggle quiet mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        await hass.services.async_call(
            DOMAIN, "set_quiet",
            {"entity_id": entity.entity_id, "quiet": "on"},
            blocking=True,
        )

        assert entity._quiet == "on"
