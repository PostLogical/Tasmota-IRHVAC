"""Tier 5: Integration tests for service calls."""

import pytest

from homeassistant.components.climate.const import HVACMode
from homeassistant.core import HomeAssistant

from custom_components.tasmota_irhvac.const import DOMAIN

from .conftest import get_climate_entity


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
