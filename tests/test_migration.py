"""Tests for config entry migration."""

import pytest

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.__init__ import async_migrate_entry

from .conftest import make_config


class TestConfigMigration:
    """Tests for async_migrate_entry."""

    @pytest.mark.asyncio
    async def test_migrate_v1_1_to_v1_3(self, hass):
        """Migration from v1.1 should add bias_entity, setpoint_weight, then disturbance_inputs."""
        config = make_config()
        # Remove keys added in v1.2 and v1.3
        config.pop("pi_disturbance_inputs", None)
        config.pop("pi_setpoint_weight", None)
        # Add old v1.1 keys
        config["pi_ff_suppress_learning_entity"] = "input_boolean.suppress"
        config["pi_ff_bias_entity"] = ""

        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=1,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == 8

        # v1.2→v1.3 migrated suppress/bias to disturbance_inputs
        # v1.3→v1.4 migrated disturbance_inputs to model_inputs
        model_inputs = entry.data.get("pi_model_inputs", [])
        assert len(model_inputs) == 1
        assert model_inputs[0]["entity_id"] == "input_boolean.suppress"
        # Old keys should be removed
        assert "pi_ff_suppress_learning_entity" not in entry.data
        assert "pi_ff_bias_entity" not in entry.data
        assert "pi_disturbance_inputs" not in entry.data

    @pytest.mark.asyncio
    async def test_migrate_v1_2_to_v1_3(self, hass):
        """Migration from v1.2 should convert suppress/bias entities to disturbance inputs."""
        config = make_config()
        config.pop("pi_disturbance_inputs", None)
        config["pi_ff_suppress_learning_entity"] = "input_boolean.suppress"
        config["pi_ff_bias_entity"] = "input_number.bias"
        config["pi_setpoint_weight"] = 0.5

        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=2,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == 8

        # v1.2→v1.3→v1.4: disturbance_inputs migrated to model_inputs
        model_inputs = entry.data.get("pi_model_inputs", [])
        assert len(model_inputs) == 2
        assert model_inputs[0]["entity_id"] == "input_boolean.suppress"
        assert model_inputs[1]["entity_id"] == "input_number.bias"

    @pytest.mark.asyncio
    async def test_migrate_v1_3_to_v1_4(self, hass):
        """Migration from v1.3 should convert disturbance_inputs to model_inputs."""
        config = make_config()
        config["pi_disturbance_inputs"] = [{
            "name": "Stove",
            "entity_id": "input_boolean.stove",
            "suppress_learning": True,
            "default_bias": -3.0,
            "gain": 1.0,
        }]
        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=3,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == 8
        model_inputs = entry.data.get("pi_model_inputs", [])
        assert len(model_inputs) == 1
        assert model_inputs[0]["entity_id"] == "input_boolean.stove"
        assert model_inputs[0]["seed_heat"] == -3.0
        assert "pi_disturbance_inputs" not in entry.data

    @pytest.mark.asyncio
    async def test_migrate_v1_4_noop(self, hass):
        """Migration of v1.4 entry should be a no-op."""
        config = make_config()
        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=4,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True
        assert entry.minor_version == 8

    @pytest.mark.asyncio
    async def test_migrate_empty_entities(self, hass):
        """Empty old entity strings should produce empty disturbance inputs."""
        config = make_config()
        config.pop("pi_disturbance_inputs", None)
        config["pi_ff_suppress_learning_entity"] = ""
        config["pi_ff_bias_entity"] = ""

        entry = MockConfigEntry(
            domain=DOMAIN,
            data=config,
            title="Test",
            version=1,
            minor_version=2,
        )
        entry.add_to_hass(hass)

        result = await async_migrate_entry(hass, entry)
        assert result is True

        inputs = entry.data.get("pi_disturbance_inputs", [])
        assert len(inputs) == 0
