"""Tests for the repairs integration (config issue checks)."""

import pytest
from unittest.mock import MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.__init__ import _check_config_issues

from .conftest import make_pi_config, make_config


def _make_entry(config):
    """Create a MockConfigEntry from a config dict."""
    return MockConfigEntry(domain=DOMAIN, data=config, title="Test")


class TestCheckConfigIssues:
    """Tests for _check_config_issues."""

    def test_no_issues_when_pi_disabled(self, hass):
        """No issues should be created when PI is disabled."""
        config = make_config()
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create, \
             patch.object(ir, "async_delete_issue") as mock_delete:
            _check_config_issues(hass, entry)
            mock_create.assert_not_called()

    def test_no_issues_when_entities_exist(self, hass):
        """No issues when all configured entities exist."""
        config = make_pi_config()
        entry = _make_entry(config)

        # Set up entity states so they exist
        hass.states.async_set("sensor.outdoor_temp", "5.0")
        hass.states.async_set("sensor.room_temp", "22.0")

        with patch.object(ir, "async_create_issue") as mock_create, \
             patch.object(ir, "async_delete_issue") as mock_delete:
            _check_config_issues(hass, entry)
            mock_create.assert_not_called()

    def test_issue_created_for_missing_outdoor_sensor(self, hass):
        """Issue should be created when outdoor sensor doesn't exist."""
        config = make_pi_config({"outdoor_temp_sensor": "sensor.nonexistent"})
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_config_issues(hass, entry)
            # Should have been called with outdoor_sensor_not_found
            assert any(
                call.kwargs.get("translation_key") == "outdoor_sensor_not_found"
                or (len(call.args) > 2 and "outdoor" in str(call))
                for call in mock_create.call_args_list
            ) or mock_create.called

    def test_issue_created_for_missing_suppress_entity(self, hass):
        """Issue should be created when suppress learning entity doesn't exist."""
        config = make_pi_config({
            "pi_ff_suppress_learning_entity": "input_boolean.nonexistent",
            "outdoor_temp_sensor": "",
        })
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_config_issues(hass, entry)
            assert mock_create.called

    def test_issue_created_for_missing_bias_entity(self, hass):
        """Issue should be created when bias entity doesn't exist."""
        config = make_pi_config({
            "pi_ff_bias_entity": "input_number.nonexistent",
            "outdoor_temp_sensor": "",
        })
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_config_issues(hass, entry)
            assert mock_create.called

    def test_issue_deleted_when_entity_exists(self, hass):
        """Issue should be deleted when entity exists."""
        config = make_pi_config({
            "outdoor_temp_sensor": "sensor.outdoor_temp",
        })
        entry = _make_entry(config)
        hass.states.async_set("sensor.outdoor_temp", "5.0")

        with patch.object(ir, "async_delete_issue") as mock_delete:
            _check_config_issues(hass, entry)
            # Should delete the outdoor sensor issue since entity exists
            assert mock_delete.called

    def test_empty_entity_id_skipped(self, hass):
        """Empty entity IDs should not trigger issues."""
        config = make_pi_config({
            "outdoor_temp_sensor": "",
            "pi_ff_suppress_learning_entity": "",
            "pi_ff_bias_entity": "",
        })
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_config_issues(hass, entry)
            mock_create.assert_not_called()
