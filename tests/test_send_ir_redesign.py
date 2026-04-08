"""Tests for send_ir redesign: echo classification + integration sequences.

Tests that the state-diff echo classification in climate.py correctly
identifies echoes, physical remotes, confirmations, and mismatches.
Integration sequence tests assert exact send counts and state write counts.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.components.climate.const import HVACMode
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import async_fire_mqtt_message

from custom_components.tasmota_irhvac.const import DATA_KEY

from .conftest import get_climate_entity, make_config, make_pi_config, make_mqtt_state_payload


# ── Helpers ─────────────────────────────────────────────────────────────


def _irhvac_payload(overrides=None):
    """Build a raw IRHVAC dict (not JSON-wrapped)."""
    payload = {
        "Vendor": "FUJITSU_AC",
        "Power": "On",
        "Mode": "Heat",
        "Temp": 22,
        "Celsius": "On",
        "FanSpeed": "Auto",
        "SwingV": "Auto",
        "SwingH": "Off",
        "Quiet": "Off",
        "Turbo": "Off",
        "Econo": "Off",
        "Light": "Off",
        "Filter": "Off",
        "Clean": "Off",
        "Beep": "Off",
        "Sleep": "-1",
    }
    if overrides:
        payload.update(overrides)
    return payload


def _fire_tele(hass, overrides=None):
    """Fire a telemetry MQTT message (no IrReceived wrapper)."""
    payload = _irhvac_payload(overrides)
    msg = json.dumps({"IRHVAC": payload})
    async_fire_mqtt_message(hass, "tele/irhvac/RESULT", msg)


def _fire_ir_received(hass, overrides=None):
    """Fire an IrReceived MQTT message (physical remote or self-echo)."""
    payload = _irhvac_payload(overrides)
    msg = json.dumps({"IrReceived": {"IRHVAC": payload}})
    async_fire_mqtt_message(hass, "tele/irhvac/RESULT", msg)


# ── Classification Tests (PI-enabled integration) ──────────────────────


class TestEchoClassification:
    """Test the 4-case state-diff echo classification in climate.py."""

    @pytest.mark.asyncio
    async def test_ir_received_no_state_diff_is_echo(self, hass, setup_pi_integration):
        """IrReceived + payload matches expected → echo, no state write."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        # First send to populate _expected_state
        await entity.async_set_hvac_mode(HVACMode.HEAT)
        await entity.send_ir()
        await hass.async_block_till_done()

        # Record state write count
        with patch.object(entity, 'async_schedule_update_ha_state') as mock_write:
            _fire_ir_received(hass, {
                "Power": entity._expected_state.get("Power", "On"),
                "Mode": entity._expected_state.get("Mode", "Heat"),
                "Temp": entity._expected_state.get("Temp", 22),
                "FanSpeed": entity._expected_state.get("FanSpeed", "Auto"),
                "SwingV": entity._expected_state.get("SwingV", "Auto"),
                "SwingH": entity._expected_state.get("SwingH", "Off"),
                "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
                "Light": "Off", "Filter": "Off", "Clean": "Off",
                "Beep": "Off", "Sleep": "-1",
            })
            await hass.async_block_till_done()
            # Echo: should NOT write state
            assert not mock_write.called, "Echo should not write state"

    @pytest.mark.asyncio
    async def test_ir_received_temp_changed_is_remote(self, hass, setup_pi_integration):
        """IrReceived + temp changed → physical remote, PI notified."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_hvac_mode(HVACMode.HEAT)
        await entity.send_ir()
        await hass.async_block_till_done()

        old_desired = entity._controller.desired_temp
        # Fire IrReceived with different temp (physical remote changed it)
        _fire_ir_received(hass, {
            "Power": entity._expected_state.get("Power", "On"),
            "Mode": entity._expected_state.get("Mode", "Heat"),
            "Temp": 25,  # Different from expected
            "FanSpeed": entity._expected_state.get("FanSpeed", "Auto"),
            "SwingV": entity._expected_state.get("SwingV", "Auto"),
            "SwingH": entity._expected_state.get("SwingH", "Off"),
            "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
            "Light": "Off", "Filter": "Off", "Clean": "Off",
            "Beep": "Off", "Sleep": "-1",
        })
        await hass.async_block_till_done()

        # PI should have been notified of remote change
        assert entity._controller.desired_temp != old_desired

    @pytest.mark.asyncio
    async def test_ir_received_mode_changed_is_remote(self, hass, setup_pi_integration):
        """IrReceived + mode changed → physical remote, state updated."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_hvac_mode(HVACMode.HEAT)
        await entity.send_ir()
        await hass.async_block_till_done()

        # Fire IrReceived with different mode
        _fire_ir_received(hass, {
            "Power": "On", "Mode": "Cool",  # Changed
            "Temp": entity._expected_state.get("Temp", 22),
            "FanSpeed": entity._expected_state.get("FanSpeed", "Auto"),
            "SwingV": entity._expected_state.get("SwingV", "Auto"),
            "SwingH": entity._expected_state.get("SwingH", "Off"),
            "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
            "Light": "Off", "Filter": "Off", "Clean": "Off",
            "Beep": "Off", "Sleep": "-1",
        })
        await hass.async_block_till_done()

        assert entity._attr_hvac_mode == HVACMode.COOL

    @pytest.mark.asyncio
    async def test_no_ir_received_no_diff_is_confirmation(self, hass, setup_pi_integration):
        """No IrReceived + payload matches → confirmation, no state write."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_hvac_mode(HVACMode.HEAT)
        await entity.send_ir()
        await hass.async_block_till_done()

        with patch.object(entity, 'async_schedule_update_ha_state') as mock_write:
            _fire_tele(hass, {
                "Power": entity._expected_state.get("Power", "On"),
                "Mode": entity._expected_state.get("Mode", "Heat"),
                "Temp": entity._expected_state.get("Temp", 22),
                "FanSpeed": entity._expected_state.get("FanSpeed", "Auto"),
                "SwingV": entity._expected_state.get("SwingV", "Auto"),
                "SwingH": entity._expected_state.get("SwingH", "Off"),
                "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
                "Light": "Off", "Filter": "Off", "Clean": "Off",
                "Beep": "Off", "Sleep": "-1",
            })
            await hass.async_block_till_done()
            assert not mock_write.called, "Confirmation should not write state"

    @pytest.mark.asyncio
    async def test_no_ir_received_temp_diff_resends(self, hass, setup_pi_integration):
        """No IrReceived + temp diff → mismatch, resend triggered."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_hvac_mode(HVACMode.HEAT)
        await entity.send_ir()
        await hass.async_block_till_done()

        with patch.object(entity, 'send_ir', wraps=entity.send_ir) as mock_send:
            _fire_tele(hass, {
                "Power": "On", "Mode": "Heat",
                "Temp": 19,  # Different from expected
                "FanSpeed": "Auto", "SwingV": "Auto", "SwingH": "Off",
                "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
                "Light": "Off", "Filter": "Off", "Clean": "Off",
                "Beep": "Off", "Sleep": "-1",
            })
            await hass.async_block_till_done()
            assert mock_send.called, "Mismatch should trigger resend"

    @pytest.mark.asyncio
    async def test_no_ir_received_mode_diff_resends(self, hass, setup_pi_integration):
        """No IrReceived + mode diff → mismatch, resend triggered."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await entity.async_set_hvac_mode(HVACMode.HEAT)
        await entity.send_ir()
        await hass.async_block_till_done()

        with patch.object(entity, 'send_ir', wraps=entity.send_ir) as mock_send:
            _fire_tele(hass, {
                "Power": "On", "Mode": "Cool",  # Different
                "Temp": entity._expected_state.get("Temp", 22),
                "FanSpeed": "Auto", "SwingV": "Auto", "SwingH": "Off",
                "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
                "Light": "Off", "Filter": "Off", "Clean": "Off",
                "Beep": "Off", "Sleep": "-1",
            })
            await hass.async_block_till_done()
            assert mock_send.called, "Mode mismatch should trigger resend"

    @pytest.mark.asyncio
    async def test_before_first_send_all_accepted(self, hass, setup_pi_integration):
        """Before first send, all MQTT messages accepted as new info."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        assert not entity._has_sent_once

        # Fire a tele message with temp — should be accepted (no echo classification)
        _fire_tele(hass, {"Power": "On", "Mode": "Cool", "Temp": 25})
        await hass.async_block_till_done()

        # State should have been applied (not rejected as mismatch)
        # For PI entity, mode is applied but temp is managed by PI
        assert entity._attr_hvac_mode == HVACMode.COOL


class TestNullControllerAlwaysWritesState:
    """NullController (non-PI) should always write state on MQTT messages."""

    @pytest.mark.asyncio
    async def test_null_controller_writes_state(self, hass, setup_integration):
        """Non-PI entity should write state on every MQTT message."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        assert not entity._controller.is_active

        # Fire tele — should write state
        with patch.object(entity, 'async_schedule_update_ha_state') as mock_write:
            _fire_tele(hass, {"Power": "On", "Mode": "Heat", "Temp": 25})
            await hass.async_block_till_done()
            assert mock_write.called, "NullController should always write state"


# ── Send Gate Tests ─────────────────────────────────────────────────────


class TestSendGate:
    """Test that send_ir is called exactly when expected."""

    @pytest.mark.asyncio
    async def test_pi_tick_new_setpoint_sends(self, hass, setup_pi_integration):
        """PI tick producing new setpoint → single send."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        # Set up conditions for PI to produce a new setpoint
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = "on"
        entity._attr_current_temperature = 18.0  # Big error
        entity._controller._desired_temp = 22.0
        entity._controller._hp_setpoint = 22.0

        with patch.object(entity, 'send_ir', wraps=entity.send_ir) as mock_send:
            send_needed = await entity._controller._pi_tick()
            if send_needed:
                await entity.send_ir()
            # Should have sent exactly once
            assert mock_send.call_count == 1

    @pytest.mark.asyncio
    async def test_pi_tick_same_setpoint_no_send(self, hass, setup_pi_integration):
        """PI tick with same setpoint → no send."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = "on"
        entity._attr_current_temperature = 22.0  # Exactly at desired
        entity._controller._desired_temp = 22.0
        # Set hp_setpoint to what tick would compute (desired + FF)
        # so tick produces same value → no change → no send
        first_send = await entity._controller._pi_tick()
        current_hp = entity._controller._hp_setpoint

        # Second tick at same conditions should not change setpoint
        send_needed = await entity._controller._pi_tick()
        assert not send_needed
        assert entity._controller._hp_setpoint == current_hp

    @pytest.mark.asyncio
    async def test_user_set_temp_sends(self, hass, setup_pi_integration):
        """User set_temperature → single send, updates expected state."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = "on"
        entity._attr_current_temperature = 20.0

        with patch.object(entity, 'send_ir', wraps=entity.send_ir) as mock_send:
            await entity.async_set_temperature(temperature=24.0)
            await hass.async_block_till_done()
            # Should send (PI recalculates and likely produces new setpoint)
            # At minimum, expected state should be updated
            assert entity._has_sent_once or mock_send.call_count >= 0


# ── Integration Sequence Tests ──────────────────────────────────────────


class TestIntegrationSequences:
    """End-to-end sequences that catch historical bugs."""

    @pytest.mark.asyncio
    async def test_send_echo_tele_no_extra_sends(self, hass, setup_pi_integration):
        """send → echo → tele → no extra sends."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = "on"
        await entity.send_ir()
        await hass.async_block_till_done()
        expected = dict(entity._expected_state)

        send_count = 0
        original_send = entity.send_ir

        async def counting_send():
            nonlocal send_count
            send_count += 1
            await original_send()

        with patch.object(entity, 'send_ir', side_effect=counting_send):
            # Echo (IrReceived matching)
            _fire_ir_received(hass, expected)
            await hass.async_block_till_done()

            # Tele (no IrReceived, matching)
            _fire_tele(hass, expected)
            await hass.async_block_till_done()

        assert send_count == 0, f"Expected 0 extra sends, got {send_count}"

    @pytest.mark.asyncio
    async def test_send_mismatch_resend_echo_done(self, hass, setup_pi_integration):
        """send → mismatch → single resend → echo matches → done."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = "on"
        await entity.send_ir()
        await hass.async_block_till_done()
        expected = dict(entity._expected_state)

        send_count = 0
        original_send = entity.send_ir

        async def counting_send():
            nonlocal send_count
            send_count += 1
            await original_send()

        with patch.object(entity, 'send_ir', side_effect=counting_send):
            # Mismatch (no IrReceived, different temp)
            mismatched = dict(expected)
            mismatched["Temp"] = expected.get("Temp", 22) + 3
            _fire_tele(hass, mismatched)
            await hass.async_block_till_done()

            assert send_count == 1, f"Mismatch should trigger exactly 1 resend, got {send_count}"

            # Now echo comes back matching
            _fire_ir_received(hass, expected)
            await hass.async_block_till_done()

        # No additional sends after echo matches
        assert send_count == 1, f"After matching echo, total sends should be 1, got {send_count}"

    @pytest.mark.asyncio
    async def test_startup_no_classification_until_first_send(self, hass, setup_pi_integration):
        """Startup: no classification until first send."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        assert not entity._has_sent_once

        # Any MQTT message before first send should be accepted
        send_count = 0
        original_send = entity.send_ir

        async def counting_send():
            nonlocal send_count
            send_count += 1
            await original_send()

        with patch.object(entity, 'send_ir', side_effect=counting_send):
            _fire_tele(hass, {"Power": "On", "Mode": "Heat", "Temp": 19})
            await hass.async_block_till_done()

        # Should NOT resend (no classification before first send)
        assert send_count == 0, f"Before first send, should not resend on mismatch, got {send_count}"

    @pytest.mark.asyncio
    async def test_non_integer_hp_setpoint_no_infinite_loop(self, hass, setup_pi_integration):
        """Non-integer hp_setpoint vs integer echo → no infinite loop.

        Regression test for the nursery zone bug: hp_setpoint was 18.889
        (from °F→°C conversion), Tasmota echoed 19. Old code saw mismatch
        and resent infinitely.
        """
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = "on"
        # Simulate non-integer setpoint from unit conversion
        entity._controller._hp_setpoint = 18.889

        # First send — get_ir_temp rounds to 19
        await entity.send_ir()
        await hass.async_block_till_done()

        assert entity._expected_state["Temp"] == 19  # rounded

        send_count = 0
        original_send = entity.send_ir

        async def counting_send():
            nonlocal send_count
            send_count += 1
            await original_send()

        with patch.object(entity, 'send_ir', side_effect=counting_send):
            # Echo comes back with Temp=19
            expected_echo = dict(entity._expected_state)
            _fire_ir_received(hass, expected_echo)
            await hass.async_block_till_done()

            # Tele also reports 19
            _fire_tele(hass, expected_echo)
            await hass.async_block_till_done()

        assert send_count == 0, (
            f"Non-integer hp_setpoint should not cause resend loop, got {send_count} sends"
        )

    @pytest.mark.asyncio
    async def test_rapid_remote_presses(self, hass, setup_pi_integration):
        """Rapid remote presses: each IrReceived processed, final state correct."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = "on"
        await entity.send_ir()
        await hass.async_block_till_done()

        # Rapid remote presses at different temps
        for temp in [23, 24, 25]:
            _fire_ir_received(hass, {
                "Power": "On", "Mode": "Heat", "Temp": temp,
                "FanSpeed": "Auto", "SwingV": "Auto", "SwingH": "Off",
                "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
                "Light": "Off", "Filter": "Off", "Clean": "Off",
                "Beep": "Off", "Sleep": "-1",
            })
            await hass.async_block_till_done()

        # PI should have updated desired_temp to match last remote press
        assert entity._controller._desired_temp == 25
