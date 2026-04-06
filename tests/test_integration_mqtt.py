"""Tier 2: Integration tests for MQTT state and command handling."""

import json
import pytest

from homeassistant.components.climate.const import (
    HVACMode,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import async_fire_mqtt_message

from custom_components.tasmota_irhvac.const import DATA_KEY

from .conftest import get_climate_entity, make_mqtt_state_payload


class TestMQTTStateUpdates:
    """Tests for MQTT state payload updating climate entity."""

    @pytest.mark.asyncio
    async def test_state_updates_hvac_mode(self, hass, setup_integration):
        """MQTT state payload should update HVAC mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = make_mqtt_state_payload({"Power": "On", "Mode": "Heat"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_hvac_mode == HVACMode.HEAT

    @pytest.mark.asyncio
    async def test_state_updates_temperature(self, hass, setup_integration):
        """MQTT state payload should update target temperature."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = make_mqtt_state_payload({"Power": "On", "Mode": "Heat", "Temp": 25})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_target_temperature == 25

    @pytest.mark.asyncio
    async def test_state_updates_fan_mode(self, hass, setup_integration):
        """MQTT state payload should update fan mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = make_mqtt_state_payload({"Power": "On", "Mode": "Heat", "FanSpeed": "Low"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_fan_mode == "low"

    @pytest.mark.asyncio
    async def test_state_updates_swing_both(self, hass, setup_integration):
        """SwingV=Auto + SwingH=Auto should set swing mode to both."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = make_mqtt_state_payload({
            "Power": "On", "Mode": "Heat",
            "SwingV": "Auto", "SwingH": "Auto",
        })
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_BOTH

    @pytest.mark.asyncio
    async def test_state_updates_swing_vertical(self, hass, setup_integration):
        """SwingV=Auto + SwingH=Off should set swing mode to vertical."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = make_mqtt_state_payload({
            "Power": "On", "Mode": "Heat",
            "SwingV": "Auto", "SwingH": "Off",
        })
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_VERTICAL

    @pytest.mark.asyncio
    async def test_state_updates_swing_horizontal(self, hass, setup_integration):
        """SwingV=Off + SwingH=Auto should set swing mode to horizontal."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = make_mqtt_state_payload({
            "Power": "On", "Mode": "Heat",
            "SwingV": "Off", "SwingH": "Auto",
        })
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_HORIZONTAL

    @pytest.mark.asyncio
    async def test_state_updates_swing_off(self, hass, setup_integration):
        """Neither swing auto should set swing mode to off."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = make_mqtt_state_payload({
            "Power": "On", "Mode": "Heat",
            "SwingV": "Off", "SwingH": "Off",
        })
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_OFF

    @pytest.mark.asyncio
    async def test_state_power_off_sets_off_mode(self, hass, setup_integration):
        """Power=Off in MQTT payload should set HVAC mode to OFF."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        # First turn on
        payload = make_mqtt_state_payload({"Power": "On", "Mode": "Heat"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()
        assert entity._attr_hvac_mode == HVACMode.HEAT

        # Now power off
        payload = make_mqtt_state_payload({"Power": "Off", "Mode": "Heat"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()
        assert entity._attr_hvac_mode == HVACMode.OFF

    @pytest.mark.asyncio
    async def test_state_ignore_off_temp(self, hass, setup_integration):
        """With ignore_off_temp, temp should not update when power is off."""
        entry = await setup_integration({"ignore_off_temp": True})
        entity = get_climate_entity(hass, entry)

        # Set initial temp
        entity._attr_target_temperature = 22.0

        # Fire power-off payload with different temp
        payload = make_mqtt_state_payload({"Power": "Off", "Mode": "Heat", "Temp": 30})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_target_temperature == 22.0  # Should NOT be 30

    @pytest.mark.asyncio
    async def test_vendor_mismatch_ignored(self, hass, setup_integration):
        """Payload from wrong vendor should be ignored."""
        entry = await setup_integration({"vendor": "FUJITSU_AC"})
        entity = get_climate_entity(hass, entry)

        entity._attr_hvac_mode = HVACMode.OFF

        # Send payload from different vendor
        payload = json.dumps({"IRHVAC": {
            "Vendor": "MITSUBISHI_AC", "Power": "On", "Mode": "Heat", "Temp": 25,
        }})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        # Should still be off (Fujitsu entity ignores Mitsubishi payloads)
        assert entity._attr_hvac_mode == HVACMode.OFF


class TestMQTTAvailability:
    """Tests for MQTT availability topic handling."""

    @pytest.mark.asyncio
    async def test_availability_online(self, hass, setup_integration):
        """Online message should make entity available."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(hass, "tele/irhvac/LWT", "Online")
        await hass.async_block_till_done()

        state = hass.states.get(entity.entity_id)
        assert state is not None
        assert state.state != "unavailable"

    @pytest.mark.asyncio
    async def test_availability_offline(self, hass, setup_integration):
        """Offline message should make entity unavailable."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(hass, "tele/irhvac/LWT", "Offline")
        await hass.async_block_till_done()

        state = hass.states.get(entity.entity_id)
        assert state is not None
        assert state.state == "unavailable"


class TestDualTopicEcho:
    """Tests for duplicate MQTT echoes from tele + stat topics."""

    @pytest.mark.asyncio
    async def test_dual_echo_single_ir_send(self, hass, setup_pi_integration):
        """Two echoes from tele + stat should not produce extra IR sends."""
        from unittest.mock import AsyncMock, patch

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Set up heating mode with a known state
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 22.0
        pi._hp_setpoint = 24
        # Ensure send time is recent (simulates we just sent a command)
        import time as _time
        pi._last_send_ir_time = _time.monotonic()
        pi._pi_last_tick_time = _time.monotonic()

        # Patch send_ir to track calls
        with patch.object(entity, 'send_ir', new_callable=AsyncMock) as mock_send:
            # First echo (tele topic) — within echo window, treated as self-echo
            payload = make_mqtt_state_payload({"Temp": 24, "Mode": "Heat"})
            async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
            await hass.async_block_till_done()

            # Second echo (stat topic) — also within echo window
            async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
            await hass.async_block_till_done()

            # send_ir should NOT have been called from echo processing
            assert mock_send.call_count == 0, (
                f"send_ir called {mock_send.call_count} times from echo — expected 0"
            )

    @pytest.mark.asyncio
    async def test_dual_echo_preserves_desired_temp(self, hass, setup_pi_integration):
        """Both echoes should preserve the user's desired temperature."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        import time as _time
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 21.5
        entity._attr_target_temperature = 21.5
        pi._last_send_ir_time = _time.monotonic()

        # Two echoes back to back
        payload = make_mqtt_state_payload({"Temp": 24, "Mode": "Heat"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        # desired_temp should still be 21.5 after both echoes
        assert pi._desired_temp == 21.5
        assert entity._attr_target_temperature == 21.5

    @pytest.mark.asyncio
    async def test_echo_with_wrong_temp_does_not_corrupt_setpoint(self, hass, setup_pi_integration):
        """Echo with different Temp (e.g. from IR receiver misdecode) must not corrupt hp_setpoint.

        Regression: prior to fix, handle_state_payload unconditionally set
        _hp_setpoint = payload["Temp"] for every echo, including our own.
        If the echo had a different Temp (stale echo, IR misdecode, Tasmota rounding),
        it would corrupt hp_setpoint and the next state write would show the wrong value.
        """
        from unittest.mock import AsyncMock, patch
        import time as _time

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        entity._attr_hvac_mode = HVACMode.HEAT
        pi._desired_temp = 22.0
        pi._hp_setpoint = 25  # PI just computed this
        pi._last_send_ir_time = _time.monotonic()
        pi._pi_last_tick_time = _time.monotonic()

        with patch.object(entity, 'send_ir', new_callable=AsyncMock) as mock_send:
            # Echo arrives with Temp=22 (wrong — we sent 25) within echo window
            # Non-IrReceived echo with mismatched temp: Case 3 (telemetry mismatch)
            # would resend, but since send_ir is recent (<2s) and no IrReceived,
            # the non-IrReceived path treats it as telemetry mismatch.
            # However, with send_ir mocked, the resend won't change hp_setpoint.
            payload = make_mqtt_state_payload({"Temp": 22, "Mode": "Heat"})
            async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
            await hass.async_block_till_done()

            # hp_setpoint must remain 25 — the mismatch triggers a resend,
            # not an hp_setpoint overwrite
            assert pi._hp_setpoint == 25, (
                f"hp_setpoint corrupted to {pi._hp_setpoint} by echo with Temp=22"
            )

            # Second echo also with wrong Temp
            async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
            await hass.async_block_till_done()

            assert pi._hp_setpoint == 25
            # Telemetry mismatch triggers resends (Case 3) — this is correct
            # behavior. The important thing is hp_setpoint was NOT corrupted.

    @pytest.mark.asyncio
    async def test_reentrant_tick_blocked_during_send(self, hass, setup_pi_integration):
        """If _pi_tick is already running (mid-send_ir), a reentrant call should be skipped.

        Simulates: timer tick starts → await send_ir() yields → sensor event
        fires → tries to call _pi_tick again → should see _pi_tick_running and bail.
        """
        from unittest.mock import AsyncMock, patch

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 24

        # Simulate a tick already in progress (mid-send_ir await)
        pi._pi_tick_running = True

        with patch.object(entity, 'send_ir', new_callable=AsyncMock) as mock_send:
            # This tick should be blocked by reentrancy guard
            await pi._pi_tick()

            assert mock_send.call_count == 0, (
                f"Reentrant tick should not send IR, but sent {mock_send.call_count} times"
            )

        # Clean up
        pi._pi_tick_running = False

    @pytest.mark.asyncio
    async def test_physical_remote_change_updates_desired_and_reticks(self, hass, setup_pi_integration):
        """Physical remote (IrReceived wrapper) should update desired_temp and re-tick.

        When someone uses the physical remote to set 22°C, the blaster captures the IR
        and publishes on tele with IrReceived wrapper. The PI should:
        1. Recognize it as physical remote (IrReceived wrapper present)
        2. Set desired_temp = 22
        3. Re-tick to compute HP setpoint = 22 + FF_offset
        4. Send the corrected HP setpoint via IR
        """
        from unittest.mock import AsyncMock, patch

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0
        pi._desired_temp = 20.0
        pi._hp_setpoint = 24
        pi._last_send_ir_time = 0  # Long ago — not a recent send

        with patch.object(entity, 'send_ir', new_callable=AsyncMock) as mock_send:
            # Physical remote sets 22°C — blaster publishes with IrReceived wrapper
            irhvac_payload = json.loads(make_mqtt_state_payload({"Temp": 22, "Mode": "Heat"}))
            ir_received_payload = json.dumps({"IrReceived": irhvac_payload})
            async_fire_mqtt_message(hass, "tele/irhvac/RESULT", ir_received_payload)
            await hass.async_block_till_done()

            # desired_temp should now be 22 (from the remote)
            assert pi._desired_temp == 22.0, (
                f"desired_temp not updated from remote: {pi._desired_temp}"
            )
            # PI should have re-ticked and sent a corrected setpoint
            # (22 + FF offset, which will be > 22 in heating mode)
            assert mock_send.call_count >= 1, (
                "PI should have sent a corrected HP setpoint after remote change"
            )

    @pytest.mark.asyncio
    async def test_telemetry_echo_does_not_overwrite_desired(self, hass, setup_pi_integration):
        """Non-IrReceived echo (telemetry/command echo) must not overwrite desired_temp.

        This is the bug that caused desired_temp corruption: periodic telemetry
        reports the HP's °C setpoint, which was being written to desired_temp.
        """
        from unittest.mock import AsyncMock, patch

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        entity._attr_hvac_mode = HVACMode.HEAT
        entity._attr_current_temperature = 20.0
        pi._desired_temp = 20.0
        pi._hp_setpoint = 24
        pi._last_send_ir_time = 0  # Long ago — not a recent send

        with patch.object(entity, 'send_ir', new_callable=AsyncMock):
            # Telemetry echo (no IrReceived wrapper) reports HP at 22°C
            payload = make_mqtt_state_payload({"Temp": 22, "Mode": "Heat"})
            async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
            await hass.async_block_till_done()

            # desired_temp must NOT change — this is telemetry, not a user action
            assert pi._desired_temp == 20.0, (
                f"desired_temp corrupted by telemetry echo: {pi._desired_temp}"
            )


class TestIrRecvWrapper:
    """Tests for IrReceived wrapper parsing."""

    @pytest.mark.asyncio
    async def test_irrecv_wrapper(self, hass, setup_integration):
        """IrReceived-wrapped IRHVAC payload should be parsed."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = json.dumps({"IrReceived": {"IRHVAC": {
            "Vendor": "FUJITSU_AC", "Power": "On", "Mode": "Cool", "Temp": 23,
            "Celsius": "On", "FanSpeed": "Auto",
            "SwingV": "Off", "SwingH": "Off",
            "Quiet": "Off", "Turbo": "Off", "Econo": "Off",
            "Light": "Off", "Filter": "Off", "Clean": "Off",
            "Beep": "Off", "Sleep": "-1",
        }}})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_hvac_mode == HVACMode.COOL

    @pytest.mark.asyncio
    async def test_missing_irhvac_key_ignored(self, hass, setup_integration):
        """Payload without IRHVAC key should be silently ignored."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        old_mode = entity._attr_hvac_mode

        payload = json.dumps({"SomeOtherKey": {"value": 123}})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()

        assert entity._attr_hvac_mode == old_mode


class TestLastOnMode:
    """Tests for last_on_mode tracking."""

    @pytest.mark.asyncio
    async def test_last_on_mode_tracked(self, hass, setup_integration):
        """last_on_mode should track the most recent non-OFF mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        # Set to heat
        payload = make_mqtt_state_payload({"Power": "On", "Mode": "Heat"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()
        assert entity._last_on_mode == HVACMode.HEAT

        # Set to cool
        payload = make_mqtt_state_payload({"Power": "On", "Mode": "Cool"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()
        assert entity._last_on_mode == HVACMode.COOL

        # Power off — last_on_mode should stay cool
        payload = make_mqtt_state_payload({"Power": "Off", "Mode": "Cool"})
        async_fire_mqtt_message(hass, "tele/irhvac/RESULT", payload)
        await hass.async_block_till_done()
        assert entity._last_on_mode == HVACMode.COOL
