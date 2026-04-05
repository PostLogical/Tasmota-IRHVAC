"""Payload snapshot tests: lock down what send_ir() actually sends to MQTT.

These tests capture the exact MQTT payload for a matrix of entity states.
They serve as the behavioral contract before any refactoring — if a payload
field changes after rewiring vendor handlers, we catch it immediately.

Run against both architecture-rework and config-flow branches to verify
we're testing genuine upstream behavior.
"""

import json
from unittest.mock import patch, AsyncMock

import pytest
from homeassistant.components.climate.const import (
    FAN_HIGH,
    FAN_LOW,
    HVACMode,
    PRESET_AWAY,
    PRESET_NONE,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)
from homeassistant.const import STATE_OFF, STATE_ON

from pytest_homeassistant_custom_component.common import async_fire_mqtt_message

from .conftest import get_climate_entity, make_config, make_mqtt_state_payload


# ── Helpers ──────────────────────────────────────────────────────────


async def capture_ir_payload(entity) -> dict:
    """Call send_ir() and return the parsed JSON payload that was published."""
    with patch(
        "homeassistant.components.mqtt.async_publish", new_callable=AsyncMock
    ) as mock_pub:
        await entity.send_ir()
        assert mock_pub.await_count == 1, "send_ir should publish exactly once"
        raw = mock_pub.await_args[0][2]  # (hass, topic, payload)
        return json.loads(raw)


# ── 1. Mode × Power matrix ──────────────────────────────────────────


class TestPayloadModePower:
    """Verify Mode and Power fields for each HVAC mode."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "hvac_mode,expected_mode,expected_power",
        [
            (HVACMode.HEAT, "heat", "on"),
            (HVACMode.COOL, "cool", "on"),
            (HVACMode.DRY, "dry", "on"),
            (HVACMode.FAN_ONLY, "fan_only", "on"),
            (HVACMode.OFF, "off", "off"),
        ],
    )
    async def test_mode_and_power(
        self, hass, setup_integration, hvac_mode, expected_mode, expected_power
    ):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        await entity.set_mode(hvac_mode)
        entity._attr_target_temperature = 22

        payload = await capture_ir_payload(entity)

        assert payload["Mode"] == expected_mode
        assert payload["Power"] == expected_power

    @pytest.mark.asyncio
    async def test_vendor_in_payload(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        payload = await capture_ir_payload(entity)
        assert payload["Vendor"] == "FUJITSU_AC"

    @pytest.mark.asyncio
    async def test_vendor_mitsubishi(self, hass, setup_integration):
        entry = await setup_integration({"vendor": "MITSUBISHI_AC"})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        payload = await capture_ir_payload(entity)
        assert payload["Vendor"] == "MITSUBISHI_AC"

    @pytest.mark.asyncio
    async def test_model_in_payload(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        payload = await capture_ir_payload(entity)
        assert payload["Model"] == -1

    @pytest.mark.asyncio
    async def test_state_mode_default(self, hass, setup_integration):
        """StateMode should be SendStore by default."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        payload = await capture_ir_payload(entity)
        assert payload["StateMode"] == "SendStore"

    @pytest.mark.asyncio
    async def test_state_mode_resets_after_send(self, hass, setup_integration):
        """StateMode should reset to SendStore after each send."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._state_mode = "StoreOnly"

        payload1 = await capture_ir_payload(entity)
        assert payload1["StateMode"] == "StoreOnly"

        payload2 = await capture_ir_payload(entity)
        assert payload2["StateMode"] == "SendStore"


# ── 2. keep_mode_when_off ────────────────────────────────────────────


class TestPayloadKeepMode:
    """Verify keep_mode_when_off affects the Mode field."""

    @pytest.mark.asyncio
    async def test_keep_mode_off_sends_current_mode(self, hass, setup_integration):
        """With keep_mode=False, Mode should be current hvac_mode."""
        entry = await setup_integration({"keep_mode_when_off": False})
        entity = get_climate_entity(hass, entry)
        await entity.set_mode(HVACMode.HEAT)

        payload = await capture_ir_payload(entity)
        assert payload["Mode"] == "heat"

    @pytest.mark.asyncio
    async def test_keep_mode_on_sends_last_on_mode(self, hass, setup_integration):
        """With keep_mode=True, Mode should be _last_on_mode even when OFF."""
        entry = await setup_integration({"keep_mode_when_off": True})
        entity = get_climate_entity(hass, entry)

        # Set heat, then turn off
        await entity.set_mode(HVACMode.HEAT)
        assert entity._last_on_mode == "heat"
        await entity.set_mode(HVACMode.OFF)

        payload = await capture_ir_payload(entity)
        assert payload["Power"] == "off"
        assert payload["Mode"] == "heat"  # last_on_mode, not "off"

    @pytest.mark.asyncio
    async def test_no_keep_mode_off_sends_off(self, hass, setup_integration):
        """With keep_mode=False, Mode should be 'off' when turned off."""
        entry = await setup_integration({"keep_mode_when_off": False})
        entity = get_climate_entity(hass, entry)
        await entity.set_mode(HVACMode.HEAT)
        await entity.set_mode(HVACMode.OFF)

        payload = await capture_ir_payload(entity)
        assert payload["Power"] == "off"
        assert payload["Mode"] == "off"


# ── 3. Fan speed ─────────────────────────────────────────────────────


class TestPayloadFanSpeed:
    """Verify FanSpeed field in payload."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fan_mode", ["auto", "low", "medium", "high"])
    async def test_fan_modes(self, hass, setup_integration, fan_mode):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_fan_mode = fan_mode

        payload = await capture_ir_payload(entity)
        assert payload["FanSpeed"] == fan_mode


# ── 4. Swing mode encoding ──────────────────────────────────────────


class TestPayloadSwing:
    """Verify SwingV and SwingH fields for each swing mode."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "swing_mode,expected_v,expected_h",
        [
            (SWING_BOTH, "auto", "auto"),
            (SWING_VERTICAL, "auto", "off"),
            (SWING_HORIZONTAL, "off", "auto"),
            (SWING_OFF, "off", "off"),
        ],
    )
    async def test_swing_encoding(
        self, hass, setup_integration, swing_mode, expected_v, expected_h
    ):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_swing_mode = swing_mode
        # Clear any fixation so we test the mode-based logic
        entity._fix_swingv = None
        entity._fix_swingh = None

        payload = await capture_ir_payload(entity)
        assert payload["SwingV"] == expected_v
        assert payload["SwingH"] == expected_h

    @pytest.mark.asyncio
    async def test_swing_fixation_preserved(self, hass, setup_integration):
        """When _fix_swingv/swingh are set, they override the default."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_swing_mode = SWING_OFF
        entity._fix_swingv = "highest"
        entity._fix_swingh = "right"

        payload = await capture_ir_payload(entity)
        assert payload["SwingV"] == "highest"
        assert payload["SwingH"] == "right"


# ── 5. Temperature ───────────────────────────────────────────────────


class TestPayloadTemperature:
    """Verify Temp and Celsius fields."""

    @pytest.mark.asyncio
    async def test_celsius_mode(self, hass, setup_integration):
        entry = await setup_integration({"celsius_mode": "on"})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_target_temperature = 22

        payload = await capture_ir_payload(entity)
        assert payload["Celsius"] == "on"
        assert payload["Temp"] == 22.0

    @pytest.mark.asyncio
    async def test_half_degree_precision(self, hass, setup_integration):
        """Temp should round to precision."""
        entry = await setup_integration({"precision": 0.5, "temp_step": 0.5})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_target_temperature = 22.3  # Should round to 22.5

        payload = await capture_ir_payload(entity)
        assert payload["Temp"] == 22.5

    @pytest.mark.asyncio
    async def test_temp_clamped_above_50c(self, hass, setup_integration):
        """Temps above 50°C should be clamped."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_target_temperature = 55  # Above 50°C clamp

        payload = await capture_ir_payload(entity)
        assert payload["Temp"] == 50.0

    @pytest.mark.asyncio
    async def test_temp_clamped_below_0c(self, hass, setup_integration):
        """Temps below 0°C should be clamped."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_target_temperature = -5  # Below 0°C clamp

        payload = await capture_ir_payload(entity)
        assert payload["Temp"] == 0.0

    @pytest.mark.asyncio
    async def test_temp_at_boundary_50c(self, hass, setup_integration):
        """Temp exactly at 50°C should NOT be clamped."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_target_temperature = 50

        payload = await capture_ir_payload(entity)
        assert payload["Temp"] == 50.0

    @pytest.mark.asyncio
    async def test_temp_at_boundary_0c(self, hass, setup_integration):
        """Temp exactly at 0°C should NOT be clamped."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._attr_target_temperature = 0

        # 0°C should pass through (it's at the boundary, not below)
        payload = await capture_ir_payload(entity)
        assert payload["Temp"] == 0.0


# ── 6. Toggle flags ──────────────────────────────────────────────────


class TestPayloadToggles:
    """Verify toggle flags in payload and reset-after-send behavior."""

    @pytest.mark.asyncio
    async def test_default_toggles_off(self, hass, setup_integration):
        """All toggles should default to 'off'."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        payload = await capture_ir_payload(entity)
        for key in ["Quiet", "Turbo", "Econo", "Light", "Filter", "Clean", "Beep"]:
            assert payload[key] == "off", f"{key} should be 'off'"
        assert payload["Sleep"] == "-1"

    @pytest.mark.asyncio
    async def test_turbo_on_in_payload(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._turbo = "on"

        payload = await capture_ir_payload(entity)
        assert payload["Turbo"] == "on"

    @pytest.mark.asyncio
    async def test_econo_on_in_payload(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._econo = "on"

        payload = await capture_ir_payload(entity)
        assert payload["Econo"] == "on"

    @pytest.mark.asyncio
    async def test_quiet_on_in_payload(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._quiet = "on"

        payload = await capture_ir_payload(entity)
        assert payload["Quiet"] == "on"

    @pytest.mark.asyncio
    async def test_toggle_list_resets_after_send(self, hass, setup_integration):
        """Toggles in _toggle_list should reset to 'off' after send_ir."""
        entry = await setup_integration({"toggle_list": ["turbo", "econo"]})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._turbo = "on"
        entity._econo = "on"

        # First send captures the "on" values
        payload1 = await capture_ir_payload(entity)
        assert payload1["Turbo"] == "on"
        assert payload1["Econo"] == "on"

        # Second send should have them reset to "off"
        payload2 = await capture_ir_payload(entity)
        assert payload2["Turbo"] == "off"
        assert payload2["Econo"] == "off"

    @pytest.mark.asyncio
    async def test_non_toggle_list_not_reset(self, hass, setup_integration):
        """Toggles NOT in _toggle_list should persist across sends."""
        entry = await setup_integration({"toggle_list": []})
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON
        entity._light = "on"

        payload1 = await capture_ir_payload(entity)
        assert payload1["Light"] == "on"

        payload2 = await capture_ir_payload(entity)
        assert payload2["Light"] == "on"  # Not reset because not in toggle_list


# ── 7. Clock and Weekday ─────────────────────────────────────────────


class TestPayloadClock:
    """Verify Clock and Weekday fields use current time."""

    @pytest.mark.asyncio
    async def test_clock_and_weekday_present(self, hass, setup_integration):
        """Clock should be 0-1439 (minutes since midnight), Weekday 0-6."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT
        entity.power_mode = STATE_ON

        payload = await capture_ir_payload(entity)
        assert 0 <= payload["Clock"] <= 1439
        assert 0 <= payload["Weekday"] <= 6


# ── 8. State receive: field processing ───────────────────────────────


class TestStateReceive:
    """Verify _handle_state_payload processes fields correctly."""

    @pytest.mark.asyncio
    async def test_mode_receive_heat(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Mode": "Heat", "Power": "On"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_hvac_mode == HVACMode.HEAT

    @pytest.mark.asyncio
    async def test_mode_receive_fan_maps_to_fan_only(self, hass, setup_integration):
        """Tasmota sends 'fan' but HA expects 'fan_only'."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Mode": "fan", "Power": "On"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_hvac_mode == HVACMode.FAN_ONLY

    @pytest.mark.asyncio
    async def test_power_off_sets_mode_off(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_hvac_mode = HVACMode.HEAT

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Power": "Off"}),
        )
        await hass.async_block_till_done()

        assert entity.power_mode == "off"

    @pytest.mark.asyncio
    async def test_temp_receive_updates_target(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Temp": 25, "Power": "On"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_target_temperature == 25

    @pytest.mark.asyncio
    async def test_impossible_temp_rejected(self, hass, setup_integration):
        """Temps outside 0-50°C range should be ignored."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_target_temperature = 22  # Set known value

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Temp": 99, "Power": "On"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_target_temperature == 22  # Unchanged

    @pytest.mark.asyncio
    async def test_ignore_off_temp_true(self, hass, setup_integration):
        """With ignore_off_temp=True, temp should not change when power is off."""
        entry = await setup_integration({"ignore_off_temp": True})
        entity = get_climate_entity(hass, entry)
        entity._attr_target_temperature = 22

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Temp": 18, "Power": "Off"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_target_temperature == 22  # Unchanged

    @pytest.mark.asyncio
    async def test_ignore_off_temp_false(self, hass, setup_integration):
        """With ignore_off_temp=False, temp should update even when power is off."""
        entry = await setup_integration({"ignore_off_temp": False})
        entity = get_climate_entity(hass, entry)
        entity._attr_target_temperature = 22

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Temp": 18, "Power": "Off"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_target_temperature == 18

    @pytest.mark.asyncio
    async def test_vendor_mismatch_ignored(self, hass, setup_integration):
        """Payloads from a different vendor should be ignored."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_target_temperature = 22

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"Vendor": "SAMSUNG_AC", "Temp": 30}),
        )
        await hass.async_block_till_done()

        assert entity._attr_target_temperature == 22  # Unchanged

    @pytest.mark.asyncio
    async def test_toggle_flags_stored(self, hass, setup_integration):
        """Toggle flags from payload should be stored on entity."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({
                "Quiet": "On", "Turbo": "On", "Econo": "Off",
                "Light": "On", "Filter": "Off", "Clean": "Off",
                "Beep": "On",
            }),
        )
        await hass.async_block_till_done()

        assert entity._quiet == "on"
        assert entity._turbo == "on"
        assert entity._econo == "off"
        assert entity._light == "on"
        assert entity._filter == "off"
        assert entity._clean == "off"
        assert entity._beep == "on"

    @pytest.mark.asyncio
    async def test_fan_speed_stored(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"FanSpeed": "High"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_fan_mode == "high"

    @pytest.mark.asyncio
    async def test_missing_fan_speed_no_crash(self, hass, setup_integration):
        """Payload without FanSpeed should not crash or change fan_mode."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_fan_mode = "auto"

        # Build payload without FanSpeed
        payload = {
            "Vendor": "FUJITSU_AC", "Power": "On", "Mode": "Heat",
            "Temp": 22, "Celsius": "On",
        }
        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            json.dumps({"IRHVAC": payload}),
        )
        await hass.async_block_till_done()

        assert entity._attr_fan_mode == "auto"  # Unchanged

    @pytest.mark.asyncio
    async def test_swing_fixation_from_payload(self, hass, setup_integration):
        """Non-auto swing values should set _fix_swingv/swingh."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"SwingV": "highest", "SwingH": "right"}),
        )
        await hass.async_block_till_done()

        assert entity._fix_swingv == "highest"
        assert entity._fix_swingh == "right"

    @pytest.mark.asyncio
    async def test_swing_auto_clears_fixation(self, hass, setup_integration):
        """Auto swing values should NOT set fixation (fixation only for non-auto)."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._fix_swingv = "highest"  # Pre-set fixation

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"SwingV": "Auto", "SwingH": "Auto"}),
        )
        await hass.async_block_till_done()

        # _fix_swingv should NOT be cleared by "auto" — only non-auto values set it
        # The existing fixation persists
        assert entity._fix_swingv == "highest"


# ── 9. Swing mode receive combinations ───────────────────────────────


class TestSwingReceive:
    """Verify swing mode derivation from SwingV + SwingH in payload."""

    @pytest.mark.asyncio
    async def test_both_auto_becomes_swing_both(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"SwingV": "Auto", "SwingH": "Auto"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_BOTH

    @pytest.mark.asyncio
    async def test_v_auto_h_off_becomes_vertical(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"SwingV": "Auto", "SwingH": "Off"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_VERTICAL

    @pytest.mark.asyncio
    async def test_v_off_h_auto_becomes_horizontal(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"SwingV": "Off", "SwingH": "Auto"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_HORIZONTAL

    @pytest.mark.asyncio
    async def test_both_off_becomes_swing_off(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            make_mqtt_state_payload({"SwingV": "Off", "SwingH": "Off"}),
        )
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_OFF

    @pytest.mark.asyncio
    async def test_only_swingv_auto(self, hass, setup_integration):
        """SwingV=Auto without SwingH should give SWING_VERTICAL."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = {
            "Vendor": "FUJITSU_AC", "Power": "On", "Mode": "Heat",
            "Temp": 22, "Celsius": "On", "SwingV": "Auto",
        }
        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            json.dumps({"IRHVAC": payload}),
        )
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_VERTICAL

    @pytest.mark.asyncio
    async def test_only_swingh_auto(self, hass, setup_integration):
        """SwingH=Auto without SwingV should give SWING_HORIZONTAL."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        payload = {
            "Vendor": "FUJITSU_AC", "Power": "On", "Mode": "Heat",
            "Temp": 22, "Celsius": "On", "SwingH": "Auto",
        }
        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            json.dumps({"IRHVAC": payload}),
        )
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_HORIZONTAL

    @pytest.mark.asyncio
    async def test_no_swing_fields_in_payload(self, hass, setup_integration):
        """Payload without SwingV/SwingH should set SWING_OFF."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_BOTH  # Start with known state

        payload = {
            "Vendor": "FUJITSU_AC", "Power": "On", "Mode": "Heat",
            "Temp": 22, "Celsius": "On",
        }
        async_fire_mqtt_message(
            hass, "tele/irhvac/RESULT",
            json.dumps({"IRHVAC": payload}),
        )
        await hass.async_block_till_done()

        assert entity._attr_swing_mode == SWING_OFF


# ── 10. Full payload golden snapshot ─────────────────────────────────


class TestGoldenPayload:
    """Complete payload snapshot for a known state — catches any field drift."""

    @pytest.mark.asyncio
    async def test_golden_heat_payload(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        await entity.set_mode(HVACMode.HEAT)
        entity._attr_target_temperature = 22
        entity._attr_fan_mode = "auto"
        entity._attr_swing_mode = SWING_VERTICAL
        entity._fix_swingv = None
        entity._fix_swingh = None

        payload = await capture_ir_payload(entity)

        # Check time fields separately (timezone-dependent)
        assert 0 <= payload.pop("Clock") <= 1439
        assert 0 <= payload.pop("Weekday") <= 6

        assert payload == {
            "StateMode": "SendStore",
            "Vendor": "FUJITSU_AC",
            "Model": -1,
            "Power": "on",
            "Mode": "heat",
            "Celsius": "on",
            "Temp": 22.0,
            "FanSpeed": "auto",
            "SwingV": "auto",
            "SwingH": "off",
            "Quiet": "off",
            "Turbo": "off",
            "Econo": "off",
            "Light": "off",
            "Filter": "off",
            "Clean": "off",
            "Beep": "off",
            "Sleep": "-1",
        }

    @pytest.mark.asyncio
    async def test_golden_cool_payload(self, hass, setup_integration):
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        await entity.set_mode(HVACMode.COOL)
        entity._attr_target_temperature = 25
        entity._attr_fan_mode = "high"
        entity._attr_swing_mode = SWING_BOTH
        entity._fix_swingv = None
        entity._fix_swingh = None

        payload = await capture_ir_payload(entity)

        assert 0 <= payload.pop("Clock") <= 1439
        assert 0 <= payload.pop("Weekday") <= 6

        assert payload == {
            "StateMode": "SendStore",
            "Vendor": "FUJITSU_AC",
            "Model": -1,
            "Power": "on",
            "Mode": "cool",
            "Celsius": "on",
            "Temp": 25.0,
            "FanSpeed": "high",
            "SwingV": "auto",
            "SwingH": "auto",
            "Quiet": "off",
            "Turbo": "off",
            "Econo": "off",
            "Light": "off",
            "Filter": "off",
            "Clean": "off",
            "Beep": "off",
            "Sleep": "-1",
        }
