"""Tests for the Fujitsu vendor handler."""

import pytest
from unittest.mock import AsyncMock

from homeassistant.components.climate.const import (
    HVACMode,
    PRESET_NONE,
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)

from homeassistant.components.climate.const import PRESET_BOOST, PRESET_ECO
from custom_components.tasmota_irhvac.const import PRESET_MIN_HEAT
from custom_components.tasmota_irhvac.vendors.base import EntityState, IRDecode
from custom_components.tasmota_irhvac.vendors.fujitsu import (
    FUJITSU_DATA_ECONO,
    FUJITSU_DATA_MIN_HEAT,
    FUJITSU_DATA_POWERFUL,
    FUJITSU_DATA_SET_H,
    FUJITSU_DATA_SET_V,
    FUJITSU_IR_ECONO,
    FUJITSU_IR_MIN_HEAT,
    FUJITSU_IR_POWERFUL,
    FUJITSU_IR_STOP,
    FUJITSU_MODEL_3,
    MIN_HEAT_TEMP_C,
    POWERFUL_TIMEOUT_SECONDS,
    FujitsuHandler,
)


def _entity_state(**kwargs) -> EntityState:
    """Build an EntityState with defaults."""
    defaults = dict(
        hvac_mode=HVACMode.COOL,
        target_temperature=22.0,
        fan_mode="auto",
        swing_mode=SWING_OFF,
        swingv="off",
        swingh="off",
        power_mode="on",
    )
    defaults.update(kwargs)
    return EntityState(**defaults)


def _decode(irhvac=None, **kwargs) -> IRDecode:
    """Build an IRDecode with defaults."""
    return IRDecode(irhvac=irhvac or {}, **kwargs)


def _model3_decode(data: str, bits: int = 56) -> IRDecode:
    """Build an IRDecode for a Model 3 / 56-bit command."""
    return IRDecode(
        irhvac={"Model": FUJITSU_MODEL_3, "Power": "On"},
        data=data,
        bits=bits,
    )


# ── Capabilities ─────────────────────────────────────────────────────


class TestFujitsuCapabilities:
    def test_extra_presets(self):
        caps = FujitsuHandler.capabilities()
        assert PRESET_BOOST in caps.extra_preset_modes
        assert PRESET_ECO in caps.extra_preset_modes
        assert PRESET_MIN_HEAT in caps.extra_preset_modes

    def test_has_raw_ir(self):
        assert FujitsuHandler.capabilities().has_raw_ir is True

    def test_supported_toggles(self):
        toggles = FujitsuHandler.capabilities().supported_toggles
        assert "beep" in toggles
        assert "turbo" in toggles
        assert "filter" not in toggles
        assert "sleep" not in toggles

    def test_models_defined(self):
        models = FujitsuHandler.capabilities().models
        assert models is not None
        ids = [m.id for m in models]
        assert "-1" in ids
        assert "1" in ids
        assert "3" in ids


# ── State restore ────────────────────────────────────────────────────


class TestFujitsuRestore:
    def test_restore_min_heat(self):
        h = FujitsuHandler()
        h.on_restore_state(PRESET_MIN_HEAT)
        assert h._min_heat is True
        assert h.should_pause_controller is True

    def test_restore_econo(self):
        h = FujitsuHandler()
        h.on_restore_state(PRESET_ECO)
        assert h._economy is True
        assert h.should_pause_controller is True

    def test_restore_powerful(self):
        h = FujitsuHandler()
        h.on_restore_state(PRESET_BOOST)
        assert h._powerful is True
        assert h.should_pause_controller is True

    def test_restore_none(self):
        h = FujitsuHandler()
        h.on_restore_state(PRESET_NONE)
        assert h.should_pause_controller is False

    def test_restore_null(self):
        h = FujitsuHandler()
        h.on_restore_state(None)
        assert h.should_pause_controller is False


# ── Post state processing: flag → preset mapping ────────────────────


class TestFujitsuFlagMapping:
    def test_turbo_maps_to_powerful(self):
        h = FujitsuHandler()
        decode = _decode(irhvac={"Power": "On", "Turbo": "on"})
        h.pre_state_processing(decode, _entity_state())
        h.post_state_processing(decode)
        assert h.active_preset == PRESET_BOOST
        assert h.should_pause_controller is True

    def test_econo_maps_to_economy(self):
        h = FujitsuHandler()
        decode = _decode(irhvac={"Power": "On", "Econo": "on"})
        h.pre_state_processing(decode, _entity_state())
        h.post_state_processing(decode)
        assert h.active_preset == PRESET_ECO
        assert h.should_pause_controller is True

    def test_clean_maps_to_min_heat(self):
        h = FujitsuHandler()
        decode = _decode(irhvac={"Power": "On", "Clean": "on"})
        h.pre_state_processing(decode, _entity_state())
        h.post_state_processing(decode)
        assert h.active_preset == PRESET_MIN_HEAT
        assert h.should_pause_controller is True

    def test_power_off_clears_all_presets(self):
        h = FujitsuHandler()
        h._powerful = True
        h._economy = True
        h._min_heat = True

        decode = _decode(irhvac={"Power": "off"})
        h.pre_state_processing(decode, _entity_state())
        h.post_state_processing(decode)

        assert h.active_preset is None
        assert h.should_pause_controller is False

    def test_no_flags_no_preset(self):
        h = FujitsuHandler()
        decode = _decode(irhvac={"Power": "On"})
        h.pre_state_processing(decode, _entity_state())
        h.post_state_processing(decode)
        assert h.active_preset is None
        assert h.should_pause_controller is False


# ── 56-bit special command detection ─────────────────────────────────


class TestFujitsu56Bit:
    def test_powerful_detected(self):
        h = FujitsuHandler()
        state = _entity_state()
        decode = _model3_decode(FUJITSU_DATA_POWERFUL)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.active_preset == PRESET_BOOST
        assert h.should_pause_controller is True
        assert h.state_restore is not None
        assert h.state_restore.hvac_mode == HVACMode.COOL  # restored

    def test_econo_detected(self):
        h = FujitsuHandler()
        state = _entity_state()
        decode = _model3_decode(FUJITSU_DATA_ECONO)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.active_preset == PRESET_ECO
        assert h.state_restore is not None

    def test_min_heat_detected(self):
        h = FujitsuHandler()
        state = _entity_state(target_temperature=22.0)
        decode = IRDecode(
            irhvac={"Model": FUJITSU_MODEL_3, "Power": "On"},
            data=FUJITSU_DATA_MIN_HEAT,
            bits=128,  # Min Heat is longer than 56 bits
        )
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.active_preset == PRESET_MIN_HEAT
        assert h.should_pause_controller is True
        assert h.should_reset_integral is True
        assert h.clear_toggles is True
        assert h.state_restore is not None
        assert h.state_restore.hvac_mode == HVACMode.HEAT
        assert h.state_restore.target_temperature == MIN_HEAT_TEMP_C
        assert h.state_restore.power_mode == "on"

    def test_set_v_updates_swing_from_both(self):
        h = FujitsuHandler()
        state = _entity_state(swing_mode=SWING_BOTH, swingv="auto", swingh="auto")
        decode = _model3_decode(FUJITSU_DATA_SET_V)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.state_restore is not None
        assert h.state_restore.swing_mode == SWING_HORIZONTAL
        assert h.state_restore.swingv is None

    def test_set_v_updates_swing_from_vertical(self):
        h = FujitsuHandler()
        state = _entity_state(swing_mode=SWING_VERTICAL)
        decode = _model3_decode(FUJITSU_DATA_SET_V)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.state_restore.swing_mode == SWING_OFF

    def test_set_h_updates_swing_from_both(self):
        h = FujitsuHandler()
        state = _entity_state(swing_mode=SWING_BOTH, swingv="auto", swingh="auto")
        decode = _model3_decode(FUJITSU_DATA_SET_H)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.state_restore.swing_mode == SWING_VERTICAL
        assert h.state_restore.swingh is None

    def test_set_h_updates_swing_from_horizontal(self):
        h = FujitsuHandler()
        state = _entity_state(swing_mode=SWING_HORIZONTAL)
        decode = _model3_decode(FUJITSU_DATA_SET_H)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.state_restore.swing_mode == SWING_OFF

    def test_set_v_noop_when_swing_off(self):
        """Vane set V with swing_mode=OFF keeps OFF."""
        h = FujitsuHandler()
        state = _entity_state(swing_mode=SWING_OFF)
        decode = _model3_decode(FUJITSU_DATA_SET_V)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.state_restore.swing_mode == SWING_OFF

    def test_set_h_noop_when_swing_off(self):
        """Vane set H with swing_mode=OFF keeps OFF."""
        h = FujitsuHandler()
        state = _entity_state(swing_mode=SWING_OFF)
        decode = _model3_decode(FUJITSU_DATA_SET_H)
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        assert h.state_restore.swing_mode == SWING_OFF

    def test_set_v_noop_when_no_state_restore(self):
        """Vane set V is a no-op when _state_restore is None."""
        h = FujitsuHandler()
        h._state_restore = None
        h._apply_vane_set_v()
        assert h.state_restore is None

    def test_set_h_noop_when_no_state_restore(self):
        """Vane set H is a no-op when _state_restore is None."""
        h = FujitsuHandler()
        h._state_restore = None
        h._apply_vane_set_h()
        assert h.state_restore is None

    def test_non_model3_ignores_data_field(self):
        """Non-Model 3 payloads shouldn't trigger 56-bit detection."""
        h = FujitsuHandler()
        state = _entity_state()
        decode = IRDecode(
            irhvac={"Model": 1, "Power": "On", "Turbo": "off"},
            data=FUJITSU_DATA_POWERFUL,
            bits=56,
        )
        h.pre_state_processing(decode, state)
        h.post_state_processing(decode)

        # No 56-bit detection — data field ignored for non-Model 3
        assert h.state_restore is None

    def test_saved_state_consumed_after_processing(self):
        """Saved state should not persist across processing cycles."""
        h = FujitsuHandler()
        state = _entity_state()

        # First cycle: Model 3 command
        decode1 = _model3_decode(FUJITSU_DATA_POWERFUL)
        h.pre_state_processing(decode1, state)
        h.post_state_processing(decode1)
        assert h.state_restore is not None

        # Second cycle: normal command (no Model 3)
        decode2 = _decode(irhvac={"Power": "On"})
        h.pre_state_processing(decode2, state)
        h.post_state_processing(decode2)
        assert h.state_restore is None


# ── Preset mode handling ─────────────────────────────────────────────


class TestFujitsuPresets:
    @pytest.mark.asyncio
    async def test_powerful_sends_ir_and_requests_timer(self):
        h = FujitsuHandler()
        send = AsyncMock()
        state = _entity_state()

        result = await h.handle_preset(PRESET_BOOST, state, send)

        assert result is not None
        send.assert_awaited_once_with(FUJITSU_IR_POWERFUL)
        assert h.active_preset == PRESET_BOOST
        assert h.should_pause_controller is True
        assert result.timer_request is not None
        assert result.timer_request.delay_seconds == POWERFUL_TIMEOUT_SECONDS
        assert result.timer_request.callback_id == "clear_powerful"

    @pytest.mark.asyncio
    async def test_powerful_idempotent(self):
        """Setting Powerful when already active doesn't re-send IR."""
        h = FujitsuHandler()
        h._powerful = True
        send = AsyncMock()

        await h.handle_preset(PRESET_BOOST, _entity_state(), send)

        send.assert_not_awaited()
        assert h.active_preset == PRESET_BOOST

    @pytest.mark.asyncio
    async def test_econo_sends_ir(self):
        h = FujitsuHandler()
        send = AsyncMock()

        result = await h.handle_preset(PRESET_ECO, _entity_state(), send)

        assert result is not None
        send.assert_awaited_once_with(FUJITSU_IR_ECONO)
        assert h.active_preset == PRESET_ECO
        assert h.should_pause_controller is True

    @pytest.mark.asyncio
    async def test_min_heat_sends_ir_and_sets_state(self):
        h = FujitsuHandler()
        send = AsyncMock()
        state = _entity_state(target_temperature=22.0)

        result = await h.handle_preset(PRESET_MIN_HEAT, state, send)

        assert result is not None
        send.assert_awaited_once_with(FUJITSU_IR_MIN_HEAT)
        assert h.active_preset == PRESET_MIN_HEAT
        assert h.should_pause_controller is True
        assert h.should_reset_integral is True
        assert h.clear_toggles is True
        assert h.saved_target_temp == 22.0
        assert h.state_restore is not None
        assert h.state_restore.hvac_mode == HVACMode.HEAT
        assert h.state_restore.target_temperature == MIN_HEAT_TEMP_C
        assert h.state_restore.power_mode == "on"

    @pytest.mark.asyncio
    async def test_none_clears_all_flags(self):
        h = FujitsuHandler()
        h._powerful = True
        h._economy = True
        h._min_heat = True
        send = AsyncMock()

        result = await h.handle_preset(PRESET_NONE, _entity_state(), send)

        assert result is None  # Falls through to base
        assert h.should_pause_controller is False
        assert h.clear_toggles is True

    @pytest.mark.asyncio
    async def test_away_falls_through(self):
        h = FujitsuHandler()
        send = AsyncMock()

        result = await h.handle_preset("away", _entity_state(), send)

        assert result is None
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exit_min_heat_before_powerful(self):
        """Switching from Min Heat to Powerful sends STOP first."""
        h = FujitsuHandler()
        h._min_heat = True
        send = AsyncMock()

        await h.handle_preset(PRESET_BOOST, _entity_state(), send)

        calls = [c.args[0] for c in send.await_args_list]
        assert calls == [FUJITSU_IR_STOP, FUJITSU_IR_POWERFUL]

    @pytest.mark.asyncio
    async def test_exit_econo_before_powerful(self):
        """Switching from Econo to Powerful toggles Econo off first."""
        h = FujitsuHandler()
        h._economy = True
        send = AsyncMock()

        await h.handle_preset(PRESET_BOOST, _entity_state(), send)

        calls = [c.args[0] for c in send.await_args_list]
        assert calls == [FUJITSU_IR_ECONO, FUJITSU_IR_POWERFUL]

    @pytest.mark.asyncio
    async def test_econo_to_econo_no_double_toggle(self):
        """Re-selecting Econo when already active doesn't toggle it off."""
        h = FujitsuHandler()
        h._economy = True
        send = AsyncMock()

        await h.handle_preset(PRESET_ECO, _entity_state(), send)

        send.assert_not_awaited()
        assert h._economy is True


# ── Timer callback ───────────────────────────────────────────────────


class TestFujitsuTimer:
    def test_clear_powerful(self):
        h = FujitsuHandler()
        h._powerful = True
        h._active_preset = PRESET_BOOST

        h.on_timer("clear_powerful")

        assert h._powerful is False
        assert h.active_preset == PRESET_NONE
        assert h.should_pause_controller is False

    def test_unknown_callback_ignored(self):
        h = FujitsuHandler()
        h.on_timer("unknown")  # Should not raise
