"""Tier 8: Button entity tests."""

import pytest

from homeassistant.components.climate.const import (
    SWING_BOTH,
    SWING_HORIZONTAL,
    SWING_OFF,
    SWING_VERTICAL,
)
from homeassistant.core import HomeAssistant

from .conftest import get_climate_entity


class TestVaneButtonSetup:
    """Tests for vane button entity creation."""

    @pytest.mark.asyncio
    async def test_set_v_button_created(self, hass, setup_integration):
        """has_set_vertical_vane=True should create a Set V button."""
        entry = await setup_integration({"has_set_vertical_vane": True})

        all_buttons = hass.states.async_all("button")
        button_ids = [s.entity_id for s in all_buttons]
        assert any("set_v" in b for b in button_ids), f"No Set V button in {button_ids}"

    @pytest.mark.asyncio
    async def test_set_h_button_created(self, hass, setup_integration):
        """has_set_horizontal_vane=True should create a Set H button."""
        entry = await setup_integration({"has_set_horizontal_vane": True})

        all_buttons = hass.states.async_all("button")
        button_ids = [s.entity_id for s in all_buttons]
        assert any("set_h" in b for b in button_ids), f"No Set H button in {button_ids}"


class TestVaneButtonPress:
    """Tests for vane button press behavior."""

    @pytest.mark.asyncio
    async def test_set_v_from_both(self, hass, setup_integration):
        """Pressing Set V from SWING_BOTH should change to HORIZONTAL."""
        entry = await setup_integration({
            "has_set_vertical_vane": True,
            "has_set_horizontal_vane": True,
        })
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_BOTH

        # Find and press the Set V button
        all_buttons = hass.states.async_all("button")
        set_v_id = next(s.entity_id for s in all_buttons if "set_v" in s.entity_id)

        await hass.services.async_call(
            "button", "press", {"entity_id": set_v_id}, blocking=True,
        )

        assert entity._attr_swing_mode == SWING_HORIZONTAL
        assert entity._swingv is None

    @pytest.mark.asyncio
    async def test_set_v_from_vertical(self, hass, setup_integration):
        """Pressing Set V from SWING_VERTICAL should change to OFF."""
        entry = await setup_integration({"has_set_vertical_vane": True})
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_VERTICAL

        all_buttons = hass.states.async_all("button")
        set_v_id = next(s.entity_id for s in all_buttons if "set_v" in s.entity_id)

        await hass.services.async_call(
            "button", "press", {"entity_id": set_v_id}, blocking=True,
        )

        assert entity._attr_swing_mode == SWING_OFF

    @pytest.mark.asyncio
    async def test_set_h_from_both(self, hass, setup_integration):
        """Pressing Set H from SWING_BOTH should change to VERTICAL."""
        entry = await setup_integration({
            "has_set_vertical_vane": True,
            "has_set_horizontal_vane": True,
        })
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_BOTH

        all_buttons = hass.states.async_all("button")
        set_h_id = next(s.entity_id for s in all_buttons if "set_h" in s.entity_id)

        await hass.services.async_call(
            "button", "press", {"entity_id": set_h_id}, blocking=True,
        )

        assert entity._attr_swing_mode == SWING_VERTICAL
        assert entity._swingh is None

    @pytest.mark.asyncio
    async def test_set_h_from_horizontal(self, hass, setup_integration):
        """Pressing Set H from SWING_HORIZONTAL should change to OFF."""
        entry = await setup_integration({"has_set_horizontal_vane": True})
        entity = get_climate_entity(hass, entry)
        entity._attr_swing_mode = SWING_HORIZONTAL

        all_buttons = hass.states.async_all("button")
        set_h_id = next(s.entity_id for s in all_buttons if "set_h" in s.entity_id)

        await hass.services.async_call(
            "button", "press", {"entity_id": set_h_id}, blocking=True,
        )

        assert entity._attr_swing_mode == SWING_OFF


class TestIRActionButton:
    """Tests for user-defined IR action buttons."""

    @pytest.mark.asyncio
    async def test_ir_action_button_created(self, hass, setup_integration):
        """IR action with type=button should create a button entity."""
        entry = await setup_integration({
            "ir_actions": [{
                "name": "Test Button",
                "type": "button",
                "ir_code": "raw,0,1234,5678",
            }],
        })

        all_buttons = hass.states.async_all("button")
        button_ids = [s.entity_id for s in all_buttons]
        assert any("test_button" in b for b in button_ids), f"No IR action button in {button_ids}"

    @pytest.mark.asyncio
    async def test_ir_action_button_press(self, hass, setup_integration):
        """Pressing IR action button should publish via MQTT."""
        entry = await setup_integration({
            "ir_actions": [{
                "name": "Test IR",
                "type": "button",
                "ir_code": "raw,0,1234,5678",
            }],
        })

        all_buttons = hass.states.async_all("button")
        ir_button_id = next(
            (s.entity_id for s in all_buttons if "test_ir" in s.entity_id), None
        )
        assert ir_button_id is not None

        # Press the button — should not raise
        await hass.services.async_call(
            "button", "press", {"entity_id": ir_button_id}, blocking=True,
        )


class TestFlushObservationBufferService:
    """Tests for the flush_observation_buffer service."""

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self, hass, setup_pi_integration):
        """Service call should clear the observation buffer."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Add some observations to the buffer
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        import time
        for i in range(5):
            obs = Observation(
                timestamp=time.monotonic() + i,
                features=[1.0, 10.0],
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=22.0,
                room_rate=0.01,
                clamped=False,
            )
            pi._observation_buffer.add(obs)

        assert len(pi._observation_buffer) == 5

        await hass.services.async_call(
            "tasmota_irhvac", "flush_observation_buffer",
            {"entity_id": entity.entity_id}, blocking=True,
        )

        assert len(pi._observation_buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_resets_batch_result(self, hass, setup_pi_integration):
        """Service call should clear the last batch result and drift history."""
        from custom_components.tasmota_irhvac.pi.batch_learning import BatchResult
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._last_batch_result = BatchResult(
            n_total=10,
            n_eligible=8,
            beta_batch=[0.0, 0.3],
            beta_current=[0.0, 0.3],
            residual_rms=0.1,
            max_coeff_change_pct=5.0,
            recommend_update=False,
            n_outliers_excluded=0,
        )
        pi._drift_correction_signs = [[1, 1, 1], [0, -1, 1]]

        await hass.services.async_call(
            "tasmota_irhvac", "flush_observation_buffer",
            {"entity_id": entity.entity_id}, blocking=True,
        )

        assert pi._last_batch_result is None
        assert pi._last_batch_timestamp is None
        assert pi._drift_correction_signs == []

    @pytest.mark.asyncio
    async def test_flush_noop_without_pi(self, hass, setup_integration):
        """Service call on non-PI entity should not raise."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)

        # Should not raise
        await hass.services.async_call(
            "tasmota_irhvac", "flush_observation_buffer",
            {"entity_id": entity.entity_id}, blocking=True,
        )

    @pytest.mark.asyncio
    async def test_flush_buffer_accepts_new_observations(self, hass, setup_pi_integration):
        """After flush, buffer should accept new observations normally."""
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        import time

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Add, flush, add again
        for i in range(3):
            pi._observation_buffer.add(Observation(
                timestamp=time.monotonic() + i,
                features=[1.0, 10.0],
                hp_setpoint=22.0, current_c=21.0, desired_c=22.0,
                room_rate=0.01, clamped=False,
            ))

        await hass.services.async_call(
            "tasmota_irhvac", "flush_observation_buffer",
            {"entity_id": entity.entity_id}, blocking=True,
        )

        # Add new observations after flush
        for i in range(2):
            pi._observation_buffer.add(Observation(
                timestamp=time.monotonic() + 100 + i,
                features=[1.0, 15.0],
                hp_setpoint=23.0, current_c=20.0, desired_c=22.0,
                room_rate=0.02, clamped=False,
            ))

        assert len(pi._observation_buffer) == 2
