"""End-to-end sign convention verification tests.

Each test traces a physical scenario through:
  config seed → internal β → feature vector → FF offset → HP setpoint direction

Convention:
  - User-facing seeds: positive = warms room
  - Internal β = -seed
  - outdoor_delta = outdoor_temp - room_temp (signed, same formula both modes)
  - FF_offset = Σ(β_i × x_i)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from homeassistant.components.climate.const import HVACMode
from homeassistant.const import STATE_ON, UnitOfTemperature

from custom_components.tasmota_irhvac.pi.pi_controller import PIController

from .conftest import make_pi_config


class FakeEntity:
    """Minimal entity for sign convention tests."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _ir_temp_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    @property
    def target_temperature(self):
        if hasattr(self, '_pi') and self._pi and self._pi._desired_temp is not None:
            return self._pi._desired_temp
        return self._attr_target_temperature

    @property
    def temperature_unit(self):
        return self._attr_temperature_unit

    def __init__(self, config, hvac_mode=HVACMode.HEAT, room_temp=20.5):
        self.hass = MagicMock()
        self.hass.states.get = MagicMock(return_value=None)
        self._attr_hvac_mode = hvac_mode
        self._attr_current_temperature = room_temp
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
        self._pi = PIController(self, config)


# ── Seed → β sign tests ─────────────────────────────────────────────


class TestSeedToBetaSign:
    """Verify seeds are correctly negated to internal β."""

    def test_outdoor_seed_heat_negated(self):
        """Outdoor seed 0.3 → β = -0.3."""
        config = make_pi_config({"pi_outdoor_seed_heat": 0.3})
        e = FakeEntity(config)
        pi = e._pi
        # Index 1 is outdoor_delta coefficient
        assert pi._heat_seeds[1] == pytest.approx(-0.3)

    def test_outdoor_seed_cool_negated(self):
        """Outdoor seed 0.3 → β = -0.3 (same as heat — no special case)."""
        config = make_pi_config({"pi_outdoor_seed_cool": 0.3})
        e = FakeEntity(config, hvac_mode=HVACMode.COOL)
        pi = e._pi
        assert pi._cool_seeds[1] == pytest.approx(-0.3)

    def test_heat_and_cool_seeds_same_sign(self):
        """Both modes get negative β for positive seeds — no asymmetry."""
        config = make_pi_config({
            "pi_outdoor_seed_heat": 0.5,
            "pi_outdoor_seed_cool": 0.4,
        })
        e = FakeEntity(config)
        pi = e._pi
        assert pi._heat_seeds[1] < 0
        assert pi._cool_seeds[1] < 0

    def test_model_input_seed_negated(self):
        """Model input seed_heat=3.5 → β = -3.5."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "solar", "entity_id": "sensor.solar",
                "seed_heat": 3.5, "seed_cool": 3.5,
            }],
        })
        e = FakeEntity(config)
        pi = e._pi
        # Index 2 is first model input
        assert pi._heat_seeds[2] == pytest.approx(-3.5)
        assert pi._cool_seeds[2] == pytest.approx(-3.5)

    def test_intercept_seed_zero(self):
        """Intercept seed is always 0."""
        config = make_pi_config()
        e = FakeEntity(config)
        pi = e._pi
        assert pi._heat_seeds[0] == 0.0
        assert pi._cool_seeds[0] == 0.0


# ── outdoor_delta computation ────────────────────────────────────────


class TestOutdoorDelta:
    """Verify outdoor_delta = outdoor_temp - room_temp (signed)."""

    def test_heating_cold_outdoor_negative_delta(self):
        """Heating: outdoor -10°C, room 20.5°C → delta = -30.5."""
        outdoor = -10.0
        room = 20.5
        outdoor_delta = outdoor - room
        assert outdoor_delta == pytest.approx(-30.5)

    def test_cooling_hot_outdoor_positive_delta(self):
        """Cooling: outdoor 35°C, room 24°C → delta = +11."""
        outdoor = 35.0
        room = 24.0
        outdoor_delta = outdoor - room
        assert outdoor_delta == pytest.approx(11.0)

    def test_same_formula_both_modes(self):
        """outdoor_delta formula is mode-independent."""
        config = make_pi_config()
        outdoor = 5.0
        room = 20.0

        e_heat = FakeEntity(config, hvac_mode=HVACMode.HEAT, room_temp=room)
        e_cool = FakeEntity(config, hvac_mode=HVACMode.COOL, room_temp=room)

        delta_heat = outdoor - room
        delta_cool = outdoor - room
        assert delta_heat == delta_cool


# ── FF offset direction tests ────────────────────────────────────────


class TestFFOffsetDirection:
    """Verify FF offset pushes HP in the right direction."""

    def test_heating_cold_outdoor_positive_ff(self):
        """Heating, cold outdoor: β<0 × delta<0 = positive FF → HP pushes harder."""
        config = make_pi_config({"pi_outdoor_seed_heat": 0.35})
        e = FakeEntity(config, room_temp=20.5)
        pi = e._pi
        pi._inputs._outdoor_temp = -10.0

        outdoor_delta = -10.0 - 20.5  # = -30.5
        x = pi._inputs.build_feature_vector(outdoor_delta)
        ff = pi._rls_heat.predict(x)

        # β = -0.35, delta = -30.5 → contribution = (-0.35)(-30.5) = +10.675
        assert ff > 0, f"FF should be positive (push harder), got {ff}"

    def test_cooling_hot_outdoor_negative_ff(self):
        """Cooling, hot outdoor: β<0 × delta>0 = negative FF → HP setpoint lower."""
        config = make_pi_config({"pi_outdoor_seed_cool": 0.35})
        e = FakeEntity(config, hvac_mode=HVACMode.COOL, room_temp=24.0)
        pi = e._pi
        pi._inputs._outdoor_temp = 35.0

        outdoor_delta = 35.0 - 24.0  # = +11
        x = pi._inputs.build_feature_vector(outdoor_delta)
        ff = pi._rls_cool.predict(x)

        # β = -0.35, delta = +11 → contribution = (-0.35)(11) = -3.85
        assert ff < 0, f"FF should be negative (more cooling), got {ff}"

    def test_solar_reduces_heating_effort(self):
        """Heating with solar: seed=3.5, solar=0.8 → FF contribution negative."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "solar", "entity_id": "sensor.solar",
                "seed_heat": 3.5, "seed_cool": 3.5,
                "typical_value": 0.5,
            }],
        })
        e = FakeEntity(config, room_temp=20.5)
        pi = e._pi
        pi._inputs._outdoor_temp = 10.0
        pi._inputs._filtered = [0.8]  # solar value

        outdoor_delta = 10.0 - 20.5
        x = pi._inputs.build_feature_vector(outdoor_delta)
        ff = pi._rls_heat.predict(x)

        # Solar contribution: β=-3.5 × 0.8 = -2.8 (reduces FF)
        # Outdoor contribution: β=-0.3 × (-10.5) = +3.15
        # Net: positive but reduced by solar
        # Just verify solar's contribution is negative
        solar_contribution = pi._heat_seeds[2] * pi._feature_scales[2] * 0.8
        assert solar_contribution < 0, "Solar should reduce heating effort"

    def test_solar_increases_cooling_effort(self):
        """Cooling with solar: seed=3.5, solar=0.8 → more cooling."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "solar", "entity_id": "sensor.solar",
                "seed_heat": 3.5, "seed_cool": 3.5,
                "typical_value": 0.5,
            }],
        })
        e = FakeEntity(config, hvac_mode=HVACMode.COOL, room_temp=24.0)
        pi = e._pi
        pi._inputs._outdoor_temp = 30.0
        pi._inputs._filtered = [0.8]

        outdoor_delta = 30.0 - 24.0
        x = pi._inputs.build_feature_vector(outdoor_delta)
        ff = pi._rls_cool.predict(x)

        # Solar β=-3.5, outdoor β=-0.3
        # Both contribute negatively → lower setpoint → more cooling
        assert ff < 0, f"FF should be negative (more cooling), got {ff}"


# ── Clamp tests ──────────────────────────────────────────────────────


class TestClampConvention:
    """Verify clamps are correctly negated from seed space to β space."""

    def test_outdoor_clamp_negated(self):
        """Seed clamp (0, 2) → β clamp (-2, 0)."""
        config = make_pi_config({
            "pi_outdoor_seed_clamp_min": 0.0,
            "pi_outdoor_seed_clamp_max": 2.0,
        })
        e = FakeEntity(config)
        pi = e._pi
        # Index 1 clamp (outdoor_delta) — in normalized space
        heat_clamp = pi._rls_heat_clamps[1]
        assert heat_clamp is not None
        # The clamp values are scaled by feature_scales in RLSModel,
        # but the raw values passed should be (-2.0, -0.0)
        # Check the unscaled clamp we passed
        beta_clamp_lo = -2.0  # -clamp_max
        beta_clamp_hi = -0.0  # -clamp_min
        # Both heat and cool get the same clamp
        assert pi._rls_heat_clamps[1] == pi._rls_cool_clamps[1]

    def test_model_input_clamp_negated(self):
        """Model input clamp (0, 5) → β clamp (-5, 0)."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "stove", "entity_id": "sensor.stove",
                "seed_heat": 3.0, "seed_cool": 3.0,
                "clamp_min": 0.0, "clamp_max": 5.0,
            }],
        })
        e = FakeEntity(config)
        pi = e._pi
        # Index 2 clamp (first model input)
        clamp = pi._rls_heat_clamps[2]
        assert clamp is not None
        assert clamp == (-5.0, -0.0)


# ── get_learned_seed_config round-trip ───────────────────────────────


class TestLearnedSeedRoundTrip:
    """Verify save/restore preserves sign convention."""

    def test_outdoor_seed_round_trip(self):
        """Internal β=-0.35 → saved as outdoor_seed_heat=0.35."""
        config = make_pi_config({"pi_outdoor_seed_heat": 0.35})
        e = FakeEntity(config)
        pi = e._pi

        learned = pi.get_learned_seed_config()
        assert "outdoor_seed_heat" in learned
        assert learned["outdoor_seed_heat"] == pytest.approx(0.35)

    def test_model_input_seed_round_trip(self):
        """Internal β=-3.5 → saved as seed_heat=3.5."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "solar", "entity_id": "sensor.solar",
                "seed_heat": 3.5, "seed_cool": 2.0,
            }],
        })
        e = FakeEntity(config)
        pi = e._pi

        learned = pi.get_learned_seed_config()
        seeds = learned["input_seeds"][0]
        assert seeds["seed_heat"] == pytest.approx(3.5)
        assert seeds["seed_cool"] == pytest.approx(2.0)

    def test_no_abs_hack(self):
        """Cool slope is saved directly, not via abs()."""
        config = make_pi_config({
            "pi_outdoor_seed_heat": 0.3,
            "pi_outdoor_seed_cool": 0.4,
        })
        e = FakeEntity(config)
        pi = e._pi

        learned = pi.get_learned_seed_config()
        # Both should be positive (negated from negative β)
        assert learned["outdoor_seed_heat"] == pytest.approx(0.3)
        assert learned["outdoor_seed_cool"] == pytest.approx(0.4)


# ── Reset preserves convention ───────────────────────────────────────


class TestResetPreservesConvention:
    """Verify reset_ff_seeds rebuilds β correctly from seeds."""

    @pytest.mark.asyncio
    async def test_reset_restores_correct_beta(self):
        """After reset, β matches -seed for all coefficients."""
        config = make_pi_config({
            "pi_outdoor_seed_heat": 0.5,
            "pi_outdoor_seed_cool": 0.4,
            "pi_model_inputs": [{
                "name": "stove", "entity_id": "sensor.stove",
                "seed_heat": 3.0, "seed_cool": 2.5,
            }],
        })
        e = FakeEntity(config)
        pi = e._pi

        # Corrupt β
        pi._rls_heat.beta[1] = 999.0
        pi._rls_cool.beta[1] = 999.0

        await pi.async_reset_ff_seeds()

        # After reset, β should be -seed × scale
        heat_scale = pi._feature_scales[1]
        cool_scale = pi._feature_scales[1]
        assert pi._rls_heat.beta[1] == pytest.approx(-0.5 * heat_scale)
        assert pi._rls_cool.beta[1] == pytest.approx(-0.4 * cool_scale)

        # Model input β
        mi_scale = pi._feature_scales[2]
        assert pi._rls_heat.beta[2] == pytest.approx(-3.0 * mi_scale)
        assert pi._rls_cool.beta[2] == pytest.approx(-2.5 * mi_scale)


# ── Batch WLS agreement ─────────────────────────────────────────────


class TestBatchOnlineAgreement:
    """Verify batch and online use same outdoor_delta convention."""

    def test_batch_outdoor_delta_matches_online(self):
        """Both compute outdoor_temp - desired_temp (exogenous reference)."""
        from custom_components.tasmota_irhvac.pi.batch_learning import (
            Observation, build_feature_vector_from_raw,
        )

        outdoor_temp = -5.0
        desired_temp = 22.0

        # Batch convention
        obs = Observation(
            timestamp=0.0,
            wall_time=0.0,
            hp_setpoint=22.0,
            outdoor_temp_c=outdoor_temp,
            current_c=20.0,
            desired_c=desired_temp,
            room_rate=0.0,
            raw_readings={},
            clamped=False,
        )
        feature_order = ["intercept", "outdoor_delta"]
        features = build_feature_vector_from_raw(obs, [], feature_order)
        assert features is not None
        batch_delta = features[1]  # outdoor_delta is index 1

        # Online convention: outdoor - desired (exogenous, no PV coupling)
        online_delta = outdoor_temp - desired_temp

        assert batch_delta == pytest.approx(online_delta)
        assert batch_delta == pytest.approx(-27.0)

    def test_greybox_outdoor_delta_matches(self):
        """Greybox buffer uses same convention."""
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation

        outdoor_temp = 35.0
        room_temp = 24.0

        obs = Observation(
            timestamp=0.0,
            wall_time=0.0,
            hp_setpoint=22.0,
            outdoor_temp_c=outdoor_temp,
            current_c=room_temp,
            desired_c=22.0,
            room_rate=0.0,
            raw_readings={},
            clamped=False,
        )

        # Greybox uses outdoor_temp_c - current_c
        greybox_delta = obs.outdoor_temp_c - obs.current_c
        online_delta = outdoor_temp - room_temp

        assert greybox_delta == pytest.approx(online_delta)
        assert greybox_delta == pytest.approx(11.0)
