"""Tests for staged learning architecture (PR 1).

Covers:
- Passive observation during OFF mode
- Per-feature confidence gating (VIF, unlock conditions)
- κ-gated learning rate
- Learning state sensor
- Frozen features as held in batch WLS
- Reset semantics (buffer reset vs seed reset)
- Migration from pre40
- 5 scenario simulations (plan validation tests)
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.climate import HVACMode
from homeassistant.const import STATE_ON, UnitOfTemperature

from custom_components.tasmota_irhvac.pi.batch_learning import (
    BatchResult,
    DiversityAwareBuffer,
    Observation,
    compare_and_report,
    compute_blended_update,
    weighted_least_squares,
)
from custom_components.tasmota_irhvac.pi.pi_controller import PIController
from custom_components.tasmota_irhvac.pi.rls_model import RLSModel

from .conftest import make_pi_config


# ── Test Helpers ─────────────────────────────────────────────────────────

_TEST_ENTITIES = [f"sensor.test_input_{i}" for i in range(10)]


def _test_model_inputs(n_extra: int, roles: list[str] | None = None) -> list[dict]:
    """Generate model input config with optional input_role."""
    inputs = []
    for i in range(n_extra):
        d: dict = {"entity_id": _TEST_ENTITIES[i], "name": f"input_{i}"}
        if roles and i < len(roles):
            d["input_role"] = roles[i]
        inputs.append(d)
    return inputs


def _test_feature_order(n_extra: int) -> list[str]:
    order = ["intercept", "outdoor_delta"]
    for i in range(n_extra):
        order.append(f"input_{i}")
    order.extend(["sin_hour", "cos_hour"])
    return order


# Hourly-spaced wall_time base — gives realistic ToD coverage so the
# sin_hour / cos_hour columns sweep the unit circle, matching production
# where buffer obs span hours/days rather than identical timestamps.
import datetime as _dt
_TEST_WALL_TIME_BASE = _dt.datetime(2026, 1, 15, 0, 0, 0).timestamp()


def _spread_wall_time(i: int, step_seconds: float = 3600.0) -> float:
    """Wall time for the i-th synthetic observation."""
    return _TEST_WALL_TIME_BASE + i * step_seconds


def _make_obs(
    features: list[float],
    sp: float | None,
    cur: float,
    des: float = 20.0,
    rate: float = 0.005,
    clamped: bool = False,
    clamped_reason: str = "",
    wall_time: float | None = None,
) -> Observation:
    """Create a test observation with optional None hp_setpoint."""
    if clamped and not clamped_reason:
        clamped_reason = "no_output"

    outdoor_temp_c: float | None = None
    if len(features) > 1:
        outdoor_temp_c = cur + features[1]

    raw_readings: dict[str, float] = {}
    for i, val in enumerate(features[2:]):
        raw_readings[_TEST_ENTITIES[i]] = val

    return Observation(
        timestamp=0.0,
        wall_time=wall_time if wall_time is not None else time.time(),
        hp_setpoint=sp,
        current_c=cur,
        desired_c=des,
        outdoor_temp_c=outdoor_temp_c,
        room_rate=rate,
        raw_readings=raw_readings,
        clamped=clamped,
        clamped_reason=clamped_reason,
    )


class FakePIEntity:
    """Minimal fake entity for testing PI controller."""

    _attr_hvac_modes = [HVACMode.HEAT, HVACMode.COOL, HVACMode.OFF]
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _temp_precision = 1.0

    def __init__(self, config):
        self.hass = MagicMock()
        self.hass.states.get.return_value = None
        self._attr_hvac_mode = HVACMode.HEAT
        self._attr_current_temperature = 21.0
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
        self.async_get_last_extra_data = AsyncMock(return_value=None)
        self._pi = PIController(self, config)

    @property
    def temperature_unit(self):
        return UnitOfTemperature.CELSIUS

    @property
    def device_info(self):
        return {"identifiers": {("test", "test")}}


def _make_config_with_inputs(n_inputs=0, roles=None, **overrides):
    """Make PI config with model inputs."""
    inputs = _test_model_inputs(n_inputs, roles=roles)
    config = {
        "pi_model_inputs": inputs,
    }
    config.update(overrides)
    return make_pi_config(config)


# ── Part 1: Passive Observation ──────────────────────────────────────────


class TestPassiveObservation:
    """Tests for _passive_tick() during hvac_mode=OFF."""

    @pytest.fixture
    def pi_entity(self):
        entity = FakePIEntity(make_pi_config())
        return entity

    async def test_off_mode_returns_false(self, pi_entity):
        """OFF mode runs passive tick, returns False (no IR)."""
        pi_entity._attr_hvac_mode = HVACMode.OFF
        pi_entity._pi._desired_temp = 22.0
        pi_entity._pi._hp_setpoint = 22.0
        result = await pi_entity._pi._pi_tick()
        assert result is False

    async def test_off_mode_zeros_integral(self, pi_entity):
        """OFF mode zeros the integral."""
        pi_entity._pi._pi_integral = 5.0
        pi_entity._attr_hvac_mode = HVACMode.OFF
        await pi_entity._pi._pi_tick()
        assert pi_entity._pi._pi_integral == 0.0

    async def test_off_mode_uninitializes_smith(self, pi_entity):
        """OFF mode uninitializes the Smith predictor."""
        if pi_entity._pi._smith is not None:
            pi_entity._pi._smith._initialized = True
            pi_entity._attr_hvac_mode = HVACMode.OFF
            await pi_entity._pi._pi_tick()
            assert not pi_entity._pi._smith._initialized

    async def test_off_mode_does_not_buffer(self, pi_entity):
        """OFF mode does NOT add observations (blocked on buffer redesign)."""
        pi_entity._attr_hvac_mode = HVACMode.OFF
        pi_entity._attr_current_temperature = 21.0
        pi_entity._pi._desired_temp = 22.0
        before = len(pi_entity._pi._observation_buffer_heat)
        with patch("custom_components.tasmota_irhvac.pi.pi_controller.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0
            mock_time.time.return_value = 1000.0
            await pi_entity._pi._pi_tick()
        after = len(pi_entity._pi._observation_buffer_heat)
        assert after == before

    async def test_passive_obs_no_sensor_returns_false(self, pi_entity):
        """OFF mode with no temp sensor returns False without error."""
        pi_entity._attr_hvac_mode = HVACMode.OFF
        pi_entity._attr_current_temperature = None
        result = await pi_entity._pi._pi_tick()
        assert result is False

    async def test_passive_tick_tracks_rate(self, pi_entity):
        """Passive tick updates room temp rate tracking."""
        pi_entity._attr_hvac_mode = HVACMode.OFF
        pi_entity._attr_current_temperature = 21.0
        with patch("custom_components.tasmota_irhvac.pi.pi_controller.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0
            mock_time.time.return_value = 1000.0
            await pi_entity._pi._pi_tick()
        assert len(pi_entity._pi._room_temp_history) > 0


class TestObservationHpSetpointNone:
    """Tests for Observation with hp_setpoint=None."""

    def test_serialize_deserialize_none_setpoint(self):
        """Observation with None setpoint round-trips through dict."""
        obs = _make_obs([1.0, 5.0], sp=None, cur=21.0,
                        clamped=True, clamped_reason="no_output")
        d = obs.as_dict()
        assert d["sp"] is None
        restored = Observation.from_dict(d)
        assert restored.hp_setpoint is None

    def test_buffer_filter_inactive_skips_none_setpoint(self):
        """filter_inactive doesn't crash on None hp_setpoint observations."""
        buf = DiversityAwareBuffer(n_features=2, max_size=100)
        buf.add(_make_obs([1.0, 5.0], sp=None, cur=21.0,
                          clamped=True, clamped_reason="no_output"))
        buf.add(_make_obs([1.0, 5.0], sp=25.0, cur=21.0))
        # Should not crash
        removed = buf.filter_inactive("heat")
        # The passive obs (sp=None) is kept — it's not "sp < cur"
        assert len(buf) >= 1

    def test_wls_excludes_passive_observations(self):
        """WLS filters out passive observations via clamped_reason."""
        n = 3
        obs_list = []
        for i in range(30):
            obs_list.append(_make_obs(
                [1.0, float(i % 10), 0.5], sp=22.0, cur=20.0,
            ))
        # Add some passive observations
        for i in range(10):
            obs_list.append(_make_obs(
                [1.0, 5.0, 0.5], sp=None, cur=21.0,
                clamped=True, clamped_reason="no_output",
            ))
        result = weighted_least_squares(
            obs_list, n_features=n, current_beta=[0.0] * n,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        # Should succeed — passive observations filtered out
        assert result is not None


# ── Part 2: Per-Feature Confidence Gating ────────────────────────────────


class TestVIFComputation:
    """Tests for DiversityAwareBuffer.compute_vif()."""

    def _make_obs(self, features, wall_time=None):
        return _make_obs(features, sp=22.0, cur=20.0, wall_time=wall_time)

    def test_vif_inf_before_recompute(self):
        """Returns all inf when xtx matrix hasn't been computed."""
        buf = DiversityAwareBuffer(n_features=3, max_size=100)
        vif = buf.compute_vif()
        assert all(math.isinf(v) for v in vif)

    def test_vif_well_conditioned(self):
        """Independent features give VIF close to 1.

        Uses the production-shape feature set including ToD; the
        sin_hour / cos_hour columns should also sit at VIF ≲ 5 because
        wall_time spreads across multiple diurnal cycles.
        """
        feature_order = _test_feature_order(1)
        buf = DiversityAwareBuffer(
            n_features=len(feature_order), max_size=200,
            feature_order=feature_order,
            model_inputs=_test_model_inputs(1),
        )
        random.seed(42)
        for i in range(100):
            x1 = float(i % 10)
            x2 = random.uniform(-1, 1)  # Independent of x1
            buf.add(self._make_obs(
                [1.0, x1, x2], wall_time=_spread_wall_time(i),
            ))
        buf.recompute_info_matrix()
        vif = buf.compute_vif()
        assert all(v < 5.0 for v in vif), f"VIF too high: {vif}"

    def test_vif_collinear_features(self):
        """Collinear features produce VIF > 10."""
        feature_order = _test_feature_order(1)
        buf = DiversityAwareBuffer(
            n_features=len(feature_order), max_size=200,
            feature_order=feature_order,
            model_inputs=_test_model_inputs(1),
        )
        for i in range(100):
            x1 = float(i % 10)
            x2 = x1 * 0.95 + 0.01 * (i % 3)  # Highly correlated
            buf.add(self._make_obs(
                [1.0, x1, x2], wall_time=_spread_wall_time(i),
            ))
        buf.recompute_info_matrix()
        vif = buf.compute_vif()
        # At least one of the constructed-collinear features should
        # produce VIF > 10 (outdoor_delta or input_0).
        assert any(v > 10.0 for v in vif[1:3]), (
            f"Expected collinear VIF > 10 for outdoor_delta or input_0, got {vif}"
        )


class TestFrozenFeaturesInWLS:
    """Tests that frozen features are treated as held in batch WLS."""

    def test_frozen_features_held_in_wls(self):
        """Frozen features are excluded from regression (treated as held)."""
        n = 4  # intercept, outdoor_delta, input_0, input_1
        obs_list = []
        for i in range(40):
            od = float(i % 10)
            x1 = float(i % 5)
            x2 = float(i % 3)
            obs_list.append(_make_obs([1.0, od, x1, x2], sp=22.0 + od * 0.3, cur=20.0))

        current = [0.0, 0.3, 0.5, -0.5]

        # Without frozen: both inputs active
        result_no_freeze = weighted_least_squares(
            obs_list, n_features=n, current_beta=current,
            feature_order=_test_feature_order(2),
            model_inputs=_test_model_inputs(2),
        )

        # With input_1 frozen: should be held at current value
        result_frozen = weighted_least_squares(
            obs_list, n_features=n, current_beta=current,
            feature_order=_test_feature_order(2),
            model_inputs=_test_model_inputs(2),
            frozen_features={3},  # index 3 = input_1
        )

        assert result_frozen is not None
        assert 3 in result_frozen.held_features
        # Frozen feature should be held at current beta value
        assert result_frozen.beta_batch[3] == pytest.approx(current[3], abs=0.01)

    def test_frozen_base_features_not_held(self):
        """Intercept and outdoor_delta (index 0, 1) are not affected by frozen_features.

        The base regression always includes them; frozen_features only
        affects model inputs (index 2+).
        """
        n = 3
        obs_list = []
        for i in range(40):
            od = float(i % 10)
            x1 = float(i % 5)
            obs_list.append(_make_obs([1.0, od, x1], sp=22.0 + od * 0.3, cur=20.0))

        current = [0.0, 0.3, 0.5]
        # Freeze outdoor_delta (index 1) — has no effect since base regression
        # always includes it (frozen_features only gates model inputs at index 2+)
        result = weighted_least_squares(
            obs_list, n_features=n, current_beta=current,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
            frozen_features={1},
        )
        assert result is not None
        # outdoor_delta (index 1) should NOT be in held
        assert 1 not in result.held_features


class TestColdStartFreezing:
    """Tests that model input features start frozen and base features don't."""

    def test_model_inputs_start_frozen(self):
        """Model input features (index 2+) start frozen at init."""
        config = _make_config_with_inputs(n_inputs=2)
        entity = FakePIEntity(config)
        pi = entity._pi
        n = pi._rls_heat.n
        # Intercept and outdoor_delta are not frozen
        assert not pi._rls_heat.frozen[0]
        assert not pi._rls_heat.frozen[1]
        # Model inputs are frozen
        for i in range(2, n):
            assert pi._rls_heat.frozen[i], f"Feature {i} should be frozen at init"
            assert pi._rls_cool.frozen[i], f"Cool feature {i} should be frozen at init"

    def test_no_model_inputs_tod_frozen(self):
        """With no model inputs, only ToD features are frozen at cold start."""
        entity = FakePIEntity(make_pi_config())
        pi = entity._pi
        # Intercept and outdoor_delta unfrozen, ToD features (sin/cos) frozen
        assert not pi._rls_heat.frozen[0]  # intercept
        assert not pi._rls_heat.frozen[1]  # outdoor_delta
        assert pi._rls_heat.frozen[2]  # sin_hour
        assert pi._rls_heat.frozen[3]  # cos_hour


class TestResetSemantics:
    """Tests for buffer reset vs seed reset freeze behavior."""

    def test_buffer_flush_does_not_refreeze(self):
        """Flushing observation buffer does NOT re-freeze features."""
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        pi = entity._pi
        # Manually unfreeze a feature
        pi._rls_heat.frozen[2] = False
        # Flush buffer
        pi.flush_observation_buffer()
        # Feature should still be unfrozen
        assert not pi._rls_heat.frozen[2]

    async def test_seed_reset_refreezes(self):
        """Resetting to seeds re-freezes model input features."""
        config = _make_config_with_inputs(n_inputs=2)
        entity = FakePIEntity(config)
        pi = entity._pi
        # Unfreeze features
        pi._rls_heat.frozen[2] = False
        pi._rls_heat.frozen[3] = False
        assert not pi._rls_heat.frozen[2]
        # Reset seeds
        await pi.async_reset_ff_seeds(mode="heat")
        # Features should be re-frozen
        assert pi._rls_heat.frozen[2]
        assert pi._rls_heat.frozen[3]
        assert not pi._rls_heat.frozen[0]  # Intercept stays unfrozen
        assert not pi._rls_heat.frozen[1]  # Outdoor delta stays unfrozen

    async def test_seed_reset_clears_manual_overrides(self):
        """Seed reset clears all manual overrides."""
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        pi = entity._pi
        pi.set_frozen("heat", 2, False, manual=True)
        assert pi._manual_override_heat[2] is True  # True = force unfrozen
        await pi.async_reset_ff_seeds(mode="heat")
        assert pi._manual_override_heat[2] is None


class TestManualFreezeUnfreeze:
    """Tests for manual freeze/unfreeze and auto-gating interaction."""

    def test_manual_unfreeze_sets_override_true(self):
        """Manual unfreeze sets override to True (force unfrozen)."""
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        pi = entity._pi
        pi.set_frozen("heat", 2, False, manual=True)
        assert pi._manual_override_heat[2] is True

    def test_manual_freeze_sets_override_false(self):
        """Manual freeze sets override to False (force frozen)."""
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        pi = entity._pi
        pi._rls_heat.frozen[2] = False  # Must be unfrozen first
        pi.set_frozen("heat", 2, True, manual=True)
        assert pi._manual_override_heat[2] is False

    def test_auto_unfreeze_leaves_override_none(self):
        """Auto-unfreeze (manual=False) does not set override."""
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        pi = entity._pi
        pi.set_frozen("heat", 2, False, manual=False)
        assert pi._manual_override_heat[2] is None


class TestMigrationFromPre40:
    """Tests for backward compatibility with pre40 models."""

    def test_restored_model_without_frozen_key_is_unfrozen(self):
        """Pre40 RLS models have no frozen key — all features default unfrozen."""
        n_inputs = 2
        model = RLSModel(n_inputs=n_inputs)
        # Simulate pre40 serialization (no frozen key)
        d = model.as_dict()
        d.pop("frozen", None)
        restored = RLSModel.from_dict(d, n_inputs)
        assert not any(restored.frozen)

    def test_restored_mature_model_stays_unfrozen(self):
        """A pre40 model with observation_count > 0 stays unfrozen after restore.

        The PIController.restore_extra_stored_data calls RLSModel.from_dict
        which defaults frozen=False for all features when no frozen key is
        present.  This is the correct backward-compat behavior.
        """
        model = RLSModel(n_inputs=2)
        model.observation_count = 50
        d = model.as_dict()
        d.pop("frozen", None)  # Pre40: no frozen state stored
        restored = RLSModel.from_dict(d, 2)
        assert restored.observation_count == 50
        assert not any(restored.frozen)


# ── Part 2: Feature Unlock Evaluation ────────────────────────────────────


class TestEvaluateFeatureUnlocks:
    """Tests for _evaluate_feature_unlocks() logic."""

    def _make_entity_with_inputs(self, n=2, roles=None):
        config = _make_config_with_inputs(n_inputs=n, roles=roles)
        entity = FakePIEntity(config)
        return entity

    def _make_full_result(self, n, std_err=None, held=None, vif=None):
        """Create a full-model BatchResult for unlock evaluation.

        This represents the result from WLS with ALL features estimated
        (no frozen_features).  std_err, held_features, and feature_vif
        are the unlock evidence.
        """
        return BatchResult(
            n_total=100,
            n_eligible=80,
            beta_batch=[0.0] * n,
            beta_current=[0.0] * n,
            residual_rms=0.5,
            max_coeff_change_pct=5.0,
            recommend_update=True,
            held_features=held or set(),
            beta_std_err=std_err or [0.1] * n,
            beta_blended=[0.0] * n,
            blend_gains=[0.5] * n,
            plant_snapshot={},
            feature_vif=vif or [1.0] * n,
        )

    def test_unlock_with_good_conditions(self):
        """Feature unlocks when full-model shows it's identifiable."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n
        assert rls.frozen[2]

        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert not rls.frozen[2], "Feature should have been unlocked"

    def test_no_unlock_when_held_in_full_model(self):
        """Feature stays frozen when held in full-model (insufficient variance)."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n

        result = self._make_full_result(n, held={2}, vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert rls.frozen[2], "Feature should remain frozen (held in full model)"

    def test_no_unlock_when_infinite_stderr(self):
        """Feature stays frozen when full-model std_err is infinite."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n

        result = self._make_full_result(n, std_err=[0.1, 0.05, float("inf")], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert rls.frozen[2], "Feature should remain frozen (infinite std_err)"

    def test_no_unlock_when_high_vif(self):
        """Feature stays frozen when VIF >= 10."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n

        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 15.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert rls.frozen[2], "Feature should remain frozen (VIF >= 10)"

    def test_adjacent_zone_gated_by_kappa(self):
        """Adjacent zone feature stays frozen when κ >= 100."""
        entity = self._make_entity_with_inputs(n=1, roles=["adjacent_zone"])
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n

        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 150.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert rls.frozen[2], "Adjacent zone should remain frozen (κ >= 100)"

    def test_adjacent_zone_unlocks_when_kappa_low(self):
        """Adjacent zone feature unlocks when κ < 100."""
        entity = self._make_entity_with_inputs(n=1, roles=["adjacent_zone"])
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n

        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 25.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert not rls.frozen[2], "Adjacent zone should unlock (κ < 100)"

    def test_manual_override_skipped(self):
        """Auto-gating skips features with any manual override."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n

        pi._manual_override_heat[2] = False  # Force frozen
        rls.frozen[2] = True

        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert rls.frozen[2], "Should remain frozen (manual override skips evaluation)"

    def test_already_unfrozen_not_reevaluated(self):
        """Features that are already unfrozen are skipped."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n
        rls.frozen[2] = False

        result = self._make_full_result(n, std_err=[0.1, 0.05, float("inf")], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        assert not rls.frozen[2], "Already unfrozen should stay unfrozen"

    def test_records_capture_all_passed_unfrozen(self):
        """Successful unlock records gate_failed=None, unfrozen=True with full-model values."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n
        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)

        records = pi._last_unlock_evaluation
        # 1 model input + 2 ToD = 3 frozen at start; sin/cos lack entity_id
        # so they end up `held` in the full_result (unset in this stub).
        # Locate the model_input record (coefficient_index=2).
        rec = next(r for r in records if r.coefficient_index == 2)
        assert rec.gate_failed is None
        assert rec.unfrozen is True
        assert rec.full_model_std_err == 0.08
        assert rec.full_model_vif == 2.0
        assert rec.is_adjacent_zone is False

    def test_records_capture_held_gate_failure(self):
        """Held-feature failure surfaces gate_failed='held' with in_full_model_held=True."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n
        result = self._make_full_result(n, held={2}, vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        rec = next(r for r in pi._last_unlock_evaluation if r.coefficient_index == 2)
        assert rec.gate_failed == "held"
        assert rec.unfrozen is False
        assert rec.in_full_model_held is True

    def test_records_capture_vif_gate_failure(self):
        """VIF≥10 surfaces gate_failed='vif' with the offending VIF preserved."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n
        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 14.3])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        rec = next(r for r in pi._last_unlock_evaluation if r.coefficient_index == 2)
        assert rec.gate_failed == "vif"
        assert rec.unfrozen is False
        assert rec.full_model_vif == 14.3
        assert rec.full_model_std_err == 0.08  # std_err passed; only VIF failed

    def test_records_capture_std_err_gate_failure(self):
        """Infinite std_err surfaces gate_failed='std_err' with std_err=None."""
        entity = self._make_entity_with_inputs(n=1)
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n
        result = self._make_full_result(n, std_err=[0.1, 0.05, float("inf")], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 15.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        rec = next(r for r in pi._last_unlock_evaluation if r.coefficient_index == 2)
        assert rec.gate_failed == "std_err"
        assert rec.unfrozen is False
        assert rec.full_model_std_err is None

    def test_records_capture_kappa_gate_failure(self):
        """Adjacent_zone with κ≥100 surfaces gate_failed='kappa'."""
        entity = self._make_entity_with_inputs(n=1, roles=["adjacent_zone"])
        pi = entity._pi
        rls = pi._rls_heat
        n = rls.n
        result = self._make_full_result(n, std_err=[0.1, 0.05, 0.08], vif=[1.0, 1.5, 2.0])
        pi._cached_kappa = 150.0

        pi._evaluate_feature_unlocks(result, rls, is_heating=True)
        rec = next(r for r in pi._last_unlock_evaluation if r.coefficient_index == 2)
        assert rec.gate_failed == "kappa"
        assert rec.unfrozen is False
        assert rec.is_adjacent_zone is True
        assert rec.kappa_at_decision == 150.0


# ── Part 3: Learning State Sensor ────────────────────────────────────────


class TestLearningState:
    """Tests for get_learning_state() method."""

    def test_learning_state_all_frozen(self):
        """All features frozen → 'Learning'."""
        entity = self._make_entity(n_inputs=2)
        state = entity._pi.get_learning_state()
        assert state["state"] == "Learning"
        # 2 model inputs + 2 ToD = 4 frozen features
        assert len(state["frozen_features"]) == 4
        assert len(state["active_features"]) == 0

    def test_learning_state_all_unfrozen(self):
        """All features unfrozen → 'Optimized'."""
        entity = self._make_entity(n_inputs=2)
        pi = entity._pi
        for i in range(2, pi._rls_heat.n):
            pi._rls_heat.frozen[i] = False
        state = pi.get_learning_state()
        assert state["state"] == "Optimized"
        assert len(state["frozen_features"]) == 0
        # 2 model inputs + 2 ToD = 4 active features
        assert len(state["active_features"]) == 4

    def test_learning_state_partial(self):
        """Some frozen, some unfrozen → 'Optimizing'."""
        entity = self._make_entity(n_inputs=2)
        pi = entity._pi
        pi._rls_heat.frozen[2] = False  # Unfreeze one model input
        state = pi.get_learning_state()
        assert state["state"] == "Optimizing"
        # 1 model input + 2 ToD still frozen = 3 frozen
        assert len(state["frozen_features"]) == 3
        assert len(state["active_features"]) == 1

    def test_learning_state_no_inputs(self):
        """No user model inputs → ToD features still exist and start frozen."""
        entity = FakePIEntity(make_pi_config())
        state = entity._pi.get_learning_state()
        # ToD features are frozen at cold start → "Learning"
        assert state["state"] == "Learning"
        assert len(state["frozen_features"]) == 2  # sin_hour, cos_hour

    def test_learning_state_attributes(self):
        """State includes expected attributes."""
        entity = self._make_entity(n_inputs=1)
        state = entity._pi.get_learning_state()
        assert "frozen_features" in state
        assert "active_features" in state
        assert "observation_count" in state
        assert "ff_confidence" in state
        assert "condition_number" in state

    def test_learning_state_kappa(self):
        """State reports cached κ."""
        entity = self._make_entity(n_inputs=1)
        entity._pi._cached_kappa = 42.5
        state = entity._pi.get_learning_state()
        assert state["condition_number"] == pytest.approx(42.5, abs=0.1)

    def test_learning_state_kappa_none(self):
        """State reports None when κ not computed."""
        entity = self._make_entity(n_inputs=1)
        entity._pi._cached_kappa = None
        state = entity._pi.get_learning_state()
        assert state["condition_number"] is None

    def _make_entity(self, n_inputs):
        config = _make_config_with_inputs(n_inputs=n_inputs)
        return FakePIEntity(config)


class TestCoeffRole:
    """Tests for _coeff_role() helper."""

    def test_intercept(self):
        entity = FakePIEntity(make_pi_config())
        assert entity._pi._coeff_role(0) == "intercept"

    def test_outdoor_delta(self):
        entity = FakePIEntity(make_pi_config())
        assert entity._pi._coeff_role(1) == "outdoor_delta"

    def test_model_input_with_role(self):
        config = _make_config_with_inputs(n_inputs=1, roles=["adjacent_zone"])
        entity = FakePIEntity(config)
        assert entity._pi._coeff_role(2) == "adjacent_zone"

    def test_model_input_default_role(self):
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        assert entity._pi._coeff_role(2) == "other"

    def test_out_of_range(self):
        entity = FakePIEntity(make_pi_config())
        assert entity._pi._coeff_role(99) == "other"


# ── Part 2 Validation: Scenario Simulations ──────────────────────────────

# These simulate realistic data conditions to verify per-feature unlock
# behavior across different seasons/configurations.


def _generate_batch_observations(
    n_obs: int,
    n_inputs: int,
    *,
    outdoor_range: tuple[float, float] = (0.0, 10.0),
    input_generators: list | None = None,
    noise_sigma: float = 0.1,
    true_beta: list[float] | None = None,
    seed: int = 42,
) -> list[Observation]:
    """Generate synthetic observations for batch learning scenarios.

    Args:
        n_obs: Number of observations.
        n_inputs: Number of model inputs (beyond intercept + outdoor_delta).
        outdoor_range: (min, max) for outdoor delta values.
        input_generators: List of callables(i, random_state) → value for each input.
            If None, each input is random uniform [-1, 1].
        noise_sigma: Observation noise standard deviation.
        true_beta: True coefficients [intercept, outdoor_delta, input_0, ...].
            Used to compute realistic hp_setpoint values.
        seed: Random seed.
    """
    rng = random.Random(seed)
    obs_list = []

    if true_beta is None:
        true_beta = [0.5, 0.3] + [0.0] * n_inputs

    for i in range(n_obs):
        od = outdoor_range[0] + (outdoor_range[1] - outdoor_range[0]) * rng.random()
        cur = 20.0 + rng.gauss(0, 0.5)

        features = [1.0, od]
        if input_generators:
            for gen in input_generators:
                features.append(gen(i, rng))
        else:
            for _ in range(n_inputs):
                features.append(rng.uniform(-1, 1))

        # Compute realistic setpoint from true model
        true_offset = sum(b * f for b, f in zip(true_beta, features))
        sp = cur + true_offset + rng.gauss(0, noise_sigma)

        obs_list.append(_make_obs(
            features, sp=sp, cur=cur, rate=rng.gauss(0, 0.005),
            wall_time=_spread_wall_time(i),
        ))

    return obs_list


class TestScenarioWinterWellSeparated:
    """Scenario 1: Winter with well-separated features.

    All features should unlock within ~2 batch cycles because:
    - Outdoor delta has wide range (cold temps)
    - Model inputs have sufficient variance
    - Features are independent (no collinearity)
    """

    def test_all_features_unlock(self):
        n_inputs = 2
        feature_order = _test_feature_order(n_inputs)
        n = len(feature_order)  # intercept + outdoor + 2 inputs + sin_hour + cos_hour
        # ToD coefficients are 0 in the synthetic truth (no real ToD signal).
        true_beta = [0.5, 0.35, -3.5, -2.0, 0.0, 0.0]  # solar-like, stove-like

        # Generate winter data: wide outdoor range, independent inputs
        obs = _generate_batch_observations(
            n_obs=80, n_inputs=n_inputs,
            outdoor_range=(5.0, 15.0),  # Cold, wide spread
            input_generators=[
                lambda i, rng: max(0, math.sin(i * 0.2)) * rng.uniform(0.5, 1.0),  # "solar"
                lambda i, rng: 1.0 if i % 10 < 3 else 0.0,  # "stove" intermittent
            ],
            true_beta=true_beta,
            noise_sigma=0.1,
        )

        current_beta = [0.0] * n
        result = weighted_least_squares(
            obs, n_features=n, current_beta=current_beta,
            feature_order=feature_order,
            model_inputs=_test_model_inputs(n_inputs),
        )
        assert result is not None
        compare_and_report(result, current_beta, feature_order)
        compute_blended_update(result, prior_std=1.0, max_step=1.0)

        # Build a real buffer to check VIF
        buf = DiversityAwareBuffer(
            n_features=n, max_size=200,
            feature_order=feature_order,
            model_inputs=_test_model_inputs(n_inputs),
        )
        for o in obs:
            buf.add(o)
        buf.recompute_info_matrix()

        vif = buf.compute_vif()
        kappa = buf.compute_condition_number()

        # Model input features (indices 2, 3) should be identifiable (VIF < 10).
        # Intercept VIF and ToD VIFs are not gated by the controller
        # (sin_hour/cos_hour are unit-circle features whose VIF is dominated
        # by their geometric coupling, not the modeled signal).
        for i in (2, 3):
            assert vif[i] < 10.0, f"Feature {feature_order[i]} VIF={vif[i]} (should be < 10)"
        assert kappa < 100.0, f"κ={kappa} should be < 100 for well-separated data"

        # Modeled features (intercept + outdoor + inputs) should have
        # blend confidence and finite std_err.  ToD features (sin_hour,
        # cos_hour) are partialled out via FWL augmented partialling and
        # not fitted as standalone coefficients (batch_learning.py
        # _ragged_partial_regression docstring) — so their blend_gain is
        # 0 by design.
        assert len(result.blend_gains) == n, (
            f"blend_gains length {len(result.blend_gains)} != n_features {n}"
        )
        for i in range(2 + n_inputs):
            assert result.blend_gains[i] > 0, (
                f"Feature {feature_order[i]} has no blend confidence"
            )
            assert math.isfinite(result.beta_std_err[i]), (
                f"Feature {feature_order[i]} std_err not finite"
            )


class TestScenarioShoulderCollinear:
    """Scenario 2: Shoulder season with collinear features.

    Solar should unlock slower; adjacent zones gated by κ.
    In shoulder season, outdoor temp doesn't vary much and solar
    correlates with outdoor temp changes.
    """

    def test_solar_collinear_high_vif(self):
        n_inputs = 2
        feature_order = _test_feature_order(n_inputs)
        n = len(feature_order)

        # Shoulder season: narrow outdoor range, solar correlated with outdoor
        obs = _generate_batch_observations(
            n_obs=80, n_inputs=n_inputs,
            outdoor_range=(4.0, 6.0),  # Narrow range
            input_generators=[
                # "solar" — correlated with outdoor delta position
                lambda i, rng: max(0, 0.8 * (i % 10) / 10 + rng.gauss(0, 0.05)),
                # "adjacent_zone" — correlated with outdoor
                lambda i, rng: 0.9 * (i % 10) / 10 + rng.gauss(0, 0.02),
            ],
            true_beta=[0.5, 0.3, -2.0, -1.0, 0.0, 0.0],  # ToD coeffs = 0
            noise_sigma=0.1,
        )

        buf = DiversityAwareBuffer(
            n_features=n, max_size=200,
            feature_order=feature_order,
            model_inputs=_test_model_inputs(n_inputs),
        )
        for o in obs:
            buf.add(o)
        buf.recompute_info_matrix()

        kappa = buf.compute_condition_number()
        vif = buf.compute_vif()

        # With narrow outdoor range + correlated features, expect elevated κ
        # and at least one of the *modeled* features (indices 2, 3) to show
        # elevated VIF.  ToD VIFs are excluded — they reflect sin/cos
        # geometric coupling, not the test's collinearity scenario.
        assert kappa > 10.0, f"κ={kappa} should be elevated for collinear data"
        max_modeled_vif = max(vif[2], vif[3])
        assert max_modeled_vif > 3.0, (
            f"Expected elevated VIF for collinear modeled inputs, got {vif}"
        )


class TestScenarioSolarSaturation:
    """Scenario 3: Solar saturation — HP idle, no active observations.

    Solar can't unlock because there are no active HP observations
    during solar gain periods (HP idles at minimum setpoint).
    """

    def test_solar_stays_frozen_no_active_data(self):
        """When all solar observations are clamped, solar stays frozen."""
        n_inputs = 1  # Just solar
        n = 2 + n_inputs

        # Only non-solar observations are unclamped
        obs = []
        for i in range(40):
            solar = 0.0  # Night time — no solar
            obs.append(_make_obs(
                [1.0, float(i % 10), solar],
                sp=22.0 + float(i % 10) * 0.3, cur=20.0,
            ))
        # Solar observations are clamped (HP idling)
        for i in range(20):
            solar = 0.8 + 0.2 * (i % 5) / 5
            obs.append(_make_obs(
                [1.0, 2.0, solar],
                sp=None, cur=23.0,
                clamped=True, clamped_reason="no_output",
            ))

        current_beta = [0.0, 0.3, 0.0]
        result = weighted_least_squares(
            obs, n_features=n, current_beta=current_beta,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        compare_and_report(result, current_beta, ["intercept", "outdoor_delta", "solar"])
        compute_blended_update(result, prior_std=1.0, max_step=1.0)

        # Solar feature should have been held (no active solar observations
        # passed through the WLS filter since all solar obs are clamped)
        assert 2 in result.held_features, (
            f"Solar should be held (only clamped solar obs). "
            f"held={result.held_features}"
        )


class TestScenarioProductionReplay:
    """Scenario 4: Production-like data (r=0.95 between zones).

    Adjacent zones stay frozen due to high correlation between
    zones on the same condenser at similar setpoints.
    """

    def test_adjacent_zones_stay_frozen_high_correlation(self):
        """Adjacent zones on the same condenser at similar setpoints → high κ.

        Both zone deltas track outdoor temp closely (same building envelope),
        with only slight noise differences.  This makes them nearly
        collinear (r≈0.95), producing elevated κ.
        """
        n_inputs = 2  # Two adjacent zones
        feature_order = _test_feature_order(n_inputs)
        n = len(feature_order)

        # Generate correlated zone data manually (lambdas can't share state)
        rng = random.Random(42)
        obs = []
        for i in range(80):
            od = 3.0 + 9.0 * rng.random()
            cur = 20.0 + rng.gauss(0, 0.3)
            # Both zones track the same base signal with slight noise
            base_signal = od * 0.5 + rng.gauss(0, 0.05)
            zone1 = base_signal + rng.gauss(0, 0.1)  # r≈0.95 with zone2
            zone2 = base_signal + rng.gauss(0, 0.1)
            sp = cur + 0.5 + od * 0.35 + zone1 * (-0.5) + zone2 * (-0.5) + rng.gauss(0, 0.1)
            obs.append(_make_obs(
                [1.0, od, zone1, zone2], sp=sp, cur=cur,
                wall_time=_spread_wall_time(i),
            ))

        buf = DiversityAwareBuffer(
            n_features=n, max_size=200,
            feature_order=feature_order,
            model_inputs=_test_model_inputs(n_inputs, roles=["adjacent_zone", "adjacent_zone"]),
        )
        for o in obs:
            buf.add(o)
        buf.recompute_info_matrix()

        kappa = buf.compute_condition_number()
        vif = buf.compute_vif()

        # High correlation between zones should give elevated κ and VIF
        assert kappa > 15.0, f"κ={kappa} should be elevated for correlated zones"
        # At least one zone should have VIF > 3 (moderate collinearity)
        max_zone_vif = max(vif[2], vif[3])
        assert max_zone_vif > 3.0, f"Expected elevated VIF for correlated zones, got {vif}"


class TestScenarioIntermittentHeatSource:
    """Scenario 5: Intermittent heat source unlocks after batch sees active periods.

    A pellet stove that runs only some evenings.  The feature should unlock
    after enough active-period observations accumulate.
    """

    def test_stove_unlocks_with_sufficient_active_data(self):
        n_inputs = 1  # stove
        n = 2 + n_inputs

        # Mix of stove-on and stove-off observations
        obs = []
        for i in range(60):
            od = float(i % 10)
            stove = 1.0 if i % 5 == 0 else 0.0  # 20% stove-on rate
            sp = 22.0 + od * 0.3 + stove * (-3.0)
            obs.append(_make_obs([1.0, od, stove], sp=sp, cur=20.0))

        current_beta = [0.0, 0.3, 0.0]
        result = weighted_least_squares(
            obs, n_features=n, current_beta=current_beta,
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        compare_and_report(result, current_beta, ["intercept", "outdoor_delta", "stove"])
        compute_blended_update(result, prior_std=1.0, max_step=1.0)

        # Stove feature should have sufficient data to unlock
        assert 2 not in result.held_features, "Stove should not be held"
        assert result.blend_gains[2] > 0, "Stove should have blend confidence"
        assert math.isfinite(result.beta_std_err[2]), "Stove std_err should be finite"

    def test_stove_stays_held_with_no_active_data(self):
        """Stove stays held when it never fires."""
        n_inputs = 1
        n = 2 + n_inputs

        obs = []
        for i in range(40):
            od = float(i % 10)
            stove = 0.0  # Never fires
            sp = 22.0 + od * 0.3
            obs.append(_make_obs([1.0, od, stove], sp=sp, cur=20.0))

        result = weighted_least_squares(
            obs, n_features=n, current_beta=[0.0, 0.3, 0.0],
            feature_order=_test_feature_order(1),
            model_inputs=_test_model_inputs(1),
        )
        assert result is not None
        # Stove should be held (zero variance — never fires)
        assert 2 in result.held_features


# ── End-to-End: Feature Unlock Through Batch Pipeline ────────────────────


class TestFeatureUnlockEndToEnd:
    """Test the full batch → unlock pipeline on a PIController instance."""

    def test_feature_unlocks_after_batch(self):
        """Feature unlocks when batch provides good evidence."""
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        pi = entity._pi

        # Verify starts frozen
        assert pi._rls_heat.frozen[2]

        # Populate buffer with good data — vary wall_time across 24h
        # so ToD features have variance (otherwise design matrix is rank-deficient)
        base_time = 1736935200.0  # 2026-01-15 08:00
        for i in range(40):
            od = float(i % 10)
            x1 = float(i % 5)
            sp = 22.0 + od * 0.3 + x1 * 0.5
            pi._observation_buffer_heat.add(_make_obs(
                [1.0, od, x1], sp=sp, cur=20.0,
                wall_time=base_time + i * 2160.0,  # 36 min apart → spans 24h
            ))

        # Run batch analysis
        pi._entity._attr_hvac_mode = HVACMode.HEAT
        pi._run_batch_analysis()

        # Feature should now be unfrozen
        assert not pi._rls_heat.frozen[2], (
            "Feature should have been unlocked by batch analysis"
        )

    def test_feature_stays_frozen_no_data(self):
        """Feature stays frozen when buffer has no data for it."""
        config = _make_config_with_inputs(n_inputs=1)
        entity = FakePIEntity(config)
        pi = entity._pi

        # Populate buffer with data but input always 0
        base_time = 1736935200.0
        for i in range(40):
            od = float(i % 10)
            pi._observation_buffer_heat.add(_make_obs(
                [1.0, od, 0.0], sp=22.0 + od * 0.3, cur=20.0,
                wall_time=base_time + i * 2160.0,
            ))

        pi._entity._attr_hvac_mode = HVACMode.HEAT
        pi._run_batch_analysis()

        # Feature should stay frozen (zero variance → held)
        assert pi._rls_heat.frozen[2], (
            "Feature should remain frozen (zero variance in data)"
        )
