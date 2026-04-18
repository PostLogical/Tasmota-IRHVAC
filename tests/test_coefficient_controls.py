"""Tests for coefficient adjustment and freeze services (prompt #7)."""

import pytest

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.pi.pi_controller import RLSModel

from .conftest import get_climate_entity


# ── RLS frozen mask (unit tests) ─────────────────────────────────────


class TestRLSFrozenMask:
    """Tests for RLSModel frozen coefficient behavior."""

    def test_frozen_defaults_all_false(self):
        """All coefficients should be unfrozen by default."""
        model = RLSModel(n_inputs=2, seed_coefficients=[0.0, 0.3, -2.0])
        assert model.frozen == [False, False, False]

    def test_frozen_coefficient_unchanged_by_update(self):
        """A frozen coefficient should not change during update."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        model.frozen[1] = True  # Freeze outdoor_delta

        beta_before = model.beta[1]
        for _ in range(50):
            model.update([1.0, 10.0], 5.0)

        assert model.beta[1] == beta_before

    def test_unfrozen_coefficient_still_learns(self):
        """Unfrozen coefficients should still update when others are frozen."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        model.frozen[1] = True  # Freeze outdoor_delta

        beta0_before = model.beta[0]
        for _ in range(50):
            model.update([1.0, 10.0], 5.0)

        # Intercept should have moved (absorbing the error)
        assert model.beta[0] != beta0_before

    def test_freeze_all_prevents_all_learning(self):
        """Freezing all coefficients should prevent any changes."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.5, 0.3])
        model.frozen = [True, True]

        beta_before = list(model.beta)
        for _ in range(50):
            model.update([1.0, 10.0], 5.0)

        assert model.beta == beta_before

    def test_frozen_still_increments_observation_count(self):
        """Observation count should still increment even with frozen coefficients."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        model.frozen = [True, True]
        model.update([1.0, 10.0], 5.0)
        assert model.observation_count == 1

    def test_frozen_still_returns_residual(self):
        """Update should still return the correct residual when frozen."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        model.frozen = [True, True]
        residual = model.update([1.0, 10.0], 5.0)
        assert residual == pytest.approx(2.0)  # 5.0 - (0 + 0.3*10)

    def test_frozen_serialization_round_trip(self):
        """Frozen state should survive serialization."""
        model = RLSModel(n_inputs=2, seed_coefficients=[1.0, 0.3, -2.0])
        model.frozen[1] = True

        data = model.as_dict()
        assert "frozen" in data

        restored = RLSModel.from_dict(data, n_inputs=2)
        assert restored.frozen == [False, True, False]

    def test_frozen_not_serialized_when_all_false(self):
        """Frozen state should be omitted from serialization when all false."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        data = model.as_dict()
        assert "frozen" not in data

    def test_frozen_handles_dimension_mismatch(self):
        """from_dict with fewer frozen entries than model dimensions should pad with False."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        model.frozen[0] = True
        data = model.as_dict()

        # Restore with more inputs
        restored = RLSModel.from_dict(data, n_inputs=2)
        assert restored.frozen == [True, False, False]

    def test_unfreeze_resumes_learning(self):
        """Unfreezing a coefficient should allow it to learn again."""
        model = RLSModel(n_inputs=1, seed_coefficients=[0.0, 0.3])
        model.frozen[1] = True

        # Update while frozen — no change
        for _ in range(10):
            model.update([1.0, 10.0], 5.0)
        beta_frozen = model.beta[1]

        # Unfreeze and update
        model.frozen[1] = False
        for _ in range(50):
            model.update([1.0, 10.0], 5.0)

        assert model.beta[1] != beta_frozen

    def test_frozen_multivariate_selective(self):
        """Freezing one coefficient in a multivariate model should only affect that one."""
        import random
        random.seed(42)

        model = RLSModel(n_inputs=2, seed_coefficients=[0.0, 0.2, -1.0])
        model.frozen[1] = True  # Freeze outdoor_delta

        beta1_before = model.beta[1]
        for _ in range(100):
            outdoor = random.uniform(0, 20)
            solar = random.uniform(0, 1)
            y = 0.5 + 0.4 * outdoor - 3.5 * solar
            model.update([1.0, outdoor, solar], y)

        # outdoor_delta frozen at seed
        assert model.beta[1] == beta1_before
        # intercept and solar should have learned
        assert model.beta[2] != -1.0


# ── PIController coefficient API (unit tests via integration) ────────


class TestPIControllerCoefficientAPI:
    """Tests for PIController.get/set_coefficient and freeze methods."""

    @pytest.mark.asyncio
    async def test_coeff_names(self, hass, setup_pi_integration):
        """coeff_names should return intercept, outdoor_delta, plus model inputs."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        names = entity._pi.coeff_names
        assert names[0] == "intercept"
        assert names[1] == "outdoor_delta"
        assert len(names) == entity._pi.n_coefficients

    @pytest.mark.asyncio
    async def test_get_coefficient(self, hass, setup_pi_integration):
        """get_coefficient should return coefficient in physical units."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        val = pi.get_coefficient("heat", 1)
        assert val is not None
        # Should match the configured heat slope seed
        assert val == pytest.approx(0.3, abs=0.01)

    @pytest.mark.asyncio
    async def test_set_coefficient(self, hass, setup_pi_integration):
        """set_coefficient should update the coefficient value."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi.set_coefficient("heat", 1, 0.5)
        assert pi.get_coefficient("heat", 1) == pytest.approx(0.5, abs=0.001)

    @pytest.mark.asyncio
    async def test_set_coefficient_mode_isolation(self, hass, setup_pi_integration):
        """Setting a heat coefficient should not affect cool."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        cool_before = pi.get_coefficient("cool", 1)
        pi.set_coefficient("heat", 1, 0.99)
        assert pi.get_coefficient("cool", 1) == pytest.approx(cool_before, abs=0.001)

    @pytest.mark.asyncio
    async def test_get_set_frozen(self, hass, setup_pi_integration):
        """get_frozen/set_frozen should read/write freeze state."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        assert pi.get_frozen("heat", 1) is False
        pi.set_frozen("heat", 1, True)
        assert pi.get_frozen("heat", 1) is True
        pi.set_frozen("heat", 1, False)
        assert pi.get_frozen("heat", 1) is False

    @pytest.mark.asyncio
    async def test_get_coefficient_out_of_range(self, hass, setup_pi_integration):
        """get_coefficient with invalid index should return None."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        assert entity._pi.get_coefficient("heat", 99) is None

    @pytest.mark.asyncio
    async def test_get_frozen_out_of_range(self, hass, setup_pi_integration):
        """get_frozen with invalid index should return False."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        assert entity._pi.get_frozen("heat", 99) is False

    @pytest.mark.asyncio
    async def test_get_coefficient_clamp(self, hass, setup_pi_integration):
        """get_coefficient_clamp should return clamp tuple for outdoor_delta."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        clamp = pi.get_coefficient_clamp("heat", 1)
        assert clamp is not None
        assert len(clamp) == 2
        assert clamp[0] <= clamp[1]

    @pytest.mark.asyncio
    async def test_get_coefficient_clamp_unclamped(self, hass, setup_pi_integration):
        """get_coefficient_clamp should return None for intercept (unclamped)."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        assert entity._pi.get_coefficient_clamp("heat", 0) is None


# ── Service integration tests ────────────────────────────────────────


class TestSetCoefficientService:
    """Tests for set_coefficient service call."""

    @pytest.mark.asyncio
    async def test_set_coefficient_service(self, hass, setup_pi_integration):
        """set_coefficient service should update the specified coefficient."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await hass.services.async_call(
            DOMAIN, "set_coefficient",
            {
                "entity_id": entity.entity_id,
                "mode": "heat",
                "name": "outdoor_delta",
                "value": 0.42,
            },
            blocking=True,
        )

        assert entity._pi.get_coefficient("heat", 1) == pytest.approx(0.42, abs=0.001)

    @pytest.mark.asyncio
    async def test_set_coefficient_intercept(self, hass, setup_pi_integration):
        """set_coefficient should work for intercept."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await hass.services.async_call(
            DOMAIN, "set_coefficient",
            {
                "entity_id": entity.entity_id,
                "mode": "cool",
                "name": "intercept",
                "value": 1.5,
            },
            blocking=True,
        )

        assert entity._pi.get_coefficient("cool", 0) == pytest.approx(1.5, abs=0.001)

    @pytest.mark.asyncio
    async def test_set_coefficient_unknown_name(self, hass, setup_pi_integration):
        """set_coefficient with unknown name should log warning, not crash."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        # Should not raise
        await hass.services.async_call(
            DOMAIN, "set_coefficient",
            {
                "entity_id": entity.entity_id,
                "mode": "heat",
                "name": "nonexistent",
                "value": 1.0,
            },
            blocking=True,
        )


class TestFreezeCoefficientService:
    """Tests for freeze_coefficient service call."""

    @pytest.mark.asyncio
    async def test_freeze_coefficient_service(self, hass, setup_pi_integration):
        """freeze_coefficient service should freeze the specified coefficient."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await hass.services.async_call(
            DOMAIN, "freeze_coefficient",
            {
                "entity_id": entity.entity_id,
                "mode": "heat",
                "name": "outdoor_delta",
                "frozen": True,
            },
            blocking=True,
        )

        assert entity._pi.get_frozen("heat", 1) is True

    @pytest.mark.asyncio
    async def test_unfreeze_coefficient_service(self, hass, setup_pi_integration):
        """freeze_coefficient with frozen=False should unfreeze."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        # Freeze first
        entity._pi.set_frozen("heat", 1, True)

        await hass.services.async_call(
            DOMAIN, "freeze_coefficient",
            {
                "entity_id": entity.entity_id,
                "mode": "heat",
                "name": "outdoor_delta",
                "frozen": False,
            },
            blocking=True,
        )

        assert entity._pi.get_frozen("heat", 1) is False

    @pytest.mark.asyncio
    async def test_freeze_unknown_name(self, hass, setup_pi_integration):
        """freeze_coefficient with unknown name should not crash."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await hass.services.async_call(
            DOMAIN, "freeze_coefficient",
            {
                "entity_id": entity.entity_id,
                "mode": "cool",
                "name": "nonexistent",
                "frozen": True,
            },
            blocking=True,
        )

    @pytest.mark.asyncio
    async def test_freeze_mode_isolation(self, hass, setup_pi_integration):
        """Freezing heat coefficient should not affect cool."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)

        await hass.services.async_call(
            DOMAIN, "freeze_coefficient",
            {
                "entity_id": entity.entity_id,
                "mode": "heat",
                "name": "outdoor_delta",
                "frozen": True,
            },
            blocking=True,
        )

        assert entity._pi.get_frozen("heat", 1) is True
        assert entity._pi.get_frozen("cool", 1) is False
