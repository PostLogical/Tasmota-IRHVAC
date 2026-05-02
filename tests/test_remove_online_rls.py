"""Contract tests for online RLS removal.

These tests pin the contracts that the removal preserved (T1, T3, T4) and the
contract it established (T2).  They serve as regression guards.

T1 — Legacy stored data with ``rls_online_enabled`` loads gracefully
     (the field was dropped from the dataclass; from_dict ignores unknown keys).
T2 — ``RLSModel.update`` is never invoked from production code paths.
T3 — FF prediction works (no online flag exists; predict reads beta only).
T4 — Bench ``production_pi`` reference scenarios pass
     (verified by ``tests/hvac_bench/scenarios/test_reference_scores.py``).
"""

from unittest.mock import patch

import pytest

from custom_components.tasmota_irhvac.pi.pi_stored_data import PIExtraStoredData
from custom_components.tasmota_irhvac.pi.rls_model import RLSModel

from .conftest import make_pi_config
from .test_pi_controller import FakePIEntity


class TestT1LegacyStoredDataLoads:
    """Legacy stored data containing ``rls_online_enabled`` must restore cleanly."""

    def test_from_dict_accepts_legacy_rls_online_enabled_field(self):
        """``from_dict`` silently ignores the legacy ``rls_online_enabled`` key.

        The field was removed from ``PIExtraStoredData`` in Phase 4; existing
        installs may have written stored data containing it before the upgrade.
        """
        legacy = {
            "pi_integral": 0.0,
            "rls_online_enabled": True,
        }
        restored = PIExtraStoredData.from_dict(legacy)
        assert restored is not None
        assert restored.pi_integral == 0.0

    def test_from_dict_accepts_missing_rls_online_enabled_field(self):
        """Stored data without the field (current state) must restore."""
        legacy = {"pi_integral": 0.0}
        restored = PIExtraStoredData.from_dict(legacy)
        assert restored is not None


class TestT2NoOnlineUpdateFromTick:
    """``RLSModel.update`` must not be invoked from any production tick path.

    A regression that re-introduces an online update would trip this immediately.
    """

    @pytest.mark.asyncio
    async def test_pi_tick_does_not_call_rls_update(self):
        from homeassistant.components.climate import HVACMode

        entity = FakePIEntity(make_pi_config())
        pi = entity._pi
        pi._inputs.outdoor_temp = 0.0
        entity._attr_current_temperature = 20.0
        pi._desired_temp = 22.0
        pi._hp_setpoint = 22.0
        entity._attr_hvac_mode = HVACMode.HEAT

        update_calls: list[tuple[list[float], float]] = []
        original_update = RLSModel.update

        def counting_update(self, x, y):
            update_calls.append((list(x), y))
            return original_update(self, x, y)

        with patch.object(RLSModel, "update", counting_update):
            for _ in range(5):
                await pi._pi_tick()

        assert update_calls == [], (
            f"Online RLS regression: RLSModel.update called "
            f"{len(update_calls)} time(s) during a 5-tick run"
        )


class TestT3FFPredictionStillWorks:
    """FF prediction must keep working post-removal (predict reads beta only)."""

    def test_predict_returns_nonzero_for_nonzero_beta(self):
        entity = FakePIEntity(make_pi_config())
        pi = entity._pi

        # Set a non-zero outdoor_delta coefficient (β index 1, in normalized space).
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]
        # Feature vector: intercept=1, outdoor_delta=10°C, rest=0.
        x = [1.0, 10.0] + [0.0] * (pi._rls_heat.n - 2)
        offset = pi._rls_heat.predict(x)
        # 0.5 * 10 = 5.0; tolerate rounding from feature-scale normalization.
        assert offset == pytest.approx(5.0, abs=0.01)

    def test_rls_state_persists_round_trip(self):
        """β/P state round-trip works (batch-only learning depends on it)."""
        entity = FakePIEntity(make_pi_config())
        pi = entity._pi
        # Manually write a coefficient (as batch would).
        pi._rls_heat.beta[1] = 0.42 * pi._rls_heat.feature_scales[1]

        snapshot = pi._rls_heat.as_dict()
        restored = RLSModel.from_dict(
            snapshot,
            n_inputs=pi._rls_heat.n - 1,
            feature_scales=pi._rls_heat.feature_scales,
            coeff_clamps=pi._rls_heat.coeff_clamps,
        )
        assert restored is not None
        assert restored.beta[1] == pytest.approx(
            0.42 * pi._rls_heat.feature_scales[1], abs=1e-9
        )
