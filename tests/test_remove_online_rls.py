"""Contract tests for online RLS removal.

These tests pin the contracts that the removal must preserve and the contracts
the removal must establish.  They serve as TDD scaffolding for phases 1-4.

T1 — Legacy stored data with ``rls_online_enabled`` loads gracefully
     (regression guard; passes today, must keep passing after Phase 4).
T2 — ``RLSModel.update`` is not invoked by ``_rls_learn_observation``
     (Phase 2 contract; FAILS today, passes after Phase 2 removes the call).
T3 — FF prediction works with online disabled
     (regression guard; passes today, must keep passing through removal).
T4 — Bench ``production_pi`` reference scenarios pass with online OFF
     (verified by running ``tests/hvac_bench/scenarios/test_reference_scores.py``;
     production_pi already overrides ``pi_rls_online_enabled=False``).
"""

from unittest.mock import patch

import pytest

from custom_components.tasmota_irhvac.pi.pi_controller import PIController
from custom_components.tasmota_irhvac.pi.pi_stored_data import PIExtraStoredData
from custom_components.tasmota_irhvac.pi.rls_model import RLSModel

from .conftest import make_pi_config
from .test_pi_controller import FakePIEntity


class TestT1LegacyStoredDataLoads:
    """Legacy stored data containing ``rls_online_enabled`` must restore cleanly."""

    def test_from_dict_accepts_legacy_rls_online_enabled_field(self):
        """Phase 4 contract: removing the field from the dataclass must not
        break loading of stored data written before the removal.

        Today this passes because the field exists.  After Phase 4 removes
        the dataclass field, ``from_dict`` will simply not pass the kwarg —
        the legacy value is silently dropped.  Either way: no crash.
        """
        legacy = {
            "pi_integral": 0.0,
            "rls_online_enabled": True,
        }
        restored = PIExtraStoredData.from_dict(legacy)
        assert restored is not None
        assert restored.pi_integral == 0.0

    def test_from_dict_accepts_missing_rls_online_enabled_field(self):
        """Stored data without the field (very old installs or post-removal)
        must restore using whatever default the dataclass currently has.
        """
        legacy = {"pi_integral": 0.0}
        restored = PIExtraStoredData.from_dict(legacy)
        assert restored is not None


class TestT2NoOnlineUpdateFromLearnObservation:
    """Phase 2 contract: ``_rls_learn_observation`` must not invoke ``RLSModel.update``.

    Today this test FAILS because ``_rls_learn_observation`` calls
    ``rls.update(x, observed_offset)`` at pi_controller.py:3849.  After Phase 2
    removes that call site, the test passes.

    The test calls ``_rls_learn_observation`` directly so it does not depend
    on the per-tick gating chain.
    """

    @pytest.mark.xfail(
        reason="Phase 2 removes the rls.update() call site; passes after Phase 2.",
        strict=True,
    )
    def test_learn_observation_does_not_call_rls_update(self):
        entity = FakePIEntity(make_pi_config())
        pi = entity._pi
        # Ensure the test seam is True so the function would call update today.
        pi._rls_online_learning = True

        update_calls: list[tuple[list[float], float]] = []
        original_update = RLSModel.update

        def counting_update(self, x, y):
            update_calls.append((list(x), y))
            return original_update(self, x, y)

        x = [1.0, 0.0] + [0.0] * (pi._rls_heat.n - 2)
        with patch.object(RLSModel, "update", counting_update):
            pi._rls_learn_observation(pi._rls_heat, x, 0.5, "test")

        assert update_calls == [], (
            f"Phase 2 contract violated: RLSModel.update called "
            f"{len(update_calls)} time(s) from _rls_learn_observation"
        )


class TestT3FFPredictionWithOnlineDisabled:
    """FF prediction must work when ``pi_rls_online_enabled`` is False.

    The user-facing flag default is already False (DEFAULT_PI_RLS_ONLINE_ENABLED).
    This test pins that the FF prediction path (rls.predict) is not gated off
    along with the online update path.
    """

    def test_predict_returns_nonzero_for_nonzero_beta(self):
        """rls.predict() reads beta only — it does not depend on the online flag."""
        config = make_pi_config({"pi_rls_online_enabled": False})
        entity = FakePIEntity(config)
        pi = entity._pi
        assert pi._pi_rls_online_enabled is False

        # Set a non-zero outdoor_delta coefficient (β index 1, in normalized space).
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]
        # Feature vector: intercept=1, outdoor_delta=10°C, rest=0.
        x = [1.0, 10.0] + [0.0] * (pi._rls_heat.n - 2)
        offset = pi._rls_heat.predict(x)
        # 0.5 * 10 = 5.0; tolerate rounding from feature-scale normalization.
        assert offset == pytest.approx(5.0, abs=0.01)

    def test_rls_state_persists_round_trip_with_online_disabled(self):
        """β/P state must round-trip even when online learning never runs.

        Confirms that turning off online does not break the persistence layer
        that batch-only learning relies on.
        """
        config = make_pi_config({"pi_rls_online_enabled": False})
        entity = FakePIEntity(config)
        pi = entity._pi
        # Manually write coefficients (as batch would).
        pi._rls_heat.beta[1] = 0.42 * pi._rls_heat.feature_scales[1]

        snapshot = pi._rls_heat.as_dict()
        restored = RLSModel.from_dict(
            snapshot,
            n_inputs=pi._rls_heat.n - 1,
            feature_scales=pi._rls_heat.feature_scales,
            coeff_clamps=pi._rls_heat.coeff_clamps,
        )
        assert restored is not None
        # β survives without going through .update().
        assert restored.beta[1] == pytest.approx(
            0.42 * pi._rls_heat.feature_scales[1], abs=1e-9
        )
