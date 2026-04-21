"""Unit tests for PlantIdentifier orchestrator — no HA dependencies."""

import pytest

from custom_components.tasmota_irhvac.pi.plant_identifier import PlantIdentifier
from custom_components.tasmota_irhvac.pi.plant_model import GainUpdate, PlantEstimate


class TestPlantIdentifier:
    """Tests for the plant identification orchestrator."""

    def _make(self, tau=60.0, lag=15.0, imc_lambda=0.0):
        return PlantIdentifier(tau_seed=tau, response_lag=lag, imc_lambda=imc_lambda)

    def test_initial_plant_from_seeds(self):
        """PlantEstimate starts from configured seeds."""
        pi = self._make(tau=60.0, lag=15.0)
        assert pi.plant.tau_fast.value == 60.0
        assert pi.plant.tau_fast.source == "seed"
        assert pi.plant.tau_slow.value == 60.0
        assert pi.plant.tau_slow.source == "seed"
        assert pi.plant.k.value == 1.0
        assert pi.plant.theta.value == 15.0

    def test_enabled_when_tau_seed_positive(self):
        pi = self._make(tau=60.0)
        assert pi.enabled

    def test_disabled_when_tau_seed_zero(self):
        pi = self._make(tau=0.0)
        assert not pi.enabled

    def test_compute_gains_from_seeds(self):
        """IMC gains derive from seed τ_slow."""
        pi = self._make(tau=60.0, lag=15.0)
        gains = pi.compute_gains()
        # λ = L/3 = 5, Kp = 60/(1*(5+15)) = 3.0
        assert gains.kp == pytest.approx(3.0)
        assert gains.ki == pytest.approx(0.15)
        assert gains.tau_fast == 60.0
        assert gains.tau_slow == 60.0
        assert gains.lag == 15.0

    def test_compute_gains_explicit_lambda(self):
        pi = self._make(tau=60.0, lag=15.0, imc_lambda=10.0)
        gains = pi.compute_gains()
        # Kp = 60/(1*(10+15)) = 2.4
        assert gains.kp == pytest.approx(2.4)

    def test_start_observation_fans_out(self):
        """start_observation activates the step response provider."""
        pi = self._make()
        pi.start_observation(0.0, 20.0, 22.0, 2.0, ff_offset=1.0)
        assert pi.active

    def test_start_observation_ignores_small_step(self):
        pi = self._make()
        pi.start_observation(0.0, 20.0, 20.3, 0.3)
        assert not pi.active

    def test_start_observation_noop_when_disabled(self):
        pi = self._make(tau=0.0)
        pi.start_observation(0.0, 20.0, 22.0, 2.0)
        assert not pi.active

    def test_check_observation_updates_tau_fast(self):
        """Step-response observation updates τ_fast in PlantEstimate."""
        pi = self._make(tau=120.0, lag=15.0)
        pi.start_observation(0.0, 20.0, 22.0, 2.0)

        # 50 min → observed τ_fast = 35
        result = pi.check_observation(3000.0, 21.27)
        assert result is not None
        assert isinstance(result, GainUpdate)
        assert pi.plant.tau_fast.source == "step_response"
        assert pi.plant.tau_fast.observations == 1
        assert pi.plant.tau_fast.value < 120.0  # Moved toward observed

    def test_check_observation_returns_none_when_not_ready(self):
        pi = self._make(tau=120.0, lag=15.0)
        pi.start_observation(0.0, 20.0, 22.0, 2.0)
        result = pi.check_observation(600.0, 20.5)
        assert result is None

    def test_cancel_observation(self):
        pi = self._make()
        pi.start_observation(0.0, 20.0, 22.0, 2.0)
        assert pi.active
        pi.cancel_observation()
        assert not pi.active

    def test_backward_compat_tau_property(self):
        """The .tau property returns tau_fast for backward compat."""
        pi = self._make(tau=42.0)
        assert pi.tau == 42.0

    def test_backward_compat_observations_property(self):
        pi = self._make(tau=60.0)
        assert pi.observations == 0

    def test_tau_slow_unchanged_by_step_response(self):
        """Step-response observation only updates tau_fast, not tau_slow."""
        pi = self._make(tau=60.0, lag=15.0)
        pi.start_observation(0.0, 20.0, 22.0, 2.0)
        pi.check_observation(3000.0, 21.27)
        # tau_slow should still be the seed
        assert pi.plant.tau_slow.source == "seed"
        assert pi.plant.tau_slow.value == 60.0


class TestPlantIdentifierPersistence:
    """Tests for save/restore round-trip."""

    def _make(self, tau=60.0, lag=15.0):
        return PlantIdentifier(tau_seed=tau, response_lag=lag, imc_lambda=0.0)

    def test_round_trip(self):
        """as_dict → restore preserves plant estimate."""
        pi = self._make(tau=120.0, lag=15.0)
        pi.start_observation(0.0, 20.0, 22.0, 2.0)
        pi.check_observation(4500.0, 21.27)  # τ_fast observation fires
        assert pi.plant.tau_fast.observations == 1

        d = pi.as_dict()
        pi2 = self._make(tau=120.0, lag=15.0)
        pi2.restore(d)

        assert pi2.plant.tau_fast.value == pytest.approx(pi.plant.tau_fast.value, abs=0.1)
        assert pi2.plant.tau_fast.observations == 1
        assert pi2.plant.tau_fast.source == "step_response"

    def test_migration_from_old_single_tau(self):
        """Old persistence data (tau_estimate/tau_observations) migrates to tau_fast."""
        pi = self._make(tau=60.0, lag=15.0)
        gains = pi.restore({
            "tau_estimate": 90.0,
            "tau_observations": 3,
        })
        assert pi.plant.tau_fast.value == 90.0
        assert pi.plant.tau_fast.observations == 3
        assert pi.plant.tau_fast.source == "step_response"
        # tau_slow stays at seed
        assert pi.plant.tau_slow.source == "seed"
        # Gains computed from restored state
        assert gains.kp > 0

    def test_backward_compat_fields_in_as_dict(self):
        """as_dict writes tau_estimate for rollback compat."""
        pi = self._make(tau=42.0)
        d = pi.as_dict()
        assert "tau_estimate" in d
        assert d["tau_estimate"] == 42.0
        assert "plant_estimate" in d

    def test_restore_with_empty_dict(self):
        """Empty dict restores cleanly — keeps seeds."""
        pi = self._make(tau=60.0, lag=15.0)
        gains = pi.restore({})
        assert pi.plant.tau_fast.value == 60.0  # seed
        assert gains.kp == pytest.approx(3.0)


class TestGainUpdateFields:
    """Verify GainUpdate has the correct dual-tau fields."""

    def test_gain_update_has_tau_fast_and_tau_slow(self):
        pi = PlantIdentifier(tau_seed=60.0, response_lag=15.0, imc_lambda=0.0)
        gains = pi.compute_gains()
        assert hasattr(gains, "tau_fast")
        assert hasattr(gains, "tau_slow")
        assert not hasattr(gains, "tau")  # Old field removed


class TestPlantIdentifierDiagnostics:
    """Tests for get_diagnostics() method."""

    def _make(self, tau=60.0, lag=15.0):
        return PlantIdentifier(tau_seed=tau, response_lag=lag, imc_lambda=0.0)

    def test_baseline_diagnostics(self):
        """Fresh identifier returns plant estimate and provider states."""
        pi = self._make()
        diag = pi.get_diagnostics()

        # Full SOPDT plant estimate
        est = diag["plant_estimate"]
        for param in ("k", "theta", "tau_fast", "tau_slow"):
            assert param in est
            assert est[param]["source"] == "seed"

        # All three providers present and inactive
        providers = diag["providers"]
        assert not providers["step_response"]["active"]
        assert not providers["area_method"]["active"]
        assert not providers["closed_loop"]["active"]

        # No cross-check or plant test initially
        assert "cross_check" not in diag
        assert "plant_test" not in diag

    def test_diagnostics_with_active_providers(self):
        """Active observation shows in provider state."""
        pi = self._make()
        pi.start_observation(0.0, 20.0, 22.0, 2.0)
        diag = pi.get_diagnostics()

        assert diag["providers"]["step_response"]["active"]
        assert diag["providers"]["area_method"]["active"]
        assert diag["providers"]["closed_loop"]["active"]

    def test_diagnostics_with_cross_check(self):
        """Cross-check data appears after closed-loop fires."""
        from custom_components.tasmota_irhvac.pi.plant_model import ParameterEstimate

        pi = self._make()
        # Manually inject cross-check data (normally set by check_observation)
        pi._last_cross_check = (
            ParameterEstimate(value=55.0, confidence=0.8, source="closed_loop", observations=3),
            ParameterEstimate(value=120.0, confidence=0.7, source="closed_loop", observations=3),
        )
        diag = pi.get_diagnostics()

        assert "cross_check" in diag
        assert diag["cross_check"]["tau_fast"]["value"] == 55.0
        assert diag["cross_check"]["tau_slow"]["value"] == 120.0
        assert diag["cross_check"]["tau_slow"]["source"] == "closed_loop"

    def test_diagnostics_with_plant_test(self):
        """Active plant test state appears in diagnostics."""
        pi = self._make()
        pi.start_plant_test(
            baseline_setpoint_c=22, amplitude_c=2,
            current_c=21.5, comfort_min_c=19.0, comfort_max_c=25.0,
        )
        diag = pi.get_diagnostics()

        assert "plant_test" in diag
        assert diag["plant_test"]["active"] is True
        assert diag["plant_test"]["phase"] == "relay_high"
        assert diag["plant_test"]["cycle_count"] == 0


class TestGreyboxTauProvider:
    """Tests for grey-box τ_eff → tau_slow integration."""

    def _make(self, tau=60.0, lag=15.0, imc_lambda=5.0):
        return PlantIdentifier(tau_seed=tau, response_lag=lag, imc_lambda=imc_lambda)

    def test_updates_seed_tau_slow(self):
        """Grey-box τ_eff should update tau_slow when source is seed."""
        pi = self._make(tau=60.0)
        assert pi.plant.tau_slow.source == "seed"

        gains = pi.update_from_greybox(tau_eff=100.0, ua_c_cv=0.1)
        assert gains is not None
        assert pi.plant.tau_slow.source == "greybox"
        assert pi.plant.tau_slow.value == 100.0

    def test_confidence_from_cv(self):
        """Confidence = max(0, 1 - 2×CV), discounted by 0.8."""
        pi = self._make(tau=60.0)
        pi.update_from_greybox(tau_eff=80.0, ua_c_cv=0.15)
        # conf = (1 - 2*0.15) * 0.8 = 0.7 * 0.8 = 0.56
        assert abs(pi.plant.tau_slow.confidence - 0.56) < 0.01

    def test_rejected_low_confidence(self):
        """High CV → low confidence → rejected."""
        pi = self._make(tau=60.0)
        gains = pi.update_from_greybox(tau_eff=100.0, ua_c_cv=0.45)
        # conf = (1 - 2*0.45) * 0.8 = 0.1 * 0.8 = 0.08 < 0.3
        assert gains is None
        assert pi.plant.tau_slow.source == "seed"  # unchanged

    def test_does_not_override_area_method(self):
        """Grey-box shouldn't replace a primary area_method estimate."""
        from custom_components.tasmota_irhvac.pi.plant_model import ParameterEstimate
        import dataclasses
        pi = self._make(tau=60.0)
        # Simulate area method having fired
        area_est = ParameterEstimate(
            value=120.0, confidence=0.9, source="area_method", observations=5
        )
        pi._plant = dataclasses.replace(pi._plant, tau_slow=area_est)

        gains = pi.update_from_greybox(tau_eff=100.0, ua_c_cv=0.1)
        assert gains is None
        assert pi.plant.tau_slow.source == "area_method"  # unchanged

    def test_overrides_closed_loop(self):
        """Grey-box should override a closed_loop interim estimate."""
        from custom_components.tasmota_irhvac.pi.plant_model import ParameterEstimate
        import dataclasses
        pi = self._make(tau=60.0)
        cl_est = ParameterEstimate(
            value=90.0, confidence=0.5, source="closed_loop", observations=1
        )
        pi._plant = dataclasses.replace(pi._plant, tau_slow=cl_est)

        gains = pi.update_from_greybox(tau_eff=100.0, ua_c_cv=0.1)
        assert gains is not None
        assert pi.plant.tau_slow.source == "greybox"

    def test_rejected_large_ratio(self):
        """τ_eff >2× current → rejected (too large a jump)."""
        pi = self._make(tau=60.0)
        gains = pi.update_from_greybox(tau_eff=200.0, ua_c_cv=0.1)
        assert gains is None
        assert pi.plant.tau_slow.source == "seed"

    def test_gains_updated(self):
        """Gain update should produce new Kp/Ki from IMC."""
        pi = self._make(tau=60.0, lag=15.0, imc_lambda=5.0)
        gains = pi.update_from_greybox(tau_eff=100.0, ua_c_cv=0.1)
        assert gains is not None
        assert gains.kp > 0
        assert gains.ki > 0

    def test_disabled_returns_none(self):
        """Disabled plant ID (tau_seed=0) → no update."""
        pi = PlantIdentifier(tau_seed=0.0, response_lag=15.0, imc_lambda=5.0)
        gains = pi.update_from_greybox(tau_eff=100.0, ua_c_cv=0.1)
        assert gains is None

    def test_successive_greybox_updates(self):
        """Grey-box can update its own previous estimate."""
        pi = self._make(tau=60.0)
        pi.update_from_greybox(tau_eff=80.0, ua_c_cv=0.15)
        assert pi.plant.tau_slow.source == "greybox"
        assert pi.plant.tau_slow.value == 80.0

        # Second update with slightly different value
        pi.update_from_greybox(tau_eff=90.0, ua_c_cv=0.1)
        assert pi.plant.tau_slow.value == 90.0
