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
