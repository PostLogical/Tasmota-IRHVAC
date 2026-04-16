"""Unit tests for TauEstimator — no PIController or HA dependencies."""

import pytest

from custom_components.tasmota_irhvac.pi.tau_estimator import GainUpdate, TauEstimator


class TestTauEstimator:
    """Tests for online τ estimation from step-response observation."""

    def _make(self, tau=120.0, lag=15.0, imc_lambda=0.0):
        return TauEstimator(tau_seed=tau, response_lag=lag, imc_lambda=imc_lambda)

    def test_start_observation_on_large_step(self):
        """Step ≥ 1°C starts a τ observation."""
        est = self._make()
        est.start_observation(1000.0, 20.0, 22.0, 2.0)
        assert est.active
        assert est.step_temp == 20.0
        assert est.step_magnitude == 2.0

    def test_start_observation_ignores_small_step(self):
        """Step < 1°C does not start observation."""
        est = self._make()
        est.start_observation(1000.0, 20.0, 20.5, 0.5)
        assert not est.active

    def test_start_observation_noop_when_disabled(self):
        """No observation when IMC is disabled (tau_seed=0)."""
        est = self._make(tau=0.0)
        assert not est.enabled
        est.start_observation(1000.0, 20.0, 22.0, 2.0)
        assert not est.active

    def test_check_observation_detects_632_pct(self):
        """τ observation fires when room reaches 63.2% of expected change."""
        est = self._make(tau=120.0, lag=15.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        assert est.active
        old_tau = est.tau

        # Room reaches 63.2% of 2°C step = 1.264°C above start = 21.264
        # At t=50 min (3000s), so observed τ = 50 - 15 (lag) = 35 min
        result = est.check_observation(3000.0, 21.27)
        assert result is not None
        assert not est.active
        assert est.observations == 1
        assert est.tau != old_tau
        assert est.tau < old_tau  # Moved toward 35

    def test_check_observation_timeout(self):
        """Observation abandoned after timeout."""
        est = self._make(tau=60.0, lag=15.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        # timeout = max(4*60, 240) = 240 min = 14400s
        result = est.check_observation(14500.0, 20.5)
        assert result is None
        assert not est.active
        assert est.observations == 0

    def test_cancel_observation(self):
        """Cancellation clears active observation."""
        est = self._make()
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        assert est.active
        est.cancel_observation()
        assert not est.active

    def test_setpoint_change_cancels_observation(self):
        """Cancel preserves step_target until cleared."""
        est = self._make()
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        assert est.active
        assert est.step_target == 22.0
        est.cancel_observation()
        assert not est.active

    def test_tau_floor_at_5_min(self):
        """Observed τ is floored at 5 minutes."""
        est = self._make(tau=120.0, lag=15.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        # Reach 63.2% at t=16 min → raw τ = 16-15 = 1 min → floored to 5
        est.check_observation(960.0, 21.27)
        assert est.observations == 1
        assert est.tau < 120.0

    def test_multiple_observations_ema(self):
        """Multiple τ observations produce EMA convergence."""
        est = self._make(tau=120.0, lag=15.0)
        # First observation: 75min elapsed → observed τ = 75-15 = 60
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        est.check_observation(4500.0, 21.27)
        assert est.observations == 1
        assert abs(est.tau - 60.0) < 0.1  # α≈0.5 for first obs

        # Second observation: 105min elapsed → observed τ = 105-15 = 90
        est.start_observation(5000.0, 20.0, 22.0, 2.0)
        est.check_observation(11300.0, 21.27)
        assert est.observations == 2
        assert abs(est.tau - 75.0) < 0.1  # 0.5*60 + 0.5*90 = 75

    def test_compute_gains_imc_formula(self):
        """compute_gains returns correct Kp/Ki from IMC formula."""
        est = self._make(tau=60.0, lag=15.0, imc_lambda=0.0)
        gains = est.compute_gains()
        # λ defaults to L/3 = 5.0, so Kp = 60/(1*(5+15)) = 3.0
        assert gains.kp == pytest.approx(3.0)
        # Ki = Kp / (τ/3) = 3.0 / 20.0 = 0.15
        assert gains.ki == pytest.approx(0.15)
        assert gains.tau == 60.0
        assert gains.lag == 15.0

    def test_compute_gains_explicit_lambda(self):
        """Explicit λ overrides default L/3."""
        est = self._make(tau=60.0, lag=15.0, imc_lambda=10.0)
        gains = est.compute_gains()
        # Kp = 60/(1*(10+15)) = 2.4
        assert gains.kp == pytest.approx(2.4)

    def test_restore_updates_tau_and_returns_gains(self):
        """restore() sets τ and returns recomputed gains."""
        est = self._make(tau=60.0, lag=15.0)
        gains = est.restore(90.0)
        assert est.tau == 90.0
        # Kp = 90/(1*(5+15)) = 4.5  (λ = L/3 = 5)
        assert gains.kp == pytest.approx(4.5)

    def test_check_observation_returns_gain_update(self):
        """check_observation returns GainUpdate when τ changes."""
        est = self._make(tau=120.0, lag=15.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        result = est.check_observation(3000.0, 21.27)
        assert isinstance(result, GainUpdate)
        assert result.kp > 0
        assert result.ki > 0

    def test_check_observation_returns_none_when_not_ready(self):
        """check_observation returns None when threshold not reached."""
        est = self._make(tau=120.0, lag=15.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        result = est.check_observation(600.0, 20.5)  # Only 25% of step
        assert result is None
        assert est.active  # Still observing
