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
        # But floor is 30 min, so observed_tau = 35 (above floor).
        # EMA: α = 1/(2+0) = 0.5, so τ = 0.5*120 + 0.5*35 = 77.5
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

    def test_tau_floor_at_15_min(self):
        """Observed τ is floored at 15 minutes."""
        est = self._make(tau=120.0, lag=15.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        # Reach 63.2% at t=16 min → raw τ = 16-15 = 1 min → floored to 15
        est.check_observation(960.0, 21.27)
        assert est.observations == 1
        # EMA: α=0.5, τ = 0.5*120 + 0.5*15 = 67.5
        assert est.tau == pytest.approx(67.5, abs=1.0)

    def test_first_observation_alpha_is_half(self):
        """First observation gets α=0.5, not 1.0 — seed is preserved."""
        est = self._make(tau=120.0, lag=15.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        # 75 min elapsed → observed τ = 75 - 15 = 60 min
        est.check_observation(4500.0, 21.27)
        assert est.observations == 1
        # α=0.5: τ = 0.5*120 + 0.5*60 = 90 (NOT 60)
        assert est.tau == pytest.approx(90.0, abs=0.1)

    def test_multiple_observations_ema(self):
        """Multiple τ observations produce EMA convergence."""
        est = self._make(tau=120.0, lag=15.0)
        # First observation: 75min elapsed → observed τ = 60
        # α=0.5: τ = 0.5*120 + 0.5*60 = 90
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        est.check_observation(4500.0, 21.27)
        assert est.observations == 1
        assert est.tau == pytest.approx(90.0, abs=0.1)

        # Second observation: 105min elapsed → observed τ = 90
        # α = max(0.3, 1/(2+1)) = 0.333: τ = 0.667*90 + 0.333*90 = 90
        est.start_observation(5000.0, 20.0, 22.0, 2.0)
        est.check_observation(11300.0, 21.27)
        assert est.observations == 2
        assert est.tau == pytest.approx(90.0, abs=0.5)

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


class TestDisturbanceGating:
    """Tests for FF-offset-based disturbance gating."""

    def _make(self, tau=120.0, lag=15.0):
        return TauEstimator(tau_seed=tau, response_lag=lag, imc_lambda=0.0)

    def test_observation_accepted_when_ff_stable(self):
        """Observation accepted when FF offset changes < threshold."""
        est = self._make()
        est.start_observation(0.0, 20.0, 22.0, 2.0, ff_offset=3.0)
        # FF changed by 0.5 — below 1.0 threshold
        result = est.check_observation(4500.0, 21.27, ff_offset=3.5)
        assert result is not None
        assert est.observations == 1

    def test_observation_rejected_when_ff_changed(self):
        """Observation rejected when FF offset changes > threshold."""
        est = self._make()
        est.start_observation(0.0, 20.0, 22.0, 2.0, ff_offset=2.0)
        # FF changed by 1.5 — above 1.0 threshold
        result = est.check_observation(4500.0, 21.27, ff_offset=3.5)
        assert result is None
        assert not est.active  # Observation cleared
        assert est.observations == 0  # Not counted

    def test_observation_rejected_when_ff_decreased(self):
        """Disturbance gate works for negative FF changes too."""
        est = self._make()
        est.start_observation(0.0, 20.0, 22.0, 2.0, ff_offset=5.0)
        result = est.check_observation(4500.0, 21.27, ff_offset=3.5)
        assert result is None
        assert est.observations == 0

    def test_ff_offset_default_zero(self):
        """Backward compat: ff_offset defaults to 0.0 at both call sites."""
        est = self._make()
        est.start_observation(0.0, 20.0, 22.0, 2.0)  # ff_offset=0.0 default
        result = est.check_observation(4500.0, 21.27)  # ff_offset=0.0 default
        assert result is not None
        assert est.observations == 1


class TestOutlierRejection:
    """Tests for outlier rejection of τ observations."""

    def _make(self, tau=120.0, lag=15.0):
        return TauEstimator(tau_seed=tau, response_lag=lag, imc_lambda=0.0)

    def _observe(self, est, elapsed_s, start_time=0.0):
        """Helper: run one clean observation at the given elapsed time."""
        est.start_observation(start_time, 20.0, 22.0, 2.0)
        result = est.check_observation(start_time + elapsed_s, 21.27)
        return result

    def test_first_two_observations_not_rejected(self):
        """Outlier rejection only kicks in after ≥2 prior observations."""
        est = self._make(tau=120.0, lag=15.0)
        # First obs: 75 min → observed τ=60, α=0.5 → τ=90
        self._observe(est, 4500.0)
        assert est.observations == 1

        # Second obs: very short, 46 min → observed τ=31 (just above floor)
        # No outlier rejection yet (only 1 prior observation)
        # α=0.333 → τ = 0.667*90 + 0.333*31 ≈ 70.3
        self._observe(est, 2760.0, start_time=5000.0)
        assert est.observations == 2

    def test_outlier_rejected_after_two_observations(self):
        """After 2 observations, extreme τ values are rejected."""
        est = self._make(tau=90.0, lag=15.0)
        # Build up 2 observations near τ≈90
        # First: 105 min → observed=90, α=0.5 → τ=90
        self._observe(est, 6300.0)
        # Second: 105 min → observed=90, α=0.333 → τ=90
        self._observe(est, 6300.0, start_time=7000.0)
        assert est.observations == 2
        tau_before = est.tau

        # Third obs: very fast, 46 min → observed=31 min
        # Ratio = 31/90 ≈ 0.34, limit = 1/3 ≈ 0.33 — just barely accepted
        # Let's try something more extreme: 45.5 min → observed=30.5→floored to 30
        # Ratio = 30/90 ≈ 0.33 — at the limit
        # Use 45 min → raw=30, floored=30, ratio=30/90=0.33 — exactly at limit
        # Use 44 min → raw=29→floored=30, ratio=30/90=0.33 — still 30 (floor)
        # We need observed_tau < 90/3=30, but floor is 30. So floor protects us
        # for very short observations. Test with a high outlier instead.

        # High outlier: 1000 min → observed=985
        # Ratio = 985/90 ≈ 10.9, limit = 3.0 → rejected
        est.start_observation(14000.0, 20.0, 22.0, 2.0)
        result = est.check_observation(14000.0 + 60000.0, 21.27)
        assert result is None
        assert est.observations == 2  # Not incremented
        assert est.tau == pytest.approx(tau_before, abs=0.1)  # Unchanged

    def test_non_outlier_accepted_after_two_observations(self):
        """Values within 3× of current estimate are accepted after 2 obs."""
        est = self._make(tau=90.0, lag=15.0)
        # Two observations near 90
        self._observe(est, 6300.0)
        self._observe(est, 6300.0, start_time=7000.0)
        assert est.observations == 2

        # Third obs: 75 min → observed=60. Ratio=60/90=0.67, within bounds
        result = self._observe(est, 4500.0, start_time=14000.0)
        assert result is not None
        assert est.observations == 3

    def test_outlier_rejection_interacts_with_floor(self):
        """The 15-min floor + outlier rejection together prevent τ collapse."""
        # Use tau=50 so that floor(15)/50 = 0.3 < 1/3 → outlier rejected
        est = self._make(tau=50.0, lag=15.0)
        # Two observations at 65 min → observed=50, keeps τ≈50
        self._observe(est, 3900.0)
        self._observe(est, 3900.0, start_time=4000.0)
        tau_stable = est.tau
        assert tau_stable == pytest.approx(50.0, abs=1.0)

        # Room warms in 16 min (raw τ=1, floored to 15).
        # 15/50 = 0.3 < 1/3 → rejected as outlier
        est.start_observation(8000.0, 20.0, 22.0, 2.0)
        result = est.check_observation(8000.0 + 960.0, 21.27)
        assert result is None
        assert est.tau == pytest.approx(tau_stable, abs=0.1)


class TestStartObservationFFOffset:
    """Tests that start_observation records FF offset correctly."""

    def test_ff_offset_recorded(self):
        est = TauEstimator(tau_seed=60.0, response_lag=15.0, imc_lambda=0.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0, ff_offset=4.5)
        assert est._step_ff_offset == 4.5

    def test_ff_offset_defaults_to_zero(self):
        est = TauEstimator(tau_seed=60.0, response_lag=15.0, imc_lambda=0.0)
        est.start_observation(0.0, 20.0, 22.0, 2.0)
        assert est._step_ff_offset == 0.0
