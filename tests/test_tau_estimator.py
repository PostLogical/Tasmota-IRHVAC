"""Unit tests for StepResponseProvider — no PIController or HA dependencies.

Tests the Layer 1 step-response τ_fast identification provider,
extracted from the original TauEstimator.
"""

import pytest

from custom_components.tasmota_irhvac.pi.plant_model import ObservationContext, ParameterEstimate
from custom_components.tasmota_irhvac.pi.providers.step_response import StepResponseProvider


def _ctx(start=0.0, temp=20.0, target=22.0, step=2.0, ff=0.0):
    """Shorthand for creating an ObservationContext."""
    return ObservationContext(
        start_time=start, baseline_temp=temp, target_temp=target,
        step_magnitude=step, ff_offset=ff,
    )


class TestStepResponseProvider:
    """Tests for online τ_fast estimation from step-response observation."""

    def _make(self, lag=15.0):
        return StepResponseProvider(response_lag=lag)

    def test_start_observation_on_large_step(self):
        """Step ≥ 1°C starts a τ observation."""
        p = self._make()
        p.start_observation(_ctx(start=1000.0))
        assert p.active
        assert p.step_temp == 20.0
        assert p.step_magnitude == 2.0

    def test_start_observation_ignores_small_step(self):
        """Step < 1°C does not start observation."""
        p = self._make()
        p.start_observation(_ctx(step=0.5))
        assert not p.active

    def test_check_observation_detects_632_pct(self):
        """τ observation fires when room reaches 63.2% of expected change."""
        p = self._make(lag=15.0)
        p.start_observation(_ctx())
        assert p.active

        # Room reaches 63.2% of 2°C step = 1.264°C above start = 21.264
        # At t=50 min (3000s), observed τ = 50 - 15 (lag) = 35 min
        result = p.check_observation(3000.0, 21.27)
        assert result is not None
        assert isinstance(result, ParameterEstimate)
        assert not p.active
        assert p.observations == 1
        assert result.source == "step_response"

    def test_check_observation_timeout(self):
        """Observation abandoned after timeout."""
        p = self._make(lag=15.0)
        # No prior observations → ref_tau=60 → timeout=240 min=14400s
        p.start_observation(_ctx())
        result = p.check_observation(14500.0, 20.5)
        assert result is None
        assert not p.active
        assert p.observations == 0

    def test_cancel_observation(self):
        """Cancellation clears active observation."""
        p = self._make()
        p.start_observation(_ctx())
        assert p.active
        p.cancel_observation()
        assert not p.active

    def test_tau_floor_at_15_min(self):
        """Observed τ is floored at 15 minutes."""
        p = self._make(lag=15.0)
        p.start_observation(_ctx())
        # Reach 63.2% at t=16 min → raw τ = 16-15 = 1 min → floored to 15
        result = p.check_observation(960.0, 21.27)
        assert p.observations == 1
        # First obs with no prior → tau_fast = 15 directly
        assert p.tau_fast == pytest.approx(15.0, abs=1.0)

    def test_first_observation_with_prior_uses_ema(self):
        """When provider has a restored prior, first observation blends via EMA."""
        p = self._make(lag=15.0)
        p.restore({"tau_fast": 120.0, "observations": 0})
        p.start_observation(_ctx())
        # 75 min elapsed → observed τ = 60
        result = p.check_observation(4500.0, 21.27)
        assert p.observations == 1
        # α=0.5: τ = 0.5*120 + 0.5*60 = 90
        assert p.tau_fast == pytest.approx(90.0, abs=0.1)

    def test_multiple_observations_ema(self):
        """Multiple observations produce EMA convergence."""
        p = self._make(lag=15.0)
        p.restore({"tau_fast": 120.0, "observations": 0})

        # First: 75min → observed=60, α=0.5 → τ=90
        p.start_observation(_ctx())
        p.check_observation(4500.0, 21.27)
        assert p.observations == 1
        assert p.tau_fast == pytest.approx(90.0, abs=0.1)

        # Second: 105min → observed=90, α=0.333 → τ=90
        p.start_observation(_ctx(start=5000.0))
        p.check_observation(11300.0, 21.27)
        assert p.observations == 2
        assert p.tau_fast == pytest.approx(90.0, abs=0.5)

    def test_check_observation_returns_none_when_not_ready(self):
        """check_observation returns None when threshold not reached."""
        p = self._make(lag=15.0)
        p.start_observation(_ctx())
        result = p.check_observation(600.0, 20.5)  # Only 25% of step
        assert result is None
        assert p.active  # Still observing

    def test_persistence_round_trip(self):
        """as_dict/restore preserves state."""
        p = self._make(lag=15.0)
        p._tau_fast = 42.0
        p._observations = 3
        d = p.as_dict()

        p2 = self._make(lag=15.0)
        p2.restore(d)
        assert p2.tau_fast == 42.0
        assert p2.observations == 3


class TestDisturbanceGating:
    """Tests for FF-offset-based disturbance gating."""

    def _make(self, lag=15.0):
        return StepResponseProvider(response_lag=lag)

    def test_observation_accepted_when_ff_stable(self):
        """Observation accepted when FF offset changes < threshold."""
        p = self._make()
        p.start_observation(_ctx(ff=3.0))
        result = p.check_observation(4500.0, 21.27, ff_offset=3.5)
        assert result is not None
        assert p.observations == 1

    def test_observation_rejected_when_ff_changed(self):
        """Observation rejected when FF offset changes > threshold."""
        p = self._make()
        p.start_observation(_ctx(ff=2.0))
        result = p.check_observation(4500.0, 21.27, ff_offset=3.5)
        assert result is None
        assert not p.active
        assert p.observations == 0

    def test_observation_rejected_when_ff_decreased(self):
        """Disturbance gate works for negative FF changes too."""
        p = self._make()
        p.start_observation(_ctx(ff=5.0))
        result = p.check_observation(4500.0, 21.27, ff_offset=3.5)
        assert result is None
        assert p.observations == 0

    def test_ff_offset_default_zero(self):
        """ff_offset defaults to 0.0 at both call sites."""
        p = self._make()
        p.start_observation(_ctx())
        result = p.check_observation(4500.0, 21.27)
        assert result is not None
        assert p.observations == 1


class TestOutlierRejection:
    """Tests for outlier rejection of τ observations."""

    def _make(self, lag=15.0):
        return StepResponseProvider(response_lag=lag)

    def _observe(self, p, elapsed_s, start_time=0.0):
        """Helper: run one clean observation at the given elapsed time."""
        p.start_observation(_ctx(start=start_time))
        return p.check_observation(start_time + elapsed_s, 21.27)

    def test_first_two_observations_not_rejected(self):
        """Outlier rejection only kicks in after ≥2 prior observations."""
        p = self._make(lag=15.0)
        p.restore({"tau_fast": 120.0, "observations": 0})
        # First obs: 75 min → observed=60, α=0.5 → τ≈90
        self._observe(p, 4500.0)
        assert p.observations == 1

        # Second obs: very short, 46 min → observed=31
        self._observe(p, 2760.0, start_time=5000.0)
        assert p.observations == 2

    def test_outlier_rejected_after_two_observations(self):
        """After 2 observations, extreme τ values are rejected."""
        p = self._make(lag=15.0)
        p.restore({"tau_fast": 90.0, "observations": 0})
        # Build up 2 observations near τ≈90
        self._observe(p, 6300.0)
        self._observe(p, 6300.0, start_time=7000.0)
        assert p.observations == 2
        tau_before = p.tau_fast

        # High outlier: 1000 min → observed=985, ratio=985/τ ≈ 10.9 → rejected
        p.start_observation(_ctx(start=14000.0))
        result = p.check_observation(14000.0 + 60000.0, 21.27)
        assert result is None
        assert p.observations == 2
        assert p.tau_fast == pytest.approx(tau_before, abs=0.1)

    def test_non_outlier_accepted_after_two_observations(self):
        """Values within 3× of current estimate are accepted after 2 obs."""
        p = self._make(lag=15.0)
        p.restore({"tau_fast": 90.0, "observations": 0})
        self._observe(p, 6300.0)
        self._observe(p, 6300.0, start_time=7000.0)
        assert p.observations == 2

        # Third obs: 75 min → observed=60. Ratio≈0.67, within bounds
        result = self._observe(p, 4500.0, start_time=14000.0)
        assert result is not None
        assert p.observations == 3

    def test_outlier_rejection_interacts_with_floor(self):
        """The 15-min floor + outlier rejection together prevent τ collapse."""
        p = self._make(lag=15.0)
        p.restore({"tau_fast": 50.0, "observations": 0})
        # Two observations at 65 min → observed=50
        self._observe(p, 3900.0)
        self._observe(p, 3900.0, start_time=4000.0)
        tau_stable = p.tau_fast
        assert tau_stable == pytest.approx(50.0, abs=1.0)

        # Room warms in 16 min → raw=1, floored=15. 15/50=0.3 < 1/3 → rejected
        p.start_observation(_ctx(start=8000.0))
        result = p.check_observation(8000.0 + 960.0, 21.27)
        assert result is None
        assert p.tau_fast == pytest.approx(tau_stable, abs=0.1)
