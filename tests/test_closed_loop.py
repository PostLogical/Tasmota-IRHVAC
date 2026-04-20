"""Unit tests for ClosedLoopProvider — SOPDT fit from closed-loop data."""

import math

import pytest

from custom_components.tasmota_irhvac.pi.plant_model import ObservationContext, ParameterEstimate
from custom_components.tasmota_irhvac.pi.providers.closed_loop import (
    ClosedLoopProvider,
    _sopdt_step,
)


def _ctx(start=0.0, temp=20.0, target=22.0, step=2.0, ff=0.0):
    return ObservationContext(
        start_time=start, baseline_temp=temp, target_temp=target,
        step_magnitude=step, ff_offset=ff,
    )


def _simulate_sopdt(
    t_min: float,
    tau_fast: float,
    tau_slow: float,
    k: float,
    setpoint_history: list[tuple[float, float]],
    initial_temp: float,
    response_lag: float = 0.0,
) -> float:
    """Simulate SOPDT plant output at time t given setpoint history.

    Superposition: each setpoint change contributes a step response.
    """
    y = initial_temp
    prev_sp = setpoint_history[0][1] if setpoint_history else initial_temp
    for t_change, sp in setpoint_history:
        delta_u = sp - prev_sp
        prev_sp = sp
        if delta_u == 0.0:
            continue
        dt = t_min - t_change - response_lag
        if dt > 0:
            y += k * delta_u * _sopdt_step(dt, tau_fast, tau_slow)
    return y


class TestSOPDTStep:
    """Verify the unit step response function."""

    def test_zero_at_t_zero(self):
        assert _sopdt_step(0.0, 15.0, 90.0) == 0.0

    def test_approaches_one(self):
        """Response approaches 1.0 at large t."""
        assert _sopdt_step(1000.0, 15.0, 90.0) == pytest.approx(1.0, abs=0.001)

    def test_monotonic_increase(self):
        """Response increases monotonically."""
        prev = 0.0
        for t in range(1, 300):
            y = _sopdt_step(float(t), 15.0, 90.0)
            assert y >= prev - 1e-10
            prev = y


class TestClosedLoopBasic:
    """Basic lifecycle tests."""

    def _make(self, lag=0.0):
        return ClosedLoopProvider(response_lag=lag)

    def test_start_observation(self):
        p = self._make()
        p.start_observation(_ctx())
        assert p.active

    def test_start_ignores_small_step(self):
        p = self._make()
        p.start_observation(_ctx(step=0.5))
        assert not p.active

    def test_cancel(self):
        p = self._make()
        p.start_observation(_ctx())
        p.cancel_observation()
        assert not p.active

    def test_accumulate_needs_hp_setpoint(self):
        """Returns None if hp_setpoint_c is not provided."""
        p = self._make()
        p.start_observation(_ctx())
        result = p.accumulate(100.0, 20.5, hp_setpoint_c=None)
        assert result is None


class TestClosedLoopFit:
    """Test SOPDT parameter recovery from simulated closed-loop data."""

    def _run_with_constant_setpoint(
        self, tau_fast, tau_slow, k=1.0, step=2.0, lag=0.0,
        tick_min=15.0, n_ticks=30,
    ):
        """Simulate constant-setpoint response and feed to provider."""
        p = ClosedLoopProvider(response_lag=lag)
        p.start_observation(_ctx(step=step))

        setpoint = 20.0 + step  # constant after step
        sp_history = [(0.0, 20.0), (0.0, setpoint)]

        result = None
        for i in range(1, n_ticks + 1):
            t_min = i * tick_min
            t_sec = t_min * 60.0
            temp = _simulate_sopdt(t_min, tau_fast, tau_slow, k, sp_history, 20.0, lag)
            result = p.accumulate(t_sec, temp, hp_setpoint_c=setpoint)
            if result is not None:
                break
        return result

    def test_recovers_tau_fast(self):
        """Closed-loop fit recovers τ_fast from constant-setpoint data."""
        result = self._run_with_constant_setpoint(
            tau_fast=15.0, tau_slow=90.0, k=1.0, n_ticks=30,
        )
        assert result is not None
        assert len(result) >= 2
        tau_fast_est = result[0]
        # Grid search has discrete values — accept nearest grid point
        assert tau_fast_est.value == pytest.approx(15.0, abs=10.0)
        assert tau_fast_est.source == "closed_loop"

    def test_recovers_tau_slow(self):
        """Closed-loop fit recovers τ_slow from constant-setpoint data."""
        result = self._run_with_constant_setpoint(
            tau_fast=15.0, tau_slow=90.0, k=1.0, n_ticks=30,
        )
        assert result is not None
        tau_slow_est = result[1]
        assert tau_slow_est.value == pytest.approx(90.0, abs=30.0)

    def test_with_varying_setpoint(self):
        """Closed-loop fit works when HP setpoint changes during observation.

        Simulates a PI controller that adjusts setpoint during the response.
        """
        p = ClosedLoopProvider(response_lag=0.0)
        p.start_observation(_ctx(step=2.0))

        # Simulate: HP starts at 22, drops to 21 at tick 5, back to 22 at tick 10
        tau_fast = 15.0
        tau_slow = 90.0
        k = 1.0
        tick_min = 15.0

        sp_schedule = {0: 22.0, 5: 21.0, 10: 22.0}
        sp_history: list[tuple[float, float]] = [(0.0, 20.0), (0.0, 22.0)]
        current_sp = 22.0

        result = None
        for i in range(1, 35):
            if i in sp_schedule:
                current_sp = sp_schedule[i]
                sp_history.append((i * tick_min, current_sp))

            t_min = i * tick_min
            temp = _simulate_sopdt(t_min, tau_fast, tau_slow, k, sp_history, 20.0)

            result = p.accumulate(t_min * 60.0, temp, hp_setpoint_c=current_sp)
            if result is not None:
                break

        assert result is not None
        # Should still recover reasonable τ values despite setpoint changes
        assert result[0].value == pytest.approx(15.0, abs=15.0)
        assert result[1].value == pytest.approx(90.0, abs=45.0)

    def test_rejects_poor_fit(self):
        """Rejects fit when R² is too low (random/noisy data)."""
        p = ClosedLoopProvider(response_lag=0.0)
        p.start_observation(_ctx(step=2.0))

        # Feed random-ish data that doesn't match any SOPDT model
        import random
        random.seed(42)
        for i in range(1, 20):
            t = i * 15.0 * 60.0
            temp = 20.0 + random.uniform(-1, 1)
            result = p.accumulate(t, temp, hp_setpoint_c=22.0)
            if result is not None:
                break

        # Should either return None (poor fit) or not fire at all
        # (settling criterion not met with random data)
        assert result is None


class TestClosedLoopOrchestration:
    """Integration with PlantIdentifier."""

    def test_closed_loop_cross_check_stored(self):
        """Closed-loop results are stored as cross-check, not applied to plant."""
        from custom_components.tasmota_irhvac.pi.plant_identifier import PlantIdentifier

        pi = PlantIdentifier(tau_seed=60.0, response_lag=0.0, imc_lambda=0.0)
        assert pi.plant.tau_fast.source == "seed"

        # Start observation
        pi.start_observation(0.0, 20.0, 22.0, 2.0)

        # Feed SOPDT response with HP setpoint
        tau_fast = 15.0
        tau_slow = 90.0
        sp_history = [(0.0, 20.0), (0.0, 22.0)]

        for i in range(1, 35):
            t_min = i * 15.0
            temp = _simulate_sopdt(t_min, tau_fast, tau_slow, 1.0, sp_history, 20.0)
            pi.check_observation(t_min * 60.0, temp, hp_setpoint_c=22.0)

        # Closed-loop is a cross-check — stored but NOT applied to plant
        # (tau_slow should still be seed unless area method fired)
        if pi._last_cross_check is not None:
            cl_tau_fast, cl_tau_slow = pi._last_cross_check
            assert cl_tau_fast.source == "closed_loop"
            assert cl_tau_fast.value == pytest.approx(15.0, abs=10.0)
        # Plant tau_slow unchanged (still seed) — closed-loop doesn't override
        assert pi.plant.tau_slow.source in ("seed", "area_method")
