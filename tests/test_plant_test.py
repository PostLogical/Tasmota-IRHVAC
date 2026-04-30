"""Unit tests for PlantTestProvider — relay test + step-hold state machine."""

import math

import pytest

from custom_components.tasmota_irhvac.pi.plant_model import PlantTestCommand
from custom_components.tasmota_irhvac.pi.providers.plant_test import PlantTestProvider


class TestPlantTestLifecycle:
    """Basic lifecycle tests."""

    def _make(self) -> PlantTestProvider:
        return PlantTestProvider()

    def test_start_activates(self):
        p = self._make()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.5,
                comfort_min_c=19.0, comfort_max_c=25.0, n_cycles=3)
        assert p.active
        assert p.phase == "relay_high"

    def test_not_active_before_start(self):
        p = self._make()
        assert not p.active
        assert p.phase == "idle"

    def test_abort(self):
        p = self._make()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.5,
                comfort_min_c=19.0, comfort_max_c=25.0)
        p.abort()
        assert not p.active
        assert p.phase == "aborted"

    def test_abort_when_not_active_is_safe(self):
        p = self._make()
        p.abort()  # Should not raise
        assert not p.active


class TestRelayPhase:
    """Tests for the relay cycling state machine."""

    def _make_and_start(self, amplitude=2, n_cycles=2, baseline=22, current=21.5):
        p = PlantTestProvider()
        p.start(baseline_setpoint_c=baseline, amplitude_c=amplitude,
                current_c=current, comfort_min_c=15.0, comfort_max_c=30.0,
                n_cycles=n_cycles)
        return p

    def test_relay_high_setpoint(self):
        """RELAY_HIGH phase sends baseline + amplitude."""
        p = self._make_and_start(baseline=22, amplitude=2)
        cmd = p.tick(100.0, 21.5)
        assert cmd.phase == "relay_high"
        assert cmd.setpoint_c == 24  # 22 + 2

    def test_midpoint_crossing_transitions_to_low(self):
        """Room crossing midpoint upward transitions to RELAY_LOW."""
        p = self._make_and_start(baseline=22, amplitude=2, current=21.0)
        # midpoint = 21.0 (starting temp)
        # First tick at t=100, room still below midpoint
        p.tick(100.0, 20.8)
        assert p.phase == "relay_high"

        # Room crosses midpoint after minimum phase duration (5 min = 300s)
        cmd = p.tick(500.0, 21.3)  # above midpoint 21.0
        assert cmd.phase == "relay_low"
        assert cmd.setpoint_c == 20  # 22 - 2

    def test_full_cycle_counting(self):
        """Complete relay cycles are counted correctly."""
        p = self._make_and_start(baseline=22, amplitude=2, current=21.0, n_cycles=2)
        t = 100.0

        # Cycle 1: high → cross up → low → cross down
        t += 400.0
        p.tick(t, 21.3)  # cross midpoint up → relay_low
        assert p.phase == "relay_low"
        t += 400.0
        p.tick(t, 20.7)  # cross midpoint down → cycle complete, relay_high
        assert p.phase == "relay_high"
        assert p.cycle_count == 1

        # Cycle 2: high → cross up → low → cross down → step_hold
        t += 400.0
        p.tick(t, 21.3)
        assert p.phase == "relay_low"
        t += 400.0
        cmd = p.tick(t, 20.7)
        assert cmd.phase == "step_hold"  # transitions after n_cycles
        assert p.cycle_count == 2

    def test_no_crossing_before_min_duration(self):
        """Crossings within minimum phase duration are ignored."""
        p = self._make_and_start(baseline=22, amplitude=2, current=21.0)
        # Room crosses midpoint at t=200s (< 5 min = 300s) → ignored
        cmd = p.tick(200.0, 21.5)
        assert cmd.phase == "relay_high"  # Not transitioned

    def test_relay_phase_timeout(self):
        """Phase timeout transitions to step_hold."""
        p = self._make_and_start(baseline=22, amplitude=2, current=21.0)
        # 181 minutes without crossing → timeout
        cmd = p.tick(100.0 + 181 * 60, 21.0)
        assert cmd.phase == "step_hold"


class TestSafetyBounds:
    """Tests for comfort bound enforcement."""

    def test_abort_on_room_too_cold(self):
        p = PlantTestProvider()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
                comfort_min_c=18.0, comfort_max_c=25.0)
        cmd = p.tick(100.0, 17.5)  # Below comfort_min
        assert cmd.phase == "aborted"
        assert not p.active

    def test_abort_on_room_too_hot(self):
        p = PlantTestProvider()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
                comfort_min_c=18.0, comfort_max_c=25.0)
        cmd = p.tick(100.0, 25.5)  # Above comfort_max
        assert cmd.phase == "aborted"

    def test_within_bounds_continues(self):
        p = PlantTestProvider()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
                comfort_min_c=18.0, comfort_max_c=25.0)
        cmd = p.tick(100.0, 21.0)
        assert cmd.phase == "relay_high"  # Continues normally


class TestStepHoldPhase:
    """Tests for the step-hold phase."""

    def _run_to_step_hold(self):
        """Helper: run relay through 2 cycles to reach step_hold."""
        p = PlantTestProvider()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
                comfort_min_c=15.0, comfort_max_c=30.0, n_cycles=2)
        t = 100.0
        for _ in range(2):
            t += 400.0
            p.tick(t, 21.3)  # cross up
            t += 400.0
            p.tick(t, 20.7)  # cross down
        assert p.phase == "step_hold"
        return p, t

    def test_step_hold_setpoint(self):
        """Step-hold phase sends baseline + amplitude."""
        p, t = self._run_to_step_hold()
        cmd = p.tick(t + 100, 21.0)
        assert cmd.setpoint_c == 24  # 22 + 2
        assert cmd.phase == "step_hold"

    def test_step_hold_creates_observation_context(self):
        """Step-hold phase creates ObservationContext for passive providers."""
        p, t = self._run_to_step_hold()
        assert p._step_hold_ctx is not None

    def test_step_hold_timeout_completes(self):
        """Step-hold times out and completes the test."""
        p, t = self._run_to_step_hold()
        # Max step-hold = 240 min = 14400s
        cmd = p.tick(t + 14500, 22.0)
        assert cmd.phase == "complete"
        assert not p.active


class TestResults:
    """Tests for relay measurement results."""

    def _run_complete_test(self, n_cycles=3, period_min=60.0, amplitude_c=0.5):
        """Run a complete relay test with simulated oscillation."""
        p = PlantTestProvider()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
                comfort_min_c=15.0, comfort_max_c=30.0, n_cycles=n_cycles)

        t = 100.0
        half_period_s = period_min * 60.0 / 2.0

        for cycle in range(n_cycles):
            # Relay high: room warms past midpoint
            t += half_period_s
            peak = 21.0 + amplitude_c
            p.tick(t, peak)  # cross midpoint upward

            # Relay low: room cools past midpoint
            t += half_period_s
            trough = 21.0 - amplitude_c
            p.tick(t, trough)  # cross midpoint downward

        # Should be in step_hold now
        assert p.phase == "step_hold"

        # Complete via timeout
        cmd = p.tick(t + 15000, 21.5)
        assert cmd.phase == "complete"

        return p

    def test_results_after_completion(self):
        """Results contain K_u, period, and amplitude."""
        p = self._run_complete_test(n_cycles=3, period_min=60.0, amplitude_c=0.5)
        results = p.get_results()
        assert results is not None
        assert "k_u" in results
        assert "period" in results
        assert "amplitude" in results

    def test_period_measurement(self):
        """T_u is computed correctly from crossing intervals."""
        p = self._run_complete_test(n_cycles=3, period_min=60.0, amplitude_c=0.5)
        results = p.get_results()
        assert results is not None
        assert results["period"].value == pytest.approx(60.0, rel=0.1)

    def test_amplitude_measurement(self):
        """Oscillation amplitude is computed from peaks and troughs."""
        p = self._run_complete_test(n_cycles=3, period_min=60.0, amplitude_c=0.5)
        results = p.get_results()
        assert results is not None
        assert results["amplitude"].value == pytest.approx(0.5, rel=0.1)

    def test_k_u_formula(self):
        """K_u = 4h/(πa) computed correctly."""
        p = self._run_complete_test(n_cycles=3, period_min=60.0, amplitude_c=0.5)
        results = p.get_results()
        assert results is not None
        h = 2.0  # relay amplitude (°C)
        a = 0.5  # room oscillation amplitude
        expected_k_u = 4.0 * h / (math.pi * a)
        assert results["k_u"].value == pytest.approx(expected_k_u, rel=0.1)

    def test_no_results_before_completion(self):
        p = PlantTestProvider()
        p.start(baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
                comfort_min_c=15.0, comfort_max_c=30.0)
        assert p.get_results() is None

    def test_results_source_is_plant_test(self):
        p = self._run_complete_test()
        results = p.get_results()
        assert results is not None
        assert results["k_u"].source == "plant_test"


class TestPlantTestOrchestration:
    """Integration with PlantIdentifier."""

    def test_start_and_tick(self):
        from custom_components.tasmota_irhvac.pi.plant_identifier import PlantIdentifier

        pi = PlantIdentifier(
            tau_fast_seed=60.0, tau_slow_seed=60.0,
            response_lag=15.0, imc_lambda=0.0, enabled=True,
        )
        pi.start_plant_test(
            baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
            comfort_min_c=15.0, comfort_max_c=30.0, n_cycles=2,
        )
        assert pi.plant_test_active

        # First tick
        cmd = pi.tick_plant_test(100.0, 21.0)
        assert cmd.phase == "relay_high"
        assert cmd.setpoint_c == 24

    def test_abort(self):
        from custom_components.tasmota_irhvac.pi.plant_identifier import PlantIdentifier

        pi = PlantIdentifier(
            tau_fast_seed=60.0, tau_slow_seed=60.0,
            response_lag=15.0, imc_lambda=0.0, enabled=True,
        )
        pi.start_plant_test(
            baseline_setpoint_c=22, amplitude_c=2, current_c=21.0,
            comfort_min_c=15.0, comfort_max_c=30.0,
        )
        assert pi.plant_test_active
        pi.abort_plant_test()
        assert not pi.plant_test_active
