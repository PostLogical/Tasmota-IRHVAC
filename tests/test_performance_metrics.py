"""Unit tests for PerformanceMetrics — no PIController or HA dependencies."""

import pytest

from custom_components.tasmota_irhvac.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Tests for performance metric accumulation."""

    def test_initial_state(self):
        m = PerformanceMetrics()
        assert m.integral_convergence == 0.0
        assert m.itae_accumulator == 0.0
        assert m.itae_tick_count == 0
        assert m.comfort_violation_hours == 0.0
        assert m.setpoint_changes == 0
        assert m.ff_load_fraction == 0.5
        assert m.batch_model_rms is None

    def test_accumulate_convergence(self):
        """EMA of abs(integral) over ~24hr time constant."""
        m = PerformanceMetrics()
        m.integral_convergence = 0.0
        m.accumulate_convergence(pi_integral=30.0)
        # alpha=0.01: 0.99*0 + 0.01*30 = 0.3
        assert m.integral_convergence == pytest.approx(0.3)

    def test_accumulate_convergence_decay(self):
        """Convergence decays toward zero when integral is small."""
        m = PerformanceMetrics()
        m.integral_convergence = 20.0
        m.accumulate_convergence(pi_integral=0.0)
        assert m.integral_convergence < 20.0

    def test_accumulate_tick_itae(self):
        """ITAE accumulates tick_count * effective_error."""
        m = PerformanceMetrics()
        m.accumulate_tick(
            abs_error=1.5, dt_seconds=900, pi_deadband=0.5,
            is_heating=True, is_cooling=False, error=1.5,
            hp_setpoint=22.0, min_temp_c=16.0, max_temp_c=30.0,
        )
        assert m.itae_tick_count == 1
        # effective_error = max(0, 1.5 - 0.5) = 1.0
        # itae_increment = 1 * 1.0 = 1.0
        assert m.itae_accumulator == pytest.approx(1.0)
        assert m.controllable_itae == pytest.approx(1.0)
        assert m.comfort_violation_hours == pytest.approx(900 / 3600)

    def test_accumulate_tick_uncontrollable(self):
        """Saturated wrong-end → uncontrollable bucket."""
        m = PerformanceMetrics()
        # Heating, error < 0 (room above target), setpoint at min → uncontrollable
        m.accumulate_tick(
            abs_error=1.5, dt_seconds=900, pi_deadband=0.5,
            is_heating=True, is_cooling=False, error=-1.5,
            hp_setpoint=16.0, min_temp_c=16.0, max_temp_c=30.0,
        )
        assert m.uncontrollable_itae > 0
        assert m.controllable_itae == 0.0
        assert m.uncontrollable_cvh > 0
        assert m.controllable_cvh == 0.0

    def test_accumulate_tick_controllable(self):
        """HP has headroom → controllable bucket."""
        m = PerformanceMetrics()
        # Heating, error > 0, setpoint NOT at min → controllable
        m.accumulate_tick(
            abs_error=1.5, dt_seconds=900, pi_deadband=0.5,
            is_heating=True, is_cooling=False, error=1.5,
            hp_setpoint=22.0, min_temp_c=16.0, max_temp_c=30.0,
        )
        assert m.controllable_itae > 0
        assert m.uncontrollable_itae == 0.0

    def test_ff_load_fraction_trends_up(self):
        """When FF dominates integral, load fraction trends toward 1."""
        m = PerformanceMetrics()
        m.ff_load_fraction = 0.3
        for _ in range(100):
            m.accumulate_ff_load(ki_integral=0.1, ff_offset=5.0)
        assert m.ff_load_fraction > 0.7

    def test_ff_load_fraction_skips_near_zero(self):
        """Both near zero → no update (avoids noise)."""
        m = PerformanceMetrics()
        m.ff_load_fraction = 0.5
        m.accumulate_ff_load(ki_integral=0.01, ff_offset=0.01)
        assert m.ff_load_fraction == 0.5  # Unchanged (total < 0.1)

    def test_record_setpoint_change(self):
        m = PerformanceMetrics()
        m.record_setpoint_change()
        m.record_setpoint_change()
        assert m.setpoint_changes == 2

    def test_persistence_round_trip(self):
        m = PerformanceMetrics()
        m.itae_accumulator = 100.0
        m.comfort_violation_hours = 2.5
        m.ff_load_fraction = 0.8
        m.batch_model_rms = 0.42

        d = m.as_dict()
        m2 = PerformanceMetrics()
        m2.restore(d)

        assert m2.itae_accumulator == 100.0
        assert m2.comfort_violation_hours == 2.5
        assert m2.ff_load_fraction == 0.8
        assert m2.batch_model_rms == 0.42
