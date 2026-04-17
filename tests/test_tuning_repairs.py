"""Tests for HA Repairs tuning recommendations (Phase 2+)."""

import pytest

from custom_components.tasmota_irhvac.pi.health_checks import (
    check_high_integral_repair,
    check_save_seeds_repair,
    check_slope_divergence_repair,
)
from .conftest import get_climate_entity


# ── check_slope_divergence_repair ───────────────────────────────────


class TestSlopeDivergenceRepair:
    """Tests for slope divergence detection."""

    def test_no_issue_within_threshold(self):
        """No issue when drift is within 30%."""
        result = check_slope_divergence_repair(
            learned_slope=0.35, configured_slope=0.30,
            sustained_cycles=10, mode="heat",
        )
        # 17% drift < 30% threshold — should not create
        assert result is None or result[2] is False

    def test_creates_issue_above_threshold(self):
        """Issue created when drift > 30% and sustained."""
        result = check_slope_divergence_repair(
            learned_slope=0.42, configured_slope=0.30,
            sustained_cycles=6, mode="heat",
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert key == "slope_divergence"
        assert placeholders["mode"] == "heat"
        assert placeholders["learned"] == "0.4200"
        assert placeholders["configured"] == "0.3000"

    def test_not_created_before_sustained(self):
        """Issue not created if sustained cycles < 6."""
        result = check_slope_divergence_repair(
            learned_slope=0.50, configured_slope=0.30,
            sustained_cycles=3, mode="heat",
        )
        assert result is None

    def test_clears_below_hysteresis(self):
        """Issue cleared when drift drops below 15%."""
        result = check_slope_divergence_repair(
            learned_slope=0.32, configured_slope=0.30,
            sustained_cycles=10, mode="heat",
        )
        assert result is not None
        assert result[2] is False  # should_create = False (clear)

    def test_hysteresis_band(self):
        """No change when in hysteresis band (15-30%)."""
        # 23% drift — between 15% clear and 30% create
        result = check_slope_divergence_repair(
            learned_slope=0.37, configured_slope=0.30,
            sustained_cycles=10, mode="heat",
        )
        assert result is None

    def test_zero_configured_slope(self):
        """Returns None for zero configured slope (avoid division by zero)."""
        result = check_slope_divergence_repair(
            learned_slope=0.5, configured_slope=0.0,
            sustained_cycles=10, mode="heat",
        )
        assert result is None

    def test_abs_floor_prevents_false_positive(self):
        """Small absolute drift doesn't trigger even if percentage is high."""
        # 0.01 vs 0.02 = 50% drift, but absolute drift is only 0.01 < 0.05 floor
        result = check_slope_divergence_repair(
            learned_slope=0.02, configured_slope=0.01,
            sustained_cycles=10, mode="heat",
        )
        assert result is None

    def test_cool_mode(self):
        """Works for cool mode too."""
        result = check_slope_divergence_repair(
            learned_slope=0.42, configured_slope=0.30,
            sustained_cycles=6, mode="cool",
        )
        assert result is not None
        assert result[1]["mode"] == "cool"
        assert result[1]["mode_cap"] == "Cool"


# ── check_save_seeds_repair ─────────────────────────────────────────


class TestSaveSeedsRepair:
    """Tests for save seeds recommendation."""

    def test_creates_when_converged(self):
        """Issue created when integral_convergence < threshold."""
        result = check_save_seeds_repair(
            integral_convergence=1.5,
            seeds_match_learned=False,
            already_notified=False,
            outdoor_delta_heat=0.42,
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert key == "save_seeds"
        assert placeholders["outdoor_delta"] == "0.4200"

    def test_clears_when_seeds_match(self):
        """Issue cleared when seeds match learned values."""
        result = check_save_seeds_repair(
            integral_convergence=1.0,
            seeds_match_learned=True,
            already_notified=True,
        )
        assert result is not None
        assert result[2] is False

    def test_one_shot_no_re_fire(self):
        """Does not re-fire after already notified."""
        result = check_save_seeds_repair(
            integral_convergence=1.0,
            seeds_match_learned=False,
            already_notified=True,
        )
        assert result is None

    def test_no_issue_when_not_converged(self):
        """No issue when convergence is above threshold."""
        result = check_save_seeds_repair(
            integral_convergence=5.0,
            seeds_match_learned=False,
            already_notified=False,
        )
        assert result is None


# ── check_high_integral_repair ──────────────────────────────────────


class TestHighIntegralRepair:
    """Tests for high integral diagnosis."""

    def test_clears_below_threshold(self):
        """Issue cleared when correction drops below 1.0°C."""
        result = check_high_integral_repair(
            ki_integral_correction=0.5, sustained_cycles=10,
            observation_count=100, learned_slope=0.3,
            configured_slope=0.3, uncontrollable_cvh=0.0,
            total_cvh=1.0, pi_ki=0.15,
            integral_convergence=3.3, mode="heat",
        )
        assert result is not None
        assert result[2] is False

    def test_no_issue_in_hysteresis(self):
        """No change when between clear and create thresholds."""
        result = check_high_integral_repair(
            ki_integral_correction=1.5, sustained_cycles=10,
            observation_count=100, learned_slope=0.3,
            configured_slope=0.3, uncontrollable_cvh=0.0,
            total_cvh=1.0, pi_ki=0.15,
            integral_convergence=10.0, mode="heat",
        )
        assert result is None

    def test_immature_model(self):
        """Immature model (< 50 obs) gets informational message."""
        result = check_high_integral_repair(
            ki_integral_correction=2.5, sustained_cycles=6,
            observation_count=30, learned_slope=0.3,
            configured_slope=0.3, uncontrollable_cvh=0.0,
            total_cvh=1.0, pi_ki=0.15,
            integral_convergence=16.7, mode="heat",
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert key == "high_integral_immature"
        assert placeholders["count"] == "30"

    def test_slope_gap(self):
        """Slope mismatch diagnosed when gap > 20%."""
        result = check_high_integral_repair(
            ki_integral_correction=2.5, sustained_cycles=6,
            observation_count=100, learned_slope=0.42,
            configured_slope=0.30, uncontrollable_cvh=0.0,
            total_cvh=1.0, pi_ki=0.15,
            integral_convergence=16.7, mode="heat",
        )
        assert result is not None
        assert result[0] == "high_integral_slope_gap"
        assert result[1]["learned"] == "0.4200"
        assert result[1]["configured"] == "0.3000"

    def test_equipment_limits(self):
        """Equipment at limits when uncontrollable > 50%."""
        result = check_high_integral_repair(
            ki_integral_correction=2.5, sustained_cycles=6,
            observation_count=100, learned_slope=0.30,
            configured_slope=0.30, uncontrollable_cvh=3.0,
            total_cvh=5.0, pi_ki=0.15,
            integral_convergence=16.7, mode="heat",
        )
        assert result is not None
        assert result[0] == "high_integral_equipment"

    def test_tuning_suggestion(self):
        """Default case suggests specific Ki value."""
        result = check_high_integral_repair(
            ki_integral_correction=2.5, sustained_cycles=6,
            observation_count=100, learned_slope=0.30,
            configured_slope=0.30, uncontrollable_cvh=0.5,
            total_cvh=5.0, pi_ki=0.15,
            integral_convergence=16.7, mode="heat",
        )
        assert result is not None
        key, placeholders, should_create = result
        assert key == "high_integral_tuning"
        assert should_create is True
        assert "suggested_ki" in placeholders
        assert "current_ki" in placeholders
        # Suggested Ki should be lower than current
        assert float(placeholders["suggested_ki"]) < float(placeholders["current_ki"])

    def test_priority_order_immature_beats_slope(self):
        """Immature model takes priority over slope gap."""
        result = check_high_integral_repair(
            ki_integral_correction=2.5, sustained_cycles=6,
            observation_count=30, learned_slope=0.50,
            configured_slope=0.30, uncontrollable_cvh=0.0,
            total_cvh=1.0, pi_ki=0.15,
            integral_convergence=16.7, mode="heat",
        )
        assert result is not None
        assert result[0] == "high_integral_immature"

    def test_priority_order_slope_beats_equipment(self):
        """Slope gap takes priority over equipment limits."""
        result = check_high_integral_repair(
            ki_integral_correction=2.5, sustained_cycles=6,
            observation_count=100, learned_slope=0.50,
            configured_slope=0.30, uncontrollable_cvh=3.0,
            total_cvh=5.0, pi_ki=0.15,
            integral_convergence=16.7, mode="heat",
        )
        assert result is not None
        assert result[0] == "high_integral_slope_gap"

    def test_not_sustained_enough(self):
        """No issue if not sustained for 6 cycles."""
        result = check_high_integral_repair(
            ki_integral_correction=3.0, sustained_cycles=3,
            observation_count=100, learned_slope=0.30,
            configured_slope=0.30, uncontrollable_cvh=0.0,
            total_cvh=1.0, pi_ki=0.15,
            integral_convergence=20.0, mode="heat",
        )
        assert result is None


# ── Integration: _check_tuning_health orchestration ─────────────────


class TestCheckTuningHealthOrchestration:
    """Test that _check_tuning_health() wires checks correctly."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_pi_disabled(self, hass, setup_integration):
        """Non-PI entity returns no tuning issues."""
        entry = await setup_integration()
        entity = get_climate_entity(hass, entry)
        # Non-PI entity has NullController, not PIController
        # _check_tuning_health is only on PIController, so this tests
        # that __init__.py correctly skips non-PI entities
        from custom_components.tasmota_irhvac.pi.controller_protocol import NullController
        assert isinstance(entity._pi, NullController) or not hasattr(entity._pi, '_check_tuning_health')

    @pytest.mark.asyncio
    async def test_slope_divergence_detected(self, hass, setup_pi_integration):
        """Slope divergence is detected when learned slope differs from configured."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Simulate learned slope diverging from configured
        # Configured is 0.3 (default), set learned to 0.5 (67% drift)
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]
        pi._rls_heat.observation_count = 100
        # Sustained for 6 cycles
        pi._tuning_alert_counters["slope_div_heat"] = 5  # will be incremented to 6

        issues = pi._check_tuning_health()
        slope_issues = [i for i in issues if "slope_divergence" in i[0]]
        assert len(slope_issues) >= 1
        assert slope_issues[0][4] is True  # should_create

    @pytest.mark.asyncio
    async def test_save_seeds_detected(self, hass, setup_pi_integration):
        """Save seeds recommended when model converges."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Simulate converged model with seeds not matching
        pi._metrics.integral_convergence = 1.0
        pi._rls_heat.observation_count = 100
        # Make seeds not match learned
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]

        issues = pi._check_tuning_health()
        save_issues = [i for i in issues if "save_seeds" in i[0]]
        assert len(save_issues) == 1
        assert save_issues[0][4] is True

    @pytest.mark.asyncio
    async def test_high_integral_detected(self, hass, setup_pi_integration):
        """High integral is detected and diagnosed."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Simulate high integral
        pi._metrics.integral_convergence = 20.0  # ki * 20 = 0.15 * 20 = 3.0°C
        pi._rls_heat.observation_count = 100
        pi._tuning_alert_counters["high_integral"] = 5  # will be incremented to 6

        issues = pi._check_tuning_health()
        integral_issues = [i for i in issues if "high_integral" in i[0]]
        assert len(integral_issues) == 1
        assert integral_issues[0][4] is True

    @pytest.mark.asyncio
    async def test_signal_fires_on_batch(self, hass, setup_pi_integration):
        """SIGNAL_PI_BATCH_COMPLETE fires after batch analysis."""
        from unittest.mock import MagicMock
        from homeassistant.helpers.dispatcher import async_dispatcher_connect
        from custom_components.tasmota_irhvac.const import SIGNAL_PI_BATCH_COMPLETE

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        signal_received = MagicMock()
        async_dispatcher_connect(
            hass, SIGNAL_PI_BATCH_COMPLETE.format(entry.entry_id), signal_received,
        )

        # Add enough observations for batch to run
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        import time
        for i in range(25):
            pi._observation_buffer.add(Observation(
                timestamp=time.monotonic() + i,
                features=[1.0, float(i % 10)],
                hp_setpoint=22.0, current_c=21.0, desired_c=22.0,
                room_rate=0.005, clamped=False,
            ))

        pi._run_batch_analysis()
        await hass.async_block_till_done()

        assert signal_received.called
