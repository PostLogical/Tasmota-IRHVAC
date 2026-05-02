"""Tests for HA Repairs tuning recommendations (Phase 2+)."""

import pytest
from homeassistant.components.climate import HVACMode

from custom_components.tasmota_irhvac.pi.health_checks import (
    check_freeze_impact_repair,
    check_high_integral_repair,
    check_intercept_absorbing_repair,
    check_model_drift_repair,
    check_multicollinearity_repair,
    check_residual_pattern_repair,
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

        issues = pi._check_tuning_health(from_batch=True)
        slope_issues = [i for i in issues if "slope_divergence" in i[0]]
        assert len(slope_issues) >= 1
        assert slope_issues[0][4] is True  # should_create

    @pytest.mark.asyncio
    async def test_startup_does_not_inflate_counters(self, hass, setup_pi_integration):
        """Startup health checks must not increment sustained-cycle counters."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Simulate conditions that would increment slope_div counter
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]
        pi._rls_heat.observation_count = 100

        # Call without from_batch (startup path) multiple times
        for _ in range(10):
            pi._check_tuning_health()

        # Counter should not have been incremented
        assert pi._tuning_alert_counters.get("slope_div_heat", 0) == 0

        # Now call with from_batch=True — counter should increment
        pi._check_tuning_health(from_batch=True)
        assert pi._tuning_alert_counters.get("slope_div_heat", 0) == 1

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

        issues = pi._check_tuning_health(from_batch=True)
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
        entity._attr_hvac_mode = HVACMode.HEAT

        signal_received = MagicMock()
        async_dispatcher_connect(
            hass, SIGNAL_PI_BATCH_COMPLETE.format(entry.entry_id), signal_received,
        )

        # Add enough observations for batch to run
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation
        import time
        for i in range(25):
            pi._observation_buffer_heat.add(Observation(
                timestamp=time.monotonic() + i,
                wall_time=1713650000.0 + i,
                hp_setpoint=22.0, current_c=21.0, desired_c=22.0,
                outdoor_temp_c=21.0 + float(i % 10),
                room_rate=0.005, raw_readings={}, clamped=False,
            ))

        pi._run_batch_analysis()
        await hass.async_block_till_done()

        assert signal_received.called


# ── check_model_drift_repair ───────────────────────────────────────


class TestModelDriftRepair:
    """Tests for model drift with maturity gate."""

    def test_suppressed_before_stable_batch(self):
        """No drift alerts before system has had a stable batch cycle."""
        result = check_model_drift_repair(
            drifting_coefficients=[(1, "outdoor_delta", 7)],
            has_had_stable_batch=False,
        )
        assert result == []

    def test_creates_after_stable_batch(self):
        """Drift alerts fire after system has stabilized once."""
        result = check_model_drift_repair(
            drifting_coefficients=[(1, "outdoor_delta", 7)],
            has_had_stable_batch=True,
        )
        assert len(result) == 1
        key, placeholders, should_create = result[0]
        assert should_create is True
        assert key == "model_drift"
        assert placeholders["coeff_name"] == "outdoor_delta"
        assert "insulation" in placeholders["suggestion"].lower()

    def test_intercept_drift_suggestion(self):
        """Intercept drift gets sensor calibration suggestion."""
        result = check_model_drift_repair(
            drifting_coefficients=[(0, "intercept", 5)],
            has_had_stable_batch=True,
        )
        assert len(result) == 1
        assert "sensor calibration" in result[0][1]["suggestion"].lower()

    def test_model_input_drift_suggestion(self):
        """Model input drift gets input-specific suggestion."""
        result = check_model_drift_repair(
            drifting_coefficients=[(2, "Pellet Stove", 6)],
            has_had_stable_batch=True,
        )
        assert len(result) == 1
        assert "Pellet Stove" in result[0][1]["suggestion"]

    def test_below_threshold_skipped(self):
        """Coefficients below min_consecutive threshold are skipped."""
        result = check_model_drift_repair(
            drifting_coefficients=[(1, "outdoor_delta", 3)],
            has_had_stable_batch=True,
        )
        assert result == []

    def test_multiple_drifting(self):
        """Multiple drifting coefficients produce multiple results."""
        result = check_model_drift_repair(
            drifting_coefficients=[
                (0, "intercept", 6),
                (1, "outdoor_delta", 8),
            ],
            has_had_stable_batch=True,
        )
        assert len(result) == 2


# ── check_intercept_absorbing_repair ────────────────────────────────


class TestInterceptAbsorbingRepair:
    """Tests for intercept absorbing coefficient detection."""

    def test_creates_when_intercept_large_and_coeff_collapsed(self):
        """Issue created when intercept is large and a coefficient has collapsed."""
        result = check_intercept_absorbing_repair(
            intercept_value=-1.5,
            coefficients=[
                ("outdoor_delta", 0.0, (0.0, 2.0), 0.001),
            ],
            delta=0.001,
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert key == "intercept_absorbing"
        assert placeholders["absorbed_name"] == "outdoor_delta"
        assert placeholders["intercept_value"] == "-1.50"

    def test_clears_when_intercept_small(self):
        """Issue cleared when intercept drops below threshold."""
        result = check_intercept_absorbing_repair(
            intercept_value=0.3,
            coefficients=[
                ("outdoor_delta", 0.0, (0.0, 2.0), 0.001),
            ],
        )
        assert result is not None
        assert result[2] is False

    def test_no_issue_when_no_collapsed_coeff(self):
        """No issue when intercept is large but no coefficient has collapsed."""
        result = check_intercept_absorbing_repair(
            intercept_value=-2.0,
            coefficients=[
                ("outdoor_delta", 0.3, (0.0, 2.0), 0.5),
            ],
        )
        assert result is None

    def test_no_issue_when_coeff_at_clamp_but_p_healthy(self):
        """No issue when coefficient at clamp but P hasn't collapsed."""
        result = check_intercept_absorbing_repair(
            intercept_value=-1.5,
            coefficients=[
                ("outdoor_delta", 0.0, (0.0, 2.0), 0.5),
            ],
        )
        assert result is None


# ── Integration: Phase 3 orchestration ──────────────────────────────


class TestPhase3Orchestration:
    """Test Phase 3 checks wired into _check_tuning_health()."""

    @pytest.mark.asyncio
    async def test_drift_suppressed_before_stable(self, hass, setup_pi_integration):
        """Model drift is suppressed before first stable batch."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._has_had_stable_batch = False
        # Simulate drifting coefficients
        pi._drift_correction_signs = [[1, 1, 1, 1, 1, 1]]

        issues = pi._check_tuning_health()
        drift_issues = [i for i in issues if "model_drift" in i[0]]
        assert len(drift_issues) == 0

    @pytest.mark.asyncio
    async def test_intercept_absorbing_detected(self, hass, setup_pi_integration):
        """Intercept absorbing is detected when conditions match."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        n = pi._rls_heat.n
        # Large intercept
        pi._rls_heat.beta[0] = -1.5 * pi._rls_heat.feature_scales[0]
        # outdoor_delta at lower clamp with collapsed P
        pi._rls_heat.beta[1] = 0.0
        pi._rls_heat.P[1 * n + 1] = 0.001
        pi._rls_heat.observation_count = 100

        issues = pi._check_tuning_health()
        absorbing_issues = [i for i in issues if "intercept_absorbing" in i[0]]
        assert len(absorbing_issues) >= 1
        assert absorbing_issues[0][4] is True


# ── check_multicollinearity_repair ─────────────────────────────────


class TestMulticollinearityRepair:
    """Tests for multicollinearity / condition number detection."""

    def test_no_issue_low_condition_number(self):
        """No issue when κ < 20 (Belsley: weak dependencies)."""
        result = check_multicollinearity_repair(
            condition_number=10.0,
            correlated_pairs=[],
            sustained_cycles=10,
        )
        assert result is not None
        assert result[2] is False  # should clear

    def test_creates_issue_above_threshold(self):
        """Issue created when κ > 30 (Belsley: moderate) and sustained."""
        result = check_multicollinearity_repair(
            condition_number=50.0,
            correlated_pairs=[("outdoor_delta", "solar", 0.92)],
            sustained_cycles=3,
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert key == "multicollinearity"
        assert placeholders["condition_number"] == "50"
        assert "outdoor_delta" in placeholders["pairs"]
        assert "solar" in placeholders["pairs"]

    def test_hysteresis_band(self):
        """No change when κ in hysteresis band (20-30)."""
        result = check_multicollinearity_repair(
            condition_number=25.0,
            correlated_pairs=[],
            sustained_cycles=10,
        )
        assert result is None

    def test_not_created_before_sustained(self):
        """Issue not created before min sustained cycles."""
        result = check_multicollinearity_repair(
            condition_number=50.0,
            correlated_pairs=[("a", "b", 0.9)],
            sustained_cycles=1,
        )
        assert result is None

    def test_clears_below_threshold(self):
        """Issue cleared when κ drops below 20 (Belsley: weak)."""
        result = check_multicollinearity_repair(
            condition_number=15.0,
            correlated_pairs=[],
            sustained_cycles=0,
        )
        assert result is not None
        assert result[2] is False

    def test_no_pairs_message(self):
        """Graceful message when κ is high but no single pair dominates."""
        result = check_multicollinearity_repair(
            condition_number=50.0,
            correlated_pairs=[],
            sustained_cycles=5,
        )
        assert result is not None
        assert result[2] is True
        assert "distributed across inputs" in result[1]["pairs"]

    def test_distributed_shows_top_pair(self):
        """When top pair is below 0.7, show it as distributed with context."""
        result = check_multicollinearity_repair(
            condition_number=50.0,
            correlated_pairs=[("outdoor_delta", "solar", 0.55)],
            sustained_cycles=5,
        )
        assert result is not None
        assert "distributed" in result[1]["pairs"]
        assert "outdoor_delta and solar" in result[1]["pairs"]

    def test_multiple_pairs_truncated(self):
        """At most 3 correlated pairs shown."""
        pairs = [
            ("a", "b", 0.9), ("c", "d", 0.85),
            ("e", "f", 0.8), ("g", "h", 0.75),
        ]
        result = check_multicollinearity_repair(
            condition_number=100.0,
            correlated_pairs=pairs,
            sustained_cycles=3,
        )
        assert result is not None
        # Should show 3 pairs, not 4
        assert result[1]["pairs"].count("r=") == 3


# ── check_residual_pattern_repair ──────────────────────────────────


class TestResidualPatternRepair:
    """Tests for time-of-day residual pattern detection."""

    def test_no_issue_small_residual(self):
        """No issue when residual is below threshold."""
        result = check_residual_pattern_repair(
            start_hour=14, end_hour=16,
            mean_residual=-0.2, n_observations=50,
            sustained_cycles=10,
        )
        assert result is not None
        assert result[2] is False  # should clear

    def test_creates_issue_negative_residual(self):
        """Issue created for negative residual (heat gain)."""
        result = check_residual_pattern_repair(
            start_hour=13, end_hour=16,
            mean_residual=-0.8, n_observations=40,
            sustained_cycles=3,
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert key == "residual_pattern"
        assert placeholders["time_range"] == "13:00–16:59"
        assert placeholders["direction"] == "negative"
        assert "solar" in placeholders["cause_hint"]

    def test_creates_issue_positive_residual(self):
        """Issue created for positive residual (heat loss)."""
        result = check_residual_pattern_repair(
            start_hour=22, end_hour=2,
            mean_residual=0.7, n_observations=35,
            sustained_cycles=3,
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert placeholders["direction"] == "positive"
        assert "heat loss" in placeholders["cause_hint"]

    def test_hysteresis_band(self):
        """No change when in hysteresis band (0.3-0.5)."""
        result = check_residual_pattern_repair(
            start_hour=14, end_hour=16,
            mean_residual=-0.4, n_observations=50,
            sustained_cycles=10,
        )
        assert result is None

    def test_not_created_before_sustained(self):
        """Issue not created before min sustained cycles."""
        result = check_residual_pattern_repair(
            start_hour=14, end_hour=16,
            mean_residual=-0.8, n_observations=50,
            sustained_cycles=1,
        )
        assert result is None

    def test_single_hour_time_range(self):
        """Single-hour pattern formats correctly."""
        result = check_residual_pattern_repair(
            start_hour=14, end_hour=14,
            mean_residual=-0.8, n_observations=30,
            sustained_cycles=3,
        )
        assert result is not None
        assert result[1]["time_range"] == "14:00–14:59"


# ── check_freeze_impact_repair ────────────────────────────────────


class TestFreezeImpactRepair:
    """Tests for frozen coefficient degradation detection."""

    def test_no_issue_rms_stable(self):
        """No issue when RMS hasn't increased."""
        result = check_freeze_impact_repair(
            coeff_name="Solar Proxy", mode="heat",
            rms_at_freeze=1.5, current_rms=1.5,
            sustained_cycles=10,
        )
        assert result is not None
        assert result[2] is False  # should clear

    def test_creates_issue_rms_increased(self):
        """Issue created when RMS increased >20% and sustained."""
        result = check_freeze_impact_repair(
            coeff_name="Solar Proxy", mode="heat",
            rms_at_freeze=1.5, current_rms=2.0,
            sustained_cycles=3,
        )
        assert result is not None
        key, placeholders, should_create = result
        assert should_create is True
        assert key == "freeze_impact"
        assert placeholders["coeff_name"] == "Solar Proxy"
        assert placeholders["mode"] == "heat"
        assert placeholders["rms_at_freeze"] == "1.500"
        assert placeholders["current_rms"] == "2.000"
        assert placeholders["increase_pct"] == "33"

    def test_hysteresis_band(self):
        """No change when increase is between 5-20%."""
        result = check_freeze_impact_repair(
            coeff_name="Solar Proxy", mode="heat",
            rms_at_freeze=1.5, current_rms=1.65,  # 10% increase
            sustained_cycles=10,
        )
        assert result is None

    def test_not_created_before_sustained(self):
        """Issue not created before min sustained cycles."""
        result = check_freeze_impact_repair(
            coeff_name="Solar Proxy", mode="heat",
            rms_at_freeze=1.5, current_rms=2.5,
            sustained_cycles=1,
        )
        assert result is None

    def test_clears_when_rms_drops(self):
        """Issue cleared when RMS increase drops below 5%."""
        result = check_freeze_impact_repair(
            coeff_name="outdoor_delta", mode="cool",
            rms_at_freeze=1.5, current_rms=1.55,  # 3.3% increase
            sustained_cycles=5,
        )
        assert result is not None
        assert result[2] is False

    def test_zero_rms_at_freeze_returns_none(self):
        """Returns None when snapshot RMS is zero (avoid division)."""
        result = check_freeze_impact_repair(
            coeff_name="Solar Proxy", mode="heat",
            rms_at_freeze=0.0, current_rms=1.5,
            sustained_cycles=5,
        )
        assert result is None

    def test_rms_decreased_clears(self):
        """Clears issue when current RMS is lower than at freeze time."""
        result = check_freeze_impact_repair(
            coeff_name="Solar Proxy", mode="heat",
            rms_at_freeze=2.0, current_rms=1.5,
            sustained_cycles=5,
        )
        assert result is not None
        assert result[2] is False


class TestFreezeImpactOrchestration:
    """Test freeze impact wired into _check_tuning_health()."""

    @pytest.mark.asyncio
    async def test_freeze_impact_detected(self, hass, setup_pi_integration):
        """Freeze impact is detected when frozen coeff degrades model fit."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Freeze outdoor_delta with RMS snapshot
        pi._metrics.batch_model_rms = 1.0
        pi.set_frozen("heat", 1, True)
        assert pi._tuning_alert_snapshots.get("freeze_rms_heat_1") == 1.0

        # Simulate RMS increasing over batch cycles
        pi._metrics.batch_model_rms = 1.5  # 50% increase
        pi._rls_heat.observation_count = 100

        # Run 3 batch cycles to build sustained counter
        for _ in range(3):
            pi._check_tuning_health(from_batch=True)

        issues = pi._check_tuning_health(from_batch=True)
        freeze_issues = [i for i in issues if "freeze_impact" in i[0]]
        assert len(freeze_issues) >= 1
        assert freeze_issues[0][4] is True

    @pytest.mark.asyncio
    async def test_freeze_impact_clears_on_unfreeze(self, hass, setup_pi_integration):
        """Freeze impact counters are cleared when coefficient is unfrozen."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._metrics.batch_model_rms = 1.0
        pi.set_frozen("heat", 1, True)
        pi._tuning_alert_counters["freeze_impact_heat_1"] = 5

        pi.set_frozen("heat", 1, False)
        assert "freeze_rms_heat_1" not in pi._tuning_alert_snapshots
        assert "freeze_impact_heat_1" not in pi._tuning_alert_counters

    @pytest.mark.asyncio
    async def test_no_freeze_impact_when_rms_stable(self, hass, setup_pi_integration):
        """No freeze impact issue when RMS is stable after freeze."""
        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._metrics.batch_model_rms = 1.0
        pi.set_frozen("heat", 1, True)
        pi._rls_heat.observation_count = 100

        # RMS stays the same
        issues = pi._check_tuning_health()
        freeze_issues = [i for i in issues if "freeze_impact" in i[0]]
        # Either no issue or should_create=False (clearing)
        assert all(not i[4] for i in freeze_issues)
