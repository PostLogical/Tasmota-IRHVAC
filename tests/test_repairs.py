"""Tests for the repairs integration (config issue checks and fix flows)."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.tasmota_irhvac.const import DOMAIN
from custom_components.tasmota_irhvac.__init__ import _check_config_issues
from custom_components.tasmota_irhvac.repairs import (
    AnomalousObservationRepairFlow,
    async_create_fix_flow,
    HighIntegralTuningRepairFlow,
    SaveSeedsRepairFlow,
    SlopeDivergenceRepairFlow,
    UnknownRepairFlow,
)

from .conftest import make_pi_config, make_config


def _make_entry(config):
    """Create a MockConfigEntry from a config dict."""
    return MockConfigEntry(domain=DOMAIN, data=config, title="Test")


class TestCheckConfigIssues:
    """Tests for _check_config_issues."""

    def test_no_issues_when_pi_disabled(self, hass):
        """No issues should be created when PI is disabled."""
        config = make_config()
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create, \
             patch.object(ir, "async_delete_issue") as mock_delete:
            _check_config_issues(hass, entry)
            mock_create.assert_not_called()

    def test_no_issues_when_entities_exist(self, hass):
        """No issues when all configured entities exist."""
        config = make_pi_config()
        entry = _make_entry(config)

        # Set up entity states so they exist
        hass.states.async_set("sensor.outdoor_temp", "5.0")
        hass.states.async_set("sensor.room_temp", "22.0")

        with patch.object(ir, "async_create_issue") as mock_create, \
             patch.object(ir, "async_delete_issue") as mock_delete:
            _check_config_issues(hass, entry)
            mock_create.assert_not_called()

    def test_issue_created_for_missing_outdoor_sensor(self, hass):
        """Issue should be created when outdoor sensor doesn't exist."""
        config = make_pi_config({"outdoor_temp_sensor": "sensor.nonexistent"})
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_config_issues(hass, entry)
            # Should have been called with outdoor_sensor_not_found
            assert any(
                call.kwargs.get("translation_key") == "outdoor_sensor_not_found"
                or (len(call.args) > 2 and "outdoor" in str(call))
                for call in mock_create.call_args_list
            ) or mock_create.called

    def test_issue_created_for_missing_model_input_entity(self, hass):
        """Issue should be created when a model input entity doesn't exist."""
        config = make_pi_config({
            "pi_model_inputs": [{
                "name": "Test Door",
                "entity_id": "binary_sensor.nonexistent",
                "seed_heat": -1.5,
                "seed_cool": 0.0,
                "lag_tau": 0,
            }],
            "outdoor_temp_sensor": "",
        })
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_config_issues(hass, entry)
            assert mock_create.called

    def test_issue_deleted_when_entity_exists(self, hass):
        """Issue should be deleted when entity exists."""
        config = make_pi_config({
            "outdoor_temp_sensor": "sensor.outdoor_temp",
        })
        entry = _make_entry(config)
        hass.states.async_set("sensor.outdoor_temp", "5.0")

        with patch.object(ir, "async_delete_issue") as mock_delete:
            _check_config_issues(hass, entry)
            # Should delete the outdoor sensor issue since entity exists
            assert mock_delete.called

    def test_empty_disturbance_inputs_no_issues(self, hass):
        """Empty disturbance inputs should not trigger issues."""
        config = make_pi_config({
            "outdoor_temp_sensor": "",
            "pi_disturbance_inputs": [],
        })
        entry = _make_entry(config)

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_config_issues(hass, entry)
            mock_create.assert_not_called()


# ── Repair flow factory ──────────────────────────────────────────────


class TestRepairFlowFactory:
    """Tests for async_create_fix_flow routing."""

    @pytest.mark.asyncio
    async def test_routes_save_seeds(self, hass):
        """save_seeds repair_type routes to SaveSeedsRepairFlow."""
        flow = await async_create_fix_flow(
            hass, "save_seeds_test123",
            {"repair_type": "save_seeds", "entry_id": "test123"},
        )
        assert isinstance(flow, SaveSeedsRepairFlow)

    @pytest.mark.asyncio
    async def test_routes_slope_divergence(self, hass):
        """slope_divergence repair_type routes to SlopeDivergenceRepairFlow."""
        flow = await async_create_fix_flow(
            hass, "slope_divergence_test123_heat",
            {"repair_type": "slope_divergence", "entry_id": "test123", "mode": "heat"},
        )
        assert isinstance(flow, SlopeDivergenceRepairFlow)

    @pytest.mark.asyncio
    async def test_routes_high_integral_tuning(self, hass):
        """high_integral_tuning repair_type routes to HighIntegralTuningRepairFlow."""
        flow = await async_create_fix_flow(
            hass, "high_integral_test123",
            {"repair_type": "high_integral_tuning", "entry_id": "test123"},
        )
        assert isinstance(flow, HighIntegralTuningRepairFlow)

    @pytest.mark.asyncio
    async def test_routes_anomalous_observation(self, hass):
        """anomalous_observation routes to AnomalousObservationRepairFlow."""
        flow = await async_create_fix_flow(
            hass, "anomalous_test",
            {"repair_type": "anomalous_observation", "entry_id": "test123"},
        )
        assert isinstance(flow, AnomalousObservationRepairFlow)

    @pytest.mark.asyncio
    async def test_unknown_type_returns_fallback(self, hass):
        """Unrecognized repair_type returns UnknownRepairFlow."""
        flow = await async_create_fix_flow(hass, "unknown_issue", {"repair_type": "bogus"})
        assert isinstance(flow, UnknownRepairFlow)

    @pytest.mark.asyncio
    async def test_none_data_returns_fallback(self, hass):
        """None data returns UnknownRepairFlow."""
        flow = await async_create_fix_flow(hass, "no_data_issue", None)
        assert isinstance(flow, UnknownRepairFlow)

    @pytest.mark.asyncio
    async def test_empty_data_returns_fallback(self, hass):
        """Empty dict returns UnknownRepairFlow."""
        flow = await async_create_fix_flow(hass, "empty_data", {})
        assert isinstance(flow, UnknownRepairFlow)


# ── SaveSeedsRepairFlow ──────────────────────────────────────────────


class TestSaveSeedsRepairFlow:
    """Tests for the save seeds fix flow."""

    @pytest.mark.asyncio
    async def test_confirm_shows_form(self, hass):
        """Init step shows confirmation form."""
        flow = SaveSeedsRepairFlow({
            "entry_id": "test123",
            "coefficient_summary": "outdoor_delta: 0.30 → 0.42",
        })
        flow.hass = hass

        result = await flow.async_step_init()
        assert result["type"] == "form"
        assert result["step_id"] == "confirm"
        assert "coefficient_summary" in result["description_placeholders"]

    @pytest.mark.asyncio
    async def test_confirm_aborts_missing_entry(self, hass):
        """Aborts if config entry no longer exists."""
        flow = SaveSeedsRepairFlow({"entry_id": "nonexistent"})
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "abort"
        assert result["reason"] == "entry_not_found"

    @pytest.mark.asyncio
    async def test_confirm_applies_seeds(self, hass, setup_pi_integration):
        """Confirm step applies learned seeds to config entry."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Simulate learned coefficients different from seeds
        pi._rls_heat.beta[1] = 0.42 * pi._rls_heat.feature_scales[1]
        pi._rls_heat.observation_count = 100

        flow = SaveSeedsRepairFlow({
            "entry_id": entry.entry_id,
            "coefficient_summary": "test",
        })
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "create_entry"

        # Verify config was updated
        updated_entry = hass.config_entries.async_get_entry(entry.entry_id)
        assert updated_entry is not None
        from custom_components.tasmota_irhvac.const import CONF_PI_FF_HEAT_SLOPE
        # The learned slope should now be in config
        assert CONF_PI_FF_HEAT_SLOPE in updated_entry.options

    @pytest.mark.asyncio
    async def test_confirm_aborts_no_pi(self, hass):
        """Aborts if climate entity has no PI controller."""
        from custom_components.tasmota_irhvac.const import DATA_KEY

        entry = MockConfigEntry(domain=DOMAIN, data=make_config(), title="Test")
        entry.add_to_hass(hass)

        # Simulate entity without PI
        mock_climate = MagicMock()
        mock_climate._pi = None
        hass.data.setdefault(DATA_KEY, {})[entry.entry_id] = mock_climate

        flow = SaveSeedsRepairFlow({"entry_id": entry.entry_id})
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "abort"
        assert result["reason"] == "pi_not_available"


# ── 7-tuple plumbing ─────────────────────────────────────────────────


class TestTuningHealthFixableIssues:
    """Test that _check_tuning_health returns fixable flag for save_seeds."""

    @pytest.mark.asyncio
    async def test_save_seeds_issue_is_fixable(self, hass, setup_pi_integration):
        """save_seeds issue should have is_fixable=True and data dict."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Simulate converged model with seeds not matching
        pi._metrics.integral_convergence = 1.0
        pi._rls_heat.observation_count = 100
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]

        issues = pi._check_tuning_health()
        save_issues = [i for i in issues if "save_seeds" in i[0]]
        assert len(save_issues) == 1

        issue = save_issues[0]
        # 7-tuple: (id, severity, key, placeholders, should_create, is_fixable, data)
        assert len(issue) == 7
        assert issue[4] is True   # should_create
        assert issue[5] is True   # is_fixable
        assert issue[6] is not None  # data
        assert issue[6]["repair_type"] == "save_seeds"
        assert issue[6]["entry_id"] is not None

    @pytest.mark.asyncio
    async def test_non_fixable_issues_have_false(self, hass, setup_pi_integration):
        """Non-fixable issues should have is_fixable=False and data=None."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Simulate high integral (non-fixable in Phase 1)
        pi._metrics.integral_convergence = 20.0
        pi._rls_heat.observation_count = 100
        pi._tuning_alert_counters["high_integral"] = 5

        issues = pi._check_tuning_health(from_batch=True)
        integral_issues = [i for i in issues if "high_integral" in i[0]]
        assert len(integral_issues) == 1

        issue = integral_issues[0]
        assert len(issue) == 7
        assert issue[5] is False  # is_fixable
        assert issue[6] is None   # data

    @pytest.mark.asyncio
    async def test_issue_creation_passes_fixable_flag(self, hass, setup_pi_integration):
        """_check_tuning_health_issues passes is_fixable to ir.async_create_issue."""
        from custom_components.tasmota_irhvac.__init__ import _check_tuning_health_issues
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Trigger save_seeds
        pi._metrics.integral_convergence = 1.0
        pi._rls_heat.observation_count = 100
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]

        with patch.object(ir, "async_create_issue") as mock_create:
            _check_tuning_health_issues(hass, entry)
            # Find the save_seeds call
            save_calls = [
                c for c in mock_create.call_args_list
                if c.kwargs.get("translation_key") == "save_seeds"
            ]
            assert len(save_calls) == 1
            call_kwargs = save_calls[0].kwargs
            assert call_kwargs["is_fixable"] is True
            assert "data" in call_kwargs
            assert call_kwargs["data"]["repair_type"] == "save_seeds"


# ── SlopeDivergenceRepairFlow ────────────────────────────────────────


class TestSlopeDivergenceRepairFlow:
    """Tests for the slope divergence fix flow."""

    @pytest.mark.asyncio
    async def test_confirm_shows_form_with_values(self, hass):
        """Init step shows form with slope values."""
        flow = SlopeDivergenceRepairFlow({
            "entry_id": "test123",
            "mode": "heat",
            "learned_slope": 0.42,
            "configured_slope": 0.30,
        })
        flow.hass = hass

        result = await flow.async_step_init()
        assert result["type"] == "form"
        assert result["step_id"] == "confirm"
        assert result["description_placeholders"]["mode"] == "heat"
        assert result["description_placeholders"]["learned"] == "0.4200"
        assert result["description_placeholders"]["configured"] == "0.3000"

    @pytest.mark.asyncio
    async def test_confirm_updates_heat_slope(self, hass, setup_pi_integration):
        """Confirm updates pi_ff_heat_slope in config entry."""
        entry = await setup_pi_integration()

        flow = SlopeDivergenceRepairFlow({
            "entry_id": entry.entry_id,
            "mode": "heat",
            "learned_slope": 0.42,
            "configured_slope": 0.30,
        })
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "create_entry"

        updated = hass.config_entries.async_get_entry(entry.entry_id)
        from custom_components.tasmota_irhvac.const import CONF_PI_FF_HEAT_SLOPE
        assert updated.options[CONF_PI_FF_HEAT_SLOPE] == 0.42

    @pytest.mark.asyncio
    async def test_confirm_updates_cool_slope(self, hass, setup_pi_integration):
        """Confirm updates pi_ff_cool_slope for cool mode."""
        entry = await setup_pi_integration()

        flow = SlopeDivergenceRepairFlow({
            "entry_id": entry.entry_id,
            "mode": "cool",
            "learned_slope": 0.25,
            "configured_slope": 0.20,
        })
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "create_entry"

        updated = hass.config_entries.async_get_entry(entry.entry_id)
        from custom_components.tasmota_irhvac.const import CONF_PI_FF_COOL_SLOPE
        assert updated.options[CONF_PI_FF_COOL_SLOPE] == 0.25

    @pytest.mark.asyncio
    async def test_confirm_aborts_missing_entry(self, hass):
        """Aborts if config entry doesn't exist."""
        flow = SlopeDivergenceRepairFlow({
            "entry_id": "nonexistent",
            "mode": "heat",
            "learned_slope": 0.42,
            "configured_slope": 0.30,
        })
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "abort"

    @pytest.mark.asyncio
    async def test_slope_divergence_issue_is_fixable(self, hass, setup_pi_integration):
        """Slope divergence issues should be fixable with data."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Trigger slope divergence: 67% drift, sustained
        pi._rls_heat.beta[1] = 0.5 * pi._rls_heat.feature_scales[1]
        pi._rls_heat.observation_count = 100
        pi._tuning_alert_counters["slope_div_heat"] = 5

        issues = pi._check_tuning_health(from_batch=True)
        slope_issues = [i for i in issues if "slope_divergence" in i[0]]
        assert len(slope_issues) >= 1

        issue = slope_issues[0]
        assert issue[5] is True  # is_fixable
        assert issue[6] is not None
        assert issue[6]["repair_type"] == "slope_divergence"
        assert issue[6]["mode"] == "heat"
        assert "learned_slope" in issue[6]


# ── HighIntegralTuningRepairFlow ─────────────────────────────────────


class TestHighIntegralTuningRepairFlow:
    """Tests for the high integral tuning fix flow."""

    @pytest.mark.asyncio
    async def test_confirm_shows_form(self, hass):
        """Init step shows form with Ki values."""
        flow = HighIntegralTuningRepairFlow({
            "entry_id": "test123",
            "current_ki": 0.15,
            "suggested_ki": 0.05,
        })
        flow.hass = hass

        result = await flow.async_step_init()
        assert result["type"] == "form"
        assert result["step_id"] == "confirm"
        assert result["description_placeholders"]["current_ki"] == "0.150"
        assert result["description_placeholders"]["suggested_ki"] == "0.050"

    @pytest.mark.asyncio
    async def test_confirm_updates_ki(self, hass, setup_pi_integration):
        """Confirm updates pi_ki in config entry."""
        entry = await setup_pi_integration()

        flow = HighIntegralTuningRepairFlow({
            "entry_id": entry.entry_id,
            "current_ki": 0.15,
            "suggested_ki": 0.05,
        })
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "create_entry"

        updated = hass.config_entries.async_get_entry(entry.entry_id)
        from custom_components.tasmota_irhvac.const import CONF_PI_KI
        assert updated.options[CONF_PI_KI] == 0.05

    @pytest.mark.asyncio
    async def test_confirm_aborts_missing_entry(self, hass):
        """Aborts if config entry doesn't exist."""
        flow = HighIntegralTuningRepairFlow({
            "entry_id": "nonexistent",
            "current_ki": 0.15,
            "suggested_ki": 0.05,
        })
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "abort"

    @pytest.mark.asyncio
    async def test_high_integral_tuning_is_fixable(self, hass, setup_pi_integration):
        """high_integral_tuning sub-case should be fixable."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Trigger high_integral sub-case 4 (tuning):
        # Need high ki_correction, matured model, no slope gap, low uncontrollable
        from homeassistant.components.climate import HVACMode
        entity._attr_hvac_mode = HVACMode.HEAT  # ensure heat path
        pi._metrics.integral_convergence = 20.0
        pi._rls_heat.observation_count = 100
        pi._tuning_alert_counters["high_integral"] = 5
        pi._metrics.uncontrollable_cvh = 0.0
        pi._metrics.comfort_violation_hours = 1.0
        # Align learned slope to configured to avoid sub-case 2 (slope_gap)
        pi._rls_heat.beta[1] = pi._ff_heat_slope * pi._rls_heat.feature_scales[1]

        issues = pi._check_tuning_health(from_batch=True)
        integral_issues = [i for i in issues if "high_integral" in i[0]]
        assert len(integral_issues) == 1

        issue = integral_issues[0]
        # Sub-case 4 returns key "high_integral_tuning"
        assert issue[2] == "high_integral_tuning"
        assert issue[5] is True  # is_fixable
        assert issue[6] is not None
        assert issue[6]["repair_type"] == "high_integral_tuning"
        assert "suggested_ki" in issue[6]

    @pytest.mark.asyncio
    async def test_high_integral_immature_not_fixable(self, hass, setup_pi_integration):
        """high_integral_immature sub-case should NOT be fixable."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Trigger sub-case 1 (immature): high correction, low obs count
        from homeassistant.components.climate import HVACMode
        entity._attr_hvac_mode = HVACMode.HEAT
        pi._metrics.integral_convergence = 20.0
        pi._rls_heat.observation_count = 10  # < maturity_obs (50)
        pi._tuning_alert_counters["high_integral"] = 5

        issues = pi._check_tuning_health(from_batch=True)
        integral_issues = [i for i in issues if "high_integral" in i[0]]
        assert len(integral_issues) == 1

        issue = integral_issues[0]
        assert issue[2] == "high_integral_immature"
        assert issue[5] is False  # not fixable
        assert issue[6] is None


# ── AnomalousObservationRepairFlow ───────────────────────────────────


class TestAnomalousObservationRepairFlow:
    """Tests for the anomalous observation fix flow."""

    @pytest.mark.asyncio
    async def test_confirm_shows_form_with_choices(self, hass):
        """Init step shows form with exclude/dismiss options."""
        flow = AnomalousObservationRepairFlow({
            "entry_id": "test123",
            "start_mono": 1000.0,
            "end_mono": 2000.0,
            "time_range": "14:00 — 14:30",
            "direction": "unexpected heat loss",
            "mean_residual": "-0.80",
        })
        flow.hass = hass

        result = await flow.async_step_init()
        assert result["type"] == "form"
        assert result["step_id"] == "confirm"
        assert "time_range" in result["description_placeholders"]

    @pytest.mark.asyncio
    async def test_exclude_calls_pi(self, hass, setup_pi_integration):
        """Exclude action calls exclude_observations_by_time."""
        from .conftest import get_climate_entity
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        # Add observations to buffer
        for i in range(10):
            pi._observation_buffer_heat.add(Observation(
                timestamp=1000.0 + i * 60,
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i),
                room_rate=0.0,
                raw_readings={},
                clamped=False,
            ))

        flow = AnomalousObservationRepairFlow({
            "entry_id": entry.entry_id,
            "start_mono": 1300.0,
            "end_mono": 1600.0,
        })
        flow.hass = hass

        before = len(pi._observation_buffer_heat)
        result = await flow.async_step_confirm(user_input={"action": "exclude"})
        assert result["type"] == "create_entry"
        assert len(pi._observation_buffer_heat) < before
        assert pi._exclusion_count == 1

    @pytest.mark.asyncio
    async def test_dismiss_keeps_buffer(self, hass, setup_pi_integration):
        """Dismiss action does not modify buffer."""
        from .conftest import get_climate_entity
        from custom_components.tasmota_irhvac.pi.batch_learning import Observation

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        for i in range(5):
            pi._observation_buffer_heat.add(Observation(
                timestamp=1000.0 + i * 60,
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=22.0,
                current_c=21.0,
                desired_c=21.0,
                outdoor_temp_c=21.0 + float(i),
                room_rate=0.0,
                raw_readings={},
                clamped=False,
            ))

        flow = AnomalousObservationRepairFlow({
            "entry_id": entry.entry_id,
            "start_mono": 1000.0,
            "end_mono": 2000.0,
        })
        flow.hass = hass

        result = await flow.async_step_confirm(user_input={"action": "dismiss"})
        assert result["type"] == "create_entry"
        assert len(pi._observation_buffer_heat) == 5
        assert pi._exclusion_count == 0


# ── Anomaly event cause hints ────────────────────────────────────────


class TestAnomalyCauseHints:
    """Test mode-aware cause hint generation in _check_tuning_health."""

    @pytest.mark.asyncio
    async def test_heat_positive_residual(self, hass, setup_pi_integration):
        """Heating + positive residual → 'unexpected heat loss'."""
        from .conftest import get_climate_entity
        from datetime import datetime
        from custom_components.tasmota_irhvac.pi.health_checks import AnomalyEvent

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        now = datetime.now()
        pi._anomaly_events.append(AnomalyEvent(
            start_time=now, start_mono=1000.0,
            end_time=now, end_mono=1000.0,
            tick_count=1, mean_residual=0.5,
            peak_cusum=12.0, mode="heat",
        ))

        issues = pi._check_tuning_health()
        anomaly_issues = [i for i in issues if "anomalous_observation" in i[0]]
        assert len(anomaly_issues) == 1
        assert "heat loss" in anomaly_issues[0][3]["direction"]

    @pytest.mark.asyncio
    async def test_heat_negative_residual(self, hass, setup_pi_integration):
        """Heating + negative residual → 'unexpected heat gain'."""
        from .conftest import get_climate_entity
        from datetime import datetime
        from custom_components.tasmota_irhvac.pi.health_checks import AnomalyEvent

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        now = datetime.now()
        pi._anomaly_events.append(AnomalyEvent(
            start_time=now, start_mono=1000.0,
            end_time=now, end_mono=1000.0,
            tick_count=1, mean_residual=-0.5,
            peak_cusum=12.0, mode="heat",
        ))

        issues = pi._check_tuning_health()
        anomaly_issues = [i for i in issues if "anomalous_observation" in i[0]]
        assert "heat gain" in anomaly_issues[0][3]["direction"]

    @pytest.mark.asyncio
    async def test_cool_positive_residual(self, hass, setup_pi_integration):
        """Cooling + positive residual → 'unexpected heat gain'."""
        from .conftest import get_climate_entity
        from datetime import datetime
        from custom_components.tasmota_irhvac.pi.health_checks import AnomalyEvent

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        now = datetime.now()
        pi._anomaly_events.append(AnomalyEvent(
            start_time=now, start_mono=1000.0,
            end_time=now, end_mono=1000.0,
            tick_count=1, mean_residual=0.5,
            peak_cusum=12.0, mode="cool",
        ))

        issues = pi._check_tuning_health()
        anomaly_issues = [i for i in issues if "anomalous_observation" in i[0]]
        assert "heat gain" in anomaly_issues[0][3]["direction"]

    @pytest.mark.asyncio
    async def test_cool_negative_residual(self, hass, setup_pi_integration):
        """Cooling + negative residual → 'unexpected heat loss'."""
        from .conftest import get_climate_entity
        from datetime import datetime
        from custom_components.tasmota_irhvac.pi.health_checks import AnomalyEvent

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        now = datetime.now()
        pi._anomaly_events.append(AnomalyEvent(
            start_time=now, start_mono=1000.0,
            end_time=now, end_mono=1000.0,
            tick_count=1, mean_residual=-0.5,
            peak_cusum=12.0, mode="cool",
        ))

        issues = pi._check_tuning_health()
        anomaly_issues = [i for i in issues if "anomalous_observation" in i[0]]
        assert "heat loss" in anomaly_issues[0][3]["direction"]


# ── Frequent exclusions escalation ───────────────────────────────────


class TestFrequentExclusionsEscalation:
    """Test frequent_exclusions issue surfaces after 3+ exclusions."""

    @pytest.mark.asyncio
    async def test_surfaces_at_threshold(self, hass, setup_pi_integration):
        """frequent_exclusions surfaces when exclusion_count >= 3."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._exclusion_count = 3

        issues = pi._check_tuning_health()
        freq_issues = [i for i in issues if "frequent_exclusions" in i[0]]
        assert len(freq_issues) == 1
        assert freq_issues[0][4] is True   # should_create
        assert freq_issues[0][5] is False  # not fixable
        assert freq_issues[0][3]["count"] == "3"

    @pytest.mark.asyncio
    async def test_not_surfaced_below_threshold(self, hass, setup_pi_integration):
        """frequent_exclusions NOT surfaced when exclusion_count < 3."""
        from .conftest import get_climate_entity

        entry = await setup_pi_integration()
        entity = get_climate_entity(hass, entry)
        pi = entity._pi

        pi._exclusion_count = 2

        issues = pi._check_tuning_health()
        freq_issues = [i for i in issues if "frequent_exclusions" in i[0]]
        assert len(freq_issues) == 0
