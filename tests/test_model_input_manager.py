"""Unit tests for ModelInputManager — no PIController or HA dependencies."""

import math

from custom_components.tasmota_irhvac.model_input_manager import ModelInputManager


STOVE_INPUT = {
    "name": "Stove",
    "entity_id": "input_boolean.stove",
    "seed_heat": -3.0,
    "seed_cool": 0.0,
    "lag_tau": 0,
}


class TestLagFilters:
    """Tests for exponential lag filter update."""

    def test_lag_filter_with_tau(self):
        """Lag filter with tau > 0 applies exponential smoothing."""
        m_input = {**STOVE_INPUT, "lag_tau": 1800}  # 30 min in seconds
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)

        mgr.values[0] = 1.0
        mgr.filtered[0] = 0.0
        mgr.update_lag_filters(900)  # dt = 15 min (half of tau)

        # Should be partially ramped up (not 0 and not 1)
        assert 0.0 < mgr.filtered[0] < 1.0
        # Exact: 1 - exp(-900/1800) ≈ 0.3935
        expected = 1.0 - math.exp(-900 / 1800)
        assert abs(mgr.filtered[0] - expected) < 0.001

    def test_lag_filter_with_zero_tau(self):
        """Lag filter with tau=0 passes through raw value."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)

        mgr.values[0] = 1.0
        mgr.filtered[0] = 0.5
        mgr.update_lag_filters(900)

        assert mgr.filtered[0] == 1.0


class TestReadValues:
    """Tests for read_values with resolved entity states."""

    def test_empty_entity_id_skipped(self):
        """Model input with empty entity_id is skipped."""
        m_input = {**STOVE_INPUT, "entity_id": ""}
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)
        mgr.values[0] = 99.0

        mgr.read_values({})
        assert mgr.values[0] == 99.0  # Unchanged

    def test_unavailable_entity_keeps_last_value(self):
        """Unavailable entity keeps previous value."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.values[0] = 1.0

        mgr.read_values({"input_boolean.stove": ("unavailable", False)})
        assert mgr.values[0] == 1.0

    def test_missing_entity_keeps_last_value(self):
        """Entity not in states dict keeps previous value."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.values[0] = 0.5

        mgr.read_values({})  # Entity not present
        assert mgr.values[0] == 0.5

    def test_numeric_entity_parsed(self):
        """Numeric state string is parsed to float."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.read_values({"input_boolean.stove": ("23.5", True)})
        assert mgr.values[0] == 23.5

    def test_binary_entity_mapped(self):
        """Non-numeric state mapped: 'on'→1.0, 'off'→0.0."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)

        mgr.read_values({"input_boolean.stove": ("on", True)})
        assert mgr.values[0] == 1.0

        mgr.read_values({"input_boolean.stove": ("off", True)})
        assert mgr.values[0] == 0.0

    def test_climate_states_mapped(self):
        """Climate states like 'heat', 'cool' map to 1.0."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        for state in ("heat", "cool", "heating", "cooling", "burning"):
            mgr.read_values({"input_boolean.stove": (state, True)})
            assert mgr.values[0] == 1.0, f"'{state}' should map to 1.0"


class TestAnyUnavailable:
    """Tests for any_unavailable check."""

    def test_empty_entity_id_not_unavailable(self):
        """Empty entity_id is skipped, not flagged unavailable."""
        m_input = {**STOVE_INPUT, "entity_id": ""}
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)
        assert mgr.any_unavailable({}) is False

    def test_unavailable_entity_returns_true(self):
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        assert mgr.any_unavailable({"input_boolean.stove": ("", False)}) is True

    def test_available_entity_returns_false(self):
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        assert mgr.any_unavailable({"input_boolean.stove": ("off", True)}) is False

    def test_missing_entity_returns_true(self):
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        assert mgr.any_unavailable({}) is True  # Not in dict


class TestFeatureVector:
    """Tests for build_feature_vector."""

    def test_basic_vector(self):
        """Feature vector is [1, outdoor_delta, filtered_inputs...]."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.filtered[0] = 0.75
        x = mgr.build_feature_vector(10.0)
        assert x == [1.0, 10.0, 0.75]

    def test_empty_inputs(self):
        """No model inputs → vector is just [1, outdoor_delta]."""
        mgr = ModelInputManager(model_inputs=[], outdoor_temp_sensor=None)
        x = mgr.build_feature_vector(5.0)
        assert x == [1.0, 5.0]


class TestOutdoorTemp:
    """Tests for outdoor temperature management."""

    def test_update_outdoor_temp_celsius(self):
        mgr = ModelInputManager(model_inputs=[], outdoor_temp_sensor="sensor.outdoor")
        mgr.update_outdoor_temp("10.5", "°C")
        assert mgr.outdoor_temp == 10.5

    def test_update_outdoor_temp_fahrenheit(self):
        mgr = ModelInputManager(model_inputs=[], outdoor_temp_sensor="sensor.outdoor")
        mgr.update_outdoor_temp("50.0", "°F")
        assert mgr.outdoor_temp is not None
        assert abs(mgr.outdoor_temp - 10.0) < 0.1

    def test_update_outdoor_temp_invalid(self):
        mgr = ModelInputManager(model_inputs=[], outdoor_temp_sensor="sensor.outdoor")
        mgr.outdoor_temp = 5.0
        mgr.update_outdoor_temp("unavailable", "°C")
        assert mgr.outdoor_temp == 5.0  # Unchanged on parse failure


class TestPersistence:
    """Tests for lag state save/restore."""

    def test_round_trip(self):
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.filtered[0] = 0.75
        states = mgr.get_lag_states()
        assert states == {"Stove": 0.75}

        mgr2 = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr2.restore_lag_states(states)
        assert mgr2.filtered[0] == 0.75
