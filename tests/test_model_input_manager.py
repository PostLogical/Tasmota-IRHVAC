"""Unit tests for ModelInputManager — no PIController or HA dependencies."""

import math

import pytest

from custom_components.tasmota_irhvac.pi.model_input_manager import ModelInputManager


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


DELTA_INPUT = {
    "name": "LR Delta",
    "entity_id": "sensor.living_room_temp",
    "seed_heat": -1.0,
    "seed_cool": 0.0,
    "lag_tau": 0,
    "delta_from_room": True,
}


class TestDeltaFromRoom:
    """Tests for delta_from_room temperature delta computation."""

    def test_delta_celsius(self):
        """delta_from_room computes (entity_c − room_c) when unit is °C."""
        mgr = ModelInputManager(model_inputs=[DELTA_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        mgr.read_values(
            {"sensor.living_room_temp": ("22.0", True)},
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - 2.0) < 0.01

    def test_delta_fahrenheit(self):
        """delta_from_room converts °F entity to °C before subtracting."""
        mgr = ModelInputManager(model_inputs=[DELTA_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°F")

        # 71.6°F = 22°C, room = 20°C → delta = 2°C
        mgr.read_values(
            {"sensor.living_room_temp": ("71.6", True)},
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - 2.0) < 0.1

    def test_delta_no_room_temp_skips(self):
        """When room_temp_c is None, delta_from_room keeps raw value."""
        mgr = ModelInputManager(model_inputs=[DELTA_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        mgr.read_values(
            {"sensor.living_room_temp": ("22.0", True)},
            room_temp_c=None,
        )
        # Raw value stored, no delta
        assert mgr.values[0] == 22.0

    def test_delta_default_unit_celsius(self):
        """When no unit cached, defaults to °C."""
        mgr = ModelInputManager(model_inputs=[DELTA_INPUT], outdoor_temp_sensor=None)
        # Don't call set_temp_unit — _temp_units[0] is None

        mgr.read_values(
            {"sensor.living_room_temp": ("22.0", True)},
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - 2.0) < 0.01

    def test_non_delta_input_ignores_room_temp(self):
        """Normal inputs are unaffected by room_temp_c parameter."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)

        mgr.read_values(
            {"input_boolean.stove": ("on", True)},
            room_temp_c=20.0,
        )
        assert mgr.values[0] == 1.0

    def test_delta_negative_when_room_warmer(self):
        """Delta is negative when room is warmer than adjacent zone."""
        mgr = ModelInputManager(model_inputs=[DELTA_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        mgr.read_values(
            {"sensor.living_room_temp": ("18.0", True)},
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - (-2.0)) < 0.01

    def test_delta_unavailable_keeps_last(self):
        """Unavailable delta input keeps its last computed delta."""
        mgr = ModelInputManager(model_inputs=[DELTA_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        # First read: compute delta
        mgr.read_values(
            {"sensor.living_room_temp": ("22.0", True)},
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - 2.0) < 0.01

        # Second read: unavailable
        mgr.read_values(
            {"sensor.living_room_temp": ("", False)},
            room_temp_c=19.0,
        )
        # Should keep 2.0, not recompute
        assert abs(mgr.values[0] - 2.0) < 0.01

    def test_set_temp_unit(self):
        """set_temp_unit caches unit at correct index."""
        mgr = ModelInputManager(
            model_inputs=[STOVE_INPUT, DELTA_INPUT],
            outdoor_temp_sensor=None,
        )
        mgr.set_temp_unit(1, "°F")
        assert mgr._temp_units[0] is None
        assert mgr._temp_units[1] == "°F"

    def test_set_temp_unit_out_of_range(self):
        """set_temp_unit with invalid index is a no-op."""
        mgr = ModelInputManager(model_inputs=[DELTA_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(5, "°F")  # Should not raise
        assert mgr._temp_units[0] is None

    def test_mixed_inputs(self):
        """Delta and non-delta inputs coexist correctly."""
        mgr = ModelInputManager(
            model_inputs=[STOVE_INPUT, DELTA_INPUT],
            outdoor_temp_sensor=None,
        )
        mgr.set_temp_unit(1, "°C")

        mgr.read_values(
            {
                "input_boolean.stove": ("on", True),
                "sensor.living_room_temp": ("23.0", True),
            },
            room_temp_c=20.0,
        )
        assert mgr.values[0] == 1.0  # Stove: binary, unaffected
        assert abs(mgr.values[1] - 3.0) < 0.01  # Delta: 23 - 20


class TestAutoScale:
    """Tests for auto-compute feature scale tracking."""

    def test_not_ready_before_min_ticks(self):
        """Auto-scale is not ready before enough outdoor_delta samples."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        for _ in range(49):
            mgr.accumulate_scales(10.0)
        assert mgr.auto_scales_ready() is False

    def test_ready_at_min_ticks(self):
        """Auto-scale is ready after 50 outdoor_delta samples."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        for _ in range(50):
            mgr.accumulate_scales(10.0)
        assert mgr.auto_scales_ready() is True

    def test_outdoor_delta_mean(self):
        """Outdoor delta scale is mean of abs(outdoor_delta) values."""
        mgr = ModelInputManager(model_inputs=[], outdoor_temp_sensor=None)
        for _ in range(25):
            mgr.accumulate_scales(8.0)
        for _ in range(25):
            mgr.accumulate_scales(12.0)
        scales = mgr.get_auto_scales()
        assert scales[0] == 1.0  # intercept
        assert scales[1] == pytest.approx(10.0)  # mean of 8 and 12

    def test_model_input_mean(self):
        """Model input scale is mean of abs(filtered) values."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        # Need ≥50 non-zero samples for auto-compute
        for tick in range(100):
            mgr.filtered[0] = 1.0 if tick % 2 == 0 else 0.5
            mgr.accumulate_scales(10.0)
        scales = mgr.get_auto_scales()
        # 50 ticks at 1.0, 50 ticks at 0.5 → mean = 0.75
        assert scales[2] == pytest.approx(0.75)

    def test_sparse_input_falls_back_to_typical_value(self):
        """Input with too few samples falls back to configured typical_value."""
        m_input = {**STOVE_INPUT, "typical_value": 0.75}
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)
        # Only accumulate outdoor_delta, not the model input
        for _ in range(50):
            mgr.accumulate_scales(10.0)
        scales = mgr.get_auto_scales()
        assert scales[2] == pytest.approx(0.75)  # fallback

    def test_zero_values_skipped(self):
        """Zero values don't count toward the mean."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        for _ in range(50):
            mgr.filtered[0] = 0.0  # Always zero
            mgr.accumulate_scales(10.0)
        # Model input had zero samples → falls back
        assert mgr._scale_count[1] == 0

    def test_commit_stops_accumulation(self):
        """After commit, accumulate_scales is a no-op."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        for _ in range(50):
            mgr.accumulate_scales(10.0)
        mgr.commit_auto_scales()

        old_count = mgr._scale_count[0]
        mgr.accumulate_scales(20.0)
        assert mgr._scale_count[0] == old_count  # Unchanged

    def test_committed_not_ready(self):
        """After commit, auto_scales_ready returns False."""
        mgr = ModelInputManager(model_inputs=[], outdoor_temp_sensor=None)
        for _ in range(50):
            mgr.accumulate_scales(10.0)
        mgr.commit_auto_scales()
        assert mgr.auto_scales_ready() is False

    def test_intercept_always_one(self):
        """Intercept scale is always 1.0."""
        mgr = ModelInputManager(model_inputs=[], outdoor_temp_sensor=None)
        for _ in range(50):
            mgr.accumulate_scales(5.0)
        scales = mgr.get_auto_scales()
        assert scales[0] == 1.0


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
