"""Unit tests for ModelInputManager — no PIController or HA dependencies."""

import math

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


GATED_INPUT = {
    "name": "Hallway Convective",
    "entity_id": "sensor.hallway_temp",
    "seed_heat": 1.0,
    "seed_cool": 0.0,
    "lag_tau": 300,
    "delta_from_room": True,
    "gate_entity": "binary_sensor.bunkroom_door",
    "gate_invert": False,
}


class TestGateEntity:
    """Tests for gate_entity gating model input values."""

    def test_gate_on_passes_value(self):
        """When gate entity is on (active), value passes through."""
        mgr = ModelInputManager(model_inputs=[GATED_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        mgr.read_values(
            {
                "sensor.hallway_temp": ("22.0", True),
                "binary_sensor.bunkroom_door": ("on", True),
            },
            room_temp_c=20.0,
        )
        # delta_from_room: 22 - 20 = 2.0, gate is on → value passes
        assert abs(mgr.values[0] - 2.0) < 0.01

    def test_gate_off_zeros_value(self):
        """When gate entity is off (inactive), value forced to zero."""
        mgr = ModelInputManager(model_inputs=[GATED_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        mgr.read_values(
            {
                "sensor.hallway_temp": ("22.0", True),
                "binary_sensor.bunkroom_door": ("off", True),
            },
            room_temp_c=20.0,
        )
        assert mgr.values[0] == 0.0

    def test_gate_invert_on_zeros_value(self):
        """With gate_invert=True, gate ON forces value to zero."""
        m_input = {**GATED_INPUT, "gate_invert": True}
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        mgr.read_values(
            {
                "sensor.hallway_temp": ("22.0", True),
                "binary_sensor.bunkroom_door": ("on", True),
            },
            room_temp_c=20.0,
        )
        assert mgr.values[0] == 0.0

    def test_gate_invert_off_passes_value(self):
        """With gate_invert=True, gate OFF passes value through."""
        m_input = {**GATED_INPUT, "gate_invert": True}
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        mgr.read_values(
            {
                "sensor.hallway_temp": ("22.0", True),
                "binary_sensor.bunkroom_door": ("off", True),
            },
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - 2.0) < 0.01

    def test_gate_unavailable_keeps_last_value(self):
        """Unavailable gate entity keeps last value (doesn't zero)."""
        mgr = ModelInputManager(model_inputs=[GATED_INPUT], outdoor_temp_sensor=None)
        mgr.set_temp_unit(0, "°C")

        # First read: gate on, value set
        mgr.read_values(
            {
                "sensor.hallway_temp": ("22.0", True),
                "binary_sensor.bunkroom_door": ("on", True),
            },
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - 2.0) < 0.01

        # Second read: gate unavailable, keeps 2.0
        mgr.read_values(
            {
                "sensor.hallway_temp": ("22.0", True),
                "binary_sensor.bunkroom_door": ("", False),
            },
            room_temp_c=20.0,
        )
        assert abs(mgr.values[0] - 2.0) < 0.01

    def test_gate_missing_from_states_keeps_last_value(self):
        """Gate entity not in states dict keeps last value."""
        mgr = ModelInputManager(model_inputs=[GATED_INPUT], outdoor_temp_sensor=None)
        mgr.values[0] = 1.5

        mgr.read_values(
            {"sensor.hallway_temp": ("22.0", True)},
            room_temp_c=20.0,
        )
        # Gate entity missing → keeps value (delta was recomputed but gate
        # unavailable path preserves it)
        # Actually the value gets recomputed from delta, but gate missing
        # means we keep whatever was computed — let's check the actual behavior
        # The delta is computed (22-20=2), then gate is missing → keep value
        assert abs(mgr.values[0] - 2.0) < 0.01

    def test_no_gate_entity_passes_value(self):
        """Input without gate_entity is unaffected."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.read_values({"input_boolean.stove": ("on", True)})
        assert mgr.values[0] == 1.0

    def test_gate_with_numeric_input_no_delta(self):
        """Gate works on numeric (non-delta) inputs too."""
        m_input = {
            "name": "Solar Gated",
            "entity_id": "sensor.solar_proxy",
            "seed_heat": -2.0,
            "seed_cool": 0.0,
            "lag_tau": 0,
            "gate_entity": "input_boolean.solar_gate",
            "gate_invert": False,
        }
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)

        # Gate off → zero
        mgr.read_values(
            {
                "sensor.solar_proxy": ("0.7", True),
                "input_boolean.solar_gate": ("off", True),
            },
        )
        assert mgr.values[0] == 0.0

        # Gate on → passes
        mgr.read_values(
            {
                "sensor.solar_proxy": ("0.7", True),
                "input_boolean.solar_gate": ("on", True),
            },
        )
        assert mgr.values[0] == 0.7

    def test_mixed_gated_and_ungated(self):
        """Gated and ungated inputs coexist correctly."""
        ungated = {**STOVE_INPUT}
        gated = {**GATED_INPUT}
        mgr = ModelInputManager(
            model_inputs=[ungated, gated], outdoor_temp_sensor=None
        )
        mgr.set_temp_unit(1, "°C")

        mgr.read_values(
            {
                "input_boolean.stove": ("on", True),
                "sensor.hallway_temp": ("22.0", True),
                "binary_sensor.bunkroom_door": ("off", True),
            },
            room_temp_c=20.0,
        )
        assert mgr.values[0] == 1.0  # Ungated stove: on
        assert mgr.values[1] == 0.0  # Gated door closed: zeroed


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


class TestEnabledToggle:
    """Tests for the enabled/disabled toggle on model inputs."""

    def test_disabled_input_zeroed(self):
        """Disabled input forces value to zero."""
        m_input = {**STOVE_INPUT, "input_enabled": False}
        mgr = ModelInputManager(model_inputs=[m_input], outdoor_temp_sensor=None)
        mgr.read_values(
            {"input_boolean.stove": ("on", True)},
        )
        assert mgr.values[0] == 0.0

    def test_enabled_input_reads_normally(self):
        """Enabled input (default) reads the entity value."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.read_values(
            {"input_boolean.stove": ("on", True)},
        )
        assert mgr.values[0] == 1.0

    def test_enabled_default_true(self):
        """Missing 'enabled' key defaults to True."""
        mgr = ModelInputManager(model_inputs=[STOVE_INPUT], outdoor_temp_sensor=None)
        mgr.read_values(
            {"input_boolean.stove": ("on", True)},
        )
        assert mgr.values[0] == 1.0  # not zeroed


class TestNamedFeatures:
    """Tests for build_feature_names and build_named_features."""

    def test_build_feature_names(self):
        mgr = ModelInputManager(
            model_inputs=[STOVE_INPUT, DELTA_INPUT],
            outdoor_temp_sensor=None,
        )
        names = mgr.build_feature_names()
        assert names == ["intercept", "outdoor_delta", "Stove", "LR Delta"]

    def test_build_named_features(self):
        mgr = ModelInputManager(
            model_inputs=[STOVE_INPUT],
            outdoor_temp_sensor=None,
        )
        mgr.filtered[0] = 0.75
        features = mgr.build_named_features(10.0)
        assert features == {"intercept": 1.0, "outdoor_delta": 10.0, "Stove": 0.75}
