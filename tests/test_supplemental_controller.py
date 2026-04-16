"""Unit tests for SupplementalController — no PIController or HA dependencies."""

from custom_components.tasmota_irhvac.pi.supplemental_controller import (
    SupplementalController,
    SupplementalResult,
)

DEFAULT_SOURCE = {
    "name": "Pellet Stove",
    "entity_id": "climate.pellet_stove",
    "failure_threshold": 900,
    "recovery_margin": 0.3,
}


class TestSupplementalController:
    """Tests for override/selector tracking mode state machine."""

    def _make(self, sources=None, deadband=0.5):
        if sources is None:
            sources = [DEFAULT_SOURCE.copy()]
        return SupplementalController(sources=sources, deadband=deadband)

    def test_no_sources_always_active(self):
        """No supplemental sources → HP always active."""
        ctrl = self._make(sources=[])
        result = ctrl.evaluate(error_c=1.0, now_mono=100.0, active_sources=[])
        assert result.hp_should_send_ir is True
        assert not ctrl.tracking_mode

    def test_active_enters_tracking(self):
        """When supplemental is heating, HP enters tracking mode."""
        ctrl = self._make()
        result = ctrl.evaluate(error_c=0.0, now_mono=100.0, active_sources=["Pellet Stove"])
        assert result.hp_should_send_ir is False
        assert ctrl.tracking_mode is True
        assert ctrl.tracking_sources == ["Pellet Stove"]

    def test_inactive_hp_active(self):
        """When supplemental is off, HP is active."""
        ctrl = self._make()
        result = ctrl.evaluate(error_c=1.0, now_mono=100.0, active_sources=[])
        assert result.hp_should_send_ir is True
        assert ctrl.tracking_mode is False

    def test_failure_detection_starts_timer(self):
        """Error above deadband starts failure timer."""
        ctrl = self._make()
        ctrl.evaluate(error_c=1.0, now_mono=100.0, active_sources=["Pellet Stove"])
        assert ctrl.failure_start == 100.0
        assert ctrl.tracking_mode is True  # Still tracking (threshold not met)

    def test_failure_threshold_triggers_assist(self):
        """After failure_threshold seconds below desired, HP assists."""
        ctrl = self._make()
        ctrl.evaluate(error_c=1.0, now_mono=100.0, active_sources=["Pellet Stove"])
        assert ctrl.assist_active is False

        # 901s later, exceeds 900s threshold
        result = ctrl.evaluate(error_c=1.0, now_mono=1001.0, active_sources=["Pellet Stove"])
        assert ctrl.assist_active is True
        assert result.hp_should_send_ir is True  # HP active (assisting)
        assert ctrl.tracking_mode is False

    def test_recovery_clears_assist(self):
        """Room recovering past margin clears assist mode."""
        ctrl = self._make()
        ctrl.assist_active = True
        ctrl.failure_start = 0.0

        # Error negative beyond recovery_margin (0.3) → recovered
        ctrl.evaluate(error_c=-0.5, now_mono=2000.0, active_sources=["Pellet Stove"])
        assert ctrl.assist_active is False
        assert ctrl.failure_start is None
        assert ctrl.tracking_mode is True  # Back to tracking

    def test_bumpless_transfer_on_supplemental_end(self):
        """When supplemental stops, should_reset_hold_timer is set."""
        ctrl = self._make()
        ctrl.tracking_mode = True
        ctrl.tracking_sources = ["Pellet Stove"]

        result = ctrl.evaluate(error_c=1.0, now_mono=2000.0, active_sources=[])
        assert result.hp_should_send_ir is True
        assert result.should_reset_hold_timer is True
        assert ctrl.tracking_mode is False

    def test_error_within_deadband_clears_failure_timer(self):
        """Error dropping within deadband clears failure start."""
        ctrl = self._make()
        ctrl.evaluate(error_c=1.0, now_mono=100.0, active_sources=["Pellet Stove"])
        assert ctrl.failure_start == 100.0

        ctrl.evaluate(error_c=0.0, now_mono=200.0, active_sources=["Pellet Stove"])
        assert ctrl.failure_start is None

    def test_empty_entity_id_skipped(self):
        """Source with empty entity_id: no active sources detected."""
        ctrl = self._make(sources=[{"name": "Bad", "entity_id": ""}])
        # No active sources because entity_id is empty — source_configs has it
        # but PIController wouldn't resolve any active state for it
        result = ctrl.evaluate(error_c=1.0, now_mono=100.0, active_sources=[])
        assert result.hp_should_send_ir is True

    def test_no_reset_when_not_transitioning(self):
        """should_reset_hold_timer is False when not transitioning out of tracking."""
        ctrl = self._make()
        # Not tracking, supplemental inactive → no transition
        result = ctrl.evaluate(error_c=1.0, now_mono=100.0, active_sources=[])
        assert result.should_reset_hold_timer is False
