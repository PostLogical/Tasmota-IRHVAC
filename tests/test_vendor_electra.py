"""Tests for the ELECTRA_AC vendor handler."""

from custom_components.tasmota_irhvac.vendors.electra import ElectraHandler


class TestElectraFanModes:
    """Fan mode normalisation for ELECTRA_AC devices."""

    def test_transforms_max_high_and_auto_max(self):
        handler = ElectraHandler()
        modes = ["auto", "min", "medium", "max_high", "auto_max"]
        result = handler.transform_fan_modes(modes)
        assert result == ["auto", "min", "medium", "high", "max"]

    def test_no_transform_without_both_markers(self):
        """Only transforms when BOTH max_high and auto_max are present."""
        handler = ElectraHandler()
        modes = ["auto", "low", "max_high"]
        assert handler.transform_fan_modes(modes) is modes

    def test_no_transform_normal_modes(self):
        handler = ElectraHandler()
        modes = ["auto", "low", "medium", "high"]
        assert handler.transform_fan_modes(modes) is modes

    def test_remap_fan_to_ir_passthrough(self):
        """ELECTRA send remapping is currently a no-op (upstream bug #184)."""
        handler = ElectraHandler()
        assert handler.remap_fan_to_ir("high") == "high"
        assert handler.remap_fan_to_ir("max") == "max"

    def test_capabilities(self):
        caps = ElectraHandler.capabilities()
        # ELECTRA uses default capabilities (no special toggles/presets)
        assert len(caps.extra_preset_modes) == 0
        assert not caps.has_raw_ir
