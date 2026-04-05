"""Tests for the vendor handler registry."""

import pytest

from custom_components.tasmota_irhvac.vendors import (
    KNOWN_VENDORS,
    get_handler,
    get_handler_class,
)
from custom_components.tasmota_irhvac.vendors.base import VendorHandler
from custom_components.tasmota_irhvac.vendors.electra import ElectraHandler
from custom_components.tasmota_irhvac.vendors.fujitsu import FujitsuHandler


class TestGetHandler:
    """Registry dispatch tests."""

    def test_exact_match_electra(self):
        handler = get_handler("ELECTRA_AC")
        assert isinstance(handler, ElectraHandler)

    def test_prefix_match_fujitsu_ac(self):
        handler = get_handler("FUJITSU_AC")
        assert isinstance(handler, FujitsuHandler)

    def test_prefix_match_fujitsu_ac176(self):
        handler = get_handler("FUJITSU_AC176")
        assert isinstance(handler, FujitsuHandler)

    def test_case_insensitive(self):
        assert isinstance(get_handler("fujitsu_ac"), FujitsuHandler)
        assert isinstance(get_handler("electra_ac"), ElectraHandler)

    def test_unknown_vendor_returns_default(self):
        handler = get_handler("SAMSUNG_AC")
        assert type(handler) is VendorHandler

    def test_empty_string_returns_default(self):
        handler = get_handler("")
        assert type(handler) is VendorHandler


class TestGetHandlerClass:
    """Class-level lookup for capabilities() before instantiation."""

    def test_returns_class_not_instance(self):
        cls = get_handler_class("FUJITSU_AC")
        assert cls is FujitsuHandler

    def test_unknown_returns_base(self):
        cls = get_handler_class("UNKNOWN_VENDOR")
        assert cls is VendorHandler


class TestKnownVendors:
    """Sanity checks on the vendor list."""

    def test_list_is_sorted(self):
        assert KNOWN_VENDORS == sorted(KNOWN_VENDORS)

    def test_contains_major_vendors(self):
        for vendor in ["FUJITSU_AC", "DAIKIN", "SAMSUNG_AC", "ELECTRA_AC", "GREE"]:
            assert vendor in KNOWN_VENDORS

    def test_no_duplicates(self):
        assert len(KNOWN_VENDORS) == len(set(KNOWN_VENDORS))


class TestDefaultHandler:
    """Base VendorHandler pass-through behaviour."""

    def test_transform_fan_modes_passthrough(self):
        handler = VendorHandler()
        modes = ["auto", "low", "high"]
        assert handler.transform_fan_modes(modes) is modes

    def test_remap_fan_to_ir_passthrough(self):
        handler = VendorHandler()
        assert handler.remap_fan_to_ir("high") == "high"

    def test_active_preset_is_none(self):
        handler = VendorHandler()
        assert handler.active_preset is None

    def test_should_pause_controller_false(self):
        handler = VendorHandler()
        assert handler.should_pause_controller is False

    def test_state_restore_is_none(self):
        handler = VendorHandler()
        assert handler.state_restore is None

    @pytest.mark.asyncio
    async def test_handle_preset_returns_none(self):
        handler = VendorHandler()
        from custom_components.tasmota_irhvac.vendors.base import EntityState
        result = await handler.handle_preset("away", EntityState(), lambda _: None)
        assert result is None
