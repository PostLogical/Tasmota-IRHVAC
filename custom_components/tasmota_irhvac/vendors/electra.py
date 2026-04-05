"""Vendor handler for ELECTRA_AC devices.

Some ELECTRA_AC units report fan modes as "max_high" and "auto_max" in
their config.  This handler normalises them to standard HA fan mode names
during entity init.

Note: upstream climate.py also has send_ir remapping guarded by the same
HVAC_FAN_MAX_HIGH/AUTO_MAX check, but after init transforms the list those
guards are always false (upstream bug #184).  This handler preserves that
current effective behaviour — transform_fan_modes runs, remap_fan_to_ir
is a pass-through.
"""

from __future__ import annotations

from homeassistant.components.climate.const import FAN_HIGH

from ..const import HVAC_FAN_AUTO_MAX, HVAC_FAN_MAX, HVAC_FAN_MAX_HIGH
from . import register
from .base import VendorHandler


@register("ELECTRA_AC")
class ElectraHandler(VendorHandler):
    """ELECTRA_AC fan mode normalisation."""

    vendor_id = "ELECTRA_AC"

    def transform_fan_modes(self, fan_modes: list[str]) -> list[str]:
        """Normalise ELECTRA fan mode labels for the HA UI.

        max_high  → FAN_HIGH  ("high")
        auto_max  → HVAC_FAN_MAX ("max")
        """
        if (
            HVAC_FAN_MAX_HIGH not in fan_modes
            or HVAC_FAN_AUTO_MAX not in fan_modes
        ):
            return fan_modes

        result: list[str] = []
        for mode in fan_modes:
            if mode == HVAC_FAN_MAX_HIGH:
                result.append(FAN_HIGH)
            elif mode == HVAC_FAN_AUTO_MAX:
                result.append(HVAC_FAN_MAX)
            else:
                result.append(mode)
        return result
