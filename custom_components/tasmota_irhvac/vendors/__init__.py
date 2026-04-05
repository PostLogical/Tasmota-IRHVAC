"""Vendor handler registry for Tasmota-IRHVAC.

Maps Tasmota vendor strings to VendorHandler subclasses.  Unknown vendors
get the default pass-through handler — no user is ever locked out.
"""

from __future__ import annotations

from .base import VendorHandler

_REGISTRY: dict[str, type[VendorHandler]] = {}


def register(vendor_key: str):
    """Class decorator — register a handler under an upper-cased vendor key.

    The key can be a prefix (e.g. "FUJITSU") to match FUJITSU_AC,
    FUJITSU_AC176, etc.  Exact matches are tried first.
    """

    def decorator(cls: type[VendorHandler]) -> type[VendorHandler]:
        _REGISTRY[vendor_key.upper()] = cls
        return cls

    return decorator


def get_handler(vendor: str) -> VendorHandler:
    """Look up a handler by Tasmota vendor string.

    Resolution order:
    1. Exact upper-case match
    2. Longest registered prefix that matches
    3. Default pass-through VendorHandler
    """
    key = vendor.upper()

    # Exact match
    if key in _REGISTRY:
        return _REGISTRY[key]()

    # Prefix match — longest wins (FUJITSU_AC beats FUJITSU if both exist)
    best: type[VendorHandler] | None = None
    best_len = 0
    for prefix, cls in _REGISTRY.items():
        if key.startswith(prefix) and len(prefix) > best_len:
            best = cls
            best_len = len(prefix)

    if best is not None:
        return best()

    return VendorHandler()


def get_handler_class(vendor: str) -> type[VendorHandler]:
    """Like get_handler but returns the class (for capabilities() before instantiation)."""
    key = vendor.upper()

    if key in _REGISTRY:
        return _REGISTRY[key]

    best: type[VendorHandler] | None = None
    best_len = 0
    for prefix, cls in _REGISTRY.items():
        if key.startswith(prefix) and len(prefix) > best_len:
            best = cls
            best_len = len(prefix)

    return best if best is not None else VendorHandler


# ── Known Tasmota IRHVAC vendors ─────────────────────────────────────
#
# From IRremoteESP8266 / Tasmota docs.  Used to populate the config flow
# dropdown.  custom_value=True on the SelectSelector means users can still
# type vendors not on this list.

KNOWN_VENDORS: list[str] = [
    "AIRTON",
    "AIRWELL",
    "AMCOR",
    "ARGO",
    "BOSCH144",
    "CARRIER_AC64",
    "COOLIX",
    "CORONA_AC",
    "DAIKIN",
    "DAIKIN128",
    "DAIKIN152",
    "DAIKIN160",
    "DAIKIN176",
    "DAIKIN2",
    "DAIKIN200",
    "DAIKIN216",
    "DAIKIN312",
    "DAIKIN64",
    "DELONGHI_AC",
    "ECOCLIM",
    "ELECTRA_AC",
    "FUJITSU_AC",
    "GOODWEATHER",
    "GREE",
    "HAIER_AC",
    "HAIER_AC160",
    "HAIER_AC176",
    "HAIER_AC_YRW02",
    "HITACHI_AC",
    "HITACHI_AC1",
    "HITACHI_AC264",
    "HITACHI_AC296",
    "HITACHI_AC344",
    "HITACHI_AC424",
    "KELON",
    "KELVINATOR",
    "LG",
    "LG2",
    "MIDEA",
    "MIRAGE",
    "MITSUBISHI112",
    "MITSUBISHI136",
    "MITSUBISHI_AC",
    "MITSUBISHI_HEAVY_152",
    "MITSUBISHI_HEAVY_88",
    "NEOCLIMA",
    "PANASONIC_AC",
    "PANASONIC_AC32",
    "RHOSS",
    "SAMSUNG_AC",
    "SANYO_AC",
    "SANYO_AC88",
    "SHARP_AC",
    "TCL112AC",
    "TECHNIBEL_AC",
    "TECO",
    "TEKNOPOINT",
    "TOSHIBA_AC",
    "TRANSCOLD",
    "TROTEC",
    "TROTEC_3550",
    "TRUMA",
    "VESTEL_AC",
    "VOLTAS",
    "WHIRLPOOL_AC",
    "YORK",
]


# ── Import handlers so @register decorators run ─────────────────────
from . import electra as _electra  # noqa: E402, F401
from . import fujitsu as _fujitsu  # noqa: E402, F401
