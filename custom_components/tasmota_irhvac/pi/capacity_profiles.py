"""HP capacity profiles — outdoor-temperature-dependent effective capacity
multipliers used by greybox ID to correct outdoor-temp bias in identified ``k_c``.

## Why this exists

Heat pumps deliver capacity that varies substantially with outdoor temperature:
standard inverter HPs lose ~30–40% capacity by 5°F; cold-climate HPs (Mitsubishi
Hyper-Heat, Fujitsu Halcyon HFI, etc.) retain 70–75% at -15°F. Greybox
identification with a constant ``k_c`` averages this variation across the data
distribution, biasing the identified gain — high for warm-weather-dominated
data, low for cold. The bias flows downstream into FF coefficients, making
cold-snap response systematically under-predicted on a static-gain model.

Letting ``k_c`` modulate by outdoor temperature corrects this. The effective
gain becomes ``k_c_rated × profile.factor(T_out, mode)`` where ``k_c_rated`` is
identified from data (interpreted as the AHRI 47°F-equivalent heating gain or
95°F-equivalent cooling gain) and ``profile.factor`` is loaded from the HP's
manufacturer engineering data.

## Boundary behavior (IMPORTANT — see [[project-greybox-stage1-validation-probes]])

Profiles use piecewise-linear interpolation between anchor points sourced from
manufacturer engineering data sheets. **Outside the anchor range we clamp to
the boundary value rather than extrapolating.** This is the conservative choice:
- Below the lowest documented temperature (typically the manufacturer's stated
  cutoff): real HPs behave non-linearly here — some shut off entirely, some
  continue at sharply reduced capacity, defrost cycles dominate. Linear
  extrapolation from a few points would invent fantasy behavior.
- Above the highest documented temperature (typically the rating point or
  slightly above): similar issue at the upper end (some HPs cap capacity
  above rated outdoor temps; some have a "boost" zone).

**Improvement path when needed**: if a user reports operation below the
documented anchor range and we have evidence the real HP has measurable
capacity there, extend the anchor list using:
1. Manufacturer cold-weather supplemental data (if published)
2. NEEP cold-climate ASHP product listing (for certified cold-climate models)
3. Field measurement / empirical fit from buffer data (last resort)

Document any anchor extension with the source in a comment on the profile.

## Per-zone profile selection

User selects profile via the ``pi_hp_capacity_profile`` config option (per zone,
since multi-zone setups commonly have different outdoor units per zone group).
The user-installed Fujitsu setup has two outdoor units:
- LR: ``fujitsu_aou24rlxfwh`` (single-zone, paired with ASU24RLF head)
- DR/BR/NU: ``fujitsu_aou36rlxfzh`` (4-zone multi-split, capacity shared)

These are shipped as named profiles with the actual published data points
(2026-05-17 extraction; cross-referenced in [[user-profile]]).

## References

- AHRI 210/240 — rating standard (47°F heating, 95°F cooling)
- NEEP cold-climate ASHP product list: https://neep.org/heating-cooling/ccashp-specification-product-list
- Fujitsu 24RLXFW1 and AOU36RLXFZH Design & Technical Manuals (specific points
  copied into the FUJITSU_* profiles below; pdf-extracted 2026-05-17)
- ASHRAE 90.1-2019 Appendix G — generic residential HP curves (for the
  ``standard_inverter`` fallback)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# AHRI 210/240 standard rating points (the temperature at which ``factor = 1.0``).
AHRI_HEAT_RATING_C: float = 8.33    # 47°F
AHRI_COOL_RATING_C: float = 35.0    # 95°F

ProfileName = Literal[
    # Generic profiles for users without HP-specific data:
    "unknown",                   # flat factor = 1.0 (legacy constant-k_c behavior)
    "standard_inverter",         # ASHRAE 90.1 typical residential
    "cold_climate_inverter",     # NEEP cold-climate generic
    "fixed_speed",               # legacy on/off compressor
    # Per-unit profiles with manufacturer engineering data:
    "fujitsu_aou24rlxfwh",       # Fujitsu Halcyon HFI single-zone (LR setup)
    "fujitsu_aou36rlxfzh",       # Fujitsu Halcyon HFI 4-zone multi-split
]


@dataclass(frozen=True)
class CapacityCurve:
    """Piecewise-linear capacity ratio curve.

    ``anchor_temps_c`` and ``anchor_ratios`` are paired arrays; the ratio at
    ``temp_c`` is the linear interpolation between bracketing anchors. **Outside
    the anchor range we hold the boundary value — see module docstring for
    rationale.**

    Convention: ratio == 1.0 at the rating point (AHRI 47°F heating, 95°F cooling).
    """
    anchor_temps_c: tuple[float, ...]
    anchor_ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.anchor_temps_c) != len(self.anchor_ratios):
            raise ValueError("anchor_temps_c and anchor_ratios must have equal length")
        if len(self.anchor_temps_c) < 2:
            raise ValueError("need at least 2 anchor points")
        for i in range(1, len(self.anchor_temps_c)):
            if self.anchor_temps_c[i] <= self.anchor_temps_c[i - 1]:
                raise ValueError("anchor_temps_c must be strictly ascending")
        if any(r < 0 for r in self.anchor_ratios):
            raise ValueError("anchor_ratios must be >= 0")

    def ratio(self, temp_c: float) -> float:
        """Capacity ratio at ``temp_c``. Clamps to boundary values outside the
        anchor range — see module docstring."""
        t = self.anchor_temps_c
        r = self.anchor_ratios
        if temp_c <= t[0]:
            return r[0]
        if temp_c >= t[-1]:
            return r[-1]
        for i in range(1, len(t)):
            if temp_c <= t[i]:
                frac = (temp_c - t[i - 1]) / (t[i] - t[i - 1])
                return r[i - 1] + frac * (r[i] - r[i - 1])
        return r[-1]  # pragma: no cover — guarded above


# ── Generic curves (fallback when user doesn't know their HP) ───────────────

# ASHRAE 90.1-2019 / AHRI 210/240 lit-typical residential inverter HP.
# Heating: 1.0 at 47°F, drops to ~0.30 at -10°F. Cooling: mild degradation
# above 95°F.
_HEAT_STANDARD = CapacityCurve(
    anchor_temps_c=(-23.0, -15.0, -8.0,  0.0,  8.33, 17.0),
    anchor_ratios=(  0.30,  0.45, 0.60, 0.78, 1.00, 1.10),
)
_COOL_STANDARD = CapacityCurve(
    anchor_temps_c=(15.0,  25.0,  35.0,  43.0,  50.0),
    anchor_ratios=( 1.05,  1.02,  1.00,  0.92,  0.82),
)

# NEEP-certified cold-climate ASHP generic (Mitsubishi MZ-FS class).
# Used when user knows their HP is "cold-climate-rated" but doesn't have the
# specific spec sheet to hand. For users with specific Fujitsu/Mitsubishi
# models, the per-unit profiles below are more accurate.
_HEAT_COLD_CLIMATE_GENERIC = CapacityCurve(
    anchor_temps_c=(-30.0, -23.0, -15.0, -8.0,  0.0,  8.33, 17.0),
    anchor_ratios=(  0.50,  0.60,  0.75, 0.85, 0.92, 1.00, 1.05),
)

# Legacy fixed-speed (on/off compressor) — no variable-speed boost mode.
_HEAT_FIXED_SPEED = CapacityCurve(
    anchor_temps_c=(-15.0, -8.0,  0.0,  8.33, 17.0),
    anchor_ratios=(  0.20, 0.40, 0.65, 1.00, 1.05),
)
_COOL_FIXED_SPEED = CapacityCurve(
    anchor_temps_c=(15.0,  25.0,  35.0,  43.0,  50.0),
    anchor_ratios=( 1.00,  1.00,  1.00,  0.80,  0.65),
)


# ── Fujitsu engineering data ────────────────────────────────────────────────

# Fujitsu AOU24RLXFWH single-zone outdoor + ASU24RLF wall-mount head.
# Source: Fujitsu 24RLXFW1 Design & Technical Manual, Heating Capacity section,
# ASU24RLF table at 70°F indoor (pdf-extracted 2026-05-17). Cooling table from
# same source. Cutoff per spec sheet: -15°F (-26°C) — below this the HP shuts off.
# Cross-reference: see same data in tests/hvac_bench/house_profiles.py
# (FUJITSU_HYPERHEAT_CAPACITY) where it parameterizes the bench thermal model.
_HEAT_FUJITSU_AOU24 = CapacityCurve(
    anchor_temps_c=(-26.0,  -20.6, -15.0, -10.0,  -5.0,   0.0,   5.0,  8.33,  10.0,  15.0),
    anchor_ratios=( 0.54,   0.62,  0.70,  0.74,   0.81,  0.89,  0.97, 1.00,  1.02,  0.97),
)
# Cooling curve for AOU24 series — not in current spec extraction; using
# the generic _COOL_STANDARD as fallback until we extract cooling data from
# the same manual.  TODO: extract cooling-mode capacity table.
_COOL_FUJITSU_AOU24 = _COOL_STANDARD

# Fujitsu AOU36RLXFZH 4-zone-capable multi-split outdoor.
# Source: AOU36RLXFZH Design & Technical Manual, Heating Capacity section,
# 36 kBtu connecting-capacity row at 70°F indoor (pdf-extracted 2026-05-17).
# Notable: wide flat top (factor ~1.0 from 0°C to 15°C) and sharp cliff
# below -20°C. Piecewise-linear captures both because we have dense anchors.
_HEAT_FUJITSU_AOU36 = CapacityCurve(
    anchor_temps_c=(-26.0,  -20.6, -15.0, -10.0,  -5.0,   0.0,   5.0,  8.33,  10.0,  15.0),
    anchor_ratios=( 0.53,   0.60,  0.87,  0.92,   0.97,  1.00,  1.00, 1.00,  1.00,  1.00),
)
_COOL_FUJITSU_AOU36 = _COOL_STANDARD  # TODO: extract cooling table

# Flat ("unknown") curve — backwards compatibility with legacy constant-k_c.
_FLAT = CapacityCurve(
    anchor_temps_c=(-50.0, 50.0),
    anchor_ratios=(  1.00, 1.00),
)


# ── Profile registry ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CapacityProfile:
    """Pair of heating/cooling capacity curves selected by config name."""
    name: ProfileName
    heat_curve: CapacityCurve
    cool_curve: CapacityCurve

    def factor(self, temp_c: float, mode: str) -> float:
        """Capacity multiplier at ``temp_c`` for ``mode`` ('heat' or 'cool').
        Returns 1.0 for unrecognized modes (defensive)."""
        if mode == "heat":
            return self.heat_curve.ratio(temp_c)
        if mode == "cool":
            return self.cool_curve.ratio(temp_c)
        return 1.0  # pragma: no cover — defensive on unrecognized mode


_PROFILES: dict[ProfileName, CapacityProfile] = {
    "unknown": CapacityProfile(
        name="unknown",
        heat_curve=_FLAT,
        cool_curve=_FLAT,
    ),
    "standard_inverter": CapacityProfile(
        name="standard_inverter",
        heat_curve=_HEAT_STANDARD,
        cool_curve=_COOL_STANDARD,
    ),
    "cold_climate_inverter": CapacityProfile(
        name="cold_climate_inverter",
        heat_curve=_HEAT_COLD_CLIMATE_GENERIC,
        cool_curve=_COOL_STANDARD,
    ),
    "fixed_speed": CapacityProfile(
        name="fixed_speed",
        heat_curve=_HEAT_FIXED_SPEED,
        cool_curve=_COOL_FIXED_SPEED,
    ),
    "fujitsu_aou24rlxfwh": CapacityProfile(
        name="fujitsu_aou24rlxfwh",
        heat_curve=_HEAT_FUJITSU_AOU24,
        cool_curve=_COOL_FUJITSU_AOU24,
    ),
    "fujitsu_aou36rlxfzh": CapacityProfile(
        name="fujitsu_aou36rlxfzh",
        heat_curve=_HEAT_FUJITSU_AOU36,
        cool_curve=_COOL_FUJITSU_AOU36,
    ),
}


def get_profile(name: str) -> CapacityProfile:
    """Resolve a profile by name. Unknown names fall back to ``unknown``
    (flat curve, legacy constant-k_c behavior)."""
    return _PROFILES.get(name, _PROFILES["unknown"])  # type: ignore[arg-type]


def available_profile_names() -> tuple[str, ...]:
    """Names of all built-in profiles, for config validation."""
    return tuple(_PROFILES.keys())
