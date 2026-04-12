"""Parameterized house archetypes for benchmark scenarios.

Each profile captures the thermal characteristics that matter for control:
time constant, HP effectiveness, and solar sensitivity. These are meant
to be representative, not exact — the goal is testing across a range of
building types, not modeling a specific house.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HouseProfile:
    """Thermal characteristics of a building zone.

    Attributes:
        name: Human-readable label.
        tau_minutes: Thermal time constant in minutes. Higher = more insulated.
            Determines how fast the room exchanges heat with outdoors.
        hp_gain: Heat pump effectiveness (°C/min per °C setpoint above equilibrium).
            Derived from tau: gain ≈ 1 / (tau * some_factor). Higher = smaller room
            or more powerful HP.
        solar_gain: Solar sensitivity coefficient. Multiplied by solar proxy (0-1)
            to get °C/min of solar heating. Higher = more windows or south-facing.
        stove_gain: Supplemental heat source gain (°C/min when active).
        description: What kind of building this represents.
    """
    name: str
    tau_minutes: float
    hp_gain: float
    solar_gain: float = 0.3
    stove_gain: float = 0.15
    description: str = ""


# ── Standard profiles ─────────────────────────────────────────────────────

PROFILES = {
    "studio_apartment": HouseProfile(
        name="Studio Apartment",
        tau_minutes=15,
        hp_gain=0.08,
        solar_gain=0.6,
        description="Small volume, fast response, high solar exposure",
    ),
    "drafty_bungalow": HouseProfile(
        name="Drafty Bungalow",
        tau_minutes=25,
        hp_gain=0.12,
        solar_gain=0.5,
        description="Older construction, poor insulation, single story — HP sized for -5°C design",
    ),
    "standard_residential": HouseProfile(
        name="Standard Residential",
        tau_minutes=50,
        hp_gain=0.06,
        solar_gain=0.4,
        description="Typical modern home with decent insulation, HP sized for -5°C design",
    ),
    "well_insulated": HouseProfile(
        name="Well Insulated",
        tau_minutes=120,
        hp_gain=0.025,
        solar_gain=0.15,
        stove_gain=0.1,
        description="High-performance envelope, triple glazing, minimal infiltration",
    ),
    "heavy_masonry": HouseProfile(
        name="Heavy Masonry",
        tau_minutes=150,
        hp_gain=0.02,
        solar_gain=0.2,
        description="Brick/concrete construction with high thermal mass",
    ),
}

# Subset for quick test runs
QUICK_PROFILES = {k: PROFILES[k] for k in ["drafty_bungalow", "standard_residential", "well_insulated"]}
