"""Parameterized house archetypes for benchmark scenarios.

Each profile captures the thermal characteristics that matter for control:
time constant, HP effectiveness, and solar sensitivity. These are meant
to be representative, not exact — the goal is testing across a range of
building types, not modeling a specific house.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HouseProfile:
    """Thermal characteristics of a building zone (1R1C model).

    Attributes:
        name: Human-readable label.
        tau_minutes: Thermal time constant in minutes. Higher = more insulated.
            Determines how fast the room exchanges heat with outdoors.
        hp_gain: Heat pump effectiveness (°C/min per °C setpoint above equilibrium).
            Derived from tau: gain ≈ 1 / (tau * some_factor). Higher = smaller room
            or more powerful HP.
        description: What kind of building this represents.
    """
    name: str
    tau_minutes: float
    hp_gain: float
    description: str = ""


@dataclass(frozen=True)
class HouseProfile2R2C:
    """Two-node thermal model: air + wall/mass (2R2C).

    Models the room as two coupled thermal nodes:
    - Air node: heated directly by HP, loses heat to outdoors through
      envelope, exchanges heat with thermal mass. The sensor reads this.
    - Wall/mass node: walls, floors, furniture. Only exchanges heat with
      air. Provides thermal inertia that buffers temperature swings.

    ODEs (all time constants in minutes):
        dT_air/dt  = (T_out - T_air)/τ_env + hp_gain*(sp - T_air)
                     + (T_wall - T_air)/τ_couple + solar + stove
        dT_wall/dt = (T_air - T_wall)/(τ_couple * mass_ratio)

    Attributes:
        name: Human-readable label.
        tau_env: Envelope time constant (min). Air-to-outdoor through
            insulation, windows, infiltration. Lower = draftier.
            Typical: 30-60 min (drafty old house), 100-200 (modern).
        tau_couple: Air-wall coupling time constant (min). How fast air
            equilibrates with thermal mass. Lower = better coupling
            (e.g., exposed brick/concrete). Typical: 60-200 min.
        mass_ratio: C_wall/C_air. Ratio of wall thermal capacitance to
            air capacitance. Higher = more thermal mass (thick masonry,
            concrete slab). Typical: 5-20. The wall time constant seen
            from the wall side is τ_couple * mass_ratio.
        hp_gain: HP effectiveness (1/min). HP heating rate per °C of
            setpoint above room temp. Typical: 0.02-0.08.
        description: What kind of building this represents.
    """
    name: str
    tau_env: float
    tau_couple: float
    mass_ratio: float
    hp_gain: float
    description: str = ""

    @property
    def true_seed(self) -> float:
        """Physical FF seed from 2R2C steady-state: 1 / (hp_gain × τ_env).

        At steady state (dT_wall/dt=0 → T_wall=T_air=desired):
          sp = desired + (desired - outdoor) / (hp_gain × τ_env)
        So FF offset = seed × (desired - outdoor) where seed = 1/(hp_gain × τ_env).
        """
        return 1.0 / (self.hp_gain * self.tau_env)

    @property
    def tau_wall(self) -> float:
        """Wall-side time constant: τ_couple * mass_ratio."""
        return self.tau_couple * self.mass_ratio

    @property
    def tau_minutes(self) -> float:
        """Apparent system time constant for IMC gain scheduling.

        Returns the fast mode time constant, which is what the PI
        controller's tau estimator would measure from step responses.
        This is the relevant tau for Kp = τ/(λ+L) gain scheduling.
        """
        return self.fast_tau

    @property
    def fast_tau(self) -> float:
        """Fast mode time constant (approximate, dominated by air node)."""
        a11 = 1.0 / self.tau_env + self.hp_gain + 1.0 / self.tau_couple
        a22 = 1.0 / self.tau_wall
        tr = a11 + a22
        det = a11 * a22 - (1.0 / self.tau_couple) * (1.0 / self.tau_wall)
        disc = max(0.0, tr * tr - 4 * det)
        lam_fast = (tr + disc ** 0.5) / 2
        return 1.0 / lam_fast if lam_fast > 0 else 999.0

    @property
    def slow_tau(self) -> float:
        """Slow mode time constant (approximate, dominated by wall node)."""
        a11 = 1.0 / self.tau_env + self.hp_gain + 1.0 / self.tau_couple
        a22 = 1.0 / self.tau_wall
        tr = a11 + a22
        det = a11 * a22 - (1.0 / self.tau_couple) * (1.0 / self.tau_wall)
        disc = max(0.0, tr * tr - 4 * det)
        lam_slow = (tr - disc ** 0.5) / 2
        return 1.0 / lam_slow if lam_slow > 0 else 999.0


# ── Standard 1R1C profiles (legacy) ───────────────────────────────────────

PROFILES_1R1C = {
    "studio_apartment": HouseProfile(
        name="Studio Apartment",
        tau_minutes=15,
        hp_gain=0.08,
        description="Small volume, fast response, high solar exposure",
    ),
    "drafty_bungalow": HouseProfile(
        name="Drafty Bungalow",
        tau_minutes=25,
        hp_gain=0.12,
        description="Older construction, poor insulation, single story — HP sized for -5°C design",
    ),
    "standard_residential": HouseProfile(
        name="Standard Residential",
        tau_minutes=50,
        hp_gain=0.06,
        description="Typical modern home with decent insulation, HP sized for -5°C design",
    ),
    "well_insulated": HouseProfile(
        name="Well Insulated",
        tau_minutes=120,
        hp_gain=0.025,
        description="High-performance envelope, triple glazing, minimal infiltration",
    ),
    "heavy_masonry": HouseProfile(
        name="Heavy Masonry",
        tau_minutes=150,
        hp_gain=0.02,
        description="Brick/concrete construction with high thermal mass",
    ),
}

# ── Standard 2R2C profiles (archetype) ───────────────────────────────────
#
# Designed to cover the same range of building types as the 1R1C profiles
# but with physically decomposed parameters. Each profile can maintain
# ~20°C at -5°C outdoor with HP at 30°C.

PROFILES = {
    "studio_apartment": HouseProfile2R2C(
        name="Studio Apartment",
        tau_env=25,
        tau_couple=20,
        mass_ratio=3,
        hp_gain=0.08,
        description="Small volume, fast air response, high solar exposure",
    ),
    "drafty_bungalow": HouseProfile2R2C(
        name="Drafty Bungalow",
        tau_env=50,
        tau_couple=40,
        mass_ratio=5,
        hp_gain=0.06,
        description="Older construction, poor insulation, single story",
    ),
    "standard_residential": HouseProfile2R2C(
        name="Standard Residential",
        tau_env=100,
        tau_couple=80,
        mass_ratio=15,
        hp_gain=0.025,
        description="Modern home, decent insulation, HP sized for -5°C design",
    ),
    "well_insulated": HouseProfile2R2C(
        name="Well Insulated",
        tau_env=250,
        tau_couple=150,
        mass_ratio=15,
        hp_gain=0.010,
        description="High-performance envelope, triple glazing, minimal infiltration",
    ),
    "heavy_masonry": HouseProfile2R2C(
        name="Heavy Masonry",
        tau_env=200,
        tau_couple=80,
        mass_ratio=25,
        hp_gain=0.012,
        description="Brick/concrete construction with very high thermal mass",
    ),
}

# Subset for quick test runs
QUICK_PROFILES = {k: PROFILES[k] for k in ["drafty_bungalow", "standard_residential", "well_insulated"]}


# ── 2R2C profiles (production-calibrated) ─────────────────────────────

PROFILES_2R2C = {
    "living_room": HouseProfile2R2C(
        name="Living Room (calibrated)",
        tau_env=100,
        tau_couple=30,
        mass_ratio=20,
        hp_gain=0.04,
        description="100yo house, single-pane sunroom exposure, mini-split head. "
                    "Calibrated from 48h production data (Apr 2026). "
                    "Design: 66°F at -15°F outdoor (marginal).",
    ),
    "bunkroom": HouseProfile2R2C(
        name="Bunkroom (calibrated)",
        tau_env=170,
        tau_couple=20,
        mass_ratio=30,
        hp_gain=0.02,
        description="100yo house, smaller zone, no direct solar. "
                    "Calibrated from 72h production data (Apr 2026). "
                    "Design: 63°F at -15°F outdoor (undersized HP). "
                    "tau_env likely over-estimated — April data lacks cold-weather signal.",
    ),
    # dining_room: NOT calibrated — only 41 active heating ticks in Apr data.
    # HP barely ran (solar/stove heated the zone). Needs winter data.
}
