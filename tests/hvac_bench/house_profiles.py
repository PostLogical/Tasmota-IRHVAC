"""Parameterized house archetypes for benchmark scenarios.

Each profile captures the thermal characteristics that matter for control:
time constant, HP effectiveness, and solar sensitivity. These are meant
to be representative, not exact — the goal is testing across a range of
building types, not modeling a specific house.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HPCapacityCurve:
    """Piecewise linear HP capacity factor vs outdoor temperature.

    Real air-source heat pumps lose capacity in cold (heating) and at high
    outdoor temps (cooling). Without this scaling, a fixed-gain bench
    underestimates how often the HP rails in cold and overestimates control
    authority during cold snaps — exactly the regime Tobit (#40) is meant
    to handle. NEEP cold-climate ASHP datasets show ~50–70% of rated
    heating capacity at design temp; standard ASHPs lose more. Cold-climate
    hyper-heat units continue operating below design at reduced capacity
    until a separate cutout temp.

    Heating curve (mode="heat"):
        outdoor ≤ heating_cutoff_t          → 0 (HP cannot heat at all)
        heating_cutoff_t .. heating_design_t → linear 0 → heating_design_factor
        heating_design_t .. heating_rated_t  → linear heating_design_factor → 1.0
        heating_rated_t  .. heating_mild_t   → linear 1.0 → heating_mild_factor
        outdoor ≥ heating_mild_t             → heating_mild_factor

    Cooling curve (mode="cool"): mirror with opposite slope.

    Defaults represent a typical residential ASHP rated at AHRI 7°C heating
    / 35°C cooling, with capacity zeroed at -15°C heating / 46°C cooling
    (heating_design_factor=0.0, heating_cutoff_t = heating_design_t — the
    cliff used pre-#49). Cold-climate / hyper-heat curves set
    heating_design_factor > 0 and a colder cutout.
    """
    heating_design_t: float = -15.0
    heating_rated_t: float = 7.0
    heating_mild_t: float = 20.0
    heating_mild_factor: float = 1.15
    # Capacity at design temp (0.0 = legacy cliff; CCASHPs typically 0.5-0.75).
    heating_design_factor: float = 0.0
    # Below this, capacity = 0. None ⇒ uses heating_design_t (legacy cliff).
    heating_cutoff_t: float | None = None
    cooling_mild_t: float = 18.0
    cooling_rated_t: float = 35.0
    cooling_design_t: float = 46.0
    cooling_mild_factor: float = 1.15

    @property
    def heating_cutoff(self) -> float:
        """Effective heating cutoff temp (below this, capacity = 0)."""
        return self.heating_design_t if self.heating_cutoff_t is None else self.heating_cutoff_t

    def factor(self, outdoor_c: float, mode: str = "heat") -> float:
        """Capacity factor (≥ 0) at the given outdoor temperature."""
        if mode == "heat":
            cutoff = self.heating_cutoff
            if outdoor_c <= cutoff:
                return 0.0
            if outdoor_c >= self.heating_mild_t:
                return self.heating_mild_factor
            if outdoor_c <= self.heating_design_t:
                # Below design, partial capacity (CCASHP / hyper-heat regime).
                span = self.heating_design_t - cutoff
                if span <= 0:
                    return 0.0
                return self.heating_design_factor * (outdoor_c - cutoff) / span
            if outdoor_c <= self.heating_rated_t:
                span = self.heating_rated_t - self.heating_design_t
                frac = (outdoor_c - self.heating_design_t) / span
                return self.heating_design_factor + frac * (1.0 - self.heating_design_factor)
            span = self.heating_mild_t - self.heating_rated_t
            frac = (outdoor_c - self.heating_rated_t) / span
            return 1.0 + frac * (self.heating_mild_factor - 1.0)
        # cooling
        if outdoor_c >= self.cooling_design_t:
            return 0.0
        if outdoor_c <= self.cooling_mild_t:
            return self.cooling_mild_factor
        if outdoor_c >= self.cooling_rated_t:
            span = self.cooling_design_t - self.cooling_rated_t
            return 1.0 - (outdoor_c - self.cooling_rated_t) / span
        span = self.cooling_rated_t - self.cooling_mild_t
        frac = (outdoor_c - self.cooling_mild_t) / span
        return self.cooling_mild_factor + frac * (1.0 - self.cooling_mild_factor)


# Default curves keyed for convenience.  STANDARD_HP_CAPACITY mirrors a
# typical residential ASHP (no cold-climate spec); COLD_CLIMATE_HP_CAPACITY
# represents a CCASHP that holds capacity well below design temp (NEEP
# cold-climate listing typical: 75% at -15°C, ~50% at -25°C).
STANDARD_HP_CAPACITY = HPCapacityCurve()

COLD_CLIMATE_HP_CAPACITY = HPCapacityCurve(
    heating_design_t=-25.0,
    heating_rated_t=7.0,
    heating_mild_t=20.0,
    heating_mild_factor=1.10,
)


# Fujitsu Halcyon hyper-heat AOU-RLF/XLTH series (NEEP cold-climate listing
# typical): rated 100% at +47°F (8.3°C), ~75% at +5°F (-15°C),
# ~50% at -15°F (-26°C, design point), continues operating with declining
# capacity to ~-25°F to -30°F (-32°C to -34°C) cutout.
# Modeled with heating_design_t=-26°C, heating_design_factor=0.5,
# heating_cutoff_t=-32°C — matches the user's deployed system class.
FUJITSU_HYPERHEAT_CAPACITY = HPCapacityCurve(
    heating_design_t=-26.0,
    heating_design_factor=0.5,
    heating_cutoff_t=-32.0,
    heating_rated_t=8.3,
    heating_mild_t=20.0,
    heating_mild_factor=1.10,
)


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
        hp_capacity: Optional capacity curve scaling hp_gain by outdoor temp.
            None = legacy fixed-gain behavior.
        description: What kind of building this represents.
    """
    name: str
    tau_minutes: float
    hp_gain: float
    description: str = ""
    hp_capacity: HPCapacityCurve | None = None


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
            concrete slab). Typical: 3-10 (Bacher & Madsen 2011 found
            C_s/C_i ≈ 5-10 for residential). The wall time constant
            seen from the wall side is τ_couple * mass_ratio.
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
    hp_capacity: HPCapacityCurve | None = None

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
        mass_ratio=8,
        hp_gain=0.025,
        description="Modern home, decent insulation, HP sized for -5°C design",
    ),
    "well_insulated": HouseProfile2R2C(
        name="Well Insulated",
        tau_env=250,
        tau_couple=150,
        mass_ratio=8,
        hp_gain=0.010,
        description="High-performance envelope, triple glazing, minimal infiltration",
    ),
    "heavy_masonry": HouseProfile2R2C(
        name="Heavy Masonry",
        tau_env=200,
        tau_couple=80,
        mass_ratio=10,
        hp_gain=0.012,
        description="Brick/concrete construction with high thermal mass",
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
        mass_ratio=8,
        hp_gain=0.04,
        description="100yo house, single-pane sunroom exposure, mini-split head. "
                    "Calibrated from 48h production data (Apr 2026). "
                    "mass_ratio reduced from 20→8 per Bacher & Madsen (C_s/C_i ≈ 5-10). "
                    "Design: 66°F at -15°F outdoor (marginal).",
    ),
    "bunkroom": HouseProfile2R2C(
        name="Bunkroom (calibrated)",
        tau_env=170,
        tau_couple=20,
        mass_ratio=8,
        hp_gain=0.025,
        description="100yo house, smaller zone, no direct solar. "
                    "Calibrated from 72h production data (Apr 2026). "
                    "mass_ratio reduced from 30→8 per Bacher & Madsen (C_s/C_i ≈ 5-10). "
                    "hp_gain 0.02→0.025: g×τ=4.25 (comparable to LR 4.0), "
                    "sp=28.9°C at -15°C outdoor (modestly undersized). "
                    "tau_env likely over-estimated — April data lacks cold-weather signal.",
    ),
    # dining_room: NOT calibrated — only 41 active heating ticks in Apr data.
    # HP barely ran (solar/stove heated the zone). Needs winter data.
}

# Capacity-curve variants of the calibrated profiles, for benches that need
# realistic cold-snap saturation (#43).  Same thermal/HP-gain parameters as
# the rated-conditions profiles above; the capacity curve scales hp_gain
# down as outdoor temp drops, so winter scenarios saturate more.  These are
# opt-in — existing tests using the non-capacity profiles are unchanged.
PROFILES_2R2C["living_room_capacity"] = HouseProfile2R2C(
    name="Living Room (calibrated, capacity curve)",
    tau_env=100,
    tau_couple=30,
    mass_ratio=8,
    hp_gain=0.04,
    description="living_room with STANDARD_HP_CAPACITY for cold-snap realism. "
                "Use when the test cares about HP saturation/Tobit-style "
                "censoring, not coefficient convergence under fixed gain.",
    hp_capacity=STANDARD_HP_CAPACITY,
)

PROFILES_2R2C["bunkroom_capacity"] = HouseProfile2R2C(
    name="Bunkroom (calibrated, capacity curve)",
    tau_env=170,
    tau_couple=20,
    mass_ratio=8,
    hp_gain=0.025,
    description="bunkroom with STANDARD_HP_CAPACITY for cold-snap realism. "
                "Already 'modestly undersized' at -15°C per calibration; "
                "with capacity curve, deep cold makes HP effectively zero.",
    hp_capacity=STANDARD_HP_CAPACITY,
)

# Fujitsu hyper-heat variants — match the deployed system class. CCASHP
# capacity holds 50% at -26°C design and runs down to -32°C cutout, instead
# of the cliff-at-design behavior of STANDARD_HP_CAPACITY.
PROFILES_2R2C["living_room_fujitsu"] = HouseProfile2R2C(
    name="Living Room (calibrated, Fujitsu hyper-heat capacity)",
    tau_env=100,
    tau_couple=30,
    mass_ratio=8,
    hp_gain=0.04,
    description="living_room with FUJITSU_HYPERHEAT_CAPACITY. Use for tests "
                "that need realistic CCASHP saturation under sub-design "
                "cold snaps without an abrupt cliff. Design temp -26°C "
                "(-15°F), cutoff -32°C.",
    hp_capacity=FUJITSU_HYPERHEAT_CAPACITY,
)

PROFILES_2R2C["bunkroom_fujitsu"] = HouseProfile2R2C(
    name="Bunkroom (calibrated, Fujitsu hyper-heat capacity)",
    tau_env=170,
    tau_couple=20,
    mass_ratio=8,
    hp_gain=0.025,
    description="bunkroom with FUJITSU_HYPERHEAT_CAPACITY. Already "
                "'modestly undersized' per calibration; CCASHP curve "
                "preserves partial capacity below -26°C instead of cliff.",
    hp_capacity=FUJITSU_HYPERHEAT_CAPACITY,
)
