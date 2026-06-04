"""Parameterized house archetypes for benchmark scenarios.

Each profile captures the thermal characteristics that matter for control:
time constant, HP effectiveness, and solar sensitivity. These are meant
to be representative, not exact — the goal is testing across a range of
building types, not modeling a specific house.
"""

from dataclasses import dataclass

# 2026-06-03 Stage 1e refactor: the bench used to define its own HPCapacityCurve
# dataclass (single-design/rated/mild/cutoff parameterization, linear ramps
# between). That implementation has been superseded by production's
# CapacityProfile registry, which uses 10-anchor piecewise-linear curves sourced
# directly from manufacturer engineering data (Fujitsu D&T Manuals, NEEP CCASHP
# spec). The bench now imports from production so there's a single source of
# truth — earlier 3-anchor linear approximations of the Fujitsu curves diverged
# ~2% from the actual manufacturer table at intermediate temps (e.g. -10°C).
#
# API compatibility: production's CapacityProfile.factor(temp_c, mode) returns
# the same scalar multiplier as the old HPCapacityCurve.factor — drop-in for
# bench consumers (ThermalModel.step, scenario tests).
from custom_components.tasmota_irhvac.pi.capacity_profiles import (
    CapacityProfile,
    get_profile as _get_capacity_profile,
)

# Convenience references to production's registry entries. These replace the
# bench-local curve constants while preserving the names that scenario tests
# already import.
STANDARD_HP_CAPACITY: CapacityProfile = _get_capacity_profile("standard_inverter")
COLD_CLIMATE_HP_CAPACITY: CapacityProfile = _get_capacity_profile("cold_climate_inverter")
FUJITSU_HYPERHEAT_CAPACITY: CapacityProfile = _get_capacity_profile("fujitsu_aou24rlxfwh")
FUJITSU_AOU36RLXFZH_CAPACITY: CapacityProfile = _get_capacity_profile("fujitsu_aou36rlxfzh")


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
    hp_capacity: CapacityProfile | None = None


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
        solar_wall_fraction: Fraction of incoming solar gain absorbed by
            the wall/mass node (rest goes directly to air). Per ASHRAE
            Fundamentals Ch. 18 (RTS method, Table 14): the *absorbed*
            solar at the glass splits ~70% radiant / 30% convective; for
            *transmitted-beam* solar through unshaded glass the canonical
            treatment is 100% radiant (deposited on interior surfaces).
            Default 0.7 is conservative-residential (mixed glazing,
            partial shading); raise to ~0.9 for sun-exposed rooms with
            single-pane / sunroom direct beam. Bacher & Madsen 2011
            §2 don't fix this ratio — they treat solar A_w·Φ_s as an
            input that enters interior or sensor node depending on model
            order. Sources: ASHRAE F18 Ch.18 RTS; Bacher & Madsen 2011.
        description: What kind of building this represents.
    """
    name: str
    tau_env: float
    tau_couple: float
    mass_ratio: float
    hp_gain: float
    description: str = ""
    hp_capacity: CapacityProfile | None = None
    solar_wall_fraction: float = 0.7

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

# ── Literature-grounded 2R2C archetypes (parallel to PROFILES_1R1C) ──────
#
# Five 2R2C entries paralleling the 1R1C archetype dict, with each parameter
# grounded in Bacher-Madsen 2011 §5 typical-value ranges:
#   τ_env       — air-to-outdoor envelope time constant; inherits 1R1C τ
#                 (the 1R1C lumped tau collapses to envelope-air timescale
#                 in the air-node-only single-state model).
#   τ_couple    — air-to-wall surface coupling time constant; lands in the
#                 Bacher-Madsen TiTm 30–90 min range, lower for lightweight/
#                 timber surfaces, upper for masonry.
#   mass_ratio  — C_wall/C_air; Bacher-Madsen residential typical 5–10,
#                 below for very lightweight construction (timber/gypsum
#                 studio), at the upper bound for masonry (also matches
#                 ASHRAE F18 Ch.18 heavyweight category).
#   hp_gain     — matches the 1R1C value so the steady-state offset
#                 (FF seed = 1/(g·τ_env)) and HP authority at the air node
#                 are identical between the 1R1C and 2R2C archetypes.
#
# Use these for bench tests that need 2R2C dynamics without relying on the
# production-calibrated living_room/bunkroom entries (which hit Reynders
# 2014 identifiability rails — see project_lr_2r2c_calibration_2026_05).

PROFILES_2R2C["studio_apartment_2r2c"] = HouseProfile2R2C(
    name="Studio Apartment (2R2C lit-archetype)",
    tau_env=15,
    tau_couple=30,
    mass_ratio=3,
    hp_gain=0.08,
    description="Parallels PROFILES_1R1C['studio_apartment'] (τ=15, g=0.08). "
                "τ_env inherits the 1R1C τ. τ_couple at the lower bound of "
                "Bacher-Madsen 2011 §5 TiTm range (30 min) — thin partition "
                "walls, fast air-surface coupling. mass_ratio=3 sits below "
                "B-M residential 5–10 to reflect lightweight construction "
                "(timber framing, gypsum, minimal interior mass) per ASHRAE "
                "F18 Ch.18 lightweight category. Note: τ_env=15 is below "
                "B-M 'typical residential 1–3hr' — represents an "
                "atypically-leaky / small-volume archetype, intentionally.",
)

PROFILES_2R2C["drafty_bungalow_2r2c"] = HouseProfile2R2C(
    name="Drafty Bungalow (2R2C lit-archetype)",
    tau_env=25,
    tau_couple=40,
    mass_ratio=5,
    hp_gain=0.12,
    description="Parallels PROFILES_1R1C['drafty_bungalow'] (τ=25, g=0.12). "
                "τ_env inherits the 1R1C τ. τ_couple=40 within Bacher-Madsen "
                "2011 §5 TiTm 30–90 min range. mass_ratio=5 at the lower "
                "edge of B-M residential C_s/C_i ≈ 5–10 — older timber "
                "framing with plaster walls and wood floors but no "
                "significant masonry mass. τ_env=25 sits below B-M "
                "'typical residential 1–3hr' as expected for a drafty/"
                "uninsulated archetype.",
)

PROFILES_2R2C["standard_residential_2r2c"] = HouseProfile2R2C(
    name="Standard Residential (2R2C lit-archetype)",
    tau_env=50,
    tau_couple=60,
    mass_ratio=7,
    hp_gain=0.06,
    description="Parallels PROFILES_1R1C['standard_residential'] (τ=50, "
                "g=0.06). τ_env inherits the 1R1C τ. τ_couple=60 at the "
                "centre of Bacher-Madsen 2011 §5 TiTm 30–90 min range, "
                "typical for modern drywall/insulation surface coupling. "
                "mass_ratio=7 at the centre of B-M residential C_s/C_i ≈ "
                "5–10. Replaces the prior literature-typical entry "
                "(τ_env=150, g=0.05, added in 6249d18) with parallel-rule "
                "values matching the 1R1C archetype.",
)

PROFILES_2R2C["well_insulated_2r2c"] = HouseProfile2R2C(
    name="Well Insulated (2R2C lit-archetype)",
    tau_env=120,
    tau_couple=70,
    mass_ratio=8,
    hp_gain=0.025,
    description="Parallels PROFILES_1R1C['well_insulated'] (τ=120, g=0.025). "
                "τ_env inherits the 1R1C τ; sits in Bacher-Madsen 2011 §5 "
                "'typical residential 1–3hr' upper-mid. τ_couple=70 within "
                "B-M TiTm range. mass_ratio=8 at the upper-mid of B-M "
                "residential 5–10 — passive-house / ICF construction often "
                "pairs high envelope insulation with significant interior "
                "thermal mass for temperature stability.",
)

PROFILES_2R2C["heavy_masonry_2r2c"] = HouseProfile2R2C(
    name="Heavy Masonry (2R2C lit-archetype)",
    tau_env=150,
    tau_couple=90,
    mass_ratio=10,
    hp_gain=0.02,
    description="Parallels PROFILES_1R1C['heavy_masonry'] (τ=150, g=0.02). "
                "τ_env inherits the 1R1C τ; near the upper end of Bacher-"
                "Madsen 2011 §5 'typical residential 1–3hr'. τ_couple=90 "
                "at the upper bound of B-M TiTm 30–90 min — slow air-to-"
                "surface coupling through thick masonry. mass_ratio=10 at "
                "the upper bound of B-M residential 5–10, matching ASHRAE "
                "F18 Ch.18 heavyweight construction category (brick/"
                "concrete/CMU).",
)

# Quick subset for fast-running tests; parallels QUICK_PROFILES selection
# (drafty / standard / well_insulated) but indexes the lit-grounded 2R2C
# entries above.
QUICK_PROFILES_2R2C = {
    k: PROFILES_2R2C[k]
    for k in ["drafty_bungalow_2r2c", "standard_residential_2r2c", "well_insulated_2r2c"]
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
    name="Living Room (calibrated, AOU24RLXFWH capacity)",
    tau_env=100,
    tau_couple=30,
    mass_ratio=8,
    hp_gain=0.04,
    description="living_room with FUJITSU_HYPERHEAT_CAPACITY = AOU24RLXFWH "
                "single-zone outdoor (the LR deployed unit per user_profile.md). "
                "Linear capacity from 1.00 at +8.3°C rated to 0.54 at -26°C "
                "cutoff (extrapolated from D&T-manual table). Use this for "
                "tests that need realistic capacity derating in cold weather.",
    hp_capacity=FUJITSU_HYPERHEAT_CAPACITY,
)

PROFILES_2R2C["bunkroom_fujitsu"] = HouseProfile2R2C(
    name="Bunkroom (calibrated, AOU36RLXFZH multi-split capacity)",
    tau_env=170,
    tau_couple=20,
    mass_ratio=8,
    hp_gain=0.025,
    description="bunkroom with FUJITSU_AOU36RLXFZH_CAPACITY = AOU36RLXFZH "
                "4-zone-capable multi-split (the DR/BR/NU deployed unit per "
                "user_profile.md). Wide flat top down to ~+5°F then cliff "
                "drop; modeled by linear fit prioritizing cold-end accuracy "
                "(1.00 at +8.3°C → 0.53 at -26°C cutoff). Note: at the "
                "user's 36 kBtu head load, the AOU36 is at full nameplate "
                "in mild weather but derates faster than a single-zone unit.",
    hp_capacity=FUJITSU_AOU36RLXFZH_CAPACITY,
)

# Lit-grounded reference Fujitsu profile — use this when the deployed-zone
# Fujitsu profiles (living_room_fujitsu / bunkroom_fujitsu) aren't suitable
# as ground truth. The living_room envelope params are calibrated on 48h of
# bad data per feedback_living_room_calibration_suspect; the bunkroom
# envelope is also uncertain. This profile combines the lit-grounded
# standard_residential_2r2c envelope (Bacher-Madsen 2011 §5 centre values)
# with the AOU24RLXFWH manufacturer capacity curve — both ends of the
# combination are independently defensible.
PROFILES_2R2C["standard_residential_fujitsu"] = HouseProfile2R2C(
    name="Standard Residential (lit-archetype) with AOU24RLXFWH capacity",
    tau_env=50,
    tau_couple=60,
    mass_ratio=7,
    hp_gain=0.06,
    description="Reference Fujitsu mini-split profile for greybox τ_hp and "
                "capacity-profile validation. Envelope (τ_env=50/τ_couple=60/"
                "mr=7/g=0.06) inherits standard_residential_2r2c's lit-grounded "
                "Bacher-Madsen 2011 §5 centre values. Capacity curve is the "
                "AOU24RLXFWH cold-climate hyper-heat data (D&T-manual extraction "
                "2026-05-17). Prefer this over living_room_fujitsu / "
                "bunkroom_fujitsu when the test needs ground-truth envelope "
                "(per feedback_living_room_calibration_suspect, the deployed-"
                "zone envelope params were calibrated on 48h of bad data).",
    hp_capacity=FUJITSU_HYPERHEAT_CAPACITY,
)
