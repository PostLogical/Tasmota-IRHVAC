"""Load Condenser A Fujitsu telemetry bundle for Phase 4 empirical-tier fits.

Source bundle: `local/debug_bundles/condenser_a_data/` (built 2026-05-01).
Schema reference: that bundle's README.md.

Outputs a `ZoneTelemetry` dataclass per zone with:
- 5-min UTC-indexed pandas DataFrame
- Derived `hp_active` regressor (proxy for HP heating output)
- Derived `q_heat_proxy_w` regressor (`nominal_capacity_w` when active)
- Per-row `valid` mask reflecting signal completeness + documented exclusions

The valid mask combines:
1. Required-signal completeness (room_temp, hp_setpoint, mode all present)
2. Documented telemetry exclusion windows (recorder gap, controller pathologies)

Window constants (TRAIN_WINDOW, VALIDATE_WINDOW) and per-zone exclusion sets
come from `project_phase4_data_survey.md`. Update there first if revising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

ProxyVariant = Literal["constant", "setpoint_modulated"]

import pandas as pd


# ── Window definitions (UTC) ─────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    """Half-open [start, end) UTC time window."""

    start: datetime
    end: datetime
    label: str

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts < self.end


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


TRAIN_WINDOW: Final = Window(
    start=_utc(2026, 3, 21, 0, 0),
    end=_utc(2026, 4, 8, 0, 0),
    label="train_pre_health_sensor",
)

VALIDATE_WINDOW: Final = Window(
    start=_utc(2026, 4, 21, 4, 13),
    end=_utc(2026, 4, 29, 22, 0),
    label="validate_clean_window",
)

# Recorder retention loss — affects ALL zones, exclude unconditionally.
RECORDER_GAP: Final = Window(
    start=_utc(2026, 4, 10, 1, 0),
    end=_utc(2026, 4, 14, 23, 11),
    label="recorder_gap",
)

# DR/BR RLS gate-starvation pathology window. LR was unaffected.
GATE_STARVATION: Final = Window(
    start=_utc(2026, 4, 11, 0, 0),
    end=_utc(2026, 4, 15, 0, 0),
    label="dr_br_gate_starvation",
)

# LR-only double-beep window (sensor listener leak).
LR_DOUBLE_BEEP: Final = Window(
    start=_utc(2026, 4, 15, 17, 43),
    end=_utc(2026, 4, 18, 0, 0),
    label="lr_double_beep",
)


# ── Zone metadata ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ZoneInfo:
    name: str
    condenser: str  # "LR" (own) or "A" (shared)
    fit_target: bool  # False for nursery — outdoor-disturbance baseline only
    exclusion_windows: tuple[Window, ...] = ()


ZONE_REGISTRY: Final[dict[str, ZoneInfo]] = {
    "living_room": ZoneInfo(
        name="living_room",
        condenser="LR",
        fit_target=True,
        exclusion_windows=(LR_DOUBLE_BEEP,),
    ),
    "dining_room": ZoneInfo(
        name="dining_room",
        condenser="A",
        fit_target=True,
        exclusion_windows=(GATE_STARVATION,),
    ),
    "bunkroom": ZoneInfo(
        name="bunkroom",
        condenser="A",
        fit_target=True,
        exclusion_windows=(GATE_STARVATION,),
    ),
    "nursery": ZoneInfo(
        name="nursery",
        condenser="A",
        fit_target=False,  # mostly off; not a fit target
        exclusion_windows=(),
    ),
}


# ── Required signals ──────────────────────────────────────────────────────


REQUIRED_SIGNALS: Final = (
    "room_temp_c",
    "hp_setpoint_c",
    "mode",
    "outdoor_temp_c_om",
    "shortwave_w_m2",
)


# ── Derivation helpers ───────────────────────────────────────────────────


def derive_hp_active(
    mode: pd.Series,
    hp_setpoint_c: pd.Series,
    room_temp_c: pd.Series,
    *,
    deadband_c: float = 0.5,
) -> pd.Series:
    """Heating-active heuristic from the bundle README §Runtime/power signal.

    Returns boolean series: True when controller is plausibly calling for
    heat AND setpoint is above (room - deadband). NaN inputs → False.
    """
    in_heat = mode == "heat"
    setpoint_above = (hp_setpoint_c - room_temp_c) > -deadband_c
    return (in_heat & setpoint_above).fillna(False).astype(bool)


def derive_q_heat_proxy_w(
    hp_active: pd.Series,
    nominal_capacity_w: float,
    *,
    variant: ProxyVariant = "constant",
    hp_setpoint_c: pd.Series | None = None,
    room_temp_c: pd.Series | None = None,
    modulation_range_c: float = 5.0,
) -> pd.Series:
    """Heat-injection proxy magnitude when controller is calling.

    variants (per `project_phase4_data_survey.md`):
      - "constant" (option a, default): nominal magnitude when hp_active.
      - "setpoint_modulated" (option b): scales magnitude by
        clip((hp_setpoint - room_temp) / modulation_range_c, 0, 1) to
        approximate variable-speed compressor rate-modulation.

    Variant (c) free-parameter scaling — fitting nominal_capacity_w as a
    PEM parameter — is documented but not implemented (worsens identifiability).
    """
    if variant == "constant":
        return hp_active.astype(float) * nominal_capacity_w
    if variant == "setpoint_modulated":
        if hp_setpoint_c is None or room_temp_c is None:
            raise ValueError(
                "setpoint_modulated proxy requires hp_setpoint_c and room_temp_c"
            )
        delta_norm = (hp_setpoint_c - room_temp_c) / modulation_range_c
        modulation = delta_norm.clip(lower=0.0, upper=1.0).fillna(0.0)
        return hp_active.astype(float) * modulation * nominal_capacity_w
    raise ValueError(f"unknown proxy variant: {variant!r}")


# ── ZoneTelemetry ────────────────────────────────────────────────────────


@dataclass
class ZoneTelemetry:
    """One zone's loaded + derived telemetry, UTC-indexed at 5-min spacing.

    `df` columns:
      Source: room_temp_c, hp_setpoint_c, desired_temp_c, user_setpoint_c,
              mode, hvac_action, pi_integral, ff_offset, outdoor_temp_c_om,
              shortwave_w_m2, solar_gain_proxy, pellet_burning, boiler_calling
      Derived: hp_active (bool), q_heat_proxy_w (float), valid (bool)
    """

    info: ZoneInfo
    df: pd.DataFrame
    nominal_capacity_w: float
    bundle_path: Path
    applied_exclusions: tuple[Window, ...] = field(default_factory=tuple)

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_valid(self) -> int:
        return int(self.df["valid"].sum())

    @property
    def fraction_valid(self) -> float:
        return self.n_valid / self.n_rows if self.n_rows else 0.0


# ── Loading ──────────────────────────────────────────────────────────────


def load_condenser_a_zone(
    bundle_path: Path | str,
    zone: str,
    *,
    nominal_capacity_w: float = 3000.0,
    deadband_c: float = 0.5,
    proxy_variant: ProxyVariant = "constant",
    modulation_range_c: float = 5.0,
) -> ZoneTelemetry:
    """Load a single zone's CSV from the Condenser A bundle.

    Args:
        bundle_path: directory containing `<zone>.csv` files.
        zone: one of ZONE_REGISTRY keys.
        nominal_capacity_w: proxy heating capacity per indoor head.
            Fujitsu rated ~3 kW per head; can override per zone.
        deadband_c: passed to `derive_hp_active`.
        proxy_variant: heat-injection proxy variant per
            `project_phase4_data_survey.md`. Default "constant" preserves
            Phase 4 lite 2026-05-01 baseline.
        modulation_range_c: passed to setpoint_modulated variant.

    Returns ZoneTelemetry with hp_active, q_heat_proxy_w, and a default
    valid mask (signal completeness only — call apply_default_exclusions
    to layer on the documented exclusion windows).
    """
    info = ZONE_REGISTRY.get(zone)
    if info is None:
        valid = ", ".join(ZONE_REGISTRY)
        raise ValueError(f"Unknown zone {zone!r}; expected one of: {valid}")

    csv_path = Path(bundle_path) / f"{zone}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Bundle CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["ts_utc"])
    df = df.set_index("ts_utc").sort_index()

    # Verify all required signals present as columns
    missing = [c for c in REQUIRED_SIGNALS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    # Signal-completeness mask — all required signals non-null on this row
    completeness = pd.Series(True, index=df.index)
    for sig in REQUIRED_SIGNALS:
        completeness &= df[sig].notna()

    # Derive HP-active proxy + heat-injection magnitude
    hp_active = derive_hp_active(
        df["mode"],
        df["hp_setpoint_c"],
        df["room_temp_c"],
        deadband_c=deadband_c,
    )
    q_heat = derive_q_heat_proxy_w(
        hp_active,
        nominal_capacity_w,
        variant=proxy_variant,
        hp_setpoint_c=df["hp_setpoint_c"],
        room_temp_c=df["room_temp_c"],
        modulation_range_c=modulation_range_c,
    )

    out = df.copy()
    out["hp_active"] = hp_active
    out["q_heat_proxy_w"] = q_heat
    out["valid"] = completeness

    return ZoneTelemetry(
        info=info,
        df=out,
        nominal_capacity_w=nominal_capacity_w,
        bundle_path=Path(bundle_path),
        applied_exclusions=(),
    )


def apply_default_exclusions(telemetry: ZoneTelemetry) -> ZoneTelemetry:
    """Apply RECORDER_GAP + zone-specific exclusions to the valid mask.

    Returns a new ZoneTelemetry; the input is not mutated. Idempotent —
    re-applying the same exclusions doesn't change anything.
    """
    df = telemetry.df.copy()
    exclusions = (RECORDER_GAP,) + telemetry.info.exclusion_windows

    valid = df["valid"].copy()
    for window in exclusions:
        ts = df.index
        in_window = (ts >= window.start) & (ts < window.end)
        valid = valid & ~in_window
    df["valid"] = valid

    return ZoneTelemetry(
        info=telemetry.info,
        df=df,
        nominal_capacity_w=telemetry.nominal_capacity_w,
        bundle_path=telemetry.bundle_path,
        applied_exclusions=exclusions,
    )


def slice_window(
    telemetry: ZoneTelemetry,
    window: Window,
) -> ZoneTelemetry:
    """Restrict to a time window. Rows outside the window are dropped.

    Does not modify the valid mask further; combine with
    apply_default_exclusions for the full pipeline:
        load → apply_default_exclusions → slice_window(TRAIN_WINDOW)
    """
    df = telemetry.df
    mask = (df.index >= window.start) & (df.index < window.end)
    return ZoneTelemetry(
        info=telemetry.info,
        df=df.loc[mask].copy(),
        nominal_capacity_w=telemetry.nominal_capacity_w,
        bundle_path=telemetry.bundle_path,
        applied_exclusions=telemetry.applied_exclusions,
    )


def load_fit_zones(
    bundle_path: Path | str,
    *,
    window: Window = TRAIN_WINDOW,
    nominal_capacity_w: float = 3000.0,
    deadband_c: float = 0.5,
    proxy_variant: ProxyVariant = "constant",
    modulation_range_c: float = 5.0,
) -> dict[str, ZoneTelemetry]:
    """Load all fit-target zones (LR/DR/BR; NU excluded) for a window.

    Convenience entry point for Phase 4c per-zone fits. Applies the default
    exclusion pipeline (RECORDER_GAP + zone-specific) and slices the
    requested window.
    """
    out: dict[str, ZoneTelemetry] = {}
    for name, info in ZONE_REGISTRY.items():
        if not info.fit_target:
            continue
        loaded = load_condenser_a_zone(
            bundle_path,
            name,
            nominal_capacity_w=nominal_capacity_w,
            deadband_c=deadband_c,
            proxy_variant=proxy_variant,
            modulation_range_c=modulation_range_c,
        )
        cleaned = apply_default_exclusions(loaded)
        sliced = slice_window(cleaned, window)
        out[name] = sliced
    return out
