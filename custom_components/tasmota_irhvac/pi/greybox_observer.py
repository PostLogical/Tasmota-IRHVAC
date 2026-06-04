"""Grey-box 1R1C energy balance observer and steady-state bridge.

Fits a physically grounded energy balance to the observation buffer and
optionally bridges its rate coefficients to WLS-compatible β for blending.
Runs alongside the existing WLS batch on the same 12h schedule.

The 1R1C energy balance divided by C_eff:

    dT_air/dt = c₀ + (UA/C) × (T_out - T_air) + (K_hp/C) × hp_offset + (α/C) × solar

Reparameterized as rate coefficients (Bacher & Madsen 2011):

    room_rate = c₀ + ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar

where c₀ absorbs unmodeled internal gains and measurement bias,
ua_c = UA/C, k_c = K_hp/C, α_c = α_solar/C.  These rate
coefficients are directly identifiable from derivative data without
the scaling ambiguity of the original parameterization.

Steady-state bridge (set dT/dt = 0, solve for hp_offset):

    hp_offset_eq = -(ua_c/k_c) × ΔT_outdoor - (α_c/k_c) × solar

This maps to WLS β: β₁ = -ua_c/k_c, β₂ = -α_c/k_c.  Standard errors
propagated via the delta method on the ratio.

Key advantage over the static WLS batch: uses ALL data including HP-off
periods (hp_offset=0), which directly inform ua_c and α_c from room
temperature trajectory.

References:
- Bacher & Madsen, "Identifying suitable models for the heat dynamics
  of buildings" (2011) -- grey-box RC models for building identification
- Madsen & Holst, "Estimation of continuous-time models for the heat
  dynamics of a building" (1995) -- rate coefficient parameterization
- Ljung, "System Identification: Theory for the User" -- prediction error,
  §16.4 cross-validation for model selection

Standalone module -- no Home Assistant dependencies.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .batch_learning import Observation
    from .capacity_profiles import CapacityProfile

from .capacity_profiles import get_profile as _get_capacity_profile

_LOGGER = logging.getLogger(__name__)

# Try to import scipy; gracefully degrade if unavailable.
try:
    from scipy.optimize import least_squares as _least_squares
    from scipy.linalg import expm as _expm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ── Parameter bounds (rate coefficients) ─────────────────────────────
#
# ua_c = UA/C in min⁻¹.  Typical residential:
#   τ = 60-300 min → ua_c = 1/300 to 1/60 = 0.003 to 0.017
#   2R2C slow mode can reach 20-100h (Bacher & Madsen 2011),
#   so lower bound accommodates τ up to 5000 min (83h).
UA_C_BOUNDS = (0.0002, 0.1)  # min⁻¹

# c0 = constant rate offset (°C/min).  Absorbs unmodeled internal gains,
# measurement bias, and other constant heat sources/sinks.
# ±0.05 °C/min ≈ ±3 °C/hr — generous for residential.
C0_BOUNDS = (-0.05, 0.05)  # °C/min

# k_c = K_hp/C in °C_room / (°C_setpoint_offset × min).
# Similar magnitude to ua_c for a well-sized HP.
K_C_BOUNDS = (0.001, 0.2)  # min⁻¹

# α_c = α_solar/C.  Positive because solar adds heat to the room.
# In the rate equation: room_rate = c0 + ua_c*(T_out-T_air) + k_c*hp + α_c*solar
# Higher solar → faster warming → α_c > 0.
# Magnitude depends on solar proxy scaling and C.
ALPHA_C_BOUNDS = (0.0, 1.0)  # °C/min per unit solar proxy

# 2R2C-only bounds.
# k_w = 1/τ_couple, air–wall coupling rate (min⁻¹).
# τ_couple ∈ ~10–600 min (bench profiles use 20–150).
K_W_BOUNDS = (0.001, 0.2)
# mass_ratio = C_wall/C_air. Bacher & Madsen (2011) report 5–10 typical
# residential; allow margins for fit, gate to a tighter range.
MASS_RATIO_BOUNDS = (1.0, 30.0)

# ── Bayesian priors for joint 2R2C fit (lit-grounded, configurable) ──
#
# Closed-loop operational residential data is partially under-determined
# for 2R2C ID (Reynders 2014 §IV; Annex 71 ST3): envelope (ua_c) and HP
# gain (k_c) are identifiable from typical weather variation, but wall
# mode (k_w, mass_ratio) and α_solar have weak observability without
# engineered excitation.
#
# Rather than hard-fixing wall mode at lit-typical values (the previous
# Stage-A approach, equivalent to a σ→0 prior), use INFORMED Bayesian
# priors with finite strength. This is the lit-canonical approach
# (CTSM-R MAP estimation, Madsen et al. 2013; Pathak et al. 2019
# "Estimating Buildings' Parameters over Time Including Prior Knowledge"
# demonstrates ±0.02 R-value accuracy on real Pecan Street smart-
# thermostat data using informed priors).
#
# The MAP estimate becomes a weighted average of MLE (data alone) and
# prior mean, with weights = inverse-variance. Strong data → MAP near
# MLE. Weak data → MAP near prior. For typical residential buildings
# (~95% of the population per Bacher-Madsen) the lit-typical wall mode
# prior matches reality closely; atypical buildings (deep masonry,
# passive house) require either user-supplied priors, transfer learning
# across seasons (posterior → next prior), or active-probe data (Phase
# 2 future work, see /Users/robbyg/.claude/plans/modular-cuddling-duckling.md).
#
# Tikhonov penalty form: ½ · (θ - μ)² / σ² appended to data residuals.
# scipy.optimize.least_squares minimizes the sum-of-squares of residuals;
# scaling each prior term by 1/σ embeds the inverse-variance weighting.

# c0: small bias offset around 0; data drives it on most buildings.
C0_PRIOR_MEAN = 0.0
C0_PRIOR_SIGMA = 0.01     # ±0.6°C/hr — generous, data dominates

# ua_c: envelope rate. Default prior centered at lit-typical (τ=200 min);
# at runtime, replaced with plant_id.tau_slow if confident (transfer-
# learning from plant ID, per Pathak §4.2). σ wide — data is informative.
UA_C_PRIOR_MEAN = 1.0 / 200.0
UA_C_PRIOR_SIGMA = 0.005   # covers τ from ~100 to ~400 min with 2σ

# k_c: HP gain rate. Bench profiles span 0.02-0.05; lit-typical 0.025-0.04.
K_C_PRIOR_MEAN = 0.03
K_C_PRIOR_SIGMA = 0.015     # covers 0.005-0.06 with 2σ

# alpha_c: solar gain rate. Wide range across buildings (window area,
# orientation, shading); keep prior loose so data can identify when
# weather provides enough solar/outdoor decoupling.
ALPHA_C_PRIOR_MEAN = 0.05
ALPHA_C_PRIOR_SIGMA = 0.05   # covers 0.0-0.15 with 2σ

# k_w: air↔wall coupling rate. The most under-determined from operational
# data (Reynders 2014, Annex 71 ST3). Tight prior at lit-typical 1/50
# (Bacher-Madsen "τ_couple ≈ 30-80 min residential"). σ small so prior
# dominates absent strong data; data can still nudge.
K_W_PRIOR_MEAN = 1.0 / 50.0    # τ_couple = 50 min
K_W_PRIOR_SIGMA = 0.005         # covers τ from ~30 to ~100 min with 2σ

# mass_ratio: C_wall/C_air. Also under-determined; tight prior at lit-
# typical 8 (Bacher-Madsen "5-10 typical residential").
MASS_RATIO_PRIOR_MEAN = 8.0
MASS_RATIO_PRIOR_SIGMA = 3.0    # covers 2-14 with 2σ

# ASHRAE Ch. 18 / bench convention: solar through a window splits ~30% to
# the air node (convective) and ~70% to the wall node (radiative). Fixed
# in v1; expose as config later if real-CSV validation flags it.
SOLAR_AIR_FRACTION = 0.3
SOLAR_WALL_FRACTION = 0.7

# Minimum observations required for a meaningful 1R1C fit.
MIN_OBSERVATIONS = 30

# 2R2C dispatch thresholds: need both enough data and enough time span
# to identify the wall mode + solar split. Below either, fall back to 1R1C.
MIN_OBSERVATIONS_2R2C = 1500
MIN_TIMESPAN_DAYS_2R2C = 14.0

# Minimum variance in hp_offset column to identify k_c.
MIN_HP_VARIANCE = 0.01


# ── HP dynamic state (Stage 1, 2026-06-03) ──────────────────────────────────
#
# τ_hp models the first-order lag of HP thermal output Q_hp toward its
# steady-state target k_c_eff·(sp−T_a), where k_c_eff is the capacity-
# profile-modulated gain at the current outdoor temperature.
#
# Lit support for the dynamic-HP construction (verified citations, see
# [[reference-hp-grey-box-lit]]):
#   * Tang et al. 2025 (Energy & Buildings, S0378778825008618): neglecting
#     HP cold-start/hot-start transients produces up to 4.9% seasonal-
#     performance bias — quantitative motivation for the third state.
#   * NIST Kim/Payne et al. 2023 ("Gray-Box Model of a Two-Stage Heat Pump
#     for Electrical Load Forecasting"): closest published analog, first-
#     order lag on residential HP (ducted, lag on electrical current).
#
# The mini-split ductless regime specifically is lit-thin — Tang & NIST are
# ducted. Mini-split has shorter τ_hp than ducted (no duct mass, no plenum),
# so the prior is biased low + wide:
#   μ = 3 min — typical inverter mini-split ramp; manufacturer engineering
#                data shows compressor 10→90% in 60–180s + ~1 min of
#                coil-air mixing to room.
#   σ = 4 min — wide because mini-split-specific lit is sparse; lets data
#                update vigorously when transients are present.
#
# Bounds (1, 30) min — below 1 min the static-gain model is fine; above
# 30 min would imply a hydronic-like emission system not present here.
#
# IDENTIFIABILITY WARNING (pre-registered, lit-consensus): Wang et al. NSF
# par/10304050 + Lin et al. arXiv:1512.08169 both argue passive thermostat
# operation gives insufficient excitation for HP-side time constants.
# **Expect τ_hp to rail under passive ID** — that is the lit-expected
# outcome and load-bearing evidence for the #138 active probe being
# necessary, NOT a Stage 1 failure. The mode-aware prior anchors τ_hp at
# the mini-split-typical value until #138 lands.
# See [[project-greybox-stage1-validation-probes]] for what tells us this
# is working vs failing.
TAU_HP_BOUNDS: tuple[float, float] = (1.0, 30.0)  # min
TAU_HP_PRIOR_MEAN: float = 3.0
TAU_HP_PRIOR_SIGMA: float = 4.0


# Lit-typical prior lookup. Used when ``PriorState`` has no promoted
# value for a given parameter (cold start, or that param was railed and
# never qualified for promotion).
#
# Names here are PARAMETER NAMES (mode-invariant for envelope params,
# base name for HP params). Mode-specific HP params (k_c_heat, k_c_cool,
# etc.) share the same lit defaults — the heat-mode k_c and cool-mode
# k_c have the same prior at cold start, then diverge as data arrives.
_LIT_PRIOR_MEAN: dict[str, float] = {
    "c0":         C0_PRIOR_MEAN,
    "ua_c":       UA_C_PRIOR_MEAN,
    "k_c":        K_C_PRIOR_MEAN,
    "alpha_c":    ALPHA_C_PRIOR_MEAN,
    "k_w":        K_W_PRIOR_MEAN,
    "mass_ratio": MASS_RATIO_PRIOR_MEAN,
    "tau_hp":     TAU_HP_PRIOR_MEAN,
}
_LIT_PRIOR_SIGMA: dict[str, float] = {
    "c0":         C0_PRIOR_SIGMA,
    "ua_c":       UA_C_PRIOR_SIGMA,
    "k_c":        K_C_PRIOR_SIGMA,
    "alpha_c":    ALPHA_C_PRIOR_SIGMA,
    "k_w":        K_W_PRIOR_SIGMA,
    "mass_ratio": MASS_RATIO_PRIOR_SIGMA,
    "tau_hp":     TAU_HP_PRIOR_SIGMA,
}
_BOUNDS_BY_NAME: dict[str, tuple[float, float]] = {
    "c0":         C0_BOUNDS,
    "ua_c":       UA_C_BOUNDS,
    "k_c":        K_C_BOUNDS,
    "alpha_c":    ALPHA_C_BOUNDS,
    "k_w":        K_W_BOUNDS,
    "mass_ratio": MASS_RATIO_BOUNDS,
    "tau_hp":     TAU_HP_BOUNDS,
}


@dataclass(frozen=True)
class PriorState:
    """Per-zone persistent priors for greybox parameters — Pathak (2019)
    §4.2 transfer learning: today's posterior is tomorrow's prior.

    Each parameter is stored as ``(mu, sigma)`` or ``None`` (cold start /
    never promoted). ``None`` entries fall back to lit-typical defaults
    via :meth:`mu_for` and :meth:`sigma_for`.

    ## Mode-invariant vs mode-specific parameters

    **Envelope params (mode-invariant)**: ``c0``, ``ua_c``, ``alpha_c``,
    ``k_w``, ``mass_ratio``. These describe building physics independent
    of whether the HP is heating or cooling — the same wall, the same
    insulation, the same thermal mass. Heat-mode fits update the same
    slots as cool-mode fits.

    **HP params (mode-specific)**: ``k_c`` and ``tau_hp`` are stored
    separately for heat and cool because the HP behaves differently
    by mode (different rated capacity, different ramp dynamics, defrost
    cycles in heating, etc.). Heat-mode fits update ``k_c_heat`` and
    ``tau_hp_heat``; cool-mode fits update ``k_c_cool`` and ``tau_hp_cool``.
    The two modes share envelope priors via the chain — what's learned
    about the wall in summer-cool informs winter-heat fits and vice versa.

    Lookup convention: :meth:`mu_for("ua_c")` (mode-invariant param —
    no mode arg) vs :meth:`mu_for("k_c", mode="heat")` (HP param —
    mode required).

    Promotion is governed by :func:`promote_posterior` with lit-grounded
    gates (no railed params per Reynders 2014; least_squares converged).
    Bridge gates are explicitly NOT a promotion criterion — they protect
    the FF coefficient path, which is a separate downstream consumer.
    The dataclass itself is immutable — promotion returns a new instance.

    Serialization is via :meth:`to_dict` / :meth:`from_dict`. Schema
    handles backwards-compatible migration from the pre-2026-06-03
    single-mode format (legacy ``k_c`` field migrates into ``k_c_heat``,
    consistent with the heating-dominant bench corpus).
    """

    # Envelope (mode-invariant)
    c0:         tuple[float, float] | None = None
    ua_c:       tuple[float, float] | None = None
    alpha_c:    tuple[float, float] | None = None
    k_w:        tuple[float, float] | None = None
    mass_ratio: tuple[float, float] | None = None
    # HP — heating mode
    k_c_heat:    tuple[float, float] | None = None
    tau_hp_heat: tuple[float, float] | None = None
    # HP — cooling mode
    k_c_cool:    tuple[float, float] | None = None
    tau_hp_cool: tuple[float, float] | None = None
    n_promotions: int = 0

    # Param name conventions:
    #   _ENVELOPE_PARAMS: mode-invariant; slot name == param name
    #   _HP_PARAMS: mode-specific; slot name == f"{param}_{mode}"
    _ENVELOPE_PARAMS = ("c0", "ua_c", "alpha_c", "k_w", "mass_ratio")
    _HP_PARAMS = ("k_c", "tau_hp")

    @classmethod
    def _resolve_slot(cls, name: str, mode: str | None) -> str:
        """Map a parameter name + optional mode to the dataclass slot name.

        Envelope params don't take a mode (raises if one is passed for an
        envelope param — caller bug). HP params require a mode."""
        if name in cls._ENVELOPE_PARAMS:
            return name
        if name in cls._HP_PARAMS:
            if mode not in ("heat", "cool"):
                raise ValueError(
                    f"mode 'heat' or 'cool' required for HP param {name!r}; got {mode!r}"
                )
            return f"{name}_{mode}"
        raise ValueError(f"unknown PriorState parameter: {name!r}")

    def mu_for(self, name: str, mode: str | None = None) -> float:
        """Return persisted μ for ``name``, or the lit-typical default if
        this parameter has never been promoted. ``mode`` required for HP
        params; ignored for envelope params."""
        slot = self._resolve_slot(name, mode)
        val = getattr(self, slot)
        return val[0] if val is not None else _LIT_PRIOR_MEAN[name]

    def sigma_for(self, name: str, mode: str | None = None) -> float:
        """Return persisted σ for ``name``, or the lit-typical default."""
        slot = self._resolve_slot(name, mode)
        val = getattr(self, slot)
        return val[1] if val is not None else _LIT_PRIOR_SIGMA[name]

    def is_promoted(self, name: str, mode: str | None = None) -> bool:
        """True iff this parameter has a persisted posterior."""
        slot = self._resolve_slot(name, mode)
        return getattr(self, slot) is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict. Schema includes both envelope and mode-specific
        slots; ``None`` entries are omitted."""
        out: dict[str, Any] = {"n_promotions": self.n_promotions}
        all_slots = (
            *self._ENVELOPE_PARAMS,
            "k_c_heat", "tau_hp_heat",
            "k_c_cool", "tau_hp_cool",
        )
        for slot in all_slots:
            val = getattr(self, slot)
            if val is not None:
                out[slot] = [float(val[0]), float(val[1])]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PriorState":
        """Restore from serialized form. ``None`` or empty → cold-start defaults.

        Backwards-compatible migration from pre-2026-06-03 format:
        - Legacy ``k_c`` field → migrates into ``k_c_heat`` (bench was
          heating-only; production users are heating-dominant in winter
          installations and the chain re-converges on cool when cool batches arrive).
        - Legacy persistence had no ``tau_hp`` (HP dynamics not modeled);
          ``tau_hp_heat`` and ``tau_hp_cool`` start fresh (cold-start).

        Malformed entries for a single parameter are silently dropped (that
        param falls back to lit default)."""
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}

        # Legacy migration: old ``k_c`` → ``k_c_heat``.  Only applied if the
        # new field isn't already present (avoid clobbering a properly-saved
        # heat-mode value).
        if "k_c" in data and "k_c_heat" not in data:
            try:
                mu, sigma = float(data["k_c"][0]), float(data["k_c"][1])
                if math.isfinite(mu) and math.isfinite(sigma) and sigma > 0:
                    kwargs["k_c_heat"] = (mu, sigma)
            except (TypeError, ValueError, IndexError):
                pass

        all_slots = (
            *cls._ENVELOPE_PARAMS,
            "k_c_heat", "tau_hp_heat",
            "k_c_cool", "tau_hp_cool",
        )
        for slot in all_slots:
            if slot in kwargs:  # already set by legacy migration
                continue
            entry = data.get(slot)
            if entry is None:
                continue
            try:
                mu, sigma = float(entry[0]), float(entry[1])
                if math.isfinite(mu) and math.isfinite(sigma) and sigma > 0:
                    kwargs[slot] = (mu, sigma)
            except (TypeError, ValueError, IndexError):
                continue
        n = data.get("n_promotions", 0)
        try:
            kwargs["n_promotions"] = max(0, int(n))
        except (TypeError, ValueError):
            kwargs["n_promotions"] = 0
        return cls(**kwargs)


def promote_posterior(
    result: "GreyboxResult",
    current_state: PriorState,
    *,
    mode: str,
    least_squares_converged: bool = True,
) -> PriorState:
    """Pathak (2019) §4.2 transfer learning: promote a passing batch's
    posterior to next batch's prior.

    ``mode`` ("heat" or "cool") routes HP-related parameters (``k_c``,
    ``tau_hp``) to the mode-specific slot. Envelope params (``c0``,
    ``ua_c``, ``alpha_c``, ``k_w``, ``mass_ratio``) update the shared
    mode-invariant slots regardless of which mode produced this batch.

    Lit-grounded promotion criteria (all must hold for the batch as a
    whole; per-parameter gates apply below):

      1. ``result.is_2r2c`` — only 2R2C fits expose all envelope params
      2. ``least_squares_converged`` — standard PEM gate (Ljung 1999)
      3. ``param_std_err`` populated — Jacobian-derived posterior σ
         must exist for the parameters we want to promote

    Per-parameter (Reynders 2014, "rails are diagnosis"):
      - A parameter at (within 1% of) either of its bounds is non-
        identifiable; the artificially-tight σ from the bound clamp is
        not a true posterior. **That parameter is not promoted** — its
        existing prior survives. Other parameters from the same fit may
        still promote independently.
      - σ must be finite and > 0.

    Note: ``bridge.gates_passed`` is deliberately NOT a criterion here.
    Those gates calibrate the WLS β-flow downstream (β_outdoor=-ua_c/k_c
    going into the WLS regression); they're tight by design for that
    consumer. The prior chain consumes the raw envelope parameters
    directly, where Reynders rail detection + wide-σ-on-disagreement
    are the correct safeguards. Smoke-test evidence (2026-06-03):
    lit-grounded parameter recovery within 7-14% of truth produced
    0/120 bridge gates passed — gating promotion on that would have
    blocked the chain entirely despite excellent envelope identification.

    Returns a new :class:`PriorState` (the input is not mutated). If no
    parameter qualifies, returns ``current_state`` unchanged.
    """
    if mode not in ("heat", "cool"):
        raise ValueError(f"mode must be 'heat' or 'cool'; got {mode!r}")
    if not least_squares_converged:
        return current_state
    if not result.is_2r2c:
        return current_state
    std_err = result.param_std_err or {}
    if not std_err:
        return current_state

    values: dict[str, float | None] = {
        "c0":         result.c0,
        "ua_c":       result.ua_c,
        "k_c":        result.k_c,
        "alpha_c":    result.alpha_c,
        "k_w":        result.k_w,
        "mass_ratio": result.mass_ratio,
        "tau_hp":     result.tau_hp,
    }

    # Updates keyed by SLOT NAME (not param name) so envelope and mode-
    # specific routing is resolved here, not when reconstructing PriorState.
    updates: dict[str, tuple[float, float]] = {}
    for name, value in values.items():
        if value is None:
            continue
        sigma = std_err.get(name)
        if sigma is None or not math.isfinite(sigma) or sigma <= 0:
            continue
        lo, hi = _BOUNDS_BY_NAME[name]
        # Rail detection: within 1% of bound span counts as railed. Reynders
        # 2014 § non-identifiability: railed params have artificially tight
        # σ from the bound clamp, not a true posterior.
        rail_tol = max((hi - lo) * 0.01, 1e-9)
        if value <= lo + rail_tol or value >= hi - rail_tol:
            continue
        # HP params route to the mode-specific slot; envelope params keep
        # their slot name (which equals their param name).
        slot = (
            f"{name}_{mode}" if name in PriorState._HP_PARAMS else name
        )
        updates[slot] = (float(value), float(sigma))

    if not updates:
        return current_state

    # Build new immutable state. Updated slots take new values; non-updated
    # slots carry over from current_state.
    all_slots = (
        *PriorState._ENVELOPE_PARAMS,
        "k_c_heat", "tau_hp_heat",
        "k_c_cool", "tau_hp_cool",
    )
    kwargs = {
        slot: updates.get(slot, getattr(current_state, slot))
        for slot in all_slots
    }
    kwargs["n_promotions"] = current_state.n_promotions + 1
    return PriorState(**kwargs)


@dataclass
class GreyboxResult:
    """Result of a grey-box energy balance fit (1R1C or 2R2C).

    1R1C is the default. When ``is_2r2c`` is True, ``k_w``, ``mass_ratio``,
    ``tau_fast`` and ``tau_slow`` carry the wall-node identification;
    ``alpha_c`` holds the steady-state-effective total (α_air + α_wall),
    so the existing bridge formulas (β_outdoor = -ua_c/k_c, β_solar =
    -alpha_c/k_c) work unchanged.
    """

    n_observations: int  # observations used in fit
    n_hp_on: int  # observations where HP was active
    n_hp_off: int  # observations where HP was off (hp_offset=0)

    # Fitted rate coefficients (÷ C_eff)
    c0: float  # constant rate offset (°C/min) — internal gains / bias
    ua_c: float  # UA/C envelope rate (min⁻¹)
    k_c: float  # K_hp/C HP gain rate (min⁻¹)
    alpha_c: float  # α_solar/C solar rate (0.0 if no solar input)

    # Derived
    tau_eff: float  # 1/ua_c (minutes) -- effective time constant (1R1C)
    residual_rms: float  # RMS of energy balance residual (°C/min)

    # Fit quality
    cost: float  # scipy cost function value
    n_function_evals: int  # number of function evaluations

    # Plant ID cross-check
    plant_tau_slow: float | None = None  # τ_slow from plant ID (minutes)
    tau_agreement_pct: float | None = None  # |τ_eff - τ_slow| / τ_slow × 100

    # Per-parameter confidence (from Jacobian if available)
    param_std_err: dict[str, float] = field(default_factory=dict)

    # 2R2C extension (None for 1R1C results)
    is_2r2c: bool = False
    k_w: float | None = None         # 1/τ_couple, air–wall coupling rate (min⁻¹)
    mass_ratio: float | None = None  # C_wall/C_air
    tau_fast: float | None = None    # natural fast eigenvalue (min)
    tau_slow: float | None = None    # natural slow eigenvalue (min)

    # HP dynamic state (Stage 1, 2026-06-03): first-order lag time constant
    # for the modeled inverter-HP ramp. ``None`` until the 3-state fit lands;
    # at that point this carries the fitted τ_hp (minutes). Required by
    # :func:`promote_posterior` so the mode-specific ``tau_hp_heat`` /
    # ``tau_hp_cool`` slots in :class:`PriorState` can be updated.
    tau_hp: float | None = None

    # Median observation interval — used by cadence-adaptive gates so the
    # residual_rms threshold scales with the data's actual sample rate.
    # None means "unknown / fallback to legacy threshold."
    dt_median_min: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "n_hp_on": self.n_hp_on,
            "n_hp_off": self.n_hp_off,
            "c0": self.c0,
            "ua_c": self.ua_c,
            "k_c": self.k_c,
            "alpha_c": self.alpha_c,
            "tau_eff": self.tau_eff,
            "residual_rms": self.residual_rms,
            "cost": self.cost,
            "n_function_evals": self.n_function_evals,
            "plant_tau_slow": self.plant_tau_slow,
            "tau_agreement_pct": self.tau_agreement_pct,
            "param_std_err": self.param_std_err,
            "is_2r2c": self.is_2r2c,
            "k_w": self.k_w,
            "mass_ratio": self.mass_ratio,
            "tau_fast": self.tau_fast,
            "tau_slow": self.tau_slow,
            "tau_hp": self.tau_hp,
            "dt_median_min": self.dt_median_min,
        }


def find_solar_entity(model_inputs: list[dict[str, Any]]) -> str | None:
    """Find the entity_id of the solar proxy input, or None."""
    for m in model_inputs:
        if m.get("input_role") == "solar":
            return m.get("entity_id")
    return None


def fit_greybox(
    observations: list[Observation],
    model_inputs: list[dict[str, Any]],
    plant_tau_slow: float | None = None,
    plant_tau_slow_confidence: float = 0.0,
    prior_state: PriorState | None = None,
    *,
    capacity_profile_name: str = "unknown",
    mode: str = "heat",
) -> GreyboxResult | None:
    """Fit a grey-box energy balance to observation buffer data.

    Dispatches between 1R1C and 2R2C based on observation count and
    time span.  When the buffer is rich enough (≥ MIN_OBSERVATIONS_2R2C
    samples spanning ≥ MIN_TIMESPAN_DAYS_2R2C days), attempts 2R2C with
    the 1R1C result as a warm start.  Falls back to 1R1C if 2R2C fails
    to converge.

    The bridge formulas (β_outdoor = -ua_c/k_c, β_solar = -α_c/k_c) are
    identical at steady state for 1R1C and 2R2C, so downstream consumers
    are unchanged.  The 2R2C win is correct identification of ua_c and
    α_total (which 1R1C confounds with wall thermal mass).

    Args:
        observations: full buffer contents (all ticks, not pre-filtered).
        model_inputs: model input config dicts (to identify solar proxy).
        plant_tau_slow: τ_slow from plant ID (minutes), if available.
        plant_tau_slow_confidence: confidence in τ_slow (0-1).
        prior_state: Pathak §4.2 transfer-learning prior state (mode-aware,
            envelope shared across heat/cool, k_c/τ_hp per-mode).
        capacity_profile_name: HP capacity-profile config name. The 2R2C
            fit applies the profile to modulate ``k_c`` by outdoor
            temperature (k_c_eff = k_c_rated · profile.factor(T_out, mode)).
            ``"unknown"`` (default) is a flat profile → legacy constant-k_c.
            See ``capacity_profiles.py`` for the registry.
        mode: HVAC mode ("heat" or "cool") for the buffer being fit.
            Determines which capacity curve applies and which PriorState
            HP slot (k_c_heat vs k_c_cool, tau_hp_heat vs tau_hp_cool)
            is used as prior + promoted on success. Envelope params are
            shared across modes.

    Returns:
        GreyboxResult (1R1C or 2R2C) or None if insufficient data or
        scipy unavailable.
    """
    if not SCIPY_AVAILABLE:
        _LOGGER.info(
            "Grey-box observer: scipy not available. "
            "Install scipy for advanced energy balance fitting."
        )
        return None

    # Filter to observations with valid outdoor temperature.
    eligible = [o for o in observations if o.outdoor_temp_c is not None]
    # GreyboxBuffer.get_all() returns storage-slot order; SlevPolicy eviction
    # overwrites slots in place (batch_learning.py:1248,1284), so post-fill
    # the slot order no longer tracks wall-clock order. The 2R2C dispatch
    # gate (timespan_days below), _compute_dt_median_min, and the sim-error
    # PEM residual loop (dt = ts[i] - ts[i-1]) all require chronological
    # ordering — without this sort, eviction produces catastrophic param
    # collapse to bounds. project_buffer_fill_collapse_bug.md (#139).
    eligible.sort(key=lambda o: o.timestamp)

    if len(eligible) < MIN_OBSERVATIONS:
        _LOGGER.debug(
            "Grey-box observer: insufficient observations (%d < %d)",
            len(eligible), MIN_OBSERVATIONS,
        )
        return None

    # Always try 1R1C first — it's cheap and gives a warm start for 2R2C.
    r_1r1c = _fit_greybox_1r1c(eligible, model_inputs, plant_tau_slow)
    if r_1r1c is None:
        return None

    # 2R2C dispatch: needs both enough data and enough time span.
    timespan_days = (eligible[-1].timestamp - eligible[0].timestamp) / 86400.0
    if (len(eligible) < MIN_OBSERVATIONS_2R2C
            or timespan_days < MIN_TIMESPAN_DAYS_2R2C):
        _LOGGER.debug(
            "Grey-box: 2R2C dispatch skipped (n=%d, span=%.1f d) — using 1R1C",
            len(eligible), timespan_days,
        )
        return r_1r1c

    # Attempt 2R2C with 1R1C warm start. Capacity profile + mode are
    # threaded through to the 3-state fit so k_c is modulated by outdoor
    # temperature and the mode-specific prior slots are used.
    capacity_profile = _get_capacity_profile(capacity_profile_name)
    r_2r2c = _fit_greybox_2r2c(
        eligible, model_inputs, r_1r1c, plant_tau_slow,
        prior_state=prior_state,
        capacity_profile=capacity_profile,
        mode=mode,
    )
    if r_2r2c is None:
        _LOGGER.info("Grey-box: 2R2C fit failed — falling back to 1R1C")
        return r_1r1c

    return r_2r2c


def _compute_dt_median_min(eligible: list[Observation]) -> float | None:
    """Median observation interval in minutes, or None if not computable.

    Used by cadence-adaptive gates. Returns the most-common dt (mode of
    rounded values to 0.1 min) since real-world streams have small jitter
    around a nominal cadence.
    """
    if len(eligible) < 2:
        return None
    dts: list[float] = []
    for i in range(1, len(eligible)):
        d = (eligible[i].timestamp - eligible[i - 1].timestamp) / 60.0
        if d > 0:
            dts.append(round(d, 1))
    if not dts:
        return None
    # Modal dt — robust to occasional gaps from missed observations.
    # Counter.most_common breaks count ties by first-encountered order,
    # which is deterministic; ``max(set(...), key=...)`` would tie-break
    # by set iteration order, which depends on PYTHONHASHSEED.
    return Counter(dts).most_common(1)[0][0]


def _fit_greybox_1r1c(
    eligible: list[Observation],
    model_inputs: list[dict[str, Any]],
    plant_tau_slow: float | None = None,
) -> GreyboxResult | None:
    """Fit a 1R1C energy balance.

        room_rate = c₀ + ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar

    Bacher & Madsen (2011) parameterization.  Rate coefficients are directly
    identifiable from derivative data; bounded to physically plausible
    ranges.  Caller has already filtered observations.
    """
    solar_entity = find_solar_entity(model_inputs)

    # Extract arrays for the fit.
    m = len(eligible)
    room_rate = [o.room_rate for o in eligible]
    t_air = [o.current_c for o in eligible]
    t_out: list[float] = [o.outdoor_temp_c for o in eligible]  # type: ignore[misc]

    # HP offset: setpoint - room temp when HP is on, 0 when off.
    hp_offset: list[float] = []
    n_hp_on = 0
    n_hp_off = 0
    for o in eligible:
        if (o.clamped_reason == "no_output" or o.hp_setpoint is None
                or o.hp_contribution_uncertain):
            hp_offset.append(0.0)
            n_hp_off += 1
        else:
            hp_offset.append(o.hp_setpoint - o.current_c)
            n_hp_on += 1

    # Solar proxy values (from raw_readings by entity_id).
    solar: list[float]
    if solar_entity is not None:
        solar = [o.raw_readings.get(solar_entity, 0.0) for o in eligible]
    else:
        solar = [0.0] * m

    has_solar = solar_entity is not None and any(abs(s) > 1e-6 for s in solar)

    # Check HP offset variance -- need some HP-on data to identify k_c.
    hp_mean = sum(hp_offset) / m
    hp_var = sum((h - hp_mean) ** 2 for h in hp_offset) / m
    fix_k_c = hp_var < MIN_HP_VARIANCE

    # ── Build residual function ──────────────────────────────────────
    #
    # room_rate[i] = ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar
    #
    # Parameters: [ua_c, k_c, alpha_c] or subsets if fixed/absent.

    if fix_k_c:
        # Fix k_c at a reasonable default (use plant ID K if available).
        # With k_c fixed, residual only fits c0, ua_c and optionally α_c.
        k_c_fixed = 0.01  # ~τ=100min default
        if has_solar:
            def residual_fn(params: list[float]) -> list[float]:
                c0, ua_c, alpha_c = params
                return [
                    room_rate[i]
                    - c0
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c_fixed * hp_offset[i]
                    - alpha_c * solar[i]
                    for i in range(m)
                ]
            x0 = [0.0, 0.01, 0.05]
            lower = [C0_BOUNDS[0], UA_C_BOUNDS[0], ALPHA_C_BOUNDS[0]]
            upper = [C0_BOUNDS[1], UA_C_BOUNDS[1], ALPHA_C_BOUNDS[1]]
            param_names = ["c0", "ua_c", "alpha_c"]
        else:
            def residual_fn(params: list[float]) -> list[float]:
                c0, ua_c = params
                return [
                    room_rate[i]
                    - c0
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c_fixed * hp_offset[i]
                    for i in range(m)
                ]
            x0 = [0.0, 0.01]
            lower = [C0_BOUNDS[0], UA_C_BOUNDS[0]]
            upper = [C0_BOUNDS[1], UA_C_BOUNDS[1]]
            param_names = ["c0", "ua_c"]
    else:
        if has_solar:
            def residual_fn(params: list[float]) -> list[float]:
                c0, ua_c, k_c, alpha_c = params
                return [
                    room_rate[i]
                    - c0
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c * hp_offset[i]
                    - alpha_c * solar[i]
                    for i in range(m)
                ]
            x0 = [0.0, 0.01, 0.02, 0.05]
            lower = [C0_BOUNDS[0], UA_C_BOUNDS[0], K_C_BOUNDS[0], ALPHA_C_BOUNDS[0]]
            upper = [C0_BOUNDS[1], UA_C_BOUNDS[1], K_C_BOUNDS[1], ALPHA_C_BOUNDS[1]]
            param_names = ["c0", "ua_c", "k_c", "alpha_c"]
        else:
            def residual_fn(params: list[float]) -> list[float]:
                c0, ua_c, k_c = params
                return [
                    room_rate[i]
                    - c0
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c * hp_offset[i]
                    for i in range(m)
                ]
            x0 = [0.0, 0.01, 0.02]
            lower = [C0_BOUNDS[0], UA_C_BOUNDS[0], K_C_BOUNDS[0]]
            upper = [C0_BOUNDS[1], UA_C_BOUNDS[1], K_C_BOUNDS[1]]
            param_names = ["c0", "ua_c", "k_c"]

    # Use plant ID τ_slow to set initial ua_c if available.
    # τ = 1/ua_c → ua_c = 1/τ  (ua_c is always at index 1, after c0)
    if plant_tau_slow is not None and plant_tau_slow > 0:
        ua_c_init = 1.0 / plant_tau_slow
        ua_c_init = max(UA_C_BOUNDS[0], min(UA_C_BOUNDS[1], ua_c_init))
        x0[1] = ua_c_init

    # ── Fit ───────────────────────────────────────────────────────────

    try:
        result = _least_squares(
            residual_fn, x0,
            bounds=(lower, upper),
            method="trf",
            loss="huber",  # robust to outliers
            f_scale=0.005,  # residual scale for Huber (°C/min)
        )
    except Exception:
        _LOGGER.exception("Grey-box observer: least_squares failed")
        return None

    # Extract fitted parameters.
    params = dict(zip(param_names, result.x))
    c0 = params.get("c0", 0.0)
    ua_c = params.get("ua_c", 0.01)
    k_c = params.get("k_c", k_c_fixed if fix_k_c else 0.02)
    alpha_c = params.get("alpha_c", 0.0)

    # Derived quantities.
    tau_eff = 1.0 / ua_c if ua_c > 1e-8 else float("inf")
    residual_rms = math.sqrt(2.0 * result.cost / m) if m > 0 else 0.0

    # Per-parameter standard errors from Jacobian.
    std_err: dict[str, float] = {}
    if result.jac is not None:  # pragma: no branch — scipy.least_squares populates jac on convergence
        try:
            import numpy as np
            J = result.jac
            n_params = len(param_names)
            sigma2 = 2.0 * result.cost / max(1, m - n_params)
            JtJ_inv = np.linalg.pinv(J.T @ J) * sigma2
            for i, name in enumerate(param_names):
                var = JtJ_inv[i, i]
                std_err[name] = math.sqrt(max(0.0, float(var)))
        except Exception:
            pass

    # Plant ID cross-check.
    tau_agreement = None
    if plant_tau_slow is not None and plant_tau_slow > 0 and not math.isinf(tau_eff):
        tau_agreement = abs(tau_eff - plant_tau_slow) / plant_tau_slow * 100.0

    return GreyboxResult(
        n_observations=m,
        n_hp_on=n_hp_on,
        n_hp_off=n_hp_off,
        c0=float(c0),
        ua_c=float(ua_c),
        k_c=float(k_c),
        alpha_c=float(alpha_c),
        tau_eff=float(tau_eff),
        residual_rms=float(residual_rms),
        cost=float(result.cost),
        n_function_evals=int(result.nfev),
        plant_tau_slow=plant_tau_slow,
        tau_agreement_pct=float(tau_agreement) if tau_agreement is not None else None,
        param_std_err=std_err,
        dt_median_min=_compute_dt_median_min(eligible),
    )


def _natural_eigenvalues(
    ua_c: float,
    k_w: float,
    mass_ratio: float,
) -> tuple[float, float]:
    """Return (τ_fast, τ_slow) of the HP-off natural 2R2C system.

    A_natural = [[-(ua_c + k_w),  k_w           ],
                 [ k_w/mass_ratio, -k_w/mass_ratio]]

    Eigenvalues λ = -1/τ; we return positive τ_fast < τ_slow (minutes).
    """
    a11 = -(ua_c + k_w)
    a22 = -k_w / mass_ratio
    a12 = k_w
    a21 = k_w / mass_ratio
    tr = a11 + a22
    det = a11 * a22 - a12 * a21
    disc = max(0.0, tr * tr - 4.0 * det)
    sq = math.sqrt(disc)
    # Both eigenvalues are negative (decay). |λ_fast| > |λ_slow|.
    lam_fast = (tr - sq) / 2.0  # most negative
    lam_slow = (tr + sq) / 2.0
    tau_fast = -1.0 / lam_fast if lam_fast < -1e-12 else float("inf")
    tau_slow = -1.0 / lam_slow if lam_slow < -1e-12 else float("inf")
    return tau_fast, tau_slow


def _fit_greybox_2r2c(
    eligible: list[Observation],
    model_inputs: list[dict[str, Any]],
    r_1r1c: GreyboxResult,
    plant_tau_slow: float | None = None,
    *,
    prior_state: PriorState | None = None,
    capacity_profile: "CapacityProfile | None" = None,
    mode: str = "heat",
) -> GreyboxResult | None:
    """Fit a 2R2C+HP grey-box energy balance with dynamic HP state.

    State: ``x = [T_a, T_w, Q_hp]`` (3-state).

    Continuous dynamics (Stage 1, 2026-06-03; see ``project-greybox-hp-dynamic-state``):

        dT_a/dt  = c₀ + ua_c·(T_out − T_a) + Q_hp + α_air·solar
                   + k_w·(T_w − T_a)
        dT_w/dt  = (k_w/mass_ratio)·(T_a − T_w) + (α_wall/mass_ratio)·solar
        dQ_hp/dt = (k_c_eff·(sp − T_a) − Q_hp) / τ_hp     [HP active]
                 = -Q_hp / τ_hp                            [HP inactive, snap to 0]

    where ``k_c_eff(T_out, mode) = k_c_rated · capacity_profile.factor(T_out, mode)``.
    ``k_c_rated`` is the fitted parameter — interpreted as the AHRI rating-point
    capacity (47°F heating / 95°F cooling). Below/above rating, the profile
    modulates delivery to match manufacturer engineering data.

    HP-off behavior (option B, locked 2026-06-03 conversation): Q_hp snaps to
    0 at the end of each tick where HP is off. Reason: Fujitsu wall-mount
    mini-splits close vanes + stop blower on idle, so residual coil delivery
    is ~0.1% of active — modeling decay would CLAIM heat delivery that
    doesn't happen and bias ua_c/c0. **This is a modeling choice for the
    ductless-mini-split regime, not a cited construction** — see
    [[reference-hp-grey-box-lit]] for why lit papers using decay (Tang,
    NIST, Sourbron) are inapplicable (ducted/hydronic).

    The 3×3 A matrix is upper block-triangular (Q_hp's row couples only to
    itself in A; the (sp − T_a) term enters via ZOH input b, so T_a → Q_hp
    feedback is held constant over each ZOH interval). This makes a closed-
    form 3×3 expm achievable cheaply by composing the 2×2 envelope expm
    with scalar exp(−dt/τ_hp), but we use scipy.linalg.expm for now per
    design Decision 4 (~6 μs vs ~1 μs closed-form). Optimize later if bench
    shows it matters.

    Per-mode parameters (locked Decision 3): the fit uses mode-specific
    priors for k_c and τ_hp (heat vs cool — different 4-way valve / defrost
    dynamics) and shared envelope priors (ua_c, k_w, mass_ratio, c0,
    alpha_c — the building is the same regardless of HP mode). Promotion
    routes k_c/τ_hp to the mode-specific PriorState slot.

    Solar split fixed at SOLAR_AIR_FRACTION / SOLAR_WALL_FRACTION (ASHRAE).
    α_total is the fitted parameter; α_air = 0.3·α_total, α_wall = 0.7·α_total.

    Bridge formulas at steady state are unchanged from 1R1C:
      β_outdoor = -ua_c / k_c_eff
      β_solar   = -α_total / k_c_eff   (the 30/70 split cancels at SS)
    where k_c_eff is evaluated at the desired operating-point T_out.

    Identifiability: τ_hp is expected to rail under passive operational
    data (Wang NSF + Lin arXiv pre-registration). The mode-aware prior
    anchors it at the mini-split-typical 3 min until #138 active probe
    lands; railing is evidence for the probe, not a Stage 1 failure.

    Args:
        capacity_profile: HP capacity curve registry entry. ``None`` falls
            back to the flat ``"unknown"`` profile (legacy constant-k_c
            behavior with τ_hp dynamics still active).
        mode: ``"heat"`` or ``"cool"``. Determines which capacity curve
            applies and which mode-specific prior slots are used. Must
            match a value the PriorState accepts.
    """
    solar_entity = find_solar_entity(model_inputs)

    m = len(eligible)
    timestamps = [o.timestamp for o in eligible]
    room_rate = [o.room_rate for o in eligible]
    t_air = [o.current_c for o in eligible]
    t_out: list[float] = [o.outdoor_temp_c for o in eligible]  # type: ignore[misc]

    # Sim-error PEM residual needs hp_setpoint (raw, not offset) when active
    # so HP feedback can enter the A matrix as -k_c·T_a + k_c·sp.
    hp_setpoint_arr: list[float | None] = []
    n_hp_on = 0
    n_hp_off = 0
    for o in eligible:
        if (o.clamped_reason == "no_output" or o.hp_setpoint is None
                or o.hp_contribution_uncertain):
            hp_setpoint_arr.append(None)
            n_hp_off += 1
        else:
            hp_setpoint_arr.append(float(o.hp_setpoint))
            n_hp_on += 1

    if solar_entity is not None:
        solar = [o.raw_readings.get(solar_entity, 0.0) for o in eligible]
    else:
        solar = [0.0] * m
    has_solar = solar_entity is not None and any(abs(s) > 1e-6 for s in solar)

    # PE check: HP must vary across observations or k_c is not separable
    # from ua_c. Use observed (sp - T_a) variance as a proxy (same logic as
    # the 1R1C rate-residual path).
    hp_offset_proxy = [
        (sp - t_air[i]) if sp is not None else 0.0
        for i, sp in enumerate(hp_setpoint_arr)
    ]
    hp_mean = sum(hp_offset_proxy) / m
    hp_var = sum((h - hp_mean) ** 2 for h in hp_offset_proxy) / m
    if hp_var < MIN_HP_VARIANCE:
        # Without HP-on data we can't separate k_c from ua_c — same constraint
        # as 1R1C. Fall back rather than fit a degenerate model.
        return None

    # Pre-compute dt[i] = (t[i] - t[i-1]) / 60 (minutes); first entry unused.
    dt_min: list[float] = [0.0] * m
    for i in range(1, m):
        d = (timestamps[i] - timestamps[i - 1]) / 60.0
        dt_min[i] = d if d > 0 else 0.0
    # Cache the typical dt so the matrix-exponential cost is paid once per
    # parameter eval, not once per observation.
    nonzero_dts = [d for d in dt_min if d > 0]
    # Use Counter.most_common for deterministic tiebreaking (see
    # ``_estimate_median_dt_minutes`` comment for the PYTHONHASHSEED reason).
    typical_dt = (Counter(nonzero_dts).most_common(1)[0][0]
                  if nonzero_dts else 0.0)

    # Parameter packing — v2 Bayesian joint fit. ALL 6 RC parameters are
    # FREE; closed-loop identifiability gaps on wall mode are addressed
    # by informed Bayesian priors (Tikhonov penalty appended to residuals
    # below), not by hard-fixing as the previous Stage-A did.
    # CTSM-R MAP estimation (Madsen et al. 2013) / Pathak 2019 §3 — the
    # MAP optimum is the inverse-variance-weighted average of likelihood
    # (data) and prior, so wall mode stays near lit-typical absent strong
    # data, but data can nudge when it's informative.
    # Prior values come from PriorState (Pathak §4.2 transfer-learning
    # chain). Lit-typical defaults apply for parameters that haven't been
    # promoted yet (cold start) or that failed promotion (e.g. railed).
    # The Tikhonov penalty (θ-μ)/σ stays identical in form; only the
    # numbers change.
    ps = prior_state if prior_state is not None else PriorState()
    # Mode-aware prior lookup: HP params (k_c, tau_hp) need the mode; envelope
    # params don't. Caller passes mode via the fit_greybox dispatcher; PriorState
    # validates mode in mu_for/sigma_for and raises ValueError on bad input.
    fit_mode = mode

    def _prior_mu(name: str) -> float:
        return ps.mu_for(name, mode=fit_mode) if name in PriorState._HP_PARAMS else ps.mu_for(name)

    def _prior_sigma(name: str) -> float:
        return ps.sigma_for(name, mode=fit_mode) if name in PriorState._HP_PARAMS else ps.sigma_for(name)

    # Capacity profile: flat "unknown" fallback preserves legacy constant-k_c
    # behavior when caller doesn't specify a profile.
    profile = capacity_profile if capacity_profile is not None else _get_capacity_profile("unknown")

    # Parameter ordering: append tau_hp at the end (Decision 5 — diff
    # minimization vs grouping with k_c; no readability difference).
    if has_solar:
        param_names = ["c0", "ua_c", "k_c", "alpha_c", "k_w", "mass_ratio", "tau_hp"]
        lower = [
            C0_BOUNDS[0], UA_C_BOUNDS[0], K_C_BOUNDS[0], ALPHA_C_BOUNDS[0],
            K_W_BOUNDS[0], MASS_RATIO_BOUNDS[0], TAU_HP_BOUNDS[0],
        ]
        upper = [
            C0_BOUNDS[1], UA_C_BOUNDS[1], K_C_BOUNDS[1], ALPHA_C_BOUNDS[1],
            K_W_BOUNDS[1], MASS_RATIO_BOUNDS[1], TAU_HP_BOUNDS[1],
        ]
        prior_mu = [_prior_mu(n) for n in param_names]
        prior_sigma = [_prior_sigma(n) for n in param_names]
    else:
        param_names = ["c0", "ua_c", "k_c", "k_w", "mass_ratio", "tau_hp"]
        lower = [
            C0_BOUNDS[0], UA_C_BOUNDS[0], K_C_BOUNDS[0],
            K_W_BOUNDS[0], MASS_RATIO_BOUNDS[0], TAU_HP_BOUNDS[0],
        ]
        upper = [
            C0_BOUNDS[1], UA_C_BOUNDS[1], K_C_BOUNDS[1],
            K_W_BOUNDS[1], MASS_RATIO_BOUNDS[1], TAU_HP_BOUNDS[1],
        ]
        prior_mu = [_prior_mu(n) for n in param_names]
        prior_sigma = [_prior_sigma(n) for n in param_names]

    # Plant ID transfer-learning: if plant_tau_slow is confidently known,
    # replace the default ua_c prior with one centered at 1/tau_slow.
    # Pathak 2019 §4.2 "Prior selection & transfer learning" — earlier
    # season's posterior becomes next season's prior. Plant ID does its
    # own probing, so its τ_slow estimate carries observational evidence
    # that informs greybox without needing to re-probe.
    ua_c_init = r_1r1c.ua_c
    if plant_tau_slow is not None and plant_tau_slow > 0:
        ua_c_init = max(UA_C_BOUNDS[0], min(UA_C_BOUNDS[1], 1.0 / plant_tau_slow))
        # Update ua_c prior mean (index 1 in param_names) to plant ID's
        # estimate; keep σ at default (data can still override).
        prior_mu[1] = ua_c_init

    # k_c warm-start rescale: r_1r1c.k_c is identified at the data's mean
    # outdoor temp, so it reflects the AVERAGE delivered capacity over the
    # buffer. The 3-state model identifies k_c_RATED (the AHRI rating-point
    # value); these differ by the inverse of the profile factor evaluated
    # at the mean operating T_out. Without this rescale, the optimizer
    # starts low and has to traverse a large parameter range — adding fit
    # iterations and risking premature convergence to a local minimum.
    mean_t_out = sum(t_out) / m if m > 0 else 0.0
    profile_mean = profile.factor(mean_t_out, fit_mode)
    if profile_mean > 1e-6:
        k_c_rated_warm = r_1r1c.k_c / profile_mean
    else:
        k_c_rated_warm = r_1r1c.k_c

    if has_solar:
        x0 = [
            r_1r1c.c0, ua_c_init, k_c_rated_warm,
            max(ALPHA_C_BOUNDS[0], r_1r1c.alpha_c),
            K_W_PRIOR_MEAN, MASS_RATIO_PRIOR_MEAN,
            TAU_HP_PRIOR_MEAN,
        ]
    else:
        x0 = [
            r_1r1c.c0, ua_c_init, k_c_rated_warm,
            K_W_PRIOR_MEAN, MASS_RATIO_PRIOR_MEAN,
            TAU_HP_PRIOR_MEAN,
        ]
    # Clamp warm-start values into bounds.
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        x0[i] = max(lo, min(hi, x0[i]))

    n_data = m
    n_priors = len(prior_mu)

    # Sim-error PEM residual: forward-simulate state x = [T_a, T_w, Q_hp] via
    # matrix-exponential; data residual = predicted T_a − observed T_a (in °C).
    #
    # 3-state continuous dynamics (Stage 1, 2026-06-03):
    #
    #   dT_a/dt  = c₀ + ua_c·(T_out − T_a) + Q_hp + α_air·solar + k_w·(T_w − T_a)
    #   dT_w/dt  = (k_w/mr)·(T_a − T_w) + (α_wall/mr)·solar
    #   dQ_hp/dt = (k_c_eff·(sp − T_a) − Q_hp) / τ_hp   [HP active]
    #            = -Q_hp / τ_hp                          [HP inactive]
    #
    # ZOH formulation with constant A across ticks:
    #
    #   A = | -(ua_c + k_w)   k_w         1           |
    #       | k_w/mr         -k_w/mr      0           |
    #       | 0               0          -1/τ_hp      |
    #
    #   b[0] = c₀ + ua_c·T_out_prev + α_air·solar_prev
    #   b[1] = α_wall·solar_prev / mr
    #   b[2] = k_c_eff(T_out_prev)·(sp_prev − T_a_prev) / τ_hp   [HP active]
    #        = 0                                                  [HP inactive]
    #
    # The (sp − T_a) feedback in dQ_hp/dt is ZOH-held at t_{i-1} (using observed
    # T_a, not predicted) — this keeps A constant across ticks (only τ_hp varies
    # with params, not data) so dt-memoization works the same as the 2-state
    # version. Error from ZOH on T_a inside Q_hp's equation is bounded by
    # ΔT_a over τ_hp (a few tenths of a degree at most), small relative to
    # other modeling error.
    #
    # The (B) HP-off snap-to-zero (Q_hp := 0 whenever HP is off at the
    # current tick) is applied AFTER the residual is computed for the current
    # tick, before propagating to the next. This avoids modeling residual
    # delivery that doesn't happen (vanes closed + blower off on Fujitsu
    # wall-mounts → ~0.1% of active delivery). See function docstring for
    # why this differs from lit (ducted/hydronic systems where decay applies).
    #
    # k_c_eff = k_c_rated · capacity_profile.factor(T_out, mode) — the AHRI-
    # rating-point capacity (the fitted scalar) is modulated by outdoor temp
    # per the manufacturer engineering data in capacity_profiles.py.
    #
    # Lit-canonical output-error PEM (Ljung) extended with HP dynamic state;
    # see [[reference-hp-grey-box-lit]] for Tang 2025 / NIST 2023 motivation.
    import numpy as np  # local-import to honour scipy-optional pattern

    # Per-fit dt-memoization. The set of distinct positive dt values is
    # data-only (depends on dt_min, not params), so compute it once outside
    # residual_fn. Inside, build per-dt (eA, ψ) caches up front; the inner
    # loop becomes a dict lookup. Reduces expm cost per residual_fn from
    # O(m) to O(unique_dts) — typically 30–100 vs 10000 for sorted-but-
    # eviction-sparse buffers.
    _unique_dts_pos = sorted({d for d in dt_min if d > 0})

    # Pre-compute capacity-profile factor at each tick's outdoor temp. Data-
    # only (independent of fit params), so cache once.
    k_c_profile_at = [profile.factor(t, fit_mode) for t in t_out]

    # TODO(perf): closed-form 3×3 expm. A is lower block-triangular (envelope
    # 2×2 block top-left, Q_hp scalar bottom-right, T_a←Q_hp coupling A[0,2]=1
    # in top-right). Can compose: top-left = closed-form 2×2 (reuse old Sylvester
    # projector formula), bottom-right = exp(-dt/τ_hp), off-diagonal block via
    # Sylvester equation. Would recover the ~6× perf gap vs scipy.linalg.expm.
    # Deferred per Stage 1c Decision 4 — optimize when bench shows it matters.
    _eye3 = np.eye(3)

    def residual_fn(params: list[float]) -> list[float]:
        if has_solar:
            c0, ua_c, k_c_rated, alpha_total, k_w, mass_ratio, tau_hp = params
        else:
            c0, ua_c, k_c_rated, k_w, mass_ratio, tau_hp = params
            alpha_total = 0.0
        alpha_air = alpha_total * SOLAR_AIR_FRACTION
        alpha_wall = alpha_total * SOLAR_WALL_FRACTION
        a_wall_rate = k_w / mass_ratio
        inv_tau_hp = 1.0 / tau_hp

        # Single 3×3 A (k_c does NOT enter A — it enters only Q_hp's b
        # component, modulated by the capacity profile per-tick).
        A = np.array([
            [-(ua_c + k_w),  k_w,          1.0          ],
            [ a_wall_rate,  -a_wall_rate,  0.0          ],
            [ 0.0,           0.0,         -inv_tau_hp   ],
        ], dtype=float)

        # dt-memoized (eA, ψ) caches. scipy.linalg.expm + numpy.linalg.solve
        # at 3×3 is ~6 μs/call; with ~50 unique dts in a typical batch, the
        # cache build cost is ~300 μs per residual_fn call (negligible vs
        # the ~10ms inner loop on 10k-tick buffers).
        cache: dict[float, tuple] = {}
        try:
            for dt_u in _unique_dts_pos:
                eA = _expm(A * dt_u)
                psi = np.linalg.solve(A, eA - _eye3)
                cache[dt_u] = (
                    float(eA[0, 0]), float(eA[0, 1]), float(eA[0, 2]),
                    float(eA[1, 0]), float(eA[1, 1]), float(eA[1, 2]),
                    float(eA[2, 0]), float(eA[2, 1]), float(eA[2, 2]),
                    float(psi[0, 0]), float(psi[0, 1]), float(psi[0, 2]),
                    float(psi[1, 0]), float(psi[1, 1]), float(psi[1, 2]),
                    float(psi[2, 0]), float(psi[2, 1]), float(psi[2, 2]),
                )
        except Exception:
            # Numerical failure (e.g. expm overflow at extreme params, or
            # A singular if a parameter rails to a degenerate point):
            # return large residuals so optimizer steers away. Length must
            # include prior terms so least_squares sees a consistent
            # residual vector size across iterations.
            return [1e6] * (n_data + n_priors)

        # Initial state: T_a/T_w start at first observation (best estimate
        # absent multi-sample warm-up), Q_hp = 0 (snap-to-zero on boot;
        # first HP-on interval will ramp it via τ_hp dynamics).
        x0 = x1 = t_air[0]
        x2 = 0.0
        residuals = [0.0] * n_data
        # First-tick residual identically zero — no inter-sample propagation
        # is possible. Optimizer learns from i=1 onward.

        for i in range(1, m):
            dt = dt_min[i]
            if dt <= 0:
                residuals[i] = 0.0
                continue
            sp_prev = hp_setpoint_arr[i - 1]
            t_a_prev = t_air[i - 1]
            # Mode-aware HP active check: heat mode delivers when room <
            # setpoint; cool mode delivers when room > setpoint. Matches
            # thermal_model.py's thermostatic cycling for the bench, and
            # is the natural production semantics.
            if fit_mode == "heat":
                active_prev = (sp_prev is not None) and (t_a_prev < sp_prev)
            else:  # cool
                active_prev = (sp_prev is not None) and (t_a_prev > sp_prev)

            (eA00, eA01, eA02, eA10, eA11, eA12, eA20, eA21, eA22,
             ps00, ps01, ps02, ps10, ps11, ps12, ps20, ps21, ps22) = cache[dt]

            b0 = c0 + ua_c * t_out[i - 1] + alpha_air * solar[i - 1]
            b1 = alpha_wall * solar[i - 1] / mass_ratio
            if active_prev:
                # k_c_eff = k_c_rated · profile(T_out_prev, mode); the
                # profile factor was pre-computed outside residual_fn.
                k_c_eff = k_c_rated * k_c_profile_at[i - 1]
                # In heat mode (sp - T_a) > 0 → b2 > 0 (delivering heat).
                # In cool mode (sp - T_a) < 0 → b2 < 0 (delivering cooling).
                # The natural sign convention; cool-mode Q_hp will be negative.
                b2 = k_c_eff * (sp_prev - t_a_prev) * inv_tau_hp
            else:
                b2 = 0.0
            # x_new = eA · x + ψ · b  — scalar expansion to skip numpy overhead.
            x0_new = (eA00 * x0 + eA01 * x1 + eA02 * x2
                      + ps00 * b0 + ps01 * b1 + ps02 * b2)
            x1_new = (eA10 * x0 + eA11 * x1 + eA12 * x2
                      + ps10 * b0 + ps11 * b1 + ps12 * b2)
            x2_new = (eA20 * x0 + eA21 * x1 + eA22 * x2
                      + ps20 * b0 + ps21 * b1 + ps22 * b2)
            x0, x1, x2 = x0_new, x1_new, x2_new
            residuals[i] = x0 - t_air[i]

            # Snap-to-zero on HP-off (Option B, see function docstring). Apply
            # AFTER computing residual at i — the snap affects only the NEXT
            # tick's starting state, not the current residual. On Fujitsu
            # wall-mount mini-splits, vanes close + blower stops on idle →
            # residual coil delivery is ~0.1% of active (negligible vs sensor
            # noise + envelope dynamics); modeling decay would inject a false
            # heat-delivery signal that biases ua_c / c0.
            sp_curr = hp_setpoint_arr[i]
            if fit_mode == "heat":
                hp_off_at_i = (sp_curr is None) or not (t_air[i] < sp_curr)
            else:
                hp_off_at_i = (sp_curr is None) or not (t_air[i] > sp_curr)
            if hp_off_at_i:
                x2 = 0.0

        # Tikhonov prior penalty terms appended to the data residuals.
        # Each term is (θ_i − μ_i) / σ_i; scipy's sum-of-squares loss makes
        # the contribution to the objective equal to ½·(θ_i−μ_i)²/σ_i², i.e.
        # the negative log of an independent Gaussian prior up to a constant.
        # The MAP optimum is the inverse-variance-weighted blend of likelihood
        # (data residuals) and priors. CTSM-R MAP / Pathak 2019 §3.1 BSSM
        # (with the simplification that scipy's huber loss is a robust
        # likelihood, not an exact Gaussian).
        prior_residuals = [
            (params[i] - prior_mu[i]) / prior_sigma[i]
            for i in range(n_priors)
        ]
        return residuals + prior_residuals

    try:
        result = _least_squares(
            residual_fn, x0,
            bounds=(lower, upper),
            method="trf",
            loss="huber",
            # Sim-error PEM residuals are in °C (T_a prediction error),
            # not °C/min (rate). With sensor noise σ=0.1°C, typical
            # residuals are 0.1–1.0°C; f_scale=0.1 puts them in the
            # Huber linear regime. Was 0.005 (rate-residual scale).
            f_scale=0.1,
            max_nfev=400,  # cap; 6-param fit usually converges in <100
        )
    except Exception:
        _LOGGER.exception("Grey-box 2R2C: least_squares failed")
        return None

    params_fit = dict(zip(param_names, result.x))
    c0 = float(params_fit["c0"])
    ua_c = float(params_fit["ua_c"])
    # k_c here is k_c_RATED (AHRI rating-point value); downstream consumers
    # interpret it as the gain at AHRI 47°F heating / 95°F cooling. To get
    # k_c at a specific operating T_out, multiply by profile.factor(T_out).
    # The GreyboxResult.k_c field semantics is the rated value going forward.
    k_c = float(params_fit["k_c"])
    alpha_total = float(params_fit.get("alpha_c", 0.0))
    # v2 Bayesian: k_w and mass_ratio are fitted (constrained by Tikhonov
    # priors, see comments at parameter packing above).
    k_w = float(params_fit["k_w"])
    mass_ratio = float(params_fit["mass_ratio"])
    # Stage 1 (2026-06-03): tau_hp is the new 7th parameter — the HP first-
    # order ramp time constant. See module-level TAU_HP_BOUNDS for bounds
    # rationale and identifiability warning.
    tau_hp = float(params_fit["tau_hp"])

    tau_fast, tau_slow = _natural_eigenvalues(ua_c, k_w, mass_ratio)
    tau_eff = tau_slow  # dominant for legacy consumers
    # residual_rms reflects data-fit quality (sim-error PEM units: °C of
    # T_a prediction error). Strip the n_priors Tikhonov terms appended
    # to result.fun before computing RMS — they're a regularization, not
    # observation residuals, and would skew the metric used by quality
    # gates and downstream diagnostics.
    if result.fun is not None and m > 0:
        data_residuals = result.fun[:n_data] if len(result.fun) >= n_data else result.fun
        residual_rms = math.sqrt(
            sum(float(r) * float(r) for r in data_residuals) / m
        )
    else:
        residual_rms = 0.0

    # Standard errors via Jacobian.
    std_err: dict[str, float] = {}
    if result.jac is not None:  # pragma: no branch — scipy.least_squares populates jac on convergence
        try:
            import numpy as np
            J = result.jac
            n_params = len(param_names)
            sigma2 = 2.0 * result.cost / max(1, m - n_params)
            JtJ_inv = np.linalg.pinv(J.T @ J) * sigma2
            for i, name in enumerate(param_names):
                var = JtJ_inv[i, i]
                std_err[name] = math.sqrt(max(0.0, float(var)))
        except Exception:
            pass

    # v2 Bayesian: Stage B perturbation-regime wall fit is GONE. k_w and
    # mass_ratio are joint-fitted with Tikhonov priors above; when
    # perturbation data is available it naturally increases the Fisher
    # information on wall mode in the likelihood term, letting the MAP
    # estimate move further from the prior. When perturbation data is
    # absent, the prior dominates and we get lit-typical wall mode —
    # the same end state Stage A's hard-fix produced, but reached
    # principally through Bayesian inference rather than ad-hoc fixing.

    tau_agreement = None
    if plant_tau_slow is not None and plant_tau_slow > 0 and not math.isinf(tau_slow):
        tau_agreement = abs(tau_slow - plant_tau_slow) / plant_tau_slow * 100.0

    return GreyboxResult(
        n_observations=m,
        n_hp_on=n_hp_on,
        n_hp_off=n_hp_off,
        c0=c0,
        ua_c=ua_c,
        k_c=k_c,  # k_c_RATED — AHRI rating-point value; multiply by
                   # capacity_profile.factor(T_out, mode) for operating-point gain
        alpha_c=alpha_total,  # SS-effective total; bridge uses α_c/k_c
        tau_eff=tau_eff,
        residual_rms=residual_rms,
        cost=float(result.cost),
        n_function_evals=int(result.nfev),
        plant_tau_slow=plant_tau_slow,
        tau_agreement_pct=float(tau_agreement) if tau_agreement is not None else None,
        param_std_err=std_err,
        is_2r2c=True,
        k_w=k_w,
        mass_ratio=mass_ratio,
        tau_fast=tau_fast,
        tau_slow=tau_slow,
        tau_hp=tau_hp,
        dt_median_min=typical_dt if typical_dt > 0 else None,
    )



def log_greybox_result(
    result: GreyboxResult,
    log_prefix: str = "",
) -> None:
    """Log grey-box fit results for diagnostics."""
    label = "2R2C" if result.is_2r2c else "1R1C"
    _LOGGER.info(
        "%sGrey-box %s: %d obs (%d HP-on, %d HP-off), RMS=%.4f °C/min",
        log_prefix, label, result.n_observations, result.n_hp_on,
        result.n_hp_off, result.residual_rms,
    )
    if result.is_2r2c:
        _LOGGER.info(
            "%s  c0=%.5f, ua_c=%.5f, k_c=%.5f, α_total=%.5f, k_w=%.5f, mass_ratio=%.2f",
            log_prefix, result.c0, result.ua_c, result.k_c,
            result.alpha_c, result.k_w or 0.0, result.mass_ratio or 0.0,
        )
        _LOGGER.info(
            "%s  τ_fast=%.0f min, τ_slow=%.0f min",
            log_prefix,
            result.tau_fast if result.tau_fast is not None else float("inf"),
            result.tau_slow if result.tau_slow is not None else float("inf"),
        )
    else:
        _LOGGER.info(
            "%s  c0=%.5f, ua_c=%.5f (τ=%.0f min), k_c=%.5f, α_c=%.5f",
            log_prefix, result.c0, result.ua_c, result.tau_eff,
            result.k_c, result.alpha_c,
        )
    _LOGGER.info(
        "%s  cost=%.6f, nfev=%d",
        log_prefix, result.cost, result.n_function_evals,
    )

    if result.param_std_err:  # pragma: no branch — param_std_err empty only on Jacobian-failure path
        parts = [f"{k}=\u00b1{v:.6f}" for k, v in result.param_std_err.items()]
        _LOGGER.info("%s  std_err: %s", log_prefix, ", ".join(parts))

    if result.tau_agreement_pct is not None:
        agree = "agree" if result.tau_agreement_pct < 30 else "disagree"
        _LOGGER.info(
            "%s  Plant ID cross-check: τ_eff=%.0f vs τ_slow=%.0f (%.0f%% %s)",
            log_prefix, result.tau_eff,
            result.plant_tau_slow, result.tau_agreement_pct, agree,
        )


# ── Steady-state bridge ─────────────────────────────────────────────
#
# At equilibrium (dT/dt = 0) the energy balance becomes algebraic:
#
#   hp_offset = -(ua_c/k_c) × ΔT_outdoor - (α_c/k_c) × solar
#
# The ratios map directly to the WLS β coefficients.
#
# Standard errors propagated via the delta method for f(a,b) = a/b:
#   σ_{a/b}² ≈ (a/b)² × (σ_a²/a² + σ_b²/b² - 2·ρ_{ab}·σ_a·σ_b/(a·b))
#
# Without the cross-correlation ρ_{ab} (not available from scipy), we
# use the conservative upper bound (ρ=0):
#   σ_{a/b}² ≈ (a/b)² × (σ_a²/a² + σ_b²/b²)

# Quality gate thresholds
GATE_MIN_TAU = 30.0    # minutes — faster implies unrealistic building (1R1C)
GATE_MAX_TAU = 1500.0  # minutes (25h) — 2R2C slow mode can reach 20-100h
GATE_MIN_TAU_FAST = 5.0     # minutes — air-node response (2R2C)
GATE_MAX_TAU_FAST = 60.0    # minutes
GATE_MIN_TAU_SLOW = 60.0    # minutes — wall mode (2R2C)
# 3500 min ≈ 58h covers lit-typical residential (15–55h per design_auto_-
# perturbation.md citation: NA-residential survey of 10,000+ buildings via
# Bacher-Madsen 2011 family). Old threshold of 1500 min cut off the upper
# half of typical residential. Passive-House range (90–200h) intentionally
# excluded — those buildings would need a separate gate.
GATE_MAX_TAU_SLOW = 3500.0  # minutes
GATE_MIN_TAU_SEPARATION = 2.0  # τ_slow / τ_fast — separation needed for 2R2C
GATE_MIN_MASS_RATIO = 1.0   # Bacher-Madsen typical 5–10; allow 1–20
GATE_MAX_MASS_RATIO = 20.0
GATE_MAX_CV = 0.5      # coefficient of variation (std_err / |estimate|)

# Cadence-adaptive residual_rms thresholds. The fit residual scales with
# both the residual formulation (rate vs state) and observation cadence
# (sensor-noise floor depends on dt for rate, dt-independent for state).
# Choose threshold = K_RMS_GATE × analytical_noise_floor so we accept fits
# within K× the irreducible measurement noise.
SIGMA_SENSOR_DEFAULT = 0.1   # °C — typical HA temp sensor σ
K_RMS_GATE = 5.0              # multiplier on noise floor (lit-typical 3-5×)
# 5-tick FD averaging window used by production room_rate computation
RATE_FD_TICKS = 5

# Legacy constant kept for back-compat with downstream consumers and the
# 1R1C gate fallback when dt_median_min is not available. Production fits
# should use _gate_max_rms() below.
GATE_MAX_RMS = 0.02    # °C/min — legacy fixed threshold (15-min cadence)


def _gate_max_rms(
    is_2r2c: bool,
    dt_median_min: float | None,
    sigma_sensor: float = SIGMA_SENSOR_DEFAULT,
) -> float:
    """Cadence-adaptive residual_rms gate threshold.

    For sim-error PEM (2R2C): residual is in °C of state-prediction error.
    Noise floor is sensor-noise σ directly, independent of dt.

    For rate-residual (1R1C): residual is in °C/min. Noise floor is the
    propagated sensor noise on the 5-tick FD: σ × √2 / (5 × dt_median_min).

    Returns K_RMS_GATE × analytical_noise_floor.
    """
    if is_2r2c:
        return K_RMS_GATE * sigma_sensor
    if dt_median_min is None or dt_median_min <= 0:
        return GATE_MAX_RMS  # legacy fallback
    rate_noise_floor = sigma_sensor * math.sqrt(2.0) / (RATE_FD_TICKS * dt_median_min)
    return K_RMS_GATE * rate_noise_floor


@dataclass
class GreyboxBridgeResult:
    """Grey-box rate coefficients mapped to WLS-compatible β."""

    # β coefficients in WLS order: [intercept, outdoor_delta, ...model_inputs]
    # intercept is None (grey-box doesn't produce it; use WLS β₀)
    beta: list[float | None]
    beta_std_err: list[float]

    # Bonus outputs WLS cannot provide
    tau_eff: float       # 1/ua_c (1R1C) or τ_slow (2R2C); dominant time const
    k_eff: float         # k_c/ua_c — process gain (currently assumed 1.0 in IMC)

    # Quality gate results
    gates_passed: bool   # True if all quality gates passed
    gate_details: dict[str, bool]  # per-gate pass/fail

    # Source grey-box result for reference
    greybox: GreyboxResult

    # 2R2C-only (None for 1R1C bridges) — exposed so plant_identifier
    # can update τ_fast and τ_slow providers separately.
    tau_fast: float | None = None
    tau_slow: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "beta_std_err": [round(s, 6) for s in self.beta_std_err],
            "tau_eff": round(self.tau_eff, 1),
            "k_eff": round(self.k_eff, 4),
            "gates_passed": self.gates_passed,
            "gate_details": self.gate_details,
            "tau_fast": round(self.tau_fast, 1) if self.tau_fast is not None else None,
            "tau_slow": round(self.tau_slow, 1) if self.tau_slow is not None else None,
        }


def _delta_method_ratio_std(
    a: float,
    b: float,
    sigma_a: float,
    sigma_b: float,
) -> float:
    """Standard error of a/b via delta method (assuming zero correlation).

    Conservative upper bound: σ_{a/b} = |a/b| × √(σ_a²/a² + σ_b²/b²).
    """
    if abs(b) < 1e-12 or abs(a) < 1e-12:
        return float("inf")
    ratio = a / b
    cv_a_sq = (sigma_a / a) ** 2
    cv_b_sq = (sigma_b / b) ** 2
    return abs(ratio) * math.sqrt(cv_a_sq + cv_b_sq)


def _check_quality_gates(
    result: GreyboxResult,
) -> dict[str, bool]:
    """Check all quality gates.  Returns {gate_name: passed}."""
    gates: dict[str, bool] = {}

    # Gate 1: hp_offset variance (already checked during fit — if k_c was
    # fixed at default, the fit ran but k_c is unreliable).
    # Use n_hp_on and n_hp_off as proxy: need both to separate k_c from ua_c.
    min_fraction = 0.1  # at least 10% HP-off OR 10% HP-on
    n = result.n_observations
    if n > 0:
        off_frac = result.n_hp_off / n
        on_frac = result.n_hp_on / n
        gates["hp_offset_diversity"] = off_frac >= min_fraction and on_frac >= min_fraction
    else:
        gates["hp_offset_diversity"] = False

    # Gate 2: parameter precision (coefficient of variation)
    se = result.param_std_err
    ua_c_cv = se.get("ua_c", float("inf")) / max(abs(result.ua_c), 1e-12)
    k_c_cv = se.get("k_c", float("inf")) / max(abs(result.k_c), 1e-12)
    gates["param_precision_ua_c"] = ua_c_cv < GATE_MAX_CV
    gates["param_precision_k_c"] = k_c_cv < GATE_MAX_CV
    # α_c precision only checked if solar was fitted
    if "alpha_c" in se:
        alpha_cv = se["alpha_c"] / max(abs(result.alpha_c), 1e-12)
        gates["param_precision_alpha_c"] = alpha_cv < GATE_MAX_CV

    # Gate 3: physical plausibility
    if result.is_2r2c:
        # 2R2C: separate τ_fast / τ_slow plausibility plus mode separation.
        tf = result.tau_fast if result.tau_fast is not None else float("inf")
        ts = result.tau_slow if result.tau_slow is not None else float("inf")
        mr = result.mass_ratio if result.mass_ratio is not None else 0.0
        kw = result.k_w if result.k_w is not None else 0.0
        gates["tau_fast_plausible"] = GATE_MIN_TAU_FAST <= tf <= GATE_MAX_TAU_FAST
        gates["tau_slow_plausible"] = GATE_MIN_TAU_SLOW <= ts <= GATE_MAX_TAU_SLOW
        # Separation: need τ_slow noticeably bigger than τ_fast or the model
        # has collapsed to 1R1C (degenerate fit).
        if tf > 0 and not math.isinf(ts):
            gates["tau_separation"] = (ts / tf) >= GATE_MIN_TAU_SEPARATION
        else:
            gates["tau_separation"] = False
        gates["mass_ratio_plausible"] = GATE_MIN_MASS_RATIO <= mr <= GATE_MAX_MASS_RATIO
        gates["k_w_positive"] = kw > 0
        # k_w precision (only when std_err available)
        if "k_w" in se:
            k_w_cv = se["k_w"] / max(abs(kw), 1e-12)
            gates["param_precision_k_w"] = k_w_cv < GATE_MAX_CV
    else:
        gates["tau_plausible"] = GATE_MIN_TAU <= result.tau_eff <= GATE_MAX_TAU
    gates["k_c_positive"] = result.k_c > 0
    gates["alpha_c_nonnegative"] = result.alpha_c >= 0.0 or "alpha_c" not in se

    # Gate 4: residual quality (cadence-adaptive — see _gate_max_rms).
    # 2R2C uses sim-error PEM (residual in °C); 1R1C uses rate residual
    # (°C/min). The threshold scales with observation cadence so the gate
    # is consistent with the analytical noise floor at any sampling rate.
    rms_threshold = _gate_max_rms(result.is_2r2c, result.dt_median_min)
    gates["residual_rms"] = result.residual_rms < rms_threshold

    return gates


def greybox_to_beta(
    result: GreyboxResult,
    model_inputs: list[dict[str, Any]],
    log_prefix: str = "",
) -> GreyboxBridgeResult:
    """Convert grey-box rate coefficients to WLS-compatible β via steady-state bridge.

    The mapping:
        β₁ (outdoor_delta) = -ua_c / k_c
        β₂ (solar input)   = -α_c / k_c   (only for the solar model input)
        β₀ (intercept)     = None           (grey-box doesn't produce this)

    Other model inputs (heat sources, adjacent zones) get None — grey-box
    doesn't identify those separately.

    Standard errors propagated via delta method on the ratio.

    Args:
        result: fitted GreyboxResult from fit_greybox().
        model_inputs: model input config dicts (same as passed to fit_greybox).
        log_prefix: logging prefix string.

    Returns:
        GreyboxBridgeResult with β, std_err, quality gates, and bonus outputs.
    """
    # Quality gates
    gate_details = _check_quality_gates(result)
    gates_passed = all(gate_details.values())

    se = result.param_std_err
    sigma_ua_c = se.get("ua_c", float("inf"))
    sigma_k_c = se.get("k_c", float("inf"))
    sigma_alpha_c = se.get("alpha_c", float("inf"))

    # β₁ (outdoor_delta) = -ua_c / k_c
    beta_outdoor = -result.ua_c / result.k_c if result.k_c > 1e-12 else 0.0
    se_outdoor = _delta_method_ratio_std(
        result.ua_c, result.k_c, sigma_ua_c, sigma_k_c,
    )

    # Build β in WLS order: [intercept, outdoor_delta, ...model_inputs]
    n_coeffs = 2 + len(model_inputs)  # intercept + outdoor_delta + inputs
    beta: list[float | None] = [None] * n_coeffs
    beta_std_err = [float("inf")] * n_coeffs

    # β₀ (intercept): grey-box doesn't produce this
    # β₁ (outdoor_delta): from bridge
    beta[1] = beta_outdoor
    beta_std_err[1] = se_outdoor

    # Model inputs: only solar gets a grey-box estimate
    for i, m in enumerate(model_inputs):
        coeff_idx = 2 + i
        if m.get("input_role") == "solar" and result.alpha_c != 0.0:
            beta_solar = -result.alpha_c / result.k_c if result.k_c > 1e-12 else 0.0
            se_solar = _delta_method_ratio_std(
                result.alpha_c, result.k_c, sigma_alpha_c, sigma_k_c,
            )
            beta[coeff_idx] = beta_solar
            beta_std_err[coeff_idx] = se_solar

    # Bonus outputs
    tau_eff = result.tau_eff
    k_eff = result.k_c / result.ua_c if result.ua_c > 1e-12 else 0.0

    bridge = GreyboxBridgeResult(
        beta=beta,
        beta_std_err=beta_std_err,
        tau_eff=tau_eff,
        k_eff=k_eff,
        gates_passed=gates_passed,
        gate_details=gate_details,
        greybox=result,
        tau_fast=result.tau_fast if result.is_2r2c else None,
        tau_slow=result.tau_slow if result.is_2r2c else None,
    )

    # Log
    _LOGGER.info(
        "%sGrey-box bridge: outdoor_delta=%.4f (σ=%.4f), τ_eff=%.0f min, K_eff=%.3f",
        log_prefix, beta_outdoor, se_outdoor, tau_eff, k_eff,
    )
    for i, m in enumerate(model_inputs):
        coeff_idx = 2 + i
        if beta[coeff_idx] is not None:
            _LOGGER.info(
                "%s  %s: β=%.4f (σ=%.4f)",
                log_prefix, m.get("name", f"input_{i}"),
                beta[coeff_idx], beta_std_err[coeff_idx],
            )
    gate_summary = ", ".join(
        f"{k}={'OK' if v else 'FAIL'}" for k, v in gate_details.items()
    )
    _LOGGER.info(
        "%s  Quality gates: %s → %s",
        log_prefix, gate_summary, "PASS" if gates_passed else "REJECT",
    )

    return bridge
