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


# Lit-typical prior lookup. Used when ``PriorState`` has no promoted
# value for a given parameter (cold start, or that param was railed and
# never qualified for promotion).
_LIT_PRIOR_MEAN: dict[str, float] = {
    "c0":         C0_PRIOR_MEAN,
    "ua_c":       UA_C_PRIOR_MEAN,
    "k_c":        K_C_PRIOR_MEAN,
    "alpha_c":    ALPHA_C_PRIOR_MEAN,
    "k_w":        K_W_PRIOR_MEAN,
    "mass_ratio": MASS_RATIO_PRIOR_MEAN,
}
_LIT_PRIOR_SIGMA: dict[str, float] = {
    "c0":         C0_PRIOR_SIGMA,
    "ua_c":       UA_C_PRIOR_SIGMA,
    "k_c":        K_C_PRIOR_SIGMA,
    "alpha_c":    ALPHA_C_PRIOR_SIGMA,
    "k_w":        K_W_PRIOR_SIGMA,
    "mass_ratio": MASS_RATIO_PRIOR_SIGMA,
}
_BOUNDS_BY_NAME: dict[str, tuple[float, float]] = {
    "c0":         C0_BOUNDS,
    "ua_c":       UA_C_BOUNDS,
    "k_c":        K_C_BOUNDS,
    "alpha_c":    ALPHA_C_BOUNDS,
    "k_w":        K_W_BOUNDS,
    "mass_ratio": MASS_RATIO_BOUNDS,
}


@dataclass(frozen=True)
class PriorState:
    """Per-zone persistent priors for greybox parameters — Pathak (2019)
    §4.2 transfer learning: today's posterior is tomorrow's prior.

    Each parameter is stored as ``(mu, sigma)`` or ``None`` (cold start /
    never promoted). ``None`` entries fall back to lit-typical defaults
    via :meth:`mu_for` and :meth:`sigma_for`.

    Promotion is governed by :func:`promote_posterior` with lit-grounded
    gates (no railed params per Reynders 2014; bridge gates pass;
    least_squares converged). The dataclass itself is immutable —
    promotion returns a new instance.

    Serialization is via :meth:`to_dict` / :meth:`from_dict`; unknown
    keys are ignored on restore so the schema can grow forward-compatibly.
    """

    c0:         tuple[float, float] | None = None
    ua_c:       tuple[float, float] | None = None
    k_c:        tuple[float, float] | None = None
    alpha_c:    tuple[float, float] | None = None
    k_w:        tuple[float, float] | None = None
    mass_ratio: tuple[float, float] | None = None
    n_promotions: int = 0

    _PARAM_NAMES = ("c0", "ua_c", "k_c", "alpha_c", "k_w", "mass_ratio")

    def mu_for(self, name: str) -> float:
        """Return persisted μ for ``name``, or the lit-typical default if
        this parameter has never been promoted."""
        val = getattr(self, name, None)
        return val[0] if val is not None else _LIT_PRIOR_MEAN[name]

    def sigma_for(self, name: str) -> float:
        """Return persisted σ for ``name``, or the lit-typical default if
        this parameter has never been promoted."""
        val = getattr(self, name, None)
        return val[1] if val is not None else _LIT_PRIOR_SIGMA[name]

    def is_promoted(self, name: str) -> bool:
        """True iff this parameter has a persisted posterior (not lit default)."""
        return getattr(self, name, None) is not None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n_promotions": self.n_promotions}
        for name in self._PARAM_NAMES:
            val = getattr(self, name)
            if val is not None:
                out[name] = [float(val[0]), float(val[1])]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PriorState":
        """Restore from serialized form. ``None`` or empty input → cold-start
        defaults. Malformed entries for a single parameter are silently
        dropped (that param falls back to lit default)."""
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}
        for name in cls._PARAM_NAMES:
            entry = data.get(name)
            if entry is None:
                continue
            try:
                mu, sigma = float(entry[0]), float(entry[1])
                if math.isfinite(mu) and math.isfinite(sigma) and sigma > 0:
                    kwargs[name] = (mu, sigma)
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
    bridge_gates_passed: bool,
    least_squares_converged: bool = True,
) -> PriorState:
    """Pathak (2019) §4.2 transfer learning: promote a passing batch's
    posterior to next batch's prior.

    Lit-grounded promotion criteria (all must hold for the batch as a
    whole; per-parameter gates apply below):

      1. ``bridge_gates_passed`` — our standing quality bar
      2. ``result.is_2r2c`` — only 2R2C fits expose all 6 envelope params
      3. ``least_squares_converged`` — standard PEM gate (Ljung 1999)
      4. ``param_std_err`` populated — Jacobian-derived posterior σ
         must exist for the parameters we want to promote

    Per-parameter (Reynders 2014, "rails are diagnosis"):
      - A parameter at (within 1% of) either of its bounds is non-
        identifiable; the artificially-tight σ from the bound clamp is
        not a true posterior. **That parameter is not promoted** — its
        existing prior survives. Other parameters from the same fit may
        still promote independently.
      - σ must be finite and > 0.

    Returns a new :class:`PriorState` (the input is not mutated). If no
    parameter qualifies, returns ``current_state`` unchanged.
    """
    if not bridge_gates_passed:
        return current_state
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
    }

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
        updates[name] = (float(value), float(sigma))

    if not updates:
        return current_state

    # Build new immutable state with the updated fields; non-updated params
    # carry over from current_state (either persisted posterior or None).
    return PriorState(
        c0=updates.get("c0", current_state.c0),
        ua_c=updates.get("ua_c", current_state.ua_c),
        k_c=updates.get("k_c", current_state.k_c),
        alpha_c=updates.get("alpha_c", current_state.alpha_c),
        k_w=updates.get("k_w", current_state.k_w),
        mass_ratio=updates.get("mass_ratio", current_state.mass_ratio),
        n_promotions=current_state.n_promotions + 1,
    )


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

    # Attempt 2R2C with 1R1C warm start.
    r_2r2c = _fit_greybox_2r2c(
        eligible, model_inputs, r_1r1c, plant_tau_slow,
        prior_state=prior_state,
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
) -> GreyboxResult | None:
    """Fit a 2R2C grey-box energy balance with forward-simulated wall state.

    Air ODE:  dT_a/dt = c₀ + ua_c·(T_out - T_a) + k_c·hp_offset
                       + α_air·solar + k_w·(T_w - T_a)
    Wall ODE: dT_w/dt = (k_w/mass_ratio)·(T_a - T_w) + (α_wall/mass_ratio)·solar

    Solar split fixed at SOLAR_AIR_FRACTION / SOLAR_WALL_FRACTION (ASHRAE).
    α_total is the fitted parameter; α_air = 0.3·α_total, α_wall = 0.7·α_total.

    Residual: predicted - observed room_rate at each tick, with t_wall
    propagated forward via exact piecewise-exponential integration.

    Bridge formulas at steady state are unchanged from 1R1C:
      β_outdoor = -ua_c / k_c
      β_solar   = -α_total / k_c   (the 30/70 split cancels at SS)
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
    if has_solar:
        param_names = ["c0", "ua_c", "k_c", "alpha_c", "k_w", "mass_ratio"]
        lower = [
            C0_BOUNDS[0], UA_C_BOUNDS[0], K_C_BOUNDS[0], ALPHA_C_BOUNDS[0],
            K_W_BOUNDS[0], MASS_RATIO_BOUNDS[0],
        ]
        upper = [
            C0_BOUNDS[1], UA_C_BOUNDS[1], K_C_BOUNDS[1], ALPHA_C_BOUNDS[1],
            K_W_BOUNDS[1], MASS_RATIO_BOUNDS[1],
        ]
        prior_mu = [ps.mu_for(n) for n in param_names]
        prior_sigma = [ps.sigma_for(n) for n in param_names]
    else:
        param_names = ["c0", "ua_c", "k_c", "k_w", "mass_ratio"]
        lower = [
            C0_BOUNDS[0], UA_C_BOUNDS[0], K_C_BOUNDS[0],
            K_W_BOUNDS[0], MASS_RATIO_BOUNDS[0],
        ]
        upper = [
            C0_BOUNDS[1], UA_C_BOUNDS[1], K_C_BOUNDS[1],
            K_W_BOUNDS[1], MASS_RATIO_BOUNDS[1],
        ]
        prior_mu = [ps.mu_for(n) for n in param_names]
        prior_sigma = [ps.sigma_for(n) for n in param_names]

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

    if has_solar:
        x0 = [
            r_1r1c.c0, ua_c_init, r_1r1c.k_c,
            max(ALPHA_C_BOUNDS[0], r_1r1c.alpha_c),
            K_W_PRIOR_MEAN, MASS_RATIO_PRIOR_MEAN,
        ]
    else:
        x0 = [
            r_1r1c.c0, ua_c_init, r_1r1c.k_c,
            K_W_PRIOR_MEAN, MASS_RATIO_PRIOR_MEAN,
        ]
    # Clamp warm-start values into bounds.
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        x0[i] = max(lo, min(hi, x0[i]))

    n_data = m
    n_priors = len(prior_mu)

    # Sim-error PEM residual: forward-simulate state x = [T_a, T_w] via
    # matrix-exponential with HP-as-feedback in A; data residual =
    # predicted T_a − observed T_a (in °C). Replaces the prior rate-
    # residual / HP-as-input formulation that produced biased β_outdoor
    # and ~14× underestimate of raw RC params on real CSV (project_grey-
    # box_2r2c_real_csv_finding.md, project_greybox_rate_convention_bug.md).
    # Formulation matches Probe 8 of the multi-restart probe series:
    #
    #   Active:   dx/dt = A_active   · x + b_active
    #             A_active   = [[-(ua_c + k_c + k_w),  k_w           ],
    #                           [ k_w/mr,             -k_w/mr        ]]
    #             b_active[0]   = c0 + ua_c·t_out + k_c·hp_setpoint
    #                                 + α_air·solar
    #
    #   Inactive: dx/dt = A_inactive · x + b_inactive
    #             A_inactive = [[-(ua_c + k_w),         k_w           ],
    #                           [ k_w/mr,              -k_w/mr        ]]
    #             b_inactive[0] = c0 + ua_c·t_out + α_air·solar
    #
    #   Always:   b[1] = α_wall·solar / mr
    #
    # Zero-order hold inputs at start of interval:
    #   x(i) = exp(A·dt)·x(i-1) + ψ(dt)·b(i-1)
    #   ψ(dt) = A⁻¹·(exp(A·dt) − I)
    #
    # Matrix exp + ψ are precomputed per (A_type, dt) pair via dt-memoization;
    # the closed-form 2×2 expm below makes each entry ~1 μs to build.
    # Lit-canonical output-error PEM (Ljung) — what Bacher-Madsen 2011 /
    # Hollick 2020 / CTSM-R use, modulo the Kalman filter (which Probe 8
    # didn't include and which we're testing whether priors substitute for).
    import numpy as np  # local-import to honour scipy-optional pattern

    # Closed-form 2×2 matrix exp via Sylvester projectors. Valid because:
    #   trace(A) = -(ua_c + k_w + k_w/mr)            < 0
    #   det(A)   = ua_c · k_w / mr   (inactive)        > 0   (k_c shift same form)
    #   discr    = trace² − 4·det     > 0  across full UA_C × K_C × K_W × MASS_RATIO box
    #     (verified by sampling 200k random points; min 2.2e-6, rel eigenvalue
    #      separation > 0.7)
    # → distinct real eigenvalues, so:
    #
    #     expm(A·dt) = e^(λ₁·dt)·P₁ + e^(λ₂·dt)·P₂
    #     ψ(dt) := A⁻¹·(expm(A·dt) − I)
    #            = (expm1(λ₁·dt)/λ₁)·P₁ + (expm1(λ₂·dt)/λ₂)·P₂
    #
    # where P₁ = (A − λ₂·I)/(λ₁ − λ₂),  P₂ = I − P₁ (Sylvester).
    # numpy.expm1 maintains precision for |λ·dt| → 0; (eˣ−1)/x naïve
    # cancels catastrophically there.
    #
    # Replaces scipy.linalg.expm + numpy.linalg.solve (~150 μs general
    # Padé+LU) with ~1 μs closed-form. The scipy path is kept as fallback
    # if the discriminant ever falls below threshold (would only happen
    # if bounds change to allow A to drift into a degenerate regime).
    _expm_eye2 = np.eye(2)

    def _expm_psi(A, dt):
        a00, a01 = A[0, 0], A[0, 1]
        a10, a11 = A[1, 0], A[1, 1]
        trace = a00 + a11
        det = a00 * a11 - a01 * a10
        discr = trace * trace - 4.0 * det
        if discr <= 1e-12:
            # Defensive fallback — unreachable across current bounds.
            eA_fb = _expm(A * dt)
            try:
                psi_fb = np.linalg.solve(A, eA_fb - _expm_eye2)
            except np.linalg.LinAlgError:
                psi_fb = np.zeros((2, 2))
            return eA_fb, psi_fb
        sqrt_d = math.sqrt(discr)
        lam1 = (trace + sqrt_d) * 0.5
        lam2 = (trace - sqrt_d) * 0.5
        inv_dlam = 1.0 / (lam1 - lam2)
        # P1 = (A − λ₂·I) · inv_dlam.  Only need P1: P2 = I − P1.
        p1_00 = (a00 - lam2) * inv_dlam
        p1_01 = a01 * inv_dlam
        p1_10 = a10 * inv_dlam
        p1_11 = (a11 - lam2) * inv_dlam
        e1 = math.exp(lam1 * dt)
        e2 = math.exp(lam2 * dt)
        de = e1 - e2
        eA = np.array([
            [de * p1_00 + e2,  de * p1_01      ],
            [de * p1_10,       de * p1_11 + e2 ],
        ], dtype=float)
        # ψ = (expm1(λ₁·dt)/λ₁)·P₁ + (expm1(λ₂·dt)/λ₂)·P₂
        psi_1 = math.expm1(lam1 * dt) / lam1
        psi_2 = math.expm1(lam2 * dt) / lam2
        dp = psi_1 - psi_2
        psi = np.array([
            [dp * p1_00 + psi_2,  dp * p1_01         ],
            [dp * p1_10,           dp * p1_11 + psi_2 ],
        ], dtype=float)
        return eA, psi

    # Per-fit dt-memoization. The set of distinct positive dt values is
    # data-only (depends on dt_min, not params), so compute it once outside
    # residual_fn. Inside, build per-(A_type, dt) (eA, ψ) caches up front;
    # the inner loop becomes a dict lookup. Reduces expm cost per
    # residual_fn from O(m) to O(unique_dts) — typically 30–100 vs 10000
    # for sorted-but-eviction-sparse buffers.
    _unique_dts_pos = sorted({d for d in dt_min if d > 0})

    # v2 Bayesian: k_w and mass_ratio are FREE parameters constrained
    # by Tikhonov priors (appended to residuals below). A matrices must
    # be rebuilt every iteration since wall structure varies with the
    # fitted params now.

    def residual_fn(params: list[float]) -> list[float]:
        if has_solar:
            c0, ua_c, k_c, alpha_total, k_w, mass_ratio = params
        else:
            c0, ua_c, k_c, k_w, mass_ratio = params
            alpha_total = 0.0
        alpha_air = alpha_total * SOLAR_AIR_FRACTION
        alpha_wall = alpha_total * SOLAR_WALL_FRACTION
        a_wall_rate = k_w / mass_ratio

        A_active = np.array([
            [-(ua_c + k_c + k_w),  k_w           ],
            [ a_wall_rate,        -a_wall_rate   ],
        ], dtype=float)
        A_inactive = np.array([
            [-(ua_c + k_w),         k_w           ],
            [ a_wall_rate,         -a_wall_rate   ],
        ], dtype=float)

        # Build flat-tuple (eA, ψ) caches for every distinct dt. Tuples of
        # 8 scalars (eA_00, eA_01, eA_10, eA_11, ψ_00, ψ_01, ψ_10, ψ_11)
        # let the inner loop do pure-scalar arithmetic — no per-tick
        # numpy array allocations, no `@` matmul overhead. The inner loop
        # runs O(m × scalar_ops) instead of O(m × numpy_op_overhead);
        # 10× faster than the array-based form on a 10000-tick buffer.
        cache_active: dict[float, tuple] = {}
        cache_inactive: dict[float, tuple] = {}
        try:
            for dt_u in _unique_dts_pos:
                eA_a, psi_a = _expm_psi(A_active, dt_u)
                eA_i, psi_i = _expm_psi(A_inactive, dt_u)
                cache_active[dt_u] = (
                    eA_a[0, 0], eA_a[0, 1], eA_a[1, 0], eA_a[1, 1],
                    psi_a[0, 0], psi_a[0, 1], psi_a[1, 0], psi_a[1, 1],
                )
                cache_inactive[dt_u] = (
                    eA_i[0, 0], eA_i[0, 1], eA_i[1, 0], eA_i[1, 1],
                    psi_i[0, 0], psi_i[0, 1], psi_i[1, 0], psi_i[1, 1],
                )
        except Exception:
            # Numerical failure (e.g. expm overflow at extreme params):
            # return large residuals so optimizer steers away. Length
            # must include prior terms so least_squares sees a
            # consistent residual vector size across iterations.
            return [1e6] * (n_data + n_priors)

        # Initial state: assume wall at air temperature at t=0 (equilibrium
        # is the best we can do from a single observation).
        x0 = x1 = t_air[0]
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
            # bench_form active flag: HP delivers heating only when
            # room_temp < setpoint (matches thermal_model.py:315 thermo-
            # static cycling). Per probe truth-validation, this halves
            # tracking RMS vs the simpler `sp is not None` proxy.
            # Heating-mode assumption (bench is heating-only).
            active_prev = (sp_prev is not None) and (t_a_prev < sp_prev)
            if active_prev:
                (eA00, eA01, eA10, eA11,
                 ps00, ps01, ps10, ps11) = cache_active[dt]
                b1 = (c0 + ua_c * t_out[i - 1] + k_c * sp_prev
                      + alpha_air * solar[i - 1])
            else:
                (eA00, eA01, eA10, eA11,
                 ps00, ps01, ps10, ps11) = cache_inactive[dt]
                b1 = (c0 + ua_c * t_out[i - 1] + alpha_air * solar[i - 1])
            b2 = alpha_wall * solar[i - 1] / mass_ratio
            # x_new = eA · x + ψ · b  expanded to scalars to skip numpy overhead.
            x0_new = eA00 * x0 + eA01 * x1 + ps00 * b1 + ps01 * b2
            x1_new = eA10 * x0 + eA11 * x1 + ps10 * b1 + ps11 * b2
            x0, x1 = x0_new, x1_new
            residuals[i] = x0 - t_air[i]

        # Tikhonov prior penalty terms appended to the data residuals.
        # Each term is (θ_i − μ_i) / σ_i; scipy's sum-of-squares loss
        # makes the contribution to the objective equal to ½·(θ_i−μ_i)²/σ_i²,
        # i.e. the negative log of an independent Gaussian prior up to
        # a constant. The MAP optimum is the inverse-variance-weighted
        # blend of likelihood (data residuals) and priors. CTSM-R MAP /
        # Pathak 2019 §3.1 BSSM (with the simplification that scipy's
        # huber loss is a robust likelihood, not an exact Gaussian).
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
    k_c = float(params_fit["k_c"])
    alpha_total = float(params_fit.get("alpha_c", 0.0))
    # v2 Bayesian: k_w and mass_ratio are fitted (constrained by Tikhonov
    # priors, see comments at parameter packing above).
    k_w = float(params_fit["k_w"])
    mass_ratio = float(params_fit["mass_ratio"])

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
        k_c=k_c,
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
