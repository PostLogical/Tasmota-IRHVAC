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
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .batch_learning import Observation

_LOGGER = logging.getLogger(__name__)

# Try to import scipy; gracefully degrade if unavailable.
try:
    from scipy.optimize import least_squares as _least_squares
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

# 2R2C Bayesian priors on the parameter pair that's not separately
# identifiable from operational rate-residual data (Marty-Stabat 2022 —
# these two are correlated in the regressor; the optimizer can rail one
# against the other to fit residuals). Priors anchor at literature values;
# implemented as ridge-style penalty residuals appended to the least-
# squares loss (Hollick 2020 occupied-home approach, MAP estimation with
# Gaussian priors).
#
# Prior means: lit-typical residential per Bacher-Madsen 2011 (mass_ratio
# 5-10, τ_couple 30-80 min so k_w ≈ 0.012-0.033).
#
# Prior stds: empirically tuned. mass_ratio at std=5 lets data move it
# noticeably when informative (CV before prior was ~0.5, often railing
# at 30; with prior, CV drops near 0 only when data carries no signal).
# k_w at std=0.001 is effectively a pin — under rate-residual at 12-h
# batch cadence, k_w is not separately identifiable from ua_c·mass_ratio
# combinations, and any prior loose enough to "release with data" lets
# k_w rail at the lower bound (data fit improves at k_w → 0 since the
# wall-coupling term in the residual vanishes). Effective k_w pin matches
# what Hollick 2020 and Marty-Stabat 2022 advise for the operational-data
# regime: fix the parameter the data can't separately identify.
#
# Once sim-error PEM (or a Kalman state estimator) replaces the rate-
# residual formulation, k_w gains a real signal source and the prior std
# can be relaxed.
MASS_RATIO_PRIOR_MEAN = 8.0
MASS_RATIO_PRIOR_STD = 5.0   # 95% CI ≈ [-2, 18], clipped by bounds [1, 30]
K_W_PRIOR_MEAN = 1.0 / 50.0  # τ_couple = 50 min
K_W_PRIOR_STD = 0.001         # effective pin under rate-residual; relax post-PEM

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
    r_2r2c = _fit_greybox_2r2c(eligible, model_inputs, r_1r1c, plant_tau_slow)
    if r_2r2c is None:
        _LOGGER.info("Grey-box: 2R2C fit failed — falling back to 1R1C")
        return r_1r1c

    return r_2r2c


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
    if result.jac is not None:
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

    if solar_entity is not None:
        solar = [o.raw_readings.get(solar_entity, 0.0) for o in eligible]
    else:
        solar = [0.0] * m
    has_solar = solar_entity is not None and any(abs(s) > 1e-6 for s in solar)

    hp_mean = sum(hp_offset) / m
    hp_var = sum((h - hp_mean) ** 2 for h in hp_offset) / m
    if hp_var < MIN_HP_VARIANCE:
        # Without HP-on data we can't separate k_c from ua_c — same constraint
        # as 1R1C. Fall back rather than fit a degenerate model.
        return None

    # Pre-compute dt[i] = (t[i] - t[i-1]) / 60 (minutes); first entry unused.
    dt_min: list[float] = [0.0] * m
    for i in range(1, m):
        d = (timestamps[i] - timestamps[i - 1]) / 60.0
        dt_min[i] = d if d > 0 else 0.0

    # Parameter packing.
    # Without solar: [c0, ua_c, k_c, k_w, mass_ratio]
    # With solar:    [c0, ua_c, k_c, alpha_total, k_w, mass_ratio]
    if has_solar:
        param_names = ["c0", "ua_c", "k_c", "alpha_c", "k_w", "mass_ratio"]
        lower = [
            C0_BOUNDS[0], UA_C_BOUNDS[0], K_C_BOUNDS[0],
            ALPHA_C_BOUNDS[0], K_W_BOUNDS[0], MASS_RATIO_BOUNDS[0],
        ]
        upper = [
            C0_BOUNDS[1], UA_C_BOUNDS[1], K_C_BOUNDS[1],
            ALPHA_C_BOUNDS[1], K_W_BOUNDS[1], MASS_RATIO_BOUNDS[1],
        ]
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

    # Warm start from 1R1C result, with plant ID hint on ua_c if available.
    ua_c_init = r_1r1c.ua_c
    if plant_tau_slow is not None and plant_tau_slow > 0:
        ua_c_init = max(UA_C_BOUNDS[0], min(UA_C_BOUNDS[1], 1.0 / plant_tau_slow))
    # Bacher-Madsen typical residential: τ_couple ≈ 30–80 min, mass_ratio 5–10.
    k_w_init = 1.0 / 50.0
    mass_ratio_init = 8.0

    if has_solar:
        x0 = [
            r_1r1c.c0, ua_c_init, r_1r1c.k_c,
            max(ALPHA_C_BOUNDS[0], r_1r1c.alpha_c),
            k_w_init, mass_ratio_init,
        ]
    else:
        x0 = [
            r_1r1c.c0, ua_c_init, r_1r1c.k_c,
            k_w_init, mass_ratio_init,
        ]
    # Clamp warm-start values into bounds.
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        x0[i] = max(lo, min(hi, x0[i]))

    # Penalty residuals for Bayesian priors on (mass_ratio, k_w). Each
    # contributes (param - prior_mean) / prior_std as an extra residual,
    # which least_squares squares and adds to the loss. Mathematically
    # equivalent to MAP estimation with Gaussian priors on those two
    # parameters. When data is informative on these params (low CV),
    # the data residuals dominate and the prior is gently pulled toward.
    # When data is uninformative (the rail-saturation case), the prior
    # holds the params at literature-typical values rather than letting
    # the optimizer route them to the corner.
    n_data = m  # data residuals come first; penalties appended after
    n_penalties = 2  # (mass_ratio, k_w)

    def residual_fn(params: list[float]) -> list[float]:
        if has_solar:
            c0, ua_c, k_c, alpha_total, k_w, mass_ratio = params
        else:
            c0, ua_c, k_c, k_w, mass_ratio = params
            alpha_total = 0.0
        alpha_air = alpha_total * SOLAR_AIR_FRACTION
        alpha_wall = alpha_total * SOLAR_WALL_FRACTION
        a_wall = k_w / mass_ratio  # wall-side rate (min⁻¹)

        # Initial wall state: assume equilibrium with air at t=0.
        t_wall = t_air[0]
        residuals = [0.0] * (n_data + n_penalties)

        # First tick: t_wall = t_air → coupling term contributes 0.
        residuals[0] = (
            room_rate[0] - c0
            - ua_c * (t_out[0] - t_air[0])
            - k_c * hp_offset[0]
            - alpha_air * solar[0]
        )

        for i in range(1, m):
            dt = dt_min[i]
            # Step t_wall using piecewise-exponential integration.
            # dT_w/dt = a_wall*(T_a - T_w) + (α_wall/mass_ratio)*solar
            # Equilibrium with previous T_a, solar held constant over dt:
            #   T_w_eq = T_a + α_wall*solar / k_w
            # (a_wall * mass_ratio = k_w cancels the mass_ratio in the
            #  forcing scaled by C_wall.)
            if dt > 0:
                t_w_eq = t_air[i - 1] + alpha_wall * solar[i - 1] / k_w
                decay = math.exp(-a_wall * dt)
                t_wall = t_w_eq + (t_wall - t_w_eq) * decay

            residuals[i] = (
                room_rate[i] - c0
                - ua_c * (t_out[i] - t_air[i])
                - k_c * hp_offset[i]
                - alpha_air * solar[i]
                - k_w * (t_wall - t_air[i])
            )

        # Bayesian prior penalty residuals (Gaussian, ridge-style). Scaled
        # by 1/prior_std so contribution is comparable across parameters
        # regardless of natural magnitude.
        residuals[n_data] = (mass_ratio - MASS_RATIO_PRIOR_MEAN) / MASS_RATIO_PRIOR_STD
        residuals[n_data + 1] = (k_w - K_W_PRIOR_MEAN) / K_W_PRIOR_STD
        return residuals

    try:
        result = _least_squares(
            residual_fn, x0,
            bounds=(lower, upper),
            method="trf",
            loss="huber",
            f_scale=0.005,
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
    k_w = float(params_fit["k_w"])
    mass_ratio = float(params_fit["mass_ratio"])

    tau_fast, tau_slow = _natural_eigenvalues(ua_c, k_w, mass_ratio)
    tau_eff = tau_slow  # dominant for legacy consumers
    # residual_rms reflects data-fit quality only — exclude the prior penalty
    # residuals (last n_penalties entries of result.fun) so the gate
    # threshold stays comparable to 1R1C and pre-prior 2R2C runs.
    if result.fun is not None and m > 0:
        data_residuals = result.fun[:n_data]
        residual_rms = math.sqrt(
            sum(float(r) * float(r) for r in data_residuals) / m
        )
    else:
        residual_rms = 0.0

    # Standard errors via Jacobian. sigma² uses data residuals only so the
    # std_errs reflect data-fit uncertainty; the prior contributions don't
    # inflate DoF the way independent observations do.
    std_err: dict[str, float] = {}
    if result.jac is not None:
        try:
            import numpy as np
            J = result.jac
            n_params = len(param_names)
            data_cost = 0.5 * sum(
                float(r) * float(r) for r in result.fun[:n_data]
            ) if result.fun is not None else result.cost
            sigma2 = 2.0 * data_cost / max(1, m - n_params)
            JtJ_inv = np.linalg.pinv(J.T @ J) * sigma2
            for i, name in enumerate(param_names):
                var = JtJ_inv[i, i]
                std_err[name] = math.sqrt(max(0.0, float(var)))
        except Exception:
            pass

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

    if result.param_std_err:
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
GATE_MAX_RMS = 0.02    # °C/min — residual quality threshold


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

    # Gate 4: residual quality
    gates["residual_rms"] = result.residual_rms < GATE_MAX_RMS

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
