"""Grey-box 1R1C energy balance observer.

Observation-mode-only module -- fits a physically grounded energy balance
to the observation buffer and logs results.  Does not modify the online
RLS model.  Runs alongside the existing WLS batch on the same 12h schedule.

The 1R1C energy balance divided by C_eff:

    dT_air/dt = (UA/C) × (T_out - T_air) + (K_hp/C) × hp_offset + (α/C) × solar

Reparameterized as rate coefficients (Bacher & Madsen 2011):

    room_rate = ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar

where ua_c = UA/C, k_c = K_hp/C, α_c = α_solar/C.  These rate
coefficients are directly identifiable from derivative data without
the scaling ambiguity of the original parameterization.

The effective time constant τ_eff = 1/ua_c (minutes) is compared
against plant ID's τ_slow as a cross-validation check.

Key advantage over the static WLS batch: uses ALL data including HP-off
periods (hp_offset=0), which directly inform ua_c and α_c from room
temperature trajectory.

References:
- Bacher & Madsen, "Identifying suitable models for the heat dynamics
  of buildings" (2011) -- grey-box RC models for building identification
- Madsen & Holst, "Estimation of continuous-time models for the heat
  dynamics of a building" (1995) -- rate coefficient parameterization
- Ljung, "System Identification: Theory for the User" -- prediction error

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
#   Allow wider range for unusual buildings.
UA_C_BOUNDS = (0.001, 0.1)  # min⁻¹

# k_c = K_hp/C in °C_room / (°C_setpoint_offset × min).
# Similar magnitude to ua_c for a well-sized HP.
K_C_BOUNDS = (0.001, 0.2)  # min⁻¹

# α_c = α_solar/C.  Negative because solar reduces room_rate deficit.
# Magnitude depends on solar proxy scaling and C.
ALPHA_C_BOUNDS = (-1.0, 0.0)  # °C/min per unit solar proxy

# Minimum observations required for a meaningful fit.
MIN_OBSERVATIONS = 30

# Minimum variance in hp_offset column to identify k_c.
MIN_HP_VARIANCE = 0.01


@dataclass
class GreyboxResult:
    """Result of a grey-box 1R1C energy balance fit."""

    n_observations: int  # observations used in fit
    n_hp_on: int  # observations where HP was active
    n_hp_off: int  # observations where HP was off (hp_offset=0)

    # Fitted rate coefficients (÷ C_eff)
    ua_c: float  # UA/C envelope rate (min⁻¹)
    k_c: float  # K_hp/C HP gain rate (min⁻¹)
    alpha_c: float  # α_solar/C solar rate (0.0 if no solar input)

    # Derived
    tau_eff: float  # 1/ua_c (minutes) -- effective time constant
    residual_rms: float  # RMS of energy balance residual (°C/min)

    # Fit quality
    cost: float  # scipy cost function value
    n_function_evals: int  # number of function evaluations

    # Plant ID cross-check
    plant_tau_slow: float | None = None  # τ_slow from plant ID (minutes)
    tau_agreement_pct: float | None = None  # |τ_eff - τ_slow| / τ_slow × 100

    # Per-parameter confidence (from Jacobian if available)
    param_std_err: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "n_hp_on": self.n_hp_on,
            "n_hp_off": self.n_hp_off,
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
        }


def _find_solar_name(model_inputs: list[dict[str, Any]]) -> str | None:
    """Find the feature name of the solar proxy input, or None."""
    for m in model_inputs:
        if m.get("input_role") == "solar":
            return m.get("name")
    return None


def fit_greybox(
    observations: list[Observation],
    model_inputs: list[dict[str, Any]],
    plant_tau_slow: float | None = None,
    plant_tau_slow_confidence: float = 0.0,
) -> GreyboxResult | None:
    """Fit a 1R1C energy balance to observation buffer data.

    Uses all observations (including HP-off) that have valid outdoor_temp_c.
    The energy balance in rate-coefficient form:

        room_rate = ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar

    Rate coefficients are directly identifiable from derivative data
    (Bacher & Madsen 2011).  Bounded to physically plausible ranges.

    Args:
        observations: full buffer contents (all ticks, not pre-filtered).
        model_inputs: model input config dicts (to identify solar proxy).
        plant_tau_slow: τ_slow from plant ID (minutes), if available.
        plant_tau_slow_confidence: confidence in τ_slow (0-1).

    Returns:
        GreyboxResult or None if insufficient data or scipy unavailable.
    """
    if not SCIPY_AVAILABLE:
        _LOGGER.info(
            "Grey-box observer: scipy not available. "
            "Install scipy for advanced energy balance fitting."
        )
        return None

    solar_name = _find_solar_name(model_inputs)

    # Filter to observations with valid outdoor temperature.
    eligible = [
        o for o in observations
        if o.outdoor_temp_c is not None
    ]

    if len(eligible) < MIN_OBSERVATIONS:
        _LOGGER.debug(
            "Grey-box observer: insufficient observations (%d < %d)",
            len(eligible), MIN_OBSERVATIONS,
        )
        return None

    # Extract arrays for the fit.
    m = len(eligible)
    room_rate = [o.room_rate for o in eligible]
    t_air = [o.current_c for o in eligible]
    t_out = [o.outdoor_temp_c for o in eligible]

    # HP offset: setpoint - room temp when HP is on, 0 when off.
    hp_offset: list[float] = []
    n_hp_on = 0
    n_hp_off = 0
    for o in eligible:
        if o.clamped_reason == "no_output":
            hp_offset.append(0.0)
            n_hp_off += 1
        else:
            hp_offset.append(o.hp_setpoint - o.current_c)
            n_hp_on += 1

    # Solar proxy values (by feature name).
    solar: list[float]
    if solar_name is not None:
        solar = [o.features.get(solar_name, 0.0) for o in eligible]
    else:
        solar = [0.0] * m

    has_solar = solar_name is not None and any(abs(s) > 1e-6 for s in solar)

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
        # With k_c fixed, residual only fits ua_c and optionally α_c.
        k_c_fixed = 0.01  # ~τ=100min default
        if has_solar:
            def residual_fn(params: list[float]) -> list[float]:
                ua_c, alpha_c = params
                return [
                    room_rate[i]
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c_fixed * hp_offset[i]
                    - alpha_c * solar[i]
                    for i in range(m)
                ]
            x0 = [0.01, -0.05]
            lower = [UA_C_BOUNDS[0], ALPHA_C_BOUNDS[0]]
            upper = [UA_C_BOUNDS[1], ALPHA_C_BOUNDS[1]]
            param_names = ["ua_c", "alpha_c"]
        else:
            def residual_fn(params: list[float]) -> list[float]:
                (ua_c,) = params
                return [
                    room_rate[i]
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c_fixed * hp_offset[i]
                    for i in range(m)
                ]
            x0 = [0.01]
            lower = [UA_C_BOUNDS[0]]
            upper = [UA_C_BOUNDS[1]]
            param_names = ["ua_c"]
    else:
        if has_solar:
            def residual_fn(params: list[float]) -> list[float]:
                ua_c, k_c, alpha_c = params
                return [
                    room_rate[i]
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c * hp_offset[i]
                    - alpha_c * solar[i]
                    for i in range(m)
                ]
            x0 = [0.01, 0.02, -0.05]
            lower = [UA_C_BOUNDS[0], K_C_BOUNDS[0], ALPHA_C_BOUNDS[0]]
            upper = [UA_C_BOUNDS[1], K_C_BOUNDS[1], ALPHA_C_BOUNDS[1]]
            param_names = ["ua_c", "k_c", "alpha_c"]
        else:
            def residual_fn(params: list[float]) -> list[float]:
                ua_c, k_c = params
                return [
                    room_rate[i]
                    - ua_c * (t_out[i] - t_air[i])
                    - k_c * hp_offset[i]
                    for i in range(m)
                ]
            x0 = [0.01, 0.02]
            lower = [UA_C_BOUNDS[0], K_C_BOUNDS[0]]
            upper = [UA_C_BOUNDS[1], K_C_BOUNDS[1]]
            param_names = ["ua_c", "k_c"]

    # Use plant ID τ_slow to set initial ua_c if available.
    # τ = 1/ua_c → ua_c = 1/τ
    if plant_tau_slow is not None and plant_tau_slow > 0:
        ua_c_init = 1.0 / plant_tau_slow
        ua_c_init = max(UA_C_BOUNDS[0], min(UA_C_BOUNDS[1], ua_c_init))
        x0[0] = ua_c_init

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


def log_greybox_result(
    result: GreyboxResult,
    log_prefix: str = "",
) -> None:
    """Log grey-box fit results for diagnostics."""
    _LOGGER.info(
        "%sGrey-box 1R1C: %d obs (%d HP-on, %d HP-off), RMS=%.4f °C/min",
        log_prefix, result.n_observations, result.n_hp_on,
        result.n_hp_off, result.residual_rms,
    )
    _LOGGER.info(
        "%s  ua_c=%.5f (τ=%.0f min), k_c=%.5f, α_c=%.5f",
        log_prefix, result.ua_c, result.tau_eff, result.k_c, result.alpha_c,
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
