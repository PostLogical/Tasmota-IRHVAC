"""Multi-restart Maximum Likelihood (PEM) fitting of RC grey-box models.

Per Bacher-Madsen 2011 / CTSM-R Kristensen-Madsen-Jorgensen 2004:
- Continuous-discrete state-space SDE, ML via Kalman conditional likelihood
- Box bounds enforce physical plausibility (CTSM-R user guide convention)
- Multi-restart with random log-uniform initial values
- Coefficient-of-variation across restarts as practical-identifiability gate
  (Cárdenas-Rangel 2022 threshold: CV ≤ 10% per parameter)
- At-bound flagging surfaces Reynders-2014-style non-identifiability rails

Reduced parameterization:
    1R1C: (tau_s, q_scale, solar_scale, sigma_w, sigma_v)  — 5 params
    2R2C: (tau_air_s, tau_wall_s, coupling_ratio, q_scale, solar_scale,
           sigma_w_i, sigma_w_e, sigma_v)                   — 8 params

C is held fixed because (R, C) is non-identifiable from temperature alone
without heat-flux measurement (Reynders 2014). q_scale and solar_scale
absorb the C scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Final

import numpy as np
from scipy.optimize import minimize

from tests.hvac_bench.empirical.rc_model import (
    RCParams1R1C,
    RCParams2R2C,
    StateSpace,
    build_1r1c,
    build_2r2c,
    kalman_log_likelihood,
)


# ── Reduced fit parameter dataclasses ────────────────────────────────────


C_NOMINAL_1R1C: Final = 1.0e7  # J/K, lumped residential capacitance
C_I_NOMINAL_2R2C: Final = 5.0e6  # J/K, interior-air capacitance


@dataclass(frozen=True)
class FitParams1R1C:
    """Reduced-ID parameters for 1R1C; C held fixed at C_nominal."""

    tau_s: float
    q_scale: float
    solar_scale: float
    sigma_w: float
    sigma_v: float
    C_nominal: float = C_NOMINAL_1R1C

    def to_full(self) -> RCParams1R1C:
        return RCParams1R1C(
            R=self.tau_s / self.C_nominal,
            C=self.C_nominal,
            q_scale=self.q_scale,
            solar_scale=self.solar_scale,
            sigma_w=self.sigma_w,
            sigma_v=self.sigma_v,
        )

    def to_log_array(self) -> np.ndarray:
        return np.log(
            np.array(
                [
                    self.tau_s,
                    self.q_scale,
                    self.solar_scale,
                    self.sigma_w,
                    self.sigma_v,
                ]
            )
        )

    @classmethod
    def from_log_array(
        cls,
        x: np.ndarray,
        C_nominal: float = C_NOMINAL_1R1C,
    ) -> FitParams1R1C:
        v = np.exp(x)
        return cls(
            tau_s=float(v[0]),
            q_scale=float(v[1]),
            solar_scale=float(v[2]),
            sigma_w=float(v[3]),
            sigma_v=float(v[4]),
            C_nominal=C_nominal,
        )


@dataclass(frozen=True)
class FitParams2R2C:
    """Reduced-ID parameters for 2R2C TiTe; C_i held fixed."""

    tau_air_s: float  # R_ie · C_i
    tau_wall_s: float  # R_ea · C_e
    coupling_ratio: float  # C_e / C_i (mass ratio)
    q_scale: float
    solar_scale: float
    sigma_w_i: float
    sigma_w_e: float
    sigma_v: float
    wall_solar_fraction: float = 0.7
    C_i_nominal: float = C_I_NOMINAL_2R2C

    def to_full(self) -> RCParams2R2C:
        C_i = self.C_i_nominal
        C_e = self.coupling_ratio * C_i
        return RCParams2R2C(
            R_ie=self.tau_air_s / C_i,
            R_ea=self.tau_wall_s / C_e,
            C_i=C_i,
            C_e=C_e,
            q_scale=self.q_scale,
            solar_scale=self.solar_scale,
            sigma_w_i=self.sigma_w_i,
            sigma_w_e=self.sigma_w_e,
            sigma_v=self.sigma_v,
            wall_solar_fraction=self.wall_solar_fraction,
        )

    def to_log_array(self) -> np.ndarray:
        return np.log(
            np.array(
                [
                    self.tau_air_s,
                    self.tau_wall_s,
                    self.coupling_ratio,
                    self.q_scale,
                    self.solar_scale,
                    self.sigma_w_i,
                    self.sigma_w_e,
                    self.sigma_v,
                ]
            )
        )

    @classmethod
    def from_log_array(
        cls,
        x: np.ndarray,
        wall_solar_fraction: float = 0.7,
        C_i_nominal: float = C_I_NOMINAL_2R2C,
    ) -> FitParams2R2C:
        v = np.exp(x)
        return cls(
            tau_air_s=float(v[0]),
            tau_wall_s=float(v[1]),
            coupling_ratio=float(v[2]),
            q_scale=float(v[3]),
            solar_scale=float(v[4]),
            sigma_w_i=float(v[5]),
            sigma_w_e=float(v[6]),
            sigma_v=float(v[7]),
            wall_solar_fraction=wall_solar_fraction,
            C_i_nominal=C_i_nominal,
        )


# ── Bounds ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Bounds:
    """Box bounds in original parameter space (not log)."""

    lower: dict[str, float]
    upper: dict[str, float]

    def keys(self) -> tuple[str, ...]:
        return tuple(self.lower)

    def log_arrays(self, keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([np.log(self.lower[k]) for k in keys]),
            np.array([np.log(self.upper[k]) for k in keys]),
        )


# Bounds derived from the lit pass:
# - τ_air ∈ [5 min, 4 h]      Madsen-Holst 1995 air mode
# - τ_slow ∈ [4 h, 200 h]     Levermore 2020 envelope mode
# - q_scale ∈ [0.01, 100]     wide; centered at 1.0 (proxy is already W-scaled)
# - solar_scale ∈ [1e-6, 100] wide; the right magnitude is unclear a priori
# - sigma_w ∈ [1e-5, 1.0]     K·s^(-1/2) process diffusion
# - sigma_v ∈ [0.02, 2.0]     K sensor noise; HA reports 0.5°F = 0.28°C typical
# 1R1C uses a single tau covering the whole [5min, 200h] range to allow the
# fitter to land where the data wants.

DEFAULT_BOUNDS_1R1C: Final = Bounds(
    lower={
        "tau_s": 30.0 * 60,  # 30 min
        "q_scale": 0.01,
        "solar_scale": 1e-6,
        "sigma_w": 1e-5,
        "sigma_v": 0.02,
    },
    upper={
        "tau_s": 200.0 * 3600,  # 200 h
        "q_scale": 100.0,
        "solar_scale": 100.0,
        "sigma_w": 1.0,
        "sigma_v": 2.0,
    },
)


DEFAULT_BOUNDS_2R2C: Final = Bounds(
    lower={
        "tau_air_s": 5.0 * 60,  # 5 min
        "tau_wall_s": 4.0 * 3600,  # 4 h
        "coupling_ratio": 0.5,
        "q_scale": 0.01,
        "solar_scale": 1e-6,
        "sigma_w_i": 1e-5,
        "sigma_w_e": 1e-5,
        "sigma_v": 0.02,
    },
    upper={
        "tau_air_s": 4.0 * 3600,  # 4 h
        "tau_wall_s": 200.0 * 3600,  # 200 h
        "coupling_ratio": 100.0,
        "q_scale": 100.0,
        "solar_scale": 100.0,
        "sigma_w_i": 1.0,
        "sigma_w_e": 1.0,
        "sigma_v": 2.0,
    },
)


# ── Result dataclasses ───────────────────────────────────────────────────


FitParams = FitParams1R1C | FitParams2R2C


@dataclass
class RestartResult:
    """Result of one optimizer run from one starting point."""

    success: bool
    log_likelihood: float
    params: FitParams
    initial: FitParams
    n_iter: int
    message: str


@dataclass
class FitResult:
    """Aggregated multi-restart fit."""

    model_name: str  # "1R1C" or "2R2C"
    best: RestartResult
    restarts: list[RestartResult]
    n_obs: int
    cv_per_param: dict[str, float] = field(default_factory=dict)
    at_bound_per_param: dict[str, bool] = field(default_factory=dict)
    bounds: Bounds = field(default_factory=lambda: DEFAULT_BOUNDS_1R1C)

    def is_practically_identifiable(self, cv_threshold: float = 0.1) -> dict[str, bool]:
        """Per Cárdenas-Rangel 2022: CV ≤ 10% across restarts."""
        return {k: v < cv_threshold for k, v in self.cv_per_param.items()}


# ── Multi-restart optimizer ──────────────────────────────────────────────


def _optimize_one(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds_log: list[tuple[float, float]],
    *,
    maxiter: int = 200,
) -> tuple[bool, float, np.ndarray, int, str]:
    """One L-BFGS-B run from a single start. Returns convergence info."""
    try:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds_log,
            options={"maxiter": maxiter, "gtol": 1e-4},
        )
        return (
            bool(result.success),
            float(-result.fun),
            np.asarray(result.x),
            int(result.nit),
            str(result.message),
        )
    except Exception as e:
        return (False, -np.inf, x0, 0, f"exception: {e}")


def _compute_cv(values: np.ndarray) -> float:
    """Coefficient of variation. Returns inf if mean magnitude is ~0."""
    mean = float(np.mean(values))
    if abs(mean) < 1e-30:
        return float("inf")
    return float(np.std(values) / abs(mean))


def _at_bound(value: float, lo: float, hi: float, *, tol_log: float = 0.05) -> bool:
    """True if value is within tol_log of either log-bound."""
    log_v = np.log(max(value, 1e-30))
    return (log_v - np.log(lo)) < tol_log or (np.log(hi) - log_v) < tol_log


def fit_1r1c(
    observations: np.ndarray,
    inputs: np.ndarray,
    valid: np.ndarray,
    dt: float,
    *,
    n_restarts: int = 4,
    bounds: Bounds = DEFAULT_BOUNDS_1R1C,
    seed: int = 0,
    C_nominal: float = C_NOMINAL_1R1C,
    maxiter: int = 200,
) -> FitResult:
    """Multi-restart PEM fit of 1R1C model.

    Each restart samples log-uniform within bounds, runs L-BFGS-B in log-space,
    and records the result. CV per parameter is computed across *successful*
    restarts; if fewer than 2 restarts succeed, CV is empty and identifiability
    is unverifiable.
    """
    rng = np.random.default_rng(seed)
    keys = ("tau_s", "q_scale", "solar_scale", "sigma_w", "sigma_v")
    log_lo, log_hi = bounds.log_arrays(keys)
    bounds_pairs = list(zip(log_lo.tolist(), log_hi.tolist()))

    def neg_ll(x_log: np.ndarray) -> float:
        try:
            params = FitParams1R1C.from_log_array(x_log, C_nominal=C_nominal)
            ss = build_1r1c(params.to_full(), dt=dt)
        except (ValueError, np.linalg.LinAlgError):
            return 1e10
        ll = kalman_log_likelihood(ss, observations, inputs, valid)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    restarts: list[RestartResult] = []
    for _ in range(n_restarts):
        x0 = log_lo + rng.random(len(log_lo)) * (log_hi - log_lo)
        success, ll, x_opt, n_iter, msg = _optimize_one(
            neg_ll, x0, bounds_pairs, maxiter=maxiter
        )
        restarts.append(
            RestartResult(
                success=success,
                log_likelihood=ll,
                params=FitParams1R1C.from_log_array(x_opt, C_nominal=C_nominal),
                initial=FitParams1R1C.from_log_array(x0, C_nominal=C_nominal),
                n_iter=n_iter,
                message=msg,
            )
        )

    best = max(restarts, key=lambda r: r.log_likelihood)
    successful = [r for r in restarts if r.success and np.isfinite(r.log_likelihood)]

    cv_per_param: dict[str, float] = {}
    if len(successful) >= 2:
        for k in keys:
            cv_per_param[k] = _compute_cv(
                np.array([getattr(r.params, k) for r in successful])
            )

    at_bound: dict[str, bool] = {}
    for k in keys:
        at_bound[k] = _at_bound(
            getattr(best.params, k), bounds.lower[k], bounds.upper[k]
        )

    return FitResult(
        model_name="1R1C",
        best=best,
        restarts=restarts,
        n_obs=int(np.sum(valid)),
        cv_per_param=cv_per_param,
        at_bound_per_param=at_bound,
        bounds=bounds,
    )


def fit_2r2c(
    observations: np.ndarray,
    inputs: np.ndarray,
    valid: np.ndarray,
    dt: float,
    *,
    n_restarts: int = 4,
    bounds: Bounds = DEFAULT_BOUNDS_2R2C,
    seed: int = 0,
    C_i_nominal: float = C_I_NOMINAL_2R2C,
    wall_solar_fraction: float = 0.7,
    maxiter: int = 200,
) -> FitResult:
    """Multi-restart PEM fit of 2R2C TiTe model. See `fit_1r1c` docstring."""
    rng = np.random.default_rng(seed)
    keys = (
        "tau_air_s",
        "tau_wall_s",
        "coupling_ratio",
        "q_scale",
        "solar_scale",
        "sigma_w_i",
        "sigma_w_e",
        "sigma_v",
    )
    log_lo, log_hi = bounds.log_arrays(keys)
    bounds_pairs = list(zip(log_lo.tolist(), log_hi.tolist()))

    def neg_ll(x_log: np.ndarray) -> float:
        try:
            params = FitParams2R2C.from_log_array(
                x_log,
                wall_solar_fraction=wall_solar_fraction,
                C_i_nominal=C_i_nominal,
            )
            ss = build_2r2c(params.to_full(), dt=dt)
        except (ValueError, np.linalg.LinAlgError):
            return 1e10
        ll = kalman_log_likelihood(ss, observations, inputs, valid)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    restarts: list[RestartResult] = []
    for _ in range(n_restarts):
        x0 = log_lo + rng.random(len(log_lo)) * (log_hi - log_lo)
        success, ll, x_opt, n_iter, msg = _optimize_one(
            neg_ll, x0, bounds_pairs, maxiter=maxiter
        )
        restarts.append(
            RestartResult(
                success=success,
                log_likelihood=ll,
                params=FitParams2R2C.from_log_array(
                    x_opt,
                    wall_solar_fraction=wall_solar_fraction,
                    C_i_nominal=C_i_nominal,
                ),
                initial=FitParams2R2C.from_log_array(
                    x0,
                    wall_solar_fraction=wall_solar_fraction,
                    C_i_nominal=C_i_nominal,
                ),
                n_iter=n_iter,
                message=msg,
            )
        )

    best = max(restarts, key=lambda r: r.log_likelihood)
    successful = [r for r in restarts if r.success and np.isfinite(r.log_likelihood)]

    cv_per_param: dict[str, float] = {}
    if len(successful) >= 2:
        for k in keys:
            cv_per_param[k] = _compute_cv(
                np.array([getattr(r.params, k) for r in successful])
            )

    at_bound: dict[str, bool] = {}
    for k in keys:
        at_bound[k] = _at_bound(
            getattr(best.params, k), bounds.lower[k], bounds.upper[k]
        )

    return FitResult(
        model_name="2R2C",
        best=best,
        restarts=restarts,
        n_obs=int(np.sum(valid)),
        cv_per_param=cv_per_param,
        at_bound_per_param=at_bound,
        bounds=bounds,
    )
