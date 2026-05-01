"""1R1C and 2R2C grey-box state-space models with Kalman log-likelihood.

Conventions follow Bacher-Madsen 2011 / CTSM-R (Kristensen-Madsen-Jorgensen 2004).

Continuous-time linear SDE:
    dx = A_c x dt + B_c u dt + dω,  cov(dω) = Sigma_c dt
    y_k = H x_k + v_k,                cov(v_k) = R

Discretized via Van Loan 1978 to discrete-time:
    x_{k+1} = A_d x_k + B_d u_k + w_k,  cov(w_k) = Q_d
    y_k     = H x_k + v_k

Kalman log-likelihood per Kristensen-Madsen-Jorgensen 2004 §3:
    log L(θ) = Σ_k [-0.5 (log(2π S_k) + e_k² / S_k)]

with `valid` mask skipping update at rows where measurement is missing.

Heat-input convention for 2R2C (matches bench `ThermalModel2R2C`):
    q_heat → 100% air node (HP heads blow air directly)
    shortwave_w_m2 × solar_scale → split between air (1-wall_solar_fraction)
                                    and wall (wall_solar_fraction). Default
                                    wall_solar_fraction = 0.7 per ASHRAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.linalg import expm


# ── Parameter dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class RCParams1R1C:
    """1R1C grey-box parameters in SI units (K, W, J, s).

    State:    x = [T_i]
    Inputs:   u = [T_outdoor (°C), q_heat_proxy (W), shortwave (W/m²)]
    """

    R: float  # K/W (envelope thermal resistance)
    C: float  # J/K (lumped thermal capacitance)
    q_scale: float  # dimensionless multiplier on q_heat_proxy_w
    solar_scale: float  # K·m²/W (lumped solar admittance)
    sigma_w: float  # K·s^(-1/2) (continuous-time process-noise diffusion)
    sigma_v: float  # K (measurement noise standard deviation)

    @property
    def tau_seconds(self) -> float:
        return self.R * self.C

    @property
    def n_states(self) -> int:
        return 1

    def validate(self) -> None:
        if self.R <= 0 or self.C <= 0:
            raise ValueError(f"R={self.R}, C={self.C} must be positive")
        if self.sigma_w < 0 or self.sigma_v <= 0:
            raise ValueError(f"sigma_w={self.sigma_w} ≥0, sigma_v={self.sigma_v} >0")


@dataclass(frozen=True)
class RCParams2R2C:
    """2R2C TiTe grey-box parameters per Bacher-Madsen 2011.

    State:    x = [T_i, T_e]      interior, envelope
    Inputs:   u = [T_outdoor (°C), q_heat_proxy (W), shortwave (W/m²)]

    Default wall_solar_fraction=0.7 matches the bench's ThermalModel2R2C
    convention (ASHRAE: ~30% solar to air convective, ~70% to wall radiative).
    """

    R_ie: float  # K/W (interior↔envelope)
    R_ea: float  # K/W (envelope↔ambient)
    C_i: float  # J/K (interior capacitance)
    C_e: float  # J/K (envelope capacitance)
    q_scale: float  # dimensionless
    solar_scale: float  # K·m²/W
    sigma_w_i: float  # K·s^(-1/2) (interior process-noise diffusion)
    sigma_w_e: float  # K·s^(-1/2) (envelope process-noise diffusion)
    sigma_v: float  # K
    wall_solar_fraction: float = 0.7

    @property
    def tau_air(self) -> float:
        """Air-mode time constant ~ R_ie·C_i (s). Approximate; coupled system."""
        return self.R_ie * self.C_i

    @property
    def tau_wall(self) -> float:
        """Wall-mode time constant ~ R_ea·C_e (s). Approximate."""
        return self.R_ea * self.C_e

    @property
    def n_states(self) -> int:
        return 2

    def validate(self) -> None:
        if self.R_ie <= 0 or self.R_ea <= 0 or self.C_i <= 0 or self.C_e <= 0:
            raise ValueError("R_ie, R_ea, C_i, C_e must all be positive")
        if not 0.0 <= self.wall_solar_fraction <= 1.0:
            raise ValueError(
                f"wall_solar_fraction={self.wall_solar_fraction} must be in [0,1]"
            )
        if self.sigma_w_i < 0 or self.sigma_w_e < 0 or self.sigma_v <= 0:
            raise ValueError("Noise diffusions ≥0 and sigma_v >0")


# ── Discrete-time state-space ────────────────────────────────────────────


@dataclass(frozen=True)
class StateSpace:
    """Discrete-time linear state-space.

        x_{k+1} = A x_k + B u_k + w_k,  cov(w_k) = Q
        y_k     = H x_k + v_k,           cov(v_k) = R
    """

    A: np.ndarray  # n × n
    B: np.ndarray  # n × m
    H: np.ndarray  # 1 × n (single measurement)
    Q: np.ndarray  # n × n process-noise covariance
    R: float  # measurement noise variance


# ── Van Loan discretization ──────────────────────────────────────────────


def _discretize_state_and_q(
    A_c: np.ndarray,
    Sigma_c: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Van Loan 1978 Eq. 18-21 — exact A_d and Q_d in one matrix exp.

    Builds the augmented matrix M = [[-A_c, Sigma_c], [0, A_c^T]] of size 2n×2n,
    exponentiates to M_d = expm(M * dt), and reads off:
        A_d = M_d[n:, n:]^T
        Q_d = A_d @ M_d[:n, n:]
    """
    n = A_c.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A_c
    M[:n, n : 2 * n] = Sigma_c
    M[n : 2 * n, n : 2 * n] = A_c.T
    M_d = expm(M * dt)
    A_d_T = M_d[n : 2 * n, n : 2 * n]
    A_d = A_d_T.T
    Q_d = A_d @ M_d[:n, n : 2 * n]
    # Symmetrize Q_d to absorb floating-point asymmetry
    Q_d = 0.5 * (Q_d + Q_d.T)
    return A_d, Q_d


def _discretize_input_zoh(
    A_c: np.ndarray,
    B_c: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Discretize input matrix for zero-order-hold via augmented expm.

    For x_dot = A_c x + B_c u with u held constant over [k, k+1)*dt:
        B_d = ∫_0^dt expm(A_c · s) ds @ B_c
    Equivalent to top-right block of expm([[A_c, B_c], [0, 0]] * dt).
    """
    n, m = B_c.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A_c
    M[:n, n:] = B_c
    M_d = expm(M * dt)
    B_d = M_d[:n, n:]
    return B_d


def discretize(
    A_c: np.ndarray,
    B_c: np.ndarray,
    Sigma_c: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Van Loan discretization of continuous-time linear SDE.

    Returns (A_d, B_d, Q_d) for use in StateSpace.
    """
    A_d, Q_d = _discretize_state_and_q(A_c, Sigma_c, dt)
    B_d = _discretize_input_zoh(A_c, B_c, dt)
    return A_d, B_d, Q_d


# ── State-space builders ─────────────────────────────────────────────────


N_INPUTS: Final = 3  # T_outdoor, q_heat_proxy_w, shortwave_w_m2


def build_1r1c(params: RCParams1R1C, dt: float) -> StateSpace:
    """Build discrete StateSpace for 1R1C model at sampling interval dt seconds."""
    params.validate()
    R, C = params.R, params.C
    A_c = np.array([[-1.0 / (R * C)]])
    # Inputs: [T_outdoor, q_heat_proxy, shortwave]
    B_c = np.array(
        [
            [
                1.0 / (R * C),
                params.q_scale / C,
                params.solar_scale / C,
            ]
        ]
    )
    Sigma_c = np.array([[params.sigma_w**2]])
    A_d, B_d, Q_d = discretize(A_c, B_c, Sigma_c, dt)
    H = np.array([[1.0]])
    return StateSpace(A=A_d, B=B_d, H=H, Q=Q_d, R=params.sigma_v**2)


def build_2r2c(params: RCParams2R2C, dt: float) -> StateSpace:
    """Build discrete StateSpace for 2R2C TiTe model at sampling interval dt seconds.

    State [T_i, T_e]; q_heat → air node only; solar splits per
    wall_solar_fraction.
    """
    params.validate()
    R_ie, R_ea = params.R_ie, params.R_ea
    C_i, C_e = params.C_i, params.C_e
    A_c = np.array(
        [
            [-1.0 / (R_ie * C_i), 1.0 / (R_ie * C_i)],
            [1.0 / (R_ie * C_e), -1.0 / (R_ie * C_e) - 1.0 / (R_ea * C_e)],
        ]
    )
    air_solar = (1.0 - params.wall_solar_fraction) * params.solar_scale
    wall_solar = params.wall_solar_fraction * params.solar_scale
    B_c = np.array(
        [
            [0.0, params.q_scale / C_i, air_solar / C_i],
            [1.0 / (R_ea * C_e), 0.0, wall_solar / C_e],
        ]
    )
    Sigma_c = np.array(
        [
            [params.sigma_w_i**2, 0.0],
            [0.0, params.sigma_w_e**2],
        ]
    )
    A_d, B_d, Q_d = discretize(A_c, B_c, Sigma_c, dt)
    H = np.array([[1.0, 0.0]])
    return StateSpace(A=A_d, B=B_d, H=H, Q=Q_d, R=params.sigma_v**2)


# ── Kalman log-likelihood ────────────────────────────────────────────────


def kalman_log_likelihood(
    ss: StateSpace,
    observations: np.ndarray,
    inputs: np.ndarray,
    valid: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> float:
    """One-step-ahead Kalman log-likelihood.

    Args:
        ss: discrete-time state-space.
        observations: shape (T,), measurement series.
        inputs: shape (T, m), input series. inputs[k] is held over [k, k+1).
        valid: shape (T,) boolean. Update step is skipped where valid is False.
        x0: optional initial state mean (n,). Defaults to [observations[first_valid], 0...].
        P0: optional initial state covariance (n,n). Defaults to 100·I (wide).

    Returns:
        Sum of conditional log-likelihoods over valid rows. Returns -inf if
        the innovation covariance becomes non-positive (numerical failure).

    Joseph-form covariance update is used for numerical stability per the
    bench's existing P-matrix-collapse precedent (debug_bundle_20260422).
    """
    obs = np.asarray(observations, dtype=float)
    u = np.asarray(inputs, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    valid = np.asarray(valid, dtype=bool)
    n = ss.A.shape[0]
    T = obs.shape[0]

    if u.shape[0] != T or valid.shape[0] != T:
        raise ValueError(
            f"Shape mismatch: obs={obs.shape}, u={u.shape}, valid={valid.shape}"
        )

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return 0.0

    if x0 is None:
        x0_arr = np.zeros(n)
        x0_arr[0] = obs[valid_idx[0]]
    else:
        x0_arr = np.asarray(x0, dtype=float).reshape(n)
    if P0 is None:
        P0_arr = np.eye(n) * 100.0
    else:
        P0_arr = np.asarray(P0, dtype=float)

    x = x0_arr.reshape(n, 1).copy()
    P = P0_arr.copy()

    log_lik = 0.0
    start = int(valid_idx[0])
    I_n = np.eye(n)

    for k in range(start, T):
        # Predict from k-1 → k via input held at u[k-1]
        if k > start:
            u_prev = u[k - 1].reshape(-1, 1)
            x = ss.A @ x + ss.B @ u_prev
            P = ss.A @ P @ ss.A.T + ss.Q

        if not valid[k]:
            continue

        # Innovation
        y_k = float(obs[k])
        Hx = (ss.H @ x).item()
        innov = y_k - Hx
        S = (ss.H @ P @ ss.H.T).item() + ss.R
        if S <= 0.0 or not np.isfinite(S):
            return -np.inf

        log_lik += -0.5 * (np.log(2.0 * np.pi * S) + innov * innov / S)

        # Kalman gain + Joseph-form update
        K = (P @ ss.H.T) / S  # (n,1)
        x = x + K * innov
        I_KH = I_n - K @ ss.H
        P = I_KH @ P @ I_KH.T + (K * ss.R) @ K.T

    return log_lik


# ── Innovation series (helper for residual diagnostics) ──────────────────


def kalman_innovations(
    ss: StateSpace,
    observations: np.ndarray,
    inputs: np.ndarray,
    valid: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-step-ahead innovation series and innovation variances.

    Useful for residual diagnostics (Bacher-Madsen battery operates on
    standardized innovations e_k / sqrt(S_k)). Both arrays are NaN on
    invalid rows so downstream tooling can mask cleanly.
    """
    obs = np.asarray(observations, dtype=float)
    u = np.asarray(inputs, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    valid = np.asarray(valid, dtype=bool)
    n = ss.A.shape[0]
    T = obs.shape[0]

    valid_idx = np.where(valid)[0]
    innovations = np.full(T, np.nan)
    variances = np.full(T, np.nan)
    if len(valid_idx) == 0:
        return innovations, variances

    if x0 is None:
        x0_arr = np.zeros(n)
        x0_arr[0] = obs[valid_idx[0]]
    else:
        x0_arr = np.asarray(x0, dtype=float).reshape(n)
    P0_arr = np.eye(n) * 100.0 if P0 is None else np.asarray(P0, dtype=float)

    x = x0_arr.reshape(n, 1).copy()
    P = P0_arr.copy()

    start = int(valid_idx[0])
    I_n = np.eye(n)

    for k in range(start, T):
        if k > start:
            u_prev = u[k - 1].reshape(-1, 1)
            x = ss.A @ x + ss.B @ u_prev
            P = ss.A @ P @ ss.A.T + ss.Q

        if not valid[k]:
            continue

        y_k = float(obs[k])
        Hx = (ss.H @ x).item()
        innov = y_k - Hx
        S = (ss.H @ P @ ss.H.T).item() + ss.R

        innovations[k] = innov
        variances[k] = S

        if S <= 0.0 or not np.isfinite(S):
            continue

        K = (P @ ss.H.T) / S
        x = x + K * innov
        I_KH = I_n - K @ ss.H
        P = I_KH @ P @ I_KH.T + (K * ss.R) @ K.T

    return innovations, variances
