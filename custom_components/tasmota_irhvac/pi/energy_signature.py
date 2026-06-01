"""Energy-signature (daily-integral) β estimator.

Recovers steady-state feed-forward gains by integrating the energy balance over
closed multi-day cycles, where the storage term telescopes and the input lag
washes out.  It therefore uses ALL operational data (no equilibrium filter, no
lag/τ estimation) and is robust to the diurnal aliasing that defeats
instantaneous lag methods (single-EMA-τ, FIR) for periodic solar.

Derivation (1R1C; generalizes to N-state — the hidden mass node drops out via
its own cycle-closure energy balance):

    C·dT/dt = k(sp − T) − UA(T − T_out) + A·S
  steady state ⇒ sp − T = (UA/k)(T − T_out) − (A/k)S,  so β_solar = −A/k.
  Integrate over a closed window [t0, t1] with T(t1) ≈ T(t0):

    Σ(sp − T) = (UA/k)·Σ(T − T_out) − (A/k)·Σ S + (C/k)·ΔT

  Regress the window-integrated demand Σ(sp − T) on the window-integrated
  loss driver Σ(T_out − T), the MEASURED window solar integral Σ S, and the
  window net ΔT (carries any residual non-closure).  The solar coefficient is
  exactly β_solar; the (T_out − T) coefficient is β_outdoor = −UA/k.

Why it works where the equilibrium WLS can't:
  - The WLS target sp − T is the energy balance with C·dT/dt dropped — valid
    only at dT/dt ≈ 0, so it must discard the solar-forced (transient) obs that
    carry the signal.  The integral puts the storage term back (as the ΔT term)
    and telescopes it over a closed cycle, so transient obs are admitted.
  - Σ_t (h * S)_t ≈ (Σ_τ h_τ)·Σ_t S_t over a full window — the daily total
    depends only on the total gain Σ_τ h_τ = A, NOT the lag distribution.  So τ
    is never estimated and diurnal aliasing cannot bite.

Window length must span a few × the building's slow time constant so the
wall/mass node closes within it; otherwise residual wall storage attenuates
β_solar (verified monotonic in thermal mass on the 2R2C bench).

Scope: only FULLY-CONTROLLED windows (HP delivering k·(sp − T) every tick) are
used.  HP-off, solar-saturated windows are a censored-data (Tobit) regime where
the latent demand is negative but observed as 0; naive inclusion inverts the
loss coefficient, so those are left to the dynamic grey-box (which handles
HP-off natively via hp_offset = 0).  When too few controlled windows exist the
estimator DECLINES (ok=False) rather than return an unidentified coefficient.

References:
- Fels, "PRISM: An Introduction" (1986) — periodic energy-signature regression.
- Hammarsten, "A critical appraisal of energy-signature models" (1987).
- Kissock et al., ASHRAE Inverse Modeling Toolkit — change-point/degree-day
  inverse models from metered operational data.

Standalone module — no Home Assistant dependencies.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .batch_learning import Observation

_LOGGER = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86400.0
# Default pre-dawn cycle boundary (local hour): solar = 0 and the room is
# settled at overnight heating, so windows close with the smallest ΔT.
_DEFAULT_BOUNDARY_HOUR = 4.0
# Coefficient layout of the daily-integral design matrix.
_C_INTERCEPT, _C_OUTDOOR, _C_SOLAR, _C_STORAGE = 0, 1, 2, 3


@dataclass
class _WindowAgg:
    """Running integrals for one closed-cycle window."""

    n: int = 0
    n_controlled: int = 0
    sum_demand: float = 0.0      # Σ over controlled ticks of (sp − T)
    sum_loss: float = 0.0        # Σ over all ticks of (T_out − T)
    sum_solar: float = 0.0       # Σ over all ticks of measured solar
    t_first: float = 0.0         # room temp at window start
    t_last: float = 0.0          # room temp at window end
    wall_first: float = math.inf
    wall_last: float = -math.inf


@dataclass
class EnergySignatureResult:
    """Result of an energy-signature daily-integral fit.

    ``ok`` is False (with ``reject_reason`` set) when the estimator declines —
    too few fully-controlled windows to identify — in which case the β fields
    are NaN and callers should hold their prior estimate rather than apply one.
    """

    ok: bool
    reject_reason: str = ""
    beta_solar: float = float("nan")
    beta_outdoor: float = float("nan")
    b_storage: float = float("nan")
    intercept: float = float("nan")
    se_solar: float = float("nan")
    se_outdoor: float = float("nan")
    # Standard errors on the storage and intercept coefficients.  Both are
    # computed from the regression covariance matrix in the same call that
    # produces se_solar / se_outdoor.  Exposed so downstream consumers can
    # gate use of the derived τ = −b_storage·dt/β_outdoor on confidence:
    # b_storage is only well-identified when window-to-window ΔT carries
    # meaningful variance.  Non-zero intercept indicates regression bias
    # (closed-cycle physics predicts intercept ≈ 0).
    se_b_storage: float = float("nan")
    se_intercept: float = float("nan")
    n_windows: int = 0          # fully-controlled windows used
    n_windows_total: int = 0    # full-coverage windows seen (controlled or not)
    window_days: int = 0
    residual_rms: float = float("nan")
    # Diagnostics: daily solar↔outdoor collinearity and the solar excitation span.
    solar_collinearity: float = float("nan")
    solar_excitation: tuple[float, float] = (float("nan"), float("nan"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reject_reason": self.reject_reason,
            "beta_solar": self.beta_solar,
            "beta_outdoor": self.beta_outdoor,
            "b_storage": self.b_storage,
            "intercept": self.intercept,
            "se_solar": self.se_solar,
            "se_outdoor": self.se_outdoor,
            "se_b_storage": self.se_b_storage,
            "se_intercept": self.se_intercept,
            "n_windows": self.n_windows,
            "n_windows_total": self.n_windows_total,
            "window_days": self.window_days,
            "residual_rms": self.residual_rms,
            "solar_collinearity": self.solar_collinearity,
            "solar_excitation": list(self.solar_excitation),
        }


def _infer_dt_seconds(wall_times: list[float]) -> float | None:
    """Median consecutive spacing of sorted observation wall-times (seconds)."""
    if len(wall_times) < 2:
        return None
    ws = sorted(wall_times)
    diffs = [b - a for a, b in zip(ws, ws[1:]) if b > a]
    if not diffs:
        return None
    return float(np.median(diffs))


def estimate_energy_signature(
    observations: list["Observation"],
    *,
    solar_entity: str | None,
    window_days: int,
    boundary_hour: float = _DEFAULT_BOUNDARY_HOUR,
    min_window_coverage: float = 0.9,
    min_windows: int = 4,
) -> EnergySignatureResult:
    """Fit β_solar / β_outdoor from a closed-cycle daily-integral regression.

    Args:
        observations: raw observations (e.g. the grey-box buffer contents, or a
            daily-aggregate accumulator's source ticks).  Must carry hp_setpoint,
            current_c, outdoor_temp_c, wall_time, clamped, and the solar reading
            in raw_readings[solar_entity].
        solar_entity: entity id of the solar proxy in raw_readings.
        window_days: closed-cycle window length; choose ≈ 2× the building slow
            time constant so the mass node closes within the window.
        boundary_hour: local hour at which windows flip (pre-dawn default).
        min_window_coverage: fraction of expected ticks a window must contain to
            count as full (guards run-edge partial windows).
        min_windows: minimum fully-controlled windows required to identify.
    """
    if window_days < 1:
        return EnergySignatureResult(ok=False, reject_reason="bad_window_days",
                                     window_days=window_days)
    if solar_entity is None:
        return EnergySignatureResult(ok=False, reject_reason="no_solar_entity",
                                     window_days=window_days)
    usable = [o for o in observations
              if o.outdoor_temp_c is not None and o.wall_time is not None]
    dt = _infer_dt_seconds([o.wall_time for o in usable])
    if dt is None or dt <= 0:
        return EnergySignatureResult(ok=False, reject_reason="insufficient_obs",
                                     window_days=window_days)

    win_seconds = window_days * _SECONDS_PER_DAY
    expected_ticks = win_seconds / dt
    boundary_s = boundary_hour * 3600.0
    # Anchor window 0 to the pre-dawn boundary of the FIRST observation's day,
    # so the windowing is reproducible and independent of the absolute epoch
    # (bucketing by absolute time would pair different days together depending
    # on where the record starts relative to the epoch).
    first_wall = min(o.wall_time for o in usable)
    origin = boundary_s + math.floor(
        (first_wall - boundary_s) / _SECONDS_PER_DAY) * _SECONDS_PER_DAY

    aggs: dict[int, _WindowAgg] = {}
    for o in usable:
        widx = math.floor((o.wall_time - origin) / win_seconds)
        a = aggs.get(widx)
        if a is None:
            a = aggs[widx] = _WindowAgg()
        room = o.current_c
        a.n += 1
        a.sum_loss += (o.outdoor_temp_c - room)
        a.sum_solar += o.raw_readings.get(solar_entity, 0.0)
        controlled = (not o.clamped) and (o.hp_setpoint is not None)
        if controlled:
            a.n_controlled += 1
            a.sum_demand += (o.hp_setpoint - room)
        if o.wall_time < a.wall_first:
            a.wall_first, a.t_first = o.wall_time, room
        if o.wall_time > a.wall_last:
            a.wall_last, a.t_last = o.wall_time, room

    rows_y, rows_out, rows_sol, rows_dt = [], [], [], []
    n_total = 0
    for a in aggs.values():
        if a.n < min_window_coverage * expected_ticks:
            continue                       # run-edge partial window
        n_total += 1
        if a.n_controlled < a.n:           # not fully controlled → censored regime
            continue
        rows_y.append(a.sum_demand)
        rows_out.append(a.sum_loss)
        rows_sol.append(a.sum_solar)
        rows_dt.append(a.t_last - a.t_first)

    n_used = len(rows_y)
    if n_used < min_windows:
        return EnergySignatureResult(
            ok=False, reject_reason="insufficient_windows",
            n_windows=n_used, n_windows_total=n_total, window_days=window_days)

    y = np.asarray(rows_y, float)
    xs = np.asarray(rows_sol, float)
    xo = np.asarray(rows_out, float)
    X = np.column_stack([np.ones(n_used), xo, xs, np.asarray(rows_dt, float)])
    # Solar excitation must vary across windows to identify its coefficient.
    sol_lo, sol_hi = float(xs.min()), float(xs.max())
    if sol_hi - sol_lo <= 0.0:
        return EnergySignatureResult(
            ok=False, reject_reason="no_solar_excitation",
            n_windows=n_used, n_windows_total=n_total, window_days=window_days)

    coef, _res, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:
        return EnergySignatureResult(
            ok=False, reject_reason="rank_deficient",
            n_windows=n_used, n_windows_total=n_total, window_days=window_days)

    resid = y - X @ coef
    rss = float(resid @ resid)
    dof = max(1, n_used - X.shape[1])
    cov = (rss / dof) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    corr = float(np.corrcoef(xs, xo)[0, 1])

    return EnergySignatureResult(
        ok=True,
        beta_solar=float(coef[_C_SOLAR]),
        beta_outdoor=float(coef[_C_OUTDOOR]),
        b_storage=float(coef[_C_STORAGE]),
        intercept=float(coef[_C_INTERCEPT]),
        se_solar=float(se[_C_SOLAR]),
        se_outdoor=float(se[_C_OUTDOOR]),
        se_b_storage=float(se[_C_STORAGE]),
        se_intercept=float(se[_C_INTERCEPT]),
        n_windows=n_used,
        n_windows_total=n_total,
        window_days=window_days,
        residual_rms=math.sqrt(rss / n_used),
        solar_collinearity=corr,
        solar_excitation=(sol_lo, sol_hi),
    )
