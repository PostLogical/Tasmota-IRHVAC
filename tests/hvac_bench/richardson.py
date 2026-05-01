"""Richardson extrapolation for solution verification (Phase 3a).

Estimates the discretization-error band on a KPI by running the same
scenario at multiple tick rates and extrapolating to h → 0. Pattern
follows Oberkampf-Roy V&V §8 (solution verification): every reported
KPI carries a numerical-error band alongside its stochastic CI.

Approach
--------
Given KPI values at tick rates ``h_1 > h_2 > ... > h_n`` (coarsest
first):

.. math::

    \\text{KPI}(h) \\approx \\text{KPI}_\\infty + C \\cdot h^p

where ``p`` is the observed convergence order. The Richardson estimate
of :math:`\\text{KPI}_\\infty` is

.. math::

    \\text{KPI}_\\infty = \\frac{r^p \\cdot \\text{KPI}(h_\\text{fine})
                                  - \\text{KPI}(h_\\text{coarse})}{r^p - 1}

with ``r = h_coarse / h_fine``. The discretization error band on the
finest grid is :math:`|\\text{KPI}(h_\\text{fine}) - \\text{KPI}_\\infty|`.

Three branches:

* **2 grids** — ``p`` must be assumed (default 1.0, first-order). The
  user can pass ``assumed_order`` when prior knowledge applies (e.g.
  RK2 → p=2).
* **≥ 3 grids, uniform refinement** — closed-form Roache (1994)::

      p = log(|(KPI_coarse - KPI_mid) / (KPI_mid - KPI_fine)|) / log(r)

* **≥ 3 grids, non-uniform refinement** — solve the analogous nonlinear
  equation via Brent's method on ``[0.1, 10.0]``.

When the convergence sequence is non-monotone (sign-changing successive
deltas) the order fit is suppressed and the report falls back to
``p=1`` on the finest pair, with ``observed_order=NaN`` flagging the
caller that the KPI is not in the asymptotic Richardson regime.

References
----------
- Richardson 1911; Richardson & Gaunt 1927 (original deferred approach
  to the limit).
- Roache, *Verification and Validation in Computational Science and
  Engineering*, 1998 — generalized Richardson formulation.
- Oberkampf & Roy, *Verification and Validation in Scientific
  Computing*, 2010, §8 (solution verification, GCI).
- Forrester et al. 2008 — practical multi-grid extrapolation guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import optimize

from tests.hvac_bench.kpis import KpiBundle


# ── Default KPI list ──────────────────────────────────────────────────────

# Numeric control-side KPIs that are well-defined for any controller and
# meaningful to extrapolate. ``settling_time_h`` is excluded because it
# can be ``None`` (never-settled scenarios) and discretization-aware
# settling needs a separate definition. Learning KPIs aren't extrapolated
# here — they get their own Phase 4 treatment.
DEFAULT_RICHARDSON_KPIS: tuple[str, ...] = (
    "tdis_tot",
    "ener_tot",
    "peak_kw",
    "cold_time_h",
    "warm_time_h",
    "setpoint_changes",
)


# ── Result type ───────────────────────────────────────────────────────────


@dataclass
class RichardsonReport:
    """Solution-verification report for a single KPI across a tick sweep.

    Attributes
    ----------
    kpi_name
        Name of the KPI being extrapolated.
    tick_rates_minutes
        Tick rates used, sorted descending (coarsest first).
    kpi_values
        Observed KPI value at each tick rate (paired with
        ``tick_rates_minutes``).
    extrapolated_value
        Best estimate of ``KPI(h → 0)``.
    observed_order
        Fitted convergence order ``p`` in ``KPI(h) = KPI∞ + C·h^p``.
        ``NaN`` if the convergence sequence is not in the asymptotic
        regime (non-monotone deltas, identical values, or fewer points
        than required to fit ``p``).
    error_band
        ``|KPI(h_finest) - extrapolated_value|``. Conservative bound on
        the discretization error of the finest-grid KPI value (Roache
        GCI without the 3.0 safety factor — caller can multiply if
        wanted).
    monotone_convergence
        True iff KPI values change in a single direction as ``h → 0``
        (sign-stable successive deltas).
    relative_error
        ``error_band / |extrapolated_value|``; ``inf`` when
        ``extrapolated_value`` is zero and ``error_band`` is positive,
        ``0`` when both are zero.
    fit_method
        How ``p`` was determined: ``"two_point_assumed_order"``,
        ``"three_point_observed_order"``, ``"nonuniform_observed_order"``,
        or ``"fallback_assumed_order_1"`` (set when ≥ 3 grids were
        provided but the deltas were non-monotone or stalled).
    """

    kpi_name: str
    tick_rates_minutes: list[float]
    kpi_values: list[float]
    extrapolated_value: float
    observed_order: float
    error_band: float
    monotone_convergence: bool
    relative_error: float
    fit_method: str


# ── Single-KPI extrapolation ──────────────────────────────────────────────


def richardson_extrapolate(
    tick_rates_minutes: Sequence[float],
    kpi_values: Sequence[float],
    *,
    assumed_order: float | None = None,
    kpi_name: str = "kpi",
) -> RichardsonReport:
    """Extrapolate a KPI to ``h → 0`` using Richardson's method.

    Args:
        tick_rates_minutes: tick rates in minutes per tick. Any order
            accepted; sorted descending internally so the smallest h
            (finest grid) is treated as the most accurate.
        kpi_values: observed KPI at each tick rate (paired).
        assumed_order: convergence order to force. Default ``None`` →
            fit from data when ≥ 3 grids; fall back to 1.0 with 2 grids.
            Pass an explicit value when the integration scheme is known
            (e.g. 1.0 for explicit Euler, 2.0 for RK2).
        kpi_name: name carried into the report for tabular display.

    Returns:
        ``RichardsonReport``.

    Raises:
        ValueError: shape mismatch, fewer than 2 grids, non-finite or
            non-positive tick rates, non-finite KPI values.
    """
    h_arr = np.asarray(tick_rates_minutes, dtype=float)
    k_arr = np.asarray(kpi_values, dtype=float)
    if h_arr.shape != k_arr.shape:
        raise ValueError(
            f"tick_rates_minutes and kpi_values must align: "
            f"{h_arr.shape} vs {k_arr.shape}"
        )
    if h_arr.ndim != 1 or h_arr.size < 2:
        raise ValueError(
            f"need >= 2 grids for Richardson; got {h_arr.size}"
        )
    if not np.all(np.isfinite(h_arr)) or not np.all(h_arr > 0):
        raise ValueError("tick_rates_minutes must be finite and positive")
    if not np.all(np.isfinite(k_arr)):
        raise ValueError("kpi_values must be finite (no NaN/inf)")
    if len(set(h_arr.tolist())) != h_arr.size:
        raise ValueError(
            f"tick_rates_minutes must be distinct: got {h_arr.tolist()}"
        )

    # Sort by h descending (coarsest first → finest last)
    order = np.argsort(-h_arr)
    h = h_arr[order]
    k = k_arr[order]

    # Monotone-convergence flag from successive deltas (coarse → fine)
    deltas = np.diff(k)
    if np.all(deltas == 0):
        monotone = True
    elif np.any(deltas == 0):
        monotone = False
    else:
        signs = np.sign(deltas)
        monotone = bool(np.all(signs == signs[0]))

    n = h.size

    if n == 2 or assumed_order is not None:
        return _extrapolate_two_point(h, k, assumed_order, kpi_name, monotone)
    return _extrapolate_three_or_more(h, k, kpi_name, monotone)


def _extrapolate_two_point(
    h: np.ndarray,
    k: np.ndarray,
    assumed_order: float | None,
    kpi_name: str,
    monotone: bool,
) -> RichardsonReport:
    """Extrapolate with assumed order on the finest pair."""
    p = float(assumed_order) if assumed_order is not None else 1.0
    h_fine = float(h[-1])
    h_coarse = float(h[-2])
    r = h_coarse / h_fine
    rp = r ** p
    if rp == 1.0:
        # Same-grid pair (or p=0); no extrapolation possible
        extrap = float(k[-1])
        err_band = 0.0
    else:
        extrap = (rp * k[-1] - k[-2]) / (rp - 1.0)
        err_band = abs(k[-1] - extrap)

    rel_err = _relative_error(err_band, extrap)
    return RichardsonReport(
        kpi_name=kpi_name,
        tick_rates_minutes=h.tolist(),
        kpi_values=k.tolist(),
        extrapolated_value=float(extrap),
        observed_order=p,
        error_band=float(err_band),
        monotone_convergence=monotone,
        relative_error=rel_err,
        fit_method="two_point_assumed_order",
    )


def _extrapolate_three_or_more(
    h: np.ndarray,
    k: np.ndarray,
    kpi_name: str,
    monotone: bool,
) -> RichardsonReport:
    """Fit p from the finest three points, then extrapolate finest pair."""
    h1, h2, h3 = float(h[-3]), float(h[-2]), float(h[-1])
    k1, k2, k3 = float(k[-3]), float(k[-2]), float(k[-1])
    d_coarse = k2 - k1
    d_fine = k3 - k2

    # Check the three-point convergence regime is well-posed.  If both
    # deltas are zero, KPI is insensitive to h on this range; if signs
    # differ or one is zero, the sequence is not in the asymptotic
    # regime and fitting p is meaningless.
    if d_coarse == 0.0 and d_fine == 0.0:
        return RichardsonReport(
            kpi_name=kpi_name,
            tick_rates_minutes=h.tolist(),
            kpi_values=k.tolist(),
            extrapolated_value=float(k3),
            observed_order=float("nan"),
            error_band=0.0,
            monotone_convergence=monotone,
            relative_error=0.0,
            fit_method="degenerate_constant_kpi",
        )
    if d_fine == 0.0 or d_coarse == 0.0 or np.sign(d_coarse) != np.sign(d_fine):
        p = 1.0
        r = h2 / h3
        rp = r ** p
        extrap = (rp * k3 - k2) / (rp - 1.0) if rp != 1.0 else k3
        err_band = abs(k3 - extrap)
        rel_err = _relative_error(err_band, extrap)
        return RichardsonReport(
            kpi_name=kpi_name,
            tick_rates_minutes=h.tolist(),
            kpi_values=k.tolist(),
            extrapolated_value=float(extrap),
            observed_order=float("nan"),
            error_band=float(err_band),
            monotone_convergence=monotone,
            relative_error=rel_err,
            fit_method="fallback_assumed_order_1",
        )

    ratio = d_coarse / d_fine
    r1 = h1 / h2
    r2 = h2 / h3
    if abs(r1 - r2) < 1e-9:
        # Uniform refinement: closed-form Roache p = log|ratio| / log r
        r = r1
        p = math.log(abs(ratio)) / math.log(r)
        method = "three_point_observed_order"
    else:
        # Non-uniform: solve f(p) = (h1^p - h2^p)/(h2^p - h3^p) - ratio = 0
        def residual(p_val: float) -> float:
            return (h1 ** p_val - h2 ** p_val) / (h2 ** p_val - h3 ** p_val) - ratio

        try:
            p = float(optimize.brentq(residual, 0.1, 10.0, xtol=1e-6))
            method = "nonuniform_observed_order"
        except (ValueError, RuntimeError):
            p = 1.0
            method = "fallback_assumed_order_1"

    r = h2 / h3
    rp = r ** p
    if rp == 1.0:
        extrap = float(k3)
        err_band = 0.0
    else:
        extrap = (rp * k3 - k2) / (rp - 1.0)
        err_band = abs(k3 - extrap)

    rel_err = _relative_error(err_band, extrap)
    return RichardsonReport(
        kpi_name=kpi_name,
        tick_rates_minutes=h.tolist(),
        kpi_values=k.tolist(),
        extrapolated_value=float(extrap),
        observed_order=float(p),
        error_band=float(err_band),
        monotone_convergence=monotone,
        relative_error=rel_err,
        fit_method=method,
    )


def _relative_error(err_band: float, extrap: float) -> float:
    if extrap == 0.0:
        return float("inf") if err_band > 0.0 else 0.0
    return err_band / abs(extrap)


# ── Bundle-level sweep ────────────────────────────────────────────────────


def kpi_richardson_sweep(
    kpi_bundles: dict[float, KpiBundle],
    *,
    kpi_names: Sequence[str] | None = None,
    assumed_order: float | None = None,
) -> dict[str, RichardsonReport]:
    """Run Richardson on each KPI across a tick-rate sweep.

    Args:
        kpi_bundles: ``{tick_minutes: KpiBundle}`` from running the same
            scenario at multiple tick rates. At least two distinct rates
            required.
        kpi_names: which KPI fields to extrapolate. Defaults to
            ``DEFAULT_RICHARDSON_KPIS`` (numeric control-side metrics).
            Names must match ``KpiBundle`` numeric field names.
        assumed_order: forwarded to ``richardson_extrapolate``.

    Returns:
        ``{kpi_name: RichardsonReport}`` with one report per requested KPI.
    """
    if len(kpi_bundles) < 2:
        raise ValueError(
            f"need >= 2 tick rates for Richardson; got {len(kpi_bundles)}"
        )
    names = tuple(kpi_names) if kpi_names is not None else DEFAULT_RICHARDSON_KPIS
    tick_rates = sorted(kpi_bundles.keys(), reverse=True)
    out: dict[str, RichardsonReport] = {}
    for kpi_name in names:
        values: list[float] = []
        for h in tick_rates:
            bundle = kpi_bundles[h]
            v = getattr(bundle, kpi_name, None)
            if v is None:
                raise ValueError(
                    f"KPI {kpi_name!r} missing or None on tick={h}; "
                    f"Richardson sweep requires all tick rates to populate it"
                )
            values.append(float(v))
        out[kpi_name] = richardson_extrapolate(
            tick_rates,
            values,
            assumed_order=assumed_order,
            kpi_name=kpi_name,
        )
    return out


# ── Reporting helpers ─────────────────────────────────────────────────────


def format_richardson_table(reports: dict[str, RichardsonReport]) -> str:
    """Render a Richardson report dict as a fixed-width table.

    Used by Phase 3 regression tests' ``-s`` print mode and by
    diagnostic scripts. Not meant for parsing.
    """
    lines: list[str] = []
    header = (
        f"{'KPI':<22} {'finest':>10} {'extrap':>12} {'err_band':>10} "
        f"{'rel_err':>9} {'p':>6} {'method':>32}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, rep in reports.items():
        finest_v = rep.kpi_values[-1]
        p_str = "  nan" if math.isnan(rep.observed_order) else f"{rep.observed_order:>5.2f}"
        rel_str = "   inf" if math.isinf(rep.relative_error) else f"{rep.relative_error:>8.3%}"
        lines.append(
            f"{name:<22} {finest_v:>10.4f} {rep.extrapolated_value:>12.4f} "
            f"{rep.error_band:>10.4f} {rel_str:>9} {p_str:>6} {rep.fit_method:>32}"
        )
    return "\n".join(lines)
