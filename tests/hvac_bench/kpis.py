"""Canonical KPI bundle for the bench reference-controller suite.

Phase 2c of the bench-validation plan. Provides a fixed list of KPIs that
every (controller × scenario) combination reports, so that bench changes
can be compared against locked numbers (Phase 2b) and discriminative
power between controllers becomes visible.

Three groups:

* **Control-side KPIs** (always computable from the per-tick history):
  ``tdis_tot``, ``ener_tot``, ``peak_kw``, ``settling_time_h``,
  ``cold_time_h``, ``warm_time_h``.  Pattern follows BOPTEST's locked KPI
  list (``tdis_tot`` = total thermal discomfort, ``ener_tot`` = total
  electric energy, ``pele`` = peak electrical demand) plus deadband
  splits useful for asymmetric heat / cool failure modes.

* **Learning-side KPIs** (Optional, populated by controllers that learn):
  ``beta_bias`` (per-feature, vs ground truth), ``residual_whiteness_pass``,
  ``residual_normality_pass``, ``n_observations``.

* **Multi-realisation KPIs** (computed by the locked-score harness across
  Monte-Carlo realisations of the same scenario, not from a single run):
  ``crlb_coverage`` — fraction of MC realisations where truth lands inside
  the per-realisation 2σ CRLB band on β.

Locked-score regression tests (Phase 2b) compare the reported bundle
against checked-in numbers with documented tolerances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np


DEADBAND_C: float = 0.5


# ── Bundle ────────────────────────────────────────────────────────────────


@dataclass
class KpiBundle:
    """Canonical KPI bundle reported per (controller × scenario) run.

    Numeric fields default to 0.0 / 0 so that controllers that do not
    populate a group (e.g. NaiveBangBang has no learning KPIs) can still
    return a complete bundle.  Learning fields default to ``None`` to
    distinguish "not applicable for this controller" from "computed and
    happens to be zero".
    """

    # ── Control-side ─────────────────────────────────────────────────
    n_ticks: int
    tick_minutes: float
    tdis_tot: float          # K·h outside deadband, USER frame (user_desired - room)
    cold_time_h: float       # hours room < user_desired - deadband
    warm_time_h: float       # hours room > user_desired + deadband
    # Effective (controller) frame: same KPIs vs the qref-biased reference the
    # PI actually tracked.  Equals the user-frame values when no supervisor is
    # active.  tdis_tot - tdis_tot_eff = intentional reference displacement.
    tdis_tot_eff: float      # K·h outside deadband, EFFECTIVE frame
    cold_time_h_eff: float   # hours room < effective_desired - deadband
    warm_time_h_eff: float   # hours room > effective_desired + deadband
    ener_tot: float          # kWh
    peak_kw: float           # peak instantaneous electrical demand (kW)
    settling_time_h: float | None  # hours to last exit of deadband; None if never settled
    setpoint_changes: int

    # ── Learning-side (None if controller does not learn) ────────────
    n_observations: int | None = None
    beta_bias: dict[str, float] | None = None    # beta_estimated - beta_truth, per feature
    beta_estimated: dict[str, float] | None = None
    beta_truth: dict[str, float] | None = None
    residual_whiteness_pass: bool | None = None  # Ljung-Box at α=0.05
    residual_normality_pass: bool | None = None  # Shapiro-Wilk / Jarque-Bera at α=0.05
    condition_number: float | None = None        # Belsley κ (standardised)

    # ── Multi-realisation (populated by aggregator, not single run) ──
    crlb_coverage: dict[str, float] | None = None  # per feature, fraction in 2σ band
    closed_loop_bias_magnitude: dict[str, float] | None = None  # |beta_cl - beta_ol|, per feature

    def as_dict(self) -> dict:
        """Flatten to a single dict for tabular reporting."""
        out = {
            "n_ticks": self.n_ticks,
            "tick_minutes": self.tick_minutes,
            "tdis_tot": self.tdis_tot,
            "cold_time_h": self.cold_time_h,
            "warm_time_h": self.warm_time_h,
            "tdis_tot_eff": self.tdis_tot_eff,
            "cold_time_h_eff": self.cold_time_h_eff,
            "warm_time_h_eff": self.warm_time_h_eff,
            "ener_tot": self.ener_tot,
            "peak_kw": self.peak_kw,
            "settling_time_h": self.settling_time_h,
            "setpoint_changes": self.setpoint_changes,
        }
        if self.n_observations is not None:
            out["n_observations"] = self.n_observations
        if self.beta_bias is not None:
            for k, v in self.beta_bias.items():
                out[f"beta_bias.{k}"] = v
        if self.residual_whiteness_pass is not None:
            out["residual_whiteness_pass"] = self.residual_whiteness_pass
        if self.residual_normality_pass is not None:
            out["residual_normality_pass"] = self.residual_normality_pass
        if self.condition_number is not None:
            out["condition_number"] = self.condition_number
        return out


# ── Computation ───────────────────────────────────────────────────────────


def compute_control_kpis(
    history: list[dict],
    *,
    tick_minutes: float,
    deadband_c: float = DEADBAND_C,
) -> KpiBundle:
    """Compute the control-side KPI group from a per-tick history.

    Expected per-tick keys: ``room_temp``, ``desired``, ``error``,
    ``hp_setpoint``, ``cumulative_kwh``.  Missing ``cumulative_kwh``
    yields zero energy KPIs (e.g. when a controller is run against
    a thermal model without a COP model attached).

    Args:
        history: per-tick dicts as produced by ``runner.run_scenario`` or
            ``run_full_stack``.
        tick_minutes: minutes per tick — needed for the energy peak rate
            (kW = kWh_per_tick / hours_per_tick).
        deadband_c: comfort deadband (single-sided, °C).  Defaults to 0.5,
            matching the production deadband and BOPTEST acceptable-band
            convention.

    Returns:
        ``KpiBundle`` with control-side fields populated; learning fields
        left as ``None``.
    """
    n_ticks = len(history)
    if n_ticks == 0:
        return KpiBundle(
            n_ticks=0,
            tick_minutes=tick_minutes,
            tdis_tot=0.0,
            cold_time_h=0.0,
            warm_time_h=0.0,
            tdis_tot_eff=0.0,
            cold_time_h_eff=0.0,
            warm_time_h_eff=0.0,
            ener_tot=0.0,
            peak_kw=0.0,
            settling_time_h=None,
            setpoint_changes=0,
        )

    dt_hours = tick_minutes / 60.0

    # Comfort accumulation.  tdis_tot is the BOPTEST-style integral of
    # discomfort: sum over ticks of max(0, |error| - deadband) * dt_hours.
    # cold/warm split this signed so a heat-mode failure mode (room cold)
    # vs cool-mode (room warm) is visible, matching the asymmetric KPI
    # split BOPTEST recommends for residential cases.
    # Dual-reference comfort.  We accumulate the BOPTEST-style discomfort
    # integral against TWO references:
    #   • user frame      (``error`` = user_desired − room): the occupant's
    #     true comfort cost — how far the room is from what was asked.
    #   • effective frame (``error_effective`` = effective_desired − room):
    #     controller tracking quality — how well it tracks the (qref-biased)
    #     target it was actually given.
    # ``tdis_tot − tdis_tot_eff`` isolates the supervisor's intentional
    # reference displacement from any genuine tracking failure.  Controllers
    # without a supervisor (no ``error_effective`` key) fall back to the user
    # error, so the two frames coincide.  See the desired-vs-effective-desired
    # branch design notes.
    def _accumulate_discomfort(err_key: str) -> tuple[float, float, float]:
        tdis = cold = warm = 0.0
        for h in history:
            if err_key in h:
                err = float(h[err_key])
            else:
                err = float(h.get("error", h.get("desired", 0.0) - h.get("room_temp", 0.0)))
            # error sign convention in this bench: desired - room.  Positive
            # error → room is cold (below desired); negative → warm.
            signed_excess_cold = max(0.0, err - deadband_c)
            signed_excess_warm = max(0.0, (-err) - deadband_c)
            tdis += (signed_excess_cold + signed_excess_warm) * dt_hours
            if signed_excess_cold > 0.0:
                cold += dt_hours
            if signed_excess_warm > 0.0:
                warm += dt_hours
        return tdis, cold, warm

    tdis_tot, cold_h, warm_h = _accumulate_discomfort("error")
    tdis_eff, cold_eff_h, warm_eff_h = _accumulate_discomfort("error_effective")

    # Energy: ener_tot is the cumulative_kwh at the last tick.  peak_kw
    # is the peak per-tick kWh divided by per-tick hours.  Both depend on
    # the thermal model wiring a COP model; if absent, both stay zero
    # rather than raising — Naive bang-bang on a no-COP model is fine
    # to score on comfort alone.
    ener_tot = float(history[-1].get("cumulative_kwh", 0.0))
    peak_kwh_per_tick = 0.0
    prev_kwh = 0.0
    for h in history:
        cur = float(h.get("cumulative_kwh", 0.0))
        delta = cur - prev_kwh
        if delta > peak_kwh_per_tick:
            peak_kwh_per_tick = delta
        prev_kwh = cur
    peak_kw = peak_kwh_per_tick / dt_hours if dt_hours > 0 else 0.0

    # Settling time: hours to the last exit of the deadband.  None if
    # the system never settles.
    last_outside = -1
    for i, h in enumerate(history):
        err = abs(float(h.get("error", h.get("desired", 0.0) - h.get("room_temp", 0.0))))
        if err >= deadband_c:
            last_outside = i
    if last_outside == -1:
        settling = 0.0
    elif last_outside == n_ticks - 1:
        settling = None
    else:
        settling = (last_outside + 1) * dt_hours

    # Setpoint-change count
    sp_changes = 0
    prev_sp = None
    for h in history:
        sp = h.get("hp_setpoint")
        if prev_sp is not None and sp != prev_sp:
            sp_changes += 1
        prev_sp = sp

    return KpiBundle(
        n_ticks=n_ticks,
        tick_minutes=tick_minutes,
        tdis_tot=round(tdis_tot, 4),
        cold_time_h=round(cold_h, 3),
        warm_time_h=round(warm_h, 3),
        tdis_tot_eff=round(tdis_eff, 4),
        cold_time_h_eff=round(cold_eff_h, 3),
        warm_time_h_eff=round(warm_eff_h, 3),
        ener_tot=round(ener_tot, 4),
        peak_kw=round(peak_kw, 4),
        settling_time_h=round(settling, 3) if settling is not None else None,
        setpoint_changes=sp_changes,
    )


def attach_learning_kpis(
    bundle: KpiBundle,
    *,
    beta_estimated: dict[str, float],
    beta_truth: dict[str, float],
    n_observations: int,
    residual_whiteness_pass: bool | None = None,
    residual_normality_pass: bool | None = None,
    condition_number: float | None = None,
) -> KpiBundle:
    """Return a copy of ``bundle`` with learning fields populated.

    ``beta_bias`` is computed for each feature present in *both*
    ``beta_estimated`` and ``beta_truth`` — features in only one are
    silently dropped, since "this controller doesn't learn this feature"
    and "ground truth has no value for this feature" are both legitimate
    reasons for asymmetry.
    """
    shared = sorted(set(beta_estimated) & set(beta_truth))
    bias = {k: round(beta_estimated[k] - beta_truth[k], 6) for k in shared}
    return replace(
        bundle,
        beta_estimated={k: round(beta_estimated[k], 6) for k in shared},
        beta_truth={k: round(beta_truth[k], 6) for k in shared},
        beta_bias=bias,
        n_observations=n_observations,
        residual_whiteness_pass=residual_whiteness_pass,
        residual_normality_pass=residual_normality_pass,
        condition_number=(round(condition_number, 4) if condition_number is not None else None),
    )


# ── Multi-realisation aggregator ──────────────────────────────────────────


def crlb_coverage(
    estimates: list[dict[str, float]],
    truths: dict[str, float],
    std_errs: list[dict[str, float]],
    *,
    k: float = 2.0,
) -> dict[str, float]:
    """Fraction of MC realisations whose ``truth`` lies in ``estimate ± k·SE``.

    Reports per-feature coverage rate.  Coverage ≈ 0.95 at k=2.0 is the
    Walter-Pronzato target; significantly less indicates the SE band is
    too tight (over-confident estimator), more indicates a wide band
    (under-confident — usually misspecification inflating residual σ²).
    """
    if not estimates:
        return {}
    features = sorted(set.intersection(*(set(e) for e in estimates)) & set(truths))
    out: dict[str, float] = {}
    for f in features:
        n = 0
        n_in = 0
        for est, se in zip(estimates, std_errs):
            if f not in est or f not in se:
                continue
            half_width = k * se[f]
            if not math.isfinite(half_width):
                # Unidentifiable in this realisation — count as out
                n += 1
                continue
            n += 1
            if abs(est[f] - truths[f]) <= half_width:
                n_in += 1
        out[f] = round(n_in / n, 4) if n > 0 else 0.0
    return out


def aggregate_mc_bundles(bundles: Iterable[KpiBundle]) -> dict[str, dict[str, float]]:
    """Aggregate per-realisation bundles into MC summary statistics.

    Returns a dict of ``{kpi_name: {"mean", "std", "p05", "p95"}}`` for
    each numeric KPI.  Used by the locked-score harness to report the
    Monte-Carlo CI alongside the point estimate.
    """
    bundles_list = list(bundles)
    if not bundles_list:
        return {}
    keys = ["tdis_tot", "cold_time_h", "warm_time_h", "ener_tot",
            "peak_kw", "setpoint_changes"]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        values = [getattr(b, k) for b in bundles_list]
        arr = np.array(values, dtype=float)
        out[k] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "p05": round(float(np.percentile(arr, 5)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "n": len(arr),
        }
    # settling_time_h handled separately (Optional)
    settled = [b.settling_time_h for b in bundles_list if b.settling_time_h is not None]
    if settled:
        arr = np.array(settled, dtype=float)
        out["settling_time_h"] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "p05": round(float(np.percentile(arr, 5)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "n_settled": len(settled),
            "n_total": len(bundles_list),
        }
    return out
