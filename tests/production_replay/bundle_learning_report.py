"""CLI: production-WLS coefficient report for a debug bundle.

Loads the controller's ACTUAL observation buffer from a bundle and re-fits the
PRODUCTION ``weighted_least_squares`` on it (the controller's own solver — no
re-implemented estimator), then prints, per feature:

  - deployed (recorded) : the beta the controller had in the field, read from
                          ``ff_contributions`` in the last tick.
  - current-code        : a fresh batch fit on the same buffer with HEAD
                          (includes the VIF dead-column fix 3da7c7f), plus the
                          fit's std-err / VIF / held flag / detected lag-tau.

…followed by a LEAVE-ONE-OUT feature A/B: drop each model input, re-fit, and
report how the remaining coefficients move.  Built to expose feature
confounds such as an adjacent-zone temperature acting as a lagged solar proxy
(future_work #115) — e.g. dropping the adjacent-zone temp and watching the
solar coefficient shift.

Generic across zones/houses: the model-input wiring (entity_id, name,
input_role, delta_from_room, lag_tau, seeds) is read from the bundle's own
``config.model_inputs`` — nothing hardcoded.  Real-data beta has NO known
truth, so this is a *report*, not a pass/fail gate: read the shifts, don't
assert magnitudes.

Usage:
    .venv/bin/python -m tests.production_replay.bundle_learning_report <bundle_dir> [heat|cool]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from custom_components.tasmota_irhvac.pi.batch_learning import (
    BatchResult,
    Observation,
    weighted_least_squares,
)

_FIXED_HEAD = ["intercept", "outdoor_delta"]
_FIXED_TAIL = ["sin_hour", "cos_hour"]


def _last_tick(bundle: Path) -> dict:
    """Last tick of tick_log.jsonl (config + ff_contributions are stable)."""
    last = None
    with (bundle / "tick_log.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        raise ValueError(f"empty tick_log.jsonl in {bundle}")
    return json.loads(last)


def feature_order(model_inputs: list[dict]) -> list[str]:
    """Production feature layout: intercept, outdoor_delta, <inputs>, sin, cos."""
    return _FIXED_HEAD + [m.get("name", m.get("entity_id", "?"))
                          for m in model_inputs] + _FIXED_TAIL


def recorded_betas(tick: dict, order: list[str]) -> list[float | None]:
    """Deployed coefficients per feature, from ff_contributions."""
    ff = tick.get("ff_contributions") or {}
    out: list[float | None] = []
    for name in order:
        cell = ff.get(name)
        out.append(cell.get("coef") if isinstance(cell, dict) else None)
    return out


def load_observations(bundle: Path, mode: str) -> list[Observation]:
    """The controller's ACTUAL retained buffer (real synchronous current_c)."""
    path = bundle / f"observation_buffer_{mode}.jsonl"
    if not path.exists():
        return []
    obs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                obs.append(Observation.from_dict(json.loads(line)))
    return obs


def _f_to_c(v: float) -> float:
    return (v - 32) * 5.0 / 9.0


def _ha_series(bundle: Path, entity: str) -> list[tuple[float, float]]:
    """Sorted [(wall_epoch, value_°C)] for an entity from ha_history.jsonl."""
    from datetime import datetime
    out = []
    with (bundle / "ha_history.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("entity_id") != entity:
                continue
            try:
                w = datetime.fromisoformat(r["last_changed"]).timestamp()
                v = float(r["state"])
            except (ValueError, KeyError, TypeError):
                continue
            out.append((w, _f_to_c(v) if 40 <= v <= 100 else v))
    out.sort()
    return out


def find_own_sensor(bundle: Path, tick: dict, model_inputs: list[dict]) -> str | None:
    """Own control sensor = the ha_history *_temperature entity that is neither
    the outdoor sensor nor any model input (generic, no hardcoded zone names)."""
    from collections import Counter
    ents = Counter()
    with (bundle / "ha_history.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                ents[json.loads(line).get("entity_id")] += 1
    outdoor = (tick.get("config") or {}).get("outdoor_temp_sensor")
    used = {m.get("entity_id") for m in model_inputs} | {outdoor}
    cand = [e for e in ents
            if e and e.endswith("_temperature") and e not in used]
    return cand[0] if len(cand) == 1 else None


def _interp(series: list[tuple[float, float]], w: float) -> float | None:
    """Linear interpolation of `series` at wall-time `w` (None if out of range)."""
    if not series or w < series[0][0] or w > series[-1][0]:
        return None
    lo, hi = 0, len(series) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if series[mid][0] <= w:
            lo = mid
        else:
            hi = mid
    (w0, v0), (w1, v1) = series[lo], series[hi]
    if w1 == w0:
        return v0
    return v0 + (v1 - v0) * (w - w0) / (w1 - w0)


def reconstruct_observations(bundle: Path, mode: str,
                             model_inputs: list[dict]) -> list[Observation]:
    """Full-span buffer from tick_log model-inputs + ha_history own-temp.

    Everything but the zone's own temperature comes from the tick_log (which
    spans the full bundle); own current_c is interpolated from the zone's own
    control sensor in ha_history (the gap #116 leaves in the tick).
    """
    tick0 = _last_tick(bundle)
    own = find_own_sensor(bundle, tick0, model_inputs)
    if own is None:
        return []
    own_series = _ha_series(bundle, own)
    obs = []
    with (bundle / "tick_log.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            o = t.get("_observation") or {}
            if o.get("mode") != mode:
                continue
            w = t.get("_ts_wall")
            cur = _interp(own_series, w) if w is not None else None
            sp = t.get("hp_setpoint")
            if cur is None or sp is None or not o.get("raw_readings"):
                continue
            desired = t.get("desired_temp")
            desired_c = (_f_to_c(desired) if isinstance(desired, (int, float))
                         and 40 <= desired <= 100 else desired)
            obs.append(Observation(
                timestamp=w, wall_time=w, hp_setpoint=float(sp), current_c=cur,
                desired_c=desired_c, effective_desired_c=t.get("_effective_desired_c"),
                outdoor_temp_c=t.get("outdoor_temp"),
                room_rate=t.get("room_temp_rate", 0.0) or 0.0,
                raw_readings=dict(o.get("raw_readings") or {}),
                clamped=bool(o.get("clamped"))))
    return obs


def validate_reconstruction(bundle: Path, mode: str,
                            model_inputs: list[dict]) -> str:
    """Compare reconstructed own-temp to the real buffer's current_c over the
    overlap — the gate before trusting any reconstructed fit."""
    tick0 = _last_tick(bundle)
    own = find_own_sensor(bundle, tick0, model_inputs)
    if own is None:
        return "  reconstruction self-check: could not uniquely identify own sensor — ABORT"
    series = _ha_series(bundle, own)
    buf = load_observations(bundle, mode)
    diffs = []
    for o in buf:
        r = _interp(series, o.wall_time)
        if r is not None:
            diffs.append(abs(r - o.current_c))
    if not diffs:
        return f"  reconstruction self-check: no overlap with buffer (own={own})"
    diffs.sort()
    rms = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
    p50 = diffs[len(diffs) // 2]
    p95 = diffs[int(len(diffs) * 0.95)]
    return (f"  reconstruction self-check (own={own}): n={len(diffs)} overlap, "
            f"|recon − buffer current_c|  rms={rms:.3f}  p50={p50:.3f}  p95={p95:.3f} °C")


def fit(obs: list[Observation], model_inputs: list[dict],
        current_beta: list[float] | None) -> BatchResult | None:
    """Production batch WLS on `obs` with the given model-input set."""
    order = feature_order(model_inputs)
    return weighted_least_squares(
        obs, n_features=len(order), current_beta=current_beta,
        feature_order=order, model_inputs=model_inputs, detect_lag=True,
    )


def _solar_name(model_inputs: list[dict]) -> str | None:
    for m in model_inputs:
        if m.get("input_role") == "solar":
            return m.get("name")
    return None


def _coef(result: BatchResult | None, order: list[str], name: str) -> float | None:
    if result is None or name not in order:
        return None
    return result.beta_batch[order.index(name)]


def format_report(bundle: Path, mode: str, tick: dict,
                  model_inputs: list[dict], obs: list[Observation]) -> str:
    order = feature_order(model_inputs)
    rec = recorded_betas(tick, order)
    current_beta = [c if c is not None else 0.0 for c in rec]
    full = fit(obs, model_inputs, current_beta)

    L = []
    zone = tick.get("_zone_label", bundle.name)
    L.append(f"\nBundle WLS report — {zone}  [{mode}]")
    L.append(f"  buffer obs:  {len(obs)}"
             + (f"   eligible (|rate|<0.02, unclamped): {full.n_eligible}"
                if full else "   (WLS returned None — insufficient eligible obs)"))
    if full is None:
        return "\n".join(L)
    L.append(f"  residual rms: {full.residual_rms:.4f}")
    if full.detected_tau:
        taus = "  ".join(f"{k}={v/3600:.1f}h" for k, v in full.detected_tau.items())
        L.append(f"  detected lag-τ: {taus}")
    L.append("")
    L.append(f"  {'feature':<24}{'deployed':>11}{'current':>11}{'std_err':>10}"
             f"{'VIF':>9}{'held':>6}")
    L.append(f"  {'-'*24}{'-'*11}{'-'*11}{'-'*10}{'-'*9}{'-'*6}")
    vif = full.feature_vif or [float('nan')] * len(order)
    for i, name in enumerate(order):
        dep = "—" if rec[i] is None else f"{rec[i]:+.3f}"
        cur = f"{full.beta_batch[i]:+.3f}"
        se = f"{full.beta_std_err[i]:.3f}" if i < len(full.beta_std_err) else "—"
        v = f"{vif[i]:.1f}" if i < len(vif) else "—"
        held = "yes" if i in full.held_features else ""
        L.append(f"  {name:<24}{dep:>11}{cur:>11}{se:>10}{v:>9}{held:>6}")

    # Leave-one-out: drop each model input, watch outdoor_delta + solar move.
    solar = _solar_name(model_inputs)
    L.append("")
    L.append("  leave-one-out (drop each model input, re-fit) — Δ vs full fit:")
    L.append(f"    {'dropped input':<24}{'outdoor_delta':>16}{'  '}{'solar':>16}")
    base_od = _coef(full, order, "outdoor_delta")
    base_solar = _coef(full, order, solar) if solar else None
    for drop in model_inputs:
        reduced = [m for m in model_inputs if m is not drop]
        r_order = feature_order(reduced)
        r_beta = [c if c is not None else 0.0
                  for c in recorded_betas(tick, r_order)]
        r = fit(obs, reduced, r_beta)
        od = _coef(r, r_order, "outdoor_delta")
        sol = _coef(r, r_order, solar) if solar and solar != drop.get("name") else None

        def _fmt(new, base):
            if new is None:
                return "—"
            d = f"{new - base:+.3f}" if base is not None else ""
            return f"{new:+.3f} ({d})" if d else f"{new:+.3f}"
        dn = drop.get("name", "?")
        L.append(f"    {dn:<24}{_fmt(od, base_od):>16}{'  '}"
                 f"{(_fmt(sol, base_solar) if solar != dn else 'self'):>16}")
    return "\n".join(L)


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: bundle_learning_report.py <bundle_dir> "
              "[heat|cool] [buffer|reconstruct]")
        print("  buffer (default): the controller's actual retained buffer "
              "(synchronous current_c, but only since the last buffer reset)")
        print("  reconstruct: full tick_log span, own current_c from "
              "ha_history (self-checked vs the buffer)")
        sys.exit(2)
    bundle = Path(args[0])
    rest = args[1:]
    modes = [a for a in rest if a in ("heat", "cool")] or ["heat", "cool"]
    source = next((a for a in rest if a in ("buffer", "reconstruct")), "buffer")
    tick = _last_tick(bundle)
    model_inputs = (tick.get("config") or {}).get("model_inputs") or []
    if not model_inputs:
        print(f"no config.model_inputs in {bundle}"); sys.exit(1)
    for mode in modes:
        if source == "reconstruct":
            print(validate_reconstruction(bundle, mode, model_inputs))
            obs = reconstruct_observations(bundle, mode, model_inputs)
        else:
            obs = load_observations(bundle, mode)
        if not obs:
            print(f"\nBundle WLS report — [{mode}/{source}]: no observations")
            continue
        print(format_report(bundle, mode, tick, model_inputs, obs))


if __name__ == "__main__":
    main()
