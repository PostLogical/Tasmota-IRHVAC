#!/usr/bin/env python
"""Analyze a pytest-regressions baseline diff and label each moved metric
better / worse / neutral — so we stop squinting at dozens of raw CSV diffs.

Three judgement modes per metric (first match wins):
  toward-truth : metric has a `<base>_truth` sibling (or maps to the file's
                 solar/outdoor truth) → improved iff |new−truth| < |old−truth|.
                 The RIGHT call for coefficients (−2.07 beats −1.94 only because
                 truth is −2.0; raw lower/higher would mislabel it).
  lower-better : errors / variance / cost / chatter (err, mae, rms, itae, std,
                 drift, |bias|, reversals, overshoot, settling, beeps, kappa…).
  higher-better: quality (comfort, COP, utilization, ff_fraction, yield).
Anything not confidently classified is FLAGGED (?), never guessed — the point is
to not trade a manual hunt for a confident-wrong summary. Coefficients with no
truth column in their file get a distinct "eyeball vs truth" flag.

Moves below the significance threshold (default 1%) are treated as run-to-run
noise: still tallied, but kept out of the magnitude-sorted REVIEW list.

Usage (defaults to HEAD vs working tree — the pre-commit review case):
    .venv/bin/python local/tools/analyze_regen.py                 # HEAD -> worktree
    .venv/bin/python local/tools/analyze_regen.py HEAD~1 HEAD     # parent -> commit
    .venv/bin/python local/tools/analyze_regen.py A B [path] [-v] [--pct=2]
"""
from __future__ import annotations

import csv
import glob
import io
import math
import os
import re
import subprocess
import sys

_DEFAULT_SCOPE = "tests/hvac_bench/regression_data"
_REL_TOL, _ABS_TOL = 1e-6, 1e-9   # repo pytest-regressions tolerance (a real move)
_SIG_PCT = 0.01                   # relative gate: below this = run-to-run noise
_ABS_FLOOR = 1e-4                 # absolute gate: kills huge-% but tiny-magnitude
                                  # from-near-zero moves (e.g. drift 0 → 3e-5)
_REVIEW_CAP = 30

# Substring rules. _verdict order: truth → |bias| → LOWER → HIGHER → COEF → NEUTRAL.
# Order matters (outdoor_delta_std hits LOWER('std') before COEF('outdoor')).
_LOWER = ("_err", "error", "mae", "rms", "itae", "overshoot", "undershoot",
          "reversal", "settling_time", "drift", "_std", "violation", "cold_ticks",
          "cold_max", "cold_time", "setpoint_changes", "changes_per_day", "beeps",
          "kappa", "abs_dev", "spread", "kwh_per_degree", "peak_irms", "tdis",
          "gate_fail", "integral", "diff")   # gate failures / windup / seed-spread
_HIGHER = ("comfort", "avg_cop", "_cop", "util", "ff_fraction", "yield",
           "gates_passed", "_ff", "first_week", "third_week")  # gates / FF-fraction (LOWER's
                                              # 'rms' catches first_week_rms before this)
_COEF = ("beta", "outdoor", "solar")   # coef value, no truth col → flag (don't guess)
_NEUTRAL = ("n_ticks", "tick_minutes", "max_size", "seed_factor", "fill_day",
            "room_temp", "setpoint_min", "setpoint_max", "obs_count", "batches",
            "unlock_day", "_tau", "detected_tau", "total_kwh", "peak_kw",
            "time_h", "warm_time", "n_windows", "intercept", "rls_obs",
            "n_residual", "n_observation", "cal_min", "cal_max", "final_cal",
            "baseline", "median_ratio", "bias_over_se", "_trace",
            "late_trace", "yield_pct", "dispatch_rate", "wls_normal",
            "wls_total", "s_without", "sat_capacity", "sat_delta", "stable_",
            "shift")                          # behaviour rates / counts / sentinels (no direction)
_SOLAR_TRUTH_COLS = ("Solar_Proxy_truth", "post_fill_Solar_Proxy_truth", "solar_truth")
_OUTDOOR_TRUTH_COLS = ("outdoor_delta_truth", "post_fill_outdoor_delta_truth", "outdoor_truth")


def _sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def _changed_files(ref_a, ref_b, scope):
    cmd = ["git", "diff", "--name-only", ref_a] + ([ref_b] if ref_b else []) + ["--", scope]
    return [f for f in _sh(*cmd).splitlines() if f.endswith(".csv")]


def _read(ref, path):
    txt = _sh("git", "show", f"{ref}:{path}") if ref else open(path).read()
    return list(csv.DictReader(io.StringIO(txt))) if txt.strip() else []


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _changed(old, new):
    if old is None or new is None:
        return old != new
    return abs(new - old) > _ABS_TOL + _REL_TOL * abs(old)


def _truth_for(metric, keys, row):
    sib = f"{metric}_truth"
    if sib in keys:
        return _f(row.get(sib))
    # A coefficient VALUE sampled at a point in time — ``X_final`` /
    # ``X_at_fill`` / ``X_at_unlock`` / ``X_traj_dNN`` — shares its
    # coefficient's ``X_truth`` sibling (system_learning_report et al.).  This
    # is what lets us judge a raw coef move toward/away truth instead of
    # punting to the "eyeball" flag.  Exclude ``_err`` columns (those are
    # already truth-distances, judged lower-better) and ``_tau`` (lag has no
    # clean truth).
    if "_err" not in metric and "_tau" not in metric:
        mch = re.match(r"(.+?)_(?:final|at_fill|at_unlock|traj_d\d+)$", metric)
        if mch and f"{mch.group(1)}_truth" in keys:
            return _f(row.get(f"{mch.group(1)}_truth"))
    low = metric.lower()
    if low.startswith(("beta_solar", "traj_solar")) or low == "solar_proxy_final":
        for c in _SOLAR_TRUTH_COLS:
            if c in keys:
                return _f(row.get(c))
    if low.startswith("beta_outdoor") or low == "outdoor_delta_final":
        for c in _OUTDOOR_TRUTH_COLS:
            if c in keys:
                return _f(row.get(c))
    return None


def _verdict(metric, old, new, truth):
    if old is None or new is None:
        return "?", "non-numeric"
    low = metric.lower()
    if truth is not None and not math.isnan(truth):
        do, dn = abs(old - truth), abs(new - truth)
        if abs(dn - do) <= _ABS_TOL + _REL_TOL * abs(do):
            return "=", f"toward-truth({truth:+.3g})"
        return ("✓" if dn < do else "✗"), f"toward-truth({truth:+.3g})"
    if low == "bias" or low.endswith("_bias"):
        return ("✓" if abs(new) < abs(old) else "✗"), "|bias|↓"
    for s in _LOWER:
        if s in low:
            return ("✓" if new < old else "✗"), "lower↓"
    for s in _HIGHER:
        if s in low:
            return ("✓" if new > old else "✗"), "higher↑"
    if "tau" in low:   # lag-τ is descriptive (no clean truth) — neutral, not a coef
        return "·", "neutral(τ)"
    if low == "od" or low.startswith("od_") or low.endswith("_od") or any(s in low for s in _COEF):
        return "?", "coef↦eyeball vs truth"
    for s in _NEUTRAL:
        if s in low:
            return "·", "neutral"
    return "?", "UNCLASSIFIED"


def _fmt(v):
    return f"{v:+.4g}" if v is not None else "—"


def _parse_log(path):
    """Pull regen-run health from a pytest output log: tests that did NOT pass
    (so --regen-all could NOT write their baseline → STALE) + skip/xpass counts."""
    txt = open(path).read()
    failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", txt, re.M)

    def _count(word):
        m = re.search(rf"(\d+) {word}", txt)
        return int(m.group(1)) if m else 0
    return {"failed": failed, "skipped": _count("skipped"),
            "xpassed": _count("xpassed"), "errored": _count("error[s ]")}


def _baselines_for(node_id):
    """Map a pytest node id to its regression baseline CSV(s) across datadirs."""
    parts = node_id.split("::")
    mod = os.path.basename(parts[0])[:-3] if parts[0].endswith(".py") else parts[0]
    func = parts[-1].split("[")[0]
    return sorted(glob.glob(f"{_DEFAULT_SCOPE}/*/{mod}/{func}*.csv"))


def _print_freshness(log_path):
    """Flag baselines that the regen did NOT actually refresh (the stale trap)."""
    info = _parse_log(log_path)
    print(f"\n  ⚑ BASELINE FRESHNESS (from {os.path.basename(log_path)}):")
    if info["failed"]:
        print(f"    {len(info['failed'])} FAILED/ERRORED → --regen-all could NOT write these (STALE):")
        for nid in info["failed"]:
            print(f"      {nid}")
            bls = _baselines_for(nid)
            for b in bls:
                print(f"          ↳ STALE baseline: {b.replace(_DEFAULT_SCOPE + '/', '')}")
            if not bls:
                print("          ↳ (no baseline matched — assert-only / non-pytest-regressions)")
    if info["skipped"]:
        print(f"    {info['skipped']} SKIPPED → their baselines were NOT regenerated. If a whole "
              f"tier was skipped (@study needs --run-studies; tick variants need --tick-minutes), "
              f"those baselines are STALE.")
    if info["xpassed"]:
        print(f"    {info['xpassed']} XPASSED → behavior changed; the xfail markers may be stale.")
    if not (info["failed"] or info["skipped"] or info["xpassed"]):
        print("    clean: no failed / skipped / xpassed reported.")
    print("    (for FULL freshness, run the regen with -rA so passed-test baselines are "
          "cross-checkable — see feedback_regen_playbook.)")


def main():
    args = sys.argv[1:]
    verbose = any(a in ("-v", "--verbose") for a in args)
    pct = next((float(a.split("=")[1]) / 100 for a in args if a.startswith("--pct=")), _SIG_PCT)
    abs_floor = next((float(a.split("=")[1]) for a in args if a.startswith("--abs=")), _ABS_FLOOR)
    log_path = next((a.split("=", 1)[1] for a in args if a.startswith("--log=")), None)
    args = [a for a in args if a not in ("-v", "--verbose")
            and not a.startswith(("--pct=", "--log=", "--abs="))]

    def _sig(rel, ov, nv):  # significant only if BOTH relative AND absolute move
        return rel >= pct and ov is not None and nv is not None and abs(nv - ov) >= abs_floor
    refs = [a for a in args if not a.startswith("tests/")]
    paths = [a for a in args if a.startswith("tests/")]
    ref_a = refs[0] if refs else "HEAD"
    ref_b = refs[1] if len(refs) > 1 else None
    scope = paths[0] if paths else _DEFAULT_SCOPE

    files = _changed_files(ref_a, ref_b, scope)
    print(f"regen diff: {ref_a} -> {ref_b or 'working tree'}   "
          f"({len(files)} changed CSVs under {scope})\n")
    if not files:
        return

    tally = {k: 0 for k in "✓✗·?="}
    sig = {k: 0 for k in "✓✗·?="}
    review, per_file, unmapped, vanished = [], {}, {}, []
    for path in files:
        old_rows, new_rows = _read(ref_a, path), _read(ref_b, path)
        short = path.replace(f"{_DEFAULT_SCOPE}/", "")
        for i in range(max(len(old_rows), len(new_rows))):
            orow = old_rows[i] if i < len(old_rows) else {}
            nrow = new_rows[i] if i < len(new_rows) else {}
            keys = list(nrow.keys()) or list(orow.keys())
            for m in keys:
                if not m or m.isdigit():
                    continue
                ov, nv = _f(orow.get(m)), _f(nrow.get(m))
                if not _changed(ov, nv):
                    continue
                sym, kind = _verdict(m, ov, nv, _truth_for(m, keys, nrow or orow))
                # Relative move with a FLOORED, symmetric denominator: a
                # near-zero baseline (drift ≈ 0) must not manufacture a giant %
                # from a tiny absolute change (0 → 2e-4 is ~100%, not 4039%).
                rel = (abs(nv - ov) / max(abs(ov), abs(nv), _ABS_FLOOR)
                       if (ov is not None and nv is not None) else math.inf)
                tally[sym] += 1
                if _sig(rel, ov, nv):
                    sig[sym] += 1
                if verbose:
                    per_file.setdefault(short, []).append((sym, m, ov, nv, kind))
                if kind == "non-numeric":   # value↔empty/NaN — structural, its own bucket
                    vanished.append((short, m, ov, nv))
                    continue
                if kind == "UNCLASSIFIED":
                    unmapped.setdefault(m, (ov, nv))
                if sym in ("✗", "?"):
                    adl = (abs(nv - ov) if (ov is not None and nv is not None)
                           else math.inf)
                    review.append((adl, rel, short, m, ov, nv, kind))

    if verbose:
        for short, rows in sorted(per_file.items()):
            print(f"━━ {short}")
            for sym, m, ov, nv, kind in rows:
                print(f"   {sym} {m:34} {_fmt(ov):>11} → {_fmt(nv):<11} {kind}")
            print()

    print("═" * 64)
    print(f"  totals : ✓ better {tally['✓']}  ✗ worse {tally['✗']}  "
          f"= no-change {tally['=']}  · neutral {tally['·']}  ? flag {tally['?']}")
    print(f"  ≥{pct:.0%} & ≥{abs_floor:g} : ✓ {sig['✓']}  ✗ {sig['✗']}  ? {sig['?']}   "
          f"(smaller = run-to-run noise)")
    # Rank by ABSOLUTE delta — "how much it actually moved" — so a tiny-but-
    # high-% near-zero move can't crowd out a genuinely large one.
    sig_rev = sorted((r for r in review if _sig(r[1], r[4], r[5])), reverse=True)
    if sig_rev:
        print(f"\n  ⚠ REVIEW — significant worse/unclassified, biggest ABSOLUTE move "
              f"first (top {min(_REVIEW_CAP, len(sig_rev))} of {len(sig_rev)}):")
        for adl, rel, short, m, ov, nv, kind in sig_rev[:_REVIEW_CAP]:
            print(f"    [Δ {adl:>9.3g}  {rel:>4.0%}] {short} :: {m}  "
                  f"{_fmt(ov)}→{_fmt(nv)}  [{kind}]")

    if vanished:
        print(f"\n  ⊘ VALUE↔EMPTY ({len(vanished)}) — metric started/stopped computing "
              f"(NaN/None); structural change, check it's intended:")
        for short, m, ov, nv in vanished[:_REVIEW_CAP]:
            print(f"    {short} :: {m}  {_fmt(ov)}→{_fmt(nv)}")

    if unmapped:
        print(f"\n  ⚠ UNMAPPED METRICS ({len(unmapped)}) — no direction rule; add to "
              f"analyze_regen.py so they're judged, not flagged:")
        for m, (ov, nv) in sorted(unmapped.items()):
            print(f"    {m:36} (e.g. {_fmt(ov)}→{_fmt(nv)})")

    if log_path:
        _print_freshness(log_path)


if __name__ == "__main__":
    main()
