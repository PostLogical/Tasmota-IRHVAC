"""Phase A diagnostic: WLS β_solar bias floor — is lag-tau the source?

Background: project_bench_solar_fidelity.md documents a ~40% bias floor on
β_solar in the full-stack bench (truth -2.0, identified ~-1.2). Hypothesis 2
of that memo: the lag-tau auto-detect picks a tau that mis-aligns the
EMA-filtered solar with the actual room response, biasing β toward zero.

This driver runs the canonical seasonal-convergence spring scenario
(90 days, real Open-Meteo CSV, living_room, _true_ff_coef=-2.0) with
default settings (detect_lag=True), then prints:

- Per-batch lag-tau search trajectory (chosen τ, β at τ, BIC accept/reject)
- Final β_solar and final β_outdoor for context
- Configured bench truth: lag_tau=120s, β_solar=-2.0

Reading guide:
- If chosen τ ≈ 120s consistently AND β_at_tau ≈ -2.0 AND final β_solar ≈ -1.2
  → lag-tau is exonerated; the bias is downstream (ToD partialling or the
  joint solve in _solve_joint).
- If chosen τ is far from 120s OR β_at_tau is itself ~-1.2 → lag-tau or the
  search's joint regression form IS plausibly the source; Phase C ToD-off
  test will discriminate further.
"""

from __future__ import annotations

import logging
import sys


def main() -> int:
    # Silence integration logging so the table reads cleanly.
    logging.getLogger("custom_components.tasmota_irhvac").setLevel(logging.ERROR)

    from tests.hvac_bench.full_stack_runner import run_full_stack
    from tests.hvac_bench.scenarios.test_seasonal_convergence import (
        _make_real_config,
    )

    config = _make_real_config("spring", n_days=90)
    mi = config.model_inputs[0]
    print("=" * 78)
    print("Phase A: lag-tau diagnostic (spring 90d, living_room, real CSV)")
    print("=" * 78)
    print(f"Bench truth:")
    print(f"  β_solar      = {mi._true_ff_coef:.4f}")
    print(f"  lag_tau      = {mi.lag_tau} s   (configured bench solar EMA lag)")
    print(f"  τ search bds = [0, 28800] s    (production WLS search range)")
    print()

    result = run_full_stack(config)

    # Final β
    print("Final coefficients:")
    for k in ("intercept", "outdoor_delta", mi.name):
        v = result.final_coefs.get(k, float("nan"))
        print(f"  {k:>20} = {v:>10.4f}")

    traj = result.coef_trajectory
    if not traj:
        print("\nNo coefficient trajectory captured.")
        return 1

    name = mi.name
    has_tau = f"{name}_tau" in traj[-1]
    if not has_tau:
        print(f"\nNo lag-tau diagnostic captured for '{name}' "
              f"— check _snapshot_coefs in full_stack_runner.py.")
        return 1

    print()
    print(f"Per-batch lag-tau trajectory for '{name}'  (~2 batches/day):")
    print(f"  {'batch':>6} {'day':>5} {'τ':>8} {'τ_raw':>8} "
          f"{'β@τ':>8} {'BIC':>8} {'thr':>6} {'acc':>5} {'β_final':>10}")
    print("  " + "-" * 76)
    n_show = max(1, len(traj) // 30)
    for i, snap in enumerate(traj):
        if i % n_show != 0 and i != len(traj) - 1:
            continue
        day = snap["batch"] / 2.0
        tau = snap.get(f"{name}_tau", 0.0)
        tau_raw = snap.get(f"{name}_tau_opt_raw", 0.0)
        beta_at_tau = snap.get(f"{name}_beta_at_tau", 0.0)
        bic_gain = snap.get(f"{name}_bic_gain", 0.0)
        bic_thr = snap.get(f"{name}_bic_threshold", 0.0)
        accepted = snap.get(f"{name}_tau_accepted", False)
        beta_final = snap.get(name, 0.0)
        print(f"  {snap['batch']:>6d} {day:>5.1f} {tau:>8.0f} {tau_raw:>8.0f} "
              f"{beta_at_tau:>8.3f} {bic_gain:>8.2f} {bic_thr:>6.2f} "
              f"{str(accepted):>5} {beta_final:>10.4f}")

    # Aggregate τ statistics across all batches
    taus = [snap.get(f"{name}_tau_opt_raw", 0.0) for snap in traj
            if f"{name}_tau_opt_raw" in snap]
    accepted_taus = [snap.get(f"{name}_tau", 0.0) for snap in traj
                     if snap.get(f"{name}_tau_accepted")]
    n_acc = sum(1 for snap in traj if snap.get(f"{name}_tau_accepted"))
    n_total = sum(1 for snap in traj if f"{name}_tau" in snap)

    print()
    print("τ search summary:")
    print(f"  batches with τ search    : {n_total}")
    print(f"  batches BIC-accepted     : {n_acc} ({100 * n_acc / max(1, n_total):.0f}%)")
    if taus:
        print(f"  τ_opt_raw  min/med/max   : "
              f"{min(taus):.0f} / {sorted(taus)[len(taus) // 2]:.0f} / {max(taus):.0f} s")
    if accepted_taus:
        print(f"  τ accepted min/med/max   : "
              f"{min(accepted_taus):.0f} / "
              f"{sorted(accepted_taus)[len(accepted_taus) // 2]:.0f} / "
              f"{max(accepted_taus):.0f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
