"""Phase D: A/B/baseline test — disentangle sin/cos role in regression vs leverage.

Three variants (spring 90d each, ~3 min × 3 = ~10 min total):

- baseline: production code (sin/cos as nuisance in WLS, sin/cos in leverage)
- B (no_tod_in_leverage): drop sin/cos from buffer feature_order; WLS unchanged
- A (no_tod_anywhere): drop from buffer + monkey-patch tod_features→(0,0) so the
  WLS variance check kills the sin/cos nuisance branch in _solve_joint/_solve_fwl

For each variant: capture β_solar trajectory + VIF(solar) per batch.

Hypothesis tests:
1. Phase C said leverage's ToD-coupled scoring rejects mid-solar daytime obs.
   B should help (β_solar closer to truth than baseline) and A should help at
   least as much (B is a strict subset of A's changes to the buffer side).
2. The user's challenge: sin/cos in WLS may not actually be load-bearing — the
   real protection against wrong-sign β_solar is clamp+gate guardrails. Test:
   does A drift further from truth than B (sin/cos removal hurts WLS)?
   - If A ≈ B in β_solar quality → sin/cos in WLS isn't doing decisive work
     in the bench; A is preferable (also lowers VIF for production unlock)
   - If A clearly worse than B → sin/cos in WLS IS doing real work; B is right

3. Production unlock hypothesis: removing sin/cos from WLS lowers VIF(solar),
   which would let unlock pass the vif gate. Direct test: VIF(solar)
   trajectory baseline vs A.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal


Variant = Literal["baseline", "no_tod_in_leverage", "no_tod_anywhere"]


def run_variant(variant: Variant, n_days: int = 90) -> dict[str, Any]:
    """Run spring N-day with a given variant. Returns trajectory + final state."""
    from custom_components.tasmota_irhvac.pi import batch_learning as bl_mod
    from custom_components.tasmota_irhvac.pi.batch_learning import (
        DiversityAwareBuffer,
    )
    from tests.hvac_bench.adapters import TasmotaPIAdapter
    from tests.hvac_bench.full_stack_runner import run_full_stack
    from tests.hvac_bench.scenarios.test_seasonal_convergence import (
        _make_real_config,
    )

    captured: dict[str, Any] = {}
    orig_init = TasmotaPIAdapter.__init__
    orig_tod = bl_mod.tod_features

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        pi = self._pi
        captured["pi"] = pi

        if variant == "baseline":
            return
        # Both B and A: replace buffer with feature_order excluding sin/cos.
        fo_full = pi._observation_buffer_heat._feature_order
        fo_no_tod = [f for f in fo_full if f not in ("sin_hour", "cos_hour")]
        n_no_tod = len(fo_no_tod)
        max_size = pi._observation_buffer_heat._max_size
        model_inputs = pi._observation_buffer_heat._model_inputs
        pi._observation_buffer_heat = DiversityAwareBuffer(
            n_features=n_no_tod, max_size=max_size,
            feature_order=fo_no_tod, model_inputs=model_inputs,
        )
        pi._observation_buffer_cool = DiversityAwareBuffer(
            n_features=n_no_tod, max_size=max_size,
            feature_order=fo_no_tod, model_inputs=model_inputs,
        )

    if variant == "no_tod_anywhere":
        # Patch tod_features so the variance-gate in _solve_joint/_solve_fwl
        # rejects the ToD-nuisance branch. Also affects _detect_optimal_tau and
        # build_feature_vector_from_raw (latter is fine — ToD columns become
        # zero in any context that still uses build_feature_vector_from_raw,
        # and the buffer's feature_order excludes them anyway).
        bl_mod.tod_features = lambda wt: (0.0, 0.0)

    TasmotaPIAdapter.__init__ = patched_init
    try:
        config = _make_real_config("spring", n_days=n_days)
        result = run_full_stack(config)
    finally:
        TasmotaPIAdapter.__init__ = orig_init
        bl_mod.tod_features = orig_tod

    return {
        "variant": variant,
        "result": result,
        "pi": captured.get("pi"),
        "n_days": n_days,
    }


def main() -> int:
    logging.getLogger("custom_components.tasmota_irhvac").setLevel(logging.ERROR)

    variants: list[Variant] = ["baseline", "no_tod_in_leverage", "no_tod_anywhere"]
    runs = []
    for v in variants:
        print(f"\n>>> Running {v} (spring 90d)...")
        runs.append(run_variant(v, n_days=90))

    name = "Solar Proxy"

    # ── Final coefficient comparison ────────────────────────────────────
    print()
    print("=" * 78)
    print("Final coefficients across variants:")
    print("=" * 78)
    print(f"  {'variant':<22} {'intercept':>11} {'outdoor':>10} "
          f"{'β_solar':>10} {'gap':>8}")
    print("  " + "-" * 65)
    for run in runs:
        r = run["result"]
        ic = r.final_coefs.get("intercept", float("nan"))
        od = r.final_coefs.get("outdoor_delta", float("nan"))
        bs = r.final_coefs.get(name, float("nan"))
        gap = abs(bs - (-2.0)) if bs == bs else float("nan")  # NaN-safe
        print(f"  {run['variant']:<22} {ic:>11.4f} {od:>10.4f} "
              f"{bs:>10.4f} {gap:>8.4f}")

    # ── β_solar trajectory side-by-side ────────────────────────────────
    print()
    print("β_solar trajectory (every ~5 days):")
    print(f"  {'day':>5} " + "  ".join(
        f"{r['variant']:<22}" for r in runs))
    print("  " + "-" * (5 + sum(24 for _ in runs)))
    trajs = [r["result"].coef_trajectory for r in runs]
    n_max = max(len(t) for t in trajs)
    n_show = max(1, n_max // 18)
    for i in range(n_max):
        if i % n_show != 0 and i != n_max - 1:
            continue
        day = i / 2.0
        cells: list[str] = []
        for t in trajs:
            if i < len(t):
                v = t[i].get(name, 0.0)
                cells.append(f"{v:>22.4f}")
            else:
                cells.append(f"{'':>22}")
        print(f"  {day:>5.1f} " + "  ".join(cells))

    # ── VIF trajectory side-by-side ────────────────────────────────────
    print()
    print("VIF(Solar Proxy) trajectory (every ~5 days):")
    print(f"  {'day':>5} " + "  ".join(
        f"{r['variant']:<22}" for r in runs))
    print("  " + "-" * (5 + sum(24 for _ in runs)))
    for i in range(n_max):
        if i % n_show != 0 and i != n_max - 1:
            continue
        day = i / 2.0
        cells = []
        for t in trajs:
            if i < len(t):
                v = t[i].get(f"{name}_vif", 0.0)
                cells.append(f"{v:>22.3f}")
            else:
                cells.append(f"{'':>22}")
        print(f"  {day:>5.1f} " + "  ".join(cells))

    # ── VIF aggregate ────────────────────────────────────
    print()
    print("VIF(Solar Proxy) summary:")
    print(f"  {'variant':<22} {'mean':>8} {'median':>8} {'max':>8}")
    print("  " + "-" * 50)
    for run in runs:
        traj = run["result"].coef_trajectory
        vifs = [snap.get(f"{name}_vif", float("nan")) for snap in traj
                if f"{name}_vif" in snap]
        if not vifs:
            continue
        vifs_sorted = sorted(vifs)
        mean_v = sum(vifs) / len(vifs)
        median_v = vifs_sorted[len(vifs_sorted) // 2]
        max_v = max(vifs)
        print(f"  {run['variant']:<22} {mean_v:>8.2f} {median_v:>8.2f} {max_v:>8.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
