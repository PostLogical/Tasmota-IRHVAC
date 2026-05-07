"""Phase E: no-eviction control. Is eviction itself the cause of β_solar drift?

Modifies the buffer to refuse new admissions once full (no Sherman-Morrison
update either — pure freeze on first 2000 observations admitted).

Spring 90d. Binary read:

- β_solar stays at the day-25 value (~-1.92): eviction IS the lever. Whatever
  the policy evicts is load-bearing for solar identifiability. Next step:
  design the right eviction restraint (representation floors, bigger buffer,
  FIFO+protection, etc.).

- β_solar still drifts toward -1.22: eviction is NOT the cause. The drift
  must come from something else compounding with more data — Sherman-Morrison
  numerical drift in the info matrix, or a bias mechanism in WLS that
  worsens with more (perhaps biased) batch updates.

Note: the user's broader concern is preserving cross-season memory while
also preserving in-season identifiability. This test doesn't address that
directly — it just isolates whether eviction is the dynamic that drives
in-season drift.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def main() -> int:
    logging.getLogger("custom_components.tasmota_irhvac").setLevel(logging.ERROR)

    from custom_components.tasmota_irhvac.pi.batch_learning import (
        BufferAddResult,
        DiversityAwareBuffer,
        Observation,
    )
    from tests.hvac_bench.adapters import TasmotaPIAdapter
    from tests.hvac_bench.full_stack_runner import run_full_stack
    from tests.hvac_bench.scenarios.test_seasonal_convergence import (
        _make_real_config,
    )

    class NoEvictionBuffer(DiversityAwareBuffer):
        """Accept until full; then refuse all new observations."""

        def add(self, obs: Observation) -> BufferAddResult:  # type: ignore[override]
            x = self._get_feature_vector(obs)
            new_score = self._compute_leverage(x)
            if len(self._buffer) < self._max_size:
                self._buffer.append(obs)
                self._sherman_morrison_update(x)
                return BufferAddResult(
                    admitted=True,
                    candidate_score=new_score,
                    evicted_timestamp=None,
                    min_incumbent_score=None,
                    rejection_reason=None,
                    policy_name="no_eviction",
                )
            return BufferAddResult(
                admitted=False,
                candidate_score=new_score,
                evicted_timestamp=None,
                min_incumbent_score=None,
                rejection_reason="no_eviction_buffer_full",
                policy_name="no_eviction",
            )

    captured: dict[str, Any] = {}
    orig_init = TasmotaPIAdapter.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        pi = self._pi
        captured["pi"] = pi
        n = pi._observation_buffer_heat._n_features
        feature_order = pi._observation_buffer_heat._feature_order
        model_inputs = pi._observation_buffer_heat._model_inputs
        max_size = pi._observation_buffer_heat._max_size
        pi._observation_buffer_heat = NoEvictionBuffer(
            n_features=n, max_size=max_size,
            feature_order=feature_order, model_inputs=model_inputs,
        )
        pi._observation_buffer_cool = NoEvictionBuffer(
            n_features=n, max_size=max_size,
            feature_order=feature_order, model_inputs=model_inputs,
        )

    TasmotaPIAdapter.__init__ = patched_init
    try:
        config = _make_real_config("spring", n_days=90)
        mi = config.model_inputs[0]
        print("=" * 78)
        print("Phase E: NO-EVICTION control (spring 90d)")
        print("=" * 78)
        print(f"Truth: β_solar = {mi._true_ff_coef:.4f}")
        print(f"Baseline (production buffer): drifts to -1.22 by day 90")
        print(f"Day-25 baseline value: ~-1.92 (just before eviction starts)")
        print()
        result = run_full_stack(config)
        pi = captured["pi"]
    finally:
        TasmotaPIAdapter.__init__ = orig_init

    name = mi.name
    print("Final coefficients:")
    for k in ("intercept", "outdoor_delta", name):
        v = result.final_coefs.get(k, float("nan"))
        print(f"  {k:>20} = {v:>10.4f}")

    buf = pi._observation_buffer_heat
    print()
    print(f"Buffer state: {len(buf.get_all())} obs (max_size={buf._max_size})")
    history = result.history
    attempts = [h for h in history if h.get("obs_admitted") is not None]
    admitted = sum(1 for h in attempts if h["obs_admitted"])
    refused = sum(1 for h in attempts if not h["obs_admitted"])
    print(f"  admission attempts: {len(attempts)}")
    print(f"  admitted: {admitted}")
    print(f"  refused (buffer full, no eviction): {refused}")

    # Trajectory
    traj = result.coef_trajectory
    print()
    print("β_solar trajectory:")
    print(f"  {'day':>5} {'β_solar':>10} {'τ':>8} {'VIF':>8}")
    print("  " + "-" * 38)
    n_show = max(1, len(traj) // 25)
    for i, snap in enumerate(traj):
        if i % n_show != 0 and i != len(traj) - 1:
            continue
        day = snap["batch"] / 2.0
        bs = snap.get(name, 0.0)
        tau = snap.get(f"{name}_tau", 0.0)
        vif = snap.get(f"{name}_vif", 0.0)
        print(f"  {day:>5.1f} {bs:>10.4f} {tau:>8.0f} {vif:>8.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
