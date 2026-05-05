"""Phase B: test the raw-vs-filtered leverage hypothesis.

Hypothesis (from Phase A code review): DiversityAwareBuffer's leverage scoring
uses RAW solar values (via build_feature_vector_from_raw with no
filtered_overrides), but the WLS regression uses EMA-filtered solar with
auto-detected τ (~2-4 hours per Phase A). The mismatch means leverage selects
for raw-solar diversity, not for the filtered-solar identifiability that
regression actually consumes.

Test: subclass DiversityAwareBuffer to use filtered solar in leverage
scoring (matching what WLS sees). Push detected τ to the buffer after each
batch. Run spring 90d. Compare β_solar trajectory vs Phase A baseline.

Reading guide:
- If β_solar holds at -2.0 (instead of drifting to -1.22) → mechanism confirmed.
- If β_solar still drifts → mechanism is something else (likely the bimodal
  solar distribution itself, where solar=0 obs dominate and the leverage
  policy has no way to weight midday peaks higher even with filtered values).
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any


def main() -> int:
    logging.getLogger("custom_components.tasmota_irhvac").setLevel(logging.ERROR)

    from custom_components.tasmota_irhvac.pi.batch_learning import (
        BufferAddResult,
        DiversityAwareBuffer,
        Observation,
        _apply_retrospective_ema,
        build_feature_vector_from_raw,
    )
    from tests.hvac_bench.adapters import TasmotaPIAdapter
    from tests.hvac_bench.full_stack_runner import run_full_stack
    from tests.hvac_bench.scenarios.test_seasonal_convergence import (
        _make_real_config,
    )

    # ── FilteredLeverageBuffer ────────────────────────────────────────────

    class FilteredLeverageBuffer(DiversityAwareBuffer):
        """Leverage scoring uses EMA-filtered values for tracked entities.

        Otherwise identical to DiversityAwareBuffer (same eviction rule,
        same Sherman-Morrison plumbing). When `_tau_per_entity` is empty,
        behaves exactly like the parent.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._tau_per_entity: dict[str, float] = {}
            # id(obs) -> {entity_id: filtered_value}
            self._filtered_by_obs_id: dict[int, dict[str, float]] = {}
            # last EMA state per entity, for extending when new obs arrive
            self._last_ema_state: dict[str, tuple[float, float]] = {}  # eid -> (ema, wt)

        def set_taus(self, taus: dict[str, float]) -> None:
            """Update τ map and recompute filtered values + info matrix."""
            if taus == self._tau_per_entity:
                return
            self._tau_per_entity = dict(taus)
            self._refresh_filtered_cache()
            self.recompute_info_matrix()

        def _refresh_filtered_cache(self) -> None:
            self._filtered_by_obs_id.clear()
            self._last_ema_state.clear()
            if not self._buffer or not self._tau_per_entity:
                return
            for eid, tau in self._tau_per_entity.items():
                filtered = _apply_retrospective_ema(self._buffer, eid, tau)
                # Track the last (ema, wall_time) for incremental extension.
                # _apply_retrospective_ema sorts by wall_time internally; iterate
                # back through buffer in original order, find the latest by wt.
                last_ema = None
                last_wt = -math.inf
                for obs, fval in zip(self._buffer, filtered):
                    if fval is not None:
                        self._filtered_by_obs_id.setdefault(id(obs), {})[eid] = fval
                        if obs.wall_time > last_wt:
                            last_wt = obs.wall_time
                            last_ema = fval
                if last_ema is not None:
                    self._last_ema_state[eid] = (last_ema, last_wt)

        def add(self, obs: Observation) -> BufferAddResult:  # type: ignore[override]
            # Pre-compute the candidate's filtered values so leverage scoring
            # uses the same metric as for incumbents.
            for eid, tau in self._tau_per_entity.items():
                raw = obs.raw_readings.get(eid)
                if raw is None:
                    continue
                state = self._last_ema_state.get(eid)
                if state is None or tau <= 0:
                    fval = float(raw)
                else:
                    last_ema, last_wt = state
                    dt = obs.wall_time - last_wt
                    if dt > 0:
                        alpha = 1.0 - math.exp(-dt / tau)
                        fval = alpha * float(raw) + (1.0 - alpha) * last_ema
                    else:
                        fval = last_ema
                self._filtered_by_obs_id.setdefault(id(obs), {})[eid] = fval
            result = super().add(obs)
            # On admission, advance last_ema_state. On eviction, refresh fully
            # (the evicted obs may have been the last for some entity).
            if result.admitted:
                for eid, tau in self._tau_per_entity.items():
                    fval = self._filtered_by_obs_id.get(id(obs), {}).get(eid)
                    if fval is None:
                        continue
                    state = self._last_ema_state.get(eid)
                    if state is None or obs.wall_time > state[1]:
                        self._last_ema_state[eid] = (fval, obs.wall_time)
            if result.evicted_timestamp is not None:
                # Evicted obs may have been the latest for some entity — full refresh.
                self._refresh_filtered_cache()
            return result

        def _get_feature_vector(self, obs: Observation) -> list[float]:
            if not self._tau_per_entity:
                return super()._get_feature_vector(obs)
            f_overrides = self._filtered_by_obs_id.get(id(obs))
            if (
                f_overrides
                and self._feature_order is not None
                and self._model_inputs is not None
            ):
                vec = build_feature_vector_from_raw(
                    obs,
                    self._model_inputs,
                    self._feature_order,
                    filtered_overrides=f_overrides,
                )
                if vec is not None:
                    return vec
            return super()._get_feature_vector(obs)

        def filter_inactive(self, mode: str) -> int:  # type: ignore[override]
            removed = super().filter_inactive(mode)
            if removed:
                self._refresh_filtered_cache()
            return removed

        def exclude_time_range(self, start: float, end: float) -> int:  # type: ignore[override]
            removed = super().exclude_time_range(start, end)
            if removed:
                self._refresh_filtered_cache()
            return removed

    # ── Buffer + batch wrapper installation ───────────────────────────────

    orig_init = TasmotaPIAdapter.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        pi = self._pi
        n = pi._observation_buffer_heat._n_features
        feature_order = pi._observation_buffer_heat._feature_order
        model_inputs = pi._observation_buffer_heat._model_inputs
        max_size = pi._observation_buffer_heat._max_size
        pi._observation_buffer_heat = FilteredLeverageBuffer(
            n_features=n, max_size=max_size,
            feature_order=feature_order, model_inputs=model_inputs,
        )
        pi._observation_buffer_cool = FilteredLeverageBuffer(
            n_features=n, max_size=max_size,
            feature_order=feature_order, model_inputs=model_inputs,
        )
        # Wrap _run_batch_analysis to push detected τ to the buffer after each batch.
        orig_batch = pi._run_batch_analysis

        def wrapped_batch(*a: Any, **kw: Any) -> Any:
            out = orig_batch(*a, **kw)
            br = pi._last_batch_result
            if br is not None and br.detected_tau:
                # Map name -> entity_id via model_inputs config
                taus_by_eid: dict[str, float] = {}
                for mi in pi._model_inputs:
                    eid = mi.get("entity_id", "")
                    name = mi.get("name", eid)
                    if eid and name in br.detected_tau:
                        taus_by_eid[eid] = br.detected_tau[name]
                if taus_by_eid:
                    pi._observation_buffer_heat.set_taus(taus_by_eid)
                    pi._observation_buffer_cool.set_taus(taus_by_eid)
            return out

        pi._run_batch_analysis = wrapped_batch

    TasmotaPIAdapter.__init__ = patched_init

    try:
        config = _make_real_config("spring", n_days=90)
        mi = config.model_inputs[0]
        print("=" * 78)
        print("Phase B: filtered-leverage buffer (spring 90d, living_room, real CSV)")
        print("=" * 78)
        print(f"Bench truth: β_solar = {mi._true_ff_coef:.4f}, lag_tau = {mi.lag_tau} s")
        print(f"Phase A baseline (raw-leverage): β_solar drifted to -1.22 by day 90")
        print()

        result = run_full_stack(config)
    finally:
        TasmotaPIAdapter.__init__ = orig_init

    print("Final coefficients (filtered-leverage):")
    for k in ("intercept", "outdoor_delta", mi.name):
        v = result.final_coefs.get(k, float("nan"))
        print(f"  {k:>20} = {v:>10.4f}")

    traj = result.coef_trajectory
    if not traj:
        print("\nNo coefficient trajectory captured.")
        return 1

    name = mi.name
    print()
    print(f"Per-batch trajectory for '{name}'  (~2 batches/day):")
    print(f"  {'batch':>6} {'day':>5} {'τ':>8} "
          f"{'β@τ':>8} {'BIC':>8} {'thr':>6} {'acc':>5} {'β_final':>10}")
    print("  " + "-" * 70)
    n_show = max(1, len(traj) // 30)
    for i, snap in enumerate(traj):
        if i % n_show != 0 and i != len(traj) - 1:
            continue
        day = snap["batch"] / 2.0
        tau = snap.get(f"{name}_tau", 0.0)
        beta_at_tau = snap.get(f"{name}_beta_at_tau", 0.0)
        bic_gain = snap.get(f"{name}_bic_gain", 0.0)
        bic_thr = snap.get(f"{name}_bic_threshold", 0.0)
        accepted = snap.get(f"{name}_tau_accepted", False)
        beta_final = snap.get(name, 0.0)
        print(f"  {snap['batch']:>6d} {day:>5.1f} {tau:>8.0f} "
              f"{beta_at_tau:>8.3f} {bic_gain:>8.2f} {bic_thr:>6.2f} "
              f"{str(accepted):>5} {beta_final:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
