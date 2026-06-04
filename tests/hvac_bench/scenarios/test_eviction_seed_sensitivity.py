"""WLS eviction-seed sensitivity — does buffer-eviction RNG move WLS coefs?

Scope (post-fill-and-wipe, 2026-06-04): this test now studies WLS only.
The grey-box observer moved to a standalone fill-and-wipe buffer in
commit 4770943, so eviction RNG no longer affects greybox fits. WLS's
DiversityAwareBuffer still uses ``SlevPolicy(alpha=0.0, seed=0)``
(uniform-random eviction), and the seed-luck concern continues to apply
to WLS-fitted Solar Proxy / outdoor_delta in low-SNR seasons.

The original observation that motivated this test: fall final Solar
Proxy was −1.10 at seed 0 but −1.59 at seed 42 (same code, same data) —
the data under-determines the solar coefficient and *which* observations
random eviction keeps shifts the final estimate.

The test runs the same scenario across N eviction seeds and pins the
spread of the WLS-learned coefficients.  A coefficient whose seed-induced
spread is large relative to its own value is being driven by RNG luck,
not data.  The pinned spread lets a future code change (deterministic
eviction, seed-ensembling, added excitation, …) be measured: did it make
the WLS learner more seed-robust?

Marked @study (N× the cost of a single run); opt in with --run-studies.
Greybox is no longer affected — its fill-and-wipe buffer is RNG-free
and the same fits would occur regardless of WLS eviction seed.
"""
from __future__ import annotations

import statistics

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import DiversityAwareBuffer
from custom_components.tasmota_irhvac.pi.buffer_policies import SlevPolicy
from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.full_stack_runner import run_full_stack
from tests.hvac_bench.scenarios.test_seasonal_convergence import _make_config

_TICK_MINUTES = 5.0
_N_DAYS = 90
_MAX_SIZE = 2000
# Low-SNR seasons (fall, spring_plus_60) are where eviction luck bites; spring
# is the high-SNR control that should stay seed-robust.
_SCENARIOS = ("fall", "spring_plus_60", "spring")
_SEEDS = (0, 1, 2, 3, 7, 13, 42, 99)
_COEFS = ("Solar Proxy", "outdoor_delta")
_TRUTH = {"Solar Proxy": -2.0, "outdoor_delta": -0.25}


def _run_with_seed(scenario: str, seed: int):
    """Run a scenario with uniform-random eviction at a given RNG seed."""
    orig_init = TasmotaPIAdapter.__init__

    def patched(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        pi = self._pi
        buf = pi._observation_buffer_heat
        kw = dict(
            n_features=buf._n_features, max_size=_MAX_SIZE,
            feature_order=buf._feature_order, model_inputs=buf._model_inputs,
        )
        pi._observation_buffer_heat = DiversityAwareBuffer(
            **kw, policy=SlevPolicy(alpha=0.0, seed=seed))
        pi._observation_buffer_cool = DiversityAwareBuffer(
            **kw, policy=SlevPolicy(alpha=0.0, seed=seed))

    TasmotaPIAdapter.__init__ = patched
    try:
        config = _make_config(scenario, n_days=_N_DAYS)
        config.tick_minutes = _TICK_MINUTES
        return run_full_stack(config)
    finally:
        TasmotaPIAdapter.__init__ = orig_init


@pytest.mark.study
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_eviction_seed_sensitivity(scenario, bench_metrics, num_regression) -> None:
    """Run one scenario across N eviction seeds; pin the coefficient spread.

    The pinned per-seed values + spread stats are the deliverable.  The only
    pass/fail gate is loose non-divergence — the point is to watch how the
    seed-induced spread changes as the code changes.
    """
    finals: dict[str, list[float]] = {c: [] for c in _COEFS}
    for seed in _SEEDS:
        r = _run_with_seed(scenario, seed)
        for c in _COEFS:
            finals[c].append(r.final_coefs.get(c, float("nan")))

    bench_metrics["n_seeds"] = float(len(_SEEDS))
    for c in _COEFS:
        key = c.replace(" ", "_")
        vals = finals[c]
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        lo, hi = min(vals), max(vals)
        bench_metrics[f"{key}_truth"] = _TRUTH[c]
        bench_metrics[f"{key}_mean"] = mean
        bench_metrics[f"{key}_std_across_seeds"] = std
        bench_metrics[f"{key}_min"] = lo
        bench_metrics[f"{key}_max"] = hi
        bench_metrics[f"{key}_range"] = hi - lo
        bench_metrics[f"{key}_mean_err"] = abs(mean - _TRUTH[c])
        for seed, v in zip(_SEEDS, vals):
            bench_metrics[f"{key}_seed{seed:02d}"] = v

    check_bench_metrics(num_regression, bench_metrics)

    # Loose non-divergence gate only (the spread itself is the pinned signal).
    for c in _COEFS:
        for v in finals[c]:
            assert -10.0 < v < 5.0, f"{scenario}: {c} diverged across seeds: {v:.4f}"
