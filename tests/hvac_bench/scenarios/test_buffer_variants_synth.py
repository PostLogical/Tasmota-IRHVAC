"""Buffer variant sweep — synth-only path for parameter sweeps (#45).

For verdict-producing buffer comparisons see ``test_buffer_variants.py``
(canonical, real-weather). This file is reserved for sweeps where
*reproducible synth knobs* matter more than weather realism — varying
``weather_amp_c``, sweeping seeds, swapping cloud-coupling, etc.

Reuses the FIFOBuffer + buffer-replacement helpers + VARIANTS list from
the canonical file; only the weather source differs (synth AR(1) here vs
real CSV in the canonical). Kept slow-marked to mirror the canonical.
"""

from __future__ import annotations

import logging

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.full_stack_runner import (
    FullStackResult,
    print_full_stack_summary,
    run_full_stack,
)
from tests.hvac_bench.scenarios.test_buffer_variants import (
    _compute_variant_results,
    _print_variant_summary,
    _replace_buffers,
)
from tests.hvac_bench.scenarios.test_seasonal_convergence import (
    _make_synth_config,
)


def _run_with_variant(season_name: str, *, max_size: int, policy: str,
                      n_days: int = 90) -> FullStackResult:
    """Run a season (synth AR(1)) with patched buffer config."""
    orig_init = TasmotaPIAdapter.__init__

    def patched(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _replace_buffers(self._pi, max_size=max_size, policy=policy)

    TasmotaPIAdapter.__init__ = patched
    try:
        return run_full_stack(_make_synth_config(season_name, n_days=n_days))
    finally:
        TasmotaPIAdapter.__init__ = orig_init


def _compute_synth_variant_results() -> dict[str, dict[str, FullStackResult]]:
    """Run all (variant × season) combos with synth AR(1) weather."""
    return _compute_variant_results(runner=_run_with_variant)


def run_buffer_variants_synth_summary() -> None:
    """CLI entry: per-run generic summary for each (variant × season),
    then the cross-variant comparison tables (synth weather)."""
    results = _compute_synth_variant_results()
    for vname, by_season in results.items():
        for sname, result in by_season.items():
            print_full_stack_summary(result, label=f"{vname} / {sname} (synth)")
    _print_variant_summary(results)


@pytest.fixture(scope="module")
def synth_variant_results() -> dict[str, dict[str, FullStackResult]]:
    """Run all (variant × season) combos with synth weather. 12 sims, ~8 min."""
    return _compute_synth_variant_results()


@pytest.mark.design
@pytest.mark.study
class TestBufferVariantsSynth:
    """Buffer size + policy sweep against synth AR(1) weather.

    Marked ``design`` (not ``slow``): synth-only sibling of
    ``test_buffer_variants``, kept for parameter sweeps where reproducible
    knobs matter. See ``feedback_synthetic_vs_real_bench.md`` — the
    FIFO/leverage verdict comes from the real-weather sibling, not this one.
    """

    def test_no_variant_diverges(self, bench_metrics, num_regression, synth_variant_results):
        """Sanity: every (variant, season) finishes with bounded coefs."""
        for vname, by_season in synth_variant_results.items():
            for sname, r in by_season.items():
                od = r.final_coefs.get("outdoor_delta", 0.0)
                solar = r.final_coefs.get("Solar Proxy", 0.0)
                # Flat-namespaced key per (variant, season) so all combos
                # land in one CSV row.
                bench_metrics[f"{vname}__{sname}__outdoor_delta"] = od
                bench_metrics[f"{vname}__{sname}__solar"] = solar
                assert -2.0 < od < 0.0, f"{vname}/{sname}: outdoor_delta={od:.4f}"
                assert -5.0 < solar < 1.0, f"{vname}/{sname}: solar={solar:.4f}"
        check_bench_metrics(num_regression, bench_metrics)
