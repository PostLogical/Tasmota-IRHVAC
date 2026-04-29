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
from tests.hvac_bench.full_stack_runner import FullStackResult, run_full_stack
from tests.hvac_bench.scenarios.test_buffer_variants import (
    VARIANTS,
    _replace_buffers,
    _print_buffer_fill,
    _print_drift_table,
    _print_final_table,
)
from tests.hvac_bench.scenarios.test_seasonal_convergence import (
    SEASONS,
    _make_synth_config,
)


def _run_with_variant(season_name: str, *, max_size: int, fifo: bool,
                      n_days: int = 90) -> FullStackResult:
    """Run a season (synth AR(1)) with patched buffer config."""
    orig_init = TasmotaPIAdapter.__init__

    def patched(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _replace_buffers(self._pi, max_size=max_size, fifo=fifo)

    TasmotaPIAdapter.__init__ = patched
    try:
        return run_full_stack(_make_synth_config(season_name, n_days=n_days))
    finally:
        TasmotaPIAdapter.__init__ = orig_init


@pytest.fixture(scope="module")
def synth_variant_results() -> dict[str, dict[str, FullStackResult]]:
    """Run all (variant × season) combos with synth weather. 12 sims, ~8 min."""
    pi_logger = logging.getLogger("custom_components.tasmota_irhvac")
    prev = pi_logger.level
    pi_logger.setLevel(logging.ERROR)
    try:
        out: dict[str, dict[str, FullStackResult]] = {}
        for vname, size, fifo in VARIANTS:
            out[vname] = {}
            for sname in SEASONS:
                out[vname][sname] = _run_with_variant(
                    sname, max_size=size, fifo=fifo, n_days=90,
                )
        return out
    finally:
        pi_logger.setLevel(prev)


@pytest.mark.slow
class TestBufferVariantsSynth:
    """Buffer size + policy sweep against synth AR(1) weather."""

    def test_print_summary(self, synth_variant_results):
        """All-in-one: final coefs, trajectory, fill rate. Always passes."""
        _print_buffer_fill(synth_variant_results)
        _print_final_table(synth_variant_results, "outdoor_delta", truth=-0.25)
        _print_final_table(synth_variant_results, "Solar Proxy", truth=-2.0)
        _print_drift_table(synth_variant_results, "Solar Proxy")

    def test_no_variant_diverges(self, synth_variant_results):
        """Sanity: every (variant, season) finishes with bounded coefs."""
        for vname, by_season in synth_variant_results.items():
            for sname, r in by_season.items():
                od = r.final_coefs.get("outdoor_delta", 0.0)
                solar = r.final_coefs.get("Solar Proxy", 0.0)
                assert -2.0 < od < 0.0, f"{vname}/{sname}: outdoor_delta={od:.4f}"
                assert -5.0 < solar < 1.0, f"{vname}/{sname}: solar={solar:.4f}"
