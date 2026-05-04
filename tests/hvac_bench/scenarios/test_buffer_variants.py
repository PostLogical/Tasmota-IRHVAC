"""Buffer variant sweep — canonical real-weather scenarios (#45).

Default weather source is real Open-Meteo CSV per ``_make_config`` in
``test_seasonal_convergence`` (#45, 2026-04-29). Compares Solar Proxy and
outdoor_delta convergence across (variant × season) for the calibrated
living_room profile under three buffer sizes (1000/2000/4000) and FIFO
vs leverage-scored eviction.

Real cloud clustering, weather fronts, dawn/dusk gradients, and seasonal
day-length shifts are exactly the distributional features a leverage policy
could be sensitive to — the synthetic v1 of this test produced a clean
finding (FIFO recovers solar truth) that real weather inverted (see
``feedback_synthetic_vs_real_bench.md``). This file is the verdict source;
``test_buffer_variants_synth.py`` is reserved for synth-only parameter
sweeps where reproducible knobs matter more than realism.
"""

from __future__ import annotations

import logging

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    DiversityAwareBuffer,
    Observation,
)
from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.full_stack_runner import (
    FullStackResult,
    print_full_stack_summary,
    run_full_stack,
)
from tests.hvac_bench.scenarios.test_seasonal_convergence import (
    SEASONS,
    _make_config,  # default: real CSV
)


# ── FIFO buffer subclass ─────────────────────────────────────────────────


class FIFOBuffer(DiversityAwareBuffer):
    """Same interface as DiversityAwareBuffer, but evicts oldest obs.

    The "regular buffer" — no leverage scoring, no clever retention.
    When full, the oldest observation falls out as a new one comes in.
    Info matrix is maintained via Sherman-Morrison downdate/update so
    WLS still operates over the current buffer contents.
    """

    def add(self, obs: Observation) -> None:
        x = self._get_feature_vector(obs)
        if len(self._buffer) < self._max_size:
            self._buffer.append(obs)
            self._sherman_morrison_update(x)
        else:
            old_x = self._get_feature_vector(self._buffer[0])
            self._sherman_morrison_downdate(old_x)
            self._buffer.pop(0)
            self._buffer.append(obs)
            self._sherman_morrison_update(x)


# ── Variant patching ─────────────────────────────────────────────────────


def _replace_buffers(pi, *, max_size: int, fifo: bool) -> None:
    """Swap heat/cool buffers on a freshly-built PI for the given variant."""
    cls = FIFOBuffer if fifo else DiversityAwareBuffer
    n = pi._observation_buffer_heat._n_features
    feature_order = pi._observation_buffer_heat._feature_order
    model_inputs = pi._observation_buffer_heat._model_inputs
    pi._observation_buffer_heat = cls(
        n_features=n,
        max_size=max_size,
        feature_order=feature_order,
        model_inputs=model_inputs,
    )
    pi._observation_buffer_cool = cls(
        n_features=n,
        max_size=max_size,
        feature_order=feature_order,
        model_inputs=model_inputs,
    )


VARIANTS: list[tuple[str, int, bool]] = [
    ("default-2000",  2000, False),
    ("half-1000",     1000, False),
    ("double-4000",   4000, False),
    ("FIFO-2000",     2000, True),
]


def _run_with_variant(season_name: str, *, max_size: int, fifo: bool,
                      n_days: int = 90) -> FullStackResult:
    """Run a season (real CSV) with patched buffer config."""
    orig_init = TasmotaPIAdapter.__init__

    def patched(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _replace_buffers(self._pi, max_size=max_size, fifo=fifo)

    TasmotaPIAdapter.__init__ = patched
    try:
        return run_full_stack(_make_config(season_name, n_days=n_days))
    finally:
        TasmotaPIAdapter.__init__ = orig_init


# ── Fixture ──────────────────────────────────────────────────────────────


def _compute_variant_results(
    *, runner=_run_with_variant, n_days: int = 90,
) -> dict[str, dict[str, FullStackResult]]:
    """Run all (variant × season) combos once. 12 sims, ~8 minutes total.

    Shared by the pytest fixture and the CLI runner. ``runner`` is injected so
    the synth sibling can reuse this loop with its own per-variant runner.
    """
    pi_logger = logging.getLogger("custom_components.tasmota_irhvac")
    prev = pi_logger.level
    pi_logger.setLevel(logging.ERROR)
    try:
        out: dict[str, dict[str, FullStackResult]] = {}
        for vname, size, fifo in VARIANTS:
            out[vname] = {}
            for sname in SEASONS:
                out[vname][sname] = runner(
                    sname, max_size=size, fifo=fifo, n_days=n_days,
                )
        return out
    finally:
        pi_logger.setLevel(prev)


def _print_variant_summary(
    results: dict[str, dict[str, FullStackResult]],
) -> None:
    """All-in-one report: buffer fill, final coefs (outdoor + solar), drift table."""
    _print_buffer_fill(results)
    _print_final_table(results, "outdoor_delta", truth=-0.25)
    _print_final_table(results, "Solar Proxy", truth=-2.0)
    _print_drift_table(results, "Solar Proxy")


def run_buffer_variants_summary() -> None:
    """CLI entry: per-run generic summary for each (variant × season),
    then the cross-variant comparison tables."""
    results = _compute_variant_results()
    for vname, by_season in results.items():
        for sname, result in by_season.items():
            print_full_stack_summary(result, label=f"{vname} / {sname}")
    _print_variant_summary(results)


@pytest.fixture(scope="module")
def variant_results() -> dict[str, dict[str, FullStackResult]]:
    """Run all (variant × season) combos once. 12 sims, ~8 minutes total."""
    return _compute_variant_results()


# ── Pretty printing ──────────────────────────────────────────────────────


def _print_final_table(
    results: dict[str, dict[str, FullStackResult]],
    coef: str,
    truth: float,
) -> None:
    seasons = list(SEASONS.keys())
    print(f"\n=== Final {coef} (truth ≈ {truth:.2f}) ===")
    header = f"{'Variant':<16}" + "".join(f"{s:>11}" for s in seasons) + f"{'Spread':>10}"
    print(header)
    print("-" * len(header))
    for vname in results:
        vals = [results[vname][s].final_coefs.get(coef, 0.0) for s in seasons]
        spread = max(vals) - min(vals)
        row = "".join(f"{v:>11.4f}" for v in vals)
        print(f"{vname:<16}{row}{spread:>10.4f}")


def _print_drift_table(
    results: dict[str, dict[str, FullStackResult]],
    coef: str,
) -> None:
    """Per-variant per-season values at days 15/30/45/60/75/90."""
    seasons = list(SEASONS.keys())
    days = [15, 30, 45, 60, 75, 90]
    print(f"\n=== {coef} trajectory (day 15 → day 90) ===")
    header = f"{'Variant':<16}{'Season':<10}" + "".join(f"d{d:>3}".rjust(11) for d in days)
    print(header)
    print("-" * len(header))
    for vname in results:
        for s in seasons:
            traj = results[vname][s].coef_trajectory
            row = []
            for d in days:
                target = int(d * 2)
                if not traj:
                    row.append(float("nan"))
                else:
                    idx = min(target, len(traj) - 1)
                    row.append(traj[idx].get(coef, 0.0))
            row_s = "".join(f"{v:>11.4f}" for v in row)
            print(f"{vname:<16}{s:<10}{row_s}")
        print()


def _print_buffer_fill(results: dict[str, dict[str, FullStackResult]]) -> None:
    """Show when each variant's buffer reaches 100%."""
    seasons = list(SEASONS.keys())
    print(f"\n=== Buffer fill (day at which utilization first reaches 100%) ===")
    header = f"{'Variant':<16}" + "".join(f"{s:>11}" for s in seasons)
    print(header)
    print("-" * len(header))
    for vname in results:
        row = []
        for s in seasons:
            r = results[vname][s]
            day_full = next(
                (i for i, u in enumerate(r.daily_buffer_utilization) if u >= 0.999),
                None,
            )
            row.append(f"{day_full}" if day_full is not None else "<90 days never")
        print(f"{vname:<16}" + "".join(f"{v:>11}" for v in row))


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.design
@pytest.mark.study
class TestBufferVariants:
    """Compare buffer size + policy across heating seasons (real weather).

    Marked ``design`` (not ``slow``): the FIFO-vs-leverage verdict is
    captured in ``project_buffer_seasonal_findings.md``. Production only
    uses ``DiversityAwareBuffer-2000``, exercised by ``test_seasonal_convergence``.
    Re-run this study only when revisiting buffer policy.
    """

    def test_no_variant_diverges(self, variant_results):
        """Sanity: every (variant, season) finishes with bounded coefs."""
        for vname, by_season in variant_results.items():
            for sname, r in by_season.items():
                od = r.final_coefs.get("outdoor_delta", 0.0)
                solar = r.final_coefs.get("Solar Proxy", 0.0)
                assert -2.0 < od < 0.0, f"{vname}/{sname}: outdoor_delta={od:.4f}"
                assert -5.0 < solar < 1.0, f"{vname}/{sname}: solar={solar:.4f}"
