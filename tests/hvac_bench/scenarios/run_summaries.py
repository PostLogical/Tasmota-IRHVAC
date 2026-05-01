"""CLI entry point for bench scenario summary reports (Future Work #50).

Verdict-recapture tables that used to live as ``@pytest.mark.design``
``test_print_*`` methods now run from this module. Pytest stays for real
regression assertions; the printed summaries are reports, not tests.

Usage::

    python -m tests.hvac_bench.scenarios.run_summaries seasonal
    python -m tests.hvac_bench.scenarios.run_summaries buffer_variants
    python -m tests.hvac_bench.scenarios.run_summaries buffer_variants_synth
    python -m tests.hvac_bench.scenarios.run_summaries hp_capacity
    python -m tests.hvac_bench.scenarios.run_summaries all
    python -m tests.hvac_bench.scenarios.run_summaries --list

Each scenario file owns its ``run_<name>_summary()`` callable; this module is a
thin registry + argparse front-end so adding a new summary is a one-line edit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from tests.hvac_bench.scenarios.test_buffer_variants import (
    run_buffer_variants_summary,
)
from tests.hvac_bench.scenarios.test_buffer_variants_synth import (
    run_buffer_variants_synth_summary,
)
from tests.hvac_bench.scenarios.test_hp_capacity_curve import (
    run_hp_capacity_summary,
)
from tests.hvac_bench.scenarios.test_seasonal_convergence import (
    run_seasonal_summary,
)
from tests.hvac_bench.scenarios.test_tobit_convergence_quality import (
    run_tobit_convergence_quality_summary,
)
from tests.hvac_bench.scenarios.test_wls_tobit import (
    run_wls_tobit_summary,
)


REGISTRY: dict[str, Callable[[], None]] = {
    "seasonal": run_seasonal_summary,
    "buffer_variants": run_buffer_variants_summary,
    "buffer_variants_synth": run_buffer_variants_synth_summary,
    "hp_capacity": run_hp_capacity_summary,
    "tobit_convergence_quality": run_tobit_convergence_quality_summary,
    "wls_tobit": run_wls_tobit_summary,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.hvac_bench.scenarios.run_summaries",
        description="Run bench scenario summary reports.",
    )
    p.add_argument(
        "scenario",
        nargs="?",
        choices=[*REGISTRY.keys(), "all"],
        help="Scenario to run, or 'all' for every scenario in sequence.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list or args.scenario is None:
        for name in REGISTRY:
            print(name)
        return 0
    if args.scenario == "all":
        for name, fn in REGISTRY.items():
            print(f"\n{'#' * 72}\n# {name}\n{'#' * 72}")
            fn()
        return 0
    REGISTRY[args.scenario]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
