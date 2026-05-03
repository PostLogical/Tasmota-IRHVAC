"""Hot-path benchmark for `_build_tick_output()` (Stage 11).

Asserts that the per-tick snapshot build cost stays within budget. The
hot path runs on every state write that triggers `fire_dispatcher` —
typically once per PI tick (1-15s in production). The earlier estimate
was 50-100 µs; we assert mean < 200 µs to leave headroom for state
machines (Stage 5c), event emitters (Stage 8), and any future
additions.

Catches regressions like accidentally adding O(n²) work to a
per-tick build site.
"""

from __future__ import annotations

import time

import pytest

from .conftest import get_climate_entity


@pytest.mark.asyncio
async def test_build_tick_output_under_200us_mean(hass, setup_pi_integration):
    """1000 iterations: mean < 200 µs, p99 < 1000 µs.

    Mean is the headline budget. p99 catches occasional spikes that
    might indicate GC pressure or non-amortized work.
    """
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    pi = get_climate_entity(hass, entry)._controller

    # Warm-up: avoid first-call dispatch + import costs polluting the
    # measurement. JIT, attribute caches, and Python's optimizations
    # stabilize after a handful of iterations.
    for _ in range(50):
        pi._build_tick_output()

    n = 1000
    timings: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        pi._build_tick_output()
        timings.append(time.perf_counter() - start)

    mean_us = (sum(timings) / n) * 1_000_000
    p99_us = sorted(timings)[int(n * 0.99)] * 1_000_000

    # Asserts with informative messages so a regression has a clear story.
    assert mean_us < 200, (
        f"_build_tick_output mean too slow: {mean_us:.1f} µs (budget 200 µs). "
        f"Hot-path regression — investigate recent additions to the snapshot "
        f"builder or sub-snapshot helpers."
    )
    assert p99_us < 1000, (
        f"_build_tick_output p99 too slow: {p99_us:.1f} µs (budget 1000 µs). "
        f"Spiky behavior suggests non-amortized work; check observation "
        f"buffer iteration or any synchronous I/O slipped in."
    )
