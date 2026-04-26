"""Monte Carlo runner for full-stack learning validation.

Wraps FullStackRunner to execute N independent runs with varied initial
conditions, then computes percentile distributions and convergence
diagnostics.

Key techniques:
- Latin Hypercube sampling across seed errors, noise seeds, forgetting factors
- Gelman-Rubin R̂ diagnostic for coefficient convergence across chains
- Percentile-based assertions (p50, p95) on comfort and learning metrics
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable

from .full_stack_runner import (
    FullStackConfig,
    FullStackResult,
    run_full_stack,
    Checkpoint,
)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo sweep."""

    # Base config (will be cloned and varied per run)
    base_config: FullStackConfig

    # Number of runs
    n_runs: int = 20

    # What to vary across runs.  Each is a list of values to sample from.
    # If None, the base_config value is used for all runs.
    noise_seeds: list[int] | None = None  # sensor noise seeds
    seed_scale_factors: list[float] | None = None  # FF seed multipliers
    # Additional pi_overrides to sweep (dict key -> list of values)
    pi_override_sweeps: dict[str, list[float]] = field(default_factory=dict)

    # Checkpoints (shared across all runs)
    checkpoints: list[Checkpoint] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    """Aggregated results from Monte Carlo sweep."""

    # Individual run results
    runs: list[FullStackResult]

    # Percentile distributions of key metrics
    itae_percentiles: dict[str, float]  # p5, p25, p50, p75, p95
    violations_percentiles: dict[str, float]
    reversals_percentiles: dict[str, float]
    integral_rms_percentiles: dict[str, float]

    # Per-coefficient convergence
    coef_percentiles: dict[str, dict[str, float]]  # coef_name -> percentiles
    coef_errors_percentiles: dict[str, dict[str, float]]

    # Gelman-Rubin R̂ per coefficient (< 1.1 = converged)
    gelman_rubin: dict[str, float]

    # Convergence rate: fraction of runs that converged
    convergence_rate: float
    batches_to_converge_percentiles: dict[str, float]


def run_monte_carlo(mc_config: MonteCarloConfig) -> MonteCarloResult:
    """Execute N independent full-stack runs with varied conditions.

    Args:
        mc_config: Monte Carlo configuration.

    Returns:
        MonteCarloResult with aggregated statistics.
    """
    n = mc_config.n_runs
    base = mc_config.base_config

    # Generate per-run configs
    noise_seeds = mc_config.noise_seeds or list(range(n))
    seed_scales = mc_config.seed_scale_factors or [1.0] * n

    # Ensure we have enough values (cycle if needed)
    noise_seeds = _cycle(noise_seeds, n)
    seed_scales = _cycle(seed_scales, n)

    results: list[FullStackResult] = []

    for i in range(n):
        cfg = _clone_config(base)
        cfg.noise_seed = noise_seeds[i]

        # Scale FF seeds
        scale = seed_scales[i]
        if "pi_outdoor_seed_heat" not in cfg.pi_overrides:
            # Will be set by runner from profile.true_seed; scale it
            cfg.pi_overrides["pi_outdoor_seed_heat_scale"] = scale
        for mi in cfg.model_inputs:
            mi.seed_heat *= scale

        # Apply pi_override sweeps
        for key, values in mc_config.pi_override_sweeps.items():
            cfg.pi_overrides[key] = values[i % len(values)]

        result = run_full_stack(cfg, checkpoints=mc_config.checkpoints)
        results.append(result)

    # Aggregate
    return _aggregate(results)


def _aggregate(results: list[FullStackResult]) -> MonteCarloResult:
    """Compute percentiles and diagnostics across runs."""
    n = len(results)

    # Scalar metric percentiles
    itae_vals = [r.total_itae for r in results]
    viol_vals = [r.total_violations for r in results]
    rev_vals = [r.total_reversals for r in results]
    irms_vals = [r.integral_rms for r in results]

    # Coefficient final values and errors
    all_coef_names = set()
    for r in results:
        all_coef_names.update(r.final_coefs.keys())

    coef_pcts: dict[str, dict[str, float]] = {}
    coef_err_pcts: dict[str, dict[str, float]] = {}
    for name in all_coef_names:
        vals = [r.final_coefs.get(name, 0.0) for r in results]
        errs = [r.coef_errors.get(name, 0.0) for r in results]
        coef_pcts[name] = _percentiles(vals)
        coef_err_pcts[name] = _percentiles(errs)

    # Gelman-Rubin R̂ per coefficient
    gr: dict[str, float] = {}
    for name in all_coef_names:
        # Use coefficient trajectory as chains
        chains = []
        for r in results:
            chain = [snap.get(name, 0.0) for snap in r.coef_trajectory
                     if name in snap]
            if chain:
                chains.append(chain)
        if len(chains) >= 2:
            gr[name] = _gelman_rubin(chains)

    # Convergence rate
    converged = [r for r in results if r.batches_to_converge is not None]
    conv_rate = len(converged) / n if n > 0 else 0.0
    conv_batches = [r.batches_to_converge for r in converged]

    return MonteCarloResult(
        runs=results,
        itae_percentiles=_percentiles(itae_vals),
        violations_percentiles=_percentiles([float(v) for v in viol_vals]),
        reversals_percentiles=_percentiles([float(v) for v in rev_vals]),
        integral_rms_percentiles=_percentiles(irms_vals),
        coef_percentiles=coef_pcts,
        coef_errors_percentiles=coef_err_pcts,
        gelman_rubin=gr,
        convergence_rate=conv_rate,
        batches_to_converge_percentiles=(
            _percentiles([float(b) for b in conv_batches])
            if conv_batches else {}
        ),
    )


# ── Gelman-Rubin R̂ ──────────────────────────────────────────────────────


def _gelman_rubin(chains: list[list[float]]) -> float:
    """Compute Gelman-Rubin R̂ convergence diagnostic.

    R̂ < 1.1 indicates chains have converged to the same distribution.
    Uses the simplified single-parameter version.

    Args:
        chains: List of M chains, each a list of N samples.
            Chains may differ in length; truncated to shortest.
    """
    if len(chains) < 2:
        return float("inf")

    # Truncate to shortest chain
    min_len = min(len(c) for c in chains)
    if min_len < 2:
        return float("inf")
    chains = [c[:min_len] for c in chains]

    m = len(chains)  # number of chains
    n = min_len  # samples per chain

    # Per-chain means and variances
    chain_means = [sum(c) / n for c in chains]
    chain_vars = [
        sum((x - mu) ** 2 for x in c) / (n - 1)
        for c, mu in zip(chains, chain_means)
    ]

    # Grand mean
    grand_mean = sum(chain_means) / m

    # Between-chain variance B
    b = n * sum((mu - grand_mean) ** 2 for mu in chain_means) / (m - 1)

    # Within-chain variance W
    w = sum(chain_vars) / m

    if w == 0:
        return 1.0 if b == 0 else float("inf")

    # Pooled variance estimate
    var_hat = (1 - 1 / n) * w + (1 / n) * b

    # R̂
    r_hat = math.sqrt(var_hat / w)
    return r_hat


# ── Helpers ──────────────────────────────────────────────────────────────


def _percentiles(
    values: list[float],
    pcts: tuple[int, ...] = (5, 25, 50, 75, 95),
) -> dict[str, float]:
    """Compute percentiles from a list of values."""
    if not values:
        return {f"p{p}": 0.0 for p in pcts}
    s = sorted(values)
    n = len(s)
    result = {}
    for p in pcts:
        idx = min(int(n * p / 100), n - 1)
        result[f"p{p}"] = s[idx]
    return result


def _cycle(lst: list, n: int) -> list:
    """Extend a list to length n by cycling."""
    if not lst:
        return [0] * n
    return [lst[i % len(lst)] for i in range(n)]


def _clone_config(cfg: FullStackConfig) -> FullStackConfig:
    """Deep-copy a FullStackConfig."""
    new = copy.copy(cfg)
    new.model_inputs = [copy.copy(mi) for mi in cfg.model_inputs]
    new.disturbances = list(cfg.disturbances)
    new.pi_overrides = dict(cfg.pi_overrides)
    if cfg.true_coefs is not None:
        new.true_coefs = dict(cfg.true_coefs)
    return new
