"""Unit tests for the Monte Carlo runner."""

from __future__ import annotations

from unittest.mock import patch

from .full_stack_runner import FullStackConfig
from .house_profiles import PROFILES_2R2C
from .monte_carlo import MonteCarloConfig, MonteCarloResult, run_monte_carlo


def _capture_per_run_overrides(mc: MonteCarloConfig) -> list[dict]:
    """Run the MC loop with run_full_stack/_aggregate stubbed so we only
    observe the per-run pi_overrides built by ``run_monte_carlo``.
    """
    captured: list[dict] = []

    def capture(cfg, checkpoints=None):
        captured.append(dict(cfg.pi_overrides))
        return None  # _aggregate is patched too, so the value is unused

    empty_result = MonteCarloResult(
        runs=[],
        itae_percentiles={},
        violations_percentiles={},
        reversals_percentiles={},
        integral_rms_percentiles={},
        coef_percentiles={},
        coef_errors_percentiles={},
        gelman_rubin={},
        convergence_rate=0.0,
        batches_to_converge_percentiles={},
    )

    with patch("tests.hvac_bench.monte_carlo.run_full_stack", side_effect=capture), \
         patch("tests.hvac_bench.monte_carlo._aggregate", return_value=empty_result):
        run_monte_carlo(mc)

    return captured


def test_seed_scale_factors_actually_scale_outdoor_seed(bench_metrics, num_regression):
    """seed_scale_factors must scale pi_outdoor_seed_heat/_cool, not write a no-op key.

    Regression test for C4: previous implementation wrote
    ``pi_outdoor_seed_heat_scale`` (a non-existent config key) into
    pi_overrides, where it was silently dropped by make_pi_config.  The
    intended behavior is that run i sees ``profile.true_seed * scales[i]``
    as the actual outdoor seed.
    """
    profile = PROFILES_2R2C["living_room"]
    base = FullStackConfig(profile_name="living_room", n_days=1)
    scales = [0.5, 1.0, 1.5]
    mc = MonteCarloConfig(
        base_config=base,
        n_runs=3,
        seed_scale_factors=scales,
    )

    captured = _capture_per_run_overrides(mc)

    assert len(captured) == 3
    expected = profile.true_seed
    for i, scale in enumerate(scales):
        overrides = captured[i]
        assert overrides.get("pi_outdoor_seed_heat") == expected * scale, (
            f"run {i} (scale={scale}) expected pi_outdoor_seed_heat="
            f"{expected * scale}, got {overrides.get('pi_outdoor_seed_heat')}"
        )
        assert overrides.get("pi_outdoor_seed_cool") == expected * scale
        assert "pi_outdoor_seed_heat_scale" not in overrides, (
            "non-existent key 'pi_outdoor_seed_heat_scale' must not be written"
        )


def test_seed_scale_factors_respect_explicit_pi_override(bench_metrics, num_regression):
    """If user sets pi_outdoor_seed_heat in pi_overrides, MC must not overwrite it."""
    base = FullStackConfig(
        profile_name="living_room",
        n_days=1,
        pi_overrides={"pi_outdoor_seed_heat": 99.9},
    )
    mc = MonteCarloConfig(
        base_config=base,
        n_runs=2,
        seed_scale_factors=[0.5, 2.0],
    )

    captured = _capture_per_run_overrides(mc)

    for overrides in captured:
        assert overrides["pi_outdoor_seed_heat"] == 99.9
