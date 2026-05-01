"""Unit tests for the Richardson extrapolation module (Phase 3a).

Pure-numerics tests on synthetic ``KPI(h) = a + b·h^p`` data — verifies
that:

* Two-grid extrapolation with the correct assumed order recovers ``a``.
* Three-grid extrapolation recovers both ``p`` and ``a`` from synthetic
  data with known order.
* The non-uniform refinement branch matches the uniform branch when
  refinement happens to be uniform.
* Degenerate cases (constant KPI, non-monotone deltas, missing values,
  shape mismatches) report sensibly without crashing.
* The bundle-level ``kpi_richardson_sweep`` walks ``KpiBundle`` fields
  correctly.

No bench scenarios are run here — that's the next test file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.hvac_bench.kpis import KpiBundle
from tests.hvac_bench.richardson import (
    DEFAULT_RICHARDSON_KPIS,
    REGIME_ORDER_MAX,
    REGIME_ORDER_MIN,
    ROACHE_FS_ASYMPTOTIC,
    ROACHE_FS_NON_ASYMPTOTIC,
    RichardsonReport,
    format_richardson_table,
    kpi_richardson_sweep,
    richardson_extrapolate,
    tick_rate_spread,
)


# ── Two-grid recovery ─────────────────────────────────────────────────────


class TestTwoGridExtrapolation:
    """Two grids with the right assumed order recover the limit exactly."""

    def test_first_order_exact_recovery(self):
        # KPI(h) = 5.0 + 2.0 * h with truth a=5.0, p=1
        h = [30.0, 15.0]
        truth = 5.0
        slope = 2.0
        k = [truth + slope * x for x in h]
        rep = richardson_extrapolate(h, k, assumed_order=1.0, kpi_name="test")
        assert rep.extrapolated_value == pytest.approx(truth, abs=1e-9)
        assert rep.observed_order == 1.0
        assert rep.fit_method == "two_point_assumed_order"
        assert rep.monotone_convergence is True
        assert rep.error_band == pytest.approx(slope * h[-1], abs=1e-9)

    def test_second_order_exact_recovery(self):
        # KPI(h) = 10.0 + 0.5 * h^2 with assumed_order=2 → recover 10.0
        h = [30.0, 10.0]
        truth = 10.0
        c = 0.5
        k = [truth + c * x ** 2 for x in h]
        rep = richardson_extrapolate(h, k, assumed_order=2.0, kpi_name="test")
        assert rep.extrapolated_value == pytest.approx(truth, abs=1e-9)
        assert rep.observed_order == 2.0
        assert rep.error_band == pytest.approx(c * h[-1] ** 2, abs=1e-9)

    def test_unsorted_input_handled(self):
        h = [15.0, 30.0]  # finest first → must be sorted internally
        k = [7.0, 9.0]    # KPI at h=15 is finer
        rep = richardson_extrapolate(h, k, assumed_order=1.0, kpi_name="test")
        # After sorting descending: h=[30,15], k=[9,7]
        # extrap = (2*7 - 9)/(2-1) = 5
        assert rep.extrapolated_value == pytest.approx(5.0, abs=1e-9)
        assert rep.tick_rates_minutes == [30.0, 15.0]
        assert rep.kpi_values == [9.0, 7.0]


# ── Three-grid order recovery ─────────────────────────────────────────────


class TestThreeGridOrderRecovery:
    """≥ 3 grids with uniform refinement recover the convergence order."""

    def test_first_order_recovers_p_one(self):
        h = [40.0, 20.0, 10.0]  # uniform refinement r=2
        truth = 5.0
        slope = 0.3
        k = [truth + slope * x for x in h]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.observed_order == pytest.approx(1.0, abs=1e-6)
        assert rep.extrapolated_value == pytest.approx(truth, abs=1e-6)
        assert rep.fit_method == "three_point_observed_order"
        assert rep.monotone_convergence is True

    def test_second_order_recovers_p_two(self):
        h = [40.0, 20.0, 10.0]  # r=2
        truth = -1.5
        c = 0.05
        k = [truth + c * x ** 2 for x in h]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.observed_order == pytest.approx(2.0, abs=1e-6)
        assert rep.extrapolated_value == pytest.approx(truth, abs=1e-6)

    def test_fractional_order(self):
        h = [27.0, 9.0, 3.0]  # uniform r=3
        truth = 100.0
        c = 1.2
        p_true = 1.5
        k = [truth + c * x ** p_true for x in h]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.observed_order == pytest.approx(p_true, abs=1e-5)
        assert rep.extrapolated_value == pytest.approx(truth, abs=1e-3)

    def test_assumed_order_overrides_three_point_fit(self):
        # Force p=1 even with 3 grids → uses two-point branch on finest pair
        h = [40.0, 20.0, 10.0]
        truth = 0.0
        c = 0.05
        k = [truth + c * x ** 2 for x in h]  # actually p=2
        rep = richardson_extrapolate(h, k, assumed_order=1.0, kpi_name="test")
        assert rep.fit_method == "two_point_assumed_order"
        assert rep.observed_order == 1.0
        # Wrong assumed order → won't recover truth exactly
        assert rep.extrapolated_value != pytest.approx(truth, abs=0.01)


class TestNonUniformRefinement:
    """Non-uniform refinement still recovers p via Brent's method."""

    def test_non_uniform_first_order(self):
        h = [30.0, 10.0, 4.0]  # non-uniform: r1=3, r2=2.5
        truth = 7.0
        slope = 0.5
        k = [truth + slope * x for x in h]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.observed_order == pytest.approx(1.0, abs=1e-5)
        assert rep.extrapolated_value == pytest.approx(truth, abs=1e-5)
        assert rep.fit_method == "nonuniform_observed_order"

    def test_non_uniform_second_order(self):
        h = [25.0, 10.0, 2.0]
        truth = -2.0
        c = 0.1
        k = [truth + c * x ** 2 for x in h]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.observed_order == pytest.approx(2.0, abs=1e-4)
        assert rep.extrapolated_value == pytest.approx(truth, abs=1e-3)


# ── Degenerate / edge cases ───────────────────────────────────────────────


class TestDegenerateCases:
    """Edge cases must not crash and must produce informative reports."""

    def test_constant_kpi(self):
        h = [30.0, 15.0, 5.0]
        k = [3.14, 3.14, 3.14]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.extrapolated_value == 3.14
        assert rep.error_band == 0.0
        assert rep.fit_method == "degenerate_constant_kpi"
        assert math.isnan(rep.observed_order)
        assert rep.monotone_convergence is True

    def test_non_monotone_three_grids_falls_back(self):
        # Non-monotone: KPI oscillates as h decreases
        h = [30.0, 15.0, 5.0]
        k = [10.0, 8.0, 9.0]  # delta_coarse=-2, delta_fine=+1 → opposite signs
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.fit_method == "fallback_assumed_order_1"
        assert math.isnan(rep.observed_order)
        assert rep.monotone_convergence is False

    def test_partially_stalled_three_grids_falls_back(self):
        # One delta is zero → can't fit p; falls back
        h = [30.0, 15.0, 5.0]
        k = [10.0, 10.0, 9.5]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.fit_method == "fallback_assumed_order_1"
        assert math.isnan(rep.observed_order)

    def test_zero_extrapolated_relative_inf(self):
        # KPI∞ = 0 with nonzero error band → relative_error reports inf
        h = [10.0, 5.0]
        k = [3.0, 1.5]  # extrap = (2*1.5 - 3)/1 = 0
        rep = richardson_extrapolate(h, k, assumed_order=1.0, kpi_name="test")
        assert rep.extrapolated_value == 0.0
        assert math.isinf(rep.relative_error)

    def test_zero_extrapolated_zero_band_relative_zero(self):
        h = [10.0, 5.0]
        k = [0.0, 0.0]
        rep = richardson_extrapolate(h, k, assumed_order=1.0, kpi_name="test")
        assert rep.extrapolated_value == 0.0
        assert rep.error_band == 0.0
        assert rep.relative_error == 0.0


class TestInputValidation:
    """Bad inputs raise informative ValueError."""

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="must align"):
            richardson_extrapolate([10.0, 5.0, 1.0], [3.0, 2.0])

    def test_too_few_grids(self):
        with pytest.raises(ValueError, match="need >= 2"):
            richardson_extrapolate([5.0], [3.0])

    def test_non_finite_tick_rates(self):
        with pytest.raises(ValueError, match="finite and positive"):
            richardson_extrapolate([10.0, float("nan")], [3.0, 2.0])

    def test_zero_tick_rate(self):
        with pytest.raises(ValueError, match="finite and positive"):
            richardson_extrapolate([10.0, 0.0], [3.0, 2.0])

    def test_negative_tick_rate(self):
        with pytest.raises(ValueError, match="finite and positive"):
            richardson_extrapolate([10.0, -5.0], [3.0, 2.0])

    def test_non_finite_kpi(self):
        with pytest.raises(ValueError, match="finite"):
            richardson_extrapolate([10.0, 5.0], [3.0, float("inf")])

    def test_duplicate_tick_rates(self):
        with pytest.raises(ValueError, match="distinct"):
            richardson_extrapolate([10.0, 10.0], [3.0, 3.5])


# ── Bundle sweep ──────────────────────────────────────────────────────────


def _bundle(tick_minutes: float, *, tdis: float, ener: float, peak: float,
            cold: float, warm: float, sp_ch: int, n_ticks: int) -> KpiBundle:
    return KpiBundle(
        n_ticks=n_ticks,
        tick_minutes=tick_minutes,
        tdis_tot=tdis,
        cold_time_h=cold,
        warm_time_h=warm,
        ener_tot=ener,
        peak_kw=peak,
        settling_time_h=None,
        setpoint_changes=sp_ch,
    )


class TestBundleSweep:
    """``kpi_richardson_sweep`` walks every default KPI."""

    def test_sweep_extrapolates_each_default_kpi(self):
        bundles = {
            30.0: _bundle(30.0, tdis=2.0, ener=6.0, peak=0.30,
                          cold=0.5, warm=0.0, sp_ch=20, n_ticks=144),
            15.0: _bundle(15.0, tdis=1.4, ener=5.7, peak=0.27,
                          cold=0.3, warm=0.0, sp_ch=24, n_ticks=288),
            5.0:  _bundle(5.0,  tdis=1.1, ener=5.5, peak=0.25,
                          cold=0.2, warm=0.0, sp_ch=30, n_ticks=864),
        }
        reports = kpi_richardson_sweep(bundles)
        assert set(reports.keys()) == set(DEFAULT_RICHARDSON_KPIS)
        for name in DEFAULT_RICHARDSON_KPIS:
            assert isinstance(reports[name], RichardsonReport)
            assert reports[name].kpi_name == name

    def test_sweep_custom_kpi_names(self):
        bundles = {
            10.0: _bundle(10.0, tdis=1.0, ener=2.0, peak=0.1,
                          cold=0.0, warm=0.0, sp_ch=10, n_ticks=144),
            5.0:  _bundle(5.0,  tdis=0.8, ener=1.9, peak=0.09,
                          cold=0.0, warm=0.0, sp_ch=12, n_ticks=288),
        }
        reports = kpi_richardson_sweep(
            bundles, kpi_names=("tdis_tot",), assumed_order=1.0,
        )
        assert list(reports.keys()) == ["tdis_tot"]

    def test_sweep_too_few_grids(self):
        bundles = {
            15.0: _bundle(15.0, tdis=1.0, ener=2.0, peak=0.1,
                          cold=0.0, warm=0.0, sp_ch=10, n_ticks=288),
        }
        with pytest.raises(ValueError, match="need >= 2 tick rates"):
            kpi_richardson_sweep(bundles)

    def test_sweep_missing_kpi_raises(self):
        bundles = {
            10.0: _bundle(10.0, tdis=1.0, ener=2.0, peak=0.1,
                          cold=0.0, warm=0.0, sp_ch=10, n_ticks=144),
            5.0:  _bundle(5.0,  tdis=0.8, ener=1.9, peak=0.09,
                          cold=0.0, warm=0.0, sp_ch=12, n_ticks=288),
        }
        with pytest.raises(ValueError, match="missing or None"):
            kpi_richardson_sweep(bundles, kpi_names=("does_not_exist",))


# ── Reporting helper ──────────────────────────────────────────────────────


class TestRegimeClassifier:
    """Asymptotic-regime classification for Roache GCI safety-factor selection."""

    def test_clean_asymptotic_data_in_regime(self):
        # KPI(h) = 5 + 0.5·h^1.5 — clean asymptotic
        h = [40.0, 20.0, 10.0]
        truth = 5.0
        c = 0.5
        p_true = 1.5
        k = [truth + c * x ** p_true for x in h]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.in_asymptotic_regime is True
        assert rep.safety_factor == ROACHE_FS_ASYMPTOTIC
        # GCI bounds the actual error generously
        assert rep.gci >= rep.error_band

    def test_non_monotone_not_in_regime(self):
        h = [30.0, 15.0, 5.0]
        k = [10.0, 8.0, 9.0]  # sign-flip in deltas
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.in_asymptotic_regime is False
        assert rep.safety_factor == ROACHE_FS_NON_ASYMPTOTIC

    def test_order_outside_physical_range_not_in_regime(self):
        # Construct a sequence with observed_order > REGIME_ORDER_MAX
        # (very steep convergence dominated by the coarsest grid).
        h = [30.0, 15.0, 5.0]
        k = [50.0, 10.0, 9.5]  # d_coarse=-40, d_fine=-0.5, ratio=80
        rep = richardson_extrapolate(h, k, kpi_name="test")
        # Observed order is well above REGIME_ORDER_MAX (4.0) — order
        # this large indicates a fit dominated by the coarsest grid,
        # not a true power-law regime.
        assert rep.observed_order > REGIME_ORDER_MAX
        assert rep.in_asymptotic_regime is False
        assert rep.safety_factor == ROACHE_FS_NON_ASYMPTOTIC

    def test_two_grid_assumed_order_not_in_regime(self):
        # Only 2 grids → fit_method is "two_point_assumed_order"; the
        # fitted p is not data-derived so we can't claim asymptotic regime.
        # Roache 1998 §5.5 explicitly recommends Fs=3.0 for 2-grid studies.
        h = [30.0, 15.0]
        k = [11.0, 8.0]
        rep = richardson_extrapolate(h, k, assumed_order=1.0, kpi_name="test")
        assert rep.in_asymptotic_regime is False
        assert rep.safety_factor == ROACHE_FS_NON_ASYMPTOTIC

    def test_constant_kpi_not_in_regime(self):
        h = [30.0, 15.0, 5.0]
        k = [3.14, 3.14, 3.14]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        # Degenerate; not asymptotic in the Roache sense (no convergence
        # to extract from). Fs=3.0 conservatively.
        assert rep.in_asymptotic_regime is False
        assert rep.safety_factor == ROACHE_FS_NON_ASYMPTOTIC
        assert rep.gci == 0.0  # but error band is genuinely zero

    def test_gci_uses_correct_safety_factor(self):
        # Asymptotic case → Fs = 1.25
        h = [40.0, 20.0, 10.0]
        c = 0.5
        k = [5.0 + c * x ** 1.5 for x in h]
        rep = richardson_extrapolate(h, k, kpi_name="test")
        assert rep.gci == pytest.approx(rep.error_band * ROACHE_FS_ASYMPTOTIC, abs=1e-9)

    def test_regime_constants_sensible(self):
        # Sanity: regime thresholds bracket the typical first/second-order
        # range expected for reasonable convergence schemes.
        assert 0 < REGIME_ORDER_MIN < 1.0 < 2.0 < REGIME_ORDER_MAX
        # Roache recommends 1.25 for asymptotic, 3.0 for non-asymptotic
        assert ROACHE_FS_ASYMPTOTIC == 1.25
        assert ROACHE_FS_NON_ASYMPTOTIC == 3.0


class TestTickRateSpread:
    """``tick_rate_spread`` returns max - min across the sweep."""

    def test_spread_returns_max_minus_min(self):
        rep = richardson_extrapolate(
            [30.0, 15.0, 5.0], [10.0, 7.0, 5.5], kpi_name="test"
        )
        assert tick_rate_spread(rep) == pytest.approx(4.5, abs=1e-9)

    def test_spread_zero_for_constant(self):
        rep = richardson_extrapolate(
            [30.0, 15.0, 5.0], [3.0, 3.0, 3.0], kpi_name="test"
        )
        assert tick_rate_spread(rep) == 0.0

    def test_spread_works_for_non_monotone(self):
        rep = richardson_extrapolate(
            [30.0, 15.0, 5.0], [5.0, 8.0, 6.0], kpi_name="test"
        )
        assert tick_rate_spread(rep) == 3.0


class TestFormatTable:
    """Smoke test on the reporting helper."""

    def test_format_runs_without_crashing(self):
        bundles = {
            30.0: _bundle(30.0, tdis=2.0, ener=6.0, peak=0.30,
                          cold=0.5, warm=0.0, sp_ch=20, n_ticks=144),
            15.0: _bundle(15.0, tdis=1.4, ener=5.7, peak=0.27,
                          cold=0.3, warm=0.0, sp_ch=24, n_ticks=288),
            5.0:  _bundle(5.0,  tdis=1.1, ener=5.5, peak=0.25,
                          cold=0.2, warm=0.0, sp_ch=30, n_ticks=864),
        }
        reports = kpi_richardson_sweep(bundles)
        text = format_richardson_table(reports)
        assert "tdis_tot" in text
        assert "extrap" in text
        assert "method" in text
        # nan and inf should not crash the formatter
        assert "p" in text or "method" in text
