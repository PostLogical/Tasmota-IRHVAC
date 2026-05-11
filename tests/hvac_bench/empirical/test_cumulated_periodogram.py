"""Tests for cumulated periodogram (Bacher-Madsen 2011 / Leprince 2022)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.hvac_bench.conftest import check_bench_metrics

from tests.hvac_bench.empirical.cumulated_periodogram import (
    KS_CRITICAL_ALPHA,
    cumulated_periodogram,
)


class TestCumulatedPeriodogramShape:
    def test_returns_arrays_of_same_length(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.standard_normal(200)
        result = cumulated_periodogram(r)
        n_freq = result.cumulated_power.size
        assert result.frequencies.size == n_freq
        assert result.diagonal.size == n_freq
        assert result.upper_band.size == n_freq
        assert result.lower_band.size == n_freq

    def test_cumulated_starts_above_zero_ends_at_one(self, bench_metrics, num_regression) -> None:
        rng = np.random.default_rng(1)
        r = rng.standard_normal(500)
        result = cumulated_periodogram(r)
        assert result.cumulated_power[0] > 0
        np.testing.assert_allclose(result.cumulated_power[-1], 1.0, atol=1e-12)
        bench_metrics["cp_start"] = float(result.cumulated_power[0])
        bench_metrics["cp_end"] = float(result.cumulated_power[-1])
        check_bench_metrics(num_regression, bench_metrics)

    def test_diagonal_increases_to_one(self, bench_metrics, num_regression) -> None:
        rng = np.random.default_rng(2)
        r = rng.standard_normal(400)
        result = cumulated_periodogram(r)
        assert (np.diff(result.diagonal) > 0).all()
        np.testing.assert_allclose(result.diagonal[-1], 1.0, atol=1e-12)
        bench_metrics["diagonal_end"] = float(result.diagonal[-1])
        bench_metrics["diagonal_start"] = float(result.diagonal[0])
        check_bench_metrics(num_regression, bench_metrics)


class TestKSBand:
    def test_band_width_scales_inverse_sqrt_n(self, bench_metrics, num_regression) -> None:
        rng = np.random.default_rng(3)
        # Larger n → tighter band.
        r1 = rng.standard_normal(100)
        r2 = rng.standard_normal(10000)
        cp1 = cumulated_periodogram(r1)
        cp2 = cumulated_periodogram(r2)
        # Band width at any non-edge point: cp1's band > cp2's band
        band_width_1 = cp1.upper_band[10] - cp1.lower_band[10]
        band_width_2 = cp2.upper_band[10] - cp2.lower_band[10]
        assert band_width_1 > band_width_2
        bench_metrics["band_width_n100"] = float(band_width_1)
        bench_metrics["band_width_n10000"] = float(band_width_2)
        bench_metrics["ratio"] = float(band_width_1 / band_width_2)
        check_bench_metrics(num_regression, bench_metrics)

    def test_unknown_alpha_raises(self) -> None:
        rng = np.random.default_rng(4)
        r = rng.standard_normal(100)
        with pytest.raises(KeyError):
            cumulated_periodogram(r, ks_alpha=0.07)


class TestWhiteNoiseInBand:
    def test_white_noise_typically_inside_band_at_alpha_05(self, bench_metrics, num_regression) -> None:
        # Repeated draws of white Gaussian; at α=0.05, P(false reject) ≤ 5%.
        # 10 draws — at most one should fail. (Probabilistic; deterministic via seed.)
        rng = np.random.default_rng(2026)
        n_pass = 0
        for _ in range(10):
            r = rng.standard_normal(500)
            cp = cumulated_periodogram(r, ks_alpha=0.05)
            if cp.inside_band:
                n_pass += 1
        assert n_pass >= 8, f"only {n_pass}/10 inside band; suspicious"
        bench_metrics["n_pass"] = n_pass
        check_bench_metrics(num_regression, bench_metrics)

    def test_n_outside_zero_when_white(self, bench_metrics, num_regression) -> None:
        rng = np.random.default_rng(11)
        r = rng.standard_normal(2000)
        cp = cumulated_periodogram(r, ks_alpha=0.05)
        assert cp.inside_band
        assert cp.n_outside == 0
        assert cp.max_excess == 0.0
        assert cp.nCPBES == 0.0
        bench_metrics["n_outside"] = int(cp.n_outside)
        bench_metrics["max_excess"] = float(cp.max_excess)
        bench_metrics["nCPBES"] = float(cp.nCPBES)
        check_bench_metrics(num_regression, bench_metrics)


class TestColoredNoiseFails:
    def test_low_pass_filtered_noise_fails_band(self, bench_metrics, num_regression) -> None:
        # Strongly autocorrelated (low-pass) signal — CP should bow toward
        # one extreme and exit the band.
        rng = np.random.default_rng(7)
        innov = rng.standard_normal(500)
        # AR(1) with phi=0.9 — strongly autocorrelated
        ar1 = np.zeros_like(innov)
        ar1[0] = innov[0]
        for k in range(1, len(innov)):
            ar1[k] = 0.9 * ar1[k - 1] + innov[k]
        cp = cumulated_periodogram(ar1, ks_alpha=0.05)
        # Strongly colored signal should exit the band.
        assert not cp.inside_band
        assert cp.n_outside > 0
        assert cp.nCPBES > 0
        bench_metrics["n_outside"] = int(cp.n_outside)
        bench_metrics["max_excess"] = float(cp.max_excess)
        bench_metrics["nCPBES"] = float(cp.nCPBES)
        check_bench_metrics(num_regression, bench_metrics)


class TestInputValidation:
    def test_too_few_residuals_raises(self) -> None:
        with pytest.raises(ValueError, match="Need ≥4"):
            cumulated_periodogram(np.array([1.0, 2.0]))

    def test_nan_raises(self) -> None:
        r = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.raises(ValueError, match="NaN"):
            cumulated_periodogram(r)

    def test_zero_variance_raises(self) -> None:
        r = np.full(50, 3.7)
        with pytest.raises(ValueError, match="zero variance"):
            cumulated_periodogram(r)


class TestKSCriticalValues:
    def test_alpha_05_has_correct_kolmogorov_value(self) -> None:
        # Per Massey 1951 / standard tables, KS one-sample 2-sided at α=0.05
        # is c=1.36 to 3 sig figs.
        np.testing.assert_allclose(KS_CRITICAL_ALPHA[0.05], 1.36, atol=1e-2)

    def test_alpha_01_tighter_than_05(self) -> None:
        assert KS_CRITICAL_ALPHA[0.01] > KS_CRITICAL_ALPHA[0.05]
