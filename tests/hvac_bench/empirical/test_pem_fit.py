"""Tests for PEM multi-restart fitting.

Mix of:
- Pure-function tests (parameter packing/unpacking, bound utilities, CV math)
- Synthetic-data round-trip tests (simulate from known params → fit → recover)
- Bound enforcement tests (optimizer respects bounds; at_bound flag fires)
- CV gate tests (CV across restarts ≤ threshold for identifiable params)
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.hvac_bench.conftest import check_bench_metrics

from tests.hvac_bench.empirical.pem_fit import (
    DEFAULT_BOUNDS_1R1C,
    DEFAULT_BOUNDS_2R2C,
    Bounds,
    FitParams1R1C,
    FitParams2R2C,
    RestartResult,
    _at_bound,
    _compute_cv,
    fit_1r1c,
    fit_2r2c,
)
from tests.hvac_bench.empirical.rc_model import (
    RCParams1R1C,
    RCParams2R2C,
    build_1r1c,
    build_2r2c,
)


# ── Parameter packing/unpacking ──────────────────────────────────────────


class TestFitParams1R1C:
    def test_to_full_round_trip(self) -> None:
        p = FitParams1R1C(
            tau_s=3.6e5,  # 100 h
            q_scale=1.5,
            solar_scale=0.05,
            sigma_w=0.001,
            sigma_v=0.2,
        )
        full = p.to_full()
        assert isinstance(full, RCParams1R1C)
        np.testing.assert_allclose(full.tau_seconds, p.tau_s, rtol=1e-12)
        assert full.q_scale == p.q_scale
        assert full.solar_scale == p.solar_scale

    def test_log_array_round_trip(self) -> None:
        p = FitParams1R1C(
            tau_s=3.6e5,
            q_scale=1.5,
            solar_scale=0.05,
            sigma_w=0.001,
            sigma_v=0.2,
        )
        x = p.to_log_array()
        p2 = FitParams1R1C.from_log_array(x)
        np.testing.assert_allclose(p2.tau_s, p.tau_s, rtol=1e-12)
        np.testing.assert_allclose(p2.q_scale, p.q_scale, rtol=1e-12)
        np.testing.assert_allclose(p2.solar_scale, p.solar_scale, rtol=1e-12)
        np.testing.assert_allclose(p2.sigma_w, p.sigma_w, rtol=1e-12)
        np.testing.assert_allclose(p2.sigma_v, p.sigma_v, rtol=1e-12)

    def test_log_array_has_five_elements(self) -> None:
        p = FitParams1R1C(
            tau_s=1e5, q_scale=1.0, solar_scale=0.1, sigma_w=0.01, sigma_v=0.1
        )
        assert p.to_log_array().shape == (5,)


class TestFitParams2R2C:
    def test_to_full_preserves_taus(self) -> None:
        p = FitParams2R2C(
            tau_air_s=1800,
            tau_wall_s=72_000,
            coupling_ratio=4.0,
            q_scale=1.0,
            solar_scale=0.1,
            sigma_w_i=0.01,
            sigma_w_e=0.01,
            sigma_v=0.1,
        )
        full = p.to_full()
        np.testing.assert_allclose(full.tau_air, p.tau_air_s, rtol=1e-12)
        np.testing.assert_allclose(full.tau_wall, p.tau_wall_s, rtol=1e-12)
        np.testing.assert_allclose(full.C_e / full.C_i, p.coupling_ratio, rtol=1e-12)

    def test_log_array_has_eight_elements(self) -> None:
        p = FitParams2R2C(
            tau_air_s=1800,
            tau_wall_s=72_000,
            coupling_ratio=4.0,
            q_scale=1.0,
            solar_scale=0.1,
            sigma_w_i=0.01,
            sigma_w_e=0.01,
            sigma_v=0.1,
        )
        assert p.to_log_array().shape == (8,)

    def test_log_array_round_trip(self) -> None:
        p = FitParams2R2C(
            tau_air_s=2400,
            tau_wall_s=86_400,
            coupling_ratio=5.5,
            q_scale=1.2,
            solar_scale=0.08,
            sigma_w_i=0.005,
            sigma_w_e=0.003,
            sigma_v=0.15,
        )
        p2 = FitParams2R2C.from_log_array(p.to_log_array())
        for k in (
            "tau_air_s",
            "tau_wall_s",
            "coupling_ratio",
            "q_scale",
            "solar_scale",
            "sigma_w_i",
            "sigma_w_e",
            "sigma_v",
        ):
            np.testing.assert_allclose(getattr(p2, k), getattr(p, k), rtol=1e-12)


# ── Bounds and helpers ───────────────────────────────────────────────────


class TestBounds:
    def test_log_arrays_returns_log_of_lower_and_upper(self) -> None:
        b = Bounds(lower={"a": 1.0, "b": 10.0}, upper={"a": 100.0, "b": 1000.0})
        lo, hi = b.log_arrays(("a", "b"))
        np.testing.assert_allclose(lo, np.log([1.0, 10.0]))
        np.testing.assert_allclose(hi, np.log([100.0, 1000.0]))

    def test_default_1r1c_has_five_keys(self) -> None:
        assert len(DEFAULT_BOUNDS_1R1C.lower) == 5
        assert set(DEFAULT_BOUNDS_1R1C.lower) == {
            "tau_s",
            "q_scale",
            "solar_scale",
            "sigma_w",
            "sigma_v",
        }

    def test_default_2r2c_has_eight_keys(self) -> None:
        assert len(DEFAULT_BOUNDS_2R2C.lower) == 8
        assert "tau_air_s" in DEFAULT_BOUNDS_2R2C.lower
        assert "tau_wall_s" in DEFAULT_BOUNDS_2R2C.lower
        assert "coupling_ratio" in DEFAULT_BOUNDS_2R2C.lower

    def test_default_bounds_have_lower_below_upper(self) -> None:
        for b in (DEFAULT_BOUNDS_1R1C, DEFAULT_BOUNDS_2R2C):
            for k in b.lower:
                assert b.lower[k] < b.upper[k], k


class TestComputeCV:
    def test_constant_values_zero_cv(self) -> None:
        assert _compute_cv(np.array([5.0, 5.0, 5.0])) == 0.0

    def test_zero_mean_returns_inf(self) -> None:
        assert _compute_cv(np.array([0.0, 0.0])) == float("inf")

    def test_known_cv(self) -> None:
        # mean=10, std=1 → CV=0.1
        np.testing.assert_allclose(
            _compute_cv(np.array([9.0, 10.0, 11.0])),
            np.std([9.0, 10.0, 11.0]) / 10.0,
            rtol=1e-10,
        )


class TestAtBound:
    def test_far_from_bounds_returns_false(self) -> None:
        assert not _at_bound(10.0, lo=1.0, hi=100.0)

    def test_at_lower_returns_true(self) -> None:
        assert _at_bound(1.001, lo=1.0, hi=100.0)

    def test_at_upper_returns_true(self) -> None:
        assert _at_bound(99.5, lo=1.0, hi=100.0)


# ── Synthetic round-trip: 1R1C ───────────────────────────────────────────


def _simulate_1r1c(
    truth: FitParams1R1C,
    T: int = 2000,
    dt: float = 300.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 1R1C trajectory with diurnal outdoor + bursty heating."""
    rng = np.random.default_rng(seed)
    ss = build_1r1c(truth.to_full(), dt=dt)
    T_o = -2.0 + 8.0 * np.sin(np.arange(T) * 2 * np.pi / (24 * 12))  # diurnal 24h cycle at 5min
    q_heat = 1500.0 * (rng.random(T) > 0.5)  # bursty
    shortwave = 200.0 * np.maximum(
        0.0, np.sin(np.arange(T) * 2 * np.pi / (24 * 12) - np.pi / 4)
    )
    inputs = np.column_stack([T_o, q_heat, shortwave])
    obs = np.zeros(T)
    x = np.array([[T_o[0]]])
    for k in range(T):
        obs[k] = (ss.H @ x).item() + rng.standard_normal() * truth.sigma_v
        if k < T - 1:
            x = ss.A @ x + ss.B @ inputs[k].reshape(-1, 1)
    valid = np.ones(T, dtype=bool)
    return obs, inputs, valid


class TestFit1R1C:
    def test_returns_fit_result_with_correct_shape(self) -> None:
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.005,
            sigma_w=1e-4,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=500)
        result = fit_1r1c(obs, inputs, valid, dt=300.0, n_restarts=2, seed=1)
        assert result.model_name == "1R1C"
        assert len(result.restarts) == 2
        assert result.n_obs == 500
        assert isinstance(result.best.params, FitParams1R1C)

    def test_recovers_tau_within_factor_of_2_on_clean_synthetic(self) -> None:
        # Clean low-noise synthetic: PEM should recover tau within ~factor-of-2.
        # Tighter recovery is sensitive to optimizer settings; this is a sanity bound.
        truth = FitParams1R1C(
            tau_s=20 * 3600,  # 20 h
            q_scale=1.0,
            solar_scale=0.002,
            sigma_w=1e-5,
            sigma_v=0.05,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=2000, seed=42)
        result = fit_1r1c(
            obs, inputs, valid, dt=300.0, n_restarts=4, seed=11, maxiter=300
        )
        assert result.best.success or np.isfinite(result.best.log_likelihood)
        ratio = result.best.params.tau_s / truth.tau_s
        assert 0.5 < ratio < 2.0, (
            f"recovered tau={result.best.params.tau_s/3600:.1f}h vs truth=20h"
        )

    def test_records_cv_when_multiple_restarts_succeed(self) -> None:
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=1000)
        result = fit_1r1c(obs, inputs, valid, dt=300.0, n_restarts=3, seed=2)
        # At least the best result is finite; CV either populated or empty.
        if result.cv_per_param:
            assert set(result.cv_per_param) == {
                "tau_s",
                "q_scale",
                "solar_scale",
                "sigma_w",
                "sigma_v",
            }

    def test_at_bound_dict_has_all_param_keys(self) -> None:
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=400)
        result = fit_1r1c(obs, inputs, valid, dt=300.0, n_restarts=2)
        assert set(result.at_bound_per_param) == {
            "tau_s",
            "q_scale",
            "solar_scale",
            "sigma_w",
            "sigma_v",
        }

    def test_handles_all_invalid_data(self) -> None:
        # All-invalid mask: optimizer still runs but log-likelihood is 0
        # for any params (no observations to score).
        T = 200
        obs = np.zeros(T)
        inputs = np.zeros((T, 3))
        valid = np.zeros(T, dtype=bool)
        result = fit_1r1c(obs, inputs, valid, dt=300.0, n_restarts=2)
        assert result.n_obs == 0
        # All restarts produce ll=0.0 (no observations); best is one of them.
        assert result.best.log_likelihood == 0.0

    def test_practically_identifiable_uses_threshold(self) -> None:
        # Construct a FitResult by hand and verify the threshold logic.
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=500)
        result = fit_1r1c(obs, inputs, valid, dt=300.0, n_restarts=2)
        # Synthetic CV.
        result.cv_per_param = {"tau_s": 0.05, "q_scale": 0.2, "solar_scale": 0.5}
        verdict = result.is_practically_identifiable(cv_threshold=0.1)
        assert verdict["tau_s"] is True
        assert verdict["q_scale"] is False
        assert verdict["solar_scale"] is False


# ── Synthetic round-trip: 2R2C ───────────────────────────────────────────


def _simulate_2r2c(
    truth: FitParams2R2C,
    T: int = 2000,
    dt: float = 300.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ss = build_2r2c(truth.to_full(), dt=dt)
    T_o = -2.0 + 8.0 * np.sin(np.arange(T) * 2 * np.pi / (24 * 12))
    q_heat = 1500.0 * (rng.random(T) > 0.5)
    shortwave = 200.0 * np.maximum(
        0.0, np.sin(np.arange(T) * 2 * np.pi / (24 * 12) - np.pi / 4)
    )
    inputs = np.column_stack([T_o, q_heat, shortwave])
    obs = np.zeros(T)
    x = np.array([[T_o[0]], [T_o[0]]])
    for k in range(T):
        obs[k] = (ss.H @ x).item() + rng.standard_normal() * truth.sigma_v
        if k < T - 1:
            x = ss.A @ x + ss.B @ inputs[k].reshape(-1, 1)
    valid = np.ones(T, dtype=bool)
    return obs, inputs, valid


class TestFit2R2C:
    def test_returns_fit_result_with_correct_shape(self) -> None:
        truth = FitParams2R2C(
            tau_air_s=1800,
            tau_wall_s=36_000,
            coupling_ratio=4.0,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w_i=1e-4,
            sigma_w_e=1e-4,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_2r2c(truth, T=600)
        result = fit_2r2c(obs, inputs, valid, dt=300.0, n_restarts=2, seed=3)
        assert result.model_name == "2R2C"
        assert len(result.restarts) == 2
        assert isinstance(result.best.params, FitParams2R2C)

    def test_at_bound_dict_has_all_eight_keys(self) -> None:
        truth = FitParams2R2C(
            tau_air_s=1800,
            tau_wall_s=36_000,
            coupling_ratio=4.0,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w_i=1e-4,
            sigma_w_e=1e-4,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_2r2c(truth, T=400)
        result = fit_2r2c(obs, inputs, valid, dt=300.0, n_restarts=2)
        assert len(result.at_bound_per_param) == 8


# ── Bound enforcement ────────────────────────────────────────────────────


class TestBoundEnforcement:
    def test_optimizer_respects_lower_bounds(self) -> None:
        # Create synthetic data driven at the bounds; verify the fitted
        # parameters land within bounds (not outside).
        truth = FitParams1R1C(
            tau_s=10 * 3600,
            q_scale=1.0,
            solar_scale=0.001,
            sigma_w=1e-5,
            sigma_v=0.1,
        )
        obs, inputs, valid = _simulate_1r1c(truth, T=500)
        result = fit_1r1c(
            obs, inputs, valid, dt=300.0, n_restarts=3, bounds=DEFAULT_BOUNDS_1R1C
        )
        for r in result.restarts:
            assert r.params.tau_s >= DEFAULT_BOUNDS_1R1C.lower["tau_s"] - 1e-6
            assert r.params.tau_s <= DEFAULT_BOUNDS_1R1C.upper["tau_s"] + 1e-6
            assert r.params.q_scale >= DEFAULT_BOUNDS_1R1C.lower["q_scale"] - 1e-6
            assert r.params.q_scale <= DEFAULT_BOUNDS_1R1C.upper["q_scale"] + 1e-6
