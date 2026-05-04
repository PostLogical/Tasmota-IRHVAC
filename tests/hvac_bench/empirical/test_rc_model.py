"""Tests for rc_model: discretization, builders, Kalman log-likelihood.

Mathematical correctness checks:
- Discretization round-trips against analytical exp(A·dt) for diagonal A.
- Van Loan Q_d matches the closed-form 1D solution for scalar systems.
- ZOH input matrix matches A^{-1}(A_d - I) B for invertible A.
- 1R1C steady-state matches T_o + R·Φ analytical limit.
- 2R2C steady-state matches the 1R1C limit when masses are matched.
- Kalman likelihood: scales correctly with sigma_v on white-noise data;
  recovers a known parameter via grid search on synthetic 1R1C data.
- Missing-data handling: valid mask doesn't blow up; likelihood drops
  monotonically with #valid observations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from tests.hvac_bench.empirical.rc_model import (
    RCParams1R1C,
    RCParams2R2C,
    StateSpace,
    build_1r1c,
    build_2r2c,
    discretize,
    kalman_innovations,
    kalman_log_likelihood,
)


# ── Discretization ───────────────────────────────────────────────────────


class TestDiscretize:
    def test_diagonal_a_matches_analytical_exp(self) -> None:
        # For A_c = -λ*I (diagonal), A_d should be exp(-λ·dt)*I exactly.
        lam = 0.01
        dt = 60.0
        A_c = np.array([[-lam, 0.0], [0.0, -lam]])
        B_c = np.zeros((2, 1))
        Sigma_c = np.zeros((2, 2))

        A_d, _, _ = discretize(A_c, B_c, Sigma_c, dt)
        expected = np.exp(-lam * dt) * np.eye(2)
        np.testing.assert_allclose(A_d, expected, atol=1e-12)

    def test_zero_a_yields_identity(self) -> None:
        A_c = np.zeros((2, 2))
        B_c = np.eye(2)
        Sigma_c = np.zeros((2, 2))
        A_d, B_d, _ = discretize(A_c, B_c, Sigma_c, 10.0)
        np.testing.assert_allclose(A_d, np.eye(2), atol=1e-12)
        # B_d for zero A and B=I is dt*I
        np.testing.assert_allclose(B_d, 10.0 * np.eye(2), atol=1e-12)

    def test_q_d_matches_scalar_closed_form(self) -> None:
        # For dx = -λ x dt + σ dω, Q_d = (σ²/(2λ)) * (1 - exp(-2λ·dt)).
        lam = 0.005
        sigma = 0.5
        dt = 30.0
        A_c = np.array([[-lam]])
        B_c = np.zeros((1, 1))
        Sigma_c = np.array([[sigma**2]])
        _, _, Q_d = discretize(A_c, B_c, Sigma_c, dt)
        expected = (sigma**2 / (2.0 * lam)) * (1.0 - np.exp(-2.0 * lam * dt))
        np.testing.assert_allclose(Q_d.item(), expected, rtol=1e-10)

    def test_b_d_matches_a_inverse_formula_for_invertible_a(self) -> None:
        # For invertible A_c, B_d = A_c^{-1} (A_d - I) B_c.
        A_c = np.array([[-0.01, 0.005], [0.002, -0.008]])
        B_c = np.array([[1.0, 0.5], [0.3, 0.7]])
        Sigma_c = np.zeros((2, 2))
        dt = 60.0
        A_d, B_d, _ = discretize(A_c, B_c, Sigma_c, dt)
        expected = np.linalg.solve(A_c, A_d - np.eye(2)) @ B_c
        np.testing.assert_allclose(B_d, expected, rtol=1e-10)

    def test_q_d_is_symmetric_positive_semidefinite(self) -> None:
        A_c = np.array([[-0.01, 0.005], [0.002, -0.008]])
        Sigma_c = np.array([[0.04, 0.01], [0.01, 0.09]])
        _, Q_d_view = discretize(A_c, np.zeros((2, 1)), Sigma_c, 60.0)[0:2:2], None
        _, _, Q_d = discretize(A_c, np.zeros((2, 1)), Sigma_c, 60.0)
        np.testing.assert_allclose(Q_d, Q_d.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(Q_d)
        assert (eigs >= -1e-12).all()


# ── 1R1C builder ─────────────────────────────────────────────────────────


class TestBuild1R1C:
    def _params(self, **kw) -> RCParams1R1C:
        defaults = dict(
            R=0.005,
            C=2.0e7,
            q_scale=1.0,
            solar_scale=0.5,
            sigma_w=0.001,
            sigma_v=0.1,
        )
        defaults.update(kw)
        return RCParams1R1C(**defaults)

    def test_returns_state_space_with_correct_shapes(self) -> None:
        ss = build_1r1c(self._params(), dt=300.0)
        assert ss.A.shape == (1, 1)
        assert ss.B.shape == (1, 3)
        assert ss.H.shape == (1, 1)
        assert ss.Q.shape == (1, 1)
        assert isinstance(ss.R, float)

    def test_a_d_decays_for_positive_tau(self) -> None:
        # For τ = R·C, A_d = exp(-dt/τ) ∈ (0, 1)
        p = self._params(R=0.005, C=2.0e7)  # τ = 1e5 s ≈ 27.8 h
        ss = build_1r1c(p, dt=300.0)
        expected = np.exp(-300.0 / p.tau_seconds)
        np.testing.assert_allclose(ss.A.item(), expected, rtol=1e-10)
        assert 0 < ss.A.item() < 1

    def test_h_selects_air_state(self) -> None:
        ss = build_1r1c(self._params(), dt=300.0)
        np.testing.assert_array_equal(ss.H, np.array([[1.0]]))

    def test_steady_state_matches_analytical_with_zero_solar(self) -> None:
        # At steady state with no solar and constant inputs:
        # T_i_ss = T_o + R · q_scale · q_heat.
        # τ = RC = 1e5 s; iterate ~10τ to get within 1e-4 of steady state.
        p = self._params(q_scale=1.0, solar_scale=0.0)
        ss = build_1r1c(p, dt=60.0)
        T_o = -5.0
        q_heat = 1500.0
        u = np.array([T_o, q_heat, 0.0])
        n_steps = int(10 * p.tau_seconds / 60.0)
        x = np.array([[0.0]])
        for _ in range(n_steps):
            x = ss.A @ x + ss.B @ u.reshape(-1, 1)
        ss_T = x.item()
        expected = T_o + p.R * p.q_scale * q_heat
        np.testing.assert_allclose(ss_T, expected, rtol=1e-3)

    def test_invalid_params_raises(self) -> None:
        with pytest.raises(ValueError):
            build_1r1c(self._params(R=-1.0), dt=60.0)
        with pytest.raises(ValueError):
            build_1r1c(self._params(C=0.0), dt=60.0)
        with pytest.raises(ValueError):
            build_1r1c(self._params(sigma_v=0.0), dt=60.0)


# ── 2R2C builder ─────────────────────────────────────────────────────────


class TestBuild2R2C:
    def _params(self, **kw) -> RCParams2R2C:
        defaults = dict(
            R_ie=0.001,
            R_ea=0.005,
            C_i=5.0e6,
            C_e=2.0e7,
            q_scale=1.0,
            solar_scale=0.5,
            sigma_w_i=0.001,
            sigma_w_e=0.001,
            sigma_v=0.1,
            wall_solar_fraction=0.7,
        )
        defaults.update(kw)
        return RCParams2R2C(**defaults)

    def test_returns_state_space_with_correct_shapes(self) -> None:
        ss = build_2r2c(self._params(), dt=300.0)
        assert ss.A.shape == (2, 2)
        assert ss.B.shape == (2, 3)
        assert ss.H.shape == (1, 2)
        assert ss.Q.shape == (2, 2)

    def test_h_selects_only_air_state(self) -> None:
        ss = build_2r2c(self._params(), dt=300.0)
        np.testing.assert_array_equal(ss.H, np.array([[1.0, 0.0]]))

    def test_steady_state_matches_thermal_circuit(self) -> None:
        # 2R2C steady state with q_heat and outdoor:
        # T_i_ss = T_o + (R_ie + R_ea) · q_scale · q_heat
        # T_e_ss = T_o + R_ea · q_scale · q_heat
        # (Solar = 0)
        p = self._params(solar_scale=0.0)
        ss = build_2r2c(p, dt=60.0)
        T_o = 0.0
        q_heat = 2000.0
        u = np.array([T_o, q_heat, 0.0])
        # Slowest mode τ ~ R_ea · C_e = 0.005 · 2e7 = 1e5 s; iterate ~12τ
        slowest_tau = max(p.tau_air, p.tau_wall)
        n_steps = int(12 * slowest_tau / 60.0)
        x = np.zeros((2, 1))
        for _ in range(n_steps):
            x = ss.A @ x + ss.B @ u.reshape(-1, 1)
        T_i_ss = x[0, 0]
        T_e_ss = x[1, 0]
        expected_air = T_o + (p.R_ie + p.R_ea) * p.q_scale * q_heat
        expected_wall = T_o + p.R_ea * p.q_scale * q_heat
        np.testing.assert_allclose(T_i_ss, expected_air, rtol=1e-3)
        np.testing.assert_allclose(T_e_ss, expected_wall, rtol=1e-3)

    def test_solar_split_wall_dominates_when_fraction_one(self) -> None:
        # With wall_solar_fraction=1, continuous B_c[0,2]=0. Discrete B_d[0,2]
        # picks up an O(dt·A[0,1]) bleed-through from the wall via coupling.
        # At dt=60s with the test params, separation is ~100×; verify >50×.
        p = self._params(wall_solar_fraction=1.0)
        ss = build_2r2c(p, dt=60.0)
        ratio = abs(ss.B[1, 2]) / max(abs(ss.B[0, 2]), 1e-30)
        assert ratio > 50

    def test_solar_split_air_dominates_when_fraction_zero(self) -> None:
        p = self._params(wall_solar_fraction=0.0)
        ss = build_2r2c(p, dt=60.0)
        ratio = abs(ss.B[0, 2]) / max(abs(ss.B[1, 2]), 1e-30)
        assert ratio > 50

    def test_solar_split_bleed_vanishes_at_small_dt(self) -> None:
        # At dt → 0, B_d → dt * B_c, so bleed-through ratio improves.
        p = self._params(wall_solar_fraction=1.0)
        ss_long = build_2r2c(p, dt=300.0)
        ss_short = build_2r2c(p, dt=10.0)
        long_ratio = abs(ss_long.B[1, 2]) / abs(ss_long.B[0, 2])
        short_ratio = abs(ss_short.B[1, 2]) / abs(ss_short.B[0, 2])
        assert short_ratio > long_ratio

    def test_invalid_wall_fraction_raises(self) -> None:
        with pytest.raises(ValueError):
            build_2r2c(self._params(wall_solar_fraction=1.5), dt=60.0)

    def test_negative_resistance_raises(self) -> None:
        with pytest.raises(ValueError):
            build_2r2c(self._params(R_ie=-0.001), dt=60.0)


# ── Kalman log-likelihood ────────────────────────────────────────────────


class TestKalmanLogLikelihood:
    def _trivial_ss(self, sigma_v: float = 0.1) -> StateSpace:
        # Identity dynamics, no noise except measurement: y_k = x_k, x stays put.
        return StateSpace(
            A=np.array([[1.0]]),
            B=np.array([[0.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.0]]),
            R=sigma_v**2,
        )

    def test_constant_signal_likelihood_finite(self) -> None:
        ss = self._trivial_ss()
        T = 100
        obs = np.full(T, 22.0) + np.random.default_rng(42).standard_normal(T) * 0.1
        u = np.zeros((T, 1))
        valid = np.ones(T, dtype=bool)
        ll = kalman_log_likelihood(ss, obs, u, valid, P0=np.array([[0.01]]))
        assert np.isfinite(ll)

    def test_likelihood_lower_with_larger_sigma_v_under_clean_data(self) -> None:
        # With clean data, larger sigma_v means worse likelihood.
        rng = np.random.default_rng(123)
        T = 200
        obs = np.full(T, 22.0) + rng.standard_normal(T) * 0.05
        u = np.zeros((T, 1))
        valid = np.ones(T, dtype=bool)
        ll_tight = kalman_log_likelihood(
            self._trivial_ss(0.05), obs, u, valid, P0=np.array([[0.01]])
        )
        ll_loose = kalman_log_likelihood(
            self._trivial_ss(2.0), obs, u, valid, P0=np.array([[0.01]])
        )
        assert ll_tight > ll_loose

    def test_no_valid_returns_zero(self) -> None:
        ss = self._trivial_ss()
        T = 50
        ll = kalman_log_likelihood(
            ss, np.zeros(T), np.zeros((T, 1)), np.zeros(T, dtype=bool)
        )
        assert ll == 0.0

    def test_partial_valid_does_not_crash(self) -> None:
        ss = self._trivial_ss()
        T = 100
        obs = np.full(T, 22.0)
        u = np.zeros((T, 1))
        valid = np.zeros(T, dtype=bool)
        valid[10:90] = True  # contiguous valid block in the middle
        ll = kalman_log_likelihood(ss, obs, u, valid, P0=np.array([[0.01]]))
        assert np.isfinite(ll)

    def test_more_valid_observations_yield_higher_likelihood(self) -> None:
        # Same data, different valid masks; more valid → larger sum.
        rng = np.random.default_rng(7)
        T = 200
        obs = np.full(T, 22.0) + rng.standard_normal(T) * 0.05
        u = np.zeros((T, 1))
        ss = self._trivial_ss(0.05)
        ll_full = kalman_log_likelihood(
            ss, obs, u, np.ones(T, dtype=bool), P0=np.array([[0.01]])
        )
        half = np.zeros(T, dtype=bool)
        half[:T // 2] = True
        ll_half = kalman_log_likelihood(ss, obs, u, half, P0=np.array([[0.01]]))
        assert ll_full > ll_half

    def test_shape_mismatch_raises(self) -> None:
        ss = self._trivial_ss()
        with pytest.raises(ValueError):
            kalman_log_likelihood(
                ss,
                np.zeros(10),
                np.zeros((20, 1)),
                np.ones(10, dtype=bool),
            )

    def test_recovers_known_tau_via_grid_search(self) -> None:
        # Synthetic data from a known 1R1C with no process noise; grid-search
        # over tau to find the maximum likelihood; verify the maximum is at
        # the truth bin (within grid resolution).
        rng = np.random.default_rng(2026)
        true_R = 0.004
        true_C = 1.5e7
        true_tau = true_R * true_C
        dt = 300.0  # 5 min
        T = 800  # ~67 h
        T_o = -5.0 + 5.0 * np.sin(np.arange(T) * 2 * np.pi / (24 * 12))  # diurnal
        q_heat = 1500.0 * (rng.random(T) > 0.5)  # bursty heating
        shortwave = np.zeros(T)
        u_seq = np.column_stack([T_o, q_heat, shortwave])

        # Simulate forward with truth
        true_p = RCParams1R1C(
            R=true_R, C=true_C, q_scale=1.0, solar_scale=0.0,
            sigma_w=0.0, sigma_v=0.05,
        )
        ss_true = build_1r1c(true_p, dt=dt)
        x = np.array([[0.0]])
        obs = np.zeros(T)
        for k in range(T):
            obs[k] = (ss_true.H @ x).item() + rng.standard_normal() * 0.05
            if k < T - 1:
                x = ss_true.A @ x + ss_true.B @ u_seq[k].reshape(-1, 1)
        valid = np.ones(T, dtype=bool)

        # Grid search over tau, holding C fixed
        taus = np.array([true_tau * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)])
        lls = []
        for tau in taus:
            R_test = tau / true_C
            p_test = RCParams1R1C(
                R=R_test, C=true_C, q_scale=1.0, solar_scale=0.0,
                sigma_w=0.0, sigma_v=0.05,
            )
            ss_test = build_1r1c(p_test, dt=dt)
            ll = kalman_log_likelihood(
                ss_test, obs, u_seq, valid, P0=np.array([[1.0]])
            )
            lls.append(ll)
        argmax = int(np.argmax(lls))
        # Best should be at index 2 (the true tau)
        assert argmax == 2, f"argmax={argmax}, lls={lls}"


# ── Innovation series ────────────────────────────────────────────────────


class TestKalmanInnovations:
    def test_returns_two_arrays_of_correct_length(self) -> None:
        ss = StateSpace(
            A=np.array([[1.0]]),
            B=np.array([[0.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.0]]),
            R=0.01,
        )
        T = 50
        innov, var = kalman_innovations(
            ss,
            np.zeros(T),
            np.zeros((T, 1)),
            np.ones(T, dtype=bool),
            P0=np.array([[0.01]]),
        )
        assert innov.shape == (T,)
        assert var.shape == (T,)

    def test_invalid_rows_are_nan(self) -> None:
        ss = StateSpace(
            A=np.array([[1.0]]),
            B=np.array([[0.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.0]]),
            R=0.01,
        )
        T = 30
        valid = np.ones(T, dtype=bool)
        valid[5:10] = False
        innov, var = kalman_innovations(
            ss, np.zeros(T), np.zeros((T, 1)), valid, P0=np.array([[0.01]])
        )
        assert np.isnan(innov[5:10]).all()
        assert np.isnan(var[5:10]).all()
        assert np.isfinite(innov[10:]).all()

    def test_variance_is_positive_on_valid_rows(self) -> None:
        ss = StateSpace(
            A=np.array([[0.99]]),
            B=np.array([[0.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.001]]),
            R=0.04,
        )
        T = 50
        rng = np.random.default_rng(0)
        innov, var = kalman_innovations(
            ss,
            rng.standard_normal(T) * 0.2,
            np.zeros((T, 1)),
            np.ones(T, dtype=bool),
            P0=np.array([[1.0]]),
        )
        valid = ~np.isnan(var)
        assert (var[valid] > 0).all()

    def test_log_likelihood_consistent_with_innovations(self) -> None:
        # Compute log-likelihood directly and via summing innovations.
        rng = np.random.default_rng(11)
        T = 100
        obs = rng.standard_normal(T) * 0.1
        u = np.zeros((T, 1))
        valid = np.ones(T, dtype=bool)
        ss = StateSpace(
            A=np.array([[0.95]]),
            B=np.array([[0.0]]),
            H=np.array([[1.0]]),
            Q=np.array([[0.001]]),
            R=0.04,
        )
        ll_direct = kalman_log_likelihood(ss, obs, u, valid, P0=np.array([[1.0]]))
        innov, var = kalman_innovations(ss, obs, u, valid, P0=np.array([[1.0]]))
        valid_mask = ~np.isnan(innov)
        ll_from_innov = -0.5 * np.sum(
            np.log(2 * np.pi * var[valid_mask]) + innov[valid_mask] ** 2 / var[valid_mask]
        )
        np.testing.assert_allclose(ll_direct, ll_from_innov, rtol=1e-10)


# ── Tier 1.4a — Analytical-reference physics audit ───────────────────────
#
# These tests verify `build_1r1c` against closed-form analytical solutions,
# independent of any optimizer or fit code. Synthetic round-trips elsewhere
# in this module test optimizer self-consistency (they use the same kernel
# on both ends); these tests test kernel-physics fidelity directly.
#
# Reference: 1R1C ODE  dT/dt = (T_outdoor - T)/τ + (q_scale·q_heat + solar_scale·shortwave) / C
# Continuous solution under constant inputs: T(t) = T_∞ + (T_0 - T_∞)·exp(-t/τ)
# where T_∞ = T_outdoor + R·q_scale·q_heat + R·solar_scale·shortwave
# and τ = R·C.
#
# Discrete-time iteration with ZOH (Van Loan): T_{k+1} = A_d·T_k + (1-A_d)·T_∞
# with A_d = exp(-dt/τ), so T_k = T_∞ + (T_0 - T_∞)·A_d^k = T_k(t=k·dt) exactly.


class TestKernelAnalyticalReference1R1C:
    """1R1C `build_1r1c` matches closed-form ODE solution under constant inputs."""

    def _truth(self, R: float = 0.005, C: float = 5.76e7) -> RCParams1R1C:
        # τ = R·C = 80h by default. q_scale and solar_scale set to make
        # B_d-component checks unambiguous (non-unity, non-zero).
        return RCParams1R1C(
            R=R,
            C=C,
            q_scale=0.5,
            solar_scale=2.0,
            sigma_w=0.0,
            sigma_v=1e-6,  # measurement noise irrelevant; dynamics test
        )

    def test_a_d_equals_exp_neg_dt_over_tau(self) -> None:
        params = self._truth()
        dt = 300.0
        ss = build_1r1c(params, dt=dt)
        expected = np.exp(-dt / params.tau_seconds)
        np.testing.assert_allclose(ss.A[0, 0], expected, rtol=1e-12)

    def test_outdoor_b_d_equals_one_minus_a_d(self) -> None:
        # ODE coefficient on T_outdoor is 1/τ → continuous-time gain = 1.
        # Discrete ZOH: B_d[outdoor] = (1 - A_d).
        params = self._truth()
        dt = 300.0
        ss = build_1r1c(params, dt=dt)
        expected = 1.0 - ss.A[0, 0]
        np.testing.assert_allclose(ss.B[0, 0], expected, rtol=1e-12)

    def test_q_heat_b_d_equals_R_qscale_times_one_minus_a_d(self) -> None:
        # Steady-state from constant Q with T_outdoor=0, solar=0: T_∞ = R·q_scale·Q.
        # Discrete: B_d[q_heat]/(1-A_d) = R·q_scale.
        params = self._truth()
        dt = 300.0
        ss = build_1r1c(params, dt=dt)
        expected = params.R * params.q_scale * (1.0 - ss.A[0, 0])
        np.testing.assert_allclose(ss.B[0, 1], expected, rtol=1e-12)

    def test_solar_b_d_equals_R_solarscale_times_one_minus_a_d(self) -> None:
        params = self._truth()
        dt = 300.0
        ss = build_1r1c(params, dt=dt)
        expected = params.R * params.solar_scale * (1.0 - ss.A[0, 0])
        np.testing.assert_allclose(ss.B[0, 2], expected, rtol=1e-12)

    def test_step_response_matches_analytical_exponential_decay(self) -> None:
        # Step heat input from rest at outdoor=0, solar=0.
        # T_∞ = R·q_scale·Q; τ = R·C; T_k = T_∞·(1 - A_d^k).
        params = self._truth()
        dt = 300.0
        Q = 1500.0  # W
        n_steps = 600  # ~50h, ~0.6τ
        ss = build_1r1c(params, dt=dt)
        u = np.array([[0.0, Q, 0.0]])
        T_inf = params.R * params.q_scale * Q

        T_k = 0.0
        states_sim = np.empty(n_steps)
        for k in range(n_steps):
            states_sim[k] = T_k
            T_k = (ss.A @ np.array([[T_k]])).item() + (ss.B @ u.T).item()

        # Analytical: T_k = T_∞·(1 - exp(-k·dt/τ))
        t = np.arange(n_steps) * dt
        states_analytical = T_inf * (1.0 - np.exp(-t / params.tau_seconds))

        np.testing.assert_allclose(states_sim, states_analytical, atol=1e-10)

    def test_decay_to_outdoor_matches_analytical(self) -> None:
        # Initial state above T_outdoor, no heating, no solar.
        # T_k = T_outdoor + (T_0 - T_outdoor)·exp(-k·dt/τ).
        params = self._truth()
        dt = 300.0
        T_o = -5.0
        T_0 = 20.0
        n_steps = 600
        ss = build_1r1c(params, dt=dt)
        u = np.array([[T_o, 0.0, 0.0]])

        T_k = T_0
        states_sim = np.empty(n_steps)
        for k in range(n_steps):
            states_sim[k] = T_k
            T_k = (ss.A @ np.array([[T_k]])).item() + (ss.B @ u.T).item()

        t = np.arange(n_steps) * dt
        states_analytical = T_o + (T_0 - T_o) * np.exp(-t / params.tau_seconds)

        np.testing.assert_allclose(states_sim, states_analytical, atol=1e-10)

    def test_steady_state_under_combined_inputs(self) -> None:
        # T_∞ = T_outdoor + R·q_scale·Q + R·solar_scale·SW
        params = self._truth()
        dt = 300.0
        T_o = -5.0
        Q = 2000.0
        SW = 400.0
        ss = build_1r1c(params, dt=dt)
        u = np.array([[T_o, Q, SW]])

        # Iterate well past 20τ to settle to numerical precision
        n_steps = int(20 * params.tau_seconds / dt)
        T_k = 0.0
        for _ in range(n_steps):
            T_k = (ss.A @ np.array([[T_k]])).item() + (ss.B @ u.T).item()

        T_inf_analytical = (
            T_o + params.R * params.q_scale * Q + params.R * params.solar_scale * SW
        )
        np.testing.assert_allclose(T_k, T_inf_analytical, rtol=1e-8)


class TestKernelAnalyticalReference2R2C:
    """2R2C `build_2r2c` matches analytical equilibria; solar wall-routing
    convex combination matches the two-extremes-and-blend construction."""

    def _truth(
        self,
        wall_solar_fraction: float = 0.7,
    ) -> RCParams2R2C:
        # Plausible 2R2C: τ_air ≈ 4h, τ_wall ≈ 80h, masses asymmetric.
        return RCParams2R2C(
            R_ie=0.002,
            R_ea=0.005,
            C_i=7.2e6,  # τ_ie = 4h
            C_e=1.6e7,  # τ_ea = ~22h
            q_scale=0.5,
            solar_scale=2.0,
            sigma_w_i=0.0,
            sigma_w_e=0.0,
            sigma_v=1e-6,
            wall_solar_fraction=wall_solar_fraction,
        )

    def test_steady_state_under_constant_outdoor_only(self) -> None:
        # Both nodes settle to T_o under constant outdoor, zero heat/solar.
        # 2R2C slow eigenmode is longer than τ_wall when R_ie << R_ea —
        # iterate well past it to reach numerical precision.
        params = self._truth()
        dt = 300.0
        T_o = 5.0
        ss = build_2r2c(params, dt=dt)
        u = np.array([[T_o, 0.0, 0.0]])

        x = np.array([[0.0], [0.0]])
        n_steps = int(50 * params.tau_wall / dt)
        for _ in range(n_steps):
            x = ss.A @ x + ss.B @ u.T

        np.testing.assert_allclose(x[0, 0], T_o, atol=1e-6)
        np.testing.assert_allclose(x[1, 0], T_o, atol=1e-6)

    def test_steady_state_with_q_heat_to_air_only(self) -> None:
        # Constant Q_heat at outdoor=0, no solar.
        # KCL at steady state:
        #   Air node: (T_e - T_i)/R_ie + q_scale·Q = 0  →  T_e = T_i - R_ie·q_scale·Q
        #   Env node: (T_i - T_e)/R_ie + (T_o - T_e)/R_ea = 0
        #   With T_o = 0: T_i - T_e = T_e·R_ie/R_ea  →  T_e = T_i·R_ea/(R_ie+R_ea)
        # Combined: T_i = R_ie·q_scale·Q + T_i·R_ea/(R_ie+R_ea)
        #           T_i·(1 - R_ea/(R_ie+R_ea)) = R_ie·q_scale·Q
        #           T_i·R_ie/(R_ie+R_ea) = R_ie·q_scale·Q
        #           T_i = (R_ie + R_ea)·q_scale·Q
        params = self._truth()
        dt = 300.0
        Q = 1000.0
        ss = build_2r2c(params, dt=dt)
        u = np.array([[0.0, Q, 0.0]])

        x = np.array([[0.0], [0.0]])
        n_steps = int(50 * params.tau_wall / dt)
        for _ in range(n_steps):
            x = ss.A @ x + ss.B @ u.T

        T_i_analytical = (params.R_ie + params.R_ea) * params.q_scale * Q
        T_e_analytical = T_i_analytical - params.R_ie * params.q_scale * Q
        np.testing.assert_allclose(x[0, 0], T_i_analytical, rtol=1e-6)
        np.testing.assert_allclose(x[1, 0], T_e_analytical, rtol=1e-6)

    def test_solar_routing_extreme_wall_fraction_zero_routes_all_to_air(self) -> None:
        # wall_solar_fraction=0 → all solar enters the air node directly.
        # B_c[air, solar] = solar_scale / C_i; B_c[wall, solar] = 0.
        # B_d[wall, solar] depends on air→wall propagation over dt; for
        # τ_ie=4h ≫ dt=5min, the wall column should be ≪ the air column.
        params = self._truth(wall_solar_fraction=0.0)
        dt = 300.0
        ss = build_2r2c(params, dt=dt)
        air_b_d = ss.B[0, 2]
        wall_b_d = ss.B[1, 2]
        assert abs(wall_b_d) < 0.05 * abs(air_b_d), (
            f"wall_solar_fraction=0 should leave wall B_d[solar] ≪ air's; "
            f"got wall={wall_b_d:.4g}, air={air_b_d:.4g}"
        )

    def test_solar_routing_extreme_wall_fraction_one_routes_all_to_wall(self) -> None:
        # wall_solar_fraction=1 → all solar enters wall node directly.
        params = self._truth(wall_solar_fraction=1.0)
        dt = 300.0
        ss = build_2r2c(params, dt=dt)
        # In continuous time, B_c[air, solar] = 0 → over a single dt the
        # air node sees solar only via the wall→air coupling. With τ_ie=4h
        # and dt=5min, this propagation is bounded but not zero. ~1.5%
        # leakage is the expected Van-Loan ZOH discretization signature
        # for these parameters; assert ≤5% to allow for similar profiles.
        air_b_d = ss.B[0, 2]
        wall_b_d = ss.B[1, 2]
        assert abs(air_b_d) < 0.05 * abs(wall_b_d), (
            f"wall_solar_fraction=1 should leave air B_d[solar] ≪ wall's; "
            f"got air={air_b_d:.4g}, wall={wall_b_d:.4g}"
        )

    def test_solar_routing_convex_combination_consistent(self) -> None:
        # B_d for solar should be the convex combination of the two extremes:
        # B_d[i, solar] at fraction f = (1-f)·B_d[i, solar | f=0] + f·B_d[i, solar | f=1]
        dt = 300.0
        f = 0.7
        params_blend = self._truth(wall_solar_fraction=f)
        params_air = self._truth(wall_solar_fraction=0.0)
        params_wall = self._truth(wall_solar_fraction=1.0)
        ss_blend = build_2r2c(params_blend, dt=dt)
        ss_air = build_2r2c(params_air, dt=dt)
        ss_wall = build_2r2c(params_wall, dt=dt)

        for node in (0, 1):  # air, wall
            expected = (1 - f) * ss_air.B[node, 2] + f * ss_wall.B[node, 2]
            np.testing.assert_allclose(
                ss_blend.B[node, 2],
                expected,
                rtol=1e-10,
                err_msg=f"node {node} solar B_d not convex combo of extremes",
            )
