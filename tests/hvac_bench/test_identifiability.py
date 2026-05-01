"""Tests for the identifiability diagnostics module (Phase 1b).

Two layers:

1. **Pure-math** — unit tests on ``fisher_information``, ``crlb_diagonal``,
   ``pe_diagnostics``, and ``estimate_sigma2_from_residuals`` against
   analytical known-truth cases (orthogonal columns, singular FIM,
   constant column, identity regressor).
2. **End-to-end** — runs an open-loop probe, builds an
   ``IdentifiabilityReport`` from its observations, and asserts the
   sanity properties any well-designed experiment should satisfy
   (rank == n_features, finite CRLB, reasonable signal-to-noise).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.hvac_bench.identifiability import (
    build_regressor_matrix,
    crlb_diagonal,
    estimate_sigma2_from_residuals,
    fisher_information,
    identifiability_report,
    pe_diagnostics,
)
from tests.hvac_bench.open_loop_runner import (
    OpenLoopConfig,
    make_step_excitation,
    run_open_loop_probe,
)


# ── Pure-math tests ───────────────────────────────────────────────────


class TestFisherInformation:
    def test_unit_weights_recover_ols_form(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        fim = fisher_information(X, weights=None, sigma2=1.0)
        # X.T @ X = [[2, 1], [1, 2]] for this X.
        np.testing.assert_allclose(fim, np.array([[2.0, 1.0], [1.0, 2.0]]))

    def test_sigma2_scales_inversely(self):
        X = np.array([[1.0], [1.0], [1.0]])
        fim_unit = fisher_information(X, sigma2=1.0)
        fim_quad = fisher_information(X, sigma2=4.0)
        # FIM ∝ 1/σ². Quadrupling σ² → 1/4 of FIM.
        np.testing.assert_allclose(fim_quad, fim_unit / 4.0)

    def test_orthogonal_columns_give_diagonal_fim(self):
        # Columns are orthogonal but not unit-normalized.
        X = np.array([[1.0, 2.0], [-1.0, 2.0], [1.0, -2.0], [-1.0, -2.0]])
        fim = fisher_information(X)
        # Off-diagonal Cov(col1, col2) = 0 by orthogonality.
        assert abs(fim[0, 1]) < 1e-12
        assert abs(fim[1, 0]) < 1e-12

    def test_weights_enter_linearly(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = np.array([2.0, 3.0])
        fim = fisher_information(X, weights=w, sigma2=1.0)
        # FIM = X^T diag(w) X = diag(w) for the identity-like X above.
        np.testing.assert_allclose(fim, np.diag([2.0, 3.0]))

    def test_rejects_invalid_sigma2(self):
        X = np.array([[1.0]])
        with pytest.raises(ValueError):
            fisher_information(X, sigma2=0.0)
        with pytest.raises(ValueError):
            fisher_information(X, sigma2=-1.0)

    def test_rejects_mismatched_weights(self):
        X = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError):
            fisher_information(X, weights=np.array([1.0, 1.0]))


class TestCrlbDiagonal:
    def test_diagonal_fim_inverts_elementwise(self):
        fim = np.diag([4.0, 9.0, 16.0])
        crlb = crlb_diagonal(fim)
        np.testing.assert_allclose(crlb, [0.25, 1.0 / 9.0, 1.0 / 16.0])

    def test_singular_fim_marks_unidentifiable(self):
        # Rank-1 FIM: only one direction is identified.
        fim = np.array([[1.0, 1.0], [1.0, 1.0]])
        crlb = crlb_diagonal(fim)
        # The null direction couples both parameters → both should be inf.
        assert math.isinf(crlb[0])
        assert math.isinf(crlb[1])

    def test_full_rank_fim_finite_crlb(self):
        # Non-trivial 2x2 PD matrix.
        fim = np.array([[2.0, 0.5], [0.5, 3.0]])
        crlb = crlb_diagonal(fim)
        assert all(math.isfinite(c) and c > 0 for c in crlb)
        # Verify against direct inv: for [[a,b],[c,d]], inv diag = [d, a]/det
        det = 2.0 * 3.0 - 0.5 * 0.5
        np.testing.assert_allclose(crlb, [3.0 / det, 2.0 / det])

    def test_rejects_non_square(self):
        with pytest.raises(ValueError):
            crlb_diagonal(np.array([[1.0, 2.0]]))


class TestPeDiagnostics:
    """Default ``standardize=True`` is the production case (Belsley κ).
    A few tests use ``standardize=False`` to exercise the raw-SVD path
    explicitly."""

    def test_identity_matrix_unstandardized_full_rank(self):
        X = np.eye(3)
        d = pe_diagnostics(X, standardize=False)
        assert d["rank"] == 3
        assert d["condition_number"] == pytest.approx(1.0, abs=1e-10)

    def test_identity_matrix_standardized_loses_one_rank(self):
        # Mean-centering every column removes the all-ones direction
        # from the column space → rank drops by 1. Expected behaviour
        # of the Belsley convention; only matters when the matrix has
        # no constant column to anchor the intercept.
        X = np.eye(3)
        d = pe_diagnostics(X, standardize=True)
        assert d["rank"] == 2

    def test_constant_column_preserved_under_standardize(self):
        # An intercept column (all 1) is left un-rescaled and not
        # mean-centered, so it still contributes a rank-1 direction.
        X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
        d = pe_diagnostics(X, standardize=True)
        assert d["rank"] == 2

    def test_constant_column_drops_rank_when_matrix_is_all_constant(self):
        # All rows identical → both columns constant → rank 1 either way.
        X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        d_unstd = pe_diagnostics(X, standardize=False)
        assert d_unstd["rank"] == 1
        d_std = pe_diagnostics(X, standardize=True)
        # Standardize preserves both constant columns → still rank 1.
        assert d_std["rank"] == 1

    def test_near_collinear_blows_up_condition(self):
        X = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-9], [2.0, 2.0]])
        d = pe_diagnostics(X, standardize=False)
        assert d["condition_number"] > 1e6

    def test_weights_reweight_singular_values(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        # standardize=False so weight effect is visible at the SV level.
        d_eq = pe_diagnostics(X, weights=np.array([1.0, 1.0]), standardize=False)
        d_asym = pe_diagnostics(X, weights=np.array([1.0, 4.0]), standardize=False)
        assert d_asym["largest_singular_value"] > d_eq["largest_singular_value"]
        assert d_asym["smallest_singular_value"] == pytest.approx(
            d_eq["smallest_singular_value"], abs=1e-10
        )

    def test_empty_matrix(self):
        X = np.zeros((0, 3))
        d = pe_diagnostics(X)
        assert d["rank"] == 0
        assert math.isinf(d["condition_number"])


class TestEstimateSigma2:
    def test_perfect_fit_zero_noise(self):
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        beta = np.array([1.0, 2.0])
        y = X @ beta  # no noise
        s2 = estimate_sigma2_from_residuals(y, X, beta)
        assert s2 == pytest.approx(0.0, abs=1e-12)

    def test_recovers_noise_variance(self):
        rng = np.random.default_rng(0)
        n = 1000
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        beta = np.array([0.5, 2.0])
        sigma_true = 0.7
        y = X @ beta + rng.normal(scale=sigma_true, size=n)
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        s2_hat = estimate_sigma2_from_residuals(y, X, beta_hat)
        # Within ~10% with n=1000.
        assert abs(s2_hat - sigma_true**2) / sigma_true**2 < 0.10

    def test_zero_dof_returns_zero(self):
        X = np.array([[1.0]])
        beta = np.array([1.0])
        y = np.array([1.0])
        # n=1, p=1 → dof = 0.
        assert estimate_sigma2_from_residuals(y, X, beta) == 0.0


# ── End-to-end probe → identifiability_report ─────────────────────────


@pytest.fixture(scope="module")
def probe_observations():
    """30-day open-loop step probe on living_room — same fixture as
    Phase 1a's test, used to verify the identifiability report wires
    end-to-end."""
    excitation = make_step_excitation(
        center_c=24.0,
        amplitude_c=1.5,
        hold_minutes=360.0,
        tick_minutes=15.0,
    )
    config = OpenLoopConfig(
        excitation=excitation,
        n_days=30,
        profile_name="living_room",
        outdoor_base_c=-2.0,
        outdoor_diurnal_c=8.0,
        desired_c=20.0,
        mode="heat",
        noise_sigma=0.05,
        noise_seed=42,
        tick_minutes=15.0,
    )
    return run_open_loop_probe(config).observations


class TestIdentifiabilityReportFromProbe:

    def test_full_rank_for_intercept_plus_outdoor(self, probe_observations):
        rep = identifiability_report(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
        )
        # 2-parameter design must be PE order ≥ 2 to be identifiable.
        assert rep.rank == 2
        assert rep.pe_order == 2
        assert rep.n_features == 2
        # Most ticks pass the room_rate filter at 6h hold ≫ τ.
        assert rep.n_observations > 1000

    def test_finite_crlb_for_each_feature(self, probe_observations):
        rep = identifiability_report(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
        )
        assert all(math.isfinite(c) and c > 0 for c in rep.crlb)
        assert all(math.isfinite(s) for s in rep.std_err_lower_bound)

    def test_outdoor_delta_well_identified(self, probe_observations):
        # With sigma2 set to the known sensor-noise variance, the CRLB
        # on outdoor_delta should be small enough that any unbiased
        # estimator can resolve β to within a few percent of typical
        # operating values.
        rep = identifiability_report(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
            sigma2=0.05**2,
        )
        outdoor_idx = rep.feature_names.index("outdoor_delta")
        # Truth is around -0.20 (open-loop) or -0.25 (closed-loop). Lower
        # bound on SE should be much smaller — at least one order of
        # magnitude — so the estimator could in principle distinguish
        # the two.
        assert rep.std_err_lower_bound[outdoor_idx] < 0.02

    def test_condition_number_reasonable(self, probe_observations):
        # Step probe with diurnal outdoor produces well-conditioned X
        # because intercept (constant) is orthogonal to centered outdoor.
        rep = identifiability_report(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
        )
        # Belsley (1980) calls κ > 30 "moderately collinear" and κ > 100
        # "strongly collinear". For a clean 2-parameter probe with
        # diurnal outdoor we expect well below 30.
        assert rep.condition_number < 30.0

    def test_sigma2_estimation_from_residuals(self, probe_observations):
        # Verifies the residual-based estimator wires correctly. The
        # reported sigma2 is the **residual variance** under the assumed
        # linear model — NOT the sensor-noise variance. Under model
        # misspecification (the 2R2C step probe has wall-transient
        # structural residuals on top of sensor noise), σ̂² is
        # appropriately larger than the configured noise variance, and
        # the resulting CRLB widens to reflect that.
        from tests.hvac_bench.identifiability import build_regressor_matrix
        X, y, w, _ = build_regressor_matrix(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
        )
        beta_hat, *_ = np.linalg.lstsq(
            np.sqrt(w)[:, None] * X, np.sqrt(w) * y, rcond=None
        )
        rep = identifiability_report(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
            sigma2=None,
            beta=beta_hat.tolist(),
        )
        assert rep.sigma2_was_estimated is True
        # Plausible band: residual variance > sensor noise (model is
        # not perfect) but bounded above by the y variance (model is
        # better than nothing).
        y_var = float(((y - y.mean()) ** 2).mean())
        sensor_noise_var = 0.05 ** 2
        assert sensor_noise_var < rep.sigma2 < y_var, (
            f"sigma2 {rep.sigma2:.4f} out of plausible band "
            f"({sensor_noise_var:.4f}, {y_var:.4f})"
        )

    def test_feature_variance_intercept_zero(self, probe_observations):
        # The intercept column is constant 1.0 → zero centered variance.
        rep = identifiability_report(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
        )
        intercept_idx = rep.feature_names.index("intercept")
        assert rep.feature_variance[intercept_idx] == pytest.approx(0.0, abs=1e-12)
        outdoor_idx = rep.feature_names.index("outdoor_delta")
        assert rep.feature_variance[outdoor_idx] > 1.0  # diurnal swing of ±8°C


class TestEmptyObservations:
    """Edge case: no eligible observations after filter → safe report."""

    def test_empty_input_returns_inf_crlb(self):
        rep = identifiability_report(
            observations=[],
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
        )
        assert rep.n_observations == 0
        assert rep.rank == 0
        assert all(math.isinf(c) for c in rep.crlb)
        assert math.isinf(rep.condition_number)


class TestRegressorMatrixMatchesWLS:
    """Bridge contract: the regressor matrix from this module must
    produce the same WLS β as ``weighted_least_squares`` on the same
    observations. Verifies no off-by-one / convention drift."""

    def test_beta_agrees_with_weighted_least_squares(self, probe_observations):
        from custom_components.tasmota_irhvac.pi.batch_learning import (
            weighted_least_squares,
        )

        X, y, w, _ = build_regressor_matrix(
            probe_observations,
            feature_order=["intercept", "outdoor_delta"],
        )
        # Plain weighted lstsq via numpy.
        Wsqrt = np.sqrt(w)[:, None]
        beta_np, *_ = np.linalg.lstsq(Wsqrt * X, Wsqrt[:, 0] * y, rcond=None)

        # Production WLS path.
        result = weighted_least_squares(
            observations=probe_observations,
            n_features=2,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=None,
            min_observations=20,
            detect_lag=False,
        )
        assert result is not None
        # The production path adds ridge + column scaling, so they won't
        # be bit-identical, but β_outdoor should agree to ~1e-3.
        assert abs(beta_np[1] - result.beta_batch[1]) < 1e-3
