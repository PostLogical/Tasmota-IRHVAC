"""Tests for the assertion harness (Phase 1e).

Layered tests:

1. **Pure-math** — fabricated reports verify each assertion fires
   correctly (raises) on invalid input and is silent on valid input.
   No probe, no I/O.
2. **End-to-end** — wires the harness through a real probe + diagnostic
   pipeline and shows the assertions accept the resulting valid report.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.hvac_bench.assertions import (
    assert_condition_number_below,
    assert_identifiable,
    assert_pe_order_at_least,
    assert_recovers_within_crlb,
    assert_recovers_within_se,
    assert_residuals_normal,
    assert_residuals_white,
    assert_split_half_stable,
)
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.identifiability import (
    IdentifiabilityReport,
    identifiability_report,
)
from tests.hvac_bench.open_loop_runner import (
    OpenLoopConfig,
    make_step_excitation,
    run_open_loop_probe,
)
from tests.hvac_bench.residual_diagnostics import (
    ResidualDiagnosticReport,
    residual_diagnostic_report,
)


# ── Helpers to fabricate reports ─────────────────────────────────────


def _make_id_report(
    *,
    feature_names: list[str] = ["intercept", "outdoor_delta"],
    crlb: list[float] | None = None,
    rank: int = 2,
    condition_number: float = 5.0,
    n_observations: int = 1000,
) -> IdentifiabilityReport:
    if crlb is None:
        crlb = [1e-3] * len(feature_names)
    return IdentifiabilityReport(
        feature_names=feature_names,
        n_observations=n_observations,
        n_features=len(feature_names),
        crlb=crlb,
        std_err_lower_bound=[
            math.sqrt(c) if math.isfinite(c) else float("inf") for c in crlb
        ],
        rank=rank,
        condition_number=condition_number,
        smallest_singular_value=1.0,
        largest_singular_value=condition_number,
        feature_variance=[1.0] * len(feature_names),
        pe_order=rank,
        sigma2=0.01,
        sigma2_was_estimated=False,
    )


def _make_residual_report(
    *,
    n_residuals: int = 1000,
    ljung_box_p: float = 0.5,
    normality_p: float = 0.5,
    split_half_max_rel: float | None = 0.05,
) -> ResidualDiagnosticReport:
    return ResidualDiagnosticReport(
        n_residuals=n_residuals,
        mean=0.0, std=1.0, skewness=0.0, kurtosis=0.0,
        autocorrelation=[1.0, 0.0, 0.0, 0.0],
        ljung_box_statistic=10.0,
        ljung_box_p_value=ljung_box_p,
        is_white_at_alpha_05=ljung_box_p >= 0.05,
        normality_test_name="shapiro-wilk",
        normality_statistic=0.99,
        normality_p_value=normality_p,
        is_normal_at_alpha_05=normality_p >= 0.05,
        split_half_beta_first=[1.0, -0.2],
        split_half_beta_second=[1.0, -0.21],
        split_half_max_rel_change=split_half_max_rel,
    )


# ── assert_identifiable ───────────────────────────────────────────────


class TestAssertIdentifiable:
    def test_passes_when_crlb_below_threshold(self, bench_metrics, num_regression):
        rep = _make_id_report(crlb=[1e-3, 5e-4])
        assert_identifiable(rep, "outdoor_delta", crlb_max=1e-3)

    def test_fails_when_crlb_above_threshold(self, bench_metrics, num_regression):
        rep = _make_id_report(crlb=[1e-3, 1.0])
        with pytest.raises(AssertionError, match="CRLB.*outdoor_delta"):
            assert_identifiable(rep, "outdoor_delta", crlb_max=1e-3)

    def test_fails_when_unidentifiable(self, bench_metrics, num_regression):
        rep = _make_id_report(crlb=[float("inf"), 1e-3], rank=1)
        with pytest.raises(AssertionError, match="unidentifiable"):
            assert_identifiable(rep, "intercept", crlb_max=1.0)

    def test_fails_when_feature_missing(self, bench_metrics, num_regression):
        rep = _make_id_report()
        with pytest.raises(AssertionError, match="not in regressor"):
            assert_identifiable(rep, "Solar Proxy", crlb_max=1.0)

    def test_se_max_bound(self, bench_metrics, num_regression):
        # CRLB = 0.04 → SE = 0.2. Bound 0.1 should fail.
        rep = _make_id_report(crlb=[0.04, 0.04])
        with pytest.raises(AssertionError, match="SE_min"):
            assert_identifiable(rep, "outdoor_delta", se_max=0.1)
        # Bound 0.3 should pass.
        assert_identifiable(rep, "outdoor_delta", se_max=0.3)

    def test_requires_some_bound(self, bench_metrics, num_regression):
        rep = _make_id_report()
        with pytest.raises(ValueError, match="crlb_max or se_max"):
            assert_identifiable(rep, "outdoor_delta")


class TestAssertPeOrder:
    def test_passes_at_threshold(self, bench_metrics, num_regression):
        rep = _make_id_report(rank=3, feature_names=["a", "b", "c"])
        assert_pe_order_at_least(rep, 3)

    def test_fails_below_threshold(self, bench_metrics, num_regression):
        rep = _make_id_report(rank=1, feature_names=["a", "b", "c"])
        with pytest.raises(AssertionError, match="PE order 1 < 3"):
            assert_pe_order_at_least(rep, 3)


class TestAssertConditionNumber:
    def test_passes_when_below(self, bench_metrics, num_regression):
        rep = _make_id_report(condition_number=15.0)
        assert_condition_number_below(rep, 30.0)

    def test_fails_when_above(self, bench_metrics, num_regression):
        rep = _make_id_report(condition_number=120.0)
        with pytest.raises(AssertionError, match="condition number 120"):
            assert_condition_number_below(rep, 100.0)


# ── Residual assertions ──────────────────────────────────────────────


class TestAssertResidualsWhite:
    def test_passes_when_p_above_alpha(self, bench_metrics, num_regression):
        rep = _make_residual_report(ljung_box_p=0.40)
        assert_residuals_white(rep, alpha=0.05)

    def test_fails_when_p_below_alpha(self, bench_metrics, num_regression):
        rep = _make_residual_report(ljung_box_p=0.001)
        with pytest.raises(AssertionError, match="Ljung-Box rejects"):
            assert_residuals_white(rep, alpha=0.05)


class TestAssertResidualsNormal:
    def test_passes_when_p_above_alpha(self, bench_metrics, num_regression):
        rep = _make_residual_report(normality_p=0.30)
        assert_residuals_normal(rep)

    def test_fails_when_p_below_alpha(self, bench_metrics, num_regression):
        rep = _make_residual_report(normality_p=0.001)
        with pytest.raises(AssertionError, match="rejects normality"):
            assert_residuals_normal(rep)


class TestAssertSplitHalfStable:
    def test_passes_when_change_small(self, bench_metrics, num_regression):
        rep = _make_residual_report(split_half_max_rel=0.05)
        assert_split_half_stable(rep, max_rel_change=0.10)

    def test_fails_when_change_large(self, bench_metrics, num_regression):
        rep = _make_residual_report(split_half_max_rel=0.30)
        with pytest.raises(AssertionError, match="split-half"):
            assert_split_half_stable(rep, max_rel_change=0.10)

    def test_fails_when_not_computed(self, bench_metrics, num_regression):
        rep = _make_residual_report(split_half_max_rel=None)
        with pytest.raises(AssertionError, match="not computed"):
            assert_split_half_stable(rep, max_rel_change=0.10)


# ── Recovery assertions ──────────────────────────────────────────────


class TestAssertRecoversWithinSe:
    def test_passes_within_band(self, bench_metrics, num_regression):
        assert_recovers_within_se(
            estimated=0.21, truth=0.20, se=0.01, k=3.0
        )

    def test_fails_outside_band(self, bench_metrics, num_regression):
        with pytest.raises(AssertionError, match=r"not within 3\.0σ"):
            assert_recovers_within_se(
                estimated=0.50, truth=0.20, se=0.01, k=3.0
            )

    def test_rejects_zero_se(self, bench_metrics, num_regression):
        with pytest.raises(ValueError, match="se must be > 0"):
            assert_recovers_within_se(
                estimated=0.20, truth=0.20, se=0.0
            )


class TestAssertRecoversWithinCrlb:
    def test_passes_within_band(self, bench_metrics, num_regression):
        rep = _make_id_report(
            feature_names=["intercept", "outdoor_delta"],
            crlb=[1e-3, 1e-4],
        )
        # SE on outdoor_delta = sqrt(1e-4) = 0.01. 3σ = 0.03.
        assert_recovers_within_crlb(
            estimated=-0.21, truth=-0.20, report=rep,
            feature="outdoor_delta", k=3.0,
        )

    def test_fails_with_estimator_diagnosis_when_gap_huge(self, bench_metrics, num_regression):
        rep = _make_id_report(crlb=[1e-3, 1e-4])
        with pytest.raises(AssertionError, match="estimator failed"):
            assert_recovers_within_crlb(
                estimated=-0.05, truth=-0.20, report=rep,
                feature="outdoor_delta", k=3.0,
            )

    def test_fails_with_design_diagnosis_when_gap_modest(self, bench_metrics, num_regression):
        # CRLB so loose that even modest gap exceeds the band — but only
        # by a small ratio. Diagnosis should be "design tight against truth".
        rep = _make_id_report(crlb=[1e-3, 0.04])
        # SE = 0.2, 3σ = 0.6. Gap of 0.7 → ratio 0.7/0.6 ≈ 1.17.
        with pytest.raises(
            AssertionError, match="experiment design"
        ):
            assert_recovers_within_crlb(
                estimated=-0.90, truth=-0.20, report=rep,
                feature="outdoor_delta", k=3.0,
            )

    def test_fails_when_unidentifiable(self, bench_metrics, num_regression):
        rep = _make_id_report(
            feature_names=["intercept", "outdoor_delta"],
            crlb=[1e-3, float("inf")], rank=1,
        )
        with pytest.raises(AssertionError, match="unidentifiable"):
            assert_recovers_within_crlb(
                estimated=-0.20, truth=-0.20, report=rep,
                feature="outdoor_delta",
            )


# ── End-to-end: harness wired through a real probe ───────────────────


@pytest.fixture(scope="module")
def probe_pipeline():
    """Run a 30-day probe + WLS + diagnostics; the assertions on this
    output should all pass."""
    excitation = make_step_excitation(
        center_c=24.0, amplitude_c=1.5, hold_minutes=360.0, tick_minutes=15.0,
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
    obs = run_open_loop_probe(config).observations

    from tests.hvac_bench.identifiability import build_regressor_matrix

    X, y, w, _ = build_regressor_matrix(
        obs, feature_order=["intercept", "outdoor_delta"]
    )
    Wsqrt = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Wsqrt * X, Wsqrt[:, 0] * y, rcond=None)

    id_rep = identifiability_report(
        observations=obs,
        feature_order=["intercept", "outdoor_delta"],
        sigma2=None,
        beta=beta.tolist(),
    )
    res_rep = residual_diagnostic_report(
        observations=obs,
        beta=beta.tolist(),
        feature_order=["intercept", "outdoor_delta"],
    )
    return beta, id_rep, res_rep


class TestEndToEnd:
    def test_outdoor_delta_identifiable(self, bench_metrics, num_regression, probe_pipeline):
        _, id_rep, _ = probe_pipeline
        outdoor_idx = id_rep.feature_names.index("outdoor_delta")
        bench_metrics["se_outdoor_delta"] = id_rep.std_err_lower_bound[outdoor_idx]
        check_bench_metrics(num_regression, bench_metrics)
        # On a 30-day probe, β_outdoor SE should be well below 0.1.
        assert_identifiable(id_rep, "outdoor_delta", se_max=0.1)

    def test_pe_order_full(self, bench_metrics, num_regression, probe_pipeline):
        _, id_rep, _ = probe_pipeline
        bench_metrics["pe_order"] = id_rep.pe_order
        bench_metrics["rank"] = id_rep.rank
        check_bench_metrics(num_regression, bench_metrics)
        assert_pe_order_at_least(id_rep, 2)

    def test_condition_number_under_30(self, bench_metrics, num_regression, probe_pipeline):
        _, id_rep, _ = probe_pipeline
        bench_metrics["condition_number"] = id_rep.condition_number
        check_bench_metrics(num_regression, bench_metrics)
        assert_condition_number_below(id_rep, 30.0)

    def test_split_half_stable(self, bench_metrics, num_regression, probe_pipeline):
        _, _, res_rep = probe_pipeline
        bench_metrics["split_half_max_rel_change"] = res_rep.split_half_max_rel_change
        check_bench_metrics(num_regression, bench_metrics)
        # Loose threshold — split-half on a misspecified model can shift
        # β by tens of percent when the second half differs in transient
        # composition. The harness still admits the reasonable case.
        assert_split_half_stable(res_rep, max_rel_change=0.50)

    def test_recovery_assertion_against_open_loop_truth_fails_explicably(
        self, bench_metrics, num_regression, probe_pipeline
    ):
        # The probe doesn't reach the open-loop asymptote (mass_ratio=8
        # wall lag), so a strict CRLB-based recovery assertion against
        # -0.20 is expected to fail. The failure message should name
        # this clearly so the test author isn't confused.
        beta, id_rep, _ = probe_pipeline
        outdoor_idx = id_rep.feature_names.index("outdoor_delta")
        with pytest.raises(AssertionError) as exc_info:
            assert_recovers_within_crlb(
                estimated=float(beta[outdoor_idx]),
                truth=-0.20,
                report=id_rep,
                feature="outdoor_delta",
                k=3.0,
            )
        # The message should mention either "estimator failed" or
        # "experiment design" — both diagnoses are meaningful.
        msg = str(exc_info.value)
        assert "estimator failed" in msg or "experiment design" in msg
