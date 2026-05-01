"""Tests for Phase 4 lite runner.

Most tests are pure-function (classification aggregation, formatting).
The end-to-end run on the bundle is marked @slow + @design — it takes
several minutes and is the verdict-into-memo deliverable, not part of
routine CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.hvac_bench.empirical.data_loader import (
    TRAIN_WINDOW,
    VALIDATE_WINDOW,
)
from tests.hvac_bench.empirical.runner import (
    CLASSIFICATION_RANK,
    CredibilityEnvelope,
    ZoneCredibility,
    _aggregate_classification,
    _one_step_rmse,
    format_envelope_memory,
    run_phase4_lite,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = REPO_ROOT / "local" / "debug_bundles" / "condenser_a_data"
BUNDLE_AVAILABLE = BUNDLE_PATH.exists() and (BUNDLE_PATH / "living_room.csv").exists()


# ── Pure helpers ─────────────────────────────────────────────────────────


class TestClassificationAggregation:
    def test_min_picks_worst(self) -> None:
        assert _aggregate_classification(["good", "close", "poor"]) == "poor"

    def test_all_good_returns_good(self) -> None:
        assert _aggregate_classification(["good", "good"]) == "good"

    def test_close_and_good_returns_close(self) -> None:
        assert _aggregate_classification(["close", "good"]) == "close"

    def test_empty_returns_poor(self) -> None:
        assert _aggregate_classification([]) == "poor"

    def test_classification_rank_orders_correctly(self) -> None:
        assert CLASSIFICATION_RANK["good"] > CLASSIFICATION_RANK["close"]
        assert CLASSIFICATION_RANK["close"] > CLASSIFICATION_RANK["poor"]


class TestOneStepRmse:
    def test_constant_zero_innovations_zero_rmse(self) -> None:
        assert _one_step_rmse(np.zeros(100)) == 0.0

    def test_handles_nans(self) -> None:
        innov = np.array([1.0, 2.0, np.nan, np.nan, 1.0])
        # RMSE over valid: sqrt((1+4+1)/3) = sqrt(2)
        np.testing.assert_allclose(_one_step_rmse(innov), np.sqrt(2.0), rtol=1e-10)

    def test_all_nan_returns_nan(self) -> None:
        assert np.isnan(_one_step_rmse(np.full(10, np.nan)))


# ── Memory formatter ─────────────────────────────────────────────────────


def _stub_envelope() -> CredibilityEnvelope:
    """Hand-constructed envelope that doesn't require a real fit."""
    from tests.hvac_bench.empirical.forward_selection import (
        ForwardSelectionResult,
        IdentifiabilityGate,
    )
    from tests.hvac_bench.empirical.pem_fit import (
        FitParams1R1C,
        FitResult,
        RestartResult,
        DEFAULT_BOUNDS_1R1C,
    )

    p = FitParams1R1C(
        tau_s=10 * 3600,
        q_scale=1.0,
        solar_scale=0.001,
        sigma_w=1e-5,
        sigma_v=0.1,
    )
    rr = RestartResult(
        success=True,
        log_likelihood=-100.0,
        params=p,
        initial=p,
        n_iter=10,
        message="ok",
    )
    fit_1r1c_stub = FitResult(
        model_name="1R1C",
        best=rr,
        restarts=[rr],
        n_obs=2000,
        cv_per_param={"tau_s": 0.05},
        at_bound_per_param={"tau_s": False, "q_scale": False, "solar_scale": False, "sigma_w": False, "sigma_v": False},
        bounds=DEFAULT_BOUNDS_1R1C,
    )
    id_gate = IdentifiabilityGate(
        cv_pass_per_param={"tau_s": True},
        at_bound_per_param={"tau_s": False},
        n_failed_cv=0,
        n_at_bound=0,
        overall_pass=True,
    )
    fwd = ForwardSelectionResult(
        selected_model="1R1C",
        classification="close",
        summary="1R1C accepted (stub)",
        fit_1r1c=fit_1r1c_stub,
        fit_2r2c=None,
        lr_test=None,
        residuals_1r1c=None,
        residuals_2r2c=None,
        identifiability_1r1c=id_gate,
        identifiability_2r2c=None,
        rejection_notes=[],
    )

    z = ZoneCredibility(
        zone="test_zone",
        n_train_obs=2000,
        n_validate_obs=500,
        train_result=fwd,
        validate_residuals=None,
        train_rmse_c=0.21,
        validate_rmse_c=0.45,
        classification="close",
        summary="test_zone: selected=1R1C, classification=close, RMSE=0.21°C",
    )
    return CredibilityEnvelope(
        bundle_path=Path("/fake/bundle"),
        zones={"test_zone": z},
        train_window=TRAIN_WINDOW,
        validate_window=VALIDATE_WINDOW,
        overall_classification="close",
        summary="stub envelope",
        rejection_notes=[],
    )


class TestFormatEnvelopeMemory:
    def test_includes_overall_classification(self) -> None:
        env = _stub_envelope()
        out = format_envelope_memory(env)
        assert "Overall classification" in out
        assert "close" in out

    def test_includes_zone_section(self) -> None:
        env = _stub_envelope()
        out = format_envelope_memory(env)
        assert "test_zone" in out
        assert "1R1C" in out
        assert "0.21" in out  # train RMSE

    def test_includes_leprince_prior_blurb(self) -> None:
        env = _stub_envelope()
        out = format_envelope_memory(env)
        assert "Leprince" in out
        assert "Annex 71" in out

    def test_renders_train_window_label(self) -> None:
        env = _stub_envelope()
        out = format_envelope_memory(env)
        assert TRAIN_WINDOW.label in out
        assert VALIDATE_WINDOW.label in out


# ── End-to-end on real bundle (slow/design) ──────────────────────────────


@pytest.mark.slow
@pytest.mark.design
@pytest.mark.skipif(not BUNDLE_AVAILABLE, reason="Condenser A bundle not present")
class TestEndToEndOnBundle:
    """Full Phase 4 lite run on the actual bundle. ~5-10 minutes wall-clock.

    Asserts the methodology fires and produces a classification per zone;
    does NOT lock specific numbers. Locked numbers belong in the
    credibility-envelope memory file (the deliverable), not in tests.
    """

    @pytest.fixture(scope="class")
    def envelope(self) -> CredibilityEnvelope:
        return run_phase4_lite(BUNDLE_PATH, n_restarts=3, seed=0)

    def test_returns_three_fit_target_zones(
        self, envelope: CredibilityEnvelope
    ) -> None:
        assert set(envelope.zones) == {"living_room", "dining_room", "bunkroom"}

    def test_overall_classification_is_a_known_value(
        self, envelope: CredibilityEnvelope
    ) -> None:
        assert envelope.overall_classification in {"good", "close", "poor"}

    def test_each_zone_has_a_selected_model(
        self, envelope: CredibilityEnvelope
    ) -> None:
        for z in envelope.zones.values():
            assert z.train_result.selected_model in {"1R1C", "2R2C"}

    def test_train_rmse_finite(self, envelope: CredibilityEnvelope) -> None:
        for z in envelope.zones.values():
            assert np.isfinite(z.train_rmse_c)
            assert z.train_rmse_c < 5.0  # sanity: shouldn't be wildly off

    def test_summary_mentions_each_zone(
        self, envelope: CredibilityEnvelope
    ) -> None:
        for name in envelope.zones:
            assert name in envelope.summary
