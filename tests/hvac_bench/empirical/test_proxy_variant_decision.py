"""Tier 1.2 discrimination: heat-injection proxy variant comparison.

Compares Phase 4 lite verdicts under proxy variant (a) "constant" vs
variant (b) "setpoint_modulated" per `project_phase4_data_survey.md`.
The hypothesis: constant-magnitude proxy has structural mismatch
(heat is on 82% of train window but proxy only scales magnitude),
producing the q_scale=0.086 rail observed in living_room at the
2026-05-01 baseline. Setpoint-modulated should lift q_scale off rail.

Marked @design — one-shot decision experiment. ~40-60 min wall-clock.
Captured output is the deliverable; no specific verdict locked in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.hvac_bench.empirical.runner import (
    CredibilityEnvelope,
    run_phase4_lite,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = REPO_ROOT / "local" / "debug_bundles" / "condenser_a_data"
BUNDLE_AVAILABLE = BUNDLE_PATH.exists() and (BUNDLE_PATH / "living_room.csv").exists()


@pytest.mark.design
@pytest.mark.skipif(not BUNDLE_AVAILABLE, reason="Condenser A bundle not present")
class TestProxyVariantDiscrimination:
    """A/B Phase 4 lite verdict at fixed n_restarts across proxy variants."""

    @pytest.fixture(scope="class")
    def envelope_constant(self) -> CredibilityEnvelope:
        return run_phase4_lite(BUNDLE_PATH, seed=0, proxy_variant="constant")

    @pytest.fixture(scope="class")
    def envelope_modulated(self) -> CredibilityEnvelope:
        return run_phase4_lite(
            BUNDLE_PATH, seed=0, proxy_variant="setpoint_modulated"
        )

    def test_variants_complete_with_valid_envelopes(
        self,
        envelope_constant: CredibilityEnvelope,
        envelope_modulated: CredibilityEnvelope,
    ) -> None:
        for env in (envelope_constant, envelope_modulated):
            assert env.overall_classification in {"good", "close", "poor"}
            assert set(env.zones) == {"living_room", "dining_room", "bunkroom"}

    def test_records_variant_comparison(
        self,
        envelope_constant: CredibilityEnvelope,
        envelope_modulated: CredibilityEnvelope,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print per-zone (q_scale, τ, rails, cv-fails, classification) for
        both variants. Output captures the Tier 1.2 decision evidence.
        """
        with capsys.disabled():
            print()
            print("=" * 78)
            print("Tier 1.2: proxy variant discrimination — Phase 4 lite verdicts")
            print("=" * 78)
            print(
                f"{'zone':<14}{'variant':<22}{'class':<8}"
                f"{'τ_h':>8}{'q_scale':>10}{'rails':>7}{'cv_fail':>9}"
            )
            for name in envelope_constant.zones:
                for label, env in (
                    ("constant", envelope_constant),
                    ("setpoint_modulated", envelope_modulated),
                ):
                    z = env.zones[name]
                    f1 = z.train_result.fit_1r1c.best.params
                    id_1 = z.train_result.identifiability_1r1c
                    print(
                        f"{name:<14}{label:<22}{z.classification:<8}"
                        f"{f1.tau_s/3600:>8.1f}{f1.q_scale:>10.3f}"
                        f"{id_1.n_at_bound:>7}{id_1.n_failed_cv:>9}"
                    )
            print()
            print(
                f"overall: constant={envelope_constant.overall_classification} "
                f"vs setpoint_modulated={envelope_modulated.overall_classification}"
            )
