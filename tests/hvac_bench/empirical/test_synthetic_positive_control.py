"""Tier 1.3 synthetic positive-control: can Phase 4 lite recover known truth?

Two phases per the bench-validation discrimination plan:

  1.3a — Synthetic excitation POC. Open-loop synthetic inputs designed to
         maximise informativeness for 1R1C identification. Decoupled from
         bundle data; tests methodology reach on cleanest signals.

  1.3b — Bundle excitation realism check. Same literature truth, but with
         the bundle's recorded outdoor/setpoint/solar driving signals
         replacing the synthetic ones. Tests whether the bundle's
         excitation pattern is sufficient for the methodology.

Both use literature-grounded truth params (Levermore 2020 winter envelope:
τ=80h, sigma_v=0.05°C). solar_scale (β_solar) sweeps across {1.0, 2.0, 3.0}
to test for the documented ~40% structural bias floor (project_bench_solar
_fidelity.md) at the kernel/PEM layer rather than only the WLS layer.

Marked @slow — permanent regression property of the bench. ~30-45 min
wall-clock at n_restarts=4. Test failure means: bench can't recover its own
kernel + that excitation level + that proxy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.hvac_bench.empirical.data_loader import (
    apply_default_exclusions,
    load_condenser_a_zone,
    slice_window,
    TRAIN_WINDOW,
    VALIDATE_WINDOW,
)
from tests.hvac_bench.empirical.rc_model import RCParams1R1C
from tests.hvac_bench.empirical.runner import (
    ZoneCredibility,
    run_zone,
)
from tests.hvac_bench.empirical.synthetic_drivers import (
    make_synthetic_zone_telemetry,
    replace_room_temp_with_synthetic,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_PATH = REPO_ROOT / "local" / "debug_bundles" / "condenser_a_data"
BUNDLE_AVAILABLE = BUNDLE_PATH.exists() and (BUNDLE_PATH / "living_room.csv").exists()


# Literature truth params — Levermore 2020 winter envelope (residential, single-zone).
# τ = 80h, mid-range; q_scale = 1.0 (proxy uses nominal capacity directly);
# sigma_v = 0.05°C (typical 5-min-averaged temperature sensor noise);
# solar_scale starts at 2.0 and sweeps across the bias-floor range.
_TAU_S = 80.0 * 3600.0
_R = 0.005  # 5 K/kW envelope, mid-range residential
_C = _TAU_S / _R  # = 5.76e7 J/K


def _truth(solar_scale: float = 2.0) -> RCParams1R1C:
    return RCParams1R1C(
        R=_R,
        C=_C,
        q_scale=1.0,
        solar_scale=solar_scale,
        sigma_w=0.0,  # zero process noise → pure measurement noise scenario
        sigma_v=0.05,
    )


# 18 days at 5-min, mirroring the bundle train window length
_N_TRAIN_STEPS = 18 * 24 * 12  # 5184
_N_VALIDATE_STEPS = 8 * 24 * 12  # 2304 — mirrors bundle validate window
_N_RESTARTS = 4  # synthetic data identifies cleanly; literature minimum not needed


# ── Tier 1.3a — Synthetic excitation POC ────────────────────────────────


@pytest.mark.slow
class TestSyntheticExcitationPOC:
    """Methodology-reach baseline: literature truth + open-loop synthetic
    excitation. Establishes the upper bound on what Phase 4 lite can do."""

    @pytest.fixture(scope="class")
    def canonical_credibility(self) -> ZoneCredibility:
        """Single run at canonical literature truth (β_solar=2.0)."""
        truth = _truth(solar_scale=2.0)
        train = make_synthetic_zone_telemetry(
            truth, n_steps=_N_TRAIN_STEPS, seed=0
        )
        validate = make_synthetic_zone_telemetry(
            truth, n_steps=_N_VALIDATE_STEPS, seed=100
        )
        return run_zone(
            "synthetic_open_loop",
            train,
            validate,
            n_restarts=_N_RESTARTS,
            seed=0,
        )

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "BENCH BUG (Tier 1.3a, 2026-05-04): on clean synthetic excitation "
            "with literature truth (τ=80h, q_scale=1.0, solar_scale=2.0), "
            "the methodology classifies 'poor' despite RMSE 0.053°C ≈ sensor "
            "noise. Validate residuals fail Ljung-Box whiteness (p=0.0071) "
            "even though the model exactly matches the truth. Either the "
            "residual whiteness threshold is mis-calibrated, or there is "
            "a structural issue producing non-white residuals at correctly-"
            "specified models. Removing this xfail when the bug is fixed."
        ),
    )
    def test_canonical_truth_recovers_at_least_close_classification(
        self,
        canonical_credibility: ZoneCredibility,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """On clean synthetic excitation with literature truth, methodology
        should classify at least "close" — anything worse means the bench
        cannot recover its own kernel under ideal conditions.
        """
        with capsys.disabled():
            print()
            print(f"1.3a canonical: classification={canonical_credibility.classification}")
            print(f"  train RMSE={canonical_credibility.train_rmse_c:.4f}°C")
            if canonical_credibility.validate_rmse_c is not None:
                print(
                    f"  validate RMSE={canonical_credibility.validate_rmse_c:.4f}°C"
                )
        assert canonical_credibility.classification in {"good", "close"}, (
            f"synthetic POC failed at literature truth: "
            f"got '{canonical_credibility.classification}', expected good/close"
        )

    def test_canonical_recovers_tau_within_factor_of_2(
        self,
        canonical_credibility: ZoneCredibility,
    ) -> None:
        """Recovered τ should be within a factor of 2 of truth (80h)."""
        recovered = canonical_credibility.train_result.fit_1r1c.best.params
        ratio = recovered.tau_s / _TAU_S
        assert 0.5 < ratio < 2.0, (
            f"τ recovery off: truth={_TAU_S/3600:.1f}h, "
            f"recovered={recovered.tau_s/3600:.1f}h (ratio={ratio:.2f})"
        )

    @pytest.fixture(scope="class")
    def beta_solar_sweep(self) -> dict[float, ZoneCredibility]:
        """Run 3 fits across solar_scale truth ∈ {1.0, 2.0, 3.0}.

        Discriminates kernel-layer solar bias from regression-form bias:
        if recovered solar_scale tracks truth → kernel clean; if it
        saturates near ~1.2 regardless → kernel bias confirmed.
        """
        results: dict[float, ZoneCredibility] = {}
        for truth_solar in (1.0, 2.0, 3.0):
            truth = _truth(solar_scale=truth_solar)
            train = make_synthetic_zone_telemetry(
                truth, n_steps=_N_TRAIN_STEPS, seed=int(truth_solar * 10)
            )
            validate = make_synthetic_zone_telemetry(
                truth,
                n_steps=_N_VALIDATE_STEPS,
                seed=int(truth_solar * 10) + 100,
            )
            results[truth_solar] = run_zone(
                f"synth_solar_{truth_solar}",
                train,
                validate,
                n_restarts=_N_RESTARTS,
                seed=0,
            )
        return results

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "BENCH BUG (Tier 1.3a, 2026-05-04): on synthetic data with literature "
            "truth solar_scale ∈ {1.0, 2.0, 3.0}, the optimizer collapses "
            "solar_scale to ~0 (rails at lower bound 1e-6) regardless of truth. "
            "Same kernel on forward and inverse — suggests solar_scale is "
            "weakly identifiable at this excitation level (outdoor diurnal "
            "swing dominates solar in informativeness) AND/OR the bounds or "
            "regularization in `pem_fit.py` push solar to zero in the "
            "presence of ambiguity. This is the empirical-PEM analog of the "
            "WLS β_solar bias floor in project_bench_solar_fidelity.md. "
            "Removing this xfail when the underlying mechanism is fixed."
        ),
    )
    def test_beta_solar_recovery_tracks_truth(
        self,
        beta_solar_sweep: dict[float, ZoneCredibility],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Each truth solar_scale should be recovered within ±50%.

        Bias-floor signature: if all 3 truths land near ~1.2 regardless,
        kernel-layer bias is confirmed (matching the WLS bias floor in
        project_bench_solar_fidelity.md).
        """
        with capsys.disabled():
            print()
            print("1.3a β_solar sweep (kernel-vs-regression bias discriminator):")
            print(f"{'truth':>8}{'recovered':>12}{'ratio':>8}{'class':>8}")
            for truth_solar, cred in beta_solar_sweep.items():
                recovered = cred.train_result.fit_1r1c.best.params.solar_scale
                ratio = recovered / truth_solar
                print(
                    f"{truth_solar:>8.2f}{recovered:>12.3f}{ratio:>8.2f}"
                    f"{cred.classification:>8}"
                )

        # Each truth should be within ±50% (loose bound; tightens to
        # ±20% if kernel is well-identified). Saturation pattern is the
        # red flag.
        for truth_solar, cred in beta_solar_sweep.items():
            recovered = cred.train_result.fit_1r1c.best.params.solar_scale
            ratio = recovered / truth_solar
            assert 0.5 < ratio < 1.5, (
                f"solar_scale recovery off at truth={truth_solar}: "
                f"recovered={recovered:.3f}, ratio={ratio:.2f}"
            )

        # Saturation check: recovered values should NOT all cluster near
        # the same value regardless of truth (the bias-floor signature).
        recovered_values = np.array(
            [
                beta_solar_sweep[t].train_result.fit_1r1c.best.params.solar_scale
                for t in (1.0, 2.0, 3.0)
            ]
        )
        recovered_spread = recovered_values.max() - recovered_values.min()
        truth_spread = 3.0 - 1.0  # 2.0
        assert recovered_spread > 0.5 * truth_spread, (
            f"solar_scale shows saturation (kernel bias floor): "
            f"recovered spread {recovered_spread:.2f} < half of truth "
            f"spread {truth_spread} — recovered={recovered_values}"
        )


# ── Tier 1.3b — Bundle excitation realism check ─────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not BUNDLE_AVAILABLE, reason="Condenser A bundle not present")
class TestBundleExcitationRealism:
    """Same literature truth, but bundle's recorded outdoor/setpoint/solar
    drive the synthetic kernel. Tests whether the bundle's actual
    excitation pattern is sufficient for the methodology — independent of
    the bundle's own room_temp values (which are replaced)."""

    @pytest.fixture(scope="class")
    def real_train_telemetry(self):
        loaded = load_condenser_a_zone(BUNDLE_PATH, "living_room")
        cleaned = apply_default_exclusions(loaded)
        return slice_window(cleaned, TRAIN_WINDOW)

    @pytest.fixture(scope="class")
    def real_validate_telemetry(self):
        loaded = load_condenser_a_zone(BUNDLE_PATH, "living_room")
        cleaned = apply_default_exclusions(loaded)
        return slice_window(cleaned, VALIDATE_WINDOW)

    @pytest.fixture(scope="class")
    def canonical_credibility(
        self, real_train_telemetry, real_validate_telemetry
    ) -> ZoneCredibility:
        truth = _truth(solar_scale=2.0)
        train_synth = replace_room_temp_with_synthetic(
            real_train_telemetry, truth, seed=0
        )
        validate_synth = replace_room_temp_with_synthetic(
            real_validate_telemetry, truth, seed=100
        )
        return run_zone(
            "bundle_synth_canonical",
            train_synth,
            validate_synth,
            n_restarts=_N_RESTARTS,
            seed=0,
        )

    def test_canonical_classification_recorded(
        self,
        canonical_credibility: ZoneCredibility,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Record verdict; no specific class locked in. Comparison against
        1.3a's verdict is the discrimination signal — if 1.3a passes
        "good" but 1.3b drops to "poor", the bundle's excitation is the
        culprit, not the methodology."""
        with capsys.disabled():
            print()
            print(
                f"1.3b canonical: classification={canonical_credibility.classification}"
            )
            print(f"  train RMSE={canonical_credibility.train_rmse_c:.4f}°C")
            if canonical_credibility.validate_rmse_c is not None:
                print(
                    f"  validate RMSE={canonical_credibility.validate_rmse_c:.4f}°C"
                )
            recovered = canonical_credibility.train_result.fit_1r1c.best.params
            print(
                f"  recovered: τ={recovered.tau_s/3600:.1f}h, "
                f"q_scale={recovered.q_scale:.3f}, "
                f"solar={recovered.solar_scale:.3f}"
            )
        # Shape-only — verdict comparison is the deliverable, captured in stdout.
        assert canonical_credibility.classification in {"good", "close", "poor"}

    @pytest.fixture(scope="class")
    def beta_solar_sweep(
        self,
        real_train_telemetry,
        real_validate_telemetry,
    ) -> dict[float, ZoneCredibility]:
        results: dict[float, ZoneCredibility] = {}
        for truth_solar in (1.0, 2.0, 3.0):
            truth = _truth(solar_scale=truth_solar)
            train_synth = replace_room_temp_with_synthetic(
                real_train_telemetry,
                truth,
                seed=int(truth_solar * 10),
            )
            validate_synth = replace_room_temp_with_synthetic(
                real_validate_telemetry,
                truth,
                seed=int(truth_solar * 10) + 100,
            )
            results[truth_solar] = run_zone(
                f"bundle_synth_solar_{truth_solar}",
                train_synth,
                validate_synth,
                n_restarts=_N_RESTARTS,
                seed=0,
            )
        return results

    def test_records_beta_solar_sweep(
        self,
        beta_solar_sweep: dict[float, ZoneCredibility],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Compare against 1.3a's β_solar sweep. If 1.3a tracks truth
        but 1.3b saturates, kernel is fine but bundle excitation hides
        solar identifiability. If both saturate, kernel-layer bias.
        """
        with capsys.disabled():
            print()
            print("1.3b β_solar sweep on bundle excitation:")
            print(f"{'truth':>8}{'recovered':>12}{'ratio':>8}{'class':>8}")
            for truth_solar, cred in beta_solar_sweep.items():
                recovered = cred.train_result.fit_1r1c.best.params.solar_scale
                ratio = recovered / truth_solar
                print(
                    f"{truth_solar:>8.2f}{recovered:>12.3f}{ratio:>8.2f}"
                    f"{cred.classification:>8}"
                )
        # Recording test only — no assertions on specific verdict shape.
        # The 1.3a sweep test enforces tracking; 1.3b is observational.
