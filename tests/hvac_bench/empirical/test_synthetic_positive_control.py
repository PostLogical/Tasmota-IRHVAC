"""Tier 1.3 synthetic positive-control: can Phase 4 lite recover known truth?

Two phases per the bench-validation discrimination plan:

  1.3a — Synthetic excitation POC. Open-loop synthetic inputs (PRBS-cycled
         q_heat, diurnal outdoor, bell-curve solar). Decoupled from bundle
         data; tests methodology reach on cleanest signals.

  1.3b — Bundle excitation realism check. Same literature truth, but with
         the bundle's recorded outdoor/setpoint/solar driving signals
         replacing the synthetic ones. Tests whether the bundle's
         excitation pattern is sufficient for the methodology.

Both use literature-grounded truth params (Levermore 2020 winter envelope:
τ=80h, sigma_v=0.05°C) with **C set to match `pem_fit.C_NOMINAL_1R1C = 1e7`**.
The 1R1C model only identifies τ = R·C (not R and C separately), and the
fit fixes C = C_nominal. If truth uses a different C, recovered q_scale
and solar_scale scale by C_nominal/C_truth — a parameterization artifact,
not a bench bug. (See _truth() docstring.)

solar_scale (β_solar) sweeps across {1.0, 2.0, 3.0} as an optimizer
self-consistency check on the solar dimension specifically.

Marked @slow — permanent regression property of the bench. ~15-20 min
wall-clock at n_restarts=4. Test failure means: bench can't recover its own
kernel + that excitation level + that proxy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.hvac_bench.conftest import check_bench_metrics

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
# sigma_v = 0.05°C (typical 5-min-averaged temperature sensor noise).
#
# IMPORTANT: C is set to match `pem_fit.C_NOMINAL_1R1C = 1e7 J/K`. The
# 1R1C model is underdetermined for separately identifying R and C from
# temperature dynamics — only τ = R·C is identifiable. The fit fixes
# C_nominal and identifies tau_s, q_scale, solar_scale relative to that
# C. If the synthetic truth uses a different C, recovered q_scale and
# solar_scale will scale by C_nominal/C_truth (a parameterization
# artifact, not a bench bug). Setting C_truth = C_nominal makes truth
# values directly comparable to recovered.
_TAU_S = 80.0 * 3600.0
_C = 1.0e7  # matches pem_fit.C_NOMINAL_1R1C — required for clean recovery comparison
_R = _TAU_S / _C  # = 0.0288 K/W
_NOMINAL_HP_W = 1500.0  # modulating mini-split typical


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
            truth,
            n_steps=_N_TRAIN_STEPS,
            seed=0,
            nominal_capacity_w=_NOMINAL_HP_W,
        )
        validate = make_synthetic_zone_telemetry(
            truth,
            n_steps=_N_VALIDATE_STEPS,
            seed=100,
            nominal_capacity_w=_NOMINAL_HP_W,
        )
        return run_zone(
            "synthetic_open_loop",
            train,
            validate,
            n_restarts=_N_RESTARTS,
            seed=0,
        )

    def test_canonical_truth_recovers_at_least_close_classification(
        self,
        bench_metrics,
        num_regression,
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

        _class_rank = {"good": 0, "close": 1, "poor": 2}
        bench_metrics["class_rank"] = _class_rank[canonical_credibility.classification]
        bench_metrics["train_rmse_c"] = canonical_credibility.train_rmse_c
        if canonical_credibility.validate_rmse_c is not None:
            bench_metrics["val_rmse_c"] = canonical_credibility.validate_rmse_c
        check_bench_metrics(num_regression, bench_metrics)

    def test_canonical_recovers_tau_within_factor_of_2(
        self,
        bench_metrics,
        num_regression,
        canonical_credibility: ZoneCredibility,
    ) -> None:
        """Recovered τ should be within a factor of 2 of truth (80h)."""
        recovered = canonical_credibility.train_result.fit_1r1c.best.params
        ratio = recovered.tau_s / _TAU_S
        assert 0.5 < ratio < 2.0, (
            f"τ recovery off: truth={_TAU_S/3600:.1f}h, "
            f"recovered={recovered.tau_s/3600:.1f}h (ratio={ratio:.2f})"
        )

        bench_metrics["tau_h"] = recovered.tau_s / 3600.0
        bench_metrics["tau_ratio"] = ratio
        check_bench_metrics(num_regression, bench_metrics)

    @pytest.fixture(scope="class")
    def beta_solar_sweep(self) -> dict[float, ZoneCredibility]:
        """Run 3 fits across solar_scale truth ∈ {1.0, 2.0, 3.0} with C
        consistent with the optimizer's C_nominal. Recovered solar_scale
        should track truth within ±20%.
        """
        results: dict[float, ZoneCredibility] = {}
        for truth_solar in (1.0, 2.0, 3.0):
            truth = _truth(solar_scale=truth_solar)
            train = make_synthetic_zone_telemetry(
                truth,
                n_steps=_N_TRAIN_STEPS,
                seed=int(truth_solar * 10),
                nominal_capacity_w=_NOMINAL_HP_W,
            )
            validate = make_synthetic_zone_telemetry(
                truth,
                n_steps=_N_VALIDATE_STEPS,
                seed=int(truth_solar * 10) + 100,
                nominal_capacity_w=_NOMINAL_HP_W,
            )
            results[truth_solar] = run_zone(
                f"synth_solar_{truth_solar}",
                train,
                validate,
                n_restarts=_N_RESTARTS,
                seed=0,
            )
        return results

    def test_beta_solar_recovery_tracks_truth(
        self,
        bench_metrics,
        num_regression,
        beta_solar_sweep: dict[float, ZoneCredibility],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Each truth solar_scale should be recovered within ±25% with
        C-consistent truth params. Verifies optimizer self-consistency
        on the solar dimension specifically.
        """
        with capsys.disabled():
            print()
            print("1.3a β_solar sweep (optimizer self-consistency check):")
            print(f"{'truth':>8}{'recovered':>12}{'ratio':>8}{'class':>8}")
            for truth_solar, cred in beta_solar_sweep.items():
                recovered = cred.train_result.fit_1r1c.best.params.solar_scale
                ratio = recovered / truth_solar
                print(
                    f"{truth_solar:>8.2f}{recovered:>12.3f}{ratio:>8.2f}"
                    f"{cred.classification:>8}"
                )

        for truth_solar, cred in beta_solar_sweep.items():
            recovered = cred.train_result.fit_1r1c.best.params.solar_scale
            ratio = recovered / truth_solar
            assert 0.75 < ratio < 1.25, (
                f"solar_scale recovery off at truth={truth_solar}: "
                f"recovered={recovered:.3f}, ratio={ratio:.2f}"
            )

        _class_rank = {"good": 0, "close": 1, "poor": 2}
        for truth_solar, cred in beta_solar_sweep.items():
            recovered = cred.train_result.fit_1r1c.best.params.solar_scale
            bench_metrics[f"truth_{truth_solar}__recovered"] = recovered
            bench_metrics[f"truth_{truth_solar}__ratio"] = recovered / truth_solar
            bench_metrics[f"truth_{truth_solar}__class_rank"] = _class_rank[
                cred.classification
            ]
        check_bench_metrics(num_regression, bench_metrics)


# ── Tier 1.3b — Bundle excitation realism check ─────────────────────────


@pytest.mark.design
@pytest.mark.skipif(not BUNDLE_AVAILABLE, reason="Condenser A bundle not present")
class TestBundleExcitationRealism:
    """Same literature truth, but bundle's recorded outdoor/setpoint/solar
    drive the synthetic kernel. Tests whether the bundle's actual
    excitation pattern is sufficient for the methodology — independent of
    the bundle's own room_temp values (which are replaced).

    Marked ``design + study`` (not ``slow``): both tests are recording-only.
    ``test_canonical_classification_recorded`` only asserts ``classification
    in {good, close, poor}`` (any value passes); ``test_records_beta_solar_sweep``
    is documented as having no real assertion. The discrimination signal vs
    1.3a (TestSyntheticExcitationPOC) lives in the printed verdict tables,
    not in regression assertions. Opt in via ``--run-studies``."""

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
        bench_metrics,
        num_regression,
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

        _class_rank = {"good": 0, "close": 1, "poor": 2}
        recovered = canonical_credibility.train_result.fit_1r1c.best.params
        bench_metrics["class_rank"] = _class_rank[canonical_credibility.classification]
        bench_metrics["train_rmse_c"] = canonical_credibility.train_rmse_c
        if canonical_credibility.validate_rmse_c is not None:
            bench_metrics["val_rmse_c"] = canonical_credibility.validate_rmse_c
        bench_metrics["tau_h"] = recovered.tau_s / 3600.0
        bench_metrics["q_scale"] = recovered.q_scale
        bench_metrics["solar_scale"] = recovered.solar_scale
        check_bench_metrics(num_regression, bench_metrics)

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
        bench_metrics,
        num_regression,
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
        _class_rank = {"good": 0, "close": 1, "poor": 2}
        for truth_solar, cred in beta_solar_sweep.items():
            recovered = cred.train_result.fit_1r1c.best.params.solar_scale
            bench_metrics[f"truth_{truth_solar}__recovered"] = recovered
            bench_metrics[f"truth_{truth_solar}__ratio"] = recovered / truth_solar
            bench_metrics[f"truth_{truth_solar}__class_rank"] = _class_rank[
                cred.classification
            ]
        check_bench_metrics(num_regression, bench_metrics)
