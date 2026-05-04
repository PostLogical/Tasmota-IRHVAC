"""Option A: linear-q_heat assumption adequacy under inverter HP physics.

The Phase 4 lite empirical pipeline assumes the heat-input column
`q_heat_proxy_w` enters the room dynamics linearly:

    dT/dt = (T_outdoor − T) / τ + (q_scale / C) · q_heat_proxy_w + …

Real Fujitsu mini-splits modulate compressor speed proportional to
(setpoint − room) and lose capacity at low outdoor temperatures
(NEEP cold-climate curve, FUJITSU_HYPERHEAT). Whether the empirical
linear proxy can absorb both nonlinearities — and at what cost — is the
Option A question deferred from `bench-id-discrimination` Tier 1 (see
`project_bench_id_discrimination.md`, "What's still open").

The control comparator is the existing Tier 1.3a synthetic positive
control (`test_synthetic_positive_control.py`): linear-kernel truth
recovers cleanly under both proxy variants. This test runs the same
literature τ through `ThermalModel`'s proportional inverter HP with the
FUJITSU_HYPERHEAT capacity curve and reports the resulting Phase 4 lite
verdicts under both proxy variants.

Marked @design — one-shot decision experiment, ~5-10 min wall-clock.
The captured output is the deliverable; assertions are shape-only.
"""

from __future__ import annotations

import pytest

from tests.hvac_bench.empirical.runner import (
    ZoneCredibility,
    run_zone,
)
from tests.hvac_bench.empirical.synthetic_drivers import (
    make_inverter_hp_zone_telemetry,
)
from tests.hvac_bench.house_profiles import (
    FUJITSU_HYPERHEAT_CAPACITY,
    HouseProfile2R2C,
)


# Literature τ to keep apples-to-apples with Tier 1.3a (Levermore 2020
# residential winter envelope: τ = 80 h). hp_gain matches PROFILES_2R2C
# living_room (0.04/min, mini-split sizing). The FUJITSU_HYPERHEAT capacity
# curve is what makes this an *inverter* HP truth, not a linear one.
# tau_couple/mass_ratio are unused by the 1R1C ThermalModel path.
_PROFILE = HouseProfile2R2C(
    name="LR-Fujitsu-Option-A-1R1C",
    tau_env=80 * 60,
    tau_couple=1.0,
    mass_ratio=1.0,
    hp_gain=0.04,
    hp_capacity=FUJITSU_HYPERHEAT_CAPACITY,
    description="1R1C-shaped 2R2C profile (tau_couple/mass_ratio unused) "
    "with FUJITSU_HYPERHEAT_CAPACITY for Option A linearization audit.",
)

_NOMINAL_HP_W = 1500.0
_N_TRAIN_STEPS = 18 * 24 * 12  # 5184, mirrors bundle train window length
_N_VALIDATE_STEPS = 8 * 24 * 12  # 2304, mirrors bundle validate window
_N_RESTARTS = 4


def _run_variant(variant: str) -> ZoneCredibility:
    train = make_inverter_hp_zone_telemetry(
        profile=_PROFILE,
        n_steps=_N_TRAIN_STEPS,
        nominal_capacity_w=_NOMINAL_HP_W,
        proxy_variant=variant,  # type: ignore[arg-type]
        seed=0,
    )
    validate = make_inverter_hp_zone_telemetry(
        profile=_PROFILE,
        n_steps=_N_VALIDATE_STEPS,
        nominal_capacity_w=_NOMINAL_HP_W,
        proxy_variant=variant,  # type: ignore[arg-type]
        seed=100,
    )
    return run_zone(
        f"inverter_hp_{variant}",
        train,
        validate,
        n_restarts=_N_RESTARTS,
        seed=0,
    )


@pytest.mark.design
class TestInverterHPLinearizationAudit:
    """Option A discrimination: linear-q_heat assumption vs proportional
    inverter HP physics with FUJITSU_HYPERHEAT capacity curve."""

    @pytest.fixture(scope="class")
    def credibility_constant(self) -> ZoneCredibility:
        return _run_variant("constant")

    @pytest.fixture(scope="class")
    def credibility_modulated(self) -> ZoneCredibility:
        return _run_variant("setpoint_modulated")

    def test_records_inverter_hp_verdicts(
        self,
        credibility_constant: ZoneCredibility,
        credibility_modulated: ZoneCredibility,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print Phase 4 lite verdicts for the inverter-HP truth under both
        proxy variants. Compared against the linear-truth control in
        `test_synthetic_positive_control.py::TestSyntheticExcitationPOC`.
        """
        with capsys.disabled():
            print()
            print("=" * 78)
            print("Option A: linear-q_heat adequacy under inverter HP physics")
            print("=" * 78)
            print(
                f"truth: ThermalModel 1R1C-shape, τ={_PROFILE.tau_env/60:.0f}h, "
                f"hp_gain={_PROFILE.hp_gain}/min, "
                f"hp_capacity=FUJITSU_HYPERHEAT (50% at -26°C, cutoff -32°C)"
            )
            print(
                f"compare to: Tier 1.3a linear-kernel truth at same τ "
                f"(passes 'good' or 'close')"
            )
            print()
            print(
                f"{'variant':<22}{'class':<8}"
                f"{'τ_h':>8}{'q_scale':>10}{'β_solar':>10}"
                f"{'rails':>7}{'cv_fail':>9}{'train_rmse':>12}{'val_rmse':>12}"
            )
            for label, cred in (
                ("constant", credibility_constant),
                ("setpoint_modulated", credibility_modulated),
            ):
                f1 = cred.train_result.fit_1r1c.best.params
                id_1 = cred.train_result.identifiability_1r1c
                val_rmse = (
                    f"{cred.validate_rmse_c:.3f}"
                    if cred.validate_rmse_c is not None
                    else "N/A"
                )
                print(
                    f"{label:<22}{cred.classification:<8}"
                    f"{f1.tau_s/3600:>8.1f}{f1.q_scale:>10.3f}"
                    f"{f1.solar_scale:>10.3f}"
                    f"{id_1.n_at_bound:>7}{id_1.n_failed_cv:>9}"
                    f"{cred.train_rmse_c:>12.3f}{val_rmse:>12}"
                )
            print()
            print(
                f"selected_model: constant="
                f"{credibility_constant.train_result.selected_model}, "
                f"setpoint_modulated="
                f"{credibility_modulated.train_result.selected_model}"
            )

        for cred in (credibility_constant, credibility_modulated):
            assert cred.classification in {"good", "close", "poor"}
