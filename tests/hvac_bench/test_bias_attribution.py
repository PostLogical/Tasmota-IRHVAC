"""Phase 1c — closed-loop bias attribution tests.

End-to-end comparison of closed-loop and open-loop β recovery on
identical thermal physics. The gap between them is the closed-loop
bias contribution per Forssell-Ljung 1999. This is the *measurement*
the rest of Phase 1 was building toward — every prior result about
"WLS β_solar = -1.2 vs truth -2.0" is unattributable until we know
how much of the gap is closed-loop.

Tests run a 30-day comparison on ``living_room``. Single-zone, single
profile is sufficient for Phase 1c — the question is "does the gap
exist", not "how does it scale". Multi-zone / multi-profile sweeps
are Phase 2 (reference controllers + locked KPIs).

Note on cost: the closed-loop arm runs the full PI controller with
RLS/WLS/buffer/health-checks. 30 days takes 5-15 seconds. Marked
``@pytest.mark.slow`` if a future runner-level marker appears; for now
the test runs unconditionally as part of bench fidelity.
"""

from __future__ import annotations

import math

import pytest

from tests.hvac_bench.bias_attribution import (
    BiasAttributionConfig,
    format_bias_report,
    run_bias_attribution,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C


# ── Single shared 30-day comparison fixture ──────────────────────────


@pytest.fixture(scope="module")
def living_room_attribution():
    """30-day comparison on ``living_room`` — the canonical test case.

    Both arms see identical outdoor + solar schedules, identical
    sensor-noise seed, identical thermal model. Only the controller
    differs (production PI vs scripted step probe).
    """
    config = BiasAttributionConfig(
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
    return run_bias_attribution(config)


# ── Smoke tests: both arms run, produce well-formed output ───────────


class TestBothArmsRun:

    def test_arms_produce_outdoor_delta_beta(self, living_room_attribution):
        # Both arms must report β for outdoor_delta — that's the minimum
        # output for the comparison to be meaningful.
        cl = living_room_attribution.closed_loop.beta
        ol = living_room_attribution.open_loop.beta
        assert "outdoor_delta" in cl, f"closed-loop missing outdoor_delta: {cl}"
        assert "outdoor_delta" in ol, f"open-loop missing outdoor_delta: {ol}"

    def test_betas_are_finite(self, living_room_attribution):
        for arm_name, arm in [
            ("closed_loop", living_room_attribution.closed_loop),
            ("open_loop", living_room_attribution.open_loop),
        ]:
            for name, val in arm.beta.items():
                assert math.isfinite(val), (
                    f"{arm_name}.{name}={val} is not finite"
                )

    def test_open_loop_identifiability_attached(self, living_room_attribution):
        # Open-loop arm must carry the IdentifiabilityReport from Phase 1b.
        ol_id = living_room_attribution.open_loop.identifiability
        assert ol_id.n_observations > 1000  # 30-day probe yields plenty
        assert ol_id.rank >= 2  # intercept + outdoor_delta identifiable
        assert ol_id.std_err_lower_bound[
            ol_id.feature_names.index("outdoor_delta")
        ] < 0.1


# ── The substantive Phase 1c claims ─────────────────────────────────


class TestClosedLoopBiasIsNonzero:
    """The gap between closed-loop and open-loop β must exist.

    If ``closed_loop_bias["outdoor_delta"] ≈ 0``, then closed-loop ID
    on this profile is bias-free — a quietly important result that
    would change the priority order for #34, #41, etc. If the bias is
    non-zero, then any closed-loop β estimate carries a known offset
    we now have a number for.

    Either way, the test is the *measurement*, not a pass/fail on
    correctness. Asserting ``|bias| > some-threshold`` codifies the
    finding that the bias is not negligible on this scenario.
    """

    def test_outdoor_delta_bias_is_measured(self, living_room_attribution):
        bias = living_room_attribution.closed_loop_bias["outdoor_delta"]
        # Print the report to stderr so it lands in test logs.
        print(format_bias_report(living_room_attribution))
        # Lower bound: the bias is at least 5× the open-loop CRLB-derived
        # standard error. If it were within the SE band, we couldn't
        # claim a real difference.
        ol_id = living_room_attribution.open_loop.identifiability
        outdoor_idx = ol_id.feature_names.index("outdoor_delta")
        ol_se = ol_id.std_err_lower_bound[outdoor_idx]
        assert abs(bias) > 5 * ol_se, (
            f"closed-loop bias |{bias:.4f}| is within 5σ of open-loop "
            f"SE ({ol_se:.4f}) — gap is not statistically resolvable; "
            f"this scenario can't attribute closed-loop bias"
        )

    def test_both_betas_in_physical_band(self, living_room_attribution):
        # Both estimators must produce physically plausible β, otherwise
        # the gap is meaningless. Living_room g·τ = 4 → asymptotic
        # closed-loop FF coefficient ≈ -0.25, open-loop ≈ -0.20.
        # Allow ±50% as a wide physical band to absorb finite-time
        # convergence and wall-mode lag.
        for arm_name, arm in [
            ("closed_loop", living_room_attribution.closed_loop),
            ("open_loop", living_room_attribution.open_loop),
        ]:
            beta = arm.beta["outdoor_delta"]
            assert -0.50 < beta < 0.0, (
                f"{arm_name}.outdoor_delta={beta:.4f} outside physical band; "
                f"controller or probe is broken"
            )


class TestBiasDirectionMatchesLiterature:
    """Heat-balance algebra predicts the closed-loop β is *more* negative
    than the open-loop β.

    For the 2R2C steady-state regression:
    - closed-loop (room ≈ desired): β_cl = -1/(g·τ)
    - open-loop (room free): β_ol = -1/(g·τ + 1)

    Since ``g·τ + 1 > g·τ``, ``|β_cl| > |β_ol|``, so β_cl < β_ol on the
    real line (both negative). This is a statement about the regression
    form's behaviour under closed loop, not a bias 'direction' claim
    about correctness — the closed-loop value is the physically-correct
    FF coefficient for tracking.
    """

    def test_closed_loop_more_negative_than_open_loop(
        self, living_room_attribution
    ):
        cl = living_room_attribution.closed_loop.beta["outdoor_delta"]
        ol = living_room_attribution.open_loop.beta["outdoor_delta"]
        profile = PROFILES_2R2C["living_room"]
        analytical_cl = -1.0 / (profile.hp_gain * profile.tau_env)
        analytical_ol = -1.0 / (profile.hp_gain * profile.tau_env + 1.0)
        assert cl < ol, (
            f"closed-loop β ({cl:.4f}) not more negative than open-loop "
            f"({ol:.4f}); literature predicts cl < ol "
            f"(analytical cl={analytical_cl:.4f}, ol={analytical_ol:.4f})"
        )
