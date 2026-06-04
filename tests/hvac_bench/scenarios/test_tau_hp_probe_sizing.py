"""Closed-loop τ_hp identifiability under active-probe excitation.

Stage 1f (2026-06-03). Counterpart to ``test_greybox_tau_hp_open_loop.py``
which used hand-crafted sinusoidal swings. This test uses the real PI
controller + production greybox observer in closed loop, with the
production auto-perturb state machine firing probes when steady state
is detected (research mode bypasses the convergence gate so probes
fire often).

**Question this test answers**:
- How many probes does it take for σ(τ_hp) to drop below a useful
  threshold (e.g. <30% of value)?
- Does each additional probe add diminishing returns, or is there a
  sharp identifiability knee?
- Do the same probes also tighten k_c and ua_c (cost/benefit
  ratio for active probing)?

**What this DOESN'T answer**:
- Whether the auto-perturb's natural probe shape is OPTIMAL for τ_hp
  (it was designed for plant-ID τ_slow, not HP-side dynamics)
- Whether multi-zone interaction changes anything (single zone here)
- Whether the closed-loop result holds across multiple seasons /
  weather patterns

Designed to be informative for the active-probe-design conversation
([[feedback_response_intensity_spectrum]] — when does STRONG fire vs
SOFT vs DEFER) without committing to a specific design.
"""

from __future__ import annotations

import math

import pytest

from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.full_stack_runner import (
    Checkpoint,
    CheckpointState,
    FullStackConfig,
    run_full_stack,
)


# Bench Q_hp lag chosen to be in the middle of TAU_HP_BOUNDS (1, 30) and
# match the production prior (3 min mean, 4 min σ). True value 5 is within
# one prior-σ of the mean — well-positioned for the data to pull the
# posterior off the prior when identifiability is good.
_BENCH_TAU_HP_MIN = 5.0


def _probe_config(*, n_days: int) -> FullStackConfig:
    """Spring-shoulder config tuned for frequent steady-state detection.

    Narrow diurnal swing + cooler base = HP often reaches setpoint and
    sits near steady → auto-perturb (research mode) fires frequently.
    """
    return FullStackConfig(
        n_days=n_days,
        profile_name="standard_residential_fujitsu",
        outdoor_base_c=5.0,
        outdoor_diurnal_c=3.0,
        desired_c=20.5,
        noise_sigma=0.1,
        noise_seed=42,
        head_sensor_offset=0.5,
        head_calibration_bounds=None,
        relax_kappa_gate=True,
        tau_hp_minutes=_BENCH_TAU_HP_MIN,
        pi_overrides={
            "pi_auto_perturb_enabled": True,
            "pi_auto_perturb_research_mode": True,
            "pi_hp_capacity_profile": "fujitsu_aou24rlxfwh",
        },
    )


def _snapshot_from(state: CheckpointState) -> dict:
    """Extract a probe/τ_hp snapshot from CheckpointState.

    Reads via getattr to be resilient to PI controller internals
    changing structure — the test's snapshot is best-effort, not a
    structural contract."""
    pi = state.pi
    probe_state = getattr(pi, "_regime_probe", None)
    probes_completed = getattr(probe_state, "_probes_completed", None) if probe_state else None
    gb = getattr(pi, "_last_greybox_result", None)
    if gb is None:
        return {
            "elapsed_days": state.elapsed_days,
            "probes_completed": probes_completed,
            "tau_hp": None,
            "tau_hp_se": None,
            "k_c": None,
            "k_c_se": None,
            "ua_c": None,
            "ua_c_se": None,
            "is_2r2c": False,
        }
    se = getattr(gb, "param_std_err", None) or {}
    return {
        "elapsed_days": state.elapsed_days,
        "probes_completed": probes_completed,
        "tau_hp": getattr(gb, "tau_hp", None),
        "tau_hp_se": se.get("tau_hp"),
        "k_c": getattr(gb, "k_c", None),
        "k_c_se": se.get("k_c"),
        "ua_c": getattr(gb, "ua_c", None),
        "ua_c_se": se.get("ua_c"),
        "is_2r2c": getattr(gb, "is_2r2c", False),
    }


class TestTauHpProbeSizing:
    """Measure τ_hp identifiability curve as probes accumulate."""

    @pytest.mark.slow
    def test_probe_sizing_curve(self, bench_metrics, num_regression):
        """Run 30 days with auto-perturb research mode + bench tau_hp=5.
        Checkpoint every 3 days, capturing (probes_completed, τ_hp,
        σ(τ_hp), k_c, σ(k_c), ua_c, σ(ua_c)).

        Reports the full snapshot table to bench_metrics for review.
        Asserts only directional minima — this test is more about
        gathering evidence than enforcing a tight numerical claim.

        Per [[project-bench-perf-30day-slow]] this run takes 5-10 minutes
        on current bench perf. Optimization is deferred future work.
        """
        config = _probe_config(n_days=30)
        snapshots: list[dict] = []

        def _callback(state: CheckpointState) -> None:
            snapshots.append(_snapshot_from(state))

        result = run_full_stack(
            config,
            checkpoints=[Checkpoint(interval_days=3, callback=_callback)],
        )

        # Flatten snapshot trajectory into bench_metrics for regression
        # capture — the curve IS the result; reviewers should inspect the
        # baseline CSV to see how τ_hp / σ(τ_hp) evolve vs probes_completed.
        for i, snap in enumerate(snapshots):
            for k, v in snap.items():
                if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
                    bench_metrics[f"snap{i:02d}_{k}"] = float(v)
                elif v is None:
                    bench_metrics[f"snap{i:02d}_{k}"] = float("nan")
        bench_metrics["final_probes"] = float(
            snapshots[-1]["probes_completed"] or 0
        )
        bench_metrics["final_tau_hp"] = float(snapshots[-1]["tau_hp"] or 0.0)
        bench_metrics["final_tau_hp_se"] = float(snapshots[-1]["tau_hp_se"] or float("nan"))
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)

        # ── Directional assertions ──
        # We don't claim that τ_hp identifies tightly — that's the
        # question we're investigating. We DO claim that the experiment
        # ran cleanly:

        # 1. Greybox dispatched to 2R2C at least once (otherwise the test
        #    didn't actually exercise the 3-state path)
        any_2r2c = any(s["is_2r2c"] for s in snapshots)
        assert any_2r2c, (
            "Greybox should have dispatched to 2R2C during the run; "
            "snapshots show no 2R2C result. Probe-sizing data is "
            "uninformative without 3-state fits."
        )

        # 2. At least one probe fired — otherwise we're just testing the
        #    passive case (which the open-loop test already covers)
        max_probes = max((s["probes_completed"] or 0) for s in snapshots)
        assert max_probes >= 1, (
            f"Auto-perturb research mode should fire at least 1 probe in "
            f"30 days; got {max_probes}. Check config tuning."
        )

        # 3. Comfort survived — probes shouldn't wreck the user experience
        #    (this also matches the active-probe-design constraint: don't
        #    inconvenience the user for no gain)
        assert result.ctrl_comfort_pct >= 55.0, (
            f"Comfort should survive probes, got {result.ctrl_comfort_pct:.1f}%"
        )
