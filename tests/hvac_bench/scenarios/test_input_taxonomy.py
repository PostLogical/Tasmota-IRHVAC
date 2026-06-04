"""Bench scenarios for the model-input taxonomy.

Maps four production realities to four distinct test classes:

1. **TestDirectActiveSource** — pellet-stove-style discrete on/off heat
   source.  Validates that an actively-firing source with independent
   timing unlocks via κ-gating, identifies its β toward truth, and
   detects its propagation lag.

2. **TestAdjacentZoneProxy** — single DR_temp aggregating modulating
   HP, occasional oil-zone bursts, outdoor coupling, and stochastic
   un-instrumented sources (oven, occupants).  Validates that a
   single aggregate proxy converges and supports comfort despite
   the mix.

3. **TestPurePassiveAdjacent** — adjacent zone temp that is a pure
   linear function of outdoor + solar (no independent variance).
   Validates that the κ-gate correctly keeps it frozen and that
   outdoor/solar coefficients absorb the passive coupling effect
   without comfort degradation.

4. **TestAnomalyRobustness** — realistic baseline with periodic
   window-open events (sustained heat-loss disturbances) and oven
   firings (heat-gain disturbances).  Neither is a model input.
   Validates that CUSUM excludes them and learning stays sound.
   Includes a tick-rate sweep (1, 5, 15 min) for the short events.

See ``project_input_taxonomy.md`` and ``project_bench_physics_fix.md``
in memory for the design rationale.
"""

from __future__ import annotations

import math

import pytest

from tests.hvac_bench.disturbances import Disturbance as ThermalDisturbance
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.constants import TICK_MINUTES_DEFAULT
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    ModelInputSpec,
    diurnal_solar,
    run_full_stack,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C


# ── Shared schedules (tick-rate aware) ──────────────────────────────────


def make_pellet_stove_schedule():
    """Pellet stove with varied firing hours for identifiability.

    Fires every other day, alternating between morning (7-10 AM) and
    evening (6-9 PM) cycles.  The alternating hours decorrelate stove
    timing from pure-evening or pure-morning patterns, giving the WLS
    independent variance to identify β.
    """
    def schedule(minute: float) -> float:
        hour = (minute / 60.0) % 24.0
        day = int(minute / (60.0 * 24.0))
        if day % 2 != 0:
            return 0.0
        # Alternate morning vs evening firing across pairs of stove days
        if (day // 2) % 2 == 0:
            return 1.0 if 18.0 <= hour < 21.0 else 0.0
        return 1.0 if 7.0 <= hour < 10.0 else 0.0
    return schedule


def make_dr_temp_schedule():
    """DR temperature aggregating multiple un-instrumented sources.

    Composition:
      - Modulating HP setpoint tracking (smooth, partly outdoor-coupled)
      - Occasional oil burst (every 5 days, 90 min, early morning)
      - Cooking event (every 3 days, 1 h, dinner time)
      - Outdoor coupling (light, 0.15 gradient)
      - Slow occupant-driven drift (deterministic, decoupled frequency)

    Returns absolute °C; consumed via ``delta_from_room=True`` in the
    model input config.
    """
    def schedule(minute: float) -> float:
        hour = (minute / 60.0) % 24.0
        day = minute / (60.0 * 24.0)
        d_int = int(day)

        # Modulating HP keeps DR around 20.5 ± wobble.  Wobble is a slow
        # function of recent outdoor (HP working harder when colder).
        outdoor = (
            -5.0
            + 6.0 * math.cos(2 * math.pi * (hour - 15) / 24)
            + 8.0 * math.sin(2 * math.pi * day / 5.0)
        )
        hp_response = 20.5 - 0.05 * (outdoor - 20.5)  # very mild HP-tracks-load

        # Light outdoor coupling through DR's own envelope
        outdoor_pull = 0.15 * (outdoor - 20.5)

        # Oil burst: every 5 days, 90 min in early morning
        oil = 0.0
        if d_int % 5 == 1 and 5.0 <= hour < 6.5:
            oil = 3.5

        # Cooking event: every 3 days, 1 h at dinner
        cooking = 0.0
        if d_int % 3 == 1 and 18.0 <= hour < 19.0:
            cooking = 1.5

        # Occupant-driven drift (decoupled frequency: ~2.7-day period
        # mixed with sub-hourly variation)
        drift = 0.4 * math.sin(2 * math.pi * day / 2.7) + 0.2 * math.sin(
            2 * math.pi * (hour * 0.7 + day * 1.1)
        )

        return hp_response + outdoor_pull + oil + cooking + drift
    return schedule


def make_passive_zone_schedule():
    """Pure-passive zone: deterministic linear function of outdoor + solar.

    No independent variance.  The WLS-from-thermal-physics view: this
    zone's temperature is fully predictable from the existing outdoor
    and solar features, so a regression on (outdoor, solar, passive)
    is rank-deficient and κ-gate should keep passive frozen.

    Light coupling coefficients (0.1 outdoor, 0.5 solar) keep the
    thermal-model contribution modest — frozen β=0 is still a decent
    approximation while outdoor catches up.  The test isn't about
    extreme heat-sink magnitudes; it's about identifiability.
    """
    def schedule(minute: float) -> float:
        hour = (minute / 60.0) % 24.0
        day = minute / (60.0 * 24.0)

        outdoor = (
            -5.0
            + 6.0 * math.cos(2 * math.pi * (hour - 15) / 24)
            + 8.0 * math.sin(2 * math.pi * day / 5.0)
        )

        solar_value = 0.0
        if 8.0 <= hour <= 18.0:
            sf = math.sin(math.pi * (hour - 8) / 10)
            cloud = 0.5 + 0.5 * math.cos(2 * math.pi * day / 3.0 + 1.0)
            solar_value = sf * cloud

        # Pure linear combination — no independent terms.
        return 20.5 + 0.1 * (outdoor - 20.5) + 0.5 * solar_value
    return schedule


# ── Scenario 1: Direct active source (pellet stove) ─────────────────────


@pytest.fixture(scope="module")
def _direct_active_source_result():
    """Run TestDirectActiveSource scenario ONCE; assertions consume the
    cached result. Replaces 4 independent run_full_stack calls (≈4×1040s)."""
    return run_full_stack(TestDirectActiveSource._make_config())


class TestDirectActiveSource:
    """Single pellet-stove-style discrete heat source.

    Validates the canonical model-input pipeline: source has independent
    activation timing, gives the WLS clean discrete signal, β converges
    and lag-tau auto-detects the propagation through wall mass.
    """

    @staticmethod
    def _make_config(n_days: int = 30) -> FullStackConfig:
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            model_inputs=[
                ModelInputSpec(
                    name="Pellet Stove",
                    entity_id="sensor.pellet_stove",
                    input_role="heat_source",
                    _true_ff_coef=-3.0,
                    seed_heat=0.0,
                    schedule=make_pellet_stove_schedule(),
                ),
            ],
            relax_kappa_gate=True,
        )

    def test_stove_unlocks_within_30d(self, bench_metrics, num_regression,
                                       _direct_active_source_result):
        """Stove model input should unlock by day 30."""
        result = _direct_active_source_result
        bench_metrics["final_stove_beta"] = result.final_coefs.get("Pellet Stove", 0.0)
        bench_metrics["n_batches"] = result.n_batches
        check_bench_metrics(num_regression, bench_metrics)
        if result.coef_trajectory:
            final = result.coef_trajectory[-1]
            assert final.get("Pellet Stove_frozen", True) is False, (
                "Stove never unlocked over 30 days"
            )

    def test_stove_beta_recovers_meaningful_magnitude(self, bench_metrics, num_regression,
                                                      _direct_active_source_result):
        """Stove β should reach at least 50% of true magnitude."""
        result = _direct_active_source_result
        beta = result.final_coefs.get("Pellet Stove", 0.0)
        bench_metrics["stove_beta"] = beta
        check_bench_metrics(num_regression, bench_metrics)
        # True is -3.0; require |β| ≥ 1.5 (half-magnitude, correct sign)
        assert beta < -1.5, (
            f"Stove β only {beta:.3f}, expected ≤ -1.5 (true -3.0)"
        )

    def test_stove_does_not_destabilize_outdoor(self, bench_metrics, num_regression,
                                                 _direct_active_source_result):
        """outdoor_delta should converge near the regression-sign truth."""
        result = _direct_active_source_result
        od = result.final_coefs.get("outdoor_delta", 0.0)
        expected = -PROFILES_2R2C["living_room"].true_seed
        bench_metrics["outdoor_delta"] = od
        bench_metrics["expected"] = expected
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(od - expected) < 0.1, (
            f"outdoor_delta diverged: {od:.3f} vs expected {expected:.3f}"
        )

    def test_comfort_above_85_pct(self, bench_metrics, num_regression,
                                   _direct_active_source_result):
        """Comfort should exceed 85% with active-source FF."""
        result = _direct_active_source_result
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)
        assert result.ctrl_comfort_pct >= 85.0, (
            f"Controllable comfort only {result.ctrl_comfort_pct:.1f}%"
        )


# ── Scenario 2: Adjacent zone proxy (DR_temp aggregate) ─────────────────


@pytest.fixture(scope="module")
def _adjacent_zone_proxy_result():
    """Run TestAdjacentZoneProxy scenario ONCE; assertions consume the
    cached result. Replaces 4 independent run_full_stack calls (≈4×1065s)."""
    return run_full_stack(TestAdjacentZoneProxy._make_config())


class TestAdjacentZoneProxy:
    """Single DR_temp input aggregating multiple un-instrumented sources.

    Most-realistic case: user has one temperature sensor in an adjacent
    zone (DR) that has its own modulating HP plus occasional oil burns,
    cooking events, and outdoor coupling.  Validates that the aggregate
    proxy converges to a meaningful β despite the mix.
    """

    @staticmethod
    def _make_config(n_days: int = 30) -> FullStackConfig:
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            model_inputs=[
                ModelInputSpec(
                    name="DR Temp",
                    entity_id="sensor.dr_temp",
                    input_role="adjacent_zone",
                    _true_ff_coef=-0.4,  # moderate party-wall coupling
                    seed_heat=0.0,
                    schedule=make_dr_temp_schedule(),
                    delta_from_room=True,
                ),
            ],
            relax_kappa_gate=True,
        )

    def test_dr_temp_unlocks_within_30d(self, bench_metrics, num_regression,
                                         _adjacent_zone_proxy_result):
        """DR_temp should unlock once oil/cooking events accumulate."""
        result = _adjacent_zone_proxy_result
        bench_metrics["dr_temp_beta"] = result.final_coefs.get("DR Temp", 0.0)
        bench_metrics["n_batches"] = result.n_batches
        check_bench_metrics(num_regression, bench_metrics)
        if result.coef_trajectory:
            final = result.coef_trajectory[-1]
            assert final.get("DR Temp_frozen", True) is False, (
                "DR_temp never unlocked despite mixed-source variance"
            )

    def test_dr_temp_beta_correct_sign(self, bench_metrics, num_regression,
                                        _adjacent_zone_proxy_result):
        """β should be negative (warmer DR → less HP needed in LR)."""
        result = _adjacent_zone_proxy_result
        beta = result.final_coefs.get("DR Temp", 0.0)
        bench_metrics["dr_temp_beta"] = beta
        check_bench_metrics(num_regression, bench_metrics)
        assert beta < 0, (
            f"DR_temp β has wrong sign: {beta:.3f}"
        )

    def test_outdoor_remains_correctly_signed(self, bench_metrics, num_regression,
                                               _adjacent_zone_proxy_result):
        """outdoor_delta should not flip sign or wildly diverge."""
        result = _adjacent_zone_proxy_result
        od = result.final_coefs.get("outdoor_delta", 0.0)
        bench_metrics["outdoor_delta"] = od
        check_bench_metrics(num_regression, bench_metrics)
        assert od < 0, f"outdoor_delta wrong sign: {od:.3f}"
        assert abs(od) < 1.0, f"outdoor_delta diverged: {od:.3f}"

    def test_comfort_above_80_pct(self, bench_metrics, num_regression,
                                   _adjacent_zone_proxy_result):
        """Comfort holds even with messy aggregate proxy."""
        result = _adjacent_zone_proxy_result
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)
        assert result.ctrl_comfort_pct >= 80.0, (
            f"Controllable comfort only {result.ctrl_comfort_pct:.1f}%"
        )


# ── Scenario 3: Pure passive adjacent zone ──────────────────────────────


@pytest.fixture(scope="module")
def _pure_passive_adjacent_result():
    """Run TestPurePassiveAdjacent scenario ONCE; assertions consume the
    cached result. Replaces 3 independent run_full_stack calls (≈3×1030s)."""
    return run_full_stack(TestPurePassiveAdjacent._make_config())


class TestPurePassiveAdjacent:
    """Adjacent zone whose temp is a pure linear function of outdoor+solar.

    Validates that κ-gate correctly identifies this as redundant and
    keeps it frozen indefinitely.  Validates that outdoor and solar
    coefficients absorb the passive coupling effect, producing correct
    FF predictions without the passive feature ever participating.
    """

    @staticmethod
    def _make_config(n_days: int = 30) -> FullStackConfig:
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            model_inputs=[
                ModelInputSpec(
                    name="Passive Zone",
                    entity_id="sensor.passive_zone",
                    input_role="adjacent_zone",
                    _true_ff_coef=-0.2,
                    seed_heat=0.0,
                    schedule=make_passive_zone_schedule(),
                    delta_from_room=True,
                ),
            ],
            # Default kappa gate (production threshold) — we want it active
            # to validate it correctly catches the redundancy.
            relax_kappa_gate=False,
        )

    def test_passive_zone_stays_frozen(self, bench_metrics, num_regression,
                                        _pure_passive_adjacent_result):
        """Pure-passive zone should remain frozen all 30 days."""
        result = _pure_passive_adjacent_result
        bench_metrics["n_snapshots"] = len(result.coef_trajectory)
        check_bench_metrics(num_regression, bench_metrics)
        # Check every snapshot — never unlocks
        for i, snap in enumerate(result.coef_trajectory):
            assert snap.get("Passive Zone_frozen", False) is True, (
                f"Passive zone unlocked at batch {i} — κ gate failed "
                f"to detect redundancy"
            )

    def test_outdoor_absorbs_passive_coupling(self, bench_metrics, num_regression,
                                               _pure_passive_adjacent_result):
        """outdoor_delta β may shift to absorb passive contribution."""
        result = _pure_passive_adjacent_result
        od = result.final_coefs.get("outdoor_delta", 0.0)
        bench_metrics["outdoor_delta"] = od
        check_bench_metrics(num_regression, bench_metrics)
        assert od < 0, f"outdoor_delta wrong sign: {od:.3f}"
        assert abs(od) < 1.0, f"outdoor_delta diverged: {od:.3f}"

    def test_comfort_holds_with_frozen_passive(self, bench_metrics, num_regression,
                                                _pure_passive_adjacent_result):
        """Comfort should not collapse despite frozen passive feature."""
        result = _pure_passive_adjacent_result
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)
        assert result.ctrl_comfort_pct >= 80.0, (
            f"Controllable comfort only {result.ctrl_comfort_pct:.1f}%"
        )


# ── Scenario 4: Anomaly robustness (windows + oven) ─────────────────────


@pytest.fixture(scope="module")
def _anomaly_with_events_result():
    """Run TestAnomalyRobustness with anomalies enabled ONCE; consumed by
    multiple test methods. Replaces 3 independent run_full_stack calls."""
    return run_full_stack(TestAnomalyRobustness._make_config(
        with_window_events=True, with_oven_events=True,
    ))


@pytest.fixture(scope="module")
def _anomaly_without_events_result():
    """Run TestAnomalyRobustness with anomalies disabled ONCE; consumed by
    multiple test methods as the no-anomaly baseline. Replaces 2 calls."""
    return run_full_stack(TestAnomalyRobustness._make_config(
        with_window_events=False, with_oven_events=False,
    ))


def _make_window_disturbances(n_days: int, tick_minutes: float = 15.0):
    """Generate periodic window-open events.

    Every ~3 days, a window opens for 60 min in the late morning.
    Modeled as a thermal disturbance with reduced tau_env (leakier
    envelope) — heat loss accelerates while the window is open.

    Note: ``tick_minutes`` parameter retained for API compatibility but
    is no longer needed — Disturbance fields are wall-clock minutes.
    """
    del tick_minutes  # no longer used; kept for backwards-compat callers
    minutes_per_day = 24 * 60
    disturbances = []
    for d in range(n_days):
        if d % 3 == 2:  # every 3 days
            # Late morning: 10 AM
            start_minute = d * minutes_per_day + 10 * 60
            disturbances.append(ThermalDisturbance(
                name=f"Window day {d}",
                heat_gain_c_per_min=0.0,
                tau_factor=0.4,  # 60% increase in heat loss
                start_minute=start_minute,
                duration_minutes=60.0,
                ramp_minutes=5.0,
            ))
    return disturbances


def _make_oven_disturbances(n_days: int, tick_minutes: float = 15.0):
    """Generate periodic oven events.

    Every 4 days at dinner time (6 PM), oven heats LR by ~1.5 kW for
    30 min.  Modeled as direct heat gain with no tau modification.
    """
    del tick_minutes
    minutes_per_day = 24 * 60
    disturbances = []
    for d in range(n_days):
        if d % 4 == 1:
            start_minute = d * minutes_per_day + 18 * 60
            disturbances.append(ThermalDisturbance(
                name=f"Oven day {d}",
                heat_gain_c_per_min=0.06,  # ~1.5°C over 30 min
                tau_factor=1.0,
                start_minute=start_minute,
                duration_minutes=30.0,
                ramp_minutes=5.0,
            ))
    return disturbances


class TestAnomalyRobustness:
    """Realistic baseline plus window-open and oven events.

    Neither anomaly is a model input.  Validates that the system's
    existing CUSUM-based exclusion + buffer behavior keeps β estimates
    sound through periodic disturbances that the user can't (or
    chooses not to) sensor.
    """

    @staticmethod
    def _make_config(n_days: int = 30, tick_minutes: float = 15.0,
                     with_window_events: bool = True,
                     with_oven_events: bool = True) -> FullStackConfig:
        thermal_disturbances = []
        if with_window_events:
            thermal_disturbances.extend(_make_window_disturbances(n_days, tick_minutes))
        if with_oven_events:
            thermal_disturbances.extend(_make_oven_disturbances(n_days, tick_minutes))
        return FullStackConfig(
            n_days=n_days,
            profile_name="living_room",
            outdoor_base_c=-5.0,
            outdoor_diurnal_c=6.0,
            desired_c=20.5,
            noise_sigma=0.1,
            noise_seed=42,
            tick_minutes=tick_minutes,
            model_inputs=[
                # Solar is the only real model input — outdoor handled implicitly.
                ModelInputSpec(
                    name="Solar Proxy",
                    entity_id="sensor.solar_proxy",
                    input_role="solar",
                    _true_ff_coef=-2.0,
                    seed_heat=0.0,
                    lag_tau=120,
                    clamp_min=0,
                    schedule=lambda m: diurnal_solar(m, peak=0.8),
                ),
            ],
            thermal_disturbances=thermal_disturbances,
            relax_kappa_gate=True,
        )

    def test_outdoor_survives_baseline_anomalies(self, bench_metrics, num_regression,
                                                  _anomaly_with_events_result,
                                                  _anomaly_without_events_result):
        """outdoor_delta β should not shift dramatically due to anomalies."""
        result_with = _anomaly_with_events_result
        result_without = _anomaly_without_events_result
        od_with = result_with.final_coefs.get("outdoor_delta", 0.0)
        od_without = result_without.final_coefs.get("outdoor_delta", 0.0)
        bench_metrics["od_with"] = od_with
        bench_metrics["od_without"] = od_without
        bench_metrics["shift"] = abs(od_with - od_without)
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(od_with - od_without) < 0.1, (
            f"outdoor_delta shifted by {abs(od_with - od_without):.3f} "
            f"due to anomalies (with={od_with:.3f}, without={od_without:.3f})"
        )

    def test_solar_survives_baseline_anomalies(self, bench_metrics, num_regression,
                                                _anomaly_with_events_result,
                                                _anomaly_without_events_result):
        """solar β should not collapse or sign-flip due to anomalies.

        Window-open events at 10 AM coincide with rising solar, creating
        a "high solar but room cooling" signal that drags β_solar toward
        zero.  We don't expect immunity — that's the realistic challenge
        the test is documenting.  We do require:
          - β stays correctly signed (negative)
          - magnitude doesn't collapse below 50% of the no-anomaly value

        If this assertion fails persistently, that's signal worth chasing
        (CUSUM tuning, exclusion logic, or per-window suppression).
        """
        result_with = _anomaly_with_events_result
        result_without = _anomaly_without_events_result
        s_with = result_with.final_coefs.get("Solar Proxy", 0.0)
        s_without = result_without.final_coefs.get("Solar Proxy", 0.0)
        bench_metrics["s_with"] = s_with
        bench_metrics["s_without"] = s_without
        check_bench_metrics(num_regression, bench_metrics)
        assert s_with < 0, f"Solar β sign-flipped under anomalies: {s_with:.3f}"
        assert abs(s_with) >= 0.5 * abs(s_without), (
            f"Solar β collapsed: |{s_with:.3f}| < 50% of "
            f"baseline |{s_without:.3f}|"
        )

    def test_comfort_holds_through_anomalies(self, bench_metrics, num_regression,
                                              _anomaly_with_events_result):
        """Comfort should remain reasonable despite periodic disturbances.

        Uses the with-events fixture — _make_config() defaults to both
        window and oven events on, which is what this test's
        "comfort despite anomalies" semantics need.
        """
        result = _anomaly_with_events_result
        bench_metrics["ctrl_comfort_pct"] = result.ctrl_comfort_pct
        check_bench_metrics(num_regression, bench_metrics)
        # Anomalies cause uncontrollable violations during events;
        # require ctrl_comfort_pct still ≥ 75%
        assert result.ctrl_comfort_pct >= 75.0, (
            f"Controllable comfort only {result.ctrl_comfort_pct:.1f}%"
        )

    @pytest.mark.parametrize("tick_minutes", [
        # 1-min ticks ⇒ 14 days × 1440 ticks = 20,160 PI iterations + full
        # RLS/WLS/buffer per tick; the run takes minutes, so it's a design
        # study (long-horizon coverage check), not regression.
        pytest.param(1.0, marks=pytest.mark.design, id="1min"),
        pytest.param(5.0, id="5min"),
        pytest.param(15.0, id="15min"),
    ])
    def test_anomaly_coverage_at_tick_rates(self, bench_metrics, num_regression, tick_minutes):
        """Tick-rate sweep: outdoor β stable across observation cadences.

        At fine ticks (1 min), short events produce many observations
        and CUSUM has plenty of evidence.  At 15-min ticks, the same
        events produce 1-4 observations and CUSUM may miss them.
        Test that β doesn't drift wildly across cadences.
        """
        # 14 days is enough for multiple anomaly cycles; full 30 days
        # at 1-min ticks runs slow.
        config = self._make_config(n_days=14, tick_minutes=tick_minutes)
        result = run_full_stack(config)
        od = result.final_coefs.get("outdoor_delta", 0.0)
        bench_metrics["tick_minutes"] = tick_minutes
        bench_metrics["outdoor_delta"] = od
        check_bench_metrics(num_regression, bench_metrics)
        # Coarse test: outdoor should not flip sign or wildly diverge
        assert od < 0, f"outdoor_delta wrong sign at {tick_minutes}min: {od:.3f}"
        assert abs(od) < 1.0, (
            f"outdoor_delta diverged at {tick_minutes}min: {od:.3f}"
        )
