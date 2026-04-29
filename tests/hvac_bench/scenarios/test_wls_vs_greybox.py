"""Comparison: WLS-only vs grey-box-only vs WLS+grey-box fused.

Uses a spring scenario where outdoor temp is mild enough that the HP
cycles on/off naturally — essential for grey-box to have HP-off data
to separate ua_c from k_c.

The grey-box observer needs both HP-on (to identify k_c) and HP-off
(to identify ua_c) observations.  Pure winter heating rarely provides
HP-off data; spring/shoulder season does.

NOTE: this scenario is currently gated by the 1R1C structural limitation
(grey-box rails at τ=1000 → tau_plausible gate blocks every batch).
Until the 2R2C grey-box upgrade lands (future_work prompt #47), the
grey-box-only and fused arms produce no meaningful coefficient delta vs
WLS-only.  This file is the validation harness for #47, not a current
verdict producer.  See project_greybox_1r1c_limitation.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig, ModelInputSpec,
    diurnal_outdoor, diurnal_solar, run_full_stack,
    TICK_MINUTES_DEFAULT,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C

import time as _time


def _spring_outdoor(tick: int) -> float:
    """Late spring weather: warm base (16°C) with ±8°C diurnal + weather fronts.

    Afternoons regularly exceed desired (20.5°C). Combined with solar,
    HP should be off 30-50% of the time — matching production patterns.
    With 5-day weather drift ±8°C, warm spells push outdoor to 32°C.
    """
    return diurnal_outdoor(tick, base_c=16.0, amplitude_c=8.0)


@dataclass
class GreyboxComparisonResult:
    """Results from WLS vs grey-box comparison."""
    wls_only_comfort: float
    wls_only_integral_rms: float
    wls_only_od_error: float
    wls_only_solar_error: float
    wls_only_coef_trajectory: list[dict]

    gb_only_comfort: float
    gb_only_integral_rms: float
    gb_only_od_error: float
    gb_only_solar_error: float
    gb_only_coef_trajectory: list[dict]

    fused_comfort: float
    fused_integral_rms: float
    fused_od_error: float
    fused_solar_error: float
    fused_coef_trajectory: list[dict]

    gb_gates_passed_count: int
    gb_total_batches: int


def _run_spring_scenario(mode: str, n_days: int = 60) -> dict:
    """Run spring scenario with specified estimator mode.

    mode: "wls_only", "gb_only", "fused"
    """
    profile = PROFILES_2R2C["living_room"]

    pi_config = {
        "pi_model_inputs": [{
            "entity_id": "sensor.solar_proxy",
            "name": "Solar Proxy",
            "input_role": "solar",
            "seed_heat": 0.0,
            "seed_cool": 0.0,
            "lag_tau": 120,
            "delta_from_room": False,
            "clamp_min": 0,
        }],
        "pi_outdoor_seed_heat": profile.true_seed,
        "pi_outdoor_seed_cool": profile.true_seed,
        "pi_ki": 0.15,
        "pi_kp": 1.5,
        "pi_deadband": 0.5,
        "pi_setpoint_weight": 0.3,
        "pi_greybox_blending": (mode == "fused"),
    }

    adapter = TasmotaPIAdapter(pi_config)
    pi = adapter._pi
    pi._batch_kappa_threshold = 10000
    pi._rls_online_learning = False  # batch-only for all modes

    if mode == "gb_only":
        # Disable WLS coefficient writes — only grey-box influences model
        pi._greybox_blending_enabled = True
        # We'll monkey-patch to skip WLS apply but keep grey-box

    from tests.hvac_bench.thermal_model import ThermalModel2R2C
    model = ThermalModel2R2C(
        profile=profile,
        initial_temp=20.5,
        outdoor_temp=16.0,
        sensor_noise_sigma=0.1,
        noise_seed=42,
        solar_gain=0.06,  # strong solar — HP off 30-50% of the time
        stove_gain=0.0,
    )

    adapter.set_desired_temp(20.5)
    adapter.set_mode("heat")

    tick_min = TICK_MINUTES_DEFAULT
    n_ticks = int(n_days * 24 * 60 / tick_min)
    batch_interval = int(12 * 60 / tick_min)

    history = []
    coef_trajectory = []
    gb_gates_passed = 0
    gb_total = 0

    # Patch for gb_only mode: override batch apply
    if mode == "gb_only":
        _orig_run_batch = pi._run_batch_analysis

        def _gb_only_batch():
            """Run batch but don't apply WLS — only grey-box writes."""
            # Save current beta
            import copy
            beta_before = list(pi._rls_heat.beta)

            # Run normal batch (which does WLS + grey-box)
            _orig_run_batch()

            # If grey-box gates didn't pass, revert WLS changes
            if (pi._last_greybox_bridge is None or
                    not pi._last_greybox_bridge.gates_passed):
                # Revert to pre-batch state (no WLS, no grey-box)
                for i in range(len(beta_before)):
                    pi._rls_heat.beta[i] = beta_before[i]
            else:
                # Grey-box passed: apply only the grey-box estimate
                bridge = pi._last_greybox_bridge
                for i in range(min(len(bridge.beta), pi._rls_heat.n)):
                    if bridge.beta[i] is not None and math.isfinite(bridge.beta_std_err[i]):
                        pi._rls_heat.beta[i] = bridge.beta[i] * pi._rls_heat.feature_scales[i]

        pi._run_batch_analysis = _gb_only_batch

    for tick in range(n_ticks):
        dt_seconds = tick_min * 60.0
        model.outdoor_temp = _spring_outdoor(tick)
        solar_val = diurnal_solar(tick, peak=0.8)  # strong spring solar

        sensor_reading = model.read_sensor()

        adapter._sim_clock += dt_seconds
        adapter._entity._attr_current_temperature = sensor_reading
        pi._inputs.outdoor_temp = model.outdoor_temp

        # Mock solar entity
        _mock_states = {}
        ms = type("MockState", (), {
            "state": str(solar_val),
            "attributes": {"unit_of_measurement": None},
        })()
        _mock_states["sensor.solar_proxy"] = ms
        pi._hass.states.get = lambda eid, _s=_mock_states: _s.get(eid)

        original = _time.monotonic
        _time.monotonic = lambda: adapter._sim_clock
        try:
            adapter._loop.run_until_complete(pi._pi_tick())
        finally:
            _time.monotonic = original

        hp_setpoint = float(pi._hp_setpoint)
        model.step(hp_setpoint=hp_setpoint, dt_minutes=tick_min,
                   tick=tick, solar_proxy=solar_val, mode="heat")

        error = 20.5 - model.room_temp
        history.append({
            "tick": tick,
            "error": error,
            "hp_setpoint": hp_setpoint,
            "room_temp": model.room_temp,
        })

        # Trigger batch
        if tick > 0 and tick % batch_interval == 0:
            pi._run_batch_analysis()
            gb_total += 1
            if (pi._last_greybox_bridge is not None and
                    pi._last_greybox_bridge.gates_passed):
                gb_gates_passed += 1

            coefs = pi._rls_heat.get_coefficients()
            coef_trajectory.append({
                "outdoor_delta": coefs.get(1, 0),
                "Solar Proxy": coefs.get(2, 0) if pi._rls_heat.n > 2 else 0,
            })

    # Compute metrics
    in_band = sum(1 for h in history if abs(h["error"]) <= 0.5)
    comfort = 100.0 * in_band / len(history)
    integral_sq = sum(h["error"] ** 2 for h in history)
    integral_rms = math.sqrt(integral_sq / len(history))

    final_coefs = pi._rls_heat.get_coefficients()
    od_final = final_coefs.get(1, 0)
    solar_final = final_coefs.get(2, 0) if pi._rls_heat.n > 2 else 0

    # True values
    od_true = profile.true_seed  # in physical (positive) space
    solar_true = -2.0  # seed convention: negative = warms room

    return {
        "comfort": comfort,
        "integral_rms": integral_rms,
        "od_error": abs(od_final - (-od_true)),  # beta convention is negated
        "solar_error": abs(solar_final - solar_true),
        "coef_trajectory": coef_trajectory,
        "gb_gates_passed": gb_gates_passed,
        "gb_total": gb_total,
    }


class TestWLSvsGreybox:
    """Compare WLS-only, grey-box-only, and fused estimators."""

    def test_spring_comparison(self):
        """Spring scenario with HP cycling — grey-box should have data."""
        wls = _run_spring_scenario("wls_only", n_days=60)
        fused = _run_spring_scenario("fused", n_days=60)
        gb = _run_spring_scenario("gb_only", n_days=60)

        print(f"\n{'='*70}")
        print(f"  WLS vs Grey-box Comparison (Spring, 60 days)")
        print(f"{'='*70}")
        print(f"  Grey-box gates passed: WLS={wls['gb_gates_passed']}/{wls['gb_total']}, "
              f"Fused={fused['gb_gates_passed']}/{fused['gb_total']}, "
              f"GB-only={gb['gb_gates_passed']}/{gb['gb_total']}")
        print(f"\n{'Metric':<25} {'WLS Only':>10} {'GB Only':>10} {'Fused':>10}")
        print(f"{'-'*55}")
        print(f"{'Comfort %':<25} {wls['comfort']:>9.1f}% {gb['comfort']:>9.1f}% {fused['comfort']:>9.1f}%")
        print(f"{'Integral RMS':<25} {wls['integral_rms']:>10.3f} {gb['integral_rms']:>10.3f} {fused['integral_rms']:>10.3f}")
        print(f"{'|outdoor_delta err|':<25} {wls['od_error']:>10.4f} {gb['od_error']:>10.4f} {fused['od_error']:>10.4f}")
        print(f"{'|Solar err|':<25} {wls['solar_error']:>10.4f} {gb['solar_error']:>10.4f} {fused['solar_error']:>10.4f}")

        # Trajectory
        print(f"\n{'outdoor_delta trajectory (every 5th batch):'}")
        max_len = min(len(wls['coef_trajectory']), len(gb['coef_trajectory']), len(fused['coef_trajectory']))
        for i in range(0, max_len, 5):
            w = wls['coef_trajectory'][i].get('outdoor_delta', 0)
            g = gb['coef_trajectory'][i].get('outdoor_delta', 0)
            f = fused['coef_trajectory'][i].get('outdoor_delta', 0)
            print(f"  Batch {i+1:3d}: WLS={w:>8.4f}  GB={g:>8.4f}  Fused={f:>8.4f}")

        # Sanity
        assert wls['comfort'] > 50.0
        assert gb['comfort'] > 50.0
        assert fused['comfort'] > 50.0
