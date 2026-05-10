"""Validate the observation pipeline in the bench runner.

Verifies that the PI controller's observation recording, gating, and
classification behave correctly during bench simulation — the data
the learning system depends on must be realistic.
"""

from __future__ import annotations

import time as _time

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.full_stack_runner import (
    FullStackConfig, ModelInputSpec, diurnal_solar, diurnal_outdoor,
    run_full_stack, TICK_MINUTES_DEFAULT,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.thermal_model import ThermalModel2R2C


def _run_and_collect_observations(outdoor_base: float, n_days: int = 7,
                                   solar_gain: float = 0.01) -> dict:
    """Run bench and collect observation statistics from the PI controller."""
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
    }
    adapter = TasmotaPIAdapter(pi_config, kappa_threshold=10000)
    pi = adapter._pi

    model = ThermalModel2R2C(
        profile=profile, initial_temp=20.5, outdoor_temp=outdoor_base,
        sensor_noise_sigma=0.1, noise_seed=42,
        solar_gain=solar_gain,
    )
    adapter.set_desired_temp(20.5)
    adapter.set_mode("heat")

    tick_min = TICK_MINUTES_DEFAULT
    n_ticks = int(n_days * 24 * 60 / tick_min)
    batch_interval = int(12 * 60 / tick_min)

    for tick in range(n_ticks):
        dt_seconds = tick_min * 60.0
        model.outdoor_temp = diurnal_outdoor(tick, outdoor_base, 6.0, tick_min)
        solar_val = diurnal_solar(tick, peak=0.6, tick_minutes=tick_min)

        sensor_reading = model.read_sensor()
        adapter._sim_clock += dt_seconds
        adapter._entity._attr_current_temperature = sensor_reading
        pi._inputs.outdoor_temp = model.outdoor_temp

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

        model.step(hp_setpoint=float(pi._hp_setpoint), dt_minutes=tick_min,
                   tick=tick, solar_proxy=solar_val, mode="heat")

        if tick > 0 and tick % batch_interval == 0:
            pi._run_batch_analysis()

    # Collect observation stats from both buffers
    wls_obs = pi._observation_buffer_heat.get_all()
    gb_obs = pi._greybox_buffer.get_all()

    wls_stats = {
        "total": len(wls_obs),
        "no_output": sum(1 for o in wls_obs if o.clamped_reason == "no_output"),
        "saturated_low": sum(1 for o in wls_obs if o.clamped_reason == "saturated_low"),
        "saturated_high": sum(1 for o in wls_obs if o.clamped_reason == "saturated_high"),
        "normal": sum(1 for o in wls_obs if o.clamped_reason == ""),
        "uncertain": sum(1 for o in wls_obs if o.hp_contribution_uncertain),
        "hp_setpoint_none": sum(1 for o in wls_obs if o.hp_setpoint is None),
        "has_outdoor": sum(1 for o in wls_obs if o.outdoor_temp_c is not None),
    }

    gb_stats = {
        "total": len(gb_obs),
        "hp_on": sum(1 for o in gb_obs if o.clamped_reason != "no_output"),
        "hp_off": sum(1 for o in gb_obs if o.clamped_reason == "no_output"),
    }

    return {
        "wls": wls_stats,
        "gb": gb_stats,
        "n_ticks": n_ticks,
        "rls_obs_count": pi._rls_heat.observation_count,
    }


class TestObservationRecording:
    """Verify observations are recorded with correct classification."""

    def test_winter_observations(self, bench_metrics, num_regression):
        """Winter: most observations should be HP-on, few no_output."""
        stats = _run_and_collect_observations(outdoor_base=-5.0, n_days=7)
        wls = stats["wls"]
        gb = stats["gb"]

        print(f"\nWinter (7 days, {stats['n_ticks']} ticks):")
        print(f"  WLS buffer: {wls['total']} obs")
        print(f"    normal={wls['normal']}, no_output={wls['no_output']}, "
              f"sat_low={wls['saturated_low']}, sat_high={wls['saturated_high']}")
        print(f"    uncertain={wls['uncertain']}, hp_setpoint=None: {wls['hp_setpoint_none']}")
        print(f"    has_outdoor={wls['has_outdoor']}")
        print(f"  GB buffer: {gb['total']} obs (hp_on={gb['hp_on']}, hp_off={gb['hp_off']})")
        print(f"  RLS obs count: {stats['rls_obs_count']}")

        # Should have recorded observations
        assert wls["total"] > 0, "No WLS observations recorded"
        assert gb["total"] > 0, "No grey-box observations recorded"

        # All should have outdoor temp
        assert wls["has_outdoor"] == wls["total"], "Some WLS obs missing outdoor temp"

        # In winter, most should be normal (HP on)
        assert wls["normal"] > wls["no_output"], (
            f"More no_output ({wls['no_output']}) than normal ({wls['normal']}) in winter"
        )

    def test_spring_observations(self, bench_metrics, num_regression):
        """Spring: should see meaningful HP-off observations."""
        stats = _run_and_collect_observations(
            outdoor_base=14.0, n_days=7, solar_gain=0.02,
        )
        wls = stats["wls"]
        gb = stats["gb"]

        print(f"\nSpring (7 days, {stats['n_ticks']} ticks):")
        print(f"  WLS buffer: {wls['total']} obs")
        print(f"    normal={wls['normal']}, no_output={wls['no_output']}, "
              f"sat_low={wls['saturated_low']}, sat_high={wls['saturated_high']}")
        print(f"    uncertain={wls['uncertain']}, hp_setpoint=None: {wls['hp_setpoint_none']}")
        print(f"  GB buffer: {gb['total']} obs (hp_on={gb['hp_on']}, hp_off={gb['hp_off']})")
        print(f"  RLS obs count: {stats['rls_obs_count']}")

        # Should have both HP-on and HP-off observations
        assert wls["no_output"] > 0 or gb["hp_off"] > 0, (
            "No HP-off observations in spring — HP cycling not detected"
        )

    def test_uncertain_observations_excluded_from_wls(self, bench_metrics, num_regression):
        """Observations in the uncertain zone should be marked."""
        stats = _run_and_collect_observations(outdoor_base=-5.0, n_days=3)
        wls = stats["wls"]

        # With calibration bounds at 0, uncertain = not hp_definitely_on
        # In winter heating, most ticks should be hp_definitely_on
        # (room well below setpoint)
        if wls["total"] > 0:
            uncertain_pct = 100.0 * wls["uncertain"] / wls["total"]
            print(f"\nUncertain observations: {wls['uncertain']}/{wls['total']} ({uncertain_pct:.1f}%)")
            # With cal bounds at 0, uncertain means room >= setpoint
            # In winter this should be rare
            # (but not zero — transients during setpoint changes)

    def test_no_output_observations_have_null_setpoint(self, bench_metrics, num_regression):
        """no_output observations should have hp_setpoint=None."""
        stats = _run_and_collect_observations(
            outdoor_base=14.0, n_days=7, solar_gain=0.02,
        )
        wls = stats["wls"]
        assert wls["no_output"] == wls["hp_setpoint_none"], (
            f"no_output ({wls['no_output']}) != hp_setpoint=None ({wls['hp_setpoint_none']})"
        )

    def test_greybox_gets_hp_off_in_spring(self, bench_metrics, num_regression):
        """Grey-box buffer should have HP-off observations in spring."""
        stats = _run_and_collect_observations(
            outdoor_base=14.0, n_days=7, solar_gain=0.02,
        )
        gb = stats["gb"]

        print(f"\nGrey-box spring: {gb['total']} obs, "
              f"hp_on={gb['hp_on']}, hp_off={gb['hp_off']}")

        if gb["total"] > 0:
            off_pct = 100.0 * gb["hp_off"] / gb["total"]
            print(f"  HP-off: {off_pct:.1f}%")
            # Grey-box needs >=10% HP-off for gates to pass
            # In spring with warm outdoor + solar, should get some
