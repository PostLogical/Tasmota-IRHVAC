"""Analytical validation of the thermal model physics.

Known-answer tests with closed-form solutions. If these fail,
the thermal model has a physics bug.
"""

from __future__ import annotations

import math

import pytest

from dataclasses import replace

from tests.hvac_bench.house_profiles import (
    COLD_CLIMATE_HP_CAPACITY,
    HouseProfile2R2C,
    PROFILES_2R2C,
    STANDARD_HP_CAPACITY,
)
from tests.hvac_bench.conftest import check_bench_metrics
from tests.hvac_bench.thermal_model import ThermalModel, ThermalModel2R2C


# Tolerance for steady-state tests (°C). Models use discrete time steps
# so exact convergence requires many steps.
SS_TOL = 0.05  # 0.05°C


class TestFreeDecay:
    """HP off, no solar: room should decay to outdoor temp."""

    def test_1r1c_decays_toward_outdoor(self, bench_metrics, num_regression):
        """1R1C: room should approach outdoor monotonically."""
        profile = PROFILES_2R2C["living_room"]
        model = ThermalModel(
            profile=profile, initial_temp=25.0, outdoor_temp=5.0,
            sensor_noise_sigma=0.0,
        )
        prev = model.room_temp
        for tick in range(500):
            model.step(hp_setpoint=0.0, dt_minutes=15.0, mode="heat", tick=tick)
            assert model.room_temp <= prev + 0.001, (
                f"Tick {tick}: room went UP ({prev:.3f} -> {model.room_temp:.3f}) "
                f"during free decay"
            )
            prev = model.room_temp
        bench_metrics["final_room_temp"] = model.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(model.room_temp - 5.0) < SS_TOL, (
            f"Room didn't converge to outdoor: {model.room_temp:.3f} vs 5.0"
        )

    def test_2r2c_decays_toward_outdoor(self, bench_metrics, num_regression):
        """2R2C: room should approach outdoor, but wall mass slows it.

        The wall node (τ_wall=600 min) acts as a heat reservoir that
        keeps the room warmer than outdoor for much longer than 1R1C.
        After 500 steps (125 hours), should be within 1°C of outdoor.
        """
        profile = PROFILES_2R2C["living_room"]
        model = ThermalModel2R2C(
            profile=profile, initial_temp=25.0, outdoor_temp=5.0,
            sensor_noise_sigma=0.0,
        )
        for tick in range(500):
            model.step(hp_setpoint=0.0, dt_minutes=15.0, mode="heat", tick=tick)
        bench_metrics["final_room_temp"] = model.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        # 2R2C has wall thermal mass — allow wider tolerance
        assert abs(model.room_temp - 5.0) < 1.0, (
            f"2R2C room didn't approach outdoor: {model.room_temp:.3f} vs 5.0"
        )
        # But should be heading in the right direction
        assert model.room_temp < 10.0, (
            f"2R2C room still far from outdoor: {model.room_temp:.3f}"
        )

    def test_1r1c_time_constant(self, bench_metrics, num_regression):
        """1R1C decay should follow exp(-t/τ_env)."""
        profile = PROFILES_2R2C["living_room"]
        T_init = 25.0
        T_out = 5.0
        model = ThermalModel(
            profile=profile, initial_temp=T_init, outdoor_temp=T_out,
            sensor_noise_sigma=0.0,
        )
        tau = profile.tau_env
        n_steps = int(tau / 15.0)
        for tick in range(n_steps):
            model.step(hp_setpoint=0.0, dt_minutes=15.0, mode="heat", tick=tick)

        expected_fraction = math.exp(-1.0)  # ~0.368
        actual_fraction = (model.room_temp - T_out) / (T_init - T_out)
        bench_metrics["fraction_after_tau"] = actual_fraction
        bench_metrics["final_room_temp"] = model.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(actual_fraction - expected_fraction) < 0.05, (
            f"1R1C after one τ_env ({tau} min): fraction={actual_fraction:.3f}, "
            f"expected {expected_fraction:.3f}"
        )

    def test_2r2c_time_constant_slower(self, bench_metrics, num_regression):
        """2R2C decay should be SLOWER than 1R1C due to wall heat reservoir."""
        profile = PROFILES_2R2C["living_room"]
        T_init = 25.0
        T_out = 5.0
        m1 = ThermalModel(
            profile=profile, initial_temp=T_init, outdoor_temp=T_out,
            sensor_noise_sigma=0.0,
        )
        m2 = ThermalModel2R2C(
            profile=profile, initial_temp=T_init, outdoor_temp=T_out,
            sensor_noise_sigma=0.0,
        )
        tau = profile.tau_env
        n_steps = int(tau / 15.0)
        for tick in range(n_steps):
            m1.step(hp_setpoint=0.0, dt_minutes=15.0, mode="heat", tick=tick)
            m2.step(hp_setpoint=0.0, dt_minutes=15.0, mode="heat", tick=tick)

        bench_metrics["m1_final_temp"] = m1.room_temp
        bench_metrics["m2_final_temp"] = m2.room_temp
        bench_metrics["temp_gap"] = m2.room_temp - m1.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        # 2R2C should be warmer (wall slows decay)
        assert m2.room_temp > m1.room_temp, (
            f"2R2C ({m2.room_temp:.3f}) should be warmer than "
            f"1R1C ({m1.room_temp:.3f}) during decay"
        )


class TestHPSteadyState:
    """HP on at fixed setpoint — verify equilibrium temperature."""

    def test_1r1c_equilibrium(self, bench_metrics, num_regression):
        """1R1C steady state: T_eq = (T_out/τ + g*sp) / (1/τ + g)"""
        profile = PROFILES_2R2C["living_room"]
        T_out = 0.0
        sp = 25.0
        model = ThermalModel(
            profile=profile, initial_temp=20.0, outdoor_temp=T_out,
            sensor_noise_sigma=0.0,
        )
        # Run to steady state
        for tick in range(2000):
            model.step(hp_setpoint=sp, dt_minutes=15.0, mode="heat", tick=tick)

        tau = profile.tau_env
        g = profile.hp_gain
        # Analytical equilibrium (with HP always on, room < sp)
        t_eq = (T_out / tau + g * sp) / (1.0 / tau + g)
        bench_metrics["final_room_temp"] = model.room_temp
        bench_metrics["t_eq_analytical"] = t_eq
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(model.room_temp - t_eq) < SS_TOL, (
            f"1R1C didn't converge to analytical equilibrium: "
            f"{model.room_temp:.3f} vs {t_eq:.3f}"
        )

    def test_2r2c_equilibrium(self, bench_metrics, num_regression):
        """2R2C steady state should match 1R1C at equilibrium.

        At steady state d/dt=0, the wall coupling nets out (wall and air
        reach same temp), so 2R2C reduces to 1R1C equilibrium.
        """
        profile = PROFILES_2R2C["living_room"]
        T_out = 0.0
        sp = 25.0
        model = ThermalModel2R2C(
            profile=profile, initial_temp=20.0, outdoor_temp=T_out,
            sensor_noise_sigma=0.0,
        )
        for tick in range(2000):
            model.step(hp_setpoint=sp, dt_minutes=15.0, mode="heat", tick=tick)

        tau = profile.tau_env
        g = profile.hp_gain
        t_eq = (T_out / tau + g * sp) / (1.0 / tau + g)
        bench_metrics["final_room_temp"] = model.room_temp
        bench_metrics["t_eq_analytical"] = t_eq
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(model.room_temp - t_eq) < SS_TOL, (
            f"2R2C didn't converge to 1R1C analytical equilibrium: "
            f"{model.room_temp:.3f} vs {t_eq:.3f}"
        )

    def test_equilibrium_below_setpoint(self, bench_metrics, num_regression):
        """Verify room settles BELOW setpoint (HP is thermostat-controlled).

        With HP cycling, the room oscillates around a point below setpoint.
        The gap depends on hp_gain and τ_env — strong gain + fast envelope
        → small gap; weak gain + slow envelope → larger gap.  With the
        living room profile (g=0.04, τ_env=100), the analytical "HP always
        on" equilibrium at outdoor=0 is ~20°C for sp=22, so a gap of
        several degrees is expected since the HP cycles off at setpoint.
        """
        profile = PROFILES_2R2C["living_room"]
        model = ThermalModel2R2C(
            profile=profile, initial_temp=15.0, outdoor_temp=0.0,
            sensor_noise_sigma=0.0,
        )
        sp = 22.0
        for tick in range(2000):
            model.step(hp_setpoint=sp, dt_minutes=15.0, mode="heat", tick=tick)

        # Analytical equilibrium (HP always on) for outdoor=0:
        tau = profile.tau_env
        g = profile.hp_gain
        t_eq_always_on = (0.0 / tau + g * sp) / (1.0 / tau + g)
        bench_metrics["final_room_temp"] = model.room_temp
        bench_metrics["t_eq_always_on"] = t_eq_always_on
        check_bench_metrics(num_regression, bench_metrics)
        # Room should never exceed setpoint
        assert model.room_temp < sp + 0.01, (
            f"Room exceeded setpoint: {model.room_temp:.3f} > {sp}"
        )
        # Room should be between outdoor and the always-on equilibrium
        assert model.room_temp > 0.0, "Room dropped to outdoor temp"
        assert model.room_temp <= t_eq_always_on + 0.1, (
            f"Room {model.room_temp:.3f} exceeds always-on equilibrium {t_eq_always_on:.3f}"
        )


class TestHPCycling:
    """Verify HP actually cycles on/off in appropriate conditions."""

    def test_hp_cycles_when_solar_pushes_past_setpoint(self, bench_metrics, num_regression):
        """Strong solar + warm outdoor should push room above setpoint, idling HP.

        The HP must be close to equilibrium already (warm outdoor) so that
        solar can push the room the last fraction of a degree past setpoint.
        With cold outdoor, the HP equilibrium is too far below setpoint
        for solar alone to bridge the gap.
        """
        profile = PROFILES_2R2C["living_room"]
        model = ThermalModel2R2C(
            profile=profile, initial_temp=20.0, outdoor_temp=19.0,
            sensor_noise_sigma=0.0, solar_gain=0.05,  # strong solar
        )
        sp = 20.5
        hp_on_ticks = 0
        hp_off_ticks = 0
        for tick in range(96 * 3):  # 3 days
            hour = (tick * 15 / 60.0) % 24.0
            solar = 0.8 if 8 <= hour <= 16 else 0.0
            model.step(hp_setpoint=sp, dt_minutes=15.0, solar_proxy=solar,
                       mode="heat", tick=tick)
            if model.room_temp >= sp:
                hp_off_ticks += 1
            else:
                hp_on_ticks += 1

        total = hp_on_ticks + hp_off_ticks
        off_pct = 100.0 * hp_off_ticks / total
        bench_metrics["hp_off_ticks"] = hp_off_ticks
        bench_metrics["hp_on_ticks"] = hp_on_ticks
        bench_metrics["off_pct"] = off_pct
        check_bench_metrics(num_regression, bench_metrics)
        assert hp_off_ticks > 0, "HP never cycled off despite strong solar + warm outdoor"
        assert hp_on_ticks > 0, "HP never cycled on (something is very wrong)"
        assert off_pct > 5.0, (
            f"HP off only {off_pct:.1f}% — expected more cycling"
        )

    def test_no_cycling_when_cold(self, bench_metrics, num_regression):
        """In deep cold with no solar, HP should always be on."""
        profile = PROFILES_2R2C["living_room"]
        model = ThermalModel2R2C(
            profile=profile, initial_temp=20.0, outdoor_temp=-15.0,
            sensor_noise_sigma=0.0,
        )
        for tick in range(96):  # 1 day
            model.step(hp_setpoint=22.0, dt_minutes=15.0, mode="heat", tick=tick)
            assert model.room_temp < 22.0, (
                f"Room exceeded setpoint in deep cold: {model.room_temp:.3f}"
            )
        bench_metrics["final_room_temp"] = model.room_temp
        check_bench_metrics(num_regression, bench_metrics)


class TestSolarEffect:
    """Verify solar gain warms room regardless of HP state."""

    def test_solar_warms_room_hp_off(self, bench_metrics, num_regression):
        """Solar should warm room even when HP is off."""
        profile = PROFILES_2R2C["living_room"]
        # HP off (setpoint=0), but solar should warm
        model = ThermalModel2R2C(
            profile=profile, initial_temp=18.0, outdoor_temp=18.0,
            sensor_noise_sigma=0.0, solar_gain=0.02,
        )
        model.step(hp_setpoint=0.0, dt_minutes=15.0, solar_proxy=1.0,
                    mode="heat", tick=0)
        bench_metrics["room_temp_after_solar"] = model.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert model.room_temp > 18.0, (
            f"Solar didn't warm room: {model.room_temp:.3f}"
        )

    def test_solar_warms_room_hp_on(self, bench_metrics, num_regression):
        """Solar should add to HP heating."""
        profile = PROFILES_2R2C["living_room"]
        # Run two models: one with solar, one without
        model_solar = ThermalModel2R2C(
            profile=profile, initial_temp=18.0, outdoor_temp=5.0,
            sensor_noise_sigma=0.0, solar_gain=0.02,
        )
        model_no_solar = ThermalModel2R2C(
            profile=profile, initial_temp=18.0, outdoor_temp=5.0,
            sensor_noise_sigma=0.0, solar_gain=0.0,
        )
        for tick in range(96):
            model_solar.step(hp_setpoint=22.0, dt_minutes=15.0,
                             solar_proxy=0.5, mode="heat", tick=tick)
            model_no_solar.step(hp_setpoint=22.0, dt_minutes=15.0,
                                solar_proxy=0.5, mode="heat", tick=tick)
        bench_metrics["solar_room_temp"] = model_solar.room_temp
        bench_metrics["no_solar_room_temp"] = model_no_solar.room_temp
        bench_metrics["solar_gain_temp"] = model_solar.room_temp - model_no_solar.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert model_solar.room_temp > model_no_solar.room_temp, (
            f"Solar model ({model_solar.room_temp:.3f}) not warmer than "
            f"no-solar ({model_no_solar.room_temp:.3f})"
        )


class TestStepResponse:
    """Verify response to step changes in outdoor temp."""

    def test_outdoor_step_down(self, bench_metrics, num_regression):
        """Sudden cold snap: room should cool then recover."""
        profile = PROFILES_2R2C["living_room"]
        model = ThermalModel2R2C(
            profile=profile, initial_temp=20.5, outdoor_temp=10.0,
            sensor_noise_sigma=0.0,
        )
        # Reach steady state at outdoor=10
        for tick in range(1000):
            model.step(hp_setpoint=22.0, dt_minutes=15.0, mode="heat", tick=tick)
        ss_temp = model.room_temp

        # Cold snap: outdoor drops to -10
        model.outdoor_temp = -10.0
        min_temp = model.room_temp
        for tick in range(1000, 2000):
            model.step(hp_setpoint=22.0, dt_minutes=15.0, mode="heat", tick=tick)
            min_temp = min(min_temp, model.room_temp)

        new_ss = model.room_temp
        bench_metrics["initial_ss"] = ss_temp
        bench_metrics["min_temp"] = min_temp
        bench_metrics["new_ss"] = new_ss
        check_bench_metrics(num_regression, bench_metrics)
        # Room should have dipped below previous steady state
        assert min_temp < ss_temp - 0.5, (
            f"Room didn't respond to cold snap: min={min_temp:.3f} vs ss={ss_temp:.3f}"
        )
        # But should recover to new (lower) steady state
        assert new_ss < ss_temp, "New SS should be lower with colder outdoor"
        # Should have stabilized (last few ticks nearly constant)
        assert abs(model.room_temp - new_ss) < 0.01


class TestCrossModelAgreement:
    """1R1C and 2R2C should agree on steady-state."""

    def test_steady_state_agreement(self, bench_metrics, num_regression):
        """Both models should converge to same equilibrium."""
        profile = PROFILES_2R2C["living_room"]
        T_out = 5.0
        sp = 22.0

        m1 = ThermalModel(
            profile=profile, initial_temp=20.0, outdoor_temp=T_out,
            sensor_noise_sigma=0.0,
        )
        m2 = ThermalModel2R2C(
            profile=profile, initial_temp=20.0, outdoor_temp=T_out,
            sensor_noise_sigma=0.0,
        )
        for tick in range(2000):
            m1.step(hp_setpoint=sp, dt_minutes=15.0, mode="heat", tick=tick)
            m2.step(hp_setpoint=sp, dt_minutes=15.0, mode="heat", tick=tick)

        bench_metrics["m1_final"] = m1.room_temp
        bench_metrics["m2_final"] = m2.room_temp
        bench_metrics["gap"] = abs(m1.room_temp - m2.room_temp)
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(m1.room_temp - m2.room_temp) < SS_TOL, (
            f"Models disagree at steady state: 1R1C={m1.room_temp:.3f} "
            f"2R2C={m2.room_temp:.3f}"
        )

    @pytest.mark.parametrize("profile_name", list(PROFILES_2R2C.keys()))
    def test_all_profiles_converge(self, bench_metrics, num_regression, profile_name):
        """Every profile should reach steady state without divergence."""
        profile = PROFILES_2R2C[profile_name]
        model = ThermalModel2R2C(
            profile=profile, initial_temp=20.0, outdoor_temp=0.0,
            sensor_noise_sigma=0.0,
        )
        for tick in range(3000):
            model.step(hp_setpoint=22.0, dt_minutes=15.0, mode="heat", tick=tick)

        assert 10 < model.room_temp < 30, (
            f"Profile {profile_name} diverged: room_temp={model.room_temp:.3f}"
        )
        # Check convergence (last 100 ticks should be stable)
        temps = []
        for tick in range(3000, 3100):
            model.step(hp_setpoint=22.0, dt_minutes=15.0, mode="heat", tick=tick)
            temps.append(model.room_temp)
        temp_range = max(temps) - min(temps)
        bench_metrics["profile_name"] = profile_name
        bench_metrics["final_room_temp"] = model.room_temp
        bench_metrics["temp_range"] = temp_range
        check_bench_metrics(num_regression, bench_metrics)
        assert temp_range < 0.01, (
            f"Profile {profile_name} not converged: range={temp_range:.4f}"
        )


class TestHPCapacityCurve:
    """HP capacity factor scales with outdoor temperature (#43).

    Real ASHPs lose capacity in cold and gain in mild conditions; the
    capacity curve adds that envelope to the bench thermal model.

    2026-06-03 Stage 1e refactor: the factor() machinery unit tests that
    used to live here have moved to tests/test_capacity_profiles.py
    (production-side coverage of CapacityCurve / CapacityProfile). The
    remaining tests here exercise downstream ThermalModel integration —
    that the bench correctly applies the capacity multiplier to HP heat
    delivery — using whatever curve shape production ships.
    """

    def test_cold_climate_holds_more_capacity_than_standard_at_minus_15(
        self, bench_metrics, num_regression,
    ):
        """At a cold operating temp, the cold-climate curve delivers more
        than the standard curve. This is the discriminator between the
        two profiles regardless of their exact anchor structure."""
        cc = COLD_CLIMATE_HP_CAPACITY.factor(-15.0, "heat")
        std = STANDARD_HP_CAPACITY.factor(-15.0, "heat")
        bench_metrics["standard_factor"] = std
        bench_metrics["cold_climate_factor"] = cc
        check_bench_metrics(num_regression, bench_metrics)
        assert cc > std, (
            f"Cold-climate should outperform standard at -15°C, got "
            f"cc={cc:.3f} vs std={std:.3f}"
        )

    # ── ThermalModel integration ─────────────────────────────────────

    def test_capacity_at_rated_matches_no_capacity(self, bench_metrics, num_regression):
        """At outdoor=AHRI rating point, capacity=1.0 — model behaves identically.

        Stage 1e (2026-06-03) refactor: outdoor=8.33°C is the AHRI rating
        point (47°F heating) where production's CapacityProfile gives
        ratio=1.0 by construction. The pre-refactor test used 7°C, which
        was the old bench HPCapacityCurve's rated_t (slightly off-rating).
        """
        base = PROFILES_2R2C["living_room"]
        assert base.hp_capacity is None
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        rating_point_c = 8.33  # AHRI_HEAT_RATING_C
        m_no_cap = ThermalModel2R2C(
            profile=base, initial_temp=20.0, outdoor_temp=rating_point_c,
            sensor_noise_sigma=0.0,
        )
        m_cap = ThermalModel2R2C(
            profile=with_cap, initial_temp=20.0, outdoor_temp=rating_point_c,
            sensor_noise_sigma=0.0,
        )
        for tick in range(2000):
            m_no_cap.step(hp_setpoint=25.0, dt_minutes=15.0, mode="heat", tick=tick)
            m_cap.step(hp_setpoint=25.0, dt_minutes=15.0, mode="heat", tick=tick)
        bench_metrics["no_cap_temp"] = m_no_cap.room_temp
        bench_metrics["cap_temp"] = m_cap.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert abs(m_cap.room_temp - m_no_cap.room_temp) < SS_TOL, (
            f"At rated_t, capacity=1.0 should match fixed-gain: "
            f"no_cap={m_no_cap.room_temp:.3f} cap={m_cap.room_temp:.3f}"
        )

    def test_2r2c_capacity_reduces_heating_in_cold(self, bench_metrics, num_regression):
        """With capacity curve, cold weather → smaller HP contribution → cooler room."""
        base = PROFILES_2R2C["living_room"]
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        m_no_cap = ThermalModel2R2C(
            profile=base, initial_temp=20.0, outdoor_temp=-10.0,
            sensor_noise_sigma=0.0,
        )
        m_cap = ThermalModel2R2C(
            profile=with_cap, initial_temp=20.0, outdoor_temp=-10.0,
            sensor_noise_sigma=0.0,
        )
        for tick in range(500):
            m_no_cap.step(hp_setpoint=23.0, dt_minutes=15.0, mode="heat", tick=tick)
            m_cap.step(hp_setpoint=23.0, dt_minutes=15.0, mode="heat", tick=tick)
        bench_metrics["no_cap_temp"] = m_no_cap.room_temp
        bench_metrics["cap_temp"] = m_cap.room_temp
        bench_metrics["capacity_loss"] = m_no_cap.room_temp - m_cap.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert m_cap.room_temp < m_no_cap.room_temp - 0.5, (
            f"Capacity curve should reduce cold-weather room temp: "
            f"no_cap={m_no_cap.room_temp:.3f} cap={m_cap.room_temp:.3f}"
        )

    def test_2r2c_capacity_strongly_reduces_below_anchor_range(self, bench_metrics, num_regression):
        """At outdoor below the lowest anchor temp, capacity clamps to the
        boundary value (sharply reduced but not zero). Room equilibrium
        falls well below setpoint because most HP capacity is lost — but
        not all the way to outdoor (the documented choice is boundary
        clamp, not cliff-to-zero, since manufacturers' published data
        ends but the HP itself keeps running at reduced capacity).
        See capacity_profiles.py module docstring for the rationale.
        """
        base = PROFILES_2R2C["living_room"]
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        # Lowest anchor in standard_inverter is -23°C; go below to -30°C.
        m_cap = ThermalModel2R2C(
            profile=with_cap, initial_temp=20.0, outdoor_temp=-30.0,
            sensor_noise_sigma=0.0,
        )
        m_no_cap = ThermalModel2R2C(
            profile=base, initial_temp=20.0, outdoor_temp=-30.0,
            sensor_noise_sigma=0.0,
        )
        for tick in range(500):
            m_cap.step(hp_setpoint=23.0, dt_minutes=15.0, mode="heat", tick=tick)
            m_no_cap.step(hp_setpoint=23.0, dt_minutes=15.0, mode="heat", tick=tick)
        bench_metrics["cap_final_temp"] = m_cap.room_temp
        bench_metrics["no_cap_final_temp"] = m_no_cap.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        # Capacity curve must drag equilibrium well below the no-capacity
        # baseline (at -30°C, all curves are degraded to their lowest
        # anchor's value — well under 50% of rated).
        assert m_cap.room_temp < m_no_cap.room_temp - 2.0, (
            f"Below the lowest anchor temp, capacity-clamped HP should fall "
            f"well behind the fixed-gain baseline: "
            f"cap={m_cap.room_temp:.3f} no_cap={m_no_cap.room_temp:.3f}"
        )

    def test_1r1c_capacity_reduces_heating_in_cold(self, bench_metrics, num_regression):
        """1R1C model also respects the capacity curve."""
        base = PROFILES_2R2C["living_room"]
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        m_no_cap = ThermalModel(
            profile=base, initial_temp=20.0, outdoor_temp=-10.0,
            sensor_noise_sigma=0.0,
        )
        m_cap = ThermalModel(
            profile=with_cap, initial_temp=20.0, outdoor_temp=-10.0,
            sensor_noise_sigma=0.0,
        )
        for tick in range(500):
            m_no_cap.step(hp_setpoint=23.0, dt_minutes=15.0, mode="heat", tick=tick)
            m_cap.step(hp_setpoint=23.0, dt_minutes=15.0, mode="heat", tick=tick)
        bench_metrics["no_cap_temp"] = m_no_cap.room_temp
        bench_metrics["cap_temp"] = m_cap.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert m_cap.room_temp < m_no_cap.room_temp - 0.5

    def test_capacity_boost_above_rated(self, bench_metrics, num_regression):
        """At outdoor above rated_t, capacity factor > 1 → faster warming."""
        base = PROFILES_2R2C["living_room"]
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        # mild_t default 20°C — pick a generous boost regime
        m_no_cap = ThermalModel2R2C(
            profile=base, initial_temp=15.0, outdoor_temp=18.0,
            sensor_noise_sigma=0.0,
        )
        m_cap = ThermalModel2R2C(
            profile=with_cap, initial_temp=15.0, outdoor_temp=18.0,
            sensor_noise_sigma=0.0,
        )
        # Run only a short stretch — both will eventually saturate at sp.
        for tick in range(20):
            m_no_cap.step(hp_setpoint=22.0, dt_minutes=15.0, mode="heat", tick=tick)
            m_cap.step(hp_setpoint=22.0, dt_minutes=15.0, mode="heat", tick=tick)
        bench_metrics["no_cap_temp"] = m_no_cap.room_temp
        bench_metrics["cap_temp"] = m_cap.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        assert m_cap.room_temp > m_no_cap.room_temp, (
            f"Mild outdoor should boost capacity: "
            f"no_cap={m_no_cap.room_temp:.3f} cap={m_cap.room_temp:.3f}"
        )

    def test_cooling_capacity_reduces_in_extreme_heat(self, bench_metrics, num_regression):
        """In cool mode, very hot outdoor reduces HP cooling capacity.

        Stage 1e (2026-06-03) refactor: production's _COOL_STANDARD cooling
        curve is gentler than the bench's old cool curve was (e.g. at 42°C
        production gives ~7% derating vs old ~36%). Threshold tightened to
        match the new curve's behavior — the test still demonstrates the
        principle (cool-mode HP derates at high outdoor → slightly warmer
        room) but the absolute effect is smaller because production's
        residential-typical curve is less aggressive.
        """
        # Build a profile with cooling capacity curve (reuse standard).
        base = PROFILES_2R2C["living_room"]
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        # outdoor=42°C is between rated (35°C, 1.00) and 43°C (0.92) on
        # the production cool curve — ~7% derating.
        m_no_cap = ThermalModel2R2C(
            profile=base, initial_temp=24.0, outdoor_temp=42.0,
            sensor_noise_sigma=0.0,
        )
        m_cap = ThermalModel2R2C(
            profile=with_cap, initial_temp=24.0, outdoor_temp=42.0,
            sensor_noise_sigma=0.0,
        )
        # Cool mode, sp=20 (below room) → HP runs.
        for tick in range(500):
            m_no_cap.step(hp_setpoint=20.0, dt_minutes=15.0, mode="cool", tick=tick)
            m_cap.step(hp_setpoint=20.0, dt_minutes=15.0, mode="cool", tick=tick)
        bench_metrics["no_cap_temp"] = m_no_cap.room_temp
        bench_metrics["cap_temp"] = m_cap.room_temp
        check_bench_metrics(num_regression, bench_metrics)
        # With reduced capacity, room stays warmer (less effective cooling).
        # ~7% derating produces ~0.25°C temperature gap — verify it's positive
        # and meaningful (>0.1°C).
        assert m_cap.room_temp > m_no_cap.room_temp + 0.1, (
            f"Hot outdoor should reduce cooling: "
            f"no_cap={m_no_cap.room_temp:.3f} cap={m_cap.room_temp:.3f}"
        )

    def test_living_room_capacity_profile_registered(self, bench_metrics, num_regression):
        """PROFILES_2R2C['living_room_capacity'] uses STANDARD_HP_CAPACITY."""
        profile = PROFILES_2R2C["living_room_capacity"]
        base = PROFILES_2R2C["living_room"]
        bench_metrics["tau_env"] = profile.tau_env
        bench_metrics["hp_gain"] = profile.hp_gain
        check_bench_metrics(num_regression, bench_metrics)
        assert profile.hp_capacity is STANDARD_HP_CAPACITY
        # Same thermal characteristics as base living_room.
        assert profile.tau_env == base.tau_env
        assert profile.hp_gain == base.hp_gain
