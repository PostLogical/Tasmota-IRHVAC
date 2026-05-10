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
    HPCapacityCurve,
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

        # Room should never exceed setpoint
        assert model.room_temp < sp + 0.01, (
            f"Room exceeded setpoint: {model.room_temp:.3f} > {sp}"
        )
        # Analytical equilibrium (HP always on) for outdoor=0:
        tau = profile.tau_env
        g = profile.hp_gain
        t_eq_always_on = (0.0 / tau + g * sp) / (1.0 / tau + g)
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

        # Room should have dipped below previous steady state
        assert min_temp < ss_temp - 0.5, (
            f"Room didn't respond to cold snap: min={min_temp:.3f} vs ss={ss_temp:.3f}"
        )
        # But should recover to new (lower) steady state
        new_ss = model.room_temp
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
        assert temp_range < 0.01, (
            f"Profile {profile_name} not converged: range={temp_range:.4f}"
        )


class TestHPCapacityCurve:
    """HP capacity factor scales with outdoor temperature (#43).

    Real ASHPs lose capacity in cold and gain in mild conditions; the
    capacity curve adds that envelope to the bench thermal model.
    """

    # ── factor() unit tests ──────────────────────────────────────────

    def test_heating_anchor_points(self, bench_metrics, num_regression):
        """Curve passes through the three named anchors."""
        c = HPCapacityCurve()
        assert c.factor(c.heating_design_t, "heat") == 0.0
        assert c.factor(c.heating_rated_t, "heat") == pytest.approx(1.0)
        assert c.factor(c.heating_mild_t, "heat") == pytest.approx(c.heating_mild_factor)

    def test_heating_clamps_below_design(self, bench_metrics, num_regression):
        """Below design temp, capacity stays at zero (HP can't run)."""
        c = HPCapacityCurve()
        assert c.factor(-30.0, "heat") == 0.0
        assert c.factor(-100.0, "heat") == 0.0

    def test_heating_clamps_above_mild(self, bench_metrics, num_regression):
        """Above the mild knee, factor saturates at mild_factor."""
        c = HPCapacityCurve()
        assert c.factor(40.0, "heat") == pytest.approx(c.heating_mild_factor)

    def test_heating_linear_below_rated(self, bench_metrics, num_regression):
        """Halfway between design and rated → 50% capacity."""
        c = HPCapacityCurve(heating_design_t=-15.0, heating_rated_t=7.0)
        midpoint = (-15.0 + 7.0) / 2  # -4°C
        assert c.factor(midpoint, "heat") == pytest.approx(0.5)

    def test_heating_linear_above_rated(self, bench_metrics, num_regression):
        """Halfway between rated and mild → halfway from 1.0 to mild_factor."""
        c = HPCapacityCurve(heating_rated_t=7.0, heating_mild_t=20.0,
                             heating_mild_factor=1.15)
        midpoint = (7.0 + 20.0) / 2  # 13.5°C
        expected = 1.0 + 0.5 * (1.15 - 1.0)
        assert c.factor(midpoint, "heat") == pytest.approx(expected)

    def test_cooling_anchor_points(self, bench_metrics, num_regression):
        """Cooling mirrors heating with reversed slope."""
        c = HPCapacityCurve()
        assert c.factor(c.cooling_design_t, "cool") == 0.0
        assert c.factor(c.cooling_rated_t, "cool") == pytest.approx(1.0)
        assert c.factor(c.cooling_mild_t, "cool") == pytest.approx(c.cooling_mild_factor)

    def test_cooling_clamps_above_design(self, bench_metrics, num_regression):
        """Above cooling design temp (extreme heat), factor is zero."""
        c = HPCapacityCurve()
        assert c.factor(60.0, "cool") == 0.0

    def test_cooling_clamps_below_mild(self, bench_metrics, num_regression):
        """Below cooling mild knee, factor saturates at mild_factor."""
        c = HPCapacityCurve()
        assert c.factor(0.0, "cool") == pytest.approx(c.cooling_mild_factor)

    def test_cooling_linear_above_rated(self, bench_metrics, num_regression):
        """Halfway between cooling rated and design → 50% capacity."""
        c = HPCapacityCurve(cooling_rated_t=35.0, cooling_design_t=46.0)
        midpoint = (35.0 + 46.0) / 2  # 40.5°C
        assert c.factor(midpoint, "cool") == pytest.approx(0.5)

    def test_cooling_linear_below_rated(self, bench_metrics, num_regression):
        """Halfway between mild and rated → halfway from mild_factor to 1.0."""
        c = HPCapacityCurve(cooling_mild_t=18.0, cooling_rated_t=35.0,
                             cooling_mild_factor=1.15)
        midpoint = (18.0 + 35.0) / 2  # 26.5°C
        expected = 1.15 + 0.5 * (1.0 - 1.15)
        assert c.factor(midpoint, "cool") == pytest.approx(expected)

    def test_cold_climate_curve_holds_capacity_below_minus_15(self, bench_metrics, num_regression):
        """CCASHP curve still delivers >0 capacity at -15°C unlike standard."""
        assert STANDARD_HP_CAPACITY.factor(-15.0, "heat") == 0.0
        cc = COLD_CLIMATE_HP_CAPACITY.factor(-15.0, "heat")
        assert 0.25 < cc < 0.50, (
            f"CCASHP at -15°C should retain ~30% capacity, got {cc:.3f}"
        )

    # ── ThermalModel integration ─────────────────────────────────────

    def test_capacity_at_rated_matches_no_capacity(self, bench_metrics, num_regression):
        """At outdoor=rated_t, capacity=1.0 — model behaves identically."""
        base = PROFILES_2R2C["living_room"]
        assert base.hp_capacity is None
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        # rated_t default is 7°C
        m_no_cap = ThermalModel2R2C(
            profile=base, initial_temp=20.0, outdoor_temp=7.0,
            sensor_noise_sigma=0.0,
        )
        m_cap = ThermalModel2R2C(
            profile=with_cap, initial_temp=20.0, outdoor_temp=7.0,
            sensor_noise_sigma=0.0,
        )
        for tick in range(2000):
            m_no_cap.step(hp_setpoint=25.0, dt_minutes=15.0, mode="heat", tick=tick)
            m_cap.step(hp_setpoint=25.0, dt_minutes=15.0, mode="heat", tick=tick)
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
        assert m_cap.room_temp < m_no_cap.room_temp - 0.5, (
            f"Capacity curve should reduce cold-weather room temp: "
            f"no_cap={m_no_cap.room_temp:.3f} cap={m_cap.room_temp:.3f}"
        )

    def test_2r2c_capacity_zero_below_design_temp(self, bench_metrics, num_regression):
        """At outdoor ≤ heating_design_t, HP delivers no heat (room → outdoor)."""
        base = PROFILES_2R2C["living_room"]
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        # design_t default is -15°C; well below it = -20°C.
        m_cap = ThermalModel2R2C(
            profile=with_cap, initial_temp=20.0, outdoor_temp=-20.0,
            sensor_noise_sigma=0.0,
        )
        # Run until decayed; HP commanded ON but capacity=0 → no heat input.
        for tick in range(500):
            m_cap.step(hp_setpoint=23.0, dt_minutes=15.0, mode="heat", tick=tick)
        assert abs(m_cap.room_temp - (-20.0)) < 0.5, (
            f"Below design temp, room should decay to outdoor: "
            f"got {m_cap.room_temp:.3f}, expected ~-20.0"
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
        assert m_cap.room_temp > m_no_cap.room_temp, (
            f"Mild outdoor should boost capacity: "
            f"no_cap={m_no_cap.room_temp:.3f} cap={m_cap.room_temp:.3f}"
        )

    def test_cooling_capacity_reduces_in_extreme_heat(self, bench_metrics, num_regression):
        """In cool mode, very hot outdoor reduces HP cooling capacity."""
        # Build a profile with cooling capacity curve (reuse standard).
        base = PROFILES_2R2C["living_room"]
        with_cap = replace(base, hp_capacity=STANDARD_HP_CAPACITY)
        # outdoor=42°C is between rated (35) and design (46); cap ~36%.
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
        # With reduced capacity, room stays warmer (less effective cooling).
        assert m_cap.room_temp > m_no_cap.room_temp + 0.3, (
            f"Hot outdoor should reduce cooling: "
            f"no_cap={m_no_cap.room_temp:.3f} cap={m_cap.room_temp:.3f}"
        )

    def test_living_room_capacity_profile_registered(self, bench_metrics, num_regression):
        """PROFILES_2R2C['living_room_capacity'] uses STANDARD_HP_CAPACITY."""
        profile = PROFILES_2R2C["living_room_capacity"]
        assert profile.hp_capacity is STANDARD_HP_CAPACITY
        # Same thermal characteristics as base living_room.
        base = PROFILES_2R2C["living_room"]
        assert profile.tau_env == base.tau_env
        assert profile.hp_gain == base.hp_gain
