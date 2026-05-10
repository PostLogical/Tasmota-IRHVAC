"""Unit tests for synthetic_drivers helpers (Tier 1.3 building blocks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.hvac_bench.empirical.data_loader import REQUIRED_SIGNALS
from tests.hvac_bench.empirical.rc_model import RCParams1R1C
from tests.hvac_bench.empirical.synthetic_drivers import (
    make_inverter_hp_zone_telemetry,
    make_open_loop_inputs,
    make_synthetic_zone_telemetry,
    replace_room_temp_with_synthetic,
    simulate_1r1c_room_temp,
    simulate_inverter_hp_room_temp,
)
from tests.hvac_bench.house_profiles import (
    FUJITSU_HYPERHEAT_CAPACITY,
    HouseProfile2R2C,
)
from tests.hvac_bench.conftest import check_bench_metrics


_LIT_TRUTH = RCParams1R1C(
    R=0.005,
    C=80.0 * 3600.0 / 0.005,  # τ=80h → C = τ/R
    q_scale=1.0,
    solar_scale=2.0,
    sigma_w=0.0,
    sigma_v=0.05,
)


class TestMakeOpenLoopInputs:
    def test_required_signals_present(self) -> None:
        df = make_open_loop_inputs(n_steps=200)
        for sig in REQUIRED_SIGNALS:
            assert sig in df.columns

    def test_index_is_5min_utc(self) -> None:
        df = make_open_loop_inputs(n_steps=10)
        assert df.index.tz is not None
        assert str(df.index.tz) == "UTC"
        # 5-minute spacing
        diffs = df.index.to_series().diff().dropna().dt.total_seconds().unique()
        assert len(diffs) == 1
        assert diffs[0] == 300.0

    def test_setpoint_alternates_between_levels(self) -> None:
        df = make_open_loop_inputs(
            n_steps=100,
            setpoint_levels_c=(20.0, 22.0),
            setpoint_period_steps=10,
        )
        unique_sps = sorted(set(df["hp_setpoint_c"]))
        assert unique_sps == [20.0, 22.0]

    def test_outdoor_diurnal_oscillation(self) -> None:
        # 24-h period at 5-min ticks = 288 steps; cover 2 cycles to detect cycling
        df = make_open_loop_inputs(n_steps=576)
        outdoor = df["outdoor_temp_c_om"].to_numpy()
        # Span should be roughly 2 × diurnal_amp (default 8°C → span ≥ 14°C)
        assert outdoor.max() - outdoor.min() > 14.0

    def test_solar_zero_at_night_positive_at_noon(self) -> None:
        df = make_open_loop_inputs(n_steps=288)  # 24 hours
        sw = df["shortwave_w_m2"]
        # Hour 0 (midnight) should be zero
        assert sw.iloc[0] == 0.0
        # Hour 12 (noon, step 144) should be positive
        assert sw.iloc[144] > 100.0

    def test_q_heat_zero_when_inactive(self) -> None:
        # PRBS produces both states; verify q_heat = 0 on inactive rows.
        df = make_open_loop_inputs(n_steps=2000, seed=7)
        inactive_rows = df[~df["hp_active"]]
        assert len(inactive_rows) > 0  # PRBS must produce some off rows
        assert (inactive_rows["q_heat_proxy_w"] == 0.0).all()

    def test_q_heat_matches_nominal_when_active(self) -> None:
        df = make_open_loop_inputs(n_steps=200, nominal_capacity_w=2500.0)
        active_rows = df[df["hp_active"]]
        assert (active_rows["q_heat_proxy_w"] == 2500.0).all()

    def test_prbs_duty_cycle_approximates_target(self) -> None:
        # On a long sequence, mean hp_active should approximate the target duty cycle.
        df = make_open_loop_inputs(n_steps=10000, seed=42, prbs_duty_cycle=0.65)
        observed_duty = df["hp_active"].mean()
        assert abs(observed_duty - 0.65) < 0.05, (
            f"observed duty {observed_duty:.3f} differs from target 0.65 by > 0.05"
        )

    def test_prbs_produces_both_states(self) -> None:
        # PRBS must include transitions; not get stuck on or off.
        df = make_open_loop_inputs(n_steps=2000, seed=0)
        unique = set(df["hp_active"].unique())
        assert unique == {True, False}, f"PRBS stuck in single state: {unique}"

    def test_prbs_average_dwell_approximates_period(self) -> None:
        # Mean on-dwell should approximate prbs_avg_period_steps.
        df = make_open_loop_inputs(
            n_steps=20000, seed=123, prbs_avg_period_steps=12, prbs_duty_cycle=0.65
        )
        active = df["hp_active"].to_numpy()
        # Run-length encode the active sequence
        diffs = np.diff(active.astype(int))
        on_starts = np.where(diffs == 1)[0]
        off_starts = np.where(diffs == -1)[0]
        if len(on_starts) > 0 and len(off_starts) > 0:
            # Match each on-start to next off-start
            if off_starts[0] < on_starts[0]:
                off_starts = off_starts[1:]
            n_pairs = min(len(on_starts), len(off_starts))
            on_dwells = off_starts[:n_pairs] - on_starts[:n_pairs]
            mean_dwell = float(np.mean(on_dwells))
            assert 8 < mean_dwell < 18, (
                f"mean on-dwell {mean_dwell:.1f} far from target ~12 steps"
            )

    def test_n_steps_parametrizes_length(self) -> None:
        df = make_open_loop_inputs(n_steps=42)
        assert len(df) == 42


class TestSimulate1R1CRoomTemp:
    def test_returns_series_indexed_like_inputs(self) -> None:
        inputs = make_open_loop_inputs(n_steps=100)
        room = simulate_1r1c_room_temp(_LIT_TRUTH, inputs, seed=0)
        assert isinstance(room, pd.Series)
        assert (room.index == inputs.index).all()
        assert len(room) == 100

    def test_zero_inputs_yield_decay_to_zero(self) -> None:
        # Outdoor=0, q_heat=0, solar=0 → room decays from initial_temp toward 0
        idx = pd.date_range("2026-01-01", periods=500, freq="5min", tz="UTC")
        zeros = pd.DataFrame(
            {
                "outdoor_temp_c_om": 0.0,
                "q_heat_proxy_w": 0.0,
                "shortwave_w_m2": 0.0,
            },
            index=idx,
        )
        # No measurement noise so we observe the deterministic decay
        params = RCParams1R1C(R=0.005, C=2.0e6, q_scale=1.0, solar_scale=0.0,
                              sigma_w=0.0, sigma_v=1e-6)
        room = simulate_1r1c_room_temp(params, zeros, initial_temp_c=20.0)
        # τ = R·C = 1e4 s ≈ 2.78 h; after ~3τ should be close to 0
        steps_per_3tau = int(3 * params.tau_seconds / 300.0)
        assert abs(room.iloc[steps_per_3tau]) < 2.0
        assert room.iloc[0] > 15.0  # close to initial

    def test_deterministic_with_seed(self) -> None:
        inputs = make_open_loop_inputs(n_steps=50, seed=7)
        a = simulate_1r1c_room_temp(_LIT_TRUTH, inputs, seed=42)
        b = simulate_1r1c_room_temp(_LIT_TRUTH, inputs, seed=42)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_different_seeds_produce_different_noise(self) -> None:
        inputs = make_open_loop_inputs(n_steps=50, seed=7)
        a = simulate_1r1c_room_temp(_LIT_TRUTH, inputs, seed=42)
        b = simulate_1r1c_room_temp(_LIT_TRUTH, inputs, seed=43)
        assert not np.allclose(a.to_numpy(), b.to_numpy())


class TestMakeSyntheticZoneTelemetry:
    def test_returns_zone_telemetry_with_synthetic_room_temp(self) -> None:
        zt = make_synthetic_zone_telemetry(_LIT_TRUTH, n_steps=200)
        assert zt.df["room_temp_c"].notna().all()
        assert zt.info.fit_target is True
        # Sanity: room temp should not be the placeholder (setpoint - 1)
        # because the simulator overwrote it
        assert not np.allclose(
            zt.df["room_temp_c"].to_numpy(),
            (zt.df["hp_setpoint_c"] - 1.0).to_numpy(),
        )

    def test_valid_mask_all_true(self) -> None:
        zt = make_synthetic_zone_telemetry(_LIT_TRUTH, n_steps=100)
        assert bool(zt.df["valid"].all())


class TestReplaceRoomTempWithSynthetic:
    @pytest.fixture
    def real_telemetry(self):
        from tests.hvac_bench.empirical.data_loader import (
            apply_default_exclusions,
            load_condenser_a_zone,
            slice_window,
            TRAIN_WINDOW,
        )
        from pathlib import Path

        REPO_ROOT = Path(__file__).resolve().parents[3]
        bundle = REPO_ROOT / "local" / "debug_bundles" / "condenser_a_data"
        if not (bundle / "living_room.csv").exists():
            pytest.skip("Condenser A bundle not present")
        loaded = load_condenser_a_zone(bundle, "living_room")
        cleaned = apply_default_exclusions(loaded)
        return slice_window(cleaned, TRAIN_WINDOW)

    def test_room_temp_is_replaced(self, real_telemetry) -> None:
        synth = replace_room_temp_with_synthetic(real_telemetry, _LIT_TRUTH, seed=0)
        assert not np.allclose(
            synth.df["room_temp_c"].to_numpy(),
            real_telemetry.df["room_temp_c"].to_numpy(),
        )

    def test_other_columns_preserved(self, real_telemetry) -> None:
        synth = replace_room_temp_with_synthetic(real_telemetry, _LIT_TRUTH, seed=0)
        # Use pandas equality with NaN-equivalence (object/float columns share NaN)
        for col in (
            "outdoor_temp_c_om",
            "shortwave_w_m2",
            "hp_setpoint_c",
            "mode",
            "valid",
        ):
            pd.testing.assert_series_equal(
                synth.df[col], real_telemetry.df[col], check_exact=False
            )

    def test_info_and_exclusions_preserved(self, real_telemetry) -> None:
        synth = replace_room_temp_with_synthetic(real_telemetry, _LIT_TRUTH, seed=0)
        assert synth.info == real_telemetry.info
        assert synth.applied_exclusions == real_telemetry.applied_exclusions


# Option A inverter HP simulator helpers (proportional + capacity curve).
_INVERTER_PROFILE = HouseProfile2R2C(
    name="LR-1R1C-shape-Fujitsu",
    tau_env=80 * 60,
    tau_couple=1.0,
    mass_ratio=1.0,
    hp_gain=0.04,
    hp_capacity=FUJITSU_HYPERHEAT_CAPACITY,
)


class TestSimulateInverterHPRoomTemp:
    def test_returns_series_indexed_like_inputs(self) -> None:
        inputs = make_open_loop_inputs(n_steps=200)
        room, hp_active = simulate_inverter_hp_room_temp(
            inputs, profile=_INVERTER_PROFILE, seed=0
        )
        assert isinstance(room, pd.Series)
        assert isinstance(hp_active, pd.Series)
        assert (room.index == inputs.index).all()
        assert (hp_active.index == inputs.index).all()
        assert len(room) == 200

    def test_hp_active_emerges_from_physics_not_prbs(self) -> None:
        # PRBS in inputs.hp_active should be ignored — physics decides.
        inputs = make_open_loop_inputs(n_steps=200, seed=0)
        room, hp_active = simulate_inverter_hp_room_temp(
            inputs, profile=_INVERTER_PROFILE, seed=0
        )
        # On a heating-dominated regime (mean outdoor -5°C, sp=20-22°C)
        # HP should be active most of the time when the simulator is running.
        assert hp_active.dtype == bool or hp_active.dtype == object
        # On heating regime, HP active ≫ 50%
        assert float(hp_active.astype(int).mean()) > 0.5

    def test_room_tracks_setpoint_when_hp_can_keep_up(self) -> None:
        # In mild conditions with no solar, room should hover near setpoint.
        idx = pd.date_range("2026-01-01", periods=600, freq="5min", tz="UTC")
        steady = pd.DataFrame(
            {
                "outdoor_temp_c_om": 0.0,
                "hp_setpoint_c": 20.0,
                "solar_gain_proxy": 0.0,
            },
            index=idx,
        )
        room, _ = simulate_inverter_hp_room_temp(
            steady, profile=_INVERTER_PROFILE, initial_temp_c=15.0,
            sensor_noise_sigma=0.0, seed=0,
        )
        # After ample time at setpoint=20, room should sit near 20
        assert abs(room.iloc[-1] - 20.0) < 0.5

    def test_capacity_curve_zeroes_below_cutoff(self) -> None:
        # FUJITSU_HYPERHEAT_CAPACITY has cutoff at -32°C. Below that, HP
        # cannot heat at all — room should drift toward outdoor.
        idx = pd.date_range("2026-01-01", periods=400, freq="5min", tz="UTC")
        cold = pd.DataFrame(
            {
                "outdoor_temp_c_om": -35.0,  # below cutoff
                "hp_setpoint_c": 20.0,
                "solar_gain_proxy": 0.0,
            },
            index=idx,
        )
        room, _ = simulate_inverter_hp_room_temp(
            cold, profile=_INVERTER_PROFILE, initial_temp_c=20.0,
            sensor_noise_sigma=0.0, seed=0,
        )
        # τ=80h, dt=5min×400=33h ≈ 0.41τ; room drops by (1-exp(-0.41))×55=18°C
        # i.e. lands around 20 - 18 ≈ 2°C. Cutoff means HP can't fight decay.
        assert room.iloc[-1] < room.iloc[0] - 5.0

    def test_deterministic_with_seed(self) -> None:
        inputs = make_open_loop_inputs(n_steps=100, seed=3)
        a, _ = simulate_inverter_hp_room_temp(
            inputs, profile=_INVERTER_PROFILE, seed=42
        )
        b, _ = simulate_inverter_hp_room_temp(
            inputs, profile=_INVERTER_PROFILE, seed=42
        )
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_missing_required_input_raises(self) -> None:
        inputs = make_open_loop_inputs(n_steps=20).drop(
            columns=["outdoor_temp_c_om"]
        )
        with pytest.raises(ValueError, match="outdoor_temp_c_om"):
            simulate_inverter_hp_room_temp(inputs, profile=_INVERTER_PROFILE)


class TestMakeInverterHPZoneTelemetry:
    def test_returns_zone_telemetry_with_required_signals(self) -> None:
        zt = make_inverter_hp_zone_telemetry(
            profile=_INVERTER_PROFILE, n_steps=200, nominal_capacity_w=1500.0
        )
        for sig in REQUIRED_SIGNALS:
            assert sig in zt.df.columns
        assert "q_heat_proxy_w" in zt.df.columns
        assert "hp_active" in zt.df.columns
        assert zt.info.fit_target is True

    def test_constant_proxy_is_binary_at_nominal(self) -> None:
        zt = make_inverter_hp_zone_telemetry(
            profile=_INVERTER_PROFILE, n_steps=200,
            nominal_capacity_w=1500.0, proxy_variant="constant",
        )
        unique = sorted(zt.df["q_heat_proxy_w"].unique().tolist())
        assert unique == pytest.approx([0.0, 1500.0])

    def test_setpoint_modulated_proxy_carries_modulation(self) -> None:
        zt = make_inverter_hp_zone_telemetry(
            profile=_INVERTER_PROFILE, n_steps=600,
            nominal_capacity_w=1500.0, proxy_variant="setpoint_modulated",
        )
        # Modulated variant: q_heat_proxy_w spans a continuous range below
        # nominal when hp_active. (Sensor noise can push the noisy reading
        # transiently above setpoint even when hp_active=True, clipping
        # the modulation to 0 — that's the data_loader's intended behaviour.)
        active = zt.df["hp_active"].astype(bool)
        active_proxy = zt.df.loc[active, "q_heat_proxy_w"]
        assert (active_proxy >= 0).all()
        assert (active_proxy > 0).mean() > 0.5  # majority modulated, some clipped
        assert active_proxy.std() > 0.0  # not pinned at any single value
        assert active_proxy.max() <= 1500.0 + 1e-9
