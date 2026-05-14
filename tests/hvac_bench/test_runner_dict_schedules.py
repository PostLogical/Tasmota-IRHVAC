"""Step-function semantic for dict-keyed schedules in runner.run_scenario.

A dict-keyed schedule like ``{150: 22.5}`` should be interpreted as
piecewise constant: at minute m the schedule's value is ``schedule[k]``
where k is the largest key ≤ m. The pre-#103 behavior was exact-match
only — ``minute in schedule`` — which silently dropped events when the
cadence did not land on a key (e.g. minute 37 at 5-min cadence).

Today's bench uses keys divisible by both 15 and 3 so the bug never
fires; this test pins the corrected semantic so a future cadence /
schedule combination cannot reintroduce silent event-loss.
"""

from __future__ import annotations

from tests.hvac_bench.house_profiles import PROFILES
from tests.hvac_bench.reference_controllers import NaiveBangBangController
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.thermal_model import ThermalModel


def _model() -> ThermalModel:
    return ThermalModel(
        profile=PROFILES["standard_residential"],
        initial_temp=20.0,
        outdoor_temp=5.0,
    )


def test_desired_schedule_off_grid_key_fires_on_next_tick() -> None:
    """Setpoint change at minute 37 lands at the first tick ≥ 37 (= minute 40 at 5-min cadence)."""
    ctrl = NaiveBangBangController()
    history = run_scenario(
        ctrl,
        _model(),
        duration_minutes=60,
        mode="heat",
        tick_interval_min=5.0,
        desired_schedule={37: 22.0},
    )

    # Minutes 0, 5, ..., 35 → controller.desired stays at default 20.5.
    pre_change = [h for h in history if h["minute"] < 37]
    assert all(h["desired"] == 20.5 for h in pre_change), (
        f"desired must hold pre-key default; got "
        f"{[(h['minute'], h['desired']) for h in pre_change if h['desired'] != 20.5]}"
    )

    # Minute 40 (first tick ≥ 37) and onward → controller.desired = 22.0.
    post_change = [h for h in history if h["minute"] >= 40]
    assert post_change, "scenario must include at least one tick at or after minute 40"
    assert all(h["desired"] == 22.0 for h in post_change), (
        f"desired must step to 22.0 at first tick ≥ key 37; got "
        f"{[(h['minute'], h['desired']) for h in post_change if h['desired'] != 22.0]}"
    )


def test_outdoor_schedule_off_grid_key_holds_step_value() -> None:
    """Outdoor at minute 22 holds 10.0 from tick at minute 25 onward (5-min cadence)."""
    ctrl = NaiveBangBangController()
    history = run_scenario(
        ctrl,
        _model(),
        duration_minutes=45,
        mode="heat",
        tick_interval_min=5.0,
        outdoor_schedule={0: 5.0, 22: 10.0},
    )

    # Pre-key-22: outdoor reads 5.0 (the largest key ≤ minute is 0).
    pre = [h for h in history if h["minute"] < 22]
    assert all(h["outdoor"] == 5.0 for h in pre), (
        f"outdoor must hold key-0 value before minute 22; got "
        f"{[(h['minute'], h['outdoor']) for h in pre if h['outdoor'] != 5.0]}"
    )

    # Post-key-22: outdoor reads 10.0 (largest key ≤ minute is 22).
    post = [h for h in history if h["minute"] >= 25]
    assert post, "scenario must include at least one tick at or after minute 25"
    assert all(h["outdoor"] == 10.0 for h in post), (
        f"outdoor must step to 10.0 at first tick ≥ key 22; got "
        f"{[(h['minute'], h['outdoor']) for h in post if h['outdoor'] != 10.0]}"
    )


def test_dict_schedule_on_grid_keys_byte_identical_to_callable() -> None:
    """Dict {0: 5.0, 30: 10.0} matches callable producing the same step function."""
    def step_fn(minute: float) -> float:
        return 5.0 if minute < 30 else 10.0

    ctrl_dict = NaiveBangBangController()
    ctrl_call = NaiveBangBangController()

    h_dict = run_scenario(
        ctrl_dict, _model(), duration_minutes=60, mode="heat",
        tick_interval_min=5.0, outdoor_schedule={0: 5.0, 30: 10.0},
    )
    h_call = run_scenario(
        ctrl_call, _model(), duration_minutes=60, mode="heat",
        tick_interval_min=5.0, outdoor_schedule=step_fn,
    )

    for d, c in zip(h_dict, h_call):
        assert d["outdoor"] == c["outdoor"], (
            f"dict and callable diverge at minute {d['minute']}: "
            f"dict={d['outdoor']} callable={c['outdoor']}"
        )
