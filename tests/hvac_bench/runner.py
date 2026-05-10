"""Simulation runner for HVAC benchmark.

Couples a ThermalModel with an HVACController and runs a scenario,
recording history at each tick.

The runner exposes two parallel scheduling conventions:

* **Tick-keyed (legacy):** ``outdoor_schedule`` / ``solar_schedule`` /
  ``desired_schedule`` / ``stove_schedule`` accept a callable(tick) or
  dict {tick: value}.  Tests migrating to wall-clock semantics should
  use the ``_minute`` variants below instead — tick-keyed schedules
  embed the cadence into the test contract, which is exactly the
  fidelity bug the cadence-migration is fixing.

* **Minute-keyed (preferred):** ``outdoor_minute_schedule`` /
  ``solar_minute_schedule`` / ``desired_minute_schedule`` /
  ``stove_minute_schedule`` accept callable(minute) or dict
  {minute: value}.  Tests using these stay correct under any tick
  cadence: ``minute = tick * tick_interval_min``.

Same story for run length: ``n_ticks`` is the legacy tick-count
parameter; ``duration_minutes`` is the cadence-independent
alternative.  Pass exactly one.
"""

from .constants import TICK_MINUTES_DEFAULT
from .thermal_model import ThermalModel
from .controller_protocol import HVACController
from .house_profiles import HouseProfile
from .disturbances import Disturbance


def run_scenario(controller: HVACController, model: ThermalModel,
                 n_ticks: int | None = None, mode: str = "heat",
                 outdoor_schedule=None, solar_schedule=None,
                 desired_schedule=None, stove_schedule=None,
                 tick_interval_min: float = TICK_MINUTES_DEFAULT,
                 solar_gain: float | None = None,
                 stove_gain: float | None = None,
                 *,
                 duration_minutes: float | None = None,
                 outdoor_minute_schedule=None,
                 solar_minute_schedule=None,
                 desired_minute_schedule=None,
                 stove_minute_schedule=None) -> list[dict]:
    """Run a benchmark scenario.

    Args:
        controller: HVACController implementation to test.
        model: ThermalModel with house profile and disturbances.
        n_ticks: Number of simulation ticks (legacy; pass exactly one of
            ``n_ticks`` or ``duration_minutes``).  Use only when the
            assertion is genuinely about a tick-counted phenomenon
            (buffer fill, batch fire counts) — for sim-duration
            semantics use ``duration_minutes`` instead.
        mode: "heat" or "cool".
        outdoor_schedule: legacy — dict {tick: temp} or callable(tick) -> temp.
        solar_schedule: legacy — dict {tick: val} or callable(tick) -> val.
        desired_schedule: legacy — dict {tick: temp} for setpoint changes.
        stove_schedule: legacy — dict {tick: val} or callable(tick) -> val.
        tick_interval_min: Minutes per tick.  Defaults to
            ``constants.TICK_MINUTES_DEFAULT`` (overridable via
            ``pytest --tick-minutes=N`` or ``BENCH_TICK_MINUTES`` env).
        solar_gain: Override model's solar gain for this run.
        stove_gain: Override model's stove gain for this run.
        duration_minutes: Cadence-independent run length (preferred).
        outdoor_minute_schedule: dict {minute: temp} or callable(minute).
        solar_minute_schedule: dict {minute: val} or callable(minute).
        desired_minute_schedule: dict {minute: temp} or callable(minute).
        stove_minute_schedule: dict {minute: val} or callable(minute).

    Returns:
        List of history dicts, one per tick.
    """
    # Length: exactly one of n_ticks / duration_minutes must be supplied.
    if (n_ticks is None) == (duration_minutes is None):
        raise ValueError(
            "run_scenario requires exactly one of n_ticks or duration_minutes"
        )
    if duration_minutes is not None:
        n_ticks = int(round(duration_minutes / tick_interval_min))

    # Schedules: legacy tick-keyed and new minute-keyed are mutually
    # exclusive per source (outdoor / solar / desired / stove).  Mixing
    # both for the same source would silently overwrite based on
    # evaluation order — surface as an error instead.
    for legacy, minute_kw, name in (
        (outdoor_schedule, outdoor_minute_schedule, "outdoor"),
        (solar_schedule, solar_minute_schedule, "solar"),
        (desired_schedule, desired_minute_schedule, "desired"),
        (stove_schedule, stove_minute_schedule, "stove"),
    ):
        if legacy is not None and minute_kw is not None:
            raise ValueError(
                f"run_scenario: pass only one of {name}_schedule (legacy) "
                f"and {name}_minute_schedule (preferred)"
            )

    if solar_gain is not None:
        model.solar_gain = solar_gain
    if stove_gain is not None:
        model.stove_gain = stove_gain
    dt_seconds = tick_interval_min * 60.0
    controller.set_mode(mode)

    history = []
    solar = 0.0
    stove = 0.0

    for tick in range(n_ticks):
        minute = tick * tick_interval_min

        # Apply schedules — legacy (tick-keyed) and minute-keyed are
        # exclusive per source by the validation above, so at most one
        # branch fires per source per tick.
        if outdoor_schedule is not None:
            if callable(outdoor_schedule):
                model.outdoor_temp = outdoor_schedule(tick)
            elif tick in outdoor_schedule:
                model.outdoor_temp = outdoor_schedule[tick]
        elif outdoor_minute_schedule is not None:
            if callable(outdoor_minute_schedule):
                model.outdoor_temp = outdoor_minute_schedule(minute)
            elif minute in outdoor_minute_schedule:
                model.outdoor_temp = outdoor_minute_schedule[minute]

        if solar_schedule is not None:
            if callable(solar_schedule):
                solar = solar_schedule(tick)
            elif tick in solar_schedule:
                solar = solar_schedule[tick]
        elif solar_minute_schedule is not None:
            if callable(solar_minute_schedule):
                solar = solar_minute_schedule(minute)
            elif minute in solar_minute_schedule:
                solar = solar_minute_schedule[minute]

        if stove_schedule is not None:
            if callable(stove_schedule):
                stove = stove_schedule(tick)
            elif tick in stove_schedule:
                stove = stove_schedule[tick]
        elif stove_minute_schedule is not None:
            if callable(stove_minute_schedule):
                stove = stove_minute_schedule(minute)
            elif minute in stove_minute_schedule:
                stove = stove_minute_schedule[minute]

        if desired_schedule is not None:
            if tick in desired_schedule:
                controller.set_desired_temp(desired_schedule[tick])
        elif desired_minute_schedule is not None:
            if minute in desired_minute_schedule:
                controller.set_desired_temp(desired_minute_schedule[minute])

        # Read sensor (with noise if configured)
        sensor_reading = model.read_sensor()

        # Build model inputs for controller
        model_inputs = {}
        if solar > 0:
            model_inputs["solar"] = solar
        if stove > 0:
            model_inputs["stove"] = stove

        # Controller tick
        hp_setpoint = controller.tick(
            room_temp_c=sensor_reading,
            outdoor_temp_c=model.outdoor_temp,
            dt_seconds=dt_seconds,
            model_inputs=model_inputs,
        )

        # Advance thermal model with HP setpoint
        model.step(
            hp_setpoint=hp_setpoint,
            dt_minutes=tick_interval_min,
            solar_proxy=solar,
            stove_active=stove,
            tick=tick,
            mode=mode,
        )

        # Get controller state
        state = controller.get_state()

        # Compute COP at this tick
        cop = model.cop_model.cop(model.outdoor_temp, hp_setpoint, mode)

        # Record history
        desired = state.get("desired_temp", 20.5)
        history.append({
            "tick": tick,
            "minute": tick * tick_interval_min,
            "room_temp": model.room_temp,
            "sensor_reading": sensor_reading,
            "desired": desired,
            "hp_setpoint": hp_setpoint,
            "integral": state.get("integral", 0.0),
            "ff_offset": state.get("ff_offset", 0.0),
            "error": desired - model.room_temp,
            "outdoor": model.outdoor_temp,
            "d_term": state.get("d_term", 0.0),
            "rls_obs_count": state.get("rls_obs_count", 0),
            "smith_correction": state.get("smith_correction", 0.0),
            "cop": cop,
            "cumulative_kwh": model.cumulative_kwh,
            "solar": solar,
            "stove": stove,
        })

    return history
