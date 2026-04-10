"""Simulation runner for HVAC benchmark.

Couples a ThermalModel with an HVACController and runs a scenario,
recording history at each tick.
"""

from .thermal_model import ThermalModel
from .controller_protocol import HVACController
from .house_profiles import HouseProfile
from .disturbances import Disturbance


def run_scenario(controller: HVACController, model: ThermalModel,
                 n_ticks: int, mode: str = "heat",
                 outdoor_schedule=None, solar_schedule=None,
                 desired_schedule=None, stove_schedule=None,
                 tick_interval_min: float = 15.0) -> list[dict]:
    """Run a benchmark scenario.

    Args:
        controller: HVACController implementation to test.
        model: ThermalModel with house profile and disturbances.
        n_ticks: Number of simulation ticks.
        mode: "heat" or "cool".
        outdoor_schedule: dict {tick: temp} or callable(tick) -> temp.
        solar_schedule: dict {tick: val} or callable(tick) -> val.
        desired_schedule: dict {tick: temp} for setpoint changes.
        stove_schedule: dict {tick: val} or callable(tick) -> val.
        tick_interval_min: Minutes per tick (default 15).

    Returns:
        List of history dicts, one per tick.
    """
    dt_seconds = tick_interval_min * 60.0
    controller.set_mode(mode)

    history = []
    solar = 0.0
    stove = 0.0

    for tick in range(n_ticks):
        # Apply schedules
        if outdoor_schedule is not None:
            if callable(outdoor_schedule):
                model.outdoor_temp = outdoor_schedule(tick)
            elif tick in outdoor_schedule:
                model.outdoor_temp = outdoor_schedule[tick]

        if solar_schedule is not None:
            if callable(solar_schedule):
                solar = solar_schedule(tick)
            elif tick in solar_schedule:
                solar = solar_schedule[tick]

        if stove_schedule is not None:
            if callable(stove_schedule):
                stove = stove_schedule(tick)
            elif tick in stove_schedule:
                stove = stove_schedule[tick]

        if desired_schedule is not None:
            if tick in desired_schedule:
                controller.set_desired_temp(desired_schedule[tick])

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
