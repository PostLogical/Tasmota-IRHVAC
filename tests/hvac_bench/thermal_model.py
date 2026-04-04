"""Enhanced thermal model for HVAC benchmark.

Single-node (1R) room model with exact exponential integration,
sensor noise injection, disturbance support, and COP tracking.
"""

import math
import random
from dataclasses import dataclass, field

from .house_profiles import HouseProfile
from .disturbances import Disturbance


@dataclass
class COPModel:
    """Heat pump coefficient of performance model.

    Default is a generic linear model: COP decreases as temperature lift
    increases. Override cop_fn for manufacturer-specific curves.
    """
    # Generic linear defaults
    heat_cop_at_zero_lift: float = 5.0
    heat_cop_slope: float = -0.08  # COP drops 0.08 per °C of lift
    cool_cop_at_zero_lift: float = 6.0
    cool_cop_slope: float = -0.10
    min_cop: float = 1.0

    # Optional custom function: (outdoor_c, setpoint_c, mode) -> COP
    cop_fn: object = None  # Callable or None

    def cop(self, outdoor_c, setpoint_c, mode="heat"):
        """Compute COP for given conditions."""
        if self.cop_fn is not None:
            return max(self.min_cop, self.cop_fn(outdoor_c, setpoint_c, mode))
        if mode == "heat":
            lift = max(0, setpoint_c - outdoor_c)
            return max(self.min_cop, self.heat_cop_at_zero_lift + self.heat_cop_slope * lift)
        else:
            lift = max(0, outdoor_c - setpoint_c)
            return max(self.min_cop, self.cool_cop_at_zero_lift + self.cool_cop_slope * lift)


class ThermalModel:
    """Single-node room thermal model with disturbances and COP tracking.

    Physics: exact exponential integration (stable at any step size).

    T_eq = (T_outdoor/τ + hp_gain*hp_setpoint + Q_solar + Q_stove + Q_disturbance)
           / (1/τ_eff + hp_gain)
    T_new = T_eq + (T_old - T_eq) * exp(-dt / τ_eff)

    where τ_eff accounts for disturbance-modified time constant.
    """

    def __init__(self, profile: HouseProfile, initial_temp: float = 20.0,
                 outdoor_temp: float = 5.0, cop_model: COPModel | None = None,
                 sensor_noise_sigma: float = 0.0, sensor_quantization: float = 0.0,
                 noise_seed: int | None = None):
        """Initialize thermal model.

        Args:
            profile: House thermal characteristics.
            initial_temp: Starting room temperature (°C).
            outdoor_temp: Initial outdoor temperature (°C).
            cop_model: COP model for energy tracking. None = no tracking.
            sensor_noise_sigma: Gaussian noise std dev on sensor readings (°C).
            sensor_quantization: Sensor resolution (e.g., 0.1°C). 0 = continuous.
            noise_seed: Random seed for reproducible noise. None = random.
        """
        self.profile = profile
        self.room_temp = initial_temp
        self.outdoor_temp = outdoor_temp
        self.cop_model = cop_model or COPModel()
        self.sensor_noise_sigma = sensor_noise_sigma
        self.sensor_quantization = sensor_quantization
        self._rng = random.Random(noise_seed)

        # Energy tracking
        self.cumulative_kwh = 0.0
        self.cumulative_cop_weighted_output = 0.0
        self.tick_count = 0

        # Active disturbances
        self._disturbances: list[Disturbance] = []

    def add_disturbance(self, disturbance: Disturbance):
        """Schedule a disturbance."""
        self._disturbances.append(disturbance)

    def step(self, hp_setpoint: float, dt_minutes: float = 15.0,
             solar_proxy: float = 0.0, stove_active: float = 0.0,
             tick: int = 0, mode: str = "heat") -> None:
        """Advance the thermal model by one time step.

        Args:
            hp_setpoint: Current HP setpoint (°C).
            dt_minutes: Time step duration in minutes.
            solar_proxy: Solar gain proxy (0-1).
            stove_active: Stove binary (0 or 1).
            tick: Current tick number (for disturbance timing).
            mode: "heat" or "cool" (for COP calculation).
        """
        tau = self.profile.tau_minutes
        hp_gain = self.profile.hp_gain

        # Apply disturbances
        extra_heat = 0.0
        tau_modifier = 1.0
        for d in self._disturbances:
            intensity = d.intensity(tick)
            if intensity > 0:
                extra_heat += d.heat_gain_c_per_min * intensity * dt_minutes
                tau_modifier *= 1.0 - (1.0 - d.tau_factor) * intensity

        tau_eff = tau * tau_modifier

        # Heat inputs
        solar_heat = self.profile.solar_gain * solar_proxy * dt_minutes
        stove_heat = self.profile.stove_gain * stove_active * dt_minutes

        # Equilibrium temperature (where room would settle with constant inputs)
        total_gain = 1.0 / tau_eff + hp_gain
        if total_gain == 0:
            return
        t_eq = (
            self.outdoor_temp / tau_eff
            + hp_gain * hp_setpoint
            + solar_heat / dt_minutes  # Convert back to rate
            + stove_heat / dt_minutes
            + extra_heat / dt_minutes
        ) / total_gain

        # Exact exponential decay toward equilibrium
        decay = math.exp(-dt_minutes / tau_eff)
        self.room_temp = t_eq + (self.room_temp - t_eq) * decay

        # Energy tracking
        thermal_output = abs(hp_setpoint - self.room_temp) * hp_gain * dt_minutes
        cop = self.cop_model.cop(self.outdoor_temp, hp_setpoint, mode)
        if cop > 0:
            electrical_input = thermal_output / cop
            self.cumulative_kwh += electrical_input / 60.0  # min → hours
            self.cumulative_cop_weighted_output += thermal_output

        self.tick_count += 1

    def read_sensor(self) -> float:
        """Read room temperature with noise and quantization.

        Returns the room temperature as a sensor would report it,
        with optional Gaussian noise and resolution quantization.
        """
        temp = self.room_temp
        if self.sensor_noise_sigma > 0:
            temp += self._rng.gauss(0, self.sensor_noise_sigma)
        if self.sensor_quantization > 0:
            temp = round(temp / self.sensor_quantization) * self.sensor_quantization
        return temp

    @property
    def average_cop(self) -> float:
        """Average COP over all ticks."""
        if self.cumulative_kwh <= 0:
            return 0.0
        return self.cumulative_cop_weighted_output / (self.cumulative_kwh * 60.0)
