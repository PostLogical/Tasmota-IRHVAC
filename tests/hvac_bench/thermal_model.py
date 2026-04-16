"""Thermal models for HVAC benchmark.

Two model variants:
- ThermalModel: Single-node (1R1C) with proportional HP coupling.
- ThermalModel2R2C: Two-node (2R2C) with separate air and wall mass
  nodes. Exact matrix-exponential integration for both.
"""

import math
import random
from dataclasses import dataclass, field

from .house_profiles import HouseProfile, HouseProfile2R2C
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
                 noise_seed: int | None = None,
                 hp_lag_minutes: float = 0.0,
                 solar_gain: float = 0.0, stove_gain: float = 0.0):
        """Initialize thermal model.

        Args:
            profile: House thermal characteristics.
            initial_temp: Starting room temperature (°C).
            outdoor_temp: Initial outdoor temperature (°C).
            cop_model: COP model for energy tracking. None = no tracking.
            sensor_noise_sigma: Gaussian noise std dev on sensor readings (°C).
            sensor_quantization: Sensor resolution (e.g., 0.1°C). 0 = continuous.
            noise_seed: Random seed for reproducible noise. None = random.
            hp_lag_minutes: First-order lag on HP response (minutes). 0 = instant.
                Models the delay from setpoint change to room temperature effect.
            solar_gain: Solar sensitivity (°C/min per unit solar proxy). Zone config.
            stove_gain: Supplemental heat gain (°C/min when active). Zone config.
        """
        self.profile = profile
        self.solar_gain = solar_gain
        self.stove_gain = stove_gain
        self.room_temp = initial_temp
        self.outdoor_temp = outdoor_temp
        self.cop_model = cop_model or COPModel()
        self.sensor_noise_sigma = sensor_noise_sigma
        self.sensor_quantization = sensor_quantization
        self._rng = random.Random(noise_seed)
        self.hp_lag_minutes = hp_lag_minutes
        self._effective_setpoint: float = initial_temp

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
        # HP response lag: effective setpoint tracks commanded setpoint
        if self.hp_lag_minutes > 0:
            lag_decay = math.exp(-dt_minutes / self.hp_lag_minutes)
            self._effective_setpoint = (
                hp_setpoint + (self._effective_setpoint - hp_setpoint) * lag_decay
            )
        else:
            self._effective_setpoint = hp_setpoint

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
        solar_heat = self.solar_gain * solar_proxy * dt_minutes
        stove_heat = self.stove_gain * stove_active * dt_minutes

        # Equilibrium temperature (where room would settle with constant inputs)
        total_gain = 1.0 / tau_eff + hp_gain
        if total_gain == 0:
            return
        t_eq = (
            self.outdoor_temp / tau_eff
            + hp_gain * self._effective_setpoint
            + solar_heat / dt_minutes  # Convert back to rate
            + stove_heat / dt_minutes
            + extra_heat / dt_minutes
        ) / total_gain

        # Exact exponential decay toward equilibrium
        decay = math.exp(-dt_minutes / tau_eff)
        self.room_temp = t_eq + (self.room_temp - t_eq) * decay

        # Energy tracking
        thermal_output = abs(self._effective_setpoint - self.room_temp) * hp_gain * dt_minutes
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


class ThermalModel2R2C:
    """Two-node room thermal model: air + wall/mass.

    Physics: exact 2x2 matrix-exponential integration via
    Cayley-Hamilton theorem (stable at any step size).

    Two coupled nodes:
        Air:  dT_a/dt = (T_out - T_a)/τ_env + g*(sp - T_a)
                        + (T_w - T_a)/τ_c + Q_solar + Q_stove
        Wall: dT_w/dt = (T_a - T_w)/τ_m

    where τ_c = tau_couple, τ_m = tau_couple * mass_ratio,
    g = hp_gain.

    The sensor reads T_air. Wall temperature provides thermal
    inertia that buffers air temperature swings.
    """

    def __init__(self, profile: HouseProfile2R2C, initial_temp: float = 20.0,
                 outdoor_temp: float = 5.0, cop_model: COPModel | None = None,
                 sensor_noise_sigma: float = 0.0, sensor_quantization: float = 0.0,
                 noise_seed: int | None = None,
                 hp_lag_minutes: float = 0.0,
                 initial_wall_temp: float | None = None,
                 solar_gain: float = 0.0, stove_gain: float = 0.0):
        self.profile = profile
        self.solar_gain = solar_gain
        self.stove_gain = stove_gain
        self.room_temp = initial_temp  # Air node — what the sensor reads
        self.wall_temp = initial_wall_temp if initial_wall_temp is not None else initial_temp
        self.outdoor_temp = outdoor_temp
        self.cop_model = cop_model or COPModel()
        self.sensor_noise_sigma = sensor_noise_sigma
        self.sensor_quantization = sensor_quantization
        self._rng = random.Random(noise_seed)
        self.hp_lag_minutes = hp_lag_minutes
        self._effective_setpoint: float = initial_temp

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
        """Advance both nodes by one time step.

        Same interface as ThermalModel for drop-in compatibility.
        """
        # HP response lag
        if self.hp_lag_minutes > 0:
            lag_decay = math.exp(-dt_minutes / self.hp_lag_minutes)
            self._effective_setpoint = (
                hp_setpoint + (self._effective_setpoint - hp_setpoint) * lag_decay
            )
        else:
            self._effective_setpoint = hp_setpoint

        p = self.profile
        tau_env = p.tau_env
        tau_c = p.tau_couple
        tau_m = p.tau_wall  # tau_couple * mass_ratio
        g = p.hp_gain

        # Apply disturbances (affect envelope only, like the 1R1C model)
        extra_heat = 0.0
        tau_modifier = 1.0
        for d in self._disturbances:
            intensity = d.intensity(tick)
            if intensity > 0:
                extra_heat += d.heat_gain_c_per_min * intensity
                tau_modifier *= 1.0 - (1.0 - d.tau_factor) * intensity
        tau_env_eff = tau_env * tau_modifier

        # Heat input rates (°C/min).
        # Solar is split: ~30% heats air convectively, ~70% is absorbed
        # by walls/furniture as radiation. This prevents the small air
        # capacitance from over-responding to solar transients.
        q_solar_total = self.solar_gain * solar_proxy
        q_solar_air = q_solar_total * 0.3
        q_solar_wall = q_solar_total * 0.7
        q_stove = self.stove_gain * stove_active
        q_extra = extra_heat

        # System matrix A and forcing vector b:
        #   d/dt [T_a, T_w]^T = A * [T_a, T_w]^T + b
        #
        # A = [[-1/τ_env - g - 1/τ_c,   1/τ_c ],
        #      [ 1/τ_m,               -1/τ_m  ]]
        #
        # b = [T_out/τ_env + g*sp + q_solar_air + q_stove + q_extra,
        #      q_solar_wall / mass_ratio]

        a11 = -(1.0 / tau_env_eff + g + 1.0 / tau_c)
        a12 = 1.0 / tau_c
        a21 = 1.0 / tau_m
        a22 = -1.0 / tau_m

        b1 = (self.outdoor_temp / tau_env_eff
              + g * self._effective_setpoint
              + q_solar_air + q_stove + q_extra)
        b2 = q_solar_wall / p.mass_ratio  # Normalized by wall capacitance ratio

        # Equilibrium: T_eq = -A^{-1} * b
        det_A = a11 * a22 - a12 * a21
        if abs(det_A) < 1e-15:
            self.tick_count += 1
            return
        t_eq_a = -(a22 * b1 - a12 * b2) / det_A
        t_eq_w = -(-a21 * b1 + a11 * b2) / det_A

        # Deviation from equilibrium
        da = self.room_temp - t_eq_a
        dw = self.wall_temp - t_eq_w

        # Matrix exponential via Cayley-Hamilton:
        #   exp(A*dt) = α₀*I + α₁*A
        # where α₀, α₁ depend on eigenvalues of A.
        #
        # Eigenvalues of 2x2: λ = (tr ± √(tr²-4det)) / 2
        tr = a11 + a22
        disc = tr * tr - 4 * det_A
        dt = dt_minutes

        if disc > 1e-12:
            # Two distinct real eigenvalues (typical case)
            sqrt_disc = math.sqrt(disc)
            lam1 = (tr + sqrt_disc) / 2
            lam2 = (tr - sqrt_disc) / 2
            e1 = math.exp(lam1 * dt)
            e2 = math.exp(lam2 * dt)
            d_lam = lam1 - lam2
            alpha0 = (lam1 * e2 - lam2 * e1) / d_lam
            alpha1 = (e1 - e2) / d_lam
        elif disc < -1e-12:
            # Complex conjugate eigenvalues (rare but possible)
            real_part = tr / 2
            imag_part = math.sqrt(-disc) / 2
            e_real = math.exp(real_part * dt)
            cos_val = math.cos(imag_part * dt)
            sin_val = math.sin(imag_part * dt)
            alpha0 = e_real * (cos_val - real_part * sin_val / imag_part)
            alpha1 = e_real * sin_val / imag_part
        else:
            # Repeated eigenvalue (degenerate)
            lam = tr / 2
            e_lam = math.exp(lam * dt)
            alpha0 = e_lam * (1 - lam * dt)
            alpha1 = e_lam * dt

        # exp(A*dt) * [da, dw]^T  =  (α₀*I + α₁*A) * [da, dw]^T
        new_da = alpha0 * da + alpha1 * (a11 * da + a12 * dw)
        new_dw = alpha0 * dw + alpha1 * (a21 * da + a22 * dw)

        self.room_temp = t_eq_a + new_da
        self.wall_temp = t_eq_w + new_dw

        # Energy tracking (same approach as 1R1C)
        thermal_output = abs(self._effective_setpoint - self.room_temp) * g * dt_minutes
        cop = self.cop_model.cop(self.outdoor_temp, hp_setpoint, mode)
        if cop > 0:
            electrical_input = thermal_output / cop
            self.cumulative_kwh += electrical_input / 60.0
            self.cumulative_cop_weighted_output += thermal_output

        self.tick_count += 1

    def read_sensor(self) -> float:
        """Read air temperature with noise and quantization."""
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
