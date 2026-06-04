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
                 tau_hp_minutes: float = 0.0,
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
            hp_lag_minutes: First-order lag on the **commanded setpoint**
                (minutes). 0 = instant. Models the delay between the user-
                requested setpoint and the HP's internal effective target
                (compressor warmup, controller hesitation). Distinct from
                ``tau_hp_minutes`` which lags the heat *output*.
            tau_hp_minutes: First-order lag on **delivered heat output Q_hp**
                (minutes). 0 (default) = instantaneous proportional gain
                (legacy behavior, byte-identical baselines). When >0, the
                model carries a Q_hp state that ramps toward steady-state
                k_c·(sp − T_a) with this time constant, then injects Q_hp
                into the room as additive heat. Matches production's
                greybox 3-state model (production has the same param at
                ``custom_components/.../pi/greybox_observer.py``). Use to
                validate τ_hp identification in greybox unit tests.
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
        self.tau_hp_minutes = tau_hp_minutes
        self._effective_setpoint: float = initial_temp
        # Q_hp state — only used when tau_hp_minutes > 0. Starts at 0,
        # matching production's snap-to-zero-on-boot initial condition.
        self._q_hp: float = 0.0

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

        tau = self.profile.tau_env
        hp_gain = self.profile.hp_gain
        if self.profile.hp_capacity is not None:
            hp_gain *= self.profile.hp_capacity.factor(self.outdoor_temp, mode)

        # Apply disturbances (minute-keyed for cadence-invariance).
        minute = tick * dt_minutes
        extra_heat = 0.0
        tau_modifier = 1.0
        for d in self._disturbances:
            intensity = d.intensity(minute)
            if intensity > 0:
                extra_heat += d.heat_gain_c_per_min * intensity * dt_minutes
                tau_modifier *= 1.0 - (1.0 - d.tau_factor) * intensity

        tau_eff = tau * tau_modifier

        # HP model: proportional controller with idle cutoff.
        # Inverter mini-splits modulate output ∝ (setpoint - room_temp).
        # The linear ODE naturally captures this via g*(sp - T) terms.
        # When room >= setpoint (heating) or room <= setpoint (cooling),
        # the HP idles — zero contribution, g_eff = 0.
        if mode == "heat":
            hp_active = self.room_temp < self._effective_setpoint
        else:
            hp_active = self.room_temp > self._effective_setpoint
        hp_gain_eff = hp_gain if hp_active else 0.0

        # Heat inputs
        solar_heat = self.solar_gain * solar_proxy * dt_minutes
        stove_heat = self.stove_gain * stove_active * dt_minutes

        if self.tau_hp_minutes > 0:
            # ── Q_hp 2-state matrix-exp path ──
            #
            # MUST match production greybox_observer.py exactly — same A
            # matrix structure, same b vector form, same matrix-exp
            # propagation. Earlier impl used Q_hp_avg as constant input
            # to T's analytical ODE, which diverged from production's
            # exact 3-state matrix-exp (production correctly accounts
            # for the time-varying Q_hp(s) within each interval). That
            # divergence made the fit's residual surface peak at the
            # WRONG τ_hp on bench data — a real model mismatch, see
            # Stage 1f finding (2026-06-03).
            #
            # State: x = [T_a, Q_hp]
            #
            # A = | -1/τ_env   1         |
            #     | 0          -1/τ_hp   |
            #
            # b = | T_out/τ_env + sources           |
            #     | hp_gain·(sp − T_a_zoh)/τ_hp     [active]
            #     | 0                                [inactive]
            #
            # Production's snap-to-zero policy: when current step is
            # inactive, Q_hp = 0 throughout. Implemented as snap at
            # start-of-step + b[1]=0; matrix-exp then keeps it at 0.
            import numpy as np
            from scipy.linalg import expm as _expm

            if not hp_active:
                self._q_hp = 0.0

            inv_tau_env = 1.0 / tau_eff
            inv_tau_hp = 1.0 / self.tau_hp_minutes

            A = np.array([
                [-inv_tau_env,  1.0          ],
                [ 0.0,         -inv_tau_hp   ],
            ], dtype=float)

            # Sources in T's row (excluding HP). dt division here cancels
            # with the explicit *dt in the heat-input expressions above.
            T_sources = (
                self.outdoor_temp * inv_tau_env
                + solar_heat / dt_minutes
                + stove_heat / dt_minutes
                + extra_heat / dt_minutes
            )
            if hp_active:
                # ZOH at start of interval: Q_hp target uses room_temp NOW.
                q_hp_target_rate = hp_gain * (self._effective_setpoint - self.room_temp)
                b1 = q_hp_target_rate * inv_tau_hp
            else:
                b1 = 0.0
            b = np.array([T_sources, b1], dtype=float)

            # x_new = exp(A·dt)·x + ψ·b   where ψ = A⁻¹·(eA − I).
            eA = _expm(A * dt_minutes)
            try:
                psi = np.linalg.solve(A, eA - np.eye(2))
            except np.linalg.LinAlgError:  # pragma: no cover — A invertible (negative eigenvalues)
                psi = np.zeros((2, 2))
            x = np.array([self.room_temp, self._q_hp], dtype=float)
            x_new = eA @ x + psi @ b
            self.room_temp = float(x_new[0])
            self._q_hp = float(x_new[1])
            if not hp_active:
                # Belt-and-suspenders: enforce snap-to-zero even after
                # matrix-exp (Q_hp should stay 0 in this branch since
                # b[1]=0 and start=0, but numerical drift is possible).
                self._q_hp = 0.0
            # Energy tracking: integral of Q_hp(s) over the interval.
            # For first-order lag, ∫₀^dt Q_hp(s) ds equals the (x_new[1] −
            # x[1]) state change times τ_hp PLUS the q_hp_target·dt term
            # contributed by b[1] over the interval. Equivalently:
            #   ∫ Q_hp ds = (q_target − Q_hp_end + Q_hp_start)·τ_hp + 0 if active
            # Simpler: use trapezoidal approximation Q_hp_avg = (Q_hp_start
            # + Q_hp_end)/2 which is correct for small dt/τ_hp and within
            # 5% for typical operating regimes.
            q_hp_avg_for_energy = (x[1] + self._q_hp) / 2.0
            thermal_output = q_hp_avg_for_energy * dt_minutes
        else:
            # ── Legacy instantaneous-gain path (tau_hp_minutes == 0) ──
            #
            # Equilibrium temperature (where room would settle with constant inputs).
            # With proportional HP: dT/dt = (T_out - T)/τ + g*(sp - T) + solar + ...
            # Rearranged: dT/dt = -(1/τ + g)*T + (T_out/τ + g*sp + solar + ...)
            # Equilibrium: T_eq = (T_out/τ + g*sp + solar) / (1/τ + g)
            total_gain = 1.0 / tau_eff + hp_gain_eff
            if total_gain == 0:
                return
            t_eq = (
                self.outdoor_temp / tau_eff
                + hp_gain_eff * self._effective_setpoint
                + solar_heat / dt_minutes  # Convert back to rate
                + stove_heat / dt_minutes
                + extra_heat / dt_minutes
            ) / total_gain

            # Exact exponential decay toward equilibrium
            decay = math.exp(-total_gain * dt_minutes)
            self.room_temp = t_eq + (self.room_temp - t_eq) * decay
            # Energy tracking — legacy "(sp − T) × g" approximation.
            thermal_output = abs(self._effective_setpoint - self.room_temp) * hp_gain_eff * dt_minutes

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
                 tau_hp_minutes: float = 0.0,
                 initial_wall_temp: float | None = None,
                 solar_gain: float = 0.0, stove_gain: float = 0.0,
                 head_sensor_offset: float = 0.0):
        """Initialize 2R2C thermal model.

        See :class:`ThermalModel` for the meaning of ``hp_lag_minutes``
        (setpoint lag) vs ``tau_hp_minutes`` (heat-output lag, matches
        production's greybox Q_hp state — default 0 preserves byte-
        identical legacy behavior).
        """
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
        self.tau_hp_minutes = tau_hp_minutes
        self._effective_setpoint: float = initial_temp
        self._q_hp: float = 0.0  # Q_hp state — only used when tau_hp_minutes > 0
        self.head_sensor_offset = head_sensor_offset

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
             q_air_extra: float = 0.0, q_wall_extra: float = 0.0,
             tick: int = 0, mode: str = "heat") -> None:
        """Advance both nodes by one time step.

        Same interface as ThermalModel for drop-in compatibility.

        ``q_air_extra`` and ``q_wall_extra`` are generic per-node heat
        injection rates (°C/min) for sources the runner has already
        split per ASHRAE convective/radiative convention or party-wall
        coupling.  They enter the same b1/b2 forcing terms as
        ``q_solar_air`` and ``q_solar_wall``.
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
        if p.hp_capacity is not None:
            g *= p.hp_capacity.factor(self.outdoor_temp, mode)

        # Apply disturbances (minute-keyed for cadence-invariance, affect
        # envelope only, like the 1R1C model).
        minute = tick * dt_minutes
        extra_heat = 0.0
        tau_modifier = 1.0
        for d in self._disturbances:
            intensity = d.intensity(minute)
            if intensity > 0:
                extra_heat += d.heat_gain_c_per_min * intensity
                tau_modifier *= 1.0 - (1.0 - d.tau_factor) * intensity
        tau_env_eff = tau_env * tau_modifier

        # Heat input rates (°C/min).
        # Solar gain split between wall (radiative absorption, released
        # via wall thermal mass) and air (convective fraction). Per
        # ASHRAE F18 Ch.18 RTS Table 14: absorbed solar at glass splits
        # ~70% radiant / 30% convective; transmitted-beam through
        # unshaded glass is ~100% radiant (lands on interior surfaces).
        # Per-profile via solar_wall_fraction (default 0.7); raise toward
        # 0.9-1.0 for sun-exposed rooms with significant direct-beam.
        wall_frac = p.solar_wall_fraction
        q_solar_total = self.solar_gain * solar_proxy
        q_solar_wall = q_solar_total * wall_frac
        q_solar_air = q_solar_total * (1.0 - wall_frac)
        q_stove = self.stove_gain * stove_active
        q_extra = extra_heat

        # HP cycling: internal thermostat turns off compressor when its
        # head sensor (room_temp + offset) is at or above setpoint (heating)
        # or at/below setpoint (cooling).  The offset models the difference
        # between the HP's head unit sensor and our room sensor.
        hp_sensed_temp = self.room_temp + self.head_sensor_offset
        if mode == "heat":
            hp_active = hp_sensed_temp < self._effective_setpoint
        else:  # cool
            hp_active = hp_sensed_temp > self._effective_setpoint
        g_eff = g if hp_active else 0.0

        # Q_hp 3-state matrix-exp path (tau_hp_minutes > 0).
        # MUST match production greybox_observer.py exactly — same A,
        # same b, same matrix-exp propagation. See ThermalModel.step for
        # the design rationale (Stage 1f fix 2026-06-03 replaced the
        # earlier Q_hp_avg-as-constant-input approach which didn't match
        # production's exact 3-state ODE).
        #
        # State: x = [T_a, T_w, Q_hp]
        #
        # A = | -(1/τ_env + 1/τ_c)   1/τ_c        1            |
        #     | 1/τ_m               -1/τ_m        0            |
        #     | 0                    0           -1/τ_hp       |
        #
        # b = | T_out/τ_env + q_solar_air + q_stove + q_extra + q_air_extra |
        #     | (q_solar_wall + q_wall_extra) / mass_ratio                  |
        #     | g·(sp − T_a_zoh) / τ_hp  [active] OR 0 [inactive]           |

        if self.tau_hp_minutes > 0:
            import numpy as np
            from scipy.linalg import expm as _expm

            if not hp_active:
                self._q_hp = 0.0

            inv_tau_env = 1.0 / tau_env_eff
            inv_tau_c = 1.0 / tau_c
            inv_tau_m = 1.0 / tau_m
            inv_tau_hp = 1.0 / self.tau_hp_minutes

            A3 = np.array([
                [-(inv_tau_env + inv_tau_c),  inv_tau_c,    1.0           ],
                [ inv_tau_m,                 -inv_tau_m,    0.0           ],
                [ 0.0,                        0.0,         -inv_tau_hp    ],
            ], dtype=float)

            b1_3 = (self.outdoor_temp * inv_tau_env
                    + q_solar_air + q_stove + q_extra + q_air_extra)
            b2_3 = (q_solar_wall + q_wall_extra) / p.mass_ratio
            if hp_active:
                b3 = g * (self._effective_setpoint - self.room_temp) * inv_tau_hp
            else:
                b3 = 0.0
            b3v = np.array([b1_3, b2_3, b3], dtype=float)

            eA3 = _expm(A3 * dt_minutes)
            try:
                psi3 = np.linalg.solve(A3, eA3 - np.eye(3))
            except np.linalg.LinAlgError:  # pragma: no cover — A invertible
                psi3 = np.zeros((3, 3))
            x3 = np.array([self.room_temp, self.wall_temp, self._q_hp], dtype=float)
            x3_new = eA3 @ x3 + psi3 @ b3v
            self.room_temp = float(x3_new[0])
            self.wall_temp = float(x3_new[1])
            self._q_hp = float(x3_new[2])
            if not hp_active:
                self._q_hp = 0.0

            # Energy tracking via trapezoidal Q_hp average (same approach
            # as 1R1C path).
            q_hp_avg_for_energy = (x3[2] + self._q_hp) / 2.0
            thermal_output = q_hp_avg_for_energy * dt_minutes
            cop = self.cop_model.cop(self.outdoor_temp, hp_setpoint, mode)
            if cop > 0:
                electrical_input = thermal_output / cop
                self.cumulative_kwh += electrical_input / 60.0
                self.cumulative_cop_weighted_output += thermal_output
            self.tick_count += 1
            return

        # ── Legacy instantaneous-gain path (tau_hp_minutes == 0) ──
        # System matrix A and forcing vector b:
        #   d/dt [T_a, T_w]^T = A * [T_a, T_w]^T + b
        #
        # A = [[-1/τ_env - g_eff - 1/τ_c,   1/τ_c ],
        #      [ 1/τ_m,                    -1/τ_m  ]]
        #
        # b = [T_out/τ_env + g_eff*sp + q_solar_air + q_stove + q_extra,
        #      q_solar_wall / mass_ratio]
        a11 = -(1.0 / tau_env_eff + g_eff + 1.0 / tau_c)
        b1 = (self.outdoor_temp / tau_env_eff
              + g_eff * self._effective_setpoint
              + q_solar_air + q_stove + q_extra + q_air_extra)

        a12 = 1.0 / tau_c
        a21 = 1.0 / tau_m
        a22 = -1.0 / tau_m

        # Wall-node forcing normalized by wall capacitance ratio.
        b2 = (q_solar_wall + q_wall_extra) / p.mass_ratio

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

        # Legacy-path energy tracking (tau_hp_minutes == 0 case only —
        # tau_hp > 0 branch above already returned with its own energy
        # accounting based on the actual Q_hp trajectory).
        thermal_output = abs(self._effective_setpoint - self.room_temp) * g_eff * dt_minutes
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
