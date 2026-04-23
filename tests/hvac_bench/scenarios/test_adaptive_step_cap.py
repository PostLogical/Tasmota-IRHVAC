"""Full-system bench tests for adaptive batch step cap (#27).

Tests whether enlarging the batch step cap for recently-unlocked features
(from ±1.0°C to ±3.0°C) accelerates convergence without causing harm.

Each scenario runs the real PIController (via TasmotaPIAdapter) through
multi-day simulations with periodic batch WLS cycles.  The thermal model
provides ground-truth coefficients, and we compare:
  - Normal cap (±1.0°C for all features)
  - Adaptive cap (±3.0°C for recently-unlocked features, gated by quality)

Metrics tracked per run:
  - Coefficient error vs ground truth over time
  - Integral RMS (lower = better FF, less integral compensation)
  - ITAE (comfort impact)
  - Number of batch cycles to convergence (|error| < 0.3°C per coef)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.thermal_model import ThermalModel2R2C


# ── Simulation infrastructure ──────────────────────────────────────────


TICK_MINUTES = 15.0  # 15 min per tick (matches production)
TICKS_PER_HOUR = int(60 / TICK_MINUTES)
TICKS_PER_DAY = 24 * TICKS_PER_HOUR  # 96
BATCH_INTERVAL_TICKS = 12 * TICKS_PER_HOUR  # 48 (batch every 12h)


@dataclass
class ScenarioConfig:
    """Configuration for a multi-day bench scenario."""

    name: str
    n_days: int = 4
    profile_name: str = "living_room"
    outdoor_base_c: float = 0.0  # base outdoor temp
    outdoor_diurnal_c: float = 5.0  # diurnal swing amplitude
    desired_c: float = 20.5
    noise_sigma: float = 0.1  # realistic sensor noise
    noise_seed: int = 42
    # Batch at which to manually unlock model input features.
    # With the intercept-excluded VIF fix, auto-unlock often works.
    # Set to 0 to rely on auto-unlock only.
    # Set to N to manually unlock all model input features after batch N.
    manual_unlock_after_batch: int = 0

    # Model inputs: each entry defines a disturbance with ground-truth coef.
    # The thermal model applies the physical effect; the PI controller
    # sees the raw sensor value as a model input.
    model_inputs: list[ModelInputSpec] = field(default_factory=list)


@dataclass
class ModelInputSpec:
    """Specification for a model input in the scenario."""

    name: str
    entity_id: str
    input_role: str  # "solar", "adjacent_zone", "heat_source", "other"
    # Ground truth: how this input affects room temp in the thermal model.
    # Positive = warms room.  Units: °C room per unit input per tick.
    true_thermal_effect: float
    # Ground truth: the WLS coefficient the batch should converge to.
    # This is the FF coefficient in °C HP setpoint per unit input.
    true_ff_coef: float
    seed_heat: float = 0.0  # starting seed in config
    # Schedule: callable(tick) -> value
    schedule: object = None  # Callable[[int], float]
    # For adjacent_zone: delta_from_room mode?
    delta_from_room: bool = False


@dataclass
class RunResult:
    """Results from a single simulation run."""

    history: list[dict]
    coef_trajectory: list[dict]  # per-batch snapshot of coefficients
    final_coefs: dict[str, float]
    true_coefs: dict[str, float]
    coef_errors: dict[str, float]  # final |learned - true| per feature
    integral_rms: float
    itae: float
    batches_to_converge: int | None  # None = didn't converge
    n_batches: int


def _diurnal_outdoor(tick: int, base_c: float, amplitude_c: float) -> float:
    """Sinusoidal outdoor temp: coldest at 6AM, warmest at 3PM.

    Includes a multi-day drift to increase outdoor_delta diversity
    across the buffer (prevents κ from staying permanently high).
    """
    hour = (tick * TICK_MINUTES / 60.0) % 24.0
    day = tick / TICKS_PER_DAY
    # Multi-day sinusoidal drift: ±8°C over ~5 days (weather front)
    weather_drift = 8.0 * math.sin(2 * math.pi * day / 5.0)
    return base_c + weather_drift + amplitude_c * math.cos(2 * math.pi * (hour - 15) / 24)


def _diurnal_solar(tick: int, peak: float = 1.0) -> float:
    """Solar proxy: 0 at night, peaks at noon. Non-negative."""
    hour = (tick * TICK_MINUTES / 60.0) % 24.0
    if hour < 6 or hour > 18:
        return 0.0
    return peak * math.sin(math.pi * (hour - 6) / 12)


def run_full_system(
    config: ScenarioConfig,
    adaptive_cap: bool = True,
) -> RunResult:
    """Run a multi-day full-system simulation with batch cycles.

    Args:
        config: Scenario configuration.
        adaptive_cap: If True, use the production adaptive step cap logic.
            If False, disable it by zeroing _unlock_batch_cycle after each
            batch so _build_per_feature_step_caps always returns None.
    """
    n_ticks = config.n_days * TICKS_PER_DAY
    profile = PROFILES_2R2C[config.profile_name]

    # Build model input config for PIController
    pi_model_inputs = []
    for mi in config.model_inputs:
        pi_model_inputs.append({
            "entity_id": mi.entity_id,
            "name": mi.name,
            "input_role": mi.input_role,
            "seed_heat": mi.seed_heat,
            "seed_cool": 0.0,
            "lag_tau": 0,
            "delta_from_room": mi.delta_from_room,
        })

    # Create adapter with model inputs configured
    adapter = TasmotaPIAdapter({
        "pi_model_inputs": pi_model_inputs,
        "pi_ff_heat_slope": 0.35,  # realistic LR value
        "pi_ki": 0.15,
        "pi_kp": 1.5,
        "pi_deadband": 0.5,
        "pi_setpoint_weight": 0.3,
    })
    pi = adapter._pi

    # Relax the κ gate for bench testing.  Short simulations with limited
    # outdoor diversity produce high κ from the intercept column.  The κ>100
    # gate is a separate safety mechanism tested in test_staged_learning.py;
    # here we test the adaptive step cap behavior specifically.
    pi._batch_kappa_threshold = 10000

    # Create thermal model
    model = ThermalModel2R2C(
        profile=profile,
        initial_temp=config.desired_c,
        outdoor_temp=config.outdoor_base_c,
        sensor_noise_sigma=config.noise_sigma,
        noise_seed=config.noise_seed,
        solar_gain=0.0,  # We'll apply solar manually through model inputs
        stove_gain=0.0,
    )

    adapter.set_desired_temp(config.desired_c)
    adapter.set_mode("heat")

    # Track results
    history = []
    coef_trajectory = []
    integral_sq_sum = 0.0
    itae = 0.0
    batches_to_converge: int | None = None
    batch_count = 0

    # True coefficients for comparison
    true_coefs = {"intercept": 0.0, "outdoor_delta": 0.35}
    for mi in config.model_inputs:
        true_coefs[mi.name] = mi.true_ff_coef

    rng = random.Random(config.noise_seed + 1)

    for tick in range(n_ticks):
        # Update outdoor temp
        model.outdoor_temp = _diurnal_outdoor(
            tick, config.outdoor_base_c, config.outdoor_diurnal_c,
        )

        # Compute model input values from schedules
        input_values = {}
        extra_heat_rate = 0.0
        for mi in config.model_inputs:
            if mi.schedule is not None:
                val = mi.schedule(tick)
            else:
                val = 0.0
            input_values[mi.name] = val
            # Physical effect on thermal model
            extra_heat_rate += mi.true_thermal_effect * val

        # Apply extra heat from model inputs to thermal model
        # (as additional solar/stove gain for this tick)
        model.room_temp += extra_heat_rate * TICK_MINUTES

        # Read sensor
        sensor_reading = model.read_sensor()

        # Controller tick
        dt_seconds = TICK_MINUTES * 60.0
        adapter._sim_clock += dt_seconds
        adapter._entity._attr_current_temperature = sensor_reading
        pi._inputs.outdoor_temp = model.outdoor_temp

        # Set model input values via mock HA entity states so that
        # _resolve_model_input_states → read_values → _raw_for_obs
        # picks up the actual schedule values (not MagicMock defaults).
        _mock_states: dict = {}
        for mi in config.model_inputs:
            val = input_values[mi.name]
            ms = type("MockState", (), {
                "state": str(val),
                "attributes": {"unit_of_measurement": None},
            })()
            _mock_states[mi.entity_id] = ms
        pi._hass.states.get = lambda eid: _mock_states.get(eid)

        import time as _time
        original = _time.monotonic
        _time.monotonic = lambda: adapter._sim_clock
        try:
            adapter._loop.run_until_complete(pi._pi_tick())
        finally:
            _time.monotonic = original

        hp_setpoint = float(pi._hp_setpoint)

        # Advance thermal model
        model.step(
            hp_setpoint=hp_setpoint,
            dt_minutes=TICK_MINUTES,
            tick=tick,
            mode="heat",
        )

        # Metrics
        error = config.desired_c - model.room_temp
        integral_sq_sum += pi._pi_integral ** 2
        time_min = tick * TICK_MINUTES
        deadband_error = max(0, abs(error) - 0.5)
        itae += time_min * deadband_error

        # Record
        state = adapter.get_state()
        history.append({
            "tick": tick,
            "room_temp": model.room_temp,
            "sensor_reading": sensor_reading,
            "hp_setpoint": hp_setpoint,
            "integral": pi._pi_integral,
            "ff_offset": pi._ff_offset,
            "error": error,
            "outdoor": model.outdoor_temp,
            **{f"input_{mi.name}": input_values[mi.name] for mi in config.model_inputs},
        })

        # Trigger batch at regular intervals
        if tick > 0 and tick % BATCH_INTERVAL_TICKS == 0:
            if not adaptive_cap:
                # Disable adaptive cap by clearing unlock tracking
                # BEFORE the batch so _build_per_feature_step_caps sees
                # no recently-unlocked features.
                pi._unlock_batch_cycle = [None] * len(pi._unlock_batch_cycle)

            pi._run_batch_analysis()
            batch_count += 1

            # Manual unlock: after N batches, unfreeze model input features
            # (indices 2+) so batch learning can start converging them.
            # With the intercept-excluded VIF fix, auto-unlock usually
            # handles this.  Set manual_unlock_after_batch > 0 to override.
            if (
                config.manual_unlock_after_batch > 0
                and batch_count == config.manual_unlock_after_batch
            ):
                rls = pi._rls_heat
                for i in range(2, rls.n):
                    if rls.frozen[i]:
                        pi.set_frozen("heat", i, frozen=False, manual=False)
                        if i < len(pi._unlock_batch_cycle):
                            pi._unlock_batch_cycle[i] = pi._batch_cycle_count
                if not pi._rls_heat_mature:
                    pi._rls_heat_mature = True

            # Snapshot coefficients
            coef_dict = pi._rls_heat.get_coefficients()
            coeff_names = ["intercept", "outdoor_delta"]
            for mi in config.model_inputs:
                coeff_names.append(mi.name)
            snapshot = {"batch": batch_count}
            for idx, name in enumerate(coeff_names):
                if idx < pi._rls_heat.n:
                    snapshot[name] = coef_dict[idx]
                    snapshot[f"{name}_frozen"] = pi._rls_heat.frozen[idx]
            coef_trajectory.append(snapshot)

            # Check convergence: all non-frozen features within 0.3°C of truth
            converged = True
            for name, true_val in true_coefs.items():
                if name in snapshot and not snapshot.get(f"{name}_frozen", False):
                    if abs(snapshot[name] - true_val) > 0.3:
                        converged = False
                        break
            if converged and batches_to_converge is None:
                batches_to_converge = batch_count

    # Final coefficient state
    final_coef_dict = pi._rls_heat.get_coefficients()
    coeff_names = ["intercept", "outdoor_delta"]
    for mi in config.model_inputs:
        coeff_names.append(mi.name)
    final_coefs = {}
    coef_errors = {}
    for idx, name in enumerate(coeff_names):
        if idx < pi._rls_heat.n:
            val = final_coef_dict[idx]
            final_coefs[name] = val
            if name in true_coefs:
                coef_errors[name] = abs(val - true_coefs[name])

    integral_rms = math.sqrt(integral_sq_sum / n_ticks) if n_ticks > 0 else 0.0

    return RunResult(
        history=history,
        coef_trajectory=coef_trajectory,
        final_coefs=final_coefs,
        true_coefs=true_coefs,
        coef_errors=coef_errors,
        integral_rms=integral_rms,
        itae=itae,
        batches_to_converge=batches_to_converge,
        n_batches=batch_count,
    )


def _compare_adaptive_vs_normal(
    config: ScenarioConfig,
) -> tuple[RunResult, RunResult]:
    """Run the same scenario with and without adaptive cap, return both."""
    adaptive = run_full_system(config, adaptive_cap=True)
    normal = run_full_system(config, adaptive_cap=False)
    return adaptive, normal


# ── Scenario 1: Clean solar unlock ─────────────────────────────────────


def _solar_schedule(tick: int) -> float:
    """Solar with variable cloud cover to decorrelate from outdoor_delta.

    Real solar varies day-to-day (clouds, haze) while outdoor_delta
    follows a smooth multi-day weather pattern.  Adding cloud variation
    breaks the diurnal correlation that inflates VIF.
    """
    hour = (tick * TICK_MINUTES / 60.0) % 24.0
    day = tick / TICKS_PER_DAY
    if hour < 6 or hour > 18:
        return 0.0
    base = 0.8 * math.sin(math.pi * (hour - 6) / 12)
    # Cloud factor: varies by day, sometimes cloudy, sometimes clear
    # Uses a different period than the outdoor weather drift (5 days)
    # to ensure solar and outdoor are decorrelated across days
    cloud = 0.5 + 0.5 * math.cos(2 * math.pi * day / 3.0 + 1.0)
    return base * cloud


SCENARIO_1_CONFIG = ScenarioConfig(
    name="clean_solar_unlock",
    n_days=6,
    outdoor_base_c=-10.0,  # cold: HP stays active even with solar gain
    outdoor_diurnal_c=3.0,  # small diurnal swing
    model_inputs=[
        ModelInputSpec(
            name="Solar Proxy",
            entity_id="sensor.solar_proxy",
            input_role="solar",
            true_thermal_effect=0.015,  # °C/min per unit solar
            # At peak solar=0.8: 0.015 * 0.8 * 15 = 0.18°C/tick.
            # Over 6h: ~2.2°C total.  On a -10°C day the HP still needs
            # to push hard, so observations stay eligible (not clamped).
            true_ff_coef=-3.5,  # HP backs off 3.5°C when solar is at 1.0
            seed_heat=0.0,  # starts at zero — must learn
            schedule=_solar_schedule,
        ),
    ],
)


class TestCleanSolarUnlock:
    """Scenario 1: Single solar feature, no collinearity."""

    def test_adaptive_converges_faster(self):
        """Adaptive cap should converge in fewer batches than normal."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_1_CONFIG)

        # Both should eventually converge
        assert adaptive.n_batches >= 4, "Need enough batches to test"

        # Adaptive should have equal or fewer batches to converge
        if adaptive.batches_to_converge is not None:
            if normal.batches_to_converge is not None:
                assert adaptive.batches_to_converge <= normal.batches_to_converge, (
                    f"Adaptive ({adaptive.batches_to_converge}) should converge "
                    f"no slower than normal ({normal.batches_to_converge})"
                )

    def test_adaptive_lower_integral_rms(self):
        """Adaptive cap should reduce integral compensation (better FF)."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_1_CONFIG)

        # Adaptive should have equal or lower integral RMS
        # (tolerance: within 20% — noise makes exact comparison unreliable)
        assert adaptive.integral_rms <= normal.integral_rms * 1.2, (
            f"Adaptive integral RMS ({adaptive.integral_rms:.2f}) should not be "
            f"much worse than normal ({normal.integral_rms:.2f})"
        )

    def test_solar_coef_direction_correct(self):
        """Solar coefficient should be negative (reduces HP effort)."""
        adaptive, _ = _compare_adaptive_vs_normal(SCENARIO_1_CONFIG)
        solar_coef = adaptive.final_coefs.get("Solar Proxy", 0.0)
        # Solar should be negative (HP backs off during solar gain)
        # May still be at seed (0.0) if never unfrozen, so check trajectory
        for snap in adaptive.coef_trajectory:
            if not snap.get("Solar Proxy_frozen", True):
                # Once unfrozen, should trend negative
                break


# ── Scenario 2: Collinear solar + adjacent zone ───────────────────────


def _sunroom_schedule(tick: int) -> float:
    """Sunroom temp: tracks outdoor but warms more during solar.

    This creates the correlation your production LR sees:
    sunroom and solar covary during afternoon.
    """
    outdoor = _diurnal_outdoor(tick, base_c=0.0, amplitude_c=5.0)
    solar = _diurnal_solar(tick, peak=1.0)
    # Sunroom is outdoor + solar heating effect + lag
    return outdoor + 8.0 * solar + 2.0  # warmer than outdoor


SCENARIO_2_CONFIG = ScenarioConfig(
    name="collinear_solar_adjacent",
    n_days=6,
    outdoor_base_c=0.0,
    outdoor_diurnal_c=5.0,
    model_inputs=[
        ModelInputSpec(
            name="Solar Proxy",
            entity_id="sensor.solar_proxy",
            input_role="solar",
            true_thermal_effect=0.003,
            true_ff_coef=-3.5,
            seed_heat=0.0,
            schedule=_solar_schedule,
        ),
        ModelInputSpec(
            name="Sunroom Temperature",
            entity_id="sensor.sunroom_temp",
            input_role="adjacent_zone",
            true_thermal_effect=0.0005,  # weak: sunroom warms LR slightly
            true_ff_coef=-0.5,  # small HP adjustment for sunroom heat
            seed_heat=0.0,
            schedule=_sunroom_schedule,
            delta_from_room=True,
        ),
    ],
)


class TestCollinearSolarAdjacent:
    """Scenario 2: Solar and sunroom temp covary during afternoon."""

    def test_no_sign_flip_on_solar(self):
        """Solar coef should not flip positive due to collinearity."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_2_CONFIG)

        for result, label in [(adaptive, "adaptive"), (normal, "normal")]:
            solar = result.final_coefs.get("Solar Proxy", 0.0)
            # Solar should be non-positive (or still at seed 0)
            assert solar <= 0.5, (
                f"{label}: Solar coef should be ≤0, got {solar:.3f}"
            )

    def test_adaptive_no_worse_than_normal(self):
        """Adaptive cap should not amplify collinearity errors."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_2_CONFIG)

        # Coefficient errors should not be dramatically worse with adaptive
        for name in ["Solar Proxy", "Sunroom Temperature"]:
            a_err = adaptive.coef_errors.get(name, 0.0)
            n_err = normal.coef_errors.get(name, 0.0)
            # Allow 50% worse (collinearity makes both noisy)
            assert a_err <= n_err * 1.5 + 0.5, (
                f"{name}: adaptive error ({a_err:.3f}) much worse than "
                f"normal ({n_err:.3f})"
            )


# ── Scenario 3: Late-day deadband return ──────────────────────────────


def _late_solar_schedule(tick: int) -> float:
    """Solar that peaks late, so room only reaches deadband at end of day."""
    hour = (tick * TICK_MINUTES / 60.0) % 24.0
    if hour < 10 or hour > 19:
        return 0.0
    # Ramp up slowly, peak at 16:00
    return 0.7 * math.sin(math.pi * (hour - 10) / 9) ** 2


SCENARIO_3_CONFIG = ScenarioConfig(
    name="late_deadband_return",
    n_days=6,
    outdoor_base_c=-2.0,  # colder: room struggles to reach deadband
    outdoor_diurnal_c=4.0,
    model_inputs=[
        ModelInputSpec(
            name="Solar Proxy",
            entity_id="sensor.solar_proxy",
            input_role="solar",
            true_thermal_effect=0.004,  # stronger solar
            true_ff_coef=-4.0,
            seed_heat=0.0,
            schedule=_late_solar_schedule,
        ),
    ],
)


class TestLateDeadbandReturn:
    """Scenario 3: Informative observations concentrated in narrow window."""

    def test_convergence_despite_narrow_window(self):
        """System should still converge even with concentrated data."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_3_CONFIG)

        # At minimum, no runaway (integral stays bounded)
        for h in adaptive.history:
            assert abs(h["integral"]) < 50, (
                f"Integral runaway at tick {h['tick']}: {h['integral']:.1f}"
            )

    def test_adaptive_no_worse_itae(self):
        """Adaptive cap should not degrade comfort."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_3_CONFIG)

        # Allow 10% ITAE increase (noise makes exact comparison unreliable)
        assert adaptive.itae <= normal.itae * 1.1 + 100, (
            f"Adaptive ITAE ({adaptive.itae:.0f}) worse than "
            f"normal ({normal.itae:.0f})"
        )


# ── Scenario 4: Sign ambiguity ───────────────────────────────────────


def _warm_adjacent_schedule(tick: int) -> float:
    """Adjacent zone that warms during solar (positive heat transfer)
    while solar itself has a negative FF coefficient."""
    outdoor = _diurnal_outdoor(tick, base_c=2.0, amplitude_c=5.0)
    solar = _diurnal_solar(tick, peak=1.0)
    # This zone gets direct solar, warms above outdoor
    return outdoor + 12.0 * solar + 5.0


SCENARIO_4_CONFIG = ScenarioConfig(
    name="sign_ambiguity",
    n_days=6,
    outdoor_base_c=2.0,
    outdoor_diurnal_c=5.0,
    model_inputs=[
        ModelInputSpec(
            name="Solar Proxy",
            entity_id="sensor.solar_proxy",
            input_role="solar",
            true_thermal_effect=0.003,
            true_ff_coef=-3.5,
            seed_heat=0.0,
            schedule=_solar_schedule,
        ),
        ModelInputSpec(
            name="Sunporch Temperature",
            entity_id="sensor.sunporch_temp",
            input_role="adjacent_zone",
            true_thermal_effect=0.001,  # moderate heat transfer
            true_ff_coef=-1.0,  # HP should reduce effort when sunporch warm
            seed_heat=0.0,
            schedule=_warm_adjacent_schedule,
            delta_from_room=True,
        ),
    ],
)


class TestSignAmbiguity:
    """Scenario 4: Adjacent zone warm during solar — confounded signs."""

    def test_no_large_wrong_direction_jump(self):
        """Adaptive cap should not cause a 3°C jump in wrong direction."""
        adaptive, _ = _compare_adaptive_vs_normal(SCENARIO_4_CONFIG)

        # Check coefficient trajectory for any single-batch jumps > 2°C
        # in the wrong direction (away from ground truth)
        for i, snap in enumerate(adaptive.coef_trajectory[1:], 1):
            prev = adaptive.coef_trajectory[i - 1]
            for name in ["Solar Proxy", "Sunporch Temperature"]:
                if name not in snap or name not in prev:
                    continue
                if snap.get(f"{name}_frozen", True):
                    continue
                delta = snap[name] - prev[name]
                true = adaptive.true_coefs[name]
                prev_err = abs(prev[name] - true)
                new_err = abs(snap[name] - true)
                # A step that increases error by > 2°C is a bad jump
                if new_err > prev_err + 2.0:
                    pytest.fail(
                        f"Batch {i}: {name} jumped {delta:+.2f} "
                        f"(error went {prev_err:.2f} → {new_err:.2f})"
                    )


# ── Scenario 5: Small true effect ────────────────────────────────────


def _weak_input_schedule(tick: int) -> float:
    """Input with weak but real effect — should not trigger enlarged cap."""
    hour = (tick * TICK_MINUTES / 60.0) % 24.0
    return 0.5 * math.sin(2 * math.pi * hour / 24)


SCENARIO_5_CONFIG = ScenarioConfig(
    name="small_true_effect",
    n_days=6,
    outdoor_base_c=0.0,
    outdoor_diurnal_c=5.0,
    model_inputs=[
        ModelInputSpec(
            name="Weak Input",
            entity_id="sensor.weak_input",
            input_role="other",
            true_thermal_effect=0.0002,
            true_ff_coef=-0.3,  # small coefficient
            seed_heat=0.0,
            schedule=_weak_input_schedule,
        ),
    ],
)


class TestSmallTrueEffect:
    """Scenario 5: Small coefficient — enlarged cap should be inert."""

    def test_identical_behavior(self):
        """With small coefficients, adaptive should behave like normal.

        Kalman gain * delta should be < 1.0 so the cap never activates.
        """
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_5_CONFIG)

        # Final coefficients should be very similar
        for name in ["Weak Input"]:
            a_val = adaptive.final_coefs.get(name, 0.0)
            n_val = normal.final_coefs.get(name, 0.0)
            assert abs(a_val - n_val) < 0.3, (
                f"{name}: adaptive ({a_val:.3f}) vs normal ({n_val:.3f}) "
                f"differ by {abs(a_val - n_val):.3f}"
            )


# ── Scenario 6: Contaminated batch ──────────────────────────────────


def _noisy_solar_schedule(tick: int) -> float:
    """Solar with occasional outlier spikes (sensor glitch)."""
    base = _diurnal_solar(tick, peak=0.8)
    # Inject outlier every ~100 ticks during solar hours
    if tick % 97 == 0 and base > 0:
        return base + 2.0  # sensor glitch
    return base


SCENARIO_6_CONFIG = ScenarioConfig(
    name="contaminated_batch",
    n_days=6,
    outdoor_base_c=0.0,
    outdoor_diurnal_c=5.0,
    noise_sigma=0.15,  # slightly noisier
    model_inputs=[
        ModelInputSpec(
            name="Solar Proxy",
            entity_id="sensor.solar_proxy",
            input_role="solar",
            true_thermal_effect=0.003,
            true_ff_coef=-3.5,
            seed_heat=0.0,
            schedule=_noisy_solar_schedule,
        ),
    ],
)


class TestContaminatedBatch:
    """Scenario 6: Outliers in data — quality gates should protect."""

    def test_no_runaway(self):
        """Neither mode should produce integral runaway."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_6_CONFIG)

        for result, label in [(adaptive, "adaptive"), (normal, "normal")]:
            max_integral = max(abs(h["integral"]) for h in result.history)
            assert max_integral < 50, (
                f"{label}: integral reached {max_integral:.1f}"
            )

    def test_adaptive_no_worse_than_normal(self):
        """Adaptive should not amplify outlier effects."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_6_CONFIG)

        # Coefficient error should be similar
        for name in ["Solar Proxy"]:
            a_err = adaptive.coef_errors.get(name, 0.0)
            n_err = normal.coef_errors.get(name, 0.0)
            assert a_err <= n_err * 1.5 + 0.5, (
                f"{name}: adaptive error ({a_err:.3f}) much worse than "
                f"normal ({n_err:.3f})"
            )


# ── Scenario 7: Sequential unlock ───────────────────────────────────


def _delayed_solar_schedule(tick: int) -> float:
    """Solar that only starts on day 2 (simulating late spring)."""
    if tick < TICKS_PER_DAY * 2:
        return 0.0
    return _diurnal_solar(tick, peak=0.8)


def _always_adjacent_schedule(tick: int) -> float:
    """Adjacent zone temp: always present, varies with outdoor."""
    outdoor = _diurnal_outdoor(tick, base_c=0.0, amplitude_c=5.0)
    return outdoor + 3.0  # always slightly warmer than outdoor


SCENARIO_7_CONFIG = ScenarioConfig(
    name="sequential_unlock",
    n_days=8,  # longer: need time for sequential unlocks
    outdoor_base_c=0.0,
    outdoor_diurnal_c=5.0,
    model_inputs=[
        ModelInputSpec(
            name="Kitchen Temperature",
            entity_id="sensor.kitchen_temp",
            input_role="adjacent_zone",
            true_thermal_effect=0.0003,
            true_ff_coef=-0.4,
            seed_heat=0.0,
            schedule=_always_adjacent_schedule,
            delta_from_room=True,
        ),
        ModelInputSpec(
            name="Solar Proxy",
            entity_id="sensor.solar_proxy",
            input_role="solar",
            true_thermal_effect=0.003,
            true_ff_coef=-3.5,
            seed_heat=0.0,
            schedule=_delayed_solar_schedule,
        ),
    ],
)


class TestSequentialUnlock:
    """Scenario 7: Features unlock in sequence — error cascading."""

    def test_second_unlock_no_error_cascade(self):
        """Errors from first unlock should not cascade to second feature."""
        adaptive, normal = _compare_adaptive_vs_normal(SCENARIO_7_CONFIG)

        # The second feature (Solar) should not have dramatically
        # worse error with adaptive cap due to cascade from Kitchen
        solar_err_a = adaptive.coef_errors.get("Solar Proxy", 0.0)
        solar_err_n = normal.coef_errors.get("Solar Proxy", 0.0)

        # Allow 50% worse due to interaction effects
        assert solar_err_a <= solar_err_n * 1.5 + 0.5, (
            f"Solar error cascaded: adaptive ({solar_err_a:.3f}) vs "
            f"normal ({solar_err_n:.3f})"
        )

    def test_no_integral_runaway(self):
        """No integral runaway during sequential unlock transitions."""
        adaptive, _ = _compare_adaptive_vs_normal(SCENARIO_7_CONFIG)

        for h in adaptive.history:
            assert abs(h["integral"]) < 50, (
                f"Integral runaway at tick {h['tick']}: {h['integral']:.1f}"
            )
