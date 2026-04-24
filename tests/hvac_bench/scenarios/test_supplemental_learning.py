"""Simulation: learning a supplemental heat source's coefficient.

Tests whether the PI controller can estimate the pellet stove's
thermal contribution from tracking-mode observations.

Questions to answer:
1. Does the tracked setpoint meaningfully differ between burn and idle phases?
2. Can we extract a burning coefficient from within-session phase differences?
3. How does this compare to a simple session-average coefficient?
4. Does outdoor temp variation invalidate the estimate?
"""

import math
import pytest

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.house_profiles import PROFILES
from tests.hvac_bench.thermal_model import ThermalModel2R2C as ThermalModel
from tests.hvac_bench.runner import run_scenario
from tests.hvac_bench.metrics import compute_all_metrics


# ── Stove Cycling Model ──────────────────────────────────────────────────


class StoveCycleModel:
    """Simulates a pellet stove with its own thermostat.

    The stove cycles: burning → idle → burning based on room temp
    relative to its setpoint and hysteresis band.
    """

    def __init__(self, setpoint_c=21.0, upper_band=0.5, lower_band=1.0,
                 heat_rate_c_per_min=0.15, burn_min_minutes=20):
        """
        Args:
            setpoint_c: Stove thermostat setpoint in °C.
            upper_band: °C above setpoint to stop burning.
            lower_band: °C below setpoint to start burning.
            heat_rate_c_per_min: Heating contribution per minute when burning.
            burn_min_minutes: Minimum burn time before stove can stop.
        """
        self.setpoint = setpoint_c
        self.upper_band = upper_band
        self.lower_band = lower_band
        self.heat_rate = heat_rate_c_per_min
        self.burn_min = burn_min_minutes
        self.burning = False
        self.thermostat_on = True
        self.burn_time = 0.0  # minutes into current burn

    def step(self, room_temp_c, dt_minutes=15.0):
        """Update stove state and return heat contribution.

        Returns:
            heat_gain_c: Temperature gain this tick (°C).
            burning: Whether stove is currently burning.
            thermostat_on: Whether thermostat is in heat mode.
        """
        if not self.thermostat_on:
            self.burning = False
            self.burn_time = 0.0
            return 0.0, False, False

        # Hysteresis logic
        if self.burning:
            self.burn_time += dt_minutes
            if room_temp_c >= self.setpoint + self.upper_band and self.burn_time >= self.burn_min:
                self.burning = False
                self.burn_time = 0.0
        else:
            if room_temp_c <= self.setpoint - self.lower_band:
                self.burning = True
                self.burn_time = 0.0

        heat = self.heat_rate * dt_minutes if self.burning else 0.0
        return heat, self.burning, self.thermostat_on

    def turn_off(self):
        self.thermostat_on = False
        self.burning = False
        self.burn_time = 0.0

    def turn_on(self):
        self.thermostat_on = True


# ── Enhanced Run with Stove ───────────────────────────────────────────────


def run_with_stove(controller, model, stove, n_ticks, mode="heat",
                   outdoor_schedule=None, stove_on_ticks=(0, None),
                   tick_interval_min=15.0):
    """Run simulation with stove cycling.

    Args:
        stove_on_ticks: (start, end) ticks when stove thermostat is on.
            None for end means stays on.

    Returns:
        history list with additional stove fields.
    """
    dt_seconds = tick_interval_min * 60.0
    controller.set_mode(mode)

    history = []
    stove_start, stove_end = stove_on_ticks

    for tick in range(n_ticks):
        # Outdoor schedule
        if outdoor_schedule is not None:
            if callable(outdoor_schedule):
                model.outdoor_temp = outdoor_schedule(tick)

        # Stove on/off schedule
        if tick == stove_start:
            stove.turn_on()
        if stove_end is not None and tick == stove_end:
            stove.turn_off()

        # Read sensor
        sensor_reading = model.read_sensor()

        # Stove step (uses actual room temp, not sensor reading)
        stove_heat, burning, thermo_on = stove.step(model.room_temp, tick_interval_min)

        # Controller tick — stove info NOT passed as model input for now
        # (we're testing whether the controller can estimate stove effect)
        hp_setpoint = controller.tick(
            room_temp_c=sensor_reading,
            outdoor_temp_c=model.outdoor_temp,
            dt_seconds=dt_seconds,
        )

        # Apply stove heat as disturbance to thermal model
        # (stove heat is an external gain, not through the HP)
        model.room_temp += stove_heat

        # Advance thermal model with HP setpoint
        model.step(
            hp_setpoint=hp_setpoint,
            dt_minutes=tick_interval_min,
            tick=tick,
            mode=mode,
        )

        state = controller.get_state()
        desired = state.get("desired_temp", 20.5)

        history.append({
            "tick": tick,
            "room_temp": model.room_temp,
            "sensor_reading": sensor_reading,
            "desired": desired,
            "hp_setpoint": hp_setpoint,
            "raw_setpoint": state.get("raw_setpoint", float(hp_setpoint)),
            "integral": state.get("integral", 0.0),
            "ff_offset": state.get("ff_offset", 0.0),
            "error": desired - model.room_temp,
            "outdoor": model.outdoor_temp,
            "rls_obs_count": state.get("rls_obs_count", 0),
            "stove_burning": burning,
            "stove_thermostat_on": thermo_on,
            "stove_heat": stove_heat,
        })

    return history


# ── Question 1: Tracked setpoint during burn vs idle ─────────────────────


class TestTrackedSetpointPhases:
    """Does the tracked setpoint meaningfully differ between burn and idle?"""

    def test_setpoint_differs_burn_vs_idle(self):
        """HP raw setpoint should be lower during burn (less HP needed)
        and higher during idle (more HP needed).

        Uses raw (pre-quantization) setpoints to test controller intent.
        Quantized setpoints with 1°C steps create limit cycles whose phase
        alignment is fragile — testing quantized values measures phase
        coincidence rather than controller behavior.
        """
        profile = PROFILES["standard_residential"]
        ctrl = TasmotaPIAdapter({"pi_outdoor_seed_heat": PROFILES["standard_residential"].true_seed})
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=2.0)
        stove = StoveCycleModel(setpoint_c=21.0, heat_rate_c_per_min=0.15)

        # Run longer to average out phase coupling between stove cycle
        # and HP limit cycle (both respond to same temperature signal).
        history = run_with_stove(ctrl, model, stove, n_ticks=96,
                                 stove_on_ticks=(4, None))

        # Collect raw setpoints during settled stove operation (after tick 32)
        burn_setpoints = [h["raw_setpoint"] for h in history
                         if h["tick"] > 32 and h["stove_burning"]]
        idle_setpoints = [h["raw_setpoint"] for h in history
                         if h["tick"] > 32 and not h["stove_burning"]
                         and h["stove_thermostat_on"]]

        if burn_setpoints and idle_setpoints:
            avg_burn = sum(burn_setpoints) / len(burn_setpoints)
            avg_idle = sum(idle_setpoints) / len(idle_setpoints)
            print(f"\n  Burn phase avg raw setpoint: {avg_burn:.2f}")
            print(f"  Idle phase avg raw setpoint: {avg_idle:.2f}")
            print(f"  Difference: {avg_idle - avg_burn:.2f}")
            # During burn, stove adds heat → HP needs less → raw setpoint should be lower
            assert avg_burn <= avg_idle + 0.5, (
                f"Expected burn raw setpoint ≤ idle, got burn={avg_burn:.2f} idle={avg_idle:.2f}"
            )
        else:
            print(f"\n  Burn ticks: {len(burn_setpoints)}, Idle ticks: {len(idle_setpoints)}")
            pytest.skip("Not enough data in both phases")


# ── Question 2: Before/after coefficient estimation ──────────────────────


class TestBeforeAfterEstimation:
    """Can we estimate the stove coefficient from before vs during tracking?"""

    def test_before_after_gives_reasonable_coefficient(self):
        """The difference in raw (pre-quantization) setpoint before and during
        stove operation should approximate the stove's thermal contribution.

        Uses raw_setpoint rather than quantized hp_setpoint because the stove
        effect can be sub-degree — invisible through 1°C integer quantization
        but measurable in the raw PI output.
        """
        profile = PROFILES["standard_residential"]
        ctrl = TasmotaPIAdapter({"pi_outdoor_seed_heat": PROFILES["standard_residential"].true_seed})
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=2.0)
        stove = StoveCycleModel(setpoint_c=21.0, heat_rate_c_per_min=0.15)

        # Phase 1: HP active, no stove (ticks 0-15)
        # Phase 2: Stove on, HP still computing (ticks 16-47)
        history = run_with_stove(ctrl, model, stove, n_ticks=48,
                                 stove_on_ticks=(16, None))

        # Capture HP state before stove (use raw setpoint)
        pre_stove = [h for h in history if 12 <= h["tick"] <= 15]
        avg_pre_raw = sum(h["raw_setpoint"] for h in pre_stove) / len(pre_stove)
        avg_pre_integral = sum(h["integral"] for h in pre_stove) / len(pre_stove)

        # Capture HP tracked state during stove (settled period)
        during_stove = [h for h in history if 32 <= h["tick"] <= 47]
        avg_during_raw = sum(h["raw_setpoint"] for h in during_stove) / len(during_stove)
        avg_during_integral = sum(h["integral"] for h in during_stove) / len(during_stove)

        diff = avg_during_raw - avg_pre_raw
        print(f"\n  Before stove: raw_sp={avg_pre_raw:.2f}, integral={avg_pre_integral:.2f}")
        print(f"  During stove: raw_sp={avg_during_raw:.2f}, integral={avg_during_integral:.2f}")
        print(f"  Raw setpoint difference: {diff:+.2f}")

        # Stove should reduce the needed HP setpoint (raw)
        assert diff < 0, f"Expected negative diff (stove reduces HP need), got {diff:+.2f}"
        # Should be roughly in the -1 to -5 range for a moderate stove
        assert -8 < diff < 0, f"Coefficient {diff:.2f} seems out of range"


# ── Question 3: Session-average vs phase-aware ────────────────────────────


class TestSessionVsPhaseCoefficient:
    """Compare session-average coefficient with burn/idle phase-aware."""

    def test_phase_aware_is_more_accurate(self):
        """Phase-aware estimation should better predict HP need during each phase."""
        profile = PROFILES["standard_residential"]
        ctrl = TasmotaPIAdapter({"pi_outdoor_seed_heat": PROFILES["standard_residential"].true_seed})
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=2.0)
        stove = StoveCycleModel(setpoint_c=21.0, heat_rate_c_per_min=0.15)

        history = run_with_stove(ctrl, model, stove, n_ticks=64,
                                 stove_on_ticks=(8, None))

        # Pre-stove baseline
        pre = [h for h in history if 4 <= h["tick"] <= 7]
        baseline_sp = sum(h["hp_setpoint"] for h in pre) / len(pre)

        # Phase-aware during stove (after settling)
        burn_sp = [h["hp_setpoint"] for h in history
                   if h["tick"] > 24 and h["stove_burning"]]
        idle_sp = [h["hp_setpoint"] for h in history
                   if h["tick"] > 24 and not h["stove_burning"]
                   and h["stove_thermostat_on"]]
        all_sp = [h["hp_setpoint"] for h in history
                  if h["tick"] > 24 and h["stove_thermostat_on"]]

        if burn_sp and idle_sp and all_sp:
            avg_burn = sum(burn_sp) / len(burn_sp)
            avg_idle = sum(idle_sp) / len(idle_sp)
            avg_all = sum(all_sp) / len(all_sp)

            coeff_burn = avg_burn - baseline_sp
            coeff_idle = avg_idle - baseline_sp
            coeff_session = avg_all - baseline_sp

            print(f"\n  Baseline setpoint: {baseline_sp:.1f}")
            print(f"  Session avg coeff: {coeff_session:.2f}")
            print(f"  Burn phase coeff: {coeff_burn:.2f}")
            print(f"  Idle phase coeff: {coeff_idle:.2f}")
            print(f"  Phase spread: {coeff_idle - coeff_burn:.2f}")

            # If phases are meaningfully different, phase-aware is better
            if abs(coeff_idle - coeff_burn) > 0.5:
                print("  → Phase-aware provides richer signal")
            else:
                print("  → Session average is sufficient (phases not very different)")
        else:
            pytest.skip("Insufficient data in phases")


# ── Question 4: Outdoor temp variation ────────────────────────────────────


class TestOutdoorVariation:
    """Does outdoor temp variation invalidate the before/after estimate?"""

    def test_before_after_robust_to_outdoor_change(self):
        """Estimate should be similar even if outdoor temp changes during stove session."""
        profile = PROFILES["standard_residential"]

        for outdoor_change, label in [(0, "constant"), (-5, "dropping"), (+3, "rising")]:
            ctrl = TasmotaPIAdapter({"pi_outdoor_seed_heat": PROFILES["standard_residential"].true_seed})
            ctrl.set_desired_temp(20.5)
            model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=5.0)
            stove = StoveCycleModel(setpoint_c=21.0, heat_rate_c_per_min=0.15)

            def outdoor(tick, change=outdoor_change):
                return 5.0 + (tick / 48.0) * change

            history = run_with_stove(ctrl, model, stove, n_ticks=48,
                                     stove_on_ticks=(16, None),
                                     outdoor_schedule=outdoor)

            pre = [h for h in history if 12 <= h["tick"] <= 15]
            during = [h for h in history if 32 <= h["tick"] <= 47]

            if pre and during:
                pre_sp = sum(h["hp_setpoint"] for h in pre) / len(pre)
                during_sp = sum(h["hp_setpoint"] for h in during) / len(during)
                diff = during_sp - pre_sp
                print(f"\n  {label} (Δoutdoor={outdoor_change:+d}°C): "
                      f"pre={pre_sp:.1f} during={during_sp:.1f} coeff={diff:+.1f}")


# ── Stove turns off: HP resume behavior ──────────────────────────────────


class TestStoveOffResume:
    """When stove stops, does the HP resume with a reasonable setpoint?"""

    def test_hp_resumes_correctly_after_stove(self):
        """After stove session ends, HP should quickly reach a correct setpoint."""
        profile = PROFILES["standard_residential"]
        ctrl = TasmotaPIAdapter({"pi_outdoor_seed_heat": PROFILES["standard_residential"].true_seed})
        ctrl.set_desired_temp(20.5)
        model = ThermalModel(profile=profile, initial_temp=20.5, outdoor_temp=2.0)
        stove = StoveCycleModel(setpoint_c=21.0, heat_rate_c_per_min=0.15)

        # Stove on ticks 8-31, off at tick 32, HP resumes
        history = run_with_stove(ctrl, model, stove, n_ticks=48,
                                 stove_on_ticks=(8, 32))

        # After stove stops, room should stay near target
        post_stove = [h for h in history if h["tick"] >= 36]
        for h in post_stove:
            assert abs(h["room_temp"] - 20.5) < 3.0, (
                f"Tick {h['tick']}: room={h['room_temp']:.1f} after stove off "
                f"(HP setpoint={h['hp_setpoint']})"
            )

        # HP setpoint should stabilize
        late = [h["hp_setpoint"] for h in history if h["tick"] >= 40]
        if late:
            sp_range = max(late) - min(late)
            print(f"\n  Post-stove setpoint range: {sp_range}°C "
                  f"(setpoints: {late})")
