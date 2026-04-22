"""Active plant identification via relay test + step-hold (Layer 3).

Pure computation — no Home Assistant dependencies.  Manages a relay
feedback test that toggles HP setpoint ±N°C to produce controlled
oscillations, followed by a step-and-hold phase for area method
validation.

At the low setpoint, the HP is off (free-fall — pure building physics).
At the high setpoint, the HP runs at full power (step response data).
With ±2°C default amplitude, the room oscillates ~±1°C within comfort.
With min/max amplitude, the HP is definitely off/on — direct relay.

The relay test produces:
- T_u (ultimate period) and a (oscillation amplitude)
- K_u = 4h/(πa) — ultimate gain
- Each half-cycle feeds step response + area method providers

Literature: Åström & Hägglund (1984), relay feedback autotuning.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..plant_model import ObservationContext, ParameterEstimate, PlantTestCommand

_LOGGER = logging.getLogger(__name__)

# Minimum time in a relay phase before accepting a midpoint crossing (minutes).
# Prevents false triggers from noise right after toggling.
_MIN_PHASE_DURATION_MIN = 5.0

# Maximum time in a single relay phase before giving up (minutes).
_MAX_PHASE_DURATION_MIN = 180.0

# Maximum total test duration (minutes) including all phases.
_MAX_TOTAL_DURATION_MIN = 480.0  # 8 hours

# Minimum step-hold duration (minutes) before completing.
_MIN_STEP_HOLD_MIN = 60.0

# Maximum step-hold duration (minutes).
_MAX_STEP_HOLD_MIN = 240.0  # 4 hours


class PlantTestProvider:
    """Relay test + step-hold state machine for active plant identification."""

    def __init__(self) -> None:
        # Configuration (set by start())
        self._baseline_c: float = 0.0
        self._amplitude_c: float = 2.0
        self._comfort_min_c: float = 0.0
        self._comfort_max_c: float = 50.0
        self._n_cycles: int = 4
        self._response_lag: float = 15.0

        # State
        self._active: bool = False
        self._phase: str = "idle"  # idle, relay_high, relay_low, step_hold, complete, aborted
        self._start_time: float = 0.0
        self._phase_start_time: float = 0.0
        self._cycle_count: int = 0  # completed full cycles (high→low = 1 cycle)
        self._half_cycle_count: int = 0  # completed half-cycles

        # Relay measurement data
        self._midpoint_c: float = 0.0  # room temp midpoint for crossing detection
        self._crossing_times: list[float] = []  # monotonic times of midpoint crossings
        self._peak_temps: list[float] = []  # peak room temps during relay_high phases
        self._trough_temps: list[float] = []  # trough room temps during relay_low phases
        self._current_peak: float = -999.0
        self._current_trough: float = 999.0

        # Step-hold tracking
        self._step_hold_start: float = 0.0
        self._step_hold_ctx: ObservationContext | None = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._active

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(
        self,
        baseline_setpoint_c: float,
        amplitude_c: float,
        current_c: float,
        comfort_min_c: float,
        comfort_max_c: float,
        n_cycles: int = 4,
        response_lag: float = 15.0,
    ) -> None:
        """Start the relay test."""
        self._baseline_c = baseline_setpoint_c
        self._amplitude_c = amplitude_c
        self._comfort_min_c = comfort_min_c
        self._comfort_max_c = comfort_max_c
        self._n_cycles = n_cycles
        self._response_lag = response_lag

        self._active = True
        self._phase = "relay_high"
        self._start_time = 0.0  # set on first tick
        self._phase_start_time = 0.0
        self._cycle_count = 0
        self._half_cycle_count = 0

        self._midpoint_c = current_c  # crossing detection relative to starting temp
        self._crossing_times = []
        self._peak_temps = []
        self._trough_temps = []
        self._current_peak = current_c
        self._current_trough = current_c

        self._step_hold_start = 0.0
        self._step_hold_ctx = None

        _LOGGER.info(
            "Plant test started: baseline=%d°C, amplitude=±%d°C, "
            "midpoint=%.1f°C, comfort=[%.1f, %.1f]°C, %d cycles",
            baseline_setpoint_c, amplitude_c, current_c,
            comfort_min_c, comfort_max_c, n_cycles,
        )

    def tick(self, now_mono: float, current_c: float) -> PlantTestCommand:
        """Advance the state machine by one tick.

        Returns a PlantTestCommand with the setpoint to send.
        """
        if not self._active:
            return PlantTestCommand(setpoint_c=self._baseline_c, phase="idle")

        # Initialize start time on first tick.
        # Don't reset _phase_start_time here — it's 0.0 from start(),
        # which means phase_elapsed is large on the first tick, allowing
        # immediate crossing detection (no relay toggle has happened yet).
        if self._start_time == 0.0:
            self._start_time = now_mono

        # ── Safety check ─────────────────────────────────────────
        if current_c < self._comfort_min_c or current_c > self._comfort_max_c:
            _LOGGER.warning(
                "Plant test ABORTED: room temp %.1f°C outside comfort "
                "bounds [%.1f, %.1f]°C",
                current_c, self._comfort_min_c, self._comfort_max_c,
            )
            return self._abort()

        # ── Total duration check ─────────────────────────────────
        total_elapsed = (now_mono - self._start_time) / 60.0
        if total_elapsed > _MAX_TOTAL_DURATION_MIN:
            _LOGGER.info(
                "Plant test completing: maximum duration %.0f min reached",
                total_elapsed,
            )
            return self._complete()

        phase_elapsed = (now_mono - self._phase_start_time) / 60.0

        # ── Relay phases ─────────────────────────────────────────
        if self._phase in ("relay_high", "relay_low"):
            return self._tick_relay(now_mono, current_c, phase_elapsed)

        # ── Step-hold phase ──────────────────────────────────────
        if self._phase == "step_hold":
            return self._tick_step_hold(now_mono, current_c, phase_elapsed)

        return PlantTestCommand(setpoint_c=self._baseline_c, phase=self._phase)

    def abort(self) -> None:
        """Abort the test externally."""
        if self._active:
            self._abort()

    def get_results(self) -> dict[str, ParameterEstimate] | None:
        """Get identification results after completion.

        Returns dict with 'k_u', 'period', 'amplitude' estimates,
        or None if test didn't complete or no valid data.
        """
        if self._phase != "complete":
            return None

        if len(self._crossing_times) < 4:
            # Need at least 2 full cycles for reliable T_u
            _LOGGER.info("Plant test: insufficient crossings (%d) for results",
                         len(self._crossing_times))
            return None

        # Compute T_u from crossing intervals
        # Each pair of same-direction crossings gives one period
        periods: list[float] = []
        for i in range(2, len(self._crossing_times)):
            # Crossings alternate direction, so every 2nd is same direction
            period = (self._crossing_times[i] - self._crossing_times[i - 2]) / 60.0
            if period > 0:
                periods.append(period)

        if not periods:
            return None

        t_u = sum(periods) / len(periods)

        # Compute amplitude from peaks and troughs
        if not self._peak_temps or not self._trough_temps:
            return None
        avg_peak = sum(self._peak_temps) / len(self._peak_temps)
        avg_trough = sum(self._trough_temps) / len(self._trough_temps)
        a = (avg_peak - avg_trough) / 2.0

        if a <= 0.05:  # Less than sensor noise
            _LOGGER.info("Plant test: oscillation amplitude too small (%.3f°C)", a)
            return None

        # K_u = 4h/(πa) where h = amplitude_c (relay half-amplitude in °C)
        h = self._amplitude_c
        k_u = 4.0 * h / (math.pi * a)

        n_cycles = len(periods)
        confidence = min(1.0, n_cycles / 3.0)

        _LOGGER.info(
            "Plant test results: T_u=%.1f min, a=%.3f°C, K_u=%.2f, "
            "h=%d°C, %d periods",
            t_u, a, k_u, self._amplitude_c, n_cycles,
        )

        return {
            "k_u": ParameterEstimate(
                value=k_u, confidence=confidence,
                source="plant_test", observations=n_cycles,
            ),
            "period": ParameterEstimate(
                value=t_u, confidence=confidence,
                source="plant_test", observations=n_cycles,
            ),
            "amplitude": ParameterEstimate(
                value=a, confidence=confidence,
                source="plant_test", observations=n_cycles,
            ),
        }

    # ── Internal state machine ───────────────────────────────────

    def _tick_relay(
        self, now_mono: float, current_c: float, phase_elapsed: float
    ) -> PlantTestCommand:
        """Handle one tick during relay_high or relay_low phase."""

        # Track peaks and troughs
        if self._phase == "relay_high":
            self._current_peak = max(self._current_peak, current_c)
            setpoint = self._baseline_c + self._amplitude_c
        else:
            self._current_trough = min(self._current_trough, current_c)
            setpoint = self._baseline_c - self._amplitude_c

        # Phase timeout
        if phase_elapsed > _MAX_PHASE_DURATION_MIN:
            _LOGGER.info(
                "Plant test: relay phase %s timed out after %.0f min, "
                "completing with available data",
                self._phase, phase_elapsed,
            )
            return self._transition_to_step_hold(now_mono, current_c)

        # Midpoint crossing detection (with minimum phase duration)
        if phase_elapsed >= _MIN_PHASE_DURATION_MIN:
            crossed = False
            if self._phase == "relay_high" and current_c > self._midpoint_c:
                crossed = True
            elif self._phase == "relay_low" and current_c < self._midpoint_c:
                crossed = True

            if crossed:
                self._crossing_times.append(now_mono)
                self._half_cycle_count += 1

                if self._phase == "relay_high":
                    self._peak_temps.append(self._current_peak)
                    _LOGGER.debug(
                        "Plant test: midpoint crossed upward (%.2f°C > %.2f°C), "
                        "peak=%.2f°C, half-cycle %d",
                        current_c, self._midpoint_c,
                        self._current_peak, self._half_cycle_count,
                    )
                    # Transition to relay_low
                    self._phase = "relay_low"
                    self._phase_start_time = now_mono
                    self._current_trough = current_c
                    setpoint = self._baseline_c - self._amplitude_c
                else:
                    self._trough_temps.append(self._current_trough)
                    self._cycle_count += 1
                    _LOGGER.debug(
                        "Plant test: midpoint crossed downward (%.2f°C < %.2f°C), "
                        "trough=%.2f°C, cycle %d/%d complete",
                        current_c, self._midpoint_c,
                        self._current_trough, self._cycle_count, self._n_cycles,
                    )
                    # Check if we have enough cycles
                    if self._cycle_count >= self._n_cycles:
                        return self._transition_to_step_hold(now_mono, current_c)
                    # Transition to relay_high
                    self._phase = "relay_high"
                    self._phase_start_time = now_mono
                    self._current_peak = current_c
                    setpoint = self._baseline_c + self._amplitude_c

        return PlantTestCommand(setpoint_c=setpoint, phase=self._phase)

    def _tick_step_hold(
        self, now_mono: float, current_c: float, phase_elapsed: float
    ) -> PlantTestCommand:
        """Handle one tick during step-hold phase."""
        setpoint = self._baseline_c + self._amplitude_c

        # Check for step-hold completion
        if phase_elapsed >= _MAX_STEP_HOLD_MIN:
            return self._complete()

        # The step-hold data is consumed externally by the orchestrator,
        # which feeds it to the step_response and area_method providers.
        # We just hold the setpoint and let the orchestrator handle it.

        return PlantTestCommand(setpoint_c=setpoint, phase="step_hold")

    def _transition_to_step_hold(
        self, now_mono: float, current_c: float
    ) -> PlantTestCommand:
        """Transition from relay to step-hold phase."""
        self._phase = "step_hold"
        self._phase_start_time = now_mono
        self._step_hold_start = now_mono

        setpoint = self._baseline_c + self._amplitude_c

        # Create observation context for the step-hold (used by orchestrator)
        self._step_hold_ctx = ObservationContext(
            start_time=now_mono,
            baseline_temp=current_c,
            target_temp=current_c + self._amplitude_c,
            step_magnitude=self._amplitude_c,
            ff_offset=0.0,  # No FF during test
        )

        _LOGGER.info(
            "Plant test: relay complete (%d cycles), transitioning to step-hold "
            "at %d°C",
            self._cycle_count, setpoint,
        )

        return PlantTestCommand(setpoint_c=setpoint, phase="step_hold")

    def _complete(self) -> PlantTestCommand:
        """Mark test as complete."""
        self._phase = "complete"
        self._active = False
        _LOGGER.info(
            "Plant test complete: %d cycles, %d crossings, "
            "%d peaks, %d troughs",
            self._cycle_count, len(self._crossing_times),
            len(self._peak_temps), len(self._trough_temps),
        )
        return PlantTestCommand(setpoint_c=self._baseline_c, phase="complete")

    def _abort(self) -> PlantTestCommand:
        """Abort the test."""
        self._phase = "aborted"
        self._active = False
        _LOGGER.warning("Plant test aborted after %d cycles", self._cycle_count)
        return PlantTestCommand(setpoint_c=self._baseline_c, phase="aborted")

    # ── Persistence ──────────────────────────────────────────────────
    # Plant test state is not persisted — it's a transient operation.
    # If HA restarts during a test, the test is simply lost.
