"""Automatic setpoint perturbation for plant identification (Layer 2.5).

Pure computation — no Home Assistant dependencies.  Injects a ±1°C offset
into the PI target when steady state is detected, holds long enough for
the room to respond, then restores.  PI stays in control throughout.

Each cycle is a single direction (±1°C → restore).  Cycles alternate
direction to get bidirectional data for gain asymmetry identification
without ever swinging >1°C from the current operating point.

Frequency is convergence-gated: perturbs often when plant ID has low
confidence, backs off as confidence grows, stops when converged.  Stalls
after N cycles without improvement → HA Repair.

Literature: Radecki & Hencey 2015 (self-excitation for building thermal
estimation), Liu & Gao 2012 (bidirectional step tests), Bouchié et al.
2022 (ISABELE binary heating signals).
"""

from __future__ import annotations

import enum
import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

AMPLITUDE_C: float = 1.0  # ±1°C, hardcoded (HP quantization minimum)
STEADY_STATE_DWELL_S: float = 600.0  # 10 min of continuous steady state
MIN_HOLD_S: float = 3600.0  # 60 min minimum at perturbed setpoint
MAX_HOLD_S: float = 10800.0  # 3 hr maximum before advancing anyway
CONVERGENCE_STOP: float = 0.9  # stop perturbing above this confidence
CONVERGENCE_DELTA: float = 0.05  # minimum improvement to reset stall counter
STALL_CAP: int = 5  # cycles without improvement before stalling
RESEARCH_STALL_CAP: int = 30  # cycles without improvement before stalling, research mode
INFORMATIVE_DELTA_C: float = 0.4  # ≈4×σ_v: cycles below this don't count toward stall
FORCE_TIMEOUT_S: float = 3600.0  # 60 min timeout for perturb_now


class PerturbState(enum.Enum):
    """State machine states."""
    IDLE = "idle"
    WAITING = "waiting"
    STEP_ACTIVE = "step_active"
    RESTORE = "restore"
    STALLED = "stalled"


class AutoPerturbation:
    """Automatic ±1°C perturbation state machine for plant identification.

    State flow:
        IDLE ──(steady 10 min)──> STEP_ACTIVE ──(hold ≥60 min + steady)──>
        RESTORE ──(steady 10 min)──> IDLE

    Each cycle uses one direction (±1°C).  Alternates on subsequent cycles.
    """

    def __init__(
        self,
        enabled: bool = False,
        window_start: int | None = None,
        window_end: int | None = None,
        research_mode: bool = False,
    ) -> None:
        self._enabled = enabled
        self._window_start = window_start
        self._window_end = window_end
        self._research_mode = research_mode
        self._stall_cap = RESEARCH_STALL_CAP if research_mode else STALL_CAP

        self._state = PerturbState.IDLE
        self._direction: float = 1.0  # +1 or -1, flips each cycle
        self._steady_since: float | None = None
        self._step_start: float = 0.0
        self._waiting_since: float | None = None

        self._cycles_completed: int = 0
        self._cycles_without_improvement: int = 0
        self._cycles_low_quality: int = 0
        self._confidence_snapshot: float = 0.0
        self._step_start_temp: float | None = None
        self._restore_start_temp: float | None = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def state(self) -> PerturbState:
        return self._state

    @property
    def offset(self) -> float:
        """Current perturbation offset (°C). Non-zero only in STEP_ACTIVE."""
        if self._state == PerturbState.STEP_ACTIVE:
            return self._direction * AMPLITUDE_C
        return 0.0

    @property
    def cycles_completed(self) -> int:
        return self._cycles_completed

    @property
    def cycles_without_improvement(self) -> int:
        return self._cycles_without_improvement

    @property
    def cycles_low_quality(self) -> int:
        return self._cycles_low_quality

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def research_mode(self) -> bool:
        return self._research_mode

    # ── Main tick ────────────────────────────────────────────────────

    def tick(
        self,
        now_mono: float,
        room_temp: float,
        room_temp_rate: float,
        integral_change_output: float,
        ff_settled_ticks: int,
        is_clamped: bool,
        supplemental_active: bool,
        learning_suppressed: bool,
        plant_test_active: bool,
        mode_heating: bool,
        plant_confidence: float,
        current_hour: int,
    ) -> float:
        """Advance the state machine. Returns the current offset (°C)."""
        if not self._enabled:
            return 0.0

        # Pack the steady-state inputs for reuse.
        conditions_met = (
            abs(room_temp_rate) < 0.015
            and integral_change_output < 0.075
            and ff_settled_ticks >= 4
            and not is_clamped
            and not supplemental_active
            and not learning_suppressed
            and not plant_test_active
        )

        # ── IDLE: wait for steady state, then start a cycle ──────
        if self._state == PerturbState.IDLE:
            convergence_ok = (
                self._research_mode or plant_confidence < CONVERGENCE_STOP
            )
            if (
                self._dwell_met(now_mono, conditions_met)
                and convergence_ok
                and self._in_window(current_hour)
            ):
                self._begin_step(now_mono, mode_heating, plant_confidence, room_temp)

        # ── WAITING (perturb_now): like IDLE but with timeout ────
        elif self._state == PerturbState.WAITING:
            if self._waiting_since is None:
                self._waiting_since = now_mono
            elif now_mono - self._waiting_since > FORCE_TIMEOUT_S:
                _LOGGER.info("Auto-perturbation: perturb_now timed out")
                self._reset()
            elif self._dwell_met(now_mono, conditions_met):
                self._begin_step(now_mono, mode_heating, plant_confidence, room_temp)

        # ── STEP_ACTIVE: hold the offset ─────────────────────────
        elif self._state == PerturbState.STEP_ACTIVE:
            if self._check_abort(is_clamped, supplemental_active,
                                 learning_suppressed, plant_test_active):
                pass  # already reset
            else:
                elapsed = now_mono - self._step_start
                if elapsed >= MAX_HOLD_S:
                    _LOGGER.info("Auto-perturbation: max hold (%.0f min)", elapsed / 60)
                    self._begin_restore(room_temp)
                elif elapsed >= MIN_HOLD_S and self._dwell_met(now_mono, conditions_met):
                    _LOGGER.info("Auto-perturbation: settled after %.0f min", elapsed / 60)
                    self._begin_restore(room_temp)
                elif elapsed < MIN_HOLD_S:
                    self._steady_since = None  # don't count dwell before min hold

        # ── RESTORE: offset=0, wait for re-stabilization ─────────
        elif self._state == PerturbState.RESTORE:
            if self._check_abort(is_clamped, supplemental_active,
                                 learning_suppressed, plant_test_active):
                pass
            elif self._dwell_met(now_mono, conditions_met):
                self._finish_cycle(plant_confidence)

        # STALLED: no-op, cleared by force_start()

        return self.offset

    # ── Transitions ──────────────────────────────────────────────────

    def _begin_step(
        self, now_mono: float, mode_heating: bool, plant_confidence: float,
        room_temp: float,
    ) -> None:
        direction = self._direction
        if not mode_heating:
            direction = -direction
        self._direction = direction  # store effective direction for this cycle
        self._state = PerturbState.STEP_ACTIVE
        self._step_start = now_mono
        self._steady_since = None
        self._confidence_snapshot = plant_confidence
        self._step_start_temp = room_temp
        self._restore_start_temp = None
        _LOGGER.info(
            "Auto-perturbation: %+.0f°C offset (cycle %d, %s)",
            direction * AMPLITUDE_C,
            self._cycles_completed + 1,
            "heating" if mode_heating else "cooling",
        )

    def _begin_restore(self, room_temp: float) -> None:
        self._state = PerturbState.RESTORE
        self._steady_since = None
        self._restore_start_temp = room_temp

    def _finish_cycle(self, plant_confidence: float) -> None:
        self._cycles_completed += 1
        # Alternate direction for next cycle
        self._direction = -self._direction

        low_quality = self._cycle_was_low_quality()

        improvement = plant_confidence - self._confidence_snapshot
        if low_quality:
            self._cycles_low_quality += 1
            delta = abs(self._restore_start_temp - self._step_start_temp)  # type: ignore[operator]
            _LOGGER.info(
                "Auto-perturbation: cycle %d done, low-quality (Δ=%.2f°C < %.2f), "
                "skipped stall counter",
                self._cycles_completed, delta, INFORMATIVE_DELTA_C,
            )
        elif improvement < CONVERGENCE_DELTA:
            self._cycles_without_improvement += 1
            _LOGGER.info(
                "Auto-perturbation: cycle %d done, no improvement "
                "(%.2f → %.2f, stall %d/%d)",
                self._cycles_completed, self._confidence_snapshot,
                plant_confidence, self._cycles_without_improvement, self._stall_cap,
            )
        else:
            self._cycles_without_improvement = 0
            _LOGGER.info(
                "Auto-perturbation: cycle %d done, improved (%.2f → %.2f)",
                self._cycles_completed, self._confidence_snapshot,
                plant_confidence,
            )

        if self._cycles_without_improvement >= self._stall_cap:
            self._state = PerturbState.STALLED
            _LOGGER.warning(
                "Auto-perturbation: stalled after %d cycles. "
                "Consider running full plant test.",
                self._cycles_completed,
            )
        else:
            self._reset()

    def _cycle_was_low_quality(self) -> bool:
        """True if the cycle's room-temp Δ is below the informativity threshold.

        Both snapshots must be present; missing snapshots fall back to "high quality"
        so we don't accidentally suppress the stall counter on stale state.
        """
        if self._step_start_temp is None or self._restore_start_temp is None:
            return False
        return abs(self._restore_start_temp - self._step_start_temp) < INFORMATIVE_DELTA_C

    def _reset(self) -> None:
        """Return to IDLE with clean timing state."""
        self._state = PerturbState.IDLE
        self._steady_since = None
        self._waiting_since = None

    # ── Helpers ───────────────────────────────────────────────────────

    def _dwell_met(self, now_mono: float, conditions_met: bool) -> bool:
        """Check if steady-state conditions have held for the required dwell."""
        if conditions_met:
            if self._steady_since is None:
                self._steady_since = now_mono
            return (now_mono - self._steady_since) >= STEADY_STATE_DWELL_S
        self._steady_since = None
        return False

    def _check_abort(
        self,
        is_clamped: bool,
        supplemental_active: bool,
        learning_suppressed: bool,
        plant_test_active: bool,
    ) -> bool:
        """If an abort condition is met, reset and return True."""
        reasons = []
        if is_clamped:
            reasons.append("HP clamped")
        if supplemental_active:
            reasons.append("supplemental active")
        if learning_suppressed:
            reasons.append("learning suppressed")
        if plant_test_active:
            reasons.append("plant test started")
        if reasons:
            self.abort(", ".join(reasons))
            return True
        return False

    def _in_window(self, hour: int) -> bool:
        """True if current hour is within configured time window (or no window)."""
        if self._window_start is None or self._window_end is None:
            return True
        s, e = self._window_start, self._window_end
        if s <= e:
            return s <= hour < e
        return hour >= s or hour < e  # wraps midnight

    # ── External API ─────────────────────────────────────────────────

    def abort(self, reason: str = "") -> None:
        """Abort any active perturbation and return to IDLE."""
        if self._state in (
            PerturbState.STEP_ACTIVE, PerturbState.RESTORE, PerturbState.WAITING,
        ):
            _LOGGER.info("Auto-perturbation aborted: %s", reason or "unknown")
            self._reset()

    def force_start(self) -> None:
        """Request a perturbation cycle (perturb_now service).

        Bypasses convergence gate and time window.  Still requires
        steady-state conditions before the step begins.
        """
        if self._state == PerturbState.STALLED:
            self._cycles_without_improvement = 0
            _LOGGER.info("Auto-perturbation: stall cleared by perturb_now")
        if self._state in (PerturbState.IDLE, PerturbState.STALLED):
            self._state = PerturbState.WAITING
            self._waiting_since = None
            self._steady_since = None
            _LOGGER.info("Auto-perturbation: perturb_now, waiting for steady state")

    # ── HA Repair ────────────────────────────────────────────────────

    def get_stall_issue(
        self, entry_id: str, mode: str,
    ) -> tuple[str, str, str, dict[str, str], bool, bool, dict[str, Any] | None] | None:
        """Return 7-tuple HA Repair issue if stalled, else None."""
        if self._state != PerturbState.STALLED:
            return None
        return (
            f"auto_perturb_stall_{entry_id}_{mode}",
            "warning",
            "auto_perturb_stall",
            {"cycles": str(self._cycles_completed), "mode": mode},
            True,
            True,
            {"repair_type": "auto_perturb_stall", "entry_id": entry_id, "mode": mode},
        )

    # ── Persistence ──────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Serialize counters (state machine resets to IDLE on restart)."""
        return {
            "cycles_completed": self._cycles_completed,
            "cycles_without_improvement": self._cycles_without_improvement,
            "cycles_low_quality": self._cycles_low_quality,
            "direction": self._direction,
            "confidence_snapshot": self._confidence_snapshot,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore counters from persisted data."""
        self._cycles_completed = int(data.get("cycles_completed", 0))
        self._cycles_without_improvement = int(data.get("cycles_without_improvement", 0))
        self._cycles_low_quality = int(data.get("cycles_low_quality", 0))
        self._direction = float(data.get("direction", 1.0))
        self._confidence_snapshot = float(data.get("confidence_snapshot", 0.0))
        if self._cycles_without_improvement >= self._stall_cap:
            self._state = PerturbState.STALLED
