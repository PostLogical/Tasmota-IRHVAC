"""Reference governor + chatter monitor for quantization defense.

Replaces q-feedback's integrator-corrupting quantization defense with a
reference-side supervisor:

  r_user → ReferenceGovernor → v(t) → inner PI+FF (unchanged) → HP
                ↑
         ChatterMonitor (counts HP-change events in sliding window)

Primary citations (verified via pdftotext in local/lit_pdfs/):
- Kolmanovsky/Garone/Di Cairano, MERL TR2014-119 (2014): SRG framework;
  closest-feasible projection is the canonical default direction.
- Schwerdtner et al., MERL TR2019-054 (2019): heat-pump-specific
  projection-based AW; nudge direction is a user-designed policy.
- Hespanha tutorial (CDC 2001) §monitoring + Hespanha-Morse 1999
  avedwell Thm 2: monitor window ≥ closed-loop τ; arbitrary chatter
  bound N0 > 0 admissible under average dwell-time τ_D.
- Bicchi/Marigo/Piccoli, IEEE TAC 47(4) 2002: under quantized inputs
  only ultimate-boundedness is achievable; |Δr| < ½ quantization step
  guarantees the nudge cannot lock the system to the wrong integer bin.
- Basseville-Nikiforov 1993 §5.2: CUSUM/ARL framework — threshold
  count sized for target MTBFA from baseline change rate.
- Forssell-Ljung Automatica 35:1215 (1999): external reference
  perturbations are benign or helpful for closed-loop RLS — directly
  opposite of integrator-state corruption.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field


# ── Defaults sized to our system (1-3 min ticks, τ ≈ 50-150 min) ─────────
DEFAULT_WINDOW_SECONDS: float = 60.0 * 60.0   # 60 min ≈ τ_fast
DEFAULT_THRESHOLD: int = 4                    # heuristic; sizable from production traces
DEFAULT_COOLDOWN_SECONDS: float = 5.0 * 60.0  # 5 min — bound spurious re-alarms

DEFAULT_NUDGE_MAX_C: float = 0.45             # < ½ × 1°C quantization (Bicchi)
DEFAULT_DRIFT_BAND_C: float = 0.5             # |room - r_user| > this means drifted out
DEFAULT_DRIFT_DURATION_SEC: float = 15.0 * 60.0  # drift must persist this long to release


@dataclass
class ChatterMonitor:
    """Sliding-window detector for raw_setpoint chatter pressure.

    Counts events where `round(raw_setpoint)` changes between consecutive
    ticks — i.e., the PI's continuous output crossed a quantization
    half-integer. This is the "would-have-flipped pre-hysteresis" rate,
    a LEADING indicator of chatter. Counting actual hp_setpoint changes
    (post-hysteresis) would be a lagging indicator since the existing
    midpoint hysteresis already absorbs many crossings.

    Also tracks the per-tick raw_setpoint history in the same window so
    the ReferenceGovernor can compute a duty-cycle lean (which side of
    the half-integer raw spent more time on) at the moment of NUDGE
    engage — a model-free signal of which integer HP the system
    naturally prefers.
    """

    window_seconds: float = DEFAULT_WINDOW_SECONDS
    threshold: int = DEFAULT_THRESHOLD
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS

    _events: collections.deque[float] = field(default_factory=collections.deque)
    _samples: collections.deque[tuple[float, float]] = field(default_factory=collections.deque)
    _last_rounded_raw: int | None = None
    _cooldown_until: float | None = None

    def update(self, raw_setpoint: float, now_mono: float) -> bool:
        """Tick the monitor; return True iff chatter pressure exceeds threshold.

        Args:
            raw_setpoint: The PI's continuous (pre-hysteresis, pre-rounding)
                setpoint output. We count when its nearest-integer value
                changes — that's a half-integer boundary crossing.
            now_mono: Monotonic clock in seconds.
        """
        # Expire stale events + samples
        cutoff = now_mono - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        # Record current sample for duty-cycle calculation
        self._samples.append((now_mono, raw_setpoint))

        # Record "raw crossed half-integer" events
        rounded_now = round(raw_setpoint)
        if self._last_rounded_raw is not None and rounded_now != self._last_rounded_raw:
            self._events.append(now_mono)
        self._last_rounded_raw = rounded_now

        # Cooldown suppression
        if self._cooldown_until is not None and now_mono < self._cooldown_until:
            return False

        # Alarm
        if len(self._events) >= self.threshold:
            self._cooldown_until = now_mono + self.cooldown_seconds
            return True
        return False

    @property
    def event_count(self) -> int:
        return len(self._events)

    def duty_cycle_lean(self) -> float:
        """Signed lean from raw_setpoint duty cycle over the recent window.

        Returns a value in [-1.0, +1.0]:
          +1.0 = raw was ABOVE its rounded value 100% of the window
                 (system strongly wants the higher integer HP)
           0.0 = balanced (raw spent equal time on each side)
          -1.0 = raw was BELOW its rounded value 100% (wants lower HP)

        Used by ReferenceGovernor to size the nudge proportionally to the
        observed lean rather than guessing a fixed magnitude.
        """
        if not self._samples:
            return 0.0
        above = sum(1 for (_, raw) in self._samples if (raw - round(raw)) > 0)
        return (above / len(self._samples)) * 2.0 - 1.0


@dataclass
class ReferenceGovernor:
    """Reference-side quantization defense.

    Engages NUDGE on chatter alarm, locks at the closest-feasible HP bin,
    and STAYS engaged until conditions change enough that the locked HP
    no longer matches comfort needs (room drifts persistently outside a
    band around r_user). Bumpless transfer at mode transitions
    (back-calc on integral) keeps PI output continuous across the change.
    """

    max_nudge_c: float = DEFAULT_NUDGE_MAX_C
    drift_band_c: float = DEFAULT_DRIFT_BAND_C
    drift_duration_sec: float = DEFAULT_DRIFT_DURATION_SEC

    mode: str = "NORMAL"           # "NORMAL" | "NUDGE"
    nudge_c: float = 0.0
    _drift_started_mono: float | None = None

    def step(
        self, r_user_c: float, room_c: float,
        chatter_alarm: bool, duty_lean: float,
        ki: float, now_mono: float,
    ) -> tuple[float, float]:
        """Advance the governor.

        Args:
            r_user_c: User-set desired temperature (°C).
            room_c: Current measured room temperature (°C).
            chatter_alarm: True iff ChatterMonitor just fired this tick.
            duty_lean: Signed lean from ChatterMonitor.duty_cycle_lean()
                — observed fraction of time raw_setpoint was above its
                rounded value, scaled to [-1, +1]. Used to pick nudge
                magnitude proportional to system's natural preference.
            ki: Current PI integral gain (for bumpless transfer).
            now_mono: Monotonic clock in seconds.

        Returns:
            (effective_desired_c, integral_delta):
              effective_desired_c = r_user + nudge (the v(t) for PI math)
              integral_delta      = back-calc adjustment for PI integral
                                    (non-zero only at mode transitions).
                                    Caller adds this to its integral state.
        """
        integral_delta = 0.0

        if self.mode == "NORMAL":
            if chatter_alarm and ki > 0:
                # Engage NUDGE.  Duty-cycle-based projection: size the
                # nudge proportionally to how strongly raw_setpoint
                # leaned toward one integer over the recent window.
                # `duty_lean` in [-1, +1] from ChatterMonitor:
                #   +1.0 → fully above ½-integer → lock at higher HP
                #   −1.0 → fully below             → lock at lower HP
                #    0   → balanced (no preference) — picks nudge=0
                #         which means effective_desired = r_user (no
                #         offset; supervisor stays NORMAL effectively).
                # Model-free; robust to FF inaccuracy. (Future: replace
                # with FF-derived equilibrium nudge once FF confidence
                # exceeds a maturity threshold — see project memory.)
                new_nudge = duty_lean * self.max_nudge_c
                new_nudge = max(-self.max_nudge_c, min(self.max_nudge_c, new_nudge))
                if new_nudge == 0.0:
                    # Balanced lean — supervisor has nothing useful to do
                    # at this engage trigger. Stay in NORMAL; will re-try
                    # on next chatter alarm with a new lean reading.
                    return r_user_c, 0.0
                # Bumpless transfer: reference shifts by Δr = new_nudge;
                # adjust integral by -Δr/Ki so PI output stays continuous.
                integral_delta = -new_nudge / ki
                self.nudge_c = new_nudge
                self.mode = "NUDGE"
                self._drift_started_mono = None

        else:  # NUDGE
            # Track sustained drift OUT of the comfort band.
            in_band = abs(room_c - r_user_c) <= self.drift_band_c
            if in_band:
                self._drift_started_mono = None
            else:
                if self._drift_started_mono is None:
                    self._drift_started_mono = now_mono
                elif now_mono - self._drift_started_mono >= self.drift_duration_sec and ki > 0:
                    # Drift sustained → conditions have changed → release.
                    # Bumpless transfer: reference shifts by Δr = -nudge_c;
                    # adjust integral by -Δr/Ki = +nudge_c/Ki.
                    integral_delta = self.nudge_c / ki
                    self.nudge_c = 0.0
                    self.mode = "NORMAL"
                    self._drift_started_mono = None

        return r_user_c + self.nudge_c, integral_delta

    @property
    def is_engaged(self) -> bool:
        return self.mode == "NUDGE"
