"""Reference governor + chatter monitor for quantization defense.

Replaces q-feedback's integrator-corrupting quantization defense with a
reference-side supervisor:

  r_user → ReferenceGovernor → v(t) → inner PI+FF (unchanged) → HP
                ↑
         ChatterMonitor (counts HP-change events in sliding window)

Magnitude rule: HYBRID. On chatter alarm, engage at full max_nudge_c
(direction by duty_lean sign, fallback by room sign). While engaged,
adaptive relax slowly tracks nudge_c toward (room − r_user), letting
the supervisor converge to the system's observed equilibrium offset
without depending on FF accuracy.

Bumpless transfer is configurable per event type (engage, relax,
release). See class docstring for the design tradeoff (controller-
output continuity vs integrator-as-thermal-signal cleanness).

Primary citations (verified via pdftotext in local/lit_pdfs/):
- Kolmanovsky/Garone/Di Cairano, MERL TR2014-119 (2014): SRG framework;
  reference shift is the canonical defense, integrator untouched.
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
- Hanus 1987 tracking conditioning: bumpless transfer preserves
  output continuity at reference shifts; correct semantics for user
  setpoint changes, debatable for supervisor-mediated detours.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field


# Multi-tier defaults. The supervisor needs to defend against chatter at
# multiple cadences — fast (sub-15-min, from quantization noise during
# quasi-steady operation) and slow (hourly, from operating-point drift
# through half-integers as outdoor temperature varies through the day).
#
# Tier sizing comes from Hespanha-Morse avg dwell-time framework: a tier
# fires when N events accumulate within W seconds, equivalent to average
# dwell time τ_D = W/N falling below the tier's tolerance.
#
# Fast: τ_D < 15 min — quasi-steady chatter where raw_setpoint wanders
# across a half-integer rapidly due to noise.
DEFAULT_FAST_WINDOW_SECONDS: float = 60.0 * 60.0    # 60 min
DEFAULT_FAST_THRESHOLD: int = 4
DEFAULT_FAST_COOLDOWN_SECONDS: float = 5.0 * 60.0   # 5 min

# Slow: τ_D < 30 min — load-driven chatter where outdoor variation slowly
# moves the equilibrium HP through one or more half-integers each hour.
# This is what real-weather winter exhibits (see supervisor_ab.py
# winter_typical scenario: HP flips 28↔29 once per hour for hours).
DEFAULT_SLOW_WINDOW_SECONDS: float = 4.0 * 60.0 * 60.0  # 4 hours
DEFAULT_SLOW_THRESHOLD: int = 8
DEFAULT_SLOW_COOLDOWN_SECONDS: float = 15.0 * 60.0     # 15 min

DEFAULT_NUDGE_MAX_C: float = 0.45             # < ½ × 1°C quantization (Bicchi)
DEFAULT_ADAPT_RATE: float = 0.02              # adaptive-relax rate per tick
DEFAULT_DRIFT_BAND_C: float = 0.5
DEFAULT_DRIFT_DURATION_SEC: float = 15.0 * 60.0


@dataclass
class QRefBiaser:
    """Reference-side persistent bias for chatter defense.

    Per tick: compute q_error = round(raw_setpoint) − raw_setpoint, EMA-filter
    it, apply the smoothed signal as a reference shift. Persistence comes from
    the EMA — same role that integration plays in q_feedback — providing a
    smooth always-on bias that keeps raw_setpoint deeper inside the rounded
    HP integer's interior.

    Compare to q_feedback: same effective mechanism (persistent bias derived
    from quantization residue), but applied through `effective_desired` instead
    of through the integrator. The integrator stays a clean thermal record.

    Gated externally on `hp_active AND in_deadband` so the biaser only acts
    when chatter actually matters (in-deadband, HP operating). Out-of-deadband
    or HP-saturated/off, bias decays toward zero.

    Not lit-canonical — this is the next-design candidate from supervisor
    investigation (2026-05-18). Hypothesis: matches q_feedback effectiveness
    on real-weather steady-state chatter without the integrator-pollution
    failure modes.
    """

    gain: float = 0.4
    ema_alpha: float = 0.1
    # Symmetric default; per-direction overrides take precedence if non-None.
    # Asymmetric caps reflect comfort asymmetry: in heating, room warmer than
    # r_user (positive bias) is more tolerable than cooler (negative bias);
    # in cooling, the inverse. Caller sets appropriate overrides per mode.
    max_bias_c: float = 0.45
    max_bias_up_c: float | None = None
    max_bias_down_c: float | None = None
    decay_alpha: float = 0.1

    _bias: float = 0.0

    def update(self, raw_setpoint: float, active_and_in_db: bool) -> float:
        """Tick. Returns current bias in °C.

        Args:
            raw_setpoint: current PI raw output (pre-rounding, pre-hysteresis).
            active_and_in_db: True iff HP is actively contributing and the
                              controller is in-deadband (|error| < deadband).
                              Caller decides this from controller state.
        """
        if not active_and_in_db:
            self._bias *= (1.0 - self.decay_alpha)
            return self._bias

        q_error = round(raw_setpoint) - raw_setpoint
        target_bias = q_error * self.gain
        up = self.max_bias_up_c if self.max_bias_up_c is not None else self.max_bias_c
        down = self.max_bias_down_c if self.max_bias_down_c is not None else self.max_bias_c
        target_bias = max(-down, min(up, target_bias))
        self._bias = (1.0 - self.ema_alpha) * self._bias + self.ema_alpha * target_bias
        return self._bias

    @property
    def bias(self) -> float:
        return self._bias


@dataclass
class ChatterTier:
    """One (window, threshold, cooldown) detection tier inside ChatterMonitor.

    A tier fires when its events deque accumulates `threshold` events within
    `window_seconds`. After firing, it suppresses further alarms for
    `cooldown_seconds`. Multiple tiers compose via OR — any tier firing
    raises the global chatter alarm.
    """

    window_seconds: float
    threshold: int
    cooldown_seconds: float
    _events: collections.deque[float] = field(default_factory=collections.deque)
    _cooldown_until: float | None = None

    def tick(self, now_mono: float, new_event: bool) -> bool:
        """Expire stale events, append new one if present, check threshold.

        Returns True iff this tier fires on this tick.
        """
        cutoff = now_mono - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        if new_event:
            self._events.append(now_mono)
        if self._cooldown_until is not None and now_mono < self._cooldown_until:
            return False
        if len(self._events) >= self.threshold:
            self._cooldown_until = now_mono + self.cooldown_seconds
            return True
        return False

    @property
    def event_count(self) -> int:
        return len(self._events)


def _default_tiers() -> list[ChatterTier]:
    return [
        ChatterTier(
            window_seconds=DEFAULT_FAST_WINDOW_SECONDS,
            threshold=DEFAULT_FAST_THRESHOLD,
            cooldown_seconds=DEFAULT_FAST_COOLDOWN_SECONDS,
        ),
        ChatterTier(
            window_seconds=DEFAULT_SLOW_WINDOW_SECONDS,
            threshold=DEFAULT_SLOW_THRESHOLD,
            cooldown_seconds=DEFAULT_SLOW_COOLDOWN_SECONDS,
        ),
    ]


@dataclass
class ChatterMonitor:
    """Multi-tier sliding-window detector for raw_setpoint chatter pressure.

    Counts `round(raw_setpoint)` transitions — the pre-hysteresis leading
    indicator of HP chatter — and checks each tier independently. Fires
    when ANY tier's threshold is met.

    Default tiers: a fast tier (60-min window, 4-event threshold) catching
    quasi-steady quantization chatter, and a slow tier (4-hour window,
    8-event threshold) catching hour-frequency load-driven chatter. The
    slow tier exists because real-weather operating-point drift produces
    chatter that's spread out over hours and never hits the fast tier's
    rate threshold — see Göbel et al. RWTH 2023 on regime-dependent
    hysteresis and Hespanha-Morse 1999 §Thm 2 on multi-scale dwell time.

    Shares one raw_setpoint sample buffer across tiers (sized to the widest
    tier window) for the duty-cycle direction signal at engage.
    """

    tiers: list[ChatterTier] = field(default_factory=_default_tiers)
    _samples: collections.deque[tuple[float, float]] = field(default_factory=collections.deque)
    _last_rounded_raw: int | None = None
    _last_fired_tier: int = -1

    @property
    def _max_window(self) -> float:
        return max((t.window_seconds for t in self.tiers), default=0)

    def update(self, raw_setpoint: float, now_mono: float) -> bool:
        # Sample buffer (sized to widest tier window) — used for duty lean
        cutoff = now_mono - self._max_window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        self._samples.append((now_mono, raw_setpoint))

        # Boundary-crossing event detection (shared across tiers)
        rounded_now = round(raw_setpoint)
        new_event = (
            self._last_rounded_raw is not None
            and rounded_now != self._last_rounded_raw
        )
        self._last_rounded_raw = rounded_now

        # Tick each tier; ANY firing raises the alarm. Record which tier fired
        # for diagnostics / supervisor engage-policy decisions.
        fired = False
        for i, tier in enumerate(self.tiers):
            if tier.tick(now_mono, new_event):
                if not fired:
                    self._last_fired_tier = i
                fired = True
        return fired

    @property
    def event_count(self) -> int:
        """Total events tracked by the fast (tier 0) detector. Preserves the
        previous single-tier semantics for callers that log this for
        diagnostics (e.g. bench full_stack_runner)."""
        return self.tiers[0].event_count if self.tiers else 0

    @property
    def tier_event_counts(self) -> list[int]:
        """Per-tier event counts, for richer diagnostics."""
        return [t.event_count for t in self.tiers]

    @property
    def last_fired_tier(self) -> int:
        """Index of the tier that most recently fired (-1 if none yet)."""
        return self._last_fired_tier

    def duty_cycle_lean(self) -> float:
        if not self._samples:
            return 0.0
        above = sum(1 for (_, raw) in self._samples if (raw - round(raw)) > 0)
        return (above / len(self._samples)) * 2.0 - 1.0


@dataclass
class ReferenceGovernor:
    """Reference-side quantization defense.

    On chatter alarm, engages at full max_nudge_c (sign chosen so the
    locked HP integer is the side raw_setpoint had been spending LESS
    duty on — committing to the under-represented side breaks the
    duty-cycle symmetry that produces chatter). While engaged, adaptive
    relax tracks nudge_c toward observed `room − r_user`, so the
    supervisor's offset converges to the actual equilibrium. Sustained
    drift outside the comfort band triggers release.

    Bumpless transfer flags (default all True) preserve PI output
    continuity at the corresponding event by back-calc adjustment to
    the integrator. Setting any flag to False makes that event a "kick"
    — the proportional term delivers the reference shift to raw_setpoint
    immediately, integrator stays a clean thermal record. Empirically
    (see local/tools/supervisor_ab*.py) kick semantics dominate on
    chatter and pollution-of-integrator metrics; bumpless wins only on
    actuator-smoothness metrics by a small margin.
    """

    max_nudge_c: float = DEFAULT_NUDGE_MAX_C
    adapt_rate: float = DEFAULT_ADAPT_RATE
    drift_band_c: float = DEFAULT_DRIFT_BAND_C
    drift_duration_sec: float = DEFAULT_DRIFT_DURATION_SEC

    engage_bumpless: bool = True
    relax_bumpless: bool = True
    release_bumpless: bool = True

    mode: str = "NORMAL"
    nudge_c: float = 0.0
    _drift_started_mono: float | None = None

    def step(
        self, r_user_c: float, room_c: float,
        chatter_alarm: bool, duty_lean: float,
        ki: float, now_mono: float,
    ) -> tuple[float, float]:
        integral_delta = 0.0

        if self.mode == "NORMAL":
            if chatter_alarm and ki > 0:
                # Hybrid engage: strong-initial commit at max_nudge_c.
                # Direction picks the LESS-represented side of duty_lean
                # (commit to break the chatter symmetry). Fallback to
                # room sign when duty_lean is exactly balanced.
                if abs(duty_lean) < 1e-9:
                    new_nudge = self.max_nudge_c if room_c > r_user_c else -self.max_nudge_c
                else:
                    new_nudge = self.max_nudge_c if duty_lean < 0 else -self.max_nudge_c
                if self.engage_bumpless:
                    integral_delta = -new_nudge / ki
                self.nudge_c = new_nudge
                self.mode = "NUDGE"
                self._drift_started_mono = None

        else:  # NUDGE
            # Adaptive relax: slowly track nudge_c toward observed
            # (room − r_user), clamped to [−max, +max]. Model-free —
            # no FF dependency.
            target = max(-self.max_nudge_c, min(self.max_nudge_c, room_c - r_user_c))
            adj = (target - self.nudge_c) * self.adapt_rate
            if abs(adj) > 1e-6:
                if self.relax_bumpless:
                    integral_delta = -adj / ki
                self.nudge_c += adj

            # Drift detection → release. The locked HP is no longer
            # serving comfort needs; hand control back to inner PI.
            in_band = abs(room_c - r_user_c) <= self.drift_band_c
            if in_band:
                self._drift_started_mono = None
            else:
                if self._drift_started_mono is None:
                    self._drift_started_mono = now_mono
                elif now_mono - self._drift_started_mono >= self.drift_duration_sec and ki > 0:
                    if self.release_bumpless:
                        integral_delta += self.nudge_c / ki
                    self.nudge_c = 0.0
                    self.mode = "NORMAL"
                    self._drift_started_mono = None

        return r_user_c + self.nudge_c, integral_delta

    @property
    def is_engaged(self) -> bool:
        return self.mode == "NUDGE"
