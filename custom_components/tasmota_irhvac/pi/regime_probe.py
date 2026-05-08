"""Active probing for HP contribution regime boundary detection.

Pure computation — no Home Assistant dependencies.  When the HP setpoint
is near room temperature, the HP's actual contribution is uncertain due
to sensor calibration mismatch between our room sensor and the HP's
internal sensor.  This module actively probes the boundary by briefly
forcing the HP to minimum setpoint and observing whether the room rate
changes, directly resolving the ambiguity.

The probe narrows an asymmetric uncertainty band around hp_setpoint ==
current_c.  The "above" side (setpoint > current but HP might have cut
out) and "below" side (setpoint < current but HP might still cycle)
shrink independently as probes confirm or deny HP contribution.

Literature: PWARX regime-switching models (building thermal), Cragg
(1971) hurdle models, dead-zone nonlinearity identification.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

BASELINE_MIN_READINGS: int = 2
"""Minimum sensor readings for baseline rate estimate."""

BASELINE_MIN_DURATION_S: float = 180.0
"""AND at least 3 min elapsed (lower bound; in practice 2 readings
in steady state take 5–10 min)."""

PROBE_MIN_READINGS: int = 2
"""Minimum sensor readings during HP-off probe."""

PROBE_MIN_DURATION_S: float = 480.0
"""AND at least 8 min elapsed.  The room rate responds on the fast
thermal time constant (~5–15 min air volume + surface convection).
At 8 min with τ_fast=10 min: 55 % of the rate change is visible.
Detectable at ≥1 °C offsets given k_c ≈ 0.02 °C/min per °C."""

RATE_CHANGE_THRESHOLD: float = 0.005
"""°C/min — minimum directional rate change to detect HP contribution.
Matches the EMA-filtered room_rate noise floor."""

STABILITY_THRESHOLD: float = 0.03
"""°C/min — moderate threshold for probe entry (not the strict
steady-state gates used for learning)."""

MIN_DELTA_BELOW_CURRENT: float = 3.0
"""Min setpoint must be ≥ 3 °C below current_c to guarantee the
compressor is off regardless of sensor offset (2 °C max offset + 1 °C
margin)."""

INITIAL_COOLDOWN_S: float = 1800.0
"""30 min — quick re-probe for confirmation."""

CONFIRMED_COOLDOWN_S: float = 14400.0
"""4 hours — after first confirmations, space out to let conditions
change so probes occur at different deltas."""

CONVERGED_COOLDOWN_S: float = 86400.0
"""24 hours — once margin has stabilised, probe rarely."""

SHRINK_CONFIRMATIONS: int = 2
"""Probes needed at similar deltas before shrinking margin."""

SHRINK_DELTA_TOLERANCE: float = 0.5
"""°C — probes within this range of each other count as confirming."""

SHRINK_FACTOR: float = 0.8
"""Shrink margin to this fraction of confirmed delta (adds safety)."""

# How many successful confirmations before moving to longer cooldown
CONFIRMATION_THRESHOLD: int = 3


class ProbeState(enum.Enum):
    """State machine states."""
    IDLE = "idle"
    BASELINE = "baseline"
    PROBE = "probe"
    ANALYZE = "analyze"
    COOLDOWN = "cooldown"


@dataclass
class RegimeProbeResult:
    """Result returned each tick."""
    probe_active: bool = False
    force_min_setpoint: bool = False


_INACTIVE = RegimeProbeResult()


class RegimeProbe:
    """Active probing state machine for HP contribution regime boundary.

    State flow::

        IDLE ──(uncertain + stable)──> BASELINE ──(readings + time)──>
        PROBE ──(readings + time)──> ANALYZE ──(update margin)──>
        COOLDOWN ──(cooldown elapsed)──> IDLE
    """

    def __init__(
        self,
        *,
        window_start: int | None = None,
        window_end: int | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._window_start = window_start
        self._window_end = window_end

        self._state = ProbeState.IDLE

        # Timing
        self._phase_start_mono: float = 0.0
        self._cooldown_end_mono: float = 0.0

        # Readings collected
        self._baseline_rates: list[float] = []
        self._probe_rates: list[float] = []

        # Context at probe start
        self._probe_hp_setpoint: int = 0
        self._probe_current_c: float = 0.0
        self._probe_is_heating: bool = True

        # Forced probe: bypass uncertainty check when boundary
        # estimator stalls and probe has never fired (IDLE state).
        self._forced_probe: bool = False

        # Evidence accumulation (persisted)
        self._contribution_evidence_above: list[float] = []
        self._contribution_evidence_below: list[float] = []
        self._no_contribution_count: int = 0
        self._confirmations_total: int = 0
        self._probes_completed: int = 0

        # Last probe result (for boundary estimator consumption)
        self._last_probe_delta: float | None = None
        self._last_probe_hp_contributing: bool | None = None

    # ── Properties ───────────────────────────────────────────────────

    @property
    def state(self) -> ProbeState:
        return self._state

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def probes_completed(self) -> int:
        return self._probes_completed

    @property
    def last_probe_delta(self) -> float | None:
        """Delta (current_c - hp_setpoint) at last completed probe."""
        return self._last_probe_delta

    @property
    def last_probe_hp_contributing(self) -> bool | None:
        """Whether HP was contributing at last probe. None if no probe yet."""
        return self._last_probe_hp_contributing

    def consume_last_probe(self) -> tuple[float, bool] | None:
        """Return and clear the last probe result for estimator consumption.

        Returns (delta, hp_was_contributing) or None if no new result.
        """
        if self._last_probe_delta is None:
            return None
        result = (self._last_probe_delta, self._last_probe_hp_contributing or False)
        self._last_probe_delta = None
        self._last_probe_hp_contributing = None
        return result

    # ── Main tick ────────────────────────────────────────────────────

    def tick(
        self,
        now_mono: float,
        room_temp_rate: float,
        hp_setpoint: int,
        current_c: float,
        min_temp_c: float,
        cal_min: float,
        cal_max: float,
        is_heating: bool,
        is_clamped: bool,
        learning_suppressed: bool,
        current_hour: int,
        auto_perturb_active: bool,
    ) -> RegimeProbeResult:
        """Advance the state machine.  Returns probe status."""
        if not self._enabled:
            return _INACTIVE

        # ── Guards (apply in all states except COOLDOWN) ─────────
        if self._state != ProbeState.COOLDOWN:
            if not self._in_window(current_hour):
                if self._state not in (ProbeState.IDLE, ProbeState.COOLDOWN):
                    self._abort("outside time window")
                return _INACTIVE
            if auto_perturb_active:
                if self._state not in (ProbeState.IDLE, ProbeState.COOLDOWN):
                    self._abort("auto-perturbation active")
                return _INACTIVE
            if is_clamped or learning_suppressed:
                if self._state not in (ProbeState.IDLE, ProbeState.COOLDOWN):
                    self._abort("clamped or learning suppressed")
                return _INACTIVE

        # ── IDLE: wait for uncertain zone + stability ────────────
        if self._state == ProbeState.IDLE:
            hp_uncertain = self._is_uncertain(
                hp_setpoint, current_c, cal_min, cal_max, is_heating,
            )
            can_probe = (
                current_c - min_temp_c >= MIN_DELTA_BELOW_CURRENT
                and not auto_perturb_active
                and not is_clamped
                and not learning_suppressed
            )
            trigger = hp_uncertain or self._forced_probe
            if trigger and can_probe and abs(room_temp_rate) < STABILITY_THRESHOLD:
                self._begin_baseline(now_mono, hp_setpoint, current_c, is_heating)

        # ── BASELINE: collect room_rate readings ─────────────────
        elif self._state == ProbeState.BASELINE:
            self._baseline_rates.append(room_temp_rate)
            elapsed = now_mono - self._phase_start_mono
            if (
                len(self._baseline_rates) >= BASELINE_MIN_READINGS
                and elapsed >= BASELINE_MIN_DURATION_S
            ):
                self._begin_probe(now_mono)
                return RegimeProbeResult(probe_active=True, force_min_setpoint=True)
            # Not ready yet — still collecting baseline
            return _INACTIVE

        # ── PROBE: HP at minimum, collect room_rate ──────────────
        elif self._state == ProbeState.PROBE:
            self._probe_rates.append(room_temp_rate)
            elapsed = now_mono - self._phase_start_mono
            if (
                len(self._probe_rates) >= PROBE_MIN_READINGS
                and elapsed >= PROBE_MIN_DURATION_S
            ):
                self._analyze(now_mono)
                return _INACTIVE
            return RegimeProbeResult(probe_active=True, force_min_setpoint=True)

        # ── ANALYZE: handled synchronously in _analyze() ─────────

        # ── COOLDOWN: wait for cooldown to elapse ────────────────
        elif self._state == ProbeState.COOLDOWN:  # pragma: no branch — COOLDOWN tick before timer expiry — exits without state change
            if now_mono >= self._cooldown_end_mono:
                self._state = ProbeState.IDLE

        return _INACTIVE

    # ── Uncertainty check ────────────────────────────────────────────

    @staticmethod
    def _is_uncertain(
        hp_setpoint: int,
        current_c: float,
        cal_min: float,
        cal_max: float,
        is_heating: bool,
    ) -> bool:
        """Check if HP contribution is uncertain.

        current_to_setpoint_delta = current_c - setpoint.  Uncertain when
        within [cal_min, cal_max] (the head unit's calibration bounds).
        """
        current_to_setpoint_delta = current_c - hp_setpoint
        return cal_min <= current_to_setpoint_delta <= cal_max

    @staticmethod
    def is_contribution_uncertain(
        hp_setpoint: int,
        current_c: float,
        cal_min: float,
        cal_max: float,
        is_heating: bool,
    ) -> bool:
        """Public static helper for use by learning gates."""
        return RegimeProbe._is_uncertain(
            hp_setpoint, current_c, cal_min, cal_max, is_heating,
        )

    # ── Transitions ──────────────────────────────────────────────────

    def _begin_baseline(
        self,
        now_mono: float,
        hp_setpoint: int,
        current_c: float,
        is_heating: bool,
    ) -> None:
        self._state = ProbeState.BASELINE
        self._phase_start_mono = now_mono
        self._baseline_rates = []
        self._forced_probe = False  # consumed
        self._probe_hp_setpoint = hp_setpoint
        self._probe_current_c = current_c
        self._probe_is_heating = is_heating
        _LOGGER.info(
            "Regime probe: baseline started "
            "(setpoint=%d°C, room=%.1f°C, delta=%.1f°C, %s)",
            hp_setpoint, current_c,
            current_c - hp_setpoint,
            "heating" if is_heating else "cooling",
        )

    def _begin_probe(self, now_mono: float) -> None:
        self._state = ProbeState.PROBE
        self._phase_start_mono = now_mono
        self._probe_rates = []
        _LOGGER.info(
            "Regime probe: testing HP contribution at delta=%.1f°C "
            "(setpoint %d°C → min, room=%.1f°C)",
            self._probe_current_c - self._probe_hp_setpoint,
            self._probe_hp_setpoint,
            self._probe_current_c,
        )

    def _analyze(self, now_mono: float) -> None:
        """Compare baseline and probe rates. Update state."""
        baseline_avg = sum(self._baseline_rates) / len(self._baseline_rates)
        probe_avg = sum(self._probe_rates) / len(self._probe_rates)
        rate_change = probe_avg - baseline_avg  # signed
        current_to_setpoint_delta = self._probe_current_c - self._probe_hp_setpoint

        # Directional check: in heating, removing HP → rate should decrease.
        if self._probe_is_heating:
            hp_was_contributing = rate_change < -RATE_CHANGE_THRESHOLD
        else:
            hp_was_contributing = rate_change > RATE_CHANGE_THRESHOLD

        self._probes_completed += 1
        self._last_probe_delta = current_to_setpoint_delta
        self._last_probe_hp_contributing = hp_was_contributing

        if not hp_was_contributing:
            # HP was NOT contributing → transition is below this point
            # → evidence to shrink cal_max.
            self._contribution_evidence_above.append(current_to_setpoint_delta)
            self._no_contribution_count += 1
            _LOGGER.info(
                "Regime probe: complete — HP was NOT contributing "
                "(current_to_sp=%.1f°C, rate %.4f→%.4f, change=%.4f)",
                current_to_setpoint_delta, baseline_avg, probe_avg, rate_change,
            )
        else:
            # HP WAS contributing → transition is above this point
            # → evidence to shrink cal_min.
            self._contribution_evidence_below.append(current_to_setpoint_delta)
            _LOGGER.info(
                "Regime probe: complete — HP WAS contributing "
                "(current_to_sp=%.1f°C, rate %.4f→%.4f, change=%.4f)",
                current_to_setpoint_delta, baseline_avg, probe_avg, rate_change,
            )

        self._begin_cooldown(now_mono)

    def _begin_cooldown(self, now_mono: float) -> None:
        if self._confirmations_total >= CONFIRMATION_THRESHOLD:
            cooldown = CONVERGED_COOLDOWN_S
        elif self._probes_completed >= SHRINK_CONFIRMATIONS:
            cooldown = CONFIRMED_COOLDOWN_S
        else:
            cooldown = INITIAL_COOLDOWN_S
        self._cooldown_end_mono = now_mono + cooldown
        self._state = ProbeState.COOLDOWN

    def _abort(self, reason: str) -> None:
        if self._state in (ProbeState.BASELINE, ProbeState.PROBE):
            _LOGGER.info("Regime probe aborted: %s", reason)
        self._state = ProbeState.IDLE
        self._baseline_rates = []
        self._probe_rates = []

    # ── Margin update ────────────────────────────────────────────────

    def compute_calibration_updates(
        self,
        current_cal_min: float,
        current_cal_max: float,
    ) -> tuple[float, float]:
        """Check evidence and return updated (cal_min, cal_max).

        Call after each probe completes.  Can shrink the band (narrow
        toward the transition) OR shift it (when all evidence points
        in one direction, the transition is outside the current band).

        Evidence types (stored as current_to_setpoint_delta at probe time):
        - evidence_above: "HP was NOT contributing" at this delta →
          transition is below this delta → shrink cal_max down.
        - evidence_below: "HP WAS contributing" at this delta →
          transition is above this delta → shrink cal_min up.

        Band shift: if we have SHRINK_CONFIRMATIONS "not contributing"
        probes and ZERO "contributing" probes, the transition is below
        the entire band.  Shift cal_min down by SHRINK_FACTOR to
        search lower.  (Mirror logic for all-contributing.)
        """
        new_min = current_cal_min
        new_max = current_cal_max

        # "HP not contributing" evidence → shrink cal_max down
        for delta in self._contribution_evidence_above:
            confirming = [
                d for d in self._contribution_evidence_above
                if abs(d - delta) < SHRINK_DELTA_TOLERANCE
            ]
            if len(confirming) >= SHRINK_CONFIRMATIONS:
                # Transition is below the lowest confirmed delta
                candidate = min(confirming) - SHRINK_DELTA_TOLERANCE
                if candidate < new_max:
                    new_max = candidate
                    self._confirmations_total += 1

        # "HP was contributing" evidence → shrink cal_min up
        for delta in self._contribution_evidence_below:
            confirming = [
                d for d in self._contribution_evidence_below
                if abs(d - delta) < SHRINK_DELTA_TOLERANCE
            ]
            if len(confirming) >= SHRINK_CONFIRMATIONS:
                # Transition is above the highest confirmed delta
                candidate = max(confirming) + SHRINK_DELTA_TOLERANCE
                if candidate > new_min:
                    new_min = candidate
                    self._confirmations_total += 1

        # Band shift: if evidence is overwhelmingly one-sided, the
        # transition is outside the band.  Shift the band to search.
        # Requires a strong majority (>= 5:1 ratio) with enough probes
        # to avoid premature shifts from noise.
        min_for_shift = max(SHRINK_CONFIRMATIONS * 2, 4)
        n_above = len(self._contribution_evidence_above)
        n_below = len(self._contribution_evidence_below)
        total = n_above + n_below
        shifted = False
        if total >= min_for_shift:
            ratio_no_hp = n_above / max(total, 1)
            ratio_has_hp = n_below / max(total, 1)
            if ratio_no_hp >= 0.8 and n_above >= min_for_shift:
                # Overwhelmingly no HP → boundary is below band
                shift = SHRINK_FACTOR * (current_cal_max - current_cal_min)
                new_min = current_cal_min - shift
                new_max = current_cal_max - shift
                shifted = True
                _LOGGER.info(
                    "Regime probe: band shifted down by %.1f°C "
                    "(%d/%d probes 'no HP')",
                    shift, n_above, total,
                )
            elif ratio_has_hp >= 0.8 and n_below >= min_for_shift:
                # Overwhelmingly HP contributing → boundary is above band
                shift = SHRINK_FACTOR * (current_cal_max - current_cal_min)
                new_min = current_cal_min + shift
                new_max = current_cal_max + shift
                shifted = True
                _LOGGER.info(
                    "Regime probe: band shifted up by %.1f°C "
                    "(%d/%d probes 'HP contributing')",
                    shift, n_below, total,
                )

        if shifted:
            # Clear evidence so the next round evaluates the new location
            self._contribution_evidence_above.clear()
            self._contribution_evidence_below.clear()

        if new_max < current_cal_max:
            _LOGGER.info(
                "Regime probe: cal_max shrunk %.1f→%.1f°C",
                current_cal_max, new_max,
            )
        if new_min > current_cal_min:
            _LOGGER.info(
                "Regime probe: cal_min shrunk %.1f→%.1f°C",
                current_cal_min, new_min,
            )

        return new_min, new_max

    # ── Window ───────────────────────────────────────────────────────

    def _in_window(self, hour: int) -> bool:
        """True if current hour is within configured window (or no window)."""
        if self._window_start is None or self._window_end is None:
            return True
        s, e = self._window_start, self._window_end
        if s <= e:
            return s <= hour < e
        return hour >= s or hour < e  # wraps midnight

    # ── External API ─────────────────────────────────────────────────

    def abort(self, reason: str = "") -> None:
        """Abort any active probe and return to IDLE."""
        self._abort(reason or "external")

    def request_early_probe(self) -> None:
        """Request a probe as soon as conditions allow.

        Called by boundary estimator when passive estimation has stalled.
        - COOLDOWN: expires cooldown timer so next probe fires immediately.
        - IDLE: sets forced flag so probe fires without requiring the HP
          to be in the uncertain zone (bypasses _is_uncertain check).
        """
        if self._state == ProbeState.COOLDOWN:
            self._cooldown_end_mono = 0.0
        elif self._state == ProbeState.IDLE:
            self._forced_probe = True

    # ── Persistence ──────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Serialise counters and evidence (state resets to IDLE on restart)."""
        return {
            "probes_completed": self._probes_completed,
            "no_contribution_count": self._no_contribution_count,
            "confirmations_total": self._confirmations_total,
            "evidence_above": list(self._contribution_evidence_above),
            "evidence_below": list(self._contribution_evidence_below),
            "forced_probe": self._forced_probe,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore counters from persisted data."""
        self._probes_completed = int(data.get("probes_completed", 0))
        self._no_contribution_count = int(data.get("no_contribution_count", 0))
        self._confirmations_total = int(data.get("confirmations_total", 0))
        self._contribution_evidence_above = [
            float(d) for d in data.get("evidence_above", [])
        ]
        self._contribution_evidence_below = [
            float(d) for d in data.get("evidence_below", [])
        ]
        self._forced_probe = bool(data.get("forced_probe", False))
