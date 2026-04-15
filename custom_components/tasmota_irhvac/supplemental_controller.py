"""Override/selector control for supplemental heat/cool sources.

Pure computation — no Home Assistant dependencies.  When a supplemental source
(e.g., fireplace) is actively heating, the HP enters "tracking mode": it
computes PI internally but doesn't send IR commands, deferring to the
supplemental.  If the supplemental can't keep up (room stays cold past a
threshold), the HP assists by resuming IR.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SupplementalResult:
    """Result of supplemental source evaluation."""

    hp_should_send_ir: bool
    should_reset_hold_timer: bool


class SupplementalController:
    """Override/selector control for supplemental heat/cool sources."""

    def __init__(
        self,
        sources: list[dict[str, Any]],
        deadband: float,
    ) -> None:
        self._sources: list[dict[str, Any]] = sources
        self._deadband: float = deadband
        self.tracking_mode: bool = False
        self.tracking_sources: list[str] = []
        self.failure_start: float | None = None
        self.assist_active: bool = False

    @property
    def has_sources(self) -> bool:
        """Whether any supplemental sources are configured."""
        return bool(self._sources)

    def evaluate(
        self,
        error_c: float,
        now_mono: float,
        get_entity_state: Callable[[str], str | None],
        *,
        pi_integral: float = 0.0,
        hp_setpoint: float | None = None,
    ) -> SupplementalResult:
        """Evaluate supplemental sources and return control decision.

        Args:
            error_c: Current error in °C (desired - current).
            now_mono: Monotonic time in seconds.
            get_entity_state: Callback that returns entity state string for a
                given entity_id, or None if unavailable/unknown.
            pi_integral: Current PI integral (for logging only).
            hp_setpoint: Current HP setpoint (for logging only).

        Returns:
            SupplementalResult with hp_should_send_ir and should_reset_hold_timer.
        """
        if not self._sources:
            return SupplementalResult(hp_should_send_ir=True, should_reset_hold_timer=False)

        active_sources = []
        for source in self._sources:
            entity_id = source.get("entity_id", "")
            if not entity_id:
                continue
            state = get_entity_state(entity_id)
            if state is None or state in ("unavailable", "unknown"):
                continue
            # Climate entity in heat or cool mode = supplemental is managing the room
            if state in ("heat", "cool"):
                active_sources.append(source.get("name", entity_id))

        was_tracking = self.tracking_mode
        should_reset_hold_timer = False

        if not active_sources:
            # No supplemental active → HP is in charge
            self.tracking_mode = False
            self.tracking_sources = []
            self.failure_start = None
            self.assist_active = False

            if was_tracking:
                _LOGGER.info(
                    "Supplemental override ended (sources: %s). HP resuming with integral=%.2f, setpoint=%s",
                    self.tracking_sources if self.tracking_sources else "none",
                    pi_integral, hp_setpoint,
                )
                # Bumpless transfer: clear hold timer so first IR send isn't blocked
                should_reset_hold_timer = True
            return SupplementalResult(hp_should_send_ir=True, should_reset_hold_timer=should_reset_hold_timer)

        # At least one supplemental is active
        self.tracking_sources = active_sources

        # Failure detection: is the supplemental keeping up?
        min_threshold = min(
            s.get("failure_threshold", 900) for s in self._sources
            if s.get("name", "") in active_sources
        ) if active_sources else 900

        # recovery_margin is stored in °C — read directly
        recovery_margin = min(
            s.get("recovery_margin", 0.3) for s in self._sources
            if s.get("name", "") in active_sources
        ) if active_sources else 0.3

        if error_c > self._deadband:
            # Room is below desired
            if self.failure_start is None:
                self.failure_start = now_mono
            time_below = now_mono - self.failure_start
            if time_below >= min_threshold:
                if not self.assist_active:
                    _LOGGER.info(
                        "Supplemental can't keep up (%.0fs below desired). HP assisting.",
                        time_below,
                    )
                self.assist_active = True
        else:
            if error_c < -recovery_margin:
                # Room above desired + margin → supplemental caught up
                if self.assist_active:
                    _LOGGER.info("Supplemental recovered. HP deferring again.")
                self.assist_active = False
            self.failure_start = None

        if self.assist_active:
            self.tracking_mode = False  # HP active (assisting)
        else:
            self.tracking_mode = True   # HP tracking (deferred)

        if self.tracking_mode and not was_tracking:
            _LOGGER.info(
                "Supplemental override started: %s. HP entering tracking mode.",
                ", ".join(active_sources),
            )

        return SupplementalResult(
            hp_should_send_ir=not self.tracking_mode,
            should_reset_hold_timer=should_reset_hold_timer,
        )
