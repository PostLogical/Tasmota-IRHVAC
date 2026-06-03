"""Observation buffer for grey-box 1R1C/2R2C identification.

Subclass of DiversityAwareBuffer specialized for state-space (sim-error
PEM) parameter identification. Defaults to ``TimeWindowPolicy`` —
keeps observations within ``DEFAULT_GREYBOX_WINDOW_SECONDS`` of the
latest admitted obs, with ``max_size`` as a memory safety net.

Why time-window, not count-based:
- Greybox 2R2C fit propagates state chronologically (sim-error PEM,
  matrix-exp ZOH). What matters for identifiability is *time-span*
  relative to the dominant time constant τ_slow, NOT raw sample count.
- A count-based policy (SlevPolicy / SlidingWindowPolicy) covers wildly
  different time-spans at different cadences: ``max_size=10000`` is ~7d
  at 60s ticks but ~104d at 15min ticks. The former is too short for
  envelope ID, the latter is wasted compute.
- Bacher & Madsen (2011), CTSM-R, Ljung §11.4 all define the window in
  time units.
- Adjacent timestamps in the retained window are uniform-spaced at the
  sensor cadence, so the per-fit dt-memoization cache in
  ``_fit_greybox_2r2c`` hits on (essentially) every tick — preserves the
  perf gains from the closed-form 2×2 expm path.

Key differences from the WLS DiversityAwareBuffer:
- Feature vector: [outdoor_delta, hp_offset, solar, room_rate] — the
  physical variables that matter for 1R1C/2R2C energy balance ID
- Admits HP-off observations (critical for isolating ua_c)
- Admits passive-tick observations (hvac_mode=OFF)
- Single buffer per zone (mode-agnostic, not split by heat/cool)
- Time-window default (7d) sized for non-transfer-learning fallback
  operation; with Pathak §4.2 posterior-chain transfer learning ahead,
  this can drop to ~72h once the prior carries forward.

Standalone module — no Home Assistant dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from .batch_learning import BufferAddResult, DiversityAwareBuffer, Observation
from .buffer_policies import TimeWindowPolicy

if TYPE_CHECKING:
    from .buffer_policies import BufferPolicy

_LOGGER = logging.getLogger(__name__)

# Default time window: 7 days. Sized so envelope τ_slow (up to ~58h
# lit-typical max) can be identified from 3+ time constants of data.
# At 60s cadence this is ~10080 obs (within default max_size); at 15min
# cadence ~672 obs (well within bounds). With Pathak §4.2 posterior-
# chain transfer learning, can be reduced to ~72h × n_batches later.
DEFAULT_GREYBOX_WINDOW_SECONDS: float = 7.0 * 24.0 * 3600.0  # 7 days

# Memory safety net. At 60s cadence × 7 days = 10080 obs; bump to
# 12000 to give the time-window first-eviction priority. Above 12000
# the FIFO falloff kicks in (only at sub-60s cadence — unusual).
DEFAULT_GREYBOX_BUFFER_SIZE = 12000

# Grey-box feature vector: 4 features for leverage scoring.
_GREYBOX_N_FEATURES = 4
_GREYBOX_FEATURE_ORDER = ["outdoor_delta", "hp_offset", "solar", "room_rate"]


class GreyboxBuffer(DiversityAwareBuffer):
    """Leverage-scored observation buffer for grey-box identification.

    Uses the same D-optimal leverage scoring as the WLS buffer but with
    a feature vector tailored for 1R1C energy balance identification:

        [outdoor_delta, hp_offset, solar, room_rate]

    Admits all observations with valid outdoor_temp_c, including HP-off
    (clamped_reason="no_output") and passive-tick observations.  HP-off
    observations are critical for isolating ua_c (envelope heat loss)
    since hp_offset=0 removes k_c from the energy balance.

    Usage::

        buf = GreyboxBuffer()
        buf.add(obs)                     # every tick (HP-on and HP-off)
        observations = buf.get_all()     # feed to fit_greybox()
        serialized = buf.as_list()       # persist
        buf = GreyboxBuffer.from_list(serialized)  # restore
    """

    def __init__(
        self,
        max_size: int = DEFAULT_GREYBOX_BUFFER_SIZE,
        solar_entity: str | None = None,
        policy: "BufferPolicy | None" = None,
        window_seconds: float = DEFAULT_GREYBOX_WINDOW_SECONDS,
    ) -> None:
        # Default to TimeWindowPolicy for greybox; sim-error PEM needs
        # contiguous recent observations, not eviction-sparse sampling.
        # Callers can pass any policy explicitly (e.g. bench tests that
        # exercise SlevPolicy / LeveragePolicy variants).
        if policy is None:
            policy = TimeWindowPolicy(window_seconds=window_seconds)
        super().__init__(
            n_features=_GREYBOX_N_FEATURES,
            max_size=max_size,
            feature_order=_GREYBOX_FEATURE_ORDER,
            model_inputs=[],
            policy=policy,
        )
        self._solar_entity = solar_entity

    @property
    def solar_entity(self) -> str | None:
        return self._solar_entity

    @solar_entity.setter
    def solar_entity(self, value: str | None) -> None:
        self._solar_entity = value

    def add(self, obs: Observation) -> BufferAddResult:
        """Admit an observation into the buffer.

        Returns a `BufferAddResult` describing the decision. Only rejects
        observations with outdoor_temp_c=None up front; otherwise
        delegates to the leverage-scored super().add(). Unlike the WLS
        buffer, admits HP-off observations.
        """
        if obs.outdoor_temp_c is None:
            return BufferAddResult(
                admitted=False,
                candidate_score=None,
                evicted_timestamp=None,
                min_incumbent_score=None,
                rejection_reason="no_outdoor_temp",
                policy_name=self._policy.name,
            )
        return super().add(obs)

    def _get_feature_vector(self, obs: Observation) -> list[float]:
        """Build the 4-feature vector for grey-box leverage scoring.

        [outdoor_delta, hp_offset, solar, room_rate]
        """
        outdoor_delta = (
            (obs.outdoor_temp_c - obs.current_c)
            if obs.outdoor_temp_c is not None
            else 0.0
        )
        hp_offset = (
            (obs.hp_setpoint - obs.current_c)
            if obs.hp_setpoint is not None
            and obs.clamped_reason != "no_output"
            else 0.0
        )
        solar = (
            obs.raw_readings.get(self._solar_entity, 0.0)
            if self._solar_entity is not None
            else 0.0
        )
        return [outdoor_delta, hp_offset, solar, obs.room_rate]

    @classmethod
    def from_list(  # type: ignore[override]
        cls,
        data: list[dict[str, Any]],
        max_size: int = DEFAULT_GREYBOX_BUFFER_SIZE,
        solar_entity: str | None = None,
        policy: "BufferPolicy | None" = None,
    ) -> "GreyboxBuffer":
        """Deserialize from stored dicts, recomputing the info matrix.

        Corrupt or unreadable entries are silently skipped.
        """
        buf = cls(max_size=max_size, solar_entity=solar_entity, policy=policy)
        observations: list[Observation] = []
        for d in data:
            try:
                obs = Observation.from_dict(d)
                if obs.outdoor_temp_c is not None:
                    observations.append(obs)
            except (KeyError, TypeError, ValueError):
                continue

        n_skipped = len(data) - len(observations)
        if n_skipped > 0:
            _LOGGER.info(
                "Grey-box buffer: skipped %d unreadable observations during restore",
                n_skipped,
            )

        if len(observations) > max_size:
            observations = observations[-max_size:]

        buf._buffer = observations
        buf.recompute_info_matrix()
        return buf

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostic information for extra_state_attributes."""
        total = len(self._buffer)
        hp_off_count = sum(
            1 for o in self._buffer
            if o.clamped_reason == "no_output" or o.hp_setpoint is None
        )
        return {
            "total": total,
            "max_size": self._max_size,
            "hp_off_pct": (
                round(100 * hp_off_count / total, 1) if total > 0 else None
            ),
            "min_leverage": round(self.get_min_leverage(), 6) if total > 0 else None,
        }
