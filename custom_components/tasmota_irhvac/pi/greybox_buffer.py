"""Observation buffer for grey-box 1R1C identification.

Subclass of DiversityAwareBuffer with a different feature vector and
admission policy.  Uses the same D-optimal leverage scoring to retain
diverse observations across the full operating range.

Key differences from the WLS DiversityAwareBuffer:
- Feature vector: [outdoor_delta, hp_offset, solar, room_rate] — the
  physical variables that matter for 1R1C energy balance identification
- Admits HP-off observations (critical for isolating ua_c)
- Admits passive-tick observations (hvac_mode=OFF)
- Single buffer per zone (mode-agnostic, not split by heat/cool)
- Larger default size (6000 vs 2000)

Standalone module — no Home Assistant dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from .batch_learning import DiversityAwareBuffer, Observation

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)

# Default capacity.  6000 observations at 4/hr fills in ~62 days.
# scipy least_squares fits 6000×3 parameters in <2s.
DEFAULT_GREYBOX_BUFFER_SIZE = 6000

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
    ) -> None:
        super().__init__(
            n_features=_GREYBOX_N_FEATURES,
            max_size=max_size,
            feature_order=_GREYBOX_FEATURE_ORDER,
            model_inputs=[],
        )
        self._solar_entity = solar_entity

    @property
    def solar_entity(self) -> str | None:
        return self._solar_entity

    @solar_entity.setter
    def solar_entity(self, value: str | None) -> None:
        self._solar_entity = value

    def add(self, obs: Observation) -> None:
        """Admit an observation into the buffer.

        Only rejects observations with outdoor_temp_c=None.
        Unlike the WLS buffer, admits HP-off observations.
        """
        if obs.outdoor_temp_c is None:
            return
        super().add(obs)

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
    ) -> "GreyboxBuffer":
        """Deserialize from stored dicts, recomputing the info matrix.

        Corrupt or unreadable entries are silently skipped.
        """
        buf = cls(max_size=max_size, solar_entity=solar_entity)
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
