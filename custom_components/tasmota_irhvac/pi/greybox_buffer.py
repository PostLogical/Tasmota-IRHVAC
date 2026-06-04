"""Append-only observation buffer for grey-box 1R1C/2R2C identification.

Architecture (2026-06-04 redesign):

Greybox uses **fill-and-wipe** semantics paired with the weekly-cadence
gate in ``pi_controller``. Each tick admits an observation (rejecting
those with no outdoor temperature); at each weekly batch fit, the
controller consumes ``get_all()``, runs ``fit_greybox()``, promotes the
posterior into the chain prior (Pathak §4.2), and calls ``clear()``.
Next week starts with an empty buffer. Across batches there is zero
observation overlap — continuity of structural parameter estimates is
carried by the persisted ``PriorState`` chain prior, not by re-fitting
shared observations.

This replaces the older ``DiversityAwareBuffer`` inheritance with its
time-window or leverage-scoring policies. That structure was inherited
from WLS where intra-day operational refresh benefits from sliding-
window admission scoring. None of it applies to greybox's structural-ID
purpose:

- No leverage scoring: there is no "which observations to evict when
  full" decision — the controller wipes everything at batch boundary.
- No D-optimal feature vector caching: nothing consumes it.
- No info-matrix tracking: nothing consumes it.
- No swappable policy abstraction: there is one behavior (append; cap
  at max_size as memory safety; clear() at batch).

Standalone module — no Home Assistant dependencies. Imports
``BufferAddResult`` and ``Observation`` only for type-compatibility with
the controller's existing observation-context plumbing.
"""

from __future__ import annotations

import logging
from typing import Any

from .batch_learning import BufferAddResult, Observation

_LOGGER = logging.getLogger(__name__)

# Memory safety net only. Under the weekly fill-and-wipe cycle the
# normal high-water mark is one week of observations: at 60 s sensor
# cadence × 7 d ≈ 10 080. Cap at 32 000 leaves room for faster cadences
# (e.g. ~20 s ticks for 7 d ≈ 30 240) without practical risk of touching
# the FIFO drop path. If max_size is ever hit, the oldest observation is
# dropped — but only as a guardrail; correctness relies on the
# controller calling ``clear()`` each batch.
DEFAULT_GREYBOX_BUFFER_SIZE = 32000


_POLICY_NAME = "append_only"


class GreyboxBuffer:
    """Append-only buffer holding observations for the current batch window.

    Usage::

        buf = GreyboxBuffer(solar_entity=...)

        # Per tick:
        result = buf.add(obs)         # admitted unless outdoor_temp_c is None

        # At batch time (weekly cadence in production):
        observations = buf.get_all()  # snapshot for fit_greybox()
        # ... fit, promote chain prior ...
        buf.clear()                   # wipe for next week

        # Persistence:
        snapshot = buf.as_list()
        buf2 = GreyboxBuffer.from_list(snapshot, solar_entity=...)
    """

    def __init__(
        self,
        max_size: int = DEFAULT_GREYBOX_BUFFER_SIZE,
        solar_entity: str | None = None,
    ) -> None:
        self._buffer: list[Observation] = []
        self._max_size = int(max_size)
        self._solar_entity = solar_entity

    # ── Solar entity (used by ``fit_greybox`` to find the solar input) ──
    @property
    def solar_entity(self) -> str | None:
        return self._solar_entity

    @solar_entity.setter
    def solar_entity(self, value: str | None) -> None:
        self._solar_entity = value

    # ── Append (with admission gate + memory safety) ────────────────────
    def add(self, obs: Observation) -> BufferAddResult:
        """Admit ``obs`` into the buffer.

        Rejects observations with ``outdoor_temp_c is None`` — the energy
        balance has ``T_out`` in every term, so without it the observation
        cannot contribute to the fit. Otherwise appends. If the buffer is
        at ``max_size`` (safety net only), drops the oldest observation.
        """
        if obs.outdoor_temp_c is None:
            return BufferAddResult(
                admitted=False,
                candidate_score=None,
                evicted_timestamp=None,
                min_incumbent_score=None,
                rejection_reason="no_outdoor_temp",
                policy_name=_POLICY_NAME,
            )

        evicted_timestamp: float | None = None
        if len(self._buffer) >= self._max_size:
            # Memory safety net — should never trip under the weekly
            # fill-and-wipe cycle. Log when it does so anomalous cadences
            # are visible.
            evicted_timestamp = self._buffer[0].timestamp
            self._buffer.pop(0)
            _LOGGER.warning(
                "Grey-box buffer hit max_size=%d; dropping oldest observation. "
                "Indicates either a missed clear() at batch boundary or a "
                "cadence faster than 7d × max_size supports.",
                self._max_size,
            )
        self._buffer.append(obs)
        return BufferAddResult(
            admitted=True,
            candidate_score=None,
            evicted_timestamp=evicted_timestamp,
            min_incumbent_score=None,
            rejection_reason=None,
            policy_name=_POLICY_NAME,
        )

    # ── Consume / wipe ──────────────────────────────────────────────────
    def get_all(self) -> list[Observation]:
        """Snapshot of all observations currently held.

        Returns a shallow copy so callers iterating the result are
        insulated from concurrent ``add()`` or ``clear()`` activity.
        """
        return list(self._buffer)

    def clear(self) -> None:
        """Wipe the buffer. Called by the controller after a successful
        batch fit + chain promotion."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    # ── Diagnostics for state logging ───────────────────────────────────
    def get_diagnostics(self) -> dict[str, Any]:
        """Buffer state for ``extra_state_attributes`` and batch logs."""
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
            # ``min_leverage`` field retained as None for backward-compat
            # with diagnostic-snapshot consumers that read it. The leverage
            # concept is meaningless under fill-and-wipe (no per-observation
            # eviction score).
            "min_leverage": None,
        }

    # ── Persistence ─────────────────────────────────────────────────────
    def as_list(self) -> list[dict[str, Any]]:
        """Serialize the buffer's observations as a list of dicts using the
        schema-versioned ``Observation.as_dict`` format. Persisted state
        from this module is fully round-trippable with state persisted by
        the prior DiversityAwareBuffer-backed implementation."""
        return [o.as_dict() for o in self._buffer]

    @classmethod
    def from_list(
        cls,
        data: list[dict[str, Any]],
        max_size: int = DEFAULT_GREYBOX_BUFFER_SIZE,
        solar_entity: str | None = None,
    ) -> "GreyboxBuffer":
        """Restore from a serialized buffer.

        Silently skips entries that fail to deserialize (corrupt persisted
        state shouldn't crash the integration on restart). Drops entries
        with ``outdoor_temp_c is None``. Caps to ``max_size`` (newest
        retained — matches the FIFO-on-overflow rule of ``add()``).
        """
        buf = cls(max_size=max_size, solar_entity=solar_entity)
        kept: list[Observation] = []
        for d in data:
            try:
                obs = Observation.from_dict(d)
            except (KeyError, TypeError, ValueError):
                continue
            if obs.outdoor_temp_c is None:
                continue
            kept.append(obs)
        n_skipped = len(data) - len(kept)
        if n_skipped > 0:
            _LOGGER.info(
                "Grey-box buffer: skipped %d unreadable / invalid "
                "observations during restore",
                n_skipped,
            )
        if len(kept) > max_size:
            kept = kept[-max_size:]
        buf._buffer = kept
        return buf
