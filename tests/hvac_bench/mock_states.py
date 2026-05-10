"""Bench-side `hass.states` substitute backed by a real dict.

Replaces the per-tick `pi._hass.states.get = lambda eid: _mock_states.get(eid)`
override the runner used to install (see #84). Tests construct one
`MockStates` and pass `mock_states.get` as the resolver; bench code calls
`mock_states.set(entity_id, value, unit)` whenever a sensor "changes."

The duck-typed `_MockState` mirrors what `_resolve_model_input_states`
reads from a real HA `State`: `.state` (str) and `.attributes` dict
keyed by `unit_of_measurement`.
"""

from __future__ import annotations


class _MockState:
    """Minimal duck-typed HA state object — only fields the controller reads."""

    __slots__ = ("state", "attributes")

    def __init__(self, state: str, unit: str | None) -> None:
        self.state = state
        self.attributes = {"unit_of_measurement": unit}


class MockStates:
    """Real-dict-backed substitute for `hass.states` in bench tests."""

    def __init__(self) -> None:
        self._states: dict[str, _MockState] = {}

    def get(self, entity_id: str) -> _MockState | None:
        return self._states.get(entity_id)

    def set(self, entity_id: str, value: object, unit: str | None = None) -> None:
        self._states[entity_id] = _MockState(str(value), unit)

    def remove(self, entity_id: str) -> None:
        self._states.pop(entity_id, None)


class _BenchHass:
    """Hand-rolled hass fake — exposes only the bench-active controller surface.

    The PI subpackage (``custom_components/tasmota_irhvac/pi/``) is
    deliberately HA-decoupled.  Its bench-active hass surface is:

      * ``hass.states.get(eid)`` — sensor reads (covered by ``MockStates``).
      * ``hass.is_running`` — boolean read once for event metadata.
      * ``hass.data`` — read by HA's ``async_dispatcher_send`` helper,
        which the controller invokes per batch (line ~1528 of pi_controller).
        Empty ``dict`` is the right value for bench: dispatcher-send
        early-exits when there are no subscribed listeners.
      * ``hass.verify_event_loop_thread(name)`` — debug guard called by
        ``async_dispatcher_send``.  Bench has its own deterministic loop
        and isn't doing cross-thread calls; the guard is a no-op.

    Other hass attributes (``async_add_executor_job``, ``config.path``,
    ``bus``) live behind feature flags the bench doesn't enable (e.g.
    ``_pi_event_log_enabled = False``).  Exposing only the active surface
    and raising ``AttributeError`` on anything else means any new
    controller hass dependency surfaces immediately as a test failure
    rather than silently consuming a truthy ``MagicMock`` child — this
    is the architectural fix for the bug class first seen in #83
    (``_FakeBenchEntity.hass = MagicMock()`` letting Mock attrs leak into
    bench math).

    Use ``__slots__`` so attribute typos on this object also fail loud.
    """

    __slots__ = ("states", "is_running", "data")

    def __init__(self, mock_states: MockStates) -> None:
        self.states = mock_states
        self.is_running = True
        # Empty dict — dispatcher_send checks ``hass.data.get(DATA_DISPATCHER)``
        # and early-exits on None.  Bench has no dispatcher subscribers so
        # this is the desired path.
        self.data: dict = {}

    def verify_event_loop_thread(self, what: str) -> None:
        """No-op: bench runs deterministically on its own loop."""
