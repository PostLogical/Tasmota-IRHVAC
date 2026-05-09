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
