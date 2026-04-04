"""Abstract controller interface for HVAC benchmark.

Any controller implementing this protocol can be benchmarked against
the thermal model and scenario suite.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class HVACController(Protocol):
    """Protocol for HVAC controllers under test."""

    def tick(self, room_temp_c: float, outdoor_temp_c: float,
             dt_seconds: float, model_inputs: dict[str, float] | None = None) -> float:
        """Run one control step.

        Args:
            room_temp_c: Current room temperature in °C.
            outdoor_temp_c: Current outdoor temperature in °C.
            dt_seconds: Time since last tick in seconds.
            model_inputs: Optional dict of named model input values
                         (e.g., {"solar": 0.7, "stove": 1.0}).

        Returns:
            HP setpoint in °C (may be float for 0.5°C-step systems).
        """
        ...

    def set_desired_temp(self, temp_c: float) -> None:
        """Set the target temperature in °C."""
        ...

    def set_mode(self, mode: str) -> None:
        """Set operating mode: 'heat' or 'cool'."""
        ...

    def get_state(self) -> dict:
        """Return controller internal state for metrics.

        Should include at minimum:
            - integral: float (PI integral accumulator)
            - ff_offset: float (feedforward contribution)

        May also include:
            - rls_obs_count: int
            - d_term: float
            - Any other diagnostic state
        """
        ...
