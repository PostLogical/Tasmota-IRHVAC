"""Modified Smith predictor for dead-time compensation (Åström Ch. 7 §7.3).

Uses an internal FOPDT model to estimate the temperature change "in the
pipeline" — corrections that have been commanded but haven't reached the
sensor due to transport delay L.
"""

from __future__ import annotations

import math


class SmithPredictor:
    """Modified Smith predictor for dead-time compensation (Åström Ch. 7 §7.3).

    Uses an internal FOPDT model to estimate the temperature change "in the
    pipeline" — corrections that have been commanded but haven't reached the
    sensor due to transport delay L.  Subtracting this from the PI error
    signal lets the controller react as if there were no delay.

    Two internal first-order models run in parallel:
      nodelay  — receives current HP setpoint immediately
      delayed  — receives HP setpoint from L minutes ago (ring buffer lookup)

    The correction term (nodelay − delayed) represents the pending temperature
    change.  At steady state the term is zero; after a setpoint change it
    transiently grows then decays as the real plant catches up.

    Robust to ±50% parameter mismatch: degrades gracefully (slower convergence)
    without instability, because the outer PI loop still closes on the real
    measurement.
    """

    def __init__(self, tau: float, lag: float, k_eff: float = 1.0) -> None:
        self.tau: float = max(tau, 1.0)   # thermal time constant (minutes)
        self.lag: float = lag             # transport delay L (minutes)
        self.k_eff: float = k_eff        # process gain (1.0 = unit gain)
        self._model_nodelay: float = 0.0  # no-delay model state (°C)
        self._model_delayed: float = 0.0  # delayed model state (°C)
        # Ring buffer: (monotonic_time, hp_setpoint) for delayed lookup
        self._setpoint_history: list[tuple[float, float]] = []
        self._initialized: bool = False

    def initialize(self, room_temp: float, hp_setpoint: float, now_mono: float) -> None:
        """Initialize model states to current room temperature.

        Both models start at the same value so correction = 0 on first tick.
        The Smith predictor is inert until the first setpoint change creates
        a divergence between the nodelay and delayed models.
        """
        self._model_nodelay = room_temp
        self._model_delayed = room_temp
        self._setpoint_history = [(now_mono, hp_setpoint)]
        self._initialized = True

    def update_params(self, tau: float, lag: float) -> None:
        """Update model parameters when τ estimate changes."""
        self.tau = max(tau, 1.0)
        self.lag = lag

    def record_setpoint(self, hp_setpoint: float, now_mono: float) -> None:
        """Record current HP setpoint for delayed lookup."""
        self._setpoint_history.append((now_mono, hp_setpoint))
        # Trim: keep enough history to look back L + margin
        cutoff = now_mono - (self.lag + 5.0) * 60.0
        while len(self._setpoint_history) > 2 and self._setpoint_history[0][0] < cutoff:
            self._setpoint_history.pop(0)

    def _delayed_setpoint(self, now_mono: float) -> float:
        """Look up HP setpoint from L minutes ago."""
        target_time = now_mono - self.lag * 60.0
        if not self._setpoint_history:
            return 0.0
        # Find last entry at or before target_time
        result = self._setpoint_history[0][1]
        for t, sp in self._setpoint_history:
            if t <= target_time:
                result = sp
            else:
                break
        return result

    def step(self, hp_setpoint: float, dt_seconds: float, now_mono: float) -> None:
        """Advance both internal models by one time step (exact exponential)."""
        if not self._initialized:
            return
        dt_min = dt_seconds / 60.0
        tau = self.tau
        decay = math.exp(-dt_min / tau) if tau > 0 else 0.0

        # No-delay model: receives current setpoint immediately
        eq_nd = self.k_eff * hp_setpoint
        self._model_nodelay = eq_nd + (self._model_nodelay - eq_nd) * decay

        # Delayed model: receives setpoint from L minutes ago
        u_delayed = self._delayed_setpoint(now_mono)
        eq_d = self.k_eff * u_delayed
        self._model_delayed = eq_d + (self._model_delayed - eq_d) * decay

    @property
    def correction(self) -> float:
        """Pipeline temperature: pending change that hasn't reached the sensor.

        Returns nodelay − delayed model prediction.  Positive means the room
        should get warmer than the sensor currently reads (heating in pipeline).
        """
        if not self._initialized:
            return 0.0
        return self._model_nodelay - self._model_delayed

    def reset(self, room_temp: float, hp_setpoint: float, now_mono: float) -> None:
        """Reset model states (mode change, large regime shift)."""
        self.initialize(room_temp, hp_setpoint, now_mono)
