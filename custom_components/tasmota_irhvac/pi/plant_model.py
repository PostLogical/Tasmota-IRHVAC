"""Data structures for plant identification.

Pure data — no computation, no Home Assistant dependencies.
All dataclasses are frozen (immutable) to prevent shared-state bugs.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ParameterEstimate:
    """Single plant parameter estimate with provenance metadata."""

    value: float
    confidence: float = 0.0  # 0.0 = seed only, grows toward 1.0
    source: str = "seed"  # "seed", "step_response", "area_method", "relay_test"
    observations: int = 0  # how many observations contributed

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "observations": self.observations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParameterEstimate:
        return cls(
            value=float(d.get("value", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            source=str(d.get("source", "seed")),
            observations=int(d.get("observations", 0)),
        )


@dataclasses.dataclass(frozen=True)
class PlantEstimate:
    """Complete SOPDT plant model: K, θ, τ_fast, τ_slow.

    All fields are ParameterEstimate with provenance.  Frozen — the
    orchestrator creates a new instance on every update.
    """

    k: ParameterEstimate  # process gain (°C room / °C HP setpoint)
    theta: ParameterEstimate  # dead time (minutes)
    tau_fast: ParameterEstimate  # fast time constant — air node (minutes)
    tau_slow: ParameterEstimate  # slow time constant — wall/mass node (minutes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k.as_dict(),
            "theta": self.theta.as_dict(),
            "tau_fast": self.tau_fast.as_dict(),
            "tau_slow": self.tau_slow.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlantEstimate | None:
        if not d:
            return None
        try:
            return cls(
                k=ParameterEstimate.from_dict(d["k"]),
                theta=ParameterEstimate.from_dict(d["theta"]),
                tau_fast=ParameterEstimate.from_dict(d["tau_fast"]),
                tau_slow=ParameterEstimate.from_dict(d["tau_slow"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def from_seeds(
        cls, tau_seed: float, response_lag: float, k_eff: float = 1.0
    ) -> PlantEstimate:
        """Create a seed-only PlantEstimate from config values."""
        return cls(
            k=ParameterEstimate(value=k_eff, source="seed"),
            theta=ParameterEstimate(value=response_lag, source="seed"),
            tau_fast=ParameterEstimate(value=tau_seed, source="seed"),
            tau_slow=ParameterEstimate(value=tau_seed, source="seed"),
        )


@dataclasses.dataclass(frozen=True)
class ObservationContext:
    """Shared context for a step-response observation.

    Created by the orchestrator when a setpoint step is detected.
    Passed to each provider independently — providers store their
    own copy, no shared mutable state.
    """

    start_time: float  # monotonic seconds
    baseline_temp: float  # room temp at step start (°C)
    target_temp: float  # desired temp (°C)
    step_magnitude: float  # signed HP setpoint change (°C)
    ff_offset: float  # FF offset at step start (°C)


@dataclasses.dataclass(frozen=True)
class GainUpdate:
    """Result of a gain recomputation.

    Produced by the orchestrator, consumed by the PI controller.
    tau_fast → Smith predictor, tau_slow → Kp scheduling.
    """

    kp: float
    ki: float
    tau_fast: float
    tau_slow: float
    lag: float
