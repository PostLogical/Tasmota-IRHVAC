"""Plant identification providers.

Each provider implements a different identification method (step response,
area method, relay test) and produces ParameterEstimate values.  Providers
are stateful but share no mutable state with each other — the orchestrator
(PlantIdentifier) coordinates them via frozen ObservationContext objects.
"""
