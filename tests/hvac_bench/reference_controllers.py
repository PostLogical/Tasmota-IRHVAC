"""Reference-controller suite for the bench locked-score harness.

Phase 2a of the bench-validation plan. Three controllers, all conforming
to ``HVACController`` protocol, plus an optional ``batch_update(tick)``
hook for controllers that learn:

* :class:`NaiveBangBangController` — pure thermostat. No FF, no learning,
  no PI. Setpoint snaps to ``max_temp``/``min_temp`` with hysteresis.
  Lower-bound on bench KPIs.

* :func:`make_well_tuned_pi` — production PI controller with learning
  *disabled* (batch WLS off, online RLS off, plant-ID off) and FF
  coefficients seeded with ground-truth values. Mid-baseline: shows what
  the controller can do when it knows the right coefficients but cannot
  improve them.

* :func:`make_production_pi` — production PI controller with learning
  *enabled* (default config). Upper baseline (within the family — an
  oracle MPC would be a true upper bound, but the plan defers that).

Pattern follows BOPTEST's reference-controller suite plus Tennessee
Eastman's "known-good vs naive" practice. The full mid-baseline / lower-
bound / upper-baseline triple gives discriminative power: a bench change
that silently breaks the lower bound below naive, or pushes naive above
WellTunedPI, is a regression that the locked-score regression test
(Phase 2b) catches.
"""

from __future__ import annotations

from typing import Iterable

from tests.hvac_bench.adapters import TasmotaPIAdapter
from tests.hvac_bench.full_stack_runner import ModelInputSpec
from tests.hvac_bench.house_profiles import PROFILES, PROFILES_2R2C, HouseProfile2R2C


# ── Naive bang-bang (lower bound) ─────────────────────────────────────────


class NaiveBangBangController:
    """Thermostat-style on/off controller with hysteresis.

    Heat mode: when room < desired - hysteresis, drive HP to ``max_temp``
    (rail). When room > desired + hysteresis, drive HP to ``min_temp``
    (off). Inside the hysteresis band, hold the previous setpoint.

    Cool mode: mirrored.

    No PI, no FF, no learning. Used to establish the bench's lower bound
    on KPIs — any controller worth deploying must beat it on ``tdis_tot``
    (or beat it on ``ener_tot`` while staying close on comfort).
    """

    def __init__(
        self,
        *,
        min_temp: float = 16.0,
        max_temp: float = 30.0,
        hysteresis_c: float = 0.5,
        initial_hp_setpoint: float | None = None,
    ):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.hysteresis_c = hysteresis_c
        self.desired = 20.5
        self._mode = "heat"
        # Initial position: rail to max in heat mode, min in cool mode, so
        # the very first tick has a defined setpoint without depending on
        # whether the room starts inside or outside the band.
        if initial_hp_setpoint is not None:
            self.hp_setpoint = float(initial_hp_setpoint)
        else:
            self.hp_setpoint = max_temp

    def tick(self, room_temp_c, outdoor_temp_c, dt_seconds, model_inputs=None):
        if self._mode == "heat":
            if room_temp_c < self.desired - self.hysteresis_c:
                self.hp_setpoint = self.max_temp
            elif room_temp_c > self.desired + self.hysteresis_c:
                self.hp_setpoint = self.min_temp
            # else: hold previous setpoint (hysteresis band)
        else:
            # cool: low setpoint = compressor pull down, high = idle
            if room_temp_c > self.desired + self.hysteresis_c:
                self.hp_setpoint = self.min_temp
            elif room_temp_c < self.desired - self.hysteresis_c:
                self.hp_setpoint = self.max_temp
        return float(self.hp_setpoint)

    def set_desired_temp(self, temp_c):
        self.desired = float(temp_c)

    def set_mode(self, mode):
        self._mode = mode
        # On mode switch, reset the rail to the active push direction so
        # the first tick after switch-on doesn't inherit the off-mode rail.
        self.hp_setpoint = self.max_temp if mode == "heat" else self.min_temp

    def get_state(self):
        return {
            "integral": 0.0,
            "ff_offset": 0.0,
            "desired_temp": self.desired,
            "hp_setpoint": self.hp_setpoint,
            "rls_obs_count": 0,
        }

    def batch_update(self, tick: int) -> None:
        """No-op — bang-bang has no learning state."""


# ── PI factory helpers ────────────────────────────────────────────────────


def _resolve_profile(profile_name: str) -> HouseProfile2R2C:
    if profile_name in PROFILES_2R2C:
        return PROFILES_2R2C[profile_name]
    if profile_name in PROFILES:
        return PROFILES[profile_name]
    raise ValueError(f"Unknown profile: {profile_name}")


def _build_pi_overrides(
    profile: HouseProfile2R2C,
    *,
    mode: str,
    learning_enabled: bool,
    seed_with_truth: bool,
    extra_overrides: dict | None,
    model_inputs: Iterable[ModelInputSpec] | None,
) -> dict:
    """Build the PI config-override dict shared by both PI factories.

    ``learning_enabled=False`` disables batch WLS and plant-ID so the
    controller's coefficients stay frozen. ``seed_with_truth=True`` uses
    the profile's true outdoor seed and the per-input ``_true_ff_coef``
    as initial coefficients (well-tuned baseline). When False, default
    seeds are used (production behavior on a fresh deployment).
    """
    overrides: dict = {}
    overrides["pi_batch_wls_enabled"] = bool(learning_enabled)
    overrides["pi_plant_id_enabled"] = bool(learning_enabled)

    if seed_with_truth:
        seed_key = "pi_outdoor_seed_heat" if mode == "heat" else "pi_outdoor_seed_cool"
        overrides[seed_key] = float(profile.true_seed)

    if model_inputs is not None:
        # Build pi_model_inputs entries with seeds.  When seeding with
        # truth, use ``_true_ff_coef``; otherwise honour the spec's
        # author-supplied ``seed_heat``/``seed_cool``.
        entries: list[dict] = []
        for mi in model_inputs:
            if seed_with_truth:
                seed_heat = mi.true_ff_coef(profile.hp_gain)
                seed_cool = mi.true_ff_coef(profile.hp_gain)
            else:
                seed_heat = mi.seed_heat
                seed_cool = mi.seed_cool
            entry: dict = {
                "entity_id": mi.entity_id,
                "name": mi.name,
                "input_role": mi.input_role,
                "seed_heat": seed_heat,
                "seed_cool": seed_cool,
                "lag_tau": mi.lag_tau,
            }
            if mi.clamp_min is not None:
                entry["clamp_min"] = mi.clamp_min
            if mi.clamp_max is not None:
                entry["clamp_max"] = mi.clamp_max
            entries.append(entry)
        overrides["pi_model_inputs"] = entries

    if extra_overrides:
        overrides.update(extra_overrides)
    return overrides


class _BatchAwarePIController:
    """TasmotaPIAdapter wrapper that adds a ``batch_update(tick)`` hook.

    Composition over inheritance — keeps the adapter's __init__ contract
    untouched and routes the optional learning hook through a single
    method the runner can call without having to know which controller
    class it has.
    """

    def __init__(self, adapter: TasmotaPIAdapter, *, learning_enabled: bool):
        self._adapter = adapter
        self._learning_enabled = learning_enabled

    # ── HVACController protocol — delegate to adapter ────────────────
    def tick(self, room_temp_c, outdoor_temp_c, dt_seconds, model_inputs=None):
        return self._adapter.tick(room_temp_c, outdoor_temp_c, dt_seconds, model_inputs)

    def set_desired_temp(self, temp_c):
        self._adapter.set_desired_temp(temp_c)

    def set_mode(self, mode):
        self._adapter.set_mode(mode)

    def get_state(self):
        return self._adapter.get_state()

    # ── Learning hook ────────────────────────────────────────────────
    def batch_update(self, tick: int) -> None:
        if not self._learning_enabled:
            return
        self._adapter._pi._run_batch_analysis()

    # ── Adapter passthrough for tests ────────────────────────────────
    @property
    def adapter(self) -> TasmotaPIAdapter:
        return self._adapter

    @property
    def pi(self):
        return self._adapter._pi


def make_well_tuned_pi(
    profile_name: str,
    *,
    mode: str = "heat",
    model_inputs: Iterable[ModelInputSpec] | None = None,
    extra_overrides: dict | None = None,
    head_calibration_bounds: tuple[float, float] = (0.0, 0.0),
) -> _BatchAwarePIController:
    """Build a production PI controller with learning **disabled** and FF
    coefficients seeded to ground-truth values.

    This is the mid-baseline: the controller knows the right physics but
    cannot improve its model. Useful for separating "the seed was good"
    from "learning helped" in KPI deltas.
    """
    profile = _resolve_profile(profile_name)
    overrides = _build_pi_overrides(
        profile,
        mode=mode,
        learning_enabled=False,
        seed_with_truth=True,
        extra_overrides=extra_overrides,
        model_inputs=model_inputs,
    )
    adapter = TasmotaPIAdapter(
        config_overrides=overrides,
        head_calibration_bounds=head_calibration_bounds,
    )
    return _BatchAwarePIController(adapter, learning_enabled=False)


def make_production_pi(
    profile_name: str,
    *,
    mode: str = "heat",
    model_inputs: Iterable[ModelInputSpec] | None = None,
    seed_with_truth: bool = False,
    extra_overrides: dict | None = None,
    head_calibration_bounds: tuple[float, float] = (0.0, 0.0),
) -> _BatchAwarePIController:
    """Build the full production PI+FF+WLS controller (learning enabled).

    Defaults to *not* seeding with truth — that's the realistic
    "fresh-deployment" condition the production controller has to recover
    from. Set ``seed_with_truth=True`` to compare against
    :func:`make_well_tuned_pi` on equal seed footing (isolating the
    learning contribution).
    """
    profile = _resolve_profile(profile_name)
    overrides = _build_pi_overrides(
        profile,
        mode=mode,
        learning_enabled=True,
        seed_with_truth=seed_with_truth,
        extra_overrides=extra_overrides,
        model_inputs=model_inputs,
    )
    adapter = TasmotaPIAdapter(
        config_overrides=overrides,
        head_calibration_bounds=head_calibration_bounds,
    )
    return _BatchAwarePIController(adapter, learning_enabled=True)
