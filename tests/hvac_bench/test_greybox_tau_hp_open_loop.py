"""Open-loop τ_hp identifiability tests using the bench's ThermalModel2R2C.

Stage 1e (2026-06-03). These tests directly call ``ThermalModel2R2C`` to
generate observations with a known ``tau_hp_minutes``, then call
``fit_greybox`` to see whether the production 3-state model can recover
τ_hp from the synthetic data.

**Scope** — what these tests cover and what they DON'T:

- **Open-loop**: setpoint trajectories are hand-crafted (sinusoidal swings),
  not produced by a real PI controller in closed loop. Real PI dynamics
  include limit cycles, supervisor nudges, deadband re-entries, etc., that
  may carry more or different τ_hp signal than a sinusoid does.
- **Single batch window**: each test fits ONE greybox batch (21 simulated
  days). The "many batches accumulate signal" question isn't probed here.
- **No real disturbances**: no cold snaps, no occupancy events, no manual
  setpoint changes — all the natural variation that production sees.

For the closed-loop probe-sizing question (how many active probes are
needed to tighten τ_hp), see ``scenarios/test_tau_hp_probe_sizing.py``.

**Cadence**: 3-min ticks per project standard (set in
``tests/hvac_bench/constants.py:TICK_MINUTES_DEFAULT``). Earlier draft
of these tests used 10-min ticks which is Nyquist-degenerate for the
5-min true τ_hp; corrected 2026-06-03.
"""

from __future__ import annotations

import math
import random as _random

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import Observation
from custom_components.tasmota_irhvac.pi.greybox_observer import (
    SCIPY_AVAILABLE,
    fit_greybox,
)
from tests.hvac_bench.house_profiles import PROFILES_2R2C
from tests.hvac_bench.thermal_model import ThermalModel2R2C


pytestmark = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason="scipy not installed"
)


def _generate_from_bench(
    *,
    n_days: float,
    tick_minutes: float = 3.0,
    bench_tau_hp_minutes: float = 5.0,
    bench_profile_name: str = "standard_residential_fujitsu",
    mean_outdoor_c: float = -3.0,
    sp_swing_amplitude_c: float = 1.5,
    sp_swing_period_minutes: float = 360.0,
    sensor_noise_sigma: float = 0.0,
    rng_seed: int = 42,
) -> list[Observation]:
    """Generate Observations from a bench ThermalModel2R2C run with known
    Q_hp lag and a sinusoidal setpoint swing.

    Defaults: 3-min ticks (project standard, ~1.7 samples per τ_hp = 5 min
    at the default lag), ±1.5°C 6-hr period swing.
    """
    profile = PROFILES_2R2C[bench_profile_name]
    m_total = int(n_days * 24 * 60 / tick_minutes)
    model = ThermalModel2R2C(
        profile=profile,
        initial_temp=20.0,
        outdoor_temp=mean_outdoor_c,
        sensor_noise_sigma=sensor_noise_sigma,
        tau_hp_minutes=bench_tau_hp_minutes,
        noise_seed=rng_seed,
    )

    rng_rate = _random.Random(rng_seed + 1)
    obs: list[Observation] = []
    # SP center chosen so HP has headroom most of the time even with capacity
    # derating: sp ≈ 23°C at -3°C outdoor for a 50-min τ_env building gives
    # steady-state ≈ sp.
    sp_center = 23.0
    # PRE-step recording convention: obs[i] records the state AT time
    # i·dt — BEFORE the i-th step is taken. obs[i].hp_setpoint is the
    # setpoint that will be applied over [obs[i], obs[i+1]]. This matches
    # production's residual_fn which interprets obs[i].hp_setpoint as the
    # ZOH input over the forward interval. Earlier post-step convention
    # caused a 1-tick mismatch and the resulting model-misspecification
    # error confounded all τ_hp identifiability conclusions (2026-06-03
    # debugging session — see _MODEL_INPUTS area docstring).
    last_room_rate = 0.0
    for i in range(m_total):
        minute = i * tick_minutes
        hour_of_day = (minute / 60.0) % 24.0
        t_out = mean_outdoor_c + 4.0 * math.sin(2 * math.pi * (hour_of_day - 6) / 24.0)
        solar = max(0.0, 0.5 * math.sin(2 * math.pi * (hour_of_day - 6) / 24.0))
        sp = sp_center + sp_swing_amplitude_c * math.sin(
            2 * math.pi * minute / sp_swing_period_minutes
        )
        # Record observation BEFORE stepping — current_c is the pre-step
        # T_a, hp_setpoint is what will be applied during the next dt.
        t_air_pre = model.room_temp
        hp_active = t_air_pre < sp
        clamped_reason = "" if hp_active else "no_output"
        obs.append(Observation(
            timestamp=float(i * tick_minutes * 60),
            wall_time=1713650000.0 + i * tick_minutes * 60,
            hp_setpoint=sp,
            current_c=t_air_pre,
            desired_c=sp_center,
            outdoor_temp_c=t_out,
            room_rate=last_room_rate,  # rate over previous interval
            raw_readings={"sensor.solar_proxy": solar},
            clamped=clamped_reason != "",
            clamped_reason=clamped_reason,
        ))
        # Now take the step using the recorded sp
        model.outdoor_temp = t_out
        model.step(hp_setpoint=sp, dt_minutes=tick_minutes, mode="heat",
                   tick=i, solar_proxy=solar)
        # Compute room_rate for NEXT observation from this step
        last_room_rate = (model.room_temp - t_air_pre) / tick_minutes
        last_room_rate += rng_rate.gauss(0, 0.001)
    return obs


_MODEL_INPUTS = [
    {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
]


class TestTauHpOpenLoop:
    """Open-loop τ_hp identifiability with bench-generated sinusoidal data.

    Companion tests for closed-loop / active-probe regimes live in
    ``scenarios/test_tau_hp_probe_sizing.py``.
    """

    def test_tau_hp_open_loop_sinusoidal_excitation_result(self):
        """Records what the 3-state fit recovers for τ_hp from 21 days of
        bench data with known τ_hp = 5 min and ±1.5°C 6-hr sinusoidal
        setpoint swings at 3-min ticks.

        This is a **documentary** test — it records the result rather than
        claiming a universal conclusion. If the recovered τ_hp changes
        substantially in a future run, that signals an algorithm change
        (greybox fit improvement, prior tuning, or capacity-profile
        behavior change) worth investigating.

        Earlier docstring claimed "passive ID is impossible" based on this
        kind of test — that claim was overstated (see
        [[feedback-synthetic-one-scenario]]).
        """
        obs = _generate_from_bench(
            n_days=21.0,
            tick_minutes=3.0,
            bench_tau_hp_minutes=5.0,
        )
        result = fit_greybox(
            obs, _MODEL_INPUTS,
            capacity_profile_name="fujitsu_aou24rlxfwh",
            mode="heat",
        )
        assert result is not None and result.is_2r2c
        assert result.tau_hp is not None
        # Wide observation bracket — this test is here to capture
        # behavioral changes, not enforce a tight numerical recovery.
        # If recovered τ_hp is wildly outside this range, the model has
        # changed behavior, worth investigating.
        assert 1.0 <= result.tau_hp <= 30.0, (
            f"τ_hp out of plausible range: got {result.tau_hp:.2f}"
        )

    def test_tau_hp_pulls_to_prior_when_no_transient_signal(self):
        """When the bench has tau_hp_minutes=0 (no Q_hp lag baked in) AND
        the setpoint barely varies (no excitation), the fit should pull
        τ_hp toward the prior (3 min, σ=4 min). This is the "data carries
        zero information about τ_hp" case — posterior = prior."""
        obs = _generate_from_bench(
            n_days=21.0,
            tick_minutes=3.0,
            bench_tau_hp_minutes=0.0,
            sp_swing_amplitude_c=0.1,  # Very mild excitation
        )
        result = fit_greybox(
            obs, _MODEL_INPUTS,
            capacity_profile_name="fujitsu_aou24rlxfwh",
            mode="heat",
        )
        assert result is not None and result.is_2r2c
        assert result.tau_hp is not None
        # Prior mean is 3 min, σ = 4. Without informative data, posterior
        # should stay within ~1 prior-σ of the prior — roughly [1, 7].
        assert 1.0 <= result.tau_hp <= 7.0, (
            f"τ_hp without transient signal should pull to prior 3 min, "
            f"got {result.tau_hp:.2f}"
        )

    def test_k_c_rated_stable_across_outdoor_temps_with_capacity_profile(self):
        """With a capacity profile applied, k_c_rated should be the AHRI-
        rating-point gain regardless of which outdoor-temperature regime
        the data is in. Cold-snap data and mild-winter data should both
        recover similar k_c_rated values — this is the operational win of
        the Stage 1c capacity-profile redesign over the old constant-k_c
        formulation, which would have shown systematic regime bias.
        """
        obs_cold = _generate_from_bench(
            n_days=21.0,
            tick_minutes=3.0,
            bench_tau_hp_minutes=0.0,
            mean_outdoor_c=-12.0,
        )
        obs_mild = _generate_from_bench(
            n_days=21.0,
            tick_minutes=3.0,
            bench_tau_hp_minutes=0.0,
            mean_outdoor_c=2.0,
            rng_seed=43,
        )
        r_cold = fit_greybox(
            obs_cold, _MODEL_INPUTS,
            capacity_profile_name="fujitsu_aou24rlxfwh",
            mode="heat",
        )
        r_mild = fit_greybox(
            obs_mild, _MODEL_INPUTS,
            capacity_profile_name="fujitsu_aou24rlxfwh",
            mode="heat",
        )
        assert r_cold is not None and r_cold.is_2r2c
        assert r_mild is not None and r_mild.is_2r2c
        # k_c_rated should agree between the two batches within 50% relative
        # — they're identifying the SAME underlying rating-point gain
        # despite operating at very different outdoor temps. Not asking for
        # tight numerical convergence (passive data has lots of variance);
        # asking for no SYSTEMATIC bias from outdoor temp.
        ratio = r_cold.k_c / r_mild.k_c if r_mild.k_c > 0 else float("inf")
        assert 0.5 <= ratio <= 2.0, (
            f"k_c_rated should be regime-invariant: cold={r_cold.k_c:.4f} "
            f"mild={r_mild.k_c:.4f} ratio={ratio:.2f}"
        )
