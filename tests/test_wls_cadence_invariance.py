"""Test WLS std_err is cadence-invariant under autocorrelated residuals.

Demonstrates the foundational assumption error behind #100: classical
IID std_err treats N (sample count) as N (information count).  At fine
cadence the same wall-clock data produces shrunken std_err — wrongly —
because consecutive samples are autocorrelated.

Currently xfail: HAC + ESS-BIC (#100 Step 2) closes the cadence gap
substantially (ratio 0.42 → 0.63) but doesn't fully reach the [0.7, 1.5]
band.  Full closure requires either (a) buffer-wall-clock-window scaling
(see plan), or (b) stronger HAC kernel (Quadratic Spectral / pre-
whitening), or (c) Reynders pre-processing.  See
`~/.claude/plans/wls_cadence_corrections.md` for the followup work.
This test is sized for the eventual fix per
`feedback_xfail_for_future_fix.md`: when the gap closes, it should pass.
"""
from __future__ import annotations

import math
import random

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import (
    Observation,
    weighted_least_squares,
)


WINDOW_MIN = 24 * 60   # 24 h wall-clock window
TRUE_BETA = 2.0        # outdoor_delta coefficient
SIGMA_NOISE = 1.0      # residual stationary std
T_CORR_MIN = 30.0      # residual autocorrelation time (continuous-time)


def _generate_observations(dt_min: float, seed: int = 42) -> list[Observation]:
    """Synthetic observations at the specified cadence over WINDOW_MIN.

    Underlying continuous-time process:
    - outdoor_delta(t) = 5·sin(2π·t/1440)  (diurnal)
    - true offset(t) = TRUE_BETA · outdoor_delta(t)
    - observed offset = true + AR(1) noise with T_CORR_MIN correlation time
    - hp_setpoint = current_c + observed_offset
    """
    rng = random.Random(seed)
    rho = math.exp(-dt_min / T_CORR_MIN)
    sigma_step = SIGMA_NOISE * math.sqrt(max(0.0, 1.0 - rho * rho))

    n = int(WINDOW_MIN / dt_min)
    eps = 0.0
    obs_list: list[Observation] = []
    base_wall_time = 1.0e9  # arbitrary epoch
    for i in range(n):
        t_min = i * dt_min
        outdoor_delta = 5.0 * math.sin(2 * math.pi * t_min / 1440.0)
        true_offset = TRUE_BETA * outdoor_delta
        eps = rho * eps + sigma_step * rng.gauss(0.0, 1.0)
        observed_offset = true_offset + eps

        cur = 20.0
        sp = cur + observed_offset
        outdoor_temp_c = cur + outdoor_delta

        obs_list.append(Observation(
            timestamp=float(i),
            wall_time=base_wall_time + t_min * 60.0,
            hp_setpoint=sp,
            current_c=cur,
            desired_c=20.0,
            outdoor_temp_c=outdoor_temp_c,
            room_rate=0.005,  # below room_rate_threshold (0.02)
            raw_readings={},
            clamped=False,
        ))
    return obs_list


class TestCadenceInvariance:
    """std_err for the same wall-clock window must not depend on sampling cadence.

    This is the unit-test analog of #100's freeze pathology: classical
    IID std_err shrinks like 1/√N regardless of whether new samples carry
    independent information. With autocorrelated residuals (T_corr=30 min,
    realistic for thermal data), 3-min and 15-min cadences see the same
    physical noise process — std_err should agree.
    """

    @pytest.mark.xfail(
        reason=(
            "HAC + ESS-BIC closes ratio 0.42 → 0.63 (substantial); full closure "
            "to [0.7, 1.5] needs buffer-wall-clock-window scaling or stronger "
            "HAC kernel.  See ~/.claude/plans/wls_cadence_corrections.md."
        ),
        strict=False,
    )
    def test_std_err_cadence_invariant(self):
        obs_coarse = _generate_observations(dt_min=15.0)
        obs_fine = _generate_observations(dt_min=3.0)

        # Sanity: fine has 5× the samples
        assert len(obs_fine) == 5 * len(obs_coarse)

        result_coarse = weighted_least_squares(
            obs_coarse, n_features=2, min_observations=20,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=[],
        )
        result_fine = weighted_least_squares(
            obs_fine, n_features=2, min_observations=20,
            feature_order=["intercept", "outdoor_delta"],
            model_inputs=[],
        )
        assert result_coarse is not None
        assert result_fine is not None

        # Point estimate sanity (both should recover TRUE_BETA roughly)
        assert abs(result_coarse.beta_batch[1] - TRUE_BETA) < 0.3
        assert abs(result_fine.beta_batch[1] - TRUE_BETA) < 0.3

        se_coarse = result_coarse.beta_std_err[1]
        se_fine = result_fine.beta_std_err[1]
        ratio = se_fine / se_coarse

        # Before HAC: ratio ≈ 1/√5 ≈ 0.45 (bug — shrunken std_err at fine cadence).
        # After HAC: ratio ≈ 1.0 (cadence-invariant — same information per
        # wall-clock window regardless of sampling rate).
        assert 0.7 < ratio < 1.5, (
            f"std_err not cadence-invariant under autocorrelated residuals: "
            f"coarse(Δt=15min)={se_coarse:.4f} "
            f"fine(Δt=3min)={se_fine:.4f} "
            f"ratio={ratio:.2f} (expected ~1.0; classical IID gives ~0.45)"
        )
