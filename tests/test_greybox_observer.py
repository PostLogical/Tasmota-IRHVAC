"""Tests for grey-box 1R1C energy balance observer.

Parameterization follows Bacher & Madsen (2011): rate coefficients
ua_c = UA/C, k_c = K_hp/C, α_c = α_solar/C are directly identifiable
from derivative (room_rate) data.  τ_eff = 1/ua_c.
"""

import math

import pytest

from custom_components.tasmota_irhvac.pi.greybox_observer import (
    SCIPY_AVAILABLE,
    GreyboxResult,
    _extract_solar_index,
    fit_greybox,
    log_greybox_result,
)
from custom_components.tasmota_irhvac.pi.batch_learning import Observation


pytestmark = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason="scipy not installed"
)


class TestExtractSolarIndex:
    def test_finds_solar_input(self):
        inputs = [
            {"name": "stove", "input_role": "heat_source"},
            {"name": "solar", "input_role": "solar"},
        ]
        assert _extract_solar_index(inputs) == 3  # offset by 2

    def test_no_solar_returns_none(self):
        inputs = [{"name": "stove", "input_role": "heat_source"}]
        assert _extract_solar_index(inputs) is None

    def test_empty_inputs(self):
        assert _extract_solar_index([]) is None


class TestFitGreybox:
    """Test 1R1C energy balance fitting on synthetic data.

    True model (rate-coefficient form):
        room_rate = ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar

    With ua_c=0.006 (τ=167 min), k_c=0.02, α_c=-0.04.
    These correspond to UA=0.3, K_hp=1.0, α_solar=-2.0, C_eff=50.0.
    """

    # True rate coefficients: UA/C=0.3/50, K/C=1/50, α/C=-2/50
    UA_C_TRUE = 0.006
    K_C_TRUE = 0.02
    ALPHA_C_TRUE = -0.04

    def _generate_observations(
        self,
        n: int = 200,
        include_hp_off: bool = True,
        noise_std: float = 0.0005,
    ) -> list[Observation]:
        """Generate synthetic observations from a known 1R1C model."""
        import random
        random.seed(42)
        obs = []
        for i in range(n):
            hour = (i / n) * 24.0
            # Outdoor: 0°C at night, 10°C midday
            t_out = 5.0 + 5.0 * math.sin(2 * math.pi * (hour - 6) / 24)
            # Solar: 0 at night, peaks at noon
            solar = max(0.0, 0.6 * math.sin(2 * math.pi * (hour - 6) / 24))
            # Room temp: ~20°C
            t_air = 20.0 + random.gauss(0, 0.1)

            hp_on = t_out < 8.0 or solar < 0.1
            if include_hp_off and not hp_on:
                hp_setpoint = t_air - 1.0
                hp_offset = 0.0
                clamped_reason = "no_output"
            else:
                hp_offset = 2.0
                hp_setpoint = t_air + hp_offset
                clamped_reason = ""

            room_rate = (
                self.UA_C_TRUE * (t_out - t_air)
                + self.K_C_TRUE * hp_offset
                + self.ALPHA_C_TRUE * solar
                + random.gauss(0, noise_std)
            )

            features = [1.0, max(0, 20.0 - t_out), solar]
            obs.append(Observation(
                timestamp=float(i * 60),
                features=features,
                hp_setpoint=hp_setpoint,
                current_c=t_air,
                desired_c=20.0,
                room_rate=room_rate,
                clamped=clamped_reason != "",
                clamped_reason=clamped_reason,
                outdoor_temp_c=t_out,
            ))
        return obs

    def test_recovers_known_parameters(self):
        """Fit should recover ua_c, k_c, α_c from clean synthetic data."""
        obs = self._generate_observations(n=500, noise_std=0.0005)
        model_inputs = [{"name": "solar", "input_role": "solar"}]

        result = fit_greybox(obs, model_inputs)
        assert result is not None
        # Rate coefficients should be close to true values.
        # ua_c=0.006 → τ≈167 min
        assert abs(result.ua_c - self.UA_C_TRUE) < 0.002
        assert abs(result.k_c - self.K_C_TRUE) < 0.005
        assert abs(result.alpha_c - self.ALPHA_C_TRUE) < 0.01
        assert abs(result.tau_eff - 1.0 / self.UA_C_TRUE) < 60.0

    def test_uses_hp_off_data(self):
        """HP-off observations should contribute to ua_c and α_c estimation."""
        obs = self._generate_observations(n=500, include_hp_off=True)
        model_inputs = [{"name": "solar", "input_role": "solar"}]
        result = fit_greybox(obs, model_inputs)
        assert result is not None
        assert result.n_hp_off > 0
        assert result.n_hp_on > 0

    def test_no_solar_input(self):
        """Works without a solar proxy -- fits ua_c, k_c only."""
        obs = self._generate_observations(n=300, noise_std=0.001)
        # Remove solar from features
        for o in obs:
            o.features[:] = o.features[:2]

        result = fit_greybox(obs, model_inputs=[])
        assert result is not None
        assert result.alpha_c == 0.0  # no solar input → 0

    def test_insufficient_observations(self):
        """Returns None with too few observations."""
        obs = self._generate_observations(n=10)
        result = fit_greybox(obs, model_inputs=[])
        assert result is None

    def test_no_outdoor_temp(self):
        """Observations without outdoor_temp_c are filtered out."""
        obs = self._generate_observations(n=50)
        for o in obs:
            o.outdoor_temp_c = None  # type: ignore[assignment]
        result = fit_greybox(obs, model_inputs=[])
        assert result is None

    def test_plant_id_cross_check(self):
        """τ_eff should be compared against plant ID τ_slow."""
        obs = self._generate_observations(n=500, noise_std=0.001)
        model_inputs = [{"name": "solar", "input_role": "solar"}]
        # True τ = 1/0.006 ≈ 167 min
        result = fit_greybox(
            obs, model_inputs,
            plant_tau_slow=170.0, plant_tau_slow_confidence=0.5,
        )
        assert result is not None
        assert result.plant_tau_slow == 170.0
        assert result.tau_agreement_pct is not None
        assert result.tau_agreement_pct < 50.0

    def test_hp_all_off_fixes_k_c(self):
        """When HP is off for entire buffer, k_c should be fixed."""
        obs = self._generate_observations(n=200, noise_std=0.001)
        for o in obs:
            o.hp_setpoint = o.current_c - 1.0
            o.clamped = True
            o.clamped_reason = "no_output"

        result = fit_greybox(obs, model_inputs=[])
        assert result is not None
        # k_c should be at the fixed default, not a fitted value
        assert result.k_c == 0.01

    def test_result_round_trip(self):
        """GreyboxResult.as_dict produces serializable output."""
        result = GreyboxResult(
            n_observations=100, n_hp_on=60, n_hp_off=40,
            ua_c=0.006, k_c=0.02, alpha_c=-0.04,
            tau_eff=166.7, residual_rms=0.005,
            cost=0.5, n_function_evals=10,
            plant_tau_slow=170.0, tau_agreement_pct=2.0,
            param_std_err={"ua_c": 0.001, "k_c": 0.003},
        )
        d = result.as_dict()
        assert d["ua_c"] == 0.006
        assert d["plant_tau_slow"] == 170.0
        assert d["param_std_err"]["ua_c"] == 0.001

    def test_log_output(self, caplog):
        """log_greybox_result should log without errors."""
        result = GreyboxResult(
            n_observations=100, n_hp_on=60, n_hp_off=40,
            ua_c=0.006, k_c=0.02, alpha_c=-0.04,
            tau_eff=166.7, residual_rms=0.005,
            cost=0.5, n_function_evals=10,
            plant_tau_slow=170.0, tau_agreement_pct=2.0,
            param_std_err={"ua_c": 0.001},
        )
        import logging
        with caplog.at_level(logging.INFO):
            log_greybox_result(result, log_prefix="[test] ")
        assert "Grey-box 1R1C" in caplog.text
        assert "ua_c=0.00600" in caplog.text
        assert "Plant ID cross-check" in caplog.text
