"""Tests for grey-box 1R1C energy balance observer.

Parameterization follows Bacher & Madsen (2011): rate coefficients
ua_c = UA/C, k_c = K_hp/C, α_c = α_solar/C are directly identifiable
from derivative (room_rate) data.  τ_eff = 1/ua_c.
"""

import math

import pytest

from custom_components.tasmota_irhvac.pi.greybox_observer import (
    GATE_MAX_CV,
    GATE_MAX_RMS,
    GATE_MAX_TAU,
    GATE_MIN_TAU,
    SCIPY_AVAILABLE,
    GreyboxBridgeResult,
    GreyboxResult,
    _check_quality_gates,
    _delta_method_ratio_std,
    find_solar_entity,
    fit_greybox,
    greybox_to_beta,
    log_greybox_result,
)
from custom_components.tasmota_irhvac.pi.batch_learning import Observation


pytestmark = pytest.mark.skipif(
    not SCIPY_AVAILABLE, reason="scipy not installed"
)


class TestFindSolarEntity:
    def test_finds_solar_input(self):
        inputs = [
            {"name": "stove", "entity_id": "sensor.stove", "input_role": "heat_source"},
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        assert find_solar_entity(inputs) == "sensor.solar_proxy"

    def test_no_solar_returns_none(self):
        inputs = [{"name": "stove", "entity_id": "sensor.stove", "input_role": "heat_source"}]
        assert find_solar_entity(inputs) is None

    def test_empty_inputs(self):
        assert find_solar_entity([]) is None


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

            obs.append(Observation(
                timestamp=float(i * 60),
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=hp_setpoint,
                current_c=t_air,
                desired_c=20.0,
                outdoor_temp_c=t_out,
                room_rate=room_rate,
                raw_readings={"sensor.solar_proxy": solar},
                clamped=clamped_reason != "",
                clamped_reason=clamped_reason,
            ))
        return obs

    def test_recovers_known_parameters(self):
        """Fit should recover ua_c, k_c, α_c from clean synthetic data."""
        obs = self._generate_observations(n=500, noise_std=0.0005)
        model_inputs = [{"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"}]

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
        model_inputs = [{"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"}]
        result = fit_greybox(obs, model_inputs)
        assert result is not None
        assert result.n_hp_off > 0
        assert result.n_hp_on > 0

    def test_no_solar_input(self):
        """Works without a solar proxy -- fits ua_c, k_c only."""
        obs = self._generate_observations(n=300, noise_std=0.001)
        # Remove solar from raw_readings
        for o in obs:
            o.raw_readings.pop("sensor.solar_proxy", None)

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
        model_inputs = [{"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"}]
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


# ── Delta method ─────────────────────────────────────────────────────

class TestDeltaMethodRatio:
    """Test σ_{a/b} propagation via delta method."""

    def test_known_values(self):
        """σ_{a/b} = |a/b| × √(σ_a²/a² + σ_b²/b²)."""
        # a=6, b=3, σ_a=1, σ_b=0.5 → ratio=2, CV_a=1/6, CV_b=0.5/3=1/6
        # σ_ratio = 2 × √(1/36 + 1/36) = 2 × √(2/36) ≈ 0.471
        se = _delta_method_ratio_std(6.0, 3.0, 1.0, 0.5)
        expected = 2.0 * math.sqrt(2 / 36)
        assert abs(se - expected) < 1e-10

    def test_zero_denominator(self):
        """Zero b → infinite std_err."""
        se = _delta_method_ratio_std(1.0, 0.0, 0.1, 0.1)
        assert math.isinf(se)

    def test_zero_numerator(self):
        """Zero a → infinite std_err (ratio is 0, CV undefined)."""
        se = _delta_method_ratio_std(0.0, 1.0, 0.1, 0.1)
        assert math.isinf(se)

    def test_no_uncertainty(self):
        """Zero σ → zero propagated error."""
        se = _delta_method_ratio_std(6.0, 3.0, 0.0, 0.0)
        assert se == 0.0


# ── Quality gates ────────────────────────────────────────────────────

def _make_result(**overrides) -> GreyboxResult:
    """Build a GreyboxResult with sensible defaults, overridable."""
    defaults = dict(
        n_observations=200, n_hp_on=120, n_hp_off=80,
        ua_c=0.006, k_c=0.02, alpha_c=-0.04,
        tau_eff=166.7, residual_rms=0.005,
        cost=0.5, n_function_evals=10,
        param_std_err={"ua_c": 0.001, "k_c": 0.003, "alpha_c": 0.005},
    )
    defaults.update(overrides)
    return GreyboxResult(**defaults)


class TestQualityGates:
    """Test individual quality gate logic."""

    def test_all_pass_with_good_data(self):
        result = _make_result()
        gates = _check_quality_gates(result)
        assert all(gates.values()), f"Failed gates: {[k for k, v in gates.items() if not v]}"

    def test_hp_off_diversity_fails_no_off(self):
        """All HP-on → can't separate k_c from ua_c."""
        result = _make_result(n_hp_on=200, n_hp_off=0)
        gates = _check_quality_gates(result)
        assert not gates["hp_offset_diversity"]

    def test_hp_off_diversity_fails_no_on(self):
        """All HP-off → can't identify k_c at all."""
        result = _make_result(n_hp_on=0, n_hp_off=200)
        gates = _check_quality_gates(result)
        assert not gates["hp_offset_diversity"]

    def test_hp_off_diversity_marginal(self):
        """Exactly 10% HP-off should pass."""
        result = _make_result(n_hp_on=180, n_hp_off=20)
        gates = _check_quality_gates(result)
        assert gates["hp_offset_diversity"]

    def test_param_precision_fails_high_cv(self):
        """High CV → uncertain parameter → gate fails."""
        # σ_ua_c = 0.005 vs ua_c = 0.006 → CV = 0.83 > 0.5
        result = _make_result(param_std_err={"ua_c": 0.005, "k_c": 0.003, "alpha_c": 0.005})
        gates = _check_quality_gates(result)
        assert not gates["param_precision_ua_c"]

    def test_tau_plausibility_too_fast(self):
        """τ < 30 min → unrealistic building."""
        result = _make_result(ua_c=0.05, tau_eff=20.0)
        gates = _check_quality_gates(result)
        assert not gates["tau_plausible"]

    def test_tau_plausibility_too_slow(self):
        """τ > 500 min → parameter at bound."""
        result = _make_result(ua_c=0.0015, tau_eff=667.0)
        gates = _check_quality_gates(result)
        assert not gates["tau_plausible"]

    def test_k_c_negative_fails(self):
        """Negative k_c is physically impossible (HP heats, doesn't cool)."""
        result = _make_result(k_c=-0.01)
        gates = _check_quality_gates(result)
        assert not gates["k_c_positive"]

    def test_alpha_c_positive_fails(self):
        """Positive α_c means solar cools the building — wrong sign."""
        result = _make_result(
            alpha_c=0.01,
            param_std_err={"ua_c": 0.001, "k_c": 0.003, "alpha_c": 0.005},
        )
        gates = _check_quality_gates(result)
        assert not gates["alpha_c_nonpositive"]

    def test_alpha_c_gate_skipped_without_solar(self):
        """No solar fit → α_c gate not checked."""
        result = _make_result(
            alpha_c=0.0,
            param_std_err={"ua_c": 0.001, "k_c": 0.003},
        )
        gates = _check_quality_gates(result)
        assert "alpha_c_nonpositive" not in gates or gates["alpha_c_nonpositive"]

    def test_residual_rms_fails(self):
        """High residual → poor model fit."""
        result = _make_result(residual_rms=0.05)
        gates = _check_quality_gates(result)
        assert not gates["residual_rms"]


# ── Steady-state bridge ─────────────────────────────────────────────

class TestGreyboxToBeta:
    """Test conversion of rate coefficients to WLS β."""

    SOLAR_INPUT = {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"}
    STOVE_INPUT = {"name": "Pellet Stove", "entity_id": "sensor.stove_temp", "input_role": "heat_source"}

    def test_outdoor_delta_mapping(self):
        """β₁ = -ua_c/k_c."""
        result = _make_result(ua_c=0.006, k_c=0.02)
        bridge = greybox_to_beta(result, model_inputs=[])
        # β₁ = -0.006/0.02 = -0.3
        assert bridge.beta[1] is not None
        assert abs(bridge.beta[1] - (-0.3)) < 1e-10

    def test_solar_mapping(self):
        """β for solar input = -α_c/k_c."""
        result = _make_result(ua_c=0.006, k_c=0.02, alpha_c=-0.04)
        bridge = greybox_to_beta(result, model_inputs=[self.SOLAR_INPUT])
        # β₂ = -(-0.04)/0.02 = 2.0
        assert bridge.beta[2] is not None
        assert abs(bridge.beta[2] - 2.0) < 1e-10

    def test_non_solar_input_gets_none(self):
        """Non-solar model inputs get None (grey-box doesn't identify them)."""
        result = _make_result()
        bridge = greybox_to_beta(result, model_inputs=[self.STOVE_INPUT])
        assert bridge.beta[2] is None
        assert math.isinf(bridge.beta_std_err[2])

    def test_intercept_is_none(self):
        """β₀ (intercept) is always None from grey-box."""
        result = _make_result()
        bridge = greybox_to_beta(result, model_inputs=[])
        assert bridge.beta[0] is None

    def test_std_err_propagation(self):
        """β std_err should be finite when rate coefficients have finite std_err."""
        result = _make_result(
            ua_c=0.006, k_c=0.02,
            param_std_err={"ua_c": 0.001, "k_c": 0.003, "alpha_c": 0.005},
        )
        bridge = greybox_to_beta(result, model_inputs=[self.SOLAR_INPUT])
        assert not math.isinf(bridge.beta_std_err[1])  # outdoor_delta
        assert not math.isinf(bridge.beta_std_err[2])  # solar
        assert bridge.beta_std_err[1] > 0
        assert bridge.beta_std_err[2] > 0

    def test_std_err_propagation_matches_delta_method(self):
        """β₁ std_err should match hand-computed delta method value."""
        result = _make_result(
            ua_c=0.006, k_c=0.02,
            param_std_err={"ua_c": 0.001, "k_c": 0.003},
        )
        bridge = greybox_to_beta(result, model_inputs=[])
        expected = _delta_method_ratio_std(0.006, 0.02, 0.001, 0.003)
        assert abs(bridge.beta_std_err[1] - expected) < 1e-10

    def test_k_eff_computed(self):
        """K_eff = k_c / ua_c."""
        result = _make_result(ua_c=0.006, k_c=0.02)
        bridge = greybox_to_beta(result, model_inputs=[])
        assert abs(bridge.k_eff - 0.02 / 0.006) < 1e-6

    def test_tau_eff_passthrough(self):
        """τ_eff from grey-box is passed through."""
        result = _make_result(tau_eff=166.7)
        bridge = greybox_to_beta(result, model_inputs=[])
        assert bridge.tau_eff == 166.7

    def test_gates_passed_with_good_data(self):
        """All gates should pass with good synthetic data."""
        result = _make_result()
        bridge = greybox_to_beta(result, model_inputs=[self.SOLAR_INPUT])
        assert bridge.gates_passed

    def test_gates_failed_propagates(self):
        """Failed gates → gates_passed=False."""
        result = _make_result(n_hp_on=200, n_hp_off=0)
        bridge = greybox_to_beta(result, model_inputs=[])
        assert not bridge.gates_passed
        assert not bridge.gate_details["hp_offset_diversity"]

    def test_multiple_model_inputs_only_solar_mapped(self):
        """With mixed inputs, only solar role gets a β value."""
        result = _make_result(alpha_c=-0.04)
        bridge = greybox_to_beta(
            result,
            model_inputs=[self.STOVE_INPUT, self.SOLAR_INPUT],
        )
        # stove (index 2) → None, solar (index 3) → mapped
        assert bridge.beta[2] is None
        assert bridge.beta[3] is not None
        assert abs(bridge.beta[3] - (-(-0.04) / 0.02)) < 1e-10

    def test_as_dict_serializable(self):
        """Bridge result should be JSON-serializable."""
        import json
        result = _make_result()
        bridge = greybox_to_beta(result, model_inputs=[self.SOLAR_INPUT])
        d = bridge.as_dict()
        json.dumps(d)  # should not raise

    def test_logging_output(self, caplog):
        """Bridge should log its results."""
        import logging
        result = _make_result()
        with caplog.at_level(logging.INFO):
            greybox_to_beta(
                result,
                model_inputs=[self.SOLAR_INPUT],
                log_prefix="[test] ",
            )
        assert "Grey-box bridge" in caplog.text
        assert "Quality gates" in caplog.text


class TestGreyboxBridgeEndToEnd:
    """End-to-end: fit grey-box on synthetic data, then bridge to β."""

    UA_C_TRUE = 0.006
    K_C_TRUE = 0.02
    ALPHA_C_TRUE = -0.04

    def _generate_observations(self, n: int = 500) -> list[Observation]:
        """Same synthetic data as TestFitGreybox."""
        import random
        random.seed(42)
        obs = []
        for i in range(n):
            hour = (i / n) * 24.0
            t_out = 5.0 + 5.0 * math.sin(2 * math.pi * (hour - 6) / 24)
            solar = max(0.0, 0.6 * math.sin(2 * math.pi * (hour - 6) / 24))
            t_air = 20.0 + random.gauss(0, 0.1)

            hp_on = t_out < 8.0 or solar < 0.1
            if not hp_on:
                hp_offset = 0.0
                hp_setpoint = t_air - 1.0
                clamped_reason = "no_output"
            else:
                hp_offset = 2.0
                hp_setpoint = t_air + hp_offset
                clamped_reason = ""

            room_rate = (
                self.UA_C_TRUE * (t_out - t_air)
                + self.K_C_TRUE * hp_offset
                + self.ALPHA_C_TRUE * solar
                + random.gauss(0, 0.0005)
            )
            obs.append(Observation(
                timestamp=float(i * 60),
                wall_time=1713650000.0 + i * 60,
                hp_setpoint=hp_setpoint,
                current_c=t_air,
                desired_c=20.0,
                outdoor_temp_c=t_out,
                room_rate=room_rate,
                raw_readings={"sensor.solar_proxy": solar},
                clamped=clamped_reason != "",
                clamped_reason=clamped_reason,
            ))
        return obs

    def test_bridged_beta_matches_physics(self):
        """β from bridge should match the true steady-state gains.

        True β₁ = -UA_C/K_C = -0.006/0.02 = -0.3
        True β₂ = -ALPHA_C/K_C = 0.04/0.02 = 2.0
        """
        obs = self._generate_observations()
        model_inputs = [{"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"}]
        result = fit_greybox(obs, model_inputs)
        assert result is not None

        bridge = greybox_to_beta(result, model_inputs)
        assert bridge.gates_passed

        # β₁ (outdoor_delta) ≈ -0.3
        assert bridge.beta[1] is not None
        assert abs(bridge.beta[1] - (-0.3)) < 0.1

        # β₂ (solar) ≈ 2.0
        assert bridge.beta[2] is not None
        assert abs(bridge.beta[2] - 2.0) < 0.5

        # Std errors should be finite and reasonable
        assert bridge.beta_std_err[1] < 0.5
        assert bridge.beta_std_err[2] < 1.0
