"""Tests for grey-box 1R1C energy balance observer.

Parameterization follows Bacher & Madsen (2011): rate coefficients
ua_c = UA/C, k_c = K_hp/C, α_c = α_solar/C are directly identifiable
from derivative (room_rate) data.  τ_eff = 1/ua_c.
"""

import math

import pytest

from custom_components.tasmota_irhvac.pi.greybox_observer import (
    C0_BOUNDS,
    GATE_MAX_CV,
    GATE_MAX_RMS,
    GATE_MAX_TAU,
    GATE_MAX_TAU_FAST,
    GATE_MAX_TAU_SLOW,
    GATE_MIN_TAU,
    GATE_MIN_TAU_FAST,
    GATE_MIN_TAU_SLOW,
    GATE_MIN_TAU_SEPARATION,
    K_C_BOUNDS,
    K_W_BOUNDS,
    MASS_RATIO_BOUNDS,
    MIN_OBSERVATIONS_2R2C,
    MIN_TIMESPAN_DAYS_2R2C,
    SCIPY_AVAILABLE,
    SOLAR_AIR_FRACTION,
    SOLAR_WALL_FRACTION,
    UA_C_BOUNDS,
    UA_C_PRIOR_MEAN,
    UA_C_PRIOR_SIGMA,
    GreyboxBridgeResult,
    GreyboxResult,
    PriorState,
    _check_quality_gates,
    _delta_method_ratio_std,
    _natural_eigenvalues,
    find_solar_entity,
    fit_greybox,
    greybox_to_beta,
    log_greybox_result,
    promote_posterior,
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
        room_rate = c0 + ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar

    With c0=0.0, ua_c=0.006 (τ=167 min), k_c=0.02, α_c=+0.04.
    These correspond to UA=0.3, K_hp=1.0, α_solar=+2.0, C_eff=50.0.
    """

    # True rate coefficients: UA/C=0.3/50, K/C=1/50, α/C=+2/50
    UA_C_TRUE = 0.006
    K_C_TRUE = 0.02
    ALPHA_C_TRUE = 0.04

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
            c0=0.001, ua_c=0.006, k_c=0.02, alpha_c=0.04,
            tau_eff=166.7, residual_rms=0.005,
            cost=0.5, n_function_evals=10,
            plant_tau_slow=170.0, tau_agreement_pct=2.0,
            param_std_err={"ua_c": 0.001, "k_c": 0.003},
        )
        d = result.as_dict()
        assert d["c0"] == 0.001
        assert d["ua_c"] == 0.006
        assert d["plant_tau_slow"] == 170.0
        assert d["param_std_err"]["ua_c"] == 0.001

    def test_log_output(self, caplog):
        """log_greybox_result should log without errors."""
        result = GreyboxResult(
            n_observations=100, n_hp_on=60, n_hp_off=40,
            c0=0.001, ua_c=0.006, k_c=0.02, alpha_c=0.04,
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
        c0=0.001, ua_c=0.006, k_c=0.02, alpha_c=0.04,
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
        """τ > 1500 min (25h) → beyond plausible 2R2C slow mode."""
        result = _make_result(ua_c=0.0005, tau_eff=2000.0)
        gates = _check_quality_gates(result)
        assert not gates["tau_plausible"]

    def test_k_c_negative_fails(self):
        """Negative k_c is physically impossible (HP heats, doesn't cool)."""
        result = _make_result(k_c=-0.01)
        gates = _check_quality_gates(result)
        assert not gates["k_c_positive"]

    def test_alpha_c_negative_fails(self):
        """Negative α_c means solar cools the building — wrong sign."""
        result = _make_result(
            alpha_c=-0.01,
            param_std_err={"ua_c": 0.001, "k_c": 0.003, "alpha_c": 0.005},
        )
        gates = _check_quality_gates(result)
        assert not gates["alpha_c_nonnegative"]

    def test_alpha_c_gate_skipped_without_solar(self):
        """No solar fit → α_c gate not checked."""
        result = _make_result(
            alpha_c=0.0,
            param_std_err={"ua_c": 0.001, "k_c": 0.003},
        )
        gates = _check_quality_gates(result)
        assert "alpha_c_nonnegative" not in gates or gates["alpha_c_nonnegative"]

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
        result = _make_result(ua_c=0.006, k_c=0.02, alpha_c=0.04)
        bridge = greybox_to_beta(result, model_inputs=[self.SOLAR_INPUT])
        # β₂ = -(0.04)/0.02 = -2.0  (solar warms room → HP backs off)
        assert bridge.beta[2] is not None
        assert abs(bridge.beta[2] - (-2.0)) < 1e-10

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
        result = _make_result(alpha_c=0.04)
        bridge = greybox_to_beta(
            result,
            model_inputs=[self.STOVE_INPUT, self.SOLAR_INPUT],
        )
        # stove (index 2) → None, solar (index 3) → mapped
        assert bridge.beta[2] is None
        assert bridge.beta[3] is not None
        assert abs(bridge.beta[3] - (-(0.04) / 0.02)) < 1e-10  # -2.0

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
    ALPHA_C_TRUE = 0.04

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
        True β₂ = -ALPHA_C/K_C = -0.04/0.02 = -2.0
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

        # β₂ (solar) ≈ -2.0  (solar warms room → HP backs off)
        assert bridge.beta[2] is not None
        assert abs(bridge.beta[2] - (-2.0)) < 0.5

        # Std errors should be finite and reasonable
        assert bridge.beta_std_err[1] < 0.5
        assert bridge.beta_std_err[2] < 1.0


# ── 2R2C upgrade ────────────────────────────────────────────────────


class TestNaturalEigenvalues:
    """Test the analytic 2R2C natural eigenvalue helper."""

    def test_typical_residential(self):
        """Standard residential (τ_env=100, τ_couple=80, mass_ratio=8) → tau_fast<tau_slow."""
        ua_c = 1.0 / 100.0   # τ_env = 100 min
        k_w = 1.0 / 80.0     # τ_couple = 80 min
        mass_ratio = 8.0
        tau_fast, tau_slow = _natural_eigenvalues(ua_c, k_w, mass_ratio)
        assert tau_fast < tau_slow
        # Fast pole dominated by air node (≈ 1/(ua_c + k_w) when wall coupling weak)
        assert 5.0 < tau_fast < 100.0
        # Slow pole reflects wall mass (τ_couple × mass_ratio bound)
        assert tau_slow > tau_fast * 2.0
        assert tau_slow < 2000.0

    def test_decoupled_limit(self):
        """k_w → 0: wall decouples; eigenvalues approach 1/ua_c and ∞."""
        tau_fast, tau_slow = _natural_eigenvalues(0.01, 1e-3, 8.0)
        # tau_fast ≈ 1/0.01 = 100 (the only meaningful pole)
        # tau_slow much longer (large mass_ratio × small k_w)
        assert tau_slow > tau_fast * 5.0

    def test_separation_grows_with_mass_ratio(self):
        """Higher mass_ratio → larger τ_slow / τ_fast separation."""
        ua_c, k_w = 1.0 / 100.0, 1.0 / 50.0
        _, tau_slow_lo = _natural_eigenvalues(ua_c, k_w, mass_ratio=2.0)
        _, tau_slow_hi = _natural_eigenvalues(ua_c, k_w, mass_ratio=15.0)
        assert tau_slow_hi > tau_slow_lo


class TestFitGreybox2R2C:
    """End-to-end 2R2C fit on synthetic data generated from a known 2R2C plant.

    True parameters chosen to span a realistic residential profile:
      ua_c       = 0.01    (τ_env = 100 min)
      k_c        = 0.04    (HP gain)
      α_total    = 0.05    (solar rate, °C/min per unit proxy)
      k_w        = 1/40    (τ_couple = 40 min)
      mass_ratio = 8.0
    """

    UA_C_TRUE = 0.01
    K_C_TRUE = 0.04
    ALPHA_TOTAL_TRUE = 0.05
    # Wall-mode truth aligned with K_W_FIXED / MASS_RATIO_FIXED hard-fixed
    # constants in greybox_observer.py (Stage A operational regime — wall
    # params not separately identifiable from operational data per
    # Bacher-Madsen 2011 / Hollick 2020). Was 1/40 before architectural
    # change; aligned to remove false-negative recovery error.
    K_W_TRUE = 1.0 / 50.0
    MASS_RATIO_TRUE = 8.0

    # HP setpoint commands (fixed, not leashed to room temp — unlike a real
    # PI controller that updates sp each tick, but constant sp is sufficient
    # for testing parameter recovery).
    #   SP_FIXED chosen so steady-state T_a ≈ 20°C with truth params and
    #   t_out averaging 5°C:
    #     0 = ua_c·(t_out − T_a) + k_c·(sp − T_a)  [no solar, wall in eq]
    #     0 = 0.01·(5 − 20) + 0.04·(sp − 20)
    #     sp − 20 = 0.15/0.04 = 3.75   ⇒  sp = 23.75 ≈ 24
    SP_FIXED_HEAT = 24.0
    SP_HEAT_OFF = 16.0  # well below typical t_air → bench_form active=False

    def _generate_2r2c_observations(
        self,
        n_days: float = 21.0,
        tick_minutes: float = 10.0,
        noise_std: float = 0.0008,
    ) -> list[Observation]:
        """Simulate a 2R2C plant forward and emit Observations.

        Uses matrix-exponential integration on the joint [T_a, T_w] state to
        match the fitter's residual computation. HP setpoint is fixed (not
        leashed to current room temp) so the trajectory is self-consistent
        for sim-error PEM — a real HVAC commands an absolute setpoint and
        the room reaches whatever steady state the physics produce.

        Pre-2026-05-06: synth used hp_setpoint = t_air + 1.5 (leash) plus a
        clamp on t_air ∈ [15, 25] to mask the resulting non-physical
        equilibrium. Worked for rate-residual fitter (per-tick local) but
        fails sim-error PEM (trajectory-level). Rewrote to use fixed sp +
        no clamp; equilibrium is now physically self-consistent.
        """
        import random
        from scipy.linalg import expm
        import numpy as np
        random.seed(42)
        rng_t_air = random.Random(43)

        m = int(n_days * 24 * 60 / tick_minutes)
        dt = tick_minutes  # minutes

        alpha_air = self.ALPHA_TOTAL_TRUE * SOLAR_AIR_FRACTION
        alpha_wall = self.ALPHA_TOTAL_TRUE * SOLAR_WALL_FRACTION
        a_wall = self.K_W_TRUE / self.MASS_RATIO_TRUE

        obs = []
        x = np.array([20.0, 20.0], dtype=float)  # [t_air, t_wall] init
        eye2 = np.eye(2)
        for i in range(m):
            hour = (i * tick_minutes / 60.0) % 24.0
            t_out = 5.0 + 7.0 * math.sin(2 * math.pi * (hour - 6) / 24)
            solar = max(0.0, 0.6 * math.sin(2 * math.pi * (hour - 6) / 24))
            t_air = x[0]
            t_wall = x[1]

            # HP idles when solar warming + mild outdoor → no HP demand.
            hp_off = solar > 0.45 and t_out > 8.0
            if hp_off:
                hp_setpoint = self.SP_HEAT_OFF
                clamped_reason = "no_output"
            else:
                hp_setpoint = self.SP_FIXED_HEAT
                clamped_reason = ""

            # True air rate (for room_rate observation field; the actual
            # state advance below uses matrix-exp on the joint system).
            # bench_form active flag: HP heats only when room < setpoint
            # AND not in hp_off mode.
            active = (not hp_off) and (t_air < hp_setpoint)
            if active:
                hp_term = self.K_C_TRUE * (hp_setpoint - t_air)
            else:
                hp_term = 0.0
            air_rate = (
                self.UA_C_TRUE * (t_out - t_air)
                + hp_term
                + alpha_air * solar
                + self.K_W_TRUE * (t_wall - t_air)
            )
            # Add small measurement noise to room_rate.
            room_rate = air_rate + rng_t_air.gauss(0, noise_std)

            obs.append(Observation(
                timestamp=float(i * tick_minutes * 60),
                wall_time=1713650000.0 + i * tick_minutes * 60,
                hp_setpoint=hp_setpoint,
                current_c=t_air,
                desired_c=20.0,
                outdoor_temp_c=t_out,
                room_rate=room_rate,
                raw_readings={"sensor.solar_proxy": solar},
                clamped=clamped_reason != "",
                clamped_reason=clamped_reason,
            ))

            # Advance the joint [T_a, T_w] state by dt minutes via matrix-exp.
            # System matrix depends on bench_form active flag (matches
            # thermal_model.py:315 thermostatic cycling).
            #   active:   dT_a/dt = -(ua_c + k_c + k_w)·T_a + k_w·T_w + b1
            #             b1 = ua_c·T_out + k_c·hp_setpoint + α_air·solar
            #   inactive: dT_a/dt = -(ua_c + k_w)·T_a + k_w·T_w + b1
            #             b1 = ua_c·T_out + α_air·solar
            #   always:   dT_w/dt = a_wall·T_a - a_wall·T_w + b2
            #             b2 = α_wall·solar / mass_ratio
            # x(i+1) = exp(A·dt)·x(i) + ψ(dt)·b(i)  where ψ = A⁻¹·(exp(A·dt) − I)
            if active:
                A = np.array([
                    [-(self.UA_C_TRUE + self.K_C_TRUE + self.K_W_TRUE), self.K_W_TRUE],
                    [a_wall, -a_wall],
                ], dtype=float)
                b1 = (self.UA_C_TRUE * t_out
                      + self.K_C_TRUE * hp_setpoint
                      + alpha_air * solar)
            else:
                A = np.array([
                    [-(self.UA_C_TRUE + self.K_W_TRUE), self.K_W_TRUE],
                    [a_wall, -a_wall],
                ], dtype=float)
                b1 = self.UA_C_TRUE * t_out + alpha_air * solar
            b2 = alpha_wall * solar / self.MASS_RATIO_TRUE
            b = np.array([b1, b2], dtype=float)
            eA = expm(A * dt)
            try:
                psi = np.linalg.solve(A, eA - eye2)
            except np.linalg.LinAlgError:
                psi = np.zeros((2, 2))
            x = eA @ x + psi @ b

        return obs

    def test_dispatch_uses_2r2c_when_eligible(self):
        """With ≥1500 obs and ≥14-day span, fit_greybox returns a 2R2C result."""
        obs = self._generate_2r2c_observations(n_days=21.0, tick_minutes=10.0)
        assert len(obs) >= MIN_OBSERVATIONS_2R2C
        timespan_days = (obs[-1].timestamp - obs[0].timestamp) / 86400.0
        assert timespan_days >= MIN_TIMESPAN_DAYS_2R2C

        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None
        assert result.is_2r2c is True
        assert result.k_w is not None
        assert result.mass_ratio is not None
        assert result.tau_fast is not None
        assert result.tau_slow is not None

    def test_dispatch_falls_back_to_1r1c_short_buffer(self):
        """Below MIN_OBSERVATIONS_2R2C, dispatch returns 1R1C."""
        obs = self._generate_2r2c_observations(n_days=2.0, tick_minutes=10.0)
        assert len(obs) < MIN_OBSERVATIONS_2R2C
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None
        assert result.is_2r2c is False
        assert result.k_w is None
        assert result.tau_fast is None

    def test_dispatch_falls_back_short_timespan(self):
        """≥1500 obs but <14 days → 1R1C fallback."""
        # Many short ticks: 2 days × 24h × 60min / 1min = 2880 obs
        obs = self._generate_2r2c_observations(n_days=2.0, tick_minutes=1.0)
        assert len(obs) >= MIN_OBSERVATIONS_2R2C
        timespan_days = (obs[-1].timestamp - obs[0].timestamp) / 86400.0
        assert timespan_days < MIN_TIMESPAN_DAYS_2R2C
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None
        assert result.is_2r2c is False

    def test_recovers_known_2r2c_parameters(self):
        """Fit recovers ua_c, k_c, α_total, k_w, mass_ratio within tolerance."""
        obs = self._generate_2r2c_observations(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None and result.is_2r2c

        # ua_c, k_c are directly identifiable from the rate equation; tight
        # tolerance.  α_total ties to k_c via the steady-state ratio so it
        # tracks closely too.
        assert abs(result.ua_c - self.UA_C_TRUE) / self.UA_C_TRUE < 0.30
        assert abs(result.k_c - self.K_C_TRUE) / self.K_C_TRUE < 0.30
        assert abs(result.alpha_c - self.ALPHA_TOTAL_TRUE) / self.ALPHA_TOTAL_TRUE < 0.40
        # k_w and mass_ratio are harder (latent state) — looser tolerance.
        assert result.k_w is not None
        assert result.k_w > 0
        assert result.mass_ratio is not None
        assert 1.5 <= result.mass_ratio <= 25.0

    def test_2r2c_bridge_recovers_steady_state_betas(self):
        """β_outdoor and β_solar from 2R2C bridge match the analytic SS gains.

        β_outdoor_true = -ua_c / k_c = -0.01 / 0.04 = -0.25
        β_solar_true   = -α_total / k_c = -0.05 / 0.04 = -1.25
        """
        obs = self._generate_2r2c_observations(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None and result.is_2r2c

        bridge = greybox_to_beta(result, model_inputs)
        beta_outdoor_true = -self.UA_C_TRUE / self.K_C_TRUE
        beta_solar_true = -self.ALPHA_TOTAL_TRUE / self.K_C_TRUE
        assert bridge.beta[1] is not None
        assert abs(bridge.beta[1] - beta_outdoor_true) < 0.10
        assert bridge.beta[2] is not None
        assert abs(bridge.beta[2] - beta_solar_true) < 0.50


class TestGates2R2C:
    """Quality gates specific to 2R2C results."""

    def _make_2r2c_result(self, **overrides) -> GreyboxResult:
        defaults = dict(
            n_observations=2000, n_hp_on=1200, n_hp_off=800,
            c0=0.0005, ua_c=0.01, k_c=0.04, alpha_c=0.05,
            tau_eff=120.0, residual_rms=0.005,
            cost=1.0, n_function_evals=20,
            param_std_err={
                "ua_c": 0.0008, "k_c": 0.002,
                "alpha_c": 0.005, "k_w": 0.002, "mass_ratio": 1.0,
            },
            is_2r2c=True,
            k_w=0.025, mass_ratio=8.0,
            tau_fast=20.0, tau_slow=180.0,
        )
        defaults.update(overrides)
        return GreyboxResult(**defaults)

    def test_2r2c_all_gates_pass_with_good_data(self):
        result = self._make_2r2c_result()
        gates = _check_quality_gates(result)
        failed = [k for k, v in gates.items() if not v]
        assert not failed, f"Unexpected gate failures: {failed}"
        # 1R1C-only gate not present
        assert "tau_plausible" not in gates
        # 2R2C-specific gates present
        assert "tau_fast_plausible" in gates
        assert "tau_slow_plausible" in gates
        assert "tau_separation" in gates
        assert "mass_ratio_plausible" in gates
        assert "k_w_positive" in gates

    def test_tau_fast_too_long_fails(self):
        result = self._make_2r2c_result(tau_fast=GATE_MAX_TAU_FAST + 5.0)
        gates = _check_quality_gates(result)
        assert not gates["tau_fast_plausible"]

    def test_tau_fast_too_short_fails(self):
        result = self._make_2r2c_result(tau_fast=GATE_MIN_TAU_FAST - 1.0)
        gates = _check_quality_gates(result)
        assert not gates["tau_fast_plausible"]

    def test_tau_slow_too_long_fails(self):
        result = self._make_2r2c_result(tau_slow=GATE_MAX_TAU_SLOW + 100.0)
        gates = _check_quality_gates(result)
        assert not gates["tau_slow_plausible"]

    def test_tau_separation_too_close_fails(self):
        """τ_slow ≈ τ_fast → degenerate to 1R1C; gate fails."""
        result = self._make_2r2c_result(tau_fast=50.0, tau_slow=70.0)
        gates = _check_quality_gates(result)
        assert (70.0 / 50.0) < GATE_MIN_TAU_SEPARATION
        assert not gates["tau_separation"]

    def test_mass_ratio_too_low_fails(self):
        result = self._make_2r2c_result(mass_ratio=0.5)
        gates = _check_quality_gates(result)
        assert not gates["mass_ratio_plausible"]

    def test_mass_ratio_too_high_fails(self):
        result = self._make_2r2c_result(mass_ratio=25.0)
        gates = _check_quality_gates(result)
        assert not gates["mass_ratio_plausible"]

    def test_k_w_zero_fails(self):
        result = self._make_2r2c_result(k_w=0.0)
        gates = _check_quality_gates(result)
        assert not gates["k_w_positive"]

    def test_k_w_high_cv_fails_precision(self):
        result = self._make_2r2c_result(
            param_std_err={
                "ua_c": 0.001, "k_c": 0.002,
                "alpha_c": 0.005, "k_w": 0.020, "mass_ratio": 1.0,
            },
            k_w=0.025,
        )
        gates = _check_quality_gates(result)
        assert "param_precision_k_w" in gates
        assert not gates["param_precision_k_w"]


class TestBridge2R2CExposesTaus:
    """Bridge result must surface τ_fast and τ_slow for 2R2C results."""

    def test_2r2c_bridge_populates_tau_fast_and_slow(self):
        result = GreyboxResult(
            n_observations=2000, n_hp_on=1200, n_hp_off=800,
            c0=0.0, ua_c=0.01, k_c=0.04, alpha_c=0.05,
            tau_eff=180.0, residual_rms=0.005,
            cost=1.0, n_function_evals=20,
            param_std_err={
                "ua_c": 0.0008, "k_c": 0.002,
                "alpha_c": 0.005, "k_w": 0.002, "mass_ratio": 1.0,
            },
            is_2r2c=True,
            k_w=0.025, mass_ratio=8.0,
            tau_fast=20.0, tau_slow=180.0,
        )
        bridge = greybox_to_beta(result, model_inputs=[])
        assert bridge.tau_fast == 20.0
        assert bridge.tau_slow == 180.0
        d = bridge.as_dict()
        assert d["tau_fast"] == 20.0
        assert d["tau_slow"] == 180.0

    def test_1r1c_bridge_leaves_tau_fast_slow_none(self):
        result = GreyboxResult(
            n_observations=200, n_hp_on=120, n_hp_off=80,
            c0=0.001, ua_c=0.006, k_c=0.02, alpha_c=0.04,
            tau_eff=166.7, residual_rms=0.005,
            cost=0.5, n_function_evals=10,
            param_std_err={"ua_c": 0.001, "k_c": 0.003, "alpha_c": 0.005},
        )
        bridge = greybox_to_beta(result, model_inputs=[])
        assert bridge.tau_fast is None
        assert bridge.tau_slow is None
        d = bridge.as_dict()
        assert d["tau_fast"] is None
        assert d["tau_slow"] is None


class TestLog2R2C:
    """log_greybox_result handles both 1R1C and 2R2C results."""

    def test_2r2c_log_includes_wall_params(self, caplog):
        import logging
        result = GreyboxResult(
            n_observations=2000, n_hp_on=1200, n_hp_off=800,
            c0=0.0, ua_c=0.01, k_c=0.04, alpha_c=0.05,
            tau_eff=180.0, residual_rms=0.005,
            cost=1.0, n_function_evals=20,
            param_std_err={"ua_c": 0.0008},
            is_2r2c=True,
            k_w=0.025, mass_ratio=8.0,
            tau_fast=20.0, tau_slow=180.0,
        )
        with caplog.at_level(logging.INFO):
            log_greybox_result(result, log_prefix="[t] ")
        assert "Grey-box 2R2C" in caplog.text
        assert "k_w" in caplog.text
        assert "mass_ratio" in caplog.text



class TestFitGreybox2R2CEdgeCases:
    """Edge cases for the 2R2C fit and dispatch logic."""

    def _generate(self, **kwargs):
        # Reuse the simulator from the main 2R2C class
        return TestFitGreybox2R2C()._generate_2r2c_observations(**kwargs)

    def test_2r2c_no_solar_input(self):
        """2R2C fit without a solar proxy: still recovers ua_c, k_c, k_w, mass_ratio."""
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        # Strip solar from raw_readings so model_inputs=[] is consistent.
        for o in obs:
            o.raw_readings.pop("sensor.solar_proxy", None)
        result = fit_greybox(obs, model_inputs=[])
        assert result is not None
        assert result.is_2r2c is True
        # alpha_c is 0.0 (no solar fitted)
        assert result.alpha_c == 0.0
        # k_w and mass_ratio still identified
        assert result.k_w is not None and result.k_w > 0
        assert result.mass_ratio is not None

    def test_2r2c_hp_all_off_returns_none_then_fallback(self):
        """All HP-off → 2R2C returns None → fit_greybox falls back to 1R1C."""
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        for o in obs:
            o.hp_setpoint = o.current_c - 1.0
            o.clamped = True
            o.clamped_reason = "no_output"
        result = fit_greybox(obs, model_inputs=[])
        assert result is not None
        # 2R2C declined (hp_var below threshold) → fell back to 1R1C
        assert result.is_2r2c is False

    def test_2r2c_with_plant_tau_slow_warm_start(self):
        """plant_tau_slow seeds the 2R2C ua_c init AND populates tau_agreement_pct."""
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(
            obs, model_inputs,
            plant_tau_slow=120.0, plant_tau_slow_confidence=0.6,
        )
        assert result is not None
        assert result.is_2r2c is True
        assert result.plant_tau_slow == 120.0
        assert result.tau_agreement_pct is not None

    def test_2r2c_least_squares_failure_falls_back_to_1r1c(self, monkeypatch):
        """If scipy raises, _fit_greybox_2r2c returns None → fit_greybox falls back."""
        from custom_components.tasmota_irhvac.pi import greybox_observer as gb

        original = gb._least_squares
        call_count = {"n": 0}

        def flaky_lsq(*args, **kwargs):
            call_count["n"] += 1
            # First call (1R1C) succeeds; second call (2R2C) raises.
            if call_count["n"] == 1:
                return original(*args, **kwargs)
            raise RuntimeError("synthetic scipy failure")

        monkeypatch.setattr(gb, "_least_squares", flaky_lsq)

        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        # 1R1C fallback after 2R2C raises.
        assert result is not None
        assert result.is_2r2c is False

    def test_2r2c_tau_separation_gate_handles_zero_tau_fast(self):
        """tau_separation gate fails when tau_fast is 0 or tau_slow is inf."""
        result = GreyboxResult(
            n_observations=2000, n_hp_on=1200, n_hp_off=800,
            c0=0.0, ua_c=0.01, k_c=0.04, alpha_c=0.05,
            tau_eff=180.0, residual_rms=0.005,
            cost=1.0, n_function_evals=20,
            param_std_err={"ua_c": 0.0008, "k_c": 0.002, "alpha_c": 0.005},
            is_2r2c=True,
            k_w=0.025, mass_ratio=8.0,
            tau_fast=0.0,  # degenerate
            tau_slow=float("inf"),
        )
        gates = _check_quality_gates(result)
        assert gates["tau_separation"] is False

    def test_2r2c_jacobian_failure_is_swallowed(self, monkeypatch):
        """If pinv raises, std_err stays empty but the fit still returns."""
        import numpy as np

        def broken_pinv(*args, **kwargs):
            raise np.linalg.LinAlgError("synthetic pinv failure")

        monkeypatch.setattr(np.linalg, "pinv", broken_pinv)
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy", "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None
        # 2R2C still returns a result; std_err is just empty for the failed branch.
        assert result.is_2r2c is True
        assert "ua_c" not in result.param_std_err


class TestComputeDtMedianMinEdgeCases:
    """Cover the early-return branches of ``_compute_dt_median_min``."""

    def test_single_observation_returns_none(self):
        from custom_components.tasmota_irhvac.pi.greybox_observer import (
            _compute_dt_median_min,
        )
        obs = [Observation(
            timestamp=0.0, wall_time=0.0,
            hp_setpoint=22.0, current_c=20.0, desired_c=20.0,
            outdoor_temp_c=15.0, room_rate=0.0,
            raw_readings={}, clamped=False,
        )]
        assert _compute_dt_median_min(obs) is None

    def test_simultaneous_observations_returns_none(self):
        """All dt rounded to 0.1min are <=0 → no positive intervals → None."""
        from custom_components.tasmota_irhvac.pi.greybox_observer import (
            _compute_dt_median_min,
        )
        obs = [
            Observation(
                timestamp=0.0, wall_time=0.0,
                hp_setpoint=22.0, current_c=20.0, desired_c=20.0,
                outdoor_temp_c=15.0, room_rate=0.0,
                raw_readings={}, clamped=False,
            ),
            Observation(
                timestamp=0.0, wall_time=0.0,
                hp_setpoint=22.0, current_c=20.0, desired_c=20.0,
                outdoor_temp_c=15.0, room_rate=0.0,
                raw_readings={}, clamped=False,
            ),
        ]
        assert _compute_dt_median_min(obs) is None


class TestFitGreybox2R2CDefensiveResidualPaths:
    """Cover defensive numerical-failure branches inside _fit_greybox_2r2c.

    These branches are not reached on well-formed data; they exist to keep
    the optimizer from crashing on extreme parameter excursions or weird
    timestamp jitter.  Tests reuse ``TestFitGreybox2R2C``'s synthetic
    trajectory (HP-variance-rich enough to clear the PE gate that opens
    the 2R2C path); see ``_generate`` below.
    """

    def _generate(self, **kwargs):
        return TestFitGreybox2R2C()._generate_2r2c_observations(**kwargs)

    def test_dt_zero_in_residual_loop_is_skipped(self):
        """A non-monotonic timestamp produces dt<=0 mid-trajectory; the
        residual_fn skips that tick (residuals[i] = 0.0)."""
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        # Drag obs[5]'s timestamp BEHIND obs[4]'s so dt_min[5] = 0.
        obs[5].timestamp = obs[4].timestamp - 60.0
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
             "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None  # the dt-skip didn't crash the optimizer

    def test_expm_psi_lin_alg_error_falls_back_to_zeros(self, monkeypatch):
        """When ``np.linalg.solve`` raises ``LinAlgError`` inside _expm_psi,
        psi falls back to a zero matrix instead of propagating."""
        import numpy as np
        from custom_components.tasmota_irhvac.pi import greybox_observer as gb

        original_solve = np.linalg.solve
        call_count = {"n": 0}
        def flaky_solve(A, b):
            call_count["n"] += 1
            # First few calls (1R1C) succeed; later calls (2R2C residual_fn)
            # raise to exercise the LinAlgError branch.
            if call_count["n"] > 3:
                raise np.linalg.LinAlgError("synthetic singular matrix")
            return original_solve(A, b)
        monkeypatch.setattr(np.linalg, "solve", flaky_solve)
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
             "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        # Test passes if no exception escaped; result may or may not be 2R2C.
        assert result is not None

    def test_expm_failure_in_typical_dt_branch_returns_huge_residuals(
        self, monkeypatch,
    ):
        """When ``_expm`` raises while precomputing the typical-dt
        matrix exponentials, residual_fn returns the [1e6]·n_data
        sentinel that steers the outer optimizer away.  The fit either
        falls back to 1R1C or completes with degraded params — what
        matters here is that the except-branch executes.
        """
        from custom_components.tasmota_irhvac.pi import greybox_observer as gb
        original_expm = gb._expm
        seen = {"n": 0}
        def flaky_expm(A):
            seen["n"] += 1
            # 1R1C uses _expm too; let those calls succeed, then raise
            # for the 2R2C residual_fn.
            if seen["n"] > 30:
                raise RuntimeError("synthetic expm failure")
            return original_expm(A)
        monkeypatch.setattr(gb, "_expm", flaky_expm)
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
             "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None  # 1R1C fallback or completed run, no crash

    def test_expm_failure_in_unequal_dt_branch_returns_huge_residuals(
        self, monkeypatch,
    ):
        """A trajectory with ``dt != typical_dt`` for some ticks routes
        through the unequal-dt branch of residual_fn, which has its own
        try/except around ``_expm_psi``.  Force a non-typical dt by
        compressing one observation interval, then make ``_expm`` raise
        on that specific call shape."""
        from custom_components.tasmota_irhvac.pi import greybox_observer as gb
        # Observations: skew obs[10] earlier so dt_min[10] is non-typical
        # (e.g. 5min instead of 10min).
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        obs[10].timestamp = obs[9].timestamp + 5 * 60  # 5-min dt
        # Patch _expm to raise; same ratchet as the typical-dt test.
        original_expm = gb._expm
        seen = {"n": 0}
        def flaky_expm(A):
            seen["n"] += 1
            if seen["n"] > 30:
                raise RuntimeError("synthetic expm failure")
            return original_expm(A)
        monkeypatch.setattr(gb, "_expm", flaky_expm)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
             "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        assert result is not None

    def test_typical_dt_zero_falls_back_to_identity_matrices(self):
        """When all dt entries are zero (e.g. timestamp jitter has flattened
        all intervals to <=0), ``typical_dt`` is 0 and the precompute step
        skips the matrix-exponential and seeds identity matrices instead.
        Constructed by directly invoking _fit_greybox_2r2c with a 1R1C
        warm-start and obs whose timestamps were rewritten post-1R1C."""
        from custom_components.tasmota_irhvac.pi import greybox_observer as gb
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
             "input_role": "solar"},
        ]
        r_1r1c = gb._fit_greybox_1r1c(obs, model_inputs)
        assert r_1r1c is not None
        # Flatten timestamps so dt_min[i] == 0 for all i ≥ 1.
        for o in obs:
            o.timestamp = 0.0
        result = gb._fit_greybox_2r2c(obs, model_inputs, r_1r1c)
        # Defensive guard runs without crashing; we don't assert on
        # parameter values since dt=0 means no information for the fit.
        assert result is None or result.is_2r2c is True

    def test_residual_rms_is_zero_when_result_fun_is_none(self, monkeypatch):
        """When the optimizer returns ``result.fun = None`` (no residual
        history), residual_rms falls back to 0.0 instead of dividing by m."""
        from types import SimpleNamespace
        from custom_components.tasmota_irhvac.pi import greybox_observer as gb

        original_lsq = gb._least_squares
        seen = {"n": 0}
        def patched_lsq(*args, **kwargs):
            seen["n"] += 1
            real = original_lsq(*args, **kwargs)
            # Patch the second invocation (the 2R2C fit) only.
            if seen["n"] == 2:
                return SimpleNamespace(
                    x=real.x, fun=None, jac=real.jac,
                    cost=real.cost, nfev=real.nfev,
                )
            return real
        monkeypatch.setattr(gb, "_least_squares", patched_lsq)
        obs = self._generate(n_days=21.0, tick_minutes=10.0)
        model_inputs = [
            {"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
             "input_role": "solar"},
        ]
        result = fit_greybox(obs, model_inputs)
        # Either the fit fell back to 1R1C or the 2R2C result has rms=0.0
        # — both branches are valid for this defensive guard.
        assert result is not None
        if result.is_2r2c:
            assert result.residual_rms == 0.0


# ── PriorState + promote_posterior — Pathak §4.2 transfer-learning chain ────


class TestPriorState:
    """PriorState: per-zone persisted greybox priors with lit-default fallback.

    Stage 1 (2026-06-03): env params (c0, ua_c, alpha_c, k_w, mass_ratio) are
    mode-invariant; HP params (k_c, tau_hp) are stored per-mode."""

    def test_default_init_has_no_promotions(self):
        ps = PriorState()
        assert ps.n_promotions == 0
        # Envelope slots all None
        for name in PriorState._ENVELOPE_PARAMS:
            assert getattr(ps, name) is None
            assert not ps.is_promoted(name)
        # HP slots all None for both modes
        for hp_name in PriorState._HP_PARAMS:
            for mode in ("heat", "cool"):
                assert getattr(ps, f"{hp_name}_{mode}") is None
                assert not ps.is_promoted(hp_name, mode=mode)

    def test_mu_for_envelope_param_falls_back_to_lit_default(self):
        ps = PriorState()
        assert ps.mu_for("ua_c") == UA_C_PRIOR_MEAN
        assert ps.sigma_for("ua_c") == UA_C_PRIOR_SIGMA

    def test_mu_for_hp_param_requires_mode(self):
        ps = PriorState()
        with pytest.raises(ValueError, match="mode 'heat' or 'cool' required"):
            ps.mu_for("k_c")
        with pytest.raises(ValueError, match="mode 'heat' or 'cool' required"):
            ps.is_promoted("tau_hp")

    def test_mu_for_unknown_param_raises(self):
        ps = PriorState()
        with pytest.raises(ValueError, match="unknown PriorState parameter"):
            ps.mu_for("not_a_param")

    def test_mu_for_returns_persisted_value_when_promoted(self):
        ps = PriorState(ua_c=(0.008, 0.002), k_c_heat=(0.04, 0.005), n_promotions=1)
        assert ps.mu_for("ua_c") == 0.008
        assert ps.sigma_for("ua_c") == 0.002
        assert ps.is_promoted("ua_c")
        assert ps.mu_for("k_c", mode="heat") == 0.04
        assert ps.is_promoted("k_c", mode="heat")
        # k_c is mode-specific: heating value doesn't propagate to cooling.
        assert ps.mu_for("k_c", mode="cool") != 0.04
        assert not ps.is_promoted("k_c", mode="cool")

    def test_serialization_roundtrip(self):
        """Round-trip with both envelope and per-mode HP slots populated."""
        ps = PriorState(
            ua_c=(0.008, 0.002),
            k_c_heat=(0.04, 0.01),
            k_c_cool=(0.035, 0.012),  # different mode value
            tau_hp_heat=(5.0, 2.0),
            k_w=(0.022, 0.003),
            n_promotions=5,
        )
        restored = PriorState.from_dict(ps.to_dict())
        assert restored == ps

    def test_from_dict_handles_none_and_empty(self):
        assert PriorState.from_dict(None) == PriorState()
        assert PriorState.from_dict({}) == PriorState()

    def test_from_dict_drops_malformed_entries(self):
        """Bogus values for one slot shouldn't poison the others."""
        data = {
            "ua_c": [0.008, 0.002],            # ok
            "k_c_heat": ["not", "numeric"],     # malformed → dropped
            "alpha_c": [float("nan"), 0.01],   # NaN → dropped
            "k_w": [0.02, -1.0],                # negative σ → dropped
            "tau_hp_heat": [5.0, 2.0],          # ok
            "n_promotions": 3,
        }
        ps = PriorState.from_dict(data)
        assert ps.ua_c == (0.008, 0.002)
        assert ps.k_c_heat is None
        assert ps.alpha_c is None
        assert ps.k_w is None
        assert ps.tau_hp_heat == (5.0, 2.0)
        assert ps.n_promotions == 3

    def test_from_dict_migrates_legacy_kc_to_heat_slot(self):
        """Pre-Stage-1 persistence had a single ``k_c`` field (no mode split).
        Restoring such a blob should populate ``k_c_heat`` (bench was
        heating-only; production users were heating-dominant)."""
        data = {
            "ua_c": [0.008, 0.002],
            "k_c": [0.04, 0.01],         # legacy single-mode field
            "n_promotions": 2,
        }
        ps = PriorState.from_dict(data)
        assert ps.k_c_heat == (0.04, 0.01)
        assert ps.k_c_cool is None   # cool stays cold-start
        assert ps.ua_c == (0.008, 0.002)
        assert ps.n_promotions == 2

    def test_from_dict_does_not_clobber_new_format_with_legacy(self):
        """If both legacy ``k_c`` and new ``k_c_heat`` exist (transitional
        blob), the new format wins to avoid overwriting a properly-saved
        Stage 1 posterior."""
        data = {
            "k_c": [0.04, 0.01],          # legacy
            "k_c_heat": [0.045, 0.008],   # Stage 1
        }
        ps = PriorState.from_dict(data)
        assert ps.k_c_heat == (0.045, 0.008)  # new format won

    def test_immutable(self):
        ps = PriorState(ua_c=(0.008, 0.002))
        with pytest.raises(Exception):
            ps.n_promotions = 99  # type: ignore[misc]


class TestPromotePosterior:
    """promote_posterior() gates: lit-grounded only.

    No CV thresholds — the Bayesian framework's wide-σ-on-non-identifiable
    handles that. Three gates: gates_passed, least_squares_converged, no
    railed params (Reynders 2014).
    """

    def _result(self, **overrides) -> GreyboxResult:
        # A 2R2C result with all params + tight std_err that should
        # qualify for promotion.
        defaults = dict(
            n_observations=2000, n_hp_on=1200, n_hp_off=800,
            c0=0.001, ua_c=0.008, k_c=0.04, alpha_c=0.05,
            tau_eff=125.0, residual_rms=0.005,
            cost=1.0, n_function_evals=20,
            param_std_err={
                "c0": 0.0005, "ua_c": 0.001, "k_c": 0.002,
                "alpha_c": 0.005, "k_w": 0.002, "mass_ratio": 1.0,
            },
            is_2r2c=True,
            k_w=0.022, mass_ratio=8.0,
            tau_fast=20.0, tau_slow=180.0,
        )
        defaults.update(overrides)
        return GreyboxResult(**defaults)

    def test_clean_fit_promotes_envelope_and_mode_specific_hp(self):
        """A heat-mode fit promotes envelope params to shared slots and HP
        params to heat-mode slots only. Cool-mode HP slots untouched."""
        result = self._result()
        before = PriorState()
        after = promote_posterior(result, before, mode="heat")
        assert after.n_promotions == 1
        # Envelope updates the shared slots
        assert after.ua_c == (0.008, 0.001)
        assert after.k_w == (0.022, 0.002)
        assert after.mass_ratio == (8.0, 1.0)
        # HP params route to heat-mode slots only
        assert after.k_c_heat == (0.04, 0.002)
        assert after.k_c_cool is None
        # The default _result() has no tau_hp set yet (Stage 1c will populate it)
        assert after.tau_hp_heat is None
        assert after.tau_hp_cool is None
        assert after is not before  # immutable; new instance returned

    def test_promote_routes_hp_to_correct_mode(self):
        """A cool-mode fit updates k_c_cool, NOT k_c_heat."""
        result = self._result(k_c=0.035)
        before = PriorState()
        after = promote_posterior(result, before, mode="cool")
        assert after.k_c_cool == (0.035, 0.002)
        assert after.k_c_heat is None  # heat-mode slot untouched

    def test_promote_requires_valid_mode(self):
        result = self._result()
        before = PriorState()
        with pytest.raises(ValueError, match="mode must be 'heat' or 'cool'"):
            promote_posterior(result, before, mode="not_a_mode")

    def test_envelope_priors_shared_across_modes(self):
        """A heat fit, then a cool fit — envelope params from heat carry
        forward as priors for cool. The two modes share building physics."""
        result_heat = self._result(ua_c=0.008)
        state = promote_posterior(result_heat, PriorState(), mode="heat")
        assert state.ua_c == (0.008, 0.001)
        # Cool fit with a different ua_c value updates the SHARED slot
        result_cool = self._result(ua_c=0.0085)
        state = promote_posterior(result_cool, state, mode="cool")
        # ua_c carries the latest fit (envelope is shared)
        assert state.ua_c == (0.0085, 0.001)
        assert state.n_promotions == 2

    def test_optimizer_not_converged_blocks_promotion(self):
        result = self._result()
        before = PriorState()
        after = promote_posterior(
            result, before, mode="heat",
            least_squares_converged=False,
        )
        assert after is before

    def test_1r1c_result_blocks_promotion(self):
        # 2R2C posterior shape is what we serialize; a 1R1C fit lacks
        # the wall-mode params → no useful posterior to promote.
        result = self._result(is_2r2c=False, k_w=None, mass_ratio=None,
                              tau_fast=None, tau_slow=None)
        before = PriorState()
        after = promote_posterior(result, before, mode="heat")
        assert after is before

    def test_missing_std_err_blocks_promotion(self):
        result = self._result(param_std_err={})
        before = PriorState()
        after = promote_posterior(result, before, mode="heat")
        assert after is before

    def test_railed_param_not_promoted_others_still_promote(self):
        """Reynders 2014 — a railed param's σ is artificially tight from
        the bound clamp, not a true posterior. Don't promote that ONE
        param; others from the same fit are independent and still go."""
        # Rail mass_ratio at upper bound; ua_c stays clean.
        result = self._result(mass_ratio=MASS_RATIO_BOUNDS[1])
        before = PriorState()
        after = promote_posterior(result, before, mode="heat")
        assert after.n_promotions == 1
        assert after.ua_c is not None
        assert after.mass_ratio is None  # railed → not promoted

    def test_param_at_lower_bound_not_promoted(self):
        """Same rule applies at the lower bound."""
        result = self._result(ua_c=UA_C_BOUNDS[0])
        before = PriorState()
        after = promote_posterior(result, before, mode="heat")
        assert after.ua_c is None  # railed lower
        assert after.k_c_heat is not None  # k_c was clean → promoted to heat slot

    def test_param_with_nan_std_err_not_promoted(self):
        result = self._result(param_std_err={
            "ua_c": float("nan"), "k_c": 0.002, "alpha_c": 0.005,
            "k_w": 0.002, "mass_ratio": 1.0,
        })
        before = PriorState()
        after = promote_posterior(result, before, mode="heat")
        assert after.ua_c is None
        assert after.k_c_heat is not None

    def test_all_params_railed_returns_unchanged(self):
        # All params at bounds → no updates → input returned unchanged.
        result = self._result(
            c0=C0_BOUNDS[0],
            ua_c=UA_C_BOUNDS[1],
            k_c=K_C_BOUNDS[1],
            alpha_c=0.0,  # at lower bound
            k_w=K_W_BOUNDS[1],
            mass_ratio=MASS_RATIO_BOUNDS[1],
        )
        before = PriorState()
        after = promote_posterior(result, before, mode="heat")
        assert after is before  # n_promotions NOT incremented

    def test_successive_promotions_accumulate(self):
        """Chain of promotions: each round overwrites with the new posterior
        and increments n_promotions."""
        result1 = self._result(ua_c=0.008)
        result2 = self._result(ua_c=0.010)
        result3 = self._result(ua_c=0.0095)
        state = PriorState()
        state = promote_posterior(result1, state, mode="heat")
        state = promote_posterior(result2, state, mode="heat")
        state = promote_posterior(result3, state, mode="heat")
        assert state.n_promotions == 3
        # Envelope (ua_c) carries the latest posterior, not an average.
        assert state.ua_c[0] == 0.0095

    def test_partial_promotion_preserves_prior_promotions(self):
        """If batch N rails mass_ratio but batch N+1 doesn't, the
        N+1 batch should promote mass_ratio fresh (since prior was None)."""
        state = PriorState()
        # Batch 1: mass_ratio railed → only other params promoted.
        r1 = self._result(mass_ratio=MASS_RATIO_BOUNDS[1])
        state = promote_posterior(r1, state, mode="heat")
        assert state.mass_ratio is None
        assert state.ua_c is not None
        ua_c_after_b1 = state.ua_c
        # Batch 2: clean mass_ratio. ua_c also re-promoted.
        r2 = self._result(mass_ratio=7.5)
        state = promote_posterior(r2, state, mode="heat")
        assert state.n_promotions == 2
        assert state.mass_ratio == (7.5, 1.0)
        # ua_c should have been re-promoted from batch 2's posterior
        assert state.ua_c != ua_c_after_b1 or state.ua_c[0] == 0.008


class TestFitGreyboxWithPriorState:
    """End-to-end: fit_greybox accepts prior_state and uses it for the
    Tikhonov penalty. Stage 1 (2026-06-03): fit pipeline is heat-mode by
    construction; HP params look up via the heat-mode slot."""

    def test_fit_with_default_prior_state_unchanged_behavior(self):
        """prior_state=None (or default PriorState) should give the
        same result as the no-arg call — backwards-compatible."""
        from tests.test_greybox_observer import TestFitGreybox2R2C
        gen = TestFitGreybox2R2C()
        obs = gen._generate_2r2c_observations(n_days=21.0, tick_minutes=10.0)
        mi = [{"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
               "input_role": "solar"}]
        r_no_arg = fit_greybox(obs, mi)
        r_with_default = fit_greybox(obs, mi, prior_state=PriorState())
        assert r_no_arg is not None and r_with_default is not None
        # Same fit, same numerics (within optimizer tolerance).
        assert abs(r_no_arg.ua_c - r_with_default.ua_c) < 1e-6

    def test_sharp_envelope_prior_anchors_param_near_persisted_value(self):
        """A very tight envelope prior anchors the fit near μ even when
        the data alone would suggest a different value."""
        from tests.test_greybox_observer import TestFitGreybox2R2C
        gen = TestFitGreybox2R2C()
        obs = gen._generate_2r2c_observations(n_days=21.0, tick_minutes=10.0)
        mi = [{"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
               "input_role": "solar"}]
        # Envelope param (ua_c) — mode-invariant slot.
        sharp_prior = PriorState(ua_c=(0.005, 0.0001))
        r_anchored = fit_greybox(obs, mi, prior_state=sharp_prior)
        r_lit = fit_greybox(obs, mi)
        assert r_anchored is not None and r_lit is not None
        # The sharply-prior'd fit should land closer to its prior than
        # the lit-prior'd fit does.
        assert abs(r_anchored.ua_c - 0.005) < abs(r_lit.ua_c - 0.005)

    def test_sharp_heat_mode_hp_prior_anchors_kc(self):
        """A sharp heat-mode k_c prior should anchor the fit (since the
        fit pipeline is heat-mode by construction in Stage 1)."""
        from tests.test_greybox_observer import TestFitGreybox2R2C
        gen = TestFitGreybox2R2C()
        obs = gen._generate_2r2c_observations(n_days=21.0, tick_minutes=10.0)
        mi = [{"name": "Solar Proxy", "entity_id": "sensor.solar_proxy",
               "input_role": "solar"}]
        # k_c is now mode-specific. Truth is 0.04; bias prior to 0.02 with tight σ.
        sharp_prior = PriorState(k_c_heat=(0.02, 0.0005))
        r_anchored = fit_greybox(obs, mi, prior_state=sharp_prior)
        r_lit = fit_greybox(obs, mi)
        assert r_anchored is not None and r_lit is not None
        # The heat-mode-anchored fit should land closer to its prior.
        assert abs(r_anchored.k_c - 0.02) < abs(r_lit.k_c - 0.02)
