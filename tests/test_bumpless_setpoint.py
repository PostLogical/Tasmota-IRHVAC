"""Spec tests for bumpless transfer on setpoint change.

A 2-DOF PI controller with proportional setpoint weighting `b` has output:

    u(t) = kp · (b · r − y)  +  ki · ∫(r − y) dτ

When the setpoint changes from r_old to r_new at time t, the proportional
term jumps by Δu_P = kp · b · (r_new − r_old). Bumpless transfer keeps the
total output u continuous by adjusting the integrator state to cancel
that jump:

    ΔI · ki = −Δu_P
    ΔI = (kp · b / ki) · (r_old − r_new)              (P-cancel formula)

The previous code used `kp · (1 − b) · (r_old − r_new)` — wrong factor
*and* wrong sign convention vs ki. These tests assert the P-cancel form.

References: Åström & Hägglund, *Advanced PID Control*, §3.4 (setpoint
weighting) + §3.5 (bumpless transfer).
"""

from __future__ import annotations

import pytest
from homeassistant.components.climate.const import HVACMode

from .conftest import get_climate_entity


async def _setup_pi(hass, setup_pi_integration, kp: float, ki: float, b: float):
    """Common setup — heat mode active, known kp/ki/b, baseline integral 0."""
    entry = await setup_pi_integration({"pi_tau_estimate": 60})
    climate = get_climate_entity(hass, entry)
    pi = climate._controller
    climate._attr_hvac_mode = HVACMode.HEAT
    pi._pi_kp = kp
    pi._pi_ki = ki
    pi._pi_setpoint_weight = b
    pi._desired_temp = 21.0
    pi._pi_integral = 0.0
    return climate, pi


def _stub_pi_tick(pi) -> None:
    """Replace _pi_tick with a no-op so we can isolate the bumpless math."""
    async def _noop_tick(now=None):
        return False
    pi._pi_tick = _noop_tick  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_set_temperature_bumpless_pcancel_classic_pi(hass, setup_pi_integration):
    """β=1 (classic PI): ΔI = kp/ki · (old − new)."""
    kp, ki, b = 3.0, 0.15, 1.0
    _, pi = await _setup_pi(hass, setup_pi_integration, kp, ki, b)
    _stub_pi_tick(pi)

    await pi.set_temperature(temperature=22.0)  # +1°C

    expected_di = (kp * b / ki) * (21.0 - 22.0)  # = -20
    assert pi._pi_integral == pytest.approx(expected_di, abs=1e-9), (
        f"Bumpless ΔI should be {expected_di} (P-cancel kp·β/ki·(old−new)); "
        f"got {pi._pi_integral}"
    )


@pytest.mark.asyncio
async def test_set_temperature_bumpless_pcancel_no_setpoint_weight(
    hass, setup_pi_integration,
):
    """β=0 (no proportional setpoint weighting): no P term to cancel, ΔI = 0."""
    kp, ki, b = 3.0, 0.15, 0.0
    _, pi = await _setup_pi(hass, setup_pi_integration, kp, ki, b)
    _stub_pi_tick(pi)

    await pi.set_temperature(temperature=22.0)

    # β=0 means setpoint weighting fully suppresses the proportional jump,
    # so no integral adjustment is needed.
    assert pi._pi_integral == pytest.approx(0.0, abs=1e-9), (
        f"With β=0 there is no P-term jump to cancel; ΔI must be 0, "
        f"got {pi._pi_integral}"
    )


@pytest.mark.asyncio
async def test_set_temperature_bumpless_pcancel_partial_weight(
    hass, setup_pi_integration,
):
    """β=0.7 (partial weighting): ΔI = kp·0.7/ki · (old − new)."""
    kp, ki, b = 3.0, 0.15, 0.7
    _, pi = await _setup_pi(hass, setup_pi_integration, kp, ki, b)
    _stub_pi_tick(pi)

    await pi.set_temperature(temperature=22.5)  # +1.5°C

    expected_di = (kp * b / ki) * (21.0 - 22.5)  # 3*0.7/0.15 * -1.5 = -21
    assert pi._pi_integral == pytest.approx(expected_di, abs=1e-9), (
        f"Bumpless ΔI should be {expected_di}; got {pi._pi_integral}"
    )


@pytest.mark.asyncio
async def test_set_temperature_bumpless_applies_for_large_jump(
    hass, setup_pi_integration,
):
    """Bumpless math applies uniformly for any |Δr|, including >2°C jumps.

    Previously the >2°C branch zeroed the integral as a "regime-shift
    escape hatch"; that branch was removed after empirical evaluation
    showed it interfered with the controller's natural settling on large
    setpoint changes without principled benefit.
    """
    kp, ki, b = 3.0, 0.15, 0.7
    _, pi = await _setup_pi(hass, setup_pi_integration, kp, ki, b)
    pi._pi_integral = 5.0
    _stub_pi_tick(pi)

    await pi.set_temperature(temperature=24.0)  # +3°C — was the threshold case

    # P-cancel formula: ΔI = (kp·b/ki)·(old − new) = 3·0.7/0.15·(21−24) = −42
    expected_di = (kp * b / ki) * (21.0 - 24.0)
    assert pi._pi_integral == pytest.approx(5.0 + expected_di, abs=1e-9), (
        f"Bumpless ΔI should be {expected_di} (P-cancel applied for any |Δr|); "
        f"integral {pi._pi_integral} vs expected {5.0 + expected_di}"
    )


@pytest.mark.asyncio
async def test_on_remote_change_bumpless_pcancel(hass, setup_pi_integration):
    """Physical remote → bumpless via the same P-cancel formula."""
    kp, ki, b = 2.0, 0.1, 0.8
    _, pi = await _setup_pi(hass, setup_pi_integration, kp, ki, b)
    _stub_pi_tick(pi)

    pi._hp_setpoint = 21.0
    await pi.on_remote_change(reported_temp_ir_unit=22.0)

    expected_di = (kp * b / ki) * (21.0 - 22.0)  # 2*0.8/0.1 * -1 = -16
    assert pi._pi_integral == pytest.approx(expected_di, abs=1e-9), (
        f"on_remote_change bumpless ΔI should be {expected_di}; "
        f"got {pi._pi_integral}"
    )


@pytest.mark.asyncio
async def test_set_temperature_bumpless_keeps_output_continuous(
    hass, setup_pi_integration,
):
    """End-to-end: P-term jump + integrator adjustment should net to zero.

    Verifies the *purpose* of the formula — controller output u(t) is
    continuous across the setpoint change (modulo numerical precision).
    """
    kp, ki, b = 3.0, 0.15, 0.7
    _, pi = await _setup_pi(hass, setup_pi_integration, kp, ki, b)
    _stub_pi_tick(pi)

    # Establish a nonzero baseline integral so we can verify it shifts
    # by the bumpless delta rather than getting reset.
    pi._pi_integral = 10.0
    integral_before = pi._pi_integral

    # Output contributions before setpoint change (assume y = some value;
    # the proportional contribution depends on b·r − y).
    y = 20.5
    r_old, r_new = 21.0, 22.0
    u_p_before = kp * (b * r_old - y)
    u_i_before = ki * integral_before

    await pi.set_temperature(temperature=r_new)

    integral_after = pi._pi_integral
    u_p_after = kp * (b * r_new - y)
    u_i_after = ki * integral_after

    assert (u_p_after + u_i_after) == pytest.approx(
        u_p_before + u_i_before, abs=1e-9,
    ), (
        f"Total controller output should be continuous across setpoint change. "
        f"Before P+I = {u_p_before + u_i_before}; "
        f"after P+I = {u_p_after + u_i_after}"
    )
