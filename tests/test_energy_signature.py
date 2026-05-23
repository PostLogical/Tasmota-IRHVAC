"""Unit tests for the energy-signature daily-integral β estimator.

Construction strategy: build observations whose PER-WINDOW INTEGRALS satisfy the
energy-signature relation exactly,

    Σ(sp − T) = β_outdoor·Σ(T_out − T) + β_solar·Σ S + b·ΔT  (+ intercept),

then assert the estimator recovers β_outdoor / β_solar / b with zero noise.
Edge cases cover every decline path (bad args, too few windows, no excitation,
rank deficiency, censored/partial windows).
"""
from __future__ import annotations

import math

import pytest

from custom_components.tasmota_irhvac.pi.batch_learning import Observation
from custom_components.tasmota_irhvac.pi.energy_signature import (
    EnergySignatureResult,
    _infer_dt_seconds,
    estimate_energy_signature,
)

SOLAR = "sensor.solar_proxy"
_BOUNDARY_H = 4.0
_BOUNDARY_S = _BOUNDARY_H * 3600.0
_DT = 300.0                      # 5-min ticks
_TICKS_PER_DAY = int(86400 / _DT)  # 288


def _make_window_obs(widx, *, n_ticks, sum_loss, sum_solar, dT, sum_demand,
                     window_days=1, clamp_one=False, base_temp=20.0):
    """Build n_ticks observations in window ``widx`` realizing the given sums.

    Middle ticks sit at base_temp; the last tick is raised by dT (so the window
    net ΔT = dT).  Per-tick loss/solar/demand are spread evenly to hit the sums.
    """
    win_s = window_days * 86400.0
    obs = []
    per_loss = sum_loss / n_ticks
    per_solar = sum_solar / n_ticks
    per_demand = sum_demand / n_ticks
    for j in range(n_ticks):
        room = base_temp + (dT if j == n_ticks - 1 else 0.0)
        wall = _BOUNDARY_S + widx * win_s + 100.0 + j * _DT
        clamped = clamp_one and j == 0
        obs.append(Observation(
            timestamp=wall,
            wall_time=wall,
            hp_setpoint=room + per_demand,
            current_c=room,
            desired_c=base_temp,
            outdoor_temp_c=room + per_loss,
            raw_readings={SOLAR: per_solar},
            clamped=clamped,
        ))
    return obs


def _build(beta_out, beta_sol, b, windows, *, intercept=0.0, **kw):
    """Assemble a multi-window observation list with exact integral relations."""
    obs = []
    for widx, (sl, ss, dT) in enumerate(windows):
        sd = intercept + beta_out * sl + beta_sol * ss + b * dT
        obs.extend(_make_window_obs(
            widx, n_ticks=_TICKS_PER_DAY, sum_loss=sl, sum_solar=ss, dT=dT,
            sum_demand=sd, **kw))
    return obs


# Eight windows: losses, solar, and ΔT vary INDEPENDENTLY (no two regressors
# affine-dependent, else the design is rank-deficient and the fit declines).
_WINDOWS = [
    (-2880.0, 288.0 * 1, -0.20),
    (-3200.0, 288.0 * 8, 0.30),
    (-2600.0, 288.0 * 3, -0.10),
    (-3500.0, 288.0 * 5, 0.25),
    (-3000.0, 288.0 * 11, -0.30),
    (-2700.0, 288.0 * 2, 0.15),
    (-3300.0, 288.0 * 7, -0.25),
    (-2900.0, 288.0 * 4, 0.10),
]


def test_recovers_known_betas_exactly():
    obs = _build(-0.30, -2.00, 0.5, _WINDOWS)
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert r.ok
    assert r.n_windows == 8
    assert r.beta_solar == pytest.approx(-2.00, abs=1e-6)
    assert r.beta_outdoor == pytest.approx(-0.30, abs=1e-6)
    assert r.b_storage == pytest.approx(0.5, abs=1e-6)
    assert r.residual_rms == pytest.approx(0.0, abs=1e-6)
    assert r.window_days == 1


def test_multiday_window_groups_correctly():
    # Two calendar days per window; 5 windows with independent regressors.
    mwins = [
        (-3456.0, 288.0 * 4, 0.10),
        (-3744.0, 288.0 * 12, -0.20),
        (-3168.0, 288.0 * 20, 0.30),
        (-4032.0, 288.0 * 8, -0.05),
        (-3600.0, 288.0 * 16, 0.20),
    ]
    obs = []
    for widx, (sl, ss, dT) in enumerate(mwins):
        sd = -0.30 * sl + -2.0 * ss + 0.5 * dT
        obs.extend(_make_window_obs(
            widx, n_ticks=2 * _TICKS_PER_DAY, sum_loss=sl, sum_solar=ss,
            dT=dT, sum_demand=sd, window_days=2))
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=2)
    assert r.ok and r.n_windows == 5
    assert r.beta_solar == pytest.approx(-2.0, abs=1e-6)
    assert r.beta_outdoor == pytest.approx(-0.30, abs=1e-6)


def test_as_dict_roundtrips_fields():
    obs = _build(-0.30, -2.00, 0.0, _WINDOWS)
    d = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1).as_dict()
    assert d["ok"] is True
    assert d["beta_solar"] == pytest.approx(-2.0, abs=1e-6)
    assert d["window_days"] == 1
    assert len(d["solar_excitation"]) == 2


# ── Decline paths ────────────────────────────────────────────────────────


def test_declines_bad_window_days():
    r = estimate_energy_signature([], solar_entity=SOLAR, window_days=0)
    assert not r.ok and r.reject_reason == "bad_window_days"


def test_declines_no_solar_entity():
    obs = _build(-0.30, -2.0, 0.0, _WINDOWS)
    r = estimate_energy_signature(obs, solar_entity=None, window_days=1)
    assert not r.ok and r.reject_reason == "no_solar_entity"


def test_declines_insufficient_obs():
    # Single observation → cannot infer dt.
    obs = _build(-0.30, -2.0, 0.0, _WINDOWS)[:1]
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert not r.ok and r.reject_reason == "insufficient_obs"


def test_declines_when_all_outdoor_missing():
    obs = _build(-0.30, -2.0, 0.0, _WINDOWS)
    for o in obs:
        o.outdoor_temp_c = None
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert not r.ok and r.reject_reason == "insufficient_obs"


def test_declines_insufficient_windows():
    obs = _build(-0.30, -2.0, 0.0, _WINDOWS[:3])  # only 3 windows < min 4
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert not r.ok and r.reject_reason == "insufficient_windows"
    assert r.n_windows == 3


def test_declines_no_solar_excitation():
    # All windows share the same solar integral → solar column constant.
    wins = [(-288 * (10 + i), 288 * 5, (i - 4) * 0.05) for i in range(8)]
    obs = _build(-0.30, -2.0, 0.0, wins)
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert not r.ok and r.reject_reason == "no_solar_excitation"


def test_declines_rank_deficient():
    # Solar integral set exactly proportional to loss integral → collinear.
    wins = [(-288 * (10 + i), -0.5 * (-288 * (10 + i)), 0.0) for i in range(8)]
    obs = _build(-0.30, -2.0, 0.0, wins)
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert not r.ok and r.reject_reason == "rank_deficient"


def test_censored_window_excluded_then_declines():
    # One clamped tick per window → no fully-controlled window survives.
    obs = _build(-0.30, -2.0, 0.0, _WINDOWS, clamp_one=True)
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert not r.ok and r.reject_reason == "insufficient_windows"
    assert r.n_windows == 0
    assert r.n_windows_total == 8  # full-coverage windows were seen


def test_partial_edge_window_dropped():
    # A short trailing window (few ticks) must not count toward n_windows_total.
    obs = _build(-0.30, -2.0, 0.0, _WINDOWS)
    obs.extend(_make_window_obs(
        99, n_ticks=10, sum_loss=-1000, sum_solar=50, dT=0.1, sum_demand=5.0))
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert r.ok and r.n_windows == 8 and r.n_windows_total == 8


def test_solar_collinearity_reported_finite():
    # The daily solar↔outdoor correlation is reported (finite, in [-1, 1]).
    obs = _build(-0.30, -2.0, 0.5, _WINDOWS)
    r = estimate_energy_signature(obs, solar_entity=SOLAR, window_days=1)
    assert r.ok
    assert not math.isnan(r.solar_collinearity)
    assert -1.0 <= r.solar_collinearity <= 1.0


# ── helper coverage ────────────────────────────────────────────────────────


def test_infer_dt_edge_cases():
    assert _infer_dt_seconds([100.0]) is None        # < 2 samples
    assert _infer_dt_seconds([5.0, 5.0, 5.0]) is None  # no positive diffs
    assert _infer_dt_seconds([0.0, 300.0, 600.0]) == pytest.approx(300.0)


def test_result_default_is_not_ok():
    r = EnergySignatureResult(ok=False, reject_reason="x")
    assert not r.ok and math.isnan(r.beta_solar)
