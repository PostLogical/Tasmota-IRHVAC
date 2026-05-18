"""Over-temp recovery scenario: stylized April-25-2026 LR pathology.

Reproduces the user-reported pattern from production:
    - Clear sunny spring day: cold morning (~-2°C), mild afternoon (~16°C),
      warm evening. Wide diurnal swing.
    - Solar peaks ~13:00 local (17:00 UTC), amplitude ~0.9 of full scale.
    - Room overshoots ~4°C above desired in late afternoon.
    - Solar wanes + outdoor cools rapidly overnight.
    - Room undershoots desired by 2°F+ for multiple hours overnight.

Production observation (April 25 LR debug bundle 20260426_2322):
    peak room 23.3°C (~3.9°C over desired 19.4°C) at 18:55 EDT;
    overnight min 18.0°C (~1.4°C / 2.5°F under desired) sustained ~5h.

Bench mechanism — uses the lit-grounded path:
    - ``solar_schedule`` + ``ModelInputSpec(input_role="solar")`` so solar
      enters both the thermal model (30/70 air/wall split via 2R2C) AND
      the controller's FF model (as a learnable feature).
    - NOT thermal_disturbances — that pathway injects to air node only,
      which is "space heater" not "solar through windows."
    - ``standard_residential_2r2c`` profile — lit-grounded archetype
      (Bacher-Madsen 2011 §5 typical values), avoids the
      ``living_room`` calibration's known identifiability rails.

The scenario is run twice externally (via local/tools/compare_overtemp_replay.py)
against (a) HEAD and (b) the bench-pre42-effective worktree to A/B the 5
controller commits that landed post-pre42.  Not asserted here; this file
provides the harness (config + per-tick capture).
"""

from __future__ import annotations

import math

from tests.hvac_bench.full_stack_runner import (
    FullStackConfig,
    ModelInputSpec,
    run_full_stack,
)


# ── April 25 outdoor profile ─────────────────────────────────────────────


def outdoor_apr25(minute: float) -> float:
    """Stylized April-25 LR outdoor in °C.

    Anchored on the production debug bundle environment_72h.csv (timestamps
    in UTC, user is EDT = UTC−4):
        10:00 UTC (06:00 EDT):  ~30 °F  ≈  -1 °C   (coldest)
        14:00 UTC (10:00 EDT):  ~47 °F  ≈   8 °C
        18:00 UTC (14:00 EDT):  ~56 °F  ≈  13 °C
        22:00 UTC (18:00 EDT):  ~60 °F  ≈  16 °C   (warmest)
        02:00 UTC+1 (22:00 EDT): ~50 °F ≈  10 °C
        10:00 UTC+1 (06:00 EDT next): ~30 °F ≈  -1 °C   (next morning low)

    Sinusoid: mean 7.5°C, amp 9°C, min at 06:00 bench-local (minute 360),
    max at 18:00 bench-local (minute 1080) — daily solar-warming pattern
    where outdoor lags solar by ~5h.  Solar peak at 13:00 local sits well
    inside the outdoor warming half, so room sees compounded heat gain.
    """
    day_minute = minute % 1440.0
    # Min at 06:00 (minute 360), max at 18:00 (minute 1080) bench-local.
    phase = 2 * math.pi * (day_minute - 1080) / 1440.0
    return 7.5 + 9.0 * math.cos(phase)


# ── April 25 solar profile ───────────────────────────────────────────────


def solar_apr25(minute: float) -> float:
    """Stylized April-25 solar proxy in [0, 1].

    Production trace peaked ~0.9 at 17 UTC (13:00 EDT — solar noon).
    Bell curve from sunrise 11:00 to sunset 23:30 UTC (~07:00 → 19:30
    bench-local), peak 0.95 to drive enough thermal gain for ~4°C
    overshoot against the standard_residential_2r2c envelope.
    """
    day_minute = minute % 1440.0
    sunrise_min = 7 * 60.0    # 07:00 bench-local
    sunset_min = 19.5 * 60.0  # 19:30 bench-local
    if day_minute < sunrise_min or day_minute > sunset_min:
        return 0.0
    # Half-sine over [sunrise, sunset]
    return 0.95 * math.sin(math.pi * (day_minute - sunrise_min) / (sunset_min - sunrise_min))


# ── Scenario builder ─────────────────────────────────────────────────────


def build_config(
    n_days: int = 2,
    desired_c: float = 19.44,   # 67°F — user's actual setpoint
    profile_name: str = "standard_residential_2r2c",
    noise_sigma: float = 0.1,
    solar_beta: float = -6.0,
    solar_seed: float = 4.0,
    extra_pi_overrides: dict | None = None,
) -> FullStackConfig:
    """Build the overtemp-overnight scenario config.

    Args:
        n_days: total simulation length. Day 0 is warm-up at the same
            diurnal pattern (lets the controller settle). Day 1 onward
            replays the same shape so the system has converged enough
            FF to actually see the pathology in the late-day overshoot.
        desired_c: setpoint in °C. Default 19.44 ≈ 67°F matches user
            production setpoint.
        profile_name: house archetype. Default standard_residential_2r2c
            (lit-grounded, no identifiability rails).
        noise_sigma: Gaussian sensor noise std-dev (°C). Default 0.1
            matches feedback_test_noise_realism.
        solar_beta: TRUE FF coefficient for solar (signed β; the actual
            physical thermal injection). -6.0 captures the strong solar
            coupling of a single-pane / sunroom-exposed room.
        solar_seed: What the CONTROLLER thinks the coefficient is (user-
            facing positive seed; β_seed = -solar_seed). Default 4.0
            matches production's clamped/learned value. solar_seed <
            |solar_beta| simulates the mis-calibration that caused the
            April 25 pathology: controller doesn't back HP off enough,
            integral has to compensate, goes deeply negative, can't
            recover overnight.

    Returns:
        FullStackConfig wired for solar + outdoor diurnal replay.
    """
    return FullStackConfig(
        n_days=n_days,
        profile_name=profile_name,
        desired_c=desired_c,
        mode="heat",
        outdoor_schedule=outdoor_apr25,
        # solar_schedule on the config is the FALLBACK diurnal; the
        # actual per-tick solar value comes from the ModelInputSpec's
        # own schedule below.
        solar_schedule=solar_apr25,
        noise_sigma=noise_sigma,
        model_inputs=[
            ModelInputSpec(
                name="Solar Proxy",
                entity_id="sensor.solar_proxy",
                input_role="solar",
                _true_ff_coef=solar_beta,
                seed_heat=solar_seed,  # what controller thinks (may differ from truth)
                schedule=solar_apr25,
            ),
        ],
        # Disable batch WLS: production on April 25 had coefficients
        # settled / clamped from weeks of prior data, not being actively
        # learned within a single day. Letting batch fire mid-scenario
        # introduces synth-artifact β updates that don't reflect reality.
        pi_overrides={"pi_batch_wls_enabled": False, **(extra_pi_overrides or {})},
        # Relax κ-gate for a short run with limited diversity (no-op
        # with batch disabled, but kept for consistency if re-enabled).
        relax_kappa_gate=True,
    )


def run_overtemp_overnight(
    n_days: int = 3,
    **kwargs,
) -> dict:
    """Run the scenario and return summary metrics + per-tick history.

    Default 3 days: day 0 warmup, day 1 pathological day (afternoon
    overshoot + first overnight), day 2 lets the overnight cold low
    fully manifest (sinusoid completes the descent by day 2's 06:00
    local, the coldest point).

    Returns a dict with:
        - peak_room_c, peak_overshoot_c
        - overnight_min_c, overnight_max_undershoot_c
        - overnight_minutes_under_2F (room < desired - 1.11°C)
        - regime_minutes_est (integral frozen while room > desired+1)
        - deepest_neg_integral
        - history (list of per-tick dicts from FullStackResult)
    """
    config = build_config(n_days=n_days, **kwargs)
    result = run_full_stack(config)
    history = result.history
    desired_c = config.desired_c

    # Pathological day = day 1 (post-warmup, with a full overnight + AM
    # window following it before sim end at day n_days).
    pathological_day = 1
    day_start_min = pathological_day * 1440.0
    day_end_min = (pathological_day + 1) * 1440.0
    day_history = [
        h for h in history
        if day_start_min <= h["minute"] < day_end_min
    ]

    # Peak over the pathological day
    peak_temp = max(h["room_temp"] for h in day_history) if day_history else desired_c

    # Overnight = 22:00 of pathological day → 09:00 next day (captures
    # the dawn cold low at minute ~1800 day-local = 06:00 local).
    overnight_start = pathological_day * 1440.0 + 22 * 60.0
    overnight_end = (pathological_day + 1) * 1440.0 + 9 * 60.0
    overnight = [
        h for h in history if overnight_start <= h["minute"] < overnight_end
    ]
    overnight_min = min((h["room_temp"] for h in overnight), default=desired_c)
    overnight_under_2f = sum(
        1 for h in overnight if h["room_temp"] < desired_c - 1.11
    ) * config.tick_minutes

    # Detect regime fire indirectly via "integral stays flat while over desired."
    regime_minutes = 0.0
    prev_integ = None
    for h in day_history:
        integ = h.get("integral", 0.0)
        if h["room_temp"] > desired_c + 1.0 and prev_integ is not None:
            if abs(integ - prev_integ) < 0.01:
                regime_minutes += config.tick_minutes
        prev_integ = integ

    overnight_integs = [h.get("integral", 0.0) for h in overnight]
    deepest_neg_integ = min(overnight_integs) if overnight_integs else 0.0

    return {
        "n_days": n_days,
        "desired_c": desired_c,
        "peak_room_c": peak_temp,
        "peak_overshoot_c": max(0.0, peak_temp - desired_c),
        "overnight_min_c": overnight_min,
        "overnight_max_undershoot_c": max(0.0, desired_c - overnight_min),
        "overnight_minutes_under_2F": overnight_under_2f,
        "regime_minutes_est": regime_minutes,
        "deepest_neg_integral": deepest_neg_integ,
        "total_itae": result.total_itae,
        "history": history,
    }


# ── Pytest harness ────────────────────────────────────────────────────────


def test_overtemp_overnight_baseline():
    """Smoke test — scenario runs and emits the expected fields.

    Does NOT assert specific behavior; the goal here is the harness, not
    the regression. The pre42-vs-HEAD A/B comparison happens in
    local/tools/compare_overtemp_replay.py which calls
    ``run_overtemp_overnight()`` against both worktrees.
    """
    out = run_overtemp_overnight(n_days=2)
    # Sanity: scenario produced ticks and a peak
    assert len(out["history"]) > 100
    assert out["peak_room_c"] > out["desired_c"] - 5.0  # not catastrophically cold
    assert out["peak_room_c"] < out["desired_c"] + 15.0  # not catastrophically hot
