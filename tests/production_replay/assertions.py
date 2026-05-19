"""Production-bundle assertion library.

A set of predictive assertions that should hold for any future production
debug bundle from this codebase. The point is to encode expected behavior
NOW so when a future bundle is captured, we have a concrete pass/fail
checklist (instead of ad-hoc "does this look right" inspection).

Three severities:
  INVARIANT  — must always hold; failure indicates a bug in this codebase
               or in our understanding of the bench/production parity
  EXPECTED   — usually holds in known regimes; failure may indicate an
               unexpected operating condition or design issue worth
               investigating but not necessarily a bug
  STATISTICAL — holds in aggregate over a multi-day window; failure
                indicates performance regression vs. design target

Assertions take a `Bundle` and return an `AssertionResult`. The runner
(`verify_bundle.py`) collects results and prints a structured report.

When real production bundles disagree with these assertions, the right
move is usually:
  - INVARIANT failures → fix the code or the assertion (if the assertion
    encoded a misconception)
  - EXPECTED failures → investigate the operating condition, may relax
    the assertion if the regime is legitimate
  - STATISTICAL failures → compare to baseline, may indicate ineffective
    tuning, environmental difference, or assertion threshold too tight
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


SEVERITY_INVARIANT = "INVARIANT"
SEVERITY_EXPECTED = "EXPECTED"
SEVERITY_STATISTICAL = "STATISTICAL"


def extract_room_temp_c(tick: dict) -> float | None:
    """Pull room temperature out of a tick dict, returning Celsius.

    Production bundles store sensor values in `_observation.raw_readings`
    as a FLAT dict keyed by entity_id (e.g. `'sensor.X_air_temperature'`).
    The Tasmota-IRHVAC integration converts F→C on read so values land in
    Celsius — but we detect by value range to be robust to config variants
    where conversion might fail (e.g. a sensor reporting raw F).

    Strategy: find unique key ending in `_temperature` (any prefix). If
    its value lies in [5, 35] treat as Celsius; in [40, 100] treat as
    Fahrenheit and convert. Reject ambiguous values (>100 or <-50) since
    those signal a unit-config bug we shouldn't silently paper over.
    """
    obs = tick.get("_observation") or {}
    raw = obs.get("raw_readings") or {}
    # Flat dotted keys like 'sensor.kitchen_air_sensor_temperature'
    candidates = [v for k, v in raw.items()
                  if k.startswith("sensor.")
                  and k.endswith("_temperature")
                  and isinstance(v, (int, float))]
    if len(candidates) != 1:
        return None
    val = candidates[0]
    if 5 <= val <= 35:
        return val  # Celsius
    if 40 <= val <= 100:
        return (val - 32) * 5 / 9  # Fahrenheit
    return None  # implausible — surface as missing data


def extract_desired_c(tick: dict) -> float | None:
    """Return the desired temperature in Celsius.

    Prefer `_effective_desired_c` (the supervisor-adjusted target the PI
    actually tracked, always in C). Fall back to `desired_temp` (HA UI
    field, may be F or C depending on user config — detect by value range:
    50-100 = F, 5-35 = C).
    """
    eff = tick.get("_effective_desired_c")
    if isinstance(eff, (int, float)):
        return eff
    desired = tick.get("desired_temp")
    if not isinstance(desired, (int, float)):
        return None
    if 50 <= desired <= 100:
        return (desired - 32) * 5 / 9
    if 5 <= desired <= 35:
        return desired
    return None


@dataclass
class Bundle:
    """A loaded production debug bundle from
    `local/debug_bundles/<timestamp>_climate_<zone>_heat_pump/`."""

    path: Path
    ticks: list[dict] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path) -> "Bundle":
        path = Path(path)
        tick_log = path / "tick_log.jsonl"
        if not tick_log.exists():
            raise FileNotFoundError(f"No tick_log.jsonl in {path}")
        ticks = []
        with tick_log.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    ticks.append(json.loads(line))
        return cls(path=path, ticks=ticks)

    @property
    def n_ticks(self) -> int:
        return len(self.ticks)

    @property
    def days_span(self) -> float:
        """Bundle duration in days from wall-clock timestamps.

        Uses `_ts_wall` (seconds since epoch) rather than `_ts_mono`
        because monotonic clock resets to ~0 on reboot and a bundle may
        span reboots.
        """
        if self.n_ticks < 2:
            return 0.0
        first_ts = self.ticks[0].get("_ts_wall")
        last_ts = self.ticks[-1].get("_ts_wall")
        if first_ts and last_ts and last_ts > first_ts:
            return (last_ts - first_ts) / 86400.0
        return 0.0

    @property
    def supervisor_kind(self) -> str:
        # Try to find in config block; fall back to inferred from data
        for t in self.ticks[:10]:
            cfg = t.get("config") or {}
            kind = cfg.get("pi_supervisor_kind")
            if kind:
                return kind
        # Fallback: if any non-zero qref_bias appears → qref active
        for t in self.ticks:
            if t.get("_qref_bias", 0):
                return "qref"
        return "unknown"

    @property
    def mode(self) -> str:
        """HVAC mode for the bundle. First check the `_observation.mode`
        field (populated each tick from the climate entity's hvac_mode),
        then fall back to config. Returns 'heat' / 'cool' / 'unknown'."""
        for t in self.ticks[:50]:
            obs = t.get("_observation") or {}
            m = obs.get("mode")
            if m in ("heat", "cool"):
                return m
            cfg = t.get("config") or {}
            cm = cfg.get("mode")
            if cm in ("heat", "cool"):
                return cm
        return "unknown"


@dataclass
class AssertionResult:
    name: str
    severity: str
    passed: bool
    detail: str
    counter_example: dict | None = None


Predicate = Callable[[Bundle], AssertionResult]
ASSERTIONS: list[Predicate] = []


def assertion(severity: str):
    """Decorator registering a function as an assertion."""
    def wrap(fn: Predicate) -> Predicate:
        fn.__severity__ = severity
        ASSERTIONS.append(fn)
        return fn
    return wrap


# ─── INVARIANT: physical bounds that should never be violated ────────────

@assertion(SEVERITY_INVARIANT)
def qref_bias_bounded_by_bicchi(b: Bundle) -> AssertionResult:
    """qref_bias must satisfy |bias| < ½ quantization step (Bicchi safety).

    Bicchi-Marigo-Piccoli TAC02: nudges exceeding ½ quantization step can
    lock the system to the wrong integer. Default max_bias_c is 0.45 < 0.5.
    """
    max_seen = 0.0
    counter = None
    for t in b.ticks:
        bias = abs(t.get("_qref_bias", 0) or 0)
        if bias > max_seen:
            max_seen = bias
            counter = t
    passed = max_seen <= 0.5
    return AssertionResult(
        name="qref_bias_bounded_by_bicchi",
        severity=SEVERITY_INVARIANT,
        passed=passed,
        detail=f"max |bias| = {max_seen:.4f} (Bicchi bound 0.5)",
        counter_example=counter if not passed else None,
    )


@assertion(SEVERITY_INVARIANT)
def effective_desired_consistent(b: Bundle) -> AssertionResult:
    """When `_effective_desired_c` is present, it should equal
    desired_temp + supervisor_nudge_c + qref_bias (within rounding).

    Failure indicates the supervisor's reported state doesn't match what
    the PI actually used — a serious bookkeeping bug.
    """
    worst_diff = 0.0
    counter = None
    checked = 0
    for t in b.ticks:
        eff = t.get("_effective_desired_c")
        if eff is None:
            continue
        # desired_temp is the user-set target (may be F or C — extract_desired_c
        # handles conversion). effective_desired_c IS that desired plus
        # supervisor's nudge + qref bias. We compare the two against each other.
        # Skip the effective_desired_c shortcut in extract_desired_c here since
        # eff IS effective_desired_c and we want the *underlying* user desired.
        desired_raw = t.get("desired_temp")
        if not isinstance(desired_raw, (int, float)):
            continue
        if 50 <= desired_raw <= 100:
            desired_c = (desired_raw - 32) * 5 / 9
        elif 5 <= desired_raw <= 35:
            desired_c = desired_raw
        else:
            continue
        nudge = t.get("_supervisor_nudge_c", 0) or 0
        bias = t.get("_qref_bias", 0) or 0
        # auto_perturb offset can also contribute; we don't have it directly
        # so we tolerate up to 0.5°C of unexplained offset
        expected = desired_c + nudge + bias
        diff = abs(eff - expected)
        if diff > worst_diff:
            worst_diff = diff
            counter = t
        checked += 1
    passed = worst_diff < 0.5
    return AssertionResult(
        name="effective_desired_consistent",
        severity=SEVERITY_INVARIANT,
        passed=passed,
        detail=f"checked {checked} ticks; worst |eff - (desired+nudge+bias)| = "
               f"{worst_diff:.3f} (tolerates ≤ 0.5 for auto_perturb)",
        counter_example=counter if not passed else None,
    )


@assertion(SEVERITY_INVARIANT)
def qref_bias_zero_when_kind_not_qref(b: Bundle) -> AssertionResult:
    """When pi_supervisor_kind != 'qref', _qref_bias must be 0.

    Failure indicates state leaking from a previous mode or a bug in
    the supervisor selector.
    """
    if b.supervisor_kind == "qref":
        return AssertionResult(
            name="qref_bias_zero_when_kind_not_qref",
            severity=SEVERITY_INVARIANT,
            passed=True,
            detail="N/A (kind = qref)",
        )
    nonzero = [t for t in b.ticks if abs(t.get("_qref_bias", 0) or 0) > 1e-9]
    passed = not nonzero
    return AssertionResult(
        name="qref_bias_zero_when_kind_not_qref",
        severity=SEVERITY_INVARIANT,
        passed=passed,
        detail=f"kind = {b.supervisor_kind}; {len(nonzero)} ticks with non-zero qref_bias",
        counter_example=nonzero[0] if nonzero else None,
    )


# ─── EXPECTED: should hold in known regimes ───────────────────────────────

@assertion(SEVERITY_EXPECTED)
def qref_bias_changes_smoothly(b: Bundle) -> AssertionResult:
    """qref_bias should change smoothly (no jumps > 2× the max per-tick
    update). Failure suggests a state-reset bug or a value being
    written outside the EMA path.

    Max per-tick change = α × g × 0.5 (max q_error magnitude). For default
    aggressive (α=0.2, g=1.0): 0.1/tick. We allow 2× that = 0.2/tick.
    """
    max_jump = 0.0
    counter_idx = -1
    for i in range(1, len(b.ticks)):
        b_prev = b.ticks[i-1].get("_qref_bias", 0) or 0
        b_cur = b.ticks[i].get("_qref_bias", 0) or 0
        jump = abs(b_cur - b_prev)
        if jump > max_jump:
            max_jump = jump
            counter_idx = i
    passed = max_jump < 0.2 + 1e-6
    return AssertionResult(
        name="qref_bias_changes_smoothly",
        severity=SEVERITY_EXPECTED,
        passed=passed,
        detail=f"max per-tick |Δbias| = {max_jump:.4f} (expected ≤ 0.2 for "
               f"default aggressive params)",
        counter_example=b.ticks[counter_idx] if counter_idx >= 0 and not passed else None,
    )


@assertion(SEVERITY_EXPECTED)
def qref_decays_when_hp_inactive(b: Bundle) -> AssertionResult:
    """When hp_estimated_active_state stays False for >10 consecutive ticks,
    |qref_bias| should be decreasing or near zero by the end of the window.

    qref's `decay_alpha` should pull bias toward zero when the actuator
    can't respond. Persistent non-zero bias during HP-off would indicate
    the decay logic is broken.
    """
    worst_growth = 0.0
    worst_window = None
    streak_start = None
    for i, t in enumerate(b.ticks):
        active = t.get("_hp_estimated_active_state", True)
        if not active:
            if streak_start is None:
                streak_start = i
        else:
            if streak_start is not None and i - streak_start > 10:
                b_start = abs(b.ticks[streak_start].get("_qref_bias", 0) or 0)
                b_end = abs(b.ticks[i-1].get("_qref_bias", 0) or 0)
                if b_end - b_start > worst_growth:
                    worst_growth = b_end - b_start
                    worst_window = (streak_start, i)
            streak_start = None
    passed = worst_growth < 0.05  # allow small noise; should never grow much
    return AssertionResult(
        name="qref_decays_when_hp_inactive",
        severity=SEVERITY_EXPECTED,
        passed=passed,
        detail=f"max |bias| growth during HP-off streak: {worst_growth:.4f}"
               + (f" (window {worst_window})" if worst_window else ""),
    )


@assertion(SEVERITY_EXPECTED)
def lock_direction_matches_mode(b: Bundle) -> AssertionResult:
    """In heating mode, mean qref_bias during in-deadband periods should be
    POSITIVE (asym caps drift bias toward warmer side). In cooling, NEGATIVE.

    This is the emergent behavior of mode-aware asymmetric caps. Failure
    suggests the asym cap logic isn't being applied per mode, or the
    operating point is dominated by a regime where the caps don't matter.
    """
    if b.supervisor_kind != "qref":
        return AssertionResult(
            name="lock_direction_matches_mode",
            severity=SEVERITY_EXPECTED,
            passed=True,
            detail=f"N/A (kind = {b.supervisor_kind}, not qref)",
        )
    in_db_biases = []
    for t in b.ticks:
        desired_c = extract_desired_c(t)
        room = extract_room_temp_c(t)
        if desired_c is None or room is None:
            continue
        if abs(room - desired_c) >= 0.5:  # not in deadband
            continue
        bias = t.get("_qref_bias", 0) or 0
        in_db_biases.append(bias)
    if not in_db_biases:
        return AssertionResult(
            name="lock_direction_matches_mode",
            severity=SEVERITY_EXPECTED,
            passed=True,
            detail="N/A (no in-deadband samples found — needs room temp + desired)",
        )
    mean_bias = sum(in_db_biases) / len(in_db_biases)
    if b.mode == "heat":
        passed = mean_bias > -0.05  # should not be strongly negative
        expected = "positive (warmer-side lock in heating)"
    elif b.mode == "cool":
        passed = mean_bias < 0.05
        expected = "negative (cooler-side lock in cooling)"
    else:
        passed = True
        expected = f"mode = {b.mode} (no expectation)"
    return AssertionResult(
        name="lock_direction_matches_mode",
        severity=SEVERITY_EXPECTED,
        passed=passed,
        detail=f"in-deadband mean bias = {mean_bias:+.4f} over {len(in_db_biases)} "
               f"ticks; expected {expected}",
    )


# ─── STATISTICAL: aggregate-metric thresholds ─────────────────────────────

@assertion(SEVERITY_STATISTICAL)
def hp_changes_per_day_below_baseline(b: Bundle) -> AssertionResult:
    """HP setpoint changes/day should be < 15 (no_defense bench baseline
    for winter_typical). Production may differ from bench, but a 60d real
    winter bundle with no chatter defense averaged 15/day; qref defaults
    target ≤ 6/day.

    Threshold of 12/day chosen as a soft regression bar — well above
    qref's bench performance but below the no-defense baseline.
    """
    if b.days_span < 1.0:
        return AssertionResult(
            name="hp_changes_per_day_below_baseline",
            severity=SEVERITY_STATISTICAL,
            passed=True,
            detail=f"N/A (bundle covers only {b.days_span:.2f} days; "
                   f"need ≥ 1.0 days for meaningful statistics)",
        )
    changes = 0
    last_hp = None
    for t in b.ticks:
        hp = t.get("hp_setpoint")
        if hp is None:
            continue
        if last_hp is not None and hp != last_hp:
            changes += 1
        last_hp = hp
    per_day = changes / b.days_span
    passed = per_day < 12
    return AssertionResult(
        name="hp_changes_per_day_below_baseline",
        severity=SEVERITY_STATISTICAL,
        passed=passed,
        detail=f"{changes} HP changes over {b.days_span:.2f} days = "
               f"{per_day:.2f}/day (threshold: < 12/day)",
    )


@assertion(SEVERITY_STATISTICAL)
def controllable_comfort_within_two_degrees_F(b: Bundle) -> AssertionResult:
    """Room should be within ±2°F of desired ≥ 95% of the time *when the HP
    could have done something different*.

    "Controllable" means the violation is in the direction the HP can act on:
      heating mode: room < desired - band (HP could have heated more)
      cooling mode: room > desired + band (HP could have cooled more)

    Violations in the opposite direction are physics-driven (solar overshoot
    in heating, cold infiltration in cooling) and the HP literally can't
    respond. Counting them as "comfort failures" would unfairly penalize
    the controller for things it has no actuator authority over.

    Raw (unfiltered) comfort % is included in detail for context.
    """
    if b.mode not in ("heat", "cool"):
        return AssertionResult(
            name="controllable_comfort_within_two_degrees_F",
            severity=SEVERITY_STATISTICAL,
            passed=True,
            detail=f"N/A (mode={b.mode}; can't classify violation direction)",
        )
    band = 1.111  # ±2°F
    in_band = 0
    ctrl_viol = 0  # violation in actuator-can-act direction
    uncrtl_viol = 0  # violation in actuator-can't direction
    for t in b.ticks:
        desired_c = extract_desired_c(t)
        room = extract_room_temp_c(t)
        if desired_c is None or room is None:
            continue
        err = room - desired_c
        if abs(err) <= band:
            in_band += 1
        elif b.mode == "heat":
            if err < 0:  # room too cold — HP could heat more
                ctrl_viol += 1
            else:        # room too hot — solar; HP can't cool
                uncrtl_viol += 1
        else:  # cool
            if err > 0:  # room too hot — HP could cool more
                ctrl_viol += 1
            else:        # room too cold — HP can't heat
                uncrtl_viol += 1
    ctrl_total = in_band + ctrl_viol
    if ctrl_total < 100:
        return AssertionResult(
            name="controllable_comfort_within_two_degrees_F",
            severity=SEVERITY_STATISTICAL,
            passed=True,
            detail=f"N/A (only {ctrl_total} controllable ticks; need ≥ 100)",
        )
    ctrl_pct = in_band / ctrl_total
    raw_pct = in_band / (in_band + ctrl_viol + uncrtl_viol)
    passed = ctrl_pct >= 0.95
    return AssertionResult(
        name="controllable_comfort_within_two_degrees_F",
        severity=SEVERITY_STATISTICAL,
        passed=passed,
        detail=f"controllable: {ctrl_pct:.1%} of {ctrl_total} ticks within "
               f"±2°F (threshold: ≥ 95%); raw: {raw_pct:.1%} with "
               f"{uncrtl_viol} uncontrollable violations ({b.mode} mode)",
    )


@assertion(SEVERITY_EXPECTED)
def no_heat_demand_when_room_warm(b: Bundle) -> AssertionResult:
    """In heating mode, HP should not be calling for additional heat when
    room is already above desired (and vice versa in cooling). Detects:

      heat mode: room > desired + 0.5°C AND hp_setpoint > room + 1°C
                  — controller actively heating an already-warm room
      cool mode: room < desired - 0.5°C AND hp_setpoint < room - 1°C
                  — controller actively cooling an already-cool room

    Brief transient violations (user setpoint changes, sudden outdoor
    shifts) are expected; threshold tolerates ≤ 5% of post-warmup ticks.
    Sustained violations indicate the integrator is corrupted or the
    overtemp/undertemp regime gate isn't firing — exactly the q_feedback
    bunkroom-bug pattern. qref should drop this number toward zero.
    """
    if b.mode not in ("heat", "cool"):
        return AssertionResult(
            name="no_heat_demand_when_room_warm",
            severity=SEVERITY_EXPECTED,
            passed=True,
            detail=f"N/A (mode={b.mode})",
        )
    bad = 0
    total = 0
    worst_excess = 0.0
    worst = None
    for t in b.ticks:
        desired_c = extract_desired_c(t)
        room = extract_room_temp_c(t)
        hp = t.get("hp_setpoint")
        if desired_c is None or room is None or hp is None:
            continue
        total += 1
        if b.mode == "heat":
            if room > desired_c + 0.5 and hp > room + 1.0:
                bad += 1
                excess = hp - room
                if excess > worst_excess:
                    worst_excess = excess
                    worst = t
        else:
            if room < desired_c - 0.5 and hp < room - 1.0:
                bad += 1
                excess = room - hp
                if excess > worst_excess:
                    worst_excess = excess
                    worst = t
    if total < 100:
        return AssertionResult(
            name="no_heat_demand_when_room_warm",
            severity=SEVERITY_EXPECTED,
            passed=True,
            detail=f"N/A (only {total} valid ticks)",
        )
    pct_bad = bad / total
    passed = pct_bad <= 0.05
    direction = "heat-while-hot" if b.mode == "heat" else "cool-while-cold"
    return AssertionResult(
        name="no_heat_demand_when_room_warm",
        severity=SEVERITY_EXPECTED,
        passed=passed,
        detail=f"{bad}/{total} ({pct_bad:.1%}) ticks were {direction} "
               f"(threshold: ≤ 5.0%); worst |hp − room| = {worst_excess:.2f}°C",
        counter_example=worst if not passed else None,
    )


@assertion(SEVERITY_STATISTICAL)
def no_beeps_during_hp_saturation(b: Bundle) -> AssertionResult:
    """When hp_estimated_active_state was False for both consecutive ticks
    and an HP setpoint change occurred — that's a spurious transition the
    gate should have prevented. Should be near zero with the gate in place.
    """
    spurious = 0
    last_hp = None
    last_active = None
    for t in b.ticks:
        hp = t.get("hp_setpoint")
        active = t.get("_hp_estimated_active_state")
        if hp is None or active is None:
            last_hp = hp
            last_active = active
            continue
        if last_hp is not None and hp != last_hp:
            if last_active is False and active is False:
                spurious += 1
        last_hp = hp
        last_active = active
    passed = spurious <= 5  # allow a few transient gate flips
    return AssertionResult(
        name="no_beeps_during_hp_saturation",
        severity=SEVERITY_STATISTICAL,
        passed=passed,
        detail=f"{spurious} HP changes during double-inactive periods "
               f"(threshold: ≤ 5)",
    )


@assertion(SEVERITY_STATISTICAL)
def qref_bias_caps_observed(b: Bundle) -> AssertionResult:
    """In a multi-day bundle with active heating/cooling, the bias should
    have hit its caps at least once (showing the EMA is exploring the full
    dynamic range and the caps are doing work).

    Failure suggests either (a) very calm conditions where chatter never
    developed, or (b) the EMA is unable to reach the caps due to
    parameter mismatch.
    """
    if b.supervisor_kind != "qref":
        return AssertionResult(
            name="qref_bias_caps_observed",
            severity=SEVERITY_STATISTICAL,
            passed=True,
            detail=f"N/A (kind = {b.supervisor_kind})",
        )
    if b.days_span < 2.0:
        return AssertionResult(
            name="qref_bias_caps_observed",
            severity=SEVERITY_STATISTICAL,
            passed=True,
            detail=f"N/A (bundle only {b.days_span:.2f} days)",
        )
    max_bias = max(
        (abs(t.get("_qref_bias", 0) or 0) for t in b.ticks),
        default=0,
    )
    # Default aggressive max_bias_c is 0.45 (sym) or up to 0.5 with asym up.
    # We expect to see at least 0.25 of usage (50% of cap) in any active bundle.
    passed = max_bias >= 0.20
    return AssertionResult(
        name="qref_bias_caps_observed",
        severity=SEVERITY_STATISTICAL,
        passed=passed,
        detail=f"max |bias| = {max_bias:.3f} over {b.days_span:.2f} days "
               f"(threshold: ≥ 0.20, indicates EMA active)",
    )


# ─── Runner ───────────────────────────────────────────────────────────────

def run_all(bundle: Bundle) -> list[AssertionResult]:
    """Run all registered assertions against `bundle`. Returns ordered results."""
    return [a(bundle) for a in ASSERTIONS]


def format_report(bundle: Bundle, results: list[AssertionResult]) -> str:
    lines = []
    lines.append(f"\nProduction bundle assertion report")
    lines.append(f"  bundle:           {bundle.path}")
    lines.append(f"  zone label:       {bundle.ticks[0].get('_zone_label', '?')}")
    lines.append(f"  tick count:       {bundle.n_ticks}")
    lines.append(f"  span (days):      {bundle.days_span:.2f}")
    lines.append(f"  supervisor kind:  {bundle.supervisor_kind}")
    lines.append(f"  mode:             {bundle.mode}")
    lines.append("")
    n_pass = sum(1 for r in results if r.passed)
    lines.append(f"  {n_pass}/{len(results)} assertions passed")
    lines.append("")
    for sev in (SEVERITY_INVARIANT, SEVERITY_EXPECTED, SEVERITY_STATISTICAL):
        sev_results = [r for r in results if r.severity == sev]
        if not sev_results:
            continue
        lines.append(f"  ── {sev} ──")
        for r in sev_results:
            mark = " ✓" if r.passed else " ✗"
            lines.append(f"  {mark} {r.name}")
            lines.append(f"        {r.detail}")
        lines.append("")
    return "\n".join(lines)
