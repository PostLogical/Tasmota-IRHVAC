# Design: Anomalous Observation Detection + Exclusion

## Status: DRAFT

## Problem

Unmodeled events (open windows, parties, cooking, space heaters) produce sustained
large residuals that corrupt RLS learning and batch WLS fits. Currently the user has
no signal that this is happening, and no way to remove the contaminated observations
from the learning buffer after the fact.

The residual time-of-day analysis (#16, already implemented) catches *recurring*
patterns across the full buffer. This feature catches *acute* anomalies in real time.

## Goals

1. Detect sustained anomalous residuals during online RLS updates
2. Surface an HA Repair identifying the time range and likely cause
3. Let the user exclude those observations from the diversity buffer
4. Escalate to proactive suppression if exclusions become frequent

## Detection

### What we're measuring

Each RLS update in `_rls_learn_observation()` returns a residual:
```
residual = observed_offset - predicted_offset
```

A single large residual is normal (sensor noise, setpoint change transient). A
*sustained* mean shift in residuals means the system is behaving differently from
what the model expects — an unmodeled disturbance is acting on the zone.

### Why CUSUM, not threshold + consecutive-tick counting

The naive approach (flag when |residual| > 3σ for N consecutive ticks) has three
documented weaknesses:

1. **Reset problem** (Hawkins & Olwell, 1998): A single sub-threshold residual
   resets the counter, losing all accumulated evidence. An intermittent anomaly
   (gusting wind through a cracked window) repeatedly resets without triggering.

2. **No optimality guarantee**: CUSUM is minimax-optimal for detecting sustained
   mean shifts (Lorden, 1971; Moustakides, 1986). Consecutive-run counting is
   strictly dominated.

3. **Heavy-tailed residuals**: RLS residuals are not Gaussian in practice — model
   mismatch, 1°C quantized actuator, and non-stationarity produce leptokurtic
   distributions (Ljung & Söderström, 1983; Agamennoni et al., 2012). A 3σ
   Gaussian threshold triggers at ~1.5% rate under Student-t(ν=5), not 0.27%.

CUSUM (Page, 1954; Basseville & Nikiforov, 1993) accumulates evidence continuously,
decays toward zero (never below), and has a built-in dead zone via the reference
value k. It is the standard method for residual-based fault detection in adaptive
systems (Gustafsson, 2000) and building FDD (Katipamula & Brambley, 2005).

### Scale estimation: MAD, not sample σ

Sample standard deviation has 0% breakdown point — a single large outlier inflates
σ̂, masking subsequent anomalies (Huber, 1981). We use the Median Absolute Deviation:

```
σ̂ = 1.4826 × median(|rᵢ - median(r)|)
```

The 1.4826 factor makes MAD consistent with σ for Gaussian data, while remaining
robust to up to 50% contamination. Computed over a rolling window of the last 60
residuals (`_residual_history`).

**Startup:** Require ≥10 residuals in history before enabling detection. Before that,
there's insufficient data for a reliable scale estimate.

**Batch RMS cross-check:** When batch WLS has run, `batch_model_rms` provides an
independent σ estimate from a larger dataset. Use `max(σ̂_MAD, 0.5 × batch_model_rms)`
as a floor to prevent the MAD from collapsing during a long quiet period and then
over-triggering on normal variation.

### Two-sided CUSUM algorithm

State on PIController:
```python
self._residual_history: deque[float] = deque(maxlen=60)  # last 60 residuals
self._cusum_pos: float = 0.0          # upper CUSUM accumulator (detects warming anomalies)
self._cusum_neg: float = 0.0          # lower CUSUM accumulator (detects cooling anomalies)
self._cusum_alarm_start: datetime | None = None   # wall-clock when alarm began
self._cusum_alarm_mono: float | None = None       # monotonic when alarm began
self._cusum_residual_sum: float = 0.0             # sum of residuals during alarm
self._cusum_alarm_ticks: int = 0                  # ticks since alarm
self._anomaly_runs_completed: list[AnomalyEvent] = []  # events to surface
self._exclusion_count: int = 0                    # lifetime counter
self._cusum_cooldown: int = 0                     # ticks remaining in cooldown
```

On each RLS update (both IDB and OODB paths):

```python
# 1. Accumulate residual history
self._residual_history.append(residual)

# 2. Cooldown check
if self._cusum_cooldown > 0:
    self._cusum_cooldown -= 1
    return

# 3. Compute robust scale
if len(self._residual_history) < MIN_RESIDUALS_FOR_DETECTION:  # 10
    return
sigma = _compute_mad_sigma(self._residual_history)
if self._metrics.batch_model_rms is not None:
    sigma = max(sigma, 0.5 * self._metrics.batch_model_rms)
sigma = max(sigma, MIN_SIGMA_FLOOR)  # absolute floor, e.g. 0.05°C

# 4. Standardize
z = residual / sigma

# 5. Update two-sided CUSUM (Basseville & Nikiforov, 1993, §2.6)
self._cusum_pos = max(0.0, self._cusum_pos + z - CUSUM_K)
self._cusum_neg = max(0.0, self._cusum_neg - z - CUSUM_K)

# 6. Check for alarm
if self._cusum_pos > CUSUM_H or self._cusum_neg > CUSUM_H:
    if self._cusum_alarm_start is None:
        # New alarm — record start
        self._cusum_alarm_start = datetime.now()
        self._cusum_alarm_mono = time.monotonic()
        self._cusum_residual_sum = residual
        self._cusum_alarm_ticks = 1
    else:
        # Ongoing alarm — accumulate
        self._cusum_residual_sum += residual
        self._cusum_alarm_ticks += 1
else:
    if self._cusum_alarm_start is not None:
        # Alarm ended — finalize event
        self._finalize_anomaly_event()
        self._cusum_cooldown = CUSUM_COOLDOWN_TICKS  # 30
```

### CUSUM parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| k (reference value) | 1.0σ | δ_min/2 where δ_min = 2σ ≈ 0.3°C. Detects shifts ≥2σ while providing a dead zone against noise (Basseville & Nikiforov, 1993, §2.6). |
| h (decision threshold) | 5.0σ | ARL₀ ≈ 500–1000 ticks = 8–17 hours between false alarms at 60s/tick (Siegmund, 1985). Conservative enough for a residential system where false alarms erode trust. |
| MIN_RESIDUALS | 10 | Minimum history for reliable MAD estimate. |
| MIN_SIGMA_FLOOR | 0.05°C | Prevents division-by-zero or pathological sensitivity when all residuals are near-identical. Below sensor quantization. |
| CUSUM_COOLDOWN_TICKS | 30 | 30 minutes at 60s/tick. Prevents re-triggering on the same event's tail. Reset on mode change (heat↔cool). |

### Expected detection performance

For our system (σ ≈ 0.15–0.2°C baseline):

| Anomaly | Shift in σ units | Expected detection (ticks) | Wall time |
|---------|-----------------|---------------------------|-----------|
| Window wide open (2°C effect) | ~10σ | 1–2 ticks | 1–2 min |
| Window cracked (1°C effect) | ~5σ | 2–3 ticks | 2–3 min |
| Cooking / space heater (0.5°C) | ~2.5σ | 5–8 ticks | 5–8 min |
| Subtle occupancy change (0.3°C) | ~1.5σ | 15–25 ticks | 15–25 min |

Detection time approximation: ARL₁ ≈ h / (δ/σ - k) for δ/σ > k
(Basseville & Nikiforov, 1993, eq. 2.24).

### AnomalyEvent dataclass

```python
@dataclass
class AnomalyEvent:
    """A completed anomalous period detected by CUSUM."""
    start_time: datetime     # wall-clock start
    start_mono: float        # monotonic start (for buffer matching)
    end_time: datetime       # wall-clock end
    end_mono: float          # monotonic end
    tick_count: int          # duration in ticks
    mean_residual: float     # mean residual during event (signed)
    peak_cusum: float        # max(S⁺, S⁻) at alarm — measures severity
```

### Cooldown

After an anomaly event completes, suppress detection for 30 minutes
(CUSUM_COOLDOWN_TICKS = 30 at 60s/tick) to avoid re-triggering on the same
event's tail. Both CUSUM accumulators are reset to zero at cooldown start.
Reset cooldown on mode change (heat↔cool).

### Literature references

- Page, E.S. (1954). "Continuous inspection schemes." *Biometrika*, 41(1-2), 100–115.
- Lorden, G. (1971). "Procedures for reacting to a change in distribution." *Annals of Mathematical Statistics*, 42(6), 1897–1908.
- Moustakides, G.V. (1986). "Optimal stopping times for detecting changes in distributions." *Annals of Statistics*, 14(4), 1379–1387.
- Huber, P.J. (1981). *Robust Statistics*. Wiley.
- Siegmund, D. (1985). *Sequential Analysis: Tests and Confidence Intervals*. Springer.
- Basseville, M. & Nikiforov, I.V. (1993). *Detection of Abrupt Changes: Theory and Application*. Prentice Hall.
- Hawkins, D.M. & Olwell, D.H. (1998). *Cumulative Sum Charts and Charting for Quality Improvement*. Springer.
- Gustafsson, F. (2000). *Adaptive Filtering and Change Detection*. Springer.
- Ljung, L. & Söderström, T. (1983). *Theory and Practice of Recursive Identification*. MIT Press.
- Agamennoni, G. et al. (2012). "Robust estimation for fault detection in adaptive systems."
- Katipamula, S. & Brambley, M.R. (2005). "Methods for fault detection, diagnostics, and prognostics for building systems." *HVAC&R Research*, 11(1), 3–25.
- Widrow, B. & Kollár, I. (2008). *Quantization Noise*. Cambridge University Press.

## Repair Flow

### Issue creation

When `_anomaly_runs_completed` has entries, surface one HA Repair per event:

```python
ir.async_create_issue(
    hass, DOMAIN,
    f"anomalous_observation_{entry_id}_{start_time_iso}",
    is_fixable=True,
    severity=ir.IssueSeverity.WARNING,
    translation_key="anomalous_observation",
    translation_placeholders={
        "time_range": f"{start_time} — {end_time}",
        "duration_min": str(duration_minutes),
        "mean_residual": f"{mean_residual:+.2f}",
        "direction": "cooler than expected" if mean < 0 else "warmer than expected",
        "cause_hint": _cause_hint(mean_residual),
    },
    data={
        "repair_type": "anomalous_observation",
        "entry_id": entry_id,
        "start_mono": start_mono,
        "end_mono": end_mono,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
    },
)
```

### Cause hints

Based on residual direction and CUSUM peak severity:

| Residual sign | Likely cause |
|---------------|-------------|
| Negative (room cooler than predicted) | Open window/door, draft, ventilation |
| Positive (room warmer than predicted) | Solar gain, cooking, space heater, occupancy |

| Peak CUSUM | Severity | Interpretation |
|------------|----------|---------------|
| 5–10σ (just above h) | Moderate | Brief or mild disturbance |
| 10–25σ | Significant | Sustained unmodeled event |
| >25σ | Severe | Major event — window wide open, HVAC malfunction |

### Repair flow (requires #18 infrastructure)

Two-option confirmation using the `RepairsFlow` from #18:

```python
class AnomalousObservationRepairFlow(RepairsFlow):

    async def async_step_init(self, user_input=None):
        return await self.async_step_confirm()

    async def async_step_confirm(self, user_input=None):
        if user_input is not None:
            action = user_input.get("action", "dismiss")
            if action == "exclude":
                # Exclude observations from buffer
                climate = self.hass.data[DATA_KEY][self._entry_id]
                pi = climate._pi
                excluded = pi.exclude_observations_by_time(
                    self._start_mono, self._end_mono
                )
                _LOGGER.info("Excluded %d observations from buffer", excluded)
            # Both actions clear the issue
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({
                vol.Required("action", default="exclude"): vol.In({
                    "exclude": "Exclude these observations from learning",
                    "dismiss": "Dismiss (keep observations)",
                }),
            }),
            description_placeholders=self._placeholders,
        )
```

### Strings

```json
"anomalous_observation": {
    "title": "Unusual temperature deviation ({time_range})",
    "description": "Room was {direction} for {duration_min} minutes (mean residual: {mean_residual}°C). This could indicate an unmodeled heat source or sink such as open windows, cooking, or a space heater.\n\nExcluding these observations prevents them from biasing the model. Dismissing keeps them in the learning buffer.",
    "fix_flow": {
        "step": {
            "confirm": {
                "title": "Handle anomalous observations",
                "description": "Room was {direction} for {duration_min} minutes ({time_range}).\n\nChoose an action:"
            }
        }
    }
}
```

## Exclusion Mechanism

### `DiversityAwareBuffer.exclude_time_range(start_mono, end_mono)`

```python
def exclude_time_range(self, start: float, end: float) -> int:
    """Remove observations within a monotonic timestamp range.

    Returns number of observations removed.
    """
    before = len(self._buffer)
    self._buffer = [
        o for o in self._buffer
        if o.timestamp < start or o.timestamp > end
    ]
    removed = before - len(self._buffer)
    if removed:
        self.recompute_info_matrix()
    return removed
```

Pattern matches existing `filter_inactive()` — filter + recompute.

### `PIController.exclude_observations_by_time(start_mono, end_mono)`

```python
def exclude_observations_by_time(self, start: float, end: float) -> int:
    """Exclude observations from active buffer by timestamp range."""
    buffer = self._active_buffer
    removed = buffer.exclude_time_range(start, end)
    self._exclusion_count += 1
    return removed
```

### What exclusion does and doesn't fix

| Component | Effect of exclusion |
|-----------|-------------------|
| Diversity buffer / batch WLS | Immediate — next batch re-solves without contaminated data |
| Online RLS coefficients | No effect — RLS updates can't be reversed. But batch WLS will correct on next cycle |
| Covariance matrix P | No effect on RLS P. Buffer info matrix is recomputed |

This is acceptable: the batch cycle (every ~50 observations) will re-derive
coefficients from clean data. The RLS will drift toward batch on the next blend.

## Escalation: Frequent Exclusions

If `_exclusion_count >= 3`, add a persistent notification (separate issue):

```json
"frequent_exclusions": {
    "title": "Consider adding a model input or using suppress_learning",
    "description": "You have excluded observations {count} times. If a recurring event causes these anomalies, consider:\n\n1. Adding a model input with a gate entity (e.g., a door/window binary sensor)\n2. Using the suppress_learning entity before the event occurs\n3. Setting up a timer-based automation: suppress learning for N hours"
}
```

This is `is_fixable=False` — it's guidance, not an action.

### Timer-based suppress automation (documentation only)

```yaml
# Example automation — suppress learning for 2 hours
automation:
  - alias: "Suppress PI learning during party"
    trigger:
      - platform: state
        entity_id: input_boolean.having_guests
        to: "on"
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.living_room_suppress_learning
      - delay: "02:00:00"
      - service: switch.turn_off
        target:
          entity_id: switch.living_room_suppress_learning
```

## Persistence

### What survives restarts

- `_exclusion_count`: Persisted in `pi_stored_data` (add to `PIStoredData` dataclass)
- `_residual_history`: NOT persisted — rebuilds naturally over ~60 ticks (1 hour)
- `_anomaly_run` (active): NOT persisted — short-lived, will re-detect if still occurring
- `_anomaly_runs_completed`: NOT persisted — HA Repairs issues are already in the
  issue registry. The repair flow data contains the timestamps needed for exclusion.

### Excluded time ranges

NOT persisted separately. Once observations are removed from the buffer, they're gone.
The buffer itself is persisted (via `pi_stored_data`), so the exclusion survives
restarts automatically.

## Implementation plan

### Phase 1: Detection (one commit)
1. Add `AnomalyEvent` dataclass and `_compute_mad_sigma()` to `health_checks.py`
2. Add CUSUM state to `PIController.__init__` (accumulators, history deque, cooldown)
3. Hook into `_rls_learn_observation()` — after each update, feed residual to CUSUM
4. Add `_update_cusum()` and `_finalize_anomaly_event()` methods
5. Tests: synthetic residual sequences — step shift detection latency matches
   ARL₁ predictions, MAD robustness to outliers, cooldown behavior, alarm
   start/end timestamps, two-sided detection (positive and negative shifts)

### Phase 2: Buffer exclusion (one commit)
1. Add `exclude_time_range()` to `DiversityAwareBuffer`
2. Add `exclude_observations_by_time()` to `PIController`
3. Add `_exclusion_count` to `PIStoredData` persistence
4. Tests: verify removal + info matrix recompute, verify RLS unaffected

### Phase 3: Repair flow (one commit, depends on #18)
1. Add `AnomalousObservationRepairFlow` to `repairs.py`
2. Wire issue creation in `_check_tuning_health()` from `_anomaly_runs_completed`
3. Add strings for `anomalous_observation` and `frequent_exclusions`
4. Tests: mock repair flow, verify exclude action calls through

### Phase 4: Escalation (one commit)
1. Add `frequent_exclusions` issue in `_check_tuning_health()`
2. Add strings
3. Tests

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| False positives during setpoint changes | Learning gate already blocks RLS during transients; CUSUM only fed when gate is open |
| σ̂ too small early (few observations) | Require ≥10 residuals for MAD; floor at max(0.5 × batch_RMS, 0.05°C) |
| σ̂ collapses during long quiet period | batch_model_rms cross-check floor prevents over-sensitivity |
| CUSUM accumulator drift during normal operation | Reference value k=1.0σ provides dead zone — normal noise (z ~ N(0,1)) causes S to hover near zero in expectation |
| Monotonic timestamps don't survive restarts | Buffer observations use monotonic time. Exclusion must happen in the same HA session as detection. After restart, the repair flow data has wall-clock times for display but monotonic times for matching — these won't match post-restart buffer entries. Accept this: stale repairs should be dismissed manually. |
| User excludes valid data | Confirmation dialog explains the tradeoff. Batch WLS will eventually re-learn from new valid observations. |

## Not in scope

- Cross-zone correlation (detecting that multiple zones see the anomaly simultaneously) —
  PIControllers are independent, no cross-zone coordinator exists
- Automatic exclusion without user confirmation — too risky, user should decide
- RLS coefficient rollback — mathematically intractable for sequential estimators
- Anomaly prediction (forecasting when anomalies will occur) — would require usage
  pattern learning, out of scope
