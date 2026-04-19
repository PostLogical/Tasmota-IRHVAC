# Design: Anomalous Observation Detection + Exclusion

## Status: IMPLEMENTED (v0.19.2-pre28)

## Problem

Unmodeled events (open windows, parties, cooking, space heaters) produce sustained
large residuals that corrupt RLS learning and batch WLS fits. The buffer adds
observations every tick regardless of the RLS learning gate (pi_controller.py:2998),
so anomalies during gated periods contaminate batch WLS even though online RLS is
protected.

Batch WLS has two existing defenses (eligibility filter, Huber outlier exclusion)
but both fail against sustained moderate anomalies (2-3σ shift across 20+
observations) — the shift biases the regression fit itself.

## Solution: Two-sided CUSUM + buffer exclusion

### Detection: CUSUM (Basseville & Nikiforov, 1993)

Two-sided CUSUM on every unclamped buffer observation, using MAD-based robust
scale estimation (Huber, 1981). Fast initial response variant (Lucas & Crosier,
1982) resets accumulators after detection.

**Algorithm:**
```
residual = (hp_setpoint - desired_c) - rls.predict(x)
σ̂ = MAD(residual_history), floored at max(0.5 × batch_RMS, 0.05°C)
z = residual / σ̂
S⁺ = max(0, S⁺ + z - k)
S⁻ = max(0, S⁻ - z - k)
If S⁺ > h or S⁻ > h → alarm, reset, enter cooldown
```

**Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| k (reference) | 1.0 | Dead zone for shifts < 2σ (Basseville & Nikiforov, 1993) |
| h (threshold) | 10.0 | ARL₀ ≈ 50,000 ticks → ~1 false alarm per year |
| Cooldown | 30 min wall-clock | Prevents re-triggering on same event |
| Residual history | deque(maxlen=60) | MAD window |

### Exclusion: DiversityAwareBuffer.exclude_time_range()

Filters buffer by monotonic timestamp, recomputes info matrix. Same pattern as
existing `filter_inactive()`. Both heat and cool buffers are cleaned.

### Repair flow: AnomalousObservationRepairFlow

Two-option `RepairsFlow`: exclude observations or dismiss. Mode-aware cause hints:

| Mode | Positive residual | Negative residual |
|------|------------------|-------------------|
| Heat | Unexpected heat loss (window/draft) | Unexpected heat gain (solar/cooking) |
| Cool | Unexpected heat gain (solar/cooking) | Unexpected heat loss (window/ventilation) |

### Escalation

After 3+ exclusions, surfaces `frequent_exclusions` repair suggesting model inputs
with gate entities or proactive `suppress_learning` usage.

## Key design decisions

1. **CUSUM over run-counting** — minimax optimal for sustained mean shifts; immune to
   intermittent dips that reset consecutive counters (Lorden 1971, Moustakides 1986).

2. **MAD over sample σ** — 50% breakdown point; robust to the very outliers we detect
   (Huber 1981). σ̂ is floored at `max(0.5 × batch_RMS, 0.05°C)`.

3. **Fast initial response** — accumulators reset to zero after alarm (Lucas & Crosier
   1982). Without this, a 20-tick -8σ anomaly builds S⁻ to 140, requiring 130 normal
   ticks to decay below h=10. Reset + cooldown is simpler and produces point-in-time
   detections rather than start/end spans.

4. **Runs on every buffer observation** — not just RLS-gated ticks. The buffer accepts
   observations unconditionally, so CUSUM must monitor what actually enters the buffer.
   Clamped observations are skipped (setpoint doesn't reflect controller intent).

5. **Wall-clock cooldown** — tick-based would be unpredictable with variable 1-15 min
   tick spacing. Wall-clock is consistent regardless of system dynamics.

## Files

| File | Role |
|------|------|
| `pi/health_checks.py` | `AnomalyEvent`, `compute_mad_sigma()`, CUSUM constants |
| `pi/pi_controller.py` | CUSUM state, `_update_cusum()`, issue wiring, `exclude_observations_by_time()` |
| `pi/batch_learning.py` | `DiversityAwareBuffer.exclude_time_range()` |
| `pi/pi_stored_data.py` | `exclusion_count` persistence |
| `repairs.py` | `AnomalousObservationRepairFlow` |
| `strings.json` / `en.json` | `anomalous_observation` and `frequent_exclusions` strings |

## Literature

- Page (1954) — CUSUM original
- Basseville & Nikiforov (1993) — *Detection of Abrupt Changes*
- Lorden (1971), Moustakides (1986) — minimax optimality
- Huber (1981) — *Robust Statistics*, MAD estimator
- Lucas & Crosier (1982) — fast initial response CUSUM
- Hawkins & Olwell (1998) — CUSUM vs runs rules
- Gustafsson (2000) — CUSUM on adaptive filter residuals
- Katipamula & Brambley (2005) — building FDD methods
