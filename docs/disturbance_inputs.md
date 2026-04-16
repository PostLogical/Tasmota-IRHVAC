# Model Inputs

Model inputs tell the PI controller about external factors that affect room temperature — pellet stoves, oil boilers, solar gain, etc. They appear as additional features in the RLS feedforward model, allowing it to learn how each factor affects the HP setpoint needed to maintain temperature.

Each model input can:
- **Be learned by RLS** — the model learns a coefficient for each input, quantifying its effect on HP setpoint
- **Suppress FF learning** — prevent the RLS from writing observations when this input is active (for atypical conditions)

## How it works

On each PI tick, the controller:

1. **Reads entity values.** Boolean entities (binary_sensor, switch, climate) map to 1.0 when active (on/heat/cool/burning/etc.), 0.0 otherwise. Numeric entities use their float value directly.
2. **Applies lag filters.** Each input has an optional exponential smoothing filter (configurable τ in seconds) to model thermal inertia — a pellet stove doesn't affect room temperature instantly.
3. **Builds feature vector.** `[1, outdoor_delta, input1_filtered, input2_filtered, ...]` is passed to the RLS model.
4. **RLS predicts FF offset.** The model computes `ff_offset = β₀ + β₁×outdoor_delta + β₂×input1 + β₃×input2 + ...` where each β is a learned coefficient.
5. **Learning suppression.** If any input with `suppress_learning: true` is active (value > 0.5), RLS observation writes are skipped — the learning gates close.

Unlike the old "disturbance input" system, model inputs don't apply a fixed bias. Instead, the RLS model *learns* how much each input shifts the required HP setpoint, adapting over time.

## Configuration

Go to **Settings → Devices → [your device] → Configure → PI Controller → Model Inputs**.

Model inputs are configured as subentries. Each has:

| Field | Description |
|---|---|
| **Name** | Display name (e.g., "Pellet Stove", "Solar Proxy") |
| **Entity** | Any HA entity to watch |
| **Seed Heat (°C)** | Initial RLS coefficient for heating mode. Expert guess for how much this input shifts setpoint. Negative = reduces HP effort. |
| **Seed Cool (°C)** | Initial RLS coefficient for cooling mode. |
| **Clamp Min / Max** | Physical bounds on the learned coefficient. E.g., a stove can only reduce heating need, so clamp_max = 0. |
| **Lag τ (seconds)** | Exponential smoothing time constant. 0 = no filtering. Use ~900s for slow thermal sources. |
| **Suppress Learning** | Check to prevent RLS observation writes when this input is active. |
| **Typical Value** | Expected magnitude when active (for feature scaling). Default 0.5. |

## Common setups

### Pellet stove / oil boiler (reduces HP workload)

When your pellet stove or oil boiler is heating the same zone as a heat pump:

- **Entity:** `binary_sensor.pellet_stove_heating` (or thermostat call entity)
- **Seed Heat:** -5.0 (stove reduces HP effort by ~5°C worth of setpoint)
- **Seed Cool:** 0.0
- **Clamp Max:** 0.0 (stove can only reduce heating need, not increase it)
- **Lag τ:** 900 (15 min — thermal mass delays the effect)
- **Suppress Learning:** Yes (HP needs less effort — don't learn that as normal FF)
- **Typical Value:** 1.0 (binary: either on or off)

The RLS model will learn the actual coefficient over time. The seed is just a starting point. If the stove stops unexpectedly, the PI integral naturally compensates — no automation needed for failover.

### Solar gain (proportional to intensity)

For solar radiation reducing heating needs:

- **Entity:** `sensor.solar_radiation_w_m2` (or a template sensor)
- **Seed Heat:** -0.005 (per W/m², so 500 W/m² → -2.5°C offset)
- **Seed Cool:** 0.005 (solar makes cooling harder)
- **Clamp Min (heat):** -0.02 (physical upper bound on solar effect)
- **Clamp Max (heat):** 0.0
- **Lag τ:** 1800 (30 min — building thermal mass buffers solar)
- **Suppress Learning:** No (solar is a natural condition, not atypical)
- **Typical Value:** 300.0 (typical midday reading for feature scaling)

### Suppress-only (no learned coefficient)

For situations where you just want to protect learning without modeling the effect:

- **Entity:** `binary_sensor.guests_visiting` (or any condition)
- **Seed Heat / Cool:** 0.0
- **Suppress Learning:** Yes

The RLS won't try to learn a coefficient for this input (seeds are zero, and learning is suppressed when it's active). It just gates the learning.

## Advanced: template sensors for complex logic

For sources with complex behavior, create a template sensor that outputs a single float. Point the model input at it.

Example for a pellet stove with energy states:
```yaml
template:
  - sensor:
      - name: "Pellet Stove Intensity"
        unit_of_measurement: "level"
        state: >
          {% set state = states('sensor.pellet_stove_energy_state') %}
          {% if state == 'active_fire' %}
            1.0
          {% elif state == 'blowing_warm' %}
            0.5
          {% elif state == 'igniting' %}
            0.2
          {% else %}
            0
          {% endif %}
```

The RLS model then learns a single coefficient for this continuous signal, capturing the proportional effect.

## Service calls

```yaml
# Suppress learning manually
service: tasmota_irhvac.suppress_ff_learning
target:
  entity_id: climate.living_room
data:
  reason: "Testing new setup"

# Resume learning
service: tasmota_irhvac.resume_ff_learning
target:
  entity_id: climate.living_room

# Reset RLS models to seeds (nuclear option)
service: tasmota_irhvac.reset_ff_seeds
target:
  entity_id: climate.living_room
```

Manual suppress clears on HA restart (ephemeral by design). Entity-based suppression from model inputs is persistent and self-healing.

## Monitoring

- **Binary sensor:** `binary_sensor.{device}_ff_learning_suppressed` shows whether learning is currently suppressed, with attributes showing manual suppress status and active entity suppressors.
- **Sensor:** `sensor.{device}_ff_offset` shows the total feedforward offset (including all model input contributions).
- **Health sensor:** `sensor.{device}_health` checks feature diversity — warns if the model lacks observations across varied conditions.
- **Diagnostics:** Download from the device page to see full model input config, current values, lag filter states, and RLS coefficients.
- **Climate entity attributes:** `ff_learning_suppressed` is included in the climate entity's extra state attributes.

## How seeds evolve

1. **Cold start (0 observations):** FF uses seed coefficients only.
2. **Blending (1-50 observations):** `alpha = obs_count / 50`. FF = `(1-α) × seed + α × RLS`.
3. **Full RLS (50+ observations):** Seed contribution fades to zero. RLS model is authoritative.
4. **If you edit a seed:** The affected RLS coefficient resets to the new seed value with increased uncertainty, allowing faster re-learning.
