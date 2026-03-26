# Disturbance Inputs

Disturbance inputs tell the PI controller about external factors that affect room temperature — pellet stoves, oil boilers, open doors, solar gain, etc. Without them, the controller may learn incorrect feedforward offsets or respond slowly to known conditions.

Each disturbance input can:
- **Suppress FF learning** — prevent the controller from writing bad data to feedforward buckets
- **Apply bias** — adjust the HP setpoint to compensate for the disturbance in real-time

## How it works

On each PI tick, the controller iterates all configured disturbance inputs:

1. **Boolean entities** (binary_sensor, input_boolean, switch): active when state is "on". Applies the configured **default bias** in °C.
2. **Numeric entities** (sensor, input_number): active when value ≠ 0. Applies **entity_value × gain** as bias in °C.
3. If any input with "suppress learning" checked is active, FF bucket writes are skipped.
4. All active biases are summed and added to the feedforward offset.
5. If the total bias changes by more than 1°C between ticks, the PI integral is reset to zero (prevents stale integral from the previous operating regime).

## Configuration

Go to **Settings → Devices → [your device] → Configure → Disturbance Inputs → Add Disturbance Input**.

| Field | Description |
|---|---|
| **Name** | Display name (e.g., "Pellet Stove", "Bunkroom Door") |
| **Entity** | Any HA entity to watch |
| **Suppress FF Learning** | Check to prevent bucket writes when this entity is active |
| **Default Bias (°C)** | Fixed offset for boolean entities. Negative = reduce HP effort, positive = increase. Ignored for numeric entities. |
| **Gain** | Multiplier for numeric entities. Bias = value × gain. Default 1.0. Ignored for boolean entities. |

## Common setups

### Pellet stove / oil boiler (reduces HP workload)

When your pellet stove or oil boiler is heating the same zone as a heat pump:

- **Entity:** `binary_sensor.pellet_stove_heating` (or thermostat call entity)
- **Suppress learning:** Yes (HP needs less effort — don't learn that as normal)
- **Default bias:** -5.0°C (adjust based on observation)
- **Gain:** 1.0

The HP setpoint drops, so it runs less while the stove handles heating. If the stove stops unexpectedly (out of pellets, malfunction), the PI still targets the original desired temperature — the integral winds up and the HP naturally takes over. No automation needed for failover.

**Calibrating the bias:** Start with -5.0°C and watch the companion sensors. If the HP still cycles on while the stove is running, increase the magnitude (e.g., -7.0°C). If the room drops too much when the stove cycles off (thermostat hysteresis), decrease it (e.g., -3.0°C).

### Door sensor (increases heat loss)

When a door to an exterior or unconditioned space is left open:

- **Entity:** `binary_sensor.bunkroom_door`
- **Suppress learning:** Yes (HP is overcompensating for heat loss)
- **Default bias:** +1.0°C (increase HP effort)
- **Gain:** 1.0

Brief door openings (a few minutes) have minimal impact — the EMA smoothing on FF buckets handles occasional noise. Suppress learning still fires instantly, which is safe (just skips one bucket write, not harmful).

If you only want bias applied after the door has been open for a while, use an HA automation with a `for: "00:15:00"` delay to set an `input_boolean`, and point the disturbance input at that instead of the raw door sensor. The raw sensor can still be used for suppress learning (separate disturbance input with suppress=yes, bias=0).

### Solar gain (proportional to intensity)

For solar radiation reducing heating needs:

- **Entity:** `sensor.solar_bias_template` (a template sensor you create)
- **Suppress learning:** No (solar is a natural condition, not atypical)
- **Default bias:** 0.0 (not used — entity is numeric)
- **Gain:** 1.0 (template already outputs °C)

Example template sensor:
```yaml
template:
  - sensor:
      - name: "Solar Bias"
        unit_of_measurement: "°C"
        state: >
          {% set lux = states('sensor.outdoor_lux') | float(0) %}
          {{ (-0.001 * lux) | round(2) }}
```

This outputs a small negative bias when it's sunny, reducing HP effort. Calibrate the multiplier using companion sensor history.

### Suppress-only (no bias adjustment)

For situations where you just want to protect learning without changing HP behavior:

- **Entity:** `binary_sensor.guests_visiting` (or any condition)
- **Suppress learning:** Yes
- **Default bias:** 0.0
- **Gain:** 1.0

## Advanced: template sensors for complex logic

For sources with complex behavior (pellet stove with multiple energy states, oil boiler with zone valves), create a template sensor that outputs the desired bias in °C. Point the disturbance input at it with gain=1.0.

The template sensor is where you encode per-setup logic. The integration just reads the number.

Example for a pellet stove with energy states:
```yaml
template:
  - sensor:
      - name: "Pellet Stove Bias"
        unit_of_measurement: "°C"
        state: >
          {% set state = states('sensor.pellet_stove_energy_state') %}
          {% if state == 'active_fire' %}
            -5.0
          {% elif state == 'blowing_warm' %}
            -2.0
          {% elif state == 'igniting' %}
            -1.0
          {% else %}
            0
          {% endif %}
```

## Manual suppress (service calls)

For ad-hoc testing or one-off situations:

```yaml
# Suppress learning manually
service: tasmota_irhvac.suppress_ff_learning
target:
  entity_id: climate.living_room
data:
  reason: "Testing new bias values"

# Resume learning
service: tasmota_irhvac.resume_ff_learning
target:
  entity_id: climate.living_room
```

Manual suppress clears on HA restart (ephemeral by design). Entity-based suppression from disturbance inputs is persistent and self-healing.

## Calibrating bias values

The companion sensors (`sensor.{device}_hp_setpoint`, `sensor.{device}_pi_integral`, `sensor.{device}_ff_offset`, `sensor.{device}_disturbance_bias`) are recorded in HA long-term statistics. Use them to calibrate:

1. Run without bias for a period with the disturbance active (suppress learning enabled)
2. Watch `hp_setpoint` and `pi_integral` — see how much the PI adjusts to compensate
3. The steady-state adjustment is approximately the bias value you should configure
4. For example: door opens, over 45 minutes the integral drives setpoint up by 1.5°C → configure default bias of +1.5°C to skip the wait next time

## Monitoring

- **Binary sensor:** `binary_sensor.{device}_ff_learning_suppressed` shows whether learning is currently suppressed, with attributes showing manual suppress status, active entity suppressors, and total bias.
- **Diagnostics:** Download from the device page to see full disturbance input config, current state, and FF buckets.
- **Climate entity attributes:** `ff_learning_suppressed` and `disturbance_bias` are included in the climate entity's extra state attributes.
