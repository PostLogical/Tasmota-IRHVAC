# Tasmota-IRHVAC Codebase Curriculum

A structured guide for an intelligent developer to understand this Home Assistant
integration — from zero to "I could extend this myself."

**Prerequisites:** Basic Python, vague awareness that Home Assistant exists, willingness
to read ~4,000 lines of code.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Physical Setup](#2-the-physical-setup)
3. [How Home Assistant Custom Integrations Work](#3-how-home-assistant-custom-integrations-work)
4. [File Map — What Lives Where](#4-file-map--what-lives-where)
5. [Lesson 1: Constants and Configuration](#lesson-1-constants-and-configuration)
6. [Lesson 2: The Base Climate Entity](#lesson-2-the-base-climate-entity)
7. [Lesson 3: MQTT — The Nervous System](#lesson-3-mqtt--the-nervous-system)
8. [Lesson 4: Sending IR Commands](#lesson-4-sending-ir-commands)
9. [Lesson 5: The PI Controller — Theory](#lesson-5-the-pi-controller--theory)
10. [Lesson 6: The PI Controller — Implementation](#lesson-6-the-pi-controller--implementation)
11. [Lesson 7: Feedforward and Auto-Learning](#lesson-7-feedforward-and-auto-learning)
12. [Lesson 8: The Fujitsu Subclass](#lesson-8-the-fujitsu-subclass)
13. [Lesson 9: Config Flow and Options](#lesson-9-config-flow-and-options)
14. [Lesson 10: Buttons, Sensors, and IR Actions](#lesson-10-buttons-sensors-and-ir-actions)
15. [Lesson 11: State Restoration and Resilience](#lesson-11-state-restoration-and-resilience)
16. [Lesson 12: End-to-End Walkthroughs](#lesson-12-end-to-end-walkthroughs)
17. [Common Gotchas](#common-gotchas)
18. [Further Reading](#further-reading)

---

## 1. The Big Picture

This integration lets Home Assistant control air conditioners (heat pumps, mini-splits)
via infrared, using a Tasmota-flashed ESP8266/ESP32 as the IR blaster.

The chain looks like this:

```
Home Assistant UI
       ↓  (HA climate entity API)
This Integration (Python)
       ↓  (MQTT publish)
Tasmota Device (ESP8266/32)
       ↓  (infrared LED)
Air Conditioner
```

And the reverse for state updates — the Tasmota device receives IR from the AC's
remote (or echoes back what it sent) and publishes state via MQTT.

**What makes this integration special** compared to a basic thermostat:
- It speaks the AC's native IR protocol (dozens of vendors supported)
- It includes a **PI controller with feedforward** — a closed-loop control system
  that continuously adjusts the temperature the AC *thinks* it should target,
  based on what the room temperature *actually* is
- It supports vendor-specific features like Fujitsu's Powerful/Econo/Min Heat presets
  via raw IR codes

---

## 2. The Physical Setup

To ground this in reality, here's what one installation looks like:

```
                    ┌─────────────────┐
                    │  Fujitsu Indoor  │
                    │   Unit (wall)    │
                    │                  │
                    │   IR receiver ◄──┼──── IR from Tasmota OR physical remote
                    └─────────────────┘
                              │
                    (refrigerant lines)
                              │
                    ┌─────────────────┐
                    │ Fujitsu Outdoor  │
                    │  Condenser Unit  │
                    └─────────────────┘

     ┌──────────────────────────────────────┐
     │ ESP8266 + IR LED (Tasmota firmware)  │
     │                                      │
     │  - Connected to home WiFi            │
     │  - Speaks MQTT to Home Assistant     │
     │  - Has IR LED pointed at AC unit     │
     │  - Has IR receiver to detect remote  │
     └──────────────────────────────────────┘
              │
         (WiFi/MQTT)
              │
     ┌──────────────────────────────────────┐
     │  Home Assistant                      │
     │  - Runs this integration             │
     │  - Has temp/humidity sensors          │
     │  - Has outdoor temp sensor           │
     │  - Displays climate card in UI       │
     └──────────────────────────────────────┘
```

The AC unit itself has a built-in thermostat, but it only measures the air right
at the unit (near the ceiling). This integration's PI controller uses a *separate*
room temperature sensor to make smarter decisions — it lies to the AC about what
temperature to target so the room reaches the *actual* desired temperature.

---

## 3. How Home Assistant Custom Integrations Work

If you're new to HA development, here's the minimum you need to know:

### The Integration Lifecycle

1. **Discovery/Setup:** User adds integration via UI → triggers `config_flow.py`
2. **Loading:** HA calls `async_setup_entry()` in `__init__.py`
3. **Platform setup:** HA forwards to platform files (`climate.py`, `sensor.py`, `button.py`)
4. **Entity creation:** Each platform creates entity objects (e.g., `TasmotaIrhvac`)
5. **Runtime:** Entities respond to user actions and external events (MQTT messages, sensor updates)
6. **Unloading:** HA calls `async_unload_entry()` for cleanup

### Key HA Concepts

| Concept | What It Is | Example in This Codebase |
|---------|-----------|------------------------|
| **ConfigEntry** | Stored configuration for one instance | One AC unit's settings |
| **Entity** | A thing with state in HA | The climate entity, sensor entities |
| **Platform** | A category of entities | `climate`, `sensor`, `button` |
| **Service** | An action you can call | `tasmota_irhvac.set_econo` |
| **RestoreEntity** | Entity that remembers state across restarts | All our entities |
| **Dispatcher** | Internal pub/sub for cross-entity communication | PI sensors listen for updates |

### The Config Entry Data Model

```
entry.data     →  Immutable connection info (vendor, MQTT topics)
entry.options  →  Mutable settings (temps, PI gains, modes, etc.)
```

This split matters: changing `entry.data` requires reconfiguration; changing
`entry.options` just triggers a reload.

**HA docs to read:**
- [Creating a custom integration](https://developers.home-assistant.io/docs/creating_integration_manifest)
- [Config Flow](https://developers.home-assistant.io/docs/config_entries_config_flow_handler)
- [Climate Entity](https://developers.home-assistant.io/docs/core/entity/climate)

---

## 4. File Map — What Lives Where

```
custom_components/tasmota_irhvac/
├── __init__.py          (158 lines)  — Entry point: setup, services, migration
├── const.py             (223 lines)  — All constants, defaults, config keys
├── climate.py          (1441 lines)  — Base climate entity: MQTT, state, IR commands
├── pi_controller.py     (568 lines)  — PI+feedforward controller mixin
├── fujitsu.py           (270 lines)  — Fujitsu vendor subclass (presets, raw IR)
├── config_flow.py       (998 lines)  — Setup wizard + options flow UI
├── button.py            (184 lines)  — Vane buttons + user-defined IR action buttons
├── sensor.py            (141 lines)  — PI diagnostic sensors (setpoint, integral, ff)
├── diagnostics.py        (84 lines)  — HA diagnostics dump
├── manifest.json                     — Integration metadata (dependencies, version)
├── strings.json                      — UI text (English, with formatjs syntax)
└── translations/
    └── en.json                       — Compiled English translations
```

**Read order for learning:** `const.py` → `climate.py` → `pi_controller.py` →
`fujitsu.py` → `config_flow.py` → everything else.

---

## Lesson 1: Constants and Configuration

**File:** `const.py` (~223 lines)

Start here. This file defines every configuration key, default value, and enum used
throughout the integration.

### What to Look For

**Config keys** — every `CONF_*` constant maps to a setting the user can configure:
```python
CONF_PI_KP = "pi_kp"           # Proportional gain
CONF_PI_KI = "pi_ki"           # Integral gain
CONF_PI_DEADBAND = "pi_deadband"  # Temperature band where PI relaxes
```

**Default values** — the `DEFAULT_*` constants tell you what happens if the user
doesn't configure something:
```python
DEFAULT_PI_KP = 1.0
DEFAULT_PI_KI = 0.02
DEFAULT_PI_DEADBAND = 0.5      # °C — half a degree
DEFAULT_PI_MIN_INTERVAL = 900  # seconds (15 minutes)
```

**Mode lists** — the `HVAC_MODE_*`, fan speed, and swing mode constants define what
the AC can do. Note the `AUTO_FAN` / `FAN_AUTO` distinction — different vendors swap
these meanings.

**Data vs. Options split** — `DATA_KEYS` lists what goes in `entry.data` (immutable
connection info), everything else goes in `entry.options`.

### Exercise
Read `const.py` end to end. For each `CONF_*` constant, ask yourself: "What would
happen if I changed this?" The defaults encode real-world tuning decisions.

---

## Lesson 2: The Base Climate Entity

**File:** `climate.py` (~1441 lines)

This is the largest file and the heart of the integration. It implements HA's
`ClimateEntity` interface — the standard API that makes this show up as a thermostat
in the UI.

### Class Hierarchy

```python
class TasmotaIrhvac(RestoreEntity, ClimateEntity):
    """Base class — works for any vendor."""
```

Or, when PI is enabled:

```python
class PIControllerMixin:
    """Injected between the subclass and the base."""

# At runtime, the actual class ends up being:
# FujitsuTasmotaIrhvac → PIControllerMixin → TasmotaIrhvac → RestoreEntity → ClimateEntity
```

This is Python's **MRO (Method Resolution Order)** in action. The mixin pattern lets
PI be vendor-agnostic — it doesn't know or care about Fujitsu.

### Key Sections to Read

**`__init__`** (~line 100-250): Initializes all state from config. Notice:
- Temperature attributes (`_attr_target_temperature`, `_attr_current_temperature`)
- Mode attributes (`_attr_hvac_mode`, `_attr_fan_mode`, `_attr_swing_mode`)
- Vendor-specific flags (`_turbo`, `_econo`, `_quiet`, `_light`, etc.)
- Toggle list (fields that reset to "off" after every IR send)

**`async_added_to_hass`** (~line 250-370): Called when HA adds the entity. Does:
1. Restore previous state (survives HA restarts)
2. Subscribe to MQTT topics
3. Set up external sensor listeners (temp, humidity, power)
4. Initialize PI controller (if enabled)

**`_handle_state_payload`** (~line 400-500): Parses MQTT JSON from Tasmota into
entity state. This is where the AC's reported state becomes HA state.

**`send_ir`** (~line 900-1000): The outbound path — builds a JSON payload and publishes
it to MQTT. This is the function that actually makes the AC do something.

**`async_set_temperature`, `async_set_hvac_mode`, etc.**: Standard HA climate methods.
Each one updates internal state and calls `send_ir()`.

### The State Model

The entity tracks two "layers" of temperature:
- **`_attr_target_temperature`** — what the user asked for (displayed in UI)
- **`_hp_setpoint`** (PI only) — what the AC is actually told (may differ by several degrees)

When PI is disabled, these are the same. When PI is enabled, the controller
continuously adjusts `_hp_setpoint` to steer the room toward `_attr_target_temperature`.

### Exercise
Find `send_ir()` and trace what happens when you call `async_set_temperature(22)`.
Follow the data from the HA service call all the way to the MQTT publish.

---

## Lesson 3: MQTT — The Nervous System

MQTT is how this integration talks to the physical IR blaster. Understanding the
message flow is essential.

### Topics

| Direction | Topic Pattern | Example | Purpose |
|-----------|--------------|---------|---------|
| **Command** | `cmnd/{device}/irhvac` | `cmnd/ir_living/irhvac` | Send IR command |
| **State** | `tele/{device}/RESULT` | `tele/ir_living/RESULT` | Receive AC state |
| **State 2** | `stat/{device}/RESULT` | `stat/ir_living/RESULT` | Alt state topic |
| **Availability** | `tele/{device}/LWT` | `tele/ir_living/LWT` | Online/Offline |
| **Raw IR** | `cmnd/{device}/irsend` | `cmnd/ir_living/irsend` | Raw IR (presets) |

### Inbound Message (AC → HA)

When the Tasmota device detects an IR signal (from the physical remote or its own
echo), it publishes:

```json
{
  "IrReceived": {
    "Protocol": "FUJITSU_AC",
    "IRHVAC": {
      "Vendor": "FUJITSU_AC",
      "Power": "on",
      "Mode": "heat",
      "Celsius": "on",
      "Temp": 22,
      "FanSpeed": "auto",
      "SwingV": "auto",
      "SwingH": "off",
      "Quiet": "off",
      "Turbo": "off",
      "Econo": "off",
      "Data": "0x...",
      "Bits": 128
    }
  }
}
```

The integration extracts `["IrReceived"]["IRHVAC"]` (or just `["IRHVAC"]` if not
nested) and passes it to `_handle_state_payload()`.

### Outbound Message (HA → AC)

When the integration wants to control the AC:

```json
{
  "StateMode": "SendStore",
  "Vendor": "FUJITSU_AC",
  "Model": "3",
  "Power": "on",
  "Mode": "heat",
  "Celsius": "on",
  "Temp": 22,
  "FanSpeed": "auto",
  "SwingV": "auto",
  "SwingH": "off",
  "Quiet": "off",
  "Turbo": "off",
  "Econo": "off",
  ...
}
```

**`StateMode: "SendStore"`** tells Tasmota to both send the IR and remember the state.
This is important — Tasmota's state memory is used for the echo-back on the state topic.

### Exercise
If you have access to an MQTT client (like MQTT Explorer), subscribe to
`tele/+/RESULT` and press buttons on a physical AC remote. Watch the messages flow.
Then compare with what the integration sends via `cmnd/+/irhvac`.

---

## Lesson 4: Sending IR Commands

**Key function:** `send_ir()` in `climate.py`

This function bridges the gap between HA's abstract climate model and the raw IR
protocol. Let's trace it:

### 1. Build the payload

```python
payload = {
    "StateMode": "SendStore",
    "Vendor": self._vendor,
    "Model": self._model,
    "Power": "on" if self._attr_hvac_mode != HVACMode.OFF else "off",
    "Mode": mode_str,
    "Celsius": self._celsius,
    "Temp": temperature,       # ← This is the critical one
    "FanSpeed": fan_str,
    "SwingV": swingv_str,
    "SwingH": swingh_str,
    # ... all the feature flags
}
```

### 2. Temperature selection

This is where PI makes its mark:

```python
# Without PI: send what the user asked for
temperature = self._attr_target_temperature

# With PI: send what the controller computed
if self._pi_enabled and mode in (HEAT, COOL):
    temperature = self._hp_setpoint  # Could be 5°C higher/lower!
```

### 3. The toggle list

Some AC features are "fire and forget" — you send `Turbo: on` once, and it sticks.
But some ACs reset flags every command. The `toggle_list` handles this:

```python
for field in self._toggle_list:
    payload[field] = "off"  # Reset after sending
```

### 4. MQTT delay

If configured, the integration sleeps before sending. This prevents current spikes
when multiple AC units on the same circuit all turn on simultaneously.

### Raw IR (for presets)

Some commands can't be expressed as IRHVAC JSON — they're special IR codes that
Tasmota's IR library doesn't decode into structured fields. For these, we bypass
the JSON interface entirely:

```python
# Fujitsu Powerful preset
await mqtt.async_publish(
    hass,
    f"cmnd/{device}/irsend",
    FUJITSU_IR_POWERFUL  # Raw IR timing data
)
```

---

## Lesson 5: The PI Controller — Theory

Before reading the code, you need to understand what a PI controller *is* and why
this integration needs one.

### The Problem

Your AC has a built-in thermostat, but it's bad at its job:
1. Its temperature sensor is at the ceiling near the unit — not where you sit
2. It doesn't know about outdoor conditions
3. It can't learn from experience
4. It overshoots and undershoots constantly

### The Solution: Closed-Loop Control

Instead of telling the AC "heat to 72°F," we tell it "heat to 78°F" (or 68°F, or
whatever it takes) so that the *room* reaches 72°F. The PI controller computes that
offset continuously.

### P = Proportional

The proportional term reacts to *current* error:

```
P_term = Kp × error
error  = desired_temp - current_temp
```

If the room is 2°C below target, and Kp = 1.0, P adds 2°C to the setpoint.

**Problem with P alone:** It can't eliminate steady-state error. If the room
stabilizes 0.3°C below target, P only adds 0.3°C — not enough to close the gap.

### I = Integral

The integral term reacts to *accumulated* error over time:

```
I_term = Ki × ∫(error × dt)
```

If the room has been 0.3°C cold for an hour, the integral builds up and adds more
offset. This is what eliminates steady-state error.

**Problem with I alone:** It's slow to react and can "wind up" — if the AC was off
for an hour with high error, the integral accumulates a huge value that causes
massive overshoot when the AC finally catches up.

### P + I Together

P handles fast response, I handles precision. Together they converge on the target
without steady-state error.

```
setpoint = desired_temp + P_term + I_term
```

### FF = Feedforward

The PI terms are *reactive* — they respond to error that already exists. Feedforward
is *proactive* — it adjusts based on outdoor temperature before error develops.

```
setpoint = desired_temp + P_term + I_term + FF_offset
```

If it's -10°C outside, the AC needs to work harder just to maintain temperature.
Feedforward adds that baseline offset so PI doesn't have to "discover" it from scratch.

### Adaptive Setpoint Weighting

Standard PI applies the full proportional gain everywhere. But near the setpoint,
you want *less* aggressive P (to avoid oscillation) and more reliance on I (for
precision). The setpoint weight `b` controls this blend:

```
Far from setpoint:  p_error = 1.0 × desired - current    (full P)
Near setpoint:      p_error = b   × desired - current    (gentler P, b=0.5 default)
```

### Deadband

Within ±0.5°C of target, the controller relaxes:
- P term goes to zero (no proportional action)
- Integral decays by 10% per tick (gentle wind-down)
- Feedforward learning activates (system is "settled")

This prevents the AC from constantly cycling on/off around the setpoint.

### Visual Mental Model

```
        Room too cold ◄──────────────────────── Room too hot
                       │                    │
   ──────────────────┬─┼────────────────────┼─┬──────────────
                     │ │    DEADBAND         │ │
                     │ │  (PI relaxes)       │ │
   P kicks in ◄──────┘ └──────────────────────┘ ──────► P kicks in
   I accumulates                                  I accumulates
   FF adds baseline offset regardless
```

**External resources:**
- [PID Controller (Wikipedia)](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller) — skip the D section, we don't use it
- [Control Theory intro (Brian Douglas, YouTube)](https://www.youtube.com/playlist?list=PLUMWjy5jgHK1NC52DXXrriwihVrYZKqjk) — excellent visual explanations

---

## Lesson 6: The PI Controller — Implementation

**File:** `pi_controller.py` (~568 lines)

Now let's see how the theory maps to code.

### Class Structure

```python
class PIControllerMixin:
    """Mixin — injected into the MRO between vendor subclass and base."""
```

It's a **mixin**, not a parent class. It overrides specific methods and calls
`super()` to chain to the base. Python's MRO ensures the right order.

### Initialization (`_pi_init`)

Sets up all PI state:
- Gains: `_pi_kp`, `_pi_ki`
- Timing: `_pi_min_interval`, `_pi_last_tick`
- State: `_pi_integral` (starts at 0), `_hp_setpoint`, `_desired_temp`
- Feedforward: `_rls_heat`, `_rls_cool` (RLS model objects), `_ff_offset`
- Learning: `_ff_settled_ticks` counter, `_stable_oodb_ticks`, `_observation_buffer`

### The Tick Function (`_pi_tick`)

This is the core algorithm. It runs:
- **On every temperature sensor update** (event-driven, with 60s minimum cooldown)
- **On a timer** every `pi_min_interval` seconds (catches outdoor temp changes)

Here's the flow, simplified:

```python
def _pi_tick(self):
    # 1. Bail if we shouldn't run
    if off or paused or no_desired_temp or no_current_temp:
        return

    # 2. Time normalization
    dt_factor = elapsed_seconds / pi_min_interval

    # 3. Error
    error = desired_temp_C - current_temp_C

    # 4. Feedforward
    ff_offset = get_ff_offset(outdoor_temp, mode)

    # 5. Are we in deadband?
    if abs(error) < deadband:
        p_term = 0
        integral *= 0.9          # Decay
        settled_ticks += 1
        maybe_learn_ff()         # Auto-learn if settled ≥2 ticks
    else:
        settled_ticks = 0
        p_term = Kp * weighted_error
        integral += (error + prev_error)/2 * dt_factor  # Trapezoidal

    # 6. Anti-windup
    integral = clamp(integral, -50, +50)

    # 7. Compute raw setpoint
    raw = desired_temp + p_term + Ki * integral + ff_offset

    # 8. Clamp to AC's min/max range
    clamped = clamp(raw, min_temp, max_temp)

    # 9. Back-calculate if clamped (anti-windup)
    if clamped != raw:
        integral += (clamped - raw) / Ki

    # 10. Hysteresis — only change if delta ≥ 0.5°C
    if abs(clamped - current_hp_setpoint) >= 0.5:
        hp_setpoint = round(clamped)
        send_ir()
```

### Key Anti-Windup Mechanisms

Integral windup is the #1 enemy of PI controllers. This implementation has *four*
defenses:

1. **Hard clamp:** Integral bounded to [-50, +50]
2. **Deadband decay:** Inside deadband, integral decays 10% per tick
3. **Back-calculation:** If setpoint hits AC min/max, integral is adjusted backward
4. **Overshoot reset:** If error crosses zero (room overshot target), integral zeroes

### Sensor Recovery

If the room temperature sensor goes offline:
- **60-second grace period:** PI keeps running with last known value
- **After 60s:** Falls back to feedforward-only (no P or I, just FF offset)
- **When sensor returns:** Full PI resumes immediately

### Method Overrides

The mixin overrides these base methods (using `super()` chaining):

| Method | What the Mixin Does |
|--------|-------------------|
| `_async_sensor_changed` | Triggers PI tick on temp update |
| `_get_ir_temp` | Returns `_hp_setpoint` instead of user target |
| `async_write_ha_state` | Sends dispatcher signal to update PI sensors |
| `async_set_temperature` | Stores `_desired_temp` separately from display temp |
| `extra_state_attributes` | Adds PI diagnostics to entity attributes |

### Exercise
Read `_pi_tick()` line by line. For each section, identify which of the theoretical
concepts from Lesson 5 it implements. Pay special attention to the deadband logic
and the four anti-windup mechanisms.

---

## Lesson 7: Feedforward and Auto-Learning

This is the most novel part of the controller. Most home HVAC PI implementations
don't have feedforward, let alone one that *learns*.

### How Feedforward Works

The outdoor temperature directly affects how hard the AC must work. Feedforward
pre-computes an offset based on outdoor temp and other conditions, so PI doesn't
have to "discover" the needed adjustment through accumulated error.

### RLS Model (`rls_model.py`)

The feedforward uses **Recursive Least Squares** — a multivariate linear model
that learns online from observations:

```
ff_offset = β₀ + β₁ × outdoor_delta + β₂ × solar_proxy + β₃ × boiler + ...
```

- `β₀` = intercept (base offset)
- `β₁` = outdoor delta coefficient (how much colder outdoor → more offset)
- `β₂...βₙ` = model input coefficients (solar, boiler, stove, etc.)

Each coefficient is learned by RLS from settled observations.  The model runs in
normalized feature space (all features scaled to O(1)) with variable forgetting
factor and ridge regularization to prevent covariance collapse.

### Seed Coefficients and Blending

Before the system has enough observations, the FF uses user-configured seeds:

```python
seed_offset = intercept_seed + outdoor_delta_seed × outdoor_delta + ...
rls_offset  = rls.predict(x)
alpha = min(observation_count / 50, 1.0)
ff_offset = (1 - alpha) × seed_offset + alpha × rls_offset
```

Seeds are expert guesses — reasonable starting points that may not be right for
every zone.  As observations accumulate (α → 1.0), the learned model takes over.

### Learning Gate (Deadband)

When the system is settled inside the deadband (< 0.5°C error) for ≥ 4
consecutive ticks with stable room temp and integral, it observes the HP
setpoint that achieved the target temperature:

```python
observed_offset = hp_setpoint - desired_c   # what the plant actually saw
rls.update(features, observed_offset)       # standard RLS update
```

This is a direct input-output observation at the operating point (Ljung,
*System Identification* §7.4).

### Out-of-Deadband (OODB) Learning

Zones with miscalibrated FF models may rarely reach the deadband, creating a
vicious cycle: bad model → room above target → can't learn → model stays bad.

OODB learning breaks this cycle by allowing observations at thermal equilibrium
outside the deadband.  At equilibrium, `hp_setpoint - current_c` tells the RLS
"what offset maintains room temp at current conditions."  This is valid at any
operating point, with bias growing as ~(K_loss/K_hp) × |error|.

Guards:
- Room temperature must be genuinely stable (|dT/dt| < 0.015°C/min)
- Integral must not be actively winding (recent change < 0.5)
- HP setpoint must not be clamped (censored data excluded)
- Distance-proportional settling: 8 + 4×|error°C| minimum ticks

### FF Confidence Scaling

When the integral opposes the FF direction (model prediction is wrong in
sign), the FF is scaled down so the integral has less to fight:

```python
if integral * ff_offset < 0:  # opposing
    model_error = |ki × integral|
    confidence = 1 / (1 + max(0, model_error - 3) / 3)
```

This is EMA-smoothed to prevent limit cycling at integer setpoint boundaries.
When integral and FF agree (both wanting more heat), confidence stays at 1.0 —
the model direction is right and reducing it would worsen an undersized-HP
situation.

### Batch WLS (Offline Analysis)

Every 12 hours, a weighted least squares analysis runs on the accumulated
observation buffer (~300 entries, ~48h).  Currently in **observe-only mode** —
it logs what it would recommend but doesn't modify the model.  This catches
systematic model errors that the real-time gate might miss:

- Filters to near-equilibrium, unclamped observations
- Weights by inverse distance to target (at-target = full weight)
- Compares batch estimate with current RLS coefficients
- Logs WARNING if any coefficient differs by > 20%

### Model Inputs (Suppression and Learning)

Model inputs are external HA entities (boiler, pellet stove, solar proxy) that
affect room temperature.  Each can:
- Suppress FF learning when active (`suppress_learning: true`)
- Have per-input lag filters (exponential smoothing)
- Have coefficient clamps (physical bounds)

### Exercise
Imagine outdoor is 0°C and desired is 20°C (= 20.5 in this example).  The RLS
model predicts ff_offset = 3.0.  The integral is at -2.0 (ki=0.15, so ki×I = -0.3).
What is the raw setpoint?  What HP setpoint does the AC get?  If the room is
at 20.0°C (in deadband), does the learning gate open?

---

## Lesson 8: The Fujitsu Subclass

**File:** `fujitsu.py` (~270 lines)

This file demonstrates how vendor-specific behavior layers on top of the generic base.

### Why a Subclass?

Fujitsu mini-splits have features that can't be controlled through the standard IRHVAC
JSON interface:

- **Powerful mode** (20-min turbo boost) — requires a specific raw IR code
- **Economy mode** — requires a different raw IR code
- **Min Heat** (10°C maintenance mode) — requires yet another raw IR code
- **Vane cycling** (Set Vertical/Horizontal) — raw IR, not in the protocol

The IRremoteESP8266 library that Tasmota uses *can* decode these when received, but
can't *encode* them for most Fujitsu models.

### Raw IR Codes

```python
FUJITSU_IR_POWERFUL = "0x146300101039C6"  # 56-bit special command
FUJITSU_IR_ECONO    = "0x146300101009F6"
FUJITSU_IR_MIN_HEAT = "0x1463001010FE09..."  # Full 128-bit command
```

These are sent via `cmnd/{device}/irsend` (not `cmnd/{device}/irhvac`).

### Preset Implementation

```python
class FujitsuTasmotaIrhvac(TasmotaIrhvac):  # or PIControllerMixin → TasmotaIrhvac

    async def async_set_preset_mode(self, preset_mode):
        if preset_mode == PRESET_POWERFUL:
            # Save current state (so we can restore later)
            self._save_state()
            # Send raw IR
            await self._send_raw_ir(FUJITSU_IR_POWERFUL)
            # Track preset state
            self._powerful = True
            # Pause PI (if enabled)
            self.pi_pause()
            # Auto-clear after 20 minutes
            self._schedule_clear(1200, self._clear_powerful)
```

### State Detection (Inbound)

When someone uses the physical remote to activate Powerful mode, the Tasmota device
receives the IR and publishes it. The Fujitsu subclass detects it:

```python
def _handle_state_payload(self, payload, raw):
    # Check for 56-bit special commands
    if payload.get("Bits") == 56:
        data = payload.get("Data", "")
        if data == "0x146300101039C6":
            self._powerful = True
            self._attr_preset_mode = PRESET_POWERFUL
            return  # Don't process as normal state update
    # Otherwise, pass to parent for normal handling
    super()._handle_state_payload(payload, raw)
```

### PI Interaction

When a preset is active, PI is *paused*:
- Integral continues to exist but doesn't accumulate
- When preset clears, PI resumes with its existing state
- Min Heat is special — it also *resets* the integral (since the AC is in a
  fundamentally different operating mode)

### Exercise
Read `fujitsu.py` and identify: what happens when a user activates Powerful, then
switches to Econo before Powerful's 20-minute timer expires? Trace the code path.

---

## Lesson 9: Config Flow and Options

**File:** `config_flow.py` (~998 lines)

This file handles the UI for setting up and modifying the integration. It's the
longest file after `climate.py`, but much of it is form definitions.

### Initial Setup (4 Steps)

```
Step 1: async_step_user()         → Device basics (name, vendor, MQTT topics)
Step 2: async_step_climate()      → Temperature, modes, fans, swings
Step 3: async_step_advanced()     → Sensors, defaults, PI toggle
Step 4: async_step_pi_controller()→ PI parameters (only if PI enabled in step 3)
```

Each step accumulates data in `self._user_input`, and the final step creates the
config entry.

### Options Flow (Menu-Based)

After setup, users can modify settings without reconfiguring:

```
Menu
 ├── MQTT Settings
 ├── Temperature
 ├── Modes
 ├── Default Values
 ├── Sensors
 ├── Advanced Options
 ├── PI Controller
 └── IR Actions (add/remove buttons and presets)
```

Each menu item is its own `async_step_*` method. The base class
`OptionsFlowWithReload` automatically reloads the integration after saving.

### Data Flow

```python
# Setup creates the entry:
self.hass.config_entries.async_create_entry(
    title=name,
    data=connection_keys_only,      # Immutable
    options=everything_else,         # Mutable via options flow
)

# Options flow updates:
self.hass.config_entries.async_update_entry(
    entry,
    options=new_options,
)
# → triggers async_unload_entry() + async_setup_entry() (reload)
```

### SelectSelector Gotcha

HA's `SelectSelector` always returns strings, even for numeric values. The integration
handles this in entity setup:

```python
kp = float(config.get(CONF_PI_KP, DEFAULT_PI_KP))
```

### Exercise
Look at `async_step_ir_actions_add()`. How does the UI validate that an IR code is
well-formed? (Trick question — trace it and see.)

---

## Lesson 10: Buttons, Sensors, and IR Actions

### Buttons (`button.py`)

Two types of buttons:

**Vane Buttons** (Fujitsu-specific):
```python
class SetVerticalVaneButton(ButtonEntity):
    async def async_press(self):
        await climate._send_raw_ir(FUJITSU_IR_SET_VERTICAL)
        climate._update_swing_after_vane_press("vertical")
```

**IR Action Buttons** (user-defined):
```python
class IRActionButton(ButtonEntity):
    async def async_press(self):
        await mqtt.async_publish(hass, command_topic, self._ir_code)
```

### Sensors (`sensor.py`)

Three diagnostic sensors, only created when PI is enabled:

| Sensor | What It Shows | Why It Matters |
|--------|-------------|---------------|
| `hp_setpoint` | Temperature sent to AC | See PI's actual output |
| `pi_integral` | Accumulated integral term | Debug windup issues |
| `ff_offset` | Current feedforward contribution | Verify FF learning |

These update via HA's **dispatcher** mechanism — the climate entity fires a signal,
and sensors listen for it:

```python
# In climate (sender):
async_dispatcher_send(self.hass, SIGNAL_PI_UPDATE.format(entry_id))

# In sensor (receiver):
async_dispatcher_connect(hass, SIGNAL_PI_UPDATE.format(entry_id), self._update)
```

### IR Actions Framework

Users can define custom IR commands through the options flow without touching code:

```python
# Stored in config entry options:
ir_actions = [
    {
        "name": "Boost Cooling",
        "type": "button",          # Creates a button entity
        "ir_code": "raw,0,3324,...",
    },
    {
        "name": "Night Mode",
        "type": "preset",          # Appears in preset dropdown
        "ir_code": "raw,0,4480,...",
        "exit_ir_code": "raw,0,...",  # Sent when leaving preset
        "auto_clear_seconds": 28800,  # 8 hours
        "pause_pi": True,
    },
]
```

**Button type:** Fire-and-forget. Press → send IR → done.

**Preset type:** Stateful. Select → send IR → track state → optionally auto-clear
after timeout → send exit IR code when deactivated.

---

## Lesson 11: State Restoration and Resilience

### RestoreEntity

All entities inherit from `RestoreEntity`, which saves state to disk. On restart:

```python
async def async_added_to_hass(self):
    old_state = await self.async_get_last_state()
    if old_state:
        self._attr_hvac_mode = old_state.state
        self._attr_target_temperature = old_state.attributes.get("temperature")
        # ... restore everything
        # Including PI state:
        self._pi_integral = old_state.attributes.get("pi_integral", 0)
        self._ff_heat_buckets = old_state.attributes.get("ff_heat_buckets", {})
```

This means:
- FF learning persists across restarts (buckets survive)
- PI integral survives (no cold-start overshoot)
- All AC settings survive (no "AC turns on in wrong mode" after restart)

### Power Sensor Integration

Optional — if configured, syncs HA state with physical AC state:

```
Physical remote turns AC on → Power sensor goes ON → HA turns entity ON
Physical remote turns AC off → Power sensor goes OFF → HA turns entity OFF
```

### Availability

The Tasmota device publishes `Online`/`Offline` on its LWT (Last Will and Testament)
topic. The integration marks the entity as unavailable when offline.

---

## Lesson 12: End-to-End Walkthroughs

### Walkthrough 1: User Sets Temperature to 72°F

```
1. User drags slider to 72°F in HA UI
2. HA calls async_set_temperature(temperature=72)
3. Integration converts: 72°F = 22.2°C
4. PI mixin stores _desired_temp = 22.2°C
5. PI tick runs:
   - Current room: 20°C, outdoor: 0°C
   - Error: 22.2 - 20 = 2.2°C (outside deadband)
   - P_term: 1.0 × 2.2 = 2.2°C
   - I_term: 0.02 × integral (small, just started)
   - FF_offset: heat_bucket[0] = 3.0°C
   - Raw setpoint: 22.2 + 2.2 + 0.1 + 3.0 = 27.5°C
   - Rounded and clamped: 28°C
6. send_ir() builds JSON with Temp: 28
7. MQTT publishes to cmnd/ir_living/irhvac
8. Tasmota sends IR to AC: "heat to 28°C"
9. AC thinks it should heat to 28°C
10. Room warms toward 22°C (the actual target)
11. As room warms, error shrinks, PI reduces offset
12. Eventually settles in deadband, PI relaxes
```

### Walkthrough 2: Outdoor Temperature Drops

```
1. Outdoor sensor: 5°C → -5°C (cold front)
2. PI timer tick fires (room temp sensor unchanged)
3. FF lookup: heat_bucket[-6] = 5.1°C (vs previous 1.5°C at bucket[6])
4. FF offset increases by ~3.6°C
5. PI proactively raises AC setpoint
6. AC starts heating harder BEFORE the room cools
7. Room temperature barely dips (feedforward prevented the error)
```

### Walkthrough 3: Powerful Preset Activated

```
1. User selects "Powerful" preset in HA UI
2. FujitsuTasmotaIrhvac.async_set_preset_mode("Powerful")
3. Current state saved (mode, temp, fan, swing)
4. Raw IR sent: cmnd/ir_living/irsend → "0x146300101039C6"
5. PI paused (pi_pause())
6. Timer scheduled: 20 minutes
7. AC blasts at max power
8. ... 20 minutes later ...
9. _clear_powerful() fires
10. Saved state restored (mode, temp, fan, swing)
11. Normal IR command sent via send_ir()
12. PI resumed (pi_resume())
13. PI tick runs, adjusts setpoint based on current conditions
```

---

## Common Gotchas

1. **Temperature units are tricky.** HA might display °F, but PI works in °C
   internally. The `_celsius` config flag controls what the *IR protocol* uses.
   These are three different things.

2. **MRO matters.** If you add a method to `PIControllerMixin`, make sure `super()`
   chains correctly. Print `YourClass.__mro__` if confused.

3. **MQTT is async and lossy.** The AC might not acknowledge every command. The
   integration uses `StateMode: "SendStore"` so Tasmota at least remembers what
   was sent.

4. **Integral windup is real.** If you change PI gains, the existing integral doesn't
   automatically adjust. A large accumulated integral can take a long time to decay.
   Use the `ff_buckets_reset` service if things get weird.

5. **Climate entity must exist before sensors/buttons.** The `__init__.py` forwards
   platforms in order: climate first, then sensor and button. This is because sensors
   reference the climate entity via `hass.data`.

6. **Config flow stores strings.** `SelectSelector` returns `"1.0"` not `1.0`.
   Always cast in entity setup.

7. **56-bit vs 128-bit Fujitsu commands.** Regular AC commands are 128-bit. Special
   commands (Powerful, Econo, vane) are 56-bit. The subclass detects this via the
   `Bits` field to distinguish presets from normal state updates.

---

## Further Reading

### Home Assistant Development
- [HA Developer Docs](https://developers.home-assistant.io/) — the official reference
- [Climate Entity docs](https://developers.home-assistant.io/docs/core/entity/climate) — the interface this implements
- [MQTT in HA](https://www.home-assistant.io/integrations/mqtt/) — how HA's MQTT works

### Tasmota IR
- [Tasmota IR docs](https://tasmota.github.io/docs/Tasmota-IR/) — how Tasmota handles IR
- [IRremoteESP8266](https://github.com/crankyoldgit/IRremoteESP8266) — the library Tasmota uses for IR encoding/decoding
- [Tasmota IRHVAC command](https://tasmota.github.io/docs/Commands/#ir-remote) — the JSON protocol

### Control Theory
- [PID Controller (Wikipedia)](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller) — comprehensive reference
- [Brian Douglas - Control Systems](https://www.youtube.com/playlist?list=PLUMWjy5jgHK1NC52DXXrriwihVrYZKqjk) — excellent YouTube series
- [Practical PID tuning](https://controlguru.com/) — real-world tuning guidance

### This Project's History
- Check `git log --oneline` for the evolution of features
- The simulation tools in `tools/` contain the thermal model used for tuning

---

## Suggested Learning Path

| Day | Focus | Activities |
|-----|-------|-----------|
| 1 | Orientation | Read this doc. Skim all files. Run `git log --oneline -30` |
| 2 | Base entity | Read `const.py` + `climate.py`. Trace `send_ir()` end to end |
| 3 | MQTT | Use MQTT Explorer to watch real traffic. Match to code |
| 4 | PI theory | Watch Brian Douglas videos. Read Wikipedia PID article |
| 5 | PI code | Read `pi_controller.py`. Trace `_pi_tick()` with pen and paper |
| 6 | Feedforward | Understand buckets, learning, seeding. Read simulation code |
| 7 | Vendor layer | Read `fujitsu.py`. Understand preset lifecycle |
| 8 | Config flow | Read `config_flow.py`. Set up a test instance if possible |
| 9 | Extras | Read `button.py`, `sensor.py`. Understand dispatcher pattern |
| 10 | Integration | Do the exercises. Modify something small. Run on real HA |

---

*Generated from the `fujitsu-config-flow` branch. ~4,067 lines of Python across 9 files.*
