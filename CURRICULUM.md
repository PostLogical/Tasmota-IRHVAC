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
- It supports vendor-specific features like Fujitsu's Boost/Eco/Min Heat presets
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
├── __init__.py           (234 lines)  — Entry point: setup, services, migration
├── const.py              (255 lines)  — All constants, defaults, config keys
├── config_model.py       (211 lines)  — Typed, frozen config dataclass (parsed once)
├── climate.py           (1899 lines)  — Base climate entity: MQTT, state, IR commands
├── config_flow.py       (1399 lines)  — Setup wizard + options flow UI
├── sensor.py             (378 lines)  — 18 PI diagnostic sensors + health sensor
├── binary_sensor.py      (194 lines)  — FF learning suppression status
├── button.py             (281 lines)  — Vane buttons + user-defined IR action buttons
├── diagnostics.py         (59 lines)  — HA diagnostics dump
├── pi/                                — PI + feedforward controller subpackage
│   ├── __init__.py                    — Public API: PIController, NullController, BatchResult
│   ├── pi_controller.py (2661 lines)  — Core PI tick, anti-windup, learning gates
│   ├── pi_stored_data.py  (106 lines) — ExtraStoredData for cross-restart persistence
│   ├── controller_protocol.py (196)   — Protocol class + NullController stub
│   ├── rls_model.py       (249 lines) — Recursive Least Squares with forgetting + ridge
│   ├── batch_learning.py (1053 lines) — Diversity-aware buffer, WLS, residual analysis
│   ├── smith_predictor.py (113 lines) — FOPDT Smith predictor (delay compensation)
│   ├── tau_estimator.py   (231 lines) — Online τ estimation + IMC gain scheduling
│   ├── model_input_manager.py (134)   — External HA entity feature management
│   ├── supplemental_controller.py (136) — Supplemental heat source coordination
│   ├── performance_metrics.py (135)   — ITAE, CVH, FF load fraction accumulators
│   └── health_checks.py  (556 lines)  — Health checks + 9 HA Repairs diagnostics
├── vendors/                           — Vendor handler registry (composition, not inheritance)
│   ├── __init__.py       (153 lines)  — Registry: maps vendor strings to handlers
│   ├── base.py           (168 lines)  — VendorHandler base + IRDecode/EntityState types
│   ├── fujitsu.py        (393 lines)  — Fujitsu presets, raw IR, vane cycling
│   └── electra.py         (49 lines)  — Electra vendor quirks
├── manifest.json                      — Integration metadata (dependencies, version)
├── strings.json                       — UI text (English, with formatjs syntax)
└── translations/
    └── en.json                        — Compiled English translations
```

**Read order for learning:** `const.py` → `climate.py` → `pi/pi_controller.py` →
`pi/rls_model.py` → `vendors/fujitsu.py` → `config_flow.py` → everything else.

---

## Lesson 1: Constants and Configuration

**File:** `const.py` (~255 lines)

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

**File:** `climate.py` (~1899 lines)

This is the largest file and the heart of the integration. It implements HA's
`ClimateEntity` interface — the standard API that makes this show up as a thermostat
in the UI.

### Architecture: Composition, Not Inheritance

The earlier mixin/subclass hierarchy has been replaced with **composition**:

```python
class TasmotaIrhvac(RestoreEntity, ClimateEntity):
    """Single climate entity — works for any vendor."""
    _controller: PIController | NullController   # PI logic (composition)
    _vendor: VendorHandler                        # Vendor-specific behavior (composition)
```

- **`PIController`** (in `pi/pi_controller.py`) is a standalone object, not a mixin.
  The climate entity delegates PI decisions to it and owns all I/O (MQTT, HA state).
- **`VendorHandler`** (in `vendors/`) uses a registry pattern. `vendors/__init__.py`
  maps vendor strings (e.g., `"FUJITSU_AC"`) to handler classes. Unknown vendors get
  a default pass-through handler — no user is ever locked out.
- **`NullController`** is a no-op stub that satisfies the same protocol as
  `PIController`, so climate.py never has `if pi_enabled:` branches.

This design means PI knows nothing about Fujitsu, Fujitsu knows nothing about PI,
and climate.py orchestrates both without subclass MRO complexity.

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
# Fujitsu Boost preset — via vendor handler callback
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
- Integration runs at full rate (the error is the signal, not throttled)
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

**File:** `pi/pi_controller.py` (~2053 lines)

Now let's see how the theory maps to code.

### Class Structure

```python
class PIController:
    """Standalone controller object — composed into the climate entity."""
```

It's a **composed object**, not a mixin or parent class. The climate entity holds
a `_controller` reference and delegates PI decisions to it. `PIController` never
touches MQTT or HA state directly — it returns values and the climate entity acts
on them. A `NullController` stub (same protocol, all no-ops) is used when PI is
disabled, eliminating `if pi_enabled:` branches in climate.py.

### Initialization

Sets up all PI state:
- Gains: `_pi_kp`, `_pi_ki` (may be overridden by IMC — see below)
- Timing: `_pi_min_interval`, `_pi_last_tick`
- State: `_pi_integral` (starts at 0), `_hp_setpoint`, `_desired_temp`
- Feedforward: `_rls_heat`, `_rls_cool` (RLS model objects), `_ff_offset`
- Smith predictor: `_smith` (FOPDT delay-compensation model)
- IMC: `_tau_estimator` (online τ estimation + gain scheduling)
- Learning: `_ff_settled_ticks`, `_stable_oodb_ticks`, `_observation_buffer`
- Metrics: `_itae_accumulator`, `_comfort_violation_hours`, `_ff_load_fraction`

### The Tick Function (`_pi_tick_inner`)

This is the core algorithm. It runs:
- **On every temperature sensor update** (event-driven, with 60s minimum cooldown)
- **On a timer** every `pi_min_interval` seconds (fallback for outdoor temp changes)

Here's the flow, simplified:

```python
def _pi_tick_inner(self):
    # 1. Bail if we shouldn't run
    if off or paused or no_desired_temp or no_current_temp:
        return

    # 2. Time normalization
    dt_factor = elapsed_seconds / pi_min_interval

    # 3. Low-pass filter on measurement (reduce sensor noise amplified by Kp)
    filtered_temp = ema_filter(raw_temp, sensor_filter_tau)

    # 4. Error (on filtered reading)
    error = desired_temp_C - filtered_temp_C

    # 5. Smith predictor correction (delay compensation)
    smith_correction = smith.get_correction()  # nodelay - delayed model
    # Only apply when correction agrees with error direction
    effective_smith = smith_correction if smith_correction * error > 0 else 0.0

    # 6. Feedforward
    ff_offset = get_ff_offset(outdoor_temp, mode)
    # Confidence scaling: reduce FF when integral opposes it
    ff_offset *= ff_confidence

    # 7. Conditional integration freeze
    if hp_at_limit and error_opposes_actuator:
        freeze_integration = True   # Don't accumulate debt we can't act on

    # 8. Deadband behavior
    if abs(error) < deadband:
        p_term = 0
        # Full-rate integration: the error IS the signal
        integral += (error + prev_error)/2 * dt_factor
        settled_ticks += 1
        maybe_learn_ff()         # IDB gate: learn if settled ≥4 ticks
    else:
        settled_ticks = 0
        # P-term uses Smith-corrected error for anticipation
        p_term = Kp * (weighted_error + effective_smith)
        integral += (error + prev_error)/2 * dt_factor  # Trapezoidal
        maybe_learn_oodb()       # OODB gate: learn at thermal equilibrium

    # 9. Leaky integrator (universal slow decay)
    integral *= 0.9999 ** dt_factor  # ~104-day time constant

    # 10. Compute raw setpoint
    raw = desired_temp + p_term + Ki * integral + ff_offset

    # 11. Clamp to AC's min/max range
    clamped = clamp(raw, min_temp, max_temp)

    # 12. Back-calculation anti-windup (if not already frozen)
    if clamped != raw and not freeze_integration:
        integral = (clamped - p_term - ff_offset) / Ki

    # 13. Quantization-error feedback (prevents 1°C step limit cycles)
    if in_deadband and misalignment between 0.3-0.5°C:
        nudge integral toward integer alignment

    # 14. Hysteresis + dwell time
    if abs(clamped - current_hp_setpoint) >= 0.5:
        if enough_dwell_time or urgent (error > 1°C):
            hp_setpoint = round(clamped)
            request_ir_send()
```

### Anti-Windup Mechanisms

Integral windup is the #1 enemy of PI controllers. This implementation has *four*
defenses:

1. **Conditional integration freeze:** When the HP setpoint is at its physical
   limit *and* the error opposes what the actuator can deliver (e.g., heating at
   min temp with room above target), integration pauses entirely. Prevents
   accumulating integral debt the controller can never act on. Also triggers
   when the HP has **no output** — setpoint below room temp in heating (or
   above in cooling), meaning the compressor is off and the controller has
   no actuator authority (Åström §6.4).

2. **Leaky integrator:** Exponential decay with α=0.9999 per nominal tick
   (~10,000 tick time constant ≈ 104 days). Bounds integral growth universally.
   α=0.9999 was chosen over 0.999 to preserve correction for slow-τ houses.

3. **Back-calculation:** If the clamped setpoint differs from the raw setpoint
   (actuator saturation), the integral is adjusted backward to the value that
   produces the clamped output. Skipped when conditional freeze already applied.

4. **Quantization-error feedback:** Nudges integral to align clamped setpoint
   with integer values, preventing 1°C HP step limit cycles. Only acts in
   deadband when misalignment is 0.3–0.5°C.

### Smith Predictor (Delay Compensation)

**File:** `pi/smith_predictor.py` (~113 lines)

Heat pumps have significant transport delay — you change the setpoint and the
room temperature doesn't respond for 15+ minutes. The Smith predictor compensates:

- Maintains two parallel FOPDT (first-order plus dead time) models:
  - `nodelay`: receives current HP setpoint immediately
  - `delayed`: receives HP setpoint from L minutes ago (ring buffer)
- Correction = `nodelay - delayed` = "pending temperature change in the pipeline"
- Applied **only to the P-term** (integral uses raw error for mismatch robustness)
- Selectively applied: only when correction has same sign as error (prevents
  overshoot fighting when the model is inaccurate)

### IMC Gain Scheduling

**File:** `pi/tau_estimator.py` (~231 lines)

Instead of fixed Kp/Ki, gains are computed from the plant's time constant using
**Internal Model Control** (Skogestad SIMC):

```
Kp = τ / (K_eff × (λ + L))
Ki = Kp / Ti,  where Ti = τ/3
```

- `τ` is estimated online by observing 63.2% step responses
- `λ` (closed-loop speed) defaults to L/3 (configurable via `pi_imc_lambda`)
- `L` = HP response lag (15 min default)
- When τ changes, gains recompute and apply to both PI and Smith predictor

### Sensor Recovery

If the room temperature sensor goes offline:
- **60-second grace period:** PI keeps running with last known value
- **After 60s:** Falls back to feedforward-only (no P or I, just FF offset)
- **When sensor returns:** Full PI resumes immediately

### Interface with Climate Entity

The controller communicates with climate.py through a clean interface:

| Climate Entity Calls | Controller Provides |
|---------------------|-------------------|
| `controller.tick()` | New HP setpoint (or None if no change) |
| `controller.get_hp_setpoint()` | Current computed setpoint |
| `controller.get_diagnostics()` | Dict of all PI state for sensors/attributes |
| `controller.set_desired_temp(t)` | Stores target, triggers recalculation |
| `controller.on_mode_change()` | Resets Smith predictor, adjusts integral |

### Exercise
Read `_pi_tick_inner()` line by line. For each section, identify which of the
theoretical concepts from Lesson 5 it implements. Pay special attention to the
deadband logic, the four anti-windup mechanisms, and how the Smith predictor
correction is selectively applied.

---

## Lesson 7: Feedforward and Auto-Learning

This is the most novel part of the controller. Most home HVAC PI implementations
don't have feedforward, let alone one that *learns*.

### How Feedforward Works

The outdoor temperature directly affects how hard the AC must work. Feedforward
pre-computes an offset based on outdoor temp and other conditions, so PI doesn't
have to "discover" the needed adjustment through accumulated error.

### RLS Model (`pi/rls_model.py`)

The feedforward uses **Recursive Least Squares** — a multivariate linear model
that learns online from observations:

```
ff_offset = β₀ + β₁ × outdoor_delta + β₂ × solar_proxy + β₃ × boiler + ...
```

- `β₀` = intercept (base offset needed regardless of conditions)
- `β₁` = outdoor delta coefficient (how much colder outdoor → more offset)
- `β₂...βₙ` = model input coefficients (solar, boiler, stove, etc.)

Each coefficient is learned by RLS from settled observations. The model runs in
normalized feature space (all features scaled to O(1)) with:
- **Variable forgetting factor** (base λ=0.99, adapts based on residual surprise)
- **Ridge regularization** (δ=1e-4) to prevent covariance collapse
- **Coefficient clamping** with P-matrix zeroing when boundaries hit

### Seed Coefficients and Blending

Before the system has enough observations, the FF uses user-configured seeds:

```python
seed_offset = intercept_seed + outdoor_delta_seed × outdoor_delta + ...
rls_offset  = rls.predict(x)
alpha = min(observation_count / 50, 1.0)
ff_offset = (1 - alpha) × seed_offset + alpha × rls_offset
```

Seeds are expert guesses — reasonable starting points that may not be right for
every zone. As observations accumulate (α → 1.0), the learned model takes over.
If the user edits a seed after the model has learned, the affected coefficient
resets to the new seed with increased uncertainty (P diagonal bump).

### FF Confidence Scaling

When the integral opposes the FF direction (model prediction is wrong in
sign), the FF is scaled down so the integral has less to fight:

```python
if integral * ff_offset < 0:  # opposing
    model_error = |ki × integral|
    confidence = 1 / (1 + max(0, model_error - 3) / 3)
```

Below 3°C model error, full FF trust. Above 3°C, smooth reduction. This is
EMA-smoothed (~10 ticks ≈ 2.5 hours) to prevent limit cycling at integer
setpoint boundaries. When integral and FF agree (both wanting more heat),
confidence stays at 1.0 — the model direction is right and reducing it would
worsen an undersized-HP situation.

### Learning Gate: In-Deadband (IDB)

When the system is settled inside the deadband (< 0.5°C error) for ≥ 4
consecutive ticks with stable room temp and integral, it observes the HP
setpoint that achieved the target temperature:

```python
observed_offset = hp_setpoint - desired_c   # what the plant actually saw
rls.update(features, observed_offset)       # standard RLS update
```

This is a direct input-output observation at the operating point (Ljung,
*System Identification* §7.4). One observation per settled window prevents
over-learning from steady state.

### Learning Gate: Out-of-Deadband (OODB)

Zones with miscalibrated FF models may rarely reach the deadband, creating a
vicious cycle: bad model → room above target → can't learn → model stays bad.

OODB learning breaks this cycle by allowing observations at thermal equilibrium
outside the deadband. At equilibrium, `hp_setpoint - current_c` tells the RLS
"what offset maintains room temp at current conditions." This is valid at any
operating point, with bias growing as ~(K_loss/K_hp) × |error|.

Guards:
- Room temperature must be genuinely stable (|dT/dt| < 0.015°C/min)
- Integral must not be actively winding (recent change < 0.5)
- HP setpoint must not be clamped (censored data excluded)
- Distance-proportional settling: 8 + 4×|error°C| minimum ticks

### Batch WLS (Offline Analysis)

**File:** `pi/batch_learning.py` (~1053 lines)

Twice daily (07:00 and 19:00 local time), a weighted least squares analysis
runs on the accumulated observation buffer. This catches systematic model
errors that the real-time learning gates might miss:

- **Diversity-aware buffer:** ~2000 slots with leverage-scored retention
  (D-optimal design). Old observations are kept if they cover rare operating
  conditions, discarded if they're redundant. Persisted across restarts.
- **Filtering:** Excludes clamped data (including hp-no-output observations
  where the compressor is off) and non-equilibrium observations
  (room rate > threshold)
- **Persistent excitation check:** Holds features with insufficient variance
  (prevents learning from narrow conditions)
- **Robust regression:** 3-sigma Huber outlier exclusion
- **Covariance-weighted blended update:** Kalman-gain fusion with prior std
- **P-aware covariance update:** After applying blended coefficients,
  P[i,i] *= (1 - K_i) so RLS treats the correction as real posterior
  information and doesn't drift back
- **Drift detection:** Tracks per-coefficient correction direction history
  to distinguish systematic drift from noise

Each observation records a `wall_hour` (0-23) for time-of-day analysis.

**Residual time-of-day analysis:** After each batch WLS fit, the system bins
residuals by wall-clock hour and detects contiguous spans where the model
consistently over- or under-predicts. A systematic negative residual in the
afternoon suggests unmodeled solar gain; a positive residual at night suggests
unmodeled heat loss. These patterns surface as HA Repairs recommendations
suggesting which model input to add.

**Multicollinearity detection:** The buffer maintains the forward information
matrix X^TX alongside its inverse. The spectral condition number
κ = √(λ_max/λ_min) is computed via power iteration. When κ exceeds 30
(Belsley, Kuh & Welsch, 1980: moderate multicollinearity), the system flags
which feature pairs are correlated (|r| > 0.7) and warns the user via HA
Repairs. At κ > 100, coefficient estimates are numerically unstable.

The batch result is persisted via `ExtraStoredData` for diagnostics continuity
across restarts.

### Model Inputs (`pi/model_input_manager.py`)

Model inputs are external HA entities (boiler, pellet stove, solar proxy) that
affect room temperature. They appear as additional features in the RLS model.
Each can:
- **Suppress FF learning** when active (`suppress_learning: true`) — prevents
  the RLS from learning during atypical conditions
- Have **per-input lag filters** (exponential smoothing with configurable τ)
- Have **coefficient clamps** (physical bounds, e.g., a stove can only reduce
  heating need, never increase it)
- Have **seed coefficients** (expert initial guess per mode)

See [docs/disturbance_inputs.md](docs/disturbance_inputs.md) for configuration
examples.

### Exercise
Imagine outdoor is 0°C and desired is 20°C. The RLS model predicts
ff_offset = 3.0. The integral is at -2.0 (ki=0.15, so ki×I = -0.3).
Confidence is 1.0 (integral agrees with FF direction). What is the raw
setpoint? What HP setpoint does the AC get? If the room is at 20.0°C
(in deadband), does the IDB learning gate open?

---

## Lesson 8: The Fujitsu Vendor Handler

**File:** `vendors/fujitsu.py` (~393 lines)

This file demonstrates how vendor-specific behavior layers on top of the generic
base using the **composition** pattern.

### Why a Vendor Handler?

Fujitsu mini-splits have features that can't be controlled through the standard IRHVAC
JSON interface:

- **Powerful/Boost mode** (20-min turbo boost) — requires a specific raw IR code
- **Economy/Eco mode** — requires a different raw IR code
- **Min Heat** (10°C maintenance mode) — requires yet another raw IR code
- **Vane cycling** (Set Vertical/Horizontal) — raw IR, not in the protocol

The IRremoteESP8266 library that Tasmota uses *can* decode these when received, but
can't *encode* them for most Fujitsu models.

### The Vendor Registry (`vendors/__init__.py`)

Handlers register themselves via a decorator:

```python
@register("FUJITSU")
class FujitsuHandler(VendorHandler):
    ...
```

At runtime, `get_handler("FUJITSU_AC")` finds `FujitsuHandler` by prefix match.
Unknown vendors get the default pass-through `VendorHandler` — no user is ever
locked out. The climate entity holds one handler instance and calls its hooks.

### Raw IR Codes

```python
FUJITSU_IR_POWERFUL = "raw,0,3324,1574,448,390,1182,..."  # 56-bit timing data
FUJITSU_IR_ECONO    = "raw,0,3324,1574,448,390,1182,..."
FUJITSU_IR_MIN_HEAT = "raw,0,3324,1574,448,390,1182,..."  # Full 128-bit
```

These are sent via `cmnd/{device}/irsend` (not `cmnd/{device}/irhvac`).

### Preset Implementation

The handler uses HA's standard `PRESET_BOOST` and `PRESET_ECO` names (not custom
strings). Presets return a `PresetResult` dataclass that tells the climate entity
what to do — the handler never touches I/O directly:

```python
def handle_preset(self, preset, state, send_raw, schedule_timer):
    if preset == PRESET_BOOST:
        self._saved_entity_state = state     # snapshot for restore
        send_raw(FUJITSU_IR_POWERFUL)        # callback to climate entity
        self._powerful = True
        schedule_timer(1200, self._clear)    # 20-minute auto-clear
        return PresetResult(
            pause_pi=True,
            active_preset=PRESET_BOOST,
        )
```

### State Detection (Inbound)

When someone uses the physical remote to activate Powerful mode, the Tasmota device
receives the IR and publishes it. The handler inspects the `IRDecode`:

```python
def handle_ir_received(self, decode: IRDecode) -> ...:
    if decode.bits == 56:
        if decode.data == FUJITSU_DATA_POWERFUL:
            self._powerful = True
            return PresetResult(active_preset=PRESET_BOOST, pause_pi=True)
    # Return None for normal IRHVAC messages → base entity handles
```

### PI Interaction

When a preset is active, the climate entity pauses PI:
- Integral continues to exist but doesn't accumulate
- When preset clears, PI resumes with its existing state
- Min Heat is special — it also signals an *integral reset* (since the AC is in a
  fundamentally different operating mode at 10°C)

### Exercise
Read `vendors/fujitsu.py` and identify: what happens when a user activates Boost,
then switches to Eco before Boost's 20-minute timer expires? Trace the code path.

---

## Lesson 9: Config Flow and Options

**File:** `config_flow.py` (~1399 lines)

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

18 diagnostic sensors + 1 health sensor, only created when PI is enabled:

| Sensor | What It Shows | Why It Matters |
|--------|-------------|---------------|
| `hp_setpoint` | Temperature sent to AC | See PI's actual output |
| `pi_integral` | Accumulated integral term | Debug windup issues |
| `ff_offset` | Current feedforward contribution | Verify FF learning |
| `integral_convergence` | Rate of convergence | Track settling behavior |
| `itae` | Integral of time-weighted absolute error | Overall performance metric |
| `comfort_violation_hours` | Time outside comfort band | User-facing quality metric |
| `setpoint_changes` | Count of HP setpoint adjustments | Activity indicator |
| `controllable_itae` / `uncontrollable_itae` | ITAE split by controller authority | Separate what PI can fix from what it can't |
| `controllable_cvh` / `uncontrollable_cvh` | CVH split by controller authority | Same split for comfort hours |
| `ff_load_fraction` | FF contribution as fraction of total offset | Model vs. integral balance |
| `batch_model_rms` | Residual RMS from last batch WLS run | Model fit quality |
| `buffer_eligible` / `buffer_total` | Observation buffer fill | Learning data availability |
| `buffer_oldest_age_hours` | Age of oldest buffered observation | Buffer diversity |
| `buffer_leverage_max` | Max leverage score in buffer | D-optimal design health |
| `batch_outliers_excluded` | Outliers removed in last batch | Data quality indicator |

The **health sensor** (`sensor.{device}_health`) is a special ENUM sensor with
states OK / Warning / Critical / Disabled. It runs multiple checks (comfort,
integral magnitude, FF confidence, intercept drift, slope drift, model drift,
feature diversity) and reports the worst status with detailed attributes.

**HA Repairs recommendations** (`health_checks.py`) — 9 tuning diagnostics that
surface as actionable items in the HA Repairs panel:

| Repair | Trigger | What It Means |
|--------|---------|--------------|
| Slope divergence | Learned slope ≠ configured >30% for 6 cycles | Building envelope changed; update seed |
| Save seeds | Model converged, seeds not saved | Checkpoint learned values against data loss |
| High integral (4 sub-causes) | Ki×I > 2°C sustained | Immature model, slope gap, equipment limits, or needs Ki reduction |
| Covariance collapse | P[i,i] ≈ δ at clamp boundary | RLS stuck, batch must correct |
| Model drift | Same-direction batch correction ≥5 cycles | Physical change (insulation, sensor, schedule) |
| Intercept absorbing | |intercept| > 1°C with collapsed coefficient | One coefficient stuck, intercept compensating |
| Batch-online disagreement | Batch corrects same direction, RLS drifts back | Transient vs. structural mismatch |
| Residual pattern | |mean residual| > 0.5°C in contiguous hours, 3 cycles | Unmodeled time-of-day disturbance (solar, occupancy) |
| Multicollinearity | Spectral κ > 30 sustained 3 cycles (Belsley 1980) | Correlated features; RLS can't separate effects |

**Binary sensor** (`binary_sensor.py`): `binary_sensor.{device}_ff_learning_suppressed`
shows whether feedforward learning is currently suppressed, with attributes listing
active entity suppressors and manual suppress status.

All sensors update via HA's **dispatcher** mechanism:

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

### RestoreEntity + ExtraStoredData

All entities inherit from `RestoreEntity`, which saves state to disk. The
integration uses two tiers of persistence:

**Tier 1: ExtraStoredData** (modern, preferred for PI state)

```python
# climate.py exposes PI data for HA's storage:
@property
def extra_restore_state_data(self) -> ExtraStoredData | None:
    return self._controller.get_extra_stored_data()
```

`PIExtraStoredData` (in `pi/pi_stored_data.py`) persists everything the
controller needs across restarts:

| Category | Fields |
|----------|--------|
| Core PI | `pi_integral`, `desired_temp`, `hp_setpoint` |
| RLS models | Full heat/cool model state (coefficients, covariance, obs count) |
| Observation buffer | Diversity-aware buffer with leverage scores |
| Batch learning | Last batch result, drift correction direction history |
| IMC | `tau_estimate` (system time constant) |
| Metrics | ITAE, CVH, convergence, setpoint changes, FF load fraction |
| Config tracking | `ki_at_save`, `heat_seeds_at_learn`, `cool_seeds_at_learn` |
| Model inputs | Lag filter states per input |

On restore, the controller handles configuration changes gracefully:
- If Ki changed since save, integral is rescaled: `integral *= (ki_at_save / current_ki)`
- If model inputs were added/removed, new inputs get seeded (not zeroed)
- If user edited a seed coefficient, the affected RLS coefficient resets to the
  new seed with increased uncertainty (P diagonal bump)
- Tau estimate triggers IMC gain recomputation

**Tier 2: State attributes** (legacy fallback)

For upgrades from pre-ExtraStoredData versions, the controller falls back to
reading `pi_integral`, `desired_temp`, and `hp_setpoint` from entity attributes.
This ensures no learning is lost during the upgrade.

This means:
- RLS learning persists across restarts (full model state survives)
- Observation buffer survives (batch WLS has history immediately)
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
4. Controller stores _desired_temp = 22.2°C
5. PI tick runs:
   - Current room: 20°C, outdoor: 0°C
   - Error: 22.2 - 20 = 2.2°C (outside deadband)
   - Smith correction: +0.5°C (pending heat in pipeline)
   - P_term: Kp × (weighted_error + smith) = 1.0 × 2.7 = 2.7°C
   - I_term: Ki × integral (small, just started) = 0.1°C
   - FF_offset: RLS predicts 3.0°C for outdoor_delta = 22.2°C
   - Raw setpoint: 22.2 + 2.7 + 0.1 + 3.0 = 28.0°C
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
3. RLS model: ff_offset(outdoor_delta=27.2) = 5.1°C (vs previous 2.5°C at delta=17.2)
4. FF offset increases by ~2.6°C
5. PI proactively raises AC setpoint
6. AC starts heating harder BEFORE the room cools
7. Room temperature barely dips (feedforward prevented the error)
```

### Walkthrough 3: Boost Preset Activated

```
1. User selects "Boost" preset in HA UI
2. climate.async_set_preset_mode("boost")
3. Vendor handler saves entity state snapshot
4. Raw IR sent via handler callback: cmnd/ir_living/irsend → raw timing data
5. Handler returns PresetResult(pause_pi=True)
6. Climate entity pauses PI controller
7. Timer scheduled: 20 minutes
8. AC blasts at max power
9. ... 20 minutes later ...
10. Timer callback fires, handler restores saved state
11. Normal IR command sent via send_ir()
12. Climate entity resumes PI controller
13. PI tick runs, adjusts setpoint based on current conditions
```

---

## Common Gotchas

1. **Temperature units are tricky.** HA might display °F, but PI works in °C
   internally. The `_celsius` config flag controls what the *IR protocol* uses.
   These are three different things.

2. **Composition, not inheritance.** The PI controller and vendor handler are
   composed objects, not mixins. Don't subclass them — extend the protocol/base
   class instead.

3. **MQTT is async and lossy.** The AC might not acknowledge every command. The
   integration uses `StateMode: "SendStore"` so Tasmota at least remembers what
   was sent.

4. **Integral windup is real.** If you change PI gains, the existing integral
   is automatically rescaled on restart (via `ki_at_save` tracking). For immediate
   issues, the `reset_ff_seeds` service resets RLS models and integral. The
   `suppress_ff_learning` / `resume_ff_learning` services control learning gates.

5. **Climate entity must exist before sensors/buttons.** The `__init__.py` forwards
   platforms in order: climate first, then sensor, binary_sensor, and button. This
   is because sensors reference the climate entity via `hass.data`.

6. **Config flow stores strings.** `SelectSelector` returns `"1.0"` not `1.0`.
   All config parsing happens in `config_model.py` — a typed, frozen dataclass
   that applies defaults and casts in one place.

7. **56-bit vs 128-bit Fujitsu commands.** Regular AC commands are 128-bit. Special
   commands (Boost, Eco, vane) are 56-bit. The vendor handler detects this via the
   `Bits` field in `IRDecode` to distinguish presets from normal state updates.

8. **ExtraStoredData is the persistence mechanism.** PI state (integral, RLS models,
   observation buffer, tau estimate) persists via `PIExtraStoredData`, not entity
   attributes. Entity attributes still carry diagnostics for display, but
   restoration reads from ExtraStoredData first.

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
| 5 | PI code | Read `pi/pi_controller.py`. Trace `_pi_tick_inner()` with pen and paper |
| 6 | Feedforward | Understand RLS model, learning gates, seeding. Read `pi/rls_model.py` |
| 7 | Vendor layer | Read `vendors/fujitsu.py`. Understand preset lifecycle + registry |
| 8 | Config flow | Read `config_flow.py`. Set up a test instance if possible |
| 9 | Extras | Read `sensor.py`, `button.py`, `pi/batch_learning.py`. Understand dispatcher pattern |
| 10 | Integration | Do the exercises. Modify something small. Run on real HA |

---

*Updated April 2026 from the `architecture-rework` branch. ~11,432 lines of Python across 22 files.*
