# Tasmota-IRHVAC Codebase Curriculum

A structured guide for an intelligent developer to understand this Home Assistant
integration — from zero to "I could extend this myself."

**Prerequisites:** Basic Python, vague awareness that Home Assistant exists, willingness
to read ~22,000 lines of code (most of it in the PI subsystem).

**Where this stands:** version `0.19.2-pre50`, branch lineage `architecture-rework →
tick-first-snapshot → bench-validation → bench-id-discrimination → greybox-redesign`.
~22,000 lines of Python across 38 files, 2,382 tests with 100% production-code coverage
on most milestone tags. The integration has matured well past a thermostat — it's now
closer to a building-thermal-identification kit that happens to drive an AC.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [The Physical Setup](#2-the-physical-setup)
3. [How Home Assistant Custom Integrations Work](#3-how-home-assistant-custom-integrations-work)
4. [File Map — What Lives Where](#4-file-map--what-lives-where)
5. [Architecture Evolution — Then vs. Now](#5-architecture-evolution--then-vs-now)
6. [Lesson 1: Constants and Configuration](#lesson-1-constants-and-configuration)
7. [Lesson 2: The Base Climate Entity](#lesson-2-the-base-climate-entity)
8. [Lesson 3: MQTT — The Nervous System](#lesson-3-mqtt--the-nervous-system)
9. [Lesson 4: Sending IR Commands](#lesson-4-sending-ir-commands)
10. [Lesson 5: The PI Controller — Theory](#lesson-5-the-pi-controller--theory)
11. [Lesson 6: The PI Controller — Implementation](#lesson-6-the-pi-controller--implementation)
12. [Lesson 7: Feedforward, Batch Learning, and the Boundary Estimator](#lesson-7-feedforward-batch-learning-and-the-boundary-estimator)
13. [Lesson 8: Plant Identification](#lesson-8-plant-identification)
14. [Lesson 9: Grey-Box Observer, Auto-Perturbation, and the Regime Probe](#lesson-9-grey-box-observer-auto-perturbation-and-the-regime-probe)
15. [Lesson 10: Tick-First Snapshot Architecture](#lesson-10-tick-first-snapshot-architecture)
16. [Lesson 11: The Fujitsu Vendor Handler](#lesson-11-the-fujitsu-vendor-handler)
17. [Lesson 12: Config Flow and Options](#lesson-12-config-flow-and-options)
18. [Lesson 13: Buttons, Sensors, Repairs, and IR Actions](#lesson-13-buttons-sensors-repairs-and-ir-actions)
19. [Lesson 14: State Restoration and Resilience](#lesson-14-state-restoration-and-resilience)
20. [Lesson 15: Service Reference](#lesson-15-service-reference)
21. [Lesson 16: End-to-End Walkthroughs](#lesson-16-end-to-end-walkthroughs)
22. [Common Gotchas](#common-gotchas)
23. [Further Reading](#further-reading)

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
├── __init__.py           (378 lines)  — Entry point: setup, services, migration (MINOR_VERSION=12)
├── const.py              (315 lines)  — All constants, defaults, config keys
├── config_model.py       (211 lines)  — Typed, frozen config dataclass (parsed once)
├── climate.py           (2217 lines)  — Base climate entity: MQTT, state, IR commands
├── config_flow.py       (1501 lines)  — Setup wizard + options flow UI
├── repairs.py            (306 lines)  — HA Repairs fix flows (fixable repairs)
├── sensor.py             (510 lines)  — 18 PI diagnostic sensors + health + learning sensor
├── binary_sensor.py      (173 lines)  — FF learning suppression status (CoordinatorEntity)
├── button.py             (282 lines)  — Vane buttons + user-defined IR action buttons
├── diagnostics.py         (59 lines)  — HA diagnostics dump
├── services.yaml                      — Service schemas (23 services)
├── pi/                                — PI + feedforward + plant ID subpackage
│   ├── __init__.py                    — Public API: PIController, NullController, BatchResult
│   ├── pi_controller.py (5512 lines)  — Core PI tick, anti-windup, learning, CUSUM, regime gate, subsystem toggles
│   ├── pi_stored_data.py  (208 lines) — ExtraStoredData for cross-restart persistence
│   ├── controller_protocol.py (245)   — Protocol class + NullController stub
│   ├── coordinator.py     (98 lines)  — TasmotaIRHVACCoordinator (DataUpdateCoordinator[TickOutput])
│   ├── snapshot.py       (1576 lines) — TickOutput / DiagnosticsBundle / OfflineBundle / TickEvent dataclasses
│   ├── event_log.py       (218 lines) — Persistent JSONL log (daily rotation, gzip rollover)
│   ├── export_bundle.py   (361 lines) — `export_debug_bundle` service implementation
│   ├── rls_model.py       (393 lines) — Recursive Least Squares (predict-only since pre45)
│   ├── batch_learning.py (2493 lines) — Diversity-aware buffer, WLS (+ VIF), residual analysis, sole estimator
│   ├── boundary_estimator.py (737)    — 3-layer Bayesian boundary estimator (HP-on/off transition)
│   ├── regime_probe.py    (556 lines) — Active HP-off probing for boundary detection (PWARX regime switching)
│   ├── greybox_observer.py (1322 lines)— 1R1C + 2R2C energy balance observer (Stage A + Stage B sim-error PEM)
│   ├── greybox_buffer.py  (172 lines) — Greybox-specific observation buffer (separate retention policy)
│   ├── smith_predictor.py (113 lines) — FOPDT Smith predictor (delay compensation)
│   ├── plant_identifier.py (652 lines)— Multi-provider plant ID orchestrator
│   ├── plant_model.py     (138 lines) — Frozen SOPDT data structures (K, θ, τ_fast, τ_slow)
│   ├── auto_perturbation.py (427 lines)— Layer 2.5: ±1°C perturbation state machine + research mode
│   ├── model_input_manager.py (355)   — External HA entity feature management
│   ├── supplemental_controller.py (136) — Supplemental heat source coordination
│   ├── performance_metrics.py (135)   — ITAE, CVH, FF load fraction accumulators
│   ├── health_checks.py  (677 lines)  — Health checks, repairs, CUSUM anomaly detection
│   └── providers/                     — Plant identification methods
│       ├── __init__.py                — Provider package docstring
│       ├── step_response.py (189)     — Layer 1: τ_fast from 63.2% crossing
│       ├── area_method.py   (329)     — Layer 2: τ_slow from step-response tail area (hardened)
│       ├── plant_test.py    (389)     — Layer 3: relay feedback + step-hold active test
│       └── closed_loop.py   (269)     — Cross-check: SOPDT fit to closed-loop data
├── vendors/                           — Vendor handler registry (composition, not inheritance)
│   ├── __init__.py       (153 lines)  — Registry: maps vendor strings to handlers
│   ├── base.py           (168 lines)  — VendorHandler base + IRDecode/EntityState types
│   ├── fujitsu.py        (393 lines)  — Fujitsu presets, raw IR, vane cycling
│   └── electra.py         (49 lines)  — Electra vendor quirks
├── manifest.json                      — Integration metadata (version 0.19.2-pre50, requires scipy)
├── strings.json                       — UI text (English, with formatjs syntax)
└── translations/
    └── en.json                        — Compiled English translations
```

**Read order for learning:** `const.py` → `climate.py` → `pi/snapshot.py` (tick contract) →
`pi/pi_controller.py` → `pi/rls_model.py` + `pi/batch_learning.py` →
`pi/boundary_estimator.py` → `pi/plant_model.py` → `pi/plant_identifier.py` →
`pi/greybox_observer.py` → `vendors/fujitsu.py` → `config_flow.py` → everything else.

---

## 5. Architecture Evolution — Then vs. Now

The integration started life as a thin MQTT-to-HA bridge for IR HVAC. The control layer
has gone through several distinct architectural eras. Knowing where you are on this
timeline makes the code much easier to read — old comments often refer to mechanisms
that have since been replaced.

### Era 1: Mixin Era (early 2025)

`TasmotaIrhvac` was assembled from a stack of mixins (`PIControlMixin`, `FujitsuMixin`,
etc.). Vendor branches lived inside the climate entity. PI gains were fixed.
**No traces remain in the code** — fully retired.

### Era 2: Composition + Online RLS (2025 → early 2026)

Mixins replaced by composition: `PIController` and `VendorHandler` are objects the
climate entity holds. Recursive Least Squares (RLS) learned feedforward coefficients
on every tick that satisfied a learning gate. A diversity-aware observation buffer
fed a twice-daily batch WLS that cross-checked the online RLS.

### Era 3: Tick-First Snapshot (pre45–pre47, 2026-04-30 → 2026-05-04)

Two structural changes landed back-to-back:

1. **Online RLS was removed** (pre45, 8 phased commits). The RLS object survives —
   it's still queried for FF predictions and persists across restarts — but
   `rls.update()` is never called. Batch WLS is now the **sole** coefficient
   estimator. The buffer fully drives learning; RLS is a cached predict-time
   evaluator.

2. **Tick-first snapshot architecture** (pre46, Stages 0–11). Sensors, diagnostics,
   the event log, and the debug bundle all read from a single typed dataclass
   `TickOutput` produced once per tick. `SIGNAL_PI_UPDATE` and `SIGNAL_FF_SUPPRESS_UPDATE`
   were deleted; everything goes through `TasmotaIRHVACCoordinator`
   (HA's `DataUpdateCoordinator`). A `CONTROLLER_RELOAD` `TickEvent` (pre47) marks
   reset boundaries so debug bundles can locate them.

### Era 4: Subsystem Gating + Persistent Logging (pre43–pre50)

Runtime toggles for each subsystem (`control`, `ff`, `batch_wls`, `plant_id`) via
the `set_subsystem` service — no reload, persists across restarts. **Observe-only
mode** disables PI output while keeping every learning subsystem alive (useful
for shadowing). Persistent JSONL event log (`set_event_log_enabled`) and
`export_debug_bundle` service ship full reproducible windows of state.

Other Era-4 additions:
- **Boundary estimator** (`boundary_estimator.py`): three-layer Bayesian estimator
  for the HP-on/off transition point — split-model RSS sweep on the buffer (Layer 1),
  setpoint-change response (Layer 2), and active probe (Layer 3).
- **Regime probe** (`regime_probe.py`): briefly forces HP to minimum setpoint to
  resolve sensor-calibration ambiguity at the boundary.
- **Over-temperature regime gate**: physical-state anti-windup that blocks integral
  growth when the room is above the cooling band even without saturation.
- **Bumpless transfer**: integral re-derives to keep raw-setpoint output continuous
  across batch-β writes, regime exits, and gain updates.
- **`learning_save` / `learning_restore`**: snapshot slots (max 3) for stashing a
  known-good learning state before a renovation, sensor swap, or experiment.

### Era 5: Grey-Box Redesign (current — `greybox-redesign` branch)

The Bench Validation work (Phases 1–4 lite) found that the static WLS was leverage-
clustered and that the 1R1C grey-box failed to estimate `ua_c` from operational data.
The 2R2C upgrade landed with `tau_fast`/`tau_slow` plant-ID feedback. Real-CSV
validation produced 0/119 gates per season (May 2026), prompting a redesign:

- **Stage A**: hard-fix wall parameters; fit air-mode rate coefficients only.
- **Stage B**: forward-simulate over the full trajectory using sim-error PEM with
  HP feedback; estimate wall-mode parameters from perturbation regimes.
- **Cadence-adaptive RMS gate**: tightens with sensor-update rate.
- **Bayesian priors on (mass_ratio, k_w)**: dropped 2026-05-06; reintroduce only
  when grey-box bridges go live and wall params need a deployment-safety pin.

**Status:** in flight. Grey-box β fusion to WLS is **default off** until the redesign
clears its bench gates. Don't trust greybox-derived β in production yet.

### Branches in flight

| Branch | Purpose | Status |
|--------|---------|--------|
| `master` | Last known good (LKG) reference | Lags pre-release tags |
| `tick-first-snapshot` | TickOutput contract + coordinator | 16 commits, not yet merged |
| `bench-validation` | Phase 4 lite credibility envelope | Phases 1–3 + 4a done |
| `bench-id-discrimination` | Real-bench-bug fixes (LB hyper-sensitivity, HP proxy) | DR shifted poor → close |
| `greybox-redesign` | **current branch** — Stage A/B fits | In progress |

The user's "what's next?" reasoning is about *substance*, not branch status. Treat
the current branch as master.

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

**File:** `climate.py` (~2217 lines)

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
5. Construct the `TasmotaIRHVACCoordinator` (when PI is enabled) and bind it to
   the controller. Sensors and binary sensors will subscribe to this coordinator
   instead of the legacy dispatcher signals.

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

**File:** `pi/pi_controller.py` (~5512 lines)

Now let's see how the theory maps to code. The size growth from ~3500 to ~5500 lines
(pre41 → pre50) reflects the tick-first contract, runtime subsystem gating,
the boundary estimator, the regime probe, the over-temperature regime gate, and
bumpless transfer machinery — none of which existed in the early-2026 curriculum.

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
- **Subsystem gates:** `_control_active`, `_pi_ff_enabled`, `_pi_batch_wls_enabled`,
  `_pi_plant_id_enabled` — runtime toggles persisted in options
- Feedforward: `_rls_heat`, `_rls_cool` (RLS model objects, **predict-only** since pre45), `_ff_offset`
- Smith predictor: `_smith` (FOPDT delay-compensation model)
- Plant ID: `_plant_id` (multi-provider SOPDT estimation + IMC gain scheduling)
- Auto-perturbation: `_auto_perturb` (Layer 2.5 state machine, with optional research mode)
- Grey-box: `_greybox` (1R1C + 2R2C energy balance observer)
- Boundary estimator: `_boundary_estimator` (3-layer Bayesian posterior on the HP-on/off transition)
- Regime probe: `_regime_probe` (active probing for boundary detection)
- Over-temperature regime: `_overtemp_regime` (physical-state anti-windup flag)
- Learning: `_ff_settled_ticks`, `_stable_oodb_ticks`, `_observation_buffer`, per-feature frozen state
- Metrics: `_itae_accumulator`, `_comfort_violation_hours`, `_ff_load_fraction`

### The Tick Function (`_pi_tick_inner`)

This is the core algorithm. It runs:
- **On every temperature sensor update** (event-driven, with 60s minimum cooldown)
- **On a timer** every `pi_min_interval` seconds (fallback for outdoor temp changes)

Here's the flow, simplified:

```python
def _pi_tick_inner(self):
    # 0. Passive tick (hvac_mode=OFF)
    if off:
        run _passive_tick()  # temp tracking, plant ID, lag filters — no buffering
        return

    # 1. Bail if we shouldn't run
    if paused or no_desired_temp or no_current_temp:
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

Integral windup is the #1 enemy of PI controllers. This implementation has *five*
defenses (the over-temperature regime gate landed pre46):

1. **Conditional integration freeze:** When the HP setpoint is at its physical
   limit *and* the error opposes what the actuator can deliver (e.g., heating at
   min temp with room above target), integration pauses entirely. Prevents
   accumulating integral debt the controller can never act on. Also triggers
   when the HP has **no output** — setpoint below room temp in heating (or
   above in cooling), meaning the compressor is off and the controller has
   no actuator authority (Åström §6.4).

2. **Over-temperature regime gate (pre46):** A *physical-state* anti-windup,
   distinct from the actuator-saturation-state defenses above. When the room
   is above the cooling band (or below the heating band) by more than the
   `cal_midpoint` hysteresis, the controller declares an "over-temp regime"
   and the integral is forced toward zero rather than continuing to accumulate
   in the wrong direction. Bumpless transfer applies on regime exit so the
   raw-setpoint output stays continuous. Solves the post-solar wind-down
   pathology (room overshoots target on a sunny afternoon, integral
   accumulates negative debt that fights heating that evening).

3. **Leaky integrator:** Exponential decay with α=0.9999 per nominal tick
   (~10,000 tick time constant ≈ 104 days). Bounds integral growth universally.
   α=0.9999 was chosen over 0.999 to preserve correction for slow-τ houses.

4. **Back-calculation:** If the clamped setpoint differs from the raw setpoint
   (actuator saturation), the integral is adjusted backward to the value that
   produces the clamped output. Skipped when conditional freeze already applied.

5. **Quantization-error feedback:** Nudges integral to align clamped setpoint
   with integer values, preventing 1°C HP step limit cycles. Only acts in
   deadband when misalignment is 0.3–0.5°C.

### Bumpless Transfer

Whenever a controller parameter changes mid-run — IMC gain update, batch β write,
regime exit — the integral is re-derived to keep `desired + p + Ki·I + d + ff`
continuous across the change. Same Åström-style back-calc principle as the
saturation case, applied to *every* discontinuous parameter change. Without it,
batch β writes would step the output and the loop would have to re-equilibrate.

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

**Files:** `pi/plant_identifier.py` (~584 lines), `pi/plant_model.py` (~128 lines)

Instead of fixed Kp/Ki, gains are computed from the plant model using
**Internal Model Control** (Skogestad SIMC):

```
Kp = τ_fast / (K_eff × (λ + L))
Ki = Kp / Ti,  where Ti = τ_fast/3
```

- Plant parameters are estimated by the `PlantIdentifier` orchestrator
  (see Lesson 8 for the full multi-provider architecture)
- `τ_fast` (air node time constant) drives Kp scheduling and the Smith predictor
- `τ_slow` (wall/mass node time constant) drives Kp upper-bound scheduling
- `λ` (closed-loop speed) defaults to L/3 (configurable via `pi_imc_lambda`)
- `L` = HP response lag (15 min default)
- When τ_fast changes, gains recompute and apply to both PI and Smith predictor

### Sensor Recovery

If the room temperature sensor goes offline:
- **60-second grace period:** PI keeps running with last known value
- **After 60s:** Falls back to feedforward-only (no P or I, just FF offset)
- **When sensor returns:** Full PI resumes immediately

### Subsystem Gating (`set_subsystem` service, persisted)

Each major subsystem can be toggled at runtime without a reload:

| Subsystem | Attribute | Effect when off |
|-----------|-----------|----------------|
| `control` | `_control_active` | **Observe-only mode** — PI computes everything but never writes the HP setpoint. Useful for shadowing, validation, vacation mode. |
| `ff` | `_pi_ff_enabled` | FF offset forced to 0; RLS no longer queried for predictions |
| `batch_wls` | `_pi_batch_wls_enabled` | Twice-daily WLS doesn't run; coefficients never get updated |
| `plant_id` | `_pi_plant_id_enabled` | Plant identifier doesn't observe step responses or relay tests; gains don't reschedule |

Toggles persist across restarts via options. The controller falls back gracefully when
a subsystem is off (e.g., FF off → uses last-known offset of 0; plant_id off → keeps
the IMC gains it last computed).

### The Over-Temperature Regime Gate

A separate state machine runs alongside `_pi_tick_inner`. It tracks whether
the room is in an "over-temp regime" — meaningfully above the cooling deadband
(or below the heating one) for a sustained window. The regime has hysteresis
on entry/exit (`cal_midpoint` — typically 0.3°C) so single-tick excursions
don't toggle it.

When the regime is active:
- Integral is forced toward zero rather than accumulating
- Bumpless transfer applies on exit so the controller doesn't snap when the
  regime clears

This is *physical-state* anti-windup: it responds to where the room actually is,
not to whether the actuator is saturated. The classic anti-windup mechanisms
(saturation back-calc, conditional freeze) only act when the HP setpoint hits
its rail; they miss the case where the HP is mid-range but the room is way past
target because of unmodeled solar gain.

### Interface with Climate Entity (Tick-First Contract)

Since pre46 (Stages 0–11), the controller exposes a typed contract via
`pi/snapshot.py`:

| Climate Entity Calls | Controller Provides |
|---------------------|-------------------|
| `controller.tick()` | Updates internal state, builds and caches `last_tick: TickOutput` |
| `controller.last_tick` | Read-only access to the most recent `TickOutput` |
| `controller.get_diagnostics()` | Returns a `DiagnosticsBundle` (typed); legacy callers get `to_dict()` |
| `controller.set_desired_temp(t)` | Stores target, triggers recalculation |
| `controller.on_mode_change()` | Resets Smith predictor, adjusts integral |
| `controller.set_subsystem(name, enabled)` | Toggles a runtime subsystem |

Sensors, the binary sensor, and the persistent event log all read from the
`TickOutput` cached on the controller — they don't reach into controller internals.
See Lesson 10 for the full snapshot architecture.

### Exercise
Read `_pi_tick_inner()` line by line. For each section, identify which of the
theoretical concepts from Lesson 5 it implements. Pay special attention to the
deadband logic, the four anti-windup mechanisms, and how the Smith predictor
correction is selectively applied.

---

## Lesson 7: Feedforward, Batch Learning, and the Boundary Estimator

This is the most novel part of the controller. Most home HVAC PI implementations
don't have feedforward, let alone one that *learns*.

> **Architectural change (pre45):** The earlier era used online RLS that updated
> on every settled tick, with batch WLS as a corrective second pass. As of pre45,
> **batch WLS is the sole estimator.** RLS persists as a predict-time evaluator —
> coefficients and covariance are still tracked, restored across restarts, and
> queried for FF predictions, but `rls.update()` is never called. The buffer is
> the single source of truth.

### How Feedforward Works

The outdoor temperature directly affects how hard the AC must work. Feedforward
pre-computes an offset based on outdoor temp and other conditions, so PI doesn't
have to "discover" the needed adjustment through accumulated error.

### RLS Model (`pi/rls_model.py`) — Predict-Only Since pre45

The feedforward uses a multivariate linear model:

```
ff_offset = β₀ + β₁ × outdoor_delta + β₂ × solar_proxy + β₃ × boiler + ...
```

- `β₀` = intercept (base offset needed regardless of conditions)
- `β₁` = outdoor delta coefficient (how much colder outdoor → more offset)
- `β₂...βₙ` = model input coefficients (solar, boiler, stove, etc.)

The class is still called `RLSModel` for git-history continuity, but **it no
longer updates on its own**. Coefficients are written to it only by the batch
WLS solver. The covariance matrix `P` is still updated (P-aware blended apply
zeroes off-diagonals that the batch corrected) so future restarts and any
hypothetical re-introduction of online learning would have a clean handoff.

What's gone since pre45:
- `rls.update()` calls during `_pi_tick_inner`
- κ-gated λ (forgetting-factor inflation)
- Maturity flags
- P-aware step-cap helpers
- Adaptive step-cap helpers
- The `pi_rls_online_enabled` user-facing flag
- Phase 4 dead Repairs about RLS-vs-batch disagreement

What remains:
- Coefficient storage, normalization, predict path
- Persistence across restarts
- Clamp-to-bounds with P-matrix zeroing on boundary hit (relevant when batch
  writes a clamped value)

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

### Observation Admission Gates: IDB and OODB

The buffer admits observations through two gate types — same gates as the old
online-learning era, but they now feed the **buffer** instead of an online RLS
update.

**In-Deadband (IDB):** When the system is settled inside the deadband (< 0.5°C
error) for ≥ 4 consecutive ticks with stable room temp and integral, the
controller admits one observation:

```python
observed_offset = hp_setpoint - desired_c   # what the plant actually saw
buffer.append(features, observed_offset)    # batch will pick this up at 07:00 / 19:00
```

This is a direct input-output observation at the operating point (Ljung,
*System Identification* §7.4). One observation per settled window prevents
over-representation of steady state in the buffer.

**Out-of-Deadband (OODB):** Zones with miscalibrated FF models may rarely
reach the deadband. OODB admission breaks the loop: at thermal equilibrium
outside the deadband, `hp_setpoint - current_c` tells the model
"what offset maintains room temp at current conditions." Valid at any operating
point, with bias growing as ~(K_loss/K_hp) × |error|.

Guards:
- Room temperature must be genuinely stable (|dT/dt| < 0.015°C/min)
- Integral must not be actively winding (recent change < 0.5)
- HP setpoint must not be clamped (censored data excluded)
- Distance-proportional settling: 8 + 4×|error°C| minimum ticks

The pre49 buffer-and-unlock observability work surfaces *why* an observation
was rejected: each tick now carries an `ObservationContext` (admitted/rejected,
which gate, raw values), debug bundles dump live buffer state, and
unlock-evaluation records show full-model gates per feature.

### Batch WLS (Sole Estimator)

**File:** `pi/batch_learning.py` (~2493 lines)

Twice daily (07:00 and 19:00 local time), a weighted least squares analysis
runs on the accumulated observation buffer. **Since pre45, this is the
only thing that updates β.** Catching model errors and adapting to drift
is its job; the online tick path can no longer drift on its own.

- **Raw-readings storage:** Observations store *what the house experienced*
  (raw sensor readings keyed by entity_id) rather than pre-computed feature
  vectors. Feature vectors are built at WLS time from raw_readings + current
  config. This means changing lag_tau, adding/removing model inputs, or
  swapping entities does NOT invalidate the buffer — old observations
  contribute to features they have data for.
- **Hierarchical regression:** When model inputs are added after some
  observations were already collected, WLS uses the full buffer for features
  present in all observations, then a subset for newer features. This avoids
  discarding data just because a feature was added later.
- **Named feature dicts:** Features are keyed by entity_id, not positional
  index. Observations survive model input add/remove without corruption.
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
- **Grey-box fusion:** After WLS, the grey-box 1R1C observer runs on the
  same buffer (see Lesson 9). Its β estimates are optionally fused with
  WLS β via inverse-variance weighting.
- **Dual WLS for per-feature gating:** Each batch cycle runs two WLS solves:
  a *partial model* (frozen features held, matching online RLS) for coefficient
  application, and a *full model* (all features estimated) for unlock evidence.
  The full model's std_err, held_features, and per-feature VIF determine whether
  each frozen feature is identifiable. VIF is computed inside WLS from the
  eligible-only regression data.
- **κ severe-multicollinearity gate:** When κ > 100, the batch refuses to write
  its recommendation — the data geometry can't reliably separate features at
  that point. This replaces the old κ-gated λ adaptation, which is gone with
  online RLS.
- **`rejection_reason` classification (pre50):** Observations rejected at the
  controller-side WLS gate now carry a structured rejection reason
  (`hp_no_output`, `clamped`, `non_equilibrium`, etc.) instead of `None`.
  Surfaces in debug bundles and Repairs.
- **`head_offset` reset target (pre50):** The `learning_reset` service accepts
  `head_offset` as a target, clearing the boundary estimator's cal-band defaults
  for cases where the room-sensor-vs-HP-internal offset has changed (sensor
  swap, HP head replacement, model_input change).

Each observation records a `wall_hour` (0-23) for time-of-day analysis and
an `outdoor_temp_c` for grey-box energy balance fitting.

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

**Buffer exclusion:** The user can exclude observations from a time range via the
anomaly detection repair flow (see HA Repairs below).
`DiversityAwareBuffer.exclude_time_range(start, end)` removes observations by
monotonic timestamp and recomputes the information matrix. This lets the user
purge contaminated data (open windows, cooking events) that would bias the next
batch WLS solve.

The batch result is persisted via `ExtraStoredData` for diagnostics continuity
across restarts.

### Model Inputs (`pi/model_input_manager.py`)

Model inputs are external HA entities (boiler, pellet stove, solar proxy) that
affect room temperature. They appear as additional features in the RLS model.
Each can:
- Be **enabled/disabled** at runtime without losing learned coefficients
  (disabled inputs are excluded from the feature vector but preserved in storage)
- Have an **input role** (`"solar"`, `"supplemental"`, or generic) — the grey-box
  observer uses the `"solar"` role to identify the solar proxy column
- **Suppress FF learning** when active (`suppress_learning: true`) — prevents
  the RLS from learning during atypical conditions
- Have **per-input lag filters** (exponential smoothing with configurable τ)
- Have **coefficient clamps** (physical bounds, e.g., a stove can only reduce
  heating need, never increase it)
- Have **seed coefficients** (expert initial guess per mode)

See [docs/disturbance_inputs.md](docs/disturbance_inputs.md) for configuration
examples.

### Boundary Estimator (`pi/boundary_estimator.py`)

A long-standing pathology: the room-temperature sensor and the HP's internal
sensor disagree by 1–2°C in practice (different mounting, different sensor
parts). When the controller sets `hp_setpoint = current_c + 0.5`, it doesn't
actually know whether the HP's compressor is on or off — the HP is comparing
the setpoint against *its* sensor, not ours.

Pre50 introduced a three-layer Bayesian boundary estimator that learns the
HP-on/off transition point in `(hp_setpoint - current_c)` space:

**Layer 1 — Split-model RSS sweep (every 12h batch):** For each candidate
breakpoint, fit a piecewise model where below the boundary the rate
includes a `k_c × hp_offset` term, above it doesn't. Score = RSS_null − RSS_split:
how much the piecewise model beats envelope-only fitting. The best breakpoint
is where adding `hp_offset` stops helping. No pre-identification of HP-off
data needed — works at any offset from the first batch.

**Layer 2 — Setpoint-change response (per-tick, opportunistic):** When the
HP setpoint changes, the room rate response (or non-response) reveals
whether the HP was affected. Each setpoint change is exogenous (the previous
tick's PI decision), making it a high-quality datum that works at any offset.

**Layer 3 — Active probe (`pi/regime_probe.py`, see Lesson 9):** When the
estimator stalls, briefly forces the HP to minimum setpoint to directly
observe whether the room rate changes. Highest-weight evidence
(literature: PWARX regime-switching, Adams & MacKay 2007 changepoint
detection).

**Bayesian fusion:** Each layer updates a Gaussian posterior `N(μ, σ²)` on
the boundary location. `cal_min`/`cal_max` (the buffer's HP-on/off cal band)
are derived as `μ ± k×σ`. The estimator escalates from passive (Layer 1) to
active (Layer 3) if the stall counter trips. Reset-target `head_offset` in
the `learning_reset` service clears this state.

### Exercise
Imagine outdoor is 0°C and desired is 20°C. The FF model (β cached on the
RLS object, last written by batch WLS) predicts `ff_offset = 3.0`. The
integral is at -2.0 (ki=0.15, so ki×I = -0.3). Confidence is 1.0 (integral
agrees with FF direction). What is the raw setpoint? What HP setpoint does
the AC get? If the room is at 20.0°C (in deadband), is an observation
admitted to the buffer? Trace which `ObservationContext` reason it carries.

---

## Lesson 8: Plant Identification

**Files:** `pi/plant_identifier.py`, `pi/plant_model.py`, `pi/providers/`

The old `tau_estimator.py` has been replaced by a multi-provider plant
identification architecture that separates τ_fast (air time constant) from
τ_slow (wall/mass time constant) and estimates a full SOPDT (Second-Order
Plus Dead Time) plant model.

### The SOPDT Plant Model (`pi/plant_model.py`)

All plant parameters are frozen dataclasses with provenance metadata:

```python
@dataclass(frozen=True)
class PlantEstimate:
    k: ParameterEstimate      # process gain (°C room / °C HP setpoint)
    theta: ParameterEstimate   # dead time (minutes)
    tau_fast: ParameterEstimate  # fast time constant — air node (minutes)
    tau_slow: ParameterEstimate  # slow time constant — wall/mass node (minutes)
```

Each `ParameterEstimate` carries `value`, `confidence` (0.0 = seed → 1.0),
`source` (which provider produced it), and `observations` count. The frozen
design prevents shared-state bugs between providers.

### 4-Layer Identification Architecture

The `PlantIdentifier` orchestrator coordinates four providers, each
specializing in different parameters:

**Layer 1 — Step Response** (`providers/step_response.py`):
- Watches for HP setpoint changes ≥ 1°C
- Measures time to 63.2% of final temperature change → τ_fast
- Same algorithm as the old tau_estimator, adapted to the provider interface
- EMA-smoothed across observations

**Layer 2 — Area Method** (`providers/area_method.py`):
- Continues observing *after* the 63.2% crossing (where Layer 1 stops)
- Integrates the remaining response tail using trapezoidal integration
- For SOPDT: `A = τ_fast + τ_slow + θ`, so `τ_slow = A - τ_fast - θ`
- Noise-robust: integration averages out sensor noise

**Layer 2.5 — Auto-Perturbation** (`pi/auto_perturbation.py`):
- See Lesson 9 for details — generates controlled ±1°C steps to feed
  Layers 1 and 2 when the system lacks natural excitation

**Layer 3 — Relay Test** (`providers/plant_test.py`):
- Active identification: toggles HP setpoint to produce controlled oscillations
- At low setpoint, HP is off (free-fall — pure building physics)
- At high setpoint, HP runs at full power (step response data)
- Produces K_u (ultimate gain) and T_u (ultimate period) from relay feedback
  (Åström & Hägglund, 1984)
- Each half-cycle also feeds the step response + area method providers

**Cross-Check — Closed Loop** (`providers/closed_loop.py`):
- Fits SOPDT model to observed closed-loop data, accounting for PI's effect
  on HP setpoint trajectory
- Time-domain grid search over (τ_fast, τ_slow) with K solved analytically
- No scipy required — runs in milliseconds
- Used to cross-validate the other providers' estimates

### How Identification Feeds the Controller

When any provider produces a new estimate:

```
Provider → ParameterEstimate → PlantIdentifier.check_observation()
    → updated PlantEstimate → GainUpdate(kp, ki, tau_fast, tau_slow)
    → PI controller applies new gains
```

The orchestrator mediates: it won't accept wild estimates (confidence
thresholds, sanity bounds) and maintains the current best `PlantEstimate`
as an immutable snapshot that the controller reads.

### Exercise
The area method needs the step response to complete first (it picks up
where the 63.2% crossing leaves off). Trace how `PlantIdentifier.start_observation()`
fans out to all providers, and how the area method uses Layer 1's τ_fast
to know where to start integrating.

---

## Lesson 9: Grey-Box Observer, Auto-Perturbation, and the Regime Probe

Three subsystems that work alongside the batch WLS to improve model quality
using fundamentally different approaches.

### Grey-Box Energy Balance Observer (`pi/greybox_observer.py` ~1322 lines)

> **Architectural status (May 2026):** This is in active redesign on the
> `greybox-redesign` branch. The previous bench validation pass (Phase 4 lite)
> found 0/119 gates passing per season on real-CSV data. The β-bridge to WLS
> is **default off** in production. Don't trust grey-box-derived coefficients
> downstream until the bench gates clear.

**Key insight:** The static WLS batch can only use equilibrium observations
where the HP is running. The grey-box observer uses *all* data including
HP-off periods, which directly inform the building's thermal characteristics
through derivative data.

#### 1R1C model (Bacher & Madsen, 2011)

```
dT_air/dt = c₀ + ua_c × (T_out - T_air) + k_c × hp_offset + α_c × solar
```

where `ua_c = UA/C`, `k_c = K_hp/C`, `α_c = α_solar/C`. `c₀` absorbs unmodeled
internal gains and measurement bias. Rate coefficients are directly identifiable
from derivative data without the scaling ambiguity of the original parameterization.

#### 2R2C upgrade (#47, mid-April 2026)

The 2R2C model adds a wall mass node with its own time constant:

```
dT_air/dt  = c₀ + ua_c×(T_out - T_air) + k_w×(T_wall - T_air) + k_c×hp + α_c×solar
dT_wall/dt = k_w/mass_ratio × (T_air - T_wall) + (envelope coupling)
```

Two free parameters: `k_w` (air↔wall coupling rate) and `mass_ratio`
(`C_wall / C_air`). The 2R2C version uses scipy `least_squares` with sensible
bounds and (briefly) Bayesian priors on (`mass_ratio`, `k_w`) — those priors
were dropped 2026-05-06 once the redesign showed the rail-pinning was
diagnostic of data limitation, not bound miscalibration (Reynders 2014).

#### The redesign (May 2026, current branch)

After the 2R2C verdict on real data was negative, the team rebuilt the fit:

- **Stage A**: hard-fix wall parameters (`k_w`, `mass_ratio`) at lit-typical
  values; estimate only the air-mode rate coefficients (`c₀`, `ua_c`, `k_c`,
  `α_c`). This is the "robust" fit used as the day-to-day estimate.
- **Stage B**: forward-simulate the 2R2C state-space model over the *full*
  trajectory (sim-error PEM, Ljung), using HP setpoint as feedback rather
  than as exogenous input. Estimate wall-mode parameters from
  perturbation regimes only (where excitation is rich enough).
- **Cadence-adaptive RMS gate**: tightens with sensor-update rate so
  whiteness checks aren't fooled by autocorrelation at sub-τ sampling.
- **Tau widening**: `GATE_MAX_TAU_SLOW` widened to lit-typical residential
  max after empirical evidence that some zones have slower wall modes than
  the prior bound allowed.

Output: a `GreyboxResult` with `ua_c`, `k_c`, `α_c`, `k_w`, `mass_ratio`,
their stds, and gate pass/fail flags. Logged via `log_greybox_result()` and
captured in the `BatchLearningSnapshot`.

#### Steady-state β bridge

When the gates pass, the β bridge maps rate coefficients to WLS-compatible β
by solving `dT/dt = 0`:

```
β₁ = -ua_c/k_c   (outdoor delta coefficient)
β₂ = -α_c/k_c    (solar coefficient)
```

Standard errors propagated via the delta method on the ratio. Inverse-variance
fusion blends grey-box β with WLS β, giving two independent measurement
sources — equilibrium-WLS and dynamic-PEM. The fusion path is gated until the
redesign clears its bench tests.

#### Cross-validation

After fitting, grey-box `τ_eff = 1/ua_c` is compared against `τ_slow` from
plant ID. Agreement builds confidence; disagreement flags model issues.
Visible in diagnostics and the `BatchLearningSnapshot`.

Requires `scipy` (now a hard dependency in `manifest.json` since it's central
to the 2R2C fit). Runs alongside the WLS batch on the same 12h schedule.

### Auto-Perturbation (`pi/auto_perturbation.py` ~427 lines)

**Problem:** Plant identification (Layers 1–2) needs HP setpoint step changes
to observe step responses. In well-tuned systems, setpoint changes are rare
— the system is too stable for its own identification needs.

**Solution:** Layer 2.5 injects controlled ±1°C perturbations while PI stays
in control. Each perturbation is a single direction, alternating on subsequent
cycles for bidirectional data.

**State machine:**
```
IDLE ──(steady 10 min)──> STEP_ACTIVE ──(hold ≥60 min + steady)──>
RESTORE ──(steady 10 min)──> IDLE
```

**Cadence-adaptive steady-state threshold (current branch):** The "system is
steady" check (`room_rate < 0.015 °C/min`) was below the noise floor at
60-second sampling with σ=0.1°C, meaning auto-perturb couldn't fire under
production conditions. The threshold now scales with sensor cadence.

**Convergence gating:** Perturbs frequently when plant ID has low confidence,
backs off as confidence grows, stops entirely above 90% confidence.

**Stall detection:** After 5 cycles without confidence improvement, transitions
to STALLED state and raises an HA Repair suggesting the relay test (Layer 3).

**Research mode (pre48, Path A0):** A user-facing flag
(`pi_auto_perturb_research_mode`, default off) lets an operator opt into
forced perturbation cycles for research purposes — bypasses the convergence
governor while keeping all safety guards.

**Safety guards:**
- Only activates during configured time window (e.g., 10:00–16:00)
- Aborts if HP saturates, supplemental activates, or learning is suppressed
- `perturb_now` service for manual triggering (60 min timeout)
- Counters persist across HA restarts

**Literature:** Radecki & Hencey 2015 (self-excitation for building thermal
estimation), Liu & Gao 2012 (bidirectional step tests), Bouchié et al. 2022
(ISABELE binary heating signals).

### Regime Probe (`pi/regime_probe.py` ~556 lines)

A separate active-probing subsystem dedicated to **boundary detection**, not
parameter identification. When the boundary estimator (Lesson 7) stalls or
needs high-confidence evidence, the regime probe forces the HP setpoint to
its minimum (≥3°C below current room temp, guaranteeing the compressor is
off regardless of sensor offset) and watches the room rate change.

State machine: `IDLE → BASELINE → PROBE_ACTIVE → COMPLETE`. Baselines for
≥3 min before probing, probes for ≥8 min (gives 55% of the rate change to
materialize on a typical air time constant). Cooldown 30 min between probes.

The result feeds back into the boundary estimator as Layer 3 evidence, with
the highest weight (5×) of any evidence source.

**Literature:** PWARX regime-switching models, Cragg (1971) hurdle models,
dead-zone nonlinearity identification.

### Exercise
Consider a zone where the FF model has good β but plant ID confidence is
only 40%. Auto-perturbation injects +1°C. Trace: how does the perturbation
offset interact with the PI setpoint calculation in `_pi_tick_inner()`?
What happens to the P and I terms during the perturbation? When does the
step response provider decide to end its observation?

---

## Lesson 10: Tick-First Snapshot Architecture

**Files:** `pi/snapshot.py` (~1576 lines), `pi/coordinator.py` (~98 lines),
`pi/event_log.py` (~218 lines), `pi/export_bundle.py` (~361 lines)

This lesson didn't exist in the early-2026 curriculum because the architecture
hadn't landed yet. As of pre46 (Stages 0–11), every consumer of PI state —
sensors, binary sensors, diagnostics, the persistent event log, the debug
bundle — reads from a single typed contract instead of reaching into controller
internals.

### Why It Was Built

Pre-pre46, every diagnostic surface called a different controller method
(`get_full_diagnostics()`, `get_buffer_state()`, ad-hoc attribute reads).
Adding a new field meant adding it to multiple places. Sensors triggered
update via `async_dispatcher_send(SIGNAL_PI_UPDATE)`; the binary sensor
listened on `SIGNAL_FF_SUPPRESS_UPDATE`. There was no single "what was the
state at tick T" object — debug bundles couldn't reproducibly capture a
window of behavior.

The fix: build the typed snapshot once per tick, cache it on the controller,
and have everyone read from it.

### The Contract: `TickOutput`

```python
@dataclass(frozen=True, slots=True)
class TickOutput:
    SCHEMA_VERSION: ClassVar[int] = 1

    # Timing + framing
    ts_mono: float
    ts_wall: float
    zone_label: str

    # Live state
    enabled: bool
    paused: bool
    desired_temp: float | None
    hp_setpoint: float | None
    integral: float
    ff_offset: float
    ff_confidence: float
    outdoor_temp: float | None

    # Subsystem snapshots
    rls_model: RLSModelSnapshot | None
    batch_learning: BatchLearningSnapshot | None
    ff_contributions: FFContributionsSnapshot | None
    observation_buffer: ObservationBufferSnapshot | None
    multicollinearity: MulticollinearityStats | None
    health: HealthSnapshot | None
    learning: LearningSnapshot | None
    greybox: GreyboxSnapshot | None
    learning_suppression: LearningSuppressionSnapshot | None
    overtemp_regime: bool
    regime_probe: dict | None
    observation_context: ObservationContext | None
    # ... and more
```

`TickOutput.to_dict()` produces output matching the legacy `get_full_diagnostics()`
shape so existing consumers don't break. New top-level fields are additive.

### `DiagnosticsBundle` and `OfflineBundle`

`DiagnosticsBundle` is the dataclass version of `get_full_diagnostics()`'s
output — used for the HA Diagnostics download. `OfflineBundle` packages
multi-tick history for offline analysis.

### `TasmotaIRHVACCoordinator`

```python
class TasmotaIRHVACCoordinator(DataUpdateCoordinator[TickOutput]):
    ...
```

A standard HA `DataUpdateCoordinator` typed on `TickOutput`. After every PI
tick, the controller calls `coordinator.async_set_updated_data(last_tick)`.
Subscribers (sensors, binary sensors, the event log writer) wake up and read
from `coordinator.data`.

The legacy `SIGNAL_PI_UPDATE` and `SIGNAL_FF_SUPPRESS_UPDATE` dispatcher
signals were **deleted** in Stage 7d. Sensors and binary sensors now inherit
from `CoordinatorEntity[TasmotaIRHVACCoordinator]` and call
`self.coordinator.data` to read state. Removing the dispatcher path eliminated
a class of "the sensor never updated" bugs caused by signal subscription
timing.

### Persistent Event Log (`pi/event_log.py`)

When `set_event_log_enabled` is true, `EventLogWriter` appends every
`TickOutput.to_dict()` to a daily JSONL file:

```
<config>/tasmota_irhvac/log/<zone>_<YYYY-MM-DD>.jsonl
```

Daily rotation; previous-day files gzip on rollover. Disk pressure ~3 GB/year
compressed for 4 zones at 1-min ticks (typical setup). Default off because it
is genuine disk pressure on small HA boxes.

`TickEvent` is a typed dataclass for the *boundary* events that aren't
per-tick state: `BATCH_RUN`, `ANOMALY_DETECTED`, `MODE_CHANGE`,
`SETPOINT_CHANGE_USER`, `MATURITY_GATE`, `LEARNING_SUPPRESSION_CHANGE`,
`AUTO_PERTURB_STATE`, `BOUNDARY_UPDATE`, `CONTROLLER_RELOAD`. Events have
typed payloads (e.g., `BatchRunPayload`, `BoundaryUpdatePayload`) with
schema version on the wrapper.

The `CONTROLLER_RELOAD` event was added pre47 specifically so a debug
bundle could locate reset boundaries — without it, you couldn't tell why
the integral suddenly jumped to zero in the middle of a window.

### Debug Bundle (`pi/export_bundle.py`)

The `export_debug_bundle` service packages a window of tick log + (optional)
HA Recorder history into:

```
<config>/tasmota_irhvac/bundles/<timestamp>_<zone>/
  ├── tick_log.jsonl       # filtered TickOutput entries
  ├── event_log.jsonl      # TickEvent entries
  ├── ha_history.jsonl     # HA Recorder state changes (when requested)
  ├── observation_buffers.jsonl  # live buffer dumps (pre49)
  ├── manifest.json        # version, zone, window, profile filter
  └── README.md            # how to interpret the bundle
```

The service requires the persistent event log to have been running for at
least the requested window — it's a no-op otherwise. Profile filters (`profile`
parameter) trim the bundle to specific subsystems for faster iteration during
debugging.

### Power-User Toggle: `set_debug_capture`

Adds the full RLS P matrix off-diagonals (~200 floats per snapshot) to
diagnostics dumps. Useful for analyzing covariance structure during
multicollinearity investigations. Default off; persists across restarts;
not exposed in the config flow.

### Why This Matters for Reading the Code

When you trace a sensor value, you no longer follow it back to a controller
attribute through dispatcher signals. The path is always:

```
controller._pi_tick_inner()
  → _build_tick_output()   (constructs TickOutput from current state)
  → coordinator.async_set_updated_data(last_tick)
  → sensor reads coordinator.data.<field>
```

This makes the "what was the state at time T" question answerable from
disk (with event log enabled) instead of requiring a live capture.

### Exercise
Run `set_event_log_enabled enabled=true`, wait an hour, then run
`export_debug_bundle window_days=0`. Open the resulting `tick_log.jsonl`
and trace one tick: find the matching `TickOutput` entry, then look at the
contemporaneous `event_log.jsonl` for any boundary events. Verify the
`ts_mono` ordering matches.

---

## Lesson 11: The Fujitsu Vendor Handler

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

## Lesson 12: Config Flow and Options

**File:** `config_flow.py` (~1501 lines)

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

## Lesson 13: Buttons, Sensors, Repairs, and IR Actions

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

The **learning sensor** (`sensor.{device}_learning`) reports the staged learning
state as Learning / Optimizing / Optimized. Attributes include frozen/active
feature lists, observation count, FF confidence, and condition number κ. Model
input features start frozen at cold start and independently unlock when batch
WLS evidence (std_err, VIF, held status) confirms they're identifiable.

**HA Repairs recommendations** (`health_checks.py`, `repairs.py`) — tuning
diagnostics that surface as actionable items in the HA Repairs panel. Three
repairs are **fixable** (the user clicks "Fix" and the integration applies the
change); the rest are informational.

**Fixable repairs** (is_fixable=True, implemented in `repairs.py`):

| Repair | Fix Action | Risk Level |
|--------|-----------|------------|
| Save seeds | Apply learned coefficients as configured seeds | Safe — same as pressing Save Seeds button |
| Slope divergence | Update configured slope to learned value | Moderate — changes feedforward baseline |
| High integral (tuning sub-case) | Reduce Ki to suggested value | Moderate — changes controller responsiveness |

The fix flow is routed by `async_create_fix_flow()` in `repairs.py`, which HA
auto-discovers. Each flow shows a confirmation dialog before applying. The
`_check_tuning_health()` return tuples carry `is_fixable` and a `data` dict
through `__init__.py` to `ir.async_create_issue()`.

**Informational repairs** (is_fixable=False):

| Repair | Trigger | What It Means |
|--------|---------|--------------|
| High integral (3 diagnostic sub-cases) | Ki×I > 2°C sustained | Immature model, slope gap, or equipment limits |
| Covariance collapse | P[i,i] ≈ δ at clamp boundary | RLS stuck, batch must correct |
| Model drift | Same-direction batch correction ≥5 cycles | Physical change (insulation, sensor, schedule) |
| Intercept absorbing | |intercept| > 1°C with collapsed coefficient | One coefficient stuck, intercept compensating |
| Batch-online disagreement | Batch corrects same direction, RLS drifts back | Transient vs. structural mismatch |
| Residual pattern | |mean residual| > 0.5°C in contiguous hours, 3 cycles | Unmodeled time-of-day disturbance (solar, occupancy) |
| Multicollinearity | Spectral κ > 30 sustained 3 cycles (Belsley 1980) | Correlated features; RLS can't separate effects |
| Freeze impact | Batch RMS increased >20% since coefficient was frozen | Frozen coefficient blocking needed adaptation |

**Real-time anomaly detection** (CUSUM, `pi_controller.py`):

A two-sided CUSUM (Page, 1954; Basseville & Nikiforov, 1993) monitors the
prediction residual on every unclamped buffer observation — not just RLS-gated
ticks. This catches unmodeled disturbances (open windows, cooking, space heaters)
that would contaminate the batch WLS regression.

- **Scale:** MAD-based robust estimate (Huber, 1981), σ̂ = 1.4826 × median|rᵢ - median(r)|
- **Parameters:** k=1.0 (dead zone for shifts < 2σ), h=10.0 (ARL₀ ≈ 50,000 ticks)
- **Reset:** Fast initial response (Lucas & Crosier, 1982) — accumulators reset to
  zero after alarm, preventing massive accumulation during prolonged anomalies
- **Cooldown:** 30 minutes wall-clock (not tick-based, since tick rate varies 1-15 min)

When the CUSUM fires, a fixable HA Repair surfaces with two options:
1. **Exclude** — removes the contaminated observations from the buffer
2. **Dismiss** — keeps them

Cause hints are mode-aware: positive residual means heat loss in heating mode but
heat gain in cooling mode (the offset residual's physical meaning depends on
direction).

After 3+ exclusions, an escalation repair suggests adding a model input with a
gate entity or using the `suppress_learning` entity proactively.

**Binary sensor** (`binary_sensor.py`): `binary_sensor.{device}_ff_learning_suppressed`
shows whether feedforward learning is currently suppressed, with attributes listing
active entity suppressors and manual suppress status.

All sensors update via the **coordinator** (since pre46, Stage 7c/7d). The
legacy `SIGNAL_PI_UPDATE` and `SIGNAL_FF_SUPPRESS_UPDATE` dispatcher signals
were deleted:

```python
# All sensors / binary sensors:
class TasmotaIRHVACSensor(CoordinatorEntity[TasmotaIRHVACCoordinator], SensorEntity):
    @property
    def native_value(self):
        return self.coordinator.data.<field>  # TickOutput is the source of truth

# In controller after each tick:
self._coordinator.async_set_updated_data(self.last_tick)
```

This eliminated a class of "the sensor never refreshed" bugs caused by
dispatcher subscription timing during entity setup.

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

## Lesson 14: State Restoration and Resilience

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
| Subsystem gates | `_control_active`, `_pi_ff_enabled`, `_pi_batch_wls_enabled`, `_pi_plant_id_enabled` |
| RLS models | Full heat/cool model state (coefficients, covariance, obs count) — predict-only since pre45 |
| Observation buffer | Diversity-aware buffer with leverage scores + raw readings |
| Batch learning | Last batch result, drift correction direction history, grey-box result |
| Plant ID | Full `PlantEstimate` (K, θ, τ_fast, τ_slow with provenance) |
| Auto-perturbation | Cycle counters, stall counter, direction, confidence snapshot, research-mode flag |
| **Boundary estimator** | Bayesian posterior `(μ, σ)`, layer evidence history, stall counter |
| **Regime probe** | Probe state, baseline rate, cooldown timer |
| **Over-temp regime** | Active flag, hysteresis state |
| Grey-box observer | Last `GreyboxResult` (rate coefficients, gate flags) |
| Metrics | ITAE, CVH, convergence, setpoint changes, FF load fraction |
| Config tracking | `ki_at_save`, `heat_seeds_at_learn`, `cool_seeds_at_learn` |
| Model inputs | Lag filter states per input |
| Learning slots | `learning_save` snapshot slots (max 3 named slots) |

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

## Lesson 15: Service Reference

The integration registers 23 services. They split into three groups by audience.

### Vendor passthrough (any user)

| Service | Purpose |
|---------|---------|
| `set_econo`, `set_turbo`, `set_quiet`, `set_clean`, `set_beep`, `set_sleep`, `set_filters`, `set_light`, `set_swingv`, `set_swingh` | Toggle vendor-specific feature flags via IR |

All of these accept `state_mode: StoreOnly | SendStore` (default `SendStore`).

### PI tuning (power user)

| Service | Purpose |
|---------|---------|
| `reset_ff_seeds` | Reset RLS coefficients to configured seeds (and clear observation count) |
| `suppress_ff_learning` / `resume_ff_learning` | Manually gate the buffer admission paths |
| `flush_observation_buffer` | Empty the diversity buffer (forces batch to relearn from scratch) |
| `set_coefficient` | Manually set a specific β coefficient |
| `freeze_coefficient` | Pin a β coefficient at its current value (frozen features survive batch updates) |
| `perturb_now` | Force an auto-perturbation cycle (waits for steady state, 60-min timeout) |
| `learning_reset` | Selectively reset subsystems: `seeds`, `buffers`, `integral`, `plant_id`, `greybox`, **`head_offset`** |
| `learning_save` | Snapshot current learning state to a named slot (max 3 slots) |
| `learning_restore` | Restore learning state from a named slot |
| `set_subsystem` | Toggle a subsystem (`control`, `ff`, `batch_wls`, `plant_id`) at runtime, persisted |

### Debugging (deep power user)

| Service | Purpose |
|---------|---------|
| `set_debug_capture` | Toggle inclusion of full RLS P matrix in diagnostics dumps |
| `set_event_log_enabled` | Toggle persistent JSONL tick log (~3 GB/year/4 zones compressed) |
| `export_debug_bundle` | Package a window of tick log + HA history into a directory |

The "observe-only mode" workflow is `set_subsystem subsystem=control enabled=false` —
PI computes everything but never writes the HP setpoint. Useful for:
- Validating a new build before letting it touch the hardware
- Vacation mode (let the AC's built-in thermostat handle minimal heating)
- Shadow-running across a software upgrade

---

## Lesson 16: End-to-End Walkthroughs

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
   issues, the `learning_reset` service with `targets: [integral]` clears just
   the integral. `learning_reset targets: [seeds, buffers]` is the
   "start from scratch" button.

5. **Online RLS is gone (pre45).** Don't add `rls.update()` calls — the design
   intent is "batch is the sole estimator". If you find yourself wanting online
   adaptation, the buffer + batch path is where to extend. Old comments in code
   may still reference "online RLS" to describe what the *partial-model* WLS
   solve is matching.

6. **Coordinator, not dispatcher.** Don't use `async_dispatcher_send` /
   `async_dispatcher_connect` for PI state. `SIGNAL_PI_UPDATE` and
   `SIGNAL_FF_SUPPRESS_UPDATE` were deleted. Sensors and binary sensors are
   `CoordinatorEntity[TasmotaIRHVACCoordinator]`; new ones should follow that
   pattern.

7. **Climate entity must exist before sensors/buttons.** The `__init__.py` forwards
   platforms in order: climate first, then sensor, binary_sensor, and button. This
   is because sensors reference the climate entity (and its coordinator) via `hass.data`.

8. **Config flow stores strings.** `SelectSelector` returns `"1.0"` not `1.0`.
   All config parsing happens in `config_model.py` — a typed, frozen dataclass
   that applies defaults and casts in one place.

9. **56-bit vs 128-bit Fujitsu commands.** Regular AC commands are 128-bit. Special
   commands (Boost, Eco, vane) are 56-bit. The vendor handler detects this via the
   `Bits` field in `IRDecode` to distinguish presets from normal state updates.

10. **ExtraStoredData is the persistence mechanism.** PI state (integral, RLS models,
    observation buffer, plant estimate, boundary estimator, regime probe, grey-box
    result) persists via `PIExtraStoredData`, not entity attributes. Entity
    attributes still carry diagnostics for display, but restoration reads from
    ExtraStoredData first.

11. **Subsystem toggles persist.** If you `set_subsystem subsystem=control
    enabled=false`, the controller stays in observe-only mode across restarts.
    There is no "reset to factory defaults" toggle — re-enabling is explicit
    via the same service. If a subsystem is unexpectedly off, check the entry
    options and the last `set_subsystem` call.

12. **Scipy is now a hard dependency.** `manifest.json` requires it (was optional
    in early-2026). The 2R2C grey-box fit needs `scipy.optimize.least_squares` and
    `scipy.linalg.expm`; degrading gracefully would mean disabling a load-bearing
    subsystem.

13. **Don't trust grey-box β downstream yet.** The 2R2C verdict on real data
    was negative; the redesign is in flight on the `greybox-redesign` branch.
    The β-bridge to WLS is default off. If you see grey-box numbers in
    diagnostics, treat them as informational.

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

### Building Thermal Identification
- Bacher & Madsen (2011) — "Identifying suitable models for the heat dynamics of buildings" — grey-box RC models, rate coefficient parameterization, forward selection
- Madsen & Holst (1995) — "Estimation of continuous-time models for the heat dynamics of a building" — rate coefficients
- Ljung (1999) — *System Identification: Theory for the User* — sim-error PEM (used in Stage B grey-box redesign), §16.4 cross-validation for model selection
- Åström & Hägglund (1984) — Relay feedback autotuning — the foundation for Layer 3 relay test
- Radecki & Hencey (2015) — Self-excitation for building thermal estimation — auto-perturbation literature basis; "excitation rotates not increases observability" caveat
- Reynders (2014) — Parameter rails at bounds = non-identifiability signal (informed Bayesian-prior retraction)
- Bouchié et al. (2022) — ISABELE binary heating signals — bidirectional perturbation method
- Annex 58 / Annex 71 (IEA EBC) — gold-standard whole-building thermal-parameter identification protocols

### Regime Detection & Boundary Estimation
- Thistlethwaite & Campbell (1960) — Regression discontinuity design (boundary estimator Layer 1)
- Adams & MacKay (2007) — Bayesian online changepoint detection
- Cragg (1971) — Hurdle models (dead-zone nonlinearity in regime probe)
- PWARX (Piecewise Affine ARX) — regime-switching models for HP on/off detection

### Change Detection & Fault Diagnosis
- Basseville & Nikiforov (1993) — *Detection of Abrupt Changes* — the definitive CUSUM reference
- Huber (1981) — *Robust Statistics* — MAD estimator, breakdown points
- Gustafsson (2000) — *Adaptive Filtering and Change Detection* — CUSUM on adaptive filter residuals
- Katipamula & Brambley (2005) — "Methods for fault detection, diagnostics, and prognostics for building systems," *HVAC&R Research*

### Verification & Validation
- Roache (1998) — *Verification and Validation in Computational Science and Engineering* — GCI, Richardson extrapolation (used in bench Phase 3a/3b)
- ASHRAE 140 / BESTEST — building energy simulation reference cases
- BOPTEST — building optimization performance test framework

### This Project's History
- Check `git log --oneline` for the evolution of features
- The simulation tools in `tools/` contain the thermal model used for tuning

---

## Suggested Learning Path

| Day | Focus | Activities |
|-----|-------|-----------|
| 1 | Orientation | Read this doc, especially the Architecture Evolution section. Skim all files. Run `git log --oneline -50` |
| 2 | Base entity | Read `const.py` + `climate.py`. Trace `send_ir()` end to end |
| 3 | MQTT | Use MQTT Explorer to watch real traffic. Match to code |
| 4 | PI theory | Watch Brian Douglas videos. Read Wikipedia PID article. Internalize windup |
| 5 | PI code | Read `pi/pi_controller.py`. Trace `_pi_tick_inner()` with pen and paper. Identify the five anti-windup defenses |
| 6 | Feedforward | Read `pi/rls_model.py` (predict-only) and `pi/batch_learning.py`. Understand admission gates feed the buffer, not RLS |
| 7 | Boundary | Read `pi/boundary_estimator.py` + `pi/regime_probe.py`. Understand the three layers of evidence |
| 8 | Plant ID | Read `pi/plant_model.py`, `pi/plant_identifier.py`, then the providers |
| 9 | Grey-box | Read `pi/greybox_observer.py`. Understand 1R1C, 2R2C, Stage A/B, why β-bridge is gated |
| 10 | Tick contract | Read `pi/snapshot.py`, `pi/coordinator.py`, `pi/event_log.py`, `pi/export_bundle.py` |
| 11 | Vendor layer | Read `vendors/fujitsu.py`. Understand preset lifecycle + registry |
| 12 | Config flow | Read `config_flow.py`. Set up a test instance if possible |
| 13 | Extras | Read `sensor.py`, `binary_sensor.py`, `button.py`, `pi/auto_perturbation.py`. Watch services in action |
| 14 | Integration | Do the exercises. Enable the event log, generate a bundle. Modify something small. Run on real HA |

---

## Update History

- **April 2026** — original curriculum at `architecture-rework` (pre41). ~16,000
  lines across 34 files, 1812 tests.
- **May 2026** — current revision at `greybox-redesign` (pre50). ~22,000 lines
  across 38 files, 2382 tests. Major additions: tick-first snapshot architecture,
  online RLS removal, subsystem gating, observe-only mode, persistent JSONL log,
  debug bundle service, 2R2C grey-box (in redesign), boundary estimator, regime
  probe, over-temperature regime gate, bumpless transfer.
