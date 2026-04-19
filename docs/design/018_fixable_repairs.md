# Design: Fixable Repairs (`is_fixable=True`)

## Status: DRAFT

## Problem

All 10 HA Repairs issues are currently informational (`is_fixable=False`). For three
of them, the integration already knows the exact fix and could apply it with one click:

| Repair | Fix action | Risk |
|--------|-----------|------|
| `save_seeds` | Call `get_learned_seed_config()` + `async_update_entry()` | Safe — same as pressing the Save Seeds button |
| `slope_divergence` | Update `pi_ff_heat_slope` or `pi_ff_cool_slope` in config to match learned value | Moderate — changes feedforward baseline |
| `high_integral_tuning` | Update `pi_ki` in config to suggested value | Moderate — changes controller responsiveness |

## HA Repairs Fixable API

### Required pieces

1. **`repairs.py`** — HA auto-discovers this file. Must export `async_create_fix_flow()`.
2. **`RepairsFlow`** subclass — implements the multi-step confirmation UI.
3. **`async_create_issue(..., is_fixable=True, data={...})`** — the `data` dict carries
   context from the health check to the repair flow (entry ID, values, mode, etc.).
4. **Strings** — each repair flow step needs `title` + `description` under
   `strings.json > "issues" > <key> > "fix_flow" > "step" > <step_id>`.

### Flow lifecycle

```
User clicks "Fix" in HA UI
  → HA calls async_create_fix_flow(hass, issue_id, data)
  → Factory returns appropriate RepairsFlow subclass
  → async_step_init() → async_step_confirm()
  → User sees confirmation dialog with details
  → User clicks "Submit"
  → async_step_confirm(user_input={}) applies the fix
  → self.async_create_entry(title="", data={}) clears the issue
```

## Design

### `repairs.py` — Factory + Flow classes

```python
# custom_components/tasmota_irhvac/repairs.py

from __future__ import annotations

from homeassistant import data_entry_flow
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant

import voluptuous as vol

from .const import DOMAIN, CONF_PI_FF_HEAT_SLOPE, CONF_PI_FF_COOL_SLOPE, CONF_PI_KI


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    """Route fixable issues to the appropriate repair flow."""
    if data is None:
        data = {}

    repair_type = data.get("repair_type", "")

    if repair_type == "save_seeds":
        return SaveSeedsRepairFlow(data)
    if repair_type == "slope_divergence":
        return SlopeDivergenceRepairFlow(data)
    if repair_type == "high_integral_tuning":
        return HighIntegralTuningRepairFlow(data)

    # Fallback — shouldn't happen, but don't crash
    return UnknownRepairFlow()
```

### Flow 1: Save Seeds

Safest fix — identical to pressing the existing Save Seeds button.

```python
class SaveSeedsRepairFlow(RepairsFlow):
    """Apply learned coefficients as new seeds."""

    def __init__(self, data: dict) -> None:
        self._entry_id: str = data["entry_id"]
        self._coefficient_summary: str = data.get("coefficient_summary", "")

    async def async_step_init(self, user_input=None):
        return await self.async_step_confirm()

    async def async_step_confirm(self, user_input=None):
        if user_input is not None:
            # Apply the fix — same logic as SaveLearnedSeedsButton.async_press()
            entry = self.hass.config_entries.async_get_entry(self._entry_id)
            if entry is None:
                return self.async_abort(reason="entry_not_found")

            from . import DATA_KEY
            climate = self.hass.data.get(DATA_KEY, {}).get(self._entry_id)
            if climate is None or climate._pi is None:
                return self.async_abort(reason="pi_not_available")

            pi = climate._pi
            seed_config = pi.get_learned_seed_config()
            new_options = {**entry.options, **seed_config}
            self.hass.config_entries.async_update_entry(entry, options=new_options)
            pi.apply_saved_seeds()

            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={
                "coefficient_summary": self._coefficient_summary,
            },
        )
```

### Flow 2: Slope Divergence

Updates the configured slope to match the learned value.

```python
class SlopeDivergenceRepairFlow(RepairsFlow):
    """Update configured FF slope to match learned value."""

    def __init__(self, data: dict) -> None:
        self._entry_id: str = data["entry_id"]
        self._mode: str = data["mode"]  # "heat" or "cool"
        self._learned: float = data["learned_slope"]
        self._configured: float = data["configured_slope"]

    async def async_step_init(self, user_input=None):
        return await self.async_step_confirm()

    async def async_step_confirm(self, user_input=None):
        if user_input is not None:
            entry = self.hass.config_entries.async_get_entry(self._entry_id)
            if entry is None:
                return self.async_abort(reason="entry_not_found")

            conf_key = (
                CONF_PI_FF_HEAT_SLOPE if self._mode == "heat"
                else CONF_PI_FF_COOL_SLOPE
            )
            new_options = {**entry.options, conf_key: round(self._learned, 4)}
            self.hass.config_entries.async_update_entry(entry, options=new_options)

            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={
                "mode": self._mode,
                "configured": f"{self._configured:.4f}",
                "learned": f"{self._learned:.4f}",
            },
        )
```

### Flow 3: High Integral Tuning

Updates Ki to the suggested value. This is the most impactful fix — confirmation
dialog should clearly state the change.

```python
class HighIntegralTuningRepairFlow(RepairsFlow):
    """Update Ki to suggested value."""

    def __init__(self, data: dict) -> None:
        self._entry_id: str = data["entry_id"]
        self._current_ki: float = data["current_ki"]
        self._suggested_ki: float = data["suggested_ki"]

    async def async_step_init(self, user_input=None):
        return await self.async_step_confirm()

    async def async_step_confirm(self, user_input=None):
        if user_input is not None:
            entry = self.hass.config_entries.async_get_entry(self._entry_id)
            if entry is None:
                return self.async_abort(reason="entry_not_found")

            new_options = {
                **entry.options,
                CONF_PI_KI: round(self._suggested_ki, 3),
            }
            self.hass.config_entries.async_update_entry(entry, options=new_options)

            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={
                "current_ki": f"{self._current_ki:.3f}",
                "suggested_ki": f"{self._suggested_ki:.3f}",
            },
        )
```

### Fallback flow

```python
class UnknownRepairFlow(RepairsFlow):
    """Fallback for unrecognized fixable issues."""

    async def async_step_init(self, user_input=None):
        return self.async_abort(reason="unknown_issue")
```

### Changes to `__init__.py`

The issue creation in `_check_tuning_health_issues()` must pass `data` and set
`is_fixable=True` for the three fixable issue types.

Current pattern:
```python
ir.async_create_issue(
    hass, DOMAIN, issue_id, is_fixable=False,
    severity=..., translation_key=..., translation_placeholders=...,
)
```

New pattern — the tuple returned by `_check_tuning_health()` gains two fields:

```python
# Old: (issue_id, severity, translation_key, placeholders, should_create)
# New: (issue_id, severity, translation_key, placeholders, should_create, is_fixable, data)
```

In `_check_tuning_health_issues()`:
```python
for issue_id, severity, translation_key, placeholders, should_create, is_fixable, data in issues:
    if should_create:
        ir.async_create_issue(
            hass, DOMAIN, issue_id,
            is_fixable=is_fixable,
            severity=...,
            translation_key=translation_key,
            translation_placeholders=placeholders,
            data=data,
        )
    else:
        ir.async_delete_issue(hass, DOMAIN, issue_id)
```

### Changes to `_check_tuning_health()`

Each `issues.append(...)` call currently builds a 5-tuple. For the three fixable
issues, add `is_fixable=True` and a `data` dict. All others get
`is_fixable=False, data=None`.

**Save seeds:**
```python
data = {
    "repair_type": "save_seeds",
    "entry_id": entry_id,
    "coefficient_summary": coeff_summary,
}
issues.append((issue_id, "warning", key, placeholders, True, True, data))
```

**Slope divergence:**
```python
data = {
    "repair_type": "slope_divergence",
    "entry_id": entry_id,
    "mode": mode,
    "learned_slope": learned,
    "configured_slope": configured,
}
issues.append((issue_id, "warning", key, placeholders, True, True, data))
```

**High integral tuning (sub-case 4 only):**
```python
data = {
    "repair_type": "high_integral_tuning",
    "entry_id": entry_id,
    "current_ki": pi_ki,
    "suggested_ki": suggested_ki,
}
issues.append((issue_id, "warning", key, placeholders, True, True, data))
```

Note: high_integral sub-cases 1-3 (immature, slope_gap, equipment) remain
`is_fixable=False` — they're diagnostic, not actionable.

### Strings additions

Under `"issues"` in `strings.json`, add `"fix_flow"` blocks:

```json
"save_seeds": {
    "title": "...",
    "description": "...",
    "fix_flow": {
        "step": {
            "confirm": {
                "title": "Save learned coefficients",
                "description": "This will update your configured seeds to match the current learned values:\n\n{coefficient_summary}\n\nThe integration will use these as starting points on next restart."
            }
        }
    }
},
"slope_divergence": {
    "title": "...",
    "description": "...",
    "fix_flow": {
        "step": {
            "confirm": {
                "title": "Update {mode} feedforward slope",
                "description": "Update configured {mode} slope from {configured} to {learned}.\n\nThis aligns the feedforward baseline with what the model has learned from your system's behavior."
            }
        }
    }
},
"high_integral_tuning": {
    "title": "...",
    "description": "...",
    "fix_flow": {
        "step": {
            "confirm": {
                "title": "Adjust integral gain (Ki)",
                "description": "Reduce Ki from {current_ki} to {suggested_ki}.\n\nThe current Ki is causing excessive integral correction. Reducing it will make the controller less aggressive but more stable."
            }
        }
    }
}
```

### Config reload after fix

`async_update_entry()` triggers `async_update_listener` which is already registered
(the integration reloads on options change). The PI controller will pick up the new
values on next tick. No explicit reload needed.

## Implementation plan

### Phase 1: Infrastructure (one commit)
1. Create `repairs.py` with factory + `SaveSeedsRepairFlow` only
2. Expand issue tuple to 7-tuple in `_check_tuning_health()` return type
3. Update `_check_tuning_health_issues()` in `__init__.py` to pass `data` and `is_fixable`
4. Add `fix_flow` strings for `save_seeds`
5. Tests: mock the repair flow, verify factory routing, verify save seeds applies correctly

### Phase 2: Slope divergence (one commit)
1. Add `SlopeDivergenceRepairFlow`
2. Wire up data in slope divergence issue creation
3. Add `fix_flow` strings
4. Tests

### Phase 3: High integral tuning (one commit)
1. Add `HighIntegralTuningRepairFlow`
2. Wire up data in high_integral_tuning sub-case only
3. Add `fix_flow` strings
4. Tests

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Ki change destabilizes control | Suggested Ki is conservative (clamped to `[0.01, current_ki]`); user confirms |
| Slope update during active season change | Slope divergence requires 6 sustained cycles — transient drift won't trigger |
| Stale data in repair flow | `data` dict is populated at issue-creation time. If conditions change before user clicks Fix, the flow re-reads the config entry and PI state. Slope/Ki values in `data` are what was recommended, but the actual write uses those values, not live state. This is correct — the recommendation was made with that context. |
| `async_update_entry` side effects | Already used by `SaveLearnedSeedsButton` — proven pattern. Options change listener reloads the entity. |

## Not in scope

- Making other repairs fixable (covariance_collapse, model_drift, etc.) — these are
  diagnostic and don't have a single clear fix action
- Multi-step repair flows with user input (e.g., letting user edit the suggested Ki) —
  adds complexity for little benefit; user can always go to config flow instead
- Undo/rollback — config entry has no built-in undo; user can manually revert via config flow
