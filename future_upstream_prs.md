
---

## Near Term

### 1. Migration Cleanup & Upstream Prep

> All `async_migrate_entry` logic in `__init__.py` and `MINOR_VERSION` bumps in `config_flow.py` are beta-only transitions. No upstream user will have these config entries. Before submitting upstream PRs 2 or 3:
>
> 1. Strip all migration paths from `async_migrate_entry` in `__init__.py`
> 2. Reset `MINOR_VERSION` to 1 in `config_flow.py`
> 3. Remove any dead migration version constants
> 4. Run the full test suite and fix any breakage
> 5. Verify a fresh install still works (no prior config entry)
>
> Do NOT touch PR 1 (config flow, already submitted as hristo-atanasov/Tasmota-IRHVAC#187).

### 2. Upstream PR 2 — Vendor Hooks

> Prepare a clean PR branch off upstream master for the vendor registry / subclass architecture. This is PR 2 of 3 (PR 1 config flow already submitted as #187). The work is done on architecture-rework but needs to be isolated into a clean diff:
>
> 1. Read `vendors/__init__.py`, `vendors/fujitsu.py`, `vendors/electra.py` and the `@register` decorator pattern
> 2. Create a branch off upstream master (hristo-atanasov/Tasmota-IRHVAC)
> 3. Cherry-pick or manually apply ONLY the vendor registry changes — no PI, no subentries, no model inputs
> 4. Strip any PI-specific imports or references from the vendor files
> 5. Ensure existing climate.py uses `get_handler()` at setup
> 6. Run tests, ensure clean diff, write PR description
>
> Goal: upstream gets an extensible vendor hook system without any of our PI/RLS/FF machinery.

### 3. Upstream PR 3 — PI Controller (optional)

> This is the most ambitious upstream PR and may be better as a separate project. Assess feasibility:
>
> 1. Read `project_architecture_decision.md` in memory — we decided to stay in Tasmota-IRHVAC and extract PI
> 2. Identify everything in `pi/` that's Fujitsu-specific vs generic
> 3. Identify config flow options that are PI-specific
> 4. Draft the minimal PI surface area that could be upstreamed: config options, sensor entities, the controller itself
> 5. Assess: is this a clean extraction or would it be better as a standalone HACS integration that depends on Tasmota-IRHVAC?
>
> Output a decision document, not code. We need to decide the approach before writing anything.

---