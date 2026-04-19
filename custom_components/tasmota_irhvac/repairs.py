"""HA Repairs fix flows for Tasmota IRHVAC."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import data_entry_flow
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant

from .const import DATA_KEY, DOMAIN

_LOGGER = logging.getLogger(__name__)


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

    return UnknownRepairFlow()


class SaveSeedsRepairFlow(RepairsFlow):
    """Apply learned coefficients as new seeds."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with issue data."""
        super().__init__()
        self._entry_id: str = data.get("entry_id", "")
        self._coefficient_summary: str = data.get("coefficient_summary", "")

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Handle init step — delegate to confirm."""
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Confirm and apply learned seeds."""
        if user_input is not None:
            entry = self.hass.config_entries.async_get_entry(self._entry_id)
            if entry is None:
                return self.async_abort(reason="entry_not_found")

            climate = self.hass.data.get(DATA_KEY, {}).get(self._entry_id)
            if climate is None or not hasattr(climate, "_pi") or climate._pi is None:
                return self.async_abort(reason="pi_not_available")

            pi = climate._pi
            if not hasattr(pi, "get_learned_seed_config"):
                return self.async_abort(reason="pi_not_available")

            from .const import (
                CONF_PI_FF_COOL_SLOPE,
                CONF_PI_FF_HEAT_SLOPE,
                CONF_PI_MODEL_INPUTS,
            )

            learned = pi.get_learned_seed_config()
            new_options = dict(entry.options)

            if "heat_slope" in learned:
                new_options[CONF_PI_FF_HEAT_SLOPE] = learned["heat_slope"]
            if "cool_slope" in learned:
                new_options[CONF_PI_FF_COOL_SLOPE] = learned["cool_slope"]

            model_inputs = list(new_options.get(CONF_PI_MODEL_INPUTS, []))
            for i, seeds in enumerate(learned.get("input_seeds", [])):
                if i < len(model_inputs):
                    updated = dict(model_inputs[i])
                    updated.update(seeds)
                    model_inputs[i] = updated
            new_options[CONF_PI_MODEL_INPUTS] = model_inputs

            self.hass.config_entries.async_update_entry(entry, options=new_options)
            pi.apply_saved_seeds()

            _LOGGER.info(
                "Repair flow: saved learned seeds for %s", self._entry_id,
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={
                "coefficient_summary": self._coefficient_summary,
            },
        )


class SlopeDivergenceRepairFlow(RepairsFlow):
    """Update configured FF slope to match learned value."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with issue data."""
        super().__init__()
        self._entry_id: str = data.get("entry_id", "")
        self._mode: str = data.get("mode", "heat")
        self._learned: float = float(data.get("learned_slope", 0.0))
        self._configured: float = float(data.get("configured_slope", 0.0))

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Handle init step — delegate to confirm."""
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Confirm and update slope."""
        if user_input is not None:
            entry = self.hass.config_entries.async_get_entry(self._entry_id)
            if entry is None:
                return self.async_abort(reason="entry_not_found")

            from .const import CONF_PI_FF_COOL_SLOPE, CONF_PI_FF_HEAT_SLOPE

            conf_key = (
                CONF_PI_FF_HEAT_SLOPE if self._mode == "heat"
                else CONF_PI_FF_COOL_SLOPE
            )
            new_options = {**entry.options, conf_key: round(self._learned, 4)}
            self.hass.config_entries.async_update_entry(entry, options=new_options)

            _LOGGER.info(
                "Repair flow: updated %s slope from %.4f to %.4f for %s",
                self._mode, self._configured, self._learned, self._entry_id,
            )
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


class HighIntegralTuningRepairFlow(RepairsFlow):
    """Update Ki to suggested value."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize with issue data."""
        super().__init__()
        self._entry_id: str = data.get("entry_id", "")
        self._current_ki: float = float(data.get("current_ki", 0.0))
        self._suggested_ki: float = float(data.get("suggested_ki", 0.0))

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Handle init step — delegate to confirm."""
        return await self.async_step_confirm()

    async def async_step_confirm(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Confirm and update Ki."""
        if user_input is not None:
            entry = self.hass.config_entries.async_get_entry(self._entry_id)
            if entry is None:
                return self.async_abort(reason="entry_not_found")

            from .const import CONF_PI_KI

            new_options = {**entry.options, CONF_PI_KI: round(self._suggested_ki, 3)}
            self.hass.config_entries.async_update_entry(entry, options=new_options)

            _LOGGER.info(
                "Repair flow: updated Ki from %.3f to %.3f for %s",
                self._current_ki, self._suggested_ki, self._entry_id,
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="confirm",
            data_schema=vol.Schema({}),
            description_placeholders={
                "current_ki": f"{self._current_ki:.3f}",
                "suggested_ki": f"{self._suggested_ki:.3f}",
            },
        )


class UnknownRepairFlow(RepairsFlow):
    """Fallback for unrecognized fixable issues."""

    async def async_step_init(
        self, user_input: dict[str, str] | None = None
    ) -> data_entry_flow.FlowResult:
        """Abort — unknown issue type."""
        return self.async_abort(reason="unknown_issue")
