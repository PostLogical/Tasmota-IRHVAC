"""Phase 4 lite per-zone runner + credibility envelope assembly.

Loads each fit-target zone, runs Bacher-Madsen forward selection on the
train window, and runs held-out one-step Kalman residual diagnostics on
the validate window using the train-fitted parameters.

Output is a CredibilityEnvelope summarising per-zone classifications
(good / close / poor per Leprince 2022) plus identifiability and
residual-diagnostic context. The summary string is suitable for writing
to a memory artifact; producing the memory file is a separate caller
responsibility (run-once, not part of automated tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tests.hvac_bench.empirical.data_loader import (
    TRAIN_WINDOW,
    VALIDATE_WINDOW,
    Window,
    ZoneTelemetry,
    load_fit_zones,
)
from tests.hvac_bench.empirical.forward_selection import (
    Classification,
    ForwardSelectionResult,
    ResidualBatteryResult,
    forward_select,
    residual_battery,
)
from tests.hvac_bench.empirical.rc_model import (
    StateSpace,
    build_1r1c,
    build_2r2c,
    kalman_innovations,
)


# ── Output dataclasses ───────────────────────────────────────────────────


@dataclass
class ZoneCredibility:
    """One zone's Phase 4 lite verdict."""

    zone: str
    n_train_obs: int
    n_validate_obs: int
    train_result: ForwardSelectionResult
    validate_residuals: ResidualBatteryResult | None
    train_rmse_c: float
    validate_rmse_c: float | None
    classification: Classification
    summary: str


@dataclass
class CredibilityEnvelope:
    """Aggregated Phase 4 lite credibility envelope for the bench."""

    bundle_path: Path
    zones: dict[str, ZoneCredibility]
    train_window: Window
    validate_window: Window
    overall_classification: Classification
    summary: str
    rejection_notes: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────


def _zone_inputs(t: ZoneTelemetry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = t.df
    obs = df["room_temp_c"].ffill().bfill().to_numpy()
    inputs = np.column_stack(
        [
            df["outdoor_temp_c_om"].to_numpy(),
            df["q_heat_proxy_w"].to_numpy(),
            df["shortwave_w_m2"].to_numpy(),
        ]
    )
    valid = df["valid"].to_numpy()
    return obs, inputs, valid


def _build_selected_state_space(
    result: ForwardSelectionResult,
    dt: float,
) -> StateSpace:
    if result.selected_model == "1R1C":
        return build_1r1c(result.fit_1r1c.best.params.to_full(), dt=dt)
    assert result.fit_2r2c is not None
    return build_2r2c(result.fit_2r2c.best.params.to_full(), dt=dt)


def _one_step_rmse(innovations: np.ndarray) -> float:
    valid = np.isfinite(innovations)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.mean(innovations[valid] ** 2)))


# ── Runner ───────────────────────────────────────────────────────────────


def run_zone(
    zone_name: str,
    train_telemetry: ZoneTelemetry,
    validate_telemetry: ZoneTelemetry,
    *,
    dt: float = 300.0,
    n_restarts: int = 4,
    seed: int = 0,
) -> ZoneCredibility:
    """Run forward selection on train, residual battery on held-out validate."""
    obs_t, in_t, valid_t = _zone_inputs(train_telemetry)
    train_result = forward_select(
        obs_t,
        in_t,
        valid_t,
        dt=dt,
        n_restarts=n_restarts,
        seed=seed,
    )

    # Train one-step RMSE
    ss = _build_selected_state_space(train_result, dt=dt)
    innov_train, var_train = kalman_innovations(ss, obs_t, in_t, valid_t)
    train_rmse = _one_step_rmse(innov_train)

    # Held-out validation
    obs_v, in_v, valid_v = _zone_inputs(validate_telemetry)
    n_validate_obs = int(np.sum(valid_v))

    validate_residuals: ResidualBatteryResult | None = None
    validate_rmse: float | None = None
    if n_validate_obs > 50:
        innov_v, var_v = kalman_innovations(ss, obs_v, in_v, valid_v)
        try:
            validate_residuals = residual_battery(innov_v, var_v)
        except ValueError:
            validate_residuals = None
        validate_rmse = _one_step_rmse(innov_v)

    classification: Classification
    if validate_residuals is None:
        # Fall back to train classification if validate is too short
        classification = train_result.classification
    elif (
        train_result.classification == "good" and validate_residuals.overall_pass
    ):
        classification = "good"
    elif train_result.classification == "good":
        # Train clean, validate fails → close (regime-shift caveat)
        classification = "close"
    elif (
        train_result.classification == "close"
        and validate_residuals.ljung_box_pass
    ):
        classification = "close"
    else:
        classification = "poor"

    summary = (
        f"{zone_name}: selected={train_result.selected_model}, "
        f"classification={classification}, "
        f"train_rmse={train_rmse:.3f}°C, "
        f"validate_rmse={validate_rmse:.3f}°C"
        if validate_rmse is not None
        else f"{zone_name}: selected={train_result.selected_model}, "
        f"classification={classification}, "
        f"train_rmse={train_rmse:.3f}°C, validate_rmse=N/A"
    )

    return ZoneCredibility(
        zone=zone_name,
        n_train_obs=int(np.sum(valid_t)),
        n_validate_obs=n_validate_obs,
        train_result=train_result,
        validate_residuals=validate_residuals,
        train_rmse_c=train_rmse,
        validate_rmse_c=validate_rmse,
        classification=classification,
        summary=summary,
    )


CLASSIFICATION_RANK = {"good": 2, "close": 1, "poor": 0}


def _aggregate_classification(values: list[Classification]) -> Classification:
    """Worst classification across zones — methodology integrity:
    one zone failing identifiability shrinks the bench's envelope.
    """
    if not values:
        return "poor"
    ranks = [CLASSIFICATION_RANK[v] for v in values]
    min_rank = min(ranks)
    return next(k for k, v in CLASSIFICATION_RANK.items() if v == min_rank)


def run_phase4_lite(
    bundle_path: Path | str,
    *,
    n_restarts: int = 4,
    seed: int = 0,
    dt: float = 300.0,
    train_window: Window = TRAIN_WINDOW,
    validate_window: Window = VALIDATE_WINDOW,
) -> CredibilityEnvelope:
    """Run forward selection + held-out validation for all fit-target zones."""
    bundle = Path(bundle_path)
    train_zones = load_fit_zones(bundle, window=train_window)
    validate_zones = load_fit_zones(bundle, window=validate_window)

    zones: dict[str, ZoneCredibility] = {}
    rejection_notes: list[str] = []
    for name in train_zones:
        cred = run_zone(
            name,
            train_zones[name],
            validate_zones[name],
            dt=dt,
            n_restarts=n_restarts,
            seed=seed,
        )
        zones[name] = cred
        rejection_notes.extend(cred.train_result.rejection_notes)

    overall = _aggregate_classification([z.classification for z in zones.values()])

    summary_lines = [
        f"Phase 4 lite credibility envelope (bench-validation):",
        f"  bundle: {bundle}",
        f"  train: {train_window.label} ({train_window.start.date()} → {train_window.end.date()})",
        f"  validate: {validate_window.label} ({validate_window.start.date()} → {validate_window.end.date()})",
        f"  overall classification: {overall}",
        f"",
        f"Per-zone:",
    ]
    for name, z in zones.items():
        summary_lines.append(f"  {z.summary}")

    return CredibilityEnvelope(
        bundle_path=bundle,
        zones=zones,
        train_window=train_window,
        validate_window=validate_window,
        overall_classification=overall,
        summary="\n".join(summary_lines),
        rejection_notes=rejection_notes,
    )


# ── Memory artifact formatter ────────────────────────────────────────────


def format_envelope_memory(envelope: CredibilityEnvelope) -> str:
    """Format the envelope as Markdown for the credibility-envelope memory file.

    Produces a standalone Markdown document (no frontmatter; caller adds).
    """
    lines: list[str] = []
    lines.append(f"# Phase 4 lite credibility envelope")
    lines.append("")
    lines.append(f"Bundle: `{envelope.bundle_path}`")
    lines.append(
        f"Train window: {envelope.train_window.label} "
        f"({envelope.train_window.start.date()} → {envelope.train_window.end.date()})"
    )
    lines.append(
        f"Validate window: {envelope.validate_window.label} "
        f"({envelope.validate_window.start.date()} → {envelope.validate_window.end.date()})"
    )
    lines.append("")
    lines.append(f"## Overall classification: **{envelope.overall_classification}**")
    lines.append("")
    lines.append(
        "Per Leprince 2022 (Annex 58 forward-selection methodology applied to "
        "247 Dutch residential buildings): expect ~38% good / ~38% close / "
        "~24% poor across population. Per Annex 71 ST3 / Madsen 2021: "
        "Bacher-Madsen-class methods on residential operational telemetry are "
        "documented as 'in principle ±15%, not robust enough for QA-grade.'"
    )
    lines.append("")
    lines.append("## Per-zone results")
    lines.append("")
    for name, z in envelope.zones.items():
        lines.append(f"### {name} — {z.classification}")
        lines.append("")
        lines.append(f"- selected model: **{z.train_result.selected_model}**")
        lines.append(f"- train obs: {z.n_train_obs}, validate obs: {z.n_validate_obs}")
        lines.append(f"- train one-step RMSE: {z.train_rmse_c:.3f}°C")
        if z.validate_rmse_c is not None:
            lines.append(f"- validate one-step RMSE: {z.validate_rmse_c:.3f}°C")
        lines.append(f"- methodology summary: {z.train_result.summary}")
        if z.train_result.fit_2r2c is not None and z.train_result.lr_test is not None:
            lr = z.train_result.lr_test
            lines.append(
                f"- LR test: stat={lr.statistic:.1f}, p={lr.p_value:.3g}, "
                f"accept_2R2C={lr.accept_full}"
            )

        # 1R1C report
        f1 = z.train_result.fit_1r1c.best.params
        id_1 = z.train_result.identifiability_1r1c
        lines.append(
            f"- 1R1C: τ={f1.tau_s/3600:.1f}h, q_scale={f1.q_scale:.3f}, "
            f"σ_v={f1.sigma_v:.3f}°C  "
            f"(rails={id_1.n_at_bound}, cv-fails={id_1.n_failed_cv})"
        )

        # 2R2C report
        if z.train_result.fit_2r2c is not None:
            f2 = z.train_result.fit_2r2c.best.params
            id_2 = z.train_result.identifiability_2r2c
            assert id_2 is not None
            lines.append(
                f"- 2R2C: τ_air={f2.tau_air_s/3600:.2f}h, "
                f"τ_wall={f2.tau_wall_s/3600:.1f}h, "
                f"coupling_ratio={f2.coupling_ratio:.2f}  "
                f"(rails={id_2.n_at_bound}, cv-fails={id_2.n_failed_cv})"
            )

        # Train residuals
        if z.train_result.residuals_1r1c is not None:
            rb = z.train_result.residuals_1r1c
            lines.append(
                f"- train residuals (1R1C): LB p={rb.ljung_box_p:.4g}, "
                f"normal p={rb.normality_p:.4g}, CP nCPBES={rb.cp.nCPBES:.4f}"
            )

        # Validate residuals
        if z.validate_residuals is not None:
            vb = z.validate_residuals
            lines.append(
                f"- validate residuals: LB p={vb.ljung_box_p:.4g}, "
                f"CP nCPBES={vb.cp.nCPBES:.4f}, overall_pass={vb.overall_pass}"
            )

        lines.append("")

    if envelope.rejection_notes:
        lines.append("## Methodology rejection notes")
        lines.append("")
        for n in envelope.rejection_notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)
