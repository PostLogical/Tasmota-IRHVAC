"""Tobit convergence-quality A/B — #40 Session 0b.

Question this answers: would Tobit materially improve convergence
quality (speed + asymptotic accuracy + std_err + recovery) versus the
current rails-dropped WLS path? If yes, proceed to Sessions 1-5. If
no, abandon prompt #40.

Method: synthetic regression data with known β_truth and σ_truth, with
right-censoring applied to a controlled fraction of the latent y values
(MNAR truncation matching the operating-rails pattern). Process data
incrementally in batches; per-batch, fit both rails-dropped WLS and
prototype Tobit on the accumulated buffer. Compare β trajectories.

Why synthetic, not full-stack: a full-stack sim adds confounds
(retrospective EMA / lag-tau detection in production WLS, controller
dynamics, imperfect rail-fraction targeting) that obscure the pure
solver A/B question we want to answer in Session 0. The full-stack
real-weather validation happens in Session 5 of #40 with the production
Tobit.

Per ``feedback_no_design_docs_in_repo.md``: this is a diagnostic
*report*, not a regression test. No assertions. Run once during Session
0; persist findings in ``project_tobit_session0_finding.md``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import numpy as np

from tests.hvac_bench.scenarios._tobit_prototype import solve_tobit_joint


# Synthetic ground truth.
# β = [intercept, outdoor_delta_signal, solar_signal]
# Magnitudes chosen so solar_signal carries a real-world-sized effect
# (β_solar = −2.0 matches `_true_ff_coef` in living_room scenarios) and
# outdoor_delta has a stronger but qualitatively similar role.
_BETA_TRUE = np.array([0.5, -0.4, -2.0])
_SIGMA_TRUE = 0.3
_FEATURE_NAMES = ["intercept", "outdoor_delta", "solar"]
_BETA_SOLAR_IDX = 2

_N_BATCHES_PRE_FLUSH = 14   # 14 batches of size BATCH_SIZE before flush
_N_BATCHES_POST_FLUSH = 14  # 14 batches after to measure recovery
_BATCH_SIZE = 60            # observations per batch
_BUFFER_CAP = 1500          # max buffer length (mimics production buffer)
_FLUSH_KEEP_FRAC = 0.05     # at flush, keep 5% of pre-flush buffer
_N_SEEDS = 5
_CONVERGENCE_THRESHOLD = 0.10  # fraction of |β_truth|


# ── Synthetic data generator ────────────────────────────────────────────


def _generate_batch(
    rng: np.random.Generator,
    n: int,
    beta_true: np.ndarray,
    sigma_true: float,
    rail_fraction_target: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one batch of synthetic regression data with right-censoring.

    Features:
    - intercept = 1
    - outdoor_delta ~ Normal(0, 1) — symmetric scale, like a normalized cold-tail
    - solar ~ Beta(2, 5) (peaked low, rare-strong-solar tail) — matches the
      production solar feature distribution shape

    Latent y* = X β + N(0, σ²). The rail is set at a quantile of y* so that
    approximately ``rail_fraction_target`` of observations are right-censored.
    Returns (X, y_obs, censor_mask) where ``censor_mask`` is +1 for censored,
    0 for uncensored.
    """
    x_outdoor = rng.normal(0.0, 1.0, size=n)
    x_solar = rng.beta(2.0, 5.0, size=n)
    X = np.column_stack([np.ones(n), x_outdoor, x_solar])
    y_latent = X @ beta_true + sigma_true * rng.normal(size=n)
    if rail_fraction_target > 0:
        # Rail at the (1 − fraction) quantile of latent y*
        rail = float(np.quantile(y_latent, 1.0 - rail_fraction_target))
        censor = (y_latent > rail).astype(int)
        y_obs = np.where(censor == 1, rail, y_latent)
    else:
        censor = np.zeros(n, dtype=int)
        y_obs = y_latent
    return X, y_obs, censor


# ── A/B solvers ─────────────────────────────────────────────────────────


def _solve_wls(
    X: np.ndarray, y: np.ndarray, w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Plain weighted least squares — matches the rails-dropped path.

    Returns (beta, std_err, sigma).
    """
    p = X.shape[1]
    if X.shape[0] < p + 1:
        return np.full(p, np.nan), np.full(p, np.nan), float("nan")
    W = np.diag(w)
    XtWX = X.T @ W @ X + 1e-6 * np.eye(p)
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    resid = y - X @ beta
    sigma_sq = float(np.average(resid * resid, weights=w))
    sigma = math.sqrt(max(sigma_sq, 1e-12))
    XtWX_inv = np.linalg.inv(XtWX)
    std_err = sigma * np.sqrt(np.maximum(np.diag(XtWX_inv), 0.0))
    return beta, std_err, sigma


@dataclass
class BatchSnapshot:
    batch_idx: int
    n_obs_total: int
    n_uncensored: int
    n_censored: int
    rail_fraction: float
    beta_wls: list[float]
    std_err_wls: list[float]
    sigma_wls: float
    beta_tobit: list[float]
    std_err_tobit: list[float]
    sigma_tobit: float


@dataclass
class ScenarioRun:
    rail_label: str
    rail_target: float
    seed: int
    snapshots: list[BatchSnapshot] = field(default_factory=list)


def _solve_pair(
    X_buf: np.ndarray, y_buf: np.ndarray, censor_buf: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, float,  # WLS
    np.ndarray, np.ndarray, float,  # Tobit
]:
    """Solve both WLS-on-uncensored and Tobit-on-all from the buffer."""
    w_buf = np.ones(X_buf.shape[0])
    u = censor_buf == 0
    if int(u.sum()) >= X_buf.shape[1] + 1:
        beta_wls, se_wls, sigma_wls = _solve_wls(X_buf[u], y_buf[u], w_buf[u])
    else:
        beta_wls = np.full(X_buf.shape[1], np.nan)
        se_wls = np.full(X_buf.shape[1], np.nan)
        sigma_wls = float("nan")
    if X_buf.shape[0] >= X_buf.shape[1] + 1:
        try:
            beta_tobit, se_tobit, sigma_tobit = solve_tobit_joint(
                X_buf, y_buf, w_buf, censor_buf,
                beta_init=(beta_wls if not np.any(np.isnan(beta_wls))
                           else None),
            )
        except Exception:
            beta_tobit = np.full(X_buf.shape[1], np.nan)
            se_tobit = np.full(X_buf.shape[1], np.nan)
            sigma_tobit = float("nan")
    else:
        beta_tobit = np.full(X_buf.shape[1], np.nan)
        se_tobit = np.full(X_buf.shape[1], np.nan)
        sigma_tobit = float("nan")
    return (
        beta_wls, se_wls, sigma_wls,
        beta_tobit, se_tobit, sigma_tobit,
    )


def _trim_buffer(
    X: np.ndarray, y: np.ndarray, censor: np.ndarray, cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] <= cap:
        return X, y, censor
    return X[-cap:], y[-cap:], censor[-cap:]


def _run_scenario(rail_label: str, rail_target: float, seed: int) -> ScenarioRun:
    """Single scenario × seed: pre-flush trajectory + flush + post-flush.

    Pre-flush: 14 batches × 60 obs = 840 obs accumulated.
    Flush at batch 14: keep last 5% (42 obs).
    Post-flush: 14 batches × 60 obs added on top of the kept 42.
    """
    rng = np.random.default_rng(seed)
    run = ScenarioRun(rail_label=rail_label, rail_target=rail_target, seed=seed)

    X_buf = np.zeros((0, 3))
    y_buf = np.zeros(0)
    cens_buf = np.zeros(0, dtype=int)

    # Pre-flush
    for batch_idx in range(_N_BATCHES_PRE_FLUSH):
        X_b, y_b, cens_b = _generate_batch(
            rng, _BATCH_SIZE, _BETA_TRUE, _SIGMA_TRUE, rail_target,
        )
        X_buf = np.vstack([X_buf, X_b])
        y_buf = np.concatenate([y_buf, y_b])
        cens_buf = np.concatenate([cens_buf, cens_b])
        X_buf, y_buf, cens_buf = _trim_buffer(X_buf, y_buf, cens_buf, _BUFFER_CAP)
        (beta_w, se_w, sig_w, beta_t, se_t, sig_t) = _solve_pair(
            X_buf, y_buf, cens_buf,
        )
        rail_frac = float(np.mean(cens_buf != 0))
        run.snapshots.append(BatchSnapshot(
            batch_idx=batch_idx + 1,
            n_obs_total=X_buf.shape[0],
            n_uncensored=int(np.sum(cens_buf == 0)),
            n_censored=int(np.sum(cens_buf != 0)),
            rail_fraction=rail_frac,
            beta_wls=beta_w.tolist(),
            std_err_wls=se_w.tolist(),
            sigma_wls=sig_w,
            beta_tobit=beta_t.tolist(),
            std_err_tobit=se_t.tolist(),
            sigma_tobit=sig_t,
        ))

    # Flush — keep only last 5% of buffer
    keep = max(int(round(_FLUSH_KEEP_FRAC * X_buf.shape[0])), 1)
    X_buf = X_buf[-keep:]
    y_buf = y_buf[-keep:]
    cens_buf = cens_buf[-keep:]

    # Post-flush
    for batch_idx in range(_N_BATCHES_POST_FLUSH):
        X_b, y_b, cens_b = _generate_batch(
            rng, _BATCH_SIZE, _BETA_TRUE, _SIGMA_TRUE, rail_target,
        )
        X_buf = np.vstack([X_buf, X_b])
        y_buf = np.concatenate([y_buf, y_b])
        cens_buf = np.concatenate([cens_buf, cens_b])
        X_buf, y_buf, cens_buf = _trim_buffer(X_buf, y_buf, cens_buf, _BUFFER_CAP)
        (beta_w, se_w, sig_w, beta_t, se_t, sig_t) = _solve_pair(
            X_buf, y_buf, cens_buf,
        )
        rail_frac = float(np.mean(cens_buf != 0))
        run.snapshots.append(BatchSnapshot(
            batch_idx=_N_BATCHES_PRE_FLUSH + batch_idx + 1,
            n_obs_total=X_buf.shape[0],
            n_uncensored=int(np.sum(cens_buf == 0)),
            n_censored=int(np.sum(cens_buf != 0)),
            rail_fraction=rail_frac,
            beta_wls=beta_w.tolist(),
            std_err_wls=se_w.tolist(),
            sigma_wls=sig_w,
            beta_tobit=beta_t.tolist(),
            std_err_tobit=se_t.tolist(),
            sigma_tobit=sig_t,
        ))

    return run


# ── Metric extraction ───────────────────────────────────────────────────


def _batches_to_converge(
    snapshots: list[BatchSnapshot],
    extract: "callable",  # type: ignore[name-defined]
    truth: float,
    upper_batch_idx: int = _N_BATCHES_PRE_FLUSH,
) -> int | None:
    """First batch (1-indexed) where |extract − truth| / |truth| ≤ threshold,
    counting only batches in [1, upper_batch_idx]. None = never converged.
    """
    if abs(truth) < 1e-9:
        return None
    for s in snapshots:
        if s.batch_idx > upper_batch_idx:
            return None
        v = extract(s)
        if math.isnan(v):
            continue
        if abs(v - truth) / abs(truth) <= _CONVERGENCE_THRESHOLD:
            return s.batch_idx
    return None


def _snapshot_at(
    snapshots: list[BatchSnapshot], batch_idx: int,
) -> BatchSnapshot | None:
    for s in snapshots:
        if s.batch_idx == batch_idx:
            return s
    return None


def _recovery_batches(
    snapshots: list[BatchSnapshot],
    extract: "callable",  # type: ignore[name-defined]
    pre_flush_batch_idx: int = _N_BATCHES_PRE_FLUSH,
    target_threshold: float = _CONVERGENCE_THRESHOLD,
) -> int | None:
    """Batches after the flush boundary until extract returns within threshold
    of the pre-flush value. None = never recovered.
    """
    pre = _snapshot_at(snapshots, pre_flush_batch_idx)
    if pre is None:
        return None
    target = extract(pre)
    if math.isnan(target) or abs(target) < 1e-9:
        return None
    post = [s for s in snapshots if s.batch_idx > pre_flush_batch_idx]
    for i, s in enumerate(post):
        v = extract(s)
        if math.isnan(v):
            continue
        if abs(v - target) / abs(target) <= target_threshold:
            return i + 1
    return None


@dataclass
class CellMetrics:
    rail_label: str
    rail_target: float
    btc_wls: list[int | None]  # batches-to-converge to truth
    btc_tobit: list[int | None]
    asym_wls: list[float]  # |β − truth| at last pre-flush batch
    asym_tobit: list[float]
    se_wls: list[float]
    se_tobit: list[float]
    rec_wls: list[int | None]
    rec_tobit: list[int | None]
    rail_fraction_mean: float
    n_seeds: int


def _compute_cell_metrics(runs: list[ScenarioRun]) -> CellMetrics:
    def get_beta(s: BatchSnapshot, src: str) -> float:
        return (s.beta_wls if src == "wls" else s.beta_tobit)[_BETA_SOLAR_IDX]

    def get_se(s: BatchSnapshot, src: str) -> float:
        return (s.std_err_wls if src == "wls" else s.std_err_tobit)[_BETA_SOLAR_IDX]

    cm = CellMetrics(
        rail_label=runs[0].rail_label, rail_target=runs[0].rail_target,
        btc_wls=[], btc_tobit=[], asym_wls=[], asym_tobit=[],
        se_wls=[], se_tobit=[], rec_wls=[], rec_tobit=[],
        rail_fraction_mean=0.0, n_seeds=len(runs),
    )
    truth_solar = float(_BETA_TRUE[_BETA_SOLAR_IDX])

    rail_fracs = []
    for r in runs:
        snaps = r.snapshots
        cm.btc_wls.append(_batches_to_converge(
            snaps, lambda s: get_beta(s, "wls"), truth_solar,
        ))
        cm.btc_tobit.append(_batches_to_converge(
            snaps, lambda s: get_beta(s, "tobit"), truth_solar,
        ))
        last_pre = _snapshot_at(snaps, _N_BATCHES_PRE_FLUSH)
        if last_pre is not None:
            cm.asym_wls.append(abs(get_beta(last_pre, "wls") - truth_solar))
            cm.asym_tobit.append(abs(get_beta(last_pre, "tobit") - truth_solar))
            cm.se_wls.append(get_se(last_pre, "wls"))
            cm.se_tobit.append(get_se(last_pre, "tobit"))
            rail_fracs.append(last_pre.rail_fraction)
        cm.rec_wls.append(_recovery_batches(
            snaps, lambda s: get_beta(s, "wls"),
        ))
        cm.rec_tobit.append(_recovery_batches(
            snaps, lambda s: get_beta(s, "tobit"),
        ))
    cm.rail_fraction_mean = (
        sum(rail_fracs) / len(rail_fracs) if rail_fracs else 0.0
    )
    return cm


# ── Reporting ───────────────────────────────────────────────────────────


def _fmt_opt(x: int | None) -> str:
    return f"{x}" if x is not None else "n/c"


def _median_or_nan(xs: list) -> float:
    vals = [float(x) for x in xs if x is not None and not (
        isinstance(x, float) and math.isnan(x))]
    return float(statistics.median(vals)) if vals else float("nan")


def _print_cell(cm: CellMetrics) -> None:
    print(f"\n--- {cm.rail_label} (rail≈{cm.rail_fraction_mean*100:.1f}%, "
          f"target {cm.rail_target*100:.0f}%, N={cm.n_seeds} seeds) ---")
    print(f"  Per-seed batches-to-converge (β_solar within "
          f"{_CONVERGENCE_THRESHOLD*100:.0f}% of {_BETA_TRUE[_BETA_SOLAR_IDX]}):")
    print(f"     WLS   = {[_fmt_opt(v) for v in cm.btc_wls]}")
    print(f"     Tobit = {[_fmt_opt(v) for v in cm.btc_tobit]}")
    print(f"  Per-seed |β_solar − truth| at batch 14 (asymptotic):")
    print(f"     WLS   = {[f'{v:.4f}' for v in cm.asym_wls]}")
    print(f"     Tobit = {[f'{v:.4f}' for v in cm.asym_tobit]}")
    print(f"  Per-seed std_err(β_solar) at batch 14:")
    print(f"     WLS   = {[f'{v:.4f}' for v in cm.se_wls]}")
    print(f"     Tobit = {[f'{v:.4f}' for v in cm.se_tobit]}")
    print(f"  Per-seed recovery batches after flush (back to pre-flush β):")
    print(f"     WLS   = {[_fmt_opt(v) for v in cm.rec_wls]}")
    print(f"     Tobit = {[_fmt_opt(v) for v in cm.rec_tobit]}")


def _print_summary_table(cells: list[CellMetrics]) -> None:
    print(f"\n{'=' * 92}")
    print("  Tobit convergence-quality A/B — Session 0 metrics (medians, "
          f"{_N_SEEDS} seeds, synthetic data)")
    print(f"{'=' * 92}")
    print()
    print(f"  β_truth = {_BETA_TRUE.tolist()}, σ_truth = {_SIGMA_TRUE}, "
          f"buffer cap = {_BUFFER_CAP}")
    print(f"  Pre-flush: {_N_BATCHES_PRE_FLUSH} batches × {_BATCH_SIZE} obs.  "
          f"Flush keeps {_FLUSH_KEEP_FRAC*100:.0f}%.  "
          f"Post-flush: {_N_BATCHES_POST_FLUSH} batches × {_BATCH_SIZE} obs.")
    print()

    print(f"{'cell':<14}{'rail%':>7}"
          f"{'btc_wls':>10}{'btc_tob':>10}{'speedup':>10}"
          f"{'asym_wls':>11}{'asym_tob':>11}{'imp%':>9}")
    print("-" * 92)
    for cm in cells:
        btc_w = _median_or_nan(cm.btc_wls)
        btc_t = _median_or_nan(cm.btc_tobit)
        speedup = (btc_w / btc_t) if (btc_t and not math.isnan(btc_t)
                                       and btc_t > 0) else float("nan")
        asym_w = _median_or_nan(cm.asym_wls)
        asym_t = _median_or_nan(cm.asym_tobit)
        imp = ((asym_w - asym_t) / asym_w * 100.0
               if asym_w and not math.isnan(asym_w) and asym_w > 0
               else float("nan"))
        print(f"{cm.rail_label:<14}{cm.rail_fraction_mean*100:>6.1f}%"
              f"{btc_w:>10.0f}{btc_t:>10.0f}{speedup:>10.2f}"
              f"{asym_w:>11.4f}{asym_t:>11.4f}{imp:>8.1f}%")

    print()
    print(f"{'cell':<14}{'se_wls':>10}{'se_tob':>10}{'se_ratio':>10}"
          f"{'rec_wls':>10}{'rec_tob':>10}{'rec_speedup':>13}")
    print("-" * 92)
    for cm in cells:
        se_w = _median_or_nan(cm.se_wls)
        se_t = _median_or_nan(cm.se_tobit)
        se_ratio = (se_t / se_w) if (se_w and not math.isnan(se_w)
                                      and se_w > 0) else float("nan")
        rec_w = _median_or_nan(cm.rec_wls)
        rec_t = _median_or_nan(cm.rec_tobit)
        rec_speedup = (rec_w / rec_t) if (rec_t and not math.isnan(rec_t)
                                           and rec_t > 0) else float("nan")
        print(f"{cm.rail_label:<14}"
              f"{se_w:>10.4f}{se_t:>10.4f}{se_ratio:>10.2f}"
              f"{rec_w:>10.0f}{rec_t:>10.0f}{rec_speedup:>13.2f}")

    print()
    print("Kill-criterion thresholds (per plan, applied to high_rails cell):")
    print("  (A) speedup        ≥ 1.43  (= 1/0.7, batches-to-converge)")
    print("  (B) asym imp       ≥ 20%  (asymptotic accuracy)")
    print("  (C) se_ratio       ≤ 0.7   (std_err reduction)")
    print("  (D) rec_speedup    ≥ 1.43  (recovery)")
    print("  ≥3 of 4 in high_rails AND ≥1 in another cell → proceed.")


# ── Scenario runner ─────────────────────────────────────────────────────


_SCENARIOS = [
    ("low_rails", 0.05),
    ("mid_rails", 0.30),
    ("high_rails", 0.60),
]


def run_tobit_convergence_quality_summary() -> None:
    """CLI entry: run all scenarios × seeds, print A/B summary."""
    cells: list[CellMetrics] = []
    for label, target in _SCENARIOS:
        print(f"\nRunning {label} (target rail {target*100:.0f}%, "
              f"{_N_SEEDS} seeds)…")
        runs = [_run_scenario(label, target, seed) for seed in range(_N_SEEDS)]
        cm = _compute_cell_metrics(runs)
        cells.append(cm)
        _print_cell(cm)
    _print_summary_table(cells)
