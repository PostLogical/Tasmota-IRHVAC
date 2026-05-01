"""Identifiability diagnostics for bench WLS experiments (Phase 1b).

Reports what an experiment design *can* support in terms of parameter
recovery, independent of which estimator is actually used. The Cramér-Rao
Lower Bound (CRLB) gives the minimum variance achievable by any unbiased
estimator on the design's data; the realised WLS estimator may or may
not achieve it. Persistence-of-excitation (PE) diagnostics — rank,
condition number, smallest singular value — describe whether the
regressor matrix is informative enough that the parameters are even
identifiable in the first place.

Together these answer "this scenario CAN identify β_solar at variance ≤
X" — the missing pre-flight check from
``project_bench_solar_fidelity.md``.

References:
- Ljung, *System Identification: Theory for the User* (1999), Ch. 13.4
- Söderström & Stoica, *System Identification* (1989), §11.5
- Walter & Pronzato, *Identification of Parametric Models from
  Experimental Data* (1997), Ch. 5-6
- Cramér 1946 / Rao 1945 for the original CRLB
- Bellman & Åström 1970 for structural identifiability

This module mirrors the preprocessing inside
``batch_learning.weighted_least_squares`` (filter on
clamped/room_rate/hp_contribution_uncertain, build feature vectors via
``build_feature_vector_from_raw``, robustness weights
``1/(1 + (rate/threshold)²)``) so the FIM reflects what the WLS path
actually sees rather than an idealised X.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from custom_components.tasmota_irhvac.pi.batch_learning import (
    Observation,
    build_feature_vector_from_raw,
)


# ── Result type ────────────────────────────────────────────────────────


@dataclass
class IdentifiabilityReport:
    """Per-experiment identifiability metrics (Phase 1b deliverable).

    Attributes
    ----------
    feature_names : list[str]
        Aligned with the regressor columns; the parameter at index ``i``
        corresponds to ``feature_names[i]``.
    n_observations, n_features
        Shape of the regressor matrix actually used (after the WLS
        eligibility filter).
    crlb
        ``Var(β̂_i)`` lower bound for any unbiased estimator. Inf for
        unidentifiable parameters.
    std_err_lower_bound
        ``sqrt(crlb[i])`` — the minimum standard error.
    rank
        Numerical rank of the (weighted) regressor matrix. Equal to
        ``n_features`` for a fully identifiable design; less for
        rank-deficient designs.
    condition_number
        Belsley-standardized κ: ``σ_max / σ_min`` after each non-constant
        column is centered and scaled to unit variance. Belsley (1980)
        thresholds: ``κ > 30`` is moderately collinear, ``κ > 100`` is
        strongly collinear. Matches the production WLS κ-gate.
    smallest_singular_value, largest_singular_value
        Singular values of the standardized weighted regressor (so
        ``σ_max / σ_min == condition_number``).
    feature_variance
        Weighted variance of each regressor column. Below
        ``MIN_FEATURE_VARIANCE`` indicates a constant column.
    pe_order
        Persistence-of-excitation order. Equal to ``rank`` for the
        regressor matrix. For an n-parameter linear model, the literature
        requires PE order ≥ n (Ljung §13.4).
    sigma2
        Noise variance used to compute FIM. If ``None`` was provided
        and ``beta`` was provided, estimated from regression residuals;
        otherwise defaults to 1.0 and CRLB is in σ²-relative units.
    sigma2_was_estimated
        ``True`` iff ``sigma2`` came from residual-based estimation
        rather than caller-supplied or default.
    """

    feature_names: list[str]
    n_observations: int
    n_features: int

    crlb: list[float]
    std_err_lower_bound: list[float]

    rank: int
    condition_number: float
    smallest_singular_value: float
    largest_singular_value: float
    feature_variance: list[float]

    pe_order: int

    sigma2: float
    sigma2_was_estimated: bool


# ── Core math ──────────────────────────────────────────────────────────


def fisher_information(
    X: np.ndarray,
    weights: np.ndarray | None = None,
    sigma2: float = 1.0,
) -> np.ndarray:
    """Compute the Fisher Information Matrix for a weighted linear model.

    Assumes ``y_i = x_i^T β + ε_i`` with ``ε_i ~ N(0, σ² / w_i)``
    independent. Then ``FIM(β) = X^T W X / σ²`` with
    ``W = diag(w_i)``. With unit weights this reduces to OLS.

    Parameters
    ----------
    X : (n, p) array
        Regressor matrix.
    weights : (n,) array or None
        Per-observation weights. ``None`` means unit weights.
    sigma2 : float
        Noise variance.

    Returns
    -------
    (p, p) array
        The Fisher Information Matrix.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be > 0")
    if weights is None:
        return X.T @ X / sigma2
    if weights.shape != (X.shape[0],):
        raise ValueError(
            f"weights shape {weights.shape} != ({X.shape[0]},)"
        )
    return (X.T * weights) @ X / sigma2


def crlb_diagonal(fim: np.ndarray) -> np.ndarray:
    """Diagonal of the inverse FIM — lower bound on Var(β̂_i).

    For a non-singular FIM this is ``np.diag(np.linalg.inv(fim))``.
    For a singular FIM, the unidentifiable directions are detected via
    SVD; their corresponding diagonal entries are reported as ``+inf``.
    """
    if fim.shape[0] != fim.shape[1]:
        raise ValueError(f"FIM must be square; got {fim.shape}")
    n = fim.shape[0]
    s = np.linalg.svd(fim, compute_uv=False)
    largest = float(s.max()) if n > 0 else 0.0
    tol = max(largest * 1e-10, 1e-12)
    if s.min() > tol:
        return np.diag(np.linalg.inv(fim)).astype(float)

    # Singular: parameters along the null space are unidentifiable.
    inv = np.linalg.pinv(fim)
    out = np.diag(inv).astype(float)
    # Mark parameters with non-trivial null-space loading as +inf.
    _, sv, vt = np.linalg.svd(fim)
    null_dirs = vt[sv <= tol]
    for direction in null_dirs:
        for i in range(n):
            if abs(direction[i]) > 1e-6:
                out[i] = float("inf")
    return out


def pe_diagnostics(
    X: np.ndarray,
    weights: np.ndarray | None = None,
    standardize: bool = True,
) -> dict:
    """Persistence-of-excitation diagnostics on the regressor matrix.

    Computes rank, condition number, and singular-value extremes of the
    (weighted) regressor matrix ``W^{1/2} X``. A regressor matrix
    ``X ∈ R^{n×p}`` is *persistently exciting of order p* iff its rank
    is p; weak excitation manifests as small ``σ_min`` even when the
    rank is full.

    Parameters
    ----------
    standardize : bool
        When True (default), each non-constant column is mean-centered
        and rescaled to unit variance before computing the condition
        number. This matches the Belsley (1980) convention under which
        thresholds like κ > 30 ("moderately collinear") and κ > 100
        ("strongly collinear") are defined; it is also what production
        WLS does for numerical conditioning before solving the normal
        equations. Constant columns (variance ≈ 0) are kept un-rescaled
        to preserve the intercept's identity. When False, returns the
        condition number of the raw (weighted) X — which is dominated
        by absolute-scale differences between columns rather than by
        collinearity.

    Returns a plain ``dict`` so it can be merged into other reports.
    """
    if weights is not None:
        if weights.shape != (X.shape[0],):
            raise ValueError(
                f"weights shape {weights.shape} != ({X.shape[0]},)"
            )
        Xw = np.sqrt(weights)[:, None] * X
    else:
        Xw = X
    if Xw.size == 0:
        return {
            "rank": 0,
            "condition_number": float("inf"),
            "smallest_singular_value": 0.0,
            "largest_singular_value": 0.0,
        }

    if standardize and Xw.shape[0] > 1:
        col_means = Xw.mean(axis=0)
        col_stds = Xw.std(axis=0, ddof=0)
        # Only standardize columns with non-trivial variance — leave
        # constant columns (intercept) untouched.
        scale = np.where(col_stds > 1e-12, col_stds, 1.0)
        offset = np.where(col_stds > 1e-12, col_means, 0.0)
        Xs = (Xw - offset) / scale
    else:
        Xs = Xw

    s = np.linalg.svd(Xs, compute_uv=False)
    smallest = float(s.min())
    largest = float(s.max())
    tol = max(largest * 1e-10, 1e-12)
    rank = int((s > tol).sum())
    cond = largest / smallest if smallest > 0 else float("inf")
    return {
        "rank": rank,
        "condition_number": cond,
        "smallest_singular_value": smallest,
        "largest_singular_value": largest,
    }


def estimate_sigma2_from_residuals(
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Estimate observation-noise variance from regression residuals.

    OLS: ``σ̂² = RSS / (n - p)``.
    WLS: ``σ̂² = Σ w_i (y_i - x_i^T β)² / (Σ w_i - p)``.

    Returns 0.0 when the effective DOF is non-positive.
    """
    residuals = y - X @ beta
    n, p = X.shape
    if weights is None:
        rss = float(residuals @ residuals)
        dof = float(n - p)
    else:
        rss = float((weights * residuals**2).sum())
        dof = float(weights.sum()) - p
    if dof <= 0:
        return 0.0
    return rss / dof


# ── Bridge: Observation list → regressor matrix ───────────────────────


def build_regressor_matrix(
    observations: list[Observation],
    feature_order: list[str],
    model_inputs: list[dict] | None = None,
    room_rate_threshold: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Observation]]:
    """Build (X, y, weights, used_obs) mirroring WLS preprocessing.

    Filters on the same eligibility criteria as
    ``batch_learning.weighted_least_squares`` (not clamped, hp_setpoint
    not None, |room_rate| < threshold, not hp_contribution_uncertain,
    has outdoor_temp_c) and constructs feature vectors via the production
    ``build_feature_vector_from_raw`` so the matrix matches exactly what
    the WLS estimator regresses against.

    Returns empty arrays if no observations pass the filter; callers
    should check ``X.shape[0] == 0`` to detect this.
    """
    eligible = [
        o for o in observations
        if not o.clamped
        and o.hp_setpoint is not None
        and abs(o.room_rate) < room_rate_threshold
        and not o.hp_contribution_uncertain
        and o.outdoor_temp_c is not None
    ]

    rows: list[list[float]] = []
    ys: list[float] = []
    ws: list[float] = []
    used: list[Observation] = []
    for obs in eligible:
        feat = build_feature_vector_from_raw(
            obs, model_inputs or [], feature_order
        )
        if feat is None:
            continue
        rows.append(feat)
        assert obs.hp_setpoint is not None  # filter guarantees this
        ys.append(obs.hp_setpoint - obs.current_c)
        ws.append(1.0 / (1.0 + (obs.room_rate / room_rate_threshold) ** 2))
        used.append(obs)

    if not rows:
        return (
            np.zeros((0, len(feature_order)), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            [],
        )
    return (
        np.array(rows, dtype=float),
        np.array(ys, dtype=float),
        np.array(ws, dtype=float),
        used,
    )


# ── Top-level entry point ─────────────────────────────────────────────


def identifiability_report(
    observations: list[Observation],
    feature_order: list[str],
    model_inputs: list[dict] | None = None,
    sigma2: float | None = None,
    beta: list[float] | None = None,
    room_rate_threshold: float = 0.02,
) -> IdentifiabilityReport:
    """Build a full identifiability report from a list of observations.

    Parameters
    ----------
    observations
        Same input the WLS path consumes. Filtered internally to
        eligible observations only.
    feature_order
        Column ordering for the regressor matrix; e.g. ``["intercept",
        "outdoor_delta", "Solar Proxy", "sin_hour", "cos_hour"]``.
    model_inputs
        Production-style model-input dicts (see ``build_feature_vector_from_raw``).
    sigma2
        Noise variance. If ``None`` and ``beta`` is supplied, estimated
        from residuals; if both are ``None``, defaults to 1.0 and the
        CRLB is reported in σ²-relative units.
    beta
        Optional WLS coefficient estimate; only used to estimate
        ``sigma2`` from residuals when ``sigma2`` is None.
    room_rate_threshold
        Maximum |room_rate| (°C/min) for an observation to be eligible.
    """
    X, y, weights, used = build_regressor_matrix(
        observations, feature_order, model_inputs, room_rate_threshold
    )
    n, p = X.shape

    sigma2_was_estimated = False
    if sigma2 is None and beta is not None and n > 0:
        sigma2_value = estimate_sigma2_from_residuals(
            y, X, np.array(beta, dtype=float), weights
        )
        sigma2_was_estimated = True
        if sigma2_value <= 0:
            sigma2_value = 1.0
            sigma2_was_estimated = False
    elif sigma2 is None:
        sigma2_value = 1.0
    else:
        if sigma2 <= 0:
            raise ValueError("sigma2 must be > 0")
        sigma2_value = float(sigma2)

    if n == 0:
        return IdentifiabilityReport(
            feature_names=list(feature_order),
            n_observations=0,
            n_features=p,
            crlb=[float("inf")] * p,
            std_err_lower_bound=[float("inf")] * p,
            rank=0,
            condition_number=float("inf"),
            smallest_singular_value=0.0,
            largest_singular_value=0.0,
            feature_variance=[0.0] * p,
            pe_order=0,
            sigma2=sigma2_value,
            sigma2_was_estimated=sigma2_was_estimated,
        )

    fim = fisher_information(X, weights, sigma2_value)
    crlb = crlb_diagonal(fim)
    pe = pe_diagnostics(X, weights)

    w_sum = float(weights.sum())
    col_means = (X * weights[:, None]).sum(axis=0) / w_sum
    col_var = (
        ((X - col_means) ** 2 * weights[:, None]).sum(axis=0) / w_sum
    ).astype(float).tolist()

    return IdentifiabilityReport(
        feature_names=list(feature_order),
        n_observations=int(n),
        n_features=int(p),
        crlb=crlb.tolist(),
        std_err_lower_bound=[
            math.sqrt(c) if math.isfinite(c) and c >= 0 else float("inf")
            for c in crlb
        ],
        rank=pe["rank"],
        condition_number=pe["condition_number"],
        smallest_singular_value=pe["smallest_singular_value"],
        largest_singular_value=pe["largest_singular_value"],
        feature_variance=col_var,
        pe_order=pe["rank"],
        sigma2=sigma2_value,
        sigma2_was_estimated=sigma2_was_estimated,
    )
