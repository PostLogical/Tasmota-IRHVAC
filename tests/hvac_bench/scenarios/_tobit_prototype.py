"""Minimal Tobit MLE prototype for #40 Session 0 convergence-quality A/B.

**Disposable.** Production version with numerical guards is Session 2.
This module exists to answer "if we had Tobit, would it materially
improve convergence?" before committing 5 sessions of work.

Approach: maximize the right/left-censored normal log-likelihood

    L(β, σ) = Σ_uncens [log φ((y − xβ)/σ) − log σ]
            + Σ_right_cens log Φ((xβ − y_obs)/σ)
            + Σ_left_cens  log Φ((y_obs − xβ)/σ)

via scipy.optimize.minimize (BFGS with analytical gradient). σ is
log-parameterized for unconstrained optimization. Warm-start β from a
WLS fit on the uncensored subset.

No numerical guards beyond a hard log-σ floor — production version owns
Mill's-ratio asymptotics, Hessian-PD checks, line-search-failure
fallback, etc.

The function ``solve_tobit_joint`` is the single public API; the helper
``__main__`` block runs a sanity self-test so you can validate the
prototype with ``python -m tests.hvac_bench.scenarios._tobit_prototype``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.stats import norm  # type: ignore[import-untyped]


def tobit_neg_loglik(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    censor_mask: np.ndarray,
) -> float:
    """Negative weighted Tobit log-likelihood.

    ``theta`` packs ``[β, log_sigma]`` for unconstrained optimization.
    ``censor_mask``: 0 = uncensored, +1 = right-censored, −1 = left-censored.
    """
    p = X.shape[1]
    beta = theta[:p]
    log_sigma = theta[p]
    sigma = float(np.exp(log_sigma))

    eta = X @ beta
    uncens = censor_mask == 0
    cens = ~uncens

    ll = 0.0
    if uncens.any():
        z_u = (y[uncens] - eta[uncens]) / sigma
        log_phi = -0.5 * np.log(2.0 * np.pi) - 0.5 * z_u * z_u - log_sigma
        ll += float(np.sum(w[uncens] * log_phi))
    if cens.any():
        # z = s * (eta - y_obs) / σ where s = ±1 = censor_mask
        z_c = censor_mask[cens] * (eta[cens] - y[cens]) / sigma
        log_Phi = norm.logcdf(z_c)
        ll += float(np.sum(w[cens] * log_Phi))
    return -ll


def tobit_neg_score(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    censor_mask: np.ndarray,
) -> np.ndarray:
    """Gradient of ``-loglik`` wrt ``[β..., log_sigma]``.

    Uses the standard Tobit score with Mill's-ratio λ(z) = φ(z)/Φ(z) for
    censored contributions. Gradients wrt ``log σ`` (chain rule: σ * ∂/∂σ).
    """
    p = X.shape[1]
    beta = theta[:p]
    log_sigma = theta[p]
    sigma = float(np.exp(log_sigma))

    eta = X @ beta
    uncens = censor_mask == 0
    cens = ~uncens

    grad_beta = np.zeros(p)
    grad_log_sigma = 0.0

    if uncens.any():
        z_u = (y[uncens] - eta[uncens]) / sigma
        x_u = X[uncens]
        w_u = w[uncens]
        # ∂/∂β_j log φ(z) = (z/σ) * x_j  →  contribution w * z/σ * x
        grad_beta += (w_u * z_u / sigma) @ x_u
        # ∂/∂log σ log φ(z) = z² − 1
        grad_log_sigma += float(np.sum(w_u * (z_u * z_u - 1.0)))

    if cens.any():
        s_c = censor_mask[cens].astype(float)
        z_c = s_c * (eta[cens] - y[cens]) / sigma
        x_c = X[cens]
        w_c = w[cens]
        # Mill's ratio λ(z) = φ(z) / Φ(z); compute log-stable.
        log_phi_c = -0.5 * np.log(2.0 * np.pi) - 0.5 * z_c * z_c
        log_Phi_c = norm.logcdf(z_c)
        lam = np.exp(log_phi_c - log_Phi_c)
        # ∂z/∂β_j = s * x_j / σ  →  contribution w * λ * s * x / σ
        grad_beta += (w_c * lam * s_c / sigma) @ x_c
        # ∂z/∂log σ = -z  →  contribution w * λ * (-z) = -w * λ * z
        grad_log_sigma += float(-np.sum(w_c * lam * z_c))

    return -np.concatenate([grad_beta, [grad_log_sigma]])


def solve_tobit_joint(
    X: np.ndarray | list[list[float]],
    y: np.ndarray | list[float],
    w: np.ndarray | list[float],
    censor_mask: np.ndarray | list[int],
    *,
    beta_init: np.ndarray | None = None,
    sigma_init: float | None = None,
    max_iter: int = 200,
    gtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Tobit β + σ via BFGS on the negative log-likelihood.

    Parameters
    ----------
    X : (m, p) design matrix
    y : (m,) targets — for censored rows, the observed rail value
    w : (m,) per-observation weights (matches the WLS weighting convention)
    censor_mask : (m,) integer array; 0 = uncensored, +1 = right, −1 = left
    beta_init : warm-start β; if None, WLS on the uncensored subset
    sigma_init : warm-start σ; if None, residual RMS from warm-start β

    Returns
    -------
    (beta, std_err, sigma)
        ``std_err`` from the BFGS-approximated inverse Hessian diagonal,
        truncated to the β block (drops the log-σ entry). NaN if BFGS
        didn't return ``hess_inv``.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    cens_arr = np.asarray(censor_mask, dtype=int)
    p = X_arr.shape[1]

    if beta_init is None:
        u = cens_arr == 0
        if int(u.sum()) < p:
            beta_init = np.zeros(p)
        else:
            W = np.diag(w_arr[u])
            XtWX = X_arr[u].T @ W @ X_arr[u] + 1e-6 * np.eye(p)
            XtWy = X_arr[u].T @ W @ y_arr[u]
            beta_init = np.linalg.solve(XtWX, XtWy)

    if sigma_init is None:
        u = cens_arr == 0
        if int(u.sum()) >= p + 1:
            resid = y_arr[u] - X_arr[u] @ beta_init
            sigma_init = max(
                float(np.sqrt(np.average(resid * resid, weights=w_arr[u]))),
                1e-3,
            )
        else:
            sigma_init = max(float(np.std(y_arr) or 1.0), 1e-3)

    theta_init = np.concatenate([beta_init, [np.log(sigma_init)]])
    result = minimize(
        tobit_neg_loglik, theta_init,
        args=(X_arr, y_arr, w_arr, cens_arr),
        jac=tobit_neg_score, method="BFGS",
        options={"maxiter": max_iter, "gtol": gtol},
    )

    beta = np.asarray(result.x[:p], dtype=float)
    sigma = float(np.exp(result.x[p]))
    if hasattr(result, "hess_inv"):
        H_inv = np.asarray(result.hess_inv, dtype=float)
        diag = np.diag(H_inv)
        # Negative diagonal can occur when BFGS hasn't converged; clip.
        diag_clipped = np.where(diag > 0, diag, np.nan)
        std_err_full = np.sqrt(diag_clipped)
        std_err = std_err_full[:p]
    else:
        std_err = np.full(p, np.nan)

    return beta, std_err, sigma


def _self_test() -> None:
    """Sanity self-test: pure-uncensored input ≈ WLS solution.

    Run via ``python -m tests.hvac_bench.scenarios._tobit_prototype``.
    """
    rng = np.random.default_rng(42)

    # Pure-uncensored case: should match WLS exactly (within MLE noise).
    n, p = 200, 3
    X = rng.normal(size=(n, p))
    X[:, 0] = 1.0  # intercept
    beta_true = np.array([0.5, -2.0, 1.0])
    sigma_true = 0.3
    y = X @ beta_true + sigma_true * rng.normal(size=n)
    w = np.ones(n)
    cens = np.zeros(n, dtype=int)

    # WLS for comparison
    XtX = X.T @ X
    Xty = X.T @ y
    beta_wls = np.linalg.solve(XtX, Xty)

    beta_tobit, se_tobit, sigma_tobit = solve_tobit_joint(X, y, w, cens)

    print("Sanity test 1: pure uncensored")
    print(f"  β_true   = {beta_true}")
    print(f"  β_wls    = {beta_wls}")
    print(f"  β_tobit  = {beta_tobit}")
    print(f"  σ_true   = {sigma_true}")
    print(f"  σ_tobit  = {sigma_tobit}")
    print(f"  std_err  = {se_tobit}")
    diff = np.max(np.abs(beta_tobit - beta_wls))
    print(f"  max |β_tobit − β_wls| = {diff:.6f}  "
          f"(should be ≪ 1)")
    assert diff < 1e-3, "Tobit should match WLS on pure uncensored data"

    # Right-censored case: censor top 30% of y.
    print("\nSanity test 2: 30% right-censored")
    threshold = np.quantile(y, 0.7)
    y_obs = np.minimum(y, threshold)
    cens_30 = np.where(y > threshold, 1, 0).astype(int)

    XtX_u = X[cens_30 == 0].T @ X[cens_30 == 0]
    Xty_u = X[cens_30 == 0].T @ y_obs[cens_30 == 0]
    beta_wls_trunc = np.linalg.solve(XtX_u, Xty_u)

    beta_tobit2, se_tobit2, sigma_tobit2 = solve_tobit_joint(
        X, y_obs, w, cens_30,
    )

    err_wls = np.linalg.norm(beta_wls_trunc - beta_true)
    err_tobit = np.linalg.norm(beta_tobit2 - beta_true)
    print(f"  β_true            = {beta_true}")
    print(f"  β_wls_truncated   = {beta_wls_trunc}")
    print(f"  β_tobit           = {beta_tobit2}")
    print(f"  σ_tobit           = {sigma_tobit2}")
    print(f"  ||β_wls − truth|| = {err_wls:.4f}")
    print(f"  ||β_tobit − truth|| = {err_tobit:.4f}")
    print(f"  Tobit improvement  = "
          f"{(err_wls - err_tobit) / err_wls * 100:.1f}%")
    assert err_tobit < err_wls, (
        "Tobit should beat WLS on truncated data"
    )

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    _self_test()
