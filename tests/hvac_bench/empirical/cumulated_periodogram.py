"""Cumulated periodogram (CP) — frequency-domain whiteness test.

Per Bacher-Madsen 2011 §1.2 and Annex 58 ST3 part 2 (Madsen-Bacher 2015):
the CP of one-step-ahead Kalman innovations must lie inside the
Kolmogorov-Smirnov band around the diagonal if the residuals are white.

Leprince et al. 2022 introduced **nCPBES** (normalized cumulated periodogram
boundary excess sum) as a single-number summary that automates the
graphical CP check.

This module operates on standardized one-step Kalman innovations
(e_k / sqrt(S_k)), which under correct model specification are
N(0, 1) i.i.d. — i.e. white noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Kolmogorov-Smirnov 1-sample critical values (two-sided)
KS_CRITICAL_ALPHA: dict[float, float] = {
    0.20: 1.07,
    0.10: 1.22,
    0.05: 1.36,
    0.025: 1.48,
    0.01: 1.63,
    0.005: 1.73,
    0.001: 1.95,
}


@dataclass
class CumulatedPeriodogramResult:
    """One-shot CP diagnostic result.

    Attributes:
        frequencies: positive-frequency bin centers (Hz at sample rate=1).
        cumulated_power: empirical CDF of normalized periodogram (∈ [0, 1]).
        diagonal: theoretical diagonal under whiteness (i/n_freq).
        upper_band: diagonal + KS critical / sqrt(n_obs).
        lower_band: diagonal - KS critical / sqrt(n_obs).
        n_obs: number of residuals used.
        ks_alpha: KS confidence level (α).
        n_outside: count of frequencies where CP is outside the band.
        max_excess: largest single-point excess outside the band (0 if all inside).
        nCPBES: normalized cumulated boundary-excess sum per Leprince 2022.
                Sum of |excess| at outside-band points, normalized by n_freq.
        inside_band: True iff every CP sample is inside the band.
    """

    frequencies: np.ndarray
    cumulated_power: np.ndarray
    diagonal: np.ndarray
    upper_band: np.ndarray
    lower_band: np.ndarray
    n_obs: int
    ks_alpha: float
    n_outside: int
    max_excess: float
    nCPBES: float
    inside_band: bool


def cumulated_periodogram(
    residuals: np.ndarray,
    *,
    ks_alpha: float = 0.05,
) -> CumulatedPeriodogramResult:
    """Compute the cumulated periodogram + KS band for whiteness testing.

    Args:
        residuals: 1-D array of residuals (NaN-free; caller should mask
            invalid entries before calling).
        ks_alpha: KS test confidence level. Defaults to 0.05.

    Returns:
        CumulatedPeriodogramResult.

    Raises:
        ValueError: if residuals are too short or contain NaNs.
        KeyError: if ks_alpha is not in KS_CRITICAL_ALPHA.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    if r.size < 4:
        raise ValueError(f"Need ≥4 residuals; got {r.size}")
    if not np.isfinite(r).all():
        raise ValueError("residuals contain NaN or Inf")
    if ks_alpha not in KS_CRITICAL_ALPHA:
        raise KeyError(
            f"ks_alpha={ks_alpha!r} not in {sorted(KS_CRITICAL_ALPHA)}"
        )

    n = r.size
    # Mean-center then take FFT; periodogram = |F(r)|² / n.
    r_centered = r - r.mean()
    spectrum = np.fft.rfft(r_centered)
    # Drop DC (k=0) and Nyquist (last bin if n is even). Per Brockwell-Davis,
    # CP uses positive frequencies in (0, π) excluding endpoints.
    if n % 2 == 0:
        power = np.abs(spectrum[1:-1]) ** 2
    else:
        power = np.abs(spectrum[1:]) ** 2

    n_freq = power.size
    if n_freq < 2:
        raise ValueError(f"Too few periodogram bins: {n_freq}")

    total_power = float(power.sum())
    if total_power <= 0:
        raise ValueError("residuals have zero variance — degenerate input")

    cum = np.cumsum(power) / total_power
    diag = np.arange(1, n_freq + 1) / n_freq

    ks_c = KS_CRITICAL_ALPHA[ks_alpha]
    # Per Brockwell-Davis Time Series & Bartlett: sup |CP(k)/CP(M) - k/M|
    # converges to a Kolmogorov distribution with denominator √M (n_freq), not √n.
    band_half_width = ks_c / np.sqrt(n_freq)
    raw_upper = diag + band_half_width
    raw_lower = diag - band_half_width
    # Clipped versions for visualization / reporting; raw bands used for the test
    upper = np.clip(raw_upper, 0.0, 1.0)
    lower = np.clip(raw_lower, 0.0, 1.0)

    # Use raw bands in the inequality so floating-point at boundaries doesn't
    # spuriously trip the test (cum[-1]=1+ε vs clipped upper[-1]=1.0).
    excess_above = np.maximum(0.0, cum - raw_upper)
    excess_below = np.maximum(0.0, raw_lower - cum)
    excess = excess_above + excess_below

    n_outside = int(np.count_nonzero(excess > 0))
    max_excess = float(excess.max())
    n_cpbes = float(excess.sum() / n_freq)
    inside = n_outside == 0

    freqs = np.fft.rfftfreq(n)[1 : 1 + n_freq]

    return CumulatedPeriodogramResult(
        frequencies=freqs,
        cumulated_power=cum,
        diagonal=diag,
        upper_band=upper,
        lower_band=lower,
        n_obs=n,
        ks_alpha=ks_alpha,
        n_outside=n_outside,
        max_excess=max_excess,
        nCPBES=n_cpbes,
        inside_band=inside,
    )
