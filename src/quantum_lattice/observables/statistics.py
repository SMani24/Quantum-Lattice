"""Batch-means uncertainty and conservative, descriptive chain diagnostics."""

from __future__ import annotations

import numpy as np


def summarize(values: np.ndarray, block_size: int = 45) -> dict[str, np.ndarray]:
    """Use every record for the mean; only complete blocks for batch-means SE.

    Neither thinning nor batch means proves that a chain has mixed. NaNs denote
    an unavailable uncertainty estimate, rather than a claim of zero error.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0 or block_size < 1:
        raise ValueError("values must be a nonempty (records, observables) array")
    count = len(values)
    blocks = count // block_size
    block_means = values[:blocks * block_size].reshape(
        blocks, block_size, values.shape[1]
    ).mean(axis=1) if blocks else np.empty((0, values.shape[1]))
    missing = np.full(values.shape[1], np.nan)
    return {
        "mean": values.mean(axis=0),
        "iid_se": values.std(axis=0, ddof=1) / np.sqrt(count) if count > 1 else missing.copy(),
        "block_se": block_means.std(axis=0, ddof=1) / np.sqrt(blocks) if blocks > 1 else missing.copy(),
        "block_means": block_means,
    }


def autocorrelation_diagnostic(values: np.ndarray) -> dict:
    """FFT autocorrelation, truncated at the first nonpositive lag pair.

    Report tau in retained-record units and ESS for this scalar alone. This is
    a short-chain diagnostic, not a lower bound on effective sample size.
    """
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count < 4 or not np.isfinite(values).all():
        return {"tau_records": None, "ess": None, "reason": "insufficient records"}
    centered = values - values.mean()
    variance = np.dot(centered, centered) / count
    if variance <= np.finfo(float).eps:
        return {"tau_records": None, "ess": None, "reason": "constant trace"}
    fft = np.fft.rfft(centered, n=1 << (2 * count - 1).bit_length())
    covariance = np.fft.irfft(fft * fft.conjugate())[:count] / np.arange(count, 0, -1)
    rho = covariance / covariance[0]
    pair_sum = 0.0
    previous = float("inf")
    for lag in range(0, count - 1, 2):
        pair = float(rho[lag] + rho[lag + 1])
        if pair <= 0:
            break
        pair = min(pair, previous)  # initial monotone positive sequence
        pair_sum += pair
        previous = pair
    tau = float(min(count, max(1.0, -1.0 + 2.0 * pair_sum)))
    return {"tau_records": tau, "ess": count / tau,
            "reason": "single-chain initial monotone sequence; tau conservatively floored at one"}
