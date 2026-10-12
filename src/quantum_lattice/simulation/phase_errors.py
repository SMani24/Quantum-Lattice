"""Numerical calibration using independently sampled Pauli Z error patterns."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantum_lattice.lattice.square import defect_fractions, validate_side_length
from quantum_lattice.metrics.calibration import exact_vertex_rate


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def sample_phase_errors(
    rng: np.random.Generator, side_length: int, error_rate: float, patterns: int
) -> NDArray[np.bool_]:
    """Draw independent Bernoulli decisions for every edge in every pattern."""
    side_length = validate_side_length(side_length)
    patterns = _positive_integer(patterns, "patterns")
    if not np.isscalar(error_rate) or not np.isfinite(error_rate) or not 0 <= error_rate <= 1:
        raise ValueError("error_rate must be a finite scalar probability")
    return rng.random((patterns, side_length, side_length, 2)) < error_rate


def run_calibration_point(
    side_length: int,
    error_rate: float,
    patterns: int,
    batch_size: int,
    seed_sequence: np.random.SeedSequence,
) -> dict:
    """Sample one calibration point and return compact reproducibility data.

    Each observation is the fraction of negative star outcomes across one
    complete independently drawn lattice. Using these observations for the
    standard error retains within-lattice covariance of neighboring stars.
    Raw patterns are discarded after each batch; retained memory is bounded
    by batch_size. The seed stream is supplied by the experiment runner.
    """
    side_length = validate_side_length(side_length)
    patterns = _positive_integer(patterns, "patterns", minimum=2)
    batch_size = _positive_integer(batch_size, "batch_size")
    if not isinstance(seed_sequence, np.random.SeedSequence):
        raise ValueError("seed_sequence must be a NumPy SeedSequence")
    # Validate the rate before allocating or sampling any error patterns.
    if not np.isscalar(error_rate) or not np.isfinite(error_rate) or not 0 <= error_rate <= 1:
        raise ValueError("error_rate must be a finite scalar probability")
    rng = np.random.default_rng(seed_sequence)
    count = 0
    total = 0.0
    total_squared = 0.0
    while count < patterns:
        batch_count = min(batch_size, patterns - count)
        errors = sample_phase_errors(rng, side_length, error_rate, batch_count)
        values = defect_fractions(errors)
        total += float(values.sum(dtype=np.float64))
        total_squared += float(np.square(values).sum(dtype=np.float64))
        count += batch_count
    mean = total / count
    sample_variance = max(0.0, (total_squared - total**2 / count) / (count - 1))
    standard_error = float(np.sqrt(sample_variance / count))
    analytical_mean = float(exact_vertex_rate(error_rate))
    difference = mean - analytical_mean
    # At deterministic endpoints (notably e_r=0), the zero difference is exact.
    z_score = difference / standard_error if standard_error else (0.0 if difference == 0 else None)
    return {
        "L": side_length,
        "e_r": float(error_rate),
        "patterns": count,
        "mean_p": mean,
        "se_p": standard_error,
        "analytic_p": analytical_mean,
        "z_score": z_score,
        "mean_stabilizer": 1 - 2 * mean,
        "stream_spawn_key": list(seed_sequence.spawn_key),
    }
