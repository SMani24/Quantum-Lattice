"""Numerical calibration using independently sampled Pauli Z error patterns."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantum_lattice.lattice.builders import square_lattice
from quantum_lattice.lattice.square import validate_side_length
from quantum_lattice.lattice.topology import Lattice
from quantum_lattice.metrics.lattice_calibration import (
    analytic_lattice_pattern_variance,
    exact_lattice_vertex_rate,
)


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _error_probability(error_rate: float) -> float:
    if not np.isscalar(error_rate) or not np.isfinite(error_rate) or not 0 <= error_rate <= 1:
        raise ValueError("error_rate must be a finite scalar probability")
    return float(error_rate)


def sample_lattice_errors(
    rng: np.random.Generator, lattice: Lattice, error_rate: float, patterns: int
) -> NDArray[np.bool_]:
    """Draw independent edge errors as (patterns, edge_id) Boolean arrays.

    Topology is constructed once and never stores mutable error flags. The
    edge-ID order determines the random-draw order and is part of provenance.
    """
    if not isinstance(lattice, Lattice):
        raise ValueError("lattice must be a Lattice")
    patterns = _positive_integer(patterns, "patterns")
    rate = _error_probability(error_rate)
    return rng.random((patterns, lattice.n_edges)) < rate


def sample_phase_errors(
    rng: np.random.Generator, side_length: int, error_rate: float, patterns: int
) -> NDArray[np.bool_]:
    """Compatibility interface returning square-grid flags (patterns,L,L,2).

    This retains the original draw order for audits and square reference tests.
    New topology-based calculations use sample_lattice_errors().
    """
    side_length = validate_side_length(side_length)
    patterns = _positive_integer(patterns, "patterns")
    rate = _error_probability(error_rate)
    return rng.random((patterns, side_length, side_length, 2)) < rate


def run_lattice_calibration_point(
    lattice: Lattice,
    error_rate: float,
    patterns: int,
    batch_size: int,
    seed_sequence: np.random.SeedSequence,
) -> dict:
    """Sample a point using vertex incidence, without square-lattice assumptions.

    Each observation is the fraction of negative star outcomes across one
    complete independently drawn lattice. Using these observations for the
    standard error retains within-lattice covariance of neighboring stars.
    Raw patterns are discarded after each batch; retained memory is bounded
    by batch_size. Both empirical and topology-aware analytic SE are retained.
    """
    if not isinstance(lattice, Lattice):
        raise ValueError("lattice must be a Lattice")
    patterns = _positive_integer(patterns, "patterns", minimum=2)
    batch_size = _positive_integer(batch_size, "batch_size")
    if not isinstance(seed_sequence, np.random.SeedSequence):
        raise ValueError("seed_sequence must be a NumPy SeedSequence")
    # Validate the rate before allocating or sampling any error patterns.
    error_rate = _error_probability(error_rate)
    rng = np.random.default_rng(seed_sequence)
    count = 0
    total = 0.0
    total_squared = 0.0
    while count < patterns:
        batch_count = min(batch_size, patterns - count)
        errors = sample_lattice_errors(rng, lattice, error_rate, batch_count)
        values = lattice.defect_fractions(errors)
        total += float(values.sum(dtype=np.float64))
        total_squared += float(np.square(values).sum(dtype=np.float64))
        count += batch_count
    mean = total / count
    sample_variance = max(0.0, (total_squared - total**2 / count) / (count - 1))
    standard_error = float(np.sqrt(sample_variance / count))
    analytical_mean = float(exact_lattice_vertex_rate(lattice, error_rate))
    difference = mean - analytical_mean
    # At deterministic endpoints (notably e_r=0), the zero difference is exact.
    z_score = difference / standard_error if standard_error else (0.0 if difference == 0 else None)
    return {
        "geometry": lattice.geometry,
        "n_vertices": lattice.n_vertices,
        "n_edges": lattice.n_edges,
        "n_faces": lattice.n_faces,
        "e_r": float(error_rate),
        "patterns": count,
        "mean_p": mean,
        "se_p": standard_error,
        "analytic_p": analytical_mean,
        "analytic_se_p": float(np.sqrt(analytic_lattice_pattern_variance(lattice, error_rate) / count)),
        "z_score": z_score,
        "mean_stabilizer": 1 - 2 * mean,
        "stream_spawn_key": list(seed_sequence.spawn_key),
    }


def run_calibration_point(
    side_length: int,
    error_rate: float,
    patterns: int,
    batch_size: int,
    seed_sequence: np.random.SeedSequence,
    *,
    lattice: Lattice | None = None,
) -> dict:
    """Square Phase 0 interface retaining the original row schema and streams.

    Supply a prebuilt lattice to reuse its topology across probability points.
    Generic geometries use run_lattice_calibration_point() instead; the paper's
    square fitting protocol and published coefficients stay square-specific.
    """
    side_length = validate_side_length(side_length)
    if lattice is None:
        lattice = square_lattice(side_length)
    if (
        not isinstance(lattice, Lattice)
        or lattice.geometry != "square_toric"
        or lattice.boundary != "periodic"
        or lattice.shape != (side_length, side_length)
        or lattice.n_vertices != side_length**2
        or lattice.n_edges != 2 * side_length**2
        or lattice.n_faces != side_length**2
        or not np.all(lattice.vertex_degrees == 4)
        or not np.all(lattice.face_degrees == 4)
    ):
        raise ValueError("The Phase 0 square interface requires the matching periodic square lattice")
    result = run_lattice_calibration_point(lattice, error_rate, patterns, batch_size, seed_sequence)
    return {
        "L": side_length,
        **{key: result[key] for key in (
            "e_r", "patterns", "mean_p", "se_p", "analytic_p", "z_score",
            "mean_stabilizer", "stream_spawn_key",
        )},
    }
