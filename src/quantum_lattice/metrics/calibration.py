"""Independent phase-error calibration and inverse-polynomial diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantum_lattice.lattice.square import validate_side_length


PUBLISHED_COEFFICIENTS = (0.2187, 0.72419, -2.5398, 4.90118)


def _probability_array(values: ArrayLike, maximum: float = 1.0) -> NDArray:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > maximum)):
        raise ValueError(f"probabilities must be finite and between 0 and {maximum}")
    return result


def exact_vertex_rate(error_rate: ArrayLike) -> NDArray:
    """Expected odd-parity star fraction for independent edge errors.

    The four-edge parity gives [1 - (1 - 2 e_r)^4] / 2. An algebraically
    equivalent expression avoids cancellation at very small error rates.
    """
    rate = _probability_array(error_rate)
    return 4 * rate * (1 - rate) * ((1 - rate) ** 2 + rate**2)


def exact_inverse(vertex_rate: ArrayLike) -> NDArray:
    """Return e_r on the low-error branch, 0 <= e_r <= 1/2.

    The forward relation cannot distinguish e_r from 1-e_r, so the branch
    choice is part of the calibration assumption.
    """
    rate = _probability_array(vertex_rate, maximum=0.5)
    result = np.empty_like(rate)
    interior = rate < 0.5
    result[interior] = -0.5 * np.expm1(0.25 * np.log1p(-2 * rate[interior]))
    result[~interior] = 0.5
    return result


def analytic_pattern_variance(error_rate: ArrayLike, side_length: int) -> NDArray:
    """Variance of one full-lattice defect fraction, accounting for neighbors.

    Neighboring stars share one error bit. Their defect covariance is
    (q^6-q^8)/4, q=1-2e_r; all other distinct pairs have zero covariance.
    Consequently Var(fraction) = (1+4q^6-5q^8)/(4 L^2), for L >= 3.
    """
    side_length = validate_side_length(side_length)
    rate = _probability_array(error_rate)
    q_squared = (1 - 2 * rate) ** 2
    # Factored expression prevents cancellation when q_squared is near 1.
    numerator = (1 - q_squared) * (
        1 + q_squared + q_squared**2 + 5 * q_squared**3
    )
    return numerator / (4 * side_length**2)


def evaluate_polynomial(
    vertex_rate: ArrayLike, coefficients: ArrayLike, zero_intercept: bool = True
) -> NDArray:
    """Evaluate ascending-power coefficients (p^1 first by default)."""
    rate = np.asarray(vertex_rate, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1 or not coefficients.size:
        raise ValueError("coefficients must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(rate)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("polynomial inputs must be finite")
    if zero_intercept:
        coefficients = np.concatenate(([0.0], coefficients))
    return np.polynomial.polynomial.polyval(rate, coefficients)


def published_inverse(vertex_rate: ArrayLike) -> NDArray:
    """Evaluate Appendix D's reported fit; this is not the exact inverse."""
    return evaluate_polynomial(vertex_rate, PUBLISHED_COEFFICIENTS)


def fit_inverse(
    vertex_rate: ArrayLike,
    error_rate: ArrayLike,
    degree: int = 4,
    zero_intercept: bool = True,
) -> dict:
    """Least-squares inverse fit with JSON-serializable coefficients/metrics.

    Inputs are the numerically estimated mean vertex probabilities and the
    known generating edge probabilities. Fit residuals are in error-rate
    units. The reported MSE is an in-sample curve-fit diagnostic.
    """
    p = _probability_array(vertex_rate, maximum=0.5)
    e = _probability_array(error_rate, maximum=0.5)
    if p.ndim != 1 or e.ndim != 1 or p.shape != e.shape:
        raise ValueError("vertex_rate and error_rate must be matching 1D arrays")
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)):
        raise ValueError("degree must be a positive integer")
    if degree < 1:
        raise ValueError("degree must be a positive integer")
    first_power = 1 if zero_intercept else 0
    powers = np.arange(first_power, degree + 1)
    if len(p) < len(powers):
        raise ValueError("too few calibration points for the requested polynomial")
    design = p[:, None] ** powers
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, e, rcond=None)
    if rank < len(powers):
        raise ValueError("calibration points do not support a full-rank polynomial fit")
    residuals = design @ coefficients - e
    mse = float(np.mean(residuals**2))
    return {
        "degree": int(degree),
        "zero_intercept": bool(zero_intercept),
        "powers": powers.tolist(),
        "coefficients": coefficients.tolist(),
        "diagnostics": {
            "points": len(p),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "max_abs_error": float(np.max(np.abs(residuals))),
            "rank": int(rank),
            "condition_number": float(singular_values[0] / singular_values[-1]),
            "p_min": float(p.min()),
            "p_max": float(p.max()),
            "e_r_min": float(e.min()),
            "e_r_max": float(e.max()),
        },
    }
