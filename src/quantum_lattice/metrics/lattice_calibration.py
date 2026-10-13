"""Independent-edge error calibration derived from lattice connectivity.

These formulas concern classical flags indicating independent Z errors on
edge qubits. They describe star-syndrome probabilities, not a ground-state
sampler or a decoder. No square-lattice coordination is assumed here.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from quantum_lattice.lattice.topology import Lattice
from quantum_lattice.metrics.calibration import exact_inverse, exact_vertex_rate


def _probability_array(values: ArrayLike, maximum: float = 1.0) -> NDArray:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > maximum)):
        raise ValueError(f"probabilities must be finite and between 0 and {maximum}")
    return result


def _degree(value: int, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("degree must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"degree must be {qualifier}")
    return int(value)


def _log_abs_q(rate: NDArray) -> NDArray:
    """Stable log|1-2e|, including e=0, 1/2, and 1."""
    reduced_rate = np.minimum(rate, 1 - rate)
    with np.errstate(divide="ignore"):
        return np.log1p(-2 * reduced_rate)


def parity_probability(error_rate: ArrayLike, degree: int) -> NDArray:
    """Probability of odd parity among ``degree`` independent error bits.

    The formula is [1-(1-2e)^degree]/2 for 0 <= e <= 1. Complementing
    every error bit preserves parity for even degree and reverses it for
    odd degree. A log/expm1 form avoids cancellation near e=0 or e=1.
    Degree zero denotes an isolated vertex, which is never defective.
    """
    rate = _probability_array(error_rate)
    degree = _degree(degree)
    if degree == 0:
        return np.zeros_like(rate)
    if degree == 4:
        # Preserve the original square experiment's analytic values exactly.
        return exact_vertex_rate(rate)
    base = -0.5 * np.expm1(degree * _log_abs_q(rate))
    if degree % 2:
        return np.where(rate > 0.5, 1 - base, base)
    return base


def inverse_parity_probability(vertex_rate: ArrayLike, degree: int) -> NDArray:
    """Invert regular-degree parity on the branch 0 <= e <= 1/2.

    This branch restricts the input to 0 <= p <= 1/2, even when an odd
    degree would allow inversion beyond p=1/2. For even degree, p alone
    cannot distinguish e from 1-e. Heterogeneous-degree lattices do not
    generally have this simple single-degree inverse.
    """
    rate = _probability_array(vertex_rate, maximum=0.5)
    degree = _degree(degree, positive=True)
    if degree == 4:
        return exact_inverse(rate)
    result = np.empty_like(rate)
    interior = rate < 0.5
    result[interior] = -0.5 * np.expm1(np.log1p(-2 * rate[interior]) / degree)
    result[~interior] = 0.5
    return result


def exact_lattice_vertex_rate(lattice: Lattice, error_rate: ArrayLike) -> NDArray:
    """Mean vertex-defect probability, allowing unequal vertex degrees.

    Each edge has the same independent generating error probability.
    Vertices contribute equally to the mean; their individual odd-parity
    probabilities are determined by their respective incident-edge counts.
    """
    rate = _probability_array(error_rate)
    degrees, counts = np.unique(lattice.vertex_degrees, return_counts=True)
    if len(degrees) == 1:
        return parity_probability(rate, int(degrees[0]))
    total = np.zeros_like(rate)
    for degree, count in zip(degrees, counts):
        total += int(count) * parity_probability(rate, int(degree))
    return total / lattice.n_vertices


def analytic_lattice_pattern_variance(lattice: Lattice, error_rate: ArrayLike) -> NDArray:
    """Variance of the defect fraction in one independent whole pattern.

    Write q=1-2e, and let vertices u,v have degrees d_u,d_v and share m
    distinct edges. Their defect covariance is

        [q^(d_u+d_v-2m) - q^(d_u+d_v)] / 4.

    Nonadjacent vertices have m=0 and zero covariance. Parallel edges
    count separately, so m may exceed one. The variance of the lattice
    mean is (sum_u Var(D_u) + 2 sum_{u<v} Cov(D_u,D_v))/V^2.

    All terms share a factor (1-q^2)/4 = e(1-e). The remaining polynomial
    has integer coefficients determined only by connectivity. Factoring
    any exact q+1 roots avoids cancellation near e=1 when negative
    covariances cancel the leading variance. Negative q is retained, so
    correlations may be negative for heterogeneous vertex degrees.
    """
    rate = _probability_array(error_rate)
    # (1-q^(2d))/(1-q^2) = sum_{j=0}^{d-1} q^(2j), including q=+/-1
    # by continuous extension. Covariance terms use the same identity.
    coefficients: Counter[int] = Counter()
    degrees, counts = np.unique(lattice.vertex_degrees, return_counts=True)
    for degree, count in zip(degrees, counts):
        for power in range(0, 2 * int(degree), 2):
            coefficients[power] += int(count)

    # One unordered pair per neighboring pair, including its edge multiplicity.
    shared_edges = Counter(tuple(sorted((int(u), int(v)))) for u, v in lattice.edge_vertex_indices)
    covariance_groups: Counter[tuple[int, int]] = Counter()
    for (u, v), multiplicity in shared_edges.items():
        exponent = int(lattice.vertex_degrees[u] + lattice.vertex_degrees[v]) - 2 * multiplicity
        covariance_groups[(exponent, multiplicity)] += 1
    for (exponent, multiplicity), count in covariance_groups.items():
        for power in range(exponent, exponent + 2 * multiplicity, 2):
            coefficients[power] += 2 * count
    if not coefficients:
        return np.zeros_like(rate)

    polynomial = [coefficients[power] for power in range(max(coefficients) + 1)]
    # S(q) may vanish at q=-1 on mixed-degree graphs. Determine its root
    # multiplicity with integer arithmetic, then evaluate those factors as
    # q+1=2(1-e), avoiding subtraction of nearly equal variance terms.
    reduced_polynomial = polynomial
    root_multiplicity = 0
    while len(reduced_polynomial) > 1 and sum(
        (-1) ** power * coefficient for power, coefficient in enumerate(reduced_polynomial)
    ) == 0:
        quotient = [reduced_polynomial[0]]
        for coefficient in reduced_polynomial[1:-1]:
            quotient.append(coefficient - quotient[-1])
        reduced_polynomial = quotient
        root_multiplicity += 1
    q = 1 - 2 * rate
    remaining = np.polynomial.polynomial.polyval(q, polynomial)
    if root_multiplicity:
        stable_remaining = (2 * (1 - rate)) ** root_multiplicity * np.polynomial.polynomial.polyval(
            q, reduced_polynomial
        )
        remaining = np.where(rate > 0.5, stable_remaining, remaining)
    return rate * (1 - rate) * remaining / lattice.n_vertices**2
