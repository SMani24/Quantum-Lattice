"""Exact tiny-graph checks for topology-derived independent-error moments."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from quantum_lattice.lattice.builders import square_lattice
from quantum_lattice.lattice.topology import Lattice
from quantum_lattice.metrics.calibration import (
    analytic_pattern_variance,
    exact_inverse,
    exact_vertex_rate,
)
from quantum_lattice.metrics.lattice_calibration import (
    analytic_lattice_pattern_variance,
    exact_lattice_vertex_rate,
    inverse_parity_probability,
    parity_probability,
)


def tetrahedron() -> Lattice:
    """Four degree-three vertices, six edges, and four triangular faces."""
    return Lattice(
        4,
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        [(0, 3, 1), (0, 4, 2), (1, 5, 2), (3, 5, 4)],
        boundary="closed",
        geometry="tetrahedron",
    )


def enumerated_moments(lattice: Lattice, rate: float) -> tuple[float, float]:
    errors = np.array(list(itertools.product([0, 1], repeat=lattice.n_edges)), dtype=bool)
    defects = lattice.star_syndrome(errors).mean(axis=-1)
    counts = errors.sum(axis=-1)
    probabilities = rate**counts * (1 - rate) ** (lattice.n_edges - counts)
    mean = float(np.sum(probabilities * defects))
    # Centered calculation remains stable at deterministic endpoints.
    variance = float(np.sum(probabilities * (defects - mean) ** 2))
    return mean, variance


class LatticeCalibrationTests(unittest.TestCase):
    def test_regular_square_preserves_exact_legacy_mean(self):
        rates = np.array([0, 1e-20, 1e-12, 0.005, 0.2, 0.5, 0.7, 1])
        for side_length in [3, 4, 8]:
            lattice = square_lattice(side_length)
            np.testing.assert_array_equal(exact_lattice_vertex_rate(lattice, rates), exact_vertex_rate(rates))
            np.testing.assert_allclose(
                analytic_lattice_pattern_variance(lattice, rates),
                analytic_pattern_variance(rates, side_length),
                rtol=1e-12,
                atol=1e-16,
            )

    def test_tetrahedral_graph_uses_three_edge_parity(self):
        lattice = tetrahedron()
        np.testing.assert_array_equal(lattice.vertex_degrees, [3, 3, 3, 3])
        rates = np.array([0, 0.03, 0.2, 0.5, 0.8, 1])
        np.testing.assert_allclose(exact_lattice_vertex_rate(lattice, rates), parity_probability(rates, 3))
        self.assertEqual(float(exact_lattice_vertex_rate(lattice, 1)), 1)

    def test_heterogeneous_degrees_are_averaged_by_vertex(self):
        lattice = Lattice(5, [(0, 1), (1, 2), (1, 3)])
        np.testing.assert_array_equal(lattice.vertex_degrees, [1, 3, 1, 1, 0])
        rates = np.array([0.1, 0.3, 0.7, 1])
        expected = (3 * rates + parity_probability(rates, 3)) / 5
        np.testing.assert_allclose(exact_lattice_vertex_rate(lattice, rates), expected)

    def test_exact_weighted_enumeration_including_parallel_edges(self):
        lattices = [
            tetrahedron(),
            Lattice(5, [(0, 1), (1, 2), (1, 3)]),
            Lattice(2, [(0, 1), (0, 1)]),
            Lattice(3, [(0, 1), (0, 1), (1, 2)]),
            Lattice(3, []),
        ]
        for lattice in lattices:
            for rate in [0, 0.01, 0.2, 0.5, 0.8, 0.99, 1]:
                with self.subTest(edges=lattice.n_edges, rate=rate):
                    mean, variance = enumerated_moments(lattice, rate)
                    self.assertAlmostEqual(float(exact_lattice_vertex_rate(lattice, rate)), mean, places=14)
                    self.assertAlmostEqual(float(analytic_lattice_pattern_variance(lattice, rate)), variance, places=14)

    def test_parallel_stars_remain_identical_at_half_error_probability(self):
        lattice = Lattice(2, [(0, 1), (0, 1)])
        # The two identical stars fluctuate together: variance is 1/4, not 1/8.
        self.assertEqual(float(analytic_lattice_pattern_variance(lattice, 0.5)), 0.25)

    def test_mixed_degree_negative_covariance_above_half_probability(self):
        lattice = Lattice(3, [(0, 1), (1, 2)])
        # Degrees 1,2,1 produce negative adjacent covariance when q < 0.
        mean, variance = enumerated_moments(lattice, 0.8)
        self.assertAlmostEqual(float(exact_lattice_vertex_rate(lattice, 0.8)), mean, places=14)
        self.assertAlmostEqual(float(analytic_lattice_pattern_variance(lattice, 0.8)), variance, places=14)

    def test_tiny_error_probability_does_not_cancel(self):
        lattice = Lattice(2, [(0, 1)])
        rate = 1e-30
        self.assertAlmostEqual(float(parity_probability(rate, 3)) / rate, 3, places=14)
        self.assertAlmostEqual(float(analytic_lattice_pattern_variance(lattice, rate)) / rate, 1, places=14)

    def test_near_unit_error_rate_handles_cancelling_covariances(self):
        lattice = Lattice(3, [(0, 1), (1, 2)])
        rates = np.array([1 - 1e-8, 1 - 1e-12, np.nextafter(1.0, 0.0)])
        # On this path, the fraction is 2/3 unless both edges are error-free.
        rare_probability = (1 - rates) ** 2
        expected = (4 / 9) * rare_probability * (1 - rare_probability)
        np.testing.assert_allclose(analytic_lattice_pattern_variance(lattice, rates), expected, rtol=1e-14)

    def test_inverse_roundtrip_and_low_error_branch(self):
        rates = np.array([0, 1e-25, 1e-10, 0.05, 0.2, 0.4, 0.5])
        for degree in [1, 3, 4, 6]:
            recovered = inverse_parity_probability(parity_probability(rates, degree), degree)
            np.testing.assert_allclose(recovered, rates, rtol=1e-10, atol=1e-13)
            self.assertEqual(float(inverse_parity_probability(0.5, degree)), 0.5)
        np.testing.assert_array_equal(inverse_parity_probability([0, 0.1, 0.5], 4), exact_inverse([0, 0.1, 0.5]))
        self.assertAlmostEqual(float(inverse_parity_probability(parity_probability(0.8, 6), 6)), 0.2)
        with self.assertRaises(ValueError):
            inverse_parity_probability(parity_probability(0.8, 3), 3)

    def test_invalid_probabilities_and_degrees_rejected(self):
        for degree in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                parity_probability(0.1, degree)
        for degree in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                inverse_parity_probability(0.1, degree)
        for value in [-0.1, 1.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                parity_probability(value, 3)
        with self.assertRaises(ValueError):
            inverse_parity_probability(0.6, 3)


if __name__ == "__main__":
    unittest.main()
