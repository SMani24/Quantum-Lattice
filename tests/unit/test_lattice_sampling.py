"""Sampling regressions against the original square implementation."""

import unittest

import numpy as np

from quantum_lattice.lattice import Lattice, square_lattice
from quantum_lattice.lattice.square import defect_fractions
from quantum_lattice.metrics.calibration import exact_vertex_rate
from quantum_lattice.simulation.phase_errors import (
    run_calibration_point,
    run_lattice_calibration_point,
    sample_lattice_errors,
    sample_phase_errors,
)


def original_square_point(side, rate, patterns, batch_size, sequence):
    """Keep the pre-refactor square reduction as an independent regression path."""
    rng = np.random.default_rng(sequence)
    count, total, total_squared = 0, 0.0, 0.0
    while count < patterns:
        size = min(batch_size, patterns - count)
        flags = sample_phase_errors(rng, side, rate, size)
        fractions = defect_fractions(flags)
        total += float(fractions.sum(dtype=np.float64))
        total_squared += float(np.square(fractions).sum(dtype=np.float64))
        count += size
    mean = total / count
    variance = max(0.0, (total_squared - total**2 / count) / (count - 1))
    se = float(np.sqrt(variance / count))
    analytic = float(exact_vertex_rate(rate))
    return {
        "L": side, "e_r": float(rate), "patterns": count,
        "mean_p": mean, "se_p": se, "analytic_p": analytic,
        "z_score": (mean - analytic) / se if se else 0.0,
        "mean_stabilizer": 1 - 2 * mean,
        "stream_spawn_key": list(sequence.spawn_key),
    }


class LatticeSamplingTests(unittest.TestCase):
    def test_flat_edge_draws_preserve_square_random_stream(self):
        lattice = square_lattice(5)
        sequence = np.random.SeedSequence(771, spawn_key=(5, 8))
        flat = sample_lattice_errors(np.random.default_rng(sequence), lattice, 0.17, 37)
        grid = sample_phase_errors(np.random.default_rng(sequence), 5, 0.17, 37)
        np.testing.assert_array_equal(flat.reshape(grid.shape), grid)

    def test_square_statistics_and_row_schema_match_original_exactly(self):
        for side in [3, 8]:
            lattice = square_lattice(side)
            for index, rate in enumerate([0, 0.005, 0.1, 0.2]):
                sequence = np.random.SeedSequence(20261012, spawn_key=(side, index))
                expected = original_square_point(side, rate, 1003, 37, sequence)
                actual = run_calibration_point(side, rate, 1003, 37, sequence, lattice=lattice)
                self.assertEqual(actual, expected)
                self.assertEqual(list(actual), list(expected))

    def test_generic_point_has_non_square_mean_and_whole_pattern_uncertainty(self):
        lattice = Lattice(
            4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            [(0, 3, 1), (0, 4, 2), (1, 5, 2), (3, 5, 4)],
            boundary="closed", geometry="tetrahedral_fixture",
        )
        row = run_lattice_calibration_point(lattice, 0.2, 6000, 137, np.random.SeedSequence(416))
        self.assertEqual((row["n_vertices"], row["n_edges"], row["n_faces"]), (4, 6, 4))
        self.assertAlmostEqual(row["analytic_p"], (1 - 0.6**3) / 2)
        self.assertLess(abs(row["z_score"]), 5)
        self.assertLess(abs(row["se_p"] / row["analytic_se_p"] - 1), 0.05)
        self.assertNotIn("L", row)

    def test_sampling_does_not_mutate_topology(self):
        lattice = square_lattice(3)
        before = (lattice.vertex_edge_indices.tobytes(), lattice.face_edge_indices.tobytes())
        run_lattice_calibration_point(lattice, 0.15, 321, 27, np.random.SeedSequence(43))
        self.assertEqual(before, (lattice.vertex_edge_indices.tobytes(), lattice.face_edge_indices.tobytes()))

    def test_invalid_lattices_and_sampling_parameters_rejected(self):
        seed = np.random.SeedSequence(1)
        with self.assertRaises(ValueError):
            run_lattice_calibration_point(None, 0.1, 100, 30, seed)
        for lattice in [Lattice(4, [(0, 1), (1, 2)]), square_lattice(4)]:
            with self.assertRaises(ValueError):
                run_calibration_point(3, 0.1, 100, 30, seed, lattice=lattice)
        for updates in [{"error_rate": -0.1}, {"patterns": 1}, {"batch_size": 0}, {"seed_sequence": 42}]:
            arguments = dict(lattice=square_lattice(3), error_rate=0.1, patterns=100, batch_size=30, seed_sequence=seed)
            arguments.update(updates)
            with self.assertRaises(ValueError):
                run_lattice_calibration_point(**arguments)


if __name__ == "__main__":
    unittest.main()
