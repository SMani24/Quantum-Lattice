"""Checks for toric syndrome geometry and independent-error calibration."""

import itertools
import unittest

import numpy as np

from quantum_lattice.lattice.square import defect_fractions, star_syndrome
from quantum_lattice.metrics.calibration import (
    analytic_pattern_variance,
    evaluate_polynomial,
    exact_inverse,
    exact_vertex_rate,
    fit_inverse,
    published_inverse,
)
from quantum_lattice.simulation.phase_errors import (
    run_calibration_point,
    sample_phase_errors,
)


class GeometryTests(unittest.TestCase):
    def test_single_horizontal_and_vertical_edges_toggle_their_endpoints(self):
        for orientation, expected in (
            (0, {(2, 3), (2, 0)}),
            (1, {(2, 3), (3, 3)}),
        ):
            errors = np.zeros((4, 4, 2), dtype=bool)
            errors[2, 3, orientation] = True
            actual = set(map(tuple, np.argwhere(star_syndrome(errors))))
            self.assertEqual(actual, expected)

    def test_two_adjacent_edges_cancel_at_shared_vertex(self):
        errors = np.zeros((4, 4, 2), dtype=bool)
        errors[1, 0, 0] = True
        errors[1, 1, 0] = True
        actual = set(map(tuple, np.argwhere(star_syndrome(errors))))
        self.assertEqual(actual, {(1, 0), (1, 2)})

    def test_closed_loop_has_no_syndrome(self):
        errors = np.zeros((4, 4, 2), dtype=bool)
        errors[0, 0, 0] = errors[1, 0, 0] = True
        errors[0, 0, 1] = errors[0, 1, 1] = True
        self.assertFalse(star_syndrome(errors).any())
        # A noncontractible error loop likewise has no local syndrome.
        errors[:] = False
        errors[2, :, 0] = True
        self.assertFalse(star_syndrome(errors).any())

    def test_total_torus_syndrome_parity_is_even(self):
        rng = np.random.default_rng(31)
        errors = sample_phase_errors(rng, 7, 0.17, 250)
        syndrome = star_syndrome(errors)
        self.assertTrue(np.all(syndrome.sum(axis=(-2, -1)) % 2 == 0))
        np.testing.assert_allclose(defect_fractions(errors), syndrome.mean(axis=(-2, -1)))

    def test_invalid_lattices_and_nonbinary_errors_are_rejected(self):
        for errors in (np.zeros((2, 2, 2)), np.zeros((3, 4, 2)), np.zeros((4, 4, 3))):
            with self.assertRaises(ValueError):
                star_syndrome(errors)
        with self.assertRaises(ValueError):
            star_syndrome(np.full((4, 4, 2), 0.2))


class ProbabilityTests(unittest.TestCase):
    def test_vertex_formula_matches_all_sixteen_local_error_patterns(self):
        for rate in (0.0, 0.003, 0.1, 0.2, 0.5, 0.83, 1.0):
            enumerated = 0.0
            for pattern in itertools.product((0, 1), repeat=4):
                errors = sum(pattern)
                probability = rate**errors * (1 - rate) ** (4 - errors)
                if errors % 2:
                    enumerated += probability
            self.assertAlmostEqual(float(exact_vertex_rate(rate)), enumerated, places=14)

    def test_full_pattern_variance_includes_shared_edge_covariance(self):
        # Enumerate every edge-error pattern on the smallest allowed torus.
        # This checks the actual periodic geometry, not independent stars.
        ids = np.arange(2**18, dtype=np.uint32)
        bits = ((ids[:, None] >> np.arange(18)) & 1).astype(bool)
        errors = bits.reshape(-1, 3, 3, 2)
        fractions = defect_fractions(errors)
        counts = bits.sum(axis=1)
        for rate in (0.1, 0.2):
            probabilities = rate**counts * (1 - rate) ** (18 - counts)
            mean = np.sum(probabilities * fractions)
            variance = np.sum(probabilities * (fractions - mean) ** 2)
            self.assertAlmostEqual(float(analytic_pattern_variance(rate, 3)), variance, places=14)

    def test_inverse_roundtrip_and_low_error_branch(self):
        rates = np.array([0, 1e-12, 0.005, 0.1, 0.2, 0.4, 0.5])
        np.testing.assert_allclose(exact_inverse(exact_vertex_rate(rates)), rates, atol=1e-13)
        self.assertAlmostEqual(float(exact_inverse(exact_vertex_rate(0.8))), 0.2)
        for rate in (-0.01, 0.51, np.nan):
            with self.assertRaises(ValueError):
                exact_inverse(rate)


class SamplingTests(unittest.TestCase):
    def test_zero_rate_is_deterministic_and_serializable(self):
        import json

        row = run_calibration_point(4, 0, 150, 37, np.random.SeedSequence(14, spawn_key=(2, 3)))
        self.assertEqual(row["mean_p"], 0)
        self.assertEqual(row["se_p"], 0)
        self.assertEqual(row["z_score"], 0)
        self.assertEqual(row["mean_stabilizer"], 1)
        self.assertEqual(row["stream_spawn_key"], [2, 3])
        json.dumps(row, allow_nan=False)

    def test_reproducible_stream_and_batching(self):
        seed = np.random.SeedSequence(111, spawn_key=(3, 6))
        first = run_calibration_point(4, 0.1, 1237, 256, seed)
        again = run_calibration_point(4, 0.1, 1237, 256, seed)
        smaller_batch = run_calibration_point(4, 0.1, 1237, 31, seed)
        self.assertEqual(first, again)
        self.assertAlmostEqual(first["mean_p"], smaller_batch["mean_p"], places=15)
        self.assertAlmostEqual(first["se_p"], smaller_batch["se_p"], places=15)

    def test_sampling_mean_and_uncertainty_match_analytic_distribution(self):
        row = run_calibration_point(8, 0.12, 10000, 113, np.random.SeedSequence(890))
        self.assertLess(abs(row["z_score"]), 5)
        expected_se = float(np.sqrt(analytic_pattern_variance(0.12, 8) / row["patterns"]))
        self.assertLess(abs(row["se_p"] / expected_se - 1), 0.05)

    def test_invalid_sampling_settings_are_rejected(self):
        seed = np.random.SeedSequence(1)
        for kwargs in (
            {"side_length": 2}, {"error_rate": -0.1}, {"error_rate": np.nan},
            {"patterns": 1}, {"batch_size": 0}, {"seed_sequence": 42},
        ):
            arguments = dict(side_length=4, error_rate=0.1, patterns=100, batch_size=30, seed_sequence=seed)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                run_calibration_point(**arguments)


class FitTests(unittest.TestCase):
    def test_fit_recovers_known_polynomial_and_zero_intercept(self):
        p = np.linspace(0, 0.45, 41)
        coefficients = [0.25, 0.3, -0.2, 0.8]
        e = evaluate_polynomial(p, coefficients)
        fit = fit_inverse(p, e)
        np.testing.assert_allclose(fit["coefficients"], coefficients, atol=1e-12)
        self.assertEqual(fit["powers"], [1, 2, 3, 4])
        self.assertEqual(float(evaluate_polynomial(0, fit["coefficients"])), 0)
        self.assertLess(fit["diagnostics"]["mse"], 1e-25)

    def test_fresh_fit_approximates_inverse_in_calibration_range(self):
        e = np.linspace(0, 0.2, 41)
        p = exact_vertex_rate(e)
        fit = fit_inverse(p, e)
        self.assertLess(fit["diagnostics"]["rmse"], 0.001)
        self.assertAlmostEqual(float(published_inverse(0.1)), 0.027062218)

    def test_unsupported_or_malformed_fits_are_rejected(self):
        for p, e in (([0, 0, 0, 0], [0, 0, 0, 0]), ([0.1], [0.05]), ([[0.1]], [[0.05]])):
            with self.assertRaises(ValueError):
                fit_inverse(p, e)


if __name__ == "__main__":
    unittest.main()
