"""Independent weighted enumeration, derivative and continuation oracles."""
import itertools
import json
import unittest

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.models.inverse_ising import fit_inverse_ising, pseudolikelihood_objective


def exact_states():
    lattice = square_lattice(3)
    tau = np.asarray([(1,) + bits for bits in itertools.product((-1, 1), repeat=8)], dtype=np.int8)
    ends = lattice.edge_vertex_indices
    return lattice, tau, tau[:, ends[:, 0]] * tau[:, ends[:, 1]]


class InverseIsingTests(unittest.TestCase):
    def test_loss_matches_vertex_conditional_oracle(self):
        lattice, tau, states = exact_states()
        fields = np.linspace(-0.3, 0.4, 18)
        weights = np.arange(1, len(states) + 1, dtype=np.float64)
        weights /= weights.sum()
        loss, gradient, hessian = pseudolikelihood_objective(states, fields, weights=weights, ridge=0.003)
        expected = 0.0
        for vertex in lattice.vertices:
            local = np.zeros(len(states))
            for edge in vertex.incident_edges:
                other = edge.vertex_ids[1] if edge.vertex_ids[0] == vertex.id else edge.vertex_ids[0]
                local += fields[edge.id] * tau[:, other]
            expected += np.dot(weights, np.logaddexp(0.0, -2.0 * tau[:, vertex.id] * local)) / 9
        expected += 0.003 / 2 * np.dot(fields, fields)
        self.assertAlmostEqual(loss, expected, places=14)
        self.assertEqual(gradient.shape, (18,))
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-16)
        self.assertGreater(np.linalg.eigvalsh(hessian).min(), 0)

    def test_analytic_gradient_and_hessian_against_finite_differences(self):
        _, _, states = exact_states()
        fields = np.linspace(-0.25, 0.5, 18)
        _, gradient, hessian = pseudolikelihood_objective(states, fields)
        step = 1e-5
        numeric_gradient = np.empty(18)
        numeric_hessian = np.empty((18, 18))
        for edge in range(18):
            offset = np.zeros(18)
            offset[edge] = step
            plus = pseudolikelihood_objective(states, fields + offset)
            minus = pseudolikelihood_objective(states, fields - offset)
            numeric_gradient[edge] = (plus[0] - minus[0]) / (2 * step)
            numeric_hessian[:, edge] = (plus[1] - minus[1]) / (2 * step)
        np.testing.assert_allclose(gradient, numeric_gradient, rtol=2e-8, atol=2e-11)
        np.testing.assert_allclose(hessian, numeric_hessian, rtol=2e-8, atol=2e-11)

    def test_exact_weighted_population_recovers_signed_couplings(self):
        _, _, states = exact_states()
        truth = np.random.default_rng(19).uniform(-0.35, 0.35, 18)
        weights = np.exp(states @ truth)
        weights /= weights.sum()
        _, gradient, _ = pseudolikelihood_objective(states, truth, weights=weights, ridge=0)
        np.testing.assert_allclose(gradient, 0, atol=2e-16)
        result = fit_inverse_ising(states, weights=weights, ridge=0, tolerance=1e-10)
        self.assertTrue(result["converged"])
        np.testing.assert_allclose(result["fields"], truth, atol=2e-8, rtol=0)

    def test_duplicate_records_equal_frequency_weights_and_uniform_zero_fit(self):
        _, _, states = exact_states()
        counts = np.arange(1, len(states) + 1) % 3 + 1
        duplicate = np.repeat(states, counts, axis=0)
        fields = np.linspace(-0.4, 0.4, 18)
        a = pseudolikelihood_objective(duplicate, fields)
        b = pseudolikelihood_objective(states, fields, weights=counts)
        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right, atol=3e-16)
        result = fit_inverse_ising(states)
        self.assertTrue(result["converged"])
        self.assertEqual(result["iteration"], 0)
        np.testing.assert_array_equal(result["fields"], np.zeros(18))

    def test_pause_resume_exact_and_monotone_objective(self):
        _, _, states = exact_states()
        weights = np.exp(states @ np.linspace(-0.6, 0.6, 18))
        history = []
        complete = fit_inverse_ising(states, weights=weights, tolerance=1e-10, callback=lambda state: history.append(state))
        paused = fit_inverse_ising(states, weights=weights, tolerance=1e-10, callback=lambda state: False)
        self.assertEqual(paused["stop_reason"], "paused")
        self.assertEqual(paused["iteration"], 1)
        restored = json.loads(json.dumps(paused["optimizer_state"], allow_nan=False))
        resumed = fit_inverse_ising(states, weights=weights, tolerance=1e-10, initial=restored)
        self.assertEqual(resumed, complete)
        self.assertTrue(all(later["objective"] <= earlier["objective"] for earlier, later in zip(history, history[1:])))
        capped = fit_inverse_ising(states, weights=weights, max_iterations=1, tolerance=1e-10)
        self.assertEqual(capped["stop_reason"], "max_iterations")
        self.assertFalse(capped["converged"])

    def test_invalid_sector_and_resume_binding_rejected(self):
        _, _, states = exact_states()
        weights = np.exp(states @ np.linspace(-0.3, 0.3, 18))
        initial = fit_inverse_ising(states, weights=weights, max_iterations=1)
        for change in ({"weights": weights[::-1]}, {"weights": weights, "ridge": 0.002}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                fit_inverse_ising(states, initial=initial, **change)
        invalid = states.copy()
        invalid[0, 0] *= -1
        with self.assertRaisesRegex(ValueError, "sector"):
            fit_inverse_ising(invalid)
        for kwargs in ({"ridge": -1}, {"tolerance": 0}, {"max_iterations": True}, {"weights": np.zeros(len(states))}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                fit_inverse_ising(states, **kwargs)


if __name__ == "__main__":
    unittest.main()
