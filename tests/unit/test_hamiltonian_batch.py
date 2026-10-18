"""Compiled batches against the separately implemented scalar D5 oracle."""

import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.lattice import Lattice
from quantum_lattice.metrics.hamiltonian import hamiltonian_error
from quantum_lattice.metrics.hamiltonian_batch import CompiledHamiltonianMetric


def _torus(L):
    """Coordinate fixture independent of the production square builder."""
    vertex = lambda x, y: (y % L) * L + x % L
    edge = lambda x, y, axis: 2 * vertex(x, y) + axis
    endpoints = []
    boundaries = []
    for y in range(L):
        for x in range(L):
            endpoints.extend([(vertex(x, y), vertex(x + 1, y)),
                              (vertex(x, y), vertex(x, y + 1))])
            boundaries.append((edge(x, y, 0), edge(x + 1, y, 1),
                               edge(x, y + 1, 0), edge(x, y, 1)))
    return Lattice(L * L, endpoints, boundaries, boundary="periodic")


class HamiltonianBatchTests(unittest.TestCase):
    def assertScalarAgreement(self, lattice, true_z, recovered_z, true_x=None,
                              recovered_x=None, model="two_channel", **tolerances):
        compiled = CompiledHamiltonianMetric(lattice, model=model)
        actual = compiled.distances(true_z, recovered_z, true_x, recovered_x)
        self.assertEqual(set(actual), {"full", "traceless"})
        for name, identity in (("full", True), ("traceless", False)):
            expected = [hamiltonian_error(
                lattice, true_z[i], recovered_z[i],
                None if true_x is None else true_x[i],
                None if recovered_x is None else recovered_x[i],
                model=model, include_identity=identity,
            ) for i in range(len(true_z))]
            self.assertEqual(actual[name].shape, (len(true_z),))
            self.assertEqual(actual[name].dtype, np.float64)
            np.testing.assert_allclose(actual[name], expected,
                                       **({"rtol": 2e-13, "atol": 2e-15} | tolerances))
        return actual

    def test_signed_random_l3_l4_both_models_match_scalar(self):
        rng = np.random.default_rng(241018)
        for L in (3, 4):
            lattice = _torus(L)
            fields = [rng.uniform(-2, 2, (7, lattice.n_edges)) for _ in range(4)]
            for model in ("z_branch", "two_channel"):
                with self.subTest(L=L, model=model):
                    self.assertScalarAgreement(lattice, *fields[:2],
                                               *(fields[2:] if model == "two_channel" else [None, None]),
                                               model=model)

    def test_zero_x_default_and_zero_fields_both_models(self):
        lattice = _torus(3)
        z = np.zeros((3, lattice.n_edges))
        recovered = np.linspace(-.7, .8, z.size).reshape(z.shape)
        for model in ("z_branch", "two_channel"):
            actual = self.assertScalarAgreement(lattice, z, recovered, model=model)
            explicit = CompiledHamiltonianMetric(lattice, model=model).distances(z, recovered, z, z)
            for key in actual:
                np.testing.assert_array_equal(actual[key], explicit[key])
        branch = CompiledHamiltonianMetric(lattice, model="z_branch").distances(z, recovered)
        both = CompiledHamiltonianMetric(lattice).distances(z, recovered)
        self.assertFalse(np.allclose(branch["full"], both["full"]))
        np.testing.assert_allclose(branch["traceless"], both["traceless"], atol=2e-15)

    def test_equal_fields_exact_zero_and_tiny_error_is_not_cancelled(self):
        lattice = _torus(4)
        rng = np.random.default_rng(7)
        z = rng.uniform(-.8, .8, (3, lattice.n_edges))
        x = rng.uniform(-.8, .8, z.shape)
        compiled = CompiledHamiltonianMetric(lattice)
        zero = compiled.distances(z, z, x, x)
        for values in zero.values():
            np.testing.assert_array_equal(values, np.zeros(len(z)))
        altered = z.copy()
        altered[:, 0] += 1e-11
        tiny = self.assertScalarAgreement(lattice, z, altered, x, x, rtol=1e-3, atol=2e-16)
        for values in tiny.values():
            self.assertTrue(np.all(values > 1e-14))
            self.assertTrue(np.all(values < 1e-10))

    def test_strong_fields_stay_finite_for_both_models(self):
        lattice = _torus(3)
        signs = np.where(np.arange(lattice.n_edges) % 2, -1., 1.)
        z = np.stack((signs * 1000, np.ones(lattice.n_edges) * -1000))
        for model in ("z_branch", "two_channel"):
            values = self.assertScalarAgreement(lattice, z, -z,
                                               -z if model == "two_channel" else None,
                                               z if model == "two_channel" else None,
                                               model=model)
            self.assertTrue(all(np.all(np.isfinite(score)) for score in values.values()))
            self.assertTrue(all(np.all((score >= 0) & (score <= 2)) for score in values.values()))
        # Unequal local log-cosh scales exercise their relative weights too,
        # rather than only testing equal-magnitude saturated tanh factors.
        mixed = z + np.linspace(-1.3, 2.1, lattice.n_edges)
        self.assertScalarAgreement(lattice, mixed, mixed + np.linspace(-.2, .3, lattice.n_edges),
                                   z, z + .07, rtol=2e-11, atol=2e-13)

    def test_repeated_supports_and_fixed_operator_collisions_merge(self):
        lattice = Lattice(3, [(0, 1), (1, 2), (2, 0)], [(0, 1, 2), (0, 1, 2)])
        z = np.array([[.4, -.7, .2], [-.9, -.2, .6]])
        x = np.array([[.6, -.3, .4], [-.2, -.1, .7]])
        self.assertScalarAgreement(lattice, z, -z / 2, x, x + .2)
        compiled = CompiledHamiltonianMetric(lattice)
        self.assertEqual(len(compiled.pauli_keys), len(set(compiled.pauli_keys)))
        self.assertIn(("X", (0, 1)), compiled.pauli_keys)

    def test_isolated_identity_collision_and_repeated_stars(self):
        # Both connected vertices have the same star X0. The isolated vertex
        # contributes -I + exp(0)I, which must cancel before normalization.
        connected = Lattice(2, [(0, 1)])
        isolated = Lattice(3, [(0, 1)])
        z = np.array([[.2], [-.7]])
        first = self.assertScalarAgreement(connected, z, -z)
        second = self.assertScalarAgreement(isolated, z, -z)
        for key in first:
            np.testing.assert_allclose(first[key], second[key], atol=2e-15)

    def test_batch_order_chunking_symmetry_and_input_preservation(self):
        lattice = _torus(3)
        rng = np.random.default_rng(932)
        true = rng.normal(size=(9, lattice.n_edges))
        recovered = rng.normal(size=true.shape)
        before = true.copy(), recovered.copy()
        compiled = CompiledHamiltonianMetric(lattice)
        all_values = compiled.distances(true, recovered)
        reverse = compiled.distances(recovered, true)
        permutation = [7, 1, 5, 2, 8, 3, 4, 0, 6]
        permuted = compiled.distances(true[permutation], recovered[permutation])
        left = compiled.distances(true[:4], recovered[:4])
        right = compiled.distances(true[4:], recovered[4:])
        for key in all_values:
            np.testing.assert_array_equal(all_values[key], reverse[key])
            np.testing.assert_array_equal(all_values[key][permutation], permuted[key])
            np.testing.assert_array_equal(all_values[key], np.r_[left[key], right[key]])
        np.testing.assert_array_equal(true, before[0])
        np.testing.assert_array_equal(recovered, before[1])

    def test_production_batch_does_not_call_scalar_oracle(self):
        lattice = _torus(3)
        fields = np.zeros((2, lattice.n_edges))
        compiled = CompiledHamiltonianMetric(lattice)
        with patch("quantum_lattice.metrics.hamiltonian.hamiltonian_error", side_effect=AssertionError), \
                patch("quantum_lattice.metrics.hamiltonian.pauli_coefficients", side_effect=AssertionError):
            result = compiled.distances(fields, fields + .1)
        self.assertTrue(np.all(result["full"] > 0))

    def test_invalid_inputs_shapes_types_fields_and_models(self):
        lattice = _torus(3)
        compiled = CompiledHamiltonianMetric(lattice)
        valid = np.zeros((2, lattice.n_edges))
        invalid = [valid[0], valid[:, :-1], valid[:0], valid[:1],
                   np.full(valid.shape, np.nan), np.full(valid.shape, np.inf),
                   np.ones(valid.shape, dtype=bool), np.zeros(valid.shape, dtype=complex),
                   np.full(valid.shape, "0"), np.full(valid.shape, object())]
        for values in invalid:
            with self.subTest(shape=np.shape(values), dtype=np.asarray(values).dtype):
                for position in range(4):
                    args = [valid, valid, valid, valid]
                    args[position] = values
                    with self.assertRaises(ValueError):
                        compiled.distances(*args)
        for model in (None, "unknown", True):
            with self.assertRaises(ValueError):
                CompiledHamiltonianMetric(lattice, model=model)
        with self.assertRaises(ValueError):
            CompiledHamiltonianMetric(object())
        branch = CompiledHamiltonianMetric(lattice, model="z_branch")
        for position in (2, 3):
            args = [valid] * 4
            args[position] = valid + .1
            with self.assertRaises(ValueError):
                branch.distances(*args)

    def test_enumeration_and_nonfinite_log_scale_guards(self):
        high_degree = Lattice(22, [(0, leaf) for leaf in range(1, 22)])
        with self.assertRaisesRegex(ValueError, "support exceeds"):
            CompiledHamiltonianMetric(high_degree)
        lattice = _torus(3)
        with patch("quantum_lattice.metrics.hamiltonian_batch._MAX_COMPILED_TERMS", 10):
            with self.assertRaisesRegex(ValueError, "safe enumeration"):
                CompiledHamiltonianMetric(lattice)
        compiled = CompiledHamiltonianMetric(lattice)
        huge = np.full((1, lattice.n_edges), np.finfo(np.float64).max)
        with self.assertRaisesRegex(ValueError, "logarithmic scale"):
            compiled.distances(huge, huge)
        with self.assertRaises(ValueError):
            CompiledHamiltonianMetric(Lattice(2, [])).distances(np.empty((1, 0)), np.empty((1, 0)))


if __name__ == "__main__":
    unittest.main()
