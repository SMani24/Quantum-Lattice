"""Independent small-matrix oracles for the normalized Hamiltonian metric.

The oracle builds tensor-product matrix exponentials directly. It never uses
the production subset expansion, and its largest matrix is only 256 x 256.
"""

from itertools import combinations
import unittest

import numpy as np

from quantum_lattice.lattice import Lattice, square_lattice
from quantum_lattice.metrics.hamiltonian import (
    PauliExpansion,
    hamiltonian_error,
    normalized_coefficient_distance,
    normalized_pauli_distance,
    pauli_coefficients,
)


IDENTITY = ("I", ())
LOCAL = {
    "I": np.eye(2),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Z": np.diag([1.0, -1.0]),
}


def _tensor(factors):
    result = np.ones((1, 1))
    for factor in factors:
        result = np.kron(result, factor)
    return result


def _pauli(n_qubits, axis, support):
    return _tensor([LOCAL[axis] if edge in support else LOCAL["I"]
                    for edge in range(n_qubits)])


def _local_exponential(axis, field):
    """Diagonalize a literal 2x2 generator instead of expanding cosh/sinh."""
    eigenvalues, eigenvectors = np.linalg.eigh(-field * LOCAL[axis])
    return (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T


def _dense_hamiltonian(lattice, b_z, b_x=None, *, model="two_channel",
                       include_identity=True):
    n = lattice.n_edges
    result = np.zeros((2 ** n, 2 ** n))
    b_z = np.asarray(b_z)
    b_x = np.zeros(n) if b_x is None else np.asarray(b_x)
    for vertex in lattice.vertices:
        result -= _pauli(n, "X", vertex.edge_ids)
        result += _tensor([
            _local_exponential("Z", b_z[edge])
            if edge in vertex.edge_ids else LOCAL["I"]
            for edge in range(n)
        ])
    for face in lattice.faces:
        result -= _pauli(n, "Z", face.edge_ids)
        if model == "two_channel":
            result += _tensor([
                _local_exponential("X", b_x[edge])
                if edge in face.edge_ids else LOCAL["I"]
                for edge in range(n)
            ])
    if not include_identity:
        result -= np.eye(2 ** n) * (np.trace(result) / (2 ** n))
    return result


def _dense_from_expansion(expansion):
    result = np.zeros((2 ** expansion.n_qubits, 2 ** expansion.n_qubits))
    for (axis, support), value in expansion.coefficients.items():
        result += value * _pauli(expansion.n_qubits, axis, support)
    return result * np.exp(expansion.log_scale)


def _dense_distance(left, right):
    return np.linalg.norm(left / np.linalg.norm(left) - right / np.linalg.norm(right))


def _triangle(*, isolated=False, doubled_face=False):
    boundaries = [(0, 1, 2)] * (2 if doubled_face else 1)
    return Lattice(4 if isolated else 3, [(0, 1), (1, 2), (2, 0)], boundaries)


def _two_by_two_torus():
    """Independent fixture: the normal square builder intentionally rejects L2."""
    return Lattice(
        4,
        [(0, 1), (0, 2), (1, 0), (1, 3),
         (2, 3), (2, 0), (3, 2), (3, 1)],
        [(0, 3, 4, 1), (2, 1, 6, 3),
         (4, 7, 0, 5), (6, 5, 2, 7)],
        geometry="independent_L2_fixture", boundary="periodic",
    )


class HamiltonianMetricTests(unittest.TestCase):
    def assertExpansionMatchesMatrix(self, lattice, b_z, b_x=None, **kwargs):
        expansion = pauli_coefficients(lattice, b_z, b_x, **kwargs)
        expected = _dense_hamiltonian(lattice, b_z, b_x, **kwargs)
        np.testing.assert_allclose(_dense_from_expansion(expansion), expected,
                                   rtol=3e-14, atol=3e-14)
        self.assertEqual(expansion.n_qubits, lattice.n_edges)
        self.assertEqual(expansion.model, kwargs.get("model", "two_channel"))
        self.assertEqual(expansion.include_identity, kwargs.get("include_identity", True))
        # The matrix trace checks every pure X/Z string, including missing keys.
        # This catches both incorrect coefficients and silently omitted strings.
        for axis in ("X", "Z"):
            for count in range(1, lattice.n_edges + 1):
                for support in combinations(range(lattice.n_edges), count):
                    key = (axis, support)
                    trace_coefficient = np.sum(_pauli(lattice.n_edges, axis, support).T
                                               * expected) / (2 ** lattice.n_edges)
                    actual = expansion.coefficients.get(key, 0.0) * np.exp(expansion.log_scale)
                    self.assertAlmostEqual(actual, trace_coefficient, delta=4e-13)
        actual_identity = expansion.coefficients.get(IDENTITY, 0.0) * np.exp(expansion.log_scale)
        self.assertAlmostEqual(actual_identity, np.trace(expected) / len(expected), delta=4e-13)
        return expansion, expected

    def test_triangle_full_and_branch_match_independent_matrix_exponential(self):
        lattice = _triangle()
        b_z, b_x = [.2, -.5, .8], [-.4, .7, .1]
        for model in ("two_channel", "z_branch"):
            for identity in (True, False):
                with self.subTest(model=model, identity=identity):
                    self.assertExpansionMatchesMatrix(
                        lattice, b_z, b_x if model == "two_channel" else None,
                        model=model, include_identity=identity)

    def test_eight_qubit_torus_full_hamiltonian_and_all_coefficients(self):
        lattice = _two_by_two_torus()
        b_z = np.linspace(-.43, .67, 8)
        b_x = np.array([.19, -.24, .36, .11, -.51, .09, .28, -.13])
        self.assertExpansionMatchesMatrix(lattice, b_z, b_x)

    def test_overlapping_operators_add_before_normalization(self):
        lattice = _triangle(doubled_face=True)
        expansion, _ = self.assertExpansionMatchesMatrix(lattice, [.3, -.2, .4], [.7, -.1, .5])
        # X0X1 occurs both as one bare star and in both face exponentials.
        expected = -1 + 2 * np.sinh(.7) * np.sinh(-.1) * np.cosh(.5)
        actual = expansion.coefficients[("X", (0, 1))] * np.exp(expansion.log_scale)
        self.assertAlmostEqual(actual, expected, places=13)
        self.assertAlmostEqual(expansion.coefficients[("Z", (0, 1, 2))]
                               * np.exp(expansion.log_scale), -2, places=13)

    def test_repeated_stars_and_isolated_vertex_cancel_correctly(self):
        self.assertExpansionMatchesMatrix(Lattice(3, [(0, 1)]), [.41], model="z_branch")
        base, _ = self.assertExpansionMatchesMatrix(_triangle(), [.2, -.1, .3])
        isolated, _ = self.assertExpansionMatchesMatrix(_triangle(isolated=True), [.2, -.1, .3])
        self.assertAlmostEqual(normalized_pauli_distance(base, isolated), 0.0, delta=1e-14)

    def test_full_metric_matches_frobenius_norm_on_signed_fields(self):
        lattice = _triangle()
        true_z, true_x = [.2, -.5, .8], [-.4, .7, .1]
        recovered_z, recovered_x = [-.1, -.4, .6], [-.2, .65, -.1]
        for model in ("two_channel", "z_branch"):
            for identity in (True, False):
                kwargs = dict(model=model, include_identity=identity)
                left = _dense_hamiltonian(lattice, true_z,
                                          true_x if model == "two_channel" else None, **kwargs)
                right = _dense_hamiltonian(lattice, recovered_z,
                                           recovered_x if model == "two_channel" else None, **kwargs)
                actual = hamiltonian_error(
                    lattice, true_z, recovered_z,
                    true_x if model == "two_channel" else None,
                    recovered_x if model == "two_channel" else None, **kwargs)
                with self.subTest(model=model, identity=identity):
                    self.assertAlmostEqual(actual, _dense_distance(left, right), places=13)

    def test_omitted_x_is_zero_x_but_two_channel_is_not_z_branch(self):
        lattice = _triangle()
        b_z = [.4, -.2, .1]
        implicit = pauli_coefficients(lattice, b_z)
        explicit = pauli_coefficients(lattice, b_z, np.zeros(3))
        self.assertAlmostEqual(normalized_pauli_distance(implicit, explicit), 0.0, delta=1e-14)
        branch = pauli_coefficients(lattice, b_z, model="z_branch")
        self.assertFalse(np.allclose(_dense_from_expansion(implicit), _dense_from_expansion(branch)))
        np.testing.assert_allclose(_dense_from_expansion(implicit) - _dense_from_expansion(branch),
                                   np.eye(8), atol=1e-14)

    def test_sign_information_changes_hamiltonian_metric(self):
        lattice = _triangle()
        fields = np.array([.3, -.5, .7])
        self.assertGreater(hamiltonian_error(lattice, fields, np.abs(fields)), .1)
        self.assertAlmostEqual(hamiltonian_error(lattice, fields, fields), 0.0, delta=1e-14)

    def test_primitive_is_unit_vector_distance_with_sparse_union_and_scale(self):
        x = ("X", (0,))
        z = ("Z", (0,))
        self.assertAlmostEqual(normalized_coefficient_distance({x: 3., z: 4.}, {x: 6., z: 8.}), 0.)
        self.assertAlmostEqual(normalized_coefficient_distance({x: 3., z: 4.}, {x: -3., z: -4.}), 2.)
        self.assertAlmostEqual(normalized_coefficient_distance({x: 1.}, {z: 1.}), np.sqrt(2))
        self.assertAlmostEqual(normalized_coefficient_distance({x: 1e300, z: 2e300},
                                                               {x: 1e-300, z: 2e-300}), 0., delta=1e-14)
        self.assertAlmostEqual(normalized_coefficient_distance({x: 1., z: 0.}, {x: 1.}), 0.)

    def test_metric_symmetry_bounds_and_log_scale_independence(self):
        lattice = _triangle()
        left = pauli_coefficients(lattice, [.2, -.3, .4])
        right = pauli_coefficients(lattice, [.1, .5, -.7])
        forward = normalized_pauli_distance(left, right)
        self.assertAlmostEqual(forward, normalized_pauli_distance(right, left), places=15)
        self.assertTrue(0 <= forward <= 2)
        scaled = PauliExpansion(left.n_qubits, left.coefficients, left.log_scale + 900,
                                left.model, left.include_identity)
        self.assertAlmostEqual(normalized_pauli_distance(left, scaled), 0., delta=1e-14)

    def test_identity_inclusion_changes_distance_under_energy_shift(self):
        x = ("X", (0,))
        left = PauliExpansion(1, {x: -1., IDENTITY: 2.}, 0., "two_channel", True)
        shifted = PauliExpansion(1, {x: -1., IDENTITY: 20.}, 0., "two_channel", True)
        self.assertGreater(normalized_pauli_distance(left, shifted), .3)
        traceless = PauliExpansion(1, {x: -1.}, 0., "two_channel", False)
        self.assertAlmostEqual(normalized_pauli_distance(traceless, traceless), 0.)

    def test_fields_thousand_stay_finite_without_materializing_exponentials(self):
        lattice = square_lattice(3)
        fields = np.full(lattice.n_edges, 1000.)
        expanded = pauli_coefficients(lattice, fields, -fields)
        self.assertTrue(np.isfinite(expanded.log_scale))
        self.assertTrue(all(np.isfinite(value) for value in expanded.coefficients.values()))
        self.assertGreater(expanded.log_scale, 1000)
        self.assertAlmostEqual(hamiltonian_error(lattice, fields, fields, -fields, -fields), 0., delta=1e-14)
        result = hamiltonian_error(lattice, fields, -fields, -fields, fields)
        self.assertTrue(np.isfinite(result))
        self.assertTrue(0 <= result <= 2)

    def test_invalid_fields_and_branch_reject_nonzero_x(self):
        lattice = _triangle()
        for fields in (1., [0., 1.], [[0., 0., 0.]], [0., np.nan, 0.],
                       [np.inf, 0., 0.], [0., -np.inf, 0.],
                       [True, False, True], [0j, 0j, 1j]):
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    pauli_coefficients(lattice, fields)
        for fields in ([0., 1.], [0., np.nan, 0.]):
            with self.subTest(x_fields=fields):
                with self.assertRaises(ValueError):
                    pauli_coefficients(lattice, [0., 0., 0.], fields)
        with self.assertRaises(ValueError):
            pauli_coefficients(lattice, [0., 0., 0.], [.1, 0., 0.], model="z_branch")
        with self.assertRaises(ValueError):
            pauli_coefficients(lattice, [0., 0., 0.], model="misspelled")

    def test_zero_and_nonfinite_coefficient_vectors_are_rejected(self):
        key = ("X", (0,))
        for invalid in ({}, {key: 0.}, {key: np.nan}, {key: np.inf}, {key: -np.inf},
                        {key: True}, {key: 1j}):
            with self.subTest(coefficients=invalid):
                with self.assertRaises(ValueError):
                    normalized_coefficient_distance(invalid, {key: 1.})
                with self.assertRaises(ValueError):
                    normalized_coefficient_distance({key: 1.}, invalid)
        # Every isolated star contributes -I + exp(0)I = 0 exactly.
        with self.assertRaises(ValueError):
            pauli_coefficients(Lattice(2, []), [])

    def test_expansion_copies_mapping_and_rejects_ambiguous_pauli_labels(self):
        source = {("X", (0,)): -1., IDENTITY: 2.}
        expanded = PauliExpansion(2, source, 0., "two_channel", True)
        source[("X", (0,))] = 99.
        self.assertEqual(expanded.coefficients[("X", (0,))], -1.)
        with self.assertRaises(TypeError):
            expanded.coefficients[("X", (0,))] = 99.
        invalid_keys = [
            ("Y", (0,)), ("X", ()), ("Z", (1, 0)),
            ("Z", (0, 0)), ("I", (0,)), ("X", (2,)), ("X", (-1,)),
        ]
        for key in invalid_keys:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    PauliExpansion(2, {key: 1.}, 0., "two_channel", True)
        for n_qubits, log_scale, model, identity in (
            (-1, 0., "two_channel", True),
            (True, 0., "two_channel", True),
            (2., 0., "two_channel", True),
            (2, np.inf, "two_channel", True),
            (2, np.nan, "two_channel", True),
            (2, 0., "typo", True),
            (2, 0., "two_channel", "false"),
        ):
            with self.subTest(n=n_qubits, scale=log_scale, model=model, identity=identity):
                with self.assertRaises(ValueError):
                    PauliExpansion(n_qubits, {("X", (0,)): 1.}, log_scale, model, identity)
        with self.assertRaises(ValueError):
            PauliExpansion(2, {IDENTITY: 1.}, 0., "two_channel", False)

    def test_expansion_comparison_rejects_incompatible_conventions(self):
        full = pauli_coefficients(_triangle(), [.1, .2, .3])
        different_n = pauli_coefficients(Lattice(2, [(0, 1)]), [.1])
        branch = pauli_coefficients(_triangle(), [.1, .2, .3], model="z_branch")
        traceless = pauli_coefficients(_triangle(), [.1, .2, .3], include_identity=False)
        for other in (different_n, branch, traceless):
            with self.subTest(other=other):
                with self.assertRaises(ValueError):
                    normalized_pauli_distance(full, other)


if __name__ == "__main__":
    unittest.main()
