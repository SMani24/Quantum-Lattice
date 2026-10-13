"""Independent amplitude and transition checks for the solvable sampler."""

import copy
import json
import math
import unittest

import numpy as np

from quantum_lattice.lattice import Lattice, square_lattice
from quantum_lattice.observables.solvable import exact_expectations, local_estimators
from quantum_lattice.simulation.solvable import StarMetropolisChain, _compiled_kernel, _proposal_kernel


class SolvableObservableTests(unittest.TestCase):
    def setUp(self):
        self.lattice = square_lattice(3)
        self.fields = np.linspace(-0.31, 0.27, self.lattice.n_edges)

    def test_estimators_and_exact_means_match_wavefunction_permutations(self):
        exact = exact_expectations(self.lattice, self.fields)
        states, probabilities = exact["states"], exact["probabilities"]
        self.assertEqual(states.shape, (256, 18))
        self.assertEqual(len({tuple(state) for state in states}), 256)
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-15)
        # Construct wavefunction independently from the listed states. A flip
        # expectation is psi dot permuted-psi, not another estimator evaluation.
        amplitude = np.exp(0.5 * (states @ self.fields))
        amplitude /= np.linalg.norm(amplitude)
        lookup = {tuple(state): index for index, state in enumerate(states)}
        features = local_estimators(self.lattice, self.fields, states)
        for name, supports in (
            ("star", [set(vertex.edge_ids) for vertex in self.lattice.vertices]),
            ("pair", [set(edge.endpoints[0].edge_ids) ^ set(edge.endpoints[1].edge_ids)
                      for edge in self.lattice.edges]),
        ):
            for feature, support in enumerate(supports):
                flipped = states.copy()
                flipped[:, list(support)] *= -1
                permutation = [lookup[tuple(state)] for state in flipped]
                oracle_ratio = amplitude[permutation] / amplitude
                np.testing.assert_allclose(features[name][:, feature], oracle_ratio, rtol=2e-15)
                self.assertAlmostEqual(exact[name][feature], amplitude @ amplitude[permutation], places=14)
        np.testing.assert_allclose(exact["z"], (amplitude * amplitude) @ states, atol=3e-16)
        self.assertTrue(np.any(features["star"] > 1.0))  # Never clip individual samples.

    def test_zero_fields_and_arbitrary_leading_batch_shape(self):
        states = exact_expectations(self.lattice, np.zeros(18))["states"][:6].reshape(2, 3, 18)
        values = local_estimators(self.lattice, np.zeros(18), states)
        np.testing.assert_array_equal(values["star"], np.ones((2, 3, 9)))
        np.testing.assert_array_equal(values["pair"], np.ones((2, 3, 18)))
        np.testing.assert_array_equal(values["z"], states)
        exact = exact_expectations(self.lattice, np.zeros(18))
        np.testing.assert_array_equal(exact["star"], np.ones(9))
        np.testing.assert_array_equal(exact["pair"], np.ones(18))
        np.testing.assert_array_equal(exact["z"], np.zeros(18))

    def test_parallel_shared_edges_cancel_in_two_star_operator(self):
        lattice = Lattice(3, [(0, 1), (0, 1), (1, 2), (2, 0)])
        fields = np.array([0.11, -0.26, 0.4, 0.2])
        state = np.array([-1, -1, 1, -1], dtype=np.int8)
        actual = local_estimators(lattice, fields, state)
        for edge in lattice.edges:
            support = set(edge.endpoints[0].edge_ids) ^ set(edge.endpoints[1].edge_ids)
            expected = math.exp(-sum(fields[index] * state[index] for index in support))
            self.assertAlmostEqual(actual["pair"][edge.id], expected)
        np.testing.assert_array_equal(actual["pair"][:2], np.repeat(actual["pair"][0], 2))

    def test_disconnected_and_isolated_vertices_have_unique_orbit_enumeration(self):
        lattice = Lattice(6, [(0, 1), (1, 2), (3, 4)])
        exact = exact_expectations(lattice, [0.1, -0.2, 0.3])
        self.assertEqual(exact["states"].shape, (8, 3))
        self.assertEqual(len({tuple(state) for state in exact["states"]}), 8)
        self.assertEqual(exact["star"][5], 1.0)

    def test_bad_shapes_and_exact_size_guard(self):
        for fields, spins in (([0.0], np.ones(18)), (np.zeros(18), np.zeros(18)),
                              (np.ones(18, dtype=complex), np.ones(18)),
                              (np.zeros(18), np.ones(17)), (np.full(18, np.nan), np.ones(18))):
            with self.assertRaises(ValueError):
                local_estimators(self.lattice, fields, spins)
        with self.assertRaises(ValueError):
            exact_expectations(square_lattice(5), np.zeros(50))


class SolvableSamplerTests(unittest.TestCase):
    def setUp(self):
        self.lattice = square_lattice(3)
        self.fields = np.linspace(-0.25, 0.3, 18)

    def test_kernel_uses_probability_ratio_and_immediate_updates(self):
        fields = np.zeros(18)
        fields[list(self.lattice.vertices[0].edge_ids)] = 0.1
        # First proposal has h=.4. Second repeats the same vertex and must see
        # the flipped state (h=-.4), so it is automatically accepted.
        spins = np.ones(18, dtype=np.int8)
        vertices = np.array([[0, 0]], dtype=np.int64)
        uniforms = np.array([[[0.9, math.exp(-0.8) - 1e-6], [0.9, 0.99]]])
        accepted, lazy = _proposal_kernel(spins, fields, self.lattice.vertex_edge_indices,
                                         self.lattice.vertex_degrees, vertices, uniforms, 0.05)
        self.assertEqual((accepted, lazy), (2, 0))
        np.testing.assert_array_equal(spins, np.ones(18))
        # Probability ratio exp(-.8), not amplitude ratio exp(-.4): reject .55.
        accepted, _ = _proposal_kernel(spins, fields, self.lattice.vertex_edge_indices,
                                      self.lattice.vertex_degrees, np.array([[0]]),
                                      np.array([[[0.9, 0.55]]]), 0.05)
        self.assertEqual(accepted, 0)

    def test_full_transition_matrix_detailed_balance_and_stationarity(self):
        exact = exact_expectations(self.lattice, self.fields)
        states, probability = exact["states"], exact["probabilities"]
        lookup = {tuple(state): index for index, state in enumerate(states)}
        transition = np.zeros((256, 256))
        lazy = 0.05
        for row, state in enumerate(states):
            transition[row, row] += lazy
            for vertex in self.lattice.vertices:
                next_state = state.copy()
                next_state[list(vertex.edge_ids)] *= -1
                column = lookup[tuple(next_state)]
                ratio = math.exp(float(self.fields @ (next_state - state)))
                acceptance = min(1.0, ratio)
                transition[row, column] += (1.0 - lazy) / 9 * acceptance
                transition[row, row] += (1.0 - lazy) / 9 * (1.0 - acceptance)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=3e-16)
        flux = probability[:, None] * transition
        np.testing.assert_allclose(flux, flux.T, atol=2e-18)
        np.testing.assert_allclose(probability @ transition, probability, atol=4e-18)

    def test_sweeps_lazy_counts_and_topological_invariants(self):
        lattice = square_lattice(4)
        chain = StarMetropolisChain(lattice, np.zeros(32), 481, backend="python")
        for _ in range(20):
            chain.advance(3)
            self.assertTrue(np.all(np.prod(chain.spins[lattice.face_edge_indices], axis=1) == 1))
            horizontal_loop = [2 * x for x in range(4)]
            vertical_loop = [2 * (4 * y) + 1 for y in range(4)]
            self.assertEqual(np.prod(chain.spins[horizontal_loop]), 1)
            self.assertEqual(np.prod(chain.spins[vertical_loop]), 1)
        self.assertEqual(chain.attempted, 60 * 16)
        self.assertEqual(chain.accepted + chain.lazy, chain.attempted)
        self.assertGreater(chain.lazy, 0)

    def test_short_chain_matches_exact_means_with_block_uncertainty(self):
        # A small validation chain only: no full dataset-generation run.
        exact = exact_expectations(self.lattice, self.fields)
        chain = StarMetropolisChain(self.lattice, self.fields, 38014, backend="auto")
        chain.advance(200)
        states = np.empty((2000, 18), dtype=np.int8)
        for sample in range(len(states)):
            chain.advance(5)
            states[sample] = chain.spins
        features = local_estimators(self.lattice, self.fields, states)
        for name, values in features.items():
            blocks = values.reshape(40, 50, -1).mean(axis=1)
            standard_error = blocks.std(axis=0, ddof=1) / math.sqrt(40)
            errors = np.abs(values.mean(axis=0) - exact[name])
            self.assertTrue(np.all(errors <= 7.0 * standard_error + 1e-12), name)

    def test_chunk_invariance_and_exact_json_resume(self):
        direct = StarMetropolisChain(self.lattice, self.fields, 142, backend="python")
        direct.advance(37)
        chunked = StarMetropolisChain(self.lattice, self.fields, 142, backend="python")
        for sweeps in (0, 3, 9, 1, 24):
            chunked.advance(sweeps)
        self.assertEqual(direct.snapshot(), chunked.snapshot())
        checkpoint = json.loads(json.dumps(chunked.snapshot()))
        resumed = StarMetropolisChain.from_snapshot(self.lattice, self.fields, checkpoint, backend="python")
        direct.advance(17)
        resumed.advance(17)
        self.assertEqual(direct.snapshot(), resumed.snapshot())
        checkpoint["spins"][0] *= -1
        with self.assertRaisesRegex(ValueError, "reference star orbit"):
            StarMetropolisChain.from_snapshot(self.lattice, self.fields, checkpoint, backend="python")

    @unittest.skipIf(_compiled_kernel is None, "Optional Numba backend is unavailable")
    def test_compiled_backend_matches_python_and_can_resume_its_checkpoint(self):
        python = StarMetropolisChain(self.lattice, self.fields, 717, backend="python")
        compiled = StarMetropolisChain(self.lattice, self.fields, 717, backend="numba")
        python.advance(151)
        compiled.advance(151)
        self.assertEqual(python.snapshot(), compiled.snapshot())
        compiled = StarMetropolisChain.from_snapshot(self.lattice, self.fields, python.snapshot(), backend="numba")
        compiled.advance(97)
        python.advance(97)
        self.assertEqual(python.snapshot(), compiled.snapshot())

    def test_snapshot_validation_and_input_guards(self):
        chain = StarMetropolisChain(self.lattice, self.fields, 31, backend="python")
        snapshot = chain.snapshot()
        for key, bad in (("accepted", 1), ("attempted", 1), ("rng_state", {}), ("version", 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                altered = copy.deepcopy(snapshot)
                altered[key] = bad
                StarMetropolisChain.from_snapshot(self.lattice, self.fields, altered, backend="python")
        with self.assertRaises(ValueError):
            StarMetropolisChain.from_snapshot(self.lattice, self.fields + 0.1, snapshot)
        with self.assertRaises(ValueError):
            StarMetropolisChain.from_snapshot(self.lattice, self.fields, snapshot, lazy_probability=0.0)
        for sweeps in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                chain.advance(sweeps)


if __name__ == "__main__":
    unittest.main()
