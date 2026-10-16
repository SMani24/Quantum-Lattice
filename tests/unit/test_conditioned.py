"""Independent-oracle checks for the optional conditioned-estimator study."""

import json
import unittest

import numpy as np

from quantum_lattice.lattice import Lattice, square_lattice
from quantum_lattice.observables.conditioned import (
    exact_reference, paired_estimators, split_rhat_from_summaries, summarize_records,
)


class ConditionedEstimatorTests(unittest.TestCase):
    def test_sech_stays_finite_without_cosh_overflow(self):
        lattice = Lattice(2, [(0, 1)])
        values = paired_estimators(lattice, [720.], np.ones((1, 1), dtype=np.int8))
        self.assertTrue(np.isfinite(values["star"]["g"]).all())
        self.assertTrue((values["star"]["g"] > 0).all())
        np.testing.assert_allclose(values["star"]["g"], 2 * np.exp(-720.), rtol=1e-12, atol=0)

    def test_direct_symmetric_difference_includes_all_parallel_edges(self):
        lattice = Lattice(3, [(0, 1), (0, 1), (1, 2), (2, 0)])
        fields = np.array([.4, -.8, 1.1, -.3])
        states = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=np.int8)
        values = paired_estimators(lattice, fields, states)
        for edge in lattice.edges:
            u, v = edge.vertex_ids
            support = sorted(set(lattice.vertices[u].edge_ids) ^ set(lattice.vertices[v].edge_ids))
            h = (states[:, support] * fields[support]).sum(axis=1)
            np.testing.assert_allclose(values["pair"]["f"][:, edge.id], np.exp(-h), atol=1e-14)
            np.testing.assert_allclose(values["pair"]["g"][:, edge.id], 1 / np.cosh(h), atol=1e-14)
        np.testing.assert_array_equal(values["pair"]["g"][:, 0], values["pair"]["g"][:, 1])

    def test_exact_oracle_matches_dense_flip_matrix_and_variance_identity(self):
        lattice = Lattice(4, [(0, 1), (1, 2), (2, 0), (2, 3)])
        fields = np.array([.4, -.2, .7, -1.1])
        reference = exact_reference(lattice, fields)
        z, p = reference["states"], reference["probabilities"]
        amplitude = np.sqrt(p)
        lookup = {tuple(row): i for i, row in enumerate(z)}
        values = paired_estimators(lattice, fields, z)
        for channel, supports in (("star", [set(v.edge_ids) for v in lattice.vertices]),
                                  ("pair", [set(lattice.vertices[e.vertex_ids[0]].edge_ids) ^ set(lattice.vertices[e.vertex_ids[1]].edge_ids) for e in lattice.edges])):
            for j, support in enumerate(supports):
                matrix = np.zeros((len(z), len(z)))
                for i, row in enumerate(z):
                    flipped = row.copy()
                    flipped[list(support)] *= -1
                    matrix[lookup[tuple(flipped)], i] = 1
                mu = amplitude @ matrix @ amplitude
                self.assertAlmostEqual(mu, reference["truth"][channel][j], places=14)
                f, g = values[channel]["f"][:, j], values[channel]["g"][:, j]
                self.assertAlmostEqual(p @ f, mu, places=14)
                self.assertAlmostEqual(p @ g, mu, places=14)
                self.assertAlmostEqual(p @ (f * f), 1, places=14)
                vf, vg = p @ (f - mu) ** 2, p @ (g - mu) ** 2
                self.assertAlmostEqual(vf - vg, p @ (1 - g * g), places=14)
                self.assertLessEqual(vg, mu * (1 - mu) + 1e-14)
        self.assertLess(reference["max_identity_error"], 1e-13)

    def test_disconnected_and_isolated_orbit(self):
        lattice = Lattice(5, [(0, 1), (2, 3), (2, 3)])
        reference = exact_reference(lattice, [.2, -.3, .8])
        self.assertEqual(reference["states"].shape, (4, 3))
        self.assertEqual(len({tuple(row) for row in reference["states"]}), 4)
        self.assertAlmostEqual(reference["truth"]["star"][4], 1)
        np.testing.assert_allclose(reference["truth"]["pair"], 1)
        self.assertLess(reference["max_identity_error"], 1e-13)

    def test_cached_orbit_states_are_shared_readonly(self):
        lattice = square_lattice(3)
        first = exact_reference(lattice, np.zeros(lattice.n_edges))
        second = exact_reference(lattice, np.full(lattice.n_edges, .1))
        self.assertIs(first["states"], second["states"])
        self.assertFalse(first["states"].flags.writeable)
        with self.assertRaises(ValueError):
            first["states"][0, 0] = -1

    def test_zero_fields_and_short_records_have_explicit_missing_uncertainty(self):
        lattice = square_lattice(3)
        reference = exact_reference(lattice, np.zeros(lattice.n_edges))
        for channel in ("star", "pair"):
            np.testing.assert_allclose(reference["truth"][channel], 1)
            np.testing.assert_allclose(reference["variance"][channel]["f"], 0, atol=1e-28)
            np.testing.assert_allclose(reference["variance"][channel]["g"], 0, atol=1e-28)
        summary, arrays = summarize_records(lattice, np.zeros(lattice.n_edges), reference["states"][:1], truth=reference["truth"])
        json.dumps(summary, allow_nan=False)
        self.assertIsNone(summary["channels"]["star"]["g"]["mean_block_se_45"])
        self.assertIsNone(summary["channels"]["star"]["g"]["coverage_45"])
        self.assertTrue(np.isnan(arrays["star_g_half_mean"]).all())

    def test_summary_means_errors_blocks_and_shared_probes(self):
        lattice = square_lattice(3)
        fields = np.linspace(-.6, .8, lattice.n_edges)
        reference = exact_reference(lattice, fields)
        draw = np.random.default_rng(11).choice(len(reference["states"]), size=180, p=reference["probabilities"])
        states = reference["states"][draw]
        summary, arrays = summarize_records(lattice, fields, states, truth=reference["truth"], block_sizes=(9, 18, 30), probe_count=3, seed=19)
        json.dumps(summary, allow_nan=False)
        x = paired_estimators(lattice, fields, states)["star"]["g"]
        np.testing.assert_allclose(arrays["star_g_mean"], x.mean(axis=0))
        np.testing.assert_allclose(arrays["star_g_block_se_9"], x.reshape(20, 9, -1).mean(axis=1).std(axis=0, ddof=1) / np.sqrt(20))
        self.assertAlmostEqual(summary["channels"]["star"]["g"]["mse"], np.mean((x.mean(axis=0) - reference["truth"]["star"]) ** 2))
        old = summary["channels"]["star"]["f"]["autocorrelation"]
        new = summary["channels"]["star"]["g"]["autocorrelation"]
        self.assertEqual([v["coordinate"] for v in old], [v["coordinate"] for v in new])

    def test_split_rhat_matches_raw_and_handles_degeneracies(self):
        x = np.random.default_rng(37).normal(size=(4, 40, 3))
        halves = np.concatenate((x[:, :20], x[:, -20:]), axis=0)
        means = np.stack((x[:, :20].mean(axis=1), x[:, -20:].mean(axis=1)), axis=1)
        variances = np.stack((x[:, :20].var(axis=1, ddof=1), x[:, -20:].var(axis=1, ddof=1)), axis=1)
        W = halves.var(axis=1, ddof=1).mean(axis=0)
        B = 20 * halves.mean(axis=1).var(axis=0, ddof=1)
        np.testing.assert_allclose(split_rhat_from_summaries(means, variances, 20), np.sqrt((19 / 20 * W + B / 20) / W))
        zero = np.zeros((4, 2, 2))
        means = zero.copy()
        means[2:, :, 1] = 1
        result = split_rhat_from_summaries(means, zero, 20)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isinf(result[1]))

    def test_invalid_inputs_and_enumeration_limit(self):
        lattice = square_lattice(3)
        with self.assertRaises(ValueError):
            exact_reference(square_lattice(8), np.zeros(128))
        with self.assertRaises(ValueError):
            paired_estimators(lattice, np.zeros(18), np.zeros((2, 18)))
        with self.assertRaises(ValueError):
            summarize_records(lattice, np.zeros(18), np.ones((4, 18)), block_sizes=(0,))
        with self.assertRaises(ValueError):
            summarize_records(lattice, np.zeros(18), np.ones((0, 18)))


if __name__ == "__main__":
    unittest.main()
