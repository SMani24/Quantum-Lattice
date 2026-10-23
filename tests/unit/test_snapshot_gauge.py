"""Finite synthetic symmetry oracles; no stored research records are read."""

import hashlib
import unittest

import numpy as np

from quantum_lattice.data.snapshot_data import reconstruct_pseudospins
from quantum_lattice.data.snapshot_gauge import (
    GAUGE_SCHEMA_VERSION, gauge_mask_spec, gauge_masks_from_vertices,
    vertex_gauge_masks)
from quantum_lattice.lattice import square_lattice
from quantum_lattice.models.inverse_ising import (
    fit_inverse_ising, pseudolikelihood_objective)


def flat_edges(maps):
    return maps.transpose(0, 2, 3, 1).reshape(len(maps), -1).astype(np.int8)


def full_orbit(length=3):
    vertex_count = length * length
    labels = np.arange(1 << (vertex_count - 1), dtype=np.uint64)
    bits = ((labels[:, None] >> np.arange(vertex_count - 1, dtype=np.uint64)) & 1)
    vertices = np.ones((len(labels), vertex_count), dtype=np.int8)
    vertices[:, 1:] = bits.astype(np.int8) * 2 - 1
    vertices = vertices.reshape(-1, length, length)
    return vertices, flat_edges(gauge_masks_from_vertices(vertices, length))


class SnapshotGaugeTests(unittest.TestCase):
    def test_edge_channel_order_matches_explicit_topology_and_preserves_inputs(self):
        for length in (3, 4):
            with self.subTest(L=length):
                vertices = np.where(np.arange(2 * length * length).reshape(2, length, length)
                                    % 3 == 0, -1, 1).astype(np.int8)
                before = vertices.copy()
                actual = gauge_masks_from_vertices(vertices, length)
                endpoints = square_lattice(length).edge_vertex_indices
                values = vertices.reshape(2, -1)
                expected = values[:, endpoints[:, 0]] * values[:, endpoints[:, 1]]
                np.testing.assert_array_equal(flat_edges(actual), expected)
                np.testing.assert_array_equal(vertices, before)
                self.assertEqual(actual.shape, (2, 2, length, length))
                self.assertEqual(actual.dtype, np.float32)
                self.assertTrue(actual.flags.c_contiguous)
                np.testing.assert_array_equal(gauge_masks_from_vertices(vertices[0], length),
                                              actual[0])

    def test_masks_are_keyed_to_uid_epoch_seed_not_batch_order_or_global_rng(self):
        uids = ["Hamiltonian-A", "Hamiltonian-B", "H-α", "Hamiltonian-A"]
        before = np.random.get_state()
        actual = vertex_gauge_masks(uids, 4, seed=2026102301, epoch=0)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(actual,
            vertex_gauge_masks(uids, 4, seed=2026102301, epoch=0))
        order = [2, 0, 3, 1]
        np.testing.assert_array_equal(actual[order], vertex_gauge_masks(
            np.asarray(uids)[order], 4, seed=2026102301, epoch=0))
        np.testing.assert_array_equal(actual, np.concatenate([
            vertex_gauge_masks(uids[:1], 4, seed=2026102301, epoch=0),
            vertex_gauge_masks(uids[1:], 4, seed=2026102301, epoch=0)]))
        np.testing.assert_array_equal(actual[0], actual[3])
        self.assertFalse(np.array_equal(actual,
            vertex_gauge_masks(uids, 4, seed=2026102301, epoch=1)))
        self.assertFalse(np.array_equal(actual,
            vertex_gauge_masks(uids, 4, seed=2026102302, epoch=0)))
        self.assertEqual(actual.dtype, np.float32)
        reconstructed = reconstruct_pseudospins(flat_edges(actual), 4)
        np.testing.assert_array_equal(reconstructed[:, 0], np.ones(len(uids)))

    def test_all_l3_root_fixed_masks_are_distinct_and_form_a_sector_preserving_group(self):
        vertices, states = full_orbit()
        self.assertEqual(len(np.unique(states, axis=0)), 256)
        reconstructed = reconstruct_pseudospins(states, 3)
        np.testing.assert_array_equal(reconstructed, vertices.reshape(256, 9))
        masks = gauge_masks_from_vertices(vertices, 3)
        np.testing.assert_array_equal(masks, gauge_masks_from_vertices(-vertices, 3))
        np.testing.assert_array_equal(masks * masks, np.ones_like(masks))
        np.testing.assert_array_equal(masks[-1], np.ones((2, 3, 3)))
        np.testing.assert_array_equal(masks * masks[::-1],
            gauge_masks_from_vertices(vertices * vertices[::-1], 3))
        lattice = square_lattice(3)
        for face in lattice.faces:
            np.testing.assert_array_equal(np.prod(states[:, face.edge_ids], axis=1),
                                          np.ones(len(states)))
        np.testing.assert_array_equal(np.prod(masks[:, 0, 0, :], axis=1), np.ones(256))
        np.testing.assert_array_equal(np.prod(masks[:, 1, :, 0], axis=1), np.ones(256))
        eta = states[37]
        transformed = states * eta
        self.assertEqual(set(map(tuple, states)), set(map(tuple, transformed)))

    def test_arbitrary_edge_and_bad_winding_masks_are_rejected_by_sector_oracle(self):
        single_edge = np.ones((1, 18), dtype=np.int8)
        single_edge[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "sector"):
            reconstruct_pseudospins(single_edge, 3)
        winding = np.ones((1, 2, 3, 3), dtype=np.int8)
        winding[:, 0, :, -1] = -1
        states = flat_edges(winding)
        for face in square_lattice(3).faces:
            self.assertEqual(int(np.prod(states[0, face.edge_ids])), 1)
        self.assertEqual(int(np.prod(winding[0, 0, 0])), -1)
        with self.assertRaisesRegex(ValueError, "sector"):
            reconstruct_pseudospins(states, 3)

    def test_exact_probability_partition_magnitude_and_symmetric_prior_identity(self):
        _, states = full_orbit()
        fields = np.linspace(-1.4, 1.6, 18)
        eta = states[83]
        transformed_fields = fields * eta
        transformed_states = states * eta
        np.testing.assert_array_equal(np.abs(transformed_fields), np.abs(fields))
        energies = states @ fields
        transformed_energies = transformed_states @ transformed_fields
        np.testing.assert_allclose(transformed_energies, energies, atol=1e-14, rtol=0)
        partition = np.exp(energies).sum()
        other_partition = np.exp(states @ transformed_fields).sum()
        self.assertAlmostEqual(partition / other_partition, 1.0, places=14)
        np.testing.assert_allclose(np.exp(transformed_energies) / other_partition,
                                   np.exp(energies) / partition, atol=1e-15, rtol=1e-13)
        # The declared product-uniform field prior has constant density inside
        # the symmetric cube; sign masks preserve both support and volume.
        self.assertTrue(np.all(np.abs(fields) <= 1.7))
        self.assertTrue(np.all(np.abs(transformed_fields) <= 1.7))
        self.assertEqual(np.prod(eta.astype(np.float64)) ** 2, 1.0)

    def test_single_mask_applies_to_all_repeated_records_and_commutes_with_mean(self):
        _, states = full_orbit()
        records = states[[1, 1, 9, 34, 255]].reshape(1, 5, 3, 3, 2).transpose(0, 1, 4, 2, 3)
        mask = vertex_gauge_masks(["repeat-bag"], 3, seed=31, epoch=2)
        transformed = records * mask[:, None]
        np.testing.assert_array_equal(transformed[:, 0], transformed[:, 1])
        np.testing.assert_array_equal(transformed.mean(1, dtype=np.float32),
                                      records.mean(1, dtype=np.float32) * mask)
        reconstruct_pseudospins(flat_edges(transformed[0]), 3)

    def test_pseudolikelihood_value_gradient_hessian_and_fitted_coupling_covariance(self):
        _, states = full_orbit()
        rng = np.random.default_rng(6051)
        truth = rng.uniform(-1.2, 1.2, 18)
        probabilities = np.exp(states @ truth)
        probabilities /= probabilities.sum()
        fields = rng.uniform(-.8, .8, 18)
        eta = states[46]
        transformed = (states * eta).astype(np.int8)
        self.assertTrue(any(np.prod(np.sign(truth[list(face.edge_ids)])) < 0
                            for face in square_lattice(3).faces))
        value, gradient, hessian = pseudolikelihood_objective(
            states, fields, weights=probabilities, ridge=.01)
        other, other_gradient, other_hessian = pseudolikelihood_objective(
            transformed, fields * eta, weights=probabilities, ridge=.01)
        self.assertAlmostEqual(value, other, places=14)
        np.testing.assert_allclose(other_gradient, eta * gradient, atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(other_hessian, eta[:, None] * hessian * eta[None, :],
                                   atol=1e-13, rtol=1e-12)
        first = fit_inverse_ising(states, weights=probabilities, ridge=.01,
                                 tolerance=1e-10, max_iterations=40)
        second = fit_inverse_ising(transformed, weights=probabilities, ridge=.01,
                                  tolerance=1e-10, max_iterations=40)
        self.assertTrue(first["converged"])
        self.assertTrue(second["converged"])
        np.testing.assert_allclose(second["fields"], eta * np.asarray(first["fields"]),
                                   atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(second["magnitudes"], first["magnitudes"],
                                   atol=1e-9, rtol=1e-9)

    def test_invalid_keys_dimensions_and_vertex_signs_are_rejected(self):
        for uids in ([], "one", [1], ["a", 1], [["a"]], [""], [b"a"]):
            with self.subTest(uids=uids), self.assertRaises(ValueError):
                vertex_gauge_masks(uids, 3, seed=1, epoch=0)
        for key, value in (("L", 2), ("L", True), ("L", 3.0),
                           ("seed", -1), ("seed", True), ("seed", 1.5),
                           ("epoch", -1), ("epoch", True), ("epoch", .5)):
            options = {"L": 3, "seed": 1, "epoch": 0}
            options[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                vertex_gauge_masks(["a"], **options)
        for vertices in (np.ones((1, 9)), np.ones((2, 3, 3), dtype=bool),
                         np.zeros((1, 3, 3)), np.ones((0, 3, 3)),
                         np.ones((1, 3, 3), dtype=complex)):
            with self.subTest(shape=vertices.shape), self.assertRaises(ValueError):
                gauge_masks_from_vertices(vertices, 3)

    def test_schema_is_owned_and_stable_mask_fixture_binds_the_recipe(self):
        self.assertEqual(GAUGE_SCHEMA_VERSION, 1)
        first = gauge_mask_spec()
        first["hash_fields"].append("wrong")
        self.assertNotIn("wrong", gauge_mask_spec()["hash_fields"])
        self.assertFalse(gauge_mask_spec()["uses_true_fields"])
        masks = vertex_gauge_masks(["Hamiltonian-A", "H-α"], 3, seed=73, epoch=0)
        # Fixed stream fixture, filled from the documented version-1 recipe.
        self.assertEqual(hashlib.sha256(masks.tobytes()).hexdigest(),
                         "4482b79a39dfe45a864d4acc1a85d01d2fbca373ccee78c315fcfc5b7c239644")


if __name__ == "__main__":
    unittest.main()
