"""Coordinate oracles for reversible, grouped periodic observable layouts."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.mlp_data import EdgeRows, LearningData
from quantum_lattice.data.observable_maps import (
    INPUT_CHANNELS, TARGET_CHANNELS, ObservableMaps, edge_rows_to_maps,
    flatten_edge_maps, map_learning_data, maps_to_edge_rows,
)


def fixture_rows(L, prefix="train", count=2):
    """Independent scalar coordinate fixture; no production geometry helpers."""
    features, targets, names, uids, edges = [], [], [], [], []
    for n in range(count):
        def star(y, x):
            return 1000 * n + 100 * (y % L) + x % L + .25
        for y in range(L):
            for x in range(L):
                for orientation in (0, 1):
                    neighbor = star(y, x+1) if orientation == 0 else star(y+1, x)
                    pair = 10000*n + 1000*orientation + 100*y + x + .5
                    features.append((star(y, x), neighbor, pair))
                    targets.append((20000*n + 1000*orientation + 100*y + x + .75,))
                    names.append(f"{prefix}-{n}")
                    uids.append(f"uid-{prefix}-{n}")
                    edges.append(2*(y*L+x)+orientation)
    return EdgeRows(np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32),
                    np.array(names), np.array(uids), np.array(edges, dtype=np.int64))


def select_rows(rows, selection):
    return EdgeRows(*(getattr(rows, name)[selection].copy() for name in
                      ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids")))


class ObservableMapTests(unittest.TestCase):
    def assertRowsEqual(self, actual, expected):
        for name in ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
            self.assertEqual(getattr(actual, name).dtype, getattr(expected, name).dtype)

    def test_coordinate_channels_targets_and_full_round_trip(self):
        for L in (3, 4, 8):
            with self.subTest(L=L):
                original = fixture_rows(L)
                actual = edge_rows_to_maps(original, L)
                self.assertEqual(actual.x.shape, (2, 3, L, L))
                self.assertEqual(actual.y.shape, (2, 2, L, L))
                self.assertEqual(len(actual), 2)
                for n in range(2):
                    for y in range(L):
                        for x in range(L):
                            self.assertEqual(actual.x[n, 0, y, x], 1000*n+100*y+x+.25)
                            for o in (0, 1):
                                self.assertEqual(actual.x[n, 1+o, y, x], 10000*n+1000*o+100*y+x+.5)
                                self.assertEqual(actual.y[n, o, y, x], 20000*n+1000*o+100*y+x+.75)
                self.assertRowsEqual(maps_to_edge_rows(actual), original)
                self.assertEqual(len(INPUT_CHANNELS), 3)
                self.assertEqual(len(TARGET_CHANNELS), 2)
                self.assertGreater(actual.x.max(), 1)  # f values remain unclipped.

    def test_periodic_wrap_uses_correct_second_endpoint(self):
        L = 4
        maps = edge_rows_to_maps(fixture_rows(L, count=1), L)
        rows = maps_to_edge_rows(maps)
        horizontal = 2 * (2*L + L-1)
        vertical = 2 * ((L-1)*L + 2) + 1
        self.assertEqual(rows.x[horizontal, 1], maps.x[0, 0, 2, 0])
        self.assertEqual(rows.x[vertical, 1], maps.x[0, 0, 0, 2])
        self.assertNotEqual(rows.x[horizontal, 1], rows.x[horizontal, 0])

    def test_interleaved_shuffled_rows_restore_original_order(self):
        original = fixture_rows(3, count=3)
        shuffled = select_rows(original, np.random.default_rng(74).permutation(len(original)))
        maps = edge_rows_to_maps(shuffled, 3)
        self.assertRowsEqual(maps_to_edge_rows(maps), shuffled)
        first_names = list(dict.fromkeys(shuffled.hamiltonian_ids))
        self.assertEqual(list(maps.hamiltonian_ids), first_names)
        for n, name in enumerate(first_names):
            for edge, source_row in enumerate(maps.source_row_indices[n]):
                self.assertEqual(shuffled.hamiltonian_ids[source_row], name)
                self.assertEqual(shuffled.edge_ids[source_row], edge)
        np.testing.assert_array_equal(flatten_edge_maps(maps, maps.y), shuffled.y)

    def test_prediction_flatten_preserves_orientation_order_and_negative_outputs(self):
        rows = fixture_rows(4)
        rows = select_rows(rows, np.random.default_rng(23).permutation(len(rows)))
        maps = edge_rows_to_maps(rows, 4)
        predictions = -maps.y
        actual = flatten_edge_maps(maps, predictions)
        np.testing.assert_array_equal(actual, -rows.y)
        self.assertFalse(actual.flags.writeable)
        self.assertFalse(np.shares_memory(actual, predictions))

    def test_translation_preserves_target_and_feature_alignment(self):
        L = 4
        maps = edge_rows_to_maps(fixture_rows(L, count=1), L)
        dy, dx = 1, -2
        shifted = replace(maps, x=np.roll(maps.x, (dy, dx), axis=(-2, -1)),
                          y=np.roll(maps.y, (dy, dx), axis=(-2, -1)))
        rows = maps_to_edge_rows(shifted)
        for y in range(L):
            for x in range(L):
                for o in (0, 1):
                    edge = 2*(y*L+x)+o
                    old_y, old_x = (y-dy) % L, (x-dx) % L
                    self.assertEqual(rows.x[edge, 0], maps.x[0, 0, old_y, old_x])
                    self.assertEqual(rows.x[edge, 2], maps.x[0, o+1, old_y, old_x])
                    self.assertEqual(rows.y[edge, 0], maps.y[0, o, old_y, old_x])
        self.assertRowsEqual(maps_to_edge_rows(edge_rows_to_maps(rows, L)), rows)

    def test_arrays_own_readonly_copies_and_do_not_mutate_source(self):
        rows = fixture_rows(3)
        expected = select_rows(rows, np.arange(len(rows)))
        maps = edge_rows_to_maps(rows, 3)
        for name in ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "source_row_indices"):
            self.assertFalse(getattr(maps, name).flags.writeable)
        self.assertFalse(np.shares_memory(maps.x, rows.x))
        self.assertFalse(np.shares_memory(maps.y, rows.y))
        self.assertRowsEqual(rows, expected)
        rows.x[:] = -999
        rows.y[:] = 999
        self.assertRowsEqual(maps_to_edge_rows(maps), expected)
        restored = maps_to_edge_rows(maps)
        for name in ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
            self.assertFalse(getattr(restored, name).flags.writeable)

    def test_independent_chain_appearances_are_rejected_not_averaged(self):
        for edge, column in ((1, 0), (4, 1), (16, 1)):
            with self.subTest(edge=edge, column=column):
                rows = fixture_rows(3)
                rows.x[edge, column] = np.nextafter(rows.x[edge, column], np.float32(np.inf))
                with self.assertRaisesRegex(ValueError, "inconsistent duplicate star.*cannot average"):
                    edge_rows_to_maps(rows, 3)

    def test_missing_repeated_and_noncanonical_edges_rejected(self):
        rows = fixture_rows(3)
        with self.assertRaisesRegex(ValueError, "complete nonempty"):
            edge_rows_to_maps(select_rows(rows, np.arange(len(rows)-1)), 3)
        for bad, message in ((0, "duplicate edge"), (-1, "canonical integer"), (18, "canonical integer")):
            changed = select_rows(rows, np.arange(len(rows)))
            changed.edge_ids[1] = bad
            with self.assertRaisesRegex(ValueError, message):
                edge_rows_to_maps(changed, 3)
        with self.assertRaisesRegex(ValueError, "canonical integer"):
            edge_rows_to_maps(replace(rows, edge_ids=rows.edge_ids.astype(float)), 3)

    def test_inconsistent_and_repeated_hamiltonian_uids_rejected(self):
        rows = fixture_rows(3)
        rows.hamiltonian_uids[1] = "changed"
        with self.assertRaisesRegex(ValueError, "inconsistent UIDs"):
            edge_rows_to_maps(rows, 3)
        rows = fixture_rows(3)
        rows.hamiltonian_uids[18:] = rows.hamiltonian_uids[0]
        with self.assertRaisesRegex(ValueError, "multiple ids"):
            edge_rows_to_maps(rows, 3)

    def test_string_identity_validation_and_unsigned_edge_indices(self):
        rows = fixture_rows(3)
        for dtype in (str, bytes):
            changed = rows.hamiltonian_ids.astype(dtype)
            changed[0] = ""
            with self.assertRaisesRegex(ValueError, "nonempty string"):
                edge_rows_to_maps(replace(rows, hamiltonian_ids=changed), 3)
        unsigned = replace(rows, edge_ids=rows.edge_ids.astype(np.uint64))
        restored = maps_to_edge_rows(edge_rows_to_maps(unsigned, 3))
        np.testing.assert_array_equal(restored.edge_ids, rows.edge_ids)
        np.testing.assert_array_equal(restored.x, rows.x)

    def test_malformed_numeric_arrays_and_labels_rejected(self):
        rows = fixture_rows(3)
        for field, value in (("x", rows.x[:, :2]), ("y", rows.y[:, 0]),
                             ("x", rows.x.astype(complex))):
            with self.subTest(field=field, shape=value.shape):
                with self.assertRaisesRegex(ValueError, "finite real numeric"):
                    edge_rows_to_maps(replace(rows, **{field: value}), 3)
        for value in (np.nan, np.inf):
            changed = fixture_rows(3)
            changed.x[0, 2] = value
            with self.assertRaisesRegex(ValueError, "finite real numeric"):
                edge_rows_to_maps(changed, 3)
        rows.y[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            edge_rows_to_maps(rows, 3)

    def test_invalid_side_lengths_and_empty_rows_rejected(self):
        for length in (2, True, 3.0):
            with self.assertRaises(ValueError):
                edge_rows_to_maps(fixture_rows(3), length)
        with self.assertRaisesRegex(ValueError, "complete nonempty"):
            edge_rows_to_maps(select_rows(fixture_rows(3), []), 3)

    def test_direct_map_constructor_checks_order_identities_and_shapes(self):
        maps = edge_rows_to_maps(fixture_rows(3), 3)
        bad_order = maps.source_row_indices.copy()
        bad_order[0, 1] = bad_order[0, 0]
        for change in ({"source_row_indices": bad_order},
                       {"source_row_indices": maps.source_row_indices.astype(float)},
                       {"hamiltonian_ids": np.repeat("same", len(maps))},
                       {"x": maps.x[:, :2]}):
            with self.subTest(keys=tuple(change)):
                with self.assertRaises(ValueError):
                    replace(maps, **change)
        with self.assertRaisesRegex(ValueError, "finite real numeric"):
            flatten_edge_maps(maps, maps.y[:, 0])

    def learning_data(self, **changes):
        data = LearningData(fixture_rows(3, "train"), fixture_rows(3, "validation", 1),
                            None, 3, 0, "all_edges",
                            {"training_eligible": False, "estimator_view": "exact", "nested": {"a": 1}},
                            Path("never-opened-fixture"))
        return replace(data, **changes)

    def test_grouped_dataset_preserves_eligibility_provenance_and_no_disk_access(self):
        data = self.learning_data()
        with patch("numpy.load", side_effect=AssertionError("no file reads")), \
                patch.object(Path, "open", side_effect=AssertionError("no file reads")):
            mapped = map_learning_data(data)
        self.assertIsNone(mapped.test)
        self.assertEqual(mapped.L, data.L)
        self.assertEqual(mapped.dataset_dir, data.dataset_dir)
        self.assertFalse(mapped.provenance["training_eligible"])
        self.assertEqual(mapped.provenance["estimator_view"], "exact")
        self.assertNotIn("observable_map", data.provenance)
        mapped.provenance["nested"]["a"] = 2
        self.assertEqual(data.provenance["nested"]["a"], 1)
        self.assertRowsEqual(maps_to_edge_rows(mapped.train), data.train)
        self.assertRowsEqual(maps_to_edge_rows(mapped.validation), data.validation)

    def test_loaded_test_and_paper_size_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "no loaded test"):
            map_learning_data(self.learning_data(test=fixture_rows(3, "test")))
        with self.assertRaisesRegex(ValueError, "all_edges"):
            map_learning_data(self.learning_data(mode="paper_size"))

    def test_no_hamiltonian_identity_or_uid_leakage_between_splits(self):
        data = self.learning_data()
        for attribute in ("hamiltonian_ids", "hamiltonian_uids"):
            changed = select_rows(data.validation, np.arange(len(data.validation)))
            getattr(changed, attribute)[:] = getattr(data.train, attribute)[0]
            with self.assertRaisesRegex(ValueError, f"leakage through {attribute}"):
                map_learning_data(replace(data, validation=changed))


if __name__ == "__main__":
    unittest.main()
