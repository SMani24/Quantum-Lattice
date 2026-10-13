"""Lossless encoding checks, including corrupt packed descriptors and padding."""

import unittest

import numpy as np

from quantum_lattice.data.state_storage import decode_states, encode_states


class StateStorageTests(unittest.TestCase):
    def test_legacy_layout_and_random_roundtrips(self):
        rng = np.random.default_rng(12)
        for edges in (1, 8, 18, 32, 1152):
            for records in (0, 1, 17):
                states = (1 - 2 * rng.integers(2, size=(records, edges))).astype(np.int8)
                with self.subTest(edges=edges, records=records):
                    legacy = encode_states(states)
                    self.assertEqual(set(legacy), {"states"})
                    np.testing.assert_array_equal(decode_states(legacy), states)
                    packed = encode_states(states, "packed_bits")
                    self.assertEqual(packed["states_packed"].shape, (records, (edges + 7) // 8))
                    np.testing.assert_array_equal(decode_states(packed), states)
                    self.assertEqual(decode_states(packed).dtype, np.int8)

    def test_little_endian_bit_convention(self):
        states = np.array([[-1, 1, -1, 1, 1, 1, 1, -1, -1]], dtype=np.int8)
        encoded = encode_states(states, "packed_bits")
        np.testing.assert_array_equal(encoded["states_packed"], [[133, 1]])

    def test_invalid_spins_or_encoding_are_rejected(self):
        for states in (np.ones((3, 4)), np.zeros((3, 4), dtype=np.int8),
                       np.ones(4, dtype=np.int8), np.ones((2, 0), dtype=np.int8)):
            with self.subTest(shape=states.shape, dtype=states.dtype):
                with self.assertRaises(ValueError):
                    encode_states(states)
                with self.assertRaises(ValueError):
                    decode_states({"states": states})
        with self.assertRaises(ValueError):
            encode_states(np.ones((1, 4), dtype=np.int8), "zip")

    def test_corrupt_packed_data_cannot_be_silently_decoded(self):
        good = encode_states(np.ones((2, 18), dtype=np.int8), "packed_bits")
        bad_padding = good["states_packed"].copy()
        bad_padding[0, -1] = 4
        mutations = [
            {"state_n_edges": np.array(0)},
            {"state_n_edges": np.array(18.0)},
            {"state_n_edges": np.array([18])},
            {"state_encoding": np.array("unknown")},
            {"states_packed": good["states_packed"].astype(np.int8)},
            {"states_packed": good["states_packed"][:, :2]},
            {"states_packed": bad_padding},
            {"states": np.ones((2, 18), dtype=np.int8)},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    decode_states({**good, **mutation})
        with self.assertRaises(ValueError):
            decode_states({"states_packed": good["states_packed"]})


if __name__ == "__main__":
    unittest.main()
