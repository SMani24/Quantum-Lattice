"""Raw-record identity, sector and resumable-cache controls on synthetic files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quantum_lattice.data.snapshot_data import (
    load_snapshot_cache, prepare_snapshot_cache, reconstruct_pseudospins,
)
from quantum_lattice.data.state_storage import encode_states
from quantum_lattice.lattice import square_lattice


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    length = 3
    lattice = square_lattice(length)
    bank = root / "codex material/bank"
    directory = bank / "instances"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(911)
    instances, origins = [], {}
    selected = {"train": [], "validation": []}
    for index, split in enumerate(("train", "train", "validation", "validation")):
        name = f"h{index}"
        instance = {"id": name, "split": split, "hamiltonian_uid": f"uid-{index}", "field_seed": [index]}
        tau = rng.choice(np.array([-1, 1], dtype=np.int8), (8, length * length))
        # One all-repeat dataset confirms records, rather than unique states, are sampled.
        if index == 0:
            tau[:] = 1
        states = tau[:, lattice.edge_vertex_indices[:, 0]] * tau[:, lattice.edge_vertex_indices[:, 1]]
        fields = np.linspace(-0.7, 0.8, lattice.n_edges)
        metadata = {"instance": instance, "retained_records": 8, "chain": {"fields": fields.tolist()}}
        path = directory / f"{name}.npz"
        np.savez(path, **encode_states(states, "packed_bits"), b_z=fields,
                 edge_vertex_indices=lattice.edge_vertex_indices, metadata_json=np.asarray(json.dumps(metadata)))
        path.with_suffix(".json").write_text(json.dumps({"id": name, "split": split, "records": 8}))
        instances.append(instance)
        selected[split].append(name)
        origins[name] = {"instance": instance, "input_sha256": sha(path)}
    # A final-test declaration exists but intentionally has no payload file.
    instances.append({"id": "test-never-open", "split": "test", "hamiltonian_uid": "test-uid"})
    (bank / "manifest.json").write_text(json.dumps({"instances": instances}))
    feature = root / "codex material/features/L3/manifest.json"
    feature.parent.mkdir(parents=True)
    feature.write_text(json.dumps({"L": length, "original_bank": str(bank.relative_to(root)),
                                   "selected_ids": selected, "origins": origins}))
    config = {"selection_seed": 81, "train_hamiltonians": 2, "validation_hamiltonians": 2, "shot_counts": [2, 4]}
    return feature, config, bank


class SnapshotDataTests(unittest.TestCase):
    def test_sector_reconstruction_roundtrip_and_global_gauge(self):
        for length in (3, 4):
            lattice = square_lattice(length)
            tau = np.random.default_rng(31).choice(np.array([-1, 1], dtype=np.int8), (7, length * length))
            ends = lattice.edge_vertex_indices
            states = tau[:, ends[:, 0]] * tau[:, ends[:, 1]]
            actual = reconstruct_pseudospins(states)
            np.testing.assert_array_equal(actual, tau * tau[:, :1])
            self.assertTrue(np.all(actual[:, 0] == 1))

    def test_zero_plaquette_flux_bad_winding_is_rejected(self):
        length = 4
        states = np.ones((1, 2 * length * length), dtype=np.int8)
        # A seam crossing each row changes one Wilson loop but no face product.
        states.reshape(1, length, length, 2)[:, :, -1, 0] = -1
        lattice = square_lattice(length)
        self.assertTrue(np.all(np.prod(states[:, lattice.face_edge_indices], axis=-1) == 1))
        with self.assertRaisesRegex(ValueError, "sector"):
            reconstruct_pseudospins(states)
        states = np.ones_like(states)
        states[0, 0] = -1
        with self.assertRaises(ValueError):
            reconstruct_pseudospins(states)

    def test_reject_invalid_spin_types_shapes_and_length(self):
        for value in (np.ones((2, 18)), np.zeros((2, 18), dtype=np.int8), np.ones((18,), dtype=np.int8), np.ones((2, 20), dtype=np.int8)):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                reconstruct_pseudospins(value)
        with self.assertRaises(ValueError):
            reconstruct_pseudospins(np.ones((2, 18), dtype=np.int8), L=4)

    def test_atomic_pause_resume_nested_records_and_original_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature, config, bank = fixture(root)
            before = {str(path.relative_to(bank)): sha(path) for path in bank.rglob("*") if path.is_file()}
            output = root / "codex material/cache"
            events = []
            prepare_snapshot_cache(root, feature, output, config, progress=events.append,
                                   should_stop=lambda: len(events) >= 1)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["state"], "paused")
            with self.assertRaises(ValueError):
                load_snapshot_cache(output)
            first_part = next((output / "parts").rglob("*.npz"))
            part_hash = sha(first_part)
            prepare_snapshot_cache(root, feature, output, config)
            self.assertEqual(sha(first_part), part_hash)
            data = load_snapshot_cache(output)
            self.assertEqual(data.train.states.shape, (2, 4, 18))
            self.assertEqual(data.validation.states.shape, (2, 4, 18))
            self.assertEqual(data.shot_counts, (2, 4))
            self.assertTrue(set(data.train.uids).isdisjoint(data.validation.uids))
            for split in (data.train, data.validation):
                np.testing.assert_array_equal(split.spin_maps(2), split.spin_maps(4)[:, :2])
                expected = split.states[:, :2].mean(axis=1, dtype=np.float32).reshape(2, 3, 3, 2).transpose(0, 3, 1, 2)
                np.testing.assert_array_equal(split.means(2), expected)
                self.assertFalse(split.states.flags.writeable)
                self.assertTrue(all(len(set(row)) == 4 for row in split.record_indices))
            row = np.flatnonzero(data.train.ids == "h0")[0]
            self.assertEqual(len(np.unique(data.train.states[row], axis=0)), 1)
            self.assertEqual(data.train.states[row].shape[0], 4)
            self.assertEqual(before, {str(path.relative_to(bank)): sha(path) for path in bank.rglob("*") if path.is_file()})
            frozen = {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()}
            prepare_snapshot_cache(root, feature, output, config)
            self.assertEqual(frozen, {str(path.relative_to(output)): sha(path) for path in output.rglob("*") if path.is_file()})

    def test_uid_selection_is_independent_of_manifest_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature, config, _ = fixture(root)
            config.update(train_hamiltonians=1, validation_hamiltonians=1)
            first = root / "codex material/first"
            prepare_snapshot_cache(root, feature, first, config)
            a = load_snapshot_cache(first)
            value = json.loads(feature.read_text())
            value["selected_ids"] = {key: list(reversed(ids)) for key, ids in value["selected_ids"].items()}
            feature.write_text(json.dumps(value))
            second = root / "codex material/second"
            prepare_snapshot_cache(root, feature, second, config)
            b = load_snapshot_cache(second)
            for name in ("train", "validation"):
                left, right = getattr(a, name), getattr(b, name)
                np.testing.assert_array_equal(left.ids, right.ids)
                np.testing.assert_array_equal(left.record_indices, right.record_indices)
                np.testing.assert_array_equal(left.states, right.states)

    def test_changed_config_input_hash_and_cache_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature, config, bank = fixture(root)
            output = root / "codex material/cache"
            prepare_snapshot_cache(root, feature, output, config)
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_snapshot_cache(root, feature, output, {**config, "selection_seed": 82})
            path = bank / "instances/h0.npz"
            with path.open("ab") as stream:
                stream.write(b"changed-source")
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare_snapshot_cache(root, feature, root / "codex material/new", config)
            archive = output / "train.npz"
            with archive.open("ab") as stream:
                stream.write(b"changed-cache")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_snapshot_cache(output)

    def test_bad_configs_and_cross_split_uid_leak_fail_before_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature, config, _ = fixture(root)
            for change in ({"shot_counts": [4, 2]}, {"shot_counts": [2, 2]}, {"train_hamiltonians": 3}, {"selection_seed": True}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    prepare_snapshot_cache(root, feature, root / "codex material/cache", {**config, **change})
            value = json.loads(feature.read_text())
            value["origins"]["h2"]["instance"]["hamiltonian_uid"] = value["origins"]["h0"]["instance"]["hamiltonian_uid"]
            feature.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "identities"):
                prepare_snapshot_cache(root, feature, root / "codex material/leak", config)


if __name__ == "__main__":
    unittest.main()
