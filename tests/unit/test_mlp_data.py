import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.generation import _bank, _canonical
from quantum_lattice.data.mlp_data import load_learning_data
from quantum_lattice.lattice import square_lattice


class MLPDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "schema_version": 1, "lattice": {"type": "square_periodic", "L": 3},
            "fields": {"channel": "z_only", "distribution": "uniform", "low": -1.7, "high": 1.7},
            "dataset": {"hamiltonians": 8, "seed": 123,
                        "split_fractions": {"train": 0.5, "validation": 0.25, "test": 0.25}},
            "sampler": {"backend": "python", "total_sweeps": 30, "burn_in_sweeps": 10,
                        "thinning_sweeps": 2, "lazy_probability": 0.05},
            "statistics": {"block_size_records": 2}, "execution": {"workers": 1, "chunk_sweeps": 10},
        }
        lattice = square_lattice(3)
        self.plan = _bank(self.config)
        (self.root / "instances").mkdir()
        entries, checks = [], []
        for item in self.plan:
            fields = np.random.default_rng(np.random.SeedSequence(item["field_seed"])).uniform(-1.7, 1.7, 18)
            stars = np.linspace(0.2, 1.2, 9) + item["index"] / 100
            pairs = np.linspace(0.1, 0.8, 18)
            ends = lattice.edge_vertex_indices
            metadata = {"schema_version": 1, "instance": item,
                        "completed_sweeps": 30, "retained_records": 10}
            path = self.root / "instances" / (item["id"] + ".npz")
            np.savez_compressed(path, b_z=fields, target_b=fields, target_abs_b=np.abs(fields),
                star_mean=stars, pair_mean=pairs, edge_vertex_indices=ends,
                mlp_X=np.column_stack((stars[ends[:, 0]], stars[ends[:, 1]], pairs)),
                metadata_json=np.array(_canonical(metadata)),
                states=np.ones((10, 18), dtype=np.int8))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append({"id": item["id"], "split": item["split"],
                "path": f"instances/{item['id']}.npz", "sha256": sha, "records": 10, "local_rows": 18})
            checks.append({"id": item["id"], "split": item["split"], "sha256": sha})
        manifest = {"schema_version": 1, "instances": self.plan,
                    "config_sha256": hashlib.sha256(_canonical(self.config).encode()).hexdigest(),
                    "source_hashes": {"simulation.py": "recorded-source"},
                    "measurement_scope": "known-field simulation estimators", "split_protocol": "fixture"}
        index = {"schema_version": 1, "instances": entries, "local_rows": 8 * 18,
                 "split_counts": {key: sum(i["split"] == key for i in self.plan)
                                  for key in ("train", "validation", "test")}}
        for name, value in (("config.json", self.config), ("manifest.json", manifest),
                            ("dataset.json", index), ("verification.json", {"schema_version": 1, "passed": True, "instances": checks}),
                            ("status.json", {"state": "complete", "completed": 8})):
            self.write_json(name, value)

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def mutate_json(self, name, change):
        document = json.loads((self.root / name).read_text())
        change(document)
        self.write_json(name, document)

    def mutate_archive(self, item, change):
        path = self.root / "instances" / (item["id"] + ".npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        change(arrays)
        np.savez_compressed(path, **arrays)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("dataset.json", "verification.json"):
            def update(doc):
                for entry in doc["instances"]:
                    if entry["id"] == item["id"]:
                        entry["sha256"] = sha
            self.mutate_json(name, update)

    def test_modes_keep_groups_and_all_test_edges(self):
        paper = load_learning_data(self.root, training_hamiltonians=2, include_test=True)
        all_rows = load_learning_data(self.root, mode="all_edges", include_test=True)
        self.assertEqual((len(paper.train), len(paper.validation), len(paper.test)), (2, 2, 36))
        self.assertEqual((len(all_rows.train), len(all_rows.validation), len(all_rows.test)), (72, 36, 36))
        np.testing.assert_array_equal(paper.test.x, all_rows.test.x)
        np.testing.assert_array_equal(paper.test.y, all_rows.test.y)
        self.assertEqual(paper.train.x.dtype, np.float32)
        self.assertEqual(paper.train.y.shape, (2, 1))
        self.assertTrue(np.all(paper.train.edge_ids == 0))
        groups = [set(rows.hamiltonian_uids) for rows in (all_rows.train, all_rows.validation, all_rows.test)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(len(paper.provenance["unused_training_ids"]), 2)
        self.assertTrue(np.any(all_rows.train.x > 1))  # Monte Carlo estimates must not be clipped.

    def test_training_never_opens_test_payload_and_never_reads_states(self):
        test_paths = {str(self.root / "instances" / (i["id"] + ".npz")) for i in self.plan if i["split"] == "test"}
        real_load = np.load
        real_getitem = np.lib.npyio.NpzFile.__getitem__
        def guarded_load(path, *args, **kwargs):
            self.assertNotIn(str(path), test_paths)
            return real_load(path, *args, **kwargs)
        def guarded_item(archive, key):
            self.assertNotIn(key, {"states", "states_packed", "star_values", "z_values"})
            return real_getitem(archive, key)
        with patch("quantum_lattice.data.mlp_data.np.load", side_effect=guarded_load), \
             patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded_item):
            training = load_learning_data(self.root, training_hamiltonians=2)
        evaluated = load_learning_data(self.root, training_hamiltonians=2, include_test=True)
        self.assertIsNone(training.test)
        self.assertEqual(training.provenance["source_fingerprint"], evaluated.provenance["source_fingerprint"])

    def test_incomplete_unverified_and_insufficient_are_rejected(self):
        self.mutate_json("status.json", lambda d: d.update(state="verifying"))
        with self.assertRaisesRegex(ValueError, "COMPLETE"):
            load_learning_data(self.root, training_hamiltonians=2)
        self.mutate_json("status.json", lambda d: d.update(state="complete"))
        self.mutate_json("verification.json", lambda d: d.update(passed=False))
        with self.assertRaisesRegex(ValueError, "passed"):
            load_learning_data(self.root, training_hamiltonians=2)
        self.mutate_json("verification.json", lambda d: d.update(passed=True))
        with self.assertRaisesRegex(ValueError, "7450"):
            load_learning_data(self.root)

    def test_config_plan_and_split_corruption_rejected(self):
        self.mutate_json("config.json", lambda d: d["dataset"].update(seed=999))
        with self.assertRaisesRegex(ValueError, "configuration hash"):
            load_learning_data(self.root, training_hamiltonians=2)
        self.write_json("config.json", self.config)
        self.mutate_json("dataset.json", lambda d: d["instances"][0].update(split="other"))
        with self.assertRaisesRegex(ValueError, "split"):
            load_learning_data(self.root, training_hamiltonians=2)

    def test_checksum_corruption_rejected(self):
        item = next(i for i in self.plan if i["split"] == "train")
        path = self.root / "instances" / (item["id"] + ".npz")
        with path.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_learning_data(self.root, training_hamiltonians=2)

    def test_feature_alignment_and_labels_rejected_even_with_updated_hashes(self):
        item = self.plan[0]
        original = load_learning_data(self.root, mode="all_edges", include_test=True)
        self.mutate_archive(item, lambda arrays: arrays["mlp_X"].__setitem__((0, 2), 99.0))
        with self.assertRaisesRegex(ValueError, "ordered star/pair"):
            load_learning_data(self.root, mode="all_edges", include_test=True)
        self.mutate_archive(item, lambda a: a.update(mlp_X=np.column_stack((
            a["star_mean"][a["edge_vertex_indices"][:, 0]], a["star_mean"][a["edge_vertex_indices"][:, 1]], a["pair_mean"]))))
        self.mutate_archive(item, lambda arrays: arrays["target_abs_b"].__setitem__(0, 0))
        with self.assertRaisesRegex(ValueError, "magnitude labels"):
            load_learning_data(self.root, mode="all_edges", include_test=True)
        self.assertEqual(original.L, 3)

    def test_wrong_metadata_identity_rejected(self):
        def wrong(arrays):
            meta = json.loads(arrays["metadata_json"].item())
            meta["instance"]["hamiltonian_uid"] = "wrong"
            arrays["metadata_json"] = np.array(json.dumps(meta))
        self.mutate_archive(self.plan[0], wrong)
        with self.assertRaisesRegex(ValueError, "identity"):
            load_learning_data(self.root, mode="all_edges", include_test=True)

    def test_duplicate_index_and_missing_id_rejected(self):
        self.mutate_json("dataset.json", lambda d: d["instances"].append(d["instances"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_learning_data(self.root, training_hamiltonians=2)

    def test_nonfinite_features_rejected(self):
        self.mutate_archive(self.plan[0], lambda arrays: arrays["mlp_X"].__setitem__((0, 0), np.nan))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_learning_data(self.root, mode="all_edges", include_test=True)


if __name__ == "__main__":
    unittest.main()
