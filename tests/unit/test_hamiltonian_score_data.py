"""Independent committed-output fixtures for read-only Hamiltonian scores."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.hamiltonian_scores import (
    load_prediction_maps, load_validation_fields, reference_sign_lift,
)
from quantum_lattice.data.mlp_data import EdgeRows, LearningData


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n")


class ScoreDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "features"
        (self.cache / "L3").mkdir(parents=True)
        self.L, self.edges = 3, 18
        self.selected = {"train": ["train-0"], "validation": ["validation-0", "validation-1"]}
        self.origins, self.field_groups = {}, {}
        rows = {}
        for split, names in self.selected.items():
            fields = []
            for index, name in enumerate(names):
                seed = [102 if split == "train" else 603, index, 0]
                self.origins[name] = {"instance": {"id": name, "split": split,
                    "hamiltonian_uid": "uid-" + name, "field_seed": seed}}
                fields.append(np.random.default_rng(np.random.SeedSequence(seed)).uniform(-1.7, 1.7, self.edges))
            fields = np.stack(fields)
            self.field_groups[split] = fields
            n = len(names) * self.edges
            rows[split] = EdgeRows(np.arange(3*n, dtype=np.float32).reshape(n, 3),
                np.abs(fields).astype(np.float32).reshape(n, 1),
                np.repeat(np.asarray(names), self.edges),
                np.repeat(np.asarray(["uid-" + name for name in names]), self.edges),
                np.tile(np.arange(self.edges, dtype=np.int64), len(names)))
        self.views = {view: LearningData(rows["train"], rows["validation"], None, 3, 0,
            "all_edges", {"training_eligible": view != "exact"}, self.cache)
            for view in ("f", "g", "exact")}
        self.size = {"origins": self.origins, "selected_ids": self.selected,
            "original_provenance": {"settings": {"fields": {
                "channel": "z_only", "distribution": "uniform", "low": -1.7, "high": 1.7}}}}
        self._commit_cache_metadata()
        self.loader = patch("quantum_lattice.data.hamiltonian_scores.load_feature_views", return_value=self.views)
        self.mock_loader = self.loader.start()
        self.addCleanup(self.loader.stop)
        self.task = {"L": 3, "key": "L3-seed23-f", "seed": 23, "view": "f"}
        self.plan_sha = "b" * 64

    def _commit_cache_metadata(self):
        write_json(self.cache / "L3/manifest.json", self.size)
        write_json(self.cache / "manifest.json", {"sizes": {"3": {
            "sha256": sha(self.cache / "L3/manifest.json")}}})

    def reference(self):
        return load_validation_fields(self.cache, 3)

    def make_job(self, reference):
        job = self.root / self.task["key"]
        job.mkdir(exist_ok=True)
        for name in ("latest.pt", "best.pt", "history.json", "per-hamiltonian.csv"):
            (job / name).write_bytes(b"byte-only verification; not a serialized checkpoint")
        evaluation = {"task": self.task, "scope": "development validation; not a new final test",
            "checkpoint_selection": "lowest own-view validation MSE",
            "prediction_policy": "raw unclipped linear predictions",
            "exact_input_policy": "crossed evaluation only; never checkpoint selection",
            "test_payload_loaded": False}
        write_json(job / "evaluation.json", evaluation)
        rows = reference.validation_rows
        self.arrays = {"target": rows.y.copy(), "hamiltonian_ids": rows.hamiltonian_ids.copy(),
            "hamiltonian_uids": rows.hamiltonian_uids.copy(), "edge_ids": rows.edge_ids.copy()}
        for k, view in enumerate(("f", "g", "exact")):
            self.arrays["prediction_" + view] = (rows.y - np.float32(k + 2)).astype(np.float32)
        self.commit_job(job)
        return job

    def commit_job(self, job):
        np.savez_compressed(job / "predictions.npz", **self.arrays)
        names = ("latest.pt", "best.pt", "history.json", "evaluation.json",
                 "per-hamiltonian.csv", "predictions.npz")
        evaluation = json.loads((job / "evaluation.json").read_text())
        write_json(job / "done.json", {"schema_version": 1, "task": self.task,
            "plan_sha256": self.plan_sha, "hashes": {name: sha(job / name) for name in names},
            "summary": evaluation})

    def load(self, job, reference):
        return load_prediction_maps(job, self.task, reference, plan_sha256=self.plan_sha)

    def test_fields_reconstructed_in_float64_and_copy_is_readonly(self):
        reference = self.reference()
        self.mock_loader.assert_called_once_with(self.cache.resolve(), 3)
        np.testing.assert_array_equal(reference.true_b_z, self.field_groups["validation"])
        self.assertEqual(reference.true_b_z.dtype, np.float64)
        self.assertEqual(reference.source_manifest_sha256, sha(self.cache / "manifest.json"))
        for array in (reference.true_b_z, reference.hamiltonian_ids, reference.hamiltonian_uids,
                      reference.validation_rows.y, reference.validation_rows.edge_ids):
            self.assertFalse(array.flags.writeable)
        original = self.views["f"].validation.y
        self.assertFalse(np.shares_memory(reference.validation_rows.y, original))
        original[:] = 0
        self.assertGreater(reference.validation_rows.y.max(), 0)

    def test_shuffled_field_rows_rejected_instead_of_reordered(self):
        original = self.views["f"].validation
        order = np.arange(len(original))[::-1]
        shuffled = EdgeRows(*(getattr(original, key)[order] for key in
            ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids")))
        self.views["f"] = replace(self.views["f"], validation=shuffled)
        with self.assertRaisesRegex(ValueError, "canonical field reference"):
            self.reference()

    def test_missing_edge_or_seed_label_corruption_rejected(self):
        for defect in ("missing", "label"):
            with self.subTest(defect=defect):
                original = self.views["g"].validation
                if defect == "missing":
                    changed = EdgeRows(*(getattr(original, key)[:-1] for key in
                        ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids")))
                else:
                    y = original.y.copy(); y[0, 0] += .5
                    changed = replace(original, y=y)
                self.views["g"] = replace(self.views["g"], validation=changed)
                with self.assertRaises(ValueError):
                    self.reference()
                self.views["g"] = replace(self.views["g"], validation=original)

    def test_test_payload_and_exact_training_eligibility_rejected(self):
        original = self.views["exact"]
        for changed in (replace(original, test=original.validation),
                        replace(original, provenance={"training_eligible": True})):
            self.views["exact"] = changed
            with self.assertRaises(ValueError):
                self.reference()

    def test_duplicate_uid_between_train_validation_rejected(self):
        self.origins["validation-0"]["instance"]["hamiltonian_uid"] = "uid-train-0"
        for view in self.views:
            data = self.views[view]
            uids = data.validation.hamiltonian_uids.copy()
            uids[:self.edges] = "uid-train-0"
            self.views[view] = replace(data, validation=replace(data.validation, hamiltonian_uids=uids))
        self._commit_cache_metadata()
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.reference()

    def test_origin_split_and_distribution_are_checked(self):
        self.origins["validation-0"]["instance"]["split"] = "test"
        self._commit_cache_metadata()
        with self.assertRaisesRegex(ValueError, "origin identity/split"):
            self.reference()
        self.origins["validation-0"]["instance"]["split"] = "validation"
        self.size["original_provenance"]["settings"]["fields"]["distribution"] = "normal"
        self._commit_cache_metadata()
        with self.assertRaisesRegex(ValueError, "uniform"):
            self.reference()

    def test_verified_predictions_preserve_signed_raw_values(self):
        reference = self.reference(); job = self.make_job(reference)
        before = {path.name: sha(path) for path in job.iterdir()}
        result = self.load(job, reference)
        self.assertEqual(set(result), {"f", "g", "exact"})
        for view, matrix in result.items():
            self.assertEqual(matrix.dtype, np.float64)
            self.assertEqual(matrix.shape, (2, self.edges))
            np.testing.assert_array_equal(matrix, self.arrays["prediction_" + view].reshape(2, self.edges))
            self.assertTrue(np.all(matrix < 0))
            self.assertFalse(matrix.flags.writeable)
        self.assertEqual(before, {path.name: sha(path) for path in job.iterdir()})

    def test_every_declared_file_hash_is_checked(self):
        reference = self.reference(); job = self.make_job(reference)
        for name in ("latest.pt", "best.pt", "history.json", "evaluation.json",
                     "per-hamiltonian.csv", "predictions.npz"):
            with self.subTest(name=name):
                path = job / name; saved = path.read_bytes()
                path.write_bytes(saved + b"changed")
                with self.assertRaisesRegex(ValueError, "checksum/path"):
                    self.load(job, reference)
                path.write_bytes(saved)

    def test_prediction_identity_labels_order_and_edges_checked_after_recommit(self):
        reference = self.reference(); job = self.make_job(reference)
        for key in ("target", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
            with self.subTest(key=key):
                saved = self.arrays[key].copy()
                if key == "target": self.arrays[key][0, 0] += .1
                elif key == "edge_ids": self.arrays[key][0] = 5
                else: self.arrays[key][0] = "wrong"
                self.commit_job(job)
                with self.assertRaisesRegex(ValueError, "identity/order/labels"):
                    self.load(job, reference)
                self.arrays[key] = saved
        # Reordering every coordinate together still violates the frozen order.
        self.arrays = {key: value[::-1] for key, value in self.arrays.items()}
        self.commit_job(job)
        with self.assertRaises(ValueError): self.load(job, reference)

    def test_missing_extra_nonfinite_and_wrong_precision_predictions_rejected(self):
        reference = self.reference(); job = self.make_job(reference)
        baseline = {key: value.copy() for key, value in self.arrays.items()}
        for defect in ("missing", "extra", "nonfinite", "precision", "shape"):
            self.arrays = {key: value.copy() for key, value in baseline.items()}
            if defect == "missing": self.arrays.pop("prediction_exact")
            elif defect == "extra": self.arrays["unexpected"] = np.zeros(1)
            elif defect == "nonfinite": self.arrays["prediction_g"][0, 0] = np.nan
            elif defect == "precision": self.arrays["prediction_g"] = self.arrays["prediction_g"].astype(np.float64)
            else: self.arrays["prediction_g"] = self.arrays["prediction_g"][:-1]
            self.commit_job(job)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.load(job, reference)

    def test_done_plan_task_inventory_and_scope_guard(self):
        reference = self.reference(); job = self.make_job(reference)
        marker_path = job / "done.json"
        baseline = json.loads(marker_path.read_text())
        for defect in ("plan", "task", "inventory", "schema", "summary"):
            marker = json.loads(json.dumps(baseline))
            if defect == "plan": marker["plan_sha256"] = "c" * 64
            elif defect == "task": marker["task"]["view"] = "g"
            elif defect == "inventory": marker["hashes"].pop("latest.pt")
            elif defect == "schema": marker["schema_version"] = True
            else: marker["summary"]["test_payload_loaded"] = True
            write_json(marker_path, marker)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.load(job, reference)
        write_json(marker_path, baseline)
        evaluation = json.loads((job / "evaluation.json").read_text())
        evaluation["test_payload_loaded"] = True
        write_json(job / "evaluation.json", evaluation); self.commit_job(job)
        with self.assertRaisesRegex(ValueError, "validation scope"):
            self.load(job, reference)

    def test_invalid_task_and_plan_digest_do_not_load_predictions(self):
        reference = self.reference()
        for task, digest in (({**self.task, "L": True}, self.plan_sha),
                             ({**self.task, "key": "../escaped"}, self.plan_sha),
                             (self.task, "not-a-digest")):
            with self.subTest(task=task), self.assertRaises(ValueError):
                load_prediction_maps(self.root / "absent", task, reference, plan_sha256=digest)

    def test_reference_sign_lift_zero_and_negative_output_policy(self):
        truth = np.array([[2., -3., 0.], [-1., 4., -5.]])
        prediction = np.array([[-.5, -.8, 19.], [.2, 6., -.7]])
        lifted = reference_sign_lift(truth, prediction)
        np.testing.assert_array_equal(lifted, [[-.5, .8, 0.], [-.2, 6., .7]])
        self.assertEqual(lifted.dtype, np.float64)
        self.assertFalse(lifted.flags.writeable)
        for bad_truth, bad_prediction in ((truth.ravel(), prediction.ravel()),
            (truth, prediction[:, :-1]), (truth, prediction + 1j),
            (truth * np.nan, prediction), (np.empty((0, 3)), np.empty((0, 3)))):
            with self.assertRaises(ValueError):
                reference_sign_lift(bad_truth, bad_prediction)


if __name__ == "__main__":
    unittest.main()
