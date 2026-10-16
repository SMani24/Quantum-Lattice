"""Independent feature construction, immutable provenance and held-out guards."""
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.estimator_features import (
    load_feature_views, prepare_feature_views,
)
from quantum_lattice.data.estimator_validation import _versions
from quantum_lattice.data.mlp_data import load_learning_data
from quantum_lattice.lattice import square_lattice
from tests.unit import test_mlp_data as original_fixture


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


class EstimatorFeatureTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the fixture builder, not its tests. Its eight independently
        # seeded original archives include states and test files to guard access.
        fixture = original_fixture.MLPDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.bank = self.root
        self.run = self.root / "codex material/artifacts/estimator_validation/fixture"
        self.output = self.root / "codex material/artifacts/estimator_mlp/fixture/features"
        self.run.mkdir(parents=True)
        sources = {}
        for name in ("data/estimator_features.py", "data/mlp_data.py", "data/generation.py"):
            path = self.root / "src/quantum_lattice" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source " + name)
            sources[str(path.relative_to(self.root))] = _sha(path)
        verification = json.loads((self.bank / "verification.json").read_text())
        verified = {item["id"]: item["sha256"] for item in verification["instances"]}
        entries, tasks = {}, []
        self.oracle = {}
        for item in fixture.plan:
            if item["split"] == "test":
                continue
            key = f"L3-{item['id']}"
            task = {"stage": "saved", "key": key, "L": 3, "id": item["id"],
                    "stratum": "complete non-test bank", "replica": None}
            tasks.append(task)
            entries[item["id"]] = {"instance": item, "sha256": verified[item["id"]]}
            directory = self.run / "jobs/saved" / key
            directory.mkdir(parents=True)
            with np.load(self.bank / "instances" / f"{item['id']}.npz") as original:
                star, pair = original["star_mean"], original["pair_mean"]
            # These distinct synthetic means are construction oracles, not
            # physically enumerated expectations. f>1 must remain unclipped.
            gstar = np.arange(9, dtype=float)/30 + .1 + item["index"]/1000
            gpair = np.arange(18, dtype=float)/40 + .2
            estar, epair = np.full(9, .61), np.full(18, .73)
            self.oracle[item["id"]] = {"g": (gstar, gpair), "exact": (estar, epair)}
            np.savez_compressed(directory / "coordinates.npz", star_f_mean=star + 1e-14,
                pair_f_mean=pair, star_g_mean=gstar, pair_g_mean=gpair,
                star_truth=estar, pair_truth=epair)
            _write(directory / "summary.json", {**task, "max_identity_error": 1e-15})
            _write(directory / "done.json", {"files": {name: _sha(directory / name)
                for name in ("coordinates.npz", "summary.json")}})
        inputs = {name: _sha(self.bank / name) for name in ("config.json", "manifest.json", "verification.json")}
        self.plan = {"schema_version": 1, "sources": sources, "inputs": inputs,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "versions": _versions()},
            "banks": {"3": {"directory": ".", "config": fixture.config, "entries": entries}}, "tasks": tasks}
        self.write_plan()

    def write_plan(self):
        _write(self.run / "plan.json", self.plan)
        _write(self.run / "plan-sha256.json", {"sha256": _sha(self.run / "plan.json")})

    def prepare(self):
        return prepare_feature_views(self.root, self.run, self.output, sizes=(3,))

    def rebind_original_hash(self, name):
        for filename in ("manifest.json", "verification.json", "config.json"):
            self.plan["inputs"][filename] = _sha(self.bank / filename)
        check = json.loads((self.bank / "verification.json").read_text())
        self.plan["banks"]["3"]["entries"][name]["sha256"] = next(
            entry["sha256"] for entry in check["instances"] if entry["id"] == name)
        self.write_plan()

    def mutate_coordinates(self, name, change, update_hash=True):
        directory = self.run / "jobs/saved" / f"L3-{name}"
        with np.load(directory / "coordinates.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        change(arrays)
        np.savez_compressed(directory / "coordinates.npz", **arrays)
        if update_hash:
            marker = json.loads((directory / "done.json").read_text())
            marker["files"]["coordinates.npz"] = _sha(directory / "coordinates.npz")
            _write(directory / "done.json", marker)

    def test_ordered_features_labels_groups_and_no_clipping(self):
        self.assertEqual(self.prepare(), self.output)
        views = load_feature_views(self.output, 3)
        baseline = load_learning_data(self.bank, mode="all_edges")
        ends = square_lattice(3).edge_vertex_indices
        self.assertEqual(set(views), {"f", "g", "exact"})
        for split in ("train", "validation"):
            old = getattr(baseline, split)
            np.testing.assert_array_equal(getattr(views["f"], split).x, old.x)
            for view in views:
                actual = getattr(views[view], split)
                self.assertIs(actual.y, getattr(views["f"], split).y)
                self.assertIsNone(views[view].test)
                self.assertFalse(actual.x.flags.writeable)
                np.testing.assert_array_equal(actual.y, old.y)
                np.testing.assert_array_equal(actual.hamiltonian_ids, old.hamiltonian_ids)
                np.testing.assert_array_equal(actual.hamiltonian_uids, old.hamiltonian_uids)
                np.testing.assert_array_equal(actual.edge_ids, old.edge_ids)
                if view != "f":
                    for name in sorted(set(old.hamiltonian_ids)):
                        star, pair = self.oracle[name][view]
                        expected = np.array([[star[u], star[v], pair[e]]
                            for e, (u,v) in enumerate(ends)], dtype=np.float32)
                        np.testing.assert_array_equal(actual.x[actual.hamiltonian_ids == name], expected)
        self.assertTrue(np.any(views["f"].train.x > 1))
        self.assertNotEqual(views["f"].provenance["source_fingerprint"], views["g"].provenance["source_fingerprint"])
        self.assertFalse(views["exact"].provenance["training_eligible"])
        self.assertTrue(views["g"].provenance["training_eligible"])

    def test_prepare_and_load_do_not_read_test_payloads_or_states(self):
        test_paths = {str(self.bank / "instances" / f"{item['id']}.npz")
                      for item in self.fixture.plan if item["split"] == "test"}
        original_load, original_item = np.load, np.lib.npyio.NpzFile.__getitem__
        def guarded_load(path, *args, **kwargs):
            self.assertNotIn(str(path), test_paths)
            return original_load(path, *args, **kwargs)
        def guarded_item(archive, key):
            self.assertNotIn(key, {"states", "states_packed", "star_values", "z_values"})
            return original_item(archive, key)
        with patch("numpy.load", side_effect=guarded_load), patch.object(np.lib.npyio.NpzFile, "__getitem__", guarded_item):
            self.prepare()
            load_feature_views(self.output, 3)
        # After preparation no original bank or coordinate file is loaded.
        def cache_only(path, *args, **kwargs):
            self.assertEqual(Path(path), self.output / "L3/views.npz")
            return original_load(path, *args, **kwargs)
        with patch("numpy.load", side_effect=cache_only):
            load_feature_views(self.output, 3)

    def test_resume_preserves_committed_cache_bytes(self):
        self.prepare()
        files = [path for path in self.output.rglob("*") if path.is_file()]
        before = {path: _sha(path) for path in files}
        self.prepare()
        self.assertEqual(before, {path: _sha(path) for path in files})
        # A missing final root commit can be recovered from a complete size.
        (self.output / "manifest.json").unlink()
        self.prepare()
        load_feature_views(self.output, 3)

    def test_source_and_plan_tampering_rejected(self):
        (self.root / next(iter(self.plan["sources"]))).write_text("changed")
        with self.assertRaisesRegex(ValueError, "pinned input/source"):
            self.prepare()
        self.plan["sources"] = {}
        self.write_plan()
        with (self.run / "plan.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "immutable plan hash"):
            self.prepare()

    def test_saved_coordinate_corruption_and_wrong_f_rejected(self):
        name = next(iter(self.plan["banks"]["3"]["entries"]))
        self.mutate_coordinates(name, lambda a: a["star_f_mean"].__setitem__(0, 99), update_hash=False)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.prepare()
        self.mutate_coordinates(name, lambda a: None)
        with self.assertRaisesRegex(ValueError, "reproduce original"):
            self.prepare()

    def test_nan_and_bounded_feature_failures(self):
        name = next(iter(self.plan["banks"]["3"]["entries"]))
        self.mutate_coordinates(name, lambda a: a["star_g_mean"].__setitem__(0, np.nan))
        with self.assertRaisesRegex(ValueError, "invalid feature coordinate"):
            self.prepare()
        self.mutate_coordinates(name, lambda a: a["star_g_mean"].__setitem__(0, 1.4))
        with self.assertRaisesRegex(ValueError, r"outside \[0,1\]"):
            self.prepare()

    def test_test_injection_and_missing_saved_group_rejected(self):
        item = next(item for item in self.fixture.plan if item["split"] == "test")
        self.plan["banks"]["3"]["entries"][item["id"]] = {"instance": item, "sha256": "0"*64}
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "cover every declared non-test"):
            self.prepare()
        del self.plan["banks"]["3"]["entries"][item["id"]]
        self.plan["tasks"].pop()
        self.write_plan()
        (self.output / "request.json").unlink()
        with self.assertRaisesRegex(ValueError, "cover every declared non-test"):
            self.prepare()

    def test_cache_archive_and_nested_manifest_corruption_rejected(self):
        self.prepare()
        path = self.output / "L3/views.npz"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "archive checksum"):
            load_feature_views(self.output, 3)
        manifest = self.output / "L3/manifest.json"
        with manifest.open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "size manifest checksum"):
            load_feature_views(self.output, 3)

    def test_rebound_cache_labels_and_group_leakage_still_rejected(self):
        self.prepare()
        directory = self.output / "L3"
        def rewrite(change_arrays=None, change_size=None):
            with np.load(directory / "views.npz") as archive:
                arrays = {key: archive[key] for key in archive.files}
            size = json.loads((directory / "manifest.json").read_text())
            if change_arrays:
                change_arrays(arrays)
            if change_size:
                change_size(size)
            np.savez_compressed(directory / "views.npz", **arrays)
            size["archive_sha256"] = _sha(directory / "views.npz")
            canonical = json.dumps(size["origins"], sort_keys=True, separators=(",", ":"), allow_nan=False)
            size["origins_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
            _write(directory / "manifest.json", size)
            root = json.loads((self.output / "manifest.json").read_text())
            root["sizes"]["3"].update(sha256=_sha(directory / "manifest.json"), archive_sha256=size["archive_sha256"])
            _write(self.output / "manifest.json", root)
        rewrite(lambda arrays: arrays["train__y"].__setitem__((0,0), 1.23))
        with self.assertRaisesRegex(ValueError, "seeded"):
            load_feature_views(self.output, 3)
        baseline = load_learning_data(self.bank, mode="all_edges")
        rewrite(lambda arrays: arrays.update(train__y=baseline.train.y))
        train_uid = baseline.train.hamiltonian_uids[0]
        val_name = baseline.validation.hamiltonian_ids[0]
        rewrite(lambda arrays: arrays["validation__hamiltonian_uids"].__setitem__(slice(0,18), train_uid),
                lambda size: size["origins"][val_name]["instance"].update(hamiltonian_uid=train_uid))
        with self.assertRaisesRegex(ValueError, "split leakage"):
            load_feature_views(self.output, 3)

    def test_exclusive_preparation_lease(self):
        self.output.mkdir(parents=True)
        with (self.output / ".prepare.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "active in another process"):
                self.prepare()
            fcntl.flock(handle, fcntl.LOCK_UN)
        self.prepare()

    def test_request_and_scope_are_frozen(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "restricted to L3/L4"):
            prepare_feature_views(self.root, self.run, self.output, sizes=(8,))
        with self.assertRaisesRegex(ValueError, "frozen feature request"):
            prepare_feature_views(self.root, self.run, self.output, sizes=(3,4))
        with self.assertRaisesRegex(ValueError, "absent"):
            load_feature_views(self.output, 4)
        with self.assertRaisesRegex(ValueError, "beneath ignored"):
            prepare_feature_views(self.root, self.run, self.root / "outside", sizes=(3,))


if __name__ == "__main__":
    unittest.main()
