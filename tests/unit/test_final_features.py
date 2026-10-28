"""Independent compact map layout, source integrity and split-access guards."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.final_features import prepare_features, load_features
from quantum_lattice.data.generation import _bank, _canonical
from quantum_lattice.lattice import square_lattice


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


class FinalFeatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bank_root = self.root / "banks"
        self.run = self.root / "codex material/artifacts/estimator_validation/fixture"
        self.output = self.root / "codex material/artifacts/final_benchmark/fixture/features"
        self.run.mkdir(parents=True)
        source = self.root / "scientific.py"
        source.write_text("fixture numerical source")
        self.plan = {"schema_version": 1, "sources": {"scientific.py": sha(source)},
                     "inputs": {}, "config": {"saved": {"exact_sizes": [3, 4]}},
                     "banks": {}, "tasks": []}
        self.oracle = {}
        for L in (3, 8):
            self.make_bank(L)
        self.write_plan()

    def make_bank(self, L):
        directory = self.bank_root / f"L{L}/run"
        (directory / "instances").mkdir(parents=True)
        config = {
            "schema_version": 1, "lattice": {"type": "square_periodic", "L": L},
            "fields": {"channel": "z_only", "distribution": "uniform", "low": -1.7, "high": 1.7},
            "dataset": {"hamiltonians": 8, "seed": 100 + L,
                        "split_fractions": {"train": .5, "validation": .25, "test": .25}},
            "sampler": {"backend": "python", "total_sweeps": 30, "burn_in_sweeps": 10,
                        "thinning_sweeps": 2, "lazy_probability": .05},
            "statistics": {"block_size_records": 2}, "execution": {"workers": 1, "chunk_sweeps": 10}}
        instances, indexed, checks, entries = _bank(config), [], [], {}
        edges, vertices = 2 * L * L, L * L
        ends = square_lattice(L).edge_vertex_indices
        for item in instances:
            fields = np.random.default_rng(np.random.SeedSequence(item["field_seed"])).uniform(-1.7, 1.7, edges)
            star = np.linspace(.2, 1.2, vertices) + item["index"] / 100
            pair = np.linspace(.1, .8, edges)
            source = directory / "instances" / f"{item['id']}.npz"
            np.savez_compressed(source, b_z=fields, target_b=fields, target_abs_b=np.abs(fields),
                star_mean=star, pair_mean=pair, edge_vertex_indices=ends,
                mlp_X=np.column_stack((star[ends[:, 0]], star[ends[:, 1]], pair)),
                metadata_json=np.array(_canonical({"schema_version": 1, "instance": item,
                    "completed_sweeps": 30, "retained_records": 10})),
                states=np.ones((10, edges), dtype=np.int8))
            indexed.append({"id": item["id"], "split": item["split"], "path": f"instances/{item['id']}.npz",
                            "sha256": sha(source), "records": 10, "local_rows": edges})
            checks.append({"id": item["id"], "split": item["split"], "sha256": sha(source)})
            if item["split"] == "test":
                continue
            task = {"stage": "saved", "L": L, "id": item["id"], "key": f"L{L}-{item['id']}",
                    "stratum": "complete non-test bank", "replica": None}
            self.plan["tasks"].append(task)
            entries[item["id"]] = {"instance": item, "sha256": sha(source)}
            job = self.run / "jobs/saved" / task["key"]
            job.mkdir(parents=True)
            # Independent construction oracles, not physically simulated data.
            gstar = np.linspace(.1, .4, vertices) + item["index"] / 1000
            gpair = np.linspace(.2, .6, edges)
            values = dict(star_f_mean=star + 1e-14, pair_f_mean=pair,
                          star_g_mean=gstar, pair_g_mean=gpair)
            if L == 3:
                values.update(star_truth=np.full(vertices, .61), pair_truth=np.full(edges, .73))
            np.savez_compressed(job / "coordinates.npz", **values)
            write(job / "summary.json", {**task, "max_identity_error": 1e-15 if L == 3 else None})
            write(job / "done.json", {"files": {n: sha(job / n) for n in ("coordinates.npz", "summary.json")}})
            self.oracle[(L, item["id"])] = dict(star=star, pair=pair, gstar=gstar, gpair=gpair, fields=fields)
        counts = {split: sum(i["split"] == split for i in instances) for split in ("train", "validation", "test")}
        docs = {"config.json": config,
                "manifest.json": {"schema_version": 1, "instances": instances,
                    "config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest()},
                "dataset.json": {"schema_version": 1, "instances": indexed, "split_counts": counts, "local_rows": len(instances)*edges},
                "verification.json": {"schema_version": 1, "passed": True, "instances": checks},
                "status.json": {"state": "complete", "completed": len(instances)}}
        for filename, value in docs.items():
            write(directory / filename, value)
            self.plan["inputs"][str((directory / filename).relative_to(self.root))] = sha(directory / filename)
        self.plan["banks"][str(L)] = {"directory": str(directory.relative_to(self.root)), "entries": entries, "config": config}

    def write_plan(self):
        write(self.run / "plan.json", self.plan)
        write(self.run / "plan-sha256.json", {"sha256": sha(self.run / "plan.json")})

    def prepare(self, sizes=(3, 8), limits=None, stop_requested=None):
        return prepare_features(self.root, self.bank_root, self.run, self.output, sizes,
                                limits=limits, stop_requested=stop_requested)

    def test_direct_layout_float32_control_fields_and_optional_exact(self):
        self.assertEqual(self.prepare(), self.output)
        for L in (3, 8):
            data = load_features(self.output, L)
            for split, count in (("train", 4), ("validation", 2)):
                a = data[split]
                self.assertEqual(a["f"].shape, (count, 3, L, L))
                self.assertEqual(a["fields"].dtype, np.float64)
                self.assertEqual("exact" in a, L == 3)
                for i, name in enumerate(a["ids"]):
                    oracle = self.oracle[L, name]
                    expected_f = np.stack([oracle["star"].reshape(L,L),
                        oracle["pair"][0::2].reshape(L,L), oracle["pair"][1::2].reshape(L,L)]).astype(np.float32)
                    expected_g = np.stack([oracle["gstar"].reshape(L,L),
                        oracle["gpair"][0::2].reshape(L,L), oracle["gpair"][1::2].reshape(L,L)]).astype(np.float32)
                    np.testing.assert_array_equal(a["f"][i], expected_f)
                    np.testing.assert_array_equal(a["g"][i], expected_g)
                    np.testing.assert_array_equal(a["fields"][i], oracle["fields"])
                    for channel in (0,1):
                        np.testing.assert_array_equal(a["y"][i,channel], np.abs(oracle["fields"])[channel::2].reshape(L,L).astype(np.float32))
                self.assertFalse(a["f"].flags.writeable)
                self.assertFalse(a["fields"].flags.writeable)
                self.assertGreater(a["f"].max(), 1)
            self.assertFalse(set(data["train"]["uids"]) & set(data["validation"]["uids"]))
            self.assertFalse(data["manifest"]["test_payload_loaded"])

    def test_never_opens_test_payload_or_decompresses_states(self):
        tests = {str(self.root / bank["directory"] / "instances" / f"{item['id']}.npz")
                 for bank in self.plan["banks"].values()
                 for item in json.loads((self.root / bank["directory"] / "manifest.json").read_text())["instances"]
                 if item["split"] == "test"}
        real_load, real_item = np.load, np.lib.npyio.NpzFile.__getitem__
        def guard_load(path,*args,**kwargs):
            self.assertNotIn(str(path), tests)
            return real_load(path,*args,**kwargs)
        def guard_item(archive,key):
            self.assertNotIn(key,{"states","states_packed","z_values","star_values"})
            return real_item(archive,key)
        with patch("numpy.load",side_effect=guard_load), patch.object(np.lib.npyio.NpzFile,"__getitem__",guard_item):
            self.prepare()
            load_features(self.output,8)
        def only_cache(path,*args,**kwargs):
            self.assertEqual(Path(path),self.output/"L8/maps.npz")
            return real_load(path,*args,**kwargs)
        with patch("numpy.load",side_effect=only_cache):
            load_features(self.output,8)

    def test_prefix_limits_are_grouped_deterministic_and_immutable(self):
        self.prepare(limits={"train":2,"validation":1})
        a=load_features(self.output,8)
        for split, count in [("train",2),("validation",1)]:
            expected=sorted(k for k,v in self.plan["banks"]["8"]["entries"].items() if v["instance"]["split"]==split)[:count]
            self.assertEqual(a[split]["ids"].tolist(),expected)
        with self.assertRaisesRegex(ValueError,"request changed"):
            self.prepare(limits={"train":3,"validation":1})

    def test_pause_then_resume_committed_sizes_preserves_bytes(self):
        self.prepare(sizes=(3,))
        # Different requests may not share one cache, so use a fresh two-size cache.
        self.output=self.output.parent/"resumable"
        calls=0
        def stop():
            nonlocal calls
            calls+=1
            return calls==9  # Commit L3 (7 checks), enter L8 then pause.
        self.prepare(stop_requested=stop)
        self.assertTrue((self.output/"L3/manifest.json").is_file())
        before={p.name:sha(p) for p in (self.output/"L3").iterdir()}
        self.assertFalse((self.output/"L8/manifest.json").exists())
        self.prepare()
        self.assertEqual(before,{p.name:sha(p) for p in (self.output/"L3").iterdir()})
        self.assertEqual(json.loads((self.output/"status.json").read_text())["state"],"complete")

    def test_checksum_coordinate_identity_and_source_pin_corruption_refused(self):
        job=self.run/"jobs/saved/L8-hamiltonian_00000"
        with (job/"coordinates.npz").open("ab") as f: f.write(b"tamper")
        with self.assertRaisesRegex(ValueError,"coordinate checksum"):
            self.prepare()
        # Fresh fixture output request stays bound; changing a source is refused
        # before resuming or opening another payload.
        (self.root/"scientific.py").write_text("changed source")
        with self.assertRaisesRegex(ValueError,"source pin changed"):
            self.prepare()

    def test_corrupt_cache_and_split_uid_rejected(self):
        self.prepare()
        with (self.output/"L8/maps.npz").open("ab") as f: f.write(b"tamper")
        with self.assertRaisesRegex(ValueError,"cache checksum"):
            load_features(self.output,8)

    def test_identity_summary_is_rejected(self):
        task=next(t for t in self.plan["tasks"] if t["L"]==3)
        job=self.run/"jobs/saved"/task["key"]
        summary=json.loads((job/"summary.json").read_text())
        summary["id"]="hamiltonian_99999"
        write(job/"summary.json",summary)
        done=json.loads((job/"done.json").read_text())
        done["files"]["summary.json"]=sha(job/"summary.json")
        write(job/"done.json",done)
        with self.assertRaisesRegex(ValueError,"summary identity"):
            self.prepare()

    def test_failed_exact_reference_is_rejected(self):
        task=next(t for t in self.plan["tasks"] if t["L"]==3)
        job=self.run/"jobs/saved"/task["key"]
        summary=json.loads((job/"summary.json").read_text())
        summary["max_identity_error"]=.01
        write(job/"summary.json",summary)
        done=json.loads((job/"done.json").read_text())
        done["files"]["summary.json"]=sha(job/"summary.json")
        write(job/"done.json",done)
        with self.assertRaisesRegex(ValueError,"exact identity check"):
            self.prepare()

    def test_changed_bank_membership_and_output_escape_rejected(self):
        bank=self.bank_root/"L8/run"
        manifest=json.loads((bank/"manifest.json").read_text())
        manifest["instances"][0]["split"]="test"
        write(bank/"manifest.json",manifest)
        with self.assertRaisesRegex(ValueError,"seeded instance"):
            self.prepare()
        with self.assertRaisesRegex(ValueError,"final_benchmark"):
            prepare_features(self.root,self.bank_root,self.run,self.root/"unsafe",[3])


if __name__ == "__main__":
    unittest.main()
