"""Small end-to-end checks of durable, Hamiltonian-grouped generation."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quantum_lattice.data.generation import _advance_job, _bank, artifact_root, prepare_run, run_generation, verify_run
from quantum_lattice.lattice import square_lattice


REPO = Path(__file__).resolve().parents[2]


def tiny_config(chunk_sweeps=17):
    """Self-contained test protocol; deliberately misalign chunks and stride."""
    return {
        "schema_version": 1,
        "lattice": {"type": "square_periodic", "L": 3},
        "fields": {"channel": "z_only", "distribution": "uniform", "low": -0.2, "high": 0.2},
        "dataset": {"hamiltonians": 4, "seed": 15,
                    "split_fractions": {"train": 0.5, "validation": 0.25, "test": 0.25}},
        "sampler": {"total_sweeps": 80, "burn_in_sweeps": 20, "thinning_sweeps": 5,
                    "lazy_probability": 0.05, "backend": "python"},
        "statistics": {"block_size_records": 4},
        "execution": {"workers": 2, "chunk_sweeps": chunk_sweeps},
    }


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GenerationIntegrationTests(unittest.TestCase):
    def setUp(self):
        root = artifact_root(REPO)
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="test-generation-", dir=root)
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_pause_resume_matches_other_chunks_and_workers_exactly(self):
        paused_run = prepare_run(REPO, tiny_config(), self.base / "paused")
        paused = run_generation(REPO, paused_run, workers=1, interactive=False, max_chunks=2)
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(paused["completed"], 0)
        self.assertEqual(paused["attempts_this_session"], 2 * 17 * 9)
        checkpoints = sorted((paused_run / "checkpoints").glob("*.npz"))
        self.assertEqual(len(checkpoints), 1)
        for checkpoint in checkpoints:
            arrays = load_arrays(checkpoint)
            self.assertEqual(arrays["states"].shape, (2, 18))
            metadata = json.loads(arrays["metadata_json"].item())
            self.assertEqual(metadata["completed_sweeps"], 34)
            self.assertEqual(metadata["chain"]["attempted"], 34 * 9)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify_run(paused_run)
        resumed = run_generation(REPO, paused_run, workers=2, interactive=False)
        self.assertEqual(resumed["state"], "complete")
        self.assertEqual(resumed["completed"], 4)
        self.assertFalse(list((paused_run / "checkpoints").glob("*.npz")))

        direct_run = prepare_run(REPO, tiny_config(chunk_sweeps=19), self.base / "direct")
        direct = run_generation(REPO, direct_run, workers=1, interactive=False)
        self.assertEqual(direct["state"], "complete")
        paused_manifest = json.loads((paused_run / "manifest.json").read_text())
        direct_manifest = json.loads((direct_run / "manifest.json").read_text())
        self.assertEqual(paused_manifest["instances"], direct_manifest["instances"])
        for instance in paused_manifest["instances"]:
            name = instance["id"] + ".npz"
            resumed_arrays = load_arrays(paused_run / "instances" / name)
            direct_arrays = load_arrays(direct_run / "instances" / name)
            self.assertEqual(set(resumed_arrays), set(direct_arrays))
            for key in resumed_arrays:
                if key in {"metadata_json", "summary_json"}:
                    left = json.loads(resumed_arrays[key].item())
                    right = json.loads(direct_arrays[key].item())
                    left.pop("compute_seconds")
                    right.pop("compute_seconds")
                    self.assertEqual(left, right, key)
                else:
                    np.testing.assert_array_equal(resumed_arrays[key], direct_arrays[key], err_msg=key)
        self.assertTrue(verify_run(paused_run)["passed"])
        self.assertTrue(verify_run(direct_run)["passed"])

    def test_saved_dataset_group_identity_and_physics_schema(self):
        run_dir = prepare_run(REPO, tiny_config(), self.base / "schema")
        run_generation(REPO, run_dir, workers=2, interactive=False)
        dataset = json.loads((run_dir / "dataset.json").read_text())
        self.assertEqual(dataset["split_counts"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(dataset["local_rows"], 4 * 18)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        split_ids = {}
        lattice = square_lattice(3)
        all_ids = set()
        for instance in manifest["instances"]:
            instance_id, split = instance["id"], instance["split"]
            self.assertNotIn(instance_id, all_ids)
            all_ids.add(instance_id)
            split_ids.setdefault(split, set()).add(instance_id)
            path = run_dir / "instances" / (instance_id + ".npz")
            arrays = load_arrays(path)
            self.assertTrue(all(array.dtype.kind != "O" for array in arrays.values()))
            metadata = json.loads(arrays["metadata_json"].item())
            summary = json.loads(arrays["summary_json"].item())
            self.assertEqual(metadata["instance"], instance)
            self.assertEqual(summary["split"], split)
            self.assertEqual(summary["id"], instance_id)
            self.assertEqual(summary["attempted"], 80 * 9)
            self.assertEqual(summary["records"], 12)
            self.assertEqual(summary["local_rows"], 18)
            self.assertEqual(json.loads(path.with_suffix(".json").read_text()), summary)
            self.assertEqual(arrays["states"].shape, (12, 18))
            self.assertEqual(arrays["star_values"].shape, (12, 9))
            self.assertEqual(arrays["pair_values"].shape, (12, 18))
            self.assertEqual(arrays["mlp_X"].shape, (18, 3))
            self.assertTrue(np.all(np.isin(arrays["states"], [-1, 1])))
            self.assertFalse(lattice.plaquette_syndrome(arrays["states"] == -1).any())
            for loop in (2 * np.arange(3), 6 * np.arange(3) + 1):
                np.testing.assert_array_equal(np.prod(arrays["states"][:, loop], axis=1), np.ones(12))
            np.testing.assert_array_equal(arrays["edge_vertex_indices"], lattice.edge_vertex_indices)
            np.testing.assert_array_equal(arrays["target_b"], arrays["b_z"])
            np.testing.assert_array_equal(arrays["target_abs_b"], np.abs(arrays["b_z"]))
            ends = lattice.edge_vertex_indices
            np.testing.assert_array_equal(arrays["mlp_X"][:, 0], arrays["star_mean"][ends[:, 0]])
            np.testing.assert_array_equal(arrays["mlp_X"][:, 1], arrays["star_mean"][ends[:, 1]])
            np.testing.assert_array_equal(arrays["mlp_X"][:, 2], arrays["pair_mean"])
        self.assertEqual(Counter(instance["split"] for instance in manifest["instances"]), Counter(train=2, validation=1, test=1))
        self.assertFalse(split_ids["train"] & split_ids["validation"])
        self.assertFalse(split_ids["train"] & split_ids["test"])
        self.assertFalse(split_ids["validation"] & split_ids["test"])

    def test_completed_instances_are_not_overwritten(self):
        run_dir = prepare_run(REPO, tiny_config(), self.base / "unchanged")
        run_generation(REPO, run_dir, workers=1, interactive=False)
        paths = sorted((run_dir / "instances").glob("*.npz"))
        before = {path.name: (digest(path), path.stat().st_mtime_ns) for path in paths}
        result = run_generation(REPO, run_dir, workers=2, interactive=False)
        self.assertEqual(result["attempts_this_session"], 0)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(before, {path.name: (digest(path), path.stat().st_mtime_ns) for path in paths})
        with self.assertRaises(FileExistsError):
            prepare_run(REPO, tiny_config(), run_dir)

    def test_production_records_survive_a_second_pause(self):
        run_dir = prepare_run(REPO, tiny_config(), self.base / "production-pause")
        result = run_generation(REPO, run_dir, workers=1, interactive=False, max_chunks=7)
        self.assertEqual(result["state"], "paused")
        before = {}
        for path in (run_dir / "checkpoints").glob("*.npz"):
            arrays = load_arrays(path)
            if len(arrays["states"]):
                before[path.name] = arrays["states"].copy()
        self.assertEqual(len(before), 1)
        self.assertTrue(all(states.shape == (2, 18) for states in before.values()))
        self.assertEqual(run_generation(REPO, run_dir, workers=2, interactive=False)["state"], "complete")
        for name, first_records in before.items():
            final = load_arrays(run_dir / "instances" / name)
            np.testing.assert_array_equal(final["states"][:len(first_records)], first_records)
        self.assertTrue(verify_run(run_dir)["passed"])

    def test_changed_scientific_config_and_unsafe_output_are_rejected(self):
        run_dir = prepare_run(REPO, tiny_config(), self.base / "guarded")
        mutated = deepcopy(tiny_config())
        mutated["fields"]["high"] = 0.19
        (run_dir / "config.json").write_text(json.dumps(mutated))
        with self.assertRaisesRegex(ValueError, "configuration has changed"):
            run_generation(REPO, run_dir, interactive=False)
        with self.assertRaisesRegex(ValueError, "inside"):
            prepare_run(REPO, tiny_config(), REPO / "tests/forbidden-output")
        self.assertFalse((REPO / "tests/forbidden-output").exists())
        with self.assertRaisesRegex(ValueError, "inside"):
            prepare_run(REPO, tiny_config(), artifact_root(REPO))

    def test_changed_source_fingerprint_is_rejected_without_touching_sources(self):
        run_dir = prepare_run(REPO, tiny_config(), self.base / "source-guard")
        path = run_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        self.assertTrue(manifest["source_hashes"])
        key = next(iter(manifest["source_hashes"]))
        manifest["source_hashes"][key] = "0" * 64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "source changed"):
            run_generation(REPO, run_dir, interactive=False)

    def test_edited_manifest_split_is_rejected_against_deterministic_bank(self):
        run_dir = prepare_run(REPO, tiny_config(), self.base / "split-guard")
        path = run_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        instance = manifest["instances"][0]
        instance["split"] = "test" if instance["split"] != "test" else "train"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "deterministic seed/group plan"):
            run_generation(REPO, run_dir, interactive=False)

    def test_numpy_version_change_is_rejected_before_resume(self):
        run_dir = prepare_run(REPO, tiny_config(), self.base / "numpy-guard")
        path = run_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["environment"]["packages"]["numpy"] = "0.0.0-test"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "NumPy version changed"):
            run_generation(REPO, run_dir, interactive=False)

    def test_corrupted_checkpoint_sweep_or_attempt_counts_are_rejected(self):
        config = tiny_config()
        run_dir = prepare_run(REPO, config, self.base / "checkpoint-guard")
        instance = json.loads((run_dir / "manifest.json").read_text())["instances"][0]
        result = _advance_job(str(run_dir), instance, config)
        self.assertFalse(result["done"])
        checkpoint = run_dir / "checkpoints" / (instance["id"] + ".npz")
        original = load_arrays(checkpoint)
        original_metadata = json.loads(original["metadata_json"].item())
        # A checkpoint must be before completion, and its attempted proposals
        # must equal exactly V times the saved number of complete sweeps.
        for corruption in ("completed", "mismatched_sweeps", "mismatched_attempts"):
            with self.subTest(corruption=corruption):
                metadata = deepcopy(original_metadata)
                if corruption == "completed":
                    metadata["completed_sweeps"] = config["sampler"]["total_sweeps"]
                    reason = "sweep count is invalid"
                elif corruption == "mismatched_sweeps":
                    metadata["completed_sweeps"] += 1
                    reason = "attempts do not match completed sweeps"
                else:
                    metadata["chain"]["attempted"] += 9
                    reason = "attempts do not match completed sweeps"
                altered = dict(original, metadata_json=np.array(json.dumps(metadata)))
                with checkpoint.open("wb") as handle:
                    np.savez_compressed(handle, **altered)
                with self.assertRaisesRegex(ValueError, reason):
                    _advance_job(str(run_dir), instance, config)

    def test_final_verification_rejects_edited_configuration_or_group_bank(self):
        config = tiny_config()
        run_dir = prepare_run(REPO, config, self.base / "verification-guard")
        manifest_path, config_path = run_dir / "manifest.json", run_dir / "config.json"
        manifest = json.loads(manifest_path.read_text())
        for instance in manifest["instances"]:
            while not _advance_job(str(run_dir), instance, config)["done"]:
                pass
        self.assertTrue(verify_run(run_dir)["passed"])
        altered_config = deepcopy(config)
        altered_config["dataset"]["seed"] += 1
        config_path.write_text(json.dumps(altered_config))
        with self.assertRaisesRegex(ValueError, "configuration or Hamiltonian seed/group plan"):
            verify_run(run_dir)
        config_path.write_text(json.dumps(config))
        altered_manifest = deepcopy(manifest)
        instance = altered_manifest["instances"][0]
        instance["split"] = "validation" if instance["split"] != "validation" else "train"
        manifest_path.write_text(json.dumps(altered_manifest))
        with self.assertRaisesRegex(ValueError, "configuration or Hamiltonian seed/group plan"):
            verify_run(run_dir)

    def test_bank_extension_preserves_groups_and_physical_identity(self):
        config = tiny_config()
        config["lattice"]["L"] = 4
        config["dataset"].update({"hamiltonians": 16, "seed": 20261013,
                                  "split_fractions": {"train": 0.75, "validation": 0.125, "test": 0.125}})
        first_bank = _bank(config)
        larger_config = deepcopy(config)
        larger_config["dataset"]["hamiltonians"] = 32
        self.assertEqual(first_bank, _bank(larger_config)[:16])
        self.assertEqual(Counter(instance["split"] for instance in first_bank), Counter(train=12, validation=2, test=2))
        original_uids = [instance["hamiltonian_uid"] for instance in first_bank]
        self.assertEqual(len(set(original_uids)), 16)

        changed_sampling = deepcopy(config)
        changed_sampling["sampler"]["total_sweeps"] = 140
        changed_sampling["execution"].update({"workers": 1, "chunk_sweeps": 19})
        self.assertEqual(original_uids, [instance["hamiltonian_uid"] for instance in _bank(changed_sampling)])
        for alteration in ("seed", "domain"):
            with self.subTest(alteration=alteration):
                changed_physics = deepcopy(config)
                if alteration == "seed":
                    changed_physics["dataset"]["seed"] += 1
                else:
                    changed_physics["fields"]["low"] = -0.19
                changed_uids = [instance["hamiltonian_uid"] for instance in _bank(changed_physics)]
                self.assertFalse(set(original_uids) & set(changed_uids))
if __name__ == "__main__":
    unittest.main()
