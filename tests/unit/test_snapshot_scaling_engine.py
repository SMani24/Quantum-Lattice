"""Synthetic controls for nested bags, immutable reuse and exact queue resume."""
import copy
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from quantum_lattice.data.state_storage import encode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.models import snapshot_campaign as pilot
from quantum_lattice.models import snapshot_scaling as scaling


REPO = Path(__file__).resolve().parents[2]


def fixture(root):
    """Three synthetic training H, two validation H and a completed tiny reference.

    The old inverse solver is run on synthetic data only; no official cache,
    model, source, configuration or physical experiment output is changed.
    """
    root = Path(root)
    for name in ("src", "experiments", "configs"):
        (root / name).symlink_to(REPO / name, target_is_directory=True)
    bank = root / "codex material/bank"
    (bank / "instances").mkdir(parents=True)
    feature = root / "codex material/artifacts/fixture-features/L4"
    feature.mkdir(parents=True)
    lattice = square_lattice(4)
    ends = lattice.edge_vertex_indices
    random = np.random.default_rng(799)
    instances, origins = [], {}
    selected = {"train": [], "validation": []}
    for index, split in enumerate(["train"] * 3 + ["validation"] * 2):
        name = f"h{index}"
        instance = {"id": name, "split": split, "hamiltonian_uid": f"uid-{index}", "field_seed": [index]}
        tau = random.choice(np.array([-1, 1], np.int8), (8, 16))
        states = tau[:, ends[:, 0]] * tau[:, ends[:, 1]]
        fields = random.uniform(-0.7, 0.7, 32)
        metadata = {"instance": instance, "retained_records": 8, "chain": {"fields": fields.tolist()}}
        path = bank / "instances" / f"{name}.npz"
        np.savez(path, **encode_states(states, "packed_bits"), b_z=fields,
                 edge_vertex_indices=ends, metadata_json=np.asarray(json.dumps(metadata)))
        path.with_suffix(".json").write_text(json.dumps({"id": name, "split": split, "records": 8}))
        instances.append(instance)
        selected[split].append(name)
        origins[name] = {"instance": instance, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    instances.append({"id": "test-no-payload", "split": "test", "hamiltonian_uid": "unopened"})
    (bank / "manifest.json").write_text(json.dumps({"instances": instances}))
    (feature / "manifest.json").write_text(json.dumps({"L": 4, "original_bank": str(bank.relative_to(root)),
                                                      "selected_ids": selected, "origins": origins}))
    old = json.loads((REPO / "configs/snapshot_learning.json").read_text())
    old["feature_cache"] = str(feature.parent.relative_to(root))
    old["output_root"] = "codex material/artifacts/snapshot_learning/campaigns/synthetic-reference"
    old["selection"] = {"selection_seed": 813, "train_hamiltonians": 2,
                        "validation_hamiltonians": 2, "shot_counts": [4]}
    old["seeds"] = [617]
    old["training"].update(batch_size_hamiltonians=2, max_epochs=2,
                            validation_batch_size=1, shot_chunk_size=2)
    old["training"]["early_stopping"].update(min_epochs=1, patience_epochs=1)
    old["inverse_ising"].update(max_iterations=3)
    old["execution"].update(checkpoint_updates=1, inverse_workers=1)
    reference_run = pilot.prepare(root, old)
    reference_plan = pilot.load_plan(root, reference_run)
    inverse_task = next(task for task in reference_plan["tasks"] if task["method"] == "inverse_ising")
    pilot._run_inverse(reference_run, reference_plan, pilot.load_snapshot_cache(reference_run / "cache"),
                       inverse_task, lambda: False, None, 1)
    config = {key: copy.deepcopy(value) for key, value in old.items() if key != "execution"}
    config["output_root"] = "codex material/artifacts/snapshot_learning/scaling/synthetic"
    config["selection"] = {"selection_seed": 813, "train_hamiltonian_counts": [2, 3],
                           "validation_hamiltonians": 2, "shots": 4}
    config["execution"] = {"checkpoint_updates": 1}
    config["inverse_reference"] = {"campaign_root": old["output_root"], "task_key": inverse_task["key"]}
    return config


def _synthetic_binding(run):
    binding = {"schema_version": 1, "accelerator": {"type": "cpu-synthetic"}}
    path = run / "device-binding.json"
    if path.exists() and scaling._read(path) != binding:
        raise ValueError("Synthetic device changed")
    if not path.exists():
        scaling._json(path, binding)
    return binding


def run_cpu(root, config, **kwargs):
    with patch("quantum_lattice.models.snapshot_training.resolve_device", return_value=torch.device("cpu")), \
            patch.object(pilot, "_gpu_binding", side_effect=_synthetic_binding):
        return scaling.run_campaign(root, config, **kwargs)


class SnapshotScalingTests(unittest.TestCase):
    def test_prepare_is_nested_read_only_and_reuses_exact_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            self.assertEqual(scaling.progress(root, config)["state"], "not_prepared")
            run = scaling.prepare(root, config)
            plan = scaling.load_plan(root, run)
            self.assertEqual([task["train_hamiltonians"] for task in plan["tasks"]], [2, 2, 3, 3])
            self.assertEqual(plan["updates_per_epoch"], {"2": 1, "3": 2})
            self.assertEqual(plan["max_optimizer_updates"], {"2": 2, "3": 4})
            self.assertFalse((run / "models").exists())
            self.assertFalse((run / "device-binding.json").exists())
            cache = pilot.load_snapshot_cache(run / "cache")
            small = scaling.prefix_cache(cache, 2)
            self.assertTrue(np.shares_memory(small.train.states, cache.train.states))
            self.assertIs(small.validation, cache.validation)
            np.testing.assert_array_equal(small.train.states, cache.train.states[:2])
            self.assertFalse(small.train.states.flags.writeable)
            with patch.object(scaling, "_json", side_effect=AssertionError("Status must be read-only")):
                status = scaling.progress(root, config)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["inverse_reference"]["state"], "complete")
            self.assertTrue(status["inverse_reference"]["reused_without_refitting"])
            self.assertFalse((root / "codex material/bank/instances/test-no-payload.npz").exists())

    def test_queue_matches_records_and_never_refits_inverse_or_rewrites_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            with patch.object(pilot, "_run_inverse", side_effect=AssertionError("No inverse refits")):
                status = run_cpu(root, config)
            self.assertEqual(status["completed"], 4)
            run = root / config["output_root"]
            cache = pilot.load_snapshot_cache(run / "cache")
            for task in scaling.load_plan(root, run)["tasks"]:
                path = run / "models" / task["key"]
                evaluation = scaling._read(path / "evaluation.json")
                self.assertEqual(evaluation["train_hamiltonians"], task["train_hamiltonians"])
                with np.load(path / "predictions.npz") as archive:
                    np.testing.assert_array_equal(archive["record_indices"], cache.validation.record_indices)
                    np.testing.assert_array_equal(archive["hamiltonian_uids"], cache.validation.uids)
            before = {str(path): scaling._sha(path) for path in (run / "models").rglob("*") if path.is_file()}
            run_cpu(root, config)
            after = {str(path): scaling._sha(path) for path in (run / "models").rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            self.assertEqual((run / "comparison.csv").read_text().count("immutable prior fit reused once"), 1)

    def test_partial_epoch_resume_is_identical_to_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            def pause(event):
                if event.get("stage") == "training":
                    signal.raise_signal(signal.SIGINT)
            status = run_cpu(root, config, progress_callback=pause)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["tasks"][0]["state"], "paused")
            self.assertEqual(status["tasks"][0]["step"], 1)
            self.assertEqual(run_cpu(root, config)["completed"], 4)
            other = copy.deepcopy(config)
            other["output_root"] += "-uninterrupted"
            self.assertEqual(run_cpu(root, other)["completed"], 4)
            for task in scaling.load_plan(root, root / config["output_root"])["tasks"]:
                with np.load(root / config["output_root"] / "models" / task["key"] / "predictions.npz") as resumed, \
                        np.load(root / other["output_root"] / "models" / task["key"] / "predictions.npz") as whole:
                    np.testing.assert_array_equal(resumed["prediction"], whole["prediction"])

    def test_corruption_and_changed_science_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            run = scaling.prepare(root, config)
            changed = copy.deepcopy(config)
            changed["seeds"] = [618]
            with self.assertRaisesRegex(ValueError, "differs"):
                scaling.prepare(root, changed)
            reference = root / config["inverse_reference"]["campaign_root"] / "models" / config["inverse_reference"]["task_key"]
            for path in (reference / "predictions.npz", reference / "instances/00000.json", run / "cache/train.npz"):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.write_bytes(original + b"corrupted")
                    with self.assertRaises(ValueError):
                        scaling.progress(root, config)
                    path.write_bytes(original)
            def pause(event):
                if event.get("stage") == "training":
                    signal.raise_signal(signal.SIGINT)
            run_cpu(root, config, progress_callback=pause)
            scaling._json(run / "device-binding.json", {"accelerator": "changed"})
            with self.assertRaisesRegex(ValueError, "device binding"):
                scaling.progress(root, config)

    def test_bad_config_and_mismatched_reference_records_are_rejected(self):
        config = json.loads((REPO / "configs/snapshot_scaling.json").read_text())
        for update in (lambda value: value["selection"].update(train_hamiltonian_counts=[512, 512]),
                       lambda value: value["selection"].update(train_hamiltonian_counts=[True]),
                       lambda value: value["training"].update(device="cpu"),
                       lambda value: value.update(output_root="../escape")):
            changed = copy.deepcopy(config)
            update(changed)
            with self.assertRaises(ValueError):
                scaling.validate_config(changed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            config["selection"]["shots"] = 3
            with self.assertRaisesRegex(ValueError, "record budget"):
                scaling.prepare(root, config)


if __name__ == "__main__":
    unittest.main()
