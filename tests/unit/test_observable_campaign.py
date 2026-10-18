"""Synthetic CPU controls for durable campaigns, never research training."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from quantum_lattice.data.hamiltonian_scores import FieldReference
from quantum_lattice.data.mlp_data import LearningData
from quantum_lattice.data.observable_maps import ObservableMaps, maps_to_edge_rows
from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian import hamiltonian_error
from quantum_lattice.models import observable_campaign as campaign
from quantum_lattice.models import observable_training


def settings(name="tiny"):
    return {"schema_version": 1, "sizes": [4, 3], "seeds": [41],
            "models": ["cnn", "map_mlp"], "views": ["f", "g"],
            "feature_cache": "codex material/artifacts/synthetic/features",
            "output_root": f"codex material/artifacts/observable_cnn/campaigns/{name}",
            "architecture": {"hidden_channels": [2], "kernel_size": 3},
            "training": {"device": "cuda", "deterministic": True, "torch_threads": 1,
                         "learning_rate": 1e-3, "batch_size_hamiltonians": 3,
                         "max_epochs": 2, "validation_batch_size": 3,
                         "early_stopping": {"enabled": False, "min_epochs": 1,
                                            "patience_epochs": 2, "min_delta": 1e-6}},
            "execution": {"checkpoint_updates": 1},
            "provenance": {"synthetic": True}}


def synthetic_split(L, split):
    count, offset = (5, 0) if split == "train" else (2, 1000)
    rng = np.random.default_rng(L + offset)
    fields = rng.uniform(-.6, .6, (count, 2 * L * L))
    inputs = rng.uniform(.3, 1.1, (count, 3, L, L)).astype(np.float32)
    target = np.abs(fields).astype(np.float32).reshape(count, L, L, 2).transpose(0, 3, 1, 2)
    names = np.array([f"{split}{n}" for n in range(count)])
    uids = np.array([f"L{L}-{name}" for name in names])
    maps = ObservableMaps(inputs, target, names, uids,
                          np.arange(count * 2 * L * L).reshape(count, -1), L)
    return maps, fields


def synthetic_views(root, L):
    result = {}
    for view, offset in (("f", 0), ("g", .02), ("exact", .01)):
        splits = []
        for split in ("train", "validation"):
            maps, _ = synthetic_split(L, split)
            variant = ObservableMaps(maps.x + np.float32(offset), maps.y, maps.hamiltonian_ids,
                                     maps.hamiltonian_uids, maps.source_row_indices, L)
            splits.append(maps_to_edge_rows(variant))
        result[view] = LearningData(*splits, None, L, 0, "all_edges",
                                   {"training_eligible": view != "exact"}, Path(root))
    return result


def synthetic_reference(root, L):
    maps, fields = synthetic_split(L, "validation")
    return FieldReference(L, maps.hamiltonian_ids, maps.hamiltonian_uids, fields,
                          maps_to_edge_rows(maps), "synthetic")


def equal_tree(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys())
        for key in left:
            equal_tree(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            equal_tree(case, a, b)
    else:
        case.assertEqual(left, right)


class ObservableCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        cache = self.root / settings()["feature_cache"]
        cache.mkdir(parents=True)
        (cache / "manifest.json").write_text('{"synthetic":true}')
        (cache / "request.json").write_text('{"synthetic":true}')
        for L in (3, 4):
            folder = cache / f"L{L}"
            folder.mkdir()
            (folder / "manifest.json").write_text('{"synthetic":true}')
            (folder / "views.npz").write_bytes(b"synthetic cache; loader explicitly substituted")
        self.patch(campaign, "_sources", lambda root: {"synthetic-source": "fixed"})
        self.patch(campaign, "load_feature_views", synthetic_views)
        self.patch(campaign, "load_validation_fields", synthetic_reference)
        # Public config still requires CUDA. Only these explicit test seams run CPU.
        self.patch(campaign, "resolve_device", lambda *args: torch.device("cpu"))
        self.patch(observable_training, "resolve_device", lambda *args: torch.device("cpu"))

    def patch(self, obj, name, value):
        active = patch.object(obj, name, value)
        active.start()
        self.addCleanup(active.stop)

    def prepare(self, name="tiny", config=None):
        return campaign.prepare(self.root, config or settings(name))

    def run_all(self, run, **kwargs):
        return campaign.run_campaign(self.root, run, output=lambda message: None, **kwargs)

    def test_prepare_freezes_all_arms_and_makes_zero_updates(self):
        run = self.prepare()
        plan = campaign.load_plan(self.root, run)
        self.assertEqual(len(plan["tasks"]), 8)
        self.assertEqual([(t["L"], t["model"], t["view"]) for t in plan["tasks"]],
                         [(L, model, view) for L in (4, 3) for model in ("cnn", "map_mlp")
                          for view in ("f", "g")])
        self.assertEqual(plan["data_bindings"]["4"]["train"]["hamiltonians"], 5)
        self.assertFalse((run / "models").exists())
        summary = campaign.summarize_campaign(self.root, run)
        self.assertEqual((summary["completed"], summary["state"]), (0, "pending"))
        self.assertIn("0/1 complete four-arm seed blocks", campaign.format_summary(summary))

    def test_strict_config_and_safe_output(self):
        mutations = [("extra", 1), ("sizes", [3, 4]), ("models", ["cnn"]),
                     ("views", ["f", "exact"]), ("seeds", [True]), ("seeds", [41, 41])]
        for key, value in mutations:
            config = settings()
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                campaign.validate_config(config)
        config = settings()
        config["training"]["device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "cuda"):
            campaign.validate_config(config)
        config = settings()
        config["training"]["unknown"] = True
        with self.assertRaises(ValueError):
            campaign.validate_config(config)
        with self.assertRaisesRegex(ValueError, "below"):
            campaign.prepare(self.root, settings(), run_dir=self.root / "outside")
        config = settings()
        config["feature_cache"] = "../../outside"
        with self.assertRaisesRegex(ValueError, "artifact"):
            campaign.prepare(self.root, config)

    def test_scientific_source_environment_plan_and_cache_guards(self):
        run = self.prepare()
        with patch.object(campaign, "_sources", lambda root: {"changed": True}):
            with self.assertRaisesRegex(ValueError, "source"):
                campaign.load_plan(self.root, run)
        with patch.object(campaign, "_environment", lambda: {"changed": True}):
            with self.assertRaisesRegex(ValueError, "environment"):
                campaign.load_plan(self.root, run)
        cache = self.root / settings()["feature_cache"] / "L3/views.npz"
        previous = cache.read_bytes()
        cache.write_bytes(previous + b"changed")
        with self.assertRaisesRegex(ValueError, "cache"):
            campaign.load_plan(self.root, run)
        cache.write_bytes(previous)
        plan = json.loads((run / "plan.json").read_text())
        plan["tasks"].pop()
        (run / "plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "hash"):
            campaign.load_plan(self.root, run)

    def test_exact_and_test_eligibility_fail_before_training(self):
        original = synthetic_views
        def bad(root, L):
            result = original(root, L)
            result["exact"].provenance["training_eligible"] = True
            return result
        with patch.object(campaign, "load_feature_views", bad):
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                self.prepare()
        def loaded_test(root, L):
            from dataclasses import replace
            result = original(root, L)
            result["f"] = replace(result["f"], test=result["f"].validation)
            return result
        with patch.object(campaign, "load_feature_views", loaded_test):
            with self.assertRaisesRegex(ValueError, "test split"):
                self.prepare("test")

    def test_preflight_cuda_failure_precedes_feature_payload(self):
        run = self.prepare()
        with patch.object(campaign, "resolve_device", side_effect=ValueError("CUDA unavailable")):
            with patch.object(campaign, "load_feature_views", side_effect=AssertionError("payload accessed")):
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    self.run_all(run)

    def test_partial_epoch_pause_resume_is_bit_identical(self):
        full, paused = self.prepare("full"), self.prepare("paused")
        self.run_all(full)
        summary = self.run_all(paused, max_chunks=1)
        self.assertEqual(summary["tasks"][0]["step"], 1)
        self.assertEqual(summary["tasks"][0]["epoch"], 0)
        self.assertEqual(summary["tasks"][0]["state"], "paused")
        self.run_all(paused)
        for task in campaign.load_plan(self.root, full)["tasks"]:
            a = torch.load(full / "models" / task["key"] / "latest.pt", weights_only=True)
            b = torch.load(paused / "models" / task["key"] / "latest.pt", weights_only=True)
            equal_tree(self, a["session"], b["session"])
        final = campaign.summarize_campaign(self.root, paused)
        self.assertEqual((final["completed"], final["state"]), (8, "complete"))

    def test_size_filter_and_completed_outputs_skip_immutably(self):
        run = self.prepare()
        summary = self.run_all(run, sizes=[3])
        self.assertEqual(summary["completed"], 4)
        self.assertTrue(all(row["state"] == "pending" for row in summary["tasks"] if row["L"] == 4))
        task = next(row for row in summary["tasks"] if row["L"] == 3)
        folder = run / "models" / task["key"]
        original = {path.name: path.read_bytes() for path in folder.iterdir()}
        self.run_all(run, sizes=[3])
        self.assertEqual(original, {path.name: path.read_bytes() for path in folder.iterdir()})
        (folder / "predictions.npz").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "output changed"):
            campaign.summarize_campaign(self.root, run)

    def test_crossed_prediction_rows_and_d5_match_scalar_oracle(self):
        run = self.prepare()
        seen = []
        actual = campaign.ObservableTrainSession
        def record(train_x, train_y, val_x, val_y, config, **kwargs):
            seen.append((config["model"], config["seed"], train_x.copy()))
            return actual(train_x, train_y, val_x, val_y, config, **kwargs)
        with patch.object(campaign, "ObservableTrainSession", record):
            self.run_all(run, sizes=[4])
        tasks = [t for t in campaign.load_plan(self.root, run)["tasks"] if t["L"] == 4]
        for task, (_, _, train_x) in zip(tasks, seen):
            maps = campaign._load_maps(self.root / settings()["feature_cache"], 4)
            self.assertTrue(np.array_equal(train_x, maps[task["view"]].train.x))
            self.assertFalse(np.array_equal(train_x, maps["exact"].train.x))
            job = run / "models" / task["key"]
            evaluation = json.loads((job / "evaluation.json").read_text())
            self.assertEqual(set(evaluation["scores"]), set(campaign.VIEWS))
            self.assertFalse(evaluation["test_payload_loaded"])
            self.assertEqual(evaluation["hamiltonian_metric"]["sign_policy"], "reference_sign_oracle")
            reference = synthetic_reference(None, 4)
            with np.load(job / "predictions.npz", allow_pickle=False) as archive:
                self.assertTrue(np.array_equal(archive["target"], reference.validation_rows.y))
                prediction = archive["prediction_f"].reshape(2, 32)
                expected = []
                for true, raw in zip(reference.true_b_z, prediction):
                    expected.append(hamiltonian_error(square_lattice(4), true,
                        np.sign(true) * raw, model="two_channel", include_identity=True))
                self.assertAlmostEqual(evaluation["scores"]["f"]["d5"]["two_channel_full"],
                                       float(np.mean(expected)), places=13)
            with (job / "hamiltonian-scores.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)
        with (run / "paired-validation.csv").open() as handle:
            pairs = list(csv.DictReader(handle))
        self.assertEqual(len(pairs), 8)
        self.assertEqual({row["metric"] for row in pairs}, {"mse", "d5_two_channel_full"})
        for row in pairs:
            self.assertAlmostEqual(float(row["interaction"]),
                float(row["g_minus_f_cnn"]) - float(row["g_minus_f_mlp"]), places=14)
        self.assertIn("All-seed primary reference-sign D5", campaign.format_summary(
            campaign.summarize_campaign(self.root, run)))

    def test_cache_mutation_during_load_is_rejected_before_any_update(self):
        run = self.prepare()
        original = campaign.load_validation_fields
        def mutate(root, L):
            result = original(root, L)
            path = Path(root) / f"L{L}/views.npz"
            path.write_bytes(path.read_bytes() + b"changed during load")
            return result
        with patch.object(campaign, "load_validation_fields", mutate):
            with self.assertRaisesRegex(ValueError, "changed during lattice loading"):
                self.run_all(run)
        self.assertFalse((run / "models").exists())

    def test_epoch_zero_pause_is_reported_as_paused(self):
        run = self.prepare()
        actual = campaign.ObservableTrainSession
        def pause_during_initialization(*args, **kwargs):
            result = actual(*args, **kwargs)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return result
        with patch.object(campaign, "ObservableTrainSession", pause_during_initialization):
            summary = self.run_all(run)
        self.assertEqual(summary["state"], "paused")
        self.assertEqual(summary["tasks"][0]["step"], 0)
        self.assertEqual(summary["tasks"][0]["epoch"], 0)

    def test_sigint_finishes_chunk_then_saves_and_restores_handler(self):
        run = self.prepare()
        previous = signal.getsignal(signal.SIGINT)
        original = campaign.ObservableTrainSession.advance
        def pause_after_update(session, count):
            result = original(session, count)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return result
        with patch.object(campaign.ObservableTrainSession, "advance", pause_after_update):
            summary = self.run_all(run)
        self.assertEqual(summary["tasks"][0]["step"], 1)
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)
        payload = torch.load(run / "models" / summary["tasks"][0]["key"] / "latest.pt", weights_only=True)
        self.assertEqual(payload["session"]["cursor"], 3)
        self.run_all(run, sizes=[4])

    def test_checkpoint_identity_and_device_corruption_are_rejected(self):
        run = self.prepare()
        summary = self.run_all(run, max_chunks=1)
        path = run / "models" / summary["tasks"][0]["key"] / "latest.pt"
        payload = torch.load(path, weights_only=True)
        payload["task"]["view"] = "exact"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_all(run)
        with patch.object(campaign, "_accelerator", lambda device: {"type": "different"}):
            with self.assertRaisesRegex(ValueError, "Accelerator changed"):
                self.run_all(run)

    def test_partial_seed_blocks_are_not_final_average(self):
        config = settings()
        config["seeds"] = [41, 42]
        run = self.prepare(config=config)
        self.run_all(run, sizes=[4], max_chunks=16)
        summary = campaign.summarize_campaign(self.root, run)
        text = campaign.format_summary(summary)
        self.assertIn("1/2 complete four-arm seed blocks", text)
        self.assertIn("all-seed comparison waits", text)
        self.assertNotIn("All-seed matched-input validation MSE", text)


if __name__ == "__main__":
    unittest.main()
