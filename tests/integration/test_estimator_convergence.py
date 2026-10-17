"""Synthetic controls for forks of completed paired-estimator campaigns.

The explicit CPU substitutions below never load research data. Parent files are
immutable evidence: the extension may only add its own checkpoints and outputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import signal
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from quantum_lattice.data.mlp_data import EdgeRows, LearningData
from quantum_lattice.models import estimator_mlp as base
from quantum_lattice.models import estimator_convergence as convergence
from quantum_lattice.models import mlp_training


def synthetic_views(root, L):
    def rows(split, offset):
        rng = np.random.default_rng(L + offset)
        return EdgeRows(
            rng.uniform(.5, 1, (12, 3)).astype(np.float32),
            rng.uniform(.1, .7, (12, 1)).astype(np.float32),
            np.repeat([f"{split}0", f"{split}1"], 6),
            np.repeat([f"L{L}-{split}0", f"L{L}-{split}1"], 6),
            np.tile(np.arange(6, dtype=np.int64), 2),
        )
    train, validation = rows("train", 0), rows("validation", 100)
    result = {}
    for view, offset in (("f", 0), ("g", .02), ("exact", .01)):
        def variant(row):
            return EdgeRows(row.x + np.float32(offset), row.y,
                            row.hamiltonian_ids, row.hamiltonian_uids, row.edge_ids)
        result[view] = LearningData(variant(train), variant(validation), None,
                                   L, 0, "all_edges", {}, root)
    return result


def file_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class ConvergenceCampaignTests(unittest.TestCase):
    real_cuda = False

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)

        def cache(repo, original, output, sizes):
            output.mkdir(parents=True, exist_ok=True)
            (output / "manifest.json").write_text('{"synthetic":true}')
            return output

        substitutions = [
            (base, "prepare_feature_views", cache),
            (base, "load_feature_views", synthetic_views),
            (base, "_sources", lambda root: {"synthetic-parent": "fixed"}),
            (convergence, "load_feature_views", synthetic_views),
            (convergence, "_sources", lambda root: {"synthetic-extension": "fixed"}),
        ]
        if not self.real_cuda:
            substitutions.extend([
                (base, "resolve_device", lambda *args: torch.device("cpu")),
                (convergence, "resolve_device", lambda *args: torch.device("cpu")),
                (mlp_training, "resolve_device", lambda *args: torch.device("cpu")),
            ])
        for module, name, value in substitutions:
            active = patch.object(module, name, value)
            active.start()
            self.addCleanup(active.stop)
        self.parent_config = {
            "schema_version": 1,
            "output_root": "codex material/artifacts/estimator_mlp/parent",
            "validation_run": "unused", "sizes": [3, 4],
            "mode": "all_edges", "seeds": [41],
            "training": {"device": "cuda", "learning_rate": 1e-3,
                         "batch_size": 4, "max_steps": 6,
                         "validation_interval": 2, "torch_threads": 1,
                         "deterministic": True,
                         "early_stopping": {"enabled": False}},
            "execution": {"checkpoint_steps": 2},
        }
        self.parent = base.prepare(self.workspace, self.parent_config)
        self.config = {
            "schema_version": 1,
            "parent_run": self.parent_config["output_root"],
            "output_root": "codex material/artifacts/estimator_convergence/child",
            "training": {"max_steps": 14,
                         "early_stopping": {"enabled": True, "min_steps": 8,
                                            "patience": 3, "min_delta": 0.0}},
            "execution": {"checkpoint_steps": 2},
        }

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def complete_parent(self):
        base.run_campaign(self.workspace, self.parent, output=lambda _: None)
        return base.load_plan(self.workspace, self.parent)

    def prepare_child(self, name="child"):
        config = copy.deepcopy(self.config)
        config["output_root"] = f"codex material/artifacts/estimator_convergence/{name}"
        return convergence.prepare(self.workspace, config)

    def parent_snapshot(self):
        plan = self.complete_parent()
        task = plan["tasks"][0]
        payload = torch.load(self.parent / "models" / task["key"] / "latest.pt",
                             map_location="cpu", weights_only=True)
        config = copy.deepcopy(payload["session"]["config"])
        config.update(copy.deepcopy(self.config["training"]))
        return task, payload, config

    def test_requires_completed_parent_and_preserves_every_task(self):
        with self.assertRaises(ValueError):
            self.prepare_child()
        parent_plan = self.complete_parent()
        run = self.prepare_child()
        plan = convergence.load_plan(self.workspace, run)
        self.assertEqual(plan["tasks"], parent_plan["tasks"])
        self.assertEqual(plan["config"]["sizes"], [3, 4])
        self.assertEqual(plan["config"]["seeds"], [41])
        self.assertEqual(plan["config"]["training"]["max_steps"], 14)
        self.assertEqual(convergence.summarize_campaign(self.workspace, run)["completed"], 0)

    def test_fork_preserves_latest_model_adam_rng_and_best_without_mutation(self):
        task, payload, config = self.parent_snapshot()
        snapshot = payload["session"]
        # Deliberately distinct selection weights make a rollback to best.pt
        # detectable even when this tiny parent's last step happened to win.
        snapshot["best_state_dict"] = {
            name: value + 1 for name, value in snapshot["best_state_dict"].items()
        }
        before = copy.deepcopy(snapshot)
        fork = convergence._fork_snapshot(snapshot, config)
        self.assert_tree_equal(snapshot, before)
        for key in snapshot:
            if key not in {"config", "stopped_early", "early_stopping_wait",
                           "early_stopping_reference"}:
                self.assert_tree_equal(snapshot[key], fork[key])
        self.assertEqual(fork["config"], mlp_training._config(config))
        self.assertFalse(fork["stopped_early"])
        self.assertEqual(fork["early_stopping_wait"], 0)
        self.assertEqual(fork["early_stopping_reference"], snapshot["best_val_mse"])
        # Mutating the fork must never mutate an in-memory source payload.
        next(iter(fork["model_state_dict"].values())).zero_()
        self.assert_tree_equal(snapshot, before)

    def test_fork_rejects_unrelated_optimizer_and_data_order_changes(self):
        _, payload, config = self.parent_snapshot()
        for key, value in (("learning_rate", .2), ("batch_size", 3),
                           ("seed", 99), ("validation_interval", 1),
                           ("deterministic", False)):
            with self.subTest(key=key):
                changed = copy.deepcopy(config)
                changed[key] = value
                with self.assertRaises(ValueError):
                    convergence._fork_snapshot(payload["session"], changed)

    def test_forked_trajectory_matches_independent_long_budget_training(self):
        task, payload, config = self.parent_snapshot()
        data = synthetic_views(self.parent / "features", task["L"])[task["view"]]
        full = mlp_training.TrainSession(data.train.x, data.train.y,
                                        data.validation.x, data.validation.y, config)
        full.advance(config["max_steps"])
        fork = convergence._fork_snapshot(payload["session"], config)
        continued = mlp_training.TrainSession(data.train.x, data.train.y,
                                             data.validation.x, data.validation.y,
                                             config, checkpoint=fork)
        continued.advance(config["max_steps"])
        self.assert_tree_equal(full.snapshot(), continued.snapshot())

    def test_early_stopped_parent_is_reactivated_with_inherited_best(self):
        task, payload, config = self.parent_snapshot()
        snapshot = payload["session"]
        snapshot["config"]["early_stopping"]["enabled"] = True
        snapshot["stopped_early"] = True
        snapshot["early_stopping_wait"] = 999
        snapshot["early_stopping_reference"] = snapshot["best_val_mse"] + .5
        fork = convergence._fork_snapshot(snapshot, config)
        data = synthetic_views(self.parent / "features", task["L"])[task["view"]]
        session = mlp_training.TrainSession(data.train.x, data.train.y,
                                          data.validation.x, data.validation.y,
                                          config, checkpoint=fork)
        old_step = session.step
        old_best = session.best_state_dict
        old_mse = session.best_val_mse
        with patch.object(session, "_mse", return_value=old_mse + 1):
            session.advance(2)
        self.assertEqual(session.step, old_step + 2)
        self.assertEqual(session.best_val_mse, old_mse)
        self.assert_tree_equal(session.best_state_dict, old_best)

    def test_fork_data_fingerprint_still_rejects_other_estimator(self):
        task, payload, config = self.parent_snapshot()
        fork = convergence._fork_snapshot(payload["session"], config)
        wrong = synthetic_views(self.parent / "features", task["L"])["g"]
        with self.assertRaisesRegex(ValueError, "training/validation data differ"):
            mlp_training.TrainSession(wrong.train.x, wrong.train.y,
                                      wrong.validation.x, wrong.validation.y,
                                      config, checkpoint=fork)

    def test_pause_resume_is_bit_identical_and_parent_files_remain_unchanged(self):
        self.complete_parent()
        before = file_hashes(self.parent)
        full = self.prepare_child("full")
        paused = self.prepare_child("paused")
        convergence.run_campaign(self.workspace, full, output=lambda _: None)
        summary = convergence.run_campaign(self.workspace, paused, max_chunks=1,
                                           output=lambda _: None)
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["tasks"][0]["step"], 8)
        convergence.run_campaign(self.workspace, paused, output=lambda _: None)
        plan = convergence.load_plan(self.workspace, full)
        for task in plan["tasks"]:
            relative = Path("models") / task["key"] / "latest.pt"
            left = torch.load(full / relative, map_location="cpu", weights_only=True)
            right = torch.load(paused / relative, map_location="cpu", weights_only=True)
            self.assert_tree_equal(left["session"], right["session"])
            self.assert_tree_equal(left["history"], right["history"])
            parent_history = json.loads((self.parent / "models" / task["key"] / "history.json").read_text())
            self.assertEqual(left["history"][:len(parent_history)], parent_history)
        self.assertEqual(file_hashes(self.parent), before)
        self.assertEqual(convergence.summarize_campaign(self.workspace, paused)["completed"], 4)

    def test_cuda_failure_occurs_before_loading_feature_payload(self):
        self.complete_parent()
        run = self.prepare_child()
        with patch.object(convergence, "resolve_device", side_effect=ValueError("no CUDA")), \
             patch.object(convergence, "load_feature_views", side_effect=AssertionError("payload accessed")):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                convergence.run_campaign(self.workspace, run)

    def test_changed_accelerator_is_rejected_before_loading_features(self):
        self.complete_parent()
        run = self.prepare_child()
        with patch.object(base, "_accelerator", return_value={"type": "cpu", "name": "different"}), \
             patch.object(convergence, "load_feature_views", side_effect=AssertionError("payload accessed")):
            with self.assertRaisesRegex(ValueError, "Accelerator changed"):
                convergence.run_campaign(self.workspace, run)

    def test_exact_never_trains_and_outputs_have_no_test_payload(self):
        self.complete_parent()
        run = self.prepare_child()
        captured = []
        actual = convergence.TrainSession
        def record(train_x, train_y, val_x, val_y, config, **kwargs):
            captured.append(train_x.copy())
            return actual(train_x, train_y, val_x, val_y, config, **kwargs)
        with patch.object(convergence, "TrainSession", record):
            result = convergence.run_campaign(self.workspace, run, output=lambda _: None)
        self.assertFalse(result["test_payload_loaded"])
        for task, train_x in zip(convergence.load_plan(self.workspace, run)["tasks"], captured):
            views = synthetic_views(self.parent / "features", task["L"])
            np.testing.assert_array_equal(train_x, views[task["view"]].train.x)
            self.assertFalse(np.array_equal(train_x, views["exact"].train.x))
            evaluation = json.loads((run / "models" / task["key"] / "evaluation.json").read_text())
            self.assertFalse(evaluation["test_payload_loaded"])
            self.assertEqual(set(evaluation["scores"]), {"f", "g", "exact"})

    def test_parent_and_child_hash_tampering_is_rejected(self):
        plan = self.complete_parent()
        run = self.prepare_child()
        task = plan["tasks"][0]
        latest = self.parent / "models" / task["key"] / "latest.pt"
        original = latest.read_bytes()
        latest.write_bytes(original + b"changed")
        with self.assertRaises(ValueError):
            convergence.load_plan(self.workspace, run)
        latest.write_bytes(original)
        child = json.loads((run / "plan.json").read_text())
        child["tasks"].reverse()
        (run / "plan.json").write_text(json.dumps(child))
        with self.assertRaises(ValueError):
            convergence.load_plan(self.workspace, run)

    def test_changed_parent_summary_and_extension_sources_are_rejected(self):
        plan = self.complete_parent()
        run = self.prepare_child()
        with patch.object(convergence, "_sources", return_value={"synthetic-extension": "changed"}):
            with self.assertRaises(ValueError):
                convergence.load_plan(self.workspace, run)
        marker = self.parent / "models" / plan["tasks"][0]["key"] / "done.json"
        done = json.loads(marker.read_text())
        done["summary"]["best_step"] = 999
        marker.write_text(json.dumps(done))
        with self.assertRaises(ValueError):
            convergence.load_plan(self.workspace, run)

    def test_configuration_change_cannot_redefine_existing_extension(self):
        self.complete_parent()
        run = self.prepare_child()
        changed = copy.deepcopy(self.config)
        changed["training"]["max_steps"] += 1
        with self.assertRaises(ValueError):
            convergence.prepare(self.workspace, changed, run_dir=run)
        for extra in ({"learning_rate": .2}, {"batch_size": 7}, {"seed": 9},
                      {"device": "cpu"}, {"validation_interval": 3}):
            invalid = copy.deepcopy(self.config)
            invalid["training"].update(extra)
            invalid["output_root"] += "-invalid"
            with self.assertRaises(ValueError):
                convergence.prepare(self.workspace, invalid)

    def test_unknown_config_fields_and_nonextended_budget_are_rejected(self):
        self.complete_parent()
        for field in ("sizes", "seeds", "mode", "extra"):
            invalid = copy.deepcopy(self.config)
            invalid[field] = []
            with self.subTest(field=field), self.assertRaises(ValueError):
                convergence.prepare(self.workspace, invalid)
        for max_steps in (1, 6, True):
            invalid = copy.deepcopy(self.config)
            invalid["training"]["max_steps"] = max_steps
            with self.subTest(max_steps=max_steps), self.assertRaises(ValueError):
                convergence.prepare(self.workspace, invalid)

    def test_paused_history_prefix_cannot_be_changed(self):
        self.complete_parent()
        run = self.prepare_child()
        convergence.run_campaign(self.workspace, run, max_chunks=1, output=lambda _: None)
        task = convergence.load_plan(self.workspace, run)["tasks"][0]
        path = run / "models" / task["key"] / "latest.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["history"][0]["validation_mse"] += .5
        torch.save(payload, path)
        with self.assertRaises(ValueError):
            convergence.run_campaign(self.workspace, run, output=lambda _: None)

    def test_new_grace_and_patience_do_not_reset_on_pause(self):
        task, payload, config = self.parent_snapshot()
        config["max_steps"] = 20
        config["early_stopping"] = {
            "enabled": True, "min_steps": 10, "patience": 3, "min_delta": 0.0,
        }
        fork = convergence._fork_snapshot(payload["session"], config)
        data = synthetic_views(self.parent / "features", task["L"])[task["view"]]
        def session(checkpoint):
            return mlp_training.TrainSession(data.train.x, data.train.y,
                                             data.validation.x, data.validation.y,
                                             config, checkpoint=checkpoint)
        continued = session(fork)
        with patch.object(continued, "_mse", return_value=continued.best_val_mse + 1):
            events = continued.advance(6)
        self.assertEqual([event["step"] for event in events], [8, 10, 12])
        self.assertEqual([event["early_stopping_wait"] for event in events], [0, 1, 2])
        restored = session(continued.snapshot())
        self.assertEqual(restored.early_stopping_wait, 2)
        with patch.object(restored, "_mse", return_value=restored.best_val_mse + 1):
            restored.advance(20)
        self.assertEqual(restored.step, 14)
        self.assertEqual(restored.stop_reason, "early_stopping")

    def test_completed_child_hashes_and_fork_metadata_remain_verified(self):
        self.complete_parent()
        run = self.prepare_child()
        convergence.run_campaign(self.workspace, run, output=lambda _: None)
        task = convergence.load_plan(self.workspace, run)["tasks"][0]
        job = run / "models" / task["key"]
        before = file_hashes(job)
        convergence.run_campaign(self.workspace, run, output=lambda _: None)
        self.assertEqual(file_hashes(job), before)
        fork = json.loads((job / "fork.json").read_text())
        fork["parent_step"] += 1
        (job / "fork.json").write_text(json.dumps(fork))
        with self.assertRaisesRegex(ValueError, "fork identity changed"):
            convergence.summarize_campaign(self.workspace, run)
        (job / "fork.json").write_text(json.dumps({**fork, "parent_step": fork["parent_step"] - 1}))
        (job / "history.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "output changed"):
            convergence.run_campaign(self.workspace, run)

    def test_completed_child_summary_cannot_override_hashed_evaluation(self):
        self.complete_parent()
        run = self.prepare_child()
        convergence.run_campaign(self.workspace, run, output=lambda _: None)
        task = convergence.load_plan(self.workspace, run)["tasks"][0]
        marker = run / "models" / task["key"] / "done.json"
        done = json.loads(marker.read_text())
        done["summary"]["scores"]["f"]["mse"] = 0.0
        marker.write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError, "marker summary"):
            convergence.summarize_campaign(self.workspace, run)

    def test_final_checkpoint_recovers_after_interrupted_export(self):
        self.complete_parent()
        run = self.prepare_child()
        with patch.object(base, "_evaluate", side_effect=OSError("export interrupted")):
            with self.assertRaisesRegex(OSError, "export interrupted"):
                convergence.run_campaign(self.workspace, run, output=lambda _: None)
        summary = convergence.run_campaign(self.workspace, run, output=lambda _: None)
        self.assertEqual(summary["completed"], 4)

    def test_real_sigint_commits_extension_and_restores_signal_handler(self):
        self.complete_parent()
        run = self.prepare_child()
        signaled = []
        def output(message):
            if "best validation" in message and not signaled:
                signaled.append(True)
                os.kill(os.getpid(), signal.SIGINT)
        before = signal.getsignal(signal.SIGINT)
        result = convergence.run_campaign(self.workspace, run, output=output)
        self.assertTrue(signaled)
        self.assertEqual(result["completed"], 0)
        self.assertEqual(result["tasks"][0]["step"], 8)
        self.assertEqual(signal.getsignal(signal.SIGINT), before)
        self.assertEqual(convergence.run_campaign(self.workspace, run, output=lambda _: None)["completed"], 4)


@unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA device required for GPU controls")
class CudaConvergenceControls(unittest.TestCase):
    """Actual CUDA execution; only the tiny synthetic data/source adapters vary."""

    real_cuda = True
    setUp = ConvergenceCampaignTests.setUp
    assert_tree_equal = ConvergenceCampaignTests.assert_tree_equal
    prepare_child = ConvergenceCampaignTests.prepare_child
    parent_snapshot = ConvergenceCampaignTests.parent_snapshot

    def complete_parent(self):
        plan = ConvergenceCampaignTests.complete_parent(self)
        for task in plan["tasks"]:
            payload = torch.load(self.parent / "models" / task["key"] / "latest.pt",
                                 map_location="cpu", weights_only=True)
            self.assertEqual(payload["accelerator"]["type"], "cuda")
            self.assertEqual(payload["session"]["device_type"], "cuda")
        return plan

    test_forked_trajectory_matches_independent_long_budget_training = (
        ConvergenceCampaignTests.test_forked_trajectory_matches_independent_long_budget_training)
    test_pause_resume_is_bit_identical_and_parent_files_remain_unchanged = (
        ConvergenceCampaignTests.test_pause_resume_is_bit_identical_and_parent_files_remain_unchanged)


if __name__ == "__main__":
    unittest.main()
