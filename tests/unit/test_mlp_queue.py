"""Queue scheduling, immutable metadata and test-boundary checks using mock runs."""
from copy import deepcopy
import csv
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import quantum_lattice.models as models
from quantum_lattice.models import mlp_queue as queue
from quantum_lattice.models.mlp_terminal import training_context


class MLPQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.path = self.repo / queue.ARTIFACT_PATH / "queues" / "fixture"
        self.config = {"schema_version": 1, "mode": "paper_size", "lattice_size": 3,
                       "dataset_path": "datasets/L3/run", "training_hamiltonians": 3, "fixed_edge": 0,
                       "training": {"seed": 12, "max_steps": 100},
                       "execution": {"checkpoint_steps": 10}}
        for size in (3, 4, 8):
            self.dataset(size)
        self.calls = []
        self.outcomes = []
        self.evaluations = []
        self.workflow = SimpleNamespace(validate_config=lambda c: deepcopy(c),
                                        train=Mock(side_effect=self.train),
                                        evaluate=Mock(side_effect=self.evaluate))
        patcher = patch.object(models, "mlp_experiment", self.workflow, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def dataset(self, size):
        root = self.repo / f"datasets/L{size}/run"
        instances = [{"id": f"{split}-{i}", "split": split, "sha256": "a" * 64}
                     for split, count in (("train", 3), ("validation", 1), ("test", 1)) for i in range(count)]
        self.write(root / "config.json", {"lattice": {"L": size}})
        self.write(root / "status.json", {"state": "complete", "completed": len(instances)})
        self.write(root / "manifest.json", {"instances": instances})
        self.write(root / "dataset.json", {"instances": instances, "split_counts": {"train": 3, "validation": 1, "test": 1}})
        self.write(root / "verification.json", {"passed": True, "instances": instances})
        (root / "unread-test-payload.npz").write_bytes(b"not an archive")

    def record(self):
        return json.loads((self.path / "queue.json").read_text())

    def train(self, config, *, run_dir, resume, **kwargs):
        self.calls.append((config["lattice_size"], config["mode"], resume, Path(run_dir)))
        run_dir.mkdir(parents=True, exist_ok=True)
        outcome = self.outcomes.pop(0) if self.outcomes else "trained"
        if outcome == "interrupt":
            raise KeyboardInterrupt
        if outcome == "partial_config":
            self.write(run_dir / "config.json", config)
            raise KeyboardInterrupt
        if outcome == "fail_init":
            (run_dir / ".run.lock").write_text("")
            raise RuntimeError("fixture initialization failed")
        self.write(run_dir / "config.json", config)
        (run_dir / "latest.pt").write_bytes(b"fixture checkpoint")
        self.write(run_dir / "manifest.json", {"row_counts": {"train": 3, "validation": 1, "test": 32}})
        self.write(run_dir / "status.json", {"state": "paused" if outcome == "paused" else "trained", "step": 100,
                   "max_steps": 100, "best_validation_mse": .05, "best_step": 70, "test_evaluated": False,
                   "training_row_passes": 5.2, "training_examples_seen": 39, "epoch": 6})
        if outcome == "quit_after_finish":
            with training_context() as context:
                context["stop_requested"]["requested"] = True
        return run_dir

    def evaluate(self, run, **kwargs):
        self.evaluations.append(Path(run))
        status = json.loads((run / "status.json").read_text())
        status.update(state="evaluated", test_evaluated=True)
        self.write(run / "status.json", status)
        self.write(run / "test_metrics.json", {"rmse": .2, "mae": .1, "mse": .04})
        return {"rmse": .2, "mae": .1, "mse": .04}

    def run_queue(self, sizes=(3, 4), **kwargs):
        return queue.run_queue(self.config, sizes=sizes, queue_dir=self.path, repo_root=self.repo,
                               interactive=False, **kwargs)

    def resume(self, **kwargs):
        return queue.run_queue(None, queue_dir=self.path, repo_root=self.repo, resume=True,
                               interactive=False, **kwargs)

    def test_train_all_sizes_and_modes_in_order_and_emit_comparison_without_test(self):
        self.assertEqual(self.run_queue(), self.path)
        self.assertEqual([item[:3] for item in self.calls], [(3, "paper_size", False), (3, "all_edges", False),
                                                          (4, "paper_size", False), (4, "all_edges", False)])
        self.workflow.evaluate.assert_not_called()
        summary = queue.queue_summary(self.path, repo_root=self.repo)
        self.assertEqual((summary["state"], summary["completed_models"], summary["total_models"]), ("trained", 4, 4))
        self.assertEqual(summary["evaluated_models"], 0)
        with (self.path / "comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["test_rmse"] == "" for row in rows))
        self.assertEqual(rows[0]["training_row_passes"], "5.2")
        self.assertIn("shared-test", (self.path / "RESULTS.md").read_text())
        self.assertIn("4/4", queue.format_queue_summary(summary))

    def test_pause_stops_queue_and_resume_skips_complete_models(self):
        self.outcomes = ["trained", "paused"]
        self.run_queue(max_chunks=1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.record()["state"], "paused")
        self.resume()
        self.assertEqual([item[:3] for item in self.calls], [(3, "paper_size", False), (3, "all_edges", False),
                                                          (3, "all_edges", True), (4, "paper_size", False), (4, "all_edges", False)])
        before = len(self.calls)
        self.resume()
        self.assertEqual(len(self.calls), before)

    def test_ctrl_c_and_quit_after_final_chunk_do_not_start_next_model(self):
        for outcome in ("interrupt", "quit_after_finish"):
            with self.subTest(outcome=outcome):
                self.path = self.path.parent / outcome
                self.calls = []
                self.outcomes = [outcome]
                self.run_queue()
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(self.record()["state"], "paused")

    def test_partial_config_and_uninitialized_attempts_are_preserved_and_retry(self):
        self.outcomes = ["partial_config"]
        self.run_queue()
        original = self.calls[0][3]
        self.resume()
        self.assertTrue((original / "config.json").exists())
        self.assertEqual(self.calls[1][3].name, "retry_001")
        self.assertFalse(self.calls[1][2])
        self.assertEqual(self.record()["state"], "trained")

    def test_all_datasets_preflight_before_any_training(self):
        self.write(self.repo / "datasets/L4/run/status.json", {"state": "sampling"})
        with self.assertRaisesRegex(ValueError, "L=4 dataset is sampling"):
            self.run_queue()
        self.assertFalse(self.path.exists())
        self.assertEqual(self.calls, [])
        ready = queue.discover_ready_sizes(self.config, sizes=(3, 4), repo_root=self.repo)
        self.assertEqual(ready[3]["state"], "complete")
        self.assertEqual(ready[4]["state"], "sampling")

    def test_metadata_drift_between_sizes_and_between_models_blocks(self):
        original = self.workflow.train.side_effect
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            self.write(self.repo / "datasets/L4/run/dataset.json", {"changed": True})
            return result
        self.workflow.train.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "L=4 dataset"):
            self.run_queue()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.record()["state"], "failed")
        self.dataset(4)
        self.workflow.train.side_effect = original
        # Existing completed L3 is skipped on resume after restoring exact data.
        self.resume()
        self.assertEqual(len(self.calls), 4)

    def test_plan_tamper_and_changed_user_settings_block_resume(self):
        self.outcomes = ["paused"]
        self.run_queue()
        changed = deepcopy(self.config)
        changed["training"]["max_steps"] = 999
        with self.assertRaisesRegex(ValueError, "immutable"):
            queue.run_queue(changed, queue_dir=self.path, repo_root=self.repo, resume=True)
        record = self.record()
        record["plan"]["groups"][0]["campaign_path"] = "../../escape"
        self.write(self.path / "queue.json", record)
        with self.assertRaisesRegex(ValueError, "plan/configuration changed"):
            self.resume()

    def test_single_mode_pauses_resumes_and_retries_partial_initialization(self):
        self.outcomes = ["partial_config"]
        self.run_queue(modes=("all_edges",))
        original = self.calls[0][3]
        self.resume()
        self.assertEqual([item[:3] for item in self.calls], [(3, "all_edges", False), (3, "all_edges", False), (4, "all_edges", False)])
        self.assertTrue((original / "config.json").exists())
        self.assertEqual(queue.queue_summary(self.path, repo_root=self.repo)["completed_models"], 2)

    def test_paths_and_lock_are_guarded(self):
        with self.assertRaisesRegex(ValueError, "must be below"):
            queue.run_queue(self.config, queue_dir=self.repo / "wrong", sizes=(3,), repo_root=self.repo)
        self.outcomes = ["paused"]
        self.run_queue()
        with (self.path / ".queue.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "already active"):
                self.resume()
            fcntl.flock(handle, fcntl.LOCK_UN)

    def test_explicit_batch_evaluation_only_after_all_complete_and_is_resumable(self):
        self.outcomes = ["paused"]
        self.run_queue()
        with self.assertRaisesRegex(ValueError, "Finish every"):
            queue.evaluate_queue(self.path, repo_root=self.repo)
        self.assertFalse(self.evaluations)
        self.resume()
        summary = queue.evaluate_queue(self.path, repo_root=self.repo)
        self.assertEqual((summary["evaluated_models"], summary["evaluation_state"]), (4, "evaluated"))
        self.assertEqual(len(self.evaluations), 4)
        queue.evaluate_queue(self.path, repo_root=self.repo)
        self.assertEqual(len(self.evaluations), 4)
        with (self.path / "comparison.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["test_rmse"] == "0.2" for row in rows))

    def test_evaluation_interrupt_preserves_completed_results(self):
        self.run_queue()
        original = self.workflow.evaluate.side_effect
        def interrupt_second(run, **kwargs):
            if self.evaluations:
                raise KeyboardInterrupt
            return original(run, **kwargs)
        self.workflow.evaluate.side_effect = interrupt_second
        summary = queue.evaluate_queue(self.path, repo_root=self.repo)
        self.assertEqual((summary["evaluation_state"], summary["evaluated_models"]), ("paused", 1))
        self.workflow.evaluate.side_effect = original
        summary = queue.evaluate_queue(self.path, repo_root=self.repo)
        self.assertEqual(summary["evaluated_models"], 4)
        self.assertEqual(len(self.evaluations), 4)

    def test_bad_verification_and_wrong_lattice_declared_not_ready(self):
        self.write(self.repo / "datasets/L3/run/verification.json", {"passed": False})
        readiness = queue.discover_ready_sizes(self.config, sizes=(3,), repo_root=self.repo)[3]
        self.assertEqual(readiness["state"], "unverified")
        self.dataset(3)
        self.write(self.repo / "datasets/L3/run/config.json", {"lattice": {"L": 99}})
        with self.assertRaisesRegex(ValueError, "lattice size differs"):
            self.run_queue(sizes=(3,))

    def test_all_seven_sizes_schedule_exactly_fourteen_models(self):
        for size in queue.SIZES:
            self.dataset(size)
        self.run_queue(sizes=queue.SIZES)
        expected = [(size, mode) for size in queue.SIZES for mode in queue.MODES]
        self.assertEqual([item[:2] for item in self.calls], expected)
        self.assertEqual(queue.queue_summary(self.path, repo_root=self.repo)["completed_models"], 14)

    def test_partial_campaign_creation_preserved_in_new_attempt(self):
        original = queue.paired.train_both
        def interrupt_before_record(config, *, campaign_dir, **kwargs):
            campaign_dir.mkdir(parents=True)
            (campaign_dir / ".campaign.lock").write_text("")
            raise KeyboardInterrupt
        with patch.object(queue.paired, "train_both", side_effect=interrupt_before_record):
            self.run_queue()
        record = self.record()
        old = self.repo / queue.ARTIFACT_PATH / record["plan"]["groups"][0]["campaign_path"]
        self.resume()
        self.assertTrue((old / ".campaign.lock").exists())
        self.assertEqual(self.record()["attempts"]["3"], 1)
        self.assertEqual(self.record()["state"], "trained")

    def test_own_dataset_drift_stops_before_second_mode(self):
        original = self.workflow.train.side_effect
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            self.write(self.repo / "datasets/L3/run/dataset.json", {"changed": True})
            return result
        self.workflow.train.side_effect = mutate
        with self.assertRaisesRegex(ValueError, "between campaign models"):
            self.run_queue()
        self.assertEqual(len(self.calls), 1)

    def test_child_config_tamper_blocks_resume_and_batch_evaluation(self):
        self.run_queue()
        run = self.calls[-1][3]
        config = json.loads((run / "config.json").read_text())
        config["training"]["seed"] += 1
        self.write(run / "config.json", config)
        with self.assertRaisesRegex(ValueError, "child training settings"):
            self.resume()
        with self.assertRaisesRegex(ValueError, "child training settings"):
            queue.evaluate_queue(self.path, repo_root=self.repo)
        self.assertFalse(self.evaluations)

    def test_summary_uses_json_only(self):
        self.run_queue()
        original = Path.read_text
        seen = []
        def spy(path, *args, **kwargs):
            seen.append(path.suffix)
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", spy):
            queue.queue_summary(self.path, repo_root=self.repo)
        self.assertEqual(set(seen), {".json"})


if __name__ == "__main__":
    unittest.main()
