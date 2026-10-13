from copy import deepcopy
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import quantum_lattice.models as models
from quantum_lattice.models.mlp_campaign import (
    ARTIFACT_PATH, format_run_summary, run_summary, train_both,
)


class MLPCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.campaign = self.repo / ARTIFACT_PATH / "campaigns" / "test-both"
        self.config = {"schema_version": 1, "mode": "paper_size", "lattice_size": 3,
                       "dataset_path": "fixture", "training_hamiltonians": 3, "fixed_edge": 0,
                       "training": {"seed": 12, "max_steps": 100,
                                    "early_stopping": {"enabled": True, "patience": 20,
                                                       "min_delta": 1e-6, "min_steps": 10}},
                       "execution": {"checkpoint_steps": 10}}
        for name in ("config.json", "manifest.json", "dataset.json", "verification.json"):
            self.write(self.repo / "fixture" / name, {"fixture": name})
        self.calls = []
        self.outcomes = []
        self.workflow = SimpleNamespace(validate_config=lambda c: deepcopy(c),
                                        train=Mock(side_effect=self.fake_train),
                                        evaluate=Mock(side_effect=AssertionError("Final test must remain explicit")))
        self.module_patch = patch.object(models, "mlp_experiment", self.workflow, create=True)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def read_record(self):
        return json.loads((self.campaign / "campaign.json").read_text())

    def fake_train(self, config, *, run_dir, resume, **kwargs):
        self.calls.append({"mode": config["mode"], "resume": resume,
                           "config": deepcopy(config), "kwargs": kwargs, "path": run_dir})
        run_dir.mkdir(parents=True, exist_ok=True)
        # The campaign must declare the currently starting child before launch.
        record = self.read_record()
        self.assertEqual(record["current_mode"], config["mode"])
        self.assertEqual(record["runs"][config["mode"]]["state"], "resuming" if resume else "starting")
        outcome = self.outcomes.pop(0) if self.outcomes else "trained"
        if outcome == "initialization_failure":
            (run_dir / ".run.lock").write_text("")
            raise RuntimeError("fixture failed before initialization")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        self.write(run_dir / "config.json", config)
        self.write(run_dir / "manifest.json", {"row_counts": {"train": 3, "validation": 2, "test": 36}})
        self.write(run_dir / "status.json", {"state": outcome, "step": 40 if outcome == "paused" else 100,
                   "max_steps": 100, "epoch": 2, "training_row_passes": 1.5,
                   "train_mse": .1, "validation_mse": .2, "best_validation_mse": .15,
                   "best_step": 30, "test_evaluated": False, "elapsed_seconds": 65})
        return run_dir

    def test_fresh_modes_run_in_order_with_same_seed_without_test(self):
        result = train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo, interactive=False)
        self.assertEqual(result, self.campaign)
        self.assertEqual([call["mode"] for call in self.calls], ["paper_size", "all_edges"])
        self.assertEqual([call["resume"] for call in self.calls], [False, False])
        self.assertEqual([call["config"]["training"]["seed"] for call in self.calls], [12, 12])
        self.assertEqual(self.read_record()["state"], "trained")
        self.assertEqual(self.read_record()["completed_modes"], ["paper_size", "all_edges"])
        self.workflow.evaluate.assert_not_called()
        self.assertEqual(self.config["mode"], "paper_size")

    def test_pause_stops_second_and_resume_continues_first(self):
        self.outcomes = ["paused"]
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo, max_chunks=1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["kwargs"]["max_chunks"], 1)
        self.assertEqual(self.read_record()["state"], "paused")
        self.assertFalse((self.campaign / "all_edges/run").exists())
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo, resume=True)
        self.assertEqual([(c["mode"], c["resume"]) for c in self.calls],
                         [("paper_size", False), ("paper_size", True), ("all_edges", False)])

    def test_resume_skips_finished_and_evaluated_models(self):
        self.outcomes = ["trained", "paused"]
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo)
        path = self.campaign / "paper_size/run/status.json"
        status = json.loads(path.read_text())
        status["state"] = "evaluated"
        self.write(path, status)
        train_both(None, campaign_dir=self.campaign, repo_root=self.repo, resume=True)
        self.assertEqual([c["mode"] for c in self.calls], ["paper_size", "all_edges", "all_edges"])
        calls_before = len(self.calls)
        train_both(None, campaign_dir=self.campaign, repo_root=self.repo, resume=True)
        self.assertEqual(len(self.calls), calls_before)
        self.workflow.evaluate.assert_not_called()

    def test_interrupt_stops_sequence(self):
        self.outcomes = ["interrupt"]
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.read_record()["state"], "paused")

    def test_initialization_failure_preserved_and_retry_uses_fresh_path(self):
        self.outcomes = ["initialization_failure"]
        with self.assertRaisesRegex(RuntimeError, "initialization"):
            train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo)
        self.assertEqual(self.read_record()["state"], "failed")
        train_both(None, campaign_dir=self.campaign, repo_root=self.repo, resume=True)
        self.assertTrue((self.campaign / "paper_size/run/.run.lock").exists())
        record = self.read_record()
        self.assertEqual(record["runs"]["paper_size"]["path"], "paper_size/retry_001")
        self.assertEqual(record["runs"]["paper_size"]["uninitialized_attempts"], ["paper_size/run"])
        self.assertEqual(record["state"], "trained")

    def test_config_is_immutable_and_child_settings_cannot_change(self):
        self.outcomes = ["paused"]
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo)
        changed = deepcopy(self.config)
        changed["training"]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "immutable"):
            train_both(changed, campaign_dir=self.campaign, repo_root=self.repo, resume=True)
        path = self.campaign / "paper_size/run/config.json"
        saved = json.loads(path.read_text())
        saved["mode"] = "all_edges"
        self.write(path, saved)
        with self.assertRaisesRegex(ValueError, "child run settings"):
            train_both(None, campaign_dir=self.campaign, repo_root=self.repo, resume=True)

    def test_normalized_nested_training_defaults_allow_resume_but_supplied_values_stay_fixed(self):
        self.config["training"]["early_stopping"] = {"enabled": True}
        self.outcomes = ["paused"]
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo)
        path = self.campaign / "paper_size/run/config.json"
        saved = json.loads(path.read_text())
        saved["training"]["early_stopping"].update(patience=20, min_delta=1e-6, min_steps=10)
        self.write(path, saved)
        # Normalization adds defaults without changing the supplied enabled flag.
        self.outcomes = ["paused"]
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo, resume=True)
        self.assertTrue(self.calls[-1]["resume"])
        # An explicitly configured nested value remains immutable.
        saved["training"]["early_stopping"]["enabled"] = False
        self.write(path, saved)
        with self.assertRaisesRegex(ValueError, "child training settings"):
            train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo, resume=True)

    def test_dataset_metadata_change_blocks_second_model_and_resume(self):
        original = self.workflow.train.side_effect
        def change_after_first(*args, **kwargs):
            path = original(*args, **kwargs)
            self.write(self.repo / "fixture/dataset.json", {"changed": True})
            return path
        self.workflow.train.side_effect = change_after_first
        with self.assertRaisesRegex(ValueError, "between campaign models"):
            train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.read_record()["state"], "failed")
        with self.assertRaisesRegex(ValueError, "since campaign creation"):
            train_both(None, campaign_dir=self.campaign, repo_root=self.repo, resume=True)

    def test_path_and_process_lock_guards(self):
        with self.assertRaisesRegex(ValueError, "must be below"):
            train_both(self.config, campaign_dir=self.repo / "published", repo_root=self.repo)
        self.outcomes = ["paused"]
        train_both(self.config, campaign_dir=self.campaign, repo_root=self.repo)
        with (self.campaign / ".campaign.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "already active"):
                train_both(None, campaign_dir=self.campaign, repo_root=self.repo, resume=True)
            fcntl.flock(handle, fcntl.LOCK_UN)

    def test_summary_handles_missing_and_legacy_records_without_payloads(self):
        run = self.campaign / "paper_size/run"
        empty = run_summary(run)
        self.assertEqual(empty["state"], "not_started")
        self.assertIn("Final test: not evaluated", format_run_summary(empty))
        self.write(run / "config.json", {"lattice_size": 4, "mode": "paper_size", "training": {"max_steps": 10000}})
        self.write(run / "status.json", {"state": "trained", "step": 10000, "best_validation_mse": .017,
                                        "best_step": 8400, "elapsed_seconds": 65})
        self.write(run / "data_selection.json", {"row_counts": {"train": 7450, "validation": 1242, "test": 39744}})
        (run / "latest.pt").write_bytes(b"not a checkpoint")
        original_read = Path.read_text
        read_paths = []
        def spy(path, *args, **kwargs):
            read_paths.append(path.name)
            return original_read(path, *args, **kwargs)
        with patch.object(Path, "read_text", spy):
            summary = run_summary(run)
        self.assertNotIn("latest.pt", read_paths)
        self.assertEqual(summary["row_counts"]["train"], 7450)
        self.assertIn("legacy", summary["stop_reason"])
        formatted = format_run_summary(summary)
        self.assertIn("00:01:05", formatted)
        self.assertIn("0.017", formatted)
        self.assertIn("RESULTS:", formatted)

    def test_summary_early_stopping_and_final_test_metrics(self):
        run = self.campaign / "paper_size/run"
        self.write(run / "config.json", self.config)
        self.write(run / "status.json", {"state": "evaluated", "step": 80, "max_steps": 100,
                                        "stop_reason": "early_stopping", "early_stopping_wait": 20,
                                        "test_evaluated": True})
        self.write(run / "test_metrics.json", {"mse": .01, "rmse": .1, "mae": .08, "count": 36})
        summary = run_summary(run)
        formatted = format_run_summary(summary)
        self.assertIn("early_stopping", formatted)
        self.assertIn("wait 20/20", formatted)
        self.assertIn("minimum 10 updates", formatted)
        self.assertIn("Final test: MSE 0.01", formatted)
        summary["early_stopping"] = {"enabled": False}
        self.assertIn("disabled (fixed update budget)", format_run_summary(summary))
        (run / "manifest.json").write_text("not json")
        self.assertTrue(run_summary(run)["read_errors"])


if __name__ == "__main__":
    unittest.main()
