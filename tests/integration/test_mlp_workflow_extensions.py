"""Tiny workflow controls; no full research training or test bank is used."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration import test_mlp_experiment as fixture_module

if torch is not None:
    from quantum_lattice.models import mlp_experiment as workflow
    from quantum_lattice.models.mlp_campaign import train_both
    from quantum_lattice.models.mlp_training import TrainSession


@unittest.skipIf(torch is None, "Use the separate MLP environment")
class MLPWorkflowExtensionTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.MLPExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.repo, self.config, self.plan = fixture.repo, copy.deepcopy(fixture.config), fixture.plan
        self.root = workflow.artifact_root(self.repo)

    def stopping_config(self):
        config = copy.deepcopy(self.config)
        config["training"].update(max_steps=50, early_stopping={
            "enabled": True, "patience": 2, "min_delta": .1, "min_steps": 0})
        return config

    def test_early_finished_checkpoint_is_frozen_and_evaluable_below_ceiling(self):
        run = self.root / "early"
        with patch.object(TrainSession, "_mse", return_value=.5):
            workflow.train(self.stopping_config(), run_dir=run, repo_root=self.repo, interactive=False)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["state"], status["step"], status["stop_reason"]), ("trained", 4, "early_stopping"))
        self.assertEqual(status["early_stopping_wait"], 2)
        self.assertEqual(status["best_step"], 1)
        self.assertFalse((run / "test_metrics.json").exists())
        self.assertEqual(workflow.evaluate(run, repo_root=self.repo)["count"], 36)
        self.assertIn("early_stopping", (run / "RESULTS.md").read_text())

    def test_real_paired_workflow_pauses_resumes_and_never_opens_test(self):
        campaign = self.root / "campaigns" / "paired"
        test_paths = {self.repo / "instances" / f"{item['id']}.npz" for item in self.plan if item["split"] == "test"}
        original = np.load
        def guarded(path, *args, **kwargs):
            self.assertNotIn(Path(path), test_paths)
            return original(path, *args, **kwargs)
        with patch("quantum_lattice.data.mlp_data.np.load", side_effect=guarded), patch.object(TrainSession, "_mse", return_value=.5):
            train_both(self.stopping_config(), campaign_dir=campaign, repo_root=self.repo,
                       interactive=False, max_chunks=1)
            self.assertFalse((campaign / "all_edges/run").exists())
            train_both(None, campaign_dir=campaign, repo_root=self.repo, resume=True,
                       interactive=False, max_chunks=1)
            self.assertEqual(json.loads((campaign / "paper_size/run/status.json").read_text())["state"], "trained")
            self.assertEqual(json.loads((campaign / "all_edges/run/status.json").read_text())["state"], "paused")
            train_both(None, campaign_dir=campaign, repo_root=self.repo, resume=True, interactive=False)
        record = json.loads((campaign / "campaign.json").read_text())
        self.assertEqual(record["state"], "trained")
        self.assertEqual(record["completed_modes"], ["paper_size", "all_edges"])
        for mode in record["modes"]:
            run = campaign / record["runs"][mode]["path"]
            self.assertFalse((run / "test_metrics.json").exists())
            self.assertTrue((run / "RESULTS.md").exists())

    def make_legacy_run(self):
        run = self.root / "legacy"
        workflow.train(self.config, run_dir=run, repo_root=self.repo, interactive=False, max_chunks=1)
        config = json.loads((run / "config.json").read_text())
        config["training"].pop("early_stopping")
        payload = torch.load(run / "latest.pt", weights_only=True)
        snapshot = payload["session"]
        snapshot["training_schema_version"] = 1
        snapshot["config"].pop("early_stopping")
        for key in ("stopped_early", "early_stopping_wait", "early_stopping_reference"):
            snapshot.pop(key)
        payload["manifest"]["source_hashes"] = workflow.LEGACY_FIXED_SOURCES.copy()
        import hashlib
        payload["manifest"]["config_sha256"] = hashlib.sha256(workflow._canonical(config).encode()).hexdigest()
        workflow._atomic_json(run / "config.json", config)
        workflow._atomic_json(run / "manifest.json", payload["manifest"])
        workflow._atomic_torch(run / "latest.pt", payload)
        return run

    def test_known_legacy_run_preserves_fixed_policy_and_can_evaluate(self):
        run = self.make_legacy_run()
        workflow.train(run_dir=run, resume=True, repo_root=self.repo, interactive=False)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual((status["state"], status["step"], status["max_steps"]), ("trained", 8, 8))
        self.assertFalse(status["early_stopping"]["enabled"])
        compatibility = json.loads((run / "source_compatibility.json").read_text())
        self.assertEqual(compatibility["original_source_hashes"], workflow.LEGACY_FIXED_SOURCES)
        self.assertEqual(workflow.evaluate(run, repo_root=self.repo)["count"], 36)
        self.assertIn("source_compatibility", json.loads((run / "evaluation_provenance.json").read_text()))

    def test_unknown_legacy_source_is_rejected(self):
        run = self.make_legacy_run()
        payload = torch.load(run / "latest.pt", weights_only=True)
        payload["manifest"]["source_hashes"]["src/quantum_lattice/models/mlp.py"] = "unknown"
        workflow._atomic_torch(run / "latest.pt", payload)
        with self.assertRaisesRegex(ValueError, "source changed"):
            workflow.train(run_dir=run, resume=True, repo_root=self.repo, interactive=False)


if __name__ == "__main__":
    unittest.main()
