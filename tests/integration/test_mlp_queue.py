"""Tiny actual PyTorch queue/checkpoint controls; no research data is used."""
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
from integration import test_mlp_experiment as fixtures

if torch is not None:
    from quantum_lattice.models import mlp_experiment as workflow
    from quantum_lattice.models import mlp_queue as queue


@unittest.skipIf(torch is None, "Install the project training dependencies")
class MLPQueueIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.MLPExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.repo, self.config, self.plan = fixture.repo, copy.deepcopy(fixture.config), fixture.plan
        self.path = workflow.artifact_root(self.repo) / "queues" / "synthetic"

    def test_real_pair_pause_resume_and_separate_explicit_batch_test(self):
        held_out = {self.repo / "instances" / f"{item['id']}.npz" for item in self.plan if item["split"] == "test"}
        original = np.load
        def guarded(path, *args, **kwargs):
            self.assertNotIn(Path(path), held_out)
            return original(path, *args, **kwargs)
        with patch("quantum_lattice.data.mlp_data.np.load", side_effect=guarded):
            queue.run_queue(self.config, sizes=(3,), queue_dir=self.path, interactive=False,
                            repo_root=self.repo, max_chunks=1)
            first = queue.queue_summary(self.path, repo_root=self.repo)
            self.assertEqual((first["state"], first["completed_models"]), ("paused", 0))
            self.assertEqual(first["tasks"][0]["step"], 3)
            self.assertEqual(first["tasks"][1]["state"], "not_started")
            queue.run_queue(None, queue_dir=self.path, resume=True, interactive=False, repo_root=self.repo)
            trained = queue.queue_summary(self.path, repo_root=self.repo)
            self.assertEqual((trained["state"], trained["completed_models"]), ("trained", 2))
            self.assertFalse(any(item["test_evaluated"] for item in trained["tasks"]))
        evaluated = queue.evaluate_queue(self.path, repo_root=self.repo)
        self.assertEqual(evaluated["evaluated_models"], 2)
        self.assertTrue(all(item["test_metrics"]["count"] == 36 for item in evaluated["tasks"]))
        self.assertEqual(evaluated["tasks"][0]["row_counts"]["test"], evaluated["tasks"][1]["row_counts"]["test"])
        self.assertIn("test_rmse", (self.path / "comparison.csv").read_text())

    def test_initial_checkpoint_interrupt_preserves_partial_attempt_and_restarts_safely(self):
        original = workflow._atomic_torch
        calls = 0
        def interrupt_initial(path, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt
            return original(path, payload)
        with patch.object(workflow, "_atomic_torch", side_effect=interrupt_initial):
            queue.run_queue(self.config, sizes=(3,), queue_dir=self.path, interactive=False, repo_root=self.repo)
        first = queue.queue_summary(self.path, repo_root=self.repo)
        partial = Path(first["tasks"][0]["run_dir"])
        self.assertTrue((partial / "config.json").is_file())
        self.assertFalse((partial / "latest.pt").exists())
        self.assertEqual(first["state"], "paused")
        queue.run_queue(None, queue_dir=self.path, resume=True, interactive=False, repo_root=self.repo)
        final = queue.queue_summary(self.path, repo_root=self.repo)
        self.assertEqual(final["completed_models"], 2)
        self.assertEqual(Path(final["tasks"][0]["run_dir"]).name, "retry_001")
        self.assertTrue((partial / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
