"""Device preflight and accelerator guards without accessing research data."""

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


@unittest.skipIf(torch is None, "Install the project training dependencies")
class MLPDeviceWorkflowTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.MLPExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.repo = fixture.repo
        self.config = copy.deepcopy(fixture.config)
        self.plan = fixture.plan
        self.run = workflow.artifact_root(self.repo) / "device-control"

    def test_unavailable_cuda_rejected_before_any_dataset_loading(self):
        self.config["training"]["device"] = "cuda"
        with patch.object(workflow, "resolve_device", side_effect=ValueError("CUDA unavailable")), \
                patch.object(workflow, "_load_data") as load:
            with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
                workflow.train(self.config, run_dir=self.run, repo_root=self.repo, interactive=False)
        load.assert_not_called()
        self.assertFalse((self.run / "latest.pt").exists())

    def test_recorded_accelerator_and_resume_identity(self):
        workflow.train(self.config, run_dir=self.run, repo_root=self.repo,
                       interactive=False, max_chunks=1)
        saved = json.loads((self.run / "manifest.json").read_text())
        self.assertEqual(saved["accelerator"], {"type": "cpu"})
        self.assertIn("cuda_runtime", saved["environment"])
        self.assertIn("Compute device: **cpu**", (self.run / "RESULTS.md").read_text())
        with patch.object(workflow, "_accelerator", return_value={"type": "changed"}):
            with self.assertRaisesRegex(ValueError, "accelerator changed"):
                workflow.train(run_dir=self.run, resume=True, repo_root=self.repo, interactive=False)
        self.assertFalse((self.run / "test_metrics.json").exists())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "Requires CUDA GPU")
    def test_actual_cuda_queue_pauses_resumes_and_records_hardware_without_test_access(self):
        from quantum_lattice.models import mlp_queue
        self.config["training"]["device"] = "cuda"
        queue = workflow.artifact_root(self.repo) / "queues" / "cuda-control"
        held_out = {self.repo / "instances" / f"{item['id']}.npz"
                    for item in self.plan if item["split"] == "test"}
        original = np.load

        def guarded(path, *args, **kwargs):
            self.assertNotIn(Path(path), held_out)
            return original(path, *args, **kwargs)

        with patch("quantum_lattice.data.mlp_data.np.load", side_effect=guarded):
            mlp_queue.run_queue(self.config, sizes=(3,), queue_dir=queue,
                                repo_root=self.repo, interactive=False, max_chunks=1)
            self.assertEqual(mlp_queue.queue_summary(queue, repo_root=self.repo)["state"], "paused")
            mlp_queue.run_queue(None, queue_dir=queue, resume=True,
                                repo_root=self.repo, interactive=False)
        summary = mlp_queue.queue_summary(queue, repo_root=self.repo)
        self.assertEqual(summary["completed_models"], 2)
        for task in summary["tasks"]:
            self.assertEqual(task["device"], "cuda")
            manifest = json.loads((Path(task["run_dir"]) / "manifest.json").read_text())
            self.assertEqual(manifest["accelerator"]["name"], torch.cuda.get_device_name(0))
            self.assertEqual(manifest["environment"]["cuda_runtime"], torch.version.cuda)
            self.assertFalse(task["test_evaluated"])
        self.assertIn("device", (queue / "comparison.csv").read_text())


if __name__ == "__main__":
    unittest.main()
