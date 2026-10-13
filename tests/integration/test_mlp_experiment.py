import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

# Permit both whole-suite and integration-only discovery.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unit import test_mlp_data as data_fixture

if torch is not None:
    from quantum_lattice.models import mlp_experiment as workflow
    from quantum_lattice.models.mlp_training import TrainSession


@unittest.skipIf(torch is None, "Install the separate MLP environment")
class MLPExperimentTests(unittest.TestCase):
    def setUp(self):
        fixture = data_fixture.MLPDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.repo = fixture.root
        self.plan = fixture.plan
        self.config = {"schema_version": 1, "dataset_path": str(fixture.root),
                       "lattice_size": 3, "mode": "paper_size", "fixed_edge": 0,
                       "training_hamiltonians": 2,
                       "training": {"seed": 12, "learning_rate": 0.001, "batch_size": 2,
                                    "max_steps": 8, "validation_interval": 2,
                                    "device": "cpu", "torch_threads": 1, "deterministic": True},
                       "execution": {"checkpoint_steps": 3},
                       "evaluation": {"bootstrap_samples": 20, "seed": 31}}
        self.root = workflow.artifact_root(self.repo)

    def train(self, name="run", **kwargs):
        return workflow.train(copy.deepcopy(self.config), run_dir=self.root / name,
                              interactive=False, repo_root=self.repo, **kwargs)

    def status(self, run):
        return json.loads((run / "status.json").read_text())

    def test_end_to_end_training_reserves_test_and_evaluation_exports(self):
        test_paths = {self.repo / "instances" / f"{i['id']}.npz" for i in self.plan if i["split"] == "test"}
        original_load = np.load
        def guarded(path, *args, **kwargs):
            self.assertNotIn(Path(path), test_paths)
            return original_load(path, *args, **kwargs)
        with patch("quantum_lattice.data.mlp_data.np.load", side_effect=guarded):
            run = self.train()
        self.assertEqual(self.status(run)["state"], "trained")
        self.assertFalse((run / "test_metrics.json").exists())
        self.assertEqual((self.root / "latest").resolve(), run)
        self.assertEqual((self.repo / "experiments/baseline_mlp/results").resolve(), self.root)
        payload = torch.load(run / "latest.pt", weights_only=True)
        frozen = torch.load(run / "best_model.pt", weights_only=True)
        for key in frozen["model_state_dict"]:
            self.assertTrue(torch.equal(frozen["model_state_dict"][key], payload["session"]["best_state_dict"][key]))
        report = workflow.evaluate(run, repo_root=self.repo)
        self.assertEqual(report["count"], 36)
        self.assertEqual(report["hamiltonians"], 2)
        self.assertEqual(self.status(run)["state"], "evaluated")
        for name in ("test_predictions.npz", "test_predictions.csv", "per_hamiltonian_metrics.csv",
                     "evaluation_provenance.json", "RESULTS.md", "figures/test_predictions.pdf"):
            self.assertTrue((run / name).is_file(), name)
        before = hashlib.sha256((run / "test_metrics.json").read_bytes()).hexdigest()
        with patch.object(workflow, "_load_data", wraps=workflow._load_data) as loader:
            self.assertEqual(workflow.evaluate(run, repo_root=self.repo), report)
            self.assertFalse(any(call.kwargs.get("test") for call in loader.call_args_list))
        self.assertEqual(before, hashlib.sha256((run / "test_metrics.json").read_bytes()).hexdigest())

    def test_checkpoint_resume_matches_uninterrupted(self):
        full = self.train("full")
        resumed = self.train("resumed", max_chunks=1)
        self.assertEqual(self.status(resumed)["state"], "paused")
        self.assertEqual(self.status(resumed)["step"], 3)
        with self.assertRaisesRegex(ValueError, "Finish training"):
            workflow.evaluate(resumed, repo_root=self.repo)
        workflow.train(run_dir=resumed, resume=True, interactive=False, repo_root=self.repo)
        a, b = [torch.load(path / "latest.pt", weights_only=True) for path in (full, resumed)]
        self.assertEqual(a["history"], b["history"])
        for key in ("model_state_dict", "best_state_dict"):
            for name in a["session"][key]:
                self.assertTrue(torch.equal(a["session"][key][name], b["session"][key][name]))
        self.assertTrue(torch.equal(a["session"]["shuffle_generator_state"], b["session"]["shuffle_generator_state"]))

    def test_mid_chunk_interrupt_reverts_to_durable_boundary(self):
        original = TrainSession.advance
        def interrupted(session, steps):
            original(session, 2)
            raise KeyboardInterrupt
        with patch.object(TrainSession, "advance", interrupted):
            run = self.train("interrupted")
        self.assertEqual((self.status(run)["state"], self.status(run)["step"]), ("paused", 0))
        workflow.train(run_dir=run, resume=True, interactive=False, repo_root=self.repo)
        full = self.train("full")
        a, b = [torch.load(path / "latest.pt", weights_only=True) for path in (full, run)]
        self.assertEqual(a["history"], b["history"])
        for key in a["session"]["model_state_dict"]:
            self.assertTrue(torch.equal(a["session"]["model_state_dict"][key], b["session"]["model_state_dict"][key]))

    def test_finished_checkpoint_can_repair_interrupted_finalization(self):
        original = workflow._atomic_torch
        def interrupted(path, payload):
            if path.name == "best_model.pt":
                raise OSError("simulated finalization failure")
            return original(path, payload)
        with patch.object(workflow, "_atomic_torch", side_effect=interrupted), self.assertRaisesRegex(OSError, "finalization"):
            self.train("repair")
        run = self.root / "repair"
        self.assertEqual(self.status(run)["state"], "training")
        workflow.train(run_dir=run, resume=True, interactive=False, repo_root=self.repo)
        self.assertEqual(self.status(run)["state"], "trained")
        self.assertTrue((run / "best_model.pt").exists())

    def test_plot_failure_leaves_training_finalization_recoverable(self):
        with patch.object(workflow, "plot_learning_history", side_effect=OSError("plot failed")), self.assertRaisesRegex(OSError, "plot failed"):
            self.train("plot_repair")
        run = self.root / "plot_repair"
        self.assertEqual(self.status(run)["state"], "training")
        workflow.train(run_dir=run, resume=True, interactive=False, repo_root=self.repo)
        self.assertEqual(self.status(run)["state"], "trained")
        self.assertTrue((run / "figures/learning_curve.pdf").exists())

    def test_configuration_source_and_frozen_model_guards(self):
        run = self.train("guards", max_chunks=1)
        with patch.object(workflow, "_source_hashes", return_value={"changed": "source"}), self.assertRaisesRegex(ValueError, "source changed"):
            workflow.train(run_dir=run, resume=True, interactive=False, repo_root=self.repo)
        config = json.loads((run / "config.json").read_text())
        config["training"]["torch_threads"] = 2
        (run / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            workflow.train(run_dir=run, resume=True, interactive=False, repo_root=self.repo)
        config["training"]["torch_threads"] = 1
        (run / "config.json").write_text(json.dumps(config))
        workflow.train(run_dir=run, resume=True, interactive=False, repo_root=self.repo)
        with (run / "best_model.pt").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            workflow.evaluate(run, repo_root=self.repo)

    def test_outputs_cannot_escape_ignored_root(self):
        with self.assertRaisesRegex(ValueError, "must be below"):
            workflow.train(self.config, run_dir=self.repo / "published-results", repo_root=self.repo)


if __name__ == "__main__":
    unittest.main()
