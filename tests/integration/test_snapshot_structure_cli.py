"""Read-only structure menu and bounded real-CUDA queue/resume controls."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_snapshot_scaling_engine import fixture as scaling_fixture
from quantum_lattice.models import snapshot_scaling as scaling
from quantum_lattice.models import snapshot_structure_campaign as campaign
from quantum_lattice.models.snapshot_structure_terminal import main, print_status


REPO = Path(__file__).resolve().parents[2]


def fixture(root):
    """Reuse synthetic saved configurations and a tiny completed inverse group."""
    source = scaling_fixture(root)
    scaling.prepare(root, source)
    config = json.loads((REPO / "configs/snapshot_structure.json").read_text())
    config["source_campaign"] = source["output_root"]
    config["output_root"] = "codex material/artifacts/snapshot_structure/campaigns/synthetic-cli"
    config["selection"] = {"selection_seed": 813, "train_hamiltonians": 2,
                           "validation_hamiltonians": 2, "shots": 4}
    config["seeds"] = [617]
    config["training"] = copy.deepcopy(source["training"])
    config["inverse_ising"] = copy.deepcopy(source["inverse_ising"])
    config["inverse_reference"] = copy.deepcopy(source["inverse_reference"])
    config["execution"] = {"checkpoint_updates": 1}
    config["diagnostics"] = {"gauge_masks": 2, "gauge_seed": 791}
    return config


def model_hashes(run):
    return {str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (run / "models").rglob("*") if path.is_file()}


class SnapshotStructureCliTests(unittest.TestCase):
    def test_status_and_menu_are_read_only_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            path = root / "structure.json"
            path.write_text(json.dumps(config))
            run = root / config["output_root"]
            output = io.StringIO()
            with contextlib.redirect_stdout(output), \
                    patch("torch.cuda.is_available", side_effect=AssertionError("Status needs no CUDA")):
                self.assertEqual(main(["--config", str(path), "--status"], repo_root=root), 0)
            self.assertIn("0/8", output.getvalue())
            self.assertIn("4 records/H", output.getvalue())
            self.assertIn("Checkpoint interval: 1", output.getvalue())
            self.assertFalse(run.exists())
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("builtins.input", side_effect=["2", "0"]):
                self.assertEqual(main(["--config", str(path)], repo_root=root), 0)
            self.assertFalse(run.exists())

    def test_prepare_without_training_and_grouped_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            path = root / "structure.json"
            path.write_text(json.dumps(config))
            previous = signal.getsignal(signal.SIGINT)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("torch.cuda.is_available", side_effect=AssertionError("Preparation needs no CUDA")):
                self.assertEqual(main(["--config", str(path), "--prepare"], repo_root=root), 0)
            self.assertEqual(signal.getsignal(signal.SIGINT), previous)
            run = root / config["output_root"]
            status = campaign.progress(root, config)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["total"], 8)
            self.assertFalse((run / "models").exists())
            self.assertFalse((run / "device-binding.json").exists())
            output = io.StringIO()
            print_status(status, stream=output)
            self.assertIn("Residual", output.getvalue())
            self.assertIn("Gauge", output.getvalue())
            self.assertIn("raw-set CNN", output.getvalue())
            self.assertIn("Inverse baseline: verified reference", output.getvalue())
            # Existing paths remain untouched when the alias repeats preparation.
            plan_before = (run / "plan.json").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(path), "--prepare-only"], repo_root=root), 0)
            self.assertEqual((run / "plan.json").read_bytes(), plan_before)

    @unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA required")
    def test_actual_cuda_pause_resume_all_arms_and_verified_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root)
            previous = signal.getsignal(signal.SIGINT)
            interrupted = []

            def pause_once(event):
                if event.get("stage") == "training" and not interrupted:
                    interrupted.append(True)
                    os.kill(os.getpid(), signal.SIGINT)

            status = campaign.run_campaign(root, config, progress_callback=pause_once)
            self.assertEqual(signal.getsignal(signal.SIGINT), previous)
            self.assertTrue(interrupted)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["tasks"][0]["state"], "paused")
            run = root / config["output_root"]
            binding = json.loads((run / "device-binding.json").read_text())
            self.assertEqual(binding["accelerator"]["type"], "cuda")
            path = root / "structure.json"
            path.write_text(json.dumps(config))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--config", str(path), "--run"], repo_root=root), 0)
            self.assertEqual(campaign.progress(root, config)["completed"], 8)
            self.assertIn("Completed seeds: 617:", output.getvalue())
            self.assertIn("Bound GPU:", output.getvalue())
            before = model_hashes(run)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(path), "--run"], repo_root=root), 0)
            self.assertEqual(before, model_hashes(run))
            plan = campaign.load_plan(root, run)
            self.assertEqual(len(plan["tasks"]), 8)
            self.assertEqual({(task["method"], task["residual"], task["gauge_augmentation"])
                              for task in plan["tasks"]},
                             {(method, residual, gauge) for method in ("mean_cnn", "set_cnn")
                              for residual in (False, True) for gauge in (False, True)})


if __name__ == "__main__":
    unittest.main()
