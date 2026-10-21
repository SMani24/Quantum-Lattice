"""Read-only menu and real-CUDA end-to-end scaling on tiny synthetic data."""
import contextlib
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
from test_snapshot_scaling_engine import fixture
from quantum_lattice.models import snapshot_scaling as campaign
from quantum_lattice.models.snapshot_scaling_terminal import main, print_status


def model_hashes(run):
    return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (run / "models").rglob("*") if p.is_file()}


class SnapshotScalingCliTests(unittest.TestCase):
    def test_status_and_simple_menu_do_not_prepare_or_require_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root)
            path = root / "local.json"
            path.write_text(json.dumps(config))
            run = root / config["output_root"]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--config", str(path), "--status"], repo_root=root), 0)
            self.assertIn("0/4", output.getvalue())
            self.assertIn("4 records/H", output.getvalue())
            self.assertFalse(run.exists())
            with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input", side_effect=["2", "0"]):
                self.assertEqual(main(["--config", str(path)], repo_root=root), 0)
            self.assertFalse(run.exists())

    def test_prepare_only_verifies_inverse_and_prints_results_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root)
            path = root / "local.json"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(path), "--prepare-only"], repo_root=root), 0)
            run = root / config["output_root"]
            status = campaign.progress(root, config)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["inverse_reference"]["state"], "complete")
            self.assertFalse((run / "models").exists())
            self.assertFalse((run / "device-binding.json").exists())
            output = io.StringIO()
            print_status(status, stream=output)
            self.assertIn("Inverse baseline: verified reference", output.getvalue())
            self.assertIn("raw-set CNN", output.getvalue())

    @unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA required")
    def test_actual_cuda_pause_resume_full_queue_and_verified_completed_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root)
            previous_handler = signal.getsignal(signal.SIGINT)
            interrupted = []

            def pause_once(event):
                if event.get("stage") == "training" and not interrupted:
                    interrupted.append(True)
                    os.kill(os.getpid(), signal.SIGINT)

            status = campaign.run_campaign(root, config, progress_callback=pause_once)
            self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)
            self.assertTrue(interrupted)
            self.assertEqual(status["completed"], 0)
            self.assertEqual(status["tasks"][0]["state"], "paused")
            run = root / config["output_root"]
            self.assertEqual(json.loads((run / "device-binding.json").read_text())["accelerator"]["type"], "cuda")
            path = root / "local.json"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(path), "--run"], repo_root=root), 0)
            self.assertEqual(campaign.progress(root, config)["completed"], 4)
            before = model_hashes(run)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(path), "--run"], repo_root=root), 0)
            self.assertEqual(before, model_hashes(run))
            plan = campaign.load_plan(root, run)
            self.assertEqual(len(plan["tasks"]), 4)
            self.assertEqual(plan["updates_per_epoch"], {"2": 1, "3": 2})


if __name__ == "__main__":
    unittest.main()
