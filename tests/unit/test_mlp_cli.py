"""Menu/config controls without starting any real training."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mlp_runner", REPO / "experiments/baseline_mlp/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MLPCliTests(unittest.TestCase):
    def test_default_early_and_explicit_fixed_budget(self):
        early = runner._config(runner.DEFAULT_CONFIG)
        fixed = runner._config(runner.DEFAULT_CONFIG, stopping="fixed")
        self.assertTrue(early["training"]["early_stopping"]["enabled"])
        self.assertEqual(early["training"]["max_steps"], 200000)
        self.assertFalse(fixed["training"]["early_stopping"]["enabled"])
        self.assertEqual(fixed["training"]["max_steps"], 10000)
        self.assertEqual(runner._config(runner.DEFAULT_CONFIG, stopping="fixed", max_steps=123)["training"]["max_steps"], 123)

    def test_saved_discovery_includes_paired_children_once_and_ignores_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "single", root / "campaigns/paired/paper_size/run", root / "campaigns/paired/all_edges/retry_001"]
            for path in paths:
                path.mkdir(parents=True)
                (path / "config.json").write_text(json.dumps({"lattice_size": 4, "mode": "paper_size"}))
                (path / "status.json").write_text(json.dumps({"state": "trained", "step": 8, "max_steps": 8}))
            (root / "latest").symlink_to(paths[0], target_is_directory=True)
            self.assertEqual({record[0] for record in runner._saved_runs(root)}, set(paths))

    def test_summary_does_not_open_checkpoint_or_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config.json").write_text(json.dumps({"lattice_size": 4, "mode": "all_edges", "training": {"max_steps": 100}}))
            (run / "status.json").write_text(json.dumps({"state": "trained", "step": 100, "max_steps": 100,
                                                       "best_step": 80, "best_validation_mse": .01}))
            (run / "latest.pt").write_bytes(b"intentionally not a valid checkpoint")
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                runner._show_run(run)
            self.assertIn("Best validation MSE: 0.01 at update 80", output.getvalue())
            self.assertIn("Final test: not evaluated", output.getvalue())
            self.assertIn(str(run / "RESULTS.md"), output.getvalue())


if __name__ == "__main__":
    unittest.main()
