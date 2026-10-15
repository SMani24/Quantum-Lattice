"""Menu/config controls without starting any real training."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mlp_runner", REPO / "experiments/baseline_mlp/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MLPCliTests(unittest.TestCase):
    def test_default_early_and_explicit_fixed_budget(self):
        early = runner._config(runner.DEFAULT_CONFIG)
        fixed = runner._config(runner.DEFAULT_CONFIG, stopping="fixed")
        self.assertTrue(early["training"]["early_stopping"]["enabled"])
        configured = json.loads(runner.DEFAULT_CONFIG.read_text())["training"]["max_steps"]
        self.assertEqual(early["training"]["max_steps"], configured)
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

    def test_selection_defaults_to_ready_sizes_and_both_modes(self):
        ready = {3: {"state": "complete"}, 4: {"state": "complete"}, 8: {"state": "unverified"}}
        self.assertEqual(runner._size_selection("", ready), (3, 4))
        self.assertEqual(runner._size_selection("4,3", ready), (4, 3))
        for choice in ("4,4", "8", "5"):
            with self.assertRaises(ValueError):
                runner._size_selection(choice, ready)
        self.assertEqual(runner._modes_selection(""), ("paper_size", "all_edges"))
        self.assertEqual(runner._modes_selection("a"), ("all_edges",))

    def test_simple_menu_default_training_starts_one_queue(self):
        workflow = Mock()
        ready = {4: {"state": "complete"}, 8: {"state": "complete"}}
        with patch.object(runner.mlp_queue, "discover_ready_sizes", return_value=ready), \
             patch.object(runner.mlp_queue, "run_queue", return_value=Path("queue")) as train, \
             patch.object(runner, "_show_queue") as show, \
             patch("builtins.input", side_effect=["1", "", "", "q"]), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            runner._menu(workflow, runner.DEFAULT_CONFIG)
        self.assertEqual(train.call_count, 1)
        self.assertEqual(train.call_args.kwargs, {"sizes": (4, 8), "modes": ("paper_size", "all_edges")})
        show.assert_called_once_with(Path("queue"))
        workflow.evaluate.assert_not_called()
        self.assertIn("1 Train all  |  2 Continue  |  3 Results  |  4 Advanced  |  q Quit", output.getvalue())
        self.assertNotIn("8  Stopping strategy", output.getvalue())

    def test_train_all_cli_passes_selection_without_evaluating(self):
        with patch.object(runner.mlp_queue, "discover_ready_sizes", return_value={4: {"state": "complete"}}), \
             patch.object(runner.mlp_queue, "run_queue", return_value=Path("queue")) as train, \
             patch.object(runner.mlp_queue, "evaluate_queue") as evaluate, \
             patch.object(runner, "_show_queue"):
            self.assertEqual(runner.main(["train-all", "--sizes", "4", "8", "--modes", "all_edges", "--plain"]), 0)
        self.assertEqual(train.call_args.kwargs["sizes"], [4, 8])
        self.assertEqual(train.call_args.kwargs["modes"], ["all_edges"])
        self.assertFalse(train.call_args.kwargs["interactive"])
        evaluate.assert_not_called()

    def test_resume_status_and_explicit_queue_evaluation_cli(self):
        with patch.object(runner.mlp_queue, "run_queue", return_value=Path("queue")) as train, \
             patch.object(runner.mlp_queue, "evaluate_queue") as evaluate, \
             patch.object(runner, "_show_queue") as show:
            self.assertEqual(runner.main(["resume-queue", "--queue-dir", "/queue", "--plain"]), 0)
            train.assert_called_once_with(queue_dir=Path("/queue"), resume=True, interactive=False)
            self.assertEqual(runner.main(["queue-status", "--queue-dir", "/queue"]), 0)
            evaluate.assert_not_called()
            self.assertEqual(runner.main(["evaluate-queue", "--queue-dir", "/queue"]), 0)
            evaluate.assert_called_once_with(Path("/queue"))
            self.assertEqual(show.call_count, 3)

    def test_discovery_excludes_nested_smoke_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real, fixture = root / "real", root / "implementation/synthetic_run"
            for path in (real, fixture):
                path.mkdir(parents=True)
                (path / "config.json").write_text(json.dumps({"lattice_size": 4, "mode": "paper_size"}))
                (path / "status.json").write_text(json.dumps({"state": "trained"}))
            self.assertEqual([record[0] for record in runner._saved_runs(root)], [real])


if __name__ == "__main__":
    unittest.main()
