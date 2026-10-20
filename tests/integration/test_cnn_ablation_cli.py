"""Routing/menu checks substitute all research work."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cnn_ablation_cli", ROOT / "experiments/cnn_ablation/train.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CnnAblationCliTests(unittest.TestCase):
    def test_redirected_default_rejects_before_access(self):
        with mock.patch.object(cli.sys.stdin, "isatty", return_value=False), \
             mock.patch.object(cli, "prepare") as prepare, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main([])
        self.assertEqual(caught.exception.code, 2)
        prepare.assert_not_called()

    def test_explicit_variant_run(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(cli, "load_plan"), \
             mock.patch.object(cli, "launch") as launch:
            self.assertEqual(cli.main(["run", "--run-dir", folder, "--models", "batchnorm", "residual",
                                       "--max-chunks", "1"]), 0)
            launch.assert_called_once_with(Path(folder), models=["batchnorm", "residual"], max_chunks=1)

    def test_status_and_prepare_never_launch(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(cli, "load_plan"), \
             mock.patch.object(cli, "launch") as launch, \
             mock.patch.object(cli, "summarize_campaign", return_value={}), \
             mock.patch.object(cli, "format_summary", return_value="0/8 complete"), redirect_stdout(io.StringIO()):
            for command in ("status", "prepare"):
                self.assertEqual(cli.main([command, "--run-dir", folder]), 0)
        launch.assert_not_called()

    def test_bounds_and_unused_variants_rejected_before_access(self):
        with mock.patch.object(cli, "prepare") as prepare, redirect_stderr(io.StringIO()):
            for args in (["run", "--max-chunks", "0"], ["status", "--models", "baseline"]):
                with self.assertRaises(SystemExit):
                    cli.main(args)
        prepare.assert_not_called()

    def test_menu_run_status_and_quit(self):
        inputs = iter(["oops", "1", "2", "q"])
        start, status, write = mock.Mock(), mock.Mock(return_value="0/8 complete"), mock.Mock()
        self.assertEqual(cli.dashboard("/campaign", status=status, start=start,
                                      read=lambda _: next(inputs), write=write), 0)
        start.assert_called_once_with()
        self.assertEqual(status.call_count, 5)

    def test_idle_interrupt_exits(self):
        start = mock.Mock()
        self.assertEqual(cli.dashboard("/campaign", status=lambda: "", start=start,
            read=mock.Mock(side_effect=KeyboardInterrupt), write=lambda _: None), 0)
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
