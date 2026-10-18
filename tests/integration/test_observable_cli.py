"""CLI routing controls; research work is substituted, never launched here."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("observable_train_cli", ROOT / "experiments/observable_cnn/train.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ObservableCliTests(unittest.TestCase):
    def test_redirected_default_does_not_prepare_or_train(self):
        with mock.patch.object(cli.sys.stdin, "isatty", return_value=False), \
             mock.patch.object(cli, "prepare") as prepare, \
             mock.patch.object(cli, "launch") as launch, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main([])
            self.assertEqual(error.exception.code, 2)
        prepare.assert_not_called()
        launch.assert_not_called()

    def test_explicit_size_run_routes_without_menu(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(cli, "load_plan"), \
             mock.patch.object(cli, "launch") as launch, \
             mock.patch.object(cli, "dashboard") as menu:
            code = cli.main(["run", "--run-dir", directory, "--sizes", "4", "--max-chunks", "1"])
            self.assertEqual(code, 0)
            launch.assert_called_once_with(Path(directory), sizes=[4], max_chunks=1)
            menu.assert_not_called()

    def test_prepare_and_status_never_launch(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(cli, "load_plan"), \
             mock.patch.object(cli, "summarize_campaign", return_value={"completed": 0}), \
             mock.patch.object(cli, "format_summary", return_value="0/40 complete"), \
             mock.patch.object(cli, "launch") as launch, redirect_stdout(io.StringIO()):
            for command in ("prepare", "status"):
                self.assertEqual(cli.main([command, "--run-dir", directory]), 0)
            launch.assert_not_called()

    def test_bad_bounds_or_ignored_sizes_rejected_before_access(self):
        with mock.patch.object(cli, "prepare") as prepare, redirect_stderr(io.StringIO()):
            for arguments in (["run", "--max-chunks", "0"], ["status", "--sizes", "3"]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                    cli.main(arguments)
                self.assertEqual(error.exception.code, 2)
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
