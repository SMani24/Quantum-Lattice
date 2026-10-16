"""Operational validation-menu controls; no research payloads are opened."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from quantum_lattice.simulation import validation_terminal as terminal


RUNNER_PATH = Path(__file__).resolve().parents[2]/"experiments/estimator_validation/run.py"
_SPEC = importlib.util.spec_from_file_location("validation_menu_runner_test", RUNNER_PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class WorkerPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)

    def test_default_saved_and_explicit_precedence(self):
        self.assertEqual(terminal.resolve_workers(self.run, 4), 4)
        terminal.save_workers(self.run, 12)
        self.assertEqual(terminal.resolve_workers(self.run, 4), 12)
        self.assertEqual(terminal.resolve_workers(self.run, 4, 8), 8)
        self.assertEqual(json.loads((self.run/"runtime-settings.json").read_text()), {"workers": 12})

    def test_preference_does_not_rewrite_scientific_files(self):
        originals = {"plan.json": b'{"execution":{"workers":4}}\n',
                     "plan-sha256.json": b'{"sha256":"fixed"}\n',
                     "config.json": b'{"seed":123,"workers":4}\n',
                     "status.json": b'{"state":"paused","completed":8230}\n'}
        for name, value in originals.items():
            (self.run/name).write_bytes(value)
        terminal.save_workers(self.run, 16)
        self.assertEqual({name: (self.run/name).read_bytes() for name in originals}, originals)
        self.assertEqual(set(p.name for p in self.run.iterdir()), set(originals)|{"runtime-settings.json"})

    def test_invalid_counts_are_rejected_without_writing(self):
        for count in (0, -1, True, False, 1.5, "8", None):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    terminal.save_workers(self.run, count)
                self.assertFalse((self.run/"runtime-settings.json").exists())

    def test_invalid_explicit_and_default_counts_are_rejected(self):
        for count in (0, -1, True, 2.5, "8"):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    terminal.resolve_workers(self.run, 4, count)
                with self.assertRaises(ValueError):
                    terminal.resolve_workers(self.run, count)

    def test_bad_saved_settings_require_recovery_or_override(self):
        for value in ("{broken", "null", "[]", "{}", '{"workers":0}', '{"workers":true}', '{"workers":"8"}'):
            with self.subTest(value=value):
                (self.run/"runtime-settings.json").write_text(value)
                with self.assertRaisesRegex(ValueError, "Saved CPU worker setting is invalid"):
                    terminal.resolve_workers(self.run, 4)
                self.assertEqual(terminal.resolve_workers(self.run, 4, 8), 8)

    def test_failed_atomic_replace_preserves_saved_preference(self):
        terminal.save_workers(self.run, 8)
        original = (self.run/"runtime-settings.json").read_bytes()
        with patch.object(terminal.os, "replace", side_effect=OSError("synthetic replace failure")):
            with self.assertRaises(OSError):
                terminal.save_workers(self.run, 12)
        self.assertEqual((self.run/"runtime-settings.json").read_bytes(), original)
        self.assertFalse(list(self.run.glob(".runtime-*.tmp")))


class MenuTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)

    def invoke(self, actions, start=None, **kwargs):
        output = io.StringIO()
        start = Mock() if start is None else start
        with patch("builtins.input", side_effect=actions), contextlib.redirect_stdout(output):
            result = terminal.dashboard(self.run, default_workers=4, start=start, **kwargs)
        return result, output.getvalue(), start

    def test_pause_change_workers_and_continue_returns_to_menu(self):
        def start(workers):
            (self.run/"status.json").write_text(json.dumps({"state": "paused", "completed": workers,
                                                          "total": 20, "pause_reason": "keyboard interrupt"}))
        callback = Mock(side_effect=start)
        result, output, _ = self.invoke(["w", "8", "1", "2", "12", "1", "q"], start=callback)
        self.assertEqual(result, 0)
        self.assertEqual([call.args for call in callback.call_args_list], [(8,), (12,)])
        self.assertEqual(terminal.resolve_workers(self.run, 4), 12)
        self.assertIn("8/20 tasks", output)
        self.assertIn("keyboard interrupt", output)
        self.assertIn("Saved 12 CPU workers", output)

    def test_saved_preference_and_cli_override_are_displayed(self):
        terminal.save_workers(self.run, 12)
        _, output, callback = self.invoke(["1", "q"], workers_override=8)
        callback.assert_called_once_with(8)
        self.assertIn("CPU workers: 8", output)
        self.assertEqual(terminal.resolve_workers(self.run, 4), 12)
        _, output, _ = self.invoke(["q"])
        self.assertIn("CPU workers: 12", output)

    def test_invalid_worker_entries_preserve_previous_value(self):
        _, output, callback = self.invoke(["w", "0", "w", "-2", "w", "no", "w", "1.5", "1", "q"])
        callback.assert_called_once_with(4)
        self.assertEqual(output.count("Worker count unchanged:"), 4)
        self.assertFalse((self.run/"runtime-settings.json").exists())

    def test_empty_worker_entry_keeps_count(self):
        _, _, callback = self.invoke(["w", "", "1", "q"])
        callback.assert_called_once_with(4)
        self.assertFalse((self.run/"runtime-settings.json").exists())

    def test_bad_saved_preference_falls_back_and_can_be_repaired(self):
        (self.run/"runtime-settings.json").write_text("not-json")
        _, output, callback = self.invoke(["w", "8", "1", "q"])
        callback.assert_called_once_with(8)
        self.assertIn("Saved CPU worker setting is invalid", output)
        self.assertEqual(terminal.resolve_workers(self.run, 4), 8)

    def test_eof_and_keyboard_interrupt_at_both_prompts_exit(self):
        for error in (EOFError, KeyboardInterrupt):
            for actions in ([error], ["w", error]):
                with self.subTest(error=error, actions=actions):
                    result, _, callback = self.invoke(actions)
                    self.assertEqual(result, 0)
                    callback.assert_not_called()

    def test_start_errors_return_to_menu_for_retry(self):
        for error in (ValueError("bad plan"), RuntimeError("already running"), OSError("disk"), ImportError("backend")):
            with self.subTest(error=error):
                callback = Mock(side_effect=[error, {"state": "paused"}])
                result, output, _ = self.invoke(["1", "1", "q"], start=callback)
                self.assertEqual(result, 0)
                self.assertEqual(callback.call_count, 2)
                self.assertIn(f"Cannot continue: {error}", output)

    def test_complete_campaign_does_not_start(self):
        (self.run/"status.json").write_text(json.dumps({"state": "complete", "completed": 20, "total": 20}))
        _, output, callback = self.invoke(["1", "3", "q"])
        callback.assert_not_called()
        self.assertIn("This campaign is complete", output)
        self.assertIn(str(self.run/"RESULTS.md"), output)
        self.assertIn(str(self.run/"execution.log"), output)

    def test_unknown_action_does_not_dispatch(self):
        _, output, callback = self.invoke(["bad", "q"])
        callback.assert_not_called()
        self.assertIn("Choose 1, 2, 3 or q.", output)

    def test_running_and_paused_status_schemas(self):
        cases = [({"state": "running", "completed_selected": 10, "total_selected": 30,
                   "stage": "multistart", "last_task": {"key": "synthetic"}},
                  ("running", "10/30 tasks", "selected stage: multistart")),
                 ({"state": "paused", "completed": 8, "total": 20,
                   "by_stage": {"saved": {"completed": 8, "total": 12}},
                   "pause_reason": "keyboard interrupt"},
                  ("paused", "8/20 tasks", "saved: 8/12", "keyboard interrupt"))]
        for status, strings in cases:
            with self.subTest(status=status):
                (self.run/"status.json").write_text(json.dumps(status))
                _, output, callback = self.invoke(["3", "q"])
                callback.assert_not_called()
                for value in strings:
                    self.assertIn(value, output)

    def test_missing_and_unreadable_json_status_are_inspectable(self):
        _, output, callback = self.invoke(["3", "q"])
        self.assertIn("not started", output)
        callback.assert_not_called()
        for content in ("{broken", "[]", "null"):
            with self.subTest(content=content):
                (self.run/"status.json").write_text(content)
                _, output, callback = self.invoke(["3", "q"])
                self.assertIn("Cannot read saved progress", output)
                callback.assert_not_called()


class RunnerRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/"campaign"
        self.run.mkdir()
        self.config = self.root/"config.json"
        self.config.write_text(json.dumps({"output_root": "campaign"}))
        self.plan = {"config": {"execution": {"workers": 4}}, "tasks": []}

    @contextlib.contextmanager
    def routing(self, stdin_tty, stdout_tty):
        stdin = Mock()
        stdin.isatty.return_value = stdin_tty
        stdout = io.StringIO()
        stdout.isatty = lambda: stdout_tty
        with patch.object(runner, "ROOT", self.root), patch.object(runner.sys, "stdin", stdin), \
             patch.object(runner.sys, "stdout", stdout), \
             patch.object(runner, "prepare", return_value=self.run) as prepare, \
             patch.object(runner, "load_plan", return_value=self.plan) as load_plan, \
             patch.object(runner, "dashboard", return_value=0) as dashboard, \
             patch.object(runner, "launch") as launch:
            yield prepare, load_plan, dashboard, launch

    def arguments(self, *args):
        return [*args, "--config", str(self.config)]

    def test_default_terminal_routes_to_menu_without_launch(self):
        with self.routing(True, True) as (prepare, load_plan, dashboard, launch):
            self.assertEqual(runner.main(self.arguments()), 0)
            prepare.assert_called_once()
            load_plan.assert_called_once_with(self.root, self.run)
            dashboard.assert_called_once()
            launch.assert_not_called()

    def test_redirected_input_or_output_preserves_one_shot_run(self):
        for stdin_tty, stdout_tty in ((False, True), (True, False), (False, False)):
            with self.subTest(stdin_tty=stdin_tty, stdout_tty=stdout_tty):
                with self.routing(stdin_tty, stdout_tty) as (_, _, dashboard, launch):
                    self.assertEqual(runner.main(self.arguments()), 0)
                    dashboard.assert_not_called()
                    launch.assert_called_once_with(self.run, 4, "all")

    def test_explicit_menu_routes_even_when_redirected(self):
        with self.routing(False, False) as (_, _, dashboard, launch):
            self.assertEqual(runner.main(self.arguments("menu", "--workers", "12", "--stage", "saved")), 0)
            dashboard.assert_called_once()
            kwargs = dashboard.call_args.kwargs
            self.assertEqual(kwargs["default_workers"], 4)
            self.assertEqual(kwargs["workers_override"], 12)
            self.assertEqual(kwargs["stage"], "saved")
            launch.assert_not_called()
            kwargs["start"](8)
            launch.assert_called_once_with(self.run, 8, "saved")

    def test_explicit_run_never_routes_to_menu(self):
        terminal.save_workers(self.run, 12)
        with self.routing(True, True) as (_, _, dashboard, launch):
            self.assertEqual(runner.main(self.arguments("run")), 0)
            dashboard.assert_not_called()
            launch.assert_called_once_with(self.run, 12, "all")
        with self.routing(True, True) as (_, _, dashboard, launch):
            self.assertEqual(runner.main(self.arguments("run", "--workers", "8", "--stage", "iid")), 0)
            launch.assert_called_once_with(self.run, 8, "iid")
        self.assertEqual(terminal.resolve_workers(self.run, 4), 12)

    def test_existing_run_does_not_prepare_new_campaign(self):
        with self.routing(True, True) as (prepare, load_plan, dashboard, _):
            self.assertEqual(runner.main(self.arguments("menu", "--run-dir", str(self.run))), 0)
            prepare.assert_not_called()
            load_plan.assert_called_once_with(self.root, self.run)
            dashboard.assert_called_once()

    def test_nonpositive_cli_workers_fail_before_plan_loading(self):
        with self.routing(True, True) as (prepare, load_plan, dashboard, launch):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    runner.main(self.arguments("run", "--workers", "0"))
            self.assertEqual(raised.exception.code, 2)
            prepare.assert_not_called()
            load_plan.assert_not_called()
            dashboard.assert_not_called()
            launch.assert_not_called()

    def test_launch_keeps_existing_campaign_api_and_log(self):
        expected = {"state": "paused", "completed": 2, "total": 5}
        with patch.object(runner, "run_campaign", return_value=expected) as run_campaign, \
             contextlib.redirect_stdout(io.StringIO()):
            result = runner.launch(self.run, 12, "multistart")
        self.assertEqual(result, expected)
        run_campaign.assert_called_once_with(runner.ROOT, self.run, workers=12, stage="multistart")
        self.assertIn("paused: 2/5 tasks.", (self.run/"execution.log").read_text())


if __name__ == "__main__":
    unittest.main()
