"""No-load fixtures for first-fault latching and bounded child supervision."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from quantum_lattice.models.final_worker import (
    EXPECTED_ONLINE, _guard, _settings, _supervise, supervise)


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def snapshot(**changes):
    return {"boot_id": "boot-fixture", "online": EXPECTED_ONLINE,
            "boost": 0, "affinity": EXPECTED_ONLINE, "cpu_c": 40.0,
            "gpu_c": 42.0, "gpu_uuid": "GPU-fixture", "compute_processes": [],
            "swap_used_kib": 0, "kernel_cursor": "cursor-fixture",
            "kernel_faults": [], **changes}


class Probe:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def snapshot(self, cursor=None):
        value = self.values[min(self.calls, len(self.values)-1)]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


class Child:
    pid = 123456

    def __init__(self, clock, *, finish_at=None, exit_on_int=True, code=0):
        self.clock = clock
        self.finish_at = finish_at
        self.exit_on_int = exit_on_int
        self.code = code
        self.signals = []
        self.forced_code = None

    def poll(self):
        if self.forced_code is not None:
            return self.forced_code
        if self.finish_at is not None and self.clock.value >= self.finish_at:
            return self.code
        return None

    def send(self, signum):
        self.signals.append((self.clock.value, signum))
        if signum == signal.SIGINT and self.exit_on_int:
            self.finish_at = self.clock.value+1
        elif signum == signal.SIGTERM:
            self.finish_at = self.clock.value+.5
            self.code = -signal.SIGTERM
        elif signum == signal.SIGKILL:
            self.forced_code = -signal.SIGKILL

    def wait(self, timeout):
        result = self.poll()
        if result is not None:
            return result
        if self.finish_at is not None and self.finish_at-self.clock.value <= timeout:
            self.clock.value = self.finish_at
            return self.poll()
        raise subprocess.TimeoutExpired(["fixture"], timeout)


class WorkerSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ql-final-worker-", dir="/tmp")
        self.directory = Path(self.temp.name)
        self.clock = Clock()
        self.child = Child(self.clock, finish_at=1)
        self.spawns = []
        self.settings = _settings({"gpu_uuid": "GPU-fixture", "drain_seconds": 2,
                                   "poll_seconds": .5})
        # These fixtures test control flow and persisted JSON content, not the
        # host disk's sync latency; real durability is used in production.
        self.sync = patch("quantum_lattice.models.final_worker.os.fsync")
        self.sync.start()

    def tearDown(self):
        self.sync.stop()
        self.temp.cleanup()

    def run_fixture(self, values=None, *, max_wall=100, stop_requested=None, popen=None):
        def launch(*args, **kwargs):
            self.spawns.append((args, kwargs))
            return self.child
        return _supervise(["fixture-not-executed"], self.directory, self.directory,
            max_wall, self.settings, self.directory/"PAUSE", probe=Probe(values or [snapshot()]),
            clock=self.clock, popen=popen or launch,
            send_signal=lambda child, signum: child.send(signum), stop_requested=stop_requested)

    def test_clean_child_runs_once_with_thread_limits_and_no_retry(self):
        value = self.run_fixture()
        self.assertEqual((value["state"], value["launch_count"], value["returncode"]), ("complete", 1, 0))
        self.assertEqual(len(self.spawns), 1)
        options = self.spawns[0][1]
        self.assertTrue(options["start_new_session"])
        self.assertEqual(options["env"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(value["restart_policy"], "never")
        self.assertEqual(self.child.signals, [])

    def test_child_failure_is_recorded_without_automatic_restart(self):
        self.child.code = 17
        value = self.run_fixture()
        self.assertEqual((value["state"], value["returncode"]), ("failed_child", 17))
        self.assertEqual(len(self.spawns), 1)
        self.assertEqual(value["launch_count"], 1)

    def test_first_guard_latches_and_drains_without_overwriting_reason(self):
        self.child.finish_at = None
        value = self.run_fixture([snapshot(), snapshot(cpu_c=81.0), snapshot(gpu_c=84.0)])
        self.assertEqual(value["state"], "failed_guard")
        self.assertEqual(value["first_guard"]["reason"], "cpu_temperature")
        self.assertEqual(value["stop_reason"], "cpu_temperature")
        self.assertEqual([sig for _, sig in self.child.signals], [signal.SIGINT])
        self.assertEqual(len(self.spawns), 1)

    def test_severe_heat_and_probe_failure_escalate_without_long_drain(self):
        for value_after_start, expected in [(snapshot(cpu_c=91.0), "severe_heat"),
                                             (RuntimeError("journal unavailable"), "supervisor_probe_failed")]:
            with self.subTest(expected=expected):
                self.clock.value = 0
                self.child = Child(self.clock, finish_at=None, exit_on_int=False)
                value = self.run_fixture([snapshot(), value_after_start])
                self.assertEqual(value["state"], "failed_guard")
                self.assertEqual(value["first_guard"]["reason"], expected)
                self.assertTrue(value["first_guard"]["severe"])
                self.assertEqual([sig for _, sig in self.child.signals], [signal.SIGINT, signal.SIGTERM])
                self.assertLess(value["active_wall_seconds"], 3)

    def test_wall_budget_reserves_drain_and_kills_unresponsive_child(self):
        self.child = Child(self.clock, finish_at=None, exit_on_int=False)
        value = self.run_fixture(max_wall=23.5)
        self.assertEqual(value["stop_reason"], "wall_budget")
        self.assertEqual(value["state"], "failed_drain")
        self.assertTrue(value["forced_kill"])
        self.assertEqual([sig for _, sig in self.child.signals], [signal.SIGINT, signal.SIGKILL])
        self.assertLessEqual(self.child.signals[0][0], 1.5)
        self.assertLessEqual(value["active_wall_seconds"], 23.5)
        self.assertEqual(len(self.spawns), 1)

    def test_missing_guard_busy_gpu_or_changed_workaround_refuses_launch(self):
        incomplete = snapshot()
        del incomplete["kernel_cursor"]
        alternatives = [incomplete, snapshot(kernel_cursor=""), snapshot(boost=1),
            snapshot(online=list(range(12))), snapshot(swap_used_kib=1),
            snapshot(compute_processes=[{"pid": 42, "gpu_uuid": "GPU-fixture"}])]
        for value in alternatives:
            with self.subTest(value=value):
                result = self.run_fixture([value])
                self.assertEqual(result["state"], "refused")
        self.assertFalse(self.spawns)
        self.assertFalse(self.child.signals)

    def test_preexisting_pause_prevents_launch_and_signal_flag_pauses_child(self):
        (self.directory/"PAUSE").touch()
        result = self.run_fixture()
        self.assertEqual((result["state"], result["launch_count"]), ("paused", 0))
        self.assertFalse(self.spawns)
        (self.directory/"PAUSE").unlink()
        self.child.finish_at = None
        result = self.run_fixture(stop_requested=lambda: self.clock.value >= .5)
        self.assertEqual(result["state"], "paused")
        self.assertEqual(result["stop_reason"], "user_pause")
        self.assertEqual([sig for _, sig in self.child.signals], [signal.SIGINT])

    def test_launch_failure_and_recording_failure_leave_no_unmonitored_child(self):
        def launch_failure(*args, **kwargs):
            raise OSError("missing executable")
        result = self.run_fixture(popen=launch_failure)
        self.assertEqual((result["state"], result["launch_count"]), ("failed_launch", 0))
        self.child.finish_at = None
        from quantum_lattice.models import final_worker
        original = final_worker._json
        def fail_running_record(path, value):
            if value.get("state") == "running":
                raise OSError("disk full")
            return original(path, value)
        with patch.object(final_worker, "_json", fail_running_record):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_fixture()
        self.assertIn(signal.SIGTERM, [sig for _, sig in self.child.signals])
        self.assertIsNotNone(self.child.poll())

    def test_guard_settings_cannot_weaken_isolation_or_limits(self):
        for change in [{"expected_boost": 1}, {"expected_online": list(range(12))},
                       {"max_cpu_c": 81}, {"max_gpu_c": 83}, {"poll_seconds": 11},
                       {"require_zero_swap": False}, {"drain_seconds": 301}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                _settings({"gpu_uuid": "GPU-fixture", **change})
        with self.assertRaises(ValueError):
            supervise(["unused"], self.directory, self.directory/"unsafe", 1000, {"gpu_uuid": "GPU-fixture"})
        with self.assertRaises(ValueError):
            supervise(["unused"], self.directory, self.directory/"unsafe", 4*24*3600+1, {"gpu_uuid": "GPU-fixture"})

    def test_public_signal_handler_requests_pause_and_restores_process_settings(self):
        from quantum_lattice.models import final_worker
        run = self.directory/"codex material/artifacts/final_benchmark/supervisor"
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        original_affinity = {0, 1}
        def engine(*args, **kwargs):
            self.assertFalse(kwargs["stop_requested"]())
            signal.raise_signal(signal.SIGTERM)
            self.assertTrue(kwargs["stop_requested"]())
            return {"state": "paused", "stop_reason": "user_pause"}
        with patch.object(final_worker.os, "sched_getaffinity", return_value=original_affinity), \
                patch.object(final_worker.os, "sched_setaffinity") as affinity, \
                patch.object(final_worker, "_supervise", side_effect=engine):
            result = supervise(["fixture-never-executed"], self.directory, run, 1000,
                               {"gpu_uuid": "GPU-fixture"})
        self.assertEqual(result["state"], "paused")
        self.assertEqual(affinity.call_args_list[0].args, (0, EXPECTED_ONLINE))
        self.assertEqual(affinity.call_args_list[-1].args, (0, original_affinity))
        self.assertTrue(all(signal.getsignal(sig) == old for sig, old in handlers.items()))
        self.assertEqual(json.loads((run/"supervisor-state.json").read_text())["state"], "paused")


if __name__ == "__main__":
    unittest.main()
