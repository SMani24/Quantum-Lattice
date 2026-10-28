"""Cold CLI guards, pause recovery and completed scientific integrity checks."""
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "experiments/final_benchmark/run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("final_runner_cli_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def forbid_numerical_imports():
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "quantum_lattice"}:
            raise AssertionError(f"Unexpected numerical import on a cold metadata path: {name}")
        return original(name, *args, **kwargs)
    with mock.patch("builtins.__import__", checked):
        yield


def campaign_module():
    from quantum_lattice.models import final_campaign
    return final_campaign


class FinalRunnerCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with forbid_numerical_imports():
            self.runner = load_runner()
        self.runner.ROOT = self.root
        self.config = json.loads((REPO / "configs/final_benchmark.json").read_text())
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(self.config))

    def call_main(self, *arguments, answers=None):
        stack = contextlib.ExitStack()
        with stack:
            stack.enter_context(mock.patch("sys.argv", [str(RUNNER), "--config", str(self.path), *arguments]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            if answers is not None:
                stack.enter_context(mock.patch("builtins.input", side_effect=answers))
            return self.runner.main()

    def test_help_and_menu_exit_are_stdlib_only(self):
        with forbid_numerical_imports(), self.assertRaises(SystemExit) as exited:
            self.call_main("--help")
        self.assertEqual(exited.exception.code, 0)
        with forbid_numerical_imports():
            self.assertEqual(self.call_main(answers=["0"]), 0)
        self.assertFalse((self.root / self.config["output_root"]).exists())

    def test_invalid_worker_counts_fail_before_numerical_import_or_output_creation(self):
        for value in ("0", "11", "-1"):
            with self.subTest(workers=value), forbid_numerical_imports(), self.assertRaises(SystemExit) as exited:
                self.call_main("--run", "--workers", value)
            self.assertEqual(exited.exception.code, 2)
        self.assertFalse((self.root / self.config["output_root"]).exists())

    def test_unprepared_status_is_stdlib_only_and_does_not_create_a_run(self):
        with forbid_numerical_imports():
            self.assertEqual(self.call_main("--status"), 0)
        self.assertFalse((self.root / self.config["output_root"]).exists())

    def test_bad_reboot_cpu_topology_refuses_before_any_numerical_import(self):
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            if str(path) == "/sys/devices/system/cpu/online":
                return "0-11\n"  # The suspect paired threads have returned.
            if str(path) == "/sys/devices/system/cpu/cpufreq/boost":
                return "0\n"
            return original_read(path, *args, **kwargs)
        with mock.patch.object(Path, "read_text", read), \
             mock.patch.object(self.runner.os, "sched_setaffinity") as affinity, \
             forbid_numerical_imports(), self.assertRaisesRegex(RuntimeError, "isolation/boost"):
            self.call_main("--run")
        affinity.assert_not_called()
        self.assertFalse((self.root / self.config["output_root"]).exists())

    def test_qualified_topology_pins_affinity_before_importing_the_training_stack(self):
        original_read, original_import = Path.read_text, builtins.__import__
        events = []
        def read(path, *args, **kwargs):
            if str(path) == "/sys/devices/system/cpu/online":
                return "0-1,3-7,9-11\n"
            if str(path) == "/sys/devices/system/cpu/cpufreq/boost":
                return "0\n"
            return original_read(path, *args, **kwargs)
        def imported(name, *args, **kwargs):
            if name.startswith("quantum_lattice"):
                events.append("numerical_import")
                raise RuntimeError("Stop after verifying guard/import order")
            return original_import(name, *args, **kwargs)
        def pin(pid, cpus):
            self.assertEqual(pid, 0)
            self.assertEqual(set(cpus), {0, 1, 3, 4, 5, 6, 7, 9, 10, 11})
            events.append("affinity_pinned")
        with mock.patch.object(Path, "read_text", read), \
             mock.patch.object(self.runner.os, "sched_setaffinity", side_effect=pin), \
             mock.patch("builtins.__import__", imported), \
             self.assertRaisesRegex(RuntimeError, "guard/import order"):
            self.runner.start(self.config, str(self.path))
        self.assertEqual(events, ["affinity_pinned", "numerical_import"])

    def test_pause_before_prepare_does_not_strand_explicit_resume(self):
        campaign = campaign_module()
        c = json.loads(self.path.read_text())
        c["raw_panel"]["enabled"] = False
        campaign.request_pause(self.root, c)
        run = self.root / c["output_root"]
        class Ids(list):
            def tolist(self):
                return list(self)
        def features(cache, L):
            return {split: {"y": range(count), "f": None, "g": None,
                           "uids": Ids(f"L{L}-{split}-{i}" for i in range(count))}
                    for split, count in [("train", 7452), ("validation", 1242)]}
        with mock.patch.object(campaign, "feature_hashes", return_value={}), \
             mock.patch.object(campaign, "_input_metadata", return_value={}), \
             mock.patch.object(campaign, "sources", return_value={}), \
             mock.patch.object(campaign, "load_features", side_effect=features):
            self.assertEqual(campaign.prepare(self.root, c), run)
        self.assertTrue((run / "plan.json").is_file())
        self.assertFalse((run / "confirmation").exists())


class CompletedInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.campaign = campaign_module()
        self.plan = {"schema_version": 1, "config": {"fixture": True}, "tasks": []}
        self.seal = {"schema_version": 1, "prohibits_reselection": True, "model_sha256": {}}
        self.campaign.atomic_json(self.run / "confirmation-seal.json", self.seal)
        (self.run / "model-scores.csv").write_text("method,mse\nfixture,0.125\n")
        self.campaign.atomic_json(self.run / "paired-comparisons.json", {"fixture": "no scientific claim"})
        inventory = {"schema_version": 1, "plan_sha256": self.campaign.digest(self.plan),
            "files": {name: self.campaign.sha(self.run / name) for name in
                      ["model-scores.csv", "paired-comparisons.json", "confirmation-seal.json"]}}
        self.campaign.atomic_json(self.run / "integrity.json", inventory)
        self.campaign.atomic_json(self.run / "complete.json", {
            "schema_version": 1, "plan_sha256": self.campaign.digest(self.plan),
            "confirmation_seal_sha256": self.campaign.digest(self.seal),
            "integrity_sha256": self.campaign.sha(self.run / "integrity.json")})

    def test_completed_deep_verification_reads_without_rewriting_artifacts(self):
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        value = self.campaign.verify_completed(self.run, self.plan)
        self.assertEqual(value, {"verified": True, "deep": True, "scientific_files": 3})
        self.assertEqual({p.name: p.read_bytes() for p in self.run.iterdir()}, before)

    def test_tampered_completed_score_csv_is_detected_even_with_intact_inventory_file(self):
        before = (self.run / "integrity.json").read_bytes()
        (self.run / "model-scores.csv").write_text("method,mse\nfixture,0\n")
        self.assertEqual((self.run / "integrity.json").read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "model-scores.csv"):
            self.campaign.verify_completed(self.run, self.plan)

    def test_missing_completed_output_and_changed_model_seal_are_rejected(self):
        (self.run / "paired-comparisons.json").unlink()
        with self.assertRaisesRegex(ValueError, "paired-comparisons.json"):
            self.campaign.verify_completed(self.run, self.plan)
        self.campaign.atomic_json(self.run / "confirmation-seal.json", {**self.seal, "prohibits_reselection": False})
        with self.assertRaisesRegex(ValueError, "seal changed"):
            self.campaign.verify_completed(self.run, self.plan)

    def test_changed_inventory_bytes_are_rejected_before_any_success_claim(self):
        with (self.run / "integrity.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.campaign.verify_completed(self.run, self.plan)


if __name__ == "__main__":
    unittest.main()
