"""Bounded CPU runner contracts; no saved research data or CUDA initialization."""

from copy import deepcopy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
if torch is not None:
    from quantum_lattice.data.mlp_data import EdgeRows, LearningData
    spec = importlib.util.spec_from_file_location(
        "observable_cnn_runner", ROOT / "experiments/observable_cnn/run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class ObservableCNNRunnerTests(unittest.TestCase):
    def config(self):
        return {
            "schema_version": 1,
            "scope": "CPU-only synthetic software check",
            "feature_cache": "not-a-real-cache",
            "architecture": {"hidden_channels": [2, 3], "kernel_size": 3},
            "checks": {"sizes": [3], "seed": 18, "synthetic_hamiltonians": 1,
                       "optimizer_steps": 1, "learning_rate": .001},
        }

    def rows(self, names, multiplier):
        """Independent canonical edge rows with matching repeated star values."""
        features, targets, ids, uids, edges = [], [], [], [], []
        for index, name in enumerate(names):
            stars = (np.arange(9, dtype=np.float32).reshape(3, 3) + 2 + index) * multiplier
            for y in range(3):
                for x in range(3):
                    for orientation in (0, 1):
                        edge = 2 * (3 * y + x) + orientation
                        end_y, end_x = ((y, (x + 1) % 3) if orientation == 0
                                        else ((y + 1) % 3, x))
                        features.append([stars[y, x], stars[end_y, end_x],
                                         multiplier * (.2 + .01 * edge)])
                        targets.append([.1 + .01 * edge])
                        ids.append(name)
                        uids.append(f"fixture-{name}")
                        edges.append(edge)
        return EdgeRows(np.array(features, dtype=np.float32),
                        np.array(targets, dtype=np.float32), np.array(ids),
                        np.array(uids), np.array(edges, dtype=np.int64))

    def views(self, directory):
        return {
            view: LearningData(
                self.rows(("train-a", "train-b"), scale),
                self.rows(("validation-a",), scale), None, 3, 0, "all_edges",
                {"feature_view": view, "training_eligible": view != "exact",
                 "fixture_metadata": {"groups": "Hamiltonian"}}, Path(directory),
            )
            for view, scale in (("f", .3), ("g", .08), ("exact", .07))
        }

    def test_tiny_check_is_cpu_only_and_never_loads_feature_cache(self):
        with patch.object(runner, "load_feature_views", side_effect=AssertionError("real cache accessed")), \
                patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
            result = runner.check(self.config())
        self.assertEqual(result["device"], "cpu")
        self.assertFalse(result["research_training_performed"])
        self.assertFalse(result["real_bank_loaded"])
        case, = result["cases"]
        self.assertEqual(case["input_shape"], [1, 3, 3, 3])
        self.assertEqual(case["target_shape"], [1, 2, 3, 3])
        self.assertTrue(case["roundtrip_exact"])
        self.assertLessEqual(case["original_vs_map_mlp_max_abs"], 1e-6)
        self.assertLessEqual(case["translation_max_abs"], 1e-5)
        self.assertEqual(case["models"]["cnn"]["parameter_count"], 121)
        self.assertEqual(case["models"]["local_mlp"]["parameter_count"], 39319)
        for result in case["models"].values():
            self.assertEqual(result["optimizer_steps"], 1)
            self.assertTrue(np.isfinite(result["initial_loss"]))
            self.assertTrue(np.isfinite(result["final_loss"]))

    def test_inspection_keeps_groups_exact_eligibility_and_no_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            manifest = directory / "manifest.json"
            manifest.write_text('{"fixture": "no real cache"}\n')
            views = self.views(directory)
            before = deepcopy(views)
            with patch.object(runner, "load_feature_views", return_value=views) as loader, \
                    patch.object(torch.optim, "Adam", side_effect=AssertionError("inspection trained")), \
                    patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")):
                result = runner.inspect(self.config(), 3, directory)
            loader.assert_called_once_with(directory, 3)
            self.assertEqual(len(result["cases"]), 6)
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertFalse(result["test_payload_loaded"])
            self.assertFalse(result["spin_records_loaded"])
            self.assertEqual(result["feature_manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
            for case in result["cases"]:
                count = 2 if case["split"] == "train" else 1
                self.assertEqual(case["hamiltonians"], count)
                self.assertEqual(case["edge_rows"], count * 18)
                self.assertTrue(case["roundtrip_exact"])
                self.assertTrue(case["source_information_unchanged"])
                self.assertEqual(case["training_eligible"], case["view"] != "exact")
                if case["view"] == "f":
                    self.assertGreater(case["maximum_input"], 1.)
            for view in views:
                self.assertEqual(views[view].provenance, before[view].provenance)
                for split in ("train", "validation"):
                    for attribute in ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
                        np.testing.assert_array_equal(getattr(getattr(views[view], split), attribute),
                                                      getattr(getattr(before[view], split), attribute))

    def test_inspection_rejects_inconsistent_star_copies_and_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            for corruption in ("star", "test", "leakage"):
                with self.subTest(corruption=corruption):
                    views = self.views(directory)
                    data = views["f"]
                    if corruption == "star":
                        data.train.x[0, 1] += .1
                    elif corruption == "test":
                        object.__setattr__(data, "test", data.validation)
                    else:
                        object.__setattr__(data, "validation", data.train)
                    with patch.object(runner, "load_feature_views", return_value=views), self.assertRaises(ValueError):
                        runner.inspect(self.config(), 3, Path(directory))

    def test_main_exports_effective_config_source_hashes_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(self.config()) + "\n")
            output = root / "artifacts/new-check"
            # Dispatch/export is checked separately from the real bounded
            # optimizer check above; no second gradient workload is needed.
            synthetic_result = {"scope": "CPU-only fixture", "cases": [], "device": "cpu"}
            with patch.object(runner, "ARTIFACT_ROOT", root / "artifacts"), \
                    patch.object(runner, "check", return_value=synthetic_result) as check, \
                    patch.object(runner, "inspect", side_effect=AssertionError("inspection dispatched")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["check", "--config", str(config), "--output", str(output)]), 0)
            check.assert_called_once_with(self.config())
            self.assertEqual(json.loads((output / "config.json").read_text()), self.config())
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertEqual(provenance["config_sha256"], hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertEqual(provenance["device"], "cpu")
            self.assertFalse(provenance["gpu_work"])
            self.assertFalse(provenance["research_training_performed"])
            self.assertIn("src/quantum_lattice/models/observable_cnn.py", provenance["source_sha256"])
            self.assertIn("src/quantum_lattice/data/observable_maps.py", provenance["source_sha256"])
            self.assertIn("experiments/observable_cnn/run.py", provenance["source_sha256"])
            for relative, digest in provenance["source_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((ROOT / relative).read_bytes()).hexdigest())
            report = (output / "RESULTS.md").read_text()
            self.assertIn("not quantum-learning accuracy", report)
            self.assertIn("No research training or GPU work", report)

    def test_main_inspect_dispatch_has_no_gradient_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps(self.config()))
            output = root / "artifacts/inspection"
            fixture = {"scope": "non-test fixture", "cases": [], "optimizer_steps": 0}
            cache = root / "synthetic-cache"
            with patch.object(runner, "ARTIFACT_ROOT", root / "artifacts"), \
                    patch.object(runner, "check", side_effect=AssertionError("gradient check dispatched")), \
                    patch.object(runner, "inspect", return_value=fixture) as inspect, \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.main(["inspect", "--L", "3", "--feature-dir", str(cache),
                             "--config", str(config), "--output", str(output)])
            inspect.assert_called_once_with(self.config(), 3, cache)
            self.assertEqual(json.loads((output / "checks.json").read_text())["optimizer_steps"], 0)

    def test_outputs_are_protected_before_any_check_or_cache_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            existing = artifacts / "existing"
            existing.mkdir(parents=True)
            sentinel = existing / "keep.txt"
            sentinel.write_bytes(b"preserve research bytes\n")
            config = root / "config.json"
            config.write_text(json.dumps(self.config()))
            escaped = root / "artifacts-elsewhere/new"
            (artifacts / "escape-link").symlink_to(root, target_is_directory=True)
            for command in ("check", "inspect"):
                for output in (existing, artifacts, escaped, artifacts / "escape-link/new"):
                    with self.subTest(command=command, output=output), \
                            patch.object(runner, "ARTIFACT_ROOT", artifacts), \
                            patch.object(runner, "check", side_effect=AssertionError("check ran")), \
                            patch.object(runner, "inspect", side_effect=AssertionError("cache inspected")), \
                            self.assertRaises(ValueError):
                        runner.main([command, "--config", str(config), "--output", str(output)])
            self.assertEqual(sentinel.read_bytes(), b"preserve research bytes\n")
            self.assertEqual(list(existing.iterdir()), [sentinel])
            self.assertFalse(escaped.exists())
            self.assertFalse((root / "new").exists())

    def test_configuration_rejects_unknown_training_fields_and_unbounded_checks(self):
        corruptions = (
            ("top", "train", True), ("architecture", "epochs", 100),
            ("checks", "train_on", "exact"), ("checks", "sizes", []),
            ("checks", "sizes", [3, 3]), ("checks", "sizes", [2]),
            ("checks", "sizes", [True]), ("checks", "sizes", [33]),
            ("checks", "optimizer_steps", 0), ("checks", "optimizer_steps", 51),
            ("checks", "synthetic_hamiltonians", 9), ("checks", "seed", -1),
            ("checks", "learning_rate", float("nan")),
            ("checks", "learning_rate", float("inf")), ("checks", "learning_rate", True),
            ("architecture", "hidden_channels", []),
            ("architecture", "hidden_channels", [129]),
            ("architecture", "kernel_size", 2), ("architecture", "kernel_size", True),
            ("top", "feature_cache", ""), ("top", "schema_version", 2),
            ("top", "schema_version", True), ("top", "scope", "  "),
            ("top", "scope", None),
        )
        for section, key, value in corruptions:
            with self.subTest(section=section, key=key, value=value):
                config = self.config()
                target = config if section == "top" else config[section]
                target[key] = value
                with self.assertRaises(ValueError):
                    runner.validate_config(config)
        self.assertEqual(runner.validate_config(self.config()), self.config())


if __name__ == "__main__":
    unittest.main()
