"""CPU-only export, freeze-before-score and CLI contracts for D5 evaluation."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.metrics import hamiltonian_evaluation as evaluation


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "hamiltonian_scores_runner", ROOT / "experiments/hamiltonian_scores/run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config():
    return json.loads((ROOT / "configs/hamiltonian_scores.json").read_text())


class HamiltonianScoresIntegrationTests(unittest.TestCase):
    def fixture(self, root):
        cfg = config()
        tasks = [{"L": 3, "seed": seed, "view": view, "key": f"L3-seed{seed}-{view}"}
                 for seed in (11, 22) for view in ("f", "g")]
        plan = {"tasks": tasks, "source_hashes": {},
                "config": {"sizes": [3], "seeds": [11, 22]}}
        features = root / "codex material/artifacts/features"
        features.mkdir(parents=True)
        (features / "manifest.json").write_text('{"synthetic":true}\n')
        feature_sha = evaluation._sha(features / "manifest.json")
        campaigns = {}
        for entry in cfg["campaigns"]:
            path = root / entry["run_dir"]
            path.mkdir(parents=True)
            (path / "preserve.bin").write_bytes(b"frozen synthetic research bytes\0")
            campaigns[entry["label"]] = {"label": entry["label"], "directory": path,
                "plan": deepcopy(plan), "plan_digest": "a"*64, "feature_dir": features,
                "feature_manifest_sha256": feature_sha, "snapshot": evaluation._snapshot(path)}
        campaigns["continued"]["plan"]["parent"] = {
            "run": cfg["campaigns"][0]["run_dir"], "plan_sha256": "a"*64}
        for name in ("src/quantum_lattice/data/hamiltonian_scores.py",
                     "src/quantum_lattice/metrics/hamiltonian.py",
                     "src/quantum_lattice/metrics/hamiltonian_batch.py",
                     "src/quantum_lattice/metrics/hamiltonian_evaluation.py",
                     "experiments/hamiltonian_scores/run.py"):
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic source pin\n")
        true = np.linspace(-.8, .9, 36).reshape(2, 18)
        ref = SimpleNamespace(source_manifest_sha256=feature_sha, true_b_z=true,
                              hamiltonian_ids=np.array(["h0", "h1"]),
                              hamiltonian_uids=np.array(["u0", "u1"]))
        output = root/evaluation.ARTIFACT_ROOT/"small-export"

        def predictions(directory, task, reference, **kwargs):
            self.assertIs(reference, ref)
            self.assertEqual(kwargs["plan_sha256"], "a"*64)
            base = abs(true) * (.8 if task["view"] == "f" else .9)
            return {view: base + offset for view, offset in (("f", .05), ("g", 0.), ("exact", -.01))}

        stack = ExitStack()
        stack.enter_context(patch.object(evaluation, "_campaign",
                                         side_effect=lambda _, entry: campaigns[entry["label"]]))
        stack.enter_context(patch.object(evaluation, "load_validation_fields", return_value=ref))
        stack.enter_context(patch.object(evaluation, "load_prediction_maps", side_effect=predictions))
        self.addCleanup(stack.close)
        return cfg, campaigns, ref, output

    def test_full_cpu_export_freezes_plan_before_any_metric_and_commits_all_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg, campaigns, _, output = self.fixture(root)
            original = evaluation.score_fields
            calls = []

            def checked(*args, **kwargs):
                # This is checked on the first (baseline) metric call as well.
                frozen = json.loads((output/"plan.json").read_text())
                digest = json.loads((output/"plan-sha256.json").read_text())
                self.assertEqual(digest, {"sha256": evaluation._digest(frozen)})
                self.assertEqual(frozen["config"], cfg)
                self.assertFalse((output/"done.json").exists())
                calls.append(1)
                return original(*args, **kwargs)

            with patch.object(evaluation, "score_fields", side_effect=checked):
                path, summary = evaluation.evaluate_campaigns(root, cfg, output=output,
                                                              output_callback=lambda _: None)
            self.assertEqual(path, output)
            self.assertEqual(len(calls), 25)  # 24 model/input cells plus baseline.
            self.assertEqual(summary["model_input_cells"], 24)
            self.assertEqual(summary["scored_hamiltonian_rows"], 48)
            self.assertEqual(summary["state"], "complete")
            self.assertFalse(summary["gpu_work"])
            self.assertFalse(summary["test_payload_access"])
            self.assertFalse(summary["training_or_new_sampling"])
            self.assertLessEqual(summary["maximum_zero_x_traceless_invariance_error"], 1e-12)
            self.assertEqual(len(summary["contrasts"]), 2*4*4)
            marker = json.loads((output/"done.json").read_text())
            self.assertEqual(marker["plan_sha256"],
                             json.loads((output/"plan-sha256.json").read_text())["sha256"])
            files = {p.name for p in output.iterdir() if p.is_file()} - {"done.json"}
            self.assertEqual(set(marker["files"]), files)
            for name, sha in marker["files"].items():
                self.assertEqual(evaluation._sha(output/name), sha)
            self.assertIn("reference-sign oracle", (output/"RESULTS.md").read_text())
            for campaign in campaigns.values():
                self.assertEqual(evaluation._snapshot(campaign["directory"]), campaign["snapshot"])

    def test_mutating_research_input_during_scoring_prevents_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg, campaigns, _, output = self.fixture(root)
            original = evaluation.score_fields
            changed = campaigns["parent"]["directory"]/"preserve.bin"

            def mutate(*args, **kwargs):
                changed.write_bytes(b"unexpected mutation")
                return original(*args, **kwargs)

            with patch.object(evaluation, "score_fields", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    evaluation.evaluate_campaigns(root, cfg, output=output, output_callback=lambda _: None)
            self.assertFalse((output/"done.json").exists())

    def test_mutating_source_during_scoring_prevents_completion_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg, _, _, output = self.fixture(root)
            original = evaluation.score_fields
            changed = root/"src/quantum_lattice/metrics/hamiltonian_evaluation.py"

            def mutate(*args, **kwargs):
                changed.write_text("# changed during execution\n")
                return original(*args, **kwargs)

            with patch.object(evaluation, "score_fields", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    evaluation.evaluate_campaigns(root, cfg, output=output, output_callback=lambda _: None)
            self.assertFalse((output/"done.json").exists())

    def test_invalid_config_and_output_guards_run_before_campaign_input_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root/evaluation.ARTIFACT_ROOT/"exists"; existing.mkdir(parents=True)
            sentinel = existing/"sentinel"; sentinel.write_text("keep")
            for output in (existing, root/"outside", root/evaluation.ARTIFACT_ROOT):
                with self.subTest(output=output), patch.object(evaluation, "_campaign") as read:
                    with self.assertRaises(ValueError):
                        evaluation.evaluate_campaigns(root, config(), output=output)
                    read.assert_not_called()
            bad = config(); bad["split"] = "test"
            with patch.object(evaluation, "_campaign") as read, self.assertRaises(ValueError):
                evaluation.evaluate_campaigns(root, bad, output=root/evaluation.ARTIFACT_ROOT/"test")
            read.assert_not_called()
            self.assertEqual(sentinel.read_text(), "keep")

    def test_two_matching_but_unrelated_campaigns_are_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg, campaigns, _, output = self.fixture(root)
            campaigns["continued"]["plan"]["parent"]["run"] = "codex material/artifacts/another-parent"
            with patch.object(evaluation, "score_fields") as score, self.assertRaisesRegex(ValueError, "must descend"):
                evaluation.evaluate_campaigns(root, cfg, output=output)
            score.assert_not_called()
            self.assertFalse(output.exists())

    def test_cli_default_and_explicit_config_dispatch_and_print_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            default = root/"configs/hamiltonian_scores.json"; default.parent.mkdir()
            default.write_text(json.dumps(config()))
            custom = root/"custom.json"; customized = config(); customized["batch_size"] = 3
            custom.write_text(json.dumps(customized))
            output = root/"outputs"
            summary = {"primary_metric": evaluation.METRICS[0], "elapsed_seconds": .1,
                       "contrasts": [{"context": "matched_pipeline", "metric": evaluation.METRICS[0],
                                      "relative_reduction": None, "campaign": "parent", "L": 3,
                                      "f_mean_d5": .2, "g_mean_d5": .3}]}
            for arguments, expected in (([], config()), (["--config", str(custom), "--output", str(output)], customized)):
                text = io.StringIO()
                with self.subTest(arguments=arguments), patch.object(runner, "ROOT", root), \
                        patch.object(runner, "evaluate_campaigns", return_value=(output, summary)) as run, \
                        redirect_stdout(text):
                    self.assertEqual(runner.main(arguments), 0)
                run.assert_called_once_with(root, expected, output=output if arguments else None)
                self.assertIn("development validation", text.getvalue())
                self.assertIn("undefined", text.getvalue())


if __name__ == "__main__":
    unittest.main()
