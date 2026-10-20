"""Synthetic CPU controls for the new screen; no research payload is opened."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian import hamiltonian_error
from quantum_lattice.models import cnn_ablation_campaign as campaign
from quantum_lattice.models import cnn_ablation_training as training
from quantum_lattice.models.cnn_variants import VARIANT_NAMES, make_model

try:
    from .test_observable_campaign import synthetic_reference, synthetic_views, equal_tree
except ImportError:  # unittest discover -s tests/unit imports top-level modules.
    from test_observable_campaign import synthetic_reference, synthetic_views, equal_tree


def settings(name="tiny"):
    return {"schema_version": 1, "sizes": [4], "seeds": [61, 62],
            "models": list(VARIANT_NAMES), "views": ["g"],
            "feature_cache": "codex material/artifacts/synthetic/features",
            "output_root": f"codex material/artifacts/cnn_ablation/campaigns/{name}",
            "training": {"device": "cuda", "deterministic": True, "torch_threads": 1,
                         "learning_rate": 1e-3, "batch_size_hamiltonians": 3,
                         "max_epochs": 2, "validation_batch_size": 3,
                         "early_stopping": {"enabled": False, "min_epochs": 1,
                                            "patience_epochs": 2, "min_delta": 1e-6}},
            "execution": {"checkpoint_updates": 1}, "provenance": {"synthetic": True}}


class CnnAblationCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / settings()["feature_cache"]
        self.cache.mkdir(parents=True)
        for name in ("manifest.json", "request.json"):
            (self.cache / name).write_text('{"synthetic":true}')
        (self.cache / "L4").mkdir()
        (self.cache / "L4/manifest.json").write_text('{"synthetic":true}')
        (self.cache / "L4/views.npz").write_bytes(b"synthetic; loader explicitly substituted")
        self.patch(campaign, "_sources", lambda root: {"synthetic-source": "fixed"})
        # Reused stateless helpers resolve these globals in their frozen module.
        self.patch(campaign._old, "load_feature_views", synthetic_views)
        self.patch(campaign._old, "load_validation_fields", synthetic_reference)
        # Public research settings remain CUDA-only. These explicit seams are CPU.
        self.patch(campaign, "resolve_device", lambda *args: torch.device("cpu"))
        self.patch(training, "resolve_device", lambda *args: torch.device("cpu"))

    def patch(self, obj, name, value):
        active = patch.object(obj, name, value)
        active.start()
        self.addCleanup(active.stop)

    def prepare(self, name="tiny", config=None):
        return campaign.prepare(self.root, settings(name) if config is None else config)

    def run_all(self, run, **kwargs):
        return campaign.run_campaign(self.root, run, output=lambda message: None, **kwargs)

    def test_prepare_freezes_eight_tasks_and_zero_optimizer_updates(self):
        with patch.object(campaign.CnnAblationTrainSession, "advance", side_effect=AssertionError("update")):
            run = self.prepare()
        plan = campaign.load_plan(self.root, run)
        self.assertEqual([(t["L"], t["seed"], t["model"], t["view"]) for t in plan["tasks"]],
                         [(4, seed, model, "g") for seed in (61, 62) for model in VARIANT_NAMES])
        self.assertEqual(plan["data_bindings"]["4"]["train"]["hamiltonians"], 5)
        self.assertEqual(set(plan["model_specs"]), set(VARIANT_NAMES))
        self.assertFalse(plan["protocol"]["test_payload_loaded"])
        self.assertFalse((run / "models").exists())
        summary = campaign.summarize_campaign(self.root, run)
        self.assertEqual((summary["completed"], summary["total"]), (0, 8))
        self.assertFalse(summary["ranking"])
        self.assertIn("waits for all eight", campaign.format_summary(summary))

    def test_config_scope_and_output_guards(self):
        bad = [("extra", True), ("schema_version", True), ("sizes", [3, 4]),
               ("views", ["f", "g"]), ("models", list(reversed(VARIANT_NAMES))),
               ("seeds", [61]), ("seeds", [61, 61]), ("seeds", [True, 62]),
               ("seeds", [61, 2**63]), ("feature_cache", "/outside"),
               ("output_root", ""), ("execution", {"checkpoint_updates": True})]
        for key, value in bad:
            config = settings()
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                campaign.validate_config(config)
        for changes in ({"device": "cpu"}, {"deterministic": False}, {"unknown": 1}):
            config = settings()
            config["training"].update(changes)
            with self.subTest(training=changes), self.assertRaises(ValueError):
                campaign.validate_config(config)
        with self.assertRaisesRegex(ValueError, "below"):
            campaign.prepare(self.root, settings(), run_dir=self.root / "outside")
        config = settings()
        config["feature_cache"] = "../../outside"
        with self.assertRaisesRegex(ValueError, "artifact"):
            self.prepare(config=config)

    def test_plan_source_environment_cache_and_architecture_pins(self):
        run = self.prepare()
        with patch.object(campaign, "_sources", lambda root: {"changed": True}):
            with self.assertRaisesRegex(ValueError, "source/environment"):
                campaign.load_plan(self.root, run)
        with patch.object(campaign, "_environment", lambda: {"changed": True}):
            with self.assertRaisesRegex(ValueError, "source/environment"):
                campaign.load_plan(self.root, run)
        original_spec = campaign.model_spec
        with patch.object(campaign, "model_spec", lambda name: {**original_spec(name), "changed": True}):
            with self.assertRaisesRegex(ValueError, "registry"):
                campaign.load_plan(self.root, run)
        path = self.cache / "L4/views.npz"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "cache"):
            campaign.load_plan(self.root, run)
        path.write_bytes(original)
        plan = json.loads((run / "plan.json").read_text())
        plan["tasks"].pop()
        (run / "plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "hash/schema"):
            campaign.load_plan(self.root, run)

    def test_cuda_preflight_precedes_feature_payload_and_rejects_changed_accelerator(self):
        run = self.prepare()
        with patch.object(campaign, "resolve_device", side_effect=ValueError("CUDA unavailable")):
            with patch.object(campaign, "_load_maps", side_effect=AssertionError("payload accessed")):
                with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
                    self.run_all(run)
        with patch.object(campaign, "_accelerator", lambda device: {"type": "different"}):
            with self.assertRaisesRegex(ValueError, "Accelerator differs"):
                self.run_all(run)

    def test_exact_eligibility_loaded_test_and_training_view_are_guarded(self):
        def bad_exact(root, L):
            views = synthetic_views(root, L)
            views["exact"].provenance["training_eligible"] = True
            return views
        with patch.object(campaign._old, "load_feature_views", bad_exact):
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                self.prepare("bad-exact")
        def test_payload(root, L):
            from dataclasses import replace
            views = synthetic_views(root, L)
            views["g"] = replace(views["g"], test=views["g"].validation)
            return views
        with patch.object(campaign._old, "load_feature_views", test_payload):
            with self.assertRaisesRegex(ValueError, "test split"):
                self.prepare("loaded-test")
        run = self.prepare()
        original = campaign._load_maps
        def ineligible(root, L):
            maps = original(root, L)
            maps["g"].provenance["training_eligible"] = False
            return maps
        with patch.object(campaign, "_load_maps", ineligible):
            with self.assertRaisesRegex(ValueError, "Evaluation-only"):
                self.run_all(run)
        self.assertFalse((run / "models").exists())

    def test_cache_change_during_reference_loading_prevents_updates(self):
        run = self.prepare()
        original = campaign._reference_for_maps
        def mutate(root, L, maps):
            reference = original(root, L, maps)
            path = self.cache / "L4/views.npz"
            path.write_bytes(path.read_bytes() + b"changed during loading")
            return reference
        with patch.object(campaign, "_reference_for_maps", mutate):
            with self.assertRaisesRegex(ValueError, "changed during loading"):
                self.run_all(run)
        self.assertFalse((run / "models").exists())

    def test_partial_checkpoint_resume_matches_uninterrupted_models_and_predictions(self):
        full, paused = self.prepare("full"), self.prepare("paused")
        self.run_all(full)
        summary = self.run_all(paused, max_chunks=1)
        self.assertEqual((summary["tasks"][0]["state"], summary["tasks"][0]["step"],
                          summary["tasks"][0]["epoch"]), ("paused", 1, 0))
        self.assertFalse(summary["ranking"])
        self.run_all(paused)
        for task in campaign.load_plan(self.root, full)["tasks"]:
            first, second = (path / "models" / task["key"] for path in (full, paused))
            a = torch.load(first / "latest.pt", weights_only=True)
            b = torch.load(second / "latest.pt", weights_only=True)
            equal_tree(self, a["session"], b["session"])
            for name in ("best.pt", "history.json", "predictions.npz", "per-hamiltonian.csv", "hamiltonian-scores.csv"):
                if name == "best.pt":
                    left = torch.load(first / name, weights_only=True)
                    right = torch.load(second / name, weights_only=True)
                    equal_tree(self, left["model_state_dict"], right["model_state_dict"])
                else:
                    self.assertEqual(campaign._sha(first / name), campaign._sha(second / name))

    def test_completed_jobs_skip_immutably_and_tampered_outputs_are_rejected(self):
        run = self.prepare()
        summary = self.run_all(run, models=["batchnorm"])
        self.assertEqual(summary["completed"], 2)
        paths = sorted((run / "models").rglob("*"))
        before = {str(path.relative_to(run)): path.read_bytes() for path in paths if path.is_file()}
        self.run_all(run, models=["batchnorm"])
        self.assertEqual(before, {str(path.relative_to(run)): path.read_bytes() for path in paths if path.is_file()})
        task = next(t for t in summary["tasks"] if t["model"] == "batchnorm")
        (run / "models" / task["key"] / "predictions.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "output changed"):
            campaign.summarize_campaign(self.root, run)

    def test_crossed_exports_magnitude_metrics_and_all_d5_conventions_match_oracles(self):
        run = self.prepare()
        actual, seen = campaign.CnnAblationTrainSession, []
        def record(train_x, train_y, val_x, val_y, config, **kwargs):
            seen.append((kwargs["variant"], config["seed"], train_x.copy()))
            return actual(train_x, train_y, val_x, val_y, config, **kwargs)
        with patch.object(campaign, "CnnAblationTrainSession", record):
            summary = self.run_all(run)
        self.assertEqual((summary["completed"], len(summary["ranking"])), (8, 4))
        plan = campaign.load_plan(self.root, run)
        maps = campaign._load_maps(self.cache, 4)
        reference, lattice = synthetic_reference(None, 4), square_lattice(4)
        for task, (variant, seed, inputs) in zip(plan["tasks"], seen):
            self.assertEqual((variant, seed), (task["model"], task["seed"]))
            np.testing.assert_array_equal(inputs, maps["g"].train.x)
            self.assertFalse(np.array_equal(inputs, maps["exact"].train.x))
            job = run / "models" / task["key"]
            evaluation = json.loads((job / "evaluation.json").read_text())
            self.assertEqual(set(evaluation["scores"]), {"f", "g", "exact"})
            self.assertFalse(evaluation["test_payload_loaded"])
            self.assertEqual(evaluation["hamiltonian_metric"]["sign_policy"], "reference_sign_oracle")
            with np.load(job / "predictions.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["target"], reference.validation_rows.y)
                for view in ("f", "g", "exact"):
                    prediction = archive[f"prediction_{view}"]
                    self.assertEqual(prediction.dtype, np.float32)
                    difference = prediction.astype(np.float64) - archive["target"].astype(np.float64)
                    self.assertAlmostEqual(evaluation["scores"][view]["mse"], float(np.mean(difference ** 2)), places=14)
                    self.assertAlmostEqual(evaluation["scores"][view]["mae"], float(np.mean(np.abs(difference))), places=14)
                    for model in ("two_channel", "z_branch"):
                        for include_identity in (True, False):
                            expected = [hamiltonian_error(lattice, true, np.sign(true) * raw,
                                        model=model, include_identity=include_identity)
                                        for true, raw in zip(reference.true_b_z, prediction.reshape(2, 32))]
                            name = f"{model}_{'full' if include_identity else 'traceless'}"
                            self.assertAlmostEqual(evaluation["scores"][view]["d5"][name],
                                                   float(np.mean(expected)), places=13)
            with (job / "per-hamiltonian.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)
            with (job / "hamiltonian-scores.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)
        self.assertEqual([r["mean_validation_mse"] for r in summary["ranking"]],
                         sorted(r["mean_validation_mse"] for r in summary["ranking"]))

    def test_ranking_waits_for_full_eight_and_pair_rows_wait_for_complete_seed_blocks(self):
        run = self.prepare()
        summary = self.run_all(run, max_chunks=12)
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(campaign._paired_rows(summary), [])
        self.assertEqual(summary["ranking"], [])
        summary = self.run_all(run, max_chunks=4)
        self.assertEqual(summary["completed"], 4)
        rows = campaign._paired_rows(summary)
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["seed"] for row in rows}, {61})
        self.assertEqual(summary["ranking"], [])
        summary = self.run_all(run)
        self.assertEqual(len(summary["ranking"]), 4)
        self.assertTrue(all(row["seeds"] == 2 for row in summary["ranking"]))
        self.assertEqual(len(campaign._paired_rows(summary)), 24)
        by_task = {(t["seed"], t["model"]): t for t in summary["tasks"]}
        for row in campaign._paired_rows(summary):
            base = by_task[(row["seed"], "baseline")]["scores"][row["input_view"]]
            score = by_task[(row["seed"], row["model"])]["scores"][row["input_view"]]
            self.assertAlmostEqual(row["mse_minus_baseline"], score["mse"] - base["mse"], places=15)
            self.assertAlmostEqual(row["d5_minus_baseline"],
                                   score["d5"]["two_channel_full"] - base["d5"]["two_channel_full"], places=15)

    def test_saved_batchnorm_selection_contains_selected_buffers_and_reproduces_predictions(self):
        run = self.prepare()
        summary = self.run_all(run, models=["batchnorm"])
        maps = campaign._load_maps(self.cache, 4)
        for task in (t for t in summary["tasks"] if t["model"] == "batchnorm"):
            job = run / "models" / task["key"]
            best = torch.load(job / "best.pt", weights_only=True)
            latest = torch.load(job / "latest.pt", weights_only=True)
            selected = latest["session"]["engine"]["best_state_dict"]
            equal_tree(self, best["model_state_dict"], selected)
            counters = [v for k, v in selected.items() if k.endswith("num_batches_tracked")]
            self.assertEqual(len(counters), 3)
            self.assertTrue(all(v.item() == best["best_step"] for v in counters))
            model = make_model("batchnorm")
            model.load_state_dict(best["model_state_dict"])
            model.eval()
            before = copy.deepcopy(model.state_dict())
            with torch.no_grad():
                predicted_maps = model(torch.from_numpy(maps["g"].validation.x.copy())).numpy()
            expected = campaign._old.flatten_edge_maps(maps["g"].validation, predicted_maps)
            with np.load(job / "predictions.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(expected, archive["prediction_g"])
            equal_tree(self, before, model.state_dict())

    def test_sigint_drains_chunk_saves_and_restores_previous_handler(self):
        run = self.prepare()
        previous = signal.getsignal(signal.SIGINT)
        actual = campaign.CnnAblationTrainSession.advance
        def pause(session, count):
            result = actual(session, count)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return result
        with patch.object(campaign.CnnAblationTrainSession, "advance", pause):
            summary = self.run_all(run)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)
        task = summary["tasks"][0]
        self.assertEqual((task["state"], task["step"], task["epoch"]), ("paused", 1, 0))
        state = torch.load(run / "models" / task["key"] / "latest.pt", weights_only=True)
        self.assertEqual(state["session"]["engine"]["cursor"], 3)
        self.run_all(run, models=["baseline"])

    def test_initialization_sigint_still_writes_epoch_zero_checkpoint(self):
        run = self.prepare()
        actual = campaign.CnnAblationTrainSession
        def pause(*args, **kwargs):
            session = actual(*args, **kwargs)
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return session
        with patch.object(campaign, "CnnAblationTrainSession", pause):
            summary = self.run_all(run)
        self.assertEqual((summary["tasks"][0]["state"], summary["tasks"][0]["step"]), ("paused", 0))

    def test_checkpoint_identity_variant_and_accelerator_corruption_are_rejected(self):
        run = self.prepare()
        summary = self.run_all(run, max_chunks=1)
        path = run / "models" / summary["tasks"][0]["key"] / "latest.pt"
        original = torch.load(path, weights_only=True)
        mutations = [lambda state: state["task"].update(view="exact"),
                     lambda state: state.update(accelerator={"type": "different"}),
                     lambda state: state["session"].update(variant="batchnorm"),
                     lambda state: state["session"]["engine"].update(step=True)]
        for mutate in mutations:
            state = copy.deepcopy(original)
            mutate(state)
            torch.save(state, path)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.run_all(run)
        torch.save(original, path)
        self.run_all(run, models=["baseline"])


if __name__ == "__main__":
    unittest.main()
