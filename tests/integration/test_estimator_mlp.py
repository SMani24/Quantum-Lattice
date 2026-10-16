"""Campaign controls using explicit synthetic CPU substitutes, never research data."""
import copy
import json
import os
from pathlib import Path
import signal
from unittest.mock import patch

import numpy as np
import unittest
import tempfile
import torch

from quantum_lattice.data.mlp_data import EdgeRows, LearningData
from quantum_lattice.models import estimator_mlp as campaign
from quantum_lattice.models import mlp_training


def settings():
    return {"schema_version": 1, "output_root": "codex material/artifacts/estimator_mlp/tiny",
            "validation_run": "unused", "sizes": [3, 4], "mode": "all_edges", "seeds": [41, 42],
            "training": {"device": "cuda", "learning_rate": 1e-3, "batch_size": 4,
                         "max_steps": 6, "validation_interval": 2, "torch_threads": 1,
                         "deterministic": True, "early_stopping": {"enabled": False}},
            "execution": {"checkpoint_steps": 2}}


def synthetic_views(root, L):
    def rows(split, offset):
        rng = np.random.default_rng(L + offset)
        x = rng.uniform(.5, 1, (12, 3)).astype(np.float32)
        return EdgeRows(x, rng.uniform(.1, .7, (12, 1)).astype(np.float32),
                        np.repeat([f"{split}0", f"{split}1"], 6),
                        np.repeat([f"L{L}-{split}0", f"L{L}-{split}1"], 6),
                        np.tile(np.arange(6, dtype=np.int64), 2))
    train, validation = rows("train", 0), rows("validation", 100)
    result = {}
    for view, offset in (("f", 0), ("g", .02), ("exact", .01)):
        def variant(row):
            return EdgeRows(row.x + np.float32(offset), row.y, row.hamiltonian_ids,
                            row.hamiltonian_uids, row.edge_ids)
        result[view] = LearningData(variant(train), variant(validation), None, L, 0,
                                    "all_edges", {"source_fingerprint": f"synthetic-{L}-{view}"}, root)
    return result




def prepare(workspace, config=None, name="tiny"):
    config = copy.deepcopy(config or settings())
    config["output_root"] = f"codex material/artifacts/estimator_mlp/{name}"
    return campaign.prepare(workspace, config)


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal_tree(a, b)
    else:
        assert left == right


class _Patcher:
    def __init__(self, case):
        self.case = case

    def setattr(self, obj, name, value):
        active = patch.object(obj, name, value)
        active.start()
        self.case.addCleanup(active.stop)


class EstimatorCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.patcher = _Patcher(self)
        def prepare_cache(repo, original, output, sizes):
            output.mkdir(parents=True, exist_ok=True)
            (output / "manifest.json").write_text('{"synthetic":true}')
            return output
        self.patcher.setattr(campaign, "prepare_feature_views", prepare_cache)
        self.patcher.setattr(campaign, "load_feature_views", synthetic_views)
        self.patcher.setattr(campaign, "_sources", lambda repo: {"synthetic": "fixed"})
        # CPU substitution exists only in these explicit synthetic tests.
        self.patcher.setattr(campaign, "resolve_device", lambda *args: torch.device("cpu"))
        self.patcher.setattr(mlp_training, "resolve_device", lambda *args: torch.device("cpu"))

    def test_summary_contrasts_count_only_completed_seed_pairs(self):
        def item(L, seed, view, own_mse, exact_mse, state="complete"):
            scores = {"f": {"mse": own_mse if view == "f" else 99.0},
                      "g": {"mse": own_mse if view == "g" else 99.0},
                      "exact": {"mse": exact_mse}}
            return {"L": L, "seed": seed, "view": view, "state": state,
                    "step": 10, "best_step": 8, "best_validation_mse": own_mse,
                    "scores": scores}
        summary = {"completed": 5, "total": 7, "state": "paused", "tasks": [
            item(3, 1, "f", .4, .2), item(3, 1, "g", .1, .17),
            item(3, 2, "f", .6, .3), item(3, 2, "g", .2, .37),
            item(3, 3, "f", 10000, 10000), item(3, 3, "g", 10000, 10000, "paused"),
            item(4, 1, "f", .9, .8, "pending")]}
        text = campaign.format_summary(summary)
        self.assertIn("L=3: 2 paired seeds; pipeline f→f MSE=0.5; g→g MSE=0.15; g-f=-0.35", text)
        self.assertIn("Common exact inputs: g-trained minus f-trained MSE=+0.02", text)
        self.assertIn("L=4: 0 completed f/g seed pairs", text)
        self.assertIn("equal-seed averages, descriptive", text)

    def test_freezes_paired_tasks_and_no_test(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        plan = campaign.load_plan(workspace, run)
        assert [(t["L"], t["seed"], t["view"]) for t in plan["tasks"]] == [
            (L, seed, view) for L in (3, 4) for seed in (41, 42) for view in ("f", "g")]
        assert plan["protocol"]["exact_view"].startswith("evaluation only")
        assert campaign.summarize_campaign(workspace, run)["completed"] == 0
        assert not (run / "models").exists()


    def test_rejects_invalid_scientific_config(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        for field, value in (("sizes", [4]), ("mode", "paper_size"), ("seeds", [41, 41]), ("seeds", [True])):
            config = settings()
            config[field] = value
            with self.assertRaises(ValueError):
                campaign.validate_config(config)


    def test_requires_cuda_config(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        config = settings()
        config["training"]["device"] = "cpu"
        with self.assertRaisesRegex(ValueError, "cuda"):
            campaign.validate_config(config)


    def test_preflight_fails_before_payload(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        monkeypatch.setattr(campaign, "resolve_device", lambda *a: (_ for _ in ()).throw(ValueError("no CUDA")))
        monkeypatch.setattr(campaign, "load_feature_views", lambda *a: self.fail("payload accessed"))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            campaign.run_campaign(workspace, run)


    def test_exact_pause_resume_matches_uninterrupted(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        full, paused = prepare(workspace, name="full"), prepare(workspace, name="paused")
        campaign.run_campaign(workspace, full, output=lambda _: None)
        summary = campaign.run_campaign(workspace, paused, max_chunks=1, output=lambda _: None)
        assert summary["completed"] == 0 and summary["tasks"][0]["step"] == 2
        assert summary["tasks"][1]["state"] == "pending"
        campaign.run_campaign(workspace, paused, output=lambda _: None)
        for task in campaign.load_plan(workspace, full)["tasks"]:
            left = torch.load(full / "models" / task["key"] / "latest.pt", weights_only=True)
            right = torch.load(paused / "models" / task["key"] / "latest.pt", weights_only=True)
            equal_tree(left["session"], right["session"])
            equal_tree(left["history"], right["history"])
        summary = campaign.summarize_campaign(workspace, paused)
        assert summary["state"] == "complete" and summary["completed"] == 8
        assert summary["test_payload_loaded"] is False


    def test_crossed_validation_outputs_and_exact_not_trained(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        captured = []
        actual = campaign.TrainSession
        def record(train_x, train_y, val_x, val_y, config, **kwargs):
            captured.append(train_x.copy())
            return actual(train_x, train_y, val_x, val_y, config, **kwargs)
        monkeypatch.setattr(campaign, "TrainSession", record)
        campaign.run_campaign(workspace, run, output=lambda _: None)
        plan = campaign.load_plan(workspace, run)
        for task, x in zip(plan["tasks"], captured):
            views = synthetic_views(run / "features", task["L"])
            assert np.array_equal(x, views[task["view"]].train.x)
            assert not np.array_equal(x, views["exact"].train.x)
            job = run / "models" / task["key"]
            result = json.loads((job / "evaluation.json").read_text())
            assert set(result["scores"]) == {"f", "g", "exact"}
            with np.load(job / "predictions.npz", allow_pickle=False) as archive:
                assert set(archive.files) == {"prediction_f", "prediction_g", "prediction_exact", "target",
                                              "hamiltonian_ids", "hamiltonian_uids", "edge_ids"}
                assert np.array_equal(archive["target"], views["f"].validation.y)
            assert len((job / "per-hamiltonian.csv").read_text().splitlines()) == 7
        # Two sizes × two seeds × three common-input contrasts plus matched pipeline.
        assert len((run / "paired-validation.csv").read_text().splitlines()) == 17


    def test_completed_outputs_are_immutable_and_verified(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        campaign.run_campaign(workspace, run, output=lambda _: None)
        task = campaign.load_plan(workspace, run)["tasks"][0]
        job = run / "models" / task["key"]
        before = campaign._sha(job / "latest.pt")
        campaign.run_campaign(workspace, run, output=lambda _: None)
        assert campaign._sha(job / "latest.pt") == before
        (job / "history.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "output changed"):
            campaign.run_campaign(workspace, run, output=lambda _: None)


    def test_done_summary_cannot_override_hashed_evaluation(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        campaign.run_campaign(workspace, run, max_chunks=3, output=lambda _: None)
        task = campaign.load_plan(workspace, run)["tasks"][0]
        marker = run / "models" / task["key"] / "done.json"
        done = json.loads(marker.read_text())
        done["summary"]["scores"]["f"]["mse"] = 0.0
        marker.write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError, "marker summary"):
            campaign.summarize_campaign(workspace, run)


    def test_real_sigint_commits_one_chunk_and_resumes(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        signaled = []
        def output(message):
            if "best validation" in message and not signaled:
                signaled.append(True)
                os.kill(os.getpid(), signal.SIGINT)
        previous = signal.getsignal(signal.SIGINT)
        result = campaign.run_campaign(workspace, run, output=output)
        assert result["tasks"][0]["step"] == 2
        assert result["completed"] == 0
        assert signal.getsignal(signal.SIGINT) == previous
        assert campaign.run_campaign(workspace, run, output=lambda _: None)["completed"] == 8


    def test_resume_final_checkpoint_after_export_crash(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        original = campaign._evaluate
        monkeypatch.setattr(campaign, "_evaluate", lambda *a, **k: (_ for _ in ()).throw(OSError("export crash")))
        with self.assertRaisesRegex(OSError, "export crash"):
            campaign.run_campaign(workspace, run, max_chunks=3, output=lambda _: None)
        monkeypatch.setattr(campaign, "_evaluate", original)
        result = campaign.run_campaign(workspace, run, max_chunks=1, output=lambda _: None)
        assert result["tasks"][0]["state"] == "complete"
        assert result["tasks"][1]["step"] == 2


    def test_source_environment_and_feature_manifest_pins(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        original = campaign._sources
        monkeypatch.setattr(campaign, "_sources", lambda repo: {"synthetic": "changed"})
        with self.assertRaisesRegex(ValueError, "source changed"):
            campaign.load_plan(workspace, run)
        monkeypatch.setattr(campaign, "_sources", original)
        (run / "features/manifest.json").write_text('{"synthetic":false}')
        with self.assertRaisesRegex(ValueError, "feature manifest"):
            campaign.load_plan(workspace, run)


    def test_config_and_plan_tampering_rejected(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        config = settings()
        config["training"]["max_steps"] = 7
        with self.assertRaisesRegex(ValueError, "Configuration differs"):
            campaign.prepare(workspace, config)
        plan = json.loads((run / "plan.json").read_text())
        plan["tasks"].reverse()
        (run / "plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "hash/schema"):
            campaign.load_plan(workspace, run)


    def test_nested_lock_rejected(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        run = prepare(workspace)
        with campaign.campaign_lock(run):
            with self.assertRaisesRegex(ValueError, "already active"):
                with campaign.campaign_lock(run):
                    pass


    def test_rejects_test_data_and_misaligned_rows(self):
        workspace = self.workspace
        monkeypatch = self.patcher
        views = synthetic_views(workspace, 3)
        data = views["g"]
        views["g"] = LearningData(data.train, data.validation, data.validation, 3, 0,
                                   "all_edges", {}, workspace)
        with self.assertRaisesRegex(ValueError, "train/validation only"):
            campaign._check_views(views, 3)
        views = synthetic_views(workspace, 3)
        views["g"].train.y[0] = .9  # Shared labels: construct a distinct target array below.
        data = views["g"]
        changed = EdgeRows(data.train.x, data.train.y + 1, data.train.hamiltonian_ids,
                           data.train.hamiltonian_uids, data.train.edge_ids)
        views["g"] = LearningData(changed, data.validation, None, 3, 0, "all_edges", {}, workspace)
        with self.assertRaisesRegex(ValueError, "row identity/labels"):
            campaign._check_views(views, 3)

if __name__ == "__main__":
    unittest.main()
