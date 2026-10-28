"""Tiny CPU orchestration through real training, sampling, scoring and reports.

The production plan/cache boundary is replaced by an explicit small fixture.
Only device selection is adapted; the actual train/resume engine, selection
seal, fresh-cohort generator, inference, Pauli metrics and reports execute.
This is an acceptance test, not a scientific result or production CPU fallback.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.data import final_confirmation as confirmation
    from quantum_lattice.data.state_storage import decode_states
    from quantum_lattice.models import final_campaign as campaign
    from quantum_lattice.models.final_evaluation import load_evaluation
    from quantum_lattice.models.final_training import (
        checkpoint_path, load_job, predict_checkpoint, train_job)


REPO = Path(__file__).resolve().parents[2]


def hashes(directory, pattern="*"):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(directory).rglob(pattern) if path.is_file()}


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class FinalCampaignFlowTests(unittest.TestCase):
    def test_real_small_flow_seals_after_training_and_completed_resume_never_refits(self):
        self.exercise_flow(pause_before_reports=False)

    def test_scientific_marker_resumes_only_reports_using_the_reserved_budget(self):
        self.exercise_flow(pause_before_reports=True)

    def exercise_flow(self, *, pause_before_reports):
        allowed = REPO / "codex material/artifacts/final_benchmark"
        allowed.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-final-campaign-flow-", dir=allowed) as temporary:
            fixture = Path(temporary)
            run, cache, prior = (fixture / name for name in ("campaign", "features", "prior"))
            for directory in (run, cache / "L3", prior):
                directory.mkdir(parents=True)
            (cache / "L3/manifest.json").write_text(json.dumps({"fixture": True, "L": 3}))
            (prior / "L3.json").write_text(json.dumps({"instances": [
                {"id": "already-exposed", "split": "test", "hamiltonian_uid": "f"*64}]}))
            old_payload = prior / "old-test.npz"
            old_payload.write_bytes(b"Not an archive; no old test payload access is allowed.")
            config = json.loads((REPO / "configs/final_benchmark.json").read_text())
            config.update(sizes=[3], size_order=[3], seeds=[701],
                feature_cache=str(cache.relative_to(REPO)), output_root=str(run.relative_to(REPO)),
                prior_manifest_root=str(prior.relative_to(REPO)),
                architecture={"hidden_channels": [2], "kernel_size": 3})
            config["training"].update(device="cpu", max_epochs=1,
                batch_size_hamiltonians=2, validation_batch_size=2)
            config["training"]["early_stopping"]["enabled"] = False
            config["confirmation"].update(count_per_size=2, seed=2026102895,
                sampling={"total_sweeps": 16, "burn_in_sweeps": 4, "thinning_sweeps": 2,
                          "chunk_sweeps": 4, "lazy_probability": .05, "backend": "python",
                          "exact_small_sizes": False, "block_size_records": 2},
                diagnostics={"enabled": False, "hamiltonians_per_size": 1, "replicas": 4})
            config["raw_panel"]["enabled"] = False
            config["statistics"]["bootstrap_samples"] = 100
            config["execution"].update(workers=1, checkpoint_updates=1, preflight_active_seconds=0)
            tasks = [{"L": 3, "seed": 701, "model": model, "view": view,
                      "key": f"L3-seed701-{model}-{view}"}
                     for model in ("cnn", "map_mlp") for view in ("f", "g")]
            plan = {"schema_version": 1, "config": config, "tasks": tasks,
                    "sources": {}, "features": {}, "input_metadata": {},
                    "scope": "tiny explicit CPU orchestration fixture"}
            campaign.atomic_json(run / "plan.json", plan)
            campaign.atomic_json(run / "plan-sha256.json", {"sha256": campaign.digest(plan)})
            campaign.atomic_json(run / "clock.json", {"active_seconds": 0,
                "inflight": False, "stage": "prepared", "last_tick_epoch": campaign.time.time()})
            rng = np.random.default_rng(901)
            data = {"L": 3}
            for split, count, symbol in [("train", 4, "a"), ("validation", 2, "b")]:
                fields = rng.uniform(-.4, .4, (count, 18))
                data[split] = {"f": rng.uniform(.1, 1.2, (count, 3, 3, 3)).astype(np.float32),
                    "g": rng.uniform(.1, 1., (count, 3, 3, 3)).astype(np.float32),
                    "y": np.abs(fields).reshape(count, 3, 3, 2).transpose(0, 3, 1, 2).astype(np.float32),
                    "fields": fields, "ids": np.array([f"{split}-{i}" for i in range(count)]),
                    "uids": np.array([symbol*63 + str(i) for i in range(count)])}
            events = []
            original_prepare = confirmation.prepare_confirmation
            original_finish = campaign._finish_reports

            def cpu_train(*args, **kwargs):
                self.assertFalse((run / "confirmation").exists())
                result = train_job(*args, **kwargs, allow_cpu=True)
                events.append(("trained", result["model"], result["seed"]))
                return result

            def cpu_predict(path, x, **kwargs):
                kwargs["device"] = "cpu"
                kwargs["allow_cpu"] = True
                return predict_checkpoint(path, x, **kwargs)

            def sealed_prepare(*args, **kwargs):
                self.assertTrue(all(load_job(run / "models" / t["key"])["state"] == "complete"
                                    for t in tasks))
                self.assertEqual(len(events), 4)
                self.assertTrue((run / "confirmation-seal.json").is_file())
                seal = campaign.read(run / "confirmation-seal.json")
                expected = {str(checkpoint_path(run / "models" / t["key"]).relative_to(REPO)):
                            campaign.sha(checkpoint_path(run / "models" / t["key"])) for t in tasks}
                self.assertEqual(seal["model_sha256"], expected)
                events.append(("confirmation_prepared",))
                return original_prepare(*args, **kwargs)

            def finish_reports(root, output, pinned, seal, budget, stop):
                if pause_before_reports:
                    self.assertTrue((run / "scientific-work-complete.json").is_file())
                    return campaign.write_status(output, pinned, "paused_reports")
                return original_finish(root, output, pinned, seal, budget, stop)

            with mock.patch.object(campaign, "load_plan", return_value=plan), \
                 mock.patch.object(campaign, "load_features", return_value=data), \
                 mock.patch.object(campaign, "train_job", side_effect=cpu_train), \
                 mock.patch.object(campaign, "predict_checkpoint", side_effect=cpu_predict), \
                 mock.patch.object(confirmation, "prepare_confirmation", side_effect=sealed_prepare), \
                 mock.patch.object(campaign, "_finish_reports", side_effect=finish_reports), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = campaign.run_campaign(REPO, run, workers=1)
            self.assertEqual(result["completed"], 4)
            self.assertEqual(result["confirmation_complete"], not pause_before_reports)
            self.assertEqual((run / "complete.json").exists(), not pause_before_reports)
            if pause_before_reports:
                # Simulate reaching the scientific cutoff after an indivisible
                # final work unit. The final-report reserve must still work.
                ledger = campaign.read(run / "clock.json")
                ledger.update(active_seconds=90*3600+1, inflight=False,
                              stage="paused_reports", last_tick_epoch=campaign.time.time())
                campaign.atomic_json(run / "clock.json", ledger)
            self.assertEqual(events[-1], ("confirmation_prepared",))
            self.assertEqual(old_payload.read_bytes(), b"Not an archive; no old test payload access is allowed.")
            for role in ("validation", "confirmation"):
                for task in tasks:
                    value = load_evaluation(run / "evaluations" / role / task["key"])
                    self.assertEqual(value["hamiltonians"], 2)
                    self.assertTrue(np.isfinite(value["metrics"]["mse"]))
            cohort = confirmation.load_confirmation_plan(REPO, run / "confirmation")
            self.assertEqual(cohort["role"], "confirmation")
            self.assertEqual(len(cohort["instances"]), 2)
            final_seal = campaign.read(run / "confirmation-seal.json")
            self.assertEqual(cohort["freeze_sha256"], campaign.digest(final_seal))
            self.assertEqual(cohort["bindings"]["model_sha256"], final_seal["model_sha256"])
            for instance in cohort["instances"]:
                path = run / "confirmation/instances" / instance["id"] / "records.npz"
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(decode_states(archive).shape, (6, 18))
            model_before = hashes(run / "models")
            evaluation_before = hashes(run / "evaluations")
            records_before = hashes(run / "confirmation/instances", "records.npz")
            seal_before = (run / "confirmation-seal.json").read_bytes()
            with mock.patch.object(campaign, "load_plan", return_value=plan), \
                 mock.patch.object(campaign, "load_features", side_effect=AssertionError("Report/completed resume must not reload learning data")), \
                 mock.patch.object(campaign, "train_job", side_effect=AssertionError("Completed models must not refit")), \
                 mock.patch.object(campaign, "predict_checkpoint", side_effect=AssertionError("Committed predictions must not rerun")), \
                 mock.patch.object(confirmation, "prepare_confirmation", side_effect=AssertionError("Report/completed resume must not create a cohort")), \
                 mock.patch.object(confirmation, "load_confirmation_features", side_effect=AssertionError("Report/completed resume must not reload cohort arrays")), \
                 mock.patch.object(confirmation, "_worker_advance", side_effect=AssertionError("Completed cohort must not resample")), \
                 contextlib.redirect_stdout(io.StringIO()):
                resumed = campaign.run_campaign(REPO, run, workers=1, resume=True)
            self.assertEqual(resumed["completed"], 4)
            self.assertTrue(resumed["confirmation_complete"])
            self.assertTrue((run / "complete.json").is_file())
            self.assertTrue((run / "model-scores.csv").is_file())
            self.assertTrue((run / "paired-comparisons.json").is_file())
            self.assertTrue((run / "RESULTS.md").is_file())
            self.assertTrue(campaign.verify_completed(run, plan)["verified"])
            if pause_before_reports:
                seconds = campaign.read(run / "clock.json")["active_seconds"]
                self.assertGreater(seconds, 90*3600)
                self.assertLess(seconds, 96*3600-330)
            self.assertEqual(hashes(run / "models"), model_before)
            self.assertEqual(hashes(run / "evaluations"), evaluation_before)
            self.assertEqual(hashes(run / "confirmation/instances", "records.npz"), records_before)
            self.assertEqual((run / "confirmation-seal.json").read_bytes(), seal_before)


if __name__ == "__main__":
    unittest.main()
