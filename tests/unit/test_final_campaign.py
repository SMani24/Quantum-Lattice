"""Final-study protocol, budget and access transitions without research runs."""
from copy import deepcopy
import hashlib
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
    from quantum_lattice.models import final_campaign as campaign


REPO = Path(__file__).resolve().parents[2]


class FakeClock:
    def __init__(self, wall=1000., monotonic=10.):
        self.wall, self.monotone = wall, monotonic

    def time(self):
        return self.wall

    def monotonic(self):
        return self.monotone

    def advance(self, seconds):
        self.wall += seconds
        self.monotone += seconds


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class FinalCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((REPO / "configs/final_benchmark.json").read_text())
        # The raw-panel adapter has its own tests. These lightweight fixtures
        # exercise the principal seven-size protocol without loading models.
        self.config["raw_panel"]["enabled"] = False

    def metadata_inputs(self):
        cache = self.root / self.config["feature_cache"]
        cache.mkdir(parents=True)
        for name in ("request.json", "manifest.json"):
            (cache / name).write_text(json.dumps({"fixture": name}))
        for L in campaign.SIZES:
            directory = cache / f"L{L}"
            directory.mkdir()
            (directory / "manifest.json").write_text(json.dumps({"L": L, "fixture": True}))
            # Not a scientific archive. A prepare operation must delegate
            # loading to the verified adapter, rather than opening test data.
            (directory / "maps.npz").write_bytes(f"development fixture {L}".encode())
        prior = self.root / self.config["prior_manifest_root"]
        prior.mkdir(parents=True)
        for L in campaign.SIZES:
            (prior / f"L{L}.json").write_text(json.dumps({"instances": []}))
        return cache

    @staticmethod
    def fake_features(cache, L, *, train_count=7452):
        return {"L": L, **{split: {
            "y": np.empty(count, dtype=np.uint8), "f": None, "g": None,
            "uids": np.array([f"L{L}-{split}-{i}" for i in range(count)])}
            for split, count in [("train", train_count), ("validation", 1242)]}}

    def prepare(self):
        self.metadata_inputs()
        with mock.patch.object(campaign, "load_features", side_effect=self.fake_features), \
             mock.patch("torch.cuda.is_available", side_effect=AssertionError("prepare must not require GPU")), \
             mock.patch("numpy.load", side_effect=AssertionError("fixture prepare must not open test archives")):
            run = campaign.prepare(self.root, self.config)
        return run, campaign.load_plan(self.root, run)

    def ledger(self, *, active=600., inflight=False, last_tick=1000.):
        run = self.root / "budget"
        run.mkdir(exist_ok=True)
        campaign.atomic_json(run / "clock.json", {"active_seconds": active,
            "inflight": inflight, "stage": "fixture", "last_tick_epoch": last_tick})
        return run

    def test_public_configuration_has_all_seven_sizes_five_fresh_paired_seeds_and_96h_cap(self):
        c = campaign.validate_config(self.config)
        self.assertEqual(c["sizes"], [3, 4, 8, 12, 16, 20, 24])
        self.assertEqual(c["seeds"], list(range(2026102801, 2026102806)))
        self.assertEqual(c["execution"]["max_active_hours"], 96)
        self.assertEqual(c["execution"]["science_active_hours"], 90)
        self.assertEqual(c["confirmation"]["count_per_size"], 512)
        sampler = c["confirmation"]["sampling"]
        self.assertEqual((sampler["total_sweeps"] - sampler["burn_in_sweeps"]) // sampler["thinning_sweeps"], 4500)
        tasks = campaign.tasks(c)
        self.assertEqual(len(tasks), 140)
        self.assertEqual(len({t["key"] for t in tasks}), 140)
        self.assertEqual([tasks[i]["L"] for i in range(0, len(tasks), 20)], c["size_order"])
        for L in c["sizes"]:
            for seed in c["seeds"]:
                arms = {(t["model"], t["view"]) for t in tasks if t["L"] == L and t["seed"] == seed}
                self.assertEqual(arms, {("cnn", "f"), ("cnn", "g"), ("map_mlp", "f"), ("map_mlp", "g")})

    def test_rejects_missing_size_cpu_fallback_and_oversized_budget(self):
        variants = []
        c = deepcopy(self.config); c["sizes"].remove(24); variants.append(c)
        c = deepcopy(self.config); c["training"]["device"] = "cpu"; variants.append(c)
        c = deepcopy(self.config); c["execution"]["max_active_hours"] = 97; variants.append(c)
        c = deepcopy(self.config); c["execution"]["science_active_hours"] = 96; variants.append(c)
        c = deepcopy(self.config); c["execution"]["workers"] = 11; variants.append(c)
        c = deepcopy(self.config); c["seeds"].append(c["seeds"][0]); variants.append(c)
        for c in variants:
            with self.subTest(config=c), self.assertRaises(ValueError):
                campaign.validate_config(c)

    def test_invalid_sampling_and_diagnostics_fail_before_training(self):
        variants = []
        c = deepcopy(self.config); c["confirmation"]["sampling"]["backend"] = "not-a-backend"; variants.append(c)
        c = deepcopy(self.config); c["confirmation"]["sampling"]["total_sweeps"] = 5; variants.append(c)
        c = deepcopy(self.config); c["confirmation"]["diagnostics"] = {}; variants.append(c)
        c = deepcopy(self.config); c["confirmation"]["diagnostics"]["replicas"] = -1; variants.append(c)
        for c in variants:
            with self.subTest(confirmation=c["confirmation"]), self.assertRaises(ValueError):
                campaign.validate_config(c)

    def test_boolean_budget_fields_are_not_treated_as_numbers(self):
        for key in ("science_active_hours", "preflight_active_seconds"):
            c = deepcopy(self.config)
            c["execution"][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                campaign.validate_config(c)

    def test_clean_pause_excludes_offline_gap_and_resumes_cumulative_active_time(self):
        run, clock = self.ledger(), FakeClock()
        budget = campaign.Budget(run, self.config["execution"], clock=clock)
        self.assertEqual(budget.active_seconds, 600)
        clock.advance(12)
        budget.tick("training")
        self.assertEqual(campaign.read(run / "clock.json")["active_seconds"], 612)
        clock.advance(8)
        budget.tick(done=True)
        clock.advance(1000)  # User pause, not active PC work.
        resumed = campaign.Budget(run, self.config["execution"], clock=clock)
        self.assertEqual(resumed.active_seconds, 620)
        self.assertEqual(resumed.unclean_charge, 0)
        clock.advance(5)
        resumed.tick(done=True)
        self.assertEqual(resumed.active_seconds, 625)
        self.assertEqual(resumed.remaining, 90*3600-625)

    def test_unclean_exit_gap_is_charged_conservatively_once_per_restart(self):
        run = self.ledger(active=600, inflight=True, last_tick=1000)
        clock = FakeClock(wall=1030, monotonic=50)
        first = campaign.Budget(run, self.config["execution"], clock=clock)
        self.assertEqual((first.unclean_charge, first.active_seconds), (30, 630))
        clock.advance(7)
        first.tick("training")
        clock.advance(10)
        second = campaign.Budget(run, self.config["execution"], clock=clock)
        self.assertEqual((second.unclean_charge, second.active_seconds), (10, 647))
        second.tick(done=True)

    def test_backward_wall_clock_does_not_subtract_spent_budget(self):
        run = self.ledger(active=600, inflight=True, last_tick=1100)
        clock = FakeClock(wall=1000)
        budget = campaign.Budget(run, self.config["execution"], clock=clock)
        self.assertEqual((budget.unclean_charge, budget.active_seconds), (0, 600))
        clock.advance(2)
        self.assertEqual(budget.active_seconds, 602)

    def test_spent_science_budget_has_zero_remaining(self):
        run = self.ledger(active=95*3600)
        budget = campaign.Budget(run, self.config["execution"], clock=FakeClock())
        self.assertEqual(budget.remaining, 0)

    def test_finalization_can_use_reserve_but_stops_before_the_hard_drain_margin(self):
        run = self.ledger(active=90*3600)
        clock = FakeClock()
        budget = campaign.Budget(run, self.config["execution"], clock=clock)
        self.assertEqual(budget.remaining, 0)  # Scientific dispatch is finished.
        budget.finalizing = True
        self.assertEqual(budget.remaining, 6*3600-330)
        budget.tick("reports")
        clock.advance(6*3600-331)
        self.assertEqual(budget.remaining, 1)
        clock.advance(1)
        self.assertEqual(budget.remaining, 0)
        self.assertEqual(budget.active_seconds, 96*3600-330)
        budget.tick(done=True)
        saved = campaign.read(run / "clock.json")
        self.assertEqual(saved["hard_limit_seconds"], 96*3600)
        self.assertEqual(saved["science_limit_seconds"], 90*3600)
        self.assertFalse(saved["inflight"])

    def test_corrupt_negative_budget_is_refused_without_rewriting_ledger(self):
        run = self.ledger(active=-1)
        before = (run / "clock.json").read_bytes()
        with self.assertRaises(ValueError):
            campaign.Budget(run, self.config["execution"], clock=FakeClock())
        self.assertEqual((run / "clock.json").read_bytes(), before)

    def test_prepare_pins_all_sizes_without_gpu_or_old_test_payload_access(self):
        run, plan = self.prepare()
        self.assertEqual(len(plan["tasks"]), 140)
        self.assertFalse(plan["confirmation_payload_loaded"])
        self.assertEqual(set(plan["groups"]), set(map(str, campaign.SIZES)))
        self.assertTrue(all(g["counts"] == {"train": 7452, "validation": 1242} for g in plan["groups"].values()))
        self.assertFalse((run / "confirmation").exists())
        self.assertFalse((run / "confirmation-seal.json").exists())
        self.assertEqual(campaign.read(run / "status.json")["completed"], 0)
        changed = self.root / self.config["feature_cache"] / "L24/maps.npz"
        changed.write_bytes(b"changed development feature cache")
        with self.assertRaisesRegex(ValueError, "input changed"):
            campaign.load_plan(self.root, run)

    def test_prepare_refuses_reduced_or_mixed_group_counts(self):
        self.metadata_inputs()
        with mock.patch.object(campaign, "load_features", side_effect=lambda root, L:
                               self.fake_features(root, L, train_count=8)), self.assertRaisesRegex(ValueError, "7,452"):
            campaign.prepare(self.root, self.config)
        run = self.root / self.config["output_root"]
        self.assertFalse((run / "plan.json").exists())

    def test_status_uses_verified_job_metadata_without_mutating_clock_or_checkpoint(self):
        run, plan = self.prepare()
        job = run / "models" / plan["tasks"][0]["key"]
        job.mkdir(parents=True)
        (job / "job-plan.json").write_text("stand-in for separately tested engine metadata")
        clock_before = (run / "clock.json").read_bytes()
        marker_before = (job / "job-plan.json").read_bytes()
        def verified_status(path, *, verify=True):
            if verify is not True:
                raise ValueError("Unchecked engine status is forbidden")
            self.assertEqual(Path(path), job)
            return {"state": "complete", "epoch": 42, "step": 100, "best_epoch": 41,
                    "best_validation_mse": .5, "active_seconds": 20, "stop_reason": "early_stopping"}
        with mock.patch.object(campaign, "load_job", side_effect=verified_status):
            status = campaign.write_status(run, plan)
        self.assertEqual(status["completed"], 1)
        self.assertEqual((run / "clock.json").read_bytes(), clock_before)
        self.assertEqual((job / "job-plan.json").read_bytes(), marker_before)
        self.assertIn("1/140", campaign.format_status(status))

    def test_pause_request_only_writes_control_file(self):
        run, plan = self.prepare()
        before = (run / "plan.json").read_bytes()
        pause = campaign.request_pause(self.root, self.config)
        self.assertEqual(pause, run / "PAUSE")
        self.assertTrue(pause.is_file())
        self.assertEqual((run / "plan.json").read_bytes(), before)
        self.assertFalse((run / "confirmation").exists())

    def test_confirmation_seal_refuses_incomplete_models_before_any_access(self):
        run, plan = self.prepare()
        with mock.patch.object(campaign, "load_job", return_value={"state": "paused"}), \
             mock.patch.object(campaign, "checkpoint_path", side_effect=AssertionError("No selected weight access yet")), \
             self.assertRaisesRegex(ValueError, "every model"):
            campaign._seal(self.root, run, plan)
        self.assertFalse((run / "confirmation-seal.json").exists())
        self.assertFalse((run / "confirmation").exists())

    def test_seal_hashes_every_frozen_model_and_rejects_later_reselection(self):
        run, plan = self.prepare()
        for task in plan["tasks"]:
            job = run / "models" / task["key"]
            job.mkdir(parents=True)
            (job / "best.pt").write_bytes(task["key"].encode())
        with mock.patch.object(campaign, "load_job", return_value={"state": "complete"}), \
             mock.patch.object(campaign, "checkpoint_path", side_effect=lambda job, kind: Path(job)/"best.pt"):
            seal = campaign._seal(self.root, run, plan)
            self.assertEqual(len(seal["model_sha256"]), 140)
            self.assertTrue(seal["prohibits_reselection"])
            self.assertFalse(seal["old_tests_are_untouched"])
            self.assertTrue(all(campaign.sha(self.root/p) == value for p, value in seal["model_sha256"].items()))
            before = (run / "confirmation-seal.json").read_bytes()
            first = run / "models" / plan["tasks"][0]["key"] / "best.pt"
            first.write_bytes(b"different selected model")
            with self.assertRaisesRegex(ValueError, "seal changed"):
                campaign._seal(self.root, run, plan)
            self.assertEqual((run / "confirmation-seal.json").read_bytes(), before)

    def test_evaluator_accepts_distinct_validation_and_confirmation_identity_schemas(self):
        run, plan = self.prepare()
        task = plan["tasks"][0]
        selected = self.root / "selected-best.pt"
        selected.write_bytes(b"selected fixture hash only")
        x = np.ones((2, 3, 4, 4), np.float32)
        fields = np.full((2, 32), .2, np.float64)
        ids, uids = np.array(["H0", "H1"]), np.array(["u0", "u1"])
        prediction = np.zeros((2, 2, 4, 4), np.float32)
        for role, data in [
            ("validation", {"f": x, "g": x, "fields": fields, "ids": ids, "uids": uids}),
            ("confirmation", {"x_f": x, "x_g": x, "fields": fields,
                              "hamiltonian_ids": ids, "hamiltonian_uids": uids})]:
            with self.subTest(role=role), \
                 mock.patch.object(campaign, "checkpoint_path", return_value=selected), \
                 mock.patch.object(campaign, "predict_checkpoint", return_value=prediction) as predict, \
                 mock.patch.object(campaign, "evaluate_predictions", return_value={"fixture": True}) as evaluate:
                value = campaign._evaluate_task(self.root, run, plan, task, data, role)
                self.assertEqual(value, {"fixture": True})
                self.assertIs(predict.call_args.args[1], x)
                self.assertIs(evaluate.call_args.args[2], fields)
                self.assertIs(evaluate.call_args.args[3], ids)
                self.assertIs(evaluate.call_args.args[4], uids)


if __name__ == "__main__":
    unittest.main()
