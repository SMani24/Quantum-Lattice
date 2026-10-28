"""Fresh-cohort role, same-record estimators and durable pause/resume checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quantum_lattice.data.final_confirmation import (
    advance_confirmation_instance, load_confirmation_features, load_confirmation_plan,
    prepare_confirmation, run_confirmation)
from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.conditioned import exact_reference, paired_estimators


REPO = Path(__file__).resolve().parents[2]


def archive(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


class FreshConfirmationTests(unittest.TestCase):
    def setUp(self):
        root = REPO / "codex material/artifacts/final_benchmark"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="test-confirmation-", dir=root)
        self.root = Path(self.temp.name)
        self.sampling = {"total_sweeps": 37, "burn_in_sweeps": 5,
                         "thinning_sweeps": 4, "lazy_probability": .05,
                         "backend": "python", "chunk_sweeps": 9,
                         "block_size_records": 3, "exact_small_sizes": False}
        self.exclusion = self.root / "old-bank-manifest.json"
        self.exclusion.write_text(json.dumps({"instances": [
            {"id": "old_test", "split": "test", "hamiltonian_uid": "f"*64}]}))
        self.payload = self.root / "old-test-payload.npz"
        self.payload.write_bytes(b"This fixture payload must not be opened as an archive.")

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, name="run", sizes=(3,), count=1, sampling=None, **kwargs):
        return prepare_confirmation(REPO, self.root/name, sizes, count, 2026102799,
            sampling or self.sampling, [str(self.exclusion.relative_to(REPO))], "a"*64, **kwargs)

    def complete(self, run, instance):
        while True:
            value = advance_confirmation_instance(REPO, run, instance, max_chunks=2)
            if value["done"]:
                return

    def test_preparation_draws_no_fields_and_keeps_confirmation_role(self):
        run = self.prepare(sizes=(3, 4), count=2)
        plan = load_confirmation_plan(REPO, run)
        self.assertEqual(plan["role"], "confirmation")
        self.assertEqual(len(plan["instances"]), 4)
        self.assertEqual(len({r["hamiltonian_uid"] for r in plan["instances"]}), 4)
        self.assertTrue(all(r["role"] == "confirmation" and "split" not in r for r in plan["instances"]))
        self.assertNotIn("f"*64, {r["hamiltonian_uid"] for r in plan["instances"]})
        self.assertTrue(all(len(r["field_seed"]) == 5 for r in plan["instances"]))
        self.assertFalse((run/"instances").exists())
        self.assertEqual(json.loads((run/"access.json").read_text())["state"], "prepared_no_fields_drawn")
        self.assertEqual(self.payload.read_bytes(), b"This fixture payload must not be opened as an archive.")
        self.assertEqual(self.prepare(sizes=(3, 4), count=2), run)

    def test_pause_resume_matches_uninterrupted_stream_and_views(self):
        run = self.prepare()
        instance = load_confirmation_plan(REPO, run)["instances"][0]
        progress = advance_confirmation_instance(REPO, run, instance["id"], max_chunks=2)
        self.assertFalse(progress["done"])
        self.assertEqual(progress["sweeps"], 18)
        job = run/"instances"/instance["id"]
        pointer = json.loads((job/"checkpoint.json").read_text())
        prefix = decode_states(archive(job/pointer["path"])).copy()
        self.assertEqual(prefix.shape, (3, 18))
        self.complete(run, instance["id"])
        resumed = archive(job/"records.npz")
        np.testing.assert_array_equal(decode_states(resumed)[:len(prefix)], prefix)
        # Different checkpoint boundaries must not change the RNG stream.
        direct_sampling = {**self.sampling, "chunk_sweeps": 37}
        other = self.prepare("direct", sampling=direct_sampling)
        other_instance = load_confirmation_plan(REPO, other)["instances"][0]
        self.complete(other, other_instance["id"])
        direct = archive(other/"instances"/other_instance["id"]/"records.npz")
        for key in ("states_packed", "fields", "y", "x_f", "x_g", "block_means_f", "block_means_g"):
            np.testing.assert_array_equal(resumed[key], direct[key])
        records = decode_states(resumed)
        self.assertEqual(records.shape, (8, 18))
        paired = paired_estimators(square_lattice(3), resumed["fields"], records)
        for view in ("f", "g"):
            expected = np.stack((paired["star"][view].mean(0).reshape(3, 3),
                paired["pair"][view].mean(0)[0::2].reshape(3, 3),
                paired["pair"][view].mean(0)[1::2].reshape(3, 3))).astype(np.float32)
            np.testing.assert_allclose(resumed[f"x_{view}"], expected, rtol=1e-7, atol=1e-7)
        features = load_confirmation_features(run, 3)
        self.assertEqual(features["x_f"].shape, (1, 3, 3, 3))
        self.assertEqual(features["y"].shape, (1, 2, 3, 3))
        self.assertEqual(features["fields"].dtype, np.float64)
        self.assertEqual(features["role"], "confirmation")
        self.assertNotIn("states_packed", features)
        self.assertFalse((job/"checkpoint.json").exists())

    def test_completed_sweep_checkpoint_survives_finalization_pause(self):
        run = self.prepare(sampling={**self.sampling, "chunk_sweeps": 37})
        instance = load_confirmation_plan(REPO, run)["instances"][0]
        calls = []
        def stop_after_sample():
            calls.append(1)
            return len(calls) > 1
        progress = advance_confirmation_instance(REPO, run, instance["id"], stop_requested=stop_after_sample)
        self.assertFalse(progress["done"])
        self.assertEqual(progress["sweeps"], 37)
        self.complete(run, instance["id"])
        self.assertEqual(load_confirmation_features(run, 3)["x_g"].shape[0], 1)

    def test_exact_reference_optional_and_never_training_role(self):
        run = self.prepare(sampling={**self.sampling, "exact_small_sizes": True})
        instance = load_confirmation_plan(REPO, run)["instances"][0]
        self.complete(run, instance["id"])
        features = load_confirmation_features(run, 3)
        truth = exact_reference(square_lattice(3), features["fields"][0])["truth"]
        np.testing.assert_allclose(features["x_exact"][0, 0].reshape(-1), truth["star"], rtol=1e-7)
        self.assertTrue(features["known_fields_for_features"])
        self.assertNotIn("train", features)

    def test_invalid_schedule_paths_and_changed_plan_are_rejected(self):
        for bad in [{**self.sampling, "total_sweeps": 36},
                    {**self.sampling, "lazy_probability": float("nan")},
                    {**self.sampling, "unexpected": True}]:
            with self.assertRaises(ValueError):
                self.prepare(sampling=bad)
        with self.assertRaises(ValueError):
            prepare_confirmation(REPO, self.root.parent.parent/"escaped", [3], 1, 0,
                                 self.sampling, [], "a"*64)
        run = self.prepare()
        with self.assertRaises(ValueError):
            self.prepare(count=2)
        path = run/"plan.json"
        plan = json.loads(path.read_text())
        plan["role"] = "validation"
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "integrity|role"):
            load_confirmation_plan(REPO, run)

    def test_prior_uid_collision_binding_change_and_exclusion_change_refuse(self):
        run = self.prepare()
        uid = load_confirmation_plan(REPO, run)["instances"][0]["hamiltonian_uid"]
        self.exclusion.write_text(json.dumps({"instances": [{"hamiltonian_uid": uid}]}))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.prepare("collision")
        with self.assertRaisesRegex(ValueError, "Excluded"):
            load_confirmation_plan(REPO, run)
        model = self.root/"model.pt"
        model.write_bytes(b"fake frozen model weights")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.exclusion.write_text(json.dumps({"instances": []}))
        pinned = self.prepare("pinned", bindings={"model_sha256": {str(model.relative_to(REPO)): digest}})
        model.write_bytes(b"changed fake weights")
        with self.assertRaisesRegex(ValueError, "method file changed"):
            load_confirmation_plan(REPO, pinned)

    def test_checkpoint_and_completed_payload_corruption_refused(self):
        run = self.prepare()
        instance = load_confirmation_plan(REPO, run)["instances"][0]
        advance_confirmation_instance(REPO, run, instance["id"])
        job = run/"instances"/instance["id"]
        head = json.loads((job/"checkpoint.json").read_text())
        checkpoint = job/head["path"]
        checkpoint.write_bytes(checkpoint.read_bytes()+b"corrupt")
        with self.assertRaisesRegex(ValueError, "pointer/hash"):
            advance_confirmation_instance(REPO, run, instance["id"])
        other = self.prepare("complete")
        other_id = load_confirmation_plan(REPO, other)["instances"][0]["id"]
        self.complete(other, other_id)
        payload = other/"instances"/other_id/"records.npz"
        payload.write_bytes(payload.read_bytes()+b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_confirmation_features(other, 3)

    def test_parallel_scheduler_pauses_resumes_and_skips_completed(self):
        run = self.prepare(count=2)
        first = run_confirmation(REPO, run, workers=2, max_chunks=2)
        self.assertEqual(first["state"], "paused")
        self.assertEqual(first["chunks_this_call"], 2)
        complete = run_confirmation(REPO, run, workers=2)
        self.assertEqual((complete["state"], complete["completed"]), ("complete", 2))
        skipped = run_confirmation(REPO, run, workers=1)
        self.assertEqual(skipped["chunks_this_call"], 0)
        self.assertGreaterEqual(skipped["active_seconds"], complete["active_seconds"])
        self.assertEqual(load_confirmation_features(run, 3)["x_g"].shape[0], 2)

    def test_larger_map_layout_and_zero_budget_are_explicit(self):
        run = self.prepare(sizes=(8,), sampling={**self.sampling, "exact_small_sizes": True})
        paused = run_confirmation(REPO, run, max_seconds=0)
        self.assertEqual(paused["state"], "paused")
        self.assertFalse((run/"instances").exists())
        instance = load_confirmation_plan(REPO, run)["instances"][0]
        self.complete(run, instance["id"])
        features = load_confirmation_features(run, 8)
        self.assertEqual(features["x_f"].shape, (1, 3, 8, 8))
        self.assertEqual(features["y"].shape, (1, 2, 8, 8))
        self.assertNotIn("x_exact", features)


if __name__ == "__main__":
    unittest.main()
