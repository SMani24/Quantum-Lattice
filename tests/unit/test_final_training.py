"""Durable training/selection boundaries on tiny synthetic CPU fixtures."""
import copy
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
    from quantum_lattice.models.final_training import (
        checkpoint_path, load_job, predict_checkpoint, train_job)
    from quantum_lattice.models.observable_cnn import PeriodicObservableCNN


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class FinalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(701)
        self.arrays = [rng.uniform(.1, 1, shape).astype(np.float32) for shape in
                       [(5, 3, 3, 3), (5, 2, 3, 3), (3, 3, 3, 3), (3, 2, 3, 3)]]
        self.settings = {"model": "cnn", "architecture": {"hidden_channels": [3], "kernel_size": 3},
                         "seed": 73, "device": "cpu", "deterministic": True,
                         "torch_threads": 1, "learning_rate": .001,
                         "batch_size_hamiltonians": 2, "max_epochs": 3,
                         "validation_batch_size": 2,
                         "early_stopping": {"enabled": False, "min_epochs": 1,
                                            "patience_epochs": 2, "min_delta": 1e-6}}
        self.bindings = {"task": "fixture-L3-cnn-f", "cache_sha256": "fixed-test-cache",
                         "scope": "synthetic train/validation only"}

    def run_job(self, name, *, settings=None, arrays=None, bindings=None, **kwargs):
        return train_job(*(self.arrays if arrays is None else arrays),
            self.settings if settings is None else settings, self.root / name,
            self.bindings if bindings is None else bindings,
            checkpoint_updates=2, allow_cpu=True, **kwargs)

    def assert_tree_equal(self, a, b):
        self.assertIs(type(a), type(b))
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_pause_resume_replays_same_optimizer_rng_and_best_for_both_models(self):
        for model in ("cnn", "map_mlp"):
            with self.subTest(model=model):
                settings = {**self.settings, "model": model}
                direct = self.run_job(model+"-direct", settings=settings)
                paused = self.run_job(model+"-resumed", settings=settings, max_updates=4)
                self.assertEqual((paused["state"], paused["step"], paused["epoch"]), ("paused", 4, 1))
                resumed = self.run_job(model+"-resumed", settings=settings)
                self.assertEqual(resumed["state"], "complete")
                self.assertGreater(resumed["active_seconds"], paused["active_seconds"])
                a = torch.load(direct["checkpoint_paths"]["latest"], weights_only=True)["session"]
                b = torch.load(resumed["checkpoint_paths"]["latest"], weights_only=True)["session"]
                self.assert_tree_equal(a, b)
                for kind in ("best", "latest"):
                    self.assertTrue(checkpoint_path(self.root/(model+"-resumed"), kind).is_file())
                generations = list((self.root/(model+"-resumed")/"checkpoints").iterdir())
                self.assertEqual(len(generations), 2)

    def test_done_job_skips_optimizer_and_tensor_reload(self):
        completed = self.run_job("done")
        with mock.patch("quantum_lattice.models.final_training.ObservableTrainSession",
                        side_effect=AssertionError("completed job must not train")), \
             mock.patch("torch.load", side_effect=AssertionError("status must not load tensors")):
            repeated = self.run_job("done")
            status = load_job(self.root/"done")
        self.assertEqual({k: v for k, v in completed.items() if k != "invocation_seconds"},
                         {k: v for k, v in repeated.items() if k != "invocation_seconds"})
        self.assertEqual(status["state"], "complete")
        pointer = json.loads((self.root/"done"/"state.json").read_text())
        self.assertEqual(status["commit_sha256"], pointer["commit_sha256"])

    def test_zero_budget_and_requested_pause_never_construct_optimizer(self):
        for name, kwargs in [("zero", {"max_seconds": 0}),
                             ("requested", {"stop_requested": lambda: True}),
                             ("zero-updates", {"max_updates": 0})]:
            with mock.patch("quantum_lattice.models.final_training.ObservableTrainSession",
                            side_effect=AssertionError("no training allowed")):
                result = self.run_job(name, **kwargs)
            self.assertEqual(result["state"], "pending")
            self.assertEqual(result["checkpoint_paths"], {})
            self.assertFalse((self.root/name/"state.json").exists())

    def test_stop_callback_saves_atomic_checkpoint_and_resumes(self):
        observed = []
        result = self.run_job("stopped", stop_requested=lambda: bool(observed),
                              progress=lambda state: observed.append(state["step"]))
        self.assertTrue(observed)
        self.assertEqual(result["state"], "paused")
        self.assertEqual(result["stop_reason"], "requested")
        result = self.run_job("stopped")
        self.assertEqual(result["state"], "complete")

    def test_unpublished_generation_after_io_failure_preserves_previous_resume(self):
        import quantum_lattice.models.final_training as engine
        saved = self.run_job("interrupted", max_updates=2)
        writer = engine._json
        def fail_pointer(path, value):
            if Path(path).name == "state.json":
                raise OSError("simulated failure before generation publication")
            return writer(path, value)
        with mock.patch.object(engine, "_json", fail_pointer), self.assertRaises(OSError):
            self.run_job("interrupted")
        current = load_job(self.root/"interrupted")
        self.assertEqual(current["generation"], saved["generation"])
        self.assertEqual(current["step"], saved["step"])
        resumed = self.run_job("interrupted")
        direct = self.run_job("uninterrupted")
        a = torch.load(resumed["checkpoint_paths"]["latest"], weights_only=True)["session"]
        b = torch.load(direct["checkpoint_paths"]["latest"], weights_only=True)["session"]
        self.assert_tree_equal(a, b)

    def test_interrupted_initial_plan_temporary_file_can_resume_before_any_updates(self):
        job = self.root/"initial-temp"
        job.mkdir()
        (job/"job-plan-sha256.json.tmp").write_text("incomplete unpublished plan")
        result = self.run_job("initial-temp", max_updates=1)
        self.assertEqual((result["state"], result["step"]), ("paused", 1))

    def test_changed_data_settings_and_binding_are_rejected_before_resume(self):
        self.run_job("bound", max_updates=2)
        arrays = [a.copy() for a in self.arrays]
        arrays[0][0, 0, 0, 0] += .01
        for changes in [dict(arrays=arrays), dict(settings={**self.settings, "seed": 74}),
                        dict(bindings={**self.bindings, "task": "different-view"})]:
            with self.subTest(changes=changes.keys()), self.assertRaisesRegex(ValueError, "changed"):
                self.run_job("bound", **changes)

    def test_tensor_history_plan_and_done_corruption_are_detected(self):
        for name in ("best.pt", "latest.pt", "history.json", "job-plan.json", "done.json"):
            with self.subTest(file=name):
                job_name = name.replace(".", "-")
                result = self.run_job(job_name)
                if name in {"job-plan.json", "done.json"}:
                    path = self.root/job_name/name
                else:
                    path = Path(result["checkpoint_paths"]["best"]).parent/name
                with path.open("ab") as handle:
                    handle.write(b" ")
                if name in {"job-plan.json", "done.json"}:
                    # Trailing whitespace leaves JSON semantically unchanged;
                    # alter a semantic field for metadata corruption checks.
                    d = json.loads(path.read_text())
                    d["schema_version"] = 999
                    path.write_text(json.dumps(d))
                with self.assertRaises(ValueError):
                    load_job(self.root/job_name)

    def test_prediction_matches_selected_state_without_adam_or_rng_changes(self):
        result = self.run_job("predict")
        path = result["checkpoint_paths"]["best"]
        payload = torch.load(path, weights_only=True)
        model = PeriodicObservableCNN(**self.settings["architecture"])
        model.load_state_dict(payload["model_state_dict"])
        with torch.no_grad():
            expected = model(torch.tensor(self.arrays[2])).numpy()
        rng_before = torch.get_rng_state().clone()
        with mock.patch("torch.optim.Adam", side_effect=AssertionError("inference cannot train")):
            prediction = predict_checkpoint(path, self.arrays[2], device="cpu", allow_cpu=True, batch_size=1)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "selected best"):
            predict_checkpoint(result["checkpoint_paths"]["latest"], self.arrays[2], device="cpu", allow_cpu=True)

    def test_legacy_selected_outputs_are_unclipped_and_transfer_explicit(self):
        folder = self.root/"legacy"
        folder.mkdir()
        model = PeriodicObservableCNN(**self.settings["architecture"])
        weights = model.state_dict()
        for value in weights.values():
            value.zero_()
        weights["layers.2.bias"].fill_(-.5)
        task = {"L": 3, "seed": 73, "model": "cnn", "view": "f"}
        path = folder/"best.pt"
        torch.save({"schema_version": 1, "plan_sha256": "legacy-plan", "task": task,
                    "architecture": self.settings["architecture"], "model_state_dict": weights}, path)
        done = {"schema_version": 1, "plan_sha256": "legacy-plan", "task": task,
                "hashes": {"best.pt": hashlib.sha256(path.read_bytes()).hexdigest()}}
        (folder/"done.json").write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError, "allow_legacy"):
            predict_checkpoint(path, self.arrays[2], device="cpu", allow_cpu=True)
        with mock.patch("torch.optim.Adam", side_effect=AssertionError("inference cannot train")):
            result = predict_checkpoint(path, self.arrays[2], device="cpu", allow_cpu=True, allow_legacy=True)
        np.testing.assert_array_equal(result, np.full((3, 2, 3, 3), -.5, dtype=np.float32))
        larger = np.ones((2, 3, 8, 8), np.float32)
        with self.assertRaisesRegex(ValueError, "size transfer"):
            predict_checkpoint(path, larger, device="cpu", allow_cpu=True, allow_legacy=True)
        result = predict_checkpoint(path, larger, device="cpu", allow_cpu=True,
                                    allow_legacy=True, allow_size_transfer=True)
        np.testing.assert_array_equal(result, np.full((2, 2, 8, 8), -.5, dtype=np.float32))

    def test_larger_size_training_accepts_readonly_maps(self):
        arrays = [np.ones(shape, np.float32) for shape in
                  [(2, 3, 8, 8), (2, 2, 8, 8), (1, 3, 8, 8), (1, 2, 8, 8)]]
        for array in arrays:
            array.flags.writeable = False
        result = self.run_job("large", arrays=arrays,
                              settings={**self.settings, "max_epochs": 1})
        self.assertEqual((result["L"], result["epoch"], result["state"]), (8, 1, "complete"))

    def test_inference_enforces_and_restores_determinism_policy(self):
        result = self.run_job("policy")
        path = result["checkpoint_paths"]["best"]
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        observed = []
        forward = PeriodicObservableCNN.forward
        def inspect(model, values):
            observed.append((torch.are_deterministic_algorithms_enabled(),
                             torch.backends.cudnn.benchmark,
                             torch.backends.cuda.matmul.allow_tf32))
            return forward(model, values)
        try:
            with mock.patch.object(PeriodicObservableCNN, "forward", inspect):
                predict_checkpoint(path, self.arrays[2], device="cpu", allow_cpu=True)
            self.assertEqual(observed, [(True, False, False)])
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_cpu_requires_explicit_optin_and_cuda_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            train_job(*self.arrays, self.settings, self.root/"implicitcpu", self.bindings)
        with mock.patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(ValueError, "unavailable"):
            train_job(*self.arrays, {**self.settings, "device": "cuda"}, self.root/"cuda", self.bindings)
        self.assertFalse((self.root/"cuda").exists())

    def test_invalid_budgets_and_nonempty_directory_refused(self):
        for kwargs in [{"max_seconds": -1}, {"max_seconds": float("nan")},
                       {"max_updates": True}, {"checkpoint_updates": 0}]:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                # Call directly because run_job fixes the chunk-size argument.
                train_job(*self.arrays, self.settings, self.root/"bad", self.bindings,
                          allow_cpu=True, **kwargs)
        directory = self.root/"occupied"
        directory.mkdir()
        (directory/"keep.txt").write_text("user material")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.run_job("occupied")
        self.assertEqual((directory/"keep.txt").read_text(), "user material")


if __name__ == "__main__":
    unittest.main()
