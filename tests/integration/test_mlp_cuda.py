"""Device preflight and short synthetic CUDA checkpoint checks.

The CUDA tests skip on hosts without a working NVIDIA device. They never open
the Monte Carlo banks or train research models.
"""

import io
import os
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.mlp_training import TrainSession, resolve_device


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class DevicePreflightTests(unittest.TestCase):
    def test_workspace_is_configured_before_cuda_query(self):
        def unavailable():
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            return False

        with patch.dict(os.environ, {}, clear=True), patch(
            "torch.cuda.is_available", side_effect=unavailable
        ):
            self.assertEqual(resolve_device("auto").type, "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                resolve_device("cuda")

    def test_explicit_cpu_does_not_touch_cuda_or_existing_workspace(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "external"}), patch(
            "torch.cuda.is_available", side_effect=AssertionError("CUDA queried for CPU")
        ):
            self.assertEqual(resolve_device("cpu").type, "cpu")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], "external")

    def test_supported_workspace_is_preserved_and_conflicts_fail_clearly(self):
        for setting in (":16:8", ":4096:8"):
            with self.subTest(setting=setting), patch.dict(
                os.environ, {"CUBLAS_WORKSPACE_CONFIG": setting}
            ), patch("torch.cuda.is_available", return_value=True):
                self.assertEqual(resolve_device("cuda").type, "cuda")
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], setting)
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "invalid"}), patch(
            "torch.cuda.is_available", side_effect=AssertionError("Queried before rejection")
        ):
            with self.assertRaisesRegex(ValueError, "before starting Python"):
                resolve_device("cuda")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], "invalid")

    def test_requested_gpu_fails_before_large_arrays_are_read(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "quantum_lattice.models.mlp_training._array",
            side_effect=AssertionError("Read arrays before device preflight"),
        ), patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                TrainSession(None, None, None, None, {"device": "cuda"})


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "Requires CUDA GPU")
class CudaTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        self.train_x = rng.uniform(size=(61, 3)).astype(np.float32)
        self.validation_x = rng.uniform(size=(19, 3)).astype(np.float32)
        self.train_y = 0.4 * self.train_x[:, :1] + 0.6 * self.train_x[:, 1:2]
        self.validation_y = 0.4 * self.validation_x[:, :1] + 0.6 * self.validation_x[:, 1:2]
        self.config = {
            "seed": 23, "learning_rate": 0.003, "batch_size": 13,
            "max_steps": 35, "validation_interval": 7, "device": "cuda",
            "torch_threads": 1, "deterministic": True,
        }

    def session(self, config=None, checkpoint=None):
        return TrainSession(self.train_x, self.train_y, self.validation_x,
                            self.validation_y, config or self.config, checkpoint)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_cuda_data_and_model_are_resident_with_independent_input_copies(self):
        session = self.session()
        for value in (session.train_x, session.train_y, session.validation_x,
                      session.validation_y, *session.model.parameters()):
            self.assertEqual(value.device.type, "cuda")
        copied = self.train_x.copy()
        self.train_x[:] = 9
        np.testing.assert_array_equal(session.train_x.cpu().numpy(), copied)
        session.advance(1)
        self.assertEqual(session.permutation.device.type, "cpu")
        self.assertEqual(session._device_permutation.device.type, "cuda")
        self.assertTrue(torch.equal(session.permutation, session._device_permutation.cpu()))
        snapshot = session.snapshot()
        self.assertEqual(snapshot["device_type"], "cuda")
        self.assertEqual(snapshot["permutation"].device.type, "cpu")
        self.assertTrue(snapshot["cuda_rng_states"])

    def test_actual_cuda_save_resume_matches_uninterrupted_adam_rng_and_selection(self):
        full = self.session()
        full_events = full.advance(35)
        full_snapshot = full.snapshot()
        partial = self.session()
        partial_events = partial.advance(12)
        saved = io.BytesIO()
        torch.save(partial.snapshot(), saved)
        saved.seek(0)
        checkpoint = torch.load(saved, weights_only=True, map_location="cpu")
        restored = self.session(checkpoint=checkpoint)
        later_events = restored.advance(23)
        self.assertEqual(full_events, partial_events + later_events)
        self.assert_tree_equal(full_snapshot, restored.snapshot())
        np.testing.assert_array_equal(full.predict(self.validation_x), restored.predict(self.validation_x))

    def test_cuda_best_weights_remain_selected_after_later_worsening(self):
        session = self.session(dict(self.config, validation_interval=1))
        session.advance(1)
        best = session.best_state_dict
        with torch.no_grad():
            session.model.layers[-1].bias.fill_(100)
        session.advance(1)
        self.assertEqual(session.best_step, 1)
        self.assert_tree_equal(best, session.best_state_dict)
        self.assertGreater(float(session.predict(self.validation_x, use_best=False).mean()), 90)
        self.assertLess(float(session.predict(self.validation_x, use_best=True).mean()), 2)

    def test_cuda_chunked_validation_matches_independent_numpy_metric(self):
        rng = np.random.default_rng(92)
        self.validation_x = rng.uniform(size=(65539, 3)).astype(np.float32)
        self.validation_y = 0.4 * self.validation_x[:, :1] + 0.6 * self.validation_x[:, 1:2]
        # A distinctive final partial chunk catches dropped tail rows.
        self.validation_y[-3:] = 20
        session = self.session()
        prediction = session.predict(self.validation_x, use_best=False)
        difference = (prediction - self.validation_y).astype(np.float64)
        expected = float(np.mean(difference ** 2))
        actual = session._mse(session.validation_x, session.validation_y)
        # CUDA chooses different GEMM kernels for inference chunk shapes;
        # retain floating-point tolerance while checking the independent metric.
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)

    def test_cuda_early_stopping_continues_exactly_after_checkpoint(self):
        config = dict(self.config, validation_interval=2, early_stopping={
            "enabled": True, "patience": 3, "min_delta": 10, "min_steps": 0,
        })
        full = self.session(config)
        events = full.advance(35)
        snapshot = full.snapshot()
        partial = self.session(config)
        earlier = partial.advance(3)
        restored = self.session(config, partial.snapshot())
        later = restored.advance(35)
        self.assertTrue(restored.stopped_early)
        self.assertEqual(restored.stop_reason, "early_stopping")
        self.assertEqual(events, earlier + later)
        self.assert_tree_equal(snapshot, restored.snapshot())

    def test_cuda_nonfinite_gradient_does_not_advance_adam(self):
        session = self.session()
        first_parameter = next(session.model.parameters())
        initial = first_parameter.detach().clone()
        handle = first_parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        try:
            with self.assertRaisesRegex(FloatingPointError, "gradients"):
                session.advance(1)
        finally:
            handle.remove()
        self.assertEqual(session.step, 0)
        self.assertTrue(torch.equal(initial, first_parameter))
        self.assertFalse(session.optimizer.state)


if __name__ == "__main__":
    unittest.main()
