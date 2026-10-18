"""Bounded CUDA controls on synthetic maps; never starts the research queue."""
import io
import unittest

import numpy as np
import torch

from quantum_lattice.models.observable_training import ObservableTrainSession


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for this integration control")
class ObservableCudaTests(unittest.TestCase):
    def equal_tree(self, a, b):
        self.assertIs(type(a), type(b))
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.equal_tree(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.equal_tree(left, right)
        else:
            self.assertEqual(a, b)

    def check_size(self, L):
        rng = np.random.default_rng(188 + L)
        arrays = (rng.uniform(0.1, 1.2, (5, 3, L, L)).astype(np.float32),
                  rng.uniform(0.0, 0.7, (5, 2, L, L)).astype(np.float32),
                  rng.uniform(0.1, 1.2, (3, 3, L, L)).astype(np.float32),
                  rng.uniform(0.0, 0.7, (3, 2, L, L)).astype(np.float32))
        for model in ("cnn", "map_mlp"):
            with self.subTest(model=model, L=L):
                config = {"model": model, "device": "cuda", "seed": 2026101898,
                          "batch_size_hamiltonians": 2, "validation_batch_size": 2,
                          "max_epochs": 3, "early_stopping": {"enabled": False}}
                straight = ObservableTrainSession(*arrays, config)
                straight.advance(7)
                expected = straight.snapshot()
                prediction = straight.selected_predictions(arrays[2])
                self.equal_tree(expected, straight.snapshot())
                split = ObservableTrainSession(*arrays, config)
                split.advance(2)  # Stop inside the first epoch.
                buffer = io.BytesIO()
                torch.save(split.snapshot(), buffer)
                buffer.seek(0)
                loaded = torch.load(buffer, map_location="cpu", weights_only=True)
                resumed = ObservableTrainSession(*arrays, config, checkpoint=loaded)
                resumed.advance(5)
                self.equal_tree(expected, resumed.snapshot())
                np.testing.assert_array_equal(prediction, resumed.selected_predictions(arrays[2]))
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_l3_exact_resume_both_architectures(self):
        self.check_size(3)

    def test_l4_exact_resume_both_architectures(self):
        self.check_size(4)


if __name__ == "__main__":
    unittest.main()
