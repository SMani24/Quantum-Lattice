"""Tiny synthetic CUDA exact-resume checks; no research data or campaign."""
import io
import unittest
import numpy as np
import torch
from quantum_lattice.models.cnn_ablation_training import CnnAblationTrainSession
from quantum_lattice.models.cnn_variants import VARIANT_NAMES


def assert_tree(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for k in left:
            assert_tree(case, left[k], right[k])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree(case, a, b)
    else:
        case.assertEqual(left, right)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for this integration control")
class CnnAblationCudaTests(unittest.TestCase):
    def test_every_variant_partial_epoch_exact_resume(self):
        for L in (3, 4):
            rng = np.random.default_rng(291 + L)
            arrays = (rng.uniform(.1, 1, (7, 3, L, L)).astype(np.float32),
                      rng.uniform(0, 1.7, (7, 2, L, L)).astype(np.float32),
                      rng.uniform(.1, 1, (3, 3, L, L)).astype(np.float32),
                      rng.uniform(0, 1.7, (3, 2, L, L)).astype(np.float32))
            config = {"model": "cnn", "architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 3},
                      "device": "cuda", "seed": 99, "max_epochs": 2, "batch_size_hamiltonians": 3,
                      "validation_batch_size": 2, "early_stopping": {"enabled": False}}
            for variant in VARIANT_NAMES:
                with self.subTest(L=L, variant=variant):
                    continuous = CnnAblationTrainSession(*arrays, config, variant=variant)
                    continuous.advance(6)
                    expected = continuous.snapshot()
                    pred = continuous.selected_predictions(arrays[2])
                    del continuous
                    partial = CnnAblationTrainSession(*arrays, config, variant=variant)
                    partial.advance(2)
                    buffer = io.BytesIO()
                    torch.save(partial.snapshot(), buffer)
                    del partial
                    buffer.seek(0)
                    resumed = CnnAblationTrainSession(*arrays, config, variant=variant,
                        checkpoint=torch.load(buffer, weights_only=True))
                    resumed.advance(4)
                    assert_tree(self, expected, resumed.snapshot())
                    np.testing.assert_array_equal(pred, resumed.selected_predictions(arrays[2]))
                    del resumed


if __name__ == "__main__":
    unittest.main()
