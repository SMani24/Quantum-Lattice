"""Short synthetic CUDA controls; never accesses a data bank or research queue."""

import io
import os
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from quantum_lattice.models.snapshot_cnn import METHODS, make_snapshot_model
from quantum_lattice.models.snapshot_training import SnapshotTrainSession


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for this bounded integration control")
class SnapshotCudaTests(unittest.TestCase):
    def assert_tree_equal(self, first, second):
        self.assertIs(type(first), type(second))
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_both_methods_l3_l4_cuda_partial_resume_exact(self):
        for L in (3, 4):
            rng = np.random.default_rng(371 + L)
            raw = rng.integers(0, 2, (5, 7, 2, L, L), dtype=np.int8) * 2 - 1
            vraw = rng.integers(0, 2, (3, 7, 2, L, L), dtype=np.int8) * 2 - 1
            y = rng.uniform(0, .8, (5, 2, L, L)).astype(np.float32)
            vy = rng.uniform(0, .8, (3, 2, L, L)).astype(np.float32)
            config = {"device": "cuda", "seed": 37, "batch_size_hamiltonians": 2,
                      "validation_batch_size": 2, "max_epochs": 2, "shot_chunk_size": 3,
                      "activation_checkpoint": True, "early_stopping": {"enabled": False}}
            for method in METHODS:
                with self.subTest(method=method, L=L):
                    x = raw if method == "set_cnn" else raw.astype(np.float32).mean(1)
                    vx = vraw if method == "set_cnn" else vraw.astype(np.float32).mean(1)
                    direct = SnapshotTrainSession(x, y, vx, vy, config, method=method)
                    direct.advance(6)
                    expected = direct.snapshot()
                    prediction = direct.selected_predictions(vx)
                    del direct
                    partial = SnapshotTrainSession(x, y, vx, vy, config, method=method)
                    partial.advance(2)
                    stream = io.BytesIO(); torch.save(partial.snapshot(), stream); stream.seek(0)
                    del partial
                    loaded = torch.load(stream, map_location="cpu", weights_only=True)
                    resumed = SnapshotTrainSession(x, y, vx, vy, config, method=method, checkpoint=loaded)
                    resumed.advance(4)
                    self.assert_tree_equal(expected, resumed.snapshot())
                    np.testing.assert_array_equal(prediction, resumed.selected_predictions(vx))
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                    del resumed

    def test_cuda_chunk_checkpoint_gradients_and_shot_permutation(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.random.default_generator.manual_seed(72)
        reference = make_snapshot_model("set_cnn", shot_chunk_size=65,
                                        activation_checkpoint=False).cuda()
        chunked = make_snapshot_model("set_cnn", shot_chunk_size=7,
                                      activation_checkpoint=True).cuda()
        chunked.load_state_dict(reference.state_dict())
        generator = torch.Generator(device="cpu").manual_seed(82)
        raw = (torch.randint(0, 2, (2, 65, 2, 4, 4), generator=generator) * 2 - 1).float().cuda()
        x, z = raw.detach().clone().requires_grad_(), raw.detach().clone().requires_grad_()
        expected, actual = reference(x), chunked(z)
        expected.square().mean().backward(); actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
        torch.testing.assert_close(x.grad, z.grad, atol=1e-7, rtol=1e-4)
        for (name, a), (_, b) in zip(reference.named_parameters(), chunked.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-4, msg=name)
        chunked.eval()
        with torch.no_grad():
            permuted = chunked(raw[:, torch.randperm(65, generator=generator)])
        torch.testing.assert_close(permuted, actual.detach(), atol=2e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
