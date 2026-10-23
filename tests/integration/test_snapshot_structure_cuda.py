"""Bounded actual-CUDA structural controls; never touches research data."""

import io
import os
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from quantum_lattice.data.snapshot_gauge import vertex_gauge_masks
from quantum_lattice.models.snapshot_structure import make_structure_model
from quantum_lattice.models.snapshot_structure_training import StructureTrainSession


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for bounded integration control")
class StructureCudaTests(unittest.TestCase):
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

    def test_residual_augmented_both_methods_cuda_partial_epoch_resume(self):
        rng = np.random.default_rng(3001)
        raw = rng.integers(0, 2, (5, 7, 2, 4, 4), dtype=np.int8) * 2 - 1
        vraw = rng.integers(0, 2, (3, 7, 2, 4, 4), dtype=np.int8) * 2 - 1
        y = rng.uniform(.05, .9, (5, 2, 4, 4)).astype(np.float32)
        vy = rng.uniform(.05, .9, (3, 2, 4, 4)).astype(np.float32)
        uids = [f"cuda-structural-H-{i}" for i in range(5)]
        config = {"device": "cuda", "seed": 311, "batch_size_hamiltonians": 2,
                  "validation_batch_size": 2, "max_epochs": 3, "shot_chunk_size": 3,
                  "activation_checkpoint": True, "early_stopping": {"enabled": False}}
        for method in ("mean_cnn", "set_cnn"):
            with self.subTest(method=method):
                x = raw if method == "set_cnn" else raw.astype(np.float32).mean(1)
                vx = vraw if method == "set_cnn" else vraw.astype(np.float32).mean(1)
                arguments = dict(method=method, residual=True, gauge_augmentation=True, train_uids=uids)
                direct = StructureTrainSession(x, y, vx, vy, config, **arguments)
                direct.advance(9)
                expected = direct.snapshot()
                expected_predictions = direct.selected_predictions(vx)
                del direct
                partial = StructureTrainSession(x, y, vx, vy, config, **arguments)
                partial.advance(4)
                self.assertEqual((partial.epoch, partial.cursor), (1, 2))
                stream = io.BytesIO()
                torch.save(partial.snapshot(), stream)
                stream.seek(0)
                del partial
                saved = torch.load(stream, map_location="cpu", weights_only=True)
                resumed = StructureTrainSession(x, y, vx, vy, config, checkpoint=saved, **arguments)
                resumed.advance(5)
                self.assert_tree_equal(expected, resumed.snapshot())
                np.testing.assert_array_equal(expected_predictions, resumed.selected_predictions(vx))
                np.testing.assert_array_equal(resumed._epoch_masks.cpu().numpy(),
                                              vertex_gauge_masks(uids, 4, seed=311, epoch=2))
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_augmented_residual_chunk_recompute_forward_and_gradients(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.random.default_generator.manual_seed(161)
        whole = make_structure_model("set_cnn", residual=True, shot_chunk_size=65,
                                     activation_checkpoint=False).cuda()
        chunked = make_structure_model("set_cnn", residual=True, shot_chunk_size=7,
                                       activation_checkpoint=True).cuda()
        chunked.load_state_dict(whole.state_dict())
        generator = torch.Generator(device="cpu").manual_seed(52)
        raw = (torch.randint(0, 2, (2, 65, 2, 4, 4), generator=generator) * 2 - 1).float()
        mask = torch.from_numpy(vertex_gauge_masks(["cuda-gradient-H0", "cuda-gradient-H1"],
                                                    4, seed=52, epoch=3))
        augmented = (raw * mask[:, None]).cuda()
        a = augmented.detach().clone().requires_grad_()
        b = augmented.detach().clone().requires_grad_()
        expected, actual = whole(a), chunked(b)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-4)
        for (name, left), (_, right) in zip(whole.named_parameters(), chunked.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=1e-4, msg=name)
        chunked.eval()
        with torch.no_grad():
            permuted = chunked(augmented[:, torch.randperm(65, generator=generator)])
        torch.testing.assert_close(permuted, actual.detach(), atol=2e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
