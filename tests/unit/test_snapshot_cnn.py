"""Synthetic architecture controls; no research data or training campaign."""

import copy
import unittest

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.snapshot_cnn import (
        METHODS, make_snapshot_model, snapshot_model_spec)


@unittest.skipIf(torch is None, "Use the shared .venv with PyTorch")
class SnapshotCNNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.random.default_generator.manual_seed(412)
        self.raw = torch.randint(0, 2, (2, 7, 2, 3, 3), dtype=torch.int8) * 2 - 1

    def pair(self, chunk=3, checkpoint=True):
        torch.random.default_generator.manual_seed(74)
        mean = make_snapshot_model("mean_cnn", shot_chunk_size=chunk, activation_checkpoint=checkpoint)
        torch.random.default_generator.manual_seed(74)
        raw = make_snapshot_model("set_cnn", shot_chunk_size=chunk, activation_checkpoint=checkpoint)
        return mean, raw

    def test_parameter_matching_and_initial_state_are_exact(self):
        mean, raw = self.pair()
        self.assertEqual(sum(p.numel() for p in mean.parameters()), 38690)
        self.assertEqual(set(mean.state_dict()), set(raw.state_dict()))
        for name, tensor in mean.state_dict().items():
            self.assertTrue(torch.equal(tensor, raw.state_dict()[name]), name)
        self.assertEqual(mean(self.raw.float().mean(1)).shape, (2, 2, 3, 3))
        self.assertEqual(raw(self.raw).shape, (2, 2, 3, 3))
        self.assertFalse(any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)) for m in raw.modules()))

    def test_shot_permutation_replication_and_translation_invariance(self):
        _, model = self.pair()
        model.eval()
        with torch.no_grad():
            expected = model(self.raw)
            actual = model(self.raw[:, [4, 1, 6, 0, 3, 5, 2]])
            repeated = model(self.raw.repeat(1, 2, 1, 1, 1))
            translated = model(torch.roll(self.raw, (1, -1), (-2, -1)))
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(repeated, expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(translated, torch.roll(expected, (1, -1), (-2, -1)), atol=2e-6, rtol=1e-5)

    def test_single_shot_equals_mean_model_and_chunks_match(self):
        mean, raw = self.pair()
        mean.eval(); raw.eval()
        with torch.no_grad():
            torch.testing.assert_close(mean(self.raw[:, 0]), raw(self.raw[:, :1]), atol=0, rtol=0)
            expected = raw(self.raw)
            for chunk in (1, 2, 7, 20):
                other = make_snapshot_model("set_cnn", shot_chunk_size=chunk)
                other.load_state_dict(raw.state_dict()); other.eval()
                torch.testing.assert_close(other(self.raw), expected, atol=2e-6, rtol=1e-5)

    def test_chunking_and_checkpointing_preserve_input_and_parameter_gradients(self):
        _, reference = self.pair(chunk=7, checkpoint=False)
        actual = make_snapshot_model("set_cnn", shot_chunk_size=2, activation_checkpoint=True)
        actual.load_state_dict(reference.state_dict())
        x = self.raw.float().requires_grad_()
        y = self.raw.float().requires_grad_()
        expected = reference(x)
        observed = actual(y)
        expected.square().mean().backward()
        observed.square().mean().backward()
        torch.testing.assert_close(observed, expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(y.grad, x.grad, atol=1e-7, rtol=3e-5)
        for (name, first), (_, second) in zip(reference.named_parameters(), actual.named_parameters()):
            torch.testing.assert_close(second.grad, first.grad, atol=1e-7, rtol=3e-5, msg=name)

    @staticmethod
    def identity_positive_path(model):
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            for index in (0, 2, 4):
                model.encoder[index].weight[0, 0, 1, 1] = 1
            model.head[0].weight[0, 0, 0, 0] = 1
            model.head[2].weight[0, 0, 0, 0] = 1

    def test_nonlinearity_preserves_information_lost_by_mean_before_encoder(self):
        mean, raw = self.pair()
        for model in (mean, raw):
            self.identity_positive_path(model)
        shots = torch.stack((torch.ones(2, 3, 3), -torch.ones(2, 3, 3)))[None]
        with torch.no_grad():
            first = mean(shots.mean(1))
            second = raw(shots)
        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(second[:, 0], torch.full_like(second[:, 0], .5)))
        self.assertTrue(torch.equal(second[:, 1], torch.zeros_like(second[:, 1])))

    def test_nonlinear_rho_applies_after_pooling_not_additive_per_shot_predictions(self):
        _, raw = self.pair()
        self.identity_positive_path(raw)
        with torch.no_grad():
            raw.head[0].bias[0] = -.75
        shots = torch.stack((torch.ones(2, 3, 3), -torch.ones(2, 3, 3)))[None]
        with torch.no_grad():
            pooled = raw(shots)
            additive = (raw(shots[:, :1]) + raw(shots[:, 1:])) / 2
        self.assertTrue(torch.equal(pooled, torch.zeros_like(pooled)))
        self.assertTrue(torch.equal(additive[:, 0], torch.full_like(additive[:, 0], .125)))

    def test_bad_shapes_options_and_owned_metadata(self):
        _, raw = self.pair()
        for bad in (self.raw[:, :0], self.raw[:, :, :1], torch.ones(2, 2, 3, 3),
                    self.raw.bool(), torch.ones(2, 3, 2, 2, 2)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                raw(bad)
        for method, options in (("unknown", {}), ("set_cnn", {"shot_chunk_size": 0}),
                                ("set_cnn", {"shot_chunk_size": True}),
                                ("mean_cnn", {"activation_checkpoint": 1})):
            with self.subTest(method=method, options=options), self.assertRaises(ValueError):
                make_snapshot_model(method, **options)
        before = torch.get_rng_state().clone()
        for method in METHODS:
            spec = snapshot_model_spec(method)
            spec["hidden_channels"][0] = 99
            self.assertEqual(snapshot_model_spec(method)["hidden_channels"], [32, 64, 32])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
