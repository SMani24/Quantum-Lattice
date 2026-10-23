"""Synthetic structure controls; no scientific data or campaign training."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from quantum_lattice.models.snapshot_cnn import make_snapshot_model
    from quantum_lattice.models.snapshot_structure import (
        METHODS, make_structure_model, structure_model_spec)


@unittest.skipIf(torch is None, "Use the shared .venv with PyTorch")
class SnapshotStructureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(284)
        self.raw = torch.randint(0, 2, (2, 7, 2, 4, 4), dtype=torch.int8) * 2 - 1

    @staticmethod
    def build(method, residual, **options):
        torch.manual_seed(913)
        return make_structure_model(method, residual=residual, **options)

    def test_plain_models_match_frozen_parameters_outputs_and_rng_exactly(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(913)
                frozen = make_snapshot_model(method, shot_chunk_size=3)
                expected_rng = torch.get_rng_state().clone()
                actual = self.build(method, False, shot_chunk_size=3)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertEqual(set(frozen.state_dict()), set(actual.state_dict()))
                for name, tensor in frozen.state_dict().items():
                    self.assertTrue(torch.equal(tensor, actual.state_dict()[name]), name)
                inputs = self.raw.float().mean(1) if method == "mean_cnn" else self.raw
                with torch.no_grad():
                    self.assertTrue(torch.equal(frozen(inputs), actual(inputs)))

    def test_all_four_treatments_share_names_initial_tensors_and_parameter_count(self):
        reference = self.build("mean_cnn", False)
        expected_rng = torch.get_rng_state().clone()
        for method in METHODS:
            for residual in (False, True):
                with self.subTest(method=method, residual=residual):
                    model = self.build(method, residual)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), 38690)
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertEqual(set(reference.state_dict()), set(model.state_dict()))
                    for name, tensor in reference.state_dict().items():
                        self.assertTrue(torch.equal(tensor, model.state_dict()[name]), name)
                    self.assertFalse(any(isinstance(layer, (torch.nn.BatchNorm2d,
                                                           torch.nn.GroupNorm))
                                         for layer in model.modules()))

    def test_residual_is_added_before_final_encoder_relu_with_no_projection(self):
        plain = self.build("mean_cnn", False)
        residual = self.build("mean_cnn", True)
        for model in (plain, residual):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.encoder[0].weight[0, 0, 1, 1] = 1
                model.encoder[4].bias[0] = -.5
                model.head[0].weight[0, 0, 0, 0] = 1
                model.head[2].weight[0, 0, 0, 0] = 1
        inputs = torch.ones(1, 2, 3, 3)
        with torch.no_grad():
            self.assertTrue(torch.equal(plain(inputs), torch.zeros(1, 2, 3, 3)))
            actual = residual(inputs)
            self.assertTrue(torch.equal(actual[:, 0], torch.full((1, 3, 3), .5)))
            self.assertTrue(torch.equal(actual[:, 1], torch.zeros(1, 3, 3)))
            residual.encoder[4].bias[0] = -2
            self.assertTrue(torch.equal(residual.encoder(inputs), torch.zeros(1, 32, 3, 3)))

    def test_residual_formula_agrees_with_independent_functional_convolutions(self):
        model = self.build("mean_cnn", True).double()
        inputs = self.raw[:, 0].double()

        def convolve(values, layer):
            padded = torch.nn.functional.pad(values, (1, 1, 1, 1), mode="circular")
            return torch.nn.functional.conv2d(padded, layer.weight, layer.bias)

        first = torch.relu(convolve(inputs, model.encoder[0]))
        second = torch.relu(convolve(first, model.encoder[2]))
        expected = torch.relu(convolve(second, model.encoder[4]) + first)
        torch.testing.assert_close(model.encoder(inputs), expected, atol=0, rtol=0)

    def test_periodic_translation_and_shot_permutation_preserve_contracts(self):
        for residual in (False, True):
            for method in METHODS:
                with self.subTest(method=method, residual=residual):
                    model = self.build(method, residual, shot_chunk_size=3).eval()
                    for length in (3, 4):
                        raw = self.raw[..., :length, :length]
                        inputs = raw.float().mean(1) if method == "mean_cnn" else raw
                        with torch.no_grad():
                            expected = model(inputs)
                            actual = model(torch.roll(inputs, (1, -1), (-2, -1)))
                            torch.testing.assert_close(actual,
                                torch.roll(expected, (1, -1), (-2, -1)),
                                atol=2e-6, rtol=1e-5)
                            if method == "set_cnn":
                                permuted = model(inputs[:, [4, 0, 6, 2, 3, 1, 5]])
                                torch.testing.assert_close(permuted, expected,
                                                           atol=2e-6, rtol=1e-5)

    def test_residual_single_shot_matches_mean_path_and_chunk_gradients_match(self):
        for residual in (False, True):
            with self.subTest(residual=residual):
                mean = self.build("mean_cnn", residual).double()
                full = self.build("set_cnn", residual, shot_chunk_size=7,
                                  activation_checkpoint=False).double()
                chunked = self.build("set_cnn", residual, shot_chunk_size=2,
                                     activation_checkpoint=True).double()
                one = self.raw[:, :1].double()
                torch.testing.assert_close(mean(one[:, 0]), full(one), atol=0, rtol=0)
                first = self.raw.double().requires_grad_()
                second = self.raw.double().requires_grad_()
                expected = full(first)
                actual = chunked(second)
                expected.square().mean().backward()
                actual.square().mean().backward()
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(second.grad, first.grad, atol=1e-12, rtol=1e-9)
                for (name, parameter), (_, other) in zip(full.named_parameters(),
                                                       chunked.named_parameters()):
                    torch.testing.assert_close(other.grad, parameter.grad,
                                               atol=1e-12, rtol=1e-9, msg=name)

    def test_options_inputs_and_metadata_are_bound_without_rng_mutation(self):
        before = torch.get_rng_state().clone()
        for method in METHODS:
            for residual in (False, True):
                spec = structure_model_spec(method, residual=residual)
                self.assertEqual(spec["parameter_count"], 38690)
                self.assertIs(spec["residual_connection"], residual)
                self.assertFalse(spec["local_vertex_gauge_invariance_enforced"])
                spec["hidden_channels"][0] = -1
                self.assertEqual(structure_model_spec(method, residual=residual)
                                 ["hidden_channels"], [32, 64, 32])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for options in ({"residual": 1}, {"residual": None},
                        {"residual": False, "shot_chunk_size": True},
                        {"residual": True, "activation_checkpoint": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_structure_model("set_cnn", **options)
        with self.assertRaises(ValueError):
            make_structure_model("unknown", residual=True)
        model = self.build("set_cnn", True)
        for bad in (self.raw[:, :0], self.raw[:, :, :1], self.raw.bool(),
                    self.raw[:, :, :, :2, :2], self.raw[:, 0]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                model(bad)


if __name__ == "__main__":
    unittest.main()
