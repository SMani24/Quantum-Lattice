"""Architecture checks independent of the training implementation."""

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.mlp import PaperMLP


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class PaperMLPTests(unittest.TestCase):
    def test_dense_shapes_relu_positions_and_initialization(self):
        model = PaperMLP()
        dense = [layer for layer in model.layers if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in dense],
                         [(3, 128), (128, 150), (150, 128), (128, 1)])
        self.assertEqual([type(layer).__name__ for layer in model.layers],
                         ["Linear", "ReLU", "Linear", "ReLU", "Linear", "ReLU", "Linear"])
        for layer in dense:
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))
            limit = np.sqrt(6 / (layer.in_features + layer.out_features))
            self.assertLessEqual(float(layer.weight.detach().abs().max()), limit)

    def test_forward_matches_independent_numpy_calculation(self):
        model = PaperMLP()
        rng = np.random.default_rng(15)
        inputs = rng.normal(size=(7, 3)).astype(np.float32)
        reference = inputs.copy()
        dense = [layer for layer in model.layers if isinstance(layer, torch.nn.Linear)]
        for index, layer in enumerate(dense):
            weights = rng.normal(scale=0.04, size=(layer.out_features, layer.in_features)).astype(np.float32)
            bias = rng.normal(scale=0.03, size=layer.out_features).astype(np.float32)
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weights))
                layer.bias.copy_(torch.from_numpy(bias))
            reference = reference @ weights.T + bias
            if index != len(dense) - 1:
                reference = np.maximum(reference, 0)
        actual = model(torch.from_numpy(inputs)).detach().numpy()
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-7)

    def test_output_is_linear_and_not_clipped(self):
        model = PaperMLP()
        with torch.no_grad():
            model.layers[-1].weight.zero_()
            model.layers[-1].bias.fill_(-2)
        actual = model(torch.ones((2, 3)))
        self.assertTrue(torch.equal(actual, torch.full((2, 1), -2.0)))

    def test_input_shape_guard(self):
        with self.assertRaises(ValueError):
            PaperMLP()(torch.ones(3))


if __name__ == "__main__":
    unittest.main()
