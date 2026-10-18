"""CPU-only architecture controls using synthetic observable maps."""

import unittest

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.mlp import PaperMLP
    from quantum_lattice.models.observable_cnn import ObservableMapMLP, PeriodicObservableCNN


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class ObservableModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_default_architecture_and_parameter_budget(self):
        model = PeriodicObservableCNN()
        convolutions = [layer for layer in model.layers if isinstance(layer, torch.nn.Conv2d)]
        self.assertEqual([(layer.in_channels, layer.out_channels, layer.kernel_size)
                          for layer in convolutions],
                         [(3, 32, (3, 3)), (32, 64, (3, 3)),
                          (64, 32, (3, 3)), (32, 2, (1, 1))])
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 37922)
        self.assertEqual(sum(parameter.numel() for parameter in ObservableMapMLP().parameters()),
                         sum(parameter.numel() for parameter in PaperMLP().parameters()))
        self.assertEqual(sum(parameter.numel() for parameter in PaperMLP().parameters()), 39319)
        for layer in convolutions[:-1]:
            self.assertEqual(layer.padding_mode, "circular")
            self.assertEqual(layer.padding, (1, 1))
        self.assertEqual([type(layer).__name__ for layer in model.layers],
                         ["Conv2d", "ReLU", "Conv2d", "ReLU", "Conv2d", "ReLU", "Conv2d"])
        for layer in convolutions:
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))

    def test_output_shape_and_size_independent_weights(self):
        for model in (PeriodicObservableCNN(), ObservableMapMLP()):
            count = sum(parameter.numel() for parameter in model.parameters())
            for size in (3, 4, 8):
                with self.subTest(model=type(model).__name__, size=size):
                    self.assertEqual(model(torch.randn(2, 3, size, size)).shape, (2, 2, size, size))
                    self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), count)

    def test_translation_equivariance_including_periodic_seams(self):
        torch.manual_seed(146)
        maps = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        for model in (PeriodicObservableCNN().double(), ObservableMapMLP().double()):
            reference = model(maps)
            for shift in ((1, 0), (0, 1), (-1, 2), (3, 3)):
                with self.subTest(model=type(model).__name__, shift=shift):
                    shifted = model(torch.roll(maps, shifts=shift, dims=(-2, -1)))
                    expected = torch.roll(reference, shifts=shift, dims=(-2, -1))
                    torch.testing.assert_close(shifted, expected, rtol=1e-12, atol=1e-12)

    def test_circular_neighbor_influence_crosses_both_seams(self):
        model = PeriodicObservableCNN(hidden_channels=(1,), kernel_size=3).double()
        convolution, _, output = model.layers
        with torch.no_grad():
            convolution.weight.zero_()
            convolution.bias.zero_()
            # Cross-correlation at (y,x) reads the next row and next column.
            convolution.weight[0, 0, 2, 1] = 1
            convolution.weight[0, 0, 1, 2] = 1
            output.weight.fill_(1)
            output.bias.zero_()
        inputs = torch.zeros(1, 3, 4, 4, dtype=torch.float64)
        inputs[0, 0, 0, 0] = 2
        expected = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
        expected[:, :, 3, 0] = 2
        expected[:, :, 0, 3] = 2
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)

    def test_one_by_one_diagnostic_has_coordinate_local_influence(self):
        model = PeriodicObservableCNN(hidden_channels=(4, 5), kernel_size=1).double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.1)
        baseline = torch.ones(1, 3, 4, 4, dtype=torch.float64)
        changed = baseline.clone()
        changed[0, 0, 1, 2] += 2
        difference = model(changed) - model(baseline)
        self.assertTrue(torch.all(difference[:, :, 1, 2] > 0))
        difference[:, :, 1, 2] = 0
        self.assertEqual(torch.count_nonzero(difference).item(), 0)
        self.assertEqual(len([layer for layer in model.layers if isinstance(layer, torch.nn.Conv2d)]), 3)

    def test_map_mlp_matches_independent_canonical_edge_rows(self):
        torch.manual_seed(984)
        original = PaperMLP().double()
        mapped = ObservableMapMLP(original)
        self.assertIs(mapped.mlp, original)
        # Distinct stars/pairs expose transposition, orientation and wrap errors.
        inputs = torch.arange(2 * 3 * 4 * 4, dtype=torch.float64).reshape(2, 3, 4, 4) / 100
        rows = []
        for batch in range(2):
            for y in range(4):
                for x in range(4):
                    rows.append(torch.stack((inputs[batch, 0, y, x],
                                             inputs[batch, 0, y, (x + 1) % 4],
                                             inputs[batch, 1, y, x])))
                    rows.append(torch.stack((inputs[batch, 0, y, x],
                                             inputs[batch, 0, (y + 1) % 4, x],
                                             inputs[batch, 2, y, x])))
        reference = original(torch.stack(rows)).reshape(2, 4, 4, 2).permute(0, 3, 1, 2)
        torch.testing.assert_close(mapped(inputs), reference, rtol=0, atol=0)

    def test_backpropagation_is_finite_for_both_models(self):
        torch.manual_seed(411)
        for model in (PeriodicObservableCNN(hidden_channels=(4, 5, 4)), ObservableMapMLP()):
            inputs = torch.randn(2, 3, 4, 4, requires_grad=True)
            targets = torch.rand(2, 2, 4, 4)
            loss = (model(inputs) - targets).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(inputs.grad).all())
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_predictions_remain_raw_linear_values(self):
        for model in (PeriodicObservableCNN(), ObservableMapMLP()):
            final = model.layers[-1] if isinstance(model, PeriodicObservableCNN) else model.mlp.layers[-1]
            with torch.no_grad():
                final.weight.zero_()
                final.bias.fill_(-2)
            actual = model(torch.full((2, 3, 4, 4), 3.0))
            torch.testing.assert_close(actual, torch.full((2, 2, 4, 4), -2.0), rtol=0, atol=0)

    def test_bad_map_shapes_are_rejected(self):
        bad_shapes = ((3, 4, 4), (2, 2, 4, 4), (2, 3, 4, 5), (1, 3, 2, 2), (0, 3, 4, 4))
        for model in (PeriodicObservableCNN(), ObservableMapMLP()):
            for shape in bad_shapes:
                with self.subTest(model=type(model).__name__, shape=shape):
                    with self.assertRaises(ValueError):
                        model(torch.ones(shape))

    def test_bad_architecture_configurations_are_rejected(self):
        for widths in ((), (0,), (-2, 4), (True,), (2.5,)):
            with self.subTest(widths=widths):
                with self.assertRaises(ValueError):
                    PeriodicObservableCNN(hidden_channels=widths)
        for kernel_size in (0, 2, 5, True, 3.0):
            with self.subTest(kernel_size=kernel_size):
                with self.assertRaises(ValueError):
                    PeriodicObservableCNN(kernel_size=kernel_size)
        with self.assertRaises(TypeError):
            ObservableMapMLP(torch.nn.Linear(3, 1))

    def test_integer_complex_and_non_tensor_maps_are_rejected(self):
        for model in (PeriodicObservableCNN(), ObservableMapMLP()):
            for inputs in (torch.ones(1, 3, 4, 4, dtype=torch.int64),
                           torch.ones(1, 3, 4, 4, dtype=torch.complex64), [1, 2, 3]):
                with self.subTest(model=type(model).__name__, input_type=type(inputs).__name__):
                    with self.assertRaisesRegex(TypeError, "real floating-point"):
                        model(inputs)


if __name__ == "__main__":
    unittest.main()
