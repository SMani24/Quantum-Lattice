"""Independent CPU controls for the prospective fixed CNN architecture arms."""

import json
import unittest

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.cnn_variants import VARIANT_NAMES, make_model, model_spec
    from quantum_lattice.models.observable_cnn import PeriodicObservableCNN


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class CNNVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_baseline_has_exact_original_class_initialization_and_outputs(self):
        torch.manual_seed(193)
        original = PeriodicObservableCNN()
        torch.manual_seed(193)
        baseline = make_model("baseline")
        self.assertIs(type(baseline), PeriodicObservableCNN)
        self.assertEqual(original.state_dict().keys(), baseline.state_dict().keys())
        for key, expected in original.state_dict().items():
            self.assertTrue(torch.equal(baseline.state_dict()[key], expected), key)
        x = torch.randn(2, 3, 4, 4)
        torch.testing.assert_close(original(x), baseline(x), rtol=0, atol=0)

    def test_independent_parameter_counts_and_shapes(self):
        expected = {"baseline": 37922, "batchnorm": 38178,
                    "groupnorm": 38178, "residual": 37922}
        self.assertEqual(tuple(expected), VARIANT_NAMES)
        for name, count in expected.items():
            with self.subTest(variant=name):
                model = make_model(name)
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                self.assertEqual(model_spec(name)["parameter_count"], count)
                for size in (3, 4, 8):
                    self.assertEqual(model(torch.randn(2, 3, size, size)).shape,
                                     (2, 2, size, size))
                convolutions = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
                self.assertEqual([m.kernel_size for m in convolutions], [(3, 3)] * 3 + [(1, 1)])
                self.assertTrue(all(m.padding_mode == "circular" for m in convolutions[:3]))
                self.assertTrue(all(m.bias is not None for m in convolutions))
                self.assertEqual(model_spec(name)["convolutional_path_receptive_field"], 7)

    def test_all_arms_pair_identical_initial_convolution_weights(self):
        torch.manual_seed(911)
        baseline = make_model("baseline")
        baseline_convs = [m for m in baseline.modules() if isinstance(m, torch.nn.Conv2d)]
        for name in VARIANT_NAMES:
            torch.manual_seed(911)
            model = make_model(name)
            convolutions = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            for actual, expected in zip(convolutions, baseline_convs):
                with self.subTest(variant=name, shape=tuple(actual.weight.shape)):
                    self.assertTrue(torch.equal(actual.weight, expected.weight))
                    self.assertTrue(torch.equal(actual.bias, expected.bias))

    def test_normalization_context_is_explicit(self):
        batchnorm = model_spec("batchnorm")["effective_context"]
        groupnorm = model_spec("groupnorm")["effective_context"]
        self.assertIn("whole minibatch and torus", batchnorm["training"])
        self.assertIn("fixed running statistics", batchnorm["evaluation"])
        self.assertEqual(groupnorm["training"], "whole torus within each channel group")
        self.assertEqual(groupnorm["evaluation"], groupnorm["training"])

    def test_specs_are_json_safe_independent_and_rng_neutral(self):
        before = torch.get_rng_state().clone()
        for name in VARIANT_NAMES:
            spec = model_spec(name)
            self.assertEqual(json.loads(json.dumps(spec)), spec)
            spec["hidden_channels"].append(999)
            self.assertNotIn(999, model_spec(name)["hidden_channels"])
            if spec["normalization"]:
                spec["normalization"]["eps"] = -1
                self.assertEqual(model_spec(name)["normalization"]["eps"], 1e-5)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_translation_equivariance_on_torus_train_and_eval(self):
        torch.manual_seed(331)
        x = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        for name in VARIANT_NAMES:
            for training in (False, True):
                model = make_model(name).double().train(training)
                reference = model(x)
                for shift in ((1, 0), (0, 1), (-1, 2), (3, 3)):
                    with self.subTest(variant=name, training=training, shift=shift):
                        actual = model(torch.roll(x, shifts=shift, dims=(-2, -1)))
                        expected = torch.roll(reference, shifts=shift, dims=(-2, -1))
                        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def test_normalization_defaults_and_initialization(self):
        for name in VARIANT_NAMES:
            model = make_model(name)
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                    self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                    self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
                    self.assertEqual(module.eps, 1e-5)
                    if isinstance(module, torch.nn.BatchNorm2d):
                        self.assertEqual(module.momentum, 0.1)
                        self.assertTrue(module.track_running_stats)
                        self.assertTrue(torch.equal(module.running_mean, torch.zeros_like(module.running_mean)))
                        self.assertTrue(torch.equal(module.running_var, torch.ones_like(module.running_var)))
                    else:
                        self.assertEqual(module.num_groups, 8)

    def test_batchnorm_buffers_update_only_in_training_and_roundtrip(self):
        torch.manual_seed(124)
        model = make_model("batchnorm")
        norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        old = {k: v.clone() for k, v in model.named_buffers()}
        model.train()
        model(torch.randn(4, 3, 4, 4) + 3)
        self.assertTrue(all(m.num_batches_tracked.item() == 1 for m in norms))
        self.assertTrue(any(not torch.equal(old[k], v) for k, v in model.named_buffers()
                            if "running_mean" in k))
        after_train = {k: v.clone() for k, v in model.named_buffers()}
        model.eval()
        x = torch.randn(3, 3, 4, 4)
        expected = model(x)
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(value, after_train[key]), key)
        restored = make_model("batchnorm")
        restored.load_state_dict(model.state_dict())
        restored.eval()
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_first_batchnorm_running_statistics_match_independent_formula(self):
        model = make_model("batchnorm").double()
        x = torch.randn(4, 3, 4, 4, dtype=torch.float64) + 2
        first_conv, first_norm = model.layers[:2]
        convolution_output = first_conv(x)
        independent_mean = convolution_output.mean(dim=(0, 2, 3))
        # BatchNorm uses unbiased sample variance to update running_var.
        independent_var = convolution_output.var(dim=(0, 2, 3), unbiased=True)
        model(x)
        torch.testing.assert_close(first_norm.running_mean, 0.1 * independent_mean,
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(first_norm.running_var, 0.9 + 0.1 * independent_var,
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(first_norm.num_batches_tracked.item(), 1)

    def test_eval_prediction_is_independent_of_batch_peers(self):
        torch.manual_seed(55)
        x = torch.randn(1, 3, 4, 4, dtype=torch.float64)
        peers = torch.randn(3, 3, 4, 4, dtype=torch.float64) * 10 + 4
        for name in VARIANT_NAMES:
            model = make_model(name).double()
            # In BatchNorm this creates non-default running statistics first.
            model(torch.randn(4, 3, 4, 4, dtype=torch.float64))
            model.eval()
            with self.subTest(variant=name):
                torch.testing.assert_close(model(x), model(torch.cat((x, peers)))[0:1],
                                           rtol=1e-12, atol=1e-12)

    def test_residual_path_matches_independent_formula(self):
        model = make_model("residual").double()
        x = torch.randn(2, 3, 4, 4, dtype=torch.float64)
        hidden = torch.relu(model.stem(x))
        residual = model.block_conv2(torch.relu(model.block_conv1(hidden)))
        expected = model.head(torch.relu(hidden + residual))
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        with torch.no_grad():
            model.block_conv2.weight.zero_()
            model.block_conv2.bias.zero_()
        torch.testing.assert_close(model(x), model.head(hidden), rtol=0, atol=0)

    def test_finite_gradients_and_raw_linear_outputs(self):
        for name in VARIANT_NAMES:
            with self.subTest(variant=name):
                model = make_model(name)
                x = torch.randn(2, 3, 4, 4, requires_grad=True)
                loss = (model(x) - torch.rand(2, 2, 4, 4)).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                head = model.head if name == "residual" else model.layers[-1]
                with torch.no_grad():
                    head.weight.zero_()
                    head.bias.fill_(-2)
                torch.testing.assert_close(model(x), torch.full((2, 2, 4, 4), -2.0),
                                           rtol=0, atol=0)

    def test_invalid_names_and_inputs_fail(self):
        for name in ("", "CNN", "batch_norm", None, [], True):
            for factory in (make_model, model_spec):
                with self.subTest(name=name, factory=factory.__name__):
                    with self.assertRaises(ValueError):
                        factory(name)
        for name in VARIANT_NAMES:
            model = make_model(name)
            for shape in ((3, 4, 4), (2, 2, 4, 4), (2, 3, 4, 5), (1, 3, 2, 2), (0, 3, 4, 4)):
                with self.assertRaises(ValueError):
                    model(torch.ones(shape))
            for x in (torch.ones(1, 3, 4, 4, dtype=torch.int64),
                      torch.ones(1, 3, 4, 4, dtype=torch.complex64), [1, 2, 3]):
                with self.assertRaises(TypeError):
                    model(x)


if __name__ == "__main__":
    unittest.main()
