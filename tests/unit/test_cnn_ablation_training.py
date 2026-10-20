"""Short CPU controls for variant-bound training; no research data is opened."""

import copy
import io
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.cnn_ablation_training import CnnAblationTrainSession
    from quantum_lattice.models.cnn_variants import VARIANT_NAMES
    from quantum_lattice.models.observable_training import ObservableTrainSession


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class CnnAblationTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(620)
        self.x = rng.uniform(0.1, 1.3, (5, 3, 3, 3)).astype(np.float32)
        self.y = rng.uniform(0.05, 0.6, (5, 2, 3, 3)).astype(np.float32)
        self.vx = rng.uniform(0.1, 1.3, (3, 3, 3, 3)).astype(np.float32)
        self.vy = rng.uniform(0.05, 0.6, (3, 2, 3, 3)).astype(np.float32)
        self.config = {"model": "cnn", "architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 3},
                       "seed": 173, "device": "cpu", "deterministic": True,
                       "torch_threads": 1, "learning_rate": 1e-3,
                       "batch_size_hamiltonians": 2, "max_epochs": 4,
                       "validation_batch_size": 2,
                       "early_stopping": {"enabled": False, "min_epochs": 1,
                                          "patience_epochs": 2, "min_delta": 1e-6}}

    def session(self, variant="baseline", checkpoint=None, config=None, **arrays):
        return CnnAblationTrainSession(
            arrays.get("train_x", self.x), arrays.get("train_y", self.y),
            arrays.get("val_x", self.vx), arrays.get("val_y", self.vy),
            self.config if config is None else config, variant=variant, checkpoint=checkpoint)

    def assert_tree_equal(self, left, right):
        self.assertIs(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_baseline_initialization_updates_and_selection_match_frozen_engine_exactly(self):
        old = ObservableTrainSession(self.x, self.y, self.vx, self.vy, self.config)
        new = self.session()
        self.assert_tree_equal(old.snapshot(), new.snapshot()["engine"])
        for chunk in (1, 3, 2, 6):
            old.advance(chunk)
            new.advance(chunk)
            self.assert_tree_equal(old.snapshot(), new.snapshot()["engine"])
        np.testing.assert_array_equal(old.selected_predictions(self.vx), new.selected_predictions(self.vx))

    def test_every_variant_resumes_mid_epoch_bit_exactly_with_weights_only_loading(self):
        for variant in VARIANT_NAMES:
            with self.subTest(variant=variant):
                direct = self.session(variant)
                direct.advance(12)
                expected = direct.snapshot()
                partial = self.session(variant)
                partial.advance(4)
                checkpoint = partial.snapshot()
                self.assertEqual((checkpoint["engine"]["epoch"], checkpoint["engine"]["cursor"]), (1, 2))
                stream = io.BytesIO()
                torch.save(checkpoint, stream)
                stream.seek(0)
                loaded = torch.load(stream, map_location="cpu", weights_only=True)
                resumed = self.session(variant, loaded)
                resumed.advance(8)
                self.assert_tree_equal(expected, resumed.snapshot())
                np.testing.assert_array_equal(direct.selected_predictions(self.vx),
                                              resumed.selected_predictions(self.vx))

    def test_batchnorm_initial_state_epoch_boundary_and_live_selected_counters(self):
        initial = self.session("batchnorm")
        self.assert_tree_equal(initial.snapshot(), self.session("batchnorm", initial.snapshot()).snapshot())
        initial.advance(3)
        boundary = initial.snapshot()
        self.assert_tree_equal(boundary, self.session("batchnorm", boundary).snapshot())
        for state_key, step_key in (("model_state_dict", "step"), ("best_state_dict", "best_step")):
            state = boundary["engine"][state_key]
            counters = [value for name, value in state.items() if name.endswith("num_batches_tracked")]
            self.assertEqual(len(counters), 3)
            self.assertTrue(all(value.dtype == torch.int64 and value.item() == boundary["engine"][step_key]
                                for value in counters))

    def test_full_validation_and_selected_inference_do_not_change_batchnorm_buffers_or_rng(self):
        session = self.session("batchnorm")
        session.advance(4)
        before = session.snapshot()
        session._mse(session.train_x, session.train_y)
        session._mse(session.validation_x, session.validation_y)
        self.assertTrue(session.model.training)
        session.selected_predictions(self.vx, batch_size=1)
        session.selected_predictions(self.vx, batch_size=3)
        self.assert_tree_equal(before, session.snapshot())
        session.model.eval()
        session._mse(session.validation_x, session.validation_y)
        session.selected_predictions(self.vx)
        self.assertFalse(session.model.training)
        self.assert_tree_equal(before, session.snapshot())

    def test_batchnorm_eval_predictions_do_not_depend_on_other_hamiltonians(self):
        session = self.session("batchnorm")
        session.advance(9)
        original = session.selected_predictions(self.vx, batch_size=3)
        changed = self.vx.copy()
        changed[1:] *= 100
        with_other_inputs = session.selected_predictions(changed, batch_size=3)
        np.testing.assert_array_equal(original[0], with_other_inputs[0])
        # Different computational batch shapes may use different float32 kernels.
        np.testing.assert_allclose(original, session.selected_predictions(self.vx, batch_size=1),
                                   rtol=1e-5, atol=1e-6)

    def test_envelope_and_variant_specification_cannot_be_replaced(self):
        saved = self.session("batchnorm").snapshot()
        for target in ("baseline", "groupnorm", "residual"):
            with self.subTest(variant=target), self.assertRaisesRegex(ValueError, "variant/model"):
                self.session(target, saved)
        corruptions = [lambda state: state.update(training_kind="observable_maps"),
                       lambda state: state.update(training_schema_version=True),
                       lambda state: state.update(training_schema_version=2),
                       lambda state: state.update(extra="unbound"),
                       lambda state: state.update(engine=[]),
                       lambda state: state["model_spec"].update(unbound_setting=True)]
        for mutate in corruptions:
            state = copy.deepcopy(saved)
            mutate(state)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.session("batchnorm", state)
        with self.assertRaises(ValueError):
            self.session("batchnorm", saved["engine"])

    def test_seed_policy_and_data_remain_bound_by_inner_engine(self):
        saved = self.session("batchnorm").snapshot()
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.session("batchnorm", saved, config={**self.config, "seed": 174})
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.session("batchnorm", saved, config={**self.config, "max_epochs": 5})
        with self.assertRaisesRegex(ValueError, "data differ"):
            self.session("batchnorm", saved, train_x=self.x * 0.9)
        with self.assertRaisesRegex(ValueError, "data differ"):
            self.session("batchnorm", saved, val_y=self.vy * 0.9)

    def test_batchnorm_corrupt_variances_and_update_counts_are_rejected(self):
        session = self.session("batchnorm")
        session.advance(4)
        saved = session.snapshot()
        keys = saved["engine"]["model_state_dict"]
        counter = next(name for name in keys if name.endswith("num_batches_tracked"))
        variance = next(name for name in keys if name.endswith("running_var"))
        mutations = [("model_state_dict", counter, lambda value: value.fill_(-1)),
                     ("model_state_dict", counter, lambda value: value.fill_(5)),
                     ("best_state_dict", counter, lambda value: value.fill_(99)),
                     ("model_state_dict", variance, lambda value: value.fill_(-0.1)),
                     ("best_state_dict", variance, lambda value: value.fill_(float("nan")))]
        for name, key, mutate in mutations:
            state = copy.deepcopy(saved)
            mutate(state["engine"][name][key])
            with self.subTest(state=name, key=key), self.assertRaisesRegex(ValueError, "BatchNorm"):
                self.session("batchnorm", state)

    def test_snapshot_and_model_spec_are_owned_copies(self):
        session = self.session("batchnorm")
        session.advance(1)
        original = session.snapshot()
        metadata = session.model_spec
        metadata["unbound_setting"] = 1
        state = session.snapshot()
        state["model_spec"]["unbound_setting"] = 2
        for name in ("model_state_dict", "best_state_dict"):
            for value in state["engine"][name].values():
                value.zero_()
        state["engine"]["history"][0]["validation_mse"] = 900
        self.assert_tree_equal(original, session.snapshot())

    def test_shuffle_order_pairs_across_variants_and_estimator_inputs(self):
        first = self.session("baseline")
        second = self.session("batchnorm", train_x=self.x * 0.8)
        third = self.session("residual")
        for _ in range(7):
            first.advance(1)
            second.advance(1)
            third.advance(1)
            self.assertTrue(torch.equal(first.permutation, second.permutation))
            self.assertTrue(torch.equal(first.permutation, third.permutation))
            self.assertEqual(first.cursor, second.cursor)
            self.assertEqual(first.cursor, third.cursor)

    def test_only_fixed_scaffold_and_registered_variant_are_accepted(self):
        for changes in ({"model": "map_mlp"}, {"architecture": {"hidden_channels": [32], "kernel_size": 3}},
                        {"architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 1}}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "common cnn"):
                self.session(config={**self.config, **changes})
        with self.assertRaises(ValueError):
            self.session("unregistered")

    def test_cpu_path_never_initializes_cuda_or_requests_cuda_rng(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")), \
             mock.patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA RNG touched")), \
             mock.patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("CUDA RNG touched")):
            session = self.session("batchnorm")
            session.advance(1)
            session.selected_predictions(self.vx)
            saved = session.snapshot()
            self.assertEqual(saved["engine"]["cuda_rng_states"], [])
            self.session("batchnorm", saved)


if __name__ == "__main__":
    unittest.main()
