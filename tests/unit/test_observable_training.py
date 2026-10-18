"""Whole-Hamiltonian training checks on tiny CPU arrays, not research runs."""

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
    from quantum_lattice.models.observable_training import ObservableTrainSession


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class ObservableTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(504)
        self.x = rng.uniform(0.1, 1.3, (5, 3, 3, 3)).astype(np.float32)
        self.y = rng.uniform(0.05, 0.6, (5, 2, 3, 3)).astype(np.float32)
        self.vx = rng.uniform(0.1, 1.3, (3, 3, 3, 3)).astype(np.float32)
        self.vy = rng.uniform(0.05, 0.6, (3, 2, 3, 3)).astype(np.float32)
        self.config = {"model": "cnn", "architecture": {"hidden_channels": [4, 5], "kernel_size": 3},
                       "seed": 73, "device": "cpu", "deterministic": True,
                       "torch_threads": 1, "learning_rate": 1e-3,
                       "batch_size_hamiltonians": 2, "max_epochs": 4,
                       "validation_batch_size": 2,
                       "early_stopping": {"enabled": False, "min_epochs": 1,
                                          "patience_epochs": 2, "min_delta": 1e-6}}

    def session(self, config=None, checkpoint=None, **arrays):
        return ObservableTrainSession(arrays.get("train_x", self.x), arrays.get("train_y", self.y),
                                      arrays.get("val_x", self.vx), arrays.get("val_y", self.vy),
                                      self.config if config is None else config, checkpoint)

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

    def test_initial_selection_and_independent_full_loss_oracle(self):
        session = self.session()
        self.assertEqual((session.step, session.epoch, session.best_step, session.best_epoch), (0, 0, 0, 0))
        self.assertEqual(len(session.history), 1)
        self.assertTrue(session.history[0]["improved"])
        with torch.no_grad():
            prediction = np.concatenate([session.model(torch.tensor(self.vx[start:start + 2])).numpy()
                                         for start in range(0, len(self.vx), 2)])
        # The engine deliberately subtracts in float32 before float64 reduction.
        expected = np.mean((prediction - self.vy).astype(np.float64) ** 2)
        self.assertAlmostEqual(session.best_val_mse, expected, places=14)
        self.assertEqual(session.selected_predictions(self.vx).dtype, np.float32)

    def test_complete_epochs_include_short_batch_and_only_epoch_checks(self):
        session = self.session()
        self.assertEqual(session.advance(0), [])
        self.assertEqual(session.advance(2), [])
        self.assertEqual((session.step, session.epoch, session.cursor, session.hamiltonians_seen), (2, 0, 4, 4))
        event = session.advance(1)[0]
        self.assertEqual((event["step"], event["epoch"], event["hamiltonians_seen"]), (3, 1, 5))
        self.assertIsNotNone(event["optimizer_epoch_mse"])
        events = session.advance(100)
        self.assertEqual([event["epoch"] for event in events], [2, 3, 4])
        self.assertEqual((session.step, session.epoch, session.hamiltonians_seen), (12, 4, 20))
        self.assertEqual(len(session.history), 5)
        self.assertTrue(session.finished)
        self.assertEqual(session.stop_reason, "max_epochs")
        self.assertEqual(session.advance(1), [])

    def test_adam_updates_match_independent_whole_hamiltonian_oracle(self):
        session = self.session()
        reference = copy.deepcopy(session.model)
        optimizer = torch.optim.Adam(reference.parameters(), lr=self.config["learning_rate"],
                                     betas=(0.9, 0.999), eps=1e-8)
        generator = torch.Generator().manual_seed(self.config["seed"])
        order = torch.randperm(len(self.x), generator=generator).numpy()
        for start in range(0, len(self.x), 2):
            ids = order[start:start + 2]
            optimizer.zero_grad(set_to_none=True)
            difference = reference(torch.tensor(self.x[ids])) - torch.tensor(self.y[ids])
            # Independently average H, orientation, y and x with the short final
            # Hamiltonian batch retaining its own ordinary minibatch mean.
            loss = difference.square().sum() / (len(ids) * 2 * 3 * 3)
            loss.backward()
            optimizer.step()
            session.advance(1)
            self.assert_tree_equal(reference.state_dict(), session.model.state_dict())
        self.assertEqual(session.epoch, 1)

    def test_partial_epoch_resume_is_bit_identical_for_both_models(self):
        for kind in ("cnn", "map_mlp"):
            with self.subTest(model=kind):
                config = {**self.config, "model": kind}
                direct = self.session(config)
                direct.advance(12)
                expected = direct.snapshot()
                partial = self.session(config)
                partial.advance(4)  # A finished epoch plus one partial batch.
                saved = partial.snapshot()
                self.assertEqual((saved["epoch"], saved["cursor"], saved["epoch_elements"]), (1, 2, 36))
                stream = io.BytesIO()
                torch.save(saved, stream)
                stream.seek(0)
                loaded = torch.load(stream, map_location="cpu", weights_only=True)
                resumed = self.session(config, loaded)
                resumed.advance(8)
                self.assert_tree_equal(expected, resumed.snapshot())

    def test_resume_at_epoch_boundary_and_initial_state(self):
        direct = self.session()
        initial = direct.snapshot()
        resumed = self.session(checkpoint=initial)
        direct.advance(3)
        resumed.advance(3)
        self.assert_tree_equal(direct.snapshot(), resumed.snapshot())
        boundary = resumed.snapshot()
        next_session = self.session(checkpoint=boundary)
        direct.advance(9)
        next_session.advance(9)
        self.assert_tree_equal(direct.snapshot(), next_session.snapshot())

    def test_hamiltonian_shuffle_order_is_architecture_and_view_independent(self):
        first = self.session()
        second = self.session({**self.config, "model": "map_mlp"}, train_x=self.x * 0.9)
        generator = torch.Generator().manual_seed(self.config["seed"])
        for epoch in range(3):
            expected = torch.randperm(len(self.x), generator=generator)
            for update in range(3):
                first.advance(1)
                second.advance(1)
                self.assertTrue(torch.equal(first.permutation, expected))
                self.assertTrue(torch.equal(second.permutation, expected))
                self.assertEqual(first.cursor, min(2 * (update + 1), len(self.x)))
                self.assertEqual(first.cursor, second.cursor)

    def test_selected_prediction_does_not_mutate_live_adam_mode_or_rng(self):
        session = self.session()
        session.advance(4)
        before = session.snapshot()
        training = session.model.training
        predictions = session.selected_predictions(self.vx, batch_size=1)
        self.assertEqual(predictions.shape, self.vy.shape)
        self.assertEqual(session.model.training, training)
        self.assert_tree_equal(before, session.snapshot())
        with torch.no_grad():
            clone = copy.deepcopy(session.model)
            clone.load_state_dict(session.best_state_dict)
            expected = clone(torch.tensor(self.vx)).numpy()
        np.testing.assert_allclose(predictions, expected, rtol=1e-6, atol=1e-7)
        session.model.eval()
        session.selected_predictions(self.vx)
        self.assertFalse(session.model.training)

    def test_best_selection_is_strict_and_independent_of_patience_delta(self):
        config = copy.deepcopy(self.config)
        config["early_stopping"] = {"enabled": True, "min_epochs": 1,
                                    "patience_epochs": 2, "min_delta": 0.05}
        # Epoch 1 sets reference=1; two small strict improvements do not reset
        # the min_delta reference, but both remain eligible best checkpoints.
        losses = iter([2.0, 1.1, 2.0, 1.0, 2.0, 0.99, 2.0, 0.98])
        with mock.patch.object(ObservableTrainSession, "_mse", side_effect=lambda *args: next(losses)):
            session = self.session(config)
            events = session.advance(100)
        self.assertEqual([event["early_stopping_wait"] for event in events], [0, 1, 2])
        self.assertEqual((session.epoch, session.best_epoch), (3, 3))
        self.assertEqual(session.best_val_mse, 0.98)
        self.assertEqual(session.stop_reason, "early_stopping")

    def test_patience_counts_only_after_minimum_and_significant_improvement_resets(self):
        config = copy.deepcopy(self.config)
        config["max_epochs"] = 10
        config["early_stopping"] = {"enabled": True, "min_epochs": 2,
                                    "patience_epochs": 2, "min_delta": 0.1}
        session = self.session(config)
        # Initial validation was real; override its loss/reference consistently.
        session.best_val_mse = 1.0
        vals = [1.0, 1.0, 0.95, 0.8, 0.79, 0.78]
        calls = iter(value for val in vals for value in (2.0, val))
        with mock.patch.object(session, "_mse", side_effect=lambda *args: next(calls)):
            events = session.advance(100)
        self.assertEqual([event["early_stopping_wait"] for event in events], [0, 0, 1, 0, 1, 2])
        self.assertEqual(session.epoch, 6)

    def test_initial_model_can_remain_selected_and_predictions_are_unclipped(self):
        session = self.session()
        initial = session.best_state_dict
        with mock.patch.object(session, "_mse", return_value=session.best_val_mse + 1):
            session.advance(3)
        self.assertEqual(session.best_epoch, 0)
        self.assert_tree_equal(initial, session.best_state_dict)
        # A selected raw linear output may be negative; no abs/sigmoid/clipping.
        for value in session._best_state_dict.values():
            value.zero_()
        last_bias = [name for name in session._best_state_dict if name.endswith("bias")][-1]
        session._best_state_dict[last_bias].fill_(-0.5)
        np.testing.assert_array_equal(session.selected_predictions(self.vx), np.full_like(self.vy, -0.5))

    def test_arrays_config_history_and_snapshots_are_owned_copies(self):
        session = self.session()
        source_prediction = session.selected_predictions(self.vx)
        self.x.fill(900)
        self.config["architecture"]["hidden_channels"][0] = 99
        self.assertNotEqual(float(session.train_x[0, 0, 0, 0]), 900)
        self.assertEqual(session.config["architecture"]["hidden_channels"], [4, 5])
        event = session.history
        event[0]["validation_mse"] = 999
        state = session.snapshot()
        next(iter(state["model_state_dict"].values())).fill_(900)
        state["history"][0]["validation_mse"] = 999
        np.testing.assert_array_equal(session.selected_predictions(self.vx), source_prediction)
        self.assertNotEqual(session.history[0]["validation_mse"], 999)

    def test_invalid_config_and_arrays_rejected_before_training(self):
        bad_configs = [{"seed": True}, {"seed": 2**63}, {"device": "auto"},
                       {"model": "other"}, {"learning_rate": float("nan")},
                       {"batch_size_hamiltonians": 0}, {"max_epochs": True},
                       {"validation_batch_size": -1}, {"deterministic": 1},
                       {"extra": 1}, {"architecture": {"hidden_channels": [4], "kernel_size": True}},
                       {"early_stopping": {"enabled": "yes"}},
                       {"early_stopping": {"min_delta": -1}},
                       {"early_stopping": {"patience_epochs": True}}]
        for changes in bad_configs:
            with self.subTest(config=changes), self.assertRaises(ValueError):
                self.session({**self.config, **changes})
        bad_arrays = [{"train_x": self.x[:0]}, {"train_x": self.x[:, :, :, :2]},
                      {"train_y": self.y[:4]}, {"train_y": -self.y},
                      {"val_x": np.full_like(self.vx, np.inf)},
                      {"val_y": np.zeros((3, 2, 4, 4))},
                      {"train_x": self.x.astype(complex)},
                      {"train_x": np.full(self.x.shape, 1e100)}]
        for arrays in bad_arrays:
            with self.subTest(arrays=list(arrays)), self.assertRaises(ValueError):
                self.session(**arrays)

    def test_checkpoint_data_settings_and_weight_corruption_are_rejected(self):
        session = self.session()
        session.advance(1)
        snapshot = session.snapshot()
        with self.assertRaisesRegex(ValueError, "data differ"):
            self.session(checkpoint=snapshot, train_x=self.x * 0.9)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.session({**self.config, "seed": 74}, snapshot)
        corruptions = [lambda state: state.update(cursor=3),
                       lambda state: state.update(epoch_elements=0),
                       lambda state: state.update(permutation=torch.zeros(5, dtype=torch.int64)),
                       lambda state: state.update(best_val_mse=99),
                       lambda state: state["model_state_dict"].update({
                           next(iter(state["model_state_dict"])): torch.zeros(1)}),
                       lambda state: state["best_state_dict"].update({
                           next(iter(state["best_state_dict"])):
                           next(iter(state["best_state_dict"].values())).double()}),
                       lambda state: state["optimizer_state_dict"]["param_groups"][0].update(lr=0.9),
                       lambda state: state["optimizer_state_dict"]["state"][0].update(
                           exp_avg=state["optimizer_state_dict"]["state"][0]["exp_avg"].double()),
                       lambda state: state["optimizer_state_dict"]["state"][0].update(step=torch.tensor(5.0)),
                       lambda state: state["optimizer_state_dict"]["state"][0]["exp_avg_sq"].fill_(-0.1)]
        for mutate in corruptions:
            state = copy.deepcopy(snapshot)
            mutate(state)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.session(checkpoint=state)

    def test_resume_replays_stopping_and_rejects_mutated_wait_reference_or_history(self):
        config = copy.deepcopy(self.config)
        config["early_stopping"] = {"enabled": True, "min_epochs": 1,
                                    "patience_epochs": 2, "min_delta": 0.05}
        losses = iter([2.0, 1.1, 2.0, 1.0, 2.0, 0.99, 2.0, 0.98])
        with mock.patch.object(ObservableTrainSession, "_mse", side_effect=lambda *args: next(losses)):
            session = self.session(config)
            session.advance(6)
            partial = session.snapshot()
            resumed = self.session(config, partial)
            resumed.advance(3)
        self.assertEqual(resumed.stop_reason, "early_stopping")
        stopped = resumed.snapshot()
        self.assertTrue(self.session(config, stopped).finished)
        corruptions = [lambda state: state.update(early_stopping_wait=0),
                       lambda state: state.update(early_stopping_reference=0.5),
                       lambda state: state.update(stopped_early=False),
                       lambda state: state["history"][-1].update(early_stopping_wait=0),
                       lambda state: state["history"][-1].update(early_stopping_reference=0.5),
                       lambda state: state["history"][-1].update(improved=False)]
        for mutate in corruptions:
            state = copy.deepcopy(stopped)
            mutate(state)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.session(config, state)

    def test_cpu_path_never_initializes_or_requests_cuda_rng(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")), \
             mock.patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA RNG touched")), \
             mock.patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("CUDA RNG touched")):
            session = self.session()
            session.advance(1)
            session.selected_predictions(self.vx)
            state = session.snapshot()
            self.assertEqual(state["cuda_rng_states"], [])
            self.session(checkpoint=state)

    def test_nonfinite_loss_and_gradient_never_advance_adam(self):
        session = self.session()
        with mock.patch.object(session.model, "forward", return_value=torch.full((2, 2, 3, 3), float("nan"))):
            with self.assertRaisesRegex(FloatingPointError, "loss"):
                session.advance(1)
        self.assertEqual(session.step, 0)
        parameter = next(session.model.parameters())
        handle = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        try:
            with self.assertRaisesRegex(FloatingPointError, "gradients"):
                session.advance(1)
        finally:
            handle.remove()
        self.assertEqual(session.step, 0)
        self.assertEqual(len(session.optimizer.state), 0)
        # Failed proposals consume no optimizer update; their already-generated
        # Hamiltonian permutation is still a valid exact-resume state.
        resumed = self.session(checkpoint=session.snapshot())
        session.advance(1)
        resumed.advance(1)
        self.assert_tree_equal(session.snapshot(), resumed.snapshot())

    def test_bad_advance_or_prediction_arguments(self):
        session = self.session()
        for count in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                session.advance(count)
        with self.assertRaises(ValueError):
            session.selected_predictions(self.vx, batch_size=True)
        with self.assertRaisesRegex(ValueError, "lattice shape"):
            session.selected_predictions(np.zeros((1, 3, 4, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
