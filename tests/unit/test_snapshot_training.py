"""Tiny CPU training/resume controls; no research files or final-test bank."""

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
    from quantum_lattice.models.snapshot_cnn import METHODS
    from quantum_lattice.models.snapshot_training import SnapshotTrainSession


@unittest.skipIf(torch is None, "Use the shared .venv with PyTorch")
class SnapshotTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(409)
        self.raw = (rng.integers(0, 2, (5, 5, 2, 3, 3), dtype=np.int8) * 2 - 1)
        self.vraw = (rng.integers(0, 2, (3, 5, 2, 3, 3), dtype=np.int8) * 2 - 1)
        self.y = rng.uniform(.05, .8, (5, 2, 3, 3)).astype(np.float32)
        self.vy = rng.uniform(.05, .8, (3, 2, 3, 3)).astype(np.float32)
        self.config = {"seed": 347, "device": "cpu", "deterministic": True, "torch_threads": 1,
                       "learning_rate": 1e-3, "batch_size_hamiltonians": 2, "max_epochs": 4,
                       "validation_batch_size": 2, "shot_chunk_size": 2, "activation_checkpoint": True,
                       "early_stopping": {"enabled": False, "min_epochs": 1,
                                          "patience_epochs": 2, "min_delta": 1e-6}}

    def session(self, method="set_cnn", checkpoint=None, config=None, **arrays):
        x = self.raw if method == "set_cnn" else self.raw.astype(np.float32).mean(1)
        vx = self.vraw if method == "set_cnn" else self.vraw.astype(np.float32).mean(1)
        return SnapshotTrainSession(arrays.get("train_x", x), arrays.get("train_y", self.y),
                                    arrays.get("val_x", vx), arrays.get("val_y", self.vy),
                                    self.config if config is None else config,
                                    method=method, checkpoint=checkpoint)

    def assert_tree_equal(self, first, second):
        self.assertIs(type(first), type(second))
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_epoch0_and_full_mse_match_independent_reduction(self):
        for method in METHODS:
            session = self.session(method)
            self.assertEqual((session.step, session.epoch, session.best_step, session.best_epoch), (0, 0, 0, 0))
            self.assertEqual(len(session.history), 1)
            with torch.no_grad():
                output = torch.cat([session.model(session.validation_x[i:i+2]) for i in range(0, 3, 2)])
                expected = (output - session.validation_y).double().square().mean().item()
            self.assertEqual(expected, session.best_val_mse)
            prediction = session.selected_predictions(self.vraw if method == "set_cnn"
                                                       else self.vraw.astype(np.float32).mean(1))
            self.assertEqual(prediction.shape, self.vy.shape)
            self.assertEqual(prediction.dtype, np.float32)

    def test_complete_short_batches_and_paired_hamiltonian_shuffle(self):
        mean, raw = self.session("mean_cnn"), self.session("set_cnn")
        generator = torch.Generator(device="cpu").manual_seed(self.config["seed"])
        for epoch in range(4):
            expected = torch.randperm(5, generator=generator)
            for update in range(3):
                mean.advance(1); raw.advance(1)
                self.assertTrue(torch.equal(mean.permutation, raw.permutation))
                self.assertTrue(torch.equal(raw.permutation, expected))
                self.assertEqual(raw.cursor, min(2 * (update + 1), 5))
            self.assertEqual(raw.hamiltonians_seen, 5 * (epoch + 1))
        self.assertEqual((raw.epoch, raw.step), (4, 12))
        self.assertTrue(raw.finished)
        self.assertEqual(raw.stop_reason, "max_epochs")
        self.assertEqual(raw.advance(1), [])

    def test_both_methods_resume_partial_epoch_bit_exact_with_weights_only(self):
        for method in METHODS:
            with self.subTest(method=method):
                direct = self.session(method); direct.advance(12)
                partial = self.session(method); partial.advance(4)
                checkpoint = partial.snapshot()
                self.assertEqual((checkpoint["engine"]["epoch"], checkpoint["engine"]["cursor"]), (1, 2))
                stream = io.BytesIO(); torch.save(checkpoint, stream); stream.seek(0)
                loaded = torch.load(stream, map_location="cpu", weights_only=True)
                resumed = self.session(method, loaded); resumed.advance(8)
                self.assert_tree_equal(direct.snapshot(), resumed.snapshot())

    def test_initial_and_epoch_boundary_resume(self):
        for method in METHODS:
            direct = self.session(method)
            restored = self.session(method, direct.snapshot())
            self.assert_tree_equal(direct.snapshot(), restored.snapshot())
            direct.advance(3); restored.advance(3)
            restored = self.session(method, restored.snapshot())
            direct.advance(9); restored.advance(9)
            self.assert_tree_equal(direct.snapshot(), restored.snapshot())

    def test_selected_inference_is_owned_unclipped_and_nonmutating(self):
        for method in METHODS:
            session = self.session(method); session.advance(4)
            values = self.vraw if method == "set_cnn" else self.vraw.astype(np.float32).mean(1)
            before = session.snapshot()
            session.selected_predictions(values, batch_size=1)
            self.assert_tree_equal(before, session.snapshot())
            session.model.eval()
            session.selected_predictions(values)
            self.assertFalse(session.model.training)
            state = session.snapshot()
            for t in state["engine"]["best_state_dict"].values():
                t.zero_()
            self.assert_tree_equal(before, session.snapshot())
            for t in session._best_state_dict.values():
                t.zero_()
            session._best_state_dict["head.2.bias"].fill_(-.5)
            np.testing.assert_array_equal(session.selected_predictions(values), np.full_like(self.vy, -.5))

    def test_best_selection_and_patience_use_distinct_thresholds(self):
        config = copy.deepcopy(self.config)
        config["early_stopping"] = {"enabled": True, "min_epochs": 1,
                                    "patience_epochs": 2, "min_delta": .05}
        losses = iter([2., 1.1, 2., 1., 2., .99, 2., .98])
        with mock.patch.object(SnapshotTrainSession, "_mse", side_effect=lambda *args: next(losses)):
            session = self.session(config=config)
            events = session.advance(100)
        self.assertEqual([e["early_stopping_wait"] for e in events], [0, 1, 2])
        self.assertEqual((session.epoch, session.best_epoch), (3, 3))
        self.assertEqual(session.best_val_mse, .98)
        self.assertEqual(session.stop_reason, "early_stopping")

    def test_epoch0_can_remain_selected(self):
        session = self.session()
        before = session.best_state_dict
        with mock.patch.object(session, "_mse", return_value=session.best_val_mse + 1):
            session.advance(6)
        self.assertEqual(session.best_epoch, 0)
        self.assert_tree_equal(before, session.best_state_dict)

    def test_checkpoints_bind_method_execution_data_and_policy(self):
        session = self.session(); session.advance(4); saved = session.snapshot()
        for method, config, arrays in (("mean_cnn", self.config, {}),
                                      ("set_cnn", {**self.config, "shot_chunk_size": 3}, {}),
                                      ("set_cnn", {**self.config, "activation_checkpoint": False}, {}),
                                      ("set_cnn", {**self.config, "max_epochs": 5}, {}),
                                      ("set_cnn", self.config, {"train_x": -self.raw})):
            with self.subTest(method=method, config=config), self.assertRaises(ValueError):
                self.session(method, saved, config, **arrays)
        for mutate in (lambda s: s.update(extra=1),
                       lambda s: s.update(training_schema_version=True),
                       lambda s: s["engine"]["history"][0].update(best_epoch=2),
                       lambda s: s["engine"].update(hamiltonians_seen=123),
                       lambda s: s["engine"]["optimizer_state_dict"]["state"][0]["step"].fill_(999)):
            corrupted = copy.deepcopy(saved); mutate(corrupted)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.session(checkpoint=corrupted)

    def test_arrays_settings_predictions_are_validated_and_owned(self):
        for changes in ({"train_x": self.raw * 0}, {"val_x": self.vraw[:, :2]},
                        {"train_x": self.raw[:, :, :1]}, {"train_y": -self.y},
                        {"val_y": self.vy[:2]}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.session(**changes)
        for config in ({**self.config, "shot_chunk_size": False}, {**self.config, "unknown": 1},
                       {**self.config, "device": "auto"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.session(config=config)
        session = self.session()
        self.raw.fill(0)
        self.assertTrue(bool(torch.all(session.train_x.abs() == 1)))
        for inputs in (self.vraw[:, :2], self.vraw.astype(np.float32) * .5):
            with self.assertRaises(ValueError):
                session.selected_predictions(inputs)
        settings = session.settings
        settings["early_stopping"]["enabled"] = True
        self.assertFalse(session.settings["early_stopping"]["enabled"])

    def test_cpu_testing_path_never_requests_cuda(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")), \
             mock.patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA RNG touched")), \
             mock.patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("CUDA RNG touched")):
            session = self.session(); session.advance(1)
            session.selected_predictions(self.vraw)
            saved = session.snapshot()
            self.assertEqual(saved["engine"]["cuda_rng_states"], [])
            self.session(checkpoint=saved)


if __name__ == "__main__":
    unittest.main()
