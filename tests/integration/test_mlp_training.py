"""Short synthetic checks; these do not train on the research dataset."""

import inspect
import io
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.mlp_training import TrainSession


@unittest.skipIf(torch is None, "PyTorch is installed in the separate MLP environment")
class TrainingSessionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.train_x = rng.uniform(0, 1, size=(61, 3)).astype(np.float32)
        self.validation_x = rng.uniform(0, 1, size=(19, 3)).astype(np.float32)
        self.train_y = (0.2 * self.train_x[:, :1] + 0.7 * self.train_x[:, 1:2]).astype(np.float32)
        self.validation_y = (0.2 * self.validation_x[:, :1] + 0.7 * self.validation_x[:, 1:2]).astype(np.float32)
        self.config = {
            "seed": 14, "learning_rate": 0.003, "batch_size": 13,
            "max_steps": 50, "validation_interval": 7, "device": "cpu",
            "torch_threads": 1, "deterministic": True,
        }

    def session(self, config=None, checkpoint=None):
        return TrainSession(self.train_x, self.train_y, self.validation_x,
                            self.validation_y, self.config if config is None else config,
                            checkpoint=checkpoint)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_learns_tiny_relation_and_checks_validation_schedule(self):
        session = self.session()
        initial_prediction = session.predict(self.validation_x, use_best=False)
        initial_mse = np.mean((initial_prediction - self.validation_y) ** 2)
        events = session.advance(100)
        self.assertTrue(session.finished)
        self.assertEqual(session.step, 50)
        self.assertEqual([event["step"] for event in events], [1, 7, 14, 21, 28, 35, 42, 49, 50])
        prediction = session.predict(self.validation_x)
        final_mse = np.mean((prediction - self.validation_y) ** 2)
        self.assertLess(final_mse, initial_mse / 10)
        self.assertAlmostEqual(final_mse, session.best_val_mse, places=7)
        self.assertEqual(session.best_step, min(events, key=lambda event: event["validation_mse"])["step"])
        self.assertEqual(session.advance(1), [])

    def test_checkpoint_resume_is_identical_including_adam_and_rng(self):
        config = dict(self.config, max_steps=35)
        full = self.session(config)
        full_events = full.advance(35)
        full_snapshot = full.snapshot()
        partial = self.session(config)
        partial_events = partial.advance(15)
        # Exercise the restricted deserializer used by the runner.
        stream = io.BytesIO()
        torch.save(partial.snapshot(), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True, map_location="cpu")
        restored = self.session(config, checkpoint=saved)
        remaining_events = restored.advance(20)
        self.assertEqual(full_events, partial_events + remaining_events)
        self.assert_tree_equal(full_snapshot, restored.snapshot())
        np.testing.assert_array_equal(full.predict(self.validation_x), restored.predict(self.validation_x))

    def test_best_state_is_preserved_when_later_validation_worsens(self):
        session = self.session(dict(self.config, validation_interval=1))
        session.advance(1)
        best = session.best_state_dict
        with torch.no_grad():
            session.model.layers[-1].bias.fill_(100)
        session.advance(1)
        self.assertEqual(session.best_step, 1)
        self.assert_tree_equal(best, session.best_state_dict)
        self.assertGreater(float(session.predict(self.validation_x, use_best=False).mean()), 90)
        self.assertLess(float(session.predict(self.validation_x, use_best=True).mean()), 2)

    def test_snapshot_is_detached_from_live_training(self):
        session = self.session()
        session.advance(2)
        snapshot = session.snapshot()
        previous = snapshot["model_state_dict"]["layers.0.weight"].clone()
        session.advance(3)
        self.assertTrue(torch.equal(previous, snapshot["model_state_dict"]["layers.0.weight"]))

    def test_checkpoint_rejects_changed_data_or_scientific_settings(self):
        session = self.session()
        session.advance(2)
        snapshot = session.snapshot()
        changed = self.train_x.copy()
        changed[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "data differ"):
            TrainSession(changed, self.train_y, self.validation_x, self.validation_y,
                         self.config, checkpoint=snapshot)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            self.session(dict(self.config, learning_rate=0.004), snapshot)

    def test_input_validation_and_no_final_test_parameter(self):
        self.assertNotIn("test_x", inspect.signature(TrainSession).parameters)
        with self.assertRaisesRegex(ValueError, "counts differ"):
            TrainSession(self.train_x[:-1], self.train_y, self.validation_x,
                         self.validation_y, self.config)
        bad = self.train_x.copy()
        bad[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            TrainSession(bad, self.train_y, self.validation_x, self.validation_y, self.config)
        with self.assertRaises(ValueError):
            self.session(dict(self.config, batch_size=0))


if __name__ == "__main__":
    unittest.main()
