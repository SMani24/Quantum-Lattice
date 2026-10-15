"""Short patience and checkpoint checks, using only synthetic datasets."""

import copy
import io
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from quantum_lattice.models.mlp_training import TrainSession


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class EarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.train_x = rng.uniform(size=(17, 3)).astype(np.float32)
        self.validation_x = rng.uniform(size=(11, 3)).astype(np.float32)
        self.train_y = self.train_x[:, :1].copy()
        self.validation_y = self.validation_x[:, :1].copy()
        self.config = {
            "seed": 12, "learning_rate": 0.001, "batch_size": 5,
            "max_steps": 30, "validation_interval": 3, "device": "cpu",
            "torch_threads": 1, "deterministic": True,
            "early_stopping": {"enabled": True, "patience": 2,
                               "min_delta": 0.01, "min_steps": 6},
        }

    def session(self, config=None, checkpoint=None):
        return TrainSession(self.train_x, self.train_y, self.validation_x,
                            self.validation_y, config or self.config, checkpoint)

    def controlled_events(self, session, validation_values):
        # Each validation computes validation first, then full-training MSE.
        values = [item for val in validation_values for item in (val, 0.2)]
        with patch.object(session, "_mse", side_effect=values):
            return session.advance(100)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_patience_counts_checks_at_minimum_and_stops_inside_chunk(self):
        session = self.session()
        self.assertIsNone(session.stop_reason)
        events = self.controlled_events(session, [1, 1, 1, 1])
        self.assertEqual([event["step"] for event in events], [1, 3, 6, 9])
        self.assertEqual([event["early_stopping_wait"] for event in events], [0, 0, 1, 2])
        self.assertEqual(session.step, 9)
        self.assertTrue(session.finished)
        self.assertTrue(session.stopped_early)
        self.assertEqual(session.stop_reason, "early_stopping")
        self.assertEqual(session.advance(20), [])
        self.assertEqual(session.step, 9)

    def test_min_delta_resets_reference_but_strict_best_is_independent(self):
        config = copy.deepcopy(self.config)
        config["validation_interval"] = 1
        config["early_stopping"] = {"enabled": True, "patience": 2,
                                   "min_delta": 0.125, "min_steps": 0}
        session = self.session(config)
        events = self.controlled_events(session, [1, 0.9375, 0.8125, 0.6875, 0.625, 0.625, 0.625])
        self.assertEqual([event["early_stopping_wait"] for event in events], [0, 1, 0, 1, 0, 1, 2])
        self.assertEqual([event["early_stopping_reference"] for event in events],
                         [1, 1, 0.8125, 0.8125, 0.625, 0.625, 0.625])
        self.assertTrue(events[1]["improved"])
        self.assertTrue(events[3]["improved"])
        self.assertEqual(session.best_step, 5)
        self.assertEqual(session.best_val_mse, 0.625)
        self.assertEqual(session.step, 7)

    def test_tiny_improvement_at_stop_still_selects_best_weights(self):
        config = copy.deepcopy(self.config)
        config["validation_interval"] = 1
        config["early_stopping"]["min_steps"] = 0
        config["early_stopping"]["min_delta"] = 0.125
        session = self.session(config)
        self.controlled_events(session, [1, 0.99, 0.98])
        self.assertEqual(session.best_step, 3)
        self.assertEqual(session.best_val_mse, 0.98)
        self.assertEqual(session.early_stopping_reference, 1)
        self.assertEqual(session.step, 3)
        self.assertEqual(session.stop_reason, "early_stopping")
        self.assert_tree_equal(session.best_state_dict, session.model.state_dict())

    def test_resume_preserves_wait_rng_optimizer_and_selected_model(self):
        config = copy.deepcopy(self.config)
        config["early_stopping"].update(patience=3, min_steps=0, min_delta=10)
        full = self.session(config)
        full_events = full.advance(30)
        full_snapshot = full.snapshot()
        partial = self.session(config)
        earlier = partial.advance(4)
        self.assertEqual(partial.early_stopping_wait, 1)
        saved = io.BytesIO()
        torch.save(partial.snapshot(), saved)
        saved.seek(0)
        checkpoint = torch.load(saved, weights_only=True, map_location="cpu")
        restored = self.session(config, checkpoint)
        self.assertEqual(restored.early_stopping_wait, 1)
        later = restored.advance(30)
        self.assertEqual(full_events, earlier + later)
        self.assertEqual(full.step, 9)
        self.assert_tree_equal(full_snapshot, restored.snapshot())
        again = self.session(config, restored.snapshot())
        self.assertEqual(again.advance(2), [])
        self.assertEqual(again.stop_reason, "early_stopping")

    def test_disabled_stopping_runs_to_budget_and_legacy_snapshot_resumes(self):
        config = copy.deepcopy(self.config)
        config.pop("early_stopping")
        config["max_steps"] = 8
        full = self.session(config)
        full.advance(8)
        snapshot = full.snapshot()
        partial = self.session(config)
        partial.advance(4)
        legacy = partial.snapshot()
        legacy["training_schema_version"] = 1
        legacy["config"].pop("early_stopping")
        for key in ("stopped_early", "early_stopping_wait", "early_stopping_reference"):
            legacy.pop(key)
        restored = self.session(config, legacy)
        restored.advance(4)
        self.assertEqual(full.stop_reason, "max_steps")
        self.assertFalse(full.stopped_early)
        self.assertEqual(full.step, 8)
        self.assert_tree_equal(snapshot, restored.snapshot())
        with self.assertRaisesRegex(ValueError, "Legacy checkpoints"):
            self.session(self.config, legacy)

    def test_minimum_can_exceed_budget_without_patience_accumulation(self):
        config = copy.deepcopy(self.config)
        config["max_steps"] = 4
        config["early_stopping"]["min_steps"] = 100
        session = self.session(config)
        self.controlled_events(session, [1, 1, 1])
        self.assertEqual(session.step, 4)
        self.assertEqual(session.early_stopping_wait, 0)
        self.assertEqual(session.stop_reason, "max_steps")

    def test_stopping_configuration_types_and_ranges(self):
        invalid = [
            {"enabled": 1}, {"patience": 0}, {"patience": True},
            {"patience": 1.5}, {"min_steps": -1}, {"min_steps": False},
            {"min_delta": -1}, {"min_delta": float("nan")},
            {"min_delta": float("inf")}, {"min_delta": True},
            {"min_delta": None}, {"min_delta": "0.001"},
        ]
        for values in invalid:
            with self.subTest(values=values):
                config = copy.deepcopy(self.config)
                config["early_stopping"].update(values)
                with self.assertRaises(ValueError):
                    self.session(config)
        with self.assertRaises(ValueError):
            self.session(dict(self.config, early_stopping=[]))


if __name__ == "__main__":
    unittest.main()
