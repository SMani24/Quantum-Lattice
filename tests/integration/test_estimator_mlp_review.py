"""Independent pairing controls for the estimator-only learning ablation.

These synthetic CPU controls exercise randomization rather than research
training. Feature values differ between the two views, while row identities,
targets, initialization seed and shuffle seed stay paired.
"""

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from quantum_lattice.models.mlp_training import TrainSession


@unittest.skipIf(torch is None, "Use the project .venv with PyTorch")
class EstimatorPairingReviewTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(107)
        self.f_train = rng.uniform(0.1, 1.3, (17, 3)).astype(np.float32)
        self.g_train = (0.7 * self.f_train + 0.05).astype(np.float32)
        self.f_validation = rng.uniform(0.1, 1.3, (11, 3)).astype(np.float32)
        self.g_validation = (0.7 * self.f_validation + 0.05).astype(np.float32)
        self.train_y = rng.uniform(0, 1.7, (17, 1)).astype(np.float32)
        self.validation_y = rng.uniform(0, 1.7, (11, 1)).astype(np.float32)
        self.config = {
            "seed": 20261016,
            "learning_rate": 1e-5,
            "batch_size": 5,
            "max_steps": 11,
            "validation_interval": 4,
            "device": "cpu",
            "torch_threads": 1,
            "deterministic": True,
            "early_stopping": {"enabled": False},
        }

    def make(self, view, checkpoint=None):
        return TrainSession(
            getattr(self, f"{view}_train"), self.train_y,
            getattr(self, f"{view}_validation"), self.validation_y,
            self.config, checkpoint=checkpoint,
        )

    def test_initial_weights_and_every_row_permutation_are_paired(self):
        f = self.make("f")
        g = self.make("g")
        for name, value in f.model.state_dict().items():
            self.assertTrue(torch.equal(value, g.model.state_dict()[name]), name)
        self.assertNotEqual(f.data_fingerprint, g.data_fingerprint)
        # Unequal feature values must not alter the independent CPU shuffle RNG,
        # including the shorter last batch and the next epoch boundary.
        for _ in range(self.config["max_steps"]):
            f.advance(1)
            g.advance(1)
            self.assertEqual(f.cursor, g.cursor)
            self.assertEqual(f.epoch, g.epoch)
            self.assertTrue(torch.equal(f.permutation, g.permutation))
            self.assertTrue(torch.equal(
                f.shuffle_generator.get_state(), g.shuffle_generator.get_state()))
        self.assertTrue(any(not torch.equal(value, g.model.state_dict()[name])
                            for name, value in f.model.state_dict().items()))

    def test_crossed_prediction_does_not_change_selected_checkpoint_or_rng(self):
        session = self.make("f")
        session.advance(5)
        model_before = {k: v.clone() for k, v in session.model.state_dict().items()}
        selected_before = session.best_state_dict
        shuffle_before = session.shuffle_generator.get_state().clone()
        torch_before = torch.get_rng_state().clone()
        first = session.predict(self.f_validation, use_best=True)
        session.predict(self.g_validation, use_best=True)
        repeated = session.predict(self.f_validation, use_best=True)
        np.testing.assert_array_equal(first, repeated)
        for name, value in model_before.items():
            self.assertTrue(torch.equal(value, session.model.state_dict()[name]))
            self.assertTrue(torch.equal(selected_before[name], session.best_state_dict[name]))
        self.assertTrue(torch.equal(shuffle_before, session.shuffle_generator.get_state()))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_a_checkpoint_cannot_be_reused_for_the_other_estimator(self):
        session = self.make("f")
        session.advance(2)
        with self.assertRaisesRegex(ValueError, "training/validation data differ"):
            self.make("g", checkpoint=session.snapshot())


if __name__ == "__main__":
    unittest.main()
