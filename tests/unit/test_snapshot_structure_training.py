"""Synthetic structural-session controls; no datasets or research jobs."""

import copy
import io
import unittest
from unittest import mock

import numpy as np
import torch

from quantum_lattice.data.snapshot_gauge import vertex_gauge_masks
from quantum_lattice.models.snapshot_structure_training import StructureTrainSession
from quantum_lattice.models.snapshot_training import SnapshotTrainSession


class StructureTrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(917)
        self.raw = rng.integers(0, 2, (5, 5, 2, 3, 3), dtype=np.int8) * 2 - 1
        self.vraw = rng.integers(0, 2, (3, 5, 2, 3, 3), dtype=np.int8) * 2 - 1
        self.y = rng.uniform(.05, .9, (5, 2, 3, 3)).astype(np.float32)
        self.vy = rng.uniform(.05, .9, (3, 2, 3, 3)).astype(np.float32)
        self.uids = [f"synthetic-H-{i}" for i in range(5)]
        self.config = {"seed": 913, "device": "cpu", "torch_threads": 1,
                       "learning_rate": 1e-3, "batch_size_hamiltonians": 2,
                       "max_epochs": 3, "validation_batch_size": 2,
                       "shot_chunk_size": 2, "activation_checkpoint": True,
                       "early_stopping": {"enabled": False}}

    def inputs(self, method):
        if method == "set_cnn":
            return self.raw, self.vraw
        return self.raw.astype(np.float32).mean(1), self.vraw.astype(np.float32).mean(1)

    def session(self, method="set_cnn", residual=True, gauge=True, checkpoint=None,
                uids=None, config=None, **arrays):
        x, vx = self.inputs(method)
        return StructureTrainSession(arrays.get("train_x", x), arrays.get("train_y", self.y),
                                     arrays.get("val_x", vx), arrays.get("val_y", self.vy),
                                     self.config if config is None else config, method=method,
                                     residual=residual, gauge_augmentation=gauge,
                                     train_uids=self.uids if uids is None else uids,
                                     checkpoint=checkpoint)

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

    def test_all_arms_partial_epoch_resume_exact_and_owned(self):
        for method in ("mean_cnn", "set_cnn"):
            for residual in (False, True):
                for gauge in (False, True):
                    with self.subTest(method=method, residual=residual, gauge=gauge):
                        direct = self.session(method, residual, gauge)
                        direct.advance(9)
                        partial = self.session(method, residual, gauge)
                        partial.advance(4)
                        self.assertEqual((partial.epoch, partial.cursor), (1, 2))
                        stream = io.BytesIO()
                        torch.save(partial.snapshot(), stream)
                        stream.seek(0)
                        saved = torch.load(stream, weights_only=True, map_location="cpu")
                        resumed = self.session(method, residual, gauge, saved)
                        resumed.advance(5)
                        self.assert_tree_equal(direct.snapshot(), resumed.snapshot())
                        vx = self.inputs(method)[1]
                        np.testing.assert_array_equal(direct.selected_predictions(vx),
                                                      resumed.selected_predictions(vx))
                        saved["engine"]["model_state_dict"]["head.2.bias"].fill_(777)
                        self.assert_tree_equal(direct.snapshot(), resumed.snapshot())

    def test_plain_unaugmented_reproduces_immutable_snapshot_baseline(self):
        for method in ("mean_cnn", "set_cnn"):
            x, vx = self.inputs(method)
            old = SnapshotTrainSession(x, self.y, vx, self.vy, self.config, method=method)
            new = self.session(method, residual=False, gauge=False)
            old.advance(9)
            new.advance(9)
            self.assert_tree_equal(old.history, new.history)
            self.assert_tree_equal(old.snapshot()["engine"]["optimizer_state_dict"],
                                   new.snapshot()["engine"]["optimizer_state_dict"])
            np.testing.assert_array_equal(old.selected_predictions(vx), new.selected_predictions(vx))

    def test_paired_bag_masks_have_no_initialization_or_shuffle_rng_cost(self):
        captures = {}
        initial = None
        for method in ("mean_cnn", "set_cnn"):
            for residual in (False, True):
                session = self.session(method, residual, gauge=True)
                if initial is None:
                    initial = copy.deepcopy(session.model.state_dict())
                self.assert_tree_equal(initial, session.model.state_dict())
                rng_before = torch.get_rng_state().clone()
                numpy_before = np.random.get_state()
                seen = []
                handle = session.model.register_forward_pre_hook(
                    lambda _model, args: seen.append(args[0].detach().cpu().numpy().copy()))
                session.advance(1)
                handle.remove()
                self.assertEqual(len(seen), 1)
                indices = session.permutation[:2].numpy()
                uids = [self.uids[i] for i in indices]
                expected_mask = vertex_gauge_masks(uids, 3, seed=self.config["seed"], epoch=0)
                original = self.inputs(method)[0][indices]
                expected = original * (expected_mask[:, None] if method == "set_cnn" else expected_mask)
                np.testing.assert_array_equal(seen[0], expected)
                np.testing.assert_array_equal(session.train_x.cpu(), self.inputs(method)[0])
                np.testing.assert_array_equal(session.train_y.cpu(), self.y)
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                numpy_after = np.random.get_state()
                self.assertEqual(numpy_before[0], numpy_after[0])
                np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
                self.assertEqual(numpy_before[2:], numpy_after[2:])
                expected_generator = torch.Generator(device="cpu").manual_seed(self.config["seed"])
                expected_permutation = torch.randperm(5, generator=expected_generator)
                self.assertTrue(torch.equal(session.permutation, expected_permutation))
                self.assertTrue(torch.equal(session.shuffle_generator.get_state(), expected_generator.get_state()))
                captures[method, residual] = seen[0]
        for residual in (False, True):
            np.testing.assert_allclose(captures["set_cnn", residual].mean(1),
                                       captures["mean_cnn", residual], atol=3e-8, rtol=0)

    def test_each_new_epoch_uses_its_uid_key_and_resume_regenerates_mask(self):
        session = self.session()
        first = vertex_gauge_masks(self.uids, 3, seed=self.config["seed"], epoch=0)
        second = vertex_gauge_masks(self.uids, 3, seed=self.config["seed"], epoch=1)
        self.assertFalse(np.array_equal(first, second))
        session.advance(3)
        saved = session.snapshot()
        resumed = self.session(checkpoint=saved)
        for run in (session, resumed):
            run.advance(1)
            indices = run.permutation[:2].numpy()
            np.testing.assert_array_equal(run._epoch_masks.cpu().numpy()[indices], second[indices])
        self.assert_tree_equal(session.snapshot(), resumed.snapshot())

    def test_validation_full_train_checks_and_selected_predictions_are_unaugmented(self):
        def independent_mse(session, inputs, targets):
            with torch.no_grad():
                output = torch.cat([session.model(inputs[i:i+2]) for i in range(0, len(inputs), 2)])
            difference = (output.numpy() - targets.numpy()).astype(np.float64)
            return float(np.square(difference).sum() / difference.size)

        with mock.patch("quantum_lattice.models.snapshot_structure_training.vertex_gauge_masks",
                        wraps=vertex_gauge_masks) as masks:
            session = self.session()
            self.assertEqual(masks.call_count, 0)
            expected = independent_mse(session, session.validation_x, session.validation_y)
            self.assertAlmostEqual(session.best_val_mse, expected, places=10)
            session.advance(3)
            self.assertEqual(masks.call_count, 1)
            v = independent_mse(session, session.validation_x, session.validation_y)
            t = independent_mse(session, session.train_x, session.train_y)
            self.assertAlmostEqual(session.history[-1]["validation_mse"], v, places=10)
            self.assertAlmostEqual(session.history[-1]["train_mse"], t, places=10)
            before = session.snapshot()
            prediction = session.selected_predictions(self.vraw, batch_size=1)
            self.assertEqual(masks.call_count, 1)
            selected = copy.deepcopy(session.model)
            selected.load_state_dict(session.best_state_dict)
            selected.eval()
            with torch.no_grad():
                expected = torch.cat([selected(session.validation_x[i:i+1]) for i in range(3)])
            np.testing.assert_array_equal(prediction, expected.numpy())
            self.assert_tree_equal(before, session.snapshot())

    def test_checkpoint_binds_factors_uid_order_schema_execution_and_data(self):
        run = self.session()
        run.advance(1)
        saved = run.snapshot()
        for changes in ({"residual": False}, {"gauge": False},
                        {"uids": list(reversed(self.uids))},
                        {"uids": [*self.uids[:-1], "replacement-H"]},
                        {"config": {**self.config, "shot_chunk_size": 3}},
                        {"train_x": -self.raw}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.session(checkpoint=saved, **changes)
        for mutate in (lambda s: s.update(residual=1),
                       lambda s: s.update(training_schema_version=True),
                       lambda s: s["augmentation_spec"].update(epoch_index="changed"),
                       lambda s: s["augmentation_spec"]["mask_spec"].update(unbound=1),
                       lambda s: s["train_uids"].reverse(),
                       lambda s: s["engine"].update(cursor=3)):
            corrupt = copy.deepcopy(saved)
            mutate(corrupt)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.session(checkpoint=corrupt)

    def test_best_selection_stopping_and_argument_validation(self):
        config = {**self.config, "max_epochs": 5,
                  "early_stopping": {"enabled": True, "min_epochs": 1,
                                     "patience_epochs": 2, "min_delta": .05}}
        losses = iter([2., 1.1, 2., 1., 2., .99, 2., .98])
        with mock.patch.object(StructureTrainSession, "_mse", side_effect=lambda *args: next(losses)):
            run = self.session(config=config)
            run.advance(100)
        self.assertEqual((run.epoch, run.best_epoch, run.stop_reason), (3, 3, "early_stopping"))
        self.assertEqual(run.best_val_mse, .98)
        for uids in ("abc", self.uids[:-1], ["same"] * 5, [*self.uids[:-1], ""], [1, 2, 3, 4, 5]):
            with self.subTest(uids=uids), self.assertRaises(ValueError):
                self.session(uids=uids)
        for changes in ({"residual": 1}, {"gauge": "yes"}, {"val_x": self.vraw[:, :2]},
                        {"train_y": -self.y}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.session(**changes)
        run = self.session()
        self.assertEqual(run.advance(0), [])
        for updates in (-1, True, .5):
            with self.assertRaises(ValueError):
                run.advance(updates)
        self.uids[0] = "caller mutation"
        self.assertEqual(run.train_uids[0], "synthetic-H-0")
        spec = run.augmentation_spec
        spec["enabled"] = False
        self.assertTrue(run.augmentation_spec["enabled"])

    def test_explicit_cpu_control_never_uses_cuda_and_default_has_no_fallback(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA touched")), \
             mock.patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA RNG touched")), \
             mock.patch("torch.cuda.get_rng_state_all", side_effect=AssertionError("CUDA RNG touched")):
            run = self.session()
            run.advance(1)
            saved = run.snapshot()
            self.session(checkpoint=saved)
            self.assertEqual(saved["engine"]["cuda_rng_states"], [])
        config = {key: value for key, value in self.config.items() if key != "device"}
        with mock.patch("quantum_lattice.models.snapshot_structure_training.resolve_device",
                        side_effect=RuntimeError("CUDA unavailable")) as resolver:
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                self.session(config=config)
            self.assertEqual(resolver.call_args.args[0], "cuda")


if __name__ == "__main__":
    unittest.main()
