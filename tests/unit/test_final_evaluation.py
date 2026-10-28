"""Independent score replay, immutable exports and grouped inference checks."""
from pathlib import Path
import csv
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian import hamiltonian_error
from quantum_lattice.metrics.hamiltonian_batch import CompiledHamiltonianMetric
from quantum_lattice.models.final_evaluation import (
    evaluate_predictions, load_evaluation, paired_comparison, compare_seed_panels,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.L, self.count = 3, 4
        rng = np.random.default_rng(204201)
        self.fields = rng.uniform(-1.7, 1.7, (self.count, 18))
        self.fields[0, 0] = 0
        self.prediction = rng.uniform(-.6, 2, (self.count, 2, 3, 3)).astype(np.float32)
        self.ids = np.array([f"H{i}" for i in range(self.count)])
        self.uids = np.array([f"uid{i}" for i in range(self.count)])
        self.bindings = {"cohort": "fixture", "checkpoint_sha256": "a"*64,
                         "scope": "synthetic score fixture; no scientific data"}

    def evaluate(self, **kwargs):
        values = dict(L=self.L, prediction=self.prediction, fields=self.fields,
                      ids=self.ids, uids=self.uids, output_dir=self.root / "evaluation",
                      bindings=self.bindings, metric_batch_size=2)
        values.update(kwargs)
        return evaluate_predictions(**values)

    def test_edge_layout_negative_predictions_and_scalar_D5_oracle(self):
        result = self.evaluate()
        # Build physical edge order independently using explicit coordinates.
        flat = np.array([[self.prediction[h, axis, y, x]
                          for y in range(3) for x in range(3) for axis in (0,1)]
                         for h in range(self.count)], dtype=np.float64)
        error = flat - np.abs(self.fields)
        self.assertAlmostEqual(result["metrics"]["mse"], np.mean(error**2), places=14)
        self.assertAlmostEqual(result["metrics"]["mae"], np.mean(np.abs(error)), places=14)
        self.assertEqual(result["metrics"]["negative_prediction_fraction"], np.mean(flat < 0))
        lattice = square_lattice(3)
        for model in ("two_channel", "z_branch"):
            for label, identity in (("full", True), ("traceless", False)):
                expected = [hamiltonian_error(lattice, self.fields[h], np.sign(self.fields[h])*flat[h],
                            model=model, include_identity=identity) for h in range(self.count)]
                np.testing.assert_allclose([r[f"reference_sign_D5_{model}_{label}"] for r in result["per_hamiltonian"]],
                                           expected, atol=3e-15, rtol=2e-13)
                self.assertAlmostEqual(result["metrics"]["reference_sign_D5"][f"{model}_{label}"], np.mean(expected), places=13)
        with np.load(self.root / "evaluation/predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["prediction_maps"], self.prediction)
            np.testing.assert_array_equal(archive["prediction"], flat)
            np.testing.assert_array_equal(archive["target"], np.abs(self.fields))
        with (self.root / "evaluation/per-hamiltonian.csv").open(newline="") as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual([r["hamiltonian_uid"] for r in rows], self.uids.tolist())

    def test_native_signed_metrics_are_separate_and_zero_sign_policy_explicit(self):
        native = self.fields.copy()
        native[:,1] *= -1
        result = self.evaluate(native_signed=native)
        self.assertIn("native_signed", result["metrics"])
        self.assertEqual(result["metrics"]["native_signed"]["zero_truth_edges"], 1)
        self.assertAlmostEqual(result["metrics"]["native_signed"]["field_mse"], np.mean((native-self.fields)**2))
        expected = [hamiltonian_error(square_lattice(3), self.fields[h], native[h]) for h in range(self.count)]
        np.testing.assert_allclose([r["native_signed_D5_two_channel_full"] for r in result["per_hamiltonian"]], expected,
                                  atol=3e-15, rtol=2e-13)
        absent = self.evaluate(output_dir=self.root / "magnitude-only")
        self.assertNotIn("native_signed", absent["metrics"])

    def test_identical_commit_is_unchanged_and_mixed_inputs_are_refused(self):
        first = self.evaluate()
        directory = self.root / "evaluation"
        before = {p.name:sha(p) for p in directory.iterdir()}
        self.assertEqual(self.evaluate(), first)
        self.assertEqual(before, {p.name:sha(p) for p in directory.iterdir()})
        self.assertEqual(load_evaluation(directory, self.bindings), first)
        changed = self.prediction.copy()
        changed[0,0,0,0] += .2
        with self.assertRaisesRegex(ValueError, "content/bindings changed"):
            self.evaluate(prediction=changed)
        with self.assertRaisesRegex(ValueError, "content/bindings changed"):
            self.evaluate(bindings={**self.bindings,"cohort":"other"})
        self.assertEqual(before, {p.name:sha(p) for p in directory.iterdir()})

    def test_corruption_rejected_and_incomplete_commit_recomputed(self):
        original = self.evaluate()
        directory = self.root / "evaluation"
        (directory / "done.json").unlink()
        self.assertEqual(self.evaluate(), original)
        with (directory / "predictions.npz").open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.evaluate()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_evaluation(directory)

    def test_large_lattice_metric_batches_are_bounded(self):
        L, count = 24, 35
        fields = np.ones((count,2*L*L), dtype=np.float64)*.03
        prediction = np.ones((count,2,L,L), dtype=np.float32)*.02
        calls=[]
        real = CompiledHamiltonianMetric.distances
        def wrapped(metric, truth, recovered, *args, **kwargs):
            calls.append((metric.model,len(truth)))
            return real(metric,truth,recovered,*args,**kwargs)
        with patch.object(CompiledHamiltonianMetric,"distances",new=wrapped):
            result=self.evaluate(L=L, fields=fields, prediction=prediction,
                ids=np.array([f"H{i}" for i in range(count)]),
                uids=np.array([f"U{i}" for i in range(count)]), metric_batch_size=16)
        self.assertEqual(result["hamiltonians"],count)
        self.assertEqual(calls,[("two_channel",16),("two_channel",16),("two_channel",3),
                               ("z_branch",16),("z_branch",16),("z_branch",3)])
        self.assertTrue(np.isfinite(result["metrics"]["reference_sign_D5"]["two_channel_full"]))

    def test_invalid_identity_finite_shape_bindings_and_batch_refused(self):
        for kwargs in [dict(uids=np.array(["same"]*4)), dict(ids=np.array(["", "b", "c", "d"])),
                       dict(prediction=self.prediction[:,0]), dict(prediction=self.prediction*np.nan),
                       dict(fields=np.zeros((4,17))), dict(bindings={}), dict(metric_batch_size=33)]:
            with self.subTest(kwargs=list(kwargs)):
                with self.assertRaises(ValueError): self.evaluate(**kwargs)
        directory=self.root / "unbound"
        directory.mkdir()
        (directory/"unrelated.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError,"nonempty unbound"):
            self.evaluate(output_dir=directory)
        self.assertEqual((directory/"unrelated.txt").read_text(),"preserve")


class PairedComparisonTests(unittest.TestCase):
    def test_paired_H_bootstrap_matches_independent_small_vector_calculation(self):
        left=np.array([1.,4.,9.]); right=np.array([2.,3.,5.])
        actual=paired_comparison(left,right,uids=["a","b","c"],bootstrap_samples=301,seed=41)
        rng=np.random.default_rng(41)
        indices=rng.integers(0,3,size=(301,3))
        manual=np.array([np.mean([left[i]-right[i] for i in row]) for row in indices])
        np.testing.assert_array_equal(actual["bootstrap_95ci_difference"], np.quantile(manual,[.025,.975]))
        self.assertAlmostEqual(actual["difference_left_minus_right"], 4/3)
        self.assertAlmostEqual(actual["relative_reduction_left_vs_right"], 1-left.mean()/right.mean())

    def test_seeds_are_retained_within_H_and_not_pooled_as_independent_H(self):
        left={10:np.array([2.,2.,2.]),20:np.array([4.,4.,4.])}
        right={10:np.ones(3),20:np.ones(3)}
        actual=compare_seed_panels(left,right,uids=["a","b","c"],bootstrap_samples=1000,seed=1)
        # All seed-averaged H differences are exactly2. Pooling six seed/H
        # cells would create false bootstrap variation, and must not happen.
        self.assertEqual(actual["hamiltonians"],3)
        self.assertEqual(actual["optimizer_seeds"],2)
        self.assertEqual(actual["bootstrap_95ci_difference"],[2.,2.])
        self.assertEqual([r["difference_left_minus_right"] for r in actual["per_seed"]],[1.,3.])
        self.assertAlmostEqual(actual["sample_sd_seed_differences"],np.sqrt(2))

    def test_uid_order_seed_pairing_and_empty_inputs_are_refused(self):
        with self.assertRaisesRegex(ValueError,"UID order"):
            paired_comparison([1,2],[2,1],uids=["a","b"],right_uids=["b","a"])
        with self.assertRaisesRegex(ValueError,"same nonempty"):
            compare_seed_panels({1:[1,2]},{2:[1,2]},uids=["a","b"])
        for left,right,uids in [([],[],[]),([1,2],[1],["a","b"]),([np.nan],[1],["a"]),([1,2],[2,3],["a","a"])]:
            with self.assertRaises(ValueError): paired_comparison(left,right,uids=uids)

    def test_zero_reference_and_single_seed_are_reported_honestly(self):
        actual=compare_seed_panels({1:[1,1]},{1:[0,0]},uids=["a","b"],bootstrap_samples=10,seed=0)
        self.assertIsNone(actual["relative_reduction_left_vs_right"])
        self.assertIsNone(actual["sample_sd_seed_differences"])


if __name__ == "__main__":
    unittest.main()
