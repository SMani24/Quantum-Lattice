import json
import unittest

import numpy as np

from quantum_lattice.metrics.regression import per_hamiltonian_metrics, regression_report


class RegressionTests(unittest.TestCase):
    def test_metrics_are_raw_and_baseline_uses_training_mean(self):
        predictions = np.array([-0.2, 2.0, 0.5, 1.0])
        targets = np.array([0.0, 1.7, 0.3, 1.0])
        report = regression_report(predictions, targets, np.array(["a", "a", "b", "b"]),
                                   np.array([0, 1, 0, 1]), training_target_mean=0.8, bootstrap_samples=50)
        self.assertAlmostEqual(report["mse"], 0.0425)
        self.assertAlmostEqual(report["mae"], 0.175)
        self.assertEqual(report["negative_prediction_fraction"], 0.25)
        self.assertEqual(report["above_1_7_prediction_fraction"], 0.25)
        self.assertAlmostEqual(report["fixed_edge"]["metrics"]["mae"], 0.2)
        self.assertAlmostEqual(report["constant_training_mean_baseline"]["mae"], 0.6)
        json.dumps(report, allow_nan=False)

    def test_hamiltonian_bootstrap_resamples_whole_groups(self):
        # A row bootstrap would invent values between 0 and 10 in a group;
        # group bootstrap draws exactly (a,a), (a,b), (b,a) or (b,b).
        predictions = np.array([0., 0., 0., 10.])
        targets = np.zeros(4)
        groups = np.array(["a", "a", "a", "b"])
        edges = np.array([0, 1, 2, 0])
        report = regression_report(predictions, targets, groups, edges, bootstrap_samples=200, seed=17)
        random = np.random.default_rng(17)
        draws = []
        for _ in range(200):
            selected = random.integers(0, 2, 2)
            n_b = np.count_nonzero(selected == 1)
            n_a = 2 - n_b
            draws.append(10 * n_b / (3 * n_a + n_b))
        np.testing.assert_allclose(report["bootstrap"]["intervals"]["mae"], np.quantile(draws, [.025, .975]))
        self.assertEqual([r["count"] for r in report["per_hamiltonian"]], [3, 1])

    def test_bins_include_last_endpoint_once(self):
        y = np.array([0., .1, .3, .6, 1., 1.4, 1.7])
        report = regression_report(y, y, np.array(["one"] * 7), np.arange(7), bootstrap_samples=0)
        bins = report["target_magnitude_bins"]
        self.assertEqual(sum(b["metrics"]["count"] for b in bins), 7)
        self.assertEqual(bins[-1]["metrics"]["count"], 2)
        self.assertIsNone(report["bootstrap"]["intervals"])

    def test_per_group_and_deterministic_intervals(self):
        args = (np.arange(8) / 8, np.zeros((8, 1)), np.array(["a"] * 4 + ["b"] * 4), np.tile(np.arange(4), 2))
        first = regression_report(*args, seed=3)
        second = regression_report(*args, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(per_hamiltonian_metrics(*args), first["per_hamiltonian"])

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            regression_report([1], [-1], np.array(["a"]), np.array([0]))
        with self.assertRaises(ValueError):
            regression_report([np.nan], [1], np.array(["a"]), np.array([0]))
        with self.assertRaises(ValueError):
            regression_report([1], [1], np.array(["a"]), np.array([0]), bootstrap_samples=-1)
        with self.assertRaises(ValueError):
            regression_report([1], [1], np.array(["a"]), np.array([0]), training_target_mean=-1)


if __name__ == "__main__":
    unittest.main()
