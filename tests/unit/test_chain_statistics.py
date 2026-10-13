"""Independent checks of recorded-chain summaries and diagnostic limits."""

import unittest

import numpy as np

from quantum_lattice.observables.statistics import autocorrelation_diagnostic, summarize


class ChainStatisticsTests(unittest.TestCase):
    def test_block_means_standard_error_and_partial_tail(self):
        values = np.array([[1, 5], [3, 1], [6, 10], [10, 2], [100, -20]], dtype=float)
        result = summarize(values, block_size=2)
        # Two complete blocks; the fifth sample still contributes to the mean.
        block_oracle = np.array([[2, 3], [8, 6]], dtype=float)
        np.testing.assert_array_equal(result["block_means"], block_oracle)
        np.testing.assert_allclose(result["mean"], [24.0, -0.4], rtol=0, atol=0)
        np.testing.assert_allclose(result["block_se"], [3.0, 1.5], atol=1e-15)
        iid_oracle = np.sqrt(((values - values.mean(axis=0)) ** 2).sum(axis=0) / (len(values) - 1) / len(values))
        np.testing.assert_allclose(result["iid_se"], iid_oracle, atol=1e-15)

    def test_missing_uncertainty_is_distinct_from_zero_uncertainty(self):
        single = summarize(np.array([[1.0, 2.0]]), block_size=4)
        self.assertTrue(np.isnan(single["iid_se"]).all())
        self.assertTrue(np.isnan(single["block_se"]).all())
        self.assertEqual(single["block_means"].shape, (0, 2))
        one_block = summarize(np.ones((4, 2)), block_size=4)
        self.assertTrue(np.isnan(one_block["block_se"]).all())
        two_blocks = summarize(np.ones((8, 2)), block_size=4)
        np.testing.assert_array_equal(two_blocks["block_se"], np.zeros(2))

    def test_iid_scalar_diagnostic_is_plausible(self):
        trace = np.random.default_rng(182).normal(size=12000)
        result = autocorrelation_diagnostic(trace)
        self.assertGreaterEqual(result["tau_records"], 1)
        self.assertLess(result["tau_records"], 1.5)
        self.assertGreater(result["ess"], 8000)
        self.assertLessEqual(result["ess"], len(trace))
        self.assertAlmostEqual(result["ess"] * result["tau_records"], len(trace))

    def test_ar1_trace_reduces_ess_in_expected_order(self):
        rng = np.random.default_rng(73)
        coefficient = 0.8
        noise = rng.normal(size=14000)
        trace = np.zeros(len(noise))
        for index in range(1, len(trace)):
            trace[index] = coefficient * trace[index - 1] + noise[index]
        result = autocorrelation_diagnostic(trace[2000:])
        theoretical_tau = (1 + coefficient) / (1 - coefficient)
        self.assertGreater(result["tau_records"], 0.55 * theoretical_tau)
        self.assertLess(result["tau_records"], 1.7 * theoretical_tau)
        self.assertGreater(result["ess"], 600)
        self.assertLess(result["ess"], 3000)

    def test_short_nonfinite_and_constant_traces_have_no_ess_claim(self):
        for trace in ([1, 2, 3], [1, np.nan, 3, 4], np.ones(30), np.zeros(30)):
            result = autocorrelation_diagnostic(np.asarray(trace))
            self.assertIsNone(result["tau_records"])
            self.assertIsNone(result["ess"])
        self.assertEqual(autocorrelation_diagnostic(np.ones(30))["reason"], "constant trace")

    def test_bad_summary_shape_and_block_length(self):
        for values, block_size in ((np.empty((0, 2)), 4), (np.ones(4), 4), (np.ones((4, 2)), 0)):
            with self.assertRaises(ValueError):
                summarize(values, block_size)


if __name__ == "__main__":
    unittest.main()
