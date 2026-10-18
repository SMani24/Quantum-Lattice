"""Aggregation and raw-output oracle contracts for frozen D5 evaluation."""

from copy import deepcopy
import unittest

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian import hamiltonian_error
from quantum_lattice.metrics.hamiltonian_batch import CompiledHamiltonianMetric
from quantum_lattice.metrics.hamiltonian_evaluation import (
    CONTEXTS, METRICS, VIEWS, score_fields, summarize_scores, validate_config,
)


def configuration():
    return {
        "schema_version": 1,
        "campaigns": [{"label": "parent", "run_dir": "codex material/artifacts/parent"},
                      {"label": "continued", "run_dir": "codex material/artifacts/continued"}],
        "split": "validation", "sign_source": "reference_sign_oracle",
        "prediction_policy": "raw_unclipped", "primary_metric": METRICS[0],
        "include_zero_field_baseline": True, "batch_size": 2,
    }


class HamiltonianEvaluationTests(unittest.TestCase):
    def test_oracle_keeps_negative_raw_predictions_and_zero_reference_rule(self):
        lattice = square_lattice(3)
        true = np.array([[0., .3, -.4] * 6, [-.2, 0., .7] * 6])
        raw = np.array([[-.9, -.1, .6] * 6, [.1, .8, -.2] * 6], dtype=np.float32)
        compiled = {name: CompiledHamiltonianMetric(lattice, model=name)
                    for name in ("two_channel", "z_branch")}
        scores, errors, residual = score_fields(true, raw, compiled, batch_size=1)
        # This is the declared lift, independently formed without the adapter.
        recovered = np.sign(true) * raw.astype(np.float64)
        self.assertEqual(recovered[0, 0], 0.)
        self.assertLess(recovered[0, 1], 0.)
        self.assertGreater(recovered[1, 0], -.2)
        for model in compiled:
            for identity in (True, False):
                name = model + ("_full" if identity else "_traceless")
                expected = [hamiltonian_error(lattice, t, r, model=model,
                                             include_identity=identity)
                            for t, r in zip(true, recovered)]
                np.testing.assert_allclose(scores[name], expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(errors["mse"], np.mean((true-recovered)**2, axis=1))
        np.testing.assert_allclose(errors["mae"], np.mean(abs(true-recovered), axis=1))
        np.testing.assert_allclose(scores[METRICS[1]], scores[METRICS[3]], atol=2e-14)
        self.assertLessEqual(residual, 1e-12)
        # Taking abs(raw) would silently repair two deliberately wrong signs.
        self.assertGreater(errors["mse"][0], np.mean((true[0]-np.sign(true[0])*abs(raw[0]))**2))

    def test_batch_boundaries_and_exact_recovery_preserve_per_hamiltonian_values(self):
        lattice = square_lattice(3)
        true = np.linspace(-.9, .8, 5*18).reshape(5, 18)
        compiled = {name: CompiledHamiltonianMetric(lattice, model=name)
                    for name in ("two_channel", "z_branch")}
        whole, errors, _ = score_fields(true, abs(true), compiled, 5)
        chunked, _, _ = score_fields(true, abs(true), compiled, 2)
        for name in METRICS:
            np.testing.assert_allclose(whole[name], 0., atol=2e-14)
            np.testing.assert_allclose(chunked[name], whole[name], atol=2e-14)
        np.testing.assert_array_equal(errors["mse"], np.zeros(5))

    def test_arithmetic_group_then_equal_seed_means_not_rms_or_mean_ratios(self):
        scores = {}
        # Seed reductions are +20% and -20%; the ratio of means is +10%.
        # g is deliberately worse on common f inputs in every seed.
        for campaign, scale in (("parent", 1.), ("continued", .9)):
            for seed, f_values, g_values in ((7, [.2, 1.], [.12, .84]),
                                            (9, [.1, .3], [.12, .36])):
                for training in ("f", "g"):
                    for view in VIEWS:
                        values = np.array(f_values if training == "f" else g_values)
                        if training == "g" and view == "f":
                            values = np.array(f_values) + .1
                        scores[campaign, 3, seed, training, view] = {
                            metric: values * scale * (1.+.05*i)
                            for i, metric in enumerate(METRICS)}
        groups, pairs, contrasts, parent = summarize_scores(
            ["parent", "continued"], [3], [7, 9], scores)
        self.assertEqual(len(groups), 2*2*3*4)
        self.assertEqual(len(pairs), 2*4*4*2)
        self.assertEqual(len(contrasts), 2*4*4)
        self.assertEqual(len(parent), 2*2*3*4)
        self.assertEqual({r["context"] for r in contrasts}, set(CONTEXTS))
        self.assertEqual({r["metric"] for r in contrasts}, set(METRICS))
        row = next(r for r in contrasts if r["campaign"] == "parent"
                   and r["context"] == "matched_pipeline" and r["metric"] == METRICS[0])
        self.assertAlmostEqual(row["f_mean_d5"], .4)
        self.assertAlmostEqual(row["g_mean_d5"], .36)
        self.assertAlmostEqual(row["relative_reduction"], .1)
        self.assertEqual(row["seeds_g_better"], 1)
        self.assertEqual(row["hamiltonians_g_better_after_seed_average"], 2)
        ratios = [r["relative_reduction"] for r in pairs
                  if r["campaign"] == "parent" and r["context"] == "matched_pipeline"
                  and r["metric"] == METRICS[0]]
        self.assertAlmostEqual(np.mean(ratios), 0.)
        self.assertNotAlmostEqual(row["f_mean_d5"], np.sqrt(np.mean([.2**2, 1.**2, .1**2, .3**2])))
        unfavorable = [r for r in contrasts if r["context"] == "f"]
        self.assertTrue(all(r["relative_reduction"] < 0 for r in unfavorable))
        self.assertTrue(all(r["seeds_g_better"] == 0 for r in unfavorable))
        self.assertTrue(all(abs(r["relative_reduction"]-.1)<1e-14 for r in parent))

    def test_zero_denominator_contrast_is_explicitly_undefined(self):
        scores = {(campaign, 3, 1, training, view): {m: np.zeros(2) for m in METRICS}
                  for campaign in ("a", "b") for training in ("f", "g") for view in VIEWS}
        _, pairs, contrasts, changes = summarize_scores(["a", "b"], [3], [1], scores)
        self.assertTrue(all(r["relative_reduction"] is None for r in pairs+contrasts+changes))

    def test_configuration_rejects_test_scope_clipping_oracle_changes_and_bool_numbers(self):
        self.assertEqual(validate_config(configuration()), configuration())
        for key, value in (("schema_version", True), ("split", "test"),
                           ("sign_source", "learned"), ("prediction_policy", "clipped"),
                           ("primary_metric", METRICS[1]), ("include_zero_field_baseline", False),
                           ("batch_size", True), ("batch_size", 0), ("batch_size", 1025),
                           ("extra", 1)):
            with self.subTest(key=key, value=value):
                bad = configuration(); bad[key] = value
                with self.assertRaises(ValueError): validate_config(bad)
        for campaigns in ([], configuration()["campaigns"][:1],
                          [configuration()["campaigns"][0]]*2,
                          [{"label": "../bad", "run_dir": "a"}, {"label": "b", "run_dir": "b"}]):
            bad = deepcopy(configuration()); bad["campaigns"] = campaigns
            with self.assertRaises(ValueError): validate_config(bad)


if __name__ == "__main__":
    unittest.main()
