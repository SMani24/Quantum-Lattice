from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.axes import Axes
from matplotlib import pyplot as plt
import numpy as np

from quantum_lattice.plotting.mlp import _binned_mae, plot_learning_history, plot_test_predictions


class MLPPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "plots"

    def check_files(self, paths):
        for path in paths:
            signature = path.read_bytes()[:8]
            if path.suffix == ".png":
                self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")
            else:
                self.assertTrue(signature.startswith(b"%PDF-"))
            self.assertGreater(path.stat().st_size, 1000)

    def test_history_and_prediction_files(self):
        history = [{"step": 10, "train_mse": .2, "validation_mse": .3},
                   {"step": 20, "train_mse": .1, "validation_mse": .15}]
        paths = plot_learning_history(history, self.output)
        self.assertEqual([p.name for p in paths], ["learning_curve.png", "learning_curve.pdf"])
        paths += plot_test_predictions(np.array([-.2, 2., .8])[:, None], np.array([0., 1.7, 1.]), self.output)
        self.assertEqual(len(paths), 6)
        self.check_files(paths)
        self.assertEqual(plt.get_fignums(), [])

    def test_scatter_preserves_out_of_range_predictions(self):
        original = Axes.scatter
        observed = []
        def spy(axis, x, y, *args, **kwargs):
            observed.append((np.asarray(x).copy(), np.asarray(y).copy()))
            return original(axis, x, y, *args, **kwargs)
        with patch.object(Axes, "scatter", spy):
            plot_test_predictions([-.3, 2.2], [0., 1.7], self.output)
        np.testing.assert_array_equal(observed[0][1], [-.3, 2.2])

    def test_bins_handle_boundaries_and_use_absolute_raw_errors(self):
        bins = _binned_mae(np.array([-.2, .4, 2.]), np.array([0., .1, 1.7]))
        self.assertEqual(sum(b["count"] for b in bins), 3)
        self.assertAlmostEqual(bins[0]["mae"], .2)
        self.assertAlmostEqual(bins[1]["mae"], .3)
        self.assertAlmostEqual(bins[-1]["mae"], .3)

    def test_invalid_data_rejected(self):
        with self.assertRaises(ValueError):
            plot_learning_history([], self.output)
        with self.assertRaises(ValueError):
            plot_learning_history([{"step": 1, "train_mse": -1, "validation_mse": 0}], self.output)
        with self.assertRaises(ValueError):
            plot_test_predictions([np.nan], [1.], self.output)
        with self.assertRaises(ValueError):
            plot_test_predictions([1.], [-1.], self.output)


if __name__ == "__main__":
    unittest.main()
