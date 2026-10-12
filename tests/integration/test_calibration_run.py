"""Small real-data checks of retained calibration runs and exports."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from quantum_lattice.simulation.calibration_experiment import (
    render_run, run_experiment, validate_config,
)


ROOT = Path(__file__).resolve().parents[2]


def small_config():
    config = json.loads((ROOT / "configs/phase0_figures_11_12.json").read_text())
    config["geometry"]["side_lengths"] = [3, 4]
    config["sampling"]["error_probability_grid"]["num_points"] = 7
    config["sampling"]["independent_patterns_per_point"] = 400
    config["sampling"]["batch_size"] = 37
    config["fit"]["side_length"] = 4
    config["plotting"]["png_dpi"] = 60
    return config


class RunTests(unittest.TestCase):
    def test_retains_numeric_results_and_valid_exports_without_overwriting(self):
        with tempfile.TemporaryDirectory(prefix="ql-calibration-") as temporary:
            root = Path(temporary)
            config = small_config()
            directory = run_experiment(config, root, run_id="smoke")
            self.assertFalse((directory / "INCOMPLETE").exists())
            rows = json.loads((directory / "data/calibration.json").read_text())
            self.assertEqual(len(rows), 14)
            self.assertEqual({tuple(row["stream_spawn_key"]) for row in rows},
                             {(side, index) for side in [3, 4] for index in range(7)})
            verification = json.loads((directory / "verification.json").read_text())
            self.assertTrue(verification["zero_error_clean"])
            self.assertTrue(all(verification["even_total_syndrome_parity"].values()))
            fit = json.loads((directory / "fit_metrics.json").read_text())
            self.assertEqual(len(fit["coefficients"]), 4)
            manifest = json.loads((directory / "artifact_manifest.json").read_text())
            for paths in manifest.values():
                for relative in paths:
                    path = directory / relative
                    self.assertGreater(path.stat().st_size, 100)
                    if path.suffix == ".pdf":
                        self.assertTrue(path.read_bytes().startswith(b"%PDF"))
                    elif path.suffix == ".png":
                        self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            data = directory / "data/calibration.csv"
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            render_run(directory, root)
            self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), digest)
            rendering = json.loads((directory / "render_provenance.json").read_text())
            self.assertEqual(rendering["input_sha256"]["data/calibration.json"],
                             hashlib.sha256((directory / "data/calibration.json").read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                run_experiment(config, root, run_id="smoke")
            self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), digest)

    def test_rejects_artifacts_outside_local_material(self):
        with tempfile.TemporaryDirectory(prefix="ql-calibration-") as temporary:
            config = small_config()
            config["output_root"] = "published-results"
            with self.assertRaises(ValueError):
                run_experiment(config, Path(temporary), run_id="unsafe")
            with self.assertRaises(ValueError):
                render_run(Path(temporary) / "published-results", Path(temporary))
            self.assertFalse((Path(temporary) / "published-results").exists())

    def test_rejects_wrong_physical_protocol_and_fit(self):
        for area, key, value in [
            ("sampling", "model", "metropolis"),
            ("sampling", "record_uncertainty_across_whole_lattice_patterns", False),
            ("fit", "zero_intercept", False),
            ("fit", "input", "edge_error_probability"),
            ("fit", "target", "vertex_defect_probability"),
            ("plotting", "x_axis", "e_r"),
        ]:
            config = copy.deepcopy(small_config())
            config[area][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
