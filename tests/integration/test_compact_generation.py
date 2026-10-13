"""Compact records preserve the full scientific result across pause/resume."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.generation import (_advance_job, _lattice, artifact_root, inspect_run, prepare_run,
                                           run_generation, validate_config, verify_run)
from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.solvable import exact_expectations
from quantum_lattice.simulation.solvable import _proposal_kernel
from tests.integration.test_generation import load_arrays, tiny_config


REPO = Path(__file__).resolve().parents[2]


class StrongFieldExactTests(unittest.TestCase):
    def test_diagnostic_reduction_is_identical_across_process_thread_limits(self):
        """Synthetic L=24 records check reproducibility without any simulation."""
        script = """import json, sys
import numpy as np
from quantum_lattice.data.generation import _log_weight_trace
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.statistics import autocorrelation_diagnostic
lattice = square_lattice(24)
rng = np.random.default_rng(154)
pseudo = (1 - 2 * rng.integers(2, size=(4500, lattice.n_vertices))).astype(np.int8)
ends = lattice.edge_vertex_indices
states = pseudo[:, ends[:, 0]] * pseudo[:, ends[:, 1]]
fields = rng.uniform(-1.7, 1.7, lattice.n_edges)
trace = _log_weight_trace(states, fields)
np.savez(sys.argv[1], trace=trace,
    diagnostic=np.array(json.dumps(autocorrelation_diagnostic(trace), sort_keys=True)))
"""
        with tempfile.TemporaryDirectory(prefix="quantum-lattice-reductions-", dir="/tmp") as root:
            paths = []
            for threads in (1, 4):
                path = Path(root) / f"threads-{threads}.npz"
                env = os.environ.copy()
                env.update({key: str(threads) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
                subprocess.run([sys.executable, "-c", script, str(path)], env=env, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                paths.append(path)
            first, second = (load_arrays(path) for path in paths)
            np.testing.assert_array_equal(first["trace"], second["trace"])
            np.testing.assert_array_equal(first["diagnostic"], second["diagnostic"])

    def test_strong_signed_fields_against_wavefunction_permutations_and_stationarity(self):
        """Finite exact check; this does not assert mixing on the campaign sizes."""
        lattice = square_lattice(3)
        fields = np.linspace(-1.7, 1.7, lattice.n_edges)
        # Independently generate the reference orbit by multiplying explicit
        # star flips, with the last star fixed to remove the global redundancy.
        states = np.ones((256, lattice.n_edges), dtype=np.int8)
        for label in range(256):
            for vertex in range(8):
                if label & (1 << vertex):
                    for edge in lattice.vertices[vertex].incident_edges:
                        states[label, edge.id] *= -1
        lookup = {row.tobytes(): i for i, row in enumerate(states)}
        log_weights = states @ fields
        weights = np.exp(log_weights - log_weights.max())
        probabilities = weights / weights.sum()
        psi = np.sqrt(probabilities)
        expected_star = np.zeros(lattice.n_vertices)
        expected_pair = np.zeros(lattice.n_edges)
        transition = np.zeros((256, 256))
        lazy = 0.05
        for i, state in enumerate(states):
            transition[i, i] = lazy
            for vertex in lattice.vertices:
                moved = state.copy()
                for edge in vertex.incident_edges:
                    moved[edge.id] *= -1
                j = lookup[moved.tobytes()]
                expected_star[vertex.id] += psi[i] * psi[j]
                acceptance = min(1.0, probabilities[j] / probabilities[i])
                transition[i, j] += (1 - lazy) * acceptance / lattice.n_vertices
                transition[i, i] += (1 - lazy) * (1 - acceptance) / lattice.n_vertices
                trial = state.copy()
                accepted, stayed = _proposal_kernel(trial, fields, lattice.vertex_edge_indices,
                    lattice.vertex_degrees, np.array([[vertex.id]]),
                    np.array([[[0.5, acceptance / 2]]]), lazy)
                self.assertEqual((accepted, stayed), (1, 0))
                np.testing.assert_array_equal(trial, moved)
                if acceptance < 1 - 1e-12:
                    trial = state.copy()
                    accepted, stayed = _proposal_kernel(trial, fields, lattice.vertex_edge_indices,
                        lattice.vertex_degrees, np.array([[vertex.id]]),
                        np.array([[[0.5, (1 + acceptance) / 2]]]), lazy)
                    self.assertEqual((accepted, stayed), (0, 0))
                    np.testing.assert_array_equal(trial, state)
            for edge in lattice.edges:
                moved = state.copy()
                for endpoint in edge.endpoints:
                    for incident in endpoint.incident_edges:
                        moved[incident.id] *= -1
                expected_pair[edge.id] += psi[i] * psi[lookup[moved.tobytes()]]
        exact = exact_expectations(lattice, fields)
        np.testing.assert_allclose(exact["star"], expected_star, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(exact["pair"], expected_pair, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(exact["z"], probabilities @ states, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(transition.sum(axis=1), 1, rtol=0, atol=1e-15)
        np.testing.assert_allclose(probabilities @ transition, probabilities, rtol=2e-14, atol=1e-16)
        flow = probabilities[:, None] * transition
        np.testing.assert_allclose(flow, flow.T, rtol=2e-14, atol=1e-16)


class CompactGenerationTests(unittest.TestCase):
    def setUp(self):
        root = artifact_root(REPO)
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="test-compact-", dir=root)
        self.base = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def compact_config(self, chunk=17):
        config = tiny_config(chunk)
        config["storage"] = {"state_encoding": "packed_bits", "retain_estimator_records": False,
                             "retain_block_means": False}
        return config

    def test_compact_and_legacy_results_agree_after_resume(self):
        compact = prepare_run(REPO, self.compact_config(), self.base / "compact")
        self.assertEqual(run_generation(REPO, compact, workers=1, interactive=False, max_chunks=2)["state"], "paused")
        checkpoints = list((compact / "checkpoints").glob("*.npz"))
        self.assertEqual(len(checkpoints), 1)
        saved = load_arrays(checkpoints[0])
        self.assertNotIn("states", saved)
        self.assertEqual(saved["states_packed"].shape, (2, 3))
        initial = decode_states(saved)
        self.assertEqual(inspect_run(compact)[2][checkpoints[0].stem]["records"], 2)
        run_generation(REPO, compact, workers=2, interactive=False)
        legacy = prepare_run(REPO, tiny_config(19), self.base / "legacy")
        run_generation(REPO, legacy, workers=1, interactive=False)
        for path in sorted((compact / "instances").glob("*.npz")):
            small, full = load_arrays(path), load_arrays(legacy / "instances" / path.name)
            np.testing.assert_array_equal(decode_states(small), full["states"])
            if path.stem == checkpoints[0].stem:
                np.testing.assert_array_equal(decode_states(small)[:2], initial)
            for key, array in full.items():
                if key == "states" or key.endswith("_values") or key.endswith("_block_means"):
                    self.assertNotIn(key, small)
                    continue
                if key in {"metadata_json", "summary_json"}:
                    left, right = json.loads(small[key].item()), json.loads(array.item())
                    left.pop("compute_seconds")
                    right.pop("compute_seconds")
                    self.assertEqual(left, right, key)
                else:
                    np.testing.assert_array_equal(small[key], array, err_msg=key)
        self.assertTrue(verify_run(compact)["passed"])
        self.assertTrue(verify_run(legacy)["passed"])

    def test_mixed_retention_choices_are_verified(self):
        for keep_records, keep_blocks in ((True, False), (False, True)):
            config = self.compact_config(chunk=80)
            config["storage"].update({"retain_estimator_records": keep_records, "retain_block_means": keep_blocks})
            run = prepare_run(REPO, config, self.base / f"mixed-{keep_records}")
            run_generation(REPO, run, workers=1, interactive=False)
            arrays = load_arrays(next((run / "instances").glob("*.npz")))
            for channel in ("star", "pair", "z"):
                self.assertEqual(channel + "_values" in arrays, keep_records)
                self.assertEqual(channel + "_block_means" in arrays, keep_blocks)
            self.assertTrue(verify_run(run)["passed"])

    def test_corrupt_padding_and_summary_fail_verification(self):
        run = prepare_run(REPO, self.compact_config(chunk=80), self.base / "corrupt")
        run_generation(REPO, run, workers=1, interactive=False)
        path = next((run / "instances").glob("*.npz"))
        arrays = load_arrays(path)
        packed = arrays["states_packed"].copy()
        arrays["states_packed"][0, -1] |= 4
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with self.assertRaisesRegex(ValueError, "padding"):
            verify_run(run)
        arrays["states_packed"] = packed
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        summary = json.loads(path.with_suffix(".json").read_text())
        summary["records"] += 1
        path.with_suffix(".json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "summary differs"):
            verify_run(run)

    def test_invalid_storage_settings_fail_before_creation(self):
        for mutation in ({"state_encoding": "uint8"}, {"retain_block_means": 0},
                         {"retain_estimator_records": "false"}, {"unexpected": True}):
            config = self.compact_config()
            config["storage"].update(mutation)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_packed_checkpoint_fields_counts_and_padding_are_validated(self):
        config = self.compact_config(chunk=34)
        run = prepare_run(REPO, config, self.base / "checkpoint-guards")
        instance = json.loads((run / "manifest.json").read_text())["instances"][0]
        _advance_job(str(run), instance, config)
        path = run / "checkpoints" / (instance["id"] + ".npz")
        original = load_arrays(path)
        for corruption in ("padding", "field", "counter", "record_count"):
            arrays = {key: value.copy() for key, value in original.items()}
            meta = json.loads(arrays["metadata_json"].item())
            if corruption == "padding":
                arrays["states_packed"][0, -1] |= 4
            elif corruption == "field":
                arrays["b_z"][0] += 0.01
            elif corruption == "counter":
                meta["chain"]["attempted"] += 9
            else:
                meta["retained_records"] += 1
            arrays["metadata_json"] = np.array(json.dumps(meta))
            with path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            with self.subTest(corruption=corruption):
                with self.assertRaises(ValueError):
                    _advance_job(str(run), instance, config)

    def test_progress_inspection_uses_only_small_metadata(self):
        config = self.compact_config(chunk=34)
        run = prepare_run(REPO, config, self.base / "light-inspection")
        instances = json.loads((run / "manifest.json").read_text())["instances"]
        for _ in range(3):
            _advance_job(str(run), instances[0], config)
        _advance_job(str(run), instances[1], config)
        with patch("quantum_lattice.data.generation._load_npz", side_effect=AssertionError("eager NPZ read")):
            _, _, progress = inspect_run(run)
        self.assertTrue(progress[instances[0]["id"]]["done"])
        self.assertEqual(progress[instances[1]["id"]]["records"], 2)
        self.assertIs(_lattice(3), _lattice(3))

    def test_verification_reports_progress_and_can_stop_without_modifying_instances(self):
        config = self.compact_config(chunk=80)
        run = prepare_run(REPO, config, self.base / "verification-progress")
        run_generation(REPO, run, workers=1, interactive=False)
        updates = []
        self.assertTrue(verify_run(run, lambda *args: updates.append(args))["passed"])
        self.assertEqual([(done, total) for done, total, _ in updates], [(i, 4) for i in range(5)])
        paths = sorted((run / "instances").glob("*.npz"))
        before = {path: path.stat().st_mtime_ns for path in paths}
        def interrupted(done, total, instance_id):
            if done == 1:
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            verify_run(run, interrupted)
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in paths})
        self.assertTrue(verify_run(run)["passed"])

    def test_interrupted_final_verification_resumes_without_sampling_again(self):
        run = prepare_run(REPO, self.compact_config(chunk=80), self.base / "verify-later")
        with patch("quantum_lattice.data.generation.verify_run", side_effect=KeyboardInterrupt):
            result = run_generation(REPO, run, workers=1, interactive=False)
        self.assertEqual(result["state"], "verification_pending")
        self.assertEqual(result["completed"], 4)
        self.assertFalse((run / "dataset.json").exists())
        before = {path: path.stat().st_mtime_ns for path in (run / "instances").glob("*.npz")}
        resumed = run_generation(REPO, run, workers=2, interactive=False)
        self.assertEqual(resumed["state"], "complete")
        self.assertEqual(resumed["attempts_this_session"], 0)
        self.assertTrue((run / "dataset.json").exists())
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in (run / "instances").glob("*.npz")})


if __name__ == "__main__":
    unittest.main()
