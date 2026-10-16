"""Tiny independent checks of crash-safe estimator-validation chains."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.simulation.estimator_validation import advance_chain, load_chain_records
from quantum_lattice.simulation.solvable import StarMetropolisChain, _in_reference_orbit


def specification(**updates):
    spec = {"seed": [20261016, 723], "start": "plus", "backend": "python",
            "lazy_probability": 0.05, "burn_in_sweeps": 11, "total_sweeps": 71,
            "thinning_sweeps": 5, "chunk_sweeps": 7}
    spec.update(updates)
    return spec


class EstimatorValidationChainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.lattice = square_lattice(3)
        self.fields = np.linspace(-0.6, 0.4, self.lattice.n_edges)

    def tearDown(self):
        self.temporary.cleanup()

    def progress(self, job):
        return json.loads((job / "progress.json").read_text())

    def finish(self, job, spec):
        for _ in range(100):
            if advance_chain(job, self.lattice, self.fields, spec):
                return
        self.fail("tiny validation chain did not finish")

    def test_chunk_resume_matches_independent_uninterrupted_chain_bit_for_bit(self):
        job, spec = self.root / "resumed", specification()
        self.assertFalse(advance_chain(job, self.lattice, self.fields, spec))
        first = self.progress(job)
        self.assertEqual(first["completed_sweeps"], 7)
        self.assertEqual(first["retained_records"], 0)
        self.assertEqual(load_chain_records(job, self.lattice).shape, (0, self.lattice.n_edges))
        self.finish(job, spec)

        direct = StarMetropolisChain(self.lattice, self.fields, spec["seed"],
                                     backend="python", lazy_probability=0.05)
        direct.advance(spec["burn_in_sweeps"])
        expected = []
        for _ in range(12):
            direct.advance(spec["thinning_sweeps"])
            expected.append(direct.spins.copy())
        np.testing.assert_array_equal(load_chain_records(job, self.lattice), expected)
        self.assertEqual(self.progress(job)["chain"], direct.snapshot())
        before = {name: (job / name).stat().st_mtime_ns for name in ("progress.json", "states_packed.npy")}
        self.assertTrue(advance_chain(job, self.lattice, self.fields, spec))
        self.assertEqual(before, {name: (job / name).stat().st_mtime_ns for name in before})

    def test_record_tail_written_before_failed_commit_is_ignored_and_overwritten(self):
        job, spec = self.root / "crash", specification(chunk_sweeps=17)
        self.assertFalse(advance_chain(job, self.lattice, self.fields, spec))
        committed = load_chain_records(job, self.lattice)
        old_progress = self.progress(job)
        with patch("quantum_lattice.simulation.estimator_validation._atomic_json", side_effect=OSError("simulated crash before commit")):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                advance_chain(job, self.lattice, self.fields, spec)
        self.assertEqual(self.progress(job), old_progress)
        np.testing.assert_array_equal(load_chain_records(job, self.lattice), committed)
        # Deliberately poison every uncommitted byte, including padding bits.
        memory = np.lib.format.open_memmap(job / "states_packed.npy", mode="r+")
        memory[len(committed):] = 255
        memory.flush()
        del memory
        np.testing.assert_array_equal(load_chain_records(job, self.lattice), committed)
        self.finish(job, spec)
        other = self.root / "uninterrupted"
        self.finish(other, spec)
        np.testing.assert_array_equal(load_chain_records(job, self.lattice), load_chain_records(other, self.lattice))
        self.assertEqual(self.progress(job)["chain"], self.progress(other)["chain"])

    def test_random_start_uses_valid_orbit_and_a_separate_initialization_rng(self):
        spec, job = specification(start="random", chunk_sweeps=1), self.root / "random"
        advance_chain(job, self.lattice, self.fields, spec)
        initial_rng = np.random.default_rng(np.random.SeedSequence(spec["seed"], spawn_key=(20261016, 119)))
        theta = initial_rng.choice(np.array([-1, 1], np.int8), size=self.lattice.n_vertices)
        ends = self.lattice.edge_vertex_indices
        initial = theta[ends[:, 0]] * theta[ends[:, 1]]
        self.assertTrue(_in_reference_orbit(self.lattice, initial))
        self.assertFalse(np.all(initial == 1))
        direct = StarMetropolisChain(self.lattice, self.fields, spec["seed"], backend="python", lazy_probability=.05)
        direct.spins[:] = initial
        direct.advance(1)
        self.assertEqual(self.progress(job)["chain"], direct.snapshot())

    def test_stationary_zero_field_start_matches_exact_categorical_draw(self):
        from quantum_lattice.observables.conditioned import exact_reference

        fields = np.zeros(self.lattice.n_edges)
        spec, job = specification(start="stationary", chunk_sweeps=1, seed=97), self.root / "stationary"
        reference = exact_reference(self.lattice, fields, max_vertices=16)
        np.testing.assert_allclose(reference["probabilities"], 1 / len(reference["probabilities"]))
        initial_rng = np.random.default_rng(np.random.SeedSequence(spec["seed"], spawn_key=(20261016, 119)))
        index = initial_rng.choice(len(reference["probabilities"]), p=reference["probabilities"])
        direct = StarMetropolisChain(self.lattice, fields, spec["seed"], backend="python", lazy_probability=.05)
        direct.spins[:] = reference["states"][index]
        direct.advance(1)
        self.assertFalse(advance_chain(job, self.lattice, fields, spec))
        self.assertEqual(self.progress(job)["chain"], direct.snapshot())

    def test_invalid_specs_and_changed_immutable_inputs_are_rejected(self):
        bad_updates = [
            {"burn_in_sweeps": 71}, {"total_sweeps": 72}, {"thinning_sweeps": 0},
            {"chunk_sweeps": 0}, {"burn_in_sweeps": True}, {"seed": [-1, 0]},
            {"seed": []}, {"seed": True}, {"start": "edges"},
            {"backend": "gpu"}, {"lazy_probability": float("nan")},
            {"lazy_probability": 1},
        ]
        for number, updates in enumerate(bad_updates):
            with self.subTest(updates=updates):
                with self.assertRaises(ValueError):
                    advance_chain(self.root / f"bad-{number}", self.lattice, self.fields, specification(**updates))
        job, spec = self.root / "immutable", specification()
        advance_chain(job, self.lattice, self.fields, spec)
        for updates in ({"seed": 42}, {"chunk_sweeps": 8}, {"start": "random"}):
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, "spec differs"):
                    advance_chain(job, self.lattice, self.fields, specification(**updates))
        with self.assertRaisesRegex(ValueError, "fields differ"):
            advance_chain(job, self.lattice, self.fields + .01, spec)

    def test_corrupted_schedule_counters_or_committed_padding_are_rejected(self):
        job, spec = self.root / "corrupted", specification(chunk_sweeps=17)
        advance_chain(job, self.lattice, self.fields, spec)
        path, original = job / "progress.json", self.progress(job)
        for corruption in ("records", "attempts", "complete", "numpy"):
            with self.subTest(corruption=corruption):
                bad = deepcopy(original)
                if corruption == "records":
                    bad["retained_records"] += 1
                elif corruption == "attempts":
                    bad["chain"]["attempted"] += self.lattice.n_vertices
                elif corruption == "complete":
                    bad["complete"] = True
                else:
                    bad["numpy_version"] = "0.0.0"
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    advance_chain(job, self.lattice, self.fields, spec)
        path.write_text(json.dumps(original))
        memory = np.lib.format.open_memmap(job / "states_packed.npy", mode="r+")
        memory[0, -1] |= np.uint8(128)
        memory.flush()
        del memory
        with self.assertRaisesRegex(ValueError, "padding"):
            load_chain_records(job, self.lattice)


if __name__ == "__main__":
    unittest.main()
