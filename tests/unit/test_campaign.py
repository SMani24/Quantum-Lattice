"""Campaign selection, provenance and storage planning without long sampling."""

from contextlib import redirect_stdout
from copy import deepcopy
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from quantum_lattice.data.campaign import (
    _benchmark, _digest, _ensure_run, _verify_with_progress, build_profiles, campaign_root, campaign_status,
    dashboard, estimate_storage_bytes,
)
from quantum_lattice.data.generation import _bank


BASE_CONFIG = {
    "schema_version": 1,
    "lattice": {"type": "square_periodic", "L": 4},
    "fields": {"channel": "z_only", "distribution": "uniform", "low": -1.7, "high": 1.7},
    "dataset": {"hamiltonians": 16, "seed": 20261013,
                "split_fractions": {"train": 0.75, "validation": 0.125, "test": 0.125}},
    "sampler": {"backend": "auto", "total_sweeps": 100000, "burn_in_sweeps": 10000,
                "thinning_sweeps": 20, "lazy_probability": 0.05},
    "statistics": {"block_size_records": 45},
    "execution": {"workers": 4, "chunk_sweeps": 1000},
    "storage": {"state_encoding": "packed_bits", "retain_estimator_records": False,
                "retain_block_means": False},
}


def specification(minimum=7450):
    return {"schema_version": 1, "campaign_name": "test_campaign",
            "lattice_sizes": [3, 4, 8, 12, 16, 20, 24],
            "training_hamiltonians_minimum": minimum, "seed": 20261013,
            "base_config": deepcopy(BASE_CONFIG)}


def fake_prepare(repo, config, path, purpose):
    path = Path(path)
    path.mkdir(parents=True)
    (path / "instances").mkdir()
    (path / "checkpoints").mkdir()
    (path / "config.json").write_text(json.dumps(config))
    (path / "manifest.json").write_text(json.dumps({"config_sha256": _digest(config)}))
    (path / "status.json").write_text(json.dumps({"state": "prepared", "completed": 0}))
    return path


class CampaignTests(unittest.TestCase):
    def test_full_profiles_are_balanced_independent_and_do_not_mutate_spec(self):
        spec = specification()
        original = deepcopy(spec)
        profiles = build_profiles(spec)
        self.assertEqual(list(profiles), [3, 4, 8, 12, 16, 20, 24])
        self.assertEqual(spec, original)
        self.assertEqual({p["dataset"]["hamiltonians"] for p in profiles.values()}, {9936})
        self.assertEqual(len({p["dataset"]["seed"] for p in profiles.values()}), 7)
        bank = _bank(profiles[4])
        self.assertEqual({name: sum(row["split"] == name for row in bank)
                          for name in ("train", "validation", "test")},
                         {"train": 7452, "validation": 1242, "test": 1242})
        for side, profile in profiles.items():
            self.assertEqual(profile["lattice"]["L"], side)
            expected = int(np.random.SeedSequence([20261013, side, 42]).generate_state(1)[0])
            self.assertEqual(profile["dataset"]["seed"], expected)
        fields3 = np.random.default_rng(profiles[3]["dataset"]["seed"]).uniform(-1.7, 1.7, 18)
        fields4 = np.random.default_rng(profiles[4]["dataset"]["seed"]).uniform(-1.7, 1.7, 32)
        self.assertFalse(np.array_equal(fields3, fields4[:18]))

    def test_invalid_campaign_fields_rejected(self):
        for key, value in (("campaign_name", "../escape"), ("campaign_name", ""),
                           ("seed", True), ("seed", -1),
                           ("training_hamiltonians_minimum", 0),
                           ("training_hamiltonians_minimum", True),
                           ("lattice_sizes", [3, 3]), ("lattice_sizes", []),
                           ("lattice_sizes", [True]), ("lattice_sizes", [2])):
            with self.subTest(key=key, value=value):
                spec = specification()
                spec[key] = value
                with self.assertRaises(ValueError):
                    build_profiles(spec)

    def test_numeric_storage_estimate_respects_compact_options(self):
        profile = build_profiles(specification(2))[4]
        # Eight Hamiltonians: packed states (4500x4) plus scientific summaries.
        expected_per_instance = 4500 * 4 + 32 * 64 + 3 * (16 + 64) * 8
        self.assertEqual(estimate_storage_bytes(profile), expected_per_instance * 8)
        original = deepcopy(profile)
        profile["storage"] = {"state_encoding": "int8", "retain_estimator_records": True,
                              "retain_block_means": True}
        expected_full = 4500 * 32 + 32 * 64 + 3 * 80 * 8 + (4500 + 100) * 80 * 8
        self.assertEqual(estimate_storage_bytes(profile), expected_full * 8)
        self.assertGreater(estimate_storage_bytes(profile), estimate_storage_bytes(original) * 100)

    def test_status_before_start_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            spec = specification(2)
            spec_path = repo / "campaign.json"
            spec_path.write_text(json.dumps(spec))
            with patch("numpy.load", side_effect=AssertionError("must not load NPZ")):
                status = campaign_status(repo, spec_path)
            self.assertEqual({row["state"] for row in status["profiles"]}, {"not started"})
            self.assertEqual({row["completed"] for row in status["profiles"]}, {0})
            self.assertFalse(campaign_root(repo, spec).exists())

    def test_saved_run_is_reused_and_progress_uses_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            spec = specification(2)
            profile = build_profiles(spec)[4]
            spec_path = repo / "campaign.json"
            spec_path.write_text(json.dumps(spec))
            with patch("quantum_lattice.data.campaign.prepare_run", side_effect=fake_prepare) as prepare:
                run = _ensure_run(repo, spec, 4, profile)
                self.assertEqual(_ensure_run(repo, spec, 4, profile), run)
                self.assertEqual(prepare.call_count, 1)
            for index in range(3):
                (run / "instances" / f"hamiltonian_{index:05d}.json").write_text("{}")
                (run / "instances" / f"hamiltonian_{index:05d}.npz").touch()
            (run / "status.json").write_text(json.dumps({"state": "paused", "completed": 3}))
            with patch("numpy.load", side_effect=AssertionError("must not load NPZ")):
                status = campaign_status(repo, spec_path)
            row = next(row for row in status["profiles"] if row["L"] == 4)
            self.assertEqual((row["completed"], row["total"], row["state"]), (3, 8, "paused"))
            self.assertEqual(row["run_dir"], str(run))
            spec["base_config"]["fields"]["high"] = 0.5
            spec_path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "new campaign_name"):
                campaign_status(repo, spec_path)

    def test_changed_saved_profile_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = specification(2)
            repo = Path(directory)
            profile = build_profiles(spec)[4]
            with patch("quantum_lattice.data.campaign.prepare_run", side_effect=fake_prepare):
                run = _ensure_run(repo, spec, 4, profile)
            changed = deepcopy(profile)
            changed["dataset"]["seed"] += 1
            (run / "config.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "saved settings"):
                _ensure_run(repo, spec, 4, profile)

    def test_index_recovered_after_interrupted_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, spec = Path(directory), specification(2)
            profile = build_profiles(spec)[4]
            run = campaign_root(repo, spec) / "L4" / "run"
            fake_prepare(repo, profile, run, "campaign-L4")
            with patch("quantum_lattice.data.campaign.prepare_run") as prepare:
                self.assertEqual(_ensure_run(repo, spec, 4, profile), run)
                prepare.assert_not_called()
            index = json.loads((campaign_root(repo, spec) / "campaign-index.json").read_text())
            self.assertEqual(index["runs"]["4"]["run_dir"], "L4/run")

    def test_campaign_lock_prevents_duplicate_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, spec = Path(directory), specification(2)
            root = campaign_root(repo, spec)
            root.mkdir(parents=True)
            with (root / ".campaign.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch("quantum_lattice.data.campaign.prepare_run") as prepare:
                    with self.assertRaisesRegex(RuntimeError, "another process"):
                        _ensure_run(repo, spec, 4, build_profiles(spec)[4])
                    prepare.assert_not_called()

    def test_dashboard_only_launches_selected_size(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, spec = Path(directory), specification(2)
            spec_path = repo / "campaign.json"
            spec_path.write_text(json.dumps(spec))
            result = {"state": "paused", "completed": 0, "total": 8}
            with patch("quantum_lattice.data.campaign.prepare_run", side_effect=fake_prepare), \
                 patch("quantum_lattice.data.campaign.run_generation", return_value=result) as run, \
                 patch("builtins.input", side_effect=["2", "1", "b", "q"]), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(dashboard(repo, spec_path, workers_override=2), 0)
            self.assertEqual(run.call_count, 1)
            self.assertIn("L4/run", str(run.call_args.args[1]))
            self.assertEqual(run.call_args.kwargs["workers"], 2)
            self.assertFalse((campaign_root(repo, spec) / "L3").exists())

    def test_dashboard_worker_selection_changes_execution_only(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, spec = Path(directory), specification(2)
            spec_path = repo / "campaign.json"
            spec_path.write_text(json.dumps(spec))
            result = {"state": "paused", "completed": 0, "total": 8}
            with patch("quantum_lattice.data.campaign.prepare_run", side_effect=fake_prepare), \
                 patch("quantum_lattice.data.campaign.run_generation", return_value=result) as run, \
                 patch("builtins.input", side_effect=["2", "w", "8", "1", "q"]), \
                 redirect_stdout(io.StringIO()):
                dashboard(repo, spec_path)
            self.assertEqual(run.call_args.kwargs["workers"], 8)
            saved = json.loads((Path(run.call_args.args[1]) / "config.json").read_text())
            self.assertEqual(saved["execution"]["workers"], 4)
            self.assertEqual(saved, build_profiles(spec)[4])

    def test_benchmark_is_separate_from_full_campaign_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, spec = Path(directory), specification(2)
            spec_path = repo / "campaign.json"
            spec_path.write_text(json.dumps(spec))
            profile = build_profiles(spec)[4]
            benchmark_path = repo / "short_benchmark"
            benchmark_path.mkdir()
            result = {"state": "complete", "completed": 4, "total": 4,
                      "workers": 2, "elapsed_seconds": 1.0, "verification_seconds": 0.1,
                      "attempts_per_second_including_startup_io": 1000.0}
            progress = {f"hamiltonian_{i:05d}": {"compute_seconds": 0.05, "attempted": 80000}
                        for i in range(4)}
            with patch("quantum_lattice.data.campaign.prepare_run", return_value=benchmark_path) as prepare, \
                 patch("quantum_lattice.data.campaign.run_generation", return_value=result), \
                 patch("quantum_lattice.data.campaign.inspect_run", return_value=(profile, {}, progress)), \
                 redirect_stdout(io.StringIO()):
                _benchmark(repo, profile, 4, 2, False)
            short = prepare.call_args.args[1]
            self.assertEqual(short["dataset"]["hamiltonians"], 4)
            self.assertEqual(short["sampler"]["total_sweeps"], 5000)
            self.assertEqual(short["execution"]["workers"], 2)
            self.assertEqual(profile, build_profiles(spec)[4])
            self.assertTrue((benchmark_path / "benchmark.json").exists())
            timing = json.loads((benchmark_path / "benchmark.json").read_text())
            self.assertAlmostEqual(timing["estimated_dataset_seconds"], 4.9)
            self.assertTrue(timing["verification_seconds_excluded_from_estimate"])
            self.assertEqual({row["state"] for row in campaign_status(repo, spec_path)["profiles"]}, {"not started"})

    def test_pending_verification_finalizes_dataset_via_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, spec = Path(directory), specification(2)
            spec_path = repo / "campaign.json"
            spec_path.write_text(json.dumps(spec))
            profile = build_profiles(spec)[4]
            with patch("quantum_lattice.data.campaign.prepare_run", side_effect=fake_prepare):
                run = _ensure_run(repo, spec, 4, profile)
            for index in range(8):
                (run / "instances" / f"hamiltonian_{index:05d}.json").write_text("{}")
                (run / "instances" / f"hamiltonian_{index:05d}.npz").touch()
            (run / "status.json").write_text(json.dumps({"state": "verification_pending", "completed": 8}))
            result = {"state": "complete", "completed": 8, "total": 8}
            with patch("quantum_lattice.data.campaign.run_generation", return_value=result) as generation, \
                 patch("quantum_lattice.data.campaign.verify_run") as verify, \
                 patch("builtins.input", side_effect=["2", "4", "q"]), \
                 redirect_stdout(io.StringIO()):
                dashboard(repo, spec_path)
            generation.assert_called_once()
            self.assertEqual(generation.call_args.args[1], run)
            verify.assert_not_called()

    def test_manual_verification_failure_marks_only_started_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            status_path = run / "status.json"
            status_path.write_text(json.dumps({"state": "complete", "completed": 8}))
            with patch("quantum_lattice.data.campaign.verify_run", side_effect=ValueError("bad header")), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    _verify_with_progress(run)
            self.assertEqual(json.loads(status_path.read_text())["state"], "complete")

            def fail_after_start(path, progress_callback):
                progress_callback(0, 8, None)
                raise ValueError("bad sample")

            with patch("quantum_lattice.data.campaign.verify_run", side_effect=fail_after_start), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    _verify_with_progress(run)
            self.assertEqual(json.loads(status_path.read_text())["state"], "verification_failed")

    def test_completed_dataset_is_never_automatically_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, spec = Path(directory), specification(2)
            spec_path = repo / "campaign.json"
            spec_path.write_text(json.dumps(spec))
            profile = build_profiles(spec)[4]
            with patch("quantum_lattice.data.campaign.prepare_run", side_effect=fake_prepare):
                run = _ensure_run(repo, spec, 4, profile)
            for index in range(8):
                (run / "instances" / f"hamiltonian_{index:05d}.json").write_text("{}")
                (run / "instances" / f"hamiltonian_{index:05d}.npz").touch()
            (run / "status.json").write_text(json.dumps({"state": "complete", "completed": 8}))
            (run / "dataset.json").write_text("{}")
            with patch("quantum_lattice.data.campaign.run_generation") as generation, \
                 patch("builtins.input", side_effect=["2", "1", "q"]), \
                 redirect_stdout(io.StringIO()) as output:
                dashboard(repo, spec_path)
            generation.assert_not_called()
            self.assertIn("already complete", output.getvalue())
            (run / "dataset.json").unlink()
            status = campaign_status(repo, spec_path)
            row = next(row for row in status["profiles"] if row["L"] == 4)
            self.assertEqual(row["state"], "inconsistent completion")


if __name__ == "__main__":
    unittest.main()
