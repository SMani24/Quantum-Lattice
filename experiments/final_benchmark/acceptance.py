#!/usr/bin/env python3
"""Bounded actual-worker acceptance and disposable steady GPU timing.

The small cache is development-only. Cyclic repetitions in the timing section
reproduce residency and batch shapes; they are not independent scientific data.
The tiny sampler cohorts are disposable API fixtures, never primary confirmation.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quantum_lattice.data.final_confirmation import (
    advance_confirmation_instance, load_confirmation_features,
    load_confirmation_plan, prepare_confirmation)
from quantum_lattice.data.final_features import load_features
from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.models.final_training import load_job, predict_checkpoint, train_job
from quantum_lattice.models.final_worker import HardwareProbe, _guard, _settings
from quantum_lattice.models.observable_training import ObservableTrainSession


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _tree_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def _settings_for(model, *, tiny=False):
    return {"model": model, "architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 3},
            "seed": 2026102898, "device": "cuda", "deterministic": True,
            "torch_threads": 1, "learning_rate": 1e-4,
            "batch_size_hamiltonians": 2 if tiny else 32,
            "max_epochs": 3 if tiny else 500,
            "validation_batch_size": 2 if tiny else 256,
            "early_stopping": {"enabled": not tiny, "min_epochs": 50,
                               "patience_epochs": 30, "min_delta": 1e-6}}


def _backend_flags():
    return (torch.are_deterministic_algorithms_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)


class Health:
    def __init__(self, gpu_uuid, started, snapshots):
        self.probe = HardwareProbe()
        self.settings = _settings({"gpu_uuid": gpu_uuid, "max_cpu_c": 75, "max_gpu_c": 78})
        self.started, self.snapshots = started, snapshots
        self.cursor, self.boot = None, None

    def check(self):
        if time.monotonic() - self.started > 1200:
            raise TimeoutError("Acceptance exceeded its twenty-minute hard allowance")
        snapshot = self.probe.snapshot(self.cursor)
        if self.boot is None:
            self.boot = snapshot["boot_id"]
        reason, severe = _guard(snapshot, self.settings, self.boot, os.getpgrp())
        # The acceptance process needs no other CUDA worker, even in its group.
        if any(p["pid"] != os.getpid() for p in snapshot["compute_processes"]):
            reason = "foreign_compute_process"
        if reason:
            raise RuntimeError(f"Acceptance hardware guard: {reason}; severe={severe}")
        self.cursor = snapshot["kernel_cursor"]
        self.snapshots.append(snapshot)
        return snapshot


def _small_arrays(features):
    if (features["manifest"].get("test_payload_loaded") is not False
            or len(features["train"]["y"]) < 8 or len(features["validation"]["y"]) < 4):
        raise ValueError("Acceptance requires at least eight train/four validation development H")
    return tuple(np.array(features[split][key][:count], copy=True)
                 for split, key, count in [("train", "g", 8), ("train", "y", 8),
                                           ("validation", "g", 4), ("validation", "y", 4)])


def _resume_case(output, features, model, health):
    L = features["L"]
    arrays = _small_arrays(features)
    settings = _settings_for(model, tiny=True)
    bindings = {"role": "disposable_acceptance", "L": L, "model": model,
                "cache_manifest_sha256": hashlib.sha256(json.dumps(features["manifest"], sort_keys=True).encode()).hexdigest()}
    direct = train_job(*arrays, settings, output/f"L{L}-{model}-direct", bindings,
                       max_updates=7, checkpoint_updates=2)
    health.check()
    paused = train_job(*arrays, settings, output/f"L{L}-{model}-resumed", bindings,
                       max_updates=3, checkpoint_updates=2)
    if (paused["state"], paused["step"]) != ("paused", 3):
        raise AssertionError("CUDA acceptance failed to pause at the requested update")
    resumed = train_job(*arrays, settings, output/f"L{L}-{model}-resumed", bindings,
                        max_updates=4, checkpoint_updates=2)
    left = torch.load(direct["checkpoint_paths"]["latest"], map_location="cpu", weights_only=True)["session"]
    right = torch.load(resumed["checkpoint_paths"]["latest"], map_location="cpu", weights_only=True)["session"]
    if not _tree_equal(left, right) or resumed["step"] != 7:
        raise AssertionError("CUDA resumed state differs from uninterrupted state")
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
    flags = _backend_flags()
    with mock.patch("torch.optim.Adam", side_effect=AssertionError("Inference must not construct Adam")):
        a = predict_checkpoint(direct["checkpoint_paths"]["best"], arrays[2], batch_size=2)
        b = predict_checkpoint(resumed["checkpoint_paths"]["best"], arrays[2], batch_size=2)
    if (not np.array_equal(a, b) or not np.isfinite(a).all()
            or not torch.equal(cpu_rng, torch.get_rng_state())
            or not all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all()))
            or flags != _backend_flags()):
        raise AssertionError("Selected inference changed RNG/backend state or predictions")
    verified = load_job(output/f"L{L}-{model}-resumed")
    if verified["step"] != 7 or verified["commit_sha256"] != resumed["commit_sha256"]:
        raise AssertionError("CUDA committed hash/progress verification failed")
    health.check()
    result = {"L": L, "model": model, "direct_updates": 7, "resume_updates": [3, 4],
              "session_bitwise_equal": True, "selected_predictions_bitwise_equal": True,
              "inference_constructs_no_optimizer": True, "rng_and_backend_flags_preserved": True,
              "committed_hashes_verified": True, "all_passed": True}
    del left, right, arrays, a, b
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _steady_case(features, model, duration, health):
    """Time cyclic-residency discarded work; never present its losses as results."""
    real_train, real_val = features["train"], features["validation"]
    ti = np.arange(7452) % len(real_train["y"])
    vi = np.arange(1242) % len(real_val["y"])
    arrays = (real_train["g"][ti], real_train["y"][ti],
              real_val["g"][vi], real_val["y"][vi])
    health.check()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.monotonic()
    session = ObservableTrainSession(*arrays, _settings_for(model))
    torch.cuda.synchronize()
    initialization = time.monotonic() - begin
    session.advance(5)
    torch.cuda.synchronize()
    first_step, first_epoch = session.step, session.epoch
    begin = time.monotonic()
    next_health = begin + 5
    updates = 0
    while time.monotonic() - begin < duration:
        session.advance(5)
        torch.cuda.synchronize()
        updates += 5
        if time.monotonic() >= next_health:
            health.check()
            next_health = time.monotonic() + 5
    torch.cuda.synchronize()
    elapsed = time.monotonic() - begin
    prediction_start = time.monotonic()
    prediction = session.selected_predictions(arrays[2])
    torch.cuda.synchronize()
    forward = time.monotonic() - prediction_start
    if prediction.shape != arrays[3].shape or not np.isfinite(prediction).all():
        raise AssertionError("Steady-case selected CUDA forward failed")
    result = {"L": 24, "model": model, "minimum_measured_seconds": duration,
              "measured_wall_seconds": elapsed, "actual_updates": session.step-first_step,
              "completed_epochs_during_measurement": session.epoch-first_epoch,
              "seconds_per_actual_update": elapsed/(session.step-first_step),
              "initialization_and_initial_validation_seconds": initialization,
              "selected_validation_forward_seconds": forward,
              "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
              "resident_train_H": 7452, "resident_validation_H": 1242,
              "independent_development_train_H": len(real_train["y"]),
              "independent_development_validation_H": len(real_val["y"]),
              "cyclic_duplicates_for_timing_only": True,
              "timing_includes": "synchronized optimizer updates, any completed-epoch validation and periodic health probes",
              "timing_excludes": "warmup, constructor, final selected forward, checkpoint I/O",
              "no_accuracy_or_convergence_claim": True}
    health.check()
    del session, prediction, arrays
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _sampler_cases(output, health):
    sampling = {"total_sweeps": 128, "burn_in_sweeps": 16, "thinning_sweeps": 4,
                "chunk_sweeps": 32, "block_size_records": 7,
                "lazy_probability": .05, "backend": "numba", "exact_small_sizes": False}
    bindings = {"purpose": "disposable acceptance fixture; not the primary confirmation cohort"}
    freeze = hashlib.sha256(b"disposable-worker-acceptance-2026102898").hexdigest()
    runs = {}
    for label, chunk in [("resumed", 32), ("direct", 128)]:
        run = prepare_confirmation(ROOT, output/f"sampler-{label}", [4, 8, 16, 24], 1,
            2026102898, {**sampling, "chunk_sweeps": chunk}, [], freeze, bindings=bindings)
        plan = load_confirmation_plan(ROOT, run)
        for instance in plan["instances"]:
            if label == "resumed":
                paused = advance_confirmation_instance(ROOT, run, instance["id"], max_chunks=1)
                if paused["done"] or paused["sweeps"] != 32:
                    raise AssertionError("Tiny sampler did not stop at its chunk boundary")
            result = advance_confirmation_instance(ROOT, run, instance["id"], max_chunks=8)
            if not result["done"] or result["records"] != 28:
                raise AssertionError("Tiny sampler failed to resume/complete")
            health.check()
        runs[label] = run
    result = []
    for L in [4, 8, 16, 24]:
        a = load_confirmation_features(runs["resumed"], L)
        b = load_confirmation_features(runs["direct"], L)
        for key in ["x_f", "x_g", "y", "fields", "hamiltonian_uids"]:
            if not np.array_equal(a[key], b[key]):
                raise AssertionError(f"Sampler pause/resume differs: L{L}/{key}")
        name = str(a["hamiltonian_ids"][0])
        left = decode_states(_archive(runs["resumed"] / "instances" / name / "records.npz"))
        right = decode_states(_archive(runs["direct"] / "instances" / name / "records.npz"))
        if not np.array_equal(left, right):
            raise AssertionError("Packed sampling records differ across chunk boundaries")
        result.append({"L": L, "records": 28, "total_sweeps": 128, "backend": "numba",
                       "pause_resume_records_bitwise_equal": True, "paired_features_bitwise_equal": True,
                       "scope": "tiny disposable portability fixture; no mixing or accuracy evidence"})
    return result


def run_acceptance(cache_root, output, gpu_uuid, *, steady_seconds=30):
    if type(steady_seconds) not in (int, float) or not 30 <= steady_seconds <= 60:
        raise ValueError("Steady timing must remain bounded to30–60seconds per model")
    output = Path(output).resolve()
    allowed = (ROOT / "codex material/artifacts/final_benchmark").resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Acceptance output must be inside ignored final_benchmark artifacts")
    if output.exists():
        raise FileExistsError("Use a fresh acceptance directory; never overwrite evidence")
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema_version": 1, "role": "disposable_worker_acceptance",
              "source_sha256": _sha(__file__), "gpu_uuid": gpu_uuid,
              "scope": "development-only acceptance and cyclic-residency timing; no primary cohort or scientific training",
              "hardware_snapshots": [], "cuda_resume": [], "steady_timing": [],
              "sampler_resume": [], "complete": False, "all_passed": False}
    health = Health(gpu_uuid, started, report["hardware_snapshots"])
    try:
        health.check()
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA acceptance is required; CPU fallback is refused")
        torch.set_num_threads(1)
        for L in [4, 8, 16, 24]:
            features = load_features(cache_root, L)
            _small_arrays(features)
            if L in [4, 24]:
                for model in ["cnn", "map_mlp"]:
                    row = _resume_case(output, features, model, health)
                    report["cuda_resume"].append(row)
                    print(json.dumps(row), flush=True)
                    _json(output / "report.progress.json", report)
            if L == 24:
                for model in ["cnn", "map_mlp"]:
                    row = _steady_case(features, model, steady_seconds, health)
                    report["steady_timing"].append(row)
                    print(json.dumps(row), flush=True)
                    _json(output / "report.progress.json", report)
            del features
            gc.collect()
        report["sampler_resume"] = _sampler_cases(output, health)
        health.check()
        report.update(complete=True, all_passed=True)
        return report
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["active_wall_seconds"] = time.monotonic() - started
        _json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Verified development-only timing cache")
    parser.add_argument("--output", required=True, help="Fresh ignored output directory")
    parser.add_argument("--gpu-uuid", required=True, help="Pinned actual-worker NVIDIA UUID")
    parser.add_argument("--steady-seconds", type=float, default=30)
    args = parser.parse_args()
    run_acceptance(args.cache, args.output, args.gpu_uuid, steady_seconds=args.steady_seconds)
