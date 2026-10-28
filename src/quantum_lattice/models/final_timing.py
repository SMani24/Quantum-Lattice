"""Disposable development-only timing, with no model selection or test access."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from quantum_lattice.data.final_features import load_features
from quantum_lattice.lattice import square_lattice
from quantum_lattice.models.observable_training import ObservableTrainSession
from quantum_lattice.simulation.solvable import StarMetropolisChain


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hardware():
    online = Path("/sys/devices/system/cpu/online").read_text().strip()
    if online != "0-1,3-7,9-11" or Path("/sys/devices/system/cpu/cpufreq/boost").read_text().strip() != "0":
        raise RuntimeError("PC isolation/boost differs from the qualified workaround")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,temperature.gpu,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=10)
    uuid, temperature, memory = result.stdout.strip().split(", ")
    if float(temperature) >= 78:
        raise RuntimeError("GPU too warm for the bounded timing probe")
    return {"online": online, "boost": 0, "gpu_uuid": uuid,
            "gpu_temperature_C": float(temperature), "gpu_memory_MiB": float(memory)}


def run_timing(cache_root, output, *, sizes=(4, 8, 16, 24), updates=20,
               train_hamiltonians=7452, validation_hamiltonians=1242):
    """Time fixed-shape work, discard all updates, preserve real-data provenance.

    A small declared non-test prefix is cyclically repeated ONLY to reproduce
    full input residency and batch shapes. These duplicates are not training
    examples for a scientific result. No weights, scores or selected model are
    retained, and actual independent-H counts are reported separately.
    """
    output, cache_root = Path(output), Path(cache_root)
    if output.exists():
        raise FileExistsError("Use a fresh timing report; never overwrite a measurement")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA is required for the PC timing probe")
    if not 1 <= updates <= 50 or train_hamiltonians < 32 * (updates + 5):
        raise ValueError("Probe must remain inside its first synthetic-residency epoch")
    started = time.perf_counter()
    report = {"schema_version": 1, "scope": "development-only disposable timing; no scientific training result",
              "configuration": {"sizes": list(sizes), "updates": updates,
                                "resident_train_count": train_hamiltonians,
                                "resident_validation_count": validation_hamiltonians},
              "source_sha256": _sha(__file__), "hardware_before": _hardware(),
              "torch": torch.__version__, "numpy": np.__version__,
              "gpu_name": torch.cuda.get_device_name(0), "training": [], "sampling": []}
    for L in sizes:
        if time.perf_counter() - started > 1200:
            raise TimeoutError("Development probe exceeded its 20-minute allowance")
        features = load_features(cache_root, L)
        real_train, real_val = features["train"], features["validation"]
        train_index = np.arange(train_hamiltonians) % len(real_train["y"])
        val_index = np.arange(validation_hamiltonians) % len(real_val["y"])
        arrays = (real_train["g"][train_index], real_train["y"][train_index],
                  real_val["g"][val_index], real_val["y"][val_index])
        for model in ("cnn", "map_mlp"):
            _hardware()
            settings = {"model": model, "architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 3},
                        "seed": 2026102798, "device": "cuda", "deterministic": True,
                        "torch_threads": 1, "learning_rate": 1e-4, "batch_size_hamiltonians": 32,
                        "max_epochs": 500, "validation_batch_size": 256,
                        "early_stopping": {"enabled": True, "min_epochs": 50,
                                           "patience_epochs": 30, "min_delta": 1e-6}}
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            before = time.perf_counter()
            session = ObservableTrainSession(*arrays, settings)
            torch.cuda.synchronize()
            initialization = time.perf_counter() - before
            session.advance(5)
            torch.cuda.synchronize()
            before = time.perf_counter()
            session.advance(updates)
            torch.cuda.synchronize()
            update_seconds = time.perf_counter() - before
            assert session.epoch == 0  # Measured updates contain no epoch validation.
            before = time.perf_counter()
            prediction = session.selected_predictions(arrays[2])
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - before
            if prediction.shape != arrays[3].shape or not np.isfinite(prediction).all():
                raise RuntimeError("Invalid CUDA forward output")
            row = {"L": L, "model": model, "view": "g", "updates": updates,
                   "independent_development_train_H": len(real_train["y"]),
                   "independent_development_validation_H": len(real_val["y"]),
                   "cyclic_duplicates_for_timing_only": True,
                   "initialization_plus_initial_validation_seconds": initialization,
                   "seconds_per_update": update_seconds / updates,
                   "forward_validation_seconds": forward_seconds,
                   "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "estimated_full_cap_seconds_one_fit": (500 *
                       (math.ceil(train_hamiltonians / 32) * update_seconds / updates + forward_seconds)
                       + initialization),
                   "projection_caveat": "full 500 epochs; no early-stop forecast; excludes checkpoint and final export; no error claim"}
            report["training"].append(row)
            print(json.dumps(row), flush=True)
            del session, prediction
            gc.collect()
            torch.cuda.empty_cache()
        fields = np.asarray(real_train["fields"][0], dtype=np.float64)
        lattice = square_lattice(L)
        before = time.perf_counter()
        chain = StarMetropolisChain(lattice, fields, 2026102797, backend="numba", lazy_probability=.05)
        chain.advance(20)
        cold = time.perf_counter() - before
        before = time.perf_counter()
        chain.advance(2000)
        warm = time.perf_counter() - before
        report["sampling"].append({"L": L, "cold_setup_and_20_sweeps_seconds": cold,
                                    "warm_2000_sweeps_seconds": warm,
                                    "estimated_100000_sweeps_one_chain_seconds": warm * 50,
                                    "scope": "one reused development field; not a mixing or parallel-throughput test"})
        del arrays, features
        gc.collect()
        report["hardware_after"] = _hardware()
        report["active_seconds"] = time.perf_counter() - started
        _json(output.with_suffix(".progress.json"), report)
    report["active_seconds"] = time.perf_counter() - started
    report["complete"] = True
    _json(output, report)
    return report
