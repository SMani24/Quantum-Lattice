"""Resumable, Hamiltonian-grouped datasets for the z-only solvable model.

One worker advances one independent chain by a bounded number of sweeps. Only
atomic on-disk checkpoints cross worker boundaries, so a pause is also safe
across shell sessions. No model training takes place here.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.solvable import local_estimators
from quantum_lattice.observables.statistics import autocorrelation_diagnostic, summarize
from quantum_lattice.simulation.solvable import StarMetropolisChain
from quantum_lattice.simulation.terminal import RunConsole


SCHEMA = 1


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config: dict) -> dict:
    """Validate scientific settings before making any files or launching workers."""
    config = deepcopy(config)
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unsupported configuration schema_version")
    lattice, fields = config["lattice"], config["fields"]
    if lattice.get("type") != "square_periodic":
        raise ValueError("this first runner supports square_periodic only")
    _integer(lattice["L"], "L", 3)
    if fields.get("channel") != "z_only" or fields.get("distribution") != "uniform":
        raise ValueError("this first runner requires z_only uniform fields")
    if not np.isfinite([fields["low"], fields["high"]]).all() or fields["low"] >= fields["high"]:
        raise ValueError("finite field bounds must satisfy low < high")
    # This bounded pilot keeps exponentials and exact checks well conditioned.
    if max(abs(fields["low"]), abs(fields["high"])) > 1.7:
        raise ValueError("field bounds beyond the paper's 1.7 need a separately reviewed protocol")
    dataset, sampler = config["dataset"], config["sampler"]
    _integer(dataset["hamiltonians"], "hamiltonians", 3)
    _integer(dataset["seed"], "seed", 0)
    fractions = dataset["split_fractions"]
    if set(fractions) != {"train", "validation", "test"}:
        raise ValueError("split_fractions needs train, validation, test")
    if not all(np.isfinite(f) and 0 < f < 1 for f in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("positive split fractions must sum to one")
    for key in ("total_sweeps", "thinning_sweeps"):
        _integer(sampler[key], key)
    _integer(sampler["burn_in_sweeps"], "burn_in_sweeps", 0)
    production = sampler["total_sweeps"] - sampler["burn_in_sweeps"]
    if production < sampler["thinning_sweeps"] or production % sampler["thinning_sweeps"]:
        raise ValueError("production sweeps must be a positive multiple of thinning_sweeps")
    if not 0 <= sampler["lazy_probability"] < 1:
        raise ValueError("lazy_probability must lie in [0,1)")
    if sampler.get("backend") not in {"auto", "python", "numba"}:
        raise ValueError("backend must be auto, python, or numba")
    _integer(config["statistics"]["block_size_records"], "block_size_records")
    for key in ("workers", "chunk_sweeps"):
        _integer(config["execution"][key], key)
    return config


def _canonical(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_npz(path: Path, **arrays):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _source_hashes(repo_root):
    paths = [repo_root / "src/quantum_lattice" / name for name in (
        "lattice/topology.py", "lattice/builders.py", "lattice/square.py", "simulation/solvable.py",
        "simulation/terminal.py", "observables/solvable.py", "observables/statistics.py",
        "data/generation.py")]
    paths += sorted((repo_root / "experiments/phase1_sample_generation").glob("*.py"))
    return {str(p.relative_to(repo_root)): _sha_file(p) for p in paths}


def _environment():
    versions = {}
    for package in ("numpy", "numba", "llvmlite", "quantum-lattice-research"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"python": sys.version, "platform": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "packages": versions}


def _bank(config):
    count = config["dataset"]["hamiltonians"]
    seed = config["dataset"]["seed"]
    names = ["train", "validation", "test"]
    fractions = [Fraction(str(config["dataset"]["split_fractions"][n])).limit_denominator(1000) for n in names]
    if sum(fractions) != 1 or any(abs(float(f) - config["dataset"]["split_fractions"][n]) > 1e-8 for n, f in zip(names, fractions)):
        raise ValueError("use simple split fractions with denominators <= 1000")
    period = math.lcm(*(f.denominator for f in fractions))
    if period > 4096:
        raise ValueError("split fractions need a smaller common denominator")
    labels = np.array([name for name, fraction in zip(names, fractions)
                       for _ in range(int(period * fraction))])
    splits = {}
    # Fixed balanced blocks preserve existing assignments when a bank grows.
    # Default period=8 gives 6/1/1; two blocks give the pilot's 12/2/2 split.
    for start in range(0, count, period):
        shuffled = np.random.default_rng(np.random.SeedSequence([seed, start // period, 2])).permutation(labels)
        for offset, name in enumerate(shuffled[:min(period, count - start)]):
            splits[start + offset] = str(name)
    if set(splits.values()) != set(names):
        raise ValueError("too few Hamiltonians for a nonempty train/validation/test split; use a whole split block")
    return [{"id": f"hamiltonian_{i:05d}", "index": i, "split": splits[i],
             "hamiltonian_uid": hashlib.sha256(_canonical({"lattice": config["lattice"], "fields": config["fields"], "field_seed": [seed, i, 0]}).encode()).hexdigest(),
             "field_seed": [seed, i, 0], "chain_seed": [seed, i, 1],
             "parent_hamiltonian_id": None} for i in range(count)]


def artifact_root(repo_root):
    return Path(repo_root).resolve() / "codex material/artifacts/phase1_sample_generation"


def _safe_run_dir(repo_root, run_dir):
    root = artifact_root(repo_root).resolve()
    path = Path(run_dir).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"run directory must be inside {root}")
    return path


def prepare_run(repo_root, config, run_dir=None, purpose="pilot") -> Path:
    config = validate_config(config)
    root = artifact_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / "runs" / f"{stamp}-{purpose}"
    run_dir = _safe_run_dir(repo_root, run_dir)
    bank = _bank(config)  # Reject impossible splits before creating a run.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "instances").mkdir()
    _atomic_json(run_dir / "config.json", config)
    _atomic_json(run_dir / "manifest.json", {
        "schema_version": SCHEMA, "purpose": purpose,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest(),
        "environment": _environment(), "source_hashes": _source_hashes(Path(repo_root)),
        "instances": bank,
        "protocol": "z-only positive-amplitude star orbit; lazy Metropolis; sweep-boundary records",
        "split_protocol": "balanced fixed-size blocks, stable under bank extension at fixed seed/fractions; regroup duplicates by hamiltonian_uid when merging banks",
        "measurement_scope": "known-field simulation estimators, not finite-shot X-basis measurements",
        "ordering": "vertex=y*L+x; edge=2*(y*L+x)+orientation; 0 outgoing horizontal, 1 outgoing vertical",
    })
    _atomic_json(run_dir / "status.json", {"state": "prepared", "completed": 0})
    return run_dir


@contextmanager
def _run_lock(run_dir):
    with (run_dir / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("this run is already active in another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _worker_init(backend):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Warm the compiled signature before task timing; startup remains in the
    # parent's elapsed session time. No retained physics data are generated.
    StarMetropolisChain(square_lattice(3), np.zeros(18), 0, backend=backend).advance(1)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _read_metadata(arrays):
    return json.loads(str(arrays["metadata_json"].item()))


def _instance_summary(path):
    arrays = _load_npz(path)
    return json.loads(str(arrays["summary_json"].item()))


def _final_arrays(lattice, fields, records, metadata, block_size):
    values = local_estimators(lattice, fields, records)
    arrays = {"states": records, "b_z": fields,
              "edge_vertex_indices": lattice.edge_vertex_indices,
              "metadata_json": np.array(_canonical(metadata))}
    for channel in ("star", "pair", "z"):
        data = values[channel]
        arrays[channel + "_values"] = data
        for key, val in summarize(data, block_size).items():
            arrays[channel + "_" + key] = val
    ends = lattice.edge_vertex_indices
    arrays["mlp_X"] = np.column_stack((arrays["star_mean"][ends[:, 0]],
                                       arrays["star_mean"][ends[:, 1]], arrays["pair_mean"]))
    arrays["target_abs_b"] = np.abs(fields)
    arrays["target_b"] = fields
    scalar_traces = {
        "log_weight": records @ fields,
        "edge_magnetization": records.mean(axis=1),
        "mean_star_estimator": values["star"].mean(axis=1),
    }
    diagnostics = {key: autocorrelation_diagnostic(trace) for key, trace in scalar_traces.items()}
    attempts = metadata["chain"]["attempted"]
    accepted = metadata["chain"]["accepted"]
    lazy = metadata["chain"]["lazy"]
    summary = {"id": metadata["instance"]["id"], "split": metadata["instance"]["split"],
               "records": len(records), "local_rows": lattice.n_edges,
               "sweeps": metadata["completed_sweeps"], "attempted": attempts,
               "accepted": accepted, "lazy_stays": lazy,
               "acceptance_per_attempt": accepted / attempts if attempts else None,
               "acceptance_per_nonlazy_proposal": accepted / (attempts - lazy) if attempts > lazy else None,
               "complete_blocks": len(records) // block_size,
               "records_excluded_from_block_se_only": len(records) % block_size,
               "compute_seconds": metadata["compute_seconds"],
               "backend": metadata["backend"], "diagnostics": diagnostics,
               "diagnostic_caveat": "thinning, batch means and scalar ESS do not establish equilibration"}
    arrays["summary_json"] = np.array(_canonical(summary))
    return arrays, summary


def _advance_job(run_dir_string, instance, config):
    """Bounded, independent worker task; returns progress after an atomic save."""
    started = time.perf_counter()
    run_dir = Path(run_dir_string)
    checkpoint = run_dir / "checkpoints" / (instance["id"] + ".npz")
    final = run_dir / "instances" / (instance["id"] + ".npz")
    if final.exists():
        return {"id": instance["id"], "done": True, **_instance_summary(final)}
    lattice = square_lattice(config["lattice"]["L"])
    sampler = config["sampler"]
    if checkpoint.exists():
        saved = _load_npz(checkpoint)
        metadata = _read_metadata(saved)
        if metadata["instance"] != instance:
            raise ValueError("checkpoint Hamiltonian identity mismatch")
        fields, records = saved["b_z"], saved["states"]
        chain = StarMetropolisChain.from_snapshot(
            lattice, fields, metadata["chain"], backend=sampler["backend"],
            lazy_probability=sampler["lazy_probability"])
        completed = metadata["completed_sweeps"]
        if not isinstance(completed, int) or not 0 <= completed < sampler["total_sweeps"]:
            raise ValueError("checkpoint sweep count is invalid")
        if chain.attempted != completed * lattice.n_vertices:
            raise ValueError("checkpoint attempts do not match completed sweeps")
        expected_count = max(0, completed - sampler["burn_in_sweeps"]) // sampler["thinning_sweeps"]
        if records.shape != (expected_count, lattice.n_edges) or records.dtype != np.int8 or not np.isin(records, [-1, 1]).all():
            raise ValueError("checkpoint retained states do not match its schedule")
        expected_fields = np.random.default_rng(np.random.SeedSequence(instance["field_seed"])).uniform(
            config["fields"]["low"], config["fields"]["high"], lattice.n_edges)
        if not np.array_equal(fields, expected_fields):
            raise ValueError("checkpoint fields do not match the seeded Hamiltonian")
    else:
        fields = np.random.default_rng(np.random.SeedSequence(instance["field_seed"])).uniform(
            config["fields"]["low"], config["fields"]["high"], lattice.n_edges)
        chain = StarMetropolisChain(lattice, fields, np.random.SeedSequence(instance["chain_seed"]),
                                    backend=sampler["backend"], lazy_probability=sampler["lazy_probability"])
        records = np.empty((0, lattice.n_edges), dtype=np.int8)
        metadata = {"instance": instance, "completed_sweeps": 0, "compute_seconds": 0.0}
    current = metadata["completed_sweeps"]
    stop = min(sampler["total_sweeps"], current + config["execution"]["chunk_sweeps"])
    retained = []
    burn, stride = sampler["burn_in_sweeps"], sampler["thinning_sweeps"]
    while current < stop:
        next_record = burn if current < burn else burn + ((current - burn) // stride + 1) * stride
        target = min(stop, next_record)
        chain.advance(target - current)
        current = target
        if current > burn and (current - burn) % stride == 0:
            retained.append(chain.spins.copy())
    if retained:
        records = np.concatenate((records, np.stack(retained)), axis=0)
    metadata.update({"schema_version": SCHEMA, "completed_sweeps": current,
                     "chain": chain.snapshot(), "backend": chain.backend,
                     "compute_seconds": metadata["compute_seconds"] + time.perf_counter() - started})
    if current == sampler["total_sweeps"]:
        expected_records = (current - burn) // stride
        if len(records) != expected_records:
            raise RuntimeError("retained-record count disagrees with sampling schedule")
        arrays, summary = _final_arrays(lattice, fields, records, metadata,
                                       config["statistics"]["block_size_records"])
        _atomic_npz(final, **arrays)
        _atomic_json(final.with_suffix(".json"), summary)
        checkpoint.unlink(missing_ok=True)
        return {"done": True, **summary}
    _atomic_npz(checkpoint, states=records, b_z=fields, metadata_json=np.array(_canonical(metadata)))
    return {"id": instance["id"], "done": False, "sweeps": current,
            "attempted": chain.attempted, "records": len(records)}


def inspect_run(run_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    progress = {}
    for instance in manifest["instances"]:
        name = instance["id"]
        final = run_dir / "instances" / (name + ".npz")
        checkpoint = run_dir / "checkpoints" / (name + ".npz")
        if final.exists():
            progress[name] = {"done": True, **_instance_summary(final)}
        elif checkpoint.exists():
            arrays = _load_npz(checkpoint)
            metadata = _read_metadata(arrays)
            progress[name] = {"done": False, "sweeps": metadata["completed_sweeps"],
                              "attempted": metadata["chain"]["attempted"], "records": len(arrays["states"])}
        else:
            progress[name] = {"done": False, "sweeps": 0, "attempted": 0, "records": 0}
    return config, manifest, progress


def verify_run(run_dir):
    """Check identities, physics constraints and file/schema consistency, not mixing."""
    run_dir = Path(run_dir)
    config, manifest, progress = inspect_run(run_dir)
    validate_config(config)
    if manifest["config_sha256"] != hashlib.sha256(_canonical(config).encode()).hexdigest() or manifest["instances"] != _bank(config):
        raise ValueError("dataset configuration or Hamiltonian seed/group plan has changed")
    if not all(p["done"] for p in progress.values()):
        raise ValueError("dataset is incomplete; resume it before verification")
    lattice = square_lattice(config["lattice"]["L"])
    checks = []
    expected = (config["sampler"]["total_sweeps"] - config["sampler"]["burn_in_sweeps"]) // config["sampler"]["thinning_sweeps"]
    for instance in manifest["instances"]:
        path = run_dir / "instances" / (instance["id"] + ".npz")
        arrays = _load_npz(path)
        metadata = _read_metadata(arrays)
        records = arrays["states"]
        if metadata["instance"] != instance or records.shape != (expected, lattice.n_edges):
            raise ValueError("instance identity or state shape mismatch")
        if not np.isin(records, [-1, 1]).all() or lattice.plaquette_syndrome(records == -1).any():
            raise ValueError("invalid spin values or a violated plaquette sector")
        # Reference Wilson loops along the two noncontractible square cycles.
        L = config["lattice"]["L"]
        loops = [2 * np.arange(L), 2 * L * np.arange(L) + 1]
        if any(np.any(np.prod(records[:, loop], axis=1) != 1) for loop in loops):
            raise ValueError("reference Wilson-loop sector was not preserved")
        fields = np.random.default_rng(np.random.SeedSequence(instance["field_seed"])).uniform(
            config["fields"]["low"], config["fields"]["high"], lattice.n_edges)
        if not np.array_equal(fields, arrays["b_z"]):
            raise ValueError("field seed does not reproduce saved Hamiltonian")
        if not np.array_equal(arrays["edge_vertex_indices"], lattice.edge_vertex_indices):
            raise ValueError("edge ordering differs from the square factory")
        calculated = local_estimators(lattice, fields, records)
        for channel in ("star", "pair", "z"):
            if not np.isfinite(arrays[channel + "_values"]).all():
                raise ValueError("nonfinite observable values")
            np.testing.assert_array_equal(arrays[channel + "_values"], calculated[channel])
            for key, value in summarize(calculated[channel], config["statistics"]["block_size_records"]).items():
                np.testing.assert_allclose(arrays[channel + "_" + key], value, rtol=0, atol=0, equal_nan=True)
        ends = lattice.edge_vertex_indices
        np.testing.assert_array_equal(arrays["mlp_X"], np.column_stack((arrays["star_mean"][ends[:, 0]], arrays["star_mean"][ends[:, 1]], arrays["pair_mean"])))
        np.testing.assert_array_equal(arrays["target_abs_b"], np.abs(fields))
        np.testing.assert_array_equal(arrays["target_b"], fields)
        if metadata["chain"]["attempted"] != config["sampler"]["total_sweeps"] * lattice.n_vertices:
            raise ValueError("attempt count disagrees with sweep definition")
        checks.append({"id": instance["id"], "split": instance["split"], "sha256": _sha_file(path)})
    report = {"schema_version": SCHEMA, "passed": True, "instances": checks,
              "local_rows": len(checks) * lattice.n_edges,
              "scope": "storage, group identity, estimators and sector invariants; not an equilibration certificate"}
    _atomic_json(run_dir / "verification.json", report)
    return report


def run_generation(repo_root, run_dir, workers=None, interactive=True, max_chunks=None):
    """Run/resume; p pauses, r resumes, q or Ctrl+C saves and returns.

    max_chunks is a bounded integration-test/validation hook. Each submitted
    chunk finishes and saves before pause/quit is acknowledged.
    """
    run_dir = _safe_run_dir(repo_root, run_dir)
    with _run_lock(run_dir):
        config, manifest, progress = inspect_run(run_dir)
        validate_config(config)
        if manifest["config_sha256"] != hashlib.sha256(_canonical(config).encode()).hexdigest():
            raise ValueError("saved scientific configuration has changed")
        if manifest["source_hashes"] != _source_hashes(Path(repo_root)):
            raise ValueError("source changed since this run was prepared; restore that source before resuming")
        if manifest["instances"] != _bank(config):
            raise ValueError("Hamiltonian manifest differs from its deterministic seed/group plan")
        if manifest["environment"]["packages"]["numpy"] != np.__version__:
            raise ValueError("NumPy version changed; use the recorded version for exact RNG continuation")
        worker_count = workers if workers is not None else config["execution"]["workers"]
        _integer(worker_count, "workers")
        worker_count = min(worker_count, len(manifest["instances"]))
        pending = deque(i for i in manifest["instances"] if not progress[i["id"]]["done"])
        started = time.perf_counter()
        active_elapsed = 0.0
        previous_tick = started
        base_attempts = sum(p["attempted"] for p in progress.values())
        attempts_total = len(progress) * config["sampler"]["total_sweeps"] * config["lattice"]["L"] ** 2
        samples_total = len(progress) * (config["sampler"]["total_sweeps"] - config["sampler"]["burn_in_sweeps"]) // config["sampler"]["thinning_sweeps"]
        paused = quitting = False
        submitted = 0
        futures = {}
        # One NumPy/BLAS thread per process; parallelize independent Hamiltonians.
        limits = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}
        for key in limits:
            os.environ[key] = "1"
        session_state = "failed"
        try:
            with RunConsole(len(progress), title="Phase 1 · solvable-model samples", interactive=interactive) as console:
                with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init, initargs=(config["sampler"]["backend"],)) as pool:
                    while pending or futures:
                        try:
                            command = console.poll()
                            if command == "pause":
                                paused = True
                            elif command == "resume":
                                paused = False
                            elif command == "quit":
                                quitting = True
                            while pending and len(futures) < worker_count and not paused and not quitting:
                                if max_chunks is not None and submitted >= max_chunks:
                                    quitting = True
                                    break
                                instance = pending.popleft()
                                future = pool.submit(_advance_job, str(run_dir), instance, config)
                                futures[future] = instance
                                submitted += 1
                            if futures:
                                ready, _ = wait(futures, timeout=0.15, return_when=FIRST_COMPLETED)
                                for future in ready:
                                    instance = futures.pop(future)
                                    result = future.result()
                                    progress[instance["id"]] = result
                                    if not result["done"]:
                                        pending.append(instance)
                            elif paused and not quitting:
                                time.sleep(0.1)
                            complete_count = sum(p["done"] for p in progress.values())
                            now = time.perf_counter()
                            if futures or not paused:
                                active_elapsed += now - previous_tick
                            previous_tick = now
                            attempts_done = sum(p["attempted"] for p in progress.values())
                            new_attempts = attempts_done - base_attempts
                            eta = (attempts_total - attempts_done) * active_elapsed / new_attempts if new_attempts else None
                            status = {"completed": complete_count, "total": len(progress), "running": len(futures),
                                      "paused": paused and not futures, "pausing": paused and bool(futures),
                                      "elapsed_seconds": now - started, "eta_seconds": eta,
                                      "attempts_done": attempts_done, "attempts_total": attempts_total,
                                      "samples_done": sum(p["records"] for p in progress.values()), "samples_total": samples_total,
                                      "workers": worker_count,
                                      "message": "Saving active chunks before exit" if quitting else "Waiting for active chunks to checkpoint" if paused and futures else "Paused; checkpoints saved" if paused else "Sampling independent Hamiltonians"}
                            console.render(status)
                            _atomic_json(run_dir / "status.json", {**status, "state": "stopping" if quitting else "pausing" if paused and futures else "paused" if paused else "running"})
                            if quitting and not futures:
                                break
                        except KeyboardInterrupt:
                            quitting = True
                            paused = False
                    session_state = "complete" if all(p["done"] for p in progress.values()) else "paused"
        finally:
            for key, previous in limits.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
            _atomic_json(run_dir / "status.json", {"state": "verifying" if session_state == "complete" else session_state,
                         "completed": sum(p["done"] for p in progress.values()), "total": len(progress),
                         "elapsed_session_seconds": time.perf_counter() - started})
        elapsed = time.perf_counter() - started
        attempts_new = sum(p["attempted"] for p in progress.values()) - base_attempts
        result = {"run_dir": str(run_dir), "state": session_state,
                  "completed": sum(p["done"] for p in progress.values()), "total": len(progress),
                  "elapsed_seconds": elapsed, "attempts_this_session": attempts_new,
                  "attempts_per_second_including_startup_io": attempts_new / elapsed if elapsed else None,
                  "workers": worker_count}
        if session_state == "complete":
            _atomic_json(run_dir / "status.json", {"state": "verifying", "completed": len(progress), "total": len(progress)})
            try:
                report = verify_run(run_dir)
            except Exception:
                _atomic_json(run_dir / "status.json", {"state": "verification_failed"})
                raise
            entries = []
            for instance in manifest["instances"]:
                path = run_dir / "instances" / (instance["id"] + ".npz")
                summary = _instance_summary(path)
                if not path.with_suffix(".json").exists():
                    _atomic_json(path.with_suffix(".json"), summary)
                entries.append({"path": str(path.relative_to(run_dir)), "sha256": _sha_file(path), **summary})
            _atomic_json(run_dir / "dataset.json", {"schema_version": SCHEMA, "instances": entries,
                         "split_counts": {s: sum(i["split"] == s for i in manifest["instances"]) for s in ("train", "validation", "test")},
                         "local_rows": report["local_rows"], "sampling_scope": manifest["measurement_scope"]})
            _atomic_json(run_dir / "status.json", {"state": "complete", "completed": len(progress), "total": len(progress),
                         "elapsed_session_seconds": time.perf_counter() - started})
        _atomic_json(run_dir / "last_session.json", result)
        return result


def benchmark_config(config, sweeps=2000, jobs=4):
    """Make explicit short-run overrides; never modify the user's pilot config."""
    config = deepcopy(config)
    stride = config["sampler"]["thinning_sweeps"]
    _integer(sweeps, "benchmark sweeps", 10 * stride)
    sweeps = sweeps // stride * stride
    config["dataset"]["hamiltonians"] = jobs
    config["dataset"]["split_fractions"] = {"train": 0.5, "validation": 0.25, "test": 0.25}
    config["sampler"]["total_sweeps"] = sweeps
    config["sampler"]["burn_in_sweeps"] = (sweeps // 10) // stride * stride
    config["execution"]["chunk_sweeps"] = min(config["execution"]["chunk_sweeps"], sweeps)
    return validate_config(config)
