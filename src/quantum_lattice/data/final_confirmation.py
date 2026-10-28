"""Fresh, sealed confirmation cohorts with chunk-resumable star sampling.

This adapter intentionally has no train/validation/test split. A caller seals
the method/model plan first, then creates an independently seeded confirmation
cohort. Old bank manifests are inspected for identities only. True fields are
used for the declared simulation estimators and kept separate from raw inputs.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import time

import numpy as np

from quantum_lattice.data.state_storage import decode_states, encode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.conditioned import exact_reference, paired_estimators
from quantum_lattice.observables.statistics import autocorrelation_diagnostic
from quantum_lattice.simulation.solvable import StarMetropolisChain


SCHEMA = 1
NAMESPACE = 2026102707
SIZES = (3, 4, 8, 12, 16, 20, 24)
SOURCE_FILES = (
    "data/final_confirmation.py", "data/state_storage.py", "lattice/topology.py",
    "lattice/builders.py", "lattice/square.py", "simulation/solvable.py",
    "observables/conditioned.py", "observables/statistics.py")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _safe_run(repo_root, run):
    allowed = (Path(repo_root).resolve() / "codex material/artifacts/final_benchmark").resolve()
    run = Path(run)
    run = (Path(repo_root) / run).resolve() if not run.is_absolute() else run.resolve()
    if run == allowed or not run.is_relative_to(allowed):
        raise ValueError(f"Confirmation output must be below {allowed}")
    return run


@contextmanager
def _lock(path):
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Confirmation cohort is already in use") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _sources(repo_root):
    return {str(Path("src/quantum_lattice") / name):
            _sha(Path(repo_root) / "src/quantum_lattice" / name) for name in SOURCE_FILES}


def _sampling(value):
    required = {"total_sweeps", "burn_in_sweeps", "thinning_sweeps", "lazy_probability", "backend"}
    optional = {"chunk_sweeps", "exact_small_sizes", "block_size_records"}
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - required - optional:
        raise ValueError("Sampling settings contain missing or unknown keys")
    value = {"chunk_sweeps": 2000, "exact_small_sizes": False,
             "block_size_records": 150, **value}
    for key in ("total_sweeps", "thinning_sweeps", "chunk_sweeps", "block_size_records"):
        _integer(value[key], key)
    _integer(value["burn_in_sweeps"], "burn_in_sweeps", 0)
    production = value["total_sweeps"] - value["burn_in_sweeps"]
    if production <= 0 or production % value["thinning_sweeps"]:
        raise ValueError("Positive production sweeps must be divisible by thinning_sweeps")
    if (type(value["lazy_probability"]) not in (int, float)
            or not np.isfinite(value["lazy_probability"]) or not 0 <= value["lazy_probability"] < 1):
        raise ValueError("lazy_probability must be finite and in [0,1)")
    if value["backend"] not in {"python", "numba", "auto"}:
        raise ValueError("Unknown sampling backend")
    if type(value["exact_small_sizes"]) is not bool:
        raise ValueError("exact_small_sizes must be boolean")
    return value


def _exclusions(repo_root, paths):
    identities, hashes = set(), {}
    for name in paths:
        path = (Path(repo_root) / name).resolve()
        if not path.is_relative_to(Path(repo_root).resolve()):
            raise ValueError("Exclusion manifest escapes repository")
        before = _sha(path)
        manifest = _read(path)
        entries = manifest.get("instances")
        if not isinstance(entries, list):
            raise ValueError("An exclusion manifest must list Hamiltonian instances")
        for item in entries:
            uid = item.get("hamiltonian_uid")
            if not isinstance(uid, str) or not re.fullmatch(r"[0-9a-f]{64}", uid):
                raise ValueError("Exclusion manifest contains an invalid Hamiltonian UID")
            identities.add(uid)
        if before != _sha(path):
            raise ValueError("Exclusion manifest changed while reading")
        hashes[str(path.relative_to(Path(repo_root).resolve()))] = before
    return identities, hashes


def _verify_bindings(repo_root, bindings):
    if not isinstance(bindings, dict):
        raise ValueError("Confirmation method bindings must be an object")
    # Arbitrary descriptive entries may accompany explicit model/source pins.
    for group in ("source_sha256", "model_sha256"):
        values = bindings.get(group, {})
        if not isinstance(values, dict):
            raise ValueError(f"{group} must map repository-relative files to hashes")
        for name, expected in values.items():
            path = (Path(repo_root) / name).resolve()
            if (not path.is_relative_to(Path(repo_root).resolve())
                    or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                    or _sha(path) != expected):
                raise ValueError(f"Confirmation method file changed: {name}")


def prepare_confirmation(repo_root, output_root, sizes, count_per_size, seed, sampling,
                         exclusion_manifests, freeze_sha256, *, bindings=None):
    """Seal fresh IDs/seeds before any fields are drawn or records generated.

    ``freeze_sha256`` identifies the caller's approved method/evaluation freeze.
    File pins in ``bindings`` (``source_sha256`` / ``model_sha256``) are verified;
    the adapter cannot infer approval merely from a hash string. Generation is a
    separate call and this preparation never opens old scientific payloads.
    """
    root = Path(repo_root).resolve()
    run = _safe_run(root, output_root)
    sizes = list(sizes)
    if not sizes or any(type(L) is not int or L not in SIZES for L in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Confirmation sizes must be unique supported reference sizes")
    _integer(seed, "seed", 0)
    if seed >= 2**63:
        raise ValueError("seed must be below2**63")
    if not isinstance(freeze_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", freeze_sha256):
        raise ValueError("A SHA256 method-freeze identity is required")
    sampler = _sampling(sampling)
    if type(count_per_size) is int:
        counts = {str(L): _integer(count_per_size, "count_per_size") for L in sizes}
    elif isinstance(count_per_size, dict) and set(map(str, count_per_size)) == set(map(str, sizes)):
        counts = {str(L): _integer(count_per_size.get(str(L), count_per_size.get(L)), "count_per_size") for L in sizes}
    else:
        raise ValueError("count_per_size must be a count or a complete per-size map")
    excluded, exclusions = _exclusions(root, exclusion_manifests)
    bindings = {} if bindings is None else json.loads(_canonical(bindings))
    _verify_bindings(root, bindings)
    instances = []
    for L in sizes:
        for index in range(counts[str(L)]):
            field_seed = [seed, NAMESPACE, L, index, 0]
            identity = {"lattice": {"type": "square_periodic", "L": L},
                        "fields": {"channel": "z_only", "distribution": "uniform", "low": -1.7, "high": 1.7},
                        "field_seed": field_seed}
            uid = _digest(identity)
            if uid in excluded:
                raise ValueError("Fresh confirmation UID overlaps an excluded prior Hamiltonian")
            instances.append({"id": f"confirmation_L{L}_{index:05d}", "L": L, "index": index,
                "role": "confirmation", "hamiltonian_uid": uid, "field_seed": field_seed,
                "chain_seed": [seed, NAMESPACE, L, index, 1], "parent_hamiltonian_id": None})
    if len({item["hamiltonian_uid"] for item in instances}) != len(instances):
        raise ValueError("Duplicate confirmation Hamiltonian identity")
    plan = {"schema_version": SCHEMA, "role": "confirmation", "freeze_sha256": freeze_sha256,
        "sizes": sizes, "count_per_size": counts, "seed": seed, "namespace": NAMESPACE,
        "sampling": sampler, "field_prior": {"distribution": "uniform", "low": -1.7, "high": 1.7},
        "source_sha256": _sources(root), "bindings": bindings, "excluded_manifests": exclusions,
        "excluded_uid_count": len(excluded), "instances": instances,
        "scope": "known-field simulation f/g views; raw states contain no true-field inputs",
        "uid_meaning": "generation-provenance identity, not a hash of actual field values"}
    run.mkdir(parents=True, exist_ok=True)
    with _lock(run / ".cohort.lock"):
        if (run / "plan.json").exists():
            if load_confirmation_plan(root, run) != plan:
                raise ValueError("Fresh cohort settings differ; use a new output directory")
            return run
        if any(p.name not in {".cohort.lock"} for p in run.iterdir()):
            raise ValueError("Refusing to prepare over nonempty confirmation output")
        _json(run / "plan.json", plan)
        _json(run / "plan-sha256.json", {"sha256": _digest(plan)})
        _json(run / "access.json", {"state": "prepared_no_fields_drawn", "freeze_sha256": freeze_sha256,
                                   "prior_payloads_loaded": False})
    return run


def load_confirmation_plan(repo_root, run):
    root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    plan = _read(run / "plan.json")
    if (plan.get("schema_version") != SCHEMA or plan.get("role") != "confirmation"
            or _read(run / "plan-sha256.json") != {"sha256": _digest(plan)}):
        raise ValueError("Confirmation plan integrity or role mismatch")
    if plan["source_sha256"] != _sources(root):
        raise ValueError("Confirmation scientific source changed")
    if _sampling(plan["sampling"]) != plan["sampling"]:
        raise ValueError("Confirmation sampling settings changed")
    for name, expected in plan["excluded_manifests"].items():
        if _sha(root / name) != expected:
            raise ValueError("Excluded prior-manifest identity changed")
    _verify_bindings(root, plan["bindings"])
    return plan


@lru_cache(maxsize=8)
def _lattice(L):
    return square_lattice(L)


def _job(run, instance):
    return Path(run) / "instances" / instance["id"]


def _done(run, plan, instance):
    directory = _job(run, instance)
    path = directory / "done.json"
    if not path.exists():
        return None
    value = _read(path)
    if (value.get("schema_version") != SCHEMA or value.get("plan_sha256") != _digest(plan) or value.get("instance") != instance
            or set(value.get("files", {})) != {"records.npz", "summary.json"}):
        raise ValueError("Completed confirmation record identity mismatch")
    if any(_sha(directory / name) != digest for name, digest in value["files"].items()):
        raise ValueError("Completed confirmation record checksum mismatch")
    return value


def _fields(instance):
    return np.random.default_rng(np.random.SeedSequence(instance["field_seed"])).uniform(
        -1.7, 1.7, 2 * instance["L"]**2)


def _checkpoint(directory, plan, instance, chain, records, completed, active_seconds):
    metadata = {"instance": instance, "plan_sha256": _digest(plan),
                "completed_sweeps": completed, "retained_records": len(records),
                "active_seconds": active_seconds, "chain": chain.snapshot(), "backend": chain.backend}
    path = directory / f"checkpoint-{completed:09d}.npz"
    previous = _read(directory / "checkpoint.json")["path"] if (directory / "checkpoint.json").exists() else None
    _npz(path, {**encode_states(records, "packed_bits"), "metadata_json": np.array(_canonical(metadata))})
    # The pointer is the commit; a crash leaves either the old valid state or
    # a harmless unreferenced new checkpoint, never mixed metadata and arrays.
    _json(directory / "checkpoint.json", {"path": path.name, "sha256": _sha(path),
        "instance_id": instance["id"], "plan_sha256": _digest(plan)})
    if previous and previous != path.name:
        (directory / previous).unlink(missing_ok=True)
    return metadata


def _restore(directory, plan, instance):
    lattice = _lattice(instance["L"])
    b, sampler = _fields(instance), plan["sampling"]
    pointer = directory / "checkpoint.json"
    if not pointer.exists():
        chain = StarMetropolisChain(lattice, b, np.random.SeedSequence(instance["chain_seed"]),
            backend=sampler["backend"], lazy_probability=sampler["lazy_probability"])
        return chain, np.empty((0, lattice.n_edges), dtype=np.int8), 0, 0.0
    head = _read(pointer)
    if (head.get("instance_id") != instance["id"] or head.get("plan_sha256") != _digest(plan)
            or not re.fullmatch(r"checkpoint-\d{9}\.npz", head.get("path", ""))
            or _sha(directory / head["path"]) != head.get("sha256")):
        raise ValueError("Confirmation checkpoint pointer/hash mismatch")
    archive = _archive(directory / head["path"])
    metadata = json.loads(archive["metadata_json"].item())
    if metadata.get("instance") != instance or metadata.get("plan_sha256") != _digest(plan):
        raise ValueError("Confirmation checkpoint identity mismatch")
    completed = metadata["completed_sweeps"]
    if type(completed) is not int or not 0 <= completed <= sampler["total_sweeps"]:
        raise ValueError("Confirmation checkpoint sweep schedule mismatch")
    records = decode_states(archive)
    expected = max(0, completed - sampler["burn_in_sweeps"]) // sampler["thinning_sweeps"]
    if records.shape != (expected, lattice.n_edges) or metadata.get("retained_records") != expected:
        raise ValueError("Confirmation checkpoint record count mismatch")
    chain = StarMetropolisChain.from_snapshot(lattice, b, metadata["chain"],
        backend=sampler["backend"], lazy_probability=sampler["lazy_probability"])
    if chain.attempted != completed * lattice.n_vertices:
        raise ValueError("Confirmation checkpoint proposal count mismatch")
    active = metadata.get("active_seconds")
    if type(active) not in (float, int) or not np.isfinite(active) or active < 0:
        raise ValueError("Confirmation checkpoint active time mismatch")
    return chain, records, completed, float(active)


def _map_records(values, L):
    return np.stack((values["star"].reshape(-1, L, L),
                     values["pair"][:, 0::2].reshape(-1, L, L),
                     values["pair"][:, 1::2].reshape(-1, L, L)), axis=1)


def _finalize(directory, plan, instance, chain, records, active_seconds):
    L, lattice, sampler = instance["L"], _lattice(instance["L"]), plan["sampling"]
    count, block_size = len(records), sampler["block_size_records"]
    sums = {view: np.zeros((3, L, L), dtype=np.float64) for view in ("f", "g")}
    blocks = {view: [] for view in ("f", "g")}
    traces = {"log_weight": np.sum(records * chain.fields, axis=1, dtype=np.float64),
              "edge_magnetization": records.mean(axis=1)}
    star_traces = {view: [] for view in ("f", "g")}
    for start in range(0, count, block_size):
        paired = paired_estimators(lattice, chain.fields, records[start:start+block_size])
        for view in ("f", "g"):
            maps = _map_records({channel: paired[channel][view] for channel in ("star", "pair")}, L)
            sums[view] += maps.sum(axis=0)
            star_traces[view].append(maps[:, 0].mean(axis=(1, 2)))
            if len(maps) == block_size:
                blocks[view].append(maps.mean(axis=0))
    arrays = {**encode_states(records, "packed_bits"), "fields": chain.fields,
        "y": np.abs(chain.fields).reshape(L, L, 2).transpose(2, 0, 1).astype(np.float32),
        "hamiltonian_id": np.array(instance["id"]),
        "hamiltonian_uid": np.array(instance["hamiltonian_uid"]), "role": np.array("confirmation")}
    for view in ("f", "g"):
        arrays[f"x_{view}"] = (sums[view] / count).astype(np.float32)
        arrays[f"block_means_{view}"] = (np.stack(blocks[view]) if blocks[view] else
                                        np.empty((0, 3, L, L), dtype=np.float64))
        traces[f"mean_{view}_star"] = np.concatenate(star_traces[view])
    for key, trace in traces.items():
        arrays["trace_" + key] = trace
    if sampler["exact_small_sizes"] and L in (3, 4):
        exact = exact_reference(lattice, chain.fields)
        arrays["x_exact"] = _map_records({key: exact["truth"][key][None, :]
            for key in ("star", "pair")}, L)[0].astype(np.float32)
    summary = {"instance": instance, "plan_sha256": _digest(plan), "records": count,
        "sweeps": sampler["total_sweeps"], "attempted": chain.attempted,
        "accepted": chain.accepted, "lazy_stays": chain.lazy, "backend": chain.backend,
        "field_sha256": hashlib.sha256(chain.fields.tobytes()).hexdigest(),
        "sampling_active_seconds": active_seconds,
        "sampling_time_scope": "sampling/checkpoint preparation; scheduler wall time includes final preprocessing and I/O",
        "known_fields_for_features": True, "states_encoding": "packed_bits",
        "block_size_records": block_size, "complete_blocks": count // block_size,
        "excluded_records_from_block_means_only": count % block_size,
        "diagnostics": {key: autocorrelation_diagnostic(trace) for key, trace in traces.items()},
        "diagnostic_scope": "single-chain descriptive traces/block means do not certify mixing; no across-H R-hat"}
    arrays["summary_json"] = np.array(_canonical(summary))
    _npz(directory / "records.npz", arrays)
    _json(directory / "summary.json", summary)
    marker = {"schema_version": SCHEMA, "instance": instance, "plan_sha256": _digest(plan),
              "files": {name: _sha(directory / name) for name in ("records.npz", "summary.json")}}
    _json(directory / "done.json", marker)
    pointer = directory / "checkpoint.json"
    if pointer.exists():
        (directory / _read(pointer)["path"]).unlink(missing_ok=True)
        pointer.unlink()
    return marker


def _advance(run, plan, instance, max_chunks=1, stop_requested=None, max_seconds=None):
    started = time.monotonic()
    committed = _done(run, plan, instance)
    if committed:
        return {"id": instance["id"], "done": True, "already_complete": True}
    directory = _job(run, instance)
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory / ".instance.lock"):
        chain, records, current, previous_active = _restore(directory, plan, instance)
        sampler = plan["sampling"]
        burn, stride, chunks = sampler["burn_in_sweeps"], sampler["thinning_sweeps"], 0
        def stopping():
            return ((stop_requested is not None and stop_requested()) or
                    (max_seconds is not None and time.monotonic() - started >= max_seconds))
        while current < sampler["total_sweeps"] and chunks < max_chunks and not stopping():
            end = min(current + sampler["chunk_sweeps"], sampler["total_sweeps"])
            retained = []
            while current < end:
                next_record = burn if current < burn else burn + ((current-burn)//stride+1)*stride
                target = min(end, next_record)
                chain.advance(target-current)
                current = target
                if current > burn and (current-burn) % stride == 0:
                    retained.append(chain.spins.copy())
            if retained:
                records = np.concatenate((records, np.stack(retained)), axis=0)
            chunks += 1
            _checkpoint(directory, plan, instance, chain, records, current,
                        previous_active + time.monotonic()-started)
        if current == sampler["total_sweeps"] and not stopping():
            _finalize(directory, plan, instance, chain, records,
                      previous_active + time.monotonic()-started)
            return {"id": instance["id"], "done": True, "sweeps": current, "records": len(records)}
        return {"id": instance["id"], "done": False, "sweeps": current,
                "records": len(records), "chunks": chunks, "budget_paused": stopping()}


def advance_confirmation_instance(repo_root, run, instance_id, *, max_chunks=1,
                                  stop_requested=None, max_seconds=None):
    """Advance a bounded chunk of one instance; safe across process restarts."""
    _integer(max_chunks, "max_chunks")
    run = _safe_run(repo_root, run)
    plan = load_confirmation_plan(repo_root, run)
    matches = [item for item in plan["instances"] if item["id"] == instance_id]
    if len(matches) != 1:
        raise ValueError("Unknown confirmation instance ID")
    _json(run / "access.json", {"state": "generating_fresh_confirmation", "freeze_sha256": plan["freeze_sha256"],
                               "prior_payloads_loaded": False})
    return _advance(run, plan, matches[0], max_chunks, stop_requested, max_seconds)


_WORKER = None


def _worker_init(run, plan):
    global _WORKER
    _WORKER = (run, plan)


def _worker_advance(instance):
    return _advance(_WORKER[0], _WORKER[1], instance)


def run_confirmation(repo_root, run, *, workers=1, max_seconds=None, stop_requested=None,
                     max_chunks=None, progress_callback=None):
    """Dispatch bounded chunk jobs and pause without losing completed records.

    A stop/budget request halts dispatch and drains at most one sampler chunk
    per worker. The caller owns its campaign-wide cumulative time ceiling.
    """
    _integer(workers, "workers")
    if max_chunks is not None:
        _integer(max_chunks, "max_chunks")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds < 0):
        raise ValueError("max_seconds must be finite and nonnegative")
    run = _safe_run(repo_root, run)
    started = time.monotonic()
    with _lock(run / ".cohort.lock"):
        plan = load_confirmation_plan(repo_root, run)
        pending = [item for item in plan["instances"] if not _done(run, plan, item)]
        completed = len(plan["instances"]) - len(pending)
        _json(run / "access.json", {"state": "generating_fresh_confirmation" if pending else "complete",
                                   "freeze_sha256": plan["freeze_sha256"], "prior_payloads_loaded": False})
        old_seconds = _read(run / "state.json").get("active_seconds", 0.0) if (run / "state.json").exists() else 0.0
        count, submitted = 0, 0
        def stopping():
            return ((stop_requested is not None and stop_requested()) or
                (max_seconds is not None and time.monotonic()-started >= max_seconds) or
                (max_chunks is not None and submitted >= max_chunks))
        def state():
            value = {"state": "complete" if completed == len(plan["instances"]) else "paused",
                "completed": completed, "total": len(plan["instances"]), "chunks_this_call": count,
                "active_seconds": old_seconds + time.monotonic()-started,
                "active_time_scope": "cohort scheduler wall time across calls, including spawn/finalization/I/O"}
            _json(run / "state.json", value)
            if progress_callback is not None:
                progress_callback(value)
            return value
        context = multiprocessing.get_context("spawn")
        # Worker initializer pins one immutable plan copy per process; large
        # cohort metadata is not reserialized or rehashed at every chunk.
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                initializer=_worker_init, initargs=(str(run), plan)) as pool:
            futures = {}
            try:
                while pending or futures:
                    while pending and len(futures) < workers and not stopping():
                        instance = pending.pop(0)
                        futures[pool.submit(_worker_advance, instance)] = instance
                        submitted += 1
                    if not futures:
                        break
                    ready, _ = wait(futures, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in ready:
                        instance = futures.pop(future)
                        result = future.result()
                        count += 1
                        if result["done"]:
                            completed += 1
                        else:
                            pending.append(instance)
                        state()
            except KeyboardInterrupt:
                # Running chunks finish under the pool's context manager.
                # Their committed files are discovered on the next call.
                pass
        # Recount after drained workers, including interruption during a chunk.
        completed = sum(_done(run, plan, item) is not None for item in plan["instances"])
        return state()


def load_confirmation_features(run, L):
    """Load only committed fresh-H features, preserving the confirmation role."""
    run = Path(run).resolve()
    plan = _read(run / "plan.json")
    if plan.get("role") != "confirmation" or _read(run / "plan-sha256.json") != {"sha256": _digest(plan)}:
        raise ValueError("Confirmation feature plan integrity mismatch")
    selected = [item for item in plan["instances"] if item["L"] == L]
    if not selected:
        raise ValueError("Lattice size not in fresh confirmation cohort")
    arrays = []
    for instance in selected:
        if not _done(run, plan, instance):
            raise ValueError("Fresh confirmation size is incomplete")
        with np.load(_job(run, instance) / "records.npz", allow_pickle=False) as archive:
            keys = {"x_f", "x_g", "y", "fields", "hamiltonian_id", "hamiltonian_uid", "role"}
            if "x_exact" in archive:
                keys.add("x_exact")
            value = {key: archive[key] for key in keys}
        if (value["role"].item() != "confirmation" or value["hamiltonian_uid"].item() != instance["hamiltonian_uid"]
                or value["hamiltonian_id"].item() != instance["id"]):
            raise ValueError("Fresh confirmation feature identity mismatch")
        for key in ("x_f", "x_g", "y", "fields", "x_exact"):
            if key not in value:
                continue
            expected = ((3, L, L) if key.startswith("x_") else
                        (2, L, L) if key == "y" else (2*L*L,))
            if value[key].shape != expected or not np.isfinite(value[key]).all():
                raise ValueError("Fresh confirmation feature shape/value mismatch")
        if not np.array_equal(value["fields"], _fields(instance)):
            raise ValueError("Fresh confirmation fields differ from the sealed seed")
        if not np.array_equal(value["y"], np.abs(value["fields"]).reshape(L, L, 2).transpose(2, 0, 1).astype(np.float32)):
            raise ValueError("Fresh confirmation magnitude labels differ from fields")
        arrays.append(value)
    result = {key: np.stack([a[key] for a in arrays]) for key in ("x_f", "x_g", "y", "fields")}
    if "x_exact" in arrays[0]:
        result["x_exact"] = np.stack([a["x_exact"] for a in arrays])
    result.update(role="confirmation", L=L,
        hamiltonian_ids=np.array([item["id"] for item in selected]),
        hamiltonian_uids=np.array([item["hamiltonian_uid"] for item in selected]),
        plan_sha256=_digest(plan), test_payload_loaded=False, known_fields_for_features=True)
    return result
