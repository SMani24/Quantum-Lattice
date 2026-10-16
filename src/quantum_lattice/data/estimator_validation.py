"""Declared, resumable train/validation-only estimator experiments.

This campaign does not alter the generator or adopt a new feature bank. Each
completed task has a hash-verified commit marker. Fresh chains checkpoint after
a bounded sweep chunk; the coordinator alone handles interrupts.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import sys
import time

import numpy as np

from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.conditioned import (
    exact_reference, paired_estimators, summarize_records,
)
from quantum_lattice.simulation.estimator_validation import advance_chain, load_chain_records


STAGES = ("saved", "iid", "multistart", "stationary")
_WORKER_CONTEXT = None


@lru_cache(maxsize=8)
def _lattice(L):
    return square_lattice(L)


def _versions():
    return {name: importlib.metadata.version(name) for name in ("numpy", "numba", "llvmlite")}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_npz(path, arrays):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def campaign_lock(run):
    with (Path(run) / ".campaign.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this validation campaign is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("unsupported validation schema")
    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    integer(config["seed"], "seed", 0)
    sizes = config["lattice_sizes"]
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("lattice sizes must be nonempty and unique")
    for L in sizes:
        integer(L, "L", 3)
    if set(config["saved"]["exact_sizes"]) - {3, 4}:
        raise ValueError("exact square-lattice checks support L3/L4 only")
    limit = config["saved"]["limit_per_size"]
    if limit is not None:
        integer(limit, "saved limit")
    for section, keys in (("iid", ("fields_per_size", "replicates", "records")),
                          ("stationary", ("fields_per_size", "replicas", "total_sweeps")),
                          ("multistart", ("random_fields_per_size", "stress_fields_per_size"))):
        for key in keys:
            integer(config[section][key], f"{section}.{key}")
    m = config["multistart"]
    for key in ("replicas", "total_sweeps", "thinning_sweeps", "chunk_sweeps"):
        integer(m[key], key)
    integer(m["burn_in_sweeps"], "burn-in", 0)
    if m["replicas"] < 2 or config["stationary"]["replicas"] < 2:
        raise ValueError("at least two replicas are required")
    if not np.isfinite(m["lazy_probability"]) or not 0 <= m["lazy_probability"] < 1:
        raise ValueError("invalid lazy probability")
    if m["backend"] not in {"python", "numba"}:
        raise ValueError("backend must be python or numba")
    for total in (m["total_sweeps"], config["stationary"]["total_sweeps"]):
        if total <= m["burn_in_sweeps"] or (total-m["burn_in_sweeps"]) % m["thinning_sweeps"]:
            raise ValueError("production must be a positive multiple of thinning")
    for start, end in m["equal_budget_windows"]:
        integer(start, "window start", 0)
        integer(end, "window end")
        if not m["burn_in_sweeps"] <= start < end <= m["total_sweeps"]:
            raise ValueError("invalid equal-budget window")
        if (start-m["burn_in_sweeps"]) % m["thinning_sweeps"] or (end-start) % m["thinning_sweeps"]:
            raise ValueError("windows must align to retained records")
    for end in m["cumulative_ends"]:
        integer(end, "cumulative endpoint")
        if not m["burn_in_sweeps"] < end <= m["total_sweeps"] or (end-m["burn_in_sweeps"]) % m["thinning_sweeps"]:
            raise ValueError("invalid cumulative endpoint")
    if config["stationary"]["fields_per_size"] > config["iid"]["fields_per_size"]:
        raise ValueError("stationary fields must fit the declared iid panel")
    for n in config["statistics"]["block_sizes"]:
        integer(n, "block size")
    if not config["statistics"]["block_sizes"]:
        raise ValueError("at least one block size required")
    integer(config["statistics"]["probe_count"], "probe count", 0)
    integer(config["execution"]["workers"], "workers")
    reserve = config["execution"].get("minimum_free_gib", 4)
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not np.isfinite(reserve) or reserve < 0:
        raise ValueError("minimum_free_gib must be finite and nonnegative")
    return config


def _sources(root):
    names = ("data/estimator_validation.py", "data/state_storage.py",
             "simulation/estimator_validation.py", "simulation/solvable.py",
             "observables/conditioned.py", "observables/statistics.py",
             "lattice/__init__.py", "lattice/topology.py", "lattice/builders.py", "lattice/square.py")
    return {str(Path("src/quantum_lattice") / n): sha(root / "src/quantum_lattice" / n) for n in names}


def prepare(root, config):
    """Declare identities/seeds from metadata, without opening any payloads."""
    root = Path(root).resolve()
    validate_config(config)
    run = (root / config["output_root"]).resolve()
    allowed = (root / "codex material/artifacts/estimator_validation").resolve()
    if not run.is_relative_to(allowed) or run == allowed:
        raise ValueError("output must be beneath ignored estimator_validation artifacts")
    if run.exists():
        plan = load_plan(root, run)
        if plan["config"] != config:
            raise ValueError("saved scientific config differs; use a new output_root")
        validate_plan(root, plan)
        return run
    selected = _read(root / config["pilot_selection"])
    inputs = {config["pilot_selection"]: sha(root / config["pilot_selection"]),
              config["derivation_pdf"]: sha(root / config["derivation_pdf"])}
    banks, tasks = {}, []
    for L in config["lattice_sizes"]:
        directory = root / config["bank_root"] / f"L{L}/run"
        meta = {name: _read(directory / name) for name in ("manifest.json", "config.json", "verification.json")}
        if not meta["verification.json"].get("passed"):
            raise ValueError(f"L{L} bank has no passing saved storage verification")
        if meta["config.json"]["lattice"] != {"L": L, "type": "square_periodic"}:
            raise ValueError("bank lattice identity mismatch")
        manifest = meta["manifest.json"]
        all_items = manifest["instances"]
        if len({m["id"] for m in all_items}) != len(all_items):
            raise ValueError("duplicate bank instance IDs")
        eligible = [m for m in all_items if m["split"] in {"train", "validation"}]
        if any(not re.fullmatch(r"hamiltonian_\d+", m["id"]) for m in eligible):
            raise ValueError("invalid bank instance filename")
        hashes = {x["id"]: x for x in meta["verification.json"]["instances"]}
        bank = {"directory": str(directory.relative_to(root)), "entries": {}, "config": meta["config.json"]}
        for item in eligible:
            checked = hashes.get(item["id"], {})
            if checked.get("split") != item["split"] or not checked.get("sha256"):
                raise ValueError("non-test identity absent from storage verification")
            bank["entries"][item["id"]] = {"instance": item, "sha256": checked["sha256"]}
        banks[str(L)] = bank
        for name in meta:
            p = directory / name
            inputs[str(p.relative_to(root))] = sha(p)
        previous = {m["id"] for m in selected if m["L"] == L}
        stress = [m["id"] for m in selected if m["L"] == L and m["selection"] == "lowest saved scalar ESS"]
        stress = stress[:config["multistart"]["stress_fields_per_size"]]
        if len(stress) < config["multistart"]["stress_fields_per_size"] or any(s not in bank["entries"] for s in stress):
            raise ValueError("insufficient predeclared non-test stress fields")
        pool = sorted(set(bank["entries"]) - previous)
        count = max(config["multistart"]["random_fields_per_size"], config["iid"]["fields_per_size"] if L in (3, 4) else 0)
        if len(pool) < count:
            raise ValueError("insufficient previously uninspected non-test random fields")
        rng = np.random.default_rng([config["seed"], L, 71])
        random_ids = [pool[i] for i in rng.choice(len(pool), count, replace=False)]
        def add(stage, ID, stratum, replica=None):
            key = f"L{L}-{ID}" + (f"-r{replica:02d}" if replica is not None else "")
            tasks.append({"stage": stage, "key": key, "L": L, "id": ID,
                          "stratum": stratum, "replica": replica})
        census = sorted(bank["entries"])
        if config["saved"]["limit_per_size"] is not None:
            census = census[:config["saved"]["limit_per_size"]]
        for ID in census:
            add("saved", ID, "complete non-test bank" if config["saved"]["limit_per_size"] is None else "limited smoke subset")
        if L in (3, 4):
            for ID in random_ids[:config["iid"]["fields_per_size"]]:
                add("iid", ID, "random, excludes prior pilot selection")
            for ID in random_ids[:config["stationary"]["fields_per_size"]]:
                for replica in range(config["stationary"]["replicas"]):
                    add("stationary", ID, "random exact-stationary panel", replica)
        for stratum, IDs in (("random", random_ids[:config["multistart"]["random_fields_per_size"]]), ("stress", stress)):
            for ID in IDs:
                for replica in range(config["multistart"]["replicas"]):
                    add("multistart", ID, stratum, replica)
    tasks.sort(key=lambda t: (STAGES.index(t["stage"]), t["L"], t["key"]))
    plan = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "inputs": inputs, "sources": _sources(root), "banks": banks,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "versions": _versions(),
                            "platform": platform.platform()}, "tasks": tasks,
            "scope": "non-test known-field estimator validation; no production adoption or model training"}
    run.mkdir(parents=True, exist_ok=False)
    (run / "jobs").mkdir()
    atomic_json(run / "plan.json", plan)
    atomic_json(run / "plan-sha256.json", {"sha256": sha(run / "plan.json")})
    atomic_json(run / "status.json", {"state": "prepared", "total": len(tasks), "completed": 0})
    return run


def validate_plan(root, plan):
    if plan["environment"]["numpy"] != np.__version__ or plan["environment"]["python"] != platform.python_version():
        raise ValueError("Python/NumPy version changed; do not mix runtime versions in a campaign")
    if plan["environment"]["versions"] != _versions():
        raise ValueError("scientific package versions changed; use a new campaign")
    for relative, expected in {**plan["inputs"], **plan["sources"]}.items():
        if sha(Path(root) / relative) != expected:
            raise ValueError(f"pinned input/source changed: {relative}; use a new campaign")


def load_plan(root, run):
    run = Path(run).resolve()
    allowed = (Path(root) / "codex material/artifacts/estimator_validation").resolve()
    if run == allowed or not run.is_relative_to(allowed):
        raise ValueError("run must be inside estimator_validation artifact root")
    if sha(run / "plan.json") != _read(run / "plan-sha256.json")["sha256"]:
        raise ValueError("immutable plan hash mismatch")
    plan = _read(run / "plan.json")
    validate_plan(root, plan)
    return plan


def _input(root, plan, task):
    """Guard manifest split and byte identity before any payload is opened."""
    bank = plan["banks"][str(task["L"])]
    entry = bank["entries"].get(task["id"])
    if entry is None or entry["instance"]["split"] not in {"train", "validation"}:
        raise ValueError("payload is not a declared non-test Hamiltonian")
    directory = Path(root) / bank["directory"]
    current = _manifest(directory / "manifest.json", plan["inputs"][str((directory / "manifest.json").relative_to(root))])
    if current.get(task["id"]) != entry["instance"]:
        raise ValueError("manifest identity or split changed")
    path = directory / "instances" / (task["id"] + ".npz")
    if path.resolve().parent != (directory / "instances").resolve():
        raise ValueError("payload path escapes its bank")
    if sha(path) != entry["sha256"]:
        raise ValueError(f"original payload hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata["instance"] != entry["instance"]:
            raise ValueError("payload Hamiltonian metadata mismatch")
        fields = archive["b_z"].copy()
        states = decode_states(archive) if task["stage"] == "saved" else None
        old = {ch: archive[ch + "_mean"].copy() for ch in ("star", "pair")} if states is not None else None
    bc = bank["config"]["fields"]
    if bc["distribution"] != "uniform" or bc["channel"] != "z_only":
        raise ValueError("only declared z-only uniform-field banks are supported")
    expected = np.random.default_rng(np.random.SeedSequence(entry["instance"]["field_seed"])).uniform(
        bc["low"], bc["high"], len(fields))
    if not np.array_equal(fields, expected):
        raise ValueError("payload fields differ from their Hamiltonian seed")
    return fields, states, old


@lru_cache(maxsize=16)
def _manifest(path, expected):
    if sha(path) != expected:
        raise ValueError("bank manifest hash changed")
    return {m["id"]: m for m in _read(path)["instances"]}


def _commit(directory, summary, arrays, extra=()):
    atomic_npz(directory / "coordinates.npz", arrays)
    atomic_json(directory / "summary.json", summary)
    files = ["coordinates.npz", "summary.json", *extra]
    atomic_json(directory / "done.json", {"files": {n: sha(directory / n) for n in files}})


def completed(directory, *, verify=True):
    marker = Path(directory) / "done.json"
    if not marker.exists():
        return False
    record = _read(marker)
    if not record.get("files"):
        raise ValueError("invalid completed-task marker")
    for name, expected in record["files"].items():
        path = Path(directory) / name
        if not path.is_file() or (verify and sha(path) != expected):
            raise ValueError(f"completed result changed: {path}")
    return True


def _iid(lattice, fields, config, seed):
    exact = exact_reference(lattice, fields)
    if exact["max_identity_error"] > 1e-10:
        raise ValueError("exact estimator identities failed tolerance")
    tables = paired_estimators(lattice, fields, exact["states"])
    rng = np.random.default_rng(seed)
    reps, count = config["replicates"], config["records"]
    arrays, channels = {}, {}
    for ch in ("star", "pair"):
        channels[ch] = {}
        for method in ("f", "g"):
            arrays[f"{ch}_{method}_replicate_mean"] = np.empty((reps, len(exact["truth"][ch])))
            arrays[f"{ch}_{method}_replicate_se"] = np.empty_like(arrays[f"{ch}_{method}_replicate_mean"])
    for replica in range(reps):
        indices = rng.choice(len(exact["probabilities"]), size=count, p=exact["probabilities"])
        for ch in ("star", "pair"):
            for method in ("f", "g"):
                values = tables[ch][method][indices]
                arrays[f"{ch}_{method}_replicate_mean"][replica] = values.mean(axis=0)
                n = count // 45
                se = values[:n*45].reshape(n, 45, -1).mean(axis=1).std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(values.shape[1], np.nan)
                arrays[f"{ch}_{method}_replicate_se"][replica] = se
    for ch in ("star", "pair"):
        arrays[f"{ch}_truth"] = exact["truth"][ch]
        for method in ("f", "g"):
            means = arrays[f"{ch}_{method}_replicate_mean"]
            errors = means - exact["truth"][ch]
            channels[ch][method] = {
                "mse": float(np.mean(errors**2)), "mae": float(np.mean(abs(errors))),
                "squared_bias": float(np.mean(errors.mean(axis=0)**2)),
                "predicted_iid_mse": float(exact["variance"][ch][method].mean()/count),
                "coverage_45": float(np.mean(abs(errors) <= 1.96*arrays[f"{ch}_{method}_replicate_se"])) if count >= 90 else None}
    return {"channels": channels, "replicates": reps, "records": count,
            "max_identity_error": exact["max_identity_error"]}, arrays


def execute_task(root, run, plan, task):
    directory = Path(run) / "jobs" / task["stage"] / task["key"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/".task.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("task still active in another worker; wait before resuming") from error
        try:
            return _execute_task(root, run, plan, task)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _execute_task(root, run, plan, task):
    started = time.monotonic()
    directory = Path(run) / "jobs" / task["stage"] / task["key"]
    directory.mkdir(parents=True, exist_ok=True)
    if completed(directory):
        return True
    config = plan["config"]
    L = task["L"]
    lattice = _lattice(L)
    stats = config["statistics"]
    seed = [config["seed"], L, int(task["id"].split("_")[-1]), STAGES.index(task["stage"]), task["replica"] or 0]
    if task["stage"] in {"saved", "iid"}:
        fields, states, old = _input(root, plan, task)
        if task["stage"] == "iid":
            summary, arrays = _iid(lattice, fields, config["iid"], seed)
        else:
            reference = exact_reference(lattice, fields) if L in config["saved"]["exact_sizes"] else None
            if reference and reference["max_identity_error"] > 1e-10:
                raise ValueError("exact estimator identities failed tolerance")
            summary, arrays = summarize_records(lattice, fields, states,
                truth=reference["truth"] if reference else None,
                block_sizes=stats["block_sizes"], probe_count=stats["probe_count"], seed=seed)
            for ch in ("star", "pair"):
                if not np.allclose(arrays[f"{ch}_f_mean"], old[ch], rtol=1e-12, atol=1e-13):
                    raise ValueError("reconstructed original feature differs from saved mean")
            summary["max_identity_error"] = reference["max_identity_error"] if reference else None
        _commit(directory, {**task, **summary, "elapsed_seconds": time.monotonic()-started}, arrays)
        return True
    # Cache the verified field alone, avoiding a payload reread for every chunk.
    if not (directory / "field.json").exists():
        fields, _, _ = _input(root, plan, task)
        atomic_npz(directory / "fields.npz", {"b_z": fields})
        atomic_json(directory / "field.json", {"sha256": sha(directory / "fields.npz"), "input_sha256": plan["banks"][str(L)]["entries"][task["id"]]["sha256"]})
    field_record = _read(directory / "field.json")
    if field_record["input_sha256"] != plan["banks"][str(L)]["entries"][task["id"]]["sha256"]:
        raise ValueError("cached field belongs to another input")
    if sha(directory / "fields.npz") != field_record["sha256"]:
        raise ValueError("cached field changed")
    with np.load(directory / "fields.npz", allow_pickle=False) as archive:
        fields = archive["b_z"]
    m = config["multistart"]
    stationary = task["stage"] == "stationary"
    spec = {k: m[k] for k in ("backend", "lazy_probability", "burn_in_sweeps", "total_sweeps", "thinning_sweeps", "chunk_sweeps")}
    spec.update(seed=seed, start="stationary" if stationary else ("plus" if task["replica"] == 0 else "random"))
    if stationary:
        spec["total_sweeps"] = config["stationary"]["total_sweeps"]
    if not advance_chain(directory, lattice, fields, spec):
        return False
    states = load_chain_records(directory, lattice)
    reference = exact_reference(lattice, fields) if L <= 4 else None
    if reference and reference["max_identity_error"] > 1e-10:
        raise ValueError("exact estimator identities failed tolerance")
    windows = [("stationary", m["burn_in_sweeps"], spec["total_sweeps"])] if stationary else (
        [(f"window_{a}_{b}", a, b) for a, b in m["equal_budget_windows"]] +
        [(f"cumulative_{end}", m["burn_in_sweeps"], end) for end in m["cumulative_ends"]])
    summaries, all_arrays = {}, {}
    for label, begin, end in windows:
        start = (begin-m["burn_in_sweeps"])//m["thinning_sweeps"]
        stop = (end-m["burn_in_sweeps"])//m["thinning_sweeps"]
        summary, arrays = summarize_records(lattice, fields, states[start:stop],
            truth=reference["truth"] if reference else None,
            block_sizes=stats["block_sizes"], probe_count=stats["probe_count"], seed=seed)
        summaries[label] = summary
        all_arrays.update({label+"__"+k: v for k, v in arrays.items()})
    _commit(directory, {**task, "windows": summaries, "start": spec["start"],
                       "sampler": spec, "elapsed_final_chunk_seconds": time.monotonic()-started}, all_arrays,
            extra=("states_packed.npy", "progress.json", "fields.npz", "field.json"))
    return True


def _worker_init(root, run):
    global _WORKER_CONTEXT
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER_CONTEXT = (root, run, load_plan(root, run))


def _worker_task(task):
    return execute_task(*_WORKER_CONTEXT, task)


def run_campaign(root, run, *, workers=4, stage="all", max_steps=None):
    """Stop after the current bounded tasks on Ctrl+C; rerunning continues."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if stage not in (*STAGES, "all"):
        raise ValueError("unknown stage")
    run = Path(run).resolve()
    with campaign_lock(run):
        plan = load_plan(root, run)
        selected = [t for t in plan["tasks"] if stage == "all" or t["stage"] == stage]
        print(f"Checking {len(selected):,} task markers; no bank payloads opened for this scan.", flush=True)
        pending = deque(t for t in selected if not completed(run/"jobs"/t["stage"]/t["key"]))
        initial = len(selected)-len(pending)
        done, steps, stop, failure, pause_reason = initial, 0, False, None, None
        pause_announced = False
        start = time.monotonic()
        previous = signal.getsignal(signal.SIGINT)
        def pause(signum, frame):
            nonlocal stop, pause_reason
            # No I/O in the handler: SIGINT can arrive during a terminal write.
            pause_reason = "keyboard interrupt"
            stop = True
        signal.signal(signal.SIGINT, pause)
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init, initargs=(str(root), str(run))) as pool:
                active = {}
                stage_start, stage_done, last_print = {}, Counter(), 0
                while pending or active:
                    if stop and not pause_announced:
                        print("\nPausing: finishing active tasks/checkpoint chunks. Please wait.", flush=True)
                        pause_announced = True
                    reserve = plan["config"]["execution"].get("minimum_free_gib", 4)*1024**3
                    if pending and not stop and shutil.disk_usage(run).free < reserve:
                        pause_reason, stop = "free disk space below configured reserve", True
                        print("Pausing: free disk space is below the configured reserve.", flush=True)
                        pause_announced = True
                    while pending and len(active) < workers and not stop:
                        task = pending.popleft()
                        stage_start.setdefault(task["stage"], time.monotonic())
                        active[pool.submit(_worker_task, task)] = task
                    if not active:
                        break
                    ready, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in ready:
                        task = active.pop(future)
                        try:
                            finished = future.result()
                        except Exception as error:
                            failure, stop = f"{task['stage']}/{task['key']}: {error}", True
                            continue
                        steps += 1
                        if finished:
                            done += 1
                            stage_done[task["stage"]] += 1
                        else:
                            # Keep active chains progressing rather than starting every chain at once.
                            pending.appendleft(task)
                        elapsed = time.monotonic()-start
                        now = time.monotonic()
                        if now-last_print >= 2 or done == len(selected):
                            stage_left = sum(t["stage"] == task["stage"] for t in pending) + sum(t["stage"] == task["stage"] for t in active.values())
                            fresh_done = stage_done[task["stage"]]
                            eta = stage_left*(now-stage_start[task["stage"]])/fresh_done if fresh_done else None
                            rate = f" | rough current-stage ETA {eta/3600:.2f}h" if eta is not None else ""
                            print(f"{done:,}/{len(selected):,} complete | {task['stage']} {task['key']} {'done' if finished else 'checkpoint'} | {elapsed/60:.1f}min{rate}", flush=True)
                            last_print = now
                        atomic_json(run/"status.json", {"state": "pausing" if stop else "running", "completed_selected": done,
                                    "total_selected": len(selected), "stage": stage, "elapsed_session_seconds": elapsed,
                                    "last_task": task, "failure": failure})
                        if max_steps is not None and steps >= max_steps:
                            stop = True
                    if stop and not active:
                        break
        finally:
            signal.signal(signal.SIGINT, previous)
        if stop and not pause_announced:
            print("Pausing: active work is saved; refreshing the partial report.", flush=True)
        report = summarize_campaign(run, plan)
        state = "failed" if failure else ("complete" if report["completed"] == report["total"] else "paused")
        atomic_json(run/"status.json", {"state": state, **report, "failure": failure, "pause_reason": pause_reason})
        if failure:
            raise RuntimeError(failure)
        return {"state": state, **report}


def _rhat(arrays):
    means = np.concatenate([a[0] for a in arrays], axis=0)
    variances = np.concatenate([a[1] for a in arrays], axis=0)
    n = arrays[0][2]
    W = variances.mean(axis=0)
    B = n * means.var(axis=0, ddof=1)
    ratio = np.divide((n-1)/n*W+B/n, W, out=np.full_like(W, np.nan), where=W>0)
    values = np.sqrt(np.where((W == 0) & (B > 0), np.inf, ratio))
    finite = values[np.isfinite(values)]
    return {"max_finite_rhat": float(finite.max()) if len(finite) else None,
            "above_1_1": int(np.sum(values > 1.1)), "undefined": int(np.sum(np.isnan(values))),
            "infinite": int(np.sum(np.isinf(values))), "coordinates": len(values)}


def summarize_campaign(run, plan=None):
    """Aggregate completed tasks only; partial campaigns are labeled partial."""
    run = Path(run)
    plan = _read(run/"plan.json") if plan is None else plan
    counts, rows, groups = Counter(), [], {}
    exact_residuals = []
    for task in plan["tasks"]:
        directory = run/"jobs"/task["stage"]/task["key"]
        if not completed(directory, verify=False):
            continue
        counts[task["stage"]] += 1
        result = _read(directory/"summary.json")
        if result.get("max_identity_error") is not None:
            exact_residuals.append(result["max_identity_error"])
        windows = result.get("windows", {"saved" if task["stage"] == "saved" else "iid": result})
        for window, summary in windows.items():
            for ch in ("star", "pair"):
                for method in ("f", "g"):
                    metrics = summary["channels"][ch][method]
                    row = {k: task[k] for k in ("stage", "L", "id", "stratum", "replica")}
                    row.update(window=window, channel=ch, method=method)
                    row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None})
                    for metric in ("ess", "tau_records"):
                        values = [p[metric] for p in metrics.get("autocorrelation", [])
                                  if p["kind"] == "fixed-seed random coordinate" and p.get(metric) is not None]
                        row["probe_"+metric+"_median"] = float(np.median(values)) if values else None
                        row["probe_"+metric+"_min"] = float(np.min(values)) if values else None
                        row["probe_"+metric+"_max"] = float(np.max(values)) if values else None
                        spatial = [p.get(metric) for p in metrics.get("autocorrelation", []) if p["kind"] == "spatial mean"]
                        row["spatial_"+metric] = spatial[0] if spatial else None
                    if method == "g":
                        row.update(summary["channels"][ch].get("comparison", {}))
                    rows.append(row)
        if task["stage"] == "multistart":
            groups.setdefault((task["L"], task["id"]), []).append((task, directory))
    report = {"completed": sum(counts.values()), "total": len(plan["tasks"]),
              "exact_identity_checks": len(exact_residuals),
              "max_identity_error": max(exact_residuals) if exact_residuals else None,
              "by_stage": {s: {"completed": counts[s], "total": sum(t["stage"] == s for t in plan["tasks"])} for s in STAGES}}
    columns = ["stage", "L", "id", "stratum", "replica", "window", "channel", "method"]
    columns += sorted(set().union(*(r.keys() for r in rows))-set(columns)) if rows else []
    tmp = run/"per-hamiltonian.csv.tmp"
    with tmp.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, run/"per-hamiltonian.csv")
    # Average replicas within H first; never treat edges/replicas as H replicates.
    buckets = {}
    for row in rows:
        key = tuple(row[k] for k in ("stage", "L", "stratum", "window", "channel", "method"))
        buckets.setdefault(key, {}).setdefault(row["id"], []).append(row)
    aggregates = []
    for key, hamiltonians in sorted(buckets.items()):
        entry = dict(zip(("stage", "L", "stratum", "window", "channel", "method"), key))
        entry["hamiltonians"] = len(hamiltonians)
        for metric in columns[8:]:
            field_values = [np.mean([r[metric] for r in rs if r.get(metric) is not None])
                            for rs in hamiltonians.values() if any(r.get(metric) is not None for r in rs)]
            if field_values:
                entry[metric] = float(np.mean(field_values))
        aggregates.append(entry)
    atomic_json(run/"aggregate-summary.json", {"progress": report, "rows": aggregates,
                "weighting": "equal Hamiltonian weight; replicas averaged within Hamiltonian; stress separate"})
    rhats = []
    for (L, ID), items in groups.items():
        if len(items) != plan["config"]["multistart"]["replicas"]:
            continue
        loaded = []
        for task, directory in items:
            with np.load(directory/"coordinates.npz", allow_pickle=False) as archive:
                loaded.append({k: archive[k] for k in archive.files})
        summaries = _read(items[0][1]/"summary.json")["windows"]
        for window, summary in summaries.items():
            for ch in ("star", "pair"):
                for method in ("f", "g"):
                    prefix = f"{window}__{ch}_{method}"
                    data = [(a[prefix+"_half_mean"], a[prefix+"_half_variance"], summary["records"]//2) for a in loaded]
                    rhats.append({"L": L, "id": ID, "stratum": items[0][0]["stratum"], "window": window,
                                  "channel": ch, "method": method, **_rhat(data)})
    atomic_json(run/"multistart-rhat.json", {"rows": rhats, "method": "conventional split-Rhat, not rank-normalized; diagnostics cannot certify mixing"})
    complete = report["completed"] == report["total"]
    text = ["# Conditioned-estimator validation", "", f"State: **{'complete' if complete else 'partial'}**; {report['completed']:,}/{report['total']:,} tasks complete.", "",
        "This is known-field simulation-estimator validation, not experimental measurements or neural-network performance. No estimator adoption, bank replacement or model retraining occurs.", "",
        f"Exact-identity tasks checked: {report['exact_identity_checks']:,}; maximum residual: {report['max_identity_error']}.", "",
        "## Progress", "", "|Stage|Complete|Planned|", "|---|---:|---:|"]
    text += [f"|{s}|{report['by_stage'][s]['completed']}|{report['by_stage'][s]['total']}|" for s in STAGES]
    text += ["", "## Saved-chain exact errors", "", "MSE means observable error. Rows without exact truth are omitted. Hamiltonians receive equal weight.", "",
             "|L|Channel|Original MSE|Conditioned MSE|Reduction|", "|---:|---|---:|---:|---:|"]
    for a in aggregates:
        if a["stage"] == "saved" and a["method"] == "f" and a.get("mse") is not None:
            b = next(x for x in aggregates if all(x[k] == a[k] for k in ("stage", "L", "stratum", "window", "channel")) and x["method"] == "g")
            reduction = 100*(1-b["mse"]/a["mse"]) if a["mse"] else 0
            text.append(f"|{a['L']}|{a['channel']}|{a['mse']:.6g}|{b['mse']:.6g}|{reduction:.2f}%|")
    for stage, title in (("iid", "Independent-draw reference"), ("stationary", "Repeated stationary finite-length MCMC"), ("multistart", "Fresh-chain exact errors")):
        text += ["", "## "+title, "", "Replicas are averaged within Hamiltonian before averaging Hamiltonians. Partial panels remain provisional.", "",
                 "|L|Stratum|Window|Channel|Hamiltonians|Original MSE|Conditioned MSE|", "|---:|---|---|---|---:|---:|---:|"]
        for a in aggregates:
            if a["stage"] != stage or a["method"] != "f" or a.get("mse") is None:
                continue
            b = next(x for x in aggregates if all(x[k] == a[k] for k in ("stage", "L", "stratum", "window", "channel")) and x["method"] == "g")
            text.append(f"|{a['L']}|{a['stratum']}|{a['window']}|{a['channel']}|{a['hamiltonians']}|{a['mse']:.6g}|{b['mse']:.6g}|")
    text += ["", "## Interpretation and files", "",
        "The iid theorem does not order Markov-chain asymptotic variances. Repeated exact-stationary chains test finite-length MCMC error; independent starts and equal-budget later windows examine sensitivity to initialization. Above L4 there is no exact target. A passing Rhat or smaller empirical block SE cannot prove adequate mixing. Random and predeclared stress fields remain separate. This full census describes the generated non-test population, not every possible Hamiltonian family.", "",
        "- [Immutable settings, identities, seeds and input/source hashes](plan.json).",
        "- [Progress record](status.json).",
        "- [Per-Hamiltonian/channel/method/window metrics](per-hamiltonian.csv).",
        "- [Aggregate summaries](aggregate-summary.json).",
        "- [Conventional split-Rhat with degeneracy counts](multistart-rhat.json).",
        "- [Task outputs](jobs/): each task contains summary.json, compact coordinate arrays and hash commit marker; fresh jobs also contain packed states, initialization and exact RNG checkpoints.", "",
        "Completing this script is not approval to replace data or retrain. Return these outputs for scientific review, precision/coverage comparisons and professor-ready figures.", ""]
    (run/"RESULTS.md").write_text("\n".join(text))
    return report
