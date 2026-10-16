#!/usr/bin/env python3
"""Resume a separate local/replica-exchange investigation of one non-test field."""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import signal
import sys
import time

import numpy as np

from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.statistics import autocorrelation_diagnostic
from quantum_lattice.simulation.replica_exchange import ReplicaExchangeChain
from quantum_lattice.simulation.solvable import StarMetropolisChain

VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    versions = {}
    for package in ("numpy", "numba", "llvmlite"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "versions": versions}


@contextmanager
def lock(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Already running: {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def positive(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_config(config):
    expected = {"schema_version", "reference_campaign", "output_root", "L",
                "hamiltonian_id", "flagged_star", "replicas", "seed", "backend",
                "lazy_probability", "total_sweeps", "burn_in_sweeps", "thinning_sweeps",
                "chunk_sweeps", "block_records", "tempering", "execution"}
    if set(config) != expected or config["schema_version"] != VERSION:
        raise ValueError("Unrecognized diagnostic configuration schema")
    for key in ("L", "replicas", "total_sweeps", "thinning_sweeps", "chunk_sweeps", "block_records"):
        positive(config[key], key)
    for key in ("flagged_star", "seed", "burn_in_sweeps"):
        positive(config[key], key, 0)
    if config["L"] < 3 or config["flagged_star"] >= config["L"] ** 2:
        raise ValueError("Invalid lattice or flagged star")
    if config["backend"] not in {"python", "numba"}:
        raise ValueError("Use an explicit python or numba backend")
    if isinstance(config["lazy_probability"], bool) or not 0 < config["lazy_probability"] < 1:
        raise ValueError("lazy_probability must be strictly between 0 and 1")
    count = config["total_sweeps"] - config["burn_in_sweeps"]
    if count <= 0 or count % config["thinning_sweeps"]:
        raise ValueError("Production sweeps must be positive and divisible by thinning_sweeps")
    if set(config["tempering"]) != {"levels", "swap_interval_sweeps"}:
        raise ValueError("Invalid tempering settings")
    positive(config["tempering"]["levels"], "levels", 2)
    positive(config["tempering"]["swap_interval_sweeps"], "swap_interval_sweeps")
    if set(config["execution"]) != {"workers"}:
        raise ValueError("Invalid execution settings")
    positive(config["execution"]["workers"], "workers")
    if not isinstance(config["hamiltonian_id"], str) or not config["hamiltonian_id"].startswith("hamiltonian_"):
        raise ValueError("Invalid Hamiltonian id")
    if "/" in config["hamiltonian_id"] or ".." in config["hamiltonian_id"]:
        raise ValueError("Invalid Hamiltonian id")
    return config


def _source_hashes(root, original):
    sources = dict(original)
    for name in ("experiments/l24_sampling_diagnostic/run.py",
                 "src/quantum_lattice/simulation/replica_exchange.py",
                 "src/quantum_lattice/data/sampling_diagnostic.py"):
        sources[name] = sha(root / name)
    return sources


def _check_files(root, values, kind):
    for name, expected in values.items():
        if sha(root / name) != expected:
            raise ValueError(f"{kind} hash differs: {name}")


def _artifact_directory(root, directory):
    directory = Path(directory).resolve()
    allowed = (root / "codex material/artifacts/estimator_validation").resolve()
    if directory == allowed or allowed not in directory.parents:
        raise ValueError("Diagnostic output must be a separate ignored estimator-validation artifact directory")
    return directory


def prepare(root, config, run, config_file=None):
    """Pin verified cached non-test inputs without reading any bank payload."""
    root, run = Path(root).resolve(), _artifact_directory(Path(root).resolve(), run)
    config = validate_config(config)
    if (run / "plan.json").exists():
        saved = load_plan(root, run)
        if saved["config"] != config:
            raise ValueError("Existing run settings differ; use a new run directory")
        return saved
    campaign = (root / config["reference_campaign"]).resolve()
    if run == campaign or campaign in run.parents:
        raise ValueError("Never write into the reference campaign")
    reference_file = campaign / "plan.json"
    if sha(reference_file) != read_json(campaign / "plan-sha256.json")["sha256"]:
        raise ValueError("Reference campaign plan hash differs")
    original = read_json(reference_file)
    _check_files(root, original["sources"], "Reference source")
    L, hid = config["L"], config["hamiltonian_id"]
    entry = original["banks"][str(L)]["entries"][hid]
    if entry["instance"]["split"] not in {"train", "validation"}:
        raise ValueError("Only declared non-test Hamiltonians are allowed")
    tasks = {(t["stage"], t["key"]): t for t in original["tasks"]}
    inputs, jobs, field_values = {}, [], None

    def pin(path):
        relative = str(Path(path).resolve().relative_to(root))
        inputs[relative] = sha(path)
        return relative

    pin(reference_file)
    pin(campaign / "plan-sha256.json")
    bankdir = root / original["banks"][str(L)]["directory"]
    for name in ("manifest.json", "config.json", "verification.json"):
        file = bankdir / name
        if sha(file) != original["inputs"][str(file.relative_to(root))]:
            raise ValueError("Reference bank metadata hash differs")
        pin(file)
    for replica in range(config["replicas"]):
        key = f"L{L}-{hid}-r{replica:02d}"
        task = tasks.get(("multistart", key))
        if task is None or task["L"] != L or task["id"] != hid:
            raise ValueError("Requested reference trajectory is unavailable")
        directory = campaign / "jobs/multistart" / key
        done = read_json(directory / "done.json")
        pin(directory / "done.json")
        for name in ("fields.npz", "field.json", "progress.json"):
            if sha(directory / name) != done["files"][name]:
                raise ValueError(f"Reference cached file hash differs: {key}/{name}")
            pin(directory / name)
        if read_json(directory / "field.json")["input_sha256"] != entry["sha256"]:
            raise ValueError("Cached field does not identify the verified Hamiltonian")
        progress = read_json(directory / "progress.json")
        if not progress["complete"] or progress["completed_sweeps"] != progress["spec"]["total_sweeps"]:
            raise ValueError("Reference trajectory is not complete")
        with np.load(directory / "fields.npz", allow_pickle=False) as data:
            fields = np.array(data["b_z"], dtype=np.float64)
        if fields.shape != (2 * L * L,) or not np.isfinite(fields).all():
            raise ValueError("Invalid cached fields")
        if field_values is not None and not np.array_equal(fields, field_values):
            raise ValueError("Reference replicas have different fields")
        field_values = fields
        lattice = square_lattice(L)
        StarMetropolisChain.from_snapshot(lattice, fields, progress["chain"],
                                          backend=config["backend"],
                                          lazy_probability=config["lazy_probability"])
        jobs.append({"key": f"local-r{replica:02d}", "method": "local", "replica": replica,
                     "initial_checkpoint": str((directory / "progress.json").relative_to(root)),
                     "initial_sweeps": progress["completed_sweeps"]})
        jobs.append({"key": f"tempered-r{replica:02d}", "method": "tempered", "replica": replica,
                     "seed": [config["seed"], L, entry["instance"]["index"], 401, replica],
                     "initial_sweeps": 0})
    if config_file is not None:
        config_file = Path(config_file).resolve()
        try:
            pin(config_file)
        except ValueError:
            raise ValueError("Config file must be inside the repository") from None
    sources = _source_hashes(root, original["sources"])
    if run.exists() and any(p.name != ".lock" for p in run.iterdir()):
        raise ValueError("Refusing to overwrite an unrecognized output directory")
    run.mkdir(parents=True, exist_ok=True)
    temporary = run / "fields.tmp.npz"
    np.savez(temporary, b_z=field_values)
    os.replace(temporary, run / "fields.npz")
    plan = {"version": VERSION, "config": config, "inputs": inputs,
            "sources": sources, "environment": environment(), "jobs": jobs,
            "instance": entry["instance"], "fields_sha256": sha(run / "fields.npz"),
            "scope": "diagnostic only; one selected non-test field; no exact L24 truth or automatic adoption"}
    atomic_json(run / "plan.json", plan)
    atomic_json(run / "plan-sha256.json", {"sha256": sha(run / "plan.json")})
    return plan


def load_plan(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    if sha(run / "plan.json") != read_json(run / "plan-sha256.json")["sha256"]:
        raise ValueError("Diagnostic plan hash differs")
    plan = read_json(run / "plan.json")
    if plan["version"] != VERSION or plan["environment"] != environment():
        raise ValueError("Diagnostic version or numerical environment differs")
    validate_config(plan["config"])
    _check_files(root, plan["sources"], "Source")
    _check_files(root, plan["inputs"], "Input")
    if sha(run / "fields.npz") != plan["fields_sha256"]:
        raise ValueError("Diagnostic field copy hash differs")
    return plan


def _read_checkpoint(directory, plan, job):
    envelope = read_json(directory / "checkpoint.json")
    progress = envelope["payload"]
    if digest(progress) != envelope["sha256"]:
        raise ValueError("Checkpoint hash differs")
    if progress["plan_sha256"] != digest(plan) or progress["job"] != job:
        raise ValueError("Checkpoint belongs to different settings/job")
    config = plan["config"]
    completed, retained = progress["completed_sweeps"], progress["retained_records"]
    positive(completed, "completed_sweeps", 0)
    positive(retained, "retained_records", 0)
    if completed > config["total_sweeps"] or retained != max(0, completed-config["burn_in_sweeps"]) // config["thinning_sweeps"]:
        raise ValueError("Checkpoint schedule counters disagree")
    return progress


def _records(directory, config, mode):
    shape = ((config["total_sweeps"]-config["burn_in_sweeps"]) // config["thinning_sweeps"],
             (2 * config["L"] ** 2 + 7) // 8)
    values = np.lib.format.open_memmap(directory / "states_packed.npy", mode=mode,
                                      dtype=np.uint8, shape=shape) if mode == "w+" else np.load(directory / "states_packed.npy", mmap_mode=mode, allow_pickle=False)
    if values.dtype != np.uint8 or values.shape != shape:
        raise ValueError("Record allocation differs")
    return values


def _prefix_hash(records, retained):
    return hashlib.sha256(np.ascontiguousarray(records[:retained]).tobytes()).hexdigest()


def _save_checkpoint(directory, progress):
    atomic_json(directory / "checkpoint.json", {"payload": progress, "sha256": digest(progress)})


def advance_job(root, run, plan, job):
    """One durable chunk. Resume discards only an uncommitted record tail."""
    root, run = Path(root), Path(run)
    config = plan["config"]
    directory = run / "jobs" / job["key"]
    with lock(directory):
        if (directory / "done.json").exists():
            for name, expected in read_json(directory / "done.json")["files"].items():
                if sha(directory / name) != expected:
                    raise ValueError(f"Completed output hash differs: {job['key']}/{name}")
            _read_checkpoint(directory, plan, job)
            return True
        with np.load(run / "fields.npz", allow_pickle=False) as data:
            fields = data["b_z"]
        lattice = square_lattice(config["L"])
        if (directory / "checkpoint.json").exists():
            progress = _read_checkpoint(directory, plan, job)
            if job["method"] == "local":
                chain = StarMetropolisChain.from_snapshot(lattice, fields, progress["chain"],
                            backend=config["backend"], lazy_probability=config["lazy_probability"])
                if chain.attempted != (job["initial_sweeps"] + progress["completed_sweeps"]) * lattice.n_vertices:
                    raise ValueError("Local checkpoint proposal counter disagrees")
            else:
                chain = ReplicaExchangeChain.from_snapshot(lattice, fields, progress["chain"], backend=config["backend"])
                if chain.sweeps != progress["completed_sweeps"]:
                    raise ValueError("Tempered checkpoint sweep counter disagrees")
            records = _records(directory, config, "r+")
            if _prefix_hash(records, progress["retained_records"]) != progress["records_sha256"]:
                raise ValueError("Committed records hash differs")
        else:
            if job["method"] == "local":
                initial = read_json(root / job["initial_checkpoint"])
                chain = StarMetropolisChain.from_snapshot(lattice, fields, initial["chain"],
                            backend=config["backend"], lazy_probability=config["lazy_probability"])
            else:
                chain = ReplicaExchangeChain(lattice, fields, job["seed"],
                    betas=np.linspace(0, 1, config["tempering"]["levels"]),
                    backend=config["backend"], lazy_probability=config["lazy_probability"],
                    swap_interval_sweeps=config["tempering"]["swap_interval_sweeps"], start="random")
            records = _records(directory, config, "w+")
            records.flush()
            progress = {"plan_sha256": digest(plan), "job": job, "completed_sweeps": 0,
                        "retained_records": 0, "wall_seconds": 0., "chain": chain.snapshot(),
                        "records_sha256": _prefix_hash(records, 0)}
            _save_checkpoint(directory, progress)
        started = time.monotonic()
        current, retained = progress["completed_sweeps"], progress["retained_records"]
        stop = min(config["total_sweeps"], current + config["chunk_sweeps"])
        burn, stride = config["burn_in_sweeps"], config["thinning_sweeps"]
        while current < stop:
            next_record = burn + (max(0, current-burn) // stride + 1) * stride
            target = min(stop, next_record)
            chain.advance(target-current)
            current = target
            if current > burn and (current-burn) % stride == 0:
                records[retained] = np.packbits(chain.spins == -1, bitorder="little")
                retained += 1
        records.flush()
        with (directory / "states_packed.npy").open("rb") as handle:
            os.fsync(handle.fileno())
        progress.update(completed_sweeps=current, retained_records=retained,
                        wall_seconds=progress["wall_seconds"] + time.monotonic()-started,
                        chain=chain.snapshot(), records_sha256=_prefix_hash(records, retained))
        _save_checkpoint(directory, progress)
        del records
        if current == config["total_sweeps"]:
            summarize_job(directory, plan, job, chain, fields)
            files = ("checkpoint.json", "states_packed.npy", "summary.json", "coordinates.npz")
            atomic_json(directory / "done.json", {"files": {name: sha(directory / name) for name in files}})
            return True
        return False


def summarize_job(directory, plan, job, chain, fields):
    config = plan["config"]
    progress = _read_checkpoint(directory, plan, job)
    packed = _records(directory, config, "r")[:progress["retained_records"]]
    lattice = square_lattice(config["L"])
    count, V, flagged = len(packed), lattice.n_vertices, config["flagged_star"]
    windows = {"whole": (0, count), "first_half": (0, count//2),
               "second_half": (count//2, count), "last_quarter": (3*count//4, count)}
    sums = {key: {method: np.zeros(V) for method in ("f", "g")} for key in windows}
    traces = {method: np.zeros(count) for method in ("f", "g")}
    scores = np.zeros(count)
    for start in range(0, count, 256):
        stop = min(count, start+256)
        states = 1 - 2*np.unpackbits(packed[start:stop], axis=1, count=lattice.n_edges,
                                    bitorder="little").astype(np.int8)
        h = np.zeros((len(states), V))
        for slot in range(lattice.vertex_edge_indices.shape[1]):
            h += states[:, lattice.vertex_edge_indices[:, slot]] * fields[lattice.vertex_edge_indices[:, slot]] * lattice.vertex_edge_mask[:, slot]
        u = np.exp(-np.abs(h))
        values = {"f": np.exp(-h), "g": 2*u/(1+u*u)}
        scores[start:stop] = np.sum(states*fields, axis=1)
        for method, value in values.items():
            traces[method][start:stop] = value[:, flagged]
            for name, (low, high) in windows.items():
                lo, hi = max(start, low), min(stop, high)
                if lo < hi:
                    sums[name][method] += value[lo-start:hi-start].sum(axis=0)
    arrays = {f"{method}_{name}": (value/(windows[name][1]-windows[name][0])
              if windows[name][1] > windows[name][0] else np.full(V, np.nan))
              for name, methods in sums.items() for method, value in methods.items()}
    arrays.update(flagged_f=traces["f"], flagged_g=traces["g"], target_score=scores,
                  retained_sweeps=config["burn_in_sweeps"] + np.arange(1, count+1)*config["thinning_sweeps"])
    temporary = directory / "coordinates.tmp.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, directory / "coordinates.npz")
    flagged_summary = {}
    for method, trace in traces.items():
        flagged_summary[method] = {name: float(trace[lo:hi].mean()) if hi > lo else None
                                  for name, (lo, hi) in windows.items()}
        block = config["block_records"]
        n = len(trace)//block
        blocks = trace[:n*block].reshape(n, block).mean(axis=1)
        flagged_summary[method]["empirical_block_se"] = float(blocks.std(ddof=1)/np.sqrt(n)) if n > 1 else None
    if job["method"] == "tempered":
        diagnostics = chain.diagnostics()
        attempts = config["total_sweeps"] * lattice.n_vertices * config["tempering"]["levels"]
    else:
        diagnostics = {"scope": "continued original local chain; no exchanges",
                       "attempted_including_previous": chain.attempted,
                       "accepted_including_previous": chain.accepted,
                       "lazy_including_previous": chain.lazy}
        attempts = config["total_sweeps"] * lattice.n_vertices
    atomic_json(directory / "summary.json", {"job": job, "records": count,
        "flagged_star": flagged, "flagged": flagged_summary,
        "empirical_wall_seconds_excluding_load_and_checkpoint": progress["wall_seconds"],
        "additional_local_proposal_attempts": attempts, "diagnostics": diagnostics,
        "limitations": "One selected Hamiltonian; no exact large-L target. Windows, block errors, exchanges and round trips do not certify convergence; PT has more proposals per sweep."})


def report(run, plan):
    config = plan["config"]
    rows, summaries, completed, sweeps = [], [], 0, 0
    for job in plan["jobs"]:
        directory = run / "jobs" / job["key"]
        n = _read_checkpoint(directory, plan, job)["completed_sweeps"] if (directory / "checkpoint.json").exists() else 0
        sweeps += n
        if (directory / "done.json").exists():
            for name, expected in read_json(directory / "done.json")["files"].items():
                if sha(directory / name) != expected:
                    raise ValueError(f"Completed output hash differs: {job['key']}/{name}")
            completed += 1
            summary = read_json(directory / "summary.json")
            summaries.append(summary)
            f, g = summary["flagged"]["f"], summary["flagged"]["g"]
            rows.append(f"|{job['key']}|{summary['records']}|{f['whole']:.6f}|{g['whole']:.6f}|{g['first_half'] if g['first_half'] is not None else 'n/a'}|{g['second_half'] if g['second_half'] is not None else 'n/a'}|")
    state = {"state": "complete" if completed == len(plan["jobs"]) else "incomplete",
             "completed_jobs": completed, "total_jobs": len(plan["jobs"]),
             "committed_sweeps": sweeps, "total_sweeps": len(plan["jobs"])*config["total_sweeps"]}
    atomic_json(run / "status.json", state)
    comparisons = compare_chains(run, plan)
    atomic_json(run / "aggregate-summary.json", {"status": state, "jobs": summaries,
                                               "between_chain_diagnostics": comparisons})
    text = f"""# L{config['L']} sampling diagnostic

Status: **{state['state']}**, {completed}/{len(plan['jobs'])} jobs complete;
{sweeps:,}/{state['total_sweeps']:,} committed global sweeps.

The selected {plan['instance']['split']} Hamiltonian is `{config['hamiltonian_id']}`.
The target is unchanged: one reference star orbit with probability proportional
to exp(sum b_e z_e). Local jobs continue their preserved original checkpoints;
tempered jobs start fresh independent random states at every temperature.
Reported local windows refer to additional sweeps after the previous trajectories.
The first {config['burn_in_sweeps']:,} additional/fresh sweeps are discarded in both.

|Job|Records|Star{config['flagged_star']} f|Star{config['flagged_star']} g|g first half|g second half|
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n\nBetween-chain window diagnostics, scalar ESS estimates and all-star mean ranges\nare saved in `aggregate-summary.json`. Missing jobs are explicitly counted;\nthese are conventional finite-window diagnostics, not exact precision measures.\n" + """

## Interpretation and limits

No exact large-lattice target is available. Agreement between runs, stable
windows, block standard errors, swap acceptance and round trips are diagnostic
evidence, not a proof of equilibration. This selected field is not a prevalence
estimate for the complete bank. The ordinary and tempered runs have equal
global-sweep counts, but the tempered method makes one temperature-level set
of local proposals each sweep: its proposal and wall-time budgets are larger.
Changing the estimator does not change the underlying trajectories. No bank,
training model or previous campaign output is overwritten.

## Files

- `plan.json` and `plan-sha256.json`: frozen settings, source/input hashes and instance identity.
- `fields.npz`: separate verified field copy; original inputs remain read-only.
- `status.json`, `aggregate-summary.json`: campaign counters and per-job diagnostics.
- `jobs/<job>/checkpoint.json`: exact chain and RNG state with committed record prefix hash.
- `jobs/<job>/states_packed.npy`: target-only states; little-endian bit1 means spin -1.
- `jobs/<job>/coordinates.npz`: all-star f/g means in whole/half/last-quarter windows;
  flagged-star f/g traces, target score trace and retained additional sweep positions.
- `jobs/<job>/summary.json`: flagged means and empirical block SE; all adjacent swap
  rates, walker visits and round trips for tempering. No acceptance edges are omitted.
- `jobs/<job>/done.json`: completed output hashes, verified before skipping a job.

Resume with the same `run.py run` command and configuration. Worker overrides
may change; scientific settings, inputs, numerical environment and source
changes are rejected for an existing run. Ctrl+C stops after active chunks.
"""
    (run / "RESULTS.md").write_text(text)
    return state


def compare_chains(run, plan):
    """Compare independent ensembles/continued trajectories; no truth is assumed."""
    result = {}
    for kind in ("local", "tempered"):
        available = [job for job in plan["jobs"] if job["method"] == kind
                     and (run / "jobs" / job["key"] / "done.json").exists()]
        detail = {"completed_chains": len(available), "expected_chains": plan["config"]["replicas"],
                  "windows": {}, "scalar_autocorrelation": {}}
        if not available:
            result[kind] = detail
            continue
        data = []
        for job in available:
            with np.load(run / "jobs" / job["key"] / "coordinates.npz", allow_pickle=False) as arrays:
                data.append({key: arrays[key] for key in arrays.files})
            detail["scalar_autocorrelation"][job["key"]] = {
                method: autocorrelation_diagnostic(data[-1][f"flagged_{method}"])
                for method in ("f", "g")}
        count = len(data[0]["flagged_g"])
        windows = {"whole": (0, count), "first_half": (0, count//2),
                   "second_half": (count//2, count), "last_quarter": (3*count//4, count)}
        for name, (lo, hi) in windows.items():
            detail["windows"][name] = {}
            for method in ("f", "g"):
                traces = np.asarray([arrays[f"flagged_{method}"][lo:hi] for arrays in data])
                coordinate_means = np.asarray([arrays[f"{method}_{name}"] for arrays in data])
                n = traces.shape[1]//2
                rhat = None
                if n >= 2 and len(data) >= 2:
                    halves = np.concatenate((traces[:, :n], traces[:, -n:]), axis=0)
                    within = float(halves.var(axis=1, ddof=1).mean())
                    mean_variance = float(halves.mean(axis=1).var(ddof=1))
                    rhat = (float(np.sqrt((n-1)/n + mean_variance/within)) if within > 0
                            else (1.0 if mean_variance == 0 else None))
                ranges = np.ptp(coordinate_means, axis=0) if len(data) > 1 else None
                detail["windows"][name][method] = {
                    "records_per_chain": hi-lo,
                    "flagged_means": [float(t.mean()) if len(t) else None for t in traces],
                    "conventional_split_rhat": rhat,
                    "all_star_max_chain_mean_range": float(ranges.max()) if ranges is not None else None,
                    "all_star_median_chain_mean_range": float(np.median(ranges)) if ranges is not None else None,
                    "all_star_largest_range_vertex": int(ranges.argmax()) if ranges is not None else None,
                }
        result[kind] = detail
    return result


def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def execute(root, run, plan, workers, max_chunks=None):
    paused = False
    def pause(signum, frame):
        nonlocal paused
        paused = True
    previous = signal.signal(signal.SIGINT, pause)
    pending = list(plan["jobs"])
    chunks, inflight, last_print = 0, {}, 0.
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_ignore_sigint) as pool:
            while pending or inflight:
                while pending and len(inflight) < workers and not paused and (max_chunks is None or chunks+len(inflight) < max_chunks):
                    job = pending.pop(0)
                    future = pool.submit(advance_job, str(root), str(run), plan, job)
                    inflight[future] = job
                if not inflight:
                    break
                finished, _ = wait(inflight, timeout=.25, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = inflight.pop(future)
                    done = future.result()
                    chunks += 1
                    if not done:
                        pending.append(job)
                if time.monotonic()-last_print >= 10 or not pending and not inflight:
                    state = report(run, plan)
                    print(f"{state['completed_jobs']}/{state['total_jobs']} jobs · {state['committed_sweeps']:,}/{state['total_sweeps']:,} sweeps" + (" · pausing after active chunks" if paused else ""), flush=True)
                    last_print = time.monotonic()
        return report(run, plan)
    finally:
        signal.signal(signal.SIGINT, previous)


def main(argv=None, *, root):
    ROOT = Path(root).resolve()
    default_config = ROOT / "configs/l24_sampling_diagnostic.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "prepare", "report"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--run-dir", type=Path, help="separate ignored artifact directory")
    parser.add_argument("--workers", type=int, help="CPU workers; may change when continuing")
    parser.add_argument("--sweeps", type=int, help="total/additional sweeps; new runs only")
    parser.add_argument("--max-chunks", type=int, help="pause after at most this many dispatched chunks")
    args = parser.parse_args(argv)
    if args.workers is not None:
        positive(args.workers, "workers")
    if args.max_chunks is not None:
        positive(args.max_chunks, "max_chunks")
    config = read_json(args.config)
    run = _artifact_directory(ROOT, args.run_dir or ROOT / config["output_root"])
    if args.command == "status":
        plan = load_plan(ROOT, run)
        state = {"state": "prepared", "completed_jobs": 0, "total_jobs": len(plan["jobs"])} if not (run / "status.json").exists() else read_json(run / "status.json")
        print(json.dumps(state, indent=2))
        print(f"Results: {run / 'RESULTS.md'}")
        return 0
    with lock(run):
        if args.sweeps is not None:
            if (run / "plan.json").exists():
                raise ValueError("--sweeps is allowed only for a new run directory")
            config["total_sweeps"] = positive(args.sweeps, "sweeps")
        # A one-time --sweeps override is stored in the immutable plan. Resume
        # that plan rather than comparing it with the unmodified template.
        plan = (load_plan(ROOT, run) if (run / "plan.json").exists()
                else prepare(ROOT, config, run, args.config))
        config = plan["config"]
        if args.command in {"prepare", "report"}:
            state = report(run, plan)
            print(json.dumps(state, indent=2))
            print(f"Results: {run / 'RESULTS.md'}")
            return 0
        workers = args.workers or config["execution"]["workers"]
        print(f"Sampling diagnostic: {run}\nCPU workers: {workers}; BLAS/OMP threads: 1 each.\nCtrl+C pauses after current chunks; repeat this command to continue.\nPT sweeps include all {config['tempering']['levels']} temperatures; budgets are not equal to local-only sweeps.\n", flush=True)
        state = execute(ROOT, run, plan, workers, args.max_chunks)
        print(f"{state['state']}: {state['completed_jobs']}/{state['total_jobs']} jobs.\nReport: {run / 'RESULTS.md'}")
    return 0
