"""Bounded, resumable validation chains without modifying a research bank.

Each call commits at most ``chunk_sweeps`` sweeps. Packed records are written
and flushed before the atomic JSON checkpoint advances its committed count.
After interruption an uncommitted tail is ignored and overwritten. Only one
worker may own a job directory at a time; the campaign controls dispatch.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from quantum_lattice.lattice import Lattice
from quantum_lattice.simulation.solvable import StarMetropolisChain, _fields


_VERSION = 1
_SPEC_KEYS = {"seed", "start", "backend", "lazy_probability", "burn_in_sweeps",
              "total_sweeps", "thinning_sweeps", "chunk_sweeps"}
_INITIALIZATION_DOMAIN = (20261016, 119)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        raise ValueError(f"chain spec must contain exactly {sorted(_SPEC_KEYS)}")
    result = dict(spec)
    seed = result["seed"]
    if isinstance(seed, (list, tuple)):
        if not seed:
            raise ValueError("seed must be a nonempty list of nonnegative integers or an integer")
        result["seed"] = [_integer(value, "seed entry", 0) for value in seed]
    else:
        result["seed"] = _integer(seed, "seed", 0)
    if not isinstance(result["start"], str) or result["start"] not in {"plus", "random", "stationary"}:
        raise ValueError("start must be plus, random, or stationary")
    if not isinstance(result["backend"], str) or result["backend"] not in {"auto", "python", "numba"}:
        raise ValueError("backend must be auto, python, or numba")
    lazy = result["lazy_probability"]
    if isinstance(lazy, (bool, np.bool_)) or not isinstance(lazy, (float, int, np.floating, np.integer)) or not np.isfinite(lazy) or not 0 <= lazy < 1:
        raise ValueError("lazy_probability must be finite and in [0, 1)")
    result["lazy_probability"] = float(lazy)
    for key, minimum in (("burn_in_sweeps", 0), ("total_sweeps", 1),
                         ("thinning_sweeps", 1), ("chunk_sweeps", 1)):
        result[key] = _integer(result[key], key, minimum)
    production = result["total_sweeps"] - result["burn_in_sweeps"]
    if production <= 0 or production % result["thinning_sweeps"]:
        raise ValueError("total minus burn-in sweeps must be positive and divisible by thinning_sweeps")
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_fields(fields: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(fields, dtype="<f8").tobytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(_canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _job_lock(job_dir: Path):
    import fcntl

    with (job_dir / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("validation chain is already running in this job directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _initial_chain(lattice: Lattice, fields: np.ndarray, spec: dict) -> StarMetropolisChain:
    chain = StarMetropolisChain(lattice, fields, spec["seed"], backend=spec["backend"],
                                lazy_probability=spec["lazy_probability"])
    # Domain separation leaves the proposal RNG untouched by initialization.
    initial_rng = np.random.default_rng(np.random.SeedSequence(
        spec["seed"], spawn_key=_INITIALIZATION_DOMAIN))
    if spec["start"] == "random":
        theta = initial_rng.choice(np.array([-1, 1], dtype=np.int8), size=lattice.n_vertices)
        ends = lattice.edge_vertex_indices
        chain.spins[:] = theta[ends[:, 0]] * theta[ends[:, 1]]
    elif spec["start"] == "stationary":
        from quantum_lattice.observables.conditioned import exact_reference

        reference = exact_reference(lattice, fields, max_vertices=16)
        index = initial_rng.choice(len(reference["probabilities"]), p=reference["probabilities"])
        chain.spins[:] = reference["states"][index]
    return chain


def _layout(lattice: Lattice, spec: dict) -> tuple[int, int]:
    return ((spec["total_sweeps"] - spec["burn_in_sweeps"]) // spec["thinning_sweeps"],
            (lattice.n_edges + 7) // 8)


def _read_progress(job_dir: Path, lattice: Lattice, fields: np.ndarray, spec: dict) -> tuple[dict, StarMetropolisChain]:
    progress = json.loads((job_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("version") != _VERSION or progress.get("numpy_version") != np.__version__:
        raise ValueError("validation checkpoint version or NumPy version differs")
    if progress.get("spec") != spec or progress.get("spec_sha256") != hashlib.sha256(_canonical(spec).encode()).hexdigest():
        raise ValueError("validation checkpoint spec differs")
    if progress.get("fields_sha256") != _hash_fields(fields):
        raise ValueError("validation checkpoint fields differ")
    if progress.get("n_edges") != lattice.n_edges or progress.get("n_vertices") != lattice.n_vertices:
        raise ValueError("validation checkpoint topology size differs")
    completed = _integer(progress.get("completed_sweeps"), "completed_sweeps", 0)
    retained = _integer(progress.get("retained_records"), "retained_records", 0)
    if completed > spec["total_sweeps"] or retained != max(0, completed - spec["burn_in_sweeps"]) // spec["thinning_sweeps"]:
        raise ValueError("validation checkpoint schedule counters disagree")
    if progress.get("complete") != (completed == spec["total_sweeps"]):
        raise ValueError("validation checkpoint completion flag disagrees")
    if progress.get("total_records") != _layout(lattice, spec)[0]:
        raise ValueError("validation checkpoint record allocation differs")
    if progress.get("initialization_domain") != list(_INITIALIZATION_DOMAIN):
        raise ValueError("validation checkpoint initialization seed domain differs")
    chain = StarMetropolisChain.from_snapshot(lattice, fields, progress["chain"],
                                              backend=spec["backend"],
                                              lazy_probability=spec["lazy_probability"])
    if chain.attempted != completed * lattice.n_vertices:
        raise ValueError("validation checkpoint attempt counter disagrees with sweeps")
    return progress, chain


def _open_records(job_dir: Path, lattice: Lattice, spec: dict, mode: str):
    records = np.lib.format.open_memmap(job_dir / "states_packed.npy", mode=mode)
    if records.dtype != np.uint8 or records.shape != _layout(lattice, spec):
        raise ValueError("validation packed-record allocation differs")
    return records


def advance_chain(job_dir: Path, lattice: Lattice, fields: Any, spec: dict) -> bool:
    """Advance one bounded chunk, atomically committing its RNG and records.

    ``total_sweeps`` includes burn-in. Records occur at ``burn_in + k*stride``
    for positive integers k, with each state copied after that sweep. Changing
    any spec entry for an existing job is rejected. Initialization uses an
    independent, domain-separated seed; stationary starts require V<=16.
    """
    job_dir, spec = Path(job_dir), _spec(spec)
    field_values = _fields(lattice, fields)
    job_dir.mkdir(parents=True, exist_ok=True)
    with _job_lock(job_dir):
        if (job_dir / "progress.json").exists():
            progress, chain = _read_progress(job_dir, lattice, field_values, spec)
        else:
            chain = _initial_chain(lattice, field_values, spec)
            records = np.lib.format.open_memmap(job_dir / "states_packed.npy", mode="w+",
                                                dtype=np.uint8, shape=_layout(lattice, spec))
            records.flush()
            del records
            with (job_dir / "states_packed.npy").open("rb") as handle:
                os.fsync(handle.fileno())
            progress = {
                "version": _VERSION, "numpy_version": np.__version__, "spec": spec,
                "spec_sha256": hashlib.sha256(_canonical(spec).encode()).hexdigest(),
                "fields_sha256": _hash_fields(field_values), "n_edges": lattice.n_edges,
                "n_vertices": lattice.n_vertices, "total_records": _layout(lattice, spec)[0],
                "completed_sweeps": 0, "retained_records": 0, "complete": False,
                "initialization_domain": list(_INITIALIZATION_DOMAIN), "chain": chain.snapshot(),
            }
            _atomic_json(job_dir / "progress.json", progress)
        records = _open_records(job_dir, lattice, spec, "r+")
        try:
            if progress["complete"]:
                return True
            current, retained = progress["completed_sweeps"], progress["retained_records"]
            stop = min(spec["total_sweeps"], current + spec["chunk_sweeps"])
            burn, stride = spec["burn_in_sweeps"], spec["thinning_sweeps"]
            while current < stop:
                next_record = burn + (max(0, current - burn) // stride + 1) * stride
                target = min(stop, next_record)
                chain.advance(target - current)
                current = target
                if current > burn and (current - burn) % stride == 0:
                    records[retained] = np.packbits(chain.spins == -1, bitorder="little")
                    retained += 1
            records.flush()
            with (job_dir / "states_packed.npy").open("rb") as handle:
                os.fsync(handle.fileno())
            progress.update(completed_sweeps=current, retained_records=retained,
                            complete=current == spec["total_sweeps"], chain=chain.snapshot())
            _atomic_json(job_dir / "progress.json", progress)
            return progress["complete"]
        finally:
            del records


def load_chain_records(job_dir: Path, lattice: Lattice) -> np.ndarray:
    """Decode committed production records only, ignoring any crash-written tail.

    This also permits inspecting an incomplete job. Campaign evaluation should
    require its completion flag separately before treating it as a full run.
    """
    job_dir = Path(job_dir)
    with _job_lock(job_dir):
        saved = json.loads((job_dir / "progress.json").read_text(encoding="utf-8"))
        spec = _spec(saved.get("spec"))
        fields = _fields(lattice, saved.get("chain", {}).get("fields"))
        progress, _ = _read_progress(job_dir, lattice, fields, spec)
        records = _open_records(job_dir, lattice, spec, "r")
        try:
            committed = records[:progress["retained_records"]]
            remainder = lattice.n_edges % 8
            if remainder and np.any(committed[:, -1] >> remainder):
                raise ValueError("validation packed records have nonzero padding bits")
            bits = np.unpackbits(committed, axis=-1, count=lattice.n_edges, bitorder="little")
            return 1 - 2 * bits.astype(np.int8)
        finally:
            del records
