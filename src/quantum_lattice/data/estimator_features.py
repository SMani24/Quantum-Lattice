"""Versioned, paired f/g/exact views of completed non-test estimator summaries.

The original bank remains authoritative for identities, fields, labels and group
membership. No spin arrays or final-test payloads are loaded. Exact features are
provided for crossed evaluation, not as an additional training treatment.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from quantum_lattice.data.estimator_validation import load_plan, sha
from quantum_lattice.data.mlp_data import EdgeRows, LearningData, load_learning_data
from quantum_lattice.lattice import square_lattice

FEATURE_VERSION = "paired-estimator-edge-v1"
VIEWS = ("f", "g", "exact")
SPLITS = ("train", "validation")


def _read(path):
    return json.loads(Path(path).read_text())


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("feature provenance path escapes its root")
    return path


def _numeric(archive, key, shape):
    result = archive[key]
    if result.shape != shape or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"invalid feature coordinate: {key}")
    return result


def _source_hashes(repo_root):
    names = ("data/estimator_features.py", "data/mlp_data.py", "data/generation.py")
    return {str(Path("src/quantum_lattice") / name): sha(repo_root / "src/quantum_lattice" / name)
            for name in names}


def _job_records(run, task):
    key = task["key"]
    if not re.fullmatch(r"L[34]-hamiltonian_\d+", key):
        raise ValueError("invalid saved task identity")
    directory = _safe(run, "jobs/saved/" + key)
    marker = _read(directory / "done.json")
    if set(marker.get("files", {})) != {"coordinates.npz", "summary.json"}:
        raise ValueError("saved task must commit coordinates and summary only")
    files = marker["files"]
    for name, expected in files.items():
        if sha(directory / name) != expected:
            raise ValueError(f"completed saved task checksum mismatch: {key}/{name}")
    summary = _read(directory / "summary.json")
    if any(summary.get(name) != task[name] for name in ("stage", "key", "L", "id", "stratum", "replica")):
        raise ValueError("saved task summary identity mismatch")
    residual = summary.get("max_identity_error")
    if not isinstance(residual, (int, float)) or not np.isfinite(residual) or residual > 1e-10 or residual < 0:
        raise ValueError("saved task lacks a passing exact identity check")
    return directory, {"done_sha256": sha(directory / "done.json"), "files": files}


def _verify_size(directory, request_sha):
    manifest = _read(directory / "manifest.json")
    if manifest.get("feature_version") != FEATURE_VERSION or manifest.get("request_sha256") != request_sha:
        raise ValueError("feature size cache belongs to a different frozen request")
    if manifest.get("schema_version") != 1 or manifest.get("archive") != "views.npz":
        raise ValueError("unsupported feature cache schema")
    if sha(directory / "views.npz") != manifest.get("archive_sha256"):
        raise ValueError("feature cache archive checksum mismatch")
    if _digest(manifest["origins"]) != manifest.get("origins_fingerprint"):
        raise ValueError("feature origin fingerprint mismatch")
    return manifest


def _build_size(repo_root, run, plan, directory, L, request_sha):
    bank = plan["banks"][str(L)]
    bank_dir = _safe(repo_root, bank["directory"])
    data = load_learning_data(bank_dir, mode="all_edges", verify_checksums=True, include_test=False)
    lattice = square_lattice(L)
    ends = lattice.edge_vertex_indices
    vertices, edges = lattice.n_vertices, lattice.n_edges
    tasks = [task for task in plan["tasks"] if task["stage"] == "saved" and task["L"] == L]
    task_map = {}
    for task in tasks:
        if task["id"] in task_map or task["key"] != f"L{L}-{task['id']}":
            raise ValueError("duplicate or inconsistent saved task identity")
        task_map[task["id"]] = task
    expected = bank["entries"]
    if set(task_map) != set(expected):
        raise ValueError("saved stage must cover every declared non-test Hamiltonian")
    if any(entry["instance"]["split"] not in SPLITS for entry in expected.values()):
        raise ValueError("feature request includes a test Hamiltonian")
    original_manifest = _read(bank_dir / "manifest.json")
    originals = {item["id"]: item for item in original_manifest["instances"]
                 if item["split"] in SPLITS}
    if set(originals) != set(expected) or any(originals[name] != expected[name]["instance"] for name in expected):
        raise ValueError("validation plan and original bank Hamiltonian identities differ")
    arrays, origins, selected, maximum_delta = {}, {}, {}, 0.0
    uid_splits = {}
    for split in SPLITS:
        base = getattr(data, split)
        names = sorted(name for name, entry in expected.items() if entry["instance"]["split"] == split)
        selected[split] = names
        if len(base) != len(names) * edges:
            raise ValueError("original feature rows differ from declared group counts")
        blocks = {view: [] for view in VIEWS}
        for index, name in enumerate(names):
            instance = expected[name]["instance"]
            uid = instance["hamiltonian_uid"]
            if uid in uid_splits and uid_splits[uid] != split:
                raise ValueError("Hamiltonian UID leaks across train and validation")
            uid_splits[uid] = split
            rows = slice(index * edges, (index + 1) * edges)
            if (not np.all(base.hamiltonian_ids[rows] == name) or not np.all(base.hamiltonian_uids[rows] == uid)
                    or not np.array_equal(base.edge_ids[rows], np.arange(edges))):
                raise ValueError("original feature row identity/order mismatch")
            job, committed = _job_records(run, task_map[name])
            # The immutable original loader verifies fields, abs labels, schedule,
            # metadata and original feature construction. Read only its tiny mean
            # arrays again to compare f before float32 conversion.
            with np.load(bank_dir / "instances" / f"{name}.npz", allow_pickle=False) as original:
                original_stars = _numeric(original, "star_mean", (vertices,))
                original_pairs = _numeric(original, "pair_mean", (edges,))
            with np.load(job / "coordinates.npz", allow_pickle=False) as archive:
                coordinate = {}
                for view in VIEWS:
                    suffix = "truth" if view == "exact" else f"{view}_mean"
                    stars = _numeric(archive, f"star_{suffix}", (vertices,))
                    pairs = _numeric(archive, f"pair_{suffix}", (edges,))
                    if view != "f" and (min(stars.min(), pairs.min()) < -1e-12 or max(stars.max(), pairs.max()) > 1+1e-12):
                        raise ValueError("bounded g/exact feature outside [0,1]")
                    coordinate[view] = np.column_stack((stars[ends[:, 0]], stars[ends[:, 1]], pairs))
                    if view == "f":
                        for new, old in ((stars, original_stars), (pairs, original_pairs)):
                            maximum_delta = max(maximum_delta, float(np.max(np.abs(new-old))))
                            if not np.allclose(new, old, rtol=1e-12, atol=1e-13):
                                raise ValueError("saved f view does not reproduce original means")
            # Keep the original training floats exactly: summation differences in
            # the independently recomputed f census must not change the control.
            blocks["f"].append(base.x[rows])
            for view in ("g", "exact"):
                blocks[view].append(coordinate[view].astype(np.float32))
            origins[name] = {"instance": instance, "input_sha256": expected[name]["sha256"],
                             **committed}
        for view in VIEWS:
            arrays[f"{split}__x_{view}"] = np.concatenate(blocks[view]).astype(np.float32)
        for field in ("y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
            arrays[f"{split}__{field}"] = getattr(base, field)
    if any(not np.isfinite(array).all() for array in arrays.values() if array.dtype.kind in "fiu"):
        raise ValueError("feature cache cannot contain nonfinite numeric arrays")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "views.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(directory / "views.npz")
    manifest = {"schema_version": 1, "feature_version": FEATURE_VERSION, "request_sha256": request_sha,
                "L": L, "archive": "views.npz", "archive_sha256": sha(directory / "views.npz"),
                "origins": origins, "origins_fingerprint": _digest(origins),
                "ordered_edge_endpoints": ends.tolist(), "features": ["mean_A_u", "mean_A_v", "mean_Au_Av"],
                "target": "abs(b_z)", "training_views": ["f", "g"], "evaluation_views": list(VIEWS),
                "f_preserves_original_float32": True, "f_reproduction_max_absolute_delta": maximum_delta,
                "f_reproduction_tolerance": {"rtol": 1e-12, "atol": 1e-13},
                "original_bank": str(bank_dir.relative_to(repo_root)), "original_provenance": data.provenance,
                "selected_ids": selected,
                "hamiltonian_counts": {split: len(selected[split]) for split in SPLITS},
                "row_counts": {split: len(arrays[f"{split}__y"]) for split in SPLITS},
                "test_payload_loaded": False, "spin_records_loaded": False,
                "measurement_scope": "known-field simulation estimators; fields are labels only"}
    _atomic_json(directory / "manifest.json", manifest)
    return manifest


def _prepare_feature_views(repo_root, validation_run, output_dir, sizes=(3, 4)):
    """Build/verify immutable all-edge views, resuming at completed size boundaries.

    Only L3/L4 and original train/validation groups are supported. A cache is a
    separate versioned artifact: it never changes the original bank or completed
    validation campaign. A later load verifies compact caches, avoiding repeated
    original-bank checks for each seed/model.
    """
    root, run, output = (Path(path).resolve() for path in (repo_root, validation_run, output_dir))
    sizes = tuple(sizes)
    if not sizes or any(type(L) is not int or L not in (3, 4) for L in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("feature sizes must be unique and restricted to L3/L4")
    allowed = root / "codex material/artifacts/estimator_mlp"
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("feature output must be beneath ignored estimator_mlp artifacts")
    plan = load_plan(root, run)  # Verify scientific runtime plus every source/input pin once.
    request = {"schema_version": 1, "feature_version": FEATURE_VERSION, "sizes": list(sizes),
               "validation_run": str(run.relative_to(root)), "validation_plan_sha256": sha(run / "plan.json"),
               "adapter_sources": _source_hashes(root), "view_order": list(VIEWS),
               "group_splits": list(SPLITS), "mode": "all_edges"}
    request_sha = _digest(request)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "request.json").exists():
        if _read(output / "request.json") != request:
            raise ValueError("frozen feature request changed; use a new cache directory")
    else:
        _atomic_json(output / "request.json", request)
    manifests = {}
    for L in sizes:
        directory = output / f"L{L}"
        if (directory / "manifest.json").exists():
            manifest = _verify_size(directory, request_sha)
            # Source-result identities are rechecked only on preparation, never
            # once per training seed. No original test payload is opened.
            tasks = {task["id"]: task for task in plan["tasks"] if task["stage"] == "saved" and task["L"] == L}
            if set(tasks) != set(manifest["origins"]):
                raise ValueError("cached source identities differ from validation plan")
            for name, origin in manifest["origins"].items():
                _, committed = _job_records(run, tasks[name])
                if any(origin[key] != committed[key] for key in ("done_sha256", "files")):
                    raise ValueError("cached feature source commit changed")
        else:
            manifest = _build_size(root, run, plan, directory, L, request_sha)
        manifests[str(L)] = {"path": f"L{L}/manifest.json", "sha256": sha(directory / "manifest.json"),
                             "archive_sha256": manifest["archive_sha256"]}
    committed = {"schema_version": 1, "feature_version": FEATURE_VERSION, "state": "complete",
                 "request_sha256": request_sha, "sizes": manifests, "test_payload_loaded": False,
                 "spin_records_loaded": False}
    if (output / "manifest.json").exists() and _read(output / "manifest.json") != committed:
        raise ValueError("completed feature manifest changed; use a new cache directory")
    _atomic_json(output / "manifest.json", committed)
    return output



def prepare_feature_views(repo_root, validation_run, output_dir, sizes=(3, 4)):
    """Build/verify paired non-test views with an exclusive preparation lease.

    Completed sizes are hash-verified and skipped when a prior preparation was
    interrupted. Run records and caches must remain under ignored artifacts.
    """
    root, output = Path(repo_root).resolve(), Path(output_dir).resolve()
    allowed = root / "codex material/artifacts/estimator_mlp"
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("feature output must be beneath ignored estimator_mlp artifacts")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("feature preparation is active in another process") from error
        try:
            return _prepare_feature_views(root, validation_run, output, sizes)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def load_feature_views(feature_dir, L):
    """Load all paired views once; returned arrays are read-only and share labels.

    The original bank and validation campaign are not revisited here. The queue
    pins the root manifest and source versions; this loader verifies its nested
    hashes and row/group contracts before handing arrays to the CUDA engine.
    """
    root = Path(feature_dir).resolve()
    manifest = _read(root / "manifest.json")
    request = _read(root / "request.json")
    if (manifest.get("schema_version") != 1 or manifest.get("feature_version") != FEATURE_VERSION
            or manifest.get("state") != "complete" or _digest(request) != manifest.get("request_sha256")):
        raise ValueError("invalid committed feature manifest/request")
    if type(L) is not int or str(L) not in manifest["sizes"]:
        raise ValueError("lattice size is absent from the feature cache")
    record = manifest["sizes"][str(L)]
    if record.get("path") != f"L{L}/manifest.json":
        raise ValueError("invalid feature size manifest path")
    directory = root / f"L{L}"
    if sha(directory / "manifest.json") != record["sha256"]:
        raise ValueError("feature size manifest checksum mismatch")
    size = _verify_size(directory, manifest["request_sha256"])
    if size["L"] != L or size["archive_sha256"] != record["archive_sha256"]:
        raise ValueError("feature size identity/hash mismatch")
    if size.get("test_payload_loaded") is not False or size.get("spin_records_loaded") is not False:
        raise ValueError("feature cache access scope is invalid")
    ends = square_lattice(L).edge_vertex_indices
    if not np.array_equal(ends, size["ordered_edge_endpoints"]):
        raise ValueError("cached feature endpoint ordering mismatch")
    with np.load(directory / "views.npz", allow_pickle=False) as archive:
        expected_keys = {f"{split}__{key}" for split in SPLITS
                         for key in ("x_f", "x_g", "x_exact", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids")}
        if set(archive.files) != expected_keys:
            raise ValueError("feature cache array schema mismatch")
        arrays = {key: archive[key] for key in expected_keys}
    edges = len(ends)
    all_uids = {}
    for split in SPLITS:
        names = size["selected_ids"][split]
        rows = len(names) * edges
        if size["row_counts"][split] != rows or size["hamiltonian_counts"][split] != len(names) or len(set(names)) != len(names):
            raise ValueError("cached feature group counts mismatch")
        y = arrays[f"{split}__y"]
        if y.shape != (rows, 1) or y.dtype != np.float32 or not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError("invalid cached magnitude labels")
        for key in ("hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
            if arrays[f"{split}__{key}"].shape != (rows,):
                raise ValueError("cached row identities have invalid shape")
        for index, name in enumerate(names):
            origin = size["origins"][name]["instance"]
            uid = origin["hamiltonian_uid"]
            if origin["split"] != split or (uid in all_uids and all_uids[uid] != split):
                raise ValueError("cached Hamiltonian split leakage")
            all_uids[uid] = split
            selection = slice(index*edges, (index+1)*edges)
            if (not np.all(arrays[f"{split}__hamiltonian_ids"][selection] == name)
                    or not np.all(arrays[f"{split}__hamiltonian_uids"][selection] == uid)
                    or not np.array_equal(arrays[f"{split}__edge_ids"][selection], np.arange(edges))):
                raise ValueError("cached row alignment differs from Hamiltonian identity")
            # Labels are independently reconstructed from the bound seed, without
            # opening original payloads or exposing fields as network features.
            settings = size["original_provenance"]["settings"]["fields"]
            fields = np.random.default_rng(np.random.SeedSequence(origin["field_seed"])).uniform(settings["low"], settings["high"], edges)
            if not np.array_equal(y[selection, 0], np.abs(fields).astype(np.float32)):
                raise ValueError("cached labels differ from seeded |b_z|")
        for view in VIEWS:
            x = arrays[f"{split}__x_{view}"]
            if x.shape != (rows, 3) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError("invalid cached feature inputs")
            if view != "f" and (np.any(x < -1e-7) or np.any(x > 1+1e-7)):
                raise ValueError("bounded cached feature outside [0,1]")
    for array in arrays.values():
        array.flags.writeable = False
    result = {}
    root_hash = sha(root / "manifest.json")
    for view in VIEWS:
        groups = {split: EdgeRows(arrays[f"{split}__x_{view}"], arrays[f"{split}__y"],
                                 arrays[f"{split}__hamiltonian_ids"], arrays[f"{split}__hamiltonian_uids"],
                                 arrays[f"{split}__edge_ids"]) for split in SPLITS}
        provenance = {**size["original_provenance"], "feature_version": FEATURE_VERSION, "estimator_view": view,
                      "source_fingerprint": _digest({"manifest_sha256": root_hash, "L": L, "view": view}),
                      "source": {**size["original_provenance"]["source"],
                                 "feature_manifest_sha256": root_hash, "feature_archive_sha256": size["archive_sha256"],
                                 "origin_fingerprint": size["origins_fingerprint"]},
                      "selection": {"mode": "all_edges", "fixed_edge": 0,
                                    "training_hamiltonians": size["hamiltonian_counts"]["train"],
                                    "selected_ids": {**size["selected_ids"], "test": []}},
                      "row_counts": {**size["row_counts"], "test": 0},
                      "hamiltonian_counts": {**size["hamiltonian_counts"], "test": 0},
                      "checksum_scope": "original non-test payloads and committed validation summaries; compact cache verified on load",
                      "test_protocol": "development validation only; no final-test payload access",
                      "test_payload_loaded": False, "spin_records_loaded": False,
                      "training_eligible": view != "exact"}
        result[view] = LearningData(groups["train"], groups["validation"], None, L, 0, "all_edges", provenance, root)
    return result
