"""Compact, paired development maps for the final multi-size benchmark.

This adapter reads verified non-test summaries, not retained configurations.
The original f values are preserved at float32; g comes from the corresponding
committed estimator-validation coordinates. Exact inputs exist only where the
validation campaign actually calculated them. Original artifacts are read-only.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from quantum_lattice.data.generation import _bank, _canonical, validate_config
from quantum_lattice.lattice import square_lattice

VERSION = "final-observable-maps-v1"
SPLITS = ("train", "validation")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError("feature source path escapes its declared root")
    return result


def _indexed(items, label):
    result = {item["id"]: item for item in items}
    if len(result) != len(items):
        raise ValueError(f"duplicate Hamiltonian IDs in {label}")
    return result


def _numeric(archive, key, shape):
    a = archive[key]
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"invalid numeric coordinate: {key}")
    return a


def _maps(star, pair, L):
    out = np.empty((3, L, L), dtype=np.float32)
    out[0] = np.asarray(star).reshape(L, L)
    out[1:] = np.asarray(pair).reshape(L, L, 2).transpose(2, 0, 1)
    if not np.isfinite(out).all():
        raise ValueError("observable maps exceed finite float32 range")
    return out


def _metadata(root, bank_root, estimator_run, plan, L, limits):
    bank = plan["banks"][str(L)]
    directory = _safe(root, bank["directory"])
    if not directory.is_relative_to(bank_root):
        raise ValueError("estimator bank differs from the declared bank_root")
    documents = {name: _read(directory / name) for name in
                 ("config.json", "manifest.json", "dataset.json", "verification.json", "status.json")}
    config = validate_config(documents["config.json"])
    manifest, index, verification, status = (documents[n] for n in
        ("manifest.json", "dataset.json", "verification.json", "status.json"))
    if config != bank["config"] or config["lattice"] != {"L": L, "type": "square_periodic"}:
        raise ValueError("source lattice/config differs from the estimator plan")
    if (status.get("state", "").lower() != "complete" or verification.get("passed") is not True
            or any(d.get("schema_version") != 1 for d in (manifest, index, verification))):
        raise ValueError("source bank must be complete and verified")
    if manifest.get("config_sha256") != _digest(config) or manifest["instances"] != _bank(config):
        raise ValueError("source configuration hash or seeded instance plan differs")
    instances = _indexed(manifest["instances"], "manifest")
    indexed, checked = _indexed(index["instances"], "index"), _indexed(verification["instances"], "verification")
    if set(instances) != set(indexed) or set(instances) != set(checked) or status.get("completed") != len(instances):
        raise ValueError("source metadata coverage is incomplete")
    seen = {}
    for name, item in instances.items():
        uid, split = item["hamiltonian_uid"], item["split"]
        if uid in seen:
            raise ValueError("Hamiltonian UID duplicated across source groups")
        seen[uid] = split
        if split not in (*SPLITS, "test"):
            raise ValueError("unknown Hamiltonian split")
        parent = item.get("parent_hamiltonian_id")
        if parent is not None and (parent not in instances or instances[parent]["split"] != split):
            raise ValueError("parent Hamiltonian split mismatch")
        if indexed[name].get("split") != split or checked[name].get("split") != split:
            raise ValueError("source split declarations differ")
        if (indexed[name].get("path") != f"instances/{name}.npz"
                or indexed[name].get("sha256") != checked[name].get("sha256")):
            raise ValueError("source path/checksum declarations differ")
    counts = {split: sum(i["split"] == split for i in instances.values()) for split in (*SPLITS, "test")}
    if index.get("split_counts") != counts or index.get("local_rows") != len(instances) * 2 * L * L:
        raise ValueError("source group/row counts differ")
    expected = bank["entries"]
    if set(expected) != {name for name, item in instances.items() if item["split"] in SPLITS}:
        raise ValueError("estimator plan must cover exactly the non-test source groups")
    for name, entry in expected.items():
        if entry["instance"] != instances[name] or entry["sha256"] != indexed[name]["sha256"]:
            raise ValueError("estimator/source instance identity or checksum differs")
    pins = {}
    for name in documents:
        path = directory / name
        relative = str(path.relative_to(root))
        actual = _sha(path)
        if relative in plan["inputs"] and plan["inputs"][relative] != actual:
            raise ValueError("pinned source bank metadata changed")
        pins[relative] = actual
    tasks = {t["id"]: t for t in plan["tasks"] if t["stage"] == "saved" and t["L"] == L}
    if len(tasks) != sum(t["stage"] == "saved" and t["L"] == L for t in plan["tasks"]) or set(tasks) != set(expected):
        raise ValueError("saved coordinate task coverage differs from non-test bank")
    selected = {}
    for split in SPLITS:
        names = sorted(name for name, item in instances.items() if item["split"] == split)
        count = limits.get(split)
        if count is not None:
            if type(count) is not int or count < 1 or count > len(names):
                raise ValueError(f"invalid deterministic {split} prefix count")
            names = names[:count]
        if not names:
            raise ValueError("each feature split must be nonempty")
        selected[split] = names
    return directory, config, instances, indexed, tasks, selected, pins


def _instance(root, directory, config, indexed, item, task, estimator_run, L):
    name = item["id"]
    if item["split"] not in SPLITS:
        raise ValueError("test payload access is prohibited")
    expected_key = f"L{L}-{name}"
    if task["key"] != expected_key or task.get("replica") is not None:
        raise ValueError("saved task identity differs")
    source = _safe(directory, indexed[name]["path"])
    if _sha(source) != indexed[name]["sha256"]:
        raise ValueError(f"original payload checksum mismatch: {name}")
    job = _safe(estimator_run, "jobs/saved/" + expected_key)
    done = _read(job / "done.json")
    if set(done.get("files", {})) != {"coordinates.npz", "summary.json"}:
        raise ValueError("saved coordinates lack an exact commit marker")
    for filename, expected_sha in done["files"].items():
        if _sha(job / filename) != expected_sha:
            raise ValueError(f"saved coordinate checksum mismatch: {expected_key}/{filename}")
    summary = _read(job / "summary.json")
    if any(summary.get(k) != task.get(k) for k in ("stage", "key", "L", "id", "stratum", "replica")):
        raise ValueError("saved coordinate summary identity mismatch")
    exact = L in (3, 4) and L in config.get("_exact_sizes", [3, 4])
    if exact:
        error = summary.get("max_identity_error")
        if isinstance(error, bool) or not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error <= 1e-10:
            raise ValueError("small-size exact identity check is missing or failed")
    vertices, edges = L * L, 2 * L * L
    lattice = square_lattice(L)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        records = (config["sampler"]["total_sweeps"] - config["sampler"]["burn_in_sweeps"]) // config["sampler"]["thinning_sweeps"]
        if (metadata.get("schema_version") != 1 or metadata.get("instance") != item
                or metadata.get("completed_sweeps") != config["sampler"]["total_sweeps"]
                or metadata.get("retained_records") != records
                or indexed[name].get("records") != records or indexed[name].get("local_rows") != edges):
            raise ValueError("source payload metadata/sampling schedule differs")
        ends = archive["edge_vertex_indices"]
        if ends.dtype.kind not in "iu" or not np.array_equal(ends, lattice.edge_vertex_indices):
            raise ValueError("source edge order differs from square topology")
        star, pair = _numeric(archive, "star_mean", (vertices,)), _numeric(archive, "pair_mean", (edges,))
        local = _numeric(archive, "mlp_X", (edges, 3))
        if not np.array_equal(local, np.column_stack((star[ends[:, 0]], star[ends[:, 1]], pair))):
            raise ValueError("original local rows disagree with shared-star summaries")
        fields = _numeric(archive, "b_z", (edges,)).astype(np.float64)
        seeded = np.random.default_rng(np.random.SeedSequence(item["field_seed"])).uniform(config["fields"]["low"], config["fields"]["high"], edges)
        if (not np.array_equal(fields, seeded) or not np.array_equal(_numeric(archive, "target_b", (edges,)), fields)
                or not np.array_equal(_numeric(archive, "target_abs_b", (edges,)), np.abs(fields))):
            raise ValueError("source fields/targets differ from seeded Hamiltonian")
    result = {"f": _maps(star, pair, L), "fields": fields,
              "y": np.abs(fields).reshape(L, L, 2).transpose(2, 0, 1).astype(np.float32)}
    with np.load(job / "coordinates.npz", allow_pickle=False) as coordinates:
        fstar = _numeric(coordinates, "star_f_mean", (vertices,))
        fpair = _numeric(coordinates, "pair_f_mean", (edges,))
        if not np.allclose(fstar, star, rtol=1e-12, atol=1e-13) or not np.allclose(fpair, pair, rtol=1e-12, atol=1e-13):
            raise ValueError("recomputed f differs from original summary")
        gstar = _numeric(coordinates, "star_g_mean", (vertices,))
        gpair = _numeric(coordinates, "pair_g_mean", (edges,))
        if min(gstar.min(), gpair.min()) < 0 or max(gstar.max(), gpair.max()) > 1 + 1e-12:
            raise ValueError("conditioned means must lie in [0,1]")
        result["g"] = _maps(gstar, gpair, L)
        if exact:
            result["exact"] = _maps(_numeric(coordinates, "star_truth", (vertices,)),
                                    _numeric(coordinates, "pair_truth", (edges,)), L)
    origin = {"instance": item, "input_sha256": indexed[name]["sha256"],
              "coordinate_done_sha256": _sha(job / "done.json"), "coordinate_files": done["files"]}
    return result, origin


def _verify_size(directory, request):
    manifest = _read(directory / "manifest.json")
    if (manifest.get("version") != VERSION or manifest.get("request_sha256") != _digest(request)
            or manifest.get("archive") != "maps.npz" or manifest.get("schema_version") != 1
            or manifest.get("archive_sha256") != _sha(directory / "maps.npz")
            or manifest.get("origins_sha256") != _digest(manifest.get("origins"))):
        raise ValueError("immutable feature cache checksum/manifest mismatch")
    return manifest


def prepare_features(repo_root, bank_root, estimator_run, output_root, sizes,
                     limits=None, stop_requested: Callable[[], bool] | None = None):
    """Prepare paired non-test maps, resuming at committed size boundaries.

    ``limits={'train':N,'validation':M}`` selects deterministic lexicographic ID
    prefixes for bounded development timing. Omit limits for the entire source
    splits. Stop requests leave completed sizes intact; an incomplete size is
    rebuilt on resume. No test arrays or retained spin arrays are decompressed.
    """
    root = Path(repo_root).resolve()
    bank_root, estimator_run, output = (_safe(root, p) for p in (bank_root, estimator_run, output_root))
    allowed = root / "codex material/artifacts/final_benchmark"
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("final features must be inside ignored final_benchmark artifacts")
    sizes = list(sizes)
    if not sizes or len(set(sizes)) != len(sizes) or any(type(L) is not int or L < 3 for L in sizes):
        raise ValueError("feature sizes must be distinct integers >=3")
    limits = {} if limits is None else dict(limits)
    if set(limits) - set(SPLITS):
        raise ValueError("limits may contain train and validation only")
    plan_path = estimator_run / "plan.json"
    plan = _read(plan_path)
    if _read(estimator_run / "plan-sha256.json") != {"sha256": _sha(plan_path)}:
        raise ValueError("estimator source plan checksum mismatch")
    exact_sizes = plan["config"]["saved"]["exact_sizes"]
    if (not isinstance(exact_sizes, list) or any(type(L) is not int or L not in (3, 4) for L in exact_sizes)
            or len(set(exact_sizes)) != len(exact_sizes)):
        raise ValueError("exact reference sizes must be a unique subset of [3,4]")
    if any(str(L) not in plan["banks"] for L in sizes):
        raise ValueError("requested size absent from completed estimator plan")
    # Rendering verified summaries on a new machine is not continuation of its
    # Monte Carlo run. Verify pinned source bytes without requiring equal OS,
    # GPU, Python or package runtime versions to its original sampling machine.
    for name, digest in plan["sources"].items():
        if _sha(_safe(root, name)) != digest:
            raise ValueError(f"estimator source pin changed: {name}")
    metadata = {L: _metadata(root, bank_root, estimator_run, plan, L, limits) for L in sizes}
    request = {"schema_version": 1, "version": VERSION, "sizes": sizes, "limits": limits,
               "bank_root": str(bank_root.relative_to(root)), "estimator_run": str(estimator_run.relative_to(root)),
               "estimator_plan_sha256": _sha(plan_path), "estimator_source_hashes": plan["sources"],
               "adapter_source_sha256": _sha(Path(__file__)),
               "bank_input_hashes": {str(L): metadata[L][-1] for L in sizes},
               "selection": {str(L): metadata[L][5] for L in sizes},
               "scope": "development train/validation only; known-field simulation features; test excluded"}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (output / "request.json").exists():
            if _read(output / "request.json") != request:
                raise ValueError("frozen feature request changed; use a new output_root")
        else:
            if any(p.name != ".prepare.lock" for p in output.iterdir()):
                raise ValueError("refusing nonempty unbound feature output")
            _json(output / "request.json", request)
        manifests = {}
        for L in sizes:
            if stop_requested is not None and stop_requested():
                _json(output / "status.json", {"state": "paused", "completed_sizes": list(manifests), "sizes": sizes})
                return output
            directory = output / f"L{L}"
            if (directory / "manifest.json").exists():
                manifests[str(L)] = _verify_size(directory, request)
                continue
            bank_dir, config, instances, indexed, tasks, selected, pins = metadata[L]
            config = {**config, "_exact_sizes": plan["config"]["saved"]["exact_sizes"]}
            arrays, origins = {}, {}
            for split in SPLITS:
                names = selected[split]
                views = ["f", "g"] + (["exact"] if L in config["_exact_sizes"] else [])
                for view in views:
                    arrays[f"{split}__{view}"] = np.empty((len(names), 3, L, L), np.float32)
                arrays[f"{split}__y"] = np.empty((len(names), 2, L, L), np.float32)
                arrays[f"{split}__fields"] = np.empty((len(names), 2 * L * L), np.float64)
                arrays[f"{split}__ids"] = np.asarray(names, dtype=str)
                arrays[f"{split}__uids"] = np.asarray([instances[name]["hamiltonian_uid"] for name in names], dtype=str)
                for i, name in enumerate(names):
                    if stop_requested is not None and stop_requested():
                        _json(output / "status.json", {"state": "paused", "completed_sizes": list(manifests), "sizes": sizes, "partial_size": L})
                        return output
                    result, origins[name] = _instance(root, bank_dir, config, indexed, instances[name], tasks[name], estimator_run, L)
                    for field, value in result.items():
                        arrays[f"{split}__{field}"][i] = value
            directory.mkdir(exist_ok=True)
            temporary = directory / "maps.npz.tmp"
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temporary.replace(directory / "maps.npz")
            manifest = {"schema_version": 1, "version": VERSION, "L": L,
                        "request_sha256": _digest(request), "archive": "maps.npz",
                        "archive_sha256": _sha(directory / "maps.npz"), "origins": origins,
                        "origins_sha256": _digest(origins), "views": views,
                        "selected_ids": selected, "counts": {s: len(selected[s]) for s in SPLITS},
                        "known_fields_for_features": True, "fields_are_network_inputs": False,
                        "test_payload_loaded": False, "spin_records_loaded": False,
                        "exact_training_eligible": False, "f_preserves_original_float32": True,
                        "input_channels": ["mean_A", "mean_Au_Av_horizontal", "mean_Au_Av_vertical"],
                        "target_channels": ["abs_b_z_horizontal", "abs_b_z_vertical"]}
            _json(directory / "manifest.json", manifest)
            manifests[str(L)] = manifest
            _json(output / "status.json", {"state": "preparing", "completed_sizes": list(manifests), "sizes": sizes})
        _json(output / "manifest.json", {"schema_version": 1, "version": VERSION,
               "request_sha256": _digest(request), "sizes": {key: _sha(output / f"L{key}/manifest.json") for key in manifests},
               "test_payload_loaded": False, "spin_records_loaded": False})
        _json(output / "status.json", {"state": "complete", "completed_sizes": list(manifests), "sizes": sizes})
    return output


def load_features(cache_root, L):
    """Load one verified size without reopening any original bank or states.

    Returns ``{'L', 'train', 'validation', 'manifest'}``. Each split is a dict
    with read-only arrays f/g (N,3,L,L), optional exact (evaluation-only), y
    (N,2,L,L), fields (N,2L² float64 scoring references), ids and uids (N).
    """
    root = Path(cache_root).resolve()
    request = _read(root / "request.json")
    if request.get("version") != VERSION or request.get("schema_version") != 1:
        raise ValueError("unsupported feature request schema")
    if type(L) is not int or L not in request["sizes"]:
        raise ValueError("requested size is not part of the frozen cache")
    directory = _safe(root, f"L{L}")
    if (root / "manifest.json").exists():
        committed = _read(root / "manifest.json")
        if (committed.get("version") != VERSION or committed.get("request_sha256") != _digest(request)
                or committed.get("sizes", {}).get(str(L)) != _sha(directory / "manifest.json")):
            raise ValueError("root feature manifest checksum mismatch")
    manifest = _verify_size(directory, request)
    if manifest["L"] != L or manifest.get("test_payload_loaded") is not False or manifest.get("spin_records_loaded") is not False:
        raise ValueError("invalid feature identity/access scope")
    out = {"L": L, "manifest": manifest}
    expected_views = ["f", "g"] + (["exact"] if L in (3, 4) and "exact" in manifest["views"] else [])
    if manifest["views"] != expected_views:
        raise ValueError("invalid feature estimator views")
    with np.load(directory / "maps.npz", allow_pickle=False) as archive:
        expected = {f"{s}__{k}" for s in SPLITS for k in [*manifest["views"], "y", "fields", "ids", "uids"]}
        if set(archive.files) != expected:
            raise ValueError("feature cache array schema differs")
        for split in SPLITS:
            count = manifest["counts"][split]
            values = {key: archive[f"{split}__{key}"] for key in [*manifest["views"], "y", "fields", "ids", "uids"]}
            for key, a in values.items():
                shape = (count,) if key in {"ids", "uids"} else ((count, 2 * L * L) if key == "fields" else (count, 2 if key == "y" else 3, L, L))
                if a.shape != shape:
                    raise ValueError("feature cache array shape differs")
                if key in {"ids", "uids"}:
                    if a.dtype.kind not in "US" or len(np.unique(a)) != count:
                        raise ValueError("invalid or duplicate cached Hamiltonian identities")
                elif a.dtype != (np.float64 if key == "fields" else np.float32) or not np.isfinite(a).all():
                    raise ValueError("invalid feature numeric dtype/values")
                a.flags.writeable = False
            if values["ids"].tolist() != manifest["selected_ids"][split]:
                raise ValueError("cached Hamiltonian order differs from manifest")
            for i, name in enumerate(values["ids"]):
                if values["uids"][i] != manifest["origins"][str(name)]["instance"]["hamiltonian_uid"]:
                    raise ValueError("cached UID differs from origin manifest")
            if np.any(values["y"] < 0) or not np.array_equal(values["y"], np.abs(values["fields"]).reshape(count, L, L, 2).transpose(0, 3, 1, 2).astype(np.float32)):
                raise ValueError("cached labels differ from scoring fields")
            out[split] = values
    if set(out["train"]["uids"]) & set(out["validation"]["uids"]):
        raise ValueError("cached Hamiltonian UID leakage across train/validation")
    return out
