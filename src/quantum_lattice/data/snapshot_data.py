"""Bounded, resumable raw-Z bags from existing non-test Hamiltonian records.

Only the selected training/validation instance files are opened. Signed fields
are stored as labels for scoring, never transformed into network inputs. Shot
subsets sample record indices without replacement; repeated configurations stay.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.lattice import square_lattice


def _json(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _array_sha(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_json(path, value):
    _atomic(path, lambda stream: stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()))


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def infer_length(edge_count):
    """Infer a nondegenerate periodic square lattice from E=2L²."""
    edges = _integer(edge_count, "edge_count")
    length = math.isqrt(edges // 2)
    if length < 3 or 2 * length * length != edges:
        raise ValueError("states must have 2*L*L edges for integer L >= 3")
    return length


def reconstruct_pseudospins(states, L=None):
    """Return τ with τ₀=+1 iff every input is in the trivial cycle sector.

    Reconstruct a spanning-tree assignment and verify *all* edges. This checks
    plaquette flux and both noncontractible torus windings, including a bad
    winding with no local plaquette defect. Leading batch axes are preserved.
    """
    states = np.asarray(states)
    if states.ndim < 2 or states.dtype != np.int8 or not np.isin(states, (-1, 1)).all():
        raise ValueError("states must be int8 +/-1 with shape (..., records, edges)")
    length = infer_length(states.shape[-1])
    if L is not None and _integer(L, "L", 3) != length:
        raise ValueError("L differs from the edge count")
    lattice = square_lattice(length)
    flat = states.reshape(-1, states.shape[-1])
    tau = np.zeros((len(flat), length * length), dtype=np.int8)
    tau[:, 0] = 1
    visited = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for edge in lattice.vertices[u].incident_edges:
            a, b = edge.vertex_ids
            v = b if a == u else a
            if v not in visited:
                tau[:, v] = tau[:, u] * flat[:, edge.id]
                visited.add(v)
                stack.append(v)
    ends = lattice.edge_vertex_indices
    if len(visited) != length * length or not np.array_equal(tau[:, ends[:, 0]] * tau[:, ends[:, 1]], flat):
        raise ValueError("raw states violate trivial plaquette/winding sector constraints")
    return tau.reshape(states.shape[:-1] + (length * length,))


@dataclass(frozen=True)
class SnapshotSplit:
    states: np.ndarray
    fields: np.ndarray
    ids: np.ndarray
    uids: np.ndarray
    record_indices: np.ndarray

    def __len__(self):
        return len(self.ids)

    @property
    def L(self):
        return infer_length(self.states.shape[-1])

    def spin_maps(self, shots):
        """Raw inputs [Hamiltonian, shot, orientation, row, column]."""
        shots = _integer(shots, "shots")
        if shots > self.states.shape[1]:
            raise ValueError("requested shots exceed the cached nested subset")
        return self.states[:, :shots].reshape(len(self), shots, self.L, self.L, 2).transpose(0, 1, 4, 2, 3)

    def means(self, shots):
        """Complete edge-Z means [Hamiltonian, orientation, row, column]."""
        return self.spin_maps(shots).mean(axis=1, dtype=np.float32)


@dataclass(frozen=True)
class SnapshotCache:
    train: SnapshotSplit
    validation: SnapshotSplit
    metadata: dict
    L: int
    shot_counts: tuple[int, ...]


def _configuration(config, length):
    expected = {"selection_seed", "train_hamiltonians", "validation_hamiltonians", "shot_counts"}
    if not isinstance(config, dict) or set(config) - (expected | {"L"}) or not expected <= set(config):
        raise ValueError("snapshot config needs selection_seed, train_hamiltonians, validation_hamiltonians, shot_counts")
    value = {"selection_seed": _integer(config["selection_seed"], "selection_seed", 0),
             "train_hamiltonians": _integer(config["train_hamiltonians"], "train_hamiltonians"),
             "validation_hamiltonians": _integer(config["validation_hamiltonians"], "validation_hamiltonians")}
    counts = config["shot_counts"]
    if not isinstance(counts, (list, tuple)) or not counts:
        raise ValueError("shot_counts must be a nonempty increasing list")
    value["shot_counts"] = [_integer(n, "shot_count") for n in counts]
    if value["shot_counts"] != sorted(set(value["shot_counts"])):
        raise ValueError("shot_counts must be strictly increasing and distinct")
    value["L"] = _integer(config.get("L", length), "L", 3)
    if value["L"] != length:
        raise ValueError("config L differs from the feature manifest")
    return value


def _safe(root, path):
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("snapshot source path escapes the repository")
    return path


def _request(root, feature_cache, config):
    feature = _safe(root, feature_cache)
    if feature.is_dir():
        nested = feature / f"L{config.get('L', 4)}/manifest.json"
        feature = nested if nested.is_file() else feature / "manifest.json"
    size = _json(feature)
    config = _configuration(config, size["L"])
    bank = _safe(root, size["original_bank"])
    bank_manifest = _json(bank / "manifest.json")
    bank_instances = {item["id"]: item for item in bank_manifest["instances"]}
    selected = {}
    all_ids, all_uids = set(), set()
    for split in ("train", "validation"):
        names = size["selected_ids"][split]
        if len(names) != len(set(names)):
            raise ValueError("feature pool has duplicate Hamiltonian IDs")
        candidates = []
        for name in names:
            if not isinstance(name, str) or name in (".", "..") or Path(name).name != name or "\\" in name:
                raise ValueError("Hamiltonian IDs must be safe single path components")
            origin = size["origins"][name]
            instance = origin["instance"]
            uid = instance["hamiltonian_uid"]
            if not isinstance(uid, str) or not uid or name != instance["id"] or instance["split"] != split or bank_instances.get(name) != instance or name in all_ids or uid in all_uids:
                raise ValueError("feature/bank Hamiltonian identities are inconsistent or leak across splits")
            all_ids.add(name)
            all_uids.add(uid)
            candidates.append({"id": name, "uid": uid, "split": split, "instance": instance,
                               "input_sha256": origin["input_sha256"]})
        count = config[f"{split}_hamiltonians"]
        if count > len(candidates):
            raise ValueError(f"requested {split} Hamiltonians exceed the existing non-test pool")
        selected[split] = sorted(candidates, key=lambda item: (_digest([config["selection_seed"], "hamiltonian", split, item["uid"]]), item["uid"]))[:count]
    source_paths = (Path(__file__).resolve(), Path(__file__).with_name("state_storage.py").resolve())
    request = {"schema_version": 1, "kind": "raw-z-trivial-sector-bags", "config": config,
               "feature_manifest": str(feature.relative_to(root)), "feature_manifest_sha256": _sha(feature),
               "original_bank": str(bank.relative_to(root)), "bank_manifest_sha256": _sha(bank / "manifest.json"),
               "sources": {str(p.relative_to(root)): _sha(p) for p in source_paths if p.is_relative_to(root)},
               "selected": selected, "sector": "all-cycle products +1; tau[0]=+1",
               "sampling": "UID-seeded record-index permutation without replacement; nested prefixes; repeated states retained",
               "test_payload_loaded": False, "fields_used_as_inputs": False}
    return request, bank


def _make_part(bank, item, config):
    path = bank / "instances" / f"{item['id']}.npz"
    info_path = path.with_suffix(".json")
    if _sha(path) != item["input_sha256"]:
        raise ValueError("original instance payload hash differs from the feature binding")
    metadata_sha = _sha(info_path)
    info = _json(info_path)
    with np.load(path, allow_pickle=False) as archive:
        metadata_text = str(archive["metadata_json"].item())
        metadata = json.loads(metadata_text)
        if metadata["instance"] != item["instance"] or info["id"] != item["id"] or info["split"] != item["split"]:
            raise ValueError("raw instance metadata does not match its bound Hamiltonian")
        states = decode_states(archive)
        fields = np.array(archive["b_z"], dtype=np.float64, copy=True)
        ends = archive["edge_vertex_indices"]
    lattice = square_lattice(config["L"])
    if not np.array_equal(ends, lattice.edge_vertex_indices) or states.shape != (metadata["retained_records"], lattice.n_edges) or info["records"] != len(states):
        raise ValueError("raw record count or canonical topology differs from its metadata")
    if fields.shape != (lattice.n_edges,) or not np.isfinite(fields).all() or not np.array_equal(fields, np.asarray(metadata["chain"]["fields"])):
        raise ValueError("raw scoring fields are invalid or inconsistent")
    reconstruct_pseudospins(states, config["L"])
    shots = max(config["shot_counts"])
    if len(states) < shots:
        raise ValueError("requested shot count exceeds available retained records")
    seed = int(_digest([config["selection_seed"], "records", item["uid"]])[:32], 16)
    indices = np.random.default_rng(seed).permutation(len(states))[:shots].astype(np.int64)
    selected = np.ascontiguousarray(states[indices])
    if _sha(path) != item["input_sha256"] or _sha(info_path) != metadata_sha:
        raise ValueError("original instance changed during snapshot extraction")
    binding = {"id": item["id"], "uid": item["uid"], "split": item["split"],
               "input_sha256": item["input_sha256"], "instance_json_sha256": metadata_sha,
               "embedded_metadata_sha256": hashlib.sha256(metadata_text.encode()).hexdigest(),
               "decoded_states_sha256": _array_sha(states), "selected_states_sha256": _array_sha(selected),
               "fields_sha256": _array_sha(fields),
               "record_indices_sha256": _array_sha(indices), "retained_records": len(states),
               "validated_sector": "trivial flux and winding"}
    return {"states": selected, "fields": fields, "record_indices": indices}, binding


def prepare_snapshot_cache(repo_root, feature_cache, output, config, progress=None, should_stop=None):
    """Prepare/continue atomic per-H parts; return output even after clean pause.

    ``progress(event_dict)`` is called after each durable completed Hamiltonian.
    ``should_stop()`` is checked between Hamiltonians. A completed cache has
    ``manifest.json: state='complete'``; ``load_snapshot_cache`` rejects pauses.
    """
    root = Path(repo_root).resolve()
    output = _safe(root, output)
    if output == root or not output.is_relative_to(root / "codex material"):
        raise ValueError("snapshot caches must be below the ignored codex material directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another process is preparing this snapshot cache") from exc
        request, bank = _request(root, feature_cache, config)
        request_sha = _digest(request)
        if (output / "request.json").exists():
            if _json(output / "request.json") != request:
                raise ValueError("snapshot inputs/config/source pins changed; use a new cache path")
        elif any(p.name != ".prepare.lock" for p in output.iterdir()):
            raise ValueError("refusing to initialize an unbound nonempty cache directory")
        else:
            _save_json(output / "request.json", request)
        if (output / "manifest.json").is_file() and _json(output / "manifest.json").get("state") == "complete":
            load_snapshot_cache(output)
            return output
        config = request["config"]
        total = sum(len(items) for items in request["selected"].values())
        completed = 0
        parts = {split: [] for split in ("train", "validation")}
        for split in parts:
            for item in request["selected"][split]:
                if should_stop is not None and should_stop():
                    _save_json(output / "manifest.json", {"state": "paused", "request_sha256": request_sha, "completed": completed, "total": total})
                    return output
                part = output / "parts" / split / f"{item['id']}.npz"
                part_meta = part.with_suffix(".json")
                if part_meta.exists():
                    record = _json(part_meta)
                    if record.get("request_sha256") != request_sha or record.get("item") != item or not part.is_file() or _sha(part) != record.get("archive_sha256"):
                        raise ValueError("snapshot part identity/hash changed")
                    source = bank / "instances" / f"{item['id']}.npz"
                    if _sha(source) != item["input_sha256"] or _sha(source.with_suffix(".json")) != record["binding"]["instance_json_sha256"]:
                        raise ValueError("original snapshot source changed since its cached part")
                else:
                    arrays, binding = _make_part(bank, item, config)
                    _atomic(part, lambda stream: np.savez(stream, **arrays))
                    record = {"request_sha256": request_sha, "item": item, "binding": binding, "archive_sha256": _sha(part)}
                    _save_json(part_meta, record)
                parts[split].append({"path": str(part.relative_to(output)), "metadata_path": str(part_meta.relative_to(output)), "sha256": _sha(part), "metadata_sha256": _sha(part_meta), "binding": record["binding"]})
                completed += 1
                event = {"state": "preparing", "completed": completed, "total": total, "split": split, "id": item["id"]}
                _save_json(output / "progress.json", event)
                if progress is not None:
                    progress(dict(event))
        archives = {}
        for split in parts:
            arrays = {key: [] for key in ("states", "fields", "record_indices")}
            for record in parts[split]:
                with np.load(output / record["path"], allow_pickle=False) as archive:
                    for key in arrays:
                        arrays[key].append(archive[key])
            final = {key: np.stack(value) for key, value in arrays.items()}
            final["ids"] = np.asarray([item["id"] for item in request["selected"][split]])
            final["uids"] = np.asarray([item["uid"] for item in request["selected"][split]])
            archive_path = output / f"{split}.npz"
            _atomic(archive_path, lambda stream: np.savez(stream, **final))
            archives[split] = {"path": archive_path.name, "sha256": _sha(archive_path), "hamiltonians": len(final["ids"])}
        manifest = {"schema_version": 1, "state": "complete", "request_sha256": request_sha, "L": config["L"],
                    "shot_counts": config["shot_counts"], "archives": archives, "parts": parts,
                    "test_payload_loaded": False, "fields_used_as_inputs": False, "completed": completed, "total": total}
        _save_json(output / "manifest.json", manifest)
        _save_json(output / "progress.json", {"state": "complete", "completed": completed, "total": total})
        return output


def load_snapshot_cache(output):
    """Load only complete, hash-bound raw bags; resulting arrays are read-only."""
    output = Path(output).resolve()
    manifest, request = _json(output / "manifest.json"), _json(output / "request.json")
    if manifest.get("schema_version") != 1 or manifest.get("state") != "complete" or manifest.get("request_sha256") != _digest(request):
        raise ValueError("snapshot cache is incomplete or its request binding changed")
    if manifest.get("test_payload_loaded") is not False or manifest.get("fields_used_as_inputs") is not False:
        raise ValueError("invalid snapshot cache access/input scope")
    if manifest.get("L") != request["config"]["L"] or manifest.get("shot_counts") != request["config"]["shot_counts"] or set(manifest.get("archives", {})) != {"train", "validation"} or set(manifest.get("parts", {})) != {"train", "validation"}:
        raise ValueError("snapshot manifest differs from the request's geometry/shots/splits")
    splits = {}
    uid_sets = {}
    for split in ("train", "validation"):
        info = manifest["archives"][split]
        path = output / info["path"]
        if not path.resolve().is_relative_to(output) or _sha(path) != info["sha256"]:
            raise ValueError("snapshot archive path/hash changed")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"states", "fields", "record_indices", "ids", "uids"}:
                raise ValueError("snapshot archive has unexpected arrays")
            arrays = {key: archive[key] for key in archive.files}
        count = info["hamiltonians"]
        if count != request["config"][f"{split}_hamiltonians"] or count != len(request["selected"][split]) or len(manifest["parts"][split]) != count:
            raise ValueError("snapshot manifest Hamiltonian counts differ from the request")
        shots, edges = max(manifest["shot_counts"]), 2 * manifest["L"] ** 2
        if arrays["states"].shape != (count, shots, edges) or arrays["fields"].shape != (count, edges) or arrays["fields"].dtype != np.float64 or not np.isfinite(arrays["fields"]).all() or arrays["record_indices"].shape != (count, shots) or arrays["record_indices"].dtype != np.int64:
            raise ValueError("invalid snapshot shapes, fields or record indices")
        if arrays["ids"].tolist() != [i["id"] for i in request["selected"][split]] or arrays["uids"].tolist() != [i["uid"] for i in request["selected"][split]]:
            raise ValueError("snapshot archive Hamiltonian identity/order changed")
        reconstruct_pseudospins(arrays["states"], manifest["L"])
        for row, record in enumerate(manifest["parts"][split]):
            item = request["selected"][split][row]
            expected_path = f"parts/{split}/{item['id']}.npz"
            if record["path"] != expected_path or record["metadata_path"] != str(Path(expected_path).with_suffix(".json")):
                raise ValueError("snapshot durable part path differs from its Hamiltonian identity")
            if _sha(output / record["path"]) != record["sha256"] or _sha(output / record["metadata_path"]) != record["metadata_sha256"]:
                raise ValueError("snapshot durable part hash changed")
            indices = arrays["record_indices"][row]
            binding = record["binding"]
            part_metadata = _json(output / record["metadata_path"])
            if part_metadata.get("item") != item or part_metadata.get("binding") != binding or part_metadata.get("request_sha256") != manifest["request_sha256"] or part_metadata.get("archive_sha256") != record["sha256"]:
                raise ValueError("snapshot durable part metadata differs from its manifest binding")
            if np.any(indices < 0) or np.any(indices >= binding["retained_records"]) or len(set(indices.tolist())) != shots or _array_sha(indices) != binding["record_indices_sha256"] or _array_sha(arrays["states"][row]) != binding["selected_states_sha256"] or _array_sha(arrays["fields"][row]) != binding["fields_sha256"]:
                raise ValueError("snapshot records are not the bound unique-index subset")
        uid_sets[split] = set(arrays["uids"].tolist())
        if len(uid_sets[split]) != count:
            raise ValueError("snapshot split has duplicate Hamiltonian UIDs")
        for array in arrays.values():
            array.flags.writeable = False
        splits[split] = SnapshotSplit(**arrays)
    if uid_sets["train"] & uid_sets["validation"]:
        raise ValueError("snapshot cache leaks a Hamiltonian across splits")
    return SnapshotCache(splits["train"], splits["validation"], {"manifest": manifest, "request": request}, manifest["L"], tuple(manifest["shot_counts"]))
