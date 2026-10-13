"""Load verified simulation summaries without opening retained spin records.

The sampler manifest owns group assignment. This module never splits edge rows
or reassigns a Hamiltonian, and it never makes the known fields input features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from quantum_lattice.data.generation import _bank, _canonical, validate_config
from quantum_lattice.lattice import square_lattice


@dataclass(frozen=True)
class EdgeRows:
    x: np.ndarray
    y: np.ndarray
    hamiltonian_ids: np.ndarray
    hamiltonian_uids: np.ndarray
    edge_ids: np.ndarray

    def __len__(self):
        return len(self.x)


@dataclass(frozen=True)
class LearningData:
    train: EdgeRows
    validation: EdgeRows
    test: EdgeRows | None
    L: int
    fixed_edge: int
    mode: str
    provenance: dict
    dataset_dir: Path


def _read_json(path):
    return json.loads(path.read_text())


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indexed(entries, label):
    result = {}
    for item in entries:
        name = item["id"]
        if name in result:
            raise ValueError(f"duplicate Hamiltonian id in {label}: {name}")
        result[name] = item
    return result


def _rows(items, n_edges):
    if not items:
        raise ValueError("each learning split must contain at least one Hamiltonian")
    x, y, names, uids, edges = [], [], [], [], []
    for instance, values, targets, selection in items:
        ids = np.arange(n_edges, dtype=np.int64) if selection is None else np.array([selection], dtype=np.int64)
        x.append(values[ids])
        y.append(targets[ids, None])
        names.extend([instance["id"]] * len(ids))
        uids.extend([instance["hamiltonian_uid"]] * len(ids))
        edges.append(ids)
    numeric_x, numeric_y = np.concatenate(x).astype(np.float32), np.concatenate(y).astype(np.float32)
    if not np.isfinite(numeric_x).all() or not np.isfinite(numeric_y).all():
        raise ValueError("features or labels cannot be represented as finite float32 values")
    return EdgeRows(numeric_x, numeric_y,
                    np.asarray(names, dtype=str), np.asarray(uids, dtype=str),
                    np.concatenate(edges))


def _finite_array(archive, key, shape):
    array = archive[key]
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"invalid shape, type or nonfinite values in {key}")
    return array


def load_learning_data(dataset_dir, mode="paper_size", fixed_edge=0,
                       training_hamiltonians=7450, verify_checksums=True,
                       include_test=False):
    """Load one complete size-specific bank, preserving its immutable groups.

    ``paper_size`` uses one fixed edge from exactly the first requested training
    Hamiltonians and from every validation Hamiltonian. ``all_edges`` uses all
    training/validation edges. Explicit ``include_test=True`` loads every edge
    of every test instance for frozen-model evaluation. The default only checks
    test index/verification declarations and does not access any test payload.
    The historical phase filter is unavailable; this is a declared adaptation,
    not recovery of the authors' historical training set.
    """
    root = Path(dataset_dir).resolve()
    if mode not in {"paper_size", "all_edges"}:
        raise ValueError("mode must be paper_size or all_edges")
    if isinstance(training_hamiltonians, bool) or not isinstance(training_hamiltonians, int) or training_hamiltonians < 1:
        raise ValueError("training_hamiltonians must be a positive integer")
    status = _read_json(root / "status.json")
    if status.get("state", "").lower() != "complete":
        raise ValueError("dataset must be COMPLETE before training")
    config = validate_config(_read_json(root / "config.json"))
    manifest = _read_json(root / "manifest.json")
    index = _read_json(root / "dataset.json")
    verification = _read_json(root / "verification.json")
    if any(document.get("schema_version") != 1 for document in (manifest, index, verification)):
        raise ValueError("unsupported dataset schema")
    if verification.get("passed") is not True:
        raise ValueError("dataset verification must have passed")
    config_sha = hashlib.sha256(_canonical(config).encode()).hexdigest()
    if manifest.get("config_sha256") != config_sha or manifest.get("instances") != _bank(config):
        raise ValueError("configuration hash or seeded Hamiltonian plan mismatch")
    planned = _indexed(manifest["instances"], "manifest")
    entries = _indexed(index["instances"], "dataset index")
    checks = _indexed(verification["instances"], "verification report")
    if set(entries) != set(planned) or set(checks) != set(planned):
        raise ValueError("dataset index and verification must cover every planned Hamiltonian")
    if status.get("completed") != len(planned):
        raise ValueError("completed count differs from the Hamiltonian plan")
    counts = {split: sum(item["split"] == split for item in planned.values())
              for split in ("train", "validation", "test")}
    if index.get("split_counts") != counts:
        raise ValueError("dataset split counts do not match the manifest")
    lattice = square_lattice(config["lattice"]["L"])
    n_edges, n_vertices = lattice.n_edges, lattice.n_vertices
    if isinstance(fixed_edge, bool) or not isinstance(fixed_edge, int) or not 0 <= fixed_edge < n_edges:
        raise ValueError("fixed_edge is outside the lattice edge range")
    if index.get("local_rows") != len(planned) * n_edges:
        raise ValueError("dataset local-row count mismatch")
    uid_splits = {}
    for item in planned.values():
        uid = item["hamiltonian_uid"]
        if uid in uid_splits and uid_splits[uid] != item["split"]:
            raise ValueError("one Hamiltonian UID appears in different splits")
        uid_splits[uid] = item["split"]
        if item.get("parent_hamiltonian_id") is not None:
            parent = planned.get(item["parent_hamiltonian_id"])
            if parent is None or parent["split"] != item["split"]:
                raise ValueError("Hamiltonian parent identity/split mismatch")
    train_ids = sorted(name for name, item in planned.items() if item["split"] == "train")
    if mode == "paper_size" and len(train_ids) < training_hamiltonians:
        raise ValueError(f"paper_size needs {training_hamiltonians} training Hamiltonians; found {len(train_ids)}")
    selected_train = train_ids[:training_hamiltonians] if mode == "paper_size" else train_ids
    selected = set(selected_train)
    groups = {"train": [], "validation": [], "test": []}
    retained_ids = {key: [] for key in groups}
    records_expected = (config["sampler"]["total_sweeps"] - config["sampler"]["burn_in_sweeps"]) // config["sampler"]["thinning_sweeps"]
    for name in sorted(planned):
        instance, entry, check = planned[name], entries[name], checks[name]
        split = instance["split"]
        if entry.get("split") != split or check.get("split") != split:
            raise ValueError("saved instance split differs from manifest")
        if entry.get("path") != f"instances/{name}.npz":
            raise ValueError("instance path differs from its canonical identity")
        path = root / entry["path"]
        if path.resolve().parent != (root / "instances").resolve():
            raise ValueError("instance path escapes the dataset")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64 or check.get("sha256") != entry["sha256"]:
            raise ValueError("dataset and verification instance hashes differ")
        if entry.get("records") != records_expected or entry.get("local_rows") != n_edges:
            raise ValueError("instance sampling count or local-row count mismatch")
        if split == "test" and not include_test:
            retained_ids[split].append(name)
            continue
        if verify_checksums and _sha_file(path) != entry["sha256"]:
            raise ValueError(f"instance checksum mismatch: {name}")
        # Access only summaries/features. np.load is lazy: raw/packed states are
        # not decompressed here, including in the checksum verification path.
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("schema_version") != 1 or metadata.get("instance") != instance:
                raise ValueError("NPZ metadata Hamiltonian identity mismatch")
            if metadata.get("completed_sweeps") != config["sampler"]["total_sweeps"] or metadata.get("retained_records") != records_expected:
                raise ValueError("NPZ metadata sampling schedule mismatch")
            ends = archive["edge_vertex_indices"]
            if ends.dtype.kind not in "iu" or not np.array_equal(ends, lattice.edge_vertex_indices):
                raise ValueError("feature endpoint order differs from lattice topology")
            values = _finite_array(archive, "mlp_X", (n_edges, 3))
            stars = _finite_array(archive, "star_mean", (n_vertices,))
            pairs = _finite_array(archive, "pair_mean", (n_edges,))
            expected_x = np.column_stack((stars[ends[:, 0]], stars[ends[:, 1]], pairs))
            if not np.array_equal(values, expected_x):
                raise ValueError("MLP inputs do not match ordered star/pair means")
            fields = _finite_array(archive, "b_z", (n_edges,))
            signed = _finite_array(archive, "target_b", (n_edges,))
            targets = _finite_array(archive, "target_abs_b", (n_edges,))
            seeded = np.random.default_rng(np.random.SeedSequence(instance["field_seed"])).uniform(
                config["fields"]["low"], config["fields"]["high"], n_edges)
            if not np.array_equal(fields, seeded) or not np.array_equal(fields, signed):
                raise ValueError("saved fields/targets differ from the seeded Hamiltonian")
            bound = max(abs(config["fields"]["low"]), abs(config["fields"]["high"]))
            if np.any(targets < 0) or np.any(targets > bound) or not np.array_equal(targets, np.abs(fields)):
                raise ValueError("magnitude labels do not equal |b_z| within declared bounds")
        if split == "train" and name not in selected:
            continue
        selection = fixed_edge if mode == "paper_size" and split != "test" else None
        groups[split].append((instance, values, targets, selection))
        retained_ids[split].append(name)
    rows = {key: _rows(items, n_edges) for key, items in groups.items()
            if key != "test" or include_test}
    source = {"dataset_index_sha256": _sha_file(root / "dataset.json"),
              "manifest_sha256": _sha_file(root / "manifest.json"),
              "verification_sha256": _sha_file(root / "verification.json"),
              "config_sha256": config_sha,
              "sampler_source_hashes": manifest.get("source_hashes", {})}
    selection = {"mode": mode, "fixed_edge": fixed_edge,
                 "training_hamiltonians": len(selected_train), "selected_ids": retained_ids}
    fingerprint = hashlib.sha256(_canonical({"source": source, "selection": selection}).encode()).hexdigest()
    provenance = {"schema_version": 1, "dataset_schema_version": index["schema_version"],
                  "source": source, "source_fingerprint": fingerprint,
                  "settings": config, "selection": selection, "split_protocol": manifest.get("split_protocol"),
                  "row_counts": {"train": len(rows["train"]), "validation": len(rows["validation"]),
                                 "test": len(retained_ids["test"]) * n_edges},
                  "hamiltonian_counts": {key: len(ids) for key, ids in retained_ids.items()},
                  "unused_training_ids": sorted(set(train_ids) - selected),
                  "checksums_verified": bool(verify_checksums),
                  "checksum_scope": "all splits" if include_test else "training and validation payloads; test declarations only",
                  "test_payload_loaded": bool(include_test),
                  "features": ["mean_A_u", "mean_A_v", "mean_Au_Av"],
                  "target": "abs(b_z)", "measurement_scope": manifest.get("measurement_scope"),
                  "phase_selection": "Historical phase filter unavailable; no reconstructed filter applied",
                  "test_protocol": "all edges of all untouched test Hamiltonians for both modes"}
    return LearningData(rows["train"], rows["validation"], rows.get("test"),
                        config["lattice"]["L"], fixed_edge, mode, provenance, root)
