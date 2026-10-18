"""Read-only field/prediction bindings for frozen Hamiltonian-score evaluation.

Only the already verified compact train/validation cache and completed model
outputs are read. Signed fields are regenerated from the cache's bound seeds;
they are references for scoring, never additional network input features.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from quantum_lattice.data.estimator_features import load_feature_views
from quantum_lattice.data.mlp_data import EdgeRows

VIEWS = ("f", "g", "exact")
OUTPUT_FILES = frozenset(("latest.pt", "best.pt", "history.json", "evaluation.json",
                          "per-hamiltonian.csv", "predictions.npz"))


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array


def _finite(value, shape, label):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise ValueError(f"invalid finite real {label}")
    return array


@dataclass(frozen=True)
class FieldReference:
    """Canonical validation groups, signed float64 fields and original row labels.

    ``true_b_z[h,e]`` follows ``hamiltonian_ids[h]`` and canonical edge ``e``.
    The original labels remain float32 ``abs(true_b_z)``; signed reconstruction
    deliberately retains the original simulation's full float64 precision.
    """
    L: int
    hamiltonian_ids: np.ndarray
    hamiltonian_uids: np.ndarray
    true_b_z: np.ndarray
    validation_rows: EdgeRows
    source_manifest_sha256: str


def _aligned_rows(rows, names, uids, fields, label):
    edges, count = fields.shape[1], len(names)
    n = count * edges
    if (rows.y.shape != (n, 1) or rows.y.dtype != np.float32
            or rows.x.shape != (n, 3) or rows.x.dtype != np.float32
            or not np.isfinite(rows.x).all()):
        raise ValueError(f"invalid {label} feature/label schema")
    expected = {"hamiltonian_ids": np.repeat(names, edges),
                "hamiltonian_uids": np.repeat(uids, edges),
                "edge_ids": np.tile(np.arange(edges, dtype=np.int64), count)}
    for key, value in expected.items():
        actual = np.asarray(getattr(rows, key))
        if (actual.shape != (n,) or not np.array_equal(actual, value)
                or (key == "edge_ids" and actual.dtype.kind not in "iu")):
            raise ValueError(f"{label} {key} differs from canonical field reference")
    if not np.array_equal(rows.y[:, 0], np.abs(fields).astype(np.float32).reshape(-1)):
        raise ValueError(f"{label} labels differ from seeded abs(b_z)")


def load_validation_fields(feature_dir, L):
    """Verify a compact cache and reconstruct its signed validation references.

    The existing loader verifies committed nested hashes, endpoint geometry,
    original float32 labels and Hamiltonian groups. This adapter additionally
    binds all three views to one canonical row order and the exact field seeds.
    No original bank, spin record or final-test payload is opened.
    """
    root = Path(feature_dir).resolve()
    source_sha = _sha(root / "manifest.json")
    views = load_feature_views(root, L)
    if set(views) != set(VIEWS):
        raise ValueError("expected all f/g/exact cached feature views")
    size_path = root / f"L{L}" / "manifest.json"
    size = _read(size_path)
    root_manifest = _read(root / "manifest.json")
    if (source_sha != _sha(root / "manifest.json")
            or _sha(size_path) != root_manifest["sizes"][str(L)]["sha256"]):
        raise ValueError("feature manifest changed while loading field references")
    settings = size["original_provenance"]["settings"]["fields"]
    if (settings.get("channel") != "z_only" or settings.get("distribution") != "uniform"
            or type(L) is not int or L < 3):
        raise ValueError("field reconstruction requires a z-only uniform square bank")
    low, high = settings["low"], settings["high"]
    if (isinstance(low, bool) or isinstance(high, bool)
            or not np.isfinite([low, high]).all() or low >= high):
        raise ValueError("invalid seeded uniform field interval")
    groups = {}
    for split in ("train", "validation"):
        names = np.asarray(size["selected_ids"][split], dtype=str)
        if (names.ndim != 1 or not len(names) or len(set(names)) != len(names)
                or np.any(np.char.str_len(names) == 0)):
            raise ValueError("invalid or duplicate Hamiltonian group names")
        origins = [size["origins"][name]["instance"] for name in names]
        if any(origin.get("id") != name or origin.get("split") != split
               for name, origin in zip(names, origins)):
            raise ValueError("seeded origin identity/split mismatch")
        uids = np.asarray([origin["hamiltonian_uid"] for origin in origins], dtype=str)
        if len(set(uids)) != len(uids) or np.any(np.char.str_len(uids) == 0):
            raise ValueError("duplicate or empty Hamiltonian UIDs")
        fields = np.stack([
            np.random.default_rng(np.random.SeedSequence(origin["field_seed"])).uniform(
                low, high, 2 * L * L) for origin in origins])
        for view in VIEWS:
            data = views[view]
            if data.L != L or data.mode != "all_edges" or data.test is not None:
                raise ValueError("field scores require non-test all-edge cache groups")
            if data.provenance.get("training_eligible") is not (view != "exact"):
                raise ValueError("cached exact-view training eligibility changed")
            _aligned_rows(getattr(data, split), names, uids, fields, f"{split}/{view}")
        groups[split] = (names, uids, fields)
    if (set(groups["train"][0]) & set(groups["validation"][0])
            or set(groups["train"][1]) & set(groups["validation"][1])):
        raise ValueError("Hamiltonian train/validation leakage")
    names, uids, fields = groups["validation"]
    rows = views["f"].validation
    copied_rows = EdgeRows(*(_readonly(getattr(rows, key)) for key in
                            ("x", "y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids")))
    return FieldReference(L, _readonly(names), _readonly(uids), _readonly(fields, np.float64),
                          copied_rows, source_sha)


def load_prediction_maps(job_dir, task, reference, *, plan_sha256):
    """Verify all six committed model outputs and return raw ``[H,E]`` views.

    ``plan_sha256`` is the existing campaign's *canonical JSON plan digest*,
    as stored in ``done.json``; it is not the pretty-printed file-byte digest.
    All model/checkpoint files are hashed as bytes, never deserialized here.
    Predictions, labels and identities must have the original canonical order;
    a shuffled, incomplete or relabeled archive is rejected rather than sorted.
    """
    if (not isinstance(plan_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None):
        raise ValueError("invalid canonical plan digest")
    if (set(task) != {"L", "key", "seed", "view"} or task["L"] != reference.L
            or type(task["L"]) is not int or type(task["seed"]) is not int
            or task["seed"] < 0 or task["view"] not in ("f", "g")
            or task["key"] != f"L{task['L']}-seed{task['seed']}-{task['view']}"):
        raise ValueError("invalid frozen model task identity")
    directory = Path(job_dir).resolve()
    done = _read(directory / "done.json")
    if (type(done.get("schema_version")) is not int or done["schema_version"] != 1
            or done.get("task") != task or done.get("plan_sha256") != plan_sha256):
        raise ValueError("completed model task/plan identity mismatch")
    hashes = done.get("hashes", {})
    if set(hashes) != OUTPUT_FILES:
        raise ValueError("completed model output inventory differs")
    for name, expected in hashes.items():
        path = directory / name
        if (not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                or not path.resolve().is_relative_to(directory) or not path.is_file()
                or _sha(path) != expected):
            raise ValueError(f"completed model checksum/path mismatch: {name}")
    evaluation = _read(directory / "evaluation.json")
    if (done.get("summary") != evaluation or evaluation.get("task") != task
            or evaluation.get("scope") != "development validation; not a new final test"
            or evaluation.get("checkpoint_selection") != "lowest own-view validation MSE"
            or evaluation.get("prediction_policy") != "raw unclipped linear predictions"
            or evaluation.get("exact_input_policy") != "crossed evaluation only; never checkpoint selection"
            or evaluation.get("test_payload_loaded") is not False):
        raise ValueError("completed evaluation identity or validation scope differs")
    rows = reference.validation_rows
    n = len(rows)
    result = {}
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        if set(archive.files) != {*(f"prediction_{view}" for view in VIEWS), "target",
                                 "hamiltonian_ids", "hamiltonian_uids", "edge_ids"}:
            raise ValueError("prediction archive schema differs")
        for key, expected in (("target", rows.y), ("hamiltonian_ids", rows.hamiltonian_ids),
                              ("hamiltonian_uids", rows.hamiltonian_uids), ("edge_ids", rows.edge_ids)):
            actual = archive[key]
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError(f"prediction archive {key} identity/order/labels differ")
        for view in VIEWS:
            prediction = _finite(archive[f"prediction_{view}"], (n, 1), f"{view} predictions")
            if prediction.dtype != np.float32:
                raise ValueError("committed predictions must retain the original float32 schema")
            result[view] = _readonly(prediction.reshape(reference.true_b_z.shape), np.float64)
    # Detect mutation while reading, including replacement of the commit marker.
    if _read(directory / "done.json") != done or _sha(directory / "predictions.npz") != hashes["predictions.npz"]:
        raise ValueError("completed predictions changed during read")
    return result


def reference_sign_lift(true_b_z, raw_predictions):
    """Apply a clearly labelled reference-sign oracle without abs or clipping.

    A negative raw magnitude prediction remains negative before multiplication
    by the known reference sign. A reference field exactly zero maps to zero:
    this oracle grants knowledge of zero support as well as nonzero signs.
    It is scoring assistance, not a sign learned by the magnitude-only network.
    """
    reference = np.asarray(true_b_z)
    if reference.ndim != 2 or not reference.size:
        raise ValueError("reference fields must be a nonempty [Hamiltonians,edges] matrix")
    _finite(reference, reference.shape, "reference fields")
    prediction = _finite(raw_predictions, reference.shape, "raw magnitude predictions")
    return _readonly(np.sign(reference.astype(np.float64)) * prediction.astype(np.float64))
