"""Frozen, inference-only field scoring and Hamiltonian-grouped comparisons.

No network, optimizer, dataset loader or test-selection mechanism lives here.
Callers supply already aligned predictions and scoring references. Reference
sign assistance is labeled and never repairs negative magnitude predictions.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from quantum_lattice.lattice import Lattice, square_lattice
from quantum_lattice.metrics.hamiltonian_batch import CompiledHamiltonianMetric

VERSION = "final-field-evaluation-v1"
FILES = frozenset(("predictions.npz", "per-hamiltonian.csv", "evaluation.json"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _array_digest(array):
    a = np.ascontiguousarray(array)
    result = hashlib.sha256()
    result.update(str(a.shape).encode())
    result.update(a.dtype.str.encode())
    result.update(a.tobytes())
    return result.hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite real values of shape {shape}")
    return a


def _identities(value, count, name):
    a = np.asarray(value)
    if (a.shape != (count,) or a.dtype.kind not in "US" or np.any(np.char.str_len(a) == 0)
            or len(np.unique(a)) != count):
        raise ValueError(f"{name} must contain one nonempty unique string per Hamiltonian")
    return a.astype(str)


def _flatten(prediction, L, count):
    a = np.array(_finite(prediction, (count, 2, L, L), "prediction"), copy=True)
    return a, a.transpose(0, 2, 3, 1).reshape(count, 2 * L * L)


def _metric_scores(lattice, truth, lifted, native, batch_size):
    outputs = {"reference_sign": {}, "native_signed": {}}
    for model in ("two_channel", "z_branch"):
        compiled = CompiledHamiltonianMetric(lattice, model=model)
        for name, prediction in (("reference_sign", lifted), ("native_signed", native)):
            if prediction is None:
                continue
            parts = {"full": [], "traceless": []}
            for start in range(0, len(truth), batch_size):
                stop = start + batch_size
                current = compiled.distances(truth[start:stop], prediction[start:stop])
                for convention in parts:
                    parts[convention].append(current[convention])
            for convention in parts:
                outputs[name][f"{model}_{convention}"] = np.concatenate(parts[convention])
    return outputs


def load_evaluation(output_dir, bindings=None):
    """Verify one committed evaluation without reopening a dataset or model."""
    directory = Path(output_dir)
    request = json.loads((directory / "request.json").read_text())
    done = json.loads((directory / "done.json").read_text())
    if (request.get("version") != VERSION or done.get("schema_version") != 1
            or done.get("request_sha256") != _digest(request)
            or set(done.get("files", {})) != FILES):
        raise ValueError("evaluation commit/request mismatch")
    if bindings is not None and request.get("bindings") != bindings:
        raise ValueError("evaluation bindings differ")
    for name, expected in done["files"].items():
        if _sha(directory / name) != expected:
            raise ValueError(f"evaluation output checksum mismatch: {name}")
    summary = json.loads((directory / "evaluation.json").read_text())
    if summary.get("request_sha256") != done["request_sha256"]:
        raise ValueError("evaluation summary belongs to a different request")
    return summary


def evaluate_predictions(L, prediction, fields, ids, uids, output_dir, bindings,
                         *, native_signed=None, metric_batch_size=16):
    """Atomically score raw magnitude maps against float64 true fields.

    ``prediction`` is [H,2,L,L]; ``fields`` is [H,2L²] in canonical interleaved
    edge order. No abs/clipping is applied to predictions. Optional native
    signed estimates are an explicitly separate [H,2L²] or [H,2,L,L] output.
    D5 is computed in batches of at most32 H, with global Pauli merging and
    the full two-channel identity convention retained. Repeated identical calls
    verify and reuse the commit; different content or bindings are refused.
    """
    if type(L) is not int or L < 3:
        raise ValueError("L must be an integer >=3")
    truth = np.asarray(fields)
    if truth.ndim != 2 or not len(truth):
        raise ValueError("fields must be a nonempty [H,2L²] array")
    count, edges = len(truth), 2 * L * L
    truth = _finite(truth, (count, edges), "fields").astype(np.float64)
    maps, flat = _flatten(prediction, L, count)
    names, identities = _identities(ids, count, "ids"), _identities(uids, count, "uids")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("nonempty JSON bindings are required")
    bindings = json.loads(_canonical(bindings))
    if type(metric_batch_size) is not int or not 1 <= metric_batch_size <= 32:
        raise ValueError("metric_batch_size must be an integer in [1,32]")
    native = None
    if native_signed is not None:
        a = np.asarray(native_signed)
        if a.shape == (count, 2, L, L):
            _, a = _flatten(a, L, count)
        native = _finite(a, (count, edges), "native_signed").astype(np.float64)
    request = {"schema_version": 1, "version": VERSION, "L": L, "hamiltonians": count,
               "bindings": bindings, "metric_batch_size": metric_batch_size,
               "input_sha256": {name: _array_digest(a) for name, a in
                    (("prediction", maps), ("fields", truth), ("ids", names), ("uids", identities))},
               "native_signed_sha256": None if native is None else _array_digest(native),
               "evaluator_source_sha256": _sha(Path(__file__)),
               "metric_source_sha256": {obj.__module__: _sha(Path(sys.modules[obj.__module__].__file__))
                    for obj in (CompiledHamiltonianMetric, square_lattice, Lattice)},
               "numpy_version": np.__version__,
               "prediction_policy": "raw unclipped magnitude output; reference signs granted only for D5",
               "primary_metric": "mean per-H magnitude MSE versus abs(float64 true b_z)",
               "D5": "unsquared unit global-Pauli coefficient distance; two_channel_full primary"}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (directory / "request.json").exists():
            if json.loads((directory / "request.json").read_text()) != request:
                raise ValueError("evaluation content/bindings changed; use a new output_dir")
        else:
            if any(p.name != ".evaluation.lock" for p in directory.iterdir()):
                raise ValueError("refusing a nonempty unbound evaluation directory")
            _json(directory / "request.json", request)
        if (directory / "done.json").exists():
            return load_evaluation(directory, bindings)
        error = flat.astype(np.float64) - np.abs(truth)
        per_mse, per_mae = np.mean(error * error, axis=1), np.mean(np.abs(error), axis=1)
        lifted = np.sign(truth) * flat.astype(np.float64)
        scores = _metric_scores(square_lattice(L), truth, lifted, native, metric_batch_size)
        rows = []
        for i in range(count):
            row = {"hamiltonian_id": str(names[i]), "hamiltonian_uid": str(identities[i]),
                   "edges": edges, "mse": float(per_mse[i]), "mae": float(per_mae[i]),
                   "rmse": float(np.sqrt(per_mse[i])),
                   "negative_prediction_fraction": float(np.mean(flat[i] < 0))}
            row.update({f"reference_sign_D5_{k}": float(v[i]) for k, v in scores["reference_sign"].items()})
            if native is not None:
                row["native_signed_mse"] = float(np.mean((native[i] - truth[i])**2))
                row.update({f"native_signed_D5_{k}": float(v[i]) for k, v in scores["native_signed"].items()})
            rows.append(row)
        metrics = {"mse": float(per_mse.mean()), "mae": float(per_mae.mean()),
                   "rmse": float(np.sqrt(per_mse.mean())), "mean_H_rmse": float(np.sqrt(per_mse).mean()),
                   "negative_prediction_fraction": float(np.mean(flat < 0)),
                   "reference_sign_D5": {k: float(v.mean()) for k, v in scores["reference_sign"].items()}}
        if native is not None:
            nonzero = truth != 0
            metrics["native_signed"] = {
                "field_mse": float(np.mean((native - truth)**2)),
                "D5": {k: float(v.mean()) for k, v in scores["native_signed"].items()},
                "sign_agreement_nonzero_truth": float(np.mean(np.sign(native[nonzero]) == np.sign(truth[nonzero]))) if nonzero.any() else None,
                "zero_truth_edges": int((~nonzero).sum()),
                "sign_agreement_scope": "pooled descriptive edges; excludes exact-zero true fields"}
        summary = {"schema_version": 1, "version": VERSION, "L": L,
                   "request_sha256": _digest(request), "bindings": bindings,
                   "hamiltonians": count, "edges_per_H": edges, "metrics": metrics,
                   "per_hamiltonian": rows, "aggregation": "equal Hamiltonians; no seed averaging or ensembling here",
                   "D5_sign_policy": "reference-sign assisted magnitude predictions; native signed separately if supplied",
                   "reference_sign_zero_policy": "sign(0)=0; no abs/clipping of predicted magnitudes",
                   "fields_argument_used_for": "scoring only; upstream feature access is declared in bindings",
                   "inference_performed_here": False,
                   "uncertainty_scope": "not computed here; paired whole-H comparisons use frozen predictions"}
        temporary = directory / "predictions.npz.tmp"
        saved = {"prediction_maps": maps, "prediction": flat, "true_b_z": truth,
                 "target": np.abs(truth), "hamiltonian_ids": names, "hamiltonian_uids": identities}
        if native is not None:
            saved["native_signed_prediction"] = native
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **saved)
        temporary.replace(directory / "predictions.npz")
        temporary = directory / "per-hamiltonian.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(directory / "per-hamiltonian.csv")
        _json(directory / "evaluation.json", summary)
        _json(directory / "done.json", {"schema_version": 1, "request_sha256": _digest(request),
                                      "files": {name: _sha(directory / name) for name in sorted(FILES)}})
        return summary


def _loss_vectors(left, right, uids, right_uids):
    raw_a, raw_b = np.asarray(left), np.asarray(right)
    if raw_a.dtype.kind not in "fiu" or raw_b.dtype.kind not in "fiu":
        raise ValueError("paired losses must be real numeric values")
    a, b = raw_a.astype(np.float64), raw_b.astype(np.float64)
    if (a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0)):
        raise ValueError("paired losses must be aligned nonempty finite nonnegative vectors")
    ids = _identities(uids, len(a), "uids")
    if right_uids is not None and not np.array_equal(ids, _identities(right_uids, len(a), "right_uids")):
        raise ValueError("paired Hamiltonian UID order differs; refusing implicit sorting")
    return a, b, ids


def paired_comparison(left, right, *, uids, right_uids=None, bootstrap_samples=10000, seed=0):
    """Compare one loss per H using paired whole-H resampling, left minus right.

    A single ``uids`` vector explicitly declares shared ordering; provide
    ``right_uids`` to verify two independently loaded orders. This interval is
    conditional on the frozen models and training data, not optimizer/training-
    bank uncertainty or a certificate against Monte Carlo bias.
    """
    a, b, identities = _loss_vectors(left, right, uids, right_uids)
    if type(bootstrap_samples) is not int or bootstrap_samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("bootstrap_samples must be positive and seed nonnegative integers")
    rng = np.random.default_rng(seed)
    difference = a - b
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for start in range(0, bootstrap_samples, 256):
        stop = min(start + 256, bootstrap_samples)
        selected = rng.integers(0, len(a), size=(stop - start, len(a)))
        draws[start:stop] = difference[selected].mean(axis=1)
    return {"hamiltonians": len(a), "uids_sha256": _array_digest(identities),
            "left_mean": float(a.mean()), "right_mean": float(b.mean()),
            "difference_left_minus_right": float(difference.mean()),
            "relative_reduction_left_vs_right": float(1 - a.mean()/b.mean()) if b.mean() != 0 else None,
            "left_lower_H": int((difference < 0).sum()), "tied_H": int((difference == 0).sum()),
            "bootstrap_95ci_difference": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap": {"method": "paired whole-Hamiltonian percentile", "samples": bootstrap_samples,
                          "seed": seed, "confidence_level": .95, "resample_unit": "Hamiltonian"},
            "interpretation": "negative difference favors left; conditional on fixed models/training bank",
            "limitations": "does not quantify optimizer or new-training-bank variation or Monte Carlo bias"}


def compare_seed_panels(left, right, *, uids, right_uids=None, bootstrap_samples=10000, seed=0):
    """Average frozen seed losses within H before paired H bootstrap.

    Seed-specific paired differences and sample SD are reported separately.
    Never pools seed×H×edge observations as independent replicates. Repeating
    one deterministic control across seed keys does not create extra fits.
    """
    if not isinstance(left, dict) or not isinstance(right, dict) or not left or set(left) != set(right):
        raise ValueError("seed panels must have the same nonempty paired seed keys")
    if any(type(k) not in (int, str) for k in left):
        raise ValueError("seed identities must be integers or strings")
    seeds = sorted(left, key=lambda k: (type(k).__name__, str(k)))
    pairs = [_loss_vectors(left[k], right[k], uids, right_uids) for k in seeds]
    a, b = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
    result = paired_comparison(a.mean(axis=0), b.mean(axis=0), uids=uids,
                              right_uids=right_uids, bootstrap_samples=bootstrap_samples, seed=seed)
    differences = (a-b).mean(axis=1)
    result.update({"optimizer_seeds": len(seeds), "seed_aggregation": "mean loss within each H, then equal-H mean; not prediction ensembling",
                   "per_seed": [{"seed": k, "left_mean": float(a[i].mean()), "right_mean": float(b[i].mean()),
                                 "difference_left_minus_right": float(differences[i])} for i,k in enumerate(seeds)],
                   "sample_sd_seed_differences": float(differences.std(ddof=1)) if len(seeds)>1 else None,
                   "left_lower_seeds": int((differences<0).sum()), "tied_seeds": int((differences==0).sum()),
                   "resampling": "whole H with all seed panels retained; fixed seeds are not resampled"})
    return result
