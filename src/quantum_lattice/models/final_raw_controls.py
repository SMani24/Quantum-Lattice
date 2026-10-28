"""Separate L4 raw-input controls on the fresh sealed confirmation cohort."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.data.final_confirmation import load_confirmation_plan, load_confirmation_features
from quantum_lattice.models.final_campaign import artifact, atomic_json, canonical, digest, read, safe, sha
from quantum_lattice.models.final_evaluation import evaluate_predictions
from quantum_lattice.models.inverse_ising import fit_inverse_ising
from quantum_lattice.models.mlp_training import resolve_device
from quantum_lattice.models.snapshot_cnn import make_snapshot_model, snapshot_model_spec


def record_indices(uid, total, maximum, seed):
    """Select record positions, preserving repeated values and nested budgets."""
    if (not isinstance(uid, str) or not uid or type(total) is not int
            or type(maximum) is not int or type(seed) is not int or seed < 0
            or not 1 <= maximum <= total):
        raise ValueError("Invalid record selection")
    raw = bytes.fromhex(digest({"uid": uid, "seed": seed, "policy": "nested-record-indices-v1"}))
    words = np.frombuffer(raw, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence(words)).choice(total, maximum, replace=False)


def _save_checked(path, payload):
    atomic_json(path, {"payload": payload, "sha256": digest(payload)})


def _read_checked(path):
    value = read(path)
    if set(value) != {"payload", "sha256"} or value["sha256"] != digest(value["payload"]):
        raise ValueError("Inverse state checksum differs")
    return value["payload"]


def _predict(path, states, method, seed):
    done = read(path.parent/"done.json")
    if done["hashes"]["best.pt"] != sha(path):
        raise ValueError("Frozen raw checkpoint differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_task = {"key": f"L4-H7452-M256-seed{seed}-{method}", "method": method,
                     "seed": seed, "shots": 256, "train_hamiltonians": 7452}
    if (payload.get("task") != expected_task or done.get("task") != expected_task
            or payload.get("model_spec") != snapshot_model_spec(method)
            or payload.get("plan_sha256") != done.get("plan_sha256")):
        raise ValueError("Raw checkpoint task/architecture binding differs")
    device = resolve_device("cuda", True)
    inputs = states.reshape(len(states), states.shape[1], 4, 4, 2).transpose(0, 1, 4, 2, 3)
    if method == "mean_cnn":
        inputs = inputs.astype(np.float32).mean(axis=1)
    from quantum_lattice.models.final_training import _inference_policy
    with _inference_policy(True), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model = make_snapshot_model(method).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(inputs), 16):
                result = model(torch.from_numpy(np.ascontiguousarray(inputs[start:start+16])).to(device))
                outputs.append(result.cpu().numpy())
        torch.cuda.synchronize()
    return np.concatenate(outputs)


def run_raw_controls(root, run, plan, confirmation, *, stop_requested, remaining_seconds):
    """No new NN fit; true fields enter scoring, never the raw model or solver."""
    root, run = Path(root), Path(run)
    config = plan["config"]["raw_panel"]
    source = artifact(root, config["source_root"])
    cp = load_confirmation_plan(root, confirmation)
    data = load_confirmation_features(confirmation, 4)
    data["ids"], data["uids"] = data["hamiltonian_ids"], data["hamiltonian_uids"]
    items = [x for x in cp["instances"] if x["L"] == 4]
    if [x["hamiltonian_uid"] for x in items] != data["uids"].tolist():
        raise ValueError("Raw/observable confirmation identity order differs")
    maximum = max(config["records"])
    bags, indices = [], []
    for item in items:
        if stop_requested() or remaining_seconds() <= 0:
            return {"state": "paused"}
        path = Path(confirmation)/"instances"/item["id"]/"records.npz"
        with np.load(path, allow_pickle=False) as arrays:
            records = decode_states(arrays)
        selected = record_indices(item["hamiltonian_uid"], len(records), maximum, config["record_seed"])
        bags.append(records[selected]); indices.append(selected)
    states = np.stack(bags)
    base = run/"raw_controls"
    selection = {"policy": "nested fixed indices without replacement; repeated configurations retained",
        "seed": config["record_seed"], "uids": data["uids"].tolist(), "indices": np.stack(indices).tolist(),
        "confirmation_plan_sha256": digest(cp)}
    if (base/"record-selection.json").exists() and read(base/"record-selection.json") != selection:
        raise ValueError("Frozen raw record selection changed")
    atomic_json(base/"record-selection.json", selection)
    for M in config["records"]:
        bags = states[:, :M]
        for seed in config["seeds"]:
            for method in config["methods"]:
                if stop_requested() or remaining_seconds() <= 0:
                    return {"state": "paused"}
                path = source/f"L4-H7452-M256-seed{seed}-{method}"/"best.pt"
                out = base/f"M{M}-{method}-seed{seed}"
                binding = {"plan_sha256": digest(plan), "role": "fresh_confirmation_raw",
                    "method": method, "seed": seed, "records": M,
                    "record_selection_sha256": sha(base/"record-selection.json"),
                    "model_sha256": sha(path), "true_fields_as_inference_inputs": False,
                    "budget_shift": M != 256}
                if (out/"done.json").exists():
                    from quantum_lattice.models.final_evaluation import load_evaluation
                    load_evaluation(out, binding)
                    continue
                before = time.perf_counter()
                prediction = _predict(path, bags, method, seed)
                seconds = time.perf_counter()-before
                evaluate_predictions(4, prediction, data["fields"], data["ids"], data["uids"], out, binding)
                atomic_json(out/"timing.json", {"seconds": seconds, "hamiltonians": len(bags),
                    "scope": "cold model load+transfer+forward+CPU prediction; excludes raw-bag selection and D5 scoring",
                    "device": "GTX1660Super", "ensemble": False})
        fitted, solver_rows = [], []
        for i, item in enumerate(items):
            if stop_requested() or remaining_seconds() <= 0:
                return {"state": "paused"}
            path = base/f"M{M}-inverse"/"instances"/(item["id"]+".json")
            binding = {"uid": item["hamiltonian_uid"], "records": M,
                       "record_selection_sha256": sha(base/"record-selection.json"),
                       "ridge": config["ridge"], "max_iterations": config["max_iterations"],
                       "tolerance": config["tolerance"], "source_sha256": sha(Path(__file__))}
            old = _read_checked(path) if path.exists() else None
            if old is not None and old.get("binding") != binding:
                raise ValueError("Inverse solver identity/data/settings differ")
            if old is not None and old["result"]["stop_reason"] != "paused":
                result = old["result"]
                spent = old["seconds"]
            else:
                before = time.perf_counter()
                def checkpoint(state):
                    _save_checked(path.with_suffix(".checkpoint.json"), {"binding": binding, "optimizer_state": state})
                    return not stop_requested() and remaining_seconds() > 0
                saved = _read_checked(path.with_suffix(".checkpoint.json")) if path.with_suffix(".checkpoint.json").exists() else None
                if saved is not None and saved.get("binding") != binding:
                    raise ValueError("Inverse iteration checkpoint binding differs")
                initial = saved["optimizer_state"] if saved else old["result"] if old else None
                result = fit_inverse_ising(bags[i], ridge=config["ridge"], max_iterations=config["max_iterations"],
                    tolerance=config["tolerance"], initial=initial, callback=checkpoint)
                spent = (old["seconds"] if old else 0.0)+time.perf_counter()-before
                _save_checked(path, {"binding": binding, "seconds": spent, "result": result})
            if result["stop_reason"] == "paused":
                return {"state": "paused"}
            fitted.append(result["fields"])
            solver_rows.append({"uid": item["hamiltonian_uid"], "converged": result["converged"],
                                "stop_reason": result["stop_reason"], "seconds": spent})
        native = np.asarray(fitted)
        prediction = np.abs(native).reshape(len(native), 4, 4, 2).transpose(0, 3, 1, 2)
        binding = {"plan_sha256": digest(plan), "method": "inverse_ising", "role": "fresh_confirmation_raw",
                   "records": M, "ridge": config["ridge"], "true_fields_as_solver_inputs": False,
                   "record_selection_sha256": sha(base/"record-selection.json")}
        evaluate_predictions(4, prediction, data["fields"], data["ids"], data["uids"], base/f"M{M}-inverse"/"evaluation",
                             binding, native_signed=native)
        atomic_json(base/f"M{M}-inverse"/"solver-summary.json", {"instances": solver_rows,
                    "failures_retained": sum(not x["converged"] for x in solver_rows),
                    "scope": "same-machine cold per-H solver work, including iteration checkpoint I/O; neural training amortization not inferred"})
    atomic_json(base/"complete.json", {"plan_sha256": digest(plan), "state": "complete"})
    return {"state": "complete"}
