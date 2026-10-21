"""Nested training-Hamiltonian scaling with an immutable inverse-Ising reference.

The tested snapshot training/session and scoring primitives are reused unchanged.
Only the training prefix varies: configurations, validation Hamiltonians, model,
optimizer and epoch policy stay fixed. This module never fits the inverse model.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal

import numpy as np
import torch

from quantum_lattice.data.snapshot_data import SnapshotCache, SnapshotSplit
from quantum_lattice.models import snapshot_campaign as pilot


ARTIFACT_ROOT = Path("codex material/artifacts/snapshot_learning/scaling")
NEURAL_METHODS = pilot.NEURAL_METHODS
_json, _read, _sha, _digest, _csv = pilot._json, pilot._read, pilot._sha, pilot._digest, pilot._csv


def _relative(value, name):
    if (not isinstance(value, str) or not value or Path(value).is_absolute()
            or ".." in Path(value).parts):
        raise ValueError(f"{name} must be a safe repository-relative path")


def validate_config(config):
    """Validate the explicit scaling schema without changing engine defaults."""
    value = json.loads(pilot._io._canonical(config))
    expected = {"schema_version", "L", "feature_cache", "output_root", "selection",
                "seeds", "training", "inverse_ising", "inverse_reference", "execution", "provenance"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Invalid snapshot scaling settings")
    selection = value["selection"]
    keys = {"selection_seed", "train_hamiltonian_counts", "validation_hamiltonians", "shots"}
    if not isinstance(selection, dict) or set(selection) != keys:
        raise ValueError("Scaling selection requires nested training counts and one shot budget")
    counts = selection["train_hamiltonian_counts"]
    if not isinstance(counts, list) or not counts:
        raise ValueError("train_hamiltonian_counts must be nonempty")
    for count in counts:
        pilot._positive(count, "train_hamiltonian_count")
    if counts != sorted(set(counts)):
        raise ValueError("train_hamiltonian_counts must be increasing and distinct")
    pilot._positive(selection["shots"], "shots")
    execution = value["execution"]
    if not isinstance(execution, dict) or set(execution) != {"checkpoint_updates"}:
        raise ValueError("Scaling execution only needs checkpoint_updates; inverse fits are reused")
    reference = value["inverse_reference"]
    if (not isinstance(reference, dict)
            or set(reference) != {"campaign_root", "task_key"}
            or not isinstance(reference["task_key"], str)
            or not reference["task_key"] or Path(reference["task_key"]).name != reference["task_key"]):
        raise ValueError("inverse_reference requires a campaign_root and safe task_key")
    for name in ("feature_cache", "output_root"):
        _relative(value[name], name)
    _relative(reference["campaign_root"], "inverse_reference.campaign_root")
    # The original validator owns the exact training/solver settings. Adapt only
    # its selection and operational fields, rather than maintaining a second copy.
    adapted = {key: item for key, item in value.items() if key != "inverse_reference"}
    adapted["selection"] = _cache_selection(value)
    adapted["selection"].pop("L")
    adapted["execution"] = {**execution, "inverse_workers": 1}
    pilot.validate_config(adapted)
    return value


def _cache_selection(config):
    selection = config["selection"]
    return {"L": config["L"], "selection_seed": selection["selection_seed"],
            "train_hamiltonians": max(selection["train_hamiltonian_counts"]),
            "validation_hamiltonians": selection["validation_hamiltonians"],
            "shot_counts": [selection["shots"]]}


def _safe_run(root, run):
    allowed = (root / ARTIFACT_ROOT).resolve()
    path = Path(run).resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError(f"Scaling runs must be below {allowed}")
    return path


def _sources(root):
    names = ("src/quantum_lattice/models/snapshot_scaling.py",
             "src/quantum_lattice/models/snapshot_scaling_terminal.py",
             "experiments/snapshot_scaling/train.py", "configs/snapshot_scaling.json")
    return {**pilot._sources(root), **{name: _sha(root / name) for name in names}}


def _tasks(config):
    return [{"key": f"L4-H{count}-M{config['selection']['shots']}-seed{seed}-{method}",
             "method": method, "shots": config["selection"]["shots"], "seed": seed,
             "train_hamiltonians": count}
            for count in config["selection"]["train_hamiltonian_counts"]
            for seed in config["seeds"] for method in NEURAL_METHODS]


def prefix_cache(cache, count):
    """A read-only prefix view: no resampling or field-derived inputs."""
    if type(count) is not int or not 1 <= count <= len(cache.train):
        raise ValueError("Training prefix exceeds the shared cache")
    train = cache.train
    split = SnapshotSplit(**{name: getattr(train, name)[:count]
                             for name in ("states", "fields", "ids", "uids", "record_indices")})
    return SnapshotCache(split, cache.validation, cache.metadata, cache.L, cache.shot_counts)


def _reference(root, config, cache):
    """Verify old provenance and exact validation bags before freezing reuse."""
    reference = config["inverse_reference"]
    run = pilot._safe_run(root, root / reference["campaign_root"])
    plan = pilot.load_plan(root, run)
    old_config = plan["config"]
    if (old_config["L"] != config["L"]
            or old_config["selection"]["selection_seed"] != config["selection"]["selection_seed"]
            or old_config["inverse_ising"] != config["inverse_ising"]):
        raise ValueError("Inverse reference geometry, record seed or solver differs")
    matches = [task for task in plan["tasks"] if task["key"] == reference["task_key"]]
    if (len(matches) != 1 or matches[0]["method"] != "inverse_ising"
            or matches[0]["shots"] != config["selection"]["shots"]):
        raise ValueError("Inverse reference must match the fixed record budget")
    task = matches[0]
    job = run / "models" / task["key"]
    marker = pilot._done(job, task, plan)
    if marker is None:
        raise ValueError("Inverse reference is not complete")
    old_cache = pilot.load_snapshot_cache(run / "cache")
    shots = task["shots"]
    for name in ("ids", "uids", "fields", "record_indices", "states"):
        old = getattr(old_cache.validation, name)
        new = getattr(cache.validation, name)
        if name in ("record_indices", "states"):
            old, new = old[:, :shots], new[:, :shots]
        if not np.array_equal(old, new):
            raise ValueError(f"Inverse reference validation {name} differ")
    if len(old_cache.train) > len(cache.train):
        raise ValueError("Shared scaling cache must include the old training prefix")
    for name in ("ids", "uids", "fields", "record_indices", "states"):
        old = getattr(old_cache.train, name)
        new = getattr(cache.train, name)[:len(old_cache.train)]
        if name in ("record_indices", "states"):
            old, new = old[:, :shots], new[:, :shots]
        if not np.array_equal(old, new):
            raise ValueError(f"Original training prefix {name} differ")
    with np.load(job / "predictions.npz", allow_pickle=False) as archive:
        for key, expected in (("hamiltonian_ids", cache.validation.ids),
                              ("hamiltonian_uids", cache.validation.uids),
                              ("record_indices", cache.validation.record_indices[:, :shots]),
                              ("target", np.abs(cache.validation.fields).astype(np.float32))):
            if not np.array_equal(archive[key], expected):
                raise ValueError(f"Inverse prediction {key} binding differs")
        native = archive["native_signed_fields"]
        prediction = archive["prediction"]
        if (native.shape != cache.validation.fields.shape or not np.isfinite(native).all()
                or not np.array_equal(prediction, np.abs(native).astype(np.float32))):
            raise ValueError("Inverse native signed fields do not match magnitude predictions")
        mse = float(np.mean((prediction.astype(np.float64)
                             - archive["target"].astype(np.float64)) ** 2))
    if not np.isclose(mse, marker["summary"]["metrics"]["mse"], rtol=0, atol=1e-14):
        raise ValueError("Inverse reference MSE does not reproduce")
    paths = ["plan.json", "plan-sha256.json", "models/" + task["key"] + "/done.json"]
    paths += ["models/" + task["key"] + "/" + name
              for name in (*marker["hashes"], *marker["instance_hashes"])]
    paths += ["cache/" + name for name in plan["cache_hashes"]]
    return {"schema_version": 1, "campaign_root": str(run.relative_to(root)),
            "task": task, "plan_sha256": _digest(plan),
            "file_hashes": {name: _sha(run / name) for name in paths},
            "summary": marker["summary"], "validation_hamiltonians": len(cache.validation),
            "reused_without_refitting": True, "repeated_across_training_counts": False}


def _verify_reference(root, binding):
    run = pilot._safe_run(root, root / binding["campaign_root"])
    plan = pilot.load_plan(root, run)
    if _digest(plan) != binding["plan_sha256"]:
        raise ValueError("Frozen inverse reference plan changed")
    for name, digest in binding["file_hashes"].items():
        if _sha(run / name) != digest:
            raise ValueError(f"Frozen inverse reference changed: {name}")


def prepare(repo_root, config, *, should_stop=None, progress=None):
    """Prepare the largest shared cache and freeze a queue; no optimization."""
    root = Path(repo_root).resolve()
    config = validate_config(config)
    run = _safe_run(root, root / config["output_root"])
    run.mkdir(parents=True, exist_ok=True)
    with pilot._io.campaign_lock(run):
        if (run / "plan.json").exists():
            plan = load_plan(root, run)
            if plan["config"] != config:
                raise ValueError("Config differs from this frozen scaling run; use a new output_root")
            return run
        feature = (root / config["feature_cache"]).resolve()
        if not feature.is_relative_to((root / "codex material/artifacts").resolve()):
            raise ValueError("Feature provenance must be inside ignored artifacts")
        pilot.prepare_snapshot_cache(root, feature, run / "cache", _cache_selection(config),
                                     progress=progress, should_stop=should_stop)
        if should_stop and should_stop():
            return run
        cache = pilot.load_snapshot_cache(run / "cache")
        selection = _cache_selection(config)
        if (cache.L != config["L"] or len(cache.train) != selection["train_hamiltonians"]
                or len(cache.validation) != selection["validation_hamiltonians"]
                or list(cache.shot_counts) != selection["shot_counts"]):
            raise ValueError("Prepared scaling cache differs from declared geometry/counts/records")
        reference = _reference(root, config, cache)
        plan = {"schema_version": 1, "workflow": "snapshot_training_scaling",
                "config": config, "tasks": _tasks(config), "source_hashes": _sources(root),
                "cache_hashes": pilot._cache_hashes(run), "inverse_reference": reference,
                "model_specs": {method: pilot.snapshot_model_spec(
                    method, shot_chunk_size=config["training"]["shot_chunk_size"],
                    activation_checkpoint=config["training"]["activation_checkpoint"])
                    for method in NEURAL_METHODS},
                "data": {"L": cache.L, "train_hamiltonians": len(cache.train),
                         "train_hamiltonian_counts": config["selection"]["train_hamiltonian_counts"],
                         "validation_hamiltonians": len(cache.validation),
                         "shot_counts": list(cache.shot_counts), "target": "abs(b_z)",
                         "fields_used_as_inputs": False, "test_payload_loaded": False},
                "updates_per_epoch": {str(count): (count + config["training"]["batch_size_hamiltonians"] - 1)
                                      // config["training"]["batch_size_hamiltonians"]
                                      for count in config["selection"]["train_hamiltonian_counts"]},
                "protocol": {
                    "scope": "L4 development scaling; reused validation; no final test",
                    "nested_training": "UID-hash ordered prefixes of one largest cache; fixed records per UID",
                    "same_initialization": "paired methods/counts share initialization seed; no warm starts",
                    "checkpoint_selection": "strict lowest own-run validation magnitude MSE including epoch0",
                    "D5": "secondary reference-sign oracle; inverse native signed D5 separate",
                    "inverse": "one immutable matched validation fit group reused, never refitted or seed replicated",
                    "epoch_policy": f"same {config['training']['max_epochs']}-epoch maximum and stopping rule; larger sets receive more updates per epoch, not a fixed-compute comparison",
                    "GPU_binding": "first neural run binds exact device and environment",
                    "preparation": "zero optimizer updates and zero inverse fits"}}
        plan["max_optimizer_updates"] = {count: updates * config["training"]["max_epochs"]
                                          for count, updates in plan["updates_per_epoch"].items()}
        _json(run / "plan-sha256.json", {"sha256": _digest(plan)})
        _json(run / "plan.json", plan)
        _json(run / "inverse-reference.json", reference)
        _report(run, plan)
    return run


def load_plan(repo_root, run):
    root = Path(repo_root).resolve()
    run = _safe_run(root, run)
    plan = _read(run / "plan.json")
    if (plan.get("schema_version") != 1 or plan.get("workflow") != "snapshot_training_scaling"
            or _read(run / "plan-sha256.json") != {"sha256": _digest(plan)}):
        raise ValueError("Invalid frozen snapshot scaling plan")
    config = validate_config(plan["config"])
    if (plan["tasks"] != _tasks(config) or plan["source_hashes"] != _sources(root)
            or plan["cache_hashes"] != pilot._cache_hashes(run)):
        raise ValueError("Scaling sources, tasks or cached data changed; restore or use a new campaign")
    if _read(run / "inverse-reference.json") != plan["inverse_reference"]:
        raise ValueError("Frozen inverse reference binding changed")
    _verify_reference(root, plan["inverse_reference"])
    return plan


def _check_progress_checkpoint(run, task, plan, payload):
    if payload.get("task") != task or payload.get("plan_sha256") != _digest(plan):
        raise ValueError("Scaling progress checkpoint identity changed")
    binding_path = run / "device-binding.json"
    if (not binding_path.exists()
            or payload.get("device_binding_sha256") != _digest(_read(binding_path))):
        raise ValueError("Scaling checkpoint device binding changed")


def _report(run, plan, *, persist=True):
    rows = []
    for task in plan["tasks"]:
        job = run / "models" / task["key"]
        done = pilot._done(job, task, plan)
        row = {**task, "state": "complete" if done else "pending"}
        if done:
            payload = torch.load(job / "latest.pt", map_location="cpu", weights_only=False)
            _check_progress_checkpoint(run, task, plan, payload)
            row.update(done["summary"])
        elif (job / "latest.pt").exists():
            payload = torch.load(job / "latest.pt", map_location="cpu", weights_only=False)
            _check_progress_checkpoint(run, task, plan, payload)
            engine = payload["session"]["engine"]
            row.update(state="paused", epoch=engine["epoch"], step=engine["step"])
        rows.append(row)
    status = {"scope": "snapshot L4 training-data scaling; reused validation; no final test",
              "completed": sum(row["state"] == "complete" for row in rows),
              "total": len(rows), "tasks": rows, "test_payload_loaded": False,
              "train_hamiltonian_counts": plan["config"]["selection"]["train_hamiltonian_counts"],
              "shots": plan["config"]["selection"]["shots"],
              "inverse_reference": {"state": "complete", **plan["inverse_reference"],
                                    "metrics": plan["inverse_reference"]["summary"]["metrics"],
                                    "converged_instances": plan["inverse_reference"]["summary"]["converged_instances"],
                                    "source": plan["inverse_reference"]["campaign_root"]}}
    if not persist:
        return status
    _json(run / "status.json", status)
    flat = [{"train_hamiltonians": row["train_hamiltonians"], "method": row["method"],
             "shots": row["shots"], "seed": row["seed"], "mse": row["metrics"]["mse"],
             "rmse": row["metrics"]["rmse"], "mae": row["metrics"]["mae"],
             "D5_reference_sign": row["metrics"]["reference_sign_D5"]["two_channel_full"],
             "elapsed_seconds": row["elapsed_seconds"], "stop_reason": row["stop_reason"],
             "provenance": "new matched CNN fit"} for row in rows if row["state"] == "complete"]
    baseline = plan["inverse_reference"]["summary"]
    flat.append({"train_hamiltonians": None, "method": "inverse_ising", "shots": plan["config"]["selection"]["shots"],
                 "seed": None, "mse": baseline["metrics"]["mse"], "rmse": baseline["metrics"]["rmse"],
                 "mae": baseline["metrics"]["mae"],
                 "D5_reference_sign": baseline["metrics"]["reference_sign_D5"]["two_channel_full"],
                 "elapsed_seconds": baseline["elapsed_seconds"], "stop_reason": "completed prior reference",
                 "provenance": "immutable prior fit reused once; no new inverse fit"})
    _csv(run / "comparison.csv", flat, tuple(flat[0]))
    text = "# L4 training-data scaling\n\nReused development validation; no final test.\n\n"
    text += f"CNN jobs complete: {status['completed']}/{status['total']}. One matched inverse reference is reused without refitting.\n\n"
    text += "| Training H | Method | Seed | State | Validation MSE |\n|---:|---|---:|---|---:|\n"
    for row in rows:
        mse = row["metrics"]["mse"] if row["state"] == "complete" else "—"
        text += f"| {row['train_hamiltonians']} | {row['method']} | {row['seed']} | {row['state']} | {mse} |\n"
    text += ("\nThe same epoch policy allows more optimizer updates with larger training sets; this is not fixed-compute scaling. "
             "Files: plan.json and inverse-reference.json; cache/ shared largest bank; device-binding.json after first CUDA run; "
             "models/<task>/ exact-resume checkpoints, selected predictions, histories, per-H errors; comparison.csv; status.json.\n")
    (run / "RESULTS.md").write_text(text)
    return status


def progress(repo_root, config):
    """Read-only monitoring; no cache/source/reference files are changed."""
    root = Path(repo_root).resolve()
    config = validate_config(config)
    run = _safe_run(root, root / config["output_root"])
    if not (run / "plan.json").exists():
        path = run / "cache/progress.json"
        return {"state": "preparing" if path.exists() else "not_prepared", "completed": 0,
                "total": len(_tasks(config)), "tasks": [],
                "train_hamiltonian_counts": config["selection"]["train_hamiltonian_counts"],
                "shots": config["selection"]["shots"],
                "data_progress": _read(path) if path.exists() else None,
                "inverse_reference": {"state": "not_verified"}}
    plan = load_plan(root, run)
    if plan["config"] != config:
        raise ValueError("Config differs from the frozen scaling run")
    return _report(run, plan, persist=False)


def run_campaign(repo_root, config, *, progress_callback=None):
    """Resume each independent CUDA model; Ctrl+C saves bounded update chunks."""
    root = Path(repo_root).resolve()
    config = validate_config(config)
    requested = {"stop": False}
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: requested.update(stop=True))
    should_stop = lambda: requested["stop"]
    try:
        run = prepare(root, config, should_stop=should_stop, progress=progress_callback)
        if should_stop() or not (run / "plan.json").exists():
            return progress(root, config)
        with pilot._io.campaign_lock(run):
            plan = load_plan(root, run)
            cache = pilot.load_snapshot_cache(run / "cache")
            for task in plan["tasks"]:
                if should_stop():
                    break
                job = pilot._job(run, task)
                if pilot._done(job, task, plan):
                    continue
                subset = prefix_cache(cache, task["train_hamiltonians"])
                pilot._run_neural(run, plan, subset, task, should_stop, progress_callback)
                _report(run, plan)
            return _report(run, plan)
    finally:
        signal.signal(signal.SIGINT, previous)
