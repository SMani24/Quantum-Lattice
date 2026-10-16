"""Paired, resumable estimator-only MLP ablation on development groups.

The completed simulation banks and baseline models are never modified. Only
train/validation feature views are accepted; exact features are evaluation-only.
One GPU model is active at a time and a checkpoint commits history and optimizer
state together. The frozen task list prevents selecting favorable seeds.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch

from quantum_lattice.data.estimator_features import prepare_feature_views, load_feature_views
from quantum_lattice.metrics.regression import per_hamiltonian_metrics
from quantum_lattice.models.mlp_training import TrainSession, resolve_device, _config

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = Path("codex material/artifacts/estimator_mlp")
VIEWS = ("f", "g", "exact")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(path, writer, binary=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb" if binary else "w") as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _json(path, value):
    _atomic(path, lambda handle: handle.write(json.dumps(value, indent=2, sort_keys=True,
                                                       allow_nan=False) + "\n"))


def _torch(path, value):
    _atomic(path, lambda handle: torch.save(value, handle), binary=True)


def _npz(path, **values):
    _atomic(path, lambda handle: np.savez_compressed(handle, **values), binary=True)


def _csv(path, rows, names):
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    _atomic(path, write)


@contextmanager
def campaign_lock(run):
    """A process-wide lease, covering preparation, updates and output commits."""
    import fcntl
    with (Path(run) / ".campaign.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This estimator MLP campaign is already active") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _safe_run(repo_root, run):
    root = (Path(repo_root) / ARTIFACT_ROOT).resolve()
    run = Path(run).resolve()
    if run == root or not run.is_relative_to(root):
        raise ValueError(f"Campaign must be below {root}")
    return run


def _environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "machine": platform.machine(), "platform": platform.platform()}


def _accelerator(device):
    if device.type != "cuda":  # Synthetic tests explicitly substitute the resolver.
        return {"type": device.type}
    properties = torch.cuda.get_device_properties(device)
    return {"type": "cuda", "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _sources(repo_root):
    paths = ("models/estimator_mlp.py", "models/mlp.py", "models/mlp_training.py",
             "data/estimator_features.py", "data/mlp_data.py", "metrics/regression.py",
             "data/estimator_validation.py", "data/state_storage.py",
             "simulation/estimator_validation.py", "simulation/solvable.py",
             "observables/conditioned.py", "observables/statistics.py",
             "lattice/__init__.py", "lattice/topology.py", "lattice/builders.py", "lattice/square.py")
    return {f"src/quantum_lattice/{name}": _sha(Path(repo_root) / "src/quantum_lattice" / name)
            for name in paths}


def validate_config(config):
    value = json.loads(_canonical(config))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported estimator MLP schema")
    if value.get("sizes") != [3, 4] or value.get("mode") != "all_edges":
        raise ValueError("This declared development batch requires L3/L4 and all_edges")
    seeds = value.get("seeds")
    if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
                   for seed in seeds)):
        raise ValueError("seeds must be distinct nonnegative integers below 2**63")
    training = value.get("training")
    if not isinstance(training, dict) or "seed" in training:
        raise ValueError("training requires shared settings; paired seeds belong in seeds")
    normalized = _config({**training, "seed": seeds[0]})
    if normalized["device"] != "cuda":
        raise ValueError("This campaign requires device=cuda; there is no CPU fallback")
    # Store the resolved defaults so changing an engine default cannot change a run.
    normalized.pop("seed")
    value["training"] = normalized
    checkpoint = value.get("execution", {}).get("checkpoint_steps")
    if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint < 1:
        raise ValueError("checkpoint_steps must be positive")
    for name in ("output_root", "validation_run"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"{name} is required")
    return value


def prepare(repo_root, config, *, run_dir=None):
    """Verify/cache non-test summaries, then freeze the exact paired task list.

    This performs no Monte Carlo and no optimizer updates. Feature manifests are
    immutable and bind their original fields, split identities and source hashes.
    """
    repo_root = Path(repo_root).resolve()
    config = validate_config(config)
    run = _safe_run(repo_root, run_dir or repo_root / config["output_root"])
    run.mkdir(parents=True, exist_ok=True)
    with campaign_lock(run):
        if (run / "plan.json").exists():
            plan = load_plan(repo_root, run)
            if plan["config"] != config:
                raise ValueError("Configuration differs from the frozen campaign")
            return run
        validation_run = Path(config["validation_run"])
        if not validation_run.is_absolute():
            validation_run = repo_root / validation_run
        prepare_feature_views(repo_root, validation_run, run / "features", sizes=tuple(config["sizes"]))
        tasks = [{"key": f"L{L}-seed{seed}-{view}", "L": L, "seed": seed, "view": view}
                 for L in config["sizes"] for seed in config["seeds"] for view in ("f", "g")]
        plan = {"schema_version": 1, "config": config, "tasks": tasks,
                "source_hashes": _sources(repo_root), "environment": _environment(),
                "feature_manifest_sha256": _sha(run / "features/manifest.json"),
                "protocol": {"scope": "development train/validation only; no final-test payloads",
                             "input_order": ["mean_A_u", "mean_A_v", "mean_Au_Av"],
                             "target": "abs(b_z); true fields are not network inputs",
                             "selection": "strictly lowest own-view validation MSE; raw predictions",
                             "paired_seeds": "identical model initialization and shuffled row order",
                             "crossed_evaluation": "each selected f/g model on f, g and exact validation inputs",
                             "exact_view": "evaluation only; never training or model selection",
                             "confirmation": "fresh unopened bank required for future confirmation",
                             "stopping": "same patience/min_delta/min_steps; each own-view validation",
                             "baseline_preservation": "original banks and fourteen models untouched"}}
        _json(run / "plan-sha256.json", {"sha256": _digest(plan)})
        _json(run / "plan.json", plan)  # Plan is the final preparation commit marker.
        _report_locked(run, plan)
    return run


def load_plan(repo_root, run):
    run = _safe_run(repo_root, run)
    plan = _read(run / "plan.json")
    if plan.get("schema_version") != 1 or _read(run / "plan-sha256.json") != {"sha256": _digest(plan)}:
        raise ValueError("Frozen campaign plan hash/schema mismatch")
    if validate_config(plan["config"]) != plan["config"]:
        raise ValueError("Frozen configuration is not normalized")
    config = plan["config"]
    tasks = [{"key": f"L{L}-seed{seed}-{view}", "L": L, "seed": seed, "view": view}
             for L in config["sizes"] for seed in config["seeds"] for view in ("f", "g")]
    if plan["tasks"] != tasks:
        raise ValueError("Frozen seed/task selection changed")
    if plan["source_hashes"] != _sources(repo_root):
        raise ValueError("Scientific source changed; preserve this run and prepare a new campaign")
    if plan["environment"] != _environment():
        raise ValueError("Scientific environment changed; exact continuation is unavailable")
    if plan["feature_manifest_sha256"] != _sha(run / "features/manifest.json"):
        raise ValueError("Frozen feature manifest changed")
    return plan


def _task_path(run, task):
    return Path(run) / "models" / task["key"]


def _done(job, plan, task):
    path = job / "done.json"
    if not path.exists():
        return None
    done = _read(path)
    if (done.get("schema_version") != 1 or done.get("plan_sha256") != _digest(plan)
            or done.get("task") != task):
        raise ValueError(f"Completed model identity mismatch: {task['key']}")
    expected = {"latest.pt", "best.pt", "history.json", "evaluation.json",
                "per-hamiltonian.csv", "predictions.npz"}
    if set(done.get("hashes", {})) != expected:
        raise ValueError("Completed model output inventory differs")
    for name, checksum in done["hashes"].items():
        if _sha(job / name) != checksum:
            raise ValueError(f"Completed model output changed: {task['key']}/{name}")
    evaluation = _read(job / "evaluation.json")
    if done.get("summary") != evaluation or evaluation.get("task") != task:
        raise ValueError("Completed marker summary differs from its verified evaluation")
    return done


def _checkpoint(job, session, history, plan, task, elapsed, accelerator):
    payload = {"workflow_schema_version": 1, "plan_sha256": _digest(plan), "task": task,
               "session": session.snapshot(), "history": history,
               "elapsed_seconds": elapsed, "accelerator": accelerator}
    _torch(job / "latest.pt", payload)
    return payload


def _load_checkpoint(job, plan, task):
    if not (job / "latest.pt").exists():
        return None
    payload = torch.load(job / "latest.pt", map_location="cpu", weights_only=True)
    if (payload.get("workflow_schema_version") != 1 or payload.get("plan_sha256") != _digest(plan)
            or payload.get("task") != task):
        raise ValueError(f"Checkpoint identity mismatch: {task['key']}")
    history = payload.get("history")
    if (not isinstance(history, list)
            or any(not isinstance(event, dict) or not isinstance(event.get("step"), int) for event in history)
            or any(left["step"] >= right["step"] for left, right in zip(history, history[1:]))
            or history and history[-1]["step"] > payload["session"]["step"]):
        raise ValueError("Checkpoint history/counters disagree")
    return payload


def _check_views(views, L):
    if set(views) != set(VIEWS):
        raise ValueError("f/g/exact feature views are all required")
    base = views["f"]
    for name, data in views.items():
        if data.L != L or data.mode != "all_edges" or data.test is not None:
            raise ValueError("Training adapter must expose all_edges train/validation only")
        for split in ("train", "validation"):
            left, right = getattr(base, split), getattr(data, split)
            for field in ("y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
                if not np.array_equal(getattr(left, field), getattr(right, field)):
                    raise ValueError(f"Paired {split} row identity/labels differ in {name}")
        if set(data.train.hamiltonian_uids) & set(data.validation.hamiltonian_uids):
            raise ValueError("Hamiltonian leakage between training and validation")
    return views


def _evaluate(job, session, history, views, plan, task, elapsed, accelerator):
    predictions, per_h, scores = {}, [], {}
    for view in VIEWS:
        rows = views[view].validation
        prediction = session.predict(rows.x, use_best=True)
        predictions[f"prediction_{view}"] = prediction
        metrics = per_hamiltonian_metrics(prediction, rows.y, rows.hamiltonian_ids, rows.edge_ids)
        for item in metrics:
            per_h.append({"input_view": view, **item})
        error = prediction.astype(np.float64) - rows.y.astype(np.float64)
        mse = float(np.mean([m["mse"] for m in metrics]))
        scores[view] = {"mse": mse, "rmse": math.sqrt(mse),
                        "mae": float(np.mean([m["mae"] for m in metrics])),
                        "row_mse": float(np.mean(error ** 2)),
                        "rows": len(rows), "hamiltonians": len(metrics),
                        "mean_hamiltonian_rmse": float(np.mean([m["rmse"] for m in metrics]))}
    rows = views["f"].validation
    _npz(job / "predictions.npz", **predictions, target=rows.y,
         hamiltonian_ids=rows.hamiltonian_ids, hamiltonian_uids=rows.hamiltonian_uids,
         edge_ids=rows.edge_ids)
    _csv(job / "per-hamiltonian.csv", per_h,
         ("input_view", "hamiltonian_id", "count", "mse", "rmse", "mae"))
    _json(job / "history.json", history)
    evaluation = {"task": task, "scope": "development validation; not a new final test",
                  "checkpoint_selection": "lowest own-view validation MSE",
                  "best_step": session.best_step, "best_validation_mse": session.best_val_mse,
                  "last_step": session.step, "stop_reason": session.stop_reason,
                  "elapsed_seconds": elapsed, "elapsed_scope": "optimizer advances and validation only; excludes loading, checkpoints and exports",
                  "accelerator": accelerator, "scores": scores,
                  "prediction_policy": "raw unclipped linear predictions",
                  "exact_input_policy": "crossed evaluation only; never checkpoint selection",
                  "test_payload_loaded": False}
    _json(job / "evaluation.json", evaluation)
    _torch(job / "best.pt", {"schema_version": 1, "plan_sha256": _digest(plan),
                             "task": task, "best_step": session.best_step,
                             "model_state_dict": session.best_state_dict})
    names = ("latest.pt", "best.pt", "history.json", "evaluation.json",
             "per-hamiltonian.csv", "predictions.npz")
    done = {"schema_version": 1, "plan_sha256": _digest(plan), "task": task,
            "hashes": {name: _sha(job / name) for name in names}, "summary": evaluation}
    _json(job / "done.json", done)  # Commit marker is written after every output is durable.
    return done


def _summary(run, plan):
    rows = []
    for task in plan["tasks"]:
        job = _task_path(run, task)
        done = _done(job, plan, task)
        payload = None if done else _load_checkpoint(job, plan, task)
        record = {**task, "state": "complete" if done else "paused" if payload else "pending",
                  "step": 0, "best_step": None, "best_validation_mse": None}
        if done:
            result = done["summary"]
            record.update(step=result["last_step"], best_step=result["best_step"],
                          best_validation_mse=result["best_validation_mse"], scores=result["scores"],
                          stop_reason=result["stop_reason"])
        elif payload:
            state = payload["session"]
            record.update(step=state["step"], best_step=state["best_step"],
                          best_validation_mse=state["best_val_mse"] if math.isfinite(state["best_val_mse"]) else None)
        rows.append(record)
    completed = sum(item["state"] == "complete" for item in rows)
    return {"schema_version": 1, "state": "complete" if completed == len(rows) else "paused",
            "completed": completed, "total": len(rows), "tasks": rows,
            "scope": "development validation only", "test_payload_loaded": False}


def format_summary(summary):
    lines = [f"Estimator MLP: {summary['completed']}/{summary['total']} complete ({summary['state']})",
             "  L  seed        view  state      updates  best step  best validation MSE"]
    for item in summary["tasks"]:
        best = "—" if item["best_validation_mse"] is None else f"{item['best_validation_mse']:.7g}"
        lines.append(f" {item['L']:2}  {item['seed']:<10}  {item['view']:>4}  {item['state']:<9}"
                     f" {item['step']:>7}  {str(item['best_step'] or '—'):>9}  {best}")
    lines.append("Paired validation: equal-seed averages, descriptive; negative g-f favors g.")
    complete = {(item["L"], item["seed"], item["view"]): item
                for item in summary["tasks"] if item["state"] == "complete"}
    for L in sorted({item["L"] for item in summary["tasks"]}):
        seeds = sorted({seed for size, seed, view in complete
                        if size == L and (L, seed, "f") in complete and (L, seed, "g") in complete})
        if not seeds:
            lines.append(f"  L={L}: 0 completed f/g seed pairs")
            continue
        f_pipeline = float(np.mean([complete[L, seed, "f"]["scores"]["f"]["mse"] for seed in seeds]))
        g_pipeline = float(np.mean([complete[L, seed, "g"]["scores"]["g"]["mse"] for seed in seeds]))
        exact_difference = float(np.mean([
            complete[L, seed, "g"]["scores"]["exact"]["mse"]
            - complete[L, seed, "f"]["scores"]["exact"]["mse"] for seed in seeds]))
        lines.append(f"  L={L}: {len(seeds)} paired seeds; pipeline f→f MSE={f_pipeline:.7g}; "
                     f"g→g MSE={g_pipeline:.7g}; g-f={g_pipeline - f_pipeline:+.7g}")
        lines.append(f"    Common exact inputs: g-trained minus f-trained MSE={exact_difference:+.7g}")
    return "\n".join(lines)


def _report_locked(run, plan):
    summary = _summary(run, plan)
    _json(run / "status.json", summary)
    comparison = []
    complete = {(r["L"], r["seed"], r["view"]): r for r in summary["tasks"] if r["state"] == "complete"}
    for L in plan["config"]["sizes"]:
        for seed in plan["config"]["seeds"]:
            if (L, seed, "f") not in complete or (L, seed, "g") not in complete:
                continue
            f, g = complete[L, seed, "f"], complete[L, seed, "g"]
            for view in (*VIEWS, "matched_pipeline"):
                left = f["scores"]["f" if view == "matched_pipeline" else view]
                right = g["scores"]["g" if view == "matched_pipeline" else view]
                comparison.append({"L": L, "seed": seed, "validation_input": view,
                                   "f_mse": left["mse"], "g_mse": right["mse"],
                                   "g_minus_f_mse": right["mse"] - left["mse"],
                                   "relative_mse_reduction": 1 - right["mse"] / left["mse"] if left["mse"] else None})
    _csv(run / "paired-validation.csv", comparison,
         ("L", "seed", "validation_input", "f_mse", "g_mse", "g_minus_f_mse", "relative_mse_reduction"))
    text = ("# Matched estimator MLP development batch\n\n"
            f"State: **{summary['state']}**; **{summary['completed']}/{summary['total']}** models complete.\n\n"
            "This batch isolates simulated feature estimator f versus conditioned g. "
            "The paper-style MLP, Hamiltonian groups, all-edge rows and paired seeds are fixed. "
            "Each model is selected by its lowest own-view validation MSE, then evaluated "
            "on the same f, g and exact validation Hamiltonians. Exact inputs never train "
            "or select models. Original baseline models and banks remain intact.\n\n"
            "These are development validation results, not a fresh held-out confirmation. "
            f"All {len(plan['config']['seeds'])} seeds are reported; no favorable seed is selected. This estimates "
            "field magnitudes, not signs, D5 Hamiltonian distance or logical error rates.\n\n"
            "## Files\n\n"
            "- `plan.json`, `plan-sha256.json`: frozen source/environment/settings/task identities.\n"
            "- `features/`: immutable train/validation f/g/exact views and provenance manifests.\n"
            "- `models/L*-seed*-f/` and `*-g/`: atomic `latest.pt` with optimizer/RNG/history; "
            "finished jobs add `best.pt`, `history.json`, `evaluation.json`, "
            "`per-hamiltonian.csv`, `predictions.npz`, and hash-verified `done.json`.\n"
            "- `status.json`: derived progress; `paired-validation.csv`: paired seed/input comparisons.\n"
            "- `execution.log`: terminal progress when launched through the runner.\n\n"
            "## Progress\n\n```text\n" + format_summary(summary) + "\n```\n")
    _atomic(run / "RESULTS.md", lambda handle: handle.write(text))
    return summary


def summarize_campaign(repo_root, run):
    run = _safe_run(repo_root, run)
    with campaign_lock(run):
        return _report_locked(run, load_plan(repo_root, run))


def run_campaign(repo_root, run, *, max_chunks=None, output=print):
    """Train sequentially; SIGINT drains one bounded chunk and commits a pause."""
    if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
        raise ValueError("max_chunks must be positive")
    repo_root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    # CUDA availability is checked before any feature payload is loaded.
    device = resolve_device("cuda", True)
    accelerator = _accelerator(device)
    with campaign_lock(run):
        plan = load_plan(repo_root, run)
        stopped = [False]
        old_handler = signal.getsignal(signal.SIGINT)
        def request_pause(signum, frame):
            stopped[0] = True
        signal.signal(signal.SIGINT, request_pause)
        chunks, current_L, views = 0, None, None
        try:
            for task in plan["tasks"]:
                job = _task_path(run, task)
                if _done(job, plan, task):
                    continue
                if stopped[0] or max_chunks is not None and chunks >= max_chunks:
                    break
                if current_L != task["L"]:
                    views = _check_views(load_feature_views(run / "features", task["L"]), task["L"])
                    current_L = task["L"]
                data = views[task["view"]]
                payload = _load_checkpoint(job, plan, task)
                if payload and payload["accelerator"] != accelerator:
                    raise ValueError("Accelerator changed; exact continuation requires the original device")
                config = {**plan["config"]["training"], "seed": task["seed"]}
                session = TrainSession(data.train.x, data.train.y, data.validation.x,
                                       data.validation.y, config,
                                       checkpoint=payload["session"] if payload else None)
                history = payload["history"] if payload else []
                elapsed = payload["elapsed_seconds"] if payload else 0.0
                job.mkdir(parents=True, exist_ok=True)
                output(f"{task['key']}: CUDA; update {session.step}/{config['max_steps']} (Ctrl+C pauses)")
                last_output = time.monotonic() - 10.0
                while not session.finished and not stopped[0] and (max_chunks is None or chunks < max_chunks):
                    started = time.monotonic()
                    history.extend(session.advance(plan["config"]["execution"]["checkpoint_steps"]))
                    elapsed += time.monotonic() - started
                    _checkpoint(job, session, history, plan, task, elapsed, accelerator)
                    chunks += 1
                    now = time.monotonic()
                    if (now - last_output >= 5.0 or session.finished or stopped[0]
                            or max_chunks is not None and chunks >= max_chunks):
                        output(f"  {task['key']} update {session.step:,}; best validation {session.best_val_mse:.7g} at {session.best_step}")
                        last_output = now
                if session.finished:
                    # A recovered final checkpoint may need only exports after a prior crash.
                    if not (job / "latest.pt").exists():
                        _checkpoint(job, session, history, plan, task, elapsed, accelerator)
                    _evaluate(job, session, history, views, plan, task, elapsed, accelerator)
                    output(f"  complete: {task['key']} ({session.stop_reason})")
                del session
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _report_locked(run, plan)
            summary = _report_locked(run, plan)
            output(format_summary(summary))
            return summary
        finally:
            signal.signal(signal.SIGINT, old_handler)
