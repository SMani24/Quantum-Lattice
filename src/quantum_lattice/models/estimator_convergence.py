"""A separately pinned continuation of every model in a completed f/g study.

Parent source files stay unchanged so their historical pins remain valid. Only
the declared training ceiling and stopping policy change, in a new output tree.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import signal
import time

import torch

from . import estimator_mlp as base
from .mlp_training import TrainSession, _config, resolve_device
from quantum_lattice.data.estimator_features import load_feature_views

ARTIFACT_ROOT = Path("codex material/artifacts/estimator_convergence")


def _safe_run(repo_root, run):
    root = (Path(repo_root) / ARTIFACT_ROOT).resolve()
    run = Path(run).resolve()
    if run == root or not run.is_relative_to(root):
        raise ValueError(f"Continuation must be below {root}")
    return run


def _sources(repo_root):
    paths = ("src/quantum_lattice/models/estimator_convergence.py",
             "experiments/estimator_convergence/run.py")
    return {**base._sources(repo_root), **{name: base._sha(Path(repo_root) / name) for name in paths}}


def validate_config(config):
    value = copy.deepcopy(config)
    required = {"schema_version", "parent_run", "output_root", "training", "execution"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != 1:
        raise ValueError("Unsupported convergence configuration/schema")
    for field in ("parent_run", "output_root"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{field} must be a path")
    training = value["training"]
    if not isinstance(training, dict) or set(training) != {"max_steps", "early_stopping"}:
        raise ValueError("Only max_steps and early_stopping may change")
    stopping = training["early_stopping"]
    if not isinstance(stopping, dict) or set(stopping) != {"enabled", "patience", "min_delta", "min_steps"}:
        raise ValueError("Declare all early-stopping settings explicitly")
    normalized = _config({**training, "device": "cuda"})
    if not normalized["early_stopping"]["enabled"]:
        raise ValueError("This continuation requires the declared early-stopping policy")
    if normalized["early_stopping"]["min_steps"] >= normalized["max_steps"]:
        raise ValueError("Stopping grace must end before the update ceiling")
    value["training"] = {key: normalized[key] for key in ("max_steps", "early_stopping")}
    execution = value["execution"]
    if (not isinstance(execution, dict) or set(execution) != {"checkpoint_steps"}
            or isinstance(execution["checkpoint_steps"], bool)
            or not isinstance(execution["checkpoint_steps"], int) or execution["checkpoint_steps"] < 1):
        raise ValueError("checkpoint_steps must be positive")
    return value


def _parent_path(repo_root, config):
    path = Path(config["parent_run"])
    return base._safe_run(repo_root, path if path.is_absolute() else Path(repo_root) / path)


def _fork_snapshot(snapshot, config):
    """Explicit policy fork; retain all trajectory state and inherited best weights."""
    old, new = _config(snapshot["config"]), _config(config)
    scientific = set(old) - {"max_steps", "early_stopping"}
    if any(old[key] != new[key] for key in scientific):
        raise ValueError("Fork may change only max_steps and early_stopping")
    if (snapshot.get("training_schema_version") != 2 or new["max_steps"] <= old["max_steps"]
            or new["early_stopping"]["min_steps"] <= snapshot["step"]
            or new["early_stopping"]["min_steps"] >= new["max_steps"]):
        raise ValueError("Fork requires a later ceiling and a future stopping grace")
    if not new["early_stopping"]["enabled"] or not math.isfinite(snapshot["best_val_mse"]):
        raise ValueError("Fork needs a finite inherited best and enabled stopping")
    result = copy.deepcopy(snapshot)
    result.update(config=new, stopped_early=False, early_stopping_wait=0,
                  early_stopping_reference=float(snapshot["best_val_mse"]))
    return result


def _capture_parent(repo_root, parent, config):
    plan = base.load_plan(repo_root, parent)
    training = {**plan["config"]["training"], **config["training"]}
    if training["max_steps"] <= plan["config"]["training"]["max_steps"]:
        raise ValueError("Continuation ceiling must exceed the parent ceiling")
    if training["device"] != "cuda" or config["execution"]["checkpoint_steps"] != plan["config"]["execution"]["checkpoint_steps"]:
        raise ValueError("CUDA and the parent's checkpoint interval must be preserved")
    names = ["plan.json", "plan-sha256.json"]
    names.extend(str(p.relative_to(parent)) for p in sorted((parent / "features").rglob("*")) if p.is_file())
    tasks = {}
    for task in plan["tasks"]:
        job = parent / "models" / task["key"]
        done = base._done(job, plan, task)
        if not done:
            raise ValueError("Every parent model must be complete before continuation")
        payload = base._load_checkpoint(job, plan, task)
        if payload is None:
            raise ValueError("Completed parent lacks latest checkpoint")
        state = payload["session"]
        expected = _config({**plan["config"]["training"], "seed": task["seed"]})
        if state["config"] != expected:
            raise ValueError("Parent checkpoint settings differ from its plan")
        if payload["accelerator"].get("type") != state["device_type"]:
            raise ValueError("Parent accelerator and checkpoint device disagree")
        if (state["step"] != done["summary"]["last_step"]
                or state["best_step"] != done["summary"]["best_step"]
                or payload["history"] != base._read(job / "history.json")):
            raise ValueError("Parent checkpoint history/counters differ from exports")
        _fork_snapshot(state, {**training, "seed": task["seed"]})
        tasks[task["key"]] = {
            "step": state["step"], "best_step": state["best_step"],
            "best_validation_mse": state["best_val_mse"],
            "last_validation": state["last_validation"],
            "stop_reason": done["summary"]["stop_reason"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "history_events": len(payload["history"]),
            "history_digest": base._digest(payload["history"]),
            "data_fingerprint": state["data_fingerprint"],
            "accelerator": payload["accelerator"],
            "scores": done["summary"]["scores"],
        }
        names.extend(f"models/{task['key']}/{name}" for name in (*done["hashes"], "done.json"))
    return plan, {"run": str(parent.relative_to(repo_root)), "plan_sha256": base._digest(plan),
                  "files": {name: base._sha(parent / name) for name in sorted(names)}, "tasks": tasks}


def prepare(repo_root, config, *, run_dir=None):
    """Freeze the complete parent and new policy without any optimizer updates."""
    repo_root = Path(repo_root).resolve()
    config = validate_config(config)
    run = _safe_run(repo_root, run_dir or repo_root / config["output_root"])
    parent = _parent_path(repo_root, config)
    if parent == run or parent.is_relative_to(run) or run.is_relative_to(parent):
        raise ValueError("Parent and continuation directories must be separate")
    run.mkdir(parents=True, exist_ok=True)
    with base.campaign_lock(run):
        if (run / "plan.json").exists():
            plan = load_plan(repo_root, run)
            if plan["extension_config"] != config:
                raise ValueError("Configuration differs from the frozen continuation")
            return run
        parent_plan, binding = _capture_parent(repo_root, parent, config)
        merged = copy.deepcopy(parent_plan["config"])
        merged["training"].update(config["training"])
        merged["execution"] = config["execution"]
        merged["output_root"] = str(run.relative_to(repo_root))
        plan = {"schema_version": 1, "workflow": "estimator_convergence",
                "extension_config": config, "config": merged, "tasks": parent_plan["tasks"],
                "parent": binding, "source_hashes": _sources(repo_root),
                "environment": base._environment(), "protocol": {
                    "scope": "development validation only; no final-test access",
                    "fork": "every parent latest model/Adam/RNG; no rollback to best weights",
                    "selection": "strict minimum own-view validation MSE over parent plus extension",
                    "stopping": "reset flag/wait for every model; reference inherited best; identical new policy",
                    "history": "literal parent prefix retained, including historical stop reasons",
                    "exact_view": "evaluation only; never training or selection",
                    "comparison": "all original sizes, seeds and views retained; unequal realized stopping lengths reported",
                    "convergence": "last-50000-update best-loss relative improvement is descriptive, not a proof"}}
        base._json(run / "plan-sha256.json", {"sha256": base._digest(plan)})
        base._json(run / "plan.json", plan)
        _report_locked(run, plan)
    return run


def load_plan(repo_root, run):
    repo_root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    plan = base._read(run / "plan.json")
    if (plan.get("schema_version") != 1 or plan.get("workflow") != "estimator_convergence"
            or base._read(run / "plan-sha256.json") != {"sha256": base._digest(plan)}):
        raise ValueError("Frozen continuation plan hash/schema mismatch")
    config = validate_config(plan["extension_config"])
    if config != plan["extension_config"] or plan["source_hashes"] != _sources(repo_root):
        raise ValueError("Continuation settings or source changed")
    if plan["environment"] != base._environment():
        raise ValueError("Environment changed; exact continuation is unavailable")
    parent = _parent_path(repo_root, config)
    original = base.load_plan(repo_root, parent)
    if (plan["parent"]["run"] != str(parent.relative_to(repo_root))
            or plan["parent"]["plan_sha256"] != base._digest(original)
            or plan["tasks"] != original["tasks"]):
        raise ValueError("Frozen parent/task identity changed")
    expected = copy.deepcopy(original["config"])
    expected["training"].update(config["training"])
    expected["execution"] = config["execution"]
    expected["output_root"] = str(run.relative_to(repo_root))
    if expected != plan["config"]:
        raise ValueError("Inherited scientific settings changed")
    for name, checksum in plan["parent"]["files"].items():
        path = (parent / name).resolve()
        if not path.is_relative_to(parent) or base._sha(path) != checksum:
            raise ValueError(f"Frozen parent file changed: {name}")
    return plan


def _fork_record(plan, task):
    origin = plan["parent"]["tasks"][task["key"]]
    return {"schema_version": 1, "plan_sha256": base._digest(plan), "task": task,
            "parent_run": plan["parent"]["run"],
            "parent_latest_sha256": plan["parent"]["files"][f"models/{task['key']}/latest.pt"],
            "parent_step": origin["step"], "parent_best_step": origin["best_step"],
            "parent_stop_reason": origin["stop_reason"],
            "parent_elapsed_seconds": origin["elapsed_seconds"],
            "reset": {"stopped_early": False, "early_stopping_wait": 0,
                      "early_stopping_reference": origin["best_validation_mse"]},
            "inherited_best_retained": True, "history_prefix_preserved": True}


def _job_state(run, plan, task):
    job = run / "models" / task["key"]
    done = base._done(job, plan, task)
    payload = base._load_checkpoint(job, plan, task)
    if (done or payload) and base._read(job / "fork.json") != _fork_record(plan, task):
        raise ValueError("Continuation fork identity changed")
    if payload:
        origin = plan["parent"]["tasks"][task["key"]]
        expected = _config({**plan["config"]["training"], "seed": task["seed"]})
        if (payload["session"]["config"] != expected
                or payload["session"]["data_fingerprint"] != origin["data_fingerprint"]
                or payload["session"]["step"] < origin["step"]
                or payload["accelerator"] != origin["accelerator"]):
            raise ValueError("Continuation checkpoint settings/data/progress changed")
        count = origin["history_events"]
        if (len(payload["history"]) < count
                or base._digest(payload["history"][:count]) != origin["history_digest"]
                or payload["session"]["last_validation"] != payload["history"][-1]):
            raise ValueError("Continuation history or literal parent prefix changed")
    return done, payload


def _summary(run, plan):
    for task in plan["tasks"]:
        _job_state(run, plan, task)
    summary = base._summary(run, plan)
    for item in summary["tasks"]:
        origin = plan["parent"]["tasks"][item["key"]]
        item["parent_step"] = origin["step"]
        if item["state"] == "pending":
            item.update(step=origin["step"], best_step=origin["best_step"],
                        best_validation_mse=origin["best_validation_mse"])
        item["additional_updates"] = item["step"] - origin["step"]
    return summary


def format_summary(summary):
    return base.format_summary(summary).replace("Estimator MLP:", "MLP convergence:") + (
        "\nUpdate counts are cumulative; pending models show their parent starting step."
        "\nSame ceiling and stopping policy; actual stop lengths may differ. Validation only.")


def _report_locked(run, plan):
    summary = _summary(run, plan)
    base._json(run / "status.json", summary)
    comparisons, extended, convergence = [], [], []
    complete = {(t["L"], t["seed"], t["view"]): t for t in summary["tasks"] if t["state"] == "complete"}
    for task in plan["tasks"]:
        key = (task["L"], task["seed"], task["view"])
        if key not in complete:
            continue
        item, origin = complete[key], plan["parent"]["tasks"][task["key"]]
        for view in base.VIEWS:
            a, b = origin["scores"][view]["mse"], item["scores"][view]["mse"]
            extended.append({**task, "validation_input": view, "parent_mse": a, "extended_mse": b,
                             "difference": b-a, "relative_reduction": 1-b/a if a else None})
        history = base._read(run / "models" / task["key"] / "history.json")
        cutoff = item["step"] - 50000
        earlier = [event["best_validation_mse"] for event in history if event["step"] <= cutoff]
        before = earlier[-1] if earlier else None
        convergence.append({**task, "last_step": item["step"], "additional_updates": item["additional_updates"],
                            "best_step": item["best_step"], "stop_reason": item["stop_reason"],
                            "best_mse_50000_updates_earlier": before,
                            "best_mse": item["best_validation_mse"],
                            "relative_best_improvement_last_50000_updates":
                                1-item["best_validation_mse"]/before if before else None})
    for L in plan["config"]["sizes"]:
        for seed in plan["config"]["seeds"]:
            if (L, seed, "f") not in complete or (L, seed, "g") not in complete:
                continue
            f, g = complete[L, seed, "f"], complete[L, seed, "g"]
            for view in (*base.VIEWS, "matched_pipeline"):
                a = f["scores"]["f" if view == "matched_pipeline" else view]["mse"]
                b = g["scores"]["g" if view == "matched_pipeline" else view]["mse"]
                comparisons.append({"L": L, "seed": seed, "validation_input": view, "f_mse": a,
                                    "g_mse": b, "g_minus_f_mse": b-a,
                                    "relative_mse_reduction": 1-b/a if a else None})
    for name, rows, fields in (
            ("paired-validation.csv", comparisons, ("L","seed","validation_input","f_mse","g_mse","g_minus_f_mse","relative_mse_reduction")),
            ("parent-comparison.csv", extended, ("key","L","seed","view","validation_input","parent_mse","extended_mse","difference","relative_reduction")),
            ("convergence.csv", convergence, ("key","L","seed","view","last_step","additional_updates","best_step","stop_reason","best_mse_50000_updates_earlier","best_mse","relative_best_improvement_last_50000_updates"))):
        base._csv(run / name, rows, fields)
    text = ("# Matched MLP convergence campaign\n\n"
            f"State: **{summary['state']}**; **{summary['completed']}/{summary['total']}** models complete.\n\n"
            f"Every parent model continues from its latest Adam/RNG checkpoint up to **{plan['config']['training']['max_steps']:,} total updates**. "
            "All parent files remain intact. The inherited best own-view validation weights remain eligible. "
            "Stopping flags/patience reset once for every model under the new frozen policy; subsequent pauses restore them exactly. "
            "Parent history remains a literal prefix, including historical stop flags. Exact inputs are evaluation-only.\n\n"
            "This is a development convergence diagnostic, not a final held-out test, proof of global optimization, "
            "EqD5 comparison or measurement protocol. All original seeds and both views are retained. "
            "Matched ceilings and stopping policy do not mean identical realized training lengths.\n\n"
            "## Files\n\n"
            "- `plan.json`, `plan-sha256.json`: effective settings, parent output pins, sources and environment.\n"
            "- `models/*/fork.json`: immutable branch boundary and stopping-policy reset.\n"
            "- `models/*/latest.pt`: full optimizer/RNG, inherited best and history for exact continuation.\n"
            "- Completed jobs add `best.pt`, `history.json`, `evaluation.json`, `predictions.npz`, "
            "`per-hamiltonian.csv`, and hash-verified `done.json`.\n"
            "- `paired-validation.csv`: all seed-wise f/g crossed contrasts.\n"
            "- `parent-comparison.csv`: each extended model against its own frozen original, all input views.\n"
            "- `convergence.csv`: final stops and last-50,000-update best-loss improvement; descriptive only.\n"
            "- `status.json`, `execution.log`: progress and terminal record. Features are read from the parent cache, not duplicated.\n\n"
            "## Progress\n\n```text\n" + format_summary(summary) + "\n```\n")
    base._atomic(run / "RESULTS.md", lambda handle: handle.write(text))
    return summary


def summarize_campaign(repo_root, run):
    run = _safe_run(repo_root, run)
    with base.campaign_lock(run):
        return _report_locked(run, load_plan(repo_root, run))


def run_campaign(repo_root, run, *, max_chunks=None, output=print):
    if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
        raise ValueError("max_chunks must be positive")
    repo_root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    device = resolve_device("cuda", True)
    accelerator = base._accelerator(device)
    with base.campaign_lock(run):
        plan = load_plan(repo_root, run)
        stopped, chunks, current_L, views = [False], 0, None, None
        old_handler = signal.getsignal(signal.SIGINT)
        def request_pause(signum, frame):
            stopped[0] = True
        signal.signal(signal.SIGINT, request_pause)
        try:
            for task in plan["tasks"]:
                done, payload = _job_state(run, plan, task)
                if done:
                    continue
                if stopped[0] or max_chunks is not None and chunks >= max_chunks:
                    break
                origin = plan["parent"]["tasks"][task["key"]]
                if accelerator != origin["accelerator"]:
                    raise ValueError("Accelerator changed; continuation requires the parent's original device")
                if current_L != task["L"]:
                    feature_root = repo_root / plan["parent"]["run"] / "features"
                    views = base._check_views(load_feature_views(feature_root, task["L"]), task["L"])
                    current_L = task["L"]
                config = {**plan["config"]["training"], "seed": task["seed"]}
                if payload:
                    snapshot, history, elapsed = payload["session"], payload["history"], payload["elapsed_seconds"]
                else:
                    parent_file = repo_root / plan["parent"]["run"] / "models" / task["key"] / "latest.pt"
                    original = torch.load(parent_file, map_location="cpu", weights_only=True)
                    snapshot = _fork_snapshot(original["session"], config)
                    history, elapsed = copy.deepcopy(original["history"]), 0.0
                data = views[task["view"]]
                session = TrainSession(data.train.x, data.train.y, data.validation.x, data.validation.y,
                                       config, checkpoint=snapshot)
                job = run / "models" / task["key"]
                job.mkdir(parents=True, exist_ok=True)
                if not payload:
                    record = _fork_record(plan, task)
                    if (job / "fork.json").exists() and base._read(job / "fork.json") != record:
                        raise ValueError("Continuation fork identity changed")
                    base._json(job / "fork.json", record)
                    base._checkpoint(job, session, history, plan, task, elapsed, accelerator)
                output(f"{task['key']}: CUDA; cumulative update {session.step:,}/{config['max_steps']:,} (Ctrl+C pauses)")
                last_output = time.monotonic() - 10
                while not session.finished and not stopped[0] and (max_chunks is None or chunks < max_chunks):
                    started = time.monotonic()
                    history.extend(session.advance(plan["config"]["execution"]["checkpoint_steps"]))
                    elapsed += time.monotonic() - started
                    base._checkpoint(job, session, history, plan, task, elapsed, accelerator)
                    chunks += 1
                    now = time.monotonic()
                    if now-last_output >= 5 or session.finished or stopped[0] or max_chunks is not None and chunks >= max_chunks:
                        output(f"  {task['key']} update {session.step:,} (+{session.step-origin['step']:,}); best validation {session.best_val_mse:.7g} at {session.best_step}")
                        last_output = now
                if session.finished:
                    base._evaluate(job, session, history, views, plan, task, elapsed, accelerator)
                    output(f"  complete: {task['key']} ({session.stop_reason})")
                del session
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _report_locked(run, plan)
            result = _report_locked(run, plan)
            output(format_summary(result))
            return result
        finally:
            signal.signal(signal.SIGINT, old_handler)
