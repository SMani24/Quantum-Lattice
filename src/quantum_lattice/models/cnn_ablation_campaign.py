"""Prospective, bounded architecture screening on the existing L4 g cache.

Prior campaigns are immutable. Stateless verified feature/metric/I/O helpers
are reused; this module owns a separate plan, checkpoints, and model registry.
Nothing here opens final-test records or raw-spin payloads.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from . import observable_campaign as _old
from .cnn_ablation_training import CnnAblationTrainSession
from .cnn_variants import VARIANT_NAMES, model_spec
from .observable_training import _config

ARTIFACT_ROOT = Path("codex material/artifacts/cnn_ablation/campaigns")
_canonical, _digest, _read, _sha = _old._canonical, _old._digest, _old._read, _old._sha
_json, _torch, _npz, _csv, _atomic = _old._json, _old._torch, _old._npz, _old._csv, _old._atomic
campaign_lock, _environment, _accelerator = _old.campaign_lock, _old._environment, _old._accelerator
resolve_device = _old.resolve_device
_feature_root, _feature_hashes = _old._feature_root, _old._feature_hashes
_load_maps, _bindings, _reference_for_maps = _old._load_maps, _old._bindings, _old._reference_for_maps
_job, _done, _d5 = _old._job, _old._done, _old._d5


def _safe_run(repo_root, run):
    root, result = (Path(repo_root) / ARTIFACT_ROOT).resolve(), Path(run).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError(f"Ablation campaigns must be below {root}")
    return result


def _sources(repo_root):
    paths = ("src/quantum_lattice/models/cnn_ablation_campaign.py",
             "src/quantum_lattice/models/cnn_ablation_training.py",
             "src/quantum_lattice/models/cnn_variants.py",
             "experiments/cnn_ablation/train.py", "configs/cnn_ablation.json")
    return {**_old._sources(repo_root), **{p: _sha(Path(repo_root) / p) for p in paths}}


def validate_config(config):
    """A deliberately small screen; larger studies require a new protocol."""
    import json
    value = json.loads(_canonical(config))
    required = {"schema_version", "sizes", "seeds", "models", "views", "feature_cache",
                "output_root", "training", "execution", "provenance"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Ablation config has missing or unknown keys")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported ablation schema")
    if value["sizes"] != [4] or value["views"] != ["g"]:
        raise ValueError("This screening protocol requires sizes=[4], views=['g']")
    if value["models"] != list(VARIANT_NAMES):
        raise ValueError("models must contain the four declared variants in frozen order")
    seeds = value["seeds"]
    if (not isinstance(seeds, list) or len(seeds) != 2 or any(type(s) is not int or not 0 <= s < 2**63
            for s in seeds) or len(set(seeds)) != 2):
        raise ValueError("The screen requires two distinct nonnegative seeds below 2**63")
    training = value["training"]
    keys = {"device", "deterministic", "torch_threads", "learning_rate", "batch_size_hamiltonians",
            "max_epochs", "validation_batch_size", "early_stopping"}
    if not isinstance(training, dict) or set(training) != keys:
        raise ValueError("All training settings must be explicit")
    if not isinstance(training["early_stopping"], dict) or set(training["early_stopping"]) != {
            "enabled", "min_epochs", "patience_epochs", "min_delta"}:
        raise ValueError("All early_stopping settings must be explicit")
    normalized = _config({**training, "model": "cnn", "seed": seeds[0],
                          "architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 3}})
    if normalized["device"] != "cuda" or not normalized["deterministic"]:
        raise ValueError("Research screening requires deterministic cuda, with no CPU fallback")
    value["training"] = {key: normalized[key] for key in keys}
    if (not isinstance(value["execution"], dict) or set(value["execution"]) != {"checkpoint_updates"}
            or type(value["execution"]["checkpoint_updates"]) is not int
            or value["execution"]["checkpoint_updates"] < 1):
        raise ValueError("execution requires positive checkpoint_updates")
    for name in ("feature_cache", "output_root"):
        if not isinstance(value[name], str) or not value[name] or Path(value[name]).is_absolute():
            raise ValueError(f"{name} must be repository relative")
    if not isinstance(value["provenance"], dict):
        raise ValueError("provenance must be an object")
    return value


def _tasks(config):
    return [{"L": 4, "seed": seed, "model": model, "view": "g",
             "key": f"L4-seed{seed}-{model}-g"}
            for seed in config["seeds"] for model in config["models"]]


def _sampling_metadata(maps):
    provenance = maps["g"].provenance
    settings = provenance.get("settings", {})
    sampler = copy.deepcopy(settings.get("sampler", {}))
    required = ("total_sweeps", "burn_in_sweeps", "thinning_sweeps")
    retained = ((sampler["total_sweeps"] - sampler["burn_in_sweeps"]) // sampler["thinning_sweeps"]
                if all(k in sampler for k in required) else None)
    return {"sampler": sampler, "retained_configurations_per_hamiltonian": retained,
            "fields": copy.deepcopy(settings.get("fields", {})),
            "train_hamiltonians": len(maps["g"].train), "validation_hamiltonians": len(maps["g"].validation),
            "measurement_scope": provenance.get("measurement_scope"), "test_payload_loaded": False}


def prepare(repo_root, config, *, run_dir=None):
    """Freeze settings and verified non-test inputs; perform zero updates."""
    root, config = Path(repo_root).resolve(), validate_config(config)
    run = _safe_run(root, run_dir or root / config["output_root"])
    run.mkdir(parents=True, exist_ok=True)
    with campaign_lock(run):
        if (run / "plan.json").exists():
            if load_plan(root, run)["config"] != config:
                raise ValueError("Config differs from the frozen screen; use a new output path")
            return run
        if any(p.name not in {".campaign.lock", "plan-sha256.json"} for p in run.iterdir()):
            raise ValueError("Refusing a nonempty uncommitted campaign directory")
        accelerator = _accelerator(resolve_device("cuda", True))
        features = _feature_root(root, config)
        before = _feature_hashes(features, [4])
        maps = _load_maps(features, 4)
        bindings, sampling = _bindings(maps), _sampling_metadata(maps)
        if before != _feature_hashes(features, [4]):
            raise ValueError("Feature files changed during preparation")
        plan = {"schema_version": 1, "workflow": "cnn_ablation", "config": config,
                "tasks": _tasks(config), "model_specs": {m: model_spec(m) for m in VARIANT_NAMES},
                "source_hashes": _sources(root), "environment": _environment(),
                "accelerator": accelerator, "feature_hashes": before, "data_bindings": {"4": bindings},
                "data_provenance": sampling,
                "protocol": {
                    "scope": "architecture screening, L4/g development validation only",
                    "data": "same cached whole Hamiltonians; actual sampler/retained counts recorded in data_provenance",
                    "training": "same Adam, whole-H shuffle, batch count, ceiling and stopping opportunity",
                    "ranking": "mean own-view validation MSE over BOTH paired seeds, only when all eight finish; ties use frozen model order",
                    "selection": "strict-best own-view validation MSE including epoch zero",
                    "secondary": "crossed f/g/exact magnitude errors and reference-sign-assisted D5; not architecture selection",
                    "normalization": "BN running statistics from optimizer minibatches only; evaluation uses frozen selected buffers; GN has global spatial context",
                    "budget": "matched epoch ceiling; runtime, FLOPs, normalization and stopped epochs can differ",
                    "confirmation": "freeze shortlist then separate multi-seed/f-g/size and unopened confirmation protocol",
                    "test_payload_loaded": False}}
        _json(run / "plan-sha256.json", {"sha256": _digest(plan)})
        _json(run / "plan.json", plan)
        _report_locked(run, plan)
    return run


def load_plan(repo_root, run):
    root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    plan = _read(run / "plan.json")
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("workflow") != "cnn_ablation"
            or _read(run / "plan-sha256.json") != {"sha256": _digest(plan)}):
        raise ValueError("Ablation plan hash/schema mismatch")
    config = validate_config(plan["config"])
    if config != plan["config"] or plan["tasks"] != _tasks(config):
        raise ValueError("Frozen configuration/tasks differ")
    if plan["model_specs"] != {m: model_spec(m) for m in VARIANT_NAMES}:
        raise ValueError("Architecture registry changed")
    if plan["source_hashes"] != _sources(root) or plan["environment"] != _environment():
        raise ValueError("Pinned source/environment changed; prepare a new campaign")
    if plan["feature_hashes"] != _feature_hashes(_feature_root(root, config), [4]):
        raise ValueError("Pinned feature cache changed")
    return plan


def _checkpoint(job, session, plan, task, elapsed):
    payload = {"workflow_schema_version": 1, "workflow": "cnn_ablation", "task": task,
               "plan_sha256": _digest(plan), "accelerator": plan["accelerator"],
               "elapsed_seconds": elapsed, "session": session.snapshot()}
    _torch(job / "latest.pt", payload)
    return payload


def _load_checkpoint(job, plan, task):
    if not (job / "latest.pt").exists():
        return None
    payload = torch.load(job / "latest.pt", map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or type(payload.get("workflow_schema_version")) is not int or payload["workflow_schema_version"] != 1
            or payload.get("workflow") != "cnn_ablation" or payload.get("task") != task
            or payload.get("plan_sha256") != _digest(plan) or payload.get("accelerator") != plan["accelerator"]):
        raise ValueError("Checkpoint identity/device mismatch")
    state = payload.get("session", {})
    if not isinstance(state, dict) or not isinstance(state.get("engine"), dict):
        raise ValueError("Checkpoint session/engine must be dictionaries")
    engine = state.get("engine", {})
    if (state.get("training_kind") != "cnn_ablation" or state.get("variant") != task["model"]
            or state.get("model_spec") != plan["model_specs"][task["model"]]
            or type(engine.get("step")) is not int or engine["step"] < 0
            or type(engine.get("epoch")) is not int or engine["epoch"] < 0
            or not isinstance(engine.get("history"), list) or len(engine["history"]) != engine["epoch"] + 1
            or type(payload.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(payload["elapsed_seconds"]) or payload["elapsed_seconds"] < 0):
        raise ValueError("Checkpoint variant/history/counters differ")
    return payload


def _evaluate(job, session, maps, reference, compiled, plan, task, elapsed):
    """Reuse the frozen metric implementation, with explicit variant metadata."""
    predictions, per_h, field_scores, scores = {}, [], [], {}
    rows = _old.maps_to_edge_rows(maps["f"].validation)
    for view in _old.VIEWS:
        mapped = maps[view].validation
        prediction = _old.flatten_edge_maps(mapped, session.selected_predictions(mapped.x))
        if prediction.dtype != np.float32:
            raise ValueError("Predictions must remain raw float32")
        predictions[f"prediction_{view}"] = prediction
        metrics = _old.per_hamiltonian_metrics(prediction, rows.y, rows.hamiltonian_ids, rows.edge_ids)
        per_h.extend({"input_view": view, **r} for r in metrics)
        distances = _d5(reference, prediction, compiled)
        scores[view] = {"mse": float(np.mean([r["mse"] for r in metrics])),
                        "mae": float(np.mean([r["mae"] for r in metrics])),
                        "rmse": math.sqrt(float(np.mean([r["mse"] for r in metrics]))),
                        "mean_hamiltonian_rmse": float(np.mean([r["rmse"] for r in metrics])),
                        "hamiltonians": len(metrics), "rows": len(rows),
                        "negative_prediction_fraction": float(np.mean(prediction < 0)),
                        "above_1_7_prediction_fraction": float(np.mean(prediction > 1.7)),
                        "d5": {k: float(np.mean(v)) for k, v in distances.items()}}
        field_scores.extend({"input_view": view, "hamiltonian_id": str(name), "hamiltonian_uid": str(uid),
                             **{k: float(v[i]) for k, v in distances.items()}}
                            for i, (name, uid) in enumerate(zip(reference.hamiltonian_ids, reference.hamiltonian_uids)))
    baseline = _d5(reference, np.zeros_like(rows.y), compiled)
    late_gain = {}
    for window in (50, 100):
        earlier = session.history[max(0, session.epoch - window)]["best_validation_mse"]
        late_gain[str(window)] = {"window_epochs": min(window, session.epoch),
                                 "best_mse_at_window_start": earlier,
                                 "relative_best_mse_reduction": (earlier - session.best_val_mse) / earlier
                                 if earlier else 0.0}
    _npz(job / "predictions.npz", **predictions, target=rows.y, hamiltonian_ids=rows.hamiltonian_ids,
         hamiltonian_uids=rows.hamiltonian_uids, edge_ids=rows.edge_ids)
    _csv(job / "per-hamiltonian.csv", per_h, ("input_view", "hamiltonian_id", "count", "mse", "rmse", "mae"))
    _csv(job / "hamiltonian-scores.csv", field_scores, ("input_view", "hamiltonian_id", "hamiltonian_uid",
         "two_channel_full", "two_channel_traceless", "z_branch_full", "z_branch_traceless"))
    _json(job / "history.json", session.history)
    evaluation = {"task": task, "model_spec": plan["model_specs"][task["model"]],
                  "scope": "architecture screening validation; not independent confirmation",
                  "best_step": session.best_step, "best_epoch": session.best_epoch,
                  "best_validation_mse": session.best_val_mse, "last_step": session.step,
                  "last_epoch": session.epoch, "stop_reason": session.stop_reason,
                  "elapsed_seconds": elapsed, "accelerator": plan["accelerator"], "scores": scores,
                  "elapsed_scope": "optimizer advances and epoch validation; excludes initial validation, loading, checkpoints and exports",
                  "late_improvement": late_gain,
                  "prediction_policy": "raw unclipped linear magnitudes",
                  "hamiltonian_metric": {"primary": "two_channel_full", "sign_policy": "reference_sign_oracle",
                      "x_fields": "zero", "zero_field_baseline": {k: float(np.mean(v)) for k, v in baseline.items()}},
                  "test_payload_loaded": False}
    _json(job / "evaluation.json", evaluation)
    _torch(job / "best.pt", {"schema_version": 1, "workflow": "cnn_ablation", "plan_sha256": _digest(plan),
                            "task": task, "model_spec": plan["model_specs"][task["model"]],
                            "best_step": session.best_step, "best_epoch": session.best_epoch,
                            "model_state_dict": session.best_state_dict})
    _json(job / "done.json", {"schema_version": 1, "plan_sha256": _digest(plan), "task": task,
                              "hashes": {name: _sha(job / name) for name in _old.OUTPUT_FILES}, "summary": evaluation})


def _summary(run, plan):
    tasks = []
    for task in plan["tasks"]:
        job = _job(run, task)
        done = _done(job, plan, task)
        payload = None if done else _load_checkpoint(job, plan, task)
        row = {**task, "state": "complete" if done else "paused" if payload else "pending",
               "step": 0, "epoch": 0, "best_validation_mse": None}
        if done:
            row.update(copy.deepcopy(done["summary"]))
            row.update(step=done["summary"]["last_step"], epoch=done["summary"]["last_epoch"])
            row.pop("task")
        elif payload:
            state = payload["session"]["engine"]
            row.update(step=state["step"], epoch=state["epoch"], best_epoch=state["best_epoch"],
                       best_validation_mse=state["best_val_mse"], elapsed_seconds=payload["elapsed_seconds"])
        tasks.append(row)
    completed = sum(t["state"] == "complete" for t in tasks)
    ranking = []
    # Do not choose architectures from a cherry-picked seed or unfinished panel.
    if completed == len(tasks):
        for model in VARIANT_NAMES:
            rows = [t for t in tasks if t["model"] == model]
            ranking.append({"model": model, "mean_validation_mse": float(np.mean([r["scores"]["g"]["mse"] for r in rows])),
                            "mean_d5": float(np.mean([r["scores"]["g"]["d5"]["two_channel_full"] for r in rows])),
                            "seeds": len(rows), "capped_models": sum(r["stop_reason"] == "max_epochs" for r in rows)})
        ranking.sort(key=lambda r: r["mean_validation_mse"])
    return {"completed": completed, "total": len(tasks), "tasks": tasks, "ranking": ranking,
            "max_epochs": plan["config"]["training"]["max_epochs"], "test_payload_loaded": False,
            "scope": "L4/g screening validation only"}


def format_summary(summary):
    lines = [f"L=4 / conditioned g: {summary['completed']}/{summary['total']} complete (screening)"]
    for row in summary["tasks"]:
        loss = "-" if row["best_validation_mse"] is None else f"{row['best_validation_mse']:.7g}"
        lines.append(f"  seed {row['seed']}  {row['model']:<10} {row['state']:<8} "
                     f"epoch {row['epoch']:4}/{summary['max_epochs']}  best MSE {loss}")
    if summary["ranking"]:
        lines.append("Two-seed mean validation MSE ranking (development only):")
        lines.extend(f"  {i}. {r['model']:<10} MSE {r['mean_validation_mse']:.7g}; D5 {r['mean_d5']:.7g}; "
                     f"capped {r['capped_models']}/{r['seeds']}" for i, r in enumerate(summary["ranking"], 1))
    else:
        lines.append("Architecture ranking waits for all eight models.")
    return "\n".join(lines)


def _paired_rows(summary):
    rows = []
    for seed in sorted({t["seed"] for t in summary["tasks"]}):
        block = {t["model"]: t for t in summary["tasks"] if t["seed"] == seed and t["state"] == "complete"}
        if set(block) != set(VARIANT_NAMES):
            continue
        for view in _old.VIEWS:
            for model in VARIANT_NAMES:
                score, base = block[model]["scores"][view], block["baseline"]["scores"][view]
                rows.append({"seed": seed, "model": model, "input_view": view, "mse": score["mse"],
                             "mse_minus_baseline": score["mse"] - base["mse"],
                             "d5": score["d5"]["two_channel_full"],
                             "d5_minus_baseline": score["d5"]["two_channel_full"] - base["d5"]["two_channel_full"]})
    return rows


def _report_locked(run, plan):
    summary = _summary(run, plan)
    _json(run / "status.json", summary)
    _csv(run / "paired-validation.csv", _paired_rows(summary),
         ("seed", "model", "input_view", "mse", "mse_minus_baseline", "d5", "d5_minus_baseline"))
    text = ("# CNN architecture screen\n\nL4/g development screening, four variants and two paired seeds. "
            "All arms, including a fresh original baseline, share the longer epoch/stopping policy and the same "
            "Hamiltonian groups. Original campaigns remain intact. No final-test records are loaded.\n\n"
            "```text\n" + format_summary(summary) + "\n```\n\n"
            "Rank by mean own-view validation MSE only after every model finishes. D5 and crossed f/g/exact inputs "
            "are secondary diagnostics. Two seeds and repeatedly used validation groups provide screening evidence, "
            "not a final winner, statistical confirmation or publication claim. BatchNorm uses frozen selected "
            "training statistics during evaluation; GroupNorm mixes spatial positions within each Hamiltonian.\n\n"
            "## Files\n\n- `plan.json`, `plan-sha256.json`: protocol, model specs, tasks, source/environment/GPU/cache hashes and H bindings.\n"
            "- `models/<task>/latest.pt`: atomic live/best model including normalization buffers, Adam, RNG, partial epoch, history.\n"
            "- Finished jobs: `best.pt`, `history.json`, `evaluation.json`, `predictions.npz`, `per-hamiltonian.csv`, "
            "`hamiltonian-scores.csv`, `done.json` (all seven output hashes).\n"
            "- `status.json`, `paired-validation.csv`: derived progress and complete four-arm paired-seed diagnostics.\n"
            "- `execution.log`: terminal log when launched through the runner.\n\n"
            "Ctrl+C requests a checkpoint pause. Resume on the pinned source, cache, environment and GPU. "
            "No Monte Carlo generation, raw-snapshot training or unopened confirmation is part of this screen.\n")
    _atomic(run / "RESULTS.md", lambda handle: handle.write(text))
    return summary


def summarize_campaign(repo_root, run):
    run = _safe_run(repo_root, run)
    with campaign_lock(run):
        return _report_locked(run, load_plan(repo_root, run))


def run_campaign(repo_root, run, *, models=None, max_chunks=None, output=print):
    """Run sequentially on CUDA; SIGINT drains a bounded chunk and saves."""
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
        raise ValueError("max_chunks must be positive")
    root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    accelerator = _accelerator(resolve_device("cuda", True))
    with campaign_lock(run):
        plan = load_plan(root, run)
        if accelerator != plan["accelerator"]:
            raise ValueError("Accelerator differs from the frozen campaign")
        selected = list(VARIANT_NAMES) if models is None else list(models)
        if not selected or len(set(selected)) != len(selected) or any(m not in VARIANT_NAMES for m in selected):
            raise ValueError("models must be a distinct nonempty subset of the frozen variants")
        stopped, chunks = [False], 0
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda signum, frame: stopped.__setitem__(0, True))
        try:
            config, maps = plan["config"], None
            for task in plan["tasks"]:
                job = _job(run, task)
                if task["model"] not in selected or _done(job, plan, task):
                    continue
                if stopped[0] or max_chunks is not None and chunks >= max_chunks:
                    break
                if maps is None:
                    features = _feature_root(root, config)
                    if _feature_hashes(features, [4]) != plan["feature_hashes"]:
                        raise ValueError("Feature cache changed before loading")
                    maps = _load_maps(features, 4)
                    if _bindings(maps) != plan["data_bindings"]["4"]:
                        raise ValueError("Hamiltonian bindings changed")
                    if _sampling_metadata(maps) != plan["data_provenance"]:
                        raise ValueError("Sampling provenance changed")
                    reference = _reference_for_maps(features, 4, maps)
                    if _feature_hashes(features, [4]) != plan["feature_hashes"]:
                        raise ValueError("Feature cache changed during loading")
                    compiled = {m: _old.CompiledHamiltonianMetric(_old.square_lattice(4), model=m)
                                for m in ("two_channel", "z_branch")}
                data = maps["g"]
                if data.provenance.get("training_eligible") is not True:
                    raise ValueError("Evaluation-only view cannot train")
                payload = _load_checkpoint(job, plan, task)
                settings = {**config["training"], "seed": task["seed"], "model": "cnn",
                            "architecture": {"hidden_channels": [32, 64, 32], "kernel_size": 3}}
                session = CnnAblationTrainSession(data.train.x, data.train.y, data.validation.x, data.validation.y,
                    settings, variant=task["model"], checkpoint=payload["session"] if payload else None)
                elapsed = payload["elapsed_seconds"] if payload else 0.0
                job.mkdir(parents=True, exist_ok=True)
                output(f"{task['key']}: CUDA, epoch {session.epoch}/{settings['max_epochs']}; Ctrl+C saves and pauses")
                last_output = time.monotonic() - 10
                while not session.finished and not stopped[0] and (max_chunks is None or chunks < max_chunks):
                    started = time.monotonic()
                    session.advance(config["execution"]["checkpoint_updates"])
                    elapsed += time.monotonic() - started
                    _checkpoint(job, session, plan, task, elapsed)
                    chunks += 1
                    now = time.monotonic()
                    if now - last_output >= 5 or session.finished or stopped[0] or max_chunks is not None and chunks >= max_chunks:
                        output(f"  epoch {session.epoch}/{settings['max_epochs']}; update {session.step:,}; "
                               f"best MSE {session.best_val_mse:.7g} at epoch {session.best_epoch}")
                        last_output = now
                if session.finished:
                    _checkpoint(job, session, plan, task, elapsed)
                    _evaluate(job, session, maps, reference, compiled, plan, task, elapsed)
                    output(f"  complete: {task['key']} ({session.stop_reason})")
                elif not (job / "latest.pt").exists():
                    _checkpoint(job, session, plan, task, elapsed)
                del session
                if accelerator["type"] == "cuda":
                    torch.cuda.empty_cache()
                _report_locked(run, plan)
            summary = _report_locked(run, plan)
            output(format_summary(summary))
            return summary
        finally:
            signal.signal(signal.SIGINT, old_handler)
