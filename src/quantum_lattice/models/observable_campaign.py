"""Frozen four-arm observable-map learning campaigns.

Only the existing non-test feature cache is read. Whole-Hamiltonian batches,
paired seeds and epoch budgets are shared by the CNN and the map-wrapped MLP.
The campaign owns durable I/O and scheduling; the training engine owns Adam,
model selection and exact-resume state. Preparing a campaign makes no updates.
"""
from __future__ import annotations

import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from quantum_lattice.data.estimator_features import load_feature_views
from quantum_lattice.data.hamiltonian_scores import load_validation_fields, reference_sign_lift
from quantum_lattice.data.observable_maps import flatten_edge_maps, map_learning_data, maps_to_edge_rows
from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian_batch import CompiledHamiltonianMetric
from quantum_lattice.metrics.regression import per_hamiltonian_metrics
from quantum_lattice.models import estimator_mlp as _io
from quantum_lattice.models.mlp_training import resolve_device
from quantum_lattice.models.observable_training import ObservableTrainSession, _config


ARTIFACT_ROOT = Path("codex material/artifacts/observable_cnn/campaigns")
VIEWS = ("f", "g", "exact")
MODELS = ("cnn", "map_mlp")
OUTPUT_FILES = frozenset(("latest.pt", "best.pt", "history.json", "evaluation.json",
                          "per-hamiltonian.csv", "hamiltonian-scores.csv", "predictions.npz"))
_canonical, _digest, _read, _sha = _io._canonical, _io._digest, _io._read, _io._sha
_json, _torch, _npz, _csv, _atomic = _io._json, _io._torch, _io._npz, _io._csv, _io._atomic
campaign_lock, _environment, _accelerator = _io.campaign_lock, _io._environment, _io._accelerator


def _safe_run(repo_root, run):
    root = (Path(repo_root) / ARTIFACT_ROOT).resolve()
    result = Path(run).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError(f"Observable campaigns must be below {root}")
    return result


def _feature_root(repo_root, config):
    path = (Path(repo_root) / config["feature_cache"]).resolve()
    allowed = (Path(repo_root) / "codex material/artifacts").resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError("Feature cache must be inside the ignored artifact directory")
    return path


def _sources(repo_root):
    paths = ("src/quantum_lattice/models/observable_campaign.py",
             "src/quantum_lattice/models/observable_training.py",
             "src/quantum_lattice/models/observable_terminal.py",
             "src/quantum_lattice/models/observable_cnn.py",
             "src/quantum_lattice/data/observable_maps.py",
             "src/quantum_lattice/data/hamiltonian_scores.py",
             "src/quantum_lattice/metrics/hamiltonian.py",
             "src/quantum_lattice/metrics/hamiltonian_batch.py",
             "experiments/observable_cnn/train.py", "configs/observable_cnn_training.json")
    return {**_io._sources(repo_root), **{name: _sha(Path(repo_root) / name) for name in paths}}


def validate_config(config):
    """Normalize explicit settings; public research campaigns require CUDA."""
    import json
    value = json.loads(_canonical(config))
    required = {"schema_version", "sizes", "seeds", "models", "views", "feature_cache",
                "output_root", "architecture", "training", "execution", "provenance"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Observable campaign settings have missing or unknown keys")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported observable campaign schema")
    if value["sizes"] != [4, 3]:
        raise ValueError("This development campaign requires sizes=[4,3], L4 first")
    for name, expected in (("models", set(MODELS)), ("views", {"f", "g"})):
        if (not isinstance(value[name], list) or len(value[name]) != len(expected)
                or set(value[name]) != expected):
            raise ValueError(f"{name} must contain each declared arm exactly once")
    seeds = value["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int
            or not 0 <= seed < 2**63 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct integers in [0,2**63)")
    training = value["training"]
    training_keys = {"device", "deterministic", "torch_threads", "learning_rate",
                     "batch_size_hamiltonians", "max_epochs", "validation_batch_size", "early_stopping"}
    if not isinstance(training, dict) or set(training) != training_keys:
        raise ValueError("training settings have missing or unknown keys")
    normalized = _config({**training, "model": value["models"][0],
                          "architecture": value["architecture"], "seed": seeds[0]})
    if normalized["device"] != "cuda":
        raise ValueError("This research campaign requires device=cuda; no CPU fallback")
    if set(training["early_stopping"]) != {"enabled", "min_epochs", "patience_epochs", "min_delta"}:
        raise ValueError("early_stopping settings must all be explicit")
    value["architecture"] = normalized.pop("architecture")
    normalized.pop("model")
    normalized.pop("seed")
    value["training"] = normalized
    execution = value["execution"]
    if (not isinstance(execution, dict) or set(execution) != {"checkpoint_updates"}
            or type(execution["checkpoint_updates"]) is not int or execution["checkpoint_updates"] < 1):
        raise ValueError("execution requires positive checkpoint_updates")
    for name in ("feature_cache", "output_root"):
        if not isinstance(value[name], str) or not value[name] or Path(value[name]).is_absolute():
            raise ValueError(f"{name} must be a nonempty repository-relative path")
    if not isinstance(value["provenance"], dict):
        raise ValueError("provenance must be an object")
    return value


def _tasks(config):
    return [{"L": L, "seed": seed, "model": model, "view": view,
             "key": f"L{L}-seed{seed}-{model}-{view}"}
            for L in config["sizes"] for seed in config["seeds"]
            for model in config["models"] for view in config["views"]]


def _feature_hashes(feature_root, sizes):
    paths = ["manifest.json", "request.json"]
    paths += [f"L{L}/{name}" for L in sizes for name in ("manifest.json", "views.npz")]
    hashes = {}
    for name in paths:
        path = feature_root / name
        if not path.resolve().is_relative_to(feature_root) or not path.is_file():
            raise ValueError(f"Missing or unsafe feature file: {name}")
        hashes[name] = _sha(path)
    return hashes


def _load_maps(feature_root, L):
    views = load_feature_views(feature_root, L)
    if set(views) != set(VIEWS):
        raise ValueError("All f/g/exact feature views are required")
    maps = {}
    for view in VIEWS:
        data = views[view]
        if data.provenance.get("training_eligible") is not (view != "exact"):
            raise ValueError("Exact inputs are evaluation-only; f/g eligibility must be explicit")
        maps[view] = map_learning_data(data)
    base = maps["f"]
    for view in VIEWS:
        if maps[view].L != L:
            raise ValueError("Cached lattice size differs")
        for split in ("train", "validation"):
            left, right = getattr(base, split), getattr(maps[view], split)
            for name in ("y", "hamiltonian_ids", "hamiltonian_uids", "source_row_indices"):
                if not np.array_equal(getattr(left, name), getattr(right, name)):
                    raise ValueError(f"Paired {split} map identities/order/labels differ in {view}")
    return maps


def _bindings(maps):
    result = {}
    for split in ("train", "validation"):
        data = getattr(maps["f"], split)
        result[split] = {"hamiltonians": len(data), "input_shape": list(data.x.shape),
                         "target_shape": list(data.y.shape),
                         "identity_sha256": _digest([data.hamiltonian_ids.tolist(),
                                                      data.hamiltonian_uids.tolist()])}
    return result


def prepare(repo_root, config, *, run_dir=None):
    """Freeze verified existing features, source, device and tasks; zero updates."""
    repo_root, config = Path(repo_root).resolve(), validate_config(config)
    run = _safe_run(repo_root, run_dir or repo_root / config["output_root"])
    features = _feature_root(repo_root, config)
    run.mkdir(parents=True, exist_ok=True)
    with campaign_lock(run):
        if (run / "plan.json").exists():
            plan = load_plan(repo_root, run)
            if plan["config"] != config:
                raise ValueError("Configuration differs from the frozen campaign")
            return run
        if any(path.name not in {".campaign.lock", "plan-sha256.json"} for path in run.iterdir()):
            raise ValueError("Refusing to prepare into a nonempty uncommitted campaign")
        device = resolve_device("cuda", config["training"]["deterministic"])
        accelerator = _accelerator(device)
        before = _feature_hashes(features, config["sizes"])
        bindings = {str(L): _bindings(_load_maps(features, L)) for L in config["sizes"]}
        if before != _feature_hashes(features, config["sizes"]):
            raise ValueError("Feature files changed during campaign preparation")
        plan = {"schema_version": 1, "config": config, "tasks": _tasks(config),
                "source_hashes": _sources(repo_root), "environment": _environment(),
                "accelerator": accelerator, "feature_hashes": before, "data_bindings": bindings,
                "protocol": {
                    "scope": "development train/validation only; no final-test payloads",
                    "input_channels": ["mean_A", "mean_Au_Av_horizontal", "mean_Au_Av_vertical"],
                    "targets": "two edge-orientation maps of abs(b_z); fields are never network inputs",
                    "batches": "identical whole-Hamiltonian shuffle order for every arm at each seed",
                    "initialization": "identical f/g weights within an architecture; architectures differ",
                    "selection": "strictly lowest own-view validation MSE including epoch zero",
                    "stopping": "grace through min_epochs; then patience on absolute min_delta improvements",
                    "exact_view": "evaluation only; never training or model selection",
                    "crossed_evaluation": "selected weights on f/g/exact validation inputs",
                    "hamiltonian_metric": "raw unsquared D5; reference_sign_oracle; zero X; two_channel/full primary",
                    "comparison": "fresh four-arm controls; older edge-batched MLPs are contextual only",
                    "budget": "same epoch ceiling and stopping opportunity, not equal FLOPs or runtime",
                    "test_payload_loaded": False}}
        _json(run / "plan-sha256.json", {"sha256": _digest(plan)})
        _json(run / "plan.json", plan)
        _report_locked(run, plan)
    return run


def load_plan(repo_root, run):
    """Verify frozen source/environment/cache without requiring CUDA for status."""
    repo_root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    plan = _read(run / "plan.json")
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or _read(run / "plan-sha256.json") != {"sha256": _digest(plan)}):
        raise ValueError("Frozen campaign plan hash/schema mismatch")
    config = validate_config(plan["config"])
    if config != plan["config"] or plan["tasks"] != _tasks(config):
        raise ValueError("Frozen configuration/task list differs")
    if plan["source_hashes"] != _sources(repo_root):
        raise ValueError("Scientific source changed; prepare a new campaign")
    if plan["environment"] != _environment():
        raise ValueError("Scientific environment changed; exact continuation is unavailable")
    if plan["feature_hashes"] != _feature_hashes(_feature_root(repo_root, config), config["sizes"]):
        raise ValueError("Frozen feature cache checksum changed")
    return plan


def _job(run, task):
    return Path(run) / "models" / task["key"]


def _done(job, plan, task):
    path = job / "done.json"
    if not path.exists():
        return None
    done = _read(path)
    if (type(done.get("schema_version")) is not int or done["schema_version"] != 1
            or done.get("task") != task or done.get("plan_sha256") != _digest(plan)
            or set(done.get("hashes", {})) != OUTPUT_FILES):
        raise ValueError(f"Completed model identity/inventory mismatch: {task['key']}")
    for name, checksum in done["hashes"].items():
        target = job / name
        if (not target.resolve().is_relative_to(job.resolve()) or not target.is_file()
                or _sha(target) != checksum):
            raise ValueError(f"Completed model output changed: {task['key']}/{name}")
    evaluation = _read(job / "evaluation.json")
    if done.get("summary") != evaluation or evaluation.get("task") != task:
        raise ValueError("Completed marker differs from verified evaluation")
    return done


def _checkpoint(job, session, plan, task, elapsed, accelerator):
    payload = {"workflow_schema_version": 1, "workflow": "observable_maps",
               "plan_sha256": _digest(plan), "task": task, "session": session.snapshot(),
               "elapsed_seconds": elapsed, "accelerator": accelerator}
    _torch(job / "latest.pt", payload)
    return payload


def _load_checkpoint(job, plan, task):
    if not (job / "latest.pt").exists():
        return None
    payload = torch.load(job / "latest.pt", map_location="cpu", weights_only=True)
    if (type(payload.get("workflow_schema_version")) is not int
            or payload["workflow_schema_version"] != 1 or payload.get("workflow") != "observable_maps"
            or payload.get("plan_sha256") != _digest(plan) or payload.get("task") != task
            or payload.get("accelerator") != plan["accelerator"]):
        raise ValueError(f"Checkpoint identity/accelerator mismatch: {task['key']}")
    state = payload.get("session", {})
    history = state.get("history")
    if (state.get("training_kind") != "observable_maps" or not isinstance(history, list)
            or not history or type(state.get("step")) is not int or state["step"] < 0
            or type(state.get("epoch")) is not int or len(history) != state["epoch"] + 1
            or not isinstance(payload.get("elapsed_seconds"), (int, float))
            or not math.isfinite(payload["elapsed_seconds"]) or payload["elapsed_seconds"] < 0):
        raise ValueError("Checkpoint history/counters differ")
    return payload


def _reference_for_maps(feature_root, L, maps):
    reference = load_validation_fields(feature_root, L)
    rows = maps_to_edge_rows(maps["f"].validation)
    for name in ("y", "hamiltonian_ids", "hamiltonian_uids", "edge_ids"):
        if not np.array_equal(getattr(rows, name), getattr(reference.validation_rows, name)):
            raise ValueError("Validation maps and D5 field references differ in identity/order/labels")
    return reference


def _d5(reference, prediction, compiled):
    lifted = reference_sign_lift(reference.true_b_z, prediction.reshape(reference.true_b_z.shape))
    values = {}
    for model, metric in compiled.items():
        blocks = {name: [] for name in ("full", "traceless")}
        for start in range(0, len(lifted), 256):
            block = metric.distances(reference.true_b_z[start:start + 256], lifted[start:start + 256])
            for name in blocks:
                blocks[name].append(block[name])
        values.update({f"{model}_{name}": np.concatenate(blocks[name]) for name in blocks})
    return values


def _evaluate(job, session, maps, reference, compiled, plan, task, elapsed, accelerator):
    predictions, per_h, field_scores, scores = {}, [], [], {}
    base_rows = maps_to_edge_rows(maps["f"].validation)
    for view in VIEWS:
        mapped = maps[view].validation
        prediction = flatten_edge_maps(mapped, session.selected_predictions(mapped.x))
        if prediction.dtype != np.float32:
            raise ValueError("Selected predictions must remain float32")
        predictions[f"prediction_{view}"] = prediction
        metrics = per_hamiltonian_metrics(prediction, base_rows.y,
                                          base_rows.hamiltonian_ids, base_rows.edge_ids)
        per_h.extend({"input_view": view, **row} for row in metrics)
        distances = _d5(reference, prediction, compiled)
        scores[view] = {"mse": float(np.mean([row["mse"] for row in metrics])),
                        "mae": float(np.mean([row["mae"] for row in metrics])),
                        "mean_hamiltonian_rmse": float(np.mean([row["rmse"] for row in metrics])),
                        "hamiltonians": len(metrics), "rows": len(base_rows),
                        "negative_prediction_fraction": float(np.mean(prediction < 0)),
                        "above_1_7_prediction_fraction": float(np.mean(prediction > 1.7)),
                        "d5": {name: float(np.mean(value)) for name, value in distances.items()}}
        scores[view]["rmse"] = math.sqrt(scores[view]["mse"])
        for index, (name, uid) in enumerate(zip(reference.hamiltonian_ids, reference.hamiltonian_uids)):
            field_scores.append({"input_view": view, "hamiltonian_id": str(name),
                                 "hamiltonian_uid": str(uid),
                                 **{key: float(value[index]) for key, value in distances.items()}})
    baseline = _d5(reference, np.zeros_like(base_rows.y), compiled)
    _npz(job / "predictions.npz", **predictions, target=base_rows.y,
         hamiltonian_ids=base_rows.hamiltonian_ids, hamiltonian_uids=base_rows.hamiltonian_uids,
         edge_ids=base_rows.edge_ids)
    _csv(job / "per-hamiltonian.csv", per_h,
         ("input_view", "hamiltonian_id", "count", "mse", "rmse", "mae"))
    _csv(job / "hamiltonian-scores.csv", field_scores,
         ("input_view", "hamiltonian_id", "hamiltonian_uid", "two_channel_full",
          "two_channel_traceless", "z_branch_full", "z_branch_traceless"))
    _json(job / "history.json", session.history)
    evaluation = {"task": task, "scope": "development validation; not a new final test",
                  "checkpoint_selection": "lowest own-view validation MSE including epoch zero",
                  "best_step": session.best_step, "best_epoch": session.best_epoch,
                  "best_validation_mse": session.best_val_mse, "last_step": session.step,
                  "last_epoch": session.epoch, "stop_reason": session.stop_reason,
                  "elapsed_seconds": elapsed,
                  "elapsed_scope": "optimizer advances and epoch validation; excludes initial validation, loading, checkpoints and exports",
                  "accelerator": accelerator, "scores": scores,
                  "prediction_policy": "raw unclipped linear predictions",
                  "exact_input_policy": "crossed evaluation only; never checkpoint selection",
                  "hamiltonian_metric": {"primary": "two_channel_full", "distance": "unsquared normalized global-Pauli D5",
                      "sign_policy": "reference_sign_oracle", "x_fields": "zero",
                      "zero_field_baseline": {name: float(np.mean(value)) for name, value in baseline.items()}},
                  "test_payload_loaded": False}
    _json(job / "evaluation.json", evaluation)
    _torch(job / "best.pt", {"schema_version": 1, "plan_sha256": _digest(plan), "task": task,
                            "architecture": plan["config"]["architecture"],
                            "best_step": session.best_step, "best_epoch": session.best_epoch,
                            "model_state_dict": session.best_state_dict})
    done = {"schema_version": 1, "plan_sha256": _digest(plan), "task": task,
            "hashes": {name: _sha(job / name) for name in OUTPUT_FILES}, "summary": evaluation}
    _json(job / "done.json", done)
    return done


def _summary(run, plan):
    rows = []
    for task in plan["tasks"]:
        job = _job(run, task)
        done = _done(job, plan, task)
        payload = None if done else _load_checkpoint(job, plan, task)
        row = {**task, "state": "complete" if done else "paused" if payload else "pending",
               "step": 0, "epoch": 0, "best_step": None, "best_epoch": None,
               "best_validation_mse": None, "elapsed_seconds": 0.0}
        if done:
            value = done["summary"]
            row.update(step=value["last_step"], epoch=value["last_epoch"],
                       best_step=value["best_step"], best_epoch=value["best_epoch"],
                       best_validation_mse=value["best_validation_mse"], scores=value["scores"],
                       stop_reason=value["stop_reason"], elapsed_seconds=value["elapsed_seconds"])
        elif payload:
            value = payload["session"]
            row.update(step=value["step"], epoch=value["epoch"], best_step=value["best_step"],
                       best_epoch=value["best_epoch"], best_validation_mse=value["best_val_mse"],
                       early_stopping_wait=value["early_stopping_wait"],
                       elapsed_seconds=payload["elapsed_seconds"])
        rows.append(row)
    completed = sum(row["state"] == "complete" for row in rows)
    state = "complete" if completed == len(rows) else "paused" if any(row["state"] == "paused" for row in rows) else "pending"
    return {"schema_version": 1, "state": state, "completed": completed, "total": len(rows),
            "tasks": rows, "seeds": plan["config"]["seeds"], "sizes": plan["config"]["sizes"],
            "max_epochs": plan["config"]["training"]["max_epochs"],
            "scope": "development validation only", "test_payload_loaded": False}


def _paired_rows(summary):
    complete = {(r["L"], r["seed"], r["model"], r["view"]): r
                for r in summary["tasks"] if r["state"] == "complete"}
    result = []
    for L in summary["sizes"]:
        for seed in summary["seeds"]:
            if not all((L, seed, model, view) in complete for model in MODELS for view in ("f", "g")):
                continue
            for input_view in (*VIEWS, "matched_pipeline"):
                for metric in ("mse", "d5_two_channel_full"):
                    def score(model, trained_view):
                        view = trained_view if input_view == "matched_pipeline" else input_view
                        value = complete[L, seed, model, trained_view]["scores"][view]
                        return value["mse"] if metric == "mse" else value["d5"]["two_channel_full"]
                    values = {f"{model}_{view}": score(model, view) for model in MODELS for view in ("f", "g")}
                    result.append({"L": L, "seed": seed, "validation_input": input_view,
                                   "metric": metric, **values,
                                   "cnn_minus_mlp_f": values["cnn_f"] - values["map_mlp_f"],
                                   "cnn_minus_mlp_g": values["cnn_g"] - values["map_mlp_g"],
                                   "g_minus_f_cnn": values["cnn_g"] - values["cnn_f"],
                                   "g_minus_f_mlp": values["map_mlp_g"] - values["map_mlp_f"],
                                   "interaction": ((values["cnn_g"] - values["cnn_f"])
                                                   - (values["map_mlp_g"] - values["map_mlp_f"]))})
    return result


def format_summary(summary):
    """Compact progress plus fully paired, explicitly descriptive summaries."""
    lines = [f"Observable CNN campaign: {summary['completed']}/{summary['total']} complete ({summary['state']})"]
    for L in summary["sizes"]:
        subset = [r for r in summary["tasks"] if r["L"] == L]
        n = sum(r["state"] == "complete" for r in subset)
        lines.append(f"  L={L}: {n}/{len(subset)} complete")
    active = [r for r in summary["tasks"] if r["state"] == "paused"]
    for row in active:
        lines.append(f"  Paused {row['key']}: epoch {row['epoch']}/{summary['max_epochs']}, "
                     f"update {row['step']:,}, best MSE {row['best_validation_mse']:.7g} "
                     f"at epoch {row['best_epoch']}")
    paired = _paired_rows(summary)
    for L in summary["sizes"]:
        rows = [r for r in paired if r["L"] == L and r["validation_input"] == "matched_pipeline"
                and r["metric"] == "mse"]
        n = len(rows)
        lines.append(f"  L={L}: {n}/{len(summary['seeds'])} complete four-arm seed blocks")
        if n == len(summary["seeds"]):
            means = {name: float(np.mean([r[name] for r in rows]))
                     for name in ("cnn_f", "cnn_g", "map_mlp_f", "map_mlp_g")}
            lines.append("    All-seed matched-input validation MSE: " + ", ".join(
                f"{name}={value:.7g}" for name, value in means.items()))
            d5_rows = [r for r in paired if r["L"] == L and r["validation_input"] == "matched_pipeline"
                       and r["metric"] == "d5_two_channel_full"]
            d5_means = {name: float(np.mean([r[name] for r in d5_rows]))
                        for name in ("cnn_f", "cnn_g", "map_mlp_f", "map_mlp_g")}
            lines.append("    All-seed primary reference-sign D5: " + ", ".join(
                f"{name}={value:.7g}" for name, value in d5_means.items()))
        elif n:
            lines.append("    Partial blocks retained in CSV; all-seed comparison waits for every declared seed.")
    lines.append("Validation selected weights; these are development results, not fresh-test or correction accuracy.")
    return "\n".join(lines)


def _report_locked(run, plan):
    summary = _summary(run, plan)
    _json(run / "status.json", summary)
    _csv(run / "paired-validation.csv", _paired_rows(summary),
         ("L", "seed", "validation_input", "metric", "cnn_f", "cnn_g", "map_mlp_f", "map_mlp_g",
          "cnn_minus_mlp_f", "cnn_minus_mlp_g", "g_minus_f_cnn", "g_minus_f_mlp", "interaction"))
    text = ("# Observable-map CNN matched campaign\n\n"
            "This campaign compares periodic CNN and unchanged shared PaperMLP controls on identical "
            "Hamiltonian groups, observable information, whole-Hamiltonian batches and epoch budgets. "
            "Both architectures use f and g inputs with all declared seeds. Older edge-batched MLP runs "
            "provide context only. Exact inputs are crossed validation controls, never training/selection.\n\n"
            "Each selected model exports raw predictions and magnitude errors on f/g/exact validation "
            "inputs. D5 is evaluated only afterwards, with known reference signs and zero X fields; "
            "two-channel/full unsquared distance is primary. This does not learn signs or demonstrate "
            "physical correction. These are development groups, not a new final test.\n\n"
            "## Progress\n\n```text\n" + format_summary(summary) + "\n```\n\n"
            "## Files\n\n"
            "- `plan.json`, `plan-sha256.json`: frozen settings, tasks, source/environment/GPU and feature hashes.\n"
            "- Existing feature cache is read in place; no new Monte Carlo or bank is created.\n"
            "- `models/<L-seed-model-view>/latest.pt`: atomic live model, Adam, best model, shuffle/CPU/CUDA RNG, "
            "partial epoch and history. Ctrl+C drains one bounded update chunk and saves this state.\n"
            "- Finished jobs: `best.pt`, `history.json`, `evaluation.json`, `predictions.npz`, "
            "`per-hamiltonian.csv`, `hamiltonian-scores.csv`; `done.json` hashes every committed output.\n"
            "- `status.json`: derived progress; `paired-validation.csv`: every complete four-arm seed block, "
            "all crossed inputs and primary MSE/D5 contrasts. Partial blocks are not presented as final averages.\n\n"
            "CUDA is required for research training. Resume on the same pinned device/environment/source. "
            "The epoch ceiling and stopping opportunity match; actual runtime, FLOPs and early-stop epochs can differ.\n")
    _atomic(run / "RESULTS.md", lambda handle: handle.write(text))
    return summary


def summarize_campaign(repo_root, run):
    run = _safe_run(repo_root, run)
    with campaign_lock(run):
        return _report_locked(run, load_plan(repo_root, run))


def run_campaign(repo_root, run, *, sizes=None, max_chunks=None, output=print):
    """Sequential GPU training; SIGINT commits the current bounded chunk and stops."""
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer")
    repo_root, run = Path(repo_root).resolve(), _safe_run(repo_root, run)
    device = resolve_device("cuda", True)  # Fail before reading any feature payload.
    accelerator = _accelerator(device)
    with campaign_lock(run):
        plan = load_plan(repo_root, run)
        config = plan["config"]
        if accelerator != plan["accelerator"]:
            raise ValueError("Accelerator changed; exact continuation requires the pinned device")
        selected = config["sizes"] if sizes is None else list(sizes)
        if (not selected or any(type(L) is not int or L not in config["sizes"] for L in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("sizes must be a nonempty distinct subset of the frozen lattice sizes")
        stopped = [False]
        old_handler = signal.getsignal(signal.SIGINT)
        def request_pause(signum, frame):
            stopped[0] = True
        signal.signal(signal.SIGINT, request_pause)
        chunks, current_L, maps, reference, compiled = 0, None, None, None, None
        try:
            for task in plan["tasks"]:
                if task["L"] not in selected:
                    continue
                job = _job(run, task)
                if _done(job, plan, task):
                    continue
                if stopped[0] or max_chunks is not None and chunks >= max_chunks:
                    break
                if current_L != task["L"]:
                    feature_root = _feature_root(repo_root, config)
                    if _feature_hashes(feature_root, config["sizes"]) != plan["feature_hashes"]:
                        raise ValueError("Frozen feature cache changed before lattice loading")
                    maps = _load_maps(feature_root, task["L"])
                    if _bindings(maps) != plan["data_bindings"][str(task["L"])]:
                        raise ValueError("Hamiltonian data bindings differ from the frozen plan")
                    reference = _reference_for_maps(feature_root, task["L"], maps)
                    if _feature_hashes(feature_root, config["sizes"]) != plan["feature_hashes"]:
                        raise ValueError("Frozen feature cache changed during lattice loading")
                    compiled = {model: CompiledHamiltonianMetric(square_lattice(task["L"]), model=model)
                                for model in ("two_channel", "z_branch")}
                    current_L = task["L"]
                data = maps[task["view"]]
                if data.provenance.get("training_eligible") is not True:
                    raise ValueError("Attempt to train an evaluation-only view")
                payload = _load_checkpoint(job, plan, task)
                engine_config = {**config["training"], "model": task["model"],
                                 "architecture": config["architecture"], "seed": task["seed"]}
                session = ObservableTrainSession(data.train.x, data.train.y, data.validation.x,
                    data.validation.y, engine_config, checkpoint=payload["session"] if payload else None)
                elapsed = payload["elapsed_seconds"] if payload else 0.0
                job.mkdir(parents=True, exist_ok=True)
                output(f"{task['key']}: CUDA; epoch {session.epoch}/{engine_config['max_epochs']} "
                       f"(Ctrl+C saves and pauses)")
                last_output = time.monotonic() - 10.0
                while not session.finished and not stopped[0] and (max_chunks is None or chunks < max_chunks):
                    started = time.monotonic()
                    session.advance(config["execution"]["checkpoint_updates"])
                    elapsed += time.monotonic() - started
                    _checkpoint(job, session, plan, task, elapsed, accelerator)
                    chunks += 1
                    now = time.monotonic()
                    if (now - last_output >= 5 or session.finished or stopped[0]
                            or max_chunks is not None and chunks >= max_chunks):
                        output(f"  epoch {session.epoch}/{engine_config['max_epochs']}; update {session.step:,}; "
                               f"best MSE {session.best_val_mse:.7g} at epoch {session.best_epoch}")
                        last_output = now
                if session.finished:
                    if not (job / "latest.pt").exists():
                        _checkpoint(job, session, plan, task, elapsed, accelerator)
                    _evaluate(job, session, maps, reference, compiled, plan, task, elapsed, accelerator)
                    output(f"  complete: {task['key']} ({session.stop_reason})")
                elif not (job / "latest.pt").exists():
                    # SIGINT during initialization still preserves eligible epoch-zero weights.
                    _checkpoint(job, session, plan, task, elapsed, accelerator)
                del session
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _report_locked(run, plan)
            summary = _report_locked(run, plan)
            output(format_summary(summary))
            return summary
        finally:
            signal.signal(signal.SIGINT, old_handler)
