"""Sequential size/mode queues around the existing, checkpointed MLP workflow.

Queue inspection reads JSON summaries only. Training never opens final-test
payloads; ``evaluate_queue`` is a separate, explicit frozen-model operation.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

from . import mlp_campaign as paired

REPO_ROOT = paired.REPO_ROOT
ARTIFACT_PATH = paired.ARTIFACT_PATH
SIZES = (3, 4, 8, 12, 16, 20, 24)
MODES = paired.MODES


def _copy(value):
    return json.loads(paired._canonical(value))


def _digest(value):
    return hashlib.sha256(paired._canonical(value).encode()).hexdigest()


def _size_config(config, size):
    """Adjust only the canonical campaign size suffix; never guess custom paths."""
    value = _copy(config)
    if size != value["lattice_size"]:
        old = f"L{value['lattice_size']}/run"
        if not value["dataset_path"].endswith(old):
            raise ValueError("Custom dataset path: configure each lattice explicitly, or use the campaign L<size>/run layout")
        value["dataset_path"] = value["dataset_path"][:-len(old)] + f"L{size}/run"
        value["lattice_size"] = size
    return value


def _choices(sizes, modes):
    sizes = tuple(SIZES if sizes is None else sizes)
    modes = tuple(modes)
    if not sizes or any(isinstance(size, bool) or not isinstance(size, int) or size not in SIZES for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError(f"sizes must be a nonempty sequence of distinct values from {SIZES}")
    if modes not in (("paper_size",), ("all_edges",), MODES):
        raise ValueError("modes must be paper_size, all_edges, or paper_size then all_edges")
    return sizes, modes


def _metadata(config, repo_root):
    record = paired._dataset_metadata(config, repo_root)
    root = Path(config["dataset_path"])
    if not root.is_absolute():
        root = Path(repo_root) / root
    status_path = root / "status.json"
    if not status_path.is_file():
        raise ValueError(f"Dataset status is missing: {status_path}")
    record["metadata_sha256"]["status.json"] = hashlib.sha256(status_path.read_bytes()).hexdigest()
    return record


def _readiness(config, repo_root):
    """Check saved completion/verification declarations, without loading NPZs."""
    root = Path(config["dataset_path"])
    if not root.is_absolute():
        root = Path(repo_root) / root
    root = root.resolve()
    result = {"dataset_path": str(root), "state": "not_started"}
    try:
        status = json.loads((root / "status.json").read_text())
        if not isinstance(status, dict):
            raise ValueError("dataset status must be a JSON object")
        result.update(status)
        if status.get("state") != "complete":
            return result
        metadata = _metadata(config, repo_root)
        records = {name: json.loads((root / name).read_text()) for name in
                   ("config.json", "manifest.json", "dataset.json", "verification.json")}
        if any(not isinstance(value, dict) for value in records.values()):
            raise ValueError("dataset metadata must contain JSON objects")
        if records["verification.json"].get("passed") is not True:
            raise ValueError("dataset verification has not passed")
        if records["config.json"].get("lattice", {}).get("L") != config["lattice_size"]:
            raise ValueError("dataset lattice size differs from requested size")
        planned = records["manifest.json"].get("instances", [])
        indexed = records["dataset.json"].get("instances", [])
        verified = records["verification.json"].get("instances", [])
        identifiers = [item["id"] for item in planned]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset Hamiltonian plan is empty or has duplicate identities")
        if status.get("completed") != len(planned):
            raise ValueError("dataset completed count differs from its plan")
        for entries, label in ((indexed, "index"), (verified, "verification")):
            if len(entries) != len(planned) or {item["id"] for item in entries} != set(identifiers):
                raise ValueError(f"dataset {label} does not cover every planned Hamiltonian")
        by_id = {item["id"]: item for item in planned}
        index = {item["id"]: item for item in indexed}
        counts = {split: sum(item.get("split") == split for item in planned)
                  for split in ("train", "validation", "test")}
        if not all(counts.values()):
            raise ValueError("dataset must declare nonempty Hamiltonian train/validation/test groups")
        if "paper_size" == config["mode"] and counts["train"] < config["training_hamiltonians"]:
            raise ValueError("dataset has too few training Hamiltonians for paper_size")
        if records["dataset.json"].get("split_counts") != counts:
            raise ValueError("dataset split counts differ from its plan")
        for check in verified:
            entry = index[check["id"]]
            if check.get("split") != by_id[check["id"]].get("split") or entry.get("split") != check.get("split"):
                raise ValueError("dataset split declarations disagree")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or check.get("sha256") != digest:
                raise ValueError("dataset checksum declarations disagree")
        result.update(metadata=metadata, split_counts=counts)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(state="unverified" if root.exists() else "not_started", error=str(error))
    return result


def discover_ready_sizes(config, sizes=SIZES, *, repo_root=REPO_ROOT):
    """Return completion declarations for each size; never read data payloads."""
    selected, _ = _choices(sizes, MODES)
    return {size: _readiness(_size_config(config, size), repo_root) for size in selected}


def _safe_queue(path, repo_root):
    root = (Path(repo_root) / ARTIFACT_PATH / "queues").resolve()
    queue = Path(path).resolve()
    if queue == root or not queue.is_relative_to(root):
        raise ValueError(f"MLP queues must be below {root}")
    return queue


def _artifact_child(relative, repo_root):
    path = Path(relative)
    root = (Path(repo_root) / ARTIFACT_PATH).resolve()
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("queue child path must be relative without traversal")
    child = (root / path).resolve()
    if child == root or not child.is_relative_to(root):
        raise ValueError("queue child path escapes the phase artifact directory")
    return child


@contextmanager
def _lock(queue):
    with (queue / ".queue.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This MLP queue is already active in another process") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validate_record(record, queue, repo_root):
    if not isinstance(record, dict) or not isinstance(record.get("plan"), dict):
        raise ValueError("Saved queue record/plan must be JSON objects")
    plan = record["plan"]
    if record.get("schema_version") != 1 or record.get("plan_sha256") != _digest(plan):
        raise ValueError("Saved queue plan/configuration changed or has an unsupported schema")
    sizes, modes = _choices(plan.get("sizes", ()), plan.get("modes", ()))
    groups = plan.get("groups", [])
    if len(groups) != len(sizes) or [group.get("size") for group in groups] != list(sizes):
        raise ValueError("Saved queue task order changed")
    token = plan.get("queue_token")
    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Invalid saved queue identity")
    for group in groups:
        expected = _size_config(plan["config"], group["size"])
        expected["mode"] = modes[0]
        if group.get("config") != expected:
            raise ValueError("Saved queue per-size configuration differs from its immutable base")
        if modes == MODES:
            canonical = f"campaigns/queue-{token}-L{group['size']}"
            if group.get("campaign_path") != canonical:
                raise ValueError("Saved queue campaign path changed")
            _artifact_child(canonical, repo_root)
        else:
            canonical = queue.relative_to((Path(repo_root) / ARTIFACT_PATH).resolve()) / f"L{group['size']}" / modes[0] / "run"
            if group.get("run_path") != str(canonical):
                raise ValueError("Saved queue model path changed")
            _artifact_child(canonical, repo_root)
    attempts = record.get("attempts", {})
    if not isinstance(attempts, dict) or any(
            size not in {str(value) for value in sizes} or isinstance(count, bool)
            or not isinstance(count, int) or count < 0
            for size, count in attempts.items()):
        raise ValueError("Invalid queue initialization-attempt record")
    return plan


def _effective_group(group, record):
    result = dict(group)
    attempt = record.get("attempts", {}).get(str(group["size"]), 0)
    if attempt:
        if "campaign_path" in group:
            result["campaign_path"] = group["campaign_path"] + f"-retry_{attempt:03d}"
        else:
            result["run_path"] = str(Path(group["run_path"]).parent / f"retry_{attempt:03d}")
    return result


def _retry_group(group, record, queue, repo_root):
    key = str(group["size"])
    attempts = record.setdefault("attempts", {})
    while True:
        attempts[key] = attempts.get(key, 0) + 1
        fresh = _effective_group(group, record)
        path = _artifact_child(fresh.get("campaign_path", fresh.get("run_path")), repo_root)
        if not path.exists():
            break
    _save(queue, record, repo_root)
    return fresh


def _initialized(run):
    return all((run / name).is_file() for name in ("config.json", "status.json", "latest.pt"))


def _check_group_settings(group, modes, repo_root):
    if modes == MODES:
        campaign = _artifact_child(group["campaign_path"], repo_root)
        record = json.loads((campaign / "campaign.json").read_text())
        expected = _copy(group["config"])
        expected["mode"] = "paper_size"
        if (record.get("schema_version") != 1 or record.get("modes") != list(MODES)
                or record.get("config") != expected
                or record.get("config_sha256") != _digest(expected)
                or record.get("common_dataset") != paired._dataset_metadata(expected, repo_root)):
            raise ValueError("Saved paired child configuration/data changed")
    for mode, run in _group_run_paths(group, modes, repo_root).items():
        paired._check_child_settings(run, group["config"], mode)


def _group_run_paths(group, modes, repo_root):
    if modes == MODES:
        campaign = _artifact_child(group["campaign_path"], repo_root)
        record_path = campaign / "campaign.json"
        if not record_path.exists():
            return {mode: campaign / mode / "run" for mode in modes}
        record = json.loads(record_path.read_text())
        if not isinstance(record, dict) or not isinstance(record.get("runs"), dict):
            raise ValueError("Saved paired campaign record/runs must be JSON objects")
        if any(not isinstance(record["runs"].get(mode), dict) or not isinstance(record["runs"][mode].get("path"), str)
               for mode in modes):
            raise ValueError("Saved paired campaign is missing queue child paths")
        return {mode: paired._child_dir(campaign, record["runs"][mode]["path"]) for mode in modes}
    return {modes[0]: _artifact_child(group["run_path"], repo_root)}


def _summary(record, queue, repo_root):
    plan = _validate_record(record, queue, repo_root)
    tasks = []
    for group in plan["groups"]:
        for mode, path in _group_run_paths(_effective_group(group, record), tuple(plan["modes"]), repo_root).items():
            summary = paired.run_summary(path)
            summary.update(L=group["size"], mode=mode)
            tasks.append(summary)
    completed = sum(item["state"] in paired.FINISHED_STATES for item in tasks)
    return {"queue_dir": str(queue), "state": record.get("state", "unknown"),
            "current_size": record.get("current_size"), "current_mode": record.get("current_mode"),
            "sizes": plan["sizes"], "modes": plan["modes"], "completed_models": completed,
            "total_models": len(tasks), "evaluated_models": sum(bool(t["test_evaluated"]) for t in tasks),
            "evaluation_state": record.get("evaluation_state", "not_started"),
            "tasks": tasks, "results_path": str(queue / "RESULTS.md"),
            "comparison_path": str(queue / "comparison.csv"), "error": record.get("error")}


def queue_summary(queue_dir, *, repo_root=REPO_ROOT):
    """Inspect queue and scalar model records; no PyTorch/checkpoint/data reads."""
    queue = _safe_queue(queue_dir, repo_root)
    return _summary(json.loads((queue / "queue.json").read_text()), queue, repo_root)


def format_queue_summary(summary):
    lines = [f"Training queue: {str(summary['state']).upper()} | {summary['completed_models']}/{summary['total_models']} models trained",
             f"Final evaluation: {summary.get('evaluation_state', 'not_started')} | {summary.get('evaluated_models', 0)}/{summary['total_models']} models evaluated",
             " L   mode         state        best validation MSE    best update    final test"]
    for item in summary["tasks"]:
        validation = paired._number(item.get("best_validation_mse"))
        step = paired._number(item.get("best_step"))
        test = ("RMSE " + paired._number(item.get("test_metrics", {}).get("rmse"))) if item["test_evaluated"] else "not evaluated"
        lines.append(f" {item['L']:<3} {item['mode']:<12} {item['state']:<12} {validation:<22} {step:<14} {test}")
    lines.append(f"Results: {summary['results_path']}")
    lines.append(f"Comparison: {summary['comparison_path']}")
    if summary.get("error"):
        lines.append(f"Stopped: {summary['error']}")
    return "\n".join(lines)


def _save(queue, record, repo_root):
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    paired._atomic_json(queue / "queue.json", record)
    summary = _summary(record, queue, repo_root)
    paired._atomic_json(queue / "comparison.json", summary)
    fields = ["L", "mode", "state", "step", "max_steps", "stop_reason", "epoch", "training_row_passes", "training_examples_seen", "elapsed_seconds", "best_validation_mse", "best_step", "train_rows", "validation_rows",
              "test_rows", "test_evaluated", "test_mse", "test_rmse", "test_mae", "run_dir"]
    temporary = queue / "comparison.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary["tasks"]:
            row = {key: item.get(key) for key in fields}
            row.update({f"{split}_rows": item["row_counts"].get(split) for split in ("train", "validation", "test")})
            row.update({f"test_{metric}": item["test_metrics"].get(metric) for metric in ("mse", "rmse", "mae")})
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, queue / "comparison.csv")
    text = f"""# Sequential MLP queue results

{format_queue_summary(summary)}

This queue schedules independent size-specific models in the recorded size order,
then paper_size before all_edges when both are requested. It preserves the
existing architecture, Adam, seeds, stopping policy and Hamiltonian-level splits.
Each size uses its own dataset. The paired modes receive the same configured
initialization seed; no edge or Hamiltonian is moved between splits.

Training opens training/validation summaries only. Final test evaluation is a
separate explicit evaluate-queue operation after all models finish. Existing
saved test scalar metrics, when present, appear separately in comparison.csv.
The two modes' validation sets contain different edge rows, so validation MSE
alone is not a shared-test comparison. Final evaluation uses all held-out edges
of the same Hamiltonians for both modes within each lattice size.

Files: queue.json (immutable ordered plan, metadata hashes and mutable progress),
comparison.json and comparison.csv (saved scalar summaries), RESULTS.md (this
report). Child run directories are listed in comparison.csv; each retains its
own config, checkpoint, validation-selected model, learning curves and RESULTS.md.
Paired child campaigns live under the phase campaigns/ directory. The queue
never deletes previous attempts or automatically reuses unrelated existing runs.

Pause or quit stops this entire sequence at its current model; resume skips
finished models. Ctrl+C retains the last atomic child checkpoint and discards
an incomplete optimizer chunk. Initialization failures without a checkpoint
remain preserved and are retried in a fresh child path. Readiness checks verify
saved metadata declarations before starting; the existing training loader also
checks training/validation payload hashes. Neither check certifies Monte Carlo
mixing or recovers the paper's unavailable historical training selection.
"""
    (queue / "RESULTS.md").write_text(text)
    return summary


def _preflight(plan, repo_root, *, frozen):
    for group in plan["groups"]:
        ready = _readiness(group["config"], repo_root)
        if ready["state"] != "complete":
            raise ValueError(f"L={group['size']} dataset is {ready['state']}: {ready.get('error', ready['dataset_path'])}")
        if frozen and group["dataset"] != ready["metadata"]:
            raise ValueError(f"L={group['size']} dataset metadata changed since queue creation")


def _run_queue(config=None, *, sizes=None, modes=MODES, queue_dir=None, resume=False,
              interactive=True, repo_root=REPO_ROOT, max_chunks=None):
    """Train selected sizes/modes one by one; an unfinished child stops the queue."""
    from quantum_lattice.models import mlp_experiment as workflow
    from .mlp_terminal import training_context, training_stop_requested

    if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer")
    if resume and queue_dir is None:
        raise ValueError("resume requires a saved queue directory")
    if not resume and config is None:
        raise ValueError("A configuration is required for a new queue")
    if queue_dir is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        queue_dir = Path(repo_root) / ARTIFACT_PATH / "queues" / name
    queue = _safe_queue(queue_dir, repo_root)
    if not resume:
        sizes, modes = _choices(sizes, modes)
        base = _copy(config)
        base["mode"] = modes[0]
        base = workflow.validate_config(base)
        token = uuid.uuid4().hex
        groups = []
        for size in sizes:
            child = workflow.validate_config(_size_config(base, size))
            ready = _readiness(child, repo_root)
            if ready["state"] != "complete":
                raise ValueError(f"L={size} dataset is {ready['state']}: {ready.get('error', ready['dataset_path'])}")
            group = {"size": size, "config": child, "dataset": ready["metadata"]}
            if modes == MODES:
                group["campaign_path"] = f"campaigns/queue-{token}-L{size}"
            else:
                group["run_path"] = str(queue.relative_to((Path(repo_root) / ARTIFACT_PATH).resolve()) / f"L{size}" / modes[0] / "run")
            groups.append(group)
        plan = {"config": base, "sizes": list(sizes), "modes": list(modes), "groups": groups, "queue_token": token}
        if queue.exists() and any(queue.iterdir()):
            raise ValueError("New queue directory must be empty; resume a saved queue")
        record = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "plan": plan, "plan_sha256": _digest(plan), "state": "prepared",
                  "current_size": None, "current_mode": None,
                  "test_protocol": "No automatic final test evaluation"}
        queue.mkdir(parents=True, exist_ok=True)
    else:
        if not (queue / "queue.json").is_file():
            raise ValueError("queue.json is missing; this is not a resumable queue")
    with _lock(queue):
        if resume:
            record = json.loads((queue / "queue.json").read_text())
            plan = _validate_record(record, queue, repo_root)
            if config is not None:
                proposed = _copy(config)
                proposed["mode"] = plan["modes"][0]
                if workflow.validate_config(proposed) != plan["config"]:
                    raise ValueError("Resume settings differ from the immutable queue configuration")
            if sizes is not None and list(sizes) != plan["sizes"]:
                raise ValueError("Resume sizes differ from the immutable queue order")
        else:
            _validate_record(record, queue, repo_root)
        try:
            _preflight(plan, repo_root, frozen=True)
            _save(queue, record, repo_root)
            selected_modes = tuple(plan["modes"])
            total = len(plan["sizes"]) * len(selected_modes)
            for index, group in enumerate(plan["groups"]):
                # Recheck every declared bank before each size, including a bank
                # that changed while an earlier model was training.
                _preflight(plan, repo_root, frozen=True)
                record.pop("error", None)
                record.update(state="training", current_size=group["size"], current_mode=selected_modes[0])
                _save(queue, record, repo_root)
                print(f"\nQueue L={group['size']} | models {index * len(selected_modes) + 1}–{(index + 1) * len(selected_modes)}/{total}: " + " → ".join(selected_modes), flush=True)
                if selected_modes == MODES:
                    effective = _effective_group(group, record)
                    campaign = _artifact_child(effective["campaign_path"], repo_root)
                    initialized = (campaign / "campaign.json").exists()
                    if campaign.exists() and any(campaign.iterdir()) and not initialized:
                        effective = _retry_group(group, record, queue, repo_root)
                        campaign = _artifact_child(effective["campaign_path"], repo_root)
                    with training_context(size=group["size"], queue_index=f"{index * len(selected_modes) + 1}–{(index + 1) * len(selected_modes)}", queue_total=total):
                        returned = paired.train_both(group["config"], campaign_dir=campaign, resume=initialized,
                                                     interactive=interactive, repo_root=repo_root, max_chunks=max_chunks)
                    if Path(returned).resolve() != campaign:
                        raise ValueError("Training returned a different queue campaign directory")
                else:
                    mode = selected_modes[0]
                    effective = _effective_group(group, record)
                    run = _artifact_child(effective["run_path"], repo_root)
                    if _initialized(run):
                        paired._check_child_settings(run, group["config"], mode)
                    summary = paired.run_summary(run)
                    if summary["state"] in paired.FINISHED_STATES and not _initialized(run):
                        raise ValueError("Finished child is missing checkpoint files; preserve it for inspection")
                    if summary["state"] not in paired.FINISHED_STATES:
                        initialized = _initialized(run)
                        if run.exists() and any(run.iterdir()) and not initialized:
                            effective = _retry_group(group, record, queue, repo_root)
                            run = _artifact_child(effective["run_path"], repo_root)
                        with training_context(size=group["size"], mode=mode, queue_index=index + 1, queue_total=total):
                            returned = workflow.train(group["config"], run_dir=run, resume=initialized,
                                                      interactive=interactive, repo_root=repo_root, max_chunks=max_chunks)
                        if Path(returned).resolve() != run:
                            raise ValueError("Training returned a different queue child directory")
                summary = _save(queue, record, repo_root)
                current = [task for task in summary["tasks"] if task["L"] == group["size"]]
                if any(task["state"] not in paired.FINISHED_STATES for task in current) or training_stop_requested():
                    unfinished = next((task["mode"] for task in current if task["state"] not in paired.FINISHED_STATES), current[-1]["mode"])
                    record.update(state="paused", current_mode=unfinished)
                    _save(queue, record, repo_root)
                    return queue
            record.update(state="trained", current_size=None, current_mode=None)
            _save(queue, record, repo_root)
            return queue
        except KeyboardInterrupt:
            record.update(state="paused")
            _save(queue, record, repo_root)
            return queue
        except Exception as error:
            record.update(state="failed", error=f"{type(error).__name__}: {error}")
            _save(queue, record, repo_root)
            raise


def evaluate_queue(queue_dir, *, repo_root=REPO_ROOT):
    """Explicitly evaluate every finished frozen model; rerun continues safely."""
    from quantum_lattice.models import mlp_experiment as workflow

    queue = _safe_queue(queue_dir, repo_root)
    with _lock(queue):
        record = json.loads((queue / "queue.json").read_text())
        plan = _validate_record(record, queue, repo_root)
        _preflight(plan, repo_root, frozen=True)
        summary = _summary(record, queue, repo_root)
        if summary["completed_models"] != summary["total_models"]:
            raise ValueError("Finish every queued model before opening final test data")
        # Validate all child configurations before evaluating the first model.
        for group in plan["groups"]:
            _check_group_settings(_effective_group(group, record), tuple(plan["modes"]), repo_root)
        try:
            record.pop("error", None)
            record["evaluation_state"] = "evaluating"
            _save(queue, record, repo_root)
            for group in plan["groups"]:
                _preflight(plan, repo_root, frozen=True)
                for mode, run in _group_run_paths(_effective_group(group, record), tuple(plan["modes"]), repo_root).items():
                    current = paired.run_summary(run)
                    if current["read_errors"]:
                        raise ValueError("Unreadable saved model summary: " + "; ".join(current["read_errors"]))
                    if current["state"] == "evaluated" and current["test_evaluated"]:
                        if not current["test_metrics"]:
                            raise ValueError("Evaluated child has no saved test metrics; preserve it for inspection")
                        continue
                    record.update(current_size=group["size"], current_mode=mode)
                    _save(queue, record, repo_root)
                    workflow.evaluate(run, repo_root=repo_root)
                    completed = paired.run_summary(run)
                    if completed["state"] != "evaluated" or not completed["test_evaluated"] or not completed["test_metrics"]:
                        raise ValueError("Evaluation did not publish complete saved test results")
                    _save(queue, record, repo_root)
            record.update(evaluation_state="evaluated", current_size=None, current_mode=None)
            return _save(queue, record, repo_root)
        except KeyboardInterrupt:
            record["evaluation_state"] = "paused"
            return _save(queue, record, repo_root)
        except Exception as error:
            record.update(evaluation_state="failed", error=f"{type(error).__name__}: {error}")
            _save(queue, record, repo_root)
            raise


def run_queue(config=None, *, sizes=None, modes=MODES, queue_dir=None, resume=False,
              interactive=True, repo_root=REPO_ROOT, max_chunks=None):
    """Run an ordered queue with one shared stop request across model boundaries."""
    from .mlp_terminal import training_context
    with training_context():
        return _run_queue(config, sizes=sizes, modes=modes, queue_dir=queue_dir,
                          resume=resume, interactive=interactive, repo_root=repo_root,
                          max_chunks=max_chunks)
