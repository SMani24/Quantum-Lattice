"""Small-JSON run summaries and sequential, resumable MLP comparisons.

Summary inspection never loads PyTorch, a checkpoint, or learning payloads.
Each campaign trains two fresh models; final-test evaluation stays explicit.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = Path("codex material/artifacts/phase1_mlp")
MODES = ("paper_size", "all_edges")
FINISHED_STATES = {"trained", "evaluated", "evaluating"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _dataset_metadata(config, repo_root):
    """Freeze common bank declarations; do not inspect any instance payload."""
    dataset = Path(config["dataset_path"])
    if not dataset.is_absolute():
        dataset = Path(repo_root) / dataset
    dataset = dataset.resolve()
    hashes = {}
    for name in ("config.json", "manifest.json", "dataset.json", "verification.json"):
        path = dataset / name
        if not path.is_file():
            raise ValueError(f"Dataset metadata is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    return {"dataset_path": config["dataset_path"], "metadata_sha256": hashes}


def _optional_json(path, errors):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("record is not a JSON object")
        return value
    except (OSError, ValueError) as error:
        errors.append(f"{path.name}: {error}")
        return {}


def run_summary(run_dir) -> dict:
    """Read saved scalar summaries only, tolerating incomplete/legacy runs."""
    run = Path(run_dir).resolve()
    errors = []
    config = _optional_json(run / "config.json", errors)
    status = _optional_json(run / "status.json", errors)
    manifest = _optional_json(run / "manifest.json", errors)
    selection = _optional_json(run / "data_selection.json", errors)
    metrics = _optional_json(run / "test_metrics.json", errors)
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    early = status.get("early_stopping", training.get("early_stopping", {}))
    if not isinstance(early, dict):
        early = {"enabled": bool(early)}
    row_counts = manifest.get("row_counts") or selection.get("row_counts") or {}
    if not isinstance(row_counts, dict):
        row_counts = {}
    state = status.get("state", "prepared" if config else "not_started")
    if not isinstance(state, str):
        state = "unknown"
    stop_reason = status.get("stop_reason")
    if not stop_reason and state in FINISHED_STATES:
        try:
            if int(status.get("step")) >= int(status.get("max_steps", training.get("max_steps"))):
                stop_reason = "max_steps (inferred from legacy record)"
        except (TypeError, ValueError):
            pass
    return {"run_dir": str(run), "L": config.get("lattice_size"), "mode": config.get("mode"),
            "device": status.get("device", manifest.get("device", training.get("device"))),
            "state": state, "step": status.get("step"),
            "max_steps": status.get("max_steps", training.get("max_steps")),
            "epoch": status.get("epoch"), "training_row_passes": status.get("training_row_passes"),
            "training_examples_seen": status.get("training_examples_seen"), "row_counts": row_counts,
            "train_mse": status.get("train_mse"), "validation_mse": status.get("validation_mse"),
            "best_validation_mse": status.get("best_validation_mse"), "best_step": status.get("best_step"),
            "stop_reason": stop_reason, "elapsed_seconds": status.get("elapsed_seconds"),
            "early_stopping": early,
            "early_stopping_wait": status.get("early_stopping_wait", status.get("checks_without_improvement")),
            "test_evaluated": bool(status.get("test_evaluated", state == "evaluated")),
            "test_metrics": {key: metrics[key] for key in ("mse", "rmse", "mae", "count", "hamiltonians") if key in metrics},
            "results_path": str(run / "RESULTS.md"), "results_exists": (run / "RESULTS.md").exists(),
            "read_errors": errors}


def _number(value, digits=6):
    if value is None:
        return "not recorded"
    try:
        numeric = float(value)
        return f"{numeric:.{digits}g}" if math.isfinite(numeric) else "not recorded"
    except (ValueError, TypeError):
        return "not recorded"


def _elapsed(seconds):
    try:
        seconds = max(0, int(float(seconds)))
    except (ValueError, TypeError, OverflowError):
        return "not recorded"
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_run_summary(summary) -> str:
    """Format saved progress, validation selection, and explicit test results."""
    rows = summary.get("row_counts", {})
    early = summary.get("early_stopping", {})
    lines = [f"L={summary.get('L') or '?'} · {summary.get('mode') or 'unknown mode'} · {str(summary.get('state', 'unknown')).upper()}",
             f"Optimizer updates: {_number(summary.get('step'))}/{_number(summary.get('max_steps'))} · elapsed {_elapsed(summary.get('elapsed_seconds'))}"]
    lines.append(f"Compute device: {summary.get('device') or 'not recorded'}")
    if summary.get("epoch") is not None or summary.get("training_row_passes") is not None:
        lines.append(f"Shuffled epoch: {_number(summary.get('epoch'))} · equivalent row passes: {_number(summary.get('training_row_passes'))}")
    lines.append("Rows: train " + _number(rows.get("train")) + " · validation " + _number(rows.get("validation")) + " · test " + _number(rows.get("test")))
    lines.append(f"MSE: train {_number(summary.get('train_mse'))} · current validation {_number(summary.get('validation_mse'))}")
    lines.append(f"Best validation MSE: {_number(summary.get('best_validation_mse'))} at update {_number(summary.get('best_step'))}")
    lines.append(f"Stop reason: {summary.get('stop_reason') or 'not recorded'}")
    if early.get("enabled", False):
        patience = early.get("patience_checks", early.get("patience"))
        lines.append(f"Early stopping: enabled · wait {_number(summary.get('early_stopping_wait'))}/{_number(patience)} validation checks · min_delta {_number(early.get('min_delta'))} · minimum {_number(early.get('min_steps'))} updates")
    elif "enabled" in early:
        lines.append("Early stopping: disabled (fixed update budget)")
    else:
        lines.append("Early stopping: absent from this legacy record")
    metrics = summary.get("test_metrics", {})
    if summary.get("test_evaluated") and metrics:
        lines.append(f"Final test: MSE {_number(metrics.get('mse'))} · RMSE {_number(metrics.get('rmse'))} · MAE {_number(metrics.get('mae'))} ({_number(metrics.get('count'))} rows)")
    elif summary.get("test_evaluated"):
        lines.append("Final test: evaluated; metric record unavailable")
    else:
        lines.append("Final test: not evaluated")
    lines.append(f"RESULTS: {summary.get('results_path', 'not recorded')}")
    if summary.get("read_errors"):
        lines.append("Unreadable records: " + "; ".join(summary["read_errors"]))
    return "\n".join(lines)


def _safe_campaign_dir(path, repo_root):
    root = (Path(repo_root) / ARTIFACT_PATH / "campaigns").resolve()
    campaign = Path(path).resolve()
    if campaign == root or not campaign.is_relative_to(root):
        raise ValueError(f"MLP campaigns must be below {root}")
    return campaign


def _child_dir(campaign, path):
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("campaign child run must have a relative path without traversal")
    child = (campaign / relative).resolve()
    if child == campaign or not child.is_relative_to(campaign):
        raise ValueError("campaign child path escapes its campaign")
    return child


@contextmanager
def _campaign_lock(campaign):
    with (campaign / ".campaign.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This MLP campaign is already active in another process") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fresh_retry_path(campaign, mode, previous):
    """Preserve an uninitialized failed directory instead of deleting it."""
    base = campaign / mode
    index = 1
    while (base / f"retry_{index:03d}").exists():
        index += 1
    return f"{mode}/retry_{index:03d}", previous


def _check_child_settings(run, base, mode):
    if not (run / "config.json").exists():
        return
    actual = json.loads((run / "config.json").read_text())
    expected = json.loads(_canonical(base))
    expected["mode"] = mode
    # The workflow records additional engine defaults in training; every
    # explicitly configured value remains immutable across the campaign.
    def supplied_values_match(recorded, configured):
        if isinstance(configured, dict):
            return isinstance(recorded, dict) and all(
                key in recorded and supplied_values_match(recorded[key], value)
                for key, value in configured.items())
        return recorded == configured

    for key, value in expected.items():
        if key == "training":
            if not supplied_values_match(actual.get(key, {}), value):
                raise ValueError("Saved child training settings differ from the campaign")
        elif actual.get(key) != value:
            raise ValueError("Saved child run settings differ from the campaign")


def _train_both(config, *, campaign_dir=None, resume=False, interactive=True,
               repo_root=REPO_ROOT, max_chunks=None) -> Path:
    """Train paper_size then all_edges; pausing stops the sequence immediately.

    A resumed campaign skips finished models and continues its first unfinished
    model. Both use the same configured initialization seed. Test evaluation is
    never called by this function.
    """
    # Import the actual training workflow only when execution is requested.
    from quantum_lattice.models import mlp_experiment as workflow

    if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer")
    new_base = None
    new_dataset = None
    if not resume:
        new_base = json.loads(_canonical(config))
        new_base["mode"] = "paper_size"
        new_base = workflow.validate_config(new_base)
        new_dataset = _dataset_metadata(new_base, repo_root)
    if resume and campaign_dir is None:
        raise ValueError("resume requires a saved campaign directory")
    if campaign_dir is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        campaign_dir = Path(repo_root) / ARTIFACT_PATH / "campaigns" / f"{name}-both"
    campaign = _safe_campaign_dir(campaign_dir, repo_root)
    if resume and not (campaign / "campaign.json").exists():
        raise ValueError("campaign.json is missing; this is not a resumable campaign")
    if not resume and campaign.exists() and any(campaign.iterdir()):
        raise ValueError("New campaign directory must be empty; resume a saved campaign")
    campaign.mkdir(parents=True, exist_ok=True)
    with _campaign_lock(campaign):
        if resume:
            record = json.loads((campaign / "campaign.json").read_text())
            if record.get("schema_version") != 1 or record.get("modes") != list(MODES):
                raise ValueError("Unsupported or changed campaign schema/mode order")
            base = workflow.validate_config(record["config"])
            if record.get("config_sha256") != hashlib.sha256(_canonical(base).encode()).hexdigest():
                raise ValueError("Saved campaign configuration changed")
            if config is not None:
                proposed = json.loads(_canonical(config))
                proposed["mode"] = "paper_size"
                if workflow.validate_config(proposed) != base:
                    raise ValueError("Resume settings differ from the immutable campaign configuration")
            if record.get("common_dataset") != _dataset_metadata(base, repo_root):
                raise ValueError("Common dataset metadata changed since campaign creation")
        else:
            base = new_base
            record = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                      "config": base, "config_sha256": hashlib.sha256(_canonical(base).encode()).hexdigest(),
                      "modes": list(MODES), "runs": {mode: {"path": f"{mode}/run", "state": "not_started"} for mode in MODES},
                      "common_dataset": new_dataset,
                      "state": "prepared", "current_mode": None, "completed_modes": [],
                      "test_protocol": "No automatic final test evaluation"}
            _atomic_json(campaign / "campaign.json", record)
        try:
            for mode in MODES:
                if record["common_dataset"] != _dataset_metadata(base, repo_root):
                    raise ValueError("Common dataset metadata changed between campaign models")
                entry = record["runs"][mode]
                run = _child_dir(campaign, entry["path"])
                has_checkpoint = all((run / name).is_file() for name in ("config.json", "status.json", "latest.pt"))
                if has_checkpoint:
                    _check_child_settings(run, base, mode)
                summary = run_summary(run)
                if summary["state"] in FINISHED_STATES:
                    if not has_checkpoint:
                        raise ValueError("Finished child is missing checkpoint files; preserve it for inspection")
                    entry["state"] = summary["state"]
                    if mode not in record["completed_modes"]:
                        record["completed_modes"].append(mode)
                    _atomic_json(campaign / "campaign.json", record)
                    continue
                has_config = has_checkpoint
                if run.exists() and any(run.iterdir()) and not has_config:
                    new_path, previous = _fresh_retry_path(campaign, mode, entry["path"])
                    entry.setdefault("uninitialized_attempts", []).append(previous)
                    entry["path"] = new_path
                    run = _child_dir(campaign, new_path)
                child_config = json.loads(_canonical(base))
                child_config["mode"] = mode
                record.pop("error", None)
                record.update(state="training", current_mode=mode)
                entry["state"] = "resuming" if has_config else "starting"
                _atomic_json(campaign / "campaign.json", record)
                from .mlp_terminal import training_context, training_stop_requested
                with training_context(size=base["lattice_size"], mode=mode):
                    result = workflow.train(child_config, run_dir=run, resume=has_config,
                                            interactive=interactive, repo_root=repo_root, max_chunks=max_chunks)
                returned = Path(result).resolve()
                if returned != run:
                    raise ValueError("Training returned a different campaign child directory")
                summary = run_summary(run)
                entry["state"] = summary["state"]
                if summary["state"] not in FINISHED_STATES or training_stop_requested():
                    record.update(state="paused", current_mode=mode)
                    _atomic_json(campaign / "campaign.json", record)
                    return campaign
                if mode not in record["completed_modes"]:
                    record["completed_modes"].append(mode)
                _atomic_json(campaign / "campaign.json", record)
            record.update(state="trained", current_mode=None)
            _atomic_json(campaign / "campaign.json", record)
            return campaign
        except KeyboardInterrupt:
            record.update(state="paused")
            _atomic_json(campaign / "campaign.json", record)
            return campaign
        except Exception as error:
            record.update(state="failed", error=f"{type(error).__name__}: {error}")
            _atomic_json(campaign / "campaign.json", record)
            raise


def train_both(config, *, campaign_dir=None, resume=False, interactive=True,
               repo_root=REPO_ROOT, max_chunks=None) -> Path:
    """Train both modes, propagating terminal stop requests across boundaries."""
    from .mlp_terminal import training_context
    with training_context():
        return _train_both(config, campaign_dir=campaign_dir, resume=resume,
                           interactive=interactive, repo_root=repo_root,
                           max_chunks=max_chunks)
