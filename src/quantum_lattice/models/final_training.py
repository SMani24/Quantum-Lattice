"""Resumable single-job I/O around the frozen observable training engine.

This module takes train/validation arrays from its caller. It does not choose
Hamiltonians, open a test set, select architectures, or construct a campaign.
Production calls require CUDA; the explicit CPU opt-in exists for small tests.
An atomic generation pointer commits tensors, selection, history and counters
together. Only the current and previous committed generations are retained.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import time
from typing import Any, Callable

import numpy as np
import torch

from .mlp_training import resolve_device
from .observable_cnn import ObservableMapMLP, PeriodicObservableCNN
from .observable_training import ObservableTrainSession, _array, _config

VERSION = 1
WORKFLOW = "final_observable_training"
GENERATION = re.compile(r"g\d{8}")
OUTPUTS = {"latest.pt", "best.pt", "history.json", "summary.json"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _torch(path, value):
    with Path(path).open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _sources():
    root = Path(__file__).resolve().parents[1]
    names = ["models/final_training.py", "models/observable_training.py",
             "models/observable_cnn.py", "models/mlp.py", "models/mlp_training.py"]
    return {name: _sha(root / name) for name in names}


def _runtime(device):
    value = {"python": platform.python_version(), "numpy": np.__version__,
             "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
             "device_type": device.type}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        value["accelerator"] = {"name": props.name, "total_memory": props.total_memory,
            "capability": [props.major, props.minor],
            "uuid": str(getattr(props, "uuid", "unavailable"))}
        value["cudnn"] = torch.backends.cudnn.version()
    return value


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _inference_policy(deterministic):
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:4]
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous[4:6]


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _seconds(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number or None")


@contextmanager
def _lock(job):
    with (job / ".job.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns this training job: {job}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _plan(job):
    plan = _read(job / "job-plan.json")
    if plan.get("schema_version") != VERSION or plan.get("workflow") != WORKFLOW:
        raise ValueError("Unsupported final-training job plan")
    if _read(job / "job-plan-sha256.json") != {"sha256": _digest(plan)}:
        raise ValueError("Final-training job plan checksum differs")
    return plan


def _commit(job, plan, session, spent, invocation_start, pause_reason):
    """Publish a self-contained generation after all substantial checkpoint I/O."""
    checkpoints = job / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    previous = _read(job / "state.json") if (job / "state.json").exists() else None
    number = int(previous["generation"][1:]) + 1 if previous else 1
    while (checkpoints / f"g{number:08d}").exists():
        number += 1  # A process killed before publication may leave an orphan.
    generation = f"g{number:08d}"
    directory = checkpoints / generation
    directory.mkdir()
    snapshot = session.snapshot()
    binding = _digest(plan)
    _torch(directory / "latest.pt", {"schema_version": VERSION, "workflow": WORKFLOW,
        "kind": "resumable", "plan_sha256": binding, "session": snapshot})
    _torch(directory / "best.pt", {"schema_version": VERSION, "workflow": WORKFLOW,
        "kind": "selected_best", "plan_sha256": binding, "config": session.config,
        "L": session.L, "best_epoch": session.best_epoch, "best_step": session.best_step,
        "best_validation_mse": session.best_val_mse,
        "model_state_dict": session.best_state_dict})
    _json(directory / "history.json", session.history)
    _synchronize(session.device)
    summary = {"schema_version": VERSION, "workflow": WORKFLOW, "plan_sha256": binding,
        "state": "complete" if session.finished else "paused", "L": session.L,
        "model": session.config["model"], "seed": session.config["seed"],
        "step": session.step, "epoch": session.epoch, "best_step": session.best_step,
        "best_epoch": session.best_epoch, "best_validation_mse": session.best_val_mse,
        "stop_reason": session.stop_reason if session.finished else pause_reason,
        "training_stop_reason": session.stop_reason,
        "parameter_count": sum(p.numel() for p in session.model.parameters()),
        "active_seconds": spent + time.monotonic() - invocation_start,
        "active_seconds_scope": "invocation entry through synchronized checkpoint tensor/history serialization; final small metadata/pointer writes excluded",
        "peak_cuda_allocated_bytes": max(
            torch.cuda.max_memory_allocated(session.device) if session.device.type == "cuda" else 0,
            _read(checkpoints / previous["generation"] / "summary.json").get("peak_cuda_allocated_bytes", 0)
                if previous else 0),
        "device_type": session.device.type, "raw_unclipped_predictions": True,
        "checkpoint_selection": "strictly lowest own-view validation MSE including epoch zero",
        "confirmation_payload_loaded_by_engine": False}
    _json(directory / "summary.json", summary)
    commit = {"schema_version": VERSION, "workflow": WORKFLOW, "generation": generation,
        "plan_sha256": binding, "files": {name: _sha(directory / name) for name in sorted(OUTPUTS)}}
    _json(directory / "commit.json", commit)
    _sync_directory(checkpoints)
    _json(job / "state.json", {"schema_version": VERSION, "generation": generation,
                              "commit_sha256": _sha(directory / "commit.json")})
    if session.finished:
        _json(job / "done.json", {"schema_version": VERSION, "plan_sha256": binding,
                                  "generation": generation, "commit_sha256": _sha(directory / "commit.json")})
    keep = {generation}
    if previous:
        keep.add(previous["generation"])
    for path in checkpoints.iterdir():
        if GENERATION.fullmatch(path.name) and path.name not in keep:
            if path.is_symlink():
                raise ValueError("Unsafe checkpoint-generation symlink")
            shutil.rmtree(path)
    return load_job(job)


def _read_committed(job, plan):
    """Retry if a concurrent status read straddles generation publication."""
    for _ in range(3):
        pointer = _read(job / "state.json")
        generation = pointer.get("generation", "")
        if (pointer.get("schema_version") != VERSION or not isinstance(generation, str)
                or not GENERATION.fullmatch(generation)):
            raise ValueError("Invalid checkpoint generation pointer")
        directory = job / "checkpoints" / generation
        if directory.is_symlink():
            raise ValueError("Unsafe checkpoint generation path")
        try:
            commit = _read(directory / "commit.json")
            if _sha(directory / "commit.json") != pointer.get("commit_sha256"):
                raise ValueError("Checkpoint generation manifest checksum differs")
            if (commit.get("schema_version") != VERSION or commit.get("workflow") != WORKFLOW
                    or commit.get("generation") != generation
                    or commit.get("plan_sha256") != _digest(plan)
                    or set(commit.get("files", {})) != OUTPUTS):
                raise ValueError("Checkpoint generation belongs to another job")
            for name, expected in commit["files"].items():
                if (directory / name).is_symlink() or _sha(directory / name) != expected:
                    raise ValueError(f"Checkpoint output checksum differs: {name}")
            summary = _read(directory / "summary.json")
        except FileNotFoundError:
            if pointer != _read(job / "state.json"):
                continue
            raise ValueError("Committed checkpoint generation is incomplete") from None
        if pointer != _read(job / "state.json"):
            continue
        if summary.get("plan_sha256") != _digest(plan):
            raise ValueError("Committed summary belongs to another job")
        if (job / "done.json").exists():
            expected = {"schema_version": VERSION, "plan_sha256": _digest(plan),
                        "generation": generation, "commit_sha256": _sha(directory / "commit.json")}
            if _read(job / "done.json") != expected or summary.get("state") != "complete":
                raise ValueError("Completed-job marker differs from committed state")
        return directory, commit, summary
    raise RuntimeError("Training job changed repeatedly during status read; retry")


def load_job(job_dir, *, verify=True):
    """Read committed progress without CUDA initialization or checkpoint loading.

    ``verify=False`` is intentionally unsupported: status must not silently
    treat a corrupt checkpoint as resumable or a corrupt done marker as done.
    """
    if verify is not True:
        raise ValueError("Unchecked final-training status is not supported")
    job = Path(job_dir).resolve()
    plan = _plan(job)
    if not (job / "state.json").exists():
        return {"state": "pending", "step": 0, "epoch": 0, "active_seconds": 0.0,
                "checkpoint_paths": {}, "plan_sha256": _digest(plan)}
    directory, commit, summary = _read_committed(job, plan)
    return {**summary, "checkpoint_paths": {name[:-3]: str(directory / name)
            for name in ("latest.pt", "best.pt")}, "generation": directory.name,
            "commit_sha256": _sha(directory / "commit.json")}


def checkpoint_path(job_dir, kind="best"):
    """Return the verified immutable path of the current selected/live state."""
    if kind not in {"best", "latest"}:
        raise ValueError("checkpoint kind must be best or latest")
    result = load_job(job_dir)
    if kind not in result["checkpoint_paths"]:
        raise ValueError("This job has no committed checkpoint yet")
    return Path(result["checkpoint_paths"][kind])


def train_job(train_x, train_y, val_x, val_y, settings, job_dir, bindings, *,
              stop_requested: Callable[[], bool] | None = None,
              max_seconds=None, checkpoint_updates=25, allow_cpu=False,
              max_updates=None, progress: Callable[[dict], None] | None = None):
    """Train/resume one immutable job; return a verified committed summary.

    ``max_seconds`` limits this invocation, whereas persisted ``active_seconds``
    accumulates resumes. The caller must enforce the campaign hard deadline;
    one update/epoch validation and an atomic checkpoint are indivisible units.
    Signals should set ``stop_requested`` rather than interrupting Adam midway.
    An optional ``max_updates`` is useful for bounded acceptance/timing tests.
    """
    entered = time.monotonic()
    _seconds(max_seconds, "max_seconds")
    _integer(checkpoint_updates, "checkpoint_updates")
    if max_updates is not None:
        _integer(max_updates, "max_updates", 0)
    config = _config(settings)
    if config["device"] == "cpu" and allow_cpu is not True:
        raise ValueError("Research training requires CUDA; CPU is an explicit test-only opt-in")
    if not isinstance(bindings, dict):
        raise ValueError("bindings must be a JSON object")
    bindings = json.loads(_canonical(bindings))
    arrays = [_array(x, channels, name) for x, channels, name in
              [(train_x, 3, "train_x"), (train_y, 2, "train_y"),
               (val_x, 3, "val_x"), (val_y, 2, "val_y")]]
    fingerprint = hashlib.sha256()
    for array in arrays:
        fingerprint.update(str(array.shape).encode("ascii"))
        fingerprint.update(array.tobytes())
    device = resolve_device(config["device"], config["deterministic"])
    plan = {"schema_version": VERSION, "workflow": WORKFLOW, "config": config,
            "bindings": bindings, "sources": _sources(), "runtime": _runtime(device),
            "data_fingerprint": fingerprint.hexdigest()}
    job = Path(job_dir).resolve()
    job.mkdir(parents=True, exist_ok=True)
    with _lock(job):
        if (job / "job-plan.json").exists():
            if _plan(job) != plan:
                raise ValueError("Training settings, source, runtime, data or caller bindings changed")
        else:
            if any(p.name not in {".job.lock", "job-plan.json.tmp", "job-plan-sha256.json",
                                  "job-plan-sha256.json.tmp"}
                   for p in job.iterdir()):
                raise ValueError("Refusing to initialize a nonempty unbound training directory")
            _json(job / "job-plan-sha256.json", {"sha256": _digest(plan)})
            _json(job / "job-plan.json", plan)
        saved = load_job(job)
        if saved["state"] == "complete":
            return {**saved, "invocation_seconds": time.monotonic()-entered}
        stopped = stop_requested or (lambda: False)
        if stopped() or max_updates == 0 or max_seconds is not None and time.monotonic()-entered >= max_seconds:
            return {**saved, "pause_reason": "requested" if stopped() else "budget",
                    "invocation_seconds": time.monotonic()-entered}
        payload = None
        if saved["checkpoint_paths"]:
            payload = torch.load(saved["checkpoint_paths"]["latest"], map_location="cpu", weights_only=True)
            if (payload.get("schema_version") != VERSION or payload.get("workflow") != WORKFLOW
                    or payload.get("kind") != "resumable" or payload.get("plan_sha256") != _digest(plan)):
                raise ValueError("Resumable tensor payload belongs to another job")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        session = ObservableTrainSession(*arrays, config,
                    checkpoint=payload["session"] if payload else None)
        spent = float(saved["active_seconds"])
        initial_step = session.step
        reason = None
        result = None
        while not session.finished:
            if stopped():
                reason = "requested"
                break
            if max_seconds is not None and time.monotonic()-entered >= max_seconds:
                reason = "time_budget"
                break
            remaining = max_updates-(session.step-initial_step) if max_updates is not None else checkpoint_updates
            if remaining <= 0:
                reason = "update_budget"
                break
            updates = min(checkpoint_updates, remaining)
            # Save before an epoch-ending update so stop/budget flags can be
            # checked immediately before that update's full validation pass.
            to_boundary = math.ceil((len(arrays[0])-session.cursor) /
                                    config["batch_size_hamiltonians"]) if session.cursor < len(arrays[0]) else math.ceil(len(arrays[0])/config["batch_size_hamiltonians"])
            updates = min(updates, max(1, to_boundary-1))
            session.advance(updates)
            result = _commit(job, plan, session, spent, entered, "checkpoint_boundary")
            if progress is not None:
                progress(result)
        if result is None or not session.finished:
            result = _commit(job, plan, session, spent, entered, reason)
        result["invocation_seconds"] = time.monotonic()-entered
        return result


def predict_checkpoint(checkpoint_input, x, *, device="cuda", batch_size=256,
                       allow_cpu=False, allow_size_transfer=False, allow_legacy=False):
    """Run selected weights with no optimizer or checkpoint reselection.

    New generation checkpoints and the established observable-campaign best.pt
    export are supported. A companion committed checksum is required in either
    case. Size transfer is explicit and does not imply per-size training.
    """
    _integer(batch_size, "batch_size")
    if device == "cpu" and allow_cpu is not True:
        raise ValueError("CPU inference requires explicit test/audit opt-in")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be cpu or cuda")
    path = Path(checkpoint_input).resolve()
    if path.name != "best.pt":
        raise ValueError("Inference requires an explicitly selected best.pt checkpoint")
    if (path.parent / "commit.json").exists():
        commit = _read(path.parent / "commit.json")
        if commit.get("files", {}).get("best.pt") != _sha(path):
            raise ValueError("Selected checkpoint checksum differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload.get("workflow") != WORKFLOW or payload.get("kind") != "selected_best"
                or payload.get("schema_version") != VERSION
                or payload.get("plan_sha256") != commit.get("plan_sha256")):
            raise ValueError("Unsupported selected final-training checkpoint")
        config, L = _config(payload["config"]), payload["L"]
    elif (path.parent / "done.json").exists():
        if allow_legacy is not True:
            raise ValueError("Legacy selected checkpoints require allow_legacy=True")
        done = _read(path.parent / "done.json")
        if done.get("hashes", {}).get("best.pt") != _sha(path):
            raise ValueError("Legacy selected checkpoint checksum differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        task = payload.get("task", {})
        if (payload.get("schema_version") != 1 or payload.get("plan_sha256") != done.get("plan_sha256")
                or task != done.get("task") or task.get("model") not in {"cnn", "map_mlp"}
                or type(task.get("L")) is not int or task["L"] < 3):
            raise ValueError("Unsupported legacy observable checkpoint")
        config = _config({"model": task["model"], "architecture": payload["architecture"],
                          "seed": task["seed"], "device": device})
        L = task["L"]
    else:
        raise ValueError("Selected checkpoint has no committed checksum manifest")
    values = _array(x, 3, "prediction inputs")
    if values.shape[2] != L and allow_size_transfer is not True:
        raise ValueError("Prediction lattice differs; size transfer must be explicit")
    target = resolve_device(device, config["deterministic"])
    devices = [torch.cuda.current_device()] if target.type == "cuda" else []
    with _inference_policy(config["deterministic"]), torch.random.fork_rng(devices=devices):
        model = (PeriodicObservableCNN(**config["architecture"]) if config["model"] == "cnn"
                 else ObservableMapMLP()).to(target)
        weights = payload["model_state_dict"]
        expected = model.state_dict()
        if set(weights) != set(expected) or any(not isinstance(weights[k], torch.Tensor)
                or weights[k].shape != expected[k].shape or weights[k].dtype != expected[k].dtype
                or not bool(torch.isfinite(weights[k]).all().item()) for k in weights):
            raise ValueError("Selected weights are incompatible or nonfinite")
        model.load_state_dict(weights)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                result = model(torch.from_numpy(values[start:start+batch_size].copy()).to(target))
                if not bool(torch.isfinite(result).all().item()):
                    raise FloatingPointError("Nonfinite selected-model predictions")
                predictions.append(result.cpu().numpy().copy())
        _synchronize(target)
    return np.concatenate(predictions, axis=0)
