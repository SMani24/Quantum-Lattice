"""Disk workflow for the size-specific magnitude baseline.

Training opens no test payload. The separate evaluation operation uses a frozen
validation-selected model. All artifacts stay inside this phase's ignored root.
"""

from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from quantum_lattice.data.mlp_data import load_learning_data
from quantum_lattice.metrics.regression import regression_report, per_hamiltonian_metrics
from quantum_lattice.models.mlp_training import TrainSession
from quantum_lattice.models.mlp_terminal import TrainingConsole
from quantum_lattice.plotting.mlp import plot_learning_history, plot_test_predictions


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_PATH = Path("codex material/artifacts/phase1_mlp")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read(path):
    return json.loads(path.read_text())


def artifact_root(repo_root=REPO_ROOT):
    return (Path(repo_root) / ARTIFACT_PATH).resolve()


def _expose(run, repo_root):
    root = artifact_root(repo_root)
    shortcut = Path(repo_root) / "experiments/baseline_mlp/results"
    shortcut.parent.mkdir(parents=True, exist_ok=True)
    if not shortcut.exists() and not shortcut.is_symlink():
        shortcut.symlink_to(os.path.relpath(root, shortcut.parent), target_is_directory=True)
    latest = root / "latest"
    if latest.exists() and not latest.is_symlink():
        raise ValueError("The phase latest path must be a shortcut, not a real directory")
    temporary = root / ".latest.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(run.relative_to(root), target_is_directory=True)
    os.replace(temporary, latest)


def _safe_run_dir(path, repo_root):
    root = artifact_root(repo_root)
    run = Path(path).resolve()
    if run == root or not run.is_relative_to(root):
        raise ValueError(f"MLP runs must be below {root}")
    return run


@contextmanager
def _lock(run):
    import fcntl
    with (run / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This MLP run is already active in another process") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _source_hashes():
    paths = [Path(__file__), REPO_ROOT / "src/quantum_lattice/models/mlp.py",
             REPO_ROOT / "src/quantum_lattice/models/mlp_training.py",
             REPO_ROOT / "src/quantum_lattice/data/mlp_data.py",
             REPO_ROOT / "src/quantum_lattice/metrics/regression.py"]
    return {str(path.relative_to(REPO_ROOT)): _sha(path) for path in paths}


def _environment():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "machine": platform.machine(),
            "platform": platform.platform()}


def validate_config(config):
    value = json.loads(_canonical(config))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported MLP configuration schema")
    if value.get("mode") not in {"paper_size", "all_edges"}:
        raise ValueError("mode must be paper_size or all_edges")
    for key in ("lattice_size", "training_hamiltonians"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    edge = value.get("fixed_edge")
    if isinstance(edge, bool) or not isinstance(edge, int) or edge < 0:
        raise ValueError("fixed_edge must be a nonnegative integer")
    if not isinstance(value.get("dataset_path"), str) or not value["dataset_path"]:
        raise ValueError("dataset_path is required")
    chunk = value.get("execution", {}).get("checkpoint_steps")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("checkpoint_steps must be a positive integer")
    evaluation = value.get("evaluation", {})
    for key in ("bootstrap_samples", "seed"):
        if isinstance(evaluation.get(key), bool) or not isinstance(evaluation.get(key), int) or evaluation[key] < 0:
            raise ValueError(f"evaluation.{key} must be a nonnegative integer")
    if not isinstance(value.get("training"), dict):
        raise ValueError("training settings are required")
    return value


def _load_data(config, repo_root, *, test=False):
    path = Path(config["dataset_path"])
    if not path.is_absolute():
        path = Path(repo_root) / path
    data = load_learning_data(path, mode=config["mode"], fixed_edge=config["fixed_edge"],
                              training_hamiltonians=config["training_hamiltonians"],
                              include_test=test)
    if data.L != config["lattice_size"]:
        raise ValueError("Configured lattice size differs from the saved dataset")
    return data


def dataset_readiness(config, repo_root=REPO_ROOT):
    """Cheap menu check; do not open any NPZ data here."""
    path = Path(config["dataset_path"])
    if not path.is_absolute():
        path = Path(repo_root) / path
    if not (path / "status.json").exists():
        return {"state": "not_started", "dataset_path": str(path)}
    status = _read(path / "status.json")
    if status.get("state") == "complete":
        if not (path / "verification.json").exists() or not _read(path / "verification.json").get("passed"):
            status["state"] = "unverified"
    return {**status, "dataset_path": str(path)}


def _save(run, session, history, manifest, state, elapsed):
    payload = {"workflow_schema_version": 1, "manifest": manifest,
               "session": session.snapshot(), "history": history,
               "elapsed_seconds": elapsed}
    _atomic_torch(run / "latest.pt", payload)
    _atomic_json(run / "history.json", history)
    event = session.last_validation or {}
    status = {"state": state, "step": session.step,
              "max_steps": session.config["max_steps"], "epoch": session.epoch,
              "best_step": session.best_step,
              "best_validation_mse": session.best_val_mse if math.isfinite(session.best_val_mse) else None,
              "train_mse": event.get("train_mse"), "validation_mse": event.get("validation_mse"),
              "device": str(session.device), "elapsed_seconds": elapsed,
              "latest_sha256": _sha(run / "latest.pt"), "test_evaluated": False}
    status["training_examples_seen"] = max(0, session.epoch - 1) * len(session.train_x) + session.cursor
    status["training_row_passes"] = status["training_examples_seen"] / len(session.train_x)
    _atomic_json(run / "status.json", status)
    return status


def _restore(run, config, data):
    payload = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    manifest = payload["manifest"]
    if payload.get("workflow_schema_version") != 1:
        raise ValueError("Unsupported MLP checkpoint schema")
    if manifest["config_sha256"] != hashlib.sha256(_canonical(config).encode()).hexdigest():
        raise ValueError("Stored run configuration changed")
    if manifest["source_hashes"] != _source_hashes():
        raise ValueError("Training source changed; retain this run and start a new one")
    if manifest["environment"] != _environment():
        raise ValueError("Training environment changed; exact continuation requires the saved environment")
    if manifest["dataset_fingerprint"] != data.provenance["source_fingerprint"]:
        raise ValueError("Learning dataset or row selection changed")
    session = TrainSession(data.train.x, data.train.y, data.validation.x, data.validation.y,
                           config["training"], checkpoint=payload["session"])
    return session, payload["history"], manifest, payload["elapsed_seconds"]


def _report(run, config, manifest, status, metrics=None):
    counts = manifest["row_counts"]
    text = f"""# MLP run results

Declared scope: {config.get('provenance', {}).get('scope', 'Magnitude baseline on the declared dataset')}.

L={config['lattice_size']}, mode `{config['mode']}`. The paper architecture
3→128→150→128→1 predicts |b_z| from (mean A_u, mean A_v, mean A_u A_v).
These inputs are known-field simulation estimators, not physical measurement shots.

Saved state: **{status['state']}**, optimizer update {status['step']}/{status['max_steps']}.
Training examples processed: {status['training_examples_seen']}; equivalent full row passes: {status['training_row_passes']:.4f}.
Training/validation/final-test row counts: {counts['train']}/{counts['validation']}/{counts['test']}.
Best validation MSE: {status['best_validation_mse']}; selected update: {status['best_step']}.
Training uses Adam, shuffled batches of {config['training']['batch_size']},
learning rate {config['training']['learning_rate']}. Batch size, update budget,
validation cadence and Hamiltonian splits are our documented choices.
PyTorch is an architecture port of the supplied TensorFlow graph, not a numerical replay.

All edges of a Hamiltonian stay in one existing split. `paper_size` uses one fixed
edge per selected training/validation instance; `all_edges` uses all their edges.
Both evaluate the same held-out bank, all edges. No clipping or normalization is applied.
No historical phase filter is recovered. Storage checks do not certify chain mixing.

Files: config.json (effective settings), manifest.json (source/environment/data hashes),
data_selection.json (split identities/counts), latest.pt (resumable Adam/RNG checkpoint),
history.json (training/validation curve), status.json (progress),
best_model.pt (frozen validation-selected weights after training), figures/ (PNG/PDF).
After explicit evaluation: test_metrics.json, test_predictions.npz and .csv,
per_hamiltonian_metrics.csv, evaluation_provenance.json and prediction/error figures.
Raw sampler data stays in the separate sample-generation run.
"""
    if metrics is None:
        text += "\nFinal test payloads have not been opened by this training run. No final-test accuracy is claimed.\n"
    else:
        summary = {key: metrics[key] for key in ("count", "hamiltonians", "mse", "rmse", "mae", "bootstrap",
                                                "constant_training_mean_baseline")}
        text += f"\nFinal-test metrics (raw linear predictions):\n\n```json\n{json.dumps(summary, indent=2)}\n```\n"
    (run / "RESULTS.md").write_text(text)


def train(config=None, *, run_dir=None, resume=False, interactive=True,
          repo_root=REPO_ROOT, max_chunks=None):
    """Run/resume bounded optimizer chunks; final test is a separate operation.

    Ctrl+C discards the incomplete chunk and retains the last atomic checkpoint.
    ``max_chunks`` is a test/smoke limit, not a change to the scientific budget.
    """
    if max_chunks is not None and (isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 1):
        raise ValueError("max_chunks must be a positive integer")
    if resume:
        if run_dir is None:
            raise ValueError("resume requires a saved run directory")
        run = _safe_run_dir(run_dir, repo_root)
        config = validate_config(_read(run / "config.json"))
    else:
        config = validate_config(config)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = _safe_run_dir(run_dir or artifact_root(repo_root) / f"{name}-L{config['lattice_size']}-{config['mode']}", repo_root)
        if run.exists() and any(run.iterdir()):
            raise ValueError("New run directory must be empty; use resume for an existing run")
        run.mkdir(parents=True, exist_ok=True)
    with _lock(run):
        if resume and _read(run / "status.json")["state"] in {"trained", "evaluated", "evaluating"}:
            raise ValueError("Training already finished; use evaluate or start a new run")
        print("Verifying training/validation summaries and checksums; test payloads stay closed.", flush=True)
        data = _load_data(config, repo_root)
        if resume:
            session, history, manifest, previous_elapsed = _restore(run, config, data)
        else:
            session = TrainSession(data.train.x, data.train.y, data.validation.x, data.validation.y, config["training"])
            # Store normalized engine defaults, rather than hiding implicit values.
            config["training"] = session.config
            manifest = {"schema_version": 1, "config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest(),
                        "source_hashes": _source_hashes(), "environment": _environment(),
                        "dataset_fingerprint": data.provenance["source_fingerprint"],
                        "row_counts": data.provenance["row_counts"],
                        "hamiltonian_counts": data.provenance["hamiltonian_counts"],
                        "device": str(session.device), "architecture": [3, 128, 150, 128, 1]}
            _atomic_json(run / "config.json", config)
            _atomic_json(run / "manifest.json", manifest)
            _atomic_json(run / "data_selection.json", data.provenance)
            history, previous_elapsed = [], 0.0
            _save(run, session, history, manifest, "prepared", previous_elapsed)
        started = time.monotonic()
        chunks, paused, stopped = 0, False, False
        with TrainingConsole(session.config["max_steps"], interactive=interactive) as console:
            try:
                while not session.finished:
                    command = console.poll()
                    if command == "quit":
                        stopped = True
                        break
                    if command == "pause":
                        paused = True
                    if command == "resume":
                        paused = False
                    elapsed = previous_elapsed + time.monotonic() - started
                    if paused:
                        status = _save(run, session, history, manifest, "paused", elapsed)
                        console.render(status)
                        # Poll without occupying a CPU core while paused.
                        while paused:
                            time.sleep(0.1)
                            command = console.poll()
                            if command == "resume":
                                paused = False
                            elif command == "quit":
                                stopped = True
                                break
                        if stopped:
                            break
                        continue
                    history.extend(session.advance(config["execution"]["checkpoint_steps"]))
                    chunks += 1
                    elapsed = previous_elapsed + time.monotonic() - started
                    status = _save(run, session, history, manifest, "training", elapsed)
                    console.render(status)
                    if max_chunks is not None and chunks >= max_chunks and not session.finished:
                        stopped = True
                        break
            except KeyboardInterrupt:
                # An interrupt can land between Adam.step and the progress counter.
                # The last complete chunk is authoritative, never a half-updated snapshot.
                session, history, manifest, _ = _restore(run, config, data)
                stopped = True
            elapsed = previous_elapsed + time.monotonic() - started
            state = "paused" if stopped else "training"
            status = _save(run, session, history, manifest, state, elapsed)
        if not stopped:
            _atomic_torch(run / "best_model.pt", {"model_state_dict": session.best_state_dict,
                          "best_step": session.best_step, "best_validation_mse": session.best_val_mse,
                          "dataset_fingerprint": manifest["dataset_fingerprint"], "architecture": manifest["architecture"]})
            status.update(state="trained", best_model_sha256=_sha(run / "best_model.pt"))
        if history:
            plot_learning_history(history, run / "figures")
        _report(run, config, manifest, status)
        # Publish final completion only after all training handoff files exist.
        _atomic_json(run / "status.json", status)
        _expose(run, repo_root)
        print(f"Saved {status['state'].upper()} run: {run}", flush=True)
        return run


def evaluate(run_dir, *, repo_root=REPO_ROOT):
    """Open final test exactly after a finished, frozen training run."""
    run = _safe_run_dir(run_dir, repo_root)
    with _lock(run):
        status = _read(run / "status.json")
        if status["state"] not in {"trained", "evaluated", "evaluating"}:
            raise ValueError("Finish training before opening the final test set")
        config = validate_config(_read(run / "config.json"))
        if _sha(run / "best_model.pt") != status.get("best_model_sha256"):
            raise ValueError("Frozen validation-selected model checksum changed")
        # Verify training identity before any final-test payload is read.
        training_data = _load_data(config, repo_root)
        session, _, manifest, _ = _restore(run, config, training_data)
        if not session.finished:
            raise ValueError("Unfinished training checkpoint cannot be evaluated")
        frozen = torch.load(run / "best_model.pt", map_location="cpu", weights_only=True)
        if frozen["best_step"] != session.best_step or any(
                not torch.equal(value, session.best_state_dict[key])
                for key, value in frozen["model_state_dict"].items()):
            raise ValueError("Frozen model differs from validation-selected checkpoint")
        if status["state"] == "evaluated":
            return _read(run / "test_metrics.json")
        status["state"] = "evaluating"
        _atomic_json(run / "status.json", status)
        print("Evaluating frozen weights on all held-out Hamiltonians and edges.", flush=True)
        data = _load_data(config, repo_root, test=True)
        if data.provenance["source_fingerprint"] != manifest["dataset_fingerprint"]:
            raise ValueError("Dataset changed during final evaluation")
        rows = data.test
        prediction = session.predict(rows.x, use_best=True)
        metrics = regression_report(prediction, rows.y, rows.hamiltonian_ids, rows.edge_ids,
                                    fixed_edge=config["fixed_edge"],
                                    training_target_mean=float(np.mean(data.train.y, dtype=np.float64)),
                                    **config["evaluation"])
        _atomic_json(run / "test_metrics.json", metrics)
        np.savez_compressed(run / "test_predictions.npz", prediction=prediction, target=rows.y,
                            hamiltonian_ids=rows.hamiltonian_ids, hamiltonian_uids=rows.hamiltonian_uids,
                            edge_ids=rows.edge_ids)
        with (run / "test_predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["hamiltonian_id", "hamiltonian_uid", "edge_id", "target_abs_b", "prediction", "error"])
            for name, uid, edge, target, pred in zip(rows.hamiltonian_ids, rows.hamiltonian_uids,
                                                    rows.edge_ids, rows.y[:, 0], prediction[:, 0]):
                writer.writerow([name, uid, int(edge), float(target), float(pred), float(pred - target)])
        groups = per_hamiltonian_metrics(prediction, rows.y, rows.hamiltonian_ids, rows.edge_ids)
        with (run / "per_hamiltonian_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
        plot_test_predictions(prediction, rows.y, run / "figures")
        provenance = {"schema_version": 1, "dataset_fingerprint": manifest["dataset_fingerprint"],
                      "best_model_sha256": status["best_model_sha256"],
                      "selected_step": session.best_step, "test_rows": len(rows),
                      "test_hamiltonians": len(set(rows.hamiltonian_ids)),
                      "test_payload_checksums_verified": True,
                      "test_protocol": "all edges; Hamiltonian-level bootstrap; no clipping",
                      "source_hashes": _source_hashes(), "environment": _environment()}
        _atomic_json(run / "evaluation_provenance.json", provenance)
        status.update(state="evaluated", test_evaluated=True)
        _atomic_json(run / "status.json", status)
        _report(run, config, manifest, status, metrics)
        _expose(run, repo_root)
        return metrics
