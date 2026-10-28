"""Fixed-method, multi-size learning and sealed fresh-H confirmation campaign.

Training uses only the immutable non-test feature cache. A separate transition
binds every selected checkpoint before the confirmation manifest is generated.
All stages are resumable; a cumulative active-time ledger spans invocations.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from quantum_lattice.data.final_features import load_features
from quantum_lattice.models.final_training import train_job, load_job, checkpoint_path, predict_checkpoint
from quantum_lattice.models.final_evaluation import evaluate_predictions, load_evaluation, compare_seed_panels

VERSION = 1
SIZES = (3, 4, 8, 12, 16, 20, 24)
MODELS = ("cnn", "map_mlp")
VIEWS = ("f", "g")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)


def safe(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path escapes the repository")
    return path


def artifact(root, relative):
    path = safe(root, relative)
    allowed = (Path(root) / "codex material/artifacts/final_benchmark").resolve()
    if path == allowed or not path.is_relative_to(allowed):
        raise ValueError("Final artifacts must be in their own ignored final_benchmark subtree")
    return path


def validate_config(config):
    from quantum_lattice.models.observable_training import _config
    c = json.loads(canonical(config))
    keys = {"schema_version", "sizes", "size_order", "seeds", "feature_cache", "output_root",
            "prior_manifest_root", "architecture", "training", "confirmation", "statistics",
            "execution", "hardware", "raw_panel", "provenance"}
    if set(c) != keys or c["schema_version"] != VERSION:
        raise ValueError("Unsupported or incomplete final campaign configuration")
    if c["sizes"] != list(SIZES) or sorted(c["size_order"]) != list(SIZES):
        raise ValueError("This final study must cover all seven declared learning sizes")
    if (not isinstance(c["seeds"], list) or not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"])
            or any(type(s) is not int or not 0 <= s < 2**63 for s in c["seeds"])):
        raise ValueError("Paired seeds must be unique nonnegative integers")
    t = _config({**c["training"], "architecture": c["architecture"], "model": "cnn", "seed": c["seeds"][0]})
    if t["device"] != "cuda":
        raise ValueError("Production training requires CUDA; no CPU fallback")
    if t["max_epochs"] != 500 or t["batch_size_hamiltonians"] != 32:
        raise ValueError("Keep the fixed 500-epoch / 32-H policy; revise the protocol explicitly for a different study")
    c["architecture"] = t.pop("architecture")
    t.pop("model"); t.pop("seed")
    c["training"] = t
    ex = c["execution"]
    if set(ex) != {"workers", "checkpoint_updates", "max_active_hours", "science_active_hours", "preflight_active_seconds"}:
        raise ValueError("Execution controls must be explicit")
    for name in ("workers", "checkpoint_updates"):
        if type(ex[name]) is not int or ex[name] < 1:
            raise ValueError(f"Invalid {name}")
    if ex["workers"] > 10:
        raise ValueError("Worker count exceeds the qualified ten online logical CPUs")
    for name in ("science_active_hours", "max_active_hours", "preflight_active_seconds"):
        if type(ex[name]) not in (int, float) or not math.isfinite(ex[name]):
            raise ValueError(f"{name} must be finite real numbers, not booleans")
    if not 0 < ex["science_active_hours"] < ex["max_active_hours"] <= 96:
        raise ValueError("The hard cumulative allowance is at most 96 active hours")
    if not 0 <= ex["preflight_active_seconds"] < ex["science_active_hours"] * 3600:
        raise ValueError("Invalid preflight time charge")
    conf = c["confirmation"]
    if set(conf) != {"count_per_size", "seed", "sampling", "diagnostics"}:
        raise ValueError("Confirmation controls must be explicit")
    if type(conf["count_per_size"]) is not int or conf["count_per_size"] < 32:
        raise ValueError("Declare at least 32 fresh Hamiltonians per size")
    if type(conf["seed"]) is not int or conf["seed"] < 0:
        raise ValueError("Invalid confirmation RNG namespace")
    from quantum_lattice.data.final_confirmation import _sampling
    conf["sampling"] = _sampling(conf["sampling"])
    diagnostics = conf["diagnostics"]
    if (not isinstance(diagnostics, dict) or set(diagnostics) != {"enabled", "hamiltonians_per_size", "replicas"}
            or type(diagnostics["enabled"]) is not bool
            or type(diagnostics["hamiltonians_per_size"]) is not int
            or not 1 <= diagnostics["hamiltonians_per_size"] <= conf["count_per_size"]
            or type(diagnostics["replicas"]) is not int or not 4 <= diagnostics["replicas"] <= 8):
        raise ValueError("Declare a bounded four-or-more-chain diagnostic panel")
    from quantum_lattice.models.final_worker import _settings
    c["hardware"] = _settings(c["hardware"])
    if c["statistics"] != {"bootstrap_samples": 10000, "seed": 2026102891,
                            "primary": "L4 CNN g minus f own-input magnitude MSE",
                            "other_size_contrasts": "prespecified secondary; no selection"}:
        raise ValueError("Keep the declared Hamiltonian-grouped uncertainty policy")
    for name in ("feature_cache", "output_root", "prior_manifest_root"):
        if not isinstance(c[name], str) or Path(c[name]).is_absolute():
            raise ValueError(f"{name} must be repository relative")
    if not isinstance(c["raw_panel"], dict) or type(c["raw_panel"].get("enabled")) is not bool:
        raise ValueError("Declare the separate raw-input control panel")
    raw = c["raw_panel"]
    raw_keys = {"enabled", "source_root", "size", "records", "record_seed", "seeds", "methods", "ridge", "max_iterations", "tolerance"}
    if (set(raw) != raw_keys or raw["size"] != 4 or raw["records"] != [256, 64]
            or raw["methods"] != ["mean_cnn", "set_cnn"] or not raw["seeds"]
            or len(set(raw["seeds"])) != len(raw["seeds"])
            or any(type(s) is not int or s < 0 for s in raw["seeds"])
            or type(raw["record_seed"]) is not int or raw["record_seed"] < 0
            or not isinstance(raw["source_root"], str) or Path(raw["source_root"]).is_absolute()
            or type(raw["max_iterations"]) is not int or raw["max_iterations"] < 1
            or type(raw["ridge"]) not in (int,float) or not math.isfinite(raw["ridge"]) or raw["ridge"] <= 0
            or type(raw["tolerance"]) not in (int,float) or not math.isfinite(raw["tolerance"]) or raw["tolerance"] <= 0):
        raise ValueError("Raw-control methods/budgets/solver settings are incomplete or invalid")
    return c


def tasks(config):
    return [{"L": L, "seed": seed, "model": model, "view": view,
             "key": f"L{L}-seed{seed}-{model}-{view}"}
            for L in config["size_order"] for seed in config["seeds"]
            for model in MODELS for view in VIEWS]


def sources(root):
    root = Path(root)
    files = list((root / "src/quantum_lattice").rglob("*.py"))
    files += list((root / "experiments/final_benchmark").glob("*.py"))
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def feature_hashes(root, config):
    cache = artifact(root, config["feature_cache"])
    names = ["request.json", "manifest.json"]
    names += [f"L{L}/{name}" for L in config["sizes"] for name in ("manifest.json", "maps.npz")]
    return {str((cache / name).relative_to(root)): sha(cache / name) for name in names}


def _input_metadata(root, config):
    base = artifact(root, config["prior_manifest_root"])
    paths = [base / f"L{L}.json" for L in config["sizes"]]
    pins = {str(p.relative_to(root)): sha(p) for p in paths}
    raw = config["raw_panel"]
    if raw["enabled"]:
        raw_root = artifact(root, raw["source_root"])
        manifest = read(raw_root / "manifest.json")
        pins[str((raw_root / "manifest.json").relative_to(root))] = sha(raw_root / "manifest.json")
        for name, expected in manifest["files"].items():
            p = safe(raw_root, name)
            if sha(p) != expected:
                raise ValueError(f"Raw reference changed: {name}")
            pins[str(p.relative_to(root))] = expected
    return pins


@contextmanager
def campaign_lock(run):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".campaign.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This campaign is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(root, config):
    """Pin all seven development banks and tasks, without drawing confirmation H."""
    root, config = Path(root).resolve(), validate_config(config)
    run = artifact(root, config["output_root"])
    with campaign_lock(run):
        if (run / "plan.json").exists():
            old = load_plan(root, run)
            if old["config"] != config:
                raise ValueError("Existing campaign settings differ; use a new output path")
            return run
        if any(p.name not in {".campaign.lock", "PAUSE"} for p in run.iterdir()):
            raise ValueError("Refusing a nonempty unbound campaign directory")
        pinned_features = feature_hashes(root, config)
        cache = artifact(root, config["feature_cache"])
        groups = {}
        for L in config["sizes"]:
            data = load_features(cache, L)
            counts = {split: len(data[split]["y"]) for split in ("train", "validation")}
            if counts != {"train": 7452, "validation": 1242}:
                raise ValueError("Production uses the complete 7,452/1,242 non-test H groups")
            if not all(view in data[split] for split in counts for view in VIEWS):
                raise ValueError("Missing paired original/conditioned maps")
            if set(data["train"]["uids"]) & set(data["validation"]["uids"]):
                raise ValueError("Hamiltonian identity leakage")
            groups[str(L)] = {"counts": counts, "identity_sha256": {
                split: digest(data[split]["uids"].tolist()) for split in counts}}
            del data
        plan = {"schema_version": VERSION, "config": config, "tasks": tasks(config),
                "sources": sources(root), "features": pinned_features,
                "input_metadata": _input_metadata(root, config), "groups": groups,
                "scope": "known-field numerical estimator recovery; grouped development only until a later sealed confirmation transition",
                "primary": config["statistics"]["primary"],
                "policy": "all four arms, all seven sizes, fixed policy and paired seeds; no architecture search",
                "confirmation_payload_loaded": False}
        if feature_hashes(root, config) != pinned_features:
            raise ValueError("Features changed during preparation")
        atomic_json(run / "plan-sha256.json", {"sha256": digest(plan)})
        atomic_json(run / "plan.json", plan)
        atomic_json(run / "clock.json", {"active_seconds": config["execution"]["preflight_active_seconds"],
                    "inflight": False, "stage": "prepared", "last_tick_epoch": time.time()})
        write_status(run, plan)
    return run


def load_plan(root, run, *, verify_data=True):
    root, run = Path(root).resolve(), artifact(root, run)
    plan = read(run / "plan.json")
    if read(run / "plan-sha256.json") != {"sha256": digest(plan)}:
        raise ValueError("Campaign plan checksum differs")
    if validate_config(plan["config"]) != plan["config"] or plan["tasks"] != tasks(plan["config"]):
        raise ValueError("Frozen task/configuration list differs")
    if sources(root) != plan["sources"]:
        raise ValueError("Source changed after freeze; do not alter a running campaign")
    if verify_data:
        for name, expected in {**plan["features"], **plan["input_metadata"]}.items():
            if sha(safe(root, name)) != expected:
                raise ValueError(f"Frozen input changed: {name}")
    return plan


class Budget:
    """Cumulative active wall time, conservatively charging an unclean exit gap."""
    def __init__(self, run, execution, *, clock=time):
        self.run, self.clock, self.execution = Path(run), clock, execution
        old = read(self.run / "clock.json")
        if (not isinstance(old, dict) or not {"active_seconds", "inflight", "stage", "last_tick_epoch"}.issubset(old)
                or type(old["active_seconds"]) not in (int,float) or not math.isfinite(old["active_seconds"])
                or old["active_seconds"] < 0 or type(old["inflight"]) is not bool
                or type(old["last_tick_epoch"]) not in (int,float) or not math.isfinite(old["last_tick_epoch"])
                or old["last_tick_epoch"] < 0 or not isinstance(old["stage"], str)):
            raise ValueError("Corrupt cumulative runtime ledger")
        self.base = float(old["active_seconds"])
        self.unclean_charge = 0.0
        if old.get("inflight"):
            self.unclean_charge = max(0.0, clock.time() - old["last_tick_epoch"])
            self.base += self.unclean_charge
        self.started = clock.monotonic()
        self.stage = "starting"
        self.finalizing = False
        self.tick()

    @property
    def active_seconds(self):
        return self.base + self.clock.monotonic() - self.started

    @property
    def remaining(self):
        limit = (self.execution["max_active_hours"] * 3600 - 330 if self.finalizing
                 else self.execution["science_active_hours"] * 3600)
        return max(0.0, limit - self.active_seconds)

    def tick(self, stage=None, *, done=False):
        if stage is not None:
            self.stage = stage
        atomic_json(self.run / "clock.json", {"active_seconds": self.active_seconds,
            "inflight": not done, "stage": self.stage, "last_tick_epoch": self.clock.time(),
            "hard_limit_seconds": self.execution["max_active_hours"] * 3600,
            "science_limit_seconds": self.execution["science_active_hours"] * 3600,
            "unclean_restart_charge_seconds": self.unclean_charge,
            "unclean_gap_policy": "conservative wall gap charge; no automatic budget extension"})


def write_status(run, plan, stage=None, *, persist=True):
    rows = []
    for task in plan["tasks"]:
        job = Path(run) / "models" / task["key"]
        value = load_job(job) if (job / "job-plan.json").exists() else {"state": "pending"}
        rows.append({**task, **{k: value.get(k) for k in ("state", "epoch", "step", "best_epoch", "best_validation_mse", "active_seconds", "stop_reason")}})
    clock = read(Path(run) / "clock.json")
    value = {"schema_version": 1, "stage": stage or clock["stage"],
             "completed": sum(r["state"] == "complete" for r in rows), "total": len(rows),
             "tasks": rows, "clock": clock, "confirmation_sealed": (Path(run)/"confirmation-seal.json").exists(),
             "confirmation_complete": (Path(run)/"complete.json").exists(),
             "scope": "development metrics select weights; fresh confirmation is later and inference-only"}
    if persist:
        atomic_json(Path(run) / "status.json", value)
    return value


def format_status(status):
    hours = status["clock"]["active_seconds"] / 3600
    lines = [f"Final campaign: {status['completed']}/{status['total']} training jobs complete",
             f"Stage: {status['stage']} · charged active time {hours:.2f} h / 96 h maximum"]
    for L in SIZES:
        rows = [r for r in status["tasks"] if r["L"] == L]
        lines.append(f"  L={L:2}: {sum(r['state']=='complete' for r in rows):2}/{len(rows)} complete")
    for row in status["tasks"]:
        if row["state"] not in ("pending", "complete"):
            lines.append(f"  {row['key']}: epoch {row.get('epoch')}, best MSE {row.get('best_validation_mse')}")
    lines.append("Confirmation: " + ("complete" if status["confirmation_complete"] else "sealed" if status["confirmation_sealed"] else "not yet accessed"))
    return "\n".join(lines)


def _seal(root, run, plan):
    """The first scientific confirmation transition requires every frozen job."""
    models = {}
    for task in plan["tasks"]:
        job = run / "models" / task["key"]
        value = load_job(job)
        if value["state"] != "complete":
            raise ValueError("Cannot open confirmation before every model is complete")
        path = Path(checkpoint_path(job, "best"))
        models[str(path.relative_to(root))] = sha(path)
    raw = plan["config"]["raw_panel"]
    if raw["enabled"]:
        base = artifact(root, raw["source_root"])
        for name, expected in read(base / "manifest.json")["files"].items():
            if name.endswith("/best.pt"):
                models[str((base/name).relative_to(root))] = expected
    seal = {"schema_version": 1, "plan_sha256": digest(plan), "source_sha256": plan["sources"],
            "model_sha256": models, "confirmation": plan["config"]["confirmation"],
            "statistics": plan["config"]["statistics"], "old_tests_are_untouched": False,
            "role": "fresh_confirmation", "prohibits_reselection": True}
    path = run / "confirmation-seal.json"
    if path.exists() and read(path) != seal:
        raise ValueError("Selected models/confirmation seal changed")
    atomic_json(path, seal)
    return seal


def verify_completed(run, plan, *, deep=True):
    """Verify completion and optionally every immutable scientific payload."""
    run = Path(run)
    complete = read(run/"complete.json")
    if (complete.get("plan_sha256") != digest(plan)
            or complete.get("integrity_sha256") != sha(run/"integrity.json")
            or complete.get("confirmation_seal_sha256") != digest(read(run/"confirmation-seal.json"))):
        raise ValueError("Completion, inventory or model seal changed")
    inventory = read(run/"integrity.json")
    if inventory.get("plan_sha256") != digest(plan) or not inventory.get("files"):
        raise ValueError("Scientific inventory belongs to a different or empty campaign")
    if deep:
        for name, expected in inventory["files"].items():
            path = safe(run, name)
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f"Completed scientific artifact changed: {name}")
    return {"verified": True, "deep": deep, "scientific_files": len(inventory["files"])}


def request_pause(root, config):
    run = artifact(root, validate_config(config)["output_root"])
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "PAUSE", {"requested_epoch": time.time(), "action": "atomic pause; preserve all checkpoints"})
    return run / "PAUSE"


def _evaluate_task(root, run, plan, task, data, role, *, exact=False):
    selected = Path(checkpoint_path(run / "models" / task["key"], "best"))
    view = "exact" if exact else task["view"]
    inputs = data.get("exact" if role == "validation" else "x_exact") if exact else data[view if role == "validation" else "x_"+view]
    bindings = {"plan_sha256": digest(plan), "task": task, "role": role,
                "input_view": view, "checkpoint_sha256": sha(selected),
                "known_fields_used_for_observable_preprocessing": True,
                "selection": "own-view development validation only; no confirmation reselection"}
    output = run / "evaluations" / role / (task["key"] + ("-exact" if exact else ""))
    if (output / "done.json").exists():
        return load_evaluation(output, bindings)
    prediction = predict_checkpoint(selected, inputs, device="cuda", batch_size=plan["config"]["training"]["validation_batch_size"])
    ids = data["ids"] if "ids" in data else data["hamiltonian_ids"]
    uids = data["uids"] if "uids" in data else data["hamiltonian_uids"]
    return evaluate_predictions(task["L"], prediction, data["fields"], ids, uids, output, bindings)


def run_campaign(root, run, *, workers=None, resume=False):
    """Run one complete queue, stopping safely on user request or active budget."""
    from quantum_lattice.data.final_confirmation import prepare_confirmation, run_confirmation, load_confirmation_features
    root, run = Path(root).resolve(), artifact(root, run)
    with campaign_lock(run):
        plan = load_plan(root, run)
        config = plan["config"]
        if (run/"complete.json").exists():
            verify_completed(run, plan)
            return write_status(run, plan, "complete")
        workers = config["execution"]["workers"] if workers is None else workers
        if type(workers) is not int or not 1 <= workers <= 10:
            raise ValueError("Choose 1–10 CPU workers")
        pause = run / "PAUSE"
        if pause.exists():
            if not resume:
                raise ValueError("Pause is pending; explicitly resume to clear it")
            pause.unlink()
        budget = Budget(run, config["execution"])
        interrupted = [False]
        def on_signal(signum, frame):
            interrupted[0] = True
        previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        def stop():
            return interrupted[0] or pause.exists() or budget.remaining <= 0
        try:
            if (run/"scientific-work-complete.json").exists():
                recorded = read(run/"scientific-work-complete.json")
                seal = _seal(root, run, plan)
                if recorded != {"plan_sha256": digest(plan), "confirmation_seal_sha256": digest(seal)}:
                    raise ValueError("Scientific completion/seal changed")
                return _finish_reports(root, run, plan, seal, budget, stop)
            cache = artifact(root, config["feature_cache"])
            data, previous_L = None, None
            budget.tick("training")
            for task in plan["tasks"]:
                if stop():
                    break
                job = run / "models" / task["key"]
                if (job / "job-plan.json").exists() and load_job(job)["state"] == "complete":
                    continue
                if previous_L != task["L"]:
                    data = load_features(cache, task["L"])
                    previous_L = task["L"]
                settings = {**config["training"], "architecture": config["architecture"],
                            "model": task["model"], "seed": task["seed"]}
                bindings = {"campaign_plan_sha256": digest(plan), "task": task,
                            "feature_manifest_sha256": sha(cache / f"L{task['L']}" / "manifest.json")}
                def progress(value):
                    budget.tick("training:"+task["key"])
                result = train_job(data["train"][task["view"]], data["train"]["y"],
                    data["validation"][task["view"]], data["validation"]["y"], settings, job, bindings,
                    stop_requested=stop, max_seconds=budget.remaining,
                    checkpoint_updates=config["execution"]["checkpoint_updates"], progress=progress)
                budget.tick("training")
                write_status(run, plan)
                print(f"{task['key']}: {result['state']} at epoch {result.get('epoch')}", flush=True)
                if result["state"] != "complete":
                    break
            if stop() or any(not (run/"models"/t["key"]/"state.json").exists()
                             or load_job(run/"models"/t["key"])["state"] != "complete" for t in plan["tasks"]):
                return write_status(run, plan, "paused_training")
            data = None
            # Development exports are post-selection and use no test payload.
            budget.tick("development_exports")
            for L in config["size_order"]:
                data = load_features(cache, L)
                for task in [t for t in plan["tasks"] if t["L"] == L]:
                    if stop():
                        return write_status(run, plan, "paused_development_exports")
                    _evaluate_task(root, run, plan, task, data["validation"], "validation")
                    budget.tick()
            data = None
            if stop():
                return write_status(run, plan, "paused_before_seal")
            seal = _seal(root, run, plan)
            budget.tick("fresh_confirmation_generation")
            prior = artifact(root, config["prior_manifest_root"])
            conf = config["confirmation"]
            confirmation = prepare_confirmation(root, run/"confirmation", config["sizes"], conf["count_per_size"],
                conf["seed"], conf["sampling"], [prior/f"L{L}.json" for L in config["sizes"]], digest(seal),
                bindings={"source_sha256": plan["sources"], "model_sha256": seal["model_sha256"]})
            last_progress = [0.0]
            def cohort_progress(value):
                if time.monotonic() - last_progress[0] >= 2:
                    budget.tick()
                    last_progress[0] = time.monotonic()
            generated = run_confirmation(root, confirmation, workers=workers, max_seconds=budget.remaining,
                                         stop_requested=stop, progress_callback=cohort_progress)
            budget.tick("confirmation_evaluation")
            if stop() or generated["state"] != "complete":
                return write_status(run, plan, "paused_confirmation_generation")
            for L in config["size_order"]:
                data = load_confirmation_features(confirmation, L)
                for task in [t for t in plan["tasks"] if t["L"] == L]:
                    if stop():
                        return write_status(run, plan, "paused_confirmation_evaluation")
                    _evaluate_task(root, run, plan, task, data, "confirmation")
                    if "x_exact" in data:
                        _evaluate_task(root, run, plan, task, data, "confirmation", exact=True)
                    budget.tick()
                data = None
            if config["raw_panel"]["enabled"]:
                from quantum_lattice.models.final_raw_controls import run_raw_controls
                budget.tick("raw_input_controls")
                raw_result = run_raw_controls(root, run, plan, confirmation, stop_requested=stop, remaining_seconds=lambda: budget.remaining)
                budget.tick()
                if stop() or raw_result["state"] != "complete":
                    return write_status(run, plan, "paused_raw_controls")
            if conf["diagnostics"]["enabled"]:
                from quantum_lattice.data.final_diagnostics import run_diagnostics
                budget.tick("sampling_diagnostics")
                result = run_diagnostics(confirmation, run/"sampling_diagnostics", repo_root=root,
                    count_per_size=conf["diagnostics"]["hamiltonians_per_size"],
                    replicas=conf["diagnostics"]["replicas"], workers=workers,
                    max_seconds=budget.remaining, stop_requested=stop)
                budget.tick()
                if stop() or result.get("state") != "complete":
                    return write_status(run, plan, "paused_sampling_diagnostics")
            atomic_json(run/"scientific-work-complete.json", {"plan_sha256": digest(plan),
                        "confirmation_seal_sha256": digest(seal)})
            return _finish_reports(root, run, plan, seal, budget, stop)
        finally:
            if not (run/"complete.json").exists():
                budget.stage = "paused:" + budget.stage
            budget.tick(done=True)
            write_status(run, plan)
            for s, handler in previous.items():
                signal.signal(s, handler)


def _comparison_reports(root, run, plan):
    rows, contrasts = [], []
    for role in ("validation", "confirmation"):
        for L in plan["config"]["sizes"]:
            cells = {}
            uid_order = None
            for task in [t for t in plan["tasks"] if t["L"] == L]:
                out = run/"evaluations"/role/task["key"]
                value = load_evaluation(out)
                ids = [r["hamiltonian_uid"] for r in value["per_hamiltonian"]]
                if uid_order is not None and uid_order != ids:
                    raise ValueError("Evaluation UID order differs across methods/seeds")
                uid_order = ids
                cells[(task["model"], task["view"], task["seed"])] = value
                rows.append({"role": role, **task, "mse": value["metrics"]["mse"],
                             "D5_reference_sign": value["metrics"]["reference_sign_D5"]["two_channel_full"],
                             "hamiltonians": value["hamiltonians"]})
            def vector(key, metric):
                return [r[metric] for r in cells[key]["per_hamiltonian"]]
            comparisons = [("g_minus_f_cnn", ("cnn", "g"), ("cnn", "f")),
                           ("g_minus_f_mlp", ("map_mlp", "g"), ("map_mlp", "f")),
                           ("cnn_minus_mlp_f", ("cnn", "f"), ("map_mlp", "f")),
                           ("cnn_minus_mlp_g", ("cnn", "g"), ("map_mlp", "g"))]
            for label, left, right in comparisons:
                for metric in ("mse", "reference_sign_D5_two_channel_full"):
                    a = {str(s): vector((*left, s), metric) for s in plan["config"]["seeds"]}
                    b = {str(s): vector((*right, s), metric) for s in plan["config"]["seeds"]}
                    result = compare_seed_panels(a, b, uids=uid_order,
                        bootstrap_samples=plan["config"]["statistics"]["bootstrap_samples"],
                        seed=plan["config"]["statistics"]["seed"])
                    contrasts.append({"role": role, "L": L, "comparison": label, "metric": metric,
                        "statistical_primary": role == "confirmation" and L == 4 and label == "g_minus_f_cnn" and metric == "mse",
                        **result})
    atomic_json(run/"paired-comparisons.json", {"scope": "whole-H intervals conditional on frozen models/training bank; other contrasts secondary", "contrasts": contrasts})
    if plan["config"]["raw_panel"]["enabled"]:
        _raw_comparisons(run, plan)
    with (run/"model-scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# Final multi-size campaign results", "", "All outcomes, including unfavorable comparisons and sampling warnings, must be reviewed before paper claims.", "",
             f"Training: {len(plan['tasks'])} fixed-policy jobs; sizes {plan['config']['sizes']}; all declared paired seeds.",
             "Fresh confirmation was generated only after all selected models were hash-sealed.",
             "Known-field observable preprocessing and raw-input controls are separate information panels.", "",
             "- model-scores.csv: all own-input development and confirmation seed scores.",
             "- paired-comparisons.json: whole-H paired uncertainty and seed outcomes; one declared primary.",
             "- models/: resumable/best checkpoints and histories; capped fits remain labeled.",
             "- evaluations/: raw predictions, per-H MSE/D5, manifests; reference signs labeled.",
             "- confirmation/: fresh Hamiltonians, packed records and shared f/g views.",
             "- raw_controls/: separate M256/M64 raw mean/set/inverse scores, convergence and timings.",
             "- sampling_diagnostics/: four-chain targeted checks; flags never remove H or certify all chains.",
             "- clock.json/status.json/confirmation-seal.json: active budget, progress and pre-access model seal.", "",
             "This is finite-budget simulation evidence, not logical error correction, experimental access or universal mixing/convergence."]
    (run/"RESULTS.md").write_text("\n".join(lines)+"\n")


def _raw_comparisons(run, plan):
    """Separate paired summaries; a deterministic solver is not five fits."""
    raw = plan["config"]["raw_panel"]
    results, rows = [], []
    for M in raw["records"]:
        inverse = load_evaluation(run/"raw_controls"/f"M{M}-inverse"/"evaluation")
        uids = [r["hamiltonian_uid"] for r in inverse["per_hamiltonian"]]
        cells = {}
        for seed in raw["seeds"]:
            for method in raw["methods"]:
                value = load_evaluation(run/"raw_controls"/f"M{M}-{method}-seed{seed}")
                if [r["hamiltonian_uid"] for r in value["per_hamiltonian"]] != uids:
                    raise ValueError("Raw-control paired Hamiltonian order differs")
                cells[method, seed] = value
                rows.append({"records": M, "method": method, "seed": seed,
                             "mse": value["metrics"]["mse"],
                             "reference_sign_D5": value["metrics"]["reference_sign_D5"]["two_channel_full"]})
        for metric in ("mse", "reference_sign_D5_two_channel_full"):
            panels = {method: {str(s): [r[metric] for r in cells[method, s]["per_hamiltonian"]]
                              for s in raw["seeds"]} for method in raw["methods"]}
            panels["inverse"] = {str(s): [r[metric] for r in inverse["per_hamiltonian"]] for s in raw["seeds"]}
            for left, right in (("set_cnn", "mean_cnn"), ("mean_cnn", "inverse"), ("set_cnn", "inverse")):
                result = compare_seed_panels(panels[left], panels[right], uids=uids,
                    bootstrap_samples=plan["config"]["statistics"]["bootstrap_samples"],
                    seed=plan["config"]["statistics"]["seed"])
                results.append({"records": M, "comparison": left+"_minus_"+right, "metric": metric,
                    "statistical_primary": False, "control_scope": "one inverse fit per H/budget, repeated across seed keys only for a paired contrast", **result})
        rows.append({"records": M, "method": "inverse", "seed": "not_applicable",
                     "mse": inverse["metrics"]["mse"],
                     "reference_sign_D5": inverse["metrics"]["reference_sign_D5"]["two_channel_full"]})
    atomic_json(run/"raw_controls"/"paired-comparisons.json", {"scope": "secondary raw-input panel; M64 is budget shift of frozen M256 networks", "contrasts": results})
    with (run/"raw_controls"/"model-scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _finish_reports(root, run, plan, seal, budget, stop):
    budget.finalizing = True
    config = plan["config"]
    if stop():
        return write_status(run, plan, "paused_reports")
    budget.tick("reports")
    _comparison_reports(root, run, plan)
    if config["sizes"] == list(SIZES) and len(config["seeds"]) == 5:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(run/"render-cache"))
        from quantum_lattice.models.final_plots import render_final_plots
        render_final_plots(run)
    inventory = {}
    last_inventory_tick = time.monotonic()
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if (relative.parts[0] in {"supervisor", "render-cache"} or path.name.startswith(".")
                or path.name.endswith(".tmp") or path.name in {"clock.json", "status.json", "integrity.json", "complete.json", "PAUSE"}):
            continue
        if stop():
            return write_status(run, plan, "paused_final_verification")
        inventory[str(relative)] = sha(path)
        if time.monotonic()-last_inventory_tick >= 2:
            budget.tick("final_verification")
            last_inventory_tick = time.monotonic()
    atomic_json(run/"integrity.json", {"schema_version": 1, "plan_sha256": digest(plan),
                "files": inventory, "scope": "scientific artifacts; excludes mutable worker/control/status records"})
    atomic_json(run/"complete.json", {"schema_version": 1, "plan_sha256": digest(plan),
                "confirmation_seal_sha256": digest(seal), "completed_epoch": time.time(),
                "active_seconds": budget.active_seconds, "state": "complete",
                "integrity_sha256": sha(run/"integrity.json")})
    budget.tick("complete")
    return write_status(run, plan, "complete")
