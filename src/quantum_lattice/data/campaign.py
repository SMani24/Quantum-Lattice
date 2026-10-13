"""Select and resume one lattice-size dataset at a time.

The campaign index tracks saved runs; the individual run manifests remain the
authority for their scientific settings and progress. Merely opening this
dashboard never creates chains or launches a simulation.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np

from quantum_lattice.data.generation import (
    _atomic_json, _canonical, _run_lock, artifact_root, benchmark_config, prepare_run,
    inspect_run, run_generation, validate_config, verify_run,
)


def build_profiles(spec: dict) -> dict[int, dict]:
    """Expand a campaign into complete, deterministic single-size settings.

    Round the requested training count up to a complete balanced split block.
    This guarantees the minimum without treating each local edge as an
    independent Hamiltonian. Size-specific seeds avoid sharing a field prefix
    between lattices of different sizes.
    """
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported campaign schema_version")
    name = spec.get("campaign_name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("campaign_name needs letters, digits, underscores or hyphens")
    sizes = spec.get("lattice_sizes")
    if not isinstance(sizes, list) or not sizes or any(
        isinstance(side, bool) or not isinstance(side, int) or side < 3 for side in sizes
    ) or len(set(sizes)) != len(sizes):
        raise ValueError("lattice_sizes must contain distinct integers >= 3")
    minimum, seed = spec.get("training_hamiltonians_minimum"), spec.get("seed")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("training_hamiltonians_minimum must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("campaign seed must be a nonnegative integer")
    base = deepcopy(spec["base_config"])
    names = ("train", "validation", "test")
    fractions = base["dataset"]["split_fractions"]
    if set(fractions) != set(names):
        raise ValueError("campaign splits need train, validation and test")
    ratios = {key: Fraction(str(fractions[key])).limit_denominator(1000) for key in names}
    if sum(ratios.values()) != 1 or any(
        value <= 0 or abs(float(value) - fractions[key]) > 1e-8
        for key, value in ratios.items()
    ):
        raise ValueError("campaign splits must be positive simple fractions summing to one")
    period = math.lcm(*(value.denominator for value in ratios.values()))
    if period > 4096:
        raise ValueError("campaign split fractions need a smaller common denominator")
    train_per_block = int(period * ratios["train"])
    total = math.ceil(minimum / train_per_block) * period
    profiles = {}
    for side in sizes:
        config = deepcopy(base)
        config["lattice"]["L"] = side
        config["dataset"]["hamiltonians"] = total
        config["dataset"]["seed"] = int(
            np.random.SeedSequence([seed, side, 42]).generate_state(1)[0]
        )
        profiles[side] = validate_config(config)
    return profiles


def campaign_root(repo_root, spec: dict) -> Path:
    build_profiles(spec)  # Validate the path component as well as the settings.
    return artifact_root(repo_root) / "campaigns" / spec["campaign_name"]


def estimate_storage_bytes(profile: dict) -> int:
    """Final numeric array payload, before compression and metadata overhead.

    Checkpoint files, JSON/text, ZIP headers, scalar counters and temporary
    verification workspace are deliberately excluded. This is a planning
    estimate rather than a disk-space guarantee.
    """
    side = profile["lattice"]["L"]
    vertices, edges = side * side, 2 * side * side
    sampler = profile["sampler"]
    records = (sampler["total_sweeps"] - sampler["burn_in_sweeps"]) // sampler["thinning_sweeps"]
    storage = profile.get("storage", {})
    state_columns = (edges + 7) // 8 if storage.get("state_encoding", "int8") == "packed_bits" else edges
    payload = records * state_columns
    # fields, endpoint indices, three summary vectors/channel, MLP rows,
    # signed and absolute targets. All scientific numeric outputs are float64.
    payload += edges * (8 + 16 + 24 + 16) + 3 * (vertices + 2 * edges) * 8
    if storage.get("retain_estimator_records", True):
        payload += records * (vertices + 2 * edges) * 8
    if storage.get("retain_block_means", True):
        blocks = records // profile["statistics"]["block_size_records"]
        payload += blocks * (vertices + 2 * edges) * 8
    return payload * profile["dataset"]["hamiltonians"]


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read_spec(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    return spec, build_profiles(spec)


@contextmanager
def _index_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".campaign.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process is updating this campaign index; try again") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read_index(root, spec):
    path = root / "campaign-index.json"
    if not path.exists():
        return {"schema_version": 1, "campaign_name": spec["campaign_name"],
                "spec_sha256": _digest(spec), "runs": {}}
    index = json.loads(path.read_text())
    if index.get("schema_version") != 1 or index.get("spec_sha256") != _digest(spec):
        raise ValueError(
            "campaign settings differ from the saved index; restore the original "
            "settings or use a new campaign_name to create a separate campaign"
        )
    if not isinstance(index.get("runs"), dict):
        raise ValueError("campaign index has an invalid run mapping")
    return index


def _indexed_path(root, entry):
    run_dir = (root / entry["run_dir"]).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("campaign run directory leaves its campaign folder")
    return run_dir


def _saved_config_matches(run_dir, profile):
    config = json.loads((run_dir / "config.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fingerprint = _digest(profile)
    if _digest(config) != fingerprint or manifest.get("config_sha256") != fingerprint:
        raise ValueError(
            f"saved settings in {run_dir} differ from this lattice profile; "
            "restore them or use a new campaign_name"
        )


def _ensure_run(repo_root, spec, side, profile):
    root = campaign_root(repo_root, spec)
    with _index_lock(root):
        index = _read_index(root, spec)
        entry = index["runs"].get(str(side))
        if entry is not None:
            if entry.get("config_sha256") != _digest(profile):
                raise ValueError("campaign profile fingerprint differs from its index")
            path = _indexed_path(root, entry)
            _saved_config_matches(path, profile)
            return path
        path = root / f"L{side}" / "run"
        if path.exists():
            # Recover an index write interrupted after a complete prepare_run.
            # Partial preparation or mismatched settings raise, never overwrite.
            _saved_config_matches(path, profile)
        else:
            prepare_run(repo_root, profile, path, purpose=f"campaign-L{side}")
        index["runs"][str(side)] = {
            "run_dir": str(path.relative_to(root)), "config_sha256": _digest(profile),
            "registered_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(root / "campaign-index.json", index)
        return path


def _profile_status(root, index, side, profile):
    total = profile["dataset"]["hamiltonians"]
    entry = index["runs"].get(str(side))
    result = {"L": side, "total": total, "completed": 0, "state": "not started",
              "run_dir": None, "numeric_payload_bytes": estimate_storage_bytes(profile),
              "split_counts": {name: round(total * fraction) for name, fraction in
                               profile["dataset"]["split_fractions"].items()}}
    if entry is None:
        return result
    if entry.get("config_sha256") != _digest(profile):
        raise ValueError("campaign profile fingerprint differs from its index")
    path = _indexed_path(root, entry)
    _saved_config_matches(path, profile)
    # Read sidecar filenames, never decompress thousands of saved NPZ archives.
    completed_indices = set()
    for file in (path / "instances").glob("hamiltonian_*.json"):
        match = re.fullmatch(r"hamiltonian_(\d+)\.json", file.name)
        if match:
            number = int(match[1])
            if number < total and file.name == f"hamiltonian_{number:05d}.json" and file.with_suffix(".npz").is_file():
                completed_indices.add(number)
    completed = len(completed_indices)
    status_path = path / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "prepared"}
    state = status.get("state", "unknown")
    if state == "complete" and (completed != total or not (path / "dataset.json").exists()):
        state = "inconsistent completion"
    result.update(completed=completed, state=state, run_dir=str(path),
                  saved_status=status)
    return result


def campaign_status(repo_root, spec_path):
    """Inspect every configured size without loading estimator/state archives."""
    spec, profiles = _read_spec(spec_path)
    root = campaign_root(repo_root, spec)
    index = _read_index(root, spec)
    disk = shutil.disk_usage(Path(repo_root))
    return {"campaign_name": spec["campaign_name"], "campaign_root": str(root),
            "free_disk_bytes": disk.free,
            "storage_estimate_scope": "final numeric payload before compression; excludes JSON/ZIP overhead and checkpoints",
            "profiles": [_profile_status(root, index, side, profile)
                         for side, profile in profiles.items()]}


def _bytes(value):
    return f"{value / 1024 ** 3:.2f} GiB" if value >= 1024 ** 3 else f"{value / 1024 ** 2:.1f} MiB"


def _display(status):
    print(f"\nQuantum Lattice · {status['campaign_name']}\n")
    print("      L    train H    total H    finished       saved state           numeric payload")
    for number, row in enumerate(status["profiles"], 1):
        state = row["state"].replace(" ", "_").upper()
        print(f"{number:>2}   {row['L']:>3}    {row['split_counts']['train']:>7}    {row['total']:>7}"
              f"    {row['completed']:>5}/{row['total']:<5}   {state:<22} {_bytes(row['numeric_payload_bytes'])}")
    print(f"\nFree disk: {_bytes(status['free_disk_bytes'])}. Payload estimates exclude checkpoints and metadata.")
    print("Choose one size; other sizes will not start automatically. q exits.\n")


def _benchmark(repo_root, profile, side, workers, interactive):
    config = benchmark_config(profile, sweeps=5000, jobs=4)
    if workers is not None:
        config["execution"]["workers"] = workers
    path = prepare_run(repo_root, config, purpose=f"benchmark-L{side}")
    print(f"\nShort benchmark: four L={side} Hamiltonians, 5,000 sweeps each.\nSaved run: {path}")
    result = run_generation(repo_root, path, workers=workers, interactive=interactive)
    if result["state"] == "complete":
        target_attempts = profile["dataset"]["hamiltonians"] * profile["sampler"]["total_sweeps"] * side * side
        _, _, progress = inspect_run(path)
        compute_seconds = sum(row["compute_seconds"] for row in progress.values())
        benchmark_attempts = sum(row["attempted"] for row in progress.values())
        parallel_compute = compute_seconds / result["workers"]
        # elapsed_seconds is the sampling session; verification runs afterward.
        # Scale worker compute, rather than multiplying compiler/process startup
        # thousands of times. Unmeasured full-bank I/O remains a limitation.
        startup_and_io = max(0.0, result["elapsed_seconds"] - parallel_compute)
        estimate = startup_and_io + parallel_compute * target_attempts / benchmark_attempts
        _atomic_json(path / "benchmark.json", {**result, "target_profile": profile,
                     "benchmark_compute_seconds": compute_seconds,
                     "benchmark_attempts": benchmark_attempts,
                     "estimated_startup_and_io_seconds": startup_and_io,
                     "verification_seconds_excluded_from_estimate": True,
                     "estimated_dataset_seconds": estimate,
                     "estimate_scope": "first local short benchmark; worker compute scaled and startup/I/O approximated once; full-bank storage/verification and mixing may differ"})
        if estimate is not None:
            print(f"Rough L={side} dataset estimate: {estimate / 3600:.2f} hours at this measured rate.")
        print("This first local short benchmark estimates sampling time; large-run I/O and verification need additional time.")
        print("The benchmark does not certify burn-in or equilibration.")
    else:
        print(f"Benchmark paused; saved run: {path}")


def _verify_with_progress(run_dir):
    run_dir = Path(run_dir)
    previous = 0.0
    started = False
    status_path = run_dir / "status.json"
    original_status = json.loads(status_path.read_text())

    def progress(done, total, instance_id):
        nonlocal previous, started
        started = True
        now = time.monotonic()
        if done == total or now - previous >= 1.0:
            print(f"Verifying {done}/{total} Hamiltonians" + (f" · {instance_id}" if instance_id else ""), flush=True)
            _atomic_json(status_path, {**original_status, "state": "verifying",
                         "completed": total, "total": total, "verified": done})
            previous = now

    try:
        with _run_lock(run_dir):
            report = verify_run(run_dir, progress_callback=progress)
    except KeyboardInterrupt:
        if started:
            _atomic_json(status_path, {**original_status, "state": "verification_pending"})
        raise
    except Exception:
        # A configuration/menu error preceding the actual verification loop
        # must not relabel an unrelated dataset as failed.
        if started:
            _atomic_json(status_path, {**original_status, "state": "verification_failed"})
        raise
    _atomic_json(status_path, {**original_status, "state": "complete"})
    print(f"Verification passed for {len(report['instances'])} Hamiltonians; mixing is not certified.")


def dashboard(repo_root, spec_path, workers_override=None, interactive=True) -> int:
    """Line-based selector; existing keyboard controls remain active while sampling."""
    spec, profiles = _read_spec(spec_path)
    if workers_override is not None and (isinstance(workers_override, bool) or
                                        not isinstance(workers_override, int) or workers_override < 1):
        raise ValueError("workers override must be a positive integer")
    preferred = spec.get("suggested_first_L", 4)
    default_side = preferred if preferred in profiles else next(iter(profiles))
    worker_count = workers_override if workers_override is not None else profiles[default_side]["execution"]["workers"]
    while True:
        status = campaign_status(repo_root, spec_path)
        _display(status)
        sides = list(profiles)
        default_number = sides.index(default_side) + 1
        try:
            choice = input(f"Lattice size [{default_number} = L{default_side}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in {"q", "quit"}:
            return 0
        choice = choice or str(default_number)
        if not choice.isdigit() or not 1 <= int(choice) <= len(sides):
            print(f"Choose 1–{len(sides)} or q.")
            continue
        side = sides[int(choice) - 1]
        default_side = side
        profile = profiles[side]
        while True:
            row = next(row for row in campaign_status(repo_root, spec_path)["profiles"] if row["L"] == side)
            print(f"\nL={side}: {row['completed']}/{row['total']} Hamiltonians saved · {row['state']}")
            print(f"Splits: {row['split_counts']} · numeric payload {_bytes(row['numeric_payload_bytes'])}")
            print(f"CPU workers: {worker_count} · one dataset at a time")
            fields, sampler = profile["fields"], profile["sampler"]
            records = (sampler["total_sweeps"] - sampler["burn_in_sweeps"]) // sampler["thinning_sweeps"]
            print(f"Fields: independent Uniform[{fields['low']}, {fields['high']}] · channel {fields['channel']}")
            print(f"Sweeps: {sampler['total_sweeps']:,} total / {sampler['burn_in_sweeps']:,} burn-in"
                  f" / retain every {sampler['thinning_sweeps']} · {records:,} records per Hamiltonian")
            print(f"Checkpoint chunk: {profile['execution']['chunk_sweeps']:,} sweeps"
                  f" · lazy stays: {100 * sampler['lazy_probability']:g}%")
            storage = profile.get("storage", {})
            traces = "retained" if storage.get("retain_estimator_records", True) else "reconstructible from states"
            blocks = "retained" if storage.get("retain_block_means", True) else "omitted; uncertainty summaries retained"
            print(f"Storage: {storage.get('state_encoding', 'int8')} states · estimator traces {traces}")
            print(f"Block means: {blocks}")
            if profile.get("provenance", {}).get("our_choices", {}).get("phase_selection"):
                print("Historical topological-phase filter is unavailable; no reconstructed filter is applied.")
            if row["run_dir"]:
                print(f"Run: {row['run_dir']}")
            print("1  Start / resume this dataset\n2  Short speed benchmark (separate run)"
                  "\n3  Inspect saved progress\n4  Verify completed dataset"
                  "\nw  Change CPU worker count\nb  Choose another size\nq  Exit")
            try:
                action = input("Action [3]: ").strip().lower() or "3"
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if action in {"q", "quit"}:
                return 0
            if action in {"b", "back"}:
                break
            if action == "w":
                try:
                    value = input(f"CPU workers [{worker_count}]: ").strip()
                    if value:
                        selected = int(value)
                        if selected < 1:
                            raise ValueError("worker count must be positive")
                        worker_count = selected
                except ValueError:
                    print("Use a positive integer; worker count is unchanged.")
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                continue
            if action == "3":
                print(json.dumps({"progress": row, "effective_profile": profile,
                                  "runtime_workers": worker_count}, indent=2))
                continue
            try:
                if action == "1":
                    if row["state"] == "complete":
                        print("This dataset is already complete. Inspect or verify it; it will not be rerun.")
                        continue
                    path = _ensure_run(repo_root, spec, side, profile)
                    controls = ("p pauses; r resumes; q or Ctrl+C saves and returns to this dashboard."
                                if interactive else "Plain progress; Ctrl+C saves and returns to this dashboard.")
                    print(f"\nSaved run: {path}\n{controls}\n")
                    result = run_generation(repo_root, path, workers=worker_count, interactive=interactive)
                    print(f"\n{result['state'].capitalize()}: {result['completed']}/{result['total']} Hamiltonians.")
                elif action == "2":
                    _benchmark(repo_root, profile, side, worker_count, interactive)
                elif action == "4":
                    if row["run_dir"] is None or row["completed"] != row["total"]:
                        print("Finish this dataset before verification.")
                    elif row["state"] != "complete":
                        # The generator finalizes dataset.json and completion
                        # status after checking all saved samples. Calling the
                        # verifier alone would leave a pending bank unfinished.
                        result = run_generation(repo_root, Path(row["run_dir"]),
                                                workers=worker_count, interactive=interactive)
                        print(f"\n{result['state'].capitalize()}: {result['completed']}/{result['total']} Hamiltonians.")
                    else:
                        _verify_with_progress(row["run_dir"])
                else:
                    print("Choose 1, 2, 3, 4, w, b or q.")
            except KeyboardInterrupt:
                print("\nStopped verification; completed samples remain saved.")
            except (ValueError, RuntimeError, ImportError, OSError, KeyError, TypeError) as error:
                print(f"Cannot continue: {error}")
