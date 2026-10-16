"""Small validation menu; runtime worker preferences stay outside the plan."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def _positive_workers(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("CPU workers must be a positive integer")
    return value


def resolve_workers(run_dir, default, override=None):
    """Explicit CLI override, saved runtime preference, then plan default."""
    if override is not None:
        return _positive_workers(override)
    path = Path(run_dir)/"runtime-settings.json"
    if not path.exists():
        return _positive_workers(default)
    try:
        settings = json.loads(path.read_text())
        return _positive_workers(settings["workers"])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("Saved CPU worker setting is invalid; choose a new count in the menu or pass --workers") from error


def save_workers(run_dir, workers):
    """Atomic, per-campaign operational preference; no plan/config mutation."""
    workers = _positive_workers(workers)
    run_dir = Path(run_dir)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=run_dir, prefix=".runtime-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({"workers": workers}, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, run_dir/"runtime-settings.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def saved_progress(run_dir):
    path = Path(run_dir)/"status.json"
    if not path.exists():
        return {"state": "not started"}
    try:
        status = json.loads(path.read_text())
        if not isinstance(status, dict):
            raise ValueError("status must be an object")
        return status
    except (ValueError, OSError) as error:
        return {"state": "unavailable", "message": f"Cannot read saved progress: {error}"}


def display_progress(run_dir, *, detailed=False):
    """Read status only: never scan jobs or rewrite scientific outputs."""
    status = saved_progress(run_dir)
    done = status.get("completed", status.get("completed_selected"))
    total = status.get("total", status.get("total_selected"))
    text = f"{done:,}/{total:,} tasks" if isinstance(done, int) and isinstance(total, int) else "counts unavailable"
    scope = f" | selected stage: {status['stage']}" if "completed_selected" in status else ""
    print(f"State: {status.get('state', 'unknown')} | {text}{scope}")
    for name, row in status.get("by_stage", {}).items():
        print(f"  {name}: {row['completed']:,}/{row['total']:,}")
    if status.get("pause_reason") or status.get("failure") or status.get("message"):
        print(status.get("failure") or status.get("pause_reason") or status.get("message"))
    if detailed:
        print(json.dumps(status, indent=2))
        print(f"Report: {Path(run_dir)/'RESULTS.md'}")
        print(f"Log: {Path(run_dir)/'execution.log'}")
    return status


def dashboard(run_dir, *, default_workers, start, workers_override=None, stage="all"):
    """Start/continue, set workers, inspect, exit; Ctrl+C run pauses return here."""
    run_dir = Path(run_dir)
    try:
        workers = resolve_workers(run_dir, default_workers, workers_override)
    except ValueError as error:
        print(error)
        workers = _positive_workers(default_workers)
    while True:
        print("\nEstimator validation")
        status = display_progress(run_dir)
        print(f"CPU workers: {workers} | available logical CPUs: {available_cpus()} | stage: {stage}")
        print("1  Start / continue tests\n2  Change CPU workers\n3  Saved progress and result paths\nq  Exit")
        try:
            action = input("Action [1]: ").strip().lower() or "1"
            if action in {"q", "quit"}:
                return 0
            if action in {"2", "w"}:
                value = input(f"CPU workers [{workers}]: ").strip()
                if value:
                    try:
                        selected = _positive_workers(int(value))
                        save_workers(run_dir, selected)
                        workers = selected
                        print(f"Saved {workers} CPU workers for the next start/continue.")
                    except (ValueError, OSError) as error:
                        print(f"Worker count unchanged: {error}")
                continue
            if action == "3":
                display_progress(run_dir, detailed=True)
                continue
            if action != "1":
                print("Choose 1, 2, 3 or q.")
                continue
            if status.get("state") == "complete":
                print(f"This campaign is complete. Report: {run_dir/'RESULTS.md'}")
                continue
            print("Ctrl+C saves active work and returns to this menu. Change workers before continuing.")
            try:
                start(workers)
            except (ValueError, RuntimeError, OSError, ImportError) as error:
                print(f"Cannot continue: {error}")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
