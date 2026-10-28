#!/usr/bin/env python3
"""Simple final-campaign menu and explicit noninteractive controls."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
LOCAL_CONFIG = "codex material/artifacts/final_benchmark/inputs/worker-config.json"


def settings(path):
    # Keep help/menu and the first CPU guard independent of numerical imports.
    value = json.loads((ROOT/path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def early_cpu_guard(config):
    expected = {0, 1, 3, 4, 5, 6, 7, 9, 10, 11}
    parts = Path("/sys/devices/system/cpu/online").read_text().strip().split(",")
    actual = set()
    for part in parts:
        if "-" in part:
            left, right = map(int, part.split("-")); actual.update(range(left, right+1))
        else:
            actual.add(int(part))
    if actual != expected or Path("/sys/devices/system/cpu/cpufreq/boost").read_text().strip() != "0":
        raise RuntimeError("PC isolation/boost differs from the qualified workaround; no numerical code was started")
    os.sched_setaffinity(0, expected)


def show(config):
    run = (ROOT/config["output_root"]).resolve()
    allowed = (ROOT/"codex material/artifacts/final_benchmark").resolve()
    if run == allowed or not run.is_relative_to(allowed):
        raise ValueError("Unsafe campaign output path")
    if not (run/"plan.json").exists():
        print("Not prepared. Verify/stage the compact inputs first.")
        return
    from quantum_lattice.models.final_campaign import read, load_plan, write_status, format_status
    plan = load_plan(ROOT, run, verify_data=False)
    status = write_status(run, plan, persist=False)
    print(format_status(status))
    for name, label in [("confirmation/state.json", "Fresh Hamiltonians"),
                        ("sampling_diagnostics/status.json", "Sampling diagnostic")]:
        if (run/name).exists():
            print(label+": "+json.dumps(read(run/name)))
    if (run/"supervisor"/"supervisor-state.json").exists():
        state = read(run/"supervisor"/"supervisor-state.json")
        print("Worker: "+str(state.get("state"))+"; guard="+str(state.get("first_guard")))
    print("Run folder:", run)


def start(config, config_path, workers=None, *, resume=False, internal=False):
    # Heavy scientific imports happen only after choosing an explicit action.
    early_cpu_guard(config)
    if internal and os.environ.get("QL_FINAL_SUPERVISOR_PID") != str(os.getppid()):
        raise RuntimeError("The internal runner requires its live monitoring supervisor; use --run")
    from quantum_lattice.models.final_campaign import prepare, read, artifact
    if internal:
        from quantum_lattice.models.final_campaign import run_campaign
        result = run_campaign(ROOT, artifact(ROOT, config["output_root"]), workers=workers, resume=resume)
        return 0 if result["confirmation_complete"] else 75
    run = prepare(ROOT, config)
    if (run/"complete.json").exists():
        from quantum_lattice.models.final_campaign import load_plan, verify_completed
        verify_completed(run, load_plan(ROOT, run))
        print("Campaign already complete; no fitting or sampling was repeated.")
        show(config)
        return 0
    if (run/"PAUSE").exists():
        if not resume:
            raise ValueError("Paused: start with --resume, or use menu option2")
        (run/"PAUSE").unlink()
    clock = read(run/"clock.json")
    remaining = config["execution"]["max_active_hours"]*3600-clock["active_seconds"]
    if remaining <= config["hardware"]["drain_seconds"]+30:
        print("The cumulative four-day budget is exhausted; no child was launched.")
        return 75
    from quantum_lattice.models.final_worker import supervise
    command = [sys.executable, str(Path(__file__).resolve()), "--config", config_path,
               "--run-noninteractive", "--resume"]
    if workers is not None:
        command += ["--workers", str(workers)]
    print("Starting monitored fixed-method campaign. Ctrl+C requests an atomic pause.", flush=True)
    result = supervise(command, ROOT, run/"supervisor", remaining, config["hardware"], stop_file=run/"PAUSE")
    print("Worker stopped:", result.get("state"), result.get("stop_reason"), flush=True)
    show(config)
    return 0 if result.get("state") == "complete" and result.get("returncode") == 0 else 75


def menu(config, path, workers):
    while True:
        print("\nFinal paper campaign · all seven sizes · 96 h maximum")
        print("1  Verify inputs / prepare\n2  Start or resume\n3  Progress and saved results\n4  Request pause\n0  Exit")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return 0
        try:
            if choice == "0":
                return 0
            if choice == "1":
                from quantum_lattice.models.final_campaign import prepare
                print("Prepared:", prepare(ROOT, config))
            elif choice == "2":
                default = workers or config["execution"]["workers"]
                value = input(f"CPU sampler workers (1–10) [{default}]: ").strip()
                selected = int(value) if value else default
                if not 1 <= selected <= 10:
                    raise ValueError("Choose 1–10 workers")
                start(config, path, selected, resume=True)
            elif choice == "3":
                show(config)
            elif choice == "4":
                from quantum_lattice.models.final_campaign import request_pause
                print("Pause requested:", request_pause(ROOT, config))
            else:
                print("Choose 0–4.")
        except (ValueError, RuntimeError, FileNotFoundError) as error:
            print("Stopped:", error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=LOCAL_CONFIG if (ROOT/LOCAL_CONFIG).exists() else "configs/final_benchmark.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--pause", action="store_true")
    group.add_argument("--verify", action="store_true", help="Verify all completed scientific output hashes")
    group.add_argument("--run-noninteractive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = settings(args.config)
    if args.workers is not None and not 1 <= args.workers <= 10:
        parser.error("--workers must be 1–10")
    if args.status:
        show(config); return 0
    if args.verify:
        from quantum_lattice.models.final_campaign import artifact, load_plan, verify_completed
        run = artifact(ROOT, config["output_root"])
        print(verify_completed(run, load_plan(ROOT, run))); return 0
    if args.pause:
        from quantum_lattice.models.final_campaign import request_pause
        print(request_pause(ROOT, config)); return 0
    if args.prepare:
        from quantum_lattice.models.final_campaign import prepare
        print(prepare(ROOT, config)); return 0
    if args.run or args.run_noninteractive:
        return start(config, args.config, args.workers, resume=args.resume, internal=args.run_noninteractive)
    return menu(config, args.config, args.workers)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Stopped safely: {error}", file=sys.stderr)
        raise SystemExit(2)
