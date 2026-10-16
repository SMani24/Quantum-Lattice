#!/usr/bin/env python3
"""Run or resume the paired-estimator validation campaign."""
import os
# Set before importing NumPy in the spawned coordinator or workers.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys

from quantum_lattice.data.estimator_validation import (
    STAGES, campaign_lock, load_plan, prepare, run_campaign, summarize_campaign,
)
from quantum_lattice.simulation.validation_terminal import dashboard, resolve_workers

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/conditioned_estimator_validation.json"


class Tee:
    def __init__(self, screen, logfile):
        self.screen, self.logfile = screen, logfile

    def write(self, value):
        self.screen.write(value)
        self.logfile.write(value)
        self.logfile.flush()
        return len(value)

    def flush(self):
        self.screen.flush()
        self.logfile.flush()


def launch(run, workers, stage):
    with (run/"execution.log").open("a", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print(f"\nEstimator validation: {run}\nCPU workers: {workers}; BLAS/OMP threads: 1 each.")
            print("Ctrl+C pauses after active tasks/checkpoint chunks.")
            print("Original data/models stay intact. Final-test payloads are excluded.\n", flush=True)
            summary = run_campaign(ROOT, run, workers=workers, stage=stage)
            print(f"\n{summary['state']}: {summary['completed']:,}/{summary['total']:,} tasks.")
            print(f"Report: {run/'RESULTS.md'}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("menu", "run", "prepare", "status", "report"),
                        help="default: menu in a terminal, run when redirected")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, help="existing campaign, otherwise use config output_root")
    parser.add_argument("--workers", type=int, help="CPU worker override; otherwise saved menu preference, then plan default")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.command is None:
        args.command = "menu" if sys.stdin.isatty() and sys.stdout.isatty() else "run"
    config = json.loads(args.config.read_text())
    run = args.run_dir.resolve() if args.run_dir else ROOT / config["output_root"]
    if args.command == "status":
        print(json.dumps(json.loads((run/"status.json").read_text()), indent=2))
        print(f"Results: {run/'RESULTS.md'}")
        return 0
    if args.run_dir:
        plan = load_plan(ROOT, run)
    else:
        run = prepare(ROOT, config)
        plan = load_plan(ROOT, run)
    if args.command == "prepare":
        counts = {s: sum(t["stage"] == s for t in plan["tasks"]) for s in STAGES}
        print(f"Prepared: {run}\nTasks: {json.dumps(counts)}\nNo payloads or simulations run.")
        return 0
    if args.command == "report":
        with campaign_lock(run):
            summary = summarize_campaign(run, plan)
        print(json.dumps(summary, indent=2))
        print(f"Report: {run/'RESULTS.md'}")
        return 0
    if args.command == "menu":
        return dashboard(run, default_workers=plan["config"]["execution"]["workers"],
                         workers_override=args.workers, stage=args.stage,
                         start=lambda workers: launch(run, workers, args.stage))
    workers = resolve_workers(run, plan["config"]["execution"]["workers"], args.workers)
    launch(run, workers, args.stage)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Validation stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
