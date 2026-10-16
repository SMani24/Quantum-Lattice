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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "prepare", "status", "report"), default="run")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, help="existing campaign, otherwise use config output_root")
    parser.add_argument("--workers", type=int, help="independent CPU workers; default 4, may change on continuation")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
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
    workers = args.workers if args.workers is not None else plan["config"]["execution"]["workers"]
    print(f"\nEstimator validation: {run}\nCPU workers: {workers}; BLAS/OMP threads: 1 each.")
    print("Ctrl+C pauses after active tasks/checkpoint chunks. Run the same command to continue.")
    print("Original data/models stay intact. Final-test payloads are excluded.\n", flush=True)
    with (run/"execution.log").open("a", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            summary = run_campaign(ROOT, run, workers=workers, stage=args.stage)
            print(f"\n{summary['state']}: {summary['completed']:,}/{summary['total']:,} tasks.")
            print(f"Report: {run/'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Validation stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
