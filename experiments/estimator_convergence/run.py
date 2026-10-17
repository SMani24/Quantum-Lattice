#!/usr/bin/env python3
"""Continue all 20 paired estimator MLPs for a matched CUDA convergence check."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys

from quantum_lattice.models.estimator_convergence import (
    format_summary, load_plan, prepare, run_campaign, summarize_campaign,
)
from quantum_lattice.models.mlp_training import resolve_device

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/estimator_convergence.json"


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


def launch(run, max_chunks=None):
    plan = load_plan(ROOT, run)
    with (run / "execution.log").open("a", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print(f"\nMatched MLP convergence check: {run}")
            print(f"Continue all {len(plan['tasks'])} CUDA models from their original optimizer and RNG states.")
            print(f"Ceiling: {plan['config']['training']['max_steps']:,} total updates per model. "
                  "Ctrl+C pauses at a checkpoint.")
            print("Original models and feature banks stay intact; validation only.\n", flush=True)
            result = run_campaign(ROOT, run, max_chunks=max_chunks)
            print(f"Report: {run / 'RESULTS.md'}")
    return result


def dashboard(run, max_chunks=None):
    while True:
        print("\nMatched MLP convergence — L=3,4; 20 existing models")
        print("1  Run / continue\n2  Progress and results\nq  Quit")
        try:
            choice = input("Choose: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in ("q", "quit", "exit"):
            return 0
        if choice == "1":
            launch(run, max_chunks)
        elif choice == "2":
            print(format_summary(summarize_campaign(ROOT, run)))
            print(f"Results: {run / 'RESULTS.md'}")
        else:
            print("Choose 1, 2 or q.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("menu", "prepare", "run", "status", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, help="existing frozen convergence campaign")
    parser.add_argument("--max-chunks", type=int, help="pause after this many checkpoint chunks (bounded probe)")
    args = parser.parse_args(argv)
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be positive")
    command = args.command or ("menu" if sys.stdin.isatty() and sys.stdout.isatty() else "run")
    if command == "run":
        resolve_device("cuda", True)
    if args.run_dir:
        run = args.run_dir.resolve()
        load_plan(ROOT, run)
    else:
        config = json.loads(args.config.read_text())
        run = (ROOT / config["output_root"]).resolve()
        if command in ("status", "report"):
            load_plan(ROOT, run)
        else:
            if not (run / "plan.json").exists():
                print("Verifying the completed parent campaign and freezing its matched continuation. "
                      "Preparation performs no optimizer updates.", flush=True)
            run = prepare(ROOT, config)
    if command == "prepare":
        print(f"Frozen 20-model continuation prepared: {run}")
        print("No training or Monte Carlo run. Choose Run / continue when ready.")
        return 0
    if command in ("status", "report"):
        print(format_summary(summarize_campaign(ROOT, run)))
        print(f"Results: {run / 'RESULTS.md'}")
        return 0
    if command == "menu":
        return dashboard(run, args.max_chunks)
    launch(run, args.max_chunks)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Estimator convergence stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
