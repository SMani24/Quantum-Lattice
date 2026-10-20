#!/usr/bin/env python3
"""Screen four periodic CNN variants on CUDA, with resumable checkpoints."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys

from quantum_lattice.models.cnn_ablation_campaign import (
    format_summary, load_plan, prepare, run_campaign, summarize_campaign,
)
from quantum_lattice.models.cnn_variants import VARIANT_NAMES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cnn_ablation.json"


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


def launch(run, *, models=None, max_chunks=None):
    with (run / "execution.log").open("a", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print("\nEight-model screening queue — CUDA, one model at a time.")
            print("Ctrl+C saves and pauses. Wait for the saved message before closing.", flush=True)
            result = run_campaign(ROOT, run, models=models, max_chunks=max_chunks)
            print("Saved. Completed jobs will be verified and skipped when you resume.")
            print(f"Report: {run / 'RESULTS.md'}", flush=True)
    return result


def dashboard(run, *, status, start, read=input, write=print):
    while True:
        write("\nCNN architecture screen — L=4, conditioned g, GPU")
        write(status())
        write("1  Run / continue all unfinished models\n2  Progress and results\nq  Quit")
        try:
            choice = read("Choose: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            write("")
            return 0
        if choice in ("q", "quit", "exit"):
            return 0
        if choice == "1":
            start()
        elif choice == "2":
            write(status())
            write(f"Results: {run}/RESULTS.md")
        else:
            write("Choose 1, 2 or q.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="menu", choices=("menu", "prepare", "run", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, help="existing frozen campaign")
    parser.add_argument("--models", nargs="+", choices=VARIANT_NAMES, help="run only these variants")
    parser.add_argument("--max-chunks", type=int, help="pause after this many checkpoint chunks")
    args = parser.parse_args(argv)
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be positive")
    if args.models and args.command != "run":
        parser.error("--models is only used with run")
    if args.command == "menu" and not sys.stdin.isatty():
        parser.error("Interactive menu needs a terminal; use prepare, status or run explicitly")
    if args.run_dir:
        run = args.run_dir.resolve()
        load_plan(ROOT, run)
    else:
        config = json.loads(args.config.read_text())
        run = (ROOT / config["output_root"]).resolve()
        if args.command == "status":
            load_plan(ROOT, run)
        else:
            print("Verifying cached Hamiltonians and settings; preparation trains nothing.", flush=True)
            run = prepare(ROOT, config)
    if args.command in ("prepare", "status"):
        print(format_summary(summarize_campaign(ROOT, run)))
        print(f"Report: {run / 'RESULTS.md'}")
        return 0
    if args.command == "run":
        launch(run, models=args.models, max_chunks=args.max_chunks)
        return 0
    return dashboard(run, status=lambda: format_summary(summarize_campaign(ROOT, run)),
                     start=lambda: launch(run, max_chunks=args.max_chunks))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"CNN screening stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
