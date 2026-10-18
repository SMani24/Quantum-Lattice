#!/usr/bin/env python3
"""Train the matched observable-map CNN / MLP campaign on CUDA."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys

from quantum_lattice.models.observable_campaign import (
    format_summary, load_plan, prepare, run_campaign, summarize_campaign,
)
from quantum_lattice.models.observable_terminal import dashboard

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/observable_cnn_training.json"


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


def launch(run, *, sizes=None, max_chunks=None):
    with (run / "execution.log").open("a", buffering=1) as logfile:
        with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
            print("\nTraining one CUDA model at a time. Ctrl+C requests a checkpoint pause.")
            print("Wait for the pause message before closing the terminal.", flush=True)
            result = run_campaign(ROOT, run, sizes=sizes, max_chunks=max_chunks)
            print(format_summary(result))
            print(f"Report: {run / 'RESULTS.md'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="menu",
                        choices=("menu", "prepare", "run", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, help="existing frozen campaign")
    parser.add_argument("--sizes", nargs="+", type=int, choices=(3, 4),
                        help="run only these sizes; default frozen order is L4 then L3")
    parser.add_argument("--max-chunks", type=int, help="bounded checkpoint chunks, then pause")
    args = parser.parse_args(argv)
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be positive")
    if args.sizes and args.command != "run":
        parser.error("--sizes is only used with run")
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
            print("Verifying cached groups and frozen settings; preparation trains nothing.", flush=True)
            run = prepare(ROOT, config)
    if args.command == "prepare":
        print(format_summary(summarize_campaign(ROOT, run)))
        print(f"Prepared without training: {run}")
        return 0
    if args.command == "status":
        print(format_summary(summarize_campaign(ROOT, run)))
        print(f"Results: {run / 'RESULTS.md'}")
        return 0
    if args.command == "run":
        launch(run, sizes=args.sizes, max_chunks=args.max_chunks)
        return 0
    return dashboard(run,
                     status=lambda: format_summary(summarize_campaign(ROOT, run)),
                     launch=lambda **kw: launch(run, max_chunks=args.max_chunks, **kw))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Observable training stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
