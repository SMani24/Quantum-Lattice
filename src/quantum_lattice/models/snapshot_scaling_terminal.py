"""Small launch and progress menu for fixed-record training-data scaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

from . import snapshot_scaling as campaign


class ProgressLine:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.last_print = 0.0
        self.last_stage = None

    def __call__(self, event):
        stage = event.get("stage", event.get("state"))
        now = time.monotonic()
        if not self.stream.isatty() and stage == self.last_stage and now - self.last_print < 5:
            return
        self.last_print = now
        self.last_stage = stage
        if stage == "training":
            message = (f"{event['task']} | epoch {event['epoch']} | "
                       f"best MSE {event['best_mse']:.6f}")
        else:
            message = (f"Preparing shared records | {event.get('completed', 0)}/"
                       f"{event.get('total', '?')} Hamiltonians")
        prefix = "\r\033[K" if self.stream.isatty() else ""
        suffix = "" if self.stream.isatty() else "\n"
        self.stream.write(prefix + message + suffix)
        self.stream.flush()

    def finish(self):
        if self.stream.isatty():
            self.stream.write("\n")
            self.stream.flush()


def print_status(status, *, stream=None):
    stream = stream or sys.stdout
    stream.write(f"\nCNNs complete: {status['completed']}/{status['total']}\n")
    tasks = status.get("tasks", [])
    if not tasks:
        data = status.get("data_progress")
        if data:
            stream.write(f"Shared data: {data.get('completed', 0)}/{data.get('total', '?')} H\n")
        else:
            stream.write("Shared data will be prepared by Run / continue.\n")
    else:
        stream.write("Train H  Method       Seed  State       Best validation MSE\n")
        for task in tasks:
            mse = task.get("metrics", {}).get("mse")
            if mse is None:
                detail = f"epoch {task['epoch']}" if task.get("epoch") is not None else "—"
            else:
                detail = f"{mse:.6f}"
            method = "mean CNN" if task["method"] == "mean_cnn" else "raw-set CNN"
            seed = str(task["seed"])[-2:]
            stream.write(f"{task['train_hamiltonians']:<9}{method:<13}{seed:<6}"
                         f"{task['state']:<12}{detail}\n")
    reference = status.get("inverse_reference", {})
    mse = reference.get("metrics", {}).get("mse")
    if mse is not None:
        stream.write(f"Inverse baseline: verified reference, MSE {mse:.6f}; no refitting.\n")
    else:
        stream.write("Inverse baseline: verified during preparation; reused once.\n")
    stream.write(f"{status.get('shots', 256)} records/H; reused development validation; no final test.\n")


def main(argv=None, *, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    parser = argparse.ArgumentParser(description="L4 fixed-record training-data scaling on CUDA")
    parser.add_argument("--config", default="configs/snapshot_scaling.json")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true", help="Prepare and run every pending CNN")
    actions.add_argument("--prepare-only", action="store_true", help="Prepare data and plan without training")
    actions.add_argument("--status", action="store_true", help="Read progress and completed metrics")
    args = parser.parse_args(argv)
    path = Path(args.config)
    if not path.is_absolute():
        path = root / path
    config = campaign.validate_config(json.loads(path.read_text()))
    run = root / config["output_root"]
    line = ProgressLine()

    def execute():
        selection = config["selection"]
        print(f"\nL4 | training H {selection['train_hamiltonian_counts']} | "
              f"{selection['validation_hamiltonians']} validation H | {selection['shots']} records/H")
        print("All pending CNNs run in order. CUDA training; inverse baseline reused.")
        print("Ctrl+C saves progress and returns to the menu.")
        try:
            status = campaign.run_campaign(root, config, progress_callback=line)
        finally:
            line.finish()
        print_status(status)
        print("Files:", run)

    if args.status:
        print_status(campaign.progress(root, config))
        return 0
    if args.prepare_only:
        stopped = {"value": False}
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: stopped.update(value=True))
        try:
            campaign.prepare(root, config, should_stop=lambda: stopped["value"], progress=line)
        finally:
            signal.signal(signal.SIGINT, previous)
            line.finish()
        print_status(campaign.progress(root, config))
        return 0
    if args.run:
        execute()
        return 0

    while True:
        count = len(config['selection']['train_hamiltonian_counts']) * len(config['seeds']) * 2
        print(f"\nL4 training-data scaling | {config['selection']['shots']} records/H | {count} CNNs")
        print("1  Run / continue\n2  Progress / results\n0  Quit")
        try:
            choice = input("Choose: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        try:
            if choice == "0":
                return 0
            if choice == "1":
                execute()
            elif choice == "2":
                print_status(campaign.progress(root, config))
            else:
                print("Choose 0, 1 or 2.")
        except (ValueError, OSError, RuntimeError) as error:
            line.finish()
            print(f"\nStopped safely: {error}")
            print("Use the shared .venv and check nvidia-smi for CUDA issues. "
                  "Restore pinned files to resume; scientific changes require a fresh campaign.")


if __name__ == "__main__":
    raise SystemExit(main())
