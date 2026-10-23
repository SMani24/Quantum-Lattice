"""Small launch, preparation and progress menu for the snapshot ablations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import sys
import time

from . import snapshot_structure_campaign as campaign


class ProgressLine:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.last_print = 0.0
        self.last_task = None

    def __call__(self, event):
        task = (event.get("stage", "preparing"), event.get("task"))
        now = time.monotonic()
        if not self.stream.isatty() and task == self.last_task and now - self.last_print < 5:
            return
        self.last_print, self.last_task = now, task
        if event.get("stage") == "training":
            message = (f"{event['task']} | epoch {event['epoch']} | step {event['step']} | "
                       f"best MSE {event['best_mse']:.6f}")
        elif event.get("stage") == "diagnostics":
            message = (f"{event['task']} | gauge diagnostics "
                       f"{event.get('completed', 0)}/{event.get('total', '?')} masks")
        else:
            message = (f"Preparing shared records | {event.get('completed', 0)}/"
                       f"{event.get('total', '?')} Hamiltonians")
        self.stream.write(("\r\033[K" if self.stream.isatty() else "") + message
                          + ("" if self.stream.isatty() else "\n"))
        self.stream.flush()

    def finish(self):
        if self.stream.isatty():
            self.stream.write("\n")
            self.stream.flush()


def print_status(status, *, stream=None):
    """Summarize all ablation cells without requiring a CUDA context."""
    stream = stream or sys.stdout
    stream.write(f"\nCNNs complete: {status['completed']}/{status['total']}\n")
    tasks = status.get("tasks", [])
    if not tasks:
        data = status.get("data_progress")
        if data:
            stream.write(f"Shared data: {data.get('completed', 0)}/{data.get('total', '?')} H\n")
        else:
            stream.write("Shared data will be prepared by Run / continue or Prepare only.\n")
    else:
        groups = defaultdict(list)
        for task in tasks:
            groups[(task["method"], task["residual"], task["gauge_augmentation"])].append(task)
        stream.write("Method       Residual  Gauge  Complete  Mean validation MSE\n")
        for (method, residual, gauge), rows in sorted(groups.items()):
            completed = [row for row in rows if row["state"] == "complete"]
            errors = [row["metrics"]["mse"] for row in completed]
            mean = f"{sum(errors) / len(errors):.6f}" if errors else "—"
            label = "mean CNN" if method == "mean_cnn" else "raw-set CNN"
            stream.write(f"{label:<13}{'yes' if residual else 'no':<10}"
                         f"{'yes' if gauge else 'no':<7}{len(completed)}/{len(rows):<8}{mean}\n")
            if completed:
                seeds = ", ".join(f"{row['seed']}: {row['metrics']['mse']:.6f}"
                                  for row in sorted(completed, key=lambda row: row["seed"]))
                stream.write(f"  Completed seeds: {seeds}\n")
        for task in tasks:
            if task["state"] == "paused":
                stream.write(f"Paused: {task['key']} | epoch {task.get('epoch', '?')} | "
                             f"step {task.get('step', '?')}; resumes from its checkpoint.\n")
    reference = status.get("inverse_reference", {})
    mse = reference.get("metrics", {}).get("mse")
    if mse is None:
        stream.write("Inverse baseline: verified during preparation; reused without refitting.\n")
    else:
        stream.write(f"Inverse baseline: verified reference, MSE {mse:.6f}; no refitting.\n")
    stream.write(f"{status.get('shots', 256)} records/H; reused development validation; no final test.\n")
    stream.write("Mean errors average completed seeds only; incomplete arms are not final rankings.\n")


def _print_context(config, run):
    selection = config["selection"]
    count = (2 * len(config["factors"]["residual"])
             * len(config["factors"]["gauge_augmentation"]) * len(config["seeds"]))
    residual_levels = "/".join("on" if value else "off"
                               for value in config["factors"]["residual"])
    gauge_levels = "/".join("on" if value else "off"
                            for value in config["factors"]["gauge_augmentation"])
    print(f"\nPeriodic L{config['L']} | {selection['train_hamiltonians']} training H | "
          f"{selection['validation_hamiltonians']} validation H | {selection['shots']} records/H")
    print(f"Mean/raw-set × residual {residual_levels} × gauge {gauge_levels} × "
          f"{len(config['seeds'])} seeds "
          f"= {count} fresh CNN fits.")
    print("Training: CUDA only, shared .venv. Preparation and status work on CPU.")
    binding = run / "device-binding.json"
    if binding.exists():
        accelerator = json.loads(binding.read_text())["accelerator"]
        print("Bound GPU:", accelerator.get("name", accelerator.get("type", "unknown")))
    else:
        print("GPU binding: created by the first training run; use a fresh output for another GPU.")
    print(f"Checkpoint interval: {config['execution']['checkpoint_updates']} optimizer updates. "
          "Ctrl+C saves bounded progress; completed fits are verified and skipped.")
    print("Files:", run)


def main(argv=None, *, repo_root=None):
    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    parser = argparse.ArgumentParser(description="Residual/gauge snapshot-CNN ablations on CUDA")
    parser.add_argument("--config", default="configs/snapshot_structure.json")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true", help="Prepare and run every pending fit")
    actions.add_argument("--prepare", "--prepare-only", dest="prepare", action="store_true",
                         help="Prepare and verify the immutable queue without training")
    actions.add_argument("--status", action="store_true", help="Read progress and result summaries")
    args = parser.parse_args(argv)
    path = Path(args.config)
    if not path.is_absolute():
        path = root / path
    try:
        config = campaign.validate_config(json.loads(path.read_text()))
    except (ValueError, OSError) as error:
        print(f"Cannot load configuration: {error}", file=sys.stderr)
        return 2
    run = root / config["output_root"]
    line = ProgressLine()

    def show():
        _print_context(config, run)
        print_status(campaign.progress(root, config))

    def execute():
        _print_context(config, run)
        print("Running every pending fit in order. Ctrl+C saves and returns to the menu.")
        try:
            status = campaign.run_campaign(root, config, progress_callback=line)
        finally:
            line.finish()
        print_status(status)

    def prepare_only():
        _print_context(config, run)
        stopped = {"value": False}
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: stopped.update(value=True))
        try:
            campaign.prepare(root, config, should_stop=lambda: stopped["value"], progress=line)
        finally:
            signal.signal(signal.SIGINT, previous)
            line.finish()
        print("Preparation paused." if stopped["value"] else "Preparation complete; no training performed.")
        print_status(campaign.progress(root, config))

    def safely(action):
        try:
            action()
            return 0
        except (ValueError, OSError, RuntimeError) as error:
            line.finish()
            print(f"\nStopped safely: {error}")
            print("Use the shared .venv and check nvidia-smi for CUDA issues. "
                  "Restore pinned files to resume; scientific changes need a fresh campaign.")
            return 2

    if args.status:
        return safely(show)
    if args.prepare:
        return safely(prepare_only)
    if args.run:
        return safely(execute)

    while True:
        count = (2 * len(config["factors"]["residual"])
                 * len(config["factors"]["gauge_augmentation"]) * len(config["seeds"]))
        print(f"\nSnapshot structure | L{config['L']} | {count} CNN fits")
        print("1  Run / continue\n2  Progress / results\n3  Prepare only\n0  Quit")
        try:
            choice = input("Choose: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice == "0":
            return 0
        if choice in {"1", "2", "3"}:
            safely({"1": execute, "2": show, "3": prepare_only}[choice])
        else:
            print("Choose 0, 1, 2 or 3.")


if __name__ == "__main__":
    raise SystemExit(main())
