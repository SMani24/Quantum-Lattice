#!/usr/bin/env python3
"""Terminal entry point; learning logic lives in quantum_lattice.models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
DEFAULT_CONFIG = REPO / "configs/phase1_mlp.json"
SIZES = (3, 4, 8, 12, 16, 20, 24)


def _config(path, size=None, mode=None, device=None):
    value = json.loads(Path(path).read_text())
    if size is not None and size != value["lattice_size"]:
        old = f"L{value['lattice_size']}/run"
        if not value["dataset_path"].endswith(old):
            raise ValueError("Custom dataset path: edit lattice_size and dataset_path together")
        value["dataset_path"] = value["dataset_path"][:-len(old)] + f"L{size}/run"
        value["lattice_size"] = size
    if mode is not None:
        value["mode"] = mode
    if device is not None:
        value["training"]["device"] = device
    return value


def _saved_runs(root):
    records = []
    for path in sorted(root.glob("*/status.json")):
        config = path.parent / "config.json"
        if config.exists():
            status = json.loads(path.read_text())
            settings = json.loads(config.read_text())
            records.append((path.parent, settings, status))
    return records


def _choose_run(workflow):
    records = _saved_runs(workflow.artifact_root())
    if not records:
        print("No MLP run has been started yet.")
        return None
    for i, (path, config, status) in enumerate(records, 1):
        print(f"  {i}. L={config['lattice_size']} {config['mode']} {status['state'].upper()} "
              f"{status['step']}/{status['max_steps']} updates | {path.name}")
    choice = input("Select run number (Enter cancels): ").strip()
    if not choice:
        return None
    number = int(choice)
    if not 1 <= number <= len(records):
        raise ValueError("Run number is outside the list")
    return records[number - 1][0]


def _menu(workflow, config_path):
    size = _config(config_path)["lattice_size"]
    while True:
        config = _config(config_path, size=size)
        ready = workflow.dataset_readiness(config)
        print(f"\nPaper-style MLP | L={size} | dataset {ready['state'].upper()}")
        if "completed" in ready:
            print(f"Hamiltonians saved: {ready['completed']}/{ready.get('total', '?')}")
        print("1  Train paper_size (one fixed-edge row per training Hamiltonian)")
        print("2  Train all_edges (additional-data comparison)")
        print("3  Resume a saved training run")
        print("4  Evaluate a finished frozen model on final test Hamiltonians")
        print("5  View saved runs and result paths")
        print(f"L  Select lattice size {SIZES}  |  q  Exit")
        choice = input("Choose: ").strip().lower()
        try:
            if choice == "q":
                return
            if choice == "l":
                selected = int(input("Lattice size: "))
                if selected not in SIZES:
                    raise ValueError(f"Choose one of {SIZES}")
                size = selected
            elif choice in {"1", "2"}:
                config["mode"] = "paper_size" if choice == "1" else "all_edges"
                workflow.train(config)
                print("Training does not open final test data. Review validation, then choose evaluation explicitly.")
            elif choice in {"3", "4", "5"}:
                run = _choose_run(workflow)
                if run is None:
                    continue
                if choice == "3":
                    workflow.train(run_dir=run, resume=True)
                elif choice == "4":
                    metrics = workflow.evaluate(run)
                    print(f"Final test RMSE {metrics['rmse']:.6g}; MAE {metrics['mae']:.6g}")
                print(f"Results: {run / 'RESULTS.md'}")
        except (ValueError, FileNotFoundError) as error:
            print(f"Cannot perform this action: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    training = commands.add_parser("train", help="Start a new train/validation run")
    training.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    training.add_argument("--size", type=int, choices=SIZES)
    training.add_argument("--mode", choices=("paper_size", "all_edges"))
    training.add_argument("--device", choices=("cpu", "cuda", "auto"))
    training.add_argument("--run-dir", type=Path)
    training.add_argument("--plain", action="store_true")
    for name in ("resume", "evaluate", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "resume":
            command.add_argument("--plain", action="store_true")
    args = parser.parse_args(argv)
    try:
        from quantum_lattice.models import mlp_experiment as workflow
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        parser.exit(2, "PyTorch is missing. Use the separate environment: ../../.venv-mlp/bin/python run.py\n")
    try:
        if args.command is None:
            if not sys.stdin.isatty():
                parser.print_help()
                return 0
            _menu(workflow, DEFAULT_CONFIG)
        elif args.command == "train":
            workflow.train(_config(args.config, args.size, args.mode, args.device),
                           run_dir=args.run_dir, interactive=not args.plain)
        elif args.command == "resume":
            workflow.train(run_dir=args.run_dir, resume=True, interactive=not args.plain)
        elif args.command == "evaluate":
            metrics = workflow.evaluate(args.run_dir)
            print(f"Final test RMSE {metrics['rmse']:.6g}; MAE {metrics['mae']:.6g}")
            print(f"Results: {args.run_dir / 'RESULTS.md'}")
        else:
            print((args.run_dir / "status.json").read_text())
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Cannot perform this action: {error}\n")
    except KeyboardInterrupt:
        print("\nStopped; any completed training chunks remain checkpointed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
