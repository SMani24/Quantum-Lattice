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
from quantum_lattice.models import mlp_campaign


def _config(path, size=None, mode=None, device=None, stopping=None, max_steps=None):
    value = json.loads(Path(path).read_text())
    if size is not None and size != value["lattice_size"]:
        old = f"L{value['lattice_size']}/run"
        if not value["dataset_path"].endswith(old):
            raise ValueError("Custom dataset path: edit lattice_size and dataset_path together")
        value["dataset_path"] = value["dataset_path"][:-len(old)] + f"L{size}/run"
        value["lattice_size"] = size
    if mode is not None:
        value["mode"] = "paper_size" if mode == "both" else mode
    if device is not None:
        value["training"]["device"] = device
    if stopping is not None:
        value["training"].setdefault("early_stopping", {})["enabled"] = stopping == "early"
        if stopping == "fixed" and max_steps is None:
            value["training"]["max_steps"] = 10000
    if max_steps is not None:
        value["training"]["max_steps"] = max_steps
    return value


def _saved_runs(root):
    records = []
    paths = list(root.glob("*/status.json")) + list((root / "campaigns").rglob("status.json"))
    for path in sorted(paths):
        if path.parent.is_symlink():
            continue
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
        summary = mlp_campaign.run_summary(path)
        metric = lambda value: "--" if value is None else f"{float(value):.6g}"
        print(f"     train MSE {metric(summary['train_mse'])} | validation MSE {metric(summary['validation_mse'])} "
              f"| best {metric(summary['best_validation_mse'])} at {summary['best_step']}")
        print(f"     stop: {summary['stop_reason'] or 'not finished'} | "
              f"test: {'evaluated' if summary['test_evaluated'] else 'not evaluated'}")
    choice = input("Select run number (Enter cancels): ").strip()
    if not choice:
        return None
    number = int(choice)
    if not 1 <= number <= len(records):
        raise ValueError("Run number is outside the list")
    return records[number - 1][0]


def _show_run(run):
    print(mlp_campaign.format_run_summary(mlp_campaign.run_summary(run)))


def _choose_campaign(workflow):
    records = sorted((workflow.artifact_root() / "campaigns").glob("*/campaign.json"))
    if not records:
        print("No paired training campaign has been started yet.")
        return None
    for i, path in enumerate(records, 1):
        saved = json.loads(path.read_text())
        print(f"  {i}. {path.parent.name} | {saved.get('state', 'unknown').upper()}")
    choice = input("Select campaign number (Enter cancels): ").strip()
    if not choice:
        return None
    index = int(choice)
    if not 1 <= index <= len(records):
        raise ValueError("Campaign number is outside the list")
    return records[index - 1].parent


def _show_campaign(campaign):
    saved = json.loads((campaign / "campaign.json").read_text())
    print(f"Paired campaign: {saved['state'].upper()} | {campaign}")
    for record in saved["runs"].values():
        path = campaign / record["path"]
        if (path / "config.json").exists():
            print()
            _show_run(path)


def _menu(workflow, config_path):
    size = _config(config_path)["lattice_size"]
    stopping = "early" if _config(config_path)["training"].get("early_stopping", {}).get("enabled") else "fixed"
    while True:
        config = _config(config_path, size=size, stopping=stopping)
        ready = workflow.dataset_readiness(config)
        print(f"\nPaper-style MLP | L={size} | dataset {ready['state'].upper()}")
        if "completed" in ready:
            print(f"Hamiltonians saved: {ready['completed']}/{ready.get('total', '?')}")
        print("1  Train paper_size (one fixed-edge row per training Hamiltonian)")
        print("2  Train all_edges (additional-data comparison)")
        print("3  Resume a saved training run")
        print("4  Evaluate a finished frozen model on final test Hamiltonians")
        print("5  View saved runs and result paths")
        print("6  Train both modes back to back (paper_size, then all_edges)")
        print("7  Resume a paired training campaign")
        print(f"8  Stopping strategy: {stopping.upper()} (change)")
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
                # The workflow also prints a summary when called outside this menu.
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
                if choice == "5":
                    _show_run(run)
            elif choice == "6":
                _show_campaign(mlp_campaign.train_both(config))
            elif choice == "7":
                campaign = _choose_campaign(workflow)
                if campaign is not None:
                    _show_campaign(mlp_campaign.train_both(None, campaign_dir=campaign, resume=True))
            elif choice == "8":
                selected = input("e: early stopping | f: fixed 10,000 updates: ").strip().lower()
                if selected not in {"e", "f"}:
                    raise ValueError("Choose e or f")
                stopping = "early" if selected == "e" else "fixed"
        except (ValueError, FileNotFoundError) as error:
            print(f"Cannot perform this action: {error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name in ("train", "train-both"):
        training = commands.add_parser(name, help="Start new train/validation run(s)")
        training.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        training.add_argument("--size", type=int, choices=SIZES)
        training.add_argument("--mode", choices=("paper_size", "all_edges", "both"))
        training.add_argument("--device", choices=("cpu", "cuda", "auto"))
        training.add_argument("--stopping", choices=("early", "fixed"))
        training.add_argument("--max-steps", type=int)
        training.add_argument("--plain", action="store_true")
        if name == "train":
            training.add_argument("--run-dir", type=Path)
        training.add_argument("--campaign-dir", type=Path)
    for name in ("resume", "evaluate", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "resume":
            command.add_argument("--plain", action="store_true")
    paired_resume = commands.add_parser("resume-both", help="Continue a saved paired campaign")
    paired_resume.add_argument("--campaign-dir", type=Path, required=True)
    paired_resume.add_argument("--plain", action="store_true")
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
        elif args.command in {"train", "train-both"}:
            config = _config(args.config, args.size, args.mode, args.device, args.stopping, args.max_steps)
            if args.command == "train-both" or args.mode == "both":
                if getattr(args, "run_dir", None) is not None:
                    raise ValueError("Use --campaign-dir for paired training")
                _show_campaign(mlp_campaign.train_both(config, campaign_dir=args.campaign_dir, interactive=not args.plain))
            else:
                if args.campaign_dir is not None:
                    raise ValueError("--campaign-dir applies only to paired training")
                workflow.train(config, run_dir=args.run_dir, interactive=not args.plain)
        elif args.command == "resume":
            workflow.train(run_dir=args.run_dir, resume=True, interactive=not args.plain)
        elif args.command == "resume-both":
            _show_campaign(mlp_campaign.train_both(None, campaign_dir=args.campaign_dir,
                                               resume=True, interactive=not args.plain))
        elif args.command == "evaluate":
            metrics = workflow.evaluate(args.run_dir)
            print(f"Final test RMSE {metrics['rmse']:.6g}; MAE {metrics['mae']:.6g}")
            print(f"Results: {args.run_dir / 'RESULTS.md'}")
        else:
            _show_run(args.run_dir)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Cannot perform this action: {error}\n")
    except KeyboardInterrupt:
        print("\nStopped; any completed training chunks remain checkpointed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
