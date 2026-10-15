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
from quantum_lattice.models import mlp_campaign, mlp_queue


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
    """Discover runs once, including queue children, without loading models."""
    records, seen = [], set()
    paths = (list(root.glob("*/status.json"))
             + list((root / "campaigns").rglob("status.json"))
             + list((root / "queues").rglob("status.json")))
    for path in sorted(paths):
        run = path.parent
        config = run / "config.json"
        if run.is_symlink() or not config.exists() or run.resolve() in seen:
            continue
        try:
            settings, status = json.loads(config.read_text()), json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if "mode" not in settings or "lattice_size" not in settings:
            continue
        records.append((run, settings, status))
        seen.add(run.resolve())
    return records


def _choose_number(count, prompt="Select number (Enter returns): "):
    choice = input(prompt).strip()
    if not choice:
        return None
    number = int(choice)
    if not 1 <= number <= count:
        raise ValueError("Number is outside the list")
    return number - 1


def _queue_child(path, root):
    parts = path.relative_to(root).parts
    return parts[0] == "queues" or (len(parts) > 1 and parts[0] == "campaigns" and parts[1].startswith("queue-"))


def _choose_run(workflow, *, unfinished=False):
    records = _saved_runs(workflow.artifact_root())
    if unfinished:
        records = [record for record in records
                   if record[2].get("state") not in mlp_campaign.FINISHED_STATES
                   and not _queue_child(record[0], workflow.artifact_root())]
    if not records:
        print("No matching MLP runs.")
        return None
    print("   L   mode         state         update    best validation MSE")
    for i, (path, config, status) in enumerate(records, 1):
        metric = status.get("best_validation_mse")
        best = "--" if metric is None else f"{float(metric):.6g}"
        print(f"{i:2} {config['lattice_size']:2}   {config['mode']:<12} {status.get('state', '?'):<13} "
              f"{status.get('step', 0):>7}   {best}")
    selected = _choose_number(len(records))
    return None if selected is None else records[selected][0]


def _show_run(run):
    print(mlp_campaign.format_run_summary(mlp_campaign.run_summary(run)))


def _choose_campaign(workflow, *, unfinished=False):
    records = []
    for path in sorted((workflow.artifact_root() / "campaigns").glob("*/campaign.json")):
        if path.parent.name.startswith("queue-"):
            continue
        saved = json.loads(path.read_text())
        if not unfinished or saved.get("state") != "trained":
            records.append((path, saved))
    if not records:
        print("No matching paired campaigns.")
        return None
    for i, (path, saved) in enumerate(records, 1):
        print(f"{i:2} L={saved['config']['lattice_size']}  {saved.get('state', 'unknown').upper():<10} {path.parent.name}")
    selected = _choose_number(len(records))
    return None if selected is None else records[selected][0].parent


def _show_campaign(campaign):
    saved = json.loads((campaign / "campaign.json").read_text())
    print(f"Paired campaign: {saved['state'].upper()} | {campaign}")
    for record in saved["runs"].values():
        path = campaign / record["path"]
        if (path / "config.json").exists():
            print()
            _show_run(path)


def _choose_queue(workflow, *, unfinished=False):
    records = []
    for path in sorted((workflow.artifact_root() / "queues").glob("*/queue.json")):
        summary = mlp_queue.queue_summary(path.parent)
        if not unfinished or summary.get("state") not in {"trained", "evaluated"}:
            records.append((path.parent, summary))
    if not records:
        print("No matching training queues.")
        return None
    print("   state          models    queue")
    for i, (path, summary) in enumerate(records, 1):
        done, total = summary.get("completed_models", 0), summary.get("total_models", "?")
        print(f"{i:2} {summary.get('state', '?'):<14} {done}/{total:<6} {path.name}")
    selected = _choose_number(len(records))
    return None if selected is None else records[selected][0]


def _show_queue(queue):
    print(mlp_queue.format_queue_summary(mlp_queue.queue_summary(queue)))


def _size_selection(text, ready):
    """An empty/all selection means verified ready banks, in canonical order."""
    if not text.strip() or text.strip().lower() == "all":
        sizes = tuple(size for size, item in ready.items() if item.get("state") == "complete")
    else:
        sizes = tuple(int(value) for value in text.replace(",", " ").split())
        if len(set(sizes)) != len(sizes) or not all(size in SIZES for size in sizes):
            raise ValueError(f"Choose distinct sizes from {SIZES}")
    if not sizes:
        raise ValueError("No verified complete datasets are ready")
    missing = [size for size in sizes if ready.get(size, {}).get("state") != "complete"]
    if missing:
        raise ValueError(f"Datasets not complete and verified: {missing}")
    return sizes


def _modes_selection(text):
    selection = text.strip().lower()
    if selection in {"", "both", "b"}:
        return ("paper_size", "all_edges")
    if selection in {"p", "paper_size"}:
        return ("paper_size",)
    if selection in {"a", "all_edges"}:
        return ("all_edges",)
    raise ValueError("Choose both, p (paper_size), or a (all_edges)")


def _continue_menu(workflow):
    print("1 Queue (all sizes)  |  2 Single run  |  3 Older paired campaign  |  Enter Back")
    choice = input("Continue: ").strip()
    if choice == "1":
        queue = _choose_queue(workflow, unfinished=True)
        if queue is not None:
            _show_queue(mlp_queue.run_queue(queue_dir=queue, resume=True))
    elif choice == "2":
        run = _choose_run(workflow, unfinished=True)
        if run is not None:
            workflow.train(run_dir=run, resume=True)
    elif choice == "3":
        campaign = _choose_campaign(workflow, unfinished=True)
        if campaign is not None:
            _show_campaign(mlp_campaign.train_both(None, campaign_dir=campaign, resume=True))
    elif choice:
        raise ValueError("Choose 1, 2, or 3")


def _results_menu(workflow):
    print("1 Queue overview  |  2 Individual model summary  |  3 Older paired campaign")
    print("4 Evaluate a finished queue on final test  |  Enter Back")
    choice = input("Results: ").strip()
    if choice == "1":
        queue = _choose_queue(workflow)
        if queue is not None:
            _show_queue(queue)
    elif choice == "2":
        run = _choose_run(workflow)
        if run is not None:
            _show_run(run)
    elif choice == "3":
        campaign = _choose_campaign(workflow)
        if campaign is not None:
            _show_campaign(campaign)
    elif choice == "4":
        queue = _choose_queue(workflow)
        if queue is not None:
            print("Opening final test for every frozen model in this finished queue.")
            mlp_queue.evaluate_queue(queue)
            _show_queue(queue)
    elif choice:
        raise ValueError("Choose 1, 2, 3, or 4")


def _advanced_menu(workflow, config, settings):
    print("1 Train one size  |  2 Evaluate frozen model on final test  |  3 Change settings  |  Enter Back")
    choice = input("Advanced: ").strip()
    if choice == "1":
        size = int(input(f"Lattice size {SIZES}: "))
        if size not in SIZES:
            raise ValueError(f"Choose one of {SIZES}")
        modes = _modes_selection(input("Mode [both / p / a, default both]: "))
        _show_queue(mlp_queue.run_queue(config, sizes=(size,), modes=modes))
    elif choice == "2":
        run = _choose_run(workflow)
        if run is not None:
            metrics = workflow.evaluate(run)
            print(f"Final test RMSE {metrics['rmse']:.6g}; MAE {metrics['mae']:.6g}")
            print(f"Results: {run / 'RESULTS.md'}")
    elif choice == "3":
        print("Settings apply only to new queues; saved runs keep their recorded settings.")
        policy = input("Stopping [e early / f fixed 10,000, Enter unchanged]: ").strip().lower()
        if policy:
            if policy not in {"e", "f"}:
                raise ValueError("Choose e or f")
            settings["stopping"] = "early" if policy == "e" else "fixed"
        device = input("Device [cpu / cuda / auto, Enter unchanged]: ").strip().lower()
        if device:
            if device not in {"cpu", "cuda", "auto"}:
                raise ValueError("Choose cpu, cuda, or auto")
            settings["device"] = device
    elif choice:
        raise ValueError("Choose 1, 2, or 3")


def _menu(workflow, config_path):
    settings = {}
    while True:
        try:
            config = _config(config_path, **settings)
            ready = mlp_queue.discover_ready_sizes(config)
            completed = [str(size) for size, item in ready.items() if item.get("state") == "complete"]
            pending = [str(size) for size, item in ready.items() if item.get("state") != "complete"]
            early = config["training"].get("early_stopping", {}).get("enabled", False)
            print("\nMLP training")
            print("Ready L: " + (", ".join(completed) or "none") + ("  |  not ready: " + ", ".join(pending) if pending else ""))
            print(f"New models: {'early stopping' if early else 'fixed budget'}; "
                  f"ceiling {config['training']['max_steps']:,} updates; {config['training']['device']}")
            print("1 Train all  |  2 Continue  |  3 Results  |  4 Advanced  |  q Quit")
            choice = input("Choose: ").strip().lower()
            if choice == "q":
                return
            if choice == "1":
                sizes = _size_selection(input("Sizes [Enter all ready, or e.g. 4,8,12]: "), ready)
                modes = _modes_selection(input("Modes [Enter both, p paper_size, a all_edges]: "))
                print(f"Training {len(sizes) * len(modes)} models sequentially; final test stays closed.")
                _show_queue(mlp_queue.run_queue(config, sizes=sizes, modes=modes))
            elif choice == "2":
                _continue_menu(workflow)
            elif choice == "3":
                _results_menu(workflow)
            elif choice == "4":
                _advanced_menu(workflow, config, settings)
            elif choice:
                raise ValueError("Choose 1, 2, 3, 4, or q")
        except (ValueError, FileNotFoundError) as error:
            print(f"Cannot perform this action: {error}")
        except KeyboardInterrupt:
            print("\nReturned to menu; completed chunks remain checkpointed.")


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
    training_all = commands.add_parser("train-all", help="Queue ready sizes, training one model at a time")
    training_all.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    training_all.add_argument("--sizes", type=int, nargs="+", choices=SIZES)
    training_all.add_argument("--modes", nargs="+", choices=("paper_size", "all_edges"), default=["paper_size", "all_edges"])
    training_all.add_argument("--device", choices=("cpu", "cuda", "auto"))
    training_all.add_argument("--stopping", choices=("early", "fixed"))
    training_all.add_argument("--max-steps", type=int)
    training_all.add_argument("--queue-dir", type=Path)
    training_all.add_argument("--plain", action="store_true")
    for name in ("resume-queue", "queue-status", "evaluate-queue"):
        queue_command = commands.add_parser(name)
        queue_command.add_argument("--queue-dir", type=Path, required=True)
        if name == "resume-queue":
            queue_command.add_argument("--plain", action="store_true")
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
        parser.exit(2, "PyTorch is missing. From the repository root run python3 scripts/setup_environment.py, then use .venv/bin/python.\n")
    try:
        if args.command is None:
            if not sys.stdin.isatty():
                parser.print_help()
                return 0
            _menu(workflow, DEFAULT_CONFIG)
        elif args.command == "train-all":
            config = _config(args.config, device=args.device, stopping=args.stopping, max_steps=args.max_steps)
            ready = mlp_queue.discover_ready_sizes(config)
            sizes = args.sizes or _size_selection("all", ready)
            _show_queue(mlp_queue.run_queue(config, sizes=sizes, modes=args.modes,
                                          queue_dir=args.queue_dir, interactive=not args.plain))
        elif args.command == "resume-queue":
            _show_queue(mlp_queue.run_queue(queue_dir=args.queue_dir, resume=True, interactive=not args.plain))
        elif args.command == "queue-status":
            _show_queue(args.queue_dir)
        elif args.command == "evaluate-queue":
            mlp_queue.evaluate_queue(args.queue_dir)
            _show_queue(args.queue_dir)
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
