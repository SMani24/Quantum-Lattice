#!/usr/bin/env python3
"""Terminal entry point for preparing, running, resuming and inspecting samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from quantum_lattice.data.generation import (
    benchmark_config, inspect_run, prepare_run, run_generation, verify_run,
)
from quantum_lattice.data.campaign import dashboard, campaign_status


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/phase1_sample_generation.json"
DEFAULT_CAMPAIGN = ROOT / "configs/phase1_paper_domain_campaign.json"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command")
    campaign = commands.add_parser("campaign", help="select paper-domain datasets and see saved progress")
    campaign.add_argument("--config", type=Path, default=DEFAULT_CAMPAIGN)
    campaign.add_argument("--workers", type=int)
    campaign.add_argument("--status", action="store_true", help="print campaign status without running anything")
    campaign.add_argument("--plain", action="store_true", help="plain progress during generation")
    for name in ("run", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--run-dir", type=Path)
        command.add_argument("--workers", type=int)
        command.add_argument("--plain", action="store_true", help="plain progress, no keyboard controls")
        if name == "benchmark":
            command.add_argument("--sweeps", type=int, default=5000)
            command.add_argument("--jobs", type=int, default=4)
    for name in ("resume", "status", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "resume":
            command.add_argument("--workers", type=int)
            command.add_argument("--plain", action="store_true")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command is None:
        if not sys.stdin.isatty():
            parser().print_help()
            return 0
        return dashboard(ROOT, DEFAULT_CAMPAIGN)
    if arguments.command == "campaign":
        if arguments.status:
            print(json.dumps(campaign_status(ROOT, arguments.config), indent=2))
            return 0
        return dashboard(ROOT, arguments.config, workers_override=arguments.workers, interactive=not arguments.plain)
    if arguments.command == "status":
        _, _, progress = inspect_run(arguments.run_dir)
        print(json.dumps({"run_dir": str(arguments.run_dir), "completed": sum(p["done"] for p in progress.values()),
                          "total": len(progress), "instances": progress}, indent=2))
        return 0
    if arguments.command == "verify":
        print(json.dumps(verify_run(arguments.run_dir), indent=2))
        return 0
    original = None
    if arguments.command in {"run", "benchmark"}:
        original = json.loads(arguments.config.read_text())
        config = benchmark_config(original, arguments.sweeps, arguments.jobs) if arguments.command == "benchmark" else original
        if arguments.workers is not None:
            config["execution"]["workers"] = arguments.workers
        run_dir = prepare_run(ROOT, config, arguments.run_dir,
                              purpose="benchmark" if arguments.command == "benchmark" else "dataset")
    else:
        run_dir = arguments.run_dir.resolve()
    print(f"\nSaved run: {run_dir}\n", flush=True)
    if arguments.plain:
        print("Plain progress; Ctrl+C saves active chunks and exits.\n", flush=True)
    else:
        print("p pauses after active chunks are saved; r resumes; q or Ctrl+C saves and exits.\n", flush=True)
    result = run_generation(ROOT, run_dir, workers=arguments.workers, interactive=not arguments.plain)
    print(f"\n{result['state'].capitalize()}: {result['completed']}/{result['total']} Hamiltonians.")
    print(f"Session wall time: {result['elapsed_seconds']:.2f} seconds.")
    if arguments.command == "benchmark" and result["state"] == "complete":
        effective, _, progress = inspect_run(run_dir)
        compute_seconds = sum(p["compute_seconds"] for p in progress.values())
        benchmark_attempts = sum(p["attempted"] for p in progress.values())
        pilot_attempts = original["dataset"]["hamiltonians"] * original["sampler"]["total_sweeps"] * original["lattice"]["L"] ** 2
        startup_and_io = max(0, result["elapsed_seconds"] - compute_seconds / result["workers"])
        estimate = startup_and_io + compute_seconds / result["workers"] * pilot_attempts / benchmark_attempts
        timing = {**result, "benchmark_config": effective,
                  "pilot_config": original, "estimated_pilot_seconds": estimate,
                  "estimated_dataset_seconds": estimate,
                  "estimated_pilot_attempts": pilot_attempts,
                  "estimate_scope": "rough same-host extrapolation from short chains; startup and I/O approximated, full checkpoint/finalization costs and mixing not certified"}
        (run_dir / "benchmark.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"Rough configured-dataset estimate on this machine with {result['workers']} workers: {estimate:.1f} seconds.")
        print("This short run measures speed; it does not certify burn-in or mixing.")
    if result["state"] == "paused":
        print(f'Continue later: python experiments/phase1_sample_generation/run.py resume --run-dir "{run_dir}"')
    else:
        print(f"Dataset index: {run_dir / 'dataset.json'}")
        print(f"Per-Hamiltonian arrays: {run_dir / 'instances'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, ImportError, OSError, KeyError, TypeError) as error:
        print(f"\nCannot continue: {error}", file=sys.stderr)
        raise SystemExit(1)
