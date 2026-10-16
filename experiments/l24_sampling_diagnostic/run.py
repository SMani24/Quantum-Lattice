#!/usr/bin/env python3
"""Run the separate checkpointed L24 sampling cross-check."""
import os
# Apply before importing NumPy in this coordinator or spawned workers.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import sys

from quantum_lattice.data.sampling_diagnostic import main

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    try:
        raise SystemExit(main(root=ROOT))
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
