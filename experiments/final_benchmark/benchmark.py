#!/usr/bin/env python3
"""Bounded development-only actual-worker timing; all optimizer updates discarded."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from quantum_lattice.models.final_timing import run_timing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 24])
    parser.add_argument("--updates", type=int, default=20)
    args = parser.parse_args()
    run_timing(args.cache, args.output, sizes=args.sizes, updates=args.updates)
