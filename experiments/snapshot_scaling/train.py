#!/usr/bin/env python3
"""Launch the fixed-record training-data scaling queue from any directory."""
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quantum_lattice.models.snapshot_scaling_terminal import main

if __name__ == "__main__":
    raise SystemExit(main(repo_root=ROOT))
