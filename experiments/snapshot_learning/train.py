#!/usr/bin/env python3
"""Launch the resumable raw-configuration pilot from any working directory."""
import os
from pathlib import Path
import sys

# Set before NumPy/Torch import, including spawned inverse-worker interpreters.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quantum_lattice.models.snapshot_terminal import main

if __name__=='__main__':
    raise SystemExit(main(repo_root=ROOT))
