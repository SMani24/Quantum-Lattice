#!/usr/bin/env python3
"""Create or check the shared CUDA .venv without upgrading satisfied packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".venv"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

SMOKE_CHECK = r"""
import importlib.metadata as metadata
import json
import os
import platform
import sys

# Set before importing Torch or creating a CUDA context, matching the trainer.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
import numba
import numpy as np
import torch
import quantum_lattice

@numba.njit(cache=False)
def compiled_sum(values):
    return values.sum()

assert compiled_sum(np.arange(4, dtype=np.int64)) == 6
torch.set_num_threads(1)
expected = {"numpy": "2.2.6", "matplotlib": "3.10.8", "numba": "0.63.1",
            "torch": "2.7.1+cu126"}
actual = {name: metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit("Installed versions differ from requirements.txt: " + repr(actual))
if not torch.cuda.is_available():
    raise SystemExit("CUDA training is required, but PyTorch cannot access an NVIDIA GPU. "
                     "Check nvidia-smi and run outside any sandbox that hides GPU devices. "
                     "No CPU fallback was used.")
torch.use_deterministic_algorithms(True)
weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", requires_grad=True)
inputs = torch.tensor([[1.0], [2.0]], device="cuda")
loss = (weights @ inputs).square().sum()
loss.backward()
torch.cuda.synchronize()
assert weights.is_cuda and weights.grad.is_cuda
assert torch.equal(weights.grad.cpu(), torch.tensor([[10.0, 20.0], [22.0, 44.0]]))
print(json.dumps({"python": platform.python_version(), "executable": sys.executable,
                  "versions": actual, "numba_compilation": "passed",
                  "torch_cuda_matmul_backward": "passed",
                  "cuda_runtime": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0),
                  "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}, indent=2))
"""


def run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the existing .venv without installing anything")
    parser.add_argument("--wheelhouse", type=Path, help="Install offline using wheels in this directory")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        parser.error("Use Python 3.10–3.13; Python 3.10.12 matches the existing research runs")
    if args.check and args.wheelhouse:
        parser.error("--wheelhouse is an installation option; omit it with --check")
    if not PYTHON.exists():
        if args.check:
            parser.error(".venv is missing; run this script without --check first")
        print(f"Creating {ENVIRONMENT}", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)
    if not args.check:
        command = [str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir"]
        requirements = ROOT / "requirements.txt"
        print("Installing missing pinned dependencies; satisfied versions are kept.", flush=True)
        if args.wheelhouse:
            wheelhouse = args.wheelhouse.resolve()
            if not wheelhouse.is_dir():
                parser.error(f"Wheel directory does not exist: {wheelhouse}")
            command += ["--no-index", "--find-links", str(wheelhouse)]
            # Requirements-file index options can override --no-index on older pip.
            # Exclude those options from the temporary offline requirements file.
            with tempfile.TemporaryDirectory(prefix="quantum-lattice-setup-") as temporary:
                offline_requirements = Path(temporary) / "requirements.txt"
                lines = requirements.read_text().splitlines()
                offline_requirements.write_text("\n".join(
                    line for line in lines
                    if not line.strip().startswith(("--index-url", "--extra-index-url"))
                ) + "\n")
                run(command + ["-r", str(offline_requirements)])
        else:
            run(command + ["-r", str(requirements)])
        run([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
             "--no-build-isolation", "--no-deps", "-e", str(ROOT)])
    run([str(PYTHON), "-c", SMOKE_CHECK])
    print("Ready. CUDA training and CPU sampling share this .venv.")
    print("  source .venv/bin/activate" if sys.platform != "win32" else r"  .venv\Scripts\activate")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
