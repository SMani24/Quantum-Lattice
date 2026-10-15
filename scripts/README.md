# Utility scripts

Small operational helpers belong here, such as checking run manifests or collecting presentation exports. Physics and model calculations belong in the shared package; experiment-specific orchestration belongs in experiments.

`setup_environment.py` creates or reuses the single project `.venv`, installs
the pinned calibration/sampling/training dependencies in `requirements.txt`,
including CUDA PyTorch 2.7.1+cu126, and installs the editable shared package.
It does not upgrade dependencies
whose versions already satisfy the requirements. Run from any directory:

```sh
python3 /path/to/Quantum-Lattice/scripts/setup_environment.py
```

From the repository root, activate it with `source .venv/bin/activate` and use
`python` for all experiment runners. Python 3.10.12 matches this machine;
the pinned stack also has wheels for Python 3.11–3.13 on Linux/Windows x86-64.
Training uses the NVIDIA GPU; NumPy/Numba sampling still uses the CPU in this
same environment. A compatible NVIDIA driver is required. The wheel supplies
the CUDA runtime, so a separate CUDA toolkit installation is unnecessary.

To check the environment without installing or training anything:

```sh
python3 scripts/setup_environment.py --check
```

The check compiles a small Numba function and performs a real CUDA matrix
multiplication and backward pass. It fails clearly when the GPU is unavailable;
it does not silently substitute the CPU. A restricted sandbox must expose the
NVIDIA devices for this check. The check and training use
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA operations.
The [official PyTorch version table](https://pytorch.org/get-started/previous-versions/#v271)
documents the 2.7.1 CUDA 12.6 wheel index.

For an offline setup, first collect all wheels for `requirements.txt` on a
compatible internet-connected machine. Then pass their directory:

```sh
python3 scripts/setup_environment.py --wheelhouse /path/to/wheels
```

The old `.venv-mlp` is not needed by current runner commands. Installing the
unified environment does not change Monte Carlo data or model training policy.
