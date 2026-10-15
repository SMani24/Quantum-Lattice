# Utility scripts

Small operational helpers belong here, such as checking run manifests or collecting presentation exports. Physics and model calculations belong in the shared package; experiment-specific orchestration belongs in experiments.

`setup_environment.py` creates or reuses the single project `.venv`, installs
the pinned calibration/sampling/training dependencies in `requirements.txt`,
and installs the editable shared package. It does not upgrade dependencies
whose versions already satisfy the requirements. Run from any directory:

```sh
python3 /path/to/Quantum-Lattice/scripts/setup_environment.py
```

From the repository root, activate it with `source .venv/bin/activate` and use
`python` for all experiment runners. Python 3.10.12 matches the retained runs;
the pinned stack also has wheels for Python 3.11–3.13. CPU Torch 2.7.1+cpu
preserves the current training implementation and checkpoint version checks.

To check the environment without installing or training anything:

```sh
python3 scripts/setup_environment.py --check
```

For an offline setup, first collect all wheels for `requirements.txt` on a
compatible internet-connected machine. Then pass their directory:

```sh
python3 scripts/setup_environment.py --wheelhouse /path/to/wheels
```

The existing `.venv-mlp` is kept locally as a backup. It is not needed by current
runner commands. Installing the unified environment does not change saved data,
model checkpoints, or their recorded training policy.
