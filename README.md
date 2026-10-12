# Quantum Lattice

Research on reconstructing and correcting toric-code Hamiltonian perturbations from local observables and measurement snapshots. The starting point is Valenti et al., *Hamiltonian learning for quantum error correction* (2019).

Phase 0 implements the independent-error calibration in Figures 11 and 12, with fresh Monte Carlo data, an inverse-polynomial fit, and presentation exports. Subsequent work compares the paper-style MLP with spatial CNNs and physics-based inference.

## Repository layout

| Path | Purpose |
|---|---|
| Reference Paper.pdf | Supplied reference paper |
| Reference Paper Codes/ | Preserved author implementation and checkpoints |
| src/quantum_lattice/ | Our shared geometry, simulation, observable, learning, inference, metric, and plotting code |
| configs/ | Versioned experiment settings and seeds |
| experiments/ | Experiment-specific runners and descriptions |
| scripts/ | Small operational utilities |
| tests/ | Physics, numerical, and pipeline checks |
| notebooks/ | Exploratory work using the shared package |
| codex material/ | Local roadmap, research notes, datasets, runs, figures, and model artifacts; excluded from Git |

## Run the calibration

See [the Phase 0 experiment description](experiments/phase0_figures_11_12/README.md) and its [configuration](configs/phase0_figures_11_12.json).

Use Python 3.10 or later. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=64'
python -m pip install -r requirements-phase0.txt
python -m pip install --no-build-isolation --no-deps -e .
python experiments/phase0_figures_11_12/run.py --config configs/phase0_figures_11_12.json
```

The default experiment draws 10,000 independent error patterns at each of 41 probabilities for seven lattice sizes. It exports numerical data, fit diagnostics, separate PDF/SVG/PNG figures, and a two-page presentation PDF under `codex material/artifacts/phase0_figures_11_12/<run-id>/`.

This samples Bernoulli edge errors, separate from ground-state Metropolis sampling. Original trial counts and raw calibration data are unavailable, so the configuration declares our reproduction choices.

To regenerate figures from an existing run, pass `--render-only "codex material/artifacts/phase0_figures_11_12/<run-id>"` to the same runner. Existing run directories cannot be overwritten by sampling.

## Checks

```sh
python -m unittest discover -s tests -v
```

The checks include exact small-lattice enumeration, periodic geometry, shared-edge covariance, seeded sampling, fitting, and the complete export pipeline. See [tests/README.md](tests/README.md).

## Reproducibility

Each run records its effective configuration, random streams, software versions, source hashes, numerical data, uncertainty checks, and generated outputs. Rendering records its own provenance without changing the sampling record. Research notes, process checklists, generated data, figures, and environments stay local and are excluded from Git.
