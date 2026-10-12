# Quantum Lattice

Research on reconstructing and correcting toric-code Hamiltonian perturbations from local observables and measurement snapshots. The starting point is Valenti et al., *Hamiltonian learning for quantum error correction* (2019).

The project scaffold is ready. The first implementation milestone is reproducing the numerical calibration in Figures 11 and 12; subsequent work compares the paper-style MLP with spatial CNNs and physics-based inference.

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

The authoritative checklist and next action are stored locally in codex material/project-management/ROADMAP.md and STATUS.md. FILE_STRUCTURE.md describes folder responsibilities, and DECISIONS.md records the adopted methodology. The research proposal and literature map are under codex material/research-proposals/.

## First milestone

See [the Phase 0 experiment description](experiments/phase0_figures_11_12/README.md) and its [configuration](configs/phase0_figures_11_12.json).

Phase 0 samples independent edge phase errors on a periodic square lattice. It is separate from ground-state Metropolis sampling. It will produce fresh numerical curves for seven lattice sizes and a new fourth-degree inverse fit, with presentation figures and reproducibility records.

The scaffold contains no experiment runner yet. Implementing and running Phase 0 is the next task.

## Reproducibility

Keep settings in configs, reusable calculations in src, and experiment orchestration in experiments. Every completed run should record its effective configuration, seed, software versions, numeric data, and generated outputs under the ignored local artifact folder. Keep snapshots and augmentations from one Hamiltonian in the same data split.
