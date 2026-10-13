# Quantum Lattice

Research on reconstructing and correcting toric-code Hamiltonian perturbations from local observables and measurement snapshots. The starting point is Valenti et al., *Hamiltonian learning for quantum error correction* (2019).

Phase 0 implements the independent-error calibration in Figures 11 and 12, with fresh Monte Carlo data, an inverse-polynomial fit, and presentation exports. Subsequent work compares the paper-style MLP with spatial CNNs and physics-based inference.

Phase 1 now has a checkpointable solvable-model sample generator with shared
star/pair estimates, CPU workers, and terminal pause/resume controls. Start with
the [short benchmark and runner instructions](experiments/phase1_sample_generation/README.md).
The dashboard selects full paper-domain datasets for seven lattice sizes,
starting with L=4, and shows saved completion/resume status. A separate weak-field
pilot remains available. The [paper-style MLP runner](experiments/baseline_mlp/README.md)
uses a separate .venv-mlp environment, Hamiltonian group splits, validation-selected
checkpoints, and explicit final-test evaluation. The full Phase 1 study remains ahead.

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

## Find the local results

Each phase has its own local results directory beside its runner. For Phase 0, start at `experiments/phase0_figures_11_12/results/README.md`. The detailed report is `results/latest/RESULTS.md` within that experiment; the presentation is `results/latest/presentation/figures_11_12_presentation.pdf`. Separate plots are in `results/latest/figures/`, numerical data in `results/latest/data/`, and fit diagnostics in `results/latest/fit_metrics.json`.

The per-phase `results/` directory is a local shortcut to that phase's retained artifacts under `codex material/artifacts/`. `latest/` selects the run being presented; timestamped runs stay available alongside it. Outputs are kept in one place and intentionally excluded from GitHub. After each phase, the report explains the method, findings, limitations, and file locations.

## Lattice topology

`lattice/topology.py` defines immutable `Vertex`, `Edge`, `Face`, and `Lattice` objects. Connectivity can be inspected through object properties; numerical star/plaquette calculations use read-only incidence tables. Sample flags stay separate, in arrays indexed by edge ID.

```python
import numpy as np
from quantum_lattice.lattice import square_lattice

lattice = square_lattice(4)
vertex = lattice.vertices[0]
print([edge.id for edge in vertex.incident_edges])
print(lattice.edges[0].endpoints)
print(lattice.faces[0].boundary_edges)

z_errors = np.zeros((100, lattice.n_edges), dtype=bool)
z_errors[:, 0] = True
star_defects = lattice.star_syndrome(z_errors)  # shape (100, lattice.n_vertices)
```

The square constructor preserves the original edge order: `edge_id = 2*(y*L+x)+orientation`, with horizontal 0 and vertical 1. Phase 0 builds one lattice per size and reuses it across all probabilities and batches. Its optional prebuilt `lattice=` argument must use this square-factory convention; metadata alone does not establish that ordering.

Other cell complexes can be supplied to `Lattice(n_vertices, edge_endpoints, face_boundaries, ...)`. Faces use ordered closed trails of distinct edges. Closed/periodic labels require two faces per edge; these are incidence checks, not a proof of embedding or manifold genus. Star and plaquette supports must intersect evenly. For physical channels, Z-error flags give star defects; X-error flags give plaquette defects.

`metrics/lattice_calibration.py` derives independent-error means and covariance from connectivity, including unequal vertex degrees. `run_lattice_calibration_point()` samples these generic topologies. Only the periodic square constructor and square Figure 11/12 experiment are provided so far; the paper's D4 coefficients remain square-specific.

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

The checks include exact small-lattice enumeration, topology navigation/immutability, periodic geometry, star–plaquette commutation, shared-edge covariance, seeded sampling, fitting, and the complete export pipeline. See [tests/README.md](tests/README.md).

## Reproducibility

Each run records its effective configuration, random streams, software versions, source hashes, numerical data, uncertainty checks, and generated outputs. Rendering records its own provenance without changing the sampling record. Research notes, process checklists, generated data, figures, and environments stay local and are excluded from Git.
