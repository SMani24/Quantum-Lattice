# Fixed-record training-data scaling

Train the unchanged mean CNN and raw-set CNN on nested subsets of the existing L4 training bank. This follows the completed raw-configuration pilot: inverse Ising was strongest, while the raw-set CNN's small average gain over the mean CNN varied with initialization.

From the repository root, use the shared environment:

```bash
.venv/bin/python experiments/snapshot_scaling/train.py
```

Choose **1 Run / continue**. The runner prepares one shared cache, verifies the existing inverse baseline, then trains all pending models in order. **Ctrl+C** saves at the next bounded checkpoint boundary and returns to the menu. Completed models are verified and skipped. **2 Progress / results** displays the saved run summaries. Neural training requires CUDA; preparation and status work on CPU. There is no CPU-training fallback or additional environment.

For unattended operation or a separate monitor:

```bash
.venv/bin/python experiments/snapshot_scaling/train.py --run
.venv/bin/python experiments/snapshot_scaling/train.py --status
```

`--prepare-only` prepares the data and immutable plan without training. Preparation is resumable between Hamiltonians. The first model binds the GPU and environment; use a fresh output directory when moving to another GPU. Scientific source/config/data changes also require a new campaign directory.

## Frozen comparison

Settings are explicit in [snapshot_scaling.json](../../configs/snapshot_scaling.json).

| Setting | Value |
|---|---|
| Geometry | Periodic square L4, 32 physical edge qubits |
| Training Hamiltonians | Nested 512, 2,048 and all 7,452 training instances |
| Validation | Same 128 development Hamiltonians throughout |
| Records per Hamiltonian | Same 256 saved records, including repeated configurations |
| Models | Mean CNN and raw-set CNN, unchanged 38,690 parameters each |
| Initialization | Five paired seeds; fresh training for every count and method |
| Queue | Three training counts × two methods × five seeds = 30 CNNs |
| Optimizer | Adam, learning rate 1e-4, batch 16 complete Hamiltonians |
| Stopping | At most 200 epochs; 30 grace / 30 patience, absolute min_delta 1e-6 |
| Selected model | Strict lowest own-run validation magnitude MSE, including epoch zero |
| Checkpoints | Every 50 optimizer updates; full optimizer/RNG/shuffle/partial-epoch state |
| Inverse control | Verified existing 256-record inverse-Ising fits, referenced once without refitting |

The same UID-hash ordering produces all training prefixes. Each UID retains the same record indices used in the pilot. All records/views of a Hamiltonian stay in its original split. True fields are training labels and scoring references; they are absent from the neural inputs. Final-test payloads are not read.

The mean CNN computes `rho(phi(mean(Z)))`; the raw-set CNN computes `rho(mean(phi(Z)))`. Both use circular 3×3 convolutions `2→32→64→32`, ReLU after each, then a nonlinear 1×1 readout `32→32→2` with a linear output. The set encoder uses 32-record chunks and activation recomputation. No normalization, residual connection, augmentation or clipping is introduced in this study.

## What the curve means

All counts use the same epoch/stopping policy. They receive **32, 128 and 466 optimizer updates per epoch**, respectively. Consequently, this is performance scaling under a common training policy, not a pure data-count comparison at equal compute. Initialization is paired across counts; shuffle permutations are paired within a count, not identical across different dataset lengths. Report stopping, caps and realized updates with the eventual scores.

Magnitude MSE is primary. D5 is secondary and uses explicitly identified reference signs; the inverse control retains its separate native signed D5. Seed averages are averages of errors, not ensemble predictions. The first two seeds repeat the earlier initialization settings; rerunning them does not create additional independent seeds.

These are reused validation groups selected for development. Five initialization seeds do not provide five independent training-bank draws. Saved MCMC records are correlated, and 256 records are not 256 effective independent samples. Complete edge sums are sufficient under the ideal fixed-support IID equilibrium likelihood; this comparison does not assume that raw sets contain extra information. A matched observable-input comparison and unopened-H confirmation remain later work.

The existing inverse baseline remains a single immutable reference at every training count. The runner verifies its plan/source/cache/output/instance hashes and exact validation states, fields, UIDs and record indices. It writes a reference binding in the new run, preserving the original fitted files. There are no repeated inverse fits or CPU-worker settings in this launcher.

## File structure

- `src/quantum_lattice/models/snapshot_scaling.py`: nested cache views, frozen plan, verified inverse reference and scheduling. It reuses the original tested training/checkpoint/scoring code without editing the pilot's pinned sources.
- `src/quantum_lattice/models/snapshot_scaling_terminal.py`: simple launch/progress menu.
- `experiments/snapshot_scaling/train.py`: thin entry point, usable from any working directory.
- `configs/snapshot_scaling.json`: explicit scientific and execution settings.

The default output is under ignored `codex material/artifacts/snapshot_learning/scaling/L4-M256-20261021/`:

- `cache/`: largest training bank, shared validation bags, source bindings and resumable per-H parts.
- `plan.json`, `plan-sha256.json`, `inverse-reference.json`: frozen tasks, model/settings/source/cache hashes and verified prior-baseline binding.
- `device-binding.json`: created at the first CUDA training run.
- `models/<task>/`: live/best checkpoints, history, selected predictions, evaluation and per-H metrics.
- `status.json`, `comparison.csv`, `RESULTS.md`: progress and completed scores. The comparison contains the inverse reference once.

Local setup reports and architecture follow-up notes are exposed through `experiments/snapshot_scaling/results/`. Generated outputs, planning notes and models remain ignored. Structural changes belong to a separately frozen comparison after reviewing this learning curve.
