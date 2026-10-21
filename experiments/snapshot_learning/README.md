# Raw-configuration learning pilot

Compare three ways to infer edge-field magnitudes from the **same saved Z-basis records**: a CNN on complete per-edge means, a CNN on a set of configurations, and a regularized inverse-Ising fit. This is a small L4 development experiment, using existing Monte Carlo outputs. It does not sample new states or open final-test data.

From the repository root, use the shared environment:

```bash
.venv/bin/python experiments/snapshot_learning/train.py
```

Choose **1 Run / continue**. Preparation is automatic and resumable. **Ctrl+C** saves a neural checkpoint after the current bounded update chunk, or finishes the bounded inverse fits already in flight, then returns to the menu. Completed tasks are verified and skipped. **2 Progress / results** shows a short summary; **3 CPU worker count** changes inverse-Ising parallelism without changing the scientific settings.

For unattended use:

```bash
.venv/bin/python experiments/snapshot_learning/train.py --run --workers 8
.venv/bin/python experiments/snapshot_learning/train.py --status
```

`--prepare-only` builds the frozen data bags and plan without fitting anything. Preparation/status need no GPU; neural training requires CUDA and has no CPU fallback. CPU workers affect only inverse-Ising fits. The first neural run binds the GPU and environment; resume on that binding. Moving to another GPU requires a fresh campaign directory. No additional packages or Python environment are needed.

## Inputs, methods and outputs

One example is **one Hamiltonian**, with `S` spin records and one field map. Each L4 record is represented by two4×4 channels: horizontal and vertical edge spins. Repeated configurations are retained. All subsets from one Hamiltonian stay in its original split.

- **Mean CNN:** average the S records to a `(2,4,4)` map, then apply a periodic spatial encoder and nonlinear readout.
- **Set CNN:** encode every `(2,4,4)` record using shared weights, average its32 feature maps across records, then apply the nonlinear readout. Shot order is irrelevant; spatial positions remain aligned.
- **Inverse Ising:** verify the trivial flux/winding sector, use `z_uv=tau_u*tau_v`, and jointly fit one shared coupling per edge from the vertex conditional probabilities. Damped Newton minimizes regularized pseudolikelihood. It receives no true fields or training labels and returns signed couplings; absolute values are used for the shared magnitude target. Solver convergence flags remain in the results.

The CNNs both have **38,690 parameters**. Their common encoder is circular3×3 convolutions `2→32→64→32`, each followed by ReLU. Their common readout is `1×1:32→32`, ReLU, `1×1:32→2` with a linear output. They differ in whether averaging occurs before or after the encoder. No BatchNorm, GroupNorm, residual connection, clipping or augmentation is included. Their initial weights and Hamiltonian shuffles match within each seed/budget.

The set model encodes32-shot chunks and recomputes encoder activations during backpropagation to bound memory. This changes computation/storage, not the intended mean-pooling operation. In floating point, shot permutations/chunking agree within checked tolerances rather than guaranteed bit identity.

## Default pilot

Settings are explicit in [snapshot_learning.json](../../configs/snapshot_learning.json):

| Setting | Value |
|---|---|
| Lattice | Periodic L4,32 physical edges |
| Data | 512 training /128 validation Hamiltonians, selected by a frozen UID hash |
| Record budgets |64 and256, nested subsets of the same4500 saved records/H |
| Neural runs |2 methods ×2 budgets ×2 paired seeds =8 |
| Inverse fits |128 validation H ×2 budgets =256 independent fits; no artificial seed repetition |
| Optimizer |Adam, learning rate1e-4,16 complete H/batch |
| Stopping |200 epochs maximum;30 grace,30 patience; absolute improvement threshold1e-6 |
| Selected weights |Strict lowest validation magnitude MSE, including epoch0 |
| Inverse solver |Ridge1e-3; up to100 Newton iterations; gradient tolerance1e-6 |
| Default CPU workers |8, adjustable in the menu or with `--workers` |

Each neural update averages magnitude error over whole Hamiltonians and all edges. Records are not treated as separate labeled Hamiltonians. Exact/b-dependent observable estimators are absent from these network inputs. True fields provide supervised labels and scoring references only. Field MSE is primary; all four reference-sign-assisted D5 conventions are secondary. The inverse solver also reports its own signed D5 separately.

Under the clean fixed-support IID equilibrium model, complete edge sums are sufficient for the likelihood. A raw model does not necessarily have more information than the mean control. This pilot compares practical representations; it does not prove a measurement-copy advantage. The saved records are from correlated MCMC chains, and record counts are not independent-sample counts.

This subset/budget pilot cannot be compared directly against the prior7452-H,4500-record observable-map models. More seeds, convergence, larger-lattice/context controls and unopened-H confirmation remain later work.

## Files

Shared implementations live in:

- `src/quantum_lattice/data/snapshot_data.py`: identity-bound, sector-checked, resumable raw bags.
- `src/quantum_lattice/models/snapshot_cnn.py`, `snapshot_training.py`: matched networks and exact-resume training state.
- `src/quantum_lattice/models/inverse_ising.py`: conditional objective, analytic derivatives and damped Newton solver.
- `src/quantum_lattice/models/snapshot_campaign.py`, `snapshot_terminal.py`: scheduling, checkpoints, metrics and menu.

The default campaign is under ignored `codex material/artifacts/snapshot_learning/campaigns/L4-pilot-20261021/`:

- `plan.json`, `plan-sha256.json`: immutable settings, tasks and source/cache hashes.
- `cache/`: original record/UID/field-label provenance, resumable per-H parts and final train/validation bags.
- `device-binding.json`: GPU/environment recorded at the first neural run.
- `models/<task>/`: live/best checkpoints, history, predictions, per-H metrics and completion hashes; inverse tasks also retain per-instance fits and convergence summaries.
- `status.json`, `comparison.csv`, `RESULTS.md`: concise progress and completed results.

Inspect the local `experiments/snapshot_learning/results/` shortcut for the setup handoff and current campaign. Do not edit scientific settings, source or data in a prepared run. To change them, keep the existing run and use a new `output_root` in a separate config passed with `--config`.
