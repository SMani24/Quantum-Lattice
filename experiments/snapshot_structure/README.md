# Snapshot CNN structural comparison

This separate study tests residual connections and training-time vertex-gauge augmentation in the mean CNN and raw-set CNN. The existing raw pilot and training-data scaling campaign stay unchanged. These are controlled changes to the current models, not a new neural-network family or a demonstrated publication contribution.

Run from the repository root with the existing shared environment:

```bash
.venv/bin/python experiments/snapshot_structure/train.py
```

The menu has four actions: **1 Run / continue**, **2 Progress / results**, **3 Prepare only**, **0 Quit**. Run prepares and verifies the data, then trains every unfinished fit in sequence. Completed outputs are checked and skipped. The results view summarizes all eight arms, their completed seeds, validation errors and paused checkpoints; means over incomplete seed groups are explicitly provisional.

For unattended execution or read-only monitoring:

```bash
.venv/bin/python experiments/snapshot_structure/train.py --run
.venv/bin/python experiments/snapshot_structure/train.py --status
.venv/bin/python experiments/snapshot_structure/train.py --prepare
```

`--prepare-only` is an alias for `--prepare`; `--config` selects another explicit settings file. Neural training requires CUDA. Preparation and status work on CPU without creating a CUDA context. No additional Python environment or CPU-training fallback is introduced. The first training step binds the run to its GPU/environment. Use a **fresh campaign output directory** when moving to the headless PC or changing scientific source/configuration/data; preserve existing runs and their bindings.

**Ctrl+C saves progress at the next bounded checkpoint boundary**, by default at most 50 optimizer updates, then returns to the menu or exits `--run`. A full checkpoint retains the optimizer, model, neural and shuffle RNG states, cursor, history, partial-epoch accumulators and selected best weights. Augmentation masks regenerate exactly from their bound seed, epoch and Hamiltonian UID; they consume no neural/shuffle RNG. Preparation can pause between verification stages; gauge diagnostics pause between masks and can be repeated on resume. A forced kill can require repeating work since the most recent completed checkpoint. Training is left for the user to launch; no official long job is part of setup verification.

## Factorial design

Settings live in [snapshot_structure.json](../../configs/snapshot_structure.json).

| Factor or setting | Default |
|---|---|
| Geometry | Periodic square L4, 32 physical edge qubits |
| Training/validation | 2,048 training / the same 128 development validation Hamiltonians |
| Records | The same 256 selected records per Hamiltonian, preserving repeated observations |
| Representation | Complete edge means or raw configuration set |
| Residual connection | Off / on |
| Vertex-gauge augmentation | Off / on during training only |
| Repeats | Five paired initialization seeds |
| Queue | 2 representations × 2 residual settings × 2 augmentation settings × 5 seeds = **40 fresh fits** |
| Optimizer | Adam, learning rate 1e-4, batch 16 complete Hamiltonians |
| Stopping/selection | Maximum 200 epochs; 30 grace / 30 patience, delta 1e-6; retain strict lowest validation magnitude MSE |
| Checkpoints | Every 50 optimizer updates by default |
| Inverse control | Verify and reuse the prior 256-record inverse-Ising reference once; no new fits |
| Gauge diagnostics | 16 fixed validation masks, separate from canonical checkpoint selection |

Each model starts afresh, including the unmodified baseline cells. Initial convolution tensors and data ordering are paired where the designs allow. All records, augmented views and edge targets for a Hamiltonian remain in its assigned split. The 2,048 training Hamiltonians are a prefix view of the immutable scaling cache, with its existing deterministic order and selected records. Preparation verifies and references that cache without rereading original raw chains or making another copy. The validation set is reused development data, not unopened confirmation. This is a common-policy comparison; early stopping can produce different realized update counts and compute.

For a reduced **20-fit study**, copy the configuration and set exactly one factor to `[false]`, keeping the other `[false, true]`. Keeping gauge augmentation off tests residual alone; keeping residual off tests augmentation alone. Preserve both representations and all five seeds. Give the copy a **fresh `output_root` before preparation**, then select it with `--config`. The terminal displays the configured levels. This reduced design estimates the chosen factor's contrast with the other factor off; it provides no residual-by-augmentation interaction estimate. Keep any already prepared 40-fit plan unchanged.

## What changes in the networks

The mean model processes `rho(phi(mean(Z)))`; the raw-set model processes `rho(mean(phi(Z)))`. Both retain periodic spatial coordinates and predict two edge-magnitude maps. Their shared scaffold has circular 3×3 convolutions `2→32→64→32`, ReLU, followed by nonlinear 1×1 readout `32→32→2` and a linear output. There is no downsampling or final clipping. For the set model, record chunks and activation recomputation manage memory while retaining every selected observation.

The residual arm adds the first hidden 32-channel activation to the third convolution's 32-channel output before its ReLU. It adds no learned parameters. Baseline and residual models therefore retain **38,690 parameters**. The identity shortcut is an established architecture; the experiment asks whether it improves this specific inference task, rather than assuming it will.

For gauge augmentation, draw vertex signs `eta_v∈{−1,+1}` with one root sign fixed to remove global redundancy, and transform physical edge records by

```text
z_uv → eta_u * eta_v * z_uv
```

One mask is shared across **all records of a Hamiltonian's bag** for that training appearance. This is not a separate random flip for every edge or record. Products around closed loops remain unchanged. Under the clean model the corresponding coupling changes as `b_uv→eta_u*eta_v*b_uv`, so the **magnitude target remains unchanged**. No true fields are needed to apply the augmentation. Randomly masking each record independently would generally mix distinct underlying Hamiltonians and is not this procedure. At validation, inputs remain in their original orientation; augmentation does not use validation labels or introduce a new observed record budget.

Residual and augmentation switches are crossed, not introduced together without controls. The four arms per representation are unchanged, residual only, gauge only, and both. A benefit in one cell does not establish synergy or a universal advantage. Gauge transformations describe a symmetry of the declared equilibrium family; finite correlated MCMC histories do not automatically inherit an exact trajectory-distribution invariance argument.

Fixed-mask validation diagnostics separately examine sensitivity to gauge transformations. They do not change the canonical validation inputs used for checkpoint selection, and are not a demonstration of exact learned invariance. Their seed and mask count are explicit in `diagnostics`.

## Scores and interpretation

Magnitude MSE is primary; all neural predictions are scored raw. D5 is secondary with explicitly identified reference signs. The old inverse-Ising control also retains its separate native signed result. The neural methods predict magnitudes and do not learn those reference signs. True fields are supervised labels/scoring references, absent from input and augmentation logic.

In the ideal IID fixed-support clean distribution, complete edge sums are sufficient statistics. The raw-set CNN therefore cannot claim extra information merely because it sees individual records. Saved MCMC observations are correlated; record count is not effective sample size. A paired seed is an optimization repeat, not an independent dataset draw. At L4 the nominal 7×7 receptive field spans the small torus, so this run establishes no large-lattice or locality advantage.

A stronger neural result, inverse baseline improvement, cross-geometry transfer, physical correction or publication novelty remains to be demonstrated. The current inverse baseline leads the completed raw pilot. Other lattice types and other network families are outside this structural comparison.

## File map

- `src/quantum_lattice/models/snapshot_structure_campaign.py`: immutable preparation, task/ablation bindings, inverse-reference verification, scheduling and summaries.
- `src/quantum_lattice/models/snapshot_structure_terminal.py`: the small menu, CLI and grouped terminal summaries.
- `experiments/snapshot_structure/train.py`: thin runner usable from any working directory.
- `configs/snapshot_structure.json`: scientific and operational choices.

Generated files live under the ignored `output_root` named in the configuration, below `codex material/artifacts/snapshot_structure/campaigns/`. `plan.json` and its binding freeze the tasks/source/settings and references to the prior scaling cache; the existing training/validation bags remain in their source campaign. The verified inverse reference also remains in its original campaign. `device-binding.json` appears at the first CUDA run. `models/<task>/` contains live/best checkpoints, histories, selected predictions, evaluations and per-Hamiltonian errors. `status.json`, `comparison.csv` and `RESULTS.md` collect completion and scores.

Local preparation evidence and the campaign are exposed through this experiment's ignored `results/` shortcuts. Each phase keeps its own results directory. Source/configuration/operational documentation may be published; data, checkpoints and local process material remain excluded from Git.
