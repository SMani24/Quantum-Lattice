# CNN architecture screen

Run from the repository root with the existing shared environment:

```bash
.venv/bin/python experiments/cnn_ablation/train.py
```

Choose **1** to run the unfinished queue. Models train sequentially on CUDA;
Ctrl+C finishes the current bounded chunk, atomically saves, and returns to the
menu. Resume with the same command. Finished jobs are hash-verified and skipped.
There is no CPU fallback. Keep the pinned source, environment, cache and GPU
unchanged while a campaign is active.

The default [config](../../configs/cnn_ablation.json) screens eight models:
L4, conditioned `g` inputs, four architectures and two paired initialization/
shuffle seeds. Baseline and residual have 37,922 parameters; BatchNorm and
GroupNorm have 38,178. All share the baseline convolutions and circular padding.
The residual arm adds one parameter-free skip. Normalization follows each hidden
convolution, before ReLU; convolution biases remain in every arm. BatchNorm uses
training-only running statistics and frozen selected buffers for inference.
GroupNorm computes statistics across the whole lattice within channel groups.

Each model uses 7,452 training and 1,242 validation Hamiltonians from the existing
paired cache, based on 4,500 retained configurations per Hamiltonian. Adam lr1e-4,
batch 32 complete Hamiltonians, maximum 1,000 epochs; early stopping has 100 grace
epochs followed by 75 stale checks at absolute delta 1e-6. Strict-best own-view
validation MSE selects weights, including epoch 0. The fresh original baseline
gets this same policy. Old 500-epoch campaigns remain unchanged.

Architecture ranking waits for all eight jobs, then uses equal-seed mean `g`
validation MSE. Crossed `f/g/exact` magnitude errors and reference-sign-assisted
D5 are secondary. Exact inputs never train or select weights. The screen does
not establish final-test performance, optimal tuning or convergence. These
validation Hamiltonians have already been used for development. Review all
seeds, failures, caps and costs before freezing a shortlist and a separate
larger-size/estimator/untouched-Hamiltonian confirmation protocol.

Noninteractive commands:

```bash
.venv/bin/python experiments/cnn_ablation/train.py prepare
.venv/bin/python experiments/cnn_ablation/train.py run
.venv/bin/python experiments/cnn_ablation/train.py status
```

`run --models batchnorm groupnorm` restricts the queue without changing its
frozen panel. `run --max-chunks 1` saves after 100 updates; it is an actual
training command. A new configuration needs a distinct `output_root`; frozen
campaign settings cannot be silently edited.

All outputs are ignored under
`codex material/artifacts/cnn_ablation/campaigns/L4-g-screen-20261020/`:

- `plan.json`: protocol, architecture specs, data bindings and source/cache/GPU hashes.
- `status.json`, `RESULTS.md`, `paired-validation.csv`, `execution.log`: progress,
  complete-panel comparisons and console output.
- `models/<task>/latest.pt`: live/best weights and normalization buffers, Adam,
  CPU/CUDA/shuffle RNG, partial epoch and complete validation history.
- Finished jobs also contain `best.pt`, `history.json`, `evaluation.json`,
  raw `predictions.npz`, per-Hamiltonian magnitude/D5 tables and `done.json` hashes.

Local setup notes and verification are linked through `results/latest/`.
No new Monte Carlo bank, raw-snapshot model or final-test evaluation is run here.
