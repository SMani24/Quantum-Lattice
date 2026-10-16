# Matched estimator MLP batch

Compare original (`f`) and conditioned (`g`) simulated observables while keeping
the paper-style MLP, Hamiltonian groups, local input order and paired seeds fixed.
The batch runs **20 fresh CUDA models**: L=3,4 × five paired seeds × two views.
It uses existing saved non-test summaries, so no Monte Carlo must be repeated.
Original models and banks stay intact.

From the repository root, use the shared environment:

```bash
.venv/bin/python experiments/estimator_mlp/run.py
```

The terminal menu offers **Run / continue**, **Progress and results**, and **Quit**.
Models train one after another on one GPU. **Ctrl+C pauses after the current
100-update chunk**; choose Run / continue, or repeat the command, to resume.
Checkpoints retain Adam state, row permutation, CPU/CUDA RNG states, validation
history, stopping counters and the best weights. CUDA is required for training,
with no CPU fallback. Explicit `run` checks CUDA before cache preparation; the
menu and `prepare` may cache summaries without a GPU, while the menu’s Run option
checks CUDA before loading training feature arrays.

Explicit commands are also available:

```bash
.venv/bin/python experiments/estimator_mlp/run.py prepare
.venv/bin/python experiments/estimator_mlp/run.py run
.venv/bin/python experiments/estimator_mlp/run.py status
.venv/bin/python experiments/estimator_mlp/run.py report
```

`prepare` caches train/validation feature summaries and freezes source, input,
environment, settings and task identities. It performs no optimizer updates.
Preparation may take a few minutes the first time. To time a short, resumable
training segment without launching the entire batch:

```bash
.venv/bin/python experiments/estimator_mlp/run.py run --max-chunks 10
```

That limit counts checkpoint chunks across the sequence, then saves and stops.
It does not alter the scientific plan. An existing run can be selected with
`--run-dir PATH`; its saved plan supplies settings.

## Fixed protocol

Each local input is `(<A_u>, <A_v>, <A_u A_v>)`; the label is `abs(b_z)`.
True fields are not neural-network inputs. Training uses all edges of the
existing training Hamiltonians. Hamiltonian identities remain separate across
training and validation. Both estimators use the same saved configurations.

The architecture is 3 → 128 ReLU → 150 ReLU → 128 ReLU → 1 linear. Adam uses
learning rate 1e-5, batch size 128 and at most 100,000 updates. Validation occurs
at the first update, every 100 updates and the ceiling. Early stopping uses
20-check patience, absolute minimum improvement 1e-6 and a 10,000-update minimum.
Each model retains its strictly lowest own-view validation MSE checkpoint;
patience and best-weight selection are separate. Equal policy can produce
different actual stopping steps, which are reported.

Each selected `f` or `g` model is evaluated on **all three validation views**:
`f`, `g`, and exact expectations. Exact data never train or select a model.
The matched `f→f` versus `g→g` comparison describes the full simulated pipeline;
common exact inputs compare the training variants without evaluation-input
sampling noise. Every seed is retained. Metrics are raw, unclipped MSE/RMSE/MAE,
with per-Hamiltonian records and aligned predictions for later paired analysis.
Saved `elapsed_seconds` counts optimizer advances and validation, excluding data
loading, checkpoint writes and export time; it is not end-to-end wall time.
The progress summary also shows equal-seed mean pipeline MSEs and the common
exact-input difference, using only completed f/g pairs. Negative differences favor
the g-trained pipeline or model respectively; these comparisons remain descriptive.

All results are **development validation results**, since validation also selects
checkpoints. No original final-test payload or fresh confirmation bank is opened.
This is magnitude regression, not sign recovery, D5 Hamiltonian distance or a
logical-error benchmark. The known-field conditioned estimator has no direct
claim of experimental measurement-copy savings.

## Files

Source is in `src/quantum_lattice/models/estimator_mlp.py`; the feature adapter is
`src/quantum_lattice/data/estimator_features.py`. Shared training remains in
`models/mlp_training.py`; settings live in `configs/estimator_mlp.json`.

Ignored outputs live under
`codex material/artifacts/estimator_mlp/matched-small-20261016/` and are exposed
through this experiment's `results/` shortcut:

- `plan.json` / `plan-sha256.json`: immutable protocol, task list and pins.
- `features/`: versioned f/g/exact arrays and source manifests.
- `models/L*-seed*-f/` and `*-g/`: `latest.pt` pause state; completed jobs add
  selected `best.pt`, `history.json`, `evaluation.json`, `per-hamiltonian.csv`,
  `predictions.npz` and a hash-verified `done.json` commit marker.
- `RESULTS.md`, `status.json`, `paired-validation.csv`: progress and comparisons.
- `execution.log`: saved terminal output.

Finished models are verified and skipped on continuation. Changing scientific
source, arrays, environment, settings or the declared seed list requires a new
campaign; existing outputs are never silently adapted or overwritten.
