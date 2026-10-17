# Matched MLP convergence check

Continue all 20 completed L=3,4 estimator models to check whether the original
100,000-update ceiling limited their comparison. Each continuation retains its
original model, Adam state, row permutation, CPU/CUDA RNG state, validation
history and best weights. This runs one model at a time on one GPU, with the
same Hamiltonians, paired seeds, architecture, batch size and optimizer.
The original campaign and feature banks remain intact.

From the repository root, use the existing shared environment:

```bash
.venv/bin/python experiments/estimator_convergence/run.py
```

Choose **1 — Run / continue** to start the queue, **2 — Progress and results**
to inspect it, or **q — Quit**. Press **Ctrl+C** during training to pause after
the current 100-update chunk. Repeat the command and choose Run / continue to
resume. CUDA is required for training; there is no CPU fallback and no additional
environment to install.

The ceiling is **500,000 total updates per model**, including its original
updates. That allows 400,000 additional updates for models that previously
reached 100,000, and 405,100 for the model that stopped at 94,900. All 20 models
restart their patience counter, including that previously stopped model. Early
stopping cannot activate before 150,000 total updates; afterwards it requires
100 validation checks without a qualifying improvement of at least 1e-6.
Validation still occurs every 100 updates, so patience covers 10,000 updates.
Best weights use the strictly lowest own-view validation MSE over **both** the
original and extended phases, independently of the patience threshold.

The original 20-model queue recorded about 78 minutes of optimizer and validation
time for approximately 2 million updates. Up to 8 million additional updates
suggest **roughly 5–6 hours** on the same GPU, including some checkpoint/export
overhead. This is an estimate rather than a measured wall-time guarantee;
early stopping, laptop power limits and other GPU workloads can change it.
Pausing is safe between sessions.

Explicit commands are available:

```bash
.venv/bin/python experiments/estimator_convergence/run.py prepare
.venv/bin/python experiments/estimator_convergence/run.py run
.venv/bin/python experiments/estimator_convergence/run.py status
```

`prepare` verifies and pins the parent models, feature manifests, source,
settings and environment without training. `status` (or `report`) summarizes
the saved continuation. To run only ten checkpoint chunks and then pause:

```bash
.venv/bin/python experiments/estimator_convergence/run.py run --max-chunks 10
```

This counts chunks across the sequential queue and preserves the declared
scientific settings. Use `--run-dir PATH` to select an existing frozen
continuation instead of the default configured run.

## Scope and files

The fixed parent is
`codex material/artifacts/estimator_mlp/matched-small-20261016/`.
Settings are in `configs/estimator_convergence.json`; shared continuation logic
is in `src/quantum_lattice/models/estimator_convergence.py`.

The study remains a **development validation comparison**, with validation used
to select checkpoints. Both estimators use the same saved Monte Carlo states;
no new Monte Carlo is needed. Selected models are evaluated on f, g and exact
validation inputs. Exact features are evaluation-only and never select models.
No original final-test payload or unopened confirmation bank is accessed.
The continuation does not establish final held-out accuracy, sign recovery,
Eq. D5 Hamiltonian distance or physical error-correction performance.

Ignored outputs live in
`codex material/artifacts/estimator_convergence/matched-long-20261017/`, exposed
through this experiment's `results/` shortcut:

- `plan.json` / `plan-sha256.json`: frozen continuation protocol and parent pins.
- `models/L*-seed*-f/` and `*-g/`: independent resumable checkpoints, full
  histories, selected weights, evaluations, predictions and completion records.
- `RESULTS.md`, `status.json`, `paired-validation.csv`: progress and comparisons.
- `execution.log`: saved terminal output.

Completed continuation models are verified and skipped when restarting. Keep
the original campaign and the shared `.venv` available until this continuation
and its review are complete. Changes to scientific settings or pinned files
require a separately prepared campaign.
