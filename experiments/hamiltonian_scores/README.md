# D5 scores for frozen MLP campaigns

Evaluate the saved L3/L4 original and continued f/g models without retraining,
new Monte Carlo or GPU work. Use the existing shared environment from the
repository root:

```bash
.venv/bin/python experiments/hamiltonian_scores/run.py
```

The default settings in `configs/hamiltonian_scores.json` select all five seeds,
both f/g training views, and f/g/exact validation inputs from each frozen
campaign. Every invocation writes a **new** ignored run below
`codex material/artifacts/hamiltonian_scores/`, exposed here through `results/`.
`--config PATH` selects another declared matched pair; `--output PATH` selects
a new subfolder of that artifact root. Existing outputs are never overwritten.
The runs and their compact feature cache must be present locally; GitHub does
not contain these generated artifacts.

## Score conventions

The primary score is the raw, **unsquared** D5 distance between normalized
global Pauli coefficient vectors, using the two-channel Hamiltonian and
retaining the combined identity coefficient and fixed stabilizers. It has
range `[0,2]`; zero means equal normalized coefficient directions. There is
no paper-figure rescaling to maximum one.

The MLP predicts field magnitudes. For this simulation benchmark, the evaluator
explicitly grants reference signs:

```text
estimated_b_z = sign(true_b_z) * raw_prediction
```

This is a **reference-sign oracle**, not learned sign recovery. Negative raw
predictions remain negative before multiplication and can reverse the resulting
sign. Predictions are neither clipped nor replaced by their absolute values.
Exactly zero reference fields map to zero, granting knowledge of zero support.
True and estimated X fields are both known zero. The signed references retain
the original seeded float64 values; cached float32 magnitude labels are checked
exactly before scoring.

Sensitivity columns report two-channel/traceless, Z-branch/full and
Z-branch/traceless conventions. With zero X fields, both traceless scores must
agree; this is an invariance check, not independent evidence. Full Hamiltonian
scores are sensitive to the extensive identity coefficient, so cross-size
values alone do not establish better learning. A zero-field reconstruction
baseline uses the same conventions for context.

The [metric guide](../hamiltonian_metric/README.md) explains the derivation and
differences from the supplied legacy scripts. These scores do not claim exact
reproduction of the paper's historical Figure 4/6 values.

## Grouping, verification and outputs

Compute D5 separately for each Hamiltonian, then average Hamiltonians equally
and the five seeds equally. Do not compute D5 on averaged fields or interpret
correlated edges/seeds as independent observations. Matched f→f versus g→g and
all common-input contrasts are retained, including unfavorable ones. Validation
selected the saved checkpoints; this is development evidence rather than a
fresh held-out result. Exact features remain evaluation-only.

The evaluator checks frozen plans/source pins, parent bindings, all six
committed output hashes per model, and exact label/UID/edge alignment. A new
evaluation plan freezes conventions and input bytes before scoring; pre/post
snapshots confirm that neither research campaign changes. Compiled NumPy
batches merge identical global Pauli strings before normalization, preserving
the scalar metric while avoiding one Python expansion per Hamiltonian.

Each completed output contains:

- `plan.json`, `plan-sha256.json`, `config.json`: settings, oracle/identity
  conventions, source/environment and complete input hashes.
- `reference-fields.npz`: verified signed validation references and group IDs.
- `hamiltonian-scores.csv`: four D5 conventions and field errors per model,
  input view and Hamiltonian.
- `model-scores.csv`, `group-means.csv`: model means and equal-seed means.
- `paired-seeds.csv`, `contrasts.csv`, `parent-comparison.csv`: every paired
  comparison, including common inputs and earlier versus continued models.
- `zero-field-baseline.csv`: per-Hamiltonian non-neural baseline.
- `summary.json`, `RESULTS.md`, `done.json`: summary, readable report and
  completion/output hashes. Absence of `done.json` means an incomplete run.

Shared logic lives in `data/hamiltonian_scores.py`,
`metrics/hamiltonian_batch.py` and `metrics/hamiltonian_evaluation.py` below
`src/quantum_lattice/`. Existing training/estimator source pins, model files,
reference material and feature archives stay unchanged.
