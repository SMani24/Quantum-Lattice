# Paper-style magnitude MLP

The size-specific model maps `(mean A_u, mean A_v, mean A_u A_v)` for one edge to
`|b_z|`. It follows the supplied 3→128→150→128→1 dense ReLU architecture with a
linear final layer. There is no input normalization or output clipping. Signs
and iterative compensation are separate later steps.

Appendix C.1 explicitly lists L=3,4,8,12,16,20,24, with a separate network for
each size. Odd L is supported by the square toric-code stabilizers and sampler.

## Environment and launch

Use the separate **.venv-mlp** environment. The existing .venv remains the
Monte Carlo environment. CPU PyTorch 2.7.1 is installed locally and checked.
From this experiment directory:

```bash
../../.venv-mlp/bin/python run.py
```

The menu shows dataset readiness, selects L and training mode, resumes saved
runs/campaigns, and performs final evaluation only when explicitly selected.
Option6 trains paper_size then all_edges; option7 resumes that saved sequence.
Option8 switches between early stopping and a fixed 10,000-update baseline.
Option5 lists validation summaries and shows the selected run's short summary,
stop reason, test status and RESULTS.md path. Summaries also print after training
or evaluation and are read from small JSON records, without loading a model.
`p` pauses,
`r` resumes, and `q` saves and exits. Ctrl+C keeps the last completed optimizer
chunk; at most the current 100-update chunk is repeated after restart. A completed
dataset with passing verification is required. Loading verifies checksums and
feature/label identities; the initial delay is data loading, not stalled training.

To recreate this environment from the repository root:

```bash
python3 -m venv --system-site-packages .venv-mlp
.venv-mlp/bin/python -m pip install 'setuptools>=64'
.venv-mlp/bin/python -m pip install -r requirements-mlp.txt --index-url https://download.pytorch.org/whl/cpu
.venv-mlp/bin/python -m pip install --no-build-isolation -e .
```

The CPU wheel follows [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/).
GPU support would require a compatible CUDA wheel in this separate environment;
the provided CPU installation cannot use CUDA. One CPU thread is the reproducible
default for this small network. Other installed dependencies are recorded per run.

## Protocol and row counts

For the configured L4 bank, the existing split is 7,452/1,242/1,242 independent
training/validation/test Hamiltonians. All 32 edges of a Hamiltonian remain in
the same group. Modes use the same frozen all-edge test bank:

| Mode | Training rows | Validation rows | Final test rows |
|---|---:|---:|---:|
| paper_size | 7,450 fixed-edge rows | 1,242 fixed-edge rows | 39,744 |
| all_edges | 238,464 rows | 39,744 rows | 39,744 |

`paper_size` chooses edge0 from the first 7,450 sorted training IDs; the remaining
two training Hamiltonians stay unused. `all_edges` uses every saved training edge.
Neighboring rows share a background and are correlated; they are not extra
independent Hamiltonians. With early stopping, both modes use the same stopping
policy and 200,000-update safety ceiling; their actual update counts can differ.
The fixed-budget option gives both 10,000 optimizer updates. Neither strategy
implies the same number of data passes. Actual processed examples, equivalent
row passes, best update and stop reason are recorded in status/results.

Defaults in [the config](../../configs/phase1_mlp.json): Adam learning rate 1e-5,
batch128, validation every100 updates, checkpoints every100 updates, deterministic
seeded shuffling. The visible author graph uses Adam at 1e-5. Batch size, stopping
budget, validation cadence, splits and fixed-edge policy are our choices because
the historical training loop is unavailable. PyTorch Adam is an adaptation, not
a bit-identical TensorFlow 1 replay. These are known-field Monte Carlo estimators,
not finite-shot experimental inputs; the historical phase filter is unavailable.

Early stopping is enabled by default. Its settings are:

| Setting | Default |
|---|---:|
| `training.max_steps` | 200,000 safety ceiling |
| `early_stopping.min_steps` | 10,000 minimum updates |
| `early_stopping.patience` | 20 eligible validation checks |
| `early_stopping.min_delta` | 0.000001 absolute validation MSE |

Every scheduled validation check updates the meaningful-improvement reference
if validation MSE decreases by more than min_delta. This resets patience. At and
after min_steps, a check without meaningful improvement increments patience;
20 consecutive such checks stop training. With the default100-update cadence,
that is roughly2,000 updates; the first eligible check is at10,000. A flat curve
can therefore stop at11,900. Small cumulative improvements can eventually reset
patience. The selected best weights still track every strict MSE improvement,
even one smaller than min_delta. The live progress bar/ETA refer to the ceiling,
not a known early-stop time. Adam's learning rate stays fixed.

The stopper's reference/counter are checkpointed, so pausing or restarting does
not reset patience. Final evaluation uses the validation-selected best weights,
including when training stops before the ceiling. The stopping settings are
our adaptation, not a recovered historical paper protocol.

Validation MSE alone selects the lowest-validation-MSE checkpoint. Training never
opens final-test payloads. After reviewing training, explicitly evaluate the
frozen best model. Test MSE/RMSE/MAE include fixed-edge and magnitude-bin summaries,
a constant training-mean baseline and whole-Hamiltonian bootstrap intervals.
Negative or above-domain outputs are reported without clipping. Test results
must not drive tuning; after test inspection, further tuning needs a new test bank.

## Commands

```bash
../../.venv-mlp/bin/python run.py train --size 4 --mode paper_size
../../.venv-mlp/bin/python run.py train --size 4 --mode all_edges
../../.venv-mlp/bin/python run.py train-both --size 4
../../.venv-mlp/bin/python run.py train --size 4 --mode paper_size --stopping fixed
../../.venv-mlp/bin/python run.py resume-both --campaign-dir '/absolute/paired/campaign/path'
../../.venv-mlp/bin/python run.py resume --run-dir '/absolute/path/printed/by/training'
../../.venv-mlp/bin/python run.py evaluate --run-dir '/absolute/path/printed/by/training'
../../.venv-mlp/bin/python run.py status --run-dir '/absolute/path/printed/by/training'
```

Starting a new run always creates a separate directory. Resume loads saved
configuration, environment, source hashes, data selection, Adam moments, best
weights, shuffle position and RNG. Source/data/environment changes reject exact
continuation. The verified bit-identical resume guarantee is within unchanged
CPU software, hardware and thread count, not across machines or device types.
Training and evaluation use a per-run process lock.

Paired campaigns create two fresh independent models with the same configured
initialization seed and unchanged Hamiltonian groups. Common dataset metadata
hashes are frozen so both modes cannot silently use different regenerated banks.
`q`/Ctrl+C during an unfinished model saves the campaign without starting the
next model; resume continues it and skips already finished models. Failed
uninitialized child files are preserved and retries use a fresh child path.
Each model has its own RESULTS.md. Final test evaluation stays a separate action.

The published `1b70fd6` fixed-budget runs remain readable/evaluable. Their saved
10,000-update budget and disabled early stopping are preserved; compatibility
is restricted to that exact source version and unchanged architecture/data logic.
Known paused runs can continue with recorded compatibility provenance. Use a
new run/campaign to apply the new stopping policy; resume never changes it.

## Files and results

Shared code:

- `src/quantum_lattice/data/mlp_data.py`: verified grouped loader; no raw state decompression.
- `src/quantum_lattice/models/mlp.py`: fixed architecture.
- `src/quantum_lattice/models/mlp_training.py`: Adam, shuffling, validation selection and snapshots.
- `src/quantum_lattice/models/mlp_experiment.py`: checkpoint/run/evaluation workflow.
- `src/quantum_lattice/models/mlp_terminal.py`: terminal progress and controls.
- `src/quantum_lattice/models/mlp_campaign.py`: saved summaries and paired training/resume.
- `src/quantum_lattice/metrics/regression.py`: regression and group uncertainty.
- `src/quantum_lattice/plotting/mlp.py`: learning and final-test figures.

The thin `run.py` handles menu/arguments. All generated files are ignored under
`codex material/artifacts/phase1_mlp/`, exposed locally at **results/**. Each run
has a RESULTS.md, config, provenance, data selection, status, latest checkpoint,
history and validation-selected weights. `figures/` contains PNG/PDF exports.
Paired campaigns live under `campaigns/<id>/`, with campaign.json recording
ordered child paths, immutable settings, dataset hashes and completion state.
Explicit evaluation adds prediction NPZ/CSV, per-Hamiltonian metrics CSV,
test metrics JSON and evaluation provenance. Raw configurations remain under
the sample-generation experiment's separate results. No general results folder
is created, and no research accuracy is implied by implementation smoke tests.
