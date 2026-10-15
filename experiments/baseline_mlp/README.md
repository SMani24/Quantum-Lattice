# Paper-style magnitude MLP

The size-specific model maps `(mean A_u, mean A_v, mean A_u A_v)` for one edge to
`|b_z|`. It follows the supplied 3→128→150→128→1 dense ReLU architecture with a
linear final layer. There is no input normalization or output clipping. Signs
and iterative compensation are separate later steps.

Appendix C.1 explicitly lists L=3,4,8,12,16,20,24, with a separate network for
each size. Odd L is supported by the square toric-code stabilizers and sampler.

## Environment and launch

Use the project's **.venv** for sampling, calibration, and neural networks.
From the repository root, set it up or check it with:

```bash
python3 scripts/setup_environment.py
python3 scripts/setup_environment.py --check
source .venv/bin/activate
```

Then launch from the repository root:

```bash
python experiments/baseline_mlp/run.py
```

Or, from this experiment directory, without activating:

```bash
../../.venv/bin/python run.py
```

The main menu has four actions:

1. **Train all**: Enter selects all verified ready lattice sizes, then Enter
   selects both `paper_size` and `all_edges`. It creates one saved queue and
   trains one fresh model at a time, size by size. Type e.g. `3,8,12` to select
   a subset instead. With all seven banks ready, both modes mean **14 models**.
2. **Continue**: resume a saved queue, an independent run, or an older paired
   campaign. Completed queue models are skipped; unfinished models retain
   their saved settings and optimizer state.
3. **Results**: compact queue/model tables and selected summaries. A separate
   explicit action evaluates every frozen model in a finished queue on final
   test data. Training never calls that action.
4. **Advanced**: one-size training, individual final-test evaluation, and
   stopping/device settings for new queues. The main screen always shows the
   actual safety ceiling and stopping strategy from the selected configuration.

During training, `p` pauses, `r` resumes, and `q` saves and exits the **whole
queue**, including a key pressed during the last optimizer chunk. Ctrl+C stops
safely at the last completed checkpoint; at most the current 100-update chunk
is repeated after restart. No model is discarded. Resume uses the original
queue's immutable settings, frozen dataset identities, and completed models.
Pause remains in the running terminal; use `q` if closing the terminal or
restarting the computer. Loading verifies checksums and feature/label
identities, so its initial delay is data verification rather than training.
If a pause arrives during a model's final chunk, that model finishes and the
queue stops before the next model; select Continue from the menu to proceed.

The setup script installs the pinned CPU PyTorch dependency into the same
`.venv` while retaining the existing sampler packages. A preexisting
`.venv-mlp` is left intact as a fallback; new commands do not require it.
One CPU thread remains the reproducible default for this small network.
Other dependency versions are recorded per run. Exact continuation rejects
incompatible software/data/source changes; merely using a different virtual
environment path does not change the recorded environment identity.

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
policy and configured safety ceiling; their actual update counts can differ.
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

| Setting | Published template default |
|---|---:|
| `training.max_steps` | 200,000 safety ceiling |
| `early_stopping.min_steps` | 10,000 minimum updates |
| `early_stopping.patience` | 20 eligible validation checks |
| `early_stopping.min_delta` | 0.000001 absolute validation MSE |

The local configuration can override these template values; the main menu
shows the actual ceiling for new queues. Saved runs retain their recorded
ceiling when resumed.

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
# All ready sizes, both modes, one model at a time:
../../.venv/bin/python run.py train-all
# Selected sizes, or only one mode:
../../.venv/bin/python run.py train-all --sizes 3 8 12 --modes paper_size all_edges
../../.venv/bin/python run.py train-all --sizes 4 --modes all_edges --stopping fixed
../../.venv/bin/python run.py resume-queue --queue-dir '/absolute/queue/path'
../../.venv/bin/python run.py queue-status --queue-dir '/absolute/queue/path'
# Explicitly open final tests only after reviewing validation-selected models:
../../.venv/bin/python run.py evaluate-queue --queue-dir '/absolute/queue/path'
# Existing individual/paired commands remain available:
../../.venv/bin/python run.py train --size 4 --mode paper_size
../../.venv/bin/python run.py train-both --size 4
../../.venv/bin/python run.py resume-both --campaign-dir '/absolute/paired/campaign/path'
../../.venv/bin/python run.py resume --run-dir '/absolute/run/path'
../../.venv/bin/python run.py evaluate --run-dir '/absolute/run/path'
../../.venv/bin/python run.py status --run-dir '/absolute/run/path'
```

Starting a new run always creates a separate directory. Resume loads saved
configuration, environment, source hashes, data selection, Adam moments, best
weights, shuffle position and RNG. Source/data/environment changes reject exact
continuation. The verified bit-identical resume guarantee is within unchanged
CPU software, hardware and thread count, not across machines or device types.
Training and evaluation use a per-run process lock.

Queues save `queue.json`, `comparison.json`, `comparison.csv`, and `RESULTS.md`
under `queues/<id>/`. They preflight the selected ready banks and freeze their
metadata identities before starting. A failure or stop leaves a resumable queue;
completed models are never retrained by resume. Existing independent/paired
runs are preserved and are not silently imported into a new queue. For a new
comparison, all selected models therefore use the queue's recorded settings.
The comparison table labels training/validation values separately from final
test values: the two modes have different validation row selections, so their
validation MSEs alone are not matched final-test accuracy. Explicit queue
evaluation uses the same frozen all-edge test bank within each size, skips
already evaluated models, and can be continued by repeating the command.

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
- `src/quantum_lattice/models/mlp_queue.py`: ready-bank selection, sequential queues, comparisons, and explicit batch evaluation.
- `src/quantum_lattice/metrics/regression.py`: regression and group uncertainty.
- `src/quantum_lattice/plotting/mlp.py`: learning and final-test figures.

The thin `run.py` handles menu/arguments. All generated files are ignored under
`codex material/artifacts/phase1_mlp/`, exposed locally at **results/**. Each run
has a RESULTS.md, config, provenance, data selection, status, latest checkpoint,
history and validation-selected weights. `figures/` contains PNG/PDF exports.
Queues live under `queues/<id>/`; their records link each size/mode to its child
run. Two-mode queue children use the existing paired-campaign mechanism under
`campaigns/queue-<id>-L<size>/`, while single-mode children live within the queue.
The queue's comparison files and reports are the entry point for the whole batch.
Older paired campaigns live under `campaigns/<id>/`, with campaign.json recording
ordered child paths, immutable settings, dataset hashes and completion state.
Explicit evaluation adds prediction NPZ/CSV, per-Hamiltonian metrics CSV,
test metrics JSON and evaluation provenance. Raw configurations remain under
the sample-generation experiment's separate results. No general results folder
is created, and no research accuracy is implied by implementation smoke tests.
