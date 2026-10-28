# Final multi-size benchmark

This is the fixed-method final campaign, not another architecture search.
Generated inputs, checkpoints, logs and results stay under the Git-ignored
`codex material/artifacts/final_benchmark/` subtree.

## Scientific design

- All reference learning sizes: **L=3,4,8,12,16,20,24**.
- Four arms: plain periodic observable CNN / map-wrapped paper MLP, each with
  original `f` / orbit-conditioned `g` features. No new normalization,
  residual, gauge or attention trials.
- Five fresh paired optimizer seeds: **140 new fits** in one primary cohort.
  Earlier development models remain intact and are not pooled as extra repeats.
- Each size uses **7,452 training and 1,242 development Hamiltonians**, with
  4,500 retained records/H. Both architectures receive the same whole-H batches.
  This is a different exposure from the paper's 7,450 local training rows.
- Adam `1e-4`, batches of 32 H, at most 500 epochs, minimum 50 and patience 30,
  absolute stopping delta `1e-6`. Strict-best own-view validation weights are
  retained, including epoch 0. A capped fit remains a capped-policy result.
- After every model is complete, seal all selected model/source/config hashes.
  Only then generate **512 fresh H per size: 3,584 total**. Original historical
  test partitions were previously evaluated and are not reused as pristine tests.
- New chains follow the declared 100,000 total / 10,000 burn-in / 20 thinning /
  5% lazy policy. Rejected and repeated configurations are retained correctly.
- Fresh shared f/g features are evaluated without optimizers or reselection.
  Exact L3/L4 inputs are diagnostics only. Known true fields enter simulated
  observable preprocessing; this is not an unknown-field measurement procedure.
- Separate L4 raw-input controls reuse the ten clean mean/set checkpoints at
  nested 256/64 retained positions plus one inverse fit per H/budget. True fields
  are scoring references, never raw-model/solver inputs. No raw-set retraining.
- Targeted sampling diagnostics use eight predetermined H per size, four valid
  physical initializations, rank/folded split-Rhat, bulk/tail ESS and batch MCSE.
  Flags retain every H; they do not certify equilibrium of the entire bank.
  The prior L24 mixing concern remains a scientific limitation to review.

Primary contrast: own-input CNN-g minus CNN-f magnitude MSE at L4. Other sizes,
architecture contrasts, raw controls and D5 are declared secondary. Uncertainty
resamples whole Hamiltonians with all frozen seed losses retained. Seed SD is
reported separately. D5 is unsquared, globally merged full two-channel distance;
reference-sign assistance and native signed inverse scores are distinct.

## Start and pause on the prepared PC

The prepared source snapshot uses the existing shared research `.venv`; no second
Python environment is created. Its private GPU binding is in ignored
`inputs/worker-config.json`; the public config contains a placeholder.

From the prepared project root:

```bash
./.venv/bin/python experiments/final_benchmark/run.py
```

Menu: verify/prepare, start/resume, progress/results, request pause, exit. The
start option asks for **1–10 CPU sampler workers**, default 5. GPU training is
serial and requires actual CUDA. BLAS/Torch threads stay bounded; sampler
worker count does not change the scientific task or random streams.

For a run that survives SSH disconnection:

```bash
bash experiments/final_benchmark/start_pc.sh
```

This starts a one-shot user service, `quantum-lattice-final.service`, with no
automatic restart or boot launch. It uses the same verified boost-off and
offline-core workaround; it never changes host settings. Linger must already
be enabled for the user, as in the worker setup.

```bash
./.venv/bin/python experiments/final_benchmark/run.py --status
./.venv/bin/python experiments/final_benchmark/run.py --pause
bash experiments/final_benchmark/start_pc.sh --workers 5
```

Ctrl+C in the foreground also requests a checkpoint pause. Resume with the
same source/config/cache bytes; completed work is verified and skipped.
The service launcher explicitly resumes a paused queue. A failed guard stops
the child and requires inspection; nothing automatically retries a hardware fault.

## Four-day budget

`max_active_hours=96` is the cumulative maximum across resumes. Scientific
dispatch stops at 90 hours, preserving six hours for draining, verification and
export. Clean pauses do not consume offline time. An unclean exit charges the
unknown gap conservatively; it never silently extends the allowance.

The supervisor reserves time for a bounded atomic checkpoint drain before its
own wall deadline. Temperature, swap, new kernel faults, GPU UUID/processes,
boost and online/affinity guards are checked during execution. No unsafe core
is restored to meet the schedule. Exhausting the allowance leaves an explicitly
incomplete campaign; incomplete size/seed panels are not reported as complete.

## Inputs and portable staging

Existing banks and completed estimator results supply the immutable non-test
feature cache. `data/final_features.py` builds direct per-H maps, avoiding an
edge-row explosion. It verifies original float32 `f`, conditioned coordinates,
seeded fields, ordering and split identities without decompressing spin records
or opening old test payloads. Only L3/L4 have an exact view.

`stage_inputs.py` stages prior UID manifests, the selected raw checkpoints and
an optional source/input bundle. It verifies original completion/export hashes
and refuses different existing bytes. The bundle excludes old test payloads,
raw training records, checkpoints for optimizer continuation, paper/literature
notes, the user's unrelated MLP configuration and credentials.

Actual-PC tests are separate:

```bash
./.venv/bin/python experiments/final_benchmark/benchmark.py --help
./.venv/bin/python experiments/final_benchmark/acceptance.py --help
```

Their optimizer updates are discarded. Cyclic development fixtures reproduce
input residency for timing; they are not additional independent training H and
provide no accuracy, convergence or mixing result.

## Files and completion evidence

Shared logic:

- `data/final_features.py`: paired development cache.
- `models/final_training.py`: durable optimizer/RNG/cursor state and inference.
- `data/final_confirmation.py`: separately sealed fresh-H generation.
- `models/final_campaign.py`: scheduling, source/data pins and active-time ledger.
- `models/final_raw_controls.py`: separate raw bags and signed inverse controls.
- `data/final_diagnostics.py`: predetermined multi-chain precision checks.
- `models/final_evaluation.py`: per-H magnitude/D5 exports and paired intervals.
- `models/final_plots.py`: complete-grid plots with separate seed/H uncertainty.
- `models/final_worker.py`: one-launch PC supervisor.

Each campaign has `plan.json`, `clock.json`, `status.json`, `models/`,
`confirmation-seal.json`, `confirmation/`, `evaluations/`, `raw_controls/`,
`sampling_diagnostics/`, all-seed CSV/paired JSON reports, `figures/` and
`RESULTS.md`. Atomic generation pointers retain current and previous optimizer
checkpoints. `supervisor/launches/` records guards, stdout/stderr and hardware
history. No old campaign is modified.

`complete.json` is written only after the entire frozen design and scientific
inventory are complete. Verify every retained scientific hash explicitly with:

```bash
./.venv/bin/python experiments/final_benchmark/run.py --verify
```

Completed resume verifies that inventory before skipping work. Progress/status
is metadata-only; no optimizer or dataset is constructed to select a new model.
Review actual sampling flags, caps, negative results, costs and uncertainty
before promoting outputs into manuscript claims.
