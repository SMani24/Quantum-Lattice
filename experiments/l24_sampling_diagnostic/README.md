# L24 sampling diagnostic

This separate experiment investigates star 540 in the selected **training**
Hamiltonian `L24/hamiltonian_02371`. Four original Metropolis trajectories
showed different local-pattern occupancies in a late window. The code audit
reproduced the discrepancy from the raw states and found no defect in the
examined proposal, indexing, sector, RNG, checkpoint or estimator paths.
The diagnostic does not replace the production sampler, data banks or models.

Shared exchange logic is in `src/quantum_lattice/simulation/replica_exchange.py`.
Plan verification, checkpointed jobs and reports are in
`src/quantum_lattice/data/sampling_diagnostic.py`; `run.py` is the thin entry point.

From the repository root, use the shared environment:

```bash
.venv/bin/python experiments/l24_sampling_diagnostic/run.py run --workers 4
```

Change `--workers` to control simultaneous CPU jobs. There are eight jobs,
so more than eight workers cannot help this panel. **Ctrl+C** stops scheduling
and lets active 1,000-sweep chunks checkpoint. Repeat the same command to
continue; the complete RNG states and committed records are restored.

```bash
.venv/bin/python experiments/l24_sampling_diagnostic/run.py status
```

`prepare` verifies and freezes the plan without simulating. `report` checks
completed outputs and rebuilds the saved report. `--max-chunks N` provides a
bounded pause for a pilot. A one-time `--sweeps N` override requires a **new**
`--run-dir`; omit the override on continuation. The production defaults live
in [the config](../../configs/l24_sampling_diagnostic.json). Scientific settings,
source files, cached inputs and numerical versions are pinned per run; workers
can change between starts. Do not edit the scientific files midway through a run.

## What runs

- Four ordinary local chains continue from their preserved 500,000-sweep
  checkpoints for another 200,000 sweeps each. Their RNG streams continue.
- Four independently seeded replica-exchange ensembles start fresh. Each has
  32 evenly spaced beta levels from 0 to 1. At every level it uses the existing
  lazy star Metropolis rule with fields `beta*b`. Adjacent exchanges alternate
  even/odd pairs every 10 sweeps; the beta0 slot uses local updates too.
- Both methods discard their first 20,000 additional/fresh sweeps and retain
  every 20th target configuration: **9,000 records per job**. For local chains
  these are original sweep positions 520,020 through 700,000; tempering records
  have fresh sweep positions 20,020 through 200,000.

The target is always `P(z) ∝ exp(sum_e b_e*z_e)` within the same reference
star orbit. Only the **beta=1** slot supplies retained tempering records.
For scores `S_i=sum b*z_i`, exchanging configurations i,j has log probability
ratio `(beta_i-beta_j)*(S_j-S_i)`. This preserves the product distribution;
the unknown partition functions cancel. The method follows
[Hukushima and Nemoto (1996)](https://arxiv.org/abs/cond-mat/9512035).
It is an established diagnostic method, not a new estimator or ground-state search.

A tempering sweep performs **32 times as many local proposals** as an ordinary
sweep. Budgets are explicitly unequal; this comparison tests whether another
valid kernel agrees, not an equal-cost speedup. Attempts and timings are saved.
All neighboring exchange rates, walker visits and cold-hot-cold round trips
are reported. Neither good exchange rates nor round trips certify equilibration.

## Files and interpretation

Outputs stay in the Git-ignored directory
`codex material/artifacts/estimator_validation/l24-diagnosis-20261016/resampling/`,
also exposed through this experiment's `results/` shortcut.

- `RESULTS.md`, `status.json`, `aggregate-summary.json`: progress, individual
  f/g estimates, whole/half/last-quarter comparisons, conventional split-Rhat,
  scalar autocorrelation diagnostics, all-star mean ranges and limitations.
- `plan.json`, `plan-sha256.json`, `fields.npz`: frozen settings, provenance,
  source/input hashes and a separate verified field copy.
- `jobs/<job>/checkpoint.json`: exact chain/RNG state and committed-prefix hash.
- `jobs/<job>/states_packed.npy`: preallocated target-only edge states. Read
  **only the committed prefix** named by the checkpoint; zero padding beyond
  that prefix is not sampled data. Bits are little endian, 1 means spin -1.
- `jobs/<job>/coordinates.npz`: all-star f/g means for four windows; flagged
  f/g traces, target scores and additional/fresh sweep positions. Neighbor-pair
  estimates can be recomputed from the saved target states if needed.
- `jobs/<job>/summary.json`, `done.json`: diagnostics and completed-file hashes.

A record tail flushed before a failed checkpoint is overwritten on continuation.
Completed jobs are verified and skipped. The output lock prevents simultaneous
coordinators. Cached inputs are checked against the previous verified campaign;
no original bank payload, final-test payload or neural network is opened.

This is one field selected because of its warning. It cannot establish the
frequency of problems throughout the bank, exact L24 bias, or a general sampler
speedup. Return the outputs for review before retraining or adopting a method.
Exact small-system transition checks and bounded pause/restart tests live in
`tests/unit/test_replica_exchange.py` and
`tests/integration/test_l24_sampling_diagnostic.py`.
