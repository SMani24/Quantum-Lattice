# Conditioned-estimator validation

Compare known-field `f=exp(-h)` and `g=sech(h)` without replacing the original
datasets, model inputs or frozen models. From the repository root:

```sh
.venv/bin/python experiments/estimator_validation/run.py
```

This opens a simple menu in your terminal:

- **1** starts or continues the tests.
- **2** changes CPU workers and remembers the choice for this campaign.
- **3** shows saved progress and result paths.
- **q** exits.

The menu shows the chosen worker count, available logical CPUs and stage
completion. **Press Ctrl+C once during execution and wait** to save and return
to the menu. You can then change workers and continue. Active saved/IID tasks finish; fresh chains
commit their current bounded 5,000-sweep chunk. Exact chain/RNG checkpoints are
retained. Completed tasks are hash-checked and skipped on continuation.

Four independent CPU workers run by default, with one BLAS/OMP thread each.
Numba runs the Metropolis kernel; GPU MLP training is a separate workload in
the same `.venv`. Choose any positive worker count in the menu; the new count
applies on the next start/continue, rather than resizing an active pool.
`runtime-settings.json` remembers it outside the immutable scientific plan.
Settings, source and numerical-library versions remain pinned.

Explicit command-line use is still available:

```sh
.venv/bin/python experiments/estimator_validation/run.py run --workers 12
```

`--workers` overrides the saved preference; otherwise the saved count wins over
the original four-worker default. The selected count also appears in the log.
`menu` explicitly opens the menu. With no command, redirected/noninteractive
invocations retain the original one-shot run behavior and never wait for input.

## Declared campaign

The [configuration](../../configs/conditioned_estimator_validation.json) defines:

|Stage|Default work|
|---|---|
|Saved census|All 60,858 non-test Hamiltonians, seven sizes; all stars/pairs on identical 4,500 records|
|Exact checks|All 17,388 L3/L4 census fields, independent amplitude-overlap truth and variance identities|
|IID reference|32 random fields per small size ×128 replicates ×4,500 exact-distribution draws|
|Multiple starts|16 random +4 prespecified stress fields per size ×4 starts ×500,000 total sweeps|
|Stationary MCMC|4 random fields per small size ×32 exact-stationary starts ×100,000 total sweeps|

New random panels exclude the previous pilot selection; stress cases reuse its
declared low-ESS fields. All payloads are guarded by manifest identity/split and
saved byte hashes. Final-test payloads/sidecars are never opened.

Fresh chains keep 5% lazy stays, 10,000 burn-in sweeps and stride 20. Equal-budget
windows (10k,100k], (100k,190k], (410k,500k] each retain 4,500 records; cumulative
estimates also end at 100k, 200k, 500k. Bounds refer to completed sweeps and the
first record follows the lower bound by 20 sweeps. Nested windows are correlated.

Plan for several hours; current-stage ETAs update from completed tasks. The
census reads roughly 15 GB of existing compressed files. Outputs require several
GB; allow approximately 10–15 GB free. The runner pauses below a 4 GiB free-space
reserve and does not delete files. Per-record floating estimator banks are not saved.

## Progress and files

```sh
.venv/bin/python experiments/estimator_validation/run.py status
.venv/bin/python experiments/estimator_validation/run.py report
```

`status` reads saved progress. `report` refreshes summaries without simulation;
run it while the campaign is stopped. Advanced `run --stage` accepts `saved`,
`iid`, `multistart` or `stationary`; the default runs all stages. `prepare`
declares metadata/seeds without payload reads. `--run-dir` continues an existing
run; a new scientific configuration needs a new `output_root`.

Outputs live in ignored `codex material/artifacts/estimator_validation/full-20261016/`,
exposed locally at `experiments/estimator_validation/results/full-20261016/`:

- `plan.json`, `plan-sha256.json`: settings, identities, seeds, input/source hashes and runtime.
- `status.json`, `execution.log`: progress and pause/error details.
- `runtime-settings.json`: optional remembered CPU worker count, separate from experiment settings.
- `jobs/<stage>/<task>/`: scalar summaries, compressed coordinate arrays, hash commit markers;
  fresh jobs also retain packed states and exact chain/RNG checkpoints.
- `per-hamiltonian.csv`, `aggregate-summary.json`: explicit replica/window IDs;
  replicas average within Hamiltonian, then Hamiltonians receive equal weight.
- `multistart-rhat.json`: conventional split-Rhat with undefined/infinite counts.
- `RESULTS.md`: progress, error tables, scope and file map; partial runs are labeled.

Reports refresh on pause/completion. Random/stress strata stay separate.
Stationary repetitions test finite-length MCMC error, not asymptotic variance.
Above L4 there is no exact target. ESS, block SE and conventional split-Rhat
cannot certify mixing. Both estimators require known simulation fields and are
not unknown-field experimental measurements. Return completed outputs for
scientific review before deciding on regeneration or retraining.
