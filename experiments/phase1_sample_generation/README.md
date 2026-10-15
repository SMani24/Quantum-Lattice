# Solvable-model datasets

This runner generates Z-only solvable-model data for the paper-style MLP and
later CNN comparisons. Its dashboard selects L=3,4,8,12,16,20,24. Finish one
dataset and inspect it before moving to the next size. Training and the paper's
other experiments remain later steps.

## Open the dashboard

Use the repository's shared `.venv`: Python 3.10.12, NumPy 2.2.6 and Numba 0.63.1,
with CPU Torch available for neural-network training in the same environment.
Set it up once with `python3 scripts/setup_environment.py`; no environment
switch is needed between generation and training.
From the repository root:

```sh
.venv/bin/python experiments/phase1_sample_generation/run.py
```

From `experiments/phase1_sample_generation/`:

```sh
../../.venv/bin/python run.py
```

The dashboard lists each size and its saved status/counts. L=4 is initially
selected; press Enter at the size prompt. Then choose 2 for a short benchmark,
1 to start/resume the full selected dataset, 3 to inspect, 4 to verify, w to
change CPU workers, b to return to the size list, or q to exit.
No other size starts automatically. Quitting a running simulation returns to
the dashboard. Reopening finds its saved run; completed datasets are not rerun.

Status/help commands do not start a simulation:

```sh
../../.venv/bin/python run.py campaign --status
../../.venv/bin/python run.py --help
```

`configs/phase1_paper_domain_campaign.json` defines all selectable profiles.
`configs/phase1_sample_generation.json` matches its full L=4 profile.
`configs/phase1_sample_generation_pilot.json` preserves the earlier weak-field pilot.

## Dataset settings and provenance

Each size has **9,936 independently seeded Hamiltonians**: 7,452 training,
1,242 validation and 1,242 test. This implements the requested minimum of
7,450 independent training backgrounds. Every edge supplies additional local
learning rows, and all records/rows of a Hamiltonian stay in its split. Balanced
blocks preserve groups when the bank grows. Persistent UIDs identify duplicates
across runs; merged banks must be deduplicated/regrouped. Timing data are excluded
from learning.

| L | All local rows | Training rows | Validation/test rows each |
|---:|---:|---:|---:|
| 3 | 178,848 | 134,136 | 22,356 |
| 4 | 317,952 | 238,464 | 39,744 |
| 8 | 1,271,808 | 953,856 | 158,976 |
| 12 | 2,861,568 | 2,146,176 | 357,696 |
| 16 | 5,087,232 | 3,815,424 | 635,904 |
| 20 | 7,948,800 | 5,961,600 | 993,600 |
| 24 | 11,446,272 | 8,584,704 | 1,430,784 |

Paper-supported choices are the seven sizes, Z-only training branch,
independent Uniform[-1.7,1.7] fields and 7,450 training examples at a fixed target
edge. Our all-edge data extend those inputs. A strict 7,450-row MLP comparison
can later use one fixed-edge row from the first 7,450 training Hamiltonians.
Our validation/test allocation is a declared choice; the paper explicitly
reports 50 evaluation examples only for the shown L=16 training curve.

The historical training generator and topological-phase filter are unavailable.
These are paper-domain profiles, not a recovered historical bank or phase filter.
The 100,000 total / 10,000 burn-in / stride20 settings come from the supplied
correction sampler, not independently established historical training settings.

One sweep is V=L² attempted slots with replacement. Nonlazy star proposals use
`min(1, exp(-2*sum_incident(b_e*s_e)))`, updating immediately. Configurable 5%
lazy stays prevent zero-field periodicity. Records follow sweeps
10,020,...,100,000: **4,500 per Hamiltonian**. This clean boundary timing differs
from the author's one-attempt offset. Rejections and lazy stays remain recorded.

Known-field estimators provide every star, neighboring-star pair (shared edges
cancel), and single-Z expectation. They are not finite-shot X-basis outcomes.
Individual estimator values may exceed one and are never clipped. Means use all
records; 45-record blocks give 100 block means. IID/block errors are separate.
Scalar ESS and block errors do not certify burn-in or mixing across this broad
field domain. Review the full L=4 bank before scaling.

## Benchmark and pause

Benchmark the selected profile first. Four short 5,000-sweep chains are stored
separately; they do not finish or replace its full dataset. Timing extrapolation
is approximate and does not fully model growing checkpoints, finalization, disk
cost or mixing. This campaign is much larger than the earlier 16-background pilot.

During generation, p pauses, r continues, and q / Ctrl+C saves and returns.
Active chunks finish checkpointing before PAUSED is confirmed. Full profiles
use 5,000-sweep chunks to reduce disk writes; measure their pause latency locally.
Four CPU processes are the default. Proposals inside each chain stay sequential.

Worker count may change on resume. Scientific configuration, NumPy version,
core source hashes, seeds and counters are checked. UI-only changes do not
invalidate a scientifically unchanged run. Use a new campaign name for changed
scientific settings. An abrupt exit may lose the unfinished chunk; its previous
atomic checkpoint supplies deterministic continuation. Verification has its
own progress display and can be interrupted, then resumed without resampling.

A directly started run can also be resumed by its printed path:

```sh
../../.venv/bin/python run.py resume --run-dir "/absolute/path/to/run" --workers 4
```

## Compact, reconstructible storage

Full profiles retain every configuration losslessly as packed bits, every
star/pair/Z mean and IID/block error, scalar diagnostics, and local learning rows.
Redundant per-record estimator/block-mean arrays are omitted; reconstruct them
from the saved states and fields whenever needed.

All seven banks contain about **20 GB of uncompressed numerical payload**, plus
metadata, archive/filesystem overhead and checkpoints. Compression may reduce
actual storage. Keeping redundant floating-point records would need about
2.75 TB. L=4's numeric payload is about 0.22 GB. These are estimates, not measured
full runs; the dashboard shows per-size estimates and current free disk space.

Load without pickle and decode either packed or legacy records:

```python
import numpy as np
from quantum_lattice.data.state_storage import decode_states
from quantum_lattice.lattice import square_lattice
from quantum_lattice.observables.solvable import local_estimators

with np.load("hamiltonian_00000.npz", allow_pickle=False) as archive:
    states = decode_states(archive)
    values = local_estimators(square_lattice(4), archive["b_z"], states)
```

## Code and saved files

| Path | Responsibility |
|---|---|
| `src/quantum_lattice/lattice/` | Immutable topology and square constructor |
| `simulation/solvable.py` | Chain, compiled/reference kernel and RNG snapshots |
| `observables/solvable.py`, `statistics.py` | Estimators, exact checks and uncertainty |
| `data/generation.py`, `state_storage.py` | Workers, atomic storage, packed states and verification |
| `data/campaign.py` | Profile selector, campaign index and saved-status dashboard |
| `simulation/terminal.py` | Active-run controls and progress |
| `experiments/phase1_sample_generation/run.py` | Entry point |

`results/` points to ignored `codex material/artifacts/phase1_sample_generation/`.
Dashboard datasets/index live under `results/campaigns/phase1_paper_domain/`;
direct runs live under `results/runs/`. Each run keeps:

| File | Contents |
|---|---|
| `config.json`, `manifest.json` | Effective settings, identities/seeds/splits, environment, source hashes |
| `checkpoints/<id>.npz` | Partial packed records and chain continuation state |
| `instances/<id>.npz`, `<id>.json` | Completed arrays and readable counts/diagnostics |
| `dataset.json` | Final index, checksums and split counts |
| `status.json`, `last_session.json` | Lifecycle/progress and session timing |
| `verification.json` | Sector/seed/identity/estimator/storage checks; not a mixing certificate |
| `benchmark.json` | Short-run timing and extrapolation |

NPZ summaries contain `b_z`, `edge_vertex_indices`, each channel's `*_mean`,
`*_iid_se`, `*_block_se`, `mlp_X`, `target_abs_b`, `target_b` and JSON metadata.
Packed records use `states_packed`, `state_n_edges`, `state_encoding`; the pilot
keeps legacy `states`, `*_values` and `*_block_means` arrays.

Square indices use vertex yL+x and edge 2(yL+x)+orientation (0 outgoing horizontal,
1 outgoing vertical). Author field imports require an explicit mapping.
Supplied reference files remain intact.

Automated tests are separate from full generation. From the repository root,
`.venv/bin/python -m unittest discover -s tests -v` runs small fixtures; changing
dataset JSON settings does not turn tests into a long simulation.
