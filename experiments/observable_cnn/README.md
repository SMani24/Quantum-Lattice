# Observable-map CNN

## Train the matched comparison

Use the existing shared `.venv`. From the repository root:

```bash
.venv/bin/python experiments/observable_cnn/train.py
```

The menu has four choices: run/continue L=4, run/continue L=3, run/continue
both, and progress/results. Start with L=4 to finish one full dataset's
comparison first. Each size has 20 models: CNN and paper MLP, each trained
with original `f` or conditioned `g` observable estimates, across five paired
seeds. Models run sequentially on CUDA. Missing CUDA stops with an error.

Press **Ctrl+C once** to request a pause. Wait for the checkpoint/pause message;
the menu then returns. Choose `q` to exit. Restart the same command and choose
the size again to continue, including an unfinished epoch and Adam state.
Completed models are hash-verified and skipped. Keep the frozen source,
environment and accelerator for exact continuation.

Preparation verifies the existing cache and freezes all 40 tasks, source/data
hashes, settings, environment and GPU. It performs zero optimizer updates.
Explicit commands are also available:

```bash
.venv/bin/python experiments/observable_cnn/train.py prepare
.venv/bin/python experiments/observable_cnn/train.py status
.venv/bin/python experiments/observable_cnn/train.py run --sizes 4
.venv/bin/python experiments/observable_cnn/train.py run --sizes 3
```

Settings are in `configs/observable_cnn_training.json`. The default run lives at
`experiments/observable_cnn/results/campaigns/matched-L34-20261018/` (a shortcut
to ignored local artifacts). `--run-dir PATH` resumes a particular frozen run;
use a new output directory when changing scientific settings. A redirected
noninteractive invocation requires an explicit `run` command; opening the
runner never implicitly launches research training.

### Data, optimization and selection

Each size uses the same existing **7,452 training and 1,242 validation
Hamiltonians**, separated by Hamiltonian identity. The cache contains the same
4,500 retained Monte Carlo configurations per Hamiltonian used in the matched
MLP comparison. No further Monte Carlo generation is needed. A training sample
is a three-channel map of star and neighboring-pair estimates; its target is
two maps of edge-field magnitudes.

Both architectures use Adam (`lr=1e-4`, betas `0.9,0.999`, epsilon `1e-8`),
32 whole Hamiltonians per shuffled batch, and a 500-epoch ceiling. All 7,452
Hamiltonians, including the last short batch, are visited once per epoch:
233 updates/epoch, at most 116,500 updates/model. A separate shuffle generator
pairs Hamiltonian order across models/views at each seed. The f/g initialization
is identical within each architecture. Float32 training uses deterministic CUDA,
TF32 disabled, raw outputs, and no scheduler or augmentation.

Full train/own-view validation MSE is measured at initialization and after
each epoch. The strictly lowest own-view validation MSE selects the saved best
model. Early stopping has a 50-epoch grace period followed by 30 checks without
an absolute `1e-6` improvement; constant loss stops earliest at epoch80.
Every strict improvement can update the best model independently of that delta.
The initial model is eligible. Pause/resume retains stopping history.

Fresh map-wrapped MLP controls share this policy with the CNN. Historical
edge-batched MLP runs used a different learning rate and data-order policy, so
their results provide context. The fresh comparison has matched data exposure
and epoch opportunities; FLOPs, runtime and actual stopping epochs can differ.

### Saved outputs

- `plan.json`, `plan-sha256.json`: frozen protocol, tasks and source/data/device
  bindings; `status.json` and `RESULTS.md`: progress and file map.
- `models/<L-seed-model-view>/latest.pt`: live weights, Adam, best weights,
  CPU/CUDA/shuffle RNG, permutation, partial epoch and full validation history.
- Finished models add `best.pt`, `history.json`, `evaluation.json`, raw
  `predictions.npz`, `per-hamiltonian.csv`, `hamiltonian-scores.csv`, and a
  hash-verified `done.json` commit marker.
- `paired-validation.csv`: every completed four-arm seed block, including
  crossed f/g/exact inputs and MSE/D5 contrasts. Terminal all-seed averages
  appear after every declared seed at that size finishes.
- `execution.log`: terminal output when training is launched through the runner.

Each selected model is evaluated on all three validation input views.
Exact inputs are evaluation-only. D5 scoring runs automatically afterwards:
raw unsquared two-channel/full distance is primary, known reference signs
lift magnitude predictions, and X fields are zero. Both model and full/traceless
conventions are retained. These are validation/model-selection results; signed
recovery and physical error correction remain later studies.

A bounded local RTX4060 Laptop timing pilot estimates about **two hours per
size, four hours for the full 40-model ceiling**. Allow roughly 2–3 hours per
size or 4–6 hours total for sustained operation and exports. Early stopping
can shorten this. Timing is an extrapolation, not a completed research run.

Shared implementation: `observable_training.py` owns Adam/selection/resume,
`observable_campaign.py` owns scheduling/integrity/exports, and
`observable_terminal.py` owns the small menu, under `src/quantum_lattice/models/`.
`train.py` is the experiment entry point. Existing datasets/models remain intact.

## Foundation checks

These commands perform small CPU checks using the same shared `.venv`:

From the repository root:

```bash
.venv/bin/python experiments/observable_cnn/run.py check
.venv/bin/python experiments/observable_cnn/run.py inspect --L 4
```

`check` runs a tiny synthetic forward/backward and optimization check, verifies
periodic translations, lossless local-feature conversion, and equivalence to
the unchanged paper MLP. `inspect` loads only the existing train/validation
feature cache, checks every f/g/exact view, and reports map/group dimensions.
It never reads spin records, original final-test payloads or model checkpoints.

Every invocation writes a new report below Git-ignored
`codex material/artifacts/observable_cnn/`, exposed through this experiment's
`results/` shortcut. `--output PATH` selects a new directory below that root.
Existing directories are never overwritten. `--config PATH` selects the
foundation settings; these are prototype choices, not a frozen research
training budget.

## Inputs and targets

One sample is one Hamiltonian. For a batch of B Hamiltonians on an L-by-L torus:

| Tensor | Shape | Channels |
|---|---|---|
| Input | `[B, 3, L, L]` | star A, neighboring pair AA-horizontal, neighboring pair AA-vertical |
| Target/prediction | `[B, 2, L, L]` | horizontal and vertical field magnitudes |

At `(y,x)`, a horizontal pair joins `(y,x)` to `(y,x+1)` and a vertical pair
joins it to `(y+1,x)`, with periodic coordinates. Edge order is
`2*(y*L+x)+orientation`, horizontal=0, vertical=1. The adapter checks repeated
star values agree exactly; it rejects independent inconsistent estimates
rather than silently averaging them. It supports shuffled edge rows and retains
their original positions for inverse conversion and prediction export.

The maps contain exactly the existing observable information, repacked spatially.
Known fields are labels only. Features and predictions remain raw, including
original-estimator values above one. Train/validation groups stay separated by
Hamiltonian UID. Exact maps preserve their evaluation-only eligibility.

## Models

- `PeriodicObservableCNN`: three circular-padded 3x3 convolutions with widths
  32,64,32 and ReLU, followed by a linear 1x1 two-channel output. It has
  **37,922 parameters** and handles varying lattice sizes without dense layers.
- `ObservableMapMLP`: gathers the same local triples `[A_u,A_v,AA_uv]` from
  maps and applies the existing shared `PaperMLP`. It adds no parameters and
  retains its **39,319 parameters**. This is the proper existing-method control.
- A kernel-size-1 CNN is available as a coordinate-local diagnostic. It is
  not equivalent to the paper MLP: it lacks the neighboring endpoint star in
  its same-coordinate input channels and has a different parameter count.

Circular padding follows the [PyTorch Conv2d API](https://docs.pytorch.org/docs/main/generated/torch.nn.Conv2d.html);
the implementation is checked on the project's installed Torch version.
Translation equivariance is tested across both periodic seams. Rotation and
reflection equivariance are not claimed. Three 3x3 layers have a nominal7x7
receptive field; at L3/L4 this wraps around and revisits the small torus.

## Foundation scientific scope

This phase validates representation and implementation. A short reduction in
synthetic training loss is a software check, not evidence of improved quantum
field estimation. No CNN has been trained on the research Hamiltonian bank.

The training protocol above now prepares MLP/f, MLP/g, CNN/f and CNN/g on
matching Hamiltonians and observable data. Every paired seed is retained.
Receptive-field, capacity and larger-lattice controls remain future work.

Shared files are `src/quantum_lattice/data/observable_maps.py` and
`src/quantum_lattice/models/observable_cnn.py`. Settings are in
`configs/observable_cnn.json`. The running campaign's pinned files, existing
feature cache, original models and supplied reference material stay intact.
