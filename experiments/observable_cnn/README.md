# Observable-map CNN foundation

This prepares the data representation and models for a later controlled CNN
comparison. The current MLP convergence campaign continues independently.
The commands here perform small **CPU-only checks**, not research GPU training.
Use the existing shared `.venv`; no new environment is needed.

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

## Scientific scope

This phase validates representation and implementation. A short reduction in
synthetic training loss is a software check, not evidence of improved quantum
field estimation. No CNN has been trained on the research Hamiltonian bank.

Later compare MLP/f, MLP/g, CNN/f and CNN/g on matching Hamiltonians and
observable data. Preserve every paired seed. Similar parameter counts do not
match optimization or computational cost: a CNN batch contains whole
Hamiltonians, while the original MLP batches contain edges. Declare data
exposure, losses, stopping, runtime and evaluation before launching that study.
Receptive-field and capacity controls remain future experimental work.

Shared files are `src/quantum_lattice/data/observable_maps.py` and
`src/quantum_lattice/models/observable_cnn.py`. Settings are in
`configs/observable_cnn.json`. The running campaign's pinned files, existing
feature cache, original models and supplied reference material stay intact.
