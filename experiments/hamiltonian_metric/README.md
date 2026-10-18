# Hamiltonian reconstruction metric

Compute the reference paper's Eq. (9)/(D5): the unsquared Euclidean distance
between normalized Hamiltonian coefficient vectors. This is a known-field
simulation benchmark, matching the numerical setting of the reference study.
It does not require a new physical-measurement protocol or any training.

Run a small deterministic example with the existing environment:

```bash
.venv/bin/python experiments/hamiltonian_metric/run.py
```

It compares signed Z fields with an estimate equal to 80% of those fields at
L=3, with zero X fields. The scores illustrate the metric; they are not network
accuracy. Generated inputs, scores, provenance and report live under ignored
`codex material/artifacts/hamiltonian_metric/`, exposed through `results/` here.

For a real pair of field vectors:

```bash
.venv/bin/python experiments/hamiltonian_metric/run.py --input /path/to/fields.json
```

The JSON schema is:

```json
{
  "schema_version": 1,
  "L": 3,
  "model": "two_channel",
  "true_b_z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "recovered_b_z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "sign_source": "provided_signed_fields"
}
```

Each field vector has `2*L*L` entries in the project's square-lattice edge order.
Optional `true_b_x` and `recovered_b_x` default to zero. Recovered means the
estimated **original** fields, not the fields remaining after compensation.
Signed inputs are used as supplied: no clipping, absolute value or sign
inference. If predictions were assigned ground-truth signs, explicitly declare
`sign_source: "reference_sign_oracle"`; this label does not perform that lift.

The model convention is explicit:

- `two_channel` (default): both exponential sums in Eq. (8)/(D7), retaining the
  zero-X channel's identity contribution even when all X fields vanish.
- `z_branch`: Eq. (6), with only the star exponential sum. Nonzero X fields are
  rejected. At zero X field the two models differ by `n_faces * I`.

The runner reports the full Hamiltonian distance and a separately named
traceless distance. Both retain the fixed `-A_s` and `-B_p` terms. The latter
removes the combined identity coefficient, making it insensitive to energy
offsets. Always compare scores using the same model and identity convention.

Shared code is in `src/quantum_lattice/metrics/hamiltonian.py`:

```python
from quantum_lattice.lattice import square_lattice
from quantum_lattice.metrics.hamiltonian import hamiltonian_error

error = hamiltonian_error(square_lattice(L), true_bz, recovered_bz,
                          model="two_channel", include_identity=True)
```

This expands each local exponential exactly into Pauli products, combines equal
global strings before normalization, and uses logarithmic scaling for numerical
stability. It never allocates an exponentially large Hilbert-space matrix.
For square lattices, work grows linearly with the number of lattice cells.
Zero coefficient vectors have undefined normalization and are rejected.

The supplied author metric scripts differ from the printed formula: they print
a squared distance, omit identity/fixed stabilizer terms, and contain confirmed
normalization and field-geometry errors. This implementation follows the stated
D5 with a declared global Pauli basis; it does not claim to reproduce unknown
historical figure-generation conventions. The paper also rescales plotted
Hamiltonian errors to maximum one. We preserve raw D5 in `[0,2]` instead.

Reference: Valenti et al., [Phys. Rev. Research 1, 033092 (2019)](https://doi.org/10.1103/PhysRevResearch.1.033092),
Eq. (9), Appendix D.2, Eqs. (D5)–(D9). Independent tests use small dense matrix
exponentials and normalized Frobenius distances, separately from the production
coefficient expansion.
